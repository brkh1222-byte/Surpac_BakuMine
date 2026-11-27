"""Block coding from a geology model (P9.T12): geology_contracts.md 11, research.md 5.7.6.

Independent checks:

- an analytic stand-in geology (planes, a slab of cover, a sphere): coded volumes against
  the closed-form volumes, convergence with the sub-block size, a rotated model;
- densities and tonnes against a pandas lookup written here;
- other attributes compared before / after at every block;
- a real (made-up) geology model: the label rules at the topography edge, brute-force
  distances, agreement with ``SolidConstraint`` on the unit meshes, replay of the steps
  file through ``bakumine blockmodel code``;
- the synthetic layered deposit: volumes against ``truth.json`` (contract section 14),
  once coding the analytic truth itself (the error of the coding alone) and once coding
  the fitted rock model of P9.T7 with the truth surfaces (model error + coding error).

The numbers of the layered deposit are printed (``pytest -s``).
"""

from __future__ import annotations

import copy
import json
import time

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from bakumine.cli import main
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.coding import code_model
from bakumine.core.blockmodel.constraints import (
    DictResolver,
    RangeConstraint,
    SolidConstraint,
    constraint_from_dict,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.geology.coding import (
    GeologyCodingResult,
    code_geology,
    coding_provenance_path,
    density_steps,
    geology_steps,
    unit_tonnage,
    write_coding_provenance,
)
from bakumine.core.geology.lithology import (
    LithologyStack,
    UnitFields,
    fit_unit_fields,
    logged_hole_points,
)
from bakumine.core.geology.model import GeologyModel
from bakumine.core.geology.provenance import Provenance
from bakumine.core.geology.units import Unit, UnitDictionary, WeatheringZone
from bakumine.core.geometry.dtm import DTM, dtm_from_grid
from bakumine.core.geometry.implicit import fit_field
from bakumine.core.geometry.isosurface import GridSpec
from bakumine.core.io.obj import read_obj
from bakumine.synth.layered import CATEGORIES

# --------------------------------------------------------------------------
# the analytic stand-in
# --------------------------------------------------------------------------

LO = np.array([1000.0, 2000.0, 300.0])
SIZE = np.array([160.0, 120.0, 100.0])
MID = LO + SIZE / 2.0
COVER, OXIDE = 4.0, 15.0
CENTRE, RADIUS = LO + [60.0, 55.0, 18.0], 14.0

UNITS = UnitDictionary(
    name="made-up deposit",
    default_density=2.7,
    units=(
        Unit(code="COV", kind="cover", density=1.9, density_source="measured"),
        Unit(code="UP", kind="lithology", density=2.5, density_source="measured"),
        Unit(code="LOW", kind="lithology"),
        Unit(code="BALL", kind="intrusion", density=2.9, density_source="assumed"),
        Unit(code="CL", kind="missing"),
    ),
    weathering_zones=(
        WeatheringZone(code="OX", density=2.3, density_source="measured"),
        WeatheringZone(code="FR"),
    ),
)


def topo(x, y):
    return LO[2] + 85.0 + 0.05 * (x - LO[0]) - 0.04 * (y - LO[1])


def contact(x, y):
    return LO[2] + 50.0 - 0.11 * (x - LO[0]) + 0.07 * (y - LO[1])


class Analytic:
    """lith: UP over LOW at a dipping plane, a ball inside LOW, 4 m of cover under a
    sloping topography, AIR above; weathering: 15 m of OX under the topography, FR below;
    fault blocks: two sides of a steep plane."""

    units = UNITS
    _categories = {
        "lith": ("UP", "LOW", "BALL", "COV", "AIR"),
        "weathering": ("OX", "FR", "AIR"),
        "fault_block": ("FA", "FB"),
    }

    def __init__(self, centre=CENTRE, radius=RADIUS, one_block: bool = False):
        self.centre, self.radius, self.one_block = np.asarray(centre), radius, one_block

    def categories(self, layer: str) -> tuple[str, ...]:
        if layer == "fault_block" and self.one_block:
            return ("ALL",)
        return self._categories[layer]

    def labels(self, points, layer: str = "lith", *, exact: bool = False) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64)
        x, y, z = pts.T
        t = topo(x, y)
        if layer == "lith":
            out = np.where(z >= contact(x, y), 0, 1)
            out[((pts - self.centre) ** 2).sum(axis=1) < self.radius**2] = 2
            out[z > t - COVER] = 3
            out[z > t] = 4
        elif layer == "weathering":
            out = np.where(z > t - OXIDE, 0, 1)
            out[z > t] = 2
        elif self.one_block:
            out = np.zeros(len(pts))
        else:
            out = np.where(x - LO[0] < 70.0 + 0.3 * (z - LO[2]), 0, 1)
        return out.astype(np.int16)


AREA = SIZE[0] * SIZE[1]
TOPO_MEAN, CONTACT_MEAN = topo(MID[0], MID[1]), contact(MID[0], MID[1])  # planes: mean = centre
BALL = 4.0 / 3.0 * np.pi * RADIUS**3
ANALYTIC = {
    "lith": {
        "AIR": AREA * (LO[2] + SIZE[2] - TOPO_MEAN),
        "COV": AREA * COVER,
        "UP": AREA * (TOPO_MEAN - COVER - CONTACT_MEAN),
        "BALL": BALL,
        "LOW": AREA * (CONTACT_MEAN - LO[2]) - BALL,
    },
    "weathering": {
        "AIR": AREA * (LO[2] + SIZE[2] - TOPO_MEAN),
        "OX": AREA * OXIDE,
        "FR": AREA * (TOPO_MEAN - OXIDE - LO[2]),
    },
    # x - x0 < 70 + 0.3 (z - z0): integral over y and z
    "fault_block": {
        "FA": SIZE[1] * (70.0 * SIZE[2] + 0.3 * SIZE[2] ** 2 / 2.0),
        "FB": SIZE.prod() - SIZE[1] * (70.0 * SIZE[2] + 0.3 * SIZE[2] ** 2 / 2.0),
    },
}


def box_model(max_depth: int = 2, size=(10.0, 10.0, 10.0)) -> BlockModel:
    shape = tuple(int(round(n)) for n in SIZE / np.array(size))
    return BlockModel(BlockModelDefinition(tuple(LO), size, shape, max_depth=max_depth))


def volumes_of(result: GeologyCodingResult, layer: str) -> dict:
    rows = result.volumes[result.volumes["layer"] == layer]
    return dict(zip(rows["category"], rows["volume"], strict=True))


def same_models(a: BlockModel, b: BlockModel) -> bool:
    if a.n_blocks != b.n_blocks or a.attribute_names != b.attribute_names:
        return False
    if not all(np.array_equal(getattr(a, s), getattr(b, s)) for s in ("parent", "level", "sub")):
        return False
    for name in a.attribute_names:
        x, y = a.attribute(name), b.attribute(name)
        if (x.kind, x.dtype, x.categories) != (y.kind, y.dtype, y.categories):
            return False
        if not np.array_equal(a[name], b[name], equal_nan=x.kind == "float"):
            return False
    return True


def values_kept(before: BlockModel, after: BlockModel, names) -> bool:
    """Every block of ``after`` holds, bit for bit, what ``before`` held at its centroid."""
    rows = before.locate(after.centroids())
    if (rows < 0).any():
        return False
    for name in names:
        old, new = before.attribute(name), after.attribute(name)
        if (old.kind, old.dtype, old.background) != (new.kind, new.dtype, new.background):
            if not (old.background != old.background and new.background != new.background):
                return False
        if old.categories != new.categories:
            return False
        if before[name][rows].tobytes() != after[name].tobytes():
            return False
    return True


@pytest.fixture(scope="module")
def coded() -> tuple[BlockModel, GeologyCodingResult]:
    model = box_model(2)
    return model, code_geology(model, Analytic(), geology_name="standin")


# --------------------------------------------------------------------------
# volumes: analytic, convergence, partition, rotation
# --------------------------------------------------------------------------


def test_volumes_against_analytic(coded):
    """2.5 m sub-blocks: within 0.5 % for the bodies bounded by planes, 3 % for the 14 m
    ball (stated tolerances; measured about 0.05 % and 1.6 %)."""
    model, result = coded
    assert result.attributes == {
        "fault_block": "fault_block", "lith": "lith", "weathering": "weathering",
    }  # fmt: skip
    assert result.skipped == {}
    for layer, truth in ANALYTIC.items():
        got = volumes_of(result, layer)
        assert list(got) == list(Analytic().categories(layer))  # no row for missing labels
        assert sum(got.values()) == pytest.approx(SIZE.prod(), rel=1e-12)
        assert sum(truth.values()) == pytest.approx(SIZE.prod(), rel=1e-12)
        for category, volume in truth.items():
            tolerance = 0.03 if category in ("BALL",) else 0.005
            assert got[category] == pytest.approx(volume, rel=tolerance), (layer, category)
    # the table equals a direct pandas sum over the blocks
    direct = pd.Series(model.volumes()).groupby(model.get("lith", decode=True)).sum()
    assert direct.to_dict() == pytest.approx(volumes_of(result, "lith"))


def test_every_block_has_exactly_one_label_equal_to_the_model(coded):
    model, _ = coded
    geology = Analytic()
    points = model.centroids()
    for layer in ("lith", "weathering", "fault_block"):
        spec = model.attribute(layer)
        assert spec.kind == "category" and spec.dtype == np.int16 and spec.background is None
        assert tuple(spec.categories) == geology.categories(layer)
        assert not model.is_missing(layer).any()
        assert np.array_equal(model[layer], geology.labels(points, layer))
    model.check_invariants()
    assert model.volumes().sum() == pytest.approx(SIZE.prod(), rel=1e-12)  # no gap, no overlap
    assert set(np.unique(model.level)) == {0, 1, 2}


def test_convergence_with_the_sub_block_size():
    errors = []
    for depth in range(4):
        result = code_geology(box_model(depth), Analytic(), geology_name="s", density=None)
        got = volumes_of(result, "lith")
        errors.append({c: abs(got[c] / v - 1.0) for c, v in ANALYTIC["lith"].items()})
    worst = [max(e.values()) for e in errors]
    print("\nstand-in: worst relative volume error by sub-block size 10 / 5 / 2.5 / 1.25 m: "
          + ", ".join(f"{100 * w:.2f} %" for w in worst))  # fmt: skip
    assert worst[0] > 0.2  # parent blocks alone: the 4 m cover and the ball are far off
    assert all(b < 0.5 * a for a, b in zip(worst, worst[1:], strict=False))
    assert worst[3] < 0.005
    for category in ("COV", "BALL", "UP"):
        series = [e[category] for e in errors]
        assert series[3] < series[1] < series[0]


def test_rotated_model_covering_the_same_region(coded):
    """A model turned by 90 degrees about z covers the same box with the same blocks, so
    the volumes must agree to rounding; a wrong local -> world conversion would put the
    sample points elsewhere."""
    _, straight = coded
    origin = (LO[0] + SIZE[0], LO[1], LO[2])
    turned = BlockModel(
        BlockModelDefinition(
            origin,
            (10.0, 10.0, 10.0),
            (12, 16, 10),
            max_depth=2,
            matrix=((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
        )  # fmt: skip
    )
    assert turned.definition.is_rotated
    box = turned.definition.bounding_box()
    assert np.allclose(box, [LO, LO + SIZE])
    result = code_geology(turned, Analytic(), geology_name="standin")
    for layer in ANALYTIC:
        a, b = volumes_of(straight, layer), volumes_of(result, layer)
        for category in a:
            assert b[category] == pytest.approx(a[category], rel=1e-6, abs=1e-6), category
    assert np.array_equal(turned["lith"], Analytic().labels(turned.centroids(), "lith"))
    assert unit_tonnage(turned)["tonnes"].sum() == pytest.approx(
        unit_tonnage(coded[0])["tonnes"].sum(), rel=1e-6
    )


def test_obliquely_rotated_model_gives_the_ball_volume():
    centre, radius = LO + [80.0, 60.0, 30.0], 25.0
    geology = Analytic(centre, radius)
    volume = 4.0 / 3.0 * np.pi * radius**3
    got = []
    for angles in ((0.0, 0.0, 0.0), (37.0, 0.0, 0.0), (37.0, -14.0, 21.0)):
        d = BlockModelDefinition(
            (0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (8, 8, 8), angles, max_depth=2
        )
        # put the middle of the (rotated) model on the ball
        origin = centre - d.local_to_world(np.array([40.0, 40.0, 40.0]))
        model = BlockModel(
            BlockModelDefinition(tuple(origin), d.block_size, d.shape, angles, max_depth=2)
        )
        assert np.allclose(model.definition.local_to_world([40.0, 40.0, 40.0]), centre)
        result = code_geology(model, geology, geology_name="s", layers=("lith",), density=None)
        got.append(volumes_of(result, "lith")["BALL"])
        assert np.array_equal(model["lith"], geology.labels(model.centroids(), "lith"))
    assert got == pytest.approx([volume] * 3, rel=0.01)


# --------------------------------------------------------------------------
# density, AIR and cover
# --------------------------------------------------------------------------


def test_density_against_a_pandas_lookup(coded):
    model, _ = coded
    spec = model.attribute("density")
    assert spec.kind == "float" and spec.unit == "t/m3"
    frame = pd.DataFrame(
        {
            "lith": model.get("lith", decode=True),
            "weathering": model.get("weathering", decode=True),
            "volume": model.volumes(),
        }
    )
    rock = {"COV": 1.9, "UP": 2.5, "BALL": 2.9}  # LOW has none: the default
    expected = frame["lith"].map(rock).fillna(2.7)
    # the zone replaces the rock density, but not in the cover (soil is not oxidised rock)
    expected = expected.mask((frame["weathering"] == "OX") & (frame["lith"] != "COV"), 2.3)
    expected = expected.mask(frame["lith"] == "AIR", np.nan)
    assert np.array_equal(model["density"], expected.to_numpy(), equal_nan=True)
    assert set(np.unique(expected.dropna())) == {1.9, 2.3, 2.5, 2.7, 2.9}
    # the cover lies wholly inside the oxide zone and keeps its own density
    cover = frame["lith"] == "COV"
    assert cover.any() and (frame.loc[cover, "weathering"] == "OX").all()
    assert (model["density"][cover.to_numpy()] == 1.9).all()
    assert ((frame["lith"] == "UP") & (expected == 2.5)).any()

    table = unit_tonnage(model).set_index("unit")
    frame["tonnes"] = (expected * frame["volume"]).fillna(0.0)
    sums = frame.groupby("lith")[["volume", "tonnes"]].sum()
    for unit in ("UP", "LOW", "BALL", "COV", "AIR"):
        assert table.loc[unit, "volume"] == pytest.approx(sums.loc[unit, "volume"], rel=1e-12)
        assert table.loc[unit, "tonnes"] == pytest.approx(sums.loc[unit, "tonnes"], rel=1e-12)
    assert table.loc["AIR", "tonnes"] == 0.0 and np.isnan(table.loc["AIR", "density"])
    assert table.loc["LOW", "density"] == pytest.approx(2.7)
    assert 2.3 < table.loc["UP", "density"] < 2.5
    assert int(table["blocks"].sum()) == model.n_blocks


def test_air_above_the_topography_and_cover_under_it(coded):
    model, _ = coded
    x, y, z = model.centroids().T
    lith = model.get("lith", decode=True)
    above = z > topo(x, y)
    assert above.any()
    assert (lith[above] == "AIR").all() and (lith[~above] != "AIR").all()
    assert (model.get("weathering", decode=True)[above] == "AIR").all()
    assert np.isnan(model["density"][above]).all()
    assert np.isfinite(model["density"][~above]).all()
    cover = ~above & (z > topo(x, y) - COVER)
    assert cover.any() and np.array_equal(lith == "COV", cover)


def test_density_steps_order():
    steps = density_steps(UNITS)
    assert [s["value"] for s in steps] == [2.7, 1.9, 2.5, 2.9, 2.3, "nan"]
    assert [s.get("constraint", {}).get("expression") for s in steps] == [
        None, 'lith == "COV"', 'lith == "UP"', 'lith == "BALL"',
        'weathering == "OX" and lith != "COV"', 'lith == "AIR"',
    ]  # fmt: skip
    assert all(s["attribute"] == "density" and s["subblock"] is False for s in steps)
    renamed = density_steps(UNITS, lith="rk", weathering=None, density="sg")
    assert [s.get("constraint", {}).get("expression") for s in renamed] == [
        None, 'rk == "COV"', 'rk == "UP"', 'rk == "BALL"', 'rk == "AIR"',
    ]  # fmt: skip
    bare = UnitDictionary(units=(Unit(code="UP", kind="lithology"),))
    assert [s["value"] for s in density_steps(bare)] == ["nan", "nan"]
    model = box_model(0)

    class NoDensity(Analytic):
        units = bare

    code_geology(model, NoDensity(), geology_name="s")
    assert np.isnan(model["density"]).all()


def test_cover_without_a_density_takes_the_default_not_the_zone():
    units = UnitDictionary(
        default_density=2.7,
        units=(
            Unit(code="COV", kind="cover"),
            Unit(code="UP", kind="lithology", density=2.5),
            Unit(code="LOW", kind="lithology"),
            Unit(code="BALL", kind="intrusion"),
        ),
        weathering_zones=(WeatheringZone(code="OX", density=2.3), WeatheringZone(code="FR")),
    )

    class Soil(Analytic):
        pass

    Soil.units = units
    model = box_model(1)
    result = code_geology(model, Soil(), geology_name="s")
    frame = pd.DataFrame(
        {"lith": model.get("lith", decode=True), "zone": model.get("weathering", decode=True)}
    )
    expected = frame["lith"].map({"UP": 2.5}).fillna(2.7)
    expected = expected.mask((frame["zone"] == "OX") & (frame["lith"] != "COV"), 2.3)
    expected = expected.mask(frame["lith"] == "AIR", np.nan)
    assert np.array_equal(model["density"], expected.to_numpy(), equal_nan=True)
    assert (model["density"][(frame["lith"] == "COV").to_numpy()] == 2.7).all()
    by_item = {e.item: e for e in result.provenance}
    assert "COV" in by_item["Default density"].used_for
    assert by_item["Density samples for unit COV"].kind == "missing"


# --------------------------------------------------------------------------
# the other attributes
# --------------------------------------------------------------------------


def precoded(max_depth: int = 2) -> BlockModel:
    """A model as the existing workflow leaves it: ore / waste / air, a grade, a density."""
    model = box_model(max_depth)
    model.add_attribute("rock", "category", background="WASTE")
    model.add_attribute("au", "float")
    model.add_attribute("zone", "int", background=-1)
    ore = RangeConstraint(x=(1033.0, 1101.0), y=(2021.0, 2083.0), z=(322.0, 351.0))
    code_model(model, [("rock", "ORE", ore, max_depth > 0), ("zone", 7, ore)])
    rng = np.random.default_rng(5)
    values = rng.lognormal(0.0, 1.0, model.n_blocks)
    values[rng.random(model.n_blocks) < 0.2] = np.nan  # missing stays missing
    model.set("au", values)
    return model


def test_existing_attributes_are_not_touched():
    before = precoded()
    after = copy.deepcopy(before)
    result = code_geology(after, Analytic(), geology_name="s", display="geo", ore_attribute="rock")
    assert after.n_blocks > before.n_blocks
    assert after.attribute_names[:3] == ["rock", "au", "zone"]
    assert values_kept(before, after, ["rock", "au", "zone"])
    assert after.attribute("rock").categories == ["WASTE", "ORE"]
    ore_before = before.volumes()[before.get("rock", decode=True) == "ORE"].sum()
    ore_after = after.volumes()[after.get("rock", decode=True) == "ORE"].sum()
    assert ore_after == pytest.approx(ore_before, rel=1e-12)

    # the display attribute: the rock unit, ORE where rock == "ORE"
    lith = after.get("lith", decode=True)
    expected = np.where(after.get("rock", decode=True) == "ORE", "ORE", lith)
    assert np.array_equal(after.get("geo", decode=True), expected)
    assert after.attribute("geo").categories == ["UP", "LOW", "BALL", "COV", "AIR", "ORE"]
    assert (expected == "ORE").any() and (lith[expected == "ORE"] != "ORE").all()
    assert [r.attribute for r in result.results].count("geo") == 7

    # without sub-blocking the stored columns are bit-identical
    flat_before = precoded()
    flat = copy.deepcopy(flat_before)
    code_geology(flat, Analytic(), geology_name="s", subblock=False)
    assert flat.n_blocks == flat_before.n_blocks
    for name in ("rock", "au", "zone"):
        assert flat[name].tobytes() == flat_before[name].tobytes()
        assert flat.attribute(name).to_dict() == flat_before.attribute(name).to_dict()

    # display without an ore attribute repeats the rock unit
    plain = box_model(1)
    code_geology(plain, Analytic(), geology_name="s", display="shown", density=None)
    assert np.array_equal(plain["shown"], plain["lith"])


def test_merge_keeps_every_value():
    before = precoded()
    plain, merged = copy.deepcopy(before), copy.deepcopy(before)
    code_geology(plain, Analytic(), geology_name="s")
    # the grade differs from block to block, so only a model without it can merge
    for model in (plain, merged, before):
        model.delete_attribute("au")
    merged = copy.deepcopy(before)
    result = code_geology(merged, Analytic(), geology_name="s", merge=True)
    assert result.n_merged > 0 and merged.n_blocks < plain.n_blocks
    assert values_kept(merged, plain, plain.attribute_names)
    assert volumes_of(result, "lith") == pytest.approx(
        dict(pd.Series(plain.volumes()).groupby(plain.get("lith", decode=True)).sum())
    )


# --------------------------------------------------------------------------
# the steps document
# --------------------------------------------------------------------------


def run_steps(model: BlockModel, spec: dict, geology, name: str) -> None:
    """What ``bakumine blockmodel code --steps`` does, from the JSON text."""
    spec = json.loads(json.dumps(spec))
    for item in spec["attributes"]:
        if item["name"] not in model.attributes:
            model.add_attribute(item.pop("name"), item.pop("kind"), **item)
    steps = []
    for item in spec["steps"]:
        if item.get("constraint") is not None:
            item["constraint"] = constraint_from_dict(item["constraint"])
        steps.append(item)
    code_model(model, steps, DictResolver(geologies={name: geology}))


def test_replaying_the_steps_gives_the_same_model():
    options = dict(display="geo", ore_attribute="rock", max_level=1)
    direct, replayed, contract = precoded(), precoded(), precoded()
    result = code_geology(direct, Analytic(), geology_name="standin", **options)
    spec = result.steps
    assert spec == geology_steps("standin", Analytic(), **options, merge=True, distance="d")
    assert len(result.results) == len(spec["steps"])
    # max_level: the geology made no block smaller than level 1 (the ore had level 2)
    assert (direct.level == 2).sum() == (precoded().level == 2).sum()
    assert (direct.level == 1).sum() > (precoded().level == 1).sum()
    run_steps(replayed, spec, Analytic(), "standin")
    assert same_models(direct, replayed)

    # shape of the document
    assert [a["name"] for a in spec["attributes"]] == [
        "fault_block", "lith", "weathering", "density", "geo",
    ]  # fmt: skip
    assert spec["attributes"][1] == {
        "name": "lith", "kind": "category", "dtype": "int16",
        "categories": ["UP", "LOW", "BALL", "COV", "AIR"],
    }  # fmt: skip
    lith = [s for s in spec["steps"] if s["attribute"] == "lith"]
    assert lith[0] == {"attribute": "lith", "value": None, "subblock": False}
    assert lith[1] == {
        "attribute": "lith", "value": "UP", "subblock": True, "max_level": 1,
        "constraint": {"type": "geology", "model": "standin", "layer": "lith", "in": ["UP"],
                       "side": "inside", "boundaries": True},
    }  # fmt: skip
    assert lith[2]["constraint"] == {
        "type": "geology", "model": "standin", "layer": "lith", "in": ["LOW"], "side": "inside",
    }  # fmt: skip
    assert [s["subblock"] for s in lith] == [False, True, False, False, False, False]
    order = [s["attribute"] for s in spec["steps"]]
    assert order == sorted(order, key=["fault_block", "lith", "weathering", "density", "geo"].index)

    # the contract's plain form (one sub-blocked step per category, no "boundaries") splits
    # the same blocks on this geology, whose bodies are thicker than the blocks
    plain = json.loads(json.dumps(spec))
    for step in plain["steps"]:
        constraint = step.get("constraint") or {}
        if constraint.get("type") == "geology":
            constraint.pop("boundaries", None)
            step["subblock"], step["max_level"] = True, 1
    run_steps(contract, plain, Analytic(), "standin")
    assert same_models(direct, contract)


def test_coding_twice_changes_nothing():
    model = precoded()
    code_geology(model, Analytic(), geology_name="s")
    again = copy.deepcopy(model)
    result = code_geology(again, Analytic(), geology_name="s")
    assert same_models(model, again)
    assert sum(r.n_split for r in result.results) == 0


def test_options_and_errors():
    geology = Analytic()
    model = box_model(1)
    result = code_geology(
        model, geology, geology_name="s", layers=("weathering", "lith"),
        attributes={"lith": "rk", "weathering": "wz"}, density="sg",
    )  # fmt: skip
    assert model.attribute_names == ["wz", "rk", "sg"]
    assert result.attributes == {"weathering": "wz", "lith": "rk"}
    assert list(result.volumes["layer"].unique()) == ["weathering", "lith"]
    assert np.array_equal(model["rk"], geology.labels(model.centroids(), "lith"))

    one = box_model(0)
    only = code_geology(one, Analytic(one_block=True), geology_name="s", layers=("fault_block",))
    assert one.attribute_names == [] and only.attributes == {} and only.results == []
    assert "no faults" in only.skipped["fault_block"] and only.volumes.empty

    weathered = box_model(0)
    code_geology(weathered, geology, geology_name="s", layers=("weathering",))
    assert weathered.attribute_names == ["weathering"]  # no density without the rock units

    untouched = precoded(0)
    reference = copy.deepcopy(untouched)
    for kwargs, message in [
        (dict(layers=("lith", "rock")), "unknown layer"),
        (dict(layers=("lith", "lith")), "twice"),
        (dict(attributes={"ore": "x"}), "unknown layer"),
        (dict(attributes={"lith": "au"}), "kind float"),
        (dict(density="rock"), "kind category"),
        (dict(display="geo", ore_attribute="nope"), "no ore attribute"),
        (dict(display="geo", ore_attribute="au"), "not a category"),
        (dict(ore_attribute="rock"), "give display"),
        (dict(display="rock", ore_attribute="rock"), "overwritten"),
        (dict(display="geo", layers=("weathering",)), "needs the layer 'lith'"),
        (dict(max_level=1), "max_level"),
        (dict(distance="far"), "distance"),
        (dict(attributes={"lith": "weathering"}), "share a name"),
    ]:
        with pytest.raises(ValueError, match=message):
            code_geology(untouched, geology, geology_name="s", **kwargs)
        assert same_models(untouched, reference), kwargs  # refused before any change

    class NoUnits(Analytic):
        units = None

    with pytest.raises(ValueError, match="unit dictionary"):
        code_geology(box_model(0), NoUnits(), geology_name="s")
    code_geology(box_model(0), NoUnits(), geology_name="s", density=None)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_entries(coded, tmp_path):
    model, result = coded
    by_item = {e.item: e for e in result.provenance}
    assert {item: e.kind for item, e in by_item.items()} == {
        "Sub-blocking at the geology boundaries": "assumed",
        "Density of unit UP": "measured",
        "Density of unit BALL": "assumed",
        "Density of unit COV": "measured",
        "Density of weathering zone OX": "measured",
        "Default density": "assumed",
        "Density samples for unit LOW": "missing",
    }
    assert all(e.task == "P9.T12" for e in result.provenance)
    assert {e.stage for e in result.provenance} == {"coding", "density"}
    assert by_item["Default density"].value == 2.7 and by_item["Density of unit UP"].value == 2.5
    assert "LOW" in by_item["Default density"].used_for

    # the shares, from pandas
    frame = pd.DataFrame(
        {
            "lith": model.get("lith", decode=True),
            "zone": model.get("weathering", decode=True),
            "volume": model.volumes(),
            "density": model["density"],
        }
    )
    frame = frame[frame["lith"] != "AIR"]
    frame["tonnes"] = frame["volume"] * frame["density"]
    total_v, total_t = frame["volume"].sum(), frame["tonnes"].sum()
    fresh = frame[frame["zone"] != "OX"]

    def text(part: pd.DataFrame) -> str:
        v, t = part["volume"].sum(), part["tonnes"].sum()
        return (
            f"{v:.0f} m3 ({100 * v / total_v:.1f} % of the block volume with a density), "
            f"{t:.0f} t ({100 * t / total_t:.1f} % of the tonnes)"
        )

    assert by_item["Default density"].detail == text(fresh[fresh["lith"] == "LOW"])
    oxide = frame[(frame["zone"] == "OX") & (frame["lith"] != "COV")]
    assert by_item["Density of weathering zone OX"].detail == text(oxide)
    assert by_item["Density of unit BALL"].detail == text(frame[frame["lith"] == "BALL"])
    up = by_item["Density of unit UP"].detail
    assert up.startswith(text(fresh[fresh["lith"] == "UP"]))
    oxidised = frame[(frame["lith"] == "UP") & (frame["zone"] == "OX")]["volume"].sum()
    assert up.endswith(f"{oxidised:.0f} m3 of the unit take a weathering-zone density instead")
    cover = frame[frame["lith"] == "COV"]
    assert by_item["Density of unit COV"].detail == text(cover)  # all of it, its own 1.9
    assert cover["volume"].sum() > 0 and (cover["density"] == 1.9).all()
    low = frame[frame["lith"] == "LOW"]["volume"].sum()
    assert by_item["Density samples for unit LOW"].detail.startswith(f"{low:.0f} m3 of blocks")
    shares = [fresh[fresh["lith"] == u]["volume"].sum() for u in ("UP", "LOW", "BALL")]
    shares += [cover["volume"].sum(), oxide["volume"].sum()]
    assert sum(shares) == pytest.approx(total_v)

    # the file: in the geology folder, three tables
    folder = tmp_path / "demo.geology"
    folder.mkdir()
    path = write_coding_provenance(folder, "model_a", result.provenance)
    assert path == folder / "coding" / "model_a.provenance.json"
    assert path == coding_provenance_path(folder, "model_a")
    record = Provenance.load(path)
    assert record.entries == tuple(result.provenance) and record.name == "model_a"
    assert [len(t) for t in record.tables().values()] == [3, 3, 1]
    with pytest.raises(ValueError, match="invalid model name"):
        write_coding_provenance(folder, "../model", result.provenance)
    with pytest.raises(FileNotFoundError):
        write_coding_provenance(tmp_path / "absent", "m", result.provenance)


def test_provenance_without_a_default_density():
    units = UnitDictionary(
        units=(
            Unit(code="COV", kind="cover"),
            Unit(code="UP", kind="lithology", density=2.5),  # no source given: assumed
            Unit(code="LOW", kind="lithology"),
            Unit(code="BALL", kind="intrusion"),
        ),
        weathering_zones=(WeatheringZone(code="OX"), WeatheringZone(code="FR")),
    )

    class Bare(Analytic):
        pass

    Bare.units = units
    model = box_model(1)
    result = code_geology(model, Bare(), geology_name="s")
    kinds = {e.item: e.kind for e in result.provenance if e.stage == "density"}
    assert kinds == {
        "Density of unit UP": "assumed",
        "Density samples for unit LOW": "missing",
        "Density samples for unit BALL": "missing",
        "Density samples for unit COV": "missing",
    }
    missing = next(e for e in result.provenance if e.item == "Density samples for unit LOW")
    assert missing.used_for == "its blocks have no density and no tonnes"
    lith = model.get("lith", decode=True)
    assert np.array_equal(np.isfinite(model["density"]), lith == "UP")
    table = unit_tonnage(model).set_index("unit")
    assert table.loc["LOW", "tonnes"] == 0.0 and table.loc["UP", "density"] == pytest.approx(2.5)


# --------------------------------------------------------------------------
# a real geology model (made up): rules at the topography edge, meshes, CLI, replay
# --------------------------------------------------------------------------

H = 5.0
GSIZE = np.array([120.0, 100.0, 90.0])
GSHAPE = tuple(int(n) for n in GSIZE / H + 1)
CAP = 15.0
NORMAL = np.array([0.25, -0.1, 1.0]) / np.linalg.norm([0.25, -0.1, 1.0])
TOPS = (NORMAL @ (LO + [60.0, 50.0, 55.0]), NORMAL @ (LO + [60.0, 50.0, 30.0]))
TOPO_EDGE = LO[0] + 90.0  # the topography ends here: beyond it, no AIR and no cover

REAL_UNITS = UnitDictionary(
    name="made-up deposit",
    default_density=2.7,
    units=(
        Unit(code="QC", kind="cover", density=1.8, density_source="measured"),
        Unit(code="BOT", kind="lithology", density=2.8, density_source="measured"),
        Unit(code="TOP", kind="lithology"),
        Unit(code="MID", kind="lithology", density=2.6, density_source="assumed"),
    ),
    weathering_zones=(
        WeatheringZone(code="OX", density=2.2, density_source="measured"),
        WeatheringZone(code="TR"),
        WeatheringZone(code="FR"),
    ),
)


def ground(x, y):
    return LO[2] + 75.0 + 6.0 * np.sin((x - LO[0]) / 30.0) + 4.0 * np.cos((y - LO[1]) / 25.0)


def lowered(depth: float):
    x = np.arange(LO[0] - 10.0, TOPO_EDGE + 1e-9, 5.0)
    y = np.arange(LO[1] - 10.0, LO[1] + GSIZE[1] + 10.0 + 1e-9, 5.0)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return dtm_from_grid(ground(xx, yy) - depth, (x[0], y[0]), (5.0, 5.0))


@pytest.fixture(scope="module")
def real_geology(tmp_path_factory):
    rng = np.random.default_rng(11)
    holes = rng.uniform(LO[:2], LO[:2] + GSIZE[:2], size=(40, 2))
    z = np.arange(LO[2] + 2.5, LO[2] + GSIZE[2], 5.0)
    points = np.column_stack([np.repeat(holes, len(z), axis=0), np.tile(z, len(holes))])
    s = points @ NORMAL
    top, bottom = s - TOPS[0], TOPS[1] - s
    data = np.clip(np.column_stack([top, np.minimum(-top, -bottom), bottom]), -CAP, CAP)
    host = UnitFields(
        ("TOP", "MID", "BOT"),
        [fit_field(points, data[:, i]) for i in range(3)],
        cap=CAP, sample_spacing=5.0, far_spacing=5.0,
        inputs=[(points, data[:, i]) for i in range(3)], hole_spacing=18.0,
    )  # fmt: skip
    geology = GeologyModel(
        name="made-up geology",
        units=REAL_UNITS,
        grid=GridSpec(LO, (H, H, H), GSHAPE),
        topography=lowered(0.0),
        stacks={"ALL": LithologyStack("ALL", (host,))},
        cover=("QC", lowered(4.0)),
        weathering=[("OX", lowered(12.0)), ("TR", lowered(20.0)), ("FR", None)],
        hole_points=points,
    )
    folder = tmp_path_factory.mktemp("geo") / "made.geology"
    geology.save(folder)
    return geology, folder


def real_block_model(max_depth: int = 1) -> BlockModel:
    model = BlockModel(BlockModelDefinition(tuple(LO), (10.0, 10.0, 10.0), (12, 10, 9),
                                            max_depth=max_depth))  # fmt: skip
    model.add_attribute("rock", "category", background="WASTE")
    model.add_attribute("au", "float")
    ore = RangeConstraint(x=(1021.0, 1069.0), y=(2031.0, 2074.0), z=(318.0, 339.0))
    code_model(model, [("rock", "ORE", ore, True)])
    model.set("au", np.random.default_rng(2).lognormal(0.0, 1.0, model.n_blocks))
    return model


def test_real_model_rules_and_distance(real_geology):
    geology, _ = real_geology
    model = real_block_model(2)
    result = code_geology(model, geology, geology_name="made", distance="geo_distance")
    assert result.skipped == {"fault_block": "the geology model has no faults (one block)"}
    assert "fault_block" not in model.attributes
    assert model.attribute("lith").categories == ["BOT", "TOP", "MID", "QC", "AIR"]
    points = model.centroids()
    for layer in ("lith", "weathering"):
        assert np.array_equal(model.get(layer, decode=True), geology.decode(
            geology.labels(points, layer), layer))  # fmt: skip
    x, y, z = points.T
    lith = model.get("lith", decode=True)
    zone = model.get("weathering", decode=True)
    beyond = x > TOPO_EDGE + 1.0  # outside the plan extent of the topography
    inside = x < TOPO_EDGE - 1.0
    assert beyond.any() and (z[beyond] > ground(x, y)[beyond]).any()
    assert not np.isin(lith[beyond], ["AIR", "QC"]).any()  # no AIR and no cover there
    assert (zone[beyond] == None).all()  # noqa: E711  (weathering unknown: missing)
    assert np.isin(lith[beyond], ["BOT", "TOP", "MID"]).all()
    assert np.isfinite(model["density"][beyond]).all()  # rock or default density, no zone
    assert set(np.unique(model["density"][beyond])) <= {2.8, 2.7, 2.6}
    # inside the extent, away from the triangulated surface by more than its faceting
    clear = inside & (np.abs(z - ground(x, y)) > 1.0)
    assert np.array_equal(lith[clear] == "AIR", (z > ground(x, y))[clear])
    assert np.isnan(model["density"][lith == "AIR"]).all()
    deep = inside & (z < ground(x, y) - 21.5)
    assert (zone[deep] == "FR").all() and (zone[lith == "AIR"] == "AIR").all()
    missing_volume = volumes_of(result, "weathering")[None]
    assert missing_volume == pytest.approx(model.volumes()[zone == None].sum())  # noqa: E711
    assert sum(volumes_of(result, "weathering").values()) == pytest.approx(GSIZE.prod())

    # the distance to the logged hole points, brute force on a sample
    spec = model.attribute("geo_distance")
    assert spec.kind == "float" and spec.unit == "m"
    rows = np.random.default_rng(1).choice(model.n_blocks, 300, replace=False)
    brute = np.sqrt(
        ((points[rows, None, :] - geology.hole_points[None, :, :]) ** 2).sum(axis=2)
    ).min(axis=1)
    assert np.allclose(model["geo_distance"][rows], brute)
    assert np.allclose(model["geo_distance"], cKDTree(geology.hole_points).query(points)[0])
    assert "geo_distance" not in [a["name"] for a in result.steps["attributes"]]


def test_agreement_with_solid_constraints_on_the_unit_meshes(real_geology):
    """Contract 11: >= 99.5 % of blocks. The meshes are cut at the cover base and the
    topography, so the comparison is the same statement: block centroid in the unit."""
    geology, _ = real_geology
    model = real_block_model(2)
    code_geology(model, geology, geology_name="made")
    lith = model.get("lith", decode=True)
    meshes = geology.meshes("lith")
    assert set(meshes) == {"BOT", "TOP", "MID"}
    for unit, mesh in meshes.items():
        agreement = float((SolidConstraint(mesh).mask(model) == (lith == unit)).mean())
        print(f"\nmade-up model: lith == {unit} equals the solid constraint on its mesh in "
              f"{100 * agreement:.3f} % of {model.n_blocks} blocks")  # fmt: skip
        assert agreement >= 0.995


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_end_to_end(real_geology, tmp_path, capsys):
    geology, folder = real_geology
    start, out, replayed = tmp_path / "start", tmp_path / "coded", tmp_path / "replayed"
    save_model(real_block_model(1), start)
    steps = tmp_path / "steps.json"
    code, text, err = run(
        capsys, "geology", "code", start, "--geology", folder, "--display", "geo",
        "--ore-attribute", "rock", "--write-steps", steps, "--out", out,
    )  # fmt: skip
    assert code == 0, err
    for title in (
        "Measured: real site data used (3)",
        "Assumed: chosen by the software (3)",
        "Missing: inputs still needed (1)",
        "Volume and tonnes per rock unit",
        "weathering: volume per category (m3)",
        "layer fault_block not coded",
    ):
        assert title in text, title
    assert "Density samples for unit TOP" in text and "Default density" in text

    coded_model = load_model(out)
    assert coded_model.attribute_names == ["rock", "au", "lith", "weathering", "density", "geo"]
    assert values_kept(load_model(start), coded_model, ["rock", "au"])
    assert np.array_equal(coded_model["lith"], geology.labels(coded_model.centroids(), "lith"))
    # the printed table is the model's
    table = unit_tonnage(coded_model).set_index("unit")
    printed = {}
    block = text.split("Volume and tonnes per rock unit")[1].split("\n\n")[0]
    for line in block.splitlines()[2:]:
        parts = line.split()
        printed[parts[0]] = (float(parts[2]), float(parts[3]))
    for unit in ("BOT", "TOP", "MID", "QC", "AIR"):
        assert printed[unit][0] == pytest.approx(table.loc[unit, "volume"], abs=0.6)
        assert printed[unit][1] == pytest.approx(table.loc[unit, "tonnes"], abs=0.6)
    assert printed["total"][0] == pytest.approx(GSIZE.prod(), abs=0.6)

    # the record of the run is in the geology folder and survives saving the block model
    record = folder / "coding" / "coded.provenance.json"
    assert record.is_file()
    assert sorted(p.name for p in out.iterdir()) == ["blocks.parquet", "model.json"]
    before = record.read_bytes()
    save_model(coded_model, out)
    save_model(coded_model, out)
    assert record.read_bytes() == before
    kinds = [e.kind for e in Provenance.load(record).entries]
    assert (kinds.count("measured"), kinds.count("assumed"), kinds.count("missing")) == (3, 3, 1)
    assert GeologyModel.load(folder).name == geology.name  # the folder still loads

    # replay of the steps file through `blockmodel code`
    document = json.loads(steps.read_text())
    assert document["steps"][1]["constraint"]["model"] == "made"
    code, _, err = run(capsys, "blockmodel", "code", start, "--steps", steps,
                       "--geometry-dir", folder.parent, "--out", replayed)  # fmt: skip
    assert code == 0, err
    assert same_models(coded_model, load_model(replayed))

    # in place, other options
    code, text, err = run(
        capsys, "geology", "code", out, "--geology", folder, "--no-density",
        "--layers", "lith", "--no-subblock", "--distance", "far", "--merge",
    )  # fmt: skip
    assert code == 0, err
    again = load_model(out)
    assert "far" in again.attributes and "0 blocks split" in text
    assert (folder / "coding" / "coded.provenance.json").read_bytes() != before

    code, _, err = run(capsys, "geology", "code", start, "--geology", tmp_path)
    assert code == 2 and "not a geology folder" in err
    code, _, err = run(capsys, "geology", "code", start, "--geology", folder,
                       "--ore-attribute", "rock")  # fmt: skip
    assert code == 2 and "--display" in err
    code, _, err = run(capsys, "geology", "code", start, "--geology", folder, "--layers", "ore")
    assert code == 2 and "unknown layer" in err
    assert same_models(load_model(start), real_block_model(1))  # failed runs wrote nothing


# --------------------------------------------------------------------------
# the synthetic layered deposit: volumes against truth.json (contract section 14)
# --------------------------------------------------------------------------

LITH = ("SST", "SLT", "AND", "DIO", "GRD", "Q4", "AIR")
TARGET = 0.05


def layered_box(deposit):
    cfg = deposit.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    return lo, hi


def layered_block_model(deposit, max_depth: int = 2) -> BlockModel:
    lo, hi = layered_box(deposit)
    size = np.array([10.0, 10.0, 5.0])
    shape = tuple(int(n) for n in np.round((hi - lo) / size))
    return BlockModel(BlockModelDefinition(tuple(lo), tuple(size), shape, max_depth=max_depth))


def truth_volumes(deposit) -> dict[str, float]:
    return {u: v["volume_m3"] for u, v in deposit.truth_json["volumes"]["units"].items()}


def report(title: str, got: dict, truth: dict) -> dict[str, float]:
    errors = {u: got[u] / truth[u] - 1.0 for u in LITH}
    lines = [f"\n{title}", f"  {'unit':<5} {'coded m3':>12} {'truth m3':>12} {'error':>8}  5 %"]
    for u in LITH:
        met = "met" if abs(errors[u]) <= TARGET else "MISSED"
        lines.append(f"  {u:<5} {got[u]:>12.0f} {truth[u]:>12.0f} {100 * errors[u]:>+7.2f}%  {met}")
    print("\n".join(lines))
    return errors


def test_layered_truth_coded_volumes(layered_small):
    """The coding alone: the analytic truth as the geology model, 10 x 10 x 5 m parents,
    2.5 x 2.5 x 1.25 m sub-blocks. Target of section 14: within 5 % per unit."""

    class Truth:
        def categories(self, layer):
            return tuple(CATEGORIES[layer])

        def labels(self, points, layer="lith", *, exact=False):
            return np.asarray(layered_small.truth.codes(points, layer)).astype(np.int16)

    model = layered_block_model(layered_small)
    t0 = time.perf_counter()
    result = code_geology(model, Truth(), geology_name="truth", density=None)
    seconds = time.perf_counter() - t0
    assert set(result.attributes) == {"fault_block", "lith", "weathering"}
    truth = truth_volumes(layered_small)
    assert tuple(truth) == LITH
    got = volumes_of(result, "lith")
    errors = report(
        f"layered-small, the truth coded ({model.n_blocks} blocks, {seconds:.1f} s):", got, truth
    )
    lo, hi = layered_box(layered_small)
    assert sum(got.values()) == pytest.approx(float(np.prod(hi - lo)), rel=1e-12)
    assert not model.is_missing("lith").any()
    for unit in ("SST", "SLT", "AND", "DIO", "Q4", "AIR"):
        assert abs(errors[unit]) <= TARGET, (unit, errors[unit])
    # NOT met: the 8 m dyke strikes 045, the centroids of 2.5 m sub-blocks cut it in rows
    # 1.77 m apart, 5 rows = 8.84 m (+10.5 %). A regression bound, not the target:
    assert TARGET < abs(errors["GRD"]) < 0.15
    # fault blocks against a lattice count of the truth (2 m, 6.2 million points)
    blocks = volumes_of(result, "fault_block")
    axes = [np.arange(lo[a] + 1.0, hi[a], 2.0) for a in range(3)]
    count = 0
    for x in axes[0]:
        plane = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        count += int(layered_small.truth.hanging_wall(plane).sum())
    assert blocks["FB2"] == pytest.approx(count * 8.0, rel=0.005)


@pytest.fixture(scope="module")
def layered_geology(layered_small, layered_small_dir, layered_small_db, layered_small_traces):
    """The rock model of P9.T7 (largest field, no fault handling, the intrusions as plain
    members of the group) with the truth topography, cover base and weathering bases."""
    folder = layered_small_dir
    units = UnitDictionary.load(folder / "units.json")
    group = ("SST", "SLT", "AND", "DIO", "GRD")
    table = layered_small_db.intervals["geology"]
    mapped = units.map_codes(table["lith"])
    fields = fit_unit_fields(table, mapped, group, layered_small_traces)
    lo, hi = layered_box(layered_small)
    geology = GeologyModel(
        name="layered-small, no faults",
        units=units,
        grid=GridSpec(lo, (5.0, 5.0, 5.0), tuple(int(n) for n in np.round((hi - lo) / 5.0) + 1)),
        topography=DTM(read_obj(folder / "topo.obj")),
        stacks={"ALL": LithologyStack("ALL", (fields,))},
        cover=("Q4", DTM(read_obj(folder / "truth_cover_base.obj"))),
        weathering=[
            ("OXIDE", DTM(read_obj(folder / "truth_oxide_base.obj"))),
            ("TRANSITION", DTM(read_obj(folder / "truth_transition_base.obj"))),
            ("FRESH", None),
        ],
        hole_points=logged_hole_points(
            table, layered_small_traces, step=5.0, known=mapped.isin(group)
        ),
    )
    geology.sample_grid()
    return geology


def test_layered_model_coded_volumes(layered_small, layered_geology):
    """The fitted rock model coded, against ``truth.json``. Only what is met is asserted;
    the misses belong to the rock model (no fault handling, no overprints: P9.T10 and
    P9.T13), which the second half shows: the coded volumes equal the model's own."""
    geology = layered_geology
    model = layered_block_model(layered_small)
    t0 = time.perf_counter()
    result = code_geology(model, geology, geology_name="layered")
    seconds = time.perf_counter() - t0
    truth = truth_volumes(layered_small)
    got = volumes_of(result, "lith")
    errors = report(
        f"layered-small, the P9.T7 rock model coded ({model.n_blocks} blocks, {seconds:.1f} s):",
        got, truth,
    )  # fmt: skip
    for unit in ("SST", "Q4", "AIR"):
        assert abs(errors[unit]) <= TARGET, (unit, errors[unit])
    missed = [u for u in LITH if abs(errors[u]) > TARGET]
    print(f"  within 5 %: {[u for u in LITH if u not in missed]}; missed: {missed}")
    # measured 2026-11-25: missed = SLT +32 %, AND -15 %, DIO +31 %, GRD -94 % (not asserted:
    # they change with the rock model, P9.T10 / P9.T13)
    assert not model.is_missing("lith").any()

    # the coding reproduces the volumes of the model it was given: a lattice count of the
    # model's own labels on an independent lattice
    lo, hi = layered_box(layered_small)
    count = np.round((hi - lo) / 4.0).astype(int) + 1  # about 4 m, not the block lattice
    axes = [lo[a] + (np.arange(count[a]) + 0.5) * (hi[a] - lo[a]) / count[a] for a in range(3)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 3)
    labels = geology.decode(geology.labels(points, "lith"), "lith")
    cell = float(np.prod(hi - lo)) / len(points)
    for unit in ("SST", "SLT", "AND", "DIO", "AIR"):
        own = float((labels == unit).sum()) * cell
        assert got[unit] == pytest.approx(own, rel=0.01), unit

    # weathering against a lattice count of the analytic truth (2.5 m)
    zones = volumes_of(result, "weathering")
    axes = [np.arange(lo[a] + 1.25, hi[a], 2.5) for a in range(3)]
    counted = dict.fromkeys(zones, 0)
    for x in axes[0]:
        plane = np.stack(np.meshgrid([x], axes[1], axes[2], indexing="ij"), -1).reshape(-1, 3)
        names, n = np.unique(layered_small.truth.labels(plane, "weathering"), return_counts=True)
        for name, k in zip(names, n, strict=True):
            counted[name] += int(k)
    for zone, volume in zones.items():
        assert volume == pytest.approx(counted[zone] * 2.5**3, rel=0.01), zone

    # densities: the dictionary of the deposit, through pandas
    frame = pd.DataFrame(
        {"lith": model.get("lith", decode=True), "zone": model.get("weathering", decode=True)}
    )
    rock = {"SST": 2.55, "SLT": 2.65, "AND": 2.75, "DIO": 2.80, "GRD": 2.70, "Q4": 1.90}
    oxidised = (frame["zone"] == "OXIDE") & (frame["lith"] != "Q4")  # the cover keeps 1.90
    expected = frame["lith"].map(rock).mask(oxidised, 2.30)
    expected = expected.mask(frame["lith"] == "AIR", np.nan)
    assert np.array_equal(model["density"], expected.to_numpy(), equal_nan=True)
    kinds = {e.item: e.kind for e in result.provenance if e.stage == "density"}
    assert set(kinds.values()) == {"measured"} and len(kinds) == 7  # 6 units + OXIDE
    table = unit_tonnage(model)
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
