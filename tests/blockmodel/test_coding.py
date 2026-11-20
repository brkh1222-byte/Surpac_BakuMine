"""P3.T7 acceptance: assigning values under constraints and coding the synthetic model."""

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.coding import AssignResult, CodingStep, assign, code_model
from bakumine.core.blockmodel.constraints import (
    AttributeConstraint,
    ConstraintError,
    DtmConstraint,
    FileResolver,
    RangeConstraint,
    SolidConstraint,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.expressions import ExpressionError, parse
from bakumine.core.blockmodel.partial import constraint_volume
from bakumine.synth.config import LensConfig, TopoConfig
from bakumine.synth.solid import LensSolid
from bakumine.synth.topo import Topography

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
TRUTH = json.loads((DATA / "truth.json").read_text())
GRID = TRUTH["blocks"]
EXACT_VOLUME = TRUTH["solid"]["volume_m3"]
ABOVE = DtmConstraint("topo", "above")
ORE_IN_ROCK = SolidConstraint("ore_solid") & DtmConstraint("topo", "below")
CODES = np.array(["AIR", "ORE", "WASTE"], dtype=object)


@pytest.fixture(scope="module")
def resolver():
    return FileResolver(DATA)


@pytest.fixture(scope="module")
def exact():
    """truth(points) -> index into CODES, from the implicit solid and the analytic topography."""
    lens_cfg, topo_cfg = LensConfig(), TopoConfig()
    # the generator defaults are the configuration the data set was made with
    for cfg, recorded in ((lens_cfg, TRUTH["config"]["lens"]), (topo_cfg, TRUTH["config"]["topo"])):
        assert json.loads(json.dumps(asdict(cfg))) == recorded
    solid, topo = LensSolid.from_config(lens_cfg), Topography(topo_cfg)

    def rock(points):
        code = np.full(len(points), 2)
        code[solid.inside(points)] = 1
        code[points[:, 2] > topo.z(points[:, 0], points[:, 1])] = 0
        return code

    return rock


def truth_model(depth):
    model = BlockModel(
        BlockModelDefinition(GRID["origin"], GRID["size"], GRID["shape"], max_depth=depth)
    )
    model.add_attribute("rock", "category", background="WASTE")
    return model


def rock_steps(max_level=None):
    return [
        CodingStep("rock", "AIR", ABOVE, True, max_level),
        CodingStep("rock", "ORE", ORE_IN_ROCK, True, max_level),
    ]


def coded_index(model, where=None):
    """Coded rock as an index into CODES."""
    labels = model.get("rock", where, decode=True)
    return np.searchsorted(CODES.astype(str), labels.astype(str))


# ------------------------------------------------------------------ the plan's criterion


def test_air_ore_waste_coding_agrees_with_truth(resolver, exact, capsys):
    d = BlockModelDefinition(GRID["origin"], GRID["size"], GRID["shape"])
    rng = np.random.default_rng(2024)
    probes = d.local_to_world(rng.uniform(0.0, 1.0, (1_500_000, 3)) * np.array(d.extent))
    probe_truth = exact(probes)
    lines, by_centroid, by_point = [], [], []
    for depth in range(4):
        model = truth_model(depth)
        t0 = time.perf_counter()
        results = code_model(model, rock_steps(), resolver)
        elapsed = time.perf_counter() - t0
        model.check_invariants()
        volume = model.volumes()
        coded = coded_index(model)
        # (a) the plan's measure: truth evaluated exactly at every (sub-)block centroid
        agree = float(volume[coded == exact(model.centroids())].sum() / volume.sum())
        # (b) stricter: truth at random points of the model against the block they fall in,
        #     which also counts the part of a cut block that is on the other side
        pointwise = float((coded[model.locate(probes)] == probe_truth).mean())
        by_centroid.append(agree)
        by_point.append(pointwise)
        ore = float(volume[coded == 1].sum())
        assert results[0].volume == pytest.approx(float(volume[coded == 0].sum()), rel=1e-12)
        assert results[1].volume == ore
        assert results[1].n_assigned == int((coded == 1).sum())
        lines.append(
            f"  {depth:>5}  {model.n_blocks:>9,}  {100 * agree:8.4f}%   {100 * pointwise:8.4f}%   "
            f"{ore:>11,.0f} ({100 * (ore / EXACT_VOLUME - 1):+.4f}%)   "
            f"{results[0].volume:>11,.0f}   {elapsed:.2f} s"
        )
    assert by_centroid[3] > 0.99  # the plan: > 99% of the model volume
    assert min(by_centroid) > 0.999  # true at every depth: only the surfaces' own error is left
    # the point-wise agreement is what sub-blocking improves
    assert by_point[0] > 0.99 and by_point[3] > 0.998
    assert all(b > a for a, b in zip(by_point, by_point[1:], strict=False))
    with capsys.disabled():
        print(
            "\n[P3.T7 coding] AIR above the DTM, ORE inside the solid mesh and below the DTM, "
            "WASTE elsewhere; truth = implicit solid + analytic topography\n"
            "  depth     blocks  at centroids  at 1.5M points  ore volume m3 (vs exact)    "
            "air volume m3   time\n" + "\n".join(lines)
        )


# ------------------------------------------------------------------ kinds of value


def small_model():
    d = BlockModelDefinition((0.0, 0.0, 100.0), (10.0, 10.0, 5.0), (6, 5, 8), max_depth=2)
    model = BlockModel(d)
    model.add_attribute("rock", "category", background="WASTE")
    model.add_attribute("density", "float")
    model.add_attribute("bench", "int", background=-1)
    return model


def test_categorical_numeric_and_expression_assignment():
    model = small_model()
    z = model.centroids()[:, 2]
    top = RangeConstraint.elevation(125.0, None)
    # category label; blocks outside the constraint keep the background
    result = assign(model, "rock", "AIR", top)
    assert isinstance(result, AssignResult)
    assert result.n_assigned == 6 * 5 * 3 and result.volume == 90 * 500.0
    assert (result.n_split, result.n_merged, result.n_invalid, result.n_blocks) == (0, 0, 0, 240)
    assert result.attribute == "rock" and result.value == "AIR" and result.subblock is None
    labels = model.get("rock", decode=True)
    assert (labels[z > 125.0] == "AIR").all() and (labels[z < 125.0] == "WASTE").all()
    assign(model, "rock", "ORE", RangeConstraint(x=(20.0, 40.0), z=(None, 125.0)))
    assert model.attribute("rock").categories == ["WASTE", "AIR", "ORE"]
    # a number, everywhere and under an attribute constraint
    everywhere = assign(model, "density", 2.7)
    assert everywhere.n_assigned == 240 and everywhere.volume == model.definition.volume
    assign(model, "density", np.float64(2.85), AttributeConstraint('rock == "ORE"'))
    assign(model, "density", float("nan"), AttributeConstraint('rock == "AIR"'))
    labels = model.get("rock", decode=True)
    np.testing.assert_array_equal(model["density"][labels == "ORE"], 2.85)
    np.testing.assert_array_equal(model["density"][labels == "WASTE"], 2.7)
    assert np.isnan(model["density"][labels == "AIR"]).all() and (labels == "ORE").sum() == 50
    # an expression (text or parsed), for the selected blocks only
    model.set("density", -1.0)
    rock = AttributeConstraint('rock != "AIR"')
    by_text = assign(model, "density", 'iif(rock == "ORE", 2.85, 2.70)', rock)
    assert by_text.value == 'iif(rock == "ORE", 2.85, 2.70)' and by_text.n_assigned == 150
    np.testing.assert_array_equal(
        model["density"], np.where(labels == "ORE", 2.85, np.where(labels == "AIR", -1.0, 2.70))
    )
    by_object = assign(model, "bench", parse("floor((z - 100) / 5)"))
    assert by_object.value == "floor((z - 100) / 5)"
    np.testing.assert_array_equal(model["bench"], model.ijk()[:, 2])
    tonnes = assign(model, "density", "density * volume / 0", rock)
    assert tonnes.n_invalid == 150 and np.isinf(model["density"][labels != "AIR"]).all()
    # None = missing category
    assign(model, "rock", None, top)
    assert model.is_missing("rock").sum() == 90
    # an empty selection is fine
    nothing = assign(model, "bench", 99, RangeConstraint(k=(50, 60)))
    assert nothing.n_assigned == 0 and nothing.volume == 0.0 and (model["bench"] != 99).all()


def test_errors_leave_the_model_untouched():
    model = small_model()
    before = {name: model[name].copy() for name in model.attribute_names}
    top = RangeConstraint.elevation(125.0, None)
    cases = [
        (KeyError, ("nope", 1.0), {}),
        (TypeError, ("rock", 3), {}),
        (ValueError, ("rock", parse("z")), {}),
        (TypeError, ("density", None), {}),
        (TypeError, ("density", True), {}),
        (TypeError, ("density", [1.0, 2.0]), {}),
        (ExpressionError, ("density", "ORE"), {}),  # a label is not a number: unknown name
        (ExpressionError, ("density", "__import__('os')"), {}),
        (ConstraintError, ("density", 1.0, "top"), {}),
        (ConstraintError, ("density", 1.0), {"subblock": True}),
        (ConstraintError, ("density", 1.0, AttributeConstraint("z > 0")), {"subblock": True}),
        (ValueError, ("density", 1.0, top), {"subblock": True, "max_level": 3}),
        (ValueError, ("bench", 1.5, top), {"subblock": True}),
    ]
    for error, args, kwargs in cases:
        with pytest.raises(error):
            assign(model, *args, **kwargs)
        assert model.is_dense
    # code_model validates every step before running the first
    cut = RangeConstraint.elevation(111.0, None)
    fresh = small_model()
    for bad in (("nope", 1), ("density", "no_such_attribute + 1"), ("rock", 5), ("rock",),
                {"attribute": "rock", "label": "X"}, "rock"):  # fmt: skip
        with pytest.raises((KeyError, ValueError, TypeError)):
            code_model(fresh, [("rock", "AIR", cut, True), bad])
        assert fresh.is_dense and (fresh.get("rock", decode=True) == "WASTE").all()
    for name, column in before.items():
        if name != "bench":
            np.testing.assert_array_equal(model[name], column)


# ------------------------------------------------------------------ sub-blocking, precedence


def test_subblock_option_selects_by_centroid_after_splitting():
    cut = RangeConstraint.elevation(111.0, None)  # 1 m into the third bench
    coarse = small_model()
    whole = assign(coarse, "rock", "AIR", cut)
    # without sub-blocking the bench 110-115 goes by its centroid (112.5 >= 111): all of it
    assert whole.volume == 6 * 5 * 100.0 * 30.0 and coarse.is_dense
    fine = small_model()
    result = assign(fine, "rock", "AIR", cut, subblock=True)
    fine.check_invariants()
    assert result.n_split == 30 + 30 * 4 == result.subblock.n_split
    assert result.n_blocks == fine.n_blocks == 240 + 7 * 150
    # 1.25 m sub-blocks: 110-111.25 has its centroid below 111 and stays rock
    assert result.volume == 3000.0 * (40.0 - 11.25)
    z = fine.centroids()[:, 2]
    np.testing.assert_array_equal(fine.get("rock", decode=True) == "AIR", z >= 111.0)
    # max_level stops earlier: 2.5 m sub-blocks, centroid 111.25 is above
    half = small_model()
    limited = assign(half, "rock", "AIR", cut, subblock=True, max_level=1)
    assert half.level.max() == 1 and limited.n_split == 30
    assert limited.volume == 3000.0 * 30.0
    # the volume helper agrees
    assert constraint_volume(fine, AttributeConstraint('rock == "AIR"')) == result.volume


def test_later_steps_overwrite_earlier_ones():
    a = RangeConstraint(x=(0.0, 40.0))
    b = RangeConstraint(x=(20.0, 60.0))
    first, second = small_model(), small_model()
    code_model(first, [("rock", "A", a), ("rock", "B", b)])
    code_model(second, [("rock", "B", b), ("rock", "A", a)])
    x = first.centroids()[:, 0]
    np.testing.assert_array_equal(first.get("rock", decode=True), np.where(x >= 20.0, "B", "A"))
    np.testing.assert_array_equal(second.get("rock", decode=True), np.where(x < 40.0, "A", "B"))
    # an expression sees the earlier steps; a later sub-blocking keeps earlier coding
    model = small_model()
    results = code_model(
        model,
        [
            ("density", 2.7),
            {"attribute": "rock", "value": "ORE", "constraint": a},
            CodingStep("density", 'iif(rock == "ORE", density + 0.15, density)'),
            ("rock", "AIR", RangeConstraint.elevation(126.25, None), True),
            ("density", float("nan"), AttributeConstraint('rock == "AIR"')),
        ],
    )
    assert [r.attribute for r in results] == ["density", "rock", "density", "rock", "density"]
    assert [r.n_split for r in results] == [0, 0, 0, 150, 0]
    assert results[4].volume == results[3].volume == 3000.0 * 13.75
    labels = model.get("rock", decode=True)
    centroid = model.centroids()
    air = centroid[:, 2] >= 126.25
    np.testing.assert_array_equal(labels[air], "AIR")
    np.testing.assert_array_equal(labels[~air], np.where(centroid[~air, 0] < 40.0, "ORE", "WASTE"))
    np.testing.assert_array_equal(
        model["density"][~air], np.where(centroid[~air, 0] < 40.0, 2.85, 2.7)
    )
    assert np.isnan(model["density"][air]).all()
    assert code_model(model, []) == []


# ------------------------------------------------------------------ merge


def test_merge_shrinks_the_model_without_changing_any_value(resolver, capsys):
    steps = [
        *rock_steps(),
        ("density", 'iif(rock == "ORE", 2.85, 2.70)', AttributeConstraint('rock != "AIR"')),
    ]
    plain, merged = truth_model(3), truth_model(3)
    for model in (plain, merged):
        model.add_attribute("density", "float")
    plain_results = code_model(plain, steps, resolver)
    t0 = time.perf_counter()
    merged_results = code_model(merged, steps, resolver, merge=True)
    elapsed = time.perf_counter() - t0
    merged.check_invariants()
    assert [r.n_merged for r in plain_results] == [0, 0, 0]
    assert merged_results[2].n_merged > 0 and merged_results[0].n_merged == 0
    assert merged_results[2].n_blocks == merged.n_blocks < 0.7 * plain.n_blocks
    for a, b in zip(plain_results, merged_results, strict=True):
        assert (a.n_assigned, a.volume, a.n_split) == (b.n_assigned, b.volume, b.n_split)
    # the value at any point is the same before and after the merge
    d = plain.definition
    rng = np.random.default_rng(8)
    points = d.local_to_world(rng.uniform(0.0, 1.0, (400_000, 3)) * np.array(d.extent))
    # ... including points next to the coded boundaries, where the sub-blocks are
    sub = plain.centroids(np.flatnonzero(plain.level == 3)[::5])
    points = np.concatenate([points, sub + rng.uniform(-0.3, 0.3, sub.shape)])
    rows_plain, rows_merged = plain.locate(points), merged.locate(points)
    np.testing.assert_array_equal(
        plain.get("rock", rows_plain, decode=True), merged.get("rock", rows_merged, decode=True)
    )
    np.testing.assert_array_equal(plain["density"][rows_plain], merged["density"][rows_merged])
    # and the volume per code is identical
    for label in ("AIR", "ORE", "WASTE"):
        test = AttributeConstraint(f'rock == "{label}"')
        assert constraint_volume(plain, test) == constraint_volume(merged, test) > 0
    # assign(..., merge=True) does the same in one call
    single = truth_model(2)
    result = assign(single, "rock", "AIR", ABOVE, resolver, subblock=True, merge=True)
    assert result.n_split > 0 and result.n_merged > 0
    assert result.n_blocks == single.n_blocks < GRID["shape"][0] * 52 * 84 + 7 * result.n_split
    with capsys.disabled():
        print(
            f"\n[P3.T7 merge] depth 3 coding (rock + density): {plain.n_blocks:,} blocks -> "
            f"{merged.n_blocks:,} after merging ({merged_results[2].n_merged:,} merges), "
            f"values identical at {len(points):,} points; code + merge {elapsed:.2f} s"
        )
