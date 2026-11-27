"""The geology constraint (P9.T12): geology_contracts.md section 11, research.md 5.7.6.

Independent checks: the labels are analytic (half-spaces, a slab), so the mask is compared
with the formula at the centroids and the box classification with :class:`PlaneConstraint`
(a separate implementation of "which side are the corners on") and with an explicit loop
over the 9 sample points. All data are made up.
"""

from __future__ import annotations

import json
import subprocess
import sys

import numpy as np
import pytest

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    INSIDE,
    OUTSIDE,
    STRADDLE,
    ConstraintError,
    DictResolver,
    FileResolver,
    GeologyConstraint,
    PlaneConstraint,
    RangeConstraint,
    constraint_from_dict,
    load_constraint,
    save_constraint,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.subblocking import subblock_to_constraint

ORIGIN = (1000.0, 2000.0, 300.0)
POINT = np.array([1041.3, 2037.7, 331.9])
NORMAL = np.array([0.31, -0.22, 1.0]) / np.linalg.norm([0.31, -0.22, 1.0])
SPLIT = np.array([0.8, 0.6, 0.1]) / np.linalg.norm([0.8, 0.6, 0.1])


class Halves:
    """lith: ``UP`` above an oblique plane, ``DOWN`` below, ``THIN`` in a 1 m slab at x;
    weathering: three bands along the plane normal; fault_block: two sides of a plane.
    Points with x beyond ``edge`` have no lith label."""

    def __init__(self, thin_at: float | None = None, edge: float = np.inf):
        self.thin_at = thin_at
        self.edge = edge
        self.calls = 0

    def categories(self, layer: str) -> tuple[str, ...]:
        return {
            "lith": ("UP", "DOWN", "THIN"),
            "weathering": ("TOP", "MID", "BASE"),
            "fault_block": ("A", "B"),
        }[layer]

    def labels(self, points, layer: str = "lith", *, exact: bool = False) -> np.ndarray:
        self.calls += 1
        pts = np.asarray(points, dtype=np.float64)
        s = (pts - POINT) @ NORMAL
        if layer == "lith":
            out = np.where(s > 0.0, 0, 1)
            if self.thin_at is not None:
                out[np.abs(pts[:, 0] - self.thin_at) < 0.5] = 2
            out[pts[:, 0] > self.edge] = -1
        elif layer == "weathering":
            out = np.where(s > 12.0, 0, np.where(s > -9.0, 1, 2))
        else:
            out = np.where((pts - POINT) @ SPLIT > 0.0, 1, 0)
        return out.astype(np.int16)


def definition(rotated: bool, max_depth: int = 2) -> BlockModelDefinition:
    angles = (35.0, -12.0, 20.0) if rotated else (0.0, 0.0, 0.0)
    return BlockModelDefinition(ORIGIN, (10.0, 8.0, 5.0), (8, 9, 12), angles, max_depth=max_depth)


def boxes(d: BlockModelDefinition, n: int = 4000, seed: int = 3):
    rng = np.random.default_rng(seed)
    lo = rng.uniform(0.0, np.array(d.extent) - 12.0, size=(n, 3))
    return lo, lo + rng.uniform(0.5, 12.0, size=(n, 3))


# --------------------------------------------------------------------------
# mask and box classification
# --------------------------------------------------------------------------


@pytest.mark.parametrize("rotated", [False, True])
def test_mask_is_the_label_of_the_centroid(rotated):
    model = BlockModel(definition(rotated))
    assert model.definition.is_rotated == rotated
    s = (model.centroids() - POINT) @ NORMAL
    up = GeologyConstraint(Halves(), "lith", ["UP"]).mask(model)
    assert up.dtype == bool and 0 < up.sum() < len(up)
    assert np.array_equal(up, s > 0.0)
    assert np.array_equal(GeologyConstraint(Halves(), "lith", ["UP"], "outside").mask(model), ~up)
    both = GeologyConstraint(Halves(), "weathering", ["TOP", "BASE"]).mask(model)
    assert np.array_equal(both, (s > 12.0) | (s <= -9.0))
    assert GeologyConstraint(Halves(), "lith", ["UP", "DOWN", "THIN"]).mask(model).all()
    side = GeologyConstraint(Halves(), "fault_block", "B").mask(model)  # one name as text
    assert np.array_equal(side, (model.centroids() - POINT) @ SPLIT > 0.0)


@pytest.mark.parametrize("rotated", [False, True])
def test_boxes_against_plane_constraint(rotated):
    """For a half-space the 8 corners decide (the centre lies between them), so the
    answer must equal the plane constraint's on the same model-frame boxes. With a
    rotated model this fails unless the boxes are taken to world coordinates."""
    d = definition(rotated)
    lo, hi = boxes(d)
    got = GeologyConstraint(Halves(), "lith", ["UP"]).classify_boxes(d, lo, hi)
    want = PlaneConstraint(POINT, NORMAL, "above").classify_boxes(d, lo, hi)
    assert got.dtype == np.int8
    assert {int(v) for v in np.unique(got)} == {INSIDE, OUTSIDE, STRADDLE}
    assert np.array_equal(got, want)
    outside = GeologyConstraint(Halves(), "lith", ["UP"], "outside").classify_boxes(d, lo, hi)
    assert np.array_equal(outside, -got)


def test_boxes_use_eight_corners_and_the_centre():
    """An explicit loop over the 9 sample points, in world coordinates built by hand."""
    d = definition(True)
    lo, hi = boxes(d, n=600, seed=5)
    geology = Halves(thin_at=1040.0)
    r = d.rotation
    members = np.empty((len(lo), 9), dtype=bool)
    labels = np.empty((len(lo), 9), dtype=np.int64)
    for b in range(len(lo)):
        local = [[x, y, z] for x in (lo[b, 0], hi[b, 0]) for y in (lo[b, 1], hi[b, 1])
                 for z in (lo[b, 2], hi[b, 2])]  # fmt: skip
        local.append(list(0.5 * (lo[b] + hi[b])))
        world = np.array(ORIGIN) + np.array(local) @ r.T
        labels[b] = geology.labels(world, "lith")
        members[b] = np.isin(labels[b], [1, 2])
    want = np.where(members.all(axis=1), INSIDE, np.where(members.any(axis=1), STRADDLE, OUTSIDE))
    constraint = GeologyConstraint(geology, "lith", ["DOWN", "THIN"])
    assert np.array_equal(constraint.classify_boxes(d, lo, hi), want)
    # boundaries=True: STRADDLE as soon as two sample points differ, also DOWN against THIN
    differ = (labels != labels[:, :1]).any(axis=1)
    refined = GeologyConstraint(geology, "lith", ["DOWN", "THIN"], boundaries=True)
    got = refined.classify_boxes(d, lo, hi)
    assert np.array_equal(got, np.where(differ, STRADDLE, want))
    assert (got != want).any()  # the flag matters on this data
    flipped = GeologyConstraint(geology, "lith", ["DOWN", "THIN"], "outside", boundaries=True)
    assert np.array_equal(flipped.classify_boxes(d, lo, hi), np.where(differ, STRADDLE, -want))


def test_points_without_a_label():
    d = definition(False, max_depth=0)
    model = BlockModel(d)
    geology = Halves(edge=1040.0)
    x = model.centroids()[:, 0]
    every = GeologyConstraint(geology, "lith", ["UP", "DOWN", "THIN"])
    assert np.array_equal(every.mask(model), x <= 1040.0)  # no label: in no category
    outside = GeologyConstraint(geology, "lith", ["UP", "DOWN", "THIN"], "outside")
    assert np.array_equal(outside.mask(model), x > 1040.0)  # ... and so in the complement
    lo, hi = model.bounds_local()
    status = every.classify_boxes(d, lo, hi)
    assert (status[hi[:, 0] + ORIGIN[0] <= 1040.0] == INSIDE).all()
    assert (status[lo[:, 0] + ORIGIN[0] > 1040.0] == OUTSIDE).all()


def test_not_conservative_a_thin_body_passes_between_the_sample_points():
    """The documented limit (research.md 5.7.6): a 1 m slab at x = 1012 crosses the 10 m
    blocks of the second column between their corners (x = 1010, 1020) and centres
    (x = 1015), so they are not reported as cut and the sub-blocking does not find it."""
    geology = Halves(thin_at=1012.0)
    d = definition(False)
    model = BlockModel(d)
    thin = GeologyConstraint(geology, "lith", ["THIN"])
    lo, hi = model.bounds_local()
    assert (thin.classify_boxes(d, lo, hi) == OUTSIDE).all()
    assert subblock_to_constraint(model, thin).n_split == 0
    assert not thin.mask(model).any()
    # the same slab through the block centres is found and refined
    centred = GeologyConstraint(Halves(thin_at=1015.0), "lith", ["THIN"])
    result = subblock_to_constraint(model, centred)
    assert result.n_split > 0
    # ... but a 1 m slab still holds no centroid of the 2.5 m sub-blocks (x = ..13.75, ..16.25)
    assert not centred.mask(model).any()


def test_combines_with_other_constraints():
    d = definition(False)
    model = BlockModel(d)
    up = GeologyConstraint(Halves(), "lith", ["UP"])
    bench = RangeConstraint(z=(320.0, 340.0))
    z = model.centroids()[:, 2]
    assert up.is_geometric and (up & bench).is_geometric
    assert np.array_equal((up & bench).mask(model), up.mask(model) & (z >= 320.0) & (z < 340.0))
    lo, hi = model.bounds_local()
    both = (up & bench).classify_boxes(d, lo, hi)
    a, b = up.classify_boxes(d, lo, hi), bench.classify_boxes(d, lo, hi)
    assert np.array_equal(both == INSIDE, (a == INSIDE) & (b == INSIDE))
    assert np.array_equal(both == OUTSIDE, (a == OUTSIDE) | (b == OUTSIDE))
    assert np.array_equal((~up).mask(model), ~up.mask(model))


# --------------------------------------------------------------------------
# arguments, JSON, resolvers
# --------------------------------------------------------------------------


def test_arguments_are_checked():
    with pytest.raises(ConstraintError, match="layer"):
        GeologyConstraint("demo", "rock", ["UP"])
    with pytest.raises(ConstraintError, match="side"):
        GeologyConstraint("demo", "lith", ["UP"], "above")
    with pytest.raises(ConstraintError, match="values"):
        GeologyConstraint("demo", "lith", [])
    with pytest.raises(ConstraintError, match="values"):
        GeologyConstraint("demo", "lith", ["UP", 3])
    with pytest.raises(ConstraintError, match="twice"):
        GeologyConstraint("demo", "lith", ["UP", "UP"])
    with pytest.raises(ConstraintError, match="model"):
        GeologyConstraint(object(), "lith", ["UP"])
    model = BlockModel(definition(False, max_depth=0))
    with pytest.raises(ConstraintError, match="no category 'SST'"):
        GeologyConstraint(Halves(), "lith", ["UP", "SST"]).mask(model)


def test_json_round_trip(tmp_path):
    document = {
        "type": "geology",
        "model": "demo",
        "layer": "lith",
        "in": ["SST", "SLT"],
        "side": "inside",
    }  # fmt: skip  (the contract's example)
    constraint = constraint_from_dict(document)
    assert isinstance(constraint, GeologyConstraint)
    assert (constraint.model, constraint.layer, constraint.values, constraint.side) == (
        "demo", "lith", ("SST", "SLT"), "inside",
    )  # fmt: skip
    assert constraint.boundaries is False
    assert constraint.to_dict() == document
    assert json.loads(json.dumps(constraint.to_dict())) == document
    flagged = GeologyConstraint("demo", "weathering", ["OX"], "outside", boundaries=True)
    assert flagged.to_dict() == {"type": "geology", "model": "demo", "layer": "weathering",
                                 "in": ["OX"], "side": "outside", "boundaries": True}  # fmt: skip
    again = constraint_from_dict(flagged.to_dict())
    assert again.to_dict() == flagged.to_dict() and again.boundaries is True
    defaults = constraint_from_dict({"type": "geology", "model": "demo", "in": ["A"]})
    assert (defaults.layer, defaults.side) == ("lith", "inside")
    with pytest.raises(ConstraintError, match="missing field 'in'"):
        constraint_from_dict({"type": "geology", "model": "demo"})
    nested = constraint & RangeConstraint(z=(0.0, 10.0))
    save_constraint(nested, tmp_path / "c.json")
    assert load_constraint(tmp_path / "c.json").to_dict() == nested.to_dict()
    with pytest.raises(ConstraintError, match="not written to JSON"):
        GeologyConstraint(Halves(), "lith", ["UP"]).to_dict()
    assert "GeologyConstraint" in repr(constraint) and "..." in repr(
        GeologyConstraint(Halves(), "lith", ["UP"])
    )


def test_resolver_lookup():
    model = BlockModel(definition(False, max_depth=0))
    geology = Halves()
    by_name = GeologyConstraint("demo", "lith", ["UP"])
    want = GeologyConstraint(geology, "lith", ["UP"]).mask(model)
    resolver = DictResolver(geologies={"demo": geology})
    assert resolver.geology("demo") is geology
    assert np.array_equal(by_name.mask(model, resolver), want)
    lo, hi = model.bounds_local()
    assert np.array_equal(
        by_name.classify_boxes(model.definition, lo, hi, resolver),
        GeologyConstraint(geology, "lith", ["UP"]).classify_boxes(model.definition, lo, hi),
    )
    with pytest.raises(ConstraintError, match="pass a resolver"):
        by_name.mask(model)
    with pytest.raises(ConstraintError, match="unknown geology 'demo'"):
        by_name.mask(model, DictResolver())

    class OldResolver:  # a resolver written before the geology constraint
        def dtm(self, name): ...
        def solid(self, name): ...
        def polygon(self, name): ...

    with pytest.raises(ConstraintError, match="cannot look up geology models"):
        by_name.mask(model, OldResolver())
    assert DictResolver().geologies == {} and DictResolver({"t": 1}).dtms == {"t": 1}


def test_file_resolver_loads_the_geology_folder(tmp_path):
    """``geology(name)`` reads ``<dir>/<name>.geology/`` once; a real (tiny) model."""
    from bakumine.core.geology.lithology import LithologyStack, UnitFields
    from bakumine.core.geology.model import GeologyModel
    from bakumine.core.geology.units import Unit, UnitDictionary
    from bakumine.core.geometry.implicit import fit_field
    from bakumine.core.geometry.isosurface import GridSpec

    lo = np.array(ORIGIN)
    rng = np.random.default_rng(4)
    holes = rng.uniform(lo[:2], lo[:2] + [80.0, 72.0], size=(25, 2))
    z = np.arange(lo[2] + 2.5, lo[2] + 60.0, 5.0)
    points = np.column_stack([np.repeat(holes, len(z), axis=0), np.tile(z, len(holes))])
    s = np.clip((points - POINT) @ NORMAL, -15.0, 15.0)
    fields = UnitFields(
        ("UP", "DOWN"),
        [fit_field(points, s), fit_field(points, -s)],
        cap=15.0, sample_spacing=5.0, far_spacing=5.0, inputs=[(points, s), (points, -s)],
        hole_spacing=15.0,
    )  # fmt: skip
    units = UnitDictionary(
        units=(Unit(code="UP", kind="lithology"), Unit(code="DOWN", kind="lithology"))
    )
    geology = GeologyModel(
        name="tiny", units=units, grid=GridSpec(lo, (5.0, 6.0, 5.0), (17, 13, 13)),
        topography=None, stacks={"ALL": LithologyStack("ALL", (fields,))},
    )  # fmt: skip
    geology.save(tmp_path / "tiny.geology")
    resolver = FileResolver(tmp_path)
    loaded = resolver.geology("tiny")
    assert resolver.geology("tiny") is loaded and loaded.name == "tiny"
    model = BlockModel(definition(False, max_depth=0))
    got = GeologyConstraint("tiny", "lith", ["UP"]).mask(model, resolver)
    assert np.array_equal(got, geology.labels(model.centroids()) == 0)
    agreement = (got == ((model.centroids() - POINT) @ NORMAL > 0.0)).mean()
    assert agreement > 0.97  # the fitted plane against the plane the data came from
    with pytest.raises(ConstraintError, match="unknown geology 'other'"):
        GeologyConstraint("other", "lith", ["UP"]).mask(model, resolver)
    with pytest.raises(ConstraintError, match="invalid geology name"):
        resolver.geology("../tiny")


def test_constraints_module_does_not_import_the_geology_package():
    code = (
        "import sys, bakumine.core.blockmodel.constraints as c; "
        "assert 'geology' in c._TYPES; "
        "assert not [m for m in sys.modules if m.startswith('bakumine.core.geology')], 'imported'"
    )
    done = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=120)
    assert done.returncode == 0, done.stderr


def test_label_shape_is_checked():
    class Broken(Halves):
        def labels(self, points, layer="lith", *, exact=False):
            return super().labels(points, layer)[:-1]

    model = BlockModel(definition(False, max_depth=0))
    with pytest.raises(ConstraintError, match="labels of shape"):
        GeologyConstraint(Broken(), "lith", ["UP"]).mask(model)
