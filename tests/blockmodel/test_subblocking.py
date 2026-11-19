"""P3.T5 acceptance: sub-blocking on geometric constraints."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pytest

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    INSIDE,
    OUTSIDE,
    STRADDLE,
    AttributeConstraint,
    ConstraintError,
    DtmConstraint,
    FileResolver,
    PlaneConstraint,
    RangeConstraint,
    SolidConstraint,
    combine,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.subblocking import (
    SubblockResult,
    merge_uniform,
    subblock_to_constraint,
)
from bakumine.core.geometry.inside import points_in_solid
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, icosphere

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
TRUTH = json.loads((DATA / "truth.json").read_text())
GRID = TRUTH["blocks"]
MESH_VOLUME = TRUTH["solid"]["mesh"]["volume_m3"]
EXACT_VOLUME = TRUTH["solid"]["volume_m3"]
ORE = SolidConstraint("ore_solid")
ABOVE_TOPO = DtmConstraint("topo", "above")
ROT_Z30 = (
    (math.cos(math.radians(30.0)), -math.sin(math.radians(30.0)), 0.0),
    (math.sin(math.radians(30.0)), math.cos(math.radians(30.0)), 0.0),
    (0.0, 0.0, 1.0),
)


@pytest.fixture(scope="module")
def resolver():
    return FileResolver(DATA)


def truth_model(depth):
    """Model on the synthetic truth grid: 10 x 10 x 5 m parents over the same region."""
    return BlockModel(
        BlockModelDefinition(GRID["origin"], GRID["size"], GRID["shape"], max_depth=depth)
    )


def structure(model):
    return model.parent.copy(), model.level.copy(), model.sub.copy()


def assert_same_structure(a, b):
    for x, y in zip(structure(a), structure(b), strict=True):
        np.testing.assert_array_equal(x, y)


def selected_volume(model, constraint, resolver=None):
    """Volume of the blocks whose centroid satisfies the constraint."""
    return float(model.volumes()[constraint.mask(model, resolver)].sum())


def cell_boxes(model, rows, level):
    """Model-frame boxes ``(lo, hi)`` of the octree cells of ``level`` containing the blocks.

    ``level`` (a number or one per row) must not exceed the block's own level; equal depths.
    """
    lo, _ = model.bounds_local(rows)
    block = np.array(model.definition.block_size)
    size = block / (2.0 ** np.broadcast_to(level, (len(lo),)))[:, None]
    corner = model.ijk(rows) * block
    cell_lo = corner + np.floor((lo - corner) / size) * size
    return cell_lo, cell_lo + size


def symmetric_difference(coarse, coarse_mask, fine, fine_mask):
    """Volume where the coding of ``coarse`` differs from the coding of ``fine``."""
    rows = coarse.locate(fine.centroids())
    assert (rows >= 0).all()
    return float(fine.volumes()[coarse_mask[rows] != fine_mask].sum())


# ------------------------------------------------------------------ the plan's criterion


def convergence(make_model, resolver, depths):
    """Sub-block on the ore solid at each depth; one dict of numbers per depth."""
    out = []
    for depth in depths:
        model = make_model(depth)
        t0 = time.perf_counter()
        result = subblock_to_constraint(model, ORE, resolver)
        elapsed = time.perf_counter() - t0
        model.check_invariants()
        mask = ORE.mask(model, resolver)
        volume = float(model.volumes()[mask].sum())
        out.append(
            {"depth": depth, "model": model, "mask": mask, "result": result, "volume": volume,
             "seconds": elapsed, "vs_mesh": volume / MESH_VOLUME - 1.0,
             "vs_exact": volume / EXACT_VOLUME - 1.0}  # fmt: skip
        )
    finest = out[-1]
    for row in out:
        row["symdiff"] = symmetric_difference(row["model"], row["mask"], finest["model"],
                                              finest["mask"])  # fmt: skip
    return out


def print_convergence(title, rows, capsys):
    lines = [
        f"\n[P3.T5 {title}] volume of blocks whose centroid is inside the ore solid; "
        f"mesh {MESH_VOLUME:,.1f} m3, exact (implicit) {EXACT_VOLUME:,.1f} m3",
        "  depth  sub-block    blocks   selected m3   vs mesh    vs exact   bound (straddle)  "
        f"differs from depth {rows[-1]['depth']}   time",
    ]
    for row in rows:
        r = row["result"]
        lines.append(
            f"  {row['depth']:>5}  1/{2 ** row['depth']:<8} {r.n_after:>9,} "
            f"{row['volume']:>13,.1f}  {100 * row['vs_mesh']:+.4f}%  {100 * row['vs_exact']:+.4f}%"
            f"   +-{100 * r.volume_straddle / MESH_VOLUME:7.3f}%      "
            f"{100 * row['symdiff'] / MESH_VOLUME:8.4f}%        {row['seconds']:.2f} s"
        )
    with capsys.disabled():
        print("\n".join(lines))


def check_convergence(rows):
    by_depth = {row["depth"]: row for row in rows}
    # the plan's number: within 1% of the mesh volume at 1/8 sub-blocks
    assert abs(by_depth[3]["vs_mesh"]) < 0.01
    # ... in fact far inside it, and against the exact volume of the implicit solid as well
    # (the mesh itself is 0.057% smaller than the implicit solid)
    assert abs(by_depth[3]["vs_mesh"]) < 0.001
    assert abs(by_depth[3]["vs_exact"]) < 0.002
    for row in rows:
        r = row["result"]
        # rigorous bracket from the box classification: whole-inside <= mesh volume <= + straddle
        assert r.volume_inside <= MESH_VOLUME <= r.volume_inside + r.volume_straddle
        assert r.volume_inside <= row["volume"] <= r.volume_inside + r.volume_straddle
        assert r.volume_inside + r.volume_outside + r.volume_straddle == pytest.approx(
            row["model"].definition.volume, rel=1e-12
        )
    # Monotone with depth: (1) the guaranteed error bound (volume of the blocks the surface
    # still cuts) -- it roughly halves per level; (2) the volume that is coded differently
    # from the finest model, i.e. the misclassified volume.
    bound = [row["result"].volume_straddle for row in rows]
    assert all(b < 0.6 * a for a, b in zip(bound, bound[1:], strict=False))
    symdiff = [row["symdiff"] for row in rows]
    assert all(b < 0.6 * a for a, b in zip(symdiff, symdiff[1:], strict=False))
    # (3) the net volume error. It is the *difference* of the over- and under-coded volumes,
    # which cancel almost completely at every depth >= 1, so beyond depth 1 it is noise at
    # the 0.01% level and NOT strictly monotone from one depth to the next (see the printed
    # table). What does hold: every refinement beats the parent blocks, the envelope
    # max(|error| at this depth or deeper) decreases, and the finest is the best.
    net = [abs(row["vs_mesh"]) for row in rows]
    assert all(e < net[0] for e in net[1:])
    assert net[-1] == min(net)
    assert all(e < 2e-3 for e in net[1:]) and all(e < 2e-4 for e in net[3:])


def test_ore_solid_volume_converges_with_depth(resolver, capsys):
    rows = convergence(truth_model, resolver, range(5))
    print_convergence("truth grid", rows, capsys)
    check_convergence(rows)
    # depth 0 splits nothing and equals the plain centroid mask of the parent model
    assert rows[0]["result"].n_split == 0 and rows[0]["model"].is_dense
    # split counts: each level only examines the children of the previous level's straddlers
    r = rows[3]["result"]
    assert r.n_before == rows[3]["model"].definition.n_parents
    assert r.n_after == r.n_before + 7 * r.n_split
    assert r.n_inside + r.n_outside + r.n_straddle == r.n_after
    assert all(b <= 8 * a for a, b in zip(r.split_per_level, r.split_per_level[1:], strict=False))


def test_rotated_model_converges(resolver, capsys):
    """The same solid in a model turned 30 degrees about Z."""
    mesh = resolver.solid("ore_solid")

    def make(depth):
        d = BlockModelDefinition.auto_extent(mesh, GRID["size"], margin=20.0, matrix=ROT_Z30,
                                             max_depth=depth)  # fmt: skip
        assert d.is_rotated
        return BlockModel(d)

    rows = convergence(make, resolver, range(5))
    print_convergence("rotated 30 deg about Z", rows, capsys)
    check_convergence(rows)


# ------------------------------------------------------------------ only the boundary is split


def test_only_boundary_blocks_are_split(resolver, capsys):
    model = truth_model(3)
    d = model.definition
    result = subblock_to_constraint(model, ORE, resolver)
    model.check_invariants()
    uniform = d.n_parents * 8**3
    assert result.n_after < 0.02 * uniform
    # every block that was split was a STRADDLE box (strict classification, no tolerance) ...
    for level in (1, 2, 3):
        rows = np.flatnonzero(model.level >= level)
        lo, hi = cell_boxes(model, rows, level - 1)
        boxes = np.unique(np.concatenate([lo, hi], axis=1), axis=0)
        status = ORE.classify_boxes(d, boxes[:, :3], boxes[:, 3:], resolver)
        assert (status == STRADDLE).all()
        assert len(boxes) == result.split_per_level[level - 1]
    # ... and no block that could still be split is one
    lo, hi = model.bounds_local()
    status = ORE.classify_boxes(d, lo, hi, resolver)
    assert not ((status == STRADDLE) & (model.level < 3)).any()
    assert int((status == STRADDLE).sum()) == result.n_straddle
    assert int((status == INSIDE).sum()) == result.n_inside
    assert int((status == OUTSIDE).sum()) == result.n_outside
    # independent check that whole blocks are uniform: random interior points of the blocks
    # next to the refinement agree with their centroid
    mesh = resolver.solid("ore_solid")
    rng = np.random.default_rng(5)
    near = np.flatnonzero(status != STRADDLE)
    near = near[rng.choice(len(near), 60_000, replace=False)]
    lo, hi = model.bounds_local(near)
    points = d.local_to_world(lo + rng.uniform(0.001, 0.999, (len(near), 3)) * (hi - lo))
    np.testing.assert_array_equal(
        points_in_solid(mesh, points), points_in_solid(mesh, model.centroids(near))
    )
    np.testing.assert_array_equal(points_in_solid(mesh, points), status[near] == INSIDE)
    # a second run finds nothing to do
    before = structure(model)
    again = subblock_to_constraint(model, ORE, resolver)
    assert again.n_split == 0 and again.n_after == again.n_before == result.n_after
    for x, y in zip(before, structure(model), strict=True):
        np.testing.assert_array_equal(x, y)
    with capsys.disabled():
        print(
            f"\n[P3.T5 locality] depth 3 on the ore solid: {result.n_after:,} blocks = "
            f"{100 * result.n_after / uniform:.2f}% of a uniform 1/8 grid ({uniform:,}); split "
            f"per level {result.split_per_level}; {result.n_straddle:,} smallest blocks still cut"
        )


def test_deterministic_and_independent_of_chunking(resolver):
    constraint = combine("below and not ore", {"below": ~ABOVE_TOPO, "ore": ORE})
    models, results = [], []
    for chunk in (1 << 18, 1 << 18, 4_999, 70_001):
        model = truth_model(2)
        results.append(subblock_to_constraint(model, constraint, resolver, chunk_size=chunk))
        models.append(model)
    for model, result in zip(models[1:], results[1:], strict=True):
        assert_same_structure(models[0], model)
        assert result == results[0]
    assert isinstance(results[0], SubblockResult) and results[0].n_split > 0


# ------------------------------------------------------------------ combinations


def test_combination_refines_along_both_boundaries(resolver, capsys):
    """WASTE = below the topography AND NOT inside the ore solid."""
    waste = combine("below and not ore", {"below": ~ABOVE_TOPO, "ore": ORE})
    model = truth_model(2)
    result = subblock_to_constraint(model, waste, resolver)
    model.check_invariants()
    # the two surfaces are > 60 m apart, so the result is the union of the two refinements
    one = truth_model(2)
    r_topo = subblock_to_constraint(one, ABOVE_TOPO, resolver)
    r_ore = subblock_to_constraint(one, ORE, resolver)
    assert r_topo.n_split > 0 and r_ore.n_split > 0
    assert_same_structure(model, one)
    assert result.n_split == r_topo.n_split + r_ore.n_split
    # sub-blocks exist along both: next to the DTM and next to the solid
    sub = np.flatnonzero(model.level > 0)
    height = resolver.dtm("topo").height_above(model.centroids(sub))
    near_topo = np.abs(height) < 12.0
    assert near_topo.sum() == 7 * r_topo.n_split + r_topo.split_per_level[0]
    assert (~near_topo).sum() == 7 * r_ore.n_split + r_ore.split_per_level[0]
    # volume: (rock below the DTM, by fine plan integration of z_at) - (mesh volume)
    d = model.definition
    step = 0.5
    gx = d.origin[0] + (np.arange(int(d.extent[0] / step)) + 0.5) * step
    gy = d.origin[1] + (np.arange(int(d.extent[1] / step)) + 0.5) * step
    z = resolver.dtm("topo").z_at(gx[:, None], gy[None, :])
    assert not np.isnan(z).any() and z.max() < d.origin[2] + d.extent[2]
    rock = float((z - d.origin[2]).sum() * step * step)
    volume = selected_volume(model, waste, resolver)
    assert volume == pytest.approx(rock - MESH_VOLUME, rel=2e-4)
    with capsys.disabled():
        print(
            f"\n[P3.T5 combined] below topo AND NOT ore, depth 2: {result.n_after:,} blocks; "
            f"waste volume {volume:,.0f} m3 vs (rock below DTM - ore mesh) "
            f"{rock - MESH_VOLUME:,.0f} m3 ({100 * (volume / (rock - MESH_VOLUME) - 1):+.4f}%)"
        )


def test_and_not_with_plane_refines_only_where_it_matters():
    """Lower half of a sphere: inside the sphere AND NOT above a horizontal plane."""
    radius, centre = 41.0, np.array([103.0, 97.0, 52.0])
    sphere = icosphere(4, radius, centre)
    plane_z = centre[2] + 0.5  # 52.5: on faces of the 2.5 m sub-blocks, inside the parents
    below = ~PlaneConstraint((0.0, 0.0, plane_z), (0.0, 0.0, 1.0), "above")
    constraint = SolidConstraint(sphere) & below
    d = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (20, 20, 10), max_depth=3)
    model = BlockModel(d)
    result = subblock_to_constraint(model, constraint)
    model.check_invariants()
    sub = model.centroids(np.flatnonzero(model.level > 0))
    r = np.linalg.norm(sub - centre, axis=1)
    diagonal = math.sqrt(300.0)
    # nothing is refined above the plane (the sphere surface there is outside the AND) ...
    assert (sub[:, 2] < plane_z + 10.0).all()
    # ... nor outside the sphere along the plane; but the plane is refined inside the sphere
    assert (r < radius + diagonal).all()
    on_plane = (np.abs(sub[:, 2] - plane_z) < 2.0) & (r < radius - diagonal)
    on_sphere = (np.abs(r - radius) < 1.5) & (sub[:, 2] < plane_z - diagonal)
    assert on_plane.sum() > 400 and on_sphere.sum() > 1000
    # the plane lies on faces of the level-2 blocks: resolved exactly there, no deeper
    levels = model.level[model.level > 0]
    assert (levels[on_plane] == 2).all() and (levels[on_sphere] == 3).any()
    # analytic volume of the spherical segment below the plane, scaled to the mesh volume
    h = radius + 0.5
    segment = math.pi * h * h * (3.0 * radius - h) / 3.0
    expected = segment * sphere.volume / (4.0 / 3.0 * math.pi * radius**3)
    volume = selected_volume(model, constraint)
    assert volume == pytest.approx(expected, rel=0.005)
    assert result.volume_inside <= expected <= result.volume_inside + result.volume_straddle


# ------------------------------------------------------------------ analytic solids


def test_sphere_volume(capsys):
    radius, centre = 60.0, (151.0, 148.5, 77.0)
    sphere = icosphere(5, radius, centre)
    analytic = 4.0 / 3.0 * math.pi * radius**3
    constraint = SolidConstraint(sphere)
    lines = []
    errors = []
    for depth in range(4):
        d = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (30, 30, 30), max_depth=depth)
        model = BlockModel(d)
        result = subblock_to_constraint(model, constraint)
        model.check_invariants()
        volume = selected_volume(model, constraint)
        errors.append(volume / sphere.volume - 1.0)
        assert result.volume_inside <= sphere.volume <= result.volume_inside + (
            result.volume_straddle
        )
        lines.append(
            f"depth {depth}: {result.n_after:,} blocks, vs mesh {100 * errors[-1]:+.4f}%, "
            f"vs 4/3 pi r^3 {100 * (volume / analytic - 1.0):+.4f}%"
        )
    assert abs(errors[3]) < 0.002
    assert abs(errors[3] + sphere.volume / analytic - 1.0) < 0.004  # vs the analytic sphere
    assert abs(errors[3]) < abs(errors[0])
    with capsys.disabled():
        print("\n[P3.T5 sphere] r = 60 m, 20 480 faces; " + "; ".join(lines))


@pytest.mark.parametrize("rotated", [False, True])
def test_box_aligned_with_block_faces_needs_no_split(rotated):
    kw = {"matrix": ROT_Z30} if rotated else {}
    origin = (1000.0, 2000.0, 300.0)
    d = BlockModelDefinition(origin, (10.0, 10.0, 5.0), (12, 10, 8), max_depth=3, **kw)
    local = box((20.0, 20.0, 10.0), (60.0, 50.0, 30.0))
    solid = Mesh(d.local_to_world(local.vertices), local.faces)
    constraint = SolidConstraint(solid)
    model = BlockModel(d)
    result = subblock_to_constraint(model, constraint)
    assert result.n_split == 0 and result.n_straddle == 0 and model.is_dense
    assert result.n_inside == 4 * 3 * 4
    assert result.volume_inside == 40.0 * 30.0 * 20.0
    if not rotated:
        assert selected_volume(model, constraint) == 24_000.0
    # a box on quarter-block positions is resolved exactly at level 2 and no deeper
    local = box((22.5, 17.5, 11.25), (57.5, 52.5, 28.75))
    solid = Mesh(d.local_to_world(local.vertices), local.faces)
    constraint = SolidConstraint(solid)
    result = subblock_to_constraint(model, constraint)
    model.check_invariants()
    assert result.n_straddle == 0 and result.split_per_level[2] == 0
    assert result.split_per_level[0] > 0 and result.split_per_level[1] > 0
    assert result.volume_inside == 35.0 * 35.0 * 17.5
    if not rotated:
        assert selected_volume(model, constraint) == 35.0 * 35.0 * 17.5
    # strictly conservative classification (no face tolerance) does split at touching faces
    strict = BlockModel(d)
    touched = subblock_to_constraint(strict, constraint, face_tolerance=0.0)
    assert touched.split_per_level[2] > 0 and touched.n_straddle > 0
    if not rotated:
        assert selected_volume(strict, constraint) == 35.0 * 35.0 * 17.5


def test_bench_range_on_block_faces_needs_no_split():
    d = BlockModelDefinition((0.0, 0.0, 100.0), (10.0, 10.0, 5.0), (6, 6, 8), max_depth=2)
    model = BlockModel(d)
    result = subblock_to_constraint(model, RangeConstraint.elevation(110.0, 125.0))
    assert result.n_split == 0 and result.n_inside == 6 * 6 * 3
    result = subblock_to_constraint(model, RangeConstraint.elevation(111.25, 125.0))
    assert result.split_per_level == (36, 36 * 4) and result.n_straddle == 0
    assert selected_volume(model, RangeConstraint.elevation(111.25, 125.0)) == 3600.0 * 13.75


# ------------------------------------------------------------------ options, attributes, merge


def test_max_level_where_and_errors(resolver):
    model = truth_model(3)
    result = subblock_to_constraint(model, ORE, resolver, max_level=1)
    assert result.max_level == 1 and model.level.max() == 1
    assert result.split_per_level[1:] == (0, 0) and result.n_straddle > 0
    full = truth_model(3)
    reference = subblock_to_constraint(full, ORE, resolver)
    assert result.split_per_level[0] == reference.split_per_level[0]
    # continuing from level 1 gives the same model as going down in one call
    subblock_to_constraint(model, ORE, resolver)
    assert_same_structure(model, full)
    # max_level 0 only reports
    dense = truth_model(3)
    report = subblock_to_constraint(dense, ORE, resolver, max_level=0)
    assert report.n_split == 0 and dense.is_dense
    assert report.n_straddle == reference.split_per_level[0]
    # where: only the selected blocks are examined
    half = truth_model(3)
    selection = half.centroids()[:, 0] < half.definition.origin[0] + 200.0
    part = subblock_to_constraint(half, ORE, resolver, where=selection)
    assert 0 < part.n_split < reference.n_split
    assert part.n_inside + part.n_outside + part.n_straddle == selection.sum() + 7 * part.n_split
    sub = half.centroids(np.flatnonzero(half.level > 0))
    assert (sub[:, 0] < half.definition.origin[0] + 200.0).all()
    by_rows = truth_model(3)
    subblock_to_constraint(by_rows, ORE, resolver, where=np.flatnonzero(selection)[::-1])
    assert_same_structure(half, by_rows)
    # errors
    with pytest.raises(ConstraintError, match="geometric"):
        subblock_to_constraint(model, AttributeConstraint("x > 0"), resolver)
    with pytest.raises(ConstraintError, match="geometric"):
        subblock_to_constraint(model, ORE & AttributeConstraint("x > 0"), resolver)
    with pytest.raises(ConstraintError):
        subblock_to_constraint(model, "ore_solid", resolver)
    with pytest.raises(ConstraintError, match="resolver"):
        subblock_to_constraint(truth_model(1), ORE)
    for bad in (-1, 4, 1.5):
        with pytest.raises(ValueError, match="max_level"):
            subblock_to_constraint(model, ORE, resolver, max_level=bad)
    with pytest.raises(ValueError, match="tolerance"):
        subblock_to_constraint(model, ORE, resolver, face_tolerance=1.0)
    with pytest.raises(ValueError, match="chunk_size"):
        subblock_to_constraint(truth_model(1), ORE, resolver, chunk_size=0)


def test_children_inherit_and_merge_uniform_keeps_every_value(resolver):
    model = truth_model(3)
    d = model.definition
    model.add_attribute("zone", "int", background=7)
    model.add_attribute("grade", "float")
    model.set("grade", np.arange(model.n_blocks, dtype=np.float64))
    rng = np.random.default_rng(11)
    points = d.local_to_world(rng.uniform(0.0, 1.0, (200_000, 3)) * np.array(d.extent))
    grade_before = model["grade"][model.locate(points)]
    subblock_to_constraint(model, ORE, resolver)
    n_sub = model.n_blocks
    rows = model.locate(points)
    np.testing.assert_array_equal(model["grade"][rows], grade_before)
    assert (model["zone"] == 7).all()
    # code, then merge: fewer blocks, same value at every point, same volume per code
    model.add_attribute("ore", "int", dtype="int8")
    mask = ORE.mask(model, resolver)
    model.set("ore", 1, mask)
    volume_before = float(model.volumes()[mask].sum())
    ore_before = model["ore"][rows]
    with pytest.raises(ValueError, match="at least one attribute"):
        merge_uniform(model, [])
    merges = merge_uniform(model)
    model.check_invariants()
    assert merges > 0 and model.n_blocks < 0.7 * n_sub
    rows = model.locate(points)
    np.testing.assert_array_equal(model["ore"][rows], ore_before)
    np.testing.assert_array_equal(model["grade"][rows], grade_before)
    assert float(model.volumes()[model["ore"] == 1].sum()) == volume_before
    # nothing more can merge, and every remaining sub-block borders the other code:
    # its sibling set (the cell one level up) holds both codes
    assert merge_uniform(model) == 0
    sub = np.flatnonzero(model.level > 0)
    lo, hi = cell_boxes(model, sub, model.level[sub].astype(np.int64) - 1)
    _, cell, count = np.unique(
        np.concatenate([lo, hi], axis=1), axis=0, return_inverse=True, return_counts=True
    )
    cell = cell.reshape(-1)
    complete = count[cell] == 8  # all eight siblings are leaves
    assert complete.sum() > 100_000
    mean = np.bincount(cell, weights=model["ore"][sub].astype(np.float64)) / count
    assert ((mean > 0.0) & (mean < 1.0))[cell][complete].all()
    # merging on a subset: "zone" is uniform everywhere, so everything collapses to parents
    assert merge_uniform(model, "zone") > 0
    assert model.is_dense


def test_merge_uniform_without_attributes_is_refused():
    d = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (2, 2, 2), max_depth=1)
    model = BlockModel(d)
    model.split(np.arange(3))
    with pytest.raises(ValueError, match="at least one attribute"):
        merge_uniform(model)
    with pytest.raises(KeyError):
        merge_uniform(model, ["nope"])


# ------------------------------------------------------------------ performance


def test_performance_depth_3_on_solid_and_topography(resolver, capsys):
    subblock_to_constraint(truth_model(1), ABOVE_TOPO | ORE, resolver)  # JIT warm-up
    model = truth_model(3)
    t0 = time.perf_counter()
    r_topo = subblock_to_constraint(model, ABOVE_TOPO, resolver)
    t_topo = time.perf_counter() - t0
    t0 = time.perf_counter()
    r_ore = subblock_to_constraint(model, ORE, resolver)
    t_ore = time.perf_counter() - t0
    model.check_invariants()
    assert t_topo + t_ore < 30.0  # measured: a few seconds
    assert model.n_blocks == r_ore.n_after == r_topo.n_after + 7 * r_ore.n_split
    with capsys.disabled():
        print(
            f"\n[P3.T5 speed] truth grid ({model.definition.n_parents:,} parents) to depth 3: "
            f"topography {t_topo:.2f} s ({r_topo.n_split:,} splits), ore solid {t_ore:.2f} s "
            f"({r_ore.n_split:,} splits) -> {model.n_blocks:,} blocks, "
            f"{model.nbytes / 1e6:.0f} MB of structure"
        )


def test_performance_five_million_leaves(resolver, capsys):
    """Depth 4 (0.625 x 0.625 x 0.3125 m sub-blocks) on topography + ore solid: 5.75 M leaves."""
    model = truth_model(4)
    t0 = time.perf_counter()
    result = subblock_to_constraint(model, ABOVE_TOPO | ORE, resolver)
    elapsed = time.perf_counter() - t0
    assert result.n_after == model.n_blocks > 5_000_000
    assert elapsed < 120.0  # measured: a few seconds
    assert model.nbytes == 8 * model.n_blocks  # structure only: 8 bytes per block
    t0 = time.perf_counter()
    model.check_invariants()
    volume = selected_volume(model, ORE, resolver)
    t_mask = time.perf_counter() - t0
    assert volume == pytest.approx(MESH_VOLUME, rel=1e-4)
    with capsys.disabled():
        print(
            f"\n[P3.T5 scale] depth 4 on topography + ore solid: {result.n_after:,} leaves in "
            f"{elapsed:.2f} s ({result.n_after / elapsed / 1e6:.1f} M leaves/s), "
            f"{model.nbytes / 1e6:.0f} MB of structure; invariants + ore centroid mask "
            f"{t_mask:.2f} s; ore volume vs mesh {100 * (volume / MESH_VOLUME - 1):+.4f}%"
        )


def test_memory_is_bounded_at_reduced_size(resolver, capsys):
    """Peak traced memory of a depth-2 run (tracemalloc slows numpy too much for depth 4)."""
    import tracemalloc

    subblock_to_constraint(truth_model(1), ABOVE_TOPO | ORE, resolver)  # caches, JIT
    peaks = {}
    for chunk in (1 << 18, 1 << 14):
        model = truth_model(2)
        tracemalloc.start()
        try:
            result = subblock_to_constraint(model, ABOVE_TOPO | ORE, resolver, chunk_size=chunk)
            peaks[chunk] = tracemalloc.get_traced_memory()[1]
        finally:
            tracemalloc.stop()
    assert result.n_after > 400_000
    # the working memory of the classification is set by the chunk size, not the model size
    assert peaks[1 << 14] < peaks[1 << 18] < 400e6
    assert peaks[1 << 14] < 150 * model.n_blocks  # bytes per final block, generous
    with capsys.disabled():
        print(
            f"\n[P3.T5 memory] depth 2, {result.n_after:,} blocks: peak traced memory "
            f"{peaks[1 << 18] / 1e6:.0f} MB with 262 144-box chunks, {peaks[1 << 14] / 1e6:.0f} MB "
            f"with 16 384-box chunks"
        )
