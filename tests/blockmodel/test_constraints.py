"""P3.T4 acceptance: constraints, combinations, JSON and box classification."""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import trimesh
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    INSIDE,
    OUTSIDE,
    STRADDLE,
    And,
    AttributeConstraint,
    BlockIdConstraint,
    CombinedConstraint,
    ConstraintError,
    DictResolver,
    DtmConstraint,
    FileResolver,
    GeometryResolver,
    Not,
    Or,
    PlaneConstraint,
    PolygonConstraint,
    RangeConstraint,
    SolidConstraint,
    combine,
    constraint_from_dict,
    load_constraint,
    save_constraint,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.geometry.dtm import DTM, dtm_from_grid
from bakumine.core.geometry.inside import points_in_solid
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, icosphere
from bakumine.core.geometry.section import plane_frame

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
ORIGIN = (500_400.0, 5_300_250.0, 825.0)
ROTATION = {"angles": (35.0, 10.0, -20.0)}


def small_model(rotated=False, depth=2, split=True, seed=0):
    kw = ROTATION if rotated else {}
    m = BlockModel(BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (8, 7, 6), max_depth=depth,
                                        **kw))  # fmt: skip
    if split:
        rng = np.random.default_rng(seed)
        for _ in range(depth):
            m.split(rng.random(m.n_blocks) < 0.2, strict=False)
    return m


def crossing_number(u, v, pu, pv):
    """Independent numpy point-in-polygon (even-odd)."""
    inside = np.zeros(len(u), dtype=bool)
    n = len(pu)
    for k in range(n):
        j = (k - 1) % n
        cond = (pv[k] > v) != (pv[j] > v)
        with np.errstate(divide="ignore", invalid="ignore"):
            xc = pu[k] + (v - pv[k]) * (pu[j] - pu[k]) / (pv[j] - pv[k])
        inside ^= cond & (u < xc)
    return inside


@pytest.fixture(scope="module")
def resolver():
    return FileResolver(DATA)


@pytest.fixture(scope="module")
def truth():
    info = json.loads((DATA / "truth.json").read_text())["blocks"]
    table = pd.read_parquet(DATA / "truth_blocks.parquet")
    d = BlockModelDefinition(info["origin"], info["size"], info["shape"], max_depth=3)
    rows = d.flat_index(table[["i", "j", "k"]].to_numpy())
    rock = np.empty(len(table), dtype=object)
    rock[rows] = table["rock"].to_numpy()
    return d, rock


# ------------------------------------------------------------------ primitives vs manual masks


@pytest.mark.parametrize("rotated", [False, True])
def test_plane_constraint(rotated):
    m = small_model(rotated)
    c = m.centroids()
    point = np.array(ORIGIN) + (37.0, 31.0, 14.0)
    normal = np.array((0.3, -0.5, 0.8))
    manual = (c - point) @ (normal / np.linalg.norm(normal)) > 0
    assert 0.05 < manual.mean() < 0.95
    np.testing.assert_array_equal(PlaneConstraint(point, normal).mask(m), manual)
    np.testing.assert_array_equal(PlaneConstraint(point, normal, "below").mask(m), ~manual)
    np.testing.assert_array_equal(PlaneConstraint(point, 7 * normal).mask(m), manual)
    # horizontal plane: a centroid exactly on the plane is "below"
    level = c[0, 2]
    flat = PlaneConstraint((0, 0, level), (0, 0, 1))
    np.testing.assert_array_equal(flat.mask(m), c[:, 2] > level)
    np.testing.assert_array_equal(PlaneConstraint((0, 0, level), (0, 0, 1), "below").mask(m),
                                  c[:, 2] <= level)  # fmt: skip
    # three points: z = 840 + 0.2 (x - x0); the normal is turned upwards whatever the order
    p1 = np.array((ORIGIN[0], ORIGIN[1], 840.0))
    p2, p3 = p1 + (10.0, 0.0, 2.0), p1 + (0.0, 10.0, 0.0)
    manual = c[:, 2] > 840.0 + 0.2 * (c[:, 0] - ORIGIN[0])
    np.testing.assert_array_equal(PlaneConstraint.from_points(p1, p2, p3).mask(m), manual)
    np.testing.assert_array_equal(PlaneConstraint.from_points(p1, p3, p2).mask(m), manual)
    # strike north, dipping 30 degrees to the east: z falls by tan(30) per metre east
    sd = PlaneConstraint.from_strike_dip(p1, 0.0, 30.0)
    manual = c[:, 2] > 840.0 - np.tan(np.pi / 6) * (c[:, 0] - ORIGIN[0])
    np.testing.assert_array_equal(sd.mask(m), manual)
    # strike east (90), right-hand rule: dips south, so the plane rises to the north
    sd = PlaneConstraint.from_strike_dip(p1, 90.0, 45.0)
    height = (c[:, 2] - 840.0) - (c[:, 1] - ORIGIN[1])
    clear = np.abs(height) > 1e-6
    assert 0.02 < (height > 0).mean() < 0.98
    np.testing.assert_array_equal(sd.mask(m)[clear], (height > 0)[clear])
    for bad in (
        lambda: PlaneConstraint(point, (0, 0, 0)),
        lambda: PlaneConstraint(point, normal, "left"),
        lambda: PlaneConstraint.from_points(p1, p1 + 1.0, p1 + 2.0),
        lambda: PlaneConstraint.from_strike_dip(p1, 0.0, 120.0),
        lambda: PlaneConstraint((0, 0), normal),
    ):
        with pytest.raises(ConstraintError):
            bad()


@pytest.mark.parametrize("rotated", [False, True])
def test_polygon_constraint(rotated):
    m = small_model(rotated)
    c = m.centroids()
    x, y, z = (c[:, a] - ORIGIN[a] for a in range(3))
    # L-shape in plan = union of two rectangles (half-open like the crossing rule)
    ell = np.array([(10, 10), (60, 10), (60, 30), (30, 30), (30, 60), (10, 60)], dtype=float)
    ell3 = np.column_stack([ell + ORIGIN[:2], np.full(6, 123.0)])  # z of the string is ignored
    manual = ((x >= 10) & (x < 60) & (y >= 10) & (y < 30)) | (
        (x >= 10) & (x < 30) & (y >= 30) & (y < 60)
    )
    edge = (np.abs(np.stack([x - 10, x - 60, x - 30, y - 10, y - 30, y - 60])) < 1e-6).any(axis=0)
    assert 0.1 < manual.mean() < 0.9
    got = PolygonConstraint(ell3).mask(m)
    np.testing.assert_array_equal(got[~edge], manual[~edge])
    if not rotated:
        np.testing.assert_array_equal(got, manual)  # exact, including centroids on the outline
    np.testing.assert_array_equal(PolygonConstraint(ell3, "outside").mask(m), ~got)
    # (n, 2) input, a repeated closing point, and reversed winding give the same mask
    closed = np.vstack([ell + ORIGIN[:2], ell[:1] + ORIGIN[:2]])
    np.testing.assert_array_equal(PolygonConstraint(closed).mask(m), got)
    np.testing.assert_array_equal(PolygonConstraint(closed[::-1]).mask(m)[~edge], got[~edge])
    # limits along the projection axis (an elevation range), half-open
    lim = PolygonConstraint(ell3, limits=(835.0, 845.0)).mask(m)
    np.testing.assert_array_equal(lim, got & (c[:, 2] >= 835.0) & (c[:, 2] < 845.0))
    np.testing.assert_array_equal(PolygonConstraint(ell3, limits=(None, 845.0)).mask(m),
                                  got & (c[:, 2] < 845.0))  # fmt: skip
    np.testing.assert_array_equal(
        PolygonConstraint(ell3, "outside", limits=(835.0, 845.0)).mask(m), ~lim
    )
    # projection along x: polygon in (y, z); along y: polygon in (x, z)
    tri_yz = np.array([(ORIGIN[1] + 5, 828.0), (ORIGIN[1] + 65, 830.0), (ORIGIN[1] + 30, 853.0)])
    np.testing.assert_array_equal(
        PolygonConstraint(tri_yz, axis="x").mask(m),
        crossing_number(c[:, 1], c[:, 2], tri_yz[:, 0], tri_yz[:, 1]),
    )
    tri_xz = np.array([(ORIGIN[0] + 5, 828.0), (ORIGIN[0] + 75, 830.0), (ORIGIN[0] + 30, 853.0)])
    along_y = PolygonConstraint(tri_xz, axis="y", limits=(ORIGIN[1] + 20, ORIGIN[1] + 50)).mask(m)
    manual = crossing_number(c[:, 0], c[:, 2], tri_xz[:, 0], tri_xz[:, 1]) & (y >= 20) & (y < 50)
    np.testing.assert_array_equal(along_y, manual)
    assert 0 < along_y.sum() < m.n_blocks
    # arbitrary direction: a triangular prism; manual test by half-planes in the plane frame
    direction = np.array((1.0, 2.0, 0.5))
    u, v, w = plane_frame(direction)
    centre = np.array(ORIGIN) + (40.0, 35.0, 15.0)
    tri = centre + np.array([(-18 * u - 12 * v), (20 * u - 10 * v), (2 * u + 22 * v)])
    pu, pv = (c - centre) @ u, (c - centre) @ v
    tu, tv = (tri - centre) @ u, (tri - centre) @ v
    signs = []
    for a in range(3):
        b = (a + 1) % 3
        signs.append((tu[b] - tu[a]) * (pv - tv[a]) - (tv[b] - tv[a]) * (pu - tu[a]))
    signs = np.array(signs)
    manual = (signs > 0).all(axis=0) | (signs < 0).all(axis=0)
    near = (np.abs(signs) < 1e-5).any(axis=0)
    got = PolygonConstraint(tri, axis=direction).mask(m)
    assert 0.02 < manual.mean() < 0.9
    np.testing.assert_array_equal(got[~near], manual[~near])
    wc = (c - tri[0]) @ w  # limits are measured from the first polygon point
    mid = float(np.median(wc))
    assert abs(mid) < 60.0
    got = PolygonConstraint(tri, axis=direction, limits=(mid, None)).mask(m)
    np.testing.assert_array_equal(got[~near], (manual & (wc >= mid))[~near])
    for bad in (
        lambda: PolygonConstraint(ell[:2]),
        lambda: PolygonConstraint(ell, axis="w"),
        lambda: PolygonConstraint(ell, axis=(0, 0, 0)),
        lambda: PolygonConstraint(ell, axis=(1, 1, 0)),  # needs 3D points
        lambda: PolygonConstraint(ell, limits=(5, 1)),
        lambda: PolygonConstraint(ell, side="in"),
    ):
        with pytest.raises(ConstraintError):
            bad()


@pytest.mark.parametrize("rotated", [False, True])
def test_range_and_block_id_constraints(rotated):
    m = small_model(rotated)
    c, ijk = m.centroids(), m.ijk()
    got = RangeConstraint(x=(ORIGIN[0] + 20, ORIGIN[0] + 50), z=(None, 845.0)).mask(m)
    manual = (c[:, 0] >= ORIGIN[0] + 20) & (c[:, 0] < ORIGIN[0] + 50) & (c[:, 2] < 845.0)
    np.testing.assert_array_equal(got, manual)
    np.testing.assert_array_equal(RangeConstraint(y=(ORIGIN[1] + 31, None)).mask(m),
                                  c[:, 1] >= ORIGIN[1] + 31)  # fmt: skip
    # index ranges are inclusive and use the parent index for sub-blocks
    got = RangeConstraint(i=(2, 4), j=(None, 3), k=(1, 1)).mask(m)
    manual = (ijk[:, 0] >= 2) & (ijk[:, 0] <= 4) & (ijk[:, 1] <= 3) & (ijk[:, 2] == 1)
    np.testing.assert_array_equal(got, manual)
    assert manual.any() and (m.level[manual] > 0).any()
    # adjacent benches partition the model
    levels = 825.0 + 5.0 * np.arange(-30, 40)
    benches = [RangeConstraint.elevation(z0, z0 + 5.0).mask(m) for z0 in levels]
    np.testing.assert_array_equal(np.sum(benches, axis=0), 1)
    if not rotated:
        np.testing.assert_array_equal(benches[30 + 2], ijk[:, 2] == 2)
    for bad in (
        lambda: RangeConstraint(),
        lambda: RangeConstraint(x=(5, 1)),
        lambda: RangeConstraint(i=(0.5, 2)),
        lambda: RangeConstraint(z=5),
    ):
        with pytest.raises(ConstraintError):
            bad()

    # explicit block ids: stable under splits of other blocks
    pick = np.zeros(m.n_blocks, dtype=bool)
    pick[[0, 5, 17, m.n_blocks - 1]] = True
    ids = BlockIdConstraint.from_rows(m, pick)
    np.testing.assert_array_equal(ids.mask(m), pick)
    np.testing.assert_array_equal(BlockIdConstraint(m.block_ids([5, 0])).mask(m).nonzero()[0],
                                  [0, 5])  # fmt: skip
    centroids = m.centroids()[pick]
    others = np.flatnonzero(~pick & (m.level < m.max_level))[:10]
    m.split(others)
    np.testing.assert_array_equal(m.centroids()[ids.mask(m)], centroids)
    target = np.flatnonzero(ids.mask(m))[0]
    if m.level[target] < m.max_level:
        m.split([target])  # a block that was split is no longer selected
        assert ids.mask(m).sum() == 3
    assert not BlockIdConstraint([]).mask(m).any()
    with pytest.raises(ConstraintError):
        BlockIdConstraint([1.5])


def test_attribute_constraint():
    m = small_model()
    rng = np.random.default_rng(2)
    m.add_attribute("au")
    m.add_attribute("rock", "category")
    au = rng.lognormal(0, 1, m.n_blocks)
    au[::7] = np.nan
    rock = rng.choice(np.array(["ORE", "WASTE", None], dtype=object), m.n_blocks)
    m.set("au", au)
    m.set("rock", rock)
    got = AttributeConstraint('au > 0.5 and rock == "ORE"').mask(m)
    np.testing.assert_array_equal(got, (au > 0.5) & (rock == "ORE"))
    assert got.any()
    np.testing.assert_array_equal(AttributeConstraint("rock != 'ORE' or z > 840").mask(m),
                                  (rock != "ORE") | (m.centroids()[:, 2] > 840))  # fmt: skip
    # a numeric expression selects non-zero, non-missing values
    np.testing.assert_array_equal(AttributeConstraint("au - 1").mask(m), ~np.isnan(au))
    np.testing.assert_array_equal(AttributeConstraint("volume < 500").mask(m), m.level > 0)
    c = AttributeConstraint("au / 0 > 1")  # no warning, no crash
    assert c.mask(m).sum() == (~np.isnan(au)).sum()
    assert not c.is_geometric
    with pytest.raises(ConstraintError, match="unknown function"):
        AttributeConstraint("__import__('os')")
    with pytest.raises(ValueError, match="unknown name"):
        AttributeConstraint("cu > 1").mask(m)
    with pytest.raises(ConstraintError, match="not geometric"):
        c.classify_boxes(m.definition, *m.bounds_local())


@pytest.mark.parametrize("rotated", [False, True])
def test_dtm_constraint(rotated):
    m = small_model(rotated)
    c = m.centroids()
    lo, hi = m.definition.bounding_box()
    # planar surface z = 840 + 0.1 dx - 0.05 dy covering only part of the model in plan
    x0, y0 = lo[0] + 15.0, lo[1] - 5.0
    nodes_x, nodes_y = x0 + np.arange(9) * 10.0, y0 + np.arange(30) * 10.0
    z = 840.0 + 0.1 * (nodes_x[:, None] - ORIGIN[0]) - 0.05 * (nodes_y[None, :] - ORIGIN[1])
    dtm = dtm_from_grid(z, (x0, y0), (10.0, 10.0))
    surface = 840.0 + 0.1 * (c[:, 0] - ORIGIN[0]) - 0.05 * (c[:, 1] - ORIGIN[1])
    covered = (c[:, 0] >= x0) & (c[:, 0] <= nodes_x[-1])
    covered &= (c[:, 1] >= y0) & (c[:, 1] <= nodes_y[-1])
    near = np.abs(c[:, 2] - surface) < 1e-6
    assert 0.2 < covered.mean() < 0.95
    above, below = c[:, 2] > surface, c[:, 2] <= surface
    for side, manual in (("above", above), ("below", below)):
        got = DtmConstraint(dtm, side).mask(m)
        np.testing.assert_array_equal(got[~near], (manual & covered)[~near])
        got = DtmConstraint(dtm, side, outside=True).mask(m)
        np.testing.assert_array_equal(got[~near], (manual | ~covered)[~near])
        with pytest.raises(ConstraintError, match="outside the plan extent"):
            DtmConstraint(dtm, side, outside="error").mask(m)
    # complementary where covered
    a, b = DtmConstraint(dtm, "above").mask(m), DtmConstraint(dtm, "below").mask(m)
    np.testing.assert_array_equal(a ^ b, covered)
    # a surface covering everything: "error" passes and equals the default
    big = dtm_from_grid(np.full((2, 2), 841.0), (lo[0] - 10, lo[1] - 10),
                        (hi[0] - lo[0] + 20, hi[1] - lo[1] + 20))  # fmt: skip
    np.testing.assert_array_equal(DtmConstraint(big, outside="error").mask(m), c[:, 2] > 841.0)
    # by name through a resolver
    res = DictResolver(dtms={"s": big})
    np.testing.assert_array_equal(DtmConstraint("s").mask(m, res), c[:, 2] > 841.0)
    with pytest.raises(ConstraintError, match="pass a resolver"):
        DtmConstraint("s").mask(m)
    with pytest.raises(ConstraintError, match="unknown dtm"):
        DtmConstraint("nope").mask(m, res)
    with pytest.raises(ConstraintError):
        DtmConstraint(big, outside="maybe")
    with pytest.raises(ConstraintError):
        DtmConstraint(3)


@pytest.mark.parametrize("rotated", [False, True])
def test_solid_constraint(rotated):
    m = small_model(rotated)
    c = m.centroids()
    # axis-aligned box on block faces: exact half-open membership in the unrotated model
    blo, bhi = np.array(ORIGIN) + (20.0, 10.0, 5.0), np.array(ORIGIN) + (60.0, 50.0, 20.0)
    solid = box(blo, bhi)
    manual = ((c >= blo) & (c < bhi)).all(axis=1)
    near = (np.abs(c - blo) < 1e-6).any(axis=1) | (np.abs(c - bhi) < 1e-6).any(axis=1)
    got = SolidConstraint(solid).mask(m)
    assert 0.05 < manual.mean() < 0.9
    np.testing.assert_array_equal(got[~near], manual[~near])
    np.testing.assert_array_equal(SolidConstraint(solid, "outside").mask(m), ~got)
    # sphere
    centre, radius = np.array(ORIGIN) + (40.0, 35.0, 15.0), 14.0
    sphere = icosphere(4, radius, centre)
    dist = np.linalg.norm(c - centre, axis=1)
    got = SolidConstraint(sphere).mask(m)
    clear = np.abs(dist - radius) > 0.1  # the icosphere is inscribed, within 0.1 m of the sphere
    assert got.sum() > 30
    np.testing.assert_array_equal(got[clear], (dist < radius)[clear])
    np.testing.assert_array_equal(SolidConstraint("s").mask(m, DictResolver(solids={"s": sphere})),
                                  got)  # fmt: skip
    # an open mesh is refused
    with pytest.raises(ValueError, match="closed"):
        SolidConstraint(sphere.with_faces(sphere.faces[:-1])).mask(m)
    with pytest.raises(ConstraintError):
        SolidConstraint(sphere, "in")
    with pytest.raises(ConstraintError):
        SolidConstraint(np.zeros((3, 3)))


# ------------------------------------------------------------------ truth


def test_solid_inside_equals_truth_ore(truth, resolver, capsys):
    d, rock = truth
    m = BlockModel(d)
    t0 = time.perf_counter()
    inside = SolidConstraint("ore_solid").mask(m, resolver)
    elapsed = time.perf_counter() - t0
    ore = rock == "ORE"
    wrong = np.flatnonzero(inside != ore)
    # the truth is the analytic lens, the solid its 20 480-face mesh: they can only differ
    # for centroids next to the surface. Distance to the mesh by trimesh (independent).
    assert len(wrong) <= 10 and len(wrong) / d.n_parents < 1e-4
    assert inside.sum() == pytest.approx(ore.sum(), rel=0.005)
    mesh = resolver.solid("ore_solid")
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    distance = np.zeros(0)
    if len(wrong):
        distance = trimesh.proximity.closest_point(tm, m.centroids()[wrong])[1]
        assert distance.max() < 0.05
    with capsys.disabled():
        print(
            f"\n[P3.T4 truth] solid inside vs truth ORE: {len(wrong)} of {d.n_parents:,} blocks "
            f"differ ({ore.sum()} ore blocks); max distance of those centroids to the mesh "
            f"{distance.max() if len(wrong) else 0.0:.4f} m; mask (incl. JIT) {elapsed:.2f} s"
        )


def test_dtm_above_equals_truth_air(truth, resolver, capsys):
    d, rock = truth
    m = BlockModel(d)
    above = DtmConstraint("topo", "above", outside="error").mask(m, resolver)
    air = rock == "AIR"
    wrong = np.flatnonzero(above != air)
    # truth = analytic topography; the DTM samples it every 25 m, so the two surfaces differ
    # by the chord error of the triangles. Only centroids that close to the DTM may differ.
    height = np.abs(resolver.dtm("topo").height_above(m.centroids()))
    assert len(wrong) / d.n_parents < 5e-4
    assert height[wrong].max() < 0.75 if len(wrong) else True
    # ... and every centroid farther than that from the surface agrees
    np.testing.assert_array_equal(above[height >= 0.75], air[height >= 0.75])
    below = DtmConstraint("topo", "below").mask(m, resolver)
    np.testing.assert_array_equal(below, ~above)
    with capsys.disabled():
        print(
            f"\n[P3.T4 truth] DTM above vs truth AIR: {len(wrong)} of {d.n_parents:,} blocks "
            f"differ ({air.sum()} air blocks); max |height above DTM| of those "
            f"{height[wrong].max() if len(wrong) else 0.0:.3f} m"
        )


# ------------------------------------------------------------------ dense fast path


def test_dense_fast_path_equals_point_path_exactly(truth, resolver):
    d, _ = truth
    mesh = resolver.solid("ore_solid")
    awkward = BlockModelDefinition((500_400.1, 5_300_250.3, 825.7), (10.0, 0.7 * 13, 1 / 0.3),
                                   (39, 57, 120), max_depth=1)  # fmt: skip
    rot = BlockModelDefinition.auto_extent(mesh, (10.0, 10.0, 5.0), margin=10.0, max_depth=1,
                                           angles=(30.0, 15.0, 55.0))  # fmt: skip
    rot2 = BlockModelDefinition.auto_extent(mesh, (7.0, 9.0, 3.0), margin=5.0, snap=False,
                                            max_depth=1, angles=(-110.0, -5.0, 20.0))  # fmt: skip
    for definition in (d, awkward, rot, rot2):
        m = BlockModel(definition)
        c = SolidConstraint(mesh)
        fast = c.mask(m)
        slow = c.mask(m, dense=False)
        assert 1000 < fast.sum() < m.n_blocks / 2, definition
        np.testing.assert_array_equal(fast, slow)
        if not definition.is_rotated:
            np.testing.assert_array_equal(fast, points_in_solid(mesh, m.centroids()))
        else:
            # against the unrotated mesh: may differ only at rounding distance; none here
            assert (fast != points_in_solid(mesh, m.centroids())).sum() <= 2
        np.testing.assert_array_equal(SolidConstraint(mesh, "outside").mask(m), ~fast)
        # after splitting, the blocks that are still parents keep their answer (point path)
        rng = np.random.default_rng(1)
        result = m.split(rng.random(m.n_blocks) < 0.05)
        assert not m.is_dense
        sub = c.mask(m)
        np.testing.assert_array_equal(sub[~result.is_child], fast[result.source[~result.is_child]])
        m.merge([])
        np.testing.assert_array_equal(c.mask(m), fast)


# ------------------------------------------------------------------ combinations


def test_composite_expressions_equal_manual_boolean_algebra(truth, resolver):
    d, _ = truth
    m = BlockModel(d)
    m.split(np.random.default_rng(0).random(m.n_blocks) < 0.01)
    m.add_attribute("au", background=0.0)
    m.set("au", np.random.default_rng(1).random(m.n_blocks))
    named = {
        "ore": SolidConstraint("ore_solid"),
        "topo": DtmConstraint("topo", "below"),
        "bench": RangeConstraint.elevation(950.0, 1050.0),
        "plane": PlaneConstraint((500_600.0, 5_300_500.0, 1000.0), (1.0, 1.0, 0.2)),
        "grade": AttributeConstraint("au > 0.4"),
    }
    masks = {name: c.mask(m, resolver) for name, c in named.items()}
    a, b, c_, p, g = (masks[k] for k in ("ore", "topo", "bench", "plane", "grade"))
    assert all(0 < v.sum() < m.n_blocks for v in masks.values())
    cases = {
        "ore and topo": a & b,
        "ore and (bench or not plane)": a & (c_ | ~p),
        "not ore and not (topo and bench)": ~a & ~(b & c_),
        "ore or topo and bench": a | (b & c_),  # and binds tighter than or
        "(ore or topo) and bench": (a | b) & c_,
        "ore AND (topo OR NOT plane) And grade": a & (b | ~p) & g,
        "not not ore": a,
        "ore": a,
        "ore and topo and bench and plane and grade": a & b & c_ & p & g,
    }
    for text, manual in cases.items():
        combined = combine(text, named)
        assert isinstance(combined, CombinedConstraint)
        np.testing.assert_array_equal(combined.mask(m, resolver), manual, err_msg=text)
    # the same as a tree and with operators
    ore, topo, bench, plane, grade = (named[k] for k in ("ore", "topo", "bench", "plane", "grade"))
    tree = And(ore, Or(bench, Not(plane)))
    np.testing.assert_array_equal(tree.mask(m, resolver), a & (c_ | ~p))
    np.testing.assert_array_equal((ore & (bench | ~plane)).mask(m, resolver), a & (c_ | ~p))
    np.testing.assert_array_equal(Or(ore, topo, grade).mask(m, resolver), a | b | g)
    np.testing.assert_array_equal((~(ore | topo) & grade).mask(m, resolver), ~(a | b) & g)
    assert isinstance(combine("ore and (bench or not plane)", named).tree, And)
    assert combine("ore and bench", named).is_geometric and tree.is_geometric
    assert not combine("ore and grade", named).is_geometric
    assert set(combine("ore and bench", named).constraints) == {"ore", "bench"}
    for text in ("ore and missing", "ore + topo", "ore and 1", "ore > topo", "ore and", "",
                 "ore.mask", "__import__('os')", "ore if topo else bench", "f(ore)"):  # fmt: skip
        with pytest.raises(ConstraintError):
            combine(text, named)
    with pytest.raises(ConstraintError):
        And()
    with pytest.raises(ConstraintError):
        Not("ore")
    with pytest.raises(ConstraintError):
        combine("ore", {"ore": 5})


# ------------------------------------------------------------------ JSON


def test_json_round_trip_gives_identical_masks(truth, resolver, tmp_path):
    d, _ = truth
    m = BlockModel(d)
    m.split(np.random.default_rng(3).random(m.n_blocks) < 0.01)
    m.add_attribute("au", background=0.0)
    m.add_attribute("rock", "category", background="WASTE")
    m.set("au", np.random.default_rng(1).random(m.n_blocks))
    m.set("rock", "ORE", SolidConstraint("ore_solid").mask(m, resolver))
    pit = np.array([(500_450.0, 5_300_300.0), (500_700.0, 5_300_320.0), (500_720.0, 5_300_700.0),
                    (500_560.0, 5_300_520.0), (500_430.0, 5_300_690.0)])  # fmt: skip
    np.save(tmp_path / "pit.npy", pit)
    for stem in ("ore_solid", "topo"):
        for part in ("vertices", "faces"):
            np.save(tmp_path / f"{stem}_{part}.npy", np.load(DATA / f"{stem}_{part}.npy"))
    files = FileResolver(tmp_path)
    assert isinstance(files, GeometryResolver) and isinstance(DictResolver(), GeometryResolver)
    primitives = {
        "dtm": DtmConstraint("topo", "below", outside=True),
        "dtm_error": DtmConstraint("topo", "above", outside="error"),
        "solid": SolidConstraint("ore_solid", "outside"),
        "attribute": AttributeConstraint('au > 0.5 and rock == "ORE"'),
        "plane": PlaneConstraint.from_strike_dip((500_600.0, 5_300_500.0, 1000.0), 30.0, 55.0),
        "polygon_inline": PolygonConstraint(pit, limits=(900.0, None)),
        "polygon_named": PolygonConstraint("pit", "outside"),
        "polygon_dir": PolygonConstraint(np.column_stack([pit, np.full(5, 1000.0)]),
                                         axis=(0.2, 0.1, 1.0), limits=(None, 100.0)),  # fmt: skip
        "polygon_x": PolygonConstraint([(5_300_300.0, 900.0), (5_300_700.0, 950.0),
                                        (5_300_500.0, 1150.0)], axis="x"),  # fmt: skip
        "range": RangeConstraint(z=(950.0, 1050.0), i=(3, 30), y=(None, 5_300_600.0)),
        "ids": BlockIdConstraint.from_rows(m, np.arange(0, m.n_blocks, 37)),
    }
    composite = combine("solid and (dtm or not plane) and not polygon_inline or ids", primitives)
    tree = And(primitives["range"], Or(primitives["attribute"], Not(primitives["polygon_named"])))
    for name, constraint in {**primitives, "composite": composite, "tree": tree}.items():
        before = constraint.mask(m, files)
        assert 0 < before.sum() < m.n_blocks, name
        data = constraint.to_dict()
        text = json.dumps(data)  # plain JSON types only
        again = constraint_from_dict(json.loads(text))
        assert type(again) is type(constraint)
        np.testing.assert_array_equal(again.mask(m, files), before, err_msg=name)
        assert again.to_dict() == data
        path = tmp_path / f"{name}.json"
        save_constraint(constraint, path)
        np.testing.assert_array_equal(load_constraint(path).mask(m, files), before, err_msg=name)
        assert "Constraint(" in repr(again) or type(again) in (And, Or, Not)
    assert json.loads((tmp_path / "composite.json").read_text())["constraint"]["expression"] == (
        "solid and (dtm or not plane) and not polygon_inline or ids"
    )
    # geometry held as objects is not written out
    for constraint in (SolidConstraint(files.solid("ore_solid")), DtmConstraint(files.dtm("topo"))):
        with pytest.raises(ConstraintError, match="not written to JSON"):
            constraint.to_dict()
        with pytest.raises(ConstraintError):
            save_constraint(constraint & primitives["range"], tmp_path / "x.json")
    for bad in ({}, {"type": "nope"}, {"type": "plane", "point": [0, 0, 0]}, [1], {"type": "and"}):
        with pytest.raises(ConstraintError):
            constraint_from_dict(bad)
    (tmp_path / "bad.json").write_text('{"constraint": {"type": "range", "z": [0, 1]}}')
    with pytest.raises(ConstraintError, match="format"):
        load_constraint(tmp_path / "bad.json")
    # the file resolver does not leave its directory
    for name in ("../ore_solid", "/etc/passwd", "a/b", "..", ""):
        with pytest.raises(ConstraintError):
            files.solid(name)
    with pytest.raises(ConstraintError, match="unknown solid"):
        SolidConstraint("missing").mask(m, files)


# ------------------------------------------------------------------ box classification


def _oracles(definition, mesh, dtm, part_dtm):
    """Constraint + independent point-wise evaluation (world points -> bool)."""
    centre = np.array(definition.local_to_world(0.5 * np.array(definition.extent)))
    extent = np.array(definition.extent)
    normal = np.array((0.4, -0.3, 0.85))
    normal /= np.linalg.norm(normal)
    ring = np.array([(-0.3, -0.35), (0.25, -0.3), (0.05, 0.0), (0.3, 0.3), (-0.2, 0.35)])
    pit = centre[:2] + ring * extent[:2]
    direction = np.array((0.3, -0.2, 1.0))
    u, v, w = plane_frame(direction)
    pit3 = np.column_stack([pit, np.full(len(pit), centre[2])])
    z_lo, z_hi = centre[2] - 0.2 * extent[2], centre[2] + 0.25 * extent[2]

    def dtm_oracle(surface, side, outside):
        def f(p):
            h = surface.height_above(p)
            sel = h > 0 if side == "above" else h <= 0
            return np.where(np.isnan(h), outside, sel)

        return f

    def ijk_range(p):
        ijk, _ = definition.world_to_ijk(p)
        return (ijk[:, 0] >= 2) & (ijk[:, 0] <= 5) & (ijk[:, 2] >= 1)

    cases = {
        "solid": (SolidConstraint(mesh), lambda p: points_in_solid(mesh, p)),
        "solid_out": (SolidConstraint(mesh, "outside"), lambda p: ~points_in_solid(mesh, p)),
        "dtm_above": (DtmConstraint(dtm, "above"), dtm_oracle(dtm, "above", False)),
        "dtm_below": (DtmConstraint(dtm, "below"), dtm_oracle(dtm, "below", False)),
        "dtm_part": (DtmConstraint(part_dtm, "below"), dtm_oracle(part_dtm, "below", False)),
        "dtm_part_true": (DtmConstraint(part_dtm, "above", outside=True),
                          dtm_oracle(part_dtm, "above", True)),  # fmt: skip
        "plane": (PlaneConstraint(centre, normal), lambda p: (p - centre) @ normal > 0),
        "plane_flat": (PlaneConstraint(centre, (0, 0, 1), "below"), lambda p: p[:, 2] <= centre[2]),
        "polygon": (PolygonConstraint(pit, limits=(z_lo, z_hi)),
                    lambda p: crossing_number(p[:, 0], p[:, 1], pit[:, 0], pit[:, 1])
                    & (p[:, 2] >= z_lo) & (p[:, 2] < z_hi)),  # fmt: skip
        "polygon_out": (PolygonConstraint(pit, "outside"),
                        lambda p: ~crossing_number(p[:, 0], p[:, 1], pit[:, 0], pit[:, 1])),
        "polygon_dir": (PolygonConstraint(pit3, axis=direction),
                        lambda p: crossing_number((p - pit3[0]) @ u, (p - pit3[0]) @ v,
                                                  (pit3 - pit3[0]) @ u, (pit3 - pit3[0]) @ v)),
        "range_xyz": (RangeConstraint(x=(centre[0] - 0.2 * extent[0], None), z=(z_lo, z_hi)),
                      lambda p: (p[:, 0] >= centre[0] - 0.2 * extent[0]) & (p[:, 2] >= z_lo)
                      & (p[:, 2] < z_hi)),  # fmt: skip
        "range_ijk": (RangeConstraint(i=(2, 5), k=(1, None)), ijk_range),
    }
    combos = {
        "and": ("solid and dtm_below and range_xyz", lambda r: r["solid"] & r["dtm_below"]
                & r["range_xyz"]),  # fmt: skip
        "or_not": ("plane or not (polygon and range_ijk)", lambda r: r["plane"]
                   | ~(r["polygon"] & r["range_ijk"])),  # fmt: skip
        "mixed": ("(solid or polygon_dir) and not dtm_part_true", lambda r: (r["solid"]
                  | r["polygon_dir"]) & ~r["dtm_part_true"]),  # fmt: skip
    }
    return cases, combos


def _classification_setup(rotated):
    mesh = Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))
    dtm = DTM(Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy")))
    kw = {"angles": (30.0, 12.0, -25.0)} if rotated else {}
    info = json.loads((DATA / "truth.json").read_text())["blocks"]
    definition = BlockModelDefinition(info["origin"], (20.0, 20.0, 10.0), (20, 26, 42),
                                      max_depth=3, **kw)  # fmt: skip
    # a second surface that crosses the model and covers only part of it in plan
    lo, hi = definition.bounding_box()
    mid = 0.5 * (lo + hi)
    gx, gy = np.meshgrid(np.arange(8), np.arange(7), indexing="ij")
    z = mid[2] + 25.0 * np.sin(gx * 0.9) + 18.0 * np.cos(gy * 1.1)
    part_dtm = dtm_from_grid(z, (mid[0] - 150.0, mid[1] - 120.0), (37.0, 41.0))
    return definition, mesh, dtm, part_dtm


_SETUPS = {}


def _setup(rotated):
    if rotated not in _SETUPS:
        definition, mesh, dtm, part_dtm = _classification_setup(rotated)
        _SETUPS[rotated] = (definition, *_oracles(definition, mesh, dtm, part_dtm))
    return _SETUPS[rotated]


def _random_boxes(definition, rng, n, aligned):
    extent = np.array(definition.extent)
    if aligned:  # blocks of a randomly refined model: what sub-blocking will ask about
        m = BlockModel(definition)
        for _ in range(3):
            m.split(rng.random(m.n_blocks) < min(0.5, 400 / m.n_blocks), strict=False)
        rows = rng.choice(m.n_blocks, n, replace=False)
        return m.bounds_local(rows)
    size = extent * rng.uniform(0.002, 0.12, (n, 3))
    lo = rng.uniform(0.0, 1.0, (n, 3)) * (extent - size)
    return lo, lo + size


@settings(max_examples=30, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(seed=st.integers(0, 2**31 - 1), rotated=st.booleans(), aligned=st.booleans())
def test_box_classification_is_conservative(seed, rotated, aligned):
    definition, cases, combos = _setup(rotated)
    rng = np.random.default_rng(seed)
    n_boxes, n_points = 250, 120
    lo, hi = _random_boxes(definition, rng, n_boxes, aligned)
    # random interior points, plus points pushed close to the faces and corners
    t = rng.uniform(0.0, 1.0, (n_boxes, n_points, 3))
    t[:, : n_points // 3] = np.where(t[:, : n_points // 3] < 0.5, 0.002, 0.998)
    t[:, n_points // 3 : n_points // 2, :] = np.clip(
        t[:, n_points // 3 : n_points // 2, :] * 1.2 - 0.1, 0.002, 0.998
    )
    local = lo[:, None, :] + t * (hi - lo)[:, None, :]
    world = definition.local_to_world(local.reshape(-1, 3))
    results, statuses = {}, {}
    for name, (constraint, oracle) in cases.items():
        assert constraint.is_geometric
        status = constraint.classify_boxes(definition, lo, hi)
        assert status.dtype == np.int8 and set(np.unique(status).tolist()) <= {-1, 0, 1}
        selected = np.asarray(oracle(world), dtype=bool).reshape(n_boxes, n_points)
        results[name], statuses[name] = selected, status
        # INSIDE: every sample selected. OUTSIDE: none selected. Never the other way round.
        assert selected[status == INSIDE].all(), name
        assert not selected[status == OUTSIDE].any(), name
    named = {name: c for name, (c, _) in cases.items()}
    flat = {name: r for name, r in results.items()}
    for name, (text, formula) in combos.items():
        constraint = combine(text, named)
        status = constraint.classify_boxes(definition, lo, hi)
        selected = formula(flat)
        assert selected[status == INSIDE].all(), name
        assert not selected[status == OUTSIDE].any(), name
    # complementary constraints give mirrored answers
    np.testing.assert_array_equal(statuses["solid_out"], -statuses["solid"])
    np.testing.assert_array_equal((~named["plane"]).classify_boxes(definition, lo, hi),
                                  -statuses["plane"])  # fmt: skip


def test_box_classification_is_reasonably_tight(truth, resolver, capsys):
    """Fraction of STRADDLE calls on the truth grid against the sampled true fraction."""
    d, _ = truth
    m = BlockModel(d)
    lo, hi = m.bounds_local()
    mesh, dtm = resolver.solid("ore_solid"), resolver.dtm("topo")
    centre = np.array(d.local_to_world(0.5 * np.array(d.extent)))
    pit = centre[:2] + np.array([(-110, -150), (120, -160), (30, 10), (140, 170), (-90, 150.0)])
    constraints = {
        "solid": (SolidConstraint(mesh), lambda p: points_in_solid(mesh, p)),
        "dtm": (DtmConstraint(dtm), lambda p: dtm.height_above(p) > 0),
        "plane": (PlaneConstraint(centre, (0.4, -0.3, 0.85)),
                  lambda p: (p - centre) @ (np.array((0.4, -0.3, 0.85))) > 0),  # fmt: skip
        "polygon": (PolygonConstraint(pit),
                    lambda p: crossing_number(p[:, 0], p[:, 1], pit[:, 0], pit[:, 1])),  # fmt: skip
    }
    # true straddling, from below: a 7 x 7 x 7 lattice reaching 0.1% from the faces
    ticks = np.linspace(0.001, 0.999, 7)
    lattice = np.stack(np.meshgrid(ticks, ticks, ticks, indexing="ij"), axis=-1).reshape(-1, 3)
    lines = []
    for name, (constraint, oracle) in constraints.items():
        t0 = time.perf_counter()
        status = constraint.classify_boxes(d, lo, hi)
        elapsed = time.perf_counter() - t0
        reported = status == STRADDLE
        # sample the reported-straddling boxes and a band of their neighbours only
        candidates = np.flatnonzero(reported)
        local = lo[candidates, None, :] + lattice[None] * (hi - lo)[candidates, None, :]
        selected = oracle(d.local_to_world(local.reshape(-1, 3))).reshape(len(candidates), -1)
        truly = selected.any(axis=1) & ~selected.all(axis=1)
        # boxes not reported as straddling are uniform (the conservative property)
        rest = np.flatnonzero(~reported)[:: max(1, (~reported).sum() // 4000)]
        local = lo[rest, None, :] + lattice[None] * (hi - lo)[rest, None, :]
        uniform = oracle(d.local_to_world(local.reshape(-1, 3))).reshape(len(rest), -1)
        assert (uniform.all(axis=1) | ~uniform.any(axis=1)).all(), name
        assert (uniform.all(axis=1) == (status[rest] == INSIDE)).all(), name
        f_reported, f_true = reported.mean(), truly.sum() / len(status)
        assert 0 < f_true <= f_reported
        assert f_reported <= 1.25 * f_true + 0.002, name  # tight
        lines.append(
            f"{name}: reported {100 * f_reported:.2f}% straddling, sampled true "
            f"{100 * f_true:.2f}% (ratio {f_reported / f_true:.3f}), {elapsed:.2f} s"
        )
    with capsys.disabled():
        print(f"\n[P3.T4 boxes] {d.n_parents:,} parent boxes; " + "; ".join(lines))


def test_box_classification_exact_cases_and_errors():
    d = BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (8, 7, 6), max_depth=2)
    m = BlockModel(d)
    lo, hi = m.bounds_local()
    ijk = m.ijk()
    # a bench range on block faces splits nothing: every block is fully in or out
    bench = RangeConstraint.elevation(835.0, 845.0)
    status = bench.classify_boxes(d, lo, hi)
    np.testing.assert_array_equal(status, np.where((ijk[:, 2] >= 2) & (ijk[:, 2] <= 3), 1, -1))
    np.testing.assert_array_equal(status == INSIDE, bench.mask(m))
    # ... a range through the middle of blocks straddles exactly that layer
    status = RangeConstraint(z=(837.0, None)).classify_boxes(d, lo, hi)
    np.testing.assert_array_equal(status, np.sign(ijk[:, 2] - 2))
    status = RangeConstraint(i=(2, 4), k=(1, 1)).classify_boxes(d, lo, hi)
    inside = (ijk[:, 0] >= 2) & (ijk[:, 0] <= 4) & (ijk[:, 2] == 1)
    np.testing.assert_array_equal(status, np.where(inside, 1, -1))
    # a plane lying on block faces does not straddle; one through block centres does
    status = PlaneConstraint((0, 0, 840.0), (0, 0, 1)).classify_boxes(d, lo, hi)
    np.testing.assert_array_equal(status, np.where(ijk[:, 2] >= 3, 1, -1))
    status = PlaneConstraint((0, 0, 842.5), (0, 0, 1), "below").classify_boxes(d, lo, hi)
    np.testing.assert_array_equal(status, -np.sign(ijk[:, 2] - 3))
    # a polygon on block faces with limits on block faces: touching boxes may straddle,
    # but never the wrong side, and boxes away from the outline are decided
    square = np.array(ORIGIN[:2]) + [(20, 20), (60, 20), (60, 50), (20, 50.0)]
    poly = PolygonConstraint(square, limits=(830.0, 850.0))
    status = poly.classify_boxes(d, lo, hi)
    mask = poly.mask(m)
    assert mask[status == INSIDE].all() and not mask[status == OUTSIDE].any()
    deep = (ijk[:, 0] >= 3) & (ijk[:, 0] <= 4) & (ijk[:, 1] == 3) & (ijk[:, 2] >= 1) & (
        ijk[:, 2] <= 4
    )
    assert (status[deep] == INSIDE).all() and (status[ijk[:, 0] == 0] == OUTSIDE).all()
    # DTM policy for boxes off the surface
    corner = (ORIGIN[0] + 25.0, ORIGIN[1] - 10.0)
    part = dtm_from_grid(np.full((2, 2), 838.0), corner, (30.0, 100.0))
    status = DtmConstraint(part, "above").classify_boxes(d, lo, hi)
    off = (ijk[:, 0] <= 1) | (ijk[:, 0] >= 6)
    assert (status[off] == OUTSIDE).all()
    assert (status[(ijk[:, 0] == 2) | (ijk[:, 0] == 5)] == STRADDLE).all()  # cross the outline
    on = (ijk[:, 0] >= 3) & (ijk[:, 0] <= 4)
    np.testing.assert_array_equal(status[on], np.sign(ijk[on, 2] - 2))
    status = DtmConstraint(part, "above", outside=True).classify_boxes(d, lo, hi)
    assert (status[off] == INSIDE).all()
    with pytest.raises(ConstraintError):
        DtmConstraint(part, "above", outside="error").classify_boxes(d, lo, hi)
    # not geometric / bad boxes
    with pytest.raises(ConstraintError, match="not geometric"):
        BlockIdConstraint([1]).classify_boxes(d, lo, hi)
    with pytest.raises(ConstraintError, match="not geometric"):
        (bench & AttributeConstraint("z > 1")).classify_boxes(d, lo, hi)
    with pytest.raises(ValueError):
        bench.classify_boxes(d, hi, lo)
    with pytest.raises(ValueError):
        bench.classify_boxes(d, lo[:, :2], hi[:, :2])
    assert bench.classify_boxes(d, lo[:0], hi[:0]).shape == (0,)
