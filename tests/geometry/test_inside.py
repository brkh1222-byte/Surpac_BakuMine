"""P2.T5 acceptance: batch point-in-solid.

Oracles: trimesh ``contains`` (random points), exact implicit function of the
synthetic ore solid, analytic half-open boxes, convex half-space tests, and a
solid-angle winding number to adjudicate disagreements with trimesh.

trimesh's ray caster costs up to 1 ms per point on the 20 480-face ore solid,
so the default run compares ``N_TRIMESH`` points per shape. Set
``BAKUMINE_FULL_ORACLE=1`` to compare the full 1 000 000 points per shape
(about 20 minutes). Our own function is always run on 1 000 000 points per
shape and checked against the fast oracles.
"""

import itertools
import os
import time
import warnings
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest
import trimesh
from numba import njit

from bakumine.core.geometry.inside import (
    ColumnCrossings,
    OpenMeshWarning,
    column_crossings,
    grid_in_solid,
    grid_points,
    points_in_solid,
)
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.predicates import _orient2d_exact, orient2d_sign, orient2d_sos
from bakumine.core.geometry.primitives import box, cube, icosphere, torus
from bakumine.synth.config import LensConfig
from bakumine.synth.solid import LensSolid

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
FAR = np.array([5.0e6, 4.0e6, 1.0e3])
N = 1_000_000
FULL = os.environ.get("BAKUMINE_FULL_ORACLE") == "1"
N_TRIMESH = {
    name: N if FULL else n
    for name, n in {
        "cube": 200_000, "icosphere": 100_000, "torus": 60_000, "concave": 200_000, "ore": 4_000,
    }.items()
}  # fmt: skip


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def l_prism() -> Mesh:
    """L-shaped prism: [0,2)x[0,1) plus [0,1)x[1,2) in plan, z from 0 to 1. Built by hand."""
    outline = np.array([[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]], dtype=float)
    n = len(outline)
    vertices = np.vstack(
        [np.column_stack([outline, np.zeros(n)]), np.column_stack([outline, np.ones(n)])]
    )
    cap = np.array([[0, 1, 2], [0, 2, 3], [0, 3, 4], [0, 4, 5]])
    sides = []
    for i in range(n):
        j = (i + 1) % n
        sides += [[i, j, j + n], [i, j + n, i + n]]
    faces = np.vstack([cap[:, ::-1], cap + n, np.array(sides)])
    return Mesh(vertices, faces)


def in_l_prism(p: np.ndarray) -> np.ndarray:
    x, y, z = p[..., 0], p[..., 1], p[..., 2]
    a = (x >= 0) & (x < 2) & (y >= 0) & (y < 1)
    b = (x >= 0) & (x < 1) & (y >= 1) & (y < 2)
    return (a | b) & (z >= 0) & (z < 1)


def ore_solid() -> Mesh:
    return Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))


def shapes() -> dict[str, Mesh]:
    rot = random_rotation(np.random.default_rng(5))
    return {
        "cube": cube(2.0).transformed(rot).translated(FAR),
        "icosphere": icosphere(3, radius=50.0, center=FAR),
        "torus": torus(20.0, 6.0).transformed(rot).translated(FAR),
        "concave": l_prism().transformed(rot).translated(FAR),
        "ore": ore_solid(),
    }


def random_points(mesh: Mesh, n: int, seed: int) -> np.ndarray:
    lo, hi = mesh.bounds
    pad = 0.05 * (hi - lo)
    return np.random.default_rng(seed).uniform(lo - pad, hi + pad, size=(n, 3))


def winding_inside(mesh: Mesh, points: np.ndarray) -> np.ndarray:
    """Independent oracle: solid-angle winding number (van Oosterom & Strackee 1983)."""
    out = np.empty(len(points), dtype=bool)
    tri = mesh.triangles
    for i, p in enumerate(points):
        a, b, c = tri[:, 0] - p, tri[:, 1] - p, tri[:, 2] - p
        la, lb, lc = (np.linalg.norm(q, axis=1) for q in (a, b, c))
        num = np.einsum("ij,ij->i", a, np.cross(b, c))
        den = (
            la * lb * lc
            + np.einsum("ij,ij->i", a, b) * lc
            + np.einsum("ij,ij->i", b, c) * la
            + np.einsum("ij,ij->i", c, a) * lb
        )
        out[i] = abs(2.0 * np.arctan2(num, den).sum() / (4.0 * np.pi)) > 0.5
    return out


def trimesh_contains(mesh: Mesh, points: np.ndarray, chunk: int = 2000) -> np.ndarray:
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    return np.concatenate(
        [tm.contains(points[i : i + chunk]) for i in range(0, len(points), chunk)]
    )


def convex_inside(mesh: Mesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Half-space test for a convex mesh: (inside, distance to the nearest face plane)."""
    worst = np.full(len(points), -np.inf)
    for normal, corner in zip(mesh.face_normals, mesh.triangles[:, 0], strict=True):
        np.maximum(worst, (points - corner) @ normal, out=worst)
    return worst < 0.0, np.abs(worst)


# ----------------------------------------------------------------- predicates


@njit
def _signs(rows):
    fast = np.empty(len(rows), dtype=np.int64)
    exact = np.empty(len(rows), dtype=np.int64)
    sos = np.empty(len(rows), dtype=np.int64)
    for i in range(len(rows)):
        r = rows[i]
        fast[i] = orient2d_sign(r[0], r[1], r[2], r[3], r[4], r[5])
        exact[i] = _orient2d_exact(r[0], r[1], r[2], r[3], r[4], r[5])
        sos[i] = orient2d_sos(r[0], r[1], r[2], r[3], r[4], r[5])
    return fast, exact, sos


def test_orient2d_is_exact_against_rational_arithmetic():
    rng = np.random.default_rng(0)
    n = 6000
    a = rng.uniform(-1, 1, (n, 2)) * 5e6
    b = a + rng.uniform(-100, 100, (n, 2))
    c = a + rng.uniform(0, 1, (n, 1)) * (b - a)  # collinear up to rounding
    c[::3] = np.nextafter(c[::3], np.inf)
    # exactly collinear, far from the origin, not representable as a tidy product
    k = np.arange(n // 2)
    a[: n // 2] = 5e6 + 0.125 * rng.integers(-1000, 1000, (n // 2, 2))
    d = 0.125 * rng.integers(-1000, 1000, (n // 2, 2))
    b[: n // 2] = a[: n // 2] + 3 * d
    c[: n // 2] = a[: n // 2] + (k % 7)[:, None] * d
    c[1 : n // 2 : 2] = np.nextafter(c[1 : n // 2 : 2], -np.inf)
    rows = np.column_stack([a, b, c])
    fast, exact, sos = _signs(rows)

    def rational(r):
        ax, ay, bx, by, cx, cy = map(Fraction, r)
        det = (ax - cx) * (by - cy) - (ay - cy) * (bx - cx)
        return (det > 0) - (det < 0)

    truth = np.array([rational(r) for r in rows.tolist()])
    assert (truth == 0).sum() > 500 and (truth > 0).sum() > 500 and (truth < 0).sum() > 500
    np.testing.assert_array_equal(fast, truth)
    np.testing.assert_array_equal(exact, truth)
    # the tie-break never returns 0 for a real edge and agrees where the sign is decided
    assert (sos[(rows[:, 0] != rows[:, 2]) | (rows[:, 1] != rows[:, 3])] != 0).all()
    np.testing.assert_array_equal(sos[truth != 0], truth[truth != 0])


# ------------------------------------------------------- random points, 5 shapes


@pytest.mark.parametrize("name", ["cube", "icosphere", "torus", "concave", "ore"])
def test_matches_trimesh_on_random_points(name):
    mesh = shapes()[name]
    assert mesh.validate(check_self_intersection=False).is_closed
    points = random_points(mesh, N, seed=11)
    ours = points_in_solid(mesh, points)
    assert 0.02 < ours.mean() < 0.9

    n = N_TRIMESH[name]
    theirs = trimesh_contains(mesh, points[:n])
    differ = np.flatnonzero(ours[:n] != theirs)
    print(f"{name}: {len(differ)} of {n} points differ from trimesh.contains")
    # trimesh is not exact; every disagreement is settled by a third method.
    assert len(differ) <= max(5, n // 20_000), f"{len(differ)} disagreements with trimesh"
    if len(differ):
        np.testing.assert_array_equal(ours[differ], winding_inside(mesh, points[differ]))

    # full 1M set against the winding number on a sample and against the volume
    sample = np.random.default_rng(3).choice(N, 300, replace=False)
    np.testing.assert_array_equal(ours[sample], winding_inside(mesh, points[sample]))
    lo, hi = mesh.bounds
    box_volume = np.prod(1.1 * (hi - lo))
    assert ours.mean() * box_volume == pytest.approx(mesh.volume, rel=0.01)


def test_convex_shapes_match_half_space_oracle_on_1m_points():
    for name in ("cube", "icosphere"):
        mesh = shapes()[name]
        points = random_points(mesh, N, seed=12)
        expected, distance = convex_inside(mesh, points)
        clear = distance > 1e-6
        np.testing.assert_array_equal(points_in_solid(mesh, points)[clear], expected[clear])
        assert clear.mean() > 0.999


def test_concave_prism_matches_analytic_on_1m_points():
    mesh = l_prism()
    points = np.random.default_rng(13).uniform(-0.2, 2.2, size=(N, 3))
    np.testing.assert_array_equal(points_in_solid(mesh, points), in_l_prism(points))


def test_synthetic_solid_matches_exact_implicit_function():
    mesh = ore_solid()
    solid = LensSolid.from_config(LensConfig())
    points = random_points(mesh, N, seed=14)
    ours = points_in_solid(mesh, points)
    exact = solid.inside(points)
    differ = np.flatnonzero(ours != exact)
    # The mesh is a 20 480-face polyhedron with its vertices on the implicit
    # surface, so the two solids differ in a thin shell between the chords
    # and the true surface. Measured: 82 of 1 000 000 points differ, all
    # within 0.065 m of the mesh surface (lens half-thickness 16 m).
    assert len(differ) / N < 2.0e-4
    tm = trimesh.Trimesh(mesh.vertices - mesh.local_origin, mesh.faces, process=False)
    _, distance, _ = trimesh.proximity.closest_point(tm, points[differ] - mesh.local_origin)
    print(f"ore vs implicit: {len(differ)} of {N} differ, max distance {distance.max():.4f} m")
    assert distance.max() < 0.1
    # the shell is the volume difference: mesh volume is 0.057 % below the exact volume
    lo, hi = mesh.bounds
    box_volume = np.prod(1.1 * (hi - lo))
    assert ours.mean() * box_volume == pytest.approx(mesh.volume, rel=0.01)
    assert exact.mean() * box_volume == pytest.approx(solid.volume(), rel=0.01)


# ---------------------------------------------------------- adversarial exactness


def half_open(points, lo, hi):
    return ((points >= lo) & (points < hi)).all(axis=-1)


@pytest.mark.parametrize("offset", [np.zeros(3), FAR, np.array([500600.1, 5300500.3, 1000.7])])
def test_axis_aligned_box_is_exactly_half_open(offset):
    lo, hi = offset + [0.0, 0.0, 0.0], offset + [4.0, 2.0, 1.0]
    mesh = box(lo, hi)
    mid = 0.5 * (lo + hi)
    eps = np.spacing(hi)
    levels = [
        sorted({lo[k] - 1.0, lo[k] - eps[k], lo[k], lo[k] + eps[k], mid[k], hi[k] - eps[k],
                hi[k], hi[k] + eps[k], hi[k] + 1.0})
        for k in range(3)
    ]  # fmt: skip
    points = np.array(list(itertools.product(*levels)))
    np.testing.assert_array_equal(points_in_solid(mesh, points), half_open(points, lo, hi))


def test_rays_through_cube_vertices_and_edges():
    mesh = cube(2.0, center=FAR)
    lo, hi = mesh.bounds
    edges = mesh.edges
    v = mesh.vertices
    xy = np.vstack([v[:, :2], 0.5 * (v[edges[:, 0], :2] + v[edges[:, 1], :2])])
    for z in (lo[2] - 1.0, lo[2], FAR[2], hi[2], hi[2] + 1.0, lo[2] - 1e6):
        points = np.column_stack([xy, np.full(len(xy), z)])
        np.testing.assert_array_equal(points_in_solid(mesh, points), half_open(points, lo, hi))


@pytest.mark.parametrize("center", [(0.0, 0.0, 0.0), tuple(FAR)])
@pytest.mark.parametrize("subdivisions", [0, 1, 3])
def test_rays_through_icosphere_vertices_and_edges(center, subdivisions):
    radius = 10.0
    mesh = icosphere(subdivisions, radius=radius, center=center)
    edges = mesh.edges
    v = mesh.vertices
    xy = np.vstack([v[:, :2], 0.5 * (v[edges[:, 0], :2] + v[edges[:, 1], :2])])
    cz = center[2]
    for z in (cz - 3 * radius, cz - 1.5 * radius, cz + 1.5 * radius, cz + 3 * radius):
        points = np.column_stack([xy, np.full(len(xy), z)])
        assert not points_in_solid(mesh, points).any()
    # Inside: the inscribed sphere of the polyhedron is certainly inside.
    inradius = np.min(np.einsum("ij,ij->i", mesh.face_normals, mesh.triangles[:, 0] - center))
    for z in np.array([-0.5, 0.0, 0.3]) * inradius + cz:
        points = np.column_stack([xy, np.full(len(xy), z)])
        r = np.linalg.norm(points - center, axis=1)
        result = points_in_solid(mesh, points)
        assert result[r < 0.999 * inradius].all()
        assert not result[r > radius].any()
        expected, distance = convex_inside(mesh, points)
        np.testing.assert_array_equal(result[distance > 1e-9], expected[distance > 1e-9])
    assert (np.linalg.norm(xy - center[:2], axis=1) < 0.99 * inradius).sum() > 3


def test_rays_through_vertices_and_edges_cross_once_per_sheet():
    # Every vertical line through a vertex or an edge midpoint must cross a
    # closed surface an even number of times; a convex one 0 or 2 times.
    for mesh in (icosphere(2, radius=3.0, center=FAR), cube(2.0, center=FAR), l_prism(), torus()):
        e = mesh.edges
        v = mesh.vertices
        xy = np.vstack([v[:, :2], 0.5 * (v[e[:, 0], :2] + v[e[:, 1], :2])])
        counts = column_crossings(mesh, xy).counts
        assert (counts % 2 == 0).all()
        assert counts.max() <= (4 if mesh.n_faces == 2304 else 2)
        assert (counts > 0).sum() > len(xy) // 4


def test_cavity_and_nested_shells():
    outer = box([0, 0, 0], [10, 10, 10])
    inner = box([3, 3, 3], [7, 7, 7])
    core = box([4, 4, 4], [6, 6, 6])
    points = np.random.default_rng(2).uniform(-1, 11, size=(200_000, 3))
    grid = np.array(list(itertools.product(np.arange(-1.0, 12.0, 0.5), repeat=3)))
    points = np.vstack([points, grid])
    in_outer = half_open(points, [0, 0, 0], [10, 10, 10])
    in_inner = half_open(points, [3, 3, 3], [7, 7, 7])
    in_core = half_open(points, [4, 4, 4], [6, 6, 6])
    for cavity in (inner.flipped(), inner):  # inward winding, and winding ignored
        mesh = Mesh.concatenate([outer, cavity])
        np.testing.assert_array_equal(points_in_solid(mesh, points), in_outer & ~in_inner)
    mesh = Mesh.concatenate([outer, inner.flipped(), core])
    np.testing.assert_array_equal(
        points_in_solid(mesh, points), (in_outer & ~in_inner) | in_core
    )
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    sample = points[:20_000]
    assert (tm.contains(sample) != points_in_solid(mesh, sample)).sum() == 0


# ------------------------------------------------------------------ grid path


def _grid_cases():
    rng = np.random.default_rng(21)
    ore = ore_solid()
    lo, hi = ore.bounds
    half = (0.5, 0.5, 0.5)
    ore_shape = tuple(np.ceil((hi - lo + 6.0) / [4.0, 5.0, 2.5]).astype(int))
    return {
        # grid planes coincide with the faces of the box
        "box_on_planes": (box(FAR, FAR + [8.0, 6.0, 4.0]), FAR - 1.0, half, (21, 17, 13)),
        "box_centroids": (box([0, 0, 0], [8, 6, 4]), (-0.75, -0.75, -0.75), half, (20, 16, 12)),
        "icosphere": (
            icosphere(3, radius=10.0, center=FAR), FAR - 10.5, (0.37, 0.41, 0.29), (58, 52, 73)
        ),
        "icosphere_through_vertices": (icosphere(2, radius=8.0), (-8.0,) * 3, half, (33, 33, 33)),
        "torus": (torus(), (-2.6, -2.6, -0.6), (0.05, 0.07, 0.03), (105, 75, 41)),
        "ore": (ore, lo - 3.0, (4.0, 5.0, 2.5), ore_shape),
        "ore_random_origin": (ore, lo + rng.uniform(-5, 0, 3), (7.3, 3.1, 1.7), (45, 150, 160)),
    }  # fmt: skip


@pytest.mark.parametrize("case", list(_grid_cases()))
def test_grid_equals_point_path_exactly(case):
    mesh, origin, spacing, shape = _grid_cases()[case]
    mask, crossings = grid_in_solid(mesh, origin, spacing, shape, return_intervals=True)
    assert mask.shape == shape and mask.dtype == bool
    assert mask.any() and not mask.all()
    points = grid_points(origin, spacing, shape)
    expected = np.asarray(origin) + np.multiply(spacing, [1, 2, 3])
    np.testing.assert_array_equal(points[1, 2, 3], expected)
    np.testing.assert_array_equal(mask, points_in_solid(mesh, points))
    np.testing.assert_array_equal(mask, grid_in_solid(mesh, origin, spacing, shape))

    # intervals describe the same thing
    assert isinstance(crossings, ColumnCrossings) and crossings.shape == shape[:2]
    assert crossings.frame == "world" and crossings.z0 == points[0, 0, 0, 2]
    assert (crossings.counts % 2 == 0).all()
    column, entry, exit_ = crossings.intervals()
    assert (exit_ >= entry).all()
    z = points[0, 0, :, 2]
    rebuilt = np.zeros(shape, dtype=bool).reshape(-1, shape[2])
    for c, a, b in zip(column, entry, exit_, strict=True):
        rebuilt[c] |= (z >= a) & (z < b)
    np.testing.assert_array_equal(rebuilt.reshape(shape), mask)
    i, j = np.unravel_index(column[len(column) // 2], shape[:2])
    pairs = crossings.column(i, j)
    assert pairs.shape[1] == 2 and (pairs[:, 0] == entry[column == column[len(column) // 2]]).all()


def test_box_grid_is_half_open_and_intervals_are_exact():
    mesh = box(FAR, FAR + [8.0, 6.0, 4.0])
    origin, spacing, shape = FAR - 1.0, (0.5, 0.5, 0.5), (21, 17, 13)
    mask, crossings = grid_in_solid(mesh, origin, spacing, shape, return_intervals=True)
    points = grid_points(origin, spacing, shape)
    np.testing.assert_array_equal(mask, half_open(points, FAR, FAR + [8.0, 6.0, 4.0]))
    assert mask.sum() == 16 * 12 * 8
    column, entry, exit_ = crossings.intervals()
    assert len(column) == 16 * 12
    assert (entry == FAR[2]).all() and (exit_ == FAR[2] + 4.0).all()
    assert crossings.thickness().sum() * 0.25 == 16 * 12 * 4.0 * 0.25


@pytest.mark.parametrize("seed", [0, 1])
def test_rotated_grid(seed):
    rng = np.random.default_rng(seed)
    rotation = random_rotation(rng)
    if seed == 1:  # a block-model style rotation about the vertical axis
        a = np.radians(30.0)
        rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    mesh = ore_solid()
    spacing, shape = (6.0, 5.0, 3.0), (110, 140, 130)
    centre = mesh.local_origin
    origin = centre - rotation @ (0.5 * np.multiply(spacing, shape))
    mask, crossings = grid_in_solid(mesh, origin, spacing, shape, rotation, return_intervals=True)
    assert crossings.frame == "grid" and crossings.z0 == 0.0

    # exact agreement with the point path on the mesh rotated into grid space
    local_mesh = Mesh((mesh.vertices - origin) @ rotation, mesh.faces)
    local_points = grid_points(np.zeros(3), spacing, shape)
    np.testing.assert_array_equal(mask, points_in_solid(local_mesh, local_points))
    # and agreement with rotating the points instead of the mesh
    world_points = grid_points(origin, spacing, shape, rotation)
    expected = origin + rotation @ np.multiply(spacing, [3, 5, 7])
    np.testing.assert_allclose(world_points[3, 5, 7], expected, rtol=0, atol=1e-8)
    by_points = points_in_solid(mesh, world_points)
    np.testing.assert_array_equal(mask, by_points)
    # the grid covers the solid: cell count times cell volume is the volume
    assert mask.sum() * np.prod(spacing) == pytest.approx(mesh.volume, rel=5e-3)
    assert crossings.thickness().sum() * spacing[0] * spacing[1] == pytest.approx(
        mesh.volume, rel=5e-3
    )


def test_rotated_grid_on_rotated_box_is_half_open():
    a = np.radians(37.0)
    rotation = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1.0]])
    tilt = random_rotation(np.random.default_rng(8))
    for rot in (rotation, tilt):
        local = box([0, 0, 0], [8, 6, 4])
        world = Mesh(FAR + local.vertices @ rot.T, local.faces)
        origin = FAR + rot @ [-3.25, -3.25, -3.25]
        mask = grid_in_solid(world, origin, (0.5, 0.5, 0.5), (30, 30, 30), rot)
        local_points = grid_points((-3.25, -3.25, -3.25), (0.5, 0.5, 0.5), (30, 30, 30))
        np.testing.assert_array_equal(mask, half_open(local_points, [0, 0, 0], [8, 6, 4]))


def test_grid_arguments_are_validated():
    mesh = cube()
    with pytest.raises(ValueError, match="spacing"):
        grid_in_solid(mesh, (0, 0, 0), (1, 0, 1), (2, 2, 2))
    with pytest.raises(ValueError, match="orthonormal"):
        grid_in_solid(mesh, (0, 0, 0), (1, 1, 1), (2, 2, 2), np.diag([1.0, 2.0, 1.0]))
    with pytest.raises(ValueError, match="shape"):
        grid_in_solid(mesh, (0, 0, 0), (1, 1, 1), (2, 2))
    assert grid_in_solid(mesh, (0, 0, 0), (1, 1, 1), (0, 3, 2)).shape == (0, 3, 2)
    identity = grid_in_solid(mesh, (-1, -1, -1), (0.25, 0.25, 0.25), (9, 9, 9), np.eye(3))
    np.testing.assert_array_equal(
        identity, grid_in_solid(mesh, (-1, -1, -1), (0.25, 0.25, 0.25), (9, 9, 9))
    )


# ------------------------------------------------------- surface, open, shapes


def test_input_shapes_and_nan():
    mesh = cube(2.0)
    points = np.zeros((4, 5, 3))
    points[0, 0] = [5, 0, 0]
    points[1, 1] = [np.nan, 0, 0]
    points[2, 2] = [0, 0, np.nan]
    result = points_in_solid(mesh, points)
    assert result.shape == (4, 5) and result.dtype == bool
    assert result.sum() == 17
    assert points_in_solid(mesh, np.zeros((0, 3))).shape == (0,)
    assert points_in_solid(mesh, [0.0, 0.0, 0.0]).shape == () and points_in_solid(mesh, [0, 0, 0])
    with pytest.raises(ValueError, match="shape"):
        points_in_solid(mesh, np.zeros((3, 2)))


def test_open_and_empty_meshes_raise_or_warn():
    closed = icosphere(1)
    opened = closed.submesh(np.arange(closed.n_faces) != 3)
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    soup = Mesh(closed.triangles.reshape(-1, 3), np.arange(closed.n_faces * 3).reshape(-1, 3))
    points = np.random.default_rng(0).uniform(-1, 1, (100, 3))
    for bad in (opened, empty, soup):
        with pytest.raises(ValueError, match="closed mesh"):
            points_in_solid(bad, points)
        with pytest.raises(ValueError, match="closed mesh"):
            grid_in_solid(bad, (0, 0, 0), (1, 1, 1), (2, 2, 2))
        with pytest.raises(ValueError, match="closed mesh"):
            column_crossings(bad, points)
        with pytest.warns(OpenMeshWarning):
            result = points_in_solid(bad, points, strict=False)
        assert result.shape == (100,)
    # welding the soup makes it usable; a closed mesh never warns
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_array_equal(
            points_in_solid(soup.weld(), points), points_in_solid(closed, points)
        )
    # an open mesh has columns with an odd number of crossings: no intervals
    with pytest.warns(OpenMeshWarning):
        crossings = column_crossings(opened, opened.triangles.mean(axis=1), strict=False)
    assert (crossings.counts % 2 == 1).any()
    with pytest.raises(ValueError, match="odd"):
        crossings.intervals()


def test_two_solids_sharing_an_edge_are_accepted():
    # four faces around one edge: not manifold, but parity is well defined
    a, b = box([0, 0, 0], [1, 1, 1]), box([1, 1, 0], [2, 2, 1])
    mesh = Mesh.concatenate([a, b]).weld()
    assert (mesh.edge_map.counts == 4).sum() == 1
    points = np.random.default_rng(4).uniform(-0.5, 2.5, (100_000, 3))
    expected = half_open(points, [0, 0, 0], [1, 1, 1]) | half_open(points, [1, 1, 0], [2, 2, 1])
    np.testing.assert_array_equal(points_in_solid(mesh, points), expected)


# ----------------------------------------------------------------- performance


def test_grid_performance_on_synthetic_solid():
    mesh = ore_solid()
    lo, hi = mesh.bounds
    shape = (100, 200, 50)  # 1M points; the full-size run is 10M (see the task log)
    spacing = (hi - lo) / shape
    origin = lo + 0.5 * spacing
    grid_in_solid(mesh, origin, spacing, (2, 2, 2))  # compile
    points_in_solid(mesh, np.zeros((2, 3)))
    start = time.perf_counter()
    mask = grid_in_solid(mesh, origin, spacing, shape)
    grid_time = time.perf_counter() - start
    points = grid_points(origin, spacing, shape).reshape(-1, 3)
    start = time.perf_counter()
    by_points = points_in_solid(mesh, points)
    point_time = time.perf_counter() - start
    np.testing.assert_array_equal(mask.reshape(-1), by_points)
    assert mask.sum() * np.prod(spacing) == pytest.approx(mesh.volume, rel=2e-3)
    assert grid_time < 2.0  # 10M must take < 20 s
    assert point_time < 5.0


@pytest.mark.skipif(not FULL, reason="set BAKUMINE_FULL_ORACLE=1 for the full-size run")
def test_grid_performance_full_size_10m():
    mesh = ore_solid()
    lo, hi = mesh.bounds
    shape = (250, 400, 100)
    spacing = (hi - lo) / shape
    grid_in_solid(mesh, lo, spacing, (2, 2, 2))
    start = time.perf_counter()
    mask = grid_in_solid(mesh, lo + 0.5 * spacing, spacing, shape)
    elapsed = time.perf_counter() - start
    print(f"10M grid centroids: {elapsed:.3f} s")
    assert mask.size == 10_000_000 and elapsed < 20.0
