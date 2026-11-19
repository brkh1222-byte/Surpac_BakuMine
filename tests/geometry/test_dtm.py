"""P2.T4 acceptance: DTM from points and strings, batch z_at.

Oracles: analytic plane and paraboloid, scipy's LinearNDInterpolator, the
generating function of the synthetic topography.
"""

import time
import warnings
from pathlib import Path

import numpy as np
import pytest
from scipy.interpolate import LinearNDInterpolator

from bakumine.core.geometry.dtm import DTM, DuplicatePointWarning, dtm_from_grid, dtm_from_points
from bakumine.core.geometry.mesh import Mesh
from bakumine.synth.config import TopoConfig
from bakumine.synth.topo import Topography

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
FAR = np.array([5.0e6, 4.0e6, 1.0e3])


def plane(x, y):
    return 0.3 * (x - FAR[0]) - 0.2 * (y - FAR[1]) + FAR[2] + 5.0


def scatter(n, seed, size=100.0):
    xy = FAR[:2] + np.random.default_rng(seed).uniform(0.0, size, (n, 2))
    return xy


def vertex_index(dtm: DTM, xy) -> int:
    hit = np.flatnonzero((dtm.mesh.vertices[:, :2] == np.asarray(xy)).all(axis=1))
    assert len(hit) == 1, f"no single vertex at {xy}"
    return int(hit[0])


def has_edge(dtm: DTM, a, b) -> bool:
    return dtm.mesh.edge_map.find(vertex_index(dtm, a), vertex_index(dtm, b)) >= 0


def topo() -> tuple[DTM, Topography]:
    mesh = Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy"))
    return DTM(mesh), Topography(TopoConfig())


# ------------------------------------------------------------ plane, paraboloid


def test_plane_is_reproduced_and_outside_is_nan():
    xy = scatter(3000, 1)
    corners = FAR[:2] + np.array([[0, 0], [100, 0], [100, 100], [0, 100.0]])
    xy = np.vstack([xy, corners])
    dtm = dtm_from_points(np.column_stack([xy, plane(xy[:, 0], xy[:, 1])]))
    assert dtm.mesh.n_vertices == len(xy)
    assert dtm.plan_area == pytest.approx(100.0 * 100.0, rel=1e-12)
    assert (dtm.mesh.face_normals[:, 2] > 0).all()

    q = FAR[:2] + np.random.default_rng(2).uniform(-20.0, 120.0, (200_000, 2))
    z = dtm.z_at(q[:, 0], q[:, 1])
    inside = ((q >= FAR[:2]) & (q <= FAR[:2] + 100.0)).all(axis=1)
    assert np.isnan(z[~inside]).all()
    assert not np.isnan(z[inside]).any()
    np.testing.assert_allclose(z[inside], plane(q[inside, 0], q[inside, 1]), rtol=0, atol=1e-9)
    # one (n, 2) argument, scalars and broadcasting behave the same
    np.testing.assert_array_equal(dtm.z_at(q), z)
    assert dtm.z_at(q.reshape(400, 500, 2)).shape == (400, 500)
    assert dtm.z_at(FAR[0] + 50.0, FAR[1] + 50.0) == pytest.approx(plane(FAR[0] + 50, FAR[1] + 50))
    assert np.isnan(dtm.z_at(np.nan, 0.0))
    assert (dtm.triangle_at(q)[inside] >= 0).all() and (dtm.triangle_at(q)[~inside] == -1).all()


def test_matches_scipy_linear_interpolation():
    rng = np.random.default_rng(3)
    xy = scatter(2000, 3)
    z = rng.normal(size=len(xy)) * 10.0
    dtm = dtm_from_points(np.column_stack([xy, z]))
    reference = LinearNDInterpolator(xy - FAR[:2], z)
    q = FAR[:2] + rng.uniform(-5.0, 105.0, (100_000, 2))
    ours = dtm.z_at(q)
    theirs = reference(q - FAR[:2])
    both = ~np.isnan(ours) & ~np.isnan(theirs)
    assert both.mean() > 0.8
    np.testing.assert_allclose(ours[both], theirs[both], rtol=0, atol=1e-7)
    # the two disagree about inside/outside only within rounding of the hull
    assert (np.isnan(ours) != np.isnan(theirs)).sum() <= 2


def test_paraboloid_error_decreases_with_density():
    def f(x, y):
        return ((x - FAR[0] - 50.0) ** 2 + (y - FAR[1] - 50.0) ** 2) / 100.0

    q = FAR[:2] + np.random.default_rng(4).uniform(10.0, 90.0, (50_000, 2))
    errors = []
    for n in (500, 2000, 8000, 32000):
        xy = scatter(n, 5)
        dtm = dtm_from_points(np.column_stack([xy, f(xy[:, 0], xy[:, 1])]))
        err = dtm.z_at(q) - f(q[:, 0], q[:, 1])
        assert not np.isnan(err).any()
        # a secant surface of a convex function lies above it
        assert err.min() > -1e-9
        # |error| <= (longest edge)^2 / 8 * second derivative (= 2 / 100)
        edges = dtm.mesh.vertices[dtm.mesh.edges]
        longest = np.linalg.norm(edges[:, 0, :2] - edges[:, 1, :2], axis=1)
        interior = longest[(np.abs(edges[:, :, :2] - FAR[:2] - 50.0) < 42.0).all(axis=(1, 2))]
        assert err.max() <= interior.max() ** 2 / 4.0 * 0.02
        errors.append(np.sqrt(np.mean(err**2)))
    ratios = np.array(errors[1:]) / np.array(errors[:-1])
    assert (ratios < 0.35).all() and (ratios > 0.18).all()  # error ~ 1 / density
    assert errors[-1] < 5e-3


# -------------------------------------------------------- vertices and edges


@pytest.mark.parametrize("far", [False, True])
def test_queries_exactly_on_vertices_and_edges(far):
    rng = np.random.default_rng(6)
    xy = scatter(1500, 6) - (0.0 if far else FAR[:2])
    z = rng.uniform(0.0, 50.0, len(xy))
    dtm = dtm_from_points(np.column_stack([xy, z]))
    v = dtm.mesh.vertices
    # every vertex, hull vertices included: exactly its own Z
    np.testing.assert_array_equal(dtm.z_at(v[:, 0], v[:, 1]), v[:, 2])
    np.testing.assert_array_equal(dtm.above_below(v), np.zeros(len(v), dtype=np.int8))
    # Points on interior edges are never missed and get the Z of the edge.
    # A computed point of an edge is off the edge by the rounding of its
    # coordinates (1e-9 m at 5e6), times the slope of this rough surface.
    atol = 1e-5 if far else 1e-9
    em = dtm.mesh.edge_map
    interior = dtm.mesh.edges[em.counts == 2]
    for t in (0.5, 0.125, 0.75):
        p = v[interior[:, 0]] + t * (v[interior[:, 1]] - v[interior[:, 0]])
        np.testing.assert_allclose(dtm.z_at(p), p[:, 2], rtol=0, atol=atol)


def test_grid_dtm_border_and_shared_edges_are_exact():
    rng = np.random.default_rng(7)
    z = rng.uniform(0.0, 10.0, (9, 7))
    dtm = dtm_from_grid(z, origin=(1000.0, 2000.0), spacing=(2.0, 4.0))
    assert dtm.mesh.n_vertices == 63 and dtm.mesh.n_faces == 2 * 8 * 6
    assert dtm.plan_area == 16.0 * 24.0
    i, j = np.meshgrid(np.arange(9), np.arange(7), indexing="ij")
    np.testing.assert_array_equal(dtm.z_at(1000.0 + 2.0 * i, 2000.0 + 4.0 * j), z)
    # midpoints of grid lines (exactly on edges, including the outer border)
    np.testing.assert_allclose(
        dtm.z_at(1001.0 + 2.0 * i[:-1], 2000.0 + 4.0 * j[:-1]), 0.5 * (z[:-1] + z[1:]), atol=1e-12
    )
    np.testing.assert_allclose(
        dtm.z_at(1000.0 + 2.0 * i[:, :-1], 2002.0 + 4.0 * j[:, :-1]),
        0.5 * (z[:, :-1] + z[:, 1:]),
        atol=1e-12,
    )
    # cell centres lie on the diagonal from (i, j) to (i + 1, j + 1)
    np.testing.assert_allclose(
        dtm.z_at(1001.0 + 2.0 * i[:-1, :-1], 2002.0 + 4.0 * j[:-1, :-1]),
        0.5 * (z[:-1, :-1] + z[1:, 1:]),
        atol=1e-12,
    )
    # just outside the border
    outside = dtm.z_at([999.999999, 1016.000001, 1008.0], [2010.0, 2010.0, 2024.000001])
    assert np.isnan(outside).all()
    assert not np.isnan(dtm.z_at([1000.0, 1016.0, 1008.0], [2010.0, 2010.0, 2024.0])).any()


def test_grid_dtm_with_missing_nodes():
    z = np.arange(20.0).reshape(4, 5)
    z[1, 1] = np.nan
    dtm = dtm_from_grid(z, (0.0, 0.0), (1.0, 1.0))
    # the six triangles around the node go; node (0, 0) loses its only two
    assert dtm.mesh.n_vertices == 18 and dtm.mesh.n_faces == 24 - 6
    assert np.isnan(dtm.z_at(0.5, 0.5)) and np.isnan(dtm.z_at(1.5, 1.5))
    assert dtm.z_at(2.5, 2.5) == pytest.approx(0.5 * (z[2, 2] + z[3, 3]))
    with pytest.raises(ValueError):
        dtm_from_grid(np.full((3, 3), np.nan), (0, 0), (1, 1))
    with pytest.raises(ValueError):
        dtm_from_grid(np.zeros((3, 3)), (0, 0), (1, 0))


# ----------------------------------------------------------------- breaklines


def test_every_breakline_segment_is_an_edge():
    rng = np.random.default_rng(8)
    xy = scatter(1200, 8)
    points = np.column_stack([xy, plane(xy[:, 0], xy[:, 1])])
    t = np.linspace(0.0, 1.0, 9)
    crest = np.column_stack([FAR[0] + 5 + 90 * t, FAR[1] + 40 + 20 * np.sin(3 * t), 60 + 0 * t])
    toe = np.column_stack([FAR[0] + 10 + 80 * t, FAR[1] + 12 + 5 * t, 20 + 0 * t])
    ring = FAR + np.array([[70, 70, 30], [90, 72, 30], [88, 92, 30], [68, 90, 30], [70, 70, 30.0]])
    long_one = FAR + np.array([[2.0, 97.0, 40.0], [98.0, 95.0, 40.0]])  # crosses many triangles
    lines = [crest, toe, ring, long_one]

    plain = dtm_from_points(np.vstack([points, *lines]))
    missing = sum(
        not has_edge(plain, a[:2], b[:2])
        for line in lines
        for a, b in zip(line[:-1], line[1:], strict=True)
    )
    assert missing > 0  # without constraints the triangulation cuts across them

    dtm = dtm_from_points(points, breaklines=lines)
    for line in lines:
        for a, b in zip(line[:-1], line[1:], strict=True):
            assert has_edge(dtm, a[:2], b[:2])
        # The surface follows the string along its whole length (the strings
        # are ~1000 m off the points: a computed point 1e-9 m off the edge
        # sits on a very steep triangle, hence the tolerance).
        for s in rng.uniform(0, 1, 5):
            p = line[:-1] + s * (line[1:] - line[:-1])
            np.testing.assert_allclose(dtm.z_at(p), p[:, 2], rtol=0, atol=1e-4)
    assert dtm.mesh.n_vertices == len(points) + sum(len(line) for line in lines) - 1
    assert dtm.plan_area == pytest.approx(plain.plan_area, rel=1e-12)
    report = dtm.mesh.validate()
    assert report.is_manifold and not report.has_degenerate_faces
    assert not report.has_self_intersections


def test_crossing_breaklines_get_a_vertex_at_the_crossing():
    xy = scatter(300, 9)
    points = np.column_stack([xy, np.zeros(len(xy))])
    a = FAR + np.array([[10.0, 10.0, 10.0], [90.0, 90.0, 10.0]])
    b = FAR + np.array([[10.0, 90.0, 10.0], [90.0, 10.0, 10.0]])
    dtm = dtm_from_points(points, breaklines=[a, b])
    assert dtm.mesh.n_vertices == len(points) + 5
    crossing = vertex_index(dtm, FAR[:2] + 50.0)
    assert dtm.mesh.vertices[crossing, 2] == pytest.approx(FAR[2] + 10.0)
    for end in (a[0], a[1], b[0], b[1]):
        assert has_edge(dtm, end[:2], FAR[:2] + 50.0)


def test_breaklines_only():
    square = np.array([[0, 0, 1], [10, 0, 1], [10, 10, 3], [0, 10, 3], [0, 0, 1.0]])
    dtm = dtm_from_points(np.zeros((0, 3)), breaklines=[square, [[0, 0, 1], [10, 10, 3]]])
    assert dtm.mesh.n_vertices == 4 and dtm.mesh.n_faces == 2
    assert has_edge(dtm, (0, 0), (10, 10))
    assert dtm.z_at(5.0, 5.0) == pytest.approx(2.0)


# ------------------------------------------------------------------- boundary


def _polygon_area(p):
    return 0.5 * abs(np.sum(p[:, 0] * np.roll(p[:, 1], -1) - np.roll(p[:, 0], -1) * p[:, 1]))


@pytest.mark.parametrize("with_z", [True, False])
def test_boundary_polygon_is_respected(with_z):
    xy = scatter(2500, 10)
    points = np.column_stack([xy, plane(xy[:, 0], xy[:, 1])])
    # concave: a notch reaches into the middle
    poly = FAR[:2] + np.array([[8, 8], [92, 10], [90, 90], [50, 40], [12, 88.0]])
    boundary = np.column_stack([poly, plane(poly[:, 0], poly[:, 1])]) if with_z else poly
    dtm = dtm_from_points(points, boundary=boundary)

    assert dtm.plan_area == pytest.approx(_polygon_area(poly - FAR[:2]), rel=1e-9)
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        assert has_edge(dtm, a, b)
    # no vertex outside, none of the outside points kept
    assert dtm.mesh.n_vertices < len(points) + len(poly)
    assert (dtm.mesh.edge_map.counts <= 2).all()
    # queries: NaN outside the polygon (also inside the hull of the data, in the notch)
    notch = FAR[:2] + np.array([[50.0, 60.0], [50.0, 45.0], [30.0, 80.0]])
    assert np.isnan(dtm.z_at(notch)).all()
    assert np.isnan(dtm.z_at(FAR[:2] + [[4.0, 50.0], [95.0, 50.0], [50.0, 5.0]])).all()


def test_boundary_inside_outside_matches_winding_test():
    xy = scatter(2500, 10)
    points = np.column_stack([xy, plane(xy[:, 0], xy[:, 1])])
    poly = FAR[:2] + np.array([[8, 8], [92, 10], [90, 90], [50, 40], [12, 88.0]])
    dtm = dtm_from_points(points, boundary=poly)
    q = FAR[:2] + np.random.default_rng(11).uniform(0, 100, (50_000, 2))
    z = dtm.z_at(q)
    # independent point-in-polygon: sum of signed angles
    d = poly[None, :, :] - q[:, None, :]
    e = np.roll(d, -1, axis=1)
    angle = np.arctan2(
        d[..., 0] * e[..., 1] - d[..., 1] * e[..., 0], (d * e).sum(axis=-1)
    ).sum(axis=1)
    inside = np.abs(angle) > np.pi
    np.testing.assert_array_equal(~np.isnan(z), inside)
    np.testing.assert_allclose(z[inside], plane(q[inside, 0], q[inside, 1]), rtol=0, atol=1e-9)


def test_boundary_errors():
    xy = scatter(200, 12)
    points = np.column_stack([xy, np.zeros(len(xy))])
    with pytest.raises(ValueError, match="outside the data"):
        dtm_from_points(points, boundary=FAR[:2] + [[-50, -50], [50, -50], [50, 50.0]])
    with pytest.raises(ValueError, match="at least 3"):
        dtm_from_points(points, boundary=FAR[:2] + [[0, 0], [50, 50.0]])
    with pytest.raises(ValueError, match="at least 3"):
        dtm_from_points(points[:2])
    with pytest.raises(ValueError):
        dtm_from_points([[0, 0, 0], [1, 1, 1], [2, 2, 2.0]])  # collinear


# ----------------------------------------------------------------- duplicates


def test_duplicate_xy_rules():
    xy = scatter(50, 13)
    points = np.column_stack([xy, np.zeros(len(xy))])
    same = points[[3, 7]]  # exact duplicates: merged silently
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        dtm = dtm_from_points(np.vstack([points, same]))
    assert dtm.mesh.n_vertices == 50

    clash = points[[5, 5, 9]] + [[0, 0, 4.0], [0, 0, 8.0], [0, 0, -3.0]]
    stacked = np.vstack([points, clash])
    expected = {"mean": (4.0, -1.5), "first": (0.0, 0.0), "last": (8.0, -3.0),
                "min": (0.0, -3.0), "max": (8.0, 0.0)}  # fmt: skip
    for rule, (z5, z9) in expected.items():
        with pytest.warns(DuplicatePointWarning, match="2 plan position"):
            dtm = dtm_from_points(stacked, duplicates=rule)
        assert dtm.mesh.n_vertices == 50
        assert dtm.z_at(*points[5, :2]) == pytest.approx(z5, abs=1e-12)
        assert dtm.z_at(*points[9, :2]) == pytest.approx(z9, abs=1e-12)
    with pytest.raises(ValueError, match="different Z"):
        dtm_from_points(stacked, duplicates="error")
    with pytest.raises(ValueError, match="duplicates must be"):
        dtm_from_points(stacked, duplicates="median")
    # a breakline through an existing point with another Z is a duplicate too
    line = np.array([points[1] + [0, 0, 2.0], points[2]])
    with pytest.warns(DuplicatePointWarning):
        dtm = dtm_from_points(points, breaklines=[line])
    assert dtm.z_at(*points[1, :2]) == pytest.approx(1.0)


# ---------------------------------------------------------------- above/below


def test_above_below_and_height():
    dtm, _ = topo()
    v = dtm.mesh.vertices
    rng = np.random.default_rng(14)
    lo, hi = dtm.bounds
    xy = rng.uniform(lo[:2] - 50.0, hi[:2] + 50.0, (20_000, 2))
    z = dtm.z_at(xy)
    off = rng.uniform(-30.0, 30.0, len(xy))
    off[::10] = 0.0
    points = np.column_stack([xy, np.where(np.isnan(z), 1200.0, z + off)])
    side = dtm.above_below(points)
    assert side.dtype == np.int8
    outside = np.isnan(z)
    assert outside.any() and (side[outside] == DTM.OUTSIDE).all()
    height = dtm.height_above(points)
    np.testing.assert_array_equal(np.isnan(height), outside)
    np.testing.assert_array_equal(side[~outside], np.sign(height[~outside]).astype(np.int8))
    assert (side == DTM.ABOVE).sum() > 5000 and (side == DTM.BELOW).sum() > 5000
    assert (side == DTM.ON).sum() >= (~outside[::10]).sum() * 0.9
    assert (dtm.above_below(v + [0, 0, 0.001]) == DTM.ABOVE).all()
    assert (dtm.above_below(v - [0, 0, 0.001]) == DTM.BELOW).all()
    assert dtm.above_below(np.zeros((3, 4, 3))).shape == (3, 4)


def test_dtm_wrapper_orients_and_rejects_vertical_faces():
    up = Mesh([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 5.0]], [[0, 1, 2], [1, 2, 3]])
    dtm = DTM(up)
    assert (dtm.mesh.face_normals[:, 2] > 0).all()
    assert dtm.z_at(1.0, 1.0) == 5.0 and dtm.z_at(0.75, 0.75) == pytest.approx(2.5)
    wall = Mesh([[0, 0, 0], [1, 0, 0], [1, 0, 1.0]], [[0, 1, 2]])
    with pytest.raises(ValueError, match="zero area in plan"):
        DTM(wall)
    empty = DTM(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64)))
    assert np.isnan(empty.z_at([1.0, 2.0], [3.0, 4.0])).all()


# -------------------------------------------------------- synthetic topography


def test_synthetic_topography_matches_generating_function():
    dtm, truth = topo()
    v = dtm.mesh.vertices
    assert dtm.mesh.n_faces == 1772
    # exactly at every vertex (grid nodes and drillhole collars, border included)
    np.testing.assert_array_equal(dtm.z_at(v[:, 0], v[:, 1]), v[:, 2])
    np.testing.assert_array_equal(dtm.z_at(v[:, 0], v[:, 1]), truth.z(v[:, 0], v[:, 1]))
    # close in between: 25 m grid on waves of 160 m and longer
    lo, hi = dtm.bounds
    q = np.random.default_rng(15).uniform(lo[:2], hi[:2], (200_000, 2))
    err = dtm.z_at(q) - truth.z(q[:, 0], q[:, 1])
    assert not np.isnan(err).any()
    print(f"topo: max |error| {np.abs(err).max():.3f} m, rms {np.sqrt(np.mean(err**2)):.3f} m")
    assert np.abs(err).max() < 1.0
    assert np.sqrt(np.mean(err**2)) < 0.3
    # rebuilding from the points gives an equally good surface (grid diagonals may differ)
    rebuilt = dtm_from_points(v)
    assert rebuilt.mesh.n_vertices == dtm.mesh.n_vertices
    assert rebuilt.plan_area == pytest.approx(dtm.plan_area, rel=1e-12)
    assert np.abs(rebuilt.z_at(q) - truth.z(q[:, 0], q[:, 1])).max() < 1.0


def test_query_performance_1m_points():
    dtm, _ = topo()
    lo, hi = dtm.bounds
    q = np.random.default_rng(16).uniform(lo[:2] - 20.0, hi[:2] + 20.0, (1_000_000, 2))
    dtm.z_at(q[:10])  # compile
    start = time.perf_counter()
    z = dtm.z_at(q[:, 0], q[:, 1])
    elapsed = time.perf_counter() - start
    print(f"z_at, 1M points: {elapsed:.3f} s")
    assert 0.8 < (~np.isnan(z)).mean() < 0.95
    assert elapsed < 2.0
