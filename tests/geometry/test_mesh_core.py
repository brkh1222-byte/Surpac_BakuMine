"""P2.T1 acceptance: mesh core on cube and tetrahedron, indexes vs brute force."""

import numpy as np
import pytest

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, cube, icosphere, subdivide, tetrahedron, torus
from bakumine.core.geometry.spatial import TriangleBVH, TriangleGridXY


def soup(mesh: Mesh) -> Mesh:
    """Triangle soup: every face gets its own three vertices."""
    return Mesh(mesh.triangles.reshape(-1, 3), np.arange(3 * mesh.n_faces).reshape(-1, 3))


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def test_mesh_stores_float64_int64_readonly_copies():
    v = [[0, 0, 0], [1, 0, 0], [0, 1, 0]]
    f = np.array([[0, 1, 2]], dtype=np.int32)
    m = Mesh(v, f)
    assert m.vertices.dtype == np.float64 and m.vertices.shape == (3, 3)
    assert m.faces.dtype == np.int64 and m.faces.shape == (1, 3)
    assert (m.n_vertices, m.n_faces) == (3, 1)
    f[0, 0] = 2
    assert m.faces[0, 0] == 0
    with pytest.raises(ValueError):
        m.vertices[0, 0] = 5.0


@pytest.mark.parametrize(
    "vertices, faces",
    [
        ([[0, 0], [1, 0], [0, 1]], [[0, 1, 2]]),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1]]),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, 3]]),
        ([[0, 0, 0], [1, 0, 0], [0, 1, 0]], [[0, 1, -1]]),
        ([[0, 0, 0], [1, 0, 0], [0, np.nan, 0]], [[0, 1, 2]]),
    ],
)
def test_mesh_rejects_bad_input(vertices, faces):
    with pytest.raises(ValueError):
        Mesh(vertices, faces)


def test_empty_mesh():
    m = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert m.n_faces == 0 and m.edge_map.n_edges == 0
    assert m.face_normals.shape == (0, 3)
    assert len(m.bvh.query_box([0, 0, 0], [1, 1, 1])) == 0
    assert len(m.bvh.self_overlaps()) == 0
    assert len(m.grid_xy.query_point(0.0, 0.0)) == 0
    assert m.weld().n_vertices == 0


# ---------------------------------------------------------------- weld


@pytest.mark.parametrize(
    "mesh, n_vertices", [(cube(), 8), (tetrahedron(), 4)], ids=["cube", "tetrahedron"]
)
def test_weld_soup_restores_shared_vertices(mesh, n_vertices):
    s = soup(mesh)
    assert s.n_vertices == 3 * mesh.n_faces
    for tol in (0.0, 1e-9):
        w = s.weld(tol)
        assert w.n_vertices == n_vertices
        assert w.n_faces == mesh.n_faces
        np.testing.assert_array_equal(w.triangles, mesh.triangles)
        assert (w.edge_map.counts == 2).all()


def test_weld_tolerance():
    v = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1e-7, 0, 0], [1, 1e-3, 0], [0, 0, 1.0]])
    f = np.array([[0, 1, 2], [3, 4, 5]])
    m = Mesh(v, f)
    assert m.weld(0.0).n_vertices == 6
    w = m.weld(1e-6)
    assert w.n_vertices == 5  # only vertex 3 merged, into vertex 0
    np.testing.assert_array_equal(w.vertices, v[[0, 1, 2, 4, 5]])
    np.testing.assert_array_equal(w.faces, [[0, 1, 2], [0, 3, 4]])
    assert m.weld(1e-2).n_vertices == 4
    with pytest.raises(ValueError):
        m.weld(-1.0)


def test_weld_is_transitive_and_idempotent():
    # chain 0 - 0.6 - 1.2 with tolerance 1: all three merge into the first
    v = np.array([[0, 0, 0], [0.6, 0, 0], [1.2, 0, 0], [5, 0, 0], [5, 5, 0]])
    m = Mesh(v, [[0, 3, 4], [1, 3, 4], [2, 3, 4]])
    w = m.weld(1.0)
    assert w.n_vertices == 3
    np.testing.assert_array_equal(w.vertices[0], [0, 0, 0])
    np.testing.assert_array_equal(w.faces, [[0, 1, 2]] * 3)
    again = w.weld(1.0)
    np.testing.assert_array_equal(again.vertices, w.vertices)
    np.testing.assert_array_equal(again.faces, w.faces)


def test_weld_drops_collapsed_faces_on_request():
    v = np.array([[0, 0, 0], [1e-12, 0, 0], [0, 1, 0], [1, 0, 0]])
    m = Mesh(v, [[0, 1, 2], [0, 3, 2]])
    assert m.weld(1e-9).n_faces == 1
    kept = m.weld(1e-9, remove_collapsed=False)
    assert kept.n_faces == 2
    np.testing.assert_array_equal(kept.faces[0], [0, 0, 1])


def test_weld_far_from_origin():
    m = soup(cube(10.0, center=(5.0e6, 4.0e6, 300.0)))
    w = m.weld(1e-6)
    assert (w.n_vertices, w.n_faces) == (8, 12)


def test_remove_unreferenced_vertices():
    c = cube()
    extra = np.vstack([[9.0, 9, 9], c.vertices[:4], [8.0, 8, 8], c.vertices[4:], [7.0, 7, 7]])
    remap = np.array([1, 2, 3, 4, 6, 7, 8, 9])
    m = Mesh(extra, remap[c.faces])
    cleaned = m.remove_unreferenced_vertices()
    np.testing.assert_array_equal(cleaned.vertices, c.vertices)
    np.testing.assert_array_equal(cleaned.faces, c.faces)
    assert c.remove_unreferenced_vertices() is c


# ------------------------------------------------------------ edge map


@pytest.mark.parametrize(
    "mesh, n_edges", [(cube(), 18), (tetrahedron(), 6)], ids=["cube", "tetrahedron"]
)
def test_edge_map_counts(mesh, n_edges):
    em = mesh.edge_map
    assert em.n_edges == n_edges
    assert mesh.n_vertices - em.n_edges + mesh.n_faces == 2  # Euler, genus 0
    assert (em.counts == 2).all()
    assert len(em.boundary_edges) == 0 and len(em.nonmanifold_edges) == 0
    assert (em.edges[:, 0] < em.edges[:, 1]).all()
    assert em.offsets[-1] == 3 * mesh.n_faces
    np.testing.assert_array_equal(mesh.edges, em.edges)


def test_edge_map_matches_brute_force():
    mesh = cube()
    em = mesh.edge_map
    expected: dict[tuple[int, int], list[int]] = {}
    for f, (a, b, c) in enumerate(mesh.faces.tolist()):
        for u, w in ((a, b), (b, c), (c, a)):
            expected.setdefault((min(u, w), max(u, w)), []).append(f)
    assert [tuple(e) for e in em.edges.tolist()] == sorted(expected)
    for e, key in enumerate(sorted(expected)):
        assert em.faces_of(e).tolist() == sorted(expected[key])
        assert em.find(*key) == e and em.find(key[1], key[0]) == e
    assert em.find(0, 6) == -1  # body diagonal is not an edge
    # face_edges / forward describe each half-edge
    for f in range(mesh.n_faces):
        for j in range(3):
            a, b = mesh.faces[f, j], mesh.faces[f, (j + 1) % 3]
            assert em.edges[em.face_edges[f, j]].tolist() == [min(a, b), max(a, b)]
            assert em.forward[f, j] == (a < b)
    # a consistently wound closed mesh crosses every edge once in each direction
    pairs = em.manifold_halfedge_pairs
    assert len(pairs) == em.n_edges
    assert (em.forward.reshape(-1)[pairs].sum(axis=1) == 1).all()


def test_edge_map_boundary_and_nonmanifold():
    c = cube()
    em = c.submesh(np.arange(1, 12)).edge_map  # one face removed
    assert len(em.boundary_edges) == 3
    assert sorted(map(tuple, em.edges[em.boundary_edges].tolist())) == [(0, 1), (0, 2), (1, 2)]
    fin = Mesh(np.vstack([c.vertices, [[0.5, 0.5, 3.0]]]), np.vstack([c.faces, [[4, 5, 8]]]))
    em = fin.edge_map
    assert em.edges[em.nonmanifold_edges].tolist() == [[4, 5]]
    assert em.counts[em.find(4, 5)] == 3
    assert em.faces_of(em.find(4, 5)).tolist() == [2, 5, 12]


# ------------------------------------------------------ normals, bounds


def test_cube_normals_areas_bounds():
    c = cube(2.0, center=(1.0, 2.0, 3.0))
    axes = [[0, 0, -1], [0, 0, 1], [0, -1, 0], [0, 1, 0], [1, 0, 0], [-1, 0, 0]]
    expected = np.repeat(axes, 2, axis=0)
    np.testing.assert_array_equal(c.face_normals, expected)
    np.testing.assert_array_equal(c.face_areas, np.full(12, 2.0))
    np.testing.assert_array_equal(c.bounds, [[0, 1, 2], [2, 3, 4]])
    np.testing.assert_array_equal(c.local_origin, [1, 2, 3])
    lo, hi = c.face_bounds
    np.testing.assert_array_equal(lo, c.triangles.min(axis=1))
    np.testing.assert_array_equal(hi, c.triangles.max(axis=1))
    # area-weighted vertex normal of corner 0: two bottom, two front, one left face
    np.testing.assert_allclose(c.vertex_normals[0], np.array([-1, -2, -2]) / 3.0)
    s = icosphere(3, 2.0, center=(1.0, 2.0, 3.0))
    np.testing.assert_allclose(s.vertex_normals, (s.vertices - [1, 2, 3]) / 2.0, atol=0.02)


def test_tetrahedron_normals_point_outward():
    t = tetrahedron(2.0)
    centre = t.vertices.mean(axis=0)
    outward = t.triangles.mean(axis=1) - centre
    assert (np.einsum("ij,ij->i", t.face_normals, outward) > 0).all()
    np.testing.assert_allclose(np.linalg.norm(t.face_normals, axis=1), 1.0)
    np.testing.assert_allclose(t.face_normals[3], np.ones(3) / np.sqrt(3))
    np.testing.assert_allclose(t.face_areas, [2, 2, 2, 2 * np.sqrt(3)])
    np.testing.assert_array_equal(t.bounds, [[0, 0, 0], [2, 2, 2]])


def test_normals_degenerate_face_and_far_origin():
    m = Mesh([[0, 0, 0], [1, 0, 0], [2, 0, 0], [0, 1, 0]], [[0, 1, 2], [0, 1, 3]])
    np.testing.assert_array_equal(m.face_normals, [[0, 0, 0], [0, 0, 1]])
    np.testing.assert_array_equal(m.face_areas, [0.0, 0.5])
    near, far = icosphere(2), icosphere(2).translated([5.0e6, 4.0e6, 1.0e3])
    np.testing.assert_allclose(far.face_normals, near.face_normals, atol=1e-8)
    np.testing.assert_allclose(far.face_areas, near.face_areas, rtol=1e-7)


def test_flipped_transformed_concatenate():
    c = cube()
    np.testing.assert_array_equal(c.flipped().face_normals, -c.face_normals)
    one = c.flipped([3])
    np.testing.assert_array_equal(one.face_normals[3], -c.face_normals[3])
    np.testing.assert_array_equal(one.face_normals[4], c.face_normals[4])
    mirror = c.transformed(np.diag([-1.0, 1.0, 1.0]))  # reflection keeps normals outward
    outward = mirror.triangles.mean(axis=1)
    assert (np.einsum("ij,ij->i", mirror.face_normals, outward) > 0).all()
    t = np.eye(4)
    t[:3, 3] = [1, 2, 3]
    np.testing.assert_array_equal(c.transformed(t).vertices, c.vertices + [1, 2, 3])
    both = Mesh.concatenate([c, c.translated([5, 0, 0])])
    assert (both.n_vertices, both.n_faces) == (16, 24)
    np.testing.assert_array_equal(both.faces[12:], c.faces + 8)


def test_primitives_are_closed_with_expected_counts():
    for mesh, n_faces, euler in [
        (cube(), 12, 2),
        (box([0, 0, 0], [1, 2, 3]), 12, 2),
        (tetrahedron(), 4, 2),
        (icosphere(2), 320, 2),
        (subdivide(cube()), 48, 2),
        (torus(2.0, 0.5, 12, 8), 192, 0),
    ]:
        em = mesh.edge_map
        assert mesh.n_faces == n_faces
        assert (em.counts == 2).all()
        assert mesh.n_vertices - em.n_edges + mesh.n_faces == euler
        assert (em.forward.reshape(-1)[em.manifold_halfedge_pairs].sum(axis=1) == 1).all()
    np.testing.assert_allclose(np.linalg.norm(icosphere(3, radius=2.5).vertices, axis=1), 2.5)


# ------------------------------------------------------- spatial index


def brute_points(lo, hi, xy):
    inside = (lo[None, :, :2] <= xy[:, None, :]) & (xy[:, None, :] <= hi[None, :, :2])
    return np.nonzero(inside.all(axis=2))


def brute_boxes(lo, hi, qlo, qhi, dims=3):
    touch = (lo[None, :, :dims] <= qhi[:, None, :]) & (qlo[:, None, :] <= hi[None, :, :dims])
    return np.nonzero(touch.all(axis=2))


def index_meshes():
    rng = np.random.default_rng(7)
    yield "cube", cube()
    yield "tetrahedron", tetrahedron()
    yield "cube_subdivided", subdivide(subdivide(cube(3.0)))
    yield (
        "sphere_rotated_far",
        icosphere(3, 40.0).transformed(random_rotation(rng)).translated([5.0e6, 4.0e6, 250.0]),
    )
    yield "torus", torus(3.0, 1.0, 24, 12).transformed(random_rotation(rng))
    # one huge triangle next to many small ones
    small = icosphere(3, 0.5)
    big = Mesh([[-50, -50, 0], [50, -50, 0], [0, 80, 5.0]], [[0, 1, 2]])
    yield "mixed_sizes", Mesh.concatenate([small, big])
    wall = Mesh([[0, 0, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1.0]], [[0, 1, 2], [0, 2, 3]])
    yield "vertical_wall", wall


INDEX_MESHES = dict(index_meshes())


def sample_points(mesh, rng, n):
    lo, hi = mesh.bounds
    pad = 0.1 * (hi - lo).max()
    pts = rng.uniform(lo - pad, hi + pad, size=(n, 3))
    # include points exactly on vertices and bounding-box corners (boundary cases)
    k = min(n - 2, mesh.n_vertices)
    pts[:k] = mesh.vertices[:k]
    pts[-2:] = [lo, hi]
    return pts


@pytest.mark.parametrize("name", INDEX_MESHES)
def test_grid_xy_point_query_matches_brute_force(name):
    mesh = INDEX_MESHES[name]
    rng = np.random.default_rng(1)
    grid = mesh.grid_xy
    lo, hi = mesh.face_bounds
    xy = sample_points(mesh, rng, 500)[:, :2]
    pi, ti = grid.query_points(xy)
    bi, bt = brute_points(lo, hi, xy)
    np.testing.assert_array_equal(pi, bi)
    np.testing.assert_array_equal(ti, bt)
    for k in (0, 3, 499):
        np.testing.assert_array_equal(grid.query_point(*xy[k]), bt[bi == k])


@pytest.mark.parametrize("name", INDEX_MESHES)
def test_grid_xy_rect_query_matches_brute_force(name):
    mesh = INDEX_MESHES[name]
    rng = np.random.default_rng(2)
    lo, hi = mesh.face_bounds
    a, b = sample_points(mesh, rng, 200)[:, :2], sample_points(mesh, rng, 200)[::-1, :2]
    qlo, qhi = np.minimum(a, b), np.maximum(a, b)
    qhi[:20] = qlo[:20]  # zero-size rectangles
    ri, ti = mesh.grid_xy.query_rects(qlo, qhi)
    bi, bt = brute_boxes(lo, hi, qlo, qhi, dims=2)
    np.testing.assert_array_equal(ri, bi)
    np.testing.assert_array_equal(ti, bt)
    np.testing.assert_array_equal(mesh.grid_xy.query_rect(qlo[50], qhi[50]), bt[bi == 50])


@pytest.mark.parametrize("name", INDEX_MESHES)
@pytest.mark.parametrize("leaf_size", [1, 8])
def test_bvh_box_query_matches_brute_force(name, leaf_size):
    mesh = INDEX_MESHES[name]
    rng = np.random.default_rng(3)
    bvh = TriangleBVH(mesh.vertices, mesh.faces, leaf_size=leaf_size)
    lo, hi = mesh.face_bounds
    a, b = sample_points(mesh, rng, 200), sample_points(mesh, rng, 200)[::-1]
    qlo, qhi = np.minimum(a, b), np.maximum(a, b)
    qhi[:20] = qlo[:20]  # point queries
    qi, ti = bvh.query_boxes(qlo, qhi)
    bi, bt = brute_boxes(lo, hi, qlo, qhi)
    np.testing.assert_array_equal(qi, bi)
    np.testing.assert_array_equal(ti, bt)
    np.testing.assert_array_equal(bvh.query_box(qlo[50], qhi[50]), bt[bi == 50])


@pytest.mark.parametrize("name", INDEX_MESHES)
def test_bvh_self_overlaps_match_brute_force(name):
    mesh = INDEX_MESHES[name]
    lo, hi = mesh.face_bounds
    i, j = brute_boxes(lo, hi, lo, hi)
    keep = i < j
    np.testing.assert_array_equal(mesh.bvh.self_overlaps(), np.column_stack([i[keep], j[keep]]))


def test_index_queries_outside_and_invalid():
    c = cube()
    assert len(c.grid_xy.query_point(10.0, 0.0)) == 0
    assert len(c.grid_xy.query_point(np.nan, 0.0)) == 0
    assert len(c.grid_xy.query_rect([2, 2], [3, 3])) == 0
    assert len(c.grid_xy.query_rect([0.4, 0.4], [-0.4, -0.4])) == 0  # inverted
    assert len(c.grid_xy.query_rect([-9, -9], [9, 9])) == 12
    assert len(c.bvh.query_box([2, 2, 2], [3, 3, 3])) == 0
    assert len(c.bvh.query_box([0.4, 0.4, 0.4], [-0.4, -0.4, -0.4])) == 0
    assert len(c.bvh.query_box([np.nan, 0, 0], [1, 1, 1])) == 0
    assert c.bvh.query_box([-9, -9, -9], [9, 9, 9]).tolist() == list(range(12))
    # a vertical line through the cube meets the boxes of all four top/bottom faces
    assert set(c.grid_xy.query_point(0.1, 0.2).tolist()) >= {0, 1, 2, 3}
    assert c.grid_xy.cell_of(-100.0, -100.0) == 0


def test_grid_structure_is_documented_csr():
    mesh = INDEX_MESHES["torus"]
    grid = TriangleGridXY(mesh.vertices, mesh.faces, cells_per_triangle=2.0)
    nx, ny = grid.shape
    assert len(grid.cell_start) == nx * ny + 1 and grid.cell_start[-1] == len(grid.cell_tris)
    x, y = mesh.vertices[5, :2]
    c = grid.cell_of(x, y)
    ix = int(np.floor((x - grid.origin[0]) * grid.inv_cell_size[0]))
    iy = int(np.floor((y - grid.origin[1]) * grid.inv_cell_size[1]))
    assert c == min(ix, nx - 1) * ny + min(iy, ny - 1)
    cell = grid.cell_tris[grid.cell_start[c] : grid.cell_start[c + 1]]
    assert (np.diff(cell) > 0).all()
    assert set(grid.query_point(x, y).tolist()) <= set(cell.tolist())


def test_index_on_large_mesh_agrees_between_grid_and_bvh():
    mesh = icosphere(6, 100.0, center=(5.0e6, 4.0e6, 0.0))  # 81,920 faces
    rng = np.random.default_rng(4)
    xy = rng.uniform(-100, 100, size=(2000, 2)) + mesh.local_origin[:2]
    pi, ti = mesh.grid_xy.query_points(xy)
    column_lo = np.column_stack([xy, np.full(len(xy), -np.inf)])
    column_hi = np.column_stack([xy, np.full(len(xy), np.inf)])
    qi, tj = mesh.bvh.query_boxes(column_lo, column_hi)
    np.testing.assert_array_equal(pi, qi)
    np.testing.assert_array_equal(ti, tj)
    assert len(pi) > 2000
