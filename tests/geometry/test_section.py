"""P2.T7 acceptance: mesh-plane section.

Oracles: analytic polygons (cube, torus rings), the circle for the sphere
(within tessellation error), trimesh ``mesh_plane`` segments on generic planes,
and the mesh volume as the integral of section areas.
"""

import math
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, cube, icosphere, torus
from bakumine.core.geometry.section import Section, plane_frame, section

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
FAR = np.array([5.0e6, 4.0e6, 1.0e3])


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def check_well_formed(s: Section, *, closed: bool = True, tol: float = 1e-9):
    """Points on the plane, no repeated points, no zero-length segments, flags consistent."""
    assert len(s.polylines) == len(s.closed)
    for pts, is_closed in zip(s.polylines, s.closed, strict=True):
        assert is_closed == closed
        assert len(pts) >= (3 if is_closed else 2)
        assert np.abs((pts - s.origin) @ s.normal).max() <= tol
        ring = np.vstack([pts, pts[:1]]) if is_closed else pts
        assert (np.linalg.norm(np.diff(ring, axis=0), axis=1) > 0).all()
        assert len(np.unique(pts, axis=0)) == len(pts)  # no point visited twice


def trimesh_section(mesh: Mesh, origin, normal, convex: bool = False):
    """Oracle: trimesh's unordered cut segments -> (area or None, total length).

    The area is only available for a convex cut (fan of triangles about the
    mean point); trimesh needs shapely, which is not installed, to chain loops.
    """
    shift = mesh.local_origin
    tm = trimesh.Trimesh(mesh.vertices - shift, mesh.faces, process=False)
    segments = trimesh.intersections.mesh_plane(
        tm, plane_normal=np.asarray(normal, dtype=float), plane_origin=np.asarray(origin) - shift
    )
    length = float(np.linalg.norm(segments[:, 1] - segments[:, 0], axis=1).sum())
    area = None
    if convex:
        centre = segments.reshape(-1, 3).mean(axis=0)
        fans = np.cross(segments[:, 0] - centre, segments[:, 1] - centre)
        area = float(0.5 * np.linalg.norm(fans, axis=1).sum())
    return area, length


# ----------------------------------------------------------------------- cube


@pytest.mark.parametrize("center", [(0.0, 0.0, 0.0), tuple(FAR)])
def test_cube_axis_aligned_sections_are_exact(center):
    c = np.array(center)
    mesh = cube(2.0, center=center)
    for axis in range(3):
        for sign in (1.0, -1.0):
            normal = np.zeros(3)
            normal[axis] = sign
            for offset in (-0.999, -0.5, 0.0, 0.25, 0.75):
                s = section(mesh, c + offset * normal, normal)
                check_well_formed(s, tol=0.0)
                assert len(s) == 1 and s.closed == (True,)
                assert s.area == 4.0 and s.length == 8.0  # counter-clockwise: positive
                uv = s.polylines_2d[0]
                assert np.abs(uv - s.to_plane(c)).max() == 1.0


def test_cube_plane_on_a_face_is_half_open():
    mesh = cube(2.0, center=FAR)
    up = (0.0, 0.0, 1.0)
    bottom = section(mesh, FAR - [0, 0, 1.0], up)
    check_well_formed(bottom, tol=0.0)
    assert len(bottom) == 1 and bottom.area == 4.0 and bottom.length == 8.0
    assert len(bottom.polylines[0]) == 4  # exactly the four corners, no extras
    assert len(section(mesh, FAR + [0, 0, 1.0], up)) == 0
    # seen from the other side the roles swap
    down = (0.0, 0.0, -1.0)
    assert section(mesh, FAR + [0, 0, 1.0], down).area == 4.0
    assert len(section(mesh, FAR - [0, 0, 1.0], down)) == 0
    # outside
    assert len(section(mesh, FAR + [0, 0, 1.5], up)) == 0
    assert section(mesh, FAR + [0, 0, 1.5], up).area == 0.0


def test_cube_planes_through_vertices_and_edges():
    mesh = cube(2.0)
    r2, r3 = math.sqrt(2.0), math.sqrt(3.0)
    # through two opposite vertical edges: a 2 x 2*sqrt(2) rectangle
    s = section(mesh, (0, 0, 0), (1, 1, 0))
    check_well_formed(s, tol=1e-15)
    assert len(s) == 1
    assert s.area == pytest.approx(4 * r2, rel=1e-14)
    assert s.length == pytest.approx(4 + 4 * r2, rel=1e-14)
    assert len(s.polylines[0]) == 6  # 4 corners + 2 crossings of the cap diagonals
    # through the centre, normal along the body diagonal: regular hexagon of side sqrt(2)
    s = section(mesh, (0, 0, 0), (1, 1, 1))
    check_well_formed(s, tol=1e-15)
    assert s.area == pytest.approx(3 * r3, rel=1e-14)
    assert s.length == pytest.approx(6 * r2, rel=1e-14)
    # through three vertices: equilateral triangle of side 2*sqrt(2); its sides
    # are face diagonals, some of them mesh edges lying in the plane
    for normal, expected in (((1, 1, 1), 2 * r3), ((-1, -1, -1), 2 * r3)):
        s = section(mesh, (1, 1, -1), normal)
        check_well_formed(s, tol=1e-15)
        assert len(s) == 1 and 3 <= len(s.polylines[0]) <= 6
        assert s.area == pytest.approx(expected, rel=1e-14)
        assert s.length == pytest.approx(6 * r2, rel=1e-14)
    # touching along one edge only, or at one vertex only: nothing
    for origin, normal in (((1, 1, 0), (1, 1, 0)), ((1, 1, 1), (1, 1, 1))):
        assert len(section(mesh, origin, normal)) == 0
        assert len(section(mesh, origin, np.negative(normal))) == 0
    # a hair inside the vertex: a small triangle
    s = section(mesh, (1 - 1e-3, 1 - 1e-3, 1 - 1e-3), (1, 1, 1))
    assert len(s) == 1 and 3 <= len(s.polylines[0]) <= 6  # corners + diagonal crossings
    assert s.area == pytest.approx(r3 / 2 * (3e-3) ** 2, rel=1e-9)


def test_cube_oblique_planes_match_trimesh_far_from_origin():
    rng = np.random.default_rng(0)
    mesh = cube(2.0).transformed(random_rotation(rng)).translated(FAR)
    hits = 0
    for _ in range(60):
        normal = rng.normal(size=3)
        origin = FAR + rng.uniform(-0.8, 0.8, 3)
        s = section(mesh, origin, normal)
        check_well_formed(s)
        area, length = trimesh_section(mesh, origin, normal, convex=True)
        assert len(s) == 1
        assert s.area == pytest.approx(area, rel=1e-9)
        assert s.length == pytest.approx(length, rel=1e-9)
        hits += 1
    assert hits == 60


# --------------------------------------------------------------------- sphere


def test_sphere_sections_match_the_circle():
    rng = np.random.default_rng(1)
    radius = 25.0
    mesh = icosphere(4, radius=radius, center=FAR)
    edge = np.linalg.norm(np.diff(mesh.vertices[mesh.edges], axis=1)[:, 0], axis=1).max()
    for k in range(40):
        normal = rng.normal(size=3) if k >= 6 else np.eye(3)[k % 3] * (1 if k < 3 else -1)
        h = rng.uniform(-0.9, 0.9) * radius if k >= 3 else 0.0
        origin = FAR + h * normal / np.linalg.norm(normal)
        s = section(mesh, origin, normal)
        check_well_formed(s)
        assert len(s) == 1 and s.closed == (True,)
        rc = math.sqrt(radius**2 - h**2)
        # the cut is a polygon with its corners on chords of the sphere: inside
        # the circle, by at most the sagitta of the longest mesh edge
        r = np.linalg.norm(s.polylines[0] - origin, axis=1)
        sagitta = radius - math.sqrt(radius**2 - (edge / 2) ** 2)
        assert r.max() <= rc + 1e-9
        assert r.min() >= rc - sagitta * radius / rc * 1.01 - 1e-9
        assert math.pi * (rc - 2.1 * sagitta * radius / rc) ** 2 < s.area < math.pi * rc**2
        assert 2 * math.pi * (rc - 2.1 * sagitta * radius / rc) < s.length < 2 * math.pi * rc
        if k >= 6:
            area, length = trimesh_section(mesh, origin, normal, convex=True)
            assert s.area == pytest.approx(area, rel=1e-9)
            assert s.length == pytest.approx(length, rel=1e-9)


def test_sphere_planes_through_many_vertices():
    # The coordinate planes contain whole rings of icosphere vertices and edges.
    mesh = icosphere(3, radius=10.0)
    for axis in range(3):
        normal = np.eye(3)[axis]
        on_plane = int((mesh.vertices[:, axis] == 0.0).sum())
        assert on_plane >= 4
        s = section(mesh, (0, 0, 0), normal)
        check_well_formed(s, tol=0.0)
        assert len(s) == 1
        # same loop as a plane a hair below, where no vertex is on the plane
        t = section(mesh, -1e-9 * normal, normal)
        check_well_formed(t)
        assert s.area == pytest.approx(t.area, rel=1e-8)
        assert s.length == pytest.approx(t.length, rel=1e-8)
        # every on-plane vertex is a corner of the loop, exactly
        loop = {tuple(p) for p in s.polylines[0].tolist()}
        for v in mesh.vertices[mesh.vertices[:, axis] == 0.0]:
            assert tuple(v.tolist()) in loop
        assert s.signed_areas[0] > 0
        assert section(mesh, (0, 0, 0), -normal).area == pytest.approx(s.area, rel=1e-12)


# ---------------------------------------------------------------------- torus


def test_torus_cut_through_its_hole_gives_two_loops():
    big, small, nu, nw = 2.0, 0.5, 48, 24
    mesh = torus(big, small, nu, nw)
    tube = 0.5 * nw * small**2 * math.sin(2 * math.pi / nw)  # area of the 24-gon
    tube_perimeter = 2 * nw * small * math.sin(math.pi / nw)
    for normal in ((0, 1, 0), (1, 0, 0), (0, -1, 0), (1, 1, 0)):
        # these planes contain the axis and whole rings of vertices (or pass
        # within rounding of them: sin(pi) is 1.2e-16, not 0)
        s = section(mesh, (0, 0, 0), normal)
        check_well_formed(s, tol=1e-15)
        assert len(s) == 2 and s.closed == (True, True)
        assert [len(p) for p in s.polylines] == [nw, nw]
        np.testing.assert_allclose(s.signed_areas, [tube, tube], rtol=1e-12)
        np.testing.assert_allclose(s.lengths, [tube_perimeter, tube_perimeter], rtol=1e-12)
        centres = sorted(float(p[:, 0].mean()) for p in s.polylines_2d)
        np.testing.assert_allclose(centres, [-big, big], atol=1e-12)
    # a generic plane through the hole
    s = section(mesh, (0.01, 0.02, 0.03), (0.1, 1.0, 0.05))
    check_well_formed(s)
    assert len(s) == 2
    _, length = trimesh_section(mesh, (0.01, 0.02, 0.03), (0.1, 1.0, 0.05))
    assert s.length == pytest.approx(length, rel=1e-9)
    assert s.signed_areas.min() > 0.9 * tube


def test_torus_equatorial_cut_is_an_annulus_with_a_clockwise_hole():
    big, small, nu, nw = 2.0, 0.5, 48, 24
    mesh = torus(big, small, nu, nw)
    k = 0.5 * nu * math.sin(2 * math.pi / nu)
    for normal in ((0, 0, 1), (0, 0, -1)):
        s = section(mesh, (0, 0, 0), normal)
        check_well_formed(s, tol=1e-15)
        assert len(s) == 2
        areas = sorted(s.signed_areas)
        np.testing.assert_allclose(
            areas, [-k * (big - small) ** 2, k * (big + small) ** 2], rtol=1e-12
        )
        assert s.area == pytest.approx(k * ((big + small) ** 2 - (big - small) ** 2), rel=1e-12)
    # resting on the plane / hanging under it: a tangent circle is not a section
    assert len(section(mesh, (0, 0, -small), (0, 0, 1))) == 0
    assert len(section(mesh, (0, 0, small), (0, 0, 1))) == 0
    # off the tangent plane by a hair: two loops again
    assert len(section(mesh, (0, 0, -small + 1e-6), (0, 0, 1))) == 2


# ------------------------------------------------------- other closed meshes


def test_cavity_is_a_hole_in_the_section():
    outer = box(FAR, FAR + 10.0)
    inner = box(FAR + 3.0, FAR + 7.0).flipped()
    mesh = Mesh.concatenate([outer, inner])
    s = section(mesh, FAR + 5.0, (0.0, 1.0, 0.0))
    check_well_formed(s)
    assert len(s) == 2
    assert sorted(s.signed_areas) == [-16.0, 100.0] and s.area == 84.0
    assert s.length == 56.0
    # through the top of the cavity (its faces are on the plane): half-open
    assert section(mesh, FAR + 7.0, (0, 0, 1.0)).area == 100.0
    assert section(mesh, FAR + 3.0, (0, 0, 1.0)).area == 84.0


def test_synthetic_solid_sections_are_closed_and_match_trimesh():
    mesh = Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))
    rng = np.random.default_rng(2)
    lo, hi = mesh.bounds
    total = 0
    for k in range(40):
        normal = rng.normal(size=3) if k % 2 else np.eye(3)[k % 3]
        origin = mesh.centroid + rng.uniform(-0.3, 0.3, 3) * (hi - lo)
        s = section(mesh, origin, normal)
        check_well_formed(s, tol=1e-8)
        _, length = trimesh_section(mesh, origin, normal)
        assert all(s.closed)
        assert s.length == pytest.approx(length, rel=1e-8, abs=1e-6)
        assert (s.signed_areas > 0).all()  # no cavities: every loop is an outer contour
        total += len(s)
    assert total >= 30
    # exactly through mesh vertices, every one of them a corner of the result
    for i in (0, 17, 4000):
        v = mesh.vertices[i]
        for normal in ((0, 0, 1.0), (1.0, 0, 0)):
            s = section(mesh, v, normal, tol=0.0)
            check_well_formed(s, tol=1e-8)
            assert all(s.closed) and len(s) >= 1
            assert any((p == v).all(axis=1).any() for p in s.polylines)
    # volume by slices (independent of the divergence-theorem volume)
    z = np.linspace(lo[2], hi[2], 401)
    mid = 0.5 * (z[:-1] + z[1:])
    areas = [section(mesh, (0.0, 0.0, zz), (0, 0, 1)).area for zz in mid]
    assert np.sum(areas) * (z[1] - z[0]) == pytest.approx(mesh.volume, rel=1e-3)


# ------------------------------------------------------------------ open mesh


def test_open_surface_gives_open_profiles():
    dtm = DTM(Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy")))
    lo, hi = dtm.bounds
    centre = 0.5 * (lo + hi)
    # a vertical east-west section looking north, on a grid line of the DTM
    for y in (centre[1], lo[1] + 25.0 * 7, centre[1] + 3.21):
        s = section(dtm.mesh, (centre[0], y, 0.0), (0.0, 1.0, 0.0))
        check_well_formed(s, closed=False, tol=1e-8)
        assert len(s) == 1 and s.closed == (False,)
        profile = s.polylines[0]
        assert profile[:, 0].min() == lo[0] and profile[:, 0].max() == hi[0]
        np.testing.assert_allclose(dtm.z_at(profile), profile[:, 2], rtol=0, atol=1e-6)
        # plane frame of a vertical section: u horizontal, v up
        np.testing.assert_allclose(s.u, [-1.0, 0.0, 0.0], atol=1e-15)
        np.testing.assert_allclose(s.v, [0.0, 0.0, 1.0], atol=1e-15)
        uv = s.polylines_2d[0]
        np.testing.assert_allclose(uv[:, 1], profile[:, 2], atol=1e-9)
        assert (np.diff(uv[:, 0]) != 0).all()
        assert np.all(np.diff(uv[:, 0]) > 0) or np.all(np.diff(uv[:, 0]) < 0)
    # horizontal plane: contour lines, open (they run off the border) or closed
    contours = section(dtm.mesh, (0, 0, 0.5 * (lo[2] + hi[2])), (0, 0, 1))
    assert len(contours) >= 1
    for pts in contours.polylines:
        np.testing.assert_allclose(pts[:, 2], 0.5 * (lo[2] + hi[2]), atol=1e-9)
    # a cube with one face removed: the cut through the opening is an open polyline
    c = cube(2.0)
    opened = c.submesh(c.face_normals[:, 2] < 0.5)
    s = section(opened, (0, 0, 0), (0, 1, 0))
    assert len(s) == 1 and s.closed == (False,) and s.length == pytest.approx(6.0)
    s = section(opened, (0, 0, 0), (0, 0, 1))
    assert len(s) == 1 and s.closed == (True,) and s.area == 4.0


# ---------------------------------------------------------------- plane frame


def test_plane_frame_and_transforms():
    rng = np.random.default_rng(3)
    mesh = icosphere(2, radius=3.0, center=FAR)
    for k in range(20):
        normal = rng.normal(size=3) if k else np.array([0.0, 0.0, 2.0])
        u, v, n = plane_frame(normal)
        np.testing.assert_allclose(np.cross(u, v), n, atol=1e-14)
        np.testing.assert_allclose([u @ u, v @ v, n @ n, u @ v, u @ n], [1, 1, 1, 0, 0], atol=1e-14)
        assert u[2] == 0.0 and v[2] >= -1e-15  # u horizontal, v never points down
        origin = FAR + rng.uniform(-1, 1, 3)
        s = section(mesh, origin, normal)
        np.testing.assert_array_equal(s.normal, n)
        pts = s.polylines[0]
        uv = s.polylines_2d[0]
        assert uv.shape == (len(pts), 2)
        np.testing.assert_allclose(s.to_world(uv), pts, rtol=0, atol=1e-8)
        np.testing.assert_array_equal(s.to_plane(pts), uv)
        homogeneous = np.column_stack([pts, np.ones(len(pts))])
        local = homogeneous @ s.to_plane_matrix.T
        np.testing.assert_allclose(local[:, :2], uv, atol=1e-8)
        np.testing.assert_allclose(local[:, 2], 0.0, atol=1e-8)
        np.testing.assert_allclose(s.to_plane_matrix @ s.to_world_matrix, np.eye(4), atol=1e-8)
        # distances are preserved, so lengths agree in 2D and 3D
        ring = np.vstack([uv, uv[:1]])
        assert np.linalg.norm(np.diff(ring, axis=0), axis=1).sum() == pytest.approx(s.length)
    u, v, n = plane_frame((0, 0, 1))
    np.testing.assert_array_equal([u, v, n], np.eye(3))
    u, v, n = plane_frame((0, 1, 0))  # looking north: east is to the left of the viewer
    np.testing.assert_array_equal([u, v], [[-1, 0, 0], [0, 0, 1]])


def test_arguments_and_empty_results():
    mesh = cube()
    with pytest.raises(ValueError, match="normal"):
        section(mesh, (0, 0, 0), (0, 0, 0))
    with pytest.raises(ValueError, match="origin"):
        section(mesh, (0, 0), (0, 0, 1))
    with pytest.raises(ValueError, match="tol"):
        section(mesh, (0, 0, 0), (0, 0, 1), tol=-1.0)
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
    for s in (section(empty, (0, 0, 0), (0, 0, 1)), section(mesh, (0, 0, 9), (0, 0, 1))):
        assert len(s) == 0 and s.polylines == () and s.closed == ()
        assert s.area == 0.0 and s.length == 0.0 and s.polylines_2d == ()
    # the snapping tolerance: a vertex 1e-13 above the plane counts as on it
    s = section(cube(2.0), (0, 0, 1.0 - 1e-13), (0, 0, 1))
    assert len(s) == 0
    s = section(cube(2.0), (0, 0, 1.0 - 1e-13), (0, 0, 1), tol=0.0)
    assert len(s) == 1 and s.area == pytest.approx(4.0)
