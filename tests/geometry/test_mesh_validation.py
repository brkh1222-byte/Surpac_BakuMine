"""P2.T2 acceptance: every defect is detected, clean meshes pass, trimesh agrees."""

import numpy as np
import pytest
import trimesh
from scipy.optimize import linprog

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import cube, icosphere, subdivide, tetrahedron, torus
from bakumine.core.geometry.validation import (
    ValidationReport,
    fix_orientation,
    misoriented_faces,
    triangle_pairs_intersect,
    validate,
)


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


def random_transform(rng, far=False) -> np.ndarray:
    t = np.eye(4)
    t[:3, :3] = random_rotation(rng) * rng.uniform(0.5, 50.0)
    t[:3, 3] = rng.uniform(-100, 100, size=3)
    if far:
        t[:3, 3] += [5.0e6, 4.0e6, 1.0e3]
    return t


def assert_only(report: ValidationReport, **expected_nonempty):
    """Check which defect lists are non-empty; all others must be empty."""
    fields = [
        "boundary_edges",
        "nonmanifold_edges",
        "nonmanifold_vertices",
        "inconsistent_edges",
        "degenerate_faces",
        "duplicate_faces",
        "self_intersections",
    ]
    for name in fields:
        value = getattr(report, name)
        if name in expected_nonempty:
            assert len(value) == expected_nonempty[name], f"{name}: {value}"
        else:
            assert len(value) == 0, f"unexpected {name}: {value}"


# ------------------------------------------------------------- clean meshes

CLEAN = {
    "cube": cube(),
    "tetrahedron": tetrahedron(),
    "sphere": icosphere(3),
    "torus": torus(2.0, 0.5, 32, 16),
    "cube_subdivided": subdivide(subdivide(cube(4.0))),
}


@pytest.mark.parametrize("name", CLEAN)
@pytest.mark.parametrize("placement", ["as_built", "rotated", "rotated_far"])
def test_clean_meshes_pass(name, placement):
    mesh = CLEAN[name]
    rng = np.random.default_rng(11)
    if placement != "as_built":
        mesh = mesh.transformed(random_transform(rng, far=placement == "rotated_far"))
    report = validate(mesh)
    assert_only(report)
    assert report.is_closed and report.is_manifold and report.is_winding_consistent
    assert report.is_outward and report.orientable and report.is_valid
    assert report.has_self_intersections is False
    assert report.n_components == 1
    assert report.issues() == []
    assert report.summary().startswith("valid solid")
    assert mesh.is_closed and mesh.is_winding_consistent
    assert fix_orientation(mesh) is mesh


def test_report_is_structured():
    report = cube().validate()
    assert isinstance(report, ValidationReport)
    d = report.to_dict()
    assert d["is_valid"] is True and d["n_faces"] == 12 and d["n_boundary_edges"] == 0
    assert d["n_self_intersections"] == 0 and d["issues"] == []
    import json

    json.dumps(d)


def test_skipping_self_intersection_is_not_valid():
    report = cube().validate(check_self_intersection=False)
    assert report.self_intersections is None and report.has_self_intersections is None
    assert report.is_closed and not report.is_valid
    assert "self-intersection not checked" in report.issues()
    assert report.to_dict()["n_self_intersections"] is None


def test_empty_mesh_is_not_valid():
    report = validate(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)))
    assert not report.is_closed and not report.is_valid
    assert report.issues() == ["mesh has no faces"]


# -------------------------------------------------- defects injected in a cube


def test_hole_is_detected():
    c = cube()
    holed = c.submesh(np.arange(12) != 5)  # remove face 5 = (0, 5, 4)
    report = validate(holed)
    assert_only(report, boundary_edges=3)
    assert sorted(map(tuple, report.boundary_edges.tolist())) == [(0, 4), (0, 5), (4, 5)]
    assert not report.is_closed and not report.is_valid
    assert report.is_manifold and report.is_winding_consistent
    assert not holed.is_closed
    assert "3 boundary edges (mesh is open)" in report.issues()
    assert report.summary().startswith("INVALID")


def test_flipped_face_is_detected():
    flipped = cube().flipped([7])
    report = validate(flipped)
    assert_only(report, inconsistent_edges=3)
    face = set(flipped.faces[7].tolist())
    for edge in report.inconsistent_edges.tolist():
        assert set(edge) <= face
    assert report.misoriented_faces.tolist() == [7]
    assert report.is_closed and not report.is_winding_consistent
    assert not report.is_outward and not report.is_valid
    assert not flipped.is_winding_consistent


def test_inverted_cube_is_consistent_but_not_outward():
    report = validate(cube().flipped())
    assert_only(report)
    assert report.is_winding_consistent and report.is_valid and not report.is_outward
    assert report.misoriented_faces.tolist() == list(range(12))
    assert "12 faces wound inward (negative volume)" in report.issues()


@pytest.mark.parametrize("far", [False, True])
def test_degenerate_face_is_detected(far):
    c = cube(10.0, center=(5.0e6, 4.0e6, 300.0)) if far else cube()
    # split edge (0, 1) of face 4 = (0, 1, 5) at its midpoint: the sliver
    # (0, 8, 1) has zero area and the mesh stays closed
    mid = 0.5 * (c.vertices[0] + c.vertices[1])
    faces = np.vstack([c.faces[np.arange(12) != 4], [[0, 8, 5], [8, 1, 5], [0, 1, 8]]])
    report = validate(Mesh(np.vstack([c.vertices, mid]), faces))
    assert report.degenerate_faces.tolist() == [13]
    assert report.has_degenerate_faces and not report.is_valid
    assert report.is_closed


def test_repeated_vertex_face_is_degenerate():
    c = cube()
    report = validate(Mesh(c.vertices, np.vstack([c.faces, [[2, 2, 6]]])))
    assert report.degenerate_faces.tolist() == [12]
    assert not report.is_valid


def test_duplicate_face_is_detected():
    c = cube()
    # same triangle again, once as is and once rotated + reversed
    faces = np.vstack([c.faces, c.faces[[3]], c.faces[[9]][:, [2, 1, 0]]])
    report = validate(Mesh(c.vertices, faces))
    assert report.duplicate_faces.tolist() == [12, 13]
    assert report.has_duplicate_faces and not report.is_valid
    assert not report.is_closed  # the doubled faces' edges now have three faces
    assert len(report.nonmanifold_edges) == 6
    assert len(report.self_intersections) == 0  # reported as duplicates, not crossings


def test_nonmanifold_edge_is_detected():
    c = cube()
    fin = Mesh(np.vstack([c.vertices, [[0.0, 0.0, 2.0]]]), np.vstack([c.faces, [[4, 5, 8]]]))
    report = validate(fin)
    assert_only(report, nonmanifold_edges=1, boundary_edges=2)
    assert report.nonmanifold_edges.tolist() == [[4, 5]]
    assert not report.is_manifold and not report.is_closed and not report.is_valid


def test_nonmanifold_vertex_is_detected():
    a = cube()  # corners at +-0.5
    b = cube().translated([1.0, 1.0, 1.0])
    joined = Mesh.concatenate([a, b]).weld()  # the cubes share exactly one corner
    assert joined.n_vertices == 15
    report = validate(joined)
    assert_only(report, nonmanifold_vertices=1)
    np.testing.assert_array_equal(joined.vertices[report.nonmanifold_vertices], [[0.5, 0.5, 0.5]])
    assert report.is_closed  # every edge still has two faces
    assert not report.is_manifold and not report.is_valid


def test_interpenetrating_cubes_self_intersect():
    a = cube()
    b = cube().translated([0.3, 0.2, 0.4])
    merged = Mesh.concatenate([a, b])
    report = validate(merged)
    assert_only(report, self_intersections=len(report.self_intersections))
    assert report.has_self_intersections and not report.is_valid
    assert report.is_closed and report.is_winding_consistent  # topology alone looks fine
    pairs = report.self_intersections
    assert len(pairs) > 0
    assert (pairs[:, 0] < 12).all() and (pairs[:, 1] >= 12).all()  # always one face of each
    # the three faces of a around its (+,+,+) corner are cut, and the three of b around (-,-,-)
    assert set(pairs[:, 0].tolist()) == {2, 3, 6, 7, 8, 9}
    assert set(pairs[:, 1].tolist()) == {12, 13, 16, 17, 22, 23}
    # exactly the pairs an independent linear-programming test finds
    tri = merged.triangles
    expected = [
        [i, j] for i in range(12) for j in range(12, 24) if triangles_intersect_lp(tri[i], tri[j])
    ]
    assert pairs.tolist() == expected
    # the same after an arbitrary rigid motion far from the origin
    moved = merged.transformed(random_transform(np.random.default_rng(5), far=True))
    np.testing.assert_array_equal(validate(moved).self_intersections, pairs)


def test_disjoint_and_nested_shells_do_not_self_intersect():
    apart = Mesh.concatenate([cube(), cube().translated([3.0, 0.0, 0.0])])
    nested = Mesh.concatenate([cube(4.0), icosphere(2, 1.0)])
    for mesh in (apart, nested):
        report = validate(mesh)
        assert_only(report)
        assert report.is_valid and report.n_components == 2


def test_touching_shells_self_intersect():
    # two unwelded cubes sharing a whole face: coplanar overlapping triangles
    stacked = Mesh.concatenate([cube(), cube().translated([0.0, 0.0, 1.0])])
    report = validate(stacked)
    assert report.has_self_intersections
    top, bottom_of_upper = {2, 3}, {12, 13}
    assert any(i in top and j in bottom_of_upper for i, j in report.self_intersections.tolist())
    # a tetrahedron standing with one vertex on the middle of the cube's top face
    spike = tetrahedron().transformed(np.diag([1.0, 1.0, -1.0])).translated([0.1, 0.2, 1.5])
    assert spike.vertices[3].tolist() == [0.1, 0.2, 0.5]
    report = validate(Mesh.concatenate([cube(), spike]))
    assert report.has_self_intersections


def test_folded_fan_self_intersects():
    # two triangles on the same side of their shared edge, in the same plane
    fold = Mesh([[0, 0, 0], [1, 0, 0], [0.3, 1, 0], [0.6, 0.5, 0]], [[0, 1, 2], [1, 0, 3]])
    assert validate(fold).self_intersections.tolist() == [[0, 1]]
    flat = Mesh([[0, 0, 0], [1, 0, 0], [0.3, 1, 0], [0.6, -0.5, 0]], [[0, 1, 2], [1, 0, 3]])
    assert len(validate(flat).self_intersections) == 0
    # two coplanar triangles sharing one vertex: overlapping, then merely touching there
    v = [[0, 0, 0], [2, 0, 0], [0, 2, 0], [1, 0.5, 0], [0.5, 1, 0], [-1, -0.5, 0], [-0.5, -1, 0]]
    assert validate(Mesh(v, [[0, 1, 2], [0, 3, 4]])).self_intersections.tolist() == [[0, 1]]
    assert len(validate(Mesh(v, [[0, 1, 2], [0, 5, 6]])).self_intersections) == 0


def triangles_intersect_lp(a: np.ndarray, b: np.ndarray) -> bool:
    """Independent oracle: do two triangles share a point? (convex feasibility LP)."""
    a_eq = np.zeros((5, 6))
    a_eq[:3, :3], a_eq[:3, 3:] = a.T, -b.T
    a_eq[3, :3] = a_eq[4, 3:] = 1.0
    res = linprog(np.zeros(6), A_eq=a_eq, b_eq=[0, 0, 0, 1, 1], bounds=(0, None), method="highs")
    return res.status == 0


def test_triangle_pair_test_matches_linear_programming_oracle():
    rng = np.random.default_rng(42)
    n = 300
    tris = rng.uniform(-1, 1, size=(2 * n, 3, 3))
    tris[n:] *= rng.uniform(0.2, 1.5, size=(n, 1, 1))
    mesh = Mesh(tris.reshape(-1, 3), np.arange(6 * n).reshape(-1, 3))
    pairs = np.column_stack([np.arange(n), np.arange(n) + n])
    got = triangle_pairs_intersect(mesh, pairs)
    expected = np.array([triangles_intersect_lp(tris[i], tris[j]) for i, j in pairs])
    np.testing.assert_array_equal(got, expected)
    assert 30 < expected.sum() < n - 30  # both outcomes are well represented


# ---------------------------------------------------------------- orientation


def test_fix_orientation_restores_random_flips():
    rng = np.random.default_rng(3)
    for mesh in (cube(), icosphere(2), torus(2.0, 0.5, 16, 8), subdivide(cube())):
        mesh = mesh.transformed(random_transform(rng, far=True))
        damaged = mesh.flipped(rng.choice(mesh.n_faces, size=mesh.n_faces // 3, replace=False))
        assert not validate(damaged).is_winding_consistent
        fixed = damaged.fix_orientation()
        np.testing.assert_array_equal(fixed.faces, mesh.faces)
        np.testing.assert_array_equal(fixed.vertices, mesh.vertices)
        assert validate(fixed).is_outward
        # the faces reported as misoriented are exactly the ones we flipped
        changed = np.flatnonzero((damaged.faces != mesh.faces).any(axis=1))
        np.testing.assert_array_equal(misoriented_faces(damaged), changed)


def test_fix_orientation_turns_inverted_shells_outward_one_by_one():
    outer, inner = cube(4.0), icosphere(2, 1.0).flipped()
    both = Mesh.concatenate([outer.flipped(), inner, cube().translated([10.0, 0, 0])])
    fixed = both.fix_orientation()
    report = validate(fixed)
    assert report.is_outward and report.n_components == 3
    oracle = trimesh.Trimesh(fixed.vertices, fixed.faces, process=False)
    for part in oracle.split(only_watertight=True):
        assert part.volume > 0


def test_fix_orientation_open_surface_points_up():
    # a 2 x 2 grid of quads on z = x/2, wound at random
    xs, ys = np.meshgrid(np.arange(3.0), np.arange(3.0), indexing="ij")
    v = np.column_stack([xs.ravel(), ys.ravel(), 0.5 * xs.ravel()])
    faces = []
    for i in range(2):
        for j in range(2):
            a, b, c, d = i * 3 + j, (i + 1) * 3 + j, (i + 1) * 3 + j + 1, i * 3 + j + 1
            faces += [(a, b, c), (a, c, d)]
    up = Mesh(v, faces)
    assert (up.face_normals[:, 2] > 0).all()
    for damaged in (up.flipped(), up.flipped([0, 3, 6])):
        fixed = damaged.fix_orientation()
        assert (fixed.face_normals[:, 2] > 0).all()
        assert validate(fixed).is_winding_consistent and not validate(fixed).is_closed


def test_moebius_band_is_not_orientable():
    n = 12
    u = np.arange(n) * (2 * np.pi / n)
    rows = []
    for s in (-0.3, 0.3):
        r = 1.0 + s * np.cos(u / 2)
        rows.append(np.column_stack([r * np.cos(u), r * np.sin(u), s * np.sin(u / 2)]))
    v = np.vstack(rows)  # vertex i on one rim, n + i on the other
    faces = []
    for i in range(n):
        a, b = i, n + i
        if i + 1 < n:
            c, d = i + 1, n + i + 1
        else:
            c, d = n, 0  # the half twist swaps the rims at the seam
        faces += [(a, c, b), (b, c, d)]
    band = Mesh(v, faces)
    report = validate(band)
    assert not report.orientable
    assert not validate(band.fix_orientation()).is_winding_consistent
    assert "surface is not orientable" in report.issues()


# -------------------------------------------------------------- trimesh oracle


def random_mesh(rng, i):
    builders = [
        lambda: cube(),
        lambda: subdivide(cube()),
        lambda: tetrahedron(),
        lambda: icosphere(int(rng.integers(0, 3))),
        lambda: torus(2.0, 0.6, int(rng.integers(5, 14)), int(rng.integers(4, 9))),
    ]
    mesh = builders[i % 5]().transformed(random_transform(rng, far=bool(rng.integers(0, 2))))
    defect = ("none", "remove", "flip", "both")[i % 4]
    if defect in ("flip", "both"):
        k = int(rng.integers(1, max(2, mesh.n_faces // 4)))
        mesh = mesh.flipped(rng.choice(mesh.n_faces, size=k, replace=False))
    if defect in ("remove", "both"):
        k = int(rng.integers(1, max(2, mesh.n_faces // 4)))
        keep = np.ones(mesh.n_faces, dtype=bool)
        keep[rng.choice(mesh.n_faces, size=k, replace=False)] = False
        mesh = mesh.submesh(keep)
    return mesh, defect


def test_closed_and_winding_agree_with_trimesh_on_20_random_meshes():
    rng = np.random.default_rng(2024)
    seen = set()
    for i in range(20):
        mesh, defect = random_mesh(rng, i)
        oracle = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        report = validate(mesh)
        context = f"mesh {i} ({defect}, {mesh.n_faces} faces)"
        assert report.is_closed == bool(oracle.is_watertight), context
        assert report.is_winding_consistent == bool(oracle.is_winding_consistent), context
        assert mesh.is_closed == report.is_closed, context
        assert mesh.is_winding_consistent == report.is_winding_consistent, context
        seen.add((report.is_closed, report.is_winding_consistent))
        # after the fix trimesh agrees the winding is consistent, and outward if closed
        fixed = trimesh.Trimesh(mesh.vertices, mesh.fix_orientation().faces, process=False)
        assert fixed.is_winding_consistent, context
        if fixed.is_watertight:
            assert fixed.volume > 0, context
    assert seen == {(True, True), (True, False), (False, True), (False, False)}


def test_cavities_are_inward_shells_inside_the_solid():
    """An inward shell inside the solid is a hole in it, not a winding defect."""
    outer, inner = cube(4.0), icosphere(2, 1.0)
    hollow = Mesh.concatenate([outer, inner.flipped()])
    report = validate(hollow)
    assert report.is_valid and report.is_oriented and not report.is_outward
    assert report.issues() == [] and report.summary().startswith("valid solid")
    np.testing.assert_array_equal(
        report.cavity_faces, np.arange(outer.n_faces, outer.n_faces + inner.n_faces)
    )
    np.testing.assert_array_equal(report.cavity_faces, report.misoriented_faces)
    assert len(report.wrongly_wound_faces) == 0
    assert report.to_dict()["n_cavity_faces"] == inner.n_faces and report.to_dict()["is_oriented"]
    assert hollow.volume == pytest.approx(outer.volume - inner.volume)
    # the same inward shell outside the cube encloses nothing: wrongly wound
    apart = Mesh.concatenate([outer, inner.flipped().translated([10.0, 0.0, 0.0])])
    report = validate(apart)
    assert len(report.cavity_faces) == 0 and not report.is_oriented
    assert report.issues() == [f"{inner.n_faces} faces wound inward (negative volume)"]
    # below the cube, where a ray going up crosses the cube twice: still outside
    below = Mesh.concatenate([outer, inner.flipped().translated([0.0, 0.0, -10.0])])
    assert len(validate(below).cavity_faces) == 0
    # an island inside a cavity is outward again; an inward island is wrongly wound
    island = icosphere(1, 0.3)
    nested = Mesh.concatenate([outer, inner.flipped(), island])
    assert validate(nested).is_oriented and validate(nested).n_components == 3
    wrong = validate(Mesh.concatenate([outer, inner.flipped(), island.flipped()]))
    assert not wrong.is_oriented and len(wrong.wrongly_wound_faces) == island.n_faces
    # a wholly inward solid is not a cavity, after any rigid motion either
    assert len(validate(outer.flipped()).cavity_faces) == 0
    moved = hollow.transformed(random_transform(np.random.default_rng(3), far=True))
    assert validate(moved).is_oriented and len(validate(moved).cavity_faces) == inner.n_faces
