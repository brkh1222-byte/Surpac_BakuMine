"""P2.T3 acceptance: volume, area, centroid vs analytic values and trimesh."""

import math

import numpy as np
import pytest
import trimesh

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, cube, icosphere, subdivide, tetrahedron, torus

FAR = np.array([5.0e6, 4.0e6, 1.0e3])


def random_rotation(rng) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] *= -1
    return q


# ------------------------------------------------------------------- analytic


def test_cube_exact():
    c = cube(2.0, center=(1.0, -3.0, 0.5))
    assert c.signed_volume == 8.0
    assert c.volume == 8.0
    assert c.area == 24.0
    np.testing.assert_array_equal(c.centroid, [1.0, -3.0, 0.5])
    np.testing.assert_array_equal(c.area_centroid, [1.0, -3.0, 0.5])
    assert c.flipped().signed_volume == -8.0
    assert c.flipped().volume == 8.0
    np.testing.assert_array_equal(c.flipped().centroid, [1.0, -3.0, 0.5])


def test_box_and_cube_exact_far_from_origin():
    b = box(FAR + [0.0, 0.0, 0.0], FAR + [20.0, 10.0, 4.0])
    assert b.volume == 800.0
    assert b.area == 2 * (200.0 + 80.0 + 40.0)
    np.testing.assert_array_equal(b.centroid, FAR + [10.0, 5.0, 2.0])
    c = subdivide(subdivide(cube(8.0, center=FAR)))
    assert c.volume == 512.0 and c.area == 384.0
    np.testing.assert_array_equal(c.centroid, FAR)


def test_tetrahedron_exact():
    t = tetrahedron(3.0)
    assert t.signed_volume == pytest.approx(4.5, rel=1e-15)
    assert t.area == pytest.approx(3 * 4.5 + 0.5 * math.sqrt(3) * 9.0, rel=1e-15)
    np.testing.assert_allclose(t.centroid, [0.75, 0.75, 0.75], rtol=1e-15)


def test_sphere_within_tessellation_error_and_converging():
    radius, centre = 3.0, np.array([10.0, -20.0, 5.0])
    exact_volume = 4.0 / 3.0 * math.pi * radius**3
    exact_area = 4.0 * math.pi * radius**2
    volume_errors, area_errors = [], []
    for level in range(1, 6):
        s = icosphere(level, radius, centre)
        # the mesh is inscribed: it can only be smaller than the sphere
        assert s.volume < exact_volume and s.area < exact_area
        volume_errors.append(1.0 - s.volume / exact_volume)
        area_errors.append(1.0 - s.area / exact_area)
        np.testing.assert_allclose(s.centroid, centre, atol=1e-12)
        # chord error bound: every face is within h^2 / (2 r) of the sphere, h = longest edge
        e = s.vertices[s.edges[:, 0]] - s.vertices[s.edges[:, 1]]
        sagitta = (np.linalg.norm(e, axis=1).max() ** 2) / (2.0 * radius)
        assert volume_errors[-1] < 3.0 * sagitta / radius
    # second order: each subdivision halves the edge and quarters the error
    for errors in (volume_errors, area_errors):
        ratios = np.array(errors[:-1]) / np.array(errors[1:])
        assert (ratios > 3.7).all() and (ratios < 4.3).all()
    assert volume_errors[-1] < 1e-3 and area_errors[-1] < 1e-3


def test_torus_exact_for_its_polygonal_shape_and_converging():
    big_r, small_r = 5.0, 1.5
    exact_volume = 2.0 * math.pi**2 * big_r * small_r**2
    exact_area = 4.0 * math.pi**2 * big_r * small_r
    errors = []
    for n in (16, 32, 64, 128):
        t = torus(big_r, small_r, 2 * n, n)
        errors.append((abs(t.volume / exact_volume - 1.0), abs(t.area / exact_area - 1.0)))
        np.testing.assert_allclose(t.centroid, [0.0, 0.0, 0.0], atol=1e-12)
        assert t.signed_volume > 0
    errors = np.array(errors)
    ratios = errors[:-1] / errors[1:]
    assert (ratios > 3.5).all() and (ratios < 4.5).all()
    assert (errors[-1] < 1e-3).all()
    # Closed form for the faceted torus itself. With ring vertices on circles,
    # Pappus on an inscribed regular n-gon tube swept along an inscribed
    # regular m-gon path is not exact, so compare against trimesh instead.
    t = torus(big_r, small_r, 40, 20)
    oracle = trimesh.Trimesh(t.vertices, t.faces, process=False)
    assert t.volume == pytest.approx(oracle.volume, rel=1e-12)
    assert t.area == pytest.approx(oracle.area, rel=1e-12)


def test_translation_and_rotation_invariance():
    rng = np.random.default_rng(8)
    base = torus(4.0, 1.0, 40, 20).translated([0.5, 0.25, -0.125])
    rot = random_rotation(rng)
    moved = base.transformed(rot).translated(FAR)
    # storing the vertices at 5e6 rounds them by ~1e-9 m, which changes the solid slightly
    assert moved.volume == pytest.approx(base.volume, rel=1e-9)
    assert moved.area == pytest.approx(base.area, rel=1e-9)
    np.testing.assert_allclose(moved.centroid, base.centroid @ rot.T + FAR, rtol=0, atol=1e-8)
    np.testing.assert_allclose(moved.area_centroid, base.area_centroid @ rot.T + FAR, atol=1e-8)


def test_scaling_laws():
    base = icosphere(2)
    scaled = base.transformed(np.diag([2.0, 3.0, 5.0]))
    assert scaled.volume == pytest.approx(30.0 * base.volume, rel=1e-14)
    mirrored = base.transformed(np.diag([1.0, 1.0, -1.0]))  # winding is kept outward
    assert mirrored.signed_volume == pytest.approx(base.volume, rel=1e-14)


def test_multiple_shells_and_cavity():
    a, b = cube(2.0), cube(1.0, center=(10.0, 0.0, 0.0))
    both = Mesh.concatenate([a, b])
    assert both.volume == 9.0 and both.area == 30.0
    np.testing.assert_allclose(both.centroid, [10.0 / 9.0, 0.0, 0.0], rtol=1e-15)
    # an inward-wound inner shell is a cavity
    hollow = Mesh.concatenate([cube(2.0), cube(1.0, center=(0.25, 0.0, 0.0)).flipped()])
    assert hollow.signed_volume == 7.0
    np.testing.assert_allclose(hollow.centroid, [-0.25 / 7.0, 0.0, 0.0], atol=1e-15)


def test_open_and_empty_meshes():
    sheet = Mesh([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0.0]], [[0, 1, 2], [0, 2, 3]])
    assert sheet.area == 4.0 and sheet.volume == 0.0
    np.testing.assert_array_equal(sheet.area_centroid, [1.0, 1.0, 0.0])
    with pytest.raises(ValueError):
        _ = sheet.centroid
    empty = Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int))
    assert empty.area == 0.0 and empty.volume == 0.0
    with pytest.raises(ValueError):
        _ = empty.centroid
    with pytest.raises(ValueError):
        _ = empty.area_centroid


# -------------------------------------------------------------- trimesh oracle


def random_solid(rng, i, min_log_size=-1.0) -> Mesh:
    builders = [
        lambda: cube(),
        lambda: tetrahedron(),
        lambda: icosphere(int(rng.integers(0, 4))),
        lambda: torus(2.0, 0.6, int(rng.integers(6, 40)), int(rng.integers(4, 20))),
        lambda: subdivide(box([0, 0, 0], rng.uniform(0.5, 3.0, size=3))),
    ]
    mesh = builders[i % 5]()
    # rotation, anisotropic scale and shear: any invertible linear map keeps it closed
    linear = random_rotation(rng) @ np.diag(rng.uniform(0.5, 2.0, size=3)) @ random_rotation(rng)
    linear *= 10.0 ** rng.uniform(min_log_size, 3)  # sizes from 0.1 m (default) to 1 km
    return mesh.transformed(linear)


@pytest.mark.parametrize("far", [False, True], ids=["near_origin", "utm_5e6"])
def test_volume_area_centroid_equal_trimesh(far):
    # trimesh integrates in absolute coordinates, so at 5e6 m its own result is
    # only good to about 1e-16 * 5e6 / size. Solids of 10 m and up keep the
    # oracle inside the 1e-9 budget; smaller ones far from the origin are
    # covered by test_far_from_origin_is_as_accurate_as_near.
    rng = np.random.default_rng(99)
    for i in range(30):
        mesh = random_solid(rng, i, min_log_size=1.0 if far else -1.0)
        offset = rng.uniform(-50, 50, size=3)
        if far:
            offset = offset + rng.choice([-1.0, 1.0], size=3) * FAR * rng.uniform(0.8, 1.2, size=3)
        mesh = mesh.translated(offset)
        oracle = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        assert oracle.is_watertight and oracle.is_winding_consistent
        context = f"mesh {i}, {mesh.n_faces} faces"
        assert mesh.signed_volume == pytest.approx(oracle.volume, rel=1e-9), context
        assert mesh.area == pytest.approx(oracle.area, rel=1e-9), context
        # 1e-9 relative to the centroid's own magnitude
        scale = np.linalg.norm(oracle.center_mass)
        error = np.linalg.norm(mesh.centroid - oracle.center_mass)
        assert error <= 1e-9 * scale, context


def test_far_from_origin_is_as_accurate_as_near():
    """Tighter than trimesh can check: the same solid, bit for bit, near and far."""
    rng = np.random.default_rng(5)
    shift = np.array([5.0e6, 4.0e6, 1.0e3])
    for i in range(20):
        base = random_solid(rng, i)
        # coordinates on a 2**-20 m lattice stay exactly representable at 5e6 m
        near = Mesh(np.round(base.vertices * 2.0**20) / 2.0**20, base.faces)
        far = Mesh(near.vertices + shift, near.faces)
        np.testing.assert_array_equal(far.vertices - shift, near.vertices)
        assert far.signed_volume == pytest.approx(near.signed_volume, rel=1e-12)
        assert far.area == pytest.approx(near.area, rel=1e-12)
        # the centroid itself can only be stored to about 1e-9 m at 5e6 m
        np.testing.assert_allclose(far.centroid - shift, near.centroid, rtol=0, atol=3e-9)
        np.testing.assert_allclose(far.area_centroid - shift, near.area_centroid, rtol=0, atol=3e-9)


def test_large_mesh_measures():
    s = icosphere(7, 250.0, center=FAR)  # 327,680 faces
    exact_volume = 4.0 / 3.0 * math.pi * 250.0**3
    assert s.volume == pytest.approx(exact_volume, rel=1e-3)
    assert s.volume < exact_volume
    assert s.area == pytest.approx(4.0 * math.pi * 250.0**2, rel=1e-3)
    np.testing.assert_allclose(s.centroid, FAR, atol=1e-6)
