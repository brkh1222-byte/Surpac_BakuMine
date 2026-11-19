"""Ore solid: valid closed mesh, known volume, implicit function consistent with the mesh."""

from __future__ import annotations

import json

import numpy as np

from bakumine.synth.solid import axes_from_angles, icosphere


def _signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """research.md 5.5, written independently of the package: (1/6) sum v1 . (v2 x v3)."""
    v = vertices - vertices[0]
    total = 0.0
    for a, b, c in faces:
        total += float(np.dot(v[a], np.cross(v[b], v[c])))
    return total / 6.0


def _inside_by_ray(points: np.ndarray, vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Independent point-in-solid test: count crossings of a +z ray (odd = inside)."""
    a, b, c = (vertices[faces[:, i]] for i in range(3))
    out = np.zeros(len(points), dtype=bool)
    for n, p in enumerate(points):
        px, py = p[0] - c[:, 0], p[1] - c[:, 1]
        d = (b[:, 1] - c[:, 1]) * (a[:, 0] - c[:, 0]) + (c[:, 0] - b[:, 0]) * (a[:, 1] - c[:, 1])
        with np.errstate(divide="ignore", invalid="ignore"):
            l1 = ((b[:, 1] - c[:, 1]) * px + (c[:, 0] - b[:, 0]) * py) / d
            l2 = ((c[:, 1] - a[:, 1]) * px + (a[:, 0] - c[:, 0]) * py) / d
        l3 = 1.0 - l1 - l2
        hit = (l1 >= 0) & (l2 >= 0) & (l3 >= 0) & (d != 0)
        z = l1 * a[:, 2] + l2 * b[:, 2] + l3 * c[:, 2]
        out[n] = np.count_nonzero(hit & (z > p[2])) % 2 == 1
    return out


def test_mesh_is_closed_and_consistently_oriented(dataset):
    faces = dataset.npy("ore_solid_faces")
    vertices = dataset.npy("ore_solid_vertices")
    assert faces.min() == 0 and faces.max() == len(vertices) - 1
    assert len(np.unique(faces)) == len(vertices)  # no unused vertices
    directed = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    assert np.all(directed[:, 0] != directed[:, 1])  # no degenerate triangles
    # Consistent orientation: every directed edge occurs once and its reverse occurs once.
    assert len(np.unique(directed, axis=0)) == len(directed)
    forward = {tuple(e) for e in directed.tolist()}
    assert all((b, a) in forward for a, b in forward)
    # Closed: every undirected edge is shared by exactly two faces.
    _, counts = np.unique(np.sort(directed, axis=1), axis=0, return_counts=True)
    assert np.all(counts == 2)
    # Euler characteristic of a sphere.
    assert len(vertices) - len(counts) + len(faces) == 2
    areas = np.linalg.norm(
        np.cross(
            vertices[faces[:, 1]] - vertices[faces[:, 0]],
            vertices[faces[:, 2]] - vertices[faces[:, 0]],
        ),
        axis=1,
    )
    assert areas.min() > 1e-6


def test_mesh_volume_matches_recorded_true_volume(dataset):
    truth = json.loads((dataset.path / "truth.json").read_text())["solid"]
    volume = _signed_volume(dataset.npy("ore_solid_vertices"), dataset.npy("ore_solid_faces"))
    assert volume > 0  # outward normals
    bound = truth["mesh"]["relative_error_bound"]
    assert bound <= 0.01
    assert abs(volume / truth["volume_m3"] - 1.0) <= bound
    assert np.isclose(volume, truth["mesh"]["volume_m3"], rtol=1e-9)


def test_true_volume_against_independent_integrations(dataset):
    solid = dataset.deposit.solid
    recorded = dataset.deposit.truth["solid"]["volume_m3"]
    # (1) The 1-D reduction by a plain trapezoid rule instead of adaptive quadrature.
    u = np.linspace(-solid.a, solid.a, 400_001)
    t = 1.0 - solid.pinch_depth * np.exp(-(((u - solid.pinch_position) / solid.pinch_width) ** 2))
    trapezoid = np.pi * solid.b * solid.c * np.trapezoid(t * (1.0 - (u / solid.a) ** 2), u)
    assert np.isclose(trapezoid, recorded, rtol=1e-8)
    # (2) A voxel count of inside() on a world-aligned grid: no use of the reduction at all.
    step = 2.0
    vertices = dataset.deposit.solid_vertices
    lo, hi = vertices.min(axis=0) - 2 * step, vertices.max(axis=0) + 2 * step
    x, y, z = (np.arange(lo[i] + 0.5 * step, hi[i], step) for i in range(3))
    xy = np.stack(np.meshgrid(x, y, indexing="ij"), axis=-1).reshape(-1, 2)
    count = 0
    for zk in z:
        count += int(solid.inside(np.column_stack([xy, np.full(len(xy), zk)])).sum())
    assert abs(count * step**3 / recorded - 1.0) < 0.005
    # (3) The generator's own voxel check is recorded with its resolution.
    assert abs(dataset.deposit.truth["solid"]["voxel_check"]["relative_error"]) < 0.005


def test_vertices_lie_on_the_implicit_surface(dataset):
    f = dataset.deposit.solid.implicit(dataset.npy("ore_solid_vertices"))
    assert np.abs(f).max() < 1e-9


def test_implicit_inside_agrees_with_ray_casting_on_the_mesh(dataset):
    solid = dataset.deposit.solid
    vertices, faces = dataset.deposit.solid_vertices, dataset.deposit.solid_faces
    rng = np.random.default_rng(0)
    local = rng.uniform(-1.05, 1.05, (400, 3)) * [solid.a, solid.b, 1.6 * solid.c]
    local[:, 2] += solid.bend(local[:, 0], local[:, 1])
    points = solid.to_world(local)
    implicit = solid.inside(points)
    assert 0.2 < implicit.mean() < 0.8  # the sample really tests both sides
    by_ray = _inside_by_ray(points, vertices, faces)
    differ = implicit != by_ray
    assert differ.mean() < 0.02
    # Any disagreement sits in the thin gap between the surface and its chordal mesh.
    assert np.all(np.abs(solid.implicit(points[differ])) < 0.1)


def test_solid_is_not_convex_and_not_axis_aligned(dataset):
    solid = dataset.deposit.solid
    vertices = dataset.deposit.solid_vertices
    rng = np.random.default_rng(1)
    i, j = rng.integers(0, len(vertices), (2, 4000))
    mid = 0.5 * (vertices[i] + vertices[j])
    assert (solid.implicit(mid) > 1e-3).mean() > 0.05  # chords that leave the solid
    # The pinch: the lens is much thinner at pinch_position than on either side.
    assert solid.thickness_factor(np.array([solid.pinch_position]))[0] < 0.5
    # No lens axis is within 10 degrees of a coordinate axis.
    assert np.abs(solid.axes).max() < np.cos(np.radians(10.0))


def test_axes_from_angles_is_a_rotation_with_the_documented_meaning():
    axes = axes_from_angles(30.0, 15.0, 55.0)
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    assert np.isclose(np.linalg.det(axes), 1.0)
    major, semi, minor = axes.T
    assert np.isclose(np.degrees(np.arctan2(major[0], major[1])), 30.0)  # bearing
    assert np.isclose(np.degrees(np.arcsin(-major[2])), 15.0)  # plunge, downward
    assert semi[2] < 0 and minor[2] > 0  # semi-major dips down; minor points up
    flat = axes_from_angles(0.0, 0.0, 0.0)
    assert np.allclose(flat, [[0, -1, 0], [1, 0, 0], [0, 0, 1]], atol=1e-12)


def test_icosphere_counts():
    vertices, faces = icosphere(3)
    assert len(faces) == 20 * 4**3 and len(vertices) == 10 * 4**3 + 2
    assert np.allclose(np.linalg.norm(vertices, axis=1), 1.0)
