"""Small closed primitive meshes, mainly for tests and benchmarks.

All primitives are closed, manifold and wound counter-clockwise seen from
outside (positive signed volume).
"""

from __future__ import annotations

import math

import numpy as np

from bakumine.core.geometry.mesh import Mesh

__all__ = ["box", "cube", "icosphere", "subdivide", "tetrahedron", "torus"]

_BOX_FACES = np.array(
    [
        [0, 2, 1], [0, 3, 2],  # bottom (z = min), normal -z
        [4, 5, 6], [4, 6, 7],  # top (z = max), normal +z
        [0, 1, 5], [0, 5, 4],  # front (y = min), normal -y
        [2, 3, 7], [2, 7, 6],  # back (y = max), normal +y
        [1, 2, 6], [1, 6, 5],  # right (x = max), normal +x
        [3, 0, 4], [3, 4, 7],  # left (x = min), normal -x
    ],
    dtype=np.int64,
)  # fmt: skip


def box(lo, hi) -> Mesh:
    """Axis-aligned box from corner ``lo`` to corner ``hi``: 8 vertices, 12 faces."""
    (x0, y0, z0), (x1, y1, z1) = np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)
    vertices = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )  # fmt: skip
    return Mesh(vertices, _BOX_FACES)


def cube(size: float = 1.0, center=(0.0, 0.0, 0.0)) -> Mesh:
    """Cube with edge length ``size`` centred on ``center``."""
    c = np.asarray(center, dtype=np.float64)
    return box(c - 0.5 * size, c + 0.5 * size)


def tetrahedron(size: float = 1.0) -> Mesh:
    """Right-angled tetrahedron with corners at the origin and ``size`` along each axis.

    Volume is ``size**3 / 6``.
    """
    vertices = size * np.array([[0.0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    faces = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]], dtype=np.int64)
    return Mesh(vertices, faces)


def subdivide(mesh: Mesh) -> Mesh:
    """Split every triangle into four at its edge midpoints (shape unchanged)."""
    em = mesh.edge_map
    n = mesh.n_vertices
    midpoints = 0.5 * (mesh.vertices[em.edges[:, 0]] + mesh.vertices[em.edges[:, 1]])
    f = mesh.faces
    ab, bc, ca = (em.face_edges[:, j] + n for j in range(3))
    a, b, c = f[:, 0], f[:, 1], f[:, 2]
    faces = np.concatenate(
        [
            np.column_stack([a, ab, ca]),
            np.column_stack([b, bc, ab]),
            np.column_stack([c, ca, bc]),
            np.column_stack([ab, bc, ca]),
        ]
    )
    return Mesh(np.concatenate([mesh.vertices, midpoints]), faces, check=False)


def icosphere(subdivisions: int = 2, radius: float = 1.0, center=(0.0, 0.0, 0.0)) -> Mesh:
    """Sphere from a subdivided icosahedron: ``20 * 4**subdivisions`` faces.

    All vertices lie exactly on the sphere, so the mesh is inscribed and its
    volume approaches ``4/3 pi r^3`` from below.
    """
    t = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, t, 0], [1, t, 0], [-1, -t, 0], [1, -t, 0],
            [0, -1, t], [0, 1, t], [0, -1, -t], [0, 1, -t],
            [t, 0, -1], [t, 0, 1], [-t, 0, -1], [-t, 0, 1],
        ],
        dtype=np.float64,
    )  # fmt: skip
    faces = np.array(
        [
            [0, 11, 5], [0, 5, 1], [0, 1, 7], [0, 7, 10], [0, 10, 11],
            [1, 5, 9], [5, 11, 4], [11, 10, 2], [10, 7, 6], [7, 1, 8],
            [3, 9, 4], [3, 4, 2], [3, 2, 6], [3, 6, 8], [3, 8, 9],
            [4, 9, 5], [2, 4, 11], [6, 2, 10], [8, 6, 7], [9, 8, 1],
        ],
        dtype=np.int64,
    )  # fmt: skip
    mesh = Mesh(vertices / np.linalg.norm(vertices[0]), faces)
    for _ in range(subdivisions):
        mesh = subdivide(mesh)
        v = mesh.vertices
        mesh = Mesh(v / np.linalg.norm(v, axis=1, keepdims=True), mesh.faces, check=False)
    return Mesh(mesh.vertices * radius + np.asarray(center, dtype=np.float64), mesh.faces)


def torus(
    major_radius: float = 2.0,
    minor_radius: float = 0.5,
    major_segments: int = 48,
    minor_segments: int = 24,
) -> Mesh:
    """Torus around the z axis: ``2 * major_segments * minor_segments`` faces.

    ``major_radius`` is the distance from the axis to the tube centre,
    ``minor_radius`` the tube radius.
    """
    if major_segments < 3 or minor_segments < 3:
        raise ValueError("a torus needs at least 3 segments each way")
    u = np.arange(major_segments) * (2 * math.pi / major_segments)
    w = np.arange(minor_segments) * (2 * math.pi / minor_segments)
    uu, ww = np.meshgrid(u, w, indexing="ij")
    ring = major_radius + minor_radius * np.cos(ww)
    x, y, z = ring * np.cos(uu), ring * np.sin(uu), minor_radius * np.sin(ww)
    vertices = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    i, j = np.meshgrid(np.arange(major_segments), np.arange(minor_segments), indexing="ij")
    i, j = i.ravel(), j.ravel()
    i1, j1 = (i + 1) % major_segments, (j + 1) % minor_segments
    p00 = i * minor_segments + j
    p10 = i1 * minor_segments + j
    p11 = i1 * minor_segments + j1
    p01 = i * minor_segments + j1
    faces = np.concatenate([np.column_stack([p00, p10, p11]), np.column_stack([p00, p11, p01])])
    return Mesh(vertices, faces)
