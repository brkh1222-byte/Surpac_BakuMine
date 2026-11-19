"""The truth ore solid: a dipping, plunging lens with a pinch.

The solid is defined by an implicit function, so membership of any point is exact and does
not depend on the mesh. In the lens frame ``(u, v, w)`` (long axis, down-dip axis, across):

    t(u)      = 1 - pinch_depth * exp(-((u - pinch_position) / pinch_width)**2)
    m(u, v)   = bend_major * (u / a)**2 + bend_semi * (v / b)**2
    f(u,v,w)  = (u / a)**2 + (v / b)**2 + ((w - m(u, v)) / (c * t(u)))**2 - 1

A point is inside when ``f < 0``. ``t`` necks the thickness (the pinch) and ``m`` bends the
mid-surface, so the solid is neither convex nor axis-aligned. Because the bend is a shear
along ``w``, the volume reduces to a one-dimensional integral:

    V = pi * b * c * integral_{-a}^{a} t(u) * (1 - u**2 / a**2) du

The mesh is the image of an icosphere under the map that sends the unit sphere onto
``f = 0``, so every vertex lies exactly on the surface.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from bakumine.synth.config import LensConfig


def axes_from_angles(bearing: float, plunge: float, dip: float) -> np.ndarray:
    """Unit axes of an oriented ellipsoid as the columns of a 3x3 matrix (world = east, north, up).

    Column 0, the major axis, points along azimuth ``bearing`` (clockwise from north) and
    plunges ``plunge`` degrees below horizontal. With ``dip = 0`` the semi-major axis
    (column 1) is horizontal, 90 degrees anticlockwise of the bearing, and the minor axis
    (column 2) points up. ``dip`` rotates both about the major axis so that the semi-major
    axis tilts downward. The matrix is a proper rotation (determinant +1).

    This is this package's own convention; it is recorded as a matrix in ``truth.json`` so
    that no assumption about another package's angle convention is needed.
    """
    b, p, d = np.radians([bearing, plunge, dip])
    major = np.array([np.sin(b) * np.cos(p), np.cos(b) * np.cos(p), -np.sin(p)])
    semi0 = np.array([-np.cos(b), np.sin(b), 0.0])
    minor0 = np.cross(major, semi0)
    semi = np.cos(d) * semi0 - np.sin(d) * minor0
    minor = np.cos(d) * minor0 + np.sin(d) * semi0
    return np.column_stack([major, semi, minor])


def icosphere(level: int) -> tuple[np.ndarray, np.ndarray]:
    """Unit icosphere: vertices (n, 3) and outward-oriented faces (m, 3)."""
    phi = (1.0 + np.sqrt(5.0)) / 2.0
    vertices = np.array(
        [
            [-1, phi, 0], [1, phi, 0], [-1, -phi, 0], [1, -phi, 0],
            [0, -1, phi], [0, 1, phi], [0, -1, -phi], [0, 1, -phi],
            [phi, 0, -1], [phi, 0, 1], [-phi, 0, -1], [-phi, 0, 1],
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
    vertices /= np.linalg.norm(vertices, axis=1, keepdims=True)
    for _ in range(level):
        n = len(vertices)
        edges = np.sort(
            np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1
        )
        unique, inverse = np.unique(edges, axis=0, return_inverse=True)
        inverse = inverse.reshape(-1)
        mid = vertices[unique[:, 0]] + vertices[unique[:, 1]]
        mid /= np.linalg.norm(mid, axis=1, keepdims=True)
        vertices = np.concatenate([vertices, mid])
        m = len(faces)
        ab, bc, ca = n + inverse[:m], n + inverse[m : 2 * m], n + inverse[2 * m :]
        a, b, c = faces[:, 0], faces[:, 1], faces[:, 2]
        faces = np.concatenate(
            [
                np.column_stack([a, ab, ca]),
                np.column_stack([b, bc, ab]),
                np.column_stack([c, ca, bc]),
                np.column_stack([ab, bc, ca]),
            ]
        )
    return vertices, faces


def mesh_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    """Signed volume of a closed mesh (positive when faces are oriented outward)."""
    v = vertices - vertices.mean(axis=0)  # centring keeps the sum well conditioned
    a, b, c = v[faces[:, 0]], v[faces[:, 1]], v[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


@dataclass(frozen=True)
class LensSolid:
    centre: np.ndarray
    axes: np.ndarray  # columns: long axis, down-dip axis, across-lens axis (world)
    a: float
    b: float
    c: float
    pinch_position: float
    pinch_width: float
    pinch_depth: float
    bend_major: float
    bend_semi: float

    @classmethod
    def from_config(cls, cfg: LensConfig) -> LensSolid:
        if not 0.0 <= cfg.pinch_depth < 1.0:
            raise ValueError("pinch_depth must be in [0, 1)")
        return cls(
            centre=np.array(cfg.centre, dtype=np.float64),
            axes=axes_from_angles(cfg.bearing, cfg.plunge, cfg.dip),
            a=cfg.half_length,
            b=cfg.half_width,
            c=cfg.half_thickness,
            pinch_position=cfg.pinch_position,
            pinch_width=cfg.pinch_width,
            pinch_depth=cfg.pinch_depth,
            bend_major=cfg.bend_major,
            bend_semi=cfg.bend_semi,
        )

    def to_local(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points, dtype=np.float64) - self.centre) @ self.axes

    def to_world(self, local: np.ndarray) -> np.ndarray:
        return self.centre + np.asarray(local, dtype=np.float64) @ self.axes.T

    def thickness_factor(self, u: np.ndarray) -> np.ndarray:
        offset = (u - self.pinch_position) / self.pinch_width
        return 1.0 - self.pinch_depth * np.exp(-(offset**2))

    def bend(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        return self.bend_major * (u / self.a) ** 2 + self.bend_semi * (v / self.b) ** 2

    def implicit(self, points: np.ndarray) -> np.ndarray:
        """``f`` at world points (n, 3): negative inside, zero on the surface, positive outside."""
        q = self.to_local(points)
        u, v, w = q[..., 0], q[..., 1], q[..., 2]
        across = (w - self.bend(u, v)) / (self.c * self.thickness_factor(u))
        return (u / self.a) ** 2 + (v / self.b) ** 2 + across**2 - 1.0

    def inside(self, points: np.ndarray) -> np.ndarray:
        return self.implicit(points) < 0.0

    def volume(self) -> float:
        """Exact volume: closed-form reduction to 1-D, evaluated by adaptive quadrature."""
        a = self.a
        value, _ = quad(
            lambda u: self.thickness_factor(u) * (1.0 - (u / a) ** 2),
            -a,
            a,
            points=[np.clip(self.pinch_position, -a, a)],
            epsabs=1e-10,
            epsrel=1e-13,
            limit=200,
        )
        return float(np.pi * self.b * self.c * value)

    def mesh(self, subdivisions: int) -> tuple[np.ndarray, np.ndarray]:
        """Closed, outward-oriented triangle mesh with every vertex on ``f = 0`` (world)."""
        sphere, faces = icosphere(subdivisions)
        u = self.a * sphere[:, 0]
        v = self.b * sphere[:, 1]
        w = self.c * self.thickness_factor(u) * sphere[:, 2] + self.bend(u, v)
        vertices = self.to_world(np.column_stack([u, v, w]))
        if mesh_volume(vertices, faces) < 0.0:
            faces = faces[:, ::-1].copy()
        return vertices, faces
