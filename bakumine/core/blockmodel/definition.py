"""Block model definition and indexing (research.md section 6.1).

A block model is a regular grid of *parent* ("user") blocks. This module holds
its geometry only -- where the grid is, how big the blocks are, how it is
turned -- and the index arithmetic. The block table lives in
:mod:`bakumine.core.blockmodel.blocks`.

Frames
------
*World*: ``(x, y, z) = (east, north, up)``, metres.

*Local* (model frame): origin at the model's **minimum corner**, axes along
the model's ``i``, ``j``, ``k`` directions, metres. The model occupies the box
``0 <= local < extent`` in this frame::

    local = ((i + 1/2) dx, (j + 1/2) dy, (k + 1/2) dz)      centroid of parent (i, j, k)
    world = origin + R @ local

``R`` is the model rotation matrix: its **columns are the world directions of
the model i (x), j (y) and k (z) axes**. It is the identity for an unrotated
model, where ``i`` runs east, ``j`` north and ``k`` up. The rotation pivots
about ``origin`` (the minimum corner), so ``origin`` is the same world point
whatever the angles.

Rotation
--------
Either three angles with a named convention, or an explicit matrix.

Angles are passed to :func:`bakumine.core.estimation.rotation.rotation_matrix`
unchanged; no new rotation code is defined here. With the default convention
``"surpac_zxy_lrl"`` they are ``(bearing, plunge, dip)`` in degrees:

* ``bearing`` -- azimuth of the model ``j`` (y) axis, clockwise from north,
* ``plunge`` -- inclination of the ``j`` axis, positive upwards,
* ``dip`` -- rotation about the ``j`` axis; positive lifts the ``+i`` end.

**[verify]** Exactly like the ellipsoid convention this mapping is implemented
from the one-line statement in research.md 7.5 and has not been compared with
a rotated model made in Surpac. In particular it is an assumption that Surpac
applies its block-model bearing/dip/plunge with the same axes, order and signs
as its ellipsoid angles, and that it pivots about the minimum corner. Until a
rotated Surpac model has been checked, define rotated models with ``matrix``
(the three axis directions), which bypasses conventions.

How Surpac describes the same things
------------------------------------
======================  ==================================================
Surpac                  here
======================  ==================================================
origin (min Y, X, Z)    ``origin`` = (x, y, z) of the minimum corner
extent (Y, X, Z)        ``extent`` = ``shape * block_size`` (x, y, z)
user block size         ``block_size`` (dx, dy, dz), the parent block
minimum block size      ``block_size / 2**max_depth``
bearing / dip / plunge  ``angles`` + ``convention`` **[verify]**, or ``matrix``
======================  ==================================================

Surpac lists coordinates as Y, X, Z (northing first); everything here is
x, y, z and the order is swapped at the I/O boundary only.

Flat ordering
-------------
Parent blocks are numbered in C order of ``(i, j, k)``, ``k`` fastest::

    flat = (i * ny + j) * nz + k

so a column of the table reshaped to ``(nx, ny, nz)`` is indexed
``a[i, j, k]`` without copying, and the blocks of one vertical column are
contiguous. (The synthetic truth table uses the other order,
``row = i + nx * (j + ny * k)``; :meth:`BlockModelDefinition.flat_index_f`
gives that number.)

Half-open rule
--------------
A block owns its minimum faces and not its maximum faces. In local
coordinates parent ``(i, j, k)`` contains the points with

    fl(i * dx) <= local_x < fl((i + 1) * dx)        (same for y, z)

evaluated in floating point exactly as written, so every point of the model
box belongs to exactly one block, a point on a shared face belongs to the
block on its positive side, and points on the model's maximum faces are
outside the model. For a rotated model the rule applies to the rotated
coordinates ``local = R.T @ (world - origin)``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

from bakumine.core.estimation.rotation import CONVENTIONS, DEFAULT_CONVENTION, rotation_matrix

__all__ = ["MAX_DEPTH", "BlockModelDefinition", "cell_index"]

#: largest sub-blocking depth per axis (sub-indices are stored in uint8)
MAX_DEPTH = 8

_ORTHO_TOL = 1e-9
_DIVISIBILITY_TOL = 1e-9


def cell_index(coord: np.ndarray, size: float, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Index of the half-open cell ``fl(n * size) <= coord < fl((n + 1) * size)``.

    Returns ``(index, inside)``: ``index`` is int64, clipped to ``[-1, count]``
    (``-1`` = before the first cell or NaN, ``count`` = past the last cell);
    ``inside`` is True where ``0 <= index < count``.
    """
    coord = np.asarray(coord, dtype=np.float64)
    with np.errstate(invalid="ignore"):
        idx = np.floor(coord / size)
        # floor(coord / size) can be off by one at a face because the division
        # rounds; make the result agree with the documented comparisons.
        idx -= coord < idx * size
        idx += coord >= (idx + 1.0) * size
        idx = np.where(np.isnan(idx), -1.0, np.clip(idx, -1.0, float(count)))
    index = idx.astype(np.int64)
    return index, (index >= 0) & (index < count)


def _triple(value: Any, name: str, kind: type) -> tuple:
    try:
        out = tuple(kind(v) for v in value)
    except TypeError:
        raise ValueError(f"{name} must have three components") from None
    if len(out) != 3:
        raise ValueError(f"{name} must have three components, got {len(out)}")
    return out


def _check_rotation(matrix: Any) -> np.ndarray:
    r = np.array(matrix, dtype=np.float64)
    if r.shape != (3, 3) or not np.isfinite(r).all():
        raise ValueError("rotation matrix must be a finite 3x3 matrix")
    if not np.allclose(r.T @ r, np.eye(3), rtol=0.0, atol=_ORTHO_TOL):
        raise ValueError("rotation matrix is not orthonormal")
    if np.linalg.det(r) < 0:
        raise ValueError("rotation matrix has determinant -1 (a reflection)")
    return r


def _geometry_points(item: Any) -> np.ndarray:
    """(n, 3) points of a Mesh, a DTM (anything with ``.mesh``) or an array."""
    if hasattr(item, "mesh"):
        item = item.mesh
    if hasattr(item, "vertices"):
        item = item.vertices
    pts = np.asarray(item, dtype=np.float64)
    if pts.ndim == 1:
        pts = pts.reshape(1, -1)
    if pts.shape[-1] != 3:
        raise ValueError(f"geometry points must have shape (n, 3), got {pts.shape}")
    return pts.reshape(-1, 3)


@dataclass(frozen=True)
class BlockModelDefinition:
    """Geometry of a block model. Immutable, hashable, JSON-serialisable.

    Parameters
    ----------
    origin : (x, y, z)
        World coordinates of the minimum corner of the model.
    block_size : (dx, dy, dz)
        Parent ("user") block size along the model axes, metres, > 0.
    shape : (nx, ny, nz)
        Number of parent blocks along the model axes, >= 1. The extent is
        ``shape * block_size``; use :meth:`from_extent` to give an extent.
    angles, convention
        Model rotation as three angles in degrees and a convention name of
        :data:`bakumine.core.estimation.rotation.CONVENTIONS`
        (default ``"surpac_zxy_lrl"``: bearing, plunge, dip). See the module
        docstring, including the **[verify]** note.
    matrix : 3x3, optional
        Explicit rotation matrix; columns are the world directions of the
        model i, j, k axes. When given, ``angles`` and ``convention`` are
        ignored.
    max_depth : int or (int, int, int)
        Maximum sub-blocking depth per axis, 0..8. The smallest block is
        ``block_size / 2**max_depth``. One number applies to all axes.
        Unequal depths give anisotropic sub-blocks (e.g. ``(3, 3, 0)`` never
        splits Z); see :mod:`bakumine.core.blockmodel.blocks`.
    """

    origin: tuple[float, float, float]
    block_size: tuple[float, float, float]
    shape: tuple[int, int, int]
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    convention: str = DEFAULT_CONVENTION
    matrix: tuple[tuple[float, float, float], ...] | None = None
    max_depth: tuple[int, int, int] = (0, 0, 0)

    def __post_init__(self) -> None:
        origin = _triple(self.origin, "origin", float)
        size = _triple(self.block_size, "block_size", float)
        if not all(np.isfinite(origin)):
            raise ValueError("origin must be finite")
        if not all(np.isfinite(s) and s > 0.0 for s in size):
            raise ValueError(f"block_size must be finite and > 0, got {size}")
        raw_shape = _triple(self.shape, "shape", float)
        if not all(n >= 1 and n == int(n) for n in raw_shape):
            raise ValueError(f"shape must be three integers >= 1, got {self.shape}")
        shape = tuple(int(n) for n in raw_shape)
        if shape[0] * shape[1] * shape[2] >= 2**31:
            raise ValueError("more than 2**31 - 1 parent blocks are not supported")
        depth = self.max_depth
        if isinstance(depth, int | np.integer):
            depth = (depth, depth, depth)
        raw_depth = _triple(depth, "max_depth", float)
        if not all(0 <= d <= MAX_DEPTH and d == int(d) for d in raw_depth):
            raise ValueError(f"max_depth must be integers in 0..{MAX_DEPTH}, got {self.max_depth}")
        angles = _triple(self.angles, "angles", float)
        if not all(np.isfinite(angles)):
            raise ValueError("angles must be finite (degrees)")
        matrix = self.matrix
        if matrix is not None:
            r = _check_rotation(matrix)
            matrix = tuple(tuple(float(v) for v in row) for row in r)
        elif self.convention not in CONVENTIONS:
            raise ValueError(
                f"unknown rotation convention {self.convention!r}; available: {sorted(CONVENTIONS)}"
            )
        for name, value in (
            ("origin", origin),
            ("block_size", size),
            ("shape", shape),
            ("angles", angles),
            ("matrix", matrix),
            ("max_depth", tuple(int(d) for d in raw_depth)),
        ):
            object.__setattr__(self, name, value)

    # ---------------------------------------------------------------- builders

    @classmethod
    def from_extent(
        cls, origin: Any, extent: Any, block_size: Any, **kwargs: Any
    ) -> BlockModelDefinition:
        """Definition from origin + extent (the way Surpac asks for it).

        ``extent`` must be a whole number of blocks on every axis (relative
        tolerance 1e-9); otherwise ``ValueError``. Other arguments as for the
        constructor.
        """
        extent = np.array(_triple(extent, "extent", float))
        size = np.array(_triple(block_size, "block_size", float))
        if not ((size > 0).all() and (extent > 0).all()):
            raise ValueError("extent and block_size must be > 0")
        counts = np.rint(extent / size)
        bad = np.abs(counts * size - extent) > _DIVISIBILITY_TOL * np.maximum(extent, size)
        if bad.any() or (counts < 1).any():
            axes = ", ".join("xyz"[a] for a in np.flatnonzero(bad | (counts < 1)))
            raise ValueError(
                f"extent {tuple(extent)} is not a whole number of blocks of size "
                f"{tuple(size)} along {axes}"
            )
        return cls(origin, tuple(size), tuple(int(c) for c in counts), **kwargs)

    @classmethod
    def auto_extent(
        cls,
        geometry: Any,
        block_size: Any,
        margin: Any = 0.0,
        snap: bool = True,
        **kwargs: Any,
    ) -> BlockModelDefinition:
        """Definition that covers the given geometry plus a margin.

        Parameters
        ----------
        geometry : a Mesh, a DTM, an (n, 3) array of points, or an iterable
            of those.
        block_size : (dx, dy, dz)
        margin : float or (mx, my, mz)
            Added on both sides of every model axis, metres, >= 0.
        snap : bool
            Move the minimum corner down to a multiple of the block size
            (measured in the model frame from the world origin; for an
            unrotated model the origin coordinates become multiples of the
            block size). The model only grows.
        **kwargs : rotation and ``max_depth`` as for the constructor. For a
            rotated model the box is fitted in the rotated frame.

        The model is the smallest whole number of blocks whose *closed* box
        contains every point with the margin; it may exceed the margin by up
        to one block. With a zero margin the extreme points can lie on the
        model's maximum faces, which the half-open rule puts outside the
        model: use a positive margin when that matters.
        """
        size = np.array(_triple(block_size, "block_size", float))
        margin = np.broadcast_to(np.asarray(margin, dtype=np.float64), (3,))
        if not (np.isfinite(margin).all() and (margin >= 0).all()):
            raise ValueError("margin must be finite and >= 0")
        if hasattr(geometry, "mesh") or hasattr(geometry, "vertices"):
            items = [geometry]
        elif isinstance(geometry, np.ndarray):
            items = [geometry]
        elif isinstance(geometry, Iterable):
            items = list(geometry)
            if items and np.isscalar(items[0]):
                items = [geometry]
        else:
            raise TypeError("geometry must be a Mesh, DTM, point array or an iterable of those")
        clouds = [p for p in (_geometry_points(item) for item in items) if len(p)]
        if not clouds:
            raise ValueError("auto_extent needs at least one point")
        points = np.concatenate(clouds)
        if not np.isfinite(points).all():
            raise ValueError("geometry contains NaN or infinite coordinates")
        probe = cls((0.0, 0.0, 0.0), tuple(size), (1, 1, 1), **kwargs)
        r = probe.rotation
        local = points if not probe.is_rotated else points @ r
        lo = local.min(axis=0) - margin
        hi = local.max(axis=0) + margin
        if snap:
            lo = np.floor(lo / size) * size
        counts = np.maximum(np.ceil((hi - lo) / size), 1.0)
        # ceil of a rounded quotient can fall one block short
        counts += lo + counts * size < hi
        origin = lo if not probe.is_rotated else r @ lo
        return cls(tuple(origin), tuple(size), tuple(int(c) for c in counts), **kwargs)

    # -------------------------------------------------------------- properties

    @cached_property
    def rotation(self) -> np.ndarray:
        """(3, 3) rotation matrix, read-only; columns = world directions of the i, j, k axes."""
        if self.matrix is not None:
            r = np.array(self.matrix, dtype=np.float64)
        else:
            r = rotation_matrix(*self.angles, convention=self.convention)
        r.setflags(write=False)
        return r

    @cached_property
    def is_rotated(self) -> bool:
        """False when the rotation matrix is exactly the identity."""
        return not np.array_equal(self.rotation, np.eye(3))

    @property
    def nx(self) -> int:
        return self.shape[0]

    @property
    def ny(self) -> int:
        return self.shape[1]

    @property
    def nz(self) -> int:
        return self.shape[2]

    @property
    def n_parents(self) -> int:
        """Number of parent blocks."""
        return self.shape[0] * self.shape[1] * self.shape[2]

    @property
    def extent(self) -> tuple[float, float, float]:
        """Model size along its axes, ``shape * block_size``."""
        return tuple(n * s for n, s in zip(self.shape, self.block_size, strict=True))

    @property
    def min_block_size(self) -> tuple[float, float, float]:
        """Smallest sub-block, ``block_size / 2**max_depth`` (exact)."""
        return tuple(s / (1 << d) for s, d in zip(self.block_size, self.max_depth, strict=True))

    @property
    def volume(self) -> float:
        """Volume of the model box."""
        ex = self.extent
        return ex[0] * ex[1] * ex[2]

    # ------------------------------------------------------------- transforms

    def local_to_world(self, local: Any) -> np.ndarray:
        """``origin + R @ local`` for points ``(..., 3)`` in the model frame.

        Computed elementwise (no BLAS), so the result for a point does not
        depend on the other points in the batch.
        """
        p = np.asarray(local, dtype=np.float64)
        if p.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {p.shape}")
        out = np.empty(p.shape, dtype=np.float64)
        if not self.is_rotated:
            for a in range(3):
                out[..., a] = self.origin[a] + p[..., a]
            return out
        r = self.rotation
        px, py, pz = p[..., 0], p[..., 1], p[..., 2]
        for a in range(3):
            out[..., a] = self.origin[a] + (px * r[a, 0] + py * r[a, 1] + pz * r[a, 2])
        return out

    def world_to_local(self, points: Any) -> np.ndarray:
        """``R.T @ (world - origin)`` for points ``(..., 3)``; inverse of :meth:`local_to_world`."""
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {p.shape}")
        out = np.empty(p.shape, dtype=np.float64)
        if not self.is_rotated:
            for a in range(3):
                out[..., a] = p[..., a] - self.origin[a]
            return out
        r = self.rotation
        qx, qy, qz = (p[..., a] - self.origin[a] for a in range(3))
        for a in range(3):
            out[..., a] = qx * r[0, a] + qy * r[1, a] + qz * r[2, a]
        return out

    def ijk_to_local(self, ijk: Any) -> np.ndarray:
        """Local centroid ``((i + 1/2) dx, (j + 1/2) dy, (k + 1/2) dz)`` of blocks ``(..., 3)``.

        ``ijk`` may be fractional (``i + 1/2`` is added to whatever is given)
        and is not range-checked.
        """
        ijk = np.asarray(ijk, dtype=np.float64)
        if ijk.shape[-1:] != (3,):
            raise ValueError(f"ijk must have shape (..., 3), got {ijk.shape}")
        return (ijk + 0.5) * np.array(self.block_size)

    def ijk_to_world(self, ijk: Any) -> np.ndarray:
        """World centroid of parent blocks ``(..., 3)``."""
        return self.local_to_world(self.ijk_to_local(ijk))

    def local_to_ijk(self, local: Any) -> tuple[np.ndarray, np.ndarray]:
        """Parent block containing points given in the model frame.

        Returns ``(ijk, inside)``: ``ijk`` is int64 ``(..., 3)`` and
        ``inside`` is True for points in the model. Points outside have their
        index clipped to ``-1`` (below the model, or NaN) or ``n`` (beyond
        it) on the offending axes. See the module docstring for the half-open
        rule.
        """
        p = np.asarray(local, dtype=np.float64)
        if p.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {p.shape}")
        ijk = np.empty(p.shape, dtype=np.int64)
        inside = np.ones(p.shape[:-1], dtype=bool)
        for a in range(3):
            ijk[..., a], ok = cell_index(p[..., a], self.block_size[a], self.shape[a])
            inside &= ok
        return ijk, inside

    def world_to_ijk(self, points: Any) -> tuple[np.ndarray, np.ndarray]:
        """Parent block containing world points ``(..., 3)``: ``(ijk, inside)``.

        See :meth:`local_to_ijk`.
        """
        return self.local_to_ijk(self.world_to_local(points))

    # ---------------------------------------------------------------- indexing

    def flat_index(self, ijk: Any) -> np.ndarray:
        """Flat parent index ``(i * ny + j) * nz + k`` (int64). Not range-checked."""
        ijk = np.asarray(ijk, dtype=np.int64)
        return (ijk[..., 0] * self.shape[1] + ijk[..., 1]) * self.shape[2] + ijk[..., 2]

    def unflat_index(self, flat: Any) -> np.ndarray:
        """``(..., 3)`` int64 ``ijk`` of flat parent indices; inverse of :meth:`flat_index`."""
        flat = np.asarray(flat, dtype=np.int64)
        out = np.empty((*flat.shape, 3), dtype=np.int64)
        out[..., 2] = flat % self.shape[2]
        rest = flat // self.shape[2]
        out[..., 1] = rest % self.shape[1]
        out[..., 0] = rest // self.shape[1]
        return out

    def flat_index_f(self, ijk: Any) -> np.ndarray:
        """Fortran-order number ``i + nx * (j + ny * k)`` (``i`` fastest).

        Not used internally; this is the row order of the synthetic truth
        table and of most block CSV exports.
        """
        ijk = np.asarray(ijk, dtype=np.int64)
        return ijk[..., 0] + self.shape[0] * (ijk[..., 1] + self.shape[1] * ijk[..., 2])

    def parent_ijk(self) -> np.ndarray:
        """(n_parents, 3) int64 ``ijk`` of every parent block in flat order."""
        return self.unflat_index(np.arange(self.n_parents))

    def parent_centroids(self) -> np.ndarray:
        """(n_parents, 3) world centroids of every parent block in flat order."""
        return self.ijk_to_world(self.parent_ijk())

    # ---------------------------------------------------------------- geometry

    def box_corners(self, lo_local: Any, hi_local: Any) -> np.ndarray:
        """World corners ``(..., 8, 3)`` of boxes given by local min / max corners ``(..., 3)``.

        Corner ``c`` is at ``(hi if c & 4 else lo, hi if c & 2 else lo, hi if
        c & 1 else lo)`` along (i, j, k): the k side changes fastest.
        """
        lo = np.asarray(lo_local, dtype=np.float64)
        hi = np.asarray(hi_local, dtype=np.float64)
        lo, hi = np.broadcast_arrays(lo, hi)
        local = np.empty((*lo.shape[:-1], 8, 3), dtype=np.float64)
        for c in range(8):
            for a, bit in enumerate((4, 2, 1)):
                local[..., c, a] = hi[..., a] if c & bit else lo[..., a]
        return self.local_to_world(local)

    def block_corners(self, ijk: Any) -> np.ndarray:
        """World corners ``(..., 8, 3)`` of parent blocks; order as in :meth:`box_corners`."""
        ijk = np.asarray(ijk, dtype=np.float64)
        size = np.array(self.block_size)
        return self.box_corners(ijk * size, (ijk + 1.0) * size)

    def bounding_box(self) -> np.ndarray:
        """(2, 3) world axis-aligned bounding box ``[min, max]`` of the model."""
        corners = self.box_corners(np.zeros(3), np.array(self.extent))
        return np.stack([corners.min(axis=0), corners.max(axis=0)])

    # ---------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; inverse of :meth:`from_dict`."""
        out: dict[str, Any] = {
            "origin": list(self.origin),
            "block_size": list(self.block_size),
            "shape": list(self.shape),
            "max_depth": list(self.max_depth),
        }
        if self.matrix is not None:
            out["rotation"] = {"matrix": [list(row) for row in self.matrix]}
        else:
            out["rotation"] = {"convention": self.convention, "angles": list(self.angles)}
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlockModelDefinition:
        """Build from the dict produced by :meth:`to_dict`."""
        rotation = data.get("rotation", {})
        matrix = rotation.get("matrix")
        return cls(
            origin=tuple(data["origin"]),
            block_size=tuple(data["block_size"]),
            shape=tuple(data["shape"]),
            angles=tuple(rotation.get("angles", (0.0, 0.0, 0.0))),
            convention=rotation.get("convention", DEFAULT_CONVENTION),
            matrix=None if matrix is None else tuple(tuple(row) for row in matrix),
            max_depth=tuple(data.get("max_depth", (0, 0, 0))),
        )

    def to_json(self, **kwargs: Any) -> str:
        """JSON text of :meth:`to_dict` (floats round-trip exactly)."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> BlockModelDefinition:
        return cls.from_dict(json.loads(text))
