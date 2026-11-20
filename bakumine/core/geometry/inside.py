"""Point-in-solid tests by vertical ray casting (research.md section 5.5).

A point is inside a closed mesh when a ray from it towards +Z crosses the
surface an odd number of times. Triangles are found through the plan-view
grid index of the mesh (:attr:`Mesh.grid_xy`).

The rule, exactly
-----------------
A query point ``p = (x, y, z)`` is classified as the point
``(x + e, y + e**2, z + e**3)`` with ``e`` an infinitesimal positive number:

1. *Which triangles the ray passes through* is decided in plan with exact
   arithmetic for the displaced point ``(x + e, y + e**2)``
   (:mod:`bakumine.core.geometry.predicates`). A ray through a shared edge
   or vertex therefore crosses exactly one of the triangles of each surface
   sheet there; vertical triangles are never crossed. No tolerance is
   involved and no crossing is ever counted twice or missed.
2. Each crossed triangle gives one crossing height ``zc`` (barycentric
   interpolation, clamped to the Z range of the triangle). The crossing is
   above the point when ``zc > z`` (strictly).
3. Inside = odd number of crossings above.

Consequences:

* For faces parallel to the coordinate planes the result is exact and
  half-open: an axis-aligned box ``lo..hi`` contains exactly the points with
  ``lo <= p < hi`` on every axis. A regular grid of blocks whose faces lie
  on the solid boundary is therefore split without gaps or overlaps.
* For a point on an oblique face the plan test is still exact, but ``zc`` is
  a rounded float: points within rounding distance of the surface (about
  1e-15 of the coordinate magnitude) may fall on either side. The answer is
  deterministic, and identical between :func:`points_in_solid` and
  :func:`grid_in_solid` for the same float coordinates.
* The winding of the faces is irrelevant (parity). A cavity (a shell inside
  a shell, any winding) is outside; nested solids alternate.

Open or invalid meshes
----------------------
Parity is only meaningful when every edge is used by an even number of
faces. Otherwise (open border, hole, unwelded triangle soup, empty mesh) the
functions raise ``ValueError``; with ``strict=False`` they emit a
:class:`OpenMeshWarning` and return the parity anyway (rays that leave
through a hole give wrong answers for whole columns). Self-intersections and
inconsistent winding are not checked here; use :meth:`Mesh.validate`.

Regular grids
-------------
:func:`grid_in_solid` exploits columns: all points of a grid that share
``(i, j)`` share one ray. The sorted crossing heights are computed once per
column and the column is filled by parity. For a rotated grid the mesh is
rotated into the grid frame once. The per-column crossings can be returned
as entry/exit intervals (:class:`ColumnCrossings`) for partial-volume and
sub-blocking decisions.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.predicates import tri_covers, tri_z
from bakumine.core.geometry.spatial import TriangleGridXY, _cell
from bakumine.core.parallel import parallel_njit

__all__ = [
    "ColumnCrossings",
    "OpenMeshWarning",
    "column_crossings",
    "grid_in_solid",
    "grid_points",
    "points_in_solid",
]


class OpenMeshWarning(UserWarning):
    """Point-in-solid was asked on a mesh that does not enclose a volume."""


# --------------------------------------------------------------------------
# kernels
# --------------------------------------------------------------------------


@parallel_njit(cache=True)
def _points_inside(pts, v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    n = pts.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        px = pts[i, 0]
        py = pts[i, 1]
        pz = pts[i, 2]
        if not (px == px and py == py and pz == pz):
            continue
        c = _cell(px, x0, invx, nx) * ny + _cell(py, y0, invy, ny)
        odd = False
        for k in range(start[c], start[c + 1]):
            t = items[k]
            if px < lo[t, 0] or px > hi[t, 0] or py < lo[t, 1] or py > hi[t, 1]:
                continue
            if hi[t, 2] <= pz:  # the whole triangle is at or below the point
                continue
            ia = f[t, 0]
            ib = f[t, 1]
            ic = f[t, 2]
            if not tri_covers(v[ia, 0], v[ia, 1], v[ib, 0], v[ib, 1], v[ic, 0], v[ic, 1], px, py):
                continue
            if lo[t, 2] > pz:  # the whole triangle is above the point
                odd = not odd
                continue
            zc = tri_z(
                v[ia, 0], v[ia, 1], v[ia, 2],
                v[ib, 0], v[ib, 1], v[ib, 2],
                v[ic, 0], v[ic, 1], v[ic, 2],
                px, py,
            )  # fmt: skip
            if zc > pz:
                odd = not odd
        out[i] = odd
    return out


@njit(cache=True, inline="always")
def _column(px, py, v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi, out, pos, write):
    """Crossing heights of the vertical line through ``(px, py)``; returns the count."""
    if not (px == px and py == py):
        return 0
    c = _cell(px, x0, invx, nx) * ny + _cell(py, y0, invy, ny)
    count = 0
    for k in range(start[c], start[c + 1]):
        t = items[k]
        if px < lo[t, 0] or px > hi[t, 0] or py < lo[t, 1] or py > hi[t, 1]:
            continue
        ia = f[t, 0]
        ib = f[t, 1]
        ic = f[t, 2]
        if not tri_covers(v[ia, 0], v[ia, 1], v[ib, 0], v[ib, 1], v[ic, 0], v[ic, 1], px, py):
            continue
        if write:
            out[pos + count] = tri_z(
                v[ia, 0], v[ia, 1], v[ia, 2],
                v[ib, 0], v[ib, 1], v[ib, 2],
                v[ic, 0], v[ic, 1], v[ic, 2],
                px, py,
            )  # fmt: skip
        count += 1
    return count


@parallel_njit(cache=True)
def _columns(xs, ys, v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    """Sorted crossing heights of the columns ``(xs[i], ys[j])`` as CSR (column = i*len(ys)+j)."""
    na = xs.shape[0]
    nb = ys.shape[0]
    offsets = np.zeros(na * nb + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.float64)
    for i in prange(na):
        for j in range(nb):
            offsets[i * nb + j + 1] = _column(
                xs[i], ys[j], v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi,
                dummy, 0, False,
            )  # fmt: skip
    for c in range(na * nb):
        offsets[c + 1] += offsets[c]
    z = np.empty(offsets[na * nb], dtype=np.float64)
    for i in prange(na):
        for j in range(nb):
            p = offsets[i * nb + j]
            q = offsets[i * nb + j + 1]
            if q > p:
                _column(
                    xs[i], ys[j], v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi,
                    z, p, True,
                )  # fmt: skip
                z[p:q].sort()
    return offsets, z


@parallel_njit(cache=True)
def _columns_xy(xy, v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    """Same as :func:`_columns` for arbitrary column positions ``xy`` (n, 2)."""
    n = xy.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.float64)
    for i in prange(n):
        offsets[i + 1] = _column(
            xy[i, 0], xy[i, 1], v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi,
            dummy, 0, False,
        )  # fmt: skip
    for c in range(n):
        offsets[c + 1] += offsets[c]
    z = np.empty(offsets[n], dtype=np.float64)
    for i in prange(n):
        p = offsets[i]
        q = offsets[i + 1]
        if q > p:
            _column(
                xy[i, 0], xy[i, 1], v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi,
                z, p, True,
            )  # fmt: skip
            z[p:q].sort()
    return offsets, z


@parallel_njit(cache=True)
def _fill(offsets, z, na, nb, zk_all):
    """mask[i, j, k] = odd number of crossings strictly above ``zk_all[k]`` (ascending)."""
    nk = zk_all.shape[0]
    mask = np.zeros((na, nb, nk), dtype=np.bool_)
    for i in prange(na):
        for j in range(nb):
            p = offsets[i * nb + j]
            q = offsets[i * nb + j + 1]
            if q == p:
                continue
            for k in range(nk):
                zk = zk_all[k]
                while p < q and z[p] <= zk:
                    p += 1
                if p == q:
                    break
                mask[i, j, k] = ((q - p) & 1) == 1
    return mask


# --------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnCrossings:
    """Heights at which vertical lines (columns) cross a mesh.

    Attributes
    ----------
    shape : tuple of int
        Shape of the set of columns: ``(nx, ny)`` for a grid, ``(n,)`` for a
        list of positions. Columns are numbered in C order
        (``column = i * ny + j`` for a grid).
    offsets : (n_columns + 1,) int64
        CSR row pointer into ``z``.
    z : (total,) float64
        Crossing heights, ascending inside each column. For a closed mesh
        every column has an even number of them and consecutive pairs
        ``(z[2q], z[2q + 1])`` are the entry and exit of the solid going
        upwards. The solid occupies the half-open interval
        ``entry <= z < exit``; a pair may have zero length.
    z0, dz : float
        For :func:`grid_in_solid`: grid point ``k`` of every column is at
        height ``z0 + k * dz``, so ``mask[i, j, k]`` is True exactly when
        that height lies in one of the intervals. NaN for
        :func:`column_crossings`.
    frame : str
        ``"world"``: heights are world Z. ``"grid"``: the grid is rotated and
        heights are distances along the grid's third axis from its origin.
    """

    shape: tuple[int, ...]
    offsets: np.ndarray
    z: np.ndarray
    z0: float = float("nan")
    dz: float = float("nan")
    frame: str = "world"

    @property
    def n_columns(self) -> int:
        return len(self.offsets) - 1

    @property
    def counts(self) -> np.ndarray:
        """Number of crossings per column, in the shape of the column set."""
        return np.diff(self.offsets).reshape(self.shape)

    def _require_pairs(self) -> None:
        if (np.diff(self.offsets) & 1).any():
            raise ValueError(
                "some columns have an odd number of crossings (open mesh): "
                "entry/exit intervals are undefined"
            )

    def column(self, *index: int) -> np.ndarray:
        """(m, 2) entry/exit heights of one column, e.g. ``column(i, j)``."""
        c = int(np.ravel_multi_index(index, self.shape))
        zs = self.z[self.offsets[c] : self.offsets[c + 1]]
        if len(zs) & 1:
            raise ValueError("odd number of crossings in this column (open mesh)")
        return zs.reshape(-1, 2).copy()

    def intervals(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """All intervals as ``(column, z_entry, z_exit)`` arrays, sorted by column then height.

        ``column`` is the flat column number; use ``np.unravel_index(column, shape)``.
        """
        self._require_pairs()
        column = np.repeat(np.arange(self.n_columns), np.diff(self.offsets) // 2)
        return column, self.z[0::2].copy(), self.z[1::2].copy()

    def thickness(self) -> np.ndarray:
        """Total length of each column that is inside the solid, in the shape of the column set."""
        column, entry, exit_ = self.intervals()
        return np.bincount(column, weights=exit_ - entry, minlength=self.n_columns).reshape(
            self.shape
        )


def _check_closed(mesh: Mesh, strict: bool) -> None:
    if mesh.n_faces == 0:
        problem = "the mesh has no faces"
    else:
        odd = int((mesh.edge_map.counts & 1).sum())
        if odd == 0:
            return
        problem = (
            f"the mesh is not closed: {odd} edge(s) are used by an odd number of faces "
            "(open border or hole; weld the mesh first if vertices are repeated per face)"
        )
    if strict:
        raise ValueError(f"point-in-solid needs a closed mesh, but {problem}")
    warnings.warn(
        f"point-in-solid on a mesh that does not enclose a volume ({problem}); "
        "results are the parity of the crossings above each point and are not reliable",
        OpenMeshWarning,
        stacklevel=3,
    )


def _arrays(vertices: np.ndarray, faces: np.ndarray, grid: TriangleGridXY):
    return (
        np.ascontiguousarray(vertices), np.ascontiguousarray(faces),
        grid.origin[0], grid.origin[1], grid.inv_cell_size[0], grid.inv_cell_size[1],
        grid.shape[0], grid.shape[1], grid.cell_start, grid.cell_tris, grid.tri_lo, grid.tri_hi,
    )  # fmt: skip


def points_in_solid(mesh: Mesh, points, *, strict: bool = True) -> np.ndarray:
    """Which points are inside a closed mesh.

    Parameters
    ----------
    mesh : Mesh
        Closed surface (any winding; cavities and several shells are fine).
    points : (..., 3) array_like
        World coordinates (east, north, up). Any leading shape.
    strict : bool
        Raise ``ValueError`` for a mesh that is not closed (default). If
        False, warn with :class:`OpenMeshWarning` and return the parity.

    Returns
    -------
    inside : bool array of shape ``points.shape[:-1]``
        See the module docstring for the exact rule on the surface: a point
        is classified as if displaced by an infinitesimal ``(+e, +e**2,
        +e**3)``. Points with a NaN coordinate are outside.
    """
    pts = np.asarray(points, dtype=np.float64)
    if pts.shape[-1:] != (3,):
        raise ValueError(f"points must have shape (..., 3), got {pts.shape}")
    _check_closed(mesh, strict)
    flat = np.ascontiguousarray(pts.reshape(-1, 3))
    if len(flat) == 0 or mesh.n_faces == 0:
        return np.zeros(pts.shape[:-1], dtype=bool)
    out = _points_inside(flat, *_arrays(mesh.vertices, mesh.faces, mesh.grid_xy))
    return out.reshape(pts.shape[:-1])


def column_crossings(mesh: Mesh, xy, *, strict: bool = True) -> ColumnCrossings:
    """Crossing heights of the vertical lines through arbitrary plan positions.

    Parameters
    ----------
    xy : (n, 2) array_like
        Column positions (extra columns are ignored).

    Returns
    -------
    ColumnCrossings with ``shape == (n,)`` in the world frame. A point
    ``(x, y, z)`` is inside the solid exactly when ``z`` lies in one of the
    half-open intervals of its column (same rule as :func:`points_in_solid`).
    """
    xy = np.asarray(xy, dtype=np.float64)
    if xy.ndim != 2 or xy.shape[1] < 2:
        raise ValueError(f"xy must have shape (n, 2), got {xy.shape}")
    _check_closed(mesh, strict)
    xy = np.ascontiguousarray(xy[:, :2])
    if len(xy) == 0 or mesh.n_faces == 0:
        return ColumnCrossings((len(xy),), np.zeros(len(xy) + 1, dtype=np.int64), np.zeros(0))
    offsets, z = _columns_xy(xy, *_arrays(mesh.vertices, mesh.faces, mesh.grid_xy))
    return ColumnCrossings((len(xy),), offsets, z)


def _grid_spec(origin, spacing, shape, rotation):
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    spacing = np.asarray(spacing, dtype=np.float64).reshape(-1)
    if origin.shape != (3,) or spacing.shape != (3,):
        raise ValueError("origin and spacing must have 3 components")
    if not (np.isfinite(origin).all() and np.isfinite(spacing).all() and (spacing > 0).all()):
        raise ValueError("origin must be finite and spacing must be finite and > 0")
    shape = tuple(int(s) for s in shape)
    if len(shape) != 3 or min(shape) < 0:
        raise ValueError("shape must be three non-negative integers (nx, ny, nz)")
    if rotation is not None:
        rotation = np.asarray(rotation, dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(
            rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-9
        ):
            raise ValueError("rotation must be an orthonormal 3x3 matrix")
        if np.array_equal(rotation, np.eye(3)):
            rotation = None
    return origin, spacing, shape, rotation


def grid_points(origin, spacing, shape, rotation=None) -> np.ndarray:
    """(nx, ny, nz, 3) world coordinates of the points tested by :func:`grid_in_solid`.

    ``point[i, j, k] = origin + rotation @ (i * dx, j * dy, k * dz)``. Without
    a rotation these are exactly the float coordinates the grid path uses, so
    ``points_in_solid(mesh, grid_points(...))`` equals ``grid_in_solid(...)``
    bit for bit.
    """
    origin, spacing, shape, rotation = _grid_spec(origin, spacing, shape, rotation)
    axes = [np.arange(n) * d for n, d in zip(shape, spacing, strict=True)]
    if rotation is None:
        axes = [o + a for o, a in zip(origin, axes, strict=True)]
    out = np.empty((*shape, 3))
    out[..., 0], out[..., 1], out[..., 2] = np.meshgrid(*axes, indexing="ij")
    if rotation is not None:
        out = origin + out @ rotation.T
    return out


def grid_in_solid(
    mesh: Mesh,
    origin,
    spacing,
    shape,
    rotation=None,
    *,
    return_intervals: bool = False,
    strict: bool = True,
):
    """Which points of a regular (optionally rotated) grid are inside a closed mesh.

    The grid points are

        point[i, j, k] = origin + rotation @ (i * dx, j * dy, k * dz)

    so ``origin`` is the position of point ``(0, 0, 0)`` -- for a block model
    pass the centroid of the first block, not the model corner.

    Parameters
    ----------
    mesh : Mesh
        Closed surface.
    origin : (3,) float
        World position of grid point ``(0, 0, 0)``.
    spacing : (3,) float
        ``(dx, dy, dz)``, all > 0.
    shape : (nx, ny, nz)
    rotation : (3, 3) array_like, optional
        Orthonormal matrix whose columns are the world directions of the grid
        ``i``, ``j`` and ``k`` axes. The mesh is rotated into the grid frame
        (``local = (vertex - origin) @ rotation``); the rays then run along
        the grid ``k`` axis.
    return_intervals : bool
        Also return the :class:`ColumnCrossings` of the ``nx * ny`` columns.
    strict : bool
        As in :func:`points_in_solid`.

    Returns
    -------
    mask : (nx, ny, nz) bool
    crossings : ColumnCrossings, only if ``return_intervals``
        In the world frame without rotation (heights are world Z, ``z0 =
        origin[2]``), otherwise in the grid frame (heights are distances along
        the ``k`` axis from ``origin``, ``z0 = 0``).

    Without a rotation the result equals ``points_in_solid(mesh,
    grid_points(...))`` exactly. With a rotation it equals
    ``points_in_solid`` on the rotated mesh exactly, and can differ from a
    test of the rotated *points* against the original mesh only for points
    within rounding distance of the surface.
    """
    origin, spacing, shape, rotation = _grid_spec(origin, spacing, shape, rotation)
    _check_closed(mesh, strict)
    nx, ny, nz = shape
    if rotation is None:
        vertices, grid = mesh.vertices, mesh.grid_xy
        base = origin
        frame = "world"
    else:
        vertices = np.ascontiguousarray((mesh.vertices - origin) @ rotation)
        grid = TriangleGridXY(vertices, mesh.faces)
        base = np.zeros(3)
        frame = "grid"
    xs = base[0] + np.arange(nx) * spacing[0]
    ys = base[1] + np.arange(ny) * spacing[1]
    if mesh.n_faces == 0 or nx * ny == 0:
        offsets, z = np.zeros(nx * ny + 1, dtype=np.int64), np.zeros(0)
    else:
        offsets, z = _columns(xs, ys, *_arrays(vertices, mesh.faces, grid))
    mask = _fill(offsets, z, nx, ny, base[2] + np.arange(nz) * spacing[2])
    if return_intervals:
        return mask, ColumnCrossings((nx, ny), offsets, z, float(base[2]), float(spacing[2]), frame)
    return mask
