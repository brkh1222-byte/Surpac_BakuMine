"""Spatial indexes over the triangles of a mesh.

Two indexes, both built in O(m log m) or better and queried from numba kernels:

* :class:`TriangleGridXY` -- a uniform grid in plan (XY). Each cell lists the
  triangles whose XY bounding box touches it. This is the index for vertical
  rays and columns: point-in-solid along +Z, DTM ``z_at(x, y)``, block columns.
* :class:`TriangleBVH` -- a bounding volume hierarchy in 3D for general
  axis-aligned box queries: which triangles may touch this block, which pairs
  of triangles may intersect each other.

Both answer *candidate* queries on bounding boxes with exact, closed
semantics: a triangle is returned if and only if its axis-aligned bounding
box touches the query (shared boundary counts). The caller runs the exact
geometric test (ray/triangle, box/triangle, triangle/triangle) on the
candidates.

All batch queries return a pair of equal-length int64 arrays
``(query_index, triangle_index)`` sorted by query and then by triangle.
The raw arrays of each index are public and documented so that later numba
kernels can walk them directly without going through Python.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from bakumine.core.parallel import parallel_njit

__all__ = ["TriangleBVH", "TriangleGridXY", "triangle_bounds"]

_EMPTY_I64 = np.empty(0, dtype=np.int64)


def triangle_bounds(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-triangle bounding boxes ``(lo, hi)``, each (m, 3) float64."""
    tri = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    if len(tri) == 0:
        return np.zeros((0, 3)), np.zeros((0, 3))
    return np.ascontiguousarray(tri.min(axis=1)), np.ascontiguousarray(tri.max(axis=1))


# --------------------------------------------------------------------------
# Plan-view uniform grid
# --------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _cell(x, x0, inv, n):
    """Grid cell of coordinate ``x`` clamped to [0, n-1]. Monotonic in x."""
    t = math.floor((x - x0) * inv)
    if not (t >= 0.0):  # negative or NaN
        return 0
    if t >= n:
        return n - 1
    return int(t)


def _cell_np(x, x0, inv, n):
    """Vectorised twin of :func:`_cell` (same floating-point operations)."""
    t = np.floor((x - x0) * inv)
    t = np.where(t >= 0.0, t, 0.0)
    return np.minimum(t, n - 1).astype(np.int64)


@njit(cache=True)
def _grid_build(lo, hi, x0, y0, invx, invy, nx, ny):
    m = lo.shape[0]
    start = np.zeros(nx * ny + 1, dtype=np.int64)
    for t in range(m):
        ix0 = _cell(lo[t, 0], x0, invx, nx)
        ix1 = _cell(hi[t, 0], x0, invx, nx)
        iy0 = _cell(lo[t, 1], y0, invy, ny)
        iy1 = _cell(hi[t, 1], y0, invy, ny)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                start[ix * ny + iy + 1] += 1
    for c in range(nx * ny):
        start[c + 1] += start[c]
    items = np.empty(start[nx * ny], dtype=np.int64)
    cursor = start[:-1].copy()
    for t in range(m):
        ix0 = _cell(lo[t, 0], x0, invx, nx)
        ix1 = _cell(hi[t, 0], x0, invx, nx)
        iy0 = _cell(lo[t, 1], y0, invy, ny)
        iy1 = _cell(hi[t, 1], y0, invy, ny)
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                c = ix * ny + iy
                items[cursor[c]] = t
                cursor[c] += 1
    return start, items


@njit(cache=True, inline="always")
def _grid_point(x, y, x0, y0, invx, invy, nx, ny, start, items, lo, hi, out, pos, write):
    if not (x == x and y == y):  # NaN never matches
        return 0
    c = _cell(x, x0, invx, nx) * ny + _cell(y, y0, invy, ny)
    count = 0
    for k in range(start[c], start[c + 1]):
        t = items[k]
        if lo[t, 0] <= x and x <= hi[t, 0] and lo[t, 1] <= y and y <= hi[t, 1]:
            if write:
                out[pos + count] = t
            count += 1
    return count


@parallel_njit(cache=True)
def _grid_query_points(xy, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    n = xy.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int64)
    for i in prange(n):
        offsets[i + 1] = _grid_point(
            xy[i, 0], xy[i, 1], x0, y0, invx, invy, nx, ny, start, items, lo, hi, dummy, 0, False
        )
    for i in range(n):
        offsets[i + 1] += offsets[i]
    total = offsets[n]
    qi = np.empty(total, dtype=np.int64)
    ti = np.empty(total, dtype=np.int64)
    for i in prange(n):
        p = offsets[i]
        c = _grid_point(
            xy[i, 0], xy[i, 1], x0, y0, invx, invy, nx, ny, start, items, lo, hi, ti, p, True
        )
        for k in range(p, p + c):
            qi[k] = i
    return qi, ti


@njit(cache=True, inline="always")
def _grid_rect(
    ax, ay, bx, by, x0, y0, invx, invy, nx, ny, start, items, lo, hi, out, pos, write
):
    if not (ax <= bx and ay <= by):  # inverted or NaN
        return 0
    qx0 = _cell(ax, x0, invx, nx)
    qx1 = _cell(bx, x0, invx, nx)
    qy0 = _cell(ay, y0, invy, ny)
    qy1 = _cell(by, y0, invy, ny)
    count = 0
    for ix in range(qx0, qx1 + 1):
        for iy in range(qy0, qy1 + 1):
            c = ix * ny + iy
            for k in range(start[c], start[c + 1]):
                t = items[k]
                if lo[t, 0] > bx or hi[t, 0] < ax or lo[t, 1] > by or hi[t, 1] < ay:
                    continue
                # A triangle sits in several cells: report it only from the
                # first cell shared by the triangle and the query rectangle.
                fx = _cell(lo[t, 0], x0, invx, nx)
                fy = _cell(lo[t, 1], y0, invy, ny)
                if ix != max(fx, qx0) or iy != max(fy, qy0):
                    continue
                if write:
                    out[pos + count] = t
                count += 1
    return count


@parallel_njit(cache=True)
def _grid_query_rects(rlo, rhi, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    n = rlo.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int64)
    for i in prange(n):
        offsets[i + 1] = _grid_rect(
            rlo[i, 0], rlo[i, 1], rhi[i, 0], rhi[i, 1],
            x0, y0, invx, invy, nx, ny, start, items, lo, hi, dummy, 0, False,
        )  # fmt: skip
    for i in range(n):
        offsets[i + 1] += offsets[i]
    total = offsets[n]
    qi = np.empty(total, dtype=np.int64)
    ti = np.empty(total, dtype=np.int64)
    for i in prange(n):
        p = offsets[i]
        c = _grid_rect(
            rlo[i, 0], rlo[i, 1], rhi[i, 0], rhi[i, 1],
            x0, y0, invx, invy, nx, ny, start, items, lo, hi, ti, p, True,
        )  # fmt: skip
        ti[p : p + c].sort()
        for k in range(p, p + c):
            qi[k] = i
    return qi, ti


class TriangleGridXY:
    """Uniform plan-view (XY) grid over triangle bounding boxes.

    Parameters
    ----------
    vertices, faces
        Mesh arrays, (n, 3) float and (m, 3) int.
    cells_per_triangle : float
        Target number of grid cells per triangle (default 1). The grid is
        coarsened automatically if large triangles would make the cell lists
        much longer than the triangle count.

    Attributes (all usable from numba kernels)
    ------------------------------------------
    origin : (2,) float64
        XY of the lower-left grid corner (minimum of the triangle bounds).
    cell_size : (2,) float64
        Cell size in X and Y.
    inv_cell_size : (2,) float64
        ``1 / cell_size``; the cell of ``x`` is
        ``clamp(floor((x - origin[0]) * inv_cell_size[0]), 0, nx - 1)``.
    shape : (nx, ny)
        Number of cells in X and Y. Flat cell id is ``ix * ny + iy``.
    cell_start : (nx * ny + 1,) int64
        CSR row pointer: triangles of cell ``c`` are
        ``cell_tris[cell_start[c]:cell_start[c + 1]]`` (ascending).
    cell_tris : (e,) int64
        Triangle indices.
    tri_lo, tri_hi : (m, 3) float64
        Triangle bounding boxes.
    """

    def __init__(self, vertices, faces, *, cells_per_triangle: float = 1.0):
        self.tri_lo, self.tri_hi = triangle_bounds(vertices, faces)
        m = len(self.tri_lo)
        self.n_triangles = m
        if m == 0:
            lo2 = np.zeros(2)
            ext = np.zeros(2)
        else:
            lo2 = self.tri_lo[:, :2].min(axis=0)
            ext = self.tri_hi[:, :2].max(axis=0) - lo2
        target = max(1.0, m * float(cells_per_triangle))
        if ext[0] > 0 and ext[1] > 0:
            side = math.sqrt(ext[0] * ext[1] / target)
            nx, ny = math.ceil(ext[0] / side), math.ceil(ext[1] / side)
        elif ext[0] > 0:
            nx, ny = math.ceil(target), 1
        elif ext[1] > 0:
            nx, ny = 1, math.ceil(target)
        else:
            nx, ny = 1, 1
        cap = int(4 * target) + 1
        nx, ny = int(min(max(nx, 1), cap)), int(min(max(ny, 1), cap))

        # Coarsen while the cell lists are much longer than the triangle list.
        while True:
            cell = np.where(ext > 0, ext / np.array([nx, ny]), 1.0)
            inv = 1.0 / cell
            if m == 0 or nx * ny == 1:
                break
            sx = _cell_np(self.tri_hi[:, 0], lo2[0], inv[0], nx) - _cell_np(
                self.tri_lo[:, 0], lo2[0], inv[0], nx
            )
            sy = _cell_np(self.tri_hi[:, 1], lo2[1], inv[1], ny) - _cell_np(
                self.tri_lo[:, 1], lo2[1], inv[1], ny
            )
            if int(((sx + 1) * (sy + 1)).sum()) <= 16 * m + 64:
                break
            nx, ny = max(1, nx // 2), max(1, ny // 2)

        self.origin = lo2.astype(np.float64)
        self.cell_size = cell
        self.inv_cell_size = inv
        self.shape = (nx, ny)
        self.cell_start, self.cell_tris = _grid_build(
            self.tri_lo, self.tri_hi, lo2[0], lo2[1], inv[0], inv[1], nx, ny
        )

    def _args(self):
        return (
            self.origin[0], self.origin[1], self.inv_cell_size[0], self.inv_cell_size[1],
            self.shape[0], self.shape[1], self.cell_start, self.cell_tris, self.tri_lo, self.tri_hi,
        )  # fmt: skip

    def cell_of(self, x: float, y: float) -> int:
        """Flat id of the (clamped) cell containing ``(x, y)``."""
        nx, ny = self.shape
        ix = int(_cell_np(np.float64(x), self.origin[0], self.inv_cell_size[0], nx))
        iy = int(_cell_np(np.float64(y), self.origin[1], self.inv_cell_size[1], ny))
        return ix * ny + iy

    def query_points(self, xy) -> tuple[np.ndarray, np.ndarray]:
        """Triangles whose XY bounding box contains each point.

        These are the candidates for a vertical line through each point.

        Parameters
        ----------
        xy : (k, 2) array of float (extra columns are ignored)

        Returns
        -------
        point_index, triangle_index : (p,) int64 each
            All pairs with ``tri_lo[t, :2] <= xy[i] <= tri_hi[t, :2]``,
            sorted by point then triangle. NaN points match nothing.
        """
        xy = np.ascontiguousarray(np.asarray(xy, dtype=np.float64).reshape(-1, np.shape(xy)[-1]))
        if len(xy) == 0 or self.n_triangles == 0:
            return _EMPTY_I64.copy(), _EMPTY_I64.copy()
        return _grid_query_points(xy, *self._args())

    def query_point(self, x: float, y: float) -> np.ndarray:
        """Triangles whose XY bounding box contains ``(x, y)`` (ascending)."""
        return self.query_points(np.array([[x, y]], dtype=np.float64))[1]

    def query_rects(self, lo, hi) -> tuple[np.ndarray, np.ndarray]:
        """Triangles whose XY bounding box touches each rectangle.

        Parameters
        ----------
        lo, hi : (k, 2) arrays of float
            Lower-left and upper-right corners.

        Returns
        -------
        rect_index, triangle_index : (p,) int64 each
            Sorted by rectangle then triangle; each pair appears once.
        """
        lo = np.ascontiguousarray(np.asarray(lo, dtype=np.float64).reshape(-1, 2))
        hi = np.ascontiguousarray(np.asarray(hi, dtype=np.float64).reshape(-1, 2))
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have the same shape")
        if len(lo) == 0 or self.n_triangles == 0:
            return _EMPTY_I64.copy(), _EMPTY_I64.copy()
        return _grid_query_rects(lo, hi, *self._args())

    def query_rect(self, lo, hi) -> np.ndarray:
        """Triangles whose XY bounding box touches one rectangle (ascending)."""
        return self.query_rects(np.reshape(lo, (1, 2)), np.reshape(hi, (1, 2)))[1]


# --------------------------------------------------------------------------
# Bounding volume hierarchy
# --------------------------------------------------------------------------


def _spread_bits(x: np.ndarray) -> np.ndarray:
    """Spread the low 21 bits of each uint64 so two zero bits follow each bit."""
    x = x & np.uint64(0x1FFFFF)
    x = (x | (x << np.uint64(32))) & np.uint64(0x1F00000000FFFF)
    x = (x | (x << np.uint64(16))) & np.uint64(0x1F0000FF0000FF)
    x = (x | (x << np.uint64(8))) & np.uint64(0x100F00F00F00F00F)
    x = (x | (x << np.uint64(4))) & np.uint64(0x10C30C30C30C30C3)
    x = (x | (x << np.uint64(2))) & np.uint64(0x1249249249249249)
    return x


def _morton_order(points: np.ndarray) -> np.ndarray:
    """Indices that sort points along a 3D Morton (Z-order) curve."""
    lo = points.min(axis=0)
    ext = points.max(axis=0) - lo
    scale = np.where(ext > 0, (2**21 - 1) / np.where(ext > 0, ext, 1.0), 0.0)
    q = ((points - lo) * scale).astype(np.uint64)
    code = (
        _spread_bits(q[:, 0])
        | (_spread_bits(q[:, 1]) << np.uint64(1))
        | (_spread_bits(q[:, 2]) << np.uint64(2))
    )
    return np.argsort(code, kind="stable").astype(np.int64)


@njit(cache=True, inline="always")
def _bvh_visit(
    node_lo, node_hi, slo, shi, order, leaf_size, first_leaf,
    ax, ay, az, bx, by, bz, min_id, out, pos, write,
):  # fmt: skip
    """Collect triangles ``t > min_id`` whose box touches [a, b]. Returns the count."""
    m = order.shape[0]
    stack = np.empty(64, dtype=np.int64)
    stack[0] = 0
    sp = 1
    count = 0
    while sp > 0:
        sp -= 1
        node = stack[sp]
        if (
            node_lo[node, 0] > bx or node_hi[node, 0] < ax
            or node_lo[node, 1] > by or node_hi[node, 1] < ay
            or node_lo[node, 2] > bz or node_hi[node, 2] < az
        ):  # fmt: skip
            continue
        if node >= first_leaf:
            s = (node - first_leaf) * leaf_size
            e = min(s + leaf_size, m)
            for k in range(s, e):
                if (
                    slo[k, 0] > bx or shi[k, 0] < ax
                    or slo[k, 1] > by or shi[k, 1] < ay
                    or slo[k, 2] > bz or shi[k, 2] < az
                ):  # fmt: skip
                    continue
                t = order[k]
                if t > min_id:
                    if write:
                        out[pos + count] = t
                    count += 1
        else:
            stack[sp] = 2 * node + 1
            stack[sp + 1] = 2 * node + 2
            sp += 2
    return count


@parallel_njit(cache=True)
def _bvh_query(node_lo, node_hi, slo, shi, order, leaf_size, first_leaf, qlo, qhi, self_mode):
    n = qlo.shape[0]
    offsets = np.zeros(n + 1, dtype=np.int64)
    dummy = np.empty(0, dtype=np.int64)
    for i in prange(n):
        if not (qlo[i, 0] <= qhi[i, 0] and qlo[i, 1] <= qhi[i, 1] and qlo[i, 2] <= qhi[i, 2]):
            continue  # inverted or NaN box
        offsets[i + 1] = _bvh_visit(
            node_lo, node_hi, slo, shi, order, leaf_size, first_leaf,
            qlo[i, 0], qlo[i, 1], qlo[i, 2], qhi[i, 0], qhi[i, 1], qhi[i, 2],
            i if self_mode else -1, dummy, 0, False,
        )  # fmt: skip
    for i in range(n):
        offsets[i + 1] += offsets[i]
    total = offsets[n]
    qi = np.empty(total, dtype=np.int64)
    ti = np.empty(total, dtype=np.int64)
    for i in prange(n):
        p = offsets[i]
        c = offsets[i + 1] - p
        if c == 0:
            continue
        _bvh_visit(
            node_lo, node_hi, slo, shi, order, leaf_size, first_leaf,
            qlo[i, 0], qlo[i, 1], qlo[i, 2], qhi[i, 0], qhi[i, 1], qhi[i, 2],
            i if self_mode else -1, ti, p, True,
        )  # fmt: skip
        ti[p : p + c].sort()
        for k in range(p, p + c):
            qi[k] = i
    return qi, ti


class TriangleBVH:
    """Bounding volume hierarchy over triangle bounding boxes.

    Triangles are sorted along a Morton curve of their box centres, cut into
    leaves of ``leaf_size`` consecutive triangles, and the leaves are the
    bottom level of a complete binary tree stored in heap order. Works well
    for meshes that mix very large and very small triangles.

    Parameters
    ----------
    vertices, faces
        Mesh arrays, (n, 3) float and (m, 3) int.
    leaf_size : int
        Triangles per leaf (default 8).

    Attributes (all usable from numba kernels)
    ------------------------------------------
    node_lo, node_hi : (2 * n_leaves - 1, 3) float64
        Node boxes in heap order: children of node ``i`` are ``2i + 1`` and
        ``2i + 2``; the root is node 0. Empty padding nodes have
        ``lo = +inf, hi = -inf`` and never match.
    first_leaf : int
        Index of the first leaf node (``n_leaves - 1``; ``n_leaves`` is a
        power of two).
    order : (m,) int64
        Triangle indices in Morton order. Leaf ``j`` (node ``first_leaf + j``)
        holds ``order[j * leaf_size:(j + 1) * leaf_size]``.
    sorted_lo, sorted_hi : (m, 3) float64
        Triangle boxes in the same order as ``order``.
    tri_lo, tri_hi : (m, 3) float64
        Triangle boxes in face order.
    """

    def __init__(self, vertices, faces, *, leaf_size: int = 8):
        if leaf_size < 1:
            raise ValueError("leaf_size must be >= 1")
        self.tri_lo, self.tri_hi = triangle_bounds(vertices, faces)
        m = len(self.tri_lo)
        self.n_triangles = m
        self.leaf_size = int(leaf_size)
        if m:
            self.order = _morton_order(0.5 * (self.tri_lo + self.tri_hi))
        else:
            self.order = _EMPTY_I64.copy()
        self.sorted_lo = np.ascontiguousarray(self.tri_lo[self.order])
        self.sorted_hi = np.ascontiguousarray(self.tri_hi[self.order])

        used = max(1, -(-m // self.leaf_size))
        n_leaves = 1 << (used - 1).bit_length()
        self.first_leaf = n_leaves - 1
        node_lo = np.full((2 * n_leaves - 1, 3), np.inf)
        node_hi = np.full((2 * n_leaves - 1, 3), -np.inf)
        if m:
            starts = np.arange(0, m, self.leaf_size)
            node_lo[self.first_leaf : self.first_leaf + len(starts)] = np.minimum.reduceat(
                self.sorted_lo, starts, axis=0
            )
            node_hi[self.first_leaf : self.first_leaf + len(starts)] = np.maximum.reduceat(
                self.sorted_hi, starts, axis=0
            )
        width, first = n_leaves, self.first_leaf
        while width > 1:
            parent_first = (first - 1) // 2
            node_lo[parent_first:first] = np.minimum(
                node_lo[first : first + width : 2], node_lo[first + 1 : first + width : 2]
            )
            node_hi[parent_first:first] = np.maximum(
                node_hi[first : first + width : 2], node_hi[first + 1 : first + width : 2]
            )
            width, first = width // 2, parent_first
        self.node_lo = node_lo
        self.node_hi = node_hi

    def _args(self):
        return (
            self.node_lo, self.node_hi, self.sorted_lo, self.sorted_hi,
            self.order, self.leaf_size, self.first_leaf,
        )  # fmt: skip

    def query_boxes(self, lo, hi) -> tuple[np.ndarray, np.ndarray]:
        """Triangles whose bounding box touches each query box.

        Parameters
        ----------
        lo, hi : (k, 3) arrays of float
            Minimum and maximum corners of the query boxes.

        Returns
        -------
        box_index, triangle_index : (p,) int64 each
            Sorted by box then triangle. Boxes with ``lo > hi`` or NaN match
            nothing.
        """
        lo = np.ascontiguousarray(np.asarray(lo, dtype=np.float64).reshape(-1, 3))
        hi = np.ascontiguousarray(np.asarray(hi, dtype=np.float64).reshape(-1, 3))
        if lo.shape != hi.shape:
            raise ValueError("lo and hi must have the same shape")
        if len(lo) == 0 or self.n_triangles == 0:
            return _EMPTY_I64.copy(), _EMPTY_I64.copy()
        return _bvh_query(*self._args(), lo, hi, False)

    def query_box(self, lo, hi) -> np.ndarray:
        """Triangles whose bounding box touches one box (ascending)."""
        return self.query_boxes(np.reshape(lo, (1, 3)), np.reshape(hi, (1, 3)))[1]

    def self_overlaps(self) -> np.ndarray:
        """(p, 2) int64 pairs ``(i, j)``, ``i < j``, of triangles whose boxes touch.

        Sorted by ``i`` then ``j``. This is the broad phase for
        self-intersection tests; neighbouring triangles are always included.
        """
        if self.n_triangles == 0:
            return np.zeros((0, 2), dtype=np.int64)
        i, j = _bvh_query(*self._args(), self.tri_lo, self.tri_hi, True)
        return np.column_stack([i, j])
