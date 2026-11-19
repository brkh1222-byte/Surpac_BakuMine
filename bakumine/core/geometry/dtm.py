"""Digital terrain models: triangulated surfaces with one Z per XY (research.md section 5.2).

Build
-----
:func:`dtm_from_points` triangulates scattered points in plan (2D Delaunay).
Strings can be honoured as *breaklines*: every breakline segment becomes an
edge of the triangulation (constrained Delaunay), so ridges, crests and toes
are not cut across. An optional closed boundary string clips the surface.
:func:`dtm_from_grid` builds a DTM from a regular grid of elevations.

Query
-----
:meth:`DTM.z_at` interpolates the surface at many ``(x, y)`` at once
(barycentric interpolation inside the triangle found through the plan-view
grid index); :meth:`DTM.height_above` and :meth:`DTM.above_below` compare
points with it.

The triangle under a point is chosen with the same exact rule as
point-in-solid (:mod:`bakumine.core.geometry.predicates`): the point is
treated as displaced by an infinitesimal ``(+e, +e**2)``, so a point on a
shared edge or vertex belongs to exactly one triangle and is never missed.
If the displaced point falls off the surface (the point is on the outer
border, on its +x / +y side), the triangle that has the point on its
boundary is used. The surface is therefore closed: border vertices and
edges return their Z, and only points strictly outside return NaN.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence

import numpy as np
from numba import njit, prange

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.predicates import tri_covers, tri_covers_closed, tri_z
from bakumine.core.geometry.spatial import _cell

__all__ = ["DTM", "DuplicatePointWarning", "dtm_from_grid", "dtm_from_points"]

_DUPLICATE_RULES = ("mean", "first", "last", "min", "max", "error")


class DuplicatePointWarning(UserWarning):
    """Input points share the same XY but have different Z."""


# --------------------------------------------------------------------------
# query kernel
# --------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _locate(x, y, v, f, x0, y0, invx, invy, nx, ny, start, items, lo, hi):
    n = x.shape[0]
    z = np.full(n, np.nan)
    tri = np.full(n, -1, dtype=np.int64)
    for i in prange(n):
        px = x[i]
        py = y[i]
        if not (px == px and py == py):
            continue
        c = _cell(px, x0, invx, nx) * ny + _cell(py, y0, invy, ny)
        hit = -1
        border = -1
        for k in range(start[c], start[c + 1]):
            t = items[k]
            if px < lo[t, 0] or px > hi[t, 0] or py < lo[t, 1] or py > hi[t, 1]:
                continue
            ia = f[t, 0]
            ib = f[t, 1]
            ic = f[t, 2]
            if tri_covers(v[ia, 0], v[ia, 1], v[ib, 0], v[ib, 1], v[ic, 0], v[ic, 1], px, py):
                hit = t
                break
            if border < 0 and tri_covers_closed(
                v[ia, 0], v[ia, 1], v[ib, 0], v[ib, 1], v[ic, 0], v[ic, 1], px, py
            ):
                border = t
        if hit < 0:
            hit = border
        if hit >= 0:
            ia = f[hit, 0]
            ib = f[hit, 1]
            ic = f[hit, 2]
            tri[i] = hit
            z[i] = tri_z(
                v[ia, 0], v[ia, 1], v[ia, 2],
                v[ib, 0], v[ib, 1], v[ib, 2],
                v[ic, 0], v[ic, 1], v[ic, 2],
                px, py,
            )  # fmt: skip
    return z, tri


# --------------------------------------------------------------------------
# DTM
# --------------------------------------------------------------------------


def _plan_cross(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Twice the signed plan area of every face (computed about the first corner)."""
    a, b, c = (vertices[faces[:, j], :2] for j in range(3))
    ab, ac = b - a, c - a
    return ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]


class DTM:
    """A triangulated surface with a single Z for each XY.

    Parameters
    ----------
    mesh : Mesh
        The triangles. Faces that are clockwise in plan are flipped so that
        all normals point up. A face with zero area in plan (vertical or
        degenerate) raises ``ValueError``: such a surface is not a DTM.

    The surface is assumed not to overlap itself in plan; this is not
    checked. Where triangles do overlap, the one with the lowest index among
    those found first in the grid cell is used.

    Attributes
    ----------
    mesh : Mesh
        Faces counter-clockwise in plan (normals up).
    """

    ABOVE = 1
    ON = 0
    BELOW = -1
    OUTSIDE = -128

    def __init__(self, mesh: Mesh):
        if mesh.n_faces:
            cross = _plan_cross(mesh.vertices, mesh.faces)
            if (cross == 0.0).any():
                n = int((cross == 0.0).sum())
                raise ValueError(
                    f"{n} face(s) have zero area in plan (vertical or degenerate): not a DTM"
                )
            if (cross < 0.0).any():
                mesh = mesh.flipped(np.flatnonzero(cross < 0.0))
        self.mesh = mesh

    def __repr__(self) -> str:
        return f"DTM(n_vertices={self.mesh.n_vertices}, n_faces={self.mesh.n_faces})"

    @property
    def bounds(self) -> np.ndarray:
        """(2, 3) bounding box of the surface."""
        return self.mesh.bounds

    @property
    def plan_area(self) -> float:
        """Area of the surface projected to plan."""
        return float(0.5 * np.abs(_plan_cross(self.mesh.vertices, self.mesh.faces)).sum())

    def _query(self, x, y):
        if y is None:
            xy = np.asarray(x, dtype=np.float64)
            if xy.shape[-1] < 2:
                raise ValueError("a single argument must have shape (..., 2) or (..., 3)")
            x, y = xy[..., 0], xy[..., 1]
        x, y = np.broadcast_arrays(np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64))
        shape = x.shape
        m = self.mesh
        if m.n_faces == 0 or x.size == 0:
            return np.full(shape, np.nan), np.full(shape, -1, dtype=np.int64)
        g = m.grid_xy
        z, tri = _locate(
            np.ascontiguousarray(x.reshape(-1)), np.ascontiguousarray(y.reshape(-1)),
            m.vertices, m.faces,
            g.origin[0], g.origin[1], g.inv_cell_size[0], g.inv_cell_size[1],
            g.shape[0], g.shape[1], g.cell_start, g.cell_tris, g.tri_lo, g.tri_hi,
        )  # fmt: skip
        return z.reshape(shape), tri.reshape(shape)

    def z_at(self, x, y=None) -> np.ndarray:
        """Surface elevation at plan positions.

        Parameters
        ----------
        x, y : array_like
            Broadcast against each other. Alternatively pass one array of
            shape ``(..., 2)`` or ``(..., 3)`` (a third column is ignored).

        Returns
        -------
        z : float64 array
            NaN strictly outside the surface (and for NaN input). At a vertex
            the result is exactly the vertex Z.
        """
        return self._query(x, y)[0]

    def triangle_at(self, x, y=None) -> np.ndarray:
        """Index of the face under each plan position, -1 outside (same rule as :meth:`z_at`)."""
        return self._query(x, y)[1]

    def height_above(self, points) -> np.ndarray:
        """``z - z_at(x, y)`` for points ``(..., 3)``: positive above the surface, NaN outside."""
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {p.shape}")
        return p[..., 2] - self._query(p[..., 0], p[..., 1])[0]

    def above_below(self, points) -> np.ndarray:
        """Side of the surface for points ``(..., 3)``.

        Returns an int8 array: ``DTM.ABOVE`` (1), ``DTM.BELOW`` (-1),
        ``DTM.ON`` (0, the point Z equals the interpolated Z exactly) and
        ``DTM.OUTSIDE`` (-128, no surface above or below the point, or NaN
        input).
        """
        h = self.height_above(points)
        out = np.full(h.shape, self.OUTSIDE, dtype=np.int8)
        out[h > 0.0] = self.ABOVE
        out[h < 0.0] = self.BELOW
        out[h == 0.0] = self.ON
        return out


# --------------------------------------------------------------------------
# construction
# --------------------------------------------------------------------------


def _as_xyz(a, name: str) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    if a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3), got {a.shape}")
    if not np.isfinite(a).all():
        raise ValueError(f"{name} contains NaN or infinite coordinates")
    return a


def _merge_duplicates(xyz: np.ndarray, rule: str):
    """Merge rows with identical XY. Returns ``(vertices, inverse)``."""
    xy, first, inverse, counts = np.unique(
        xyz[:, :2], axis=0, return_index=True, return_inverse=True, return_counts=True
    )
    inverse = inverse.reshape(-1)
    n = len(xy)
    z = xyz[:, 2]
    if len(xyz) == n:
        return np.column_stack([xy, z[first]]), inverse
    zmin = np.full(n, np.inf)
    zmax = np.full(n, -np.inf)
    np.minimum.at(zmin, inverse, z)
    np.maximum.at(zmax, inverse, z)
    conflict = zmax > zmin
    if conflict.any():
        k = int(conflict.sum())
        worst = float((zmax - zmin).max())
        message = (
            f"{k} plan position(s) have points with different Z (largest spread {worst:g} m)"
        )
        if rule == "error":
            raise ValueError(message)
        warnings.warn(
            f"{message}; using the {rule} Z at each", DuplicatePointWarning, stacklevel=3
        )
    if rule in ("mean", "error"):
        # mean about zmin: exact when all Z are equal
        zz = zmin + np.bincount(inverse, weights=z - zmin[inverse], minlength=n) / counts
    elif rule == "first":
        zz = z[first]
    elif rule == "last":
        last = np.zeros(n, dtype=np.int64)
        np.maximum.at(last, inverse, np.arange(len(z)))
        zz = z[last]
    elif rule == "min":
        zz = zmin
    else:
        zz = zmax
    return np.column_stack([xy, zz]), inverse


def _points_in_polygon(xy: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Even-odd test of points (n, 2) against a closed polygon (k, 2)."""
    inside = np.zeros(len(xy), dtype=bool)
    x, y = xy[:, 0], xy[:, 1]
    for a, b in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if a[1] == b[1]:
            continue
        straddle = (a[1] > y) != (b[1] > y)
        xc = a[0] + (y - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        inside ^= straddle & (x < xc)
    return inside


def _delaunay(xy: np.ndarray, segments: np.ndarray, z: np.ndarray):
    """Triangulate in plan. Returns ``(faces, extra_xy, extra_z)``.

    ``extra_*`` are vertices added where constraint segments cross each other.
    """
    if len(segments) == 0:
        from scipy.spatial import Delaunay, QhullError

        try:
            faces = Delaunay(xy).simplices.astype(np.int64)
        except QhullError as exc:
            raise ValueError("the points cannot be triangulated (all collinear?)") from exc
        if len(np.unique(faces)) == len(xy):
            return faces, np.zeros((0, 2)), np.zeros(0)
        # Qhull left some points out (nearly coincident or on the hull): use
        # the exact-arithmetic triangulator instead.
    import triangle

    data = {"vertices": xy, "vertex_attributes": z[:, None]}
    if len(segments):
        data["segments"] = segments.astype(np.int32)
    out = triangle.triangulate(data, "pc" if len(segments) else "c")
    if "triangles" not in out or len(out["triangles"]) == 0:
        raise ValueError("the points cannot be triangulated (all collinear?)")
    faces = np.asarray(out["triangles"], dtype=np.int64)
    n = len(xy)
    extra_xy = np.asarray(out["vertices"], dtype=np.float64)[n:]
    extra_z = np.asarray(out["vertex_attributes"], dtype=np.float64).reshape(-1)[n:]
    return faces, extra_xy, extra_z


def dtm_from_points(
    points,
    breaklines: Sequence | None = None,
    boundary=None,
    *,
    duplicates: str = "mean",
) -> DTM:
    """Build a DTM from scattered points, optional breaklines and an optional boundary.

    Parameters
    ----------
    points : (n, 3) array_like
        Surveyed points (east, north, up). May be empty if breaklines carry
        all the data.
    breaklines : sequence of (k, 3) array_like, optional
        Open strings. Their points become vertices and each of their
        segments becomes an edge of the triangulation (repeat the first point
        at the end for a closed string). Where two breaklines cross in plan a
        vertex is added at the crossing, with Z interpolated along one of
        them; a segment that passes exactly through another vertex is split
        there.
    boundary : (k, 3) or (k, 2) array_like, optional
        Closed polygon (do not repeat the first point; a repeated last point
        is dropped). Its edges become edges of the triangulation and only
        triangles inside it are kept. With three columns its points are
        surface points. With two columns the Z of each boundary point is
        interpolated from the unclipped surface of the other data, and a
        boundary point outside that surface raises ``ValueError``.
    duplicates : {"mean", "first", "last", "min", "max", "error"}
        What to do with points that have the same XY (exactly) and different
        Z, over points, breaklines and boundary together: the merged vertex
        gets the mean (default), first, last, lowest or highest Z and a
        :class:`DuplicatePointWarning` is emitted; ``"error"`` raises
        ``ValueError`` instead. Points equal in X, Y and Z are merged
        silently.

    Returns
    -------
    DTM
        Faces counter-clockwise in plan. Vertices not used by any triangle
        (outside the boundary) are removed. Vertex order is not the input
        order.
    """
    if duplicates not in _DUPLICATE_RULES:
        raise ValueError(f"duplicates must be one of {_DUPLICATE_RULES}")
    parts = [_as_xyz(np.reshape(points, (-1, 3)) if np.size(points) == 0 else points, "points")]
    seg_parts = []
    offset = len(parts[0])
    for line in breaklines or ():
        line = _as_xyz(line, "breakline")
        if len(line) >= 2:
            idx = offset + np.arange(len(line))
            seg_parts.append(np.column_stack([idx[:-1], idx[1:]]))
        parts.append(line)
        offset += len(line)

    poly = None
    if boundary is not None:
        b = np.asarray(boundary, dtype=np.float64)
        if b.ndim != 2 or b.shape[1] not in (2, 3) or not np.isfinite(b).all():
            raise ValueError("boundary must be a finite array of shape (k, 2) or (k, 3)")
        if len(b) > 1 and (b[0] == b[-1]).all():
            b = b[:-1]
        if len(b) < 3:
            raise ValueError("boundary needs at least 3 distinct points")
        if b.shape[1] == 2:
            base = dtm_from_points(points, breaklines, None, duplicates=duplicates)
            zb = base.z_at(b[:, 0], b[:, 1])
            if np.isnan(zb).any():
                raise ValueError(
                    f"{int(np.isnan(zb).sum())} boundary point(s) lie outside the data: "
                    "give the boundary Z values or shrink it"
                )
            b = np.column_stack([b, zb])
        idx = offset + np.arange(len(b))
        seg_parts.append(np.column_stack([idx, np.roll(idx, -1)]))
        parts.append(b)
        poly = b[:, :2]

    xyz = np.concatenate(parts)
    if len(xyz) < 3:
        raise ValueError("a DTM needs at least 3 points")
    vertices, inverse = _merge_duplicates(xyz, duplicates)
    if len(vertices) < 3:
        raise ValueError("a DTM needs at least 3 points with distinct XY")

    if seg_parts:
        segments = np.sort(inverse[np.concatenate(seg_parts)], axis=1)
        segments = np.unique(segments[segments[:, 0] != segments[:, 1]], axis=0)
    else:
        segments = np.zeros((0, 2), dtype=np.int64)

    # Triangulate about the centre of the data: full precision for the predicates.
    centre = 0.5 * (vertices[:, :2].min(axis=0) + vertices[:, :2].max(axis=0))
    local = vertices[:, :2] - centre
    faces, extra_xy, extra_z = _delaunay(local, segments, vertices[:, 2])
    if len(extra_xy):
        vertices = np.concatenate([vertices, np.column_stack([extra_xy + centre, extra_z])])
        local = vertices[:, :2] - centre

    cross = _plan_cross(local, faces)
    faces = faces[cross != 0.0]
    cross = cross[cross != 0.0]
    faces[cross < 0.0] = faces[cross < 0.0][:, ::-1]
    if poly is not None and len(faces):
        centroids = local[faces].mean(axis=1)
        faces = faces[_points_in_polygon(centroids, poly - centre)]
    if len(faces) == 0:
        raise ValueError("the triangulation is empty (collinear points or empty boundary)")
    return DTM(Mesh(vertices, faces, check=False).remove_unreferenced_vertices())


def dtm_from_grid(z, origin, spacing) -> DTM:
    """DTM from a regular grid of elevations.

    Parameters
    ----------
    z : (nx, ny) array_like
        ``z[i, j]`` is the elevation at ``(x0 + i * dx, y0 + j * dy)``. NaN
        marks a missing node; triangles with a missing corner are left out.
    origin : (x0, y0)
    spacing : (dx, dy), both > 0

    Each grid cell is split into two triangles along the diagonal from node
    ``(i, j)`` to node ``(i + 1, j + 1)``.
    """
    z = np.asarray(z, dtype=np.float64)
    if z.ndim != 2 or min(z.shape) < 2:
        raise ValueError(f"z must have shape (nx, ny) with nx, ny >= 2, got {z.shape}")
    x0, y0 = (float(c) for c in origin)
    dx, dy = (float(c) for c in spacing)
    if not (dx > 0 and dy > 0):
        raise ValueError("spacing must be > 0")
    nx, ny = z.shape
    xx, yy = np.meshgrid(x0 + np.arange(nx) * dx, y0 + np.arange(ny) * dy, indexing="ij")
    vertices = np.column_stack([xx.ravel(), yy.ravel(), z.ravel()])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij"))
    a = i * ny + j
    b = a + ny
    c = b + 1
    d = a + 1
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    ok = np.isfinite(vertices[:, 2])
    faces = faces[ok[faces].all(axis=1)]
    if len(faces) == 0:
        raise ValueError("no grid cell has four valid corners")
    vertices[~ok, 2] = 0.0  # unreferenced; removed below
    return DTM(Mesh(vertices, faces, check=False).remove_unreferenced_vertices())
