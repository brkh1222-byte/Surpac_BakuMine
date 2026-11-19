"""Solid from sections: join closed outlines on parallel sections into a closed mesh.

A geologist digitises a closed string around the ore on each cross-section
(research.md sections 5.1 and 5.3). This module turns the stack of outlines
into a closed triangulated solid:

* :func:`triangulate_between` joins two rings with a band of triangles;
* :func:`triangulate_cap` closes a (possibly concave) ring with a flat cap;
* :func:`solid_from_sections` does the whole stack: ring matching between
  adjacent sections, bands, simple bifurcations, end caps or tapered ends,
  welding, outward orientation and validation;
* :func:`solid_from_strings` groups the segments of a string file into
  sections and calls :func:`solid_from_sections`;
* :func:`sections_from_mesh` cuts an existing solid into parallel sections
  (the inverse operation, used for tests and for re-sectioning).

Stitching method
----------------
The band between two rings ``P`` (m vertices) and ``Q`` (n vertices) is a
closed path on the ``m x n`` torus of vertex pairs: each step advances one
ring by one vertex and adds one triangle (Keppel 1975; Fuchs, Kedem and
Uselton 1977). Three ways to choose the path are available:

``"min_area"`` (default)
    Dynamic programme over ``(i, j)`` that minimises the total triangle
    area (the Fuchs-Kedem-Uselton criterion). It is the default because it
    is a global optimum: it cannot paint itself into a corner the way a
    greedy walk can (one ring used up early, the rest closed by a fan), and
    the cost is small: ``O(m n)`` per start vertex, compiled with numba
    (500 x 500 vertices take a few milliseconds).
``"min_span"``
    The same dynamic programme, minimising the sum of the lengths of the
    cross edges ("spans").
``"greedy"``
    The classic shortest-diagonal walk (Christiansen and Sederberg 1978):
    at each step add the triangle with the shorter new diagonal. ``O(m + n)``.

All three measure cost in a *normalised* frame, not in world coordinates:
both rings are expressed in the section frame ``(u, v, w)``, the centres of
their bounding boxes are moved to a common point and their in-plane extents
are scaled towards a common size (Christiansen and Sederberg's "unit square"
mapping, with the scale limited to a factor 3). This makes the result
independent of a lateral shift between sections (plunge) and of a change in
size (taper). The triangles themselves use the original vertices.

The dynamic programme needs a start pair ``(i0, j0)``. Several candidates are
tried (``start_candidates`` vertices spread around ``P``, each paired with
the nearest vertex of ``Q`` in the normalised frame) and the cheapest band
is kept; this is what removes twist when the two rings start at different
vertices. Winding is made consistent first: both rings are traversed
counter-clockwise about the section normal, whatever their stored direction.

Tie lines (pairs of vertices that must be joined by an edge) cut the path
into independent pieces, each solved on its own.

Limitations
-----------
* Rings must be simple (not self-crossing) when seen along the section normal.
* A ring inside another ring on the same section (a hole) is not supported
  and raises ``NotImplementedError``.
* Bifurcation is one ring to exactly two rings (or two to one). One to
  three or more, and many-to-many groups that cannot be reduced by dropping
  the weakest overlaps, raise :class:`SectionSolidError`.
* The method works per pair of sections. Strongly dissimilar outlines on
  adjacent sections (a "C" next to an "O", very different positions) can
  give a band that crosses itself or a neighbouring band; the validation
  step reports the pair of sections so that the strings can be fixed or
  tie lines added.

Everything is computed relative to a local origin, so mine coordinates of
order 1e6 lose no precision. Input vertices are copied unchanged into the
result. Coordinates are ``x = east, y = north, z = up``.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.spatial import cKDTree

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.section import Section, plane_frame, section
from bakumine.core.geometry.validation import ValidationReport
from bakumine.core.io.strings import StringFile

__all__ = [
    "METHODS",
    "SectionProblem",
    "SectionSolidError",
    "densify_ring",
    "resample_ring",
    "sections_from_mesh",
    "solid_from_sections",
    "solid_from_strings",
    "triangulate_between",
    "triangulate_cap",
]

METHODS = ("min_area", "min_span", "greedy")
_END_MODES = ("flat", "point", "taper")


# --------------------------------------------------------------------------
# Errors and diagnostics
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SectionProblem:
    """One defect of a solid, located between two sections.

    Attributes
    ----------
    kind : str
        ``"self_intersection"``, ``"open_edge"``, ``"nonmanifold_edge"``,
        ``"nonmanifold_vertex"``, ``"inconsistent_winding"``,
        ``"degenerate_face"``, ``"duplicate_face"``, or a construction step
        that failed: ``"band"``, ``"cap"``, ``"bifurcation"``, ``"matching"``,
        ``"section"``.
    sections : (int, int)
        Indices, in the caller's input order, of the two sections the defect
        lies between. Both are the same for a defect at one section (an end
        cap or a tapered end).
    positions : (float, float)
        Positions of those sections along the section normal (metres from
        the local origin).
    count : int
        Number of defective elements (faces, edges, face pairs).
    message : str
    """

    kind: str
    sections: tuple[int, int]
    positions: tuple[float, float]
    count: int
    message: str


class SectionSolidError(ValueError):
    """The sections cannot be made into a valid solid.

    Attributes
    ----------
    problems : list of SectionProblem
        Where the trouble is; empty for errors that are not tied to sections.
    mesh : Mesh or None
        The invalid mesh, when one could be built (for display).
    report : ValidationReport or None
    """

    def __init__(
        self,
        message: str,
        problems: Sequence[SectionProblem] = (),
        mesh: Mesh | None = None,
        report: ValidationReport | None = None,
    ):
        super().__init__(message)
        self.problems = list(problems)
        self.mesh = mesh
        self.report = report

    @property
    def section_pairs(self) -> list[tuple[int, int]]:
        """Distinct section pairs named by the problems, in order."""
        out: list[tuple[int, int]] = []
        for p in self.problems:
            if p.sections not in out:
                out.append(p.sections)
        return out


class _StepError(Exception):
    """Internal: a construction step failed; the caller adds the section pair."""


# --------------------------------------------------------------------------
# Ring helpers
# --------------------------------------------------------------------------


def _clean_ring(points, tol: float = 1e-9) -> np.ndarray:
    """(n, 3) float64 ring without the closing point and without repeated points."""
    p = np.array(points, dtype=np.float64)
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"a ring must have shape (n, 3), got {p.shape}")
    if not np.isfinite(p).all():
        raise ValueError("ring coordinates must be finite")
    if len(p) > 1:
        keep = np.ones(len(p), dtype=bool)
        keep[1:] = np.linalg.norm(np.diff(p, axis=0), axis=1) > tol
        p = p[keep]
    while len(p) > 1 and np.linalg.norm(p[-1] - p[0]) <= tol:
        p = p[:-1]
    if len(p) < 3:
        raise ValueError("a closed ring needs at least 3 distinct points")
    return p


def _newell(p: np.ndarray) -> np.ndarray:
    """Area vector of a closed 3D polygon (normal times area)."""
    q = p - p.mean(axis=0)
    return 0.5 * np.cross(q, np.roll(q, -1, axis=0)).sum(axis=0)


def _estimate_normal(rings: Sequence[np.ndarray]) -> np.ndarray:
    """Common normal of a set of roughly parallel rings (sign: largest component > 0)."""
    m = np.zeros((3, 3))
    for r in rings:
        a = _newell(r)
        length = np.linalg.norm(a)
        if length > 0:
            m += np.outer(a, a) / length
    if not np.trace(m) > 0:
        raise SectionSolidError("cannot find the section normal: all rings have zero area")
    _, vec = np.linalg.eigh(m)
    n = vec[:, -1]
    if n[np.argmax(np.abs(n))] < 0:
        n = -n
    return n / np.linalg.norm(n)


def _unit(normal) -> np.ndarray:
    return plane_frame(normal)[2]


def _signed_area(xy: np.ndarray) -> float:
    q = xy - xy[0]
    return 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))


def _polygon_centroid(xy: np.ndarray) -> np.ndarray:
    """Area centroid of a simple polygon; vertex mean if the area vanishes."""
    c0 = xy.mean(axis=0)
    q = xy - c0
    r = np.roll(q, -1, axis=0)
    w = q[:, 0] * r[:, 1] - r[:, 0] * q[:, 1]
    total = w.sum()
    if abs(total) <= 1e-300:
        return c0
    return c0 + ((q + r) * w[:, None]).sum(axis=0) / (3.0 * total)


def _points_in_polygon(pts: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Crossing-number test of (k, 2) points against a closed (n, 2) polygon."""
    x = pts[:, 0][:, None]
    y = pts[:, 1][:, None]
    x0 = poly[:, 0][None, :]
    y0 = poly[:, 1][None, :]
    x1 = np.roll(poly[:, 0], -1)[None, :]
    y1 = np.roll(poly[:, 1], -1)[None, :]
    straddle = (y0 > y) != (y1 > y)
    with np.errstate(divide="ignore", invalid="ignore"):
        xi = x0 + (y - y0) * (x1 - x0) / (y1 - y0)
    return (np.count_nonzero(straddle & (x < xi), axis=1) & 1).astype(bool)


def _densify(ring: np.ndarray, max_edge: float) -> tuple[np.ndarray, np.ndarray]:
    """Insert points so that no edge is longer than ``max_edge``.

    Returns the new ring and, for each new vertex, the index of the original
    vertex it copies (-1 for inserted points).
    """
    if not max_edge > 0:
        raise ValueError("max_edge must be > 0")
    nxt = np.roll(ring, -1, axis=0)
    length = np.linalg.norm(nxt - ring, axis=1)
    pieces = np.maximum(1, np.ceil(length / max_edge - 1e-9).astype(np.int64))
    if (pieces == 1).all():
        return ring, np.arange(len(ring))
    start = np.concatenate([[0], np.cumsum(pieces)[:-1]])
    total = int(pieces.sum())
    owner = np.repeat(np.arange(len(ring)), pieces)
    k = np.arange(total) - start[owner]
    t = (k / pieces[owner])[:, None]
    out = ring[owner] + t * (nxt[owner] - ring[owner])
    out[start] = ring
    src = np.full(total, -1, dtype=np.int64)
    src[start] = np.arange(len(ring))
    return out, src


def densify_ring(ring, max_edge: float) -> np.ndarray:
    """Ring with extra points so that no edge is longer than ``max_edge``.

    The original vertices are kept, in order, starting with the first.
    """
    return _densify(_clean_ring(ring), max_edge)[0]


def resample_ring(ring, n: int) -> np.ndarray:
    """``n`` points equally spaced by arc length along the ring, from its first vertex.

    The original vertices (other than the first) are in general not kept.
    """
    if n < 3:
        raise ValueError("n must be >= 3")
    p = _clean_ring(ring)
    closed = np.vstack([p, p[:1]])
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(closed, axis=0), axis=1))])
    target = np.arange(n) * (s[-1] / n)
    return np.column_stack([np.interp(target, s, closed[:, k]) for k in range(3)])


# --------------------------------------------------------------------------
# Tiling kernels
# --------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _area(p, q, r):
    ax = q[0] - p[0]
    ay = q[1] - p[1]
    az = q[2] - p[2]
    bx = r[0] - p[0]
    by = r[1] - p[1]
    bz = r[2] - p[2]
    cx = ay * bz - az * by
    cy = az * bx - ax * bz
    cz = ax * by - ay * bx
    return 0.5 * np.sqrt(cx * cx + cy * cy + cz * cz)


@njit(cache=True, inline="always")
def _dist(p, q):
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    dz = q[2] - p[2]
    return np.sqrt(dx * dx + dy * dy + dz * dz)


@njit(cache=True)
def _dp_tile(P, Q, span):
    """Cheapest monotone path from (0, 0) to (a, b); P has a + 1 points, Q has b + 1.

    Returns ``moves`` (0 = advance along P, 1 = advance along Q) and the cost.
    """
    a = P.shape[0] - 1
    b = Q.shape[0] - 1
    back = np.zeros((a + 1, b + 1), dtype=np.uint8)
    prev = np.zeros(b + 1)
    cur = np.zeros(b + 1)
    for j in range(1, b + 1):
        c = _dist(P[0], Q[j]) if span else _area(P[0], Q[j - 1], Q[j])
        prev[j] = prev[j - 1] + c
        back[0, j] = 1
    for i in range(1, a + 1):
        c = _dist(P[i], Q[0]) if span else _area(P[i - 1], P[i], Q[0])
        cur[0] = prev[0] + c
        for j in range(1, b + 1):
            if span:
                c0 = _dist(P[i], Q[j])
                c1 = c0
            else:
                c0 = _area(P[i - 1], P[i], Q[j])
                c1 = _area(P[i], Q[j - 1], Q[j])
            c0 += prev[j]
            c1 += cur[j - 1]
            if c0 <= c1:
                cur[j] = c0
            else:
                cur[j] = c1
                back[i, j] = 1
        tmp = prev
        prev = cur
        cur = tmp
    moves = np.empty(a + b, dtype=np.uint8)
    i = a
    j = b
    k = a + b
    while k > 0:
        k -= 1
        mv = back[i, j]
        moves[k] = mv
        if mv == 0:
            i -= 1
        else:
            j -= 1
    return moves, prev[b]


@njit(cache=True)
def _greedy_tile(P, Q):
    """Shortest-diagonal walk from (0, 0) to (a, b); cost is the band area."""
    a = P.shape[0] - 1
    b = Q.shape[0] - 1
    moves = np.empty(a + b, dtype=np.uint8)
    i = 0
    j = 0
    cost = 0.0
    for k in range(a + b):
        if i == a:
            mv = 1
        elif j == b:
            mv = 0
        elif _dist(P[i + 1], Q[j]) <= _dist(P[i], Q[j + 1]):
            mv = 0
        else:
            mv = 1
        moves[k] = mv
        if mv == 0:
            cost += _area(P[i], P[i + 1], Q[j])
            i += 1
        else:
            cost += _area(P[i], Q[j], Q[j + 1])
            j += 1
    return moves, cost


def _tile(na, nb, i0, j0, a, b, method):
    """Tile from pair (i0, j0), advancing ``a`` steps on ring A and ``b`` on ring B."""
    P = np.ascontiguousarray(na[(i0 + np.arange(a + 1)) % len(na)])
    Q = np.ascontiguousarray(nb[(j0 + np.arange(b + 1)) % len(nb)])
    if method == "greedy":
        return _greedy_tile(P, Q)
    return _dp_tile(P, Q, method == "min_span")


def _normalise_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map two rings in frame coordinates (u, v, w) to a common centre and size."""
    lo_a, hi_a = a[:, :2].min(axis=0), a[:, :2].max(axis=0)
    lo_b, hi_b = b[:, :2].min(axis=0), b[:, :2].max(axis=0)
    ext_a, ext_b = hi_a - lo_a, hi_b - lo_b
    target = 0.5 * (ext_a + ext_b)
    floor = 1e-12 * max(float(target.max()), 1e-300)
    out = []
    for ring, lo, hi, ext in ((a, lo_a, hi_a, ext_a), (b, lo_b, hi_b, ext_b)):
        scale = np.where(ext > floor, target / np.maximum(ext, floor), 1.0)
        scale = np.clip(scale, 1.0 / 3.0, 3.0)
        n = ring.copy()
        n[:, :2] = (ring[:, :2] - 0.5 * (lo + hi)) * scale
        out.append(n)
    return out[0], out[1]


def _order_ties(ties, m: int, n: int):
    """Sort tie pairs around the rings; raise if they cross."""
    uniq = sorted(set(ties))
    i0, j0 = uniq[0]
    rel = sorted(((i - i0) % m, (j - j0) % n) for i, j in uniq)
    for (da0, db0), (da1, db1) in zip(rel[:-1], rel[1:], strict=True):
        if db1 < db0:
            raise _StepError(
                "tie lines cross each other: they must join the two rings in the same order"
            )
        if da1 == da0 and db1 == db0:
            raise _StepError("duplicate tie line")
    return i0, j0, rel


def _band(A, B, frame, method: str, ties, n_start: int) -> np.ndarray:
    """Triangles of the band between rings A (m, 3) and B (n, 3).

    Coordinates are local. ``ties`` is a list of ``(ia, ib)`` index pairs.
    Returns (m + n, 3) vertex indices into ``concatenate([A, B])``.
    """
    u, v, w = frame
    basis = np.column_stack([u, v, w])
    a = A @ basis
    b = B @ basis
    m, n = len(a), len(b)
    area_a = _signed_area(a[:, :2])
    area_b = _signed_area(b[:, :2])
    size = max(float(np.ptp(a[:, :2], axis=0).max()), float(np.ptp(b[:, :2], axis=0).max()))
    for area in (area_a, area_b):
        if not abs(area) > 1e-12 * size * size:
            raise _StepError(
                "a ring has no area seen along the section normal "
                "(collinear points, a figure of eight, or a ring that is not on a section)"
            )
    ia = np.arange(m) if area_a > 0 else np.arange(m)[::-1]
    ib = np.arange(n) if area_b > 0 else np.arange(n)[::-1]
    na, nb = _normalise_pair(a[ia], b[ib])

    if ties:
        pos_a = np.empty(m, dtype=np.int64)
        pos_a[ia] = np.arange(m)
        pos_b = np.empty(n, dtype=np.int64)
        pos_b[ib] = np.arange(n)
        i0, j0, rel = _order_ties([(int(pos_a[i]), int(pos_b[j])) for i, j in ties], m, n)
        rel = [*rel, (m, n)]
        parts = []
        for (da0, db0), (da1, db1) in zip(rel[:-1], rel[1:], strict=True):
            parts.append(_tile(na, nb, i0 + da0, j0 + db0, da1 - da0, db1 - db0, method)[0])
        moves = np.concatenate(parts)
    else:
        k = max(1, min(int(n_start), m))
        starts = np.unique(np.round(np.linspace(0, m, k, endpoint=False)).astype(np.int64) % m)
        best = None
        for i in starts:
            j = int(np.argmin(((nb - na[i]) ** 2).sum(axis=1)))
            mv, cost = _tile(na, nb, int(i), j, m, n, method)
            if best is None or cost < best[0]:
                best = (cost, mv, int(i), j)
        _, moves, i0, j0 = best

    step_a = moves == 0
    ci = i0 + np.concatenate([[0], np.cumsum(step_a)[:-1]])
    cj = j0 + np.concatenate([[0], np.cumsum(~step_a)[:-1]])
    p0 = ia[ci % m]
    p1 = ia[(ci + 1) % m]
    q0 = m + ib[cj % n]
    q1 = m + ib[(cj + 1) % n]
    return np.where(
        step_a[:, None], np.column_stack([p0, p1, q0]), np.column_stack([p0, q1, q0])
    ).astype(np.int64)


def _check_method(method: str) -> None:
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def triangulate_between(
    ring_a,
    ring_b,
    *,
    normal=None,
    method: str = "min_area",
    tie_lines: Iterable[tuple[int, int]] | None = None,
    max_edge: float | None = None,
    start_candidates: int = 8,
) -> Mesh:
    """Join two closed 3D polylines with a band of triangles.

    Parameters
    ----------
    ring_a, ring_b : (m, 3), (n, 3) array_like
        Closed rings; the closing point may be repeated or not. They need
        not be planar, may have different vertex counts, any start vertex
        and either winding.
    normal : (3,) array_like, optional
        Section direction, used to make the two windings consistent and to
        centre the rings on each other. Default: the common normal of the
        two rings (from their area vectors).
    method : {"min_area", "min_span", "greedy"}
        See the module docstring. Default ``"min_area"`` (dynamic programme).
    tie_lines : iterable of (int, int), optional
        Pairs ``(i, j)``: vertex ``i`` of ``ring_a`` must be joined to
        vertex ``j`` of ``ring_b`` by an edge. Indices refer to the rings as
        returned in the mesh when ``max_edge`` is None, else to the input
        rings (without a repeated closing point). Tie lines must not cross.
    max_edge : float, optional
        Densify both rings first so that no ring edge is longer than this.
    start_candidates : int
        Number of start pairs tried when there are no tie lines.

    Returns
    -------
    Mesh
        Open band with ``m + n`` faces. Its vertices are ``ring_a`` followed
        by ``ring_b`` (after cleaning and densifying), in their input order.
        The winding is consistent; which side is "out" is not defined for a
        band.
    """
    _check_method(method)
    a = _clean_ring(ring_a)
    b = _clean_ring(ring_b)
    ties = [(int(i), int(j)) for i, j in tie_lines] if tie_lines is not None else []
    for i, j in ties:
        if not (0 <= i < len(a) and 0 <= j < len(b)):
            raise ValueError(f"tie line ({i}, {j}) is out of range")
    if max_edge is not None:
        a, src_a = _densify(a, max_edge)
        b, src_b = _densify(b, max_edge)
        new_a = np.flatnonzero(src_a >= 0)
        new_b = np.flatnonzero(src_b >= 0)
        ties = [(int(new_a[i]), int(new_b[j])) for i, j in ties]
    origin = 0.5 * (a.mean(axis=0) + b.mean(axis=0))
    la, lb = a - origin, b - origin
    n = _estimate_normal([la, lb]) if normal is None else _unit(normal)
    try:
        faces = _band(la, lb, plane_frame(n), method, ties, start_candidates)
    except _StepError as exc:
        raise SectionSolidError(str(exc)) from None
    return Mesh(np.vstack([a, b]), faces)


# --------------------------------------------------------------------------
# Caps
# --------------------------------------------------------------------------


def _cap(local: np.ndarray, frame) -> np.ndarray:
    """Constrained Delaunay triangles (k - 2, 3) of a ring in local coordinates."""
    import triangle

    k = len(local)
    if k == 3:
        return np.array([[0, 1, 2]], dtype=np.int64)
    u, v, _ = frame
    xy = np.column_stack([local @ u, local @ v])
    xy = xy - 0.5 * (xy.min(axis=0) + xy.max(axis=0))
    scale = float(np.abs(xy).max())
    if not scale > 0:
        raise _StepError("the outline has no extent in its plane")
    segments = np.column_stack([np.arange(k), (np.arange(k) + 1) % k]).astype(np.int32)
    out = triangle.triangulate({"vertices": xy / scale, "segments": segments}, "pQ")
    tris = np.asarray(out.get("triangles", np.zeros((0, 3))), dtype=np.int64).reshape(-1, 3)
    if len(out["vertices"]) != k or len(tris) != k - 2:
        raise _StepError(
            "the outline cannot be capped: it crosses itself or repeats a point "
            "when projected on its plane"
        )
    return tris


def triangulate_cap(ring, normal=None) -> np.ndarray:
    """Triangulate the inside of a closed, roughly planar ring.

    The ring is projected on its best-fit plane (or on the plane with the
    given ``normal``) and triangulated by constrained Delaunay with the ring
    edges as constraints, so a concave outline gets no triangle outside it.
    No vertex is added: the triangles index the ring's own 3D vertices.

    Parameters
    ----------
    ring : (k, 3) array_like
        Closed ring without repeated points (a repeated closing point is
        dropped). It must be simple in the projection.
    normal : (3,) array_like, optional

    Returns
    -------
    (k - 2, 3) int64
        Vertex indices into the ring.
    """
    p = _clean_ring(ring)
    local = p - p.mean(axis=0)
    n = _estimate_normal([local]) if normal is None else _unit(normal)
    try:
        return _cap(local, plane_frame(n))
    except _StepError as exc:
        raise SectionSolidError(str(exc)) from None


# --------------------------------------------------------------------------
# Section input
# --------------------------------------------------------------------------


def _rings_of(item, index: int) -> list[np.ndarray]:
    """Rings of one section given as a Section, one ring, or a list of rings."""
    if isinstance(item, Section):
        if not all(item.closed):
            raise SectionSolidError(
                f"section {index} has an open polyline; only closed rings can make a solid"
            )
        raw = list(item.polylines)
    else:
        try:
            arr = np.asarray(item, dtype=np.float64)
        except (ValueError, TypeError):
            arr = None
        if arr is not None and arr.ndim == 2:
            raw = [arr] if arr.size else []
        else:
            raw = list(item)
    rings = []
    for r, ring in enumerate(raw):
        try:
            rings.append(_clean_ring(ring))
        except ValueError as exc:
            raise SectionSolidError(f"section {index}, ring {r}: {exc}") from None
    return rings


def _ring_radius(xy: np.ndarray, centre: np.ndarray) -> float:
    return float(np.linalg.norm(xy - centre, axis=1).max())


def _match(polys_a, polys_b, reach_limit=None) -> tuple[list[tuple[int, int]], str | None]:
    """Link rings of two adjacent sections by overlap in the section plane.

    Returns the links and, if a group of rings could not be reduced to
    one-to-one or one-to-two, a message.
    """
    p, q = len(polys_a), len(polys_b)
    score = np.zeros((p, q))
    box_a = [(r.min(axis=0), r.max(axis=0)) for r in polys_a]
    box_b = [(r.min(axis=0), r.max(axis=0)) for r in polys_b]
    for i in range(p):
        for j in range(q):
            if (box_a[i][0] > box_b[j][1]).any() or (box_b[j][0] > box_a[i][1]).any():
                continue
            fa = _points_in_polygon(polys_a[i], polys_b[j]).mean()
            fb = _points_in_polygon(polys_b[j], polys_a[i]).mean()
            score[i, j] = max(fa, fb)
    cen_a = [_polygon_centroid(r) for r in polys_a]
    cen_b = [_polygon_centroid(r) for r in polys_b]
    # Rings with no overlap at all (a thin, plunging body): pair orphans whose
    # bounding circles overlap, nearest first.
    free_a = [i for i in range(p) if not score[i].any()]
    free_b = [j for j in range(q) if not score[:, j].any()]
    pairs = []
    for i in free_a:
        for j in free_b:
            d = float(np.linalg.norm(cen_a[i] - cen_b[j]))
            if reach_limit is not None:
                reach = reach_limit
            else:
                reach = _ring_radius(polys_a[i], cen_a[i]) + _ring_radius(polys_b[j], cen_b[j])
            if d <= reach:
                pairs.append((d, i, j))
    used_a: set[int] = set()
    used_b: set[int] = set()
    for _, i, j in sorted(pairs):
        if i not in used_a and j not in used_b:
            score[i, j] = 1e-9
            used_a.add(i)
            used_b.add(j)

    linked = score > 0
    # Reduce many-to-many groups by dropping their weakest link.
    while True:
        deg_a = linked.sum(axis=1)
        deg_b = linked.sum(axis=0)
        bad = linked & (deg_a[:, None] > 1) & (deg_b[None, :] > 1)
        if not bad.any():
            break
        cand = np.where(bad, score, np.inf)
        i, j = np.unravel_index(int(np.argmin(cand)), cand.shape)
        linked[i, j] = False
    message = None
    if (linked.sum(axis=1) > 2).any() or (linked.sum(axis=0) > 2).any():
        message = "one ring corresponds to more than two rings on the next section"
    return [(int(i), int(j)) for i, j in zip(*np.nonzero(linked), strict=True)], message


def _split_chord(parent: np.ndarray, child_1: np.ndarray, child_2: np.ndarray):
    """Where the dividing line between two children cuts the parent ring.

    All arguments are 2D polygons in the section plane. The dividing line is
    the perpendicular bisector of the closest pair of points of the two
    children. Returns ``((edge_a, t_a), (edge_b, t_b))``: the two cut points
    as edge index and parameter along the edge.
    """
    dist, idx = cKDTree(child_2).query(child_1)
    k = int(np.argmin(dist))
    p1, p2 = child_1[k], child_2[int(idx[k])]
    gap = p2 - p1
    size = float(np.ptp(parent, axis=0).max())
    if np.linalg.norm(gap) <= 1e-9 * size:
        gap = _polygon_centroid(child_2) - _polygon_centroid(child_1)
    if not np.linalg.norm(gap) > 0:
        raise _StepError("the two rings of the bifurcation coincide")
    d = gap / np.linalg.norm(gap)
    along = np.array([-d[1], d[0]])
    mid = 0.5 * (p1 + p2)
    s = (parent - mid) @ d
    side = s > 0
    edges = np.flatnonzero(side != np.roll(side, -1))
    if len(edges) < 2:
        raise _StepError("the dividing line between the two rings does not cross the parent ring")
    nxt = (edges + 1) % len(parent)
    t = s[edges] / (s[edges] - s[nxt])
    points = parent[edges] + t[:, None] * (parent[nxt] - parent[edges])
    tau = (points - mid) @ along
    order = np.argsort(tau)
    below = int(np.searchsorted(tau[order], 0.0))
    if below % 2 == 1:
        lo = below - 1
    else:  # midpoint outside the parent: take the nearest stretch inside it
        starts = np.arange(0, len(order) - 1, 2)
        gaps = np.minimum(np.abs(tau[order][starts]), np.abs(tau[order][starts + 1]))
        lo = int(starts[np.argmin(gaps)])
    ea, eb = order[lo], order[lo + 1]
    return (int(edges[ea]), float(t[ea])), (int(edges[eb]), float(t[eb]))


def _refine(local: np.ndarray, requests: list[tuple[int, float]]):
    """Insert points on ring edges.

    ``requests`` are ``(edge, t)``. Returns the new ring, the source index of
    each new vertex (-1 if inserted) and the new index of each request. A
    request close to an existing vertex is snapped to it.
    """
    n = len(local)
    snap = 1e-6
    targets: list[tuple] = []
    for edge, t in requests:
        if t <= snap:
            targets.append(("v", edge % n))
        elif t >= 1.0 - snap:
            targets.append(("v", (edge + 1) % n))
        else:
            targets.append(("e", edge % n, t))
    inserts: dict[int, list[float]] = {}
    for tg in targets:
        if tg[0] == "e":
            ts = inserts.setdefault(tg[1], [])
            if not any(abs(t - tg[2]) <= 1e-9 for t in ts):
                ts.append(tg[2])
    for ts in inserts.values():
        ts.sort()
    pts = []
    src = []
    new_of_vertex = np.empty(n, dtype=np.int64)
    new_of_insert: dict[tuple[int, float], int] = {}
    for i in range(n):
        new_of_vertex[i] = len(pts)
        pts.append(local[i])
        src.append(i)
        for t in inserts.get(i, ()):
            new_of_insert[(i, t)] = len(pts)
            pts.append(local[i] + t * (local[(i + 1) % n] - local[i]))
            src.append(-1)
    index = []
    for tg in targets:
        if tg[0] == "v":
            index.append(int(new_of_vertex[tg[1]]))
        else:
            t = next(t for t in inserts[tg[1]] if abs(t - tg[2]) <= 1e-9)
            index.append(new_of_insert[(tg[1], t)])
    return np.array(pts), np.array(src, dtype=np.int64), index


def _arc(ids: np.ndarray, start: int, stop: int) -> np.ndarray:
    """Ring entries from ``start`` to ``stop`` inclusive, going forward cyclically."""
    k = len(ids)
    count = (stop - start) % k + 1
    return ids[(start + np.arange(count)) % k]


# --------------------------------------------------------------------------
# solid_from_sections
# --------------------------------------------------------------------------


def _end_modes(ends) -> tuple[str, str]:
    pair = (ends, ends) if isinstance(ends, str) else tuple(ends)
    if len(pair) != 2 or any(e not in _END_MODES for e in pair):
        raise ValueError(f"ends must be one of {_END_MODES} or a pair of them, got {ends!r}")
    return pair


class _Builder:
    """Accumulates vertices (local and world) and faces."""

    def __init__(self, origin: np.ndarray):
        self.origin = origin
        self.local: list[np.ndarray] = []
        self.world: list[np.ndarray] = []
        self.faces: list[np.ndarray] = []
        self.count = 0

    def add(self, local: np.ndarray, world: np.ndarray | None = None) -> np.ndarray:
        local = np.atleast_2d(local)
        self.local.append(local)
        self.world.append(local + self.origin if world is None else world)
        ids = np.arange(self.count, self.count + len(local))
        self.count += len(local)
        return ids

    def coords(self) -> np.ndarray:
        if len(self.local) > 1:
            self.local = [np.vstack(self.local)]
        return self.local[0]


def solid_from_sections(
    sections,
    *,
    normal=None,
    method: str = "min_area",
    ends="flat",
    end_distance: float | None = None,
    taper_scale: float = 0.5,
    tie_lines=None,
    tie_tolerance: float = 1e-6,
    max_edge: float | None = None,
    bifurcation_lift: float = 0.5,
    match_distance: float | None = None,
    start_candidates: int = 8,
    validate: bool = True,
    check_self_intersection: bool = True,
    weld_tolerance: float = 1e-9,
) -> Mesh:
    """Build a closed solid from closed outlines on successive sections.

    Parameters
    ----------
    sections : sequence
        One item per section. An item is a :class:`Section` (from
        :func:`bakumine.core.geometry.section.section`), one ring as an
        ``(n, 3)`` array, or a list of rings. Rings are closed; the closing
        point may be repeated or not. The order of the items does not
        matter: sections are sorted by their position along ``normal``.
    normal : (3,) array_like, optional
        Section direction. Default: the normal of the first item if it is a
        :class:`Section`, else the common normal of all rings.
    method : {"min_area", "min_span", "greedy"}
        Band stitching method (module docstring). Default ``"min_area"``.
    ends : {"flat", "point", "taper"} or a pair of them
        How rings without a partner are closed. ``"flat"``: a cap in the
        ring's own plane. ``"point"``: a cone to a point above the ring's
        centroid, ``end_distance`` beyond the section. ``"taper"``: a copy
        of the ring shrunk by ``taper_scale`` about its centroid,
        ``end_distance`` beyond the section, with a flat cap. A pair gives
        the modes for ends facing down and up the normal (first and last
        section, and limbs that stop in between).
    end_distance : float, optional
        Default: half the spacing to the neighbouring section.
    taper_scale : float
        Size of the shrunken outline for ``"taper"`` (0 < scale <= 1).
    tie_lines : (t, 2, 3) array_like, optional
        Pairs of 3D points. Each point is snapped to the nearest ring vertex
        (within ``tie_tolerance``); the two vertices must lie on rings that
        are joined together, and the band then contains the edge between them.
    max_edge : float, optional
        Densify every ring first so that no ring edge is longer than this.
    bifurcation_lift : float
        Where one ring splits into two, the parent ring is divided by a
        bridge. Its interior points are raised towards the two children by
        this fraction of the section spacing (0 keeps the bridge in the
        parent's plane, 0.5 puts the saddle half-way).
    match_distance : float, optional
        Rings on adjacent sections are joined when they overlap seen along
        the normal. Rings that overlap nothing (a thin, plunging body) are
        joined to the nearest such ring on the next section if their
        centroids, seen along the normal, are at most this far apart.
        Default: the sum of the two rings' radii (largest distance from the
        centroid to a vertex).
    start_candidates : int
        Start pairs tried per band.
    validate : bool
        Check the result with :func:`validate` and raise
        :class:`SectionSolidError` if it is not a valid solid.
    check_self_intersection : bool
        Include the triangle/triangle test (the slow part) in the validation.
    weld_tolerance : float
        Tolerance (metres) of the final vertex weld.

    Returns
    -------
    Mesh
        Closed, welded, outward-oriented solid. Input vertices are kept
        bit-for-bit.

    Raises
    ------
    NotImplementedError
        A ring lies inside another ring on the same section (a hole).
    SectionSolidError
        The input is unusable or the result is not a valid solid.
        ``error.problems`` names the pair(s) of sections at fault (indices
        into ``sections`` as given), ``error.mesh`` is the invalid mesh and
        ``error.report`` its :class:`ValidationReport`.
    """
    _check_method(method)
    end_modes = _end_modes(ends)
    if not 0 < taper_scale <= 1:
        raise ValueError("taper_scale must be in (0, 1]")
    if end_distance is not None and not end_distance > 0:
        raise ValueError("end_distance must be > 0")

    items = list(sections)
    parsed = [(i, _rings_of(item, i)) for i, item in enumerate(items)]
    parsed = [(i, rings) for i, rings in parsed if rings]
    if len(parsed) < 2:
        raise SectionSolidError("at least two sections with rings are needed to make a solid")
    all_points = np.vstack([r for _, rings in parsed for r in rings])
    origin = np.round(0.5 * (all_points.min(axis=0) + all_points.max(axis=0)))

    if normal is None:
        first = next((it for it in items if isinstance(it, Section)), None)
        if first is not None:
            n = first.normal
        else:
            n = _estimate_normal([r - origin for _, rings in parsed for r in rings])
    else:
        n = _unit(normal)
    frame = plane_frame(n)
    u, v, n = frame
    uv = np.column_stack([u, v])

    position = np.array(
        [np.mean([((r - origin) @ n).mean() for r in rings]) for _, rings in parsed]
    )
    order = np.argsort(position, kind="stable")
    parsed = [parsed[k] for k in order]
    position = position[order]
    label = [i for i, _ in parsed]
    n_sec = len(parsed)
    extent = float(np.linalg.norm(np.ptp(all_points, axis=0)))

    def problem(kind: str, ka: int, kb: int, message: str, count: int = 1) -> SectionProblem:
        return SectionProblem(
            kind, (label[ka], label[kb]), (float(position[ka]), float(position[kb])), count, message
        )

    def fail(kind: str, ka: int, kb: int, message: str):
        where = (
            f"section {label[ka]}" if ka == kb else f"between sections {label[ka]} and {label[kb]}"
        )
        text = f"{where} (position {position[ka]:.3f}"
        text += f" to {position[kb]:.3f} m): " if ka != kb else " m): "
        return SectionSolidError(text + message, [problem(kind, ka, kb, message)])

    for k in range(n_sec - 1):
        if not position[k + 1] - position[k] > 1e-9 * max(extent, 1.0):
            raise fail(
                "section",
                k,
                k + 1,
                "the two sections are at the same position along the section normal",
            )

    # Rings in local coordinates, with the world coordinates of every vertex.
    ring_local: list[list[np.ndarray]] = []
    ring_world: list[list[np.ndarray]] = []
    for _, rings in parsed:
        loc, wor = [], []
        for r in rings:
            l0 = r - origin
            if max_edge is not None:
                l0, src = _densify(l0, max_edge)
                w0 = l0 + origin
                w0[src >= 0] = r
            else:
                w0 = r
            loc.append(l0)
            wor.append(w0)
        ring_local.append(loc)
        ring_world.append(wor)

    def polys(k: int) -> list[np.ndarray]:
        return [r @ uv for r in ring_local[k]]

    # Holes and overlapping rings on one section.
    for k in range(n_sec):
        pk = polys(k)
        for i in range(len(pk)):
            for j in range(len(pk)):
                if i == j:
                    continue
                inside = _points_in_polygon(pk[i], pk[j])
                if inside.all():
                    raise NotImplementedError(
                        f"section {label[k]}: ring {i} lies inside ring {j}. Holes in a "
                        "section outline are not supported; model the hole as a separate "
                        "solid and subtract it"
                    )
                if inside.any():
                    raise fail("section", k, k, f"rings {i} and {j} overlap each other")

    # Match rings between adjacent sections.
    has_up = [np.zeros(len(r), dtype=bool) for r in ring_local]
    has_down = [np.zeros(len(r), dtype=bool) for r in ring_local]
    bands: list[tuple[int, int, int]] = []  # (k, ring on k, ring on k + 1)
    splits: list[tuple[int, int, int, tuple[int, int]]] = []  # (k, parent side, parent, children)
    for k in range(n_sec - 1):
        links, message = _match(polys(k), polys(k + 1), match_distance)
        if message:
            raise fail("matching", k, k + 1, message)
        deg_a = np.bincount([i for i, _ in links], minlength=len(ring_local[k]))
        deg_b = np.bincount([j for _, j in links], minlength=len(ring_local[k + 1]))
        for i, j in links:
            has_up[k][i] = True
            has_down[k + 1][j] = True
            if deg_a[i] == 1 and deg_b[j] == 1:
                bands.append((k, i, j))
        for i in np.flatnonzero(deg_a == 2):
            kids = tuple(j for a, j in links if a == i)
            splits.append((k, 0, int(i), kids))
        for j in np.flatnonzero(deg_b == 2):
            kids = tuple(i for i, b in links if b == j)
            splits.append((k, 1, int(j), kids))

    for k in range(n_sec):
        for r in np.flatnonzero(~has_up[k] & ~has_down[k]):
            if end_modes == ("flat", "flat"):
                raise fail(
                    "matching",
                    k,
                    k,
                    f"ring {r} has no partner on a neighbouring section (no overlap seen along "
                    "the section normal), so it encloses no volume; check the string, or "
                    "raise match_distance if the body plunges",
                )

    # Bridges of the bifurcations: cut points on the parent rings.
    requests: dict[tuple[int, int], list[tuple[int, float]]] = {}
    split_slots = []
    for k, side, parent, kids in splits:
        kp, kc = (k, k + 1) if side == 0 else (k + 1, k)
        pc = polys(kc)
        try:
            cut_a, cut_b = _split_chord(ring_local[kp][parent] @ uv, pc[kids[0]], pc[kids[1]])
        except _StepError as exc:
            raise fail("bifurcation", k, k + 1, str(exc)) from None
        slot = requests.setdefault((kp, parent), [])
        split_slots.append(len(slot))
        slot.extend([cut_a, cut_b])

    build = _Builder(origin)
    ring_ids: list[list[np.ndarray]] = []
    cut_index: dict[tuple[int, int], list[int]] = {}
    for k in range(n_sec):
        ids_k = []
        for r in range(len(ring_local[k])):
            loc, wor = ring_local[k][r], ring_world[k][r]
            req = requests.get((k, r))
            if req:
                new, src, index = _refine(loc, req)
                world = new + origin
                world[src >= 0] = wor[src[src >= 0]]
                loc, wor = new, world
                ring_local[k][r] = loc
                cut_index[(k, r)] = index
            ids_k.append(build.add(loc, wor))
        ring_ids.append(ids_k)

    # Sub-rings of the bifurcations (need new bridge vertices).
    jobs: list[tuple[int, np.ndarray, np.ndarray]] = []  # (k, ids on k side, ids on k + 1 side)
    for k, i, j in bands:
        jobs.append((k, ring_ids[k][i], ring_ids[k + 1][j]))
    for (k, side, parent, kids), slot in zip(splits, split_slots, strict=True):
        kp, kc = (k, k + 1) if side == 0 else (k + 1, k)
        ids = ring_ids[kp][parent]
        loc = ring_local[kp][parent]
        ia, ib = cut_index[(kp, parent)][slot : slot + 2]
        if ia == ib:
            raise fail("bifurcation", k, k + 1, "the bridge across the parent ring has no length")
        pa, pb = loc[ia], loc[ib]
        chord = float(np.linalg.norm(pb - pa))
        edge = float(np.linalg.norm(np.roll(loc, -1, axis=0) - loc, axis=1).mean())
        n_inner = int(np.clip(round(chord / edge) - 1, 1, 500))
        t = (np.arange(1, n_inner + 1) / (n_inner + 1))[:, None]
        rise = bifurcation_lift * (position[kc] - position[kp])
        inner = pa + t * (pb - pa) + rise * np.sin(np.pi * t) * n
        bridge = build.add(inner)
        sub_1 = np.concatenate([_arc(ids, ia, ib), bridge[::-1]])
        sub_2 = np.concatenate([_arc(ids, ib, ia), bridge])
        c1 = (loc[_arc(np.arange(len(loc)), ia, ib)] @ uv).mean(axis=0)
        c2 = (loc[_arc(np.arange(len(loc)), ib, ia)] @ uv).mean(axis=0)
        k1, k2 = (_polygon_centroid(ring_local[kc][c] @ uv) for c in kids)
        straight = np.linalg.norm(c1 - k1) + np.linalg.norm(c2 - k2)
        swapped = np.linalg.norm(c1 - k2) + np.linalg.norm(c2 - k1)
        first, second = (kids[0], kids[1]) if straight <= swapped else (kids[1], kids[0])
        for sub, child in ((sub_1, first), (sub_2, second)):
            child_ids = ring_ids[kc][child]
            jobs.append((k, sub, child_ids) if side == 0 else (k, child_ids, sub))

    # Tie lines: snap to vertices.
    ties: list[tuple[int, int]] = []
    tie_used: list[bool] = []
    if tie_lines is not None:
        pts = np.asarray(tie_lines, dtype=np.float64)
        if pts.ndim != 3 or pts.shape[1:] != (2, 3):
            raise ValueError(f"tie_lines must have shape (t, 2, 3), got {pts.shape}")
        coords = build.coords()
        dist, idx = cKDTree(coords).query(pts.reshape(-1, 3) - origin)
        if (dist > tie_tolerance).any():
            bad = int(np.argmax(dist > tie_tolerance)) // 2
            raise SectionSolidError(
                f"tie line {bad}: an end point is {dist.reshape(-1, 2)[bad].max():.6g} m from the "
                f"nearest ring vertex (tolerance {tie_tolerance:g} m)"
            )
        ties = [(int(a), int(b)) for a, b in idx.reshape(-1, 2)]
        tie_used = [False] * len(ties)

    coords = build.coords()
    for k, ids_a, ids_b in jobs:
        local_ties = []
        if ties:
            where_a = {int(g): i for i, g in enumerate(ids_a)}
            where_b = {int(g): i for i, g in enumerate(ids_b)}
            for t_index, (ga, gb) in enumerate(ties):
                if ga in where_a and gb in where_b:
                    local_ties.append((where_a[ga], where_b[gb]))
                    tie_used[t_index] = True
                elif gb in where_a and ga in where_b:
                    local_ties.append((where_a[gb], where_b[ga]))
                    tie_used[t_index] = True
        try:
            f = _band(coords[ids_a], coords[ids_b], frame, method, local_ties, start_candidates)
        except _StepError as exc:
            raise fail("band", k, k + 1, str(exc)) from None
        build.faces.append(np.concatenate([ids_a, ids_b])[f])
    if ties and not all(tie_used):
        bad = tie_used.index(False)
        raise SectionSolidError(
            f"tie line {bad} does not join two rings that are triangulated together"
        )

    # Ends: rings without a partner below or above.
    for k in range(n_sec):
        for r, ids in enumerate(ring_ids[k]):
            for sign, linked in ((-1, has_down[k][r]), (1, has_up[k][r])):
                if linked:
                    continue
                mode = end_modes[0] if sign < 0 else end_modes[1]
                loc = build.coords()[ids]
                try:
                    if mode == "flat":
                        build.faces.append(ids[_cap(loc, frame)])
                        continue
                    if end_distance is not None:
                        dist_end = end_distance
                    else:
                        other = k + sign if 0 <= k + sign < n_sec else k - sign
                        dist_end = 0.5 * abs(position[other] - position[k])
                    centre2 = _polygon_centroid(loc @ uv)
                    centre = centre2[0] * u + centre2[1] * v + (loc @ n).mean() * n
                    shift = sign * dist_end * n
                    nxt = np.roll(ids, -1)
                    if mode == "point":
                        apex = build.add(centre + shift)
                        build.faces.append(np.column_stack([ids, nxt, np.repeat(apex, len(ids))]))
                    else:
                        small_local = centre + taper_scale * (loc - centre) + shift
                        small = build.add(small_local)
                        small_next = np.roll(small, -1)
                        build.faces.append(np.column_stack([ids, nxt, small]))
                        build.faces.append(np.column_stack([nxt, small_next, small]))
                        build.faces.append(small[_cap(small_local, frame)])
                except _StepError as exc:
                    raise fail("cap", k, k, str(exc)) from None

    vertices = np.vstack(build.world)
    mesh = Mesh(vertices, np.vstack(build.faces))
    mesh = mesh.weld(weld_tolerance).fix_orientation()
    if not validate:
        return mesh
    report = mesh.validate(check_self_intersection=check_self_intersection)
    ok = (
        report.is_closed
        and report.is_manifold
        and report.is_winding_consistent
        and not report.has_degenerate_faces
        and not report.has_duplicate_faces
        and not report.has_self_intersections  # None when the check was skipped
    )
    if ok:
        return mesh
    problems = _locate(mesh, report, origin, n, position, label)
    issues = [t for t in report.issues() if t != "self-intersection not checked"]
    lines = [f"the sections do not make a valid solid ({'; '.join(issues)})"]
    lines += [f"  {p.message}" for p in problems]
    raise SectionSolidError("\n".join(lines), problems, mesh, report)


_KIND_TEXT = {
    "self_intersection": "pair(s) of triangles cross each other",
    "open_edge": "open edge(s)",
    "nonmanifold_edge": "edge(s) shared by more than two triangles",
    "nonmanifold_vertex": "vertex(es) where the surface touches itself",
    "inconsistent_winding": "edge(s) with inconsistent winding",
    "degenerate_face": "zero-area triangle(s)",
    "duplicate_face": "duplicate triangle(s)",
}


def _locate(mesh: Mesh, report: ValidationReport, origin, n, position, label):
    """Turn a validation report into problems located between sections."""
    w = (mesh.vertices - origin) @ n
    last = len(position) - 1
    tol = 1e-6 * max(float(position[-1] - position[0]), 1.0)

    def interval(vertex_sets: np.ndarray) -> np.ndarray:
        """(k, 2) section indices spanned by each row of vertex indices."""
        ws = w[vertex_sets]
        lo, hi = ws.min(axis=1), ws.max(axis=1)
        mid = 0.5 * (lo + hi)
        a = np.clip(np.searchsorted(position, mid + tol, side="right") - 1, 0, last)
        b = np.clip(np.searchsorted(position, mid - tol, side="left"), 0, last)
        return np.column_stack([np.minimum(a, b), np.maximum(a, b)])

    faces = mesh.faces
    groups: list[tuple[str, np.ndarray]] = []
    if report.self_intersections is not None and len(report.self_intersections):
        pairs = report.self_intersections
        groups.append(("self_intersection", np.hstack([faces[pairs[:, 0]], faces[pairs[:, 1]]])))
    for kind, edges in (
        ("open_edge", report.boundary_edges),
        ("nonmanifold_edge", report.nonmanifold_edges),
        ("inconsistent_winding", report.inconsistent_edges),
    ):
        if len(edges):
            groups.append((kind, np.asarray(edges)))
    if len(report.nonmanifold_vertices):
        groups.append(("nonmanifold_vertex", np.asarray(report.nonmanifold_vertices)[:, None]))
    for kind, idx in (
        ("degenerate_face", report.degenerate_faces),
        ("duplicate_face", report.duplicate_faces),
    ):
        if len(idx):
            groups.append((kind, faces[np.asarray(idx)]))

    problems = []
    for kind, sets in groups:
        spans, counts = np.unique(interval(sets), axis=0, return_counts=True)
        for (ka, kb), count in zip(spans.tolist(), counts.tolist(), strict=True):
            if ka == kb:
                where = f"at section {label[ka]} (position {position[ka]:.3f} m)"
            else:
                where = (
                    f"between sections {label[ka]} and {label[kb]} "
                    f"(positions {position[ka]:.3f} and {position[kb]:.3f} m)"
                )
            problems.append(
                SectionProblem(
                    kind,
                    (label[ka], label[kb]),
                    (float(position[ka]), float(position[kb])),
                    int(count),
                    f"{count} {_KIND_TEXT[kind]} {where}",
                )
            )
    problems.sort(key=lambda p: (p.positions, p.kind))
    return problems


# --------------------------------------------------------------------------
# Convenience
# --------------------------------------------------------------------------


def solid_from_strings(
    strings: StringFile,
    *,
    normal=None,
    tolerance: float | None = None,
    numbers: Iterable[int] | None = None,
    closed_tolerance: float = 1e-6,
    **kwargs,
) -> Mesh:
    """Build a solid from the closed segments of a string file.

    Segments are grouped into sections by their mean position along the
    common normal: after sorting, a new section starts wherever the position
    jumps by more than ``tolerance``.

    Parameters
    ----------
    strings : StringFile
    normal : (3,) array_like, optional
        Section direction. Default: the common normal of all segments.
    tolerance : float, optional
        Largest difference in position (metres) between rings of the same
        section. Default: 1 % of the mean spacing the segments would have if
        each were on its own section.
    numbers : iterable of int, optional
        Use only segments with these string numbers.
    closed_tolerance : float
        A segment is closed when its last point repeats its first within
        this distance (metres).
    **kwargs
        Passed to :func:`solid_from_sections`. Section indices in error
        diagnostics count the sections in order of position.

    Raises
    ------
    SectionSolidError
        If a segment is open (naming it), or as :func:`solid_from_sections`.
    """
    wanted = None if numbers is None else {int(k) for k in numbers}
    rings = []
    for index, seg in enumerate(strings.segments):
        if seg.n_points == 0 or (wanted is not None and seg.number not in wanted):
            continue
        if not seg.closed(closed_tolerance):
            gap = float(np.linalg.norm(seg.points[-1] - seg.points[0])) if seg.n_points > 1 else 0.0
            raise SectionSolidError(
                f"segment {index} (string {seg.number}, {seg.n_points} points) is open: its "
                f"first and last points are {gap:.6g} m apart. Close the string to use it "
                "in a solid"
            )
        try:
            rings.append(_clean_ring(seg.points, min(1e-9, closed_tolerance)))
        except ValueError as exc:
            raise SectionSolidError(f"segment {index} (string {seg.number}): {exc}") from None
    if len(rings) < 2:
        raise SectionSolidError("at least two closed segments are needed to make a solid")
    origin = np.mean([r.mean(axis=0) for r in rings], axis=0)
    n = _estimate_normal([r - origin for r in rings]) if normal is None else _unit(normal)
    position = np.array([((r - origin) @ n).mean() for r in rings])
    order = np.argsort(position, kind="stable")
    if tolerance is None:
        tolerance = max(0.01 * float(np.ptp(position)) / (len(rings) - 1), 1e-9)
    sections: list[list[np.ndarray]] = []
    previous = None
    for k in order:
        if previous is None or position[k] - previous > tolerance:
            sections.append([])
        sections[-1].append(rings[k])
        previous = position[k]
    return solid_from_sections(sections, normal=n, **kwargs)


def sections_from_mesh(mesh: Mesh, origin, normal, spacing: float) -> list[Section]:
    """Cut a mesh with parallel planes ``origin + k * spacing * normal``.

    Returns the non-empty sections in order of increasing ``k`` (all integer
    ``k`` for which the plane meets the mesh). The result can be passed
    directly to :func:`solid_from_sections`.
    """
    if not spacing > 0:
        raise ValueError("spacing must be > 0")
    origin = np.asarray(origin, dtype=np.float64).reshape(3)
    n = _unit(normal)
    if mesh.n_faces == 0:
        return []
    d = (mesh.vertices - origin) @ n
    k0 = int(np.ceil(d.min() / spacing))
    k1 = int(np.floor(d.max() / spacing))
    out = []
    for k in range(k0, k1 + 1):
        s = section(mesh, origin + (k * spacing) * n, n)
        if len(s):
            out.append(s)
    return out
