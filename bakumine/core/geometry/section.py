"""Mesh-plane section: cut a mesh with an arbitrary plane and return polylines.

The plane is given by a point and a normal. The result is a set of
polylines: closed loops for a closed mesh, open polylines where the plane
runs off the border of an open mesh (a DTM profile, for example).

How vertices on the plane are handled
-------------------------------------
Every vertex gets a signed distance ``d`` to the plane and is classified as
*positive* (``d > 0``) or *not positive* (``d <= 0``); vertices within
``tol`` of the plane have ``d`` set to exactly 0. Everything else follows
from these per-vertex classes, so the topology of the result is always
consistent:

* an edge crosses the plane when its ends are in different classes, and it
  has one crossing point, computed once per edge (never once per face). If
  the non-positive end is on the plane, the crossing point is exactly that
  vertex;
* a face contributes one segment when it has two crossing edges, else none.

This is the section by the plane shifted infinitesimally towards its
normal. Consequences: a vertex or an edge lying on the plane never produces
a duplicate point or a gap; a face lying in the plane shows up as its
outline when the solid continues on the positive side (the cut at the
bottom face of a box is the full rectangle) and not at all when the solid
ends there (the cut at the top face is empty) -- the same half-open
convention as point-in-solid. Zero-length segments, repeated points,
zero-area spikes and pairs of coincident opposite loops created at on-plane
vertices are removed.

Orientation
-----------
In the plane frame ``(u, v)`` (see :class:`Section`), loops run with the
solid on their left for an outward-wound mesh: outer contours are
counter-clockwise and contours of holes clockwise, so the sum of the signed
loop areas is the area of the cut.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bakumine.core.geometry.mesh import Mesh

__all__ = ["Section", "plane_frame", "section"]


def plane_frame(normal) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Right-handed orthonormal frame ``(u, v, n)`` of a plane, ``u x v = n``.

    ``u`` is horizontal: ``u = z x n`` normalised, so for a vertical section
    plane ``v`` is straight up and ``u`` points to the right for a viewer who
    looks against the normal. For a horizontal plane ``u`` is east (+x) and
    ``v = n x u`` (north for ``n = +z``).
    """
    n = np.asarray(normal, dtype=np.float64).reshape(-1)
    if n.shape != (3,) or not np.isfinite(n).all() or not np.linalg.norm(n) > 0:
        raise ValueError("normal must be a finite, non-zero 3-vector")
    n = n / np.linalg.norm(n)
    u = np.array([-n[1], n[0], 0.0])  # z x n
    length = np.linalg.norm(u)
    u = np.array([1.0, 0.0, 0.0]) if length < 1e-12 else u / length
    v = np.cross(n, u)
    return u, v / np.linalg.norm(v), n


def _signed_area(p: np.ndarray) -> float:
    q = p - p[0]
    return 0.5 * float(np.sum(q[:, 0] * np.roll(q[:, 1], -1) - np.roll(q[:, 0], -1) * q[:, 1]))


@dataclass(frozen=True)
class Section:
    """Result of :func:`section`.

    Attributes
    ----------
    origin : (3,) float64
        The point that defines the plane; origin of the plane frame.
    u, v, normal : (3,) float64
        Orthonormal plane frame in world coordinates, ``u x v = normal``
        (see :func:`plane_frame`).
    polylines : tuple of (k, 3) float64
        World coordinates. A closed loop does *not* repeat its first point.
    closed : tuple of bool
        Whether each polyline is a closed loop.
    """

    origin: np.ndarray
    u: np.ndarray
    v: np.ndarray
    normal: np.ndarray
    polylines: tuple[np.ndarray, ...]
    closed: tuple[bool, ...]

    def __len__(self) -> int:
        return len(self.polylines)

    @property
    def to_plane_matrix(self) -> np.ndarray:
        """4x4 transform from world to plane coordinates ``(u, v, w)``; ``w`` is the offset."""
        m = np.eye(4)
        m[:3, :3] = np.array([self.u, self.v, self.normal])
        m[:3, 3] = -m[:3, :3] @ self.origin
        return m

    @property
    def to_world_matrix(self) -> np.ndarray:
        """4x4 transform from plane coordinates ``(u, v, w)`` to world."""
        m = np.eye(4)
        m[:3, :3] = np.column_stack([self.u, self.v, self.normal])
        m[:3, 3] = self.origin
        return m

    def to_plane(self, points) -> np.ndarray:
        """(..., 2) plane coordinates ``(u, v)`` of world points (..., 3)."""
        d = np.asarray(points, dtype=np.float64) - self.origin
        return np.stack([d @ self.u, d @ self.v], axis=-1)

    def to_world(self, uv) -> np.ndarray:
        """(..., 3) world coordinates of plane points (..., 2)."""
        uv = np.asarray(uv, dtype=np.float64)
        return self.origin + uv[..., :1] * self.u + uv[..., 1:2] * self.v

    @property
    def polylines_2d(self) -> tuple[np.ndarray, ...]:
        """The polylines in plane coordinates, each (k, 2)."""
        return tuple(self.to_plane(p) for p in self.polylines)

    @property
    def signed_areas(self) -> np.ndarray:
        """Signed area of each polyline in the plane frame (0 for open polylines)."""
        return np.array(
            [
                _signed_area(p) if c else 0.0
                for p, c in zip(self.polylines_2d, self.closed, strict=True)
            ]
        )

    @property
    def area(self) -> float:
        """Area of the cut: sum of the signed loop areas (holes count negative).

        Meaningful for a closed, outward-wound mesh.
        """
        return float(self.signed_areas.sum())

    @property
    def lengths(self) -> np.ndarray:
        """Length of each polyline (perimeter for closed loops)."""
        out = []
        for p, c in zip(self.polylines, self.closed, strict=True):
            q = np.vstack([p, p[:1]]) if c else p
            out.append(float(np.linalg.norm(np.diff(q, axis=0), axis=1).sum()))
        return np.array(out)

    @property
    def length(self) -> float:
        """Total length of all polylines."""
        return float(self.lengths.sum())


def _chain(start: np.ndarray, end: np.ndarray, n_nodes: int) -> list[list[int]]:
    """Join directed segments (node pairs) into node paths.

    Paths follow the segment direction where possible. A closed path repeats
    its first node at the end.
    """
    start = start.tolist()
    end = end.tolist()
    incident: list[list[int]] = [[] for _ in range(n_nodes)]
    for s, (a, b) in enumerate(zip(start, end, strict=True)):
        incident[a].append(s)
        incident[b].append(s)
    used = [False] * len(start)

    def walk(node: int, s: int) -> list[int]:
        path = [node]
        forward = 0
        steps = 0
        while True:
            used[s] = True
            steps += 1
            if start[s] == node:
                forward += 1
                node = end[s]
            else:
                node = start[s]
            path.append(node)
            nxt = -1
            for c in incident[node]:
                if not used[c]:
                    if start[c] == node:
                        nxt = c
                        break
                    if nxt < 0:
                        nxt = c
            if nxt < 0:
                break
            s = nxt
        if 2 * forward < steps:
            path.reverse()
        return path

    paths = []
    # Open polylines first: they start at nodes with an odd number of segments.
    for node in range(n_nodes):
        if len(incident[node]) & 1:
            for s in incident[node]:
                if not used[s]:
                    paths.append(walk(node, s))
                    break
    for s in range(len(start)):
        if not used[s]:
            paths.append(walk(start[s], s))
    return paths


def _clean(points: np.ndarray, closed: bool) -> np.ndarray:
    """Drop repeated points and zero-area spikes (``a, b, a``)."""
    pts = [tuple(p) for p in points.tolist()]
    while True:
        out: list[tuple] = []
        for p in pts:
            if out and out[-1] == p:
                continue
            if len(out) >= 2 and out[-2] == p:
                out.pop()
                continue
            out.append(p)
        if closed:
            changed = True
            while changed and len(out) >= 2:
                changed = False
                if out[0] == out[-1]:
                    out.pop()
                    changed = True
                elif len(out) >= 3 and out[1] == out[-1]:
                    out = out[1:-1]
                    changed = True
                elif len(out) >= 3 and out[0] == out[-2]:
                    out = out[:-2]
                    changed = True
        if len(out) == len(pts):
            break
        pts = out
    return np.array(out, dtype=np.float64).reshape(-1, 3)


def _drop_reversed_pairs(polylines: list, closed: list) -> tuple[list, list]:
    """Remove pairs of closed loops through the same points in opposite directions.

    They enclose nothing: the mesh touches the plane from the positive side
    along a closed curve of on-plane edges (a torus resting on the plane).
    """
    unmatched: dict[tuple, int] = {}
    drop: set[int] = set()
    for i, (pts, is_closed) in enumerate(zip(polylines, closed, strict=True)):
        if not is_closed:
            continue
        rows = [tuple(p) for p in pts.tolist()]
        j = rows.index(min(rows))
        fwd = tuple(rows[j:] + rows[:j])
        rev = (fwd[0], *fwd[:0:-1])
        partner = unmatched.pop(rev, None)
        if partner is None:
            unmatched[fwd] = i
        else:
            drop.update((i, partner))
    keep = [i for i in range(len(polylines)) if i not in drop]
    return [polylines[i] for i in keep], [closed[i] for i in keep]


def section(mesh: Mesh, origin, normal, *, tol: float | None = None) -> Section:
    """Cut a mesh with the plane through ``origin`` with normal ``normal``.

    Parameters
    ----------
    mesh : Mesh
        Closed or open. Shared vertices must be shared by index (weld first).
    origin : (3,) float
        A point on the plane; also the origin of the plane frame.
    normal : (3,) float
        Plane normal (any length).
    tol : float, optional
        Vertices closer than this to the plane are treated as lying on it.
        Default: 1e-12 times the diagonal of the mesh bounding box. Use 0 to
        snap only exact zeros.

    Returns
    -------
    Section
        Polylines in world coordinates with their closed flags, the plane
        frame, and helpers for plane coordinates (``polylines_2d``,
        ``to_plane``, ``to_plane_matrix``), ``area`` and ``length``. See the
        module docstring for the treatment of vertices, edges and faces lying
        on the plane. For a closed manifold mesh every polyline is closed.
    """
    origin = np.asarray(origin, dtype=np.float64).reshape(-1)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError("origin must be a finite 3-vector")
    u, v, n = plane_frame(normal)
    empty = Section(origin, u, v, n, (), ())
    if mesh.n_faces == 0:
        return empty
    if tol is None:
        tol = 1e-12 * float(np.linalg.norm(mesh.bounds[1] - mesh.bounds[0]))
    if not tol >= 0:
        raise ValueError("tol must be >= 0")

    vertices = mesh.vertices
    d = (vertices - origin) @ n
    d[np.abs(d) <= tol] = 0.0
    positive = d > 0.0

    faces = mesh.faces
    pf = positive[faces]
    count = pf.sum(axis=1)
    crossing = np.flatnonzero((count == 1) | (count == 2))
    if len(crossing) == 0:
        return empty
    pf = pf[crossing]
    lone_positive = count[crossing] == 1
    # slot of the corner that is alone on its side
    lone = np.where(lone_positive, pf.argmax(axis=1), pf.argmin(axis=1))
    face_edges = mesh.edge_map.face_edges[crossing]
    rows = np.arange(len(crossing))
    e_ab = face_edges[rows, lone]  # edge lone -> next corner
    e_ca = face_edges[rows, (lone + 2) % 3]  # edge previous corner -> lone
    # Solid on the left seen against the normal (see module docstring).
    seg_start = np.where(lone_positive, e_ab, e_ca)
    seg_end = np.where(lone_positive, e_ca, e_ab)

    edge_ids, inverse = np.unique(np.concatenate([seg_start, seg_end]), return_inverse=True)
    inverse = inverse.reshape(-1)
    ends = mesh.edge_map.edges[edge_ids]
    first_positive = positive[ends[:, 0]]
    neg = np.where(first_positive, ends[:, 1], ends[:, 0])
    pos = np.where(first_positive, ends[:, 0], ends[:, 1])
    t = d[neg] / (d[neg] - d[pos])  # 0 <= t < 1; exactly 0 when the vertex is on the plane
    node_points = vertices[neg] + t[:, None] * (vertices[pos] - vertices[neg])

    k = len(seg_start)
    paths = _chain(inverse[:k], inverse[k:], len(edge_ids))
    polylines = []
    closed = []
    for path in paths:
        is_closed = len(path) > 2 and path[0] == path[-1]
        pts = _clean(node_points[path], is_closed)
        if len(pts) < (3 if is_closed else 2):
            continue
        polylines.append(pts)
        closed.append(is_closed)
    polylines, closed = _drop_reversed_pairs(polylines, closed)
    for pts in polylines:
        pts.flags.writeable = False
    return Section(origin, u, v, n, tuple(polylines), tuple(closed))
