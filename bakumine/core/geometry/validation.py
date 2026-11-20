"""Mesh validation and orientation repair (research.md section 5.4).

A solid is usable for volumes and constraints only if it is closed, manifold,
consistently oriented and free of degenerate, duplicate and self-intersecting
triangles. :func:`validate` checks all of these and returns a structured
:class:`ValidationReport` that says *where* each defect is.
:func:`fix_orientation` makes the winding consistent and outward.

Validation works on vertex indices: two faces are neighbours only if they
share vertex *indices*. Weld the mesh first (``Mesh.weld``) if it comes from a
format that repeats coordinates per face.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numba import njit, prange
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.parallel import parallel_njit

__all__ = [
    "ValidationReport",
    "fix_orientation",
    "misoriented_faces",
    "triangle_pairs_intersect",
    "validate",
]


@dataclass(frozen=True)
class ValidationReport:
    """Result of :func:`validate`. Every defect list is empty for a valid solid.

    Attributes
    ----------
    n_vertices, n_faces : int
    n_components : int
        Number of face groups connected through two-face edges.
    boundary_edges : (k, 2) int64
        Vertex pairs of edges used by only one face (holes, open borders).
    nonmanifold_edges : (k, 2) int64
        Vertex pairs of edges used by more than two faces.
    nonmanifold_vertices : (k,) int64
        Vertices whose faces form more than one fan (shapes touching at a
        point only).
    inconsistent_edges : (k, 2) int64
        Vertex pairs of two-face edges whose faces wind in opposite senses
        (one of the two faces is flipped).
    misoriented_faces : (k,) int64
        Faces that :func:`fix_orientation` would flip to make the winding
        consistent and outward.
    orientable : bool
        False if no consistent winding exists (e.g. a Moebius band).
    degenerate_faces : (k,) int64
        Faces with zero area (repeated vertex or collinear corners).
    duplicate_faces : (k,) int64
        Faces that use the same three vertices as an earlier face.
    self_intersections : (k, 2) int64 or None
        Pairs of faces ``(i, j)``, ``i < j``, that cross or overlap other
        than along a shared vertex or edge. None if the check was skipped.
    cavity_faces : (k,) int64
        The faces of ``misoriented_faces`` that belong to a **cavity**: a closed
        shell with negative volume that lies inside an odd number of the other
        shells (a hole in the solid). Its inward winding is the correct one: the
        normals point out of the material and ``Mesh.volume`` is the volume of
        the outer shell less that of the hole. Found only for a closed mesh with
        consistent winding.
    """

    n_vertices: int
    n_faces: int
    n_components: int
    boundary_edges: np.ndarray
    nonmanifold_edges: np.ndarray
    nonmanifold_vertices: np.ndarray
    inconsistent_edges: np.ndarray
    misoriented_faces: np.ndarray
    orientable: bool
    degenerate_faces: np.ndarray
    duplicate_faces: np.ndarray
    self_intersections: np.ndarray | None
    cavity_faces: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))

    @property
    def is_closed(self) -> bool:
        """Every edge is shared by exactly two faces (and there is at least one face)."""
        return self.n_faces > 0 and not len(self.boundary_edges) and not len(self.nonmanifold_edges)

    @property
    def is_manifold(self) -> bool:
        """No edge with more than two faces and no vertex with more than one fan."""
        return not len(self.nonmanifold_edges) and not len(self.nonmanifold_vertices)

    @property
    def is_winding_consistent(self) -> bool:
        """Every two-face edge is traversed once in each direction."""
        return not len(self.inconsistent_edges)

    @property
    def is_outward(self) -> bool:
        """Closed, consistently wound and every component has positive volume."""
        return self.is_closed and self.is_winding_consistent and not len(self.misoriented_faces)

    @property
    def wrongly_wound_faces(self) -> np.ndarray:
        """Faces of inward-wound shells that are not cavities (``misoriented`` less ``cavity``)."""
        return np.setdiff1d(self.misoriented_faces, self.cavity_faces)

    @property
    def is_oriented(self) -> bool:
        """Closed, consistently wound, every shell outward except the cavities (inward)."""
        return (
            self.is_closed and self.is_winding_consistent and not len(self.wrongly_wound_faces)
        )

    @property
    def has_degenerate_faces(self) -> bool:
        return bool(len(self.degenerate_faces))

    @property
    def has_duplicate_faces(self) -> bool:
        return bool(len(self.duplicate_faces))

    @property
    def has_self_intersections(self) -> bool | None:
        """None if the self-intersection check was skipped."""
        return None if self.self_intersections is None else bool(len(self.self_intersections))

    @property
    def is_valid(self) -> bool:
        """Usable as a solid: closed, manifold, consistently wound, no bad faces.

        A skipped self-intersection check counts as not valid. Inward winding
        of a consistent solid is not a defect here; see :attr:`is_outward`.
        """
        return (
            self.is_closed
            and self.is_manifold
            and self.is_winding_consistent
            and not self.has_degenerate_faces
            and not self.has_duplicate_faces
            and self.has_self_intersections is False
        )

    def issues(self) -> list[str]:
        """Human-readable list of the defects found (empty if none)."""
        out = []
        if self.n_faces == 0:
            out.append("mesh has no faces")
        for items, text in (
            (self.boundary_edges, "boundary edges (mesh is open)"),
            (self.nonmanifold_edges, "edges shared by more than two faces"),
            (self.nonmanifold_vertices, "non-manifold vertices"),
            (self.inconsistent_edges, "edges with inconsistent winding"),
            (self.degenerate_faces, "degenerate (zero-area) faces"),
            (self.duplicate_faces, "duplicate faces"),
        ):
            if len(items):
                out.append(f"{len(items)} {text}")
        if not self.orientable:
            out.append("surface is not orientable")
        if self.self_intersections is None:
            out.append("self-intersection not checked")
        elif len(self.self_intersections):
            out.append(f"{len(self.self_intersections)} self-intersecting face pairs")
        if self.is_closed and self.is_winding_consistent and len(self.wrongly_wound_faces):
            out.append(f"{len(self.wrongly_wound_faces)} faces wound inward (negative volume)")
        return out

    def summary(self) -> str:
        head = f"{self.n_faces} faces, {self.n_vertices} vertices, {self.n_components} component(s)"
        issues = self.issues()
        if not issues:
            return f"valid solid: {head}"
        return f"{'valid' if self.is_valid else 'INVALID'}: {head}; " + "; ".join(issues)

    def to_dict(self) -> dict:
        """JSON-friendly flags and defect counts (no index arrays)."""
        n_self = None if self.self_intersections is None else len(self.self_intersections)
        return {
            "n_vertices": self.n_vertices,
            "n_faces": self.n_faces,
            "n_components": self.n_components,
            "is_valid": self.is_valid,
            "is_closed": self.is_closed,
            "is_manifold": self.is_manifold,
            "is_winding_consistent": self.is_winding_consistent,
            "is_outward": self.is_outward,
            "is_oriented": self.is_oriented,
            "n_cavity_faces": len(self.cavity_faces),
            "orientable": self.orientable,
            "n_boundary_edges": len(self.boundary_edges),
            "n_nonmanifold_edges": len(self.nonmanifold_edges),
            "n_nonmanifold_vertices": len(self.nonmanifold_vertices),
            "n_inconsistent_edges": len(self.inconsistent_edges),
            "n_misoriented_faces": len(self.misoriented_faces),
            "n_degenerate_faces": len(self.degenerate_faces),
            "n_duplicate_faces": len(self.duplicate_faces),
            "n_self_intersections": n_self,
            "issues": self.issues(),
        }


# --------------------------------------------------------------------------
# Topology checks
# --------------------------------------------------------------------------


def _inconsistent_edge_ids(mesh: Mesh) -> np.ndarray:
    em = mesh.edge_map
    pairs = em.manifold_halfedge_pairs
    forward = em.forward.reshape(-1)
    same = forward[pairs[:, 0]] == forward[pairs[:, 1]]
    return em.face_edges.reshape(-1)[pairs[same, 0]]


def _nonmanifold_vertices(mesh: Mesh) -> np.ndarray:
    """Vertices whose incident faces are not one edge-connected fan."""
    em = mesh.edge_map
    faces = mesh.faces
    m = mesh.n_faces
    if m == 0:
        return np.zeros(0, dtype=np.int64)
    flat = faces.reshape(-1)  # corner 3f+j is vertex faces[f, j]
    # Link the corners of consecutive faces around every edge, at both ends.
    edge_of_slot = np.repeat(np.arange(em.n_edges), em.counts)
    link = edge_of_slot[:-1] == edge_of_slot[1:]
    h1, h2, e = em.halfedges[:-1][link], em.halfedges[1:][link], edge_of_slot[:-1][link]
    rows, cols = [], []
    for end in (0, 1):
        v = em.edges[e, end]
        c1 = np.where(flat[h1] == v, h1, 3 * (h1 // 3) + (h1 % 3 + 1) % 3)
        c2 = np.where(flat[h2] == v, h2, 3 * (h2 // 3) + (h2 % 3 + 1) % 3)
        rows.append(c1)
        cols.append(c2)
    # Corners of one face that are the same vertex (degenerate face) are one corner.
    base = 3 * np.arange(m)
    for j, k in ((0, 1), (1, 2), (2, 0)):
        rep = faces[:, j] == faces[:, k]
        rows.append(base[rep] + j)
        cols.append(base[rep] + k)
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(3 * m, 3 * m))
    _, fan = connected_components(graph, directed=False)
    fans = np.unique(np.column_stack([flat, fan]), axis=0)
    return np.flatnonzero(np.bincount(fans[:, 0], minlength=mesh.n_vertices) > 1)


def _degenerate_faces(mesh: Mesh, rel_tol: float) -> np.ndarray:
    f = mesh.faces
    if mesh.n_faces == 0:
        return np.zeros(0, dtype=bool)
    tri = mesh.triangles
    e = np.stack([tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 1], tri[:, 0] - tri[:, 2]], axis=1)
    longest2 = np.einsum("ijk,ijk->ij", e, e).max(axis=1)
    two_area = 2.0 * mesh.face_areas
    # Height below the rounding of the stored coordinates is zero for all purposes.
    noise = 16.0 * np.finfo(np.float64).eps * np.abs(mesh.vertices).max()
    threshold = np.maximum(rel_tol * longest2, noise * np.sqrt(longest2))
    repeated = (f[:, 0] == f[:, 1]) | (f[:, 1] == f[:, 2]) | (f[:, 2] == f[:, 0])
    return repeated | (two_area <= threshold)


def _duplicate_faces(mesh: Mesh) -> np.ndarray:
    if mesh.n_faces < 2:
        return np.zeros(0, dtype=np.int64)
    s = np.sort(mesh.faces, axis=1)
    order = np.lexsort((np.arange(len(s)), s[:, 2], s[:, 1], s[:, 0]))
    ss = s[order]
    same = (ss[1:] == ss[:-1]).all(axis=1)
    return np.sort(order[1:][same])


# --------------------------------------------------------------------------
# Orientation
# --------------------------------------------------------------------------


@njit(cache=True)
def _propagate_orientation(adjacent, forward, m):
    """Flood fill across two-face edges choosing flips that make winding agree.

    Returns ``(flip, component, n_components, conflict)``.
    """
    flip = np.zeros(m, dtype=np.bool_)
    component = np.full(m, -1, dtype=np.int64)
    stack = np.empty(max(m, 1), dtype=np.int64)
    n_components = 0
    conflict = False
    for seed in range(m):
        if component[seed] >= 0:
            continue
        component[seed] = n_components
        stack[0] = seed
        sp = 1
        while sp > 0:
            sp -= 1
            f = stack[sp]
            for j in range(3):
                h = 3 * f + j
                g = adjacent[h]
                if g < 0:
                    continue
                nf = g // 3
                want = flip[f] != (forward[h] == forward[g])
                if component[nf] < 0:
                    component[nf] = n_components
                    flip[nf] = want
                    stack[sp] = nf
                    sp += 1
                elif flip[nf] != want and nf != f:
                    conflict = True
        n_components += 1
    return flip, component, n_components, conflict


def _orientation(mesh: Mesh) -> tuple[np.ndarray, int, bool]:
    """Faces to flip (bool mask), component count, orientable flag."""
    m = mesh.n_faces
    if m == 0:
        return np.zeros(0, dtype=bool), 0, True
    em = mesh.edge_map
    pairs = em.manifold_halfedge_pairs
    adjacent = np.full(3 * m, -1, dtype=np.int64)
    adjacent[pairs[:, 0]] = pairs[:, 1]
    adjacent[pairs[:, 1]] = pairs[:, 0]
    flip, component, n_components, conflict = _propagate_orientation(
        adjacent, np.ascontiguousarray(em.forward.reshape(-1)), m
    )
    # Choose the sign of each component: closed -> positive volume,
    # open -> normals up on balance (the DTM convention, z = up).
    tri = mesh.triangles - mesh.local_origin
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    sign = np.where(flip, -1.0, 1.0)
    six_volume = sign * np.einsum("ij,ij->i", tri[:, 0], cross)
    volume = np.bincount(component, weights=six_volume, minlength=n_components)
    up = np.bincount(component, weights=sign * cross[:, 2], minlength=n_components)
    open_faces = (adjacent.reshape(m, 3) < 0).any(axis=1)
    is_open = np.bincount(component, weights=open_faces, minlength=n_components) > 0
    reverse = np.where(is_open, up < 0, volume < 0)
    return flip ^ reverse[component], n_components, not conflict


def _cavity_faces(mesh: Mesh, inward: np.ndarray) -> np.ndarray:
    """Faces of the inward-wound shells of a closed, consistent mesh that are cavities.

    A shell is enclosed when a vertical ray going up from its highest vertex crosses the
    mesh an odd number of times: the ray cannot cross the shell itself again, and shells
    do not cross each other, so the count is the number of shells around it.
    """
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    from bakumine.core.geometry.inside import column_crossings

    f, n = mesh.faces, mesh.n_vertices
    rows = np.concatenate([f[:, 0], f[:, 1]])
    cols = np.concatenate([f[:, 1], f[:, 2]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    vertex_label = connected_components(graph, directed=False)[1]
    face_label = vertex_label[f[:, 0]]
    shells = np.unique(face_label[inward])
    # a shell is inward as a whole when the winding is consistent
    shells = shells[[bool(inward[face_label == c].all()) for c in shells]]
    if not len(shells):
        return np.zeros(0, dtype=np.int64)
    z = mesh.vertices[:, 2]
    order = np.lexsort((z, vertex_label))  # by shell, then height: the last of a shell is its top
    last = np.flatnonzero(np.r_[vertex_label[order][1:] != vertex_label[order][:-1], True])
    top = dict(zip(vertex_label[order][last].tolist(), order[last].tolist(), strict=True))
    tops = mesh.vertices[[top[int(c)] for c in shells]]
    crossings = column_crossings(mesh, tops[:, :2])
    enclosed = np.zeros(len(shells), dtype=bool)
    for i in range(len(shells)):
        zs = crossings.z[crossings.offsets[i] : crossings.offsets[i + 1]]
        above = zs > tops[i, 2] + 1e-9 * (1.0 + abs(tops[i, 2]))
        enclosed[i] = bool(int(above.sum()) & 1)
    return np.flatnonzero(inward & np.isin(face_label, shells[enclosed]))


def misoriented_faces(mesh: Mesh) -> np.ndarray:
    """Indices of the faces :func:`fix_orientation` would flip."""
    return np.flatnonzero(_orientation(mesh)[0])


def fix_orientation(mesh: Mesh) -> Mesh:
    """Return a mesh with consistent, outward winding.

    Faces are grouped into components connected through two-face edges. In
    each component the winding is made consistent, then the whole component
    is flipped if needed so that

    * a closed component has positive signed volume (normals outward);
    * an open component (a surface) has normals pointing up on balance.

    Every shell is made outward on its own, including a shell nested inside
    another. Components joined only through non-manifold edges are treated
    separately. A non-orientable surface cannot be fixed; the result then
    still has inconsistent edges, which :func:`validate` reports. Vertices
    and face order are unchanged.
    """
    flip = _orientation(mesh)[0]
    if not flip.any():
        return mesh
    return mesh.flipped(np.flatnonzero(flip))


# --------------------------------------------------------------------------
# Self-intersection (narrow phase)
# --------------------------------------------------------------------------


@njit(cache=True, inline="always")
def _orient2d(ax, ay, bx, by, cx, cy):
    return (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)


@njit(cache=True)
def _point_in_tri_2d(px, py, ax, ay, bx, by, cx, cy):
    o1 = _orient2d(ax, ay, bx, by, px, py)
    o2 = _orient2d(bx, by, cx, cy, px, py)
    o3 = _orient2d(cx, cy, ax, ay, px, py)
    return (o1 >= 0 and o2 >= 0 and o3 >= 0) or (o1 <= 0 and o2 <= 0 and o3 <= 0)


@njit(cache=True)
def _seg_seg_2d(px, py, qx, qy, rx, ry, sx, sy):
    """Closed 2D segments pq and rs share a point."""
    d1 = _orient2d(rx, ry, sx, sy, px, py)
    d2 = _orient2d(rx, ry, sx, sy, qx, qy)
    d3 = _orient2d(px, py, qx, qy, rx, ry)
    d4 = _orient2d(px, py, qx, qy, sx, sy)
    if ((d1 > 0 and d2 < 0) or (d1 < 0 and d2 > 0)) and (
        (d3 > 0 and d4 < 0) or (d3 < 0 and d4 > 0)
    ):
        return True
    if d1 == 0 and min(rx, sx) <= px <= max(rx, sx) and min(ry, sy) <= py <= max(ry, sy):
        return True
    if d2 == 0 and min(rx, sx) <= qx <= max(rx, sx) and min(ry, sy) <= qy <= max(ry, sy):
        return True
    if d3 == 0 and min(px, qx) <= rx <= max(px, qx) and min(py, qy) <= ry <= max(py, qy):
        return True
    if d4 == 0 and min(px, qx) <= sx <= max(px, qx) and min(py, qy) <= sy <= max(py, qy):
        return True
    return False


@njit(cache=True)
def _seg_tri(v, p, q, a, b, c):
    """Closed segment pq touches closed triangle abc (vertex indices into v)."""
    e1x, e1y, e1z = v[b, 0] - v[a, 0], v[b, 1] - v[a, 1], v[b, 2] - v[a, 2]
    e2x, e2y, e2z = v[c, 0] - v[a, 0], v[c, 1] - v[a, 1], v[c, 2] - v[a, 2]
    nx = e1y * e2z - e1z * e2y
    ny = e1z * e2x - e1x * e2z
    nz = e1x * e2y - e1y * e2x
    dp = nx * (v[p, 0] - v[a, 0]) + ny * (v[p, 1] - v[a, 1]) + nz * (v[p, 2] - v[a, 2])
    dq = nx * (v[q, 0] - v[a, 0]) + ny * (v[q, 1] - v[a, 1]) + nz * (v[q, 2] - v[a, 2])
    if (dp > 0 and dq > 0) or (dp < 0 and dq < 0):
        return False
    # Project on the coordinate plane most parallel to the triangle.
    ux, uy, uz = abs(nx), abs(ny), abs(nz)
    if ux >= uy and ux >= uz:
        i, j = 1, 2
    elif uy >= uz:
        i, j = 0, 2
    else:
        i, j = 0, 1
    if dp == 0 and dq == 0:
        # Segment lies in the triangle's plane.
        if _point_in_tri_2d(v[p, i], v[p, j], v[a, i], v[a, j], v[b, i], v[b, j], v[c, i], v[c, j]):
            return True
        if _point_in_tri_2d(v[q, i], v[q, j], v[a, i], v[a, j], v[b, i], v[b, j], v[c, i], v[c, j]):
            return True
        return (
            _seg_seg_2d(v[p, i], v[p, j], v[q, i], v[q, j], v[a, i], v[a, j], v[b, i], v[b, j])
            or _seg_seg_2d(v[p, i], v[p, j], v[q, i], v[q, j], v[b, i], v[b, j], v[c, i], v[c, j])
            or _seg_seg_2d(v[p, i], v[p, j], v[q, i], v[q, j], v[c, i], v[c, j], v[a, i], v[a, j])
        )
    # The crossing point lies on the segment, so a segment that is nearly in
    # the plane but away from the triangle is still rejected reliably.
    t = dp / (dp - dq)
    xi = v[p, i] + t * (v[q, i] - v[p, i])
    xj = v[p, j] + t * (v[q, j] - v[p, j])
    return _point_in_tri_2d(xi, xj, v[a, i], v[a, j], v[b, i], v[b, j], v[c, i], v[c, j])


@njit(cache=True)
def _fold_overlap(v, u, w, a, b):
    """Triangles (u, w, a) and (u, w, b) lie on top of each other across edge uw."""
    dx, dy, dz = v[w, 0] - v[u, 0], v[w, 1] - v[u, 1], v[w, 2] - v[u, 2]
    dd = dx * dx + dy * dy + dz * dz
    if dd == 0:
        return False
    ax, ay, az = v[a, 0] - v[u, 0], v[a, 1] - v[u, 1], v[a, 2] - v[u, 2]
    bx, by, bz = v[b, 0] - v[u, 0], v[b, 1] - v[u, 1], v[b, 2] - v[u, 2]
    ta = (ax * dx + ay * dy + az * dz) / dd
    tb = (bx * dx + by * dy + bz * dz) / dd
    ax, ay, az = ax - ta * dx, ay - ta * dy, az - ta * dz  # parts normal to the edge
    bx, by, bz = bx - tb * dx, by - tb * dy, bz - tb * dz
    dot = ax * bx + ay * by + az * bz
    if dot <= 0:
        return False
    cx, cy, cz = ay * bz - az * by, az * bx - ax * bz, ax * by - ay * bx
    cross2 = cx * cx + cy * cy + cz * cz
    # Same side of the edge and dihedral angle below 1e-9 rad.
    return cross2 <= 1e-18 * (ax * ax + ay * ay + az * az) * (bx * bx + by * by + bz * bz)


@njit(cache=True)
def _tri_tri(v, a0, a1, a2, b0, b1, b2):
    """Triangles intersect beyond the vertices they share by index."""
    a = (a0, a1, a2)
    b = (b0, b1, b2)
    shared = 0
    sa = -1
    sb = -1
    ua = 0  # sum of slots of A that are shared
    ub = 0
    for i in range(3):
        for j in range(3):
            if a[i] == b[j]:
                shared += 1
                sa, sb = i, j
                ua += i
                ub += j
    if shared == 0:
        return (
            _seg_tri(v, a0, a1, b0, b1, b2)
            or _seg_tri(v, a1, a2, b0, b1, b2)
            or _seg_tri(v, a2, a0, b0, b1, b2)
            or _seg_tri(v, b0, b1, a0, a1, a2)
            or _seg_tri(v, b1, b2, a0, a1, a2)
            or _seg_tri(v, b2, b0, a0, a1, a2)
        )
    if shared == 1:
        # Any contact beyond the shared vertex reaches one of the opposite edges.
        p, q = a[(sa + 1) % 3], a[(sa + 2) % 3]
        r, s = b[(sb + 1) % 3], b[(sb + 2) % 3]
        return _seg_tri(v, p, q, b0, b1, b2) or _seg_tri(v, r, s, a0, a1, a2)
    if shared == 2:
        # Slots sum to 3; the unshared slot is what is left.
        ia, ib = 3 - ua, 3 - ub
        return _fold_overlap(v, a[(ia + 1) % 3], a[(ia + 2) % 3], a[ia], b[ib])
    return False  # same three vertices: reported as a duplicate face


@parallel_njit(cache=True)
def _pairs_intersect(v, faces, pairs, skip):
    n = pairs.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for k in prange(n):
        i, j = pairs[k, 0], pairs[k, 1]
        if skip[i] or skip[j]:
            continue
        out[k] = _tri_tri(
            v, faces[i, 0], faces[i, 1], faces[i, 2], faces[j, 0], faces[j, 1], faces[j, 2]
        )
    return out


def triangle_pairs_intersect(mesh: Mesh, pairs, *, skip=None) -> np.ndarray:
    """Exact-as-floats triangle/triangle test for selected face pairs.

    Parameters
    ----------
    mesh : Mesh
    pairs : (k, 2) int array of face indices
    skip : (m,) bool, optional
        Faces to ignore (default: degenerate faces).

    Returns
    -------
    (k,) bool
        True where the two faces share a point that is not explained by the
        vertices they have in common by index:

        * no shared vertex: any contact, including touching;
        * one shared vertex: contact anywhere else;
        * a shared edge: the faces fold back onto each other (coplanar, same
          side of the edge);
        * the same three vertices: False (that is a duplicate face).

    Coordinates are taken relative to the mesh's local origin, so the test is
    as accurate at UTM coordinates as at the origin. Contacts within floating
    point rounding of merely touching may be reported either way.
    """
    pairs = np.ascontiguousarray(np.asarray(pairs, dtype=np.int64).reshape(-1, 2))
    if skip is None:
        skip = _degenerate_faces(mesh, 1e-12)
    if len(pairs) == 0:
        return np.zeros(0, dtype=bool)
    local = np.ascontiguousarray(mesh.vertices - mesh.local_origin)
    return _pairs_intersect(local, mesh.faces, pairs, np.ascontiguousarray(skip, dtype=np.bool_))


# --------------------------------------------------------------------------
# Public entry point
# --------------------------------------------------------------------------


def validate(
    mesh: Mesh, *, check_self_intersection: bool = True, degenerate_tol: float = 1e-12
) -> ValidationReport:
    """Check a mesh for every defect that makes a solid unusable.

    Parameters
    ----------
    mesh : Mesh
    check_self_intersection : bool
        Run the triangle/triangle test (the only step that is not linear in
        the face count). If False the report has ``self_intersections=None``
        and ``is_valid`` is False.
    degenerate_tol : float
        A face is degenerate if twice its area is at most
        ``degenerate_tol * longest_edge**2``, or if its height is below the
        rounding of the stored coordinates, or if it repeats a vertex.

    Returns
    -------
    ValidationReport
    """
    em = mesh.edge_map
    degenerate = _degenerate_faces(mesh, degenerate_tol)
    flip, n_components, orientable = _orientation(mesh)
    cavity = np.zeros(0, dtype=np.int64)
    closed = mesh.n_faces > 0 and not (em.counts != 2).any()
    if flip.any() and closed and not len(_inconsistent_edge_ids(mesh)):
        cavity = _cavity_faces(mesh, flip)
    if check_self_intersection:
        candidates = mesh.bvh.self_overlaps()
        hit = triangle_pairs_intersect(mesh, candidates, skip=degenerate)
        self_intersections = candidates[hit]
    else:
        self_intersections = None
    return ValidationReport(
        n_vertices=mesh.n_vertices,
        n_faces=mesh.n_faces,
        n_components=n_components,
        boundary_edges=em.edges[em.boundary_edges],
        nonmanifold_edges=em.edges[em.nonmanifold_edges],
        nonmanifold_vertices=_nonmanifold_vertices(mesh),
        inconsistent_edges=em.edges[np.sort(_inconsistent_edge_ids(mesh))],
        misoriented_faces=np.flatnonzero(flip),
        orientable=orientable,
        degenerate_faces=np.flatnonzero(degenerate),
        duplicate_faces=_duplicate_faces(mesh),
        self_intersections=self_intersections,
        cavity_faces=cavity,
    )
