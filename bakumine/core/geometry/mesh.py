"""Triangle mesh core: vertex/face arrays, welding, edge map, normals, bounds.

Coordinates are ``x = east, y = north, z = up`` in metres (float64). A
:class:`Mesh` is immutable: its arrays are read-only and every operation
returns a new mesh, so derived data (edge map, normals, spatial indexes) can
be cached safely.

Face winding convention: counter-clockwise seen from outside, so face normals
of a closed solid point outward and its signed volume is positive.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

if TYPE_CHECKING:
    from bakumine.core.geometry.spatial import TriangleBVH, TriangleGridXY
    from bakumine.core.geometry.validation import ValidationReport

__all__ = ["EdgeMap", "Mesh"]


@dataclass(frozen=True)
class EdgeMap:
    """Undirected edges of a mesh and the faces around each of them.

    A *half-edge* is one side of one face: half-edge ``3*f + j`` runs from
    ``faces[f, j]`` to ``faces[f, (j + 1) % 3]``. Every half-edge belongs to
    exactly one undirected edge.

    Attributes
    ----------
    edges : (k, 2) int64
        Unique undirected edges as vertex pairs ``(a, b)`` with ``a <= b``,
        sorted lexicographically.
    face_edges : (m, 3) int64
        ``face_edges[f, j]`` is the edge index of half-edge ``3*f + j``.
    forward : (m, 3) bool
        True where the half-edge runs ``a -> b`` (low to high vertex index).
        Two faces around an edge are consistently wound when exactly one of
        them is forward.
    counts : (k,) int64
        Number of half-edges (adjacent faces) per edge. 1 = boundary,
        2 = manifold interior, more = non-manifold.
    offsets : (k + 1,) int64
        CSR row pointer into ``halfedges``.
    halfedges : (3m,) int64
        Half-edge ids grouped by edge, ascending inside each group. The face
        is ``halfedges // 3`` and the slot in that face ``halfedges % 3``.
    """

    edges: np.ndarray
    face_edges: np.ndarray
    forward: np.ndarray
    counts: np.ndarray
    offsets: np.ndarray
    halfedges: np.ndarray
    n_vertices: int

    @property
    def n_edges(self) -> int:
        return len(self.edges)

    def faces_of(self, edge: int) -> np.ndarray:
        """Faces adjacent to edge index ``edge`` (ascending)."""
        return self.halfedges[self.offsets[edge] : self.offsets[edge + 1]] // 3

    def find(self, a: int, b: int) -> int:
        """Index of the edge between vertices ``a`` and ``b``, or -1."""
        lo, hi = (a, b) if a <= b else (b, a)
        keys = self.edges[:, 0] * self.n_vertices + self.edges[:, 1]
        key = lo * self.n_vertices + hi
        i = int(np.searchsorted(keys, key))
        return i if i < len(keys) and keys[i] == key else -1

    @property
    def boundary_edges(self) -> np.ndarray:
        """Indices of edges with exactly one adjacent face."""
        return np.flatnonzero(self.counts == 1)

    @property
    def nonmanifold_edges(self) -> np.ndarray:
        """Indices of edges with more than two adjacent faces."""
        return np.flatnonzero(self.counts > 2)

    @cached_property
    def manifold_halfedge_pairs(self) -> np.ndarray:
        """(p, 2) half-edge pairs of the edges that have exactly two faces."""
        starts = self.offsets[:-1][self.counts == 2]
        return np.column_stack([self.halfedges[starts], self.halfedges[starts + 1]])


def _as_readonly(a: np.ndarray) -> np.ndarray:
    a.flags.writeable = False
    return a


class Mesh:
    """Indexed triangle mesh.

    Parameters
    ----------
    vertices : (n, 3) array_like of float
        Vertex coordinates (east, north, up).
    faces : (m, 3) array_like of int
        Vertex indices of each triangle.
    check : bool
        Verify shapes, index range and finiteness (default True).

    The input arrays are copied; the stored arrays are read-only.
    """

    def __init__(self, vertices, faces, *, check: bool = True):
        v = np.array(vertices, dtype=np.float64, order="C", ndmin=2, copy=True)
        f = np.array(faces, dtype=np.int64, order="C", ndmin=2, copy=True)
        if v.size == 0:
            v = v.reshape(0, 3)
        if f.size == 0:
            f = f.reshape(0, 3)
        if check:
            if v.ndim != 2 or v.shape[1] != 3:
                raise ValueError(f"vertices must have shape (n, 3), got {v.shape}")
            if f.ndim != 2 or f.shape[1] != 3:
                raise ValueError(f"faces must have shape (m, 3), got {f.shape}")
            if not np.isfinite(v).all():
                raise ValueError("vertices contain NaN or infinite coordinates")
            if f.size and (f.min() < 0 or f.max() >= len(v)):
                raise ValueError("faces reference vertices outside [0, n_vertices)")
        v.flags.writeable = False
        f.flags.writeable = False
        self.vertices: np.ndarray = v
        self.faces: np.ndarray = f

    # ------------------------------------------------------------------ basics

    @property
    def n_vertices(self) -> int:
        return len(self.vertices)

    @property
    def n_faces(self) -> int:
        return len(self.faces)

    def __repr__(self) -> str:
        return f"Mesh(n_vertices={self.n_vertices}, n_faces={self.n_faces})"

    @cached_property
    def triangles(self) -> np.ndarray:
        """(m, 3, 3) corner coordinates of every face."""
        return _as_readonly(self.vertices[self.faces])

    @cached_property
    def bounds(self) -> np.ndarray:
        """(2, 3) axis-aligned bounding box ``[[xmin, ymin, zmin], [xmax, ...]]``.

        Covers all vertices, referenced or not. NaN for a mesh with no vertices.
        """
        if self.n_vertices == 0:
            return _as_readonly(np.full((2, 3), np.nan))
        return _as_readonly(np.array([self.vertices.min(axis=0), self.vertices.max(axis=0)]))

    @cached_property
    def face_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Per-face bounding boxes as ``(lo, hi)``, each (m, 3)."""
        tri = self.triangles
        return _as_readonly(tri.min(axis=1)), _as_readonly(tri.max(axis=1))

    @cached_property
    def local_origin(self) -> np.ndarray:
        """(3,) centre of the bounding box.

        Numerical routines subtract it first so that results do not lose
        precision for meshes far from the coordinate origin (UTM-sized values).
        """
        if self.n_vertices == 0:
            return _as_readonly(np.zeros(3))
        return _as_readonly(0.5 * (self.bounds[0] + self.bounds[1]))

    # ----------------------------------------------------------------- normals

    @cached_property
    def _face_cross(self) -> np.ndarray:
        """(m, 3) ``(b - a) x (c - a)``: length is twice the face area."""
        tri = self.triangles
        return np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])

    @cached_property
    def face_areas(self) -> np.ndarray:
        """(m,) area of every face."""
        return _as_readonly(0.5 * np.linalg.norm(self._face_cross, axis=1))

    @cached_property
    def face_normals(self) -> np.ndarray:
        """(m, 3) unit normals (right-hand rule on the winding).

        Zero-area faces get a zero normal.
        """
        cross = self._face_cross
        length = np.linalg.norm(cross, axis=1, keepdims=True)
        out = np.zeros_like(cross)
        np.divide(cross, length, out=out, where=length > 0)
        return _as_readonly(out)

    @cached_property
    def vertex_normals(self) -> np.ndarray:
        """(n, 3) unit vertex normals, area-weighted mean of adjacent faces.

        Vertices with no faces (or cancelling faces) get a zero normal.
        """
        acc = np.zeros((self.n_vertices, 3))
        cross = self._face_cross
        for j in range(3):
            for axis in range(3):
                acc[:, axis] += np.bincount(
                    self.faces[:, j], weights=cross[:, axis], minlength=self.n_vertices
                )
        length = np.linalg.norm(acc, axis=1, keepdims=True)
        out = np.zeros_like(acc)
        np.divide(acc, length, out=out, where=length > 0)
        return _as_readonly(out)

    # ---------------------------------------------------------------- topology

    @cached_property
    def edge_map(self) -> EdgeMap:
        """Unique edges and their adjacent faces (see :class:`EdgeMap`)."""
        f = self.faces
        n = max(self.n_vertices, 1)
        tail = f.reshape(-1)  # half-edge 3f+j starts at faces[f, j]
        head = f[:, [1, 2, 0]].reshape(-1)
        lo = np.minimum(tail, head)
        hi = np.maximum(tail, head)
        keys = lo * n + hi
        unique_keys, inverse, counts = np.unique(keys, return_inverse=True, return_counts=True)
        inverse = inverse.reshape(-1)
        edges = np.column_stack([unique_keys // n, unique_keys % n]).astype(np.int64)
        offsets = np.zeros(len(unique_keys) + 1, dtype=np.int64)
        np.cumsum(counts, out=offsets[1:])
        halfedges = np.argsort(inverse, kind="stable").astype(np.int64)
        return EdgeMap(
            edges=_as_readonly(edges),
            face_edges=_as_readonly(inverse.astype(np.int64).reshape(-1, 3)),
            forward=_as_readonly((tail <= head).reshape(-1, 3)),
            counts=_as_readonly(counts.astype(np.int64)),
            offsets=_as_readonly(offsets),
            halfedges=_as_readonly(halfedges),
            n_vertices=n,
        )

    @property
    def edges(self) -> np.ndarray:
        """(k, 2) unique undirected edges, each as ``(low, high)`` vertex index."""
        return self.edge_map.edges

    @property
    def is_closed(self) -> bool:
        """Every edge is shared by exactly two faces (and there is at least one face)."""
        return self.n_faces > 0 and bool((self.edge_map.counts == 2).all())

    @property
    def is_winding_consistent(self) -> bool:
        """Every two-face edge is traversed once in each direction."""
        em = self.edge_map
        pairs = em.manifold_halfedge_pairs
        forward = em.forward.reshape(-1)
        return bool((forward[pairs[:, 0]] != forward[pairs[:, 1]]).all())

    # -------------------------------------------------------------- validation

    def validate(
        self, *, check_self_intersection: bool = True, degenerate_tol: float = 1e-12
    ) -> ValidationReport:
        """Full defect report; see :func:`bakumine.core.geometry.validation.validate`."""
        from bakumine.core.geometry.validation import validate

        return validate(
            self, check_self_intersection=check_self_intersection, degenerate_tol=degenerate_tol
        )

    def fix_orientation(self) -> Mesh:
        """Mesh with consistent, outward winding (positive volume for closed shells).

        See :func:`bakumine.core.geometry.validation.fix_orientation`.
        """
        from bakumine.core.geometry.validation import fix_orientation

        return fix_orientation(self)

    # ----------------------------------------------------------------- cleanup

    def weld(self, tolerance: float = 1e-9, *, remove_collapsed: bool = True) -> Mesh:
        """Merge vertices closer than ``tolerance`` (metres) into one.

        Merging is transitive: a chain of vertices each within ``tolerance``
        of the next becomes a single vertex. Each cluster keeps the
        coordinates of its lowest-index member, and surviving vertices keep
        their relative order, so welding a welded mesh changes nothing.

        Faces that collapse (two corners merged into the same vertex) are
        dropped unless ``remove_collapsed`` is False. Unreferenced vertices
        are kept; see :meth:`remove_unreferenced_vertices`.
        """
        if tolerance < 0:
            raise ValueError("tolerance must be >= 0")
        n = self.n_vertices
        if n == 0:
            return self
        v = self.vertices
        # Exact duplicates first (cheap, and the common case for triangle soup).
        order = np.lexsort((v[:, 2], v[:, 1], v[:, 0]))
        sv = v[order]
        new_group = np.ones(n, dtype=bool)
        new_group[1:] = (sv[1:] != sv[:-1]).any(axis=1)
        group_sorted = np.cumsum(new_group) - 1
        n_groups = int(group_sorted[-1]) + 1
        label = np.empty(n, dtype=np.int64)
        label[order] = group_sorted

        if tolerance > 0:
            first = order[new_group]  # one member per exact group
            pairs = cKDTree(v[first]).query_pairs(tolerance, output_type="ndarray")
            if len(pairs):
                graph = coo_matrix(
                    (np.ones(len(pairs), dtype=np.int8), (pairs[:, 0], pairs[:, 1])),
                    shape=(n_groups, n_groups),
                )
                _, merged = connected_components(graph, directed=False)
                label = merged[label]

        # Representative of each cluster = lowest original index; keep order.
        n_clusters = int(label.max()) + 1
        rep = np.full(n_clusters, n, dtype=np.int64)
        np.minimum.at(rep, label, np.arange(n))
        keep = np.sort(rep)
        rank = np.empty(n_clusters, dtype=np.int64)
        rank[np.argsort(rep)] = np.arange(n_clusters)
        remap = rank[label]

        faces = remap[self.faces]
        if remove_collapsed and len(faces):
            ok = (
                (faces[:, 0] != faces[:, 1])
                & (faces[:, 1] != faces[:, 2])
                & (faces[:, 2] != faces[:, 0])
            )
            faces = faces[ok]
        return Mesh(v[keep], faces, check=False)

    def remove_unreferenced_vertices(self) -> Mesh:
        """Drop vertices that no face uses; remaining vertices keep their order."""
        used = np.zeros(self.n_vertices, dtype=bool)
        used[self.faces.reshape(-1)] = True
        if used.all():
            return self
        remap = np.cumsum(used) - 1
        return Mesh(self.vertices[used], remap[self.faces], check=False)

    # -------------------------------------------------------------- derivation

    def with_faces(self, faces) -> Mesh:
        """Same vertices, different faces."""
        return Mesh(self.vertices, faces)

    def submesh(self, face_mask) -> Mesh:
        """Keep the faces selected by a boolean mask or index array (vertices unchanged)."""
        return Mesh(self.vertices, self.faces[np.asarray(face_mask)], check=False)

    def flipped(self, faces=None) -> Mesh:
        """Reverse the winding of the selected faces (all faces if None)."""
        f = self.faces.copy()
        sel = slice(None) if faces is None else np.asarray(faces)
        f[sel] = f[sel][..., ::-1]
        return Mesh(self.vertices, f, check=False)

    def translated(self, offset) -> Mesh:
        """Mesh moved by ``offset`` (3,)."""
        return Mesh(self.vertices + np.asarray(offset, dtype=np.float64), self.faces, check=True)

    def transformed(self, matrix) -> Mesh:
        """Apply a 4x4 homogeneous (or 3x3 linear) transform to the vertices.

        Winding is flipped automatically when the transform is a reflection
        (negative determinant) so that normals keep pointing the same way
        relative to the surface.
        """
        m = np.asarray(matrix, dtype=np.float64)
        if m.shape == (4, 4):
            linear, shift = m[:3, :3], m[:3, 3]
        elif m.shape == (3, 3):
            linear, shift = m, np.zeros(3)
        else:
            raise ValueError("matrix must be 3x3 or 4x4")
        v = self.vertices @ linear.T + shift
        f = self.faces[:, ::-1] if np.linalg.det(linear) < 0 else self.faces
        return Mesh(v, f, check=True)

    @staticmethod
    def concatenate(meshes: Sequence[Mesh]) -> Mesh:
        """Join meshes into one (no welding; vertex indices are offset)."""
        if not meshes:
            return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))
        offsets = np.cumsum([0] + [m.n_vertices for m in meshes[:-1]])
        vertices = np.concatenate([m.vertices for m in meshes])
        faces = np.concatenate([m.faces + o for m, o in zip(meshes, offsets, strict=True)])
        return Mesh(vertices, faces, check=False)

    # ---------------------------------------------------------- spatial index

    @cached_property
    def grid_xy(self) -> TriangleGridXY:
        """Plan-view uniform grid over the triangles (built once, cached).

        Use for vertical ray / column queries; see
        :class:`bakumine.core.geometry.spatial.TriangleGridXY`.
        """
        from bakumine.core.geometry.spatial import TriangleGridXY

        return TriangleGridXY(self.vertices, self.faces)

    @cached_property
    def bvh(self) -> TriangleBVH:
        """Bounding volume hierarchy over the triangles (built once, cached).

        Use for general box queries; see
        :class:`bakumine.core.geometry.spatial.TriangleBVH`.
        """
        from bakumine.core.geometry.spatial import TriangleBVH

        return TriangleBVH(self.vertices, self.faces)
