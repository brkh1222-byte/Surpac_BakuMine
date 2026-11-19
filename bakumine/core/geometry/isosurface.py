"""Isosurface of a scalar field sampled on a regular grid: marching tetrahedra.

The field is known at the nodes of a regular (optionally rotated) grid

    node[i, j, k] = origin + rotation @ (i * dx, j * dy, k * dz)

Every grid cell is split into six tetrahedra that share the cell's main
diagonal (node ``(0,0,0)`` to node ``(1,1,1)`` of the cell; the Kuhn /
Freudenthal subdivision). Tetrahedron ``(a, b, c)``, one per ordering of the
three axes, has the vertices

    0, e_a, e_a + e_b, e_a + e_b + e_c

The subdivision is the same in every cell (it is invariant under grid
translations), so the diagonals drawn on the shared face of two neighbouring
cells agree and the tetrahedra form a proper simplicial complex of the whole
grid. Its edges run from a node in one of seven directions: three axes, three
face diagonals and the body diagonal.

Each node is classified *inside* or *outside* (a strict comparison with the
level: a node exactly at the level is outside). An edge whose ends differ
carries exactly one surface vertex, placed by linear interpolation. Within a
tetrahedron one or three inside nodes give one triangle, two give a
quadrilateral (two triangles). Because the classification is binary and a
tetrahedron has no ambiguous configuration, every mixed triangular face of the
complex carries exactly one surface segment, shared by the two tetrahedra on
that face: the surface is closed and manifold by construction, provided no
mixed edge is cut by the boundary of the grid. Padding the field with one
layer of outside nodes guarantees that.

Vertices are identified by the *edge* they sit on (an integer key, node index
times 8 plus the direction mask), never by comparing coordinates, so welding
is exact at any coordinate magnitude. The interpolation parameter is kept away
from the two ends of the edge by ``min_fraction``; this keeps the vertices of
distinct edges that meet at a node apart, so no triangle is degenerate. It
moves the surface by at most ``min_fraction`` of an edge length.

Triangles are wound so that the normal points from the inside to the outside
nodes (outward for a solid whose interior is the inside set).

The method follows the published idea (marching cubes of Lorensen & Cline
1987 applied to a tetrahedral subdivision, e.g. Doi & Koide 1991, Bloomenthal
1994); the case tables below are generated at import time from the geometry,
not copied from any implementation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from bakumine.core.geometry.mesh import Mesh

__all__ = [
    "GridSpec",
    "marching_tetrahedra",
    "mesh_components",
    "remove_small_components",
]


# --------------------------------------------------------------------------
# grid description
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class GridSpec:
    """A regular grid of nodes, same convention as ``grid_in_solid``.

    ``node[i, j, k] = origin + rotation @ (i * dx, j * dy, k * dz)``;
    ``rotation`` columns are the world directions of the i, j, k axes
    (``None`` = world axes).
    """

    origin: np.ndarray
    spacing: np.ndarray
    shape: tuple[int, int, int]
    rotation: np.ndarray | None = None

    def __post_init__(self) -> None:
        origin = np.asarray(self.origin, dtype=np.float64).reshape(-1)
        spacing = np.asarray(self.spacing, dtype=np.float64).reshape(-1)
        if origin.shape != (3,) or spacing.shape != (3,):
            raise ValueError("origin and spacing must have 3 components")
        if not (np.isfinite(origin).all() and np.isfinite(spacing).all() and (spacing > 0).all()):
            raise ValueError("origin must be finite and spacing must be finite and > 0")
        shape = tuple(int(s) for s in self.shape)
        if len(shape) != 3 or min(shape) < 0:
            raise ValueError("shape must be three non-negative integers")
        rotation = self.rotation
        if rotation is not None:
            rotation = np.asarray(rotation, dtype=np.float64)
            if rotation.shape != (3, 3) or not np.allclose(
                rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-9
            ):
                raise ValueError("rotation must be an orthonormal 3x3 matrix")
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "spacing", spacing)
        object.__setattr__(self, "shape", shape)
        object.__setattr__(self, "rotation", rotation)

    @property
    def n_nodes(self) -> int:
        return int(np.prod(self.shape))

    @property
    def axes(self) -> np.ndarray:
        """(3, 3) matrix, columns = world directions of the grid axes."""
        return np.eye(3) if self.rotation is None else self.rotation

    def to_world(self, index_coords) -> np.ndarray:
        """World coordinates of (fractional) node indices ``(..., 3)``."""
        local = np.asarray(index_coords, dtype=np.float64) * self.spacing
        if self.rotation is not None:
            local = local @ self.rotation.T
        return self.origin + local

    def slab_points(self, i0: int, i1: int) -> np.ndarray:
        """(i1 - i0, ny, nz, 3) world coordinates of the nodes with ``i0 <= i < i1``."""
        _, ny, nz = self.shape
        idx = np.empty((i1 - i0, ny, nz, 3))
        idx[..., 0] = np.arange(i0, i1)[:, None, None]
        idx[..., 1] = np.arange(ny)[None, :, None]
        idx[..., 2] = np.arange(nz)[None, None, :]
        return self.to_world(idx)

    def to_dict(self) -> dict:
        return {
            "origin": self.origin.tolist(),
            "spacing": self.spacing.tolist(),
            "shape": list(self.shape),
            "rotation": None if self.rotation is None else self.rotation.tolist(),
        }


# --------------------------------------------------------------------------
# case tables, generated from the geometry
# --------------------------------------------------------------------------


def _build_tables():
    """Per tetrahedron and inside-pattern: the triangles, as edges of the cell.

    Returns ``(tets, table)``. ``tets`` is (6, 4) corner numbers (corner number
    = dx + 2 dy + 4 dz). ``table[t][case]`` is an int array (n_tri, 3, 2):
    for each triangle corner the edge it sits on, as (base corner, direction
    mask); ``case`` has bit ``m`` set when tetrahedron vertex ``m`` is inside.
    """
    unit = np.eye(3, dtype=np.int64)
    tets = []
    for perm in permutations(range(3)):
        v = np.zeros(3, dtype=np.int64)
        corners = [0]
        for axis in perm:
            v = v + unit[axis]
            corners.append(int(v[0] + 2 * v[1] + 4 * v[2]))
        tets.append(corners)
    tets = np.array(tets, dtype=np.int64)

    def position(corner: int) -> np.ndarray:
        return np.array([corner & 1, (corner >> 1) & 1, (corner >> 2) & 1], dtype=np.float64)

    def edge(c0: int, c1: int) -> tuple[int, int]:
        # the vertices of a Kuhn tetrahedron form a chain, so one end dominates
        lo, hi = (c0, c1) if c0 < c1 else (c1, c0)
        assert lo & hi == lo
        return lo, hi - lo

    def midpoint(c0: int, c1: int) -> np.ndarray:
        return 0.5 * (position(c0) + position(c1))

    table = []
    for corners in tets:
        cases = [np.zeros((0, 3, 2), dtype=np.int64)]
        for case in range(1, 15):
            inside = [c for m, c in enumerate(corners) if case >> m & 1]
            outside = [c for m, c in enumerate(corners) if not case >> m & 1]
            if len(inside) == 1:
                polygon = [(inside[0], o) for o in outside]
            elif len(outside) == 1:
                polygon = [(i, outside[0]) for i in inside]
            else:
                (a, b), (c, d) = inside, outside
                polygon = [(a, c), (a, d), (b, d), (b, c)]  # cyclic around the quad
            tris = [polygon[:3]]
            if len(polygon) == 4:
                tris.append([polygon[0], polygon[2], polygon[3]])
            p = [midpoint(*e) for e in tris[0]]
            normal = np.cross(p[1] - p[0], p[2] - p[0])
            outward = np.mean([position(c) for c in outside], axis=0) - np.mean(
                [position(c) for c in inside], axis=0
            )
            dot = float(normal @ outward)
            assert abs(dot) > 1e-9
            if dot < 0.0:
                tris = [t[::-1] for t in tris]
            cases.append(np.array([[edge(*e) for e in t] for t in tris], dtype=np.int64))
        cases.append(np.zeros((0, 3, 2), dtype=np.int64))
        table.append(cases)
    return tets, table


_TETS, _TABLE = _build_tables()


# --------------------------------------------------------------------------
# extraction
# --------------------------------------------------------------------------


def marching_tetrahedra(
    values,
    origin=(0.0, 0.0, 0.0),
    spacing=(1.0, 1.0, 1.0),
    rotation=None,
    *,
    level: float = 0.0,
    inside: str = "above",
    pad: bool = True,
    min_fraction: float = 1e-3,
) -> Mesh:
    """Triangulate the surface ``values == level`` of a field on a regular grid.

    Parameters
    ----------
    values : (nx, ny, nz) array_like of float
        Field at the grid nodes. NaN nodes are outside.
    origin, spacing, rotation
        The grid: ``node[i, j, k] = origin + rotation @ (i dx, j dy, k dz)``
        (``rotation`` columns = world directions of the grid axes; None = world
        axes). Vertices are computed in grid coordinates and moved to the world
        last, so large coordinates cost no precision in the topology.
    level : float
        Isovalue.
    inside : "above" or "below"
        Which side is the interior: ``values > level`` (default, the convention
        of a signed distance that is positive inside) or ``values < level``.
    pad : bool
        Surround the grid with one layer of outside nodes (default), so the
        surface is closed where the interior reaches the grid boundary: it is
        capped there by flat faces on the boundary planes. With ``pad=False``
        the surface is left open at the boundary.
    min_fraction : float
        Smallest distance of a vertex from either end of its edge, as a
        fraction of the edge (0 < min_fraction < 0.5).

    Returns
    -------
    Mesh
        Vertices shared between triangles (one per cut edge). Closed, manifold
        and outward-oriented when ``pad`` is True. Empty when no edge is cut.
    """
    if inside not in ("above", "below"):
        raise ValueError(f"inside must be 'above' or 'below', got {inside!r}")
    if not 0.0 < min_fraction < 0.5:
        raise ValueError("min_fraction must be in (0, 0.5)")
    field = np.asarray(values, dtype=np.float64)
    if field.ndim != 3:
        raise ValueError(f"values must be a 3-D array, got shape {field.shape}")
    grid = GridSpec(origin, spacing, field.shape, rotation)
    level = float(level)
    if not np.isfinite(level):
        raise ValueError("level must be finite")
    if inside == "below":
        field, level = -field, -level
    shift = 0
    if pad:
        field = np.pad(field, 1, mode="constant", constant_values=-np.inf)
        shift = 1
    nx, ny, nz = field.shape
    if min(nx, ny, nz) < 2:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    inside_node = field > level  # NaN -> False
    # cells with both kinds of corner
    any_in = np.zeros((nx - 1, ny - 1, nz - 1), dtype=bool)
    all_in = np.ones((nx - 1, ny - 1, nz - 1), dtype=bool)
    for corner in range(8):
        dx, dy, dz = corner & 1, (corner >> 1) & 1, (corner >> 2) & 1
        view = inside_node[dx : dx + nx - 1, dy : dy + ny - 1, dz : dz + nz - 1]
        any_in |= view
        all_in &= view
    ci, cj, ck = np.nonzero(any_in & ~all_in)
    del any_in, all_in
    if ci.size == 0:
        return Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=np.int64))

    stride = np.array([ny * nz, nz, 1], dtype=np.int64)
    cell_node = ci * stride[0] + cj * stride[1] + ck  # flat index of the cell's corner 0
    corner_offset = np.array(
        [(c & 1) * stride[0] + ((c >> 1) & 1) * stride[1] + ((c >> 2) & 1) for c in range(8)],
        dtype=np.int64,
    )
    flat_inside = inside_node.reshape(-1)
    corner_in = np.stack([flat_inside[cell_node + corner_offset[c]] for c in range(8)], axis=1)

    key_blocks = []
    for t in range(6):
        bits = corner_in[:, _TETS[t]]
        case = bits[:, 0] + 2 * bits[:, 1] + 4 * bits[:, 2] + 8 * bits[:, 3].astype(np.int64)
        for c in range(1, 15):
            cells = cell_node[case == c]
            if cells.size == 0:
                continue
            tris = _TABLE[t][c]  # (n_tri, 3, 2)
            base = corner_offset[tris[..., 0]]  # (n_tri, 3)
            keys = (cells[:, None, None] + base[None]) * 8 + tris[None, ..., 1]
            key_blocks.append(keys.reshape(-1, 3))
    tri_keys = np.concatenate(key_blocks)
    keys, faces = np.unique(tri_keys.reshape(-1), return_inverse=True)
    faces = faces.reshape(-1, 3)

    # vertex positions from the edge keys
    node_a = keys >> 3
    mask = keys & 7
    step = np.stack([mask & 1, (mask >> 1) & 1, (mask >> 2) & 1], axis=1)
    node_b = node_a + step @ stride
    flat = field.reshape(-1)
    va, vb = flat[node_a], flat[node_b]
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        frac = (level - va) / (vb - va)
    # a non-finite end (padding, NaN, infinity): the vertex goes next to the other end
    frac = np.where(np.isfinite(vb), frac, 0.0)
    frac = np.where(np.isfinite(va), frac, 1.0)
    frac = np.where(np.isfinite(va) | np.isfinite(vb), frac, 0.5)
    frac = np.where(np.isfinite(frac), frac, 0.5)
    frac = np.clip(frac, min_fraction, 1.0 - min_fraction)

    ijk = np.stack(np.unravel_index(node_a, (nx, ny, nz)), axis=1).astype(np.float64)
    vertices = grid.to_world(ijk + frac[:, None] * step - shift)
    return Mesh(vertices, faces)


# --------------------------------------------------------------------------
# connected components
# --------------------------------------------------------------------------


def mesh_components(mesh: Mesh) -> tuple[int, np.ndarray, np.ndarray]:
    """Edge-connected components of a mesh.

    Returns ``(n, face_label, signed_volume)``: the number of components, the
    component of each face, and each component's signed volume (positive for
    an outward-oriented closed shell, negative for a cavity), computed about
    the mesh's local origin.
    """
    if mesh.n_faces == 0:
        return 0, np.zeros(0, dtype=np.int64), np.zeros(0)
    f = mesh.faces
    n = mesh.n_vertices
    rows = np.concatenate([f[:, 0], f[:, 1]])
    cols = np.concatenate([f[:, 1], f[:, 2]])
    graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    _, vertex_label = connected_components(graph, directed=False)
    face_label = vertex_label[f[:, 0]]
    used, face_label = np.unique(face_label, return_inverse=True)
    tri = mesh.vertices[f] - mesh.local_origin
    six_v = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2]))
    volume = np.bincount(face_label, weights=six_v, minlength=len(used)) / 6.0
    return len(used), face_label, volume


def remove_small_components(
    mesh: Mesh, min_volume: float, *, keep_largest: bool = True
) -> tuple[Mesh, int]:
    """Drop the connected components whose ``|volume|`` is below ``min_volume``.

    A small cavity is removed like a small shell. With ``keep_largest`` the
    component of largest volume is never removed. Returns ``(mesh, n_removed)``.
    """
    n, label, volume = mesh_components(mesh)
    if n <= 1 or not min_volume > 0.0:
        return mesh, 0
    keep = np.abs(volume) >= float(min_volume)
    if keep_largest:
        keep[int(np.argmax(volume))] = True
    if keep.all():
        return mesh, 0
    out = mesh.submesh(keep[label])
    return out, int((~keep).sum())
