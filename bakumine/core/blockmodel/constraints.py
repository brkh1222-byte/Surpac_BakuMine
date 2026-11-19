"""Constraints: logical tests that select blocks (research.md section 6.4).

A :class:`Constraint` answers two questions:

``mask(model, resolver=None)``
    One boolean per row of the block table: does the block's **centroid**
    pass the test?

``classify_boxes(definition, lo, hi, resolver=None)``
    For boxes that are axis-aligned in the model frame (``lo``, ``hi`` are
    (n, 3) minimum / maximum corners in model-local coordinates, e.g. from
    :meth:`BlockModel.bounds_local`): is the box entirely selected
    (:data:`INSIDE`), entirely rejected (:data:`OUTSIDE`) or cut by the
    boundary (:data:`STRADDLE`)? Sub-blocking uses this to decide which
    blocks to split. Available for the geometric primitives and their
    combinations; attribute and block-id constraints raise
    :class:`ConstraintError`.

    The answer is **conservative**: ``INSIDE`` / ``OUTSIDE`` are only
    reported when every point in the interior of the box gives the same
    answer; ``STRADDLE`` may be reported for a box that is in fact on one
    side (for instance when the boundary merely touches it). Precisely:

    * solid, DTM: exact triangle-box overlap tests (separating axes) against
      every triangle near the box, with the box grown by a rounding margin
      (64 ulp of the largest coordinate, about 6e-8 m at northing 5.3e6);
      a box that no triangle reaches is classified by its centre. A box
      that crosses the plan outline of a DTM is ``STRADDLE``. For a rotated
      model the outline test uses the box's world bounding rectangle, which
      can over-report.
    * plane: sign of the distance of the 8 corners. A corner exactly on the
      plane (computed distance 0) counts for either side, so a plane lying
      on block faces does not make those blocks ``STRADDLE``; any other
      distance smaller than the rounding margin does.
    * polygon: polygon edges against the bounding rectangle of the box in
      the projection plane (exact rectangle for an unrotated model and a
      projection along x, y or z; otherwise it over-reports), then the
      limits along the projection axis.
    * coordinate / index ranges: exact interval comparisons under the
      half-open rule (with the rounding margin for world-coordinate ranges
      on a rotated model).

    Combinations propagate the three-valued answer (``AND`` is inside when
    all parts are, outside when any part is; and so on), which stays
    conservative but can report ``STRADDLE`` where the parts' boundaries
    cancel.

Primitives
----------
====================  =====================================================
:class:`DtmConstraint`      above / below a surface, with a policy outside its
                            plan extent
:class:`SolidConstraint`    inside / outside a closed mesh
:class:`AttributeConstraint` an expression over attributes: ``au > 0.5 and
                            rock == "ORE"``
:class:`PlaneConstraint`    above / below a plane (point + normal, three
                            points, or strike / dip)
:class:`PolygonConstraint`  inside / outside a closed polygon projected along
                            x, y, z or any direction, optionally limited
                            along that direction
:class:`RangeConstraint`    x / y / z coordinate ranges (bench / elevation
                            range) and i / j / k index ranges
:class:`BlockIdConstraint`  an explicit set of block ids
====================  =====================================================

Sides are complementary: ``below`` is exactly ``not above`` and ``outside``
is ``not inside`` (except where a DTM does not cover the block; see
:class:`DtmConstraint`). A centroid exactly on a surface or plane is
``below`` it; on the boundary of a solid or polygon the half-open rules of
the underlying tests decide.

Combination
-----------
``a & b``, ``a | b``, ``~a`` or :class:`And`, :class:`Or`, :class:`Not` build
an expression tree. :func:`combine` builds one from text over named
constraints, as Surpac users write them::

    combine("ore and (below_topo or not dyke)", {"ore": ..., "below_topo": ..., "dyke": ...})

Saving
------
:meth:`Constraint.to_dict` / :func:`constraint_from_dict` and
:func:`save_constraint` / :func:`load_constraint` (JSON). Meshes are not
written into the JSON: a DTM or solid is referred to **by name**, and the
name is turned into geometry by a *resolver* passed to ``mask`` /
``classify_boxes`` (:class:`GeometryResolver`; :class:`DictResolver` and
:class:`FileResolver` are provided, and a project system can supply its own).
Planes, ranges, block ids and polygons given as coordinates are embedded.
"""

from __future__ import annotations

import ast
import json
import re
import weakref
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, ClassVar, Protocol, runtime_checkable

import numpy as np
from numba import njit, prange

from bakumine.core.blockmodel import expressions as _expressions
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.inside import grid_in_solid, points_in_solid
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.section import plane_frame

__all__ = [
    "INSIDE",
    "OUTSIDE",
    "STRADDLE",
    "And",
    "AttributeConstraint",
    "BlockIdConstraint",
    "CombinedConstraint",
    "Constraint",
    "ConstraintError",
    "DictResolver",
    "DtmConstraint",
    "FileResolver",
    "GeometryResolver",
    "Not",
    "Or",
    "PlaneConstraint",
    "PolygonConstraint",
    "RangeConstraint",
    "SolidConstraint",
    "combine",
    "constraint_from_dict",
    "load_constraint",
    "save_constraint",
]

#: box classification codes (int8)
INSIDE = 1
OUTSIDE = -1
STRADDLE = 0

FORMAT_VERSION = 1


class ConstraintError(ValueError):
    """A constraint is invalid, cannot be resolved, or does not support the operation."""


# --------------------------------------------------------------------------
# geometry resolvers
# --------------------------------------------------------------------------


@runtime_checkable
class GeometryResolver(Protocol):
    """Turns the geometry names stored in constraints into objects.

    Implement these three methods to plug in a project system. Each raises
    ``KeyError`` (or :class:`ConstraintError`) for an unknown name.
    """

    def dtm(self, name: str) -> DTM: ...

    def solid(self, name: str) -> Mesh: ...

    def polygon(self, name: str) -> np.ndarray: ...


class DictResolver:
    """Resolver backed by dictionaries of objects already in memory."""

    def __init__(
        self,
        dtms: Mapping[str, DTM] | None = None,
        solids: Mapping[str, Mesh] | None = None,
        polygons: Mapping[str, Any] | None = None,
    ):
        self.dtms = dict(dtms or {})
        self.solids = dict(solids or {})
        self.polygons = dict(polygons or {})

    def dtm(self, name: str) -> DTM:
        return self.dtms[name]

    def solid(self, name: str) -> Mesh:
        return self.solids[name]

    def polygon(self, name: str) -> np.ndarray:
        return np.asarray(self.polygons[name], dtype=np.float64)


class FileResolver:
    """Resolver that loads geometry from numpy files in one directory.

    ``solid("ore")`` and ``dtm("topo")`` read ``<name>_vertices.npy`` and
    ``<name>_faces.npy``; ``polygon("pit")`` reads ``<name>.npy`` ((n, 2) or
    (n, 3)). Names are plain file stems (letters, digits, ``_``, ``-``,
    ``.``); anything that could leave the directory is rejected. Loaded
    objects are cached.
    """

    _NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self._cache: dict[tuple[str, str], Any] = {}

    def _path(self, name: str, suffix: str) -> Path:
        if not isinstance(name, str) or not self._NAME.match(name) or ".." in name:
            raise ConstraintError(f"invalid geometry name {name!r}")
        path = self.directory / f"{name}{suffix}"
        if not path.is_file():
            raise KeyError(f"geometry {name!r}: {path} not found")
        return path

    def _mesh(self, name: str) -> Mesh:
        key = ("mesh", name)
        if key not in self._cache:
            vertices = np.load(self._path(name, "_vertices.npy"), allow_pickle=False)
            faces = np.load(self._path(name, "_faces.npy"), allow_pickle=False)
            self._cache[key] = Mesh(vertices, faces)
        return self._cache[key]

    def solid(self, name: str) -> Mesh:
        return self._mesh(name)

    def dtm(self, name: str) -> DTM:
        key = ("dtm", name)
        if key not in self._cache:
            self._cache[key] = DTM(self._mesh(name))
        return self._cache[key]

    def polygon(self, name: str) -> np.ndarray:
        key = ("polygon", name)
        if key not in self._cache:
            self._cache[key] = np.load(self._path(name, ".npy"), allow_pickle=False)
        return self._cache[key]


def _resolve(ref: Any, kind: str, resolver: Any) -> Any:
    if not isinstance(ref, str):
        return ref
    if resolver is None:
        raise ConstraintError(
            f"the constraint refers to the {kind} {ref!r} by name; pass a resolver to look it up"
        )
    try:
        return getattr(resolver, kind)(ref)
    except KeyError as exc:
        raise ConstraintError(f"unknown {kind} {ref!r}: {exc}") from None


def _reference(ref: Any, kind: str) -> str:
    if isinstance(ref, str):
        return ref
    raise ConstraintError(
        f"this constraint holds a {kind} object, which is not written to JSON; "
        "create the constraint with the geometry's name and use a resolver"
    )


# --------------------------------------------------------------------------
# numeric helpers
# --------------------------------------------------------------------------


@njit(cache=True, parallel=True)
def _tri_box_pairs(vertices, faces, lo, hi, box_index, tri_index):
    """Exact triangle / box overlap (separating axis theorem) for candidate pairs.

    Touching counts as overlapping. Degenerate triangles (segments) are fine.
    """
    n = box_index.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for p in prange(n):
        b = box_index[p]
        t = tri_index[p]
        c = np.empty(3)
        h = np.empty(3)
        v = np.empty((3, 3))
        for a in range(3):
            c[a] = 0.5 * (lo[b, a] + hi[b, a])
            h[a] = 0.5 * (hi[b, a] - lo[b, a])
        for k in range(3):
            for a in range(3):
                v[k, a] = vertices[faces[t, k], a] - c[a]
        separated = False
        # box face normals
        for a in range(3):
            if min(v[0, a], v[1, a], v[2, a]) > h[a] or max(v[0, a], v[1, a], v[2, a]) < -h[a]:
                separated = True
        if not separated:
            # cross products of triangle edges and box axes
            for k in range(3):
                k2 = (k + 1) % 3
                ex = v[k2, 0] - v[k, 0]
                ey = v[k2, 1] - v[k, 1]
                ez = v[k2, 2] - v[k, 2]
                for a in range(3):
                    if a == 0:
                        ax, ay, az = 0.0, -ez, ey
                    elif a == 1:
                        ax, ay, az = ez, 0.0, -ex
                    else:
                        ax, ay, az = -ey, ex, 0.0
                    p0 = ax * v[0, 0] + ay * v[0, 1] + az * v[0, 2]
                    p1 = ax * v[1, 0] + ay * v[1, 1] + az * v[1, 2]
                    p2 = ax * v[2, 0] + ay * v[2, 1] + az * v[2, 2]
                    r = h[0] * abs(ax) + h[1] * abs(ay) + h[2] * abs(az)
                    if min(p0, p1, p2) > r or max(p0, p1, p2) < -r:
                        separated = True
        if not separated:
            # triangle plane
            e0x = v[1, 0] - v[0, 0]
            e0y = v[1, 1] - v[0, 1]
            e0z = v[1, 2] - v[0, 2]
            e1x = v[2, 0] - v[0, 0]
            e1y = v[2, 1] - v[0, 1]
            e1z = v[2, 2] - v[0, 2]
            nx = e0y * e1z - e0z * e1y
            ny = e0z * e1x - e0x * e1z
            nz = e0x * e1y - e0y * e1x
            d = nx * v[0, 0] + ny * v[0, 1] + nz * v[0, 2]
            r = h[0] * abs(nx) + h[1] * abs(ny) + h[2] * abs(nz)
            if abs(d) > r:
                separated = True
        out[p] = not separated
    return out


def _boxes_hit_mesh(mesh: Mesh, lo: np.ndarray, hi: np.ndarray, eps: float) -> np.ndarray:
    """True for boxes (grown by ``eps``) that a triangle of ``mesh`` touches."""
    hit = np.zeros(len(lo), dtype=bool)
    if mesh.n_faces == 0 or len(lo) == 0:
        return hit
    lo = np.ascontiguousarray(lo - eps)
    hi = np.ascontiguousarray(hi + eps)
    box_index, tri_index = mesh.bvh.query_boxes(lo, hi)
    if len(box_index):
        ok = _tri_box_pairs(mesh.vertices, mesh.faces, lo, hi, box_index, tri_index)
        hit[box_index[ok]] = True
    return hit


@njit(cache=True, parallel=True)
def _points_in_polygon(u, v, pu, pv):
    """Even-odd rule; an edge owns its lower end (half-open), so shared edges count once."""
    n = u.shape[0]
    m = pu.shape[0]
    out = np.zeros(n, dtype=np.bool_)
    for i in prange(n):
        x = u[i]
        y = v[i]
        inside = False
        j = m - 1
        for k in range(m):
            if (pv[k] > y) != (pv[j] > y):
                xc = pu[k] + (y - pv[k]) * (pu[j] - pu[k]) / (pv[j] - pv[k])
                if x < xc:
                    inside = not inside
            j = k
        out[i] = inside
    return out


def _rounding_margin(*arrays: Any) -> float:
    """64 ulp of the largest coordinate magnitude involved."""
    scale = 1.0
    for a in arrays:
        a = np.asarray(a, dtype=np.float64)
        if a.size:
            scale = max(scale, float(np.abs(a).max()))
    return 64.0 * float(np.spacing(scale))


def _interval_status(
    lo_box: np.ndarray, hi_box: np.ndarray, lo: float, hi: float, eps: float
) -> np.ndarray:
    """Box extent ``[lo_box, hi_box)`` against the half-open range ``[lo, hi)``."""
    status = np.zeros(len(lo_box), dtype=np.int8)
    status[(lo_box >= lo + eps) & (hi_box <= hi - eps)] = INSIDE
    status[(hi_box <= lo - eps) | (lo_box >= hi + eps)] = OUTSIDE
    return status


def _status_and(parts: list[np.ndarray]) -> np.ndarray:
    stack = np.stack(parts)
    out = np.zeros(stack.shape[1], dtype=np.int8)
    out[(stack == INSIDE).all(axis=0)] = INSIDE
    out[(stack == OUTSIDE).any(axis=0)] = OUTSIDE
    return out


def _boxes(lo: Any, hi: Any) -> tuple[np.ndarray, np.ndarray]:
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    if lo.ndim != 2 or lo.shape[1] != 3 or lo.shape != hi.shape:
        raise ValueError("lo and hi must both have shape (n, 3)")
    if not (hi > lo).all():
        raise ValueError("boxes must have hi > lo on every axis")
    return lo, hi


_LOCAL_MESHES: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _mesh_in_frame(mesh: Mesh, shift: tuple, rotation: np.ndarray | None) -> Mesh:
    """``(vertices - shift) @ rotation`` as a Mesh, cached per source mesh and frame."""
    per_mesh = _LOCAL_MESHES.setdefault(mesh, {})
    key = (shift, None if rotation is None else rotation.tobytes())
    if key not in per_mesh:
        vertices = mesh.vertices - np.array(shift)
        if rotation is not None:
            vertices = np.ascontiguousarray(vertices @ rotation)
        if len(per_mesh) >= 4:  # a few frames per mesh are enough
            per_mesh.clear()
        per_mesh[key] = Mesh(vertices, mesh.faces, check=False)
    return per_mesh[key]


def _model_frame_mesh(mesh: Mesh, definition: BlockModelDefinition) -> Mesh:
    rotation = definition.rotation if definition.is_rotated else None
    return _mesh_in_frame(mesh, definition.origin, rotation)


def _side(value: str, allowed: tuple[str, str]) -> str:
    if value not in allowed:
        raise ConstraintError(f"side must be {allowed[0]!r} or {allowed[1]!r}, got {value!r}")
    return value


def _vector(value: Any, name: str) -> np.ndarray:
    v = np.asarray(value, dtype=np.float64).reshape(-1)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ConstraintError(f"{name} must be three finite numbers")
    return v


# --------------------------------------------------------------------------
# base class and combinations
# --------------------------------------------------------------------------

_TYPES: dict[str, type[Constraint]] = {}


class Constraint:
    """Base class of all constraints; see the module docstring."""

    type_name: ClassVar[str] = ""

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        if cls.type_name:
            _TYPES[cls.type_name] = cls

    # -- evaluation
    def mask(self, model: BlockModel, resolver: GeometryResolver | None = None) -> np.ndarray:
        """(n_blocks,) bool: True where the block centroid satisfies the constraint."""
        raise NotImplementedError

    def classify_boxes(
        self,
        definition: BlockModelDefinition,
        lo: Any,
        hi: Any,
        resolver: GeometryResolver | None = None,
    ) -> np.ndarray:
        """(n,) int8 of :data:`INSIDE` / :data:`OUTSIDE` / :data:`STRADDLE` for model-frame boxes.

        Conservative; see the module docstring. Raises :class:`ConstraintError`
        for constraints that are not geometric.
        """
        raise ConstraintError(
            f"a {self.type_name} constraint is not geometric: boxes cannot be classified"
        )

    @property
    def is_geometric(self) -> bool:
        """True when :meth:`classify_boxes` is available."""
        return True

    # -- combination
    def __and__(self, other: Constraint) -> Constraint:
        return And(self, other)

    def __or__(self, other: Constraint) -> Constraint:
        return Or(self, other)

    def __invert__(self) -> Constraint:
        return Not(self)

    # -- serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; :func:`constraint_from_dict` is the inverse."""
        raise NotImplementedError

    @classmethod
    def _from_dict(cls, data: dict[str, Any]) -> Constraint:
        raise NotImplementedError

    def __repr__(self) -> str:
        try:
            body = {k: v for k, v in self.to_dict().items() if k != "type"}
        except ConstraintError:
            body = "..."
        return f"{type(self).__name__}({body})"


def _check_children(children: Iterable[Any]) -> tuple[Constraint, ...]:
    out = tuple(children)
    if not out or not all(isinstance(c, Constraint) for c in out):
        raise ConstraintError("AND / OR need one or more constraints")
    return out


class And(Constraint):
    """All of the children."""

    type_name = "and"

    def __init__(self, *children: Constraint):
        self.children = _check_children(children)

    def mask(self, model, resolver=None):
        out = self.children[0].mask(model, resolver)
        for child in self.children[1:]:
            out = out & child.mask(model, resolver)
        return out

    def classify_boxes(self, definition, lo, hi, resolver=None):
        return _status_and([c.classify_boxes(definition, lo, hi, resolver) for c in self.children])

    @property
    def is_geometric(self):
        return all(c.is_geometric for c in self.children)

    def to_dict(self):
        return {"type": self.type_name, "children": [c.to_dict() for c in self.children]}

    @classmethod
    def _from_dict(cls, data):
        return cls(*(constraint_from_dict(c) for c in data["children"]))


class Or(And):
    """Any of the children."""

    type_name = "or"

    def mask(self, model, resolver=None):
        out = self.children[0].mask(model, resolver)
        for child in self.children[1:]:
            out = out | child.mask(model, resolver)
        return out

    def classify_boxes(self, definition, lo, hi, resolver=None):
        parts = [-c.classify_boxes(definition, lo, hi, resolver) for c in self.children]
        return -_status_and(parts)


class Not(Constraint):
    """The complement of the child."""

    type_name = "not"

    def __init__(self, child: Constraint):
        if not isinstance(child, Constraint):
            raise ConstraintError("NOT needs a constraint")
        self.child = child

    def mask(self, model, resolver=None):
        return ~self.child.mask(model, resolver)

    def classify_boxes(self, definition, lo, hi, resolver=None):
        return -self.child.classify_boxes(definition, lo, hi, resolver)

    @property
    def is_geometric(self):
        return self.child.is_geometric

    def to_dict(self):
        return {"type": self.type_name, "child": self.child.to_dict()}

    @classmethod
    def _from_dict(cls, data):
        return cls(constraint_from_dict(data["child"]))


class CombinedConstraint(Constraint):
    """Named constraints joined by a text expression; create with :func:`combine`.

    Attributes
    ----------
    expression : the text, e.g. ``"a and (b or not c)"``.
    constraints : name -> constraint (only the names the expression uses).
    tree : the equivalent :class:`And` / :class:`Or` / :class:`Not` tree.
    """

    type_name = "expression"

    def __init__(self, expression: str, constraints: Mapping[str, Constraint]):
        try:
            parsed = _expressions.parse(expression)
        except _expressions.ExpressionError as exc:
            raise ConstraintError(str(exc)) from None
        missing = sorted(parsed.names - set(constraints))
        if missing:
            raise ConstraintError(
                f"constraint expression {expression!r} uses undefined name(s): "
                f"{', '.join(missing)}"
            )
        self.expression = parsed.source
        self.constraints = {name: constraints[name] for name in sorted(parsed.names)}
        if not all(isinstance(c, Constraint) for c in self.constraints.values()):
            raise ConstraintError("every named item must be a Constraint")
        self.tree = self._build(parsed._tree.body)

    def _build(self, node: Any) -> Constraint:
        if isinstance(node, ast.Name):
            if node.id not in self.constraints:
                raise ConstraintError(f"{node.id!r} is not a constraint name")
            return self.constraints[node.id]
        if isinstance(node, ast.BoolOp):
            parts = [self._build(v) for v in node.values]
            return And(*parts) if isinstance(node.op, ast.And) else Or(*parts)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            return Not(self._build(node.operand))
        raise ConstraintError(
            "a constraint expression may only contain constraint names, and, or, not "
            "and parentheses"
        )

    def mask(self, model, resolver=None):
        return self.tree.mask(model, resolver)

    def classify_boxes(self, definition, lo, hi, resolver=None):
        return self.tree.classify_boxes(definition, lo, hi, resolver)

    @property
    def is_geometric(self):
        return self.tree.is_geometric

    def to_dict(self):
        return {
            "type": self.type_name,
            "expression": self.expression,
            "constraints": {name: c.to_dict() for name, c in self.constraints.items()},
        }

    @classmethod
    def _from_dict(cls, data):
        named = {name: constraint_from_dict(c) for name, c in data["constraints"].items()}
        return cls(data["expression"], named)


def combine(expression: str, constraints: Mapping[str, Constraint]) -> CombinedConstraint:
    """Combine named constraints with ``and`` / ``or`` / ``not`` and parentheses.

    ``combine("a and (b or not c)", {"a": ..., "b": ..., "c": ...})``. The
    keywords may be written in any letter case. Names follow the rules of
    attribute names. The text is parsed by the safe expression parser
    (:mod:`bakumine.core.blockmodel.expressions`), never evaluated as code.
    """
    return CombinedConstraint(expression, constraints)


# --------------------------------------------------------------------------
# primitives
# --------------------------------------------------------------------------


class DtmConstraint(Constraint):
    """Blocks whose centroid is above / below a surface.

    Parameters
    ----------
    surface : name of a DTM (looked up through the resolver), or a
        :class:`~bakumine.core.geometry.dtm.DTM` object (then the constraint
        cannot be saved).
    side : ``"above"`` (centroid Z > surface Z) or ``"below"`` (centroid Z <=
        surface Z).
    outside : what to answer for centroids outside the plan extent of the
        surface, whatever the side: ``False`` (default: not selected),
        ``True`` (selected) or ``"error"`` (raise :class:`ConstraintError`).
    """

    type_name = "dtm"

    def __init__(self, surface: str | DTM, side: str = "above", outside: bool | str = False):
        if not isinstance(surface, str | DTM):
            raise ConstraintError("surface must be a DTM or the name of one")
        if outside not in (True, False, "error"):
            raise ConstraintError("outside must be True, False or 'error'")
        self.surface = surface
        self.side = _side(side, ("above", "below"))
        self.outside = outside

    def _pick(self, height: np.ndarray) -> np.ndarray:
        covered = ~np.isnan(height)
        if self.outside == "error" and not covered.all():
            raise ConstraintError(
                f"{int((~covered).sum())} point(s) lie outside the plan extent of the DTM "
                f"{self.surface if isinstance(self.surface, str) else ''}".rstrip()
            )
        with np.errstate(invalid="ignore"):
            selected = height > 0.0 if self.side == "above" else height <= 0.0
        return np.where(covered, selected, bool(self.outside) and self.outside != "error")

    def mask(self, model, resolver=None):
        dtm = _resolve(self.surface, "dtm", resolver)
        return self._pick(dtm.height_above(model.centroids()))

    def classify_boxes(self, definition, lo, hi, resolver=None):
        dtm = _resolve(self.surface, "dtm", resolver)
        lo, hi = _boxes(lo, hi)
        corners = definition.box_corners(lo, hi)
        eps = _rounding_margin(dtm.mesh.vertices, corners)
        straddle = _boxes_hit_mesh(_model_frame_mesh(dtm.mesh, definition), lo, hi, eps)
        # the plan outline of the surface, as degenerate triangles at z = 0, against the
        # world bounding rectangle of each box
        edge_map = dtm.mesh.edge_map
        border = edge_map.edges[edge_map.boundary_edges]
        if len(border):
            flat = dtm.mesh.vertices.copy()
            flat[:, 2] = 0.0
            outline = Mesh(flat, np.column_stack([border, border[:, 1]]), check=False)
            rect_lo = corners.min(axis=1)
            rect_hi = corners.max(axis=1)
            rect_lo[:, 2], rect_hi[:, 2] = -1.0, 1.0
            crosses_outline = _boxes_hit_mesh(outline, rect_lo, rect_hi, eps)
        else:
            crosses_outline = np.zeros(len(lo), dtype=bool)
        if self.outside == "error" and crosses_outline.any():
            raise ConstraintError(
                f"{int(crosses_outline.sum())} box(es) cross the plan outline of the DTM"
            )
        centre = definition.local_to_world(0.5 * (lo + hi))
        status = np.where(self._pick(dtm.height_above(centre)), INSIDE, OUTSIDE).astype(np.int8)
        status[straddle | crosses_outline] = STRADDLE
        return status

    def to_dict(self):
        return {
            "type": self.type_name,
            "surface": _reference(self.surface, "DTM"),
            "side": self.side,
            "outside": self.outside,
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(data["surface"], data.get("side", "above"), data.get("outside", False))


class SolidConstraint(Constraint):
    """Blocks whose centroid is inside / outside a closed mesh.

    Parameters
    ----------
    solid : name of a solid (looked up through the resolver), or a
        :class:`~bakumine.core.geometry.mesh.Mesh` (then the constraint
        cannot be saved).
    side : ``"inside"`` or ``"outside"`` (= not inside).

    The test is :func:`bakumine.core.geometry.inside.points_in_solid` (rule
    on the surface: see that module). Evaluation:

    * unrotated model: world centroids against the mesh. When the model has
      no sub-blocks the regular-grid routine ``grid_in_solid`` is used
      instead (one ray per column); it is only used when its grid
      coordinates are bit-for-bit the centroids, so both paths give
      identical masks.
    * rotated model: the mesh is rotated into the model frame once (about
      the centroid of block (0, 0, 0)) and the centroids are tested there;
      again the grid routine and the point routine see identical
      coordinates and give identical masks. Relative to testing world
      centroids against the unrotated mesh this can differ only for
      centroids within rounding distance (~1e-9 m) of the surface.

    An open mesh raises ``ValueError``.
    """

    type_name = "solid"

    def __init__(self, solid: str | Mesh, side: str = "inside"):
        if not isinstance(solid, str | Mesh):
            raise ConstraintError("solid must be a Mesh or the name of one")
        self.solid = solid
        self.side = _side(side, ("inside", "outside"))

    def mask(self, model, resolver=None, *, dense: bool | None = None):
        """As :meth:`Constraint.mask`.

        ``dense``: ``None`` (default) uses the regular-grid fast path when
        possible; ``False`` forces the per-point path (same result).
        """
        mesh = _resolve(self.solid, "solid", resolver)
        d = model.definition
        first = d.ijk_to_world((0, 0, 0))
        use_grid = model.is_dense and dense is not False
        if not d.is_rotated:
            if use_grid:
                for a in range(3):
                    steps = np.arange(d.shape[a])
                    grid_axis = first[a] + steps * d.block_size[a]
                    model_axis = d.origin[a] + (steps + 0.5) * d.block_size[a]
                    use_grid = use_grid and np.array_equal(grid_axis, model_axis)
            if use_grid:
                inside = grid_in_solid(mesh, first, d.block_size, d.shape).reshape(-1)
            else:
                inside = points_in_solid(mesh, model.centroids())
        elif use_grid:
            inside = grid_in_solid(mesh, first, d.block_size, d.shape, d.rotation).reshape(-1)
        else:
            local = _mesh_in_frame(mesh, tuple(first), d.rotation)
            inside = points_in_solid(local, model.centroids_grid())
        return inside if self.side == "inside" else ~inside

    def classify_boxes(self, definition, lo, hi, resolver=None):
        mesh = _resolve(self.solid, "solid", resolver)
        lo, hi = _boxes(lo, hi)
        local = _model_frame_mesh(mesh, definition)
        eps = _rounding_margin(mesh.vertices, definition.bounding_box())
        straddle = _boxes_hit_mesh(local, lo, hi, eps)
        inside = points_in_solid(local, 0.5 * (lo + hi))
        status = np.where(inside, INSIDE, OUTSIDE).astype(np.int8)
        if self.side == "outside":
            status = -status
        status[straddle] = STRADDLE
        return status

    def to_dict(self):
        return {
            "type": self.type_name,
            "solid": _reference(self.solid, "solid"),
            "side": self.side,
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(data["solid"], data.get("side", "inside"))


class AttributeConstraint(Constraint):
    """Blocks for which an expression over attributes is true.

    ``AttributeConstraint('au > 0.5 and rock == "ORE"')``. The language is
    that of :mod:`bakumine.core.blockmodel.expressions` (safe, no ``eval``);
    the built-in variables ``x y z dx dy dz volume`` are available. A numeric
    result selects blocks where it is non-zero and not NaN. Not geometric.
    """

    type_name = "attribute"

    def __init__(self, expression: str):
        try:
            self._expression = _expressions.parse(expression)
        except _expressions.ExpressionError as exc:
            raise ConstraintError(str(exc)) from None
        self.expression = self._expression.source

    def mask(self, model, resolver=None):
        values, _ = model.evaluate(self._expression, return_invalid=True)
        if values.dtype == bool:
            return values
        return (values != 0.0) & ~np.isnan(values)

    @property
    def is_geometric(self):
        return False

    def to_dict(self):
        return {"type": self.type_name, "expression": self.expression}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["expression"])


class PlaneConstraint(Constraint):
    """Blocks whose centroid is above / below a plane.

    Parameters
    ----------
    point : a world point on the plane.
    normal : the plane normal (any length). ``"above"`` is the side the
        normal points to: ``(centroid - point) . normal > 0``; ``"below"`` is
        the rest (``<= 0``, including the plane itself). For a vertical
        plane "above" therefore means "in front of", towards the normal.
    side : ``"above"`` or ``"below"``.

    Also :meth:`from_points` and :meth:`from_strike_dip`.
    """

    type_name = "plane"

    def __init__(self, point: Any, normal: Any, side: str = "above"):
        self.point = _vector(point, "point")
        normal = _vector(normal, "normal")
        length = float(np.linalg.norm(normal))
        if not length > 0.0:
            raise ConstraintError("the plane normal must not be zero")
        self.normal = normal / length
        self.side = _side(side, ("above", "below"))

    @classmethod
    def from_points(cls, p1: Any, p2: Any, p3: Any, side: str = "above") -> PlaneConstraint:
        """Plane through three points.

        The normal is turned to point upwards (positive Z), so "above" has
        its everyday meaning. For a vertical plane the normal is
        ``(p2 - p1) x (p3 - p1)`` as given.
        """
        p1, p2, p3 = (_vector(p, "point") for p in (p1, p2, p3))
        normal = np.cross(p2 - p1, p3 - p1)
        scale = float(np.linalg.norm(p2 - p1) * np.linalg.norm(p3 - p1))
        if not np.linalg.norm(normal) > 1e-12 * max(scale, 1e-300):
            raise ConstraintError("the three points are collinear and do not define a plane")
        if normal[2] < 0.0:
            normal = -normal
        return cls(p1, normal, side)

    @classmethod
    def from_strike_dip(
        cls, point: Any, strike: float, dip: float, side: str = "above"
    ) -> PlaneConstraint:
        """Plane from strike and dip (degrees), right-hand rule.

        ``strike`` is the azimuth of the strike line, clockwise from north;
        the plane dips ``dip`` degrees (0 = horizontal, 90 = vertical) towards
        azimuth ``strike + 90``. The normal is the upward one,
        ``(sin(dd) sin(dip), cos(dd) sin(dip), cos(dip))`` with ``dd = strike
        + 90``, so "above" is the hanging-wall side.
        """
        if not (np.isfinite(strike) and 0.0 <= dip <= 90.0):
            raise ConstraintError("strike must be finite and dip within 0..90 degrees")
        dd = np.deg2rad(strike + 90.0)
        delta = np.deg2rad(dip)
        normal = (np.sin(dd) * np.sin(delta), np.cos(dd) * np.sin(delta), np.cos(delta))
        return cls(point, normal, side)

    def _distance(self, points: np.ndarray) -> np.ndarray:
        n, p = self.normal, self.point
        return (
            (points[..., 0] - p[0]) * n[0]
            + (points[..., 1] - p[1]) * n[1]
            + (points[..., 2] - p[2]) * n[2]
        )

    def mask(self, model, resolver=None):
        above = self._distance(model.centroids()) > 0.0
        return above if self.side == "above" else ~above

    def classify_boxes(self, definition, lo, hi, resolver=None):
        lo, hi = _boxes(lo, hi)
        corners = definition.box_corners(lo, hi)
        s = self._distance(corners)
        eps = _rounding_margin(corners, self.point)
        # exact zeros (a plane on box faces) are trusted; other tiny distances are not
        s_min, s_max = s.min(axis=1), s.max(axis=1)
        unsure = ((np.abs(s) < eps) & (s != 0.0)).any(axis=1)
        status = np.zeros(len(lo), dtype=np.int8)
        status[(s_min >= 0.0) & ~unsure] = INSIDE
        status[(s_max <= 0.0) & ~unsure] = OUTSIDE
        return status if self.side == "above" else -status

    def to_dict(self):
        return {
            "type": self.type_name,
            "point": self.point.tolist(),
            "normal": self.normal.tolist(),
            "side": self.side,
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(data["point"], data["normal"], data.get("side", "above"))


_AXIS_FRAMES = {
    # projection axis -> (index of u, index of v, index of w) in world coordinates
    "x": (1, 2, 0),
    "y": (0, 2, 1),
    "z": (0, 1, 2),
}


class PolygonConstraint(Constraint):
    """Blocks whose centroid is inside / outside a closed polygon projected along an axis.

    Parameters
    ----------
    polygon : (n, 3) world points of a closed string (the closing segment is
        implied; a repeated first point is dropped), or the name of one
        (looked up through the resolver). With ``axis`` ``"x"``, ``"y"`` or
        ``"z"`` an (n, 2) array is also accepted: the coordinates in the
        projection plane, i.e. (y, z), (x, z) or (x, y).
    side : ``"inside"`` or ``"outside"`` (= not inside, limits included).
    axis : direction of projection: ``"z"`` (default; a plan polygon),
        ``"x"``, ``"y"``, or any world direction vector.
    limits : optional ``(low, high)`` along the projection axis: the
        centroid's coordinate ``w`` along the axis must satisfy ``low <= w <
        high``. For ``"x"``/``"y"``/``"z"`` ``w`` is the world coordinate
        itself (e.g. an elevation range). For a direction vector ``w`` is the
        distance along the direction **from the first polygon point**,
        ``(centroid - polygon[0]) . unit(direction)``, so ``(-25, 25)``
        selects a 50 m slab centred on a section string. ``None`` on either
        end leaves it open.

    Even-odd rule, so a self-intersecting string selects its odd regions. A
    point exactly on the outline is inside or outside by the half-open rule
    of the crossing test (lower / left edges in, upper / right edges out for
    an axis-aligned rectangle).
    """

    type_name = "polygon"

    def __init__(
        self,
        polygon: Any,
        side: str = "inside",
        axis: Any = "z",
        limits: Any = None,
    ):
        self.side = _side(side, ("inside", "outside"))
        if isinstance(axis, str):
            if axis not in _AXIS_FRAMES:
                raise ConstraintError("axis must be 'x', 'y', 'z' or a direction vector")
            self.axis: Any = axis
        else:
            direction = _vector(axis, "axis")
            if not np.linalg.norm(direction) > 0.0:
                raise ConstraintError("the projection direction must not be zero")
            self.axis = tuple(float(c) for c in direction)
        if limits is None:
            limits = (None, None)
        try:
            low, high = limits
        except (TypeError, ValueError):
            raise ConstraintError("limits must be (low, high)") from None
        self.limits = (
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
        if not self.limits[0] < self.limits[1]:
            raise ConstraintError("limits must satisfy low < high")
        if isinstance(polygon, str):
            self.polygon: Any = polygon
        else:
            self.polygon = self._check_polygon(polygon)

    def _check_polygon(self, polygon: Any) -> np.ndarray:
        pts = np.array(polygon, dtype=np.float64)
        columns = (2, 3) if isinstance(self.axis, str) else (3,)
        if pts.ndim != 2 or pts.shape[1] not in columns or not np.isfinite(pts).all():
            raise ConstraintError(
                f"a polygon must be a finite (n, {' or '.join(map(str, columns))}) array"
            )
        if len(pts) > 1 and np.array_equal(pts[0], pts[-1]):
            pts = pts[:-1]
        if len(pts) < 3:
            raise ConstraintError("a polygon needs at least three distinct points")
        return pts

    def _setup(self, resolver):
        """``(project, pu, pv)``: world points (..., 3) -> (u, v, w), and the outline in (u, v)."""
        polygon = self.polygon
        if isinstance(polygon, str):
            polygon = self._check_polygon(_resolve(polygon, "polygon", resolver))
        if isinstance(self.axis, str):
            iu, iv, iw = _AXIS_FRAMES[self.axis]

            def project(p):
                return p[..., iu], p[..., iv], p[..., iw]

            if polygon.shape[1] == 2:
                pu, pv = polygon[:, 0], polygon[:, 1]
            else:
                pu, pv = polygon[:, iu], polygon[:, iv]
        else:
            frame = plane_frame(self.axis)
            base = polygon[0]

            def project(p):
                # relative to the first polygon point: small numbers, and w = 0 on the string
                return tuple(
                    (p[..., 0] - base[0]) * a[0]
                    + (p[..., 1] - base[1]) * a[1]
                    + (p[..., 2] - base[2]) * a[2]
                    for a in frame
                )

            pu, pv, _ = project(polygon)
        return project, np.ascontiguousarray(pu), np.ascontiguousarray(pv)

    def mask(self, model, resolver=None):
        project, pu, pv = self._setup(resolver)
        u, v, w = project(model.centroids())
        inside = _points_in_polygon(np.ascontiguousarray(u), np.ascontiguousarray(v), pu, pv)
        inside &= (w >= self.limits[0]) & (w < self.limits[1])
        return inside if self.side == "inside" else ~inside

    def classify_boxes(self, definition, lo, hi, resolver=None):
        project, pu, pv = self._setup(resolver)
        lo, hi = _boxes(lo, hi)
        corners = definition.box_corners(lo, hi)
        u, v, w = project(corners)
        exact = isinstance(self.axis, str) and not definition.is_rotated
        eps = _rounding_margin(corners, pu, pv)
        # polygon edges as degenerate triangles in the plane z = 0 against the (u, v) rectangles
        n = len(pu)
        vertices = np.column_stack([pu, pv, np.zeros(n)])
        nxt = (np.arange(n) + 1) % n
        outline = Mesh(vertices, np.column_stack([np.arange(n), nxt, nxt]), check=False)
        rect_lo = np.column_stack([u.min(axis=1), v.min(axis=1), np.full(len(lo), -1.0)])
        rect_hi = np.column_stack([u.max(axis=1), v.max(axis=1), np.full(len(lo), 1.0)])
        crossed = _boxes_hit_mesh(outline, rect_lo, rect_hi, eps)
        centre_u = np.ascontiguousarray(0.5 * (rect_lo[:, 0] + rect_hi[:, 0]))
        centre_v = np.ascontiguousarray(0.5 * (rect_lo[:, 1] + rect_hi[:, 1]))
        in_plan = np.where(_points_in_polygon(centre_u, centre_v, pu, pv), INSIDE, OUTSIDE)
        in_plan = in_plan.astype(np.int8)
        in_plan[crossed] = STRADDLE
        along = _interval_status(
            w.min(axis=1), w.max(axis=1), *self.limits, 0.0 if exact else eps
        )
        status = _status_and([in_plan, along])
        return status if self.side == "inside" else -status

    def to_dict(self):
        low, high = self.limits
        polygon = self.polygon if isinstance(self.polygon, str) else self.polygon.tolist()
        return {
            "type": self.type_name,
            "polygon": polygon,
            "side": self.side,
            "axis": self.axis if isinstance(self.axis, str) else list(self.axis),
            "limits": [None if np.isinf(low) else low, None if np.isinf(high) else high],
        }

    @classmethod
    def _from_dict(cls, data):
        return cls(
            data["polygon"], data.get("side", "inside"), data.get("axis", "z"), data.get("limits")
        )


class RangeConstraint(Constraint):
    """Blocks inside coordinate and / or index ranges (all given ranges must hold).

    Parameters
    ----------
    x, y, z : ``(min, max)`` on the world centroid coordinate, half-open:
        ``min <= c < max``. Use ``z`` for a bench or elevation range;
        adjacent benches ``(900, 905)``, ``(905, 910)`` do not overlap.
    i, j, k : ``(min, max)`` on the parent block index, **inclusive** on both
        ends: ``min <= index <= max`` (sub-blocks have their parent's index).

    ``None`` for either end leaves it open; at least one range is needed.
    """

    type_name = "range"
    _KEYS = ("x", "y", "z", "i", "j", "k")

    def __init__(self, *, x=None, y=None, z=None, i=None, j=None, k=None):
        self.ranges: dict[str, tuple[float, float]] = {}
        for key, value in zip(self._KEYS, (x, y, z, i, j, k), strict=True):
            if value is None:
                continue
            try:
                low, high = value
            except (TypeError, ValueError):
                raise ConstraintError(f"range {key} must be (min, max)") from None
            low = -np.inf if low is None else float(low)
            high = np.inf if high is None else float(high)
            if np.isnan(low) or np.isnan(high) or low > high:
                raise ConstraintError(f"range {key}: need min <= max")
            if key in "ijk":
                for end in (low, high):
                    if np.isfinite(end) and end != int(end):
                        raise ConstraintError(f"range {key}: block indices must be integers")
            self.ranges[key] = (low, high)
        if not self.ranges:
            raise ConstraintError("a range constraint needs at least one range")

    @classmethod
    def elevation(cls, low: float | None, high: float | None) -> RangeConstraint:
        """Bench / elevation range ``low <= z < high``."""
        return cls(z=(low, high))

    def mask(self, model, resolver=None):
        out = np.ones(model.n_blocks, dtype=bool)
        centroid = ijk = None
        for key, (low, high) in self.ranges.items():
            if key in "xyz":
                if centroid is None:
                    centroid = model.centroids()
                c = centroid[:, "xyz".index(key)]
                out &= (c >= low) & (c < high)
            else:
                if ijk is None:
                    ijk = model.ijk()
                index = ijk[:, "ijk".index(key)]
                out &= (index >= low) & (index <= high)
        return out

    def classify_boxes(self, definition, lo, hi, resolver=None):
        lo, hi = _boxes(lo, hi)
        parts = []
        corners = None
        for key, (low, high) in self.ranges.items():
            if key in "xyz":
                if corners is None:
                    corners = definition.box_corners(lo, hi)
                c = corners[..., "xyz".index(key)]
                eps = _rounding_margin(corners) if definition.is_rotated else 0.0
                parts.append(_interval_status(c.min(axis=1), c.max(axis=1), low, high, eps))
            else:
                a = "ijk".index(key)
                size = definition.block_size[a]
                # index range [low, high] = local coordinates [low * size, (high + 1) * size)
                parts.append(
                    _interval_status(lo[:, a], hi[:, a], low * size, (high + 1.0) * size, 0.0)
                )
        return _status_and(parts)

    def to_dict(self):
        out: dict[str, Any] = {"type": self.type_name}
        for key, (low, high) in self.ranges.items():
            ends = [None if np.isinf(low) else low, None if np.isinf(high) else high]
            out[key] = [e if e is None or key in "xyz" else int(e) for e in ends]
        return out

    @classmethod
    def _from_dict(cls, data):
        return cls(**{key: tuple(data[key]) for key in cls._KEYS if data.get(key) is not None})


class BlockIdConstraint(Constraint):
    """An explicit set of blocks, by id (:meth:`BlockModel.block_ids`).

    Ids, unlike row numbers, stay valid when other blocks are split or
    merged. An id whose block no longer exists as a leaf selects nothing (its
    children do not inherit the selection). Not geometric.
    """

    type_name = "block_ids"

    def __init__(self, ids: Any):
        arr = np.asarray(ids)
        if arr.size and arr.dtype.kind not in "iu":
            raise ConstraintError("block ids must be integers")
        self.ids = np.unique(arr.astype(np.int64).reshape(-1))

    @classmethod
    def from_rows(cls, model: BlockModel, where: Any) -> BlockIdConstraint:
        """The blocks currently at the given rows (bool mask or row numbers)."""
        return cls(model.block_ids(where))

    def mask(self, model, resolver=None):
        return np.isin(model.block_ids(), self.ids)

    @property
    def is_geometric(self):
        return False

    def to_dict(self):
        return {"type": self.type_name, "ids": self.ids.tolist()}

    @classmethod
    def _from_dict(cls, data):
        return cls(data["ids"])


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def constraint_from_dict(data: Mapping[str, Any]) -> Constraint:
    """Rebuild a constraint from :meth:`Constraint.to_dict` output."""
    if not isinstance(data, Mapping) or "type" not in data:
        raise ConstraintError("a constraint must be an object with a 'type'")
    try:
        cls = _TYPES[data["type"]]
    except (KeyError, TypeError):
        raise ConstraintError(
            f"unknown constraint type {data['type']!r}; known: {', '.join(sorted(_TYPES))}"
        ) from None
    try:
        return cls._from_dict(dict(data))
    except KeyError as exc:
        raise ConstraintError(f"{data['type']} constraint: missing field {exc}") from None


def save_constraint(constraint: Constraint, path: str | Path) -> None:
    """Write a constraint as JSON. Geometry held as objects (not names) raises."""
    document = {"bakumine_constraint": FORMAT_VERSION, "constraint": constraint.to_dict()}
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_constraint(path: str | Path) -> Constraint:
    """Read a constraint written by :func:`save_constraint`."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "constraint" not in document:
        raise ConstraintError(f"{path} is not a BakuMine constraint file")
    if document.get("bakumine_constraint") != FORMAT_VERSION:
        raise ConstraintError(
            f"{path}: unsupported constraint format {document.get('bakumine_constraint')!r}"
        )
    return constraint_from_dict(document["constraint"])
