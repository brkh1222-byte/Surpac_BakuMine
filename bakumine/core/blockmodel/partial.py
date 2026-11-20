"""Partial percentages: the fraction of each block inside a constraint (research.md 6.5).

Sub-blocking resolves a boundary by making blocks smaller. The alternative
(or supplement) is to keep the blocks and store, per block, **which fraction
of its volume lies inside** a solid, below a surface, ...; volumes are then
reported as ``sum(fraction * block volume)``.

:func:`partial_fraction` computes that fraction for a geometric constraint or
a combination of geometric constraints. Blocks the boundary does not cut get
exactly 1.0 or 0.0 from the box classification
(:meth:`Constraint.classify_boxes`, with the face tolerance of
:mod:`bakumine.core.blockmodel.subblocking`) and are never sampled. For the
blocks that are cut there are two methods.

``method="lattice"`` (default) -- any geometric constraint
    The block is filled with ``nx * ny * nz`` equal cells (``samples``) and
    the constraint is tested at each cell centre, with exactly the code that
    tests block centroids (so an ``n = 2**k`` lattice gives precisely the
    volume fraction of the level-``k`` sub-blocks that centroid coding would
    select). The fraction is the share of centres that pass: a multiple of
    ``1 / (nx ny nz)``.

    *Error.* Per block, a smooth boundary is misjudged by a fraction of the
    layer of cells it passes through: the error is bounded by ``~ 1 / n`` of
    the block and is typically ``~ 1 / n**2``. Measured on the 10 x 10 x 5 m
    blocks cut by the synthetic ore solid, against a 40 x 40 column
    reference, mean (maximum) absolute error in block fractions: 0.033
    (0.21) for ``n = 2``, 0.0077 (0.050) for ``n = 4``, 0.0048 (0.043) for
    ``n = 5``, 0.0018 (0.022) for ``n = 8``; on the blocks cut by the
    topography 0.0093 (0.078) for ``n = 5``. The errors of different blocks
    have both signs and largely cancel in a total, so a total volume is far
    more accurate than a single block: the ore volume is within 0.012 % of
    the mesh volume for every ``n >= 2``, a sphere of r = 60 m within 0.07 %.
    Boundaries parallel to the lattice (a flat bench floor, a box) do not
    average out: the fraction is then simply rounded to the nearest cell
    layer, a systematic error of up to ``1 / (2 n)`` of the block along that
    axis. Use the column method, or sub-blocking, for those.

``method="column"`` -- solids, DTMs, planes, coordinate ranges, combinations
    Only the plan section of the block (model ``i``, ``j``) is sampled, on an
    ``nx * ny`` lattice; along the model ``k`` axis the integration is
    **exact**. Through each lattice point runs a line parallel to ``k``; the
    points where it pierces each surface are computed (ray / triangle
    crossings in the model frame, so a rotated model is handled the same
    way; a plane or range limit analytically), the line is cut at those
    points, each piece is classified by its midpoint, and the lengths of the
    pieces inside are added, clipped to the block. The fraction of the block
    is the mean over its lines.

    *Error.* None along ``k``. In plan it is a midpoint rule: ``~ 1 / n**2``
    where the inside length varies smoothly across the block, ``~ 1 / n``
    where it jumps (a vertical wall, the silhouette of a solid as seen along
    ``k``), again with cancelling signs between blocks. A flat horizontal
    boundary is exact with ``n = 1``. Measured as above, mean (maximum)
    absolute error: ore solid 0.020 (0.15) for ``n = 2``, 0.0044 (0.046) for
    ``n = 4``, 0.0027 (0.024) for ``n = 5``, 0.0010 (0.014) for ``n = 8``;
    topography 0.0003 (0.0016) for ``n = 5`` -- thirty times better than the
    lattice, because a near-horizontal surface is exactly the case the
    lattice handles worst. Totals: ore volume within 0.003 % of the mesh
    volume for ``n >= 4``, the sphere within 0.05 % (0.001 % at ``n = 10``).
    Cost per block is ``nx * ny`` rays instead of ``nx * ny * nz`` point
    tests.

    Limitation: for a DTM in a model whose ``k`` axis is tilted (rotation
    with dip or plunge), a line can leave the plan extent of the surface
    without piercing it; such a line is not cut there and its last piece is
    classified as a whole. Lines inside the extent, and all lines of models
    rotated about Z only, are exact. Polygon constraints are not supported by
    this method (:class:`ConstraintError`); use the lattice.

Fractions belong to the blocks they were computed for: sub-blocks created
later inherit the stored value of their parent, which is then no longer
their own fraction. Compute partial percentages after any sub-blocking.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    INSIDE,
    STRADDLE,
    And,
    CombinedConstraint,
    Constraint,
    ConstraintError,
    DtmConstraint,
    GeometryResolver,
    Not,
    PlaneConstraint,
    RangeConstraint,
    SolidConstraint,
    _model_frame_mesh,
    _resolve,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.subblocking import (
    BOX_CHUNK,
    _classify_rows,
    _face_tolerance,
    _require_geometric,
    _row_numbers,
)
from bakumine.core.geometry.inside import OpenMeshWarning, column_crossings

__all__ = ["METHODS", "assign_partial", "constraint_volume", "partial_fraction"]

METHODS = ("lattice", "column")

#: sample points (lattice) or lines (column) evaluated at a time
POINT_CHUNK = 1 << 20


class _PointModel:
    """Arbitrary points presented to :meth:`Constraint.mask` as if they were block centroids.

    The geometric constraints only ask a model for its definition, its
    centroids (world, and grid-relative for rotated solids) and the parent
    index of each block; this object answers those for free points given in
    the model frame. It makes the sample tests of this module run through
    exactly the code that tests centroids.
    """

    is_dense = False

    def __init__(self, definition: BlockModelDefinition, local: np.ndarray, parent: np.ndarray):
        self.definition = definition
        self._local = local
        self._parent = parent
        self._world: np.ndarray | None = None

    @property
    def n_blocks(self) -> int:
        return len(self._local)

    def centroids_local(self, where: Any = None) -> np.ndarray:
        return self._local

    def centroids(self, where: Any = None) -> np.ndarray:
        if self._world is None:
            self._world = self.definition.local_to_world(self._local)
        return self._world

    def centroids_grid(self, where: Any = None) -> np.ndarray:
        # relative to the centroid of parent (0, 0, 0), as BlockModel.centroids_grid
        return self._local - 0.5 * np.array(self.definition.block_size)

    def ijk(self, where: Any = None) -> np.ndarray:
        return self.definition.unflat_index(self._parent)


def _samples(samples: Any, count: int) -> tuple[int, ...]:
    if isinstance(samples, int | np.integer) and not isinstance(samples, bool):
        out = (int(samples),) * count
    else:
        try:
            out = tuple(int(s) for s in samples)
            exact = all(int(s) == s for s in samples)
        except (TypeError, ValueError):
            raise ValueError("samples must be an integer or a tuple of integers") from None
        if not exact or len(out) not in (count, 3):
            raise ValueError(f"samples must be an integer or {count} integers, got {samples!r}")
        out = out[:count]
    if min(out) < 1:
        raise ValueError("samples must be >= 1")
    return out


def _ticks(n: int) -> np.ndarray:
    """Centres of ``n`` equal cells of the unit interval."""
    return (np.arange(n) + 0.5) / n


# --------------------------------------------------------------------------
# lattice
# --------------------------------------------------------------------------


def _lattice_fraction(
    model: BlockModel,
    constraint: Constraint,
    resolver: GeometryResolver | None,
    rows: np.ndarray,
    samples: tuple[int, ...],
) -> np.ndarray:
    lo, hi = model.bounds_local(rows)
    size = hi - lo
    grid = np.meshgrid(*(_ticks(n) for n in samples), indexing="ij")
    offsets = np.stack(grid, axis=-1).reshape(-1, 3)
    local = lo[:, None, :] + offsets[None, :, :] * size[:, None, :]
    points = _PointModel(
        model.definition, local.reshape(-1, 3), np.repeat(model.parent[rows], len(offsets))
    )
    inside = constraint.mask(points, resolver).reshape(len(rows), len(offsets))
    return inside.sum(axis=1) / float(len(offsets))


# --------------------------------------------------------------------------
# columns
# --------------------------------------------------------------------------


def _primitives(constraint: Constraint) -> list[Constraint]:
    """The primitive constraints of a tree, for the column method."""
    if isinstance(constraint, And):  # Or is a subclass of And
        return [p for child in constraint.children for p in _primitives(child)]
    if isinstance(constraint, Not):
        return _primitives(constraint.child)
    if isinstance(constraint, CombinedConstraint):
        return _primitives(constraint.tree)
    if isinstance(constraint, SolidConstraint | DtmConstraint | PlaneConstraint | RangeConstraint):
        return [constraint]
    raise ConstraintError(
        f"the column method does not support a {constraint.type_name} constraint; "
        'use method="lattice"'
    )


def _mesh_breaks(mesh: Any, definition: BlockModelDefinition, uv: np.ndarray, closed: bool):
    local = _model_frame_mesh(mesh, definition)
    if closed:
        crossings = column_crossings(local, uv)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", OpenMeshWarning)
            crossings = column_crossings(local, uv, strict=False)
    column = np.repeat(np.arange(len(uv)), np.diff(crossings.offsets))
    return column, crossings.z


def _linear_breaks(coefficients: np.ndarray, offset: float, uv: np.ndarray):
    """Where ``c0 u + c1 v + c2 t = offset`` along each line; nothing if ``c2 == 0``."""
    if coefficients[2] == 0.0:
        return np.zeros(0, dtype=np.int64), np.zeros(0)
    t = (offset - coefficients[0] * uv[:, 0] - coefficients[1] * uv[:, 1]) / coefficients[2]
    return np.arange(len(uv)), t


def _breaks(
    primitive: Constraint,
    definition: BlockModelDefinition,
    uv: np.ndarray,
    resolver: GeometryResolver | None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``(column, t)`` pairs: heights (model k) where lines through ``uv`` meet the boundary."""
    rotation = definition.rotation
    if isinstance(primitive, SolidConstraint):
        mesh = _resolve(primitive.solid, "solid", resolver)
        return [_mesh_breaks(mesh, definition, uv, True)]
    if isinstance(primitive, DtmConstraint):
        dtm = _resolve(primitive.surface, "dtm", resolver)
        return [_mesh_breaks(dtm.mesh, definition, uv, False)]
    if isinstance(primitive, PlaneConstraint):
        normal = primitive.normal @ rotation  # the normal in the model frame
        point = definition.world_to_local(primitive.point)
        return [_linear_breaks(normal, float(normal @ point), uv)]
    out = []
    for key, limits in primitive.ranges.items():
        for end, limit in enumerate(limits):
            if not np.isfinite(limit):
                continue
            if key in "xyz":
                axis = "xyz".index(key)
                out.append(_linear_breaks(rotation[axis], limit - definition.origin[axis], uv))
            elif key == "k":
                # parent index range [low, high] = local heights [low dz, (high + 1) dz)
                height = (limit + end) * definition.block_size[2]
                out.append((np.arange(len(uv)), np.full(len(uv), height)))
    return out


def _column_fraction(
    model: BlockModel,
    constraint: Constraint,
    resolver: GeometryResolver | None,
    rows: np.ndarray,
    samples: tuple[int, ...],
    primitives: list[Constraint],
) -> np.ndarray:
    definition = model.definition
    lo, hi = model.bounds_local(rows)
    size = hi - lo
    nu, nv = samples
    per_block = nu * nv
    u = lo[:, None, 0] + _ticks(nu)[None, :] * size[:, None, 0]
    v = lo[:, None, 1] + _ticks(nv)[None, :] * size[:, None, 1]
    uv = np.empty((len(rows), nu, nv, 2))
    uv[..., 0] = u[:, :, None]
    uv[..., 1] = v[:, None, :]
    uv = uv.reshape(-1, 2)
    n_lines = len(uv)
    bottom = np.repeat(lo[:, 2], per_block)
    top = np.repeat(hi[:, 2], per_block)
    line = np.arange(n_lines)
    columns, heights = [line], [bottom]
    for primitive in primitives:
        for column, t in _breaks(primitive, definition, uv, resolver):
            keep = (t > bottom[column]) & (t < top[column])
            columns.append(column[keep])
            heights.append(t[keep])
    columns.append(line)
    heights.append(top)
    column = np.concatenate(columns)
    height = np.concatenate(heights)
    order = np.lexsort((height, column))
    column, height = column[order], height[order]
    piece = np.flatnonzero((column[1:] == column[:-1]) & (height[1:] > height[:-1]))
    start, end, column = height[piece], height[piece + 1], column[piece]
    local = np.column_stack([uv[column, 0], uv[column, 1], 0.5 * (start + end)])
    points = _PointModel(definition, local, np.repeat(model.parent[rows], per_block)[column])
    inside = constraint.mask(points, resolver)
    length = np.bincount(column[inside], weights=(end - start)[inside], minlength=n_lines)
    fraction = (length / (top - bottom)).reshape(len(rows), per_block).mean(axis=1)
    return np.clip(fraction, 0.0, 1.0)


# --------------------------------------------------------------------------
# public functions
# --------------------------------------------------------------------------


def partial_fraction(
    model: BlockModel,
    constraint: Constraint,
    resolver: GeometryResolver | None = None,
    *,
    samples: Any = 5,
    method: str = "lattice",
    where: Any = None,
    face_tolerance: float | None = None,
    chunk_size: int = POINT_CHUNK,
) -> np.ndarray:
    """Fraction of each block's volume inside a geometric constraint.

    Parameters
    ----------
    model : the model (not modified).
    constraint : a geometric constraint or a boolean combination of them;
        ``inside solid and below topo``, ``not inside pit`` ... are fine.
    resolver : looks up the geometry the constraint names.
    samples : sampling density inside the blocks the boundary cuts: one
        integer ``n`` or one per model axis. ``method="lattice"``: ``(nx, ny,
        nz)`` cell centres. ``method="column"``: ``(nx, ny)`` lines along the
        model ``k`` axis (a third number is ignored).
    method : ``"lattice"`` or ``"column"``; see the module docstring for
        what each supports and how accurate it is.
    where : optional bool mask or row numbers; the result then has one value
        per selected block, in the order given.
    face_tolerance : as in :func:`~bakumine.core.blockmodel.subblocking.
        subblock_to_constraint`: a boundary that touches a block within this
        distance of its faces does not make it a cut block (it gets exactly
        0 or 1; the error is below ``tolerance / block size``).
    chunk_size : sample points (lattice) or lines (column) processed at a
        time; bounds the temporary memory and does not change the result.

    Returns
    -------
    (n,) float64 in [0, 1]: exactly 1.0 for blocks entirely inside, exactly
    0.0 for blocks entirely outside, sampled in between.
    """
    _require_geometric(constraint, "a partial percentage")
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    counts = _samples(samples, 3 if method == "lattice" else 2)
    primitives = _primitives(constraint) if method == "column" else []
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    rows = _row_numbers(model, where)
    tolerance = _face_tolerance(model, face_tolerance)
    status = _classify_rows(model, constraint, resolver, rows, tolerance, BOX_CHUNK)
    fraction = (status == INSIDE).astype(np.float64)
    cut = np.flatnonzero(status == STRADDLE)
    step = max(1, chunk_size // int(np.prod(counts)))
    for start in range(0, len(cut), step):
        part = cut[start : start + step]
        if method == "lattice":
            fraction[part] = _lattice_fraction(model, constraint, resolver, rows[part], counts)
        else:
            fraction[part] = _column_fraction(
                model, constraint, resolver, rows[part], counts, primitives
            )
    return fraction


def assign_partial(
    model: BlockModel,
    attribute: str,
    constraint: Constraint,
    resolver: GeometryResolver | None = None,
    *,
    scale: float = 1.0,
    **options: Any,
) -> np.ndarray:
    """Compute :func:`partial_fraction` for every block and store it in a float attribute.

    Parameters
    ----------
    attribute : name of a float attribute, e.g. ``"ore_pct"``. Created if it
        does not exist (float64, background 0, described by the constraint).
    scale : stored value = ``scale * fraction``; pass 100 to store percent.
    **options : ``samples``, ``method``, ``face_tolerance``, ``chunk_size``
        of :func:`partial_fraction`.

    Returns the fractions (unscaled), one per block.

    The stored values are those of the blocks as they are now: compute them
    after sub-blocking, and again if the model is split or merged later.
    """
    if "where" in options:
        raise TypeError("assign_partial fills every block; it takes no 'where'")
    if attribute in model and model.attribute(attribute).kind != "float":
        raise ValueError(f"attribute {attribute!r} is not a float attribute")
    fraction = partial_fraction(model, constraint, resolver, **options)
    if attribute not in model:
        unit = "%" if scale == 100.0 else ""
        model.add_attribute(
            attribute, "float", background=0.0, unit=unit,
            description="fraction of the block volume inside a constraint",
        )  # fmt: skip
    model.set(attribute, fraction if scale == 1.0 else scale * fraction)
    return fraction


def constraint_volume(
    model: BlockModel,
    constraint: Constraint | None = None,
    resolver: GeometryResolver | None = None,
    *,
    partial: Any = None,
) -> float:
    """Volume of the blocks selected by a constraint, optionally weighted by a partial fraction.

    ``sum(volume)`` over the blocks whose **centroid** satisfies
    ``constraint`` (all blocks when ``None``; any constraint, including
    attribute tests). With ``partial`` -- the name of an attribute holding
    fractions in [0, 1], or an array with one fraction per block -- each
    block counts ``fraction * volume``::

        constraint_volume(model, SolidConstraint("ore"), resolver)      # centroid rule
        constraint_volume(model, partial="ore_pct")                     # partial percentages

    A minimal helper; grouped reports (tonnes, grade, benches) are the
    business of :mod:`bakumine.core.report`.
    """
    volume = model.volumes()
    if partial is not None:
        weights = model.get(partial) if isinstance(partial, str) else np.asarray(partial)
        if weights.shape != volume.shape:
            raise ValueError(f"partial must have one value per block, got shape {weights.shape}")
        volume = volume * weights
    if constraint is not None:
        volume = volume[constraint.mask(model, resolver)]
    return float(volume.sum())
