"""Sub-blocking on geometric constraints (research.md sections 6.2 and 6.4).

A parent block that is cut by an ore contact or by the topography is neither
ore nor waste, neither rock nor air. :func:`subblock_to_constraint` refines
the model along the boundary of a geometric constraint: every block the
boundary cuts is split, its children are examined again, and so on down to
the smallest block the definition allows. Blocks that are entirely on one
side are never touched, so the model stays small: the number of blocks grows
with the *area* of the boundary, not with the volume of the model.

After sub-blocking, a constraint is applied as always, by the **centroid** of
each (sub-)block (:meth:`Constraint.mask`); see
:func:`bakumine.core.blockmodel.coding.assign`.

How a block is judged
---------------------
With :meth:`Constraint.classify_boxes`, which answers ``INSIDE``, ``OUTSIDE``
or ``STRADDLE`` for a box conservatively (a box is only called inside /
outside when it certainly is). Only ``STRADDLE`` blocks are split. Boolean
combinations are refined along the boundaries of all their parts that matter:
for ``A and B`` a block outside ``A`` is not split however ``B`` cuts it.
Constraints that are not geometric (attribute tests, block-id lists) cannot
be sub-blocked on and raise :class:`ConstraintError`.

Face tolerance
--------------
``classify_boxes`` counts a boundary that merely *touches* a box as cutting
it. A flat boundary lying on block faces -- a bench floor, the flat top of a
solid, a box solid snapped to the grid -- would then split the blocks on both
sides down to the smallest size for nothing. Therefore the boxes handed to
the classifier are shrunk by a **face tolerance** on every side (by default
four times the rounding margin of the classifier: 256 ulp of the largest
model coordinate, about 2.4e-7 m at northing 5.3e6, 5.7e-14 m near the
origin). A block is split only when the boundary enters it deeper than that.
The price is that a block left whole may contain a sliver of the other side
thinner than the tolerance; this cannot move a centroid across the boundary
and changes volumes by less than ``tolerance * boundary area``. Pass
``face_tolerance=0.0`` for the strictly conservative behaviour.

Cost and determinism
--------------------
Each level is one vectorised pass: classify the candidate boxes (in chunks of
``chunk_size`` boxes, so temporary memory is bounded whatever the model
size), split the straddling ones with :meth:`BlockModel.split`, and take
their children as the next candidates. No Python loop runs per block. The
result depends only on the model, the constraint and ``max_level`` -- not on
``chunk_size``, not on the number of threads, and the table order is the
canonical (parent, Morton) order of :mod:`bakumine.core.blockmodel.blocks`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    INSIDE,
    OUTSIDE,
    STRADDLE,
    Constraint,
    ConstraintError,
    GeometryResolver,
    _rounding_margin,
)

__all__ = ["SubblockResult", "merge_uniform", "subblock_to_constraint"]

#: boxes classified per call of ``classify_boxes``
BOX_CHUNK = 1 << 18


@dataclass(frozen=True)
class SubblockResult:
    """Statistics of :func:`subblock_to_constraint`.

    "Examined" blocks are the selected blocks (``where``, default all) and
    every block created from them; blocks outside the selection are not
    counted anywhere but in ``n_before`` / ``n_after``.

    Attributes
    ----------
    n_before, n_after : number of blocks in the model before and after.
    max_level : the level limit that was applied.
    split_per_level : tuple, one entry per level ``0 .. model.max_level - 1``
        Number of blocks of that level that were split (each into 8, or 4 /
        2 children for a partial split).
    n_inside, n_outside : examined blocks left whole because they are
        entirely inside / outside the constraint.
    n_straddle : examined blocks the boundary still cuts: they are at
        ``max_level`` and cannot be split further. Their centroid decides
        which side they are coded to.
    volume_inside, volume_outside, volume_straddle : the volumes of those
        three groups. The true volume inside the constraint, within the
        examined blocks, lies between ``volume_inside`` and ``volume_inside +
        volume_straddle`` (up to the face tolerance).
    """

    n_before: int
    n_after: int
    max_level: int
    split_per_level: tuple[int, ...]
    n_inside: int
    n_outside: int
    n_straddle: int
    volume_inside: float
    volume_outside: float
    volume_straddle: float

    @property
    def n_split(self) -> int:
        """Total number of blocks that were split."""
        return int(sum(self.split_per_level))


# --------------------------------------------------------------------------
# helpers shared with partial.py and coding.py
# --------------------------------------------------------------------------


def _require_geometric(constraint: Any, what: str) -> None:
    if not isinstance(constraint, Constraint):
        raise ConstraintError(f"{what} needs a Constraint, got {type(constraint).__name__}")
    if not constraint.is_geometric:
        raise ConstraintError(
            f"{what} needs a geometric constraint (DTM, solid, plane, polygon, range and "
            "their combinations); this one contains an attribute or block-id test"
        )


def _row_numbers(model: BlockModel, where: Any) -> np.ndarray:
    """Row numbers (int64) of a selection: None = all, a bool mask, or row numbers as given."""
    n = model.n_blocks
    if where is None:
        return np.arange(n, dtype=np.int64)
    sel = np.asarray(where)
    if sel.dtype == bool:
        if sel.shape != (n,):
            raise ValueError(f"mask must have shape ({n},), got {sel.shape}")
        return np.flatnonzero(sel)
    if sel.size == 0:
        return np.zeros(0, dtype=np.int64)
    if sel.dtype.kind not in "iu":
        raise TypeError("select blocks with a boolean mask or integer row numbers")
    sel = sel.astype(np.int64).reshape(-1)
    if sel.min() < 0 or sel.max() >= n:
        raise IndexError("block row number out of range")
    return sel


def _face_tolerance(model: BlockModel, face_tolerance: float | None) -> float:
    if face_tolerance is None:
        tol = 4.0 * _rounding_margin(model.definition.bounding_box())
    else:
        tol = float(face_tolerance)
        if not (np.isfinite(tol) and tol >= 0.0):
            raise ValueError("face_tolerance must be finite and >= 0")
    smallest = min(model.definition.min_block_size)
    if not tol < 0.25 * smallest:
        raise ValueError(
            f"face tolerance {tol:g} m is not small against the smallest block ({smallest:g} m)"
        )
    return tol


def _classify_rows(
    model: BlockModel,
    constraint: Constraint,
    resolver: GeometryResolver | None,
    rows: np.ndarray,
    tolerance: float,
    chunk_size: int,
) -> np.ndarray:
    """``classify_boxes`` for the blocks at ``rows``, shrunk by ``tolerance``, in chunks."""
    chunk_size = int(chunk_size)
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    status = np.empty(len(rows), dtype=np.int8)
    for start in range(0, len(rows), chunk_size):
        part = rows[start : start + chunk_size]
        lo, hi = model.bounds_local(part)
        if tolerance:
            lo = lo + tolerance
            hi = hi - tolerance
        status[start : start + len(part)] = constraint.classify_boxes(
            model.definition, lo, hi, resolver
        )
    return status


def _level_limit(model: BlockModel, max_level: int | None) -> int:
    if max_level is None:
        return model.max_level
    if isinstance(max_level, bool) or int(max_level) != max_level:
        raise ValueError("max_level must be an integer")
    if not 0 <= max_level <= model.max_level:
        raise ValueError(
            f"max_level must be within 0..{model.max_level} "
            f"(max_depth of the model is {model.definition.max_depth}), got {max_level}"
        )
    return int(max_level)


# --------------------------------------------------------------------------
# public functions
# --------------------------------------------------------------------------


def subblock_to_constraint(
    model: BlockModel,
    constraint: Constraint,
    resolver: GeometryResolver | None = None,
    *,
    max_level: int | None = None,
    where: Any = None,
    face_tolerance: float | None = None,
    chunk_size: int = BOX_CHUNK,
) -> SubblockResult:
    """Split the blocks cut by the boundary of a geometric constraint.

    Parameters
    ----------
    model : the model; modified in place (row numbers change).
    constraint : a geometric constraint or a boolean combination of geometric
        constraints (:class:`ConstraintError` otherwise).
    resolver : looks up DTMs / solids / polygons the constraint names.
    max_level : deepest block level to create. Default (``None``) the
        model's own limit ``max(definition.max_depth)``, i.e. down to the
        minimum block size; a lower number stops earlier (0 splits nothing
        and only reports). Blocks are ``block_size / 2**level`` (per axis
        capped by that axis' ``max_depth``).
    where : optional bool mask or row numbers: only these blocks (and the
        blocks created from them) are examined and split.
    face_tolerance : see the module docstring; ``None`` = automatic, ``0.0``
        = strictly conservative.
    chunk_size : boxes classified at a time; bounds the temporary memory and
        does not change the result.

    Children inherit all attribute values of the block they come from
    (:meth:`BlockModel.split`): sub-blocking never changes what is stored at
    any point of the model, it only makes room for values to differ.

    Blocks that are already sub-blocks are handled like any other: a
    straddling block of any level below ``max_level`` is split. Calling the
    function again with the same arguments changes nothing.

    Returns
    -------
    SubblockResult
    """
    _require_geometric(constraint, "sub-blocking")
    limit = _level_limit(model, max_level)
    tolerance = _face_tolerance(model, face_tolerance)
    n_before = model.n_blocks
    candidates = np.unique(_row_numbers(model, where))
    split_per_level = np.zeros(model.max_level, dtype=np.int64)
    counts = {INSIDE: 0, OUTSIDE: 0, STRADDLE: 0}
    volumes = {INSIDE: 0.0, OUTSIDE: 0.0, STRADDLE: 0.0}
    while len(candidates):
        status = _classify_rows(model, constraint, resolver, candidates, tolerance, chunk_size)
        level = model.level[candidates]
        volume = model.volumes(candidates)
        to_split = (status == STRADDLE) & (level < limit)
        for code in (INSIDE, OUTSIDE, STRADDLE):
            final = (status == code) & ~to_split
            counts[code] += int(final.sum())
            volumes[code] += float(volume[final].sum())
        if not to_split.any():
            break
        split_per_level += np.bincount(level[to_split], minlength=model.max_level)[
            : model.max_level
        ]
        result = model.split(candidates[to_split])
        candidates = np.flatnonzero(result.is_child)
    return SubblockResult(
        n_before=n_before,
        n_after=model.n_blocks,
        max_level=limit,
        split_per_level=tuple(int(c) for c in split_per_level),
        n_inside=counts[INSIDE],
        n_outside=counts[OUTSIDE],
        n_straddle=counts[STRADDLE],
        volume_inside=volumes[INSIDE],
        volume_outside=volumes[OUTSIDE],
        volume_straddle=volumes[STRADDLE],
    )


def merge_uniform(model: BlockModel, attributes: Any = None, where: Any = None) -> int:
    """Merge complete sets of sibling sub-blocks that carry identical values.

    The counterpart of :func:`subblock_to_constraint`, to be run after
    coding: sub-blocks that turned out to hold the same values as all their
    siblings become one block again (repeatedly, up to parent blocks), which
    keeps the model small.

    Parameters
    ----------
    attributes : names of the attributes that must be identical across the
        siblings. Default (``None``): **all** attributes, which loses
        nothing -- every point of the model keeps the value of every
        attribute. With an explicit subset the merged block takes the other
        attributes from its first child (see :meth:`BlockModel.merge`), so
        name every attribute whose detail matters. An empty selection (no
        names, or a model without attributes) is refused: it would merge on
        structure alone and undo the sub-blocking.
    where : optional bool mask or row numbers; only sets whose siblings are
        all selected are merged.

    A merge never crosses a boundary that was coded: the blocks on its two
    sides differ in the coded attribute, so their sibling sets are not
    uniform. (NaN equals NaN and missing equals missing for this purpose.)

    Returns the number of merges (sibling sets replaced by one block).
    """
    names = model.attribute_names if attributes is None else list(attributes)
    if isinstance(attributes, str):
        names = [attributes]
    if not names:
        raise ValueError(
            "merge_uniform needs at least one attribute to compare; merging on structure "
            "alone would undo the sub-blocking (use BlockModel.merge([]) if that is intended)"
        )
    return model.merge(names, where)
