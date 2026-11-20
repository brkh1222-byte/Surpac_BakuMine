"""Assigning values to blocks under a constraint: "coding" the model (research.md 6.4, 6.6).

:func:`assign` fills one attribute with a constant or an expression in the
blocks a constraint selects. :func:`code_model` applies an ordered list of
such assignments, the usual way a model is coded::

    model.add_attribute("rock", "category", background="WASTE")
    model.add_attribute("density", "float")
    code_model(model, [
        ("rock", "AIR", DtmConstraint("topo", "above"), True),
        ("rock", "ORE", SolidConstraint("ore") & DtmConstraint("topo", "below"), True),
        ("density", 'iif(rock == "ORE", 2.85, 2.70)', AttributeConstraint('rock != "AIR"')),
    ], resolver)

Which blocks get the value -- precisely
---------------------------------------
1. With ``subblock=True`` the model is first refined along the boundary of
   the constraint (:func:`~bakumine.core.blockmodel.subblocking.
   subblock_to_constraint`): every block the boundary cuts is split, down to
   the minimum block size (or ``max_level``). This needs a geometric
   constraint.
2. Then, with or without sub-blocking, a block is selected when its
   **centroid** satisfies the constraint (:meth:`Constraint.mask`). There is
   no partial selection: a block is assigned whole or not at all. A block
   still cut by the boundary -- a parent block when ``subblock=False``, a
   smallest sub-block otherwise -- goes to the side its centroid is on. (For
   the tie rules of a centroid exactly on a surface see
   :mod:`bakumine.core.blockmodel.constraints`.)

This is the behaviour research.md 6.4 describes for Surpac ("blocks cut by
the boundary are split until the minimum size, and the test is then made on
each sub-block's centroid"). It has not been compared with Surpac output.

Values
------
* a number, for a float or int attribute;
* a label (string) or ``None`` (missing), for a category attribute; new
  labels are added to the attribute's dictionary;
* an expression, for a float or int attribute: a string or a parsed
  :class:`~bakumine.core.blockmodel.expressions.Expression`, evaluated for
  the selected blocks only, from the values the model holds at that moment
  (:meth:`BlockModel.calculate`). So for a numeric attribute a string is
  always an expression, for a category attribute always a label; a category
  cannot be computed from an expression.

The attribute must exist (:meth:`BlockModel.add_attribute`): its kind and
background are a decision, not something to guess from the first value.

Precedence
----------
Steps are applied in the order given and each one simply overwrites: where
the selections of two steps overlap, the **later** step wins. Write general
rules first and exceptions after them (waste as background, then ore, then a
dyke that cuts the ore). An expression sees the results of all earlier steps.

Sub-blocking and earlier steps: a later step's sub-blocking splits blocks
coded by earlier steps; the children inherit the values, so nothing already
coded changes. An earlier step is, however, only as fine as the blocks were
when it ran: code each boundary with ``subblock=True`` in the step that
depends on it.

Merging
-------
``merge=True`` runs :func:`~bakumine.core.blockmodel.subblocking.
merge_uniform` over **all** attributes afterwards: sibling sub-blocks that
ended up with identical values in every attribute become one block again.
This changes no value at any point of the model; it only removes sub-blocks
that turned out not to be needed (typically well over a third of them).
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from bakumine.core.blockmodel.attributes import BUILTIN_VARIABLES
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import Constraint, ConstraintError, GeometryResolver
from bakumine.core.blockmodel.expressions import Expression, parse
from bakumine.core.blockmodel.subblocking import (
    SubblockResult,
    _level_limit,
    _require_geometric,
    merge_uniform,
    subblock_to_constraint,
)

__all__ = ["AssignResult", "CodingStep", "assign", "code_model"]


@dataclass(frozen=True)
class CodingStep:
    """One assignment of :func:`code_model`; the arguments of :func:`assign`."""

    attribute: str
    value: Any
    constraint: Constraint | None = None
    subblock: bool = False
    max_level: int | None = None


@dataclass(frozen=True)
class AssignResult:
    """Summary of one assignment.

    Attributes
    ----------
    attribute, value : what was assigned (the expression text for an expression).
    n_assigned : number of blocks that received the value, counted after
        sub-blocking and before any merge.
    volume : their total volume.
    n_split : blocks split by the sub-blocking of this step (0 without).
    n_merged : sibling sets merged afterwards (0 without ``merge``).
    n_invalid : for an expression, the number of values that became inf / NaN
        from finite operands (division by zero, ...); they are stored as such.
    n_blocks : number of blocks in the model when the step was finished.
    subblock : the :class:`SubblockResult` of this step, or ``None``.
    """

    attribute: str
    value: Any
    n_assigned: int
    volume: float
    n_split: int
    n_merged: int
    n_invalid: int
    n_blocks: int
    subblock: SubblockResult | None = None


def _is_expression(model: BlockModel, attribute: str, value: Any) -> bool:
    """Validate attribute / value; True when the value is an expression."""
    spec = model.attribute(attribute)
    if isinstance(value, Expression):
        if spec.kind == "category":
            raise ValueError(
                f"attribute {attribute!r} is a category: it cannot be computed from an expression"
            )
        return True
    if spec.kind == "category":
        if value is not None and not isinstance(value, str):
            raise TypeError(
                f"attribute {attribute!r} is a category: assign a label (string) or None, "
                f"got {type(value).__name__}"
            )
        return False
    if isinstance(value, str):
        # reject a malformed expression or an unknown name before anything is changed
        parse(value, [*model.attribute_names, *BUILTIN_VARIABLES])
        return True
    if isinstance(value, bool) or not isinstance(value, int | float | np.integer | np.floating):
        raise TypeError(
            f"attribute {attribute!r} is {spec.kind}: assign a number or an expression, "
            f"got {type(value).__name__}"
        )
    spec.to_stored(value)  # e.g. a fraction for an integer attribute: refuse it now
    return False


def _check(model: BlockModel, step: CodingStep) -> bool:
    is_expression = _is_expression(model, step.attribute, step.value)
    if step.constraint is not None and not isinstance(step.constraint, Constraint):
        raise ConstraintError(
            f"the constraint must be a Constraint or None, got {type(step.constraint).__name__}"
        )
    if step.subblock:
        if step.constraint is None:
            raise ConstraintError("subblock=True needs a constraint to sub-block on")
        _require_geometric(step.constraint, "sub-blocking")
        _level_limit(model, step.max_level)
    return is_expression


def _apply(
    model: BlockModel, step: CodingStep, resolver: GeometryResolver | None, merge: bool
) -> AssignResult:
    is_expression = _check(model, step)
    sub = None
    if step.subblock:
        sub = subblock_to_constraint(model, step.constraint, resolver, max_level=step.max_level)
    mask = None if step.constraint is None else step.constraint.mask(model, resolver)
    volume = model.volumes()
    n_assigned = model.n_blocks if mask is None else int(mask.sum())
    total = float(volume.sum() if mask is None else volume[mask].sum())
    n_invalid = 0
    if is_expression:
        n_invalid = model.calculate(step.attribute, step.value, mask)
    else:
        model.set(step.attribute, step.value, mask)
    n_merged = merge_uniform(model) if merge else 0
    value = step.value.source if isinstance(step.value, Expression) else step.value
    return AssignResult(
        attribute=step.attribute,
        value=value,
        n_assigned=n_assigned,
        volume=total,
        n_split=0 if sub is None else sub.n_split,
        n_merged=n_merged,
        n_invalid=n_invalid,
        n_blocks=model.n_blocks,
        subblock=sub,
    )


def assign(
    model: BlockModel,
    attribute: str,
    value: Any,
    constraint: Constraint | None = None,
    resolver: GeometryResolver | None = None,
    *,
    subblock: bool = False,
    max_level: int | None = None,
    merge: bool = False,
) -> AssignResult:
    """Fill an attribute with a value or an expression in the blocks a constraint selects.

    Parameters
    ----------
    model : modified in place.
    attribute : name of an existing attribute.
    value : a number, a category label / ``None``, or an expression (string
        or :class:`Expression`) for a numeric attribute; see the module
        docstring.
    constraint : which blocks; ``None`` = every block. Any constraint,
        including attribute tests and combinations.
    resolver : looks up the geometry the constraint names.
    subblock : first split the blocks the constraint's boundary cuts (the
        constraint must then be geometric).
    max_level : limit of that sub-blocking; default the model's minimum
        block size.
    merge : afterwards merge sibling sub-blocks that are identical in all
        attributes.

    Selection is by block **centroid**, after the optional sub-blocking; see
    the module docstring. Blocks not selected keep their value. Everything is
    validated before the model is touched.

    Returns
    -------
    AssignResult
    """
    step = CodingStep(attribute, value, constraint, bool(subblock), max_level)
    return _apply(model, step, resolver, bool(merge))


def _as_step(item: Any) -> CodingStep:
    if isinstance(item, CodingStep):
        return item
    if isinstance(item, Mapping):
        try:
            return CodingStep(**item)
        except TypeError as exc:
            raise ValueError(f"invalid coding step {dict(item)!r}: {exc}") from None
    if isinstance(item, tuple | list) and 2 <= len(item) <= 5:
        return CodingStep(*item)
    raise ValueError(
        "a coding step is a CodingStep, a mapping with its fields, or a tuple "
        "(attribute, value[, constraint[, subblock[, max_level]]])"
    )


def code_model(
    model: BlockModel,
    steps: Iterable[Any],
    resolver: GeometryResolver | None = None,
    *,
    merge: bool = False,
) -> list[AssignResult]:
    """Apply an ordered list of assignments.

    Parameters
    ----------
    steps : each a :class:`CodingStep`, a mapping with its fields, or a tuple
        ``(attribute, value[, constraint[, subblock[, max_level]]])``.
    resolver : used by every step.
    merge : merge uniform sibling sub-blocks once, after the last step (over
        all attributes); the number of merges is reported in the last
        result's ``n_merged``.

    The steps run in order and later steps overwrite earlier ones where
    their selections overlap (see "Precedence" in the module docstring). All
    steps are validated against the model before the first one runs;
    geometry names are only resolved when their step runs.

    Returns one :class:`AssignResult` per step: blocks assigned, their
    volume, blocks split.
    """
    plan = [_as_step(item) for item in steps]
    for step in plan:
        _check(model, step)
    results = []
    for number, step in enumerate(plan):
        results.append(_apply(model, step, resolver, merge and number == len(plan) - 1))
    return results
