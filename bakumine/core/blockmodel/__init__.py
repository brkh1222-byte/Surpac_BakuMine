"""Block model definition, sub-blocks, attributes, constraints and storage."""

from bakumine.core.blockmodel.attributes import Attribute
from bakumine.core.blockmodel.blocks import BlockModel, SplitResult
from bakumine.core.blockmodel.constraints import (
    And,
    AttributeConstraint,
    BlockIdConstraint,
    Constraint,
    ConstraintError,
    DictResolver,
    DtmConstraint,
    FileResolver,
    Not,
    Or,
    PlaneConstraint,
    PolygonConstraint,
    RangeConstraint,
    SolidConstraint,
    combine,
    constraint_from_dict,
    load_constraint,
    save_constraint,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.expressions import ExpressionError, ExpressionWarning
from bakumine.core.blockmodel.partial import (
    assign_partial,
    constraint_volume,
    partial_fraction,
)
from bakumine.core.blockmodel.subblocking import (
    SubblockResult,
    merge_uniform,
    subblock_to_constraint,
)

__all__ = [
    "And",
    "Attribute",
    "AttributeConstraint",
    "BlockIdConstraint",
    "BlockModel",
    "BlockModelDefinition",
    "Constraint",
    "ConstraintError",
    "DictResolver",
    "DtmConstraint",
    "ExpressionError",
    "ExpressionWarning",
    "FileResolver",
    "Not",
    "Or",
    "PlaneConstraint",
    "PolygonConstraint",
    "RangeConstraint",
    "SolidConstraint",
    "SplitResult",
    "combine",
    "constraint_from_dict",
    "load_constraint",
    "save_constraint",
]

# sub-blocking on constraints (P3.T5)
__all__ += ["SubblockResult", "merge_uniform", "subblock_to_constraint"]

# partial percentages (P3.T6)
__all__ += ["assign_partial", "constraint_volume", "partial_fraction"]
