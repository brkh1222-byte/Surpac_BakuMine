"""Resource classification of estimated blocks into a category attribute.

The class of a block is decided from the diagnostics the estimation wrote next to the
grade (:mod:`bakumine.core.estimation.block_estimation`: ``<attr>_pass``, ``_ns``,
``_dist``, ``_ndist``, ``_kv``, ``_slope``, ``_ke``) and from the drillhole support
computed here (:func:`hole_support`). The result is one category attribute (default
``res_class``), so ``bakumine blockmodel report --group-by res_class`` reports by class.

Rules
-----
``categories`` is an ordered list of rule sets ``{"label", "criteria", "where"}``. The
labels are free text: ``Measured / Indicated / Inferred``, ``B / C1 / C2 / P1``, ...

* Only **estimated** blocks are classified: blocks whose grade attribute is not missing.
* Rule sets are applied in the order given. A block gets the label of the **first** rule
  set whose criteria it meets all of; later rule sets never change it. So every block has
  at most one class, and a rule set without criteria takes every estimated block that is
  left ("Inferred: the rest").
* An estimated block that meets no rule set stays unclassified (missing), like the blocks
  that are not estimated.

Criteria (all optional, all must hold; a missing value never meets a criterion):

======================== ================== ==========================================
criterion                attribute          meets when
======================== ================== ==========================================
``max_pass``             ``<attr>_pass``    ``1 <= pass <= value``
``min_samples``          ``<attr>_ns``      ``samples >= value``
``max_mean_distance``    ``<attr>_dist``    ``mean distance <= value``
``max_nearest_distance`` ``<attr>_ndist``   ``nearest distance <= value``
``max_kriging_variance`` ``<attr>_kv``      ``variance <= value``
``min_slope``            ``<attr>_slope``   ``slope of regression >= value``
``min_efficiency``       ``<attr>_ke``      ``kriging efficiency >= value``
``min_holes``            ``<attr>_nholes``  ``holes within the radius >= value``
``max_hole_distance``    ``<attr>_dhole``   ``distance to the n-th hole <= value``
======================== ================== ==========================================

``inputs`` renames the attribute a criterion reads (``{"n_samples": "au_count"}``; keys:
``pass_number, n_samples, mean_distance, nearest_distance, variance, slope, efficiency,
holes, hole_distance``). ``where`` is an optional block expression that must hold too
(``rock == "ORE"``). The two distances of the estimation are the anisotropic distances of
the search ellipsoid of the pass that estimated the block, as that module stores them.

Drillhole support
-----------------
The estimation does not record which holes it used. With ``hole_support`` (``radius``,
``nth``) and a composite table, :func:`hole_support` computes for every estimated block

* ``<attr>_nholes``: the number of distinct holes with at least one composite within
  ``radius`` of the block centroid, and
* ``<attr>_dhole``: the distance from the centroid to the ``nth`` nearest hole (default
  the 3rd), the distance to a hole being that to its nearest composite; NaN when there
  are fewer than ``nth`` holes.

Both are **isotropic** straight-line distances in metres between the block centroid and
composite midpoints (``x, y, z`` columns), over all composites of the table (with
``value_column``: those with a value in that column). They describe the drilling around
the block; they are not "the holes the estimate used". Without composites the two
attributes must already be in the model.

Smoothing (optional)
--------------------
``smoothing = {"iterations": 1, "min_agree": 4, "max_same": 0}`` removes isolated blocks
with a majority filter on the **parent block grid**; it is off unless asked for:

1. Each parent block that holds at least one classified block gets a cell class: the
   class with the largest classified volume among its blocks (for a model without
   sub-blocks, the class of the block). On a tie the later rule set (lower confidence).
2. For each such cell, its up to six face neighbours (``i +- 1``, ``j +- 1``, ``k +- 1``
   in the model grid) that have a cell class are counted by class.
3. A cell is changed when at most ``max_same`` neighbours share its class and at least
   ``min_agree`` neighbours share one other class; it takes that class (on a tie the
   later rule set). All cells are judged on the classes before the pass.
4. Every classified block of a changed cell gets the new class. Unclassified blocks
   stay unclassified, classified blocks stay classified, and blocks of an unchanged cell
   keep their own class.

Smoothing can move a block to a class whose criteria it does not meet: that is its
purpose, and the summary counts the changed blocks.

Parameter file (JSON, round trip through :class:`ClassificationParams`)::

    {"bakumine_classification": 1, "attribute": "au", "output": "res_class",
     "hole_support": {"radius": 50.0, "nth": 3},
     "categories": [
       {"label": "Measured", "criteria": {"max_pass": 1, "min_samples": 8, "min_holes": 3,
                                          "max_nearest_distance": 25.0}},
       {"label": "Indicated", "criteria": {"max_pass": 2, "min_holes": 2}},
       {"label": "Inferred"}],
     "smoothing": null}
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.attributes import check_name
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.estimation.block_estimation import OutputNames

__all__ = [
    "CRITERIA",
    "ClassRule",
    "ClassificationParams",
    "ClassificationSummary",
    "HoleSupport",
    "Smoothing",
    "add_hole_support",
    "classify",
    "hole_support",
    "smooth_classes",
]

FORMAT_VERSION = 1
MAX_CLASSES = 100

#: criterion -> (input key, comparison)
CRITERIA: dict[str, tuple[str, str]] = {
    "max_pass": ("pass_number", "<="),
    "min_samples": ("n_samples", ">="),
    "max_mean_distance": ("mean_distance", "<="),
    "max_nearest_distance": ("nearest_distance", "<="),
    "max_kriging_variance": ("variance", "<="),
    "min_slope": ("slope", ">="),
    "min_efficiency": ("efficiency", ">="),
    "min_holes": ("holes", ">="),
    "max_hole_distance": ("hole_distance", "<="),
}
_HOLE_KEYS = {"holes": "_nholes", "hole_distance": "_dhole"}
INPUT_KEYS = (*OutputNames().names("").keys(), *_HOLE_KEYS)


def _number(value: Any, what: str, *, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise ValueError(f"{what} must be a finite number, got {value!r}")
    if minimum is not None and value < minimum:
        raise ValueError(f"{what} must be >= {minimum:g}, got {value!r}")
    return float(value)


def _only(data: dict[str, Any], known: tuple[str, ...], what: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{what} must be an object")
    unknown = sorted(set(data) - set(known))
    if unknown:
        raise ValueError(f"{what}: unknown key(s) {', '.join(unknown)}")


@dataclass(frozen=True)
class ClassRule:
    """One rule set: a label, criteria (see :data:`CRITERIA`) and an optional expression."""

    label: str
    criteria: dict[str, float] = field(default_factory=dict)
    where: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("a class label must be a non-empty string")
        if not isinstance(self.criteria, dict):
            raise ValueError(f"class {self.label!r}: criteria must be an object")
        unknown = sorted(set(self.criteria) - set(CRITERIA))
        if unknown:
            raise ValueError(
                f"class {self.label!r}: unknown criterion {', '.join(unknown)}; "
                f"known: {', '.join(CRITERIA)}"
            )
        clean = {
            k: _number(v, f"class {self.label!r}: {k}")
            for k, v in self.criteria.items()
            if v is not None
        }
        object.__setattr__(self, "criteria", clean)
        if self.where is not None and (not isinstance(self.where, str) or not self.where.strip()):
            object.__setattr__(self, "where", None)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"label": self.label, "criteria": dict(self.criteria)}
        if self.where:
            out["where"] = self.where
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassRule:
        _only(data, ("label", "criteria", "where"), "a class")
        return cls(data.get("label"), dict(data.get("criteria") or {}), data.get("where"))  # type: ignore[arg-type]


@dataclass(frozen=True)
class HoleSupport:
    """Parameters of :func:`hole_support`."""

    radius: float
    nth: int = 3
    hole_column: str = "hole_id"
    value_column: str | None = None

    def __post_init__(self) -> None:
        radius = _number(self.radius, "hole_support radius")
        if radius <= 0:
            raise ValueError("hole_support radius must be positive")
        object.__setattr__(self, "radius", radius)
        if isinstance(self.nth, bool) or int(self.nth) != self.nth or self.nth < 1:
            raise ValueError(f"hole_support nth must be a positive integer, got {self.nth!r}")
        object.__setattr__(self, "nth", int(self.nth))

    def to_dict(self) -> dict[str, Any]:
        return {
            "radius": self.radius,
            "nth": self.nth,
            "hole_column": self.hole_column,
            "value_column": self.value_column,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HoleSupport:
        _only(data, ("radius", "nth", "hole_column", "value_column"), "hole_support")
        if "radius" not in data:
            raise ValueError("hole_support needs a radius")
        return cls(
            data["radius"],
            data.get("nth", 3),
            data.get("hole_column") or "hole_id",
            data.get("value_column") or None,
        )


@dataclass(frozen=True)
class Smoothing:
    """Parameters of :func:`smooth_classes` (module docstring, "Smoothing")."""

    iterations: int = 1
    min_agree: int = 4
    max_same: int = 0

    def __post_init__(self) -> None:
        for name, low, high in (("iterations", 1, 100), ("min_agree", 1, 6), ("max_same", 0, 5)):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int | float) or (
                int(value) != value or not low <= value <= high
            ):  # fmt: skip
                raise ValueError(f"smoothing {name} must be an integer {low}..{high}")
            object.__setattr__(self, name, int(value))

    def to_dict(self) -> dict[str, int]:
        return {
            "iterations": self.iterations,
            "min_agree": self.min_agree,
            "max_same": self.max_same,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Smoothing:
        _only(data, ("iterations", "min_agree", "max_same"), "smoothing")
        return cls(**data)


@dataclass(frozen=True)
class ClassificationParams:
    """Everything :func:`classify` needs; JSON through :meth:`to_dict` / :meth:`from_dict`."""

    attribute: str
    categories: tuple[ClassRule, ...]
    output: str = "res_class"
    hole_support: HoleSupport | None = None
    smoothing: Smoothing | None = None
    inputs: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.attribute, str) or not self.attribute:
            raise ValueError("give the estimated grade attribute")
        if not isinstance(self.output, str) or not self.output:
            raise ValueError("give the name of the class attribute")
        rules = tuple(self.categories)
        if not rules:
            raise ValueError("give at least one class")
        if len(rules) > MAX_CLASSES:
            raise ValueError(f"at most {MAX_CLASSES} classes")
        labels = [r.label for r in rules]
        if len(set(labels)) != len(labels):
            raise ValueError("class labels must be distinct")
        object.__setattr__(self, "categories", rules)
        unknown = sorted(set(self.inputs) - set(INPUT_KEYS))
        if unknown:
            raise ValueError(f"inputs: unknown key(s) {unknown}; known: {list(INPUT_KEYS)}")
        names = self.input_names()
        if self.output == self.attribute or self.output in names.values():
            raise ValueError(f"the class attribute {self.output!r} is an input of the rules")
        # attributes classify may create: valid names, known before anything is written
        check_name(self.output)
        if self.hole_support is not None:
            check_name(names["holes"])
            check_name(names["hole_distance"])

    @property
    def labels(self) -> list[str]:
        return [r.label for r in self.categories]

    def input_names(self) -> dict[str, str]:
        """``{input key: attribute name}`` (defaults by suffix, then ``inputs``)."""
        names = OutputNames().names(self.attribute)
        names.update({k: self.attribute + s for k, s in _HOLE_KEYS.items()})
        names.update({k: str(v) for k, v in self.inputs.items() if v})
        return names

    def used_inputs(self) -> list[str]:
        """Input keys some criterion reads, in table order."""
        used = {CRITERIA[c][0] for r in self.categories for c in r.criteria}
        return [k for k in INPUT_KEYS if k in used]

    def to_dict(self) -> dict[str, Any]:
        return {
            "bakumine_classification": FORMAT_VERSION,
            "attribute": self.attribute,
            "output": self.output,
            "inputs": dict(self.inputs),
            "hole_support": self.hole_support.to_dict() if self.hole_support else None,
            "categories": [r.to_dict() for r in self.categories],
            "smoothing": self.smoothing.to_dict() if self.smoothing else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ClassificationParams:
        keys = ("bakumine_classification", "attribute", "output", "inputs", "hole_support",
                "categories", "smoothing")  # fmt: skip
        _only(data, keys, "classification parameters")
        if data.get("bakumine_classification", FORMAT_VERSION) != FORMAT_VERSION:
            raise ValueError(
                f"unsupported classification format {data.get('bakumine_classification')!r}"
            )
        categories = data.get("categories")
        if not isinstance(categories, list):
            raise ValueError("categories must be a list of classes")
        inputs = data.get("inputs") or {}
        if not isinstance(inputs, dict):
            raise ValueError("inputs must be an object")
        return cls(
            attribute=data.get("attribute"),  # type: ignore[arg-type]
            categories=tuple(ClassRule.from_dict(c) for c in categories),
            output=data.get("output") or "res_class",
            hole_support=(
                HoleSupport.from_dict(data["hole_support"]) if data.get("hole_support") else None
            ),
            smoothing=Smoothing.from_dict(data["smoothing"]) if data.get("smoothing") else None,
            inputs=dict(inputs),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(
            json.dumps(self.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> ClassificationParams:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class ClassificationSummary:
    attribute: str
    output: str
    n_blocks: int
    n_estimated: int
    n_classified: int
    n_smoothed: int
    #: label, blocks, volume -- in rule order
    classes: list[dict[str, Any]]
    #: attributes written besides ``output`` (drillhole support)
    written: list[str]

    @property
    def n_unclassified(self) -> int:
        """Estimated blocks that met no rule set."""
        return self.n_estimated - self.n_classified

    def to_dict(self) -> dict[str, Any]:
        return {
            "attribute": self.attribute,
            "output": self.output,
            "n_blocks": self.n_blocks,
            "n_estimated": self.n_estimated,
            "n_classified": self.n_classified,
            "n_unclassified": self.n_unclassified,
            "n_not_estimated": self.n_blocks - self.n_estimated,
            "n_smoothed": self.n_smoothed,
            "classes": self.classes,
            "written": self.written,
        }


# ------------------------------------------------------------------- drillhole support


def hole_support(
    points: Any,
    composites: pd.DataFrame,
    *,
    radius: float,
    nth: int = 3,
    hole_column: str = "hole_id",
    value_column: str | None = None,
    chunk_size: int = 20_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Holes around points: ``(n_holes_within_radius, distance_to_nth_nearest_hole)``.

    ``points`` is (m, 3); ``composites`` needs ``x, y, z`` and ``hole_column``. Distances
    are isotropic and to composite midpoints (module docstring). The count is int32; the
    distance is NaN where fewer than ``nth`` holes exist.
    """
    from scipy.spatial import cKDTree

    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    for column in ("x", "y", "z", hole_column):
        if column not in composites.columns:
            raise ValueError(f"the composite table has no column {column!r}")
    keep = composites[["x", "y", "z"]].notna().all(axis=1) & composites[hole_column].notna()
    if value_column is not None:
        if value_column not in composites.columns:
            raise ValueError(f"the composite table has no column {value_column!r}")
        keep &= composites[value_column].notna()
    xyz = composites.loc[keep, ["x", "y", "z"]].to_numpy(dtype=np.float64)
    holes = pd.factorize(composites.loc[keep, hole_column])[0].astype(np.int64)
    count = np.zeros(len(points), dtype=np.int32)
    distance = np.full(len(points), np.nan)
    n = len(xyz)
    if n == 0 or len(points) == 0:
        return count, distance
    tree = cKDTree(xyz)
    for start in range(0, len(points), chunk_size):
        pending = np.arange(start, min(start + chunk_size, len(points)))
        k = min(n, max(8, 4 * nth))
        while pending.size:
            step = max(1, 4_000_000 // k)  # bounds the (rows, k) arrays
            left = []
            for a in range(0, pending.size, step):
                rows = pending[a : a + step]
                d, index = tree.query(points[rows], k=k)
                d, index = d.reshape(len(rows), k), index.reshape(len(rows), k)
                h = holes[index]
                # first composite of each hole in distance order
                order = np.argsort(h, axis=1, kind="stable")
                h_sorted = np.take_along_axis(h, order, axis=1)
                first_sorted = np.ones_like(h_sorted, dtype=bool)
                first_sorted[:, 1:] = h_sorted[:, 1:] != h_sorted[:, :-1]
                first = np.zeros_like(first_sorted)
                np.put_along_axis(first, order, first_sorted, axis=1)
                seen = np.cumsum(first, axis=1)
                done = np.full(len(rows), k == n)
                done |= (seen[:, -1] >= nth) & (d[:, -1] > radius)
                count[rows[done]] = (first & (d <= radius)).sum(axis=1)[done]
                hit = first & (seen == nth)
                has = hit.any(axis=1) & done
                distance[rows[has]] = d[has, hit.argmax(axis=1)[has]]
                left.append(rows[~done])
            pending = np.concatenate(left)
            k = min(n, 2 * k)
    return count, distance


def add_hole_support(
    model: BlockModel,
    composites: pd.DataFrame,
    *,
    radius: float,
    nth: int = 3,
    count_name: str,
    distance_name: str,
    hole_column: str = "hole_id",
    value_column: str | None = None,
    where: Any = None,
) -> tuple[str, str]:
    """Write :func:`hole_support` of the block centroids as two attributes.

    ``count_name`` (int32, 0 outside ``where``) and ``distance_name`` (float64, NaN
    outside ``where`` or with fewer than ``nth`` holes) are created or overwritten.
    ``where`` is an optional bool mask over the blocks (default all).
    """
    mask = np.ones(model.n_blocks, dtype=bool) if where is None else np.asarray(where, dtype=bool)
    count, distance = hole_support(
        model.centroids()[mask],
        composites,
        radius=radius,
        nth=nth,
        hole_column=hole_column,
        value_column=value_column,
    )
    _write_hole_support(model, mask, count, distance, count_name, distance_name, radius, nth)
    return count_name, distance_name


def _write_hole_support(
    model: BlockModel,
    mask: np.ndarray,
    count: np.ndarray,
    distance: np.ndarray,
    count_name: str,
    distance_name: str,
    radius: float,
    nth: int,
) -> None:
    specs = (
        (count_name, "int", count, f"distinct holes with a composite within {radius:g} m"),
        (distance_name, "float", distance, f"distance to the nearest hole number {nth}"),
    )
    for name, kind, _, _ in specs:
        if name in model and model.attribute(name).kind != kind:
            raise ValueError(f"attribute {name!r} exists and is not {kind}")
    for name, kind, values, description in specs:
        if name in model:
            model.reset(name)
        else:
            model.add_attribute(name, kind, description=description)
        model.set(name, values, mask)


# --------------------------------------------------------------------------- smoothing


def smooth_classes(
    model: BlockModel, codes: np.ndarray, n_classes: int, smoothing: Smoothing
) -> np.ndarray:
    """Majority filter of class codes (-1 = unclassified) on the parent grid.

    Exactly the four steps of the module docstring ("Smoothing"); returns new codes.
    """
    codes = np.array(codes, dtype=np.int64)
    definition = model.definition
    shape = np.asarray(definition.shape, dtype=np.int64)
    volume = model.volume_units().astype(np.float64)
    parent = np.asarray(model.parent, dtype=np.int64)
    offsets = np.array(
        [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)], dtype=np.int64
    )
    for _ in range(smoothing.iterations):
        rows = np.flatnonzero(codes >= 0)
        if rows.size == 0:
            break
        cells, inverse = np.unique(parent[rows], return_inverse=True)
        m = len(cells)
        held = np.bincount(
            inverse * n_classes + codes[rows], weights=volume[rows], minlength=m * n_classes
        ).reshape(m, n_classes)
        # largest volume; on a tie the later class
        cell = n_classes - 1 - np.argmax(held[:, ::-1], axis=1)
        grid = np.full(tuple(shape), -1, dtype=np.int8)
        ijk = definition.unflat_index(cells)
        grid[ijk[:, 0], ijk[:, 1], ijk[:, 2]] = cell
        votes = np.zeros((m, n_classes), dtype=np.int64)
        for offset in offsets:
            q = ijk + offset
            ok = ((q >= 0) & (q < shape)).all(axis=1)
            neighbour = np.full(m, -1, dtype=np.int64)
            neighbour[ok] = grid[q[ok, 0], q[ok, 1], q[ok, 2]]
            has = neighbour >= 0
            np.add.at(votes, (np.flatnonzero(has), neighbour[has]), 1)
        index = np.arange(m)
        same = votes[index, cell]
        others = votes.copy()
        others[index, cell] = -1
        best = n_classes - 1 - np.argmax(others[:, ::-1], axis=1)
        change = (same <= smoothing.max_same) & (others[index, best] >= smoothing.min_agree)
        if not change.any():
            break
        new = np.where(change, best, cell)
        moved = change[inverse]
        codes[rows[moved]] = new[inverse[moved]]
    return codes


# ------------------------------------------------------------------------- classify


def classify(
    model: BlockModel,
    params: ClassificationParams,
    composites: pd.DataFrame | None = None,
) -> ClassificationSummary:
    """Classify the estimated blocks of ``model`` in place; see the module docstring.

    Writes the category attribute ``params.output`` (replaced when it exists; its
    dictionary is the labels in rule order) and, with ``params.hole_support`` and
    ``composites``, the two drillhole-support attributes. Nothing is written when the
    parameters do not fit the model (``ValueError``).
    """
    if params.attribute not in model:
        raise ValueError(f"the model has no attribute {params.attribute!r}")
    if model.attribute(params.attribute).kind != "float":
        raise ValueError(f"{params.attribute!r} is not a float (grade) attribute")
    if params.output in model and model.attribute(params.output).kind != "category":
        raise ValueError(f"attribute {params.output!r} exists and is not a category")
    names = params.input_names()
    used = params.used_inputs()
    estimated = ~np.isnan(np.asarray(model[params.attribute], dtype=np.float64))

    computed: dict[str, np.ndarray] = {}
    if params.hole_support is not None and composites is not None:
        hs = params.hole_support
        for key, kind in (("holes", "int"), ("hole_distance", "float")):
            if names[key] in model and model.attribute(names[key]).kind != kind:
                raise ValueError(f"attribute {names[key]!r} exists and is not {kind}")
        count, distance = hole_support(
            model.centroids()[estimated],
            composites,
            radius=hs.radius,
            nth=hs.nth,
            hole_column=hs.hole_column,
            value_column=hs.value_column,
        )
        computed["holes"] = np.zeros(model.n_blocks, dtype=np.int32)
        computed["holes"][estimated] = count
        computed["hole_distance"] = np.full(model.n_blocks, np.nan)
        computed["hole_distance"][estimated] = distance
    columns: dict[str, np.ndarray] = {}
    for key in used:
        if key in computed:
            columns[key] = computed[key]
        elif names[key] in model:
            if model.attribute(names[key]).kind == "category":
                raise ValueError(f"attribute {names[key]!r} is a category, not a number")
            columns[key] = np.asarray(model[names[key]])
        elif key in _HOLE_KEYS:
            raise ValueError(
                f"the rules use the drillhole support ({names[key]!r}): give hole_support "
                "and the composites, or add the attribute first"
            )
        else:
            raise ValueError(
                f"the rules use {names[key]!r}, which the model does not have "
                f"(was {params.attribute!r} estimated with that output?)"
            )
    conditions = [
        None if rule.where is None else np.asarray(model.evaluate(rule.where)).astype(bool)
        for rule in params.categories
    ]

    codes = np.full(model.n_blocks, -1, dtype=np.int64)
    for number, rule in enumerate(params.categories):
        meets = estimated & (codes < 0)
        for criterion, value in rule.criteria.items():
            key, comparison = CRITERIA[criterion]
            column = columns[key]
            with np.errstate(invalid="ignore"):
                meets &= (column <= value) if comparison == "<=" else (column >= value)
            if key == "pass_number":
                meets &= column >= 1
        if conditions[number] is not None:
            meets &= conditions[number]
        codes[meets] = number
    n_smoothed = 0
    if params.smoothing is not None:
        smoothed = smooth_classes(model, codes, len(params.categories), params.smoothing)
        n_smoothed = int((smoothed != codes).sum())
        codes = smoothed

    written: list[str] = []
    if computed:
        hs = params.hole_support
        _write_hole_support(
            model, estimated, computed["holes"][estimated], computed["hole_distance"][estimated],
            names["holes"], names["hole_distance"], hs.radius, hs.nth,  # type: ignore[union-attr]
        )  # fmt: skip
        written = [names["holes"], names["hole_distance"]]
    description = f"resource class of {params.attribute}"
    if params.output in model:
        # replaced in place, so that the attribute keeps its position in the model
        spec = model.attribute(params.output)
        spec.categories, spec.background, spec.description = params.labels, None, description
    else:
        model.add_attribute(
            params.output, "category", categories=params.labels, description=description
        )
    model[params.output][:] = codes
    volume = model.volumes()
    classes = [
        {
            "label": label,
            "blocks": int((codes == number).sum()),
            "volume": float(volume[codes == number].sum()),
        }
        for number, label in enumerate(params.labels)
    ]
    return ClassificationSummary(
        attribute=params.attribute,
        output=params.output,
        n_blocks=int(model.n_blocks),
        n_estimated=int(estimated.sum()),
        n_classified=int((codes >= 0).sum()),
        n_smoothed=n_smoothed,
        classes=classes,
        written=written,
    )


def read_composites(path: str | Path) -> pd.DataFrame:
    """A composite table written by ``bakumine composite`` (Parquet or CSV)."""
    path = Path(path)
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"hole_id": str})


def classify_model_folder(
    model_path: str | Path,
    params: ClassificationParams,
    composites: str | Path | pd.DataFrame | None = None,
    *,
    out: str | Path | None = None,
) -> ClassificationSummary:
    """:func:`classify` on a saved block model (what the CLI and the API run).

    The model is saved to ``out`` (default: in place) only when classification succeeded.
    """
    from bakumine.core.blockmodel.storage import load_model, save_model

    model = load_model(model_path)
    if composites is not None and not isinstance(composites, pd.DataFrame):
        composites = read_composites(composites)
    summary = classify(model, params, composites)
    save_model(model, out or model_path)
    return summary
