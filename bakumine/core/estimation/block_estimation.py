"""Multi-pass estimation of a block-model attribute, per domain (research.md 6.6, 7.6, 7.7).

This module joins the estimators, which work on plain arrays
(:func:`~bakumine.core.estimation.nearest.nearest_neighbour`,
:func:`~bakumine.core.estimation.idw.inverse_distance`,
:func:`~bakumine.core.estimation.kriging.krige`), to the block model
(:class:`~bakumine.core.blockmodel.blocks.BlockModel`, attributes,
constraints). It adds no numerics of its own: for one pass the values written
to the model are exactly what the array function returns for the same
centroids, sizes and parameters.

Domains (hard boundaries)
-------------------------
A domain is a pair: a **block constraint** (which blocks are estimated, e.g.
``rock == "ORE"``) and a **sample filter** (which composites may be used, e.g.
``domain == "ORE"``). Blocks of the constraint only ever see samples passing
the filter, so no grade crosses the boundary. :func:`estimate_attribute` runs
one such pair; :func:`estimate_domains` runs several in sequence. Soft
boundaries (letting samples within some distance of the contact take part in
the neighbouring domain) are out of scope.

Passes
------
An :class:`EstimationPass` is a method (``"nn"``, ``"idw"``, ``"ok"``,
``"sk"``), a search ellipsoid and search options, and the method's own
parameters. Passes run in the order given. A pass estimates only blocks of
the constraint that have no value yet; a block estimated in pass ``p`` is
never touched by a later pass. The pass number (1-based) is stored next to
the estimate. ``radius_factor`` scales the search ellipsoid, so the usual
plan "same ellipsoid x 1, x 2, x 3 with relaxed minimum samples" is three
passes that differ in two fields (:meth:`EstimationPass.scaled`).

Which blocks are estimated
--------------------------
``overwrite=False`` (default): blocks of the constraint whose target
attribute is missing (NaN). Existing estimates, and their diagnostics, are
kept. ``overwrite=True``: every block of the constraint; all output
attributes of those blocks are reset to their background first, so a block
that no pass can estimate ends up missing even if it had a value before.

Block support
-------------
``support="block"``
    Every block, parent or sub-block, is estimated at its own centroid with
    its own size.
``support="parent"``
    Each parent block that has at least one block to estimate is estimated
    once, at the parent centroid with the parent size, and the value (and its
    diagnostics) is written to all of that parent's blocks that are to be
    estimated: "estimate parents, assign to children". This is the usual
    practice and the recommended one for a sub-blocked model: sub-blocks
    exist to follow a boundary, not to carry grade detail; kriging each tiny
    sub-block separately costs many times more and only adds noise at a
    support the data cannot resolve.
``support="auto"`` (default)
    ``"parent"`` when the model has sub-blocks, else ``"block"`` (for a model
    without sub-blocks the two are the same thing).

The model rotation is passed to the estimators as ``grid_rotation``, so the
discretisation points of a rotated model's blocks are where the blocks are.
Nearest neighbour has no block support: it uses the centroid.

Outputs
-------
Attributes are named from the target attribute by suffix (:class:`OutputNames`;
``au`` gives ``au``, ``au_pass``, ``au_ns``, ``au_dist``, ``au_ndist``,
``au_kv``, ``au_slope``, ``au_ke`` by default) and created when absent:

====================  =======  =======  ======================================
key                   suffix   type     content
====================  =======  =======  ======================================
(estimate)            (none)   float64  the estimate; NaN = not estimated
``pass_number``       _pass    int8     1-based pass; 0 = not estimated
``n_samples``         _ns      int32    samples in the neighbourhood; 0 = none
``mean_distance``     _dist    float64  mean anisotropic distance; NaN = none
``nearest_distance``  _ndist   float64  smallest anisotropic distance
``variance``          _kv      float64  kriging variance
``slope``             _slope   float64  slope of regression
``efficiency``        _ke      float64  kriging efficiency
====================  =======  =======  ======================================

Integer attributes have no missing marker in the block model, so the pass
number and the sample count use the background 0 for "not estimated".
The last three are only created when the run has a kriging pass; blocks
estimated by a non-kriging pass keep NaN there. Distances are anisotropic
distances of the pass's search ellipsoid (research.md 7.5).

Atomicity
---------
Nothing is written to the model until every pass of every domain has been
computed. If the ``cancel`` hook returns true (it is polled between chunks
of blocks), or an exception is raised, the model is exactly as before the
call: no attribute created, no value changed.

Parameter files
---------------
:class:`EstimationPlan` (attribute, options, domains with their passes) is
JSON-serialisable (:meth:`EstimationPlan.save` / :meth:`EstimationPlan.load`):
ellipsoids through ``Ellipsoid.to_dict``, variograms through the variogram
JSON schema, block constraints through the constraint JSON (geometry by name,
resolved by the ``resolver`` given at run time). Example::

    {
      "bakumine_estimation": 1,
      "attribute": "au",
      "value_column": "au",
      "support": "auto",
      "domains": [
        {
          "name": "ORE",
          "block_constraint": {"type": "attribute", "expression": "rock == \\"ORE\\""},
          "sample_filter": {"column": "domain", "equals": "ORE"},
          "capping": 30.0,
          "passes": [
            {
              "method": "ok",
              "ellipsoid": {"major_radius": 60.0, "semi_major_ratio": 1.7,
                            "minor_ratio": 8.0, "convention": "surpac_zxy_lrl",
                            "angles": [35.0, 10.0, 50.0]},
              "radius_factor": 1.0,
              "search": {"max_samples": 16, "min_samples": 6, "max_per_hole": 4},
              "variogram": {"schema": "...", "nugget": 0.15, "structures": [...]},
              "discretisation": [4, 4, 2]
            }
          ]
        }
      ]
    }
"""

from __future__ import annotations

import dataclasses
import json
import math
import reprlib
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.attributes import check_name
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import Constraint, constraint_from_dict
from bakumine.core.drillholes.capping import cap_grades
from bakumine.core.estimation.dynamic_anisotropy import DynamicAnisotropy, run_dynamic
from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import VariogramModel

__all__ = [
    "FORMAT_VERSION",
    "METHODS",
    "SUPPORTS",
    "ColumnFilter",
    "DomainSummary",
    "EstimationDomain",
    "EstimationPass",
    "EstimationPlan",
    "EstimationSummary",
    "OutputNames",
    "PassSummary",
    "estimate_attribute",
    "estimate_domains",
    "load_passes",
    "save_passes",
]

FORMAT_VERSION = 1

#: estimation methods of a pass
METHODS = ("nn", "idw", "ok", "sk")
_KRIGING = {"ok": "ordinary", "sk": "simple"}

#: block-support modes
SUPPORTS = ("auto", "parent", "block")

#: default block discretisation of IDW and kriging passes
DEFAULT_DISCRETISATION = (4, 4, 2)

_MAX_PASSES = 127  # pass numbers are stored as int8
_NN_RADIUS = 1.0e30  # "no limit" radius of the declustering nearest-neighbour run

Progress = Callable[[float, str], None]
Cancel = Callable[[], bool]


# --------------------------------------------------------------------------
# passes
# --------------------------------------------------------------------------


def _search_to_dict(params: SearchParams) -> dict[str, Any]:
    default = SearchParams()
    out: dict[str, Any] = {"max_samples": params.max_samples, "min_samples": params.min_samples}
    for f in dataclasses.fields(SearchParams):
        value = getattr(params, f.name)
        if f.name not in out and value != getattr(default, f.name):
            out[f.name] = value
    return out


def _search_from_dict(data: Mapping[str, Any]) -> SearchParams:
    known = {f.name for f in dataclasses.fields(SearchParams)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ValueError(f"unknown search option(s) {unknown}; known: {sorted(known)}")
    return SearchParams(**dict(data))


@dataclass(frozen=True)
class EstimationPass:
    """One estimation pass: method, search and method parameters.

    Parameters
    ----------
    method : ``"nn"`` (nearest neighbour), ``"idw"`` (inverse distance),
        ``"ok"`` (ordinary kriging) or ``"sk"`` (simple kriging).
    ellipsoid : search ellipsoid (orientation, major radius, ratios). For IDW
        it is also the anisotropy of the weights.
    search : :class:`~bakumine.core.estimation.search.SearchParams`: minimum
        / maximum samples, maximum per hole, sectors, maximum vertical
        distance. Nearest neighbour uses only ``radius`` and
        ``max_vertical`` of these.
    radius_factor : multiplier of the search radius: the pass searches
        ``ellipsoid`` with its major radius (and ``search.radius`` when that
        is set) times this factor; ratios and orientation are unchanged.
    power : IDW exponent.
    variogram : variogram model, required for kriging.
    mean : the known mean, required for simple kriging.
    discretisation : ``(nx, ny, nz)`` points per block for IDW and kriging;
        ``None`` = 4 x 4 x 2. ``(1, 1, 1)`` is a point estimate at the
        centroid. Ignored by nearest neighbour.
    negative_weight_correction : kriging option (see
        :func:`~bakumine.core.estimation.kriging.krige`).
    name : free label used in progress messages and the summary.
    dynamic : optional
        :class:`~bakumine.core.estimation.dynamic_anisotropy.DynamicAnisotropy`:
        per-block angles (and ratios) read from block attributes re-orient the
        search ellipsoid and, for kriging, the variogram of each block (see
        that module). ``None`` (default): the fixed ``ellipsoid``.
    """

    method: str
    ellipsoid: Ellipsoid
    search: SearchParams = field(default_factory=SearchParams)
    radius_factor: float = 1.0
    power: float = 2.0
    variogram: VariogramModel | None = None
    mean: float | None = None
    discretisation: tuple[int, int, int] | None = None
    negative_weight_correction: bool = False
    name: str = ""
    dynamic: DynamicAnisotropy | None = None

    def __post_init__(self) -> None:
        if isinstance(self.dynamic, Mapping):
            object.__setattr__(self, "dynamic", DynamicAnisotropy.from_dict(self.dynamic))
        if self.dynamic is not None and not isinstance(self.dynamic, DynamicAnisotropy):
            raise TypeError("dynamic must be a DynamicAnisotropy or None")
        if self.method not in METHODS:
            raise ValueError(f"method must be one of {METHODS}, got {self.method!r}")
        if not isinstance(self.ellipsoid, Ellipsoid):
            raise TypeError("ellipsoid must be an Ellipsoid")
        if not isinstance(self.search, SearchParams):
            raise TypeError("search must be a SearchParams")
        factor = float(self.radius_factor)
        if not (math.isfinite(factor) and factor > 0.0):
            raise ValueError("radius_factor must be a positive finite number")
        object.__setattr__(self, "radius_factor", factor)
        power = float(self.power)
        if not (math.isfinite(power) and power > 0.0):
            raise ValueError("power must be a positive finite number")
        object.__setattr__(self, "power", power)
        if self.method in _KRIGING:
            if not isinstance(self.variogram, VariogramModel):
                raise ValueError(f"a {self.method!r} pass needs a variogram model")
            if self.method == "sk":
                if self.mean is None or not math.isfinite(float(self.mean)):
                    raise ValueError("a simple-kriging pass needs a finite mean")
                object.__setattr__(self, "mean", float(self.mean))
        elif self.variogram is not None and not isinstance(self.variogram, VariogramModel):
            raise TypeError("variogram must be a VariogramModel")
        if self.discretisation is not None:
            try:
                disc = tuple(int(v) for v in self.discretisation)
            except (TypeError, ValueError):
                raise ValueError("discretisation must be three integers (nx, ny, nz)") from None
            if len(disc) != 3 or min(disc) < 1:
                raise ValueError("discretisation must be three integers >= 1")
            object.__setattr__(self, "discretisation", disc)

    # ---------------------------------------------------------- derived values

    @property
    def is_kriging(self) -> bool:
        return self.method in _KRIGING

    @property
    def search_ellipsoid(self) -> Ellipsoid:
        """The ellipsoid actually searched: ``ellipsoid`` scaled by ``radius_factor``."""
        if self.radius_factor == 1.0:
            return self.ellipsoid
        return dataclasses.replace(
            self.ellipsoid, major_radius=self.ellipsoid.major_radius * self.radius_factor
        )

    @property
    def search_params(self) -> SearchParams:
        """The search options actually used (an explicit radius scaled by ``radius_factor``)."""
        if self.radius_factor == 1.0 or self.search.radius is None:
            return self.search
        return dataclasses.replace(self.search, radius=self.search.radius * self.radius_factor)

    @property
    def block_discretisation(self) -> tuple[int, int, int]:
        """Discretisation used: the given one, 4 x 4 x 2 by default, a point for NN."""
        if self.method == "nn":
            return (1, 1, 1)
        return DEFAULT_DISCRETISATION if self.discretisation is None else self.discretisation

    @property
    def label(self) -> str:
        return self.name or self.method

    def scaled(self, radius_factor: float, **changes: Any) -> EstimationPass:
        """A copy with another ``radius_factor`` and any other fields changed.

        ``search_min_samples`` / ``search_max_samples`` / ... change single
        fields of ``search``::

            p1 = EstimationPass("ok", ellipsoid, SearchParams(16, 6), variogram=model)
            p2 = p1.scaled(2.0, search_min_samples=3)
            p3 = p1.scaled(3.0, search_min_samples=1)
        """
        search_changes = {
            key[len("search_") :]: changes.pop(key)
            for key in list(changes)
            if key.startswith("search_")
        }
        if search_changes:
            if "search" in changes:
                raise TypeError("give either search or search_<field> changes, not both")
            changes["search"] = dataclasses.replace(self.search, **search_changes)
        return dataclasses.replace(self, radius_factor=radius_factor, **changes)

    # ------------------------------------------------------------ serialisation

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; inverse of :meth:`from_dict`."""
        out: dict[str, Any] = {}
        if self.name:
            out["name"] = self.name
        out["method"] = self.method
        out["ellipsoid"] = self.ellipsoid.to_dict()
        out["radius_factor"] = self.radius_factor
        out["search"] = _search_to_dict(self.search)
        if self.method == "idw" or self.power != 2.0:
            out["power"] = self.power
        if self.variogram is not None:
            out["variogram"] = self.variogram.to_dict()
        if self.mean is not None:
            out["mean"] = self.mean
        if self.discretisation is not None:
            out["discretisation"] = list(self.discretisation)
        if self.negative_weight_correction:
            out["negative_weight_correction"] = True
        if self.dynamic is not None:
            out["dynamic"] = self.dynamic.to_dict()
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationPass:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown pass field(s) {unknown}; known: {sorted(known)}")
        for key in ("method", "ellipsoid"):
            if key not in data:
                raise ValueError(f"a pass needs {key!r}")
        variogram = data.get("variogram")
        disc = data.get("discretisation")
        return cls(
            method=data["method"],
            ellipsoid=Ellipsoid.from_dict(data["ellipsoid"]),
            search=_search_from_dict(data.get("search", {})),
            radius_factor=data.get("radius_factor", 1.0),
            power=data.get("power", 2.0),
            variogram=None if variogram is None else VariogramModel.from_dict(variogram),
            mean=data.get("mean"),
            discretisation=None if disc is None else tuple(disc),
            negative_weight_correction=bool(data.get("negative_weight_correction", False)),
            name=data.get("name", ""),
            dynamic=data.get("dynamic"),
        )


def _as_passes(passes: Any) -> list[EstimationPass]:
    if isinstance(passes, EstimationPass | Mapping):
        passes = [passes]
    out = [p if isinstance(p, EstimationPass) else EstimationPass.from_dict(p) for p in passes]
    if not out:
        raise ValueError("at least one estimation pass is needed")
    if len(out) > _MAX_PASSES:
        raise ValueError(f"at most {_MAX_PASSES} passes are supported")
    return out


def save_passes(passes: Iterable[EstimationPass], path: str | Path) -> None:
    """Write a list of passes as JSON."""
    document = {
        "bakumine_estimation_passes": FORMAT_VERSION,
        "passes": [p.to_dict() for p in _as_passes(list(passes))],
    }
    Path(path).write_text(json.dumps(document, indent=2), encoding="utf-8")


def load_passes(path: str | Path) -> list[EstimationPass]:
    """Read a list of passes written by :func:`save_passes`."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or "passes" not in document:
        raise ValueError(f"{path} is not a BakuMine estimation-pass file")
    if document.get("bakumine_estimation_passes") != FORMAT_VERSION:
        raise ValueError(
            f"{path}: unsupported format {document.get('bakumine_estimation_passes')!r}"
        )
    return _as_passes(document["passes"])


# --------------------------------------------------------------------------
# sample filters
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ColumnFilter:
    """Samples whose ``column`` equals one of ``values`` (the serialisable sample filter)."""

    column: str
    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.column, str) or not self.column:
            raise ValueError("a sample filter needs a column name")
        values = self.values
        if isinstance(values, str) or not isinstance(values, Iterable):
            values = (values,)
        values = tuple(v.item() if isinstance(v, np.generic) else v for v in values)
        if not values:
            raise ValueError("a sample filter needs at least one value")
        nested = [v for v in values if isinstance(v, (list, tuple, set, dict))]
        if nested:  # one cell never equals a list: the filter would keep nothing
            raise ValueError(
                f"a sample filter value is one value, not {reprlib.repr(nested[0])}; "
                "several values are given as {'in': [...]}"
            )
        object.__setattr__(self, "values", values)

    def mask(self, table: pd.DataFrame) -> np.ndarray:
        if self.column not in table.columns:
            raise ValueError(f"sample filter: column {self.column!r} is not in the sample table")
        return table[self.column].isin(list(self.values)).to_numpy(dtype=bool)

    def to_dict(self) -> dict[str, Any]:
        if len(self.values) == 1:
            return {"column": self.column, "equals": self.values[0]}
        return {"column": self.column, "in": list(self.values)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ColumnFilter:
        if "column" not in data or ("equals" in data) == ("in" in data):
            raise ValueError("a sample filter is {'column': name, 'equals': value | 'in': [...]}")
        if set(data) - {"column", "equals", "in"}:
            raise ValueError(f"unknown sample-filter field(s) in {dict(data)}")
        if "equals" in data:
            values: tuple[Any, ...] = (data["equals"],)
        else:
            within = data["in"]
            if isinstance(within, (str, bytes, Mapping)) or not isinstance(within, Iterable):
                # "in": "MIN" would be the letters M, I, N
                raise ValueError(f"sample filter 'in' takes a list of values, got {within!r}")
            values = tuple(within)
        return cls(data["column"], values)


def _as_filter(sample_filter: Any) -> Any:
    """Normalise: None, ColumnFilter, a callable or a boolean array."""
    if sample_filter is None or isinstance(sample_filter, ColumnFilter) or callable(sample_filter):
        return sample_filter
    if isinstance(sample_filter, Mapping):
        return ColumnFilter.from_dict(sample_filter)
    if (
        isinstance(sample_filter, tuple)
        and len(sample_filter) == 2
        and isinstance(sample_filter[0], str)
    ):
        return ColumnFilter(sample_filter[0], sample_filter[1])
    array = np.asarray(sample_filter)
    if array.dtype != bool or array.ndim != 1:
        raise TypeError(
            "sample_filter must be None, (column, value), {'column':..., 'equals':...}, "
            "a ColumnFilter, a boolean array with one entry per sample, or a callable"
        )
    return array


def _filter_mask(table: pd.DataFrame, sample_filter: Any) -> np.ndarray:
    if sample_filter is None:
        return np.ones(len(table), dtype=bool)
    if isinstance(sample_filter, ColumnFilter):
        return sample_filter.mask(table)
    mask = sample_filter(table) if callable(sample_filter) else sample_filter
    mask = np.asarray(mask)
    if mask.dtype != bool or mask.shape != (len(table),):
        raise ValueError("a sample filter must give one boolean per row of the sample table")
    return mask


# --------------------------------------------------------------------------
# output names
# --------------------------------------------------------------------------

_OUTPUT_KEYS = (
    "pass_number",
    "n_samples",
    "mean_distance",
    "nearest_distance",
    "variance",
    "slope",
    "efficiency",
)
_KRIGING_KEYS = ("variance", "slope", "efficiency")
# key -> (attribute kind, dtype, description)
_OUTPUT_SPEC = {
    "pass_number": ("int", np.int8, "estimation pass (0 = not estimated)"),
    "n_samples": ("int", np.int32, "number of samples used"),
    "mean_distance": ("float", np.float64, "mean anisotropic distance to the samples"),
    "nearest_distance": ("float", np.float64, "anisotropic distance to the nearest sample"),
    "variance": ("float", np.float64, "kriging variance"),
    "slope": ("float", np.float64, "slope of regression"),
    "efficiency": ("float", np.float64, "kriging efficiency"),
}


@dataclass(frozen=True)
class OutputNames:
    """Which diagnostics are written, and the suffix of each attribute name.

    A field is the suffix appended to the target attribute name, or ``None``
    to leave that output out. ``OutputNames()`` writes everything;
    :meth:`only` selects by key.
    """

    pass_number: str | None = "_pass"
    n_samples: str | None = "_ns"
    mean_distance: str | None = "_dist"
    nearest_distance: str | None = "_ndist"
    variance: str | None = "_kv"
    slope: str | None = "_slope"
    efficiency: str | None = "_ke"

    def __post_init__(self) -> None:
        used = [s for s in dataclasses.astuple(self) if s is not None]
        if any(not isinstance(s, str) or not s for s in used):
            raise ValueError("an output suffix must be a non-empty string or None")
        if len(set(used)) != len(used):
            raise ValueError("output suffixes must be distinct")

    @classmethod
    def only(cls, *keys: str) -> OutputNames:
        """Default suffixes for the given keys, nothing else (``only()`` = estimate alone)."""
        unknown = sorted(set(keys) - set(_OUTPUT_KEYS))
        if unknown:
            raise ValueError(f"unknown output(s) {unknown}; known: {list(_OUTPUT_KEYS)}")
        default = cls()
        return cls(**{k: (getattr(default, k) if k in keys else None) for k in _OUTPUT_KEYS})

    def names(self, attribute: str) -> dict[str, str]:
        """``{key: attribute name}`` of the outputs that are switched on."""
        return {
            key: attribute + getattr(self, key)
            for key in _OUTPUT_KEYS
            if getattr(self, key) is not None
        }

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OutputNames:
        unknown = sorted(set(data) - set(_OUTPUT_KEYS))
        if unknown:
            raise ValueError(f"unknown output(s) {unknown}; known: {list(_OUTPUT_KEYS)}")
        return cls(**dict(data))


def _as_outputs(outputs: Any) -> OutputNames:
    if outputs is None:
        return OutputNames()
    if isinstance(outputs, OutputNames):
        return outputs
    if isinstance(outputs, Mapping):
        return OutputNames.from_dict(outputs)
    if isinstance(outputs, str):
        outputs = (outputs,)
    return OutputNames.only(*outputs)


# --------------------------------------------------------------------------
# summary
# --------------------------------------------------------------------------


def _json_value(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {k: _json_value(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_value(v) for v in value]
    return value


@dataclass
class PassSummary:
    """What one pass did in one domain.

    ``n_targeted`` blocks were still without a value when the pass started;
    ``n_estimated`` of them got one, ``n_unestimated`` did not (and go to
    the next pass). With parent support ``n_units_*`` count the parent
    blocks actually estimated; with block support they equal the block
    counts. ``mean_estimate`` and ``nn_mean`` are volume-weighted over the
    blocks estimated in this pass; ``nn_mean`` is the nearest-neighbour
    (polygonal-declustered) mean of the samples over the same blocks, NaN
    when the check was switched off. ``n_negative`` counts negative
    estimates (before clipping, if clipping was requested).
    ``mean_samples`` is the mean number of samples per estimated block.
    """

    number: int
    name: str
    method: str
    n_targeted: int = 0
    n_estimated: int = 0
    n_unestimated: int = 0
    n_units_targeted: int = 0
    n_units_estimated: int = 0
    mean_samples: float = math.nan
    mean_estimate: float = math.nan
    min_estimate: float = math.nan
    max_estimate: float = math.nan
    nn_mean: float = math.nan
    n_negative: int = 0
    seconds: float = 0.0


@dataclass
class DomainSummary:
    """Result of one domain (block constraint + sample filter).

    ``n_constraint`` blocks pass the constraint; ``n_targeted`` of them were
    to be estimated (all with ``overwrite``, else those missing a value);
    ``n_estimated`` got a value in some pass and ``n_unestimated`` did not.
    ``n_samples`` samples passed the filter and had a finite value
    (``n_samples_missing`` passed the filter without one); ``n_capped`` of
    them were cut to the cap. ``sample_mean / min / max`` are the plain
    statistics of the values used (after capping). ``mean_estimate`` and
    ``nn_mean`` are volume-weighted over the blocks estimated;
    ``estimate_vs_nn`` is their relative difference ``(est - nn) / nn``.
    """

    name: str
    support: str
    n_constraint: int = 0
    n_targeted: int = 0
    n_estimated: int = 0
    n_unestimated: int = 0
    n_samples: int = 0
    n_samples_missing: int = 0
    n_capped: int = 0
    sample_mean: float = math.nan
    sample_min: float = math.nan
    sample_max: float = math.nan
    mean_estimate: float = math.nan
    min_estimate: float = math.nan
    max_estimate: float = math.nan
    nn_mean: float = math.nan
    estimate_vs_nn: float = math.nan
    n_negative: int = 0
    seconds: float = 0.0
    passes: list[PassSummary] = field(default_factory=list)


@dataclass
class EstimationSummary:
    """Result of :func:`estimate_attribute` / :func:`estimate_domains`.

    ``cancelled`` true means the run was stopped by the cancel hook and the
    model was not changed (``domains`` then holds what had been computed).
    ``outputs`` maps output keys (``"estimate"``, ``"pass_number"``, ...) to
    the attribute names written.
    """

    attribute: str
    outputs: dict[str, str] = field(default_factory=dict)
    domains: list[DomainSummary] = field(default_factory=list)
    cancelled: bool = False
    clip_negative: bool = False
    seconds: float = 0.0

    @property
    def n_targeted(self) -> int:
        return sum(d.n_targeted for d in self.domains)

    @property
    def n_estimated(self) -> int:
        return sum(d.n_estimated for d in self.domains)

    @property
    def n_unestimated(self) -> int:
        return sum(d.n_unestimated for d in self.domains)

    def domain(self, name: str) -> DomainSummary:
        for d in self.domains:
            if d.name == name:
                return d
        raise KeyError(name)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (NaN written as ``None``)."""
        return _json_value(dataclasses.asdict(self))

    def to_frame(self) -> pd.DataFrame:
        """One row per domain and pass."""
        rows = []
        for d in self.domains:
            for p in d.passes:
                rows.append({"domain": d.name, **dataclasses.asdict(p)})
        return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# domains and plans
# --------------------------------------------------------------------------


@dataclass
class EstimationDomain:
    """A named domain: which blocks, which samples, which passes.

    Parameters
    ----------
    name : label (summary, progress messages).
    passes : list of :class:`EstimationPass` (or their dicts).
    block_constraint : a :class:`~bakumine.core.blockmodel.constraints.Constraint`,
        a boolean mask over the block table, or ``None`` (all blocks).
    sample_filter : ``None`` (all samples), ``(column, value)``,
        ``{"column": name, "equals": value}`` / ``{"column": name, "in":
        [...]}``, a :class:`ColumnFilter`, a boolean array with one entry per
        row of the sample table, or a callable ``table -> boolean array``.
    capping : optional top-cut of this domain's sample values (research.md
        7.7), applied before estimation; overrides the run-level ``capping``.
    """

    name: str
    passes: list[EstimationPass]
    block_constraint: Any = None
    sample_filter: Any = None
    capping: float | None = None

    def __post_init__(self) -> None:
        self.name = str(self.name)
        self.passes = _as_passes(self.passes)
        self.sample_filter = _as_filter(self.sample_filter)
        if isinstance(self.block_constraint, Mapping):
            self.block_constraint = constraint_from_dict(self.block_constraint)
        if self.capping is not None:
            self.capping = float(self.capping)

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict. Masks and callables cannot be saved (``TypeError``)."""
        out: dict[str, Any] = {"name": self.name}
        if self.block_constraint is not None:
            if not isinstance(self.block_constraint, Constraint):
                raise TypeError(
                    f"domain {self.name!r}: only a Constraint can be saved as block_constraint"
                )
            out["block_constraint"] = self.block_constraint.to_dict()
        if self.sample_filter is not None:
            if not isinstance(self.sample_filter, ColumnFilter):
                raise TypeError(
                    f"domain {self.name!r}: only a column == value sample filter can be saved"
                )
            out["sample_filter"] = self.sample_filter.to_dict()
        if self.capping is not None:
            out["capping"] = self.capping
        out["passes"] = [p.to_dict() for p in self.passes]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationDomain:
        known = {"name", "passes", "block_constraint", "sample_filter", "capping"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown domain field(s) {unknown}; known: {sorted(known)}")
        if "passes" not in data:
            raise ValueError("a domain needs 'passes'")
        return cls(
            name=data.get("name", ""),
            passes=data["passes"],
            block_constraint=data.get("block_constraint"),
            sample_filter=data.get("sample_filter"),
            capping=data.get("capping"),
        )


def _as_domain(domain: Any) -> EstimationDomain:
    if isinstance(domain, EstimationDomain):
        return domain
    if isinstance(domain, Mapping):
        return EstimationDomain.from_dict(domain)
    raise TypeError("a domain must be an EstimationDomain or a dict")


@dataclass
class EstimationPlan:
    """A complete, reproducible estimation run: the content of a parameter file.

    Fields are the arguments of :func:`estimate_domains`. ``capping`` here is
    a single top-cut for every domain that does not set its own.
    """

    attribute: str
    domains: list[EstimationDomain]
    value_column: str | None = None
    support: str = "auto"
    outputs: OutputNames = field(default_factory=OutputNames)
    overwrite: bool = False
    clip_negative: bool = False
    capping: float | None = None
    declustered_check: bool = True

    def __post_init__(self) -> None:
        check_name(self.attribute)
        self.domains = [_as_domain(d) for d in self.domains]
        if not self.domains:
            raise ValueError("a plan needs at least one domain")
        if self.support not in SUPPORTS:
            raise ValueError(f"support must be one of {SUPPORTS}, got {self.support!r}")
        self.outputs = _as_outputs(self.outputs)
        if self.capping is not None:
            self.capping = float(self.capping)

    def run(
        self,
        model: BlockModel,
        samples: pd.DataFrame,
        *,
        resolver: Any = None,
        chunk_size: int | None = None,
        progress: Progress | None = None,
        cancel: Cancel | None = None,
    ) -> EstimationSummary:
        """Run the plan on ``model`` with the composite table ``samples``."""
        options: dict[str, Any] = {} if chunk_size is None else {"chunk_size": chunk_size}
        return estimate_domains(
            model,
            samples,
            attribute=self.attribute,
            value_column=self.value_column,
            domains=self.domains,
            resolver=resolver,
            outputs=self.outputs,
            overwrite=self.overwrite,
            support=self.support,
            clip_negative=self.clip_negative,
            capping=self.capping,
            declustered_check=self.declustered_check,
            progress=progress,
            cancel=cancel,
            **options,
        )

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "bakumine_estimation": FORMAT_VERSION,
            "attribute": self.attribute,
            "value_column": self.value_column or self.attribute,
            "support": self.support,
            "outputs": self.outputs.to_dict(),
            "overwrite": self.overwrite,
            "clip_negative": self.clip_negative,
            "declustered_check": self.declustered_check,
        }
        if self.capping is not None:
            out["capping"] = self.capping
        out["domains"] = [d.to_dict() for d in self.domains]
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EstimationPlan:
        if not isinstance(data, Mapping) or "domains" not in data or "attribute" not in data:
            raise ValueError("an estimation plan needs 'attribute' and 'domains'")
        version = data.get("bakumine_estimation", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported estimation-plan format {version!r}")
        known = {f.name for f in dataclasses.fields(cls)} | {"bakumine_estimation"}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown plan field(s) {unknown}; known: {sorted(known)}")
        return cls(
            attribute=data["attribute"],
            domains=list(data["domains"]),
            value_column=data.get("value_column"),
            support=data.get("support", "auto"),
            outputs=_as_outputs(data.get("outputs")),
            overwrite=bool(data.get("overwrite", False)),
            clip_negative=bool(data.get("clip_negative", False)),
            capping=data.get("capping"),
            declustered_check=bool(data.get("declustered_check", True)),
        )

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> EstimationPlan:
        return cls.from_dict(json.loads(text))

    def save(self, path: str | Path) -> None:
        """Write the plan as a JSON parameter file."""
        Path(path).write_text(self.to_json(indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> EstimationPlan:
        """Read a parameter file written by :meth:`save`."""
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


# --------------------------------------------------------------------------
# internals
# --------------------------------------------------------------------------


class _Cancelled(Exception):
    pass


@dataclass
class _Samples:
    xyz: np.ndarray
    values: np.ndarray
    holes: np.ndarray | None
    n_missing: int
    n_capped: int


def _prepare_samples(
    table: pd.DataFrame,
    value_column: str,
    sample_filter: Any,
    capping: Any,
    capping_domain: str | None,
) -> _Samples:
    mask = _filter_mask(table, sample_filter)
    sub = table.loc[mask]
    n_capped = 0
    if capping is not None:
        sub, report = cap_grades(sub, {value_column: capping}, domain=capping_domain)
        n_capped = int(report["n_cut"].sum())
    values = sub[value_column].to_numpy(dtype=np.float64)
    xyz = sub[["x", "y", "z"]].to_numpy(dtype=np.float64)
    holes = None
    if "hole_id" in sub.columns:
        holes = pd.factorize(sub["hole_id"])[0].astype(np.int64)
    ok = np.isfinite(values)
    return _Samples(
        np.ascontiguousarray(xyz[ok]),
        np.ascontiguousarray(values[ok]),
        None if holes is None else holes[ok],
        int((~ok).sum()),
        n_capped,
    )


def _check_table(table: Any, value_column: str) -> None:
    if not isinstance(table, pd.DataFrame):
        raise TypeError("samples must be a pandas DataFrame with x, y, z and the value column")
    missing = [c for c in ("x", "y", "z", value_column) if c not in table.columns]
    if missing:
        raise ValueError(f"the sample table has no column(s) {missing}")


def _run_estimator(
    p: EstimationPass, smp: _Samples, xyz: np.ndarray, sizes: Any, grid: np.ndarray
) -> dict[str, np.ndarray]:
    """Call the array estimator of pass ``p`` for one chunk of targets."""
    ellipsoid = p.search_ellipsoid
    params = p.search_params
    if p.method == "nn":
        r = nearest_neighbour(
            smp.xyz,
            smp.values,
            xyz,
            ellipsoid,
            radius=params.radius,
            max_vertical=params.max_vertical,
        )
    elif p.method == "idw":
        r = inverse_distance(
            smp.xyz,
            smp.values,
            xyz,
            ellipsoid,
            params,
            power=p.power,
            hole_ids=smp.holes,
            block_size=sizes,
            discretisation=p.block_discretisation,
            grid_rotation=grid,
        )
    else:
        r = krige(
            smp.xyz,
            smp.values,
            xyz,
            p.variogram,
            ellipsoid,
            params,
            method=_KRIGING[p.method],
            mean=p.mean if p.method == "sk" else None,
            hole_ids=smp.holes,
            block_size=sizes,
            discretisation=p.block_discretisation,
            grid_rotation=grid,
            negative_weight_correction=p.negative_weight_correction,
        )
    out = {
        "estimate": r.estimate,
        "ok": r.estimated & np.isfinite(r.estimate),
        "n_samples": r.n_samples,
        "mean_distance": r.mean_distance,
        "nearest_distance": r.nearest_distance,
    }
    if p.is_kriging:
        out["variance"] = r.variance
        out["slope"] = r.slope
        out["efficiency"] = r.efficiency
    return out


def _run_dynamic_estimator(
    p: EstimationPass,
    smp: _Samples,
    xyz: np.ndarray,
    sizes: Any,
    grid: np.ndarray,
    units: _Units,
    sel: np.ndarray,
) -> dict[str, np.ndarray]:
    """:func:`_run_estimator` with per-block orientation (dynamic anisotropy)."""
    assert p.dynamic is not None
    params = p.dynamic.block_parameters(units.model, units.representative_rows(sel), p.ellipsoid)
    per_target = np.ndim(sizes) == 2

    def run(fixed: EstimationPass, idx: np.ndarray) -> dict[str, np.ndarray]:
        return _run_estimator(fixed, smp, xyz[idx], sizes[idx] if per_target else sizes, grid)

    return run_dynamic(p, params, run)


class _Units:
    """The things estimated: the target blocks themselves, or their parents.

    ``rows`` are the block-table rows to fill; unit ``u`` supplies the value
    of the rows with ``unit_of_row == u`` (``None``: unit i is row i).
    Geometry is produced per chunk, so nothing of size (n, 3) is held for the
    whole run.
    """

    def __init__(self, model: BlockModel, rows: np.ndarray, support: str):
        self.model = model
        self.rows = rows
        self.parent_support = support == "parent"
        if self.parent_support:
            self.parents, inverse = np.unique(model.parent[rows], return_inverse=True)
            self.unit_of_row = inverse.reshape(-1).astype(np.int64)
            self.n = len(self.parents)
            self.volume = np.bincount(
                self.unit_of_row, weights=model.volumes(rows), minlength=self.n
            )
        else:
            self.unit_of_row = None
            self.n = len(rows)
            self.volume = model.volumes(rows)

    def geometry(self, units: np.ndarray) -> tuple[np.ndarray, Any]:
        """World centroids (m, 3) and sizes ((3,) when all equal, else (m, 3))."""
        d = self.model.definition
        if self.parent_support:
            xyz = d.ijk_to_world(d.unflat_index(self.parents[units]))
            return xyz, np.array(d.block_size, dtype=np.float64)
        rows = self.rows[units]
        sizes = self.model.sizes(rows)
        if len(sizes) and (sizes == sizes[0]).all():
            sizes = sizes[0].copy()
        return self.model.centroids(rows), sizes

    def representative_rows(self, units: np.ndarray) -> np.ndarray:
        """One block-table row per unit: the block itself, or the parent's
        first targeted block (used to read per-block dynamic anisotropy)."""
        if not self.parent_support:
            return self.rows[units]
        if not hasattr(self, "_first_row"):
            _, first = np.unique(self.model.parent[self.rows], return_index=True)
            self._first_row = self.rows[first]
        return self._first_row[units]

    def to_rows(self, per_unit: np.ndarray) -> np.ndarray:
        return per_unit if self.unit_of_row is None else per_unit[self.unit_of_row]

    def row_count(self, unit_mask: np.ndarray) -> int:
        if self.unit_of_row is None:
            return int(unit_mask.sum())
        return int(unit_mask[self.unit_of_row].sum())


@dataclass
class _Pending:
    """Computed, not yet written, result of one domain."""

    units: _Units
    buffers: dict[str, np.ndarray]  # per unit; "estimate" NaN where unestimated
    summary: DomainSummary


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0.0 else math.nan


def _compute_domain(
    model: BlockModel,
    table: pd.DataFrame,
    domain: EstimationDomain,
    *,
    attribute: str,
    value_column: str,
    resolver: Any,
    keys: set[str],
    overwrite: bool,
    support: str,
    clip_negative: bool,
    capping: Any,
    capping_domain: str | None,
    declustered_check: bool,
    chunk_size: int,
    taken: np.ndarray | None,
    report: Callable[[float, str], None],
    cancel: Cancel | None,
) -> _Pending:
    t_domain = time.perf_counter()
    label = f"{domain.name}: " if domain.name else ""

    def check_cancel() -> None:
        if cancel is not None and cancel():
            raise _Cancelled

    # ---- blocks
    constraint = domain.block_constraint
    if constraint is None:
        mask = np.ones(model.n_blocks, dtype=bool)
    elif isinstance(constraint, Constraint):
        mask = np.asarray(constraint.mask(model, resolver), dtype=bool)
    else:
        mask = np.asarray(constraint)
        if mask.dtype != bool or mask.shape != (model.n_blocks,):
            raise ValueError(
                "block_constraint must be a Constraint or a boolean mask over the block table"
            )
    n_constraint = int(mask.sum())
    target = mask.copy()
    if not overwrite and attribute in model:
        target &= model.is_missing(attribute)
    if taken is not None:
        target &= ~taken
    rows = np.flatnonzero(target)
    units = _Units(model, rows, support)
    summary = DomainSummary(
        name=domain.name, support=support, n_constraint=n_constraint, n_targeted=len(rows)
    )

    # ---- samples
    cap = domain.capping if domain.capping is not None else capping
    smp = _prepare_samples(table, value_column, domain.sample_filter, cap, capping_domain)
    summary.n_samples = len(smp.values)
    summary.n_samples_missing = smp.n_missing
    summary.n_capped = smp.n_capped
    if len(smp.values):
        summary.sample_mean = float(smp.values.mean())
        summary.sample_min = float(smp.values.min())
        summary.sample_max = float(smp.values.max())
    if any(p.search.max_per_hole is not None for p in domain.passes) and smp.holes is None:
        raise ValueError("max_per_hole needs a 'hole_id' column in the sample table")
    for p in domain.passes:
        if p.dynamic is not None:
            p.dynamic.check_model(model)

    # ---- buffers (per unit)
    n = units.n
    buffers: dict[str, np.ndarray] = {"estimate": np.full(n, np.nan)}
    for key in _OUTPUT_KEYS:
        if key in keys:
            _, dtype, _ = _OUTPUT_SPEC[key]
            fill = 0 if np.issubdtype(dtype, np.integer) else np.nan
            buffers[key] = np.full(n, fill, dtype=dtype)
    pass_of_unit = np.zeros(n, dtype=np.int8)
    grid = np.array(model.definition.rotation)
    share = 0.95 if declustered_check else 1.0
    n_passes = len(domain.passes)

    pending = np.arange(n, dtype=np.int64)
    pass_summaries: list[PassSummary] = []
    for number, p in enumerate(domain.passes, start=1):
        t_pass = time.perf_counter()
        ps = PassSummary(number=number, name=p.label, method=p.method)
        ps.n_units_targeted = len(pending)
        done_mask = np.zeros(len(pending), dtype=bool)
        message = f"{label}pass {number}/{n_passes} ({p.label})"
        base = share * (number - 1) / n_passes
        report(base, message)
        for start in range(0, len(pending), chunk_size):
            check_cancel()
            sel = pending[start : start + chunk_size]
            if len(smp.values):
                xyz, sizes = units.geometry(sel)
                if p.dynamic is None:
                    res = _run_estimator(p, smp, xyz, sizes, grid)
                else:
                    res = _run_dynamic_estimator(p, smp, xyz, sizes, grid, units, sel)
                ok = res["ok"]
                hit = sel[ok]
                buffers["estimate"][hit] = res["estimate"][ok]
                for key, buffer in buffers.items():
                    if key in res and key != "estimate":
                        buffer[hit] = res[key][ok]
                pass_of_unit[hit] = number
                done_mask[start : start + len(sel)] = ok
            report(base + share / n_passes * min(1.0, (start + len(sel)) / len(pending)), message)
        estimated_units = pending[done_mask]
        pending = pending[~done_mask]
        ps.n_units_estimated = len(estimated_units)
        ps.seconds = time.perf_counter() - t_pass
        pass_summaries.append(ps)
    check_cancel()

    # ---- negatives
    estimate = buffers["estimate"]
    done = pass_of_unit > 0
    negative = done & (estimate < 0.0)
    if clip_negative:
        estimate[negative] = 0.0

    # ---- declustered (nearest-neighbour) mean over the estimated blocks
    nn = np.full(n, np.nan)
    if declustered_check and len(smp.values) and done.any():
        todo = np.flatnonzero(done)
        ellipsoid = domain.passes[0].ellipsoid
        for start in range(0, len(todo), chunk_size):
            check_cancel()
            sel = todo[start : start + chunk_size]
            xyz, _ = units.geometry(sel)
            nn[sel] = nearest_neighbour(
                smp.xyz, smp.values, xyz, ellipsoid, radius=_NN_RADIUS
            ).estimate
            report(
                share + (1.0 - share) * (start + len(sel)) / len(todo),
                f"{label}declustered mean",
            )

    # ---- statistics
    volume = units.volume
    remaining = len(rows)
    for ps in pass_summaries:
        in_pass = pass_of_unit == ps.number
        ps.n_targeted = remaining
        ps.n_estimated = units.row_count(in_pass)
        remaining -= ps.n_estimated
        ps.n_unestimated = remaining
        ps.n_negative = units.row_count(in_pass & negative)
        if in_pass.any():
            w = volume[in_pass]
            ps.mean_estimate = _weighted_mean(estimate[in_pass], w)
            ps.min_estimate = float(estimate[in_pass].min())
            ps.max_estimate = float(estimate[in_pass].max())
            if "n_samples" in buffers:
                ps.mean_samples = _weighted_mean(buffers["n_samples"][in_pass].astype(float), w)
            has_nn = in_pass & np.isfinite(nn)
            if has_nn.any():
                ps.nn_mean = _weighted_mean(nn[has_nn], volume[has_nn])
    summary.passes = pass_summaries
    summary.n_estimated = units.row_count(done)
    summary.n_unestimated = len(rows) - summary.n_estimated
    summary.n_negative = units.row_count(negative)
    if done.any():
        summary.mean_estimate = _weighted_mean(estimate[done], volume[done])
        summary.min_estimate = float(estimate[done].min())
        summary.max_estimate = float(estimate[done].max())
        has_nn = done & np.isfinite(nn)
        if has_nn.any():
            summary.nn_mean = _weighted_mean(nn[has_nn], volume[has_nn])
            if summary.nn_mean != 0.0:
                summary.estimate_vs_nn = (
                    _weighted_mean(estimate[has_nn], volume[has_nn]) - summary.nn_mean
                ) / summary.nn_mean
    if "pass_number" in buffers:
        buffers["pass_number"] = pass_of_unit
    buffers["_done"] = done
    summary.seconds = time.perf_counter() - t_domain
    return _Pending(units, buffers, summary)


def _check_outputs(model: BlockModel, attribute: str, names: dict[str, str]) -> None:
    """Fail before any work if an output attribute cannot be created or has the wrong type."""
    check_name(attribute)
    if attribute in model:
        spec = model.attribute(attribute)
        if spec.kind != "float":
            raise ValueError(f"attribute {attribute!r} is {spec.kind}; estimates need a float")
    for key, name in names.items():
        kind = _OUTPUT_SPEC[key][0]
        if name in model:
            if model.attribute(name).kind != kind:
                raise ValueError(
                    f"output attribute {name!r} exists and is {model.attribute(name).kind}; "
                    f"the {key} needs kind {kind!r}"
                )
        else:
            check_name(name)


def _commit(
    model: BlockModel,
    attribute: str,
    names: dict[str, str],
    pendings: list[_Pending],
    overwrite: bool,
) -> None:
    if attribute not in model:
        model.add_attribute(attribute, "float")
    for key, name in names.items():
        if name not in model:
            kind, dtype, description = _OUTPUT_SPEC[key]
            model.add_attribute(
                name, kind, dtype=dtype, description=f"{attribute}: {description}"
            )
    for pending in pendings:
        units, buffers = pending.units, pending.buffers
        rows = units.rows
        if len(rows) == 0:
            continue
        done = units.to_rows(buffers["_done"])
        if overwrite:
            model.reset(attribute, rows)
            for name in names.values():
                model.reset(name, rows)
        hit = rows[done]
        model.set(attribute, units.to_rows(buffers["estimate"])[done], hit)
        for key, name in names.items():
            values = units.to_rows(buffers[key])[done]
            column = model[name]
            if column.dtype.kind in "iu":
                info = np.iinfo(column.dtype)
                values = np.clip(values, info.min, info.max)
            column[hit] = values


def _resolve_support(model: BlockModel, support: str) -> str:
    if support not in SUPPORTS:
        raise ValueError(f"support must be one of {SUPPORTS}, got {support!r}")
    if support == "auto":
        return "block" if model.is_dense else "parent"
    return support


# --------------------------------------------------------------------------
# public functions
# --------------------------------------------------------------------------


def estimate_domains(
    model: BlockModel,
    samples: pd.DataFrame,
    *,
    attribute: str,
    domains: Sequence[EstimationDomain | Mapping[str, Any]],
    value_column: str | None = None,
    resolver: Any = None,
    outputs: Any = None,
    overwrite: bool = False,
    support: str = "auto",
    clip_negative: bool = False,
    capping: Any = None,
    capping_domain: str | None = None,
    declustered_check: bool = True,
    chunk_size: int = 250_000,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
) -> EstimationSummary:
    """Estimate ``attribute`` in several domains, one after the other.

    Each domain (:class:`EstimationDomain` or a dict with ``name``,
    ``passes`` and optionally ``block_constraint``, ``sample_filter``,
    ``capping``) is estimated from its own samples only: hard boundaries.
    The other parameters are those of :func:`estimate_attribute`.

    All block constraints are evaluated on the model as it is before the
    call, and nothing is written until every domain is computed (see the
    module docstring: a cancelled or failed run leaves the model unchanged).
    If constraints overlap, a block estimated by an earlier domain is not
    estimated again by a later one.
    """
    t0 = time.perf_counter()
    if not isinstance(model, BlockModel):
        raise TypeError("model must be a BlockModel")
    domain_list = [_as_domain(d) for d in domains]
    if not domain_list:
        raise ValueError("at least one domain is needed")
    value_column = attribute if value_column is None else value_column
    _check_table(samples, value_column)
    if int(chunk_size) < 1:
        raise ValueError("chunk_size must be >= 1")
    chunk_size = int(chunk_size)
    out_names = _as_outputs(outputs)
    names = out_names.names(attribute)
    if not any(p.is_kriging for d in domain_list for p in d.passes):
        names = {k: v for k, v in names.items() if k not in _KRIGING_KEYS}
    _check_outputs(model, attribute, names)
    support = _resolve_support(model, support)

    last = [0.0]

    def emit(fraction: float, message: str) -> None:
        if progress is None:
            return
        fraction = min(1.0, max(last[0], float(fraction)))
        last[0] = fraction
        progress(fraction, message)

    summary = EstimationSummary(
        attribute=attribute,
        outputs={"estimate": attribute, **names},
        clip_negative=bool(clip_negative),
    )
    pendings: list[_Pending] = []
    taken = np.zeros(model.n_blocks, dtype=bool) if len(domain_list) > 1 else None
    n_domains = len(domain_list)
    try:
        for index, domain in enumerate(domain_list):

            def report(fraction: float, message: str, index: int = index) -> None:
                # keep 1.0 for the moment the model has been written
                emit(min((index + fraction) / n_domains, 0.999), message)

            pending = _compute_domain(
                model,
                samples,
                domain,
                attribute=attribute,
                value_column=value_column,
                resolver=resolver,
                keys=set(names),
                overwrite=overwrite,
                support=support,
                clip_negative=clip_negative,
                capping=capping,
                capping_domain=capping_domain,
                declustered_check=declustered_check,
                chunk_size=chunk_size,
                taken=taken,
                report=report,
                cancel=cancel,
            )
            pendings.append(pending)
            summary.domains.append(pending.summary)
            if taken is not None:
                rows = pending.units.rows
                taken[rows[pending.units.to_rows(pending.buffers["_done"])]] = True
        if cancel is not None and cancel():
            raise _Cancelled
    except _Cancelled:
        summary.cancelled = True
        summary.seconds = time.perf_counter() - t0
        return summary
    _commit(model, attribute, names, pendings, overwrite)
    summary.seconds = time.perf_counter() - t0
    emit(1.0, "done")
    return summary


def estimate_attribute(
    model: BlockModel,
    samples: pd.DataFrame,
    *,
    attribute: str,
    passes: Sequence[EstimationPass | Mapping[str, Any]] | EstimationPass,
    value_column: str | None = None,
    block_constraint: Any = None,
    sample_filter: Any = None,
    resolver: Any = None,
    outputs: Any = None,
    overwrite: bool = False,
    support: str = "auto",
    clip_negative: bool = False,
    capping: Any = None,
    capping_domain: str | None = None,
    declustered_check: bool = True,
    chunk_size: int = 250_000,
    progress: Progress | None = None,
    cancel: Cancel | None = None,
    name: str = "",
) -> EstimationSummary:
    """Estimate a block-model attribute in one domain with a list of passes.

    Parameters
    ----------
    model : the block model; the results are written to it.
    samples : composite table (``pandas.DataFrame``) with columns ``x``,
        ``y``, ``z`` (east, north, up), the value column and, when
        ``max_per_hole`` is used, ``hole_id``. Rows without a finite value
        are left out (missing stays missing, never zero).
    attribute : target attribute; created (float, background NaN) if absent.
    passes : :class:`EstimationPass` objects (or their dicts), run in order.
    value_column : column of ``samples`` holding the values; default: the
        attribute name.
    block_constraint : the blocks of the domain: a
        :class:`~bakumine.core.blockmodel.constraints.Constraint` (evaluated
        with ``resolver``), a boolean mask over the block table, or ``None``
        for all blocks.
    sample_filter : the samples of the domain: ``(column, value)``,
        ``{"column": name, "equals": value}``, a boolean array, a callable
        ``table -> boolean array``, or ``None`` for all samples. Pairing a
        block constraint with a sample filter is how a domain is honoured
        (hard boundary; soft boundaries are not supported).
    resolver : geometry resolver for constraints that name a DTM / solid.
    outputs : which diagnostics to write and how to name them: ``None`` (all,
        default suffixes), an :class:`OutputNames`, a dict ``{key: suffix or
        None}`` or an iterable of keys. See the module docstring.
    overwrite : estimate every block of the constraint (resetting all
        outputs there first) instead of only the blocks without a value.
    support : ``"auto"`` (default), ``"parent"`` or ``"block"``; see the
        module docstring. Parent support is recommended for sub-blocked
        models and is what ``"auto"`` selects for them.
    clip_negative : set negative estimates (possible with kriging) to 0.
        Off by default; negatives are counted in the summary either way.
    capping : optional top-cut applied to the value column before estimation
        (:func:`bakumine.core.drillholes.capping.cap_grades`): a number, or
        a mapping ``{domain code: cap}`` with ``capping_domain`` naming the
        code column. The caller's table is not changed.
    declustered_check : also run an unlimited-radius nearest-neighbour
        estimate (first pass's anisotropy) over the estimated blocks and
        report its mean, the declustered sample mean, next to the mean
        estimate. Costs one extra KD-tree query per block.
    chunk_size : blocks (parent blocks with parent support) handed to the
        estimator at a time. Bounds memory and sets how often ``cancel`` is
        polled and ``progress`` called; results do not depend on it.
    progress : ``progress(fraction, message)``; fractions never decrease and
        the last call, after the model has been written, is ``1.0``.
    cancel : ``cancel() -> bool``, polled before every chunk. When it returns
        true the function returns a summary with ``cancelled=True`` and the
        model is exactly as it was.
    name : label of the domain in the summary.

    Returns
    -------
    :class:`EstimationSummary` with one :class:`DomainSummary`.
    """
    domain = EstimationDomain(
        name=name,
        passes=passes,  # type: ignore[arg-type]
        block_constraint=block_constraint,
        sample_filter=sample_filter,
    )
    return estimate_domains(
        model,
        samples,
        attribute=attribute,
        domains=[domain],
        value_column=value_column,
        resolver=resolver,
        outputs=outputs,
        overwrite=overwrite,
        support=support,
        clip_negative=clip_negative,
        capping=capping,
        capping_domain=capping_domain,
        declustered_check=declustered_check,
        chunk_size=chunk_size,
        progress=progress,
        cancel=cancel,
    )
