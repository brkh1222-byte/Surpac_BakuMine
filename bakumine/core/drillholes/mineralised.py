"""Mineralised intervals ("economic composites") and significant intercepts from assays.

Given an assay interval table and one grade column, down-hole runs of samples are flagged
as mineralised with the rules below. Everything is a parameter (:class:`InterceptParams`).

Segments
--------
The samples of one hole are sorted by depth. Zero-length rows are dropped; overlapping
samples raise ``ValueError``. Each sample is

* **ore**      ``grade >= cutoff``,
* **waste**    ``grade < cutoff``,
* **missing**  grade is NaN. Depth between two samples that no sample covers is missing
  too. Adjacent missing stretches are joined.

Missing is never treated as zero. A missing stretch longer than ``gap_tolerance`` ends
every run: the hole is cut there into independent pieces. A shorter one (a **gap**) is
bridged: it adds length but no grade and no metal.

Runs (cut-off and internal waste)
---------------------------------
How runs are built is chosen with ``mode``: ``"strict"`` (the default, described first),
``"average"`` or ``"moving-average"`` (section "Other modes" below).

With ``e_i = (g_i - cutoff) * l_i`` the *excess* of a sample (0 for a gap) and ``E`` the sum
of excesses, runs are built down the hole (collar to toe) within one piece:

1. A run starts at the first ore sample that is not in a run and takes the ore samples
   that follow it directly.
2. Let ``W`` be the stretch of waste samples (and gaps) after the run and ``O`` the ore
   samples directly after ``W``. The run is extended by ``W + O`` when

   * the length of ``W`` (waste and the gaps inside it) ``<= max_internal_waste``,
   * the internal waste so far plus ``W`` ``<= max_total_waste`` (when given),
   * ``E(run) + E(W) >= 0``   -- the run still makes the cut-off at the end of the waste,
   * ``E(W) + E(O) >= 0``     -- the ore after the waste carries it.

   A stretch of gaps without any waste sample is not waste: it is bridged without
   conditions. Step 2 repeats until an extension fails; the run is then closed and step 1
   continues with the next ore sample.

Consequences (the property tests check them): a run starts and ends with an ore sample,
its length-weighted grade including the internal waste is ``>= cutoff``, and so is the
grade of every part of it measured from either end -- there is no waste at the edges that
the rest of the run would have to carry. The procedure is greedy down-hole: it does not
search for the grouping with the most metal.

Other modes
-----------
The strict rule lets no part of a run fall below the cut-off when measured from either
end. Common practice is looser; two documented alternatives:

``mode = "average"``
    A run is accepted when it **starts and ends on an ore sample** and its length-weighted
    grade **including the internal waste** is ``>= cutoff`` (``E(run) >= 0``), with every
    internal waste stretch ``<= max_internal_waste`` and the total internal waste
    ``<= max_total_waste`` (when given), measured as in step 2. Down the hole: a run starts
    at the first ore sample that is not in a run and is the **longest** run from there
    that meets these conditions; the search goes on behind its last sample. Unlike
    ``"strict"`` a low-grade edge may be carried by the rest of the run, and a run may pass
    a point where its grade so far is below the cut-off when the ore further down brings
    it back.

    Example (``cutoff 0.2, max_internal_waste 2``, 1 m samples from 0 m):
    ``0.25, 0.1, 0.1, 1.0, 0.05``. Strict: ``E(0-1) + E(1-3) = 0.05 - 0.2 < 0``, so two
    runs, 0-1 m and 3-4 m (2 m). Average: 0-4 m has ``E = 0.05 - 0.2 + 0.8 >= 0``: one
    run of 4 m at ``(0.25 + 0.1 + 0.1 + 1.0) / 4 = 0.3625``.

``mode = "moving-average"`` (needs ``window``, a length in metres)
    Each sample gets the length-weighted mean grade of the assayed length within
    ``window / 2`` up and down the hole from its middle (the window is cut at the ends of
    the piece; gaps carry no grade and no weight). A sample is mineralised when that mean
    is ``>= cutoff``; a run is a maximal stretch of such samples (gaps between two of them
    are bridged). With 1 m samples and ``window = 3`` this is the three-sample moving
    average. ``max_internal_waste`` and ``max_total_waste`` are **not used**: the window
    decides how much waste is carried. A run may start or end on a waste sample next to
    high grade, and its own grade is not guaranteed to reach the cut-off.

    Example (``cutoff 0.2, window 3``, 1 m samples from 0 m): ``0.05, 0.1, 0.9, 0.1,
    0.05, 0.05``. Means: ``0.075`` (window cut at 0 m), ``0.35, 0.3667, 0.35, 0.0667,
    0.05``. One run 1-4 m, 3 m at ``1.1 / 3 = 0.3667``; strict and average give 2-3 m.

The thickness rules, "including" sub-intercepts and the outputs are the same in all modes.

Thickness
---------
``thickness = depth_to - depth_from`` of the run (gaps included). With ``body_dip`` and
``body_dip_azimuth`` (degrees; azimuth of the dip direction) it is the true thickness
``length * |d . n|``, where ``d`` is the unit direction of the desurveyed hole at the
middle of the run and ``n = (sin dip sin az, sin dip cos az, cos dip)`` is the normal of
the body (east, north, up). A run is accepted when

* ``thickness >= min_thickness`` (rule ``"thickness"``), or
* ``accumulation_rule`` is on and ``metal * thickness / length >= cutoff * min_thickness``
  (rule ``"accumulation"``: a short high-grade run), where ``metal = sum(g_i * l_i)`` over
  the assayed samples of the run.

Rejected runs are waste. Runs are not re-grouped after a rejection.

Outputs
-------
``intervals``
    ``hole_id, depth_from, depth_to, <code_column>`` with ``ore_code`` on accepted runs and
    ``waste_code`` on the rest of every piece. Missing stretches longer than the tolerance,
    and gaps at the ends of a piece, have no row (they are not waste). This is an interval
    table like geology: it can be stored in the drillhole database and used as the domain
    table for compositing.
``intercepts``
    One row per accepted run: ``hole_id, depth_from, depth_to, length, true_thickness``
    (NaN without a body orientation), ``grade`` (length-weighted over the assayed length),
    ``grade_cut`` (the same with grades above ``top_cut`` set to it; NaN without a top
    cut), ``grade_x_m`` (``sum(g_i * l_i)``, grade x metres), ``assayed_length``,
    ``waste_length`` (waste samples inside), ``missing_length`` (bridged gaps),
    ``n_samples``, ``cutoff``, ``rule``, ``level`` and ``parent``.
    ``level`` is ``"intercept"``; rows with ``level == "including"`` are sub-intercepts
    found by running the same rules, with a higher cut-off from ``including_cutoffs`` and
    ``including_min_thickness``, on the samples of one intercept only. ``parent`` is the
    row number of that intercept (-1 for an intercept). A sub-intercept equal to its
    whole parent is left out. The top cut is for reporting only: flagging uses the raw
    grades.

Hand-worked example (``cutoff 0.2, min_thickness 2, max_internal_waste 2``), 1 m samples
from 0 m: ``0.1, 0.5, 0.3, 0.1, 0.1, 0.6, 0.05``. Ore at 1-3 (E = +0.4), waste 3-5
(length 2, E = -0.2), ore 5-6 (E = +0.4). ``0.4 - 0.2 >= 0`` and ``-0.2 + 0.4 >= 0``: one
run 1-6 m, 5 m at ``(0.5 + 0.3 + 0.1 + 0.1 + 0.6) / 5 = 0.32``, 1.6 g x m, 2 m internal waste.
"""

from __future__ import annotations

import dataclasses
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.drillholes.desurvey import HoleTraces

__all__ = [
    "INTERCEPT_COLUMNS",
    "MODES",
    "InterceptParams",
    "MineralisedResult",
    "body_normal",
    "mineralised_intervals",
]

FORMAT_VERSION = 1
MODES = ("strict", "average", "moving-average")
_ORE, _WASTE, _GAP = 0, 1, 2
#: absolute tolerance on depths and lengths (the same order as the compositing one)
_TOL = 1e-9

INTERCEPT_COLUMNS = (
    "hole_id",
    "depth_from",
    "depth_to",
    "length",
    "true_thickness",
    "grade",
    "grade_cut",
    "grade_x_m",
    "assayed_length",
    "waste_length",
    "missing_length",
    "n_samples",
    "cutoff",
    "rule",
    "level",
    "parent",
)


@dataclass(frozen=True)
class InterceptParams:
    """Rules of :func:`mineralised_intervals`; see the module docstring.

    cutoff : grade a sample needs to be ore (``>=``).
    mode : ``"strict"`` (default), ``"average"`` or ``"moving-average"``: how runs are built.
    window : length in metres of the moving average (``mode = "moving-average"`` only).
    min_thickness : smallest accepted thickness (0 = any).
    max_internal_waste : longest single stretch of internal waste (0 = none allowed).
    max_total_waste : largest total internal waste of one run, or None (no limit).
    accumulation_rule : accept thinner runs whose metal makes up for it.
    gap_tolerance : longest missing stretch that does not end a run.
    top_cut : grades above it are cut to it in ``grade_cut`` (reporting only), or None.
    including_cutoffs : higher cut-offs for "including" sub-intercepts.
    including_min_thickness : their minimum thickness (default ``min_thickness``).
    body_dip, body_dip_azimuth : orientation of the body for true thickness (both or none).
    ore_code, waste_code, code_column : labels and column name of the interval table.
    """

    cutoff: float
    min_thickness: float = 0.0
    max_internal_waste: float = 0.0
    max_total_waste: float | None = None
    accumulation_rule: bool = False
    gap_tolerance: float = 0.0
    top_cut: float | None = None
    including_cutoffs: tuple[float, ...] = ()
    including_min_thickness: float | None = None
    body_dip: float | None = None
    body_dip_azimuth: float | None = None
    ore_code: str = "MIN"
    waste_code: str = "WASTE"
    code_column: str = "code"
    mode: str = "strict"
    window: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {', '.join(MODES)}, got {self.mode!r}")
        def number(name: str, *, optional: bool = False, minimum: float | None = 0.0) -> None:
            value = getattr(self, name)
            if value is None and optional:
                return
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise ValueError(f"{name} must be a number, got {value!r}")
            value = float(value)
            if not math.isfinite(value) or (minimum is not None and value < minimum):
                raise ValueError(f"{name} must be a finite number >= {minimum}, got {value!r}")
            object.__setattr__(self, name, value)

        number("cutoff", minimum=None)
        number("min_thickness")
        number("max_internal_waste")
        number("max_total_waste", optional=True)
        number("gap_tolerance")
        number("top_cut", optional=True, minimum=None)
        number("including_min_thickness", optional=True)
        number("body_dip", optional=True, minimum=None)
        number("body_dip_azimuth", optional=True, minimum=None)
        number("window", optional=True)
        if self.mode == "moving-average":
            if self.window is None or not self.window > 0:
                raise ValueError("mode moving-average needs a window length in metres (> 0)")
        elif self.window is not None:
            raise ValueError("window is used by mode moving-average only")
        if (self.body_dip is None) != (self.body_dip_azimuth is None):
            raise ValueError("give both body_dip and body_dip_azimuth, or neither")
        cutoffs = self.including_cutoffs
        if isinstance(cutoffs, int | float):
            cutoffs = (cutoffs,)
        cutoffs = tuple(sorted({float(c) for c in cutoffs or ()}))
        if any(not math.isfinite(c) or c <= self.cutoff for c in cutoffs):
            raise ValueError("including_cutoffs must be above the cut-off")
        object.__setattr__(self, "including_cutoffs", cutoffs)
        object.__setattr__(self, "accumulation_rule", bool(self.accumulation_rule))
        for name in ("ore_code", "waste_code", "code_column"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} must be a non-empty string")
        if self.ore_code == self.waste_code:
            raise ValueError("ore_code and waste_code must differ")
        if self.code_column in ("hole_id", "depth_from", "depth_to"):
            raise ValueError(f"code_column cannot be {self.code_column!r}")

    @property
    def true_thickness(self) -> bool:
        return self.body_dip is not None

    def to_dict(self) -> dict[str, Any]:
        data = dataclasses.asdict(self)
        data["including_cutoffs"] = list(self.including_cutoffs)
        return {"bakumine_intercepts": FORMAT_VERSION, **data}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> InterceptParams:
        data = dict(data)
        version = data.pop("bakumine_intercepts", FORMAT_VERSION)
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported intercept parameter format {version!r}")
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(f"unknown intercept parameter(s): {', '.join(unknown)}")
        if "cutoff" not in data:
            raise ValueError("give the cut-off grade")
        if data.get("including_cutoffs") is not None:
            value = data["including_cutoffs"]
            data["including_cutoffs"] = (
                (value,) if isinstance(value, int | float) else tuple(value)
            )
        else:
            data.pop("including_cutoffs", None)
        return cls(**data)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.write_text(json.dumps(self.to_dict(), indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> InterceptParams:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass
class MineralisedResult:
    """``intervals`` (the coded interval table) and ``intercepts``; see the module docstring."""

    intervals: pd.DataFrame
    intercepts: pd.DataFrame
    params: InterceptParams
    grade: str

    def summary(self) -> dict[str, Any]:
        """Counts and totals for logs and dialogs (JSON-safe)."""
        code = self.params.code_column
        length = self.intervals["depth_to"] - self.intervals["depth_from"]
        is_ore = self.intervals[code] == self.params.ore_code
        main = self.intercepts[self.intercepts["level"] == "intercept"]
        metal, assayed = float(main["grade_x_m"].sum()), float(main["assayed_length"].sum())
        return {
            "grade": self.grade,
            "cutoff": self.params.cutoff,
            "mode": self.params.mode,
            "n_holes": int(self.intervals["hole_id"].nunique()),
            "n_holes_mineralised": int(main["hole_id"].nunique()),
            "n_intercepts": int(len(main)),
            "n_including": int(len(self.intercepts) - len(main)),
            "mineralised_length": float(length[is_ore].sum()),
            "waste_length": float(length[~is_ore].sum()),
            "grade_x_m": metal,
            "mean_grade": metal / assayed if assayed > 0 else None,
        }


def body_normal(dip: float, dip_azimuth: float) -> np.ndarray:
    """Unit normal (east, north, up) of a plane dipping ``dip`` degrees towards ``dip_azimuth``."""
    d, a = math.radians(dip), math.radians(dip_azimuth)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


# ----------------------------------------------------------------------------- one hole


@dataclass
class _Run:
    start: int  # first and last segment of the piece (both ore)
    end: int
    rule: str = ""
    factor: float = float("nan")  # true thickness / down-hole length


def _pieces(
    frm: np.ndarray, to: np.ndarray, grade: np.ndarray, gap_tolerance: float, hole: object
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Cut one hole (sorted, positive lengths) into pieces: ``(from, to, grade, assayed)``.

    A piece is a list of touching segments; ``assayed`` is False for a bridged gap (its
    grade is NaN). Gaps at the ends of a piece are removed.
    """
    overlap = np.flatnonzero(frm[1:] < to[:-1] - _TOL)
    if overlap.size:
        i = int(overlap[0])
        raise ValueError(
            f"hole {hole!r}: samples overlap at {frm[i + 1]:g} m "
            f"({frm[i]:g}-{to[i]:g} and {frm[i + 1]:g}-{to[i + 1]:g})"
        )
    # every stretch, sampled or not, in depth order; missing stretches joined
    seg: list[list[float]] = []  # [from, to, grade or NaN]
    for f, t, g in zip(frm.tolist(), to.tolist(), grade.tolist(), strict=True):
        if seg and f > seg[-1][1] + _TOL:
            seg.append([seg[-1][1], f, math.nan])
        seg.append([f, t, g])
    joined: list[list[float]] = []
    for s in seg:
        if joined and math.isnan(s[2]) and math.isnan(joined[-1][2]):
            joined[-1][1] = s[1]
        else:
            joined.append(s)
    pieces, current = [], []
    for s in joined:
        if math.isnan(s[2]) and s[1] - s[0] > gap_tolerance + _TOL:
            pieces.append(current)
            current = []
        else:
            current.append(s)
    pieces.append(current)
    out = []
    for piece in pieces:
        while piece and math.isnan(piece[0][2]):
            piece = piece[1:]
        while piece and math.isnan(piece[-1][2]):
            piece = piece[:-1]
        if piece:
            a = np.array(piece, dtype=np.float64)
            out.append((a[:, 0], a[:, 1], a[:, 2], ~np.isnan(a[:, 2])))
    return out


def _build_runs(
    length: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    max_internal_waste: float,
    max_total_waste: float | None,
) -> list[_Run]:
    """Steps 1 and 2 of the module docstring on the segments of one piece."""
    n = len(length)
    kind = np.where(~assayed, _GAP, np.where(grade >= cutoff, _ORE, _WASTE))
    excess = np.where(assayed, (np.where(assayed, grade, 0.0) - cutoff) * length, 0.0)
    slack = _TOL * (1.0 + abs(cutoff))
    runs: list[_Run] = []
    i = 0
    while i < n:
        if kind[i] != _ORE:
            i += 1
            continue
        start = end = i
        total, waste_total = float(excess[i]), 0.0
        j = i + 1
        while True:
            while j < n and kind[j] == _ORE:
                total += float(excess[j])
                end = j
                j += 1
            k = j
            while k < n and kind[k] != _ORE:
                k += 1
            if k >= n:
                break
            m = k
            while m < n and kind[m] == _ORE:
                m += 1
            e_waste = float(excess[j:k].sum())
            e_ore = float(excess[k:m].sum())
            waste = float(length[j:k].sum()) if (kind[j:k] == _WASTE).any() else 0.0
            if waste > 0.0:
                if waste > max_internal_waste + _TOL:
                    break
                if max_total_waste is not None and waste_total + waste > max_total_waste + _TOL:
                    break
                if total + e_waste < -slack or e_waste + e_ore < -slack:
                    break
            total += e_waste + e_ore
            waste_total += waste
            end = m - 1
            j = m
        runs.append(_Run(start, end))
        i = j
    return runs


def _build_runs_average(
    length: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    max_internal_waste: float,
    max_total_waste: float | None,
) -> list[_Run]:
    """``mode = "average"``: from each start the longest run whose grade makes the cut-off."""
    n = len(length)
    kind = np.where(~assayed, _GAP, np.where(grade >= cutoff, _ORE, _WASTE))
    excess = np.where(assayed, (np.where(assayed, grade, 0.0) - cutoff) * length, 0.0)
    slack = _TOL * (1.0 + abs(cutoff))
    runs: list[_Run] = []
    i = 0
    while i < n:
        if kind[i] != _ORE:
            i += 1
            continue
        end = i
        total, waste_total = 0.0, 0.0
        j = i
        while True:
            while j < n and kind[j] == _ORE:  # an ore stretch: a possible end of the run
                total += float(excess[j])
                j += 1
            if total >= -slack:
                end = j - 1
            k = j
            while k < n and kind[k] != _ORE:
                k += 1
            if k >= n:
                break
            waste = float(length[j:k].sum()) if (kind[j:k] == _WASTE).any() else 0.0
            if waste > max_internal_waste + _TOL:
                break
            if max_total_waste is not None and waste_total + waste > max_total_waste + _TOL:
                break
            total += float(excess[j:k].sum())
            waste_total += waste
            j = k
        runs.append(_Run(i, end))
        i = end + 1
    return runs


def _build_runs_moving_average(
    frm: np.ndarray,
    to: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    window: float,
) -> list[_Run]:
    """``mode = "moving-average"``: samples whose windowed mean grade makes the cut-off."""
    n = len(frm)
    half = 0.5 * window
    slack = _TOL * (1.0 + abs(cutoff))
    flagged = np.zeros(n, dtype=bool)
    rows = np.flatnonzero(assayed)
    a_frm, a_to, a_grade = frm[rows], to[rows], grade[rows]
    for i in rows:
        middle = 0.5 * (frm[i] + to[i])
        overlap = np.minimum(a_to, middle + half) - np.maximum(a_frm, middle - half)
        overlap = np.where(overlap > _TOL, overlap, 0.0)
        covered = float(overlap.sum())
        above = a_grade[overlap > 0.0] >= cutoff
        if above.all() or not above.any():  # exact: no rounding of the mean decides
            flagged[i] = bool(above.all()) and covered > 0.0
        else:
            flagged[i] = float((a_grade * overlap).sum()) / covered >= cutoff - slack
    runs: list[_Run] = []
    i = 0
    while i < n:
        if not flagged[i]:
            i += 1
            continue
        end = j = i
        while j + 1 < n and (flagged[j + 1] or not assayed[j + 1]):
            j += 1
            if flagged[j]:
                end = j
        runs.append(_Run(i, end))
        i = end + 1
    return runs


def _runs_of(
    frm: np.ndarray,
    to: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    params: InterceptParams,
) -> list[_Run]:
    """The runs of one piece under ``params.mode`` with the cut-off ``cutoff``."""
    if params.mode == "moving-average":
        return _build_runs_moving_average(frm, to, grade, assayed, cutoff, params.window)  # type: ignore[arg-type]
    build = _build_runs_average if params.mode == "average" else _build_runs
    return build(
        to - frm, grade, assayed, cutoff, params.max_internal_waste, params.max_total_waste
    )


def _accept(
    runs: list[_Run],
    frm: np.ndarray,
    to: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    min_thickness: float,
    accumulation_rule: bool,
    factor_at: Any,
) -> list[_Run]:
    """The thickness rules; ``factor_at(depth)`` is true thickness per metre down-hole."""
    kept = []
    for run in runs:
        a, b = run.start, run.end + 1
        span = float(to[b - 1] - frm[a])
        run.factor = float(factor_at(0.5 * (frm[a] + to[b - 1]))) if factor_at else float("nan")
        scale = run.factor if factor_at else 1.0
        if span * scale >= min_thickness - _TOL:
            run.rule = "thickness"
        elif accumulation_rule:
            ok = assayed[a:b]
            metal = float((grade[a:b][ok] * (to[a:b] - frm[a:b])[ok]).sum())
            if metal * scale >= cutoff * min_thickness - _TOL * (1.0 + abs(cutoff)):
                run.rule = "accumulation"
        if run.rule:
            kept.append(run)
    return kept


def _row(
    hole: object,
    run: _Run,
    frm: np.ndarray,
    to: np.ndarray,
    grade: np.ndarray,
    assayed: np.ndarray,
    cutoff: float,
    params: InterceptParams,
    level: str,
    parent: int,
) -> dict[str, Any]:
    a, b = run.start, run.end + 1
    length = (to[a:b] - frm[a:b])[assayed[a:b]]
    g = grade[a:b][assayed[a:b]]
    span = float(to[b - 1] - frm[a])
    metal, covered = float((g * length).sum()), float(length.sum())
    cut = float("nan")
    if params.top_cut is not None:
        cut = float((np.minimum(g, params.top_cut) * length).sum()) / covered
    return {
        "hole_id": hole,
        "depth_from": float(frm[a]),
        "depth_to": float(to[b - 1]),
        "length": span,
        "true_thickness": span * run.factor,
        "grade": metal / covered,
        "grade_cut": cut,
        "grade_x_m": metal,
        "assayed_length": covered,
        "waste_length": float(length[g < cutoff].sum()),
        "missing_length": span - covered,
        "n_samples": int(len(g)),
        "cutoff": float(cutoff),
        "rule": run.rule,
        "level": level,
        "parent": parent,
    }


# ----------------------------------------------------------------------------- the table


def _thickness_factor(traces: Any, normal: np.ndarray, hole: object) -> Any:
    def factor_at(depth: float) -> float:
        direction = traces.xyz_at(hole, depth)[1]
        if not np.isfinite(direction).all():
            raise ValueError(
                f"hole {hole!r}: no desurveyed direction at {depth:g} m for true thickness"
            )
        return abs(float(direction @ normal))

    return factor_at


def mineralised_intervals(
    assays: pd.DataFrame,
    grade: str,
    params: InterceptParams,
    *,
    traces: HoleTraces | None = None,
) -> MineralisedResult:
    """Flag mineralised runs in ``assays`` (``hole_id, depth_from, depth_to, <grade>``).

    ``traces`` (desurveyed holes) is needed for true thickness only. The result does not
    depend on the order of the rows: holes come out sorted by id, runs by depth. Rows with
    a missing hole id or depth are ignored; ``depth_to < depth_from`` raises.
    """
    for column in ("hole_id", "depth_from", "depth_to", grade):
        if column not in assays.columns:
            raise ValueError(f"the assay table has no column {column!r}")
    if not pd.api.types.is_numeric_dtype(assays[grade]):
        raise ValueError(f"grade column {grade!r} is not numeric")
    values = assays[grade].to_numpy(dtype=np.float64, na_value=np.nan)
    if params.true_thickness and traces is None:
        raise ValueError("true thickness needs the desurveyed holes (traces)")
    normal = None
    if params.true_thickness:
        normal = body_normal(params.body_dip, params.body_dip_azimuth)  # type: ignore[arg-type]
    table = pd.DataFrame(
        {
            "hole_id": assays["hole_id"].to_numpy(),
            "f": pd.to_numeric(assays["depth_from"], errors="coerce").to_numpy(dtype=np.float64),
            "t": pd.to_numeric(assays["depth_to"], errors="coerce").to_numpy(dtype=np.float64),
            "g": np.where(np.isfinite(values), values, np.nan),
        }
    ).dropna(subset=["hole_id", "f", "t"])
    if (table["t"] < table["f"]).any():
        bad = table[table["t"] < table["f"]].iloc[0]
        raise ValueError(
            f"hole {bad['hole_id']!r}: depth_to {bad['t']:g} < depth_from {bad['f']:g}"
        )
    table = table[table["t"] - table["f"] > _TOL]
    # a stable order that does not depend on the input order (grade breaks exact ties,
    # which are overlaps and raise anyway)
    table = table.sort_values(["hole_id", "f", "t", "g"], kind="stable")

    interval_rows: list[tuple[object, float, float, str]] = []
    rows: list[dict[str, Any]] = []
    inner_min = (
        params.min_thickness
        if params.including_min_thickness is None
        else params.including_min_thickness
    )
    for hole, group in table.groupby("hole_id", sort=True):
        frm_all, to_all, g_all = (group[c].to_numpy() for c in ("f", "t", "g"))
        factor_at = None if normal is None else _thickness_factor(traces, normal, hole)
        for frm, to, g, assayed in _pieces(frm_all, to_all, g_all, params.gap_tolerance, hole):
            runs = _accept(
                _runs_of(frm, to, g, assayed, params.cutoff, params),
                frm, to, g, assayed, params.cutoff, params.min_thickness,
                params.accumulation_rule, factor_at,
            )  # fmt: skip
            cursor = 0
            for run in runs:
                if run.start > cursor:
                    interval_rows.append(
                        (hole, frm[cursor], to[run.start - 1], params.waste_code)
                    )
                interval_rows.append((hole, frm[run.start], to[run.end], params.ore_code))
                cursor = run.end + 1
                parent = len(rows)
                rows.append(
                    _row(hole, run, frm, to, g, assayed, params.cutoff, params, "intercept", -1)
                )
                sl = slice(run.start, run.end + 1)
                for higher in params.including_cutoffs:
                    inner = _accept(
                        _runs_of(frm[sl], to[sl], g[sl], assayed[sl], higher, params),
                        frm[sl], to[sl], g[sl], assayed[sl], higher, inner_min,
                        params.accumulation_rule, factor_at,
                    )  # fmt: skip
                    for sub in inner:
                        if sub.start == 0 and sub.end == run.end - run.start:
                            continue
                        rows.append(
                            _row(hole, sub, frm[sl], to[sl], g[sl], assayed[sl], higher,
                                 params, "including", parent)  # fmt: skip
                        )
            if cursor < len(frm):
                interval_rows.append((hole, frm[cursor], to[-1], params.waste_code))

    intervals = pd.DataFrame(
        interval_rows, columns=["hole_id", "depth_from", "depth_to", params.code_column]
    ).astype({"depth_from": np.float64, "depth_to": np.float64})
    if not len(intervals):
        intervals["hole_id"] = intervals["hole_id"].astype(assays["hole_id"].dtype)
    intercepts = pd.DataFrame(rows, columns=list(INTERCEPT_COLUMNS))
    if not rows:
        intercepts = intercepts.astype(
            {c: np.float64 for c in INTERCEPT_COLUMNS[1:11]} | {"n_samples": int, "parent": int}
        )
    return MineralisedResult(intervals, intercepts, params, grade)


def flag_database(
    database: Any,
    table: str,
    grade: str,
    params: InterceptParams,
    *,
    save_as: str | None = None,
    overwrite: bool = False,
) -> MineralisedResult:
    """Run :func:`mineralised_intervals` on an interval table of a drillhole database.

    With ``save_as`` the coded interval table is added to ``database.intervals`` under
    that name (the caller saves the database). An existing table is replaced only with
    ``overwrite``; the source table itself cannot be replaced.
    """
    from bakumine.core.drillholes.importer import _check_table_name

    if table not in database.intervals:
        raise ValueError(f"no interval table {table!r}; database has {list(database.intervals)}")
    if save_as is not None:
        _check_table_name(save_as)
        if save_as == table:
            raise ValueError("save the intervals under another name than the assay table")
        if save_as in database.intervals and not overwrite:
            raise ValueError(f"the database already has a table {save_as!r} (use overwrite)")
    traces = database.desurvey(method="minimum_curvature") if params.true_thickness else None
    result = mineralised_intervals(database.intervals[table], grade, params, traces=traces)
    if save_as is not None:
        database.intervals[save_as] = result.intervals
    return result


def flag_database_folder(
    folder: str | Path,
    table: str,
    grade: str,
    params: InterceptParams,
    *,
    save_as: str | None = None,
    overwrite: bool = False,
    out: str | Path | None = None,
) -> tuple[MineralisedResult, Path]:
    """:func:`flag_database` on a saved database folder (what the CLI and the API run).

    Writes the intercept table to ``out`` (default
    ``<folder>/intercepts_<save_as or table>.csv``) and, with ``save_as``, saves the
    database with the new interval table. Returns the result and the CSV path.
    """
    from bakumine.core.drillholes.importer import load_database

    folder = Path(folder)
    database = load_database(folder)
    result = flag_database(database, table, grade, params, save_as=save_as, overwrite=overwrite)
    path = Path(out) if out is not None else folder / f"intercepts_{save_as or table}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    result.intercepts.to_csv(path, index=False)
    if save_as is not None:
        database.save(folder)
    return result, path
