"""Compositing: re-cut the samples of every hole into intervals of equal support.

Theory: ``research.md`` section 4. One interval table (``hole_id, depth_from, depth_to`` +
numeric attributes) goes in; one composite table comes out, with columns

``hole_id, depth_from, depth_to, length``  the composite and its real downhole length;
``x, y, z``                                its midpoint on the desurveyed arc;
``dx, dy, dz``                             unit direction of the hole at the midpoint;
``domain``                                 only when a domain source is given;
``<attribute>``                            the composited value of every attribute;
``<attribute>_coverage``                   the sampled length that supports that value.

Where the composite boundaries are
----------------------------------
A *run* is a stretch of one hole inside which composites are formed. Without domains the
run is the whole hole from the collar (depth 0) to the end of its deepest sample. With a
domain source a run is a maximal stretch of consecutive segments with the same domain code,
so no composite ever crosses a domain contact.

* ``mode="downhole"`` - boundaries every ``length`` from the start of the run; the last
  composite of a run ends at the run's end and may be short.
* ``mode="domain"``   - the same, and a domain source is required.
* ``mode="bench"``    - boundaries where the desurveyed path crosses the horizontal levels
  ``bench_reference + k * bench_height``. The crossing depths are solved on the path itself
  (``HoleTraces.xyz_at``), so they are right for curved holes.
* ``best_fit=True``   - (downhole / domain) each run is divided into the whole number of
  equal composites whose length is nearest to ``length``; there is no short remnant.

How a value is computed
-----------------------
Length-weighted mean over the part of each sample inside the composite,
``g = sum(g_i * l_i) / sum(l_i)``; with ``density="col"`` the weight is ``l_i * rho_i``.
A NaN sample and an unsampled gap contribute neither grade nor length to that attribute
(missing is never zero), unless ``fill_missing`` gives an explicit value.
``<attribute>_coverage`` is ``sum(l_i)`` over the samples that did contribute, so
``value * coverage`` is the metal accumulation, which compositing conserves.

Domain sources (pluggable)
--------------------------
``domain`` may be

* a column name of the interval table;
* an array with one code per row of the interval table;
* a callable ``f(intervals) -> codes`` (one per row). A later "inside solid X" test only
  has to supply such a callable (or the interval table below); this module does not change;
* :class:`IntervalDomains` - another interval table with a categorical column (geology
  ``rock``). The samples are first split at its contacts with ``merge_intervals``
  (:func:`split_at_domains`).

Everything works on whole arrays for all holes at once.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from .desurvey import HoleTraces
from .importer import DERIVED_INTERVAL_COLUMNS, INTERVAL_COLUMNS
from .intervals import merge_intervals

__all__ = [
    "MODES",
    "SHORT_RULES",
    "DomainSource",
    "IntervalDomains",
    "bench_crossings",
    "composite",
    "coverage_column",
    "split_at_domains",
]

Mode = Literal["downhole", "bench", "domain"]
ShortRule = Literal["keep", "discard", "merge"]
OnInvalid = Literal["raise", "drop"]

MODES: tuple[str, ...] = ("downhole", "bench", "domain")
SHORT_RULES: tuple[str, ...] = ("keep", "discard", "merge")

#: Depths closer than this (relative to the composite length) are one boundary.
_REL_TOL = 1e-9
_COVERAGE_SUFFIX = "_coverage"


@dataclass(frozen=True)
class IntervalDomains:
    """Domain codes from a categorical column of another interval table (e.g. geology)."""

    table: pd.DataFrame
    column: str


DomainSource = str | IntervalDomains | Callable[[pd.DataFrame], object] | Sequence | np.ndarray


def coverage_column(attribute: str) -> str:
    """Name of the column holding the sampled length that supports ``attribute``."""
    return f"{attribute}{_COVERAGE_SUFFIX}"


# --------------------------------------------------------------------------------------
# domains
# --------------------------------------------------------------------------------------
def split_at_domains(
    intervals: pd.DataFrame,
    domains: IntervalDomains,
    *,
    name: str = "domain",
    on_invalid: OnInvalid = "raise",
) -> pd.DataFrame:
    """Split the samples at every domain contact and give each piece its domain code.

    Returns ``hole_id, depth_from, depth_to``, the columns of ``intervals`` (unchanged
    values: a split sample keeps its grade on both sides) and the column ``name``. A
    stretch covered by the domain table only has missing sample attributes; a stretch
    covered by the samples only has a missing domain.
    """
    if domains.column not in domains.table.columns:
        raise ValueError(f"domain table has no column {domains.column!r}")
    if name in intervals.columns:
        raise ValueError(f"the interval table already has a column {name!r}")
    merged = merge_intervals(
        {
            "sample": intervals[list(INTERVAL_COLUMNS)],
            "domain": domains.table[list(INTERVAL_COLUMNS)],
        },
        on_invalid=on_invalid,
        source_rows=True,
    )
    sample_row = merged["sample_row"].fillna(-1).to_numpy(dtype=np.int64)
    domain_row = merged["domain_row"].fillna(-1).to_numpy(dtype=np.int64)
    out = merged[list(INTERVAL_COLUMNS)].copy()
    attributes = intervals.drop(columns=list(INTERVAL_COLUMNS)).reset_index(drop=True)
    taken = attributes.reindex(sample_row).reset_index(drop=True)  # -1 -> missing
    for column in taken.columns:
        out[column] = taken[column]
    codes = domains.table[domains.column].reset_index(drop=True)
    out[name] = codes.reindex(domain_row).reset_index(drop=True)
    return out


def _domain_values(intervals: pd.DataFrame, domain: object) -> pd.Series:
    """One domain code per row of ``intervals`` from a column name, callable or array."""
    if isinstance(domain, str):
        if domain not in intervals.columns:
            raise ValueError(
                f"domain column {domain!r} is not in the interval table; "
                f"has {list(intervals.columns)}"
            )
        return intervals[domain].reset_index(drop=True)
    values = domain(intervals) if callable(domain) else domain
    series = pd.Series(values).reset_index(drop=True)
    if len(series) != len(intervals):
        raise ValueError(
            f"the domain source gave {len(series)} codes for {len(intervals)} intervals"
        )
    return series


def _locate_midpoints(
    traces: HoleTraces, ids: np.ndarray, middle: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Midpoint coordinates and directions. A composite of samples that go deeper than
    the stated hole depth has its midpoint past the end of the trace: it is located on
    the straight continuation of the hole (direction at the end), with a warning, rather
    than left without coordinates (the estimation refuses such samples)."""
    xyz, direction = traces.xyz_at(ids, middle)
    end = pd.Series(traces.max_depth, index=traces.hole_ids).reindex(ids).to_numpy(np.float64)
    with np.errstate(invalid="ignore"):
        past = ~np.isfinite(xyz).all(axis=1) & (middle > end)
    if past.any():
        xyz, direction = xyz.copy(), direction.copy()
        end_xyz, end_dir = traces.xyz_at(ids[past], end[past])
        over = middle[past] - end[past]
        xyz[past] = end_xyz + over[:, None] * end_dir
        direction[past] = end_dir
        holes = sorted({str(h) for h in ids[past]})
        warnings.warn(
            f"{int(past.sum())} composite(s) in {len(holes)} hole(s) lie below the stated "
            f"hole depth (by up to {float(over.max()):g}; {', '.join(holes[:6])}"
            f"{' ...' if len(holes) > 6 else ''}): located on the straight continuation "
            "of the hole. Check the hole depth of the collar table",
            stacklevel=3,
        )
    return xyz, direction


# --------------------------------------------------------------------------------------
# bench levels on the desurveyed path
# --------------------------------------------------------------------------------------
def _path_z(traces: HoleTraces, ids: np.ndarray, depth: np.ndarray) -> tuple[np.ndarray, ...]:
    xyz, direction = traces.xyz_at(ids, depth)
    return xyz[:, 2], direction[:, 2]


def bench_crossings(
    traces: HoleTraces, bench_height: float, bench_reference: float = 0.0
) -> pd.DataFrame:
    """Depths at which every hole path crosses the levels ``reference + k * height``.

    Returns ``hole_id, depth, level`` ordered by hole and depth. The path between two
    stations is first cut where it turns from descending to ascending (or back), so that
    the elevation is monotone on each piece; each crossing is then solved on the path with
    a bracketed Newton iteration (the derivative of z along the hole is the z component of
    the direction). A hole may cross the same level more than once.
    """
    if not (np.isfinite(bench_height) and bench_height > 0):
        raise ValueError(f"bench_height must be > 0, got {bench_height!r}")
    if not np.isfinite(bench_reference):
        raise ValueError(f"bench_reference must be finite, got {bench_reference!r}")
    stations = traces.stations()
    empty = pd.DataFrame({"hole_id": np.empty(0, dtype=object), "depth": [], "level": []})
    if len(stations) < 2:
        return empty
    ids = np.asarray(traces.hole_ids, dtype=object)
    hole = pd.Index(ids).get_indexer(stations["hole_id"].to_numpy(dtype=object))
    depth = stations["depth"].to_numpy(dtype=np.float64)
    z = stations["z"].to_numpy(dtype=np.float64)
    dz = stations["dz"].to_numpy(dtype=np.float64)
    start = np.flatnonzero((hole[1:] == hole[:-1]) & (depth[1:] > depth[:-1]))
    ph, d0, d1 = hole[start], depth[start], depth[start + 1]
    z0, z1, g0, g1 = z[start], z[start + 1], dz[start], dz[start + 1]

    # Cut at the turning point (direction z changes sign at most once per interval).
    turn = np.flatnonzero(g0 * g1 < 0.0)
    if turn.size:
        lo, hi = d0[turn].copy(), d1[turn].copy()
        turn_ids, sign0 = ids[ph[turn]], np.sign(g0[turn])
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            _, g = _path_z(traces, turn_ids, mid)
            before = np.sign(g) == sign0
            lo, hi = np.where(before, mid, lo), np.where(before, hi, mid)
        t = 0.5 * (lo + hi)
        zt, _ = _path_z(traces, turn_ids, t)
        ph = np.concatenate((ph, ph[turn]))
        d0 = np.concatenate((d0, t))
        z0 = np.concatenate((z0, zt))
        d1 = np.concatenate((d1, d1[turn]))
        z1 = np.concatenate((z1, z1[turn]))
        d1[turn], z1[turn] = t, zt

    # Levels crossed by each monotone piece: open at its upper end, closed at its lower end.
    u0, u1 = (z0 - bench_reference) / bench_height, (z1 - bench_reference) / bench_height
    down = z1 < z0
    k_first = np.where(down, np.ceil(u1), np.floor(u0) + 1.0)
    k_last = np.where(down, np.ceil(u0) - 1.0, np.floor(u1))
    count = np.maximum(k_last - k_first + 1.0, 0.0).astype(np.int64)
    total = int(count.sum())
    if total == 0:
        return empty
    piece = np.repeat(np.arange(count.size), count)
    within = np.arange(total) - np.repeat(np.cumsum(count) - count, count)
    level = bench_reference + (k_first[piece] + within) * bench_height
    lo, hi = d0[piece].copy(), d1[piece].copy()
    za, zb = z0[piece], z1[piece]
    rising = np.where(zb > za, 1.0, -1.0)
    level = np.clip(level, np.minimum(za, zb), np.maximum(za, zb))
    x = lo + (level - za) / (zb - za) * (hi - lo)
    x = np.clip(x, lo, hi)
    piece_ids = ids[ph[piece]]
    tolerance = np.maximum(1e-10, 8.0 * np.finfo(np.float64).eps * np.abs(level))
    for _ in range(80):
        zx, gx = _path_z(traces, piece_ids, x)
        f = zx - level
        done = np.abs(f) <= tolerance
        if done.all():
            break
        before = f * rising < 0.0  # x is above (shallower than) the crossing
        lo = np.where(before & ~done, x, lo)
        hi = np.where(~before & ~done, x, hi)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = x - f / gx
        bad = ~np.isfinite(newton) | (newton <= lo) | (newton >= hi)
        x = np.where(done, x, np.where(bad, 0.5 * (lo + hi), newton))
    order = np.lexsort((x, ph[piece]))
    return pd.DataFrame(
        {"hole_id": piece_ids[order], "depth": x[order], "level": level[order]}
    )


# --------------------------------------------------------------------------------------
# composite boundaries
# --------------------------------------------------------------------------------------
def _fixed_length_cuts(
    run_start: np.ndarray, run_end: np.ndarray, length: float, best_fit: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composites of every run: ``(run index, depth_from, depth_to)``."""
    span = run_end - run_start
    if best_fit:
        n = np.maximum(np.rint(span / length), 1.0)
        step = span / n
    else:
        n = np.maximum(np.ceil(span / length - _REL_TOL), 1.0)
        step = np.full(span.shape, float(length))
    n = n.astype(np.int64)
    run = np.repeat(np.arange(n.size), n)
    k = np.arange(run.size) - np.repeat(np.cumsum(n) - n, n)
    start = run_start[run] + k * step[run]
    end = run_start[run] + (k + 1) * step[run]
    last = k == n[run] - 1
    end[last] = run_end[run[last]]
    return run, start, end


def _cuts_at_depths(
    run_hole: np.ndarray,
    run_start: np.ndarray,
    run_end: np.ndarray,
    cut_hole: np.ndarray,
    cut_depth: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Composites of every run when the boundaries are given depths (bench crossings).

    Runs are ordered by hole and depth and do not overlap. A cut belongs to the run with
    the latest start at or above it in the same hole, if it lies strictly inside that run.
    """
    n_run = run_hole.size
    hole = np.concatenate((run_hole, cut_hole))
    depth = np.concatenate((run_start, cut_depth))
    is_cut = np.concatenate((np.zeros(n_run, dtype=bool), np.ones(cut_hole.size, dtype=bool)))
    order = np.lexsort((is_cut, depth, hole))
    position = np.arange(order.size)
    latest = np.maximum.accumulate(np.where(~is_cut[order], position, -1))
    owner = np.where(latest >= 0, order[np.maximum(latest, 0)], 0)  # a run index
    cut_sorted = is_cut[order]
    run_of = owner[cut_sorted]
    d = depth[order][cut_sorted]
    inside = (
        (latest[cut_sorted] >= 0)
        & (run_hole[run_of] == hole[order][cut_sorted])
        & (d > run_start[run_of] + tolerance)
        & (d < run_end[run_of] - tolerance)
    )
    run_of, d = run_of[inside], d[inside]
    if d.size:  # two crossings closer than the tolerance are one boundary
        distinct = np.ones(d.size, dtype=bool)
        distinct[1:] = (run_of[1:] != run_of[:-1]) | (d[1:] - d[:-1] > tolerance)
        run_of, d = run_of[distinct], d[distinct]
    run = np.concatenate((np.arange(n_run), run_of))
    start = np.concatenate((run_start, d))
    order = np.lexsort((start, run))
    run, start = run[order], start[order]
    end = np.empty_like(start)
    end[:-1] = start[1:]
    last = np.ones(run.size, dtype=bool)
    last[:-1] = run[1:] != run[:-1]
    end[last] = run_end[run[last]]
    return run, start, end


def _apply_short_rule(
    run: np.ndarray,
    start: np.ndarray,
    end: np.ndarray,
    short: np.ndarray,
    rule: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if rule == "keep" or not short.any():
        return run, start, end
    if rule == "discard":
        return run[~short], start[~short], end[~short]
    first = np.ones(run.size, dtype=bool)
    first[1:] = run[1:] != run[:-1]
    last = np.ones(run.size, dtype=bool)
    last[:-1] = run[1:] != run[:-1]
    begins = np.ones(run.size, dtype=bool)  # composite i starts a merged composite
    begins[short & ~first] = False  # joins the previous composite of its run
    begins[np.flatnonzero(short & first & ~last) + 1] = False  # the next one joins it
    closes = np.ones(run.size, dtype=bool)
    closes[:-1] = begins[1:]
    return run[begins], start[begins], end[closes]


# --------------------------------------------------------------------------------------
# overlap of composites with segments
# --------------------------------------------------------------------------------------
def _overlaps(
    seg_hole: np.ndarray,
    seg_from: np.ndarray,
    seg_to: np.ndarray,
    comp_hole: np.ndarray,
    comp_from: np.ndarray,
    comp_to: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All (composite, segment, overlap length > 0) triples.

    Both sets are ordered by hole and depth and neither overlaps itself. A single sorted
    key ``hole * span + depth`` finds the candidate segments of each composite; because
    that key is rounded, the candidate range is widened by one segment on each side and
    the overlap itself is computed from the true depths, so nothing is lost or invented.
    """
    n_seg, n_comp = seg_hole.size, comp_hole.size
    if n_seg == 0 or n_comp == 0:
        none = np.empty(0, dtype=np.int64)
        return none, none, np.empty(0)
    span = max(float(seg_to.max()), float(comp_to.max()), 0.0) + 1.0
    key_from, key_to = seg_hole * span + seg_from, seg_hole * span + seg_to
    lo = np.searchsorted(key_to, comp_hole * span + comp_from, side="right") - 1
    hi = np.searchsorted(key_from, comp_hole * span + comp_to, side="left") + 1
    lo, hi = np.clip(lo, 0, n_seg), np.clip(hi, 0, n_seg)
    count = np.maximum(hi - lo, 0)
    comp = np.repeat(np.arange(n_comp), count)
    seg = lo[comp] + (np.arange(comp.size) - np.repeat(np.cumsum(count) - count, count))
    length = np.minimum(seg_to[seg], comp_to[comp]) - np.maximum(seg_from[seg], comp_from[comp])
    keep = (seg_hole[seg] == comp_hole[comp]) & (length > 0.0)
    return comp[keep], seg[keep], length[keep]


# --------------------------------------------------------------------------------------
# the compositor
# --------------------------------------------------------------------------------------
def _fill_values(
    fill_missing: float | Mapping[str, float] | None, attributes: list[str], density: str | None
) -> dict[str, float]:
    if fill_missing is None:
        return {}
    if isinstance(fill_missing, Mapping):
        known = set(attributes) | ({density} if density else set())
        unknown = [k for k in fill_missing if k not in known]
        if unknown:
            raise ValueError(f"fill_missing names unknown attributes {unknown}")
        fill = {str(k): float(v) for k, v in fill_missing.items()}
    else:
        fill = {a: float(fill_missing) for a in attributes}
    bad = [k for k, v in fill.items() if not np.isfinite(v)]
    if bad:
        raise ValueError(f"fill_missing must be finite; got a non-finite value for {bad}")
    return fill


def composite(
    intervals: pd.DataFrame,
    traces: HoleTraces | None = None,
    *,
    length: float | None = None,
    mode: Mode = "downhole",
    attributes: Sequence[str] | None = None,
    bench_height: float | None = None,
    bench_reference: float = 0.0,
    domain: DomainSource | None = None,
    best_fit: bool = False,
    density: str | None = None,
    fill_missing: float | Mapping[str, float] | None = None,
    min_length: float = 0.0,
    short: ShortRule = "keep",
    min_coverage: float = 0.0,
    drop_empty: bool = True,
    on_invalid: OnInvalid = "raise",
) -> pd.DataFrame:
    """Composite the numeric attributes of one interval table. See the module docstring.

    ``intervals``      ``hole_id, depth_from, depth_to`` + attributes. Intervals of one
                       hole must not overlap (``ValueError``).
    ``traces``         desurveyed holes (:func:`desurvey`), for the midpoint coordinates and
                       the bench levels. ``None`` (or a hole without a trace) gives NaN
                       coordinates; ``mode="bench"`` needs traces and skips, with a
                       ``UserWarning``, the holes that have none.
    ``length``         composite length ``L`` (``downhole`` and ``domain`` modes).
    ``mode``           ``"downhole"``, ``"bench"`` or ``"domain"``. A ``domain`` source is
                       honoured in every mode (composites never cross a domain contact);
                       ``"domain"`` is ``"downhole"`` that insists on one.
    ``attributes``     columns to composite; default: every numeric column.
    ``bench_height`` / ``bench_reference``
                       bench levels are ``bench_reference + k * bench_height`` (elevations).
    ``domain``         a column name, an array of codes per row, a callable
                       ``f(intervals) -> codes`` or :class:`IntervalDomains`.
    ``best_fit``       equal division of each run, nearest to ``length`` (not for bench).
    ``density``        name of a density column: weights become ``length * density``. The
                       density column itself is composited by length only. A sample without
                       density cannot be weighted and contributes nothing.
    ``fill_missing``   a value (all attributes) or ``{attribute: value}`` (may include the
                       density column) used for NaN samples and unsampled gaps. Default
                       ``None``: missing stays missing. Filled length counts as coverage.
    ``min_length``     a composite is *short* when it is below this fraction of ``L``
                       (bench mode: its elevation range below this fraction of the bench
                       height).
    ``short``          ``"keep"`` short composites, ``"discard"`` them, or ``"merge"`` each
                       into the previous composite of its run (the first of a run into the
                       next; a run that is one short composite is kept).
    ``min_coverage``   an attribute whose coverage is below this fraction of the composite
                       length is set to NaN.
    ``drop_empty``     leave out composites in which no attribute has any coverage.
    ``on_invalid``     rows with a missing ``hole_id``, a non-finite or negative depth or
                       ``depth_from >= depth_to``: ``"raise"`` (default) or ``"drop"``.

    Rows are ordered by hole (first appearance) and depth.
    """
    # ------------------------------------------------------------------ arguments
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if short not in SHORT_RULES:
        raise ValueError(f"short must be one of {SHORT_RULES}, got {short!r}")
    if on_invalid not in ("raise", "drop"):
        raise ValueError(f"on_invalid must be 'raise' or 'drop', got {on_invalid!r}")
    if not 0.0 <= min_length <= 1.0:
        raise ValueError(f"min_length is a fraction of the composite length, got {min_length!r}")
    if not 0.0 <= min_coverage <= 1.0:
        raise ValueError(f"min_coverage is a fraction in 0..1, got {min_coverage!r}")
    missing = [c for c in INTERVAL_COLUMNS if c not in intervals.columns]
    if missing:
        raise ValueError(f"interval table is missing columns {missing}")
    if mode == "bench":
        if bench_height is None or not (np.isfinite(bench_height) and bench_height > 0):
            raise ValueError(f"bench mode needs bench_height > 0, got {bench_height!r}")
        if traces is None:
            raise ValueError("bench mode needs the desurveyed traces")
        if best_fit:
            raise ValueError("best_fit does not apply to bench mode")
        nominal = float(bench_height)
    else:
        if length is None or not (np.isfinite(length) and length > 0):
            raise ValueError(f"{mode} mode needs length > 0, got {length!r}")
        if mode == "domain" and domain is None:
            raise ValueError("domain mode needs a domain source")
        nominal = float(length)
    tolerance = _REL_TOL * max(nominal, 1.0)

    # ------------------------------------------------------------------ domains
    domain_name: str | None = None
    if isinstance(domain, IntervalDomains):
        domain_name = "domain"
        while domain_name in intervals.columns:
            domain_name = f"_{domain_name}"
        if attributes is None:
            attributes = _numeric_attributes(intervals, exclude=None)
        intervals = split_at_domains(intervals, domain, name=domain_name, on_invalid=on_invalid)
        domain = domain_name
    codes = None if domain is None else _domain_values(intervals, domain)

    # ------------------------------------------------------------------ attributes
    if attributes is None:
        attributes = _numeric_attributes(intervals, domain if isinstance(domain, str) else None)
    attributes = [str(a) for a in attributes]
    if len(set(attributes)) != len(attributes):
        raise ValueError(f"attributes are listed twice: {attributes}")
    for column in [*attributes, *([density] if density else [])]:
        if column not in intervals.columns:
            raise ValueError(f"column {column!r} is not in the interval table")
        if not pd.api.types.is_numeric_dtype(intervals[column]):
            raise ValueError(f"column {column!r} is not numeric and cannot be composited")
    reserved = {*INTERVAL_COLUMNS, *DERIVED_INTERVAL_COLUMNS}
    clash = [a for a in attributes if a in reserved or coverage_column(a) in attributes]
    if clash:
        raise ValueError(
            f"attribute names clash with composite columns: {clash}; composite named "
            "attributes only (--attributes) or rename the column at import"
        )
    fill = _fill_values(fill_missing, attributes, density)

    # ------------------------------------------------------------------ usable, sorted segments
    ids = intervals["hole_id"]
    seg_from = intervals["depth_from"].to_numpy(dtype=np.float64)
    seg_to = intervals["depth_to"].to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        good = (
            ids.notna().to_numpy()
            & np.isfinite(seg_from)
            & np.isfinite(seg_to)
            & (seg_from >= 0.0)
            & (seg_from < seg_to)
        )
    if not good.all() and on_invalid == "raise":
        bad = int(np.flatnonzero(~good)[0])
        raise ValueError(
            f"{int((~good).sum())} interval(s) cannot be composited; first is row {bad}: "
            f"hole_id={ids.iloc[bad]!r}, depth_from={seg_from[bad]!r}, depth_to={seg_to[bad]!r}. "
            "Fix the data or pass on_invalid='drop'."
        )
    hole_code, hole_labels = pd.factorize(ids.where(good), sort=False)
    hole_code = hole_code.astype(np.int64)
    rows = np.flatnonzero(good)
    rows = rows[np.lexsort((seg_from[rows], hole_code[rows]))]
    seg_hole, seg_from, seg_to = hole_code[rows], seg_from[rows], seg_to[rows]
    if rows.size > 1:
        # An overlap below the depth tolerance is float noise (it is weighed as it is).
        overlap = (seg_hole[1:] == seg_hole[:-1]) & (
            (seg_from[1:] < seg_to[:-1] - tolerance) | (seg_to[1:] < seg_to[:-1])
        )
        if overlap.any():
            k = int(np.flatnonzero(overlap)[0])
            raise ValueError(
                f"{int(overlap.sum())} overlapping interval pair(s) in the same hole; first is "
                f"hole {hole_labels[seg_hole[k]]!r}: {seg_from[k]:.10g}-{seg_to[k]:.10g} and "
                f"{seg_from[k + 1]:.10g}-{seg_to[k + 1]:.10g}. Resolve overlaps before compositing."
            )
    hole_ids = np.asarray(hole_labels, dtype=object)

    # ------------------------------------------------------------------ runs
    new_run = np.ones(rows.size, dtype=bool)
    if rows.size:
        new_run[1:] = seg_hole[1:] != seg_hole[:-1]
    domain_levels = None
    if codes is not None:
        seg_code, domain_levels = pd.factorize(codes.iloc[rows], sort=False)  # missing -> -1
        if rows.size:
            new_run[1:] |= seg_code[1:] != seg_code[:-1]
    first = np.flatnonzero(new_run)
    last = np.concatenate((first[1:] - 1, [rows.size - 1])) if rows.size else first
    run_hole = seg_hole[first]
    run_start = seg_from[first] if codes is not None else np.zeros(first.size)
    run_end = seg_to[last]
    run_code = seg_code[first] if codes is not None else None

    # ------------------------------------------------------------------ composite boundaries
    if mode == "bench":
        traced = pd.Index(traces.hole_ids).get_indexer(hole_ids) >= 0
        if not traced.all():
            warnings.warn(
                f"bench compositing skipped {int((~traced).sum())} hole(s) without a trace, "
                f"e.g. {hole_ids[~traced][:5].tolist()}",
                UserWarning,
                stacklevel=2,
            )
            keep = traced[run_hole]
            run_hole, run_start, run_end = run_hole[keep], run_start[keep], run_end[keep]
            run_code = run_code[keep] if run_code is not None else None
        crossings = bench_crossings(traces, nominal, bench_reference)
        cut_hole = pd.Index(hole_ids).get_indexer(crossings["hole_id"].to_numpy(dtype=object))
        known = cut_hole >= 0
        comp_run, comp_from, comp_to = _cuts_at_depths(
            run_hole,
            run_start,
            run_end,
            cut_hole[known],
            crossings["depth"].to_numpy()[known],
            tolerance,
        )
    else:
        comp_run, comp_from, comp_to = _fixed_length_cuts(run_start, run_end, nominal, best_fit)

    if min_length > 0.0 and short != "keep" and comp_run.size:
        if mode == "bench":
            ends = np.concatenate((comp_from, comp_to))
            z_ends, _ = _path_z(traces, hole_ids[run_hole[np.tile(comp_run, 2)]], ends)
            measure = np.abs(z_ends[: comp_run.size] - z_ends[comp_run.size :])
            measure = np.where(np.isfinite(measure), measure, comp_to - comp_from)
        else:
            measure = comp_to - comp_from
        is_short = measure < min_length * nominal - tolerance
        comp_run, comp_from, comp_to = _apply_short_rule(
            comp_run, comp_from, comp_to, is_short, short
        )
    comp_hole = run_hole[comp_run]
    comp_length = comp_to - comp_from
    n_comp = comp_run.size

    # ------------------------------------------------------------------ weighted sums
    comp, seg, piece = _overlaps(seg_hole, seg_from, seg_to, comp_hole, comp_from, comp_to)
    gap = np.maximum(comp_length - np.bincount(comp, weights=piece, minlength=n_comp), 0.0)
    gap[gap <= 8.0 * np.finfo(np.float64).eps * np.maximum(comp_to, 1.0)] = 0.0
    source = rows[seg]
    if density is not None:
        rho = intervals[density].to_numpy(dtype=np.float64)[source]
        if density in fill:
            rho = np.where(np.isfinite(rho), rho, fill[density])
            gap_weight = gap * fill[density]
        else:
            gap_weight = np.zeros(n_comp)
        weight = piece * rho
    else:
        weight, gap_weight = piece, gap

    values: dict[str, np.ndarray] = {}
    coverage: dict[str, np.ndarray] = {}
    any_coverage = np.zeros(n_comp, dtype=bool)
    for attribute in attributes:
        v = intervals[attribute].to_numpy(dtype=np.float64)[source]
        by_length = attribute == density
        w = piece if by_length else weight
        gw = gap if by_length else gap_weight
        if attribute in fill:
            v = np.where(np.isfinite(v), v, fill[attribute])
        ok = np.isfinite(v) & np.isfinite(w)
        numerator = np.bincount(comp[ok], weights=(v * w)[ok], minlength=n_comp)
        denominator = np.bincount(comp[ok], weights=w[ok], minlength=n_comp)
        support = np.bincount(comp[ok], weights=piece[ok], minlength=n_comp)
        if attribute in fill:
            numerator += fill[attribute] * gw
            denominator += gw
            support += np.where(gw > 0.0, gap, 0.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = np.where(denominator > 0.0, numerator / denominator, np.nan)
        any_coverage |= support > 0.0
        if min_coverage > 0.0:
            value = np.where(support < min_coverage * comp_length - tolerance, np.nan, value)
        values[attribute], coverage[attribute] = value, support

    # ------------------------------------------------------------------ output
    keep = any_coverage if (drop_empty and attributes) else np.ones(n_comp, dtype=bool)
    comp_ids = hole_ids[comp_hole[keep]]
    middle = 0.5 * (comp_from[keep] + comp_to[keep])
    if traces is not None and keep.any():
        xyz, direction = _locate_midpoints(traces, comp_ids, middle)
    else:
        xyz = direction = np.full((int(keep.sum()), 3), np.nan)
    out: dict[str, object] = {
        "hole_id": pd.Series(hole_labels.take(comp_hole[keep])).reset_index(drop=True),
        "depth_from": comp_from[keep],
        "depth_to": comp_to[keep],
        "length": comp_length[keep],
        "x": xyz[:, 0],
        "y": xyz[:, 1],
        "z": xyz[:, 2],
        "dx": direction[:, 0],
        "dy": direction[:, 1],
        "dz": direction[:, 2],
    }
    if run_code is not None:
        code = run_code[comp_run[keep]]
        levels = pd.Series(domain_levels).reset_index(drop=True)
        out["domain"] = levels.reindex(code).reset_index(drop=True)  # -1 -> missing
    for attribute in attributes:
        out[attribute] = values[attribute][keep]
    for attribute in attributes:
        out[coverage_column(attribute)] = coverage[attribute][keep]
    return pd.DataFrame(out)


def _numeric_attributes(intervals: pd.DataFrame, exclude: str | None) -> list[str]:
    return [
        str(c)
        for c in intervals.columns
        if c not in INTERVAL_COLUMNS
        and c != exclude
        and pd.api.types.is_numeric_dtype(intervals[c])
        and not pd.api.types.is_bool_dtype(intervals[c])
    ]
