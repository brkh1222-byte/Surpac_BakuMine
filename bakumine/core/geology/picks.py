"""Per-hole picks of cover and weathering unit bases, with bounds (research.md 5.7.4).

A hole gives one row per surface (geology_contracts.md 4):

- ``exact``: the unit ends against another logged unit;
- ``lower``: the surface is at least this deep (the hole ends in the unit, or nothing is
  logged below it);
- ``upper``: the surface is at most this deep;
- ``between``: the contact lies in an unlogged stretch (core loss, a gap) with the unit
  above and another unit below;
- ``none``: the hole gives no information. This is never read as zero depth.

Depths are along the hole. ``below_lo`` / ``below_hi`` are vertical depths below the
topography at the pick's own (x, y), not at the collar, so inclined holes are handled.

Sources of intervals: a logged table (:meth:`UnitDictionary.map_codes`), a rule on an assay
column (:func:`threshold_units`, e.g. sulphur as a stand-in for the oxide zone) and an extra
CSV file (:func:`read_extra_intervals`). :func:`combine_picks` merges the pick tables of
several sources and flags the holes where they disagree.
"""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.geology.provenance import ProvenanceEntry
from bakumine.core.geometry.dtm import DTM

PICK_COLUMNS = (
    "hole_id",
    "surface",
    "kind",
    "source",
    "depth",
    "depth_max",
    "x",
    "y",
    "z",
    "x2",
    "y2",
    "z2",
    "topo_z",
    "topo_z2",
    "below_lo",
    "below_hi",
)
PICK_KINDS = ("exact", "lower", "upper", "between", "none")
PICK_SOURCES = ("log", "threshold", "extra")
_FLOAT_COLUMNS = PICK_COLUMNS[4:]
_TASK = "P9.T6"
_STAGE = "picks"

_UNKNOWN, _OUT, _IN = 0, 1, 2


@dataclass
class PickResult:
    """Picks of one surface from one source."""

    table: pd.DataFrame
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------- one hole


def _runs(
    top: np.ndarray, bottom: np.ndarray, cls: np.ndarray, tolerance: float
) -> tuple[list[tuple[int, float, float]], bool]:
    """Runs ``(class, top, bottom)`` of one hole, top to bottom, and whether intervals overlap.

    Gaps longer than ``tolerance`` are UNKNOWN runs. Where intervals overlap, IN wins over
    OUT and OUT over UNKNOWN. An UNKNOWN stretch between two IN runs does not end the run.
    """
    edges = np.unique(np.concatenate([top, bottom]))
    keep = np.ones(edges.size, dtype=bool)
    last = edges[0]
    for i in range(1, edges.size):  # snap edges closer than the tolerance
        if edges[i] - last <= tolerance:
            keep[i] = False
        else:
            last = edges[i]
    edges = edges[keep]
    if edges.size < 2:
        return [], False

    def edge_index(values: np.ndarray) -> np.ndarray:
        i = np.searchsorted(edges, values, side="right") - 1
        return np.clip(i, 0, edges.size - 1)

    first, last_edge = edge_index(top), edge_index(bottom)
    seg = np.zeros(edges.size - 1, dtype=np.int8)
    cover = np.zeros(edges.size - 1, dtype=np.int64)
    for a, b, c in zip(first, last_edge, cls, strict=True):
        if b > a:
            seg[a:b] = np.maximum(seg[a:b], c)
            cover[a:b] += 1
    overlap = bool((cover > 1).any())

    runs: list[tuple[int, float, float]] = []
    for k in range(seg.size):
        c = int(seg[k])
        if runs and runs[-1][0] == c:
            runs[-1] = (c, runs[-1][1], float(edges[k + 1]))
        else:
            runs.append((c, float(edges[k]), float(edges[k + 1])))
    joined: list[tuple[int, float, float]] = []
    for k, run in enumerate(runs):
        inside = (
            run[0] == _UNKNOWN
            and joined
            and joined[-1][0] == _IN
            and k + 1 < len(runs)
            and runs[k + 1][0] == _IN
        )
        if joined and joined[-1][0] == _IN and (inside or run[0] == _IN):
            joined[-1] = (_IN, joined[-1][1], run[2])
        else:
            joined.append(run)
    return joined, overlap


def _base_pick(
    runs: list[tuple[int, float, float]], mode: str, absent: str
) -> tuple[str, float, float, str]:
    """``(kind, depth, depth_max, reason)`` of the base of the IN runs (research.md 5.7.4).

    ``reason`` is ``""`` for a pick read from the unit itself, ``"absent"`` for a bound that
    comes from the ``absent = zero`` policy, and for ``none``: ``"not_logged"`` (other units
    logged, the target never) or ``"no_data"`` (nothing usable logged).
    """
    inside = [k for k, run in enumerate(runs) if run[0] == _IN]
    if not inside:
        first_out = next((run for run in runs if run[0] == _OUT), None)
        if first_out is None:
            return "none", np.nan, np.nan, "no_data"
        if absent == "zero":
            return "upper", first_out[1], np.nan, "absent"
        return "none", np.nan, np.nan, "not_logged"
    k = inside[-1] if mode == "deepest" else inside[0]
    bottom = runs[k][2]
    if k + 1 < len(runs):
        nxt = runs[k + 1]
        if nxt[0] == _OUT:
            return "exact", bottom, np.nan, ""
        if nxt[0] == _UNKNOWN and k + 2 < len(runs) and runs[k + 2][0] == _OUT:
            return "between", bottom, nxt[2], ""
    return "lower", bottom, np.nan, ""


def _hole_pick(
    top: np.ndarray,
    bottom: np.ndarray,
    cls: np.ndarray,
    *,
    side: str,
    mode: str,
    absent: str,
    tolerance: float,
) -> tuple[str, float, float, str, bool]:
    """Pick of one hole: ``(kind, depth, depth_max, reason, overlap)``.

    The top of a unit is the base rule applied to the hole turned upside down: a lower bound
    of the mirrored depth is an upper bound of the depth, and the bracket is swapped.
    """
    if side == "base":
        runs, overlap = _runs(top, bottom, cls, tolerance)
        return (*_base_pick(runs, mode, absent), overlap)
    runs, overlap = _runs(-bottom, -top, cls, tolerance)
    kind, depth, depth_max, reason = _base_pick(runs, mode, absent)
    if kind == "exact":
        return kind, -depth, np.nan, reason, overlap
    if kind == "lower":
        return "upper", -depth, np.nan, reason, overlap
    if kind == "upper":
        return "lower", -depth, np.nan, reason, overlap
    if kind == "between":
        return kind, -depth_max, -depth, reason, overlap
    return kind, np.nan, np.nan, reason, overlap


# ---------------------------------------------------------------------------- picks


def _empty_table() -> pd.DataFrame:
    frame = pd.DataFrame({c: pd.Series(dtype=object) for c in PICK_COLUMNS[:4]})
    for c in _FLOAT_COLUMNS:
        frame[c] = pd.Series(dtype=np.float64)
    return frame


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def unit_picks(
    table: pd.DataFrame,
    units: pd.Series,
    target: Collection[str],
    traces: HoleTraces,
    topography: DTM,
    *,
    surface: str,
    side: str = "base",
    mode: str = "deepest",
    absent: str = "unknown",
    unknown_codes: Collection[str] = (),
    source: str = "log",
    depth_tolerance: float = 1e-6,
) -> PickResult:
    """Picks of the base (or top) of the units ``target``, one row per hole with a trace.

    Parameters
    ----------
    table : interval table with ``hole_id, depth_from, depth_to``.
    units : unit code per row of ``table`` (same length, taken by position); NaN = unknown.
        The output of :meth:`UnitDictionary.map_codes` or of :func:`threshold_units`.
    target : the unit codes that make up the zone (IN). Every other code is OUT, except
        ``unknown_codes`` (core loss and the like), which are UNKNOWN like NaN and gaps.
    surface : name written to the ``surface`` column, e.g. ``"base:OXIDE"``.
    side : ``"base"`` or ``"top"`` (the same rules, upward).
    mode : ``"deepest"``: the bottom of the deepest IN run (``top``: the top of the
        shallowest). ``"first_run"``: the bottom of the first IN run from the top of the
        logging (``top``: the top of the first run from the bottom).
    absent : what a hole with other units logged and the target never means. ``"unknown"``
        (default): no information. ``"zero"``: the zone is absent, so its base is not deeper
        than the top of the first OUT interval (an upper bound).
    source : ``"log"``, ``"threshold"`` or ``"extra"``; written to the ``source`` column.

    Rows of holes without a trace are left out and listed in ``diagnostics``. A pick outside
    the topography keeps its kind and depth; its ``topo_z`` and ``below_*`` are NaN.
    """
    if side not in ("base", "top"):
        raise ValueError(f"side must be 'base' or 'top', got {side!r}")
    if mode not in ("deepest", "first_run"):
        raise ValueError(f"mode must be 'deepest' or 'first_run', got {mode!r}")
    if absent not in ("unknown", "zero"):
        raise ValueError(f"absent must be 'unknown' or 'zero', got {absent!r}")
    if source not in PICK_SOURCES:
        raise ValueError(f"source must be one of {', '.join(PICK_SOURCES)}, got {source!r}")
    if not isinstance(surface, str) or not surface.strip():
        raise ValueError("surface must be a name such as 'base:OXIDE'")
    if not depth_tolerance >= 0.0:
        raise ValueError("depth_tolerance must be >= 0")
    if isinstance(target, str):
        target = (target,)
    target_set = {str(code) for code in target}
    if not target_set:
        raise ValueError("target must name at least one unit code")
    unknown_set = {
        str(code)
        for code in ((unknown_codes,) if isinstance(unknown_codes, str) else unknown_codes)
    }
    if target_set & unknown_set:
        both = ", ".join(sorted(target_set & unknown_set))
        raise ValueError(f"codes both in target and in unknown_codes: {both}")
    missing = [c for c in ("hole_id", "depth_from", "depth_to") if c not in table.columns]
    if missing:
        raise ValueError(f"the interval table lacks the columns: {', '.join(missing)}")
    if len(units) != len(table):
        raise ValueError(f"units has {len(units)} values for {len(table)} table rows")

    hole = table["hole_id"].to_numpy(dtype=object)
    top = pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(dtype=np.float64)
    bottom = pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(dtype=np.float64)
    codes = pd.Series(np.asarray(units, dtype=object))
    known = codes.notna().to_numpy()
    text = codes.where(codes.notna(), "").astype(str)
    cls = np.full(len(table), _UNKNOWN, dtype=np.int8)
    cls[known & ~text.isin(list(unknown_set)).to_numpy()] = _OUT
    cls[known & text.isin(list(target_set)).to_numpy()] = _IN

    usable = pd.notna(hole) & np.isfinite(top) & np.isfinite(bottom) & (bottom > top)
    n_dropped = int((~usable).sum())
    hole, top, bottom, cls = hole[usable], top[usable], bottom[usable], cls[usable]

    traced = np.array([h in traces for h in hole], dtype=bool)
    skipped = list(dict.fromkeys(hole[~traced].tolist()))
    groups = pd.Series(np.flatnonzero(traced)).groupby(hole[traced], sort=False).indices
    rows_of = {h: np.flatnonzero(traced)[i] for h, i in groups.items()}

    ids = list(traces.hole_ids)
    n = len(ids)
    kinds = np.full(n, "none", dtype=object)
    depth = np.full(n, np.nan)
    depth_max = np.full(n, np.nan)
    reasons = np.full(n, "no_data", dtype=object)
    overlapping: list[Any] = []
    for i, hole_id in enumerate(ids):
        rows = rows_of.get(hole_id)
        if rows is None:
            continue
        kinds[i], depth[i], depth_max[i], reasons[i], overlap = _hole_pick(
            top[rows],
            bottom[rows],
            cls[rows],
            side=side,
            mode=mode,
            absent=absent,
            tolerance=depth_tolerance,
        )
        if overlap:
            overlapping.append(hole_id)

    limit = np.asarray(traces.max_depth, dtype=np.float64) if n else np.zeros(0)
    id_array = np.empty(n, dtype=object)
    id_array[:] = ids

    def locate(d: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        xyz = np.full((n, 3), np.nan)
        topo = np.full(n, np.nan)
        sel = np.flatnonzero(np.isfinite(d))
        beyond = np.zeros(n, dtype=bool)
        if sel.size:
            beyond[sel] = d[sel] > limit[sel] + depth_tolerance
            query = np.clip(d[sel], 0.0, limit[sel])
            xyz[sel] = traces.xyz_at(id_array[sel], query)[0]
            topo[sel] = topography.z_at(xyz[sel, 0], xyz[sel, 1])
        return xyz, topo, beyond

    xyz, topo, beyond = locate(depth)
    xyz2, topo2, beyond2 = locate(depth_max)
    below = topo - xyz[:, 2]
    below2 = topo2 - xyz2[:, 2]
    below_lo = np.where(np.isin(kinds, ("exact", "lower", "between")), below, np.nan)
    below_hi = np.where(
        np.isin(kinds, ("exact", "upper")), below, np.where(kinds == "between", below2, np.nan)
    )

    frame = pd.DataFrame(
        {
            "hole_id": id_array,
            "surface": np.full(n, surface, dtype=object),
            "kind": kinds,
            "source": np.full(n, source, dtype=object),
            "depth": depth,
            "depth_max": depth_max,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "x2": xyz2[:, 0],
            "y2": xyz2[:, 1],
            "z2": xyz2[:, 2],
            "topo_z": topo,
            "topo_z2": topo2,
            "below_lo": below_lo,
            "below_hi": below_hi,
        },
        columns=list(PICK_COLUMNS),
    )
    if n == 0:
        frame = _empty_table()

    picked = kinds != "none"
    outside = picked & (np.isnan(topo) | ((kinds == "between") & np.isnan(topo2)))
    counts = {kind: int((kinds == kind).sum()) for kind in PICK_KINDS}
    from_policy = picked & (reasons == "absent")
    n_bounds = int((picked & (kinds != "exact") & ~from_policy).sum())
    n_not_logged = int((reasons == "not_logged").sum())
    threshold = getattr(units, "attrs", {}).get("threshold")
    diagnostics = {
        "surface": surface,
        "source": source,
        "side": side,
        "mode": mode,
        "absent": absent,
        "counts": counts,
        "holes": n,
        "holes_skipped": skipped,
        "holes_outside_topography": [ids[i] for i in np.flatnonzero(outside)],
        "holes_without_intervals": [h for h in ids if h not in rows_of],
        "holes_target_not_logged": [ids[i] for i in np.flatnonzero(reasons == "not_logged")],
        "holes_absent_policy": [ids[i] for i in np.flatnonzero(from_policy)],
        "holes_overlapping_intervals": overlapping,
        "holes_depth_beyond_trace": [ids[i] for i in np.flatnonzero(beyond | beyond2)],
        "rows_dropped": n_dropped,
        "threshold": threshold,
    }

    what = {"log": "logged intervals", "threshold": "threshold on an assay column"}.get(
        source, "extra intervals file"
    )
    units_text = ", ".join(sorted(target_set))
    origin = f"{what}; units {units_text}"
    provenance: list[ProvenanceEntry] = []

    def entry(kind: str, item: str, src: str, detail: str, count: int, **extra: Any) -> None:
        provenance.append(
            ProvenanceEntry(
                item=item,
                kind=kind,
                source=src,
                used_for=f"surface {surface}",
                detail=detail,
                stage=_STAGE,
                task=_TASK,
                count=count,
                **extra,
            )
        )

    if counts["exact"]:
        entry(
            "measured",
            f"Exact picks of {surface} ({source})",
            origin,
            f"{_plural(counts['exact'], 'hole')} with the contact against another logged unit",
            counts["exact"],
        )
    if n_bounds:
        parts = [
            f"{int(((kinds == kind) & ~from_policy).sum())} {kind}"
            for kind in ("lower", "upper", "between")
            if ((kinds == kind) & ~from_policy).any()
        ]
        entry(
            "measured",
            f"Bounds of {surface} ({source})",
            origin,
            f"{_plural(n_bounds, 'hole')}: {', '.join(parts)}",
            n_bounds,
        )
    if threshold and (counts["exact"] or n_bounds):
        entry(
            "measured",
            f"Threshold rule for {surface}",
            f"column {threshold['column']} {threshold['operator']} {threshold['value']:g}",
            f"{_plural(int(threshold['inside_count']), 'interval')} inside the rule, "
            f"{int(threshold['outside_count'])} outside, {int(threshold['missing_count'])} missing",
            int(threshold["inside_count"]),
            value=float(threshold["value"]),
        )
    if absent == "zero":
        n_policy = int(from_policy.sum())
        bound = "upper" if side == "base" else "lower"
        entry(
            "assumed",
            f"Holes without the units of {surface} ({source})",
            "option absent = zero",
            f"{_plural(n_policy, 'hole')} read as the zone being absent: {bound} bound at the "
            "first other unit",
            n_policy,
        )
    else:
        entry(
            "assumed",
            f"Holes without the units of {surface} ({source})",
            "default (absent = unknown)",
            f"{_plural(n_not_logged, 'hole')} with other units logged give no information, "
            "not zero depth",
            n_not_logged,
        )
    if counts["none"]:
        entry(
            "missing",
            f"{surface} not logged ({source})",
            "logs of the zone, or assays of an element that marks it (e.g. sulphur)",
            f"{_plural(counts['none'], 'hole')} of {n} give no pick and no bound",
            counts["none"],
        )
    return PickResult(table=frame, provenance=provenance, diagnostics=diagnostics)


# ---------------------------------------------------------------------------- sources


def threshold_units(
    table: pd.DataFrame,
    column: str,
    *,
    below: float | None = None,
    above: float | None = None,
    inside: str = "IN",
    outside: str = "OUT",
) -> pd.Series:
    """Class per row from a rule on a numeric column: ``inside`` / ``outside``; NaN stays NaN.

    Exactly one of ``below`` (inside where ``value < below``) and ``above`` (inside where
    ``value > above``). The result has the index of ``table`` and carries the rule in
    ``.attrs["threshold"]``, which :func:`unit_picks` writes to its provenance.
    """
    if (below is None) == (above is None):
        raise ValueError("give exactly one of below / above")
    if column not in table.columns:
        raise ValueError(f"the table has no column {column!r}")
    if inside == outside:
        raise ValueError("inside and outside must be different codes")
    limit = float(below if below is not None else above)
    if not np.isfinite(limit):
        raise ValueError("the threshold must be a finite number")
    values = pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)
    finite = np.isfinite(values)
    with np.errstate(invalid="ignore"):
        is_inside = finite & ((values < limit) if below is not None else (values > limit))
    out = np.full(len(table), np.nan, dtype=object)
    out[finite] = outside
    out[is_inside] = inside
    series = pd.Series(out, index=table.index, dtype=object, name=column)
    series.attrs["threshold"] = {
        "column": str(column),
        "operator": "<" if below is not None else ">",
        "value": limit,
        "inside": inside,
        "outside": outside,
        "inside_count": int(is_inside.sum()),
        "outside_count": int((finite & ~is_inside).sum()),
        "missing_count": int((~finite).sum()),
    }
    return series


def read_extra_intervals(
    path: str | Path,
    *,
    hole_id: str = "hole_id",
    depth_from: str = "depth_from",
    depth_to: str = "depth_to",
    code: str = "code",
) -> pd.DataFrame:
    """Extra logged intervals from a CSV file: ``hole_id, depth_from, depth_to, code``.

    For zones that are logged outside the drillhole database (a weathering log kept in a
    separate sheet). The keyword arguments name the columns of the file. Hole ids and codes
    are stripped; an empty code is NaN (unknown). Depths that are not numbers, or an interval
    with ``depth_to <= depth_from``, raise ``ValueError`` naming the file line.
    """
    path = Path(path)
    raw = pd.read_csv(path, dtype=str, keep_default_na=False, skipinitialspace=True)
    raw.columns = [str(c).strip() for c in raw.columns]
    wanted = {"hole_id": hole_id, "depth_from": depth_from, "depth_to": depth_to, "code": code}
    lacking = [name for name in wanted.values() if name not in raw.columns]
    if lacking:
        raise ValueError(f"{path.name}: missing column(s): {', '.join(lacking)}")
    out = pd.DataFrame(index=raw.index)
    out["hole_id"] = raw[hole_id].str.strip().astype(object)
    for name in ("depth_from", "depth_to"):
        cells = raw[wanted[name]].str.strip()
        values = pd.to_numeric(cells, errors="coerce")
        bad = np.flatnonzero(~np.isfinite(values.to_numpy(dtype=np.float64)))
        if bad.size:
            raise ValueError(
                f"{path.name}, line {bad[0] + 2}: {wanted[name]} {cells.iloc[bad[0]]!r} "
                "is not a number"
            )
        out[name] = values.astype(np.float64)
    codes = raw[code].str.strip().astype(object)
    out["code"] = codes.where(codes != "", np.nan)
    empty = np.flatnonzero((out["hole_id"] == "").to_numpy())
    if empty.size:
        raise ValueError(f"{path.name}, line {empty[0] + 2}: empty hole id")
    wrong = np.flatnonzero((out["depth_to"] <= out["depth_from"]).to_numpy())
    if wrong.size:
        raise ValueError(f"{path.name}, line {wrong[0] + 2}: depth_to is not below depth_from")
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------- combining


def _bound(row: Any, which: str, order: int) -> dict[str, Any]:
    """The lower (``which = "lo"``) or upper bound a pick row carries, with its position."""
    second = which == "hi" and row.kind == "between"
    return {
        "order": order,
        "source": row.source,
        "depth": float(row.depth_max if second else row.depth),
        "x": row.x2 if second else row.x,
        "y": row.y2 if second else row.y,
        "z": row.z2 if second else row.z,
        "topo_z": row.topo_z2 if second else row.topo_z,
        "below": row.below_hi if which == "hi" else row.below_lo,
    }


def _from_bounds(
    hole_id: Any, surface: str, lower: dict | None, upper: dict | None, tolerance: float
) -> dict[str, Any]:
    out: dict[str, Any] = {c: np.nan for c in PICK_COLUMNS}
    out.update(hole_id=hole_id, surface=surface)
    first = lower or upper

    def put(b: dict, suffix: str = "") -> None:
        for name in ("x", "y", "z"):
            out[name + suffix] = b[name]
        out["topo_z" + suffix] = b["topo_z"]

    if lower is not None and upper is not None:
        sources = list(dict.fromkeys([lower["source"], upper["source"]]))
        out["source"] = "+".join(sources)
        out["depth"] = lower["depth"]
        put(lower)
        out["below_lo"] = lower["below"]
        if upper["depth"] - lower["depth"] <= tolerance:
            out["kind"] = "exact"
            out["below_hi"] = lower["below"]
        else:
            out["kind"] = "between"
            out["depth_max"] = upper["depth"]
            put(upper, "2")
            out["below_hi"] = upper["below"]
        return out
    out["source"] = first["source"]
    out["depth"] = first["depth"]
    put(first)
    if lower is not None:
        out["kind"] = "lower"
        out["below_lo"] = lower["below"]
    else:
        out["kind"] = "upper"
        out["below_hi"] = upper["below"]
    return out


def combine_picks(*tables: pd.DataFrame, tolerance: float = 1e-6) -> pd.DataFrame:
    """One pick per hole and surface from the pick tables of several sources.

    The tables are given in order of priority (e.g. log, extra, threshold).

    - An exact pick wins: the first one in argument order.
    - Otherwise the tightest bounds: the deepest lower bound and the shallowest upper bound
      over all tables. Both together make a ``between`` row (an ``exact`` row when they meet);
      its ``source`` joins the two sources with ``+`` when they differ.
    - No pick and no bound anywhere: one ``none`` row.

    The result has :data:`PICK_COLUMNS` and a boolean ``conflict`` column, true where the
    sources contradict each other by more than ``tolerance`` metres along the hole: a lower
    bound below an upper bound, a bound on the wrong side of the exact pick, or two exact
    picks at different depths. A row with a conflict keeps the exact pick, or else the bounds
    of the first table that has any; it should be looked at before it is used.
    """
    if not tolerance >= 0.0:
        raise ValueError("tolerance must be >= 0")
    columns = [*PICK_COLUMNS, "conflict"]
    parts = []
    for order, table in enumerate(tables):
        lacking = [c for c in PICK_COLUMNS if c not in table.columns]
        if lacking:
            raise ValueError(f"pick table {order + 1} lacks the columns: {', '.join(lacking)}")
        bad = sorted(set(table["kind"].unique()) - set(PICK_KINDS))
        if bad:
            raise ValueError(
                f"pick table {order + 1} has unknown kinds: {', '.join(map(str, bad))}"
            )
        part = table[list(PICK_COLUMNS)].copy()
        part["order_"] = order
        parts.append(part)
    if not parts or not sum(len(p) for p in parts):
        empty = _empty_table()
        empty["conflict"] = pd.Series(dtype=bool)
        return empty[columns]
    stacked = pd.concat(parts, ignore_index=True)

    rows: list[dict[str, Any]] = []
    for (hole_id, surface), group in stacked.groupby(["hole_id", "surface"], sort=False):
        group = group.sort_values("order_", kind="stable")
        lowers = [
            _bound(r, "lo", r.order_)
            for r in group.itertuples(index=False)
            if r.kind in ("lower", "between")
        ]
        uppers = [
            _bound(r, "hi", r.order_)
            for r in group.itertuples(index=False)
            if r.kind in ("upper", "between")
        ]
        exact = group[group["kind"] == "exact"]
        if len(exact):
            chosen = exact.iloc[0]
            d = float(chosen["depth"])
            conflict = (
                bool((np.abs(exact["depth"].to_numpy(dtype=np.float64) - d) > tolerance).any())
                or any(b["depth"] > d + tolerance for b in lowers)
                or any(b["depth"] < d - tolerance for b in uppers)
            )
            row = {c: chosen[c] for c in PICK_COLUMNS}
        elif lowers or uppers:
            lower = max(lowers, key=lambda b: (b["depth"], -b["order"])) if lowers else None
            upper = min(uppers, key=lambda b: (b["depth"], b["order"])) if uppers else None
            conflict = (
                lower is not None
                and upper is not None
                and lower["depth"] > upper["depth"] + tolerance
            )
            if conflict:
                first = min(b["order"] for b in (*lowers, *uppers))
                mine_lo = [b for b in lowers if b["order"] == first]
                mine_hi = [b for b in uppers if b["order"] == first]
                lower = max(mine_lo, key=lambda b: b["depth"]) if mine_lo else None
                upper = min(mine_hi, key=lambda b: b["depth"]) if mine_hi else None
                if lower and upper and lower["depth"] > upper["depth"] + tolerance:
                    upper = None
            row = _from_bounds(hole_id, surface, lower, upper, tolerance)
        else:
            conflict = False
            row = {c: group.iloc[0][c] for c in PICK_COLUMNS}
        row["conflict"] = bool(conflict)
        rows.append(row)

    out = pd.DataFrame(rows, columns=columns)
    for c in _FLOAT_COLUMNS:
        out[c] = out[c].astype(np.float64)
    out["conflict"] = out["conflict"].astype(bool)
    return out
