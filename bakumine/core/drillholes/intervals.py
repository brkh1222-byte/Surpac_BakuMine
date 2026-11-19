"""Interval merge: several interval tables -> one segment table per hole.

Each input table has ``hole_id, depth_from, depth_to`` and any attribute columns (assay
grades, rock codes, ...). The output is cut at every boundary of every input table, so
each output segment lies inside at most one interval of each table and can carry that
interval's attributes unchanged:

::

    assay    |----a1----|----a2----|          |--a3--|
    geology  |------g1-------|--------g2---------|
    merged   |----1-----|--2-|--3--|----4-----|-5|-6-|
             a1,g1      a2,g1 a2,g2  -,g2    a3,g2 a3,-

Segments exist wherever at least one table has an interval (the union of covered depth);
where a table has nothing, its attributes are NaN / missing. Nothing is averaged or
interpolated here; that is compositing (``research.md`` section 4).

The work is done on whole arrays for all holes at once: one sort of all interval ends,
then a forward fill per table. There is no Python loop over rows or holes.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import numpy as np
import pandas as pd

from .importer import INTERVAL_COLUMNS

__all__ = ["merge_intervals"]

OnInvalid = Literal["raise", "drop"]


def _output_names(tables: Mapping[str, pd.DataFrame]) -> dict[str, dict[str, str]]:
    """Attribute column -> output name. A name used by several tables gets a table prefix."""
    seen: dict[str, int] = {}
    for table in tables.values():
        for column in table.columns:
            if column not in INTERVAL_COLUMNS:
                seen[str(column)] = seen.get(str(column), 0) + 1
    reserved = set(INTERVAL_COLUMNS) | {f"{name}_row" for name in tables}
    names: dict[str, dict[str, str]] = {}
    used: set[str] = set()
    for name, table in tables.items():
        names[name] = {}
        for column in table.columns:
            if column in INTERVAL_COLUMNS:
                continue
            out = str(column)
            if seen[out] > 1 or out in reserved:
                out = f"{name}_{column}"
            if out in used or out in reserved:
                raise ValueError(f"cannot give a unique output name to {name}.{column}")
            used.add(out)
            names[name][column] = out
    return names


def _sort_by_hole_then_depth(hole: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Indices that order the points by hole code, then depth (ties in any order).

    Sorting one combined float key is several times faster than ``np.lexsort`` on millions
    of points, but rounding in the key can misorder nearly equal depths. So the result is
    checked, and ``np.lexsort`` is used whenever it is not exactly sorted.
    """
    if hole.size == 0:
        return np.empty(0, dtype=np.int64)
    low = float(depth.min())
    span = float(depth.max()) - low + 1.0
    order = np.argsort(hole * span + (depth - low))
    h, d = hole[order], depth[order]
    in_order = (h[1:] > h[:-1]) | ((h[1:] == h[:-1]) & (d[1:] >= d[:-1]))
    if in_order.all():
        return order
    return np.lexsort((depth, hole))


def merge_intervals(
    tables: Mapping[str, pd.DataFrame],
    *,
    tolerance: float = 0.0,
    on_invalid: OnInvalid = "raise",
    source_rows: bool = False,
) -> pd.DataFrame:
    """Merge interval tables into one segment table, split at every boundary.

    ``tables``       name -> table with ``hole_id, depth_from, depth_to`` + attributes.
    ``tolerance``    boundaries of one hole closer than this are treated as one (the
                     shallowest is kept), so float noise does not create sliver segments.
                     ``0`` splits at every distinct depth.
    ``on_invalid``   rows with a missing ``hole_id``, a non-finite depth or
                     ``depth_from >= depth_to`` cannot be merged: ``"raise"`` (default)
                     stops with ``ValueError``, ``"drop"`` leaves them out.
    ``source_rows``  add one ``<table>_row`` column per table with the 0-based position of
                     the source row (``<NA>`` where the table has no interval).

    Returns columns ``hole_id, depth_from, depth_to``, the optional row columns, then each
    table's attributes in table order. An attribute name that occurs in more than one table
    is prefixed ``<table>_``. Rows are ordered by hole (first appearance over the tables in
    the order given) and depth. Numeric attributes with gaps come back as float (NaN).

    Overlapping intervals inside one table make "the" value of a segment ambiguous and
    always raise ``ValueError``; find them with ``validate`` first.
    """
    if not tables:
        raise ValueError("merge_intervals needs at least one interval table")
    if on_invalid not in ("raise", "drop"):
        raise ValueError(f"on_invalid must be 'raise' or 'drop', got {on_invalid!r}")
    if not tolerance >= 0:
        raise ValueError(f"tolerance must be >= 0, got {tolerance!r}")
    for name, table in tables.items():
        missing = [c for c in INTERVAL_COLUMNS if c not in table.columns]
        if missing:
            raise ValueError(f"{name} table is missing columns {missing}")
    names = _output_names(tables)

    # --- usable rows of every table --------------------------------------------------
    rows: dict[str, np.ndarray] = {}  # source row position of each usable interval
    ids, starts, ends = [], [], []
    for name, table in tables.items():
        hole = table["hole_id"]
        start = table["depth_from"].to_numpy(dtype=np.float64)
        end = table["depth_to"].to_numpy(dtype=np.float64)
        with np.errstate(invalid="ignore"):
            good = hole.notna().to_numpy() & np.isfinite(start) & np.isfinite(end) & (start < end)
        if not good.all():
            if on_invalid == "raise":
                bad = int(np.flatnonzero(~good)[0])
                raise ValueError(
                    f"{name}: {int((~good).sum())} interval(s) cannot be merged; first is row "
                    f"{bad}: hole_id={hole.iloc[bad]!r}, depth_from={start[bad]!r}, "
                    f"depth_to={end[bad]!r}. Fix the data or pass on_invalid='drop'."
                )
            keep = np.flatnonzero(good)
            hole, start, end = hole.iloc[keep], start[keep], end[keep]
        else:
            keep = np.arange(len(table))
        rows[name] = keep
        ids.append(hole.astype("str"))
        starts.append(start)
        ends.append(end)

    # --- every interval end of every table, sorted by hole then depth ------------------
    counts = [len(k) for k in rows.values()]
    total = int(sum(counts))
    codes, holes = pd.factorize(pd.concat(ids, ignore_index=True), sort=False)
    codes = codes.astype(np.int64)
    end_hole = np.concatenate([codes, codes])  # first all starts, then all ends
    end_depth = np.concatenate([*starts, *ends])
    order = _sort_by_hole_then_depth(end_hole, end_depth)
    hole_sorted, depth_sorted = end_hole[order], end_depth[order]
    new = np.ones(2 * total, dtype=bool)
    if total:
        new[1:] = (hole_sorted[1:] != hole_sorted[:-1]) | (
            depth_sorted[1:] - depth_sorted[:-1] > tolerance
        )
    boundary_of = np.empty(2 * total, dtype=np.int64)  # boundary index of every interval end
    boundary_of[order] = np.cumsum(new) - 1
    boundary_hole, boundary_depth = hole_sorted[new], depth_sorted[new]
    n_boundary = boundary_depth.size
    position = np.arange(n_boundary)

    # --- per table: which interval owns the segment that starts at each boundary ---------
    owners: dict[str, np.ndarray] = {}
    covered_any = np.zeros(n_boundary, dtype=bool)
    offset = 0
    for name, n in zip(tables, counts, strict=True):
        first = boundary_of[offset : offset + n]
        last = boundary_of[total + offset : total + offset + n]
        offset += n
        by_start = np.argsort(first, kind="stable")
        clash = last[by_start][:-1] > first[by_start][1:]
        if clash.any():
            k = int(np.flatnonzero(clash)[0])
            a, b = rows[name][by_start[k]], rows[name][by_start[k + 1]]
            table = tables[name]
            span = [
                f"{float(table['depth_from'].iloc[r]):.10g}-{float(table['depth_to'].iloc[r]):.10g}"
                for r in (a, b)
            ]
            raise ValueError(
                f"{name}: {int(clash.sum())} overlapping interval pair(s) in the same hole; "
                f"first is hole {table['hole_id'].iloc[a]!r}, rows {int(a)} ({span[0]}) and "
                f"{int(b)} ({span[1]}). Resolve overlaps before merging (or set a tolerance "
                "if they are float noise)."
            )
        # Forward fill the latest interval start; it owns the segment if it has not ended.
        # A sliver interval whose two ends snapped together (first == last) owns nothing.
        starter = np.full(n_boundary, -1, dtype=np.int64)
        starter[first] = np.arange(n)
        latest = np.maximum.accumulate(np.where(starter >= 0, position, -1))
        owner = np.where(latest >= 0, starter[np.maximum(latest, 0)], -1)
        has_owner = owner >= 0
        has_owner[has_owner] = last[owner[has_owner]] > position[has_owner]
        owner[~has_owner] = -1
        owners[name] = owner
        covered_any |= has_owner

    # --- output ----------------------------------------------------------------------------
    segment = np.flatnonzero(covered_any)
    out: dict[str, object] = {
        "hole_id": pd.Series(np.asarray(holes, dtype=object)[boundary_hole[segment]], dtype="str"),
        "depth_from": boundary_depth[segment],
        "depth_to": boundary_depth[segment + 1] if segment.size else np.empty(0),
    }
    source: dict[str, np.ndarray] = {}
    for name in tables:
        owner = owners[name][segment]
        if rows[name].size:
            source[name] = np.where(owner >= 0, rows[name][np.maximum(owner, 0)], -1)
        else:
            source[name] = np.full(segment.size, -1, dtype=np.int64)
        if source_rows:
            row = pd.array(source[name], dtype="Int64")
            row[source[name] < 0] = pd.NA
            out[f"{name}_row"] = row
    for name, table in tables.items():
        if not names[name]:
            continue
        attributes = table[list(names[name])].reset_index(drop=True)
        taken = attributes.reindex(source[name]).reset_index(drop=True)  # -1 -> missing
        for column, out_name in names[name].items():
            out[out_name] = taken[column]
    return pd.DataFrame(out)
