"""Drillhole database validation (``research.md`` section 2.5).

``validate(db)`` runs every check and returns a :class:`ValidationReport`. It never
changes the data and never raises on bad data: every problem is a row of
``report.findings`` with columns

``severity``  ``"error"`` (the data is wrong or unusable), ``"warning"`` (suspect, or will
              be repaired by a later step) or ``"info"``;
``code``      the check code, a key of :data:`CHECKS`;
``table``     ``"collar"``, ``"survey"`` or the interval table's name;
``hole_id``   the hole concerned;
``row``       0-based row position in that table (the data row of the source file), or
              ``<NA>`` when the finding concerns a hole as a whole;
``message``   a sentence with the offending values.

``report.summary`` has one row per hole (every ``hole_id`` seen in any table).

Reading the checks:

* Interval checks (``overlap``, ``gap``, ``interval_past_max_depth``) only use rows whose
  depths are finite with ``depth_from < depth_to``; a bad row is reported once, by
  ``from_ge_to`` or ``invalid_interval_depth``, and does not cascade.
* ``gap`` reports unlogged stretches *between* intervals of one table. An interval table
  that starts below the collar or stops above ``max_depth`` is not a gap.
* ``sudden_deviation`` is the angle between the directions of consecutive stations
  (the dogleg of ``research.md`` section 3) per 30 m of hole. Stations already reported as
  invalid, out of range or duplicate are left out of it.
* ``survey_past_max_depth`` also catches "straight to end of hole" sentinel stations
  such as depth 90000.
* ``missing_collar`` is reported once per table and hole, with the number of rows.
* ``collar_outlier`` is a robust distance test: a collar is flagged when its plan distance
  from the median collar exceeds ``outlier_factor`` times the median of those distances
  (and likewise for elevation).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .desurvey import direction_vectors
from .importer import LENGTH_UNITS, DrillholeDatabase

__all__ = ["CHECKS", "SEVERITIES", "ValidationReport", "validate"]

SEVERITIES: tuple[str, ...] = ("error", "warning", "info")

#: check code -> (severity, what it means)
CHECKS: dict[str, tuple[str, str]] = {
    "duplicate_hole_id": ("error", "hole_id occurs more than once in the collar table"),
    "non_finite_coordinate": ("error", "collar x, y or z is missing or not finite"),
    "invalid_max_depth": ("warning", "collar max_depth is missing, not finite or not positive"),
    "collar_outlier": ("warning", "collar lies far outside the cluster of the other collars"),
    "missing_collar": ("error", "survey or interval rows whose hole_id has no collar"),
    "no_survey": ("error", "hole has a collar but no survey station"),
    "invalid_survey_station": ("error", "survey depth, dip or azimuth not finite, or depth < 0"),
    "no_survey_at_zero": ("warning", "hole has no survey station at depth 0"),
    "dip_out_of_range": ("error", "dip outside -90..90 degrees"),
    "azimuth_out_of_range": ("error", "azimuth outside 0..360 degrees"),
    "duplicate_survey_depth": ("error", "two survey stations of one hole at the same depth"),
    "sudden_deviation": ("warning", "direction change between stations above the threshold"),
    "survey_past_max_depth": ("warning", "survey station deeper than the hole's max_depth"),
    "invalid_interval_depth": ("error", "interval depth missing, not finite or negative"),
    "from_ge_to": ("error", "interval with depth_from >= depth_to"),
    "overlap": ("error", "interval starts before an earlier interval of the hole ends"),
    "gap": ("info", "unlogged stretch between two intervals of the hole"),
    "interval_past_max_depth": ("error", "interval ends below the hole's max_depth"),
}

_FINDING_COLUMNS = ("severity", "code", "table", "hole_id", "row", "message")
_SUMMARY_COLUMNS = (
    "hole_id", "in_collar", "survey_stations", "interval_rows", "errors", "warnings", "infos",
    "status", "codes",
)


@dataclass
class ValidationReport:
    """Result of :func:`validate`. ``findings`` and ``summary`` are plain DataFrames."""

    findings: pd.DataFrame
    summary: pd.DataFrame

    @property
    def ok(self) -> bool:
        """True when there is no finding of severity ``error``."""
        return not (self.findings["severity"] == "error").any()

    @property
    def errors(self) -> pd.DataFrame:
        return self.findings[self.findings["severity"] == "error"]

    @property
    def warnings(self) -> pd.DataFrame:
        return self.findings[self.findings["severity"] == "warning"]

    def counts(self) -> dict[str, int]:
        """Number of findings per check code (codes with no finding are absent)."""
        return {str(k): int(v) for k, v in self.findings["code"].value_counts(sort=False).items()}

    def for_hole(self, hole_id: str) -> pd.DataFrame:
        return self.findings[self.findings["hole_id"] == hole_id]

    def with_code(self, code: str) -> pd.DataFrame:
        return self.findings[self.findings["code"] == code]


class _Findings:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str, object, object, str]] = []

    def add(self, code: str, table: str, hole_id: object, row: int | None, message: str) -> None:
        hole = None if pd.isna(hole_id) else str(hole_id)
        self.rows.append((CHECKS[code][0], code, table, hole, row, message))

    def frame(self) -> pd.DataFrame:
        frame = pd.DataFrame(self.rows, columns=list(_FINDING_COLUMNS))
        for column in ("severity", "code", "table", "hole_id", "message"):
            frame[column] = frame[column].astype("str")
        frame["row"] = pd.array([r for *_, r, _ in self.rows], dtype="Int64")
        return frame


def _g(value: float) -> str:
    return f"{value:.10g}"


def _ids(table: pd.DataFrame) -> np.ndarray:
    return table["hole_id"].to_numpy(dtype=object)


def _floats(table: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(table[column], errors="coerce").to_numpy(dtype=np.float64)


def _check_missing_collar(
    found: _Findings, name: str, ids: np.ndarray, known: np.ndarray
) -> None:
    orphan = pd.Series(np.flatnonzero(~known))
    if orphan.empty:
        return
    keys = pd.Series(ids[~known]).fillna("").to_numpy(dtype=object)
    for hole, rows in orphan.groupby(keys, sort=False):
        found.add(
            "missing_collar",
            name,
            hole,
            int(rows.iloc[0]),
            f"{len(rows)} {name} row(s) for hole {hole!r}, which is not in the collar table",
        )


def _check_collar(
    found: _Findings, collar: pd.DataFrame, outlier_factor: float, min_outlier_holes: int
) -> None:
    ids = _ids(collar)
    xyz = np.column_stack([_floats(collar, c) for c in ("x", "y", "z")])
    depth = _floats(collar, "max_depth")

    duplicate = pd.Series(ids).duplicated().to_numpy()
    for i in np.flatnonzero(duplicate):
        found.add("duplicate_hole_id", "collar", ids[i], int(i),
                  f"hole_id {ids[i]!r} is already used by an earlier collar row")

    finite = np.isfinite(xyz).all(axis=1)
    for i in np.flatnonzero(~finite):
        values = ", ".join(f"{a}={_g(v)}" for a, v in zip("xyz", xyz[i], strict=True))
        found.add("non_finite_coordinate", "collar", ids[i], int(i),
                  f"collar coordinate is not finite ({values})")

    with np.errstate(invalid="ignore"):
        bad_depth = ~(np.isfinite(depth) & (depth > 0))
    for i in np.flatnonzero(bad_depth):
        found.add("invalid_max_depth", "collar", ids[i], int(i),
                  f"max_depth is {_g(depth[i])}; it must be a positive length")

    use = np.flatnonzero(finite & ~duplicate)
    if use.size < max(min_outlier_holes, 3):
        return
    p = xyz[use]
    centre = np.median(p, axis=0)
    plan = np.hypot(p[:, 0] - centre[0], p[:, 1] - centre[1])
    rise = np.abs(p[:, 2] - centre[2])
    plan_scale = float(np.median(plan))
    # Flat ground has zero elevation spread; fall back to a small share of the plan spread.
    rise_scale = max(float(np.median(rise)), 1e-3 * plan_scale)
    far_plan = plan > outlier_factor * plan_scale if plan_scale > 0 else plan > 0
    far_rise = rise > outlier_factor * rise_scale if rise_scale > 0 else rise > 0
    for k in np.flatnonzero(far_plan | far_rise):
        parts = []
        if far_plan[k]:
            parts.append(
                f"{_g(plan[k])} in plan from the median collar "
                f"(typical distance {_g(plan_scale)})"
            )
        if far_rise[k]:
            parts.append(
                f"{_g(rise[k])} in elevation from the median collar "
                f"(typical difference {_g(rise_scale)})"
            )
        i = int(use[k])
        found.add("collar_outlier", "collar", ids[i], i, "collar is " + " and ".join(parts))


def _check_survey(
    found: _Findings,
    survey: pd.DataFrame,
    collar_ids: pd.Index,
    max_depth: pd.Series,
    max_rate: float,
    metres_per_unit: float,
    tol: float,
) -> None:
    ids = _ids(survey)
    depth, dip, azimuth = (_floats(survey, c) for c in ("depth", "dip", "azimuth"))
    known = collar_ids.get_indexer(ids) >= 0
    _check_missing_collar(found, "survey", ids, known)

    surveyed = set(pd.unique(ids))
    for hole in collar_ids:
        if hole not in surveyed:
            found.add("no_survey", "survey", hole, None, f"hole {hole!r} has no survey station")

    with np.errstate(invalid="ignore"):
        valid = np.isfinite(depth) & np.isfinite(dip) & np.isfinite(azimuth) & (depth >= 0)
        bad_dip = valid & (np.abs(dip) > 90.0)
        bad_azimuth = valid & ((azimuth < 0.0) | (azimuth > 360.0))
    for i in np.flatnonzero(~valid):
        found.add(
            "invalid_survey_station", "survey", ids[i], int(i),
            f"unusable station: depth={_g(depth[i])}, dip={_g(dip[i])}, "
            f"azimuth={_g(azimuth[i])}",
        )
    for i in np.flatnonzero(bad_dip):
        found.add("dip_out_of_range", "survey", ids[i], int(i),
                  f"dip {_g(dip[i])} at depth {_g(depth[i])} is outside -90..90")
    for i in np.flatnonzero(bad_azimuth):
        found.add("azimuth_out_of_range", "survey", ids[i], int(i),
                  f"azimuth {_g(azimuth[i])} at depth {_g(depth[i])} is outside 0..360")

    limit = pd.Series(ids).map(max_depth).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        past = valid & (depth > limit + tol)
    for i in np.flatnonzero(past):
        found.add(
            "survey_past_max_depth", "survey", ids[i], int(i),
            f"station at depth {_g(depth[i])} is below max_depth {_g(limit[i])}",
        )

    # Per-hole sequence checks on the usable stations, sorted by depth (stable: file order).
    rows = np.flatnonzero(valid)
    if rows.size == 0:
        return
    codes = pd.factorize(pd.Series(ids[rows]).fillna(""))[0]
    order = np.lexsort((depth[rows], codes))
    rows, codes = rows[order], codes[order]
    d = depth[rows]
    same = np.zeros(rows.size, dtype=bool)
    same[1:] = codes[1:] == codes[:-1]

    first = ~same
    for k in np.flatnonzero(first & (d > tol)):
        i = int(rows[k])
        found.add("no_survey_at_zero", "survey", ids[i], None,
                  f"first survey station of hole {ids[i]!r} is at depth {_g(d[k])}, not 0")

    duplicate = np.zeros(rows.size, dtype=bool)
    duplicate[1:] = same[1:] & (d[1:] == d[:-1])
    for k in np.flatnonzero(duplicate):
        i = int(rows[k])
        found.add("duplicate_survey_depth", "survey", ids[i], i,
                  f"depth {_g(d[k])} already has a station (row {int(rows[k - 1])})")

    # Deviation between consecutive good stations. The dip sign convention does not change
    # the angle between two directions, so the default is used.
    good = ~duplicate & ~bad_dip[rows] & ~bad_azimuth[rows]
    rows, codes, d = rows[good], codes[good], d[good]
    if rows.size < 2:
        return
    direction = direction_vectors(dip[rows], azimuth[rows]).reshape(-1, 3)
    cos = np.clip(np.einsum("ij,ij->i", direction[1:], direction[:-1]), -1.0, 1.0)
    cross = np.linalg.norm(np.cross(direction[1:], direction[:-1]), axis=1)
    dogleg = np.degrees(np.arctan2(cross, cos))
    step = (d[1:] - d[:-1]) * metres_per_unit
    with np.errstate(divide="ignore", invalid="ignore"):
        rate = np.where(step > 0, dogleg / step * 30.0, 0.0)
    sudden = (codes[1:] == codes[:-1]) & (rate > max_rate)
    for k in np.flatnonzero(sudden):
        i = int(rows[k + 1])
        found.add(
            "sudden_deviation", "survey", ids[i], i,
            f"direction changes {_g(round(dogleg[k], 3))} deg between depths {_g(d[k])} and "
            f"{_g(d[k + 1])} ({_g(round(rate[k], 2))} deg per 30 m, limit {_g(max_rate)})",
        )


def _check_intervals(
    found: _Findings,
    name: str,
    table: pd.DataFrame,
    collar_ids: pd.Index,
    max_depth: pd.Series,
    tol: float,
) -> None:
    ids = _ids(table)
    start, end = _floats(table, "depth_from"), _floats(table, "depth_to")
    known = collar_ids.get_indexer(ids) >= 0
    _check_missing_collar(found, name, ids, known)

    with np.errstate(invalid="ignore"):
        finite = np.isfinite(start) & np.isfinite(end) & (start >= 0)
        reversed_ = finite & (start >= end)
    for i in np.flatnonzero(~finite):
        found.add("invalid_interval_depth", name, ids[i], int(i),
                  f"interval depths are not usable: from={_g(start[i])}, to={_g(end[i])}")
    for i in np.flatnonzero(reversed_):
        found.add("from_ge_to", name, ids[i], int(i),
                  f"depth_from {_g(start[i])} is not less than depth_to {_g(end[i])}")
    valid = finite & ~reversed_

    limit = pd.Series(ids).map(max_depth).to_numpy(dtype=np.float64)
    with np.errstate(invalid="ignore"):
        past = valid & (end > limit + tol)
    for i in np.flatnonzero(past):
        found.add(
            "interval_past_max_depth", name, ids[i], int(i),
            f"interval {_g(start[i])}-{_g(end[i])} ends below max_depth {_g(limit[i])}",
        )

    rows = np.flatnonzero(valid)
    if rows.size < 2:
        return
    codes = pd.factorize(pd.Series(ids[rows]).fillna(""))[0]
    order = np.lexsort((start[rows], codes))
    rows, codes = rows[order], codes[order]
    s, e = start[rows], end[rows]
    reach = pd.Series(e).groupby(codes).cummax().to_numpy()  # deepest end so far in the hole
    same = codes[1:] == codes[:-1]
    before = reach[:-1]
    overlap = same & (s[1:] < before - tol)
    gap = same & (s[1:] > before + tol)
    for k in np.flatnonzero(overlap):
        i = int(rows[k + 1])
        found.add(
            "overlap", name, ids[i], i,
            f"interval {_g(s[k + 1])}-{_g(e[k + 1])} starts before depth {_g(before[k])}, "
            f"where an earlier interval ends (overlap {_g(min(before[k], e[k + 1]) - s[k + 1])})",
        )
    for k in np.flatnonzero(gap):
        i = int(rows[k + 1])
        found.add("gap", name, ids[i], i,
                  f"nothing logged from {_g(before[k])} to {_g(s[k + 1])} "
                  f"({_g(s[k + 1] - before[k])} long)")


def _summary(db: DrillholeDatabase, findings: pd.DataFrame) -> pd.DataFrame:
    collar_ids = pd.unique(db.collar["hole_id"].dropna())
    others = [t["hole_id"].dropna() for t in (db.survey, *db.intervals.values())]
    holes = pd.unique(pd.concat([pd.Series(collar_ids, dtype="str"), *others], ignore_index=True))
    index = pd.Index(holes)

    def per_hole(series: pd.Series) -> np.ndarray:
        return series.value_counts().reindex(index, fill_value=0).to_numpy(dtype=np.int64)

    summary = pd.DataFrame({"hole_id": pd.Series(holes, dtype="str")})
    summary["in_collar"] = index.isin(collar_ids)
    summary["survey_stations"] = per_hole(db.survey["hole_id"])
    interval_rows = np.zeros(len(index), dtype=np.int64)
    for table in db.intervals.values():
        interval_rows += per_hole(table["hole_id"])
    summary["interval_rows"] = interval_rows
    for severity, column in zip(SEVERITIES, ("errors", "warnings", "infos"), strict=True):
        summary[column] = per_hole(findings.loc[findings["severity"] == severity, "hole_id"])
    status = np.where(
        summary["errors"] > 0,
        "error",
        np.where(summary["warnings"] > 0, "warning", np.where(summary["infos"] > 0, "info", "ok")),
    )
    summary["status"] = pd.Series(status, dtype="str")
    codes = findings.groupby("hole_id")["code"].agg(lambda c: ",".join(sorted(set(c))))
    summary["codes"] = codes.reindex(index).fillna("").to_numpy(dtype=object)
    summary["codes"] = summary["codes"].astype("str")
    return summary[list(_SUMMARY_COLUMNS)]


def validate(
    db: DrillholeDatabase,
    *,
    max_deviation_deg_per_30m: float = 10.0,
    depth_tolerance: float = 1e-6,
    outlier_factor: float = 10.0,
    min_outlier_holes: int = 5,
) -> ValidationReport:
    """Run every check of :data:`CHECKS` on a database. The database is not modified.

    ``max_deviation_deg_per_30m``  threshold of ``sudden_deviation`` (degrees of direction
                                   change per 30 m of hole; 30 m is converted to the
                                   database's length unit).
    ``depth_tolerance``            slack, in the database's length unit, for depth
                                   comparisons (overlap, gap, past max_depth, depth 0).
    ``outlier_factor``             ``collar_outlier`` flags distances above this many times
                                   the median distance from the median collar.
    ``min_outlier_holes``          the outlier test is skipped with fewer usable collars.
    """
    if max_deviation_deg_per_30m <= 0 or outlier_factor <= 0 or depth_tolerance < 0:
        raise ValueError("thresholds must be positive (depth_tolerance may be 0)")
    found = _Findings()
    collar_ids = pd.Index(pd.unique(db.collar["hole_id"].dropna()))
    first = db.collar.drop_duplicates("hole_id")  # a duplicated hole uses its first row
    max_depth = pd.Series(_floats(first, "max_depth"), index=_ids(first))
    max_depth = max_depth[max_depth.index.notna()]

    _check_collar(found, db.collar, outlier_factor, min_outlier_holes)
    _check_survey(
        found,
        db.survey,
        collar_ids,
        max_depth,
        max_deviation_deg_per_30m,
        LENGTH_UNITS.get(db.length_unit, 1.0),
        depth_tolerance,
    )
    for name, table in db.intervals.items():
        _check_intervals(found, name, table, collar_ids, max_depth, depth_tolerance)

    findings = found.frame()
    return ValidationReport(findings=findings, summary=_summary(db, findings))
