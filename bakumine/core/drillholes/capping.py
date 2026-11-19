"""Grade capping (top-cut): ``research.md`` section 7.7.

Values above a threshold are set to the threshold before compositing or estimation, so a
few extreme assays do not dominate the estimate. The cap is given per attribute and,
optionally, per domain. The uncapped values stay in the table next to the capped ones
(``au`` becomes the capped grade, ``au_uncut`` keeps the original), and a report says how
many values were cut and how much metal that removed.

Works on any table with the attribute columns: an interval table (samples) or a composite
table. "Metal" is ``sum(grade * weight)``; the weight of a row is, in this order, the
column named by ``weight``, the attribute's ``<attribute>_coverage`` column (composites),
``length``, ``depth_to - depth_from``, or 1.

Missing grades stay missing: NaN is neither cut nor counted.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

__all__ = ["ANY_DOMAIN", "REPORT_COLUMNS", "cap_grades"]

#: Key of a per-domain cap mapping that applies to every domain not listed by name.
ANY_DOMAIN = "*"

REPORT_COLUMNS: tuple[str, ...] = (
    "attribute",
    "domain",
    "cap",
    "n_values",
    "n_cut",
    "max_uncut",
    "mean_uncut",
    "mean_cut",
    "metal_uncut",
    "metal_cut",
    "metal_removed",
    "metal_removed_pct",
)


def _weights(table: pd.DataFrame, attribute: str, weight: str | None) -> np.ndarray:
    if weight is not None:
        if weight not in table.columns:
            raise ValueError(f"weight column {weight!r} is not in the table")
        w = table[weight].to_numpy(dtype=np.float64)
    elif f"{attribute}_coverage" in table.columns:
        w = table[f"{attribute}_coverage"].to_numpy(dtype=np.float64)
    elif "length" in table.columns:
        w = table["length"].to_numpy(dtype=np.float64)
    elif "depth_from" in table.columns and "depth_to" in table.columns:
        w = (table["depth_to"] - table["depth_from"]).to_numpy(dtype=np.float64)
    else:
        w = np.ones(len(table))
    return np.where(np.isfinite(w) & (w > 0.0), w, 0.0)


def _check_cap(attribute: str, cap: object) -> float:
    try:
        value = float(cap)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        raise ValueError(f"cap of {attribute!r} must be a number, got {cap!r}") from None
    if not np.isfinite(value):
        raise ValueError(f"cap of {attribute!r} must be finite, got {cap!r}")
    return value


def _report_row(
    attribute: str,
    domain: object,
    cap: float,
    uncut: np.ndarray,
    weight: np.ndarray,
    rows: np.ndarray,
) -> dict[str, object]:
    use = rows & np.isfinite(uncut)
    v, w = uncut[use], weight[use]
    cut = np.minimum(v, cap)
    metal_uncut, metal_cut = float((v * w).sum()), float((cut * w).sum())
    total = float(w.sum())
    return {
        "attribute": attribute,
        "domain": domain,
        "cap": cap,
        "n_values": int(use.sum()),
        "n_cut": int((v > cap).sum()),
        "max_uncut": float(v.max()) if v.size else np.nan,
        "mean_uncut": metal_uncut / total if total > 0 else np.nan,
        "mean_cut": metal_cut / total if total > 0 else np.nan,
        "metal_uncut": metal_uncut,
        "metal_cut": metal_cut,
        "metal_removed": metal_uncut - metal_cut,
        "metal_removed_pct": (
            100.0 * (metal_uncut - metal_cut) / metal_uncut if metal_uncut != 0 else 0.0
        ),
    }


def cap_grades(
    table: pd.DataFrame,
    caps: Mapping[str, float | Mapping[object, float]],
    *,
    domain: str | None = None,
    weight: str | None = None,
    suffix: str = "_uncut",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Top-cut attribute columns. Returns ``(capped table, report)``; the input is not changed.

    ``caps``    ``{attribute: cap}`` or ``{attribute: {domain code: cap}}``. In a per-domain
                mapping the key ``"*"`` is the cap of every domain not listed (a missing
                domain code included); without it, unlisted domains are not capped.
    ``domain``  name of the domain column; needed for per-domain caps.
    ``weight``  name of the column that weights the metal in the report (see the module
                docstring for the default).
    ``suffix``  the original values are kept in ``<attribute><suffix>``, placed right after
                the attribute. If that column already exists the cap is applied to it again
                (so re-capping with another threshold starts from the uncut grades).

    The report has one row per attribute (and per capped domain) with the columns of
    :data:`REPORT_COLUMNS`: ``n_values`` non-missing values, ``n_cut`` of them above the
    cap, weighted means before and after, and ``metal_removed = metal_uncut - metal_cut``.
    """
    if not suffix:
        raise ValueError("suffix must not be empty")
    out = table.copy()
    report: list[dict[str, object]] = []
    for attribute, cap in caps.items():
        attribute = str(attribute)
        if attribute not in table.columns:
            raise ValueError(f"column {attribute!r} is not in the table")
        if not pd.api.types.is_numeric_dtype(table[attribute]):
            raise ValueError(f"column {attribute!r} is not numeric and cannot be capped")
        keep = f"{attribute}{suffix}"
        source = keep if keep in table.columns else attribute
        uncut = table[source].to_numpy(dtype=np.float64)
        w = _weights(table, attribute, weight)
        limit = np.full(len(table), np.inf)
        if isinstance(cap, Mapping):
            if domain is None:
                raise ValueError(f"per-domain caps of {attribute!r} need the domain column name")
            if domain not in table.columns:
                raise ValueError(f"domain column {domain!r} is not in the table")
            codes = table[domain]
            listed = np.zeros(len(table), dtype=bool)
            for code, value in cap.items():
                if code == ANY_DOMAIN:
                    continue
                value = _check_cap(attribute, value)
                rows = (codes == code).fillna(False).to_numpy(dtype=bool)
                listed |= rows
                limit[rows] = value
                report.append(_report_row(attribute, code, value, uncut, w, rows))
            if ANY_DOMAIN in cap:
                value = _check_cap(attribute, cap[ANY_DOMAIN])
                limit[~listed] = value
                report.append(_report_row(attribute, ANY_DOMAIN, value, uncut, w, ~listed))
        else:
            value = _check_cap(attribute, cap)
            limit[:] = value
            everything = np.ones(len(table), dtype=bool)
            report.append(_report_row(attribute, None, value, uncut, w, everything))
        out[attribute] = np.minimum(uncut, limit)  # NaN stays NaN
        if keep not in out.columns:
            out.insert(out.columns.get_loc(attribute) + 1, keep, uncut)
    frame = pd.DataFrame(report, columns=list(REPORT_COLUMNS))
    return out, frame
