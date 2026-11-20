"""Volume, tonnes, grade and metal report (research.md section 8).

For the blocks selected by a constraint, per group::

    volume  = sum of  V_i * p_i                 p = partial fraction (1 if none)
    tonnes  = sum of  V_i * p_i * rho_i         rho = density, t/m3
    grade   = sum of (t_i * g_i) / sum of t_i   over the blocks that HAVE a grade
    metal   = sum of  t_i * g_i                 over the same blocks

Missing stays missing
---------------------
A block whose grade is NaN is **not** counted as zero grade. Its tonnes are
reported in the separate column ``<grade>_unestimated_tonnes`` and are left
out of the grade average and of the metal. So ``<grade>_grade`` is the mean
grade of the estimated tonnes (``tonnes - <grade>_unestimated_tonnes``), and
``metal = estimated tonnes * grade``. With nothing estimated the grade is NaN.

The same holds for density: a block whose density is NaN has no tonnes; its
volume is reported in ``volume_without_density``.

Grouping
--------
``group_by``
    One or more category or integer attributes (rock code, pass number...).
    A missing category is its own group (shown as ``None``).
bench
    ``bench_height`` slices the elevation into benches
    ``[reference + n * height, reference + (n + 1) * height)``. **A block
    belongs to the bench that contains its centroid** (world z); it is never
    divided between benches. A block (or sub-block) that straddles a bench
    floor is therefore counted whole in the bench of its centroid, and a
    centroid exactly on a floor belongs to the bench above. Choose the height
    and reference to match the block layout if that matters.
cut-offs
    A grade-tonnage table: for each cut-off ``c`` the row holds the blocks
    with ``grade >= c`` (cumulative, "at or above cut-off"). Blocks without a
    value of the cut-off grade cannot be classified: they are in no cut-off
    row, and their tonnes are shown in ``unclassified_tonnes`` (the same
    number on every cut-off row of the group).

Result
------
A tidy :class:`pandas.DataFrame`, one row per group (and cut-off), then the
total row(s) with ``row == "total"``. Columns, in order:

* the ``group_by`` attributes (strings / integers; ``None`` in total rows),
* ``bench_from``, ``bench_to`` (with ``bench_height``),
* ``cutoff`` (with ``cutoffs``),
* ``row`` -- ``"group"`` or ``"total"``,
* ``blocks``, ``volume``,
* ``tonnes``, ``volume_without_density`` (with ``density``),
* ``unclassified_tonnes`` (with ``cutoffs``),
* per grade attribute ``g``: ``g_grade``, the metal column(s) and
  ``g_unestimated_tonnes``.

Metal columns depend on the unit of the grade:

=============  =========================================================
unit           columns
=============  =========================================================
``g/t``        ``g_metal_g`` (grams = tonnes x g/t) and ``g_metal_oz``
(``ppm``)      (troy ounces, 31.1034768 g)
``%``          ``g_metal_t`` (tonnes of metal = tonnes x % / 100)
none           ``g_metal`` = tonnes x grade, in "tonnes x grade unit"
=============  =========================================================

The unit comes from ``units={name: unit}``, else from the attribute's own
``unit``, else none. ``frame.attrs["notes"]`` states the rules that were
applied; ``frame.attrs["units"]`` the unit of each grade.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.blocks import BlockModel

__all__ = ["TROY_OUNCE_G", "ReportError", "format_table", "report", "to_csv"]

#: grams in one troy ounce (exact by definition)
TROY_OUNCE_G = 31.1034768

_UNITS = {"g/t": "g/t", "gpt": "g/t", "ppm": "g/t", "%": "%", "pct": "%", "percent": "%"}
_PARTIAL_SLACK = 1e-9


class ReportError(ValueError):
    """The report cannot be made as asked."""


def _unit(text: Any) -> str | None:
    if text is None or str(text).strip() == "":
        return None
    try:
        return _UNITS[str(text).strip().lower()]
    except KeyError:
        return ""  # an unknown unit: plain tonnes x grade


def _names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return [str(v) for v in value]


def report(
    model: BlockModel,
    *,
    constraint: Any = None,
    resolver: Any = None,
    where: Any = None,
    group_by: str | Iterable[str] = (),
    cutoffs: Sequence[float] | None = None,
    cutoff_attribute: str | None = None,
    grade_attributes: str | Iterable[str] = (),
    density: str | float | None = None,
    partial: str | None = None,
    bench_height: float | None = None,
    bench_reference: float = 0.0,
    units: Mapping[str, str] | None = None,
    totals: bool = True,
) -> pd.DataFrame:
    """Volume / tonnes / grade / metal of the selected blocks; see the module docstring.

    Parameters
    ----------
    constraint, resolver : optional
        :class:`~bakumine.core.blockmodel.constraints.Constraint` selecting blocks.
    where : optional bool mask over the block table, ANDed with the constraint.
    group_by : category or integer attribute name(s).
    cutoffs : grade cut-offs for a grade-tonnage table (``grade >= cutoff``).
    cutoff_attribute : the grade the cut-offs apply to; default the first of
        ``grade_attributes``.
    grade_attributes : float attribute name(s) to average, tonnage-weighted.
    density : attribute name, or a constant in t/m3. Without it there are no
        tonnes, and grades cannot be reported.
    partial : name of an attribute holding the fraction (0..1) of each block
        to count, e.g. the fraction inside the ore solid; default the whole block.
    bench_height, bench_reference : group by bench (by block centroid).
    units : grade name -> ``"g/t"``, ``"ppm"``, ``"%"`` or ``""`` (none);
        overrides the attributes' own units.
    totals : append the total row(s).
    """
    group_names = _names(group_by)
    grades = _names(grade_attributes)
    if len(set(grades)) != len(grades) or len(set(group_names)) != len(group_names):
        raise ReportError("an attribute is listed twice")
    for name in (*group_names, *grades):
        model.attribute(name)
    if grades and density is None:
        raise ReportError(
            "grades are tonnage-weighted: give the density (an attribute name or a constant)"
        )
    notes: list[str] = []

    # ------------------------------------------------------------ selection
    mask = np.ones(model.n_blocks, dtype=bool)
    if constraint is not None:
        mask &= np.asarray(constraint.mask(model, resolver), dtype=bool)
    if where is not None:
        sel = np.asarray(where)
        if sel.dtype != bool or sel.shape != mask.shape:
            raise ReportError("where must be a boolean mask with one value per block")
        mask &= sel
    rows = np.flatnonzero(mask)
    n = len(rows)

    # ----------------------------------------------------- volume and tonnes
    volume = model.volumes(rows)
    if partial is not None:
        spec = model.attribute(partial)
        if spec.kind != "float":
            raise ReportError(f"partial attribute {partial!r} must be a float fraction (0..1)")
        fraction = model.get(partial, rows).astype(np.float64)
        bad = ~((fraction >= 0.0) & (fraction <= 1.0 + _PARTIAL_SLACK))  # NaN is bad too
        if bad.any():
            n_nan = int(np.isnan(fraction).sum())
            raise ReportError(
                f"partial attribute {partial!r}: {int(bad.sum())} selected block(s) are not a "
                f"fraction between 0 and 1 ({n_nan} missing); fill or constrain them first"
            )
        volume = volume * fraction
        notes.append(f"volume = block volume x {partial}")
    tonnes = no_density = None
    if density is not None:
        if isinstance(density, str):
            if model.attribute(density).kind != "float":
                raise ReportError(f"density attribute {density!r} must be a float attribute")
            rho = model.get(density, rows).astype(np.float64)
            notes.append(f"density from attribute {density}")
        else:
            rho = np.full(n, float(density))
            notes.append(f"constant density {float(density):g} t/m3")
        if (rho <= 0.0).any() or np.isinf(rho).any():
            raise ReportError("density must be positive and finite (or NaN for unknown)")
        known = ~np.isnan(rho)
        tonnes = np.where(known, volume * np.where(known, rho, 0.0), 0.0)
        no_density = np.where(known, 0.0, volume)
        if not known.all():
            notes.append(
                "blocks without density have no tonnes; their volume is in volume_without_density"
            )

    grade_values = {g: _grade(model, g, rows) for g in grades}
    unit_of: dict[str, str | None] = {}
    for g in grades:
        given = (units or {}).get(g)
        unit_of[g] = _unit(model.attribute(g).unit if given is None else given) or None
        if given not in (None, "") and unit_of[g] is None:
            raise ReportError(f"unit {given!r} of {g!r} is not one of g/t, ppm, %")

    # --------------------------------------------------------------- cut-offs
    cut_values: list[float] = []
    cut_grade = None
    if cutoffs is not None:
        cut_values = sorted({float(c) for c in cutoffs})
        if not cut_values or not np.isfinite(cut_values).all():
            raise ReportError("cutoffs must be one or more finite numbers")
        cut_name = cutoff_attribute or (grades[0] if grades else None)
        if cut_name is None:
            raise ReportError("cut-offs need a grade: give cutoff_attribute or grade_attributes")
        cut_grade = grade_values.get(cut_name)
        if cut_grade is None:
            cut_grade = _grade(model, cut_name, rows)
        notes.append(
            f"cut-off rows hold the blocks with {cut_name} >= cutoff (cumulative); blocks "
            f"without {cut_name} are in no cut-off row"
            + (", their tonnes are in unclassified_tonnes" if tonnes is not None else "")
        )
    elif cutoff_attribute is not None:
        raise ReportError("cutoff_attribute given without cutoffs")

    # ----------------------------------------------------------------- groups
    keys: list[tuple[str, np.ndarray, Any]] = []  # (column, code per block, decoder)
    for name in group_names:
        spec = model.attribute(name)
        if spec.kind == "float":
            raise ReportError(
                f"cannot group by the float attribute {name!r}; use cutoffs, or group by a "
                "category or integer attribute"
            )
        codes = model.get(name, rows).astype(np.int64)
        if spec.kind == "category":
            table = np.array([*spec.categories, None], dtype=object)
            keys.append((name, codes, lambda c, table=table: table[c]))
        else:
            keys.append((name, codes, lambda c: c.astype(object)))
    if bench_height is not None:
        height = float(bench_height)
        if not (np.isfinite(height) and height > 0.0):
            raise ReportError("bench_height must be a positive number")
        reference = float(bench_reference)
        z = model.centroids(rows)[:, 2]
        keys.append(("bench", np.floor((z - reference) / height).astype(np.int64), None))
        notes.append(
            f"benches of {height:g} m from elevation {reference:g}; a block is counted whole "
            "in the bench that contains its centroid"
        )
    group = np.zeros(n, dtype=np.int64)
    uniques: list[np.ndarray] = []
    for _, codes, _ in keys:
        unique, inverse = np.unique(codes, return_inverse=True)
        group = group * len(unique) + inverse
        uniques.append(unique)
    present, group = np.unique(group, return_inverse=True)
    n_groups = len(present) if keys else 0
    key_columns: dict[str, np.ndarray] = {}
    rest = present.copy()
    for (name, _, decode), unique in zip(reversed(keys), reversed(uniques), strict=True):
        values = unique[rest % len(unique)]
        rest //= len(unique)
        if name == "bench":
            key_columns["bench_to"] = reference + (values + 1) * height
            key_columns["bench_from"] = reference + values * height
        else:
            key_columns[name] = decode(values)
    key_columns = dict(reversed(list(key_columns.items())))

    # ------------------------------------------------------------ aggregation
    def aggregate(index: np.ndarray, size: int, keep: np.ndarray | None) -> dict[str, np.ndarray]:
        def total(weights: np.ndarray | None, extra: np.ndarray | None = None) -> np.ndarray:
            use = keep if extra is None else (extra if keep is None else keep & extra)
            if use is not None:
                idx = index[use]
                weights = None if weights is None else weights[use]
            else:
                idx = index
            counts = np.bincount(idx, weights=weights, minlength=size)
            return counts if weights is None else counts.astype(np.float64)

        out: dict[str, np.ndarray] = {
            "blocks": total(None).astype(np.int64),
            "volume": total(volume),
        }
        if tonnes is not None:
            out["tonnes"] = total(tonnes)
            out["volume_without_density"] = total(no_density)
            if cut_grade is not None:
                # not restricted to the cut-off: everything in the group that has no grade
                unknown = np.isnan(cut_grade)
                out["unclassified_tonnes"] = np.bincount(
                    index[unknown], weights=tonnes[unknown], minlength=size
                ).astype(np.float64)
        for g in grades:
            values = grade_values[g]
            has = ~np.isnan(values)
            estimated = total(tonnes, has)
            product = total(tonnes * np.where(has, values, 0.0), has)
            unestimated = total(tonnes, ~has)
            with np.errstate(invalid="ignore", divide="ignore"):
                out[f"{g}_grade"] = np.where(estimated > 0.0, product / estimated, np.nan)
            # nothing estimated but tonnes present: the metal is unknown, not zero
            metal = np.where((estimated == 0.0) & (unestimated > 0.0), np.nan, product)
            unit = unit_of[g]
            if unit == "g/t":
                out[f"{g}_metal_g"] = metal
                out[f"{g}_metal_oz"] = metal / TROY_OUNCE_G
            elif unit == "%":
                out[f"{g}_metal_t"] = metal / 100.0
            else:
                out[f"{g}_metal"] = metal
            out[f"{g}_unestimated_tonnes"] = unestimated
        return out

    cut_list: list[float | None] = cut_values if cut_values else [None]
    frames: list[pd.DataFrame] = []
    for kind, index, size in (
        ("group", group, n_groups),
        ("total", np.zeros(n, dtype=np.int64), 1 if totals or not keys else 0),
    ):
        if size == 0:
            continue
        parts = []
        for cut in cut_list:
            keep = None if cut is None else cut_grade >= cut
            data: dict[str, Any] = {}
            for name, values in key_columns.items():
                if kind == "group":
                    data[name] = values
                else:
                    data[name] = np.full(size, np.nan) if name.startswith("bench_") else [None]
            if cut is not None:
                data["cutoff"] = np.full(size, cut)
            data["row"] = [kind] * size
            data.update(aggregate(index, size, keep))
            parts.append(pd.DataFrame(data))
        frame = pd.concat(parts, ignore_index=True)
        if kind == "group" and len(cut_list) > 1:
            # group by group, cut-offs ascending inside each
            order = np.arange(len(frame)).reshape(len(cut_list), size).T.reshape(-1)
            frame = frame.iloc[order]
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    for name in key_columns:
        if not name.startswith("bench_"):  # plain Python values; None = missing / total
            values = [None if v is None or v != v else v for v in result[name].tolist()]
            result[name] = pd.Series(values, dtype=object, index=result.index)
    result.attrs["notes"] = notes
    result.attrs["units"] = {g: unit_of[g] or "" for g in grades}
    return result


def _grade(model: BlockModel, name: str, rows: np.ndarray) -> np.ndarray:
    if model.attribute(name).kind != "float":
        raise ReportError(f"grade attribute {name!r} must be a float attribute")
    return model.get(name, rows).astype(np.float64)


# -------------------------------------------------------------------- output


def to_csv(frame: pd.DataFrame, path: str | Path, *, missing: str = "") -> Path:
    """Write a report to CSV. NaN (e.g. the grade of nothing) is written as ``missing``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, na_rep=missing, lineterminator="\n")
    return path


def _decimals_for(column: str, decimals: Mapping[str, int]) -> int:
    if column in decimals:
        return decimals[column]
    for suffix, value in (("_grade", "grade"), ("cutoff", "cutoff")):
        if column.endswith(suffix):
            return decimals.get(value, 3)
    if column.startswith("bench_"):
        return decimals.get("bench", 1)
    return decimals.get("quantity", 0)


def format_table(frame: pd.DataFrame, *, decimals: Mapping[str, int] | None = None) -> str:
    """A report as a plain-text table.

    ``decimals`` maps a column name -- or ``"grade"``, ``"cutoff"``, ``"bench"``,
    ``"quantity"`` (volumes, tonnes, metal) -- to a number of decimals. Defaults:
    grades and cut-offs 3, benches 1, quantities 0 with thousands separators.
    Total rows are labelled ``TOTAL``; missing values are shown as ``-``.
    """
    decimals = dict(decimals or {})
    columns = [c for c in frame.columns if c != "row"]
    is_total = (frame["row"] == "total").to_numpy() if "row" in frame else np.zeros(len(frame))
    cells: list[list[str]] = []
    numeric: list[bool] = []
    for position, column in enumerate(columns):
        series = frame[column]
        number = pd.api.types.is_numeric_dtype(series.dtype)
        numeric.append(number)
        out = []
        for value, total in zip(series.tolist(), is_total, strict=True):
            if position == 0 and total and (value is None or value != value):
                out.append("TOTAL")
            elif value is None or (isinstance(value, float) and value != value):
                out.append("-")
            elif number and series.dtype.kind in "iu":
                out.append(f"{value:,d}")
            elif number:
                out.append(f"{value:,.{_decimals_for(column, decimals)}f}")
            else:
                out.append(str(value))
        cells.append(out)
    widths = [max(len(c), *(len(v) for v in col)) for c, col in zip(columns, cells, strict=True)]

    def line(values: Sequence[str]) -> str:
        parts = [
            v.rjust(w) if num else v.ljust(w)
            for v, w, num in zip(values, widths, numeric, strict=True)
        ]
        return "  ".join(parts).rstrip()

    lines = [line(columns), line(["-" * w for w in widths])]
    lines += [line([col[r] for col in cells]) for r in range(len(frame))]
    return "\n".join(lines)
