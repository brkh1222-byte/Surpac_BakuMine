"""From spreadsheets as exploration teams keep them to one plain table per source.

A *grid* is a sheet (or a CSV file) read without interpreting anything: every cell is
text, there is no header. This module finds the header in a grid, stacks the sheets of a
workbook, and applies the row rules of one source of an import recipe
(:mod:`bakumine.core.drillholes.recipe`):

* :func:`read_grids`        - every sheet of an XLSX / XLS workbook, or a CSV, as grids;
* :func:`locate_header`     - the header row (not always the first), one or two rows
                              (a merged group header ``Interval`` over ``from`` / ``to``),
                              units rows and column-number rows below it;
* :func:`table_from_grid`   - the data rows under that header;
* :func:`normalise_hole_ids` - ``DH_02``, ``DH-002`` -> one spelling, with a pattern;
* :func:`prepare_source`    - all of the above for one source of a recipe.

Nothing here parses assay values: the result is still text, typed by
:func:`bakumine.core.drillholes.importer.import_table`.
"""

from __future__ import annotations

import csv
import fnmatch
import io
import re
import string
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .importer import (
    _SYNONYMS,
    _cell_text,
    _normalise_header,
    _resolve_column_map,
    suggest_column_map,
)

__all__ = [
    "Header",
    "Prepared",
    "locate_header",
    "normalise_hole_ids",
    "prepare_source",
    "read_grids",
    "select_sheets",
    "table_from_grid",
]

SHEET_COLUMN = "__sheet__"
_HEADER_SCAN_ROWS = 60
_NUMERIC = re.compile(r"^[<>]?\s*[+-]?(?:\d+[.,]?\d*|[.,]\d+)(?:[eE][+-]?\d+)?$")
_UNITS = frozenset(
    {
        "m", "м", "ft", "g/t", "г/т", "ppm", "ppb", "%", "deg", "°", "град", "градус",
        "мг/кг", "mg/kg", "kg", "кг", "t", "т", "g", "г", "m3", "м3", "t/m3", "т/м3", "cm", "см",
        "mm", "мм", "wt%", "pct", "oz/t", "gpt",
    }
)  # fmt: skip
_KNOWN_HEADERS = frozenset(
    s for kind in _SYNONYMS.values() for names in kind.values() for s in names
)


# --------------------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------------------
def _grid(frame: pd.DataFrame) -> pd.DataFrame:
    """Any DataFrame of cells -> text grid with integer columns and index."""
    cells = {i: [_cell_text(v) for v in frame.iloc[:, i].tolist()] for i in range(frame.shape[1])}
    return pd.DataFrame(cells, columns=range(frame.shape[1]), dtype=object).reset_index(drop=True)


def _read_csv_grid(path: Path, delimiter: str | None, encoding: str) -> pd.DataFrame:
    text = path.read_text(encoding=encoding)
    if delimiter is None:
        try:
            delimiter = csv.Sniffer().sniff(text[:65536], delimiters=",;\t|").delimiter
        except csv.Error:
            delimiter = ","
    rows = list(csv.reader(io.StringIO(text, newline=""), delimiter=delimiter))
    width = max((len(r) for r in rows), default=0)
    padded = [r + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, columns=range(width), dtype=object)


def read_grids(
    path: str | Path, *, delimiter: str | None = ",", encoding: str = "utf-8-sig"
) -> dict[str, pd.DataFrame]:
    """Every sheet of a workbook as a text grid, in workbook order. A CSV file is one grid
    under the name ``""``. Rows of a CSV may have different lengths (title lines)."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".xlsx", ".xlsm", ".xls"):
        sheets = pd.read_excel(
            path,
            sheet_name=None,
            header=None,
            dtype=object,
            na_filter=False,
            engine="xlrd" if suffix == ".xls" else "openpyxl",
        )
        return {str(name): _grid(frame) for name, frame in sheets.items()}
    if suffix in (".csv", ".txt", ".tsv", ".dat"):
        if suffix == ".tsv" and delimiter == ",":
            delimiter = "\t"
        return {"": _read_csv_grid(path, delimiter, encoding)}
    raise ValueError(f"unsupported table file type {suffix!r}: {path}")


def select_sheets(
    names: Sequence[str],
    sheets: str | int | Sequence[str | int] | None = None,
    exclude: Sequence[str] | None = None,
) -> list[str]:
    """Sheet names chosen by ``sheets``: None = the first sheet; a name, a 0-based index or a
    wildcard pattern (``"DH*"``, ``"*"``), or a list of these. Workbook order is kept."""
    names = list(names)
    if sheets is None:
        chosen = names[:1]
    else:
        wanted = [sheets] if isinstance(sheets, str | int) else list(sheets)
        chosen = []
        for item in wanted:
            if isinstance(item, int) and not isinstance(item, bool):
                if not 0 <= item < len(names):
                    raise ValueError(f"sheet index {item} is outside 0..{len(names) - 1}")
                found = [names[item]]
            elif str(item) in names:
                found = [str(item)]
            else:
                found = [n for n in names if fnmatch.fnmatchcase(n, str(item))]
                if not found:
                    raise ValueError(f"no sheet matches {item!r}; the file has {names}")
            chosen += [n for n in found if n not in chosen]
        chosen.sort(key=names.index)
    dropped = [p for p in (exclude or [])]
    return [n for n in chosen if not any(n == p or fnmatch.fnmatchcase(n, p) for p in dropped)]


# --------------------------------------------------------------------------------------
# header
# --------------------------------------------------------------------------------------
@dataclass
class Header:
    """Where the header of a grid is and what it says.

    ``row``        0-based index of the first header row (-1: the grid has no header).
    ``n_rows``     1 or 2.
    ``names``      one name per grid column ("" for a column that is blank throughout).
    ``data_start`` 0-based index of the first data row (after units / numbering rows).
    ``skipped``    what the rows between the header and the data are (``"units"``, ...).
    """

    row: int
    n_rows: int
    names: list[str]
    data_start: int
    skipped: list[str] = field(default_factory=list)


def _is_number(text: str) -> bool:
    return bool(_NUMERIC.match(text.strip()))


def _is_unit(text: str) -> bool:
    return re.sub(r"[\s()\[\]]+", "", text.casefold()).strip("/,.") in _UNITS


def _row(grid: pd.DataFrame, index: int) -> list[str]:
    return [str(v).strip() for v in grid.iloc[index].tolist()]


def _known(cells: Iterable[str]) -> int:
    return sum(1 for c in cells if c and _normalise_header(c) in _KNOWN_HEADERS)


def _is_sub_header(top: list[str], below: list[str]) -> bool:
    """Is ``below`` the second row of a two-row header under ``top``?"""
    filled = [c for c in below if c]
    if not any(top) or not filled or any(_is_number(c) for c in filled):
        return False
    if all(_is_unit(c) for c in filled):
        return False
    under_blank = sum(1 for t, b in zip(top, below, strict=True) if b and not t)
    if under_blank == 0:
        return False
    # a first data row of text only has (almost) every column filled; a sub-header does
    # not: the columns whose header spans both rows are empty in it
    spanning = sum(1 for t, b in zip(top, below, strict=True) if t and not b)
    return spanning > 0 or _known(filled) > 0


def _numbering_row(cells: list[str]) -> bool:
    filled = [c for c in cells if c]
    if len(filled) < 3 or not all(c.isdigit() for c in filled):
        return False
    numbers = [int(c) for c in filled]
    return numbers == list(range(numbers[0], numbers[0] + len(numbers))) and numbers[0] in (0, 1)


def _names(top: list[str], sub: list[str] | None, has_data: list[bool]) -> list[str]:
    names: list[str] = []
    group, run = "", 1
    for j, cell in enumerate(top):
        if cell:
            group, run = cell, 1
        below = sub[j] if sub else ""
        if below and not (cell and _is_unit(below)):
            name = below
        elif cell:
            name = cell
        elif has_data[j]:
            run += 1
            name = f"{group} {run}" if group else f"column {j + 1}"
        else:
            name = ""
        names.append(name)
    # the same sub-name under two groups ("from" under "Core" and under "Sludge")
    counts = Counter(n for n in names if n)
    if sub:
        group = ""
        for j, cell in enumerate(top):
            group = cell or group
            if names[j] and counts[names[j]] > 1 and sub[j] and group and group != names[j]:
                names[j] = f"{group} {names[j]}"
    used: set[str] = set()
    for j, name in enumerate(names):
        if not name:
            continue
        unique, n = name, 1
        while unique in used:  # also against the names made up for blank header cells
            n += 1
            unique = f"{name} {n}"
        names[j] = unique
        used.add(unique)
    return names


def locate_header(
    grid: pd.DataFrame, header_row: int | None = None, header_rows: int | None = None
) -> Header:
    """Find the header of a grid.

    ``header_row`` is the 0-based row of the (first) header row; None looks for the row
    among the first rows with the most known column names (hole, from, to, east ... in
    English, Russian or Mongolian), else takes the first row with two or more cells.
    ``-1`` means there is no header: the columns are named ``"0", "1", ...``.
    ``header_rows`` is 1 or 2; None decides from the row below (text only, sitting under
    blank cells of a merged group header). Units rows (``m, m, g/t``) and a row of column
    numbers (``1, 2, 3 ...``) right below the header are skipped.
    """
    n = len(grid)
    width = grid.shape[1]
    if header_row is not None and header_row < 0:
        return Header(-1, 0, [str(j) for j in range(width)], 0)
    if n == 0:
        return Header(0, 1, [""] * width, 0)
    if header_row is None:
        best, best_score = None, 1
        first_wide = None
        for i in range(min(n, _HEADER_SCAN_ROWS)):
            cells = _row(grid, i)
            filled = [c for c in cells if c]
            if first_wide is None and len(filled) >= 2:
                first_wide = i
            score = _known(cells)
            if i + 1 < n and _is_sub_header(cells, _row(grid, i + 1)):
                score += _known(_row(grid, i + 1))
            if score > best_score:
                best, best_score = i, score
        header_row = best if best is not None else (first_wide or 0)
    if header_row >= n:
        raise ValueError(f"header row {header_row + 1} is beyond the last row ({n})")
    top = _row(grid, header_row)
    sub: list[str] | None = None
    if header_rows not in (None, 1, 2):
        raise ValueError(f"header_rows must be 1 or 2, got {header_rows!r}")
    if header_row + 1 < n and header_rows != 1:
        below = _row(grid, header_row + 1)
        if header_rows == 2 or _is_sub_header(top, below):
            sub = below
    start = header_row + (2 if sub else 1)
    skipped: list[str] = []
    while start < n and len(skipped) < 3:
        cells = _row(grid, start)
        filled = [c for c in cells if c]
        if not filled:
            start += 1
            continue
        if all(_is_unit(c) for c in filled):
            skipped.append("units")
        elif _numbering_row(cells):
            skipped.append("column numbers")
        else:
            break
        start += 1
    body = grid.iloc[start:]
    has_data = [bool((body[j].astype(str).str.strip() != "").any()) for j in grid.columns]
    return Header(header_row, 2 if sub else 1, _names(top, sub, has_data), start, skipped)


def table_from_grid(grid: pd.DataFrame, header: Header) -> tuple[pd.DataFrame, int]:
    """The data rows of a grid under ``header`` as a text table. Blank rows, columns blank
    throughout, and repeats of the header row among the data are dropped. Returns
    ``(table, n_repeated_headers)``; the table's index is the 0-based grid row."""
    keep = [j for j, name in enumerate(header.names) if name]
    body = grid.iloc[header.data_start :, keep].copy()
    body.columns = [header.names[j] for j in keep]
    for column in body.columns:
        body[column] = body[column].astype(str).str.strip()
    cells = body.to_numpy(dtype=object)
    blank = (cells == "").all(axis=1) if len(body) else np.zeros(0, dtype=bool)
    names = np.array(list(body.columns), dtype=object)
    same = (cells == names) | (cells == "")
    repeated = same.all(axis=1) & ((cells == names).sum(axis=1) >= 2) if len(body) else blank
    return body.loc[~(blank | repeated)], int(repeated.sum())


# --------------------------------------------------------------------------------------
# hole ids
# --------------------------------------------------------------------------------------
class _GroupFormatter(string.Formatter):
    """``{1:02d}`` works on a regex group: the text is made an int for integer formats."""

    def format_field(self, value: Any, format_spec: str) -> str:
        if format_spec and format_spec[-1] in "dxXobn":
            value = int(value)
        elif format_spec and format_spec[-1] in "eEfFgG%":
            value = float(value)
        return super().format_field(value, format_spec)


def normalise_hole_ids(
    ids: pd.Series, pattern: str | None, template: str | None
) -> tuple[pd.Series, list[str]]:
    """Rewrite hole ids that match ``pattern`` (a regular expression, matched against the
    whole id, case-insensitively) as ``template`` (``{0}`` = the whole id, ``{1}``, ``{2}``
    = the groups; ``{1:02d}`` pads a number). Returns ``(ids, ids that did not match)``;
    those are left as they are. Without a pattern the ids are only stripped."""
    ids = ids.astype(str).str.strip()
    if not pattern:
        return ids, []
    try:
        compiled = re.compile(pattern, re.IGNORECASE)
    except re.error as exc:
        raise ValueError(f"hole id pattern {pattern!r}: {exc}") from None
    template = template or "{0}"
    formatter = _GroupFormatter()
    known: dict[str, str | None] = {}
    for value in ids.unique():
        match = compiled.fullmatch(value)
        if match is None:
            known[value] = None
            continue
        try:
            known[value] = formatter.format(template, match.group(0), *match.groups(default=""))
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise ValueError(f"hole id template {template!r} on {value!r}: {exc}") from None
    unmatched = [v for v, new in known.items() if new is None and v]
    return ids.map(lambda v: known[v] if known[v] is not None else v), unmatched


# --------------------------------------------------------------------------------------
# one source of a recipe
# --------------------------------------------------------------------------------------
@dataclass
class Prepared:
    """One source after the row rules: text, canonical column names.

    ``table``   the rows to import (``hole_id`` ... as text).
    ``qc``      the rows set aside (interval tables), with ``qc_type``, ``sheet``, ``row``.
    ``mapping`` source column -> canonical name, as used.
    ``info``    what was read and decided: sheets (name, header row, rows), columns,
                missing canonical columns, QC and sample-number summaries.
    ``notes``   importer notes (table, column, kind, count ...).
    """

    table: pd.DataFrame
    qc: pd.DataFrame
    mapping: dict[str, str]
    info: dict[str, Any]
    notes: list[dict[str, Any]]


def _stack(frames: list[pd.DataFrame]) -> pd.DataFrame:
    columns: list[str] = []
    for frame in frames:
        columns += [c for c in frame.columns if c not in columns]
    if not frames:
        return pd.DataFrame(columns=columns, dtype=object)
    stacked = pd.concat([f.reindex(columns=columns) for f in frames], ignore_index=True)
    return stacked.fillna("").astype(object)


def prepare_source(
    spec: Mapping[str, Any],
    kind: str,
    *,
    name: str | None = None,
    grids: Mapping[str, pd.DataFrame] | None = None,
    path: str | Path | None = None,
    delimiter: str | None = ",",
    encoding: str = "utf-8-sig",
    hole_id: Mapping[str, Any] | None = None,
    keep_unmapped: bool = True,
) -> Prepared:
    """Apply one source of a recipe (see :mod:`.recipe` for the keys of ``spec``).

    ``grids`` are the sheets already read (else ``path`` is read). ``hole_id`` is the
    recipe-wide ``{"pattern", "template"}`` used when the source has none of its own.
    """
    name = name or kind
    if grids is None:
        if path is None:
            raise ValueError(f"{name}: no file")
        grids = read_grids(path, delimiter=delimiter, encoding=encoding)
    notes: list[dict[str, Any]] = []

    def note(column: str, what: str, count: int, **extra: Any) -> None:
        if count:
            notes.append({"table": name, "column": column, "kind": what, "count": count, **extra})

    chosen = select_sheets(list(grids), spec.get("sheets"), spec.get("exclude_sheets"))
    if not chosen:
        raise ValueError(f"{name}: no sheet selected")
    frames, sheet_info = [], []
    repeated_headers = 0
    # in a recipe the header row is the spreadsheet's row number (1 = first, 0 = none)
    given = spec.get("header_row")
    header_row = None if given in (None, "", "auto") else int(given) - 1
    header_rows = spec.get("header_rows")
    header_rows = None if header_rows in (None, "", "auto") else int(header_rows)
    for sheet in chosen:
        grid = grids[sheet]
        header = locate_header(grid, header_row, header_rows)
        table, repeated = table_from_grid(grid, header)
        repeated_headers += repeated
        frame = table.reset_index(names="__row__")
        frame["__row__"] = frame["__row__"] + 1  # spreadsheet row number
        frame[SHEET_COLUMN] = sheet
        frames.append(frame)
        sheet_info.append(
            {
                "name": sheet,
                "header_row": header.row + 1,
                "header_rows": header.n_rows,
                "skipped": header.skipped,
                "rows": int(len(table)),
                "columns": list(table.columns),
            }
        )
    raw = _stack(frames)
    note("*", "repeated_header_rows_dropped", repeated_headers)
    headers = [c for c in raw.columns if c not in (SHEET_COLUMN, "__row__")]

    from_sheet = bool(spec.get("hole_id_from_sheet"))
    mapping = spec.get("map")
    if not mapping:
        mapping = suggest_column_map(headers, kind).mapping  # type: ignore[arg-type]
    mapping = {str(k): str(v) for k, v in mapping.items() if str(v).strip()}
    if from_sheet:
        mapping = {k: v for k, v in mapping.items() if v != "hole_id"}
    resolved = _resolve_column_map(headers, mapping, True, name)
    if not keep_unmapped:
        raw = raw[[c for c in raw.columns if c in resolved or c in (SHEET_COLUMN, "__row__")]]
    renamed = raw.rename(columns=resolved)
    clash = [c for c in renamed.columns if list(renamed.columns).count(c) > 1]
    if clash:
        raise ValueError(
            f"{name}: unmapped source column(s) {sorted(set(clash))} collide with a mapped "
            "name; map them to another name"
        )
    if from_sheet:
        renamed["hole_id"] = renamed[SHEET_COLUMN]
    targets = set(renamed.columns)

    rule = spec.get("hole_id") or hole_id or {}
    if "hole_id" in targets:
        renamed["hole_id"], unmatched = normalise_hole_ids(
            renamed["hole_id"], rule.get("pattern"), rule.get("template")
        )
        note("hole_id", "hole_id_pattern_not_matched", len(unmatched), examples=unmatched[:5])

    info: dict[str, Any] = {"sheets": sheet_info, "columns": headers}
    qc_rows = pd.DataFrame()
    present = set(renamed.columns) - {SHEET_COLUMN, "__row__"}
    info["rows"] = int(len(renamed))
    info["targets"] = sorted(present)
    return Prepared(
        table=renamed.drop(columns=[SHEET_COLUMN, "__row__"]).reset_index(drop=True),
        qc=qc_rows,
        mapping={str(k): v for k, v in resolved.items()},
        info=info,
        notes=notes,
    )
