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
* :func:`depths_from_sample_ids` - ``to`` = the number at the end of the sample id,
                              ``from = to - length`` (or an explicit ``a-b``);
* :func:`classify_qc`       - blanks, standards and duplicates among the samples;
* :func:`is_total_label`    - ``Total`` / ``Нийт`` / ``Итого`` rows, which are dropped
                              together with the rows that name no hole;
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
    element_attribute,
    suggest_column_map,
)

__all__ = [
    "Header",
    "Prepared",
    "QC_WORDS",
    "check_sample_depths",
    "classify_qc",
    "depths_from_sample_ids",
    "is_total_label",
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

#: QC sample kinds and the words (lower case, any language met so far) that announce them
QC_WORDS: dict[str, tuple[str, ...]] = {
    "blank": ("blank", "blk", "бланк", "холостая", "холостой", "хоосон"),
    "standard": ("standard", "standart", "std", "crm", "srm", "стандарт", "эталон"),
    "duplicate": (
        "duplicate", "dup", "dupl", "replicate", "rep", "дубликат", "дубль", "давхар",
        "давтан",
    ),  # fmt: skip
}


#: words that open the label of a total / summary row (English, Mongolian, Russian)
TOTAL_WORDS: tuple[str, ...] = (
    "total", "totals", "subtotal", "sum", "нийт", "нэгтгэл", "дүн", "бүгд",
    "итого", "всего",
)  # fmt: skip


def is_total_label(text: object) -> bool:
    """Does a cell label a total / summary row: ``Total``, ``Нийт``, ``НЭГТГЭЛ``,
    ``Итого по участку``, ``Grand total:``, ``Sum`` ... Its first word (or the one after
    ``grand`` / ``sub``) is a total word and it holds no digit, so that a hole named
    ``SUM-01`` is not one."""
    words = [w for w in re.split(r"[\W_]+", str(text).casefold()) if w]
    if not words or any(c.isdigit() for c in str(text)):
        return False
    if words[0] in ("grand", "sub", "нийт") and len(words) > 1 and words[1] in TOTAL_WORDS:
        return True
    return words[0] in TOTAL_WORDS


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
# depths from sample numbers
# --------------------------------------------------------------------------------------
_SEPARATOR = r"[\s_\-./]"
_DEPTH = r"\d+(?:\.\d+)?"
_TRAILING_INT = re.compile(r"(\d+)\D*$")


def _hole_prefix(hole: str) -> re.Pattern[str]:
    """A loose pattern for the hole id at the start of a sample id: letters in any case,
    numbers with or without leading zeros, any separators (``DH-01`` ~ ``dh_001``)."""
    parts = re.findall(r"[^\W\d_]+|\d+", hole)
    body = (_SEPARATOR + "*").join(
        f"0*{int(p)}" if p.isdigit() else re.escape(p) for p in parts
    )
    return re.compile(rf"^\s*{body}(?:{_SEPARATOR}+|$)", re.IGNORECASE)


_AUTO_TO = re.compile(rf"^({_DEPTH})$")
_AUTO_RANGE = re.compile(rf"^({_DEPTH})\s*[-–_]\s*({_DEPTH})$")


def _hole_patterns(patterns: Sequence[str], hole: str) -> list[re.Pattern[str]]:
    number = _TRAILING_INT.search(hole)
    values = {"hole": re.escape(hole), "holenum": str(int(number.group(1))) if number else ""}
    compiled = []
    for pattern in patterns:
        try:
            text = re.sub(r"\{(hole|holenum)\}", lambda m: values[m.group(1)], pattern)
            compiled.append(re.compile(text, re.IGNORECASE))
        except re.error as exc:
            raise ValueError(f"sample number pattern {pattern!r}: {exc}") from None
        if "to" not in compiled[-1].groupindex:
            raise ValueError(f"sample number pattern {pattern!r} has no (?P<to>...) group")
    return compiled


def _without_hole_number(hole: str, numbers: Sequence[str]) -> tuple[np.ndarray, bool] | None:
    """The numbers of a hole's sample ids with the hole's own number taken off their
    front (``4002`` of hole 4 -> 2), when every one of them begins with it and what is
    left is a depth above 0. Also: was any of them zero-padded (``002``)?"""
    found = _TRAILING_INT.search(hole)
    if found is None or not numbers:
        return None
    for form in sorted({found.group(1), str(int(found.group(1)))}, key=len, reverse=True):
        if not all(n.startswith(form) and len(n) > len(form) for n in numbers):
            continue
        rest = [n[len(form) :] for n in numbers]
        if any(r.startswith(".") for r in rest):
            continue
        values = np.array([float(r) for r in rest], dtype=np.float64)
        if (values > 0).all():
            return values, any(r.startswith("0") for r in rest)
    return None


def depths_from_sample_ids(
    hole_ids: Sequence[str],
    sample_ids: Sequence[str],
    length: float | Sequence[float] = 1.0,
    patterns: Sequence[str] | None = None,
    *,
    max_depth: Mapping[str, float] | None = None,
    tolerance: float = 5.0,
    report: dict[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(depth_from, depth_to)`` read from sample numbers; NaN where the rule does not apply.

    The default rule: strip the hole id from the start of the sample id when it is there
    (loosely: ``DH01-152`` belongs to ``DH-01``); what is left must be a number, the
    depth of the *end* of the sample, ``to``, and ``from = to - length``; or an explicit
    ``from-to`` pair (``DH01-151-152``). ``length`` is one number or one per row. A row
    whose ``from`` would be negative or not below ``to`` gets no depths.

    Some ids repeat the hole's number in front of the metre: ``DH04-4002`` is sample
    ``002`` of hole 4, not metre 4002. The number is taken off, hole by hole, when every
    number of the hole begins with it and what is left is a depth, and either the hole's
    depth says so (``max_depth``: hole id -> depth; the plain reading goes more than
    ``tolerance`` beyond it and the other does not) or, the depth being unknown, the hole
    has three or more samples and the numbers are zero-padded or all 1000 and above.
    ``report["repeated_hole_number"]`` lists the holes read that way.

    ``patterns`` replaces the default rule with regular expressions tried in order against
    the whole sample id (case-insensitive, full match). Each needs a ``(?P<to>...)`` group
    and may have ``(?P<from>...)``; ``{hole}`` stands for the row's hole id and
    ``{holenum}`` for the number at its end without leading zeros, e.g.
    ``DH0*{holenum}-{holenum}(?P<to>\\d+)`` for ids that repeat the hole number.
    """
    n = len(sample_ids)
    lengths = np.broadcast_to(np.asarray(length, dtype=np.float64), (n,))
    start = np.full(n, np.nan)
    end = np.full(n, np.nan)
    cache: dict[str, Any] = {}
    plain: dict[str, list[tuple[int, str]]] = {}  # hole -> rows whose id is one number
    for i, (hole, sample) in enumerate(zip(hole_ids, sample_ids, strict=True)):
        hole, sample = str(hole).strip(), str(sample).strip()
        if not sample:
            continue
        low = high = None
        if patterns:
            if hole not in cache:
                cache[hole] = _hole_patterns(patterns, hole)
            for pattern in cache[hole]:
                match = pattern.fullmatch(sample)
                if match is not None:
                    high = float(match.group("to"))
                    if "from" in pattern.groupindex and match.group("from") is not None:
                        low = float(match.group("from"))
                    break
        else:
            if hole not in cache:
                cache[hole] = _hole_prefix(hole) if hole else None
            prefix = cache[hole].match(sample) if cache[hole] is not None else None
            rest = sample[prefix.end() :] if prefix else sample
            single = _AUTO_TO.match(rest)
            pair = _AUTO_RANGE.match(rest)
            if single:
                plain.setdefault(hole, []).append((i, single.group(1)))
                continue
            elif pair:
                low, high = float(pair.group(1)), float(pair.group(2))
        if high is None:
            continue
        if low is None:
            low = high - lengths[i]
        if np.isfinite(low) and low < high and low >= 0:
            start[i], end[i] = low, high
    repeated: list[str] = []
    for hole, found in plain.items():
        rows = np.array([i for i, _ in found], dtype=np.int64)
        values = np.array([float(text) for _, text in found], dtype=np.float64)
        other = _without_hole_number(hole, [text for _, text in found])
        if other is not None:
            limit = (max_depth or {}).get(hole)
            if limit is not None and np.isfinite(limit):
                take = values.max() > limit + tolerance >= other[0].max()
            else:
                take = len(found) >= 3 and (other[1] or values.min() >= 1000.0)
            if take:
                values = other[0]
                repeated.append(hole)
        lows = values - lengths[rows]
        good = np.isfinite(lows) & (lows < values) & (lows >= 0)
        start[rows[good]], end[rows[good]] = lows[good], values[good]
    if report is not None:
        report["repeated_hole_number"] = repeated
    return start, end


def check_sample_depths(
    hole_ids: Sequence[str], depth_from: np.ndarray, depth_to: np.ndarray
) -> dict[str, Any]:
    """How plausible derived intervals are, per hole and in total: rows, first ``from``,
    last ``to``, repeated intervals, places where the depth goes backwards (in file order)
    and gaps or overlaps between consecutive samples."""
    frame = pd.DataFrame(
        {"hole_id": list(hole_ids), "a": np.asarray(depth_from), "b": np.asarray(depth_to)}
    ).dropna()
    holes = []
    for hole, part in frame.groupby("hole_id", sort=False):
        a, b = part["a"].to_numpy(), part["b"].to_numpy()
        step = a[1:] - b[:-1] if len(a) > 1 else np.zeros(0)
        holes.append(
            {
                "hole_id": str(hole),
                "rows": int(len(part)),
                "from": float(a.min()),
                "to": float(b.max()),
                "duplicates": int(part.duplicated(["a", "b"]).sum()),
                "backwards": int((np.diff(b) < 0).sum()),
                "gaps": int((step > 1e-9).sum()),
                "overlaps": int((step < -1e-9).sum()),
            }
        )
    keys = ("rows", "duplicates", "backwards", "gaps", "overlaps")
    return {**{k: int(sum(h[k] for h in holes)) for k in keys}, "holes": holes}


# --------------------------------------------------------------------------------------
# QC rows
# --------------------------------------------------------------------------------------
def _qc_word(text: str) -> str:
    """The QC kind a cell announces ("" when none)."""
    words = [w for w in re.split(r"[\W\d_]+", text.casefold()) if w]
    for kind, names in QC_WORDS.items():
        if any(w in names for w in words):
            return kind
    return ""


_LETTER_SUFFIX = re.compile(r"\d[^\W\d_]{1,3}$")


def classify_qc(
    table: pd.DataFrame,
    *,
    columns: Sequence[str] = (),
    letter_suffix: bool = True,
) -> np.ndarray:
    """QC kind of each row of an interval table still in text: ``""`` for a sample, else
    ``blank`` / ``standard`` / ``duplicate`` / ``other`` / ``letter_suffix``.

    * a QC word (:data:`QC_WORDS`) in ``sample_type``, in ``depth_from`` / ``depth_to``
      or in one of ``columns``;
    * any other text in ``depth_from`` / ``depth_to`` where a number belongs: ``other``;
    * with ``letter_suffix``: a sample id ending in a letter (``190043A``) in a table where
      at least four fifths of the ids end in a digit.
    """
    n = len(table)
    kinds = np.full(n, "", dtype=object)
    for column in ("sample_type", *columns):
        if column in table.columns:
            found = np.array([_qc_word(str(v)) for v in table[column].tolist()], dtype=object)
            kinds = np.where((kinds == "") & (found != ""), found, kinds)
    for column in ("depth_from", "depth_to"):
        if column not in table.columns:
            continue
        for i, value in enumerate(table[column].tolist()):
            text = str(value).strip()
            if kinds[i] or not text or _is_number(text) or text in ("-", "--"):
                continue
            kinds[i] = _qc_word(text) or "other"
    if letter_suffix and "sample_id" in table.columns:
        ids = table["sample_id"].astype(str).str.strip()
        filled = ids != ""
        lettered = ids.str.contains(_LETTER_SUFFIX).to_numpy(dtype=bool)
        digits = ids.str.contains(r"\d$").to_numpy(dtype=bool)
        if filled.sum() and digits.sum() >= 0.8 * filled.sum():
            kinds = np.where((kinds == "") & lettered, "letter_suffix", kinds)
    return kinds


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
                missing canonical columns, units of element columns, rows dropped (total
                rows, rows without a hole id), QC and sample-number summaries.
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
    hole_depths: Mapping[str, float] | None = None,
) -> Prepared:
    """Apply one source of a recipe (see :mod:`.recipe` for the keys of ``spec``).

    ``grids`` are the sheets already read (else ``path`` is read). ``hole_id`` is the
    recipe-wide ``{"pattern", "template"}`` used when the source has none of its own.
    ``keep_unmapped=False`` drops the columns the column map does not name; when the map
    is left to the suggester, an interval table keeps its known columns (hole, from, to,
    sample id / type / length) and the element columns only. ``hole_depths`` (hole id ->
    depth of the hole, in this source's unit) lets the sample-number rule be checked: a
    hole whose derived depths go beyond its depth is not imported that way.
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

    if list(grids) == [""]:  # a CSV file has no sheets: a table-wide choice passes it by
        chosen = [""]
    else:
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
        if not keep_unmapped and kind == "interval":
            # the suggester proposes every other column of an interval table as an
            # attribute; "unmapped" are then the columns it did not recognise: neither a
            # known column nor an element ("No", remarks, laboratory batch ...)
            known = set(_SYNONYMS["interval"])
            extra = [
                column for column, target in mapping.items()
                if target not in known and element_attribute(column) is None
            ]  # fmt: skip
            mapping = {c: t for c, t in mapping.items() if c not in extra}
            note("*", "unrecognised_columns_dropped", len(extra), examples=extra[:8])
    mapping = {str(k): str(v) for k, v in mapping.items() if str(v).strip()}
    if from_sheet:
        mapping = {k: v for k, v in mapping.items() if v != "hole_id"}
    resolved = _resolve_column_map(headers, mapping, True, name)
    every_cell = raw[headers].to_numpy(dtype=object) if headers else None
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
    dropped: dict[str, int] = {}
    # total / summary rows: the hole id cell or the first filled cell says "Total", "Нийт"
    if len(renamed) and every_cell is not None:
        labels = [next((str(c) for c in row if str(c).strip()), "") for row in every_cell]
        total = np.array([is_total_label(label) for label in labels], dtype=bool)
        if "hole_id" in targets and not from_sheet:
            total |= np.array([is_total_label(v) for v in renamed["hole_id"]], dtype=bool)
        if total.any():
            dropped["total_rows"] = int(total.sum())
            note("*", "total_rows_dropped", int(total.sum()),
                 examples=sorted({label for label, t in zip(labels, total, strict=True) if t})[:5],
                 rows=renamed["__row__"].to_numpy()[total][:20].tolist())  # fmt: skip
            renamed = renamed.loc[~total].reset_index(drop=True)
    # the unit a header gives for an element column ("Au, g/t"), under the mapped name
    elements = {target: element_attribute(column) for column, target in resolved.items()}
    info["units"] = {target: found[1] for target, found in elements.items() if found and found[1]}
    qc_rows = pd.DataFrame()
    if kind == "interval" and len(renamed):
        renamed, qc_rows = _interval_rules(renamed, spec, name, info, note, hole_depths)
    # rows that name no hole are nobody's collar, station or sample (QC rows are aside)
    if "hole_id" in renamed.columns and len(renamed):
        nameless = (renamed["hole_id"].astype(str).str.strip() == "").to_numpy(dtype=bool)
        if nameless.any():
            dropped["rows_without_hole_id"] = int(nameless.sum())
            note("hole_id", "rows_without_hole_id_dropped", int(nameless.sum()),
                 rows=renamed["__row__"].to_numpy()[nameless][:20].tolist())  # fmt: skip
            renamed = renamed.loc[~nameless].reset_index(drop=True)
    info["dropped"] = dropped
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


def _interval_rules(
    table: pd.DataFrame, spec: Mapping[str, Any], name: str, info: dict[str, Any], note: Any,
    hole_depths: Mapping[str, float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:  # fmt: skip
    """QC rows and sample-number depths of one interval source. Returns (samples, qc)."""
    qc_spec = spec.get("qc")
    if qc_spec is None or qc_spec is True:
        qc_spec = {}
    separate = qc_spec is not False and qc_spec.get("mode", "separate") != "off"
    kinds = np.full(len(table), "", dtype=object)
    if separate:
        kinds = classify_qc(
            table,
            columns=tuple(qc_spec.get("columns") or ()),
            letter_suffix=bool(qc_spec.get("letter_suffix", True)),
        )

    def blank(column: str) -> np.ndarray:
        if column not in table.columns:
            return np.ones(len(table), dtype=bool)
        return table[column].astype(str).str.strip().isin(["", "-", "--"]).to_numpy(dtype=bool)

    rule = spec.get("sample_depth")
    if rule:
        rule = {} if rule is True else dict(rule)
        column = rule.get("column", "sample_id")
        if column not in table.columns or "hole_id" not in table.columns:
            raise ValueError(
                f"{name}: the sample-number rule needs the columns hole_id and {column!r} "
                "(map the sample number column to it)"
            )
        todo = kinds == ""
        if rule.get("when", "missing") != "always":
            todo &= blank("depth_from") & blank("depth_to")
        length: Any = float(rule.get("length", 1.0))
        length_column = rule.get("length_column")
        if length_column:
            if length_column not in table.columns:
                raise ValueError(f"{name}: no column {length_column!r} for the sample length")
            given = pd.to_numeric(
                table[length_column].astype(str).str.replace(",", "."), errors="coerce"
            ).to_numpy(dtype=np.float64)
            length = np.where(np.isfinite(given), given, length)[todo]
        tolerance = float(rule.get("depth_tolerance", 5.0))
        decided: dict[str, Any] = {}
        start, end = depths_from_sample_ids(
            table["hole_id"].to_numpy()[todo],
            table[column].to_numpy()[todo],
            length,
            rule.get("patterns") or None,
            max_depth=hole_depths,
            tolerance=tolerance,
            report=decided,
        )
        for target in ("depth_from", "depth_to"):
            if target not in table.columns:
                table[target] = ""
        ok = np.isfinite(end)
        rows = np.flatnonzero(todo)
        # a rule that puts samples below the bottom of the hole has misread the ids
        beyond: list[dict[str, Any]] = []
        unchecked = 0
        too_deep = np.zeros(len(rows), dtype=bool)
        derived_holes = table["hole_id"].to_numpy()[rows].astype(str)
        for hole in dict.fromkeys(derived_holes[ok].tolist()):
            limit = (hole_depths or {}).get(hole)
            if limit is None or not np.isfinite(limit):
                unchecked += 1
                continue
            mine = ok & (derived_holes == hole)
            deepest = float(end[mine].max())
            if deepest > limit + tolerance:
                too_deep |= mine
                beyond.append(
                    {"hole_id": hole, "derived_to": deepest, "max_depth": float(limit),
                     "rows": int(mine.sum()),
                     "example": str(table[column].iat[rows[np.flatnonzero(mine)[-1]]])}
                )  # fmt: skip
        ok &= ~too_deep
        for target, values in (("depth_from", start), ("depth_to", end)):
            table.iloc[rows[ok], table.columns.get_loc(target)] = [
                repr(float(v)) for v in values[ok]
            ]
        failed = rows[~ok & ~too_deep]
        kinds[failed] = "no_depth"
        kinds[rows[too_deep]] = "depth_beyond_hole"
        check = check_sample_depths(table["hole_id"].to_numpy()[rows[ok]], start[ok], end[ok])
        check["unmatched"] = int(len(failed))
        check["unmatched_examples"] = table[column].to_numpy()[failed][:8].tolist()
        check["repeated_hole_number"] = decided.get("repeated_hole_number", [])
        check["depth_tolerance"] = tolerance
        check["beyond_hole"] = beyond
        check["beyond_hole_rows"] = int(too_deep.sum())
        check["holes_without_known_depth"] = unchecked if hole_depths is not None else None
        check["preview"] = [
            {
                "hole_id": str(table["hole_id"].iat[r]),
                "sample_id": str(table[column].iat[r]),
                "depth_from": float(a),
                "depth_to": float(b),
            }
            for r, a, b in zip(rows[ok][:8], start[ok][:8], end[ok][:8], strict=True)
        ]
        info["sample_depth"] = check
        note("depth_from,depth_to", "from_sample_number", int(ok.sum()))
        note("depth_from,depth_to", "sample_number_not_understood", int(len(failed)),
             examples=check["unmatched_examples"][:5])  # fmt: skip
        stripped = check["repeated_hole_number"]
        note("depth_from,depth_to", "sample_number_repeats_hole_number", len(stripped),
             examples=stripped[:5])  # fmt: skip
        note("depth_from,depth_to", "sample_depths_beyond_hole_not_imported",
             int(too_deep.sum()),
             examples=[f"{b['hole_id']}: {b['derived_to']:g} > {b['max_depth']:g}"
                       for b in beyond[:5]],
             hint="the sample numbers are not read as meant: give sample_depth patterns",
        )  # fmt: skip
    if separate:
        no_interval = (kinds == "") & blank("depth_from") & blank("depth_to")
        kinds[no_interval] = "no_interval"
    is_qc = kinds != ""
    qc = table.loc[is_qc].copy()
    qc.insert(0, "qc_type", kinds[is_qc])
    qc = qc.rename(columns={SHEET_COLUMN: "sheet", "__row__": "row"})
    qc = qc.astype(str).reset_index(drop=True)
    counts = Counter(kinds[is_qc].tolist())
    info["qc"] = dict(sorted(counts.items()))
    for what, count in sorted(counts.items()):
        note("*", f"qc_{what}_set_aside", count)
    return table.loc[~is_qc].reset_index(drop=True), qc
