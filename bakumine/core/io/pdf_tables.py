"""Tables out of text PDFs (report appendices), as CSV.

Assay and collar tables often reach us only as an appendix of a report: one table that
runs over hundreds of pages. :func:`extract_tables` reads the ruled tables of a page range
with ``pdfplumber`` (MIT licence; it reads the cell rectangles the PDF draws) and joins
them into whole tables:

* a table continues on the next page when that page starts with data or repeats the
  header (repeated headers are dropped): the same words, closely, in the same columns of
  the page. A header with other words, another number of columns or other column
  positions starts a new table, and nothing is carried down across that boundary. The
  first row of a continuation page is a header
  only when it matches the known header, or is clearly another header (it has text over
  the number columns of the rows below, or other columns): then it starts a new table
  (``one_table=True`` never does). A text-only row in the columns of the running table is
  data;
* a row at the top or the bottom of a page whose outer rule is not drawn (the table is
  left open at the page break) is closed and read like the others;
* every page is accounted for: ruled rows found against rows written, and text lying in
  the table's columns outside its rows (see :func:`table_report`);
* columns are matched by position on the page, not by counting cells, so that empty
  cells, a missing optional last column, padding columns and a table drawn in two pieces
  side by side do not shift anything;
* a two-row header (a group over sub-headers) is flattened like a workbook's;
* a row with a single wide cell is a section title: dropped and counted;
* a merged cell that names the hole once for many rows is carried down (``carry_down``);
* a value broken over two lines inside a cell is joined (``DH25_0`` + ``1``).

Scanned PDFs (images) have no text: nothing is found, and that is reported, not guessed.
Never used on Surpac files; a PDF is a text export like any other.
"""

from __future__ import annotations

import csv
import difflib
import json
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

__all__ = [
    "ExtractedTable",
    "extract_tables",
    "page_warnings",
    "parse_pages",
    "table_report",
    "write_tables",
]

_NUMERIC = re.compile(r"^[<>]?\s*[+-]?(?:\d+[.,]?\d*|[.,]\d+)$")
_Y_TOLERANCE = 2.5  # points: rows of two table pieces on one line
_X_TOLERANCE = 1.5  # points: column boundaries that are the same line
_MIN_COLUMN = 2.0  # points: narrower intervals between boundaries are rulings, not columns
_MAX_TITLES = 100
_MIN_ROW = 4.0  # points: a band of rulings lower than this is not a row
_JOINED = 1.5  # points: ruling pieces closer than this are one line
_SAME_COLUMN = 3.0  # points: header cells of a repeated header lie this close to the first


@dataclass
class _Cell:
    x0: float
    x1: float
    text: str


@dataclass
class _Row:
    top: float
    cells: list[_Cell]
    page: int

    def texts(self) -> list[str]:
        return [c.text for c in self.cells if c.text]


@dataclass
class ExtractedTable:
    """One table joined over its pages. ``rows`` are text; ``pages[i]`` is the 1-based
    page of row *i*. ``notes`` counts what was dropped, joined or carried down."""

    columns: list[str]
    rows: list[list[str]]
    pages: list[int]
    notes: dict[str, Any] = field(default_factory=dict)

    @property
    def first_page(self) -> int:
        return self.pages[0] if self.pages else int(self.notes.get("header_page", 0))

    @property
    def last_page(self) -> int:
        return self.pages[-1] if self.pages else int(self.notes.get("header_page", 0))


def parse_pages(text: str | None, n_pages: int) -> list[int]:
    """``"3-5,9"`` -> ``[3, 4, 5, 9]`` (1-based, inclusive); None = every page."""
    if not text:
        return list(range(1, n_pages + 1))
    pages: list[int] = []
    for part in text.split(","):
        first, sep, last = part.strip().partition("-")
        try:
            a = int(first)
            b = int(last) if sep and last.strip() else (n_pages if sep else a)
        except ValueError:
            raise ValueError(f"pages must look like '3-5,9', got {text!r}") from None
        if a < 1 or b < a:
            raise ValueError(f"bad page range {part.strip()!r}")
        pages += [p for p in range(a, min(b, n_pages) + 1) if p not in pages]
    if not pages:
        raise ValueError(f"the file has {n_pages} pages; none of {text!r} exist")
    return pages


def _clean(text: str | None, header: bool = False) -> str:
    """Cell text on one line. A data value broken over lines (no spaces in any piece) is
    joined without a space; anything else with one."""
    if not text:
        return ""
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return ""
    if not header and len(lines) > 1 and not any(" " in line for line in lines):
        return "".join(lines)
    return " ".join(lines)


def _open_ends(page: Any, tables: list[Any]) -> tuple[list[dict[str, Any]], int]:
    """Rows left open at the top or bottom of a table: the column rulings run on beyond
    the outermost horizontal rule (a table cut by a page break is often drawn without the
    rule at the cut) and there is text between them. Returns the horizontal lines that
    close them, each with the number of text lines it brings in (``text_lines``)."""
    verticals = [e for e in page.edges if e["orientation"] == "v"]
    if not verticals or not tables:
        return []
    words: list[dict[str, Any]] | None = None
    lines: list[dict[str, Any]] = []
    for table in tables:
        x0, top, x1, bottom = (float(v) for v in table.bbox)
        inside = [e for e in verticals if x0 - 2.0 <= float(e["x0"]) <= x1 + 2.0]
        for upward in (True, False):
            edge = top if upward else bottom
            reached = edge
            grown = True
            while grown:  # the rulings come in pieces, one per row
                grown = False
                for e in inside:
                    a, b = float(e["top"]), float(e["bottom"])
                    if upward and b >= reached - _JOINED and a < reached - _JOINED:
                        reached, grown = a, True
                    elif not upward and a <= reached + _JOINED and b > reached + _JOINED:
                        reached, grown = b, True
            low, high = (reached, edge) if upward else (edge, reached)
            if high - low < _MIN_ROW:
                continue
            columns = {
                round(float(e["x0"]))
                for e in inside
                if float(e["top"]) < high - _JOINED and float(e["bottom"]) > low + _JOINED
            }
            taken = any(
                o is not table
                and o.bbox[1] < high - _JOINED and o.bbox[3] > low + _JOINED
                and o.bbox[0] < x1 and o.bbox[2] > x0
                for o in tables
            )  # fmt: skip
            if len(columns) < 2 or taken:
                continue
            if words is None:
                words = page.extract_words()
            held = {
                round(float(w["top"]))
                for w in words
                if float(w["top"]) >= low - _JOINED
                and float(w["bottom"]) <= high + _JOINED
                and float(w["x0"]) >= x0 - 2.0
                and float(w["x1"]) <= x1 + 2.0
            }
            if not held:
                continue
            lines.append(
                {
                    "object_type": "line", "x0": x0, "x1": x1, "top": reached,
                    "bottom": reached, "width": x1 - x0, "height": 0.0, "orientation": "h",
                    "text_lines": len(held),
                }
            )  # fmt: skip
    return lines


def _page_groups(
    page: Any, number: int, settings: dict[str, Any]
) -> list[tuple[list[_Row], dict[str, int]]]:
    """The tables of a page as groups of rows; pieces drawn side by side are one group.
    With each group what was done about rows open at a page break: ``closed`` (rules
    added) and ``outside`` (text lines still between its column rulings but in no row)."""
    groups: list[tuple[tuple[float, float, float, float], list[_Row]]] = []
    found = page.find_tables(table_settings=settings)
    closing: list[dict[str, Any]] = []
    left_out: list[dict[str, Any]] = []
    if settings.get("vertical_strategy") == "lines":
        closing = _open_ends(page, found)
        if closing:
            found = page.find_tables(
                table_settings={**settings, "explicit_horizontal_lines": closing}
            )
        left_out = _open_ends(page, found)

    def inside(inner: Any, outer: Any) -> bool:
        a, b = inner.bbox, outer.bbox
        return (
            inner is not outer
            and a[0] >= b[0] - 1 and a[1] >= b[1] - 1 and a[2] <= b[2] + 1 and a[3] <= b[3] + 1
            and (a[2] - a[0]) * (a[3] - a[1]) < 0.5 * (b[2] - b[0]) * (b[3] - b[1])
        )  # fmt: skip

    tables = [t for t in found if not any(inside(t, other) for other in found)]
    for table in sorted(tables, key=lambda t: t.bbox[1]):
        texts = table.extract()
        rows: list[_Row] = []
        for row, row_texts in zip(table.rows, texts, strict=True):
            cells = [
                _Cell(float(box[0]), float(box[2]), text or "")
                for box, text in zip(row.cells, row_texts, strict=True)
                if box is not None
            ]
            if cells:
                rows.append(_Row(float(row.bbox[1]), cells, number))
        if not rows:
            continue
        box = tuple(float(v) for v in table.bbox)
        for other_box, other_rows in groups:
            overlap = min(box[3], other_box[3]) - max(box[1], other_box[1])
            beside = box[0] >= other_box[2] - _X_TOLERANCE or box[2] <= other_box[0] + _X_TOLERANCE
            if beside and overlap > 0.5 * min(box[3] - box[1], other_box[3] - other_box[1]):
                _merge_beside(other_rows, rows)
                break
        else:
            groups.append((box, rows))  # type: ignore[arg-type]
    out = []
    for box, rows in groups:
        def mine(line: dict[str, Any], box: tuple[float, ...] = box) -> bool:
            return (
                box[1] - _MIN_ROW <= line["top"] <= box[3] + _MIN_ROW
                and line["x0"] < box[2] and line["x1"] > box[0]
            )  # fmt: skip

        check = {
            "closed": sum(1 for line in closing if mine(line)),
            # a line still open after closing is past the box: look one row further
            "outside": sum(
                line["text_lines"] for line in left_out
                if line["x0"] < box[2] and line["x1"] > box[0]
                and box[1] - 60.0 <= line["top"] <= box[3] + 60.0
            ),
        }  # fmt: skip
        out.append((rows, check))
    return out


def _merge_beside(rows: list[_Row], more: list[_Row]) -> None:
    for extra in more:
        near = min(rows, key=lambda r: abs(r.top - extra.top))
        if abs(near.top - extra.top) <= _Y_TOLERANCE:
            near.cells = sorted(near.cells + extra.cells, key=lambda c: c.x0)
        else:
            rows.append(extra)
    rows.sort(key=lambda r: r.top)


def _is_number(text: str) -> bool:
    return bool(_NUMERIC.match(text.strip()))


def _numbering(texts: list[str]) -> bool:
    if len(texts) < 3 or not all(t.isdigit() for t in texts):
        return False
    numbers = [int(t) for t in texts]
    return numbers[0] == 1 and numbers == list(range(1, len(numbers) + 1))


def _width(rows: Iterable[_Row]) -> tuple[float, float]:
    cells = [c for r in rows for c in r.cells]
    return min(c.x0 for c in cells), max(c.x1 for c in cells)


def _is_title(row: _Row, left: float, right: float) -> bool:
    filled = [c for c in row.cells if c.text.strip()]
    return (
        len(filled) == 1
        and not _is_number(filled[0].text)
        and (filled[0].x1 - filled[0].x0) > 0.4 * (right - left)
    )


def _leading_header(rows: list[_Row], left: float, right: float) -> int:
    """How many leading rows of a table piece are header rows: text only, the first with
    two or more cells; a row of column numbers 1, 2, 3 ... directly below counts too. A
    row with a single narrow cell (the middle line of a three-line header) counts when
    more header follows it; a single wide cell is a section title, not header."""
    def kind(row: _Row) -> str:
        texts = [t for t in (_clean(t) for t in row.texts()) if t]
        if _numbering(texts):
            return "numbers"
        if not texts or any(_is_number(t) for t in texts) or _is_title(row, left, right):
            return "data"
        return "wide" if len(texts) >= 2 else "single"

    kinds = [kind(row) for row in rows[:6]]
    n = 0
    while n < len(kinds):
        if kinds[n] == "wide":
            n += 1
        elif n and kinds[n] == "numbers":
            return n + 1
        elif n and kinds[n] == "single" and n + 1 < len(kinds) and kinds[n + 1] == "wide":
            n += 1
        else:
            break
    return n


class _Builder:
    """One output table: its column intervals (from the header) and the rows so far."""

    def __init__(self, header: list[_Row], first: _Row) -> None:
        self.header_key = [[_clean(t, True) for t in r.texts()] for r in header]
        self.header_columns = _header_columns(header)
        self.header_page = (header[0] if header else first).page
        source = header or [first]
        edges = sorted({round(e, 1) for r in source for c in r.cells for e in (c.x0, c.x1)})
        bounds: list[float] = []
        for edge in edges:
            if not bounds or edge - bounds[-1] > _X_TOLERANCE:
                bounds.append(edge)
        self.intervals = [
            (a, b) for a, b in zip(bounds, bounds[1:], strict=False) if b - a > _MIN_COLUMN
        ]
        named = [r for r in header if not _numbering([_clean(t) for t in r.texts() if t])]
        self.top = self._header_line(named[:1])
        self.sub = self._header_line(named[1:]) if len(named) > 1 else None
        self.rows: list[dict[int, str]] = []
        self.pages: list[int] = []
        self.notes: dict[str, Any] = {
            "header_page": self.header_page,
            "header_rows": len(header),
            "repeated_headers": 0,
            "title_rows": 0,
            "titles": [],
            "cells_joined": 0,
            "open_rows_closed": 0,
            "pages": [],
        }

    def page_entry(self, page: int) -> dict[str, int]:
        """The accounting of one page of this table (made on first use)."""
        pages = self.notes["pages"]
        if not pages or pages[-1]["page"] != page:
            pages.append(
                {"page": page, "rows_found": 0, "rows_written": 0, "header_rows": 0,
                 "title_rows": 0, "numbering_rows": 0, "empty_rows": 0,
                 "open_rows_closed": 0, "text_lines_outside": 0}  # fmt: skip
            )
        return pages[-1]

    def continues_with(self, header: list[_Row], body: list[_Row]) -> bool:
        """Are the text-only leading rows of a later piece (``header``, not the known
        header) data rows of this table? Yes when the piece has this table's columns and
        the rows do not look like a header: no text over a column that holds numbers in the
        rows below, or a value this table already has in the same column."""
        if any(_numbering([_clean(t) for t in r.texts() if t]) for r in header):
            return False
        starts = [a for a, _ in self.intervals]
        ends = [b for _, b in self.intervals]
        cells = [c for r in (header + body)[:8] for c in r.cells]
        fits = sum(
            1 for c in cells
            if min(abs(c.x0 - a) for a in starts) <= 2 * _X_TOLERANCE
            and min(abs(c.x1 - b) for b in ends) <= 2 * _X_TOLERANCE
        )  # fmt: skip
        if not cells or fits < 0.9 * len(cells):
            return False
        seen: dict[int, set[str]] = {}
        for row in self.rows[-200:]:
            for j, text in row.items():
                seen.setdefault(j, set()).add(text)
        over_numbers = shared = 0
        for row in header:
            for cell in row.cells:
                text = _clean(cell.text)
                if not text:
                    continue
                middle = 0.5 * (cell.x0 + cell.x1)
                below = [
                    _clean(c.text) for r in body for c in r.cells
                    if c.x0 - _X_TOLERANCE <= middle <= c.x1 + _X_TOLERANCE
                ]  # fmt: skip
                below = [t for t in below if t]
                if below and sum(_is_number(t) for t in below) >= 0.5 * len(below):
                    over_numbers += 1
                if text in below or text in seen.get(self._column(cell), ()):
                    shared += 1
        return shared > 0 or over_numbers == 0

    def _header_line(self, rows: list[_Row]) -> list[str]:
        """Header text per interval; a merged cell gives its text to its first interval."""
        line = [""] * len(self.intervals)
        for row in rows:
            for cell in row.cells:
                text = _clean(cell.text, True)
                if not text:
                    continue
                j = self._column(cell, start_only=True)
                line[j] = f"{line[j]} {text}".strip()
        return line

    def _column(self, cell: _Cell, start_only: bool = False) -> int:
        """The column of a cell: the interval it overlaps most. A header cell
        (``start_only``) goes to the first full-width interval it covers. A data cell outside every
        interval (a column the header page did not have) opens a new one."""
        if start_only:
            under = [
                j
                for j, (a, b) in enumerate(self.intervals)
                if cell.x0 - _X_TOLERANCE <= 0.5 * (a + b) <= cell.x1 + _X_TOLERANCE
            ]
            if under:  # not a sliver of padding at the cell's edge
                widest = max(self.intervals[j][1] - self.intervals[j][0] for j in under)
                return next(
                    j for j in under
                    if self.intervals[j][1] - self.intervals[j][0] >= 0.5 * widest
                )  # fmt: skip
        best, best_overlap = -1, 0.0
        for j, (a, b) in enumerate(self.intervals):
            overlap = min(cell.x1, b) - max(cell.x0, a)
            if overlap > best_overlap:
                best, best_overlap = j, overlap
        if best >= 0:
            return best
        self.intervals.append((cell.x0, cell.x1))
        self.top.append("")
        if self.sub is not None:
            self.sub.append("")
        return len(self.intervals) - 1

    def add(self, row: _Row) -> bool:
        out: dict[int, str] = {}
        for cell in row.cells:
            text = _clean(cell.text)
            if not text:
                continue
            j = self._column(cell)
            if j in out:
                out[j] = f"{out[j]} {text}"
                self.notes["cells_joined"] += 1
            else:
                out[j] = text
        if out:
            self.rows.append(out)
            self.pages.append(row.page)
        return bool(out)

    def finish(self, carry_down: Sequence[str | int] | str | None) -> ExtractedTable:
        from bakumine.core.drillholes.importer import suggest_column_map
        from bakumine.core.drillholes.tables import _names, is_total_label

        # left to right (columns met after the header page were appended at the end)
        order = sorted(range(len(self.intervals)), key=lambda j: self.intervals[j][0])
        has_data = [any(j in row for row in self.rows) for j in order]
        names = _names(
            [self.top[j] for j in order],
            [self.sub[j] for j in order] if self.sub is not None else None,
            has_data,
        )
        keep = [(j, name) for j, name in zip(order, names, strict=True) if name]
        columns = [name for _, name in keep]
        rows = [[row.get(j, "") for j, _ in keep] for row in self.rows]
        if carry_down in (None, "auto"):
            mapping = suggest_column_map(columns, "interval").mapping
            carried = [c for c in columns if mapping.get(c) == "hole_id"]
        elif carry_down in ("none", "off", ()):
            carried = []
        else:
            carried = []
            for item in [carry_down] if isinstance(carry_down, str | int) else carry_down:
                if isinstance(item, int) or str(item).isdigit():
                    if not 1 <= int(item) <= len(columns):
                        raise ValueError(f"no column {item} to carry down; there are {columns}")
                    carried.append(columns[int(item) - 1])
                elif str(item) in columns:
                    carried.append(str(item))
                else:
                    raise ValueError(f"no column {item!r} to carry down; there are {columns}")
        filled = 0
        # a total row ("Total", "Нийт" in its first cell) belongs to no hole
        totals = [is_total_label(next((c for c in row if c), "")) for row in rows]
        for name in carried:
            j = columns.index(name)
            last = ""
            for row, total in zip(rows, totals, strict=True):
                if row[j]:
                    last = row[j]
                elif last and not total:
                    row[j] = last
                    filled += 1
        notes = {
            **self.notes,
            "carried_down": {"columns": carried, "cells": filled},
            "total_rows": int(sum(totals)),
        }
        return ExtractedTable(columns, rows, list(self.pages), notes)


def extract_tables(
    path: str | Path,
    pages: str | Sequence[int] | None = None,
    *,
    carry_down: Sequence[str | int] | str | None = "auto",
    one_table: bool = False,
    strategy: str = "lines",
    progress: Callable[[int, int], None] | None = None,
) -> list[ExtractedTable]:
    """Read the tables of ``pages`` (``"3-5,9"``, a list of 1-based pages, or None = all).

    ``carry_down``  columns whose empty cells take the value above (a merged cell that
                    names the hole once): ``"auto"`` = the hole-id column, found by its
                    header; a list of column names or 1-based positions; ``"none"``.
    ``one_table``   treat everything as one table: only the first header counts.
    ``strategy``    ``"lines"`` (ruled tables, the default) or ``"text"`` (pdfplumber
                    guesses the columns from the text alignment; less reliable).
    """
    import pdfplumber

    if strategy not in ("lines", "text"):
        raise ValueError(f"strategy must be 'lines' or 'text', got {strategy!r}")
    settings = {"vertical_strategy": strategy, "horizontal_strategy": strategy}
    builders: list[_Builder] = []
    current: _Builder | None = None
    with pdfplumber.open(str(path)) as pdf:
        numbers = (
            parse_pages(pages, len(pdf.pages))
            if pages is None or isinstance(pages, str)
            else [int(p) for p in pages]
        )
        for done, number in enumerate(numbers):
            if not 1 <= number <= len(pdf.pages):
                raise ValueError(f"page {number} is outside 1..{len(pdf.pages)}")
            page = pdf.pages[number - 1]
            for rows, check in _page_groups(page, number, settings):
                left, right = _width(rows)
                n_header = _leading_header(rows, left, right)
                header, body = rows[:n_header], rows[n_header:]
                key = [[_clean(t, True) for t in r.texts()] for r in header]
                # the known header again: the same words (closely) in the same columns
                repeated = bool(
                    current is not None
                    and header
                    and _same_header(key, current.header_key)
                    and _same_columns(_header_columns(header), current.header_columns)
                )
                if current is not None and header and not repeated:
                    if one_table or current.continues_with(header, body):
                        header, body, key = [], rows, []  # data that happens to be text
                if current is None or (header and not repeated):
                    if not header and not body:
                        continue
                    current = _Builder(header, (body or header)[0])
                    builders.append(current)
                elif header:
                    current.notes["repeated_headers"] += 1
                entry = current.page_entry(number)
                entry["rows_found"] += len(rows)
                entry["header_rows"] += len(header)
                for row in body:
                    if _is_title(row, left, right):
                        current.notes["title_rows"] += 1
                        entry["title_rows"] += 1
                        if len(current.notes["titles"]) < _MAX_TITLES:
                            current.notes["titles"].append(
                                {"page": row.page, "text": _clean(row.texts()[0], True)}
                            )
                    elif _numbering([_clean(t) for t in row.texts() if t]) and row is body[0]:
                        entry["numbering_rows"] += 1  # the column-number row of each page
                    elif current.add(row):
                        entry["rows_written"] += 1
                    else:
                        entry["empty_rows"] += 1
                entry["open_rows_closed"] += check["closed"]
                entry["text_lines_outside"] += check["outside"]
                current.notes["open_rows_closed"] += check["closed"]
            page.close()  # frees the page's parsed objects: long files stay small in memory
            if progress is not None:
                progress(done + 1, len(numbers))
    return [b.finish(carry_down) for b in builders]


def _header_columns(header: list[_Row]) -> list[tuple[float, float]]:
    """Where the named cells of a header's first row lie (left and right edge)."""
    return [(c.x0, c.x1) for c in header[0].cells if _clean(c.text, True)] if header else []


def _same_columns(a: list[tuple[float, float]], b: list[tuple[float, float]]) -> bool:
    """The same number of header cells at the same places on the page. Another table,
    even one with nearly the same header words, is drawn with other column widths."""
    return len(a) == len(b) and all(
        abs(p[0] - q[0]) <= _SAME_COLUMN and abs(p[1] - q[1]) <= _SAME_COLUMN
        for p, q in zip(a, b, strict=True)
    )


def _same_header(a: list[list[str]], b: list[list[str]]) -> bool:
    def flat(rows: list[list[str]]) -> list[str]:
        return [re.sub(r"\s+", " ", t).casefold() for row in rows for t in row if t]

    first_a, first_b = flat(a[:1]), flat(b[:1])
    if not first_a or not first_b:
        return False
    if flat(a) == flat(b) or first_a == first_b:
        return True
    # closely: the same header with a word broken differently or a footnote mark
    close = difflib.SequenceMatcher(None, " | ".join(first_a), " | ".join(first_b)).ratio()
    return len(first_a) == len(first_b) and close >= 0.9


def _numbering_gaps(table: ExtractedTable) -> dict[str, Any] | None:
    """The row-number column of a table (1, 2, 3 ...) and the numbers it skips."""
    if len(table.rows) < 3:
        return None
    for j, name in enumerate(table.columns[:2]):
        cells = [row[j] for row in table.rows]
        numbers = [int(c) for c in cells if c.isdigit()]
        if len(numbers) < 0.8 * len(cells) or len(numbers) < 3:
            continue
        steps = [b - a for a, b in zip(numbers, numbers[1:], strict=False)]
        if sum(1 for s in steps if s == 1) < 0.8 * len(steps):
            continue
        missing = [
            n for a, s in zip(numbers, steps, strict=False) if 1 < s <= 50
            for n in range(a + 1, a + s)
        ]  # fmt: skip
        return {"column": name, "missing": missing[:100], "n_missing": len(missing)}
    return None


def page_warnings(table: ExtractedTable) -> list[str]:
    """What may have been lost, per page: ruled rows that were not written, text between
    the column rulings but in no row, and numbers skipped by the row-number column."""
    warnings = []
    for entry in table.notes.get("pages", []):
        accounted = (
            entry["rows_written"] + entry["header_rows"] + entry["title_rows"]
            + entry["numbering_rows"]
        )  # fmt: skip
        if accounted < entry["rows_found"]:
            warnings.append(
                f"page {entry['page']}: {entry['rows_found']} ruled rows, "
                f"{entry['rows_written']} written; {entry['rows_found'] - accounted} "
                "had no text"
            )
        if entry["text_lines_outside"]:
            warnings.append(
                f"page {entry['page']}: {entry['text_lines_outside']} text line(s) between "
                "the column rulings lie outside the ruled rows and were NOT extracted"
            )
    gaps = _numbering_gaps(table)
    if gaps and gaps["n_missing"]:
        listed = ", ".join(str(n) for n in gaps["missing"][:12])
        warnings.append(
            f"the numbering column {gaps['column']!r} skips {gaps['n_missing']} number(s) "
            f"({listed}): rows may be missing"
        )
    return warnings


def table_report(table: ExtractedTable) -> dict[str, Any]:
    """Rows per hole, depth range and where ``from`` goes backwards, for a table whose
    header names a hole and from / to columns (else only the row and page counts).

    Always: ``pages`` (per page the ruled rows found, the rows written, header / title /
    numbering / empty rows, rows closed at a page break, text lines left outside),
    ``numbering`` (numbers the row-number column skips) and ``warnings``."""
    import numpy as np
    import pandas as pd

    from bakumine.core.drillholes.importer import suggest_column_map

    report: dict[str, Any] = {
        "rows": len(table.rows),
        "columns": list(table.columns),
        "first_page": table.first_page,
        "last_page": table.last_page,
        **{k: v for k, v in table.notes.items() if k != "titles"},
        "numbering": _numbering_gaps(table),
        "warnings": page_warnings(table),
    }
    mapping = suggest_column_map(table.columns, "interval").mapping
    found = {target: source for source, target in mapping.items()}
    if not {"hole_id", "depth_from", "depth_to"} <= set(found) or not table.rows:
        return report
    frame = pd.DataFrame(table.rows, columns=table.columns)
    hole = frame[found["hole_id"]]
    numbers = {
        k: pd.to_numeric(frame[found[k]].str.replace(",", "."), errors="coerce").to_numpy()
        for k in ("depth_from", "depth_to")
    }
    start, end = numbers["depth_from"], numbers["depth_to"]
    holes: list[dict[str, Any]] = []
    runs = (hole != hole.shift()).cumsum().to_numpy()
    seen: dict[str, int] = {}
    for run in np.unique(runs):
        rows = np.flatnonzero(runs == run)
        name = str(hole.iat[rows[0]])
        a, b = start[rows], end[rows]
        seen[name] = seen.get(name, 0) + 1
        backwards = int((np.diff(a[np.isfinite(a)]) < 0).sum())
        holes.append(
            {
                "hole_id": name,
                "rows": int(len(rows)),
                "from": float(np.nanmin(a)) if np.isfinite(a).any() else None,
                "to": float(np.nanmax(b)) if np.isfinite(b).any() else None,
                "from_backwards": backwards,
                "first_page": int(table.pages[rows[0]]),
                "last_page": int(table.pages[rows[-1]]),
            }
        )
    report.update(
        hole_column=found["hole_id"],
        n_holes=len(seen),
        rows_without_hole=int((hole == "").sum()),
        rows_without_depths=int((~np.isfinite(start) | ~np.isfinite(end)).sum()),
        holes_from_backwards=[h["hole_id"] for h in holes if h["from_backwards"]],
        holes_in_several_runs=sorted(name for name, n in seen.items() if n > 1),
        holes=holes,
    )
    return report


def write_tables(
    tables: Sequence[ExtractedTable], out_dir: str | Path, stem: str = "table"
) -> tuple[list[Path], Path, list[dict[str, Any]]]:
    """Write ``<stem>_1.csv``, ``<stem>_2.csv`` ... (UTF-8, one header row) and
    ``<stem>_report.json``. Returns ``(csv paths, report path, reports)``."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths, reports = [], []
    for k, table in enumerate(tables, start=1):
        path = out / f"{stem}_{k}.csv"
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(table.columns)
            writer.writerows(table.rows)
        paths.append(path)
        reports.append({"file": path.name, **table_report(table), "titles": table.notes["titles"]})
    report_path = out / f"{stem}_report.json"
    report_path.write_text(
        json.dumps({"tables": reports}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return paths, report_path, reports
