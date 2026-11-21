"""Surpac string file ``.str`` (text): read and write.

PROVISIONAL — not yet validated against a file exported by Surpac.
==================================================================

This module is written clean-room from the layout described in research.md
section 5.6. On 2026-11-21 it was compared with the published "Appendix D -
File Formats" of the Surpac manual (1996; URL in the docstring of
:mod:`bakumine.core.io.surpac_dtm`), which CONFIRMS items 1 to 5 below (header
``location, date, purpose, memo``; axis record; ``string, Y, X, Z, d1, ...``;
null record; ``END`` record) and adds: string numbers 1 to 32000, up to 100
description fields of 32 characters, 512 in total (not enforced here). Items 6
to 8 remain assumptions, and no file exported by a current Surpac was read:

1. Line 1 is a header of comma-separated fields ``location, date, purpose,
   memo``. Anything after the third comma is kept verbatim as the memo, so
   extra header fields that real files may carry are not lost.
2. Line 2 is the axis record: string number ``0`` followed by two points,
   ``0, Y1, X1, Z1, Y2, X2, Z2``.
3. Every other record is ``string_number, Y, X, Z, d1, d2, ...``. The first
   coordinate column is the NORTHING (Y) and the second the EASTING (X). The
   order is held in :data:`COORD_COLUMNS`.
4. A record with string number 0 and zero coordinates ends a segment.
5. ``0, 0, 0, 0, END`` ends the file. The keyword spelling is :data:`KEYWORD_END`
   and is matched without regard to case.
6. Description fields are separated by commas, so a description cannot
   contain a comma. No quoting convention is assumed.
7. The writer's number layout (``", "`` separator, fixed decimals, a trailing
   empty description field on records without descriptions) is our choice of
   canonical form; whether Surpac requires fixed column widths is unknown.
8. A binary variant of ``.str`` exists in Surpac. It is NOT supported: binary
   content is detected and rejected with :class:`BinaryFileError`.

The reader is tolerant where real files might differ (see
:func:`parse_surpac_str`); the writer is strict. All layout knowledge sits in
the constants and the four small record functions directly below, so it can
be corrected quickly once a real file is available.

Coordinates: the model is internal ``x = east, y = north, z = up``; the swap
to the file's ``Y, X, Z`` happens only in :func:`_file_to_internal` and
:func:`_internal_to_file`.
"""

from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

import numpy as np

from bakumine.core.io._text import BinaryFileError, FormatWarning, parse_float, read_text
from bakumine.core.io.strings import AxisRecord, StringFile, StringHeader, StringSegment

__all__ = [
    "BinaryFileError",
    "FormatWarning",
    "SurpacFormatError",
    "format_surpac_str",
    "parse_surpac_str",
    "read_surpac_str",
    "write_surpac_str",
]

# --------------------------------------------------------------------------
# Layout knowledge (PROVISIONAL). Change here when a real file says otherwise.
# --------------------------------------------------------------------------

#: Meaning of the three coordinate columns of a point record, in file order.
COORD_COLUMNS: tuple[str, str, str] = ("y", "x", "z")
#: Field separator of every record.
FIELD_SEPARATOR = ","
#: String number that marks axis, segment-end and file-end records.
SEPARATOR_STRING_NUMBER = 0
#: Keyword in the description field of the last record.
KEYWORD_END = "END"
#: Number of fields of the header that have a name; the rest is the memo.
HEADER_FIELDS = ("location", "date", "purpose", "memo")
#: Separator the writer puts between fields.
WRITE_SEPARATOR = ", "
#: Separator the writer puts between description fields.
WRITE_DESCRIPTION_SEPARATOR = ", "

_TO_INTERNAL = [COORD_COLUMNS.index(c) for c in ("x", "y", "z")]
_TO_FILE = [("x", "y", "z").index(c) for c in COORD_COLUMNS]


class SurpacFormatError(ValueError):
    """The file does not follow the layout (raised in strict mode or on write)."""


def _file_to_internal(columns) -> tuple[float, float, float]:
    """Coordinate columns in file order -> ``(east, north, up)``."""
    return columns[_TO_INTERNAL[0]], columns[_TO_INTERNAL[1]], columns[_TO_INTERNAL[2]]


def _internal_to_file(xyz) -> tuple[float, float, float]:
    """``(east, north, up)`` -> coordinate columns in file order."""
    return xyz[_TO_FILE[0]], xyz[_TO_FILE[1]], xyz[_TO_FILE[2]]


def _fixed(value: float, decimals: int) -> str:
    text = f"{value:.{decimals}f}"
    if float(text) == 0.0:  # never write "-0.000"
        text = text.lstrip("-")
    return text


def _format_header(header: StringHeader) -> str:
    fields = [header.location, header.date, header.purpose, header.memo]
    return WRITE_SEPARATOR.join(fields).rstrip(" ")


def _format_axis_record(axis: AxisRecord | None, decimals: int) -> str:
    axis = axis or AxisRecord()
    values = [*_internal_to_file(axis.start), *_internal_to_file(axis.end)]
    fields = [str(SEPARATOR_STRING_NUMBER)] + [_fixed(v, decimals) for v in values]
    return WRITE_SEPARATOR.join(fields)


def _format_point_record(number: int, xyz, descriptions, decimals: int) -> str:
    fields = [str(number)] + [_fixed(v, decimals) for v in _internal_to_file(xyz)]
    fields.append(WRITE_DESCRIPTION_SEPARATOR.join(descriptions))
    return WRITE_SEPARATOR.join(fields).rstrip(" ")


def _format_separator_record(decimals: int, *, end: bool) -> str:
    fields = [str(SEPARATOR_STRING_NUMBER)] + [_fixed(0.0, decimals)] * 3
    fields.append(KEYWORD_END if end else "")
    return WRITE_SEPARATOR.join(fields).rstrip(" ")


def _split_record(line: str) -> list[str]:
    """Fields of one record. Comma separated; whitespace separated if no comma."""
    if FIELD_SEPARATOR in line:
        return [t.strip() for t in line.split(FIELD_SEPARATOR)]
    return line.split()


def _parse_string_number(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        value = parse_float(token)
        if not math.isfinite(value) or value != int(value):
            raise ValueError(f"string number {token!r} is not an integer") from None
        return int(value)


def _parse_point_record(fields: list[str]):
    """``(number, (x, y, z), descriptions)`` from the fields of a point record.

    Raises ValueError when the record has no string number plus three
    coordinates.
    """
    if len(fields) < 4:
        raise ValueError("fewer than four fields")
    number = _parse_string_number(fields[0])
    columns = [parse_float(t) for t in fields[1:4]]
    if not all(math.isfinite(c) for c in columns):
        raise ValueError("non-finite coordinate")
    return number, _file_to_internal(columns), tuple(fields[4:])


def _is_end_record(fields: list[str]) -> bool:
    return any(f.upper() == KEYWORD_END for f in fields)


# --------------------------------------------------------------------- read


def parse_surpac_str(text: str, *, strict: bool = False, source: str = "<string>") -> StringFile:
    """Parse the text of a Surpac string file.

    Tolerated on reading (each case adds a note to ``StringFile.warnings`` when
    it is a real deviation, and raises :class:`SurpacFormatError` instead when
    ``strict`` is True):

    * any amount of whitespace around fields, tabs, blank lines, Windows or
      old Mac line endings, a UTF-8 byte-order mark;
    * trailing commas and missing description fields;
    * numbers in scientific notation (``5.3E+06``, also Fortran ``5.3D+06``),
      string numbers written as ``1.0``;
    * records separated by whitespace only (no commas at all);
    * keyword ``END`` in any case, or alone on a line;
    * a missing axis record, a missing header line (warning);
    * a missing separator between segments with different string numbers
      (warning; the segment is split where the number changes);
    * a missing ``END`` record (warning);
    * lines that cannot be parsed, and anything after ``END``: skipped and
      listed in ``StringFile.skipped`` (warning).

    Returns a :class:`StringFile` in internal coordinates (east, north, up).
    """
    out = StringFile()
    notes: list[str] = out.warnings

    def note(message: str) -> None:
        if strict:
            raise SurpacFormatError(f"{source}: {message}")
        notes.append(message)

    lines = [(i, ln.strip()) for i, ln in enumerate(text.splitlines(), start=1)]
    lines = [(i, ln) for i, ln in lines if ln and ln != "\x1a"]
    pos = 0

    # Header line.
    if lines:
        i, line = lines[0]
        fields = _split_record(line)
        try:
            _parse_point_record(fields)
            looks_like_data = True
        except ValueError:
            looks_like_data = False
        if looks_like_data:
            note(f"line {i}: no header line; the file starts with a record")
        else:
            parts = [p.strip() for p in line.split(FIELD_SEPARATOR, len(HEADER_FIELDS) - 1)]
            parts += [""] * (len(HEADER_FIELDS) - len(parts))
            out.header = StringHeader(*parts)
            pos = 1
    else:
        note("file is empty")

    # Axis record.
    if pos < len(lines):
        i, line = lines[pos]
        fields = _split_record(line)
        is_axis = False
        if not _is_end_record(fields):
            try:
                is_axis = _parse_string_number(fields[0]) == SEPARATOR_STRING_NUMBER
            except (ValueError, IndexError):
                is_axis = False
        if is_axis:
            values = []
            for token in fields[1:7]:
                try:
                    values.append(parse_float(token))
                except ValueError:
                    values.append(0.0)
            values += [0.0] * (6 - len(values))
            out.axis = AxisRecord(_file_to_internal(values[:3]), _file_to_internal(values[3:]))
            pos += 1
        else:
            note(f"line {i}: no axis record after the header")

    current_number: int | None = None
    points: list[tuple[float, float, float]] = []
    descriptions: list[tuple[str, ...]] = []

    def flush() -> None:
        nonlocal current_number
        if points:
            out.segments.append(StringSegment(current_number, points, descriptions))
        current_number = None
        points.clear()
        descriptions.clear()

    ended = False
    n_read = 0  # point records so far = row of the next point in all_points()
    while pos < len(lines):
        i, line = lines[pos]
        pos += 1
        fields = _split_record(line)
        if fields and fields[0].upper() == KEYWORD_END:
            ended = True
            break
        try:
            number, xyz, desc = _parse_point_record(fields)
        except ValueError as exc:
            if _is_end_record(fields):
                ended = True
                break
            if strict:
                raise SurpacFormatError(f"{source}: line {i}: {exc}: {line!r}") from None
            out.skipped.append((i, line))
            out.record_rows.append(-1)
            continue
        if number == SEPARATOR_STRING_NUMBER:
            if _is_end_record(list(desc)):
                ended = True
                break
            if any(c != 0.0 for c in xyz):
                note(f"line {i}: string number 0 with non-zero coordinates; taken as a separator")
            flush()
            out.record_rows.append(-1)
            continue
        if current_number is not None and number != current_number:
            note(
                f"line {i}: string number changes from {current_number} to {number} "
                "without a separator record; segment split"
            )
            flush()
        current_number = number
        points.append(xyz)
        descriptions.append(desc)
        out.record_rows.append(n_read)
        n_read += 1
    flush()

    if not ended:
        note(f"no {KEYWORD_END} record at the end of the file")
    elif pos < len(lines):
        if strict:
            raise SurpacFormatError(f"{source}: line {lines[pos][0]}: data after {KEYWORD_END}")
        out.skipped.extend(lines[pos:])
    if out.skipped:
        first = out.skipped[0]
        notes.append(
            f"{len(out.skipped)} line(s) skipped; first is line {first[0]}: {first[1][:60]!r}"
        )
    for message in notes:
        warnings.warn(f"{source}: {message}", FormatWarning, stacklevel=3)
    return out


def read_surpac_str(path, *, strict: bool = False) -> StringFile:
    """Read a Surpac text string file. See :func:`parse_surpac_str`.

    Raises :class:`BinaryFileError` for a binary ``.str``.
    """
    text = read_text(path, format_name=".str")
    return parse_surpac_str(text, strict=strict, source=os.fspath(path))


# -------------------------------------------------------------------- write


def _check_text_field(value: str, what: str, *, allow_separator: bool = False) -> None:
    if len((value + "x").splitlines()) != 1:  # any character str.splitlines breaks at
        raise SurpacFormatError(f"{what} contains a line break: {value!r}")
    if not allow_separator and FIELD_SEPARATOR in value:
        raise SurpacFormatError(f"{what} contains a comma, which the format cannot hold: {value!r}")


def format_surpac_str(strings, *, decimals: int = 3) -> str:
    """Text of a Surpac string file (lines joined with ``\\n``, final newline).

    ``strings`` is a :class:`StringFile` or an iterable of
    :class:`StringSegment`. Coordinates are written with exactly ``decimals``
    decimals (default 3 = millimetres); reading the result back gives the
    coordinates rounded to that precision.

    Strict: raises :class:`SurpacFormatError` for an empty segment, a string
    number below 1, a non-finite coordinate, or a header field or description
    containing a comma or line break (the memo may contain commas) — anything
    that would not read back unchanged. Header fields are stripped.
    """
    if decimals < 0:
        raise ValueError("decimals must be >= 0")
    if not isinstance(strings, StringFile):
        strings = StringFile(segments=list(strings))
    header = strings.header
    for name in HEADER_FIELDS:
        _check_text_field(getattr(header, name), f"header {name}", allow_separator=name == "memo")
    header = StringHeader(*(getattr(header, n).strip() for n in HEADER_FIELDS))

    header_line = _format_header(header)
    try:  # the reader would take a header that parses as a point record for data
        _parse_point_record(_split_record(header_line))
    except ValueError:
        pass
    else:
        raise SurpacFormatError(
            f"header {header_line!r} would be read back as a point record; "
            "make its first field non-numeric"
        )
    lines = [header_line, _format_axis_record(strings.axis, decimals)]
    for k, segment in enumerate(strings.segments):
        if segment.n_points == 0:
            raise SurpacFormatError(f"segment {k} has no points")
        if segment.number <= SEPARATOR_STRING_NUMBER:
            raise SurpacFormatError(f"segment {k}: string number {segment.number} is below 1")
        if not np.isfinite(segment.points).all():
            raise SurpacFormatError(f"segment {k} has NaN or infinite coordinates")
        for xyz, desc in zip(segment.points.tolist(), segment.descriptions, strict=True):
            for d in desc:
                _check_text_field(d, f"segment {k} description")
            lines.append(_format_point_record(segment.number, xyz, desc, decimals))
        lines.append(_format_separator_record(decimals, end=False))
    lines.append(_format_separator_record(decimals, end=True))
    return "\n".join(lines) + "\n"


def write_surpac_str(path, strings, *, decimals: int = 3, newline: str = "\n") -> None:
    """Write a Surpac text string file. See :func:`format_surpac_str`.

    ``newline`` is the line terminator (``"\\r\\n"`` for Windows-style files).
    """
    text = format_surpac_str(strings, decimals=decimals)
    if newline != "\n":
        text = text.replace("\n", newline)
    Path(path).write_bytes(text.encode("utf-8"))
