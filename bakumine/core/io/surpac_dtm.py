"""Surpac DTM / solid file ``.dtm`` (text): read and write, with its ``.str``.

Written to the PUBLISHED layout; not yet checked against a current export.
==========================================================================

A ``.dtm`` holds only topology: triangles as index triples into the points of
the paired ``.str`` file (same name, other extension). It is meaningless
without that file, so reading takes both and writing produces both.

Source of the layout: "Appendix D - File Formats" of the Surpac manual (1996,
Surpac Software International), public at
https://www.cse.unr.edu/~fredh/papers/working/vr-mining/string.html. It
describes the text ``.str`` and ``.dtm`` files with a small example pair (a
box). Clean room: only that published description is used; the example is not
copied into this repository (the tests use hand-written pairs that follow the
same rules). The document is from 1996; whether current Surpac versions write
the same text is NOT verified (no Surpac export was available).

What the document states, and what this module does:

1. Line 1: ``<string file name>,<checksum>`` where the checksum part
   (``0680000000;algorithm=standard;fields=x,y`` in the example) is optional
   and "should not be included" by converters. We write ``<name>.str,`` and
   never a checksum (the algorithm is not published). The line is not
   interpreted on reading.
2. Line 2 of the example is ``0, 0.000, 0.000, 0.000, END``. We write it
   (:data:`WRITE_PREAMBLE_LINES`); the reader keeps any lines before the first
   ``OBJECT`` record in ``SurpacDtm.preamble``.
3. ``OBJECT, <id>,`` then ``TRISOLATION, <id>, name=value,...``. Both must
   exist. The example's values are ``neighbours=yes,closed=yes,
   direction=solid``; the document calls them optional, assigned by
   processing modules. We write ``neighbours`` and ``closed``, and
   ``direction=solid`` for a closed mesh. ``validated=`` is NOT in the
   document: it is written only when the caller gives it, and read when there.
   The reader accepts yes/no/true/false for any flag, any case.
4. Triangle record: ``triangle, v1, v2, v3, n1, n2, n3``; triangle numbers
   start at one in each trisolation.
5. Vertex numbers count the records of the ``.str`` file from the first
   record after its two header lines, and "you MUST include the null points":
   the zero separator records are counted (:func:`_point_rows`). In the
   example the second ring starts at point 7 after a ring of 5 and its null
   record.
6. Neighbours are triangle numbers of the same trisolation, 0 = none.
   ``n1`` is opposite vertex 3 (across edge ``v1-v2``), ``n2`` opposite vertex
   1 (``v2-v3``), ``n3`` opposite vertex 2 (``v3-v1``):
   :data:`NEIGHBOUR_EDGES`. They may all be zero in converted data.
7. The example has no record between the last triangle and ``END``. We write
   none. A block ends at the next ``TRISOLATION`` / ``OBJECT`` / ``END``.

Still assumptions (the document is silent):

8. The writer puts the vertices of every mesh into the ``.str`` as one
   segment per mesh, all with string number
   :data:`DEFAULT_VERTEX_STRING_NUMBER`.
9. Winding. The document does not state it, but its example box, read with
   the ``Y, X, Z`` columns as north, east, up, has every triangle CLOCKWISE
   seen from outside (signed volume -100^3 in a right-handed east, north, up
   frame). Our meshes are counter-clockwise seen from outside, so the reader
   and the writer swap the second and third vertex of every triangle (and the
   neighbour columns with them): :data:`FILE_WINDING_REVERSED`. Evidence: that
   one example. Whether open surfaces (DTMs) follow the same rule is unknown;
   the same swap is applied to them.
10. A closed string repeats its first point at the end; a file may refer to
    either copy. The reader does not weld points with equal coordinates.
11. A binary variant of ``.dtm`` / ``.str`` exists in Surpac. It is NOT
    supported: binary content is rejected with :class:`BinaryFileError`.

Files of the earlier, provisional writer of this module (before 2026-11-21)
numbered vertices WITHOUT the null records and ended every trisolation with a
record of zeros. The reader recognises them (zero records and no line 2) and
reads them the old way, with a warning; ``indexing=`` forces either way.

All layout knowledge sits in the constants and small record functions directly
below. The reader is tolerant (see :func:`parse_dtm_topology`); the writer is
strict and canonical.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io import surpac_str
from bakumine.core.io._text import (
    BinaryFileError,
    FormatWarning,
    looks_binary,
    parse_float,
    read_text,
)
from bakumine.core.io.strings import StringFile, StringHeader, StringSegment
from bakumine.core.io.surpac_str import SurpacFormatError

__all__ = [
    "BinaryFileError",
    "SurpacDtm",
    "SurpacFormatError",
    "SurpacTrisolation",
    "describe_surpac_file",
    "mesh_neighbours",
    "parse_dtm_topology",
    "read_surpac_dtm",
    "write_surpac_dtm",
]

# --------------------------------------------------------------------------
# Layout knowledge (PROVISIONAL). Change here when a real file says otherwise.
# --------------------------------------------------------------------------

KEYWORD_OBJECT = "OBJECT"
KEYWORD_TRISOLATION = "TRISOLATION"
KEYWORD_END = "END"
FIELD_SEPARATOR = ","
WRITE_SEPARATOR = ", "

FLAG_NEIGHBOURS = "neighbours"
FLAG_VALIDATED = "validated"
FLAG_CLOSED = "closed"
#: Other spellings the reader maps onto the canonical flag names.
FLAG_ALIASES = {"neighbors": FLAG_NEIGHBOURS, "neighbour": FLAG_NEIGHBOURS, "valid": FLAG_VALIDATED}
#: Words the writer uses for (True, False), per flag.
FLAG_WORDS = {
    FLAG_NEIGHBOURS: ("yes", "no"),
    FLAG_VALIDATED: ("true", "false"),
    FLAG_CLOSED: ("yes", "no"),
}
#: Order in which the writer emits the flags of a TRISOLATION record (a flag
#: that is not given is left out).
FLAG_ORDER = (FLAG_NEIGHBOURS, FLAG_VALIDATED, FLAG_CLOSED)
#: Written after the flags of a closed mesh, as in the published example.
FLAG_DIRECTION = "direction"
DIRECTION_SOLID = "solid"
#: Separator between flags in a TRISOLATION record.
WRITE_FLAG_SEPARATOR = ","
_TRUE_WORDS = {"yes", "y", "true", "t", "1", "on"}
_FALSE_WORDS = {"no", "n", "false", "f", "0", "off"}

#: Columns of a triangle record.
TRIANGLE_COLUMNS = ("triangle_id", "v1", "v2", "v3", "n1", "n2", "n3")
#: Index of the first point of the .str file as used by v1..v3.
VERTEX_INDEX_BASE = 1
#: Id of the first triangle of a trisolation, as written.
TRIANGLE_ID_BASE = 1
#: Value of n1..n3 meaning "no neighbour".
NO_NEIGHBOUR = 0
#: Edge (as corner slots of the triangle) that neighbour n1, n2, n3 lies across.
NEIGHBOUR_EDGES = ((0, 1), (1, 2), (2, 0))
#: Triangles in the file are clockwise seen from outside (east, north, up frame);
#: ours are counter-clockwise: vertices 2 and 3 are swapped on reading and writing.
FILE_WINDING_REVERSED = True
#: Column order that swaps the winding of (v1, v2, v3) and of (n1, n2, n3).
_SWAP_VERTICES = [0, 2, 1]
_SWAP_NEIGHBOURS = [2, 1, 0]
#: String number the writer gives to the vertex segments in the .str file.
DEFAULT_VERTEX_STRING_NUMBER = 1
#: Lines the writer puts between the header and the first OBJECT record.
WRITE_PREAMBLE_LINES: tuple[str, ...] = ("0, 0.000, 0.000, 0.000, END",)
#: Whether the writer ends a trisolation with a record of zeros (the published
#: example has none; the provisional writer of this module wrote one).
WRITE_ZERO_RECORD = False
#: Whether records end with a trailing separator (``1, 2, 3, 4, 0, 0, 0,``).
WRITE_TRAILING_SEPARATOR = True


def _finish_record(fields: Sequence[str]) -> str:
    line = WRITE_SEPARATOR.join(fields)
    return line + FIELD_SEPARATOR if WRITE_TRAILING_SEPARATOR else line


def _format_object_record(object_id: int) -> str:
    return _finish_record([KEYWORD_OBJECT, str(object_id)])


def _format_trisolation_record(trisolation_id: int, flags: dict[str, bool]) -> str:
    words = [
        f"{name}={FLAG_WORDS[name][0 if flags[name] else 1]}"
        for name in FLAG_ORDER
        if flags.get(name) is not None
    ]
    if flags.get(FLAG_CLOSED):
        words.append(f"{FLAG_DIRECTION}={DIRECTION_SOLID}")
    return WRITE_SEPARATOR.join(
        [KEYWORD_TRISOLATION, str(trisolation_id), WRITE_FLAG_SEPARATOR.join(words)]
    )


def _format_triangle_records(faces: np.ndarray, neighbours: np.ndarray, offset: int) -> list[str]:
    """Triangle records of one trisolation.

    ``faces`` are 0-based rows of this mesh's vertices, ``offset`` the number
    of .str records (points and null records) before its first vertex, ``neighbours`` 0-based face
    indices per edge slot (-1 = none).
    """
    ids = np.arange(len(faces)) + TRIANGLE_ID_BASE
    v = faces + offset + VERTEX_INDEX_BASE
    n = np.where(neighbours >= 0, neighbours + TRIANGLE_ID_BASE, NO_NEIGHBOUR)
    table = np.column_stack([ids, v, n]).tolist()
    return [_finish_record([str(x) for x in row]) for row in table]


def _format_zero_record() -> str:
    return _finish_record(["0"] * len(TRIANGLE_COLUMNS))


def _point_rows(vertex_ids: np.ndarray, record_rows: np.ndarray | None = None) -> np.ndarray:
    """Rows of ``StringFile.all_points()`` addressed by the v columns of a .dtm.

    With ``record_rows`` (``StringFile.record_rows``) the numbers count every
    record of the ``.str`` after its two header lines, null records included
    (the published rule); a number that names a null record or no record gives
    -1. Without it they count point records only (the old provisional rule).
    """
    index = vertex_ids - VERTEX_INDEX_BASE
    if record_rows is None:
        return index
    inside = (index >= 0) & (index < len(record_rows))
    rows = np.full(index.shape, -1, dtype=np.int64)
    rows[inside] = record_rows[index[inside]]
    return rows


def mesh_neighbours(mesh: Mesh) -> np.ndarray:
    """(m, 3) neighbour face index per neighbour column, -1 = none.

    Column ``k`` is the face across the edge given by ``NEIGHBOUR_EDGES[k]``,
    taken from the mesh edge map. Boundary edges and edges with more than two
    faces give -1.
    """
    em = mesh.edge_map
    across = np.full(3 * mesh.n_faces, -1, dtype=np.int64)
    pairs = em.manifold_halfedge_pairs
    across[pairs[:, 0]] = pairs[:, 1] // 3
    across[pairs[:, 1]] = pairs[:, 0] // 3
    across = across.reshape(-1, 3)  # slot j: edge faces[f, j] -> faces[f, (j + 1) % 3]
    slot_of = {(0, 1): 0, (1, 0): 0, (1, 2): 1, (2, 1): 1, (2, 0): 2, (0, 2): 2}
    columns = [slot_of[edge] for edge in NEIGHBOUR_EDGES]
    return across[:, columns]


# -------------------------------------------------------------------- model


@dataclass
class SurpacTrisolation:
    """One trisolation of a .dtm: a mesh with its ids and flags.

    ``neighbours`` is (m, 3) int64: for every face the index (0-based, into
    ``mesh.faces``) of the face named by the n1, n2, n3 columns, -1 = none. It
    is what the FILE says, not recomputed. ``flags`` holds every ``key=value``
    of the TRISOLATION record as text (keys in lower case), including ones
    this module does not know.
    """

    mesh: Mesh
    object_id: int = 1
    trisolation_id: int = 1
    neighbours: np.ndarray | None = None
    has_neighbours: bool | None = None
    validated: bool | None = None
    closed: bool | None = None
    flags: dict[str, str] = field(default_factory=dict)


@dataclass
class SurpacDtm:
    """Result of reading a ``.dtm`` + ``.str`` pair."""

    trisolations: list[SurpacTrisolation] = field(default_factory=list)
    header: str = ""
    preamble: list[str] = field(default_factory=list)
    strings: StringFile | None = None
    warnings: list[str] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)

    @property
    def meshes(self) -> list[Mesh]:
        """One mesh per trisolation, in file order."""
        return [t.mesh for t in self.trisolations]

    @property
    def ids(self) -> list[tuple[int, int]]:
        """``(object_id, trisolation_id)`` per trisolation, in file order."""
        return [(t.object_id, t.trisolation_id) for t in self.trisolations]

    def get(self, object_id: int, trisolation_id: int = 1) -> SurpacTrisolation:
        for t in self.trisolations:
            if (t.object_id, t.trisolation_id) == (object_id, trisolation_id):
                return t
        raise KeyError((object_id, trisolation_id))

    def __iter__(self):
        return iter(self.trisolations)

    def __len__(self) -> int:
        return len(self.trisolations)


@dataclass
class _RawTrisolation:
    """A trisolation exactly as the file states it (ids, not array rows)."""

    object_id: int
    trisolation_id: int
    flags: dict[str, str]
    line: int
    records: list[list[int]] = field(default_factory=list)

    def arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(triangle_ids (m,), vertex_ids (m, 3), neighbour_ids (m, 3))``."""
        if not self.records:
            empty = np.zeros((0, 3), dtype=np.int64)
            return np.zeros(0, dtype=np.int64), empty, empty.copy()
        table = np.array(self.records, dtype=np.int64)
        return table[:, 0], table[:, 1:4], table[:, 4:7]


@dataclass
class _RawDtm:
    header: str = ""
    preamble: list[str] = field(default_factory=list)
    trisolations: list[_RawTrisolation] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)
    ended: bool = False
    zero_records: int = 0


# --------------------------------------------------------------------- read


def _parse_int(token: str) -> int:
    try:
        return int(token)
    except ValueError:
        value = parse_float(token)
        if value != int(value):
            raise ValueError(f"{token!r} is not an integer") from None
        return int(value)


def _parse_flag(value: str) -> bool | None:
    word = value.strip().lower()
    if word in _TRUE_WORDS:
        return True
    if word in _FALSE_WORDS:
        return False
    return None


def _parse_flags(fields: list[str]) -> dict[str, str]:
    flags: dict[str, str] = {}
    for item in fields:
        for part in item.split():
            key, sep, value = part.partition("=")
            if not sep:
                continue
            key = key.strip().lower()
            flags[FLAG_ALIASES.get(key, key)] = value.strip()
    return flags


def parse_dtm_topology(text: str, *, strict: bool = False, source: str = "<string>") -> _RawDtm:
    """Parse the text of a ``.dtm`` without its ``.str``: blocks, flags, records.

    Tolerated on reading (deviations add a note to ``warnings`` and raise
    :class:`SurpacFormatError` instead when ``strict`` is True):

    * keywords in any case; any whitespace; blank lines; trailing commas;
      Windows line endings; records separated by whitespace only;
    * extra lines between the header and the first ``OBJECT`` record (kept in
      ``preamble``, no warning);
    * flags in any order, ``yes/no/true/false/y/n/1/0`` for any flag,
      ``neighbors`` for ``neighbours``, unknown flags (kept as text), flags
      separated by spaces or commas, no flags at all;
    * triangle records without the neighbour columns (taken as no neighbours)
      or with extra columns (ignored);
    * a ``TRISOLATION`` with no ``OBJECT`` before it, triangle records with no
      ``TRISOLATION`` before them, a missing id (warning; ids are made up);
    * a record of zeros at the end of a trisolation (counted in
      ``zero_records``; the published layout has none), a missing ``END``
      (warning);
    * lines that cannot be parsed and anything after ``END``: skipped and
      listed in ``skipped`` (warning).
    """
    raw = _RawDtm()

    def note(message: str) -> None:
        if strict:
            raise SurpacFormatError(f"{source}: {message}")
        raw.warnings.append(message)

    lines = [(i, ln.strip()) for i, ln in enumerate(text.splitlines(), start=1)]
    lines = [(i, ln) for i, ln in lines if ln and ln != "\x1a"]
    if not lines:
        note("file is empty")
        return raw

    def split(line: str) -> list[str]:
        if FIELD_SEPARATOR in line:
            fields = [t.strip() for t in line.split(FIELD_SEPARATOR)]
        else:
            fields = line.split()
        while fields and fields[-1] == "":
            fields.pop()
        return fields

    pos = 0
    first = split(lines[0][1])
    if first and first[0].upper() in (KEYWORD_OBJECT, KEYWORD_TRISOLATION):
        note("no header line; the file starts with a block record")
    else:
        raw.header = lines[0][1]
        pos = 1

    object_id: int | None = None
    block: _RawTrisolation | None = None
    seen_block = False

    def open_block(trisolation_id: int | None, flags: dict[str, str], line: int) -> None:
        nonlocal block, object_id, seen_block
        if object_id is None:
            object_id = 1
            note(f"line {line}: {KEYWORD_TRISOLATION} before any {KEYWORD_OBJECT}; object 1 used")
        if trisolation_id is None:
            used = [t.trisolation_id for t in raw.trisolations if t.object_id == object_id]
            trisolation_id = max(used, default=0) + 1
        block = _RawTrisolation(object_id, trisolation_id, flags, line)
        raw.trisolations.append(block)
        seen_block = True

    while pos < len(lines):
        i, line = lines[pos]
        pos += 1
        fields = split(line)
        keyword = fields[0].upper() if fields else ""
        if keyword == KEYWORD_END:
            raw.ended = True
            break
        if keyword == KEYWORD_OBJECT:
            try:
                object_id = _parse_int(fields[1])
            except (IndexError, ValueError):
                object_id = max((t.object_id for t in raw.trisolations), default=0) + 1
                note(f"line {i}: {KEYWORD_OBJECT} record without an id; {object_id} assumed")
            block = None
            seen_block = True
            continue
        if keyword == KEYWORD_TRISOLATION:
            rest = fields[1:]
            trisolation_id = None
            if rest and "=" not in rest[0]:
                try:
                    trisolation_id = _parse_int(rest[0].split()[0])
                except (ValueError, IndexError):
                    trisolation_id = None
                rest = [" ".join(rest[0].split()[1:])] + rest[1:]
            flags = _parse_flags(rest)
            if trisolation_id is None:
                note(f"line {i}: {KEYWORD_TRISOLATION} record without an id; numbered in order")
            open_block(trisolation_id, flags, i)
            continue
        try:
            values = [_parse_int(t) for t in fields]
        except ValueError:
            values = None
        if not seen_block:
            raw.preamble.append(line)  # e.g. a string-style "0, 0.000, 0.000, 0.000, END"
            continue
        if values is None or len(values) < 4:
            if strict:
                raise SurpacFormatError(f"{source}: line {i}: not a triangle record: {line!r}")
            raw.skipped.append((i, line))
            continue
        if values[0] == 0 and not any(values[1:4]):
            block = None
            raw.zero_records += 1
            continue
        if block is None:
            note(f"line {i}: triangle record outside a {KEYWORD_TRISOLATION}; one is assumed")
            open_block(None, {}, i)
        row = values[:7] + [NO_NEIGHBOUR] * (7 - len(values[:7]))
        block.records.append(row)

    if not raw.ended:
        note(f"no {KEYWORD_END} record at the end of the file")
    elif pos < len(lines):
        if strict:
            raise SurpacFormatError(f"{source}: line {lines[pos][0]}: data after {KEYWORD_END}")
        raw.skipped.extend(lines[pos:])
    if raw.skipped:
        first_skipped = raw.skipped[0]
        raw.warnings.append(
            f"{len(raw.skipped)} line(s) skipped; first is line {first_skipped[0]}: "
            f"{first_skipped[1][:60]!r}"
        )
    return raw


def _neighbour_rows(triangle_ids: np.ndarray, neighbour_ids: np.ndarray) -> tuple[np.ndarray, int]:
    """Neighbour triangle ids -> 0-based face rows (-1 = none); count of unknown ids."""
    out = np.full(neighbour_ids.shape, -1, dtype=np.int64)
    if len(triangle_ids) == 0:
        return out, 0
    order = np.argsort(triangle_ids, kind="stable")
    sorted_ids = triangle_ids[order]
    wanted = neighbour_ids != NO_NEIGHBOUR
    where = np.searchsorted(sorted_ids, neighbour_ids)
    where = np.clip(where, 0, len(sorted_ids) - 1)
    found = wanted & (sorted_ids[where] == neighbour_ids)
    out[found] = order[where[found]]
    return out, int((wanted & ~found).sum())


def _paired_path(path: Path, suffix: str) -> Path:
    """Sibling file with another extension, matching the case found on disk."""
    for candidate in (path.with_suffix(suffix), path.with_suffix(suffix.upper())):
        if candidate.exists():
            return candidate
    return path.with_suffix(suffix)


def read_surpac_dtm(
    dtm_path, str_path=None, *, strict: bool = False, indexing: str = "auto"
) -> SurpacDtm:
    """Read a Surpac text ``.dtm`` together with its ``.str``.

    Parameters
    ----------
    dtm_path : path
        The topology file.
    str_path : path, optional
        The paired string file; default is ``dtm_path`` with extension ``.str``.
    strict : bool
        Raise :class:`SurpacFormatError` on any deviation from the layout
        instead of recording a warning.
    indexing : {"auto", "records", "points"}
        How the vertex numbers count the ``.str``: ``"records"`` counts every
        record after the two header lines, null records included (the
        published rule); ``"points"`` counts point records only (files of this
        module's provisional writer). ``"auto"`` takes ``"points"`` for a file
        that looks like one of those (records of zeros, no line 2) and
        ``"records"`` otherwise.

    Returns
    -------
    SurpacDtm
        ``.trisolations`` holds one :class:`SurpacTrisolation` (mesh, object
        id, trisolation id, flags, neighbours) per trisolation; ``.meshes``
        and ``.ids`` are shortcuts.

    Every mesh is in internal coordinates (east, north, up) and holds only the
    points its triangles use, in ``.str`` order; triangle order is as in the file
    (winding: see the module docstring, item 9). Vertex numbers that name a null
    record or no record of the ``.str`` raise :class:`SurpacFormatError` — the symptom of files that
    are not a pair, or of the other counting rule (``indexing``).
    Raises :class:`BinaryFileError` if either file is binary.
    """
    dtm_path = Path(dtm_path)
    str_path = _paired_path(dtm_path, ".str") if str_path is None else Path(str_path)
    source = os.fspath(dtm_path)
    raw = parse_dtm_topology(read_text(dtm_path, format_name=".dtm"), strict=strict, source=source)
    if not str_path.exists():
        raise FileNotFoundError(
            f"{str_path}: the string file paired with {dtm_path.name} is missing; "
            "a .dtm holds only triangle indices and cannot be read without it"
        )
    strings = surpac_str.read_surpac_str(str_path, strict=strict)
    points = strings.all_points()
    if indexing not in ("auto", "records", "points"):
        raise ValueError(f"indexing must be 'auto', 'records' or 'points', got {indexing!r}")
    if indexing == "auto":
        legacy = raw.zero_records > 0 and not raw.preamble
        indexing = "points" if legacy else "records"
        if legacy and strings.n_segments > 1:
            raw.warnings.append(
                "the file ends its trisolations with records of zeros and has no second "
                "header line: read as a file of the provisional BakuMine writer (vertex "
                "numbers do not count the null records of the .str); pass "
                "indexing='records' for the published rule"
            )
    record_rows = np.asarray(strings.record_rows, dtype=np.int64) if indexing == "records" else None

    out = SurpacDtm(
        header=raw.header,
        preamble=raw.preamble,
        strings=strings,
        warnings=raw.warnings,
        skipped=raw.skipped,
    )
    for block in raw.trisolations:
        name = f"object {block.object_id} trisolation {block.trisolation_id}"
        triangle_ids, vertex_ids, neighbour_ids = block.arrays()
        rows = _point_rows(vertex_ids, record_rows)
        if rows.size and (rows.min() < 0 or rows.max() >= len(points)):
            bad = vertex_ids[(rows < 0) | (rows >= len(points))]
            counted = (
                f"{len(strings.record_rows)} records ({len(points)} points and the null records)"
                if record_rows is not None
                else f"{len(points)} point records"
            )
            raise SurpacFormatError(
                f"{source}: {name} uses point numbers {int(vertex_ids.min())}.."
                f"{int(vertex_ids.max())}; {len(np.unique(bad))} of them "
                f"(first: {int(bad.flat[0])}) name a null record or no record of "
                f"{str_path.name}, which has {counted}. "
                "Either the files are not a pair or the vertex numbers count the other way "
                "(indexing='points' / 'records'); run describe_surpac_file() on both files."
            )
        if FILE_WINDING_REVERSED and record_rows is not None:  # not for the provisional files
            rows = rows[:, _SWAP_VERTICES]
            neighbour_ids = neighbour_ids[:, _SWAP_NEIGHBOURS]
        used = np.unique(rows)
        remap = np.full(len(points), -1, dtype=np.int64)
        remap[used] = np.arange(len(used))
        mesh = Mesh(points[used], remap[rows], check=False)
        if len(np.unique(triangle_ids)) != len(triangle_ids):
            out.warnings.append(f"{name}: duplicate triangle ids; neighbours may be wrong")
        neighbours, unknown = _neighbour_rows(triangle_ids, neighbour_ids)
        if unknown:
            out.warnings.append(f"{name}: {unknown} neighbour id(s) name no triangle; set to none")
        flags = block.flags
        out.trisolations.append(
            SurpacTrisolation(
                mesh=mesh,
                object_id=block.object_id,
                trisolation_id=block.trisolation_id,
                neighbours=neighbours,
                has_neighbours=_parse_flag(flags.get(FLAG_NEIGHBOURS, "")),
                validated=_parse_flag(flags.get(FLAG_VALIDATED, "")),
                closed=_parse_flag(flags.get(FLAG_CLOSED, "")),
                flags=dict(flags),
            )
        )
    if strict and out.warnings:
        raise SurpacFormatError(f"{source}: {out.warnings[0]}")
    for message in out.warnings:
        warnings.warn(f"{source}: {message}", FormatWarning, stacklevel=2)
    return out


# -------------------------------------------------------------------- write


def write_surpac_dtm(
    dtm_path,
    meshes,
    str_path=None,
    *,
    object_ids: Sequence[int] | None = None,
    trisolation_ids: Sequence[int] | None = None,
    validated: bool | Sequence[bool] | None = None,
    write_neighbours: bool = True,
    decimals: int = 3,
    string_number: int = DEFAULT_VERTEX_STRING_NUMBER,
    header: StringHeader | None = None,
    newline: str = "\n",
) -> None:
    """Write meshes as a Surpac text ``.dtm`` + ``.str`` pair.

    Parameters
    ----------
    dtm_path : path
        Topology file to write.
    meshes : Mesh, SurpacTrisolation, or a sequence of them
        One trisolation is written per item.
    str_path : path, optional
        Point file to write; default is ``dtm_path`` with extension ``.str``.
    object_ids, trisolation_ids : sequences of int, optional
        Ids per mesh. Defaults: the ids of a :class:`SurpacTrisolation` item;
        for plain meshes each mesh becomes its own object (1, 2, ...) holding
        trisolation 1. Meshes that share an object id must be adjacent.
    validated : bool or sequence of bool, optional
        Value of a ``validated`` flag. The published layout has no such flag:
        it is written only when given here or set on a
        :class:`SurpacTrisolation` item, and left out otherwise.
    write_neighbours : bool
        Compute neighbours from the mesh edge map and write them with
        ``neighbours=yes`` (default). False writes zeros and ``neighbours=no``.
    decimals : int
        Decimals of the coordinates in the ``.str`` (default 3).
    string_number : int
        String number of the vertex segments in the ``.str``.
    header : StringHeader, optional
        Header for both files; location defaults to the ``.str`` file name.

    The ``closed`` flag is ``mesh.is_closed``. All vertices of each mesh are
    written (also unreferenced ones); the reader returns only referenced ones.
    Vertices are not welded or rounded apart: two vertices that become equal
    at the written precision stay two points, so topology survives a round trip.
    """
    dtm_path = Path(dtm_path)
    str_path = dtm_path.with_suffix(".str") if str_path is None else Path(str_path)
    if isinstance(meshes, Mesh | SurpacTrisolation):
        meshes = [meshes]
    items = list(meshes)
    n = len(items)

    def per_item(values, name):
        if values is None:
            return [None] * n
        values = list(values)
        if len(values) != n:
            raise ValueError(f"{name} must have one entry per mesh")
        return values

    object_ids = per_item(object_ids, "object_ids")
    trisolation_ids = per_item(trisolation_ids, "trisolation_ids")
    if validated is None or isinstance(validated, bool):
        validated = [validated] * n
    validated = per_item(validated, "validated")

    blocks: list[tuple[int, int, Mesh, bool | None]] = []
    for k, item in enumerate(items):
        if isinstance(item, SurpacTrisolation):
            mesh = item.mesh
            oid = item.object_id if object_ids[k] is None else object_ids[k]
            tid = item.trisolation_id if trisolation_ids[k] is None else trisolation_ids[k]
            flag = item.validated if validated[k] is None else validated[k]
        else:
            mesh = item
            oid = k + 1 if object_ids[k] is None else object_ids[k]
            tid = 1 if trisolation_ids[k] is None else trisolation_ids[k]
            flag = validated[k]
        if not isinstance(mesh, Mesh):
            raise TypeError(f"item {k} is not a Mesh")
        if mesh.n_faces == 0:
            raise SurpacFormatError(f"mesh {k} has no faces")
        if int(oid) < 1 or int(tid) < 1:
            raise SurpacFormatError(f"mesh {k}: object and trisolation ids must be >= 1")
        blocks.append((int(oid), int(tid), mesh, None if flag is None else bool(flag)))

    keys = [(oid, tid) for oid, tid, _, _ in blocks]
    if len(set(keys)) != len(keys):
        raise SurpacFormatError(f"duplicate (object, trisolation) ids: {keys}")
    finished: set[int] = set()
    for k, (oid, _) in enumerate(keys):
        if oid in finished:
            raise SurpacFormatError(f"meshes of object {oid} are not adjacent in the input")
        if k + 1 < n and keys[k + 1][0] != oid:
            finished.add(oid)

    header = header or StringHeader()
    if not header.location.strip():
        header = StringHeader(str_path.name, header.date, header.purpose, header.memo)

    segments = [StringSegment(string_number, mesh.vertices) for _, _, mesh, _ in blocks]
    surpac_str.write_surpac_str(
        str_path, StringFile(segments=segments, header=header), decimals=decimals, newline=newline
    )

    surpac_str._check_text_field(str_path.name, "string file name")
    lines = [str_path.name + FIELD_SEPARATOR]  # "<string file>,<checksum>" without a checksum
    lines.extend(WRITE_PREAMBLE_LINES)
    offset = 0
    previous_object = None
    for oid, tid, mesh, flag in blocks:
        if oid != previous_object:
            lines.append(_format_object_record(oid))
            previous_object = oid
        flags = {
            FLAG_NEIGHBOURS: write_neighbours,
            FLAG_VALIDATED: flag,
            FLAG_CLOSED: mesh.is_closed,
        }
        lines.append(_format_trisolation_record(tid, flags))
        if write_neighbours:
            neighbours = mesh_neighbours(mesh)
        else:
            neighbours = np.full((mesh.n_faces, 3), -1, dtype=np.int64)
        faces = mesh.faces
        if FILE_WINDING_REVERSED:
            faces, neighbours = faces[:, _SWAP_VERTICES], neighbours[:, _SWAP_NEIGHBOURS]
        lines.extend(_format_triangle_records(faces, neighbours, offset))
        if WRITE_ZERO_RECORD:
            lines.append(_format_zero_record())
        offset += mesh.n_vertices + 1  # the points of the segment and its null record
    lines.append(KEYWORD_END)
    dtm_path.write_bytes((newline.join(lines) + newline).encode("utf-8"))


# ----------------------------------------------------------------- describe


def describe_surpac_file(path, *, max_records: int = 5) -> str:
    """Plain-text dump of what the parser understood of a ``.str`` or ``.dtm``.

    The debugging tool for the first real Surpac export: it never raises on
    content (a binary file is reported, not rejected), shows the raw first
    lines next to the parsed result, counts, index ranges, flags as written,
    every warning and the first skipped lines. For a ``.dtm`` it also looks
    for the paired ``.str`` and compares the largest vertex index with the
    number of point records, which is the test of the indexing assumption.
    """
    path = Path(path)
    out = [f"file: {path}"]
    data = path.read_bytes()
    out.append(f"size: {len(data)} bytes")
    if looks_binary(data):
        out.append("kind: BINARY (not supported; re-save as text in Surpac)")
        out.append(f"first bytes: {data[:32].hex(' ')}")
        return "\n".join(out)
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    raw_lines = text.splitlines()
    crlf = text.count("\r\n")
    out.append(f"lines: {len(raw_lines)} ({'CRLF' if crlf else 'LF'} line endings)")
    out.append(f"first {min(max_records, len(raw_lines))} raw lines:")
    out.extend(f"  {i + 1}: {ln!r}" for i, ln in enumerate(raw_lines[:max_records]))

    upper = text[:20000].upper()
    is_dtm = path.suffix.lower() == ".dtm" or (
        path.suffix.lower() != ".str" and (KEYWORD_TRISOLATION in upper or KEYWORD_OBJECT in upper)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", FormatWarning)
        if is_dtm:
            _describe_dtm(path, text, out, max_records)
        else:
            _describe_str(text, out, max_records)
    return "\n".join(out)


def _describe_notes(notes: list[str], skipped: list[tuple[int, str]], out: list[str], limit: int):
    out.append(f"warnings: {len(notes)}")
    out.extend(f"  - {w}" for w in notes)
    out.append(f"skipped lines: {len(skipped)}")
    out.extend(f"  {i}: {ln!r}" for i, ln in skipped[:limit])


def _describe_str(text: str, out: list[str], limit: int) -> None:
    out.append("kind: Surpac string file (.str), text  [PROVISIONAL parser]")
    sf = surpac_str.parse_surpac_str(text)
    h = sf.header
    out.append(
        f"header: location={h.location!r} date={h.date!r} purpose={h.purpose!r} memo={h.memo!r}"
    )
    if sf.axis is None:
        out.append("axis record: none found")
    else:
        out.append(
            f"axis record (east, north, up): {sf.axis.start.tolist()} -> {sf.axis.end.tolist()}"
        )
    out.append(f"segments: {sf.n_segments}; point records: {sf.n_points}")
    for number in sf.string_numbers:
        segs = sf.segments_of(number)
        n_closed = sum(s.is_closed for s in segs)
        out.append(
            f"  string {number}: {len(segs)} segment(s), {sum(s.n_points for s in segs)} point(s), "
            f"{n_closed} closed"
        )
    if sf.n_points:
        pts = sf.all_points()
        lo, hi = pts.min(axis=0), pts.max(axis=0)
        out.append(
            "extent after the Y/X swap: "
            f"east {lo[0]:.3f}..{hi[0]:.3f}, north {lo[1]:.3f}..{hi[1]:.3f}, "
            f"up {lo[2]:.3f}..{hi[2]:.3f}"
        )
        n_desc = max((len(d) for s in sf.segments for d in s.descriptions), default=0)
        out.append(f"description fields per point: up to {n_desc}")
        out.append(f"first {limit} parsed points (string, east, north, up, descriptions):")
        shown = 0
        for s in sf.segments:
            for p, d in zip(s.points.tolist(), s.descriptions, strict=True):
                if shown == limit:
                    break
                out.append(f"  {s.number}, {p[0]!r}, {p[1]!r}, {p[2]!r}, {list(d)}")
                shown += 1
    _describe_notes(sf.warnings, sf.skipped, out, limit)


def _describe_dtm(path: Path, text: str, out: list[str], limit: int) -> None:
    out.append("kind: Surpac DTM file (.dtm), text  [layout of the 1996 format appendix]")
    raw = parse_dtm_topology(text)
    out.append(f"header: {raw.header!r}")
    out.append(f"lines between header and first block: {len(raw.preamble)}")
    out.extend(f"  {ln!r}" for ln in raw.preamble[:limit])
    objects = sorted({t.object_id for t in raw.trisolations})
    out.append(f"objects: {len(objects)} {objects[:20]}; trisolations: {len(raw.trisolations)}")
    max_vertex = 0
    for t in raw.trisolations:
        triangle_ids, vertex_ids, neighbour_ids = t.arrays()
        out.append(
            f"  object {t.object_id} trisolation {t.trisolation_id} (line {t.line}): "
            f"{len(triangle_ids)} triangle(s), flags as written {t.flags}"
        )
        if len(triangle_ids):
            max_vertex = max(max_vertex, int(vertex_ids.max()))
            out.append(
                f"    triangle ids {int(triangle_ids.min())}..{int(triangle_ids.max())}, "
                f"vertex indices {int(vertex_ids.min())}..{int(vertex_ids.max())}, "
                f"neighbour ids {int(neighbour_ids.min())}..{int(neighbour_ids.max())} "
                f"({int((neighbour_ids != NO_NEIGHBOUR).sum())} non-zero)"
            )
            out.extend(f"    {r}" for r in t.records[:limit])
    out.append(f"END record found: {raw.ended}")
    _describe_notes(raw.warnings, raw.skipped, out, limit)

    str_path = _paired_path(path, ".str")
    if not str_path.exists():
        out.append(f"paired string file: {str_path.name} NOT FOUND")
        return
    data = str_path.read_bytes()
    if looks_binary(data):
        out.append(f"paired string file: {str_path.name} is BINARY (not supported)")
        return
    try:
        sf = surpac_str.read_surpac_str(str_path)
    except (ValueError, OSError) as exc:
        out.append(f"paired string file: {str_path.name} could not be read: {exc}")
        return
    n_records = len(sf.record_rows)
    out.append(
        f"paired string file: {str_path.name}, {sf.n_points} point record(s) in "
        f"{sf.n_segments} segment(s), {n_records} record(s) with the null records; "
        f"largest vertex number in the .dtm is {max_vertex}"
    )
    record_rows = np.asarray(sf.record_rows, dtype=np.int64)
    for t in raw.trisolations:
        _, vertex_ids, _ = t.arrays()
        if not len(vertex_ids):
            continue
        by_record = int((_point_rows(vertex_ids, record_rows) < 0).sum())
        by_point = int((_point_rows(vertex_ids) >= sf.n_points).sum())
        out.append(
            f"  object {t.object_id} trisolation {t.trisolation_id}: counting all records "
            f"(published rule) {by_record} vertex number(s) name no point; counting point "
            f"records only {by_point}"
        )
