"""ASCII DXF: read and write strings, meshes and points, grouped by layer.

Written by hand from the public DXF reference (group-code / value pairs); no
DXF library is used.

Reading understands these entities of the ``ENTITIES`` section:

* ``3DFACE`` — a triangle, or a quad that is split into two triangles
  ``(0, 1, 2)`` and ``(0, 2, 3)``;
* ``POLYLINE`` + ``VERTEX`` ... ``SEQEND`` — 3D polylines, 2D polylines (with
  the polyline's elevation), polyface meshes (flag 64) and polygon M x N
  meshes (flag 16);
* ``LWPOLYLINE`` — with its elevation (group 38);
* ``LINE`` — an open string of two points;
* ``POINT``.

Everything else (other entities, other sections, block definitions) is
ignored and counted in ``DxfDocument.skipped_entities`` /
``skipped_sections``. Not handled, and noted in ``DxfDocument.notes`` when
met: bulges (arcs inside polylines are read as straight chords) and extrusion
directions other than +Z (object coordinate systems). Binary DXF is rejected.

DXF has no string numbers and no point descriptions. On reading, a string on
a layer whose name is a positive integer gets that number, otherwise 1; on
writing, a :class:`StringFile` goes to layers named after the string numbers.
Descriptions are not written.

Coordinates are assumed to be ``X = east, Y = north, Z = up``, as in the
internal model. ``swap_xy=True`` is for files whose X column holds the
northing: the two columns are exchanged on the way in or out. That is a
relabelling of columns, not a mirror of the geometry, so triangle winding is
left as it is.
"""

from __future__ import annotations

import os
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io._text import BinaryFileError, parse_float, read_text
from bakumine.core.io.strings import StringFile, StringSegment

__all__ = ["DxfDocument", "DxfFormatError", "parse_dxf", "read_dxf", "write_dxf", "format_dxf"]

DEFAULT_LAYER = "0"
#: Largest vertex / face count of one polyface mesh (16-bit indices in DXF).
POLYFACE_LIMIT = 32767

_POLYLINE_CLOSED = 1
_POLYLINE_3D = 8
_POLYLINE_MESH = 16
_POLYLINE_MESH_CLOSED_N = 32
_POLYLINE_POLYFACE = 64
_VERTEX_3D_POLYLINE = 32
_VERTEX_POLYFACE_COORD = 64
_VERTEX_POLYFACE = 128


class DxfFormatError(ValueError):
    """The file is not an ASCII DXF, or it is damaged beyond tolerance."""


@dataclass
class DxfDocument:
    """Geometry of a DXF file by layer, in internal coordinates.

    Attributes
    ----------
    strings : dict layer -> list of StringSegment
        Polylines and lines. A closed polyline gets its first point repeated
        at the end, which is how the string model marks a closed segment.
    meshes : dict layer -> Mesh
        All 3DFACEs, polyface meshes and polygon meshes of the layer joined
        into one mesh; identical vertices are merged (see ``weld_tolerance``).
    points : dict layer -> (n, 3) array
        POINT entities.
    skipped_entities, skipped_sections : Counter
        Entity types and section names that were ignored, with counts.
    notes : Counter
        Things that were read approximately (bulges, extrusions, bad records).
    """

    strings: dict[str, list[StringSegment]] = field(default_factory=dict)
    meshes: dict[str, Mesh] = field(default_factory=dict)
    points: dict[str, np.ndarray] = field(default_factory=dict)
    skipped_entities: Counter = field(default_factory=Counter)
    skipped_sections: Counter = field(default_factory=Counter)
    notes: Counter = field(default_factory=Counter)

    @property
    def layers(self) -> list[str]:
        """Names of all layers that hold geometry, in first-seen order."""
        seen: dict[str, None] = {}
        for group in (self.strings, self.meshes, self.points):
            for name in group:
                seen.setdefault(name)
        return list(seen)

    @property
    def n_skipped(self) -> int:
        return sum(self.skipped_entities.values())

    def string_file(self) -> StringFile:
        """All strings of all layers as one :class:`StringFile` (layer order)."""
        return StringFile(segments=[s for segs in self.strings.values() for s in segs])

    def mesh(self) -> Mesh:
        """All meshes of all layers joined into one (no welding across layers)."""
        return Mesh.concatenate(list(self.meshes.values()))


# --------------------------------------------------------------------- read


def _pairs(text: str) -> list[tuple[int, str]]:
    """Group-code / value pairs of an ASCII DXF."""
    lines = text.splitlines()
    while lines and not lines[-1].strip():
        lines.pop()
    pairs = []
    for k in range(0, len(lines) - 1, 2):
        code_text = lines[k].strip()
        try:
            code = int(code_text)
        except ValueError:
            raise DxfFormatError(
                f"line {k + 1}: expected a DXF group code, found {code_text[:40]!r}"
            ) from None
        pairs.append((code, lines[k + 1].strip()))
    return pairs


def _records(pairs: list[tuple[int, str]]) -> list[tuple[str, list[tuple[int, str]]]]:
    """Split pairs at every group code 0: ``(name, [(code, value), ...])``."""
    records: list[tuple[str, list[tuple[int, str]]]] = []
    for code, value in pairs:
        if code == 0:
            records.append((value.upper(), []))
        elif records:
            records[-1][1].append((code, value))
    return records


class _Fields:
    """Group codes of one entity. First value wins, except in ``all``."""

    def __init__(self, items: list[tuple[int, str]]):
        self.items = items
        self.first: dict[int, str] = {}
        for code, value in items:
            self.first.setdefault(code, value)

    def has(self, code: int) -> bool:
        return code in self.first

    def num(self, code: int, default: float = 0.0) -> float:
        try:
            return parse_float(self.first[code])
        except (KeyError, ValueError):
            return default

    def integer(self, code: int, default: int = 0) -> int:
        try:
            return int(parse_float(self.first[code]))
        except (KeyError, ValueError):
            return default

    def text(self, code: int, default: str = "") -> str:
        return self.first.get(code, default)

    def point(self, index: int = 0) -> tuple[float, float, float]:
        return self.num(10 + index), self.num(20 + index), self.num(30 + index)

    @property
    def layer(self) -> str:
        return self.text(8, DEFAULT_LAYER) or DEFAULT_LAYER

    @property
    def tilted(self) -> bool:
        """Extrusion direction given and not +Z."""
        return (
            abs(self.num(210, 0.0)) > 1e-12
            or abs(self.num(220, 0.0)) > 1e-12
            or abs(self.num(230, 1.0) - 1.0) > 1e-12
        )


class _Collector:
    def __init__(self) -> None:
        self.doc = DxfDocument()
        self.strings: dict[str, list[np.ndarray]] = {}
        self.tri: dict[str, list[np.ndarray]] = {}
        self.points: dict[str, list[tuple[float, float, float]]] = {}

    def add_string(self, layer: str, pts, closed: bool) -> None:
        p = np.asarray(pts, dtype=np.float64).reshape(-1, 3)
        if len(p) == 0:
            self.doc.notes["polyline without vertices"] += 1
            return
        if closed and len(p) >= 2 and not np.array_equal(p[0], p[-1]):
            p = np.vstack([p, p[:1]])
        self.strings.setdefault(layer, []).append(p)

    def add_faces(self, layer: str, corners: np.ndarray, faces: np.ndarray) -> None:
        """Faces (k, 3) indexing ``corners`` (n, 3): stored as triangle soup."""
        if len(faces):
            self.tri.setdefault(layer, []).append(corners[faces])

    def add_polygon(self, layer: str, corners) -> None:
        """A 3- or 4-corner face; the fourth corner may repeat the third."""
        c = np.asarray(corners, dtype=np.float64)
        if len(c) == 4 and np.array_equal(c[3], c[2]):
            c = c[:3]
        if len(c) == 3:
            self.add_faces(layer, c, np.array([[0, 1, 2]]))
        else:
            self.add_faces(layer, c, np.array([[0, 1, 2], [0, 2, 3]]))


def _read_3dface(fields: _Fields, col: _Collector) -> None:
    corners = [fields.point(k) for k in range(3)]
    if fields.has(13) or fields.has(23) or fields.has(33):
        corners.append(fields.point(3))
    col.add_polygon(fields.layer, corners)


def _read_line(fields: _Fields, col: _Collector) -> None:
    col.add_string(fields.layer, [fields.point(0), fields.point(1)], closed=False)


def _read_lwpolyline(fields: _Fields, col: _Collector) -> None:
    elevation = fields.num(38, 0.0)
    pts: list[list[float]] = []
    for code, value in fields.items:
        try:
            if code == 10:
                pts.append([parse_float(value), 0.0, elevation])
            elif code == 20 and pts:
                pts[-1][1] = parse_float(value)
            elif code == 42 and parse_float(value) != 0.0:
                col.doc.notes["bulge ignored (arc read as chord)"] += 1
        except ValueError:
            col.doc.notes["unreadable number"] += 1
    if fields.tilted:
        col.doc.notes["extrusion direction ignored"] += 1
    col.add_string(fields.layer, pts, closed=bool(fields.integer(70) & _POLYLINE_CLOSED))


def _read_polyline(fields: _Fields, vertices: list[_Fields], col: _Collector) -> None:
    flags = fields.integer(70)
    layer = fields.layer
    if flags & _POLYLINE_POLYFACE:
        coords, faces = [], []
        for v in vertices:
            vflags = v.integer(70)
            if vflags & _VERTEX_POLYFACE and not vflags & _VERTEX_POLYFACE_COORD:
                ids = [abs(v.integer(code)) for code in (71, 72, 73, 74)]  # negative: hidden edge
                ids = [i for i in ids if i != 0]
                if len(ids) >= 3:
                    faces.append(ids)
            else:
                coords.append(v.point())
        corners = np.asarray(coords, dtype=np.float64).reshape(-1, 3)
        tris = []
        for ids in faces:
            if max(ids) > len(corners):
                col.doc.notes["polyface record with an index out of range"] += 1
                continue
            a, b, c = ids[0] - 1, ids[1] - 1, ids[2] - 1
            tris.append((a, b, c))
            if len(ids) == 4 and ids[3] != ids[2]:
                tris.append((a, c, ids[3] - 1))
        col.add_faces(layer, corners, np.asarray(tris, dtype=np.int64).reshape(-1, 3))
        return
    if flags & _POLYLINE_MESH:
        m, n = fields.integer(71), fields.integer(72)
        corners = np.asarray([v.point() for v in vertices], dtype=np.float64).reshape(-1, 3)
        if m < 2 or n < 2 or len(corners) != m * n:
            col.doc.notes["polygon mesh with inconsistent M x N"] += 1
            return
        mm = m if flags & _POLYLINE_CLOSED else m - 1
        nn = n if flags & _POLYLINE_MESH_CLOSED_N else n - 1
        i, j = np.meshgrid(np.arange(mm), np.arange(nn), indexing="ij")
        i, j = i.ravel(), j.ravel()
        a = i * n + j
        b = ((i + 1) % m) * n + j
        c = ((i + 1) % m) * n + (j + 1) % n
        d = i * n + (j + 1) % n
        col.add_faces(layer, corners, np.concatenate([np.c_[a, b, c], np.c_[a, c, d]]))
        return
    is_3d = bool(flags & _POLYLINE_3D)
    elevation = fields.num(30, 0.0)
    pts = []
    for v in vertices:
        x, y, z = v.point()
        if not is_3d and not v.has(30):
            z = elevation
        if v.num(42, 0.0) != 0.0:
            col.doc.notes["bulge ignored (arc read as chord)"] += 1
        pts.append((x, y, z))
    if not is_3d and fields.tilted:
        col.doc.notes["extrusion direction ignored"] += 1
    col.add_string(layer, pts, closed=bool(flags & _POLYLINE_CLOSED))


def _string_number(layer: str) -> int:
    return int(layer) if layer.isdigit() and int(layer) > 0 else 1


def parse_dxf(
    text: str, *, swap_xy: bool = False, weld_tolerance: float = 0.0, source: str = "<string>"
) -> DxfDocument:
    """Parse the text of an ASCII DXF file. See :func:`read_dxf`."""
    try:
        records = _records(_pairs(text))
    except DxfFormatError as exc:
        raise DxfFormatError(f"{source}: {exc}") from None
    if not records:
        raise DxfFormatError(f"{source}: no DXF records found")
    col = _Collector()
    doc = col.doc

    has_sections = any(name == "SECTION" for name, _ in records)
    section = None if has_sections else "ENTITIES"
    k = 0
    while k < len(records):
        name, items = records[k]
        k += 1
        if name == "EOF":
            break
        if name == "SECTION":
            section = dict(items).get(2, "").upper()
            if section != "ENTITIES":
                doc.skipped_sections[section or "?"] += 1
            continue
        if name == "ENDSEC":
            section = None
            continue
        if section != "ENTITIES":
            continue
        fields = _Fields(items)
        if name == "3DFACE":
            _read_3dface(fields, col)
        elif name == "LINE":
            _read_line(fields, col)
        elif name == "POINT":
            col.points.setdefault(fields.layer, []).append(fields.point())
        elif name == "LWPOLYLINE":
            _read_lwpolyline(fields, col)
        elif name == "POLYLINE":
            vertices = []
            while k < len(records) and records[k][0] == "VERTEX":
                vertices.append(_Fields(records[k][1]))
                k += 1
            if k < len(records) and records[k][0] == "SEQEND":
                k += 1
            else:
                doc.notes["POLYLINE without SEQEND"] += 1
            _read_polyline(fields, vertices, col)
        else:
            doc.skipped_entities[name] += 1

    def orient(a: np.ndarray) -> np.ndarray:
        return a[..., [1, 0, 2]] if swap_xy else a

    for layer, arrays in col.strings.items():
        number = _string_number(layer)
        doc.strings[layer] = [StringSegment(number, orient(a)) for a in arrays]
    for layer, soups in col.tri.items():
        soup = orient(np.concatenate(soups).reshape(-1, 3))
        mesh = Mesh(soup, np.arange(len(soup)).reshape(-1, 3))
        doc.meshes[layer] = mesh.weld(weld_tolerance, remove_collapsed=False)
    for layer, pts in col.points.items():
        doc.points[layer] = orient(np.asarray(pts, dtype=np.float64).reshape(-1, 3))
    return doc


def read_dxf(path, *, swap_xy: bool = False, weld_tolerance: float = 0.0) -> DxfDocument:
    """Read an ASCII DXF file.

    Parameters
    ----------
    swap_xy : bool
        The file stores the northing in X and the easting in Y; exchange them.
    weld_tolerance : float
        Distance (metres) below which face corners become one mesh vertex.
        The default 0 merges only identical coordinates, which is what
        3DFACE triangle soup needs to become an indexed mesh again.

    Returns a :class:`DxfDocument`. Unknown entities and sections are counted
    there, not reported as errors. Raises :class:`BinaryFileError` for a
    binary DXF and :class:`DxfFormatError` if the text is not group-code pairs.
    """
    data = Path(path).read_bytes()
    if data.startswith(b"AutoCAD Binary DXF"):
        raise BinaryFileError(f"{path}: binary DXF is not supported; re-save as ASCII DXF")
    text = read_text(path, format_name="DXF")
    return parse_dxf(text, swap_xy=swap_xy, weld_tolerance=weld_tolerance, source=os.fspath(path))


# -------------------------------------------------------------------- write


def _number(value: float, decimals: int | None) -> str:
    if decimals is None:
        return repr(float(value))
    text = f"{value:.{decimals}f}"
    return text.lstrip("-") if float(text) == 0.0 else text


def _layer_name(name) -> str:
    text = str(name)
    if not text or any(ch in text for ch in "\r\n"):
        raise DxfFormatError(f"invalid layer name {text!r}")
    return text


def _strings_by_layer(strings) -> dict[str, list[StringSegment]]:
    if strings is None:
        return {}
    if isinstance(strings, Mapping):
        out = {}
        for layer, value in strings.items():
            segments = value.segments if isinstance(value, StringFile) else value
            if isinstance(segments, StringSegment):
                segments = [segments]
            out[_layer_name(layer)] = list(segments)
        return out
    segments = strings.segments if isinstance(strings, StringFile) else list(strings)
    out = {}
    for s in segments:
        out.setdefault(str(s.number), []).append(s)
    return out


def _meshes_by_layer(meshes) -> dict[str, Mesh]:
    if meshes is None:
        return {}
    if isinstance(meshes, Mesh):
        return {DEFAULT_LAYER: meshes}
    if isinstance(meshes, Mapping):
        return {_layer_name(k): v for k, v in meshes.items()}
    return {f"MESH_{k + 1}": m for k, m in enumerate(meshes)}


def format_dxf(
    *,
    strings=None,
    meshes=None,
    points: Mapping[str, Sequence] | None = None,
    polyface: bool = False,
    decimals: int | None = None,
    swap_xy: bool = False,
) -> str:
    """Text of an ASCII DXF (R12-style entities). See :func:`write_dxf`."""
    string_layers = _strings_by_layer(strings)
    mesh_layers = _meshes_by_layer(meshes)
    point_layers = {
        _layer_name(k): np.asarray(v, dtype=np.float64).reshape(-1, 3)
        for k, v in (points or {}).items()
    }
    layers: dict[str, None] = {DEFAULT_LAYER: None}
    for group in (string_layers, mesh_layers, point_layers):
        for name in group:
            layers.setdefault(name)

    out: list[str] = []

    def emit(code: int, value) -> None:
        out.append(str(code))
        out.append(str(value))

    def emit_point(xyz, index: int = 0) -> None:
        x, y, z = xyz
        if swap_xy:
            x, y = y, x
        emit(10 + index, _number(x, decimals))
        emit(20 + index, _number(y, decimals))
        emit(30 + index, _number(z, decimals))

    emit(0, "SECTION")
    emit(2, "TABLES")
    emit(0, "TABLE")
    emit(2, "LAYER")
    emit(70, len(layers))
    for name in layers:
        emit(0, "LAYER")
        emit(2, name)
        emit(70, 0)
        emit(62, 7)
        emit(6, "CONTINUOUS")
    emit(0, "ENDTAB")
    emit(0, "ENDSEC")

    emit(0, "SECTION")
    emit(2, "ENTITIES")
    for layer, segments in string_layers.items():
        for segment in segments:
            pts = segment.points
            if len(pts) == 0:
                continue
            if not np.isfinite(pts).all():
                raise DxfFormatError(f"layer {layer!r}: string has NaN or infinite coordinates")
            closed = segment.is_closed
            if closed:
                pts = pts[:-1]
            emit(0, "POLYLINE")
            emit(8, layer)
            emit(66, 1)
            emit_point((0.0, 0.0, 0.0))
            emit(70, _POLYLINE_3D | (_POLYLINE_CLOSED if closed else 0))
            for p in pts.tolist():
                emit(0, "VERTEX")
                emit(8, layer)
                emit_point(p)
                emit(70, _VERTEX_3D_POLYLINE)
            emit(0, "SEQEND")
            emit(8, layer)
    for layer, mesh in mesh_layers.items():
        if polyface:
            if mesh.n_vertices > POLYFACE_LIMIT or mesh.n_faces > POLYFACE_LIMIT:
                raise DxfFormatError(
                    f"layer {layer!r}: a polyface mesh holds at most {POLYFACE_LIMIT} vertices "
                    "and faces; write 3DFACEs instead (polyface=False)"
                )
            emit(0, "POLYLINE")
            emit(8, layer)
            emit(66, 1)
            emit_point((0.0, 0.0, 0.0))
            emit(70, _POLYLINE_POLYFACE)
            emit(71, mesh.n_vertices)
            emit(72, mesh.n_faces)
            for p in mesh.vertices.tolist():
                emit(0, "VERTEX")
                emit(8, layer)
                emit_point(p)
                emit(70, _VERTEX_POLYFACE | _VERTEX_POLYFACE_COORD)
            for a, b, c in (mesh.faces + 1).tolist():
                emit(0, "VERTEX")
                emit(8, layer)
                emit_point((0.0, 0.0, 0.0))
                emit(70, _VERTEX_POLYFACE)
                emit(71, a)
                emit(72, b)
                emit(73, c)
            emit(0, "SEQEND")
            emit(8, layer)
        else:
            for tri in mesh.triangles.tolist():
                emit(0, "3DFACE")
                emit(8, layer)
                emit_point(tri[0], 0)
                emit_point(tri[1], 1)
                emit_point(tri[2], 2)
                emit_point(tri[2], 3)  # fourth corner repeats the third: a triangle
    for layer, pts in point_layers.items():
        for p in pts.tolist():
            emit(0, "POINT")
            emit(8, layer)
            emit_point(p)
    emit(0, "ENDSEC")
    emit(0, "EOF")
    return "\n".join(out) + "\n"


def write_dxf(
    path,
    *,
    strings=None,
    meshes=None,
    points: Mapping[str, Sequence] | None = None,
    polyface: bool = False,
    decimals: int | None = None,
    swap_xy: bool = False,
) -> None:
    """Write strings, meshes and points to an ASCII DXF file.

    Parameters
    ----------
    strings : dict layer -> segments / StringFile, or a StringFile, or segments
        Written as 3D ``POLYLINE`` entities. Without layer names (a
        :class:`StringFile` or a list of segments) each string number goes to
        the layer named after it. A closed segment is written without its
        repeated last point and with the closed flag.
    meshes : dict layer -> Mesh, or a Mesh (layer "0"), or a list of meshes
        Written as ``3DFACE`` entities, or as one polyface mesh per layer when
        ``polyface`` is True (at most 32767 vertices and faces each). A list
        goes to layers ``MESH_1``, ``MESH_2``, ...
    points : dict layer -> (n, 3) array
        Written as ``POINT`` entities.
    decimals : int, optional
        Fixed decimals. Default None writes the shortest text that reads back
        to the identical float64, so a round trip is lossless.
    swap_xy : bool
        Write the northing into X and the easting into Y.

    Layer names are written as given. A DXF read back with :func:`read_dxf`
    gives the same layers, strings and (after index remapping) faces.
    """
    text = format_dxf(
        strings=strings,
        meshes=meshes,
        points=points,
        polyface=polyface,
        decimals=decimals,
        swap_xy=swap_xy,
    )
    Path(path).write_text(text, encoding="utf-8", newline="\n")
