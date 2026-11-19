"""Stanford PLY, ASCII variant: read and write triangle meshes.

Reading takes the ``x``, ``y``, ``z`` properties of the ``vertex`` element and
the first list property of the ``face`` element (usually ``vertex_indices`` or
``vertex_index``); other properties and elements are skipped. Faces with more
than three corners become a triangle fan. Binary PLY is rejected with a clear
message.

Coordinates are assumed ``x = east, y = north, z = up``; ``swap_xy`` exchanges
the first two columns (a relabelling: winding is not changed).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io._text import BinaryFileError, parse_float
from bakumine.core.io.obj import fan_triangles, format_vertex_lines

__all__ = ["PlyFormatError", "format_ply", "parse_ply", "read_ply", "write_ply"]


class PlyFormatError(ValueError):
    """The file is not a readable ASCII PLY."""


def parse_ply(text: str, *, swap_xy: bool = False, source: str = "<string>") -> Mesh:
    """Parse ASCII PLY text into a :class:`Mesh`. See :func:`read_ply`."""
    lines = text.splitlines()
    if not lines or lines[0].strip() != "ply":
        raise PlyFormatError(f"{source}: not a PLY file (first line is not 'ply')")
    elements: list[tuple[str, int, list[list[str]]]] = []
    body_start = None
    for k, line in enumerate(lines[1:], start=1):
        tokens = line.split()
        if not tokens or tokens[0] in ("comment", "obj_info"):
            continue
        if tokens[0] == "format":
            if len(tokens) < 2 or tokens[1] != "ascii":
                raise BinaryFileError(
                    f"{source}: binary PLY ({' '.join(tokens[1:])}) is not supported; "
                    "re-save as ASCII PLY"
                )
        elif tokens[0] == "element":
            try:
                elements.append((tokens[1], int(tokens[2]), []))
            except (IndexError, ValueError):
                raise PlyFormatError(f"{source}: line {k + 1}: bad element record") from None
        elif tokens[0] == "property":
            if not elements:
                raise PlyFormatError(f"{source}: line {k + 1}: property before any element")
            elements[-1][2].append(tokens[1:])
        elif tokens[0] == "end_header":
            body_start = k + 1
            break
        else:
            raise PlyFormatError(f"{source}: line {k + 1}: unknown header record {tokens[0]!r}")
    if body_start is None:
        raise PlyFormatError(f"{source}: no end_header record")

    body = [ln for ln in lines[body_start:] if ln.strip()]
    vertices = np.zeros((0, 3))
    faces: list[tuple[int, int, int]] = []
    pos = 0
    for name, count, properties in elements:
        rows = body[pos : pos + count]
        if len(rows) < count:
            raise PlyFormatError(f"{source}: element {name!r} has {len(rows)} of {count} rows")
        pos += count
        try:
            if name == "vertex":
                if any(p[0] == "list" for p in properties):
                    raise PlyFormatError(f"{source}: list property in the vertex element")
                names = [p[-1] for p in properties]
                columns = [names.index(axis) for axis in ("x", "y", "z")]
                table = [row.split() for row in rows]
                vertices = np.array(
                    [[parse_float(t[c]) for c in columns] for t in table], dtype=np.float64
                ).reshape(-1, 3)
            elif name == "face":
                skip = 0
                for p in properties:
                    if p[0] == "list":
                        break
                    skip += 1
                else:
                    raise PlyFormatError(f"{source}: the face element has no list property")
                for row in rows:
                    tokens = row.split()
                    n = int(tokens[skip])
                    corners = [int(t) for t in tokens[skip + 1 : skip + 1 + n]]
                    if len(corners) != n:
                        raise ValueError("face record is shorter than its count")
                    faces.extend(fan_triangles(corners))
        except PlyFormatError:
            raise
        except (ValueError, IndexError) as exc:
            raise PlyFormatError(f"{source}: element {name!r}: {exc}") from None
    if swap_xy:
        vertices = vertices[:, [1, 0, 2]]
    try:
        return Mesh(vertices, np.asarray(faces, dtype=np.int64).reshape(-1, 3))
    except ValueError as exc:
        raise PlyFormatError(f"{source}: {exc}") from None


def read_ply(path, *, swap_xy: bool = False) -> Mesh:
    """Read an ASCII PLY file as a mesh (vertex and face order kept)."""
    data = Path(path).read_bytes()
    head = data[:2048].split(b"end_header", 1)[0].decode("latin-1")
    if data[:3] == b"ply" and "format binary" in head:
        raise BinaryFileError(f"{path}: binary PLY is not supported; re-save as ASCII PLY")
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    return parse_ply(text, swap_xy=swap_xy, source=os.fspath(path))


def format_ply(
    mesh: Mesh, *, decimals: int | None = 6, swap_xy: bool = False, comment: str | None = None
) -> str:
    """ASCII PLY text of a mesh. See :func:`write_ply`."""
    lines = ["ply", "format ascii 1.0"]
    lines.extend(f"comment {c}" for c in (comment.splitlines() if comment else []))
    lines += [
        f"element vertex {mesh.n_vertices}",
        "property double x",
        "property double y",
        "property double z",
        f"element face {mesh.n_faces}",
        "property list uchar int vertex_indices",
        "end_header",
    ]
    v = mesh.vertices[:, [1, 0, 2]] if swap_xy else mesh.vertices
    lines.extend(format_vertex_lines(v, decimals, ""))
    lines.extend(f"3 {a} {b} {c}" for a, b, c in mesh.faces.tolist())
    return "\n".join(lines) + "\n"


def write_ply(
    path,
    mesh: Mesh,
    *,
    decimals: int | None = 6,
    swap_xy: bool = False,
    comment: str | None = None,
) -> None:
    """Write a mesh as ASCII PLY (double coordinates, 0-based triangle indices).

    ``decimals`` fixes the number of decimals (default 6); None writes the
    shortest text that reads back to the identical float64.
    """
    text = format_ply(mesh, decimals=decimals, swap_xy=swap_xy, comment=comment)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
