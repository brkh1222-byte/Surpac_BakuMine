"""Wavefront OBJ: read and write triangle meshes.

Only geometry is handled: ``v`` (vertex) and ``f`` (face) records. Texture
coordinates, normals, materials, groups, objects and smoothing groups are
ignored; all groups of a file end up in one mesh. Faces with more than three
corners are split into a triangle fan from their first corner. Negative
(relative) vertex indices are resolved.

Coordinates are assumed ``x = east, y = north, z = up`` as in the internal
model; ``swap_xy`` exchanges the first two columns for files that hold the
northing first. That is a relabelling of columns, so winding is not changed.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io._text import parse_float, read_text

__all__ = ["ObjFormatError", "format_obj", "parse_obj", "read_obj", "write_obj"]


class ObjFormatError(ValueError):
    """A ``v`` or ``f`` record cannot be understood."""


def fan_triangles(corners: list[int]) -> list[tuple[int, int, int]]:
    """Triangle fan ``(c0, ci, ci+1)`` of a polygon given by vertex indices."""
    return [(corners[0], corners[i], corners[i + 1]) for i in range(1, len(corners) - 1)]


def format_vertex_lines(vertices: np.ndarray, decimals: int | None, prefix: str) -> list[str]:
    """One text line per vertex; ``decimals=None`` gives round-trip exact text."""
    if decimals is None:
        return [f"{prefix}{x!r} {y!r} {z!r}" for x, y, z in vertices.tolist()]
    d = decimals
    return [f"{prefix}{x:.{d}f} {y:.{d}f} {z:.{d}f}" for x, y, z in vertices.tolist()]


def parse_obj(text: str, *, swap_xy: bool = False, source: str = "<string>") -> Mesh:
    """Parse OBJ text into one :class:`Mesh`. See :func:`read_obj`."""
    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    pending = ""
    for number, line in enumerate(text.splitlines(), start=1):
        line = pending + line.split("#", 1)[0].strip()
        if line.endswith("\\"):  # continuation line
            pending = line[:-1] + " "
            continue
        pending = ""
        tokens = line.split()
        if not tokens:
            continue
        kind = tokens[0]
        try:
            if kind == "v":
                vertices.append(
                    (parse_float(tokens[1]), parse_float(tokens[2]), parse_float(tokens[3]))
                )
            elif kind == "f":
                corners = []
                for token in tokens[1:]:
                    index = int(token.split("/", 1)[0])
                    if index > 0:
                        index -= 1
                    elif index < 0:
                        index += len(vertices)
                    else:
                        raise ValueError("vertex index 0")
                    if not 0 <= index < len(vertices):
                        raise ValueError(f"vertex index {token} is not defined yet")
                    corners.append(index)
                faces.extend(fan_triangles(corners))
        except (ValueError, IndexError) as exc:
            raise ObjFormatError(f"{source}: line {number}: {exc}: {line!r}") from None
    v = np.asarray(vertices, dtype=np.float64).reshape(-1, 3)
    if swap_xy:
        v = v[:, [1, 0, 2]]
    return Mesh(v, np.asarray(faces, dtype=np.int64).reshape(-1, 3))


def read_obj(path, *, swap_xy: bool = False) -> Mesh:
    """Read an OBJ file as one mesh.

    Vertices keep their file order (also those no face uses), faces keep
    theirs; n-gons become fans. Raises :class:`ObjFormatError` on a bad ``v``
    or ``f`` record; every other record type is ignored.
    """
    text = read_text(path, format_name="OBJ")
    return parse_obj(text, swap_xy=swap_xy, source=os.fspath(path))


def format_obj(
    mesh: Mesh, *, decimals: int | None = 6, swap_xy: bool = False, comment: str | None = None
) -> str:
    """OBJ text of a mesh. See :func:`write_obj`."""
    lines = [f"# {c}" for c in (comment.splitlines() if comment else [])]
    v = mesh.vertices[:, [1, 0, 2]] if swap_xy else mesh.vertices
    lines.extend(format_vertex_lines(v, decimals, "v "))
    lines.extend(f"f {a} {b} {c}" for a, b, c in (mesh.faces + 1).tolist())
    return "\n".join(lines) + "\n"


def write_obj(
    path,
    mesh: Mesh,
    *,
    decimals: int | None = 6,
    swap_xy: bool = False,
    comment: str | None = None,
) -> None:
    """Write a mesh as OBJ (``v`` and ``f`` records, 1-based indices).

    ``decimals`` fixes the number of decimals (default 6 = micrometres);
    None writes the shortest text that reads back to the identical float64.
    ``comment`` lines are written at the top, each prefixed with ``#``.
    """
    text = format_obj(mesh, decimals=decimals, swap_xy=swap_xy, comment=comment)
    Path(path).write_text(text, encoding="utf-8", newline="\n")
