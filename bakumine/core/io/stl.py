"""STL (stereolithography), ASCII and binary: read triangle meshes.

An STL file is a list of triangles, each with its own three corner coordinates
(and a normal, which is ignored: the corner order defines the winding). There is
no shared vertex table, so corners with bit-identical coordinates are merged into
one vertex on reading; the first occurrence keeps its place, and the triangle
order and winding of the file are kept. Triangles that repeat a corner are kept
too (``Mesh.weld`` / ``import_mesh`` drop them).

Binary STL: an 80-byte header, a little-endian uint32 triangle count and 50
bytes per triangle (normal and three corners as float32, a 2-byte attribute).
The coordinates are therefore single precision: at coordinates of a few million
the resolution is about 0.25 of the file's unit. A file is taken as binary when
its length is exactly ``84 + 50 n`` for the stored count ``n``, whatever its
first bytes are (binary files starting with ``solid`` exist); otherwise it is
parsed as ASCII (``facet`` / ``outer loop`` / ``vertex x y z``; several
``solid`` blocks are joined).

STL carries no unit. ``scale`` multiplies the coordinates (0.3048 for a file in
feet); ``swap_xy`` exchanges the first two columns (a relabelling: winding is
not changed). Writing is not implemented: single precision loses decimetres at
projected coordinates, use OBJ or PLY.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io._text import parse_float

__all__ = ["StlFormatError", "parse_stl", "read_stl"]

_HEADER = 84
_RECORD = np.dtype([("normal", "<f4", 3), ("corners", "<f4", (3, 3)), ("attribute", "<u2")])


class StlFormatError(ValueError):
    """The file is not a readable STL."""


def _binary_count(data: bytes) -> int | None:
    """Triangle count when ``data`` has the exact length of a binary STL, else None."""
    if len(data) < _HEADER:
        return None
    count = int.from_bytes(data[80:84], "little")
    return count if len(data) == _HEADER + _RECORD.itemsize * count else None


def _ascii_corners(data: bytes, source: str) -> np.ndarray:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = data.decode("latin-1")
    if text.lstrip()[:5].lower() != "solid":
        raise StlFormatError(
            f"{source}: not an STL file: it does not start with 'solid', and its length "
            f"({len(data)} bytes) is not that of a binary STL (84 + 50 per triangle)"
        )
    corners: list[tuple[float, float, float]] = []
    in_loop = False
    start = 0
    for k, line in enumerate(text.splitlines(), start=1):
        tokens = line.split()
        if not tokens:
            continue
        key = tokens[0].lower()
        if key == "outer":
            if in_loop:
                raise StlFormatError(f"{source}: line {k}: 'outer loop' inside a loop")
            in_loop, start = True, len(corners)
        elif key == "vertex":
            if not in_loop:
                raise StlFormatError(f"{source}: line {k}: vertex outside an 'outer loop'")
            try:
                x, y, z = (parse_float(t) for t in tokens[1:4])
            except ValueError:
                raise StlFormatError(f"{source}: line {k}: bad vertex record") from None
            corners.append((x, y, z))
        elif key == "endloop":
            if not in_loop or len(corners) - start != 3:
                raise StlFormatError(
                    f"{source}: line {k}: a facet must have exactly 3 vertices, "
                    f"got {len(corners) - start if in_loop else 0}"
                )
            in_loop = False
        elif key not in ("solid", "facet", "endfacet", "endsolid"):
            raise StlFormatError(f"{source}: line {k}: unknown STL record {tokens[0]!r}")
    if in_loop:
        raise StlFormatError(f"{source}: the file ends inside a facet (truncated?)")
    return np.asarray(corners, dtype=np.float64).reshape(-1, 3)


def parse_stl(
    data: bytes, *, scale: float = 1.0, swap_xy: bool = False, source: str = "<bytes>"
) -> Mesh:
    """Parse the bytes of an ASCII or binary STL file into a :class:`Mesh`."""
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    count = _binary_count(data)
    if count is not None:
        records = np.frombuffer(data, dtype=_RECORD, count=count, offset=_HEADER)
        corners = records["corners"].reshape(-1, 3).astype(np.float64)
    else:
        corners = _ascii_corners(data, source)
    if not np.isfinite(corners).all():
        raise StlFormatError(f"{source}: NaN or infinite coordinates")
    # merge bit-identical corners; vertices in order of first appearance
    unique, first, inverse = np.unique(corners, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    rank = np.empty(len(order), dtype=np.int64)
    rank[order] = np.arange(len(order))
    vertices = unique[order] * float(scale)
    if swap_xy:
        vertices = vertices[:, [1, 0, 2]]
    return Mesh(vertices, rank[np.asarray(inverse).reshape(-1)].reshape(-1, 3))


def read_stl(path, *, scale: float = 1.0, swap_xy: bool = False) -> Mesh:
    """Read an ASCII or binary STL file as a mesh with shared vertices."""
    return parse_stl(
        Path(path).read_bytes(), scale=scale, swap_xy=swap_xy, source=os.fspath(path)
    )
