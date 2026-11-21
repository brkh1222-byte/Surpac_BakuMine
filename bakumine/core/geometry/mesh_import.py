"""Bring an existing surface or solid file into the form the project stores.

Reads ``.dxf``, ``.obj``, ``.ply``, ``.stl`` or a Surpac text ``.dtm`` (with its ``.str``
next to it) through :mod:`bakumine.core.io.files`, welds vertices that coincide exactly (DXF
3DFACEs and some OBJ writers repeat them per triangle) and checks the mesh as the kind it
is imported as:

* ``dtm`` - must be a single-valued surface: not closed, no vertical triangle and, when
  the winding is consistent, no overturned triangle; triangles are turned normals-up;
* ``solid`` - validated (closed, consistently wound, no self-intersection); an invalid
  solid is still imported and the issues are reported, because a constraint needs to
  know and the user may want to look at it;
* ``auto`` - ``solid`` when the mesh is closed, else ``dtm``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from bakumine.core.geometry.mesh import Mesh

__all__ = ["KINDS", "import_mesh"]

KINDS = ("auto", "dtm", "solid")
#: coordinate rounding (m) of a binary STL above which the import says so
PRECISION_WARNING = 1e-3


def import_mesh(
    path: str | Path, kind: str = "auto", scale: float = 1.0
) -> tuple[Mesh, dict[str, Any]]:
    """The mesh of ``path`` ready to be stored, and a summary of it.

    The summary has ``kind`` (``dtm`` or ``solid``), ``n_vertices``, ``n_faces``,
    ``closed``, ``bounds`` and for a solid ``valid``, ``outward``, ``volume``, ``issues``;
    ``warnings`` (a list of sentences) when the file itself limits the result (a binary STL
    in large coordinates).

    ``scale`` multiplies every coordinate before anything is checked or measured: 0.3048
    for a file in feet (the project is in metres). It is in the summary when it is not 1.
    """
    import math

    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.io.files import read_mesh

    if kind not in KINDS:
        raise ValueError(f"kind must be one of {', '.join(KINDS)}")
    if isinstance(scale, bool) or not isinstance(scale, int | float):
        raise ValueError("scale must be a number")
    scale = float(scale)
    if not (math.isfinite(scale) and scale > 0.0):
        raise ValueError(f"scale must be a positive number, got {scale!r}")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"{path} not found")
    mesh = read_mesh(path)
    if mesh.n_faces == 0:
        raise ValueError(f"{path.name} holds no triangles")
    if scale != 1.0:
        mesh = Mesh(mesh.vertices * scale, mesh.faces, check=False)
    mesh = mesh.weld(0.0, remove_collapsed=True).remove_unreferenced_vertices()
    if mesh.n_faces == 0:
        raise ValueError(f"{path.name} holds only degenerate triangles")
    closed = bool(mesh.is_closed)
    if kind == "auto":
        kind = "solid" if closed else "dtm"
    info: dict[str, Any] = {"kind": kind, "closed": closed}
    if kind == "dtm":
        # a consistently wound surface with one Z per XY has all its triangles turned the
        # same way in plan; a mix means it folds over itself (or is a closed solid)
        a, b, c = (mesh.vertices[mesh.faces[:, j], :2] for j in range(3))
        ab, ac = b - a, c - a
        turn = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        if closed or (mesh.is_winding_consistent and (turn > 0).any() and (turn < 0).any()):
            raise ValueError(
                f"{path.name} is not a surface with one Z per XY: it "
                + ("is a closed solid" if closed else "folds over itself in plan")
            )
        try:
            mesh = DTM(mesh).mesh
        except ValueError as exc:
            raise ValueError(f"{path.name} is not a surface with one Z per XY: {exc}") from None
    else:
        report = mesh.validate()
        info.update(
            valid=bool(report.is_valid),
            outward=bool(report.is_outward),
            volume=float(mesh.volume) if closed else None,
            issues=report.issues(),
        )
    info.update(
        n_vertices=int(mesh.n_vertices),
        n_faces=int(mesh.n_faces),
        bounds=[[float(v) for v in row] for row in mesh.bounds],
    )
    if scale != 1.0:
        info["scale"] = scale
    if path.suffix.lower() == ".stl":
        from bakumine.core.io.stl import single_precision_step

        step = single_precision_step(path.read_bytes())
        if step is not None and step * scale > PRECISION_WARNING:
            info["warnings"] = [
                f"binary STL stores single-precision numbers: at these coordinates every "
                f"vertex is rounded to about {step * scale:.2g} m, which can turn a valid "
                "solid into one with degenerate or crossing triangles. Export it as ASCII "
                "STL, OBJ or DXF, or with a local origin"
            ]
    return mesh, info
