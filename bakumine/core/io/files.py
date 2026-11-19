"""Read and write strings and meshes by file extension.

==========  =======================  ================================================
extension   holds                    module
==========  =======================  ================================================
``.str``    strings                  :mod:`bakumine.core.io.surpac_str` (provisional)
``.dtm``    meshes (+ its ``.str``)  :mod:`bakumine.core.io.surpac_dtm` (provisional)
``.dxf``    strings and meshes       :mod:`bakumine.core.io.dxf`
``.obj``    one mesh                 :mod:`bakumine.core.io.obj`
``.ply``    one mesh                 :mod:`bakumine.core.io.ply`
==========  =======================  ================================================

Everything returned is in internal coordinates ``x = east, y = north,
z = up``. Keyword options are passed to the format's own function (for
example ``decimals=``, ``swap_xy=``); use those functions directly for
anything format-specific such as layers, object ids or flags.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io import dxf, obj, ply, surpac_dtm, surpac_str
from bakumine.core.io.strings import StringFile

__all__ = [
    "MESH_EXTENSIONS",
    "STRING_EXTENSIONS",
    "read_mesh",
    "read_meshes",
    "read_strings",
    "write_mesh",
    "write_strings",
]

MESH_EXTENSIONS = (".dtm", ".dxf", ".obj", ".ply")
STRING_EXTENSIONS = (".str", ".dxf")


def _extension(path, allowed: tuple[str, ...], what: str) -> str:
    ext = Path(path).suffix.lower()
    if ext not in allowed:
        raise ValueError(
            f"{path}: cannot handle {what} with extension {ext!r}; supported: {', '.join(allowed)}"
        )
    return ext


def read_meshes(path, **options) -> dict[str, Mesh]:
    """All meshes of a file by name.

    Names are ``"<object>.<trisolation>"`` for ``.dtm`` (its ``.str`` is found
    next to it, or pass ``str_path=``), layer names for ``.dxf`` and the file
    stem for ``.obj`` / ``.ply``.
    """
    ext = _extension(path, MESH_EXTENSIONS, "a mesh file")
    if ext == ".dtm":
        result = surpac_dtm.read_surpac_dtm(path, **options)
        return {f"{t.object_id}.{t.trisolation_id}": t.mesh for t in result.trisolations}
    if ext == ".dxf":
        return dict(dxf.read_dxf(path, **options).meshes)
    reader = obj.read_obj if ext == ".obj" else ply.read_ply
    return {Path(path).stem: reader(path, **options)}


def read_mesh(path, **options) -> Mesh:
    """The file's meshes joined into one :class:`Mesh` (no welding between them)."""
    meshes = list(read_meshes(path, **options).values())
    return meshes[0] if len(meshes) == 1 else Mesh.concatenate(meshes)


def write_mesh(path, meshes, **options) -> None:
    """Write one mesh, a list of meshes or a ``{name: mesh}`` dict.

    ``.dtm`` writes the ``.str`` next to it and one trisolation per mesh;
    ``.dxf`` writes 3DFACEs with the dict keys as layer names; ``.obj`` and
    ``.ply`` hold a single mesh, so several are joined first.
    """
    ext = _extension(path, MESH_EXTENSIONS, "a mesh file")
    if ext == ".dtm":
        items = list(meshes.values()) if isinstance(meshes, Mapping) else meshes
        surpac_dtm.write_surpac_dtm(path, items, **options)
    elif ext == ".dxf":
        dxf.write_dxf(path, meshes=meshes, **options)
    else:
        if isinstance(meshes, Mapping):
            meshes = list(meshes.values())
        mesh = meshes if isinstance(meshes, Mesh) else Mesh.concatenate(list(meshes))
        (obj.write_obj if ext == ".obj" else ply.write_ply)(path, mesh, **options)


def read_strings(path, **options) -> StringFile:
    """Strings of a ``.str`` or ``.dxf`` file as one :class:`StringFile`."""
    ext = _extension(path, STRING_EXTENSIONS, "a string file")
    if ext == ".str":
        return surpac_str.read_surpac_str(path, **options)
    return dxf.read_dxf(path, **options).string_file()


def write_strings(path, strings, **options) -> None:
    """Write a :class:`StringFile` (or segments) to a ``.str`` or ``.dxf`` file."""
    ext = _extension(path, STRING_EXTENSIONS, "a string file")
    if ext == ".str":
        surpac_str.write_surpac_str(path, strings, **options)
    else:
        dxf.write_dxf(path, strings=strings, **options)
