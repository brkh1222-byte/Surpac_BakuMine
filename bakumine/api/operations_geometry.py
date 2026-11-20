"""Geometry operations behind the UI dialogs: surfaces from points and mesh import. Like
:mod:`bakumine.api.operations` they wrap the core functions the CLI uses (``bakumine
surface from-points``, ``surface import``), write a job-log entry and add the result to
the project. (:mod:`bakumine.api.operations_implicit` has the implicit solid.)

Files for these operations are uploaded to ``geometry_uploads/<token>/<file name>`` (one
token may hold several files: a ``.dtm`` needs its ``.str`` next to it). A request names a
source either by project object id (``"drillholes"``) or as ``{"upload": token, "file":
name}``. The files a job read are copied to ``inputs/<job id>/`` so the log can be
replayed.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.api.operations import INPUTS, OperationError, _fail, _record, _slug
from bakumine.core import joblog
from bakumine.core.project import Project, ProjectObject

GEOMETRY_UPLOADS = "geometry_uploads"
POINT_SUFFIXES = (".csv", ".txt", ".xlsx", ".xlsm")
STRING_SUFFIXES = (".str", ".dxf")
MESH_SUFFIXES = (".dtm", ".dxf", ".obj", ".ply")
_TOKEN = re.compile(r"[0-9a-f]{12}\Z")
_ERRORS = (ValueError, KeyError, TypeError, FileNotFoundError, OverflowError)


# ------------------------------------------------------------------ uploads


def _safe_name(filename: str) -> str:
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", Path(str(filename).replace("\\", "/")).name)
    name = name.lstrip(".")
    if not name or Path(name).suffix.lower() not in {
        *POINT_SUFFIXES,
        *STRING_SUFFIXES,
        *MESH_SUFFIXES,
    }:
        raise OperationError(
            f"{filename!r}: use a CSV / XLSX table of points, a .str or .dxf string file, "
            "or a .dxf / .obj / .ply / .dtm mesh"
        )
    return name


def _upload_dir(folder: Path, token: Any) -> Path:
    if not isinstance(token, str) or not _TOKEN.match(token):
        raise OperationError(f"invalid upload token {token!r}")
    path = folder / GEOMETRY_UPLOADS / token
    if not path.is_dir():
        raise OperationError(f"no upload {token!r} (upload the file again)")
    return path


def _describe(path: Path) -> dict[str, Any]:
    """What an uploaded file holds, for the dialog."""
    suffix = path.suffix.lower()
    out: dict[str, Any] = {"file": path.name, "roles": []}
    try:
        if suffix in POINT_SUFFIXES:
            from bakumine.core.drillholes.importer import read_raw_table
            from bakumine.core.geometry.surface_points import read_points_table

            raw = read_raw_table(path)
            out.update(columns=[str(c) for c in raw.columns], rows=int(len(raw)))
            try:
                _, info = read_points_table(path)
                out.update(xyz=info["columns"], skipped=info["skipped"])
            except ValueError:
                out["xyz"] = None
            out["roles"] = ["points"]
        if suffix in STRING_SUFFIXES:
            from bakumine.core.io.files import read_strings

            strings = read_strings(path)
            out.update(n_strings=int(strings.n_segments), n_string_points=int(strings.n_points))
            if strings.n_points:
                out["roles"].append("strings")
        if suffix in MESH_SUFFIXES and not (suffix == ".dtm" and not _companion(path)):
            from bakumine.core.io.files import read_mesh

            mesh = read_mesh(path)
            out.update(n_vertices=int(mesh.n_vertices), n_faces=int(mesh.n_faces))
            if mesh.n_faces:
                out["roles"].append("mesh")
        if suffix == ".dtm" and not _companion(path):
            out["needs"] = path.with_suffix(".str").name
    except OperationError:
        raise
    except Exception as exc:  # unreadable file
        raise OperationError(f"{path.name}: {_fail(exc)}") from None
    return out


def _companion(dtm: Path) -> Path | None:
    """The ``.str`` of a Surpac ``.dtm`` (same stem, any case of the extension)."""
    for other in dtm.parent.iterdir():
        if other.stem == dtm.stem and other.suffix.lower() == ".str":
            return other
    return None


def store_geometry_upload(
    folder: Path, filename: str, content: bytes, token: str | None = None
) -> dict[str, Any]:
    """Keep an uploaded file; returns ``{token, files: [description, ...]}``."""
    name = _safe_name(filename)
    if token:
        target = _upload_dir(folder, token)
    else:
        token = uuid.uuid4().hex[:12]
        target = folder / GEOMETRY_UPLOADS / token
        target.mkdir(parents=True)
    (target / name).write_bytes(content)
    try:
        files = [_describe(p) for p in sorted(target.iterdir()) if p.is_file()]
    except OperationError:
        (target / name).unlink(missing_ok=True)
        raise
    return {"token": token, "file": name, "files": files}


def remove_geometry_upload(folder: Path, token: str) -> None:
    shutil.rmtree(_upload_dir(folder, token))


# ------------------------------------------------------------------ sources


class _Sources:
    """Resolves the sources of one request and notes them in the job-log entry."""

    def __init__(self, folder: Path, rec: joblog.Recorder) -> None:
        self.folder = folder
        self.rec = rec
        self.project = Project.open(folder)

    def obj(self, object_id: Any, *kinds: str) -> ProjectObject:
        try:
            obj = self.project.get(str(object_id))
            if obj.kind not in kinds:
                raise KeyError(object_id)
        except KeyError:
            what = " or ".join(kinds)
            raise OperationError(f"no {what} {object_id!r} in the project") from None
        if obj.to_dict() not in self.rec.entry["inputs"]:
            self.rec.entry["inputs"].append(obj.to_dict())
        return obj

    def upload(self, ref: Any, suffixes: tuple[str, ...]) -> tuple[Path, str]:
        """The uploaded file of ``ref`` copied to ``inputs/<job id>/``; (path, relative)."""
        if not isinstance(ref, dict) or "upload" not in ref:
            raise OperationError(f"invalid source {ref!r}")
        source = _upload_dir(self.folder, ref["upload"])
        files = sorted(p for p in source.iterdir() if p.is_file())
        wanted = ref.get("file")
        if wanted:
            chosen = [p for p in files if p.name == wanted]
        else:
            chosen = [
                p for p in files if p.suffix.lower() in suffixes and p.suffix.lower() != ".str"
            ]
            chosen = chosen or [p for p in files if p.suffix.lower() in suffixes]
        if not chosen or chosen[0].suffix.lower() not in suffixes:
            raise OperationError(
                f"upload {ref['upload']!r} has no {' / '.join(suffixes)} file"
                + (f" named {wanted!r}" if wanted else "")
            )
        kept = self.folder / INPUTS / self.rec.id / str(ref["upload"])
        kept.mkdir(parents=True, exist_ok=True)
        for file in files:  # with companions (.str of a .dtm)
            if not (kept / file.name).exists():
                shutil.copy2(file, kept / file.name)
        path = kept / chosen[0].name
        return path, path.relative_to(self.folder).as_posix()

    def strings(self, ref: Any) -> tuple[Any, dict[str, str]]:
        """A string file from a project string object or an upload; (strings, replay)."""
        from bakumine.core.io.files import read_strings

        if isinstance(ref, str):
            obj = self.obj(ref, "string")
            return read_strings(self.project.path_of(obj)), {"object": obj.id, "path": obj.path}
        path, relative = self.upload(ref, STRING_SUFFIXES)
        return read_strings(path), {"file": relative}


def _add_mesh(folder: Path, kind: str, name: str, mesh: Any, comment: str) -> ProjectObject:
    from bakumine.core.io.obj import write_obj

    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / f"{_slug(name)}.obj"
        write_obj(file, mesh, decimals=None, comment=comment)
        return Project.open(folder).add(kind, file, name)


def _name(request: dict[str, Any], default: str) -> str:
    name = str(request.get("name") or default).strip()
    if not name or len(name) > 80:
        raise OperationError("give a name of at most 80 characters")
    return name


def _list(value: Any, what: str) -> list[Any]:
    if value in (None, ""):
        return []
    if not isinstance(value, list):
        raise OperationError(f"{what} must be a list")
    return value


# ------------------------------------------------------------------ surface from points


def surface_from_points(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine surface from-points``: collars, point tables and strings to a DTM."""
    return _record(
        folder, "surface_from_points", request, lambda rec: _surface(folder, request, rec)
    )


def _surface(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.geometry import surface_points as sp

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    name = _name(request, "topography")
    sources = _Sources(folder, rec)
    replay: dict[str, Any] = {"databases": [], "points": [], "strings": [], "breaklines": []}
    clouds: list[np.ndarray] = []
    counts: dict[str, int] = {}
    try:
        for ref in _list(request.get("databases"), "databases"):
            obj = sources.obj(ref, "database")
            pts = sp.collar_points(load_database(sources.project.path_of(obj)))
            clouds.append(pts)
            counts["collars"] = counts.get("collars", 0) + len(pts)
            replay["databases"].append({"object": obj.id, "path": obj.path})
        for ref in _list(request.get("points"), "points"):
            path, relative = sources.upload(ref, POINT_SUFFIXES)
            pts, info = sp.read_points_table(path, ref.get("columns") or None)
            clouds.append(pts)
            counts["table points"] = counts.get("table points", 0) + len(pts)
            replay["points"].append({"file": relative, "columns": info["columns"]})
        for ref in _list(request.get("strings"), "strings"):
            strings, where = sources.strings(ref)
            pts = sp.string_points(strings)
            clouds.append(pts)
            counts["string points"] = counts.get("string points", 0) + len(pts)
            replay["strings"].append(where)
        breaklines: list[np.ndarray] = []
        for ref in _list(request.get("breaklines"), "breaklines"):
            strings, where = sources.strings(ref)
            breaklines += sp.string_lines(strings)
            replay["breaklines"].append(where)
        if not clouds and not breaklines:
            raise OperationError("pick at least one source of points")

        extend = request.get("extend") or {}
        if not isinstance(extend, dict):
            raise OperationError("extend must be an object")
        mode = str(extend.get("mode") or "none")
        rectangle, padding = None, None
        if mode == "none":
            pass
        elif mode == "padding":
            padding = float(extend.get("padding") or 0.0)
        elif mode == "rectangle":
            rectangle = extend.get("rectangle")
            padding = float(extend.get("padding") or 0.0) or None
        elif mode == "model":
            from bakumine.core.blockmodel.storage import load_model

            obj = sources.obj(extend.get("model"), "model")
            rectangle = sp.model_rectangle(load_model(sources.project.path_of(obj)).definition)
            padding = float(extend.get("padding") or 0.0) or None
            replay["model"] = {"object": obj.id, "path": obj.path}
        else:
            raise OperationError("extend mode must be none, padding, rectangle or model")
        z = extend.get("z", "nearest")
        if z in (None, ""):
            z = "nearest"
        if not isinstance(z, str):
            z = float(z)
        elif z != "nearest":
            z = float(z)
        step = extend.get("step")
        result = sp.surface_from_points(
            clouds,
            breaklines,
            rectangle=rectangle,
            padding=padding,
            extend_z=z,
            step=float(step) if step not in (None, "") else None,
            duplicates=str(request.get("duplicates") or "mean"),
        )
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    rec.entry["replay"] = replay
    obj = _add_mesh(folder, "dtm", name, result.dtm.mesh, "surface from points")
    rec.entry["outputs"] = [obj.to_dict()]
    return {"id": obj.id, "name": obj.name, "kind": "dtm", "sources": counts, **result.info}


# ------------------------------------------------------------------ mesh import


def import_mesh(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine surface import``: an uploaded DXF / OBJ / PLY / .dtm as a DTM or solid."""
    return _record(folder, "import_mesh", request, lambda rec: _import_mesh(folder, request, rec))


def _import_mesh(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.geometry.mesh_import import import_mesh as read

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    sources = _Sources(folder, rec)
    path, relative = sources.upload(request, MESH_SUFFIXES)
    try:
        mesh, info = read(path, str(request.get("kind") or "auto"))
    except _ERRORS as exc:
        raise _fail(exc) from None
    name = _name(request, path.stem)
    rec.entry["replay"] = {"file": relative, "kind": info["kind"]}
    obj = _add_mesh(folder, info["kind"], name, mesh, f"imported from {path.name}")
    rec.entry["outputs"] = [obj.to_dict()]
    return {"id": obj.id, "name": obj.name, "source": path.name, **info}
