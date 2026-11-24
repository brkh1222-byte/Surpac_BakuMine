"""Geometry operations behind the UI dialogs: surfaces from points or from elevation rasters,
and mesh import. Like :mod:`bakumine.api.operations` they wrap the core functions the CLI
uses (``bakumine surface from-points``, ``surface from-raster``, ``surface import``), write a
job-log entry and add the result to the project. (:mod:`bakumine.api.operations_implicit`
has the implicit solid.)

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

from bakumine.api.operations import (
    INPUTS,
    OperationError,
    _fail,
    _record,
    _slug,
    object_name,
)
from bakumine.core import joblog
from bakumine.core.project import Project, ProjectObject

GEOMETRY_UPLOADS = "geometry_uploads"
POINT_SUFFIXES = (".csv", ".txt", ".xlsx", ".xlsm")
STRING_SUFFIXES = (".str", ".dxf")
MESH_SUFFIXES = (".dtm", ".dxf", ".obj", ".ply", ".stl")
RASTER_SUFFIXES = (".tif", ".tiff", ".asc")
_TOKEN = re.compile(r"[0-9a-f]{12}\Z")
_ERRORS = (ValueError, KeyError, TypeError, FileNotFoundError, OverflowError)


# ------------------------------------------------------------------ uploads


#: longest stored name of an uploaded file (file systems allow 255 bytes)
MAX_FILE_NAME = 120


def _safe_name(filename: str) -> str:
    if len(str(filename)) > 4096:
        raise OperationError(f"the file name is too long (at most {MAX_FILE_NAME} characters)")
    name = re.sub(r"[^A-Za-z0-9_.\-]+", "_", Path(str(filename).replace("\\", "/")).name)
    name = name.lstrip(".")
    if len(name) > MAX_FILE_NAME:
        raise OperationError(
            f"the file name {name[:40]!r}... is too long (at most {MAX_FILE_NAME} characters)"
        )
    if not name or Path(name).suffix.lower() not in {
        *POINT_SUFFIXES,
        *STRING_SUFFIXES,
        *MESH_SUFFIXES,
        *RASTER_SUFFIXES,
    }:
        raise OperationError(
            f"{filename!r}: use a CSV / XLSX table of points, a .str or .dxf string file, "
            "a .dxf / .obj / .ply / .stl / .dtm mesh or a .tif / .asc elevation grid"
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
        if suffix in RASTER_SUFFIXES:
            from bakumine.core.geometry.raster_dem import read_raster

            r = read_raster(path)
            known = np.isfinite(r.z)
            out.update(
                cells=[int(r.shape[1]), int(r.shape[0])],
                cell=[float(abs(r.dx)), float(abs(r.dy))],
                crs=r.crs,
                geographic=bool(r.geographic),
                heights=[float(r.z[known].min()), float(r.z[known].max())] if known.any() else None,
            )
            out["roles"].append("raster")
    except OperationError:
        raise
    except Exception as exc:  # unreadable file
        # readers name the file by its full path: the client is told the file name only
        text = str(_fail(exc)).replace(str(path), path.name).replace(str(path.parent), "")
        if not text.startswith(path.name):
            text = f"{path.name}: {text}"
        raise OperationError(text) from None
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
    created = not token
    if token:
        target = _upload_dir(folder, token)
    else:
        token = uuid.uuid4().hex[:12]
        target = folder / GEOMETRY_UPLOADS / token
        target.mkdir(parents=True)
    try:
        (target / name).write_bytes(content)
        files = [_describe(p) for p in sorted(target.iterdir()) if p.is_file()]
    except (OperationError, OSError) as exc:
        (target / name).unlink(missing_ok=True)
        if created:  # a rejected first file leaves no empty token folder behind
            shutil.rmtree(target, ignore_errors=True)
        if isinstance(exc, OSError):
            raise OperationError(f"{name}: the file could not be stored") from None
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
    return object_name(request.get("name"), default[:80], 80)


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
        dem = request.get("raster")
        dem_info = None
        if dem:
            if not isinstance(dem, dict):
                raise OperationError("raster must be an object")
            if z != "nearest" or step not in (None, ""):
                raise OperationError("the boundary elevation and step do not apply with a raster")
            dem_info = _dem_around(sources, dem, clouds, breaklines, rectangle, padding, replay)
            clouds.append(dem_info.pop("points"))
            rectangle = padding = None
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
    out = {"id": obj.id, "name": obj.name, "kind": "dtm", "sources": counts, **result.info}
    if dem_info is not None:
        out["dem"] = dem_info
    return out


def _dem_around(
    sources: _Sources,
    dem: dict[str, Any],
    clouds: list[np.ndarray],
    breaklines: list[np.ndarray],
    rectangle: Any,
    padding: float | None,
    replay: dict[str, Any],
) -> dict[str, Any]:
    """The raster nodes around the precise sources (``points``) and what was done; as
    ``bakumine surface from-points --raster``."""
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.geometry import raster_dem as rd

    survey = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    every = np.concatenate([survey, *breaklines]) if breaklines else survey
    if rectangle is None:
        rectangle = [*every[:, :2].min(axis=0), *every[:, :2].max(axis=0)]
    rectangle = [float(v) for v in rectangle]
    pad = float(padding or 0.0)
    area = [rectangle[0] - pad, rectangle[1] - pad, rectangle[2] + pad, rectangle[3] + pad]
    if not (area[2] > area[0] and area[3] > area[1]):
        raise OperationError("the area of the raster is empty: extend the surface by a padding")
    database_crs = None
    if replay["databases"]:
        first = sources.project.path_of(sources.obj(replay["databases"][0]["object"], "database"))
        database_crs = load_database(first).metadata.crs
    z, origin, spacing, grid, crs, notes, raster_replay = _raster_grid(
        sources, dem, area, database_crs
    )
    rule = _shift_rule(dem.get("z_shift")) or "median"
    drop = dem.get("drop_within")
    drop = None if drop in (None, "") else float(drop)
    try:
        points, info = rd.dem_fill(
            survey, breaklines, z, origin, spacing, drop_within=drop, z_shift=rule
        )
    except ValueError as exc:
        raise OperationError(str(exc)) from None
    replay["raster"] = {**raster_replay, "z_shift": rule, "drop_within": drop}
    return {"points": points, **info, "grid": grid, "crs": crs, "notes": notes}


# ------------------------------------------------------------------ elevation rasters


def _area(sources: _Sources, area: Any, collars: np.ndarray | None, replay: dict) -> list[float]:
    """Plan rectangle of ``{mode: collars | model | rectangle, padding}``, grown by padding."""
    from bakumine.core.geometry import surface_points as sp

    area = area or {}
    if not isinstance(area, dict):
        raise OperationError("area must be an object")
    mode = str(area.get("mode") or "collars")
    if mode == "collars":
        if collars is None or len(collars) == 0:
            raise OperationError("pick the drillhole database whose collars give the area")
        rectangle = [*collars[:, :2].min(axis=0), *collars[:, :2].max(axis=0)]
    elif mode == "model":
        from bakumine.core.blockmodel.storage import load_model

        obj = sources.obj(area.get("model"), "model")
        rectangle = sp.model_rectangle(load_model(sources.project.path_of(obj)).definition)
        replay["model"] = {"object": obj.id, "path": obj.path}
    elif mode == "rectangle":
        rectangle = [float(v) for v in area.get("rectangle") or []]
        if len(rectangle) != 4:
            raise OperationError("rectangle needs four numbers: xmin, ymin, xmax, ymax")
        replay["rectangle"] = rectangle
    else:
        raise OperationError("area mode must be collars, model or rectangle")
    pad = float(area.get("padding") or 0.0)
    if not (np.isfinite(pad) and pad >= 0):
        raise OperationError("padding must be >= 0")
    if pad:
        replay["padding"] = pad
    return [rectangle[0] - pad, rectangle[1] - pad, rectangle[2] + pad, rectangle[3] + pad]


def _raster_grid(
    sources: _Sources, block: dict[str, Any], rectangle: list[float], database_crs: str | None
) -> tuple[Any, ...]:
    """Read the uploaded rasters of ``block`` and grid them over ``rectangle``.

    Returns ``(z, origin, spacing, grid, crs, notes, replay)``.
    """
    from bakumine.core.geometry import raster_dem as rd

    refs = _list(block.get("rasters"), "rasters")
    if not refs:
        raise OperationError("upload at least one elevation file (.tif / .asc)")
    raster_crs = block.get("raster_crs") or None
    rasters, files = [], []
    for ref in refs:
        path, relative = sources.upload(ref, RASTER_SUFFIXES)
        rasters.append(rd.read_raster(path, raster_crs))
        files.append(relative)
    try:
        crs, notes = rd.settle_crs(rasters, block.get("crs") or None, database_crs)
    except ValueError:
        raise OperationError(
            "give the projected CRS of the model (e.g. EPSG:32633 for WGS 84 / UTM 33N)"
        ) from None
    spacing = block.get("spacing")
    if spacing in (None, ""):
        spacing = rd.default_spacing(rasters, crs, rectangle)
    else:
        spacing = float(spacing)
        if not (np.isfinite(spacing) and spacing > 0):
            raise OperationError("the grid spacing must be > 0")
    z, origin, grid = rd.grid_from_rasters(rasters, rectangle, spacing, crs)
    if grid["nodes_with_height"] == 0:
        raise OperationError(
            "no node of the area lies on the elevation file: check the CRS and the area"
        )
    replay = {"files": files, "raster_crs": raster_crs, "crs": crs, "spacing": spacing}
    return z, origin, spacing, grid, crs, notes, replay


def _shift_rule(value: Any) -> float | str | None:
    if value in (None, ""):
        return None
    if value in ("median", "none"):
        return str(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise OperationError("the vertical shift must be median, none or metres") from None


def surface_from_raster(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine surface from-raster``: uploaded GeoTIFF / ASCII grids to a DTM, tied to the
    collars of a database."""
    return _record(
        folder, "surface_from_raster", request, lambda rec: _from_raster(folder, request, rec)
    )


def _from_raster(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.geometry import raster_dem as rd
    from bakumine.core.geometry import surface_points as sp

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    name = _name(request, "topography")
    sources = _Sources(folder, rec)
    replay: dict[str, Any] = {}
    try:
        db = None
        if request.get("database"):
            obj = sources.obj(request["database"], "database")
            db = load_database(sources.project.path_of(obj))
            replay["database"] = {"object": obj.id, "path": obj.path}
        collars = sp.collar_points(db) if db is not None else None
        rectangle = _area(sources, request.get("area"), collars, replay)
        z, origin, spacing, grid, crs, notes, raster_replay = _raster_grid(
            sources, request, rectangle, db.metadata.crs if db is not None else None
        )
        replay.update(raster_replay)
        rule = _shift_rule(request.get("z_shift"))
        tolerance = float(request.get("collar_tolerance") or 5.0)
        dtm, tie, _ = rd.raster_surface(
            z, origin, spacing, db.collar if db is not None else None,
            z_shift=rule, tolerance=tolerance,
        )  # fmt: skip
        replay.update(z_shift=rule, collar_tolerance=tolerance)
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    rec.entry["replay"] = replay
    obj = _add_mesh(folder, "dtm", name, dtm.mesh, "surface from elevation raster")
    rec.entry["outputs"] = [obj.to_dict()]
    v = dtm.mesh.vertices
    return {
        "id": obj.id,
        "name": obj.name,
        "kind": "dtm",
        "crs": crs,
        "grid": grid,
        "notes": notes,
        "z_shift": tie["z_shift"],
        "collars": tie["collars"],
        "n_vertices": int(dtm.mesh.n_vertices),
        "n_faces": int(dtm.mesh.n_faces),
        "bounds": [[float(x) for x in v.min(axis=0)], [float(x) for x in v.max(axis=0)]],
    }


# ------------------------------------------------------------------ mesh import


def import_mesh(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine surface import``: an uploaded DXF / OBJ / PLY / STL / .dtm as a DTM or
    solid; ``scale`` multiplies the coordinates (0.3048 for a file in feet)."""
    return _record(folder, "import_mesh", request, lambda rec: _import_mesh(folder, request, rec))


def _import_mesh(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.geometry.mesh_import import import_mesh as read

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    _name(request, "mesh")  # a bad name is refused before the file is read
    sources = _Sources(folder, rec)
    path, relative = sources.upload(request, MESH_SUFFIXES)
    try:
        scale = request.get("scale")
        mesh, info = read(path, str(request.get("kind") or "auto"), 1.0 if scale is None else scale)
    except _ERRORS as exc:
        raise _fail(exc) from None
    name = _name(request, path.stem)
    rec.entry["replay"] = {"file": relative, "kind": info["kind"]}
    if "scale" in info:
        rec.entry["replay"]["scale"] = info["scale"]
    obj = _add_mesh(folder, info["kind"], name, mesh, f"imported from {path.name}")
    rec.entry["outputs"] = [obj.to_dict()]
    return {"id": obj.id, "name": obj.name, "source": path.name, **info}
