"""Blast BakuMine endpoints, under ``/api/v1/blast`` (plan_blast.md task B5).

    GET  /blast/defaults    default request for the open project (or the demo bench)
    POST /blast/simulate    request JSON -> result bundle (plan_blast.md section 3)
    GET  /blast/result      the latest saved bundle of the open project, 404 when none
    GET  /blast/imagery     satellite imagery of the project: the saved tile plan, or what is needed
    POST /blast/imagery     download satellite tiles over the topography (an explicit action:
                            the tile numbers tell the imagery provider where the site is)
    GET  /blast/imagery/tile/{z}/{y}/{x}   one downloaded tile (never fetched on demand)

When the server serves a project with a topography, the blast is put on it (the first
``dtm`` object, or the one named by ``topo`` in the body) and the damage is computed for
the blocks of the block model named by ``model`` in the body (none by default: a grid of
blocks is generated). Otherwise it runs on the synthetic demo bench. With a project the
bundle is saved as ``<project>/blast/latest.json`` and its design as
``<project>/blast/latest_design.json``; nothing is added to ``project.json``.

They wrap :func:`bakumine.blast.simulate.simulate`, the function the CLI runs.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.routes_ops import Json, Source
from bakumine.api.scene import SceneSource
from bakumine.blast import imagery
from bakumine.blast import simulate as sim
from bakumine.blast.bench import project_site

router = APIRouter(prefix="/api/v1/blast", tags=["blast"])

BLAST_FOLDER = "blast"
RESULT_FILE = "latest.json"
DESIGN_FILE = "latest_design.json"
IMAGERY_FOLDER = "imagery"
IMAGERY_PLAN = "plan.json"
IMAGERY_MAX_TILES = 400
# Default block: 16 x 16 tiles of 256 px fill the 4096 px mosaic the viewer builds.
IMAGERY_DEFAULT_TILES = 256


def _folder(source: SceneSource) -> Path | None:
    """The project folder, or None when the scene source is not a project."""
    return source.folder if isinstance(source, ProjectSceneSource) else None


def _has_topography(folder: Path) -> bool:
    from bakumine.core.project import Project

    return bool(Project.open(folder).list("dtm"))


def _write_atomic(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _optional_id(body: dict[str, Any], key: str) -> str | None:
    value = body.get(key)
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{key} must be an object id (text) or null")
    return value


@router.get("/defaults")
def get_defaults(source: Source) -> dict[str, Any]:
    folder = _folder(source)
    try:
        if folder is not None and _has_topography(folder):
            return sim.defaults(project_site(folder)[0])
        return sim.defaults()
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None


@router.post("/simulate")
def post_simulate(body: Json, source: Source) -> dict[str, Any]:
    folder = _folder(source)
    try:
        topo, model = _optional_id(body, "topo"), _optional_id(body, "model")
        if folder is not None and (topo is not None or _has_topography(folder)):
            ground, blocks, _ = project_site(folder, topo, model)
            bundle = sim.simulate(body, ground=ground, blocks=blocks)
        else:
            bundle = sim.simulate(body)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    if folder is not None:
        target = folder / BLAST_FOLDER
        _write_atomic(target / RESULT_FILE, json.dumps(bundle, allow_nan=False))
        _write_atomic(target / DESIGN_FILE, json.dumps(bundle["design"], indent=2, allow_nan=False))
    return bundle


@router.get("/result")
def get_result(source: Source) -> dict[str, Any]:
    folder = _folder(source)
    file = None if folder is None else folder / BLAST_FOLDER / RESULT_FILE
    if file is None or not file.is_file():
        raise HTTPException(404, "no blast result yet: run a simulation first")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError:
        raise HTTPException(404, "the saved blast result cannot be read: run it again") from None


def _database_crs(folder: Path) -> str | None:
    """The coordinate system of the project's first drillhole database, when it states one."""
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.project import Project

    project = Project.open(folder)
    for obj in project.list("database"):
        try:
            crs = load_database(project.path_of(obj)).metadata.crs
        except (OSError, ValueError):
            continue
        if crs:
            return str(crs)
    return None


def _imagery_plan(folder: Path) -> dict[str, Any] | None:
    file = folder / BLAST_FOLDER / IMAGERY_FOLDER / IMAGERY_PLAN
    if not file.is_file():
        return None
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError:
        return None


@router.get("/imagery")
def get_imagery(source: Source) -> dict[str, Any]:
    folder = _folder(source)
    if folder is None:
        return {"available": False, "plan": None, "crs": None, "project": False}
    plan = _imagery_plan(folder)
    crs = plan["crs"] if plan else _database_crs(folder)
    return {"available": plan is not None, "plan": plan, "crs": crs, "project": True}


@router.post("/imagery")
def post_imagery(body: Json, source: Source) -> dict[str, Any]:
    """Body: ``{"crs": "EPSG:...", "rectangle": [xmin, ymin, xmax, ymax], "max_tiles": 256}``;
    ``crs`` defaults to the drillhole database's, ``rectangle`` to the topography's extent."""
    folder = _folder(source)
    if folder is None:
        raise HTTPException(400, "satellite imagery needs an open project with real coordinates")
    try:
        crs = body.get("crs") or _database_crs(folder)
        if not crs:
            raise ValueError(
                "the project does not state its coordinate system: give one, e.g. EPSG:32633"
            )
        rectangle = body.get("rectangle")
        if rectangle is None:
            bounds = project_site(folder, _optional_id(body, "topo"))[0].bounds
            rectangle = [bounds[0][0], bounds[0][1], bounds[1][0], bounds[1][1]]
        max_tiles = int(body.get("max_tiles", IMAGERY_DEFAULT_TILES))
        if not 1 <= max_tiles <= IMAGERY_MAX_TILES:
            raise ValueError(f"max_tiles must be between 1 and {IMAGERY_MAX_TILES}")
        target = folder / BLAST_FOLDER / IMAGERY_FOLDER
        plan, counts = imagery.fetch_best(
            [float(v) for v in rectangle], str(crs), target, max_tiles=max_tiles
        )
    except (ValueError, TypeError) as exc:
        raise HTTPException(400, str(exc)) from None
    if counts["downloaded"] + counts["cached"] == 0:
        raise HTTPException(400, "the imagery server has no tiles for this area")
    plan["fetched"] = counts
    _write_atomic(target / IMAGERY_PLAN, json.dumps(plan, allow_nan=False))
    return {"available": True, "plan": plan, "crs": plan["crs"], "project": True}


@router.get("/imagery/tile/{z}/{y}/{x}")
def get_imagery_tile(z: int, y: int, x: int, source: Source) -> FileResponse:
    folder = _folder(source)
    file = (
        None
        if folder is None
        else imagery.tile_path(folder / BLAST_FOLDER / IMAGERY_FOLDER, z, x, y)
    )
    if file is None or not file.is_file():
        raise HTTPException(404, "this tile was not downloaded")
    return FileResponse(file, media_type="image/jpeg")
