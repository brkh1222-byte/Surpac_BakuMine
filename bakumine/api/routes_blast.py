"""Blast BakuMine endpoints, under ``/api/v1/blast`` (plan_blast.md task B5).

    GET  /blast/defaults    default request for the open project (or the demo bench)
    POST /blast/simulate    request JSON -> result bundle (plan_blast.md section 3)
    GET  /blast/result      the latest saved bundle of the open project, 404 when none

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

from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.routes_ops import Json, Source
from bakumine.api.scene import SceneSource
from bakumine.blast import simulate as sim
from bakumine.blast.bench import project_site

router = APIRouter(prefix="/api/v1/blast", tags=["blast"])

BLAST_FOLDER = "blast"
RESULT_FILE = "latest.json"
DESIGN_FILE = "latest_design.json"


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
