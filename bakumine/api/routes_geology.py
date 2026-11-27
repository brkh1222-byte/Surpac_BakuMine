"""Geology endpoints (geology_contracts.md section 12).

    POST /api/v1/ops/geology/build          {"spec": {...}, "name": "..."} -> a geology object
    POST /api/v1/ops/geology/code           {"model": id, "geology": id, ...} -> coded model
    POST /api/v1/ops/geology/uncode         {"model": id, "merge"?, "attributes"?} -> the
                                            attributes dropped and those kept
    POST /api/v1/ops/geology/remove         {"geology": id} -> the object taken out
    GET  /api/v1/geology/{id}               name, layers, blocks, units, faults, surfaces,
                                            diagnostics
    GET  /api/v1/geology/{id}/provenance    {"build": provenance.json, "coding": {model: record}}

They wrap :mod:`bakumine.api.operations_geology`. The operations run to the end in the
request and write a job-log entry (``geology_build``, ``geology_code``, ``geology_uncode``,
``geology_remove``; the last three can be undone from the job log); the meshes of a
geology object are listed in the scene manifest and served by the mesh endpoint.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Response

from bakumine.api import operations_geology as geo
from bakumine.api.routes_ops import Json, Source, _guard, _project

router = APIRouter(prefix="/api/v1", tags=["geology"])
CACHE_CONTROL = "private, no-cache"


@router.post("/ops/geology/build")
def post_geology_build(body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    return _guard(lambda: geo.geology_build(project, body))


@router.post("/ops/geology/code")
def post_geology_code(body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    return _guard(lambda: geo.geology_code(project, body))


@router.post("/ops/geology/uncode")
def post_geology_uncode(body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    return _guard(lambda: geo.geology_uncode(project, body))


@router.post("/ops/geology/remove")
def post_geology_remove(body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    return _guard(lambda: geo.geology_remove(project, body))


def _or_404(call):
    from fastapi import HTTPException

    try:
        return call()
    except geo.OperationError as exc:
        raise HTTPException(404, str(exc)) from None


@router.get("/geology/{object_id}")
def get_geology(object_id: str, source: Source, response: Response) -> dict[str, Any]:
    project = _project(source)
    response.headers["Cache-Control"] = CACHE_CONTROL
    name = next((o.name for o in project.objects() if o.id == object_id), None)
    return _or_404(lambda: geo.geology_info(project.folder, object_id, name))


@router.get("/geology/{object_id}/provenance")
def get_geology_provenance(object_id: str, source: Source, response: Response) -> dict[str, Any]:
    project = _project(source)
    response.headers["Cache-Control"] = CACHE_CONTROL
    return _or_404(lambda: geo.geology_provenance(project.folder, object_id))
