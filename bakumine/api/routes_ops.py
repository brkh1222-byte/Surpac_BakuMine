"""Operation endpoints behind the UI dialogs (P6.T2 - P6.T5), under ``/api/v1/ops``.

    GET    /ops/project                       objects and what dialogs pick from
    PUT    /ops/uploads/{table}?filename=     raw file body; returns sheets, columns, mapping,
                                              preview (&add=true: one more file of the table)
    POST   /ops/uploads/{table}/preview       the same for {index, spec}: sheets, header row,
                                              sample-number rule, QC (a recipe source)
    DELETE /ops/uploads/{table}
    POST   /ops/import                        import + validate; saves the database
    POST   /ops/databases/{id}/validate
    POST   /ops/composite
    POST   /ops/models                        create a block model
    POST   /ops/models/{id}/code              attributes + coding steps, in place
    POST   /ops/estimate                      starts a job; returns {id, ...}
    GET    /ops/jobs/{id}                     status, progress, summary
    POST   /ops/jobs/{id}/cancel
    GET    /ops/log                           job log (jobs.jsonl), newest first; undo target
    POST   /ops/undo                          revert the newest model-changing job (P6.T7)

They wrap :mod:`bakumine.api.operations`, which calls the core functions the CLI uses.
Plain ``def`` endpoints run in FastAPI's thread pool; estimation runs on its own thread.
"""

from __future__ import annotations

from typing import Annotated, Any

from fastapi import APIRouter, Body, Depends, HTTPException, Request

from bakumine.api import operations as ops
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import SceneSource, get_scene_source

router = APIRouter(prefix="/api/v1/ops", tags=["operations"])
Source = Annotated[SceneSource, Depends(get_scene_source)]
Json = Annotated[dict[str, Any], Body()]


def _project(source: SceneSource) -> ProjectSceneSource:
    if not isinstance(source, ProjectSceneSource):
        raise HTTPException(
            409, "the server is not serving a project (start it with BAKUMINE_PROJECT)"
        )
    return source


def _guard(call):
    try:
        return call()
    except ops.BusyError as exc:
        raise HTTPException(409, str(exc)) from None
    except ops.OperationError as exc:
        raise HTTPException(400, str(exc)) from None


@router.get("/project")
def get_project(source: Source) -> dict[str, Any]:
    return ops.project_summary(_project(source).folder)


#: largest uploaded table file (HTTP 413 above)
MAX_UPLOAD_BYTES = 1024**3


async def _read_limited(request: Request, limit: int) -> bytes:
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(413, f"the file is larger than {limit // 1024**2} MB")
    chunks, size = [], 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            raise HTTPException(413, f"the file is larger than {limit // 1024**2} MB")
        chunks.append(chunk)
    return b"".join(chunks)


@router.put("/uploads/{table}")
async def put_upload(
    table: str, filename: str, request: Request, source: Source, add: bool = False
) -> dict[str, Any]:
    folder = _project(source).folder
    content = await _read_limited(request, MAX_UPLOAD_BYTES)
    return _guard(lambda: ops.store_upload(folder, table, filename, content, add))


@router.post("/uploads/{table}/preview")
def post_upload_preview(table: str, body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder

    def run() -> dict[str, Any]:
        ops._check_upload_name(table)
        index = body.get("index", 0)
        if not isinstance(index, int) or isinstance(index, bool):
            raise ops.OperationError("index must be a whole number")
        return ops.preview_upload(
            folder, table, index=index, spec=body.get("spec"), hole_id=body.get("hole_id")
        )

    return _guard(run)


@router.delete("/uploads/{table}")
def delete_upload(table: str, source: Source) -> dict[str, str]:
    folder = _project(source).folder
    _guard(lambda: ops.remove_upload(folder, table))
    return {"removed": table}


@router.post("/import")
def post_import(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(
        lambda: ops.import_database(
            folder,
            body.get("name", "drillholes"),
            body.get("maps"),
            body.get("options"),
            body.get("tables"),
            body.get("hole_id"),
            body.get("merge"),
            body.get("holes"),
        )
    )


@router.post("/databases/{object_id}/validate")
def post_validate(object_id: str, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: ops.validate_database(folder, object_id))


@router.post("/composite")
def post_composite(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: ops.composite_database(folder, body))


@router.post("/models")
def post_model(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: ops.create_model(folder, body))


@router.post("/models/{object_id}/code")
def post_code(object_id: str, body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    return _guard(lambda: ops.code_model_in_project(project, object_id, body))


@router.post("/estimate")
def post_estimate(body: Json, source: Source) -> dict[str, Any]:
    project = _project(source)
    model, composites = body.get("model"), body.get("composites")
    if not model or not composites:
        raise HTTPException(400, "give the model and the composites")
    job = _guard(lambda: ops.start_estimation(project, model, composites, body.get("plan") or {}))
    return job.to_dict()


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    try:
        return ops.get_job(job_id).to_dict()
    except KeyError:
        raise HTTPException(404, f"no job {job_id!r}") from None


@router.post("/jobs/{job_id}/cancel")
def cancel_job(job_id: str) -> dict[str, Any]:
    try:
        return ops.cancel_job(job_id).to_dict()
    except KeyError:
        raise HTTPException(404, f"no job {job_id!r}") from None


@router.get("/log")
def get_log(source: Source) -> dict[str, Any]:
    return ops.job_log(_project(source).folder)


@router.post("/undo")
def post_undo(source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: ops.undo_last(folder))
