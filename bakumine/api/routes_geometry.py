"""Geometry operation endpoints, under ``/api/v1/ops`` next to :mod:`routes_ops`.

    PUT    /ops/geometry/uploads?filename=&token=   raw file body; what the file holds
    DELETE /ops/geometry/uploads/{token}
    POST   /ops/surfaces/from-points                collars / point tables / strings -> DTM
    POST   /ops/meshes/import                       uploaded DXF / OBJ / PLY / .dtm -> DTM or solid

They wrap :mod:`bakumine.api.operations_geometry`; every operation writes a job-log entry
and adds its result to the project, so the scene shows it after the next manifest load.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request

from bakumine.api import operations_geometry as geo
from bakumine.api.routes_ops import (
    MAX_UPLOAD_BYTES,
    Json,
    Source,
    _guard,
    _project,
    _read_limited,
)

router = APIRouter(prefix="/api/v1/ops", tags=["geometry operations"])


@router.put("/geometry/uploads")
async def put_geometry_upload(
    filename: str, request: Request, source: Source, token: str | None = None
) -> dict[str, Any]:
    folder = _project(source).folder
    content = await _read_limited(request, MAX_UPLOAD_BYTES)
    return _guard(lambda: geo.store_geometry_upload(folder, filename, content, token))


@router.delete("/geometry/uploads/{token}")
def delete_geometry_upload(token: str, source: Source) -> dict[str, str]:
    folder = _project(source).folder
    _guard(lambda: geo.remove_geometry_upload(folder, token))
    return {"removed": token}


@router.post("/surfaces/from-points")
def post_surface(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: geo.surface_from_points(folder, body))


@router.post("/meshes/import")
def post_mesh_import(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: geo.import_mesh(folder, body))
