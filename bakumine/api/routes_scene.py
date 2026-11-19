"""Scene endpoints, versioned under ``/api/v1``.

    GET /api/v1/scene                                   JSON manifest
    GET /api/v1/scene/objects/{id}/drillholes           BKMA; ?table=<interval table>
                                                              &tolerance=<metres>
    GET /api/v1/scene/objects/{id}/mesh                 BKMA; ?normals=1

Block models will follow the same scheme (``.../objects/{id}/blocks`` with attribute and
level-of-detail query parameters); the manifest's ``kind`` and ``href`` of each object
tell the client which endpoint to call.

Binary responses carry a strong ``ETag`` and ``Cache-Control: private, no-cache``: the
browser keeps the payload and revalidates it with ``If-None-Match`` (304 when unchanged).
Compression is applied by the app's gzip middleware. Built payloads are kept in a small
in-process LRU keyed by the source version, so a revalidation costs no rebuild.
"""

from __future__ import annotations

import hashlib
import threading
from collections import OrderedDict
from collections.abc import Callable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response

from bakumine.api.binary import MEDIA_TYPE
from bakumine.api.payloads import (
    DEFAULT_TOLERANCE,
    drillhole_payload,
    mesh_payload,
    scene_manifest,
)
from bakumine.api.scene import SceneSource, get_scene_source

router = APIRouter(prefix="/api/v1", tags=["scene"])

Source = Annotated[SceneSource, Depends(get_scene_source)]
CACHE_CONTROL = "private, no-cache"


class PayloadCache:
    """Small thread-safe LRU of built payloads: key -> (bytes, etag)."""

    def __init__(self, capacity: int = 16) -> None:
        self.capacity = capacity
        self._items: OrderedDict[tuple[Any, ...], tuple[bytes, str]] = OrderedDict()
        self._lock = threading.Lock()

    def get(self, key: tuple[Any, ...], build: Callable[[], bytes]) -> tuple[bytes, str]:
        with self._lock:
            hit = self._items.get(key)
            if hit is not None:
                self._items.move_to_end(key)
                return hit
        payload = build()
        item = (payload, hashlib.sha1(payload).hexdigest()[:20])
        with self._lock:
            self._items[key] = item
            while len(self._items) > self.capacity:
                self._items.popitem(last=False)
        return item

    def clear(self) -> None:
        with self._lock:
            self._items.clear()


payload_cache = PayloadCache()


def _etag_matches(header: str | None, etag: str) -> bool:
    if not header:
        return False
    candidates = [c.strip().removeprefix("W/").strip('"') for c in header.split(",")]
    return "*" in candidates or etag in candidates


def _binary(request: Request, payload: bytes, etag: str) -> Response:
    headers = {"ETag": f'"{etag}"', "Cache-Control": CACHE_CONTROL}
    if _etag_matches(request.headers.get("if-none-match"), etag):
        return Response(status_code=304, headers=headers)
    return Response(content=payload, media_type=MEDIA_TYPE, headers=headers)


def _cache_key(source: SceneSource, *parts: Any) -> tuple[Any, ...]:
    return (id(source), source.project_id, source.version, *parts)


@router.get("/scene")
def get_scene(source: Source, response: Response) -> dict[str, Any]:
    """The scene manifest: objects, world bounds, attributes with ranges or categories."""
    response.headers["Cache-Control"] = CACHE_CONTROL
    return scene_manifest(source)


@router.get("/scene/objects/{object_id}/drillholes")
def get_drillholes(
    object_id: str,
    request: Request,
    source: Source,
    table: Annotated[str | None, Query(description="interval table to include")] = None,
    tolerance: Annotated[float, Query(gt=0, le=100)] = DEFAULT_TOLERANCE,
) -> Response:
    try:
        data = source.drillholes(object_id)
    except KeyError:
        raise HTTPException(404, f"no drillhole object {object_id!r}") from None
    if table is not None and table not in data.database.intervals:
        raise HTTPException(404, f"drillhole object {object_id!r} has no table {table!r}")
    payload, etag = payload_cache.get(
        _cache_key(source, "drillholes", object_id, table, tolerance),
        lambda: drillhole_payload(data, object_id, table, tolerance),
    )
    return _binary(request, payload, etag)


@router.get("/scene/objects/{object_id}/mesh")
def get_mesh(
    object_id: str,
    request: Request,
    source: Source,
    normals: Annotated[bool, Query(description="include per-vertex normals")] = False,
) -> Response:
    try:
        mesh = source.mesh(object_id)
    except KeyError:
        raise HTTPException(404, f"no mesh object {object_id!r}") from None
    payload, etag = payload_cache.get(
        _cache_key(source, "mesh", object_id, normals),
        lambda: mesh_payload(mesh, object_id, normals),
    )
    return _binary(request, payload, etag)
