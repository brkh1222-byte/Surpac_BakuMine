"""Resource classification dialog: ``POST /api/v1/ops/models/{id}/classify``.

Request::

    {"params": {"attribute": "au", "output": "res_class", "categories": [...],
                "hole_support": {...} | null, "smoothing": {...} | null,
                "overwrite_output": false},   # true: replace an attribute named like
                                              # "output" that no classification wrote
                # hole_support may carry a sample_filter, and a class its own
                # hole_support (radius, nth, ...)
     "composites": id | null}        # for the drillhole support

Runs what ``bakumine classify`` runs
(:func:`bakumine.core.estimation.classification.classify_model_folder`) on the project's
model, in place. The job is logged as operation ``classify``; it changes attributes only,
so its undo snapshot holds the class attribute and the two drillhole-support attributes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter

from bakumine.api import operations as ops
from bakumine.api.routes_ops import Json, Source, _guard, _project
from bakumine.core import joblog
from bakumine.core.project import Project

router = APIRouter(prefix="/api/v1/ops", tags=["operations"])


def classify_model(folder: Path, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
    params = {"model": model_id, **request}
    return ops._record(folder, "classify", params, lambda rec: _run(folder, model_id, request, rec))


def _run(
    folder: Path, model_id: str, request: dict[str, Any], rec: joblog.Recorder
) -> dict[str, Any]:
    from bakumine.core.estimation.classification import (
        ClassificationParams,
        classify_model_folder,
    )

    path = ops._model_path(folder, model_id)
    spec = request.get("params")
    if not isinstance(spec, dict):
        raise ops.OperationError("params must be an object (attribute, categories, ...)")
    try:
        params = ClassificationParams.from_dict(spec)
    except (ValueError, KeyError, TypeError) as exc:
        raise ops._fail(exc) from None
    rec.entry["replay"] = {"params": params.to_dict()}
    project = Project.open(folder)
    composites = None
    composites_id = request.get("composites") or None
    if composites_id is not None:
        try:
            if project.get(composites_id).kind != "composite":
                raise KeyError(composites_id)
        except KeyError:
            raise ops.OperationError(f"no composites {composites_id!r}") from None
        composites = ops._read_composites(project.path_of(composites_id))
    columns = params.written_names()
    ops._claim(model_id)
    try:
        ops._will_modify(rec, folder, model_id, path, columns)
        if composites_id is not None:
            rec.entry["inputs"].insert(1, project.get(composites_id).to_dict())
        summary = classify_model_folder(path, params, composites)
    except ops.OperationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise ops._fail(exc) from None
    finally:
        ops._release(model_id)
    return summary.to_dict()


@router.post("/models/{object_id}/classify")
def post_classify(object_id: str, body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: classify_model(folder, object_id, body))
