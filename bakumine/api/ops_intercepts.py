"""Mineralised intervals / significant intercepts dialog: ``POST /api/v1/ops/intercepts``.

Request::

    {"database": id, "table": "assay", "grade": "au",
     "params": {"cutoff": 0.2, "min_thickness": 2, ...},   # InterceptParams fields
     "save_as": "mineralised", "overwrite": false}

Runs what ``bakumine intercepts`` runs
(:func:`bakumine.core.drillholes.mineralised.flag_database_folder`) on the project's
database: the intercept table is written to ``intercepts_<name>.csv`` in the database
folder and, with ``save_as``, the coded intervals become an interval table of the database.
The job is logged as operation ``intercepts`` (replayed by ``bakumine intercepts``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from fastapi import APIRouter

from bakumine.api import operations as ops
from bakumine.api.routes_ops import Json, Source, _guard, _project
from bakumine.core import joblog
from bakumine.core.project import Project

router = APIRouter(prefix="/api/v1/ops", tags=["operations"])

#: intercept rows returned to the dialog (the CSV has all of them)
MAX_ROWS = 2000


def intercepts(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    return ops._record(folder, "intercepts", request, lambda rec: _run(folder, request, rec))


def _run(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.drillholes.mineralised import InterceptParams, flag_database_folder

    project = Project.open(folder)
    try:
        obj = project.get(request["database"])
        if obj.kind != "database":
            raise KeyError(request["database"])
    except KeyError:
        raise ops.OperationError(f"no drillhole database {request.get('database')!r}") from None
    rec.entry["inputs"] = [obj.to_dict()]
    grade = request.get("grade")
    if not grade or not isinstance(grade, str):
        raise ops.OperationError("give the grade column")
    spec = request.get("params")
    if not isinstance(spec, dict):
        raise ops.OperationError("params must be an object with at least the cutoff")
    save_as = request.get("save_as") or None
    try:
        params = InterceptParams.from_dict(spec)
        rec.entry["replay"] = {"params": params.to_dict()}
        result, path = flag_database_folder(
            project.path_of(obj),
            str(request.get("table") or "assay"),
            grade,
            params,
            save_as=save_as,
            overwrite=bool(request.get("overwrite")),
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise ops._fail(exc) from None
    if save_as:
        rec.entry["outputs"] = [obj.to_dict()]
    rows = result.intercepts.head(MAX_ROWS).replace({np.nan: None}).to_dict(orient="records")
    return {
        **result.summary(),
        "database": obj.id,
        "interval_table": save_as,
        "file": path.relative_to(folder).as_posix(),
        "n_intervals": int(len(result.intervals)),
        "warnings": list(result.warnings),
        "truncated": bool(len(result.intercepts) > MAX_ROWS),
        "table": rows,
    }


@router.post("/intercepts")
def post_intercepts(body: Json, source: Source) -> dict[str, Any]:
    folder = _project(source).folder
    return _guard(lambda: intercepts(folder, body))
