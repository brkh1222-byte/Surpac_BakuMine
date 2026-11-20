"""Report endpoints (P6.T6): the volume / tonnes / grade table of a block model.

    POST /api/v1/scene/objects/{id}/report       JSON: {columns, rows, notes}; NaN is null
    POST /api/v1/scene/objects/{id}/report.csv   the same table as a CSV download

The body carries the arguments of :func:`bakumine.core.report.report`, the same ones as
``bakumine blockmodel report``.
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from bakumine.api.blocks import BlockQueryError, parse_constraint
from bakumine.api.routes_scene import _block_data
from bakumine.api.scene import SceneSource, get_scene_source

router = APIRouter(prefix="/api/v1", tags=["report"])
Source = Annotated[SceneSource, Depends(get_scene_source)]


class ReportRequest(BaseModel):
    where: str | None = Field(None, description="constraint JSON or an attribute expression")
    group_by: list[str] = []
    cutoffs: list[float] | None = None
    cutoff_attribute: str | None = None
    #: attribute names, each optionally ``name:unit`` (g/t, ppm, %)
    grade_attributes: list[str] = []
    #: density attribute name or a constant in t/m3
    density: str | float | None = None
    partial: str | None = None
    bench_height: float | None = Field(None, gt=0)
    bench_reference: float = 0.0


def _frame(source: SceneSource, object_id: str, body: ReportRequest) -> pd.DataFrame:
    from bakumine.core.report import ReportError, report

    data = _block_data(source, object_id)
    grades: list[str] = []
    units: dict[str, str] = {}
    for item in body.grade_attributes:
        name, sep, unit = item.partition(":")
        grades.append(name.strip())
        if sep:
            units[name.strip()] = unit.strip()
    density: str | float | None = body.density
    if isinstance(density, str):
        try:
            density = float(density)
        except ValueError:
            pass
    try:
        return report(
            data.model,
            constraint=parse_constraint(body.where),
            resolver=data.resolver,
            group_by=body.group_by,
            cutoffs=body.cutoffs,
            cutoff_attribute=body.cutoff_attribute,
            grade_attributes=grades,
            density=density,
            partial=body.partial,
            bench_height=body.bench_height,
            bench_reference=body.bench_reference,
            units=units,
        )
    except (BlockQueryError, ReportError, ValueError, KeyError) as exc:
        text = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
        raise HTTPException(400, str(text)) from None


def _logged_frame(source: SceneSource, object_id: str, body: ReportRequest) -> pd.DataFrame:
    """The report table; on a project it is also written to the job log (P6.T7)."""
    from bakumine.api.operations import record_report
    from bakumine.api.project_source import ProjectSceneSource

    if not isinstance(source, ProjectSceneSource):
        return _frame(source, object_id, body)
    return record_report(
        source.folder, object_id, body.model_dump(), lambda: _frame(source, object_id, body)
    )


@router.post("/scene/objects/{object_id}/report")
def post_report(object_id: str, body: ReportRequest, source: Source) -> dict[str, Any]:
    table = _logged_frame(source, object_id, body)
    return {
        "columns": [str(c) for c in table.columns],
        "rows": json.loads(table.to_json(orient="records", double_precision=15)),
        "notes": [str(n) for n in table.attrs.get("notes", [])],
    }


@router.post("/scene/objects/{object_id}/report.csv")
def post_report_csv(object_id: str, body: ReportRequest, source: Source) -> Response:
    from bakumine.core.report import to_csv

    table = _logged_frame(source, object_id, body)
    with tempfile.TemporaryDirectory() as folder:
        text = to_csv(table, Path(folder) / "report.csv").read_text(encoding="utf-8")
    return Response(
        content=text,
        media_type="text/csv",
        headers={"Content-Disposition": f'attachment; filename="{object_id}-report.csv"'},
    )
