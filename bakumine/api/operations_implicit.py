"""Implicit solid behind the UI dialog: ``bakumine solid implicit`` on project objects.

Wraps :func:`bakumine.core.geometry.implicit_request.run_implicit` (the function the CLI
calls), writes a job-log entry and adds the solid to the project.
"""

from __future__ import annotations

from typing import Any

from bakumine.api.operations import OperationError, _fail, _record
from bakumine.api.operations_geometry import _ERRORS, _add_mesh, _name, _Sources
from bakumine.core import joblog


def implicit_solid(source: Any, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine solid implicit``: a solid of logged codes straight from the drillholes."""
    return _record(
        source.folder, "implicit_solid", request, lambda rec: _implicit(source, request, rec)
    )


def _implicit(source: Any, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.geometry.implicit_request import run_implicit

    if not isinstance(request, dict):
        raise OperationError("the request must be an object")
    folder = source.folder
    sources = _Sources(folder, rec)
    database = sources.obj(request.get("database"), "database")
    topography = None
    try:
        if request.get("topography"):
            topo = sources.obj(request["topography"], "dtm")
            topography = DTM(source.mesh(topo.id))
        codes = request.get("codes", request.get("code"))
        default = "-".join(str(c) for c in codes) if isinstance(codes, list) else str(codes or "")
        name = _name(request, f"{default or 'domain'} solid")
        result, summary = run_implicit(
            load_database(sources.project.path_of(database)), request, topography
        )
    except OperationError:
        raise
    except _ERRORS as exc:
        raise _fail(exc) from None
    obj = _add_mesh(folder, "solid", name, result.mesh, "implicit solid from drillholes")
    rec.entry["outputs"] = [obj.to_dict()]
    return {"id": obj.id, "name": obj.name, "kind": "solid", **summary}
