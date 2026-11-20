"""Project operations behind the UI dialogs (P6.T2 - P6.T5).

Every function wraps the core functions the CLI commands use (``bakumine import``,
``validate``, ``composite``, ``blockmodel create`` / ``code``, ``estimate``) with the same
defaults, and adds the result to the project folder. The routes in ``routes_ops`` are thin
wrappers around these functions; the tests call both and compare with the CLI.
"""

from __future__ import annotations

import tempfile
import threading
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.project import Project

UPLOADS = "uploads"
MAX_ISSUES = 500
PREVIEW_ROWS = 8


class OperationError(ValueError):
    """A request is not valid (HTTP 400)."""


class BusyError(OperationError):
    """The model is being estimated (HTTP 409)."""


_busy: set[str] = set()
_busy_lock = threading.Lock()


def _fail(exc: Exception) -> OperationError:
    text = exc.args[0] if isinstance(exc, KeyError) and exc.args else exc
    return OperationError(str(text))


# ------------------------------------------------------------------ uploads / import


def _table_kind(name: str) -> str:
    return name if name in ("collar", "survey") else "interval"


def _upload_path(folder: Path, name: str) -> Path | None:
    for file in sorted((folder / UPLOADS).glob(f"{name}.*")):
        return file
    return None


def store_upload(folder: Path, name: str, filename: str, content: bytes) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import _check_table_name

    if name not in ("collar", "survey"):
        try:
            _check_table_name(name)
        except ValueError as exc:
            raise _fail(exc) from None
    suffix = Path(filename).suffix.lower()
    if suffix not in (".csv", ".txt", ".xlsx", ".xlsm"):
        raise OperationError(f"{filename!r}: use a CSV or XLSX file")
    target = folder / UPLOADS
    target.mkdir(exist_ok=True)
    for old in target.glob(f"{name}.*"):
        old.unlink()
    (target / f"{name}{suffix}").write_bytes(content)
    return preview_upload(folder, name)


def remove_upload(folder: Path, name: str) -> None:
    for old in (folder / UPLOADS).glob(f"{name}.*"):
        old.unlink()


def preview_upload(folder: Path, name: str, delimiter: str | None = ",") -> dict[str, Any]:
    from bakumine.core.drillholes.importer import read_raw_table, suggest_column_map

    path = _upload_path(folder, name)
    if path is None:
        raise OperationError(f"no uploaded file for table {name!r}")
    try:
        raw = read_raw_table(path, delimiter=delimiter)
        suggestion = suggest_column_map(raw.columns, _table_kind(name))  # type: ignore[arg-type]
    except Exception as exc:  # unreadable file
        raise _fail(exc) from None
    return {
        "name": name,
        "kind": _table_kind(name),
        "file": path.name,
        "rows": int(len(raw)),
        "columns": [str(c) for c in raw.columns],
        "preview": raw.head(PREVIEW_ROWS).astype(str).to_dict(orient="records"),
        "mapping": {str(k): v for k, v in suggestion.mapping.items()},
        "missing": list(suggestion.missing),
        "unmatched": [str(c) for c in suggestion.unmatched],
    }


def list_uploads(folder: Path) -> list[dict[str, Any]]:
    names = (
        sorted({p.stem for p in (folder / UPLOADS).glob("*.*")})
        if (folder / UPLOADS).is_dir()
        else []
    )
    return [preview_upload(folder, n) for n in names]


def validation_dict(report: Any) -> dict[str, Any]:
    from bakumine.core.drillholes.validate import CHECKS

    findings = report.findings
    severity = findings["severity"].value_counts()
    counts = [
        {"code": code, "severity": CHECKS[code][0], "description": CHECKS[code][1], "count": n}
        for code, n in sorted(report.counts().items(), key=lambda kv: (-kv[1], kv[0]))
    ]
    order = {"error": 0, "warning": 1, "info": 2}
    ranked = findings.assign(_o=findings["severity"].map(order)).sort_values("_o", kind="stable")
    issues = [
        {k: (None if pd.isna(v) else (int(v) if k == "row" else str(v))) for k, v in row.items()}
        for row in ranked.drop(columns="_o").head(MAX_ISSUES).to_dict(orient="records")
    ]
    return {
        "n_holes": int(len(report.summary)),
        "errors": int(severity.get("error", 0)),
        "warnings": int(severity.get("warning", 0)),
        "infos": int(severity.get("info", 0)),
        "holes_with_errors": int((report.summary["errors"] > 0).sum()),
        "counts": counts,
        "issues": issues,
        "truncated": bool(len(findings) > MAX_ISSUES),
    }


def validate_database(folder: Path, object_id: str) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.drillholes.validate import validate

    project = Project.open(folder)
    try:
        path = project.path_of(object_id)
        if project.get(object_id).kind != "database":
            raise KeyError(object_id)
    except KeyError:
        raise OperationError(f"no drillhole database {object_id!r}") from None
    # CLI defaults: max_deviation 10 deg / 30 m, depth tolerance 1e-6
    report = validate(load_database(path), max_deviation_deg_per_30m=10.0, depth_tolerance=1e-6)
    report.findings.to_csv(path / "validation_findings.csv", index=False)
    return validation_dict(report)


def import_database(
    folder: Path,
    name: str,
    maps: dict[str, dict[str, str]] | None = None,
    options: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``bakumine import`` + ``validate`` on the uploaded tables; saves the database."""
    from bakumine.core.drillholes.importer import (
        ImportOptions,
        TableSource,
        import_drillholes,
        read_raw_table,
        suggest_column_map,
    )
    from bakumine.core.drillholes.validate import validate

    maps = dict(maps or {})
    uploads = (
        {p.stem: p for p in (folder / UPLOADS).glob("*.*")} if (folder / UPLOADS).is_dir() else {}
    )
    for needed in ("collar", "survey"):
        if needed not in uploads:
            raise OperationError(f"upload the {needed} table first")
    try:
        opts = ImportOptions(**{k: v for k, v in (options or {}).items() if v not in (None, "")})
        sources: dict[str, TableSource] = {}
        for table, path in uploads.items():
            mapping = maps.get(table)
            if not mapping:
                raw = read_raw_table(path, delimiter=opts.delimiter, encoding=opts.encoding)
                suggestion = suggest_column_map(raw.columns, _table_kind(table))  # type: ignore[arg-type]
                if suggestion.missing:
                    raise OperationError(
                        f"no column found for {suggestion.missing} in the {table} table"
                    )
                mapping = suggestion.mapping
            sources[table] = TableSource(path, mapping)
        intervals = {n: s for n, s in sources.items() if n not in ("collar", "survey")}
        db = import_drillholes(sources["collar"], sources["survey"], intervals, options=opts)
        report = validate(db, max_deviation_deg_per_30m=10.0, depth_tolerance=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            saved = db.save(Path(tmp) / "database")
            report.findings.to_csv(saved / "validation_findings.csv", index=False)
            obj = Project.open(folder).add("database", saved, name or "drillholes")
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        raise _fail(exc) from None
    result = validation_dict(report)
    result.update(
        {
            "id": obj.id,
            "name": obj.name,
            "tables": {n: int(len(t)) for n, t in db.tables.items()},
            "notes": [
                f"{n['table']}.{n['column']}: {n['kind']} x {n['count']}"
                for n in db.metadata.notes[:20]
            ],
        }
    )
    return result


# ------------------------------------------------------------------ project summary


def project_summary(folder: Path) -> dict[str, Any]:
    """Objects with what the dialogs pick from: tables and columns, attributes."""
    from bakumine.core.blockmodel.storage import load_model
    from bakumine.core.drillholes.importer import load_database

    project = Project.open(folder)
    out: list[dict[str, Any]] = []
    for o in project.list():
        entry: dict[str, Any] = {"id": o.id, "kind": o.kind, "name": o.name}
        path = project.path_of(o)
        try:
            if o.kind == "database":
                db = load_database(path)
                entry["tables"] = {
                    n: {
                        "columns": [str(c) for c in t.columns],
                        "attributes": db.attribute_columns(n),
                        "rows": int(len(t)),
                    }
                    for n, t in db.intervals.items()
                }
            elif o.kind == "composite":
                table = _read_composites(path)
                entry["columns"] = [str(c) for c in table.columns]
                entry["rows"] = int(len(table))
            elif o.kind == "model":
                model = load_model(path)
                entry["attributes"] = [
                    {"name": n, "kind": str(getattr(a, "kind", "float"))}
                    for n, a in model.attributes.items()
                ]
                entry["n_blocks"] = int(model.n_blocks)
        except Exception as exc:  # a damaged object must not hide the others
            entry["error"] = str(exc)
        out.append(entry)
    return {"name": project.name, "objects": out, "uploads": list_uploads(folder)}


def _read_composites(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(path)
    return pd.read_csv(path, dtype={"hole_id": str})


# ------------------------------------------------------------------ composite


def composite_database(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine composite`` with the CLI defaults for everything not in ``request``."""
    from bakumine.core.drillholes.composite import (
        IntervalDomains,
        composite,
        split_at_domains,
    )
    from bakumine.core.drillholes.importer import load_database

    project = Project.open(folder)
    try:
        db_obj = project.get(request["database"])
        if db_obj.kind != "database":
            raise KeyError(request["database"])
    except KeyError:
        raise OperationError(f"no drillhole database {request.get('database')!r}") from None
    db = load_database(project.path_of(db_obj))
    table = request.get("table") or "assay"
    mode = request.get("mode") or "downhole"
    if table not in db.intervals:
        raise OperationError(f"no interval table {table!r}; database has {list(db.intervals)}")
    samples = db.intervals[table]
    domain_column = request.get("domain_column") or None
    domain_table = request.get("domain_table") or None
    domain: str | None = None
    try:
        if domain_column:
            domain_table = domain_table or table
            if domain_table not in db.intervals:
                raise OperationError(f"no interval table {domain_table!r}")
            if domain_column not in db.intervals[domain_table].columns:
                raise OperationError(f"table {domain_table!r} has no column {domain_column!r}")
            if domain_table == table:
                domain = domain_column
            else:
                domain = "domain"
                while domain in samples.columns:
                    domain = f"_{domain}"
                samples = split_at_domains(
                    samples,
                    IntervalDomains(db.intervals[domain_table], domain_column),
                    name=domain,
                    on_invalid="raise",
                )
        if mode == "domain" and domain is None:
            raise OperationError("mode 'domain' needs a domain column")
        attributes = request.get("attributes") or None
        out = composite(
            samples,
            db.desurvey(method="minimum_curvature"),
            length=request.get("length"),
            mode=mode,
            attributes=attributes,
            bench_height=request.get("bench_height"),
            bench_reference=float(request.get("bench_reference") or 0.0),
            domain=domain,
            best_fit=False,
            density=None,
            fill_missing=None,
            min_length=0.0,
            short="keep",
            min_coverage=0.0,
            drop_empty=True,
            on_invalid="raise",
        )
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from None
    if not len(out):
        raise OperationError("no composites were produced")
    name = request.get("name") or f"composites_{table}"
    with tempfile.TemporaryDirectory() as tmp:
        file = Path(tmp) / f"{name}.parquet"
        out.to_parquet(file)
        obj = Project.open(folder).add("composite", file, name)
    return {
        "id": obj.id,
        "name": obj.name,
        "rows": int(len(out)),
        "holes": int(out["hole_id"].nunique()),
        "length_mean": float(out["length"].mean()),
        "length_total": float(out["length"].sum()),
        "columns": [str(c) for c in out.columns],
        "domains": (
            {str(k): int(v) for k, v in out["domain"].fillna("(none)").value_counts().items()}
            if "domain" in out.columns
            else {}
        ),
    }


# ------------------------------------------------------------------ block model


def _triple(value: Any, what: str) -> tuple[float, ...]:
    try:
        values = tuple(float(v) for v in value)
    except (TypeError, ValueError):
        values = ()
    if len(values) != 3:
        raise OperationError(f"{what} needs three numbers")
    return values


def create_model(folder: Path, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine blockmodel create``: explicit grid, or fitted to drillholes / solids."""
    from bakumine.core.blockmodel.blocks import BlockModel
    from bakumine.core.blockmodel.definition import BlockModelDefinition
    from bakumine.core.blockmodel.storage import save_model
    from bakumine.core.drillholes.importer import load_database

    project = Project.open(folder)
    size = _triple(request.get("size"), "size")
    options: dict[str, Any] = {"max_depth": int(request.get("max_depth") or 0)}
    if request.get("angles"):
        options["angles"] = _triple(request["angles"], "angles")
    try:
        if request.get("extent_from"):
            clouds = []
            for ref in request["extent_from"]:
                obj = project.get(ref)
                if obj.kind == "database":
                    collar = load_database(project.path_of(obj)).collar
                    top = collar[["x", "y", "z"]].to_numpy(dtype=float)
                    bottom = top.copy()
                    bottom[:, 2] -= collar["max_depth"].to_numpy(dtype=float)
                    clouds += [top, bottom]
                elif obj.kind in ("solid", "dtm"):
                    from bakumine.core.io.obj import read_obj

                    clouds.append(np.asarray(read_obj(project.path_of(obj)).vertices))
                else:
                    raise OperationError(
                        f"{ref!r} is a {obj.kind}: pick drillholes, a solid or a DTM"
                    )
            padding = request.get("padding", 0.0)
            margin = (float(padding),) * 3 if np.ndim(padding) == 0 else _triple(padding, "padding")
            definition = BlockModelDefinition.auto_extent(
                np.concatenate(clouds), size, margin=margin, **options
            )
        else:
            origin = _triple(request.get("origin"), "origin")
            shape = tuple(int(v) for v in _triple(request.get("shape"), "shape"))
            definition = BlockModelDefinition(origin, size, shape, **options)
        model = BlockModel(definition)
        with tempfile.TemporaryDirectory() as tmp:
            header = save_model(model, Path(tmp) / "model")
            obj = project.add("model", Path(tmp) / "model", request.get("name") or "model")
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from None
    d = definition
    return {
        "id": obj.id,
        "name": obj.name,
        "origin": [float(v) for v in d.origin],
        "size": [float(v) for v in d.block_size],
        "shape": [int(d.nx), int(d.ny), int(d.nz)],
        "n_blocks": int(header.n_blocks),
    }


def project_resolver(source: Any) -> Any:
    """Geometry resolver of the project's solids and DTMs by object id."""
    from bakumine.core.blockmodel.constraints import DictResolver
    from bakumine.core.geometry.dtm import DTM

    project = Project.open(source.folder)
    return DictResolver(
        dtms={o.id: DTM(source.mesh(o.id)) for o in project.list("dtm")},
        solids={o.id: source.mesh(o.id) for o in project.list("solid")},
    )


def _model_path(folder: Path, model_id: str) -> Path:
    project = Project.open(folder)
    try:
        if project.get(model_id).kind != "model":
            raise KeyError(model_id)
        return project.path_of(model_id)
    except KeyError:
        raise OperationError(f"no block model {model_id!r}") from None


def _claim(model_id: str) -> None:
    with _busy_lock:
        if model_id in _busy:
            raise BusyError(f"block model {model_id!r} is busy")
        _busy.add(model_id)


def _release(model_id: str) -> None:
    with _busy_lock:
        _busy.discard(model_id)


def code_model_in_project(source: Any, model_id: str, request: dict[str, Any]) -> dict[str, Any]:
    """``bakumine blockmodel code``: attributes and steps; the model is saved in place."""
    from bakumine.core.blockmodel.coding import code_model
    from bakumine.core.blockmodel.constraints import constraint_from_dict
    from bakumine.core.blockmodel.storage import load_model, save_model

    path = _model_path(source.folder, model_id)
    _claim(model_id)
    try:
        model = load_model(path)
        for item in request.get("attributes", []):
            item = dict(item)
            if item.get("name") in model.attributes:
                continue
            model.add_attribute(item.pop("name"), item.pop("kind", "float"), **item)
        steps = []
        for item in request.get("steps", []):
            item = dict(item)
            if item.get("constraint") is not None:
                item["constraint"] = constraint_from_dict(item["constraint"])
            if request.get("subblock"):
                item.setdefault("subblock", True)
            steps.append(item)
        results = code_model(
            model, steps, project_resolver(source), merge=bool(request.get("merge"))
        )
        header = save_model(model, path)
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from None
    finally:
        _release(model_id)
    return {
        "n_blocks": int(header.n_blocks),
        "steps": [
            {
                "attribute": r.attribute,
                "value": str(r.value),
                "n_assigned": int(r.n_assigned),
                "volume": float(r.volume),
                "n_split": int(r.n_split),
                "n_blocks": int(r.n_blocks),
            }
            for r in results
        ],
    }


# ------------------------------------------------------------------ estimation


def build_plan(spec: dict[str, Any]) -> Any:
    """An :class:`EstimationPlan` from the dialog's request.

    ``spec``: attribute, value_column, method (nn / idw / ok), power, ellipsoid {radii
    [major, semi, minor], angles [bearing, plunge, dip]}, variogram {nugget, structures
    [{type, sill, ranges?, angles?}]}, domain {name, block_constraint, sample_filter},
    passes [{name, radius_factor, min_samples, max_samples, max_per_hole}], discretisation.
    """
    from bakumine.core.estimation.block_estimation import (
        EstimationDomain,
        EstimationPass,
        EstimationPlan,
    )
    from bakumine.core.estimation.rotation import Ellipsoid
    from bakumine.core.estimation.search import SearchParams
    from bakumine.core.estimation.variogram import Structure, VariogramModel

    try:
        method = str(spec.get("method", "ok")).lower()
        e = spec.get("ellipsoid") or {}
        radii = _triple(e.get("radii", (100, 100, 100)), "ellipsoid radii")
        angles = _triple(e.get("angles", (0, 0, 0)), "ellipsoid angles")
        ellipsoid = Ellipsoid.from_radii(radii, angles)  # type: ignore[arg-type]
        model = None
        if method == "ok":
            v = spec.get("variogram") or {}
            structures = []
            for s in v.get("structures") or []:
                s_radii = _triple(s.get("ranges") or radii, "structure ranges")
                s_angles = _triple(s.get("angles") or angles, "structure angles")
                structures.append(
                    Structure(s.get("type", "spherical"), float(s["sill"]), s_radii, s_angles)  # type: ignore[arg-type]
                )
            if not structures:
                raise OperationError("kriging needs at least one variogram structure")
            model = VariogramModel(float(v.get("nugget", 0.0)), tuple(structures))
        discretisation = (
            tuple(int(x) for x in spec["discretisation"]) if spec.get("discretisation") else None
        )
        passes = []
        for i, p in enumerate(spec.get("passes") or [{}]):
            per_hole = p.get("max_per_hole")
            passes.append(
                EstimationPass(
                    method,
                    ellipsoid,
                    SearchParams(
                        max_samples=int(p.get("max_samples", 16)),
                        min_samples=int(p.get("min_samples", 1)),
                        max_per_hole=int(per_hole) if per_hole else None,
                    ),
                    radius_factor=float(p.get("radius_factor", 1.0)),
                    power=float(spec.get("power", 2.0)),
                    variogram=model,
                    discretisation=discretisation,  # type: ignore[arg-type]
                    name=str(p.get("name") or f"pass {i + 1}"),
                )
            )
        d = spec.get("domain") or {}
        domain = EstimationDomain(
            d.get("name") or "all",
            passes,
            block_constraint=d.get("block_constraint") or None,
            sample_filter=d.get("sample_filter") or None,
        )
        attribute = spec.get("attribute")
        if not attribute:
            raise OperationError("give the grade attribute")
        return EstimationPlan(attribute, [domain], value_column=spec.get("value_column") or None)
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from None


class Job:
    def __init__(self, kind: str) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.kind = kind
        self.status = "queued"  # queued | running | done | cancelled | error
        self.progress = 0.0
        self.message = ""
        self.summary: dict[str, Any] | None = None
        self.error: str | None = None
        self.cancel_requested = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "status": self.status,
            "progress": self.progress,
            "message": self.message,
            "summary": self.summary,
            "error": self.error,
        }


_jobs: dict[str, Job] = {}
_jobs_lock = threading.Lock()


def get_job(job_id: str) -> Job:
    with _jobs_lock:
        if job_id not in _jobs:
            raise KeyError(job_id)
        return _jobs[job_id]


def run_estimation(
    source: Any,
    model_id: str,
    composites_id: str,
    plan: Any,
    on_progress: Callable[[float, str], None] | None = None,
    cancel: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """``bakumine estimate``: run the plan; the model is saved only when it completed."""
    from bakumine.core.blockmodel.storage import load_model, save_model

    path = _model_path(source.folder, model_id)
    project = Project.open(source.folder)
    try:
        if project.get(composites_id).kind != "composite":
            raise KeyError(composites_id)
        table = _read_composites(project.path_of(composites_id))
    except KeyError:
        raise OperationError(f"no composites {composites_id!r}") from None
    model = load_model(path)
    try:
        summary = plan.run(
            model, table, resolver=project_resolver(source), progress=on_progress, cancel=cancel
        )
    except (ValueError, KeyError, TypeError) as exc:
        raise _fail(exc) from None
    if not summary.cancelled:
        save_model(model, path)
    result = summary.to_dict()
    result.update(
        n_targeted=int(summary.n_targeted),
        n_estimated=int(summary.n_estimated),
        n_unestimated=int(summary.n_unestimated),
    )
    result["table"] = summary.to_frame().replace({np.nan: None}).to_dict(orient="records")
    return result


def start_estimation(source: Any, model_id: str, composites_id: str, spec: dict[str, Any]) -> Job:
    """Validate the request, then run the estimation on a worker thread."""
    plan = build_plan(spec)
    _model_path(source.folder, model_id)
    _claim(model_id)
    job = Job("estimate")
    with _jobs_lock:
        _jobs[job.id] = job

    def work() -> None:
        job.status = "running"

        def progress(fraction: float, message: str) -> None:
            job.progress, job.message = float(fraction), message

        try:
            result = run_estimation(
                source, model_id, composites_id, plan, progress, lambda: job.cancel_requested
            )
            job.summary = result
            job.status = "cancelled" if result.get("cancelled") else "done"
            if job.status == "done":
                job.progress = 1.0
        except Exception as exc:  # reported to the client
            job.error, job.status = str(exc), "error"
        finally:
            _release(model_id)

    threading.Thread(target=work, name=f"estimate-{job.id}", daemon=True).start()
    return job


def cancel_job(job_id: str) -> Job:
    job = get_job(job_id)
    job.cancel_requested = True
    return job
