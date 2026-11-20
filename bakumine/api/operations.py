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

from bakumine.core import joblog
from bakumine.core.project import Project, _slug

UPLOADS = "uploads"
#: uploaded files an import used are copied here (``inputs/<job id>/``) for replay
INPUTS = "inputs"
MAX_ISSUES = 500
PREVIEW_ROWS = 8
#: limits on what one request may ask for (memory and time of a local server)
MAX_MODEL_BLOCKS = 100_000_000  # parent blocks of a new model (about 1.2 GB)
MAX_DISCRETISATION = 1000  # points per block (nx * ny * nz)
MAX_SAMPLES = 1000  # samples per estimate
MAX_RUNNING_JOBS = 4
MAX_FINISHED_JOBS = 100  # older finished jobs are forgotten


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


#: file types the import accepts (CSVs from `bakumine extract-tables` included)
UPLOAD_SUFFIXES = (".csv", ".txt", ".tsv", ".xlsx", ".xlsm", ".xls")
#: rows / columns of the raw sheet shown for picking the header row
GRID_ROWS, GRID_COLUMNS = 14, 40
MAX_SHEETS_LISTED = 400
#: keys of a source of an import recipe a request may set (never ``file``)
_SPEC_KEYS = frozenset(
    {
        "sheets", "exclude_sheets", "header_row", "header_rows", "map", "hole_id_from_sheet",
        "hole_id", "sample_depth", "qc", "options",
    }
)  # fmt: skip


def _table_files(folder: Path, name: str) -> list[Path]:
    """The uploaded files of a table: ``<name>.<ext>``, then ``<name>+2.<ext>``, ..."""
    uploads = folder / UPLOADS
    first = sorted(p for p in uploads.glob(f"{name}.*") if p.is_file())[:1]

    def number(path: Path) -> int:
        tail = path.stem.rpartition("+")[2]
        return int(tail) if tail.isdigit() else 0

    more = [p for p in uploads.glob(f"{name}+*.*") if p.is_file() and number(p) > 0]
    return first + sorted(more, key=number)


def _upload_path(folder: Path, name: str) -> Path | None:
    files = _table_files(folder, name)
    return files[0] if files else None


def _uploaded_tables(folder: Path) -> list[str]:
    uploads = folder / UPLOADS
    if not uploads.is_dir():
        return []
    return sorted({p.stem.partition("+")[0] for p in uploads.glob("*.*") if p.is_file()})


def _check_upload_name(name: str) -> None:
    """Table names are also glob patterns of the uploads folder: no wildcards."""
    from bakumine.core.drillholes.importer import _check_table_name

    if name not in ("collar", "survey"):
        try:
            _check_table_name(name)
        except ValueError as exc:
            raise _fail(exc) from None


def store_upload(
    folder: Path, name: str, filename: str, content: bytes, add: bool = False
) -> dict[str, Any]:
    """Keep an uploaded file for a table. ``add`` appends it as one more source of the
    table (merged on import) instead of replacing what was uploaded for it."""
    _check_upload_name(name)
    suffix = Path(filename).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        raise OperationError(f"{filename!r}: use a CSV, XLSX or XLS file")
    target = folder / UPLOADS
    target.mkdir(exist_ok=True)
    existing = _table_files(folder, name)
    if add and existing:
        index = len(existing)
        path = target / f"{name}+{index + 1}{suffix}"
    else:
        for old in existing:
            old.unlink()
        index, path = 0, target / f"{name}{suffix}"
    path.write_bytes(content)
    try:
        return preview_upload(folder, name, index=index)
    except OperationError:
        path.unlink(missing_ok=True)  # an unreadable file is not kept
        raise


def remove_upload(folder: Path, name: str) -> None:
    _check_upload_name(name)
    for old in _table_files(folder, name):
        old.unlink()


_grid_cache: dict[tuple[str, int, int, str | None], dict[str, pd.DataFrame]] = {}
_grid_lock = threading.Lock()


def _grids(path: Path, delimiter: str | None) -> dict[str, pd.DataFrame]:
    """The sheets of an uploaded file; the last few files read are remembered, so that
    moving the header row of a large workbook does not read it again."""
    from bakumine.core.drillholes.tables import read_grids

    stat = path.stat()
    key = (str(path), stat.st_mtime_ns, stat.st_size, delimiter)
    with _grid_lock:
        if key in _grid_cache:
            return _grid_cache[key]
    grids = read_grids(path, delimiter=delimiter)
    with _grid_lock:
        _grid_cache[key] = grids
        while len(_grid_cache) > 4:
            _grid_cache.pop(next(iter(_grid_cache)))
    return grids


def _clean_spec(spec: Any) -> dict[str, Any]:
    if spec is None:
        return {}
    if not isinstance(spec, dict):
        raise OperationError("a source description must be an object")
    unknown = sorted(set(spec) - _SPEC_KEYS - {"preview_sheet"})
    if unknown:
        raise OperationError(f"unknown source option(s) {unknown}")
    return {k: v for k, v in spec.items() if v is not None and v != "" and v != []}


def preview_upload(
    folder: Path,
    name: str,
    delimiter: str | None = ",",
    index: int = 0,
    spec: dict[str, Any] | None = None,
    hole_id: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """What an uploaded file holds and what the import would make of it with ``spec``
    (the keys of a recipe source: sheets, header row, column map, sample-number rule, QC)."""
    from bakumine.core.drillholes.importer import missing_required
    from bakumine.core.drillholes.tables import locate_header, prepare_source, table_from_grid

    files = _table_files(folder, name)
    if not 0 <= index < len(files):
        raise OperationError(f"no uploaded file for table {name!r}")
    path = files[index]
    spec = _clean_spec(spec)
    shown = spec.pop("preview_sheet", None)
    kind = _table_kind(name)
    try:
        grids = _grids(path, delimiter)
        spec.pop("options", None)  # units, dips, axes: applied at import, not to the text
        prepared = prepare_source(spec, kind, name=name, grids=grids, hole_id=hole_id)
        chosen = prepared.info["sheets"]
        shown = shown if shown in grids else chosen[0]["name"]
        grid = grids[shown]
        used = next((s for s in chosen if s["name"] == shown), None)
        if used is not None:
            header = locate_header(grid, used["header_row"] - 1, used["header_rows"] or None)
        else:
            header = locate_header(grid)
        raw, _ = table_from_grid(grid, header)
    except Exception as exc:  # unreadable file, bad pattern, missing column
        raise _fail(exc) from None
    targets = prepared.info["targets"]
    sample_depth = prepared.info.get("sample_depth")
    if sample_depth:
        sample_depth = {**sample_depth, "holes": sample_depth["holes"][:60]}
    return {
        "name": name,
        "kind": kind,
        "index": index,
        "n_files": len(files),
        "file": path.name,
        "rows": prepared.info["rows"],
        "columns": [str(c) for c in prepared.info["columns"]],
        "preview": raw.head(PREVIEW_ROWS).astype(str).to_dict(orient="records"),
        "mapping": dict(prepared.mapping),
        "missing": missing_required(kind, targets),
        "unmatched": [c for c in prepared.info["columns"] if c not in prepared.mapping],
        "sheets": [
            {"name": sheet, "rows": int(len(g))}
            for sheet, g in list(grids.items())[:MAX_SHEETS_LISTED]
        ],
        "selected": chosen,
        "grid_sheet": shown,
        "grid_header_row": header.row + 1,
        "grid_header_rows": header.n_rows,
        "grid": grid.iloc[:GRID_ROWS, :GRID_COLUMNS].astype(str).to_numpy().tolist(),
        "prepared": {
            "columns": [str(c) for c in prepared.table.columns],
            "rows": prepared.table.head(PREVIEW_ROWS).astype(str).to_numpy().tolist(),
        },
        "qc": prepared.info.get("qc") or {},
        "sample_depth": sample_depth,
        "notes": [_note_text(n) for n in prepared.notes[:20]],
    }


def _note_text(note: dict[str, Any]) -> str:
    text = f"{note['table']}.{note['column']}: {note['kind']} x {note['count']}"
    values = note.get("values") or {}
    if values:
        text += " (" + ", ".join(f"{k} x {v}" for k, v in list(values.items())[:6]) + ")"
    elif note.get("examples"):
        text += " (e.g. " + ", ".join(str(e) for e in note["examples"][:3]) + ")"
    return text


def list_uploads(folder: Path) -> list[dict[str, Any]]:
    out = []
    for name in _uploaded_tables(folder):
        try:
            out.append(preview_upload(folder, name))
        except OperationError:
            continue  # a file that can no longer be read does not block the project
    return out


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
    return _record(
        folder, "validate", {"database": object_id}, lambda rec: _validate(folder, object_id, rec)
    )


def _validate(folder: Path, object_id: str, rec: joblog.Recorder) -> dict[str, Any]:
    from bakumine.core.drillholes.importer import load_database
    from bakumine.core.drillholes.validate import validate

    project = Project.open(folder)
    try:
        path = project.path_of(object_id)
        if project.get(object_id).kind != "database":
            raise KeyError(object_id)
    except KeyError:
        raise OperationError(f"no drillhole database {object_id!r}") from None
    rec.entry["inputs"] = [project.get(object_id).to_dict()]
    # CLI defaults: max_deviation 10 deg / 30 m, depth tolerance 1e-6
    report = validate(load_database(path), max_deviation_deg_per_30m=10.0, depth_tolerance=1e-6)
    report.findings.to_csv(path / "validation_findings.csv", index=False)
    return validation_dict(report)


def import_database(
    folder: Path,
    name: str,
    maps: dict[str, dict[str, str]] | None = None,
    options: dict[str, Any] | None = None,
    tables: dict[str, list[dict[str, Any] | None]] | None = None,
    hole_id: dict[str, Any] | None = None,
    merge: dict[str, Any] | None = None,
    holes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``bakumine import`` + ``validate`` on the uploaded tables; saves the database.

    ``tables`` gives, per table, one recipe source per uploaded file (sheets, header row,
    column map, sample-number rule, QC; see ``bakumine.core.drillholes.recipe``), and then
    only the tables it names are imported (files uploaded earlier for other tables are left
    alone); ``maps`` is the short form: one column map per table, every uploaded table is
    imported. What is left out is detected. ``holes`` = ``{"include": [...], "exclude":
    [...]}``: hole names or wildcard patterns kept / left out in every table."""
    params = {"name": name, "maps": maps, "options": options}
    for key, value in (
        ("tables", tables), ("hole_id", hole_id), ("merge", merge), ("holes", holes),
    ):  # fmt: skip
        if value:
            params[key] = value
    return _record(
        folder,
        "import",
        params,
        lambda rec: _import(folder, name, maps, options, tables, hole_id, merge, rec, holes),
    )


def _import(
    folder: Path,
    name: str,
    maps: dict[str, dict[str, str]] | None,
    options: dict[str, Any] | None,
    tables: dict[str, list[dict[str, Any] | None]] | None,
    hole_id: dict[str, Any] | None,
    merge: dict[str, Any] | None,
    rec: joblog.Recorder,
    holes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    import shutil

    from bakumine.core.drillholes.recipe import (
        MissingColumns,
        make_recipe,
        options_from_dict,
        run_recipe,
    )
    from bakumine.core.drillholes.validate import validate

    for what, value in (("maps", maps), ("options", options), ("tables", tables),
                        ("hole_id", hole_id), ("merge", merge), ("holes", holes)):  # fmt: skip
        if not isinstance(value or {}, dict):
            raise OperationError(f"{what} must be an object")
    maps, tables = dict(maps or {}), dict(tables or {})
    uploads = {table: _table_files(folder, table) for table in _uploaded_tables(folder)}
    if tables:  # the dialog says which tables it shows; older uploads are not its business
        absent = sorted(set(tables) - set(uploads))
        if absent:
            raise OperationError(f"no uploaded file for table(s) {absent}")
        uploads = {table: files for table, files in uploads.items() if table in tables}
    if "collar" not in uploads:
        raise OperationError("upload the collar table first")
    # keep the files this import reads, so that the job log can be replayed
    kept = folder / INPUTS / rec.id
    kept.mkdir(parents=True, exist_ok=True)
    sources: dict[str, list[dict[str, Any]]] = {}
    for table, files in uploads.items():
        specs = tables.get(table) or []
        if not isinstance(specs, list):
            raise OperationError(f"tables.{table} must be a list with one entry per file")
        sources[table] = []
        for i, path in enumerate(files):
            copied = Path(shutil.copy2(path, kept / path.name))
            source = _clean_spec(specs[i] if i < len(specs) else None)
            source.pop("preview_sheet", None)
            if not source.get("map") and maps.get(table):
                source["map"] = maps[table]
            source["file"] = copied.relative_to(folder).as_posix()
            sources[table].append(source)
    try:
        recipe = make_recipe(
            sources, options_from_dict(options), hole_id=hole_id, merge=merge, holes=holes
        )
        db = run_recipe(recipe, folder)
        resolved = db.metadata.recipe
        rec.entry["replay"] = {
            "files": {t: s[0]["file"] for t, s in resolved["tables"].items()},
            "maps": {t: dict(s[0].get("map") or {}) for t, s in resolved["tables"].items()},
            "recipe": resolved,
        }
        report = validate(db, max_deviation_deg_per_30m=10.0, depth_tolerance=1e-6)
        with tempfile.TemporaryDirectory() as tmp:
            saved = db.save(Path(tmp) / "database")
            report.findings.to_csv(saved / "validation_findings.csv", index=False)
            obj = Project.open(folder).add("database", saved, name or "drillholes")
            rec.entry["outputs"] = [obj.to_dict()]
    except OperationError:
        raise
    except MissingColumns as exc:
        raise OperationError(
            f"no column found for {exc.missing} in the {exc.table} table"
        ) from None
    except (ValueError, KeyError, TypeError, FileNotFoundError) as exc:
        raise _fail(exc) from None
    result = validation_dict(report)
    reports = db.metadata.reports
    result.update(
        {
            "id": obj.id,
            "name": obj.name,
            "tables": {n: int(len(t)) for n, t in db.tables.items()},
            "notes": [_note_text(n) for n in db.metadata.notes[:40]],
            "qc": reports.get("qc", {}),
            "conflicts": {
                t: {**c, "rows": c["rows"][:50]} for t, c in reports.get("conflicts", {}).items()
            },
            "coordinates": reports.get("coordinates"),
            "dropped_rows": reports.get("dropped_rows", {}),
            "hole_filter": reports.get("hole_filter"),
            "sources": reports.get("sources", {}),
            "units": reports.get("units", {}),
            "survey": reports.get("survey"),
            "sample_depth": {
                t: [{k: v for k, v in c.items() if k not in ("holes", "preview")} for c in checks]
                for t, checks in reports.get("sample_depth", {}).items()
            },
            "dip_convention": db.dip_convention,
            "coordinate_order": db.metadata.coordinate_order,
            "recipe": resolved,
        }
    )
    return result


# ------------------------------------------------------------------ job log (P6.T7)

#: result keys left out of the log summary (long lists the dialogs show)
_LONG = {
    "issues", "counts", "preview", "table", "notes", "columns", "steps", "rows",
    "recipe", "conflicts", "coordinates", "sample_depth", "survey", "dropped_rows", "units",
    "hole_filter", "sources",
}  # fmt: skip


def _brief(result: Any) -> Any:
    if not isinstance(result, dict):
        return None
    return {k: v for k, v in result.items() if k not in _LONG}


def _record(
    folder: Path, operation: str, params: dict[str, Any], work: Callable[[joblog.Recorder], Any]
) -> Any:
    """Run ``work`` and append its job-log entry (also when it fails)."""
    rec = joblog.Recorder(folder, operation, params)
    try:
        result = work(rec)
    except Exception as exc:
        rec.finish("error", error=str(_fail(exc) if not isinstance(exc, OperationError) else exc))
        raise
    rec.finish("done", summary=_brief(result))
    return result


def _geometry_objects(folder: Path) -> list[dict[str, str]]:
    """The solids and DTMs a project resolver offers (constraints may name any of them)."""
    project = Project.open(folder)
    return [o.to_dict() for o in project.list() if o.kind in ("solid", "dtm")]


def _will_modify(
    rec: joblog.Recorder, folder: Path, model_id: str, path: Path, columns: list[str] | None
) -> None:
    """Note the model the job changes and snapshot it for undo."""
    obj = Project.open(folder).get(model_id).to_dict()
    rec.entry.update(modifies=model_id, outputs=[obj])
    rec.entry["inputs"] = [obj, *_geometry_objects(folder)]
    if joblog.keep_snapshots() > 0:
        joblog.take_snapshot(folder, rec.id, model_id, path, columns)
        rec.entry["snapshot"] = True


def _coded_columns(request: dict[str, Any]) -> list[str] | None:
    """Columns coding changes, or None when it may change the block structure."""
    steps = request.get("steps") or []
    if request.get("subblock") or request.get("merge") or any(s.get("subblock") for s in steps):
        return None
    names = [a.get("name") for a in request.get("attributes") or []]
    return [str(n) for n in [*names, *(s.get("attribute") for s in steps)] if n]


def job_log(folder: Path) -> dict[str, Any]:
    """The log, newest first, and the job an undo would revert."""
    candidate = joblog.undo_candidate(folder)
    return {
        "jobs": list(reversed(joblog.jobs(folder))),
        "undo": candidate["id"] if candidate else None,
        "keep_snapshots": joblog.keep_snapshots(),
    }


def undo_last(folder: Path) -> dict[str, Any]:
    """Restore the snapshot of the newest model-changing job and mark it undone."""
    target = joblog.undo_candidate(folder)
    if target is None:
        raise BusyError("nothing to undo")
    model_id = str(target["modifies"])
    _claim(model_id)
    rec = joblog.Recorder(folder, "undo", {"job": target["id"]})
    try:
        joblog.restore_snapshot(folder, target["id"])
    except Exception as exc:
        rec.finish("error", error=str(exc))
        raise OperationError(str(exc)) from None
    finally:
        _release(model_id)
    rec.entry["outputs"] = [Project.open(folder).get(model_id).to_dict()]
    result = {"undone": target["id"], "operation": target["operation"], "model": model_id}
    rec.finish("done", summary=result)
    return result


def record_report(folder: Path, model_id: str, params: dict[str, Any], work: Callable[[], Any]):
    """Run a report (``work`` returns its table) and log it."""

    def run(rec: joblog.Recorder) -> Any:
        try:
            rec.entry["inputs"] = [
                Project.open(folder).get(model_id).to_dict(),
                *_geometry_objects(folder),
            ]
        except KeyError:
            pass
        return work()

    rec = joblog.Recorder(folder, "report", {"model": model_id, **params})
    try:
        table = run(rec)
    except Exception as exc:
        rec.finish("error", error=str(getattr(exc, "detail", exc)))
        raise
    rec.finish("done", summary={"rows": int(len(table)), "columns": int(len(table.columns))})
    return table


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
    return _record(folder, "composite", request, lambda rec: _composite(folder, request, rec))


def _composite(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
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
    rec.entry["inputs"] = [db_obj.to_dict()]
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
    name = str(request.get("name") or f"composites_{table}")
    with tempfile.TemporaryDirectory() as tmp:
        # The name is display text: the file name is a slug of it, never a path.
        file = Path(tmp) / f"{_slug(name)}.parquet"
        out.to_parquet(file)
        obj = Project.open(folder).add("composite", file, name)
    rec.entry["outputs"] = [obj.to_dict()]
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
    return _record(folder, "create_model", request, lambda rec: _create(folder, request, rec))


def _create(folder: Path, request: dict[str, Any], rec: joblog.Recorder) -> dict[str, Any]:
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
                rec.entry["inputs"].append(obj.to_dict())
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
        if definition.n_parents > MAX_MODEL_BLOCKS:
            raise OperationError(
                f"{definition.n_parents:,} blocks: at most {MAX_MODEL_BLOCKS:,} per model"
            )
        model = BlockModel(definition)
        with tempfile.TemporaryDirectory() as tmp:
            header = save_model(model, Path(tmp) / "model")
            obj = project.add("model", Path(tmp) / "model", request.get("name") or "model")
            rec.entry["outputs"] = [obj.to_dict()]
    except OperationError:
        raise
    except (ValueError, KeyError, TypeError, OverflowError) as exc:
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
    params = {"model": model_id, **request}
    return _record(
        source.folder, "code", params, lambda rec: _code(source, model_id, request, rec)
    )


def _code(source: Any, model_id: str, request: dict[str, Any], rec: joblog.Recorder) -> dict:
    from bakumine.core.blockmodel.coding import code_model
    from bakumine.core.blockmodel.constraints import constraint_from_dict
    from bakumine.core.blockmodel.storage import load_model, save_model

    path = _model_path(source.folder, model_id)
    _claim(model_id)
    try:
        _will_modify(rec, source.folder, model_id, path, _coded_columns(request))
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


#: nested variogram structures of one model (besides the nugget)
MAX_STRUCTURES = 3


def _orientation(item: dict[str, Any], default: Any, what: str) -> tuple[float, ...]:
    """Angles of ``item``: ``angles`` [bearing, plunge, dip], or ``plane`` {dip, dip_azimuth,
    pitch} (see :mod:`bakumine.core.estimation.orientation`); else ``default``."""
    from bakumine.core.estimation.orientation import angles_from_plane

    plane = item.get("plane")
    if plane:
        if item.get("angles"):
            raise OperationError(f"{what}: give angles or a plane, not both")
        try:
            return angles_from_plane(
                float(plane["dip"]), float(plane["dip_azimuth"]), float(plane.get("pitch") or 0.0)
            )
        except KeyError as exc:
            raise OperationError(f"{what} plane needs {exc.args[0]}") from None
    return _triple(item.get("angles") or default, f"{what} angles")


def build_plan(spec: dict[str, Any]) -> Any:
    """An :class:`EstimationPlan` from the dialog's request.

    ``spec``: attribute, value_column, method (nn / idw / ok), power, ellipsoid {radii
    [major, semi, minor], angles [bearing, plunge, dip] or plane {dip, dip_azimuth, pitch?}},
    variogram {nugget, structures [{type, sill, ranges?, angles? or plane?}]} (up to three
    structures; ranges and orientation default to the ellipsoid's), domain {name,
    block_constraint, sample_filter}, passes [{name, radius_factor, min_samples,
    max_samples, max_per_hole}], discretisation.
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
        angles = _orientation(e, (0, 0, 0), "ellipsoid")
        ellipsoid = Ellipsoid.from_radii(radii, angles)  # type: ignore[arg-type]
        model = None
        if method == "ok":
            v = spec.get("variogram") or {}
            structures = []
            if len(v.get("structures") or []) > MAX_STRUCTURES:
                raise OperationError(f"at most {MAX_STRUCTURES} variogram structures")
            for s in v.get("structures") or []:
                s_radii = _triple(s.get("ranges") or radii, "structure ranges")
                s_angles = _orientation(s, angles, "structure")
                structures.append(
                    Structure(s.get("type", "spherical"), float(s["sill"]), s_radii, s_angles)  # type: ignore[arg-type]
                )
            if not structures:
                raise OperationError("kriging needs at least one variogram structure")
            model = VariogramModel(float(v.get("nugget", 0.0)), tuple(structures))
        discretisation = (
            tuple(int(x) for x in spec["discretisation"]) if spec.get("discretisation") else None
        )
        if discretisation is not None and np.prod(discretisation) > MAX_DISCRETISATION:
            raise OperationError(f"discretisation: at most {MAX_DISCRETISATION} points per block")
        passes = []
        for i, p in enumerate(spec.get("passes") or [{}]):
            per_hole = p.get("max_per_hole")
            if int(p.get("max_samples", 16)) > MAX_SAMPLES:
                raise OperationError(f"max_samples: at most {MAX_SAMPLES}")
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
    # AttributeError: a list or a number where the plan has an object (``.get``)
    except (ValueError, KeyError, TypeError, OverflowError, AttributeError) as exc:
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


def _estimated_columns(plan: Any) -> list[str]:
    return [plan.attribute, *plan.outputs.names(plan.attribute).values()]


def start_estimation(source: Any, model_id: str, composites_id: str, spec: dict[str, Any]) -> Job:
    """Validate the request, then run the estimation on a worker thread."""
    plan = build_plan(spec)
    _model_path(source.folder, model_id)
    project = Project.open(source.folder)
    try:
        if project.get(composites_id).kind != "composite":
            raise KeyError(composites_id)
    except KeyError:
        raise OperationError(f"no composites {composites_id!r}") from None
    job = Job("estimate")
    with _jobs_lock:
        running = sum(j.status in ("queued", "running") for j in _jobs.values())
        if running >= MAX_RUNNING_JOBS:
            raise BusyError(f"{running} jobs are running; wait for one to finish")
        _claim(model_id)
        finished = [k for k, j in _jobs.items() if j.status not in ("queued", "running")]
        for key in finished[: max(0, len(finished) - MAX_FINISHED_JOBS + 1)]:
            del _jobs[key]
        _jobs[job.id] = job
    folder = source.folder
    rec = joblog.Recorder(
        folder, "estimate", {"model": model_id, "composites": composites_id, "plan": spec}, job.id
    )
    rec.entry["replay"] = {"plan": plan.to_dict()}

    def work() -> None:
        job.status = "running"

        def progress(fraction: float, message: str) -> None:
            job.progress, job.message = float(fraction), message

        try:
            path = _model_path(folder, model_id)
            _will_modify(rec, folder, model_id, path, _estimated_columns(plan))
            try:
                rec.entry["inputs"].insert(1, Project.open(folder).get(composites_id).to_dict())
            except KeyError:
                pass
            result = run_estimation(
                source, model_id, composites_id, plan, progress, lambda: job.cancel_requested
            )
            job.summary = result
            job.status = "cancelled" if result.get("cancelled") else "done"
            if job.status == "done":
                job.progress = 1.0
            rec.finish(job.status, summary=_brief(result))
        except Exception as exc:  # reported to the client
            job.error, job.status = str(exc), "error"
            rec.finish("error", error=str(exc))
        finally:
            _release(model_id)

    threading.Thread(target=work, name=f"estimate-{job.id}", daemon=True).start()
    return job


def cancel_job(job_id: str) -> Job:
    job = get_job(job_id)
    job.cancel_requested = True
    return job
