"""Project job log (``jobs.jsonl``) and undo snapshots of block models (P6.T7).

Every operation run on a project through the operations API appends one JSON line to
``<project>/jobs.jsonl``::

    {"id", "time", "operation", "params", "inputs": [{id, kind, name, path}],
     "outputs": [...], "status": "done" | "error" | "cancelled", "duration_s",
     "summary", "error", "modifies": model id | null, "snapshot": bool}

A line is written with a single ``write`` on a file opened for appending, then fsync'd, so
a crash leaves at most one damaged last line, which readers skip. Nothing is rewritten:
undoing a job appends an ``"operation": "undo"`` line whose ``params.job`` names it.

Undo snapshots live in ``<project>/.snapshots/<job id>/``. An operation that only changes
attribute values (estimation) saves the affected columns; one that may change the block
structure (coding with sub-blocks or merging) copies the whole model folder. Only the
newest :data:`KEEP_SNAPSHOTS` snapshots are kept (``BAKUMINE_UNDO_SNAPSHOTS`` overrides).
"""

from __future__ import annotations

import json
import math
import os
import shutil
import threading
import time
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

LOG_FILE = "jobs.jsonl"
SNAPSHOTS = ".snapshots"
KEEP_SNAPSHOTS = 5

_lock = threading.Lock()


def keep_snapshots() -> int:
    try:
        return max(0, int(os.environ.get("BAKUMINE_UNDO_SNAPSHOTS", KEEP_SNAPSHOTS)))
    except ValueError:
        return KEEP_SNAPSHOTS


def new_id() -> str:
    return uuid.uuid4().hex[:12]


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="milliseconds")


def _plain(value: Any) -> Any:
    """JSON-safe copy: numpy scalars to Python, non-finite floats to None."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


# ------------------------------------------------------------------------ the log


def append(folder: Path, entry: dict[str, Any]) -> dict[str, Any]:
    """Append one entry as one line (single write, fsync)."""
    line = (json.dumps(_plain(entry), ensure_ascii=False, allow_nan=False) + "\n").encode()
    with _lock:
        fd = os.open(Path(folder) / LOG_FILE, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)
        try:
            os.write(fd, line)
            os.fsync(fd)
        finally:
            os.close(fd)
    return entry


def read(path: Path) -> list[dict[str, Any]]:
    """Entries of a log file (or of the project folder holding it); damaged lines skipped."""
    path = Path(path)
    if path.is_dir():
        path = path / LOG_FILE
    if not path.is_file():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            entry = json.loads(line)
        except ValueError:
            continue
        if isinstance(entry, dict) and "operation" in entry:
            out.append(entry)
    return out


def undone_ids(entries: Iterable[dict[str, Any]]) -> set[str]:
    return {
        str(e["params"].get("job"))
        for e in entries
        if e.get("operation") == "undo" and e.get("status") == "done"
    }


def jobs(folder: Path) -> list[dict[str, Any]]:
    """Log entries with ``undone`` set, oldest first."""
    entries = read(folder)
    undone = undone_ids(entries)
    return [dict(e, undone=e.get("id") in undone) for e in entries]


def undo_candidate(folder: Path) -> dict[str, Any] | None:
    """The newest model-changing job that is not undone, if its snapshot still exists."""
    for entry in reversed(jobs(folder)):
        if entry.get("operation") == "undo" or entry["undone"]:
            continue
        if entry.get("status") != "done" or not entry.get("modifies"):
            continue
        return entry if snapshot_dir(folder, entry["id"]).is_dir() else None
    return None


class Recorder:
    """Times one operation and appends its entry when it ends."""

    def __init__(
        self, folder: Path, operation: str, params: dict[str, Any], job_id: str | None = None
    ):
        self.folder = Path(folder)
        self.entry: dict[str, Any] = {
            "id": job_id or new_id(),
            "time": now(),
            "operation": operation,
            "params": params,
            "inputs": [],
            "outputs": [],
            "status": "running",
            "duration_s": None,
            "summary": None,
            "error": None,
            "modifies": None,
            "snapshot": False,
        }
        self._start = time.perf_counter()

    @property
    def id(self) -> str:
        return str(self.entry["id"])

    def finish(self, status: str, summary: Any = None, error: str | None = None) -> None:
        self.entry.update(
            status=status,
            duration_s=round(time.perf_counter() - self._start, 6),
            summary=summary,
            error=error,
        )
        if status != "done" and self.entry["snapshot"]:
            drop_snapshot(self.folder, self.id)  # the model was not changed
            self.entry["snapshot"] = False
        append(self.folder, self.entry)


# ------------------------------------------------------------------- snapshots


def snapshot_dir(folder: Path, job_id: str) -> Path:
    return Path(folder) / SNAPSHOTS / job_id


def drop_snapshot(folder: Path, job_id: str) -> None:
    shutil.rmtree(snapshot_dir(folder, job_id), ignore_errors=True)


def take_snapshot(
    folder: Path,
    job_id: str,
    model_id: str,
    model_path: Path,
    columns: Iterable[str] | None = None,
) -> Path:
    """Save what is needed to undo a change of the model at ``model_path``.

    ``columns`` None copies the whole model folder (structure may change); otherwise only
    those attribute columns (the ones that exist) and the attribute list are saved.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq

    from bakumine.core.blockmodel.storage import load_model, read_header

    model_path = Path(model_path)
    target = snapshot_dir(folder, job_id)
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    header = read_header(model_path)
    meta: dict[str, Any] = {
        "job": job_id,
        "model": model_id,
        "model_path": model_path.relative_to(folder).as_posix(),
        "created_ns": time.time_ns(),
        "checksum": header.checksum,
    }
    if columns is None:
        meta["mode"] = "folder"
        shutil.copytree(model_path, target / "model")
    else:
        model = load_model(model_path)
        saved = [c for c in dict.fromkeys(columns) if c in model.attribute_names]
        meta.update(
            mode="columns",
            attributes=[model.attribute(n).to_dict() for n in model.attribute_names],
            columns=saved,
        )
        table = pa.table({n: pa.array(np.asarray(model[n])) for n in saved})
        pq.write_table(table, target / "columns.parquet")
    (target / "meta.json").write_text(json.dumps(_plain(meta), indent=1), encoding="utf-8")
    prune(folder)
    return target


def prune(folder: Path, keep: int | None = None) -> None:
    keep = keep_snapshots() if keep is None else keep
    root = Path(folder) / SNAPSHOTS
    if not root.is_dir():
        return
    found = []
    for d in root.iterdir():
        try:
            created = json.loads((d / "meta.json").read_text(encoding="utf-8"))["created_ns"]
        except (OSError, ValueError, KeyError):
            created = 0
        found.append((created, d))
    found.sort()
    for _, d in found[: max(0, len(found) - keep)]:
        shutil.rmtree(d, ignore_errors=True)


def restore_snapshot(folder: Path, job_id: str) -> dict[str, Any]:
    """Put the model back as it was before job ``job_id``; returns the snapshot's meta.

    Raises ``ValueError`` if the restored model's content checksum differs from the one
    recorded when the snapshot was taken.
    """
    import pyarrow.parquet as pq

    from bakumine.core.blockmodel.attributes import Attribute
    from bakumine.core.blockmodel.storage import load_model, read_header, save_model

    source = snapshot_dir(folder, job_id)
    meta = json.loads((source / "meta.json").read_text(encoding="utf-8"))
    path = Path(folder).joinpath(*meta["model_path"].split("/"))
    if meta["mode"] == "folder":
        tmp = path.parent / f".{path.name}.undo-{job_id}"
        old = path.parent / f".{path.name}.undone-{job_id}"
        shutil.rmtree(tmp, ignore_errors=True)
        shutil.copytree(source / "model", tmp)
        if path.exists():
            os.rename(path, old)
        os.rename(tmp, path)
        shutil.rmtree(old, ignore_errors=True)
    else:
        model = load_model(path)
        specs = {a["name"]: a for a in meta["attributes"]}
        for name in model.attribute_names:
            if name not in specs:
                model.delete_attribute(name)
        table = pq.read_table(source / "columns.parquet")
        for name in meta["columns"]:
            values = table.column(name).to_numpy(zero_copy_only=False)
            spec = Attribute.from_dict(specs[name])
            if name not in model.attribute_names:
                model.add_attribute(name, spec.kind, dtype=spec.dtype)
            model._attributes[name] = spec  # noqa: SLF001 - exact restore of the description
            model._columns[name] = np.ascontiguousarray(values, dtype=spec.dtype)  # noqa: SLF001
        save_model(model, path)
    if read_header(path).checksum != meta["checksum"]:
        raise ValueError(f"undo of {job_id}: the restored model differs from the snapshot")
    drop_snapshot(folder, job_id)
    return meta
