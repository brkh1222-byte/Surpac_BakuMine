"""A project is a folder: ``project.json`` plus one subfolder per kind of object.

    project.json            name, version, objects: [{id, kind, name, path}]
    database/               drillhole database folders (``bakumine import`` output)
    composites/             composited interval tables (Parquet / CSV)
    strings/                Surpac-style string files
    dtm/                    open surfaces (OBJ)
    solids/                 closed solids (OBJ)
    models/                 block model folders (``blockmodel.storage``)
    constraints/            constraint JSON files

``path`` in the registry is relative to the project folder and always uses ``/``. Adding
an object copies the file or folder into the project, byte for byte, so data read back are
identical to what went in. ``project.json`` is written atomically (temporary file, fsync,
rename) under an exclusive lock on ``.project.lock``; every change re-reads the
registry inside the lock first, so two handles (processes) adding objects do not lose each
other's entries.

Registry paths and object ids are checked: a path must be relative, use only plain
components (no ``..``) and lie in its kind's subfolder; an id must be a plain token
(letters, digits, ``_``, ``-``, ``.``, not starting with ``.``). A ``project.json`` that
breaks these rules is reported as damaged instead of being followed outside the folder.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_FILE = "project.json"
LOCK_FILE = ".project.lock"
FORMAT_VERSION = 1

#: kind -> subfolder
KIND_FOLDERS: dict[str, str] = {
    "database": "database",
    "composite": "composites",
    "string": "strings",
    "dtm": "dtm",
    "solid": "solids",
    "model": "models",
    "constraint": "constraints",
}
KINDS = tuple(KIND_FOLDERS)


class ProjectError(ValueError):
    """A project folder or request is not valid."""


@dataclass(frozen=True)
class ProjectObject:
    id: str
    kind: str
    name: str
    path: str  # relative to the project folder, "/" separated

    def to_dict(self) -> dict[str, str]:
        return {"id": self.id, "kind": self.kind, "name": self.name, "path": self.path}


_ID = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")
_PART = re.compile(r"[^/\\\x00]+\Z")


def _check_id(object_id: str) -> str:
    if not isinstance(object_id, str) or not _ID.match(object_id) or ".." in object_id:
        raise ProjectError(
            f"invalid object id {object_id!r}: use letters, digits, '_', '-' and '.' only"
        )
    return object_id


def _check_object(obj: ProjectObject) -> ProjectObject:
    """Reject registry entries whose id or path could reach outside the project."""
    _check_id(obj.id)
    if obj.kind not in KIND_FOLDERS:
        raise ProjectError(f"unknown kind {obj.kind!r}")
    parts = obj.path.split("/") if isinstance(obj.path, str) else []
    if (
        len(parts) < 2
        or parts[0] != KIND_FOLDERS[obj.kind]
        or any(p in ("", ".", "..") or not _PART.match(p) for p in parts)
    ):
        raise ProjectError(f"object {obj.id!r} has an invalid path {obj.path!r}")
    return obj


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-") or "object"


class Project:
    """An open project folder. Every change is written to ``project.json`` immediately."""

    def __init__(self, folder: Path, name: str, objects: list[ProjectObject]) -> None:
        self.folder = folder
        self.name = name
        self._objects = objects

    # ------------------------------------------------------------ create / open

    @classmethod
    def create(cls, folder: str | Path, name: str | None = None) -> Project:
        folder = Path(folder)
        if (folder / PROJECT_FILE).exists():
            raise ProjectError(f"{folder} already is a project")
        folder.mkdir(parents=True, exist_ok=True)
        for sub in KIND_FOLDERS.values():
            (folder / sub).mkdir(exist_ok=True)
        project = cls(folder, name or folder.resolve().name, [])
        project._save()
        return project

    @classmethod
    def open(cls, folder: str | Path) -> Project:
        folder = Path(folder)
        file = folder / PROJECT_FILE
        if not file.is_file():
            raise ProjectError(f"{folder} is not a project (no {PROJECT_FILE})")
        try:
            doc = json.loads(file.read_text(encoding="utf-8"))
            version = doc["version"]
            name = str(doc["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"{file} is damaged: {exc}") from None
        if version != FORMAT_VERSION:
            raise ProjectError(f"unsupported project version {version!r}")
        return cls(folder, name, cls._read_objects(file, doc))

    @staticmethod
    def _read_objects(file: Path, doc: Any = None) -> list[ProjectObject]:
        try:
            if doc is None:
                doc = json.loads(file.read_text(encoding="utf-8"))
            objects = [_check_object(ProjectObject(**o)) for o in doc["objects"]]
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"{file} is damaged: {exc}") from None
        return objects

    @contextlib.contextmanager
    def _locked(self):
        """Exclusive lock on the registry; reloads the objects from disk on entry."""
        with open(self.folder / LOCK_FILE, "a+") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                file = self.folder / PROJECT_FILE
                if file.is_file():
                    self._objects = self._read_objects(file)
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    # ------------------------------------------------------------ objects

    def list(self, kind: str | None = None) -> list[ProjectObject]:
        return [o for o in self._objects if kind is None or o.kind == kind]

    def get(self, object_id: str) -> ProjectObject:
        for o in self._objects:
            if o.id == object_id:
                return o
        raise KeyError(object_id)

    def path_of(self, obj: ProjectObject | str) -> Path:
        if isinstance(obj, str):
            obj = self.get(obj)
        _check_object(obj)
        return self.folder.joinpath(*obj.path.split("/"))

    def add(
        self, kind: str, source: str | Path, name: str | None = None, object_id: str | None = None
    ) -> ProjectObject:
        """Copy the file or folder ``source`` into the project's ``kind`` subfolder."""
        if kind not in KIND_FOLDERS:
            raise ProjectError(f"unknown kind {kind!r}; choose from {', '.join(KINDS)}")
        source = Path(source)
        if not source.exists():
            raise FileNotFoundError(f"{source} does not exist")
        if object_id is not None:
            _check_id(object_id)
        stem = source.stem if source.is_file() else source.name
        name = name or stem
        suffix = source.suffix if source.is_file() else ""
        with self._locked():
            object_id = object_id or self._unique_id(_slug(name))
            if any(o.id == object_id for o in self._objects):
                raise ProjectError(f"object id {object_id!r} already exists")
            relative = f"{KIND_FOLDERS[kind]}/{source.name}"
            if (
                any(o.path == relative for o in self._objects)
                or self.folder.joinpath(*relative.split("/")).exists()
            ):
                relative = f"{KIND_FOLDERS[kind]}/{stem}-{object_id}{suffix}"
            obj = _check_object(ProjectObject(object_id, kind, name, relative))
            target = self.folder.joinpath(*relative.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            if source.is_dir():
                shutil.copytree(source, target)
            else:
                shutil.copy2(source, target)
            self._objects.append(obj)
            self._save()
        return obj

    def remove(self, object_id: str, *, delete_files: bool = True) -> ProjectObject:
        with self._locked():
            obj = self.get(object_id)
            self._objects.remove(obj)
            self._save()
        if delete_files:
            path = self.path_of(obj)
            if path.is_dir():
                shutil.rmtree(path)
            elif path.exists():
                path.unlink()
        return obj

    def _unique_id(self, base: str) -> str:
        ids = {o.id for o in self._objects}
        candidate, n = base, 2
        while candidate in ids:
            candidate, n = f"{base}-{n}", n + 1
        return candidate

    # ------------------------------------------------------------ persistence

    def _save(self) -> None:
        doc: dict[str, Any] = {
            "name": self.name,
            "version": FORMAT_VERSION,
            "objects": [o.to_dict() for o in self._objects],
        }
        fd, tmp = tempfile.mkstemp(prefix=".project-", suffix=".tmp", dir=self.folder)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps(doc, indent=2, ensure_ascii=False) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, self.folder / PROJECT_FILE)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
