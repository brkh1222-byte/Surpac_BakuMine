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
rename).
"""

from __future__ import annotations

import json
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

PROJECT_FILE = "project.json"
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
            objects = [ProjectObject(**o) for o in doc["objects"]]
            name = str(doc["name"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ProjectError(f"{file} is damaged: {exc}") from None
        if version != FORMAT_VERSION:
            raise ProjectError(f"unsupported project version {version!r}")
        return cls(folder, name, objects)

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
        stem = source.stem if source.is_file() else source.name
        name = name or stem
        object_id = object_id or self._unique_id(_slug(name))
        if any(o.id == object_id for o in self._objects):
            raise ProjectError(f"object id {object_id!r} already exists")
        suffix = source.suffix if source.is_file() else ""
        relative = f"{KIND_FOLDERS[kind]}/{source.name}"
        if any(o.path == relative for o in self._objects) or self.folder.joinpath(
            *relative.split("/")
        ).exists():
            relative = f"{KIND_FOLDERS[kind]}/{stem}-{object_id}{suffix}"
        target = self.folder.joinpath(*relative.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target)
        else:
            shutil.copy2(source, target)
        obj = ProjectObject(object_id, kind, name, relative)
        self._objects.append(obj)
        self._save()
        return obj

    def remove(self, object_id: str, *, delete_files: bool = True) -> ProjectObject:
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
