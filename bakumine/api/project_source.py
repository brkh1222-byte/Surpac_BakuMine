"""A project folder (:mod:`bakumine.core.project`) served as a scene source.

Database objects become drillhole objects, ``dtm`` objects surfaces, ``solid`` objects
solids and ``model`` objects block models. Composites, strings and constraints are in the
project but not drawn. Block models resolve the project's solids and DTMs by object id in
constraints. Data are loaded on first use and kept until the folder changes.
"""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path
from typing import TYPE_CHECKING

from bakumine.api.scene import Category, DrillholeData, Kind, ObjectInfo
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.project import PROJECT_FILE, Project

if TYPE_CHECKING:
    from bakumine.api.blocks import BlockData

#: project kind -> (manifest category, data kind)
_DRAWN: dict[str, tuple[Category, Kind]] = {
    "database": ("drillholes", "drillholes"),
    "dtm": ("surfaces", "mesh"),
    "solid": ("solids", "mesh"),
    "model": ("blockModels", "blocks"),
}


class ProjectSceneSource:
    def __init__(self, folder: str | Path) -> None:
        self.folder = Path(folder)
        self._lock = threading.RLock()
        self._cache: dict[tuple[str, str], object] = {}
        self._loaded_version = ""
        project = Project.open(self.folder)
        self.project_id = project.name
        self.project_name = project.name

    def _project(self) -> Project:
        return Project.open(self.folder)

    @property
    def version(self) -> str:
        digest = hashlib.sha1()
        project = self._project()
        digest.update((self.folder / PROJECT_FILE).read_bytes())
        for obj in project.list():
            path = project.path_of(obj)
            for file in sorted([path, *path.rglob("*")] if path.is_dir() else [path]):
                if file.is_file():
                    stat = file.stat()
                    digest.update(f"{file.name}:{stat.st_size}:{stat.st_mtime_ns};".encode())
        return digest.hexdigest()[:12]

    def objects(self) -> list[ObjectInfo]:
        return [
            ObjectInfo(o.id, o.name, *_DRAWN[o.kind])
            for o in self._project().list()
            if o.kind in _DRAWN
        ]

    # ------------------------------------------------------------------ data

    def _cached(self, kind: str, object_id: str, build):
        version = self.version
        with self._lock:
            if version != self._loaded_version:
                self._cache.clear()
                self._loaded_version = version
            key = (kind, object_id)
            if key not in self._cache:
                self._cache[key] = build()
            return self._cache[key]

    def _path(self, object_id: str, *kinds: str) -> Path:
        project = self._project()
        obj = project.get(object_id)  # KeyError when unknown
        if obj.kind not in kinds:
            raise KeyError(object_id)
        return project.path_of(obj)

    def drillholes(self, object_id: str) -> DrillholeData:
        path = self._path(object_id, "database")

        def build() -> DrillholeData:
            from bakumine.core.drillholes.importer import load_database

            database = load_database(path)
            return DrillholeData(database, database.desurvey())

        return self._cached("drillholes", object_id, build)  # type: ignore[no-any-return]

    def mesh(self, object_id: str) -> Mesh:
        path = self._path(object_id, "dtm", "solid")

        def build() -> Mesh:
            from bakumine.core.io.obj import read_obj

            return read_obj(path)

        return self._cached("mesh", object_id, build)  # type: ignore[no-any-return]

    def blocks(self, object_id: str) -> BlockData:
        path = self._path(object_id, "model")

        def build() -> BlockData:
            from bakumine.api.blocks import BlockData
            from bakumine.core.blockmodel.constraints import DictResolver
            from bakumine.core.blockmodel.storage import load_model
            from bakumine.core.geometry.dtm import DTM

            project = self._project()
            solids = {o.id: (o.name, self.mesh(o.id)) for o in project.list("solid")}
            dtms = {o.id: (o.name, self.mesh(o.id)) for o in project.list("dtm")}
            return BlockData(
                load_model(path),
                DictResolver(
                    dtms={k: DTM(m) for k, (_, m) in dtms.items()},
                    solids={k: m for k, (_, m) in solids.items()},
                ),
                solids={k: n for k, (n, _) in solids.items()},
                surfaces={k: n for k, (n, _) in dtms.items()},
            )

        return self._cached("blocks", object_id, build)  # type: ignore[no-any-return]
