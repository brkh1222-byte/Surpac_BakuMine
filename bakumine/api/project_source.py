"""A project folder (:mod:`bakumine.core.project`) served as a scene source.

Database objects become drillhole objects, ``dtm`` objects surfaces, ``solid`` objects
solids and ``model`` objects block models. Composites, strings and constraints are in the
project but not drawn. A ``geology`` object is listed as itself (category and kind
``geology``) followed by its meshes as ``mesh`` objects that name it as their ``parent``:
``<id>.unit.<CODE>`` (group ``Units``, the unit's colour), ``<id>.fault.<NAME>`` (group
``Faults``) and ``<id>.surface.<stem>`` (group ``Surfaces``, the zone's colour). The unit
dictionaries of the geology objects give the colours of the categories of block and
drillhole attributes (the first geology object wins).

Block models resolve the project's solids and DTMs by object id in constraints, and open
the way :func:`bakumine.api.blocks.default_display` reads them: coloured by the estimated
grade, without the air. The project's solids are all listed as the model's constraint
geometry, so the viewer opens them unticked and see-through: an opaque ore solid would hide
the blocks inside it. Data are loaded on first use and kept until the folder changes.
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
        project = self._project()
        out = [
            ObjectInfo(o.id, o.name, *_DRAWN[o.kind]) for o in project.list() if o.kind in _DRAWN
        ]
        for o in project.list("geology"):
            folder = project.path_of(o)
            if not (folder / "geology.json").is_file():
                continue
            out.append(ObjectInfo(o.id, o.name, "geology", "geology"))
            out.extend(
                ObjectInfo(c["id"], c["name"], "geology", "mesh", c["colour"], c["group"], o.id)
                for c in self._geology_children(o.id, folder)
            )
        return out

    # ------------------------------------------------------------------ geology

    def _geology_children(self, object_id: str, folder: Path) -> list[dict]:
        def build() -> list[dict]:
            from bakumine.api.operations_geology import geology_children

            return geology_children(object_id, folder)

        return self._cached("geology-children", object_id, build)  # type: ignore[no-any-return]

    def _geology_folder(self, object_id: str) -> Path:
        folder = self._path(object_id, "geology")
        if not (folder / "geology.json").is_file():
            raise KeyError(object_id)
        return folder

    def geology(self, object_id: str) -> dict:
        """``bounds`` (the grid box) and ``stats`` of a geology object, for the manifest."""
        folder = self._geology_folder(object_id)

        def build() -> dict:
            import json

            manifest = json.loads((folder / "geology.json").read_text(encoding="utf-8"))
            grid = manifest["grid"]
            lo = [float(v) for v in grid["origin"]]
            hi = [
                o + float(s) * (int(n) - 1)
                for o, s, n in zip(lo, grid["spacing"], grid["shape"], strict=True)
            ]
            return {
                "bounds": {"min": lo, "max": hi},
                "stats": {
                    "units": len([c for c in manifest["layers"]["lith"] if c != "AIR"]),
                    "blocks": len(manifest.get("blocks") or []),
                    "faults": len(manifest.get("faults") or []),
                },
            }

        return self._cached("geology", object_id, build)  # type: ignore[no-any-return]

    def _geology_mesh_file(self, object_id: str) -> Path:
        """File of ``<geology id>.<unit | fault | surface>.<key>``; ids are parsed from the
        right, because an object id may contain dots and a key does not."""
        parts = object_id.rsplit(".", 2)
        if len(parts) != 3 or parts[1] not in ("unit", "fault", "surface"):
            raise KeyError(object_id)
        folder = self._geology_folder(parts[0])
        for child in self._geology_children(parts[0], folder):
            if child["id"] == object_id:
                return folder / child["file"]
        raise KeyError(object_id)

    def category_colours(self) -> dict[str, str]:
        """Colour of unit codes, weathering zones and folded log spellings, from the unit
        dictionaries of the project's geology objects; the first object wins."""

        def build() -> dict[str, str]:
            from bakumine.core.geology.units import UnitDictionary

            out: dict[str, str] = {}
            project = self._project()
            for o in project.list("geology"):
                try:
                    units = UnitDictionary.load(project.path_of(o) / "units.json")
                except (OSError, ValueError):
                    continue
                colours = units.colours()
                for code, colour in colours.items():
                    out.setdefault(code, colour)
                for spelling, code in units.lookup().items():
                    if code in colours:
                        out.setdefault(spelling, colours[code])
            return out

        return self._cached("colours", "", build)  # type: ignore[no-any-return]

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
            return DrillholeData(database, database.desurvey(), colours=self.category_colours())

        return self._cached("drillholes", object_id, build)  # type: ignore[no-any-return]

    def mesh(self, object_id: str) -> Mesh:
        try:
            path = self._path(object_id, "dtm", "solid")
        except KeyError:
            path = self._geology_mesh_file(object_id)

        def build() -> Mesh:
            from bakumine.core.io.obj import read_obj

            return read_obj(path)

        return self._cached("mesh", object_id, build)  # type: ignore[no-any-return]

    def blocks(self, object_id: str) -> BlockData:
        path = self._path(object_id, "model")

        def build() -> BlockData:
            from bakumine.api.blocks import BlockData, default_display
            from bakumine.core.blockmodel.constraints import DictResolver
            from bakumine.core.blockmodel.storage import load_model
            from bakumine.core.geometry.dtm import DTM

            project = self._project()
            solids = {o.id: (o.name, self.mesh(o.id)) for o in project.list("solid")}
            dtms = {o.id: (o.name, self.mesh(o.id)) for o in project.list("dtm")}
            model = load_model(path)
            return BlockData(
                model,
                DictResolver(
                    dtms={k: DTM(m) for k, (_, m) in dtms.items()},
                    solids={k: m for k, (_, m) in solids.items()},
                ),
                solids={k: n for k, (n, _) in solids.items()},
                surfaces={k: n for k, (n, _) in dtms.items()},
                display=default_display(model),
                colours=self.category_colours(),
            )

        return self._cached("blocks", object_id, build)  # type: ignore[no-any-return]
