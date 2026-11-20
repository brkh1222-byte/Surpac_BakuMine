"""Scene sources: where the viewer's objects come from.

A :class:`SceneSource` lists drawable objects and hands out the core data behind them
(a drillhole database with its desurveyed traces, a triangle mesh, a block model). The endpoints in
``routes_scene`` turn that into the manifest and the binary payloads; they know nothing
about folders or files. Today the only source is :class:`DemoSceneSource`, built from the
synthetic benchmark deposit. The project system (P6.T1) replaces it by implementing the
same protocol and calling :func:`set_scene_source`.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import numpy as np

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.drillholes.importer import (
    DrillholeDatabase,
    TableSource,
    import_drillholes,
    read_raw_table,
    suggest_column_map,
)
from bakumine.core.geometry.mesh import Mesh

if TYPE_CHECKING:
    from bakumine.api.blocks import BlockData

Category = Literal["drillholes", "surfaces", "solids", "blockModels"]
Kind = Literal["drillholes", "mesh", "blocks"]

#: Default demo folder: ``data/synthetic/default`` next to the package (source checkout).
DEMO_FOLDER = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
#: Environment variable that points the demo source at another folder.
DEMO_FOLDER_ENV = "BAKUMINE_DEMO_DIR"


@dataclass(frozen=True)
class ObjectInfo:
    """One entry of the object tree. ``kind`` selects the data endpoint."""

    id: str
    name: str
    category: Category
    kind: Kind


@dataclass(frozen=True)
class DrillholeData:
    """A drillhole database, its desurveyed traces and display hints."""

    database: DrillholeDatabase
    traces: HoleTraces
    #: attribute column -> unit label ("g/t", "%"), where known
    units: dict[str, str] = field(default_factory=dict)


class SceneSource(Protocol):
    """What the scene endpoints need from a project."""

    project_id: str
    project_name: str

    @property
    def version(self) -> str:
        """Changes whenever any object's data change (used for cache keys and ETags)."""
        ...

    def objects(self) -> list[ObjectInfo]: ...

    def drillholes(self, object_id: str) -> DrillholeData:
        """Raises ``KeyError`` when ``object_id`` is not a drillhole object."""
        ...

    def mesh(self, object_id: str) -> Mesh:
        """Raises ``KeyError`` when ``object_id`` is not a mesh object."""
        ...

    def blocks(self, object_id: str) -> BlockData:
        """Raises ``KeyError`` when ``object_id`` is not a block model."""
        ...


class EmptySceneSource:
    """A project with nothing in it (no demo data on disk)."""

    project_id = "empty"
    project_name = "Empty project"
    version = "0"

    def objects(self) -> list[ObjectInfo]:
        return []

    def drillholes(self, object_id: str) -> DrillholeData:
        raise KeyError(object_id)

    def mesh(self, object_id: str) -> Mesh:
        raise KeyError(object_id)

    def blocks(self, object_id: str) -> BlockData:
        raise KeyError(object_id)


class DemoSceneSource:
    """The synthetic benchmark deposit (``bakumine synth`` output) as a scene.

    Objects: ``drillholes`` (collar/survey/assay/geology CSVs, imported with
    ``import_drillholes`` and desurveyed by minimum curvature), ``topo`` (open DTM) and
    ``ore-solid`` (closed solid) and ``block-model``: the truth grid as a block model,
    sub-blocked two levels along the ore contact (``bakumine.api.blocks``). Everything is
    loaded on first use and kept in memory.
    """

    project_id = "demo"
    project_name = "Synthetic demo deposit"

    _MESHES = {
        "topo": ("Topography", "surfaces", "topo"),
        "ore-solid": ("Ore solid", "solids", "ore_solid"),
    }
    _CSV = ("collar", "survey", "assay", "geology")
    _BLOCKS = ("block-model", "Demo block model")
    _BLOCK_FILES = ("truth_blocks.parquet", "truth.json")

    def __init__(self, folder: str | Path = DEMO_FOLDER) -> None:
        self.folder = Path(folder)
        self._lock = threading.Lock()
        self._drillholes: DrillholeData | None = None
        self._meshes: dict[str, Mesh] = {}
        self._blocks: BlockData | None = None

    # ------------------------------------------------------------------ what exists

    def _has_drillholes(self) -> bool:
        return all((self.folder / f"{name}.csv").is_file() for name in self._CSV)

    def _mesh_files(self, stem: str) -> tuple[Path, Path]:
        return self.folder / f"{stem}_vertices.npy", self.folder / f"{stem}_faces.npy"

    def _files(self) -> list[Path]:
        files = [self.folder / f"{name}.csv" for name in self._CSV]
        for _, _, stem in self._MESHES.values():
            files.extend(self._mesh_files(stem))
        files.extend(self.folder / name for name in self._BLOCK_FILES)
        return [f for f in files if f.is_file()]

    def _has_blocks(self) -> bool:
        return all((self.folder / name).is_file() for name in self._BLOCK_FILES)

    @property
    def version(self) -> str:
        digest = hashlib.sha1()
        for path in self._files():
            stat = path.stat()
            digest.update(f"{path.name}:{stat.st_size}:{stat.st_mtime_ns};".encode())
        return digest.hexdigest()[:12]

    def objects(self) -> list[ObjectInfo]:
        out: list[ObjectInfo] = []
        for object_id, (name, category, stem) in self._MESHES.items():
            if all(f.is_file() for f in self._mesh_files(stem)):
                out.append(ObjectInfo(object_id, name, category, "mesh"))  # type: ignore[arg-type]
        if self._has_drillholes():
            out.append(ObjectInfo("drillholes", "Drillholes", "drillholes", "drillholes"))
        if self._has_blocks():
            out.append(ObjectInfo(*self._BLOCKS, "blockModels", "blocks"))
        return out

    # ------------------------------------------------------------------ data

    def drillholes(self, object_id: str) -> DrillholeData:
        if object_id != "drillholes" or not self._has_drillholes():
            raise KeyError(object_id)
        with self._lock:
            if self._drillholes is None:
                self._drillholes = self._load_drillholes()
            return self._drillholes

    def _load_drillholes(self) -> DrillholeData:
        kinds = {"collar": "collar", "survey": "survey", "assay": "interval", "geology": "interval"}
        sources: dict[str, TableSource] = {}
        for name, kind in kinds.items():
            path = self.folder / f"{name}.csv"
            suggestion = suggest_column_map(read_raw_table(path).columns, kind)  # type: ignore[arg-type]
            if suggestion.missing:
                raise ValueError(f"{path.name}: no column found for {suggestion.missing}")
            sources[name] = TableSource(path, suggestion.mapping)
        database = import_drillholes(
            sources["collar"],
            sources["survey"],
            {"assay": sources["assay"], "geology": sources["geology"]},
        )
        return DrillholeData(database, database.desurvey(), self._units())

    def _units(self) -> dict[str, str]:
        """Grade unit from ``truth.json`` (``grade.element`` / ``grade.units``)."""
        try:
            grade = json.loads((self.folder / "truth.json").read_text())["grade"]
            return {str(grade["element"]): str(grade["units"])}
        except (OSError, ValueError, KeyError, TypeError):
            return {}

    def mesh(self, object_id: str) -> Mesh:
        if object_id not in self._MESHES:
            raise KeyError(object_id)
        vertices, faces = self._mesh_files(self._MESHES[object_id][2])
        if not (vertices.is_file() and faces.is_file()):
            raise KeyError(object_id)
        with self._lock:
            if object_id not in self._meshes:
                self._meshes[object_id] = Mesh(np.load(vertices), np.load(faces))
            return self._meshes[object_id]


    def blocks(self, object_id: str) -> BlockData:
        if object_id != self._BLOCKS[0] or not self._has_blocks():
            raise KeyError(object_id)
        from bakumine.api.blocks import BlockData, build_demo_block_model
        from bakumine.core.blockmodel.constraints import DictResolver
        from bakumine.core.geometry.dtm import DTM

        def mesh_or_none(name: str) -> Mesh | None:
            try:
                return self.mesh(name)
            except KeyError:
                return None

        solid, topo = mesh_or_none("ore-solid"), mesh_or_none("topo")
        with self._lock:
            if self._blocks is None:
                model = build_demo_block_model(self.folder, solid, topo)
                solids = {"ore-solid": solid} if solid is not None else {}
                dtms = {"topo": DTM(topo)} if topo is not None else {}
                self._blocks = BlockData(
                    model,
                    DictResolver(dtms=dtms, solids=solids),
                    solids={k: self._MESHES[k][0] for k in solids},
                    surfaces={k: self._MESHES[k][0] for k in dtms},
                    # Every block at once, air included, is an opaque brick: open on the ore.
                    display={
                        "color_by": "au",
                        "category_filter": {"attribute": "rock", "values": ["ORE"]},
                    },
                )
            return self._blocks


# ---------------------------------------------------------------------- active source

_source: SceneSource | None = None
_source_lock = threading.Lock()


def get_scene_source() -> SceneSource:
    """The active source (FastAPI dependency). Defaults to the demo project."""
    global _source
    with _source_lock:
        if _source is None:
            folder = Path(os.environ.get(DEMO_FOLDER_ENV) or DEMO_FOLDER)
            _source = DemoSceneSource(folder) if folder.is_dir() else EmptySceneSource()
        return _source


def set_scene_source(source: SceneSource | None) -> None:
    """Replace the active source; ``None`` returns to the default on next use."""
    global _source
    with _source_lock:
        _source = source
