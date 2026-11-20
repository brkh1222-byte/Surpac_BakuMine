"""A project on ``data/synthetic/default`` for the geometry operation tests, and helpers.

The project holds the synthetic drillholes (collar, survey, geology), the true topography,
an empty block model with the extent of the truth model and one string object.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import save_model
from bakumine.core.drillholes.importer import TableSource, import_drillholes, load_database
from bakumine.core.geometry.surface_points import collar_points
from bakumine.core.io.files import write_strings
from bakumine.core.io.obj import read_obj
from bakumine.core.io.strings import StringFile, StringSegment
from bakumine.core.project import Project

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "synthetic" / "default"
OPS = "/api/v1/ops"
HAVE_SYNTH = (SYNTH / "collar.csv").is_file()


def synthetic_database(folder: Path) -> Path:
    def source(table: str) -> TableSource:
        columns = pd.read_csv(SYNTH / f"{table}.csv", nrows=0).columns
        return TableSource(SYNTH / f"{table}.csv", {c: c for c in columns})

    db = import_drillholes(source("collar"), source("survey"), {"geology": source("geology")})
    return db.save(folder)


def geometry_env(tmp_path_factory):
    """Generator for a module-scoped fixture: the project, served by the app."""
    base = tmp_path_factory.mktemp("geometry")
    folder = base / "project"
    project = Project.create(folder, "geometry")
    project.add("database", synthetic_database(base / "db"), "Drillholes", "drillholes")
    project.add("dtm", SYNTH / "topo.obj", "Topography", "topo")
    truth = json.loads((SYNTH / "truth.json").read_text())["blocks"]
    definition = BlockModelDefinition(truth["origin"], truth["size"], truth["shape"])
    save_model(BlockModel(definition), base / "model")
    project.add("model", base / "model", "Model", "model")
    # a string object in the project: one contour-like line across the collars
    database = load_database(project.path_of("drillholes"))
    collars = collar_points(database)
    order = np.argsort(collars[:, 0])
    line = collars[order[[0, len(order) // 2, -1]]] + [3.0, 7.0, 0.5]
    write_strings(base / "ridge.str", StringFile([StringSegment(1, line)]))
    project.add("string", base / "ridge.str", "Ridge", "ridge")
    set_scene_source(ProjectSceneSource(folder))
    client = TestClient(app)
    yield {"client": client, "folder": folder, "base": base, "database": database,
           "collars": collars, "definition": definition, "line": line}  # fmt: skip
    set_scene_source(None)


def project_mesh(folder: Path, object_id: str):
    return read_obj(Project.open(folder).path_of(object_id))


def same_mesh(a, b) -> None:
    np.testing.assert_array_equal(a.vertices, b.vertices)
    np.testing.assert_array_equal(a.faces, b.faces)


def upload(client: TestClient, path: Path, token: str | None = None) -> dict:
    params = {"filename": path.name, **({"token": token} if token else {})}
    r = client.put(f"{OPS}/geometry/uploads", params=params, content=path.read_bytes())
    assert r.status_code == 200, r.text
    return r.json()


def npy_mesh(folder: Path, name: str):
    from bakumine.core.blockmodel.constraints import FileResolver

    return FileResolver(folder).solid(name)
