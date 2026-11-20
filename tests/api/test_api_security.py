"""P7.T5 review of the API: requests that must not reach outside the project, run
away with resources or end in a 500."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.core.project import Project

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "synthetic" / "default"

needs_synth = pytest.mark.skipif(
    not (SYNTH / "collar.csv").is_file(), reason="data/synthetic/default has not been generated"
)


@pytest.fixture
def project(tmp_path):
    folder = tmp_path / "p"
    Project.create(folder, "sec")
    set_scene_source(ProjectSceneSource(folder))
    yield TestClient(app), folder
    set_scene_source(None)


def _import(client: TestClient) -> str:
    for table in ("collar", "survey", "assay", "geology"):
        r = client.put(
            f"/api/v1/ops/uploads/{table}",
            params={"filename": f"{table}.csv"},
            content=(SYNTH / f"{table}.csv").read_bytes(),
        )
        assert r.status_code == 200, r.text
    r = client.post("/api/v1/ops/import", json={"name": "dh"})
    assert r.status_code == 200, r.text
    return r.json()["id"]


@needs_synth
@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_composite_name_cannot_write_outside_the_project(project, tmp_path, kind):
    client, folder = project
    db = _import(client)
    outside = tmp_path / "outside"
    outside.mkdir()
    name = str(outside / "pwned") if kind == "absolute" else "../../outside/pwned"
    r = client.post(
        "/api/v1/ops/composite",
        json={"database": db, "table": "assay", "length": 2, "name": name},
    )
    assert r.status_code == 200, r.text
    assert list(outside.iterdir()) == []
    obj = Project.open(folder).get(r.json()["id"])
    assert obj.name == name  # kept as display text
    assert Project.open(folder).path_of(obj).is_relative_to(folder)
