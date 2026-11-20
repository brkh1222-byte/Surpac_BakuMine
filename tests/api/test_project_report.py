"""Project folder (P6.T1) and report endpoint (P6.T6)."""

from __future__ import annotations

import io
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import DemoSceneSource, get_scene_source
from bakumine.cli import main
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.drillholes.importer import load_database
from bakumine.core.io.obj import read_obj
from bakumine.core.project import Project, ProjectError

DEMO = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"

pytestmark = pytest.mark.skipif(
    not (DEMO / "truth_blocks.parquet").is_file(),
    reason="data/synthetic/default has not been generated",
)


@pytest.fixture(scope="module")
def demo() -> DemoSceneSource:
    return DemoSceneSource(DEMO)


@pytest.fixture(scope="module")
def sources(demo, tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("sources")
    demo.drillholes("drillholes").database.save(root / "db")
    save_model(demo.blocks("block-model").model, root / "model")
    return {"db": root / "db", "model": root / "model", "solid": DEMO / "ore_solid.obj"}


def test_project_reopen_returns_identical_objects_and_data(tmp_path, sources):
    folder = tmp_path / "proj"
    project = Project.create(folder, "Test project")
    db = project.add("database", sources["db"], name="Drillholes")
    solid = project.add("solid", sources["solid"], name="Ore solid")
    model = project.add("model", sources["model"], name="Model")
    assert [o.id for o in project.list()] == ["drillholes", "ore-solid", "model"]

    again = Project.open(folder)
    assert again.name == "Test project"
    assert again.list() == project.list()
    assert not list(folder.glob(".project-*"))
    for sub in ("database", "composites", "strings", "dtm", "solids", "models", "constraints"):
        assert (folder / sub).is_dir()

    a, b = load_database(sources["db"]), load_database(again.path_of(db))
    for name in a.tables:
        pd.testing.assert_frame_equal(a.tables[name], b.tables[name], check_exact=True)
    ma, mb = read_obj(sources["solid"]), read_obj(again.path_of(solid))
    np.testing.assert_array_equal(ma.vertices, mb.vertices)
    np.testing.assert_array_equal(ma.faces, mb.faces)
    xa, xb = load_model(sources["model"]), load_model(again.path_of(model))
    assert xa.n_blocks == xb.n_blocks
    np.testing.assert_array_equal(xa.get("au"), xb.get("au"))

    again.remove("ore-solid")
    assert [o.id for o in Project.open(folder).list()] == ["drillholes", "model"]
    assert not again.path_of(solid).exists()


def test_project_errors(tmp_path):
    with pytest.raises(ProjectError):
        Project.open(tmp_path)
    project = Project.create(tmp_path / "p")
    with pytest.raises(ProjectError):
        Project.create(tmp_path / "p")
    with pytest.raises(ProjectError):
        project.add("nonsense", tmp_path)
    with pytest.raises(FileNotFoundError):
        project.add("solid", tmp_path / "missing.obj")


def test_cli_project_init_add_list(tmp_path, sources, capsys):
    folder = str(tmp_path / "cli")
    assert main(["project", "init", folder, "--name", "CLI"]) == 0
    assert main(["project", "add", folder, "solid", str(sources["solid"])]) == 0
    assert main(["project", "add", folder, "solid", str(sources["solid"])]) == 0
    capsys.readouterr()
    assert main(["project", "list", folder]) == 0
    out = capsys.readouterr().out
    assert "ore_solid" in out and "ore-solid-2" in out
    assert main(["project", "list", str(tmp_path / "nope")]) == 2


@pytest.fixture
def project_client(tmp_path, sources):
    project = Project.create(tmp_path / "api", "API project")
    project.add("database", sources["db"], name="Drillholes")
    project.add("solid", sources["solid"], name="Ore solid")
    project.add("model", sources["model"], name="Model")
    source = ProjectSceneSource(project.folder)
    app.dependency_overrides[get_scene_source] = lambda: source
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_api_manifest_lists_project_objects(project_client):
    manifest = project_client.get("/api/v1/scene").json()
    assert manifest["project"]["name"] == "API project"
    found = {o["id"]: (o["category"], o["kind"]) for o in manifest["objects"]}
    assert found == {
        "drillholes": ("drillholes", "drillholes"),
        "ore-solid": ("solids", "mesh"),
        "model": ("blockModels", "blocks"),
    }
    assert project_client.get("/api/v1/scene/objects/ore-solid/mesh").status_code == 200
    assert project_client.get("/api/v1/scene/objects/model/blocks").status_code == 200


def test_env_var_selects_project(tmp_path, monkeypatch):
    from bakumine.api import scene

    Project.create(tmp_path / "envp", "Env project")
    monkeypatch.setenv(scene.PROJECT_FOLDER_ENV, str(tmp_path / "envp"))
    scene.set_scene_source(None)
    try:
        assert scene.get_scene_source().project_name == "Env project"
    finally:
        scene.set_scene_source(None)


# ------------------------------------------------------------------ report

REPORT = "/api/v1/scene/objects/block-model/report"
CLI_ARGS = [
    "--where", "au > 0.3",
    "-g", "rock",
    "--cutoffs", "0.5,1",
    "--cutoff-attribute", "au",
    "--grade", "au:g/t",
    "--density", "density",
]  # fmt: skip
BODY = {
    "where": "au > 0.3",
    "group_by": ["rock"],
    "cutoffs": [0.5, 1],
    "cutoff_attribute": "au",
    "grade_attributes": ["au:g/t"],
    "density": "density",
}


@pytest.fixture
def demo_client(demo):
    app.dependency_overrides[get_scene_source] = lambda: demo
    yield TestClient(app)
    app.dependency_overrides.clear()


def test_report_endpoint_equals_cli(demo, demo_client, tmp_path, capsys):
    model_dir = tmp_path / "m"
    save_model(demo.blocks("block-model").model, model_dir)
    out_csv = tmp_path / "cli.csv"
    assert main(["blockmodel", "report", str(model_dir), *CLI_ARGS, "--out", str(out_csv)]) == 0
    capsys.readouterr()
    cli = pd.read_csv(out_csv)

    response = demo_client.post(REPORT, json=BODY)
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["columns"] == list(cli.columns)
    api = pd.DataFrame(body["rows"], columns=body["columns"])
    pd.testing.assert_frame_equal(api, cli, check_dtype=False, rtol=1e-9)

    csv = demo_client.post(REPORT + ".csv", json=BODY)
    assert csv.status_code == 200 and csv.headers["content-type"].startswith("text/csv")
    assert "attachment" in csv.headers["content-disposition"]
    assert csv.text == out_csv.read_text()
    pd.testing.assert_frame_equal(pd.read_csv(io.StringIO(csv.text)), cli)


def test_report_constant_density_bench_and_errors(demo_client):
    ok = demo_client.post(REPORT, json={"density": 2.7, "bench_height": 10, "partial": None})
    assert ok.status_code == 200
    rows = ok.json()["rows"]
    assert rows[-1]["row"] == "total" and rows[-1]["tonnes"] == pytest.approx(
        rows[-1]["volume"] * 2.7
    )
    assert demo_client.post(REPORT, json={"group_by": ["nope"]}).status_code == 400
    assert demo_client.post(REPORT, json={"where": "au >"}).status_code == 400
    assert demo_client.post(REPORT, json={"where": json.dumps({"type": "x"})}).status_code == 400
    missing = "/api/v1/scene/objects/nothing/report"
    assert demo_client.post(missing, json={}).status_code == 404
