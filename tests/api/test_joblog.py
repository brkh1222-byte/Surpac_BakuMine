"""P6.T7: job log, replay with the CLI and undo.

The synthetic chain (import -> validate -> composite -> create -> code -> estimate -> report)
runs through ``/api/v1/ops``. Coding and estimation are each undone once and redone; the
undo must give the model back bit for bit. The log is then replayed with ``bakumine project
replay`` into a new project, whose model must equal the original bit for bit (the task
allowed rtol 1e-12; the CLI runs the same code on the same inputs, so it is exact).
"""

from __future__ import annotations

import json
import subprocess

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from test_ops_api import SYNTH, coding, estimate_spec, wait

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.blockmodel.storage import load_model, read_header
from bakumine.core.project import Project

pytestmark = pytest.mark.skipif(
    not (SYNTH / "truth_blocks.parquet").is_file(),
    reason="data/synthetic/default has not been generated",
)


def same_model(a, b, *, rtol: float = 0.0) -> None:
    """Structure exactly; float attributes to ``rtol`` (0 = bit for bit), others exactly."""
    assert a.definition.to_dict() == b.definition.to_dict()
    assert a.n_blocks == b.n_blocks
    np.testing.assert_array_equal(a.parent, b.parent)
    np.testing.assert_array_equal(a.level, b.level)
    np.testing.assert_array_equal(a.sub, b.sub)
    assert a.attribute_names == b.attribute_names
    for name in a.attribute_names:
        assert a.attribute(name).to_dict() == b.attribute(name).to_dict(), name
        x, y = a[name], b[name]
        assert x.dtype == y.dtype, name
        if x.dtype.kind == "f" and rtol:
            np.testing.assert_allclose(x, y, rtol=rtol, atol=0, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(x, y, err_msg=name)


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("joblog") / "source"
    project = Project.create(folder, "source")
    project.add("solid", SYNTH / "ore_solid.obj", "Ore solid", "ore-solid")
    project.add("dtm", SYNTH / "topo.obj", "Topography", "topo")
    set_scene_source(ProjectSceneSource(folder))
    client = TestClient(app)
    truth = json.loads((SYNTH / "truth.json").read_text())
    states: dict = {}

    def ok(response):
        assert response.status_code == 200, response.text
        return response.json()

    for table in ("collar", "survey", "assay", "geology"):
        ok(client.put(f"/api/v1/ops/uploads/{table}", params={"filename": f"{table}.csv"},
                      content=(SYNTH / f"{table}.csv").read_bytes()))  # fmt: skip
    db = ok(client.post("/api/v1/ops/import", json={"name": "drillholes"}))["id"]
    ok(client.post(f"/api/v1/ops/databases/{db}/validate"))
    # a failing request is logged as an error and skipped by replay
    assert client.post("/api/v1/ops/composite", json={"database": "nope"}).status_code == 400
    comp = ok(client.post(
        "/api/v1/ops/composite",
        json={"database": db, "table": "assay", "mode": "domain", "length": 2,
              "domain_table": "geology", "domain_column": "rock", "name": "composites_assay"},
    ))["id"]  # fmt: skip
    info = truth["blocks"]
    model_id = ok(client.post(
        "/api/v1/ops/models",
        json={"name": "model", "origin": info["origin"], "size": info["size"],
              "shape": info["shape"], "max_depth": 1},
    ))["id"]  # fmt: skip
    path = Project.open(folder).path_of(model_id)
    states["created"] = (load_model(path), read_header(path).checksum)

    ok(client.post(f"/api/v1/ops/models/{model_id}/code", json=coding(truth)))
    states["coded"] = load_model(path)
    log = ok(client.get("/api/v1/ops/log"))
    assert log["jobs"][0]["operation"] == "code" and log["undo"] == log["jobs"][0]["id"]
    states["undo_code"] = ok(client.post("/api/v1/ops/undo"))
    states["after_undo_code"] = (load_model(path), read_header(path).checksum)
    ok(client.post(f"/api/v1/ops/models/{model_id}/code", json=coding(truth)))
    states["recoded"] = (load_model(path), read_header(path).checksum)

    def estimate():
        job = ok(client.post("/api/v1/ops/estimate",
                             json={"model": model_id, "composites": comp,
                                   "plan": estimate_spec()}))  # fmt: skip
        assert wait(client, job["id"])["status"] == "done"

    estimate()
    states["estimated"] = load_model(path)
    states["undo_estimate"] = ok(client.post("/api/v1/ops/undo"))
    states["after_undo_estimate"] = (load_model(path), read_header(path).checksum)
    estimate()
    ok(client.post(f"/api/v1/scene/objects/{model_id}/report",
                   json={"grade_attributes": ["au"], "density": "density",
                         "group_by": ["rock"]}))  # fmt: skip
    states["final"] = load_model(path)
    states["log"] = ok(client.get("/api/v1/ops/log"))
    yield {"client": client, "folder": folder, "db": db, "comp": comp, "model": model_id,
           **states}  # fmt: skip
    set_scene_source(None)


def test_log_has_every_operation(run):
    entries = joblog.read(run["folder"])
    ops = [e["operation"] for e in entries]
    assert ops == ["import", "validate", "composite", "composite", "create_model", "code",
                   "undo", "code", "estimate", "undo", "estimate", "report"]  # fmt: skip
    assert [e["status"] for e in entries].count("error") == 1
    for e in entries:
        assert {"id", "time", "params", "inputs", "outputs", "status", "duration_s"} <= set(e)
        assert e["duration_s"] >= 0
    imported = entries[0]
    for rel in imported["replay"]["files"].values():  # uploads were archived in the project
        assert (run["folder"] / rel).is_file()
    assert imported["outputs"][0]["id"] == run["db"]
    estimate = entries[8]
    assert estimate["params"]["plan"] == estimate_spec()
    assert estimate["summary"]["n_estimated"] > 0
    assert {o["id"] for o in estimate["inputs"]} >= {run["model"], run["comp"], "ore-solid"}
    # the API view: newest first, undone flags, nothing left to undo except the last estimate
    jobs = run["log"]["jobs"]
    assert jobs[0]["operation"] == "report"
    undone = {j["id"] for j in jobs if j["undone"]}
    assert undone == {entries[5]["id"], entries[8]["id"]}
    assert run["log"]["undo"] == entries[10]["id"]


def test_undo_code_restores_the_model_bit_for_bit(run):
    before, checksum = run["created"]
    after, after_checksum = run["after_undo_code"]
    assert run["undo_code"]["operation"] == "code"
    assert after_checksum == checksum
    same_model(after, before)
    assert run["coded"].n_blocks != before.n_blocks  # coding did sub-block the model
    redone, _ = run["recoded"]
    same_model(redone, run["coded"])


def test_undo_estimate_restores_the_model_bit_for_bit(run):
    before, checksum = run["recoded"]
    after, after_checksum = run["after_undo_estimate"]
    assert run["undo_estimate"]["operation"] == "estimate"
    assert "au" in run["estimated"].attribute_names and "au" not in after.attribute_names
    assert after_checksum == checksum
    same_model(after, before)
    same_model(run["final"], run["estimated"])


def test_replay_gives_an_identical_project(run, tmp_path):
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(run["folder"] / joblog.LOG_FILE), "--into",
                 str(target)]) == 0  # fmt: skip
    new, old = Project.open(target), Project.open(run["folder"])
    assert {o.to_dict()["id"]: o.to_dict() for o in new.list()} == {
        o.to_dict()["id"]: o.to_dict() for o in old.list()
    }
    same_model(load_model(new.path_of(run["model"])), run["final"])
    pd.testing.assert_frame_equal(
        pd.read_parquet(new.path_of(run["comp"])), pd.read_parquet(old.path_of(run["comp"]))
    )
    for table in ("collar", "survey", "interval_assay", "interval_geology"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(new.path_of(run["db"]) / f"{table}.parquet"),
            pd.read_parquet(old.path_of(run["db"]) / f"{table}.parquet"),
        )


def test_export_log_writes_runnable_scripts(run, tmp_path):
    sh, py = tmp_path / "replay.sh", tmp_path / "replay.py"
    log = str(run["folder"])
    assert main(["project", "export-log", log, "--out", str(sh)]) == 0
    assert main(["project", "export-log", log, "--out", str(py)]) == 0
    text = sh.read_text()
    subprocess.run(["sh", "-n", str(sh)], check=True)  # syntax only; replay runs the steps
    for command in ("$BAKUMINE import ", "$BAKUMINE validate ", "$BAKUMINE composite ",
                    "$BAKUMINE blockmodel create ", "$BAKUMINE blockmodel code ",
                    "$BAKUMINE estimate ", "$BAKUMINE blockmodel report ",
                    "$BAKUMINE project add "):  # fmt: skip
        assert command in text, command
    assert text.count("$BAKUMINE estimate ") == 1  # the undone estimate is not replayed
    compile(py.read_text(), str(py), "exec")


def test_nothing_to_undo_is_409(tmp_path):
    Project.create(tmp_path / "p")
    set_scene_source(ProjectSceneSource(tmp_path / "p"))
    try:
        client = TestClient(app)
        assert client.post("/api/v1/ops/undo").status_code == 409
        assert client.get("/api/v1/ops/log").json() == {"jobs": [], "undo": None,
                                                         "keep_snapshots": 5}  # fmt: skip
    finally:
        set_scene_source(None)


def test_log_skips_a_damaged_last_line(tmp_path):
    joblog.append(tmp_path, {"id": "a", "operation": "x", "params": {}})
    with open(tmp_path / joblog.LOG_FILE, "a") as handle:
        handle.write('{"id": "b", "operat')
    assert [e["id"] for e in joblog.read(tmp_path)] == ["a"]


def test_only_the_newest_snapshots_are_kept(tmp_path, monkeypatch):
    from bakumine.core.blockmodel.blocks import BlockModel
    from bakumine.core.blockmodel.definition import BlockModelDefinition
    from bakumine.core.blockmodel.storage import save_model

    monkeypatch.setenv("BAKUMINE_UNDO_SNAPSHOTS", "2")
    path = tmp_path / "models" / "m"
    save_model(BlockModel(BlockModelDefinition((0, 0, 0), (1, 1, 1), (2, 2, 2))), path)
    for job in ("j1", "j2", "j3"):
        joblog.take_snapshot(tmp_path, job, "m", path, ["x"])
    assert sorted(p.name for p in (tmp_path / joblog.SNAPSHOTS).iterdir()) == ["j2", "j3"]
