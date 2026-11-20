"""``POST /api/v1/ops/models/{id}/classify`` equals ``bakumine classify``; log, undo, replay.

The synthetic chain (import -> composite -> create -> code -> estimate) runs through the
operations API; the estimated model is then classified through the endpoint and, on a copy
of the model folder, with the CLI. The two models must be equal bit for bit.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_joblog import same_model
from test_ops_api import SYNTH, coding, estimate_spec, wait

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.blockmodel.storage import load_model, read_header
from bakumine.core.estimation.classification import ClassificationParams
from bakumine.core.project import Project

pytestmark = pytest.mark.skipif(
    not (SYNTH / "truth_blocks.parquet").is_file(),
    reason="data/synthetic/default has not been generated",
)

PARAMS = {
    "attribute": "au",
    "hole_support": {"radius": 40.0, "nth": 3},
    "categories": [
        {"label": "Measured",
         "criteria": {"max_pass": 2, "min_samples": 12, "min_holes": 4, "min_slope": 0.9}},
        {"label": "Indicated", "criteria": {"max_pass": 2, "max_hole_distance": 40.0}},
        {"label": "Inferred"},
    ],
    "smoothing": {"iterations": 1, "min_agree": 4, "max_same": 0},
}  # fmt: skip


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    folder = tmp_path_factory.mktemp("classify") / "p"
    project = Project.create(folder, "classify")
    project.add("solid", SYNTH / "ore_solid.obj", "Ore solid", "ore-solid")
    project.add("dtm", SYNTH / "topo.obj", "Topography", "topo")
    set_scene_source(ProjectSceneSource(folder))
    client = TestClient(app)
    truth = json.loads((SYNTH / "truth.json").read_text())

    def ok(response):
        assert response.status_code == 200, response.text
        return response.json()

    for table in ("collar", "survey", "assay", "geology"):
        ok(client.put(f"/api/v1/ops/uploads/{table}", params={"filename": f"{table}.csv"},
                      content=(SYNTH / f"{table}.csv").read_bytes()))  # fmt: skip
    db = ok(client.post("/api/v1/ops/import", json={"name": "drillholes"}))["id"]
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
    ok(client.post(f"/api/v1/ops/models/{model_id}/code", json=coding(truth)))
    job = ok(client.post("/api/v1/ops/estimate",
                         json={"model": model_id, "composites": comp,
                               "plan": estimate_spec()}))  # fmt: skip
    assert wait(client, job["id"])["status"] == "done"
    path = Project.open(folder).path_of(model_id)
    estimated = folder.parent / "estimated_copy"
    shutil.copytree(path, estimated)
    result = ok(client.post(f"/api/v1/ops/models/{model_id}/classify",
                            json={"params": PARAMS, "composites": comp}))  # fmt: skip
    yield {"client": client, "folder": folder, "model": model_id, "comp": comp, "path": path,
           "estimated": estimated, "result": result, "ok": ok}  # fmt: skip
    set_scene_source(None)


def test_endpoint_equals_cli(run, tmp_path, capsys):
    params = tmp_path / "classes.json"
    params.write_text(json.dumps(PARAMS), encoding="utf-8")
    copy, summary = tmp_path / "model", tmp_path / "summary.json"
    shutil.copytree(run["estimated"], copy)
    composites = Project.open(run["folder"]).path_of(run["comp"])
    code = main(["classify", str(copy), "--params", str(params), "--composites",
                 str(composites), "--summary", str(summary)])  # fmt: skip
    assert code == 0
    out = capsys.readouterr().out
    assert "Measured" in out and "classified" in out
    same_model(load_model(copy), load_model(run["path"]))
    assert json.loads(summary.read_text()) == run["result"]
    # --out leaves the input model alone
    other = tmp_path / "other"
    before = read_header(copy).checksum
    assert main(["classify", str(copy), "--params", str(params), "--composites",
                 str(composites), "--out", str(other)]) == 0  # fmt: skip
    assert read_header(copy).checksum == before == read_header(other).checksum
    assert main(["classify", str(copy), "--params", str(tmp_path / "nope.json")]) == 2
    assert main(["classify", str(tmp_path), "--params", str(params)]) == 2


def test_result_is_a_sensible_classification(run):
    result, model = run["result"], load_model(run["path"])
    assert result["output"] == "res_class"
    assert result["written"] == ["au_nholes", "au_dhole"]
    counts = {c["label"]: c["blocks"] for c in result["classes"]}
    assert all(n > 0 for n in counts.values()), counts
    estimated = ~np.isnan(model["au"])
    assert result["n_estimated"] == int(estimated.sum()) > 0
    assert result["n_classified"] == result["n_estimated"] and result["n_unclassified"] == 0
    assert sum(counts.values()) == result["n_classified"]
    # unestimated blocks stay unclassified; every estimated block has exactly one class
    np.testing.assert_array_equal(model["res_class"] >= 0, estimated)
    assert model.attribute("res_class").categories == ["Measured", "Indicated", "Inferred"]
    assert (model["au_nholes"][~estimated] == 0).all()
    assert result["n_smoothed"] > 0
    # without smoothing every Measured block meets the Measured criteria
    plain = load_model(run["estimated"])
    from bakumine.core.estimation.classification import classify, read_composites

    spec = ClassificationParams.from_dict({**PARAMS, "smoothing": None})
    composites = read_composites(Project.open(run["folder"]).path_of(run["comp"]))
    classify(plain, spec, composites)
    measured = plain.get("res_class", decode=True) == "Measured"
    assert measured.any()
    assert (plain["au_pass"][measured] <= 2).all() and (plain["au_ns"][measured] >= 12).all()
    assert (plain["au_nholes"][measured] >= 4).all() and (plain["au_slope"][measured] >= 0.9).all()
    rest = ~measured & (plain["res_class"] >= 0)
    fails = (
        (plain["au_pass"][rest] > 2)
        | (plain["au_ns"][rest] < 12)
        | (plain["au_nholes"][rest] < 4)
        | ~(plain["au_slope"][rest] >= 0.9)
    )
    assert fails.all()  # rules in order: nothing that meets Measured is in a later class


def test_job_log_undo_and_redo(run):
    client, ok, folder = run["client"], run["ok"], run["folder"]
    entry = [e for e in joblog.read(folder) if e["operation"] == "classify"][-1]
    assert entry["status"] == "done" and entry["modifies"] == run["model"] and entry["snapshot"]
    assert [o["kind"] for o in entry["inputs"]][:2] == ["model", "composite"]
    assert entry["summary"]["n_classified"] == run["result"]["n_classified"]
    assert entry["replay"]["params"]["categories"][0]["label"] == "Measured"
    classified = load_model(run["path"])
    # classify again with other labels, then undo: back to the first classification
    other = {"attribute": "au", "categories": [{"label": "C1", "criteria": {"max_pass": 1}},
                                                 {"label": "C2"}]}  # fmt: skip
    again = ok(client.post(f"/api/v1/ops/models/{run['model']}/classify", json={"params": other}))
    assert [c["label"] for c in again["classes"]] == ["C1", "C2"] and again["written"] == []
    assert load_model(run["path"]).attribute("res_class").categories == ["C1", "C2"]
    undone = ok(client.post("/api/v1/ops/undo"))
    assert undone["operation"] == "classify"
    same_model(load_model(run["path"]), classified)
    # undo of the first one: the estimated model, without the three attributes
    ok(client.post("/api/v1/ops/undo"))
    same_model(load_model(run["path"]), load_model(run["estimated"]))
    assert "res_class" not in load_model(run["path"])
    redo = ok(client.post(f"/api/v1/ops/models/{run['model']}/classify",
                          json={"params": PARAMS, "composites": run["comp"]}))  # fmt: skip
    assert redo == run["result"]
    same_model(load_model(run["path"]), classified)


def test_report_by_class_sums_to_the_total(run):
    client, ok = run["client"], run["ok"]
    url = f"/api/v1/scene/objects/{run['model']}/report"
    body = {"grade_attributes": ["au"], "density": "density"}
    grouped = ok(client.post(url, json={**body, "group_by": ["res_class"]}))["rows"]
    plain = ok(client.post(url, json=body))["rows"]
    total = next(r for r in plain if r["row"] == "total")
    groups = [r for r in grouped if r["row"] == "group"]
    assert {r["res_class"] for r in groups} == {"Measured", "Indicated", "Inferred", None}
    for column, value in total.items():
        if column in ("row", "au_grade", "res_class") or value is None:
            continue
        assert sum(r[column] or 0 for r in groups) == pytest.approx(value, rel=1e-9), column
    by_class = {r["res_class"]: r["blocks"] for r in groups}
    for entry in run["result"]["classes"]:
        assert by_class[entry["label"]] == entry["blocks"]


def test_bad_requests_are_400_and_change_nothing(run):
    client = run["client"]
    url = f"/api/v1/ops/models/{run['model']}/classify"
    before = read_header(run["path"]).checksum
    rules = {"attribute": "au", "categories": [{"label": "A", "criteria": {"min_holes": 2}}]}
    for body in (
        {},
        {"params": {"attribute": "au", "categories": []}},
        {"params": {"attribute": "nope", "categories": [{"label": "A"}]}},
        {"params": {"attribute": "au", "categories": [{"label": "A", "criteria": {"x": 1}}]}},
        {"params": {"attribute": "au", "output": "class", "categories": [{"label": "A"}]}},
        {"params": {"attribute": "au", "output": "rock", "categories": [{"label": "A"}],
                    "inputs": {"holes": "zz"}}, "composites": "nope"},
        {"params": {**rules, "inputs": {"holes": "missing_attribute"}}},
        {"params": PARAMS, "composites": run["model"]},
    ):  # fmt: skip
        r = client.post(url, json=body)
        assert r.status_code == 400, (body, r.text)
        assert read_header(run["path"]).checksum == before
    missing = client.post("/api/v1/ops/models/nope/classify", json={"params": PARAMS})
    assert missing.status_code == 400
    assert joblog.read(run["folder"])[-1]["status"] == "error"
    assert not joblog.read(run["folder"])[-1]["snapshot"]


def test_replay_gives_an_identical_model(run, tmp_path):
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(run["folder"]), "--into", str(target)]) == 0
    same_model(load_model(Project.open(target).path_of(run["model"])), load_model(run["path"]))
    script = tmp_path / "replay.sh"
    assert main(["project", "export-log", str(run["folder"]), "--out", str(script)]) == 0
    assert "$BAKUMINE classify " in script.read_text()
