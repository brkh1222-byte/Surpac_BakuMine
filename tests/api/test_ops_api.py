"""Dialog endpoints (P6.T2 - P6.T5): same parameters as the CLI give the same results.

The CLI side is ``scripts/gate_synthetic.py`` (import -> validate -> composite -> create ->
code -> estimate); the API side runs the same chain through ``/api/v1/ops``.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app, operations
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.core.blockmodel.storage import load_model
from bakumine.core.project import Project

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "synthetic" / "default"

pytestmark = pytest.mark.skipif(
    not (SYNTH / "truth_blocks.parquet").is_file(),
    reason="data/synthetic/default has not been generated",
)

ELLIPSOID = {
    "radii": [120.0, 70.0, 15.0],
    "angles": [35.0, -10.0, 50.0],
}  # same ellipsoid as truth.json's axes


def _gate():
    spec = importlib.util.spec_from_file_location(
        "gate_synthetic", ROOT / "scripts/gate_synthetic.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate_synthetic"] = module
    spec.loader.exec_module(module)
    return module


def coding(truth: dict) -> dict:
    density = truth["density_t_m3"]
    return {
        "attributes": [
            {"name": "rock", "kind": "category", "background": "WASTE"},
            {"name": "density", "kind": "float", "unit": "t/m3"},
        ],
        "steps": [
            {"attribute": "rock", "value": "AIR", "subblock": True,
             "constraint": {"type": "dtm", "surface": "topo", "side": "above", "outside": True}},
            {"attribute": "rock", "value": "ORE", "subblock": True,
             "constraint": {"type": "solid", "solid": "ore-solid", "side": "inside"}},
            {"attribute": "density",
             "value": f'iif(rock == "ORE", {density["ORE"]}, {density["WASTE"]})',
             "constraint": {"type": "attribute", "expression": 'rock != "AIR"'}},
        ],
        "merge": True,
    }  # fmt: skip


def estimate_spec() -> dict:
    ore = {
        "name": "ORE",
        "block_constraint": {"type": "attribute", "expression": 'rock == "ORE"'},
        "sample_filter": {"column": "domain", "equals": "ORE"},
    }
    return {
        "attribute": "au",
        "method": "ok",
        "ellipsoid": ELLIPSOID,
        "variogram": {"nugget": 0.15, "structures": [{"type": "spherical", "sill": 0.85}]},
        "discretisation": [4, 4, 2],
        "domain": ore,
        "passes": [
            {"name": "tight", "radius_factor": 0.25, "min_samples": 6,
             "max_samples": 16, "max_per_hole": 4},
            {"name": "medium", "radius_factor": 0.5, "min_samples": 3,
             "max_samples": 16, "max_per_hole": 4},
            {"name": "wide", "radius_factor": 1.5, "min_samples": 1, "max_samples": 16},
        ],
    }  # fmt: skip


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory) -> Path:
    out = tmp_path_factory.mktemp("cli") / "gate"
    _gate().run(out)
    return out


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    folder = tmp_path_factory.mktemp("proj") / "p"
    project = Project.create(folder, "api")
    project.add("solid", SYNTH / "ore_solid.obj", "Ore solid", "ore-solid")
    project.add("dtm", SYNTH / "topo.obj", "Topography", "topo")
    set_scene_source(ProjectSceneSource(folder))
    client = TestClient(app)
    yield client, folder
    set_scene_source(None)


def wait(client: TestClient, job_id: str, timeout: float = 600) -> dict:
    end = time.time() + timeout
    while time.time() < end:
        job = client.get(f"/api/v1/ops/jobs/{job_id}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.2)
    raise AssertionError("job did not finish")


@pytest.fixture(scope="module")
def chain(api, cli_run):
    client, folder = api
    truth = json.loads((SYNTH / "truth.json").read_text())
    for table in ("collar", "survey", "assay", "geology"):
        r = client.put(
            f"/api/v1/ops/uploads/{table}",
            params={"filename": f"{table}.csv"},
            content=(SYNTH / f"{table}.csv").read_bytes(),
        )
        assert r.status_code == 200, r.text
        assert r.json()["missing"] == []
    imported = client.post("/api/v1/ops/import", json={"name": "drillholes"})
    assert imported.status_code == 200, imported.text
    db = imported.json()["id"]
    comp = client.post(
        "/api/v1/ops/composite",
        json={"database": db, "table": "assay", "mode": "domain", "length": 2,
              "domain_table": "geology", "domain_column": "rock", "name": "composites_assay"},
    )  # fmt: skip
    assert comp.status_code == 200, comp.text
    info = truth["blocks"]
    model = client.post(
        "/api/v1/ops/models",
        json={"name": "model", "origin": info["origin"], "size": info["size"],
              "shape": info["shape"], "max_depth": 1},
    )  # fmt: skip
    assert model.status_code == 200, model.text
    model_id = model.json()["id"]
    coded = client.post(f"/api/v1/ops/models/{model_id}/code", json=coding(truth))
    assert coded.status_code == 200, coded.text
    started = client.post(
        "/api/v1/ops/estimate",
        json={"model": model_id, "composites": comp.json()["id"], "plan": estimate_spec()},
    )
    assert started.status_code == 200, started.text
    job = wait(client, started.json()["id"])
    assert job["status"] == "done", job
    return {"db": db, "composites": comp.json()["id"], "model": model_id, "job": job,
            "imported": imported.json()}  # fmt: skip


def test_import_reports_validation(chain):
    imported = chain["imported"]
    assert imported["errors"] == 0
    assert imported["n_holes"] > 0
    assert imported["tables"]["assay"] > 0
    assert all(c["count"] > 0 for c in imported["counts"])


def test_database_and_composites_equal_cli(api, chain, cli_run):
    _, folder = api
    project = Project.open(folder)
    mine = pd.read_parquet(project.path_of(chain["composites"]))
    cli = pd.read_parquet(cli_run / "db" / "composites_assay.parquet")
    pd.testing.assert_frame_equal(mine, cli)
    for table in ("collar", "survey", "interval_assay", "interval_geology"):
        pd.testing.assert_frame_equal(
            pd.read_parquet(project.path_of(chain["db"]) / f"{table}.parquet"),
            pd.read_parquet(cli_run / "db" / f"{table}.parquet"),
        )


def test_model_equals_cli(api, chain, cli_run):
    _, folder = api
    mine = load_model(Project.open(folder).path_of(chain["model"]))
    cli = load_model(cli_run / "model")
    assert mine.n_blocks == cli.n_blocks
    assert sorted(mine.attributes) == sorted(cli.attributes)
    np.testing.assert_array_equal(mine.volumes(), cli.volumes())
    for name in cli.attributes:
        a, b = mine.get(name), cli.get(name)
        if a.dtype.kind == "f":
            np.testing.assert_allclose(a, b, rtol=1e-6, atol=0, equal_nan=True, err_msg=name)
        else:
            np.testing.assert_array_equal(a, b, err_msg=name)
    assert chain["job"]["summary"]["n_estimated"] > 0


def test_manifest_lists_the_new_objects(api, chain):
    client, _ = api
    ids = {o["id"] for o in client.get("/api/v1/scene").json()["objects"]}
    assert {chain["db"], chain["model"], "ore-solid", "topo"} <= ids
    summary = client.get("/api/v1/ops/project").json()
    kinds = {o["id"]: o["kind"] for o in summary["objects"]}
    assert kinds[chain["composites"]] == "composite"


def test_cancel_leaves_the_model_unchanged(api, chain):
    client, folder = api
    path = Project.open(folder).path_of(chain["model"])
    spec = estimate_spec() | {"attribute": "au2", "value_column": "au"}
    before = load_model(path)
    plan = operations.build_plan(spec)
    result = operations.run_estimation(
        ProjectSceneSource(folder), chain["model"], chain["composites"], plan, cancel=lambda: True
    )
    assert result["cancelled"] is True
    after = load_model(path)
    assert sorted(after.attributes) == sorted(before.attributes)
    assert "au2" not in after.attributes
    # and through the endpoints
    started = client.post(
        "/api/v1/ops/estimate",
        json={"model": chain["model"], "composites": chain["composites"],
              "plan": spec},
    )  # fmt: skip
    assert started.status_code == 200, started.text
    job_id = started.json()["id"]
    client.post(f"/api/v1/ops/jobs/{job_id}/cancel")
    job = wait(client, job_id)
    assert job["status"] == "cancelled", job
    assert sorted(load_model(path).attributes) == sorted(before.attributes)


def test_bad_requests_are_400(api, chain):
    client, _ = api
    assert client.post("/api/v1/ops/composite", json={"database": "nope"}).status_code == 400
    assert client.post("/api/v1/ops/models", json={"size": [1, 1]}).status_code == 400
    bad = client.post(
        "/api/v1/ops/estimate",
        json={"model": chain["model"], "composites": chain["composites"], "plan": {"method": "ok"}},
    )
    assert bad.status_code == 400
    assert client.get("/api/v1/ops/jobs/none").status_code == 404
