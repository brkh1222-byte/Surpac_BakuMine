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


# ---------------------------------------------------------------------- uploads

CSV = b"hole_id,x,y,z,max_depth\nA,0,0,0,10\n"


@pytest.mark.parametrize("pattern", ["*", "c*", "[cs]ollar", "%3Follar"])
def test_deleting_an_upload_takes_no_wildcards(project, pattern):
    client, folder = project
    for table in ("collar", "survey"):
        r = client.put(f"/api/v1/ops/uploads/{table}", params={"filename": "t.csv"}, content=CSV)
        assert r.status_code == 200, r.text
    assert client.delete(f"/api/v1/ops/uploads/{pattern}").status_code == 400
    assert sorted(p.name for p in (folder / "uploads").iterdir()) == ["collar.csv", "survey.csv"]
    assert client.delete("/api/v1/ops/uploads/collar").status_code == 200
    assert [p.name for p in (folder / "uploads").iterdir()] == ["survey.csv"]


def test_uploads_are_limited_in_size(project, monkeypatch):
    from bakumine.api import routes_ops

    client, folder = project
    monkeypatch.setattr(routes_ops, "MAX_UPLOAD_BYTES", len(CSV) - 1)
    r = client.put("/api/v1/ops/uploads/collar", params={"filename": "c.csv"}, content=CSV)
    assert r.status_code == 413
    # a streamed body with no Content-Length is counted as it arrives
    r = client.put(
        "/api/v1/ops/uploads/collar", params={"filename": "c.csv"}, content=iter([CSV[:5], CSV[5:]])
    )
    assert r.status_code == 413
    assert not (folder / "uploads").exists() or not any((folder / "uploads").iterdir())
    monkeypatch.setattr(routes_ops, "MAX_UPLOAD_BYTES", len(CSV))
    r = client.put("/api/v1/ops/uploads/collar", params={"filename": "c.csv"}, content=CSV)
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------- 400, not 500


def _model(client: TestClient) -> str:
    r = client.post(
        "/api/v1/ops/models",
        json={"origin": [0, 0, 0], "size": [10, 10, 10], "shape": [3, 3, 3], "name": "m"},
    )
    assert r.status_code == 200, r.text
    return r.json()["id"]


def _deep_not(depth: int) -> dict:
    doc: dict = {"type": "attribute", "expression": "1 > 0"}
    for _ in range(depth):
        doc = {"type": "not", "child": doc}
    return doc


def test_malformed_requests_are_client_errors(project):
    import json

    client, _ = project
    client = TestClient(app, raise_server_exceptions=False)
    model = _model(client)
    ops = "/api/v1/ops"
    bad = [
        client.post(f"{ops}/models", json={"origin": [0, 0, 0], "size": [1, 1, 1],
                                           "shape": ["inf", 2, 2]}),
        client.post(f"{ops}/estimate", json={"model": model, "composites": "c", "plan": [1]}),
        client.post(f"{ops}/estimate", json={"model": model, "composites": "c",
                                             "plan": {"attribute": "a", "passes": "ab"}}),
        client.post(f"{ops}/import", json={"maps": "x"}),
        client.post(f"{ops}/import", json={"options": [1]}),
        client.post(f"{ops}/models/{model}/code",
                    json={"steps": [{"attribute": "a", "value": 1, "constraint": _deep_not(900)}]}),
        client.post(f"/api/v1/scene/objects/{model}/report",
                    json={"where": json.dumps(_deep_not(900))}),
        client.get(f"/api/v1/scene/objects/{model}/blocks",
                   params={"where": '{"type": ' + "[" * 3000 + "]" * 3000 + "}"}),
    ]  # fmt: skip
    assert [r.status_code for r in bad] == [400] * len(bad), [r.text for r in bad]
    r = client.get(f"/api/v1/scene/objects/{model}/blocks/detail", params={"block_id": "9" * 40})
    assert r.status_code == 404
    # a reasonable depth still works
    r = client.post(
        f"/api/v1/scene/objects/{model}/report", json={"where": json.dumps(_deep_not(40))}
    )
    assert r.status_code == 200, r.text


# ---------------------------------------------------------------------- resource limits


def test_requests_too_large_to_run_are_refused(project, monkeypatch):
    from bakumine.api.routes_report import MAX_CUTOFFS

    client, folder = project
    model = _model(client)
    r = client.post(
        "/api/v1/ops/models",
        json={"origin": [0, 0, 0], "size": [1, 1, 1], "shape": [1000, 1000, 101]},
    )
    assert r.status_code == 400 and "at most" in r.json()["detail"]
    r = client.post(
        f"/api/v1/scene/objects/{model}/report",
        json={"cutoffs": [0.1 * i for i in range(MAX_CUTOFFS + 1)], "cutoff_attribute": "x"},
    )
    assert r.status_code == 422
    plan = {"attribute": "au", "method": "idw"}
    for extra in ({"discretisation": [100, 100, 100]}, {"passes": [{"max_samples": 10**8}]}):
        r = client.post("/api/v1/ops/estimate",
                        json={"model": model, "composites": "c", "plan": plan | extra})  # fmt: skip
        assert r.status_code == 400 and "at most" in r.json()["detail"], r.text
    # unknown composites: refused before a job starts
    r = client.post("/api/v1/ops/estimate", json={"model": model, "composites": "c", "plan": plan})
    assert r.status_code == 400 and "no composites" in r.json()["detail"]


def test_jobs_are_bounded(project, monkeypatch):
    from bakumine.api import operations

    client, folder = project
    monkeypatch.setattr(operations, "_jobs", {})
    monkeypatch.setattr(operations, "MAX_FINISHED_JOBS", 3)
    for _ in range(5):
        done = operations.Job("estimate")
        done.status = "done"
        operations._jobs[done.id] = done
    running = [operations.Job("estimate") for _ in range(operations.MAX_RUNNING_JOBS)]
    for job in running:
        job.status = "running"
        operations._jobs[job.id] = job
    # one more is refused while MAX_RUNNING_JOBS run (409); old finished jobs are dropped
    monkeypatch.setattr(operations, "build_plan", lambda spec: None)
    monkeypatch.setattr(operations, "_model_path", lambda folder, model: None)
    monkeypatch.setattr(operations.Project, "open", lambda folder: _FakeProject())
    with pytest.raises(operations.BusyError):
        operations.start_estimation(_Source(folder), "m", "c", {})
    running[0].status = "done"
    monkeypatch.setattr(operations, "run_estimation", lambda *a, **k: {"cancelled": False})
    job = operations.start_estimation(_Source(folder), "m", "c", {})
    finished = [j for j in operations._jobs.values() if j.status not in ("queued", "running")]
    assert len(finished) <= 3 and job.id in operations._jobs


class _Source:
    def __init__(self, folder):
        self.folder = folder


class _FakeProject:
    def get(self, object_id):
        from types import SimpleNamespace

        return SimpleNamespace(kind="composite")


# ---------------------------------------------------------------------- request guard


@pytest.mark.parametrize(
    ("host", "status"),
    [
        ("evil.example", 400),
        ("evil.example:8000", 400),
        ("127.0.0.1.evil.example", 400),
        ("localhost:8000", 200),
        ("127.0.0.1:5173", 200),
        ("[::1]:8000", 200),
        ("LOCALHOST", 200),
    ],
)
def test_dns_rebinding_hosts_are_refused(host, status):
    client = TestClient(app)
    assert client.get("/api/health", headers={"host": host}).status_code == status


def test_allowed_hosts_from_the_environment(monkeypatch):
    client = TestClient(app)
    assert client.get("/api/health", headers={"host": "box.lan:8000"}).status_code == 400
    monkeypatch.setenv("BAKUMINE_ALLOWED_HOSTS", "box.lan, other")
    assert client.get("/api/health", headers={"host": "box.lan:8000"}).status_code == 200


@pytest.mark.parametrize(
    ("origin", "status"),
    [
        ("https://evil.example", 403),
        ("null", 403),
        ("http://localhost:9999", 403),  # another local page is another origin
        ("http://localhost:8000", 404),  # same origin: reaches the route (no such job)
        (None, 404),  # not a browser
    ],
)
def test_cross_origin_state_changes_are_refused(origin, status):
    client = TestClient(app)
    headers = {"host": "localhost:8000"}
    if origin is not None:
        headers["origin"] = origin
    assert client.post("/api/v1/ops/jobs/nope/cancel", headers=headers).status_code == status
    # reads are not refused for an Origin (no CORS headers: the page cannot read them)
    assert client.get("/api/health", headers=headers).status_code == 200
