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
