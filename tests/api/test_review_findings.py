"""Defects found by the final walk through the browser (API side), one test per finding.

A small project is built directly with the core: a 6 x 6 x 3 block model with a ``rock``
category, and composites of four vertical holes in two domains.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.project import Project

OPS = "/api/v1/ops"


def small_model() -> BlockModel:
    model = BlockModel(BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (6, 6, 3)))
    model.add_attribute("rock", "category", categories=["WASTE", "ORE"])
    model["rock"][:] = np.arange(model.n_blocks) % 2
    return model


def small_composites() -> pd.DataFrame:
    rows = []
    for h, (x, y) in enumerate([(12.0, 12.0), (48.0, 14.0), (15.0, 47.0), (45.0, 45.0)]):
        for k in range(6):
            rows.append(
                {"hole_id": f"H{h + 1}", "x": x, "y": y, "z": 2.5 + 5.0 * k, "length": 5.0,
                 "au": 0.5 + 0.1 * h + 0.05 * k, "au_coverage": 1.0,
                 "domain": "ORE" if k % 2 else "WASTE"}
            )  # fmt: skip
    return pd.DataFrame(rows)


@pytest.fixture
def env(tmp_path):
    folder = tmp_path / "project"
    project = Project.create(folder, "review")
    save_model(small_model(), tmp_path / "model")
    project.add("model", tmp_path / "model", "Model", "model")
    small_composites().to_parquet(tmp_path / "composites.parquet")
    project.add("composite", tmp_path / "composites.parquet", "Composites", "composites")
    set_scene_source(ProjectSceneSource(folder))
    yield TestClient(app), folder
    set_scene_source(None)


def plan(**more) -> dict:
    return {"attribute": "au", "method": "nn", "ellipsoid": {"radii": [200, 200, 200]}} | more


def estimate(client: TestClient, spec: dict) -> dict:
    r = client.post(f"{OPS}/estimate", json={"model": "model", "composites": "composites",
                                             "plan": spec})  # fmt: skip
    assert r.status_code == 200, r.text
    end = time.time() + 60
    while time.time() < end:
        job = client.get(f"{OPS}/jobs/{r.json()['id']}").json()
        if job["status"] not in ("queued", "running"):
            return job
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ 1. re-estimating


def test_estimating_again_says_why_nothing_happened_and_overwrite_does_it(env):
    client, folder = env
    first = estimate(client, plan())
    assert first["status"] == "done" and first["summary"]["n_estimated"] == 108
    assert first["message"] == ""  # the status already says "done"
    path = Project.open(folder).path_of("model")
    before = np.array(load_model(path)["au"])

    again = estimate(client, plan(domain={"sample_filter": {"column": "domain", "equals": "ORE"}}))
    assert again["status"] == "done" and again["summary"]["n_targeted"] == 0
    assert "already have a value" in again["message"]
    assert "Estimate again blocks that already have a value" in again["message"]
    np.testing.assert_array_equal(load_model(path)["au"], before)

    spec = plan(overwrite=True, domain={"sample_filter": {"column": "domain", "equals": "ORE"}})
    forced = estimate(client, spec)
    assert forced["status"] == "done" and forced["summary"]["n_estimated"] == 108
    assert forced["message"] == ""
    assert not np.array_equal(load_model(path)["au"], before)
    logged = [e for e in joblog.read(folder) if e["operation"] == "estimate"]
    assert [e["replay"]["plan"]["overwrite"] for e in logged] == [False, False, True]

    bad = client.post(f"{OPS}/estimate", json={"model": "model", "composites": "composites",
                                               "plan": plan(overwrite="yes")})  # fmt: skip
    assert bad.status_code == 400 and "overwrite" in bad.json()["detail"]


def test_constraint_without_blocks_and_filter_without_composites_say_so(env):
    client, folder = env
    job = estimate(client, plan(domain={"sample_filter": {"column": "domain", "equals": "GOLD"}}))
    assert job["status"] == "error"
    assert "no composite matches the sample filter" in job["error"]
    assert "'GOLD'" in job["error"] and "'ORE'" in job["error"]  # asked for; what there is
    assert "au" not in load_model(Project.open(folder).path_of("model"))

    nothing = estimate(
        client,
        plan(domain={"block_constraint": {"type": "attribute", "expression": 'rock == "NONE"'}}),
    )
    assert nothing["status"] == "done" and nothing["summary"]["n_targeted"] == 0
    assert "select no block" in nothing["message"]



# ------------------------------------------------------------------ 2 / 5. classify


def classify(client: TestClient, params: dict, composites: str | None = "composites"):
    return client.post(f"{OPS}/models/model/classify",
                       json={"params": params, "composites": composites})  # fmt: skip


CLASSES = [{"label": "Measured", "criteria": {"max_pass": 1, "min_holes": 3}},
           {"label": "Inferred"}]  # fmt: skip
SUPPORT = {"radius": 40.0, "nth": 2}


def test_classify_refuses_to_replace_an_attribute_it_did_not_write(env, tmp_path, capsys):
    client, folder = env
    assert estimate(client, plan())["status"] == "done"
    path = Project.open(folder).path_of("model")
    rock = np.array(load_model(path)["rock"])
    params = {"attribute": "au", "output": "rock", "hole_support": SUPPORT, "categories": CLASSES}

    refused = classify(client, params)
    assert refused.status_code == 400, refused.text
    detail = refused.json()["detail"]
    assert "'rock'" in detail and "overwrite_output" in detail
    after = load_model(path)
    np.testing.assert_array_equal(after["rock"], rock)
    assert after.attribute("rock").categories == ["WASTE", "ORE"]

    # the same rule from the command line
    file = tmp_path / "classes.json"
    file.write_text(json.dumps(params), encoding="utf-8")
    table = Project.open(folder).path_of("composites")
    assert main(["classify", str(path), "--params", str(file), "--composites", str(table)]) == 2
    assert "overwrite_output" in capsys.readouterr().err
    np.testing.assert_array_equal(load_model(path)["rock"], rock)

    # a new attribute, and again into what classify wrote: no flag needed
    params["output"] = "res_class"
    first = classify(client, params)
    assert first.status_code == 200, first.text
    second = classify(client, params | {"categories": [{"label": "B"}, {"label": "C"}]})
    assert second.status_code == 200, second.text
    assert load_model(path).attribute("res_class").categories == ["B", "C"]

    # asked for explicitly: replaced
    forced = classify(client, params | {"output": "rock", "overwrite_output": True})
    assert forced.status_code == 200, forced.text
    assert load_model(path).attribute("rock").categories == ["Measured", "Inferred"]


@pytest.mark.parametrize(
    ("change", "text"),
    [
        ({"categories": [{"label": "M", "criteria": {"max_pass": 0}}]}, "max_pass"),
        ({"categories": [{"label": "M", "criteria": {"max_pass": -2}}]}, "max_pass"),
        ({"categories": [{"label": "M", "criteria": {"min_samples": -1}}]}, "min_samples"),
        ({"categories": [{"label": "M", "criteria": {"min_holes": -3}}]}, "min_holes"),
        ({"categories": [{"label": "M", "criteria": {"max_hole_distance": -5}}]},
         "max_hole_distance"),
        ({"categories": [{"label": "M" * 100_000}]}, "64 characters"),
        ({"categories": [{"label": f"c{i}"} for i in range(101)]}, "at most 100 classes"),
        ({"hole_support": {"radius": 1e12}}, "radius"),
        ({"hole_support": {"radius": 0}}, "radius"),
        ({"hole_support": {"radius": 40, "nth": 10**9}}, "nth"),
        ({"hole_support": {"radius": 40, "nth": 0}}, "nth"),
        ({"categories": [{"label": "M", "criteria": {"min_holes": 1},
                          "hole_support": {"radius": 1e12}}]}, "radius"),
        ({"categories": [{"label": "M", "criteria": {"min_holes": 1},
                          "hole_support": {"nth": 5000}}]}, "nth"),
    ],
)  # fmt: skip
def test_classify_rejects_values_that_cannot_be_meant(env, change, text):
    client, folder = env
    path = Project.open(folder).path_of("model")
    model = load_model(path)
    model.add_attribute("au", "float")
    model.set("au", np.ones(model.n_blocks))
    model.add_attribute("au_pass", "int")
    save_model(model, path)
    params = {"attribute": "au", "hole_support": SUPPORT, "categories": CLASSES} | change
    start = time.perf_counter()
    r = classify(client, params)
    assert r.status_code == 400, r.text
    assert text in r.json()["detail"]
    assert len(r.json()["detail"]) < 600
    assert time.perf_counter() - start < 5.0
    assert "res_class" not in load_model(path)


def test_classify_accepts_a_radius_of_the_size_of_the_model(env):
    client, folder = env
    assert estimate(client, plan())["status"] == "done"
    params = {"attribute": "au", "hole_support": {"radius": 500.0, "nth": 4},
              "categories": CLASSES}  # fmt: skip
    r = classify(client, params)
    assert r.status_code == 200, r.text


# ------------------------------------------------------------------ 3. hole id template


COLLAR = "hole_id,x,y,z,max_depth\nA1,0,0,100,50\nA2,10,0,100,50\n"


@pytest.mark.parametrize(
    "template", ["{0:>1000000}", "{1:.1000000f}", "{0:0>65}", "{0}" * 200, "{0:>{1}000000}"]
)
def test_hole_id_template_is_bounded(env, template):
    client, _ = env
    up = client.put(f"{OPS}/uploads/collar", params={"filename": "collar.csv"}, content=COLLAR)
    assert up.status_code == 200, up.text
    hole_id = {"pattern": r"A(\d+)", "template": template}
    for r in (
        client.post(f"{OPS}/uploads/collar/preview", json={"index": 0, "hole_id": hole_id}),
        client.post(f"{OPS}/import", json={"name": "holes", "hole_id": hole_id}),
    ):
        assert r.status_code == 400, r.text[:300]
        assert len(r.content) < 2000
        assert "hole id" in r.json()["detail"]


def test_hole_id_template_with_padding_still_works(env):
    client, _ = env
    client.put(f"{OPS}/uploads/collar", params={"filename": "collar.csv"}, content=COLLAR)
    hole_id = {"pattern": r"A(\d+)", "template": "DH-{1:03d}"}
    r = client.post(f"{OPS}/uploads/collar/preview", json={"index": 0, "hole_id": hole_id})
    assert r.status_code == 200, r.text
    assert [row[0] for row in r.json()["prepared"]["rows"]] == ["DH-001", "DH-002"]


# ------------------------------------------------------------------ 4. job log


def test_rejected_requests_are_stored_capped(env):
    client, folder = env
    huge = {"database": "x" * 200_000, "attributes": [f"a{i}" for i in range(50_000)],
            "nested": {"deep": ["y" * 5000] * 300}}  # fmt: skip
    assert client.post(f"{OPS}/composite", json=huge).status_code == 400
    line = (folder / joblog.LOG_FILE).read_text(encoding="utf-8").splitlines()[-1]
    assert len(line) < 8000
    entry = json.loads(line)
    assert entry["status"] == "error" and entry["operation"] == "composite"
    assert entry["params"]["database"].startswith("xxxx")
    assert "more)" in entry["params"]["database"]
    assert "more)" in str(entry["params"]["attributes"][-1])
    assert len(entry["error"]) < 2100


def test_log_endpoint_pages_newest_first(env):
    client, folder = env
    for i in range(7):
        joblog.Recorder(folder, "validate", {"database": f"db{i}"}).finish("done", summary={})
    page = client.get(f"{OPS}/log", params={"limit": 3}).json()
    assert page["total"] == 7 and page["has_more"] is True
    assert [j["params"]["database"] for j in page["jobs"]] == ["db6", "db5", "db4"]
    older = client.get(f"{OPS}/log", params={"limit": 3, "before": page["jobs"][-1]["id"]}).json()
    assert [j["params"]["database"] for j in older["jobs"]] == ["db3", "db2", "db1"]
    assert older["total"] == 7 and older["has_more"] is True
    last = client.get(f"{OPS}/log", params={"limit": 3, "before": older["jobs"][-1]["id"]}).json()
    assert [j["params"]["database"] for j in last["jobs"]] == ["db0"] and not last["has_more"]
    everything = client.get(f"{OPS}/log").json()  # default: the newest 200
    assert len(everything["jobs"]) == 7 and everything["has_more"] is False
    assert client.get(f"{OPS}/log", params={"limit": 0}).status_code == 422
    assert client.get(f"{OPS}/log", params={"before": "nope"}).status_code == 400
    for i in range(250):
        joblog.Recorder(folder, "validate", {"database": f"more{i}"}).finish("done", summary={})
    default = client.get(f"{OPS}/log").json()
    assert len(default["jobs"]) == 200 and default["total"] == 257 and default["has_more"]


def test_successful_entries_keep_their_parameters_for_replay(env):
    """The file keeps what replay reads; only the endpoint shortens what it shows."""
    client, folder = env
    attributes = [{"name": f"attribute_{i:04d}", "kind": "float"} for i in range(300)]
    r = client.post(f"{OPS}/models/model/code", json={"attributes": attributes, "steps": []})
    assert r.status_code == 200, r.text
    stored = [e for e in joblog.read(folder) if e["operation"] == "code"][-1]
    assert stored["params"]["attributes"] == attributes
    shown = client.get(f"{OPS}/log").json()["jobs"][0]
    assert shown["operation"] == "code" and len(shown["params"]["attributes"]) < 30
    assert "more)" in str(shown["params"]["attributes"][-1])


# ------------------------------------------------------------------ 6. unhandled 500s


def test_long_geometry_file_name_is_400_and_leaves_nothing_behind(env):
    client, folder = env
    r = client.put(f"{OPS}/geometry/uploads", params={"filename": "a" * 5000 + ".obj"},
                   content=b"v 0 0 0\n")  # fmt: skip
    assert r.status_code == 400, r.text[:300]
    assert "file name" in r.json()["detail"] and len(r.json()["detail"]) < 300
    uploads = folder / "geometry_uploads"
    assert not uploads.exists() or list(uploads.iterdir()) == []


def test_long_table_name_of_an_upload_is_400(env):
    client, folder = env
    for call in (
        lambda t: client.put(f"{OPS}/uploads/{t}", params={"filename": "t.csv"}, content=COLLAR),
        lambda t: client.post(f"{OPS}/uploads/{t}/preview", json={"index": 0}),
        lambda t: client.delete(f"{OPS}/uploads/{t}"),
    ):
        r = call("t" * 300)
        assert r.status_code == 400, r.text[:300]
        assert "table name" in r.json()["detail"]
    assert not any((folder / "uploads").glob("ttt*"))


def test_rejected_geometry_upload_leaves_no_folder_and_no_server_path(env):
    client, folder = env
    r = client.put(f"{OPS}/geometry/uploads", params={"filename": "broken.ply"},
                   content=b"this is not a mesh")  # fmt: skip
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert "broken.ply" in detail
    assert str(folder) not in detail and str(folder.resolve()) not in detail
    assert "geometry_uploads" not in detail
    uploads = folder / "geometry_uploads"
    assert not uploads.exists() or list(uploads.iterdir()) == []
    # a second file that fails does not remove the first of its token
    good = client.put(f"{OPS}/geometry/uploads", params={"filename": "points.csv"},
                      content=b"x,y,z\n1,2,3\n4,5,6\n7,8,1\n")  # fmt: skip
    assert good.status_code == 200, good.text
    token = good.json()["token"]
    bad = client.put(f"{OPS}/geometry/uploads", content=b"nope",
                     params={"filename": "broken.ply", "token": token})  # fmt: skip
    assert bad.status_code == 400
    assert [p.name for p in (uploads / token).iterdir()] == ["points.csv"]


@pytest.mark.parametrize("name", [["a"], {"a": 1}, 5, True, "n" * 500])
def test_names_must_be_short_strings(env, name):
    client, _ = env
    client.put(f"{OPS}/uploads/collar", params={"filename": "collar.csv"}, content=COLLAR)
    requests = [
        ("/import", {"name": name}),
        ("/composite", {"database": "nope", "name": name}),
        ("/models", {"name": name, "origin": [0, 0, 0], "size": [1, 1, 1], "shape": [2, 2, 2]}),
        ("/surfaces/from-points", {"name": name, "databases": ["nope"]}),
        ("/meshes/import", {"name": name, "upload": "0" * 12}),
        ("/solids/implicit", {"name": name, "database": "nope"}),
    ]
    for route, body in requests:
        r = client.post(f"{OPS}{route}", json=body)
        assert r.status_code == 400, (route, r.status_code, r.text[:200])
        assert "name" in r.json()["detail"], (route, r.json()["detail"])


# ------------------------------------------------------------------ 9. sub-block depth


def test_sub_block_depth_is_limited_to_what_is_supported(env, tmp_path, capsys):
    client, _ = env
    body = {"name": "deep", "origin": [0, 0, 0], "size": [8, 8, 8], "shape": [2, 2, 2]}
    ok = client.post(f"{OPS}/models", json=body | {"max_depth": 3})
    assert ok.status_code == 200, ok.text
    for depth in (4, 8, -1, 1.5, "x"):
        r = client.post(f"{OPS}/models", json=body | {"max_depth": depth})
        assert r.status_code == 400, (depth, r.text)
        assert "0 to 3" in r.json()["detail"]
    argv = ["blockmodel", "create", "--origin", "0,0,0", "--size", "8,8,8", "--shape", "2,2,2"]
    assert main([*argv, str(tmp_path / "m3"), "--max-depth", "3"]) == 0
    capsys.readouterr()
    assert main([*argv, str(tmp_path / "m4"), "--max-depth", "4"]) == 2
    assert "0 to 3" in capsys.readouterr().err
    assert not (tmp_path / "m4").exists()


def test_other_fields_of_the_wrong_type_are_400(env):
    """Found by sweeping the endpoints with lists / objects where text or a list belongs."""
    client, _ = env
    for route, body in [
        ("/composite", {"database": "nope", "table": ["a"]}),
        ("/models/model/code", {"attributes": ["a"]}),
        ("/models/model/code", {"attributes": "abc"}),
        ("/models/model/code", {"steps": ["a"]}),
        ("/models/model/code", {"steps": {"a": 1}}),
    ]:
        r = client.post(f"{OPS}{route}", json=body)
        assert r.status_code == 400, (route, body, r.status_code, r.text[:200])
