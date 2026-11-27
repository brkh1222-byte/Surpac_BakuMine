"""Taking the composition off again: ``geology uncode`` (the coded attributes of a block
model) and ``geology remove`` (the geology model itself), over the API, the CLI and replay.

A project on the small synthetic layered deposit, the geology built through the API from
``base_spec``. Independent checks: a block model with its coding removed against the
untouched copy made before any coding (blocks and values); the files of a geology folder
restored by undo against SHA-256 digests taken before the removal; tonnes of a kept
density against the sum over the blocks before the removal; the project rebuilt by
``bakumine project replay`` against the source project.

The tests of this module run in order and share the project.
"""

from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient
from test_geology_spec import base_spec

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.commands.replay import replay_steps
from bakumine.core import joblog
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, read_header, save_model
from bakumine.core.geology.coding import coding_record_path, read_coding_record
from bakumine.core.project import Project

OPS = "/api/v1/ops"
LAYER_ATTRIBUTES = ["fault_block", "lith", "weathering"]
USER_DENSITY = 2.5


def api_spec(cfg, units: dict) -> dict:
    """``base_spec`` as the API takes it: project objects by id, the dictionary inline."""
    spec = base_spec(cfg, faults=True)
    del spec["name"]
    spec.update(
        database={"object": "drillholes"},
        units=units,
        topography={"object": "topo", "source": "synthetic survey"},
        resolution=10.0,
    )
    return spec


def empty_model(cfg, *, density: bool) -> BlockModel:
    lo = (cfg.x_range[0], cfg.y_range[0], cfg.z_range[0])
    model = BlockModel(BlockModelDefinition(lo, (20.0, 20.0, 10.0), (20, 20, 31), max_depth=1))
    model.add_attribute("rock", "category", background="WASTE")
    model.add_attribute("au", "float")
    model.set("au", np.linspace(0.0, 3.0, model.n_blocks))
    if density:  # a density of the user's own, there before any coding
        model.add_attribute("density", "float", unit="t/m3")
        model.set("density", np.full(model.n_blocks, USER_DENSITY))
    return model


@pytest.fixture(scope="module")
def env(layered_small, layered_small_dir, layered_small_db, tmp_path_factory):
    base = tmp_path_factory.mktemp("geology_remove")
    folder = base / "project"
    project = Project.create(folder, "composition")
    project.add("database", layered_small_db.save(base / "db"), "Drillholes", "drillholes")
    project.add("dtm", layered_small_dir / "topo.obj", "Topography", "topo")
    for name, density in (("model", False), ("owned", True)):
        save_model(empty_model(layered_small.config, density=density), base / name)
        project.add("model", base / name, name.title(), name)
    set_scene_source(ProjectSceneSource(folder))
    units = json.loads((layered_small_dir / "units.json").read_text())
    client = TestClient(app)
    try:
        r = client.post(
            f"{OPS}/geology/build",
            json={"spec": api_spec(layered_small.config, units), "name": "Layered small"},
        )
        assert r.status_code == 200, r.text
        yield {"client": client, "folder": folder, "base": base, "gid": r.json()["id"]}
    finally:
        set_scene_source(None)


def same_model(a: BlockModel, b: BlockModel) -> None:
    """Same blocks, same attributes in the same order, same values (NaN equal to NaN)."""
    assert a.attribute_names == b.attribute_names
    np.testing.assert_array_equal(a.block_ids(), b.block_ids())
    for name in a.attribute_names:
        assert a.attribute(name).to_dict() == b.attribute(name).to_dict(), name
        np.testing.assert_array_equal(np.asarray(a[name]), np.asarray(b[name]), err_msg=name)


def digests(folder) -> dict[str, str]:
    return {
        p.relative_to(folder).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(folder.rglob("*"))
        if p.is_file()
    }


def post(env, operation: str, **body):
    return env["client"].post(f"{OPS}/geology/{operation}", json=body)


def model_path(env, model_id: str):
    return Project.open(env["folder"]).path_of(model_id)


def scene_object(env, object_id: str) -> dict:
    scene = env["client"].get("/api/v1/scene").json()
    return next(o for o in scene["objects"] if o["id"] == object_id)


def test_coding_records_what_it_wrote(env):
    client, gid = env["client"], env["gid"]
    assert read_coding_record(model_path(env, "model")) is None
    assert "composition" not in scene_object(env, "model")
    for model_id in ("model", "owned"):
        r = post(env, "code", model=model_id, geology=gid)
        assert r.status_code == 200, r.text
        assert r.json()["n_split"] > 0

    record = read_coding_record(model_path(env, "model"))
    assert record["geology"] == gid
    assert record["attributes"] == [
        {"name": "fault_block", "role": "fault_block", "created": True},
        {"name": "lith", "role": "lith", "created": True},
        {"name": "weathering", "role": "weathering", "created": True},
        {"name": "density", "role": "density", "created": True},
    ]
    owned = {
        a["name"]: a["created"] for a in read_coding_record(model_path(env, "owned"))["attributes"]
    }
    assert owned == {"fault_block": True, "lith": True, "weathering": True, "density": False}

    # the viewer is told which attribute holds which layer, and the names of the categories
    entry = scene_object(env, "model")
    assert entry["composition"] == {
        "geology": gid,
        "layers": {"fault_block": "fault_block", "lith": "lith", "weathering": "weathering"},
    }
    attributes = {a["name"]: a for a in entry["attributes"]}
    units = json.loads(
        (
            model_path(env, "model").parents[1] / "geology" / "layered-small" / "units.json"
        ).read_text()
    )
    names = {u["code"]: u["name"] for u in units["units"] if u.get("name")}
    assert attributes["lith"]["labels"]["SST"] == names["SST"] == "Sandstone"
    assert set(attributes["lith"]["labels"]) <= set(attributes["lith"]["categories"])
    assert "AIR" not in attributes["lith"]["labels"]
    assert attributes["weathering"]["labels"]["OXIDE"] == "Oxidised"
    assert "labels" not in attributes["fault_block"] and "labels" not in attributes["rock"]
    summary = client.get(f"{OPS}/project").json()
    models = {o["id"]: o for o in summary["objects"] if o["kind"] == "model"}
    assert models["model"]["geology_coding"]["layers"]["lith"] == "lith"
    assert models["model"]["geology_coding"]["attributes"] == record["attributes"]

    # coding again keeps what the first record says: the attributes are there now, but
    # the coding made them
    r = post(env, "code", model="model", geology=gid, layers=["lith"], density=None)
    assert r.status_code == 200, r.text
    assert read_coding_record(model_path(env, "model"))["attributes"] == [
        record["attributes"][1],
        record["attributes"][0],
        *record["attributes"][2:],
    ]


def test_uncode_gives_back_the_model_and_undo_gives_back_the_coding(env):
    client, gid, folder = env["client"], env["gid"], env["folder"]
    path = model_path(env, "model")
    pristine = load_model(env["base"] / "model")
    coded = load_model(path)
    assert coded.n_blocks > pristine.n_blocks
    assert set(coded.attribute_names) == {"rock", "au", *LAYER_ATTRIBUTES, "density"}
    geology_folder = Project.open(folder).path_of(gid)
    provenance = geology_folder / "coding" / "model.provenance.json"
    checksum = read_header(path).checksum
    record = coding_record_path(path).read_bytes()
    assert provenance.is_file()
    assert client.get(f"/api/v1/geology/{gid}").json()["coded_models"] == ["model", "owned"]

    r = post(env, "uncode", model="model")
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["removed"] == ["lith", "fault_block", "weathering", "density"]
    assert result["kept"] == {} and result["n_merged"] > 0 and result["geology"] == [gid]
    assert result["n_blocks"] == pristine.n_blocks
    same_model(load_model(path), pristine)  # blocks merged back, own values untouched
    assert not coding_record_path(path).exists() and not provenance.exists()
    assert client.get(f"/api/v1/geology/{gid}").json()["coded_models"] == ["owned"]
    assert "composition" not in scene_object(env, "model")
    summary = client.get(f"{OPS}/project").json()
    assert "geology_coding" not in next(o for o in summary["objects"] if o["id"] == "model")

    entry = joblog.jobs(folder)[-1]
    assert (entry["operation"], entry["status"], entry["modifies"]) == (
        "geology_uncode",
        "done",
        "model",
    )
    assert entry["snapshot"] is True
    assert entry["replay"] == {"attributes": result["removed"], "merge": True}
    assert {o["id"] for o in entry["inputs"]} >= {"model", gid}
    log = client.get(f"{OPS}/log").json()
    assert log["undo"] == entry["id"]

    # undo: the coded model, its record and the geology model's record are back
    undone = client.post(f"{OPS}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json() == {"undone": entry["id"], "operation": "geology_uncode", "model": "model"}
    assert read_header(path).checksum == checksum
    same_model(load_model(path), coded)
    assert coding_record_path(path).read_bytes() == record and provenance.is_file()
    assert scene_object(env, "model")["composition"]["layers"]["lith"] == "lith"

    # without merging the blocks stay split; then a second removal has nothing to do
    r = post(env, "uncode", model="model", merge=False)
    assert r.status_code == 200, r.text
    assert r.json()["n_merged"] == 0 and r.json()["n_blocks"] == coded.n_blocks
    left = load_model(path)
    assert left.attribute_names == ["rock", "au"]
    np.testing.assert_array_equal(left.block_ids(), coded.block_ids())
    np.testing.assert_array_equal(left["au"], coded["au"])


def test_undo_of_the_coding_takes_its_records_back(env):
    client, gid = env["client"], env["gid"]
    path = model_path(env, "model")
    before = load_model(path)
    provenance = Project.open(env["folder"]).path_of(gid) / "coding" / "model.provenance.json"
    assert post(env, "code", model="model", geology=gid).status_code == 200
    assert coding_record_path(path).is_file() and provenance.is_file()
    assert client.post(f"{OPS}/undo").status_code == 200
    same_model(load_model(path), before)
    assert not coding_record_path(path).exists() and not provenance.exists()
    # coded again: the state the later tests start from
    assert post(env, "code", model="model", geology=gid).status_code == 200


def test_a_density_the_model_had_before_survives(env):
    path = model_path(env, "owned")
    coded = load_model(path)
    tonnes = float(np.nansum(coded.volumes() * np.asarray(coded["density"])))
    assert not np.allclose(np.asarray(coded["density"])[~np.isnan(coded["density"])], USER_DENSITY)
    r = post(env, "uncode", model="owned")
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["removed"] == LAYER_ATTRIBUTES
    assert list(result["kept"]) == ["density"]
    assert "had this attribute before the geology coding" in result["kept"]["density"]
    left = load_model(path)
    assert left.attribute_names == ["rock", "au", "density"]
    assert left.attribute("density").unit == "t/m3"
    # the values are the coded ones (said in "kept"): no tonne is lost by the merge
    after = float(np.nansum(left.volumes() * np.asarray(left["density"])))
    assert after == pytest.approx(tonnes, rel=1e-12)
    assert left.volumes().sum() == pytest.approx(coded.volumes().sum(), rel=1e-12)
    assert not coding_record_path(path).exists()


def test_removing_what_is_not_there_names_it_and_changes_nothing(env):
    folder, gid = env["folder"], env["gid"]
    before = (Project.open(folder).list(), read_header(model_path(env, "owned")).checksum)
    cases = [
        ("uncode", {"model": "owned"}, "block model 'owned' has no geology coding to remove"),
        ("uncode", {"model": "nope"}, "no block model 'nope'"),
        ("uncode", {"model": gid}, f"no block model {gid!r}"),
        ("uncode", {}, "model: the id of the block model"),
        (
            "uncode",
            {"model": "owned", "attributes": ["au", "nope"]},
            "block model 'owned': attributes: the block model has no attribute 'nope'",
        ),
        ("uncode", {"model": "owned", "attributes": "au"}, "attributes must be a list"),
        ("uncode", {"model": "model", "merge": "yes"}, "merge must be true or false"),
        ("remove", {"geology": "nope"}, "no geology model 'nope' in the project"),
        ("remove", {"geology": "topo"}, "no geology model 'topo' in the project"),
        ("remove", {}, "no geology model None in the project"),
    ]
    for operation, body, text in cases:
        r = post(env, operation, **body)
        assert r.status_code == 400, (operation, body, r.text)
        assert text in r.json()["detail"], r.json()["detail"]
    assert (Project.open(folder).list(), read_header(model_path(env, "owned")).checksum) == before
    failed = joblog.jobs(folder)[-len(cases) :]
    assert [e["status"] for e in failed] == ["error"] * len(cases)
    assert not any(e["snapshot"] for e in failed)
    assert not any(joblog.snapshot_dir(folder, e["id"]).exists() for e in failed)

    # a record whose attributes are all gone from the model: nothing to remove
    path = model_path(env, "owned")
    coding_record_path(path).write_text(
        json.dumps(
            {
                "format": "bakumine.geology-coding",
                "version": 1,
                "model": "owned",
                "geology": gid,
                "attributes": [{"name": "lith", "role": "lith", "created": True}],
            }
        )
    )
    r = post(env, "uncode", model="owned")
    assert r.status_code == 400 and "nothing to remove" in r.json()["detail"], r.text
    assert "lith" in r.json()["detail"]
    coding_record_path(path).unlink()


def test_remove_takes_the_geology_out_and_undo_brings_it_back(env, monkeypatch):
    client, gid, folder = env["client"], env["gid"], env["folder"]
    project = Project.open(folder)
    obj, geology_folder = project.get(gid), project.path_of(gid)
    files = digests(geology_folder)
    assert len(files) > 20 and "geology.json" in files and "coding/model.provenance.json" in files
    children = [
        o["id"] for o in client.get("/api/v1/scene").json()["objects"] if o.get("parent") == gid
    ]
    assert len(children) >= 5
    coded = load_model(model_path(env, "model"))
    assert "labels" in next(
        a for a in scene_object(env, "model")["attributes"] if a["name"] == "lith"
    )

    r = post(env, "remove", geology=gid)
    assert r.status_code == 200, r.text
    assert r.json() == {
        "removed": gid,
        "name": "Layered small",
        "path": obj.path,
        "undoable": True,
        "coded_models": ["model"],
    }
    assert Project.open(folder).list("geology") == []
    assert not geology_folder.exists() and list((folder / "geology").iterdir()) == []
    scene = client.get("/api/v1/scene").json()
    assert not [o for o in scene["objects"] if o["category"] == "geology"]
    assert client.get(f"/api/v1/geology/{gid}").status_code == 404
    assert client.get(f"/api/v1/scene/objects/{children[0]}/mesh").status_code == 404
    # the coded block model keeps its attributes and its record; the dictionary colours
    # and names went with the geology model
    same_model(load_model(model_path(env, "model")), coded)
    lith = next(a for a in scene_object(env, "model")["attributes"] if a["name"] == "lith")
    assert "labels" not in lith and "colours" not in lith
    assert scene_object(env, "model")["composition"]["layers"]["lith"] == "lith"

    entry = joblog.jobs(folder)[-1]
    assert (entry["operation"], entry["status"]) == ("geology_remove", "done")
    assert entry["removes"] == [obj.to_dict()] and entry["inputs"] == [obj.to_dict()]
    assert entry["modifies"] == gid and entry["snapshot"] is True
    assert post(env, "remove", geology=gid).status_code == 400  # it is gone
    assert client.get(f"{OPS}/log").json()["undo"] == entry["id"]

    undone = client.post(f"{OPS}/undo")
    assert undone.status_code == 200, undone.text
    assert undone.json()["operation"] == "geology_remove"
    assert Project.open(folder).get(gid) == obj  # same id, name and path
    assert digests(geology_folder) == files  # every file, byte for byte
    scene = client.get("/api/v1/scene").json()
    assert [o["id"] for o in scene["objects"] if o.get("parent") == gid] == children
    assert client.get(f"/api/v1/geology/{gid}").json()["coded_models"] == ["model"]
    assert not joblog.snapshot_dir(folder, entry["id"]).exists()

    # the coding can still be taken off after the geology model is gone for good
    monkeypatch.setenv("BAKUMINE_UNDO_SNAPSHOTS", "0")
    r = post(env, "remove", geology=gid)
    assert r.status_code == 200 and r.json()["undoable"] is False, r.text
    assert not geology_folder.exists() and Project.open(folder).list("geology") == []
    assert joblog.jobs(folder)[-1]["snapshot"] is False
    # not undoable: Undo names the newest model change before it instead
    assert client.get(f"{OPS}/log").json()["undo_job"]["modifies"] == "owned"
    monkeypatch.delenv("BAKUMINE_UNDO_SNAPSHOTS")
    r = post(env, "uncode", model="model")
    assert r.status_code == 200, r.text
    assert r.json()["removed"] == ["fault_block", "lith", "weathering", "density"]
    assert r.json()["geology"] == []
    same_model(load_model(model_path(env, "model")), load_model(env["base"] / "model"))


def test_replay_reproduces_the_project(env, tmp_path, capsys):
    """Runs last: the job log, with its removals and undone jobs, redone by the CLI."""
    folder, gid = env["folder"], env["gid"]
    done = [
        e
        for e in joblog.jobs(folder)
        if e["status"] == "done" and not e["undone"] and e["operation"] != "undo"
    ]
    assert [e["operation"] for e in done] == [
        "geology_build",
        "geology_code",
        "geology_code",
        "geology_code",
        "geology_uncode",
        "geology_code",
        "geology_uncode",
        "geology_remove",
        "geology_uncode",
    ]
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(folder), "--into", str(target)]) == 0, capsys.readouterr()
    out = capsys.readouterr().out
    assert f"geology remove {target} {gid}" in out and "geology uncode" in out
    source, replayed = Project.open(folder), Project.open(target)
    assert [o.to_dict() for o in replayed.list()] == [o.to_dict() for o in source.list()]
    assert replayed.list("geology") == [] and list((target / "geology").iterdir()) == []
    for model_id in ("model", "owned"):
        same_model(load_model(replayed.path_of(model_id)), load_model(source.path_of(model_id)))
        assert not coding_record_path(replayed.path_of(model_id)).exists()
    same_model(load_model(replayed.path_of("model")), load_model(env["base"] / "model"))


def test_replay_skips_the_removal_of_an_object_the_log_did_not_make():
    obj = {"id": "by-hand", "kind": "geology", "name": "By hand", "path": "geology/by-hand"}
    entry = {
        "id": "j1",
        "time": "t",
        "operation": "geology_remove",
        "status": "done",
        "params": {"geology": "by-hand"},
        "inputs": [obj],
        "outputs": [],
        "removes": [obj],
    }
    steps = replay_steps([entry])
    assert [s["kind"] for s in steps] == ["run", "comment"]
    assert "by-hand was not made by this log" in steps[1]["text"]
    # made by the log, removed, made again under the same id: add, remove, add
    built = {
        "id": "j0",
        "time": "t",
        "operation": "geology_build",
        "status": "done",
        "params": {},
        "inputs": [],
        "outputs": [obj],
        "replay": {"spec": {"name": "x"}},
    }
    steps = replay_steps([built, entry, dict(built, id="j2")])
    runs = [s["argv"][:2] for s in steps if s["kind"] == "run"]
    assert runs == [
        ["project", "init"],
        ["geology", "build"],
        ["project", "add"],
        ["geology", "remove"],
        ["geology", "build"],
        ["project", "add"],
    ]


def test_cli_uncode_and_remove(
    env, layered_small, layered_small_dir, layered_small_db, tmp_path, capsys
):
    """The command line on its own folders: a geology folder built by ``geology build``."""
    spec = base_spec(layered_small.config, faults=False)
    spec.update(
        resolution=10.0,
        database=str(layered_small_db.save(tmp_path / "db")),
        units=str(layered_small_dir / "units.json"),
        topography={"file": str(layered_small_dir / "topo.obj"), "source": "synthetic"},
    )
    (tmp_path / "spec.json").write_text(json.dumps(spec))
    geology = tmp_path / "small.geology"
    assert (
        main(["geology", "build", "--spec", str(tmp_path / "spec.json"), "--out", str(geology)])
        == 0
    ), capsys.readouterr()
    model = tmp_path / "owned"
    shutil.copytree(env["base"] / "owned", model)
    pristine = load_model(model)

    assert main(["geology", "uncode", str(model)]) == 2
    assert "has no geology coding to remove" in capsys.readouterr().err
    assert main(["geology", "code", str(model), "--geology", str(geology)]) == 0
    assert "owned.geology-coding.json" in capsys.readouterr().out
    record = read_coding_record(model)
    assert {a["name"]: a["created"] for a in record["attributes"]} == {
        "lith": True,
        "weathering": True,
        "density": False,  # one fault block: not coded
    }
    assert record["geology"] == "small"
    provenance = geology / "coding" / "owned.provenance.json"
    assert provenance.is_file()

    # a model coded before records were kept: the layers count as the coding's, the
    # density does not; naming an attribute that is not there is an error naming it
    coding_record_path(model).unlink()
    assert main(["geology", "uncode", str(model), "--attributes", "lith,nope"]) == 2
    assert "no attribute 'nope'" in capsys.readouterr().err
    assert main(["geology", "code", str(model), "--geology", str(geology)]) == 0
    assert {a["name"]: a["created"] for a in read_coding_record(model)["attributes"]} == {
        "lith": True,
        "weathering": True,
        "density": False,
    }
    coded = load_model(model)
    capsys.readouterr()

    assert main(["geology", "uncode", str(model), "--geology", str(geology)]) == 0
    out = capsys.readouterr().out
    assert "removed lith, weathering" in out and "kept density" in out
    left = load_model(model)
    assert left.attribute_names == pristine.attribute_names
    assert left.n_blocks < coded.n_blocks
    np.testing.assert_array_equal(
        np.unique(np.asarray(left["rock"])), np.unique(np.asarray(pristine["rock"]))
    )
    assert not coding_record_path(model).exists() and not provenance.exists()
    assert main(["geology", "uncode", str(model)]) == 2
    capsys.readouterr()

    # explicit attributes on a model without a record
    assert main(["geology", "uncode", str(model), "--attributes", "au", "--no-merge"]) == 0
    assert load_model(model).attribute_names == ["rock", "density"]
    capsys.readouterr()

    project = Project.create(tmp_path / "project", "p")
    project.add("geology", geology, "Small", "small")
    project.add("model", model, "Owned", "owned")
    for bad in ("nope", "owned"):
        assert main(["geology", "remove", str(tmp_path / "project"), bad]) == 2
        err = capsys.readouterr().err
        assert f"no geology model {bad!r} in the project" in err and "small" in err
    assert main(["geology", "remove", str(tmp_path / "project"), "small"]) == 0
    assert "removed geology 'small'" in capsys.readouterr().out
    after = Project.open(tmp_path / "project")
    assert [o.id for o in after.list()] == ["owned"]
    assert not (tmp_path / "project" / "geology" / "small.geology").exists()
