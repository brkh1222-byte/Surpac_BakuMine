"""Geology over the API (P9.T14): build, code, the scene manifest and job-log replay.

A project on the small synthetic layered deposit. Independent checks: the endpoint's
model against the CLI's from the same spec (identical labels), against the truth of the
deposit (IoU), and against the model rebuilt by ``bakumine project replay``; the manifest
against the files of the geology folder and the unit dictionary; the coded block model
against ``bakumine geology code`` on a copy.
"""

from __future__ import annotations

import json
import shutil

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.binary import unpack_arrays
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.geology.model import LAYERS, GeologyModel
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.io.obj import read_obj
from bakumine.core.project import Project
from bakumine.synth.layered import LAYERED_PRESETS
from bakumine.synth.layered_generate import generate_layered, import_layered

OPS = "/api/v1/ops"
GROUP = ("SST", "SLT", "AND", "DIO", "GRD")


def api_spec(cfg, units: dict) -> dict:
    """The build spec as the panel sends it: project objects by id, the dictionary inline."""
    return {
        "format": "bakumine.geology-spec", "version": 1,
        "database": {"object": "drillholes"},
        "units": units,
        "topography": {"object": "topo", "source": "synthetic survey"},
        "extent": {"model": {"object": "model"}, "padding": 0},
        "resolution": 5.0,
        "faults": {
            "blocks": {"table": "assay", "column": "wf", "pattern": r"^(FB\d)-"},
            "post_fault": ["DIO", "GRD"],
            "surfaces": [{"name": "F1", "between": ["FB1", "FB2"]}],
        },
        "lithology": {
            "events": [
                {"mode": "sequence", "units": ["SST", "SLT", "AND"]},
                {"mode": "intrusion", "units": ["DIO"]},
                {"mode": "intrusion", "units": ["GRD"]},
            ]
        },
        "cover": {"unit": "Q4"},
        "weathering": [
            {"zone": "OXIDE",
             "base": [{"units": ["OX"]},
                      {"threshold": {"table": "assay", "column": "s_ppm", "below": 1000}}]},
            {"zone": "TRANSITION",
             "base": [{"threshold": {"table": "assay", "column": "s_ppm", "below": 3000}}]},
        ],
        "interpolation": {"spacing": 10.0},
    }  # fmt: skip


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    base = tmp_path_factory.mktemp("geology_api")
    deposit = generate_layered(LAYERED_PRESETS["layered-small"], seed=42)
    deposit.write(base / "synth")
    folder = base / "project"
    project = Project.create(folder, "geology")
    project.add("database", import_layered(base / "synth").save(base / "db"), "Drillholes",
                "drillholes")  # fmt: skip
    project.add("dtm", base / "synth" / "topo.obj", "Topography", "topo")
    project.add("dtm", base / "synth" / "truth_fault_F1.obj", "Fault F1, mapped", "f1-mapped")
    cfg = deposit.config
    lo = (cfg.x_range[0], cfg.y_range[0], cfg.z_range[0])
    definition = BlockModelDefinition(lo, (20.0, 20.0, 10.0), (20, 20, 31), max_depth=1)
    model = BlockModel(definition)
    model.add_attribute("rock", "category", background="WASTE")
    save_model(model, base / "model")
    project.add("model", base / "model", "Model", "model")
    shutil.copytree(base / "model", base / "model_copy" / "model")  # for the CLI comparison
    set_scene_source(ProjectSceneSource(folder))
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    units = json.loads((base / "synth" / "units.json").read_text())
    try:
        yield {"client": TestClient(app), "folder": folder, "base": base, "deposit": deposit,
               "box": (np.array(lo, dtype=float), hi), "units": units,
               "spec": api_spec(cfg, units)}  # fmt: skip
    finally:
        set_scene_source(None)


def geology_folder(env, object_id: str):
    return Project.open(env["folder"]).path_of(object_id)


def without_time(document):
    """A JSON document without the fields that differ from run to run."""
    if isinstance(document, dict):
        return {k: without_time(v) for k, v in document.items()
                if k not in ("created", "timing")}  # fmt: skip
    if isinstance(document, list):
        return [without_time(v) for v in document]
    return document


def test_build_refuses_a_bad_request_and_adds_nothing(env):
    client, spec = env["client"], env["spec"]
    cases = [
        ({"spec": dict(spec, resolution=-1), "name": "bad"}, "resolution: must be > 0"),
        ({"spec": dict(spec, database={"object": "nope"})}, "no database 'nope'"),
        ({"spec": dict(spec, database="db")}, "no database 'db'"),
        ({"spec": dict(spec, topography={"object": "drillholes"})}, "no dtm 'drillholes'"),
        ({"spec": dict(spec, units="units.json")}, "spec.units: give the unit dictionary"),
        ({"spec": dict(spec, extra_intervals=[{"file": "a.csv"}])}, "give the intervals as 'rows'"),
        ({"spec": dict(spec, colour="red")}, "unknown key(s) colour"),
        ({"spec": 5}, "spec must be an object"),
        ({"spec": spec, "name": 5}, "name must be text"),
    ]
    for body, text in cases:
        r = client.post(f"{OPS}/geology/build", json=body)
        assert r.status_code == 400, r.text
        assert text in r.json()["detail"], r.json()["detail"]
    assert Project.open(env["folder"]).list("geology") == []
    log = joblog.jobs(env["folder"])
    assert [e["status"] for e in log] == ["error"] * len(cases)
    assert client.get("/api/v1/geology/nope").status_code == 404
    assert client.get("/api/v1/geology/drillholes/provenance").status_code == 404


def test_build_endpoint_equals_the_cli_and_meets_the_truth(env, tmp_path, capsys):
    client, folder = env["client"], env["folder"]
    r = client.post(f"{OPS}/geology/build", json={"spec": env["spec"], "name": "Layered small"})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["kind"] == "geology" and result["name"] == "Layered small"
    assert result["layers"]["lith"] == [*GROUP, "Q4", "AIR"]
    assert result["blocks"] == ["FB1", "FB2"] and result["faults"][0]["name"] == "F1"
    assert result["provenance"]["measured"] > 0 and result["provenance"]["missing"] > 0
    obj = Project.open(folder).get(result["id"])
    assert (obj.kind, obj.path, obj.name) == ("geology", "geology/layered-small", "Layered small")
    env["id"] = result["id"]
    api_folder = geology_folder(env, obj.id)
    api = GeologyModel.load(api_folder)

    entry = joblog.jobs(folder)[-1]
    assert entry["operation"] == "geology_build" and entry["status"] == "done"
    assert [o["id"] for o in entry["inputs"]] == ["drillholes", "topo", "model"]
    assert entry["outputs"] == [obj.to_dict()]
    logged = entry["replay"]["spec"]
    paths = {o.id: o.path for o in Project.open(folder).list()}
    assert logged["database"] == paths["drillholes"]
    assert logged["extent"]["model"] == paths["model"]
    assert logged["topography"] == {"file": paths["topo"], "source": "synthetic survey"}
    assert json.loads((api_folder / "spec.json").read_text()) == logged

    # the CLI on the same inputs, written by hand: files by path, the dictionary as a file
    base = env["base"]
    cli = dict(
        env["spec"],
        name="Layered small",
        database=f"project/{paths['drillholes']}",
        units="synth/units.json",
        topography={"file": f"project/{paths['topo']}", "source": "synthetic survey"},
        extent={"model": f"project/{paths['model']}", "padding": 0},
    )
    (base / "cli_spec.json").write_text(json.dumps(cli), encoding="utf-8")
    out = tmp_path / "cli.geology"
    assert main(["geology", "build", "--spec", str(base / "cli_spec.json"), "--out", str(out)]) == 0
    capsys.readouterr()
    by_cli = GeologyModel.load(out)
    lo, hi = env["box"]
    pts = np.random.default_rng(8).uniform(lo - 10.0, hi + 10.0, size=(30000, 3))
    for layer in LAYERS:
        np.testing.assert_array_equal(api.labels(pts, layer), by_cli.labels(pts, layer))
    np.testing.assert_array_equal(
        api.labels(pts[:3000], exact=True), by_cli.labels(pts[:3000], exact=True)
    )
    for name in ("geology.json", "diagnostics.json"):
        a = json.loads((api_folder / name).read_text())
        b = json.loads((out / name).read_text())
        assert without_time(a) == without_time(b), name
    a = json.loads((api_folder / "provenance.json").read_text())["entries"]
    b = json.loads((out / "provenance.json").read_text())["entries"]
    differing = [x["item"] for x, y in zip(a, b, strict=True) if x != y]
    assert differing == ["Rock-type log"]  # names how the dictionary was given, nothing else
    assert sorted(p.name for p in api_folder.rglob("*") if p.is_file()) == sorted(
        p.name for p in out.rglob("*") if p.is_file()
    )

    # and against the truth of the deposit (geology_contracts.md 14), inside the drilled hull
    from scipy.spatial import Delaunay

    axes = [np.arange(a + 2.5, b, 5.0) for a, b in zip(lo, hi, strict=True)]
    grid = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    grid = grid[Delaunay(api.hole_points).find_simplex(grid) >= 0]
    true = env["deposit"].truth.labels(grid, "lith")
    got = api.decode(api.labels(grid), "lith")
    ground = (true != "AIR") & (got != "AIR")
    scores = {}
    for u in GROUP:
        both, either = (got == u) & (true == u) & ground, ((got == u) | (true == u)) & ground
        scores[u] = float(both.sum() / max(either.sum(), 1))
    print("\nAPI build, IoU in the hull: " + ", ".join(f"{u} {v:.3f}" for u, v in scores.items()))
    assert min(scores[u] for u in ("SST", "SLT", "AND")) >= 0.85
    assert scores["DIO"] >= 0.80 and scores["GRD"] >= 0.50


def test_geology_endpoints_and_scene_manifest(env):
    client, gid = env["client"], env["id"]
    folder = geology_folder(env, gid)
    manifest = json.loads((folder / "geology.json").read_text())
    dictionary = UnitDictionary.load(folder / "units.json")
    colours = dictionary.colours()

    info = client.get(f"/api/v1/geology/{gid}")
    assert info.status_code == 200, info.text
    info = info.json()
    assert {"name", "layers", "blocks", "faults", "surfaces", "diagnostics"} <= set(info)
    assert info["name"] == "Layered small" and info["layers"] == manifest["layers"]
    assert info["blocks"] == ["FB1", "FB2"] and info["pooled_blocks"] == []
    assert [u["code"] for u in info["units"]] == [*GROUP, "Q4"]
    assert info["units"][0] == {"code": "SST", "name": "Sandstone", "kind": "lithology",
                                "colour": "#c9b28f", "mesh": f"{gid}.unit.SST"}  # fmt: skip
    assert info["units"][-1]["mesh"] is None  # the cover has a base surface, not a mesh
    fault = info["faults"][0]
    assert fault["name"] == "F1" and fault["source"] == "inferred-plane"
    assert fault["mesh"] == f"{gid}.fault.F1" and fault["blocks"] == ["FB1", "FB2"]
    assert fault["dip"] == pytest.approx(65.0, abs=5.0)
    assert [(s["name"], s["role"], s["mesh"]) for s in info["surfaces"]] == [
        ("base:Q4", "cover", f"{gid}.surface.base_Q4"),
        ("base:OXIDE", "weathering", f"{gid}.surface.base_OXIDE"),
        ("base:TRANSITION", "weathering", f"{gid}.surface.base_TRANSITION"),
    ]
    assert info["surfaces"][1]["colour"] == "#d98c3a"
    assert info["surfaces"][1]["diagnostics"]["used"]["exact"] == 64
    assert info["diagnostics"] == json.loads((folder / "diagnostics.json").read_text())
    assert info["spec"]["name"] == "Layered small" and info["coded_models"] == []

    record = client.get(f"/api/v1/geology/{gid}/provenance").json()
    assert record["build"] == json.loads((folder / "provenance.json").read_text())
    assert record["coding"] == {}
    assert {e["kind"] for e in record["build"]["entries"]} == {"measured", "assumed", "missing"}

    scene = client.get("/api/v1/scene").json()
    objects = {o["id"]: o for o in scene["objects"]}
    entry = objects[gid]
    grid = manifest["grid"]
    top = [o + s * (n - 1) for o, s, n in zip(grid["origin"], grid["spacing"], grid["shape"],
                                              strict=True)]  # fmt: skip
    assert entry == {
        "id": gid, "name": "Layered small", "category": "geology", "kind": "geology",
        "attributes": [], "href": f"/api/v1/geology/{gid}",
        "bounds": {"min": grid["origin"], "max": top},
        "stats": {"units": 6, "blocks": 2, "faults": 1},
    }  # fmt: skip
    children = [o for o in scene["objects"] if o.get("parent") == gid]
    assert [o["id"] for o in children] == [
        *(f"{gid}.unit.{u}" for u in GROUP), f"{gid}.fault.F1",
        f"{gid}.surface.base_Q4", f"{gid}.surface.base_OXIDE", f"{gid}.surface.base_TRANSITION",
    ]  # fmt: skip
    assert [o["group"] for o in children] == ["Units"] * 5 + ["Faults"] + ["Surfaces"] * 3
    assert all(o["category"] == "geology" and o["kind"] == "mesh" for o in children)
    for unit in GROUP:
        child = objects[f"{gid}.unit.{unit}"]
        assert child["style"] == {"colour": colours[unit]}
        assert child["name"] == dictionary.unit(unit).name
        assert child["href"] == f"/api/v1/scene/objects/{gid}.unit.{unit}/mesh"
        mesh = read_obj(folder / manifest["meshes"]["lith"][unit])
        assert child["stats"] == {"vertices": mesh.n_vertices, "faces": mesh.n_faces,
                                  "closed": True}  # fmt: skip
    assert "style" not in objects[f"{gid}.fault.F1"]
    assert objects[f"{gid}.surface.base_Q4"]["style"] == {"colour": colours["Q4"]}
    assert objects[f"{gid}.surface.base_OXIDE"]["name"] == "Base of Oxidised"
    assert "parent" not in objects["topo"] and "style" not in objects["topo"]

    # the meshes come from the existing mesh endpoint, bit for bit the files of the folder
    for object_id, file in ((f"{gid}.unit.DIO", manifest["meshes"]["lith"]["DIO"]),
                            (f"{gid}.fault.F1", "faults/F1.obj"),
                            (f"{gid}.surface.base_OXIDE", "surfaces/base_OXIDE.obj")):  # fmt: skip
        r = client.get(f"/api/v1/scene/objects/{object_id}/mesh")
        assert r.status_code == 200, object_id
        meta, arrays = unpack_arrays(r.content)
        mesh = read_obj(folder / file)
        assert meta["meta"]["object_id"] == object_id
        np.testing.assert_array_equal(arrays["vertices"], mesh.vertices)
        np.testing.assert_array_equal(arrays["faces"], mesh.faces)
    for bad in (f"{gid}.unit.XXX", f"{gid}.unit", f"{gid}.layer.SST", "nope.unit.SST", gid):
        assert client.get(f"/api/v1/scene/objects/{bad}/mesh").status_code == 404, bad

    # category colours of the drillhole log: every spelling takes the colour of its unit
    lith = next(a for a in objects["drillholes"]["attributes"]
                if a["table"] == "geology" and a["name"] == "lith")  # fmt: skip
    assert set(lith["colours"]) <= set(lith["categories"])
    for spelling, colour in lith["colours"].items():
        assert colour == colours[dictionary.lookup()[dictionary.fold(spelling)]], spelling
    assert lith["colours"]["SOIL"] == "#d8c37a" and lith["colours"]["Sst"] == "#c9b28f"
    assert set(lith["categories"]) - set(lith["colours"]) == set()
    wf = next(a for a in objects["drillholes"]["attributes"] if a["name"] == "wf")
    assert "colours" not in wf


def test_code_endpoint_equals_the_cli(env, capsys):
    client, gid, folder = env["client"], env["id"], env["folder"]
    for body, text in (
        ({"model": "nope", "geology": gid}, "no block model 'nope'"),
        ({"model": "model", "geology": "topo"}, "no geology model 'topo'"),
        ({"model": "model", "geology": gid, "layers": "lith"}, "layers must be a list"),
        ({"model": "model", "geology": gid, "layers": ["rock"]}, "rock"),
    ):
        r = client.post(f"{OPS}/geology/code", json=body)
        assert r.status_code == 400 and text in r.json()["detail"], r.text
    body = {
        "model": "model",
        "geology": gid,
        "layers": ["fault_block", "lith", "weathering"],
        "subblock": True,
        "density": "density",
        "display": "geo",
        "ore_attribute": "rock",
    }
    r = client.post(f"{OPS}/geology/code", json=body)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["attributes"] == {"fault_block": "fault_block", "lith": "lith",
                                    "weathering": "weathering"}  # fmt: skip
    assert result["n_split"] > 0 and result["skipped"] == {}
    assert {row["unit"] for row in result["tonnage"]} >= set(GROUP)
    entry = joblog.jobs(folder)[-1]
    assert entry["operation"] == "geology_code" and entry["modifies"] == "model"
    assert {o["kind"] for o in entry["inputs"]} >= {"model", "geology"}

    project = Project.open(folder)
    coded = load_model(project.path_of("model"))
    assert {"fault_block", "lith", "weathering", "density", "geo"} <= set(coded.attributes)
    # the CLI on the untouched copy of the model, with the same geology folder
    copy = env["base"] / "model_copy" / "model"
    assert main(["geology", "code", str(copy), "--geology", str(geology_folder(env, gid)),
                 "--display", "geo", "--ore-attribute", "rock"]) == 0  # fmt: skip
    capsys.readouterr()
    by_cli = load_model(copy)
    assert by_cli.n_blocks == coded.n_blocks == result["n_blocks"]
    np.testing.assert_array_equal(by_cli.block_ids(), coded.block_ids())
    for name in ("fault_block", "lith", "weathering", "geo"):
        assert by_cli.attribute(name).categories == coded.attribute(name).categories
        np.testing.assert_array_equal(by_cli[name], coded[name])
    np.testing.assert_array_equal(by_cli["density"], coded["density"])

    record = client.get(f"/api/v1/geology/{gid}/provenance").json()
    assert list(record["coding"]) == ["model"]
    assert record["coding"]["model"]["object"]["kind"] == "geology coding"
    assert client.get(f"/api/v1/geology/{gid}").json()["coded_models"] == ["model"]

    # the block attributes carry the dictionary's colours for their categories
    scene = client.get("/api/v1/scene").json()
    blocks = next(o for o in scene["objects"] if o["id"] == "model")
    attributes = {a["name"]: a for a in blocks["attributes"]}
    colours = UnitDictionary.load(geology_folder(env, gid) / "units.json").colours()
    assert attributes["lith"]["colours"] == {u: colours[u] for u in (*GROUP, "Q4")}
    assert attributes["weathering"]["colours"] == {
        z: colours[z] for z in ("OXIDE", "TRANSITION", "FRESH")
    }
    assert attributes["geo"]["colours"]["ORE"] == "#e8c547"
    assert "colours" not in attributes["fault_block"] and "colours" not in attributes["rock"]


def test_build_with_a_fault_surface_of_the_project(env):
    """The geologist's fault is a project object: it replaces the inferred plane, is an
    input of the job, and the units dictionary can be taken from an existing geology
    model. The second model lands beside the first and wins nothing from it."""
    client, folder, gid = env["client"], env["folder"], env["id"]
    spec = json.loads(json.dumps(env["spec"]))
    spec.update(resolution=10.0, units={"object": gid}, cover=None, weathering=[])
    spec["faults"]["surfaces"][0]["replace_with"] = {"file": {"object": "f1-mapped"}}
    r = client.post(f"{OPS}/geology/build", json={"spec": spec, "name": "Mapped fault"})
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["faults"][0]["source"] == "imported-mesh"
    assert result["faults"][0]["dip"] == pytest.approx(65.0, abs=0.01)
    assert result["layers"]["weathering"] == ["OXIDE", "TRANSITION", "FRESH", "AIR"]
    entry = joblog.jobs(folder)[-1]
    assert [o["id"] for o in entry["inputs"]] == ["drillholes", "topo", "model", gid, "f1-mapped"]
    paths = {o.id: o.path for o in Project.open(folder).list()}
    logged = entry["replay"]["spec"]
    assert logged["units"] == f"{paths[gid]}/units.json"
    assert logged["faults"]["surfaces"][0]["replace_with"] == {"file": paths["f1-mapped"]}
    env["second"] = result["id"]
    record = client.get(f"/api/v1/geology/{result['id']}/provenance").json()["build"]
    items = {e["item"]: e for e in record["entries"]}
    assert "Fault surfaces" not in items  # nothing inferred, so nothing asked for
    assert {"Cover base", "Weathering bases"} <= {
        e["item"] for e in record["entries"] if e["kind"] == "missing"
    }
    scene = client.get("/api/v1/scene").json()
    parents = {o["id"]: o.get("parent") for o in scene["objects"]}
    assert parents[f"{result['id']}.fault.F1"] == result["id"] and parents[f"{gid}.fault.F1"] == gid
    assert f"{result['id']}.surface.base_Q4" not in parents


def test_replay_rebuilds_the_same_geology_and_coding(env, tmp_path, capsys):
    """Runs last: the job log is redone by the CLI into a new project."""
    folder, gid = env["folder"], env["id"]
    done = [e for e in joblog.jobs(folder) if e["status"] == "done"]
    assert [e["operation"] for e in done] == ["geology_build", "geology_code", "geology_build"]
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(folder), "--into", str(target)]) == 0, capsys.readouterr()
    capsys.readouterr()
    source, replayed = Project.open(folder), Project.open(target)
    obj = replayed.get(gid)
    assert (obj.kind, obj.path) == ("geology", source.get(gid).path)
    a = GeologyModel.load(source.path_of(gid))
    b = GeologyModel.load(replayed.path_of(gid))
    lo, hi = env["box"]
    pts = np.random.default_rng(9).uniform(lo, hi, size=(30000, 3))
    for layer in LAYERS:
        np.testing.assert_array_equal(a.labels(pts, layer), b.labels(pts, layer))
    np.testing.assert_array_equal(
        a.labels(pts[:3000], exact=True), b.labels(pts[:3000], exact=True)
    )
    for name in ("spec.json", "provenance.json", "diagnostics.json"):
        first = json.loads((source.path_of(gid) / name).read_text())
        second = json.loads((replayed.path_of(gid) / name).read_text())
        if name == "provenance.json":
            for x, y in zip(first["entries"], second["entries"], strict=True):
                assert x == y, (x, y)
        assert without_time(first) == without_time(second), name
    coded, again = load_model(source.path_of("model")), load_model(replayed.path_of("model"))
    np.testing.assert_array_equal(coded.block_ids(), again.block_ids())
    for name in ("fault_block", "lith", "weathering", "geo", "density"):
        np.testing.assert_array_equal(coded[name], again[name])
    assert (replayed.path_of(gid) / "coding" / "model.provenance.json").is_file()
    second = env["second"]
    assert replayed.get(second).path == source.get(second).path
    c = GeologyModel.load(source.path_of(second))
    d = GeologyModel.load(replayed.path_of(second))
    for layer in ("lith", "fault_block"):
        np.testing.assert_array_equal(c.labels(pts, layer), d.labels(pts, layer))
