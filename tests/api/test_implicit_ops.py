"""Implicit solid from drillhole codes: endpoint == CLI == Python API on the synthetic deposit.

The endpoint's solid is read back from the project folder, the CLI's from ``--out`` and
``--geometry-dir``, and both are compared with ``implicit_solid`` called directly. The job
log is then replayed with ``bakumine project replay`` and must give the same solids again.
"""

from __future__ import annotations

import json

import pytest
from geometry_fixtures import (
    HAVE_SYNTH,
    OPS,
    SYNTH,
    geometry_env,
    npy_mesh,
    project_mesh,
    same_mesh,
)

from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.estimation.orientation import ellipsoid_from_plane
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.implicit import implicit_solid
from bakumine.core.io.obj import read_obj
from bakumine.core.project import Project

pytestmark = pytest.mark.skipif(
    not HAVE_SYNTH, reason="data/synthetic/default has not been generated"
)

#: the plane of the synthetic lens (what the automatic estimate finds, rounded)
PLANE = {"mode": "plane", "dip": 56.0, "dip_azimuth": 310.0, "pitch": 20.0, "ratios": "16:9:1"}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    yield from geometry_env(tmp_path_factory)


def test_implicit_solid_endpoint_cli_python(env, tmp_path, capsys):
    client, folder = env["client"], env["folder"]
    truth = json.loads((SYNTH / "truth.json").read_text())
    request = {"database": "drillholes", "table": "geology", "column": "rock", "codes": ["ORE"],
               "anisotropy": PLANE, "max_extrapolation": 60, "topography": "topo",
               "min_component_volume": 1000, "name": "Ore implicit"}  # fmt: skip
    r = client.post(f"{OPS}/solids/implicit", json=request)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["kind"] == "solid" and result["valid"] and result["n_components"] == 1
    assert result["volume"] == pytest.approx(truth["solid"]["volume_m3"], rel=0.1)
    a = result["anisotropy"]
    assert a["source"] == "user" and a["plane"] == {"dip": 56.0, "dip_azimuth": 310.0}
    assert a["ratios"] == [1.0, 0.5625, 0.0625]
    assert result["max_extrapolation"] == 60 and result["n_holes"] == 80
    api_mesh = project_mesh(folder, result["id"])
    assert api_mesh.volume == pytest.approx(result["volume"])

    project = Project.open(folder)
    assert main(["solid", "implicit", str(project.path_of("drillholes")), "--column", "rock",
                 "--code", "ORE", "--dip", "56", "--dip-azimuth", "310", "--pitch", "20",
                 "--ratios", "16:9:1", "--max-extrapolation", "60",
                 "--topography", str(project.path_of("topo")), "--min-component-volume", "1000",
                 "--out", str(tmp_path / "ore.obj"), "--geometry-dir", str(tmp_path),
                 "--name", "ore"]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "plane dips 56.0 towards 310.0" in out and "valid" in out
    same_mesh(read_obj(tmp_path / "ore.obj"), api_mesh)
    same_mesh(npy_mesh(tmp_path, "ore"), api_mesh)

    ellipsoid = ellipsoid_from_plane((16.0, 9.0, 1.0), 56.0, 310.0, 20.0)
    python = implicit_solid(
        env["database"], "geology", "rock", "ORE", anisotropy=ellipsoid, max_extrapolation=60.0,
        topography=DTM(read_obj(SYNTH / "topo.obj")), min_component_volume=1000.0,
    )  # fmt: skip
    same_mesh(python.mesh, api_mesh)
    entry = joblog.jobs(folder)[-1]
    assert entry["operation"] == "implicit_solid"
    assert [o["id"] for o in entry["inputs"]] == ["drillholes", "topo"]
    assert entry["summary"]["volume"] == result["volume"]
    assert result["id"] in {o["id"] for o in client.get("/api/v1/scene").json()["objects"]}


def test_implicit_solid_auto_reports_the_anisotropy_it_used(env, tmp_path, capsys):
    client, folder = env["client"], env["folder"]
    r = client.post(
        f"{OPS}/solids/implicit",
        json={"database": "drillholes", "column": "rock", "codes": "ORE", "anisotropy": "auto"},
    )
    assert r.status_code == 200, r.text
    result = r.json()
    a = result["anisotropy"]
    assert a["source"] == "auto" and a["method"] == "intercept midpoints"
    assert a["plane"]["dip"] == pytest.approx(PLANE["dip"], abs=2.0)
    assert a["plane"]["dip_azimuth"] == pytest.approx(PLANE["dip_azimuth"], abs=2.0)
    assert a["hole_angle_deg"] > 20.0 and result["warnings"] == []
    assert result["name"] == "ORE solid"
    db_path = Project.open(folder).path_of("drillholes")
    assert main(["solid", "implicit", str(db_path), "--column", "rock", "--code", "ORE",
                 "--anisotropy", "auto", "--out", str(tmp_path / "auto.obj")]) == 0  # fmt: skip
    assert "anisotropy (auto)" in capsys.readouterr().out
    same_mesh(read_obj(tmp_path / "auto.obj"), project_mesh(folder, result["id"]))


def test_bad_requests_are_400_and_logged(env, capsys):
    client, folder = env["client"], env["folder"]
    before = len(joblog.jobs(folder))
    base = {"database": "drillholes", "column": "rock", "codes": ["ORE"], "anisotropy": "auto"}
    cases = [
        {**base, "anisotropy": None},  # the orientation must be stated
        {**base, "codes": ["GOLD"]},
        {**base, "anisotropy": {"mode": "plane", "dip": 50, "dip_azimuth": 10}},  # no ratios
        {**base, "anisotropy": {"mode": "angles", "angles": [1, 2], "ratios": "4:2:1"}},
        {**base, "topography": "drillholes"},
        {**base, "database": "model"},
        {**base, "column": "lith"},
        {**base, "resolution": "fine"},
    ]
    for body in cases:
        r = client.post(f"{OPS}/solids/implicit", json=body)
        assert r.status_code == 400, (body, r.text)
        assert isinstance(r.json()["detail"], str)
    entries = joblog.jobs(folder)[before:]
    assert len(entries) == len(cases) and all(e["status"] == "error" for e in entries)
    # the CLI refuses a solid without a stated anisotropy
    db_path = str(Project.open(folder).path_of("drillholes"))
    command = ["solid", "implicit", db_path, "--column", "rock", "--code", "ORE", "--out", "x.obj"]
    assert main(command) == 2
    assert "orientation" in capsys.readouterr().err
    assert main([*command, "--dip", "50"]) == 2
    assert main([*command, "--dip", "50", "--dip-azimuth", "10"]) == 2  # no ratios
    assert main([*command, "--anisotropy", "auto", "--angles", "1,2,3"]) == 2
    capsys.readouterr()
    plane = client.post(f"{OPS}/orientation", json={"plane": {"dip": 69, "dip_azimuth": 216}})
    assert plane.json()["angles"] == [126.0, 0.0, -69.0]
    assert plane.json()["major"] == {"azimuth": 126.0, "plunge": 0.0}
    assert client.post(f"{OPS}/orientation", json={"plane": {"dip": 120}}).status_code == 400


def test_replay_rebuilds_the_same_solids(env, tmp_path, capsys):
    """Runs last: the implicit-solid jobs above are redone by the CLI from the job log."""
    folder = env["folder"]
    done = [e for e in joblog.jobs(folder) if e["status"] == "done"]
    assert [e["operation"] for e in done] == ["implicit_solid", "implicit_solid"]
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(folder), "--into", str(target)]) == 0, capsys.readouterr()
    capsys.readouterr()
    for entry in done:
        out = entry["outputs"][0]
        replayed = Project.open(target).get(out["id"])
        assert (replayed.kind, replayed.path) == (out["kind"], out["path"])
        same_mesh(project_mesh(target, out["id"]), project_mesh(folder, out["id"]))


@pytest.mark.parametrize(
    ("extra", "text"),
    [
        ({"resolution": 1e-6}, "coarsen the resolution"),
        ({"resolution": 1e-6, "max_extrapolation": 60}, "coarsen the resolution"),
        ({"max_extrapolation": 1e9}, "max_extrapolation"),
    ],
)
def test_implicit_solid_refuses_a_grid_that_cannot_be_meant(env, extra, text):
    """Final review: these answered with numpy's "array is too big; arr.size * ..."."""
    client, folder = env["client"], env["folder"]
    before = len(Project.open(folder).list("solid"))
    request = {"database": "drillholes", "table": "geology", "column": "rock", "codes": ["ORE"],
               "anisotropy": "none", "name": "too fine"}  # fmt: skip
    r = client.post(f"{OPS}/solids/implicit", json=request | extra)
    assert r.status_code == 400, r.text
    detail = r.json()["detail"]
    assert text in detail and "arr.size" not in detail and "array is too big" not in detail
    assert len(Project.open(folder).list("solid")) == before
