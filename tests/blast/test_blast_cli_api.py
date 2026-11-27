"""``bakumine blast`` and ``/api/v1/blast``: both run ``bakumine.blast.simulate.simulate``,
so the same request gives the same bundle; with a project the blast sits on its topography
and the API keeps the latest result in ``<project>/blast``."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import EmptySceneSource, set_scene_source
from bakumine.blast.simulate import defaults, validate_bundle
from bakumine.cli import main
from bakumine.core.geometry.dtm import dtm_from_grid
from bakumine.core.io.obj import write_obj
from bakumine.core.project import Project

API = "/api/v1/blast"


def run(*argv) -> int:
    return main([str(a) for a in argv])


@pytest.fixture
def no_project():
    set_scene_source(EmptySceneSource())
    yield TestClient(app)
    set_scene_source(None)


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    """A project whose topography slopes up to the north: z = 300 + 0.04 (y - 5000),
    x 1000..1400, y 5000..5400 on a 10 m grid."""
    base = tmp_path_factory.mktemp("blast")
    i = np.arange(41)
    z = 300.0 + 0.04 * (10.0 * i)[None, :] + np.zeros((41, 1))
    write_obj(
        base / "topo.obj", dtm_from_grid(z, (1000.0, 5000.0), (10.0, 10.0)).mesh, decimals=None
    )
    folder = base / "project"
    Project.create(folder, "blast").add("dtm", base / "topo.obj", "Topography", "topo")
    return folder


@pytest.fixture
def served(project):
    set_scene_source(ProjectSceneSource(project))
    yield TestClient(app)
    set_scene_source(None)


def quick(request: dict) -> dict:
    """The request on a smaller block (30 x 14 m around the same centre), to run fast."""
    from bakumine.blast.bench import demo_site, rectangle

    polygon = request["polygon"] or demo_site().polygon
    block = rectangle(np.mean(polygon, axis=0), 30.0, 14.0, request["face_azimuth"])
    return {**request, "polygon": block, "diameter": 0.2, "particles": 300}


# ------------------------------------------------------------------------------ no project


def test_cli_demo_equals_api_simulate(tmp_path, no_project, capsys):
    out, design = tmp_path / "out" / "result.json", tmp_path / "design.json"
    assert run("blast", "demo", "--out", out, "--design", design) == 0
    assert "100 holes" in capsys.readouterr().out
    cli = json.loads(out.read_text())
    assert validate_bundle(cli) == []

    r = no_project.get(f"{API}/defaults")
    assert r.status_code == 200 and r.json() == defaults()
    r = no_project.post(f"{API}/simulate", json=r.json())
    assert r.status_code == 200, r.text
    assert r.json() == cli
    assert json.loads(design.read_text()) == cli["design"]
    assert cli["design"]["format"] == "bakumine-blast-design"
    # nothing is kept without a project
    assert no_project.get(f"{API}/result").status_code == 404


def test_cli_defaults_and_simulate_without_project(tmp_path, capsys):
    assert run("blast", "defaults") == 0
    request = json.loads(capsys.readouterr().out)
    assert request == defaults()
    params = tmp_path / "request.json"
    params.write_text(json.dumps(quick(request) | {"name": "Bench 990"}))
    assert run("blast", "simulate", "--params", params, "--out", tmp_path / "r.json") == 0
    bundle = json.loads((tmp_path / "r.json").read_text())
    assert bundle["design"]["name"] == "Bench 990" and bundle["design"]["floor_z"] == 990.0
    assert len(bundle["throw"]["particles"]["kind"]) == 300


def test_api_errors_are_400_with_the_message(no_project):
    r = no_project.post(f"{API}/simulate", json=defaults() | {"diameter": 165})
    assert r.status_code == 400 and "diameter is in metres" in r.json()["detail"]
    bad = defaults() | {"timing": {"type": "hole_by_hole", "inter_hole_ms": 20, "inter_row_ms": 40}}
    r = no_project.post(f"{API}/simulate", json=bad)
    assert r.status_code == 400 and "same time" in r.json()["detail"]
    r = no_project.post(f"{API}/simulate", json=defaults() | {"model": 5})
    assert r.status_code == 400 and "model must be" in r.json()["detail"]


def test_cli_errors(tmp_path, capsys):
    assert (
        run("blast", "simulate", "--params", tmp_path / "none.json", "--out", tmp_path / "r") == 2
    )
    assert "does not exist" in capsys.readouterr().err
    params = tmp_path / "request.json"
    params.write_text(json.dumps(defaults() | {"pattern": "hex"}))
    assert run("blast", "simulate", "--params", params, "--out", tmp_path / "r.json") == 2
    assert "pattern must be" in capsys.readouterr().err
    assert not (tmp_path / "r.json").exists()
    assert run("blast", "simulate", "--params", params, "--topo", "x", "--out", tmp_path / "r") == 2
    assert "need --project" in capsys.readouterr().err
    (tmp_path / "bad.json").write_text('{"format": "bakumine-blast-result"}')
    assert run("blast", "report", tmp_path / "bad.json") == 2
    assert "not a valid blast result" in capsys.readouterr().err


# ------------------------------------------------------------------------------ project


def test_api_on_a_project_saves_the_latest_result(served, project, tmp_path, capsys):
    before = (project / "project.json").read_text()
    assert served.get(f"{API}/result").status_code == 404

    r = served.get(f"{API}/defaults")
    assert r.status_code == 200
    request = r.json()
    polygon = np.array(request["polygon"])  # 60 x 35 m at the centre of the topography
    np.testing.assert_allclose(polygon.mean(axis=0), [1200.0, 5200.0])
    (monitor,) = request["monitors"]
    # 300 m behind the block (north): off the topography, so at the block centre's elevation
    assert monitor["xyz"] == pytest.approx([1200.0, 5500.0, 308.0])

    r = served.post(f"{API}/simulate", json=quick(request))
    assert r.status_code == 200, r.text
    bundle = r.json()
    assert validate_bundle(bundle) == []
    for h in bundle["timeline"]["holes"]:  # collars on the sloping ground
        assert h["collar"][2] == pytest.approx(300.0 + 0.04 * (h["collar"][1] - 5000.0), abs=1e-6)
    assert bundle["design"]["floor_z"] == pytest.approx(308.0 - 10.0, abs=0.05)
    assert any("Free face excavated" in note for note in bundle["summary"]["notes"])

    saved = project / "blast" / "latest.json"
    assert json.loads(saved.read_text()) == bundle
    assert json.loads((project / "blast" / "latest_design.json").read_text()) == bundle["design"]
    r = served.get(f"{API}/result")
    assert r.status_code == 200 and r.json() == bundle
    assert (project / "project.json").read_text() == before  # no new project objects
    assert [o.kind for o in Project.open(project).list()] == ["dtm"]

    # the CLI on the same project and request gives the same bundle
    params = tmp_path / "request.json"
    params.write_text(json.dumps(quick(request)))
    assert run("blast", "simulate", "--params", params, "--project", project,
               "--out", tmp_path / "cli.json") == 0  # fmt: skip
    assert "topography 'topo'" in capsys.readouterr().out
    assert json.loads((tmp_path / "cli.json").read_text()) == bundle
    assert run("blast", "defaults", "--project", project) == 0
    assert json.loads(capsys.readouterr().out) == request

    # a failed run keeps the saved result; an unknown topography is a 400
    r = served.post(f"{API}/simulate", json=quick(request) | {"topo": "nope"})
    assert r.status_code == 400 and "no dtm object 'nope'" in r.json()["detail"]
    assert served.get(f"{API}/result").json() == bundle


def test_report_prints_the_key_numbers(served, tmp_path, capsys):
    bundle = served.get(f"{API}/result").json()
    result = tmp_path / "result.json"
    result.write_text(json.dumps(bundle))
    assert run("blast", "report", result) == 0
    text = capsys.readouterr().out
    s, (monitor,) = bundle["summary"], bundle["monitors"]
    for want in (
        f"BLAST REPORT: {bundle['design']['name']}",
        f"{s['holes']} / {s['rows']}",
        f"{s['explosive_kg']:,.0f} kg",
        f"{s['powder_factor_kg_m3']:.3f} kg/m3",
        f"{s['volume_m3']:,.0f} m3",
        f"{s['mic_kg']:,.1f} kg",
        f"{s['x50_m']:.3f} m",
        f"{s['p80_m']:.3f} m",
        f"{monitor['ppv_mm_s']:,.2f} mm/s",
        f"(limit {monitor['limit_mm_s']:.1f} mm/s)",
        "ABOVE LIMIT" if not monitor["ok"] else "within limit",
        f"{monitor['airblast_dbl']:.1f} dBL",
        f"{bundle['flyrock']['clearance_m']:,.0f} m",
        f"{s['backbreak_m']:.1f} m",
        "200 mm",
        "uncalibrated",
        "Free face excavated in front of the block",
        "launch velocity",
    ):
        assert want in text, want
    assert run("blast", "report", result, "--out", tmp_path / "report.txt") == 0
    assert (tmp_path / "report.txt").read_text() == text


def test_project_site_prefers_the_surface_made_for_the_blast(tmp_path):
    """Without an id, the dtm with the id ``blast-terrain`` is the ground, wherever it is listed."""
    from bakumine.blast import bench

    def surface(name: str, z: float, side: int) -> str:
        path = tmp_path / f"{name}.obj"
        write_obj(
            path,
            dtm_from_grid(np.full((side, side), z), (0.0, 0.0), (10.0, 10.0)).mesh,
            decimals=None,
        )
        return str(path)

    folder = tmp_path / "project"
    project = Project.create(folder, "site")
    project.add("dtm", surface("small", 100.0, 5), "Topography", "topo")
    dtm, _, ids = bench.project_site(folder)
    assert ids["topo"] == "topo" and float(dtm.z_at(20.0, 20.0)) == pytest.approx(100.0)

    project.add("dtm", surface("wide", 130.0, 21), "Blast terrain", bench.BLAST_TERRAIN_ID)
    dtm, _, ids = bench.project_site(folder)
    assert ids["topo"] == bench.BLAST_TERRAIN_ID and float(dtm.z_at(20.0, 20.0)) == pytest.approx(
        130.0
    )
    # An id that is asked for still wins.
    dtm, _, ids = bench.project_site(folder, "topo")
    assert ids["topo"] == "topo" and float(dtm.z_at(20.0, 20.0)) == pytest.approx(100.0)
    with pytest.raises(ValueError, match="nope"):
        bench.project_site(folder, "nope")
