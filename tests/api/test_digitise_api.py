"""Section digitising endpoints (P5.T7).

Acceptance: five sections of the synthetic deposit "digitised by script" (the ore outlines
cut from the reference solid by ``section``), posted as strings, built into a solid. The
solid must be valid and watertight, and its volume must match the reference over the same
span. The reference volume is computed independently of the solid builder: the area of the
reference solid's cross-section integrated along the normal on 401 planes (trapezoid rule),
cross-checked against trimesh's volume of the built mesh.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import trimesh
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import DemoSceneSource, get_scene_source
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.section import section
from bakumine.core.geometry.sections_to_solid import sections_from_mesh
from bakumine.core.io.obj import read_obj
from bakumine.core.io.surpac_str import parse_surpac_str
from bakumine.core.project import Project

DEMO = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"

pytestmark = pytest.mark.skipif(
    not (DEMO / "ore_solid_vertices.npy").is_file(),
    reason="data/synthetic/default has not been generated",
)

NORMAL = np.array([0.0, 1.0, 0.0])  # W-E sections, stepping north


@pytest.fixture(scope="module")
def ore() -> Mesh:
    return Mesh(np.load(DEMO / "ore_solid_vertices.npy"), np.load(DEMO / "ore_solid_faces.npy"))


@pytest.fixture(scope="module")
def northings(ore) -> np.ndarray:
    lo, hi = ore.vertices[:, 1].min(), ore.vertices[:, 1].max()
    return np.round(np.linspace(lo + 0.15 * (hi - lo), hi - 0.15 * (hi - lo), 5))


def digitise(ore: Mesh, northings) -> list[dict]:
    """One closed string per outline per section, as the viewer would post them."""
    strings = []
    for k, y in enumerate(northings):
        cut = sections_from_mesh(ore, [0.0, y, 0.0], NORMAL, 1e6)
        assert len(cut) == 1
        for j, (line, closed) in enumerate(zip(cut[0].polylines, cut[0].closed, strict=True)):
            assert closed
            strings.append(
                {
                    "id": f"s{k}-{j}",
                    "section": f"az090-dip90-N{y:.1f}",
                    "closed": True,
                    "points": line.tolist(),
                }
            )
    return strings


def reference_volume(ore: Mesh, y0: float, y1: float, n: int = 401) -> float:
    """Integral of the reference cross-section area from y0 to y1 (trapezoid rule)."""
    ys = np.linspace(y0, y1, n)
    areas = []
    for y in ys:
        total = 0.0
        for p in section(ore, [0.0, y, 0.0], NORMAL).polylines:
            x, z = p[:, 0], p[:, 2]
            total += abs(0.5 * np.sum(x * np.roll(z, -1) - np.roll(x, -1) * z))
        areas.append(total)
    return float(np.trapezoid(areas, ys))


@pytest.fixture
def demo_client():
    source = DemoSceneSource(DEMO)
    app.dependency_overrides[get_scene_source] = lambda: source
    yield TestClient(app), source
    app.dependency_overrides.clear()


@pytest.fixture
def project_client(tmp_path):
    project = Project.create(tmp_path / "proj", "Digitise project")
    source = ProjectSceneSource(project.folder)
    app.dependency_overrides[get_scene_source] = lambda: source
    yield TestClient(app), project.folder
    app.dependency_overrides.clear()


def test_five_digitised_sections_make_a_valid_solid(demo_client, ore, northings):
    client, source = demo_client
    strings = digitise(ore, northings)
    saved = client.put("/api/v1/digitise/strings", json={"strings": strings})
    assert saved.status_code == 200, saved.text
    assert saved.json()["n_strings"] == len(strings) == 5

    built = client.post("/api/v1/digitise/solid", json={"name": "Ore digitised"})
    assert built.status_code == 200, built.text
    result = built.json()
    assert result["valid"] and result["watertight"] and result["outward"]
    assert result["issues"] == []
    assert result["n_sections"] == 5 and result["skipped_open"] == 0

    reference = reference_volume(ore, northings[0], northings[-1])
    assert result["volume"] == pytest.approx(reference, rel=0.03)

    # The solid is in the scene, and trimesh agrees on its volume and closure.
    manifest = client.get("/api/v1/scene").json()
    entry = next(o for o in manifest["objects"] if o["id"] == result["id"])
    assert entry["category"] == "solids" and entry["kind"] == "mesh"
    assert client.get(entry["href"]).status_code == 200
    mesh = source.mesh(result["id"])
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    assert tm.is_watertight and tm.is_winding_consistent
    assert tm.volume == pytest.approx(result["volume"], rel=1e-9)
    # Spans exactly the digitised sections (flat ends).
    assert mesh.vertices[:, 1].min() == pytest.approx(northings[0], abs=1e-3)
    assert mesh.vertices[:, 1].max() == pytest.approx(northings[-1], abs=1e-3)

    # Removing it takes it out of the scene; demo data cannot be removed.
    assert client.delete(f"/api/v1/digitise/solid/{result['id']}").status_code == 200
    assert result["id"] not in {o["id"] for o in client.get("/api/v1/scene").json()["objects"]}
    assert client.delete(f"/api/v1/digitise/solid/{result['id']}").status_code == 404
    assert client.delete("/api/v1/digitise/solid/ore-solid").status_code == 404


def test_strings_round_trip_and_str_export(demo_client):
    client, _ = demo_client
    square = [
        [500500.0, 5300500.0, 1000.0],
        [500510.0, 5300500.0, 1000.0],
        [500510.0, 5300500.0, 1010.0],
    ]
    body = {
        "strings": [
            {"id": "a", "section": "az090-dip90-N5300500.0", "closed": True, "points": square},
            {"id": "b", "section": "az090-dip90-N5300500.0", "closed": False, "points": square[:2]},
        ]
    }
    assert client.put("/api/v1/digitise/strings", json=body).status_code == 200
    back = client.get("/api/v1/digitise/strings").json()["strings"]
    assert [(s["id"], s["closed"], len(s["points"])) for s in back] == [
        ("a", True, 3),
        ("b", False, 2),
    ]
    np.testing.assert_allclose(back[0]["points"], square, atol=5e-4)
    assert back[0]["section"] == "az090-dip90-N5300500.0"

    text = client.get("/api/v1/digitise/strings.str").text
    parsed = parse_surpac_str(text)
    assert parsed.n_segments == 2 and parsed.segments[0].is_closed
    # Surpac order Y, X, Z in the file: the first coordinate column is the northing.
    first = text.splitlines()[2].split(",")
    assert float(first[1]) == pytest.approx(5300500.0) and float(first[2]) == pytest.approx(
        500500.0
    )

    # One open and one closed string: not enough for a solid, explained.
    bad = client.post("/api/v1/digitise/solid", json={})
    assert bad.status_code == 422
    assert "two closed" in bad.json()["detail"]["message"]


def test_bad_requests(demo_client):
    client, _ = demo_client
    two = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    r = client.put(
        "/api/v1/digitise/strings", json={"strings": [{"id": "x", "closed": True, "points": two}]}
    )
    assert r.status_code == 422
    dup = {"id": "x", "points": two}
    assert client.put("/api/v1/digitise/strings", json={"strings": [dup, dup]}).status_code == 422
    comma = {"id": "x", "section": "a,b", "points": two}
    assert client.put("/api/v1/digitise/strings", json={"strings": [comma]}).status_code == 422
    assert client.get("/api/v1/digitise/strings?set=../x").status_code == 400
    assert client.get("/api/v1/digitise/strings.str?set=nothing").status_code == 404


def test_project_stores_strings_and_solid_in_the_folder(project_client, ore, northings):
    client, folder = project_client
    strings = digitise(ore, northings[:3])
    assert (
        client.put("/api/v1/digitise/strings?set=ore", json={"strings": strings}).json()[
            "object_id"
        ]
        == "strings-ore"
    )
    # Saving again overwrites the same object.
    assert (
        client.put("/api/v1/digitise/strings?set=ore", json={"strings": strings}).status_code == 200
    )
    project = Project.open(folder)
    assert [(o.id, o.kind, o.path) for o in project.list()] == [
        ("strings-ore", "string", "strings/ore.str")
    ]
    assert parse_surpac_str((folder / "strings" / "ore.str").read_text()).n_segments == 3

    built = client.post("/api/v1/digitise/solid", json={"set": "ore", "name": "Ore A"}).json()
    assert built["valid"]
    project = Project.open(folder)
    solid = project.get("ore-a")
    assert solid.kind == "solid"
    mesh = read_obj(project.path_of(solid))
    assert mesh.volume == pytest.approx(built["volume"], rel=1e-9)
    assert any(o["id"] == "ore-a" for o in client.get("/api/v1/scene").json()["objects"])
    # Building again replaces the solid rather than adding a second one.
    client.post("/api/v1/digitise/solid", json={"set": "ore", "name": "Ore A"})
    assert [o.id for o in Project.open(folder).list("solid")] == ["ore-a"]
    assert client.delete("/api/v1/digitise/solid/strings-ore").status_code == 404
    assert client.delete("/api/v1/digitise/solid/ore-a").status_code == 200
    assert Project.open(folder).list("solid") == []
