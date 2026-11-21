"""Surface and mesh-import operations: endpoint == CLI == Python API.

Both run on ``data/synthetic/default``. The endpoint's result is read back from the
project folder, the CLI's from ``--out`` / ``--geometry-dir``, and both are compared with
the core functions called directly. The job log of the endpoint calls is then replayed
with ``bakumine project replay`` and must give the same meshes again.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from geometry_fixtures import (
    HAVE_SYNTH,
    OPS,
    SYNTH,
    geometry_env,
    npy_mesh,
    project_mesh,
    same_mesh,
    upload,
)

from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.surface_points import (
    extension_points,
    model_rectangle,
    string_lines,
    surface_from_points,
)
from bakumine.core.io.files import write_mesh, write_strings
from bakumine.core.io.obj import read_obj
from bakumine.core.io.strings import StringFile, StringSegment
from bakumine.core.project import Project

pytestmark = pytest.mark.skipif(
    not HAVE_SYNTH, reason="data/synthetic/default has not been generated"
)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    yield from geometry_env(tmp_path_factory)


# ------------------------------------------------------------------ surface from points


def test_topography_from_collars_endpoint_cli_python(env, tmp_path, capsys):
    client, folder = env["client"], env["folder"]
    r = client.post(
        f"{OPS}/surfaces/from-points",
        json={"name": "Topo from collars", "databases": ["drillholes"],
              "extend": {"mode": "padding", "padding": 100}},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["kind"] == "dtm" and result["sources"] == {"collars": 80}
    assert result["n_points"] == 80 and result["n_extension_points"] > 0
    assert result["extend_z"] == "nearest"
    api_mesh = project_mesh(folder, result["id"])

    db_path = Project.open(folder).path_of("drillholes")
    assert main(["surface", "from-points", "--database", str(db_path), "--extend-padding", "100",
                 "--out", str(tmp_path / "topo.obj"), "--geometry-dir", str(tmp_path / "geo"),
                 "--name", "topo_cli", "--summary", str(tmp_path / "s.json")]) == 0  # fmt: skip
    out = capsys.readouterr().out
    assert "80 collars" in out and "elevation of the nearest data point" in out
    same_mesh(read_obj(tmp_path / "topo.obj"), api_mesh)
    same_mesh(npy_mesh(tmp_path / "geo", "topo_cli"), api_mesh)
    assert json.loads((tmp_path / "s.json").read_text())["n_faces"] == result["n_faces"]

    python = surface_from_points(env["collars"], padding=100.0)
    same_mesh(python.dtm.mesh, api_mesh)
    # it is a usable DTM through every collar, and the scene lists it
    dtm = DTM(api_mesh)
    c = env["collars"]
    np.testing.assert_allclose(dtm.z_at(c[:, 0], c[:, 1]), c[:, 2], atol=1e-9)
    scene = {o["id"]: o for o in client.get("/api/v1/scene").json()["objects"]}
    assert result["id"] in scene
    assert client.get(f"/api/v1/scene/objects/{result['id']}/mesh").status_code == 200
    kinds = {o["id"]: o["kind"] for o in client.get(f"{OPS}/project").json()["objects"]}
    assert kinds[result["id"]] == "dtm"
    entry = joblog.jobs(folder)[-1]
    assert entry["operation"] == "surface_from_points" and entry["status"] == "done"
    assert [o["id"] for o in entry["inputs"]] == ["drillholes"]
    assert entry["outputs"][0]["id"] == result["id"]


def test_combined_sources_breaklines_and_model_extent(env, tmp_path, capsys):
    client, folder, base = env["client"], env["folder"], env["base"]
    # a points table (Surpac column order: northing first) with a blank and a duplicate row
    rng = np.random.default_rng(2)
    topo = np.load(SYNTH / "topo_vertices.npy")
    pick = topo[rng.choice(len(topo), 120, replace=False)]
    table = pd.DataFrame({"Northing": pick[:, 1], "Easting": pick[:, 0], "RL": pick[:, 2]})
    table = pd.concat([table, table.iloc[:4].assign(RL=table["RL"].iloc[:4] + 1.0)])
    table.loc[len(table)] = [np.nan, 1.0, 2.0]
    table.to_csv(base / "survey points.csv", index=False)
    # an uploaded breakline file
    c = env["collars"]
    order = np.argsort(c[:, 1])
    crest = c[order[[2, len(order) // 3, -3]]] + [-5.0, 2.0, 1.5]
    write_strings(base / "crest.str", StringFile([StringSegment(7, crest)]))

    points = upload(client, base / "survey points.csv")
    described = points["files"][0]
    assert described["roles"] == ["points"] and described["rows"] == 125
    assert described["xyz"] == ["Easting", "Northing", "RL"] and described["skipped"] == 1
    assert points["file"] == "survey_points.csv"
    lines = upload(client, base / "crest.str")
    assert lines["files"][0]["roles"] == ["strings"] and lines["files"][0]["n_strings"] == 1

    request = {
        "name": "Topo combined",
        "databases": ["drillholes"],
        "points": [{"upload": points["token"]}],
        "strings": ["ridge"],
        "breaklines": [{"upload": lines["token"]}],
        "extend": {"mode": "model", "model": "model", "padding": 25, "z": 1200},
        "duplicates": "max",
    }
    r = client.post(f"{OPS}/surfaces/from-points", json=request)
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["sources"] == {"collars": 80, "table points": 124, "string points": 3}
    assert result["n_breaklines"] == 1 and result["n_conflicts"] == 4
    # the synthetic topography passes through the collars: some picked points are collars
    collar_xy = {tuple(row) for row in env["collars"][:, :2]}
    on_collars = sum(tuple(row) in collar_xy for row in pick[:, :2])
    assert result["n_duplicates"] == 4 + on_collars and result["duplicates"] == "max"
    box = model_rectangle(env["definition"])
    assert result["rectangle"] == pytest.approx(
        [box[0] - 25, box[1] - 25, box[2] + 25, box[3] + 25]
    )
    api_mesh = project_mesh(folder, result["id"])

    project = Project.open(folder)
    assert main(["surface", "from-points",
                 "--database", str(project.path_of("drillholes")),
                 "--points", str(base / "survey points.csv"),
                 "--strings", str(project.path_of("ridge")),
                 "--breaklines", str(base / "crest.str"),
                 "--extend-model", str(project.path_of("model")), "--extend-padding", "25",
                 "--extend-z", "1200", "--duplicates", "max",
                 "--out", str(tmp_path / "combined.obj")]) == 0  # fmt: skip
    assert "4 locations with different Z resolved by 'max'" in capsys.readouterr().out
    same_mesh(read_obj(tmp_path / "combined.obj"), api_mesh)

    table_points = table[["Easting", "Northing", "RL"]].to_numpy()[:-1]
    python = surface_from_points(
        [c, table_points, env["line"]],
        string_lines(StringFile([StringSegment(7, crest)])),
        rectangle=box, padding=25.0, extend_z=1200.0, duplicates="max",
    )  # fmt: skip
    np.testing.assert_allclose(python.dtm.mesh.vertices, api_mesh.vertices, atol=1e-3)
    np.testing.assert_array_equal(python.dtm.mesh.faces, api_mesh.faces)
    dtm = DTM(api_mesh)
    # the boundary points (those outside the hull of the data) are at the given elevation
    grown = [box[0] - 25, box[1] - 25, box[2] + 25, box[3] + 25]
    everything = np.concatenate([c, table_points, env["line"], crest])
    extra = extension_points(everything, grown, 1200.0)
    assert len(extra) == result["n_extension_points"] > 0
    np.testing.assert_allclose(dtm.z_at(extra[:, 0], extra[:, 1]), 1200.0, atol=1e-9)
    np.testing.assert_allclose(
        dtm.z_at(table_points[:4, 0], table_points[:4, 1]), table_points[:4, 2] + 1.0, atol=1e-6
    )
    entry = joblog.jobs(folder)[-1]
    assert {o["id"] for o in entry["inputs"]} == {"drillholes", "ridge", "model"}
    for item in entry["replay"]["points"] + entry["replay"]["breaklines"]:
        assert (folder / item["file"]).is_file()  # archived for replay


# ------------------------------------------------------------------ mesh import


def test_mesh_import_endpoint_cli_python(env, tmp_path, capsys):
    client, folder, base = env["client"], env["folder"], env["base"]
    truth = json.loads((SYNTH / "truth.json").read_text())

    solid = upload(client, SYNTH / "ore_solid.obj")
    assert solid["files"][0]["roles"] == ["mesh"] and solid["files"][0]["n_faces"] == 20480
    r = client.post(
        f"{OPS}/meshes/import", json={"upload": solid["token"], "name": "Ore wireframe"}
    )
    assert r.status_code == 200, r.text
    result = r.json()
    assert result["kind"] == "solid" and result["valid"] and result["closed"]
    assert result["volume"] == pytest.approx(truth["solid"]["volume_m3"], rel=2e-3)
    assert main(["surface", "import", str(SYNTH / "ore_solid.obj"),
                 "--geometry-dir", str(tmp_path), "--name", "ore"]) == 0  # fmt: skip
    assert "solid: 10242 vertices, 20480 triangles, valid" in capsys.readouterr().out
    same_mesh(npy_mesh(tmp_path, "ore"), project_mesh(folder, result["id"]))
    same_mesh(read_obj(SYNTH / "ore_solid.obj"), project_mesh(folder, result["id"]))

    # a Surpac .dtm needs its .str: two files under one token
    write_mesh(base / "pit.dtm", read_obj(SYNTH / "topo.obj"))
    first = upload(client, base / "pit.dtm")
    assert first["files"][0]["roles"] == [] and first["files"][0]["needs"] == "pit.str"
    alone = client.post(f"{OPS}/meshes/import", json={"upload": first["token"]})
    assert alone.status_code == 400
    both = upload(client, base / "pit.str", first["token"])
    by_name = {f["file"]: f for f in both["files"]}
    assert by_name["pit.dtm"]["roles"] == ["mesh"]
    r = client.post(f"{OPS}/meshes/import", json={"upload": first["token"], "kind": "dtm"})
    assert r.status_code == 200, r.text
    surface = r.json()
    assert surface["kind"] == "dtm" and surface["name"] == "pit" and surface["n_faces"] == 1772
    assert main(["surface", "import", str(base / "pit.dtm"), "--kind", "dtm",
                 "--out", str(tmp_path / "pit.obj")]) == 0  # fmt: skip
    same_mesh(read_obj(tmp_path / "pit.obj"), project_mesh(folder, surface["id"]))
    original = DTM(read_obj(SYNTH / "topo.obj"))
    v = project_mesh(folder, surface["id"]).vertices
    np.testing.assert_allclose(original.z_at(v[:, 0], v[:, 1]), v[:, 2], atol=2e-3)
    kinds = {o["id"]: o["kind"] for o in client.get(f"{OPS}/project").json()["objects"]}
    assert kinds[result["id"]] == "solid" and kinds[surface["id"]] == "dtm"
    assert joblog.jobs(folder)[-1]["operation"] == "import_mesh"

    # an STL in feet: a tetrahedron with 10 ft legs, scaled to metres (analytic volume)
    corners = {"o": "0 0 0", "e": "10 0 0", "n": "0 10 0", "u": "0 0 10"}
    facets = ["one", "oeu", "oun", "enu"]  # counter-clockwise seen from outside
    text = "solid t\n" + "".join(
        "facet normal 0 0 0\nouter loop\n"
        + "".join(f"vertex {corners[c]}\n" for c in f)
        + "endloop\nendfacet\n"
        for f in facets
    ) + "endsolid t\n"  # fmt: skip
    (base / "feet.stl").write_text(text)
    feet = upload(client, base / "feet.stl")
    assert feet["files"][0]["roles"] == ["mesh"] and feet["files"][0]["n_faces"] == 4
    r = client.post(f"{OPS}/meshes/import", json={"upload": feet["token"], "scale": 0.3048})
    assert r.status_code == 200, r.text
    scaled = r.json()
    assert scaled["kind"] == "solid" and scaled["valid"] and scaled["scale"] == 0.3048
    assert scaled["volume"] == pytest.approx(1000.0 / 6.0 * 0.3048**3, rel=1e-12)
    assert joblog.jobs(folder)[-1]["replay"]["scale"] == 0.3048
    assert main(["surface", "import", str(base / "feet.stl"), "--scale", "0.3048",
                 "--geometry-dir", str(tmp_path), "--name", "feet"]) == 0  # fmt: skip
    assert "(coordinates x 0.3048)" in capsys.readouterr().out
    same_mesh(npy_mesh(tmp_path, "feet"), project_mesh(folder, scaled["id"]))
    for wrong in (0, -1, "big", True):
        bad = client.post(f"{OPS}/meshes/import", json={"upload": feet["token"], "scale": wrong})
        assert bad.status_code == 400 and "scale" in bad.json()["detail"], wrong

    # a closed solid cannot be imported as a DTM; unknown files are refused at upload
    bad = client.post(f"{OPS}/meshes/import", json={"upload": solid["token"], "kind": "dtm"})
    assert bad.status_code == 400 and "not a surface" in bad.json()["detail"]
    refused = client.put(f"{OPS}/geometry/uploads", params={"filename": "x.exe"}, content=b"..")
    assert refused.status_code == 400
    broken = client.put(f"{OPS}/geometry/uploads", params={"filename": "x.obj"}, content=b"f 1 2 3")
    assert broken.status_code == 400
    assert client.delete(f"{OPS}/geometry/uploads/{solid['token']}").status_code == 200
    assert client.post(f"{OPS}/meshes/import", json={"upload": solid["token"]}).status_code == 400
    assert client.post(f"{OPS}/meshes/import", json={"upload": "../../etc"}).status_code == 400


def test_bad_requests_are_400_and_logged(env, capsys):
    client, folder = env["client"], env["folder"]
    before = len(joblog.jobs(folder))
    cases = [
        ("surfaces/from-points", {}),
        ("surfaces/from-points", {"databases": ["nope"]}),
        ("surfaces/from-points", {"databases": ["topo"]}),
        ("surfaces/from-points", {"databases": ["drillholes"], "extend": {"mode": "pit"}}),
        ("surfaces/from-points", {"databases": ["drillholes"], "duplicates": "median"}),
        ("surfaces/from-points", {"databases": ["drillholes"],
                                  "extend": {"mode": "rectangle", "rectangle": [0, 0, 1]}}),
    ]  # fmt: skip
    for path, body in cases:
        r = client.post(f"{OPS}/{path}", json=body)
        assert r.status_code == 400, (path, body, r.text)
        assert isinstance(r.json()["detail"], str)
    entries = joblog.jobs(folder)[before:]
    assert len(entries) == len(cases) and all(e["status"] == "error" for e in entries)
    db_path = str(Project.open(folder).path_of("drillholes"))
    assert main(["surface", "from-points", "--database", db_path]) == 2
    assert main(["surface", "from-points", "--out", "x.obj"]) == 2
    assert main(["surface", "import", "missing.obj", "--out", "x.obj"]) == 2
    capsys.readouterr()


def test_replay_rebuilds_the_same_meshes(env, tmp_path, capsys):
    """Runs last: every geometry job above is redone by the CLI from the job log."""
    folder = env["folder"]
    done = [e for e in joblog.jobs(folder) if e["status"] == "done"]
    assert {e["operation"] for e in done} == {"surface_from_points", "import_mesh"}
    target = tmp_path / "replayed"
    assert main(["project", "replay", str(folder), "--into", str(target)]) == 0, capsys.readouterr()
    capsys.readouterr()
    for entry in done:
        out = entry["outputs"][0]
        replayed = Project.open(target).get(out["id"])
        assert (replayed.kind, replayed.path) == (out["kind"], out["path"])
        same_mesh(project_mesh(target, out["id"]), project_mesh(folder, out["id"]))
