"""``bakumine geology build`` (P9.T14), end to end on a small made-up log.

Independent check: the model's labels against the analytic contact plane the log was
written from. All data are made up.
"""

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.geology.model import GeologyModel
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io.obj import write_obj

N, STEP, LENGTH = 5, 20.0, 60.0  # 5 x 5 vertical holes 20 m apart, 60 m deep


def contact_depth(x, y):
    """Depth of the UP / LOW contact below the flat ground at z = 0."""
    return 22.0 + 0.2 * x + 0.1 * y


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    folder = tmp_path_factory.mktemp("geology_build_cli")
    axis = STEP * np.arange(N)
    x, y = (v.ravel() for v in np.meshgrid(axis, axis, indexing="ij"))
    ids = [f"H{i + 1:02d}" for i in range(len(x))]
    collar = pd.DataFrame({"hole_id": ids, "x": x, "y": y, "z": 0.0, "max_depth": LENGTH})
    survey = pd.DataFrame({"hole_id": ids, "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    rows = []
    for hole, depth in zip(ids, contact_depth(x, y), strict=True):
        rows += [(hole, 0.0, depth, "up"), (hole, depth, LENGTH, "LOW")]
    log = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "lith"])
    for name, frame in {"collar": collar, "survey": survey, "geology": log}.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    assert main(["import", "--collar", str(folder / "collar.csv"), "--survey",
                 str(folder / "survey.csv"), "--geology", str(folder / "geology.csv"),
                 "--out", str(folder / "db"), "--auto"]) == 0  # fmt: skip
    units = {
        "format": "bakumine.units", "version": 1, "name": "made-up",
        "source": {"table": "geology", "column": "lith"},
        "units": [{"code": "UP", "kind": "lithology", "colour": "#aa8844", "density": 2.5,
                   "density_source": "measured"},
                  {"code": "LOW", "kind": "lithology", "density": 2.7,
                   "density_source": "assumed"}],
        "weathering_zones": [],
    }  # fmt: skip
    (folder / "units.json").write_text(json.dumps(units), encoding="utf-8")
    ground = np.array([[-50.0, -50, 0], [150, -50, 0], [150, 150, 0], [-50, 150, 0]])
    write_obj(folder / "ground.obj", Mesh(ground, np.array([[0, 1, 2], [0, 2, 3]])), decimals=None)
    spec = {
        "format": "bakumine.geology-spec", "version": 1, "name": "made-up layers",
        "database": "db", "units": "units.json", "topography": "ground.obj",
        "extent": {"rectangle": [-10, -10, 90, 90], "z": [-70, 10]}, "resolution": 2.5,
        "lithology": {"events": [{"mode": "sequence", "units": ["UP", "LOW"]}]},
        "faults": None, "cover": None,
    }  # fmt: skip
    (folder / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    return folder, spec, len(ids), len(log)


def test_geology_build_end_to_end(data, capsys):
    folder, _, n_holes, n_intervals = data
    out, summary = folder / "out" / "made.geology", folder / "out" / "summary.json"
    code, text, err = run(capsys, "geology", "build", "--spec", folder / "spec.json",
                          "--out", out, "--summary", summary)  # fmt: skip
    assert code == 0, err
    assert "geology model 'made-up layers'" in text and "grid 41 x 41 x 33 nodes at 2.5 m" in text
    assert f"{n_holes} holes ({n_holes} with a trace), {n_intervals} logged intervals" in text
    assert "lith: UP, LOW, AIR" in text and "Volume per unit (m3)" in text
    for title in ("Measured: real site data used", "Assumed", "Missing"):
        assert title in text
    assert "Fault data" in text and "Density samples" in text and "Topography" in text
    for name in ("geology.json", "units.json", "spec.json", "provenance.json", "diagnostics.json",
                 "topography.obj", "meshes/lith/UP.obj", "meshes/lith/LOW.obj"):  # fmt: skip
        assert (out / name).is_file(), name
    figures = json.loads(summary.read_text())
    assert figures["inputs"]["holes"] == n_holes
    assert figures["provenance"]["missing"] >= 2 and figures["provenance"]["measured"] >= 4
    record = json.loads((out / "provenance.json").read_text())
    kinds = {e["kind"] for e in record["entries"]}
    assert kinds == {"measured", "assumed", "missing"}
    counted = {e["item"]: e.get("count") for e in record["entries"]}
    assert counted["Drillhole database"] == n_holes and counted["Rock-type log"] == n_intervals
    assert counted["Log of unit UP"] == n_holes and counted["Log of unit LOW"] == n_holes
    assert sum(figures["provenance"].values()) == len(record["entries"])

    # the model against the plane the log was written from, inside the drilled area
    model = GeologyModel.load(out)
    pts = np.random.default_rng(5).uniform([0, 0, -LENGTH], [80, 80, 5], size=(40000, 3))
    upper = -pts[:, 2] < contact_depth(pts[:, 0], pts[:, 1])
    true = np.where(pts[:, 2] > 0, "AIR", np.where(upper, "UP", "LOW"))
    got = model.decode(model.labels(pts), "lith")
    accuracy = float((got == true).mean())
    print(f"\nmade-up layers: labels equal the analytic model at {accuracy:.4%} of 40000 points")
    assert accuracy >= 0.99
    # UP inside the drilled hull: 80 x 80 m in plan, mean contact depth 22 + 0.2 * 40 + 0.1 * 40
    volumes = json.loads((out / "diagnostics.json").read_text())["volumes"]["units"]
    assert volumes["UP"]["inside_hull"] == pytest.approx(80 * 80 * 34.0, rel=0.02)

    # an existing folder is kept unless --overwrite is given
    code, _, err = run(capsys, "geology", "build", "--spec", folder / "spec.json", "--out", out)
    assert code == 2 and "--overwrite" in err
    code, _, err = run(capsys, "geology", "build", "--spec", folder / "spec.json", "--out", out,
                       "--overwrite")  # fmt: skip
    assert code == 0, err
    again = GeologyModel.load(out)
    np.testing.assert_array_equal(again.labels(pts, "lith"), model.labels(pts, "lith"))


def test_geology_build_database_argument_base_and_errors(data, tmp_path, capsys):
    folder, spec, _, _ = data
    # the spec in another folder: --base says where its paths start; DB replaces the database
    moved = dict(spec, database=None, resolution=5.0)
    (tmp_path / "moved.json").write_text(json.dumps(moved), encoding="utf-8")
    out = tmp_path / "moved.geology"
    code, _, err = run(capsys, "geology", "build", "--spec", tmp_path / "moved.json", "--out", out)
    assert code == 2 and "units: no file" in err
    code, _, err = run(capsys, "geology", "build", "--spec", tmp_path / "moved.json",
                       "--base", folder, "--out", out)  # fmt: skip
    assert code == 2 and "database" in err and not out.exists()
    code, text, err = run(capsys, "geology", "build", folder / "db", "--spec",
                          tmp_path / "moved.json", "--base", folder, "--out", out)  # fmt: skip
    assert code == 0, err
    assert "grid 21 x 21 x 17 nodes at 5 m" in text
    assert json.loads((out / "spec.json").read_text())["database"] is None

    bad = dict(spec, lithology={"events": [{"mode": "layers", "units": ["UP", "LOW"]}]})
    (folder / "bad.json").write_text(json.dumps(bad), encoding="utf-8")
    code, _, err = run(capsys, "geology", "build", "--spec", folder / "bad.json",
                       "--out", tmp_path / "bad.geology")  # fmt: skip
    assert code == 2 and not (tmp_path / "bad.geology").exists()
    assert "lithology.events[0].mode: must be one of argmax, sequence, intrusion" in err
    code, _, err = run(capsys, "geology", "build", "--spec", folder / "none.json",
                       "--out", tmp_path / "bad.geology")  # fmt: skip
    assert code == 2 and "does not exist" in err
