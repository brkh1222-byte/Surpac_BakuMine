"""`bakumine geology cv` (P9.T11), end to end on a small made-up log."""

import json

import numpy as np
import pandas as pd

from bakumine.cli import main

HOLES = 6
LENGTH = 60.0


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_geology_cv_end_to_end(tmp_path, capsys):
    ids = [f"H{i + 1}" for i in range(HOLES)]
    x = 20.0 * np.arange(HOLES)
    contact = 25.0 + 0.2 * x
    collar = pd.DataFrame(
        {"hole_id": ids, "x": x, "y": 15.0 * (np.arange(HOLES) % 2), "z": 0.0,
         "max_depth": LENGTH}
    )  # fmt: skip
    survey = pd.DataFrame({"hole_id": ids, "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    rows = []
    for hole, depth in zip(ids, contact, strict=True):
        rows += [(hole, 0.0, depth, "up"), (hole, depth, LENGTH, "LOW")]
    log = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "lith"])
    for name, frame in {"collar": collar, "survey": survey, "geology": log}.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    db = tmp_path / "db"
    code, _, err = run(
        capsys, "import", "--collar", tmp_path / "collar.csv", "--survey", tmp_path / "survey.csv",
        "--geology", tmp_path / "geology.csv", "--out", db, "--auto",
    )  # fmt: skip
    assert code == 0, err
    units = {
        "format": "bakumine.units", "version": 1, "name": "made-up",
        "source": {"table": "geology", "column": "lith"},
        "units": [{"code": "UP", "kind": "lithology"}, {"code": "LOW", "kind": "lithology"}],
        "weathering_zones": [],
    }  # fmt: skip
    (tmp_path / "units.json").write_text(json.dumps(units), encoding="utf-8")
    spec = {
        "format": "bakumine.geology-spec", "version": 1, "units": "units.json",
        "lithology_source": {"table": "geology", "column": "lith"},
        "lithology": {"events": [{"mode": "argmax", "units": ["UP", "LOW"]}]},
        "faults": None,
    }  # fmt: skip
    (tmp_path / "spec.json").write_text(json.dumps(spec), encoding="utf-8")

    out, summary = tmp_path / "out" / "cv.csv", tmp_path / "out" / "cv.json"
    code, text, err = run(
        capsys, "geology", "cv", db, "--spec", tmp_path / "spec.json", "--folds", 3,
        "--seed", 1, "--step", 0.5, "--out", out, "--summary", summary,
    )  # fmt: skip
    assert code == 0, err
    assert "3 folds by hole, 6 holes, seed 1" in text
    assert "What this does not prove" in text and "beyond the drilling" in text

    pieces = pd.read_csv(out)
    assert set(pieces["layer"]) == {"lith"} and set(pieces["hole_id"]) == set(ids)
    assert np.isclose(pieces["length"].sum(), HOLES * LENGTH)
    agreement = pieces.loc[pieces["logged"] == pieces["predicted"], "length"].sum() / (
        HOLES * LENGTH
    )
    data = json.loads(summary.read_text(encoding="utf-8"))
    assert np.isclose(data["layers"]["lith"]["agreement"], agreement)
    assert f"agreement by length: {agreement:.4f}" in text
    kinds = [e["kind"] for e in data["layers"]["lith"]["provenance"]]
    assert kinds.count("measured") >= 2 and kinds.count("assumed") == 1

    code, _, err = run(capsys, "geology", "cv", db, "--spec", tmp_path / "none.json",
                       "--out", out)  # fmt: skip
    assert code == 2 and "does not exist" in err
