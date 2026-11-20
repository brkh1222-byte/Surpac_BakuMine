"""``POST /api/v1/ops/intercepts`` equals ``bakumine intercepts``; log, replay, use as domains."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.drillholes.importer import load_database
from bakumine.core.project import Project

ROOT = Path(__file__).resolve().parents[2]
SYNTH = ROOT / "data" / "synthetic" / "default"

pytestmark = pytest.mark.skipif(
    not (SYNTH / "assay.csv").is_file(), reason="data/synthetic/default has not been generated"
)

PARAMS = {
    "cutoff": 0.5,
    "min_thickness": 2.0,
    "max_internal_waste": 2.0,
    "max_total_waste": 6.0,
    "accumulation_rule": True,
    "gap_tolerance": 0.5,
    "top_cut": 10.0,
    "including_cutoffs": [2.0],
    "including_min_thickness": 1.0,
    "body_dip": 50.0,
    "body_dip_azimuth": 125.0,
}
CLI_OPTIONS = [
    "--cutoff", "0.5", "--min-thickness", "2", "--max-internal-waste", "2",
    "--max-total-waste", "6", "--accumulation-rule", "--gap-tolerance", "0.5",
    "--top-cut", "10", "--including", "2", "--including-min-thickness", "1",
    "--body-dip", "50", "--body-dip-azimuth", "125",
]  # fmt: skip


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    folder = tmp_path_factory.mktemp("intercepts") / "p"
    Project.create(folder, "intercepts")
    set_scene_source(ProjectSceneSource(folder))
    client = TestClient(app)
    for table in ("collar", "survey", "assay", "geology"):
        r = client.put(
            f"/api/v1/ops/uploads/{table}",
            params={"filename": f"{table}.csv"},
            content=(SYNTH / f"{table}.csv").read_bytes(),
        )
        assert r.status_code == 200, r.text
    db = client.post("/api/v1/ops/import", json={"name": "drillholes"}).json()["id"]
    yield client, folder, db
    set_scene_source(None)


@pytest.fixture(scope="module")
def flagged(api):
    client, _, db = api
    r = client.post(
        "/api/v1/ops/intercepts",
        json={"database": db, "table": "assay", "grade": "au", "params": PARAMS,
              "save_as": "mineralised"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    return r.json()


@pytest.fixture(scope="module")
def cli_db(tmp_path_factory) -> Path:
    db = tmp_path_factory.mktemp("intercepts_cli") / "db"
    code = main(
        ["import", "--collar", str(SYNTH / "collar.csv"), "--survey", str(SYNTH / "survey.csv"),
         "--assay", str(SYNTH / "assay.csv"), "--geology", str(SYNTH / "geology.csv"),
         "--auto", "--out", str(db)]
    )  # fmt: skip
    assert code == 0
    code = main(
        ["intercepts", str(db), "--grade", "au", *CLI_OPTIONS, "--save-as", "mineralised"]
    )
    assert code == 0
    return db


def test_endpoint_equals_cli(api, flagged, cli_db):
    _, folder, db = api
    path = Project.open(folder).path_of(db)
    ours, theirs = load_database(path), load_database(cli_db)
    assert len(ours.intervals["mineralised"]) > 10
    pd.testing.assert_frame_equal(ours.intervals["mineralised"], theirs.intervals["mineralised"])
    relative = Path(path).relative_to(folder).as_posix()
    assert flagged["file"] == f"{relative}/intercepts_mineralised.csv"
    assert (folder / flagged["file"]).read_bytes() == (
        cli_db / "intercepts_mineralised.csv"
    ).read_bytes()
    # the response is the same table (None for NaN) with its summary
    table = pd.read_csv(folder / flagged["file"])
    assert len(flagged["table"]) == len(table) == flagged["n_intercepts"] + flagged["n_including"]
    assert flagged["n_intercepts"] > 0 and flagged["n_including"] > 0
    assert flagged["table"][0]["hole_id"] == table["hole_id"][0]
    assert flagged["table"][0]["true_thickness"] == pytest.approx(table["true_thickness"][0])
    main_rows = table[table["level"] == "intercept"]
    assert flagged["grade_x_m"] == pytest.approx(main_rows["grade_x_m"].sum())
    assert (main_rows["grade"] >= 0.5 - 1e-9).all()
    assert set(ours.intervals["mineralised"]["code"]) == {"MIN", "WASTE"}


def test_params_file_gives_the_same_result(cli_db, tmp_path, capsys):
    file = tmp_path / "params.json"
    file.write_text(json.dumps(PARAMS))
    out = tmp_path / "again.csv"
    intervals = tmp_path / "intervals.csv"
    code = main(
        ["intercepts", str(cli_db), "--grade", "au", "--params", str(file), "--out", str(out),
         "--intervals-out", str(intervals)]
    )  # fmt: skip
    assert code == 0
    assert out.read_bytes() == (cli_db / "intercepts_mineralised.csv").read_bytes()
    assert "intercepts" in capsys.readouterr().out
    saved = load_database(cli_db).intervals["mineralised"]
    pd.testing.assert_frame_equal(pd.read_csv(intervals), saved, check_exact=False, rtol=1e-12)
    # an option overrides the file
    assert main(["intercepts", str(cli_db), "--grade", "au", "--params", str(file),
                 "--cutoff", "1.9", "--out", str(out)]) == 0  # fmt: skip
    again = pd.read_csv(out)
    assert len(again) and set(again["cutoff"]) == {1.9, 2.0}
    assert (again["grade"] >= again["cutoff"] - 1e-9).all()


def test_cli_errors(cli_db, capsys):
    assert main(["intercepts", str(cli_db), "--grade", "au"]) == 2  # no cut-off
    assert main(["intercepts", str(cli_db), "--grade", "nope", "--cutoff", "1"]) == 2
    assert main(["intercepts", str(cli_db), "--grade", "au", "--cutoff", "1",
                 "--save-as", "mineralised"]) == 2  # exists, no --overwrite  # fmt: skip
    assert "already has a table" in capsys.readouterr().err
    assert main(["intercepts", str(cli_db / "nope"), "--grade", "au", "--cutoff", "1"]) == 2


def test_job_is_logged_and_table_is_listed(api, flagged):
    client, folder, db = api
    entry = [e for e in joblog.read(folder) if e["operation"] == "intercepts"][-1]
    assert entry["status"] == "done" and entry["modifies"] is None
    assert entry["inputs"][0]["id"] == db and entry["outputs"][0]["id"] == db
    assert entry["summary"]["n_intercepts"] == flagged["n_intercepts"]
    assert "table" not in entry["summary"]
    assert entry["replay"]["params"]["cutoff"] == 0.5
    tables = next(
        o for o in client.get("/api/v1/ops/project").json()["objects"] if o["id"] == db
    )["tables"]
    assert tables["mineralised"]["attributes"] == ["code"]


def test_intervals_are_a_domain_table_for_compositing(api, flagged):
    client, folder, db = api
    r = client.post(
        "/api/v1/ops/composite",
        json={"database": db, "table": "assay", "mode": "domain", "length": 2,
              "domain_table": "mineralised", "domain_column": "code", "name": "comp_min"},
    )  # fmt: skip
    assert r.status_code == 200, r.text
    assert set(r.json()["domains"]) >= {"MIN", "WASTE"}
    composites = pd.read_parquet(Project.open(folder).path_of(r.json()["id"]))
    by_domain = composites.groupby("domain")["au"].mean()
    assert by_domain["MIN"] > 0.5 > by_domain["WASTE"]


def test_replay_gives_the_same_table(api, flagged, tmp_path):
    _, folder, db = api
    new = tmp_path / "replayed"
    assert main(["project", "replay", str(folder), "--into", str(new)]) == 0
    ours = load_database(Project.open(folder).path_of(db)).intervals["mineralised"]
    theirs = load_database(Project.open(new).path_of(db)).intervals["mineralised"]
    pd.testing.assert_frame_equal(ours, theirs)


def test_bad_requests_are_400_and_logged(api, flagged):
    client, folder, db = api
    post = lambda body: client.post("/api/v1/ops/intercepts", json=body)  # noqa: E731
    assert post({"database": "nope", "grade": "au", "params": {"cutoff": 1}}).status_code == 400
    assert post({"database": db, "params": {"cutoff": 1}}).status_code == 400
    assert post({"database": db, "grade": "au"}).status_code == 400
    assert post({"database": db, "grade": "au", "params": {}}).status_code == 400
    assert post({"database": db, "grade": "au", "params": {"cutoff": 1, "x": 2}}).status_code == 400
    assert post({"database": db, "grade": "zz", "params": {"cutoff": 1}}).status_code == 400
    exists = post({"database": db, "grade": "au", "params": {"cutoff": 1},
                   "save_as": "mineralised"})  # fmt: skip
    assert exists.status_code == 400 and "already has a table" in exists.json()["detail"]
    assert post({"database": db, "grade": "au", "params": {"cutoff": 1},
                 "save_as": "assay"}).status_code == 400  # fmt: skip
    assert post({"database": db, "grade": "au", "params": {"cutoff": 1},
                 "save_as": "../x"}).status_code == 400  # fmt: skip
    assert joblog.read(folder)[-1]["status"] == "error"
    # without save_as nothing in the database changes, and overwrite replaces
    before = load_database(Project.open(folder).path_of(db)).intervals["mineralised"]
    assert post({"database": db, "grade": "au", "params": {"cutoff": 1}}).status_code == 200
    after = load_database(Project.open(folder).path_of(db)).intervals["mineralised"]
    pd.testing.assert_frame_equal(before, after)
