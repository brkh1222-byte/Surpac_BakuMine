"""The import endpoints on real-world table shapes (synthetic fixtures of
``tests/drillholes/messy_fixtures.py``): what the Import panel does, without the browser.

The result must be the database a clean-CSV import of the same data gives, the job log
must hold the recipe, and ``bakumine project replay`` must rebuild the database from it.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.drillholes.importer import (
    ImportOptions,
    TableSource,
    import_drillholes,
    load_database,
)
from bakumine.core.project import Project

_SPEC = importlib.util.spec_from_file_location(
    "messy_fixtures", Path(__file__).parents[1] / "drillholes" / "messy_fixtures.py"
)
messy = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(messy)

HOLE_ID = {"pattern": r"SYN[_-]?0*(\d+)", "template": "SYN-{1:02d}"}
OPS = "/api/v1/ops"


@pytest.fixture
def setup(tmp_path):
    folder = tmp_path / "project"
    Project.create(folder, "shapes")
    set_scene_source(ProjectSceneSource(folder))
    tables = messy.truth()
    files = tmp_path / "files"
    files.mkdir()
    counts = messy.write_sheet_per_hole(
        files / "assays.xlsx", tables["assay"], depths="sample", qc=True
    )
    messy.write_collar_dms(
        files / "collars.xlsx", tables["collar"], tables["survey"],
        shifted={messy.hole_name(4): 120.0},
    )  # fmt: skip
    clean = messy.write_clean(files / "clean", tables)
    yield {"client": TestClient(app), "folder": folder, "files": files, "tables": tables,
           "qc": counts, "clean": clean}  # fmt: skip
    set_scene_source(None)


def ok(response):
    assert response.status_code == 200, response.text
    return response.json()


def upload(client, table: str, path: Path, add: bool = False):
    params = {"filename": path.name, **({"add": "true"} if add else {})}
    return client.put(f"{OPS}/uploads/{table}", params=params, content=path.read_bytes())


def clean_database(paths):
    return import_drillholes(
        TableSource(paths["collar"], {c: c for c in ("hole_id", "x", "y", "z", "max_depth")}),
        TableSource(paths["survey"], {c: c for c in ("hole_id", "depth", "dip", "azimuth")}),
        {"assay": TableSource(paths["assay"], {"hole_id": "hole_id", "from": "depth_from",
                                               "to": "depth_to", "ag": "ag", "au": "au"})},
        options=ImportOptions(delimiter=";", dip_convention="auto"),
    )  # fmt: skip


ASSAY_SPEC = {"sheets": ["SYN*"], "hole_id_from_sheet": True, "sample_depth": {"length": 1.0}}


def test_workbook_preview_sheets_header_and_rules(setup):
    client, tables = setup["client"], setup["tables"]
    first = ok(upload(client, "assay", setup["files"] / "assays.xlsx"))
    assert [s["name"] for s in first["sheets"]][-1] == "Summary"
    assert len(first["sheets"]) == messy.N_HOLES + 1
    assert [s["name"] for s in first["selected"]] == [first["sheets"][0]["name"]]
    assert (first["grid_header_row"], first["grid_header_rows"]) == (3, 2)
    assert first["grid"][2][4] == "Интервал" and first["grid"][3][4] == "аас"
    assert first["mapping"]["аас"] == "depth_from" and first["mapping"]["Au"] == "au"
    # no hole id column, and the depths are in the sample numbers: nothing usable yet
    assert first["missing"] == ["hole_id"] and first["rows"] == 0
    assert first["qc"]["no_interval"] > 0

    body = {"index": 0, "spec": {**ASSAY_SPEC, "preview_sheet": "Summary"}, "hole_id": HOLE_ID}
    preview = ok(client.post(f"{OPS}/uploads/assay/preview", json=body))
    assert len(preview["selected"]) == messy.N_HOLES and preview["missing"] == []
    assert preview["rows"] == len(tables["assay"])
    assert preview["qc"] == setup["qc"]
    assert preview["sample_depth"]["rows"] == len(tables["assay"])
    assert preview["sample_depth"]["unmatched"] == 0
    assert preview["grid_sheet"] == "Summary" and preview["grid"][0][:2] == ["Hole", "Samples"]
    columns = preview["prepared"]["columns"]
    row = dict(zip(columns, preview["prepared"]["rows"][0], strict=True))
    assert row["hole_id"] == messy.hole_name(1)
    want = tables["assay"].iloc[0]
    assert (float(row["depth_from"]), float(row["depth_to"])) == (want["from"], want["to"])

    # the user moves the header: one row, the second one -> the group names are gone
    moved = ok(client.post(f"{OPS}/uploads/assay/preview",
                           json={"spec": {"header_row": 4, "header_rows": 1}}))  # fmt: skip
    assert moved["grid_header_row"] == 4 and "аас" in moved["columns"]
    assert "Дээж №" not in moved["columns"]
    for bad, text in [
        ({"spec": {"sheets": ["Nope*"]}}, "no sheet matches"),
        ({"spec": {"file": "/etc/passwd"}}, "unknown source option"),
        ({"spec": {"header_row": 99}}, "beyond the last row"),
        ({"spec": {"sample_depth": {"patterns": ["("]}, "hole_id_from_sheet": True}},
         "sample number pattern"),
        ({"index": 3}, "no uploaded file"),
        ({"index": "0"}, "whole number"),
        ({"hole_id": {"pattern": "("}, "spec": {"hole_id_from_sheet": True}}, "hole id pattern"),
    ]:  # fmt: skip
        response = client.post(f"{OPS}/uploads/assay/preview", json=bad)
        assert response.status_code == 400 and text in response.json()["detail"], response.text
    assert client.post(f"{OPS}/uploads/a*/preview", json={}).status_code == 400
    assert upload(client, "assay", setup["files"] / "collars.xlsx").status_code == 200
    bad = setup["files"] / "report.pdf"
    bad.write_bytes(b"%PDF-1.4")
    assert upload(client, "assay", bad).status_code == 400


def test_import_equals_clean_and_replays_from_the_job_log(setup, tmp_path, capsys):
    client, folder = setup["client"], setup["folder"]
    ok(upload(client, "collar", setup["files"] / "collars.xlsx"))
    ok(upload(client, "assay", setup["files"] / "assays.xlsx"))
    # a file left from an earlier session for a table the dialog does not show is not imported
    ok(upload(client, "geology", setup["clean"]["assay"]))
    response = client.post(f"{OPS}/import", json={"tables": {"collar": [{}], "lith": [{}]}})
    assert response.status_code == 400 and "no uploaded file for table(s) ['lith']" in response.text
    result = ok(
        client.post(
            f"{OPS}/import",
            json={"name": "shapes", "tables": {"collar": [{}], "assay": [ASSAY_SPEC]},
                  "hole_id": HOLE_ID,
                  "options": {"dip_convention": "auto", "coordinate_order": "auto"}},
        )  # fmt: skip
    )
    want = clean_database(setup["clean"])
    assert result["tables"] == {"collar": messy.N_HOLES, "survey": messy.N_HOLES,
                                "assay": len(want.intervals["assay"])}  # fmt: skip
    assert result["qc"] == {"assay": setup["qc"]}
    assert result["survey"]["generated_from_collar"] == messy.N_HOLES
    assert result["coordinates"]["disagree"] == 1
    assert result["coordinates"]["rows"][0]["hole_id"] == messy.hole_name(4)
    assert (result["dip_convention"], result["coordinate_order"]) == ("positive_down", "yxz")
    assert result["sample_depth"]["assay"][0]["unmatched"] == 0
    assert result["errors"] == 0
    assert any("below_detection" in note for note in result["notes"])

    project = Project.open(folder)
    db = load_database(project.path_of(result["id"]))
    shared = list(want.intervals["assay"].columns)
    pd.testing.assert_frame_equal(db.intervals["assay"][shared], want.intervals["assay"])
    assert db.intervals["assay"]["sample_id"].str.startswith("SYN").all()
    pd.testing.assert_frame_equal(db.survey, want.survey)
    moved = want.collar["hole_id"] == messy.hole_name(4)
    pd.testing.assert_frame_equal(
        db.collar.loc[~moved, ["hole_id", "x", "y", "z", "max_depth"]], want.collar[~moved]
    )
    assert len(db.qc["assay"]) == sum(setup["qc"].values())
    assert db.metadata.recipe == result["recipe"]

    # the job log holds the recipe with the archived files, and replays with the CLI
    done = [e for e in joblog.read(folder) if e["operation"] == "import" and e["status"] == "done"]
    (entry,) = done
    recipe = entry["replay"]["recipe"]
    assert recipe["tables"]["assay"][0]["sheets"] == ["SYN*"]
    assert recipe["tables"]["assay"][0]["map"]["Au"] == "au"
    for sources in recipe["tables"].values():
        assert (folder / sources[0]["file"]).is_file()
    assert "recipe" not in entry["summary"]
    capsys.readouterr()
    assert main(["project", "replay", str(folder), "--into", str(tmp_path / "again")]) == 0
    assert "import --recipe" in capsys.readouterr().out
    replayed = Project.open(tmp_path / "again")
    again = load_database(replayed.path_of(result["id"]))
    for name, table in db.tables.items():
        pd.testing.assert_frame_equal(again.tables[name], table)
    pd.testing.assert_frame_equal(again.qc["assay"], db.qc["assay"])
    script = tmp_path / "replay.sh"
    assert main(["project", "export-log", str(folder), "--out", str(script)]) == 0
    assert "$BAKUMINE import --recipe" in script.read_text()


def test_several_files_per_table_are_merged(setup, tmp_path):
    client, folder, tables = setup["client"], setup["folder"], setup["tables"]
    assay = tables["assay"]
    cut = len(assay) // 2
    parts = [tmp_path / "first.csv", tmp_path / "second.csv"]
    assay.iloc[: cut + 3].to_csv(parts[0], index=False)
    late = assay.iloc[cut:].copy()
    late.iloc[0, late.columns.get_loc("ag")] = "77"
    late.to_csv(parts[1], index=False)
    for name in ("collar", "survey"):
        frame = pd.read_csv(setup["clean"][name], sep=";")
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
        ok(upload(client, name, tmp_path / f"{name}.csv"))
    assert ok(upload(client, "assay", parts[0]))["n_files"] == 1
    second = ok(upload(client, "assay", parts[1], add=True))
    assert (second["index"], second["n_files"], second["file"]) == (1, 2, "assay+2.csv")
    assert second["rows"] == len(assay) - cut
    assert sorted(p.name for p in (folder / "uploads").iterdir()) == [
        "assay+2.csv", "assay.csv", "collar.csv", "survey.csv"]  # fmt: skip
    summary = ok(client.get(f"{OPS}/project"))
    assert [u["name"] for u in summary["uploads"]] == ["assay", "collar", "survey"]
    body = {"name": "merged", "options": {"dip_convention": "auto"}}
    result = ok(client.post(f"{OPS}/import", json=body))
    assert result["tables"]["assay"] == len(assay)
    conflicts = result["conflicts"]["assay"]
    assert (conflicts["merged"], conflicts["conflicts"]) == (3, 1)
    assert conflicts["precedence"] == "first"
    assert conflicts["rows"][0]["dropped"] == 77.0
    assert conflicts["rows"][0]["dropped_source"] == "assay+2.csv"
    db = load_database(Project.open(folder).path_of(result["id"]))
    want = clean_database(setup["clean"])
    pd.testing.assert_frame_equal(db.intervals["assay"], want.intervals["assay"])
    body = {"name": "merged2", "merge": {"precedence": "last"},
            "options": {"dip_convention": "auto"}}  # fmt: skip
    last = ok(client.post(f"{OPS}/import", json=body))
    assert last["conflicts"]["assay"]["rows"][0]["kept"] == 77.0
    # replacing the table's file drops the added ones; removing removes all
    assert ok(upload(client, "assay", parts[0]))["n_files"] == 1
    ok(upload(client, "assay", parts[1], add=True))
    ok(client.delete(f"{OPS}/uploads/assay"))
    assert sorted(p.name for p in (folder / "uploads").iterdir()) == ["collar.csv", "survey.csv"]
    # without a survey table and without azimuth / dip in the collar: said, not guessed
    ok(client.delete(f"{OPS}/uploads/survey"))
    response = client.post(f"{OPS}/import", json={"name": "x"})
    assert response.status_code == 400 and "no survey table" in response.json()["detail"]
    ok(client.delete(f"{OPS}/uploads/collar"))
    response = client.post(f"{OPS}/import", json={"name": "x"})
    assert response.status_code == 400 and "upload the collar table" in response.json()["detail"]
