"""Rows that are not data: totals, rows without a hole id, collars without coordinates."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.recipe import make_recipe, run_recipe
from bakumine.core.drillholes.tables import is_total_label


@pytest.mark.parametrize(
    "text", ["Total", "TOTAL:", "Нийт", "НИЙТ", "НЭГТГЭЛ", "Итого", "Итого по участку",
             "Всего", "Sum", "Grand total", "Sub-total", "Нийт дүн", "Дүн"],
)  # fmt: skip
def test_total_labels(text):
    assert is_total_label(text)


@pytest.mark.parametrize(
    "text", ["", "SUM-01", "Total2", "SYN-01", "17", "Summit", "Totalizer", "Core", "итог"]
)
def test_not_total_labels(text):
    assert not is_total_label(text)


def write_tables(folder):
    holes = ["SYN-01", "SYN-02", "SUM-03", "SYN-04"]
    collar = pd.DataFrame(
        {
            "No": ["1", "2", "3", "4", "5", "Нийт", "", ""],
            "Hole": [*holes, "SYN-05", "", "", "Total"],
            "East": ["500100", "500200", "500300", "500400", "", "", "", ""],
            "North": ["5300100", "5300200", "5300300", "5300400", "", "", "", ""],
            "RL": ["1200", "1201", "1202", "1203", "", "", "", ""],
            "Depth": ["30", "30", "30", "30", "30", "150", "", "150"],
            "Azimuth": ["0"] * 5 + ["", "", ""],
            "Dip": ["-60"] * 5 + ["", "", ""],
            "Remark": ["", "", "", "", "planned", "", "4 drilled, 1 planned", ""],
        }
    )
    collar.to_csv(folder / "collar.csv", index=False)
    rows = []
    for hole in holes:
        for k in range(3):
            rows.append([hole, str(10 * k), str(10 * k + 10), f"{0.1 * (k + 1):.1f}"])
        rows.append(["Итого", "", "", "0.6"])  # a subtotal under each hole
    rows.append(["", "0", "10", "9.9"])  # a sample of no hole
    rows.append(["Всего", "", "", "2.4"])
    assay = pd.DataFrame(rows, columns=["Hole", "From", "To", "Au, g/t"])
    assay.to_csv(folder / "assay.csv", index=False)
    return holes


def test_totals_and_nameless_rows_are_dropped_and_counted(tmp_path, capsys):
    holes = write_tables(tmp_path)
    recipe = make_recipe(
        {"collar": [{"file": "collar.csv"}], "assay": [{"file": "assay.csv"}]},
        {"dip_convention": "negative_down"},
    )
    db = run_recipe(recipe, tmp_path)
    assert db.collar["hole_id"].tolist() == holes  # SUM-03 is a hole, not a sum
    assert db.survey["hole_id"].tolist() == holes
    assay = db.intervals["assay"]
    assert len(assay) == 12 and set(assay["hole_id"]) == set(holes)
    np.testing.assert_allclose(assay["au"].sum(), 4 * 0.6)
    assert db.metadata.reports["dropped_rows"] == {
        "collar": {"total_rows": 2, "rows_without_hole_id": 1, "collars_without_coordinates": 1,
                   "holes_without_coordinates": ["SYN-05"]},
        "assay": {"total_rows": 5, "rows_without_hole_id": 1},
    }  # fmt: skip
    notes = {(n["table"], n["kind"]): n for n in db.metadata.notes}
    assert notes[("collar", "total_rows_dropped")]["examples"] == ["Total", "Нийт"]
    assert notes[("collar", "total_rows_dropped")]["rows"] == [7, 9]  # spreadsheet rows
    assert notes[("collar", "rows_without_hole_id_dropped")]["rows"] == [8]
    assert notes[("collar", "collars_without_coordinates_dropped")]["examples"] == ["SYN-05"]
    assert notes[("assay", "total_rows_dropped")]["count"] == 5
    # no QC row was made of them
    assert "assay" not in db.qc

    code = main(["import", "--collar", str(tmp_path / "collar.csv"), "--assay",
                 str(tmp_path / "assay.csv"), "--auto", "--dry-run"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert ("collar rows dropped: 2 total / summary row(s), 1 without a hole id, "
            "1 collar(s) without coordinates (SYN-05)") in out  # fmt: skip
    assert "assay rows dropped: 5 total / summary row(s), 1 without a hole id" in out
    assert "holes 4," in out


def test_a_collar_gets_its_coordinates_from_another_file_before_it_is_judged(tmp_path):
    write_tables(tmp_path)
    more = pd.DataFrame(
        {"Hole": ["SYN-05"], "East": [500500.0], "North": [5300500.0], "RL": [1204.0]}
    )
    more.to_csv(tmp_path / "survey_office.csv", index=False)
    recipe = make_recipe({"collar": [{"file": "collar.csv"}, {"file": "survey_office.csv"}]})
    db = run_recipe(recipe, tmp_path)
    assert db.collar["hole_id"].tolist()[-1] == "SYN-05" and len(db.collar) == 5
    row = db.collar.iloc[-1]
    assert (row["x"], row["y"], row["z"], row["max_depth"]) == (500500.0, 5300500.0, 1204.0, 30.0)
    assert "collars_without_coordinates" not in db.metadata.reports["dropped_rows"]["collar"]


def test_longitude_and_latitude_count_as_coordinates(tmp_path):
    collar = pd.DataFrame(
        {
            "Hole": ["SYN-01", "SYN-02", "SYN-03"],
            "Longitude": ["100 30 00", "100 30 10", ""],
            "Latitude": ["45 00 00", "45 00 05", ""],
            "RL": [1200, 1201, 1202],
            "Depth": [50, 50, 50],
            "Azimuth": [0, 0, 0],
            "Dip": [-90, -90, -90],
        }
    )
    collar.to_csv(tmp_path / "collar.csv", index=False)
    db = run_recipe(make_recipe({"collar": [{"file": "collar.csv"}]}), tmp_path)
    assert db.collar["hole_id"].tolist() == ["SYN-01", "SYN-02"]
    assert db.collar[["x", "y"]].notna().all().all()
    assert db.metadata.reports["dropped_rows"]["collar"]["holes_without_coordinates"] == ["SYN-03"]


def test_sheet_names_are_not_total_labels(tmp_path):
    # one sheet per hole: the hole id is the sheet name, whatever its first cell says
    path = tmp_path / "assays.xlsx"
    frame = pd.DataFrame({"From": [0, 1, 2], "To": [1, 2, 3], "Au": [0.1, 0.2, 0.3]})
    total = pd.DataFrame({"From": ["Total"], "To": [""], "Au": [0.6]})
    with pd.ExcelWriter(path) as writer:
        pd.concat([frame, total]).to_excel(writer, sheet_name="SYN-01", index=False)
    collar = pd.DataFrame({"Hole": ["SYN-01"], "East": [1.0], "North": [2.0], "RL": [3.0],
                           "Azimuth": [0], "Dip": [-90]})  # fmt: skip
    collar.to_csv(tmp_path / "collar.csv", index=False)
    recipe = make_recipe(
        {"collar": [{"file": "collar.csv"}],
         "assay": [{"file": "assays.xlsx", "sheets": "*", "hole_id_from_sheet": True}]}
    )  # fmt: skip
    db = run_recipe(recipe, tmp_path)
    assert len(db.intervals["assay"]) == 3
    assert db.metadata.reports["dropped_rows"]["assay"] == {"total_rows": 1}


# ------------------------------------------------------------------ hole id once per hole
def write_merged_layout(folder):
    """The hole id written once per hole (a merged cell), in the assay and survey files."""
    (folder / "collar.csv").write_text(
        "Hole,East,North,RL,Depth\nA,0,0,100,10\nB,50,0,100,10\n", encoding="utf-8"
    )
    (folder / "survey.csv").write_text(
        "Hole,Depth,Dip,Azimuth\nA,0,-60,90\n,5,-61,91\nB,0,-90,0\n,,,\n,5,-89,0\n",
        encoding="utf-8",
    )
    (folder / "assay.csv").write_text(
        "Hole,From,To,Au\n"
        "A,0,1,0.1\n,1,2,0.2\n,\"2,5\",3,0.3\n"
        ",,,0.6\n"  # a sum under the hole: no depths, not a sample
        "B,0,1,0.4\n,1,2,0.5\n",
        encoding="utf-8",
    )


def test_a_hole_id_written_once_per_hole_is_carried_down(tmp_path, capsys):
    write_merged_layout(tmp_path)
    tables = {
        "collar": [{"file": "collar.csv"}],
        "survey": [{"file": "survey.csv"}],
        "assay": [{"file": "assay.csv", "qc": {"mode": "off"}}],
    }
    db = run_recipe(make_recipe(tables), tmp_path)
    assay = db.intervals["assay"]
    assert assay["hole_id"].tolist() == ["A", "A", "A", "B", "B"]
    assert assay["depth_from"].tolist() == [0.0, 1.0, 2.5, 0.0, 1.0]
    assert assay["au"].tolist() == [0.1, 0.2, 0.3, 0.4, 0.5]
    assert db.survey["hole_id"].tolist() == ["A", "A", "B", "B"]
    assert db.survey["depth"].tolist() == [0.0, 5.0, 0.0, 5.0]
    notes = {(n["table"], n["kind"]): n["count"] for n in db.metadata.notes}
    assert notes[("assay", "hole_id_carried_down")] == 3
    assert notes[("survey", "hole_id_carried_down")] == 2
    # the row without depths is still dropped and counted
    assert db.metadata.reports["dropped_rows"]["assay"] == {"rows_without_hole_id": 1}
    assert ("collar", "hole_id_carried_down") not in notes

    code = main(["import", "--collar", str(tmp_path / "collar.csv"), "--survey",
                 str(tmp_path / "survey.csv"), "--assay", str(tmp_path / "assay.csv"),
                 "--auto", "--dry-run"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert "assay: 3 row(s) without a hole id took the id of the row above" in out
    assert "survey: 2 row(s) without a hole id took the id of the row above" in out


def test_carry_down_can_be_turned_off_and_is_not_for_collars(tmp_path, capsys):
    write_merged_layout(tmp_path)
    tables = {
        "collar": [{"file": "collar.csv"}],
        "survey": [{"file": "survey.csv", "carry_down": False}],
        "assay": [{"file": "assay.csv", "qc": {"mode": "off"}, "carry_down": False}],
    }
    db = run_recipe(make_recipe(tables), tmp_path)
    assert db.intervals["assay"]["hole_id"].tolist() == ["A", "B"]
    assert db.survey["hole_id"].tolist() == ["A", "B"]
    assert db.metadata.reports["dropped_rows"]["assay"] == {"rows_without_hole_id": 4}
    assert not [n for n in db.metadata.notes if n["kind"] == "hole_id_carried_down"]
    code = main(["import", "--collar", str(tmp_path / "collar.csv"), "--survey",
                 str(tmp_path / "survey.csv"), "--assay", str(tmp_path / "assay.csv"),
                 "--auto", "--dry-run", "--source-option", "assay.csv:carry-down=no"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0 and "assay intervals 2" in out and "survey stations 4" in out

    # a collar row without a hole id is nobody's collar, whatever the row above says
    (tmp_path / "collar.csv").write_text(
        "Hole,East,North,RL,Depth\nA,0,0,100,10\n,50,0,100,10\n", encoding="utf-8"
    )
    (tmp_path / "survey.csv").write_text("Hole,Depth,Dip,Azimuth\nA,0,-60,90\n")
    db = run_recipe(
        make_recipe({"collar": [{"file": "collar.csv"}], "survey": [{"file": "survey.csv"}]}),
        tmp_path,
    )
    assert db.collar["hole_id"].tolist() == ["A"]
    assert db.metadata.reports["dropped_rows"]["collar"] == {"rows_without_hole_id": 1}


def test_carry_down_stays_on_its_sheet_and_skips_totals(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    book = openpyxl.Workbook()
    first = book.active
    first.title = "one"
    for row in (["Hole", "From", "To", "Au"], ["A", 0, 1, 0.1], [None, 1, 2, 0.2],
                ["Total", 0, 2, 0.3]):  # fmt: skip
        first.append(row)
    second = book.create_sheet("two")
    for row in (["Hole", "From", "To", "Au"], [None, 0, 1, 0.9], ["B", 1, 2, 0.4],
                [None, 2, 3, 0.5]):  # fmt: skip
        second.append(row)
    book.save(tmp_path / "assay.xlsx")
    (tmp_path / "collar.csv").write_text(
        "Hole,East,North,RL,Depth,Azimuth,Dip\nA,0,0,100,10,0,-90\nB,50,0,100,10,0,-90\n",
        encoding="utf-8",
    )
    recipe = make_recipe(
        {"collar": [{"file": "collar.csv"}],
         "assay": [{"file": "assay.xlsx", "sheets": "*", "qc": {"mode": "off"}}]}  # fmt: skip
    )
    db = run_recipe(recipe, tmp_path)
    assay = db.intervals["assay"]
    # the first row of sheet "two" has no row above it on its sheet: dropped, not hole A
    assert list(zip(assay["hole_id"], assay["depth_from"], strict=True)) == [
        ("A", 0.0), ("A", 1.0), ("B", 1.0), ("B", 2.0),
    ]  # fmt: skip
    assert db.metadata.reports["dropped_rows"]["assay"] == {
        "total_rows": 1, "rows_without_hole_id": 1,
    }  # fmt: skip
