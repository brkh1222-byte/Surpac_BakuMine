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
