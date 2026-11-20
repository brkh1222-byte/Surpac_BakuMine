"""Options of one source file of a table: a table made of a CSV and a workbook, files
with different dip conventions, units, axis orders and coordinate systems."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import load_recipe, make_recipe, run_recipe

N = 8
HOLES = [f"SYN-{i:02d}" for i in range(1, N + 1)]


def truth() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    return pd.DataFrame(
        {
            "hole_id": HOLES,
            "x": 512000 + rng.uniform(0, 800, N).round(2),
            "y": 5310000 + rng.uniform(0, 800, N).round(2),
            "z": 1200 + rng.uniform(0, 40, N).round(2),
            "max_depth": rng.integers(100, 300, N).astype(float),
            "azimuth": rng.choice([30.0, 120.0, 210.0], N),
            "dip": -rng.choice([60.0, 70.0, 85.0], N),  # negative down
        }
    )


@pytest.fixture
def two_files(tmp_path):
    """The collars of the first five holes as a CSV (dips negative down); those of the
    last five in the second sheet of a workbook, under a title row, dips positive down.
    Holes 4 and 5 are in both."""
    collar = truth()
    first = collar.iloc[:5].rename(
        columns={"hole_id": "Hole", "x": "East", "y": "North", "z": "RL", "max_depth": "Depth",
                 "azimuth": "Azimuth", "dip": "Dip"}  # fmt: skip
    )
    first.to_csv(tmp_path / "drilling.csv", index=False)
    second = collar.iloc[3:].assign(dip=lambda t: -t["dip"]).rename(
        columns={"hole_id": "BHID", "x": "Easting", "y": "Northing", "z": "Elevation",
                 "max_depth": "EOH", "azimuth": "Azi", "dip": "Inclination"}  # fmt: skip
    )
    with pd.ExcelWriter(tmp_path / "gis.xlsx") as writer:
        pd.DataFrame({"note": ["not a table"]}).to_excel(writer, sheet_name="Readme", index=False)
        second.to_excel(writer, sheet_name="Summary", index=False, startrow=2)
        sheet = writer.sheets["Summary"]
        sheet["A1"] = "Collars of the 2025 programme"
    return tmp_path, collar


def run_cli(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_a_table_mixes_a_csv_and_a_workbook_sheet(two_files, capsys):
    folder, collar = two_files
    out = folder / "db"
    # a table-wide --sheet passes the CSV by; the header row is the workbook's own
    code, printed, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "gis.xlsx",
        "--auto", "--sheet", "collar=Summary", "--source-option", "gis.xlsx:header-row=3",
        "--dry-run",
    )  # fmt: skip
    assert code == 0, err
    assert "holes 8," in printed
    # the same said for the file only
    code, printed, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "gis.xlsx",
        "--auto", "--source-option", f"{folder / 'gis.xlsx'}:sheet=Summary", "--out", out,
    )  # fmt: skip
    assert code == 0, err
    assert "collar (auto, drilling.csv): Hole->hole_id" in printed
    assert "collar (auto, gis.xlsx): BHID->hole_id" in printed
    assert "collar: 2 file(s), 2 sheet(s) stacked" in printed
    db = load_database(out)
    assert db.collar["hole_id"].tolist() == HOLES
    np.testing.assert_allclose(db.collar[["x", "y", "z"]], collar[["x", "y", "z"]])
    recipe = load_recipe(out / "import_recipe.json")
    assert "sheets" not in recipe["tables"]["collar"][0]
    assert recipe["tables"]["collar"][1]["sheets"] == ["Summary"]


def test_source_option_takes_a_column_map_and_layout_per_file(two_files, capsys):
    folder, collar = two_files
    own = folder / "gis_map.json"
    own.write_text(json.dumps({"BHID": "hole_id", "Easting": "x", "Northing": "y",
                               "Elevation": "z"}), encoding="utf-8")  # fmt: skip
    code, printed, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "gis.xlsx",
        "--auto", "--source-option", "gis.xlsx:sheets=1", "--source-option",
        "gis.xlsx:header-row=3", "--source-option", "gis.xlsx:header-rows=1",
        "--source-option", f"gis.xlsx:map={own}", "--save-recipe", folder / "r.json", "--dry-run",
    )  # fmt: skip
    assert code == 0, err
    source = load_recipe(folder / "r.json")["tables"]["collar"][1]
    assert (source["sheets"], source["header_row"], source["header_rows"]) == ([1], 3, 1)
    assert source["map"] == {"BHID": "hole_id", "Easting": "x", "Northing": "y", "Elevation": "z"}
    assert "collar (auto, gis.xlsx): BHID->hole_id, Easting->x, Northing->y, Elev" in printed


@pytest.mark.parametrize(
    ("option", "message"),
    [
        ("gis.xlsx:sheet", "expects FILE:KEY=VALUE"),
        ("gis.xlsx:colour=red", "expects FILE:KEY=VALUE"),
        ("other.xlsx:sheet=A", "is not one of the source files"),
        ("gis.xlsx:header-row=two", "expects a number"),
        ("gis.xlsx:hole-from-sheet=maybe", "expects yes or no"),
        ("gis.xlsx:sheet=Nope", "no sheet matches 'Nope'"),
        ("gis.xlsx:map=missing.json", "column map file not found"),
    ],
)
def test_source_option_errors(two_files, capsys, option, message):
    folder, _ = two_files
    code, _, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "gis.xlsx",
        "--auto", "--source-option", option, "--dry-run",
    )  # fmt: skip
    assert code == 2 and message in err
    code, _, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "gis.xlsx",
        "--auto", "--source-option", "gis.xlsx:sheet=Summary", "--source-option",
        "gis.xlsx:sheets=Readme", "--dry-run",
    )  # fmt: skip
    assert code == 2 and "twice" in err


def test_sheets_of_a_recipe_source_are_ignored_for_a_csv(two_files):
    folder, _ = two_files
    recipe = make_recipe(
        {"collar": [{"file": "drilling.csv", "sheets": ["Summary"]},
                    {"file": "gis.xlsx", "sheets": ["Summary"]}]}  # fmt: skip
    )
    assert run_recipe(recipe, folder).collar["hole_id"].tolist() == HOLES
