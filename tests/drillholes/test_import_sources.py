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


# --------------------------------------------------------------------------------------
# files of one table that differ: dip convention, length unit, axis order, CRS
# --------------------------------------------------------------------------------------
def check_dips(db, collar):
    assert db.collar["hole_id"].tolist() == HOLES
    assert db.dip_convention == "negative_down"
    assert db.metadata.source_dip_convention == "mixed"
    np.testing.assert_array_equal(db.survey["dip"], collar["dip"])  # all negative down
    assert (db.survey["dip"] < 0).all()
    np.testing.assert_array_equal(db.survey["azimuth"], collar["azimuth"])
    # holes 4 and 5 are in both files: after the negation they agree
    merged = db.metadata.reports["conflicts"]["collar"]
    assert (merged["merged"], merged["conflicts"]) == (2, 0)


def test_dip_conventions_per_file_merge_into_negative_down(two_files, capsys):
    folder, collar = two_files
    sources = [{"file": "drilling.csv"},
               {"file": "gis.xlsx", "sheets": "Summary",
                "options": {"dip_convention": "positive_down"}}]  # fmt: skip
    db = run_recipe(make_recipe({"collar": sources}), folder)
    check_dips(db, collar)
    assert db.metadata.recipe["tables"]["collar"][1]["options"] == {
        "dip_convention": "positive_down"
    }
    assert db.metadata.reports["sources"]["collar"] == [
        {"file": "drilling.csv", "dip_convention": "negative_down"},
        {"file": "gis.xlsx", "dip_convention": "positive_down"},
    ]
    negated = [n for n in db.metadata.notes if n["kind"] == "dip_negated_to_negative_down"]
    assert [(n["source"], n["count"]) for n in negated] == [("gis.xlsx", 5)]
    # without the per-file option the two files contradict each other
    plain = run_recipe(
        make_recipe({"collar": [sources[0], {"file": "gis.xlsx", "sheets": "Summary"}]}), folder
    )
    assert plain.metadata.reports["conflicts"]["collar"]["conflicts"] == 2
    assert (plain.survey["dip"] > 0).sum() == 3

    # the CLI: said for the file, or decided per file by --dip-convention auto
    for how in (["--source-option", "gis.xlsx:dip-convention=positive_down"],
                ["--dip-convention", "auto"]):  # fmt: skip
        out = folder / f"db{len(how)}"
        code, printed, err = run_cli(
            capsys, "import", "--collar", folder / "drilling.csv", "--collar",
            folder / "gis.xlsx", "--auto", "--source-option", "gis.xlsx:sheet=Summary", *how,
            "--out", out,
        )  # fmt: skip
        assert code == 0, err
        check_dips(load_database(out), collar)
        assert "collar gis.xlsx: dip convention positive_down" in printed
        assert "dip convention negative_down" in printed
    # a recipe saved by the CLI replays it
    again = folder / "again"
    assert main(["import", "--recipe", str(out / "import_recipe.json"), "--out", str(again)]) == 0
    check_dips(load_database(again), collar)


def test_files_that_share_a_convention_are_left_alone(two_files):
    folder, collar = two_files
    first = collar.iloc[:5].assign(dip=lambda t: -t["dip"])
    first.to_csv(folder / "positive.csv", index=False)
    recipe = make_recipe(
        {"collar": [{"file": "positive.csv"}, {"file": "gis.xlsx", "sheets": "Summary"}]},
        {"dip_convention": "auto"},
    )
    db = run_recipe(recipe, folder)
    assert db.dip_convention == db.metadata.source_dip_convention == "positive_down"
    np.testing.assert_array_equal(db.survey["dip"], -collar["dip"])


def test_length_unit_per_file(two_files):
    folder, collar = two_files
    feet = collar.iloc[3:].copy()
    feet[["x", "y", "z", "max_depth"]] /= 0.3048
    feet.to_csv(folder / "feet.csv", index=False)
    rng = np.random.default_rng(2)
    assay = pd.DataFrame(
        [{"hole_id": h, "depth_from": float(k), "depth_to": float(k + 2), "au": rng.random()}
         for h in HOLES for k in range(0, 20, 2)]  # fmt: skip
    )
    in_feet = assay[assay["hole_id"].isin(HOLES[4:])].copy()
    in_feet[["depth_from", "depth_to"]] /= 0.3048
    assay[assay["hole_id"].isin(HOLES[:4])].to_csv(folder / "assay_m.csv", index=False)
    in_feet.to_csv(folder / "assay_ft.csv", index=False)
    recipe = make_recipe(
        {"collar": [{"file": "drilling.csv"},
                    {"file": "feet.csv", "options": {"length_unit": "ft"}}],
         "assay": [{"file": "assay_m.csv"},
                   {"file": "assay_ft.csv", "options": {"length_unit": "ft"}}]}  # fmt: skip
    )
    db = run_recipe(recipe, folder)
    assert db.length_unit == "m" and db.metadata.source_length_unit == "mixed"
    np.testing.assert_allclose(
        db.collar[["x", "y", "z", "max_depth"]], collar[["x", "y", "z", "max_depth"]], rtol=1e-12
    )
    assert db.metadata.reports["conflicts"]["collar"]["conflicts"] == 0
    np.testing.assert_allclose(
        db.intervals["assay"][["depth_from", "depth_to"]], assay[["depth_from", "depth_to"]],
        atol=1e-9,
    )  # fmt: skip
    converted = [n for n in db.metadata.notes if n["kind"] == "converted_to_metres"]
    assert {(n["table"], n["source"], n["unit"]) for n in converted} == {
        ("collar", "feet.csv", "ft"), ("assay", "assay_ft.csv", "ft"),
    }  # fmt: skip


def test_axis_order_per_file(two_files, capsys):
    folder, collar = two_files
    # a report table: X is the northing
    report = collar.iloc[3:].rename(columns={"x": "Y", "y": "X", "hole_id": "Hole"})
    report.to_csv(folder / "report.csv", index=False)
    for options, own in (({}, {"coordinate_order": "yxz"}), ({"coordinate_order": "auto"}, {})):
        recipe = make_recipe(
            {"collar": [{"file": "drilling.csv"}, {"file": "report.csv", "options": own}]}, options
        )
        db = run_recipe(recipe, folder)
        np.testing.assert_allclose(db.collar[["x", "y"]], collar[["x", "y"]])
        assert db.metadata.reports["conflicts"]["collar"]["conflicts"] == 0
        assert db.metadata.coordinate_order == "mixed"
        orders = [e["coordinate_order"] for e in db.metadata.reports["sources"]["collar"]]
        assert orders == ["xyz", "yxz"]
    code, printed, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--collar", folder / "report.csv",
        "--auto", "--source-option", "report.csv:coordinate-order=yxz", "--dry-run",
    )  # fmt: skip
    assert code == 0, err
    assert "collar report.csv: coordinate order yxz" in printed
    assert "coordinate order mixed" in printed


def test_crs_per_file_is_reprojected_into_the_database_crs(tmp_path):
    from test_import_collars import kruger_utm

    # holes either side of the 18 E meridian: zone 33 on the left, zone 34 on the right
    lon = np.array([17.90, 17.95, 17.99, 18.01, 18.05, 18.10])
    lat = np.array([45.00, 45.02, 45.04, 45.01, 45.03, 45.05])
    names = [f"SYN-{i:02d}" for i in range(1, 7)]
    in_33 = np.array([kruger_utm(a, b, 33) for a, b in zip(lon, lat, strict=True)])
    in_34 = np.array([kruger_utm(a, b, 34) for a, b in zip(lon, lat, strict=True)])
    west = pd.DataFrame({"Hole": names[:3], "East": in_33[:3, 0], "North": in_33[:3, 1]})
    east = pd.DataFrame({"Hole": names[2:], "East": in_34[2:, 0], "North": in_34[2:, 1]})
    for frame, file in ((west, "west.csv"), (east, "east.csv")):
        frame.assign(RL=100.0, Depth=50.0, Azimuth=0.0, Dip=-90.0).to_csv(
            tmp_path / file, index=False
        )
    recipe = make_recipe(
        {"collar": [{"file": "west.csv"},
                    {"file": "east.csv", "options": {"crs": "EPSG:32634"}}]},  # fmt: skip
        {"crs": "EPSG:32633"},
    )
    db = run_recipe(recipe, tmp_path)
    assert db.collar["hole_id"].tolist() == names
    # an independent series (Krueger) gives the zone 33 coordinates of every hole
    np.testing.assert_allclose(db.collar[["x", "y"]], in_33, atol=0.01)
    assert abs(in_34[3, 0] - in_33[3, 0]) > 100_000  # the two zones differ by far more
    merged = db.metadata.reports["conflicts"]["collar"]
    assert (merged["merged"], merged["conflicts"]) == (1, 0)  # SYN-03 is in both files
    moved = [n for n in db.metadata.notes if n["kind"] == "reprojected"]
    assert [(n["source"], n["from"], n["to"], n["count"]) for n in moved] == [
        ("east.csv", "EPSG:32634", "EPSG:32633", 4)
    ]
    # without a recipe-wide CRS the first CRS named by a collar file is the database's
    recipe = make_recipe(
        {"collar": [{"file": "east.csv", "options": {"crs": "EPSG:32634"}},
                    {"file": "west.csv", "options": {"crs": "EPSG:32633"}}]}  # fmt: skip
    )
    db = run_recipe(recipe, tmp_path)
    np.testing.assert_allclose(
        db.collar.set_index("hole_id").loc[names, ["x", "y"]], in_34, atol=0.01
    )
    feet = make_recipe(
        {"collar": [{"file": "west.csv"}, {"file": "east.csv", "options": {"crs": "EPSG:32634"}}]},
        {"crs": "EPSG:32633", "length_unit": "ft"},
    )
    with pytest.raises(ValueError, match="cannot be reprojected"):
        run_recipe(feet, tmp_path)


def test_source_options_are_checked(two_files, capsys):
    folder, _ = two_files
    with pytest.raises(ValueError, match="cannot set \\['delimiter'\\] for itself"):
        make_recipe({"collar": [{"file": "a.csv", "options": {"delimiter": ";"}}]})
    with pytest.raises(ValueError, match="source options: dip_convention must be one of"):
        make_recipe({"collar": [{"file": "a.csv", "options": {"dip_convention": "down"}}]})
    with pytest.raises(ValueError, match="'options' must be an object"):
        make_recipe({"collar": [{"file": "a.csv", "options": "ft"}]})
    # nothing set: nothing stored (recipes written before stay as they were)
    empty = make_recipe({"collar": [{"file": "a.csv", "options": {"crs": ""}}]})
    assert empty["tables"]["collar"] == [{"file": "a.csv"}]
    code, _, err = run_cli(
        capsys, "import", "--collar", folder / "drilling.csv", "--auto", "--source-option",
        "drilling.csv:length-unit=yards", "--dry-run",
    )  # fmt: skip
    assert code == 2 and "length_unit must be one of" in err


def test_axis_order_per_file_when_only_one_file_has_longitude_and_latitude(tmp_path):
    from test_import_collars import kruger_utm

    lon = np.array([15.10, 15.12, 15.14, 15.16])
    lat = np.array([47.00, 47.01, 47.02, 47.03])
    names = [f"SYN-{i:02d}" for i in range(1, 5)]
    utm = np.array([kruger_utm(a, b, 33) for a, b in zip(lon, lat, strict=True)]).round(2)
    # a report table (X = northing), a plain table, and a table in degrees
    pd.DataFrame({"Hole": names[:2], "X": utm[:2, 1], "Y": utm[:2, 0], "RL": 300.0}).to_csv(
        tmp_path / "report.csv", index=False
    )
    pd.DataFrame({"Hole": names[1:3], "East": utm[1:3, 0], "North": utm[1:3, 1],
                  "Azimuth": 0.0, "Dip": -90.0, "Depth": 40.0}).to_csv(
        tmp_path / "plain.csv", index=False
    )  # fmt: skip
    pd.DataFrame({"Hole": names, "Longitude": lon, "Latitude": lat, "RL": 300.0,
                  "Azimuth": 0.0, "Dip": -90.0, "Depth": 40.0}).to_csv(
        tmp_path / "degrees.csv", index=False
    )  # fmt: skip
    recipe = make_recipe(
        {"collar": [{"file": "report.csv"}, {"file": "plain.csv"}, {"file": "degrees.csv"}]},
        {"coordinate_order": "auto"},
    )
    db = run_recipe(recipe, tmp_path)
    assert db.collar["hole_id"].tolist() == names
    np.testing.assert_allclose(db.collar[["x", "y"]], utm, atol=0.02)  # the last from degrees
    orders = [e["coordinate_order"] for e in db.metadata.reports["sources"]["collar"]]
    assert orders == ["yxz", "xyz", "xyz"]
    assert db.metadata.crs == "EPSG:32633"
    assert db.metadata.reports["conflicts"]["collar"]["conflicts"] == 0
