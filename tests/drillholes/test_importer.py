"""P1.T1 acceptance tests: table import, column mapping, options, storage."""

import json

import numpy as np
import pandas as pd
import pytest
from drillhole_samples import (
    BABBITT_MAPS,
    BABBITT_OPTIONS,
    BABBITT_RAW,
    REPO,
    SMALL_MAPS,
    import_small,
)
from pandas.testing import assert_frame_equal

from bakumine.core.drillholes.importer import (
    COLLAR_COLUMNS,
    DrillholeDatabase,
    ImportOptions,
    TableSource,
    import_drillholes,
    import_table,
    load_database,
    read_raw_table,
    suggest_column_map,
)

FT = 0.3048


def assert_db_equal(a: DrillholeDatabase, b: DrillholeDatabase, *, metadata: bool = True) -> None:
    assert_frame_equal(a.collar, b.collar)
    assert_frame_equal(a.survey, b.survey)
    assert list(a.intervals) == list(b.intervals)
    for name in a.intervals:
        assert_frame_equal(a.intervals[name], b.intervals[name])
    if metadata:
        assert a.metadata == b.metadata


# --------------------------------------------------------------------------------------
# Babbitt: a real dataset imported from its raw files
# --------------------------------------------------------------------------------------
def test_babbitt_imports_from_raw_files(babbitt_db):
    db = babbitt_db
    assert len(db.collar) == 399
    assert len(db.survey) == 2628
    assert len(db.intervals["assay"]) == 35616
    assert list(db.collar.columns) == list(COLLAR_COLUMNS)
    assert list(db.survey.columns) == ["hole_id", "depth", "dip", "azimuth"]
    assert list(db.intervals["assay"].columns) == [
        "hole_id", "depth_from", "depth_to", "cu", "ni", "s", "fe",
    ]  # fmt: skip
    # Units and dip sign are recorded as given, not changed.
    assert db.length_unit == "ftUS"
    assert db.dip_convention == "positive_down"
    assert db.metadata.column_maps == BABBITT_MAPS
    assert db.survey["dip"].between(32, 90).all()


def test_babbitt_hole_ids_are_strings(babbitt_db):
    for table in babbitt_db.tables.values():
        assert table["hole_id"].dtype == "str"
        assert all(isinstance(v, str) for v in table["hole_id"])
    assert "34873" in set(babbitt_db.collar["hole_id"])  # numeric-looking ID kept as text
    assert babbitt_db.collar["hole_id"].is_unique


def test_babbitt_numeric_columns_are_float64_and_blank_is_nan(babbitt_db):
    assay = babbitt_db.intervals["assay"]
    for column in ("depth_from", "depth_to", "cu", "ni", "s", "fe"):
        assert assay[column].dtype == np.float64
    # Counts of assayed values from SOURCE.md: blanks are NaN, not zero.
    assert assay["cu"].notna().sum() == 23685
    assert assay["ni"].notna().sum() == 23439
    assert assay["s"].notna().sum() == 23545
    assert assay["fe"].notna().sum() == 24
    first = assay.iloc[0]  # hole 34873, 0 to 2515 ft, unassayed
    assert (first["hole_id"], first["depth_from"], first["depth_to"]) == ("34873", 0.0, 2515.0)
    assert np.isnan(first["cu"])


def test_babbitt_matches_independent_pandas_read(babbitt_db):
    """Same numbers as a plain pandas read of the raw files."""
    collar = pd.read_csv(BABBITT_RAW / "collar_BABBITT.csv", dtype={"BHID": str})
    survey = pd.read_csv(BABBITT_RAW / "survey_BABBITT.csv", dtype={"BHID": str})
    assay = pd.read_csv(BABBITT_RAW / "assay_BABBITT.csv", dtype={"BHID": str})
    np.testing.assert_array_equal(babbitt_db.collar["x"], collar["XCOLLAR"])
    np.testing.assert_array_equal(babbitt_db.collar["y"], collar["YCOLLAR"])
    np.testing.assert_array_equal(babbitt_db.collar["z"], collar["ZCOLLAR"])
    np.testing.assert_array_equal(babbitt_db.survey["azimuth"], survey["AZ"])
    np.testing.assert_array_equal(babbitt_db.survey["dip"], survey["DIP"])
    np.testing.assert_array_equal(babbitt_db.intervals["assay"]["cu"], assay["CU"])
    # max_depth is absent from the raw collar: derived from the deepest assay interval.
    deepest = assay.groupby("BHID")["TO"].max()
    np.testing.assert_array_equal(
        babbitt_db.collar["max_depth"], collar["BHID"].map(deepest).to_numpy()
    )
    assert babbitt_db.collar["max_depth"].sum() == pytest.approx(541230.0)
    assert {
        "table": "collar",
        "column": "max_depth",
        "kind": "derived_from_deepest_interval",
        "count": 399,
    } in babbitt_db.metadata.notes


def test_babbitt_options_convert_units_and_dip():
    from drillhole_samples import import_babbitt

    raw = import_babbitt()
    db = import_babbitt(
        ImportOptions(
            length_unit="ftUS",
            dip_convention="positive_down",
            convert_to_metres=True,
            normalise_dip=True,
        )
    )
    k = 1200.0 / 3937.0
    assert db.length_unit == "m" and db.metadata.source_length_unit == "ftUS"
    assert db.dip_convention == "negative_down"
    assert db.metadata.source_dip_convention == "positive_down"
    np.testing.assert_allclose(db.collar[["x", "y", "z", "max_depth"]],
                               raw.collar[["x", "y", "z", "max_depth"]] * k, rtol=1e-15)
    np.testing.assert_allclose(db.survey["depth"], raw.survey["depth"] * k, rtol=1e-15)
    np.testing.assert_array_equal(db.survey["dip"], -raw.survey["dip"])
    np.testing.assert_array_equal(db.survey["azimuth"], raw.survey["azimuth"])
    np.testing.assert_array_equal(db.intervals["assay"]["cu"], raw.intervals["assay"]["cu"])


def test_babbitt_database_feeds_desurvey(babbitt_db):
    traces = babbitt_db.desurvey()
    assert len(traces) == 399
    # Every hole goes down (dips are positive-down and recorded as such).
    bottom, _ = traces.xyz_at("B1-001", 520.0)
    top, _ = traces.xyz_at("B1-001", 0.0)
    np.testing.assert_array_equal(top, [2294148.2, 420495.9, 1620.9])
    # 520 ft at dip 60 down: 520 sin(60) below the collar.
    assert top[2] - bottom[2] == pytest.approx(520.0 * np.sin(np.radians(60.0)))


def test_suggester_gets_babbitt_raw_headers_right():
    for file, kind, name in (
        ("collar_BABBITT.csv", "collar", "collar"),
        ("survey_BABBITT.csv", "survey", "survey"),
        ("assay_BABBITT.csv", "interval", "assay"),
    ):
        headers = list(read_raw_table(BABBITT_RAW / file).columns)
        suggestion = suggest_column_map(headers, kind)
        assert suggestion.mapping == BABBITT_MAPS[name]
        assert suggestion.missing == []
        assert suggestion.unmatched == []


def test_suggested_maps_import_babbitt(babbitt_db):
    maps = {
        name: suggest_column_map(
            read_raw_table(BABBITT_RAW / f"{name}_BABBITT.csv").columns, kind
        ).mapping
        for name, kind in (("collar", "collar"), ("survey", "survey"), ("assay", "interval"))
    }
    db = import_drillholes(
        TableSource(BABBITT_RAW / "collar_BABBITT.csv", maps["collar"]),
        TableSource(BABBITT_RAW / "survey_BABBITT.csv", maps["survey"]),
        {"assay": TableSource(BABBITT_RAW / "assay_BABBITT.csv", maps["assay"])},
        options=BABBITT_OPTIONS,
    )
    assert_db_equal(db, babbitt_db)


# --------------------------------------------------------------------------------------
# column-map suggester
# --------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("kind", "headers", "expected"),
    [
        (
            "collar",
            ["HOLEID", "NORTH", "EAST", "RL", "TOTAL_DEPTH"],
            {"HOLEID": "hole_id", "NORTH": "y", "EAST": "x", "RL": "z",
             "TOTAL_DEPTH": "max_depth"},
        ),
        (
            "collar",
            ["Hole_ID", "Easting (m)", "Northing (m)", "Elevation", "EOH", "Comments"],
            {"Hole_ID": "hole_id", "Easting (m)": "x", "Northing (m)": "y", "Elevation": "z",
             "EOH": "max_depth"},
        ),
        (
            "collar",  # misspelt headers are matched fuzzily
            ["hole id", "Eastings", "Northng", "Elevaton", "Max Depth"],
            {"hole id": "hole_id", "Eastings": "x", "Northng": "y", "Elevaton": "z",
             "Max Depth": "max_depth"},
        ),
        (
            "survey",
            ["BHID", "DEPTH", "AZI", "DIP"],
            {"BHID": "hole_id", "DEPTH": "depth", "AZI": "azimuth", "DIP": "dip"},
        ),
        (
            "survey",
            ["Hole", "Survey Depth", "Bearing", "Inclination"],
            {"Hole": "hole_id", "Survey Depth": "depth", "Bearing": "azimuth",
             "Inclination": "dip"},
        ),
        (
            "interval",
            ["HOLE-ID", "Depth_From", "Depth_To", "Au ppm", "LITH"],
            {"HOLE-ID": "hole_id", "Depth_From": "depth_from", "Depth_To": "depth_to",
             "Au ppm": "au_ppm", "LITH": "lith"},
        ),
        (
            "interval",  # N and E are grades here, not northing/easting
            ["DHID", "FROM_M", "TO_M", "N", "E", "S"],
            {"DHID": "hole_id", "FROM_M": "depth_from", "TO_M": "depth_to", "N": "n", "E": "e",
             "S": "s"},
        ),
    ],
)  # fmt: skip
def test_suggester_common_headers(kind, headers, expected):
    assert suggest_column_map(headers, kind).mapping == expected


def test_suggester_reports_missing_and_unmatched_and_is_editable():
    suggestion = suggest_column_map(["Hole", "Foo", "North", "RL"], "collar")
    assert suggestion.mapping == {"Hole": "hole_id", "North": "y", "RL": "z"}
    assert suggestion.missing == ["x"]
    assert suggestion.unmatched == ["Foo"]
    suggestion.mapping["Foo"] = "x"  # the user edits the proposal
    table, _ = import_table(
        pd.DataFrame([["A", "1", "2", "3"]], columns=["Hole", "Foo", "North", "RL"]),
        "collar",
        suggestion.mapping,
    )
    assert table.loc[0, ["x", "y", "z"]].tolist() == [1.0, 2.0, 3.0]


# --------------------------------------------------------------------------------------
# small dataset: typing, axis order, units, missing values
# --------------------------------------------------------------------------------------
def test_small_dataset_import(small_db):
    db = small_db
    assert list(db.intervals) == ["assay", "geology"]
    assert db.collar["hole_id"].tolist() == ["DH001", "DH002", "DH003", "1004", "DH005", "DH006"]
    assert db.collar.loc[1, ["x", "y", "z", "max_depth"]].tolist() == [1050.0, 5000.0, 201.0, 100]
    assert db.survey["dip"].dtype == np.float64
    assay, geology = db.intervals["assay"], db.intervals["geology"]
    assert assay["au"].dtype == np.float64 and assay["cu"].dtype == np.float64
    assert assay["au"].isna().sum() == 1  # the one blank assay
    assert (assay["au"].dropna() > 0).all()
    assert geology["rock"].dtype == "str"  # text attribute kept as text
    assert geology["rock"].unique().tolist() == ["OVB", "ORE", "WST"]
    assert geology["depth_to"].dtype == np.float64
    assert db.attribute_columns("assay") == ["au", "cu"]
    assert db.metadata.column_maps == SMALL_MAPS


def test_csv_files_give_the_same_result_as_frames(small_tables, small_db, tmp_path):
    for name, table in small_tables.items():
        table.to_csv(tmp_path / f"{name}.csv", index=False)
    db = import_drillholes(
        TableSource(tmp_path / "collar.csv", SMALL_MAPS["collar"]),
        TableSource(tmp_path / "survey.csv", SMALL_MAPS["survey"]),
        {n: TableSource(tmp_path / f"{n}.csv", SMALL_MAPS[n]) for n in ("assay", "geology")},
    )
    assert_db_equal(db, small_db, metadata=False)
    assert db.metadata.sources["collar"] == str(tmp_path / "collar.csv")


def test_yx_swapped_file_imports_to_the_same_result(small_tables, small_db, tmp_path):
    """Surpac-style Y, X, Z column order, with and without a header row."""
    collar = small_tables["collar"]
    # Header-less file in Surpac order: hole, northing, easting, elevation, depth.
    swapped = collar[["HOLE", "NORTH", "EAST", "RL", "EOH"]]
    swapped.to_csv(tmp_path / "collar_yxz.csv", index=False, header=False)
    positional = {0: "hole_id", 1: "x", 2: "y", 3: "z", 4: "max_depth"}
    others = {n: TableSource(small_tables[n], SMALL_MAPS[n]) for n in ("assay", "geology")}
    survey = TableSource(small_tables["survey"], SMALL_MAPS["survey"])

    db = import_drillholes(
        TableSource(tmp_path / "collar_yxz.csv", positional, header=False),
        survey,
        others,
        options=ImportOptions(coordinate_order="yxz"),
    )
    assert_db_equal(db, small_db, metadata=False)
    assert db.metadata.coordinate_order == "yxz"

    # Without the option the same file comes in with east and north exchanged.
    wrong = import_drillholes(
        TableSource(tmp_path / "collar_yxz.csv", positional, header=False), survey, others
    )
    np.testing.assert_array_equal(wrong.collar["x"], small_db.collar["y"])
    np.testing.assert_array_equal(wrong.collar["y"], small_db.collar["x"])

    # Headed file whose "X" column holds the northing: same option, same result.
    headed = swapped.set_axis(["HOLE", "X", "Y", "Z", "EOH"], axis=1)
    db = import_drillholes(
        TableSource(headed, {"HOLE": "hole_id", "X": "x", "Y": "y", "Z": "z",
                             "EOH": "max_depth"}),
        survey,
        others,
        options=ImportOptions(coordinate_order="yxz"),
    )  # fmt: skip
    assert_db_equal(db, small_db, metadata=False)


def test_feet_to_metres_is_applied_to_coordinates_and_depths(small_tables, small_db):
    in_feet = import_small(small_tables, ImportOptions(length_unit="ft"))
    assert in_feet.length_unit == "ft"
    assert_db_equal(in_feet, small_db, metadata=False)  # recorded, not converted

    db = import_small(small_tables, ImportOptions(length_unit="ft", convert_to_metres=True))
    assert db.length_unit == "m"
    assert db.metadata.source_length_unit == "ft"
    for column in ("x", "y", "z", "max_depth"):
        np.testing.assert_allclose(db.collar[column], small_db.collar[column] * FT, rtol=1e-15)
    np.testing.assert_allclose(db.survey["depth"], small_db.survey["depth"] * FT, rtol=1e-15)
    for name in ("assay", "geology"):
        for column in ("depth_from", "depth_to"):
            np.testing.assert_allclose(
                db.intervals[name][column], small_db.intervals[name][column] * FT, rtol=1e-15
            )
    # A 100 ft hole is 30.48 m; angles and grades are not lengths.
    assert db.collar["max_depth"].iloc[0] == pytest.approx(30.48)
    np.testing.assert_array_equal(db.survey["dip"], small_db.survey["dip"])
    np.testing.assert_array_equal(db.survey["azimuth"], small_db.survey["azimuth"])
    grades = ["au", "cu"]
    assert_frame_equal(db.intervals["assay"][grades], small_db.intervals["assay"][grades])
    # The converted database still desurveys consistently: bottom is 30.48 m along the hole.
    trace = db.desurvey().hole("DH001")
    assert trace.max_depth == pytest.approx(30.48)


def test_dip_convention_recorded_and_normalised(small_tables, small_db):
    flipped = {**small_tables, "survey": small_tables["survey"].copy()}
    flipped["survey"]["DIP"] = [str(-float(v)) for v in flipped["survey"]["DIP"]]
    recorded = import_small(flipped, ImportOptions(dip_convention="positive_down"))
    assert recorded.dip_convention == "positive_down"
    assert (recorded.survey["dip"] > 0).all()
    normalised = import_small(
        flipped, ImportOptions(dip_convention="positive_down", normalise_dip=True)
    )
    assert normalised.dip_convention == "negative_down"
    assert normalised.metadata.source_dip_convention == "positive_down"
    assert_frame_equal(normalised.survey, small_db.survey)
    # Both describe the same holes in space.
    a, _ = recorded.desurvey().xyz_at("DH001", 100.0)
    b, _ = normalised.desurvey().xyz_at("DH001", 100.0)
    np.testing.assert_allclose(a, b, atol=1e-12)
    assert a[2] < 200.0  # below the collar


ASSAY_MAP = {"HOLE": "hole_id", "FROM": "depth_from", "TO": "depth_to", "AU": "au", "LITH": "lith"}


def _assay(cells: list[str], lith: list[str] | None = None) -> pd.DataFrame:
    n = len(cells)
    return pd.DataFrame(
        {
            "HOLE": ["A"] * n,
            "FROM": [str(i) for i in range(n)],
            "TO": [str(i + 1) for i in range(n)],
            "AU": cells,
            "LITH": lith or ["x"] * n,
        }
    )


def test_missing_tokens_become_nan_never_zero():
    cells = ["1.5", "", " ", "-", "NA", "n/a", "NaN", "null", "-99", "-999.0", "-9999", "0", "2"]
    table, notes = import_table(_assay(cells), "interval", ASSAY_MAP)
    au = table["au"]
    assert au.dtype == np.float64
    assert au.isna().tolist() == [False] + [True] * 10 + [False, False]
    assert au.dropna().tolist() == [1.5, 0.0, 2.0]  # a real 0 stays 0; no NaN became 0
    assert {"table": "interval", "column": "au", "kind": "sentinel_set_to_nan", "count": 3} in notes


def test_missing_tokens_are_user_configurable():
    cells = ["1.5", "-99", "NS", "-1", "", "-"]
    options = ImportOptions(missing_tokens=("NS", -1))
    table, _ = import_table(_assay(cells), "interval", ASSAY_MAP, options)
    # -99 is now a value, "NS" and -1 are missing, blank is always missing; "-" is no longer
    # a token, so the column is not numeric unless forced.
    assert table["au"].dtype == "str"
    options = ImportOptions(missing_tokens=("NS", -1), numeric_columns=("au",))
    table, notes = import_table(_assay(cells), "interval", ASSAY_MAP, options)
    np.testing.assert_array_equal(table["au"], [1.5, -99.0, np.nan, np.nan, np.nan, np.nan])
    unparseable = [n for n in notes if n["kind"] == "unparseable_set_to_nan"]
    assert unparseable == [
        {"table": "interval", "column": "au", "kind": "unparseable_set_to_nan", "count": 1,
         "examples": ["-"]}
    ]  # fmt: skip


def test_numeric_sentinels_do_not_touch_coordinates_or_angles():
    collar = pd.DataFrame([["A", "-99", "-999", "-9999", "NA"]],
                          columns=["HOLE", "EAST", "NORTH", "RL", "EOH"])  # fmt: skip
    table, _ = import_table(collar, "collar", SMALL_MAPS["collar"])
    assert table.loc[0, ["x", "y", "z"]].tolist() == [-99.0, -999.0, -9999.0]
    assert np.isnan(table.loc[0, "max_depth"])  # text tokens do apply


@pytest.mark.parametrize(
    ("rule", "expected"),
    [("nan", [np.nan, np.nan, np.nan]), ("half", [0.005, 0.25, 1.0]), ("limit", [0.01, 0.5, 2.0])],
)
def test_below_detection_rules(rule, expected):
    cells = ["<0.01", "< 0.5", "<=2", "0.3"]
    table, notes = import_table(
        _assay(cells), "interval", ASSAY_MAP, ImportOptions(below_detection=rule)
    )
    assert table["au"].dtype == np.float64
    np.testing.assert_array_equal(table["au"], [*expected, 0.3])
    assert {"table": "interval", "column": "au", "kind": "below_detection", "count": 3,
            "rule": rule} in notes  # fmt: skip


def test_text_attributes_stay_text_and_missing_text_is_missing():
    lith = ["GRANITE", " ore ", "", "NA", "12"]
    table, _ = import_table(_assay(["1"] * 5, lith), "interval", ASSAY_MAP)
    assert table["lith"].dtype == "str"
    assert table["lith"].tolist()[:2] == ["GRANITE", "ore"]
    assert table["lith"].isna().tolist() == [False, False, True, True, False]
    assert table["lith"].iloc[4] == "12"
    # A column of numeric rock codes is numeric unless the user says it is text.
    codes = _assay(["1"] * 3, ["10", "20", "030"])
    assert import_table(codes, "interval", ASSAY_MAP)[0]["lith"].tolist() == [10.0, 20.0, 30.0]
    as_text = import_table(codes, "interval", ASSAY_MAP, ImportOptions(text_columns=("lith",)))[0]
    assert as_text["lith"].tolist() == ["10", "20", "030"]


def test_hole_id_is_stripped_text_even_when_numeric():
    collar = pd.DataFrame(
        [[" 007 ", "1", "2", "3", "10"], ["34873", "1", "2", "3", "10"]],
        columns=["HOLE", "EAST", "NORTH", "RL", "EOH"],
    )
    table, _ = import_table(collar, "collar", SMALL_MAPS["collar"])
    assert table["hole_id"].tolist() == ["007", "34873"]
    assert table["hole_id"].dtype == "str"


def test_max_depth_derived_only_where_missing(small_tables):
    tables = dict(small_tables)
    tables["collar"] = tables["collar"].copy()
    tables["collar"].loc[0, "EOH"] = ""  # DH001: no depth in the collar
    tables["collar"].loc[1, "EOH"] = "250"  # DH002: collar depth wins over intervals
    db = import_small(tables)
    assert db.collar["max_depth"].tolist() == [100.0, 250.0, 100.0, 100.0, 100.0, 100.0]
    assert db.metadata.notes[-1]["count"] == 1
    off = import_small(tables, ImportOptions(derive_max_depth=False))
    assert np.isnan(off.collar["max_depth"].iloc[0])

    # No max_depth column at all.
    no_depth = {**tables, "collar": tables["collar"].drop(columns="EOH")}
    collar_map = {k: v for k, v in SMALL_MAPS["collar"].items() if k != "EOH"}
    db = import_drillholes(
        TableSource(no_depth["collar"], collar_map),
        TableSource(no_depth["survey"], SMALL_MAPS["survey"]),
        {"geology": TableSource(no_depth["geology"], SMALL_MAPS["geology"])},
    )
    assert db.collar["max_depth"].tolist() == [100.0] * 6


def test_unmapped_columns_are_kept_or_dropped(small_tables):
    collar = small_tables["collar"].assign(PROSPECT="North Pit", YEAR="2021")
    kept, _ = import_table(collar, "collar", SMALL_MAPS["collar"])
    assert list(kept.columns) == [*COLLAR_COLUMNS, "PROSPECT", "YEAR"]
    assert kept["PROSPECT"].dtype == "str" and kept["YEAR"].dtype == np.float64
    dropped, _ = import_table(
        collar, "collar", SMALL_MAPS["collar"], ImportOptions(keep_unmapped=False)
    )
    assert list(dropped.columns) == list(COLLAR_COLUMNS)


def test_column_map_errors(small_tables):
    collar = small_tables["collar"]
    with pytest.raises(ValueError, match="not found"):
        import_table(collar, "collar", {**SMALL_MAPS["collar"], "NOPE": "foo"})
    with pytest.raises(ValueError, match=r"no source for \['x'\]"):
        import_table(collar, "collar", {"HOLE": "hole_id", "NORTH": "y", "RL": "z"})
    with pytest.raises(ValueError, match="several source columns"):
        import_table(collar, "collar", {**SMALL_MAPS["collar"], "EOH": "z"})
    # Header matching is case-insensitive.
    lower = {k.lower(): v for k, v in SMALL_MAPS["collar"].items()}
    assert_frame_equal(
        import_table(collar, "collar", lower)[0],
        import_table(collar, "collar", SMALL_MAPS["collar"])[0],
    )


def test_option_validation():
    with pytest.raises(ValueError, match="length_unit"):
        ImportOptions(length_unit="yd")
    with pytest.raises(ValueError, match="dip_convention"):
        ImportOptions(dip_convention="up")
    with pytest.raises(ValueError, match="coordinate_order"):
        ImportOptions(coordinate_order="xy")
    with pytest.raises(ValueError, match="below_detection"):
        ImportOptions(below_detection="zero")
    with pytest.raises(ValueError, match="invalid interval table name"):
        import_small_named("../assay")


def import_small_named(name: str) -> DrillholeDatabase:
    from drillhole_samples import small_raw_tables

    tables = small_raw_tables()
    return import_drillholes(
        TableSource(tables["collar"], SMALL_MAPS["collar"]),
        TableSource(tables["survey"], SMALL_MAPS["survey"]),
        {name: TableSource(tables["assay"], SMALL_MAPS["assay"])},
    )


# --------------------------------------------------------------------------------------
# XLSX
# --------------------------------------------------------------------------------------
def test_xlsx_round_trip(small_tables, small_db, tmp_path):
    """Tables written to one workbook (typed cells, blanks) import to the same database."""
    path = tmp_path / "holes.xlsx"
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for name, table in small_tables.items():
            typed = table.copy()
            for column in typed.columns:
                if column not in ("HOLE", "ROCK"):
                    typed[column] = pd.to_numeric(typed[column])  # real numeric cells
            if name == "collar":
                # A numeric hole ID stored as a number cell must still come back as "1004".
                typed["HOLE"] = [int(h) if h.isdigit() else h for h in typed["HOLE"]]
            typed.to_excel(writer, sheet_name=name, index=False)
    db = import_drillholes(
        TableSource(path, SMALL_MAPS["collar"], sheet="collar"),
        TableSource(path, SMALL_MAPS["survey"], sheet="survey"),
        {n: TableSource(path, SMALL_MAPS[n], sheet=n) for n in ("assay", "geology")},
    )
    assert_db_equal(db, small_db, metadata=False)
    assert db.collar["hole_id"].iloc[3] == "1004"
    assert db.intervals["assay"]["au"].isna().sum() == 1


def test_xlsx_text_cells_and_default_sheet(tmp_path):
    path = tmp_path / "assay.xlsx"
    _assay(["0.5", "<0.01", "NA", "-99"]).to_excel(path, index=False)  # all text cells
    table, _ = import_table(path, "interval", ASSAY_MAP)
    np.testing.assert_array_equal(table["au"], [0.5, 0.005, np.nan, np.nan])
    assert table["hole_id"].tolist() == ["A"] * 4


# --------------------------------------------------------------------------------------
# storage
# --------------------------------------------------------------------------------------
def test_parquet_save_load_is_lossless(small_tables, tmp_path):
    tables = dict(small_tables)
    tables["geology"] = tables["geology"].copy()
    tables["geology"].loc[2, "ROCK"] = ""  # a missing text value must survive
    db = import_small(tables, ImportOptions(length_unit="ft", dip_convention="negative_down"))
    folder = db.save(tmp_path / "db")
    assert sorted(p.name for p in folder.iterdir()) == [
        "collar.parquet",
        "database.json",
        "interval_assay.parquet",
        "interval_geology.parquet",
        "survey.parquet",
    ]
    loaded = load_database(folder)
    assert_db_equal(loaded, db)
    assert loaded.intervals["geology"]["rock"].isna().sum() == 1
    for name, table in db.tables.items():
        assert loaded.tables[name].dtypes.tolist() == table.dtypes.tolist()
        # Bit-exact floats, not merely close.
        for column in table.select_dtypes("float64").columns:
            assert (
                loaded.tables[name][column].to_numpy().tobytes()
                == table[column].to_numpy().tobytes()
            )
    document = json.loads((folder / "database.json").read_text(encoding="utf-8"))
    assert document["metadata"]["length_unit"] == "ft"
    assert document["metadata"]["column_maps"] == SMALL_MAPS
    # Saving again over the same folder and reloading changes nothing.
    assert_db_equal(DrillholeDatabase.load(loaded.save(folder)), db)


def test_parquet_round_trip_babbitt(babbitt_db, tmp_path):
    loaded = DrillholeDatabase.load(babbitt_db.save(tmp_path / "babbitt"))
    assert_db_equal(loaded, babbitt_db)
    assert loaded.collar["hole_id"].dtype == "str"


def test_load_rejects_foreign_folder(tmp_path):
    (tmp_path / "database.json").write_text('{"format": "other"}', encoding="utf-8")
    with pytest.raises(ValueError, match="not a bakumine-drillholes"):
        load_database(tmp_path)


# --------------------------------------------------------------------------------------
# synthetic benchmark deposit (P0.T3), if it has been generated
# --------------------------------------------------------------------------------------
def test_synthetic_default_dataset_imports():
    folder = REPO / "data" / "synthetic" / "default"
    files = {name: folder / f"{name}.csv" for name in ("collar", "survey", "assay", "geology")}
    if not all(path.is_file() for path in files.values()):
        pytest.skip("data/synthetic/default has not been generated yet (P0.T3)")
    kinds = {"collar": "collar", "survey": "survey", "assay": "interval", "geology": "interval"}
    sources = {}
    for name, path in files.items():
        suggestion = suggest_column_map(read_raw_table(path).columns, kinds[name])
        assert suggestion.missing == [], f"{name}: no column found for {suggestion.missing}"
        sources[name] = TableSource(path, suggestion.mapping)
    db = import_drillholes(
        sources["collar"],
        sources["survey"],
        {"assay": sources["assay"], "geology": sources["geology"]},
    )
    assert len(db.collar) > 0 and db.collar["hole_id"].is_unique
    assert np.isfinite(db.collar[["x", "y", "z", "max_depth"]].to_numpy()).all()
    for table in db.intervals.values():
        assert set(table["hole_id"]) <= set(db.collar["hole_id"])
        assert (table["depth_to"] > table["depth_from"]).all()
    assert len(db.desurvey()) == len(db.collar)
