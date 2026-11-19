"""P1.T2 acceptance tests: validation report."""

import copy
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from bakumine.core.drillholes.importer import DrillholeDatabase
from bakumine.core.drillholes.validate import CHECKS, SEVERITIES, validate


def _append(table: pd.DataFrame, rows: list[dict]) -> pd.DataFrame:
    extra = pd.DataFrame(rows, columns=table.columns).astype(table.dtypes.to_dict())
    return pd.concat([table, extra], ignore_index=True)


def _set(table: pd.DataFrame, hole: str, where: dict, **values: float) -> int:
    """Change one row (selected by hole and column values) in place; return its position."""
    mask = table["hole_id"] == hole
    for column, value in where.items():
        mask &= table[column] == value
    (row,) = np.flatnonzero(mask)
    for column, value in values.items():
        table.loc[row, column] = value
    return int(row)


def corrupt(db: DrillholeDatabase) -> tuple[DrillholeDatabase, dict[str, tuple]]:
    """Inject every check exactly once. Returns the database and, per check code, the
    expected ``(table, hole_id, row)`` of its single finding."""
    collar, survey = db.collar.copy(), db.survey.copy()
    assay, geology = db.intervals["assay"].copy(), db.intervals["geology"].copy()
    expected: dict[str, tuple] = {}
    nan = np.nan

    def hole(hole_id, x=1025.0, y=5025.0, z=202.0, max_depth=100.0, surveyed=True):
        nonlocal collar, survey
        collar = _append(collar, [dict(hole_id=hole_id, x=x, y=y, z=z, max_depth=max_depth)])
        if surveyed:
            survey = _append(survey, [dict(hole_id=hole_id, depth=0.0, dip=-90.0, azimuth=0.0)])
        return len(collar) - 1

    # --- collar ---
    collar = _append(collar, [collar.iloc[0].to_dict()])  # DH001 twice
    expected["duplicate_hole_id"] = ("collar", "DH001", len(collar) - 1)
    expected["non_finite_coordinate"] = ("collar", "NF1", hole("NF1", x=nan))
    expected["invalid_max_depth"] = ("collar", "ND1", hole("ND1", max_depth=nan))
    expected["collar_outlier"] = ("collar", "FAR1", hole("FAR1", x=1.0e6))
    hole("NS1", x=1075.0, surveyed=False)
    expected["no_survey"] = ("survey", "NS1", None)

    # --- survey ---
    survey = survey[~((survey["hole_id"] == "DH003") & (survey["depth"] == 0))]
    survey = survey.reset_index(drop=True)
    expected["no_survey_at_zero"] = ("survey", "DH003", None)
    row = _set(survey, "DH005", {"depth": 50}, dip=-95.0)
    expected["dip_out_of_range"] = ("survey", "DH005", row)
    row = _set(survey, "DH006", {"depth": 50}, azimuth=370.0)
    expected["azimuth_out_of_range"] = ("survey", "DH006", row)
    row = _set(survey, "1004", {"depth": 100}, azimuth=150.0)
    expected["sudden_deviation"] = ("survey", "1004", row)
    survey = _append(
        survey,
        [
            dict(hole_id="DH002", depth=nan, dip=-60.0, azimuth=90.0),
            dict(hole_id="DH001", depth=50.0, dip=-59.0, azimuth=91.0),
            dict(hole_id="DH002", depth=150.0, dip=-58.0, azimuth=92.0),
        ],
    )
    expected["invalid_survey_station"] = ("survey", "DH002", len(survey) - 3)
    expected["duplicate_survey_depth"] = ("survey", "DH001", len(survey) - 2)
    expected["survey_past_max_depth"] = ("survey", "DH002", len(survey) - 1)

    # --- intervals ---
    assay = _append(assay, [dict(hole_id="GHOST", depth_from=0.0, depth_to=10.0, au=1.0, cu=1.0)])
    expected["missing_collar"] = ("assay", "GHOST", len(assay) - 1)
    row = _set(assay, "DH002", {"depth_from": 90}, depth_to=90.0)
    expected["from_ge_to"] = ("assay", "DH002", row)
    _set(assay, "DH003", {"depth_from": 10}, depth_to=22.0)
    row = _set(assay, "DH003", {"depth_from": 20})
    expected["overlap"] = ("assay", "DH003", row)
    row = _set(assay, "DH005", {"depth_from": 30}, depth_from=32.0)
    expected["gap"] = ("assay", "DH005", row)
    row = _set(geology, "DH006", {"depth_from": 62.5}, depth_to=105.0)
    expected["interval_past_max_depth"] = ("geology", "DH006", row)
    geology = _append(geology, [dict(hole_id="DH001", depth_from=nan, depth_to=5.0, rock="ORE")])
    expected["invalid_interval_depth"] = ("geology", "DH001", len(geology) - 1)

    corrupted = DrillholeDatabase(
        collar, survey, {"assay": assay, "geology": geology}, replace(db.metadata)
    )
    return corrupted, expected


def test_clean_dataset_has_no_findings(small_db):
    report = validate(small_db)
    assert report.ok
    assert report.findings.empty
    assert list(report.findings.columns) == [
        "severity", "code", "table", "hole_id", "row", "message",
    ]  # fmt: skip
    assert report.counts() == {}
    summary = report.summary
    assert summary["hole_id"].tolist() == small_db.collar["hole_id"].tolist()
    assert summary["in_collar"].all()
    assert (summary["survey_stations"] == 3).all()
    assert (summary["interval_rows"] == 13).all()  # 10 assays + 3 rock intervals
    assert (summary[["errors", "warnings", "infos"]] == 0).all().all()
    assert (summary["status"] == "ok").all()


def test_each_injected_defect_triggers_its_check_exactly_once(small_db):
    db, expected = corrupt(small_db)
    assert set(expected) == set(CHECKS), "the corrupted dataset must cover every check"
    report = validate(db)
    assert report.counts() == {code: 1 for code in CHECKS}
    assert not report.ok

    findings = report.findings.set_index("code")
    for code, (table, hole, row) in expected.items():
        finding = findings.loc[code]
        assert finding["severity"] == CHECKS[code][0], code
        assert finding["table"] == table, code
        assert finding["hole_id"] == hole, code
        if row is None:
            assert pd.isna(finding["row"]), code
        else:
            assert finding["row"] == row, code
        assert isinstance(finding["message"], str) and finding["message"], code
    assert set(report.findings["severity"]) == set(SEVERITIES)
    assert len(report.errors) == sum(s == "error" for s, _ in CHECKS.values())
    assert len(report.warnings) == sum(s == "warning" for s, _ in CHECKS.values())


def test_messages_carry_the_offending_values(small_db):
    db, _ = corrupt(small_db)
    message = validate(db).findings.set_index("code")["message"]
    assert message["overlap"] == (
        "interval 20-30 starts before depth 22, where an earlier interval ends (overlap 2)"
    )
    assert message["gap"] == "nothing logged from 30 to 32 (2 long)"
    assert message["interval_past_max_depth"] == (
        "interval 62.5-105 ends below max_depth 100"
    )
    assert "-95" in message["dip_out_of_range"]
    assert "370" in message["azimuth_out_of_range"]
    assert "150" in message["survey_past_max_depth"]
    assert "GHOST" in message["missing_collar"] and "1 assay row" in message["missing_collar"]
    assert "deg per 30 m" in message["sudden_deviation"]


def test_per_hole_summary(small_db):
    db, expected = corrupt(small_db)
    report = validate(db)
    summary = report.summary.set_index("hole_id")
    assert summary.index.is_unique
    assert set(summary.index) == set(db.collar["hole_id"]) | {"GHOST"}
    assert not summary.loc["GHOST", "in_collar"]
    assert summary.loc["GHOST", "status"] == "error"
    assert summary.loc["NS1", "survey_stations"] == 0
    # Totals agree with the findings table.
    assert summary["errors"].sum() == len(report.errors)
    assert summary["warnings"].sum() == len(report.warnings)
    assert summary["infos"].sum() == 1
    # DH005: one error (dip) and one info (gap).
    row = summary.loc["DH005"]
    assert (row["errors"], row["warnings"], row["infos"]) == (1, 0, 1)
    assert row["status"] == "error"
    assert row["codes"] == "dip_out_of_range,gap"
    assert summary.loc["FAR1", "status"] == "warning"
    assert summary.loc["DH003", "codes"] == "no_survey_at_zero,overlap"
    for code, (_, hole, _) in expected.items():
        assert code in summary.loc[hole, "codes"].split(",")
    assert report.for_hole("DH005")["code"].tolist() == ["dip_out_of_range", "gap"]


def test_validation_never_mutates_the_data(small_db):
    db, _ = corrupt(small_db)
    before = copy.deepcopy(db)
    validate(db)
    for name, table in db.tables.items():
        assert_frame_equal(table, before.tables[name])
    assert db.metadata == before.metadata


def test_deviation_threshold_is_configurable_and_unit_aware(small_db):
    # Clean data: 1 degree of dip and 1 of azimuth per 50 m is about 0.67 deg per 30 m.
    assert validate(small_db, max_deviation_deg_per_30m=1.0).findings.empty
    strict = validate(small_db, max_deviation_deg_per_30m=0.5)
    assert strict.counts() == {"sudden_deviation": 12}  # two station pairs in each of 6 holes
    # Independent dogleg of the first pair: (dip -60, az 90) -> (dip -59, az 91) over 50 m.
    d1 = np.array([np.cos(np.radians(60)), 0.0, -np.sin(np.radians(60))])
    az = np.radians(91)
    d2 = np.array([np.cos(np.radians(59)) * np.sin(az), np.cos(np.radians(59)) * np.cos(az),
                   -np.sin(np.radians(59))])  # fmt: skip
    rate = np.degrees(np.arccos(d1 @ d2)) / 50.0 * 30.0
    assert f"({rate:.2f} deg per 30 m" in strict.findings["message"].iloc[0]

    # The same depths read as feet are a 3.28 times tighter bend: 0.67 -> 2.2 deg per 30 m.
    in_feet = DrillholeDatabase(
        small_db.collar, small_db.survey, small_db.intervals,
        replace(small_db.metadata, length_unit="ft"),
    )  # fmt: skip
    assert validate(in_feet, max_deviation_deg_per_30m=1.0).counts() == {"sudden_deviation": 12}
    assert validate(in_feet, max_deviation_deg_per_30m=2.5).findings.empty

    db, _ = corrupt(small_db)
    assert "sudden_deviation" not in validate(db, max_deviation_deg_per_30m=25.0).counts()


def test_outlier_factor_and_minimum_holes(small_db):
    db, _ = corrupt(small_db)
    assert "collar_outlier" not in validate(db, outlier_factor=1e6).counts()
    assert "collar_outlier" not in validate(db, min_outlier_holes=50).counts()
    # An elevation typo (a dropped digit) is an outlier too.
    collar = small_db.collar.copy()
    collar.loc[2, "z"] = 20.2
    typo = DrillholeDatabase(collar, small_db.survey, small_db.intervals, small_db.metadata)
    report = validate(typo)
    assert report.counts() == {"collar_outlier": 1}
    assert report.findings["hole_id"].iloc[0] == "DH003"
    assert "elevation" in report.findings["message"].iloc[0]


def test_missing_collar_is_reported_once_per_table_and_hole(small_db):
    collar = small_db.collar[small_db.collar["hole_id"] != "DH002"].reset_index(drop=True)
    db = DrillholeDatabase(collar, small_db.survey, small_db.intervals, small_db.metadata)
    report = validate(db)
    assert report.counts() == {"missing_collar": 3}  # survey, assay, geology
    assert report.findings["table"].tolist() == ["survey", "assay", "geology"]
    assert report.findings["row"].tolist() == [3, 10, 3]  # first row of DH002 in each table
    assert "10 assay row(s)" in report.with_code("missing_collar")["message"].iloc[1]


def test_depth_tolerance(small_db):
    assay = small_db.intervals["assay"].copy()
    assay.loc[1, "depth_from"] = 10.0 + 1e-9  # float noise at a shared boundary
    db = DrillholeDatabase(
        small_db.collar, small_db.survey, {"assay": assay}, small_db.metadata
    )
    assert validate(db).findings.empty
    assert validate(db, depth_tolerance=0.0).counts() == {"gap": 1}


def test_invalid_thresholds_raise(small_db):
    with pytest.raises(ValueError):
        validate(small_db, max_deviation_deg_per_30m=0.0)
    with pytest.raises(ValueError):
        validate(small_db, depth_tolerance=-1.0)


# --------------------------------------------------------------------------------------
# Babbitt: findings must agree with the quirks documented in SOURCE.md
# --------------------------------------------------------------------------------------
def test_babbitt_findings_match_documented_quirks(babbitt_db):
    report = validate(babbitt_db)
    counts = report.counts()

    # "hole IDs are unique in the collar and all three tables cover exactly the same 399
    # holes", "every hole has a station at 0", dips 32..90 and azimuths 0..359, "no
    # duplicate depths", intervals "contiguous (no gaps, no overlaps)": no errors, no gaps.
    assert report.ok
    assert set(counts) <= {"survey_past_max_depth", "sudden_deviation"}

    # "70 holes have exactly two stations: one at 0 and one at AT = 90000 ... No other
    # station lies deeper than its hole."
    sentinel = report.with_code("survey_past_max_depth")
    assert len(sentinel) == 70
    assert sentinel["hole_id"].is_unique
    assert (sentinel["severity"] == "warning").all()
    rows = babbitt_db.survey.iloc[sentinel["row"].to_numpy(dtype=int)]
    assert (rows["depth"] == 90000).all()
    assert rows["hole_id"].tolist() == sentinel["hole_id"].tolist()
    station_count = babbitt_db.survey.groupby("hole_id").size()
    assert (station_count[sentinel["hole_id"]] == 2).all()
    assert sentinel["message"].str.startswith("station at depth 90000 is below max_depth").all()

    # A sentinel repeats the dip and azimuth of the station at 0, so it is not a deviation.
    deviation = report.with_code("sudden_deviation")
    assert set(deviation["hole_id"]).isdisjoint(sentinel["hole_id"])
    assert len(deviation) == counts.get("sudden_deviation", 0) < 20

    summary = report.summary
    assert len(summary) == 399 and summary["in_collar"].all()
    assert summary["survey_stations"].sum() == 2628
    assert summary["interval_rows"].sum() == 35616
    assert (summary["errors"] == 0).all()
    assert (summary["status"] == "warning").sum() == sentinel["hole_id"].nunique() + deviation[
        "hole_id"
    ].nunique()


def test_babbitt_deviation_findings_against_independent_dogleg(babbitt_db):
    """Recompute the flagged station pairs with the textbook dogleg formula."""
    survey = babbitt_db.survey
    report = validate(babbitt_db)
    foot = 1200.0 / 3937.0
    flagged = set()
    for _, stations in survey.groupby("hole_id", sort=False):
        s = stations.sort_values("depth")
        inc = np.radians(90.0 - s["dip"].to_numpy())  # inclination from vertical
        azi = np.radians(s["azimuth"].to_numpy())
        cos_dl = np.cos(inc[1:] - inc[:-1]) - np.sin(inc[:-1]) * np.sin(inc[1:]) * (
            1.0 - np.cos(azi[1:] - azi[:-1])
        )
        dogleg = np.degrees(np.arccos(np.clip(cos_dl, -1.0, 1.0)))
        length_m = np.diff(s["depth"].to_numpy()) * foot
        rate = dogleg / length_m * 30.0
        flagged.update(s.index[1:][rate > 10.0].tolist())
    assert set(report.with_code("sudden_deviation")["row"].tolist()) == flagged


def test_babbitt_in_metres_gives_the_same_findings(babbitt_db):
    from drillhole_samples import import_babbitt

    from bakumine.core.drillholes.importer import ImportOptions

    metres = import_babbitt(
        ImportOptions(length_unit="ftUS", dip_convention="positive_down", convert_to_metres=True,
                      normalise_dip=True)
    )  # fmt: skip
    a, b = validate(babbitt_db).findings, validate(metres).findings
    assert_frame_equal(a.drop(columns="message"), b.drop(columns="message"))
