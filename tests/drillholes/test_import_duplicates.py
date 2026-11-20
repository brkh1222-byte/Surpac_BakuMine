"""An interval that is in a table twice (an unmarked repeat assay), and intervals of two
files whose end differs by a rounding."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import (
    make_recipe,
    merge_sources,
    resolve_duplicate_intervals,
    run_recipe,
)
from bakumine.core.drillholes.validate import validate

HOLES = ["SYN-01", "SYN-02", "SYN-03"]
REPEATED = [("SYN-01", 4.0), ("SYN-01", 9.0), ("SYN-03", 0.0)]  # (hole, from)


@pytest.fixture
def files(tmp_path):
    rng = np.random.default_rng(8)
    collar = pd.DataFrame(
        {"Hole": HOLES, "East": [500000.0, 500050.0, 500100.0], "North": 5300000.0,
         "RL": 1200.0, "Depth": 20.0, "Azimuth": 0.0, "Dip": -90.0}  # fmt: skip
    )
    collar.to_csv(tmp_path / "collar.csv", index=False)
    rows = []
    for hole in HOLES:
        for k in range(20):
            row = {"Hole": hole, "From": float(k), "To": float(k + 1), "Lab": "first",
                   "Au": round(float(rng.lognormal(-1, 1)), 3),
                   "Ag": round(float(k), 1)}  # fmt: skip
            rows.append(row)
            if (hole, float(k)) in REPEATED:  # the repeat follows its original, unmarked
                rows.append({**row, "Lab": "repeat", "Au": round(row["Au"] + 0.5, 3),
                             "Ag": np.nan})  # fmt: skip
    assay = pd.DataFrame(rows)
    assay.to_csv(tmp_path / "assay.csv", index=False)
    return tmp_path, assay


def recipe(**merge):
    return make_recipe(
        {"collar": [{"file": "collar.csv"}], "assay": [{"file": "assay.csv"}]},
        merge=merge or None,
    )


def test_default_keeps_the_first_row_and_reports(files):
    folder, assay = files
    assert recipe()["merge"]["duplicate_intervals"] == "keep_first"
    db = run_recipe(recipe(), folder)
    table = db.intervals["assay"]
    first = assay[assay["Lab"] == "first"].reset_index(drop=True)
    assert len(table) == 60 and table["lab"].eq("first").all()
    np.testing.assert_array_equal(table["au"], first["Au"])
    report = db.metadata.reports["duplicates"]["assay"]
    assert (report["rule"], report["intervals"], report["rows_removed"], report["holes"]) == (
        "keep_first", 3, 3, 2,
    )  # fmt: skip
    assert report["examples"][0] == {"hole_id": "SYN-01", "depth_from": 4.0, "depth_to": 5.0,
                                     "rows": 2}  # fmt: skip
    assert report["listing"] == "duplicates_assay.csv"
    listing = db.listings["duplicates_assay"]  # both rows of each such interval
    assert len(listing) == 6 and listing["kept"].tolist() == [True, False] * 3
    assert listing["lab"].tolist() == ["first", "repeat"] * 3
    note = next(n for n in db.metadata.notes if n["kind"] == "duplicate_intervals_keep_first")
    assert note["count"] == 3
    # what the validator said of the repeats before: overlapping samples. Not any more
    codes = set(validate(db).findings["code"])
    assert not {c for c in codes if "overlap" in c or "duplicate" in c}


def test_mean_of_the_rows(files):
    folder, assay = files
    db = run_recipe(recipe(duplicate_intervals="mean"), folder)
    table = db.intervals["assay"].set_index(["hole_id", "depth_from"])
    assert len(table) == 60
    for hole, start in REPEATED:
        rows = assay[(assay["Hole"] == hole) & (assay["From"] == start)]
        assert len(rows) == 2
        np.testing.assert_allclose(table.loc[(hole, start), "au"], rows["Au"].mean())
        # the repeat has no silver: the mean of what there is; text is the first row's
        np.testing.assert_allclose(table.loc[(hole, start), "ag"], rows["Ag"].iloc[0])
        assert table.loc[(hole, start), "lab"] == "first"
    untouched = assay[assay["Lab"] == "first"].set_index(["Hole", "From"])["Au"]
    np.testing.assert_allclose(table.loc[("SYN-02", 7.0), "au"], untouched.loc[("SYN-02", 7.0)])
    assert db.metadata.reports["duplicates"]["assay"]["rule"] == "mean"


def test_error_stops_the_import(files):
    folder, _ = files
    with pytest.raises(ValueError, match=r"assay: 3 interval\(s\) are in the table more than "
                                         r"once \(SYN-01 4-5 x 2; SYN-01 9-10 x 2"):  # fmt: skip
        run_recipe(recipe(duplicate_intervals="error"), folder)
    with pytest.raises(ValueError, match="duplicate_intervals must be one of"):
        recipe(duplicate_intervals="median")


def test_cli_option_summary_and_listing(files, capsys):
    folder, _ = files
    common = ["import", "--collar", str(folder / "collar.csv"), "--assay",
              str(folder / "assay.csv"), "--auto"]  # fmt: skip
    out = folder / "db"
    assert main([*common, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    said = ("assay duplicate intervals (keep-first): 3 interval(s) in 2 hole(s) were there "
            "more than once, 3 row(s) removed; listed in duplicates_assay.csv")  # fmt: skip
    assert said in printed
    assert "duplicates_assay.csv (6 rows, the full listing)" in printed
    assert "assay intervals 60" in printed
    saved = pd.read_csv(out / "duplicates_assay.csv")
    assert saved["kept"].tolist() == [True, False] * 3 and set(saved["rule"]) == {"keep_first"}
    assert load_database(out).metadata.recipe["merge"]["duplicate_intervals"] == "keep_first"
    assert main([*common, "--duplicate-intervals", "mean", "--dry-run"]) == 0
    assert "duplicate intervals (mean): 3 interval(s)" in capsys.readouterr().out
    assert main([*common, "--duplicate-intervals", "error", "--dry-run"]) == 2
    assert "are in the table more than once" in capsys.readouterr().err


def test_no_duplicates_no_report(files):
    folder, assay = files
    assay[assay["Lab"] == "first"].to_csv(folder / "assay.csv", index=False)
    db = run_recipe(recipe(duplicate_intervals="error"), folder)
    assert "duplicates" not in db.metadata.reports and not db.listings
    table, report, listing = resolve_duplicate_intervals(db.intervals["assay"])
    assert table is db.intervals["assay"] and report["intervals"] == 0 and listing.empty


def test_ends_within_the_tolerance_are_one_interval_when_files_merge(files):
    folder, assay = files
    first = assay[assay["Lab"] == "first"].copy()
    second = first.copy()
    second["To"] = second["To"] - 0.004  # the other file rounds the end differently
    second.loc[second.index[0], "To"] -= 0.5  # this one really is another interval
    second["Au"] = np.nan
    second["Cu"] = 0.25
    first.to_csv(folder / "a.csv", index=False)
    second.to_csv(folder / "b.csv", index=False)
    both = make_recipe(
        {"collar": [{"file": "collar.csv"}], "assay": [{"file": "a.csv"}, {"file": "b.csv"}]}
    )
    db = run_recipe(both, folder)
    table = db.intervals["assay"]
    merged = db.metadata.reports["conflicts"]["assay"]
    assert merged["merged"] == 59 and merged["conflicts"] == 0
    assert len(table) == 61
    joined = table[table["cu"].notna() & table["au"].notna()]
    assert len(joined) == 59 and (joined["depth_to"] == joined["depth_from"] + 1.0).all()
    # with a tolerance of nothing they are different intervals, as before
    strict = make_recipe(both["tables"], merge={"tolerance": 0.0})
    assert run_recipe(strict, folder).metadata.reports["conflicts"]["assay"]["merged"] == 0

    keys = ("hole_id", "depth_from", "depth_to")
    pair = pd.DataFrame({"hole_id": ["A", "A", "A"], "depth_from": [0.0, 0.0, 0.0],
                         "depth_to": [1.0, 1.009, 1.02], "au": [1.0, np.nan, 3.0]})  # fmt: skip
    out, report = merge_sources(pair, np.array([0, 1, 1]), keys, tolerance=0.01)
    assert out["depth_to"].tolist() == [1.0, 1.02] and report["merged"] == 1
