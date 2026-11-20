"""The same interval written a little differently in two files (--interval-tolerance), and
the intervals that still overlap: reported at import, named by ``bakumine intercepts``."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import (
    interval_overlaps,
    make_recipe,
    near_intervals,
    run_recipe,
)

HOLES = ["SYN-01", "SYN-02", "SYN-03"]


def test_near_intervals_are_grouped_by_from_to_and_overlap():
    holes = ["A", "A", "A", "A", "A", "A", "B", "A", None]
    start = np.array([120.4, 120.4, 10.0, 10.4, 0.0, 0.0, 120.4, 120.3, 1.0])
    end = np.array([121.6, 121.3, 10.4, 10.8, 1.0, 0.4, 121.3, 121.5, 2.0])
    group = near_intervals(holes, start, end, 0.5)
    assert group[0] == group[1] == group[7]  # one sample, three ways of writing its ends
    assert group[2] != group[3]  # two short samples that follow each other: never one
    assert group[4] != group[5]  # from equal, to 0.6 apart
    assert group[6] not in (group[0], -1)  # another hole
    assert group[8] == -1  # no hole
    assert len(set(group.tolist())) == 7
    exact = near_intervals(holes, start, end, 0.0)  # tolerance 0: equal from and to only
    assert len(set(exact[:8].tolist())) == 8
    twice = near_intervals(["A", "A"], np.array([1.0, 1.0]), np.array([2.0, 2.0]), 0.0)
    assert twice[0] == twice[1]
    assert near_intervals(["A"], np.array([np.nan]), np.array([2.0]), 0.5).tolist() == [-1]

    table = pd.DataFrame({"hole_id": holes[:8], "depth_from": start[:8], "depth_to": end[:8]})
    pairs = interval_overlaps(table, list("abcdefgh"))
    found = {(r.hole_id, r.depth_from, r.depth_to, r.source, r.overlaps_from, r.overlaps_to,
              r.overlaps_source) for r in pairs.itertuples()}  # fmt: skip
    assert found == {
        ("A", 0.0, 1.0, "e", 0.0, 0.4, "f"),
        ("A", 120.4, 121.3, "b", 120.3, 121.5, "h"),
        ("A", 120.4, 121.6, "a", 120.3, 121.5, "h"),  # against the deepest end so far
    }
    assert sorted(pairs["overlap"].tolist()) == [0.4, 0.9, 1.1]
    assert interval_overlaps(table.iloc[2:4]).empty  # touching is not overlapping


@pytest.fixture
def two_files(tmp_path):
    """Three holes of 20 one-metre samples in ``a.csv``; the same in ``b.csv`` (another
    laboratory, other values) that ends three samples elsewhere: SYN-01 4-5 as 4-5.3,
    SYN-02 7-8 as 7.1-7.9 and SYN-03 12-13 as 12-12.4 (too far to be the same)."""
    rng = np.random.default_rng(8)
    pd.DataFrame(
        {"Hole": HOLES, "East": [500000.0, 500050.0, 500100.0], "North": 5300000.0,
         "RL": 1200.0, "Depth": 20.0, "Azimuth": 0.0, "Dip": -90.0}  # fmt: skip
    ).to_csv(tmp_path / "collar.csv", index=False)
    first = pd.DataFrame(
        [{"Hole": hole, "From": float(k), "To": float(k + 1), "Lab": "first",
          "Au": round(float(rng.lognormal(-1, 1)), 3)}
         for hole in HOLES for k in range(20)]  # fmt: skip
    )
    second = first.copy()
    second["Lab"] = "other"
    second["Au"] = second["Au"] + 1.0
    for hole, start, new in (("SYN-01", 4.0, (4.0, 5.3)), ("SYN-02", 7.0, (7.1, 7.9)),
                             ("SYN-03", 12.0, (12.0, 12.4))):  # fmt: skip
        row = (second["Hole"] == hole) & (second["From"] == start)
        second.loc[row, ["From", "To"]] = new
    first.to_csv(tmp_path / "a.csv", index=False)
    second.to_csv(tmp_path / "b.csv", index=False)
    return tmp_path, first, second


def both(**merge):
    return make_recipe(
        {"collar": [{"file": "collar.csv"}], "assay": [{"file": "a.csv"}, {"file": "b.csv"}]},
        merge=merge or None,
    )


def test_two_files_that_end_a_sample_differently_give_one_interval(two_files):
    folder, first, _ = two_files
    assert both()["merge"]["interval_tolerance"] == 0.5
    # the tolerance for values (0.01) is not the one for depths: before, 4-5 and 4-5.3
    # both stayed
    db = run_recipe(both(), folder)
    table = db.intervals["assay"]
    merged = db.metadata.reports["conflicts"]["assay"]
    assert (merged["merged"], merged["near_intervals"]) == (59, 2)
    assert merged["interval_tolerance"] == 0.5
    assert merged["near_rows"] == [
        {"hole_id": "SYN-01", "depth_from": 4.0, "depth_to": 5.0, "dropped_from": 4.0,
         "dropped_to": 5.3, "kept_source": "a.csv", "dropped_source": "b.csv"},
        {"hole_id": "SYN-02", "depth_from": 7.0, "depth_to": 8.0, "dropped_from": 7.1,
         "dropped_to": 7.9, "kept_source": "a.csv", "dropped_source": "b.csv"},
    ]  # fmt: skip
    # the first file's intervals and values, plus the one row that is another interval
    assert len(table) == 61 and "__file__" not in table.columns
    mine = table[table["lab"] == "first"].reset_index(drop=True)
    np.testing.assert_array_equal(mine["depth_to"], first["To"])
    np.testing.assert_array_equal(mine["au"], first["Au"])
    listed = db.listings["conflicts_assay"]
    near = listed[listed["column"] == "depth_from,depth_to"]
    assert near["kept"].tolist() == ["4-5", "7-8"]
    assert near["dropped"].tolist() == ["4-5.3", "7.1-7.9"] and near["informational"].all()
    # what is left overlapping is reported with both files
    over = db.metadata.reports["overlaps"]["assay"]
    assert (over["count"], over["holes"], over["listing"]) == (1, 1, "overlaps_assay.csv")
    assert over["rows"] == [
        {"hole_id": "SYN-03", "depth_from": 12.0, "depth_to": 13.0, "source": "a.csv",
         "overlaps_from": 12.0, "overlaps_to": 12.4, "overlaps_source": "b.csv", "overlap": 0.4}
    ]  # fmt: skip
    assert len(db.listings["overlaps_assay"]) == 1
    note = next(n for n in db.metadata.notes if n["kind"] == "intervals_overlap")
    assert note["examples"] == ["SYN-03 12-13 / 12-12.4"]

    # last file wins: its from / to stay
    last = run_recipe(both(precedence="last"), folder).intervals["assay"]
    row = last[(last["hole_id"] == "SYN-01") & (last["depth_from"] == 4.0)]
    assert row["depth_to"].tolist() == [5.3] and row["lab"].tolist() == ["other"]
    # a wider tolerance does not take the third pair: 12-12.4 is less than half of 12-13
    wide = run_recipe(both(interval_tolerance=0.6), folder)
    assert wide.metadata.reports["conflicts"]["assay"]["near_intervals"] == 2
    assert wide.metadata.reports["overlaps"]["assay"]["count"] == 1
    # no tolerance: all three stay and overlap (4-5.3 reaches into 5-6 as well)
    none = run_recipe(both(interval_tolerance=0.0), folder)
    assert len(none.intervals["assay"]) == 63
    assert none.metadata.reports["conflicts"]["assay"]["near_intervals"] == 0
    assert none.metadata.reports["overlaps"]["assay"]["count"] == 4
    with pytest.raises(ValueError, match="interval_tolerance must be 0 or more"):
        both(interval_tolerance=-1)


def test_cli_interval_tolerance_overlaps_and_the_intercepts_message(two_files, capsys):
    folder, _, _ = two_files
    common = ["import", "--collar", str(folder / "collar.csv"), "--assay", str(folder / "a.csv"),
              "--assay", str(folder / "b.csv"), "--auto"]  # fmt: skip
    out = folder / "db"
    assert main([*common, "--out", str(out)]) == 0
    printed = capsys.readouterr().out
    assert "2 of the rows joined had another from/to in the other file (within 0.5)" in printed
    assert "SYN-01 4-5 (a.csv) = 4-5.3 (b.csv)" in printed
    assert ("WARNING: assay: 1 interval(s) in 1 hole(s) overlap another interval of their hole "
            "(listed in overlaps_assay.csv)") in printed  # fmt: skip
    assert "SYN-03: 12-13 (a.csv) overlaps 12-12.4 (b.csv)" in printed
    saved = pd.read_csv(out / "overlaps_assay.csv")
    assert saved[["hole_id", "source", "overlaps_source"]].values.tolist() == [
        ["SYN-03", "a.csv", "b.csv"]
    ]
    assert load_database(out).metadata.recipe["merge"]["interval_tolerance"] == 0.5
    # intercepts says which hole and which intervals
    intercepts = ["--grade", "au", "--cutoff", "0.1", "--out", str(folder / "i.csv")]
    assert main(["intercepts", str(out), *intercepts]) == 2
    error = capsys.readouterr().err
    assert "samples overlap in 1 hole(s), 1 place(s): hole 'SYN-03': 12-12.4 and 12-13" in error
    assert "--interval-tolerance" in error

    strict = folder / "strict"
    assert main([*common, "--interval-tolerance", "0", "--out", str(strict)]) == 0
    printed = capsys.readouterr().out
    assert "4 interval(s) in 3 hole(s) overlap" in printed
    assert "SYN-01: 4-5.3 (b.csv) overlaps 4-5 (a.csv)" in printed
    assert main(["intercepts", str(strict), *intercepts]) == 2
    error = capsys.readouterr().err
    assert "samples overlap in 3 hole(s), 4 place(s)" in error
    assert ("hole 'SYN-01': 4-5 and 4-5.3; 4-5.3 and 5-6; hole 'SYN-02': 7-8 and 7.1-7.9; "
            "hole 'SYN-03': 12-12.4 and 12-13") in error  # fmt: skip

    clean = folder / "clean"  # without the hole that overlaps, nothing is said and all runs
    assert main([*common, "--exclude-holes", "SYN-03", "--out", str(clean)]) == 0
    assert "overlap" not in capsys.readouterr().out
    assert main(["intercepts", str(clean), *intercepts]) == 0


def test_near_duplicates_within_one_file_follow_the_duplicate_rule(two_files):
    folder, first, _ = two_files
    again = first[(first["Hole"] == "SYN-02") & (first["From"] == 3.0)].copy()
    again[["To", "Lab"]] = [4.2, "repeat"]
    again["Au"] = again["Au"] + 1.0
    pd.concat([first, again]).to_csv(folder / "assay.csv", index=False)

    def recipe(**merge):
        return make_recipe(
            {"collar": [{"file": "collar.csv"}], "assay": [{"file": "assay.csv"}]},
            merge=merge or None,
        )

    db = run_recipe(recipe(), folder)
    table = db.intervals["assay"]
    assert len(table) == 60 and table["lab"].eq("first").all()
    report = db.metadata.reports["duplicates"]["assay"]
    assert (report["intervals"], report["rows_removed"], report["near"]) == (1, 1, 1)
    assert report["tolerance"] == 0.5
    assert report["examples"] == [
        {"hole_id": "SYN-02", "depth_from": 3.0, "depth_to": 4.0, "rows": 2}
    ]
    listing = db.listings["duplicates_assay"]
    assert listing["depth_to"].tolist() == [4.0, 4.2] and listing["kept"].tolist() == [True, False]
    assert listing["source"].tolist() == ["assay.csv"] * 2
    assert "overlaps" not in db.metadata.reports
    mean = run_recipe(recipe(duplicate_intervals="mean"), folder).intervals["assay"]
    row = mean[(mean["hole_id"] == "SYN-02") & (mean["depth_from"] == 3.0)]
    assert row["depth_to"].tolist() == [4.0]  # the first row's depths, the mean of the values
    assert row["au"].iloc[0] == pytest.approx(float(again["Au"].iloc[0]) - 0.5)
    with pytest.raises(ValueError, match=r"1 interval\(s\) are in the table more than once"):
        run_recipe(recipe(duplicate_intervals="error"), folder)
    # with no tolerance the two stay, and the import says that they overlap
    strict = run_recipe(recipe(interval_tolerance=0.0), folder)
    assert len(strict.intervals["assay"]) == 61 and "duplicates" not in strict.metadata.reports
    assert strict.metadata.reports["overlaps"]["assay"]["rows"][0]["source"] == "assay.csv"
