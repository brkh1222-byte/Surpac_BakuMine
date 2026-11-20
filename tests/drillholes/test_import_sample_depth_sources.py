"""The sample-number rule file by file: one file of a table needs it, another has from / to
of its own and running sample numbers. The rule is set per file, never replaces a from / to
that is there, and is refused where stated intervals contradict it."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import make_recipe, run_recipe
from bakumine.core.drillholes.tables import _rule_against_known

DEPTH = 60
LOST = (20, 21, 22, 40, 41)  # rows of each hole of the second file that lost their from / to


@pytest.fixture
def files(tmp_path):
    """``first.csv``: holes 1-2, no from / to, the metre in the sample number.
    ``second.csv``: holes 3-4 with from / to; the sample numbers are running numbers that
    begin with the hole number (3001, 3002 ...), and a few rows have no from / to."""
    rng = np.random.default_rng(21)
    holes = [f"SYN-{i:02d}" for i in range(1, 5)]
    pd.DataFrame(
        {"Hole": holes, "East": 500000.0 + 50 * np.arange(4), "North": 5300000.0, "RL": 1200.0,
         "Depth": float(DEPTH), "Azimuth": 0.0, "Dip": -90.0}  # fmt: skip
    ).to_csv(tmp_path / "collar.csv", index=False)
    first = pd.DataFrame(
        [{"Hole": hole, "Sample No": f"{hole.replace('-', '')}-{m:03d}",
          "Au": round(float(rng.lognormal(-1, 1)), 3)}
         for hole in holes[:2] for m in range(1, DEPTH + 1)]  # fmt: skip
    )
    first.to_csv(tmp_path / "first.csv", index=False)
    rows = []
    for number, hole in ((3, holes[2]), (4, holes[3])):
        depth = 2.3
        for k in range(45):
            length = float(rng.choice([0.8, 1.0, 1.2, 1.5]))
            lost = k in LOST
            rows.append(
                {"Hole": hole, "Sample No": f"{number}{k + 1:03d}",
                 "From": "" if lost else round(depth, 2),
                 "To": "" if lost else round(depth + length, 2),
                 "Au": round(float(rng.lognormal(-1, 1)), 3)}  # fmt: skip
            )
            depth += length
    second = pd.DataFrame(rows)
    second.to_csv(tmp_path / "second.csv", index=False)
    return tmp_path, first, second


def recipe(first=True, second=True, **rule):
    def source(name, on):
        return {"file": name, **({"sample_depth": {"length": 1.0, **rule}} if on else {})}

    return make_recipe(
        {"collar": [{"file": "collar.csv"}],
         "assay": [source("first.csv", first), source("second.csv", second)]}  # fmt: skip
    )


def stated(second: pd.DataFrame) -> pd.DataFrame:
    return second[second["From"] != ""].reset_index(drop=True)


def check_tables(db, first, second, kind):
    table = db.intervals["assay"]
    one = table[table["hole_id"].isin(["SYN-01", "SYN-02"])]
    assert len(one) == len(first) == 2 * DEPTH  # every row of the first file, by the rule
    np.testing.assert_array_equal(one["depth_to"], list(range(1, DEPTH + 1)) * 2)
    np.testing.assert_array_equal(one["depth_from"], list(range(DEPTH)) * 2)
    two = table[table["hole_id"].isin(["SYN-03", "SYN-04"])].reset_index(drop=True)
    want = stated(second)
    assert len(two) == len(want) == 2 * (45 - len(LOST))  # only the rows with from / to
    np.testing.assert_array_equal(two["depth_from"], want["From"].astype(float))
    np.testing.assert_array_equal(two["depth_to"], want["To"].astype(float))
    np.testing.assert_array_equal(two["au"], want["Au"])
    qc = db.qc["assay"]
    assert set(qc["qc_type"]) == {kind} and len(qc) == 2 * len(LOST)
    assert set(qc["source"]) == {"second.csv"}


def test_rule_for_the_whole_table_spares_the_file_with_its_own_depths(files, capsys):
    folder, first, second = files
    db = run_recipe(recipe(), folder)
    # before: 3021 ... were read as 20-21 m ..., among intervals that say otherwise
    check_tables(db, first, second, "sample_number_not_a_depth")
    one, two = db.metadata.reports["sample_depth"]["assay"]
    assert (one["file"], one["rows"], one["rows_with_own_depths"]) == ("first.csv", 120, 0)
    assert one["contradicted"] == [] and one["contradicts_sources"] == []
    assert (two["file"], two["rows"], two["rows_with_own_depths"]) == ("second.csv", 0, 80)
    assert two["contradicted_rows"] == 2 * len(LOST) and two["repeated_hole_number"] == []
    assert [c["hole_id"] for c in two["contradicted"]] == ["SYN-03", "SYN-04"]
    found = two["contradicted"][0]
    assert (found["rows"], found["judged_by"], found["compared"]) == (len(LOST), "hole", 40)
    assert found["agreed"] < 0.8 * found["compared"]
    assert found["example"]["sample_id"].startswith("30")
    note = next(n for n in db.metadata.notes
                if n["kind"] == "sample_numbers_contradict_own_depths_not_imported")  # fmt: skip
    assert note["count"] == 2 * len(LOST)

    out = folder / "db"
    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "first.csv"), "--assay", str(folder / "second.csv"), "--auto",
                 "--sample-depth", "assay", "--out", str(out)])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert ("assay first.csv: depths from sample numbers: 120 rows in 2 holes (0 rows keep "
            "the from/to they have)") in printed  # fmt: skip
    assert ("assay second.csv: depths from sample numbers: 0 rows in 0 holes (80 rows keep "
            "the from/to they have)") in printed  # fmt: skip
    assert ("WARNING: assay second.csv: the sample numbers are not depths in 2 hole(s) "
            "(SYN-03, SYN-04): the file's own intervals say so") in printed  # fmt: skip
    assert "10 row(s) without from/to were NOT given depths" in printed
    assert len(load_database(out).intervals["assay"]) == 200


def test_source_option_sets_the_rule_per_file(files, capsys):
    folder, first, second = files
    base = ["import", "--collar", str(folder / "collar.csv"), "--assay",
            str(folder / "first.csv"), "--assay", str(folder / "second.csv"), "--auto"]  # fmt: skip

    def sources(*flags):
        path = folder / "recipe.json"
        assert main([*base, *flags, "--dry-run", "--save-recipe", str(path)]) == 0
        capsys.readouterr()
        return json.loads(path.read_text())["tables"]["assay"]

    # on for one file only
    one, two = sources("--source-option", "first.csv:sample-depth=on")
    assert one["sample_depth"] == {"length": 1.0} and "sample_depth" not in two
    # the table's rule, one file taken out
    one, two = sources("--sample-depth", "assay", "--sample-length", "2",
                       "--source-option", "second.csv:sample-depth=off")  # fmt: skip
    assert one["sample_depth"] == {"length": 2.0} and "sample_depth" not in two
    # length, patterns and tolerance of one file; the other keeps the table's
    one, two = sources(
        "--sample-depth", "assay", "--sample-pattern", r"X(?P<to>\d+)",
        "--source-option", "first.csv:sample-length=0.5",
        "--source-option", r"first.csv:sample-pattern=SYN0*{holenum}-(?P<to>\d+)",
        "--source-option", r"first.csv:sample-pattern=S:(?P<from>\d+)=(?P<to>\d+)",
        "--source-option", "first.csv:sample-depth-tolerance=2",
    )  # fmt: skip
    assert one["sample_depth"] == {
        "length": 0.5, "depth_tolerance": 2.0,
        "patterns": [r"SYN0*{holenum}-(?P<to>\d+)", r"S:(?P<from>\d+)=(?P<to>\d+)"],
    }  # fmt: skip
    assert two["sample_depth"] == {"length": 1.0, "patterns": [r"X(?P<to>\d+)"]}
    # a length alone says the file has the rule
    one, two = sources("--source-option", "first.csv:sample-length=1.5")
    assert one["sample_depth"] == {"length": 1.5} and "sample_depth" not in two

    out = folder / "db"
    code = main([*base, "--source-option", "first.csv:sample-depth=on", "--out", str(out)])
    printed = capsys.readouterr().out
    assert code == 0 and "second.csv: depths from sample numbers" not in printed
    # rows of the second file without from / to are rows without an interval, as ever
    check_tables(load_database(out), first, second, "no_interval")

    for flags, message in (
        (["--source-option", "collar.csv:sample-depth=on"], "is for interval tables"),
        (["--source-option", "first.csv:sample-depth=maybe"], "expects yes or no"),
        (["--source-option", "first.csv:sample-length=long"], "expects a number"),
        (["--source-option", "first.csv:sample-depth=on", "--source-option",
          "first.csv:sample-depth=off"], "twice"),
    ):  # fmt: skip
        assert main([*base, *flags, "--dry-run"]) == 2
        assert message in capsys.readouterr().err


def test_a_from_to_that_is_there_is_never_replaced(files):
    folder, first, second = files
    # "always" reads the sample number wherever the row has no valid from / to
    db = run_recipe(recipe(when="always"), folder)
    check_tables(db, first, second, "sample_number_not_a_depth")
    # ids that are depths, in a file that states some of its intervals: the stated ones
    # stay as they are (0.5 m long here), the others are read from the number
    rows = []
    for m in range(1, 31):
        half = m % 3 == 0
        rows.append({"Hole": "SYN-01", "Sample No": f"SYN01-{m:03d}", "Au": 0.1 * m,
                     "From": m - 0.5 if half else "", "To": float(m) if half else ""})  # fmt: skip
    pd.DataFrame(rows).to_csv(folder / "first.csv", index=False)
    for when in ("missing", "always"):
        db = run_recipe(recipe(second=False, when=when), folder)
        table = db.intervals["assay"]
        table = table[table["hole_id"] == "SYN-01"]
        np.testing.assert_array_equal(table["depth_to"], np.arange(1.0, 31.0))
        length = (table["depth_to"] - table["depth_from"]).to_numpy()
        np.testing.assert_array_equal(length, [1.0, 1.0, 0.5] * 10)
        (check,) = (c for c in db.metadata.reports["sample_depth"]["assay"]
                    if c["file"] == "first.csv")  # fmt: skip
        assert (check["rows"], check["rows_with_own_depths"]) == (20, 10)
        assert check["contradicted"] == []  # the stated intervals agree with the numbers


def test_too_few_own_intervals_in_a_hole_are_judged_by_the_file():
    holes = np.array(["A"] * 8 + ["B"] * 3)
    ids = np.array([str(i) for i in range(11)])
    derived = np.array([1.0, 2, 3, 4, 5, 6, 7, 8, 1, 2, 3])
    known = np.array([2.6, 3.9, 5.1, 6.0, 7.7, 9.0, np.nan, np.nan, 2.2, np.nan, np.nan])
    wanted = ~np.isfinite(known)
    refused, found = _rule_against_known(holes, ids, derived, known, np.ones(11), wanted)
    np.testing.assert_array_equal(refused, wanted)  # A by its 6 intervals, B by the file's 7
    assert [(f["hole_id"], f["judged_by"], f["compared"], f["agreed"]) for f in found] == [
        ("A", "hole", 6, 0), ("B", "file", 7, 0),
    ]  # fmt: skip
    # numbers that are depths: nothing is refused, whatever the lengths
    known = np.where(np.isfinite(known), derived, np.nan)
    refused, found = _rule_against_known(holes, ids, derived, known, np.ones(11), wanted)
    assert not refused.any() and found == []
    # nothing to compare with: the rule is applied as before
    none = np.full(11, np.nan)
    refused, found = _rule_against_known(holes, ids, derived, none, np.ones(11), ~wanted)
    assert not refused.any() and found == []


def test_derived_depths_that_contradict_another_file_are_an_error_for_the_file(files, capsys):
    folder, first, _ = files
    # another file states the intervals of hole 2: they start at 0.4 m. Hole 1 is there
    # too, exactly as the sample numbers say
    other = pd.DataFrame(
        [{"Hole": "SYN-02", "From": m + 0.4, "To": m + 1.4, "Ag": 1.0} for m in range(30)]
        + [{"Hole": "SYN-01", "From": float(m), "To": m + 1.0, "Ag": 2.0} for m in range(10)]
    )
    other.to_csv(folder / "second.csv", index=False)
    db = run_recipe(recipe(second=False), folder)
    table = db.intervals["assay"]
    one = table[table["hole_id"] == "SYN-01"]
    assert len(one) == DEPTH and one["ag"].notna().sum() == 10  # merged with the other file
    two = table[table["hole_id"] == "SYN-02"]
    assert len(two) == 30 and two["au"].isna().all()  # only the stated intervals
    np.testing.assert_allclose(two["depth_from"], np.arange(30) + 0.4)
    check = db.metadata.reports["sample_depth"]["assay"][0]
    assert check["contradicts_sources_rows"] == DEPTH
    (found,) = check["contradicts_sources"]
    assert (found["hole_id"], found["rows"], found["same_interval"]) == ("SYN-02", DEPTH, 0)
    assert found["overlapping"] == 31  # 0-1 ... 30-31 reach into 0.4-1.4 ... 29.4-30.4
    assert found["example"] == {
        "sample_id": "SYN02-001", "depth_from": 0.0, "depth_to": 1.0, "other_from": 0.4,
        "other_to": 1.4, "other_file": "second.csv",
    }  # fmt: skip
    qc = db.qc["assay"]
    assert set(qc["qc_type"]) == {"depth_contradicts_other_source"} and len(qc) == DEPTH
    assert set(qc["hole_id"]) == {"SYN-02"} and set(qc["source"]) == {"first.csv"}

    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "first.csv"), "--assay", str(folder / "second.csv"), "--auto",
                 "--source-option", "first.csv:sample-depth=on", "--dry-run"])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert ("ERROR: assay first.csv: depths read from sample numbers contradict the intervals "
            "of other files in 1 hole(s) (SYN-02), e.g. SYN-02 sample SYN02-001 read as 0-1 "
            "overlaps 0.4-1.4 of second.csv") in printed  # fmt: skip
    assert "The 60 derived row(s) of these holes were NOT imported" in printed
