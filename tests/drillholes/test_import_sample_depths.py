"""Depths read from sample numbers: ids that repeat the hole number before the metre,
and the check of every derived depth against the depth of the hole."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import make_recipe, run_recipe
from bakumine.core.drillholes.tables import depths_from_sample_ids

DEPTH = 250


def plain_ids(hole: str, metres) -> list[str]:
    """``SYN04-152``: the metre after the hole."""
    return [f"{hole.replace('-', '')}-{m:03d}" for m in metres]


def repeated_ids(hole: str, metres) -> list[str]:
    """``SYN04-4152``: the hole number again, then the metre in three digits."""
    number = int(hole[-2:])
    return [f"{hole.replace('-', '')}-{number}{m:03d}" for m in metres]


def test_both_id_styles_give_the_same_depths():
    metres = list(range(1, DEPTH + 1))
    for hole in ("SYN-01", "SYN-04", "SYN-12", "SYN-25"):
        want = np.array(metres, dtype=float)
        for ids in (plain_ids(hole, metres), repeated_ids(hole, metres)):
            for depth in ({hole: float(DEPTH)}, None):  # the hole depth known or not
                found: dict = {}
                start, end = depths_from_sample_ids(
                    [hole] * len(ids), ids, 1.0, max_depth=depth, report=found
                )
                np.testing.assert_array_equal(end, want)
                np.testing.assert_array_equal(start, want - 1.0)
                repeats = ids[0] != plain_ids(hole, metres)[0]
                assert found["repeated_hole_number"] == ([hole] if repeats else [])


def test_the_plain_reading_stays_where_it_is_plausible():
    # hole 4, sampled from 40 to 49 m: every number begins with 4, and 40 is not "4, 0"
    ids = plain_ids("SYN-04", range(40, 50))
    _, end = depths_from_sample_ids(["SYN-04"] * 10, [i.replace("-0", "-") for i in ids])
    np.testing.assert_array_equal(end, np.arange(40.0, 50.0))
    # hole 1, metres 100 to 199 (three digits, no padding to see): 1000 m would be needed
    ids = [f"SYN01-{m}" for m in range(100, 200)]
    _, end = depths_from_sample_ids(["SYN-01"] * 100, ids)
    np.testing.assert_array_equal(end, np.arange(100.0, 200.0))
    # ... and with the depth of the hole known, the plain reading that fits it is taken
    _, end = depths_from_sample_ids(["SYN-01"] * 100, ids, max_depth={"SYN-01": 200.0})
    np.testing.assert_array_equal(end, np.arange(100.0, 200.0))
    # two samples and no depth to go by: too little to reinterpret
    _, end = depths_from_sample_ids(["SYN-04"] * 2, ["SYN04-4001", "SYN04-4002"])
    np.testing.assert_array_equal(end, [4001.0, 4002.0])
    # the hole number written with its zero: 04 002
    found: dict = {}
    _, end = depths_from_sample_ids(
        ["SYN-04"] * 3, ["SYN04-04001", "SYN04-04002", "SYN04-04003"], report=found
    )
    np.testing.assert_array_equal(end, [1.0, 2.0, 3.0])
    assert found["repeated_hole_number"] == ["SYN-04"]
    # an explicit from-to pair is never touched
    start, end = depths_from_sample_ids(["SYN-04"], ["SYN04-4001-4002"])
    assert (start[0], end[0]) == (4001.0, 4002.0)


@pytest.fixture
def files(tmp_path):
    """Six holes of 250 m: holes 1-3 with plain ids, 4-5 with the hole number repeated,
    hole 6 with laboratory numbers that mean nothing as depths. No from / to columns."""
    rng = np.random.default_rng(13)
    holes = [f"SYN-{i:02d}" for i in range(1, 7)]
    collar = pd.DataFrame(
        {"Hole": holes, "East": 500000.0 + 50 * np.arange(6), "North": 5300000.0, "RL": 1200.0,
         "Depth": float(DEPTH), "Azimuth": 0.0, "Dip": -90.0}  # fmt: skip
    )
    collar.to_csv(tmp_path / "collar.csv", index=False)
    metres = list(range(1, DEPTH + 1))
    rows = []
    for i, hole in enumerate(holes, start=1):
        if i <= 3:
            ids = plain_ids(hole, metres)
        elif i <= 5:
            ids = repeated_ids(hole, metres)
        else:
            ids = [f"SYN06-{70000 + m}" for m in metres]
        rows += [{"Hole": hole, "Sample No": s, "From": "", "To": "",
                  "Au": round(float(rng.lognormal(-1, 1)), 3)} for s in ids]  # fmt: skip
    assay = pd.DataFrame(rows)
    assay.to_csv(tmp_path / "assay.csv", index=False)
    return tmp_path, holes, assay


def recipe(**rule):
    return make_recipe(
        {"collar": [{"file": "collar.csv"}],
         "assay": [{"file": "assay.csv", "sample_depth": {"length": 1.0, **rule}}]}  # fmt: skip
    )


def test_import_reads_both_styles_and_refuses_depths_below_the_hole(files, capsys):
    folder, holes, assay = files
    db = run_recipe(recipe(), folder)
    table = db.intervals["assay"]
    good = holes[:5]
    assert table["hole_id"].unique().tolist() == good and len(table) == 5 * DEPTH
    for hole in good:  # 0-1, 1-2 ... 249-250, whatever the id style
        part = table[table["hole_id"] == hole]
        np.testing.assert_array_equal(part["depth_to"], np.arange(1.0, DEPTH + 1))
        np.testing.assert_array_equal(part["depth_from"], np.arange(0.0, DEPTH))
    assert table["depth_to"].max() == DEPTH  # no hole 4 km deep
    np.testing.assert_array_equal(table["au"], assay["Au"].to_numpy()[: 5 * DEPTH])
    (check,) = db.metadata.reports["sample_depth"]["assay"]
    assert check["repeated_hole_number"] == ["SYN-04", "SYN-05"]
    assert check["beyond_hole"] == [
        {"hole_id": "SYN-06", "derived_to": 70250.0, "max_depth": 250.0, "rows": DEPTH,
         "example": "SYN06-70250"}
    ]  # fmt: skip
    assert check["beyond_hole_rows"] == DEPTH and check["holes_without_known_depth"] == 0
    assert (check["backwards"], check["gaps"], check["overlaps"]) == (0, 0, 0)
    # the rows of hole 6 are kept apart, not lost and not imported
    qc = db.qc["assay"]
    assert set(qc["qc_type"]) == {"depth_beyond_hole"} and len(qc) == DEPTH
    assert set(qc["hole_id"]) == {"SYN-06"}
    kinds = {n["kind"]: n for n in db.metadata.notes if n["table"] == "assay"}
    assert kinds["sample_depths_beyond_hole_not_imported"]["examples"] == ["SYN-06: 70250 > 250"]
    assert kinds["sample_number_repeats_hole_number"]["examples"] == ["SYN-04", "SYN-05"]

    out = folder / "db"
    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "assay.csv"), "--auto", "--sample-depth", "assay",
                 "--out", str(out)])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert "the sample numbers repeat the hole number" in printed and "SYN-04, SYN-05" in printed
    assert ("WARNING: assay SYN-06: the sample numbers give depths to 70250 but the hole is "
            "250 deep (e.g. SYN06-70250): its 250 rows were NOT imported") in printed  # fmt: skip
    assert load_database(out).intervals["assay"]["depth_to"].max() == DEPTH
    # a pattern for the odd ids brings hole 6 in
    patterns = [r"SYN06-70(?P<to>\d+)", r"SYN0*{holenum}-{holenum}?(?P<to>\d{3})"]
    full = run_recipe(recipe(patterns=patterns), folder)
    assert len(full.intervals["assay"]) == 6 * DEPTH and "assay" not in full.qc


def test_tolerance_and_holes_of_unknown_depth(files):
    folder, holes, assay = files
    collar = pd.read_csv(folder / "collar.csv")
    collar.loc[0, "Depth"] = 245.0  # logged a little shorter than it was sampled
    collar.loc[1, "Depth"] = 200.0  # far too short for its samples
    collar.loc[2, "Depth"] = np.nan  # not known
    collar.to_csv(folder / "collar.csv", index=False)
    db = run_recipe(recipe(), folder)
    (check,) = db.metadata.reports["sample_depth"]["assay"]
    assert [b["hole_id"] for b in check["beyond_hole"]] == ["SYN-02", "SYN-06"]
    assert check["holes_without_known_depth"] == 1
    assert set(db.intervals["assay"]["hole_id"]) == {"SYN-01", "SYN-03", "SYN-04", "SYN-05"}
    # the hole of unknown depth takes it from its samples, as before
    depth = db.collar.set_index("hole_id")["max_depth"]
    assert depth["SYN-03"] == DEPTH and depth["SYN-01"] == 245.0
    wide = run_recipe(recipe(depth_tolerance=60.0), folder)
    (check,) = wide.metadata.reports["sample_depth"]["assay"]
    assert [b["hole_id"] for b in check["beyond_hole"]] == ["SYN-06"]
    tight = run_recipe(recipe(depth_tolerance=1.0), folder)
    (check,) = tight.metadata.reports["sample_depth"]["assay"]
    assert [b["hole_id"] for b in check["beyond_hole"]] == ["SYN-01", "SYN-02", "SYN-06"]


def test_depths_going_backwards_are_warned_of(files, capsys):
    folder, holes, assay = files
    part = assay[assay["Hole"] == "SYN-01"].copy()
    swapped = part.iloc[[0, 1, 5, 3, 4, 2, *range(6, 20)]]
    swapped.to_csv(folder / "assay.csv", index=False)
    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "assay.csv"), "--auto", "--sample-depth", "assay",
                 "--dry-run"])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert "WARNING: assay: the depths read from sample numbers go backwards" in printed
