"""Merge conflicts: counted per column, listed in full beside the database, and columns
whose differences are only informational (sample ids)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import make_recipe, merge_sources, run_recipe

HOLES = [f"SYN-{i:02d}" for i in range(1, 7)]
PER_HOLE = 60


@pytest.fixture
def files(tmp_path):
    """Two laboratory files of the same 360 samples: the sample ids are spelled
    differently in every row, three gold values and one silver value really differ."""
    rng = np.random.default_rng(4)
    collar = pd.DataFrame(
        {"Hole": HOLES, "East": 500000.0 + np.arange(6) * 50, "North": 5300000.0, "RL": 1200.0,
         "Depth": float(PER_HOLE), "Azimuth": 0.0, "Dip": -90.0}  # fmt: skip
    )
    collar.to_csv(tmp_path / "collar.csv", index=False)
    rows = [
        {"Hole": h, "From": float(k), "To": float(k + 1), "Sample": f"{h}-{k + 1:03d}",
         "Au": round(float(rng.lognormal(-1, 1)), 3), "Ag": round(float(rng.uniform(1, 9)), 2)}
        for h in HOLES for k in range(PER_HOLE)
    ]  # fmt: skip
    first = pd.DataFrame(rows)
    second = first.copy()
    second["Sample"] = [f"L{i:05d}" for i in range(len(second))]
    second.loc[[5, 100, 250], "Au"] += 1.0
    second.loc[300, "Ag"] += 2.0
    second.loc[10, "Au"] += 0.004  # inside the tolerance: no conflict
    first.to_csv(tmp_path / "lab_a.csv", index=False)
    second.to_csv(tmp_path / "lab_b.csv", index=False)
    return tmp_path, first, second


def recipe(**merge):
    return make_recipe(
        {"collar": [{"file": "collar.csv"}],
         "assay": [{"file": "lab_a.csv"}, {"file": "lab_b.csv"}]},  # fmt: skip
        merge=merge or None,
    )


def test_sample_ids_are_informational_and_real_conflicts_stand_out(files):
    folder, first, second = files
    assert recipe()["merge"]["ignore_columns"] == ["sample_id"]  # the default
    db = run_recipe(recipe(), folder)
    report = db.metadata.reports["conflicts"]["assay"]
    n = len(first)
    assert (report["merged"], report["conflicts"]) == (n, 4)
    assert report["by_column"] == {"au": 3, "ag": 1}  # most first
    assert report["informational"] == {"sample_id": n}
    assert [r["column"] for r in report["rows"]] == ["au", "au", "au", "ag"]  # ids not among them
    assert report["listing"] == "conflicts_assay.csv"
    # the first file's values (ids included) are kept
    assert db.intervals["assay"]["sample_id"].tolist() == first["Sample"].tolist()
    np.testing.assert_array_equal(db.intervals["assay"]["au"], first["Au"])
    # the full listing: every difference, flagged
    listing = db.listings["conflicts_assay"]
    assert len(listing) == n + 4
    assert int(listing["informational"].sum()) == n
    real = listing[~listing["informational"]]
    assert real["column"].tolist() == ["au", "au", "au", "ag"]
    np.testing.assert_allclose(real["dropped"].astype(float) - real["kept"].astype(float),
                               [1.0, 1.0, 1.0, 2.0])  # fmt: skip
    assert set(real["dropped_source"]) == {"lab_b.csv"}
    note = next(n for n in db.metadata.notes if n["kind"] == "sources_conflicts")
    assert note["count"] == 4 and note["values"] == {"au": 3, "ag": 1}


def test_listing_is_not_capped_and_is_saved_with_the_database(files, capsys):
    folder, first, _ = files
    out = folder / "db"
    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "lab_a.csv"), "--assay", str(folder / "lab_b.csv"), "--auto",
                 "--merge-ignore", "none", "--out", str(out)])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    n = len(first)
    assert f"conflicts by column: sample_id {n}, au 3, ag 1" in printed
    assert "every difference: conflicts_assay.csv in the database folder" in printed
    assert f"conflicts_assay.csv ({n + 4} rows, the full listing)" in printed
    saved = pd.read_csv(out / "conflicts_assay.csv")
    assert len(saved) == n + 4 > 200  # database.json keeps the first 200 only
    assert list(saved.columns) == [
        "hole_id", "depth_from", "depth_to", "column", "kept", "dropped", "kept_source",
        "dropped_source", "informational",
    ]  # fmt: skip
    db = load_database(out)
    assert len(db.metadata.reports["conflicts"]["assay"]["rows"]) == 200
    assert db.metadata.reports["conflicts"]["assay"]["conflicts"] == n + 4
    assert len(db.listings["conflicts_assay"]) == n + 4  # read back with the database
    assert db.metadata.recipe["merge"]["ignore_columns"] == []

    # the default, and several columns named
    code = main(["import", "--collar", str(folder / "collar.csv"), "--assay",
                 str(folder / "lab_a.csv"), "--assay", str(folder / "lab_b.csv"), "--auto",
                 "--merge-ignore", "sample_id,ag", "--dry-run"])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    assert "360 rows joined, 0 cells filled, 3 conflicts" in printed
    assert "conflicts by column: au 3" in printed
    assert f"(informational): sample_id {n}, ag 1" in printed


def test_merge_sources_counts_and_lists():
    table = pd.DataFrame(
        {"hole_id": ["A", "A", "A", "A"], "depth_from": [0.0, 1.0, 0.0, 1.0],
         "depth_to": [1.0, 2.0, 1.0, 2.0], "sample_id": ["a1", "a2", "x1", "x2"],
         "au": [1.0, 2.0, 1.5, 2.0]}  # fmt: skip
    )
    source = np.array([0, 0, 1, 1])
    listing: list = []
    merged, report = merge_sources(
        table, source, ("hole_id", "depth_from", "depth_to"), ignore=["sample_id"],
        listing=listing,
    )  # fmt: skip
    assert len(merged) == 2 and report["conflicts"] == 1
    assert report["by_column"] == {"au": 1} and report["informational"] == {"sample_id": 2}
    assert [(r["column"], r["informational"]) for r in listing] == [
        ("sample_id", True), ("au", False), ("sample_id", True)
    ]
    # nothing ignored (as before): every difference is a conflict
    _, plain = merge_sources(table, source, ("hole_id", "depth_from", "depth_to"))
    assert plain["conflicts"] == 3 and plain["informational"] == {}


def test_merge_settings_are_checked():
    with pytest.raises(ValueError, match="unknown merge key"):
        recipe(ignore=["sample_id"])
    assert recipe(ignore_columns="au, ")["merge"]["ignore_columns"] == ["au,"]
    assert recipe(ignore_columns=["au", " ", "au"])["merge"]["ignore_columns"] == ["au"]
    # a recipe written before the key existed
    old = recipe()
    del old["merge"]["ignore_columns"]
    assert make_recipe(old["tables"], merge=old["merge"])["merge"]["ignore_columns"] == [
        "sample_id"
    ]
