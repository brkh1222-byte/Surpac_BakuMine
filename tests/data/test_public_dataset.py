"""Integrity checks for the public drillhole dataset in data/public/babbitt."""

import hashlib
from pathlib import Path

import pandas as pd
import pytest

DATASET = Path(__file__).resolve().parents[2] / "data" / "public" / "babbitt"

RAW_SHA256 = {
    "collar_BABBITT.csv": "2dbce5d751c283d0d71d9fc2afbebace62d74d6e3db2c92af9cbcac7b7f62c32",
    "survey_BABBITT.csv": "5f41cfaf7adf4cdd7f9652e2db6d53520c5565489123ff6cec0fc0cf6a300c09",
    "assay_BABBITT.csv": "121956eb0e158af5d6c13e66ea1d57b66bee36e9577c20d80b624b1a88140f2f",
}

# the domain wireframe of the PyGSLIB tutorial (binary STL, feet); see SOURCE.md
WIREFRAME_SHA256 = {
    "Mpz.stl": "a324327f6dd4802ad95c6570cce8488ea397311459a48ea2a75c20f6cd5e867b",
}

COLUMNS = {
    "collar": ["hole_id", "easting", "northing", "elevation", "depth"],
    "survey": ["hole_id", "depth", "dip", "azimuth"],
    "assay": ["hole_id", "from", "to", "cu", "ni", "s", "fe"],
}
ROWS = {"collar": 399, "survey": 2628, "assay": 35616}


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(DATASET / f"{name}.csv", dtype={"hole_id": str})


def all_csvs() -> list[Path]:
    return sorted(DATASET.rglob("*.csv"))


def test_licence_and_source_files_exist():
    licence = DATASET / "LICENSE"
    assert licence.is_file()
    assert "MIT License" in licence.read_text(encoding="utf-8")
    assert (DATASET / "SOURCE.md").is_file()


def test_dataset_is_small():
    total = sum(p.stat().st_size for p in DATASET.rglob("*") if p.is_file())
    assert total < 10 * 1024 * 1024


def test_csv_inventory():
    names = {p.relative_to(DATASET).as_posix() for p in all_csvs()}
    assert names == {f"{n}.csv" for n in COLUMNS} | {f"raw/{n}" for n in RAW_SHA256}


@pytest.mark.parametrize("path", all_csvs(), ids=lambda p: p.relative_to(DATASET).as_posix())
def test_every_csv_loads(path: Path):
    frame = pd.read_csv(path)
    assert len(frame) > 0
    assert len(frame.columns) >= 4


@pytest.mark.parametrize("name", sorted(RAW_SHA256 | WIREFRAME_SHA256))
def test_raw_files_are_unmodified(name: str):
    digest = hashlib.sha256((DATASET / "raw" / name).read_bytes()).hexdigest()
    assert digest == (RAW_SHA256 | WIREFRAME_SHA256)[name]


def test_raw_inventory():
    names = {p.name for p in (DATASET / "raw").iterdir()}
    assert names == set(RAW_SHA256) | set(WIREFRAME_SHA256)


@pytest.mark.parametrize("name", sorted(COLUMNS))
def test_columns_and_row_counts(name: str):
    frame = load(name)
    assert list(frame.columns) == COLUMNS[name]
    assert len(frame) == ROWS[name]
    assert frame["hole_id"].notna().all()


def test_collar_is_complete_and_unique():
    collar = load("collar")
    assert collar["hole_id"].is_unique
    assert collar.notna().all().all()
    assert (collar["depth"] > 0).all()


@pytest.mark.parametrize("name", ["survey", "assay"])
def test_hole_ids_exist_in_collar(name: str):
    collar_ids = set(load("collar")["hole_id"])
    missing = set(load(name)["hole_id"]) - collar_ids
    assert not missing


def test_assay_intervals():
    assay = load("assay")
    assert assay[["from", "to"]].notna().all().all()
    assert (assay["from"] >= 0).all()
    assert (assay["from"] < assay["to"]).all()
    # At least one grade column carries real values.
    assert assay["cu"].notna().sum() > 20000
    assert (assay[["cu", "ni", "s", "fe"]].min() >= 0).all()


def test_assay_intervals_do_not_overlap_and_match_collar_depth():
    assay = load("assay").sort_values(["hole_id", "from"], kind="stable")
    previous_to = assay.groupby("hole_id")["to"].shift()
    assert (assay["from"] >= previous_to).loc[previous_to.notna()].all()
    max_to = assay.groupby("hole_id")["to"].max()
    depth = load("collar").set_index("hole_id")["depth"]
    assert (max_to == depth.loc[max_to.index]).all()


def test_survey_convention_is_positive_down():
    """SOURCE.md documents dips as positive downward; fail loudly if that ever changes."""
    survey = load("survey")
    assert survey.notna().all().all()
    assert survey["dip"].between(0, 90).all()
    assert (survey["dip"] > 0).all()
    assert survey["azimuth"].between(0, 360, inclusive="left").all()
    assert (survey["depth"] >= 0).all()
    # Every hole has a station at the collar.
    assert (survey.groupby("hole_id")["depth"].min() == 0).all()
