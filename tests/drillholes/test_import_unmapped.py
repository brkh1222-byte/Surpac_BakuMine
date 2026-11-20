"""``keep_unmapped: false``: extra columns of a source do not reach the database."""

from __future__ import annotations

import pandas as pd
import pytest

from bakumine.core.drillholes.recipe import make_recipe, run_recipe


@pytest.fixture
def files(tmp_path):
    collar = pd.DataFrame(
        {"Hole": ["SYN-01", "SYN-02"], "East": [500100.0, 500200.0],
         "North": [5300100.0, 5300200.0], "RL": [1200.0, 1201.0], "Azimuth": [0, 0],
         "Dip": [-90, -90], "Drilled by": ["rig 1", "rig 2"]}  # fmt: skip
    )
    collar.to_csv(tmp_path / "collar.csv", index=False)
    with pd.ExcelWriter(tmp_path / "assays.xlsx") as writer:
        for hole in ("SYN-01", "SYN-02"):
            sheet = pd.DataFrame(
                {"No": [1, 2, 3], "From": [0, 1, 2], "To": [1, 2, 3], "Sample No": ["a", "b", "c"],
                 "Au, g/t": [0.1, 0.2, 0.3], "Ag (г/т)": [1.0, 2.0, 3.0],
                 "Batch": ["B1", "B1", "B2"], "Remark": ["", "vein", ""]}  # fmt: skip
            )
            sheet.to_excel(writer, sheet_name=hole, index=False)
    return tmp_path


def recipe(keep: bool, **assay):
    source = {"file": "assays.xlsx", "sheets": "SYN*", "hole_id_from_sheet": True, **assay}
    return make_recipe(
        {"collar": [{"file": "collar.csv"}], "assay": [source]}, {"keep_unmapped": keep}
    )


def test_suggested_map_keeps_known_and_element_columns_only(files):
    kept = run_recipe(recipe(True), files)
    assert list(kept.intervals["assay"].columns) == [
        "hole_id", "depth_from", "depth_to", "no", "sample_id", "au", "ag", "batch", "remark"
    ]  # fmt: skip
    assert "Drilled by" in kept.collar.columns
    db = run_recipe(recipe(False), files)
    assert list(db.intervals["assay"].columns) == [
        "hole_id", "depth_from", "depth_to", "sample_id", "au", "ag"
    ]
    assert len(db.intervals["assay"]) == 6
    assert "Drilled by" not in db.collar.columns
    dropped = [n for n in db.metadata.notes if n["kind"] == "unrecognised_columns_dropped"]
    assert [(n["table"], n["count"], n["examples"]) for n in dropped] == [
        ("assay", 3, ["No", "Batch", "Remark"])
    ]
    # the recipe written back names what was kept, and replays the same
    assert set(db.metadata.recipe["tables"]["assay"][0]["map"]) == {
        "From", "To", "Sample No", "Au, g/t", "Ag (г/т)"
    }
    again = run_recipe(db.metadata.recipe, files)
    assert list(again.intervals["assay"].columns) == list(db.intervals["assay"].columns)


def test_given_map_decides(files):
    given = {"From": "depth_from", "To": "depth_to", "Au, g/t": "gold", "Batch": "batch"}
    db = run_recipe(recipe(False, map=given), files)
    assert list(db.intervals["assay"].columns) == [
        "hole_id", "depth_from", "depth_to", "gold", "batch"
    ]
    kept = run_recipe(recipe(True, map=given), files)
    assert {"No", "Sample No", "Ag (г/т)", "Remark"} <= set(kept.intervals["assay"].columns)
