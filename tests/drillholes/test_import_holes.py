"""Hole filters: import only some holes (names, wildcard patterns, a list file)."""

from __future__ import annotations

import json

import pytest
from messy_fixtures import CLEAN_MAPS, OPTIONS, clean_fixture, recipe_for, tables_fixture

from bakumine.cli import main
from bakumine.core.drillholes.importer import load_database
from bakumine.core.drillholes.recipe import HoleFilter, load_recipe, make_recipe, run_recipe

tables = tables_fixture()
clean = clean_fixture()


def assay_source(paths):
    return {"file": str(paths["assay"]), "map": CLEAN_MAPS["assay"]}


def rows_of(table, holes):
    return int(table["hole_id"].isin(holes).sum())


def test_include_and_exclude_apply_to_every_table(clean):
    paths, clean_db = clean
    recipe = recipe_for(
        paths, assay_source(paths),
        holes={"include": ["syn-0[1-4]", "SYN-06", "NOPE-*"], "exclude": ["SYN-03"]},
    )  # fmt: skip
    assert recipe["holes"] == {"include": ["syn-0[1-4]", "SYN-06", "NOPE-*"],
                               "exclude": ["SYN-03"]}  # fmt: skip
    db = run_recipe(recipe)
    kept = ["SYN-01", "SYN-02", "SYN-04", "SYN-06"]
    assert db.collar["hole_id"].tolist() == kept
    assert db.survey["hole_id"].tolist() == kept
    assay, all_assays = db.intervals["assay"], clean_db.intervals["assay"]
    assert set(assay["hole_id"]) == set(kept)
    assert len(assay) == rows_of(all_assays, kept)
    report = db.metadata.reports["hole_filter"]
    assert report["tables"] == {
        "collar": {"rows_kept": 4, "rows_dropped": 2, "holes_dropped": 2},
        "survey": {"rows_kept": 4, "rows_dropped": 2, "holes_dropped": 2},
        "assay": {"rows_kept": len(assay), "rows_dropped": len(all_assays) - len(assay),
                  "holes_dropped": 2},
    }  # fmt: skip
    assert report["unmatched"] == ["NOPE-*"]  # a pattern that matched nothing is said
    assert any(n["kind"] == "hole_filter_matched_nothing" for n in db.metadata.notes)
    assert db.metadata.recipe["holes"] == recipe["holes"]  # stored for replay
    # exclusion alone; the values of the kept rows are untouched
    only = run_recipe(recipe_for(paths, assay_source(paths), holes={"exclude": "SYN-0[12]"}))
    assert only.collar["hole_id"].tolist() == ["SYN-03", "SYN-04", "SYN-05", "SYN-06"]
    left = all_assays[~all_assays["hole_id"].isin(["SYN-01", "SYN-02"])].reset_index(drop=True)
    assert only.intervals["assay"]["ag"].equals(left["ag"])


def test_filter_sees_the_rewritten_hole_ids_and_the_qc_rows(tmp_path, tables, clean):
    from messy_fixtures import HOLE_ID, write_sheet_per_hole

    paths, _ = clean
    book = tmp_path / "assays.xlsx"
    write_sheet_per_hole(book, tables["assay"], depths="sample", qc=True)
    source = {"file": str(book), "sheets": "*", "hole_id_from_sheet": True,
              "sample_depth": {"length": 1.0}}  # fmt: skip
    everything = run_recipe(recipe_for(paths, source))
    # QC rows of every hole (and of a "Summary" sheet that is no hole at all)
    assert set(everything.qc["assay"]["hole_id"]) > set(tables["collar"]["hole_id"])
    # the sheets are SYN_01, SYN-002 ...: the filter names the hole as it is written out
    db = run_recipe(recipe_for(paths, source, holes={"include": ["SYN-02", "SYN-05"]}))
    assert HOLE_ID["template"].startswith("SYN-")
    assert set(db.intervals["assay"]["hole_id"]) == {"SYN-02", "SYN-05"}
    assert set(db.qc["assay"]["hole_id"]) <= {"SYN-02", "SYN-05"}
    qc_all = everything.qc["assay"]
    assert len(db.qc["assay"]) == rows_of(qc_all, ["SYN-02", "SYN-05"])
    assert db.metadata.reports["hole_filter"]["tables"]["assay (QC)"]["rows_kept"] == len(
        db.qc["assay"]
    )


def test_cli_filters_by_names_patterns_and_a_list_file(tmp_path, clean, capsys):
    paths, _ = clean
    maps = tmp_path / "maps.json"
    maps.write_text(json.dumps(CLEAN_MAPS), encoding="utf-8")
    listed = tmp_path / "holes.txt"
    listed.write_text("SYN-01\n# the twin of 01\nSYN-02  # drilled later\n\nSYN-05\n",
                      encoding="utf-8")  # fmt: skip
    common = ["import", "--collar", str(paths["collar"]), "--survey", str(paths["survey"]),
              "--assay", str(paths["assay"]), "--map", str(maps), "--delimiter", ";",
              "--dip-convention", "auto"]  # fmt: skip
    out = tmp_path / "db"
    code = main([*common, "--include-holes", str(listed), "--include-holes", "SYN-06,ZZ*",
                 "--exclude-holes", "syn-02", "--out", str(out)])  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    db = load_database(out)
    assert db.collar["hole_id"].tolist() == ["SYN-01", "SYN-05", "SYN-06"]
    assert "hole filter (include SYN-01, SYN-02, SYN-05, SYN-06, ZZ*; exclude syn-02): " in printed
    assert "kept collar 3 of 6 rows, survey 3 of 6 rows, assay " in printed
    assert "WARNING: hole filter matched no hole: ZZ*" in printed
    assert "holes 3," in printed
    recipe = load_recipe(out / "import_recipe.json")
    assert recipe["holes"] == {"include": ["SYN-01", "SYN-02", "SYN-05", "SYN-06", "ZZ*"],
                               "exclude": ["syn-02"]}  # fmt: skip
    # the recipe replays the filter; a filter given with --recipe replaces it
    again = tmp_path / "again"
    assert main(["import", "--recipe", str(out / "import_recipe.json"), "--out", str(again)]) == 0
    assert load_database(again).collar["hole_id"].tolist() == ["SYN-01", "SYN-05", "SYN-06"]
    other = tmp_path / "other"
    assert main(["import", "--recipe", str(out / "import_recipe.json"), "--include-holes",
                 "SYN-04", "--out", str(other)]) == 0  # fmt: skip
    assert load_database(other).collar["hole_id"].tolist() == ["SYN-04"]
    capsys.readouterr()
    assert main([*common, "--include-holes", " , ", "--dry-run"]) == 2
    assert "--include-holes ' , ' names no hole" in capsys.readouterr().err


def test_filter_documents_are_checked(clean):
    paths, _ = clean
    collar = {"collar": {"file": str(paths["collar"])}}
    assert "holes" not in make_recipe(collar, OPTIONS)  # old recipes have none
    assert "holes" not in make_recipe(collar, OPTIONS, holes={"include": [], "exclude": [" "]})
    with pytest.raises(ValueError, match="unknown holes key"):
        make_recipe(collar, OPTIONS, holes={"only": ["A"]})
    with pytest.raises(ValueError, match="holes must be"):
        make_recipe(collar, OPTIONS, holes=["A"])
    assert not HoleFilter() and HoleFilter(exclude=["A"])
    # everything filtered away: the import says so instead of writing an empty database
    with pytest.raises(ValueError, match="no hole"):
        run_recipe(make_recipe(
            {"collar": {"file": str(paths["collar"]), "map": CLEAN_MAPS["collar"]},
             "survey": {"file": str(paths["survey"]), "map": CLEAN_MAPS["survey"]}},
            OPTIONS, holes={"include": ["NOPE"]},
        ))  # fmt: skip
