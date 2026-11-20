"""Import recipes: saved with the database, replayed by ``bakumine import --recipe``;
and the CLI options for the real-world shapes.

The fixtures are synthetic (``messy_fixtures``); the reference is always
``import_drillholes`` on three plain CSV files, which reads them with the plain reader
(``read_raw_table``) and knows nothing of sheets, header rows, rules or recipes.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest
from messy_fixtures import (
    HOLE_ID,
    N_HOLES,
    OPTIONS,
    UTM,
    clean_fixture,
    hole_name,
    same_assays,
    tables_fixture,
    write_collar_dms,
    write_sheet_per_hole,
)

from bakumine.cli import main
from bakumine.core.drillholes.importer import (
    load_database,
)
from bakumine.core.drillholes.recipe import (
    MissingColumns,
    load_recipe,
    make_recipe,
    run_recipe,
)

tables = tables_fixture()
clean = clean_fixture()


def run_cli(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def test_cli_imports_the_shapes_and_replays_the_recipe(tmp_path, tables, clean, capsys):
    paths, clean_db = clean
    assays, collars = tmp_path / "assays.xlsx", tmp_path / "collars.xlsx"
    counts = write_sheet_per_hole(assays, tables["assay"], depths="sample", qc=True)
    write_collar_dms(collars, tables["collar"], tables["survey"], shifted={hole_name(4): 120.0})
    db = tmp_path / "db"
    code, out, err = run_cli(
        capsys, "import", "--collar", collars, "--assay", assays, "--out", db, "--auto",
        "--sheet", "assay=SYN*", "--hole-from-sheet", "assay", "--sample-depth", "assay",
        "--hole-id-pattern", HOLE_ID["pattern"], "--hole-id-template", HOLE_ID["template"],
        "--dip-convention", "auto", "--coordinate-order", "auto",
        "--save-recipe", tmp_path / "recipe.json",
    )  # fmt: skip
    assert code == 0, err
    assert f"assay: 1 file(s), {N_HOLES} sheet(s) stacked" in out
    assert f"assay QC rows set aside ({sum(counts.values())})" in out
    assert "depths from sample numbers" in out and "not understood 0" in out
    assert f"collar longitude / latitude -> {UTM}" in out and "1 differ by more than 30" in out
    assert f"survey made from the collar table: {N_HOLES} holes" in out
    assert "dip convention positive_down, coordinate order yxz" in out
    assert "below_detection" in out and len(out.splitlines()) < 45
    loaded = load_database(db)
    same_assays(loaded, clean_db)
    pd.testing.assert_frame_equal(loaded.survey, clean_db.survey)
    assert len(loaded.qc["assay"]) == sum(counts.values())

    # the recipe is saved with the database, and replays on (updated) files elsewhere
    saved = json.loads((db / "import_recipe.json").read_text())
    assert saved == json.loads((tmp_path / "recipe.json").read_text()) == loaded.metadata.recipe
    assert saved["tables"]["assay"][0]["sample_depth"] == {"length": 1.0}
    assert saved["tables"]["assay"][0]["map"]["Au"] == "au"
    moved = tmp_path / "next"
    moved.mkdir()
    update = tables["assay"].iloc[:-3]  # the updated export has fewer rows
    write_sheet_per_hole(moved / "assays.xlsx", update, depths="sample", qc=True)
    write_collar_dms(moved / "collars.xlsx", tables["collar"], tables["survey"])
    recipe = load_recipe(db / "import_recipe.json")
    for sources in recipe["tables"].values():
        for source in sources:
            source["file"] = Path(source["file"]).name
    (moved / "recipe.json").write_text(json.dumps(recipe))
    code, out, err = run_cli(
        capsys, "import", "--recipe", moved / "recipe.json", "--out", tmp_path / "db2"
    )
    assert code == 0, err
    again = load_database(tmp_path / "db2")
    assert len(again.intervals["assay"]) == len(clean_db.intervals["assay"]) - 3
    pd.testing.assert_frame_equal(again.collar[["hole_id", "x", "y"]],
                                  clean_db.collar[["hole_id", "x", "y"]])  # fmt: skip
    assert again.metadata.reports["coordinates"]["disagree"] == 0
    # --base points a recipe at another folder; --dry-run writes nothing
    code, out, err = run_cli(
        capsys, "import", "--recipe", moved / "recipe.json", "--base", tmp_path, "--dry-run"
    )
    assert code == 0 and "dry run: no database written" in out
    assert len(load_database(tmp_path / "db2").intervals["assay"]) == len(again.intervals["assay"])


def test_cli_merges_files_and_reports_usage_errors(tmp_path, tables, clean, capsys):
    paths, clean_db = clean
    assay = tables["assay"]
    half = len(assay) // 2
    first, second = tmp_path / "a.csv", tmp_path / "b.csv"
    assay.iloc[: half + 4].to_csv(first, index=False, sep=";")
    assay.iloc[half:].to_csv(second, index=False, sep=";")
    base = ["import", "--collar", paths["collar"], "--survey", paths["survey"], "--delimiter", ";",
            "--dip-convention", "auto"]  # fmt: skip
    code, out, err = run_cli(
        capsys, *base, "--assay", first, "--assay", second, "--auto", "--out", tmp_path / "db"
    )
    assert code == 0, err
    assert "assay sources merged (first file wins): 4 rows joined, 0 cells filled" in out
    same_assays(load_database(tmp_path / "db"), clean_db)
    code, _, err = run_cli(capsys, *base, "--auto")
    assert code == 2 and "--out" in err
    code, _, err = run_cli(capsys, "import", "--recipe", tmp_path / "none.json", "--out", "x")
    assert code == 2 and "recipe not found" in err
    code, _, err = run_cli(
        capsys, "import", "--recipe", first, "--collar", first, "--out", tmp_path / "x"
    )
    assert code == 2 and "--recipe replaces" in err
    (tmp_path / "bad.json").write_text('{"format": "something else"}')
    code, _, err = run_cli(capsys, "import", "--recipe", tmp_path / "bad.json", "--out", "x")
    assert code == 2 and "not a bakumine-import-recipe" in err
    code, _, err = run_cli(capsys, *base, "--auto", "--sheet", "geology=1", "--out", "x")
    assert code == 2 and "tables that were not given" in err
    assert not (tmp_path / "x").exists()


def test_recipe_documents_are_checked(tmp_path, clean):
    paths, _ = clean
    with pytest.raises(ValueError, match="needs a collar"):
        make_recipe({"assay": {"file": "a.csv"}})
    with pytest.raises(ValueError, match="unknown source key"):
        make_recipe({"collar": {"file": "a.csv", "sheet": "x"}})
    with pytest.raises(ValueError, match="every source needs a 'file'"):
        make_recipe({"collar": {"sheets": "x"}})
    with pytest.raises(ValueError, match="unknown import option"):
        make_recipe({"collar": {"file": "a.csv"}}, {"lenght_unit": "m"})
    with pytest.raises(ValueError, match="precedence"):
        make_recipe({"collar": {"file": "a.csv"}}, merge={"precedence": "newest"})
    with pytest.raises(FileNotFoundError, match="source file not found"):
        run_recipe(make_recipe({"collar": {"file": "nowhere.csv"}}), tmp_path)
    headerless = tmp_path / "c.csv"
    headerless.write_text("a;b;c\n1;2;3\n")
    with pytest.raises(MissingColumns) as error:
        run_recipe(make_recipe({"collar": {"file": str(headerless)}}, OPTIONS))
    assert error.value.table == "collar" and "hole_id" in error.value.missing
