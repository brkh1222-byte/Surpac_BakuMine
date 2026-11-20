"""Workbooks with one sheet per hole and headers that are not one clean first row
import to the same database as the clean CSVs of the same data.

The fixtures are synthetic (``messy_fixtures``); the reference is always
``import_drillholes`` on three plain CSV files, which reads them with the plain reader
(``read_raw_table``) and knows nothing of sheets, header rows, rules or recipes.
"""

from __future__ import annotations

import pandas as pd
import pytest
from messy_fixtures import (
    HOLE_ID,
    N_HOLES,
    clean_fixture,
    hole_name,
    recipe_for,
    same_assays,
    tables_fixture,
    write_sheet_per_hole,
)

from bakumine.core.drillholes.importer import (
    suggest_column_map,
)
from bakumine.core.drillholes.recipe import (
    run_recipe,
)
from bakumine.core.drillholes.tables import (
    locate_header,
    normalise_hole_ids,
    read_grids,
    select_sheets,
    table_from_grid,
)

tables = tables_fixture()
clean = clean_fixture()


@pytest.mark.parametrize("mongolian", [True, False])
def test_one_sheet_per_hole_with_two_row_header_equals_clean(tmp_path, tables, clean, mongolian):
    paths, clean_db = clean
    book = tmp_path / "assays.xlsx"
    write_sheet_per_hole(book, tables["assay"], mongolian=mongolian)
    recipe = recipe_for(
        paths, {"file": str(book), "sheets": "SYN*", "hole_id_from_sheet": True}
    )
    db = run_recipe(recipe)
    same_assays(db, clean_db)
    pd.testing.assert_frame_equal(db.collar, clean_db.collar)
    report = db.metadata.reports["tables"]["assay"]["sources"][0]
    assert [s["name"] for s in report["sheets"]] == [
        n for n in read_grids(book) if n != "Summary"
    ]
    assert {(s["header_row"], s["header_rows"]) for s in report["sheets"]} == {(3, 2)}
    # the column map was found from the (Mongolian or English) sub-headers and is saved
    found = db.metadata.recipe["tables"]["assay"][0]["map"]
    lower = ("аас", "хүртэл") if mongolian else ("from", "to")
    assert found[lower[0]] == "depth_from" and found[lower[1]] == "depth_to"
    assert found["Au"] == "au" and found["Ag"] == "ag"
    assert "hole_id" not in found.values()  # from the sheet name


def test_sheet_selection_and_hole_ids_from_sheet_names(tmp_path, tables):
    book = tmp_path / "assays.xlsx"
    write_sheet_per_hole(book, tables["assay"])
    names = list(read_grids(book))
    assert names[-1] == "Summary" and len(names) == N_HOLES + 1
    assert select_sheets(names) == names[:1]
    assert select_sheets(names, "SYN*") == names[:-1]
    assert select_sheets(names, "*", ["Summary"]) == names[:-1]
    assert select_sheets(names, [names[2], 0]) == [names[0], names[2]]
    with pytest.raises(ValueError, match="no sheet matches"):
        select_sheets(names, "DH*")
    ids, unmatched = normalise_hole_ids(
        pd.Series([*names, " syn_7 "]), HOLE_ID["pattern"], HOLE_ID["template"]
    )
    assert ids.tolist() == [*[hole_name(i) for i in range(1, N_HOLES + 1)], "Summary", "SYN-07"]
    assert unmatched == ["Summary"]
    with pytest.raises(ValueError, match="hole id pattern"):
        normalise_hole_ids(pd.Series(["a"]), "(", None)
    # an optional group that did not match is empty text, not "None"
    ids, _ = normalise_hole_ids(
        pd.Series(["SYN25_7", "SYN-012"]), r"SYN(25)?[_-]?0*(\d+)", "SYN{1}-{2:02d}"
    )
    assert ids.tolist() == ["SYN25-07", "SYN-12"]


def grid(rows: list[list[object]]) -> pd.DataFrame:
    width = max(len(r) for r in rows)
    padded = [[("" if v is None else str(v)) for v in r] + [""] * (width - len(r)) for r in rows]
    return pd.DataFrame(padded, dtype=object)


def test_header_below_titles_with_units_and_numbering_rows():
    g = grid([
        ["Project X, assay register"],
        [],
        ["", "Скважина", "Интервал", "", "Содержание", ""],
        ["", "", "от", "до", "Au", "Ag"],
        ["", "", "м", "м", "г/т", "г/т"],
        ["", "1", "2", "3", "4", "5"],
        ["", "C-1", "0", "1,5", "0,3", "<1"],
        ["", "", "", "", "", ""],
        ["", "C-1", "1,5", "3", "1.2", "2"],
    ])  # fmt: skip
    header = locate_header(g)
    assert (header.row, header.n_rows, header.data_start) == (2, 2, 6)
    assert header.skipped == ["units", "column numbers"]
    assert header.names == ["", "Скважина", "от", "до", "Au", "Ag"]
    table, repeated = table_from_grid(g, header)
    assert repeated == 0 and table.shape == (2, 5)  # blank column and blank row dropped
    assert suggest_column_map(table.columns, "interval").mapping == {
        "Скважина": "hole_id", "от": "depth_from", "до": "depth_to", "Au": "au", "Ag": "ag",
    }  # fmt: skip
    # the user's choice wins over detection
    forced = locate_header(g, header_row=3, header_rows=1)
    assert forced.names[2:4] == ["от", "до"] and forced.row == 3
    none = locate_header(g, header_row=-1)
    assert none.names == ["0", "1", "2", "3", "4", "5"] and none.data_start == 0


def test_merged_header_without_sub_header_and_repeated_header_rows():
    g = grid([
        ["№№", "Цооног дугаар", "Cорьцын төрөл", "Сорьц дугаар", "Интервал", "",
         "Сорьцын урт (м)", "Au"],
        ["1", "S-1", "Чөмгөн", "1001", "0", "1", "1", "0.2"],
        ["№№", "Цооног дугаар", "Cорьцын төрөл", "Сорьц дугаар", "Интервал", "",
         "Сорьцын урт (м)", "Au"],
        ["2", "S-1", "Чөмгөн", "1002", "1", "2", "1", "0.4"],
    ])  # fmt: skip
    header = locate_header(g)
    assert header.n_rows == 1 and header.names[4:6] == ["Интервал", "Интервал 2"]
    table, repeated = table_from_grid(g, header)
    assert repeated == 1 and len(table) == 2
    mapping = suggest_column_map(table.columns, "interval").mapping
    # "Cорьцын" starts with a Latin C in the source: look-alike letters are folded
    assert mapping == {
        "№№": "column", "Цооног дугаар": "hole_id", "Cорьцын төрөл": "sample_type",
        "Сорьц дугаар": "sample_id", "Интервал": "depth_from", "Интервал 2": "depth_to",
        "Сорьцын урт (м)": "sample_length", "Au": "au",
    }  # fmt: skip


@pytest.mark.parametrize(
    ("kind", "headers", "expected"),
    [
        ("collar", ["Цооног", "Уртраг", "Өргөрөг", "Өндөршил", "Азимут /градус/",
                    "Цооногийн налуу /градус/", "Цооногийн гүн /м/"],
         ["hole_id", "lon", "lat", "z", "azimuth", "dip", "max_depth"]),
        ("collar", ["№ скважины", "Долгота", "Долгота 2", "Долгота 3", "Широта", "Широта 2",
                    "Широта 3", "Абс. отметка", "Глубина, м"],
         ["hole_id", "lon_deg", "lon_min", "lon_sec", "lat_deg", "lat_min", "lat_sec", "z",
          "max_depth"]),
        ("survey", ["Скважина", "Глубина", "Азимут", "Угол наклона"],
         ["hole_id", "depth", "azimuth", "dip"]),
        ("interval", ["скв.", "От", "До", "Проба", "Au г/т"],
         ["hole_id", "depth_from", "depth_to", "sample_id", "au"]),
    ],
)  # fmt: skip
def test_non_english_headers_are_suggested(kind, headers, expected):
    suggestion = suggest_column_map(headers, kind)
    assert [suggestion.mapping.get(h) for h in headers] == expected
    assert suggestion.missing == ([] if kind != "collar" or "x" in expected else [])


def test_csv_with_title_lines_and_ragged_rows(tmp_path, clean):
    paths, clean_db = clean
    text = paths["assay"].read_text()
    messy = tmp_path / "assay.csv"
    messy.write_text("Assays exported 2024\n\n" + text)
    db = run_recipe(recipe_for(paths, {"file": str(messy)}))
    same_assays(db, clean_db)
    assert db.metadata.reports["tables"]["assay"]["sources"][0]["sheets"][0]["header_row"] == 3
