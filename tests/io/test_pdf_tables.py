"""Tables out of a text PDF: a generated 30+ page appendix comes back row for row."""

from __future__ import annotations

import csv
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from pdf_fixture import COLUMNS, ROWS_PER_PAGE, appendix_rows, write_appendix

from bakumine.cli import main
from bakumine.core.drillholes.recipe import make_recipe, run_recipe
from bakumine.core.io.pdf_tables import extract_tables, parse_pages, table_report, write_tables


@pytest.fixture(scope="module")
def appendix(tmp_path_factory):
    path = tmp_path_factory.mktemp("pdf") / "report volume 2.pdf"
    rows = write_appendix(path)
    return path, rows


@pytest.fixture(scope="module")
def extracted(appendix):
    path, _ = appendix
    return extract_tables(path)


def n_pages(path) -> int:
    import pdfplumber

    with pdfplumber.open(str(path)) as pdf:
        return len(pdf.pages)


def test_long_table_comes_back_row_for_row(appendix, extracted):
    path, rows = appendix
    pages = n_pages(path)
    assay_pages = -(-len(rows) // ROWS_PER_PAGE)
    assert pages == assay_pages + 1 >= 31
    summary, assay = extracted  # the small table before it is a table of its own
    assert summary.columns == ["No", "Hole", "X", "Y", "Depth, m"]
    assert len(summary.rows) == 5 and summary.first_page == summary.last_page == 1
    assert summary.rows[0] == ["1", "SYN25_01", "5204031.5", "412017.25", "251.0"]

    assert assay.columns == COLUMNS  # the two-row header, flattened
    assert len(assay.rows) == len(rows)
    assert assay.rows == rows  # ids carried down and re-joined, empty cells in place
    assert (assay.first_page, assay.last_page) == (2, pages)
    assert assay.pages == [2 + i // ROWS_PER_PAGE for i in range(len(rows))]
    notes = assay.notes
    assert notes["header_rows"] == 3
    assert notes["repeated_headers"] == assay_pages - 2  # one page has no header at all
    assert notes["title_rows"] == 1
    assert notes["titles"][0]["text"] == "Infill drilling, second season"
    holes = Counter(r[1] for r in rows)
    assert notes["carried_down"] == {"columns": ["Hole"], "cells": len(rows) - len(holes)}
    assert sum(1 for r in rows if r[7] == "") > 5 and sum(1 for r in rows if r[8] == "") > 100


def test_report_counts_rows_and_depth_ranges_per_hole(appendix, extracted):
    _, rows = appendix
    report = table_report(extracted[1])
    frame = pd.DataFrame(rows, columns=COLUMNS).astype({"from": float, "to": float})
    assert report["rows"] == len(rows) and report["n_holes"] == frame["Hole"].nunique()
    assert report["rows_without_hole"] == 0 and report["rows_without_depths"] == 0
    assert report["holes_from_backwards"] == [] and report["holes_in_several_runs"] == []
    want = frame.groupby("Hole", sort=False).agg(n=("No", "size"), a=("from", "min"),
                                                 b=("to", "max"))  # fmt: skip
    got = pd.DataFrame(report["holes"]).set_index("hole_id")
    assert got.index.tolist() == want.index.tolist()
    np.testing.assert_array_equal(got["rows"], want["n"])
    np.testing.assert_array_equal(got["from"], want["a"])
    np.testing.assert_array_equal(got["to"], want["b"])
    assert got["first_page"].is_monotonic_increasing


def test_from_going_backwards_and_split_holes_are_reported(tmp_path):
    rows = appendix_rows(n_holes=2, seed=5)
    first = [r for r in rows if r[1] == "SYN25_01"]
    second = [r for r in rows if r[1] == "SYN25_02"]
    # the second half of hole 1 printed after hole 2, and two rows of hole 2 swapped
    second[10], second[11] = second[11], second[10]
    shuffled = first[:50] + second + first[50:]
    path = tmp_path / "shuffled.pdf"
    write_appendix(path, shuffled)
    table = extract_tables(path, f"2-{n_pages(path)}")[0]
    assert table.rows == shuffled
    report = table_report(table)
    assert report["holes_from_backwards"] == ["SYN25_02"]
    assert report["holes_in_several_runs"] == ["SYN25_01"]
    assert [h["hole_id"] for h in report["holes"]] == ["SYN25_01", "SYN25_02", "SYN25_01"]


def test_page_ranges_and_options(appendix, extracted):
    path, rows = appendix
    assert parse_pages("3-5,9", 40) == [3, 4, 5, 9]
    assert parse_pages("38-", 40) == [38, 39, 40] and parse_pages(None, 3) == [1, 2, 3]
    for bad in ("a-b", "5-3", "0"):
        with pytest.raises(ValueError):
            parse_pages(bad, 40)
    with pytest.raises(ValueError, match="none of"):
        parse_pages("50-60", 40)
    (part,) = extract_tables(path, "5-7")  # a slice of the long table
    want = [list(r) for r in rows[3 * ROWS_PER_PAGE : 6 * ROWS_PER_PAGE]]
    # the hole was named on an earlier page: until the next hole starts it is unknown
    known = next(i for i, r in enumerate(part.rows) if r[1])
    assert [r[1] for r in part.rows[:known]] == [""] * known and known > 0
    assert [r[:1] + r[2:] for r in part.rows] == [r[:1] + r[2:] for r in want]
    assert part.rows[known:] == want[known:]
    assert table_report(part)["rows_without_hole"] == known
    (plain,) = extract_tables(path, "5-6", carry_down="none")
    assert sum(1 for r in plain.rows if r[1]) < 4
    (by_position,) = extract_tables(path, "5-6", carry_down=[2])
    assert by_position.rows == part.rows[: 2 * ROWS_PER_PAGE]
    with pytest.raises(ValueError, match="no column 'Nope'"):
        extract_tables(path, "5", carry_down=["Nope"])
    (single,) = extract_tables(path, "1-3", one_table=True)  # the second header is data
    assert {"No", "Hole", "X", "Y", "Depth, m"} <= set(single.columns)
    assert len(single.rows) > 5 + 2 * ROWS_PER_PAGE


def test_cli_writes_csv_and_report_and_the_csv_imports(appendix, tmp_path, capsys):
    path, rows = appendix
    out = tmp_path / "tables"
    code = main(["extract-tables", str(path), "--pages", "2-", "--out", str(out)])
    printed = capsys.readouterr()
    assert code == 0, printed.err
    stem = "report_volume_2"
    assert f"wrote {out / (stem + '_1.csv')} ({len(rows)} rows, 9 columns" in printed.out
    assert "12 holes; rows without a hole 0, without depths 0" in printed.out
    assert "1 title rows dropped" in printed.out and "carried down" in printed.out
    with (out / f"{stem}_1.csv").open(newline="", encoding="utf-8") as handle:
        written = list(csv.reader(handle))
    assert written[0] == COLUMNS and written[1:] == rows
    report = json.loads((out / f"{stem}_report.json").read_text(encoding="utf-8"))
    assert report["tables"][0]["rows"] == len(rows) and len(report["tables"][0]["holes"]) == 12
    assert report["tables"][0]["titles"][0]["page"] > 2

    # the CSV is an ordinary import source: header and columns are found by themselves
    frame = pd.DataFrame(rows, columns=COLUMNS)
    collar = tmp_path / "collar.csv"
    pd.DataFrame(
        {"hole_id": frame["Hole"].unique(), "x": 0.0, "y": 0.0, "z": 0.0, "azimuth": 0.0,
         "dip": -90.0}  # fmt: skip
    ).to_csv(collar, index=False)
    recipe = make_recipe(
        {"collar": {"file": str(collar)}, "assay": {"file": str(out / f"{stem}_1.csv")}}
    )
    db = run_recipe(recipe)
    assay = db.intervals["assay"]
    assert len(assay) == len(rows)
    assert assay["hole_id"].tolist() == frame["Hole"].tolist()
    np.testing.assert_array_equal(assay["depth_from"], frame["from"].astype(float))
    np.testing.assert_array_equal(assay["depth_to"], frame["to"].astype(float))
    au = pd.to_numeric(frame["Au, g/t"], errors="coerce")
    np.testing.assert_array_equal(assay["au_g_t"], au)
    ag = frame["Ag, g/t"].replace({"<1.0": "0.5", "": "nan"}).astype(float)
    np.testing.assert_array_equal(assay["ag_g_t"], ag)
    assert assay["sample_id"].tolist() == frame["Sample No"].tolist()


def test_cli_errors(tmp_path, capsys):
    assert main(["extract-tables", str(tmp_path / "none.pdf"), "--out", str(tmp_path)]) == 2
    assert "PDF not found" in capsys.readouterr().err
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import Paragraph, SimpleDocTemplate

    text_only = tmp_path / "text.pdf"
    SimpleDocTemplate(str(text_only)).build(
        [Paragraph("No tables here.", getSampleStyleSheet()["Normal"])]
    )
    assert main(["extract-tables", str(text_only), "--out", str(tmp_path / "o")]) == 2
    assert "no table found" in capsys.readouterr().err
    broken = tmp_path / "broken.pdf"
    broken.write_bytes(b"not a pdf")
    assert main(["extract-tables", str(broken), "--out", str(tmp_path / "o")]) == 2
    assert "cannot read broken.pdf" in capsys.readouterr().err
    assert write_tables([], tmp_path / "empty")[0] == []
