"""Depths carried by the sample number, QC rows among the samples and special lab
values: the import equals a clean-CSV import of the same data.

The fixtures are synthetic (``messy_fixtures``); the reference is always
``import_drillholes`` on three plain CSV files, which reads them with the plain reader
(``read_raw_table``) and knows nothing of sheets, header rows, rules or recipes.
"""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from messy_fixtures import (
    N_HOLES,
    clean_fixture,
    recipe_for,
    same_assays,
    tables_fixture,
    write_sheet_per_hole,
)

from bakumine.core.drillholes.importer import (
    ImportOptions,
    import_table,
    load_database,
)
from bakumine.core.drillholes.recipe import (
    run_recipe,
)
from bakumine.core.drillholes.tables import (
    classify_qc,
    depths_from_sample_ids,
)

tables = tables_fixture()
clean = clean_fixture()


def test_depths_from_sample_numbers_equal_clean(tmp_path, tables, clean):
    paths, clean_db = clean
    book = tmp_path / "assays.xlsx"
    write_sheet_per_hole(book, tables["assay"], depths="sample")
    source = {"file": str(book), "sheets": "SYN*", "hole_id_from_sheet": True,
              "sample_depth": True}  # fmt: skip
    db = run_recipe(recipe_for(paths, source))
    same_assays(db, clean_db)
    (check,) = db.metadata.reports["sample_depth"]["assay"]
    assert check["rows"] == len(tables["assay"]) and check["unmatched"] == 0
    assert (check["duplicates"], check["backwards"], check["overlaps"]) == (0, 0, 0)
    assert len(check["holes"]) == N_HOLES
    first = tables["assay"].groupby("hole_id").agg(a=("from", "min"), b=("to", "max"))
    for hole in check["holes"]:
        assert (hole["from"], hole["to"]) == tuple(first.loc[hole["hole_id"]])
    assert check["preview"][0]["sample_id"].startswith("SYN")
    # without the rule the rows have no interval: set aside, nothing imported
    del source["sample_depth"]
    db = run_recipe(recipe_for(paths, source))
    assert len(db.intervals["assay"]) == 0
    assert db.metadata.reports["qc"]["assay"] == {"no_interval": len(tables["assay"])}


def test_sample_number_patterns_and_what_is_not_understood():
    holes = ["DH-04"] * 5 + ["DH-11"]
    ids = ["DH04-4123", "DH04-4124", "DH04-124A", "DH04-5-6", "junk", "DH011-1107"]
    # the default rule strips the hole and takes what is left: here that is wrong on purpose
    start, end = depths_from_sample_ids(holes, ids)
    assert end.tolist()[:2] == [4123.0, 4124.0] and np.isnan(end[2]) and np.isnan(end[4])
    assert (start[3], end[3]) == (5.0, 6.0)
    # ids that repeat the hole number before the depth need a pattern
    patterns = [r"DH0*{holenum}-{holenum}(?P<to>\d+)", r"DH0*{holenum}-(?P<from>\d+)-(?P<to>\d+)"]
    start, end = depths_from_sample_ids(holes, ids, 1.0, patterns)
    np.testing.assert_array_equal(end, [123, 124, np.nan, 6, np.nan, 7])
    np.testing.assert_array_equal(start, [122, 123, np.nan, 5, np.nan, 6])
    with pytest.raises(ValueError, match="no \\(\\?P<to>"):
        depths_from_sample_ids(holes, ids, 1.0, [r"KT(\d+)"])
    with pytest.raises(ValueError, match="sample number pattern"):
        depths_from_sample_ids(holes, ids, 1.0, [r"(?P<to>\d+"])
    # a hole id is never matched half way: DH-1 does not own DH-12-5
    _, end = depths_from_sample_ids(["DH-1", "DH-12"], ["DH-12-5", "dh_012/5"])
    assert np.isnan(end[0]) and end[1] == 5.0


_LETTERS = st.text("ABCDEFGHKLMNPRSTXYZ", min_size=1, max_size=5)
_SEPARATORS = st.sampled_from(["-", "_", "/", " "])


@st.composite
def numbered_samples(draw):
    """A hole id, a sample id written by someone else (other case, padding, separators),
    and the interval it stands for."""
    prefix, number = draw(_LETTERS), draw(st.integers(1, 999))
    hole = f"{prefix}{draw(st.sampled_from(['', '-', '_']))}{number:0{draw(st.integers(1, 4))}d}"
    written = f"{prefix}{draw(st.sampled_from(['', '-', '_', ' ']))}"
    written += f"{number:0{draw(st.integers(1, 4))}d}"
    if draw(st.booleans()):
        written = written.lower()
    length = draw(st.sampled_from([0.5, 1.0, 1.5, 2.0, 3.0]))
    steps = draw(st.integers(1, 4000))
    to = steps * 0.5 if draw(st.booleans()) else float(steps)
    start = to - length
    explicit = draw(st.booleans())
    if start < 0 and not explicit:
        to, start = to + length, to
    text = f"{to:g}" if not explicit else f"{max(start, 0):g}-{to:g}"
    sample_id = f"{written}{draw(_SEPARATORS)}{text}"
    return hole, sample_id, max(start, 0) if explicit else start, to, length


@settings(max_examples=300, deadline=None)
@given(numbered_samples())
def test_sample_number_rule_recovers_the_interval(sample):
    hole, sample_id, start, to, length = sample
    got_from, got_to = depths_from_sample_ids([hole], [sample_id], length)
    assert got_to[0] == to
    assert got_from[0] == pytest.approx(start, abs=1e-12)
    # a different hole's number never yields a depth through the hole prefix
    other = re.sub(r"\d+$", lambda m: str(int(m.group()) + 1000), hole)
    lost_from, lost_to = depths_from_sample_ids([other], [sample_id], length)
    assert np.isnan(lost_to[0]) and np.isnan(lost_from[0])


@pytest.mark.parametrize("depths", ["columns", "sample"])
def test_qc_rows_are_set_aside_and_kept(tmp_path, tables, clean, depths):
    paths, clean_db = clean
    book = tmp_path / "assays.xlsx"
    counts = write_sheet_per_hole(book, tables["assay"], depths=depths, qc=True)
    assert min(counts.values()) > 0
    source = {"file": str(book), "sheets": "SYN*", "hole_id_from_sheet": True,
              "sample_depth": depths == "sample"}  # fmt: skip
    db = run_recipe(recipe_for(paths, source))
    same_assays(db, clean_db)
    assert db.metadata.reports["qc"]["assay"] == counts
    qc = db.qc["assay"]
    assert len(qc) == sum(counts.values())
    assert Counter(qc["qc_type"]) == counts
    assert set(qc["hole_id"]) == set(clean_db.collar["hole_id"])
    assert {"sample_id", "sheet", "row", "source", "au"} <= set(qc.columns)
    assert set(qc.loc[qc["qc_type"] == "standard", "au"]) == {"1.05"}
    # saved with the database and loaded back
    loaded = load_database(db.save(tmp_path / "db"))
    pd.testing.assert_frame_equal(loaded.qc["assay"], qc)
    # with QC handling off the rows stay (and the validator has to deal with them)
    source["qc"] = {"mode": "off"}
    if depths == "columns":
        kept = run_recipe(recipe_for(paths, source))
        assert len(kept.intervals["assay"]) == len(clean_db.intervals["assay"]) + sum(
            counts.values()
        )
        assert not kept.qc


def test_qc_words_in_a_sample_type_column():
    table = pd.DataFrame(
        {
            "hole_id": ["A"] * 6,
            "depth_from": ["0", "1", "", "", "2", "3"],
            "depth_to": ["1", "2", "", "", "3", "4"],
            "sample_type": ["Чөмгөн", "core", "Blank", "CRM OREAS-1", "Давхар", "core"],
            "sample_id": ["1", "2", "3", "4", "5", "6"],
        }
    )
    assert classify_qc(table).tolist() == ["", "", "blank", "standard", "duplicate", ""]


def test_special_values_are_numeric_and_counted_by_kind(tables, clean):
    _, clean_db = clean
    au_text = tables["assay"]["au"]
    au = clean_db.intervals["assay"]["au"]
    assert au.dtype == np.float64 and clean_db.intervals["assay"]["ag"].dtype == np.float64
    below = au_text == "<0.01"
    above = au_text == ">100"
    comma = au_text.str.contains(",")
    assert (au[below.to_numpy()] == 0.005).all()  # half the limit
    assert (au[above.to_numpy()] == 100.0).all()
    np.testing.assert_allclose(
        au[comma.to_numpy()], au_text[comma].str.replace(",", ".").astype(float)
    )
    assert au[(au_text.isin(["LNR", "-"])).to_numpy()].isna().all()
    notes = {n["kind"]: n for n in clean_db.metadata.notes if n["column"] == "au"}
    assert notes["below_detection"]["count"] == below.sum() > 0
    assert notes["above_detection"]["count"] == above.sum() > 0
    assert notes["decimal_comma"]["count"] == comma.sum() > 0
    assert notes["unparseable_set_to_nan"]["values"] == {"LNR": int((au_text == "LNR").sum())}
    assert notes["missing_token"]["values"] == {"-": int((au_text == "-").sum())}
    ag = {n["kind"]: n for n in clean_db.metadata.notes if n["column"] == "ag"}
    assert ag["missing_token"]["values"] == {"n/a": int((tables["assay"]["ag"] == "n/a").sum())}


def test_special_value_rules():
    frame = pd.DataFrame(
        {"h": ["A"] * 6, "f": list("012345"), "t": list("123456"),
         "au": ["<0,5", "> 10", "1,25", "IS", "2", "-"]}  # fmt: skip
    )
    column_map = {"h": "hole_id", "f": "depth_from", "t": "depth_to", "au": "au"}
    table, _ = import_table(frame, "interval", column_map)
    np.testing.assert_array_equal(table["au"], [0.25, 10.0, 1.25, np.nan, 2.0, np.nan])
    options = ImportOptions(above_detection="nan", below_detection="limit", decimal_comma=False)
    table, notes = import_table(frame, "interval", column_map, options)
    # without the decimal comma rule three cells are text, three are numbers: a text column
    assert table["au"].dtype == "str"
    with pytest.raises(ValueError, match="above_detection"):
        ImportOptions(above_detection="half")
