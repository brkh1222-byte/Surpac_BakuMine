"""P1.T4 acceptance tests: interval merge."""

import math
import time

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from pandas.testing import assert_frame_equal

from bakumine.core.drillholes.intervals import _sort_by_hole_then_depth, merge_intervals


def table(rows: list[tuple], columns: list[str]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", *columns])
    frame["hole_id"] = frame["hole_id"].astype("str")
    return frame.astype({"depth_from": np.float64, "depth_to": np.float64})


def union_length(tables: dict[str, pd.DataFrame]) -> dict[str, float]:
    """Covered depth per hole by a plain sweep over sorted intervals (independent check)."""
    spans: dict[str, list[tuple[float, float]]] = {}
    for frame in tables.values():
        for hole, start, end in zip(
            frame["hole_id"], frame["depth_from"], frame["depth_to"], strict=True
        ):
            spans.setdefault(hole, []).append((float(start), float(end)))
    lengths = {}
    for hole, items in spans.items():
        total, reach = 0.0, -math.inf
        for start, end in sorted(items):
            if end > reach:
                total += end - max(start, reach)
                reach = end
        lengths[hole] = total
    return lengths


def merged_length(merged: pd.DataFrame) -> dict[str, float]:
    length = merged["depth_to"] - merged["depth_from"]
    return length.groupby(merged["hole_id"]).sum().to_dict()


# --------------------------------------------------------------------------------------
# hand-worked example
# --------------------------------------------------------------------------------------
ASSAY = table(
    [("A", 0, 10, 1.0), ("A", 10, 20, 2.0), ("A", 30, 40, 3.0), ("B", 0, 4, 9.0)], ["au"]
)
GEOLOGY = table(
    [("A", 0, 15, "OXIDE", 1), ("A", 15, 35, "FRESH", 2), ("C", 5, 8, "DYKE", 7)],
    ["rock", "code"],
)


def test_hand_worked_two_table_example():
    """
    Hole A   assay    0----10----20      30----40
             geology  0-------15-----------35
             merged   0----10-15--20-----30-35-40   (6 segments, 20-30 has geology only)
    Hole B   assay only; hole C geology only.
    """
    merged = merge_intervals({"assay": ASSAY, "geology": GEOLOGY}, source_rows=True)
    nan, na = np.nan, pd.NA
    expected = pd.DataFrame(
        [
            ("A", 0.0, 10.0, 0, 0, 1.0, "OXIDE", 1.0),
            ("A", 10.0, 15.0, 1, 0, 2.0, "OXIDE", 1.0),
            ("A", 15.0, 20.0, 1, 1, 2.0, "FRESH", 2.0),
            ("A", 20.0, 30.0, na, 1, nan, "FRESH", 2.0),
            ("A", 30.0, 35.0, 2, 1, 3.0, "FRESH", 2.0),
            ("A", 35.0, 40.0, 2, na, 3.0, None, nan),
            ("B", 0.0, 4.0, 3, na, 9.0, None, nan),
            ("C", 5.0, 8.0, na, 2, nan, "DYKE", 7.0),
        ],
        columns=["hole_id", "depth_from", "depth_to", "assay_row", "geology_row", "au", "rock",
                 "code"],
    )  # fmt: skip
    expected = expected.astype(
        {"hole_id": "str", "assay_row": "Int64", "geology_row": "Int64", "rock": "str"}
    )
    assert_frame_equal(merged, expected)

    # Length: hole A is covered 0-40 without a hole in it, B 0-4, C 5-8.
    assert merged_length(merged) == {"A": 40.0, "B": 4.0, "C": 3.0}
    # Metal (grade x length) of the assay table is unchanged: 10*1 + 10*2 + 10*3 and 4*9.
    metal = (merged["au"] * (merged["depth_to"] - merged["depth_from"])).groupby(
        merged["hole_id"]
    ).sum()
    assert metal.to_dict() == {"A": 60.0, "B": 36.0, "C": 0.0}


def test_default_output_has_no_row_columns_and_keeps_table_order():
    merged = merge_intervals({"geology": GEOLOGY, "assay": ASSAY})
    assert list(merged.columns) == ["hole_id", "depth_from", "depth_to", "rock", "code", "au"]
    assert merged["hole_id"].unique().tolist() == ["A", "C", "B"]  # first appearance
    assert merged["hole_id"].dtype == "str" and merged["rock"].dtype == "str"
    assert merged["au"].dtype == np.float64


def test_single_table_comes_back_sorted_and_unchanged():
    shuffled = ASSAY.iloc[[2, 3, 0, 1]].reset_index(drop=True)
    merged = merge_intervals({"assay": shuffled})
    assert_frame_equal(merged, ASSAY)
    full = table([("A", 0, 1, 5), ("A", 1, 2, 6)], ["n"])
    assert merge_intervals({"t": full})["n"].dtype == np.int64  # no gap, dtype kept


def test_row_order_of_the_input_does_not_matter():
    rng = np.random.default_rng(3)
    a = ASSAY.iloc[rng.permutation(len(ASSAY))].reset_index(drop=True)
    g = GEOLOGY.iloc[rng.permutation(len(GEOLOGY))].reset_index(drop=True)
    key = ["hole_id", "depth_from"]
    assert_frame_equal(
        merge_intervals({"assay": a, "geology": g}).sort_values(key, ignore_index=True),
        merge_intervals({"assay": ASSAY, "geology": GEOLOGY}).sort_values(key, ignore_index=True),
    )


def test_three_tables_and_shared_attribute_names():
    density = table([("A", 5, 12, 2.7, 1)], ["sg", "code"])
    merged = merge_intervals({"assay": ASSAY, "geology": GEOLOGY, "density": density})
    assert list(merged.columns) == [
        "hole_id", "depth_from", "depth_to", "au", "rock", "geology_code", "sg", "density_code",
    ]  # fmt: skip
    a = merged[merged["hole_id"] == "A"]
    assert a["depth_from"].tolist() == [0, 5, 10, 12, 15, 20, 30, 35]
    assert a["depth_to"].tolist() == [5, 10, 12, 15, 20, 30, 35, 40]
    np.testing.assert_array_equal(a["sg"], [np.nan, 2.7, 2.7, *[np.nan] * 5])
    np.testing.assert_array_equal(a["geology_code"], [1, 1, 1, 1, 2, 2, 2, np.nan])


def test_overlap_within_a_table_raises():
    bad = table([("A", 0, 10, 1.0), ("A", 8, 12, 2.0)], ["au"])
    with pytest.raises(ValueError, match=r"assay: 1 overlapping interval pair.*'A'.*0-10.*8-12"):
        merge_intervals({"assay": bad, "geology": GEOLOGY})
    same_start = table([("A", 0, 10, 1.0), ("A", 0, 10, 2.0)], ["au"])
    with pytest.raises(ValueError, match="overlapping"):
        merge_intervals({"assay": same_start})
    # The same depths in different holes are not an overlap.
    merge_intervals({"assay": table([("A", 0, 10, 1.0), ("B", 0, 10, 2.0)], ["au"])})


def test_invalid_rows_raise_or_are_dropped():
    bad = table(
        [("A", 0, 10, 1.0), ("A", 10, 10, 2.0), ("A", 25, 20, 3.0), ("A", np.nan, 30, 4.0),
         (None, 0, 5, 5.0)],
        ["au"],
    )  # fmt: skip
    with pytest.raises(ValueError, match="assay: 4 interval.*row 1"):
        merge_intervals({"assay": bad})
    merged = merge_intervals({"assay": bad}, on_invalid="drop", source_rows=True)
    assert merged["depth_to"].tolist() == [10.0]
    assert merged["assay_row"].tolist() == [0]
    with pytest.raises(ValueError, match="on_invalid"):
        merge_intervals({"assay": bad}, on_invalid="ignore")
    with pytest.raises(ValueError, match="at least one"):
        merge_intervals({})
    with pytest.raises(ValueError, match="missing columns"):
        merge_intervals({"assay": ASSAY.drop(columns="depth_to")})


def test_tolerance_snaps_float_noise():
    a = table([("A", 0, 10, 1.0), ("A", 10, 20, 2.0)], ["au"])
    g = table([("A", 0, 10.0000001, "X"), ("A", 10.0000001, 20, "Y")], ["rock"])
    exact = merge_intervals({"a": a, "g": g})
    assert len(exact) == 3  # a sliver 10 - 10.0000001
    snapped = merge_intervals({"a": a, "g": g}, tolerance=1e-6)
    assert snapped["depth_from"].tolist() == [0.0, 10.0]
    assert snapped["depth_to"].tolist() == [10.0, 20.0]
    assert snapped["au"].tolist() == [1.0, 2.0] and snapped["rock"].tolist() == ["X", "Y"]
    # A tiny overlap inside the tolerance is float noise, not an error.
    noisy = table([("A", 0, 10.0000001, 1.0), ("A", 10, 20, 2.0)], ["au"])
    with pytest.raises(ValueError, match="overlapping"):
        merge_intervals({"a": noisy})
    assert len(merge_intervals({"a": noisy}, tolerance=1e-6)) == 2


def test_empty_tables():
    empty = ASSAY.iloc[:0]
    merged = merge_intervals({"assay": empty, "geology": GEOLOGY})
    assert len(merged) == 3 and merged["au"].isna().all()
    nothing = merge_intervals({"assay": empty}, source_rows=True)
    assert len(nothing) == 0
    assert list(nothing.columns) == ["hole_id", "depth_from", "depth_to", "assay_row", "au"]


def test_sort_falls_back_when_the_combined_key_rounds():
    """Depths 1e-9 apart at large hole codes are equal in the combined key."""
    rng = np.random.default_rng(0)
    hole = np.repeat(np.arange(999_000, 1_000_000), 6)
    depth = rng.permutation(np.tile(1000.0 + 1e-9 * np.arange(6), 1000))
    order = _sort_by_hole_then_depth(hole, depth)
    expected = np.lexsort((depth, hole))
    np.testing.assert_array_equal(hole[order], hole[expected])
    np.testing.assert_array_equal(depth[order], depth[expected])


# --------------------------------------------------------------------------------------
# real tables
# --------------------------------------------------------------------------------------
def test_small_database_assay_and_geology(small_db):
    merged = merge_intervals(small_db.intervals)
    assert list(merged.columns) == ["hole_id", "depth_from", "depth_to", "au", "cu", "rock"]
    # Per hole: assay boundaries every 10 m plus the rock contacts at 15 and 62.5.
    assert (merged.groupby("hole_id").size() == 12).all()
    assert merged_length(merged) == {hole: 100.0 for hole in small_db.collar["hole_id"]}
    hole = merged[merged["hole_id"] == "DH001"]
    assert hole["depth_from"].tolist() == [0, 10, 15, 20, 30, 40, 50, 60, 62.5, 70, 80, 90]
    assert hole["rock"].tolist() == ["OVB"] * 2 + ["ORE"] * 6 + ["WST"] * 4
    assert hole["au"].tolist()[:4] == [0.1, 0.2, 0.2, 0.3]
    assert merged["au"].isna().sum() == 1  # the unassayed interval is still unassayed


def test_babbitt_length_and_metal_are_preserved(babbitt_db):
    assay = babbitt_db.intervals["assay"]
    # A second, coarser table: 50 ft benches down each hole.
    depth = babbitt_db.collar.set_index("hole_id")["max_depth"]
    benches = [
        (hole, start, min(start + 50.0, end), int(start // 50))
        for hole, end in depth.items()
        for start in np.arange(0.0, end, 50.0)
    ]
    bench = table(benches, ["bench"])
    merged = merge_intervals({"assay": assay, "bench": bench})

    expected = union_length({"assay": assay})
    got = merged_length(merged)
    assert got.keys() == expected.keys()
    for hole, length in expected.items():
        assert got[hole] == pytest.approx(length, rel=1e-12, abs=1e-9)
    assert sum(got.values()) == pytest.approx(541230.0)  # SOURCE.md: total drilled feet

    length = merged["depth_to"] - merged["depth_from"]
    source_length = assay["depth_to"] - assay["depth_from"]
    for grade in ("cu", "ni", "s", "fe"):
        before = (assay[grade] * source_length).groupby(assay["hole_id"]).sum()
        after = (merged[grade] * length).groupby(merged["hole_id"]).sum()
        np.testing.assert_allclose(after[before.index], before, rtol=1e-9, atol=1e-9)
        # Missing stays missing: the unassayed length is the same.
        assert length[merged[grade].isna()].sum() == pytest.approx(
            source_length[assay[grade].isna()].sum()
        )
    assert merged["bench"].notna().all()
    assert (merged.groupby(["hole_id", "bench"])["depth_to"].max()
            - merged.groupby(["hole_id", "bench"])["depth_from"].min()).max() <= 50.0  # fmt: skip


# --------------------------------------------------------------------------------------
# property test
# --------------------------------------------------------------------------------------
HOLES = ["H1", "H2", "H3"]


@st.composite
def interval_tables(draw) -> dict[str, pd.DataFrame]:
    """1-3 tables of non-overlapping intervals on a 0.25 grid (so boundaries coincide
    between tables often), with gaps, in shuffled row order."""
    tables = {}
    for t in range(draw(st.integers(1, 3))):
        rows = []
        for hole in draw(st.lists(st.sampled_from(HOLES), unique=True, max_size=3)):
            points = sorted(draw(st.sets(st.integers(0, 60), min_size=2, max_size=12)))
            for start, end in zip(points[:-1], points[1:], strict=True):
                if draw(st.booleans()):
                    value = draw(st.floats(-1e3, 1e3, allow_nan=False))
                    label = draw(st.sampled_from(["ORE", "WST", "DYKE"]))
                    rows.append((hole, start * 0.25, end * 0.25, value, label))
        rows = draw(st.permutations(rows))
        tables[f"t{t}"] = table(rows, [f"value{t}", f"label{t}"])
    return tables


@settings(max_examples=300, deadline=None)
@given(interval_tables())
def test_property_segments_carry_exactly_their_source_interval(tables):
    merged = merge_intervals(tables, source_rows=True)
    seg_from, seg_to = merged["depth_from"].to_numpy(), merged["depth_to"].to_numpy()
    assert (seg_to > seg_from).all()

    # Segments of a hole are sorted and do not overlap.
    for _, hole in merged.groupby("hole_id"):
        assert (hole["depth_from"].to_numpy()[1:] >= hole["depth_to"].to_numpy()[:-1]).all()

    for t, (name, source) in enumerate(tables.items()):
        records = list(source.itertuples(index=False, name=None))
        for i, hole in enumerate(merged["hole_id"]):
            inside, touching = [], 0
            for row, (source_hole, start, end, _, _) in enumerate(records):
                if source_hole != hole:
                    continue
                if start <= seg_from[i] and seg_to[i] <= end:
                    inside.append(row)
                elif max(start, seg_from[i]) < min(end, seg_to[i]):
                    touching += 1
            # Inside exactly one or zero intervals of this table; never straddling one.
            assert touching == 0
            assert len(inside) <= 1
            value, label = merged[f"value{t}"].iloc[i], merged[f"label{t}"].iloc[i]
            if inside:
                assert merged[f"{name}_row"].iloc[i] == inside[0]
                assert value == records[inside[0]][3]  # exactly that interval's values
                assert label == records[inside[0]][4]
            else:
                assert pd.isna(merged[f"{name}_row"].iloc[i])
                assert np.isnan(value) and pd.isna(label)

    # Every segment is covered by at least one table.
    rows = merged[[f"{name}_row" for name in tables]]
    assert rows.notna().any(axis=1).all()

    # Total length per hole = union of covered depth.
    expected = {h: v for h, v in union_length(tables).items()}
    got = merged_length(merged)
    assert got.keys() == expected.keys()
    for hole in expected:
        assert got[hole] == pytest.approx(expected[hole], abs=1e-9)

    # Length-weighted sums of each attribute are conserved, per hole and table.
    length = seg_to - seg_from
    for t, source in enumerate(tables.values()):
        for hole in set(source["hole_id"]):
            s = source[source["hole_id"] == hole]
            before = math.fsum(s[f"value{t}"] * (s["depth_to"] - s["depth_from"]))
            m = (merged["hole_id"] == hole).to_numpy() & merged[f"value{t}"].notna().to_numpy()
            after = math.fsum(merged[f"value{t}"].to_numpy()[m] * length[m])
            assert after == pytest.approx(before, rel=1e-9, abs=1e-9)
            # ...and so is the length carrying a value.
            assert math.fsum(length[m]) == pytest.approx(
                math.fsum(s["depth_to"] - s["depth_from"]), abs=1e-9
            )

    # The output is cut at every input boundary that lies inside covered depth.
    for source in tables.values():
        for hole, start, end, _, _ in source.itertuples(index=False, name=None):
            cuts = merged.loc[merged["hole_id"] == hole, ["depth_from", "depth_to"]].to_numpy()
            assert start in cuts[:, 0] and end in cuts[:, 1]


# --------------------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------------------
def _big_table(holes: int, per_hole: int, seed: int, column: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    end = np.cumsum(rng.uniform(0.5, 1.5, (holes, per_hole)), axis=1)
    start = np.concatenate([np.zeros((holes, 1)), end[:, :-1]], axis=1)
    ids = np.repeat(np.array([f"H{i:05d}" for i in range(holes)], dtype=object), per_hole)
    return pd.DataFrame(
        {
            "hole_id": pd.Series(ids, dtype="str"),
            "depth_from": start.ravel(),
            "depth_to": end.ravel(),
            column: rng.normal(size=holes * per_hole),
        }
    )


def test_performance_5000_holes_x_500_intervals():
    """Two tables of 5 000 holes x 500 intervals each (2.5 million rows per table), with
    unrelated boundaries: 5 million output segments. Must run in a few seconds."""
    assay = _big_table(5000, 500, 1, "grade")
    geology = _big_table(5000, 500, 2, "density")
    geology["rock"] = pd.Series(np.where(geology["density"] > 0, "ORE", "WST"), dtype="str")
    started = time.perf_counter()
    merged = merge_intervals({"assay": assay, "geology": geology})
    elapsed = time.perf_counter() - started
    print(f"\nmerge_intervals: {len(merged):,} segments in {elapsed:.2f} s")
    assert elapsed < 10.0, f"interval merge took {elapsed:.1f} s"

    # Both tables start at 0 with no gaps and never share a boundary below it: a hole has
    # 1 + 500 + 500 distinct boundaries, so 1000 segments.
    assert len(merged) == 5000 * 1000
    length = (merged["depth_to"] - merged["depth_from"]).to_numpy()
    assert (length > 0).all()
    deepest = np.maximum(
        assay.groupby("hole_id", sort=False)["depth_to"].max().to_numpy(),
        geology.groupby("hole_id", sort=False)["depth_to"].max().to_numpy(),
    )
    np.testing.assert_allclose(
        pd.Series(length).groupby(merged["hole_id"].to_numpy(), sort=False).sum().to_numpy(),
        deepest,
        rtol=1e-12,
    )
    # Length-weighted grade is conserved over the whole table.
    grade = merged["grade"].to_numpy()
    carried = ~np.isnan(grade)
    before = float((assay["grade"] * (assay["depth_to"] - assay["depth_from"])).sum())
    assert float((grade[carried] * length[carried]).sum()) == pytest.approx(before, abs=1e-6)
