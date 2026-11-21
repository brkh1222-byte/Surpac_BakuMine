"""Run-building modes of the mineralised intervals: ``strict``, ``average``, ``moving-average``.

Every expected value is worked by hand in the comments (cut-off 0.2 and 1 m samples from
0 m unless stated; ``E`` is the excess ``sum((g - cutoff) * length)``). The property tests
at the end check the rules of each mode on random tables with a separate implementation.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from test_mineralised import PARAMS, assay_tables, codes, hole, inside, run, spans

from bakumine.cli import main
from bakumine.core.drillholes.mineralised import MODES, InterceptParams, mineralised_intervals


def test_modes_and_their_parameters():
    assert MODES == ("strict", "average", "moving-average")
    assert InterceptParams(0.2).mode == "strict" and InterceptParams(0.2).window is None
    with pytest.raises(ValueError, match="mode must be one of"):
        InterceptParams(0.2, mode="loose")
    with pytest.raises(ValueError, match="needs a window"):
        InterceptParams(0.2, mode="moving-average")
    with pytest.raises(ValueError, match="needs a window"):
        InterceptParams(0.2, mode="moving-average", window=0.0)
    with pytest.raises(ValueError, match="moving-average only"):
        InterceptParams(0.2, mode="average", window=3.0)
    # parameter files: old ones have no mode; new ones round-trip
    assert InterceptParams.from_dict({"bakumine_intercepts": 1, "cutoff": 0.2}).mode == "strict"
    moving = InterceptParams(0.2, mode="moving-average", window=3)
    assert moving.window == 3.0 and InterceptParams.from_dict(moving.to_dict()) == moving
    assert moving.to_dict()["mode"] == "moving-average"


# ---------------------------------------------------------------------------- average


def test_average_docstring_example():
    table = hole([0.25, 0.1, 0.1, 1.0, 0.05])
    # strict: E(0-1) = +0.05, waste 1-3 E = -0.2: 0.05 - 0.2 < 0, the run stops at 1 m
    assert spans(run(table, max_internal_waste=2)) == [(0, 1), (3, 4)]
    # average: 0-4 has E = 0.05 - 0.2 + 0.8 = +0.65, starts and ends on ore
    r = run(table, max_internal_waste=2, mode="average")
    assert codes(r) == [(0, 4, "MIN"), (4, 5, "WASTE")]
    row = r.intercepts.iloc[0]
    assert row["grade"] == pytest.approx((0.25 + 0.1 + 0.1 + 1.0) / 4)  # 0.3625
    assert row["waste_length"] == 2.0 and row["n_samples"] == 4 and row["rule"] == "thickness"
    assert r.summary()["mode"] == "average" and r.summary()["mineralised_length"] == 4.0


def test_average_takes_the_longest_run_and_may_pass_a_weak_point():
    table = hole([0.3, 0.05, 0.05, 0.21, 0.05, 2.0])
    # E so far at the ore ends: 0-1: +0.10; 0-4: 0.10 - 0.30 + 0.01 = -0.19 (not an end);
    # 0-6: -0.19 - 0.15 + 1.80 = +1.46. The longest run that makes the cut-off is 0-6.
    r = run(table, max_internal_waste=2, mode="average")
    assert spans(r) == [(0, 6)]
    assert r.intercepts.iloc[0]["grade"] == pytest.approx(2.66 / 6)
    assert r.intercepts.iloc[0]["waste_length"] == 3.0
    # strict keeps the three ore samples apart: 0.10 - 0.30 < 0 and 0.01 - 0.15 < 0
    assert spans(run(table, max_internal_waste=2)) == [(0, 1), (3, 4), (5, 6)]


def test_average_stops_where_the_grade_cannot_be_kept():
    # 0-4: E = 0.01 - 0.40 + 0.05 = -0.34 < 0: the second ore sample is its own run
    r = run(hole([0.21, 0.0, 0.0, 0.25]), max_internal_waste=2, mode="average")
    assert spans(r) == [(0, 1), (3, 4)]
    # a weak first sample is carried by what follows: E = 0 - 0.2 + 4.8
    weak = hole([0.2, 0.0, 5.0])
    assert spans(run(weak, max_internal_waste=1, mode="average")) == [(0, 3)]
    assert spans(run(weak, max_internal_waste=1)) == [(0, 1), (2, 3)]  # strict: 0 - 0.2 < 0


def test_average_respects_the_waste_limits():
    table = hole([1.0, 0.0, 0.0, 0.21, 0.0, 0.0, 0.21])
    # E at the ore ends: +0.8; 0.8 - 0.4 + 0.01 = +0.41; 0.41 - 0.4 + 0.01 = +0.02
    r = run(table, max_internal_waste=2, mode="average")
    assert spans(r) == [(0, 7)]
    assert r.intercepts.iloc[0]["grade"] == pytest.approx(1.42 / 7)  # 0.2029 >= 0.2
    # total internal waste 2 + 2 = 4 m > 3 m: the run ends at 4 m, the last ore is alone
    r = run(table, max_internal_waste=2, max_total_waste=3, mode="average")
    assert spans(r) == [(0, 4), (6, 7)]
    # a single stretch of 2 m > 1 m allowed: no waste is taken at all
    r = run(table, max_internal_waste=1, mode="average")
    assert spans(r) == [(0, 1), (3, 4), (6, 7)]
    assert spans(run(hole([1.0, 0.0, 0.0, 0.0, 1.0]), max_internal_waste=2, mode="average")) == [
        (0, 1),
        (4, 5),
    ]


def test_average_gaps_and_thickness_rules():
    # a bridged gap adds length but no grade; a long one cuts the hole
    table = hole([0.5, np.nan, 0.5])
    r = run(table, gap_tolerance=1, mode="average")
    assert spans(r) == [(0, 3)] and r.intercepts.iloc[0]["missing_length"] == 1.0
    assert r.intercepts.iloc[0]["grade"] == pytest.approx(0.5)
    assert spans(run(table, mode="average")) == [(0, 1), (2, 3)]
    # the minimum thickness applies to the run of the mode: 4 m passes, strict's 1 m do not
    table = hole([0.25, 0.1, 0.1, 1.0, 0.05])
    assert spans(run(table, max_internal_waste=2, min_thickness=3, mode="average")) == [(0, 4)]
    assert spans(run(table, max_internal_waste=2, min_thickness=3)) == []
    # including sub-intercepts use the same mode inside the intercept: at cut-off 0.5 the
    # only ore sample is 3-4 m
    r = run(table, max_internal_waste=2, mode="average", including_cutoffs=(0.5,))
    assert spans(r, "including") == [(3, 4)]


# --------------------------------------------------------------------- moving average


def test_moving_average_docstring_example():
    table = hole([0.05, 0.1, 0.9, 0.1, 0.05, 0.05])
    # window 3 m centred on each sample (cut at the ends of the hole):
    # 0: (0.05 + 0.1) / 2 = 0.075   1: 1.05 / 3 = 0.35   2: 1.1 / 3 = 0.3667
    # 3: 1.05 / 3 = 0.35            4: 0.2 / 3 = 0.0667  5: 0.1 / 2 = 0.05
    r = run(table, mode="moving-average", window=3)
    assert codes(r) == [(0, 1, "WASTE"), (1, 4, "MIN"), (4, 6, "WASTE")]
    row = r.intercepts.iloc[0]
    assert row["grade"] == pytest.approx(1.1 / 3) and row["waste_length"] == 2.0
    assert spans(run(table, max_internal_waste=2)) == [(2, 3)]
    assert spans(run(table, max_internal_waste=2, mode="average")) == [(2, 3)]
    # a window of one sample is the plain cut-off
    assert spans(run(table, mode="moving-average", window=1)) == [(2, 3)]


def test_moving_average_is_length_weighted_in_metres():
    # samples 0-2, 2-3, 3-4 m; window 2 m
    # 0-2 (middle 1.0): window 0-2 -> 0.1
    # 2-3 (middle 2.5): window 1.5-3.5 -> (0.5 * 0.1 + 1 * 0.6 + 0.5 * 0.1) / 2 = 0.35
    # 3-4 (middle 3.5): window 2.5-4.5 -> (0.5 * 0.6 + 1 * 0.1) / 1.5 = 0.2667
    table = hole([0.1, 0.6, 0.1], lengths=[2.0, 1.0, 1.0])
    r = run(table, mode="moving-average", window=2)
    assert spans(r) == [(2, 4)]
    assert r.intercepts.iloc[0]["grade"] == pytest.approx(0.35)
    # at the cut-off exactly: (0.5 * 0.6 + 1 * 0.1) / 1.5 = 0.2667 >= 0.2667 is in
    assert spans(run(table, mode="moving-average", window=2, cutoff=0.4 / 1.5)) == [(2, 4)]
    assert spans(run(table, mode="moving-average", window=2, cutoff=0.27)) == [(2, 3)]


def test_moving_average_ignores_the_internal_waste_limit():
    table = hole([0.9, 0.0, 0.0, 0.0, 0.9])
    # window 5 m: 0.9 / 3 = 0.3, 0.9 / 4 = 0.225, 1.8 / 5 = 0.36, 0.225, 0.3: all in
    r = run(table, mode="moving-average", window=5, max_internal_waste=0, max_total_waste=0)
    assert spans(r) == [(0, 5)] and r.intercepts.iloc[0]["waste_length"] == 3.0
    assert spans(run(table, mode="moving-average", window=3)) == [(0, 2), (3, 5)]


def test_moving_average_gaps_and_pieces():
    # a bridged gap has no grade and no weight; it joins two mineralised samples
    r = run(hole([0.5, np.nan, 0.5]), gap_tolerance=1, mode="moving-average", window=1)
    assert spans(r) == [(0, 3)] and r.intercepts.iloc[0]["missing_length"] == 1.0
    # ... but it is not kept at the end of a run
    r = run(hole([0.5, np.nan, 0.05]), gap_tolerance=1, mode="moving-average", window=1)
    assert spans(r) == [(0, 1)]
    # the window does not reach across a missing stretch that cuts the hole: 0.05 stays out
    r = run(hole([0.05, np.nan, 5.0]), mode="moving-average", window=5)
    assert spans(r) == [(2, 3)]
    # with the gap bridged the 5.0 carries its neighbour: (0.05 + 5.0) / 2 >= 0.2
    r = run(hole([0.05, np.nan, 5.0]), gap_tolerance=1, mode="moving-average", window=5)
    assert spans(r) == [(0, 3)]
    # thickness rule and sub-intercepts as in the other modes
    table = hole([0.05, 0.1, 0.9, 0.1, 0.05, 0.05])
    assert spans(run(table, mode="moving-average", window=3, min_thickness=4)) == []
    # sub-intercepts: the same rule on the samples of the intercept only.
    # means: 0.175, 1.25 / 3 = 0.4167, 0.7, 0.7, 0.4167, 0.175 -> intercept 1-5 m;
    # inside it (window cut at 1 m and 5 m): 1.2 / 2 = 0.6, 0.7, 0.7, 0.6 -> 2-4 m at 0.65
    table = hole([0.05, 0.3, 0.9, 0.9, 0.3, 0.05])
    r = run(table, mode="moving-average", window=3, including_cutoffs=(0.65,))
    assert spans(r) == [(1, 5)] and spans(r, "including") == [(2, 4)]


# ------------------------------------------------------------------------------- CLI


def test_cli_modes(tmp_path, capsys):
    from bakumine.core.drillholes.importer import DrillholeDatabase, load_database

    assays = pd.concat(
        [
            hole([0.25, 0.1, 0.1, 1.0, 0.05], hole_id="A"),
            hole([0.05, 0.1, 0.9, 0.1, 0.05, 0.05], hole_id="B"),
        ],
        ignore_index=True,
    )
    collar = pd.DataFrame(
        {"hole_id": ["A", "B"], "x": [0.0, 50.0], "y": 0.0, "z": 0.0, "max_depth": [5.0, 6.0]}
    )
    survey = pd.DataFrame({"hole_id": ["A", "B"], "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    folder = tmp_path / "db"
    DrillholeDatabase(collar, survey, {"assay": assays}).save(folder)
    lengths = {}
    for mode, extra in (
        ("strict", []),
        ("average", []),
        ("moving-average", ["--window", "3"]),
    ):
        argv = ["intercepts", str(folder), "--grade", "au", "--cutoff", "0.2",
                "--max-internal-waste", "2", "--mode", mode, *extra,
                "--save-as", f"dom_{mode.replace('-', '_')}",
                "--out", str(tmp_path / f"{mode}.csv")]  # fmt: skip
        assert main(argv) == 0
        out = capsys.readouterr().out
        assert f"au >= 0.2 ({mode})" in out
        table = load_database(folder).intervals[f"dom_{mode.replace('-', '_')}"]
        ore = table[table["code"] == "MIN"]
        lengths[mode] = float((ore["depth_to"] - ore["depth_from"]).sum())
    # A: strict 0-1 and 3-4, average 0-4, moving average 2-5 (means 0.175, 0.15, 0.4,
    # 0.383, 0.525);
    # B: strict 2-3, average 2-3, moving average 1-4
    assert lengths == {"strict": 3.0, "average": 5.0, "moving-average": 6.0}
    assert main(["intercepts", str(folder), "--grade", "au", "--cutoff", "0.2",
                 "--mode", "moving-average"]) == 2  # fmt: skip
    assert "needs a window" in capsys.readouterr().err


# ------------------------------------------------------------------------ properties

MODE_PARAMS = st.one_of(
    PARAMS.map(lambda p: dataclasses.replace(p, mode="average")),
    st.builds(
        lambda p, w: dataclasses.replace(p, mode="moving-average", window=w),
        PARAMS,
        st.sampled_from([0.5, 1.0, 3.0, 7.5]),
    ),
)


def window_mean(samples: pd.DataFrame, middle: float, window: float) -> float:
    """Length-weighted mean grade of the assayed samples within window / 2 of ``middle``."""
    lo, hi = middle - window / 2, middle + window / 2
    total = weight = 0.0
    for s in samples.dropna(subset=["au"]).itertuples():
        part = min(s.depth_to, hi) - max(s.depth_from, lo)
        if part > 1e-9:
            total += s.au * part
            weight += part
    return total / weight if weight else float("nan")


@settings(max_examples=300, deadline=None)
@given(assay_tables(), MODE_PARAMS)
def test_runs_of_the_other_modes_obey_their_rules(table, params):
    result = mineralised_intervals(table, "au", params)
    main_rows = result.intercepts[result.intercepts["level"] == "intercept"]
    ore = result.intervals[result.intervals[params.code_column] == params.ore_code]
    assert len(ore) == len(main_rows)
    for row in main_rows.itertuples():
        samples = inside(table, row)
        assayed = samples.dropna(subset=["au"])
        g = assayed["au"].to_numpy()
        length = (assayed["depth_to"] - assayed["depth_from"]).to_numpy()
        assert row.grade == pytest.approx(float((g * length).sum() / length.sum()))
        # a run starts and ends on an assayed sample
        assert assayed["depth_from"].iloc[0] == pytest.approx(row.depth_from)
        assert assayed["depth_to"].iloc[-1] == pytest.approx(row.depth_to)
        if params.mode == "average":
            assert g[0] >= params.cutoff and g[-1] >= params.cutoff
            assert row.grade >= params.cutoff - 1e-9
            # internal waste: every stretch between two ore samples that holds a waste
            # sample counts with the gaps inside it; single stretches and their total
            total, stretch, has_waste, depth = 0.0, 0.0, False, row.depth_from
            for s in samples.itertuples():
                stretch += s.depth_from - depth  # unsampled depth before this sample
                if not np.isnan(s.au) and s.au >= params.cutoff:
                    if has_waste:
                        assert stretch <= params.max_internal_waste + 1e-6
                        total += stretch
                    stretch, has_waste = 0.0, False
                else:
                    stretch += s.depth_to - s.depth_from
                    has_waste |= not np.isnan(s.au)
                depth = s.depth_to
            if params.max_total_waste is not None:
                assert total <= params.max_total_waste + 1e-6
        else:
            # every assayed sample of the run has a window mean at or above the cut-off,
            # the window taken over the samples of its piece
            hole_rows = table[table["hole_id"] == row.hole_id]
            piece = hole_piece(hole_rows, row, params.gap_tolerance)
            for s in assayed.itertuples():
                mean = window_mean(piece, 0.5 * (s.depth_from + s.depth_to), params.window)
                assert mean >= params.cutoff - 1e-7


def hole_piece(hole_rows: pd.DataFrame, row, gap_tolerance: float) -> pd.DataFrame:
    """The samples of the piece of the hole that holds ``row`` (cut at long missing stretches)."""
    t = hole_rows[hole_rows["depth_to"] > hole_rows["depth_from"]].sort_values("depth_from")
    keep, lo, hi = [], row.depth_from, row.depth_to
    rows = list(t.itertuples())
    # walk outwards from the run while the missing stretch between assayed samples is short
    assayed = [r for r in rows if not np.isnan(r.au)]
    inside_run = [r for r in assayed if r.depth_from >= lo - 1e-9 and r.depth_to <= hi + 1e-9]
    keep = list(inside_run)
    before = [r for r in assayed if r.depth_to <= lo + 1e-9][::-1]
    edge = lo
    for r in before:
        if edge - r.depth_to > gap_tolerance + 1e-9:
            break
        keep.append(r)
        edge = r.depth_from
    edge = hi
    for r in [r for r in assayed if r.depth_from >= hi - 1e-9]:
        if r.depth_from - edge > gap_tolerance + 1e-9:
            break
        keep.append(r)
        edge = r.depth_to
    return pd.DataFrame(
        [(r.depth_from, r.depth_to, r.au) for r in keep], columns=["depth_from", "depth_to", "au"]
    )


@settings(max_examples=200, deadline=None)
@given(assay_tables(), PARAMS)
def test_a_tiny_window_is_the_plain_cutoff(table, params):
    """Moving average over less than a sample = strict without internal waste."""
    plain = dataclasses.replace(params, max_internal_waste=0.0, max_total_waste=None)
    # the waste limits are set on both sides: the moving average does not use them for
    # the intercepts, but the "including" intervals are built with the strict rule
    tiny = dataclasses.replace(plain, mode="moving-average", window=0.01)
    a = mineralised_intervals(table, "au", plain)
    b = mineralised_intervals(table, "au", tiny)
    pd.testing.assert_frame_equal(a.intervals, b.intervals)
    pd.testing.assert_frame_equal(a.intercepts, b.intercepts)


# ------------------------------------------------------------------ "including" in every mode
@pytest.mark.parametrize("mode", MODES)
def test_including_intervals_are_the_high_grade_in_every_mode(mode):
    # 0.05, 0.1, 0.9, 0.1, 0.05, 0.05 (cut-off 0.2). Moving average, window 3: the run is
    # 1-4 m (module docstring); strict and average: 2-3 m. "Including" at 0.5 is found
    # with the strict rule on the samples of the intercept: the 0.9 sample at 2-3 m, never
    # the 0.1 samples beside it.
    window = {"window": 3} if mode == "moving-average" else {}
    r = run(hole([0.05, 0.1, 0.9, 0.1, 0.05, 0.05]), mode=mode, including_cutoffs=(0.5,),
            min_thickness=0.0, **window)  # fmt: skip
    if mode == "moving-average":
        assert spans(r) == [(1, 4)]
        assert spans(r, "including") == [(2, 3)]
        sub = r.intercepts[r.intercepts["level"] == "including"].iloc[0]
        assert sub["grade"] == pytest.approx(0.9) and sub["parent"] == 0
        assert sub["n_samples"] == 1 and sub["cutoff"] == 0.5
    else:
        assert spans(r) == [(2, 3)]
        assert spans(r, "including") == []  # it would be the whole intercept: left out


def test_including_intervals_start_and_end_at_their_cutoff():
    # two high-grade stretches inside one moving-average run; every "including" interval
    # starts and ends on a sample >= its cut-off and makes that cut-off
    grades = [0.05, 0.3, 1.2, 0.1, 0.1, 0.8, 0.9, 0.3, 0.05, 0.05]
    table = hole(grades)
    for mode, extra in (("strict", {}), ("average", {}), ("moving-average", {"window": 3})):
        r = run(table, mode=mode, including_cutoffs=(0.5,), min_thickness=0.0,
                max_internal_waste=2.0, **extra)  # fmt: skip
        inner = r.intercepts[r.intercepts["level"] == "including"]
        assert [(s.depth_from, s.depth_to) for s in inner.itertuples()] == [(2, 3), (5, 7)], mode
        for sub in inner.itertuples():
            assert grades[int(sub.depth_from)] >= 0.5 and grades[int(sub.depth_to) - 1] >= 0.5
            assert sub.grade >= 0.5
            parent = r.intercepts.iloc[sub.parent]
            assert parent["depth_from"] <= sub.depth_from and sub.depth_to <= parent["depth_to"]
