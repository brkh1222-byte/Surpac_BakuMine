"""Mineralised intervals and significant intercepts: hand-worked rules and properties.

Every expected number below is worked by hand in the comment next to it; the property
tests re-derive each rule from the returned tables with plain Python sums.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.drillholes.mineralised import (
    INTERCEPT_COLUMNS,
    InterceptParams,
    body_normal,
    mineralised_intervals,
)


def hole(grades, *, start=0.0, lengths=None, hole_id="A") -> pd.DataFrame:
    lengths = [1.0] * len(grades) if lengths is None else lengths
    to = start + np.cumsum(lengths)
    return pd.DataFrame(
        {"hole_id": hole_id, "depth_from": to - np.asarray(lengths), "depth_to": to, "au": grades}
    )


def run(table, **params):
    params.setdefault("cutoff", 0.2)
    return mineralised_intervals(table, "au", InterceptParams(**params))


def spans(result, level="intercept"):
    t = result.intercepts[result.intercepts["level"] == level]
    return [(r.depth_from, r.depth_to) for r in t.itertuples()]


def codes(result):
    return [(r.depth_from, r.depth_to, r.code) for r in result.intervals.itertuples()]


# ------------------------------------------------------------------ hand-worked rules


def test_docstring_example():
    # ore 1-3 (E = 0.3 + 0.1 = +0.4), waste 3-5 (2 m, E = -0.2), ore 5-6 (E = +0.4)
    r = run(hole([0.1, 0.5, 0.3, 0.1, 0.1, 0.6, 0.05]), min_thickness=2, max_internal_waste=2)
    assert codes(r) == [(0, 1, "WASTE"), (1, 6, "MIN"), (6, 7, "WASTE")]
    row = r.intercepts.iloc[0]
    assert row["length"] == 5.0 and row["n_samples"] == 5
    assert row["grade"] == pytest.approx(1.6 / 5)
    assert row["grade_x_m"] == pytest.approx(1.6)
    assert row["waste_length"] == pytest.approx(2.0) and row["missing_length"] == 0.0
    assert row["rule"] == "thickness" and row["parent"] == -1
    assert list(r.intercepts.columns) == list(INTERCEPT_COLUMNS)


def test_cutoff_is_inclusive_and_runs_are_contiguous_ore():
    r = run(hole([0.2, 0.19, 0.2, 0.2]))  # no internal waste allowed
    assert spans(r) == [(0, 1), (2, 4)]


def test_minimum_thickness():
    table = hole([0.5, 0.0, 0.0, 0.0, 0.5, 0.5])
    assert spans(run(table, min_thickness=2)) == [(4, 6)]  # the 1 m run is rejected
    assert codes(run(table, min_thickness=2)) == [(0, 4, "WASTE"), (4, 6, "MIN")]
    assert spans(run(table, min_thickness=2.5)) == []


def test_minimum_thickness_of_summed_float_lengths():
    # 0.7 + 0.7 + 0.6 is 1.9999999999999998 in floating point; it is 2 m
    assert spans(run(hole([1, 1, 1], lengths=[0.7, 0.7, 0.6]), min_thickness=2)) == [
        (0, pytest.approx(2.0))
    ]


def test_internal_waste_run_length():
    table = hole([1.0, 0.0, 0.0, 1.0])  # 2 m of waste between two 1 m ore samples
    assert spans(run(table, max_internal_waste=2)) == [(0, 4)]
    assert spans(run(table, max_internal_waste=1.99)) == [(0, 1), (3, 4)]
    assert spans(run(table)) == [(0, 1), (3, 4)]  # default: no internal waste


def test_total_internal_waste():
    table = hole([1.0, 0.0, 1.0, 0.0, 1.0])  # two 1 m waste runs
    assert spans(run(table, max_internal_waste=1)) == [(0, 5)]
    assert spans(run(table, max_internal_waste=1, max_total_waste=1)) == [(0, 3), (4, 5)]
    assert spans(run(table, max_internal_waste=1, max_total_waste=2)) == [(0, 5)]


def test_waste_must_be_carried_by_the_run_before_it():
    # ore 0-1: E = (0.25 - 0.2) * 1 = +0.05; waste 1-2: E = -0.2; ore 2-3: E = +9.8.
    # The whole 0-3 averages 3.42, but 0-2 averages 0.125 < 0.2: waste at the edge.
    r = run(hole([0.25, 0.0, 10.0]), max_internal_waste=5)
    assert spans(r) == [(0, 1), (2, 3)]


def test_waste_must_be_carried_by_the_ore_after_it():
    # mirror image: 1-3 would average 0.125
    r = run(hole([10.0, 0.0, 0.25]), max_internal_waste=5)
    assert spans(r) == [(0, 1), (2, 3)]


def test_grade_including_internal_waste_meets_the_cutoff():
    # 0.3 | 0.1 | 0.3: E = +0.1 - 0.1 = 0 at the end of the waste: exactly at cut-off, kept
    assert spans(run(hole([0.3, 0.1, 0.3]), max_internal_waste=1)) == [(0, 3)]
    # 0.29: E = +0.09 - 0.1 < 0: split
    assert spans(run(hole([0.29, 0.1, 0.3]), max_internal_waste=1)) == [(0, 1), (2, 3)]


def test_accumulation_rule_for_short_high_grade():
    table = hole([0.0, 0.5, 0.0])  # 1 m at 0.5: 0.5 g x m >= 0.2 * 2 = 0.4
    assert spans(run(table, min_thickness=2)) == []
    r = run(table, min_thickness=2, accumulation_rule=True)
    assert spans(r) == [(1, 2)] and r.intercepts.iloc[0]["rule"] == "accumulation"
    # 1 m at 0.39: 0.39 < 0.4
    assert spans(run(hole([0.0, 0.39, 0.0]), min_thickness=2, accumulation_rule=True)) == []


def test_missing_assay_ends_a_run_and_is_not_zero():
    table = hole([0.5, np.nan, 0.5])
    r = run(table, max_internal_waste=5)
    assert spans(r) == [(0, 1), (2, 3)]
    # the missing metre is neither mineralised nor waste
    assert codes(r) == [(0, 1, "MIN"), (2, 3, "MIN")]
    # as zero it would have been internal waste: (0.5 + 0 + 0.5) / 3 = 0.33
    assert spans(run(hole([0.5, 0.0, 0.5]), max_internal_waste=5)) == [(0, 3)]


def test_gap_tolerance_bridges_short_missing_stretches():
    table = hole([0.5, np.nan, 0.5])
    r = run(table, gap_tolerance=1, min_thickness=3)
    assert spans(r) == [(0, 3)]
    row = r.intercepts.iloc[0]
    # grade over the assayed 2 m only; the bridged metre adds length, not metal
    assert row["grade"] == pytest.approx(0.5) and row["grade_x_m"] == pytest.approx(1.0)
    assert row["assayed_length"] == 2.0 and row["missing_length"] == pytest.approx(1.0)
    assert row["waste_length"] == 0.0
    assert spans(run(table, gap_tolerance=0.99)) == [(0, 1), (2, 3)]


def test_unsampled_depth_is_missing_too():
    table = pd.concat([hole([0.5, 0.5]), hole([0.5], start=2.5)])  # nothing from 2.0 to 2.5
    assert spans(run(table)) == [(0, 2), (2.5, 3.5)]
    assert spans(run(table, gap_tolerance=0.5)) == [(0, 3.5)]
    # an unsampled stretch next to a missing assay is one missing stretch: 0.5 + 1 > 1
    table = pd.concat([hole([0.5, 0.5]), hole([np.nan, 0.5], start=2.5)])
    assert spans(run(table, gap_tolerance=1)) == [(0, 2), (3.5, 4.5)]


def test_gap_inside_waste_counts_as_waste_length():
    # waste 1-2, missing 2-2.5, waste 2.5-3.5: an internal stretch of 2.5 m
    table = pd.concat([hole([5.0, 0.0]), hole([0.0, 5.0], start=2.5)])
    assert spans(run(table, gap_tolerance=0.5, max_internal_waste=2.5)) == [(0, 4.5)]
    assert spans(run(table, gap_tolerance=0.5, max_internal_waste=2.4)) == [(0, 1), (3.5, 4.5)]


def test_top_cut_is_reporting_only():
    r = run(hole([0.3, 10.0]), top_cut=2.0)
    row = r.intercepts.iloc[0]
    assert row["grade"] == pytest.approx(5.15) and row["grade_cut"] == pytest.approx(1.15)
    assert math.isnan(run(hole([0.3, 10.0])).intercepts.iloc[0]["grade_cut"])


def test_including_sub_intercepts():
    r = run(
        hole([0.3, 2.0, 3.0, 0.3, 0.3, 5.0, 0.3]),
        including_cutoffs=(1.0,),
        including_min_thickness=2,
    )
    assert spans(r) == [(0, 7)]
    assert spans(r, "including") == [(1, 3)]  # the single metre at 5.0 is too thin
    sub = r.intercepts.iloc[1]
    assert sub["parent"] == 0 and sub["cutoff"] == 1.0 and sub["grade"] == pytest.approx(2.5)
    r = run(hole([0.3, 2.0, 0.3]), including_cutoffs=(1.0, 0.25))
    # at 0.25 the sub-intercept would be the whole parent: left out
    assert spans(r, "including") == [(1, 2)]


def test_true_thickness_against_analytic_directions():
    collar = pd.DataFrame(
        {"hole_id": ["V", "I"], "x": 0.0, "y": 0.0, "z": 100.0, "max_depth": 50.0}
    )
    survey = pd.DataFrame(
        {"hole_id": ["V", "I"], "depth": 0.0, "dip": [-90.0, -60.0], "azimuth": [0.0, 90.0]}
    )
    traces = desurvey(collar, survey)
    table = pd.concat([hole([1.0] * 4, hole_id="V"), hole([1.0] * 4, hole_id="I")])
    # body dipping 30 degrees to the west: normal (-1/2, 0, sqrt(3)/2). The inclined hole
    # (east, 60 degrees down) is perpendicular to it; the vertical one makes 30 degrees.
    params = InterceptParams(0.2, min_thickness=3.6, body_dip=30, body_dip_azimuth=270)
    r = mineralised_intervals(table, "au", params, traces=traces)
    assert list(r.intercepts["hole_id"]) == ["I"]  # V: 4 * cos(30) = 3.46 < 3.6
    assert r.intercepts.iloc[0]["true_thickness"] == pytest.approx(4.0)
    params = InterceptParams(0.2, min_thickness=3.4, body_dip=30, body_dip_azimuth=270)
    r = mineralised_intervals(table, "au", params, traces=traces)
    assert dict(zip(r.intercepts["hole_id"], r.intercepts["true_thickness"], strict=True)) == {
        "I": pytest.approx(4.0),
        "V": pytest.approx(4 * math.cos(math.radians(30))),
    }
    np.testing.assert_allclose(body_normal(0, 123), [0, 0, 1], atol=1e-15)
    np.testing.assert_allclose(body_normal(90, 90), [1, 0, 0], atol=1e-15)
    with pytest.raises(ValueError, match="traces"):
        mineralised_intervals(table, "au", params)


def test_true_thickness_past_the_end_of_a_trace_and_without_one():
    collar = pd.DataFrame(
        {"hole_id": ["I", "C"], "x": 0.0, "y": 0.0, "z": 100.0, "max_depth": [8.0, 20.0]}
    )
    # I: straight, 60 degrees down to the east. C: turns from vertical at 0 m to 60
    # degrees down to the east at its end (20 m).
    survey = pd.DataFrame(
        {"hole_id": ["I", "C", "C"], "depth": [0.0, 0.0, 20.0], "dip": [-60.0, -90.0, -60.0],
         "azimuth": 90.0}  # fmt: skip
    )
    traces = desurvey(collar, survey)
    # body dipping 30 degrees to the west: the direction (east, 60 down) is its normal
    params = InterceptParams(0.2, min_thickness=1.0, body_dip=30, body_dip_azimuth=270)
    table = pd.concat(
        [
            hole([1.0] * 4, start=7.0, hole_id="I"),  # 7-11 m: middle 9 m, the hole ends at 8
            hole([1.0] * 4, start=22.0, hole_id="C"),  # 22-26 m, below the end at 20 m
            hole([1.0] * 4, hole_id="X"),  # no collar, no survey: judged down-hole
        ]
    )
    with pytest.warns(UserWarning) as caught:
        r = mineralised_intervals(table, "au", params, traces=traces)
    messages = [str(w.message) for w in caught]
    below = [m for m in messages if "below the stated hole depth" in m]
    none = [m for m in messages if "no desurveyed direction" in m]
    assert len(below) == 1 and "2 hole(s) (C, I)" in below[0]
    assert len(none) == 1 and "1 hole(s) (X)" in none[0]
    thickness = dict(zip(r.intercepts["hole_id"], r.intercepts["true_thickness"], strict=True))
    assert list(thickness) == ["C", "I", "X"]
    assert thickness["I"] == pytest.approx(4.0)  # the straight hole goes on as it went
    # the curved hole continues on its direction at the end (perpendicular to the body),
    # not on the direction somewhere up the hole
    assert thickness["C"] == pytest.approx(4.0)
    assert math.isnan(thickness["X"])
    rules = dict(zip(r.intercepts["hole_id"], r.intercepts["rule"], strict=True))
    assert rules == {"C": "thickness", "I": "thickness", "X": "thickness"}
    # an intercept inside the trace gives no warning
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        inside_trace = mineralised_intervals(
            hole([1.0] * 4, hole_id="I"), "au", params, traces=traces
        )
    assert inside_trace.intercepts["true_thickness"].iloc[0] == pytest.approx(4.0)
    assert inside_trace.warnings == [] and r.warnings == messages


def test_cli_says_which_holes_have_intercepts_below_their_depth(tmp_path, capsys):
    from bakumine.cli import main
    from bakumine.core.drillholes.importer import DrillholeDatabase

    collar = pd.DataFrame({"hole_id": ["I"], "x": 0.0, "y": 0.0, "z": 100.0, "max_depth": 8.0})
    survey = pd.DataFrame({"hole_id": ["I"], "depth": 0.0, "dip": -60.0, "azimuth": 90.0})
    assays = hole([1.0] * 4, start=7.0, hole_id="I")
    folder = tmp_path / "db"
    DrillholeDatabase(collar, survey, {"assay": assays}).save(folder)
    code = main(["intercepts", str(folder), "--grade", "au", "--cutoff", "0.2",
                 "--body-dip", "30", "--body-dip-azimuth", "270"])  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0
    assert "WARNING: true thickness: intercepts of 1 hole(s) (I) lie below the stated" in out
    table = pd.read_csv(folder / "intercepts_assay.csv")
    assert table["true_thickness"].tolist() == pytest.approx([4.0])


def test_input_errors():
    with pytest.raises(ValueError, match="overlap"):
        run(pd.concat([hole([1.0, 1.0]), hole([1.0], start=1.5)]))
    with pytest.raises(ValueError, match="no column"):
        mineralised_intervals(hole([1.0]), "cu", InterceptParams(0.2))
    with pytest.raises(ValueError, match="not numeric"):
        mineralised_intervals(hole(["a"]), "au", InterceptParams(0.2))
    for bad in (
        {"cutoff": float("nan")},
        {"cutoff": 0.2, "min_thickness": -1},
        {"cutoff": 0.2, "including_cutoffs": (0.1,)},
        {"cutoff": 0.2, "body_dip": 30},
        {"cutoff": 0.2, "ore_code": "X", "waste_code": "X"},
    ):
        with pytest.raises(ValueError):
            InterceptParams(**bad)


def test_parameter_file_round_trip(tmp_path):
    params = InterceptParams(
        0.2, 2.0, 2.0, 4.0, True, 0.5, 30.0, (1.0, 5.0), 1.0, 45.0, 270.0, "ORE", "W", "zone"
    )
    assert InterceptParams.load(params.save(tmp_path / "p.json")) == params
    assert InterceptParams.from_dict({"cutoff": 0.2}) == InterceptParams(0.2)
    with pytest.raises(ValueError, match="unknown"):
        InterceptParams.from_dict({"cutoff": 0.2, "cut_off": 1})
    with pytest.raises(ValueError, match="cut-off"):
        InterceptParams.from_dict({})


def test_empty_and_all_waste_tables():
    r = run(hole([0.0, 0.1]))
    assert len(r.intercepts) == 0 and codes(r) == [(0, 2, "WASTE")]
    assert r.summary()["n_intercepts"] == 0 and r.summary()["mean_grade"] is None
    r = run(hole([]))
    assert len(r.intervals) == 0 and len(r.intercepts) == 0


# ------------------------------------------------------------------------- properties

LENGTHS = st.sampled_from([0.5, 1.0, 1.5, 2.0, 0.3])
GRADES = st.one_of(
    st.sampled_from([0.0, 0.05, 0.1, 0.19, 0.2, 0.21, 0.3, 0.5, 1.0, 5.0, 40.0]),
    st.floats(0.0, 3.0, allow_nan=False),
    st.just(float("nan")),
)
SAMPLE = st.tuples(st.sampled_from([0.0, 0.0, 0.0, 0.4, 1.0, 3.0]), LENGTHS, GRADES)


@st.composite
def assay_tables(draw):
    rows = []
    for h in range(draw(st.integers(1, 3))):
        depth = draw(st.sampled_from([0.0, 2.5]))
        for gap, length, grade in draw(st.lists(SAMPLE, min_size=0, max_size=14)):
            depth += gap
            rows.append((f"H{h}", depth, depth + length, grade))
            depth += length
    return pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "au"]).astype(
        {"hole_id": object, "depth_from": float, "depth_to": float, "au": float}
    )


PARAMS = st.builds(
    InterceptParams,
    cutoff=st.sampled_from([0.2, 0.2, 0.5, 0.0]),
    min_thickness=st.sampled_from([0.0, 1.0, 2.0, 3.5]),
    max_internal_waste=st.sampled_from([0.0, 1.0, 2.0, 5.0]),
    max_total_waste=st.sampled_from([None, None, 1.0, 3.0]),
    accumulation_rule=st.booleans(),
    gap_tolerance=st.sampled_from([0.0, 0.4, 1.0]),
    top_cut=st.sampled_from([None, 2.0]),
    including_cutoffs=st.sampled_from([(), (1.0,), (0.6, 2.0)]),
    including_min_thickness=st.sampled_from([None, 0.0, 1.0]),
)
EPS = 1e-7


def inside(table: pd.DataFrame, row) -> pd.DataFrame:
    """Samples of the hole within [depth_from, depth_to] of ``row``, by depth."""
    t = table[
        (table["hole_id"] == row.hole_id)
        & (table["depth_from"] >= row.depth_from - 1e-9)
        & (table["depth_to"] <= row.depth_to + 1e-9)
    ]
    return t.sort_values("depth_from")


def check_run(table: pd.DataFrame, row, params: InterceptParams, min_thickness: float) -> None:
    """Every rule, re-derived from the samples of one intercept row."""
    cutoff = row.cutoff
    samples = inside(table, row)
    f, t, g = (samples[c].to_numpy() for c in ("depth_from", "depth_to", "au"))
    # starts and ends on an ore sample
    assert f[0] == row.depth_from and t[-1] == row.depth_to
    assert g[0] >= cutoff and g[-1] >= cutoff
    # the reported numbers are those of the assays (metal is conserved)
    ok = ~np.isnan(g)
    metal = float((g[ok] * (t - f)[ok]).sum())
    assert row.grade_x_m == pytest.approx(metal, rel=1e-9, abs=1e-12)
    assert row.assayed_length == pytest.approx(float((t - f)[ok].sum()))
    assert row.length == pytest.approx(row.depth_to - row.depth_from)
    assert row.missing_length == pytest.approx(row.length - row.assayed_length, abs=1e-9)
    assert row.grade == pytest.approx(metal / row.assayed_length)
    assert row.n_samples == int(ok.sum())
    # cut-off, also from either end (no waste at the edges)
    assert row.grade >= cutoff - EPS
    excess = np.where(ok, (np.where(ok, g, 0) - cutoff) * (t - f), 0.0)
    assert np.cumsum(excess).min() >= -EPS
    assert np.cumsum(excess[::-1]).min() >= -EPS
    # thickness or accumulation
    if row.rule == "thickness":
        assert row.length >= min_thickness - EPS
    else:
        assert row.rule == "accumulation" and params.accumulation_rule
        assert row.length < min_thickness
        assert row.grade_x_m >= cutoff * min_thickness - EPS
    # internal waste and missing stretches, walked sample by sample
    waste_total = stretch = missing = 0.0
    has_waste = False
    depth = f[0]
    for a, b, grade in zip(f, t, g, strict=True):
        hole_gap = a - depth
        depth = b
        missing += hole_gap
        if math.isnan(grade):
            missing += b - a  # pending until the next assayed sample
            continue
        assert missing <= params.gap_tolerance + EPS
        if grade >= cutoff:
            if has_waste:  # the stretch since the last ore sample, gaps included
                stretch += missing
                assert stretch <= params.max_internal_waste + EPS
                waste_total += stretch
            stretch, has_waste = 0.0, False
        else:
            stretch += missing + (b - a)
            has_waste = True
        missing = 0.0
    if params.max_total_waste is not None:
        assert waste_total <= params.max_total_waste + EPS
    assert row.waste_length == pytest.approx(
        float((t - f)[ok & (np.nan_to_num(g, nan=np.inf) < cutoff)].sum())
    )
    if params.top_cut is None:
        assert math.isnan(row.grade_cut)
    else:
        cut = float((np.minimum(g[ok], params.top_cut) * (t - f)[ok]).sum()) / row.assayed_length
        assert row.grade_cut == pytest.approx(cut)
        assert row.grade_cut <= row.grade + EPS


@settings(max_examples=300, deadline=None)
@given(assay_tables(), PARAMS)
def test_accepted_runs_obey_every_rule(table, params):
    result = mineralised_intervals(table, "au", params)
    inner_min = (
        params.min_thickness
        if params.including_min_thickness is None
        else params.including_min_thickness
    )
    main = result.intercepts[result.intercepts["level"] == "intercept"]
    for row in result.intercepts.itertuples():
        if row.level == "intercept":
            assert row.parent == -1 and row.cutoff == params.cutoff
            check_run(table, row, params, params.min_thickness)
        else:
            parent = result.intercepts.iloc[row.parent]
            assert parent["level"] == "intercept" and parent["hole_id"] == row.hole_id
            assert parent["depth_from"] <= row.depth_from and row.depth_to <= parent["depth_to"]
            assert (row.depth_from, row.depth_to) != (parent["depth_from"], parent["depth_to"])
            assert row.cutoff in params.including_cutoffs
            check_run(table, row, params, inner_min)
    # the interval table: MIN rows are the intercepts; no overlaps; no row over a long gap
    code = result.intervals[params.code_column]
    assert set(code) <= {"MIN", "WASTE"}
    ore = result.intervals[code == "MIN"]
    assert [tuple(r) for r in ore[["hole_id", "depth_from", "depth_to"]].to_numpy()] == [
        tuple(r) for r in main[["hole_id", "depth_from", "depth_to"]].to_numpy()
    ]
    for _, g in result.intervals.groupby("hole_id"):
        assert (g["depth_from"].to_numpy()[1:] >= g["depth_to"].to_numpy()[:-1] - 1e-9).all()
        assert (g["depth_to"] > g["depth_from"]).all()


@settings(max_examples=300, deadline=None)
@given(assay_tables(), PARAMS)
def test_metal_is_conserved(table, params):
    result = mineralised_intervals(table, "au", params)
    length = table["depth_to"] - table["depth_from"]
    total = float((table["au"] * length).sum(skipna=True))
    by_code = {"MIN": 0.0, "WASTE": 0.0}
    covered = 0
    for row in result.intervals.itertuples():
        samples = inside(table, row).dropna(subset=["au"])
        covered += len(samples)
        by_code[getattr(row, params.code_column)] += float(
            (samples["au"] * (samples["depth_to"] - samples["depth_from"])).sum()
        )
    main = result.intercepts[result.intercepts["level"] == "intercept"]
    # every assayed sample is in exactly one coded interval
    assert covered == int(table["au"].notna().sum())
    assert by_code["MIN"] + by_code["WASTE"] == pytest.approx(total, rel=1e-9, abs=1e-9)
    assert float(main["grade_x_m"].sum()) == pytest.approx(by_code["MIN"], rel=1e-9, abs=1e-9)
    assert result.summary()["grade_x_m"] == pytest.approx(by_code["MIN"], rel=1e-9, abs=1e-9)
    # no ore-grade sample is left as waste unless a thickness rule rejected its run
    if params.min_thickness == 0:
        for row in result.intervals[result.intervals[params.code_column] == "WASTE"].itertuples():
            assert not (inside(table, row)["au"] >= params.cutoff).any()


@settings(max_examples=200, deadline=None)
@given(assay_tables(), PARAMS, st.randoms(use_true_random=False))
def test_result_does_not_depend_on_row_order(table, params, rng):
    order = list(range(len(table)))
    rng.shuffle(order)
    a = mineralised_intervals(table, "au", params)
    b = mineralised_intervals(table.iloc[order], "au", params)
    pd.testing.assert_frame_equal(a.intervals, b.intervals)
    pd.testing.assert_frame_equal(a.intercepts, b.intercepts)
