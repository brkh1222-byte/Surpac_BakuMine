"""Compositing (P1.T5): hand-worked examples, conservation, geometry, properties."""

import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.optimize import brentq

from bakumine.core.drillholes.composite import (
    IntervalDomains,
    bench_crossings,
    composite,
    split_at_domains,
)
from bakumine.core.drillholes.desurvey import desurvey

SYNTH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
NAN = float("nan")


# --------------------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def synth():
    tables = {
        name: pd.read_csv(SYNTH / f"{name}.csv", dtype={"hole_id": str})
        for name in ("collar", "survey", "assay", "geology", "truth_samples")
    }
    tables["traces"] = desurvey(tables["collar"], tables["survey"])
    return tables


def hand_samples() -> pd.DataFrame:
    """One hole: a NaN sample at 5-6 and an unsampled gap at 6-7."""
    return pd.DataFrame(
        {
            "hole_id": ["H"] * 5,
            "depth_from": [0.0, 2.0, 5.0, 7.0, 10.0],
            "depth_to": [2.0, 5.0, 6.0, 10.0, 11.5],
            "g": [1.0, 3.0, NAN, 2.0, 4.0],
        }
    )


def straight_trace(dip: float, z: float = 100.0, depth: float = 20.0):
    collar = pd.DataFrame({"hole_id": ["H"], "x": [0.0], "y": [0.0], "z": [z],
                           "max_depth": [depth]})
    survey = pd.DataFrame({"hole_id": ["H"], "depth": [0.0], "dip": [dip], "azimuth": [90.0]})
    return desurvey(collar, survey)


def curved_trace():
    """A hole that steepens, flattens, and finally climbs: z is not monotone."""
    collar = pd.DataFrame({"hole_id": ["C"], "x": [10.0], "y": [20.0], "z": [103.3],
                           "max_depth": [400.0]})
    survey = pd.DataFrame(
        {
            "hole_id": ["C"] * 6,
            "depth": [0.0, 80.0, 160.0, 240.0, 320.0, 400.0],
            "dip": [-75.0, -50.0, -20.0, 5.0, 25.0, 40.0],
            "azimuth": [10.0, 25.0, 40.0, 60.0, 75.0, 90.0],
        }
    )
    return desurvey(collar, survey)


def accumulation(table: pd.DataFrame, attribute: str, weight: np.ndarray) -> pd.Series:
    """Sum of grade x length per hole over the rows where the grade exists."""
    value = table[attribute].to_numpy(dtype=float)
    product = np.where(np.isfinite(value), value * weight, 0.0)
    return pd.Series(product).groupby(table["hole_id"].to_numpy()).sum()


def assert_conserved(samples: pd.DataFrame, composites: pd.DataFrame, attributes) -> None:
    for attribute in attributes:
        length = (samples["depth_to"] - samples["depth_from"]).to_numpy()
        before = accumulation(samples, attribute, length)
        after = accumulation(
            composites, attribute, composites[f"{attribute}_coverage"].to_numpy()
        ).reindex(before.index, fill_value=0.0)
        scale = np.maximum(np.abs(before.to_numpy()), 1e-300)
        relative = np.abs(after.to_numpy() - before.to_numpy()) / scale
        relative[before.to_numpy() == 0.0] = np.abs(after.to_numpy())[before.to_numpy() == 0.0]
        assert relative.max() < 1e-9, (attribute, relative.max())


# --------------------------------------------------------------------------------------
# hand-worked examples
# --------------------------------------------------------------------------------------
def test_downhole_hand_example():
    out = composite(hand_samples(), length=4.0)
    # Boundaries from the collar: 0-4, 4-8, 8-11.5 (end of the deepest sample).
    np.testing.assert_allclose(out["depth_from"], [0.0, 4.0, 8.0])
    np.testing.assert_allclose(out["depth_to"], [4.0, 8.0, 11.5])
    np.testing.assert_allclose(out["length"], [4.0, 4.0, 3.5])
    # 0-4 : sample 0-2 (g 1, 2 m) + first 2 m of sample 2-5 (g 3)  -> (1*2 + 3*2) / 4 = 2
    # 4-8 : last 1 m of sample 2-5 (g 3) + NaN 5-6 + gap 6-7 + first 1 m of 7-10 (g 2)
    #       -> (3*1 + 2*1) / 2 = 2.5 on 2 m of support
    # 8-11.5 : last 2 m of 7-10 (g 2) + 10-11.5 (g 4) -> (2*2 + 4*1.5) / 3.5 = 10 / 3.5
    np.testing.assert_allclose(out["g"], [2.0, 2.5, 10.0 / 3.5], rtol=1e-14)
    np.testing.assert_allclose(out["g_coverage"], [4.0, 2.0, 3.5], rtol=1e-14)
    assert np.isnan(out[["x", "y", "z", "dx", "dy", "dz"]].to_numpy()).all()  # no traces given
    assert "domain" not in out.columns


def test_missing_is_never_zero_unless_asked():
    # Same 4-8 composite with the NaN metre and the gap metre counted as grade 0:
    # (3*1 + 0*1 + 0*1 + 2*1) / 4 = 1.25 on 4 m.
    out = composite(hand_samples(), length=4.0, fill_missing=0.0)
    np.testing.assert_allclose(out["g"], [2.0, 1.25, 10.0 / 3.5], rtol=1e-14)
    np.testing.assert_allclose(out["g_coverage"], [4.0, 4.0, 3.5], rtol=1e-14)
    # A non-zero fill: (3*1 + 0.5*2 + 2*1) / 4 = 1.5
    out = composite(hand_samples(), length=4.0, fill_missing={"g": 0.5})
    np.testing.assert_allclose(out["g"], [2.0, 1.5, 10.0 / 3.5], rtol=1e-14)


def test_min_coverage_sets_attribute_to_nan():
    # Composite 4-8 has 2 m of support in 4 m = 0.5 < 0.6.
    out = composite(hand_samples(), length=4.0, min_coverage=0.6)
    assert out["g"].isna().tolist() == [False, True, False]
    np.testing.assert_allclose(out["g_coverage"], [4.0, 2.0, 3.5])
    out = composite(hand_samples(), length=4.0, min_coverage=0.5)
    assert not out["g"].isna().any()


def test_bench_hand_example():
    # Straight hole dipping 30 degrees from a collar at z = 100.5: z falls 0.5 m per metre.
    # Levels every 2.5 m (reference 0): z = 100 at depth 1, 97.5 at depth 6, 95 at depth 11.
    traces = straight_trace(-30.0, z=100.5)
    out = composite(hand_samples(), traces, mode="bench", bench_height=2.5)
    np.testing.assert_allclose(out["depth_from"], [0.0, 1.0, 6.0, 11.0], atol=1e-9)
    np.testing.assert_allclose(out["depth_to"], [1.0, 6.0, 11.0, 11.5], atol=1e-9)
    # 0-1    : 1 m of sample 0-2 (g 1)                                    -> 1 on 1 m
    # 1-6    : 1 m of g 1 + 3 m of g 3 + NaN 5-6   -> (1 + 9) / 4 = 2.5   on 4 m
    # 6-11   : gap 6-7 + 3 m of g 2 + 1 m of g 4   -> (6 + 4) / 4 = 2.5   on 4 m
    # 11-11.5: 0.5 m of g 4                                               -> 4 on 0.5 m
    np.testing.assert_allclose(out["g"], [1.0, 2.5, 2.5, 4.0], rtol=1e-9)
    np.testing.assert_allclose(out["g_coverage"], [1.0, 4.0, 4.0, 0.5], rtol=1e-9)
    # Midpoint of 1-6 is depth 3.5: x = 3.5 cos 30, z = 100.5 - 1.75.
    np.testing.assert_allclose(out["x"][1], 3.5 * np.cos(np.radians(30.0)), atol=1e-9)
    np.testing.assert_allclose(out["z"][1], 98.75, atol=1e-9)
    # A different reference elevation moves the levels: 99.5, 97, 94.5 -> depths 2, 7, 12.
    out = composite(hand_samples(), traces, mode="bench", bench_height=2.5, bench_reference=2.0)
    np.testing.assert_allclose(out["depth_from"], [0.0, 2.0, 7.0], atol=1e-9)
    # 0-2: g 1.   2-7: 3 m of g 3 (NaN and gap skipped) -> 3.   7-11.5: (2*3 + 4*1.5)/4.5
    np.testing.assert_allclose(out["g"], [1.0, 3.0, 12.0 / 4.5], rtol=1e-9)


GEOLOGY = pd.DataFrame(
    {
        "hole_id": ["H"] * 3,
        "depth_from": [0.0, 3.0, 9.0],
        "depth_to": [3.0, 9.0, 20.0],
        "rock": ["A", "B", "A"],
    }
)


def test_domain_hand_example():
    out = composite(
        hand_samples(), mode="domain", length=4.0, domain=IntervalDomains(GEOLOGY, "rock")
    )
    # Runs: A 0-3, B 3-9, A 9-20. Sample 2-5 (g 3) is split at the contact at 3.
    # A 0-3  : (1*2 + 3*1) / 3 = 5/3                             on 3 m
    # B 3-7  : 2 m of g 3 + NaN 5-6 + gap 6-7 -> 3               on 2 m
    # B 7-9  : 2 m of g 2 -> 2                                   on 2 m
    # A 9-13 : 1 m of g 2 + 1.5 m of g 4 -> (2 + 6) / 2.5 = 3.2  on 2.5 m
    # A 13-17 and 17-20 have no sample at all and are left out.
    assert out["domain"].tolist() == ["A", "B", "B", "A"]
    np.testing.assert_allclose(out["depth_from"], [0.0, 3.0, 7.0, 9.0])
    np.testing.assert_allclose(out["depth_to"], [3.0, 7.0, 9.0, 13.0])
    np.testing.assert_allclose(out["g"], [5.0 / 3.0, 3.0, 2.0, 3.2], rtol=1e-14)
    np.testing.assert_allclose(out["g_coverage"], [3.0, 2.0, 2.0, 2.5], rtol=1e-14)
    everything = composite(
        hand_samples(),
        mode="domain",
        length=4.0,
        domain=IntervalDomains(GEOLOGY, "rock"),
        drop_empty=False,
    )
    np.testing.assert_allclose(everything["depth_to"], [3.0, 7.0, 9.0, 13.0, 17.0, 20.0])
    assert everything["g"].isna().tolist() == [False, False, False, False, True, True]


def test_best_fit_hand_example():
    out = composite(
        hand_samples(),
        mode="domain",
        length=2.5,
        best_fit=True,
        domain=IntervalDomains(GEOLOGY, "rock"),
    )
    # A 0-3  : 3 / 2.5 = 1.2  -> 1 composite of 3 m
    # B 3-9  : 6 / 2.5 = 2.4  -> 2 composites of 3 m: 3-6, 6-9
    # A 9-20 : 11 / 2.5 = 4.4 -> 4 composites of 2.75 m; only 9-11.75 has samples.
    np.testing.assert_allclose(out["depth_from"], [0.0, 3.0, 6.0, 9.0])
    np.testing.assert_allclose(out["depth_to"], [3.0, 6.0, 9.0, 11.75])
    # 3-6: 2 m of g 3 (+ NaN) -> 3.  6-9: gap + 2 m of g 2 -> 2.  9-11.75: (2 + 6) / 2.5.
    np.testing.assert_allclose(out["g"], [5.0 / 3.0, 3.0, 2.0, 3.2], rtol=1e-14)
    # Without domains the run is the hole 0-11.5: 11.5 / 2.5 = 4.6 -> 5 composites of 2.3 m.
    plain = composite(hand_samples(), length=2.5, best_fit=True)
    np.testing.assert_allclose(plain["length"], [2.3] * 5, rtol=1e-12)
    np.testing.assert_allclose(plain["depth_to"].iloc[-1], 11.5, rtol=0, atol=0)


@pytest.mark.parametrize(
    ("rule", "ends", "grades", "coverage"),
    [
        # L = 4, threshold 0.9 * 4 = 3.6: the end composite 8-11.5 (3.5 m) is short.
        ("keep", [4.0, 8.0, 11.5], [2.0, 2.5, 10.0 / 3.5], [4.0, 2.0, 3.5]),
        ("discard", [4.0, 8.0], [2.0, 2.5], [4.0, 2.0]),
        # merged 4-11.5: 1 m of g 3 + 3 m of g 2 + 1.5 m of g 4 = 15 on 5.5 m
        ("merge", [4.0, 11.5], [2.0, 15.0 / 5.5], [4.0, 5.5]),
    ],
)
def test_min_length_rules(rule, ends, grades, coverage):
    out = composite(hand_samples(), length=4.0, min_length=0.9, short=rule)
    np.testing.assert_allclose(out["depth_to"], ends)
    np.testing.assert_allclose(out["g"], grades, rtol=1e-14)
    np.testing.assert_allclose(out["g_coverage"], coverage, rtol=1e-14)
    # Below the threshold nothing is short: 3.5 >= 0.8 * 4.
    same = composite(hand_samples(), length=4.0, min_length=0.8, short=rule)
    np.testing.assert_allclose(same["depth_to"], [4.0, 8.0, 11.5])


def test_min_length_merge_in_domain_and_bench_modes():
    # Domain runs with L = 4: A 0-3 (one short composite: nothing to merge with, kept),
    # B 3-7 + 7-9 (2 m is short -> one composite 3-9), A 9-13 ...
    domains = IntervalDomains(GEOLOGY, "rock")
    out = composite(hand_samples(), mode="domain", length=4.0, domain=domains,
                    min_length=0.8, short="merge")
    np.testing.assert_allclose(out["depth_from"], [0.0, 3.0, 9.0])
    np.testing.assert_allclose(out["depth_to"], [3.0, 9.0, 13.0])
    # B 3-9: 2 m of g 3 + 2 m of g 2 -> 2.5 on 4 m
    np.testing.assert_allclose(out["g"], [5.0 / 3.0, 2.5, 3.2], rtol=1e-14)
    out = composite(hand_samples(), mode="domain", length=4.0, domain=domains,
                    min_length=0.8, short="discard")
    np.testing.assert_allclose(out["depth_from"], [3.0, 9.0])
    # Bench (see test_bench_hand_example): composites 0-1, 1-6, 6-11, 11-11.5 span
    # 0.5, 2.5, 2.5 and 0.25 m of elevation; with threshold 0.5 * 2.5 the first merges into
    # the next and the last into the previous.
    traces = straight_trace(-30.0, z=100.5)
    out = composite(hand_samples(), traces, mode="bench", bench_height=2.5,
                    min_length=0.5, short="merge")
    np.testing.assert_allclose(out["depth_from"], [0.0, 6.0], atol=1e-9)
    np.testing.assert_allclose(out["depth_to"], [6.0, 11.5], atol=1e-9)
    # 0-6: 2 m of g 1 + 3 m of g 3 -> 11 / 5.   6-11.5: 3 m of g 2 + 1.5 m of g 4 -> 12 / 4.5
    np.testing.assert_allclose(out["g"], [11.0 / 5.0, 12.0 / 4.5], rtol=1e-9)
    out = composite(hand_samples(), traces, mode="bench", bench_height=2.5,
                    min_length=0.5, short="discard")
    np.testing.assert_allclose(out["depth_from"], [1.0, 6.0], atol=1e-9)


def test_density_weighting_hand_example():
    samples = pd.DataFrame(
        {
            "hole_id": ["H"] * 4,
            "depth_from": [0.0, 2.0, 4.0, 6.0],
            "depth_to": [2.0, 4.0, 6.0, 8.0],
            "g": [1.0, 3.0, 5.0, 2.0],
            "sg": [2.0, 4.0, NAN, 3.0],
        }
    )
    out = composite(samples, length=4.0, density="sg")
    # 0-4: (1*2*2 + 3*2*4) / (2*2 + 2*4) = 28 / 12; density by length: (2*2 + 4*2) / 4 = 3
    # 4-8: the sample without density cannot be weighted -> only 6-8 counts: g 2 on 2 m;
    #      density (3*2) / 2 = 3 on 2 m.
    np.testing.assert_allclose(out["g"], [28.0 / 12.0, 2.0], rtol=1e-14)
    np.testing.assert_allclose(out["sg"], [3.0, 3.0], rtol=1e-14)
    np.testing.assert_allclose(out["g_coverage"], [4.0, 2.0])
    np.testing.assert_allclose(out["sg_coverage"], [4.0, 2.0])
    # Plain length weighting for comparison: (1*2 + 3*2) / 4 = 2 and (5*2 + 2*2) / 4 = 3.5.
    plain = composite(samples, length=4.0)
    np.testing.assert_allclose(plain["g"], [2.0, 3.5], rtol=1e-14)
    # With a density given for missing values: (5*2*2.5 + 2*2*3) / (2*2.5 + 2*3) = 37 / 11.
    filled = composite(samples, length=4.0, density="sg", fill_missing={"sg": 2.5})
    np.testing.assert_allclose(filled["g"], [28.0 / 12.0, 37.0 / 11.0], rtol=1e-14)
    # Mass-weighted accumulation sum(g * l * rho) is conserved: 28 + 12 = 40.
    mass = samples["g"] * 2.0 * samples["sg"]
    np.testing.assert_allclose(
        (out["g"] * out["g_coverage"] * out["sg"]).sum(), mass.sum(skipna=True), rtol=1e-14
    )


# --------------------------------------------------------------------------------------
# domain sources
# --------------------------------------------------------------------------------------
def test_split_at_domains_keeps_grades_and_adds_codes():
    split = split_at_domains(hand_samples(), IntervalDomains(GEOLOGY, "rock"))
    np.testing.assert_allclose(
        split["depth_from"], [0.0, 2.0, 3.0, 5.0, 6.0, 7.0, 9.0, 10.0, 11.5]
    )
    assert split["domain"].tolist() == ["A", "A", "B", "B", "B", "B", "A", "A", "A"]
    np.testing.assert_allclose(split["g"], [1, 3, 3, NAN, NAN, 2, 2, 4, NAN])
    with pytest.raises(ValueError, match="no column"):
        split_at_domains(hand_samples(), IntervalDomains(GEOLOGY, "nope"))


def test_domain_sources_are_interchangeable():
    """Column name, array and callable give the same composites as the interval table."""
    reference = composite(
        hand_samples(), mode="domain", length=4.0, domain=IntervalDomains(GEOLOGY, "rock")
    )
    split = split_at_domains(hand_samples(), IntervalDomains(GEOLOGY, "rock"), name="code")
    by_column = composite(split, mode="domain", length=4.0, domain="code")
    by_array = composite(
        split.drop(columns="code"), mode="domain", length=4.0, domain=split["code"].to_numpy()
    )
    calls = []

    def inside_solid(intervals: pd.DataFrame) -> np.ndarray:
        """Stand-in for a later 'inside solid X' test: one code per segment."""
        calls.append(len(intervals))
        middle = 0.5 * (intervals["depth_from"] + intervals["depth_to"]).to_numpy()
        return np.where((middle > 3.0) & (middle < 9.0), "B", "A")

    by_callable = composite(
        split.drop(columns="code"), mode="domain", length=4.0, domain=inside_solid
    )
    assert calls == [len(split)]
    for other in (by_column, by_array, by_callable):
        pd.testing.assert_frame_equal(other, reference)
    with pytest.raises(ValueError, match="codes for"):
        composite(hand_samples(), mode="domain", length=4.0, domain=np.array(["A", "B"]))


def test_numeric_domain_codes_and_missing_domain():
    samples = hand_samples()
    samples["zone"] = [1, 1, 2, 2, 2]
    out = composite(samples, mode="domain", length=10.0, domain="zone")
    assert out["domain"].tolist() == [1, 2]
    assert "zone" not in out.columns  # the domain column is not averaged
    np.testing.assert_allclose(out["depth_to"], [5.0, 11.5])
    # A stretch that the domain table does not cover is its own run with a missing code.
    short_geology = GEOLOGY.iloc[:2]  # covers 0-9 only
    out = composite(samples.drop(columns="zone"), mode="domain", length=10.0,
                    domain=IntervalDomains(short_geology, "rock"))
    assert out["domain"].isna().tolist() == [False, False, True]
    np.testing.assert_allclose(out["depth_from"], [0.0, 3.0, 9.0])


def test_domain_is_honoured_in_bench_mode():
    traces = straight_trace(-30.0, z=100.5)  # levels at depths 1, 6, 11
    out = composite(hand_samples(), traces, mode="bench", bench_height=2.5,
                    domain=IntervalDomains(GEOLOGY, "rock"))
    np.testing.assert_allclose(out["depth_from"], [0.0, 1.0, 3.0, 6.0, 9.0, 11.0], atol=1e-9)
    assert out["domain"].tolist() == ["A", "A", "B", "B", "A", "A"]
    assert_conserved(hand_samples(), out, ["g"])


# --------------------------------------------------------------------------------------
# argument and data errors
# --------------------------------------------------------------------------------------
def test_bad_arguments_and_bad_data():
    samples = hand_samples()
    with pytest.raises(ValueError, match="mode must be"):
        composite(samples, length=2.0, mode="uphole")
    with pytest.raises(ValueError, match="needs length"):
        composite(samples)
    with pytest.raises(ValueError, match="domain source"):
        composite(samples, length=2.0, mode="domain")
    with pytest.raises(ValueError, match="bench_height"):
        composite(samples, straight_trace(-60.0), mode="bench")
    with pytest.raises(ValueError, match="traces"):
        composite(samples, mode="bench", bench_height=5.0)
    with pytest.raises(ValueError, match="best_fit"):
        composite(samples, straight_trace(-60.0), mode="bench", bench_height=5.0, best_fit=True)
    with pytest.raises(ValueError, match="short must be"):
        composite(samples, length=2.0, short="drop")
    with pytest.raises(ValueError, match="not in the interval table"):
        composite(samples, length=2.0, attributes=["cu"])
    with pytest.raises(ValueError, match="unknown attributes"):
        composite(samples, length=2.0, fill_missing={"cu": 0.0})
    text = samples.assign(rock="A")
    with pytest.raises(ValueError, match="not numeric"):
        composite(text, length=2.0, attributes=["rock"])
    assert "rock" not in composite(text, length=2.0).columns  # text columns are skipped
    overlapping = pd.concat([samples, samples.iloc[[1]].assign(depth_from=4.0, depth_to=5.5)])
    with pytest.raises(ValueError, match="overlapping"):
        composite(overlapping, length=2.0)
    reversed_row = samples.copy()
    reversed_row.loc[0, "depth_to"] = -1.0
    with pytest.raises(ValueError, match="cannot be composited"):
        composite(reversed_row, length=2.0)
    dropped = composite(reversed_row, length=4.0, on_invalid="drop")
    # Without sample 0-2 the composite 0-4 holds only 2 m of g 3.
    np.testing.assert_allclose(dropped["g"].iloc[0], 3.0)
    np.testing.assert_allclose(dropped["g_coverage"].iloc[0], 2.0)


def test_empty_table_and_row_order():
    empty = composite(hand_samples().iloc[:0], length=2.0)
    assert len(empty) == 0
    assert list(empty.columns)[:4] == ["hole_id", "depth_from", "depth_to", "length"]
    # Rows in any order, two holes: output ordered by hole (first appearance) and depth.
    two = pd.concat([hand_samples().assign(hole_id="Z"), hand_samples()]).sample(
        frac=1.0, random_state=3
    )
    first_hole = two["hole_id"].iloc[0]
    out = composite(two, length=4.0)
    assert out["hole_id"].tolist() == [first_hole] * 3 + [h for h in "HZ" if h != first_hole] * 3
    np.testing.assert_allclose(out["g"], [2.0, 2.5, 10.0 / 3.5] * 2, rtol=1e-14)


def test_bench_mode_skips_holes_without_a_trace():
    samples = pd.concat([hand_samples(), hand_samples().assign(hole_id="NOTRACE")])
    with pytest.warns(UserWarning, match="without a trace"):
        out = composite(samples, straight_trace(-30.0, z=100.5), mode="bench", bench_height=2.5)
    assert set(out["hole_id"]) == {"H"}
    # In the other modes such a hole is composited, with NaN coordinates.
    out = composite(samples, straight_trace(-30.0), length=4.0)
    assert out.loc[out["hole_id"] == "NOTRACE", "x"].isna().all()
    assert out.loc[out["hole_id"] == "H", "x"].notna().all()


# --------------------------------------------------------------------------------------
# conservation of metal on the real tables
# --------------------------------------------------------------------------------------
def synth_modes(synth) -> dict[str, dict]:
    domains = IntervalDomains(synth["geology"], "rock")
    return {
        "downhole": dict(length=2.0),
        "downhole_best_fit": dict(length=2.0, best_fit=True),
        "downhole_merge": dict(length=5.0, min_length=0.5, short="merge"),
        "bench": dict(mode="bench", bench_height=5.0, bench_reference=1.25),
        "domain": dict(mode="domain", length=2.0, domain=domains),
        "domain_best_fit": dict(mode="domain", length=2.0, domain=domains, best_fit=True),
        "domain_bench": dict(mode="bench", bench_height=10.0, domain=domains),
    }


@pytest.mark.parametrize("name", [
    "downhole", "downhole_best_fit", "downhole_merge", "bench", "domain", "domain_best_fit",
    "domain_bench",
])
def test_metal_is_conserved_on_synthetic(synth, name):
    out = composite(synth["assay"], synth["traces"], **synth_modes(synth)[name])
    assert len(out) > 1000
    assert set(out["hole_id"]) == set(synth["assay"]["hole_id"])
    assert_conserved(synth["assay"], out, ["au"])
    # Every composite lies on its hole's path at its midpoint depth.
    middle = 0.5 * (out["depth_from"] + out["depth_to"]).to_numpy()
    xyz, direction = synth["traces"].xyz_at(out["hole_id"].to_numpy(dtype=object), middle)
    np.testing.assert_array_equal(out[["x", "y", "z"]].to_numpy(), xyz)
    np.testing.assert_array_equal(out[["dx", "dy", "dz"]].to_numpy(), direction)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
    np.testing.assert_allclose(out["length"], out["depth_to"] - out["depth_from"], rtol=0, atol=0)


def babbitt_domain(intervals: pd.DataFrame) -> np.ndarray:
    """A callable domain source: grade shells on copper (missing where not assayed)."""
    cu = intervals["cu"].to_numpy()
    return np.where(np.isnan(cu), None, np.where(cu >= 0.2, "HG", "LG"))


@pytest.mark.parametrize(
    "options",
    [
        dict(length=10.0),
        dict(length=10.0, best_fit=True),
        dict(mode="bench", bench_height=20.0),
        dict(mode="domain", length=10.0, domain=babbitt_domain),
        dict(mode="domain", length=10.0, domain=babbitt_domain, best_fit=True),
        dict(mode="domain", length=10.0, domain=babbitt_domain, min_length=0.5, short="merge"),
    ],
    ids=["downhole", "best_fit", "bench", "domain", "domain_best_fit", "domain_merge"],
)
def test_metal_is_conserved_on_babbitt(babbitt_db, options):
    assay = babbitt_db.intervals["assay"]
    traces = babbitt_db.desurvey()
    out = composite(assay, traces, **options)
    assert len(out) > 5000
    assert_conserved(assay, out, ["cu", "ni", "s", "fe"])
    assert np.isfinite(out[["x", "y", "z"]].to_numpy()).all()
    if "domain" in out.columns:
        # No composite mixes the two shells: its grade is on the right side of the cut.
        high = out["domain"] == "HG"
        assert (out.loc[high, "cu"] >= 0.2 - 1e-12).all()
        assert (out.loc[out["domain"] == "LG", "cu"] < 0.2).all()
        assert out.loc[out["domain"].isna(), "cu"].isna().all()


# --------------------------------------------------------------------------------------
# domain mode on the synthetic deposit
# --------------------------------------------------------------------------------------
def test_no_composite_spans_two_domains(synth):
    geology = synth["geology"]
    length = 2.0
    out = composite(
        synth["assay"], synth["traces"], mode="domain", length=length,
        domain=IntervalDomains(geology, "rock"),
    )
    assert set(out["domain"]) == {"ORE", "WASTE"}
    checked = 0
    for hole, comps in out.groupby("hole_id"):
        log = geology[geology["hole_id"] == hole].sort_values("depth_from")
        starts = log["depth_from"].to_numpy()
        ends = log["depth_to"].to_numpy()
        rocks = log["rock"].to_numpy()
        # The geology interval that holds the composite's start must hold all of it.
        i = np.searchsorted(starts, comps["depth_from"].to_numpy(), side="right") - 1
        assert (comps["depth_to"].to_numpy() <= ends[i] + 1e-9).all()
        assert (rocks[i] == comps["domain"].to_numpy()).all()
        checked += len(comps)
    assert checked == len(out)
    assert (out["length"] <= length + 1e-9).all()


def test_domain_matches_truth_rock_away_from_contacts(synth):
    geology, truth = synth["geology"], synth["truth_samples"]
    length = 2.0
    out = composite(
        synth["assay"], synth["traces"], mode="domain", length=length,
        domain=IntervalDomains(geology, "rock"),
    )
    middle = 0.5 * (out["depth_from"] + out["depth_to"]).to_numpy()
    compared = ore = 0
    for hole, comps in out.groupby("hole_id"):
        mid = middle[comps.index.to_numpy()]
        log = geology[geology["hole_id"] == hole].sort_values("depth_from")
        contacts = log["depth_from"].to_numpy()[1:]
        near = np.zeros(mid.size, dtype=bool)
        if contacts.size:
            near = np.abs(mid[:, None] - contacts[None, :]).min(axis=1) <= length
        samples = truth[truth["hole_id"] == hole].sort_values("depth_from")
        i = np.searchsorted(samples["depth_from"].to_numpy(), mid, side="right") - 1
        inside = (i >= 0) & (mid <= samples["depth_to"].to_numpy()[np.maximum(i, 0)])
        use = inside & ~near
        expected = samples["rock"].to_numpy()[i[use]]
        got = comps["domain"].to_numpy()[use]
        assert (got == expected).all(), hole
        compared += int(use.sum())
        ore += int((got == "ORE").sum())
    assert compared > 0.8 * len(out)
    assert ore > 100


# --------------------------------------------------------------------------------------
# bench levels on curved holes
# --------------------------------------------------------------------------------------
def distance_to_level(z: np.ndarray, height: float, reference: float) -> np.ndarray:
    u = (z - reference) / height
    return np.abs(u - np.rint(u)) * height


def test_bench_crossings_match_an_independent_root_finder():
    traces = curved_trace()
    height, reference = 7.5, 1.0
    found = bench_crossings(traces, height, reference)

    def z_at(depth: float) -> float:
        return float(traces.xyz_at("C", depth)[0][2])

    # Independent: sample z densely, bracket every sign change, polish with brentq.
    grid = np.linspace(0.0, 400.0, 40001)
    z = traces.xyz_at("C", grid)[0][:, 2]
    assert z.min() < z[-1] - 5.0  # the hole really turns upward
    expected = []
    for level in reference + height * np.arange(np.floor(z.min() / height) - 1,
                                                np.ceil(z.max() / height) + 2):
        f = z - level
        for i in np.flatnonzero(f[:-1] * f[1:] < 0.0):
            expected.append((brentq(lambda d, lv=level: z_at(d) - lv, grid[i], grid[i + 1],
                                    xtol=1e-12), level))
    expected.sort()
    assert len(expected) == len(found) > 20
    np.testing.assert_allclose(found["depth"], [d for d, _ in expected], atol=1e-8)
    np.testing.assert_allclose(found["level"], [lv for _, lv in expected], atol=1e-9)
    levels, counts = np.unique(found["level"], return_counts=True)
    assert counts.max() == 2  # levels crossed going down and again coming up


@pytest.mark.parametrize("method", ["minimum_curvature", "balanced_tangential", "tangent"])
def test_bench_composite_ends_lie_on_levels_for_curved_holes(synth, method):
    height, reference = 5.0, 1.25
    curved = curved_trace()
    samples = pd.DataFrame(
        {"hole_id": "C", "depth_from": np.arange(0.0, 399.0, 1.5),
         "depth_to": np.arange(1.5, 400.5, 1.5)}
    )
    samples["g"] = np.sin(samples["depth_from"]) + 2.0
    cases = [(samples, curved, "g")]
    cases.append(
        (synth["assay"], desurvey(synth["collar"], synth["survey"], method=method), "au")
    )
    for table, traces, attribute in cases:
        out = composite(table, traces, mode="bench", bench_height=height,
                        bench_reference=reference, drop_empty=False)
        assert_conserved(table, out, [attribute])
        ids = out["hole_id"].to_numpy(dtype=object)
        z_from = traces.xyz_at(ids, out["depth_from"].to_numpy())[0][:, 2]
        z_to = traces.xyz_at(ids, out["depth_to"].to_numpy())[0][:, 2]
        hole_end = table.groupby("hole_id")["depth_to"].max().reindex(ids).to_numpy()
        starts_at_collar = out["depth_from"].to_numpy() == 0.0
        ends_at_bottom = out["depth_to"].to_numpy() == hole_end
        assert starts_at_collar.sum() == ends_at_bottom.sum() == out["hole_id"].nunique()
        assert distance_to_level(z_from[~starts_at_collar], height, reference).max() < 1e-6
        assert distance_to_level(z_to[~ends_at_bottom], height, reference).max() < 1e-6
        # Composites tile each hole, and none spans more than one bench.
        same = ids[1:] == ids[:-1]
        np.testing.assert_array_equal(
            out["depth_from"].to_numpy()[1:][same], out["depth_to"].to_numpy()[:-1][same]
        )
        assert (np.abs(z_from - z_to) <= height + 1e-6).all()
        # No level is crossed inside a composite: its ends (moved 1e-6 inwards in elevation)
        # and points inside it are all in one bench.
        for fraction in (0.25, 0.5, 0.75):
            depth = out["depth_from"] + fraction * out["length"]
            z_in = traces.xyz_at(ids, depth.to_numpy())[0][:, 2]
            low = np.minimum(np.minimum(z_from, z_to) + 1e-6, z_in)
            high = np.maximum(np.maximum(z_from, z_to) - 1e-6, z_in)
            wide = high > low
            bench_low = np.floor((low - reference) / height)
            bench_high = np.floor((high - reference) / height)
            assert (bench_low[wide] == bench_high[wide]).all()


# --------------------------------------------------------------------------------------
# properties on random holes
# --------------------------------------------------------------------------------------
@st.composite
def random_hole(draw):
    n = draw(st.integers(min_value=1, max_value=25))
    lengths = draw(st.lists(st.floats(0.05, 6.0), min_size=n, max_size=n))
    gaps = draw(st.lists(st.sampled_from([0.0, 0.0, 0.0, 0.7, 3.3]), min_size=n, max_size=n))
    grades = draw(
        st.lists(st.one_of(st.none(), st.floats(0.0, 500.0)), min_size=n, max_size=n)
    )
    codes = draw(st.lists(st.sampled_from(["A", "B", None]), min_size=n, max_size=n))
    start = draw(st.sampled_from([0.0, 1.25]))
    depth_from, depth_to, position = [], [], start
    for length, gap in zip(lengths, gaps, strict=True):
        position += gap
        depth_from.append(position)
        position += length
        depth_to.append(position)
    return pd.DataFrame(
        {
            "hole_id": "R",
            "depth_from": depth_from,
            "depth_to": depth_to,
            "g": [NAN if g is None else g for g in grades],
            "code": codes,
        }
    )


@settings(max_examples=150, deadline=None)
@given(
    samples=random_hole(),
    length=st.floats(0.3, 12.0),
    mode=st.sampled_from(["downhole", "domain", "bench"]),
    best_fit=st.booleans(),
    short=st.sampled_from(["keep", "merge"]),
    min_length=st.sampled_from([0.0, 0.5, 0.75]),
    dip=st.floats(-90.0, -5.0),
)
def test_properties_on_random_holes(samples, length, mode, best_fit, short, min_length, dip):
    end = float(samples["depth_to"].max())
    collar = pd.DataFrame({"hole_id": ["R"], "x": [0.0], "y": [0.0], "z": [50.0],
                           "max_depth": [end]})
    survey = pd.DataFrame({"hole_id": ["R", "R"], "depth": [0.0, end],
                           "dip": [dip, min(dip + 30.0, 20.0)], "azimuth": [0.0, 45.0]})
    traces = desurvey(collar, survey)
    options = dict(min_length=min_length, short=short, drop_empty=False, attributes=["g"])
    if mode == "bench":
        options.update(mode="bench", bench_height=length)
    else:
        options.update(mode=mode, length=length, best_fit=best_fit)
    if mode == "domain":
        options["domain"] = "code"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = composite(samples, traces, **options)

    start, stop = out["depth_from"].to_numpy(), out["depth_to"].to_numpy()
    sample_from, sample_to = samples["depth_from"].to_numpy(), samples["depth_to"].to_numpy()
    grade = samples["g"].to_numpy()
    # --- lengths
    assert (out["length"].to_numpy() > 0).all()
    np.testing.assert_array_equal(out["length"].to_numpy(), stop - start)
    assert (start[1:] >= stop[:-1]).all()  # ordered, no overlap
    if mode == "domain":
        # Runs: consecutive samples with the same code (gaps inside a run stay in the run).
        code = samples["code"].fillna("?").to_numpy()
        first = np.flatnonzero(np.r_[True, code[1:] != code[:-1]])
        last = np.r_[first[1:] - 1, len(code) - 1]
        expected_total = float((sample_to[last] - sample_from[first]).sum())
        assert out["domain"].fillna("?").tolist() == [
            code[first][np.searchsorted(sample_from[first], s, side="right") - 1] for s in start
        ]
    else:
        expected_total = end  # from the collar to the end of the deepest sample
        np.testing.assert_array_equal(start[1:], stop[:-1])  # composites tile the hole
        assert start[0] == 0.0
        assert stop[-1] == end
    np.testing.assert_allclose(out["length"].sum(), expected_total, rtol=1e-9)
    if mode != "bench" and not best_fit and short == "keep":
        assert (out["length"] <= length * (1 + 1e-9)).all()
    if mode == "bench":
        z = traces.xyz_at("R", np.r_[start, stop])[0][:, 2]
        assert (np.abs(z[: len(out)] - z[len(out):]) <= length + 1e-6).all() or short == "merge"
    # --- coverage and conservation
    finite = np.isfinite(grade)
    np.testing.assert_allclose(
        out["g_coverage"].sum(), (sample_to - sample_from)[finite].sum(), rtol=1e-9, atol=1e-12
    )
    assert (out["g_coverage"] <= out["length"] * (1 + 1e-9) + 1e-12).all()
    metal = float((grade * (sample_to - sample_from))[finite].sum())
    np.testing.assert_allclose(
        np.nansum(out["g"] * out["g_coverage"]), metal, rtol=1e-9, atol=1e-12
    )
    # --- every weighted mean lies between the smallest and largest contributing sample
    for a, b, value, cover in zip(start, stop, out["g"], out["g_coverage"], strict=True):
        inside = finite & (np.minimum(sample_to, b) - np.maximum(sample_from, a) > 0)
        if not inside.any():
            assert np.isnan(value)
            assert cover == 0.0
            continue
        low, high = grade[inside].min(), grade[inside].max()
        slack = 1e-9 * max(1.0, abs(high))
        assert low - slack <= value <= high + slack


# --------------------------------------------------------------------------------------
# performance
# --------------------------------------------------------------------------------------
def test_five_thousand_holes_of_five_hundred_samples_in_seconds():
    holes, per_hole = 5000, 500
    rng = np.random.default_rng(7)
    lengths = rng.uniform(0.5, 1.5, (holes, per_hole))
    depth_to = np.cumsum(lengths, axis=1)
    depth_from = np.concatenate((np.zeros((holes, 1)), depth_to[:, :-1]), axis=1)
    names = np.char.add("H", np.arange(holes).astype(str)).astype(object)
    samples = pd.DataFrame(
        {
            "hole_id": np.repeat(names, per_hole),
            "depth_from": depth_from.ravel(),
            "depth_to": depth_to.ravel(),
            "au": rng.lognormal(size=holes * per_hole),
            "cu": rng.lognormal(size=holes * per_hole),
        }
    )
    samples.loc[rng.random(len(samples)) < 0.05, "au"] = NAN
    collar = pd.DataFrame({"hole_id": names, "x": rng.uniform(0, 1e4, holes),
                           "y": rng.uniform(0, 1e4, holes), "z": 500.0,
                           "max_depth": depth_to[:, -1]})
    survey = pd.DataFrame({"hole_id": np.repeat(names, 2), "depth": np.tile([0.0, 300.0], holes),
                           "dip": np.tile([-70.0, -50.0], holes),
                           "azimuth": np.repeat(rng.uniform(0, 360, holes), 2)})
    traces = desurvey(collar, survey)
    started = time.perf_counter()
    out = composite(samples, traces, length=2.0)
    elapsed = time.perf_counter() - started
    assert len(out) > 1_000_000
    assert out["hole_id"].nunique() == holes
    assert np.isfinite(out[["x", "y", "z"]].to_numpy()).all()
    length = (samples["depth_to"] - samples["depth_from"]).to_numpy()
    for attribute in ("au", "cu"):
        np.testing.assert_allclose(
            np.nansum(out[attribute] * out[f"{attribute}_coverage"]),
            np.nansum(samples[attribute] * length),
            rtol=1e-9,
        )
    # "A few seconds": about 1.5 s on a 2024 laptop; the bound leaves room for slow CI.
    assert elapsed < 15.0, f"compositing took {elapsed:.1f} s"


def test_composite_below_the_stated_hole_depth_is_located_with_a_warning():
    """A sample that ends 0.6 m below the collar table's hole depth: its composite is on
    the straight continuation of the hole (analytic for a straight hole), not NaN."""
    collar = pd.DataFrame({"hole_id": ["S"], "x": [100.0], "y": [200.0], "z": [50.0],
                           "max_depth": [10.0]})  # fmt: skip
    survey = pd.DataFrame({"hole_id": ["S"], "depth": [0.0], "dip": [-60.0], "azimuth": [90.0]})
    traces = desurvey(collar, survey)
    samples = pd.DataFrame({"hole_id": ["S"] * 3, "depth_from": [8.0, 9.0, 10.0],
                            "depth_to": [9.0, 10.0, 10.6], "g": [1.0, 2.0, 3.0]})  # fmt: skip
    with pytest.warns(UserWarning, match="1 composite.*below the stated hole depth.*0.3"):
        out = composite(samples, length=1.0, traces=traces)
    np.testing.assert_allclose(out["depth_to"], [9.0, 10.0, 10.6])
    middle = out["depth_from"].to_numpy() / 2 + out["depth_to"].to_numpy() / 2
    np.testing.assert_allclose(out["x"], 100.0 + middle * np.cos(np.radians(60.0)), atol=1e-9)
    np.testing.assert_allclose(out["y"], 200.0, atol=1e-9)
    np.testing.assert_allclose(out["z"], 50.0 - middle * np.sin(np.radians(60.0)), atol=1e-9)
    np.testing.assert_allclose(out["dz"], -np.sin(np.radians(60.0)), atol=1e-12)
    # a hole without a trace still has no coordinates
    other = composite(samples.assign(hole_id="X"), length=1.0, traces=traces)
    assert np.isnan(other[["x", "y", "z"]].to_numpy()).all()
