"""Kuz-Ram fragmentation: hand arithmetic, Rosin-Rammler identities, a scipy check.

The expected numbers are evaluated by hand from the published formulas (written out in each
test). No published worked example is reproduced: none was available to check against.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from bakumine.blast import fragmentation as fr
from bakumine.blast.model import ANFO, EMULSION, BlastDesign, Hole, Rock


def make_design(pattern: str = "staggered", rock: Rock | None = None) -> BlastDesign:
    """2 x 3 holes, 3 m x 3.6 m, 10 m bench, 100 mm holes, 2 m stemming, no subdrill."""
    holes = []
    for row in range(2):
        for i in range(3):
            x, y = 3.6 * i, -3.0 * row
            holes.append(
                Hole(
                    id=f"R{row}H{i}",
                    collar=(x, y, 10.0),
                    toe=(x, y, 0.0),
                    diameter=0.1,
                    charge_top=2.0,
                    charge_bottom=10.0,
                    explosive=ANFO,
                    row=row,
                    index=i,
                    burden=3.0,
                    spacing=3.6,
                )
            )
    return BlastDesign(
        name="t",
        holes=holes,
        rock=rock or Rock(),
        bench_height=10.0,
        burden=3.0,
        spacing=3.6,
        pattern=pattern,
    )


def test_kuznetsov_by_hand():
    # A = 7, K = 0.5 kg/m3, Q = 100 kg, ANFO (RWS 100):
    # 0.5^-0.8 = 2^0.8 = 1.741101; 100^(1/6) = 2.154435; 1.15^0.95 = 1.141994
    # x50 = 7 * 1.741101 * 2.154435 * 1.141994 = 29.986 cm
    assert fr.kuznetsov_x50(7.0, 0.5, 100.0, 100.0) == pytest.approx(0.29986, rel=2e-5)
    # TNT-equivalent strength 115 removes the last factor: 7 * 1.741101 * 2.154435 = 26.2576 cm
    assert fr.kuznetsov_x50(7.0, 0.5, 100.0, 115.0) == pytest.approx(0.262576, rel=2e-5)


def test_kuznetsov_trends_and_errors():
    base = fr.kuznetsov_x50(7.0, 0.5, 100.0)
    assert fr.kuznetsov_x50(7.0, 0.8, 100.0) < base  # more explosive per m3: finer
    assert fr.kuznetsov_x50(10.0, 0.5, 100.0) > base  # harder rock: coarser
    assert fr.kuznetsov_x50(7.0, 0.5, 100.0, EMULSION.rws) < base  # stronger explosive: finer
    with pytest.raises(ValueError):
        fr.kuznetsov_x50(7.0, 0.0, 100.0)


def test_rock_factor_by_hand():
    # jointed: RMD = JF = 20 + 30 = 50; RDI = 25 * 2.7 - 50 = 17.5; E = 60 GPa >= 50 so
    # HF = UCS / 5 = 150 / 5 = 30; A = 0.06 * 97.5 = 5.85
    a = fr.rock_factor(2700.0, jps=20.0, jpa=30.0, youngs_gpa=60.0, ucs_mpa=150.0)
    assert a == pytest.approx(5.85)
    # E = 30 GPa < 50: HF = 30 / 3 = 10; massive RMD = 50; A = 0.06 * 77.5 = 4.65
    assert fr.rock_factor(2700.0, rmd=50.0, youngs_gpa=30.0, ucs_mpa=150.0) == pytest.approx(4.65)
    # friable, only UCS known: HF = 20 / 5 = 4; RDI = 25 * 2.0 - 50 = 0; A = 0.06 * 14 = 0.84
    assert fr.rock_factor(2000.0, rmd=10.0, ucs_mpa=20.0) == pytest.approx(0.84)
    with pytest.raises(ValueError):
        fr.rock_factor(2700.0, rmd=50.0, jps=20.0, jpa=30.0, ucs_mpa=100.0)
    with pytest.raises(ValueError):
        fr.rock_factor(2700.0, jps=20.0, ucs_mpa=100.0)
    with pytest.raises(ValueError):
        fr.rock_factor(2700.0, rmd=50.0)


def test_rock_factor_from_rock():
    assert fr.rock_factor_from_rock(Rock(rock_factor=9.5)) == 9.5
    # default Rock: joint spacing 1.0 m >= oversize 1.0 -> JPS 50, JPA 30; RDI 17.5;
    # E = 50 GPa (not < 50) -> HF = 100 / 5 = 20; A = 0.06 * 117.5 = 7.05
    assert fr.rock_factor_from_rock(Rock()) == pytest.approx(7.05)
    # joints at 0.5 m: JPS 20 -> A = 0.06 * 87.5 = 5.25
    assert fr.rock_factor_from_rock(Rock(joint_spacing=0.5)) == pytest.approx(5.25)
    # joints at 0.05 m: JPS 10 -> 0.06 * 77.5 = 4.65
    assert fr.rock_factor_from_rock(Rock(joint_spacing=0.05)) == pytest.approx(4.65)
    # massive against a 4 m pattern: RMD 50 -> 0.06 * 87.5 = 5.25; friable: 0.06 * 47.5 = 2.85
    assert fr.rock_factor_from_rock(Rock(joint_spacing=5.0), pattern_size=4.0) == pytest.approx(
        5.25
    )
    assert fr.rock_factor_from_rock(Rock(), friable=True) == pytest.approx(2.85)


def test_uniformity_by_hand():
    # B = 3, S = 3.6, d = 100 mm, W = 0.3, L = 8, H = 10, single column charge:
    # (2.2 - 14*3/100) = 1.78; sqrt((1 + 1.2)/2) = 1.048809; (1 - 0.1) = 0.9;
    # (8/8 + 0.1)^0.1 = 1.009577; L/H = 0.8 -> n = 1.357026
    n = fr.uniformity_index(3.0, 3.6, 0.1, 8.0, 10.0, drill_deviation=0.3)
    assert n == pytest.approx(1.357026, rel=2e-6)
    n_st = fr.uniformity_index(3.0, 3.6, 0.1, 8.0, 10.0, drill_deviation=0.3, staggered=True)
    assert n_st == pytest.approx(1.1 * n)
    # equal bottom and column charge: the charge term is 0.1^0.1 = 0.794328
    n_eq = fr.uniformity_index(
        3.0, 3.6, 0.1, 8.0, 10.0, bottom_charge_length=4.0, column_charge_length=4.0
    )
    assert n_eq == pytest.approx(1.78 * 1.048809 * 0.794328 * 0.8, rel=2e-6)
    # only the bottom charge given: the column is the rest (|2 - 6| / 8 = 0.5)
    n_b = fr.uniformity_index(3.0, 3.6, 0.1, 8.0, 10.0, bottom_charge_length=2.0)
    assert n_b == pytest.approx(1.78 * 1.048809 * 0.6**0.1 * 0.8, rel=2e-6)


def test_uniformity_outside_model_is_nan():
    assert math.isnan(fr.uniformity_index(20.0, 20.0, 0.1, 8.0, 10.0))  # 2.2 - 14*20/100 < 0
    assert math.isnan(fr.uniformity_index(3.0, 3.6, 0.1, 8.0, 10.0, drill_deviation=3.0))
    with pytest.raises(ValueError):
        fr.uniformity_index(0.0, 3.6, 0.1, 8.0, 10.0)


def test_rosin_rammler_identities():
    rr = fr.RosinRammler(0.3, 1.4)
    assert float(rr.passing(0.3)) == 0.5  # exactly
    assert float(rr.passing(rr.xc)) == pytest.approx(1.0 - 1.0 / math.e, rel=1e-14)
    assert rr.xc == pytest.approx(0.3 / math.log(2.0) ** (1 / 1.4))
    assert float(rr.passing(0.0)) == 0.0
    assert float(rr.passing(-1.0)) == 0.0
    assert math.isnan(float(rr.passing(math.nan)))
    assert float(rr.size_at(0.5)) == pytest.approx(0.3)
    assert float(rr.size_at(0.0)) == 0.0
    assert math.isinf(float(rr.size_at(1.0)))
    assert math.isnan(float(rr.size_at(1.2)))
    # P80 by hand: x50 * (ln 5 / ln 2)^(1/n) = 0.3 * 2.321928^(1/1.4)
    assert float(rr.size_at(0.8)) == pytest.approx(0.3 * 2.321928 ** (1 / 1.4), rel=1e-6)
    x = np.array([0.01, 0.1, 0.5, 2.0])
    np.testing.assert_allclose(rr.fraction_above(x) + rr.fraction_below(x), 1.0, rtol=1e-14)
    with pytest.raises(ValueError):
        fr.RosinRammler(0.3, 0.0)


def test_rosin_rammler_is_a_weibull_distribution():
    # independent implementation: Weibull CDF with shape n and scale xc
    rr = fr.RosinRammler(0.25, 1.1)
    x = np.geomspace(1e-3, 5.0, 40)
    np.testing.assert_allclose(
        rr.passing(x), stats.weibull_min.cdf(x, 1.1, scale=rr.xc), rtol=1e-12, atol=1e-15
    )
    p = np.array([0.05, 0.2, 0.5, 0.8, 0.99])
    np.testing.assert_allclose(rr.size_at(p), stats.weibull_min.ppf(p, 1.1, scale=rr.xc))


@settings(max_examples=60, deadline=None)
@given(
    x50=st.floats(0.01, 3.0),
    n=st.floats(0.4, 3.0),
    a=st.floats(1e-4, 20.0),
    b=st.floats(1e-4, 20.0),
    p=st.floats(0.001, 0.999),
)
def test_rosin_rammler_properties(x50, n, a, b, p):
    rr = fr.RosinRammler(x50, n)
    lo, hi = min(a, b), max(a, b)
    p_lo, p_hi = float(rr.passing(lo)), float(rr.passing(hi))
    assert 0.0 <= p_lo <= p_hi <= 1.0  # monotone, bounded
    assert float(rr.passing(rr.size_at(p))) == pytest.approx(p, rel=1e-9)  # inverse


def test_curve():
    rr = fr.RosinRammler(0.3, 1.4)
    sizes, passing = rr.curve(40)
    assert sizes.shape == passing.shape == (40,)
    assert np.all(np.diff(sizes) > 0) and np.all(np.diff(passing) > 0)
    assert passing[0] == pytest.approx(0.001) and passing[-1] == pytest.approx(0.999)
    np.testing.assert_allclose(np.diff(np.log(sizes)), np.log(sizes[1] / sizes[0]))  # log-spaced
    with pytest.raises(ValueError):
        rr.curve(1)


def test_swebrec():
    assert float(fr.swebrec(0.3, 0.3, 2.0, 2.5)) == pytest.approx(0.5)
    assert float(fr.swebrec(2.0, 0.3, 2.0, 2.5)) == 1.0
    assert float(fr.swebrec(5.0, 0.3, 2.0, 2.5)) == 1.0
    assert float(fr.swebrec(0.0, 0.3, 2.0, 2.5)) == 0.0
    assert math.isnan(float(fr.swebrec(math.nan, 0.3, 2.0, 2.5)))
    # by hand: x = 1, x50 = 0.5, xmax = 2, b = 2: (ln 2 / ln 4)^2 = 0.25 -> 1 / 1.25 = 0.8
    assert float(fr.swebrec(1.0, 0.5, 2.0, 2.0)) == pytest.approx(0.8)
    p = fr.swebrec(np.geomspace(1e-3, 2.0, 50), 0.3, 2.0, 2.5)
    assert np.all(np.diff(p) > 0)
    with pytest.raises(ValueError):
        fr.swebrec(1.0, 2.0, 1.0, 2.0)


def test_predict_by_hand():
    design = make_design()
    res = fr.predict(design, drill_deviation=0.3, oversize=1.0, fines=0.01)
    # charge per hole: pi/4 * 0.1^2 * 850 * 8 = 53.407 kg; 6 holes = 320.44 kg
    q = math.pi / 4 * 0.01 * 850.0 * 8.0
    assert q == pytest.approx(53.407, rel=1e-5)
    # no polygon: volume = 6 * 3 * 3.6 * 10 = 648 m3 -> K = 0.49451 kg/m3
    k = 6 * q / 648.0
    # default Rock, joint spacing 1 m >= oversize 1 m, < pattern 3.6 m: A = 7.05 (see above)
    assert res.rock_factor == pytest.approx(7.05)
    assert not res.rock_factor_calibrated
    x50 = 7.05 * k**-0.8 * q ** (1 / 6) * 1.15**0.95 / 100.0
    assert res.x50 == pytest.approx(x50, rel=1e-12)
    assert 0.25 < res.x50 < 0.30  # 7.05 * 1.7565 * 1.9406 * 1.14199 = 27.44 cm
    assert res.x50 == pytest.approx(0.2744, rel=1e-3)
    assert res.n == pytest.approx(1.1 * 1.357026, rel=2e-6)  # staggered, see the hand value
    rr = fr.RosinRammler(res.x50, res.n)
    assert res.xc == pytest.approx(rr.xc)
    assert res.p80 == pytest.approx(float(rr.size_at(0.8)))
    assert res.p20 < res.x50 < res.p80
    assert res.oversize_fraction == pytest.approx(1.0 - float(rr.passing(1.0)))
    assert res.fines_fraction == pytest.approx(float(rr.passing(0.01)))
    assert res.inputs["powder_factor_kg_m3"] == pytest.approx(k)
    assert res.inputs["rws"] == pytest.approx(100.0)
    d = res.to_dict()
    assert set(d) == {"sizes_m", "passing", "x50_m", "n", "xc_m"}
    assert len(d["sizes_m"]) == len(d["passing"]) == 60


def test_predict_options_and_errors():
    square = fr.predict(make_design("square"))
    stag = fr.predict(make_design("staggered"))
    assert stag.n == pytest.approx(1.1 * square.n)
    assert math.isnan(square.oversize_fraction) and math.isnan(square.fines_fraction)
    cal = fr.predict(make_design(rock=Rock(rock_factor=10.0)))
    assert cal.rock_factor == 10.0 and cal.rock_factor_calibrated
    assert cal.x50 == pytest.approx(stag.x50 * 10.0 / 7.05)

    design = make_design()
    design.burden = math.nan  # falls back to the holes' burden, volume becomes NaN
    with pytest.raises(ValueError, match="powder factor"):
        fr.predict(design)
    design = make_design()
    for h in design.holes:
        h.charge_top = h.charge_bottom
    with pytest.raises(ValueError, match="no charged holes"):
        fr.predict(design)
    with pytest.raises(ValueError, match="uniformity"):
        fr.predict(make_design(), drill_deviation=3.0)
