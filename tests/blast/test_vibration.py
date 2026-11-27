"""Scaled-distance vibration and airblast: hand arithmetic, brute force, scipy regression."""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from bakumine.blast import vibration as vb
from bakumine.blast.model import ANFO, BlastDesign, Hole


def make_design(delays: list[float], spacing: float = 5.0) -> BlastDesign:
    """One row of vertical holes along x: 0.1 m, charge from 2 m to 10 m (53.407 kg each)."""
    holes = [
        Hole(
            id=f"H{i}",
            collar=(spacing * i, 0.0, 10.0),
            toe=(spacing * i, 0.0, 0.0),
            diameter=0.1,
            charge_top=2.0,
            charge_bottom=10.0,
            explosive=ANFO,
            index=i,
            burden=4.0,
            spacing=spacing,
            delay_ms=t,
        )
        for i, t in enumerate(delays)
    ]
    return BlastDesign(name="t", holes=holes, burden=4.0, spacing=spacing)


Q = math.pi / 4 * 0.1**2 * 850.0 * 8.0  # 53.407 kg per hole


def test_ppv_by_hand():
    # R = 100 m, W = 100 kg: SD = 10; 10^-1.6 = 0.0251189; 1140 * 0.0251189 = 28.635 mm/s
    assert float(vb.ppv(100.0, 100.0)) == pytest.approx(28.635, rel=1e-4)
    # R = 500 m, W = 400 kg: SD = 25; 25^-1.6 = 0.0057999; PPV = 6.6119 mm/s
    assert float(vb.ppv(500.0, 400.0)) == pytest.approx(1140.0 * 25.0**-1.6)
    assert float(vb.ppv(500.0, 400.0)) == pytest.approx(6.612, rel=1e-3)
    assert float(vb.scaled_distance(500.0, 400.0)) == pytest.approx(25.0)
    assert float(vb.scaled_distance(270.0, 27.0, 1 / 3)) == pytest.approx(90.0)
    assert math.isnan(float(vb.ppv(100.0, 0.0)))
    assert math.isnan(float(vb.ppv(math.nan, 10.0)))
    assert math.isinf(float(vb.ppv(0.0, 10.0)))


def test_site_law_presets_and_inverse():
    law = vb.AS2187_AVERAGE
    assert (law.k, law.beta, law.calibrated) == (1140.0, 1.6, False)
    assert law.distance_for(float(law.ppv(250.0, 80.0)), 80.0) == pytest.approx(250.0)
    assert law.charge_for(float(law.ppv(250.0, 80.0)), 250.0) == pytest.approx(80.0)
    # imperial 160 (ft/lb^0.5)^-1.6 in/s -> metric, by hand:
    # 1 m/kg^0.5 = 3.28084 / sqrt(2.20462) = 2.20961 ft/lb^0.5; 2.20961^-1.6 = 0.28124;
    # 160 * 25.4 * 0.28124 = 1143
    assert vb.ISEE_TYPICAL.k == pytest.approx(1143.0, rel=1e-3)
    assert vb.ISEE_UPPER_BOUND.k / vb.ISEE_TYPICAL.k == pytest.approx(242.0 / 160.0)
    # the same physical case in both unit systems: 100 ft, 4 lb -> SD 50 -> 160 * 50^-1.6 in/s
    v_in_s = 160.0 * 50.0**-1.6
    v_mm_s = float(vb.ISEE_TYPICAL.ppv(100.0 * 0.3048, 4.0 * 0.45359237))
    assert v_mm_s == pytest.approx(v_in_s * 25.4)
    with pytest.raises(ValueError):
        vb.SiteLaw(k=-1.0)


@settings(max_examples=60, deadline=None)
@given(
    r=st.floats(1.0, 5000.0),
    dr=st.floats(0.1, 5000.0),
    w=st.floats(0.1, 5000.0),
    dw=st.floats(0.1, 5000.0),
)
def test_ppv_monotone(r, dr, w, dw):
    assert float(vb.ppv(r + dr, w)) < float(vb.ppv(r, w))  # farther: lower
    assert float(vb.ppv(r, w + dw)) > float(vb.ppv(r, w))  # more charge: higher


def brute_mic(times: list[float], masses: list[float], window: float) -> float:
    best = 0.0
    for t in times:
        best = max(best, sum(m for u, m in zip(times, masses, strict=True) if t <= u < t + window))
    return best


def test_charge_per_delay():
    design = make_design([0.0, 0.0, 5.0, 8.0, 17.0, 42.0])
    groups, mic = vb.charge_per_delay(design, 8.0)
    # windows [0, 8): H0 H1 H2; [5, 13): H2 H3; [17, 25): H4; [42, 50): H5
    assert [g.hole_ids for g in groups] == [("H0", "H1", "H2"), ("H2", "H3"), ("H4",), ("H5",)]
    assert [g.t_ms for g in groups] == [0.0, 5.0, 17.0, 42.0]
    assert mic == pytest.approx(3 * Q)
    assert groups[0].charge_kg == pytest.approx(3 * Q)
    # centroid: equal charges at x = 0, 5, 10, mid charge depth z = 10 - 6 = 4
    assert groups[0].centroid == pytest.approx((5.0, 0.0, 4.0))
    # hole by hole with a zero window: every firing time alone, simultaneous holes together
    groups0, mic0 = vb.charge_per_delay(design, 0.0)
    assert len(groups0) == 5 and mic0 == pytest.approx(2 * Q)
    # one long window: everything
    assert vb.charge_per_delay(design, 1000.0)[1] == pytest.approx(6 * Q)


def test_mic_against_brute_force():
    rng = np.random.default_rng(7)
    for _ in range(20):
        times = np.round(rng.uniform(0, 120, 25), 0).tolist()
        design = make_design(times)
        masses = [h.charge_mass for h in design.holes]
        for window in (8.0, 17.0, 1.0):
            assert vb.charge_per_delay(design, window)[1] == pytest.approx(
                brute_mic(times, masses, window)
            )


def test_charge_per_delay_errors():
    with pytest.raises(ValueError, match="firing time"):
        vb.charge_per_delay(make_design([0.0, math.nan]))
    design = make_design([0.0])
    design.holes[0].charge_top = design.holes[0].charge_bottom
    with pytest.raises(ValueError, match="no charged holes"):
        vb.charge_per_delay(design)


def test_ppv_at_mic_mode_by_hand():
    design = make_design([0.0, 25.0, 50.0])  # one hole per delay, MIC = Q
    pts = np.array([[210.0, 0.0, 4.0], [-300.0, 0.0, 4.0], [np.nan, 0.0, 0.0]])
    res = vb.ppv_at(pts, design, mode="mic")
    # nearest charge centre: H2 at x = 10 -> R = 200; H0 at x = 0 -> R = 300
    np.testing.assert_allclose(res.distance[:2], [200.0, 300.0])
    np.testing.assert_allclose(res.scaled_distance[:2], np.array([200.0, 300.0]) / math.sqrt(Q))
    np.testing.assert_allclose(
        res.ppv_mm_s[:2], 1140.0 * (np.array([200.0, 300.0]) / math.sqrt(Q)) ** -1.6
    )
    # 200 / sqrt(53.407) = 27.367; 27.367^-1.6 = 0.005018; PPV = 5.72 mm/s
    assert res.ppv_mm_s[0] == pytest.approx(5.72, rel=2e-3)
    assert math.isnan(res.ppv_mm_s[2]) and res.group[2] == -1 and math.isnan(res.charge_kg[2])
    assert res.law is vb.AS2187_AVERAGE and res.mode == "mic"
    # a single (3,) point is accepted
    assert vb.ppv_at(pts[0], design).ppv_mm_s.shape == (1,)


def test_ppv_at_envelope():
    # two far-apart groups: a big one (3 holes together) at x ~ 5 and a single hole at x = 1000
    design = make_design([0.0, 0.0, 0.0, 100.0])
    hole = design.holes[3]
    hole.collar, hole.toe = (1000.0, 0.0, 10.0), (1000.0, 0.0, 0.0)
    pts = np.array([[5.0, 200.0, 4.0], [1000.0, 50.0, 4.0]])
    env = vb.ppv_at(pts, design, mode="per_hole")
    assert env.group.tolist() == [0, 1]  # each point is governed by the group next to it
    np.testing.assert_allclose(env.distance, [200.0, 50.0])
    np.testing.assert_allclose(env.charge_kg, [3 * Q, Q])
    expected = 1140.0 * (np.array([200.0, 50.0]) / np.sqrt([3 * Q, Q])) ** -1.6
    np.testing.assert_allclose(env.ppv_mm_s, expected)
    # brute force over groups
    for i, p in enumerate(pts):
        best = max(
            1140.0 * (math.dist(p, g.centroid) / math.sqrt(g.charge_kg)) ** -1.6 for g in env.groups
        )
        assert env.ppv_mm_s[i] == pytest.approx(best)
    # the classical MIC number is the larger one near the small group
    mic = vb.ppv_at(pts, design, mode="mic")
    assert mic.ppv_mm_s[1] > env.ppv_mm_s[1]
    with pytest.raises(ValueError, match="unknown mode"):
        vb.ppv_at(pts, design, mode="nearest")
    with pytest.raises(ValueError, match="points"):
        vb.ppv_at(np.zeros((2, 2)), design)


def test_fit_recovers_exact_constants():
    rng = np.random.default_rng(1)
    r = rng.uniform(50, 2000, 40)
    w = rng.uniform(20, 800, 40)
    fit = vb.fit_site_law(r, w, 850.0 * (r / np.sqrt(w)) ** -1.45)
    assert fit.law.k == pytest.approx(850.0, rel=1e-9)
    assert fit.law.beta == pytest.approx(1.45, rel=1e-9)
    assert fit.r2 == pytest.approx(1.0)
    assert fit.law.calibrated and fit.n == 40
    assert fit.k_95 == pytest.approx(850.0, rel=1e-6)  # no scatter: the bound is the line


def test_fit_against_scipy_and_numpy():
    rng = np.random.default_rng(2)
    r = rng.uniform(50, 2000, 200)
    w = rng.uniform(20, 800, 200)
    v = 1140.0 * (r / np.sqrt(w)) ** -1.6 * 10 ** rng.normal(0.0, 0.15, 200)
    v[3], r[10], w[20] = np.nan, np.nan, -1.0  # missing records are dropped
    fit = vb.fit_site_law(r, w, v)
    ok = np.isfinite(v) & np.isfinite(r) & (w > 0)
    assert fit.n == int(ok.sum()) == 197
    x, y = np.log10(r[ok] / np.sqrt(w[ok])), np.log10(v[ok])
    lin = stats.linregress(x, y)
    assert fit.law.beta == pytest.approx(-lin.slope, rel=1e-10)
    assert fit.law.k == pytest.approx(10**lin.intercept, rel=1e-10)
    assert fit.r2 == pytest.approx(lin.rvalue**2, rel=1e-10)
    slope, intercept = np.polyfit(x, y, 1)
    assert fit.law.beta == pytest.approx(-slope, rel=1e-9)
    assert fit.law.k == pytest.approx(10**intercept, rel=1e-9)
    # close to the truth, and the scatter is recovered
    assert fit.law.beta == pytest.approx(1.6, abs=0.1)
    assert fit.se_log10 == pytest.approx(0.15, abs=0.03)
    resid = y - (lin.intercept + lin.slope * x)
    assert fit.se_log10 == pytest.approx(math.sqrt(np.sum(resid**2) / (fit.n - 2)))
    assert fit.k_95 == pytest.approx(fit.law.k * 10 ** (1.645 * fit.se_log10))
    # about 95 % of the records lie below the upper line
    below = np.mean(v[ok] <= fit.upper_law.ppv(r[ok], w[ok]))
    assert 0.90 <= below <= 0.99
    assert fit.upper_law.beta == fit.law.beta


def test_fit_errors():
    with pytest.raises(ValueError, match="at least 3"):
        vb.fit_site_law([100.0, 200.0], [50.0, 50.0], [10.0, 4.0])
    with pytest.raises(ValueError, match="same scaled distance"):
        vb.fit_site_law([100.0] * 3, [50.0] * 3, [10.0, 11.0, 12.0])
    with pytest.raises(ValueError, match="does not decrease"):
        vb.fit_site_law([100.0, 200.0, 400.0], [50.0] * 3, [1.0, 2.0, 4.0])
    with pytest.raises(ValueError, match="same length"):
        vb.fit_site_law([100.0, 200.0], [50.0], [1.0, 2.0])


def test_limits_and_check():
    assert vb.LIMITS["usbm_ri8507_above_40hz"].value == pytest.approx(2.0 * 25.4)
    assert vb.LIMITS["usbm_ri8507_drywall_low_freq"].value == pytest.approx(0.75 * 25.4)
    assert vb.LIMITS["usbm_ri8507_plaster_low_freq"].value == pytest.approx(0.5 * 25.4)
    assert vb.LIMITS["osm_0_300ft"].value == pytest.approx(1.25 * 25.4)
    assert vb.LIMITS["din4150_residential"].value == 5.0
    for table, unit in ((vb.LIMITS, "mm/s"), (vb.AIRBLAST_LIMITS, "dBL")):
        for limit in table.values():
            assert limit.unit == unit and limit.source and limit.value > 0
    ok = vb.check(np.array([4.0, 5.0, 6.0, np.nan]), vb.LIMITS["din4150_residential"])
    assert ok.tolist() == [True, True, False, False]
    assert bool(vb.check(110.0, 115.0))


def test_dbl_by_hand():
    assert float(vb.pressure_to_dbl(2.0)) == pytest.approx(100.0)  # 20 log10(1e5)
    assert float(vb.pressure_to_dbl(20e-6)) == pytest.approx(0.0)
    assert float(vb.pressure_to_dbl(200.0)) == pytest.approx(140.0)
    assert float(vb.pressure_to_dbl(20.0)) == pytest.approx(120.0)
    assert math.isnan(float(vb.pressure_to_dbl(0.0)))
    assert float(vb.dbl_to_pressure(100.0)) == pytest.approx(2.0)
    x = np.array([95.0, 115.0, 133.0])
    np.testing.assert_allclose(vb.pressure_to_dbl(vb.dbl_to_pressure(x)), x)


def test_airblast_by_hand():
    # R = 1000 m, W = 1000 kg: cube-root SD = 100; 100^-1.45 = 10^-2.9 = 1.258925e-3
    # K_a = 100 -> 0.1258925 kPa = 125.89 Pa; K_a = 10 -> 12.589 Pa
    assert float(vb.airblast_pressure(1000.0, 1000.0)) == pytest.approx(125.8925, rel=1e-6)
    assert float(vb.airblast_pressure(1000.0, 1000.0, ka_kpa=10.0)) == pytest.approx(12.58925)
    # 20 log10(12.58925 / 2e-5) = 20 * (1.1 + 4.69897) = 115.98 dBL
    assert float(vb.pressure_to_dbl(12.58925)) == pytest.approx(115.98, abs=0.005)
    # doubling the distance: -20 * 1.45 * log10(2) = -8.73 dB
    d = vb.pressure_to_dbl(vb.airblast_pressure(np.array([500.0, 1000.0]), 300.0))
    assert d[0] - d[1] == pytest.approx(8.73, abs=0.005)


def test_airblast_at():
    design = make_design([0.0, 0.0, 30.0])  # MIC = 2 Q
    pts = np.array([[10.0, 400.0, 10.0], [np.nan, 0.0, 0.0]])
    res = vb.airblast_at(pts, design, ka_kpa=10.0)
    assert res.charge_kg == pytest.approx(2 * Q)
    assert res.distance[0] == pytest.approx(400.0)  # nearest collar is H2 at (10, 0, 10)
    sd = 400.0 / (2 * Q) ** (1 / 3)
    assert res.scaled_distance[0] == pytest.approx(sd)
    assert res.pressure_pa[0] == pytest.approx(10_000.0 * sd**-1.45)
    assert res.dbl[0] == pytest.approx(20 * math.log10(res.pressure_pa[0] / 20e-6))
    assert math.isnan(res.pressure_pa[1]) and math.isnan(res.dbl[1])
    assert not res.calibrated
    assert vb.airblast_at(pts, design).pressure_pa[0] == pytest.approx(10 * res.pressure_pa[0])
