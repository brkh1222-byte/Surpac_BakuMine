"""Holmberg-Persson near-field PPV against scipy quadrature, the closed form and the
point-charge limit; damage classes, back-break and the block attribute by hand."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import quad

from bakumine.blast.damage import (
    BAUER_CALDER,
    DamageThresholds,
    HPConstants,
    backbreak,
    block_damage,
    classify,
    critical_ppv,
    hole_ppv,
    hp_integral,
    points_in_polygon,
    ppv_field,
    thresholds_for,
)
from bakumine.blast.model import ANFO, BlastDesign, Hole, Rock


def _hole(**kw):
    base = dict(id="A1", collar=(0, 0, 110), toe=(0, 0, 99), diameter=0.2,
                charge_top=4.0, charge_bottom=11.0, explosive=ANFO)  # fmt: skip
    return Hole(**(base | kw))


def _quad_ppv(point, hole, c):
    """Independent evaluation: scipy adaptive quadrature in the original variable."""
    d = hole.direction
    rel = np.asarray(point, dtype=float) - np.asarray(hole.collar)
    x0 = float(rel @ d)
    r0 = float(np.linalg.norm(rel - x0 * d))
    p = c.beta / (2 * c.alpha)
    integral, _ = quad(
        lambda x: (r0**2 + (x - x0) ** 2) ** -p, hole.charge_top, hole.charge_bottom,
        points=[min(max(x0, hole.charge_top), hole.charge_bottom)], epsabs=0, epsrel=1e-12,
    )  # fmt: skip
    return c.k * (hole.linear_density * integral) ** c.alpha


POINTS = np.array(
    [
        [0.5, 0.0, 104.0],
        [3.0, 1.0, 102.5],
        [1.0, -2.0, 112.0],
        [8.0, 6.0, 95.0],
        [0.3, 0.2, 99.5],
        [40.0, -25.0, 110.0],
        [2.0, 0.0, 106.0],
    ]  # fmt: skip
)


@pytest.mark.parametrize(
    "constants",
    [HPConstants(), HPConstants(k=400.0, alpha=0.8, beta=1.9), HPConstants(alpha=0.9, beta=1.4)],
)
def test_numeric_integral_matches_scipy_quad(constants):
    hole = _hole()
    got = hole_ppv(POINTS, hole, constants)
    want = [_quad_ppv(p, hole, constants) for p in POINTS]
    assert got == pytest.approx(want, rel=2e-5)


def test_inclined_hole_matches_scipy_quad():
    hole = _hole(toe=(3.0, 4.0, 98.0), charge_top=3.0, charge_bottom=13.0)  # 3-4-12, length 13
    got = hole_ppv(POINTS[[1, 2, 3, 5]], hole)
    want = [_quad_ppv(p, hole, HPConstants()) for p in POINTS[[1, 2, 3, 5]]]
    assert got == pytest.approx(want, rel=2e-5)
    # a point on the perpendicular through mid-charge of the inclined hole equals the same
    # geometry on a vertical hole
    vertical = _hole(collar=(0, 0, 0), toe=(0, 0, -13), charge_top=3.0, charge_bottom=13.0)
    d = hole.direction
    normal = np.cross(d, [0.0, 0.0, 1.0])
    normal /= np.linalg.norm(normal)
    p_inclined = hole.point_at(8.0) + 2.5 * normal
    assert hole_ppv(p_inclined, hole)[0] == pytest.approx(
        hole_ppv([2.5, 0.0, -8.0], vertical)[0], rel=1e-10
    )


def test_closed_form_when_beta_is_two_alpha():
    c = HPConstants(k=700.0, alpha=0.7, beta=1.4)
    hole = _hole()
    q = hole.linear_density
    for p in POINTS:
        r0 = math.hypot(p[0], p[1])
        x0 = 110.0 - p[2]
        arctan = (math.atan((11.0 - x0) / r0) - math.atan((4.0 - x0) / r0)) / r0
        assert hole_ppv(p, hole, c)[0] == pytest.approx(700.0 * (q * arctan) ** 0.7, rel=1e-12)
        assert hole_ppv(p, hole, c)[0] == pytest.approx(_quad_ppv(p, hole, c), rel=1e-8)
    # the Gauss-Legendre branch tends to the closed form as the exponent tends to 1
    near = hp_integral([0.5, 3.0], [6.0, 2.0], 4.0, 11.0, 1.0 + 1e-9)
    exact = hp_integral([0.5, 3.0], [6.0, 2.0], 4.0, 11.0, 1.0)
    assert near == pytest.approx(exact, rel=1e-7)


def test_far_field_is_the_point_charge_law():
    c = HPConstants()
    hole = _hole()
    w = hole.charge_mass
    centre = hole.charge_centre
    for direction in ([1.0, 0.0, 0.0], [0.6, 0.0, 0.8], [0.0, 0.0, -1.0]):
        r = 2000.0
        got = hole_ppv(centre + r * np.asarray(direction), hole, c)[0]
        assert got == pytest.approx(c.k * w**c.alpha / r**c.beta, rel=1e-4)
    # and it is not the point-charge law close to the column (smaller: the charge is spread)
    near = hole_ppv(centre + [2.0, 0.0, 0.0], hole, c)[0]
    assert near < 0.8 * c.k * w**c.alpha / 2.0**c.beta


def test_ppv_decays_with_distance():
    hole = _hole()
    r = np.geomspace(0.1, 500.0, 60)
    pts = np.column_stack([r, np.zeros_like(r), np.full_like(r, 102.5)])
    ppv = hole_ppv(pts, hole)
    assert np.all(np.diff(ppv) < 0)
    # inside the hole radius the value is the value at the wall
    assert hole_ppv([0.0, 0.0, 102.5], hole)[0] == pytest.approx(ppv[0])
    assert hole_ppv(pts, _hole(charge_top=11.0)) == pytest.approx(0.0)  # no charge


def _design(delays=(0.0, 4.0, 100.0), **kw):
    holes = [
        _hole(
            id=f"H{i}", collar=(5.0 * i, 0, 110), toe=(5.0 * i, 0, 99), delay_ms=t, row=0, index=i
        )
        for i, t in enumerate(delays)
    ]
    return BlastDesign("t", holes, floor_z=100.0, bench_height=10.0, face_azimuth=180.0, **kw)


def test_field_is_the_maximum_over_holes():
    design = _design()
    per_hole = np.array([hole_ppv(POINTS, h) for h in design.holes])
    assert ppv_field(POINTS, design) == pytest.approx(per_hole.max(axis=0), rel=1e-12)


def test_window_adds_the_holes_of_one_window():
    design = _design()
    c = HPConstants()
    terms = np.array([(hole_ppv(POINTS, h, c) / c.k) ** (1 / c.alpha) for h in design.holes])
    # H0 and H1 fire 4 ms apart (one 8 ms window); H2 fires alone
    want = c.k * np.maximum(terms[0] + terms[1], terms[2]) ** c.alpha
    assert ppv_field(POINTS, design, c, combine="window") == pytest.approx(want, rel=1e-12)
    assert np.all(ppv_field(POINTS, design, c, combine="window") >= ppv_field(POINTS, design, c))
    # a window narrower than the gap is the same as "max"
    assert ppv_field(POINTS, design, c, combine="window", window_ms=1.0) == pytest.approx(
        ppv_field(POINTS, design, c)
    )
    with pytest.raises(ValueError):
        ppv_field(POINTS, _design(delays=(0.0, math.nan)), combine="window")
    with pytest.raises(ValueError):
        ppv_field(POINTS, design, combine="sum")


def test_critical_ppv_and_thresholds_by_hand():
    rock = Rock()  # sigma_t 10 MPa, E 50 GPa, cp 4714.05 m/s
    # 10e6 * 4714.05 / 50e9 = 0.94281 m/s
    assert critical_ppv(rock) == pytest.approx(942.81, abs=0.01)
    th = thresholds_for(rock)
    assert th.as_list() == pytest.approx([377.12, 942.81, 3771.24], abs=0.01)
    assert th.calibrated is False
    # the multiples are the ratios of the Bauer and Calder steps (10 / 25 / 100 in/s)
    bc = BAUER_CALDER.as_list()
    assert bc == [250.0, 635.0, 2540.0]
    assert bc[0] / bc[1] == pytest.approx(0.4, abs=0.01) and bc[2] / bc[1] == pytest.approx(4.0)
    with pytest.raises(ValueError):
        DamageThresholds(500.0, 400.0, 900.0)
    with pytest.raises(ValueError):
        thresholds_for(rock, damaged_factor=1.5)


def test_classify_by_hand():
    th = DamageThresholds(250.0, 635.0, 2540.0)
    ppv = [0.0, 250.0, 250.1, 635.0, 700.0, 2540.0, 1e5, math.nan]
    assert classify(ppv, th).tolist() == [0, 0, 1, 1, 2, 2, 3, -1]


def test_backbreak_is_where_the_ppv_meets_the_threshold():
    design = _design()
    th = thresholds_for(design.rock)
    bb = backbreak(design, thresholds=th)
    assert bb > 0.1
    # face looks south (azimuth 180), so "behind" is north (+y), at mid-charge depth
    behind = np.array([[h.collar[0], bb, 102.5] for h in design.holes])
    assert ppv_field(behind, design).max() == pytest.approx(th.fractured_mm_s, rel=2e-3)


def test_backbreak_is_monotone():
    design = _design()
    limits = [300.0, 600.0, 1200.0, 2400.0, 4800.0]
    depths = [backbreak(design, thresholds=DamageThresholds(v / 2, v, 2 * v)) for v in limits]
    assert all(a > b for a, b in zip(depths, depths[1:], strict=False))
    # more charge per metre: deeper
    by_diameter = []
    for diameter in (0.1, 0.15, 0.2, 0.3):
        holes = [_hole(diameter=diameter, delay_ms=0.0)]
        by_diameter.append(backbreak(BlastDesign("t", holes, face_azimuth=180.0)))
    assert all(a < b for a, b in zip(by_diameter, by_diameter[1:], strict=False))
    # larger K: deeper; a threshold nothing reaches: 0; no charge: NaN
    assert backbreak(design, HPConstants(k=1400.0), th := thresholds_for(design.rock)) > backbreak(
        design, HPConstants(), th
    )
    assert backbreak(design, thresholds=DamageThresholds(1e8, 1e9, 1e10)) == 0.0
    assert math.isnan(backbreak(BlastDesign("t", [_hole(charge_top=11.0)])))


def test_points_in_polygon():
    square = [(0, 0), (10, 0), (10, 10), (0, 10)]
    pts = [(5, 5), (-1, 5), (11, 5), (5, 11), (9.99, 0.01), (5, -0.01)]
    assert points_in_polygon(pts, square).tolist() == [True, False, False, False, True, False]
    assert points_in_polygon(pts, square[::-1]).tolist() == points_in_polygon(pts, square).tolist()
    # concave L shape, closed ring
    ell = [(0, 0), (10, 0), (10, 4), (4, 4), (4, 10), (0, 10), (0, 0)]
    assert points_in_polygon([(2, 8), (8, 2), (8, 8), (4.5, 3.5)], ell).tolist() == [
        True, True, False, True,
    ]  # fmt: skip
    # area by Monte Carlo against the shoelace area (84)
    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, (20000, 2))
    assert points_in_polygon(xy, ell).mean() * 100 == pytest.approx(64.0, rel=0.03)
    with pytest.raises(ValueError):
        points_in_polygon(pts, [(0, 0), (1, 1)])


def test_block_damage():
    design = _design(polygon=[(-2.5, -4.0), (12.5, -4.0), (12.5, 2.0), (-2.5, 2.0)])
    centres = np.array(
        [
            [0.0, -2.0, 105.0],  # inside the block, 2 m from H0
            [0.0, -2.0, 99.0],  # below the floor
            [0.0, -2.0, 111.0],  # above the bench top
            [0.0, 30.0, 105.0],  # far behind: intact
            [10.0, 3.0, 104.0],  # 1 m behind the polygon, 3 m from H2: outside
            [10.5, 0.0, 104.0],  # 0.5 m from H2 (fires at 100 ms)
        ]
    )
    sizes = np.full_like(centres, 1.0)
    out = block_damage(centres, sizes, design)
    assert out["blasted"].tolist() == [True, False, False, False, False, True]
    assert out["ppv_mm_s"] == pytest.approx(ppv_field(centres, design))
    assert out["cls"].tolist() == classify(out["ppv_mm_s"], thresholds_for(design.rock)).tolist()
    assert out["cls"][3] == 0 and math.isnan(out["t_break_ms"][3])
    cp = 4714.05
    assert out["t_break_ms"][0] == pytest.approx(0.0 + 2.0 / cp * 1000, rel=1e-5)
    assert out["t_break_ms"][5] == pytest.approx(100.0 + 0.5 / cp * 1000, rel=1e-5)
    assert out["centre"].shape == (6, 3) and out["size"].shape == (6, 3)
    # a fractured block outside the polygon still gets a break time
    assert out["cls"][1] >= 2 and math.isfinite(out["t_break_ms"][1])
    # untimed holes: the break time is unknown, not zero
    untimed = _design(delays=(math.nan,) * 3, polygon=design.polygon)
    assert math.isnan(block_damage(centres, sizes, untimed)["t_break_ms"][0])
    with pytest.raises(ValueError, match="polygon"):
        block_damage(centres, sizes, _design())
