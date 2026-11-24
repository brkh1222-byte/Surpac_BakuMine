"""Bench blast design rules and layout (bakumine.blast.design), research_blast_design.md.

Independent checks: the Konya burden against the imperial form of the formula evaluated in
feet and inches, the powder-factor solver against scipy's root finder, hole counts and the
powder factor of a rectangle against hand arithmetic, point-in-polygon against an analytic
description of the shape.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.optimize import brentq

from bakumine.blast import design as bd
from bakumine.blast.model import ANFO, EMULSION, BlastDesign, Rock

ROCK = Rock(density=2700.0)


def rectangle(width: float, depth: float) -> list[tuple[float, float]]:
    return [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)]


# ------------------------------------------------------------------------------ rules


def test_konya_burden_hand_worked():
    """Konya and Walter (1990): B[ft] = (2 SGe / SGr + 1.5) De[in].

    The expected value is derived here by hand arithmetic (not a number quoted from the
    book): ANFO SGe = 0.85, rock SGr = 2.70, De = 4 in = 101.6 mm:
    2 * 0.85 / 2.70 = 0.629630; + 1.5 = 2.129630; * 4 in = 8.518519 ft = 2.596444 m.
    """
    assert bd.burden_konya(0.1016, 2700.0, 850.0) == pytest.approx(2.596444, abs=1e-6)


@pytest.mark.parametrize("d_in, sg_e, sg_r", [(3.0, 0.8, 2.6), (6.5, 1.2, 2.4), (12.25, 1.15, 3.1)])
def test_konya_si_form_equals_imperial_form(d_in, sg_e, sg_r):
    """Second route: evaluate the imperial formula and convert feet to metres."""
    burden_ft = (2.0 * sg_e / sg_r + 1.5) * d_in
    got = bd.burden_konya(d_in * 0.0254, sg_r * 1000.0, sg_e * 1000.0)
    assert got == pytest.approx(burden_ft * 0.3048, rel=1e-12)


def test_ash_burden():
    """Ash (1963): B = Kb d. By hand: 30 * 0.1 m = 3.0 m; 12 in hole, Kb 25 = 25 ft."""
    assert bd.burden_ash(0.1) == pytest.approx(3.0)
    assert bd.burden_ash(12.0 / 12.0, kb=25.0) == pytest.approx(25.0)  # feet in, feet out
    for bad in (19.9, 40.1):
        with pytest.raises(ValueError, match="Kb"):
            bd.burden_ash(0.1, kb=bad)


@pytest.mark.parametrize(
    "q, h, d, m", [(0.5, 10.0, 0.115, 1.15), (0.9, 15.0, 0.2, 1.0), (0.3, 6.0, 0.089, 1.25)]
)
def test_powder_factor_burden_against_scipy_root(q, h, d, m):
    w = math.pi / 4.0 * d**2 * EMULSION.density

    def residual(b: float) -> float:
        charge = w * (h + 0.3 * b - 0.7 * b)
        return charge / (b * m * b * h) - q

    expected = brentq(residual, 0.05, 50.0, xtol=1e-13)
    got = bd.burden_from_powder_factor(q, h, d, EMULSION, spacing_ratio=m)
    assert got == pytest.approx(expected, rel=1e-9)


def test_suggest_konya_by_hand():
    """d = 0.1016 m, H = 10 m, ANFO in 2.7 rock, staggered. By hand from B = 2.596444 m:
    S = 1.15 B = 2.985911, T = 0.7 B = 1.817511, J = 0.3 B = 0.778933, L = 10.778933,
    charge length = 8.961422, w = pi/4 * 0.1016^2 * 850 = 6.891222 kg/m, Q = 61.75515 kg,
    volume per hole = B S H = 77.52752 m3, q = 0.796558 kg/m3, H/B = 3.851421."""
    s = bd.suggest(0.1016, 10.0, ROCK, ANFO)
    assert s.burden == pytest.approx(2.596444, abs=1e-6)
    assert s.spacing == pytest.approx(2.985911, abs=1e-6)
    assert s.stemming == pytest.approx(1.817511, abs=1e-6)
    assert s.subdrill == pytest.approx(0.778933, abs=1e-6)
    assert s.hole_length == pytest.approx(10.778933, abs=1e-6)
    assert s.charge_length == pytest.approx(8.961422, abs=1e-6)
    assert s.charge_per_hole == pytest.approx(61.75515, abs=1e-4)
    assert s.powder_factor == pytest.approx(0.796558, abs=1e-5)
    assert s.powder_factor_kg_t == pytest.approx(0.796558 / 2.7, abs=1e-5)
    assert s.stiffness_ratio == pytest.approx(3.851421, abs=1e-5)
    assert s.warnings == ()
    assert s.to_dict()["warnings"] == []


def test_suggest_methods_and_patterns():
    ash = bd.suggest(0.1, 10.0, ROCK, ANFO, pattern="square", method="ash", kb=25.0)
    assert ash.burden == pytest.approx(2.5)
    assert ash.spacing == pytest.approx(2.5)  # square: S = B
    wide = bd.suggest(0.1, 10.0, ROCK, ANFO, pattern="square", method="ash", spacing_ratio=1.25)
    assert wide.spacing == pytest.approx(1.25 * 3.0)
    pf = bd.suggest(0.115, 10.0, ROCK, EMULSION, method="powder_factor", powder_factor=0.6)
    assert pf.powder_factor == pytest.approx(0.6, rel=1e-12)  # the solver hits the target
    with pytest.raises(ValueError, match="powder_factor"):
        bd.suggest(0.115, 10.0, ROCK, EMULSION, method="powder_factor")
    with pytest.raises(ValueError, match="method"):
        bd.suggest(0.115, 10.0, ROCK, EMULSION, method="magic")
    with pytest.raises(ValueError, match="pattern"):
        bd.suggest(0.115, 10.0, ROCK, EMULSION, pattern="hex")


def test_stiffness_warnings():
    low = bd.suggest(0.25, 6.0, ROCK, ANFO, method="ash")  # B = 7.5, H/B = 0.8
    assert low.stiffness_ratio == pytest.approx(0.8)
    assert any("< 2" in w for w in low.warnings)
    high = bd.suggest(0.05, 12.0, ROCK, ANFO, method="ash")  # B = 1.5, H/B = 8
    assert any("> 4" in w for w in high.warnings)
    assert bd.stiffness_warning(2.0) is None and bd.stiffness_warning(4.0) is None
    # stemming longer than the hole: no charge, said so
    none = bd.suggest(0.3, 2.0, ROCK, ANFO, method="ash", stemming_ratio=1.0)
    assert none.charge_length == 0.0 and none.charge_per_hole == 0.0
    assert any("no room" in w for w in none.warnings)


# ----------------------------------------------------------------------------- layout


def test_points_in_polygon_against_analytic_l_shape():
    """L shape = [0,4]x[0,2] united with [0,2]x[0,4]; the oracle is that description."""
    poly = [(0, 0), (4, 0), (4, 2), (2, 2), (2, 4), (0, 4)]
    rng = np.random.default_rng(7)
    x, y = rng.uniform(-1, 5, 4000), rng.uniform(-1, 5, 4000)
    expected = ((x >= 0) & (x <= 4) & (y >= 0) & (y <= 2)) | (
        (x >= 0) & (x <= 2) & (y >= 0) & (y <= 4)
    )
    assert np.array_equal(bd.points_in_polygon(x, y, poly), expected)
    # vertices, edge points and a reversed ring
    bx, by = np.array([0.0, 4.0, 2.0, 3.0, 2.0, 0.0]), np.array([0.0, 2.0, 2.0, 2.0, 3.0, 1.0])
    assert bd.points_in_polygon(bx, by, poly).all()
    assert np.array_equal(bd.points_in_polygon(x, y, poly[::-1]), expected)


def make(pattern: str, **kw) -> BlastDesign:
    args = dict(
        polygon=rectangle(14.0, 9.0),
        top=110.0,
        floor_z=100.0,
        burden=3.0,
        spacing=3.5,
        pattern=pattern,
        face_azimuth=0.0,
        diameter=0.1,
        explosive=ANFO,
        stemming=2.1,
        subdrill=0.9,
        rock=ROCK,
    )
    args.update(kw)
    return bd.layout(**args)


def test_square_rectangle_hand_count_and_positions():
    """14 m (4 S) wide, 9 m (3 B) deep, face looking north (edge y = 9): rows at y = 6, 3, 0
    with holes at x = 1.75, 5.25, 8.75, 12.25 -> 3 x 4 = 12 holes."""
    d = make("square")
    assert len(d.holes) == 12
    assert [h.id for h in d.holes] == [f"{r}{i}" for r in "ABC" for i in range(1, 5)]
    for h in d.holes:
        assert h.collar[1] == pytest.approx(6.0 - 3.0 * h.row)
        assert h.collar[0] == pytest.approx(1.75 + 3.5 * h.index)
        assert h.collar[2] == 110.0
        assert h.toe == pytest.approx((h.collar[0], h.collar[1], 99.1))
        assert (h.burden, h.spacing, h.subdrill) == (3.0, 3.5, 0.9)
        assert (h.charge_top, h.charge_bottom) == (2.1, pytest.approx(10.9))
        assert math.isnan(h.delay_ms)
    assert d.bench_height == 10.0 and d.pattern == "square" and not d.timed


def test_staggered_rectangle_hand_count():
    """Same rectangle: rows A and C as above (4 holes), row B shifted by S / 2 to
    x = 0, 3.5, 7, 10.5, 14 (both side edges count as inside) -> 4 + 5 + 4 = 13."""
    d = make("staggered")
    by_row = {r: [h for h in d.holes if h.row == r] for r in range(3)}
    assert [len(by_row[r]) for r in range(3)] == [4, 5, 4]
    assert [h.collar[0] for h in by_row[1]] == pytest.approx([0.0, 3.5, 7.0, 10.5, 14.0])
    assert [h.id for h in by_row[1]] == ["B1", "B2", "B3", "B4", "B5"]
    assert [h.collar[0] for h in by_row[2]] == pytest.approx([1.75, 5.25, 8.75, 12.25])


def test_powder_factor_of_rectangle_by_hand():
    """12 holes, each 10.9 m long with 8.8 m of ANFO in a 0.1 m hole:
    w = pi/4 * 0.01 * 850 = 6.675884 kg/m, Q = 58.74778 kg, total 704.9734 kg;
    volume 14 * 9 * 10 = 1260 m3 = 3402 t; q = 0.559503 kg/m3 = 0.207223 kg/t;
    drilled 130.8 m -> 0.103810 m/m3; H/B = 3.333."""
    s = bd.summary(make("square"))
    assert s["holes"] == 12 and s["rows"] == 3
    assert s["drilled_m"] == pytest.approx(130.8)
    assert s["explosive_kg"] == pytest.approx(704.9734, abs=1e-3)
    assert s["volume_m3"] == pytest.approx(1260.0)
    assert s["tonnes"] == pytest.approx(3402.0)
    assert s["powder_factor_kg_m3"] == pytest.approx(0.559503, abs=1e-6)
    assert s["powder_factor_kg_t"] == pytest.approx(0.207223, abs=1e-6)
    assert s["specific_drilling_m_m3"] == pytest.approx(0.103810, abs=1e-6)
    assert s["stiffness_ratio"] == pytest.approx(10.0 / 3.0)


@pytest.mark.parametrize("azimuth", [0.0, 90.0, 137.0, 250.0])
@pytest.mark.parametrize("pattern", ["square", "staggered"])
def test_rotated_bench_same_count_all_inside(azimuth, pattern):
    """Turn the rectangle with the face: the count is that of the unrotated case and every
    collar, taken back to the bench axes, is inside the box; row 0 is one burden from the
    face edge."""
    az = math.radians(azimuth)
    f = np.array([math.sin(az), math.cos(az)])
    u = np.array([math.cos(az), -math.sin(az)])
    origin = np.array([5000.0, 7000.0])
    poly = [tuple(origin + s * u + t * f) for s, t in rectangle(14.0, 9.0)]
    d = make(pattern, polygon=poly, face_azimuth=azimuth)
    assert len(d.holes) == (12 if pattern == "square" else 13)
    xy = np.array([h.collar[:2] for h in d.holes]) - origin
    s, t = xy @ u, xy @ f
    assert ((s > -1e-6) & (s < 14 + 1e-6) & (t > -1e-6) & (t < 9 + 1e-6)).all()
    rows = np.array([h.row for h in d.holes])
    assert t[rows == 0] == pytest.approx(6.0, abs=1e-6)
    assert t[rows == 2] == pytest.approx(0.0, abs=1e-6)
    assert d.volume == pytest.approx(1260.0)


def test_irregular_polygon_all_holes_inside():
    """Triangle with the face on its long north side: every hole satisfies the three
    half-plane inequalities (analytic), rows get shorter toward the back."""
    poly = [(0.0, 30.0), (40.0, 30.0), (20.0, 0.0)]
    d = make("staggered", polygon=poly, burden=3.0, spacing=3.5)
    assert len(d.holes) > 20
    counts = [sum(h.row == r for h in d.holes) for r in range(max(h.row for h in d.holes) + 1)]
    assert counts == sorted(counts, reverse=True) or max(np.diff(counts)) <= 1
    for h in d.holes:
        x, y = h.collar[:2]
        assert y <= 30.0 + 1e-6
        assert y >= 1.5 * abs(x - 20.0) - 1e-6  # sides: y = 1.5 |x - 20|
    assert len({h.id for h in d.holes}) == len(d.holes)


class Plane:
    """Sloping bench top with the ``z_at`` of a DTM; NaN east of x = 13."""

    def z_at(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        return np.where(x > 13.0, np.nan, 110.0 + 0.2 * x - 0.1 * y)


def test_collars_follow_sloping_top():
    d = make("staggered", top=Plane())
    for h in d.holes:
        x, y, z = h.collar
        assert z == pytest.approx(110.0 + 0.2 * x - 0.1 * y)
        assert h.toe[2] == pytest.approx(99.1)
        assert h.length == pytest.approx(z - 99.1)
        assert h.charge_bottom == pytest.approx(h.length)
    # the hole at x = 14 (row B) has no surface: dropped and recorded, never z = NaN
    assert len(d.holes) == 12
    assert [n["reason"] for n in d.notes["dropped_holes"]] == ["no top"]
    assert [h.id for h in d.holes if h.row == 1] == ["B1", "B2", "B3", "B4"]
    mean_z = np.mean([h.collar[2] for h in d.holes])
    assert d.bench_height == pytest.approx(mean_z - 100.0)


def test_inclined_holes_lean_toward_the_face():
    """dip 70 deg, face looking east: L = H / sin(70) + J; toe moved east by L cos(70) and
    down to floor - J sin(70)."""
    d = make("square", polygon=rectangle(9.0, 14.0), face_azimuth=90.0, hole_dip=70.0)
    sin70, cos70 = math.sin(math.radians(70.0)), math.cos(math.radians(70.0))
    length = 10.0 / sin70 + 0.9
    for h in d.holes:
        assert h.length == pytest.approx(length)
        assert h.toe[0] - h.collar[0] == pytest.approx(length * cos70)
        assert h.toe[1] == pytest.approx(h.collar[1])
        assert h.toe[2] == pytest.approx(100.0 - 0.9 * sin70)
        assert h.direction @ np.array([cos70, 0.0, -sin70]) == pytest.approx(1.0)


def test_defaults_and_uncharged_and_errors():
    d = make("square", stemming=None, subdrill=None)
    assert d.holes[0].charge_top == pytest.approx(0.7 * 3.0)
    assert d.holes[0].subdrill == pytest.approx(0.3 * 3.0)
    assert d.notes["layout"]["stemming_rule"] == "0.7 B"
    # stemming not shorter than the hole: kept, uncharged, recorded
    short = make("square", stemming=20.0)
    assert all(h.charge_mass == 0.0 for h in short.holes)
    assert short.notes["uncharged_holes"] == [h.id for h in short.holes]
    assert short.explosive_mass == 0.0
    with pytest.raises(ValueError, match="no row fits"):
        make("square", burden=9.5)
    with pytest.raises(ValueError, match="pattern"):
        make("hexagonal")
    with pytest.raises(ValueError, match="hole_dip"):
        make("square", hole_dip=0.0)
    with pytest.raises(ValueError, match="no hole fits"):
        make("square", top=90.0)  # top below the floor
    with pytest.raises(ValueError, match="three"):
        make("square", polygon=[(0, 0), (1, 1)])


def test_json_round_trip_and_row_letters():
    d = make("staggered", name="bench 1200")
    back = BlastDesign.from_dict(d.to_dict())
    assert [h.id for h in back.holes] == [h.id for h in d.holes]
    assert back.holes[5].collar == d.holes[5].collar
    assert [bd._row_letters(i) for i in (0, 25, 26, 27, 701, 702)] == [
        "A", "Z", "AA", "AB", "ZZ", "AAA",
    ]  # fmt: skip
