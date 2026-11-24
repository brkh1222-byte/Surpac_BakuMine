"""Flyrock: Richards-Moore by hand, ballistics against the closed form and scipy solve_ivp."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bakumine.blast import flyrock as fl
from bakumine.blast.model import ANFO, BlastDesign, Hole


def make_design(stemming: float = 2.5, inclined: bool = False) -> BlastDesign:
    """Two rows of two 0.115 m holes, 10 m bench; ``inclined`` leans row 0 at 70 degrees."""
    holes = []
    for row in range(2):
        for i in range(2):
            x, y = 4.0 * i, -3.5 * row
            toe = (x, y, 0.0)
            if inclined and row == 0:
                toe = (x, y + 10.0 / math.tan(math.radians(70.0)), 0.0)
            holes.append(
                Hole(
                    id=f"R{row}H{i}",
                    collar=(x, y, 10.0),
                    toe=toe,
                    diameter=0.115,
                    charge_top=stemming,
                    charge_bottom=10.0,
                    explosive=ANFO,
                    row=row,
                    index=i,
                    burden=3.5,
                    spacing=4.0,
                )
            )
    return BlastDesign(name="t", holes=holes, burden=3.5, spacing=4.0)


def test_ranges_by_hand():
    # m = 25 kg/m, B = 5 m: sqrt(m)/B = 1 -> L = k^2/g = 729 / 9.81 = 74.312 m
    assert float(fl.face_burst_range(25.0, 5.0)) == pytest.approx(74.312, rel=1e-5)
    # soft rock k = 13.5: a quarter of that
    assert float(fl.face_burst_range(25.0, 5.0, k=13.5)) == pytest.approx(74.312 / 4, rel=1e-5)
    # m = 16, SH = 2: (4/2)^2.6 = 6.062866 -> 74.312 * 6.062866 = 450.55 m
    assert float(fl.cratering_range(16.0, 2.0)) == pytest.approx(450.55, rel=1e-4)
    # stemming ejection at 60 degrees from horizontal: * sin(120 deg) = 0.866025
    assert float(fl.stemming_ejection_range(16.0, 2.0, math.radians(60))) == pytest.approx(
        450.55 * 0.866025, rel=1e-4
    )
    assert float(fl.stemming_ejection_range(16.0, 2.0, math.pi / 2)) == pytest.approx(0, abs=1e-9)
    assert float(fl.stemming_ejection_range(16.0, 2.0, math.pi / 4)) == pytest.approx(
        float(fl.cratering_range(16.0, 2.0))
    )
    # launch velocity: k * 2^1.3 = 27 * 2.462289 = 66.482 m/s, and L = v0^2 / g
    v0 = float(fl.launch_velocity(16.0, 2.0))
    assert v0 == pytest.approx(66.482, rel=1e-5)
    assert v0**2 / 9.81 == pytest.approx(float(fl.cratering_range(16.0, 2.0)))
    assert math.isinf(float(fl.cratering_range(16.0, 0.0)))
    assert math.isnan(float(fl.cratering_range(16.0, -1.0)))
    # more confinement: shorter throw
    b = np.array([2.0, 3.0, 4.0, 6.0])
    assert np.all(np.diff(fl.face_burst_range(10.0, b)) < 0)


def test_predict_vertical_holes():
    design = make_design()
    res = fl.predict(design)
    m = math.pi / 4 * 0.115**2 * 850.0  # 8.8289 kg/m
    assert m == pytest.approx(8.8289, rel=1e-4)
    face = 729.0 / 9.81 * (math.sqrt(m) / 3.5) ** 2.6
    crater = 729.0 / 9.81 * (math.sqrt(m) / 2.5) ** 2.6
    assert res.face_burst_m == pytest.approx(face)
    assert res.cratering_m == pytest.approx(crater)
    assert res.stemming_ejection_m == pytest.approx(0.0, abs=1e-9)  # vertical holes
    assert res.max_range_m == pytest.approx(crater)  # stemming 2.5 m < burden 3.5 m
    assert res.governing_mechanism == "cratering"
    assert res.governing_hole == "R0H0"
    assert res.holes["face_burst"].startswith("R0")
    assert res.clearance_equipment_m == pytest.approx(2 * crater)
    assert res.clearance_people_m == pytest.approx(4 * crater)
    assert res.max_launch_velocity == pytest.approx(27.0 * (math.sqrt(m) / 2.5) ** 1.3)
    assert not res.calibrated and res.k == 27.0
    d = res.to_dict()
    assert d["clearance_m"] == pytest.approx(4 * crater) and d["safety_factor"] == 4.0
    assert set(d) == {
        "face_burst_m",
        "cratering_m",
        "stemming_ejection_m",
        "max_range_m",
        "clearance_m",
        "safety_factor",
    }
    soft = fl.predict(design, k=13.5, safety_factor_people=3.0)
    assert soft.max_range_m == pytest.approx(crater / 4)
    assert soft.clearance_people_m == pytest.approx(3 * crater / 4)


def test_predict_options():
    # long stemming: the face governs
    res = fl.predict(make_design(stemming=5.0))
    assert res.governing_mechanism == "face_burst" and res.governing_hole.startswith("R0")
    # inclined front row: stemming ejection along the hole, 70 degrees from horizontal
    inc = fl.predict(make_design(inclined=True))
    m = math.pi / 4 * 0.115**2 * 850.0
    crater = 729.0 / 9.81 * (math.sqrt(m) / 2.5) ** 2.6
    assert inc.stemming_ejection_m == pytest.approx(crater * math.sin(math.radians(140.0)))
    assert inc.holes["stemming_ejection"].startswith("R0")
    # no burden anywhere: the face burst is not evaluated, the rest is
    design = make_design()
    design.burden = math.nan
    for h in design.holes:
        h.burden = math.nan
    res = fl.predict(design)
    assert math.isnan(res.face_burst_m) and res.max_range_m == pytest.approx(crater)
    # a hole charged to the collar has no bound
    assert math.isinf(fl.predict(make_design(stemming=0.0)).max_range_m)
    for h in design.holes:
        h.charge_top = h.charge_bottom
    with pytest.raises(ValueError, match="no charged holes"):
        fl.predict(design)


def test_range_on_level():
    v0 = 40.0
    for deg in (10.0, 30.0, 45.0, 60.0, 85.0):
        a = math.radians(deg)
        assert float(fl.range_on_level(v0, a)) == pytest.approx(v0**2 * math.sin(2 * a) / 9.81)
    # 45 degrees is the maximum on level ground: 1600 / 9.81 = 163.10 m
    assert float(fl.range_on_level(v0, math.pi / 4)) == pytest.approx(163.10, rel=1e-4)
    # horizontal launch from 20 m: t = sqrt(2 h / g) = 2.01928 s -> 80.771 m
    assert float(fl.range_on_level(v0, 0.0, h0=20.0)) == pytest.approx(80.771, rel=1e-5)
    assert math.isnan(float(fl.range_on_level(5.0, math.pi / 4, h0=-10.0)))  # cannot reach


def test_land_level_ground_closed_form():
    angles = np.radians([15.0, 45.0, 75.0])
    v0 = np.column_stack([30.0 * np.cos(angles), np.zeros(3), 30.0 * np.sin(angles)])
    p0 = np.array([[0.0, 0.0, 12.0]] * 3)
    t, p = fl.land(p0, v0, 2.0)  # launch 10 m above the level
    np.testing.assert_allclose(p[:, 0], fl.range_on_level(30.0, angles, h0=10.0), rtol=1e-12)
    np.testing.assert_allclose(p[:, 2], 2.0)
    np.testing.assert_allclose(p0[:, 2] + v0[:, 2] * t - 0.5 * 9.81 * t**2, 2.0, atol=1e-9)
    # the stepped solver (surface object) agrees with the closed form
    t2, p2 = fl.land(p0, v0, Plane(0.0, 0.0, 2.0), dt=0.05)
    np.testing.assert_allclose(t2, t, atol=1e-9)
    np.testing.assert_allclose(p2, p, atol=1e-7)
    # single particle given as vectors; a particle that never comes down within t_max
    t1, p1 = fl.land(p0[0], v0[0], 2.0)
    assert t1.shape == (1,) and p1.shape == (1, 3)
    t3, p3 = fl.land([0.0, 0.0, 0.0], [0.0, 0.0, 1000.0], 0.0, t_max=10.0)
    assert math.isnan(t3[0]) and np.isnan(p3).all()
    # below the level and too slow to reach it; NaN input stays NaN
    t4, _ = fl.land([[0.0, 0.0, -5.0], [np.nan, 0.0, 0.0]], [1.0, 0.0, 2.0], 0.0)
    assert np.isnan(t4).all()
    # starts 1 m below the level but fast enough: lands on the way down
    t5, p5 = fl.land([0.0, 0.0, -1.0], [3.0, 0.0, 20.0], 0.0)
    assert t5[0] == pytest.approx((20.0 + math.sqrt(400.0 - 2 * 9.81)) / 9.81)


class Plane:
    """z = a x + b y + c."""

    def __init__(self, a: float, b: float, c: float) -> None:
        self.a, self.b, self.c = a, b, c

    def z_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return self.a * np.asarray(x) + self.b * np.asarray(y) + self.c


def ivp_landing(p0, v0, plane: Plane, c: float = 0.0) -> tuple[float, np.ndarray]:
    """Independent integration with scipy and a ground-contact event."""

    def rhs(_t, s):
        v = s[3:]
        a = -c * np.linalg.norm(v) * v
        a[2] -= 9.81
        return np.concatenate([v, a])

    def hit(_t, s):
        return s[2] - (plane.a * s[0] + plane.b * s[1] + plane.c)

    hit.terminal = True
    hit.direction = -1
    sol = solve_ivp(rhs, (0.0, 200.0), np.concatenate([p0, v0]), events=hit, rtol=1e-11, atol=1e-12)
    assert sol.t_events[0].size == 1
    return float(sol.t_events[0][0]), sol.y_events[0][0][:3]


def test_land_on_slope_against_solve_ivp():
    plane = Plane(0.1, -0.05, -5.0)
    p0 = np.array([[0.0, 0.0, 10.0], [5.0, 5.0, 0.0], [0.0, 0.0, -5.0]])  # last: on the ground
    v0 = np.array([[20.0, 5.0, 15.0], [-12.0, 3.0, 25.0], [8.0, 0.0, 12.0]])
    t, p = fl.land(p0, v0, plane, dt=0.03)
    for i in range(3):
        t_ref, p_ref = ivp_landing(p0[i], v0[i], plane)
        assert t[i] == pytest.approx(t_ref, abs=1e-7)
        np.testing.assert_allclose(p[i], p_ref, atol=1e-6)
    np.testing.assert_allclose(p[:, 2], plane.z_at(p[:, 0], p[:, 1]), atol=1e-9)


def test_land_with_drag_against_solve_ivp():
    plane = Plane(0.05, 0.0, 0.0)
    drag = fl.Drag(size=np.array([0.05, 0.3]), density=2700.0)
    # c = 3 * 1.225 * 0.47 / (4 * 2700 * 0.05) = 0.0031986 1/m
    assert drag.coefficient[0] == pytest.approx(0.0031986, rel=1e-4)
    p0 = np.array([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]])
    v0 = np.array([[35.0, 0.0, 35.0], [35.0, 0.0, 35.0]])
    t, p = fl.land(p0, v0, plane, drag=drag, dt=0.01)
    for i in range(2):
        t_ref, p_ref = ivp_landing(p0[i], v0[i], plane, float(drag.coefficient[i]))
        assert t[i] == pytest.approx(t_ref, abs=1e-6)
        np.testing.assert_allclose(p[i], p_ref, atol=1e-5)
    # drag shortens the throw, more for the small fragment
    _, p_free = fl.land(p0[:1], v0[:1], plane)
    assert p[0, 0] < p[1, 0] < p_free[0, 0]
    # level ground with drag goes through the stepped solver too
    _, p_lvl = fl.land(p0[:1], v0[:1], 0.0, drag=fl.Drag(size=0.05))
    _, p_lvl_free = fl.land(p0[:1], v0[:1], 0.0)
    assert p_lvl[0, 0] < p_lvl_free[0, 0]


def test_land_surface_with_gaps():
    class Patch:
        def z_at(self, x, y):
            return np.where(np.asarray(x) < 50.0, 0.0, np.nan)  # undefined beyond x = 50

    t, p = fl.land(
        [[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], [[5.0, 0.0, 10.0], [60.0, 0.0, 10.0]], Patch(), t_max=5
    )
    assert np.isfinite(t[0]) and p[0, 2] == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(t[1])  # flies off the surface: no contact, not a guess


def test_trajectory():
    p0, v0 = np.array([1.0, 2.0, 3.0]), np.array([10.0, -4.0, 20.0])
    t, p = fl.trajectory(p0, v0, dt=0.1)
    assert p.shape == (t.size, 3) and t[0] == 0.0
    assert t[-1] == pytest.approx(math.ceil(2 * 20.0 / 9.81 / 0.1) * 0.1)
    expected = p0 + np.outer(t, v0)
    expected[:, 2] -= 0.5 * 9.81 * t**2
    np.testing.assert_allclose(p, expected, atol=1e-12)
    # several particles, with drag, against solve_ivp at the sample times
    drag = fl.Drag(size=0.1)
    c = float(drag.coefficient)
    t, pd = fl.trajectory(np.zeros((2, 3)), [[30.0, 0.0, 30.0], [0.0, 20.0, 40.0]], drag=drag)
    assert pd.shape == (t.size, 2, 3)

    def rhs(_t, s):
        v = s[3:]
        a = -c * np.linalg.norm(v) * v
        a[2] -= 9.81
        return np.concatenate([v, a])

    sol = solve_ivp(rhs, (0, t[-1]), [0, 0, 0, 30.0, 0.0, 30.0], t_eval=t, rtol=1e-11, atol=1e-12)
    np.testing.assert_allclose(pd[:, 0, :], sol.y[:3].T, atol=1e-6)
    _, pf = fl.trajectory(np.zeros(3), [30.0, 0.0, 30.0], t_end=t[-1])
    assert pd[-1, 0, 0] < pf[-1, 0]  # drag: less far at the same time
    with pytest.raises(ValueError):
        fl.trajectory(p0, v0, dt=0.0)
    with pytest.raises(ValueError):
        fl.land(np.zeros((2, 2)), np.zeros((2, 2)), 0.0)
