"""Blast data model: geometry, charge mass and JSON round trip."""

from __future__ import annotations

import math

import pytest

from bakumine.blast.model import ANFO, EMULSION, BlastDesign, Hole, Monitor, Rock


def _hole(**kw):
    base = dict(id="A1", collar=(0, 0, 110), toe=(0, 0, 99), diameter=0.2,
                charge_top=4.0, charge_bottom=11.0, explosive=ANFO)  # fmt: skip
    return Hole(**(base | kw))


def test_charge_mass_is_the_cylinder_of_explosive():
    h = _hole()
    assert h.length == 11.0 and h.stemming == 4.0 and h.charge_length == 7.0
    # pi/4 * 0.2^2 * 850 = 26.7035 kg/m (by hand)
    assert h.linear_density == pytest.approx(26.7035, abs=1e-4)
    assert h.charge_mass == pytest.approx(7 * 26.7035, abs=1e-3)
    assert tuple(h.charge_centre) == (0.0, 0.0, 110 - 7.5)


def test_inclined_hole_points():
    h = _hole(toe=(3.0, 4.0, 98.0), charge_bottom=13.0)  # length 13 (3-4-12 triangle)
    assert h.length == pytest.approx(13.0)
    assert tuple(h.point_at(6.5)) == pytest.approx((1.5, 2.0, 104.0))


def test_bad_holes_are_refused():
    with pytest.raises(ValueError):
        _hole(charge_top=5.0, charge_bottom=4.0)
    with pytest.raises(ValueError):
        _hole(charge_bottom=12.0)
    with pytest.raises(ValueError):
        _hole(diameter=0.0)
    with pytest.raises(ValueError):
        Rock(poisson=0.5)


def test_design_totals_and_round_trip(tmp_path):
    holes = [_hole(id=f"A{i}", collar=(5.0 * i, 0, 110), toe=(5.0 * i, 0, 99), index=i,
                   explosive=EMULSION, delay_ms=25.0 * i) for i in range(4)]  # fmt: skip
    rock = Rock("granite", rock_factor=8.0)
    d = BlastDesign("test", holes, rock, bench_height=10.0, floor_z=100.0,
                    burden=4.0, spacing=5.0, polygon=[(0, 0), (20, 0), (20, 4), (0, 4)],
                    monitors=[Monitor("house", (300.0, 0.0, 110.0), 10.0, 120.0)])  # fmt: skip
    assert d.plan_area == 80.0 and d.volume == 800.0
    mass = 4 * 7.0 * math.pi / 4 * 0.04 * 1150.0
    assert d.explosive_mass == pytest.approx(mass)
    assert d.powder_factor == pytest.approx(mass / 800.0)
    assert d.drilled_length == 44.0 and d.timed
    back = BlastDesign.load(d.save(tmp_path / "d.json"))
    assert back.to_dict() == d.to_dict()
    assert back.holes[2].explosive == EMULSION and back.monitors[0].limit_mm_s == 10.0
    untimed = _hole()
    assert math.isnan(Hole.from_dict(untimed.to_dict()).delay_ms)
