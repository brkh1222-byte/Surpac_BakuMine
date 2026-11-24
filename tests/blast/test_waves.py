"""Wave speeds, pressures, crushed zone and wave sources against hand calculations."""

from __future__ import annotations

import math

import numpy as np
import pytest

from bakumine.blast.model import ANFO, BlastDesign, Hole, Rock
from bakumine.blast.waves import (
    arrival_time,
    borehole_pressure,
    crushed_zone_radius,
    detonation_pressure,
    sources,
    wave_speeds,
)


def _hole(**kw):
    base = dict(id="A1", collar=(0, 0, 110), toe=(0, 0, 99), diameter=0.2,
                charge_top=4.0, charge_bottom=11.0, explosive=ANFO)  # fmt: skip
    return Hole(**(base | kw))


def test_wave_speeds_by_hand():
    # E = 50 GPa, nu = 0.25, rho = 2700: M = E (1-nu) / ((1+nu)(1-2nu)) = 60 GPa,
    # cp = sqrt(60e9 / 2700) = 4714.05; G = 20 GPa, cs = sqrt(20e9 / 2700) = 2721.66
    cp, cs = wave_speeds(Rock())
    assert cp == pytest.approx(4714.05, abs=0.01)
    assert cs == pytest.approx(2721.66, abs=0.01)


@pytest.mark.parametrize("nu", [0.0, 0.15, 0.25, 0.33, 0.45])
def test_speed_ratio_identity(nu):
    cp, cs = wave_speeds(Rock(poisson=nu, youngs_gpa=37.0, density=2550.0))
    assert cs / cp == pytest.approx(math.sqrt((1 - 2 * nu) / (2 * (1 - nu))), rel=1e-12)
    # the same speeds from the Lame constants
    e = 37e9
    lam = e * nu / ((1 + nu) * (1 - 2 * nu))
    mu = e / (2 * (1 + nu))
    assert cp == pytest.approx(math.sqrt((lam + 2 * mu) / 2550.0), rel=1e-12)
    assert cs == pytest.approx(math.sqrt(mu / 2550.0), rel=1e-12)


def test_measured_p_wave_overrides():
    cp, cs = wave_speeds(Rock(p_wave=5000.0, poisson=0.25))
    assert cp == 5000.0
    assert cs == pytest.approx(5000.0 / math.sqrt(3.0))
    with pytest.raises(ValueError):
        wave_speeds(Rock(p_wave=-1.0))


def test_pressures_by_hand():
    # ANFO 850 kg/m3, 4000 m/s: 850 * 16e6 / 4 = 3.4 GPa; borehole 1.7 GPa
    assert detonation_pressure(ANFO) == pytest.approx(3.4e9)
    assert borehole_pressure(ANFO) == pytest.approx(1.7e9)


def test_crushed_zone_by_hand():
    rock = Rock()  # UCS 100 MPa, E 50 GPa, nu 0.25
    hole = _hole()
    # Esen: K = 50e9 / 1.25 = 4e10; CZI = (1.7e9)^3 / (4e10 * 1e16) = 12.2825
    czi = 1.7e9**3 / (4e10 * 1e16)
    assert czi == pytest.approx(12.2825)
    assert crushed_zone_radius(hole, rock) == pytest.approx(0.812 * 0.1 * czi**0.219)
    assert crushed_zone_radius(hole, rock) == pytest.approx(0.1406, abs=2e-4)
    # pressure ratio: 0.1 * sqrt(3.4e9 / 1e8) = 0.5831
    assert crushed_zone_radius(hole, rock, "pressure_ratio") == pytest.approx(0.5831, abs=1e-4)
    # stronger rock: smaller zone, never inside the hole
    strong = Rock(ucs_mpa=5000.0)
    assert crushed_zone_radius(hole, strong) == pytest.approx(0.1)
    assert crushed_zone_radius(hole, Rock(ucs_mpa=50.0)) > crushed_zone_radius(hole, rock)
    with pytest.raises(ValueError):
        crushed_zone_radius(hole, rock, "other")


def test_arrival_time():
    pts = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, -5000.0]])
    t = arrival_time(pts, (0.0, 0.0, 0.0), 0.025, 5000.0)
    assert t == pytest.approx([0.026, 0.025, 1.025])
    assert float(arrival_time((3.0, 4.0, 0.0), (0, 0, 0), 0.0, 5.0)) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        arrival_time(pts, (0, 0, 0), 0.0, 0.0)


def test_sources_one_per_charged_hole_in_firing_order():
    holes = [
        _hole(id="A1", delay_ms=42.0),
        _hole(id="A2", collar=(5, 0, 110), toe=(5, 0, 99), delay_ms=0.0),
        _hole(id="A3", collar=(10, 0, 110), toe=(10, 0, 99), charge_top=11.0),  # no charge
    ]
    out = sources(BlastDesign("t", holes))
    assert [s["hole"] for s in out] == ["A2", "A1"]
    assert out[0]["t_ms"] == 0.0 and out[1]["t_ms"] == 42.0
    assert out[1]["centre"] == [0.0, 0.0, 102.5]
    assert out[1]["charge_kg"] == pytest.approx(7 * math.pi / 4 * 0.04 * 850)


def test_sources_need_delays():
    with pytest.raises(ValueError, match="delay_ms"):
        sources(BlastDesign("t", [_hole()]))
