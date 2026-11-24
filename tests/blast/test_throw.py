"""Throw and muckpile: ballistic landing against scipy's ODE solver, volume and repose
slope of the pile, Rosin-Rammler sampling, determinism."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from bakumine.blast.model import ANFO, BlastDesign, Hole, Rock
from bakumine.blast.throw import (
    Heightmap,
    ThrowParams,
    _ballistic_landing,
    deposit,
    land,
    max_slope_excess,
    particles,
    relax,
    rosin_rammler_size,
    simulate_throw,
)

G = 9.81


def _design(top=110.0, floor=100.0, rock=None, timed=True):
    holes = []

    def delay(i, row):
        return 25.0 * i + 42.0 * row if timed else math.nan

    for row in range(2):
        for i in range(3):
            x, y = 2.5 + 5.0 * i, 2.0 + 4.0 * row
            holes.append(
                Hole(f"R{row}H{i}", (x, y, top), (x, y, floor - 1.0), 0.15, 3.0, top - floor + 1.0,
                     ANFO, row=row, index=i, delay_ms=delay(i, row))
            )  # fmt: skip
    return BlastDesign(
        "t", holes, rock=rock or Rock(), bench_height=top - floor, floor_z=floor, burden=4.0,
        spacing=5.0, face_azimuth=180.0, polygon=[(0, 0), (15, 0), (15, 8), (0, 8)],
    )  # fmt: skip


def _ivp_landing(p0, v0, surface):
    """Independent flight: integrate the equations of motion until the ground is met."""

    def rhs(_t, s):
        return [s[3], s[4], s[5], 0.0, 0.0, -G]

    def hit(_t, s):
        return s[2] - surface(s[0], s[1])

    hit.terminal = True
    hit.direction = -1
    sol = solve_ivp(rhs, (0.0, 100.0), [*p0, *v0], events=hit, rtol=1e-11, atol=1e-12)
    return sol.t_events[0][0], sol.y_events[0][0][:3]


SHOTS = [
    ((0.0, 0.0, 10.0), (12.0, 3.0, 8.0)),
    ((1.0, -2.0, 3.0), (-6.0, 9.0, 0.5)),
    ((5.0, 5.0, 25.0), (20.0, 0.0, -3.0)),
    ((0.0, 0.0, 4.0), (0.5, 0.2, 15.0)),
]


@pytest.mark.parametrize(
    "surface",
    [lambda x, y: 2.0 + 0.0 * x, lambda x, y: 0.1 * x + 0.05 * y, lambda x, y: -0.3 * x + 1.0],
    ids=["flat", "slope", "downhill"],
)
def test_landing_matches_solve_ivp(surface):
    p0 = np.array([s[0] for s in SHOTS])
    v0 = np.array([s[1] for s in SHOTS])
    t, p = _ballistic_landing(p0, v0, surface, z_min=0.0, g=G)
    for k, (a, b) in enumerate(SHOTS):
        t_ref, p_ref = _ivp_landing(a, b, surface)
        assert t[k] == pytest.approx(t_ref, rel=1e-8)
        assert p[k] == pytest.approx(p_ref, abs=1e-6)


def test_landing_on_flat_ground_closed_form():
    # z0 = 10, vz = 8, ground 0: t = (8 + sqrt(64 + 2 g 10)) / g
    t, p = land([(0.0, 0.0, 10.0)], [(12.0, 3.0, 8.0)], 0.0, g=G)
    want = (8 + math.sqrt(64 + 2 * G * 10)) / G
    assert t[0] == pytest.approx(want, rel=1e-12)
    assert p[0] == pytest.approx([12 * want, 3 * want, 0.0])
    # starting under the ground: lands at once, on the ground
    t, p = land([(1.0, 2.0, -5.0)], [(3.0, 0.0, 9.0)], 0.0)
    assert t[0] == 0.0 and p[0].tolist() == [1.0, 2.0, 0.0]


def test_landing_on_a_heightmap_step():
    # ground 0 for x < 10, a 5 m high bench beyond (ramp between the nodes x = 10 and 11)
    z = np.zeros((31, 3))
    z[11:, :] = 5.0
    grid = Heightmap((0.0, 0.0), 1.0, z)
    t, p = land([(2.0, 1.0, 1.0), (2.0, 1.0, 1.0)], [(10.0, 0.0, 12.0), (3.0, 0.0, 2.0)], grid)
    # the fast one clears the edge and lands on the bench top, the slow one on the floor
    assert p[0][2] == pytest.approx(5.0) and p[0][0] > 11.0
    assert p[1][2] == pytest.approx(0.0) and p[1][0] < 10.0
    for k, v in enumerate([(10.0, 0.0, 12.0), (3.0, 0.0, 2.0)]):
        t_ref, p_ref = _ivp_landing((2.0, 1.0, 1.0), v, lambda x, y: float(grid.z_at(x, y)))
        assert t[k] == pytest.approx(t_ref, rel=1e-6)
        assert p[k] == pytest.approx(p_ref, abs=1e-5)


def test_heightmap_is_bilinear():
    xx, yy = np.meshgrid(np.arange(4) * 2.0 + 10.0, np.arange(5) * 2.0 - 3.0, indexing="ij")
    grid = Heightmap((10.0, -3.0), 2.0, 0.5 * xx - 0.25 * yy + 1.0)
    x, y = np.array([10.0, 11.3, 15.9, 16.0]), np.array([-3.0, 0.1, 4.4, 5.0])
    assert grid.z_at(x, y) == pytest.approx(0.5 * x - 0.25 * y + 1.0)
    assert grid.z_at(100.0, 0.0) == pytest.approx(0.5 * 16.0 + 1.0)  # clamped to the edge
    assert [int(v) for v in grid.cell_of(13.1, -2.1)] == [2, 0]


def test_rosin_rammler_inverse():
    assert rosin_rammler_size(0.5, 0.3, 1.4) == pytest.approx(0.3)
    u = np.array([0.05, 0.2, 0.5, 0.8, 0.99])
    x = rosin_rammler_size(u, 0.3, 1.4)
    assert 1 - np.exp(-math.log(2) * (x / 0.3) ** 1.4) == pytest.approx(u)
    # characteristic size: 63.2 % passing at xc = x50 / ln(2)^(1/n)
    assert rosin_rammler_size(1 - math.exp(-1), 0.3, 1.4) == pytest.approx(
        0.3 / math.log(2) ** (1 / 1.4)
    )
    with pytest.raises(ValueError):
        rosin_rammler_size(0.5, 0.0, 1.0)


def test_particles_fill_the_block_and_start_after_their_hole():
    design = _design()
    parts = particles(design, 3000, seed=3, x50=0.3, uniformity=1.4)
    p0 = parts["p0"]
    assert p0.shape == (3000, 3) and parts["v0"].shape == (3000, 3)
    assert p0[:, 0].min() >= 0 and p0[:, 0].max() <= 15
    assert p0[:, 1].min() >= 0 and p0[:, 1].max() <= 8
    assert p0[:, 2].min() >= 100 and p0[:, 2].max() <= 110
    # uniform: the centre of the samples is the centre of the box, 15 x 8 x 10
    assert p0.mean(axis=0) == pytest.approx([7.5, 4.0, 105.0], abs=0.3)
    assert parts["blasted_volume_m3"] == pytest.approx(1200.0, rel=1e-12)
    # nearest hole in plan, by brute force
    collars = np.array([h.collar[:2] for h in design.holes])
    brute = np.argmin(np.linalg.norm(p0[:, None, :2] - collars[None], axis=2), axis=1)
    assert np.array_equal(parts["hole"], brute)
    delay = np.array([h.delay_ms for h in design.holes])[brute]
    wait = parts["t0_ms"] - delay
    assert np.all(wait >= 0) and wait.max() < 12.0 / 4714.0 * 1000  # P wave over < 12 m
    # median fragment size is x50
    assert np.median(parts["size_m"]) == pytest.approx(0.3, rel=0.08)


def test_launch_velocity_model():
    design = _design()
    quiet = ThrowParams(velocity_scatter=0.0, angle_scatter_deg=0.0)
    parts = particles(design, 500, seed=1, x50=0.3, uniformity=1.4, params=quiet)
    v, p0 = parts["v0"], parts["p0"]
    row = np.array([h.row for h in design.holes])[parts["hole"]]
    speed = np.linalg.norm(v, axis=1)
    assert speed == pytest.approx(15.0 * 0.8**row)
    # azimuth 180: thrown to the south (-y), nothing east-west
    assert np.all(v[:, 1] < 0) and v[:, 0] == pytest.approx(0.0, abs=1e-12)
    elevation = np.degrees(np.arcsin(v[:, 2] / speed))
    assert elevation == pytest.approx(5.0 + 20.0 * (p0[:, 2] - 100.0) / 10.0)
    # azimuth 90: to the east
    design.face_azimuth = 90.0
    east = particles(design, 50, seed=1, x50=0.3, uniformity=1.4, params=quiet)["v0"]
    assert np.all(east[:, 0] > 0) and east[:, 1] == pytest.approx(0.0, abs=1e-12)
    # large fragments are slowed only when asked
    slow = ThrowParams(velocity_scatter=0.0, angle_scatter_deg=0.0, size_exponent=0.5)
    parts = particles(design, 500, seed=1, x50=0.3, uniformity=1.4, params=slow)
    factor = np.minimum(1.0, (0.3 / parts["size_m"]) ** 0.5)
    assert np.linalg.norm(parts["v0"], axis=1) == pytest.approx(15.0 * 0.8**row * factor)


def test_particles_refuse_incomplete_designs():
    with pytest.raises(ValueError, match="delay_ms"):
        particles(_design(timed=False), 10, x50=0.3, uniformity=1.4)
    design = _design()
    design.polygon = []
    with pytest.raises(ValueError, match="polygon"):
        particles(design, 10, x50=0.3, uniformity=1.4)
    with pytest.raises(ValueError):
        ThrowParams(row_factor=1.5)


def test_relax_cone_conserves_volume_and_obeys_repose():
    tan = math.tan(math.radians(37.0))
    base = np.zeros((41, 41))
    t0 = np.zeros((41, 41))
    t0[20, 20] = 60.0  # a 60 m3 column on a 1 m grid
    t, iterations, converged = relax(base, t0, 1.0, tan, tol=1e-5)
    assert converged and iterations > 1
    assert t.sum() == pytest.approx(60.0, rel=1e-12)
    assert t.min() >= 0.0
    assert max_slope_excess(base, t, 1.0, tan) < 1e-3
    assert max_slope_excess(base, t0, 1.0, tan) > 50.0
    # a cone of volume V at the repose angle has height (3 V tan^2 / pi)^(1/3) = 3.19 m;
    # the grid heap is 8-sided and its apex one cell wide, so only roughly that
    assert t.max() == pytest.approx((3 * 60.0 * tan**2 / math.pi) ** (1 / 3), rel=0.2)
    assert t[20, 20] == t.max()
    # the heap stays where the rock fell (the sweep order makes it slightly lopsided)
    ii, jj = np.meshgrid(np.arange(41), np.arange(41), indexing="ij")
    assert (t * ii).sum() / 60.0 == pytest.approx(20.0, abs=0.25)
    assert (t * jj).sum() / 60.0 == pytest.approx(20.0, abs=0.25)


def test_relax_keeps_the_ground_and_stable_piles():
    tan = math.tan(math.radians(37.0))
    base = np.zeros((12, 6))
    base[6:, :] = 10.0  # a bench wall steeper than repose: bare ground does not move
    flat = np.full((12, 6), 0.5)
    t, iterations, converged = relax(base, flat, 1.0, tan)
    assert converged
    assert t.sum() == pytest.approx(flat.sum(), rel=1e-12)
    # the layer on the bench slides down over the wall until what is left is no steeper
    assert t[6, 0] < 0.5 and t[5, 0] > 0.5
    assert max_slope_excess(base, t, 1.0, tan) < 1e-3 or t[6].max() < 1e-3
    # an even layer on level ground is left alone
    t, iterations, _ = relax(np.zeros((5, 5)), np.full((5, 5), 2.0), 1.0, tan)
    assert iterations == 1 and np.all(t == 2.0)
    with pytest.raises(ValueError):
        relax(np.zeros((3, 3)), -np.ones((3, 3)), 1.0, tan)


def test_deposit_conserves_volume():
    grid = Heightmap((0.0, 0.0), 2.0, np.zeros((4, 4)))
    xy = np.array([[0.9, 0.9], [1.1, 0.9], [1.1, 0.9], [100.0, -50.0]])
    t = deposit(grid, xy, 8.0)
    assert t[0, 0] == 2.0 and t[1, 0] == 4.0 and t[3, 0] == 2.0  # 8 m3 on a 4 m2 cell
    assert t.sum() * 4.0 == pytest.approx(32.0)


@pytest.fixture(scope="module")
def result():
    return simulate_throw(_design(), 100.0, x50=0.3, uniformity=1.4, n=1500, seed=7)


def test_muckpile_volume_is_swell_times_blasted_volume(result):
    s, pile = result["summary"], result["muckpile"]
    nx, ny = pile["shape"]
    z_before = pile["z_before"].reshape(nx, ny)
    z_after = pile["z_after"].reshape(nx, ny)
    assert z_before.shape == (nx, ny) and np.all(z_before == 100.0)
    # ground was level at the floor, so the pile is z_after - z_before
    volume = (z_after - z_before).sum() * pile["spacing"] ** 2
    assert s["blasted_volume_m3"] == pytest.approx(15 * 8 * 10)
    assert volume == pytest.approx(1.3 * 1200.0, rel=1e-9)  # well inside the 1 % asked
    assert s["swell_check"] == pytest.approx(1.3, rel=1e-9)
    assert s["pile_volume_m3"] == pytest.approx(volume)
    assert s["landed_outside_grid"] == 0 and s["volume_on_unknown_ground_m3"] == 0.0
    assert np.all(z_after >= z_before - 1e-12)
    assert s["muckpile_max_height_m"] == pytest.approx((z_after - z_before).max())


def test_muckpile_slopes_do_not_exceed_repose(result):
    s, pile = result["summary"], result["muckpile"]
    assert s["relax_converged"]
    nx, ny = pile["shape"]
    z = pile["z_after"].reshape(nx, ny)
    d = pile["spacing"]
    tan = math.tan(math.radians(37.0))
    # independent of the module's own check: level ground, so every slope is a pile slope
    slopes = [
        np.abs(np.diff(z, axis=0)).max() / d,
        np.abs(np.diff(z, axis=1)).max() / d,
        np.abs(z[1:, 1:] - z[:-1, :-1]).max() / (d * math.sqrt(2)),
        np.abs(z[1:, :-1] - z[:-1, 1:]).max() / (d * math.sqrt(2)),
    ]
    assert max(slopes) <= tan + 5e-3
    assert s["max_slope_excess"] <= 5e-3


def test_particles_land_on_the_ground_and_rest_on_the_pile(result):
    p = result["throw"]["particles"]
    pile = result["muckpile"]
    nx, ny = pile["shape"]
    assert result["throw"]["g"] == 9.81
    assert p["p_land"][:, 2] == pytest.approx(100.0, abs=1e-9)  # the ground it was given
    flight = (p["t_land_ms"] - p["t0_ms"]) / 1000.0
    assert np.all(flight > 0)
    # the landing point is the ballistic position at the landing time
    want = p["p0"] + p["v0"] * flight[:, None]
    want[:, 2] -= 0.5 * 9.81 * flight**2
    assert p["p_land"] == pytest.approx(want, abs=1e-6)
    after = Heightmap(tuple(pile["origin"]), pile["spacing"], pile["z_after"].reshape(nx, ny))
    assert p["p_rest"][:, :2] == pytest.approx(p["p_land"][:, :2])
    assert p["p_rest"][:, 2] == pytest.approx(after.z_at(p["p_land"][:, 0], p["p_land"][:, 1]))
    assert np.all(p["kind"] == 0) and len(p["size_m"]) == 1500
    # thrown south (azimuth 180), a few metres to a few tens of metres
    s = result["summary"]
    assert p["p_land"][:, 1].mean() < p["p0"][:, 1].mean()
    assert 3.0 < s["throw_com_m"] < s["throw_max_m"] < 15.0**2 * 1.3**2 / 9.81 + 30.0


def test_same_seed_same_result(result):
    again = simulate_throw(_design(), 100.0, x50=0.3, uniformity=1.4, n=1500, seed=7)
    assert np.array_equal(again["muckpile"]["z_after"], result["muckpile"]["z_after"])
    for key in ("p0", "v0", "p_land", "p_rest", "size_m", "t_land_ms"):
        assert np.array_equal(again["throw"]["particles"][key], result["throw"]["particles"][key])
    other = simulate_throw(_design(), 100.0, x50=0.3, uniformity=1.4, n=300, seed=8, max_iter=50)
    assert not np.array_equal(
        other["throw"]["particles"]["p0"][:300], result["throw"]["particles"]["p0"][:300]
    )


class _Slope:
    """A ground object with ``z_at`` like the DTM: a bench cut into a hillside, NaN far east."""

    def z_at(self, x, y):
        x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        z = np.where(y >= 0.0, 110.0, 100.0 + 0.05 * y)  # bench top north of the face
        return np.where(x > 40.0, np.nan, z)


def test_throw_on_a_surface_object_with_gaps():
    out = simulate_throw(
        _design(), _Slope(), x50=0.3, uniformity=1.4, n=600, seed=2, spacing=1.0,
        origin=(-20.0, -60.0), shape=(71, 81),
    )  # fmt: skip
    pile, s, p = out["muckpile"], out["summary"], out["throw"]["particles"]
    z_before = pile["z_before"].reshape(71, 81)
    z_after = pile["z_after"].reshape(71, 81)
    assert np.isnan(z_before[65, 10]) and np.isnan(z_after[65, 10])  # unknown stays unknown
    assert z_before[25, 62] == 110.0  # node (5, 2): inside the block, bench top before
    assert z_after[25, 62] < 110.0  # the block is gone
    assert s["swell_check"] == pytest.approx(1.3, rel=1e-9)
    assert s["relax_converged"] and s["max_slope_excess"] <= 5e-3
    # landing points lie on the ground with the block removed
    base = np.where(np.isfinite(z_before), z_before, 100.0)
    xx, yy = np.meshgrid(-20.0 + np.arange(71), -60.0 + np.arange(81), indexing="ij")
    inside = (xx >= 0) & (xx < 15) & (yy >= 0) & (yy < 8)  # nodes on the west / south edge count
    base[inside] = 100.0
    ground = Heightmap((-20.0, -60.0), 1.0, base)
    assert p["p_land"][:, 2] == pytest.approx(ground.z_at(p["p_land"][:, 0], p["p_land"][:, 1]))
    assert np.all(p["p_rest"][:, 2] >= p["p_land"][:, 2] - 1e-9)
