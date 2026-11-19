"""Acceptance tests for P4.T4 inverse distance weighting (research.md 7.3)."""

import time

import numpy as np
import pytest

from bakumine.core.estimation.idw import IDWResult, inverse_distance
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams


# ----------------------------------------------------------------- reference
def aniso_distance(samples, point, ellipsoid):
    """research.md 7.5, written out: project on the axes, stretch, norm."""
    diff = samples - point
    u = diff @ ellipsoid.major_axis
    v = diff @ ellipsoid.semi_major_axis
    w = diff @ ellipsoid.minor_axis
    return np.sqrt(u**2 + (v * ellipsoid.semi_major_ratio) ** 2 + (w * ellipsoid.minor_ratio) ** 2)


def brute_select(samples, holes, target, ellipsoid, p):
    """Selection rule of search.py's docstring as a plain Python loop.
    Returns (indices, distances, estimated)."""
    radius = ellipsoid.major_radius if p.radius is None else p.radius
    diff = samples - target
    d = aniso_distance(samples, target, ellipsoid)
    bits = {None: 0, "quadrant": 2, "octant": 3}[p.sectors]
    sx = diff @ ellipsoid.semi_major_axis >= 0
    sy = diff @ ellipsoid.major_axis >= 0
    sz = diff @ ellipsoid.minor_axis >= 0
    chosen, per_hole, per_sector = [], {}, {}
    for i in np.lexsort((np.arange(len(d)), d)):
        if d[i] > radius or len(chosen) == p.max_samples:
            break
        if p.exclude_zero_distance and d[i] == 0.0:
            continue
        if p.max_vertical is not None and abs(diff[i, 2]) > p.max_vertical:
            continue
        sector = 0
        if bits:
            sector = int(sx[i]) + 2 * int(sy[i]) + (4 * int(sz[i]) if bits == 3 else 0)
            if p.max_per_sector is not None and per_sector.get(sector, 0) >= p.max_per_sector:
                continue
        if p.max_per_hole is not None and per_hole.get(holes[i], 0) >= p.max_per_hole:
            continue
        chosen.append(i)
        per_hole[holes[i]] = per_hole.get(holes[i], 0) + 1
        per_sector[sector] = per_sector.get(sector, 0) + 1
    ok = len(chosen) >= p.min_samples
    if p.min_sectors is not None:
        ok = ok and len(per_sector) >= p.min_sectors
    idx = np.array(chosen, dtype=np.int64)
    return idx, d[idx], ok


def brute_point_idw(d, z, power):
    """Textbook formula with explicit 1 / d^p weights."""
    if (d == 0.0).any():
        return z[d == 0.0].mean()
    w = 1.0 / d**power
    return (w * z).sum() / w.sum()


def brute_idw(samples, values, holes, targets, ellipsoid, p, power):
    est = np.full(len(targets), np.nan)
    n = np.zeros(len(targets), dtype=np.int32)
    mean_d = np.full(len(targets), np.nan)
    near_d = np.full(len(targets), np.nan)
    for t, target in enumerate(targets):
        idx, d, ok = brute_select(samples, holes, target, ellipsoid, p)
        n[t] = len(idx)
        if len(idx):
            mean_d[t], near_d[t] = d.mean(), d.min()
        if ok and len(idx):
            est[t] = brute_point_idw(d, values[idx], power)
    return est, n, mean_d, near_d


# --------------------------------------------------------------- hand cases
def test_hand_example_isotropic_power_2():
    # distances 3, 4, 6  ->  weights 1/9, 1/16, 1/36 = (16, 9, 4) / 144
    samples = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, -6.0]])
    values = np.array([10.0, 20.0, 30.0])
    res = inverse_distance(samples, values, np.zeros((1, 3)), Ellipsoid(100.0), power=2.0)
    # (16*10 + 9*20 + 4*30) / (16 + 9 + 4) = 460 / 29
    assert res.estimate[0] == pytest.approx(460.0 / 29.0, rel=1e-14)
    assert res.n_samples[0] == 3
    assert res.mean_distance[0] == pytest.approx(13.0 / 3.0)
    assert res.nearest_distance[0] == pytest.approx(3.0)
    assert isinstance(res, IDWResult) and res.estimated[0] and res.n_coincident[0] == 0


def test_hand_example_anisotropic_power_1():
    # bearing 0: major axis north, semi-major east (ratio 2), minor up (ratio 3)
    # sample 3 m east -> d = 6; 4 m north -> d = 4; 2 m up -> d = 6; 5 m north -> d = 5
    ell = Ellipsoid(100.0, 2.0, 3.0)
    samples = np.array([[3.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 2.0], [0.0, -5.0, 0.0]])
    values = np.array([10.0, 20.0, 30.0, 40.0])
    res = inverse_distance(samples, values, np.zeros((1, 3)), ell, power=1.0)
    # weights 1/6, 1/4, 1/6, 1/5 = (10, 15, 10, 12) / 60
    # (10*10 + 15*20 + 10*30 + 12*40) / 47 = 1180 / 47
    assert res.estimate[0] == pytest.approx(1180.0 / 47.0, rel=1e-14)
    assert res.mean_distance[0] == pytest.approx(21.0 / 4.0)
    assert res.nearest_distance[0] == pytest.approx(4.0)
    # power 3: weights 1/216, 1/64, 1/216, 1/125 -> (1000, 3375, 1000, 1728) / 216000
    res3 = inverse_distance(samples, values, np.zeros((1, 3)), ell, power=3.0)
    assert res3.estimate[0] == pytest.approx(
        (1000 * 10 + 3375 * 20 + 1000 * 30 + 1728 * 40) / 7103.0, rel=1e-14
    )


# --------------------------------------------------------------- brute force
OPTIONS = {
    "plain": dict(max_samples=8),
    "radius_min": dict(max_samples=12, min_samples=4, radius=35.0),
    "per_hole": dict(max_samples=10, max_per_hole=2),
    "octant": dict(max_samples=16, sectors="octant", max_per_sector=2, min_sectors=3),
    "quadrant": dict(max_samples=8, sectors="quadrant", max_per_sector=3, min_sectors=2),
    "vertical": dict(max_samples=8, max_vertical=6.0),
    "no_zero": dict(max_samples=8, exclude_zero_distance=True),
    "all": dict(
        max_samples=12,
        min_samples=3,
        radius=45.0,
        max_per_hole=3,
        sectors="octant",
        max_per_sector=3,
        min_sectors=2,
        max_vertical=20.0,
        exclude_zero_distance=True,
    ),
}


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(11)
    samples = rng.uniform(0.0, 150.0, (900, 3))
    values = rng.lognormal(0.0, 1.0, 900)
    values[rng.choice(900, 30, replace=False)] = np.nan
    holes = rng.integers(0, 60, 900)
    targets = np.vstack([rng.uniform(-10.0, 160.0, (500, 3)), samples[:40]])  # 40 on samples
    return samples, values, holes, targets


@pytest.mark.parametrize("power", [1.0, 2.0, 3.0])
@pytest.mark.parametrize("option", list(OPTIONS))
def test_equals_brute_force(cloud, power, option):
    samples, values, holes, targets = cloud
    ell = Ellipsoid(50.0, 1.8, 4.0, angles=(40.0, -25.0, 15.0))
    params = SearchParams(**OPTIONS[option])
    res = inverse_distance(
        samples, values, targets, ell, params, power=power, hole_ids=holes, chunk_size=137
    )
    ok = ~np.isnan(values)
    est, n, mean_d, near_d = brute_idw(
        samples[ok], values[ok], holes[ok], targets, ell, params, power
    )
    assert res.n_samples_excluded == 30
    np.testing.assert_array_equal(np.isnan(res.estimate), np.isnan(est))
    np.testing.assert_allclose(res.estimate, est, rtol=1e-10)
    np.testing.assert_array_equal(res.n_samples, n)
    np.testing.assert_allclose(res.mean_distance, mean_d, rtol=1e-12)
    np.testing.assert_allclose(res.nearest_distance, near_d, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(res.estimated, ~np.isnan(est))
    assert 0 < np.isnan(est).sum() < len(est)  # both outcomes exercised


# ----------------------------------------------------------------- exactness
@pytest.mark.parametrize("power", [0.5, 2.0, 7.0])
def test_exact_at_coincident_samples(power):
    rng = np.random.default_rng(5)
    samples = rng.uniform(0.0, 50.0, (200, 3))
    values = rng.normal(5.0, 2.0, 200)
    ell = Ellipsoid(30.0, 2.0, 3.0, angles=(70.0, 30.0, -20.0))
    res = inverse_distance(samples, values, samples, ell, power=power)
    np.testing.assert_array_equal(res.estimate, values)  # bit-exact
    np.testing.assert_array_equal(res.n_coincident, 1)
    assert (res.nearest_distance == 0.0).all()


def test_several_coincident_samples_give_their_mean():
    samples = np.array([[1.0, 2.0, 3.0], [9.0, 2.0, 3.0], [1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    values = np.array([4.0, 100.0, 8.0, np.nan])  # the NaN duplicate is ignored
    res = inverse_distance(samples, values, samples[:1], Ellipsoid(50.0))
    assert res.estimate[0] == 6.0 and res.n_coincident[0] == 2 and res.n_samples[0] == 3


def test_large_power_approaches_nearest_neighbour():
    rng = np.random.default_rng(8)
    samples = rng.uniform(0.0, 100.0, (500, 3))
    values = rng.lognormal(0.0, 1.0, 500)
    targets = rng.uniform(0.0, 100.0, (2000, 3))
    ell = Ellipsoid(40.0, 2.0, 2.5, angles=(20.0, 10.0, 5.0))
    nn = nearest_neighbour(samples, values, targets, ell)
    assert nn.estimated.sum() > 1000
    errors = []
    for power in (2.0, 8.0, 32.0, 128.0, 2000.0):
        res = inverse_distance(samples, values, targets, ell, power=power)
        np.testing.assert_array_equal(res.estimated, nn.estimated)
        assert np.isfinite(res.estimate[res.estimated]).all()  # no overflow / 0-division
        errors.append(np.nanmean(np.abs(res.estimate - nn.estimate)))
    assert all(a > b for a, b in zip(errors[:-1], errors[1:], strict=True)), errors
    assert errors[-1] < 1e-3 * errors[0], errors
    # in the limit only targets with two almost equidistant samples differ
    close = np.abs(res.estimate - nn.estimate) < 1e-9
    assert close[nn.estimated].mean() > 0.97


def test_low_power_approaches_plain_average():
    rng = np.random.default_rng(9)
    samples = rng.uniform(0.0, 20.0, (30, 3))
    values = rng.normal(size=30)
    res = inverse_distance(
        samples, values, np.full((1, 3), 10.0), Ellipsoid(100.0), SearchParams(30), power=1e-9
    )
    assert res.estimate[0] == pytest.approx(values.mean(), abs=1e-7)


# -------------------------------------------------------------- block support
def lattice(centre, size, disc, rot):
    """Cell centres of a (nx, ny, nz) subdivision of a rotated block."""
    pts = []
    for k in range(disc[2]):
        for j in range(disc[1]):
            for i in range(disc[0]):
                local = np.array(
                    [
                        ((i + 0.5) / disc[0] - 0.5) * size[0],
                        ((j + 0.5) / disc[1] - 0.5) * size[1],
                        ((k + 0.5) / disc[2] - 0.5) * size[2],
                    ]
                )
                pts.append(centre + rot @ local)
    return np.array(pts)


def rot_z(deg):
    c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


@pytest.mark.parametrize("rotated", [False, True])
@pytest.mark.parametrize("per_target_sizes", [False, True])
def test_block_estimate_is_mean_of_point_estimates(rotated, per_target_sizes):
    rng = np.random.default_rng(21)
    samples = rng.uniform(0.0, 100.0, (400, 3))
    values = rng.lognormal(0.0, 1.0, 400)
    targets = rng.uniform(10.0, 90.0, (60, 3))
    ell = Ellipsoid(35.0, 1.5, 3.0, angles=(30.0, 20.0, 10.0))
    params = SearchParams(max_samples=10)
    disc = (3, 2, 4)
    rot = rot_z(30.0) @ np.array([[1.0, 0, 0], [0, 0.8, -0.6], [0, 0.6, 0.8]]) if rotated else None
    if per_target_sizes:
        sizes = rng.choice([5.0, 10.0, 20.0], (60, 3))
    else:
        sizes = np.array([10.0, 12.0, 5.0])
    res = inverse_distance(
        samples,
        values,
        targets,
        ell,
        params,
        power=2.0,
        block_size=sizes,
        discretisation=disc,
        grid_rotation=rot,
        chunk_size=17,
    )
    r = np.eye(3) if rot is None else rot
    for t, target in enumerate(targets):
        idx, d, ok = brute_select(samples, np.zeros(400), target, ell, params)
        size = sizes[t] if per_target_sizes else sizes
        pts = lattice(target, size, disc, r)
        assert len(pts) == 24
        expected = np.mean(
            [brute_point_idw(aniso_distance(samples[idx], p, ell), values[idx], 2.0) for p in pts]
        )
        assert res.estimate[t] == pytest.approx(expected, rel=1e-10) if ok else np.isnan(
            res.estimate[t]
        )
        assert res.n_samples[t] == len(idx)


def test_block_discretisation_one_equals_point_and_coincident_point():
    rng = np.random.default_rng(22)
    samples = rng.uniform(0.0, 100.0, (300, 3))
    values = rng.lognormal(0.0, 1.0, 300)
    targets = rng.uniform(10.0, 90.0, (200, 3))
    ell = Ellipsoid(40.0, 2.0, 2.0, angles=(10.0, 0.0, 0.0))
    point = inverse_distance(samples, values, targets, ell)
    block = inverse_distance(samples, values, targets, ell, block_size=(10.0, 10.0, 5.0))
    np.testing.assert_allclose(block.estimate, point.estimate, rtol=1e-12, equal_nan=True)

    # a sample exactly on one of the 2 discretisation points of a block:
    # that point takes the sample's value, the other is ordinary IDW
    s = np.array([[2.5, 0.0, 0.0], [20.0, 0.0, 0.0]])
    v = np.array([7.0, 1.0])
    res = inverse_distance(
        s, v, np.zeros((1, 3)), Ellipsoid(100.0), block_size=(10.0, 4.0, 4.0),
        discretisation=(2, 1, 1),
    )  # points at x = -2.5 and x = +2.5
    other = (7.0 / 5.0**2 + 1.0 / 22.5**2) / (1 / 5.0**2 + 1 / 22.5**2)
    assert res.estimate[0] == pytest.approx((7.0 + other) / 2.0, rel=1e-13)
    assert res.n_coincident[0] == 1


def test_validation_and_empty():
    ell = Ellipsoid(10.0)
    s, v, t = np.zeros((3, 3)), np.ones(3), np.zeros((2, 3))
    for bad in (0.0, -1.0, np.nan, np.inf):
        with pytest.raises(ValueError):
            inverse_distance(s, v, t, ell, power=bad)
    with pytest.raises(ValueError):
        inverse_distance(s, v, t, ell, block_size=(1.0, 0.0, 1.0))
    with pytest.raises(ValueError):
        inverse_distance(s, v, t, ell, block_size=(1.0, 1.0, 1.0), discretisation=(0, 1, 1))
    with pytest.raises(ValueError):
        inverse_distance(s, v, t, ell, block_size=(1.0, 1.0, 1.0), grid_rotation=np.ones((3, 3)))
    with pytest.raises(ValueError):
        inverse_distance(s, v, t, ell, SearchParams(max_per_hole=1))  # needs hole ids
    res = inverse_distance(np.zeros((0, 3)), np.zeros(0), t, ell)
    assert np.isnan(res.estimate).all() and not res.estimated.any()
    assert inverse_distance(s, v, np.zeros((0, 3)), ell).estimate.shape == (0,)
    # min_samples not met: left NaN, neighbours still counted
    far = inverse_distance(s, v, t, ell, SearchParams(max_samples=8, min_samples=5))
    assert np.isnan(far.estimate).all() and (far.n_samples == 3).all()


# ---------------------------------------------------------------- performance
def test_performance_reduced():
    """plan.md target: 1M targets x 20k samples, max 24 neighbours, plain
    search, under 60 s. Asserted here at 1/10 of the targets with a generous
    bound (the full-size time is measured and reported with the task)."""
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1000.0, (20_000, 3))
    values = rng.lognormal(0.0, 1.0, 20_000)
    targets = rng.uniform(0.0, 1000.0, (100_000, 3))
    ell = Ellipsoid(150.0, 2.0, 4.0, angles=(30.0, 20.0, 10.0))
    params = SearchParams(max_samples=24)
    inverse_distance(samples, values, targets[:100], ell, params)  # compile
    start = time.perf_counter()
    res = inverse_distance(samples, values, targets, ell, params)
    elapsed = time.perf_counter() - start
    assert res.estimated.mean() > 0.9
    assert elapsed < 30.0, f"100k targets took {elapsed:.1f} s"
