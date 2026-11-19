"""Acceptance tests for P4.T6 ordinary and simple kriging (research.md 7.4).

Independent references used here:

* hand-built systems with explicit numbers, solved with ``numpy.linalg``;
* ``dense_krige`` below: a dense numpy solver of the *bordered* system with
  its own covariance function (``ref_cov``), sharing no code with
  ``kriging.py`` or ``variogram.py``;
* PyKrige ``OrdinaryKriging3D`` and gstools ``krige.Ordinary`` (dev
  dependencies, oracles only).
"""

import math
import time

import gstools as gs
import numpy as np
import pytest
from pykrige.ok3d import OrdinaryKriging3D
from scipy.spatial.transform import Rotation

from bakumine.core.estimation.kriging import (
    STATUS_ESTIMATED,
    STATUS_NOT_SELECTED,
    STATUS_SINGULAR,
    KrigingResult,
    block_covariance,
    krige,
    ordinary_kriging,
    simple_kriging,
)
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel

BIG = Ellipsoid(1.0e6)  # isotropic search that reaches everything


def all_samples(n):
    """Search options that put every sample in every neighbourhood."""
    return SearchParams(max_samples=n, radius=1.0e6)


# ----------------------------------------------------------------- reference
def ref_cov(h, model, nugget=True):
    """Covariance of lag vectors h (..., 3), written from research.md 7.4 /
    7.5 with explicit axis projections. ``nugget=False`` gives the
    structures-only covariance used for block terms."""
    h = np.asarray(h, dtype=float)
    cov = np.zeros(h.shape[:-1])
    for s in model.structures:
        e = s.ellipsoid
        a_major, a_semi, a_minor = s.ranges
        r = np.sqrt(
            (h @ e.major_axis / a_major) ** 2
            + (h @ e.semi_major_axis / a_semi) ** 2
            + (h @ e.minor_axis / a_minor) ** 2
        )
        if s.type == "spherical":
            g = np.where(r < 1.0, 1.5 * r - 0.5 * r**3, 1.0)
        elif s.type == "exponential":
            g = 1.0 - np.exp(-3.0 * r)
        else:
            g = 1.0 - np.exp(-3.0 * r**2)
        cov += s.sill * (1.0 - g)
    if nugget:
        cov = cov + model.nugget * (np.abs(h).sum(axis=-1) == 0.0)
    return cov


def lattice(centre, size, disc, rot=None):
    """Cell centres of an (nx, ny, nz) subdivision of a (rotated) block."""
    rot = np.eye(3) if rot is None else rot
    ax = [((np.arange(n) + 0.5) / n - 0.5) * s for n, s in zip(disc, size, strict=True)]
    local = np.array([[x, y, z] for z in ax[2] for y in ax[1] for x in ax[0]])
    return centre + local @ rot.T


def dense_krige(xs, zs, model, target=None, points=None, mean=None):
    """Bordered-system kriging with numpy. ``target`` (3,) = point kriging;
    ``points`` (N, 3) = block kriging over those discretisation points.
    Returns dict(weights, mu, estimate, variance, rhs, cmat, c00)."""
    n = len(xs)
    cmat = ref_cov(xs[:, None, :] - xs[None, :, :], model)
    if points is None:
        rhs = ref_cov(xs - target, model)
        c00 = model.total_sill
    else:
        rhs = ref_cov(xs[:, None, :] - points[None, :, :], model, nugget=False).mean(axis=1)
        c00 = ref_cov(points[:, None, :] - points[None, :, :], model, nugget=False).mean()
    if mean is None:
        a = np.zeros((n + 1, n + 1))
        a[:n, :n] = cmat
        a[:n, n] = a[n, :n] = 1.0
        sol = np.linalg.solve(a, np.append(rhs, 1.0))
        w, mu = sol[:n], sol[n]
        est = w @ zs
        var = c00 - w @ rhs - mu
    else:
        w, mu = np.linalg.solve(cmat, rhs), np.nan
        est = mean + w @ (zs - mean)
        var = c00 - w @ rhs
    return dict(weights=w, mu=mu, estimate=est, variance=var, rhs=rhs, cmat=cmat, c00=c00)


MODELS = {
    "sph_iso": VariogramModel(0.2, (Structure("spherical", 0.8, 60.0),)),
    "sph_aniso": VariogramModel(
        0.2, (Structure("spherical", 0.8, (60.0, 35.0, 15.0), (35.0, 20.0, 50.0)),)
    ),
    "exp_iso": VariogramModel(0.2, (Structure("exponential", 0.8, 60.0),)),
    "exp_aniso": VariogramModel(
        0.2, (Structure("exponential", 0.8, (60.0, 35.0, 15.0), (35.0, 20.0, 50.0)),)
    ),
    "gau_iso": VariogramModel(0.2, (Structure("gaussian", 0.8, 60.0),)),
    "gau_aniso": VariogramModel(
        0.2, (Structure("gaussian", 0.8, (60.0, 35.0, 15.0), (35.0, 20.0, 50.0)),)
    ),
}
NESTED = VariogramModel(
    0.1,
    (
        Structure("spherical", 0.5, (40.0, 25.0, 10.0), (30.0, 20.0, 10.0)),
        Structure("exponential", 0.4, (150.0, 90.0, 30.0), (60.0, -10.0, 0.0), "gslib"),
    ),
)


@pytest.fixture(scope="module")
def cloud():
    rng = np.random.default_rng(1)
    samples = rng.uniform(0.0, 100.0, (150, 3))
    values = rng.lognormal(0.0, 0.7, 150)
    targets = rng.uniform(0.0, 100.0, (300, 3))
    return samples, values, targets


# ------------------------------------------------------------ hand examples
def test_four_sample_ordinary_kriging_solved_by_hand():
    """Spherical, nugget 0.2, sill 0.8, range 100 (isotropic). Target at the
    origin; four samples whose mutual distances are written out below."""

    def cov(d):  # C(h) for h > 0: 0.8 * (1 - 1.5 h/100 + 0.5 (h/100)^3)
        return 0.8 * (1.0 - 1.5 * d / 100.0 + 0.5 * (d / 100.0) ** 3)

    samples = np.array(
        [[30.0, 0.0, 0.0], [0.0, 40.0, 0.0], [-30.0, -40.0, 0.0], [0.0, 0.0, 20.0]]
    )
    values = np.array([1.2, 2.5, 0.7, 3.1])
    d12, d13, d14 = 50.0, math.sqrt(5200.0), math.sqrt(1300.0)
    d23, d24, d34 = math.sqrt(7300.0), math.sqrt(2000.0), math.sqrt(2900.0)
    d01, d02, d03, d04 = 30.0, 40.0, 50.0, 20.0
    c0 = 1.0  # C(0) = nugget + sill
    a = np.array(
        [
            [c0, cov(d12), cov(d13), cov(d14), 1.0],
            [cov(d12), c0, cov(d23), cov(d24), 1.0],
            [cov(d13), cov(d23), c0, cov(d34), 1.0],
            [cov(d14), cov(d24), cov(d34), c0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    b = np.array([cov(d01), cov(d02), cov(d03), cov(d04), 1.0])
    sol = np.linalg.solve(a, b)
    w, mu = sol[:4], sol[4]
    estimate = w @ values
    variance = c0 - w @ b[:4] - mu

    # the same system in the variogram form of research.md 7.4
    def gam(d):
        return 0.2 + 0.8 * (1.5 * d / 100.0 - 0.5 * (d / 100.0) ** 3)

    g = np.array(
        [
            [0.0, gam(d12), gam(d13), gam(d14), 1.0],
            [gam(d12), 0.0, gam(d23), gam(d24), 1.0],
            [gam(d13), gam(d23), 0.0, gam(d34), 1.0],
            [gam(d14), gam(d24), gam(d34), 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0, 0.0],
        ]
    )
    gb = np.array([gam(d01), gam(d02), gam(d03), gam(d04), 1.0])
    gsol = np.linalg.solve(g, gb)
    np.testing.assert_allclose(gsol[:4], w, atol=1e-12)
    assert gsol[4] == pytest.approx(-mu, abs=1e-12)  # multiplier changes sign
    assert gsol[:4] @ gb[:4] + gsol[4] == pytest.approx(variance, abs=1e-12)

    model = VariogramModel(0.2, (Structure("spherical", 0.8, 100.0),))
    res = ordinary_kriging(
        samples, values, np.zeros((1, 3)), model, BIG, all_samples(4), return_weights_for=[0]
    )
    order = res.weight_samples[0]
    np.testing.assert_array_equal(np.sort(order), np.arange(4))
    np.testing.assert_allclose(res.weights[0], w[order], atol=1e-10, rtol=0)
    assert res.estimate[0] == pytest.approx(estimate, abs=1e-10)
    assert res.variance[0] == pytest.approx(variance, abs=1e-10)
    assert res.lagrange[0] == pytest.approx(mu, abs=1e-10)
    assert res.sum_weights[0] == pytest.approx(1.0, abs=1e-10)
    assert res.n_samples[0] == 4 and res.status[0] == STATUS_ESTIMATED
    assert res.mean_distance[0] == pytest.approx(35.0)
    assert res.nearest_distance[0] == pytest.approx(20.0)
    assert isinstance(res, KrigingResult) and res.method == "ordinary" and res.support == "point"


def test_simple_kriging_hand_examples():
    # one sample: w = C(d) / C(0); exponential, sill 2, range 30, no nugget
    model = VariogramModel(0.0, (Structure("exponential", 2.0, 30.0),))
    c_d = 2.0 * math.exp(-3.0 * 10.0 / 30.0)  # d = 10
    w = c_d / 2.0
    res = simple_kriging(
        np.array([[10.0, 0.0, 0.0]]), np.array([5.0]), np.zeros((1, 3)), model, BIG, mean=3.0
    )
    assert res.estimate[0] == pytest.approx(3.0 + w * (5.0 - 3.0), abs=1e-12)
    assert res.variance[0] == pytest.approx(2.0 - c_d**2 / 2.0, abs=1e-12)
    assert res.sum_weights[0] == pytest.approx(w, abs=1e-12)
    assert np.isnan(res.lagrange[0]) and res.method == "simple"
    assert res.slope[0] == pytest.approx(1.0, abs=1e-12)

    # three samples, nugget 0.3 + spherical sill 0.7 range 50; mean 1.5
    def cov(d):
        return 0.7 * (1.0 - 1.5 * d / 50.0 + 0.5 * (d / 50.0) ** 3)

    samples = np.array([[6.0, 8.0, 0.0], [-20.0, 0.0, 0.0], [0.0, 0.0, 15.0]])
    values = np.array([2.0, 0.5, 1.9])
    d12, d13, d23 = math.sqrt(26.0**2 + 64.0), math.sqrt(100.0 + 225.0), 25.0
    c = np.array([[1.0, cov(d12), cov(d13)], [cov(d12), 1.0, cov(d23)], [cov(d13), cov(d23), 1.0]])
    rhs = np.array([cov(10.0), cov(20.0), cov(15.0)])
    w = np.linalg.solve(c, rhs)
    model = VariogramModel(0.3, (Structure("spherical", 0.7, 50.0),))
    res = simple_kriging(
        samples, values, np.zeros((1, 3)), model, BIG, all_samples(3), mean=1.5,
        return_weights_for=[0],
    )  # fmt: skip
    np.testing.assert_allclose(res.weights[0], w[res.weight_samples[0]], atol=1e-10, rtol=0)
    assert res.estimate[0] == pytest.approx(1.5 + w @ (values - 1.5), abs=1e-10)
    assert res.variance[0] == pytest.approx(1.0 - w @ rhs, abs=1e-10)
    assert res.sum_weights[0] == pytest.approx(w.sum(), abs=1e-10)
    # far from all data simple kriging returns the mean with variance C(0)
    far = simple_kriging(samples, values, np.full((1, 3), 500.0), model, BIG, mean=1.5)
    assert far.estimate[0] == pytest.approx(1.5) and far.variance[0] == pytest.approx(1.0)


# ------------------------------------------------- weights and exactness
@pytest.mark.parametrize("name", [*MODELS, "nested"])
def test_ordinary_weights_sum_to_one_and_match_dense_solver(cloud, name):
    samples, values, targets = cloud
    model = NESTED if name == "nested" else MODELS[name]
    ell = Ellipsoid(45.0, 1.5, 2.5, angles=(35.0, 20.0, 50.0))
    params = SearchParams(max_samples=16, min_samples=3)
    rows = np.arange(len(targets))
    res = ordinary_kriging(
        samples, values, targets, model, ell, params, return_weights_for=rows, chunk_size=64
    )
    ok = res.estimated
    assert ok.sum() > 150 and res.n_singular == 0 and res.n_invalid == 0
    np.testing.assert_allclose(np.nansum(res.weights[ok], axis=1), 1.0, atol=1e-10, rtol=0)
    np.testing.assert_allclose(res.sum_weights[ok], 1.0, atol=1e-10, rtol=0)
    for t in np.flatnonzero(ok):
        idx = res.weight_samples[t, : res.n_samples[t]]
        ref = dense_krige(samples[idx], values[idx], model, target=targets[t])
        w = res.weights[t, : res.n_samples[t]]
        np.testing.assert_allclose(w, ref["weights"], atol=1e-8, rtol=0)
        assert res.estimate[t] == pytest.approx(ref["estimate"], abs=1e-8)
        assert res.variance[t] == pytest.approx(ref["variance"], abs=1e-8)
        assert res.lagrange[t] == pytest.approx(ref["mu"], abs=1e-8)
        assert res.estimate[t] == pytest.approx(w @ values[idx], abs=1e-12)
        assert res.sum_negative_weights[t] == pytest.approx(w[w < 0].sum(), abs=1e-12)
        # by-products from their definitions
        var_est = w @ ref["cmat"] @ w
        assert res.slope[t] == pytest.approx(w @ ref["rhs"] / var_est, abs=1e-8)
        assert res.slope[t] == pytest.approx(
            w @ ref["rhs"] / (w @ ref["rhs"] - ref["mu"]), abs=1e-8
        )
        efficiency = 1.0 - ref["variance"] / model.total_sill
        assert res.efficiency[t] == pytest.approx(efficiency, abs=1e-8)
    # unestimated rows carry no numbers
    for field in ("estimate", "variance", "lagrange", "slope", "efficiency", "sum_weights"):
        assert np.isnan(getattr(res, field)[~ok]).all()
    assert (res.status[~ok] == STATUS_NOT_SELECTED).all() and res.n_not_selected == (~ok).sum()


@pytest.mark.parametrize("method", ["ordinary", "simple"])
@pytest.mark.parametrize("kind", ["spherical", "exponential"])
def test_exact_interpolation_without_nugget(cloud, method, kind):
    samples, values, _ = cloud
    model = VariogramModel(0.0, (Structure(kind, 1.3, (60.0, 40.0, 25.0), (30.0, 20.0, 10.0)),))
    mean = 2.0 if method == "simple" else None
    res = krige(
        samples, values, samples, model, Ellipsoid(50.0), SearchParams(12), method=method, mean=mean
    )
    assert res.estimated.all()
    np.testing.assert_allclose(res.estimate, values, atol=1e-10, rtol=0)
    np.testing.assert_allclose(res.variance, 0.0, atol=1e-10, rtol=0)
    assert (res.variance >= 0.0).all()
    np.testing.assert_allclose(res.efficiency, 1.0, atol=1e-10)


def test_nugget_behaviour_at_and_near_a_sample(cloud):
    """Documented behaviour: with a nugget, point kriging is still exact AT a
    sample (value reproduced, variance 0) but discontinuous: 1 mm away the
    estimate is the smoothed one and the variance is at least the nugget."""
    samples, values, _ = cloud
    model = VariogramModel(0.4, (Structure("spherical", 0.6, 60.0),))
    params = SearchParams(12)
    at = ordinary_kriging(samples, values, samples, model, Ellipsoid(50.0), params)
    np.testing.assert_allclose(at.estimate, values, atol=1e-10, rtol=0)
    np.testing.assert_allclose(at.variance, 0.0, atol=1e-10, rtol=0)
    near = ordinary_kriging(samples, values, samples + [1e-3, 0, 0], model, Ellipsoid(50.0), params)
    assert (near.variance >= 0.4).all()
    assert np.abs(near.estimate - values).mean() > 0.1  # smoothed, not reproduced
    # the smoothed estimate is the limit of the dense solution next to the sample
    idx = np.argsort(np.linalg.norm(samples - samples[0], axis=1))[:12]
    ref = dense_krige(samples[idx], values[idx], model, target=samples[0] + [1e-3, 0, 0])
    assert near.estimate[0] == pytest.approx(ref["estimate"], abs=1e-9)
    assert near.variance[0] == pytest.approx(ref["variance"], abs=1e-9)


# ------------------------------------------------------------ PyKrige oracle
def pykrige_parameters(model):
    """Map a nugget + ONE structure BakuMine model to PyKrige's arguments.

    * Variogram function. PyKrige (variogram_models.py, documented formulas):
        spherical    psill (1.5 d/range - 0.5 d^3/range^3) + nugget, d <= range
        exponential  psill (1 - exp(-d / (range/3))) + nugget
        gaussian     psill (1 - exp(-d^2 / (4/7 range)^2)) + nugget
      Spherical and exponential are research.md's forms with range = a.
      PyKrige's gaussian has exponent (49/16) d^2/range^2 where ours has
      3 d^2/a^2, so  range_pykrige = a * 7 / (4 sqrt(3)).
      ``psill`` = partial sill = our structure sill (NOT the total sill; the
      alternative key 'sill' would be nugget + psill).
    * Nugget / zero lag. PyKrige builds the system from -gamma with a zero
      diagonal and sets the right-hand side to 0 at zero distance
      (``exact_values=True``, the default): gamma(0) = 0 and the nugget
      applies to every non-zero lag - the same as C(0) = total sill here.
    * Anisotropy. PyKrige transforms coordinates as
        x_adj = S @ Ez(angle_z) @ Ey(angle_y) @ Ex(angle_x) @ (x - centre)
      with S = diag(1, scaling_y, scaling_z) and each E the right-hand
      rotation about its axis by MINUS the angle (degrees); found by probing
      ``pykrige.core._adjust_for_anisotropy`` with unit vectors. Distances
      are Euclidean in x_adj and ``range`` applies along adjusted x. So the
      rows of Q = Ez Ey Ex must be our (major, semi-major, -minor) axes
      (minor negated for det +1), scaling_y = a_major / a_semi and
      scaling_z = a_major / a_minor. Q = Rz(-az) Ry(-ay) Rx(-ax) is scipy's
      extrinsic 'xyz' Euler form with angles (-ax, -ay, -az).
      The mapping is not trusted on its own: agreement of the kriging
      results to 1e-6 for a generic three-angle rotation is the check.
    """
    (s,) = model.structures
    e = s.ellipsoid
    q = np.vstack([e.major_axis, e.semi_major_axis, -e.minor_axis])
    ax, ay, az = -np.rad2deg(Rotation.from_matrix(q).as_euler("xyz"))
    rng = s.ranges[0] * (7.0 / (4.0 * math.sqrt(3.0)) if s.type == "gaussian" else 1.0)
    return dict(
        variogram_model=s.type,
        variogram_parameters={"psill": s.sill, "range": rng, "nugget": model.nugget},
        anisotropy_scaling_y=s.ranges[0] / s.ranges[1],
        anisotropy_scaling_z=s.ranges[0] / s.ranges[2],
        anisotropy_angle_x=ax,
        anisotropy_angle_y=ay,
        anisotropy_angle_z=az,
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_pykrige_ordinary_kriging_3d(cloud, name):
    """Estimates AND variances against PyKrige on 300 targets, every sample
    in every neighbourhood (so neighbour selection cannot differ)."""
    samples, values, targets = cloud
    targets = targets.copy()
    targets[:5] = samples[:5]  # include targets that coincide with samples
    model = MODELS[name]
    res = ordinary_kriging(samples, values, targets, model, BIG, all_samples(len(samples)))
    assert res.estimated.all() and (res.n_samples == len(samples)).all()
    oracle = OrdinaryKriging3D(
        samples[:, 0], samples[:, 1], samples[:, 2], values, **pykrige_parameters(model)
    )
    est, var = oracle.execute("points", targets[:, 0], targets[:, 1], targets[:, 2])
    np.testing.assert_allclose(res.estimate, np.asarray(est), atol=1e-6, rtol=0)
    np.testing.assert_allclose(res.variance, np.asarray(var), atol=1e-6, rtol=0)
    np.testing.assert_allclose(res.estimate[:5], values[:5], atol=1e-9)
    assert res.variance.max() > 0.3  # not a trivial comparison


# ------------------------------------------------------------ gstools oracle
def gstools_model(model):
    """gstools CovModel of a nugget + one structure model; mapping as in
    tests/estimation/test_variogram.py::to_gstools (verified there against
    1000 random lags): rescale 1 / 3 / sqrt(3) makes len_scale our range,
    len_scale = (major, semi, minor) ranges, and the angles are read off
    Q = [major, semi-major, -minor] (columns) = Rx(gamma) Ry(beta) Rz(alpha);
    the axes are asserted to coincide."""
    (s,) = model.structures
    e = s.ellipsoid
    q = np.column_stack([e.major_axis, e.semi_major_axis, -e.minor_axis])
    beta = math.asin(max(-1.0, min(1.0, q[0, 2])))
    alpha = math.atan2(-q[0, 1], q[0, 0])
    gamma = math.atan2(-q[1, 2], q[2, 2])
    axes = gs.rotated_main_axes(3, [alpha, beta, gamma])
    np.testing.assert_allclose(axes, q.T, atol=1e-12)
    cls, rescale = {
        "spherical": (gs.Spherical, 1.0),
        "exponential": (gs.Exponential, 3.0),
        "gaussian": (gs.Gaussian, math.sqrt(3.0)),
    }[s.type]
    return cls(
        dim=3,
        var=s.sill,
        len_scale=list(s.ranges),
        angles=[alpha, beta, gamma],
        nugget=model.nugget,
        rescale=rescale,
    )


@pytest.mark.parametrize("name", list(MODELS))
def test_matches_gstools_ordinary_kriging(cloud, name):
    """Second oracle. ``exact=True`` makes gstools reproduce the data at
    conditioning points (nugget as discontinuity, as here); with its default
    ``exact=False`` it differs AT coincident targets only (measured), where
    it treats the nugget as measurement error."""
    samples, values, targets = cloud
    targets = targets.copy()
    targets[:5] = samples[:5]
    model = MODELS[name]
    res = ordinary_kriging(samples, values, targets, model, BIG, all_samples(len(samples)))
    oracle = gs.krige.Ordinary(
        gstools_model(model), (samples[:, 0], samples[:, 1], samples[:, 2]), values, exact=True
    )
    est, var = oracle((targets[:, 0], targets[:, 1], targets[:, 2]), return_var=True)
    np.testing.assert_allclose(res.estimate, est, atol=1e-6, rtol=0)
    np.testing.assert_allclose(res.variance, var, atol=1e-6, rtol=0)


def test_matches_gstools_simple_kriging(cloud):
    samples, values, targets = cloud
    model = MODELS["exp_aniso"]
    res = simple_kriging(
        samples, values, targets, model, BIG, all_samples(len(samples)), mean=1.3
    )
    oracle = gs.krige.Simple(
        gstools_model(model),
        (samples[:, 0], samples[:, 1], samples[:, 2]),
        values,
        mean=1.3,
        exact=True,
    )
    est, var = oracle((targets[:, 0], targets[:, 1], targets[:, 2]), return_var=True)
    np.testing.assert_allclose(res.estimate, est, atol=1e-6, rtol=0)
    np.testing.assert_allclose(res.variance, var, atol=1e-6, rtol=0)
    np.testing.assert_allclose(res.slope, 1.0, atol=1e-8)


# -------------------------------------------------------------- block kriging
def rot_grid():
    c, s = math.cos(math.radians(30.0)), math.sin(math.radians(30.0))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ np.array(
        [[1.0, 0.0, 0.0], [0.0, 0.8, -0.6], [0.0, 0.6, 0.8]]
    )


@pytest.mark.parametrize("method", ["ordinary", "simple"])
@pytest.mark.parametrize("rotated", [False, True])
def test_block_kriging_matches_dense_solver(cloud, method, rotated):
    """Block OK / SK against the numpy solver, with per-target block sizes
    and (optionally) a rotated grid; fixed neighbourhood = all samples."""
    samples, values, targets = cloud
    samples, values, targets = samples[:40], values[:40], targets[:25]
    rng = np.random.default_rng(4)
    sizes = rng.choice([4.0, 10.0, 25.0], (25, 3))
    rot = rot_grid() if rotated else None
    disc = (3, 2, 4)
    mean = 1.1 if method == "simple" else None
    res = krige(
        samples,
        values,
        targets,
        NESTED,
        BIG,
        all_samples(40),
        method=method,
        mean=mean,
        block_size=sizes,
        discretisation=disc,
        grid_rotation=rot,
        return_weights_for=np.arange(25),
        chunk_size=7,
    )
    assert res.support == "block" and res.estimated.all()
    for t in range(25):
        pts = lattice(targets[t], sizes[t], disc, rot)
        idx = res.weight_samples[t]
        ref = dense_krige(samples[idx], values[idx], NESTED, points=pts, mean=mean)
        np.testing.assert_allclose(res.weights[t], ref["weights"], atol=1e-9, rtol=0)
        assert res.estimate[t] == pytest.approx(ref["estimate"], abs=1e-9)
        assert res.variance[t] == pytest.approx(ref["variance"], abs=1e-9)
        assert res.efficiency[t] == pytest.approx(1.0 - ref["variance"] / ref["c00"], abs=1e-9)
        assert block_covariance(NESTED, sizes[t], disc, rot) == pytest.approx(ref["c00"], abs=1e-12)


def test_block_estimate_is_mean_of_point_estimates_same_neighbourhood(cloud):
    """Exact identity (linearity of the weights in the right-hand side) when
    the neighbourhood is the same and no lattice point sits on a sample."""
    samples, values, targets = cloud
    samples, values, targets = samples[:60], values[:60], targets[:20]
    model = MODELS["sph_aniso"]
    size, disc = (12.0, 8.0, 6.0), (4, 3, 2)
    params = all_samples(60)
    block = ordinary_kriging(
        samples, values, targets, model, BIG, params, block_size=size, discretisation=disc
    )
    for t in range(20):
        pts = lattice(targets[t], size, disc)
        point = ordinary_kriging(samples, values, pts, model, BIG, params)
        assert block.estimate[t] == pytest.approx(point.estimate.mean(), abs=1e-10)
    # and close to the mean of point estimates over a much denser lattice
    dense = ordinary_kriging(
        samples, values, lattice(targets[0], size, (12, 12, 12)), model, BIG, params
    )
    assert block.estimate[0] == pytest.approx(dense.estimate.mean(), abs=5e-3)
    # block variance is below the point variance at the centroid
    centre = ordinary_kriging(samples, values, targets, model, BIG, params)
    assert (block.variance < centre.variance).all()


def test_block_variance_decreases_with_block_size_and_discretisation_converges(cloud):
    samples, values, targets = cloud
    targets = targets[:50]
    model = MODELS["sph_iso"]
    params = all_samples(len(samples))
    variances = []
    for edge in (2.0, 5.0, 10.0, 20.0, 40.0):
        res = ordinary_kriging(
            samples, values, targets, model, BIG, params,
            block_size=(edge, edge, edge), discretisation=(5, 5, 5),
        )  # fmt: skip
        assert res.estimated.all() and (res.variance > 0).all()
        variances.append(res.variance)
    for small, large in zip(variances[:-1], variances[1:], strict=True):
        assert (large < small).all()
    # block covariance shrinks with size as well, and stays below C(0) - nugget
    cbb = [block_covariance(model, (e, e, e), (5, 5, 5)) for e in (2.0, 10.0, 40.0)]
    assert 0.8 > cbb[0] > cbb[1] > cbb[2] > 0.0

    out = {}
    for disc in ((2, 2, 2), (4, 4, 4), (8, 8, 8)):
        out[disc] = ordinary_kriging(
            samples, values, targets, model, BIG, params,
            block_size=(10.0, 10.0, 10.0), discretisation=disc,
        )  # fmt: skip
    e2, e4, e8 = (out[d].estimate for d in out)
    v2, v4, v8 = (out[d].variance for d in out)
    assert np.abs(e4 - e8).max() < 5e-3
    assert np.abs(v4 - v8).max() < 5e-3
    assert np.abs(e4 - e8).max() < np.abs(e2 - e8).max()
    assert np.abs(v4 - v8).max() < np.abs(v2 - v8).max()


def test_block_with_single_discretisation_point_is_point_kriging(cloud):
    samples, values, targets = cloud
    model = MODELS["exp_iso"]
    ell, params = Ellipsoid(50.0), SearchParams(10)
    point = ordinary_kriging(samples, values, targets, model, ell, params)
    block = ordinary_kriging(
        samples, values, targets, model, ell, params, block_size=(10.0, 10.0, 5.0)
    )
    assert block.support == "point"
    np.testing.assert_array_equal(block.estimate, point.estimate)
    np.testing.assert_array_equal(block.variance, point.variance)


# ----------------------------------------------------- duplicates, failures
def test_duplicate_samples_are_merged_and_reported(cloud):
    samples, values, targets = cloud
    samples, values, targets = samples[:50], values[:50], targets[:40]
    model = MODELS["sph_iso"]
    rng = np.random.default_rng(2)
    dup_values = values[:6] + rng.normal(0.0, 0.3, 6)
    with_dups = np.vstack([samples, samples[:6], samples[:1]])  # sample 0 three times
    with_dup_values = np.concatenate([values, dup_values, [9.0]])
    params = all_samples(57)
    res = ordinary_kriging(
        with_dups, with_dup_values, targets, model, BIG, params, return_weights_for=[0]
    )
    assert res.estimated.all() and np.isfinite(res.estimate).all()
    assert (res.n_merged == 7).all() and (res.n_samples == 57).all()
    assert res.n_targets_with_duplicates == 40 and res.n_duplicates_merged == 280
    assert res.n_singular == 0

    # equals kriging of the data with the duplicates averaged beforehand
    merged = values.copy()
    merged[0] = (values[0] + dup_values[0] + 9.0) / 3.0
    merged[1:6] = (values[1:6] + dup_values[1:]) / 2.0
    ref = ordinary_kriging(samples, merged, targets, model, BIG, all_samples(50))
    np.testing.assert_allclose(res.estimate, ref.estimate, atol=1e-10, rtol=0)
    np.testing.assert_allclose(res.variance, ref.variance, atol=1e-10, rtol=0)
    # reported weights: per original sample, the group's weight split equally
    w, idx = res.weights[0], res.weight_samples[0]
    assert w.sum() == pytest.approx(1.0, abs=1e-10)
    assert w @ with_dup_values[idx] == pytest.approx(res.estimate[0], abs=1e-10)
    three = w[np.isin(idx, [0, 50, 56])]
    assert three.max() - three.min() < 1e-15

    # tolerance 0 still merges exact coincidences
    raw = ordinary_kriging(
        with_dups, with_dup_values, targets, model, BIG, params, duplicate_tolerance=0.0
    )
    assert (raw.n_merged == 7).all()
    np.testing.assert_allclose(raw.estimate, ref.estimate, atol=1e-10, rtol=0)
    near = with_dups.copy()
    # almost coincident and not merged (tolerance 0), no nugget: the system
    # is ill-conditioned -> detected, counted, nothing written
    near[50:] += 1e-9
    bad = ordinary_kriging(
        near, with_dup_values, targets, VariogramModel(0.0, model.structures), BIG, params,
        duplicate_tolerance=0.0,
    )  # fmt: skip
    assert bad.n_singular == 40 and (bad.status == STATUS_SINGULAR).all()
    assert np.isnan(bad.estimate).all() and np.isnan(bad.variance).all()
    assert not bad.estimated.any()


def test_degenerate_model_is_reported_not_estimated(cloud):
    samples, values, targets = cloud
    zero = VariogramModel(0.0, (Structure("spherical", 0.0, 50.0),))
    res = ordinary_kriging(samples, values, targets[:10], zero, Ellipsoid(60.0))
    assert res.n_singular == 10 and np.isnan(res.estimate).all()
    # pure nugget is a valid model: OK = neighbourhood mean, variance c0 (1 + 1/n)
    nug = VariogramModel(0.5, ())
    res = ordinary_kriging(samples, values, targets[:10], nug, BIG, all_samples(150))
    np.testing.assert_allclose(res.estimate, values.mean(), atol=1e-12)
    np.testing.assert_allclose(res.variance, 0.5 * (1.0 + 1.0 / 150.0), atol=1e-12)


def test_missing_values_search_flags_and_chunking(cloud):
    samples, values, targets = cloud
    values = values.copy()
    values[::7] = np.nan
    model = MODELS["sph_aniso"]
    ell = Ellipsoid(30.0, 1.5, 2.0, angles=(35.0, 20.0, 50.0))
    params = SearchParams(max_samples=8, min_samples=4, sectors="octant", max_per_sector=2)
    holes = np.arange(150) % 20
    kwargs = dict(hole_ids=holes, return_weights_for=np.arange(300))
    res = ordinary_kriging(samples, values, targets, model, ell, params, **kwargs)
    assert res.n_samples_excluded == int(np.isnan(values).sum())
    assert 0 < res.estimated.sum() < 300
    assert (res.n_samples[res.estimated] >= 4).all()
    assert (res.n_samples[~res.estimated] < 4).all()
    assert np.isnan(res.estimate[~res.estimated]).all()
    used = res.weight_samples[res.weight_samples >= 0]
    assert not np.isnan(values[used]).any()  # original indices, never a NaN sample
    small = ordinary_kriging(samples, values, targets, model, ell, params, chunk_size=13, **kwargs)
    for field in ("estimate", "variance", "weights", "weight_samples", "status", "n_samples"):
        np.testing.assert_array_equal(getattr(small, field), getattr(res, field))


def test_validation():
    s, v, t = np.zeros((3, 3)), np.ones(3), np.zeros((1, 3))
    model = MODELS["sph_iso"]
    with pytest.raises(ValueError):
        krige(s, v, t, model, BIG, method="universal")
    with pytest.raises(ValueError):
        krige(s, v, t, model, BIG, method="simple")  # no mean
    with pytest.raises(ValueError):
        krige(s, v, t, model, BIG, mean=1.0)  # mean with ordinary
    with pytest.raises(TypeError):
        krige(s, v, t, {"nugget": 0}, BIG)
    with pytest.raises(ValueError):
        krige(s, v, t, model, BIG, return_weights_for=[5])
    with pytest.raises(ValueError):
        krige(s, v, t, model, BIG, block_size=(1.0, -1.0, 1.0), discretisation=(2, 2, 2))
    empty = krige(np.zeros((0, 3)), np.zeros(0), t, model, BIG)
    assert np.isnan(empty.estimate).all() and empty.n_not_selected == 1
    assert krige(s, v, np.zeros((0, 3)), model, BIG).estimate.shape == (0,)


# ---------------------------------------------------- negative-weight option
def test_negative_weight_correction():
    """Screen effect: samples lined up behind a near one get negative
    weights with a continuous (gaussian-like / no nugget) model."""
    model = VariogramModel(0.0, (Structure("spherical", 1.0, 100.0),))
    samples = np.array(
        [[5.0, 0.0, 0.0], [12.0, 0.0, 0.0], [30.0, 1.0, 0.0], [-40.0, 0.0, 0.0], [0.0, 45.0, 0.0]]
    )
    values = np.array([3.0, 1.0, 8.0, 2.0, 4.0])
    target = np.zeros((1, 3))
    params = all_samples(5)
    plain = ordinary_kriging(samples, values, target, model, BIG, params, return_weights_for=[0])
    assert plain.sum_negative_weights[0] < -1e-3 and (plain.weights[0] < 0).any()

    fixed = ordinary_kriging(
        samples, values, target, model, BIG, params,
        return_weights_for=[0], negative_weight_correction=True,
    )  # fmt: skip
    w0, w1 = plain.weights[0], fixed.weights[0]
    np.testing.assert_array_equal(fixed.weight_samples, plain.weight_samples)
    expected = np.where(w0 > 0, w0, 0.0)
    expected /= expected.sum()
    np.testing.assert_allclose(w1, expected, atol=1e-12)
    assert (w1 >= 0).all() and w1.sum() == pytest.approx(1.0, abs=1e-12)
    idx = fixed.weight_samples[0]
    assert fixed.estimate[0] == pytest.approx(w1 @ values[idx], abs=1e-12)
    assert values.min() <= fixed.estimate[0] <= values.max()
    # the reported negative-weight sum is that of the solved system
    assert fixed.sum_negative_weights[0] == pytest.approx(plain.sum_negative_weights[0])
    # variance of the corrected (no longer optimal) estimator, general formula
    ref = dense_krige(samples[idx], values[idx], model, target=target[0])
    general = ref["c00"] - 2.0 * w1 @ ref["rhs"] + w1 @ ref["cmat"] @ w1
    assert fixed.variance[0] == pytest.approx(general, abs=1e-12)
    assert fixed.variance[0] > plain.variance[0]
    # no negatives -> the option changes nothing
    a = ordinary_kriging(samples[[0, 3, 4]], values[[0, 3, 4]], target, model, BIG, params)
    b = ordinary_kriging(
        samples[[0, 3, 4]], values[[0, 3, 4]], target, model, BIG, params,
        negative_weight_correction=True,
    )  # fmt: skip
    assert a.sum_negative_weights[0] == 0.0
    assert a.estimate[0] == b.estimate[0] and a.variance[0] == b.variance[0]
    # simple kriging: the sum of weights is preserved
    sk0 = simple_kriging(
        samples, values, target, model, BIG, params, mean=3.0, return_weights_for=[0]
    )
    sk1 = simple_kriging(
        samples, values, target, model, BIG, params, mean=3.0,
        return_weights_for=[0], negative_weight_correction=True,
    )  # fmt: skip
    assert (sk0.weights[0] < 0).any() and (sk1.weights[0] >= 0).all()
    assert sk1.weights[0].sum() == pytest.approx(sk0.sum_weights[0], abs=1e-12)
    assert sk1.sum_weights[0] == pytest.approx(sk0.sum_weights[0], abs=1e-12)


# ---------------------------------------------------------------- performance
def test_performance_reduced():
    """plan.md target: 1M targets, max 24 samples, point OK under 5 minutes.
    Asserted here at 1/20 of the targets with a generous bound; the measured
    full-size times (point and 4x4x2 block) are reported with the task."""
    rng = np.random.default_rng(0)
    samples = rng.uniform(0.0, 1000.0, (20_000, 3))
    values = rng.lognormal(0.0, 1.0, 20_000)
    targets = rng.uniform(0.0, 1000.0, (50_000, 3))
    ell = Ellipsoid(150.0, 2.0, 4.0, angles=(30.0, 20.0, 10.0))
    model = VariogramModel(
        0.15, (Structure("spherical", 0.85, (120.0, 70.0, 30.0), (30.0, 20.0, 10.0)),)
    )
    params = SearchParams(max_samples=24)
    ordinary_kriging(samples, values, targets[:100], model, ell, params)  # compile
    start = time.perf_counter()
    res = ordinary_kriging(samples, values, targets, model, ell, params)
    elapsed = time.perf_counter() - start
    assert res.estimated.mean() > 0.9 and res.n_singular == 0
    assert elapsed < 30.0, f"50k targets took {elapsed:.1f} s"
