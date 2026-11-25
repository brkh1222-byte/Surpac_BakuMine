"""2-D interpolation with exact values and bounds (geology_contracts.md 5).

Independent checks (none of them shares code with ``surface_interp.py``):

* PyKrige ``OrdinaryKriging`` / ``UniversalKriging`` with every sample in the
  system, and the project's own ``krige`` with a search that holds every
  sample (the usual weights form; the module solves the dual form);
* ``scipy.interpolate.RBFInterpolator`` for the thin-plate spline;
* ``scipy.optimize.minimize(method="SLSQP")`` on the quadratic programme
  written in all coefficients (inactive bound points included), with the
  covariance function ``ref_cov`` of this file;
* the KKT conditions of that programme, which are sufficient for a convex QP;
* a two-point case solved by hand.

All data are made-up numbers.
"""

import json
import time

import numpy as np
import pytest
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging
from scipy.interpolate import RBFInterpolator
from scipy.optimize import minimize

from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel
from bakumine.core.geometry.surface_interp import (
    Surface2D,
    SurfaceFitError,
    default_variogram,
    fit_surface,
)

SPH = VariogramModel(0.0, (Structure("spherical", 4.0, 60.0),))
SPH_NUGGET = VariogramModel(0.5, (Structure("spherical", 3.5, 45.0),))
EXP = VariogramModel(0.0, (Structure("exponential", 2.0, 80.0),))
ISOTROPIC = {"spherical": SPH, "spherical_nugget": SPH_NUGGET, "exponential": EXP}


# ----------------------------------------------------------------- reference
def ref_gamma(model, d):
    """Variogram of a nugget + one isotropic structure model at distance d
    (research.md 7.4), written out again here."""
    (s,) = model.structures
    r = np.asarray(d, dtype=float) / s.ranges[0]
    if s.type == "spherical":
        g = np.where(r < 1.0, 1.5 * r - 0.5 * r**3, 1.0)
    elif s.type == "exponential":
        g = 1.0 - np.exp(-3.0 * r)
    else:
        raise AssertionError(s.type)
    return np.where(r > 0.0, model.nugget + s.sill * g, 0.0)


def ref_cov(model, a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    return model.nugget + model.structures[0].sill - ref_gamma(model, d)


def ref_tps(a, b):
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=2)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d > 0.0, d * d * np.log(d), 0.0)


def monomials(xy, drift):
    if drift == "constant":
        return np.ones((len(xy), 1))
    return np.column_stack([np.ones(len(xy)), xy])


def qp_oracle(kernel, drift, xy, z, lxy, lo, uxy, up, targets):
    """SLSQP solve of: minimise (1/2) b^T K b over (b, c) with f = z at the
    exact points, f >= lo, f <= up and P^T b = 0. Returns f at the targets and
    at the bound points."""
    pts = np.vstack([xy, lxy, uxy])
    ne, nl, n = len(xy), len(lxy), len(xy) + len(lxy) + len(uxy)
    k = kernel(pts, pts)
    p = monomials(pts, drift)
    q = p.shape[1]
    a = np.hstack([k, p])  # f at the points = a @ [b, c]
    grad_pad = np.zeros(q)
    cons = [
        {"type": "eq", "fun": lambda w: a[:ne] @ w - z, "jac": lambda w: a[:ne]},
        {
            "type": "eq",
            "fun": lambda w: p.T @ w[:n],
            "jac": lambda w: np.hstack([p.T, np.zeros((q, q))]),
        },
    ]
    if nl:
        cons.append(
            {
                "type": "ineq",
                "fun": lambda w: a[ne : ne + nl] @ w - lo,
                "jac": lambda w: a[ne : ne + nl],
            }
        )
    if n > ne + nl:
        cons.append(
            {"type": "ineq", "fun": lambda w: up - a[ne + nl :] @ w, "jac": lambda w: -a[ne + nl :]}
        )
    res = minimize(
        lambda w: 0.5 * w[:n] @ k @ w[:n],
        np.zeros(n + q),
        jac=lambda w: np.concatenate([k @ w[:n], grad_pad]),
        constraints=cons,
        method="SLSQP",
        options={"ftol": 1e-12, "maxiter": 2000},
    )
    assert res.success, res.message
    w = res.x
    at_targets = kernel(targets, pts) @ w[:n] + monomials(targets, drift) @ w[n:]
    return at_targets, a[ne:] @ w


def trend(xy):
    return (
        20.0 + 0.05 * xy[:, 0] - 0.03 * xy[:, 1] + 3.0 * np.sin(xy[:, 0] / 17.0 + xy[:, 1] / 29.0)
    )


def bounded_case(seed, n=None):
    """Random exact points, lower and upper bounds (some active, some not) and
    targets in a 100 m square."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(10, 41)) if n is None else n
    ne = max(4, n // 2)
    nl = (n - ne + 1) // 2
    nu = n - ne - nl
    pts = rng.uniform(0.0, 100.0, size=(n, 2))
    xy, lxy, uxy = pts[:ne], pts[ne : ne + nl], pts[ne + nl :]
    z = trend(xy) + rng.normal(0.0, 1.0, ne)
    lo = trend(lxy) + rng.normal(0.0, 2.0, nl)
    up = trend(uxy) + rng.normal(0.0, 2.0, nu)
    targets = rng.uniform(-10.0, 110.0, size=(60, 2))
    return xy, z, lxy, lo, uxy, up, targets


def exact_case(seed=11, n=40, offset=(0.0, 0.0)):
    rng = np.random.default_rng(seed)
    xy = rng.uniform(0.0, 100.0, size=(n, 2)) + np.asarray(offset)
    z = trend(xy - np.asarray(offset)) + rng.normal(0.0, 1.0, n)
    targets = rng.uniform(-10.0, 110.0, size=(200, 2)) + np.asarray(offset)
    targets[:5] = xy[:5]
    return xy, z, targets


def pykrige_args(model):
    (s,) = model.structures
    return dict(
        variogram_model=s.type,
        variogram_parameters={"psill": s.sill, "range": s.ranges[0], "nugget": model.nugget},
    )


# ------------------------------------------------- exact data: kriging oracles
@pytest.mark.parametrize("name", list(ISOTROPIC))
def test_exact_only_matches_pykrige_ordinary_kriging(name):
    model = ISOTROPIC[name]
    xy, z, targets = exact_case()
    surface = fit_surface(xy, z, variogram=model)
    oracle = OrdinaryKriging(xy[:, 0], xy[:, 1], z, **pykrige_args(model))
    est, _ = oracle.execute("points", targets[:, 0], targets[:, 1])
    got = surface(targets)
    np.testing.assert_allclose(got, np.asarray(est), atol=1e-6, rtol=0)
    assert np.ptp(got) > 3.0  # not a flat comparison
    assert surface.iterations == 1 and surface.max_violation == 0.0
    assert surface.active_lower.size == 0 and surface.active_upper.size == 0


def test_exact_only_matches_pykrige_with_anisotropy():
    """Major axis at bearing 30 (clockwise from north) = 60 degrees
    anticlockwise from east, the angle PyKrige rotates by; ratio 2.5."""
    model = VariogramModel(
        0.2, (Structure("spherical", 3.0, (75.0, 30.0, 30.0), (30.0, 0.0, 0.0)),)
    )
    xy, z, targets = exact_case(seed=5)
    surface = fit_surface(xy, z, variogram=model)
    oracle = OrdinaryKriging(
        xy[:, 0],
        xy[:, 1],
        z,
        variogram_model="spherical",
        variogram_parameters={"psill": 3.0, "range": 75.0, "nugget": 0.2},
        anisotropy_scaling=2.5,
        anisotropy_angle=60.0,
    )
    est, _ = oracle.execute("points", targets[:, 0], targets[:, 1])
    np.testing.assert_allclose(surface(targets), np.asarray(est), atol=1e-6, rtol=0)
    # the anisotropy matters: the isotropic model gives another surface
    iso = fit_surface(xy, z, variogram=VariogramModel(0.2, (Structure("spherical", 3.0, 75.0),)))
    assert np.abs(iso(targets) - surface(targets)).max() > 0.1


@pytest.mark.parametrize(
    "model",
    [
        SPH,
        SPH_NUGGET,
        EXP,
        VariogramModel(0.1, (Structure("spherical", 2.0, (90.0, 40.0, 20.0), (25.0, 0.0, 0.0)),)),
    ],
)
def test_exact_only_matches_own_krige_with_all_samples(model):
    xy, z, targets = exact_case(seed=3, n=35)
    surface = fit_surface(xy, z, variogram=model)
    samples = np.column_stack([xy, np.zeros(len(xy))])
    tg = np.column_stack([targets, np.zeros(len(targets))])
    res = krige(
        samples, z, tg, model, Ellipsoid(1.0e6), SearchParams(max_samples=len(xy), radius=1.0e6)
    )
    assert res.estimated.all() and (res.n_samples == len(xy)).all()
    np.testing.assert_allclose(surface(targets), res.estimate, atol=1e-9, rtol=0)


def test_linear_drift_matches_pykrige_universal_kriging():
    xy, z, targets = exact_case(seed=8)
    surface = fit_surface(xy, z, variogram=SPH_NUGGET, drift="linear")
    oracle = UniversalKriging(
        xy[:, 0], xy[:, 1], z, drift_terms=["regional_linear"], **pykrige_args(SPH_NUGGET)
    )
    est, _ = oracle.execute("points", targets[:, 0], targets[:, 1])
    np.testing.assert_allclose(surface(targets), np.asarray(est), atol=1e-6, rtol=0)
    constant = fit_surface(xy, z, variogram=SPH_NUGGET)
    assert np.abs(constant(targets) - surface(targets)).max() > 1e-3


def test_rbf_matches_scipy_thin_plate_spline():
    xy, z, targets = exact_case(seed=21)
    surface = fit_surface(xy, z, method="rbf", drift="constant")
    assert surface.drift == "linear" and surface.variogram is None
    oracle = RBFInterpolator(xy, z, kernel="thin_plate_spline", degree=1)
    np.testing.assert_allclose(surface(targets), oracle(targets), atol=1e-7, rtol=0)


# ------------------------------------------------------- exactness, constants
@pytest.mark.parametrize(
    "method,drift", [("kriging", "constant"), ("kriging", "linear"), ("rbf", "linear")]
)
@pytest.mark.parametrize("offset", [(0.0, 0.0), (512300.0, 4871200.0)])
def test_exact_at_data_points(method, drift, offset):
    xy, z, _ = exact_case(seed=2, offset=offset)
    surface = fit_surface(xy, z, method=method, drift=drift, variogram=SPH_NUGGET)
    np.testing.assert_allclose(surface(xy), z, atol=1e-9, rtol=0)
    assert surface.points.shape == (len(xy), 2) and surface.coefficients.shape == (len(xy),)


@pytest.mark.parametrize(
    "method,drift", [("kriging", "constant"), ("kriging", "linear"), ("rbf", "linear")]
)
def test_constant_field_is_reproduced(method, drift):
    xy, _, targets = exact_case(seed=4)
    surface = fit_surface(xy, np.full(len(xy), 37.5), method=method, drift=drift)
    np.testing.assert_allclose(surface(targets * 5.0), 37.5, atol=1e-9, rtol=0)
    np.testing.assert_allclose(surface.coefficients, 0.0, atol=1e-9)


@pytest.mark.parametrize("method", ["kriging", "rbf"])
@pytest.mark.parametrize("offset", [(0.0, 0.0), (512300.0, 4871200.0)])
def test_plane_is_reproduced_with_a_linear_drift(method, offset):
    xy, _, targets = exact_case(seed=6, offset=offset)

    def plane(p):
        return 12.0 + 0.07 * (p[:, 0] - offset[0]) - 0.11 * (p[:, 1] - offset[1])

    surface = fit_surface(xy, plane(xy), method=method, drift="linear", variogram=SPH)
    far = targets + 300.0  # well outside the data: the drift carries the plane
    np.testing.assert_allclose(surface(targets), plane(targets), atol=1e-9, rtol=0)
    np.testing.assert_allclose(surface(far), plane(far), atol=1e-8, rtol=0)
    # with a constant drift the plane is NOT reproduced away from the data
    flat = fit_surface(xy, plane(xy), variogram=SPH)
    assert np.abs(flat(far) - plane(far)).max() > 1.0


# ------------------------------------------------------------------- bounds
CASES = [("kriging", "constant"), ("kriging", "linear"), ("rbf", "linear")]


def kernel_of(method, model):
    if method == "rbf":
        return ref_tps
    return lambda a, b: ref_cov(model, a, b)


@pytest.mark.parametrize("method,drift", CASES)
@pytest.mark.parametrize("seed", range(8))
def test_bounds_match_slsqp_quadratic_programme(method, drift, seed):
    xy, z, lxy, lo, uxy, up, targets = bounded_case(seed)
    if method == "rbf":
        # the same data in a 1 x 1 square: with r^2 log r on distances of tens
        # of metres SLSQP stops at about 1e-5 of the solution (badly scaled
        # programme); test_rbf_bounds_do_not_depend_on_the_length_unit covers
        # the 100 m square
        xy, lxy, uxy, targets = xy / 100.0, lxy / 100.0, uxy / 100.0, targets / 100.0
    surface = fit_surface(
        xy,
        z,
        lower_xy=lxy,
        lower=lo,
        upper_xy=uxy,
        upper=up,
        method=method,
        drift=drift,
        variogram=SPH,
    )
    want, want_bounds = qp_oracle(kernel_of(method, SPH), drift, xy, z, lxy, lo, uxy, up, targets)
    np.testing.assert_allclose(surface(targets), want, atol=1e-6, rtol=0)
    np.testing.assert_allclose(surface(np.vstack([lxy, uxy])), want_bounds, atol=1e-6, rtol=0)
    assert surface.active_lower.size + surface.active_upper.size > 0  # the bounds matter
    assert surface.iterations > 1


@pytest.mark.parametrize("method,drift", CASES)
@pytest.mark.parametrize("seed", range(20, 32))
def test_bounds_satisfy_the_kkt_conditions(method, drift, seed):
    """Feasible, exact at the data, active bounds met with equality and with
    multipliers of the right sign, no coefficient on inactive bounds: for a
    convex programme these conditions are sufficient for the minimum."""
    xy, z, lxy, lo, uxy, up, _ = bounded_case(seed)
    s = fit_surface(
        xy,
        z,
        lower_xy=lxy,
        lower=lo,
        upper_xy=uxy,
        upper=up,
        method=method,
        drift=drift,
        variogram=EXP,
    )
    ne, nl_act = len(xy), s.active_lower.size
    np.testing.assert_allclose(s(xy), z, atol=1e-9, rtol=0)
    f_lo, f_up = s(lxy), s(uxy)
    assert (f_lo >= lo - 1e-9).all() and (f_up <= up + 1e-9).all()
    assert s.max_violation <= 1e-9
    np.testing.assert_allclose(f_lo[s.active_lower], lo[s.active_lower], atol=1e-9, rtol=0)
    np.testing.assert_allclose(f_up[s.active_upper], up[s.active_upper], atol=1e-9, rtol=0)
    # points = exact, then active bounds (in input order: lower before upper)
    assert s.points.shape[0] == ne + nl_act + s.active_upper.size
    np.testing.assert_array_equal(s.points[:ne], xy)
    np.testing.assert_array_equal(s.points[ne : ne + nl_act], lxy[s.active_lower])
    np.testing.assert_array_equal(s.points[ne + nl_act :], uxy[s.active_upper])
    b = s.coefficients[ne:]
    assert (b[:nl_act] >= 0.0).all() and (b[nl_act:] <= 0.0).all()
    # P^T b = 0 (the monomials of the module's own frame)
    u = (s.points - s.origin) / s.scale
    p = np.ones((len(u), 1)) if s.drift == "constant" else np.column_stack([np.ones(len(u)), u])
    np.testing.assert_allclose(p.T @ s.coefficients, 0.0, atol=1e-9 * np.abs(s.coefficients).sum())
    # strict inactivity elsewhere: the remaining bounds are slack
    slack_lo = np.delete(f_lo - lo, s.active_lower)
    slack_up = np.delete(up - f_up, s.active_upper)
    assert (slack_lo > -1e-9).all() and (slack_up > -1e-9).all()


def test_rbf_bounds_do_not_depend_on_the_length_unit():
    """A change of unit adds only a linear polynomial to r^2 log r sums with
    P^T b = 0, so the thin-plate surface is the same in any length unit."""
    xy, z, lxy, lo, uxy, up, targets = bounded_case(2)
    kw = dict(lower=lo, upper=up, method="rbf")
    a = fit_surface(xy, z, lower_xy=lxy, upper_xy=uxy, **kw)
    b = fit_surface(xy / 100.0, z, lower_xy=lxy / 100.0, upper_xy=uxy / 100.0, **kw)
    np.testing.assert_allclose(a(targets), b(targets / 100.0), atol=1e-9, rtol=0)
    np.testing.assert_array_equal(a.active_lower, b.active_lower)
    np.testing.assert_array_equal(a.active_upper, b.active_upper)
    assert a.active_lower.size + a.active_upper.size > 0


@pytest.mark.parametrize("method,drift", CASES)
def test_inactive_bounds_change_nothing(method, drift):
    xy, z, lxy, _, uxy, _, targets = bounded_case(3, n=36)
    free = fit_surface(xy, z, method=method, drift=drift, variogram=SPH)
    lo = free(lxy) - 0.5  # satisfied by the exact-only surface
    up = free(uxy) + 0.5
    s = fit_surface(
        xy,
        z,
        lower_xy=lxy,
        lower=lo,
        upper_xy=uxy,
        upper=up,
        method=method,
        drift=drift,
        variogram=SPH,
    )
    np.testing.assert_array_equal(s(targets), free(targets))
    np.testing.assert_array_equal(s.coefficients, free.coefficients)
    assert s.iterations == 1 and s.max_violation == 0.0
    assert s.active_lower.size == 0 and s.active_upper.size == 0
    assert s.points.shape == (len(xy), 2)


def test_an_active_bound_is_met_with_equality_and_pulls_the_surface():
    xy, z, lxy, _, uxy, _, targets = bounded_case(4, n=30)
    free = fit_surface(xy, z, variogram=SPH)
    lo = free(lxy) - 0.5
    lo[0] = free(lxy[:1])[0] + 2.0  # the only violated lower bound
    up = free(uxy) + 0.5
    up[1] = free(uxy[1:2])[0] - 1.5  # the only violated upper bound
    s = fit_surface(xy, z, lower_xy=lxy, lower=lo, upper_xy=uxy, upper=up, variogram=SPH)
    assert 0 in s.active_lower and 1 in s.active_upper
    assert s(lxy[:1])[0] == pytest.approx(lo[0], abs=1e-9)
    assert s(uxy[1:2])[0] == pytest.approx(up[1], abs=1e-9)
    assert (s(lxy) >= lo - 1e-9).all() and (s(uxy) <= up + 1e-9).all()
    assert np.abs(s(targets) - free(targets)).max() > 0.1


def test_two_point_case_solved_by_hand():
    """One exact value z0 at 0 and one lower bound l > z0 at distance h,
    ordinary kriging. Exact-only: f = z0 everywhere, so the bound is active.
    Two equalities: b1 = -b2 = (z0 - l) / (2 gamma(h)), c = (z0 + l) / 2 and

        f(x) = (z0 + l)/2 + (l - z0) (gamma(x - x0) - gamma(x - x1)) / (2 gamma(h))
    """
    a, sill, h, z0, low = 50.0, 2.0, 20.0, 10.0, 13.0
    model = VariogramModel(0.0, (Structure("spherical", sill, a),))

    def gamma(d):
        r = min(abs(d) / a, 1.0)
        return sill * (1.5 * r - 0.5 * r**3)

    s = fit_surface([[0.0, 0.0]], [z0], lower_xy=[[h, 0.0]], lower=[low], variogram=model)
    np.testing.assert_array_equal(s.active_lower, [0])
    assert s.iterations == 2
    np.testing.assert_allclose(
        s.coefficients, np.array([1.0, -1.0]) * (z0 - low) / (2 * gamma(h)), atol=1e-12
    )
    np.testing.assert_allclose(s.drift_coefficients, [(z0 + low) / 2], atol=1e-12)
    xs = np.array([-200.0, -30.0, 0.0, 5.0, 10.0, 20.0, 35.0, 300.0])
    want = [(z0 + low) / 2 + (low - z0) * (gamma(x) - gamma(x - h)) / (2 * gamma(h)) for x in xs]
    np.testing.assert_allclose(s(np.column_stack([xs, np.zeros(len(xs))])), want, atol=1e-12)
    # spot values: the data, the middle, and far away (the mean of the two)
    np.testing.assert_allclose(
        s([[0.0, 0.0], [h, 0.0], [h / 2, 0.0], [0.0, 900.0]]), [10.0, 13.0, 11.5, 11.5], atol=1e-12
    )
    # a bound below z0 is inactive and the surface stays flat
    flat = fit_surface([[0.0, 0.0]], [z0], lower_xy=[[h, 0.0]], lower=[9.0], variogram=model)
    np.testing.assert_allclose(flat([[7.0, 3.0], [h, 0.0]]), z0, atol=1e-12)
    assert flat.active_lower.size == 0


def test_between_pick_is_one_point_with_two_bounds():
    xy, z, lxy, _, _, _, targets = bounded_case(6, n=24)
    free = fit_surface(xy, z, variogram=SPH)
    base = free(lxy)
    lo = base - 1.0
    up = base + 1.0
    lo[0], up[0] = base[0] + 0.5, base[0] + 1.5  # active from below
    lo[1], up[1] = base[1] - 1.5, base[1] - 0.5  # active from above
    s = fit_surface(xy, z, lower_xy=lxy, lower=lo, upper_xy=lxy, upper=up, variogram=SPH)
    f = s(lxy)
    assert (f >= lo - 1e-9).all() and (f <= up + 1e-9).all()
    assert 0 in s.active_lower and 1 in s.active_upper
    want, _ = qp_oracle(kernel_of("kriging", SPH), "constant", xy, z, lxy, lo, lxy, up, targets)
    np.testing.assert_allclose(s(targets), want, atol=1e-6, rtol=0)


# ---------------------------------------------------------------- degenerate
def test_coincident_exact_points_are_averaged():
    xy, z, targets = exact_case(seed=9, n=20)
    xy2 = np.vstack([xy, xy[:3], xy[:1]])
    z2 = np.concatenate([z, z[:3] + 2.0, [z[0] + 4.0]])
    merged = z.copy()
    merged[1:3] += 1.0
    merged[0] += 2.0
    s = fit_surface(xy2, z2, variogram=SPH)
    want = fit_surface(xy, merged, variogram=SPH)
    assert s.points.shape == (20, 2)
    np.testing.assert_allclose(s(targets), want(targets), atol=1e-10, rtol=0)
    np.testing.assert_allclose(s(xy[:3]), merged[:3], atol=1e-9, rtol=0)


def test_contradictory_bounds_at_one_place_raise():
    xy, z, _ = exact_case(seed=9, n=12)
    spot = [[200.0, 200.0]]
    with pytest.raises(SurfaceFitError, match="inconsistent"):
        fit_surface(xy, z, lower_xy=spot, lower=[5.0], upper_xy=spot, upper=[4.0], variogram=SPH)
    # the tightest of several bounds at one place counts
    two = [[200.0, 200.0], [200.0, 200.0]]
    with pytest.raises(SurfaceFitError, match="inconsistent"):
        fit_surface(
            xy, z, lower_xy=two, lower=[1.0, 5.0], upper_xy=two, upper=[4.0, 9.0], variogram=SPH
        )


def test_equal_lower_and_upper_bound_pin_the_value():
    xy, z, targets = exact_case(seed=9, n=12)
    spot = np.array([[40.0, 60.0]])
    s = fit_surface(xy, z, lower_xy=spot, lower=[30.0], upper_xy=spot, upper=[30.0], variogram=SPH)
    want = fit_surface(np.vstack([xy, spot]), np.append(z, 30.0), variogram=SPH)
    np.testing.assert_allclose(s(targets), want(targets), atol=1e-10, rtol=0)
    np.testing.assert_array_equal(s.active_lower, [0])
    np.testing.assert_array_equal(s.active_upper, [0])


def test_several_bounds_at_one_place_use_the_tightest():
    xy, z, targets = exact_case(seed=9, n=12)
    spot = np.array([[40.0, 60.0]])
    top = fit_surface(xy, z, variogram=SPH)(spot)[0]
    three = np.repeat(spot, 3, axis=0)
    s = fit_surface(xy, z, lower_xy=three, lower=[top + 1.0, top + 3.0, top + 2.0], variogram=SPH)
    one = fit_surface(xy, z, lower_xy=spot, lower=[top + 3.0], variogram=SPH)
    np.testing.assert_array_equal(s.active_lower, [1])
    np.testing.assert_allclose(s(targets), one(targets), atol=1e-12, rtol=0)
    assert s(spot)[0] == pytest.approx(top + 3.0, abs=1e-9)


def test_bound_at_an_exact_point_is_dropped_or_rejected():
    xy, z, targets = exact_case(seed=9, n=12)
    free = fit_surface(xy, z, variogram=SPH)
    ok = fit_surface(
        xy, z, lower_xy=xy[:2], lower=z[:2] - 1.0, upper_xy=xy[2:3], upper=z[2:3], variogram=SPH
    )
    np.testing.assert_array_equal(ok(targets), free(targets))
    assert ok.active_lower.size == 0 and ok.active_upper.size == 0 and ok.max_violation == 0.0
    with pytest.raises(SurfaceFitError, match="inconsistent"):
        fit_surface(xy, z, lower_xy=xy[:1], lower=z[:1] + 0.1, variogram=SPH)
    with pytest.raises(SurfaceFitError, match="inconsistent"):
        fit_surface(xy, z, upper_xy=xy[:1], upper=z[:1] - 0.1, variogram=SPH)


def test_too_few_exact_points_raise():
    with pytest.raises(SurfaceFitError, match="at least one exact"):
        fit_surface(np.zeros((0, 2)), [], lower_xy=[[0.0, 0.0]], lower=[1.0])
    line = np.column_stack([np.arange(5.0) * 10.0, np.arange(5.0) * 20.0])
    for method, drift in [("kriging", "linear"), ("rbf", "constant")]:
        with pytest.raises(SurfaceFitError, match="not on one line"):
            fit_surface(line, np.arange(5.0), method=method, drift=drift)
        with pytest.raises(SurfaceFitError, match="not on one line"):
            fit_surface(line[:2], [1.0, 2.0], method=method, drift=drift)
    # coincident points do not count twice
    with pytest.raises(SurfaceFitError, match="not on one line"):
        fit_surface(
            [[0.0, 0.0], [0.0, 0.0], [5.0, 0.0], [0.0, 5.0 * 0.0]],
            [1.0, 1.0, 2.0, 3.0],
            drift="linear",
        )
    # a single exact point is enough for a constant drift
    one = fit_surface([[3.0, 4.0]], [7.0])
    np.testing.assert_allclose(one([[3.0, 4.0], [500.0, -80.0]]), 7.0, atol=1e-12)


def test_bad_arguments_and_non_finite_data():
    xy, z, _ = exact_case(seed=9, n=12)
    with pytest.raises(ValueError, match="method"):
        fit_surface(xy, z, method="idw")
    with pytest.raises(ValueError, match="drift"):
        fit_surface(xy, z, drift="quadratic")
    with pytest.raises(ValueError, match="together"):
        fit_surface(xy, z, lower_xy=xy[:1])
    with pytest.raises(ValueError, match="same length"):
        fit_surface(xy, z[:-1])
    bad = z.copy()
    bad[3] = np.nan
    with pytest.raises(SurfaceFitError, match="finite"):
        fit_surface(xy, bad)
    with pytest.raises(SurfaceFitError, match="finite"):
        fit_surface(xy, z, upper_xy=[[1.0, 2.0]], upper=[np.inf])
    with pytest.raises(SurfaceFitError, match="singular"):
        fit_surface(xy, z, variogram=VariogramModel(0.0, (Structure("spherical", 0.0, 50.0),)))


def test_iteration_cap_raises():
    xy, z, lxy, lo, uxy, up, _ = bounded_case(0)
    kw = dict(lower_xy=lxy, lower=lo, upper_xy=uxy, upper=up, variogram=SPH)
    done = fit_surface(xy, z, **kw)
    assert 1 < done.iterations <= 10 * (len(lo) + len(up)) + 10
    with pytest.raises(SurfaceFitError, match="did not finish"):
        fit_surface(xy, z, max_iterations=done.iterations - 1, **kw)
    again = fit_surface(xy, z, max_iterations=done.iterations, **kw)
    np.testing.assert_array_equal(again.coefficients, done.coefficients)


# ------------------------------------------------- variogram, grid, round trip
def test_default_variogram():
    # 4 x 3 lattice with 10 m spacing: every nearest-neighbour distance is 10
    gx, gy = np.meshgrid(np.arange(4) * 10.0, np.arange(3) * 10.0, indexing="ij")
    xy = np.column_stack([gx.ravel(), gy.ravel()])
    values = np.arange(12.0)
    model = default_variogram(xy, values)
    (s,) = model.structures
    assert model.nugget == 0.0 and s.type == "spherical"
    assert s.sill == pytest.approx(np.var(values))
    assert s.ranges == pytest.approx((30.0, 30.0, 30.0))
    assert default_variogram(xy, np.full(12, 3.0)).structures[0].sill == 1.0
    assert default_variogram([[1.0, 2.0]], [5.0]).structures[0].ranges == (1.0, 1.0, 1.0)
    # fit_surface uses it when no model is given
    z = values + 0.3 * xy[:, 1]
    np.testing.assert_array_equal(
        fit_surface(xy, z)([[7.0, 8.0]]),
        fit_surface(xy, z, variogram=default_variogram(xy, z))([[7.0, 8.0]]),
    )
    assert fit_surface(xy, z).variogram == default_variogram(xy, z)


def test_grid_layout():
    xy, z, _ = exact_case(seed=13, n=25)
    s = fit_surface(xy, z, variogram=SPH)
    g = s.grid((10.0, 20.0), (5.0, 2.5), (7, 4))
    assert g.shape == (7, 4)
    assert g[3, 2] == pytest.approx(s([[10.0 + 3 * 5.0, 20.0 + 2 * 2.5]])[0], abs=1e-12)
    assert g[6, 0] == pytest.approx(s([[40.0, 20.0]])[0], abs=1e-12)
    assert s.grid((0.0, 0.0), (1.0, 1.0), (0, 3)).shape == (0, 3)


@pytest.mark.parametrize("method,drift", CASES)
def test_dict_round_trip(method, drift):
    xy, z, lxy, lo, uxy, up, targets = bounded_case(1)
    s = fit_surface(
        xy,
        z,
        lower_xy=lxy,
        lower=lo,
        upper_xy=uxy,
        upper=up,
        method=method,
        drift=drift,
        variogram=SPH_NUGGET,
    )
    back = Surface2D.from_dict(json.loads(json.dumps(s.to_dict())))
    np.testing.assert_array_equal(back(targets), s(targets))
    assert back.method == s.method and back.drift == s.drift and back.variogram == s.variogram
    np.testing.assert_array_equal(back.active_lower, s.active_lower)
    np.testing.assert_array_equal(back.active_upper, s.active_upper)
    assert back.iterations == s.iterations and back.max_violation == s.max_violation
    with pytest.raises(ValueError, match="schema"):
        Surface2D.from_dict({**s.to_dict(), "schema": "other/1"})


# ----------------------------------------------------------------------- speed
def test_speed_500_points_and_a_200_by_200_grid():
    rng = np.random.default_rng(42)
    pts = rng.uniform(0.0, 2000.0, size=(500, 2))
    xy, lxy, uxy = pts[:300], pts[300:400], pts[400:]

    def field(p):
        return 30.0 + 8.0 * np.sin(p[:, 0] / 300.0) * np.cos(p[:, 1] / 250.0)

    z = field(xy)
    lo = field(lxy) - rng.uniform(-2.0, 6.0, 100)
    up = field(uxy) + rng.uniform(-2.0, 6.0, 100)
    start = time.perf_counter()
    s = fit_surface(xy, z, lower_xy=lxy, lower=lo, upper_xy=uxy, upper=up)
    fitted = time.perf_counter()
    g = s.grid((0.0, 0.0), (10.0, 10.0), (200, 200))
    done = time.perf_counter()
    print(
        f"\nfit {fitted - start:.2f} s ({s.iterations} solves, "
        f"{s.active_lower.size + s.active_upper.size} active of 200 bounds), "
        f"grid 200 x 200 {done - fitted:.2f} s"
    )
    assert g.shape == (200, 200) and np.isfinite(g).all()
    assert s.active_lower.size > 5 and s.active_upper.size > 5
    assert (s(lxy) >= lo - 1e-9).all() and (s(uxy) <= up + 1e-9).all()
    np.testing.assert_allclose(s(xy), z, atol=1e-9, rtol=0)
    assert done - start < 30.0
