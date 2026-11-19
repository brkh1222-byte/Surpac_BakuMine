"""Acceptance tests for P4.T5 variogram models (research.md section 7.4)."""

import json
import math

import gstools as gs  # independent oracle, dev dependency only
import numpy as np
import pytest

from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.variogram import (
    EXPONENTIAL,
    GAUSSIAN,
    SCHEMA,
    SPHERICAL,
    Structure,
    VariogramArrays,
    VariogramModel,
    lag_covariance,
    point_covariance,
)

TYPES = ["spherical", "exponential", "gaussian"]


def along(direction, distances):
    return np.outer(np.atleast_1d(distances), np.asarray(direction, dtype=float))


# ------------------------------------------------------------ analytic values
@pytest.mark.parametrize("kind", TYPES)
def test_values_at_zero_range_and_far(kind):
    nugget, sill, a = 0.2, 1.3, 80.0
    m = VariogramModel(nugget, (Structure(kind, sill, a),))
    assert m.total_sill == pytest.approx(1.5)
    h = along([1, 0, 0], [0.0, 1e-9, a, 1000 * a])
    g = m.gamma(h)
    c = m.covariance(h)
    # h = 0: gamma is exactly 0, covariance is the total sill
    assert g[0] == 0.0 and c[0] == pytest.approx(1.5)
    # just off zero: the nugget appears as a jump
    assert g[1] == pytest.approx(nugget, abs=1e-6)
    # at the range: spherical reaches the sill, the others 95 % (practical range)
    at_range = {"spherical": 1.0, "exponential": 1 - math.exp(-3), "gaussian": 1 - math.exp(-3)}
    assert g[2] == pytest.approx(nugget + sill * at_range[kind], rel=1e-14)
    # far beyond the range: total sill, zero covariance
    assert g[3] == pytest.approx(1.5, rel=1e-12)
    assert c[3] == pytest.approx(0.0, abs=1e-12)
    np.testing.assert_allclose(g + c, 1.5, rtol=1e-14)


def test_hand_values_inside_range():
    a, h = 100.0, 40.0
    lag = [[0.0, h, 0.0]]
    sph = VariogramModel(0.0, (Structure("spherical", 2.0, a),)).gamma(lag)[0]
    assert sph == pytest.approx(2.0 * (1.5 * 0.4 - 0.5 * 0.4**3))  # = 1.136
    assert sph == pytest.approx(1.136)
    exp = VariogramModel(0.0, (Structure("exponential", 2.0, a),)).gamma(lag)[0]
    assert exp == pytest.approx(2.0 * (1 - math.exp(-1.2)))
    gau = VariogramModel(0.0, (Structure("gaussian", 2.0, a),)).gamma(lag)[0]
    assert gau == pytest.approx(2.0 * (1 - math.exp(-0.48)))
    # spherical is flat beyond the range
    far = VariogramModel(0.0, (Structure("spherical", 2.0, a),)).gamma([[0, 0, 100.0001]])
    assert far[0] == 2.0


def test_nested_structures_add():
    s1 = Structure("spherical", 0.5, (60, 30, 10), (40, -15, 5))
    s2 = Structure("exponential", 0.3, (200, 150, 40), (10, 0, 0))
    s3 = Structure("gaussian", 0.2, 90.0)
    nugget = 0.1
    full = VariogramModel(nugget, (s1, s2, s3))
    rng = np.random.default_rng(3)
    h = rng.normal(size=(500, 3)) * 80
    parts = sum(VariogramModel(0.0, (s,)).gamma(h) for s in (s1, s2, s3))
    np.testing.assert_allclose(full.gamma(h), nugget + parts, rtol=1e-13)
    assert full.total_sill == pytest.approx(1.1)
    np.testing.assert_allclose(full.covariance(h), 1.1 - full.gamma(h), rtol=1e-13)
    # pure nugget model
    pure = VariogramModel(0.7)
    np.testing.assert_array_equal(pure.gamma([[0, 0, 0], [1, 0, 0]]), [0.0, 0.7])
    np.testing.assert_array_equal(pure.covariance([[0, 0, 0], [1, 0, 0]]), [0.7, 0.0])


@pytest.mark.parametrize("kind", TYPES)
def test_anisotropic_ranges_honoured_along_each_axis(kind):
    ranges = (120.0, 60.0, 20.0)
    s = Structure(kind, 1.0, ranges, angles=(30.0, 20.0, 10.0))
    m = VariogramModel(0.0, (s,))
    iso = VariogramModel(0.0, (Structure(kind, 1.0, 1.0),))
    e = s.ellipsoid
    assert e.radii == pytest.approx(ranges)
    fractions = np.array([0.1, 0.5, 1.0, 1.7])
    reference = iso.gamma(along([1, 0, 0], fractions))  # same curve vs h / range
    for axis, a in zip((e.major_axis, e.semi_major_axis, e.minor_axis), ranges, strict=True):
        np.testing.assert_allclose(m.gamma(along(axis, fractions * a)), reference, rtol=1e-12)
        np.testing.assert_allclose(m.gamma(along(-axis, fractions * a)), reference, rtol=1e-12)


# -------------------------------------------------------------------- gstools
def to_gstools(structure, nugget=0.0):
    """Build the gstools model equivalent to one BakuMine structure.

    Mapping, checked piece by piece below:

    * Function shape. gstools evaluates ``cor(rescale * h / len_scale)`` with
      ``Spherical.cor(s) = 1 - 1.5 s + 0.5 s^3`` (s <= 1),
      ``Exponential.cor(s) = exp(-s)``, ``Gaussian.cor(s) = exp(-s^2)``; its
      DEFAULT rescale factors (1, 1, sqrt(pi)/2) make ``len_scale`` an
      integral scale, not our practical range. To get research.md's
      ``exp(-3h/a)`` and ``exp(-3h^2/a^2)`` with ``len_scale = a`` we pass
      rescale = 1 (spherical), 3 (exponential), sqrt(3) (gaussian).
    * Sill -> ``var``; nugget -> ``nugget`` (gstools also applies it only for
      h > 0).
    * Ranges. ``len_scale=[a_major, a_semi, a_minor]``: gstools' first main
      axis carries the first length, i.e. it is our MAJOR axis (our local
      frame is ordered semi-major, major, minor instead).
    * Angles. gstools uses radians and three angles (alpha, beta, gamma).
      The rows of ``gs.rotated_main_axes(3, angles)`` are the world
      directions of its main axes; as a matrix Q with those axes as COLUMNS,
        Q = Rx(gamma) @ Ry(beta) @ Rz(alpha)      (right-hand elementary
      matrices, applied about the FIXED world axes z, then y, then x).
      This is unlike our bearing (clockwise from north, about z first, then
      about the already-turned axes), so the angles are not converted one by
      one. Instead we take our axes Q = [major, semi-major, -minor] (minor
      negated to keep det +1 in this ordering) and read the angles off the
      product above:
        Q[0, 2] = sin(beta)
        Q[0, 0] = cos(beta) cos(alpha),  Q[0, 1] = -cos(beta) sin(alpha)
        Q[1, 2] = -sin(gamma) cos(beta), Q[2, 2] = cos(gamma) cos(beta)
      Rather than trusting this derivation, the function asserts that
      gstools' own main axes then coincide with ours.
    """
    e = structure.ellipsoid
    q = np.column_stack([e.major_axis, e.semi_major_axis, -e.minor_axis])
    beta = math.asin(max(-1.0, min(1.0, q[0, 2])))
    assert abs(math.cos(beta)) > 1e-6, "test case in gimbal lock; pick other angles"
    alpha = math.atan2(-q[0, 1], q[0, 0])
    gamma = math.atan2(-q[1, 2], q[2, 2])
    angles = [alpha, beta, gamma]

    axes = gs.rotated_main_axes(3, angles)
    np.testing.assert_allclose(axes[0], e.major_axis, atol=1e-12)
    np.testing.assert_allclose(axes[1], e.semi_major_axis, atol=1e-12)
    np.testing.assert_allclose(axes[2], -e.minor_axis, atol=1e-12)

    cls, rescale = {
        "spherical": (gs.Spherical, 1.0),
        "exponential": (gs.Exponential, 3.0),
        "gaussian": (gs.Gaussian, math.sqrt(3.0)),
    }[structure.type]
    return cls(
        dim=3,
        var=structure.sill,
        len_scale=list(structure.ranges),
        angles=angles,
        nugget=nugget,
        rescale=rescale,
    )


def gstools_gamma_cov(model, h):
    """Sum of gstools structures at lag vectors h (n, 3)."""
    pos = (h[:, 0], h[:, 1], h[:, 2])
    gamma = np.zeros(len(h))
    cov = np.zeros(len(h))
    for i, s in enumerate(model.structures):
        g = to_gstools(s, nugget=model.nugget if i == 0 else 0.0)
        gamma += g.vario_spatial(pos)
        cov += g.cov_spatial(pos)
    return gamma, cov


def random_lags(seed, scale, n=1000):
    rng = np.random.default_rng(seed)
    directions = rng.normal(size=(n, 3))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    # lengths from well inside to well beyond the range
    return directions * rng.uniform(0.01, 2.5, size=(n, 1)) * scale


@pytest.mark.parametrize("kind", TYPES)
def test_matches_gstools_isotropic(kind):
    m = VariogramModel(0.15, (Structure(kind, 0.85, 75.0),))
    h = random_lags(11, 75.0)
    gamma, cov = gstools_gamma_cov(m, h)
    np.testing.assert_allclose(m.gamma(h), gamma, rtol=1e-11, atol=1e-13)
    np.testing.assert_allclose(m.covariance(h), cov, rtol=1e-11, atol=1e-13)
    # isotropic: depends on |h| only, and equals the 1-D gstools variogram
    g1d = to_gstools(m.structures[0], nugget=0.15)
    np.testing.assert_allclose(
        m.gamma(h), g1d.variogram(np.linalg.norm(h, axis=1)), rtol=1e-11, atol=1e-13
    )


@pytest.mark.parametrize("kind", TYPES)
@pytest.mark.parametrize("convention", ["surpac_zxy_lrl", "gslib"])
@pytest.mark.parametrize(
    "angles", [(30.0, 20.0, 10.0), (250.0, -65.0, -40.0), (0.0, 0.0, 0.0), (135.0, 0.0, 90.0)]
)
def test_matches_gstools_anisotropic_rotated(kind, convention, angles):
    m = VariogramModel(0.1, (Structure(kind, 1.4, (150.0, 70.0, 25.0), angles, convention),))
    h = random_lags(5, 100.0)
    gamma, cov = gstools_gamma_cov(m, h)
    np.testing.assert_allclose(m.gamma(h), gamma, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m.covariance(h), cov, rtol=1e-10, atol=1e-12)


def test_matches_gstools_nested_anisotropic():
    m = VariogramModel(
        0.05,
        (
            Structure("spherical", 0.4, (60.0, 40.0, 12.0), (30.0, 20.0, 10.0)),
            Structure("exponential", 0.35, (220.0, 90.0, 45.0), (300.0, -10.0, 55.0)),
            Structure("gaussian", 0.2, (130.0, 130.0, 30.0), (75.0, 35.0, 0.0), "gslib"),
        ),
    )
    h = random_lags(99, 120.0)
    gamma, cov = gstools_gamma_cov(m, h)
    np.testing.assert_allclose(m.gamma(h), gamma, rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(m.covariance(h), cov, rtol=1e-10, atol=1e-12)


# ----------------------------------------------------------- matrices / shapes
def test_pairwise_matrices():
    m = VariogramModel(
        0.1, (Structure("spherical", 0.9, (100.0, 50.0, 20.0), (30.0, 20.0, 10.0)),)
    )
    rng = np.random.default_rng(1)
    a = rng.uniform(0, 150, size=(12, 3))
    b = rng.uniform(0, 150, size=(5, 3))
    caa = m.covariance_matrix(a)
    assert caa.shape == (12, 12)
    np.testing.assert_allclose(caa, caa.T, rtol=1e-14)
    np.testing.assert_allclose(np.diag(caa), m.total_sill)
    assert np.linalg.eigvalsh(caa).min() > 0  # positive definite
    cab = m.covariance_matrix(a, b)
    gab = m.gamma_matrix(a, b)
    assert cab.shape == (12, 5)
    for i, j in [(0, 0), (3, 4), (11, 2)]:
        assert cab[i, j] == pytest.approx(m.covariance(a[i] - b[j]))
        assert gab[i, j] == pytest.approx(m.gamma(a[i] - b[j]))
    np.testing.assert_array_equal(np.diag(m.gamma_matrix(a)), 0.0)
    # arbitrary leading shape and a single vector
    assert m.gamma(np.zeros((4, 6, 3))).shape == (4, 6)
    assert m.gamma([10.0, 0.0, 0.0]).shape == ()
    with pytest.raises(ValueError):
        m.gamma(np.zeros((4, 2)))
    with pytest.raises(ValueError):
        m.covariance_matrix(np.zeros(3))


# ------------------------------------------------------------------------ JSON
def test_json_round_trip_and_schema():
    axes = Ellipsoid(1.0, angles=(12.0, 34.0, 56.0)).rotation
    m = VariogramModel(
        0.05,
        (
            Structure("spherical", 0.4, (60.0, 40.0, 12.0), (30.0, 20.0, 10.0)),
            Structure("exponential", 0.35, (220.0, 90.0, 45.0), (300.0, -10.0, 55.0), "gslib"),
            Structure("gaussian", 0.2, 130.0, axes=axes),
        ),
    )
    text = m.to_json(indent=2)
    data = json.loads(text)
    assert data["schema"] == SCHEMA == "bakumine.variogram/1"
    assert data["nugget"] == 0.05
    assert data["structures"][0] == {
        "type": "spherical",
        "sill": 0.4,
        "ranges": [60.0, 40.0, 12.0],
        "rotation": {"convention": "surpac_zxy_lrl", "angles": [30.0, 20.0, 10.0]},
    }
    assert data["structures"][1]["rotation"]["convention"] == "gslib"
    assert set(data["structures"][2]["rotation"]) == {"axes"}

    back = VariogramModel.from_json(text)
    assert back == m
    h = random_lags(2, 100.0, n=200)
    np.testing.assert_array_equal(back.gamma(h), m.gamma(h))

    # minimal document: rotation optional, isotropic scalar range accepted
    mini = VariogramModel.from_dict(
        {"nugget": 1, "structures": [{"type": "Spherical", "sill": 2, "ranges": 50}]}
    )
    assert mini.structures[0].ranges == (50.0, 50.0, 50.0)
    assert mini.structures[0].type == "spherical"
    with pytest.raises(ValueError, match="schema"):
        VariogramModel.from_dict({"schema": "other/9", "nugget": 0})
    with pytest.raises(ValueError, match="missing"):
        VariogramModel.from_dict({"structures": [{"type": "spherical", "sill": 1}]})


def test_structure_from_ellipsoid():
    e = Ellipsoid(100.0, 2.0, 5.0, angles=(30.0, 20.0, 10.0), convention="gslib")
    s = Structure.from_ellipsoid("exponential", 0.6, e)
    assert s.ranges == pytest.approx((100.0, 50.0, 20.0))
    assert s.ellipsoid == Ellipsoid.from_radii(s.ranges, e.angles, "gslib")
    np.testing.assert_allclose(s.ellipsoid.rotation, e.rotation)


def test_validation():
    with pytest.raises(ValueError, match="nugget"):
        VariogramModel(-0.1)
    with pytest.raises(ValueError, match="sill"):
        Structure("spherical", -1.0, 10.0)
    with pytest.raises(ValueError, match="ranges"):
        Structure("spherical", 1.0, (10.0, 0.0, 5.0))
    with pytest.raises(ValueError, match="ranges"):
        Structure("spherical", 1.0, (10.0, 5.0))
    with pytest.raises(ValueError, match="ranges"):
        Structure("spherical", 1.0, -3.0)
    with pytest.raises(ValueError, match="unknown structure type"):
        Structure("cubic", 1.0, 10.0)
    with pytest.raises(ValueError, match="unknown rotation convention"):
        Structure("spherical", 1.0, 10.0, convention="nope")
    with pytest.raises(TypeError):
        VariogramModel(0.0, ({"type": "spherical"},))
    Structure("spherical", 0.0, 10.0)  # zero sill is allowed


# ------------------------------------------------------------------------ njit
def test_njit_equals_numpy():
    m = VariogramModel(
        0.05,
        (
            Structure("spherical", 0.4, (60.0, 40.0, 12.0), (30.0, 20.0, 10.0)),
            Structure("exponential", 0.35, (220.0, 90.0, 45.0), (300.0, -10.0, 55.0)),
            Structure("gaussian", 0.2, (130.0, 130.0, 30.0), (75.0, 35.0, 0.0), "gslib"),
        ),
    )
    arrays = m.as_arrays()
    assert isinstance(arrays, VariogramArrays)
    assert arrays.nugget == 0.05
    assert arrays.types.dtype == np.int64 and arrays.sills.dtype == np.float64
    np.testing.assert_array_equal(arrays.types, [SPHERICAL, EXPONENTIAL, GAUSSIAN])
    assert arrays.transforms.shape == (3, 3, 3) and arrays.transforms.flags.c_contiguous

    h = np.vstack([random_lags(8, 100.0), np.zeros((1, 3)), [[60.0, 0.0, 0.0]]])
    expected = m.covariance(h)
    got = np.array([lag_covariance(dx, dy, dz, *arrays) for dx, dy, dz in h])
    np.testing.assert_allclose(got, expected, rtol=1e-12, atol=1e-14)
    assert got[-2] == pytest.approx(m.total_sill)  # zero lag includes the nugget

    rng = np.random.default_rng(4)
    p = rng.uniform(0, 100, size=(50, 3))
    q = rng.uniform(0, 100, size=(50, 3))
    got = np.array([point_covariance(p[i], q[i], *arrays) for i in range(50)])
    np.testing.assert_allclose(got, m.covariance(p - q), rtol=1e-12, atol=1e-14)
    assert point_covariance(p[0], p[0], *arrays) == pytest.approx(m.total_sill)

    # pure nugget: empty structure arrays still type-check in numba
    pure = VariogramModel(0.3).as_arrays()
    assert pure.transforms.shape == (0, 3, 3)
    assert lag_covariance(0.0, 0.0, 0.0, *pure) == 0.3
    assert lag_covariance(1.0, 0.0, 0.0, *pure) == 0.0
