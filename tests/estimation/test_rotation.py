"""Acceptance tests for P4.T1 rotation conventions (research.md section 7.5).

These tests check internal consistency and hand-computed vectors. They do NOT
establish that ``surpac_zxy_lrl`` matches Surpac: no Surpac reference was
available (see the module docstring of ``rotation.py``).
"""

import json
import math

import numpy as np
import pytest

from bakumine.core.estimation.rotation import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
    Ellipsoid,
    rotation_from_axes,
    rotation_from_sequence,
    rotation_matrix,
)

EAST = np.array([1.0, 0.0, 0.0])
NORTH = np.array([0.0, 1.0, 0.0])
UP = np.array([0.0, 0.0, 1.0])

ANGLE_SETS = [
    (0.0, 0.0, 0.0),
    (30.0, 20.0, 10.0),
    (-135.0, 60.0, -75.0),
    (275.0, -40.0, 170.0),
    (359.0, 89.0, 1.0),
]


def test_default_convention_and_registry():
    assert DEFAULT_CONVENTION == "surpac_zxy_lrl"
    assert set(CONVENTIONS) == {"surpac_zxy_lrl", "gslib"}
    with pytest.raises(ValueError, match="unknown rotation convention"):
        rotation_matrix(0, 0, 0, "datamine_zxz")  # deliberately not provided


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_identity_at_zero_angles(convention):
    np.testing.assert_array_equal(rotation_matrix(0, 0, 0, convention), np.eye(3))
    e = Ellipsoid(angles=(0, 0, 0), convention=convention)
    np.testing.assert_array_equal(e.major_axis, NORTH)
    np.testing.assert_array_equal(e.semi_major_axis, EAST)
    np.testing.assert_array_equal(e.minor_axis, UP)


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
@pytest.mark.parametrize("angles", ANGLE_SETS)
def test_orthonormal_proper(convention, angles):
    r = rotation_matrix(*angles, convention)
    np.testing.assert_allclose(r.T @ r, np.eye(3), atol=1e-14)
    assert np.linalg.det(r) == pytest.approx(1.0, abs=1e-14)


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_bearing_alone_is_azimuth_clockwise_from_north(convention):
    # bearing 90 -> major axis points east
    r = rotation_matrix(90, 0, 0, convention)
    np.testing.assert_allclose(r[:, 1], EAST, atol=1e-15)
    # semi-major swings from east to south, minor stays up
    np.testing.assert_allclose(r[:, 0], -NORTH, atol=1e-15)
    np.testing.assert_allclose(r[:, 2], UP, atol=1e-15)
    for bearing in (0, 30, 135, 200, 315):
        major = rotation_matrix(bearing, 0, 0, convention)[:, 1]
        azimuth = math.degrees(math.atan2(major[0], major[1])) % 360
        assert azimuth == pytest.approx(bearing, abs=1e-12)
        assert major[2] == pytest.approx(0.0, abs=1e-15)


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
@pytest.mark.parametrize("plunge", [-60.0, -20.0, 10.0, 45.0])
def test_plunge_alone_tilts_major_axis_positive_up(convention, plunge):
    # Documented sign for both conventions: positive second angle = major
    # axis rises above horizontal; negative = plunges below.
    r = rotation_matrix(0, plunge, 0, convention)
    major = r[:, 1]
    inclination = math.degrees(math.asin(major[2]))
    assert inclination == pytest.approx(plunge, abs=1e-12)
    # stays in the north-up plane, still pointing north-ish
    assert major[0] == pytest.approx(0.0, abs=1e-15)
    assert major[1] > 0
    # the semi-major axis (the rotation axis) is untouched
    np.testing.assert_allclose(r[:, 0], EAST, atol=1e-15)
    # with a bearing the inclination is unchanged
    major = rotation_matrix(77, plunge, 0, convention)[:, 1]
    assert math.degrees(math.asin(major[2])) == pytest.approx(plunge, abs=1e-12)


def test_dip_alone_sign_surpac_and_gslib():
    # Surpac ZXY LRL: third rotation is LEFT-hand about the major axis
    # (north here): the east end of the semi-major axis goes UP for dip > 0.
    r = rotation_matrix(0, 0, 25, "surpac_zxy_lrl")
    np.testing.assert_allclose(r[:, 1], NORTH, atol=1e-15)
    np.testing.assert_allclose(
        r[:, 0], [math.cos(math.radians(25)), 0, math.sin(math.radians(25))], atol=1e-15
    )
    # GSLIB: same rotation with the opposite sign (east end goes DOWN).
    g = rotation_matrix(0, 0, 25, "gslib")
    np.testing.assert_allclose(
        g[:, 0], [math.cos(math.radians(25)), 0, -math.sin(math.radians(25))], atol=1e-15
    )
    np.testing.assert_allclose(
        rotation_matrix(12, 34, 56, "surpac_zxy_lrl"),
        rotation_matrix(12, 34, -56, "gslib"),
        atol=1e-15,
    )


def test_worked_case_bearing30_plunge20_dip10_hand_computed():
    """plan.md P4.T1 worked case, convention surpac_zxy_lrl.

    Hand computation (independent of the code under test):

    Major axis: start at north (0, 1, 0).
      * dip (about the major axis itself) does not move it.
      * plunge +20 lifts it in the north-up plane:
          (0, cos20, sin20) = (0, 0.9396926, 0.3420201)
      * bearing 30 turns the horizontal part clockwise from north to
        azimuth 030:
          east  = cos20 * sin30 = 0.9396926 * 0.5       = 0.4698463
          north = cos20 * cos30 = 0.9396926 * 0.8660254 = 0.8137977
          up    = sin20                                 = 0.3420201

    Semi-major axis: start at east (1, 0, 0).
      * dip +10 left-hand about north raises the east end:
          (cos10, 0, sin10) = (0.9848078, 0, 0.1736482)
      * plunge +20 right-hand about east, (y, z) -> (y c - z s, y s + z c):
          (0.9848078, -0.1736482*0.3420201, 0.1736482*0.9396926)
        = (0.9848078, -0.0593912, 0.1631759)
      * bearing 30 clockwise about up, (x, y) -> (x c + y s, -x s + y c):
          east  = 0.9848078*0.8660254 + (-0.0593912)*0.5       = 0.8231729
          north = -0.9848078*0.5      + (-0.0593912)*0.8660254 = -0.5438382
          up    = 0.1631759

    Minor axis = semi-major x major:
          east  = (-0.5438382)(0.3420201) - (0.1631759)(0.8137977) = -0.3187958
          north = (0.1631759)(0.4698463) - (0.8231729)(0.3420201)  = -0.2048741
          up    = (0.8231729)(0.8137977) - (-0.5438382)(0.4698463) = 0.9254166
    """
    e = Ellipsoid(100.0, 2.0, 4.0, angles=(30.0, 20.0, 10.0))
    assert e.convention == "surpac_zxy_lrl"
    np.testing.assert_allclose(e.major_axis, [0.4698463, 0.8137977, 0.3420201], atol=5e-7)
    np.testing.assert_allclose(e.semi_major_axis, [0.8231729, -0.5438382, 0.1631759], atol=5e-7)
    np.testing.assert_allclose(e.minor_axis, [-0.3187958, -0.2048741, 0.9254166], atol=5e-7)
    np.testing.assert_allclose(
        rotation_matrix(30, 20, 10)[:, 1], [0.4698463, 0.8137977, 0.3420201], atol=5e-7
    )


# --------------------------------------------------------------------- GSLIB
def gslib_setrot(ang1, ang2, ang3, anis1, anis2):
    """Independent transcription of the rotation-matrix formulas published for
    GSLIB's ``setrot`` (Deutsch & Journel 1998, section II.4), written from
    the formulas, not from the code under test.

    Returns the 3x3 matrix M such that the anisotropic squared distance of a
    world lag (dx, dy, dz) is |M @ (dx, dy, dz)|^2. Row 1 is the principal
    direction; rows 2 and 3 are divided by the anisotropy factors
    anis1 = (range horizontal-minor / range principal) and
    anis2 = (range vertical-minor / range principal).
    """
    deg2rad = math.pi / 180.0
    if 0.0 <= ang1 < 270.0:
        alpha = (90.0 - ang1) * deg2rad
    else:
        alpha = (450.0 - ang1) * deg2rad
    beta = -1.0 * ang2 * deg2rad
    theta = ang3 * deg2rad
    sina, sinb, sint = math.sin(alpha), math.sin(beta), math.sin(theta)
    cosa, cosb, cost = math.cos(alpha), math.cos(beta), math.cos(theta)
    afac1 = 1.0 / anis1
    afac2 = 1.0 / anis2
    m = np.empty((3, 3))
    m[0, 0] = cosb * cosa
    m[0, 1] = cosb * sina
    m[0, 2] = -sinb
    m[1, 0] = afac1 * (-cost * sina + sint * sinb * cosa)
    m[1, 1] = afac1 * (cost * cosa + sint * sinb * sina)
    m[1, 2] = afac1 * (sint * cosb)
    m[2, 0] = afac2 * (sint * sina + cost * sinb * cosa)
    m[2, 1] = afac2 * (-sint * cosa + cost * sinb * sina)
    m[2, 2] = afac2 * (cost * cosb)
    return m


@pytest.mark.parametrize(
    "angles", [(0.0, 0.0, 0.0), (30.0, 20.0, 10.0), (225.0, -35.0, 80.0), (300.0, 50.0, -25.0)]
)
def test_gslib_matches_setrot_formulas(angles):
    r = rotation_matrix(*angles, "gslib")
    m = gslib_setrot(*angles, 1.0, 1.0)
    # GSLIB's rotated frame is (principal, horizontal-minor, vertical-minor).
    # At zero angles its rows are (north, WEST, up); ours are columns
    # (east, north, up) = (semi-major, major, minor). Hence:
    np.testing.assert_allclose(r[:, 1], m[0], atol=1e-14)  # major = principal
    np.testing.assert_allclose(r[:, 0], -m[1], atol=1e-14)  # semi-major = -row 2
    np.testing.assert_allclose(r[:, 2], m[2], atol=1e-14)  # minor = row 3

    # Sign-free check: anisotropic distances agree for random lags.
    rng = np.random.default_rng(7)
    lags = rng.normal(size=(200, 3)) * 50
    a_major, a_semi, a_minor = 120.0, 45.0, 15.0
    e = Ellipsoid.from_radii((a_major, a_semi, a_minor), angles, "gslib")
    m = gslib_setrot(*angles, a_semi / a_major, a_minor / a_major)
    d_gslib = np.sqrt(((lags @ m.T) ** 2).sum(axis=1))
    np.testing.assert_allclose(e.distance(lags, np.zeros(3)), d_gslib, rtol=1e-13)


def test_gslib_angle_meaning():
    # ang1 azimuth clockwise from north; ang2 negative = dipping down.
    major = rotation_matrix(90, -30, 0, "gslib")[:, 1]
    np.testing.assert_allclose(
        major, [math.cos(math.radians(30)), 0, -math.sin(math.radians(30))], atol=1e-15
    )


# ------------------------------------------------------------ generic builders
def test_rotation_from_sequence_is_intrinsic():
    # 90 right-hand about z, then 90 right-hand about the NEW x axis (which
    # now points north): local y (after step 1: west) is lifted to up.
    r =rotation_from_sequence("zxy", "rrr", (90, 90, 0))
    np.testing.assert_allclose(r[:, 0], NORTH, atol=1e-15)
    np.testing.assert_allclose(r[:, 1], UP, atol=1e-15)
    np.testing.assert_allclose(r[:, 2], EAST, atol=1e-15)
    with pytest.raises(ValueError):
        rotation_from_sequence("zxq", "rrr", (1, 2, 3))
    with pytest.raises(ValueError):
        rotation_from_sequence("zxy", "rrx", (1, 2, 3))


def test_from_axes_bypasses_conventions():
    ref = Ellipsoid(80.0, 2.0, 5.0, angles=(30.0, 20.0, 10.0))
    e = Ellipsoid.from_axes(
        ref.major_axis * 3.0, ref.semi_major_axis * 0.2, major_radius=80.0,
        semi_major_ratio=2.0, minor_ratio=5.0,
    )
    np.testing.assert_allclose(e.rotation, ref.rotation, atol=1e-14)
    # minor may be given with either sign (an axis is a line)
    for sign in (1.0, -1.0):
        r = rotation_from_axes(ref.major_axis, ref.semi_major_axis, sign * ref.minor_axis)
        np.testing.assert_allclose(r, ref.rotation, atol=1e-14)
    m = Ellipsoid.from_matrix(ref.rotation, 80.0, 2.0, 5.0)
    np.testing.assert_allclose(m.rotation, ref.rotation, atol=0)
    with pytest.raises(ValueError, match="perpendicular"):
        rotation_from_axes([0, 1, 0], [1, 1, 0])
    with pytest.raises(ValueError, match="perpendicular"):
        rotation_from_axes([0, 1, 0], [1, 0, 0], [0, 1, 1])
    with pytest.raises(ValueError, match="reflection"):
        Ellipsoid.from_matrix(np.diag([1.0, 1.0, -1.0]))
    with pytest.raises(ValueError, match="orthonormal"):
        Ellipsoid.from_matrix(np.diag([1.0, 2.0, 1.0]))


# ------------------------------------------------------------------ Ellipsoid
@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_to_isotropic_distance_equals_research_formula(convention):
    """research.md 7.5:
        [u v w]^T = R^-1 (x_sample - x_block)
        d_aniso   = sqrt(u^2 + (v r_semi)^2 + (w r_minor)^2)
    with u, v, w along the major, semi-major and minor axes."""
    rng = np.random.default_rng(42)
    e = Ellipsoid(150.0, 2.5, 6.0, angles=(30.0, 20.0, 10.0), convention=convention)
    samples = rng.uniform(-200, 200, size=(300, 3))
    blocks = rng.uniform(-200, 200, size=(300, 3))
    r_semi, r_minor = 2.5, 6.0
    diff = samples - blocks
    u = diff @ e.major_axis  # explicit projections onto each axis
    v = diff @ e.semi_major_axis
    w = diff @ e.minor_axis
    expected = np.sqrt(u**2 + (v * r_semi) ** 2 + (w * r_minor) ** 2)

    iso_s = e.to_isotropic(samples)
    iso_b = e.to_isotropic(blocks)
    np.testing.assert_allclose(np.linalg.norm(iso_s - iso_b, axis=1), expected, rtol=1e-12)
    np.testing.assert_allclose(e.distance(samples, blocks), expected, rtol=1e-12)
    centre = np.array([10.0, -20.0, 30.0])
    np.testing.assert_allclose(
        e.to_isotropic(samples, centre), iso_s - e.to_isotropic(centre), atol=1e-9
    )
    # full matrix by broadcasting
    dm = e.distance(samples[:5, None, :], blocks[None, :7, :])
    assert dm.shape == (5, 7)
    assert dm[2, 2] == pytest.approx(expected[2])
    # unstretched local frame preserves ordinary length
    np.testing.assert_allclose(
        np.linalg.norm(e.to_local(diff), axis=1), np.linalg.norm(diff, axis=1), rtol=1e-12
    )


def test_axis_end_points_are_on_the_surface():
    e = Ellipsoid(100.0, 2.0, 4.0, angles=(30.0, 20.0, 10.0))
    assert e.radii == (100.0, 50.0, 25.0)
    centre = np.array([500.0, 600.0, 70.0])
    tips = centre + np.array(
        [100.0 * e.major_axis, 50.0 * e.semi_major_axis, -25.0 * e.minor_axis]
    )
    np.testing.assert_allclose(e.distance(tips, centre), 100.0, rtol=1e-12)
    assert e.contains(centre + 0.999 * (tips - centre), centre).all()
    assert not e.contains(centre + 1.001 * (tips - centre), centre).any()
    # isotropic ellipsoid = Euclidean distance
    iso = Ellipsoid(10.0)
    assert iso.distance([3.0, 4.0, 12.0], [0.0, 0.0, 0.0]) == pytest.approx(13.0)


def test_ellipsoid_validation_equality_and_json():
    with pytest.raises(ValueError):
        Ellipsoid(0.0)
    with pytest.raises(ValueError):
        Ellipsoid(10.0, semi_major_ratio=-1.0)
    with pytest.raises(ValueError):
        Ellipsoid(10.0, minor_ratio=float("nan"))
    with pytest.raises(ValueError, match="unknown rotation convention"):
        Ellipsoid(10.0, convention="nope")
    with pytest.raises(ValueError):
        Ellipsoid(10.0).to_isotropic(np.zeros((4, 2)))

    a = Ellipsoid(100, 2, 4, angles=(30, 20, 10), convention="gslib")
    b = Ellipsoid.from_dict(json.loads(json.dumps(a.to_dict())))
    assert a == b and hash(a) == hash(b)
    assert b.convention == "gslib" and b.angles == (30.0, 20.0, 10.0)

    c = Ellipsoid.from_matrix(a.rotation, 100, 2, 4)
    d = Ellipsoid.from_dict(json.loads(json.dumps(c.to_dict())))
    assert c == d
    np.testing.assert_array_equal(c.rotation, d.rotation)
    assert c != a  # different parameterisation, even though same geometry
    with pytest.raises(ValueError):
        a.rotation[0, 0] = 5.0  # read-only


def test_transform_is_bit_reproducible_per_point():
    """A point's transform must not depend on the batch it is in (the search
    needs coincident sample/target pairs to give a distance of exactly 0)."""
    rng = np.random.default_rng(0)
    e = Ellipsoid(60.0, 1.5, 3.0, angles=(30.0, 20.0, 10.0))
    pts = rng.uniform(-1e4, 1e4, size=(1001, 3))
    full = e.to_isotropic(pts)
    for size in (1, 2, 3, 5, 7, 64):
        for start in (0, 1, 17, 500):
            np.testing.assert_array_equal(
                e.to_isotropic(pts[start : start + size]), full[start : start + size]
            )
    np.testing.assert_array_equal(e.to_isotropic(pts[::-1])[::-1], full)
    np.testing.assert_array_equal(e.to_isotropic(np.asfortranarray(pts)), full)
    assert (e.distance(pts, pts.copy()) == 0.0).all()
