"""Dip / dip azimuth / pitch input and the inverse of the rotation conventions."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from bakumine.core.estimation.orientation import (
    angles_from_plane,
    angles_from_rotation,
    angles_from_spec,
    axis_trend_plunge,
    describe_axes,
    ellipsoid_from_plane,
    parse_ratios,
    plane_axes,
)
from bakumine.core.estimation.rotation import CONVENTIONS, Ellipsoid, rotation_matrix


def direction(azimuth: float, plunge: float) -> np.ndarray:
    """Unit vector plunging ``plunge`` degrees below horizontal towards ``azimuth``."""
    a, p = np.radians([azimuth, plunge])
    return np.array([np.sin(a) * np.cos(p), np.cos(a) * np.cos(p), -np.sin(p)])


def same_axis(a, b) -> bool:
    return bool(abs(abs(float(np.dot(a, b))) - 1.0) < 1e-9)


def test_plane_dipping_69_towards_216():
    """The case from the field: major along strike 306/126, semi-major down dip, minor = pole."""
    ellipsoid = ellipsoid_from_plane((100.0, 60.0, 10.0), dip=69.0, dip_azimuth=216.0)
    major, semi, minor = ellipsoid.major_axis, ellipsoid.semi_major_axis, ellipsoid.minor_axis
    assert abs(major[2]) < 1e-12  # horizontal
    assert same_axis(major, direction(126.0, 0.0)) and same_axis(major, direction(306.0, 0.0))
    assert same_axis(semi, direction(216.0, 69.0))  # straight down dip
    # the pole of the plane: perpendicular to strike and to the dip line, 21 degrees from
    # horizontal, on the side away from the dip direction
    assert same_axis(minor, direction(36.0, 21.0))
    assert (
        abs(minor @ direction(126.0, 0.0)) < 1e-12 and abs(minor @ direction(216.0, 69.0)) < 1e-12
    )
    assert ellipsoid.radii == pytest.approx((100.0, 60.0, 10.0))
    # points 100 m along strike, 60 m down dip and 10 m across are all on the ellipsoid
    for axis, radius in ((direction(306.0, 0.0), 100.0), (direction(216.0, 69.0), 60.0)):
        assert ellipsoid.distance(np.zeros(3), radius * axis) == pytest.approx(100.0)
    assert ellipsoid.distance(np.zeros(3), 10.0 * direction(36.0, 21.0)) == pytest.approx(100.0)

    described = describe_axes(ellipsoid)
    assert described["major"] == {"azimuth": 126.0, "plunge": 0.0}
    assert described["semi_major"] == {"azimuth": 216.0, "plunge": 69.0}
    assert described["minor"] == {"azimuth": 36.0, "plunge": 21.0}
    assert described["plane"] == {"dip": 69.0, "dip_azimuth": 216.0}
    assert described["angles"] == [126.0, 0.0, -69.0]


@pytest.mark.parametrize("dip", [0.0, 20.0, 69.0, 90.0])
@pytest.mark.parametrize("dip_azimuth", [0.0, 37.0, 216.0, 359.0])
@pytest.mark.parametrize("pitch", [0.0, 30.0, 90.0, 140.0])
def test_plane_axes_geometry(dip, dip_azimuth, pitch):
    major, semi, minor = plane_axes(dip, dip_azimuth, pitch)
    down_dip = direction(dip_azimuth, dip)
    strike = direction(dip_azimuth - 90.0, 0.0)
    # an orthonormal right-handed frame (semi-major, major, minor) with the pole up
    frame = np.column_stack([semi, major, minor])
    np.testing.assert_allclose(frame.T @ frame, np.eye(3), atol=1e-12)
    assert np.linalg.det(frame) == pytest.approx(1.0)
    assert minor[2] >= -1e-12
    assert abs(minor @ down_dip) < 1e-12 and abs(minor @ strike) < 1e-12
    # the pitch is the angle in the plane from the strike direction towards down dip
    assert major @ strike == pytest.approx(np.cos(np.radians(pitch)), abs=1e-12)
    assert major @ down_dip == pytest.approx(np.sin(np.radians(pitch)), abs=1e-12)
    # the angles of the project convention reproduce these axes
    r = rotation_matrix(*angles_from_plane(dip, dip_azimuth, pitch))
    np.testing.assert_allclose(r, frame, atol=1e-8)


def test_pitch_90_puts_the_major_axis_down_dip():
    major, semi, _ = plane_axes(60.0, 90.0, 90.0)
    assert same_axis(major, direction(90.0, 60.0))
    assert same_axis(semi, direction(0.0, 0.0))


@pytest.mark.parametrize("convention", sorted(CONVENTIONS))
def test_angles_from_rotation_is_the_inverse_of_rotation_matrix(convention):
    rng = np.random.default_rng(7)
    for _ in range(500):
        angles = (rng.uniform(0, 360), rng.uniform(-89.9, 89.9), rng.uniform(-179.9, 179.9))
        r = rotation_matrix(*angles, convention=convention)
        back = angles_from_rotation(r, convention)
        np.testing.assert_allclose(back, angles, atol=1e-6)
        np.testing.assert_allclose(rotation_matrix(*back, convention=convention), r, atol=1e-8)
    for angles in ((10.0, 90.0, 0.0), (200.0, -90.0, 0.0), (0.0, 0.0, 180.0)):  # gimbal lock
        r = rotation_matrix(*angles, convention=convention)
        back = angles_from_rotation(r, convention)
        np.testing.assert_allclose(rotation_matrix(*back, convention=convention), r, atol=1e-8)


def test_angles_from_rotation_against_scipy():
    """Independent check: scipy's intrinsic Z-X-Y Euler angles of the same matrix."""
    rng = np.random.default_rng(11)
    for _ in range(200):
        r = Rotation.random(random_state=rng).as_matrix()
        alpha, beta, gamma = Rotation.from_matrix(r).as_euler("ZXY", degrees=True)
        # surpac_zxy_lrl: left, right, left hand = (-alpha, +beta, -gamma)
        bearing, plunge, dip = angles_from_rotation(r, "surpac_zxy_lrl")
        assert (bearing + alpha) % 360.0 == pytest.approx(0.0, abs=1e-6) or (
            bearing + alpha
        ) % 360.0 == pytest.approx(360.0, abs=1e-6)
        assert plunge == pytest.approx(beta, abs=1e-6)
        assert dip == pytest.approx(-gamma, abs=1e-6)
        assert angles_from_rotation(r, "gslib")[2] == pytest.approx(gamma, abs=1e-6)


def test_axis_trend_plunge_and_ratios():
    assert axis_trend_plunge(direction(216.0, 69.0)) == pytest.approx((216.0, 69.0))
    assert axis_trend_plunge(-direction(216.0, 69.0)) == pytest.approx((216.0, 69.0))
    assert axis_trend_plunge([0.0, 0.0, 1.0]) == (0.0, 90.0)
    assert axis_trend_plunge([-1.0, 0.0, 0.0]) == pytest.approx((90.0, 0.0))  # horizontal: 0-180
    assert parse_ratios("4:2:1") == (4.0, 2.0, 1.0)
    assert parse_ratios([3, 3, 1]) == (3.0, 3.0, 1.0)
    assert parse_ratios("4, 2, 1") == (4.0, 2.0, 1.0)
    for bad in ("4:2", "4:0:1", "a:b:c", None, [1, 2]):
        with pytest.raises(ValueError):
            parse_ratios(bad)


def test_angles_from_spec():
    assert angles_from_spec({"angles": [35, -10, 50]}) == (35.0, -10.0, 50.0)
    assert angles_from_spec({"plane": {"dip": 69, "dip_azimuth": 216}}) == (126.0, 0.0, -69.0)
    assert angles_from_spec({}, (1, 2, 3)) == (1.0, 2.0, 3.0)
    assert angles_from_spec(None) is None
    for bad in (
        {"angles": [1, 2]},
        {"plane": {"dip": 69}},
        {"plane": {"dip": 100, "dip_azimuth": 0}},
        {"plane": {"dip": 10, "dip_azimuth": 0, "rake": 3}},
        {"plane": {"dip": 10, "dip_azimuth": 0}, "angles": [0, 0, 0]},
    ):
        with pytest.raises(ValueError):
            angles_from_spec(bad)
    assert describe_axes(None) is None
    assert describe_axes(Ellipsoid(10.0, 2.0, 5.0))["ratios"] == [1.0, 0.5, 0.2]
