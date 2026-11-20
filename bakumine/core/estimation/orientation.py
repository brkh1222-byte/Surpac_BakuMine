"""Orientation of an ellipsoid entered the way a geologist measures a plane.

A tabular body (vein, lode, bed) is described by the **dip** of its plane, the **dip
azimuth** (the direction it dips towards, clockwise from north) and optionally a **pitch**
(rake) of the long axis within the plane. The ellipsoid axes follow from them:

* **minor** axis = the pole (normal) of the plane, pointing up;
* with ``pitch = 0`` the **major** axis is horizontal along strike, in the right-hand-rule
  strike direction ``dip_azimuth - 90`` (the plane dips to the right when looking along
  it), and the **semi-major** axis points straight down dip;
* a pitch turns the major axis within the plane, away from that strike direction towards
  the down-dip direction: ``pitch = 90`` puts the major axis down dip, ``pitch = 180`` is
  the opposite strike direction. The semi-major axis stays in the plane, perpendicular
  to the major axis.

So a plane dipping 69 degrees towards 216 has its major axis horizontal at azimuth 126
(= 306, an axis is a line), the semi-major axis plunging 69 degrees towards 216 and the
minor axis plunging 21 degrees towards 036.

The axes are converted to the three angles of a rotation convention of
:mod:`bakumine.core.estimation.rotation` with :func:`angles_from_rotation`, which is the
exact inverse of :func:`~bakumine.core.estimation.rotation.rotation_matrix` (tested by
round trip). Nothing here assumes what Surpac does: the plane defines the axes, and the
angles are whatever the project's convention needs to reproduce those axes.

Coordinates are x = east, y = north, z = up; angles in degrees.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from bakumine.core.estimation.rotation import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
    Ellipsoid,
    rotation_from_axes,
)

__all__ = [
    "angles_from_plane",
    "angles_from_rotation",
    "angles_from_spec",
    "axis_trend_plunge",
    "describe_axes",
    "ellipsoid_from_plane",
    "parse_ratios",
    "plane_axes",
]


def plane_axes(
    dip: float, dip_azimuth: float, pitch: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(major, semi-major, minor) unit vectors of a plane; see the module docstring."""
    dip, dip_azimuth, pitch = float(dip), float(dip_azimuth), float(pitch)
    if not (np.isfinite(dip) and 0.0 <= dip <= 90.0):
        raise ValueError(f"dip must be between 0 and 90 degrees, got {dip}")
    if not np.isfinite(dip_azimuth):
        raise ValueError("dip azimuth must be a finite number")
    if not np.isfinite(pitch):
        raise ValueError("pitch must be a finite number")
    a, d, p = np.radians([dip_azimuth, dip, pitch])
    down_dip = np.array([np.sin(a) * np.cos(d), np.cos(a) * np.cos(d), -np.sin(d)])
    strike = np.array([-np.cos(a), np.sin(a), 0.0])  # azimuth dip_azimuth - 90
    pole = np.array([np.sin(a) * np.sin(d), np.cos(a) * np.sin(d), np.cos(d)])
    major = np.cos(p) * strike + np.sin(p) * down_dip
    semi = -np.sin(p) * strike + np.cos(p) * down_dip
    return major, semi, pole


def angles_from_rotation(
    rotation: Any, convention: str = DEFAULT_CONVENTION
) -> tuple[float, float, float]:
    """The three angles of ``convention`` that give the rotation matrix ``rotation``.

    Inverse of :func:`~bakumine.core.estimation.rotation.rotation_matrix` for the
    ``zxy`` conventions (``surpac_zxy_lrl``, ``gslib``). The first angle is returned in
    ``[0, 360)``, the second in ``[-90, 90]``, the third in ``(-180, 180]``. When the
    major axis is vertical the first and third angle turn about the same line; the third
    is then returned as 0.
    """
    try:
        axes, hands = CONVENTIONS[convention]
    except KeyError:
        raise ValueError(f"unknown rotation convention {convention!r}") from None
    if axes != "zxy":
        raise ValueError(f"convention {convention!r} is not a z-x-y sequence")
    r = np.asarray(rotation, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError("rotation must be a 3x3 matrix")
    # R = Rz(alpha) @ Rx(beta) @ Ry(gamma), right-hand elementary rotations
    beta = float(np.arcsin(np.clip(r[2, 1], -1.0, 1.0)))
    if abs(np.cos(beta)) < 1e-9:
        gamma = 0.0
        alpha = float(np.arctan2(r[1, 0], r[0, 0]))
    else:
        alpha = float(np.arctan2(-r[0, 1], r[1, 1]))
        gamma = float(np.arctan2(-r[2, 0], r[2, 2]))
    signs = [1.0 if h == "r" else -1.0 for h in hands]
    first, second, third = (
        np.degrees(v) * s for v, s in zip((alpha, beta, gamma), signs, strict=True)
    )
    first = float(first % 360.0)
    if abs(first - 360.0) < 1e-9:
        first = 0.0
    # + 0.0 turns -0.0 into 0.0
    return (
        round(first, 9) + 0.0,
        round(float(second), 9) + 0.0,
        round(float(third), 9) + 0.0,
    )


def angles_from_plane(
    dip: float, dip_azimuth: float, pitch: float = 0.0, convention: str = DEFAULT_CONVENTION
) -> tuple[float, float, float]:
    """Rotation angles of ``convention`` for a plane given by dip / dip azimuth / pitch."""
    major, semi, minor = plane_axes(dip, dip_azimuth, pitch)
    return angles_from_rotation(rotation_from_axes(major, semi, minor), convention)


def ellipsoid_from_plane(
    radii: tuple[float, float, float],
    dip: float,
    dip_azimuth: float,
    pitch: float = 0.0,
    convention: str = DEFAULT_CONVENTION,
) -> Ellipsoid:
    """Ellipsoid with radii (along the major axis, the semi-major axis, across the plane)."""
    return Ellipsoid.from_radii(
        radii, angles_from_plane(dip, dip_azimuth, pitch, convention), convention
    )


def angles_from_spec(
    spec: Mapping[str, Any] | None, default: tuple[float, ...] | None = None
) -> tuple[float, float, float] | None:
    """Angles (bearing, plunge, dip) of a request object.

    ``spec`` may carry ``angles: [bearing, plunge, dip]`` or ``plane: {dip, dip_azimuth,
    pitch?}`` (not both). With neither, ``default`` is returned.
    """
    spec = spec or {}
    plane, angles = spec.get("plane"), spec.get("angles")
    if plane is not None and angles is not None:
        raise ValueError("give either angles or a plane (dip, dip azimuth), not both")
    if plane is not None:
        if not isinstance(plane, Mapping):
            raise ValueError("plane must be an object with dip and dip_azimuth")
        unknown = set(plane) - {"dip", "dip_azimuth", "pitch"}
        if unknown:
            raise ValueError(f"plane: unknown keys {sorted(unknown)}")
        if plane.get("dip") is None or plane.get("dip_azimuth") is None:
            raise ValueError("plane needs dip and dip_azimuth")
        return angles_from_plane(
            float(plane["dip"]), float(plane["dip_azimuth"]), float(plane.get("pitch") or 0.0)
        )
    if angles is not None:
        try:
            values = tuple(float(v) for v in angles)
        except (TypeError, ValueError):
            values = ()
        if len(values) != 3:
            raise ValueError("angles needs three numbers (bearing, plunge, dip)")
        return values  # type: ignore[return-value]
    return None if default is None else tuple(float(v) for v in default)  # type: ignore[return-value]


def parse_ratios(value: Any) -> tuple[float, float, float]:
    """``"4:2:1"`` (or three numbers): extent along major : semi-major : minor axis."""
    if isinstance(value, str):
        parts: list[Any] = [p for p in value.replace(",", ":").replace(" ", ":").split(":") if p]
    else:
        try:
            parts = list(value)
        except TypeError:
            parts = []
    try:
        ratios = tuple(float(p) for p in parts)
    except (TypeError, ValueError):
        ratios = ()
    if len(ratios) != 3 or not all(np.isfinite(r) and r > 0 for r in ratios):
        raise ValueError(
            "ratios need three positive numbers major:semi-major:minor "
            "(along strike : down dip : across), for example 4:2:1"
        )
    return ratios  # type: ignore[return-value]


def axis_trend_plunge(vector: Any) -> tuple[float, float]:
    """(azimuth, plunge) of an axis, taken at its downward end; plunge is >= 0.

    A horizontal axis is reported with the azimuth in ``[0, 180)``.
    """
    v = np.asarray(vector, dtype=np.float64)
    v = v / np.linalg.norm(v)
    eps = 1e-12
    flat = abs(v[2]) <= eps
    if v[2] > eps or (flat and (v[0] < -eps or (abs(v[0]) <= eps and v[1] < 0))):
        v = -v
    plunge = float(np.degrees(np.arcsin(np.clip(-v[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(v[0], v[1])) % 360.0) if plunge < 90.0 - 1e-9 else 0.0
    return round(azimuth, 6) % 360.0, round(plunge, 6) + 0.0


def describe_axes(ellipsoid: Ellipsoid | None) -> dict[str, Any] | None:
    """JSON-ready description of an ellipsoid's orientation for result panels."""
    if ellipsoid is None:
        return None
    out: dict[str, Any] = {
        "ratios": [
            1.0,
            round(1.0 / ellipsoid.semi_major_ratio, 6),
            round(1.0 / ellipsoid.minor_ratio, 6),
        ],
        "angles": list(angles_from_rotation(ellipsoid.rotation)),
    }
    for name, axis in (
        ("major", ellipsoid.major_axis),
        ("semi_major", ellipsoid.semi_major_axis),
        ("minor", ellipsoid.minor_axis),
    ):
        azimuth, plunge = axis_trend_plunge(axis)
        out[name] = {"azimuth": azimuth, "plunge": plunge}
    # the plane the minor axis is the pole of
    pole_azimuth, pole_plunge = out["minor"]["azimuth"], out["minor"]["plunge"]
    out["plane"] = {
        "dip": round(90.0 - pole_plunge, 6),
        "dip_azimuth": round((pole_azimuth + 180.0) % 360.0, 6),
    }
    return out
