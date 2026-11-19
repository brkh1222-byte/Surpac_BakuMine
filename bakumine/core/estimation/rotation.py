"""Rotation conventions and the anisotropy ellipsoid (research.md section 7.5).

Frames
------
World coordinates are ``(x, y, z) = (east, north, up)`` in metres.

An ellipsoid carries a *local frame* ``(x', y', z')`` whose axes are

    ``x'`` = semi-major axis,  ``y'`` = major axis,  ``z'`` = minor axis.

With all angles zero the local frame coincides with the world frame: the
major axis points north, the semi-major axis east and the minor axis up.
(The major axis is ``y'`` and not ``x'`` so that the zero-angle matrix is the
identity *and* a bearing of zero means north.)

Every function here returns a proper rotation matrix ``R`` (orthonormal,
determinant +1) whose **columns are the world directions of the local axes**::

    R[:, 0] = semi-major axis,  R[:, 1] = major axis,  R[:, 2] = minor axis

    world_vector = R @ local_vector          local_vector = R.T @ world_vector

Conventions
-----------
All implemented conventions are *intrinsic* rotation sequences: each rotation
is about an axis of the frame as already turned by the previous rotations.
"Right-hand" (R) means counter-clockwise when looking from the positive end of
the axis towards the origin; "left-hand" (L) is the opposite. Angles are in
degrees.

``"surpac_zxy_lrl"`` (default) - angles ``(bearing, plunge, dip)``:

1. **bearing**: rotate about ``z`` (up), left-hand. Seen from above this is
   clockwise, so the major axis swings from north towards east: bearing is
   the azimuth of the major axis, bearing 90 points it east.
2. **plunge**: rotate about the turned ``x'`` (semi-major) axis, right-hand.
   This lifts the forward (bearing-direction) end of the major axis:
   **positive plunge = major axis points upwards**, a major axis plunging
   20 degrees below horizontal has plunge -20.
3. **dip**: rotate about the turned ``y'`` (major) axis, left-hand. Standing
   behind the ellipsoid and looking along the major axis in the bearing
   direction, the semi-major axis' right-hand end (east for bearing 0) goes
   **up** for positive dip; a semi-major axis dipping down to the right has
   negative dip.

   ``R = Rz(-bearing) @ Rx(+plunge) @ Ry(-dip)`` with the usual right-hand
   elementary matrices, giving

   ``major = (sin b cos p, cos b cos p, sin p)``.

``"gslib"`` - angles ``(ang1, ang2, ang3)`` of Deutsch & Journel, *GSLIB*
(2nd ed., 1998), section II.4 / subroutine ``setrot``: ``ang1`` = azimuth of
the principal direction clockwise from north, ``ang2`` = its dip, negative
downwards, ``ang3`` = rotation about the principal direction. Written in the
same language it is ZXY left, right, right:
``R = Rz(-ang1) @ Rx(+ang2) @ Ry(+ang3)``, i.e. it differs from the Surpac
convention above only in the sign of the third angle. GSLIB's own rotated
frame is ordered (principal, horizontal-minor, vertical-minor) and equals
``(y', -x', z')`` of the local frame used here; the tests check this against
an independent transcription of the published ``setrot`` formulas.

Generic sequences: :func:`rotation_from_sequence` builds any intrinsic
sequence (for example ``"zxz"``, ``"rrr"``); :func:`rotation_from_axes` and
:meth:`Ellipsoid.from_axes` take the axis directions directly and bypass
conventions altogether.

Uncertainty - read this
-----------------------
* **Surpac.** ``surpac_zxy_lrl`` implements the convention *as research.md
  states it* ("rotate about Z, then X, then Y, with left-, right-, left-hand
  rules"), which research.md itself marks **[verify]**. No Surpac manual or
  Surpac output was available. In particular, (a) that the sequence is
  intrinsic, (b) that positive plunge is upwards and (c) the sign of dip are
  consequences of that one-line statement as derived above, not facts checked
  against Surpac. The tests verify internal consistency and hand-computed
  vectors only. Before trusting a model built from Surpac parameters, compare
  one ellipsoid with a Surpac export; if a sign differs, negate that angle or
  use :meth:`Ellipsoid.from_axes`.
* **GSLIB.** Implemented from the published formulas (from memory of the
  book; no GSLIB source was consulted or copied) and cross-checked in the
  tests against a separate transcription of those formulas, not against a
  GSLIB executable.
* **Datamine ZXZ** is deliberately **not** provided as a named convention:
  the handedness of its three rotations and which local axis carries the
  major radius could not be stated with confidence. Use
  :func:`rotation_from_sequence` or :meth:`Ellipsoid.from_axes` once the
  convention has been confirmed from Datamine documentation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Any

import numpy as np

__all__ = [
    "CONVENTIONS",
    "DEFAULT_CONVENTION",
    "Ellipsoid",
    "rotation_from_axes",
    "rotation_from_sequence",
    "rotation_matrix",
]

DEFAULT_CONVENTION = "surpac_zxy_lrl"

#: convention name -> (axis sequence, handedness of each rotation)
CONVENTIONS: dict[str, tuple[str, str]] = {
    "surpac_zxy_lrl": ("zxy", "lrl"),
    "gslib": ("zxy", "lrr"),
}

_ORTHO_TOL = 1e-9


def _elementary(axis: str, angle_rad: float) -> np.ndarray:
    """Right-hand rotation matrix about a coordinate axis."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == "x":
        return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])
    if axis == "y":
        return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])
    if axis == "z":
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    raise ValueError(f"axis must be 'x', 'y' or 'z', got {axis!r}")


def rotation_from_sequence(axes: str, hands: str, angles: tuple[float, float, float]) -> np.ndarray:
    """Rotation matrix of an intrinsic three-rotation sequence.

    Parameters
    ----------
    axes : three letters from ``"xyz"``, e.g. ``"zxy"``: the axis of each
        rotation, in the order applied. Each axis is the one of the frame as
        turned by the preceding rotations (intrinsic).
    hands : three letters, ``"r"`` (right-hand, counter-clockwise seen from
        the positive axis end) or ``"l"`` (left-hand), one per rotation.
    angles : the three angles in degrees.

    Returns
    -------
    (3, 3) float64 matrix ``R``; columns are the world directions of the
    local ``x'``, ``y'``, ``z'`` axes (see the module docstring).
    """
    axes, hands = axes.lower(), hands.lower()
    if len(axes) != 3 or len(hands) != 3 or len(angles) != 3:
        raise ValueError("axes, hands and angles must each have three entries")
    result = np.eye(3)
    for axis, hand, angle in zip(axes, hands, angles, strict=True):
        if hand not in "rl":
            raise ValueError(f"hand must be 'r' or 'l', got {hand!r}")
        sign = 1.0 if hand == "r" else -1.0
        # intrinsic sequence = post-multiplication by each elementary rotation
        result = result @ _elementary(axis, sign * np.deg2rad(float(angle)))
    return result


def rotation_matrix(
    angle1: float, angle2: float, angle3: float, convention: str = DEFAULT_CONVENTION
) -> np.ndarray:
    """Rotation matrix for three angles (degrees) in a named convention.

    For ``"surpac_zxy_lrl"`` the angles are (bearing, plunge, dip); for
    ``"gslib"`` they are (ang1 azimuth, ang2 dip, ang3 rake). See the module
    docstring for the exact meaning and signs.

    Returns the (3, 3) matrix whose columns are the world (east, north, up)
    directions of the semi-major, major and minor axes, in that order.
    """
    try:
        axes, hands = CONVENTIONS[convention]
    except KeyError:
        raise ValueError(
            f"unknown rotation convention {convention!r}; available: {sorted(CONVENTIONS)}"
        ) from None
    return rotation_from_sequence(axes, hands, (angle1, angle2, angle3))


def _check_rotation(matrix: Any) -> np.ndarray:
    r = np.array(matrix, dtype=np.float64)
    if r.shape != (3, 3):
        raise ValueError(f"rotation matrix must be 3x3, got shape {r.shape}")
    if not np.all(np.isfinite(r)):
        raise ValueError("rotation matrix contains non-finite values")
    if not np.allclose(r.T @ r, np.eye(3), atol=_ORTHO_TOL):
        raise ValueError("rotation matrix is not orthonormal")
    if np.linalg.det(r) < 0:
        raise ValueError("rotation matrix has determinant -1 (a reflection)")
    return r


def rotation_from_axes(major: Any, semi_major: Any, minor: Any = None) -> np.ndarray:
    """Rotation matrix from the axis directions themselves.

    Parameters
    ----------
    major, semi_major : world (east, north, up) direction vectors of the major
        and semi-major axes. Any length; they must be perpendicular.
    minor : optional direction of the minor axis. It must be perpendicular to
        the other two. An ellipsoid axis is a line, not an arrow, so if the
        given ``minor`` would make the frame left-handed it is negated.
        Default ``semi_major x major``.

    Returns
    -------
    (3, 3) matrix with columns (semi-major, major, minor) unit vectors.
    """
    cols = []
    for name, vec in (("major", major), ("semi_major", semi_major)):
        v = np.asarray(vec, dtype=np.float64)
        if v.shape != (3,):
            raise ValueError(f"{name} must be a 3-vector")
        norm = np.linalg.norm(v)
        if not np.isfinite(norm) or norm == 0.0:
            raise ValueError(f"{name} must be a finite non-zero vector")
        cols.append(v / norm)
    y_axis, x_axis = cols
    if abs(float(x_axis @ y_axis)) > _ORTHO_TOL:
        raise ValueError("major and semi_major are not perpendicular")
    z_axis = np.cross(x_axis, y_axis)
    if minor is not None:
        m = np.asarray(minor, dtype=np.float64)
        if m.shape != (3,) or not np.linalg.norm(m) > 0.0:
            raise ValueError("minor must be a non-zero 3-vector")
        m = m / np.linalg.norm(m)
        if abs(abs(float(m @ z_axis)) - 1.0) > _ORTHO_TOL:
            raise ValueError("minor is not perpendicular to major and semi_major")
    return _check_rotation(np.column_stack([x_axis, y_axis, z_axis]))


@dataclass(frozen=True)
class Ellipsoid:
    """Anisotropy / search ellipsoid: orientation, major radius and two ratios.

    Parameters
    ----------
    major_radius : length of the major semi-axis in metres (search distance,
        or variogram range, along the major axis). Must be > 0.
    semi_major_ratio : major radius / semi-major radius (> 0, normally >= 1).
    minor_ratio : major radius / minor radius (> 0, normally >= 1).
    angles : three angles in degrees, interpreted by ``convention``
        ((bearing, plunge, dip) for the default).
    convention : name in :data:`CONVENTIONS`.
    axes : optional explicit rotation matrix (3x3, columns = world directions
        of semi-major, major, minor axes). When given it overrides ``angles``
        and ``convention``, which are then ignored. Prefer
        :meth:`from_axes` / :meth:`from_matrix` to construct this form.

    The object is immutable and hashable. It does not depend on any block
    model or drillhole class: all methods take plain coordinate arrays.
    """

    major_radius: float = 1.0
    semi_major_ratio: float = 1.0
    minor_ratio: float = 1.0
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    convention: str = DEFAULT_CONVENTION
    axes: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        for name in ("major_radius", "semi_major_ratio", "minor_ratio"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be a positive finite number, got {value}")
            object.__setattr__(self, name, value)
        angles = tuple(float(a) for a in self.angles)
        if len(angles) != 3 or not all(np.isfinite(a) for a in angles):
            raise ValueError("angles must be three finite numbers (degrees)")
        object.__setattr__(self, "angles", angles)
        if self.axes is not None:
            r = _check_rotation(self.axes)
            object.__setattr__(self, "axes", tuple(tuple(float(v) for v in row) for row in r))
        elif self.convention not in CONVENTIONS:
            raise ValueError(
                f"unknown rotation convention {self.convention!r}; "
                f"available: {sorted(CONVENTIONS)}"
            )

    # ---------------------------------------------------------------- builders
    @classmethod
    def from_matrix(
        cls,
        matrix: Any,
        major_radius: float = 1.0,
        semi_major_ratio: float = 1.0,
        minor_ratio: float = 1.0,
    ) -> Ellipsoid:
        """Ellipsoid from a rotation matrix (columns semi-major, major, minor)."""
        r = _check_rotation(matrix)
        return cls(major_radius, semi_major_ratio, minor_ratio, axes=tuple(map(tuple, r)))

    @classmethod
    def from_axes(
        cls,
        major: Any,
        semi_major: Any,
        minor: Any = None,
        major_radius: float = 1.0,
        semi_major_ratio: float = 1.0,
        minor_ratio: float = 1.0,
    ) -> Ellipsoid:
        """Ellipsoid from axis direction vectors; see :func:`rotation_from_axes`."""
        r = rotation_from_axes(major, semi_major, minor)
        return cls(major_radius, semi_major_ratio, minor_ratio, axes=tuple(map(tuple, r)))

    @classmethod
    def from_radii(
        cls,
        radii: tuple[float, float, float],
        angles: tuple[float, float, float] = (0.0, 0.0, 0.0),
        convention: str = DEFAULT_CONVENTION,
        axes: Any = None,
    ) -> Ellipsoid:
        """Ellipsoid from the three radii (major, semi-major, minor) in metres."""
        major, semi, minor = (float(v) for v in radii)
        if not (major > 0.0 and semi > 0.0 and minor > 0.0):
            raise ValueError("radii must be positive")
        return cls(major, major / semi, major / minor, angles, convention, axes)

    # -------------------------------------------------------------- properties
    @cached_property
    def rotation(self) -> np.ndarray:
        """(3, 3) rotation matrix; columns = world directions of the
        semi-major (x'), major (y') and minor (z') axes. Read-only."""
        if self.axes is not None:
            r = np.array(self.axes, dtype=np.float64)
        else:
            r = rotation_matrix(*self.angles, convention=self.convention)
        r.setflags(write=False)
        return r

    @property
    def major_axis(self) -> np.ndarray:
        """Unit world vector (east, north, up) along the major axis."""
        return self.rotation[:, 1].copy()

    @property
    def semi_major_axis(self) -> np.ndarray:
        """Unit world vector along the semi-major axis."""
        return self.rotation[:, 0].copy()

    @property
    def minor_axis(self) -> np.ndarray:
        """Unit world vector along the minor axis."""
        return self.rotation[:, 2].copy()

    @property
    def radii(self) -> tuple[float, float, float]:
        """Radii (major, semi-major, minor) in metres."""
        r = self.major_radius
        return (r, r / self.semi_major_ratio, r / self.minor_ratio)

    @cached_property
    def scale(self) -> np.ndarray:
        """Stretch factors applied to local (x', y', z'):
        ``(semi_major_ratio, 1, minor_ratio)``."""
        s = np.array([self.semi_major_ratio, 1.0, self.minor_ratio])
        s.setflags(write=False)
        return s

    @cached_property
    def isotropic_matrix(self) -> np.ndarray:
        """(3, 3) matrix ``A = diag(scale) @ R.T``: ``A @ world_vector`` is the
        vector in isotropic space; its length is the anisotropic distance."""
        a = self.scale[:, None] * self.rotation.T
        a.setflags(write=False)
        return a

    # ------------------------------------------------------------- transforms
    def to_local(self, points: Any, centre: Any = None) -> np.ndarray:
        """Rotate world points into the local frame without stretching.

        Returns an array of the same shape ``(..., 3)`` with columns
        ``(x', y', z')`` = coordinates along (semi-major, major, minor).
        """
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1] != 3:
            raise ValueError("points must have shape (..., 3)")
        if centre is not None:
            p = p - np.asarray(centre, dtype=np.float64)
        return p @ self.rotation

    def to_isotropic(self, points: Any, centre: Any = None) -> np.ndarray:
        """Rotate world points into the ellipsoid's axes and stretch the
        short axes, so that anisotropic distance becomes Euclidean distance.

        ``out = ((points - centre) @ R) * (semi_major_ratio, 1, minor_ratio)``

        The Euclidean distance between two transformed points equals
        ``sqrt(u^2 + (v r_semi)^2 + (w r_minor)^2)`` of research.md 7.5, with
        ``u, v, w`` the components of their separation along the major,
        semi-major and minor axes. A point is inside the ellipsoid centred at
        ``centre`` when the length of its transformed vector is
        ``<= major_radius``.

        Parameters
        ----------
        points : (..., 3) world coordinates (east, north, up).
        centre : optional (3,) or broadcastable origin; default the world
            origin (the transform is linear, so differences of transformed
            points do not depend on it).

        Returns
        -------
        (..., 3) float64, columns along (semi-major, major, minor) in metres
        of equivalent major-axis distance.
        """
        return self.to_local(points, centre) * self.scale

    def distance(self, a: Any, b: Any) -> np.ndarray:
        """Anisotropic distance between world points ``a`` and ``b``.

        ``a`` and ``b`` have shape ``(..., 3)`` and broadcast against each
        other (``a[:, None, :]`` vs ``b[None, :, :]`` gives a full matrix).
        """
        diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
        iso = self.to_isotropic(diff)
        return np.sqrt(np.einsum("...i,...i->...", iso, iso))

    def contains(self, points: Any, centre: Any = None) -> np.ndarray:
        """True where a point lies inside or on the ellipsoid at ``centre``."""
        iso = self.to_isotropic(points, centre)
        return np.einsum("...i,...i->...", iso, iso) <= self.major_radius**2

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict; inverse of :meth:`from_dict`."""
        out: dict[str, Any] = {
            "major_radius": self.major_radius,
            "semi_major_ratio": self.semi_major_ratio,
            "minor_ratio": self.minor_ratio,
        }
        if self.axes is not None:
            out["axes"] = [list(row) for row in self.axes]
        else:
            out["convention"] = self.convention
            out["angles"] = list(self.angles)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Ellipsoid:
        """Build from the dict produced by :meth:`to_dict`."""
        axes = data.get("axes")
        return cls(
            major_radius=data.get("major_radius", 1.0),
            semi_major_ratio=data.get("semi_major_ratio", 1.0),
            minor_ratio=data.get("minor_ratio", 1.0),
            angles=tuple(data.get("angles", (0.0, 0.0, 0.0))),
            convention=data.get("convention", DEFAULT_CONVENTION),
            axes=None if axes is None else tuple(tuple(row) for row in axes),
        )
