"""Variogram models: nugget + nested anisotropic structures (research.md 7.4).

Model
-----
::

    gamma(h) = 0                                   for h = 0
    gamma(h) = nugget + sum_k sill_k * g_k(r_k)    for h != 0

where ``r_k`` is the *reduced* lag of structure ``k``: the lag vector is
rotated into that structure's axes and each component divided by the range
along its axis, so ``r = 1`` on the surface of the range ellipsoid::

    r = sqrt( (h_major/a_major)^2 + (h_semi/a_semi)^2 + (h_minor/a_minor)^2 )

and ``g`` is one of (practical-range forms: ``g(1) = 1`` for spherical,
``g(1) = 1 - exp(-3) ~ 0.95`` for the other two)::

    spherical     g(r) = 1.5 r - 0.5 r^3   for r <= 1,   1 beyond
    exponential   g(r) = 1 - exp(-3 r)
    gaussian      g(r) = 1 - exp(-3 r^2)

Covariance form: ``C(h) = total_sill - gamma(h)`` with
``total_sill = nugget + sum_k sill_k``; so ``C(0) = total_sill`` and the
nugget drops out for any non-zero lag. "Zero lag" means all three components
are exactly 0.0; no tolerance is applied.

Orientation of each structure uses the conventions of
:mod:`bakumine.core.estimation.rotation` (default ``surpac_zxy_lrl``; see the
uncertainty note there) or an explicit axes matrix.

JSON schema (``VariogramModel.to_dict`` / ``to_json``)
------------------------------------------------------
::

    {
      "schema": "bakumine.variogram/1",
      "nugget": 0.1,                       # >= 0
      "structures": [                      # zero or more, summed
        {
          "type": "spherical",             # | "exponential" | "gaussian"
          "sill": 0.9,                     # >= 0, partial sill of this structure
          "ranges": [120.0, 60.0, 20.0],   # > 0, metres: major, semi-major, minor
          "rotation": {                    # optional, default no rotation
            "convention": "surpac_zxy_lrl",
            "angles": [30.0, -20.0, 10.0]  # degrees, meaning set by convention
          }
          # or, bypassing conventions:
          # "rotation": {"axes": [[..3..], [..3..], [..3..]]}
          #   3x3 rotation matrix, columns = world (east, north, up)
          #   directions of the semi-major, major and minor axes
        }
      ]
    }

Compiled use
------------
:meth:`VariogramModel.as_arrays` flattens a model to plain arrays, and
:func:`lag_covariance` / :func:`point_covariance` (numba ``@njit``) evaluate
the covariance from them inside compiled kriging loops.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import cached_property
from typing import Any, NamedTuple

import numpy as np
from numba import njit

from bakumine.core.estimation.rotation import DEFAULT_CONVENTION, Ellipsoid

__all__ = [
    "GAUSSIAN",
    "EXPONENTIAL",
    "SCHEMA",
    "SPHERICAL",
    "STRUCTURE_TYPES",
    "Structure",
    "VariogramArrays",
    "VariogramModel",
    "lag_covariance",
    "point_covariance",
]

SCHEMA = "bakumine.variogram/1"

#: integer codes used in the plain-array representation
SPHERICAL = 0
EXPONENTIAL = 1
GAUSSIAN = 2
STRUCTURE_TYPES: dict[str, int] = {
    "spherical": SPHERICAL,
    "exponential": EXPONENTIAL,
    "gaussian": GAUSSIAN,
}


def _unit_variogram(code: int, r: np.ndarray) -> np.ndarray:
    """g(r) of the module docstring for reduced lag ``r`` (numpy)."""
    if code == SPHERICAL:
        rc = np.minimum(r, 1.0)
        return 1.5 * rc - 0.5 * rc**3
    if code == EXPONENTIAL:
        return 1.0 - np.exp(-3.0 * r)
    return 1.0 - np.exp(-3.0 * r * r)


@dataclass(frozen=True)
class Structure:
    """One nested structure of a variogram model.

    Parameters
    ----------
    type : ``"spherical"``, ``"exponential"`` or ``"gaussian"``.
    sill : partial sill contributed by this structure (>= 0).
    ranges : (major, semi-major, minor) ranges in metres (> 0); a single
        number means an isotropic structure. For exponential and gaussian
        these are practical ranges (95 % of the sill).
    angles : three rotation angles in degrees, per ``convention``
        ((bearing, plunge, dip) for the default).
    convention : rotation convention name, see ``rotation.CONVENTIONS``.
    axes : optional explicit 3x3 rotation matrix (columns = world directions
        of semi-major, major, minor axes); overrides angles/convention.
    """

    type: str
    sill: float
    ranges: tuple[float, float, float]
    angles: tuple[float, float, float] = (0.0, 0.0, 0.0)
    convention: str = DEFAULT_CONVENTION
    axes: tuple[tuple[float, float, float], ...] | None = None

    def __post_init__(self) -> None:
        kind = str(self.type).lower()
        if kind not in STRUCTURE_TYPES:
            raise ValueError(
                f"unknown structure type {self.type!r}; available: {sorted(STRUCTURE_TYPES)}"
            )
        object.__setattr__(self, "type", kind)
        sill = float(self.sill)
        if not np.isfinite(sill) or sill < 0.0:
            raise ValueError(f"sill must be a non-negative finite number, got {self.sill}")
        object.__setattr__(self, "sill", sill)
        if np.ndim(self.ranges) == 0:
            ranges = (float(self.ranges),) * 3
        else:
            ranges = tuple(float(a) for a in self.ranges)
        if len(ranges) != 3 or not all(np.isfinite(a) and a > 0.0 for a in ranges):
            raise ValueError(f"ranges must be three positive finite numbers, got {self.ranges}")
        object.__setattr__(self, "ranges", ranges)
        # Ellipsoid validates and normalises angles / convention / axes.
        ell = Ellipsoid.from_radii(ranges, self.angles, self.convention, self.axes)
        object.__setattr__(self, "angles", ell.angles)
        object.__setattr__(self, "axes", ell.axes)

    @classmethod
    def from_ellipsoid(cls, type: str, sill: float, ellipsoid: Ellipsoid) -> Structure:
        """Structure whose range ellipsoid is ``ellipsoid`` (its major radius
        is the major range, its ratios give the other two)."""
        return cls(
            type, sill, ellipsoid.radii, ellipsoid.angles, ellipsoid.convention, ellipsoid.axes
        )

    @cached_property
    def ellipsoid(self) -> Ellipsoid:
        """Range ellipsoid: radii = ranges, orientation of this structure."""
        return Ellipsoid.from_radii(self.ranges, self.angles, self.convention, self.axes)

    @cached_property
    def transform(self) -> np.ndarray:
        """(3, 3) matrix ``T``; ``|T @ h|`` is the reduced lag of world lag
        vector ``h`` (1.0 on the range ellipsoid)."""
        t = self.ellipsoid.isotropic_matrix / self.ranges[0]
        t.setflags(write=False)
        return t

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "sill": self.sill, "ranges": list(self.ranges)}
        if self.axes is not None:
            out["rotation"] = {"axes": [list(row) for row in self.axes]}
        else:
            out["rotation"] = {"convention": self.convention, "angles": list(self.angles)}
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Structure:
        for key in ("type", "sill", "ranges"):
            if key not in data:
                raise ValueError(f"variogram structure is missing {key!r}")
        rot = data.get("rotation") or {}
        axes = rot.get("axes")
        return cls(
            type=data["type"],
            sill=data["sill"],
            ranges=data["ranges"],
            angles=tuple(rot.get("angles", (0.0, 0.0, 0.0))),
            convention=rot.get("convention", DEFAULT_CONVENTION),
            axes=None if axes is None else tuple(tuple(row) for row in axes),
        )


class VariogramArrays(NamedTuple):
    """Plain-array form of a model for ``@njit`` code.

    nugget : float
    types : (k,) int64, codes SPHERICAL=0, EXPONENTIAL=1, GAUSSIAN=2
    sills : (k,) float64, partial sills
    transforms : (k, 3, 3) float64 C-contiguous; ``transforms[k] @ h`` is the
        reduced lag vector of structure ``k`` (length 1 at the range)
    """

    nugget: float
    types: np.ndarray
    sills: np.ndarray
    transforms: np.ndarray


@dataclass(frozen=True)
class VariogramModel:
    """Nugget plus nested structures.

    Parameters
    ----------
    nugget : nugget variance (>= 0).
    structures : sequence of :class:`Structure` (may be empty: pure nugget).
    """

    nugget: float = 0.0
    structures: tuple[Structure, ...] = ()

    def __post_init__(self) -> None:
        nugget = float(self.nugget)
        if not np.isfinite(nugget) or nugget < 0.0:
            raise ValueError(f"nugget must be a non-negative finite number, got {self.nugget}")
        object.__setattr__(self, "nugget", nugget)
        structures = tuple(self.structures)
        if not all(isinstance(s, Structure) for s in structures):
            raise TypeError("structures must be Structure instances")
        object.__setattr__(self, "structures", structures)

    # ------------------------------------------------------------- properties
    @property
    def total_sill(self) -> float:
        """nugget + sum of structure sills = C(0)."""
        return self.nugget + sum(s.sill for s in self.structures)

    # ------------------------------------------------------------- evaluation
    def gamma(self, h_vectors: Any) -> np.ndarray:
        """Variogram value for world lag vectors.

        Parameters
        ----------
        h_vectors : (..., 3) array of lag vectors (east, north, up), metres.

        Returns
        -------
        (...) float64. Exactly 0 at the zero vector, nugget included elsewhere.
        """
        h = np.asarray(h_vectors, dtype=np.float64)
        if h.shape[-1] != 3:
            raise ValueError("h_vectors must have shape (..., 3)")
        out = np.full(h.shape[:-1], self.nugget, dtype=np.float64)
        for s in self.structures:
            red = h @ s.transform.T
            r = np.sqrt(np.einsum("...i,...i->...", red, red))
            out += s.sill * _unit_variogram(STRUCTURE_TYPES[s.type], r)
        zero = ~h.any(axis=-1)
        return np.where(zero, 0.0, out)

    def covariance(self, h_vectors: Any) -> np.ndarray:
        """Covariance ``C(h) = total_sill - gamma(h)`` for lag vectors
        ``(..., 3)``; ``C(0) = total_sill``."""
        return self.total_sill - self.gamma(h_vectors)

    def _pairwise_lags(self, a: Any, b: Any) -> np.ndarray:
        a = np.asarray(a, dtype=np.float64)
        b = a if b is None else np.asarray(b, dtype=np.float64)
        if a.ndim != 2 or a.shape[1] != 3 or b.ndim != 2 or b.shape[1] != 3:
            raise ValueError("point sets must have shape (n, 3)")
        return a[:, None, :] - b[None, :, :]

    def gamma_matrix(self, a: Any, b: Any = None) -> np.ndarray:
        """Variogram matrix between point sets.

        ``a`` is (n, 3), ``b`` is (m, 3) (default ``a`` itself); returns
        (n, m) with element ``[i, j] = gamma(a[i] - b[j])``. Memory is
        O(n * m); call per neighbourhood, not for whole data sets.
        """
        return self.gamma(self._pairwise_lags(a, b))

    def covariance_matrix(self, a: Any, b: Any = None) -> np.ndarray:
        """Covariance matrix (n, m) between point sets, ``[i, j] =
        C(a[i] - b[j])``: sample-sample with ``b=None``, sample-point with
        ``b`` the target (or block discretisation) points."""
        return self.covariance(self._pairwise_lags(a, b))

    # --------------------------------------------------------- compiled form
    def as_arrays(self) -> VariogramArrays:
        """Plain arrays for :func:`lag_covariance` / :func:`point_covariance`."""
        k = len(self.structures)
        types = np.array([STRUCTURE_TYPES[s.type] for s in self.structures], dtype=np.int64)
        sills = np.array([s.sill for s in self.structures], dtype=np.float64)
        transforms = np.zeros((k, 3, 3), dtype=np.float64)
        for i, s in enumerate(self.structures):
            transforms[i] = s.transform
        return VariogramArrays(self.nugget, types, sills, transforms)

    # ---------------------------------------------------------- serialisation
    def to_dict(self) -> dict[str, Any]:
        """Dict following the schema in the module docstring."""
        return {
            "schema": SCHEMA,
            "nugget": self.nugget,
            "structures": [s.to_dict() for s in self.structures],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> VariogramModel:
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported variogram schema {schema!r}, expected {SCHEMA!r}")
        return cls(
            nugget=data.get("nugget", 0.0),
            structures=tuple(Structure.from_dict(s) for s in data.get("structures", ())),
        )

    def to_json(self, **kwargs: Any) -> str:
        """JSON text (``kwargs`` go to ``json.dumps``)."""
        return json.dumps(self.to_dict(), **kwargs)

    @classmethod
    def from_json(cls, text: str) -> VariogramModel:
        return cls.from_dict(json.loads(text))


@njit(cache=True)
def lag_covariance(dx, dy, dz, nugget, types, sills, transforms):
    """Covariance for one lag vector ``(dx, dy, dz)`` (world east, north, up).

    ``nugget, types, sills, transforms`` are the fields of
    :class:`VariogramArrays` (``model.as_arrays()``). Equals
    ``model.covariance([dx, dy, dz])``. numba ``@njit``: callable from other
    compiled functions.
    """
    cov = 0.0
    for k in range(types.shape[0]):
        t = transforms[k]
        u = t[0, 0] * dx + t[0, 1] * dy + t[0, 2] * dz
        v = t[1, 0] * dx + t[1, 1] * dy + t[1, 2] * dz
        w = t[2, 0] * dx + t[2, 1] * dy + t[2, 2] * dz
        r2 = u * u + v * v + w * w
        kind = types[k]
        if kind == 0:  # spherical
            if r2 < 1.0:
                r = np.sqrt(r2)
                cov += sills[k] * (1.0 - 1.5 * r + 0.5 * r * r2)
        elif kind == 1:  # exponential
            cov += sills[k] * np.exp(-3.0 * np.sqrt(r2))
        else:  # gaussian
            cov += sills[k] * np.exp(-3.0 * r2)
    if dx == 0.0 and dy == 0.0 and dz == 0.0:
        cov += nugget
    return cov


@njit(cache=True)
def point_covariance(p, q, nugget, types, sills, transforms):
    """Covariance between two points ``p`` and ``q`` (length-3 arrays);
    see :func:`lag_covariance`."""
    return lag_covariance(
        p[0] - q[0], p[1] - q[1], p[2] - q[2], nugget, types, sills, transforms
    )
