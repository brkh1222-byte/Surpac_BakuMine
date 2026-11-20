"""Dynamic anisotropy: a per-block search ellipsoid and variogram orientation
(research.md 7.5, last paragraph; plan.md P4.T10).

A fixed ellipsoid cannot follow a folded or curved orebody: on one limb of a
fold the continuity dips east, on the other west. With dynamic anisotropy the
three orientation angles (and optionally the two anisotropy ratios) are read
**per block** from block-model attributes, so each block is estimated with an
ellipsoid aligned to the local geometry.

What changes per block
----------------------
For a pass with ``dynamic=DynamicAnisotropy(bearing="ang_b", plunge="ang_p",
dip="ang_d")`` every block ``B`` is estimated exactly as a fixed-ellipsoid
pass would estimate it, except that

* the **search ellipsoid** is the pass ellipsoid (same major radius, same
  ``radius_factor``) with the block's angles (in ``convention``, default
  ``surpac_zxy_lrl`` of :mod:`bakumine.core.estimation.rotation`) and, if
  ratio attributes are named, the block's semi-major and minor ratios;
* for kriging, **every structure of the variogram** gets the block's angles
  (its own ranges and sills are kept; per-block ratios change only the
  search, not the variogram ranges). The orientation is *replaced*, not
  composed with the structure's own rotation.
* for IDW the anisotropic distances (and so the weights) are those of the
  block's ellipsoid; for nearest neighbour the "nearest" sample is nearest in
  the block's anisotropic metric.

**Kriging choice (stated, as is common practice):** all covariances of the
system of block ``B`` - sample-to-sample *and* sample-to-block - use block
``B``'s local anisotropy. A sample's own location's orientation is not used.
This keeps each system a valid (positive-definite) kriging system of a
stationary model, which a mixture of per-sample anisotropies would not be;
the cost is that two neighbouring blocks see slightly different covariances
between the same pair of samples.

Blocks without an orientation
-----------------------------
If any of a block's dynamic attributes is missing (NaN), ``missing``
decides: ``"fixed"`` (default) estimates that block with the pass's own fixed
ellipsoid and variogram; ``"skip"`` leaves it unestimated by this pass (a
later pass may still estimate it).

Parent support
--------------
With parent support (sub-blocked models) a parent is estimated once, with
the attributes of its first targeted sub-block (lowest row). Sub-blocks of
one parent normally share the orientation, which is interpolated at a
scale far coarser than the parent block.

Implementation and cost
-----------------------
Blocks are grouped by their exact parameter tuple; each group is one call
of the ordinary array estimator (:func:`~bakumine.core.estimation.idw.inverse_distance`,
:func:`~bakumine.core.estimation.kriging.krige`, ...) with that group's
ellipsoid and variogram. So a constant orientation reproduces the
fixed-ellipsoid result bit for bit, and the numerics are the tested ones.
Each group builds its own KD-tree, so the cost grows with the number of
*distinct* orientations; ``angle_step`` (degrees) rounds the angles to a
grid to bound it (``None`` = exact angles).

Orientation fields
------------------
:func:`interpolate_orientations` gives per-block angles from a scattered set
of oriented points (e.g. measured on trend strings or a surface): it averages
the rotation matrices (not the raw angles, which wrap at 360 and are
ambiguous), with inverse-distance weights, after aligning each matrix to the
nearest one modulo the ellipsoid's 180-degree symmetries, then projects the
mean to the nearest rotation (SVD) and converts back to angles with
:func:`angles_from_matrix`.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from bakumine.core.estimation.rotation import (
    CONVENTIONS,
    DEFAULT_CONVENTION,
    Ellipsoid,
    rotation_matrix,
)
from bakumine.core.estimation.variogram import Structure, VariogramModel

__all__ = [
    "MISSING_MODES",
    "DynamicAnisotropy",
    "angles_from_matrix",
    "interpolate_orientations",
    "local_ellipsoid",
    "local_variogram",
    "run_dynamic",
]

#: what to do with a block whose dynamic attributes are missing
MISSING_MODES = ("fixed", "skip")

_FIELDS = ("bearing", "plunge", "dip", "semi_major_ratio", "minor_ratio")


@dataclass(frozen=True)
class DynamicAnisotropy:
    """Names of the block attributes that orient each block's ellipsoid.

    Parameters
    ----------
    bearing, plunge, dip : block attribute names holding the three angles
        (degrees) in ``convention`` ((bearing, plunge, dip) for the default
        ``surpac_zxy_lrl``; (ang1, ang2, ang3) for ``gslib``). An angle left
        ``None`` takes the pass ellipsoid's own value for every block (which
        needs that ellipsoid to be given by angles in the same convention).
        At least one angle attribute is required.
    semi_major_ratio, minor_ratio : optional attribute names of per-block
        ratios (major / semi-major, major / minor; > 0). ``None`` keeps the
        pass ellipsoid's ratio. They affect the search ellipsoid (and IDW
        distances) only, not the variogram ranges.
    convention : rotation convention of the angle attributes.
    missing : ``"fixed"`` or ``"skip"``; see the module docstring.
    angle_step : optional rounding step of the angles in degrees (cost
        control; ``None`` = exact).
    """

    bearing: str | None = None
    plunge: str | None = None
    dip: str | None = None
    semi_major_ratio: str | None = None
    minor_ratio: str | None = None
    convention: str = DEFAULT_CONVENTION
    missing: str = "fixed"
    angle_step: float | None = None

    def __post_init__(self) -> None:
        for name in _FIELDS:
            value = getattr(self, name)
            if value is not None and (not isinstance(value, str) or not value):
                raise ValueError(f"{name} must be an attribute name or None")
        if self.bearing is None and self.plunge is None and self.dip is None:
            raise ValueError("dynamic anisotropy needs at least one angle attribute")
        if self.convention not in CONVENTIONS:
            raise ValueError(
                f"unknown rotation convention {self.convention!r}; available: {sorted(CONVENTIONS)}"
            )
        if self.missing not in MISSING_MODES:
            raise ValueError(f"missing must be one of {MISSING_MODES}, got {self.missing!r}")
        if self.angle_step is not None:
            step = float(self.angle_step)
            if not (math.isfinite(step) and step > 0.0):
                raise ValueError("angle_step must be a positive number of degrees")
            object.__setattr__(self, "angle_step", step)

    # ------------------------------------------------------------- attributes

    @property
    def attributes(self) -> list[str]:
        """The block attribute names used."""
        return [getattr(self, f) for f in _FIELDS if getattr(self, f) is not None]

    def check_model(self, model: Any) -> None:
        """Raise if an attribute is absent from ``model`` or not numeric."""
        for name in self.attributes:
            if name not in model:
                raise ValueError(f"dynamic anisotropy: block attribute {name!r} does not exist")
            kind = model.attribute(name).kind
            if kind not in ("float", "int"):
                raise ValueError(
                    f"dynamic anisotropy: attribute {name!r} is {kind}; it must be numeric"
                )

    def block_parameters(self, model: Any, rows: np.ndarray, ellipsoid: Ellipsoid) -> np.ndarray:
        """(m, 5) float64 ``(angle1, angle2, angle3, semi_major_ratio,
        minor_ratio)`` of the blocks ``rows``; a row with any NaN is missing.

        Angles without an attribute and ratios without an attribute come
        from ``ellipsoid``.
        """
        rows = np.asarray(rows, dtype=np.int64)
        m = len(rows)
        out = np.empty((m, 5), dtype=np.float64)
        need_angles = any(getattr(self, f) is None for f in _FIELDS[:3])
        if need_angles and (ellipsoid.axes is not None or ellipsoid.convention != self.convention):
            raise ValueError(
                "dynamic anisotropy: an angle without an attribute is taken from the pass "
                "ellipsoid, which must then be given by angles in the same convention "
                f"({self.convention!r})"
            )
        fixed = (*ellipsoid.angles, ellipsoid.semi_major_ratio, ellipsoid.minor_ratio)
        for j, name in enumerate(_FIELDS):
            attr = getattr(self, name)
            if attr is None:
                out[:, j] = fixed[j]
            else:
                out[:, j] = np.asarray(model.get(attr, rows), dtype=np.float64)
        ratios = out[:, 3:]
        bad = np.isfinite(ratios) & ~(ratios > 0.0)
        if bad.any():
            raise ValueError("dynamic anisotropy: per-block ratios must be > 0")
        out[~np.isfinite(out)] = np.nan
        if self.angle_step is not None:
            out[:, :3] = np.round(out[:, :3] / self.angle_step) * self.angle_step
        return out

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {f: getattr(self, f) for f in _FIELDS if getattr(self, f)}
        out["convention"] = self.convention
        out["missing"] = self.missing
        if self.angle_step is not None:
            out["angle_step"] = self.angle_step
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> DynamicAnisotropy:
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ValueError(
                f"unknown dynamic-anisotropy field(s) {unknown}; known: {sorted(known)}"
            )
        return cls(**dict(data))


# --------------------------------------------------------------------------
# local ellipsoid / variogram
# --------------------------------------------------------------------------


def local_ellipsoid(
    ellipsoid: Ellipsoid, parameters: Any, convention: str = DEFAULT_CONVENTION
) -> Ellipsoid:
    """The pass ellipsoid re-oriented (and re-shaped) for one block.

    ``parameters`` = ``(angle1, angle2, angle3, semi_major_ratio,
    minor_ratio)``; the major radius is kept.
    """
    a1, a2, a3, semi, minor = (float(v) for v in parameters)
    return Ellipsoid(ellipsoid.major_radius, semi, minor, (a1, a2, a3), convention)


def local_variogram(
    model: VariogramModel, angles: Any, convention: str = DEFAULT_CONVENTION
) -> VariogramModel:
    """``model`` with every structure oriented by ``angles`` (ranges, sills,
    nugget unchanged)."""
    a = tuple(float(v) for v in angles)
    structures = tuple(Structure(s.type, s.sill, s.ranges, a, convention) for s in model.structures)
    return VariogramModel(model.nugget, structures)


# --------------------------------------------------------------------------
# grouped estimation
# --------------------------------------------------------------------------


def _fill_for(array: np.ndarray) -> Any:
    if array.dtype == np.bool_:
        return False
    if np.issubdtype(array.dtype, np.integer):
        return 0
    return np.nan


def run_dynamic(
    estimation_pass: Any,
    parameters: np.ndarray,
    run: Callable[[Any, np.ndarray], dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Estimate targets with per-target ellipsoid and variogram orientation.

    Parameters
    ----------
    estimation_pass : an ``EstimationPass`` (or any dataclass with fields
        ``ellipsoid``, ``variogram`` and ``dynamic``) whose ``dynamic`` is a
        :class:`DynamicAnisotropy`.
    parameters : (m, 5) from :meth:`DynamicAnisotropy.block_parameters`.
    run : ``run(fixed_pass, idx)`` estimates targets ``idx`` (int array into
        the m targets) with the fixed pass ``fixed_pass`` and returns a dict
        of per-target arrays of length ``len(idx)``.

    Returns
    -------
    dict of (m,) arrays, as ``run`` returns, with ``"ok"`` false (and NaN /
    0 elsewhere) for skipped targets.
    """
    dyn: DynamicAnisotropy = estimation_pass.dynamic
    params = np.asarray(parameters, dtype=np.float64)
    m = params.shape[0]
    base = dataclasses.replace(estimation_pass, dynamic=None)
    groups: list[tuple[Any, np.ndarray]] = []
    missing = np.isnan(params).any(axis=1)
    if missing.any() and dyn.missing == "fixed":
        groups.append((base, np.flatnonzero(missing)))
    have = np.flatnonzero(~missing)
    if have.size:
        unique, inverse = np.unique(params[have], axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        order = np.argsort(inverse, kind="stable")
        bounds = np.searchsorted(inverse[order], np.arange(len(unique) + 1))
        for g, row in enumerate(unique):
            idx = have[order[bounds[g] : bounds[g + 1]]]
            changes: dict[str, Any] = {
                "ellipsoid": local_ellipsoid(base.ellipsoid, row, dyn.convention)
            }
            if base.variogram is not None:
                changes["variogram"] = local_variogram(base.variogram, row[:3], dyn.convention)
            groups.append((dataclasses.replace(base, **changes), idx))

    out: dict[str, np.ndarray] = {}
    for fixed_pass, idx in groups:
        res = run(fixed_pass, idx)
        for key, values in res.items():
            values = np.asarray(values)
            if key not in out:
                out[key] = np.full(m, _fill_for(values), dtype=values.dtype)
            out[key][idx] = values
    if "ok" not in out:  # every target skipped
        probe = run(base, np.zeros(0, dtype=np.int64))
        for key, values in probe.items():
            values = np.asarray(values)
            out[key] = np.full(m, _fill_for(values), dtype=values.dtype)
    out["ok"] = out["ok"].astype(bool)
    return out


# --------------------------------------------------------------------------
# angles <-> matrices, orientation interpolation
# --------------------------------------------------------------------------


def angles_from_matrix(matrix: Any, convention: str = DEFAULT_CONVENTION) -> np.ndarray:
    """Angles (degrees) of rotation matrices in a named convention.

    Inverse of :func:`~bakumine.core.estimation.rotation.rotation_matrix`
    for the ``"zxy"`` conventions whose first two rotations are left- then
    right-handed (``surpac_zxy_lrl`` and ``gslib``). ``matrix`` is (3, 3) or
    (..., 3, 3); returns (3,) or (..., 3). The second angle is in
    [-90, 90]; at +-90 (gimbal lock) the first angle is set to 0.

    Derivation (``surpac_zxy_lrl``, ``R = Rz(-b) Rx(p) Ry(-d)``): column 1 is
    the major axis ``(sin b cos p, cos b cos p, sin p)`` and row 2 is
    ``(cos p sin d, sin p, cos p cos d)``. ``gslib`` differs only in the sign
    of the third angle.
    """
    try:
        axes, hands = CONVENTIONS[convention]
    except KeyError:
        raise ValueError(f"unknown rotation convention {convention!r}") from None
    if axes != "zxy" or hands[:2] != "lr":
        raise NotImplementedError(f"angles_from_matrix does not support {convention!r}")
    r = np.asarray(matrix, dtype=np.float64)
    if r.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape (..., 3, 3)")
    sin_p = np.clip(r[..., 2, 1], -1.0, 1.0)
    p = np.arcsin(sin_p)
    cos_p = np.cos(p)
    lock = cos_p < 1e-12
    b = np.where(lock, 0.0, np.arctan2(r[..., 0, 1], r[..., 1, 1]))
    d_free = np.arctan2(r[..., 2, 0], r[..., 2, 2])
    # gimbal lock with b = 0: R = Rx(p) Ry(-d), column 0 = (cos d, -sin p sin d, cos p sin d)
    d_lock = np.arctan2(-r[..., 1, 0] * np.sign(sin_p), r[..., 0, 0])
    d = np.where(lock, d_lock, d_free)
    if hands[2] == "r":
        d = -d
    return np.degrees(np.stack([b, p, d], axis=-1))


# 180-degree turns about each local axis: the ellipsoid is unchanged by them
_SYMMETRIES = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], dtype=np.float64)


def interpolate_orientations(
    points: Any,
    angles: Any,
    targets: Any,
    *,
    convention: str = DEFAULT_CONVENTION,
    power: float = 2.0,
    max_points: int = 8,
) -> np.ndarray:
    """Per-target angles interpolated from oriented points.

    Parameters
    ----------
    points : (n, 3) world coordinates of the orientation data.
    angles : (n, 3) angles in degrees at those points (``convention``).
    targets : (m, 3) world coordinates (e.g. block centroids).
    power : inverse-distance exponent (Euclidean distance).
    max_points : nearest orientation points used per target.

    Returns
    -------
    (m, 3) angles in ``convention``. The angles describe the interpolated
    *ellipsoid*; they may differ from the input angles by an equivalent
    representation (e.g. bearing + 180 with the plunge negated).

    Method: rotation matrices (not raw angles) are averaged with weights
    ``1 / d^power``; each neighbour is first turned by whichever of the
    ellipsoid's four 180-degree symmetries brings it closest to the nearest
    neighbour's matrix; the weighted mean matrix is projected onto the
    nearest rotation (SVD). A target that coincides with a data point takes
    that point's orientation.
    """
    pts = np.asarray(points, dtype=np.float64)
    ang = np.asarray(angles, dtype=np.float64)
    tg = np.asarray(targets, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3 or ang.shape != pts.shape:
        raise ValueError("points and angles must both have shape (n, 3)")
    if tg.ndim != 2 or tg.shape[1] != 3:
        raise ValueError("targets must have shape (m, 3)")
    if len(pts) == 0:
        raise ValueError("at least one orientation point is needed")
    if not (np.isfinite(pts).all() and np.isfinite(ang).all() and np.isfinite(tg).all()):
        raise ValueError("points, angles and targets must be finite")
    if not (math.isfinite(power) and power > 0.0):
        raise ValueError("power must be positive")
    rot = np.stack([rotation_matrix(*a, convention=convention) for a in ang])
    k = int(min(max(1, max_points), len(pts)))
    dist, idx = cKDTree(pts).query(tg, k=k)
    dist = np.asarray(dist, dtype=np.float64).reshape(len(tg), k)
    idx = np.asarray(idx, dtype=np.int64).reshape(len(tg), k)
    q = rot[idx]  # (m, k, 3, 3)
    ref = q[:, 0]  # nearest neighbour
    # column-wise dot products with the reference: (m, k, 3)
    dots = np.einsum("mij,mkij->mkj", ref, q)
    scores = dots @ _SYMMETRIES.T  # (m, k, 4) = trace(ref' q S)
    best = _SYMMETRIES[np.argmax(scores, axis=-1)]  # (m, k, 3)
    q = q * best[:, :, None, :]
    with np.errstate(divide="ignore"):
        w = 1.0 / dist**power
    exact = dist[:, 0] == 0.0
    w[exact] = 0.0
    w[exact, 0] = 1.0
    mean = np.einsum("mk,mkij->mij", w, q)
    u, _, vt = np.linalg.svd(mean)
    det = np.linalg.det(u @ vt)
    u[:, :, 2] *= det[:, None]
    result = u @ vt
    return angles_from_matrix(result, convention)
