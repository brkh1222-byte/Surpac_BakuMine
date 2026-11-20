"""Experimental variograms and a least-squares seed fit (research.md 7.4).

This is the *minimal* calculator the plan asks for: enough to obtain kriging
parameters without another program. It is not a variogram modelling tool.

Definitions
-----------
For a set of sample pairs ``(i, j)`` assigned to a lag, with values ``z``:

``"semivariogram"`` (traditional)::

    gamma = sum (z_i - z_j)^2 / (2 N)

``"pairwise_relative"`` (robust for skewed, non-negative grades; each
squared difference is scaled by the squared pair mean; a pair whose two
values are both 0 contributes 0)::

    gamma = sum [ (z_i - z_j)^2 / ((z_i + z_j) / 2)^2 ] / (2 N)

``"correlogram"`` (symmetric form: every pair is used both ways round, so
head and tail share one mean ``m`` and one variance ``s^2`` per lag)::

    gamma = 1 - (mean(z_i z_j) - m^2) / s^2          (standardised, sill 1)

``"normal_scores"``: the traditional semivariogram of the normal scores of
the values (:func:`normal_score`, equal weights), sill about 1.

Pair selection (the exact rule the tests check against brute force)
--------------------------------------------------------------------
With ``h = x_j - x_i`` and ``d = |h| = sqrt(dx*dx + dy*dy + dz*dz)``:

* pairs with ``d == 0`` (duplicates) are never used;
* **lags**: lag ``k = 1 .. n_lags`` has centre ``k * lag_size``; the pair
  belongs to it when ``k * lag_size - lag_tolerance <= d < k * lag_size +
  lag_tolerance`` (half-open, so with the default tolerance of half a lag
  every pair falls in exactly one lag; a larger tolerance makes the lags
  overlap and a pair is then counted in each lag it falls in);
* **direction** (unit vector ``u``; a variogram is symmetric, so ``u`` and
  ``-u`` are the same direction): ``|h . u| >= cos(angular_tolerance) * d``.
  A tolerance of 90 degrees or more accepts every pair (omnidirectional);
* **bandwidths**: the offset of ``h`` perpendicular to ``u`` is split along
  two unit vectors ``e_h`` and ``e_v`` (both perpendicular to ``u`` and to
  each other); the pair is kept when ``|h . e_h| <= bandwidth_h`` and
  ``|h . e_v| <= bandwidth_v``. By default ``e_h`` is horizontal
  (``up x u``, normalised; east for a vertical ``u``) and ``e_v = u x e_h``,
  so for a horizontal direction they are the horizontal and vertical
  bandwidths of GSLIB ``gamv``. :func:`principal_directions` uses the other
  two ellipsoid axes instead;
* optional: only pairs within the same hole (down-hole variogram) and / or
  the same domain code.

Direction angles: ``azimuth`` is degrees clockwise from north, ``plunge`` is
degrees **below** horizontal (the geological meaning; note that the
``surpac_zxy_lrl`` *rotation* convention of ``rotation.py`` counts its plunge
angle positive upwards)::

    u = (sin az cos pl, cos az cos pl, -sin pl)          (east, north, up)

Algorithm: the samples are sorted along the coordinate axis of largest
extent and each sample is paired with the following ones until the sorted
coordinate differs by more than the largest lag distance (a sweep), inside a
numba-compiled parallel loop. Memory is O(n + directions * lags); no pair
list or distance matrix is ever stored. Work is at worst O(n^2) pair tests
(about 2e8 for 20 000 samples, well under a second) and less when the data
extend beyond the maximum lag distance.

Sources: Journel & Huijbregts (1978); Isaaks & Srivastava (1989), chapters 4
and 7; Deutsch & Journel (1998), the *GSLIB* book's description of ``gamv``
and ``nscore``. No third-party code was consulted or copied.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import prange
from scipy.optimize import least_squares
from scipy.special import ndtri

from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.variogram import (
    STRUCTURE_TYPES,
    Structure,
    VariogramModel,
    _unit_variogram,
)
from bakumine.core.parallel import parallel_njit

__all__ = [
    "ESTIMATORS",
    "FIT_WEIGHTS",
    "Direction",
    "ExperimentalVariogram",
    "VariogramFit",
    "downhole_variogram",
    "experimental_variogram",
    "fit_variogram",
    "normal_score",
    "principal_directions",
    "principal_variograms",
    "variogram_chart_data",
]

ESTIMATORS = ("semivariogram", "pairwise_relative", "correlogram", "normal_scores")
FIT_WEIGHTS = ("pairs/distance", "pairs/distance^2", "pairs", "equal")

# The sweep is split into this many fixed slabs, each accumulated on its own
# and summed in order afterwards, so the floating-point result does not
# depend on the number of threads.
_N_CHUNKS = 64


# --------------------------------------------------------------------------
# normal scores
# --------------------------------------------------------------------------
def normal_score(values: Any, weights: Any = None) -> np.ndarray:
    """Normal-score (Gaussian anamorphosis by ranks) transform of ``values``.

    The values are ranked; value ``i`` gets the cumulative probability at the
    middle of its own weight, ``p_i = (sum of weights of smaller values +
    w_i / 2) / sum of all weights``, and the score is the standard normal
    quantile of ``p_i``. Tied values all receive the quantile of the mean of
    their probabilities (deterministic, no random tie-breaking), so a data
    set with many equal values does not become exactly Gaussian.

    Parameters
    ----------
    values : (n,) array. NaN stays NaN and takes no part in the ranking.
    weights : optional (n,) non-negative declustering weights (default equal).

    Returns
    -------
    (n,) float64 scores with (weighted) mean about 0 and variance slightly
    below 1 for small ``n``.
    """
    z = np.asarray(values, dtype=np.float64)
    if z.ndim != 1:
        raise ValueError("values must be one-dimensional")
    ok = np.isfinite(z)
    if weights is None:
        w = np.ones(z.shape)
    else:
        w = np.asarray(weights, dtype=np.float64)
        if w.shape != z.shape:
            raise ValueError("weights must have one entry per value")
        if not (np.isfinite(w[ok]).all() and (w[ok] >= 0.0).all()):
            raise ValueError("weights must be finite and non-negative")
    out = np.full(z.shape, np.nan)
    zz, ww = z[ok], w[ok]
    if zz.size == 0:
        return out
    total = ww.sum()
    if not total > 0.0:
        raise ValueError("the weights of the valid values sum to zero")
    order = np.argsort(zz, kind="stable")
    ws = ww[order]
    p_sorted = (np.cumsum(ws) - 0.5 * ws) / total
    p = np.empty(zz.size)
    p[order] = p_sorted
    _, inverse, counts = np.unique(zz, return_inverse=True, return_counts=True)
    inverse = np.asarray(inverse).reshape(-1)
    p = (np.bincount(inverse, weights=p) / counts)[inverse]
    out[ok] = ndtri(p)
    return out


# --------------------------------------------------------------------------
# directions
# --------------------------------------------------------------------------
def _unit(vector: Any, name: str) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    if v.shape != (3,) or not np.isfinite(v).all():
        raise ValueError(f"{name} must be a finite 3-vector")
    norm = float(np.linalg.norm(v))
    if norm == 0.0:
        raise ValueError(f"{name} must not be the zero vector")
    return v / norm


@dataclass(frozen=True)
class Direction:
    """One variogram direction with its tolerances (module docstring).

    Parameters
    ----------
    vector : direction (east, north, up); any length, normalised on
        construction. Irrelevant when ``angular_tolerance >= 90`` and no
        bandwidth is set (omnidirectional).
    angular_tolerance : half-angle of the acceptance cone, degrees.
    bandwidth_h, bandwidth_v : maximum offset (metres) perpendicular to the
        direction along ``e_h`` / ``e_v``; ``None`` = unlimited.
    label : name carried into the output table.
    reference : optional vector fixing ``e_h`` (its part perpendicular to
        ``vector``, normalised). Default: horizontal, see module docstring.
    """

    vector: tuple[float, float, float] = (0.0, 1.0, 0.0)
    angular_tolerance: float = 22.5
    bandwidth_h: float | None = None
    bandwidth_v: float | None = None
    label: str = ""
    reference: tuple[float, float, float] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "vector", tuple(float(c) for c in _unit(self.vector, "vector")))
        tol = float(self.angular_tolerance)
        if not (np.isfinite(tol) and tol > 0.0):
            raise ValueError("angular_tolerance must be a positive number of degrees")
        object.__setattr__(self, "angular_tolerance", tol)
        for name in ("bandwidth_h", "bandwidth_v"):
            value = getattr(self, name)
            if value is not None:
                value = float(value)
                if not (np.isfinite(value) and value >= 0.0):
                    raise ValueError(f"{name} must be >= 0 or None")
                object.__setattr__(self, name, value)
        if self.reference is not None:
            ref = _unit(self.reference, "reference")
            u = np.array(self.vector)
            if np.linalg.norm(ref - (ref @ u) * u) < 1e-9:
                raise ValueError("reference must not be parallel to vector")
            object.__setattr__(self, "reference", tuple(float(c) for c in ref))

    @classmethod
    def from_angles(cls, azimuth: float, plunge: float = 0.0, **options: Any) -> Direction:
        """Direction from ``azimuth`` (degrees clockwise from north) and
        ``plunge`` (degrees below horizontal)."""
        az, pl = np.radians(float(azimuth)), np.radians(float(plunge))
        vector = (np.sin(az) * np.cos(pl), np.cos(az) * np.cos(pl), -np.sin(pl))
        options.setdefault("label", f"az {float(azimuth):g} pl {float(plunge):g}")
        return cls(vector, **options)

    @classmethod
    def omnidirectional(cls, label: str = "omni") -> Direction:
        """All pairs, whatever their orientation."""
        return cls((0.0, 1.0, 0.0), 90.0, None, None, label)

    @property
    def is_omnidirectional(self) -> bool:
        return (
            self.angular_tolerance >= 90.0
            and self.bandwidth_h is None
            and self.bandwidth_v is None
        )

    def frame(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Unit vectors ``(u, e_h, e_v)``: the direction and the two
        perpendicular axes the bandwidths are measured along."""
        u = np.array(self.vector, dtype=np.float64)
        if self.reference is not None:
            ref = np.array(self.reference, dtype=np.float64)
            e_h = ref - (ref @ u) * u
        else:
            e_h = np.array([-u[1], u[0], 0.0])  # up x u
            if np.linalg.norm(e_h) < 1e-12:
                e_h = np.array([1.0, 0.0, 0.0])
        e_h = e_h / np.linalg.norm(e_h)
        e_v = np.cross(u, e_h)
        return u, e_h, e_v / np.linalg.norm(e_v)


def principal_directions(
    ellipsoid: Ellipsoid,
    *,
    angular_tolerance: float | Sequence[float] = 22.5,
    bandwidths: Sequence[float | None] | None = None,
) -> tuple[Direction, Direction, Direction]:
    """Directions along the major, semi-major and minor axes of an ellipsoid.

    Only the orientation of ``ellipsoid`` is used. Bandwidths are measured
    along the *other two* ellipsoid axes:

    =========== ================= =================
    direction   ``bandwidth_h``   ``bandwidth_v``
    =========== ================= =================
    major       along semi-major  along minor
    semi-major  along major       along minor
    minor       along major       along semi-major
    =========== ================= =================

    Parameters
    ----------
    angular_tolerance : one value or three (major, semi-major, minor).
    bandwidths : optional ``(across_major, across_semi_major, across_minor)``
        = the largest allowed offset along each ellipsoid axis when that axis
        is not the direction itself (``None`` entries = unlimited). E.g.
        ``(None, None, 5.0)`` keeps the major and semi-major variograms
        within 5 m of the plane of the two long axes.
    """
    tol = np.broadcast_to(np.asarray(angular_tolerance, dtype=np.float64), (3,))
    bw = (None, None, None) if bandwidths is None else tuple(bandwidths)
    if len(bw) != 3:
        raise ValueError("bandwidths must have three entries (major, semi-major, minor)")
    major = tuple(ellipsoid.major_axis)
    semi = tuple(ellipsoid.semi_major_axis)
    minor = tuple(ellipsoid.minor_axis)
    return (
        Direction(major, float(tol[0]), bw[1], bw[2], "major", semi),
        Direction(semi, float(tol[1]), bw[0], bw[2], "semi_major", major),
        Direction(minor, float(tol[2]), bw[0], bw[1], "minor", major),
    )


# --------------------------------------------------------------------------
# result table
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ExperimentalVariogram:
    """Experimental variogram of one direction: one row per lag.

    Attributes
    ----------
    lag_centre : (n_lags,) nominal lag distance ``k * lag_size``.
    mean_distance : (n_lags,) mean separation of the pairs in the lag (NaN
        for an empty lag); plot and fit against this, not the centre.
    gamma : (n_lags,) variogram value (NaN for an empty lag).
    pairs : (n_lags,) int64 number of pairs.
    direction : the :class:`Direction` used.
    lag_size, lag_tolerance : as used.
    estimator : one of :data:`ESTIMATORS`.
    n_samples : samples with a finite value that took part.
    variance : population variance of those values (of the normal scores for
        ``"normal_scores"``): the usual reference sill of the traditional
        semivariogram. The correlogram's sill is 1; the pairwise relative
        variogram has no such reference.
    kind : ``"spatial"`` or ``"downhole"``.
    """

    lag_centre: np.ndarray
    mean_distance: np.ndarray
    gamma: np.ndarray
    pairs: np.ndarray
    direction: Direction
    lag_size: float
    lag_tolerance: float
    estimator: str
    n_samples: int
    variance: float
    kind: str = "spatial"

    @property
    def label(self) -> str:
        return self.direction.label

    def to_frame(self) -> pd.DataFrame:
        """Table with columns ``lag_centre, mean_distance, gamma, pairs``."""
        return pd.DataFrame(
            {
                "lag_centre": self.lag_centre,
                "mean_distance": self.mean_distance,
                "gamma": self.gamma,
                "pairs": self.pairs,
            }
        )


# --------------------------------------------------------------------------
# pair accumulation
# --------------------------------------------------------------------------
@parallel_njit(cache=True)
def _accumulate(
    key,
    xyz,
    z,
    hole,
    same_hole,
    dom,
    same_dom,
    u,
    e_h,
    e_v,
    cos_tol,
    bw_h,
    bw_v,
    lag,
    n_lags,
    tol,
    max_d,
    n_chunks,
):
    """Sweep over pairs of samples sorted by ``key`` (ascending).

    Returns ``(count, acc)`` with shapes (n_dir, max_lags) and
    (n_dir, max_lags, 6); the six sums per lag are: distance, squared
    difference, ``z_i + z_j``, ``z_i^2 + z_j^2``, ``z_i z_j`` and the
    pairwise-relative term. ``cos_tol <= 0`` accepts every angle; a negative
    bandwidth means unlimited.
    """
    n = xyz.shape[0]
    n_dir = u.shape[0]
    max_lags = 0
    for a in range(n_dir):
        if n_lags[a] > max_lags:
            max_lags = n_lags[a]
    count = np.zeros((n_chunks, n_dir, max_lags), dtype=np.int64)
    acc = np.zeros((n_chunks, n_dir, max_lags, 6), dtype=np.float64)
    for c in prange(n_chunks):
        lo = (c * n) // n_chunks
        hi = ((c + 1) * n) // n_chunks
        for i in range(lo, hi):
            for j in range(i + 1, n):
                if key[j] - key[i] > max_d:
                    break
                if same_hole and hole[i] != hole[j]:
                    continue
                if same_dom and dom[i] != dom[j]:
                    continue
                dx = xyz[j, 0] - xyz[i, 0]
                dy = xyz[j, 1] - xyz[i, 1]
                dz = xyz[j, 2] - xyz[i, 2]
                d = np.sqrt(dx * dx + dy * dy + dz * dz)
                if d == 0.0 or d >= max_d:
                    continue
                diff = z[i] - z[j]
                sq = diff * diff
                total = z[i] + z[j]
                rel = 0.0
                if total != 0.0:
                    rel = 4.0 * sq / (total * total)
                for a in range(n_dir):
                    if cos_tol[a] > 0.0:
                        along = dx * u[a, 0] + dy * u[a, 1] + dz * u[a, 2]
                        if abs(along) < cos_tol[a] * d:
                            continue
                    if bw_h[a] >= 0.0:
                        off = dx * e_h[a, 0] + dy * e_h[a, 1] + dz * e_h[a, 2]
                        if abs(off) > bw_h[a]:
                            continue
                    if bw_v[a] >= 0.0:
                        off = dx * e_v[a, 0] + dy * e_v[a, 1] + dz * e_v[a, 2]
                        if abs(off) > bw_v[a]:
                            continue
                    k_lo = int(np.floor((d - tol[a]) / lag[a]))
                    k_hi = int(np.floor((d + tol[a]) / lag[a])) + 1
                    if k_lo < 1:
                        k_lo = 1
                    if k_hi > n_lags[a]:
                        k_hi = n_lags[a]
                    for k in range(k_lo, k_hi + 1):
                        centre = k * lag[a]
                        if d >= centre - tol[a] and d < centre + tol[a]:
                            count[c, a, k - 1] += 1
                            acc[c, a, k - 1, 0] += d
                            acc[c, a, k - 1, 1] += sq
                            acc[c, a, k - 1, 2] += total
                            acc[c, a, k - 1, 3] += z[i] * z[i] + z[j] * z[j]
                            acc[c, a, k - 1, 4] += z[i] * z[j]
                            acc[c, a, k - 1, 5] += rel
    count_out = np.zeros((n_dir, max_lags), dtype=np.int64)
    acc_out = np.zeros((n_dir, max_lags, 6), dtype=np.float64)
    for c in range(n_chunks):
        count_out += count[c]
        acc_out += acc[c]
    return count_out, acc_out


def _per_direction(value: Any, n_dir: int, name: str, dtype: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full(n_dir, arr, dtype=dtype)
    if arr.shape != (n_dir,):
        raise ValueError(f"{name} must be one value or one per direction")
    return arr


def _codes(ids: Any, n: int, name: str) -> np.ndarray:
    arr = np.asarray(ids)
    if arr.shape != (n,):
        raise ValueError(f"{name} must have one entry per sample")
    _, inverse = np.unique(arr, return_inverse=True)
    return np.ascontiguousarray(np.asarray(inverse).reshape(-1), dtype=np.int64)


def experimental_variogram(
    coords: Any,
    values: Any,
    *,
    lag_size: float | Sequence[float],
    n_lags: int | Sequence[int],
    lag_tolerance: float | Sequence[float] | None = None,
    directions: Direction | Sequence[Direction] | None = None,
    estimator: str = "semivariogram",
    hole_ids: Any = None,
    same_hole_only: bool = False,
    domains: Any = None,
) -> list[ExperimentalVariogram]:
    """Experimental variogram in one or more directions.

    Parameters
    ----------
    coords : (n, 3) sample coordinates (east, north, up), metres.
    values : (n,) sample values. Samples with a NaN / infinite value are
        left out (missing stays missing).
    lag_size, n_lags, lag_tolerance : lag spacing (metres), number of lags
        and half-width of each lag (default ``lag_size / 2``); one value, or
        one per direction (the minor axis of an orebody usually needs a much
        shorter lag than the major axis).
    directions : a :class:`Direction` or a sequence of them; ``None`` = one
        omnidirectional variogram.
    estimator : one of :data:`ESTIMATORS` (module docstring).
    hole_ids : optional (n,) drillhole ids; needed for ``same_hole_only``.
    same_hole_only : use only pairs of samples from the same hole (down-hole
        variogram; see :func:`downhole_variogram`).
    domains : optional (n,) domain codes; when given, only pairs of samples
        with the same code are used (no pairs across a contact).

    Returns
    -------
    A list with one :class:`ExperimentalVariogram` per direction, in the
    order given. Empty lags have ``pairs = 0`` and NaN values.
    """
    if estimator not in ESTIMATORS:
        raise ValueError(f"estimator must be one of {ESTIMATORS}, got {estimator!r}")
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("coords must have shape (n, 3)")
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != (pts.shape[0],):
        raise ValueError("values must have one entry per sample")
    n_all = pts.shape[0]
    if same_hole_only and hole_ids is None:
        raise ValueError("same_hole_only needs hole_ids")
    hole = None if hole_ids is None else _codes(hole_ids, n_all, "hole_ids")
    dom = None if domains is None else _codes(domains, n_all, "domains")

    if directions is None:
        dirs: list[Direction] = [Direction.omnidirectional()]
    elif isinstance(directions, Direction):
        dirs = [directions]
    else:
        dirs = list(directions)
        if not dirs or not all(isinstance(d, Direction) for d in dirs):
            raise ValueError("directions must be Direction objects")
    n_dir = len(dirs)
    lag = _per_direction(lag_size, n_dir, "lag_size", np.float64)
    if not (np.isfinite(lag).all() and (lag > 0.0).all()):
        raise ValueError("lag_size must be positive")
    nl = _per_direction(n_lags, n_dir, "n_lags", np.int64)
    if (nl < 1).any():
        raise ValueError("n_lags must be >= 1")
    if lag_tolerance is None:
        tol = 0.5 * lag
    else:
        tol = _per_direction(lag_tolerance, n_dir, "lag_tolerance", np.float64)
        if not (np.isfinite(tol).all() and (tol > 0.0).all()):
            raise ValueError("lag_tolerance must be positive")

    ok = np.isfinite(vals)
    pts, vals = pts[ok], vals[ok]
    if not np.isfinite(pts).all():
        raise ValueError("samples with a value have non-finite coordinates")
    if estimator == "pairwise_relative" and (vals < 0.0).any():
        raise ValueError("the pairwise relative variogram needs non-negative values")
    if estimator == "normal_scores":
        vals = normal_score(vals)
    n = pts.shape[0]
    variance = float(np.var(vals)) if n else float("nan")

    u = np.zeros((n_dir, 3))
    e_h = np.zeros((n_dir, 3))
    e_v = np.zeros((n_dir, 3))
    cos_tol = np.zeros(n_dir)
    bw_h = np.full(n_dir, -1.0)
    bw_v = np.full(n_dir, -1.0)
    for a, d in enumerate(dirs):
        u[a], e_h[a], e_v[a] = d.frame()
        if d.angular_tolerance < 90.0:
            cos_tol[a] = np.cos(np.radians(d.angular_tolerance))
        if d.bandwidth_h is not None:
            bw_h[a] = d.bandwidth_h
        if d.bandwidth_v is not None:
            bw_v[a] = d.bandwidth_v
    max_d = float(np.max(nl * lag + tol))

    max_lags = int(nl.max())
    if n >= 2:
        axis = int(np.argmax(np.ptp(pts, axis=0)))
        order = np.argsort(pts[:, axis], kind="stable")
        xyz = np.ascontiguousarray(pts[order])
        empty = np.zeros(0, dtype=np.int64)
        count, acc = _accumulate(
            np.ascontiguousarray(xyz[:, axis]),
            xyz,
            np.ascontiguousarray(vals[order]),
            empty if hole is None else np.ascontiguousarray(hole[ok][order]),
            bool(same_hole_only),
            empty if dom is None else np.ascontiguousarray(dom[ok][order]),
            dom is not None,
            u,
            e_h,
            e_v,
            cos_tol,
            bw_h,
            bw_v,
            np.ascontiguousarray(lag),
            np.ascontiguousarray(nl),
            np.ascontiguousarray(tol),
            max_d,
            min(_N_CHUNKS, n),
        )
    else:
        count = np.zeros((n_dir, max_lags), dtype=np.int64)
        acc = np.zeros((n_dir, max_lags, 6))

    out = []
    for a, d in enumerate(dirs):
        k = int(nl[a])
        pairs = count[a, :k].copy()
        sums = acc[a, :k]
        has = pairs > 0
        npairs = np.where(has, pairs, 1).astype(np.float64)
        mean_distance = np.where(has, sums[:, 0] / npairs, np.nan)
        if estimator in ("semivariogram", "normal_scores"):
            gamma = sums[:, 1] / (2.0 * npairs)
        elif estimator == "pairwise_relative":
            gamma = sums[:, 5] / (2.0 * npairs)
        else:  # correlogram
            mean = sums[:, 2] / (2.0 * npairs)
            var = sums[:, 3] / (2.0 * npairs) - mean * mean
            cov = sums[:, 4] / npairs - mean * mean
            with np.errstate(divide="ignore", invalid="ignore"):
                gamma = np.where(var > 0.0, 1.0 - cov / var, np.nan)
        gamma = np.where(has, gamma, np.nan)
        out.append(
            ExperimentalVariogram(
                lag_centre=np.arange(1, k + 1) * float(lag[a]),
                mean_distance=mean_distance,
                gamma=gamma,
                pairs=pairs,
                direction=d,
                lag_size=float(lag[a]),
                lag_tolerance=float(tol[a]),
                estimator=estimator,
                n_samples=n,
                variance=variance,
                kind="downhole" if same_hole_only else "spatial",
            )
        )
    return out


def downhole_variogram(
    coords: Any,
    values: Any,
    hole_ids: Any,
    *,
    lag_size: float,
    n_lags: int,
    lag_tolerance: float | None = None,
    estimator: str = "semivariogram",
    domains: Any = None,
) -> ExperimentalVariogram:
    """Down-hole variogram: pairs of samples from the same hole only.

    Along a hole the samples are close together and evenly spaced, so the
    first lags are well informed at distances far shorter than the drill
    spacing. Extrapolating this variogram to zero distance is the standard
    way to obtain the nugget. Use a lag equal to the composite length.

    Parameters
    ----------
    coords : (n, 3) sample centre coordinates (the pair distance is then the
        straight-line distance between centres), **or** an (n,) array of
        along-hole depths of the sample centres (distance = depth
        difference).
    values, hole_ids : (n,) values and hole identifiers.
    lag_size, n_lags, lag_tolerance, estimator, domains : as in
        :func:`experimental_variogram`.
    """
    pts = np.asarray(coords, dtype=np.float64)
    if pts.ndim == 1:
        pts = np.column_stack([pts, np.zeros_like(pts), np.zeros_like(pts)])
    return experimental_variogram(
        pts,
        values,
        lag_size=lag_size,
        n_lags=n_lags,
        lag_tolerance=lag_tolerance,
        directions=Direction.omnidirectional("downhole"),
        estimator=estimator,
        hole_ids=hole_ids,
        same_hole_only=True,
        domains=domains,
    )[0]


def principal_variograms(
    coords: Any,
    values: Any,
    ellipsoid: Ellipsoid,
    *,
    lag_size: float | Sequence[float],
    n_lags: int | Sequence[int],
    lag_tolerance: float | Sequence[float] | None = None,
    angular_tolerance: float | Sequence[float] = 22.5,
    bandwidths: Sequence[float | None] | None = None,
    estimator: str = "semivariogram",
    hole_ids: Any = None,
    domains: Any = None,
) -> tuple[ExperimentalVariogram, ExperimentalVariogram, ExperimentalVariogram]:
    """Directional variograms along the three axes of an ellipsoid.

    Returns ``(major, semi_major, minor)``, ready for :func:`fit_variogram`.
    Only the orientation of ``ellipsoid`` is used; lag parameters may be one
    value or three (major, semi-major, minor). See
    :func:`principal_directions` for the tolerances.
    """
    dirs = principal_directions(
        ellipsoid, angular_tolerance=angular_tolerance, bandwidths=bandwidths
    )
    major, semi, minor = experimental_variogram(
        coords,
        values,
        lag_size=lag_size,
        n_lags=n_lags,
        lag_tolerance=lag_tolerance,
        directions=dirs,
        estimator=estimator,
        hole_ids=hole_ids,
        domains=domains,
    )
    return major, semi, minor


# --------------------------------------------------------------------------
# least-squares fit
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class VariogramFit:
    """Result of :func:`fit_variogram`.

    Attributes
    ----------
    model : the fitted :class:`VariogramModel` (ranges ordered major,
        semi-major, minor; orientation from the ``ellipsoid`` argument).
    nugget, sills, ranges : the parameters; ``sills`` is (n_structures,) and
        ``ranges`` (n_structures, 3), structures ordered by increasing
        major range.
    total_sill : nugget + sum of sills.
    nugget_ratio : nugget / total sill.
    types : structure types.
    axes : for each experimental variogram, the model axis (0 major, 1
        semi-major, 2 minor) it was fitted as; ``-1`` = isotropic (all axes).
    distance, gamma, fitted, weight : per variogram, the points used in the
        fit (lags with enough pairs), the model values there and the
        normalised least-squares weights.
    weighted_rmse : sqrt(sum w (model - gamma)^2 / sum w).
    r_squared : 1 - weighted SSE / weighted total sum of squares about the
        weighted mean.
    n_points : number of experimental points used.
    n_parameters : number of free parameters.
    at_bounds : names of free parameters that ended on a bound (a warning
        sign: the data do not constrain them).
    success, message : from ``scipy.optimize.least_squares`` (best start).
    """

    model: VariogramModel
    nugget: float
    sills: np.ndarray
    ranges: np.ndarray
    total_sill: float
    nugget_ratio: float
    types: tuple[str, ...]
    axes: tuple[int, ...]
    distance: tuple[np.ndarray, ...]
    gamma: tuple[np.ndarray, ...]
    fitted: tuple[np.ndarray, ...]
    weight: tuple[np.ndarray, ...]
    weighted_rmse: float
    r_squared: float
    n_points: int
    n_parameters: int
    at_bounds: tuple[str, ...] = field(default_factory=tuple)
    success: bool = True
    message: str = ""


_AXIS_NAMES = {"major": 0, "semi_major": 1, "minor": 2, "isotropic": -1, "downhole": -1}


def fit_variogram(
    variograms: ExperimentalVariogram | Sequence[ExperimentalVariogram],
    *,
    types: str | Sequence[str] = "spherical",
    ellipsoid: Ellipsoid | None = None,
    axes: Sequence[int | str] | None = None,
    nugget: float | None = None,
    sill: float | None = None,
    nugget_bounds: tuple[float, float] | None = None,
    range_bounds: Any = None,
    weights: str = "pairs/distance",
    balance_directions: bool = True,
    min_pairs: int = 30,
    max_distance: float | Sequence[float | None] | None = None,
) -> VariogramFit:
    """Weighted least-squares fit of a nugget + 1 or 2 nested structures.

    **This is a seed for kriging parameters, not a modelling tool.** It fits
    a fixed, simple model family automatically; it does not replace looking
    at the experimental points (:func:`variogram_chart_data`) and judging
    whether the result is sensible. Check ``at_bounds`` and the pair counts.

    Model. Along model axis ``a`` (major, semi-major or minor)::

        gamma_a(h) = c0 + sum_k c_k g_k(h / range[k, a])

    with ``g`` as in ``variogram.py`` (practical ranges). The nugget ``c0``
    and the partial sills ``c_k`` are shared by all directions; every
    structure has its own range along each axis (geometric anisotropy). The
    experimental variograms are fitted jointly.

    Objective::

        minimise  sum_d sum_l  w[d, l] * (gamma_model(hbar[d, l]) - gamma[d, l])^2

    over the lags ``l`` of every variogram ``d`` with at least ``min_pairs``
    pairs, evaluated at the mean pair distance ``hbar``. Weights
    (``weights``):

    * ``"pairs/distance"`` (default): ``w = N / hbar``. Lags with many pairs
      are more reliable, and the short lags matter most for kriging, so both
      count; a milder emphasis on short lags than ``N / hbar^2``.
    * ``"pairs/distance^2"``: ``w = N / hbar^2``.
    * ``"pairs"``: ``w = N``.
    * ``"equal"``: ``w = 1``.

    With ``balance_directions`` (default) the weights of each variogram are
    divided by their sum, so every direction has the same total weight and
    the direction with the most pairs does not dictate the shared sill.

    Solver: ``scipy.optimize.least_squares`` (trust-region reflective, with
    bounds) from several starting points; the best solution is returned.

    Parameters
    ----------
    variograms : one experimental variogram (isotropic fit) or several.
    types : structure type, or a sequence of one or two types
        (``"spherical"``, ``"exponential"``, ``"gaussian"``).
    ellipsoid : orientation given to the fitted structures (only its
        rotation is used). Default: no rotation.
    axes : for each variogram, the model axis it measures: 0 / ``"major"``,
        1 / ``"semi_major"``, 2 / ``"minor"`` or -1 / ``"isotropic"``.
        Default: taken from the direction labels set by
        :func:`principal_variograms`; a single variogram, an omnidirectional
        or a down-hole one is isotropic. Model axes not measured by any
        variogram take the range of the major axis if that was fitted, else
        of the nearest fitted axis.
    nugget : fix the nugget at this value (e.g. read from the down-hole
        variogram) instead of fitting it.
    sill : fix the total sill (nugget + structures), e.g. 1.0 for normal
        scores or the sample variance. Default: fitted.
    nugget_bounds : ``(low, high)`` for a fitted nugget. Default
        ``(0, total sill)`` or ``(0, 1.5 * max gamma)``.
    range_bounds : ``(low, high)`` applied to all ranges, or a sequence of
        three such pairs (major, semi-major, minor). Default per axis:
        from half the shortest mean pair distance to twice the longest of
        the variograms on that axis.
    weights, balance_directions, min_pairs : see above.
    max_distance : ignore lags whose mean distance exceeds this (one value
        or one per variogram).

    Returns
    -------
    :class:`VariogramFit`
    """
    if isinstance(variograms, ExperimentalVariogram):
        vgs = [variograms]
    else:
        vgs = list(variograms)
    if not vgs or not all(isinstance(v, ExperimentalVariogram) for v in vgs):
        raise ValueError("variograms must be ExperimentalVariogram objects")
    kinds = (types,) if isinstance(types, str) else tuple(types)
    if len(kinds) not in (1, 2):
        raise ValueError("fit one or two structures")
    kinds = tuple(str(k).lower() for k in kinds)
    for k in kinds:
        if k not in STRUCTURE_TYPES:
            raise ValueError(f"unknown structure type {k!r}")
    n_struct = len(kinds)
    if weights not in FIT_WEIGHTS:
        raise ValueError(f"weights must be one of {FIT_WEIGHTS}, got {weights!r}")

    # ---- which model axis each variogram measures
    if axes is None:
        ax = []
        for v in vgs:
            if len(vgs) == 1 or v.kind == "downhole" or v.direction.is_omnidirectional:
                ax.append(-1)
            elif v.label in _AXIS_NAMES:
                ax.append(_AXIS_NAMES[v.label])
            else:
                raise ValueError(
                    f"cannot tell which axis variogram {v.label!r} measures; pass axes="
                )
    else:
        ax = [_AXIS_NAMES[a] if isinstance(a, str) else int(a) for a in axes]
        if len(ax) != len(vgs) or any(a not in (-1, 0, 1, 2) for a in ax):
            raise ValueError("axes must give -1, 0, 1 or 2 for each variogram")
    isotropic = all(a == -1 for a in ax)
    if not isotropic and any(a == -1 for a in ax):
        # an isotropic variogram next to directional ones has no single range
        raise ValueError("mix of isotropic and directional variograms; pass axes= explicitly")
    fitted_axes = [0] if isotropic else sorted(set(ax))
    col = {a: i for i, a in enumerate(fitted_axes)}  # model axis -> range column
    n_ax = len(fitted_axes)

    # ---- points
    limits = np.broadcast_to(
        np.array(
            [np.inf if m is None else float(m) for m in np.atleast_1d(max_distance)]
            if max_distance is not None
            else [np.inf]
        ),
        (len(vgs),),
    )
    hs, gs, ws, cols = [], [], [], []
    for v, a, limit in zip(vgs, ax, limits, strict=True):
        use = (v.pairs >= int(min_pairs)) & np.isfinite(v.gamma) & (v.mean_distance <= limit)
        h = v.mean_distance[use]
        g = v.gamma[use]
        n_pairs = v.pairs[use].astype(np.float64)
        if weights == "pairs/distance":
            w = n_pairs / h
        elif weights == "pairs/distance^2":
            w = n_pairs / (h * h)
        elif weights == "pairs":
            w = n_pairs
        else:
            w = np.ones_like(h)
        if balance_directions and w.size:
            w = w / w.sum()
        hs.append(h)
        gs.append(g)
        ws.append(w)
        cols.append(0 if isotropic else col[a])
    n_points = int(sum(h.size for h in hs))
    if n_points == 0:
        raise ValueError("no lag has enough pairs to fit (see min_pairs)")
    w_all = np.concatenate(ws)
    w_all = w_all / w_all.sum()
    ws = list(np.split(w_all, np.cumsum([h.size for h in hs])[:-1]))
    g_all = np.concatenate(gs)
    g_max = float(g_all.max())
    if not g_max > 0.0:
        raise ValueError("the experimental variogram is zero everywhere")

    # ---- parameters: [c0] [T] [q] ranges(n_struct, n_ax); gamma in units of `scale`
    scale = float(sill) if sill is not None else g_max
    if sill is not None and not (np.isfinite(sill) and sill > 0.0):
        raise ValueError("sill must be positive")
    if nugget is not None:
        if not (np.isfinite(nugget) and nugget >= 0.0):
            raise ValueError("nugget must be >= 0")
        if sill is not None and nugget > sill:
            raise ValueError("nugget exceeds the fixed sill")
    if nugget_bounds is None:
        nug_lo, nug_hi = 0.0, (float(sill) if sill is not None else 1.5 * g_max)
    else:
        nug_lo, nug_hi = (float(b) for b in nugget_bounds)
        if not 0.0 <= nug_lo < nug_hi:
            raise ValueError("nugget_bounds must satisfy 0 <= low < high")
        if sill is not None:
            nug_hi = min(nug_hi, float(sill))

    r_lo = np.zeros(n_ax)
    r_hi = np.zeros(n_ax)
    for i, a in enumerate(fitted_axes):
        mine = [h for h, c in zip(hs, cols, strict=True) if c == i and h.size]
        if not mine:
            raise ValueError(f"no usable lag for axis {a}")
        r_lo[i] = 0.5 * min(float(h.min()) for h in mine)
        r_hi[i] = 2.0 * max(float(h.max()) for h in mine)
    if range_bounds is not None:
        rb = np.asarray(range_bounds, dtype=np.float64)
        if rb.shape == (2,):
            rb = np.tile(rb, (3, 1))
        if rb.shape != (3, 2) or not ((rb[:, 0] > 0.0) & (rb[:, 0] < rb[:, 1])).all():
            raise ValueError("range_bounds must be (low, high) or three such pairs, 0 < low < high")
        for i, a in enumerate(fitted_axes):
            r_lo[i], r_hi[i] = rb[max(a, 0)]

    names: list[str] = []
    lower: list[float] = []
    upper: list[float] = []
    if nugget is None:
        names.append("nugget")
        lower.append(nug_lo / scale)
        upper.append(nug_hi / scale)
    if sill is None:
        names.append("structured_sill")
        lower.append(0.0)
        upper.append(3.0)
    if n_struct == 2:
        names.append("sill_split")
        lower.append(0.0)
        upper.append(1.0)
    n_head = len(names)
    for k in range(n_struct):
        for i, a in enumerate(fitted_axes):
            suffix = "" if isotropic else "." + ("major", "semi_major", "minor")[a]
            names.append(f"range[{k}]{suffix}")
            lower.append(r_lo[i])
            upper.append(r_hi[i])
    lower_arr = np.array(lower)
    upper_arr = np.array(upper)
    codes = [STRUCTURE_TYPES[k] for k in kinds]

    def unpack(p: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
        """(nugget, sills, ranges) in units of `scale` / metres."""
        pos = 0
        if nugget is None:
            c0 = p[pos]
            pos += 1
        else:
            c0 = nugget / scale
        if sill is None:
            structured = p[pos]
            pos += 1
        else:
            structured = max(1.0 - c0, 0.0)
        if n_struct == 2:
            q = p[pos]
            pos += 1
            sills = np.array([structured * q, structured * (1.0 - q)])
        else:
            sills = np.array([structured])
        return c0, sills, p[n_head:].reshape(n_struct, n_ax)

    def model_values(p: np.ndarray) -> list[np.ndarray]:
        c0, sills, ranges = unpack(p)
        out = []
        for h, c in zip(hs, cols, strict=True):
            g = np.full(h.shape, c0)
            for k in range(n_struct):
                g = g + sills[k] * _unit_variogram(codes[k], h / ranges[k, c])
            out.append(g)
        return out

    sqrt_w = np.sqrt(w_all)
    target = g_all / scale

    def residuals(p: np.ndarray) -> np.ndarray:
        return sqrt_w * (np.concatenate(model_values(p)) - target)

    # ---- multi-start
    best = None
    span = upper_arr - lower_arr
    for frac in (0.15, 0.3, 0.5, 0.75):
        for nug0 in (0.1, 0.4):
            start: list[float] = []
            if nugget is None:
                start.append(nug0)
            if sill is None:
                start.append(max(1.0 - nug0, 0.1))
            if n_struct == 2:
                start.append(0.5)
            for k in range(n_struct):
                f = frac if n_struct == 1 or k == 1 else frac / 3.0
                start.extend(r_lo + f * (r_hi - r_lo))
            x0 = np.clip(np.array(start), lower_arr + 1e-9 * span, upper_arr - 1e-9 * span)
            sol = least_squares(residuals, x0, bounds=(lower_arr, upper_arr), x_scale=span)
            if best is None or sol.cost < best.cost:
                best = sol
    assert best is not None
    p = best.x
    c0, sills, ranges_fit = unpack(p)
    fitted = [g * scale for g in model_values(p)]
    c0 *= scale
    sills = sills * scale

    # ---- full (n_struct, 3) range table
    ranges = np.zeros((n_struct, 3))
    for k in range(n_struct):
        for a in range(3):
            if isotropic:
                ranges[k, a] = ranges_fit[k, 0]
            elif a in col:
                ranges[k, a] = ranges_fit[k, col[a]]
            else:
                nearest = 0 if 0 in col else min(fitted_axes, key=lambda b, a=a: abs(b - a))
                ranges[k, a] = ranges_fit[k, col[nearest]]
    order = np.argsort(ranges[:, 0], kind="stable")
    ranges, sills = ranges[order], sills[order]
    kinds = tuple(kinds[i] for i in order)

    rotation = None if ellipsoid is None else ellipsoid.rotation
    structures = tuple(
        Structure(
            kinds[k],
            float(sills[k]),
            tuple(float(r) for r in ranges[k]),
            axes=None if rotation is None else tuple(map(tuple, rotation)),
        )
        for k in range(n_struct)
    )
    model = VariogramModel(float(c0), structures)

    fit_all = np.concatenate(fitted)
    sse = float(np.sum(w_all * (fit_all - g_all) ** 2))
    mean_w = float(np.sum(w_all * g_all))
    sst = float(np.sum(w_all * (g_all - mean_w) ** 2))
    eps = 1e-6
    at_bounds = tuple(
        name
        for name, value, lo, hi in zip(names, p, lower_arr, upper_arr, strict=True)
        if value - lo <= eps * (hi - lo) or hi - value <= eps * (hi - lo)
    )
    total = float(c0 + sills.sum())
    return VariogramFit(
        model=model,
        nugget=float(c0),
        sills=sills,
        ranges=ranges,
        total_sill=total,
        nugget_ratio=float(c0 / total) if total > 0.0 else float("nan"),
        types=kinds,
        axes=tuple(ax),
        distance=tuple(hs),
        gamma=tuple(gs),
        fitted=tuple(fitted),
        weight=tuple(ws),
        weighted_rmse=float(np.sqrt(sse)),
        r_squared=1.0 - sse / sst if sst > 0.0 else float("nan"),
        n_points=n_points,
        n_parameters=len(names),
        at_bounds=at_bounds,
        success=bool(best.success),
        message=str(best.message),
    )


# --------------------------------------------------------------------------
# chart data
# --------------------------------------------------------------------------
def variogram_chart_data(
    variograms: ExperimentalVariogram | Sequence[ExperimentalVariogram],
    model: VariogramModel | VariogramFit | None = None,
    *,
    n_points: int = 101,
) -> dict[str, Any]:
    """Plain lists for a simple variogram chart (JSON-ready; no plotting here).

    Returns::

        {
          "estimator": "...",
          "sill": <reference sill or None>,       # model total sill if a model
                                                  # is given, else data variance
          "series": [
            {
              "label": "major", "kind": "spatial",
              "vector": [e, n, u] or None,        # None = omnidirectional
              "lag_centre": [...], "distance": [...],   # mean pair distance
              "gamma": [...], "pairs": [...],           # empty lags omitted
              "model_distance": [...], "model_gamma": [...]   # if a model
            }, ...
          ]
        }

    The model curve of a directional series is the model along that
    direction's unit vector, from 0 to the last lag centre (``n_points``
    points; the first point is at distance 0 where gamma = 0, the second
    already carries the nugget). For an omnidirectional or down-hole series
    there is no single direction: the curve is drawn along the major axis of
    the model's first structure for ``"omni"`` and along its minor axis for
    ``"downhole"``, a convention for display only.
    """
    vgs = [variograms] if isinstance(variograms, ExperimentalVariogram) else list(variograms)
    if isinstance(model, VariogramFit):
        model = model.model
    if n_points < 2:
        raise ValueError("n_points must be >= 2")
    series = []
    for v in vgs:
        has = v.pairs > 0
        omni = v.direction.is_omnidirectional
        item: dict[str, Any] = {
            "label": v.label,
            "kind": v.kind,
            "vector": None if omni else [float(c) for c in v.direction.vector],
            "lag_centre": [float(x) for x in v.lag_centre[has]],
            "distance": [float(x) for x in v.mean_distance[has]],
            "gamma": [float(x) for x in v.gamma[has]],
            "pairs": [int(x) for x in v.pairs[has]],
        }
        if model is not None:
            if not omni:
                unit = np.array(v.direction.vector)
            elif model.structures:
                first = model.structures[0].ellipsoid
                unit = first.minor_axis if v.kind == "downhole" else first.major_axis
            else:
                unit = np.array([0.0, 1.0, 0.0])
            dist = np.linspace(0.0, float(v.lag_centre[-1]), int(n_points))
            item["model_distance"] = [float(x) for x in dist]
            item["model_gamma"] = [float(x) for x in model.gamma(dist[:, None] * unit[None, :])]
        series.append(item)
    if model is not None:
        sill: float | None = float(model.total_sill)
    else:
        ref = vgs[0].variance if vgs else float("nan")
        if vgs and vgs[0].estimator == "correlogram":
            ref = 1.0
        if vgs and vgs[0].estimator == "pairwise_relative":
            ref = float("nan")
        sill = float(ref) if np.isfinite(ref) else None
    return {
        "estimator": vgs[0].estimator if vgs else None,
        "sill": sill,
        "series": series,
    }
