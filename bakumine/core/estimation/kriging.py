"""Ordinary and simple kriging, point and block support (research.md 7.4).

Formulas are the textbook ones (Journel & Huijbregts 1978, *Mining
Geostatistics*; Isaaks & Srivastava 1989; Deutsch & Journel 1998, *GSLIB*
book text). No third-party kriging code was consulted or copied.

System (covariance form)
------------------------
For a target ``V`` (a point or a block) and its ``n`` neighbours selected by
:class:`~bakumine.core.estimation.search.SampleSearch`, with ``C`` the
``n x n`` sample-to-sample covariance matrix and ``c`` the vector of
sample-to-target covariances:

* **Simple kriging** (known mean ``m``)::

      C w = c
      z*      = m + sum_i w_i (z_i - m)
      sigma^2 = C(V,V) - sum_i w_i c_i

* **Ordinary kriging** (unknown local mean, weights sum to 1)::

      | C   1 | | w  |   | c |
      | 1'  0 | | mu | = | 1 |
      z*      = sum_i w_i z_i
      sigma^2 = C(V,V) - sum_i w_i c_i - mu

  ``mu`` is the Lagrange multiplier *of this covariance-form system*. The
  variogram-form system of research.md 7.4 has the multiplier ``-mu``; the
  weights and the variance are the same in both forms.

Covariances come from a :class:`~bakumine.core.estimation.variogram.VariogramModel`:
``C(h) = total_sill - gamma(h)``, with ``C(0) = total_sill`` (nugget
included) and no nugget at any non-zero lag.

Support and the nugget (GSLIB ``kt3d`` treatment)
-------------------------------------------------
The treatment below follows the one documented for GSLIB's ``kt3d``. It was
written from the published description, from memory; it was **not** checked
against a ``kt3d`` executable.

* **Sample-to-sample** terms always use the full covariance: the diagonal is
  ``C(0) = total_sill`` (nugget included), off-diagonal terms carry no nugget.
* **Point kriging** (no ``block_size``, or a block with discretisation
  ``(1, 1, 1)``, which ``kt3d`` also treats as a point):
  ``c_i = C(x_i - x0)`` and ``C(V,V) = C(0) = total_sill``. The nugget enters
  ``c_i`` only if the target coincides *exactly* with sample ``i``. Kriging
  is then an exact interpolator **with or without a nugget**: at a sample
  location the estimate equals the sample value and the variance is 0. With
  a nugget the estimate is discontinuous there: an arbitrarily small step
  away, the nugget no longer enters ``c``, the estimate is the smoothed
  value and the variance is at least the nugget (the nugget part of the
  unknown value is uncorrelated with every sample).
* **Block kriging** (``block_size`` and more than one discretisation point):
  the block is represented by ``N = nx * ny * nz`` points, the centres of
  equal cells (:func:`~bakumine.core.estimation._common.discretisation_lattice`),
  which follow ``grid_rotation``. The nugget is treated as variability that
  averages out inside any block, so it is **excluded from both block terms**:

      c_i    = (1/N)   sum_p     C'(x_i - u_p)
      C(V,V) = (1/N^2) sum_p sum_q C'(u_p - u_q)

  where ``C'`` is the covariance of the structures only (no nugget), also
  when a discretisation point coincides with a sample and for the ``p = q``
  terms. ``C(V,V)`` uses the same lattice as ``c_i`` and includes the
  ``p = q`` terms (value ``total_sill - nugget``); it therefore converges to
  the true block covariance from above as the lattice is refined. Block
  kriging is not exact at a sample.
  Consequence used by the tests: with the same neighbourhood, the block
  estimate equals the mean of the point estimates at the discretisation
  points whenever none of them coincides with a sample.

Solver
------
Cholesky factorisation ``C = L L'`` of the sample covariance block, then, for
ordinary kriging, the constraint is eliminated with the Schur complement::

    C a = c,   C b = 1,   mu = (1'a - 1) / (1'b),   w = a - mu b

Chosen over LU with partial pivoting of the bordered matrix because (a) the
sample block is symmetric positive definite for a valid model and distinct
locations, so Cholesky needs no pivoting and half the work; (b) the bordered
matrix is indefinite, while the Schur form only ever divides by Cholesky
pivots and by ``1'C^-1 1 > 0``; (c) the Cholesky pivots are a direct test of
positive definiteness: the factorisation is abandoned as soon as a pivot is
``<= pivot_tolerance * C(0)`` (default 1e-10), which catches singular and
badly conditioned systems. Simple kriging is the same code without the
constraint step.

A target whose system fails this test, or whose result is not finite or has
a clearly negative variance, is **left unestimated** (estimate NaN) with
``status`` 2 or 3 and counted in the result's diagnostics. Variances between
``-1e-9 * C(0)`` and 0 (rounding at exact interpolation) are set to 0.

Duplicate samples
-----------------
Two samples at the same place make ``C`` singular. Inside each neighbourhood
samples closer than ``duplicate_tolerance`` (world Euclidean distance,
default 1e-6 m) to an earlier (nearer / lower-index) sample of the
neighbourhood are **merged** with it: the system gets one sample at the
earlier sample's location carrying the *mean* of the merged values. The
weight of the merged sample is reported split equally among its members.
``n_merged`` counts, per target, the samples merged away; totals are in the
diagnostics. ``n_samples`` is the number of samples the search selected
(before merging).

Negative weights
----------------
``sum_negative_weights`` is the sum of the negative weights of the solved
system. With ``negative_weight_correction=True`` negative weights are set to
0 and the remaining weights rescaled so that the sum of weights is unchanged
(1 for ordinary kriging; for simple kriging, if the original sum is not
positive all weights become 0 and the estimate is the mean). The estimate
then uses the corrected weights and the variance is recomputed for them with
the general formula ``C(V,V) - 2 w'c + w'C w``; ``lagrange`` keeps the value
of the solved system. Off by default.

By-products
-----------
* ``slope`` - slope of the regression of the true value on the estimate,
  ``Cov(Z_V, Z*) / Var(Z*) = w'c / w'C w``. For simple kriging it is 1; for
  ordinary kriging it equals ``w'c / (w'c - mu)``, the usual
  ``(BV - KV + |mu|) / (BV - KV + 2|mu|)``.
* ``efficiency`` - kriging efficiency (Krige 1996),
  ``(BV - KV) / BV`` with ``BV = C(V,V)`` (the block variance; ``C(0)`` for
  point kriging) and ``KV`` the kriging variance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange

from bakumine.core.estimation._common import (
    as_targets,
    block_size_array,
    clean_samples,
    discretisation_lattice,
    distance_summaries,
    grid_rotation_matrix,
)
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SampleSearch, SearchParams
from bakumine.core.estimation.variogram import VariogramModel, lag_covariance
from bakumine.core.parallel import parallel_njit

__all__ = [
    "STATUS_ESTIMATED",
    "STATUS_INVALID",
    "STATUS_NOT_SELECTED",
    "STATUS_SINGULAR",
    "KrigingResult",
    "block_covariance",
    "krige",
    "ordinary_kriging",
    "simple_kriging",
]

#: per-target status codes
STATUS_ESTIMATED = 0
STATUS_NOT_SELECTED = 1  # the search did not flag the target as estimable
STATUS_SINGULAR = 2  # Cholesky pivot below tolerance: singular / ill-conditioned
STATUS_INVALID = 3  # non-finite result or clearly negative variance

_NEGATIVE_VARIANCE_TOLERANCE = 1e-9


@dataclass(frozen=True)
class KrigingResult:
    """Kriging estimate of ``m`` targets. All per-target arrays have length
    ``m``; float fields are NaN where the target is not estimated.

    Attributes
    ----------
    estimate : kriged value.
    variance : kriging variance (see module docstring for the support).
    n_samples : int32, samples selected by the search (before merging
        duplicates; reported for unestimated targets too).
    mean_distance, nearest_distance : mean / smallest anisotropic distance
        (search ellipsoid) from the target centroid to its neighbours.
    estimated : bool, ``status == 0``.
    status : int8, 0 estimated, 1 not selected by the search (too few
        samples / sectors), 2 singular or ill-conditioned system, 3 invalid
        numerical result.
    sum_weights : sum of the sample weights (1 for ordinary kriging; for
        simple kriging the mean receives ``1 - sum_weights``).
    lagrange : Lagrange multiplier ``mu`` of the covariance-form system
        (ordinary kriging; NaN for simple kriging).
    sum_negative_weights : sum of the negative weights (<= 0) of the solved
        system, before any correction.
    slope : slope of regression of truth on estimate.
    efficiency : kriging efficiency.
    n_merged : int32, samples merged into another as duplicates.
    weights : ``None``, or (len(weight_targets), max_samples) float64: the
        weight of each neighbour (NaN padding), for the targets requested
        with ``return_weights_for``.
    weight_samples : matching (.., max_samples) int64 rows into the caller's
        ``samples`` array (-1 padding), nearest first.
    weight_targets : the requested target rows.
    method : ``"ordinary"`` or ``"simple"``; support : ``"point"`` or
        ``"block"``.
    n_samples_excluded : samples dropped before the search (NaN / infinite
        value).
    n_not_selected, n_singular, n_invalid : number of targets per status.
    n_targets_with_duplicates, n_duplicates_merged : duplicate statistics
        over the targets for which a system was built.
    """

    estimate: np.ndarray
    variance: np.ndarray
    n_samples: np.ndarray
    mean_distance: np.ndarray
    nearest_distance: np.ndarray
    estimated: np.ndarray
    status: np.ndarray
    sum_weights: np.ndarray
    lagrange: np.ndarray
    sum_negative_weights: np.ndarray
    slope: np.ndarray
    efficiency: np.ndarray
    n_merged: np.ndarray
    weights: np.ndarray | None
    weight_samples: np.ndarray | None
    weight_targets: np.ndarray | None
    method: str
    support: str
    n_samples_excluded: int
    n_not_selected: int
    n_singular: int
    n_invalid: int
    n_targets_with_duplicates: int
    n_duplicates_merged: int


@njit(cache=True)
def _block_covariance(unit, size, grid, types, sills, transforms):
    """Mean nugget-free covariance between all pairs of discretisation
    points (``p = q`` pairs included) of one block."""
    n = unit.shape[0]
    pts = np.empty((n, 3))
    for p in range(n):
        lx = unit[p, 0] * size[0]
        ly = unit[p, 1] * size[1]
        lz = unit[p, 2] * size[2]
        for k in range(3):
            pts[p, k] = grid[k, 0] * lx + grid[k, 1] * ly + grid[k, 2] * lz
    total = 0.0
    for p in range(n):
        total += lag_covariance(0.0, 0.0, 0.0, 0.0, types, sills, transforms)
        for q in range(p):
            total += 2.0 * lag_covariance(
                pts[p, 0] - pts[q, 0],
                pts[p, 1] - pts[q, 1],
                pts[p, 2] - pts[q, 2],
                0.0,
                types,
                sills,
                transforms,
            )
    return total / (n * n)


def block_covariance(
    model: VariogramModel,
    block_size: Any,
    discretisation: Any,
    grid_rotation: Any = None,
) -> float:
    """Block-to-block average covariance ``C(V,V)`` of one block.

    Mean of the nugget-free covariance over all ``N^2`` pairs of the
    ``nx * ny * nz`` discretisation points (module docstring, "Support and
    the nugget"). ``block_size = (dx, dy, dz)``.
    """
    size = block_size_array(block_size, 1)[0]
    arrays = model.as_arrays()
    return float(
        _block_covariance(
            discretisation_lattice(discretisation),
            size,
            grid_rotation_matrix(grid_rotation),
            arrays.types,
            arrays.sills,
            arrays.transforms,
        )
    )


@parallel_njit(cache=True)
def _krige_kernel(
    indices,
    counts,
    estimated,
    targets,
    coords,
    values,
    nugget,
    types,
    sills,
    transforms,
    ordinary,
    sk_mean,
    block,
    unit,
    sizes,
    grid,
    cbb,
    dup_tol2,
    pivot_tol,
    correct_negative,
    keep_weights,
    out_est,
    out_var,
    out_sumw,
    out_mu,
    out_neg,
    out_slope,
    out_eff,
    out_merged,
    out_status,
    out_w,
):
    """Build and solve the kriging system of every target of one chunk.

    ``indices, counts, estimated`` are search result rows; ``coords`` /
    ``values`` the (cleaned) samples. Block support: ``unit`` (N, 3) lattice,
    ``sizes`` (m, 3) or (1, 3), ``grid`` rotation, ``cbb`` (m,) block
    covariances. Float outputs must be pre-filled with NaN; they are written
    only for targets that end with status 0.
    """
    m = indices.shape[0]
    n_points = unit.shape[0]
    shared = sizes.shape[0] == 1
    c_zero = lag_covariance(0.0, 0.0, 0.0, nugget, types, sills, transforms)
    for t in prange(m):
        n0 = counts[t]
        out_merged[t] = 0
        if not estimated[t] or n0 == 0:
            out_status[t] = 1
            continue

        # ---- merge coincident samples (mean value, first location)
        group = np.empty(n0, dtype=np.int64)
        gx = np.empty((n0, 3))
        gz = np.empty(n0)
        gsize = np.empty(n0)
        n = 0
        for a in range(n0):
            i = indices[t, a]
            found = -1
            for g in range(n):
                dx = coords[i, 0] - gx[g, 0]
                dy = coords[i, 1] - gx[g, 1]
                dz = coords[i, 2] - gx[g, 2]
                if dx * dx + dy * dy + dz * dz <= dup_tol2:
                    found = g
                    break
            if found < 0:
                gx[n, 0] = coords[i, 0]
                gx[n, 1] = coords[i, 1]
                gx[n, 2] = coords[i, 2]
                gz[n] = values[i]
                gsize[n] = 1.0
                group[a] = n
                n += 1
            else:
                gz[found] += values[i]
                gsize[found] += 1.0
                group[a] = found
        for g in range(n):
            gz[g] /= gsize[g]
        out_merged[t] = n0 - n

        # ---- sample-to-sample covariance (full matrix, nugget on diagonal)
        c = np.empty((n, n))
        for a in range(n):
            c[a, a] = c_zero
            for b in range(a):
                v = lag_covariance(
                    gx[a, 0] - gx[b, 0],
                    gx[a, 1] - gx[b, 1],
                    gx[a, 2] - gx[b, 2],
                    nugget,
                    types,
                    sills,
                    transforms,
                )
                c[a, b] = v
                c[b, a] = v

        # ---- sample-to-target covariance and C(V,V)
        r = np.empty(n)
        if block:
            size = sizes[0] if shared else sizes[t]
            pts = np.empty((n_points, 3))
            for p in range(n_points):
                lx = unit[p, 0] * size[0]
                ly = unit[p, 1] * size[1]
                lz = unit[p, 2] * size[2]
                for k in range(3):
                    pts[p, k] = targets[t, k] + grid[k, 0] * lx + grid[k, 1] * ly + grid[k, 2] * lz
            for a in range(n):
                total = 0.0
                for p in range(n_points):
                    total += lag_covariance(
                        gx[a, 0] - pts[p, 0],
                        gx[a, 1] - pts[p, 1],
                        gx[a, 2] - pts[p, 2],
                        0.0,
                        types,
                        sills,
                        transforms,
                    )
                r[a] = total / n_points
            c00 = cbb[t]
        else:
            for a in range(n):
                r[a] = lag_covariance(
                    gx[a, 0] - targets[t, 0],
                    gx[a, 1] - targets[t, 1],
                    gx[a, 2] - targets[t, 2],
                    nugget,
                    types,
                    sills,
                    transforms,
                )
            c00 = c_zero

        # ---- Cholesky C = L L' with pivot test
        low = np.empty((n, n))
        singular = False
        for j in range(n):
            s = c[j, j]
            for k in range(j):
                s -= low[j, k] * low[j, k]
            if not (s > pivot_tol * c[j, j]):
                singular = True
                break
            diag = np.sqrt(s)
            low[j, j] = diag
            for i in range(j + 1, n):
                v = c[i, j]
                for k in range(j):
                    v -= low[i, k] * low[j, k]
                low[i, j] = v / diag
        if singular:
            out_status[t] = 2
            continue

        # ---- solve C a = r (and C b = 1 for the constraint)
        w = np.empty(n)
        for i in range(n):
            v = r[i]
            for k in range(i):
                v -= low[i, k] * w[k]
            w[i] = v / low[i, i]
        for i in range(n - 1, -1, -1):
            v = w[i]
            for k in range(i + 1, n):
                v -= low[k, i] * w[k]
            w[i] = v / low[i, i]
        mu = 0.0
        if ordinary:
            ones = np.empty(n)
            for i in range(n):
                v = 1.0
                for k in range(i):
                    v -= low[i, k] * ones[k]
                ones[i] = v / low[i, i]
            for i in range(n - 1, -1, -1):
                v = ones[i]
                for k in range(i + 1, n):
                    v -= low[k, i] * ones[k]
                ones[i] = v / low[i, i]
            sum_a = 0.0
            sum_b = 0.0
            for i in range(n):
                sum_a += w[i]
                sum_b += ones[i]
            if not (sum_b > 0.0):
                out_status[t] = 2
                continue
            mu = (sum_a - 1.0) / sum_b
            for i in range(n):
                w[i] -= mu * ones[i]

        # ---- weights: diagnostics and optional correction
        sum_w = 0.0
        neg = 0.0
        for i in range(n):
            sum_w += w[i]
            if w[i] < 0.0:
                neg += w[i]
        corrected = False
        if correct_negative and neg < 0.0:
            corrected = True
            pos = sum_w - neg
            factor = sum_w / pos if (sum_w > 0.0 and pos > 0.0) else 0.0
            for i in range(n):
                w[i] = w[i] * factor if w[i] > 0.0 else 0.0
            sum_w = sum_w if factor > 0.0 else 0.0

        wr = 0.0
        wcw = 0.0
        est = 0.0
        for i in range(n):
            wr += w[i] * r[i]
            est += w[i] * gz[i]
            row = 0.0
            for k in range(n):
                row += c[i, k] * w[k]
            wcw += w[i] * row
        if not ordinary:
            est += (1.0 - sum_w) * sk_mean
        if corrected:
            var = c00 - 2.0 * wr + wcw
        else:
            var = c00 - wr - mu
        if not (np.isfinite(est) and np.isfinite(var)):
            out_status[t] = 3
            continue
        if var < 0.0:
            if var < -_NEGATIVE_VARIANCE_TOLERANCE * c_zero:
                out_status[t] = 3
                continue
            var = 0.0

        out_status[t] = 0
        out_est[t] = est
        out_var[t] = var
        out_sumw[t] = sum_w
        if ordinary:
            out_mu[t] = mu
        out_neg[t] = neg
        if wcw > 0.0:
            out_slope[t] = wr / wcw
        if c00 > 0.0:
            out_eff[t] = (c00 - var) / c00
        if keep_weights:
            for a in range(n0):
                g = group[a]
                out_w[t, a] = w[g] / gsize[g]


def krige(
    samples: Any,
    values: Any,
    targets: Any,
    model: VariogramModel,
    ellipsoid: Ellipsoid,
    params: SearchParams | None = None,
    *,
    method: str = "ordinary",
    mean: float | None = None,
    hole_ids: Any = None,
    block_size: Any = None,
    discretisation: Any = (1, 1, 1),
    grid_rotation: Any = None,
    negative_weight_correction: bool = False,
    duplicate_tolerance: float = 1e-6,
    pivot_tolerance: float = 1e-10,
    return_weights_for: Any = None,
    chunk_size: int = 100_000,
    workers: int = -1,
) -> KrigingResult:
    """Ordinary or simple kriging of points or blocks.

    Parameters
    ----------
    samples : (n, 3) world coordinates (east, north, up) of the samples.
    values : (n,) sample values; NaN / infinite values are excluded before
        the search (count in ``n_samples_excluded``).
    targets : (m, 3) points to estimate, or block centroids.
    model : variogram model (supplies the covariances; its anisotropy is its
        own and independent of the search ellipsoid).
    ellipsoid : search ellipsoid (neighbour selection and the reported
        distances only).
    params : search options (default ``SearchParams()``: up to 24 samples).
    method : ``"ordinary"`` or ``"simple"``.
    mean : the known mean, required for simple kriging.
    hole_ids : optional (n,) drillhole ids, needed for ``max_per_hole``.
    block_size : ``None`` for point kriging; ``(dx, dy, dz)`` or an (m, 3)
        array of per-target block dimensions for block kriging.
    discretisation : ``(nx, ny, nz)`` points per block; ``(1, 1, 1)`` is
        point kriging at the centroid (as in GSLIB ``kt3d``).
    grid_rotation : optional (3, 3) rotation of the block grid; columns are
        the world directions of the blocks' local x, y, z edges.
    negative_weight_correction : reset negative weights to zero and
        renormalise (module docstring). Default off.
    duplicate_tolerance : samples of one neighbourhood closer than this
        (metres, world distance) are merged by averaging.
    pivot_tolerance : relative Cholesky pivot below which a system is
        declared singular / ill-conditioned and the target left unestimated.
    return_weights_for : optional target rows for which the weights are
        returned (tests, debugging).
    chunk_size : targets per batch (memory only; results do not depend on
        it).
    workers : threads for the KD-tree query (-1 = all cores).

    Returns
    -------
    :class:`KrigingResult`. See the module docstring for conventions.
    """
    if method not in ("ordinary", "simple"):
        raise ValueError("method must be 'ordinary' or 'simple'")
    ordinary = method == "ordinary"
    if ordinary:
        if mean is not None:
            raise ValueError("mean is only used by simple kriging")
        sk_mean = 0.0
    else:
        if mean is None or not np.isfinite(mean):
            raise ValueError("simple kriging needs a finite mean")
        sk_mean = float(mean)
    if not isinstance(model, VariogramModel):
        raise TypeError("model must be a VariogramModel")
    if not (duplicate_tolerance >= 0.0 and np.isfinite(duplicate_tolerance)):
        raise ValueError("duplicate_tolerance must be >= 0")
    if not (0.0 <= pivot_tolerance < 1.0):
        raise ValueError("pivot_tolerance must be in [0, 1)")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    params = SearchParams() if params is None else params
    clean = clean_samples(samples, values, hole_ids)
    tg = as_targets(targets)
    m = tg.shape[0]
    arrays = model.as_arrays()
    grid = grid_rotation_matrix(grid_rotation)

    unit = discretisation_lattice(discretisation)
    block = block_size is not None and unit.shape[0] > 1
    if block_size is not None:
        sizes = block_size_array(block_size, m)
    if not block:
        sizes = np.ones((1, 3))
        unit = np.zeros((1, 3))
    shared_cbb = None
    if block and sizes.shape[0] == 1:
        shared_cbb = _block_covariance(
            unit, sizes[0], grid, arrays.types, arrays.sills, arrays.transforms
        )

    want = None
    if return_weights_for is not None:
        want = np.atleast_1d(np.asarray(return_weights_for, dtype=np.int64))
        if want.ndim != 1 or ((want < 0) | (want >= m)).any():
            raise ValueError("return_weights_for must hold target rows in [0, m)")
    width = int(params.max_samples)

    def full(dtype=np.float64, fill=np.nan):
        return np.full(m, fill, dtype=dtype)

    estimate, variance, sum_w, lagrange = full(), full(), full(), full()
    neg, slope, eff = full(), full(), full()
    mean_distance, nearest_distance = full(), full()
    n_samples = full(np.int32, 0)
    n_merged = full(np.int32, 0)
    status = full(np.int8, STATUS_NOT_SELECTED)
    weights = weight_samples = None
    if want is not None:
        weights = np.full((want.size, width), np.nan)
        weight_samples = np.full((want.size, width), -1, dtype=np.int64)

    search = SampleSearch(clean.coords, ellipsoid, clean.hole_ids)
    for start in range(0, m, chunk_size):
        stop = min(m, start + chunk_size)
        chunk = tg[start:stop]
        found = search.query(chunk, params, chunk_size=chunk_size, workers=workers)
        if block:
            if shared_cbb is not None:
                chunk_sizes = sizes
                cbb = np.full(stop - start, shared_cbb)
            else:
                chunk_sizes = sizes[start:stop]
                unique, inverse = np.unique(chunk_sizes, axis=0, return_inverse=True)
                per_size = np.array(
                    [
                        _block_covariance(
                            unit, size, grid, arrays.types, arrays.sills, arrays.transforms
                        )
                        for size in unique
                    ]
                )
                cbb = np.ascontiguousarray(per_size[np.asarray(inverse).reshape(-1)])
        else:
            chunk_sizes = sizes
            cbb = np.zeros(1)
        rows = None
        if want is not None:
            rows = np.flatnonzero((want >= start) & (want < stop))
        keep = rows is not None and rows.size > 0
        out_w = np.full((stop - start, width), np.nan) if keep else np.zeros((1, 1))
        _krige_kernel(
            found.indices,
            found.counts,
            found.estimated,
            chunk,
            clean.coords,
            clean.values,
            arrays.nugget,
            arrays.types,
            arrays.sills,
            arrays.transforms,
            ordinary,
            sk_mean,
            block,
            unit,
            chunk_sizes,
            grid,
            cbb,
            float(duplicate_tolerance) ** 2,
            float(pivot_tolerance),
            bool(negative_weight_correction),
            keep,
            estimate[start:stop],
            variance[start:stop],
            sum_w[start:stop],
            lagrange[start:stop],
            neg[start:stop],
            slope[start:stop],
            eff[start:stop],
            n_merged[start:stop],
            status[start:stop],
            out_w,
        )
        n_samples[start:stop] = found.counts
        mean_distance[start:stop], nearest_distance[start:stop] = distance_summaries(found)
        if keep:
            local = want[rows] - start
            good = status[start:stop][local] == STATUS_ESTIMATED
            weights[rows[good]] = out_w[local[good]]
            idx = found.indices[local]
            weight_samples[rows] = np.where(idx >= 0, clean.kept[np.maximum(idx, 0)], -1)

    built = status != STATUS_NOT_SELECTED
    return KrigingResult(
        estimate=estimate,
        variance=variance,
        n_samples=n_samples,
        mean_distance=mean_distance,
        nearest_distance=nearest_distance,
        estimated=status == STATUS_ESTIMATED,
        status=status,
        sum_weights=sum_w,
        lagrange=lagrange,
        sum_negative_weights=neg,
        slope=slope,
        efficiency=eff,
        n_merged=n_merged,
        weights=weights,
        weight_samples=weight_samples,
        weight_targets=want,
        method=method,
        support="block" if block else "point",
        n_samples_excluded=clean.n_excluded,
        n_not_selected=int((status == STATUS_NOT_SELECTED).sum()),
        n_singular=int((status == STATUS_SINGULAR).sum()),
        n_invalid=int((status == STATUS_INVALID).sum()),
        n_targets_with_duplicates=int((built & (n_merged > 0)).sum()),
        n_duplicates_merged=int(n_merged[built].sum()),
    )


def ordinary_kriging(
    samples: Any,
    values: Any,
    targets: Any,
    model: VariogramModel,
    ellipsoid: Ellipsoid,
    params: SearchParams | None = None,
    **options: Any,
) -> KrigingResult:
    """Ordinary kriging; see :func:`krige` for the parameters."""
    return krige(samples, values, targets, model, ellipsoid, params, method="ordinary", **options)


def simple_kriging(
    samples: Any,
    values: Any,
    targets: Any,
    model: VariogramModel,
    ellipsoid: Ellipsoid,
    params: SearchParams | None = None,
    *,
    mean: float,
    **options: Any,
) -> KrigingResult:
    """Simple kriging with the known ``mean``; see :func:`krige`."""
    return krige(
        samples, values, targets, model, ellipsoid, params, method="simple", mean=mean, **options
    )
