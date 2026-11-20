"""Inverse distance weighting (research.md section 7.3).

Point estimate
--------------
With ``d_i`` the *anisotropic* distance (research.md 7.5) from the target to
neighbour ``i`` and power ``p > 0``::

    z* = sum_i w_i z_i / sum_i w_i,        w_i = 1 / d_i^p

evaluated as ``w_i = (d_min / d_i)^p`` (the common factor cancels), which
neither overflows for tiny distances nor underflows for large powers.

**Coincident samples.** If one or more neighbours are at distance exactly 0
the estimate is the mean of the values of those neighbours (a single
coincident sample: its value); the other neighbours get no weight.

The neighbours are those selected by
:class:`~bakumine.core.estimation.search.SampleSearch` around the target
(block centroid), so every :class:`SearchParams` option applies, and targets
the search flags as not estimated stay NaN.

Block estimate (optional)
-------------------------
With ``block_size`` given, the block is discretised into ``nx * ny * nz``
points (centres of equal cells, see
:func:`~bakumine.core.estimation._common.discretisation_lattice`; the points
follow ``grid_rotation``) and the block estimate is **the average of the
point estimates** at those points, each computed with the rule above
(including the coincident-sample rule, applied per point) from the *same*
neighbourhood - the one found around the block centroid. This is a linear
combination of the samples with weights equal to the mean of the normalised
point weights; it is *not* the alternative "weight by the mean inverse
distance" variant, which was not chosen because it gives a sample that sits
on one discretisation point an unbounded share of the whole block.
``discretisation = (1, 1, 1)`` is the point estimate at the centroid.

``n_samples``, ``mean_distance`` and ``nearest_distance`` always refer to the
centroid's neighbourhood.
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
from bakumine.core.parallel import parallel_njit

__all__ = ["IDWResult", "inverse_distance"]


@dataclass(frozen=True)
class IDWResult:
    """IDW estimate of ``m`` targets.

    Attributes
    ----------
    estimate : (m,) float64; NaN where unestimated.
    n_samples : (m,) int32. Neighbours selected by the search (also reported
        for unestimated targets, e.g. fewer than ``min_samples``).
    mean_distance : (m,) float64. Mean anisotropic distance from the target
        (centroid) to its neighbours; NaN without neighbours.
    nearest_distance : (m,) float64. Smallest such distance.
    estimated : (m,) bool.
    n_coincident : (m,) int32. Neighbours at distance exactly 0 from the
        target point (point support) or from any discretisation point,
        summed over the points (block support).
    n_samples_excluded : samples dropped before the search because their
        value was NaN or infinite.
    """

    estimate: np.ndarray
    n_samples: np.ndarray
    mean_distance: np.ndarray
    nearest_distance: np.ndarray
    estimated: np.ndarray
    n_coincident: np.ndarray
    n_samples_excluded: int


@njit(cache=True)
def _idw_point(d, z, n, power):
    """IDW of ``n`` leading (distance, value) pairs. Returns (estimate,
    number of coincident samples)."""
    d_min = np.inf
    for a in range(n):
        if d[a] < d_min:
            d_min = d[a]
    if d_min == 0.0:
        total = 0.0
        hits = 0
        for a in range(n):
            if d[a] == 0.0:
                total += z[a]
                hits += 1
        return total / hits, hits
    num = 0.0
    den = 0.0
    for a in range(n):
        w = (d_min / d[a]) ** power
        num += w * z[a]
        den += w
    return num / den, 0


@parallel_njit(cache=True)
def _idw_kernel(
    indices,
    distances,
    counts,
    estimated,
    values,
    power,
    block,
    target_iso,
    sample_iso,
    unit,
    sizes,
    iso_grid,
    out_est,
    out_hits,
):
    """IDW for one chunk of targets.

    ``indices, distances, counts, estimated`` are the search result rows.
    Point support uses the search's distances directly. Block support
    (``block`` True): point ``p`` of target ``t`` sits at
    ``target_iso[t] + iso_grid @ (unit[p] * sizes[t])`` in isotropic space
    (``sizes`` has one row per target or a single shared row).
    """
    m, width = indices.shape
    n_points = unit.shape[0]
    shared = sizes.shape[0] == 1
    for t in prange(m):
        n = counts[t]
        if not estimated[t] or n == 0:
            continue
        z = np.empty(n)
        for a in range(n):
            z[a] = values[indices[t, a]]
        if not block:
            out_est[t], out_hits[t] = _idw_point(distances[t], z, n, power)
            continue
        size = sizes[0] if shared else sizes[t]
        d = np.empty(n)
        total = 0.0
        hits_total = 0
        for p in range(n_points):
            lx = unit[p, 0] * size[0]
            ly = unit[p, 1] * size[1]
            lz = unit[p, 2] * size[2]
            px = target_iso[t, 0] + iso_grid[0, 0] * lx + iso_grid[0, 1] * ly + iso_grid[0, 2] * lz
            py = target_iso[t, 1] + iso_grid[1, 0] * lx + iso_grid[1, 1] * ly + iso_grid[1, 2] * lz
            pz = target_iso[t, 2] + iso_grid[2, 0] * lx + iso_grid[2, 1] * ly + iso_grid[2, 2] * lz
            for a in range(n):
                i = indices[t, a]
                ex = sample_iso[i, 0] - px
                ey = sample_iso[i, 1] - py
                ez = sample_iso[i, 2] - pz
                d[a] = np.sqrt(ex * ex + ey * ey + ez * ez)
            est, hits = _idw_point(d, z, n, power)
            total += est
            hits_total += hits
        out_est[t] = total / n_points
        out_hits[t] = hits_total


def inverse_distance(
    samples: Any,
    values: Any,
    targets: Any,
    ellipsoid: Ellipsoid,
    params: SearchParams | None = None,
    *,
    power: float = 2.0,
    hole_ids: Any = None,
    block_size: Any = None,
    discretisation: Any = (1, 1, 1),
    grid_rotation: Any = None,
    chunk_size: int = 200_000,
    workers: int = -1,
) -> IDWResult:
    """Inverse-distance-weighted estimate at points or blocks.

    Parameters
    ----------
    samples : (n, 3) world coordinates (east, north, up) of the samples.
    values : (n,) sample values; NaN / infinite values are excluded before
        the search (count in ``n_samples_excluded``).
    targets : (m, 3) points to estimate, or block centroids.
    ellipsoid : anisotropy of the distance used both for the search and for
        the weights; its major radius is the default search radius.
    params : search options (default ``SearchParams()``: up to 24 samples).
    power : exponent ``p`` (> 0).
    hole_ids : optional (n,) drillhole ids, needed for ``max_per_hole``.
    block_size : ``None`` for point estimates; ``(dx, dy, dz)`` or an (m, 3)
        array of per-target block dimensions for block estimates.
    discretisation : ``(nx, ny, nz)`` points per block (used with
        ``block_size`` only).
    grid_rotation : optional (3, 3) rotation of the block grid; columns are
        the world directions of the blocks' local x, y, z edges.
    chunk_size : targets per batch (memory only).
    workers : threads for the KD-tree query (-1 = all cores).

    Returns
    -------
    :class:`IDWResult`. See the module docstring for the exact rules.
    """
    power = float(power)
    if not (np.isfinite(power) and power > 0.0):
        raise ValueError("power must be a positive finite number")
    if chunk_size < 1:
        raise ValueError("chunk_size must be >= 1")
    params = SearchParams() if params is None else params
    clean = clean_samples(samples, values, hole_ids)
    tg = as_targets(targets)
    m = tg.shape[0]

    block = block_size is not None
    if block:
        sizes = block_size_array(block_size, m)
        unit = discretisation_lattice(discretisation)
        iso_grid = np.ascontiguousarray(
            ellipsoid.isotropic_matrix @ grid_rotation_matrix(grid_rotation)
        )
    else:
        sizes = np.ones((1, 3))
        unit = np.zeros((1, 3))
        iso_grid = np.eye(3)

    search = SampleSearch(clean.coords, ellipsoid, clean.hole_ids)
    estimate = np.full(m, np.nan)
    n_samples = np.zeros(m, dtype=np.int32)
    mean_distance = np.full(m, np.nan)
    nearest_distance = np.full(m, np.nan)
    estimated = np.zeros(m, dtype=np.bool_)
    n_coincident = np.zeros(m, dtype=np.int32)

    for start in range(0, m, chunk_size):
        stop = min(m, start + chunk_size)
        chunk = tg[start:stop]
        found = search.query(chunk, params, chunk_size=chunk_size, workers=workers)
        if block:
            target_iso = np.ascontiguousarray(ellipsoid.to_isotropic(chunk))
            chunk_sizes = sizes if sizes.shape[0] == 1 else sizes[start:stop]
        else:
            target_iso = np.zeros((1, 3))
            chunk_sizes = sizes
        _idw_kernel(
            found.indices,
            found.distances,
            found.counts,
            found.estimated,
            clean.values,
            power,
            block,
            target_iso,
            search.samples_iso,
            unit,
            chunk_sizes,
            iso_grid,
            estimate[start:stop],
            n_coincident[start:stop],
        )
        n_samples[start:stop] = found.counts
        mean_distance[start:stop], nearest_distance[start:stop] = distance_summaries(found)
        estimated[start:stop] = found.estimated & (found.counts > 0)

    return IDWResult(
        estimate=estimate,
        n_samples=n_samples,
        mean_distance=mean_distance,
        nearest_distance=nearest_distance,
        estimated=estimated,
        n_coincident=n_coincident,
        n_samples_excluded=clean.n_excluded,
    )
