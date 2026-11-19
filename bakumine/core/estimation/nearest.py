"""Nearest-neighbour estimation (research.md section 7.2).

Each target takes the value of the sample at the smallest *anisotropic*
distance (research.md 7.5), provided that distance is within the search
radius. The weights are 1 for that sample and 0 for all others, which makes
the estimate a Voronoi (polygonal) assignment in the ellipsoid's isotropic
space; its global mean over a domain is the polygonal-declustered mean of the
samples.

Selection is delegated to :class:`~bakumine.core.estimation.search.SampleSearch`
with ``max_samples = 1``, so the rule is exactly the search's: candidates are
ordered by ``(distance, sample index)`` and **ties go to the sample with the
lowest row index** in the caller's ``samples`` array, independent of thread
count and chunking. The search (KD-tree query and its numba selection kernel)
is the parallel part; what remains here is a vectorised gather.

A block is represented by its centroid: nearest neighbour has no block
discretisation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from bakumine.core.estimation._common import as_targets, clean_samples
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SampleSearch, SearchParams

__all__ = ["NearestResult", "nearest_neighbour"]


@dataclass(frozen=True)
class NearestResult:
    """Nearest-neighbour estimate of ``m`` targets.

    Attributes
    ----------
    estimate : (m,) float64. Value of the chosen sample; NaN where no sample
        qualified.
    sample_index : (m,) int64. Row of the chosen sample in the ``samples`` /
        ``values`` arrays as passed by the caller (before NaN removal);
        ``-1`` where unestimated.
    n_samples : (m,) int32. 1 where estimated, else 0.
    mean_distance, nearest_distance : (m,) float64. Both are the anisotropic
        distance to the chosen sample (kept as two fields so that all
        estimators share one layout); NaN where unestimated.
    estimated : (m,) bool.
    n_samples_excluded : samples dropped before the search because their
        value was NaN or infinite.
    """

    estimate: np.ndarray
    sample_index: np.ndarray
    n_samples: np.ndarray
    mean_distance: np.ndarray
    nearest_distance: np.ndarray
    estimated: np.ndarray
    n_samples_excluded: int


def nearest_neighbour(
    samples: Any,
    values: Any,
    targets: Any,
    ellipsoid: Ellipsoid,
    *,
    radius: float | None = None,
    max_vertical: float | None = None,
    exclude_zero_distance: bool = False,
    chunk_size: int = 200_000,
    workers: int = -1,
) -> NearestResult:
    """Nearest-neighbour estimate at target points.

    Parameters
    ----------
    samples : (n, 3) world coordinates (east, north, up) of the samples.
    values : (n,) sample values. NaN / infinite values are excluded before
        the search (count in ``n_samples_excluded``).
    targets : (m, 3) points to estimate (block centroids).
    ellipsoid : anisotropy of the distance; its major radius is the default
        search radius.
    radius : search radius as anisotropic distance; ``None`` = the
        ellipsoid's major radius. A sample at exactly the radius qualifies.
    max_vertical : optional maximum ``|dz|`` (world Z) between sample and
        target; a nearer sample outside this limit is skipped.
    exclude_zero_distance : ignore samples at distance exactly 0
        (leave-one-out use).
    chunk_size : targets per batch (memory only; results do not depend on
        it).
    workers : threads for the KD-tree query (-1 = all cores).

    Returns
    -------
    :class:`NearestResult`
    """
    clean = clean_samples(samples, values)
    tg = as_targets(targets)
    m = tg.shape[0]
    search = SampleSearch(clean.coords, ellipsoid)
    params = SearchParams(
        max_samples=1,
        min_samples=1,
        radius=radius,
        max_vertical=max_vertical,
        exclude_zero_distance=exclude_zero_distance,
    )
    found = search.query(tg, params, chunk_size=chunk_size, workers=workers)
    ok = found.estimated
    local = found.indices[:, 0]
    estimate = np.full(m, np.nan)
    estimate[ok] = clean.values[local[ok]]
    index = np.full(m, -1, dtype=np.int64)
    index[ok] = clean.kept[local[ok]]
    distance = np.where(ok, found.distances[:, 0], np.nan)
    return NearestResult(
        estimate=estimate,
        sample_index=index,
        n_samples=ok.astype(np.int32),
        mean_distance=distance,
        nearest_distance=distance.copy(),
        estimated=ok.copy(),
        n_samples_excluded=clean.n_excluded,
    )
