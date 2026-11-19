"""Anisotropic sample search (research.md sections 7.5 and 7.6).

:class:`SampleSearch` is built once from sample coordinates and an
:class:`~bakumine.core.estimation.rotation.Ellipsoid`. The samples are
transformed into the ellipsoid's isotropic space (rotate into its axes,
stretch the short axes), where anisotropic distance is ordinary Euclidean
distance, and indexed with a ``scipy.spatial.cKDTree``. Queries for many
target points are answered in batches; the option logic runs in a
numba-compiled parallel kernel. Nothing here depends on block-model or
drillhole classes: inputs are plain arrays.

Selection rule (the exact definition the tests check against brute force)
------------------------------------------------------------------------
For one target point:

1. **Candidates** are the samples with anisotropic distance ``d <= radius``
   (radius defaults to the ellipsoid's major radius), and, if set,
   ``|z_sample - z_target| <= max_vertical`` (world Z, metres), and, if
   ``exclude_zero_distance``, ``d > 0``.
2. Candidates are ordered by ``(d, sample index)`` ascending. **Tie-break:**
   samples at exactly equal distance are taken in order of their row index
   in the ``samples`` array, so results do not depend on thread count,
   chunking or KD-tree internals.
3. Walk the ordered candidates and accept each one unless it would exceed
   ``max_per_hole`` (accepted samples with the same hole id) or
   ``max_per_sector`` (accepted samples in the same sector); a skipped
   candidate is not reconsidered. Stop once ``max_samples`` are accepted.
4. The target is flagged ``estimated`` when the number accepted is
   ``>= min_samples`` and the number of sectors holding at least one
   accepted sample is ``>= min_sectors``.

Sectors are defined in the ellipsoid's rotated (and stretched, which does
not change signs) frame, centred on the target, from the signs of the
sample's offsets along the axes (an offset of exactly 0 counts as positive):

* ``"octant"``: 8 sectors, ``code = (x' >= 0) + 2 (y' >= 0) + 4 (z' >= 0)``
  with ``x'`` = semi-major, ``y'`` = major, ``z'`` = minor axis;
* ``"quadrant"``: 4 sectors, ``code = (x' >= 0) + 2 (y' >= 0)``: quadrants of
  the major / semi-major plane, the minor axis is ignored.

Distances are the anisotropic distances of research.md 7.5 in metres of
equivalent major-axis distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit, prange
from scipy.spatial import cKDTree

from bakumine.core.estimation.rotation import Ellipsoid

__all__ = ["SECTOR_MODES", "SampleSearch", "SearchParams", "SearchResult"]

#: sector mode -> number of sign bits used (2 = quadrant, 3 = octant)
SECTOR_MODES: dict[str | None, int] = {None: 0, "none": 0, "quadrant": 2, "octant": 3}

# Upper bound on (rows x candidates) handled in one KD-tree call: bounds the
# temporary memory at about 16 bytes x this number.
_MAX_CANDIDATE_CELLS = 8_000_000


@dataclass(frozen=True)
class SearchParams:
    """Options of one search pass. See the module docstring for exact rules.

    Parameters
    ----------
    max_samples : maximum samples per target (width of the result arrays).
    min_samples : targets with fewer accepted samples are flagged
        unestimated.
    radius : search radius as anisotropic distance (= distance along the
        major axis), metres. ``None`` = the ellipsoid's ``major_radius``.
    max_per_hole : maximum accepted samples per drillhole (needs hole ids).
    sectors : ``None``, ``"quadrant"`` or ``"octant"``.
    max_per_sector : maximum accepted samples per sector (needs ``sectors``).
    min_sectors : minimum number of sectors with at least one accepted
        sample for the target to be estimated (needs ``sectors``).
    max_vertical : maximum ``|dz|`` in world Z between sample and target.
    exclude_zero_distance : drop samples at distance exactly 0 (the target
        itself in leave-one-out cross-validation; note that every sample
        coincident with the target is dropped, duplicates included).
    """

    max_samples: int = 24
    min_samples: int = 1
    radius: float | None = None
    max_per_hole: int | None = None
    sectors: str | None = None
    max_per_sector: int | None = None
    min_sectors: int | None = None
    max_vertical: float | None = None
    exclude_zero_distance: bool = False

    def __post_init__(self) -> None:
        if int(self.max_samples) < 1:
            raise ValueError("max_samples must be >= 1")
        if int(self.min_samples) < 0 or int(self.min_samples) > int(self.max_samples):
            raise ValueError("min_samples must be between 0 and max_samples")
        if self.radius is not None and not (np.isfinite(self.radius) and self.radius > 0):
            raise ValueError("radius must be a positive finite number")
        if self.max_per_hole is not None and int(self.max_per_hole) < 1:
            raise ValueError("max_per_hole must be >= 1")
        if self.sectors not in SECTOR_MODES:
            raise ValueError(f"sectors must be None, 'quadrant' or 'octant', got {self.sectors!r}")
        n_sectors = self.n_sectors
        if self.max_per_sector is not None:
            if n_sectors == 0:
                raise ValueError("max_per_sector needs sectors='quadrant' or 'octant'")
            if int(self.max_per_sector) < 1:
                raise ValueError("max_per_sector must be >= 1")
        if self.min_sectors is not None:
            if n_sectors == 0:
                raise ValueError("min_sectors needs sectors='quadrant' or 'octant'")
            if not 0 <= int(self.min_sectors) <= n_sectors:
                raise ValueError(f"min_sectors must be between 0 and {n_sectors}")
        if self.max_vertical is not None and not self.max_vertical >= 0:
            raise ValueError("max_vertical must be >= 0")

    @property
    def n_sectors(self) -> int:
        """0 (no sector search), 4 or 8."""
        bits = SECTOR_MODES[self.sectors]
        return 0 if bits == 0 else 2**bits


@dataclass(frozen=True)
class SearchResult:
    """Padded neighbour arrays for ``m`` targets, in the layout compiled
    estimators consume.

    Attributes
    ----------
    indices : (m, max_samples) int64. Row indices into the ``samples`` array
        given to :class:`SampleSearch`, nearest first (ties by index);
        unused slots are ``-1``.
    distances : (m, max_samples) float64. Anisotropic distance of each
        neighbour; unused slots are ``inf``.
    counts : (m,) int32. Number of valid leading entries in each row.
    sectors_informed : (m,) int8. Number of sectors with at least one
        accepted sample (0 when no sector search was requested).
    estimated : (m,) bool. ``counts >= min_samples`` and
        ``sectors_informed >= min_sectors``. Rows flagged ``False`` still
        list the samples that were found, for diagnostics; estimators must
        skip them.
    """

    indices: np.ndarray
    distances: np.ndarray
    counts: np.ndarray
    sectors_informed: np.ndarray
    estimated: np.ndarray


@njit(parallel=True, cache=True)
def _select(
    rows,
    cand_d,
    cand_i,
    exhaustive,
    radius,
    target_iso,
    target_z,
    sample_iso,
    sample_z,
    hole_codes,
    max_samples,
    max_per_hole,
    sector_bits,
    max_per_sector,
    max_vertical,
    exclude_zero,
    out_idx,
    out_dist,
    out_count,
    out_sectors,
    resolved,
):
    """Apply the selection rule to KD-tree candidates.

    ``cand_d`` / ``cand_i`` (r, K) are the K nearest samples of target
    ``rows[r]`` (sorted by distance, missing = index n / distance inf); they
    are re-ordered in place. Writes the accepted neighbours of each target
    into the ``out_*`` arrays at row ``rows[r]`` and sets ``resolved[r]``
    False when the K candidates were not enough to decide the result (the
    caller then retries with a larger K).
    """
    n_rows, n_cand = cand_d.shape
    n_samples = sample_iso.shape[0]
    for r in prange(n_rows):
        row = rows[r]
        # valid candidates: a leading run with d <= radius
        n_valid = 0
        for j in range(n_cand):
            if cand_i[r, j] < n_samples and cand_d[r, j] <= radius:
                n_valid += 1
            else:
                break
        # order ties by sample index (insertion sort; input is sorted by d)
        for j in range(1, n_valid):
            d = cand_d[r, j]
            i = cand_i[r, j]
            k = j - 1
            while k >= 0 and (cand_d[r, k] > d or (cand_d[r, k] == d and cand_i[r, k] > i)):
                cand_d[r, k + 1] = cand_d[r, k]
                cand_i[r, k + 1] = cand_i[r, k]
                k -= 1
            cand_d[r, k + 1] = d
            cand_i[r, k + 1] = i

        sector_count = np.zeros(8, dtype=np.int64)
        count = 0
        last = -1
        for j in range(n_valid):
            d = cand_d[r, j]
            i = cand_i[r, j]
            if exclude_zero and d == 0.0:
                continue
            if max_vertical >= 0.0 and abs(sample_z[i] - target_z[row]) > max_vertical:
                continue
            sector = 0
            if sector_bits > 0:
                if sample_iso[i, 0] - target_iso[row, 0] >= 0.0:
                    sector += 1
                if sample_iso[i, 1] - target_iso[row, 1] >= 0.0:
                    sector += 2
                if sector_bits == 3 and sample_iso[i, 2] - target_iso[row, 2] >= 0.0:
                    sector += 4
                if max_per_sector > 0 and sector_count[sector] >= max_per_sector:
                    continue
            if max_per_hole > 0:
                hole = hole_codes[i]
                same = 0
                for a in range(count):
                    if hole_codes[out_idx[row, a]] == hole:
                        same += 1
                if same >= max_per_hole:
                    continue
            out_idx[row, count] = i
            out_dist[row, count] = d
            sector_count[sector] += 1
            count += 1
            last = j
            if count == max_samples:
                break

        # Is the answer final? Yes if every sample within the radius was
        # among the candidates, or if the selection completed strictly
        # inside the candidate list's last distance (so no unseen sample
        # can tie with or precede anything that was examined).
        if exhaustive or n_valid < n_cand:
            resolved[r] = True
        elif count == max_samples and cand_d[r, last] < cand_d[r, n_cand - 1]:
            resolved[r] = True
        else:
            resolved[r] = False
            for a in range(count):  # reset; the row will be recomputed
                out_idx[row, a] = -1
                out_dist[row, a] = np.inf
            count = 0
            for s in range(8):
                sector_count[s] = 0
        out_count[row] = count
        informed = 0
        if sector_bits > 0:
            for s in range(8):
                if sector_count[s] > 0:
                    informed += 1
        out_sectors[row] = informed


class SampleSearch:
    """Neighbour search around many targets inside an anisotropic ellipsoid.

    Parameters
    ----------
    samples : (n, 3) float64 world coordinates (east, north, up) of the
        samples (composites). Must be finite.
    ellipsoid : orientation, major radius and ratios of the search
        ellipsoid.
    hole_ids : optional (n,) array of per-sample drillhole identifiers (any
        dtype ``numpy.unique`` accepts); required for ``max_per_hole``.
    leafsize : KD-tree leaf size.

    Attributes
    ----------
    samples : the (n, 3) coordinates as given (float64 copy).
    samples_iso : (n, 3) samples in isotropic space
        (``ellipsoid.to_isotropic(samples)``).
    hole_codes : (n,) int64 codes of the hole ids, or ``None``.
    tree : the ``cKDTree`` over ``samples_iso``.
    """

    def __init__(
        self,
        samples: Any,
        ellipsoid: Ellipsoid,
        hole_ids: Any = None,
        *,
        leafsize: int = 16,
    ) -> None:
        pts = np.array(samples, dtype=np.float64, order="C")
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError("samples must have shape (n, 3)")
        if not np.isfinite(pts).all():
            raise ValueError("samples contain non-finite coordinates")
        self.samples = pts
        self.ellipsoid = ellipsoid
        self.samples_iso = np.ascontiguousarray(ellipsoid.to_isotropic(pts))
        self._sample_z = np.ascontiguousarray(pts[:, 2])
        if hole_ids is None:
            self.hole_codes = None
        else:
            ids = np.asarray(hole_ids)
            if ids.shape != (len(pts),):
                raise ValueError("hole_ids must have one entry per sample")
            _, codes = np.unique(ids, return_inverse=True)
            self.hole_codes = np.ascontiguousarray(codes.reshape(-1), dtype=np.int64)
        self.tree = cKDTree(self.samples_iso, leafsize=leafsize) if len(pts) else None

    @property
    def n_samples(self) -> int:
        return self.samples.shape[0]

    def query(
        self,
        targets: Any,
        params: SearchParams | None = None,
        *,
        chunk_size: int = 200_000,
        workers: int = -1,
        **options: Any,
    ) -> SearchResult:
        """Find the neighbours of each target point.

        Parameters
        ----------
        targets : (m, 3) float64 world coordinates of the points to estimate
            (block centroids, discretisation points, sample locations for
            cross-validation). Must be finite.
        params : a :class:`SearchParams`; alternatively pass its fields as
            keyword arguments (``search.query(t, max_samples=12)``).
        chunk_size : targets processed per batch (memory control only; the
            result does not depend on it).
        workers : threads for the KD-tree query (``-1`` = all cores).

        Returns
        -------
        :class:`SearchResult` with arrays of shape ``(m, max_samples)``.
        """
        if params is None:
            params = SearchParams(**options)
        elif options:
            raise TypeError("pass either params or keyword options, not both")
        tg = np.ascontiguousarray(targets, dtype=np.float64)
        if tg.ndim != 2 or tg.shape[1] != 3:
            raise ValueError("targets must have shape (m, 3)")
        if not np.isfinite(tg).all():
            raise ValueError("targets contain non-finite coordinates")
        if params.max_per_hole is not None and self.hole_codes is None:
            raise ValueError("max_per_hole needs hole_ids given to SampleSearch")
        if chunk_size < 1:
            raise ValueError("chunk_size must be >= 1")

        m = tg.shape[0]
        n = self.n_samples
        max_samples = int(params.max_samples)
        radius = float(self.ellipsoid.major_radius if params.radius is None else params.radius)
        sector_bits = SECTOR_MODES[params.sectors]
        max_per_hole = 0 if params.max_per_hole is None else int(params.max_per_hole)
        max_per_sector = 0 if params.max_per_sector is None else int(params.max_per_sector)
        max_vertical = -1.0 if params.max_vertical is None else float(params.max_vertical)
        hole_codes = self.hole_codes if max_per_hole > 0 else np.zeros(0, dtype=np.int64)

        out_idx = np.full((m, max_samples), -1, dtype=np.int64)
        out_dist = np.full((m, max_samples), np.inf, dtype=np.float64)
        out_count = np.zeros(m, dtype=np.int32)
        out_sectors = np.zeros(m, dtype=np.int8)

        if n > 0 and m > 0:
            constrained = (
                max_per_hole > 0
                or max_per_sector > 0
                or max_vertical >= 0.0
                or params.exclude_zero_distance
            )
            # one spare candidate lets the kernel detect ties at the cut-off
            k_first = max_samples + 1 if not constrained else 3 * max_samples + 2
            # the tree's bound may be strict or inclusive; ask slightly wide
            # and let the kernel apply `d <= radius` exactly
            bound = np.nextafter(radius, np.inf)
            target_z = np.ascontiguousarray(tg[:, 2])
            for start in range(0, m, chunk_size):
                stop = min(m, start + chunk_size)
                target_iso = np.ascontiguousarray(self.ellipsoid.to_isotropic(tg[start:stop]))
                pending = np.arange(stop - start, dtype=np.int64)
                k = min(n, k_first)
                while pending.size:
                    still = []
                    step = max(1, _MAX_CANDIDATE_CELLS // k)
                    for a in range(0, pending.size, step):
                        rows = pending[a : a + step]
                        d, i = self.tree.query(
                            target_iso[rows], k=k, distance_upper_bound=bound, workers=workers
                        )
                        d = np.ascontiguousarray(d, dtype=np.float64).reshape(rows.size, k)
                        i = np.ascontiguousarray(i, dtype=np.int64).reshape(rows.size, k)
                        resolved = np.zeros(rows.size, dtype=np.bool_)
                        _select(
                            rows,
                            d,
                            i,
                            k >= n,
                            radius,
                            target_iso,
                            target_z[start:stop],
                            self.samples_iso,
                            self._sample_z,
                            hole_codes,
                            max_samples,
                            max_per_hole,
                            sector_bits,
                            max_per_sector,
                            max_vertical,
                            params.exclude_zero_distance,
                            out_idx[start:stop],
                            out_dist[start:stop],
                            out_count[start:stop],
                            out_sectors[start:stop],
                            resolved,
                        )
                        if not resolved.all():
                            still.append(rows[~resolved])
                    pending = np.concatenate(still) if still else np.zeros(0, dtype=np.int64)
                    k = min(n, k * 4)

        estimated = out_count >= int(params.min_samples)
        if params.min_sectors is not None:
            estimated &= out_sectors >= int(params.min_sectors)
        return SearchResult(out_idx, out_dist, out_count, out_sectors, estimated)
