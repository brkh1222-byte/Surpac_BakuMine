"""Validation of an estimated model (research.md section 9).

Everything here works on plain arrays. A "block model" is whatever the caller
has: an (m, 3) array of block centroids, an (m,) array of values and, if
wanted, an (m,) array of weights (block volume, or volume x density for
tonnage weighting). Nothing depends on a block-model or drillhole class.

Contents
--------
* :func:`swath_table` - mean of blocks and of samples in slices along X, Y,
  Z or any direction (research.md 9, check 3).
* :func:`global_comparison` - block mean against the nearest-neighbour
  (declustered) mean and the naive sample mean (check 2).
* :func:`cell_declustering_weights`, :func:`cell_declustered_mean`,
  :func:`cell_size_scan` - cell declustering, an estimate of the declustered
  mean that does not use the estimators of this package.
* :func:`cross_validate` - leave-one-out and leave-one-hole-out
  cross-validation of IDW and ordinary kriging, built on the existing
  estimators; :func:`error_summary` for the statistics.
* :func:`compare_to_truth` - model against a known truth (synthetic tests,
  check 6).

Missing values (NaN) are never treated as zero: a block or sample without a
value is left out of every mean and count.

Sources: Isaaks & Srivastava (1989), chapters 10 (declustering) and 15
(cross-validation); Deutsch & Journel (1998), the *GSLIB* book's description
of ``declus``. No third-party code was consulted or copied.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SampleSearch, SearchParams
from bakumine.core.estimation.variogram import VariogramModel

__all__ = [
    "CrossValidationResult",
    "DeclusterScan",
    "GlobalComparison",
    "TruthComparison",
    "cell_declustered_mean",
    "cell_declustering_weights",
    "cell_size_scan",
    "compare_to_truth",
    "cross_validate",
    "error_summary",
    "global_comparison",
    "swath_table",
]

_AXES = {"x": (1.0, 0.0, 0.0), "y": (0.0, 1.0, 0.0), "z": (0.0, 0.0, 1.0)}


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
def _coords(points: Any, name: str) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"{name} must have shape (n, 3)")
    return pts


def _values(values: Any, n: int, name: str) -> np.ndarray:
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != (n,):
        raise ValueError(f"{name} must have one entry per point")
    return vals


def _weights(weights: Any, n: int, name: str) -> np.ndarray:
    if weights is None:
        return np.ones(n)
    w = _values(weights, n, name)
    if (w[np.isfinite(w)] < 0.0).any():
        raise ValueError(f"{name} must be non-negative")
    return w


def _axis_vector(axis: Any) -> np.ndarray:
    """Unit vector of ``"x"``, ``"y"``, ``"z"`` or a 3-vector (east, north, up)."""
    if isinstance(axis, str):
        if axis.lower() not in _AXES:
            raise ValueError("axis must be 'x', 'y', 'z' or a 3-vector")
        return np.array(_AXES[axis.lower()])
    vec = np.asarray(axis, dtype=np.float64)
    if vec.shape != (3,) or not np.isfinite(vec).all() or not np.linalg.norm(vec) > 0.0:
        raise ValueError("axis must be 'x', 'y', 'z' or a non-zero 3-vector")
    return vec / np.linalg.norm(vec)


def _position(coords: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """Coordinate of each point along the axis. For a coordinate axis this is
    the coordinate column itself (no arithmetic, so no rounding)."""
    for column in range(3):
        if vector[column] == 1.0:
            return coords[:, column].copy()
    return coords[:, 0] * vector[0] + coords[:, 1] * vector[1] + coords[:, 2] * vector[2]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(weights.sum())
    return float((values * weights).sum() / total) if total > 0.0 else float("nan")


def _relative(a: float, b: float) -> float:
    return (a - b) / b if np.isfinite(a) and np.isfinite(b) and b != 0.0 else float("nan")


def _grouped(index: np.ndarray, values: np.ndarray, weights: np.ndarray) -> pd.DataFrame:
    """count, weight sum and weighted mean per slice index."""
    frame = pd.DataFrame({"slice": index, "w": weights, "wv": weights * values})
    out = frame.groupby("slice").agg(count=("w", "size"), weight=("w", "sum"), wv=("wv", "sum"))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["mean"] = np.where(out["weight"] > 0.0, out["wv"] / out["weight"], np.nan)
    return out.drop(columns="wv")


# --------------------------------------------------------------------------
# swath
# --------------------------------------------------------------------------
def swath_table(
    block_coords: Any,
    block_values: Any,
    sample_coords: Any = None,
    sample_values: Any = None,
    *,
    width: float,
    axis: Any = "x",
    origin: float | None = None,
    block_weights: Any = None,
    sample_weights: Any = None,
) -> pd.DataFrame:
    """Mean of blocks and of samples in slices along an axis.

    A point with position ``p`` along the axis (its x, y or z coordinate, or
    ``coords . unit_vector`` for an arbitrary direction) belongs to slice
    ``k = floor((p - origin) / width)``, which covers
    ``[origin + k width, origin + (k + 1) width)``.

    The block curve should follow the sample curve, smoother (research.md
    section 9). Where they part, look at the count columns first: slices
    with few samples are not informative.

    Parameters
    ----------
    block_coords, block_values : (m, 3) block centroids and (m,) values.
        Blocks with a NaN value (unestimated) are left out.
    sample_coords, sample_values : optional (n, 3) and (n,) samples
        (composites). Samples with a NaN value are left out.
    width : slice width, metres.
    axis : ``"x"`` (east), ``"y"`` (north), ``"z"`` (up) or a 3-vector
        (east, north, up) of any length for a rotated swath, e.g.
        ``ellipsoid.major_axis`` for slices along strike.
    origin : position of a slice boundary. Default: the largest multiple of
        ``width`` not above the smallest position of any block or sample
        used, so that slices are aligned to round coordinates.
    block_weights : optional (m,) weights for the block mean: block volume
        (needed when sub-blocks differ in size) or volume x density for a
        tonnage-weighted mean. Default equal weights.
    sample_weights : optional (n,) declustering weights for the sample mean.

    Returns
    -------
    ``pandas.DataFrame`` with one row per slice that holds at least one
    block or sample, in increasing order, and columns ``slice`` (the index
    ``k``), ``lower``, ``upper``, ``centre``, ``block_count``,
    ``block_weight`` (sum of weights), ``block_mean``, ``sample_count``,
    ``sample_weight``, ``sample_mean``. Means are NaN where a slice has no
    blocks (or samples) or zero total weight.
    """
    width = float(width)
    if not (np.isfinite(width) and width > 0.0):
        raise ValueError("width must be a positive number")
    vector = _axis_vector(axis)

    def prepare(coords: Any, values: Any, weights: Any, name: str):
        pts = _coords(coords, f"{name}_coords")
        vals = _values(values, pts.shape[0], f"{name}_values")
        w = _weights(weights, pts.shape[0], f"{name}_weights")
        ok = np.isfinite(vals) & np.isfinite(w)
        pos = _position(pts[ok], vector)
        if not np.isfinite(pos).all():
            raise ValueError(f"{name}s with a value have non-finite coordinates")
        return pos, vals[ok], w[ok]

    b_pos, b_val, b_w = prepare(block_coords, block_values, block_weights, "block")
    if sample_coords is None:
        s_pos, s_val, s_w = np.zeros(0), np.zeros(0), np.zeros(0)
    else:
        if sample_values is None:
            raise ValueError("sample_values is required with sample_coords")
        s_pos, s_val, s_w = prepare(sample_coords, sample_values, sample_weights, "sample")

    columns = [
        "slice", "lower", "upper", "centre", "block_count", "block_weight", "block_mean",
        "sample_count", "sample_weight", "sample_mean",
    ]  # fmt: skip
    everything = np.concatenate([b_pos, s_pos])
    if everything.size == 0:
        return pd.DataFrame(columns=columns)
    if origin is None:
        origin = float(np.floor(everything.min() / width) * width)
    origin = float(origin)

    blocks = _grouped(np.floor((b_pos - origin) / width).astype(np.int64), b_val, b_w)
    samples = _grouped(np.floor((s_pos - origin) / width).astype(np.int64), s_val, s_w)
    table = blocks.add_prefix("block_").join(samples.add_prefix("sample_"), how="outer")
    table = table.sort_index()
    for name in ("block_count", "sample_count"):
        table[name] = table[name].fillna(0).astype(np.int64)
    for name in ("block_weight", "sample_weight"):
        table[name] = table[name].fillna(0.0)
    k = table.index.to_numpy().astype(np.float64)
    table.insert(0, "centre", origin + (k + 0.5) * width)
    table.insert(0, "upper", origin + (k + 1.0) * width)
    table.insert(0, "lower", origin + k * width)
    table = table.reset_index(names="slice")
    return table[columns]


# --------------------------------------------------------------------------
# global means
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class GlobalComparison:
    """Global means and their relative differences ``(a - b) / b``.

    Attributes
    ----------
    block_mean : (weighted) mean of the estimated blocks.
    nn_mean : (weighted) mean of the nearest-neighbour model over the same
        blocks: the polygonal-declustered mean of the samples.
    naive_mean : plain mean of the samples.
    declustered_mean : weighted mean of the samples when ``sample_weights``
        were given (e.g. cell declustering), else NaN.
    n_blocks : blocks with both an estimate and an NN value.
    n_samples : samples with a value.
    block_vs_nn : ``(block_mean - nn_mean) / nn_mean``. This is the check of
        research.md section 9: it should be within a few percent.
    block_vs_naive, nn_vs_naive : the same against the naive mean; a large
        ``nn_vs_naive`` only says that the drilling is clustered.
    block_vs_declustered, nn_vs_declustered : against ``declustered_mean``.
    """

    block_mean: float
    nn_mean: float
    naive_mean: float
    declustered_mean: float
    n_blocks: int
    n_samples: int
    block_vs_nn: float
    block_vs_naive: float
    nn_vs_naive: float
    block_vs_declustered: float
    nn_vs_declustered: float


def global_comparison(
    block_values: Any,
    nn_values: Any,
    sample_values: Any,
    *,
    block_weights: Any = None,
    sample_weights: Any = None,
) -> GlobalComparison:
    """Block mean against the NN-declustered mean and the naive sample mean.

    Parameters
    ----------
    block_values : (m,) estimated block values (IDW, kriging, ...).
    nn_values : (m,) nearest-neighbour values of the *same* blocks
        (``nearest_neighbour(...).estimate``). Only blocks where both are
        finite are compared, so the two means cover the same volume.
    sample_values : (n,) sample (composite) values; NaN left out.
    block_weights : optional (m,) block volumes or tonnages.
    sample_weights : optional (n,) declustering weights
        (:func:`cell_declustering_weights`), reported as an independent
        declustered mean.
    """
    est = np.asarray(block_values, dtype=np.float64)
    if est.ndim != 1:
        raise ValueError("block_values must be one-dimensional")
    nn = _values(nn_values, est.size, "nn_values")
    bw = _weights(block_weights, est.size, "block_weights")
    both = np.isfinite(est) & np.isfinite(nn) & np.isfinite(bw)
    block_mean = _weighted_mean(est[both], bw[both])
    nn_mean = _weighted_mean(nn[both], bw[both])

    smp = np.asarray(sample_values, dtype=np.float64)
    if smp.ndim != 1:
        raise ValueError("sample_values must be one-dimensional")
    ok = np.isfinite(smp)
    naive = float(smp[ok].mean()) if ok.any() else float("nan")
    declustered = float("nan")
    if sample_weights is not None:
        sw = _weights(sample_weights, smp.size, "sample_weights")
        use = ok & np.isfinite(sw)
        declustered = _weighted_mean(smp[use], sw[use])
    return GlobalComparison(
        block_mean=block_mean,
        nn_mean=nn_mean,
        naive_mean=naive,
        declustered_mean=declustered,
        n_blocks=int(both.sum()),
        n_samples=int(ok.sum()),
        block_vs_nn=_relative(block_mean, nn_mean),
        block_vs_naive=_relative(block_mean, naive),
        nn_vs_naive=_relative(nn_mean, naive),
        block_vs_declustered=_relative(block_mean, declustered),
        nn_vs_declustered=_relative(nn_mean, declustered),
    )


# --------------------------------------------------------------------------
# cell declustering
# --------------------------------------------------------------------------
def _cell_size(cell_size: Any) -> np.ndarray:
    size = np.asarray(cell_size, dtype=np.float64)
    if size.ndim == 0:
        size = np.full(3, float(size))
    if size.shape != (3,) or not (np.isfinite(size).all() and (size > 0.0).all()):
        raise ValueError("cell_size must be a positive number or three of them")
    return size


def cell_declustering_weights(coords: Any, cell_size: Any, *, n_offsets: int = 8) -> np.ndarray:
    """Cell-declustering weights (mean 1).

    A grid of cells of ``cell_size`` is laid over the samples. In a cell
    holding ``n_c`` samples each sample gets ``1 / n_c``, so every occupied
    cell carries the same total weight: samples in densely drilled areas are
    down-weighted. The weights are scaled to sum to the number of samples.

    The result depends on where the grid starts, so it is averaged over
    ``n_offsets`` grids: grid ``o`` has its origin at ``min(coords) -
    cell_size * (o / n_offsets)`` (shifted along the cell diagonal), with a
    sample at ``x`` in cell ``floor((x - origin) / cell_size)``.

    Parameters
    ----------
    coords : (n, 3) sample coordinates, all finite.
    cell_size : one edge length (cubic cells) or ``(dx, dy, dz)``. Use flat
        or elongated cells where the drilling pattern is anisotropic.
    n_offsets : number of grid origins averaged (>= 1; 1 = a single grid
        whose corner is the minimum coordinate).

    Returns
    -------
    (n,) float64 weights with mean exactly 1 (sum ``n``).
    """
    pts = _coords(coords, "coords")
    if not np.isfinite(pts).all():
        raise ValueError("coords contain non-finite values")
    size = _cell_size(cell_size)
    n_offsets = int(n_offsets)
    if n_offsets < 1:
        raise ValueError("n_offsets must be >= 1")
    n = pts.shape[0]
    if n == 0:
        return np.zeros(0)
    low = pts.min(axis=0)
    total = np.zeros(n)
    for o in range(n_offsets):
        origin = low - size * (o / n_offsets)
        cells = np.floor((pts - origin) / size).astype(np.int64)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        inverse = np.asarray(inverse).reshape(-1)
        total += n / (counts.size * counts[inverse])
    return total / n_offsets


def cell_declustered_mean(
    coords: Any, values: Any, cell_size: Any, *, n_offsets: int = 8
) -> float:
    """Cell-declustered mean of ``values``; samples with a NaN value are
    left out before the weights are computed."""
    pts = _coords(coords, "coords")
    vals = _values(values, pts.shape[0], "values")
    ok = np.isfinite(vals)
    if not ok.any():
        return float("nan")
    weights = cell_declustering_weights(pts[ok], cell_size, n_offsets=n_offsets)
    return _weighted_mean(vals[ok], weights)


@dataclass(frozen=True)
class DeclusterScan:
    """Declustered mean for a series of cell sizes.

    Attributes
    ----------
    cell_sizes : (k,) the scanned sizes (the x edge; other edges follow
        ``anisotropy``).
    means : (k,) declustered mean for each size.
    naive_mean : the equal-weight mean (the limit for very small and for
        very large cells).
    best_size, best_mean : the size giving the lowest (``choose="min"``) or
        highest (``"max"``) declustered mean, and that mean.
    weights : (n,) weights for ``best_size`` (mean 1 over the samples with a
        value; NaN for samples without one).
    """

    cell_sizes: np.ndarray
    means: np.ndarray
    naive_mean: float
    best_size: float
    best_mean: float
    weights: np.ndarray


def cell_size_scan(
    coords: Any,
    values: Any,
    cell_sizes: Any,
    *,
    anisotropy: Any = (1.0, 1.0, 1.0),
    n_offsets: int = 8,
    choose: str = "min",
) -> DeclusterScan:
    """Declustered mean against cell size, and the size that moves it most.

    The usual rule: when the high-grade areas are the densely drilled ones
    (the normal case) take the cell size with the *lowest* declustered mean
    (``choose="min"``); when low-grade areas were drilled more densely, the
    highest (``"max"``). A natural alternative is simply the spacing of the
    sparse, regular part of the drilling.

    Parameters
    ----------
    cell_sizes : sequence of sizes to try (metres).
    anisotropy : cell shape; a size ``s`` means cells of ``s * anisotropy``
        (dx, dy, dz).
    """
    if choose not in ("min", "max"):
        raise ValueError("choose must be 'min' or 'max'")
    pts = _coords(coords, "coords")
    vals = _values(values, pts.shape[0], "values")
    sizes = np.atleast_1d(np.asarray(cell_sizes, dtype=np.float64))
    if sizes.ndim != 1 or sizes.size == 0 or not (np.isfinite(sizes).all() and (sizes > 0).all()):
        raise ValueError("cell_sizes must be positive numbers")
    shape = _cell_size(anisotropy)
    ok = np.isfinite(vals)
    if not ok.any():
        raise ValueError("no sample has a value")
    means = np.array(
        [cell_declustered_mean(pts[ok], vals[ok], s * shape, n_offsets=n_offsets) for s in sizes]
    )
    best = int(np.argmin(means) if choose == "min" else np.argmax(means))
    weights = np.full(vals.shape, np.nan)
    weights[ok] = cell_declustering_weights(pts[ok], sizes[best] * shape, n_offsets=n_offsets)
    return DeclusterScan(
        cell_sizes=sizes,
        means=means,
        naive_mean=float(vals[ok].mean()),
        best_size=float(sizes[best]),
        best_mean=float(means[best]),
        weights=weights,
    )


# --------------------------------------------------------------------------
# error statistics
# --------------------------------------------------------------------------
def error_summary(
    true: Any, estimate: Any, variance: Any = None, *, weights: Any = None
) -> dict[str, float]:
    """Summary statistics of estimates against true (or held-out) values.

    Pairs where either value is NaN are left out. With ``e = estimate -
    true`` (positive = over-estimation):

    ``n``
        number of pairs.
    ``mean_true``, ``mean_estimate``
        the two means.
    ``mean_error`` (ME)
        mean of ``e``; ``relative_bias`` = ``mean_error / mean_true``.
    ``rmse``, ``mae``
        root mean squared error and mean absolute error.
    ``correlation``
        Pearson correlation of estimate and true.
    ``slope``, ``intercept``
        regression of the true value on the estimate, ``true ~ intercept +
        slope * estimate``: ``slope = cov(true, estimate) / var(estimate)``.
        A slope below 1 means conditional bias: high estimates are too high
        and low estimates too low.
    ``mean_std_error``, ``var_std_error``
        only with ``variance`` (kriging): mean and (population) variance of
        the standardised errors ``e / sqrt(variance)`` over the pairs with
        a positive variance. Near 0 and near 1 when the variogram model is
        consistent with the data. ``n_std_error`` is their number.

    ``weights`` (optional, e.g. block volumes) make every mean a weighted
    mean.
    """
    t = np.asarray(true, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    if est.shape != t.shape:
        raise ValueError("true and estimate must have the same length")
    w = _weights(weights, t.size, "weights")
    ok = np.isfinite(t) & np.isfinite(est) & np.isfinite(w)
    out: dict[str, float] = {"n": int(ok.sum())}
    names = (
        "mean_true", "mean_estimate", "mean_error", "relative_bias", "rmse", "mae",
        "correlation", "slope", "intercept",
    )  # fmt: skip
    total = float(w[ok].sum())
    if not total > 0.0:
        out.update({name: float("nan") for name in names})
    else:
        tt, ee, ww = t[ok], est[ok], w[ok] / total
        err = ee - tt
        mean_t = float((ww * tt).sum())
        mean_e = float((ww * ee).sum())
        var_t = float((ww * (tt - mean_t) ** 2).sum())
        var_e = float((ww * (ee - mean_e) ** 2).sum())
        cov = float((ww * (tt - mean_t) * (ee - mean_e)).sum())
        slope = cov / var_e if var_e > 0.0 else float("nan")
        defined = var_t > 0.0 and var_e > 0.0
        correlation = float(cov / np.sqrt(var_t * var_e)) if defined else float("nan")
        out.update(
            mean_true=mean_t,
            mean_estimate=mean_e,
            mean_error=mean_e - mean_t,
            relative_bias=_relative(mean_e, mean_t),
            rmse=float(np.sqrt((ww * err**2).sum())),
            mae=float((ww * np.abs(err)).sum()),
            correlation=correlation,
            slope=slope,
            intercept=mean_t - slope * mean_e,
        )
    if variance is not None:
        var = _values(variance, t.size, "variance")
        use = ok & np.isfinite(var) & (var > 0.0)
        out["n_std_error"] = int(use.sum())
        if use.any() and float(w[use].sum()) > 0.0:
            std = (est[use] - t[use]) / np.sqrt(var[use])
            ww = w[use] / w[use].sum()
            mean_std = float((ww * std).sum())
            out["mean_std_error"] = mean_std
            out["var_std_error"] = float((ww * (std - mean_std) ** 2).sum())
        else:
            out["mean_std_error"] = float("nan")
            out["var_std_error"] = float("nan")
    return out


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class CrossValidationResult:
    """Cross-validation estimates at the sample locations (length ``n`` =
    the caller's samples; NaN / -1 for samples without a value or not
    estimated).

    Attributes
    ----------
    true : (n,) the sample values.
    estimate : (n,) estimate of each sample from the others.
    error : (n,) ``estimate - true`` (positive = over-estimation).
    variance : (n,) kriging variance (NaN for IDW).
    std_error : (n,) ``error / sqrt(variance)`` (kriging; NaN where the
        variance is not positive, and for IDW).
    n_samples : (n,) int32 number of neighbours the search selected.
    neighbours : (n, max_samples) int64 rows of the caller's sample arrays
        selected as neighbours of each sample, nearest first, ``-1``
        padding. Obtained by running the estimator's own search
        (:class:`SampleSearch` with the same samples, ellipsoid, hole ids
        and parameters) once more; the estimators do not return their
        neighbour lists, except kriging through ``return_weights_for``,
        against which the tests check this field.
    estimated : (n,) bool.
    method : ``"idw"`` or ``"ok"``; leave : ``"one"`` or ``"hole"``.
    summary : :func:`error_summary` of the estimated samples.
    """

    true: np.ndarray
    estimate: np.ndarray
    error: np.ndarray
    variance: np.ndarray
    std_error: np.ndarray
    n_samples: np.ndarray
    neighbours: np.ndarray
    estimated: np.ndarray
    method: str
    leave: str
    summary: dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        """Per-sample table (without the neighbour lists)."""
        return pd.DataFrame(
            {
                "true": self.true,
                "estimate": self.estimate,
                "error": self.error,
                "variance": self.variance,
                "std_error": self.std_error,
                "n_samples": self.n_samples,
                "estimated": self.estimated,
            }
        )


def cross_validate(
    coords: Any,
    values: Any,
    ellipsoid: Ellipsoid,
    params: SearchParams | None = None,
    *,
    method: str = "ok",
    model: VariogramModel | None = None,
    power: float = 2.0,
    leave: str = "one",
    hole_ids: Any = None,
    **options: Any,
) -> CrossValidationResult:
    """Cross-validate IDW or ordinary kriging at the sample locations.

    Each sample is estimated (point support) from the other samples with the
    existing estimators (:func:`inverse_distance`, :func:`krige`) and the
    given search, then compared with its own value.

    ``leave="one"`` (leave-one-out)
        The estimator runs once with the samples as targets and the search
        option ``exclude_zero_distance=True``, which drops the sample itself.
        As documented in ``search.py``, it drops *every* sample at distance
        exactly 0, so exact duplicates of a sample do not help to estimate
        it. Along a drillhole the neighbours 1-2 m up and down the hole
        remain and dominate the estimate, so leave-one-out flatters any
        estimator on drillhole data.

    ``leave="hole"`` (leave-one-hole-out; needs ``hole_ids``)
        More honest for drillholes: all samples of a hole are estimated from
        the samples of the *other* holes only. Implemented by running the
        estimator once per hole with that hole's samples removed from the
        sample set (nothing was added to ``search.py``); cost is one
        estimator call per hole. ``exclude_zero_distance`` is left as given
        in ``params``.

    Parameters
    ----------
    coords, values : (n, 3) sample coordinates and (n,) values. Samples
        with a NaN value take no part (neither estimated nor used).
    ellipsoid : search ellipsoid (and distance anisotropy for IDW).
    params : search options (default ``SearchParams()``).
    method : ``"idw"`` or ``"ok"`` (ordinary kriging; needs ``model``).
    model : variogram model for kriging.
    power : IDW power.
    leave : ``"one"`` or ``"hole"``.
    hole_ids : optional (n,) drillhole ids (required for ``leave="hole"``
        and for ``max_per_hole``).
    options : passed on to the estimator (e.g. ``workers``,
        ``negative_weight_correction``); block-support options are not
        accepted, cross-validation is at point support.

    Returns
    -------
    :class:`CrossValidationResult`
    """
    if method not in ("idw", "ok"):
        raise ValueError("method must be 'idw' or 'ok'")
    if leave not in ("one", "hole"):
        raise ValueError("leave must be 'one' or 'hole'")
    if method == "ok" and not isinstance(model, VariogramModel):
        raise ValueError("ordinary kriging needs a VariogramModel")
    for name in ("block_size", "discretisation", "grid_rotation", "method", "mean"):
        if name in options:
            raise TypeError(f"cross_validate does not accept {name!r}")
    params = SearchParams() if params is None else params
    pts = _coords(coords, "coords")
    vals = _values(values, pts.shape[0], "values")
    n = pts.shape[0]
    holes = None
    if hole_ids is not None:
        holes = np.asarray(hole_ids)
        if holes.shape != (n,):
            raise ValueError("hole_ids must have one entry per sample")
    if leave == "hole" and holes is None:
        raise ValueError("leave='hole' needs hole_ids")

    kept = np.flatnonzero(np.isfinite(vals))
    c = np.ascontiguousarray(pts[kept])
    v = vals[kept]
    h = None if holes is None else holes[kept]
    if not np.isfinite(c).all():
        raise ValueError("samples with a value have non-finite coordinates")
    k = kept.size
    width = int(params.max_samples)
    est = np.full(k, np.nan)
    var = np.full(k, np.nan)
    count = np.zeros(k, dtype=np.int32)
    done = np.zeros(k, dtype=np.bool_)
    neigh = np.full((k, width), -1, dtype=np.int64)

    def run(rows: np.ndarray, source: np.ndarray, search_params: SearchParams) -> None:
        """Estimate samples ``rows`` from samples ``source`` (indices into the
        kept arrays)."""
        if rows.size == 0 or source.size == 0:
            return
        src_c, src_v = c[source], v[source]
        src_h = None if h is None else h[source]
        targets = c[rows]
        if method == "idw":
            res = inverse_distance(
                src_c, src_v, targets, ellipsoid, search_params,
                power=power, hole_ids=src_h, **options,
            )  # fmt: skip
        else:
            res = krige(
                src_c, src_v, targets, model, ellipsoid, search_params,
                method="ordinary", hole_ids=src_h, **options,
            )  # fmt: skip
            var[rows] = res.variance
        est[rows] = res.estimate
        count[rows] = res.n_samples
        done[rows] = res.estimated
        found = SampleSearch(src_c, ellipsoid, src_h).query(targets, search_params)
        idx = found.indices
        neigh[rows] = np.where(idx >= 0, kept[source[np.maximum(idx, 0)]], -1)

    everything = np.arange(k, dtype=np.int64)
    if leave == "one":
        run(everything, everything, replace(params, exclude_zero_distance=True))
    else:
        _, codes = np.unique(h, return_inverse=True)
        codes = np.asarray(codes).reshape(-1)
        for code in range(int(codes.max()) + 1 if k else 0):
            inside = codes == code
            run(everything[inside], everything[~inside], params)

    def spread(values_kept: np.ndarray, fill: Any, dtype: Any) -> np.ndarray:
        out = np.full((n,) + values_kept.shape[1:], fill, dtype=dtype)
        out[kept] = values_kept
        return out

    est = np.where(done, est, np.nan)
    var = np.where(done, var, np.nan)
    error = est - v
    with np.errstate(divide="ignore", invalid="ignore"):
        std = np.where(var > 0.0, error / np.sqrt(var), np.nan)
    summary = error_summary(v, est, var if method == "ok" else None)
    return CrossValidationResult(
        true=vals.copy(),
        estimate=spread(est, np.nan, np.float64),
        error=spread(error, np.nan, np.float64),
        variance=spread(var, np.nan, np.float64),
        std_error=spread(std, np.nan, np.float64),
        n_samples=spread(count, 0, np.int32),
        neighbours=spread(neigh, -1, np.int64),
        estimated=spread(done, False, np.bool_),
        method=method,
        leave=leave,
        summary=summary,
    )


# --------------------------------------------------------------------------
# model against truth
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class TruthComparison:
    """Estimated model against known true values (synthetic tests).

    Attributes
    ----------
    n : blocks with both an estimate and a truth.
    mean_estimate, mean_truth : (weighted) means.
    bias : ``mean_estimate - mean_truth``; relative_bias : ``bias /
        mean_truth``.
    rmse, mae : (weighted) root mean squared and mean absolute error.
    correlation : Pearson correlation of estimate and truth.
    slope : slope of the regression of truth on estimate.
    by_slice : ``None``, or a table with one row per slice (see
        :func:`swath_table` for the slicing) and columns ``slice, lower,
        upper, centre, count, weight, estimate_mean, truth_mean, bias, rmse,
        mae``.
    """

    n: int
    mean_estimate: float
    mean_truth: float
    bias: float
    relative_bias: float
    rmse: float
    mae: float
    correlation: float
    slope: float
    by_slice: pd.DataFrame | None = None


def compare_to_truth(
    estimate: Any,
    truth: Any,
    *,
    weights: Any = None,
    coords: Any = None,
    axis: Any = "z",
    width: float | None = None,
    origin: float | None = None,
) -> TruthComparison:
    """Compare estimated block values with the true ones.

    Parameters
    ----------
    estimate, truth : (m,) arrays; blocks where either is NaN are left out.
    weights : optional (m,) block volumes or tonnages.
    coords, axis, width, origin : give ``coords`` (m, 3) and ``width`` to
        also get the errors by slice along ``axis`` (as in
        :func:`swath_table`): where in the deposit the model is biased.
    """
    est = np.asarray(estimate, dtype=np.float64)
    if est.ndim != 1:
        raise ValueError("estimate must be one-dimensional")
    tru = _values(truth, est.size, "truth")
    w = _weights(weights, est.size, "weights")
    stats = error_summary(tru, est, weights=w)
    by_slice = None
    if (coords is None) != (width is None):
        raise ValueError("by-slice errors need both coords and width")
    if coords is not None:
        width = float(width)  # type: ignore[arg-type]
        if not (np.isfinite(width) and width > 0.0):
            raise ValueError("width must be a positive number")
        pts = _coords(coords, "coords")
        if pts.shape[0] != est.size:
            raise ValueError("coords must have one row per block")
        ok = np.isfinite(est) & np.isfinite(tru) & np.isfinite(w)
        pos = _position(pts[ok], _axis_vector(axis))
        if origin is None:
            origin = float(np.floor(pos.min() / width) * width) if pos.size else 0.0
        index = np.floor((pos - origin) / width).astype(np.int64)
        e, t, ww = est[ok], tru[ok], w[ok]
        frame = pd.DataFrame(
            {
                "slice": index,
                "w": ww,
                "we": ww * e,
                "wt": ww * t,
                "wsq": ww * (e - t) ** 2,
                "wabs": ww * np.abs(e - t),
            }
        )
        g = frame.groupby("slice").agg(
            count=("w", "size"), weight=("w", "sum"), we=("we", "sum"), wt=("wt", "sum"),
            wsq=("wsq", "sum"), wabs=("wabs", "sum"),
        )  # fmt: skip
        k = g.index.to_numpy().astype(np.float64)
        with np.errstate(divide="ignore", invalid="ignore"):
            total = g["weight"].to_numpy()
            total = np.where(total > 0.0, total, np.nan)
            est_mean = g["we"].to_numpy() / total
            tru_mean = g["wt"].to_numpy() / total
            by_slice = pd.DataFrame(
                {
                    "slice": g.index.to_numpy(),
                    "lower": origin + k * width,
                    "upper": origin + (k + 1.0) * width,
                    "centre": origin + (k + 0.5) * width,
                    "count": g["count"].to_numpy(),
                    "weight": g["weight"].to_numpy(),
                    "estimate_mean": est_mean,
                    "truth_mean": tru_mean,
                    "bias": est_mean - tru_mean,
                    "rmse": np.sqrt(g["wsq"].to_numpy() / total),
                    "mae": g["wabs"].to_numpy() / total,
                }
            )
    return TruthComparison(
        n=int(stats["n"]),
        mean_estimate=stats["mean_estimate"],
        mean_truth=stats["mean_true"],
        bias=stats["mean_error"],
        relative_bias=stats["relative_bias"],
        rmse=stats["rmse"],
        mae=stats["mae"],
        correlation=stats["correlation"],
        slope=stats["slope"],
        by_slice=by_slice,
    )
