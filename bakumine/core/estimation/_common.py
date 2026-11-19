"""Helpers shared by the estimators (nearest neighbour, IDW, kriging).

Plain arrays only; nothing here depends on block-model or drillhole classes.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import numpy as np

from bakumine.core.estimation.search import SearchResult

__all__ = [
    "CleanSamples",
    "as_targets",
    "block_size_array",
    "clean_samples",
    "discretisation_lattice",
    "distance_summaries",
    "grid_rotation_matrix",
]


class CleanSamples(NamedTuple):
    """Samples with a usable value.

    coords : (n_kept, 3) float64, C-contiguous
    values : (n_kept,) float64
    hole_ids : (n_kept,) array or ``None``
    kept : (n_kept,) int64, row of each kept sample in the caller's arrays
        (ascending, so index order - and with it the search's tie-break - is
        the caller's)
    n_excluded : number of samples dropped because their value was NaN or
        infinite
    """

    coords: np.ndarray
    values: np.ndarray
    hole_ids: np.ndarray | None
    kept: np.ndarray
    n_excluded: int


def clean_samples(samples: Any, values: Any, hole_ids: Any = None) -> CleanSamples:
    """Validate sample arrays and drop samples without a value.

    A sample whose value is NaN (a missing assay) or infinite is removed
    *before* the search, so it can neither be selected nor take a slot from a
    valid sample. Missing stays missing: it is never treated as zero.
    Non-finite coordinates of a sample that has a value are an error.
    """
    pts = np.asarray(samples, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("samples must have shape (n, 3)")
    vals = np.asarray(values, dtype=np.float64)
    if vals.shape != (pts.shape[0],):
        raise ValueError("values must have one entry per sample")
    holes = None
    if hole_ids is not None:
        holes = np.asarray(hole_ids)
        if holes.shape != (pts.shape[0],):
            raise ValueError("hole_ids must have one entry per sample")
    ok = np.isfinite(vals)
    kept = np.flatnonzero(ok).astype(np.int64)
    pts = np.ascontiguousarray(pts[ok])
    if not np.isfinite(pts).all():
        raise ValueError("samples with a value have non-finite coordinates")
    return CleanSamples(
        pts,
        np.ascontiguousarray(vals[ok]),
        None if holes is None else holes[ok],
        kept,
        int(vals.size - kept.size),
    )


def as_targets(targets: Any) -> np.ndarray:
    """(m, 3) float64 C-contiguous target coordinates, validated."""
    tg = np.ascontiguousarray(targets, dtype=np.float64)
    if tg.ndim != 2 or tg.shape[1] != 3:
        raise ValueError("targets must have shape (m, 3)")
    if not np.isfinite(tg).all():
        raise ValueError("targets contain non-finite coordinates")
    return tg


def discretisation_lattice(discretisation: Any) -> np.ndarray:
    """Unit discretisation lattice of a block.

    ``discretisation = (nx, ny, nz)`` splits the block into ``nx * ny * nz``
    equal cells; the points are the cell centres, in units of the block
    dimensions and relative to the block centroid::

        u = (i + 0.5) / nx - 0.5,   i = 0 .. nx - 1      (same for y, z)

    so every coordinate lies strictly inside (-0.5, 0.5) and ``(1, 1, 1)``
    is the centroid alone. Returns (nx * ny * nz, 3) float64; x varies
    fastest. The world position of point ``p`` of a block with centroid ``c``,
    dimensions ``s`` and grid rotation ``G`` is ``c + G @ (u[p] * s)``.
    """
    try:
        nx, ny, nz = (int(v) for v in discretisation)
    except (TypeError, ValueError):
        raise ValueError("discretisation must be three integers (nx, ny, nz)") from None
    if min(nx, ny, nz) < 1:
        raise ValueError("discretisation counts must be >= 1")
    ax = [(np.arange(n) + 0.5) / n - 0.5 for n in (nx, ny, nz)]
    zz, yy, xx = np.meshgrid(ax[2], ax[1], ax[0], indexing="ij")
    return np.ascontiguousarray(np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()]))


def block_size_array(block_size: Any, m: int) -> np.ndarray:
    """Block dimensions as an (m, 3) or (1, 3) float64 array.

    ``block_size`` is one common ``(dx, dy, dz)`` (returned with shape
    (1, 3)) or per-target dimensions of shape (m, 3). Dimensions are the full
    edge lengths along the block's own x, y, z axes, in metres, and must be
    positive.
    """
    size = np.asarray(block_size, dtype=np.float64)
    if size.shape == (3,):
        size = size.reshape(1, 3)
    elif size.shape != (m, 3):
        raise ValueError("block_size must be (dx, dy, dz) or an array of shape (m, 3)")
    if not (np.isfinite(size).all() and (size > 0.0).all()):
        raise ValueError("block dimensions must be positive finite numbers")
    return np.ascontiguousarray(size)


def grid_rotation_matrix(grid_rotation: Any) -> np.ndarray:
    """Validated (3, 3) block-grid rotation matrix (identity for ``None``).

    Columns are the world (east, north, up) directions of the block's local
    x, y, z edges: ``world_offset = G @ local_offset``.
    """
    if grid_rotation is None:
        return np.eye(3)
    g = np.array(grid_rotation, dtype=np.float64)
    if g.shape != (3, 3) or not np.isfinite(g).all():
        raise ValueError("grid_rotation must be a finite 3x3 matrix")
    if not np.allclose(g.T @ g, np.eye(3), atol=1e-9):
        raise ValueError("grid_rotation is not orthonormal")
    return np.ascontiguousarray(g)


def distance_summaries(result: SearchResult) -> tuple[np.ndarray, np.ndarray]:
    """Mean and smallest anisotropic distance of each target's neighbours.

    Returns ``(mean_distance, nearest_distance)``, both (m,) float64 and NaN
    for targets without any neighbour.
    """
    d = result.distances
    valid = np.isfinite(d)
    counts = result.counts.astype(np.float64)
    total = np.where(valid, d, 0.0).sum(axis=1)
    has = result.counts > 0
    mean = np.full(d.shape[0], np.nan)
    np.divide(total, counts, out=mean, where=has)
    nearest = np.where(has, d[:, 0], np.nan) if d.shape[1] else np.full(d.shape[0], np.nan)
    return mean, nearest
