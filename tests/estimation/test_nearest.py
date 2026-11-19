"""Acceptance tests for P4.T3 nearest neighbour (research.md 7.2)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import cKDTree

from bakumine.core.estimation.nearest import NearestResult, nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid

SYNTH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"


def brute_nearest(samples, values, targets, ellipsoid, radius=None, max_vertical=None):
    """Reference written from research.md 7.5: explicit axis projections,
    stretch, full distance matrix; first minimum = lowest index on ties.
    Shares no code with nearest.py or search.py."""
    radius = ellipsoid.major_radius if radius is None else radius
    major, semi, minor = ellipsoid.major_axis, ellipsoid.semi_major_axis, ellipsoid.minor_axis
    est = np.full(len(targets), np.nan)
    idx = np.full(len(targets), -1, dtype=np.int64)
    dist = np.full(len(targets), np.nan)
    for t, target in enumerate(targets):
        diff = samples - target
        u, v, w = diff @ major, diff @ semi, diff @ minor
        d = np.sqrt(u**2 + (v * ellipsoid.semi_major_ratio) ** 2 + (w * ellipsoid.minor_ratio) ** 2)
        d[np.isnan(values)] = np.inf
        if max_vertical is not None:
            d[np.abs(diff[:, 2]) > max_vertical] = np.inf
        i = int(np.argmin(d))  # first occurrence of the minimum
        if d[i] <= radius:
            est[t], idx[t], dist[t] = values[i], i, d[i]
    return est, idx, dist


@pytest.mark.parametrize(
    "ellipsoid",
    [
        Ellipsoid(20.0),
        Ellipsoid(60.0, 2.0, 5.0, angles=(30.0, 20.0, 10.0)),
        Ellipsoid(40.0, 1.5, 3.0, angles=(250.0, -40.0, 75.0), convention="gslib"),
    ],
)
@pytest.mark.parametrize("max_vertical", [None, 12.0])
def test_equals_brute_force(ellipsoid, max_vertical):
    rng = np.random.default_rng(3)
    samples = rng.uniform(0.0, 200.0, (600, 3))
    values = rng.lognormal(0.0, 1.0, 600)
    values[rng.choice(600, 40, replace=False)] = np.nan  # missing assays
    targets = rng.uniform(-20.0, 220.0, (3000, 3))

    res = nearest_neighbour(
        samples, values, targets, ellipsoid, max_vertical=max_vertical, chunk_size=700
    )
    est, idx, dist = brute_nearest(samples, values, targets, ellipsoid, max_vertical=max_vertical)

    assert isinstance(res, NearestResult)
    assert res.n_samples_excluded == 40
    np.testing.assert_array_equal(res.sample_index, idx)
    np.testing.assert_array_equal(res.estimate, est)  # NaN where unestimated
    np.testing.assert_allclose(res.nearest_distance, dist, rtol=1e-12, atol=1e-12)
    np.testing.assert_array_equal(res.mean_distance, res.nearest_distance)
    np.testing.assert_array_equal(res.estimated, idx >= 0)
    np.testing.assert_array_equal(res.n_samples, (idx >= 0).astype(np.int32))
    # the case must exercise both outcomes, and never pick a NaN sample
    assert 0 < res.estimated.sum() < len(targets)
    assert not np.isnan(res.estimate[res.estimated]).any()


def test_ties_are_deterministic_lowest_index():
    # targets on the integer lattice, samples on the half-odd lattice: every
    # target has 8 samples at exactly the same distance
    g = np.arange(0.0, 6.0)
    samples = np.array([[x, y, z] for x in g for y in g for z in g]) + 0.5
    rng = np.random.default_rng(0)
    order = rng.permutation(len(samples))
    samples = samples[order]
    values = np.arange(len(samples), dtype=float)
    g = np.arange(1.0, 6.0)
    targets = np.array([[x, y, z] for x in g for y in g for z in g])
    ell = Ellipsoid(10.0)

    res = nearest_neighbour(samples, values, targets, ell)
    d = np.linalg.norm(samples[None] - targets[:, None], axis=2)
    tied = np.isclose(d, np.sqrt(0.75))
    assert (tied.sum(axis=1) == 8).all()
    expected = np.argmax(tied, axis=1)  # lowest index among the tied samples
    np.testing.assert_array_equal(res.sample_index, expected)
    np.testing.assert_array_equal(res.estimate, values[expected])

    # independent of chunking and thread count
    for chunk, workers in ((1, 1), (7, 2), (10_000, -1)):
        again = nearest_neighbour(samples, values, targets, ell, chunk_size=chunk, workers=workers)
        np.testing.assert_array_equal(again.sample_index, res.sample_index)

    # a NaN at the winning index hands the tie to the next lowest index
    values2 = values.copy()
    values2[expected[0]] = np.nan
    res2 = nearest_neighbour(samples, values2, targets[:1], ell)
    assert res2.sample_index[0] == np.flatnonzero(tied[0])[1]


def test_radius_anisotropy_and_exclusion():
    # major axis north (bearing 0): a sample 10 m north is at distance 10,
    # a sample 4 m east is at 4 * ratio 3 = 12
    ell = Ellipsoid(11.0, 3.0, 3.0)
    samples = np.array([[0.0, 10.0, 0.0], [4.0, 0.0, 0.0]])
    values = np.array([1.0, 2.0])
    res = nearest_neighbour(samples, values, np.zeros((1, 3)), ell)
    assert res.sample_index[0] == 0 and res.estimate[0] == 1.0
    assert res.nearest_distance[0] == pytest.approx(10.0)
    # radius is inclusive; just inside it nothing is found
    assert nearest_neighbour(samples, values, np.zeros((1, 3)), ell, radius=10.0).estimated[0]
    out = nearest_neighbour(samples, values, np.zeros((1, 3)), ell, radius=9.999)
    assert not out.estimated[0] and np.isnan(out.estimate[0]) and out.sample_index[0] == -1
    assert out.n_samples[0] == 0 and np.isnan(out.nearest_distance[0])
    # a target on a sample takes it at distance 0, unless excluded
    hit = nearest_neighbour(samples, values, samples[1:], ell, radius=20.0)
    assert hit.sample_index[0] == 1 and hit.nearest_distance[0] == 0.0
    loo = nearest_neighbour(
        samples, values, samples[1:], ell, radius=20.0, exclude_zero_distance=True
    )
    assert loo.sample_index[0] == 0


def test_empty_and_invalid_inputs():
    ell = Ellipsoid(10.0)
    res = nearest_neighbour(np.zeros((0, 3)), np.zeros(0), np.zeros((2, 3)), ell)
    assert not res.estimated.any() and np.isnan(res.estimate).all()
    res = nearest_neighbour(np.zeros((2, 3)), np.array([np.nan, np.nan]), np.zeros((2, 3)), ell)
    assert not res.estimated.any() and res.n_samples_excluded == 2
    res = nearest_neighbour(np.zeros((2, 3)), np.ones(2), np.zeros((0, 3)), ell)
    assert res.estimate.shape == (0,)
    with pytest.raises(ValueError):
        nearest_neighbour(np.zeros((2, 3)), np.ones(3), np.zeros((1, 3)), ell)
    with pytest.raises(ValueError):
        nearest_neighbour(np.zeros((2, 2)), np.ones(2), np.zeros((1, 3)), ell)


# ------------------------------------------------------- synthetic deposit
def inside_lens(points, truth):
    """Implicit ore-lens function transcribed from truth.json ``solid``."""
    solid, lens = truth["solid"], truth["config"]["lens"]
    local = (points - np.array(solid["centre"])) @ np.array(solid["axes"])
    a, b, c = solid["half_axes"]
    u, v, w = local[:, 0], local[:, 1], local[:, 2]
    pinch = np.exp(-(((u - lens["pinch_position"]) / lens["pinch_width"]) ** 2))
    t = 1.0 - lens["pinch_depth"] * pinch
    mid = lens["bend_major"] * (u / a) ** 2 + lens["bend_semi"] * (v / b) ** 2
    return (u / a) ** 2 + (v / b) ** 2 + ((w - mid) / (c * t)) ** 2 < 1.0


def test_global_mean_equals_polygonal_declustered_mean():
    """NN global mean over the ore lens = polygonal-declustered sample mean.

    NN side: `nearest_neighbour` on the truth block centroids (10 x 10 x 5 m)
    of blocks that are more than half ore, isotropic distance.
    Reference side, computed here without bakumine code: polygonal
    declustering weights = volume of each sample's Voronoi cell clipped to
    the lens, measured by counting the nodes of a 2 x 2 x 1 m lattice
    (independent of the block grid, lens membership from the implicit
    function in truth.json) that are nearest to the sample (scipy cKDTree).
    The two means differ only by the coarseness of the block grid and the
    half-ore block selection versus the exact lens boundary.
    Tolerance: 2 % relative. Measured when written: NN 1.9273, polygonal
    1.9383 (0.57 % apart), naive sample mean 1.9744 (1.9 % above polygonal).
    """
    truth = json.loads((SYNTH / "truth.json").read_text())
    smp = pd.read_csv(SYNTH / "truth_samples.csv")
    smp = smp[smp["rock"] == "ORE"]
    xyz = smp[["x", "y", "z"]].to_numpy()
    au = smp["au_true"].to_numpy()
    blocks = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    centroids = blocks.loc[blocks["ore_fraction"] > 0.5, ["x", "y", "z"]].to_numpy()

    res = nearest_neighbour(xyz, au, centroids, Ellipsoid(1.0e4))
    assert res.estimated.all()
    nn_mean = res.estimate.mean()

    lo, hi = centroids.min(axis=0) - 10.0, centroids.max(axis=0) + 10.0
    axes = [np.arange(lo[k], hi[k], step) for k, step in enumerate((2.0, 2.0, 1.0))]
    weights = np.zeros(len(xyz))
    tree = cKDTree(xyz)
    for z in axes[2]:  # slab by slab to bound memory
        gx, gy = np.meshgrid(axes[0], axes[1], indexing="ij")
        nodes = np.column_stack([gx.ravel(), gy.ravel(), np.full(gx.size, z)])
        nodes = nodes[inside_lens(nodes, truth)]
        if len(nodes):
            weights += np.bincount(tree.query(nodes)[1], minlength=len(xyz))
    declustered = float((weights * au).sum() / weights.sum())
    naive = float(au.mean())

    rel = abs(nn_mean - declustered) / declustered
    assert rel < 0.02, (nn_mean, declustered, naive, rel)
    # the lattice volume is the lens volume (sanity of the reference itself)
    assert weights.sum() * 4.0 == pytest.approx(truth["solid"]["volume_m3"], rel=0.01)
    print(f"NN mean {nn_mean:.4f}  polygonal {declustered:.4f}  naive {naive:.4f}  rel {rel:.2e}")
