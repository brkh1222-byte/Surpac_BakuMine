"""Validation tools (P4.T9).

Independent checks: pandas ``groupby`` for the swath tables, a known field
for declustering, brute-force re-estimation loops for cross-validation, a
``gstools`` random field (oracle only) for the standardised errors, and the
truth of the synthetic deposit.
"""

import gstools as gs
import numpy as np
import pandas as pd
import pytest
import synthetic_ore as so

from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige, ordinary_kriging
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.validation import (
    cell_declustered_mean,
    cell_declustering_weights,
    cell_size_scan,
    compare_to_truth,
    cross_validate,
    error_summary,
    global_comparison,
    swath_table,
)
from bakumine.core.estimation.variogram import Structure, VariogramModel


# --------------------------------------------------------------------------
# swath tables
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def model_and_samples():
    rng = np.random.default_rng(21)
    gx, gy, gz = np.meshgrid(
        1000.0 + 10.0 * np.arange(14) + 5.0,
        2000.0 + 10.0 * np.arange(9) + 5.0,
        300.0 + 5.0 * np.arange(6) + 2.5,
        indexing="ij",
    )
    blocks = np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])
    grade = rng.lognormal(size=len(blocks))
    grade[rng.random(len(blocks)) < 0.15] = np.nan  # unestimated blocks
    volume = rng.choice([500.0, 62.5], size=len(blocks))  # parents and sub-blocks
    samples = rng.uniform([1000.0, 2000.0, 300.0], [1140.0, 2090.0, 330.0], size=(400, 3))
    assay = rng.lognormal(size=400)
    assay[rng.random(400) < 0.1] = np.nan
    weight = rng.uniform(0.2, 3.0, size=400)
    return blocks, grade, volume, samples, assay, weight


def groupby_reference(position, value, weight, width, origin):
    """Weighted mean per slice with a plain pandas groupby."""
    frame = pd.DataFrame({"p": position, "v": value, "w": weight}).dropna(subset=["v"])
    frame["slice"] = np.floor((frame["p"] - origin) / width).astype(int)
    frame["wv"] = frame["w"] * frame["v"]
    g = frame.groupby("slice")
    return pd.DataFrame(
        {"count": g.size(), "weight": g["w"].sum(), "mean": g["wv"].sum() / g["w"].sum()}
    )


@pytest.mark.parametrize(
    "axis, column, width", [("x", 0, 20.0), ("y", 1, 10.0), ("z", 2, 5.0), ("Z", 2, 7.5)]
)
@pytest.mark.parametrize("weighted", [False, True])
def test_swath_equals_pandas_groupby(model_and_samples, axis, column, width, weighted):
    blocks, grade, volume, samples, assay, weight = model_and_samples
    bw = volume if weighted else np.ones(len(blocks))
    sw = weight if weighted else np.ones(len(samples))
    table = swath_table(
        blocks, grade, samples, assay, axis=axis, width=width,
        block_weights=bw if weighted else None, sample_weights=sw if weighted else None,
    )  # fmt: skip
    low = min(blocks[np.isfinite(grade), column].min(), samples[np.isfinite(assay), column].min())
    origin = np.floor(low / width) * width
    ref_b = groupby_reference(blocks[:, column], grade, bw, width, origin)
    ref_s = groupby_reference(samples[:, column], assay, sw, width, origin)
    ref = ref_b.add_prefix("block_").join(ref_s.add_prefix("sample_"), how="outer").sort_index()
    assert list(table["slice"]) == list(ref.index)
    np.testing.assert_array_equal(table["block_count"], ref["block_count"].fillna(0))
    np.testing.assert_array_equal(table["sample_count"], ref["sample_count"].fillna(0))
    np.testing.assert_allclose(table["block_mean"], ref["block_mean"], rtol=1e-12)
    np.testing.assert_allclose(table["sample_mean"], ref["sample_mean"], rtol=1e-12)
    np.testing.assert_allclose(table["block_weight"], ref["block_weight"].fillna(0), rtol=1e-12)
    np.testing.assert_allclose(table["lower"], origin + table["slice"] * width)
    np.testing.assert_allclose(table["upper"] - table["lower"], width)
    np.testing.assert_allclose(table["centre"], table["lower"] + width / 2)
    assert table["block_count"].sum() == np.isfinite(grade).sum()
    assert table["sample_count"].sum() == np.isfinite(assay).sum()
    assert list(table.columns) == [
        "slice", "lower", "upper", "centre", "block_count", "block_weight", "block_mean",
        "sample_count", "sample_weight", "sample_mean",
    ]  # fmt: skip


def test_swath_along_rotated_axis_and_options(model_and_samples):
    blocks, grade, volume, samples, assay, weight = model_and_samples
    ell = Ellipsoid(100.0, 2.0, 4.0, (35.0, -10.0, 50.0))
    axis = ell.major_axis
    table = swath_table(
        blocks, grade, samples, assay, axis=3.0 * axis, width=15.0, origin=2200.0,
        block_weights=volume,
    )  # fmt: skip
    ref_b = groupby_reference(blocks @ axis, grade, volume, 15.0, 2200.0)
    ref_s = groupby_reference(samples @ axis, assay, np.ones(len(assay)), 15.0, 2200.0)
    ref = ref_b.add_prefix("block_").join(ref_s.add_prefix("sample_"), how="outer").sort_index()
    assert list(table["slice"]) == list(ref.index) and len(table) > 5
    np.testing.assert_array_equal(table["block_count"], ref["block_count"].fillna(0))
    np.testing.assert_allclose(table["block_mean"], ref["block_mean"], rtol=1e-10)
    np.testing.assert_allclose(table["sample_mean"], ref["sample_mean"], rtol=1e-10)
    # a slice with samples but no blocks is reported with a NaN block mean
    far = swath_table(blocks[:5], grade[:5] * 0 + 1.0, samples, assay, axis="x", width=20.0)
    assert (far["block_count"] == 0).any() and far["block_mean"].isna().any()
    assert far.loc[far["block_count"] > 0, "block_mean"].eq(1.0).all()
    # blocks only
    only = swath_table(blocks, grade, axis="z", width=5.0)
    assert (only["sample_count"] == 0).all() and only["sample_mean"].isna().all()
    assert len(only) == 6
    with pytest.raises(ValueError):
        swath_table(blocks, grade, axis="q", width=5.0)
    with pytest.raises(ValueError):
        swath_table(blocks, grade, axis="x", width=0.0)
    with pytest.raises(ValueError):
        swath_table(blocks, grade, samples, axis="x", width=5.0)


# --------------------------------------------------------------------------
# global comparison
# --------------------------------------------------------------------------
def test_global_comparison_by_hand():
    est = np.array([1.0, 2.0, 3.0, np.nan, 10.0])
    nn = np.array([2.0, 2.0, 5.0, 7.0, np.nan])  # only the first three blocks have both
    vol = np.array([1.0, 1.0, 2.0, 1.0, 1.0])
    samples = np.array([1.0, np.nan, 3.0, 8.0])
    g = global_comparison(est, nn, samples, block_weights=vol, sample_weights=[1.0, 5.0, 1.0, 0.5])
    assert g.n_blocks == 3 and g.n_samples == 3
    assert g.block_mean == pytest.approx((1 + 2 + 6) / 4)
    assert g.nn_mean == pytest.approx((2 + 2 + 10) / 4)
    assert g.naive_mean == pytest.approx(4.0)
    assert g.declustered_mean == pytest.approx((1 + 3 + 4) / 2.5)
    assert g.block_vs_nn == pytest.approx(2.25 / 3.5 - 1)
    assert g.block_vs_naive == pytest.approx(2.25 / 4 - 1)
    assert g.nn_vs_naive == pytest.approx(3.5 / 4 - 1)
    assert g.block_vs_declustered == pytest.approx(2.25 / 3.2 - 1)
    assert g.nn_vs_declustered == pytest.approx(3.5 / 3.2 - 1)
    plain = global_comparison(est, nn, samples)
    assert plain.block_mean == pytest.approx(2.0) and plain.nn_mean == pytest.approx(3.0)
    assert np.isnan(plain.declustered_mean) and np.isnan(plain.block_vs_declustered)


# --------------------------------------------------------------------------
# cell declustering
# --------------------------------------------------------------------------
def test_cell_weights_by_hand():
    # cells of 10 from the minimum corner: two samples share a cell, one is alone
    pts = np.array([[0.0, 0, 0], [3.0, 4.0, 0], [25.0, 0, 0]])
    w = cell_declustering_weights(pts, 10.0, n_offsets=1)
    np.testing.assert_allclose(w, [0.75, 0.75, 1.5])
    assert cell_declustered_mean(pts, [1.0, 3.0, 10.0], 10.0, n_offsets=1) == pytest.approx(6.0)
    # every sample in its own cell, or all in one cell: equal weights
    np.testing.assert_allclose(cell_declustering_weights(pts, 1.0), 1.0)
    np.testing.assert_allclose(cell_declustering_weights(pts, 1000.0, n_offsets=1), 1.0)
    # anisotropic cells and several origins; weights always average 1
    rng = np.random.default_rng(0)
    cloud = rng.normal(size=(300, 3)) * [50.0, 20.0, 5.0]
    for size in (7.0, (30.0, 10.0, 2.0)):
        w = cell_declustering_weights(cloud, size, n_offsets=5)
        assert w.mean() == pytest.approx(1.0) and (w > 0).all()
    # NaN values take no part
    assert cell_declustered_mean(pts, [1.0, np.nan, 10.0], 10.0, n_offsets=1) == pytest.approx(5.5)
    with pytest.raises(ValueError):
        cell_declustering_weights(pts, -1.0)


def test_cell_declustered_mean_is_closer_to_truth_for_clustered_samples():
    def field(x, y):  # background 1 with a high-grade zone around (300, 300)
        return 1.0 + 6.0 * np.exp(-((x - 300.0) ** 2 + (y - 300.0) ** 2) / (2 * 120.0**2))

    g = np.linspace(0.5, 999.5, 1000)
    true_mean = field(*np.meshgrid(g, g)).mean()
    rng = np.random.default_rng(4)
    # regular 100 m coverage plus infill drilling on the high-grade zone
    gx, gy = np.meshgrid(np.arange(50.0, 1000.0, 100.0), np.arange(50.0, 1000.0, 100.0))
    regular = np.column_stack([gx.ravel(), gy.ravel()]) + rng.uniform(-10, 10, size=(100, 2))
    infill = np.clip(rng.normal(300.0, 60.0, size=(400, 2)), 0.0, 1000.0)
    xy = np.vstack([regular, infill])
    coords = np.column_stack([xy, np.zeros(len(xy))])
    values = field(xy[:, 0], xy[:, 1])
    naive = values.mean()
    declustered = cell_declustered_mean(coords, values, (100.0, 100.0, 1.0))
    scan = cell_size_scan(
        coords, values, np.arange(20.0, 301.0, 20.0), anisotropy=(1.0, 1.0, 0.01)
    )
    print(
        f"\ntrue mean {true_mean:.3f}  naive {naive:.3f}  cell 100 m {declustered:.3f}"
        f"  scan best {scan.best_mean:.3f} at {scan.best_size:.0f} m"
    )
    assert naive > 1.5 * true_mean  # the clustering really biases the naive mean
    assert abs(declustered - true_mean) < 0.2 * abs(naive - true_mean)
    assert declustered == pytest.approx(true_mean, rel=0.1)
    # the scan: every size improves on the naive mean; the chosen one is the lowest
    assert scan.naive_mean == pytest.approx(naive)
    assert (np.abs(scan.means - true_mean) < abs(naive - true_mean)).all()
    assert scan.best_mean == scan.means.min() and 60.0 <= scan.best_size <= 300.0
    assert np.average(values, weights=scan.weights) == pytest.approx(scan.best_mean)
    assert cell_size_scan(coords, values, [50.0, 100.0], choose="max").best_mean == max(
        cell_size_scan(coords, values, [50.0, 100.0]).means
    )


# --------------------------------------------------------------------------
# error statistics and model against truth
# --------------------------------------------------------------------------
def test_error_summary_and_compare_to_truth():
    rng = np.random.default_rng(6)
    truth = rng.lognormal(size=500)
    est = 0.3 + 0.8 * truth + rng.normal(scale=0.4, size=500)
    var = rng.uniform(0.1, 0.5, size=500)
    est[:20] = np.nan
    truth[20:30] = np.nan
    ok = np.isfinite(est) & np.isfinite(truth)
    e, t = est[ok], truth[ok]
    s = error_summary(truth, est, var)
    assert s["n"] == 470 == s["n_std_error"]
    assert s["mean_error"] == pytest.approx((e - t).mean())
    assert s["relative_bias"] == pytest.approx((e - t).mean() / t.mean())
    assert s["rmse"] == pytest.approx(np.sqrt(((e - t) ** 2).mean()))
    assert s["mae"] == pytest.approx(np.abs(e - t).mean())
    assert s["correlation"] == pytest.approx(np.corrcoef(e, t)[0, 1])
    slope, intercept = np.polyfit(e, t, 1)  # regression of truth on estimate
    assert s["slope"] == pytest.approx(slope) and s["intercept"] == pytest.approx(intercept)
    std = (e - t) / np.sqrt(var[ok])
    assert s["mean_std_error"] == pytest.approx(std.mean())
    assert s["var_std_error"] == pytest.approx(std.var())
    assert "mean_std_error" not in error_summary(truth, est)

    coords = rng.uniform(0, 100, size=(500, 3))
    vol = rng.uniform(1, 5, size=500)
    c = compare_to_truth(est, truth, weights=vol, coords=coords, axis="y", width=25.0)
    w = vol[ok] / vol[ok].sum()
    assert c.n == 470
    assert c.bias == pytest.approx((w * (e - t)).sum())
    assert c.mean_estimate - c.mean_truth == pytest.approx(c.bias)
    assert c.relative_bias == pytest.approx(c.bias / (w * t).sum())
    assert c.rmse == pytest.approx(np.sqrt((w * (e - t) ** 2).sum()))
    assert c.mae == pytest.approx((w * np.abs(e - t)).sum())
    frame = pd.DataFrame({"e": e, "t": t, "w": vol[ok], "k": np.floor(coords[ok, 1] / 25.0)})
    ref = frame.groupby("k").apply(
        lambda g: pd.Series(
            {
                "count": len(g),
                "bias": np.average(g["e"] - g["t"], weights=g["w"]),
                "rmse": np.sqrt(np.average((g["e"] - g["t"]) ** 2, weights=g["w"])),
                "truth_mean": np.average(g["t"], weights=g["w"]),
            }
        ),
        include_groups=False,
    )
    assert len(c.by_slice) == 4 and list(c.by_slice["lower"]) == [0.0, 25.0, 50.0, 75.0]
    for name in ("count", "bias", "rmse", "truth_mean"):
        np.testing.assert_allclose(c.by_slice[name], ref[name], rtol=1e-10)
    plain = compare_to_truth(est, truth)
    assert plain.by_slice is None and plain.rmse == pytest.approx(s["rmse"])
    assert plain.correlation == pytest.approx(s["correlation"])
    assert plain.slope == pytest.approx(s["slope"])
    with pytest.raises(ValueError):
        compare_to_truth(est, truth, coords=coords)


# --------------------------------------------------------------------------
# cross-validation
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def drilled():
    """12 short 'holes' of 8 samples each in a 100 m cube, one missing assay."""
    rng = np.random.default_rng(13)
    collars = rng.uniform(0.0, 100.0, size=(12, 2))
    xyz = np.array(
        [[cx, cy, 90.0 - 4.0 * k] for cx, cy in collars for k in range(8)]
    ) + rng.normal(scale=0.2, size=(96, 3))
    holes = np.repeat([f"H{i:02d}" for i in range(12)], 8)
    values = np.exp(0.02 * xyz[:, 0] + rng.normal(scale=0.5, size=96))
    values[17] = np.nan
    ell = Ellipsoid(80.0, 1.5, 2.0, (30.0, 0.0, 20.0))
    model = VariogramModel(0.1, (Structure("spherical", 0.9, (60.0, 40.0, 30.0)),))
    return xyz, values, holes, ell, model


def test_leave_one_out_idw_equals_brute_force_loop(drilled):
    xyz, values, holes, ell, _ = drilled
    params = SearchParams(max_samples=6, min_samples=2, max_per_hole=3)
    cv = cross_validate(xyz, values, ell, params, method="idw", power=2.5, hole_ids=holes)
    assert cv.method == "idw" and cv.leave == "one"
    n = len(values)
    for i in range(n):
        if np.isnan(values[i]):
            assert not cv.estimated[i] and np.isnan(cv.estimate[i])
            assert (cv.neighbours[i] == -1).all()
            continue
        others = np.arange(n) != i
        ref = inverse_distance(
            xyz[others], values[others], xyz[i : i + 1], ell, params,
            power=2.5, hole_ids=holes[others],
        )  # fmt: skip
        assert ref.estimated[0] and cv.estimated[i]
        assert cv.estimate[i] == pytest.approx(ref.estimate[0], rel=1e-12)
        assert cv.n_samples[i] == ref.n_samples[0]
        used = cv.neighbours[i][cv.neighbours[i] >= 0]
        assert i not in used and 17 not in used and len(used) == cv.n_samples[i]
    ok = cv.estimated
    np.testing.assert_allclose(cv.error[ok], cv.estimate[ok] - values[ok])
    assert np.isnan(cv.variance).all() and np.isnan(cv.std_error).all()
    assert cv.summary == error_summary(values, cv.estimate)
    assert cv.summary["n"] == 95 and "mean_std_error" not in cv.summary
    assert len(cv.to_frame()) == n


def test_leave_one_out_ok_equals_brute_force_loop(drilled):
    xyz, values, holes, ell, model = drilled
    params = SearchParams(max_samples=8)
    cv = cross_validate(xyz, values, ell, params, method="ok", model=model, hole_ids=holes)
    n = len(values)
    for i in (0, 5, 16, 18, 40, 95):
        others = np.arange(n) != i
        ref = ordinary_kriging(xyz[others], values[others], xyz[i : i + 1], model, ell, params)
        assert cv.estimate[i] == pytest.approx(ref.estimate[0], rel=1e-10)
        assert cv.variance[i] == pytest.approx(ref.variance[0], rel=1e-10)
    ok = cv.estimated
    assert ok.sum() == 95
    np.testing.assert_allclose(cv.std_error[ok], cv.error[ok] / np.sqrt(cv.variance[ok]))
    assert cv.summary["var_std_error"] == pytest.approx(cv.std_error[ok].var())
    # the neighbour lists are those the estimator itself used
    direct = krige(
        xyz, values, xyz, model, ell, SearchParams(max_samples=8, exclude_zero_distance=True),
        return_weights_for=np.arange(n),
    )  # fmt: skip
    np.testing.assert_array_equal(cv.neighbours[ok], direct.weight_samples[ok])
    with pytest.raises(ValueError):
        cross_validate(xyz, values, ell, method="ok")
    with pytest.raises(ValueError):
        cross_validate(xyz, values, ell, method="nn")
    with pytest.raises(TypeError):
        cross_validate(xyz, values, ell, method="idw", block_size=(1, 1, 1))


@pytest.mark.parametrize("method", ["idw", "ok"])
def test_leave_one_hole_out_never_uses_the_left_out_hole(drilled, method):
    xyz, values, holes, ell, model = drilled
    params = SearchParams(max_samples=8, min_samples=1)
    kw = dict(method=method, model=model if method == "ok" else None, hole_ids=holes)
    cv = cross_validate(xyz, values, ell, params, leave="hole", **kw)
    loo = cross_validate(xyz, values, ell, params, leave="one", **kw)
    assert cv.leave == "hole" and cv.estimated.sum() == 95
    # through the returned neighbour lists
    used = cv.neighbours >= 0
    assert used.sum() == cv.n_samples.sum() > 0
    own = np.broadcast_to(holes[:, None], cv.neighbours.shape)
    assert (holes[cv.neighbours[used]] != own[used]).all()
    assert (cv.neighbours[used] != 17).all()  # the missing assay is never used
    # ... whereas leave-one-out leans on the same hole
    lused = loo.neighbours >= 0
    same = holes[loo.neighbours[lused]] == np.broadcast_to(holes[:, None], lused.shape)[lused]
    assert same.mean() > 0.3
    # black box: changing the values of a hole does not change its own estimates
    changed = values.copy()
    inside = holes == "H03"
    changed[inside] = 1e6
    again = cross_validate(xyz, changed, ell, params, leave="hole", **kw)
    np.testing.assert_array_equal(again.estimate[inside], cv.estimate[inside])
    rest = ~inside & cv.estimated
    assert not np.allclose(again.estimate[rest], cv.estimate[rest])
    # brute force: estimate each hole from all the others
    for hole in ("H00", "H02", "H07"):
        inside = holes == hole
        if method == "idw":
            ref = inverse_distance(
                xyz[~inside], values[~inside], xyz[inside], ell, params, hole_ids=holes[~inside]
            )
        else:
            ref = ordinary_kriging(xyz[~inside], values[~inside], xyz[inside], model, ell, params)
            direct = krige(
                xyz[~inside], values[~inside], xyz[inside], model, ell, params,
                return_weights_for=np.arange(inside.sum()),
            )  # fmt: skip
            rows = np.flatnonzero(~inside)
            mapped = np.where(
                direct.weight_samples >= 0, rows[np.maximum(direct.weight_samples, 0)], -1
            )
            shown = np.isfinite(values[inside])
            np.testing.assert_array_equal(cv.neighbours[inside][shown], mapped[shown])
        shown = np.isfinite(values[inside])
        np.testing.assert_allclose(cv.estimate[inside][shown], ref.estimate[shown], rtol=1e-12)
    with pytest.raises(ValueError):
        cross_validate(xyz, values, ell, params, leave="hole", method="idw")


def simulated(kind):
    """2000 scattered samples of a field with a known variogram model.

    ``exponential``: a ``gstools`` random field (5000 Fourier modes) plus an
    independent nugget. ``spherical``: exact simulation, ``L @ normal`` with
    ``L`` the Cholesky factor of the model's covariance matrix (gstools'
    Fourier-mode sampler did not reproduce the short-range behaviour of a
    spherical model here: measured variance of standardised errors 1.67,
    against 1.01 for the exact simulation).
    """
    rng = np.random.default_rng(31)
    xyz = rng.uniform(0.0, [400.0, 400.0, 100.0], size=(2000, 3))
    ell = Ellipsoid.from_radii((80.0, 50.0, 25.0), (60.0, 0.0, 0.0))
    model = VariogramModel(0.2, (Structure.from_ellipsoid(kind, 0.8, ell),))
    if kind == "exponential":
        structured = gs.Exponential(dim=3, var=0.8, len_scale=[50.0, 80.0, 25.0], rescale=3.0)
        field = gs.SRF(structured, seed=5, mode_no=5000)((xyz @ ell.rotation).T)
        values = 3.0 + field + rng.normal(scale=np.sqrt(0.2), size=len(xyz))
    else:
        low = np.linalg.cholesky(model.covariance_matrix(xyz))
        values = 3.0 + low @ rng.normal(size=len(xyz))
    return xyz, values, ell, model


@pytest.mark.parametrize("kind", ["exponential", "spherical"])
def test_ok_standardised_errors_have_unit_variance_for_the_true_variogram(kind):
    # data simulated from the model that is then used for kriging
    xyz, values, ell, model = simulated(kind)
    params = SearchParams(max_samples=16)
    cv = cross_validate(xyz, values, ell, params, method="ok", model=model)
    s = cv.summary
    print(
        f"\nLOO OK, true {kind} model: ME {s['mean_error']:.4f}  RMSE {s['rmse']:.4f}"
        f"  mean std err {s['mean_std_error']:.4f}  var std err {s['var_std_error']:.4f}"
        f"  slope {s['slope']:.3f}  r {s['correlation']:.3f}  n {s['n']}"
    )
    assert s["n"] == 2000
    assert abs(s["mean_std_error"]) < 0.1
    assert 0.8 < s["var_std_error"] < 1.25
    assert s["slope"] == pytest.approx(1.0, abs=0.15)
    # a model with the wrong sill is detected: 4 x the sill -> 1/4 of the variance of
    # the standardised errors, while the weights and so the estimates stay the same
    wrong = VariogramModel(0.8, (Structure.from_ellipsoid(kind, 3.2, ell),))
    w = cross_validate(xyz, values, ell, params, method="ok", model=wrong).summary
    assert w["var_std_error"] == pytest.approx(s["var_std_error"] / 4.0, rel=1e-6)
    assert w["rmse"] == pytest.approx(s["rmse"], rel=1e-9)
    # IDW on the same data is no better than kriging with the true model
    idw = cross_validate(xyz, values, ell, params, method="idw").summary
    print(f"LOO IDW2 on the same data: RMSE {idw['rmse']:.4f}")
    assert idw["rmse"] >= s["rmse"]


# --------------------------------------------------------------------------
# the plan's criterion on the synthetic deposit
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def ore_blocks():
    blocks = pd.read_parquet(so.SYNTH / "truth_blocks.parquet")
    return blocks[blocks["ore_fraction"] > 0.5].reset_index(drop=True)


def estimate_blocks(data, blocks, model):
    """NN, IDW2 and OK of the ore blocks; set-up of test_estimators_synthetic."""
    xyz = data[["x", "y", "z"]].to_numpy()
    au = data["au"].to_numpy()
    centroids = blocks[["x", "y", "z"]].to_numpy()
    search = so.true_range_ellipsoid(180.0)
    params = SearchParams(max_samples=16, min_samples=1)
    block = dict(block_size=so.truth()["blocks"]["size"], discretisation=(4, 4, 2))
    nn = nearest_neighbour(xyz, au, centroids, search)
    idw = inverse_distance(xyz, au, centroids, search, params, power=2.0, **block)
    ok = ordinary_kriging(xyz, au, centroids, model, search, params, **block)
    assert ok.n_singular == 0 and ok.n_invalid == 0
    return nn.estimate, idw.estimate, ok.estimate


@pytest.mark.parametrize("support", ["samples", "composites"])
def test_synthetic_rmse_ordering_ok_idw_nn(ore_blocks, support):
    data = so.SUPPORTS[support][0]()
    true = ore_blocks["au_ore"].to_numpy()
    fitted = so.ore_variography(support).fit.model
    lines = [f"\n{support}: {len(data)} values, {len(ore_blocks)} ore blocks"]
    for name, model in (("true variogram", so.true_model()), ("fitted variogram", fitted)):
        nn, idw, ok = estimate_blocks(data, ore_blocks, model)
        scores = {
            label: compare_to_truth(est, true)
            for label, est in (("NN", nn), ("IDW2", idw), ("OK", ok))
        }
        for label, c in scores.items():
            lines.append(
                f"  {name:16s} {label:4s} rmse {c.rmse:.4f}  r {c.correlation:.4f}"
                f"  bias {c.relative_bias:+.4f}  slope {c.slope:.3f}  n {c.n}"
            )
        assert scores["OK"].n == scores["NN"].n > 0.99 * len(true)
        assert scores["OK"].rmse <= scores["IDW2"].rmse <= scores["NN"].rmse
        assert scores["OK"].correlation >= scores["IDW2"].correlation >= scores["NN"].correlation
        for c in scores.values():
            assert abs(c.relative_bias) < 0.05
    print("\n".join(lines))


def test_synthetic_validation_run(ore_blocks):
    """The checks of research.md section 9 on the synthetic model."""
    data = so.ore_composites()
    xyz = data[["x", "y", "z"]].to_numpy()
    au = data["au"].to_numpy()
    holes = data["hole_id"].to_numpy()
    model = so.ore_variography("composites").fit.model
    nn, idw, ok = estimate_blocks(data, ore_blocks, model)
    centroids = ore_blocks[["x", "y", "z"]].to_numpy()
    true = ore_blocks["au_ore"].to_numpy()

    # global means: block models against NN-declustered, cell-declustered and naive
    scan = cell_size_scan(xyz, au, np.arange(20.0, 121.0, 10.0), anisotropy=(1.0, 1.0, 0.25))
    g = global_comparison(ok, nn, au, sample_weights=scan.weights)
    print(
        f"\nglobal: truth {np.nanmean(true):.4f}  OK {g.block_mean:.4f}  NN {g.nn_mean:.4f}"
        f"  naive {g.naive_mean:.4f}  cell-declustered {g.declustered_mean:.4f}"
        f" (cell {scan.best_size:.0f} m)  OK vs NN {g.block_vs_nn:+.4f}"
        f"  OK vs naive {g.block_vs_naive:+.4f}"
    )
    assert abs(g.block_vs_nn) < 0.05
    assert abs(global_comparison(idw, nn, au).block_vs_nn) < 0.05
    assert g.nn_mean == pytest.approx(np.nanmean(true), rel=0.05)

    # swaths: the block curve follows the composite curve and is smoother
    for axis, width in (("x", 40.0), ("y", 40.0), ("z", 20.0)):
        t = swath_table(centroids, ok, xyz, au, axis=axis, width=width)
        well = t[(t["sample_count"] >= 30) & (t["block_count"] >= 30)]
        r = np.corrcoef(well["block_mean"], well["sample_mean"])[0, 1]
        print(
            f"swath {axis}: {len(t)} slices, {len(well)} with >= 30 samples and blocks,"
            f" r(block, composite) {r:.3f}, largest relative gap"
            f" {(well['block_mean'] / well['sample_mean'] - 1).abs().max():.3f}"
        )
        assert len(well) >= 5 and r > 0.6
        truth_swath = compare_to_truth(ok, true, coords=centroids, axis=axis, width=width).by_slice
        assert len(truth_swath) >= len(well)

    # cross-validation at the composites
    search = so.true_range_ellipsoid(180.0)
    params = SearchParams(max_samples=16, min_samples=1)
    results = {}
    for leave in ("one", "hole"):
        for method in ("idw", "ok"):
            cv = cross_validate(
                xyz, au, search, params, method=method, model=model, hole_ids=holes, leave=leave
            )
            s = results[leave, method] = cv.summary
            extra = ""
            if method == "ok":
                extra = (
                    f"  mean std err {s['mean_std_error']:+.3f}  var std err"
                    f" {s['var_std_error']:.3f} (variogram of normal scores: not interpretable)"
                )
            print(
                f"CV leave-{leave}-out {method.upper():3s}: n {s['n']}"
                f"  ME {s['mean_error']:+.4f}  RMSE {s['rmse']:.4f}  MAE {s['mae']:.4f}"
                f"  r {s['correlation']:.3f}  slope {s['slope']:.3f}{extra}"
            )
            if leave == "hole":
                used = cv.neighbours >= 0
                own = np.broadcast_to(holes[:, None], used.shape)
                assert (holes[cv.neighbours[used]] != own[used]).all()
    for method in ("idw", "ok"):
        assert results["hole", method]["rmse"] > results["one", method]["rmse"]
        assert results["hole", method]["n"] > 0.95 * len(au)
