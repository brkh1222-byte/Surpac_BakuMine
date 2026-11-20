"""Dynamic anisotropy: per-block ellipsoid and variogram orientation (P4.T10).

Checks
------
* constant per-block angles reproduce the fixed-ellipsoid run bit for bit
  (NN, IDW, OK, SK; block and parent support);
* each block's ellipsoid / variogram rotation equals ``rotation_matrix`` of
  that block's angles, and each block's estimate equals an independent
  fixed-ellipsoid call of the array estimator with that orientation;
* plan JSON round trip;
* ``angles_from_matrix`` inverts ``rotation_matrix``; orientation
  interpolation averages rotations, not angles;
* folded synthetic deposit: dynamic anisotropy following the fold beats a
  fixed flat-lying ellipsoid (same radii and variogram) against the known
  truth, in RMSE and correlation, for IDW and OK.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pandas as pd
import pytest

from bakumine.core.blockmodel import BlockModel, BlockModelDefinition
from bakumine.core.estimation.block_estimation import (
    EstimationPass,
    EstimationPlan,
    estimate_attribute,
    load_passes,
    save_passes,
)
from bakumine.core.estimation.dynamic_anisotropy import (
    DynamicAnisotropy,
    angles_from_matrix,
    interpolate_orientations,
    local_ellipsoid,
    local_variogram,
    run_dynamic,
)
from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid, rotation_matrix
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel

ANGLES = (35.0, 10.0, 50.0)
DYN = DynamicAnisotropy(bearing="ang_b", plunge="ang_p", dip="ang_d")
SUFFIXES = ("", "_pass", "_ns", "_dist", "_ndist", "_kv", "_slope", "_ke")


# ------------------------------------------------------------------ helpers


def small_model(max_depth: int = 0) -> BlockModel:
    d = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (8, 7, 6), max_depth=max_depth)
    return BlockModel(d)


def random_samples(seed: int = 3, n_holes: int = 25) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for h in range(n_holes):
        x, y = rng.uniform(0, 80), rng.uniform(0, 70)
        for z in np.arange(1.0, 30.0, 2.0):
            rows.append((f"H{h}", x, y, z, 1.0 + np.sin(x / 13) + 0.05 * z + rng.normal(0, 0.2)))
    return pd.DataFrame(rows, columns=["hole_id", "x", "y", "z", "au"])


def variogram(angles=ANGLES) -> VariogramModel:
    return VariogramModel(
        0.1,
        (
            Structure("spherical", 0.6, (60.0, 40.0, 15.0), angles),
            Structure("exponential", 0.3, (90.0, 50.0, 20.0), angles),
        ),
    )


def make_pass(method: str, **kw) -> EstimationPass:
    extra = {}
    if method in ("ok", "sk"):
        extra["variogram"] = variogram()
    if method == "sk":
        extra["mean"] = 1.5
    return EstimationPass(
        method,
        Ellipsoid(60.0, 1.5, 4.0, ANGLES),
        SearchParams(max_samples=12, min_samples=2, max_per_hole=4),
        discretisation=(2, 2, 2),
        **extra,
        **kw,
    )


def add_angles(model: BlockModel, b, p, d) -> None:
    for name, value in (("ang_b", b), ("ang_p", p), ("ang_d", d)):
        model.add_attribute(name, "float")
        model.set(name, np.broadcast_to(np.asarray(value, dtype=float), (model.n_blocks,)).copy())


def outputs(model: BlockModel, attribute: str = "au") -> dict[str, np.ndarray]:
    return {s: model[attribute + s].copy() for s in SUFFIXES if attribute + s in model}


# ------------------------------------------------- constant angles == fixed


@pytest.mark.parametrize("method", ["nn", "idw", "ok", "sk"])
@pytest.mark.parametrize("max_depth", [0, 1])
def test_constant_angles_equal_fixed(method, max_depth):
    samples = random_samples()
    fixed_model = small_model(max_depth)
    if max_depth:
        fixed_model.split(np.arange(0, fixed_model.n_blocks, 7))  # sub-blocks -> parent support
    add_angles(fixed_model, *ANGLES)
    dyn_model = BlockModel.from_arrays(
        fixed_model.definition, fixed_model.parent, fixed_model.level, fixed_model.sub
    )
    add_angles(dyn_model, *ANGLES)

    estimate_attribute(fixed_model, samples, attribute="au", passes=[make_pass(method)])
    estimate_attribute(dyn_model, samples, attribute="au", passes=[make_pass(method, dynamic=DYN)])
    a, b = outputs(fixed_model), outputs(dyn_model)
    assert set(a) == set(b)
    assert np.isfinite(a[""]).sum() > 0.5 * fixed_model.n_blocks
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)


def test_constant_ratios_equal_fixed():
    samples = random_samples()
    fixed_model, dyn_model = small_model(), small_model()
    for m in (fixed_model, dyn_model):
        add_angles(m, *ANGLES)
        m.add_attribute("r_semi", "float")
        m.add_attribute("r_minor", "float")
        m.set("r_semi", 1.5)
        m.set("r_minor", 4.0)
    dyn = dataclasses.replace(DYN, semi_major_ratio="r_semi", minor_ratio="r_minor")
    estimate_attribute(fixed_model, samples, attribute="au", passes=[make_pass("ok")])
    estimate_attribute(dyn_model, samples, attribute="au", passes=[make_pass("ok", dynamic=dyn)])
    for key, values in outputs(fixed_model).items():
        np.testing.assert_array_equal(values, outputs(dyn_model)[key], err_msg=key)


# ------------------------------------------- per-block rotation is correct


def test_local_ellipsoid_and_variogram_rotation():
    rng = np.random.default_rng(0)
    base = Ellipsoid(50.0, 2.0, 5.0, (0.0, 0.0, 0.0))
    for _ in range(20):
        ang = rng.uniform([-180, -89, -180], [180, 89, 180])
        for conv in ("surpac_zxy_lrl", "gslib"):
            e = local_ellipsoid(base, (*ang, 1.7, 3.0), conv)
            np.testing.assert_allclose(e.rotation, rotation_matrix(*ang, convention=conv))
            assert e.radii == pytest.approx((50.0, 50.0 / 1.7, 50.0 / 3.0))
            v = local_variogram(variogram((0.0, 0.0, 0.0)), ang, conv)
            for s, s0 in zip(v.structures, variogram().structures, strict=True):
                assert s.ranges == s0.ranges and s.sill == s0.sill
                np.testing.assert_allclose(
                    s.ellipsoid.rotation, rotation_matrix(*ang, convention=conv)
                )


def test_each_block_uses_its_own_orientation():
    """Per-block results equal independent fixed-ellipsoid calls at each block's angles."""
    samples = random_samples(5)
    model = small_model()
    c = model.centroids()
    rng = np.random.default_rng(1)
    # a few distinct orientations spread over the blocks
    choices = rng.uniform([-90, -40, -60], [90, 40, 60], size=(5, 3)).round(1)
    pick = rng.integers(0, 5, size=model.n_blocks)
    add_angles(model, choices[pick, 0], choices[pick, 1], choices[pick, 2])

    seen = []

    def recorder(fixed, idx):
        seen.append((fixed, idx))
        return {"ok": np.ones(len(idx), bool), "estimate": np.zeros(len(idx))}

    p = make_pass("ok", dynamic=DYN)
    params = DYN.block_parameters(model, np.arange(model.n_blocks), p.ellipsoid)
    run_dynamic(p, params, recorder)
    assert len(seen) == len(np.unique(pick))
    covered = np.zeros(model.n_blocks, int)
    for fixed, idx in seen:
        covered[idx] += 1
        expect = rotation_matrix(*choices[pick[idx[0]]])
        assert (pick[idx] == pick[idx[0]]).all()
        assert fixed.dynamic is None
        np.testing.assert_allclose(fixed.ellipsoid.rotation, expect)
        assert fixed.ellipsoid.major_radius == p.ellipsoid.major_radius
        for s in fixed.variogram.structures:
            np.testing.assert_allclose(s.ellipsoid.rotation, expect)
    assert (covered == 1).all()

    coords = samples[["x", "y", "z"]].to_numpy()
    values, holes = samples["au"].to_numpy(), samples["hole_id"].to_numpy()
    for method in ("nn", "idw", "ok"):
        m = small_model()
        add_angles(m, choices[pick, 0], choices[pick, 1], choices[pick, 2])
        q = make_pass(method, dynamic=DYN)
        estimate_attribute(m, samples, attribute="au", passes=[q])
        for k in range(5):
            rows = np.flatnonzero(pick == k)
            ell = Ellipsoid(60.0, 1.5, 4.0, tuple(choices[k]))
            if method == "nn":
                r = nearest_neighbour(coords, values, c[rows], ell)
            elif method == "idw":
                r = inverse_distance(
                    coords, values, c[rows], ell, q.search, hole_ids=holes,
                    block_size=(10.0, 10.0, 5.0), discretisation=(2, 2, 2),
                )  # fmt: skip
            else:
                r = krige(
                    coords, values, c[rows], variogram(tuple(choices[k])), ell, q.search,
                    hole_ids=holes, block_size=(10.0, 10.0, 5.0), discretisation=(2, 2, 2),
                )  # fmt: skip
            expect = np.where(r.estimated, r.estimate, np.nan)
            np.testing.assert_array_equal(m.get("au", rows), expect, err_msg=f"{method} {k}")


def test_missing_angles_fixed_or_skip():
    samples = random_samples()
    angles = np.full(small_model().n_blocks, ANGLES[0])
    angles[::3] = np.nan
    ref = small_model()
    add_angles(ref, *ANGLES)
    estimate_attribute(ref, samples, attribute="au", passes=[make_pass("idw")])
    for missing in ("fixed", "skip"):
        m = small_model()
        add_angles(m, angles, ANGLES[1], ANGLES[2])
        dyn = dataclasses.replace(DYN, missing=missing)
        estimate_attribute(m, samples, attribute="au", passes=[make_pass("idw", dynamic=dyn)])
        keep = np.isfinite(angles)
        np.testing.assert_array_equal(m["au"][keep], ref["au"][keep])
        if missing == "fixed":
            np.testing.assert_array_equal(m["au"][~keep], ref["au"][~keep])
        else:
            assert np.isnan(m["au"][~keep]).all()
            assert (m["au_pass"][~keep] == 0).all()


def test_missing_attribute_is_an_error():
    m = small_model()
    with pytest.raises(ValueError, match="ang_b"):
        estimate_attribute(
            m, random_samples(), attribute="au", passes=[make_pass("idw", dynamic=DYN)]
        )
    with pytest.raises(ValueError, match="at least one angle"):
        DynamicAnisotropy()


# --------------------------------------------------------------- JSON


def test_plan_json_round_trip(tmp_path):
    dyn = DynamicAnisotropy(
        bearing="ang_b", dip="ang_d", minor_ratio="r_minor", missing="skip", angle_step=2.5
    )
    passes = [make_pass("ok", dynamic=dyn), make_pass("idw")]
    plan = EstimationPlan("au", [{"name": "ALL", "passes": passes}])
    text = plan.to_json()
    back = EstimationPlan.from_json(text)
    assert back.to_dict() == plan.to_dict()
    assert back.domains[0].passes[0].dynamic == dyn
    assert back.domains[0].passes[1].dynamic is None
    assert '"dynamic"' in text and "r_minor" in text
    save_passes(passes, tmp_path / "p.json")
    assert load_passes(tmp_path / "p.json") == passes


# ------------------------------------------------ angles and interpolation


@pytest.mark.parametrize("conv", ["surpac_zxy_lrl", "gslib"])
def test_angles_from_matrix_inverts_rotation_matrix(conv):
    rng = np.random.default_rng(2)
    ang = rng.uniform([-179, -89, -179], [179, 89, 179], size=(200, 3))
    mats = np.stack([rotation_matrix(*a, convention=conv) for a in ang])
    np.testing.assert_allclose(angles_from_matrix(mats, conv), ang, atol=1e-9)
    # gimbal lock: the matrix is still reproduced
    for a in ((30.0, 90.0, 20.0), (-50.0, -90.0, 70.0)):
        r = rotation_matrix(*a, convention=conv)
        np.testing.assert_allclose(
            rotation_matrix(*angles_from_matrix(r, conv), convention=conv), r, atol=1e-9
        )


def _same_ellipsoid(r1, r2):
    """Equal up to the ellipsoid's symmetries: each axis equal up to sign."""
    return np.allclose(np.abs(np.einsum("ij,ij->j", r1, r2)), 1.0, atol=1e-9)


def test_interpolate_orientations():
    pts = np.array([[0.0, 0.0, 0.0], [100.0, 0.0, 0.0]])
    # constant field, given with two equivalent representations (bearing +180,
    # plunge and dip negated: every axis flips sign except one -> same ellipsoid)
    a0 = (20.0, 15.0, 30.0)
    r0 = rotation_matrix(*a0)
    alt = r0 * np.array([-1.0, -1.0, 1.0])
    a1 = angles_from_matrix(alt)
    tg = np.array([[30.0, 5.0, 0.0], [70.0, -5.0, 3.0], [0.0, 0.0, 0.0]])
    out = interpolate_orientations(pts, np.array([a0, a1]), tg)
    for a in out:
        assert _same_ellipsoid(rotation_matrix(*a), r0)
    # dip 10 and 30 either side: the midpoint gets dip 20 (raw-angle averaging
    # would too here; a wrap case below distinguishes)
    out = interpolate_orientations(pts, [(0, 0, 10), (0, 0, 30)], [[50.0, 0, 0]])
    np.testing.assert_allclose(out[0], (0, 0, 20), atol=1e-9)
    # bearings 170 and -170 (= 190): the mean is 180, not 0
    out = interpolate_orientations(pts, [(170, 0, 0), (-170, 0, 0)], [[50.0, 0, 0]])
    assert _same_ellipsoid(rotation_matrix(*out[0]), rotation_matrix(180.0, 0, 0))
    # exact hit takes the data orientation
    out = interpolate_orientations(pts, [(0, 0, 10), (0, 0, 30)], [[100.0, 0, 0]])
    np.testing.assert_allclose(out[0], (0, 0, 30), atol=1e-9)


# ------------------------------------------------ folded synthetic deposit

AMP, WAVE = 50.0, 400.0  # fold amplitude and wavelength (m), fold axis north-south
PERIOD = 24.0  # grade banding period across the layering (m)


def fold_z(x):
    return AMP * np.sin(2 * np.pi * x / WAVE)


def fold_slope(x):
    return AMP * 2 * np.pi / WAVE * np.cos(2 * np.pi * x / WAVE)


def fold_truth(xyz):
    """Grade banded parallel to the folded layering, slowly varying along it."""
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    normal = (z - fold_z(x)) / np.sqrt(1 + fold_slope(x) ** 2)
    return (
        3.0
        + 1.5 * np.cos(2 * np.pi * normal / PERIOD)
        + 0.4 * np.sin(2 * np.pi * x / 350.0)
        + 0.3 * np.cos(2 * np.pi * y / 300.0)
    )


def test_folded_deposit_dynamic_beats_fixed():
    d = BlockModelDefinition((0.0, 0.0, -60.0), (10.0, 10.0, 4.0), (40, 10, 30))
    rng = np.random.default_rng(11)
    rows = []
    for i, x in enumerate(np.arange(5.0, 400.0, 20.0)):
        for j, y in enumerate(np.arange(10.0, 100.0, 30.0)):
            z = np.arange(-59.0, 60.0, 2.0)
            xyz = np.column_stack([np.full_like(z, x), np.full_like(z, y), z])
            g = fold_truth(xyz) + rng.normal(0, 0.1, len(z))
            for zz, gg in zip(z, g, strict=True):
                rows.append((f"H{i}_{j}", x, y, zz, gg))
    samples = pd.DataFrame(rows, columns=["hole_id", "x", "y", "z", "au"])

    model = BlockModel(d)
    c = model.centroids()
    dip = np.degrees(np.arctan(fold_slope(c[:, 0])))  # semi-major axis down the layer
    add_angles(model, 0.0, 0.0, dip)
    truth = fold_truth(c)

    search = SearchParams(max_samples=16, min_samples=4, max_per_hole=3)
    # major axis = fold axis (north), semi-major = across strike in the layer,
    # minor = across the layering
    ell = Ellipsoid(120.0, 1.5, 12.0, (0.0, 0.0, 0.0))
    vario = VariogramModel(0.02, (Structure("spherical", 1.0, (200.0, 120.0, 10.0)),))

    results = {}
    for method in ("idw", "ok"):
        kw = {"variogram": vario} if method == "ok" else {}
        fixed = EstimationPass(method, ell, search, discretisation=(1, 1, 1), **kw)
        for label, p in (("fixed", fixed), ("dynamic", dataclasses.replace(fixed, dynamic=DYN))):
            m = BlockModel(d)
            add_angles(m, 0.0, 0.0, dip)
            estimate_attribute(m, samples, attribute="au", passes=[p], declustered_check=False)
            est = m["au"]
            ok = np.isfinite(est)
            assert ok.mean() > 0.95
            rmse = float(np.sqrt(np.mean((est[ok] - truth[ok]) ** 2)))
            corr = float(np.corrcoef(est[ok], truth[ok])[0, 1])
            results[(method, label)] = (rmse, corr)
            print(f"folded {method:3s} {label:7s}: RMSE {rmse:.3f}  corr {corr:.3f}")

    for method in ("idw", "ok"):
        fixed_rmse, fixed_corr = results[(method, "fixed")]
        dyn_rmse, dyn_corr = results[(method, "dynamic")]
        assert dyn_rmse < 0.8 * fixed_rmse
        assert dyn_corr > fixed_corr
