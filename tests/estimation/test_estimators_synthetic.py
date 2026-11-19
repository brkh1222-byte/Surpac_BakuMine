"""Cross-method sanity test on the synthetic deposit (P4.T3 / T4 / T6).

Ore-domain grades are estimated at the centroids of the truth blocks that
are more than half ore, from the ORE assay intervals only, with nearest
neighbour, IDW (power 2) and ordinary kriging, all with the same search.
Expected ordering of the error against the true block grades
(``au_ore``): OK <= IDW <= NN.

Set-up
------
* Samples: rows of ``truth_samples.csv`` with rock ORE (interval midpoints
  x, y, z); values = ``au`` of the row-aligned ``assay.csv`` (the rounded
  assays a user would have, not the unrounded truth).
* Variogram: the TRUE model of ``truth.json`` (spherical, nugget 0.15, sill
  0.85, ranges 120 / 70 / 15 m). It is the variogram of the Gaussian variable
  behind the lognormal grades; ordinary-kriging weights do not change when a
  variogram is multiplied by a constant, so it is used here as the relative
  structure of the grades (the kriging *variance* is then in Gaussian units
  and is not interpreted).
* Orientation: ``truth.json`` gives the axes as matrix columns (major,
  semi-major, minor); ``Ellipsoid.from_axes`` takes exactly those vectors.
* Search: the variogram's ellipsoid scaled to a 180 m major radius (ratios
  120/70 and 120/15), at most 16 samples, at least 1.
* Support: truth is a block average, so IDW and OK estimate 10 x 10 x 5 m
  blocks with a 4 x 4 x 2 discretisation; NN uses the centroid.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import ordinary_kriging
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel

SYNTH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"


@pytest.fixture(scope="module")
def deposit():
    truth = json.loads((SYNTH / "truth.json").read_text())
    smp = pd.read_csv(SYNTH / "truth_samples.csv")
    assay = pd.read_csv(SYNTH / "assay.csv")
    key = ["hole_id", "depth_from", "depth_to"]
    assert (smp[key].to_numpy() == assay[key].to_numpy()).all()  # row-aligned
    ore = (smp["rock"] == "ORE").to_numpy()
    xyz = smp.loc[ore, ["x", "y", "z"]].to_numpy()
    au = assay.loc[ore, "au"].to_numpy()
    holes = smp.loc[ore, "hole_id"].to_numpy()
    blocks = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    blocks = blocks[blocks["ore_fraction"] > 0.5]
    return truth, xyz, au, holes, blocks


def test_rmse_ordering_ok_idw_nn(deposit):
    truth, xyz, au, holes, blocks = deposit
    centroids = blocks[["x", "y", "z"]].to_numpy()
    true = blocks["au_ore"].to_numpy()
    size = truth["blocks"]["size"]

    v = truth["variogram"]
    axes = np.array(v["axes"])
    a_major, a_semi, a_minor = v["ranges_m"]
    range_ell = Ellipsoid.from_axes(
        axes[:, 0], axes[:, 1], axes[:, 2], a_major, a_major / a_semi, a_major / a_minor
    )
    # the convention of truth.json really is what from_axes was given
    np.testing.assert_allclose(range_ell.major_axis, axes[:, 0], atol=1e-12)
    np.testing.assert_allclose(range_ell.semi_major_axis, axes[:, 1], atol=1e-12)
    assert abs(range_ell.minor_axis @ axes[:, 2]) == pytest.approx(1.0)
    assert v["model"] == "spherical"
    structure = Structure.from_ellipsoid("spherical", v["sill"], range_ell)
    model = VariogramModel(v["nugget"], (structure,))
    search = Ellipsoid.from_axes(
        axes[:, 0], axes[:, 1], axes[:, 2], 180.0, a_major / a_semi, a_major / a_minor
    )
    params = SearchParams(max_samples=16, min_samples=1)
    block = dict(block_size=size, discretisation=(4, 4, 2))

    nn = nearest_neighbour(xyz, au, centroids, search)
    idw = inverse_distance(xyz, au, centroids, search, params, power=2.0, **block)
    ok = ordinary_kriging(xyz, au, centroids, model, search, params, **block)
    assert ok.n_singular == 0 and ok.n_invalid == 0
    both = nn.estimated & idw.estimated & ok.estimated
    assert both.mean() > 0.99, both.mean()
    np.testing.assert_array_equal(nn.estimated, idw.estimated)
    np.testing.assert_array_equal(idw.estimated, ok.estimated)

    def score(est):
        err = est[both] - true[both]
        return float(np.sqrt(np.mean(err**2))), float(np.corrcoef(est[both], true[both])[0, 1])

    rmse_nn, r_nn = score(nn.estimate)
    rmse_idw, r_idw = score(idw.estimate)
    rmse_ok, r_ok = score(ok.estimate)
    print(
        f"\nblocks {both.sum()} of {len(both)}  true mean {true[both].mean():.4f}\n"
        f"NN    rmse {rmse_nn:.4f}  r {r_nn:.4f}  mean {nn.estimate[both].mean():.4f}\n"
        f"IDW2  rmse {rmse_idw:.4f}  r {r_idw:.4f}  mean {idw.estimate[both].mean():.4f}\n"
        f"OK    rmse {rmse_ok:.4f}  r {r_ok:.4f}  mean {ok.estimate[both].mean():.4f}"
    )
    assert rmse_ok <= rmse_idw <= rmse_nn, (rmse_ok, rmse_idw, rmse_nn)
    assert r_ok >= r_idw >= r_nn, (r_ok, r_idw, r_nn)
    # all three are globally unbiased to within 5 % of the true mean
    for est in (nn.estimate, idw.estimate, ok.estimate):
        assert est[both].mean() == pytest.approx(true[both].mean(), rel=0.05)
