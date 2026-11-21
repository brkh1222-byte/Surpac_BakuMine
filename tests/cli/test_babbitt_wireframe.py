"""P8.T12: scripts/run_babbitt_wireframe.py against the published PyGSLIB tutorial (~10 s).

The tutorial (source pinned in data/public/babbitt/SOURCE.md) estimates Cu inside the public
wireframe ``Mpz.stl`` with GSLIB ``kt3d`` and prints block means and data statistics.  The run
repeats it with the tutorial's parameters through the ``bakumine`` CLI; the numbers here pin
how close an independent implementation gets.

Tolerances
----------
``CLOSE`` = 0.5 % for every quantity computed the same way on both sides.  Measured
differences are 0.00 to 0.18 %; what is left differs by construction and was not separated
further: composite midpoints (actual versus nominal interval), the desurvey method, float32
coordinates in PyGSLIB, the origin of the declustering grid, the repaired wireframe.  0.5 % is
about three times the largest measured value, and five times tighter than the 2.3 % spread
between the tutorial's own three estimators, so a wrong unit, convention or domain cannot
hide in it.

Two quantities are NOT computed the same way, and are pinned as such, not loosened:

* Ordinary kriging over all blocks is 0.9 % below the published mean.  Cause (measured
  here, inferred for the tutorial from the ``kt3d`` source; PyGSLIB was not run): two pairs
  of twin holes (B1-184 / B1-184B, B1-100A / B1-100B) put 95 pairs of composites at
  identical coordinates.  ``kt3d`` does not merge duplicates, its matrix is singular for a
  neighbourhood holding such a pair and the block gets no kriged value, so it drops out of
  the tutorial's mean; ``bakumine`` merges the duplicates and estimates the block.  Those
  blocks are low grade (0.10 % Cu).  Without them our mean is within ``CLOSE``.
* ``bakumine`` inverse distance weights by the anisotropic search distance, the tutorial's
  by the isotropic distance: +1.2 %.  Recomputed with isotropic weights from the same
  neighbourhoods it is within ``CLOSE``.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
CLOSE = 0.005


@pytest.fixture(scope="module")
def metrics(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "run_babbitt_wireframe", REPO / "scripts" / "run_babbitt_wireframe.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("babbitt_wireframe") / "run"
    result = module.run(out)
    assert json.loads((out / "metrics.json").read_text())["ours"] == result["ours"]
    for name in ("model/model.json", "report.csv", "plan_ok.json", "mpz_closed.obj",
                 "geometry/mpz_vertices.npy", "estimation_summary_ok.json"):  # fmt: skip
        assert (out / name).is_file(), name
    return result


def test_published_numbers_are_the_tutorials(metrics):
    # as printed in doc/source/Tutorial.md of the pinned commit
    p = metrics["published"]
    assert (p["ok_mean"], p["id2_mean"], p["nn_mean"]) == (0.21166003, 0.20810474, 0.20687895)
    assert p["data_mean"] == 0.24149141734342264
    assert p["declustered_mean"] == 0.2251903672467954


def test_wireframe_repair_and_units(metrics):
    w = metrics["wireframe"]
    # as published: 63 749 triangles, three double-sided fins and one missing triangle
    assert (w["triangles_in_file"], w["fin_triangles_removed"], w["hole_triangles_added"]) == (
        63749, 6, 1)  # fmt: skip
    assert w["closed"] and w["outward"]
    assert w["n_faces"] == 63749 - 6 + 1
    # the repair leaves one vertex where two sheets of the surface touch
    assert w["issues"] == ["1 non-manifold vertices"]
    # feet -> metres on both sides: the solid holds 46 % of the composites and 79 % of the
    # Cu metal, at over four times the grade outside.  With the wireframe left in feet no
    # composite would be inside at all.
    d = metrics["domain"]
    assert d["composites_inside"] == pytest.approx(24780, abs=25)
    assert d["holes_with_composites_inside"] >= 330
    assert d["share_of_cu_metal_inside"] > 0.75
    assert d["mean_cu_inside"] > 4.0 * d["mean_cu_outside"]


def test_data_statistics_match(metrics):
    ours, pub, rel = metrics["ours"], metrics["published"], metrics["relative_difference"]
    assert ours["n_composites"] == pub["n_composites"] == 54189
    assert ours["data_max"] == pytest.approx(pub["data_max"], rel=1e-6)
    assert abs(rel["data_mean"]) < CLOSE  # measured +0.01 %
    assert abs(rel["declustered_mean"]) < CLOSE  # measured -0.12 %
    assert abs(rel["declustered_variance"]) < CLOSE  # measured -0.18 %
    assert abs(rel["data_cv_declustered"]) < CLOSE  # measured +0.03 %


def test_volume_of_the_coded_domain(metrics):
    v, b = metrics["volume"], metrics["blocks"]
    # the wireframe reaches 54 ft south of the tutorial's model; that part (0.4 %) cannot be
    # coded and is measured separately
    assert 0.003 < v["mesh_south_of_model_m3"] / v["mesh_m3"] < 0.005
    # block fractions (column method, exact in Z, 5 x 5 in plan): measured -0.02 %; the
    # documented accuracy of the method for a total is a few 0.01 %
    assert abs(v["partial_vs_mesh_inside_model"]) < 0.001
    # whole blocks by centroid, 30.48 x 30.48 x 9.144 m blocks, no sub-blocks: measured
    # +0.20 %; the errors of the boundary blocks only cancel statistically
    assert abs(v["centroid_vs_mesh_inside_model"]) < 0.005
    assert b["touched_d1_gt_0"] > b["centroid_inside"] > 200_000
    assert 0 < b["not_estimated"] < 0.15 * b["touched_d1_gt_0"]  # fewer than 4 samples in reach


def test_nearest_neighbour_mean(metrics):
    assert abs(metrics["relative_difference"]["nn_mean"]) < CLOSE  # measured +0.12 %


def test_ordinary_kriging_mean(metrics):
    d, rel = metrics["diagnostics"], metrics["relative_difference"]
    # on the blocks kt3d can solve: measured +0.16 %
    assert abs(d["ok_without_those_blocks_vs_published"]) < CLOSE
    # all blocks: lower, because the blocks around the twin holes are estimated (see the
    # module docstring); measured -0.86 %
    assert d["twin_holes"] == ["B1-100A = B1-100B", "B1-184 = B1-184B"]
    assert d["coincident_sample_pairs_in_domain"] == 95
    share = d["blocks_with_coincident_samples"] / metrics["blocks"]["estimated"]
    assert 0.015 < share < 0.022  # measured 1.86 %
    assert d["ok_mean_in_those_blocks"] < 0.5 * metrics["ours"]["ok_mean"]
    assert -0.012 < rel["ok_mean"] < -0.005


def test_inverse_distance_mean(metrics):
    d, rel = metrics["diagnostics"], metrics["relative_difference"]
    # the tutorial's definition (isotropic weights), from our neighbourhoods: measured +0.17 %
    assert abs(d["id2_isotropic_vs_published"]) < CLOSE
    # bakumine's own (anisotropic weights): a different estimator, measured +1.23 %
    assert 0.005 < rel["id2_mean"] < 0.02


def test_missing_stays_missing_variant(metrics):
    """Not a comparison with the tutorial: what its zero-filling does.  A third of the
    composites in the wireframe were never assayed; left missing (bakumine's default) the
    grade is about half as high again."""
    v, d = metrics["variant_missing_stays_missing"], metrics["domain"]
    assert v["composites_inside_with_cu"] == d["composites_inside_originally_assayed"]
    assert 0.6 < v["composites_inside_with_cu"] / d["composites_inside"] < 0.68
    assert v["ok_mean"] > 1.4 * metrics["ours"]["ok_mean"]
    assert v["data_mean"] > 1.5 * metrics["ours"]["data_mean"]


def test_report_uses_block_fractions(metrics):
    t, v = metrics["tonnage"], metrics["volume"]
    total = t["report_rows"][0]
    assert total["cutoff"] == 0.0 and total["blocks"] == metrics["blocks"]["estimated"]
    assert total["tonnes"] == pytest.approx(total["volume"] * 2.7, rel=1e-9)
    assert total["tonnes"] / 1e6 == pytest.approx(t["estimated_partial_Mt"], rel=1e-9)
    assert total["volume"] < v["partial_m3"]  # unestimated blocks are not in the report
    assert total["cu_grade"] == pytest.approx(
        metrics["diagnostics"]["ok_mean_weighted_by_d1"], rel=1e-9
    )
