"""Real-data run: the Babbitt Cu-Ni drillholes (data/public/babbitt) through the CLI.

    uv run python scripts/run_babbitt.py [OUT_DIR] [--no-cv]

Choices (also written to ``metrics.json`` under ``choices``):

* Units: the files are in feet, dips positive downward, with a sentinel survey station at
  90000 ft.  ``bakumine import --length-unit ft --to-metres --dip-convention positive_down
  --normalise-dip`` converts to metres and negative-down dips; the sentinel stations are
  reported by ``validate`` / ``desurvey`` as warnings and ignored past the hole end.
* Composites: 10 ft = 3.048 m downhole, short remnants merged.  Cu only is estimated (Ni, S are
  composited too).  Blank assays stay NaN.
* Domain: NONE.  There is no geology table and the mineralised-zone wireframe is not shipped,
  so every block with enough nearby composites is estimated.  The report therefore covers
  "estimated blocks", not a resource inside an ore shell.
* Block model: parent 100 x 100 x 20 m (median nearest-hole spacing is ~100 m; blocks
  smaller than that are mostly extrapolation), no sub-blocks, extent from the drillholes.
* Variogram: experimental Cu variograms along E / N / Z are fitted with ``fit_variogram``.  The
  fit is accepted only if it is good and geologically sensible (horizontal range >= vertical,
  R^2 >= 0.5); otherwise a manual spherical model is used (see ``variogram`` in metrics.json).
* Density: 2.9 t/m3, an ASSUMPTION for Duluth Complex gabbro-hosted sulphide, not measured.
* Estimation: ordinary kriging in 3 passes (0.5, 1.0, 1.5 x search) plus a nearest-neighbour
  model of the same blocks for the global-mean check.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "public" / "babbitt"
DEFAULT_OUT = REPO / "data" / "reference" / "babbitt_out"
FT = 0.3048
DENSITY = 2.9
BLOCK = (100.0, 100.0, 20.0)
SEARCH_MAJOR = 250.0  # metres, horizontal; vertical = major / VERT_RATIO
VERT_RATIO = 2.5
# No extrapolation more than 10 m above / below a composite: holes end in barren footwall and
# blocks below hole ends otherwise take the grade of the last composites (see metrics).
MAX_VERTICAL = 10.0
MANUAL = {"nugget": 0.085, "sill": 0.06, "ranges_m": [200.0, 200.0, 100.0]}


def cli(*argv: object) -> None:
    from bakumine.cli import main

    code = main([str(a) for a in argv])
    if code != 0:
        raise SystemExit(f"bakumine {' '.join(map(str, argv[:2]))} failed with exit code {code}")


def search_ellipsoid():
    from bakumine.core.estimation.rotation import Ellipsoid

    return Ellipsoid.from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1), SEARCH_MAJOR, 1.0, VERT_RATIO)


def choose_variogram(comp: pd.DataFrame) -> tuple[object, dict]:
    from bakumine.core.estimation.experimental import (
        experimental_variogram,
        fit_variogram,
        principal_directions,
    )
    from bakumine.core.estimation.rotation import Ellipsoid
    from bakumine.core.estimation.variogram import Structure, VariogramModel

    xyz, cu = comp[["x", "y", "z"]].to_numpy(), comp["cu"].to_numpy()
    ell = Ellipsoid.from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1), 300.0, 1.0, 3.0)
    exp = experimental_variogram(
        xyz, cu, lag_size=[40, 40, 10], n_lags=[10, 10, 15],
        directions=list(principal_directions(ell)), hole_ids=comp["hole_id"].to_numpy(),
    )  # fmt: skip
    fit = fit_variogram(exp, ellipsoid=ell, types="spherical", max_distance=[400, 400, 150],
                        min_pairs=100)  # fmt: skip
    r = fit.ranges[0]
    info = {
        "fit": {"nugget": float(fit.nugget), "sill": float(fit.sills[0]),
                "ranges_m_EW_NS_Z": [float(v) for v in r], "r_squared": float(fit.r_squared),
                "at_bounds": list(fit.at_bounds), "sample_variance": float(exp[0].variance)},
    }  # fmt: skip
    # fit ranges are ordered along the model axes E, N, Z here
    sensible = fit.r_squared >= 0.5 and min(r[0], r[1]) >= r[2]
    if sensible:
        info["used"] = "fitted"
        return fit.model, info
    info["used"] = "manual"
    info["manual"] = MANUAL
    mr = MANUAL["ranges_m"]
    ellipsoid = Ellipsoid.from_axes((1, 0, 0), (0, 1, 0), (0, 0, 1), mr[0], mr[0] / mr[1],
                                    mr[0] / mr[2])  # fmt: skip
    model = VariogramModel(MANUAL["nugget"], (Structure.from_ellipsoid(
        "spherical", MANUAL["sill"], ellipsoid),))  # fmt: skip
    return model, info


def make_passes(method: str, model):
    from bakumine.core.estimation.block_estimation import EstimationPass
    from bakumine.core.estimation.search import SearchParams

    ell = search_ellipsoid()
    if method == "nn":
        return [EstimationPass("nn", ell, SearchParams(max_samples=1, max_vertical=MAX_VERTICAL),
                               radius_factor=1.5,
                               name="nn")]  # fmt: skip
    first = EstimationPass("ok", ell, SearchParams(max_samples=16, min_samples=6, max_per_hole=4,
                                                   max_vertical=MAX_VERTICAL),
                           radius_factor=0.5, variogram=model, discretisation=(3, 3, 2),
                           name="tight")  # fmt: skip
    return [first, first.scaled(1.0, search_min_samples=3, name="medium"),
            first.scaled(1.5, search_min_samples=1, search_max_per_hole=None,
                         name="wide")]  # fmt: skip


def write_plans(out: Path, model) -> tuple[Path, Path]:
    from bakumine.core.estimation.block_estimation import (
        EstimationDomain,
        EstimationPlan,
        OutputNames,
    )

    ok = EstimationPlan("cu", [EstimationDomain("all", make_passes("ok", model))])
    nn = EstimationPlan("cu_nn", [EstimationDomain("all", make_passes("nn", model))],
                        value_column="cu", outputs=OutputNames.only())  # fmt: skip
    ok.save(out / "plan_ok.json")
    nn.save(out / "plan_nn.json")
    return out / "plan_ok.json", out / "plan_nn.json"


def swath_checks(m, comp: pd.DataFrame) -> dict:
    from bakumine.core.estimation.validation import swath_table

    ok = np.isfinite(m.get("cu"))
    coords, vol, val = m.centroids()[ok], m.volumes()[ok], m.get("cu")[ok]
    result = {}
    for axis, width in (("x", 200.0), ("y", 200.0), ("z", 40.0)):
        t = swath_table(coords, val, comp[["x", "y", "z"]].to_numpy(), comp["cu"].to_numpy(),
                        width=width, axis=axis, block_weights=vol)  # fmt: skip
        use = t[(t.block_count >= 5) & (t.sample_count >= 20)]
        corr = float(np.corrcoef(use.block_mean, use.sample_mean)[0, 1]) if len(use) > 2 else None
        result[axis] = {"width_m": width, "slices_used": int(len(use)), "correlation": corr,
                        "table": [{k: (None if pd.isna(v) else float(v)) for k, v in row.items()}
                                  for row in use.to_dict("records")]}  # fmt: skip
    return result


def run(out: Path, cross_validation: bool = True) -> dict:
    from bakumine.core.blockmodel.storage import load_model
    from bakumine.core.estimation.search import SearchParams
    from bakumine.core.estimation.validation import (
        cell_declustering_weights,
        cross_validate,
        global_comparison,
    )

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    db, model_dir = out / "db", out / "model"

    cli("import", "--collar", SRC / "collar.csv", "--survey", SRC / "survey.csv",
        "--assay", SRC / "assay.csv", "--out", db, "--auto", "--length-unit", "ft",
        "--to-metres", "--dip-convention", "positive_down", "--normalise-dip")  # fmt: skip
    cli("validate", db)
    cli("desurvey", db)
    cli("composite", db, "--table", "assay", "--mode", "downhole", "--length", 10 * FT,
        "--attributes", "cu,ni,s", "--short", "merge", "--min-length", 0.5)  # fmt: skip
    comp_path = db / "composites_assay.parquet"
    comp = pd.read_parquet(comp_path)
    cu_comp = comp.dropna(subset=["cu"]).reset_index(drop=True)

    model, vinfo = choose_variogram(cu_comp)
    plan_ok, plan_nn = write_plans(out, model)

    cli("blockmodel", "create", model_dir, "--size", ",".join(map(str, BLOCK)),
        "--extent-from", f"drillholes:{db}", "--padding", "50,50,0", "--max-depth", 0)  # fmt: skip
    cli("estimate", model_dir, "--composites", comp_path, "--plan", plan_ok,
        "--summary", out / "estimation_summary_ok.json")  # fmt: skip
    cli("estimate", model_dir, "--composites", comp_path, "--plan", plan_nn,
        "--summary", out / "estimation_summary_nn.json")  # fmt: skip
    cli("blockmodel", "report", model_dir, "--grade", "cu:%", "--density", DENSITY,
        "--cutoffs", "0,0.2,0.3,0.4", "--out", out / "report.csv")  # fmt: skip

    m = load_model(model_dir)
    vol = m.volumes()
    cu, nn = m.get("cu"), m.get("cu_nn")
    both = np.isfinite(cu) & np.isfinite(nn)
    weights = cell_declustering_weights(cu_comp[["x", "y", "z"]].to_numpy(), 100.0)
    g = global_comparison(cu[both], nn[both], cu_comp["cu"].to_numpy(), block_weights=vol[both],
                          sample_weights=weights)  # fmt: skip
    est = np.isfinite(cu)
    metrics = {
        "choices": {
            "units": "feet -> metres at import; dips positive_down -> negative_down",
            "composite_length_m": 10 * FT, "domain": "none (no geology table)",
            "block_size_m": list(BLOCK), "sub_blocks": False,
            "search_major_m": SEARCH_MAJOR, "search_vertical_m": SEARCH_MAJOR / VERT_RATIO,
            "density_t_m3": DENSITY, "density_is_assumption": True,
        },  # fmt: skip
        "variogram": vinfo,
        "composites": {"n_total": int(len(comp)), "n_with_cu": int(len(cu_comp)),
                       "mean_cu_pct": float(cu_comp.cu.mean())},  # fmt: skip
        "blocks": {"n": int(m.n_blocks), "estimated": int(est.sum()),
                   "volume_estimated_m3": float(vol[est].sum()),
                   "tonnes_estimated": float(vol[est].sum() * DENSITY),
                   "mean_cu_pct_volume_weighted": float(
                       np.average(cu[est], weights=vol[est]))},  # fmt: skip
        "global_mean": {
            "ok_mean": g.block_mean, "nn_mean": g.nn_mean, "naive_sample_mean": g.naive_mean,
            "declustered_sample_mean": g.declustered_mean, "n_blocks": g.n_blocks,
            "ok_vs_nn": g.block_vs_nn, "ok_vs_declustered": g.block_vs_declustered,
            "nn_vs_declustered": g.nn_vs_declustered,
        },  # fmt: skip
        "swath": swath_checks(m, cu_comp),
    }
    if cross_validation:
        cv = cross_validate(cu_comp[["x", "y", "z"]].to_numpy(), cu_comp["cu"].to_numpy(),
                            search_ellipsoid(), SearchParams(max_samples=16, min_samples=3,
                                                             max_per_hole=4),
                            method="ok", model=model, leave="hole",
                            hole_ids=cu_comp["hole_id"].to_numpy())  # fmt: skip
        metrics["cross_validation_leave_one_hole_out"] = {
            k: float(v) for k, v in cv.summary.items()} | {"n_estimated": int(cv.estimated.sum())}
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--no-cv", action="store_true", help="skip cross-validation")
    args = parser.parse_args(argv)
    m = run(args.out, not args.no_cv)
    for key in ("variogram", "blocks", "global_mean"):
        print(key, json.dumps(m[key], indent=1, default=str))
    for axis, s in m["swath"].items():
        print("swath", axis, "slices", s["slices_used"], "corr", s["correlation"])
    print("cv", m.get("cross_validation_leave_one_hole_out"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
