"""Gate A + Gate B from the command line, on the committed synthetic deposit.

    uv run python scripts/gate_synthetic.py [OUT_DIR] [--depth N]

Steps (each a ``bakumine`` command): import -> validate -> desurvey -> composite
(2 m, geology.rock domains) -> blockmodel create (truth grid) -> blockmodel code
(AIR above the topography, ORE inside the ore solid, density; sub-blocked) ->
estimate (OK in 3 passes with the true variogram) -> blockmodel report.
Then it compares the model with ``truth.json`` / ``truth_blocks.parquet`` and writes
``metrics.json`` into OUT_DIR (default ``data/synthetic/gate_out``, git-ignored).
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
SYNTH = REPO / "data" / "synthetic" / "default"
DEFAULT_OUT = REPO / "data" / "synthetic" / "gate_out"


def cli(*argv: object) -> None:
    from bakumine.cli import main

    code = main([str(a) for a in argv])
    if code != 0:
        raise SystemExit(f"bakumine {' '.join(map(str, argv[:2]))} failed with exit code {code}")


def write_configs(out: Path) -> tuple[Path, Path]:
    """The coding file and the estimation parameter file (true variogram, 3 passes)."""
    from bakumine.core.estimation.block_estimation import (
        EstimationDomain,
        EstimationPass,
        EstimationPlan,
    )
    from bakumine.core.estimation.rotation import Ellipsoid
    from bakumine.core.estimation.search import SearchParams
    from bakumine.core.estimation.variogram import Structure, VariogramModel

    truth = json.loads((SYNTH / "truth.json").read_text())
    density = truth["density_t_m3"]
    coding = {
        "attributes": [
            {"name": "rock", "kind": "category", "background": "WASTE"},
            {"name": "density", "kind": "float", "unit": "t/m3"},
        ],
        "steps": [
            {"attribute": "rock", "value": "AIR", "subblock": True,
             "constraint": {"type": "dtm", "surface": "topo", "side": "above", "outside": True}},
            {"attribute": "rock", "value": "ORE", "subblock": True,
             "constraint": {"type": "solid", "solid": "ore_solid", "side": "inside"}},
            {"attribute": "density",
             "value": f'iif(rock == "ORE", {density["ORE"]}, {density["WASTE"]})',
             "constraint": {"type": "attribute", "expression": 'rock != "AIR"'}},
        ],
    }  # fmt: skip
    coding_path = out / "coding.json"
    coding_path.write_text(json.dumps(coding, indent=2))

    v = truth["variogram"]
    axes = np.array(v["axes"])
    major, semi, minor = v["ranges_m"]
    ellipsoid = Ellipsoid.from_axes(
        axes[:, 0], axes[:, 1], axes[:, 2], major, major / semi, major / minor
    )
    structure = Structure.from_ellipsoid("spherical", v["sill"], ellipsoid)
    model = VariogramModel(v["nugget"], (structure,))
    tight = EstimationPass(
        "ok", ellipsoid, SearchParams(max_samples=16, min_samples=6, max_per_hole=4),
        radius_factor=0.25, variogram=model, discretisation=(4, 4, 2), name="tight",
    )  # fmt: skip
    passes = [
        tight,
        tight.scaled(0.5, search_min_samples=3, name="medium"),
        tight.scaled(1.5, search_min_samples=1, search_max_per_hole=None, name="wide"),
    ]
    domain = EstimationDomain(
        "ORE", passes,
        block_constraint={"type": "attribute", "expression": 'rock == "ORE"'},
        sample_filter={"column": "domain", "equals": "ORE"},
    )  # fmt: skip
    plan_path = out / "plan.json"
    EstimationPlan("au", [domain]).save(plan_path)
    return coding_path, plan_path


def run(out: Path, depth: int = 1) -> dict:
    from bakumine.core.blockmodel.storage import load_model

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    truth = json.loads((SYNTH / "truth.json").read_text())
    info = truth["blocks"]
    db, model = out / "db", out / "model"
    coding, plan = write_configs(out)

    cli("import", "--collar", SYNTH / "collar.csv", "--survey", SYNTH / "survey.csv",
        "--assay", SYNTH / "assay.csv", "--geology", SYNTH / "geology.csv", "--out", db, "--auto",
        )  # fmt: skip
    cli("validate", db)
    cli("desurvey", db)
    cli("composite", db, "--table", "assay", "--mode", "domain", "--length", 2,
        "--domain-table", "geology", "--domain-column", "rock")  # fmt: skip
    cli("blockmodel", "create", model, "--origin", ",".join(map(str, info["origin"])),
        "--size", ",".join(map(str, info["size"])), "--shape", ",".join(map(str, info["shape"])),
        "--max-depth", depth)  # fmt: skip
    cli("blockmodel", "code", model, "--steps", coding, "--geometry-dir", SYNTH, "--merge")
    coded = load_model(model)
    ore = coded.get("rock", decode=True) == "ORE"
    gate_a = float(coded.volumes()[ore].sum())
    cli("estimate", model, "--composites", db / "composites_assay.parquet", "--plan", plan,
        "--geometry-dir", SYNTH, "--summary", out / "estimation_summary.json")  # fmt: skip
    cli("blockmodel", "report", model, "--where", 'rock == "ORE"', "--grade", "au:g/t",
        "--density", "density", "--out", out / "report.csv")  # fmt: skip

    # ---- compare with the truth -------------------------------------------------------
    m = load_model(model)
    sel = m.get("rock", decode=True) == "ORE"
    vol, au, dens = m.volumes()[sel], m.get("au")[sel], m.get("density")[sel]
    ok = np.isfinite(au)
    tonnes = float((vol * dens).sum())
    metal = float((vol * dens * au)[ok].sum())
    # parent-level correlation with the true block grades
    table = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    flat = m.definition.flat_index(table[["i", "j", "k"]].to_numpy())
    n_parents = m.definition.n_parents
    true_au = np.full(n_parents, np.nan)
    true_au[flat] = table["au_ore"].to_numpy()
    parent = m.parent[sel][ok]
    sums = np.bincount(parent, vol[ok] * au[ok], n_parents)
    wts = np.bincount(parent, vol[ok], n_parents)
    est = np.divide(sums, wts, out=np.full_like(sums, np.nan), where=wts > 0)
    both = np.isfinite(est) & np.isfinite(true_au)
    totals = truth["totals"]
    metrics = {
        "blocks": int(m.n_blocks),
        "ore_volume_m3": gate_a,
        "truth_ore_volume_m3": totals["ore_volume_m3"],
        "volume_error": gate_a / totals["ore_volume_m3"] - 1,
        "ore_tonnes": tonnes,
        "truth_ore_tonnes": totals["ore_tonnes"],
        "tonnes_error": tonnes / totals["ore_tonnes"] - 1,
        "ore_metal_g": metal,
        "truth_ore_metal_g": totals["ore_metal_g"],
        "metal_error": metal / totals["ore_metal_g"] - 1,
        "ore_blocks_unestimated": int((~ok).sum()),
        "n_parents_compared": int(both.sum()),
        "correlation": float(np.corrcoef(est[both], true_au[both])[0, 1]),
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--depth", type=int, default=1, help="sub-block depth (default 1)")
    args = parser.parse_args(argv)
    metrics = run(args.out, args.depth)
    print("\n== gate metrics ==")
    for key, value in metrics.items():
        print(f"{key:24} {value:.6g}" if isinstance(value, float) else f"{key:24} {value}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
