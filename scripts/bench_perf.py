"""Performance benchmark on a 5M-block model (plan P7.T4).

    uv run python scripts/bench_perf.py [--scale 1.0] [--out data/bench_perf.json] [--profile STEP]

Builds a 200 x 200 x 125 parent grid (5M blocks, 10 x 10 x 5 m, max depth 3), 20k random
composites, and times with numba warm: IDW 1M blocks, OK block kriging 1M blocks (24 samples),
3-pass ``estimate_attribute`` on 5M blocks, sub-blocking + coding to depth 3, save / load.
``--scale`` shrinks the linear grid size and composite count for a smoke run.
"""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.coding import code_model
from bakumine.core.blockmodel.constraints import DictResolver, SolidConstraint
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.estimation.block_estimation import EstimationPass, estimate_attribute
from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel
from bakumine.core.geometry.primitives import icosphere

TARGETS = {"idw_1M": 60.0, "ok_1M": 300.0}


def build(scale: float, seed: int = 7):
    n = max(4, round(200 * scale))
    shape = (n, n, max(2, round(125 * scale)))
    definition = BlockModelDefinition((0, 0, 0), (10, 10, 5), shape, max_depth=3)
    model = BlockModel(definition)
    rng = np.random.default_rng(seed)
    ext = np.array(shape) * np.array([10, 10, 5.0])
    m = max(200, round(20_000 * scale**2))
    xyz = rng.uniform(0, 1, (m, 3)) * ext
    val = np.exp(0.8 * rng.standard_normal(m) + 0.002 * xyz[:, 0])
    samples = pd.DataFrame({"x": xyz[:, 0], "y": xyz[:, 1], "z": xyz[:, 2], "au": val})
    return model, samples, ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--out", default="data/bench_perf.json")
    ap.add_argument("--profile", default=None, help="step name to cProfile")
    ap.add_argument("--only", default=None, help="comma list of steps to run")
    args = ap.parse_args()

    model, samples, ext = build(args.scale)
    xyz = samples[["x", "y", "z"]].to_numpy()
    vals = samples["au"].to_numpy()
    n_blocks = model.n_blocks
    n1m = min(1_000_000, n_blocks)
    sel = model.centroids()[:n1m]
    ell = Ellipsoid(major_radius=120.0, semi_major_ratio=1.5, minor_ratio=3.0, angles=(30, 0, 0))
    params = SearchParams(max_samples=24, min_samples=1)
    vario = VariogramModel(0.2, (Structure.from_ellipsoid("spherical", 0.8, ell),))
    disc = (2, 2, 2)
    block = (10.0, 10.0, 5.0)
    only = args.only.split(",") if args.only else None

    results: dict[str, dict] = {}

    def timed(name, fn, warm=None):
        if only and name not in only:
            return None
        if warm:
            warm()
        prof = cProfile.Profile() if args.profile == name else None
        t0 = time.perf_counter()
        if prof:
            prof.enable()
        out = fn()
        if prof:
            prof.disable()
        dt = time.perf_counter() - t0
        results[name] = {"seconds": round(dt, 3), "target": TARGETS.get(name)}
        print(f"{name:>22s} {dt:9.2f} s", flush=True)
        if prof:
            pstats.Stats(prof).sort_stats("cumulative").print_stats(18)
        return out

    small = slice(0, 2000)
    r = timed(
        "idw_1M",
        lambda: inverse_distance(
            xyz, vals, sel, ell, params, power=2.0, block_size=block, discretisation=disc
        ),
        warm=lambda: inverse_distance(
            xyz, vals, sel[small], ell, params, block_size=block, discretisation=disc
        ),
    )
    if r is not None:
        results["idw_1M"]["estimated_fraction"] = float(r.estimated.mean())
        results["idw_1M"]["mean_samples"] = float(r.n_samples.mean())
    r = timed(
        "ok_1M",
        lambda: krige(xyz, vals, sel, vario, ell, params, block_size=block, discretisation=disc),
        warm=lambda: krige(
            xyz, vals, sel[small], vario, ell, params, block_size=block, discretisation=disc
        ),
    )

    if r is not None:
        results["ok_1M"]["estimated_fraction"] = float(r.estimated.mean())
        results["ok_1M"]["mean_samples"] = float(r.n_samples.mean())

    p1 = EstimationPass("idw", ell, params, radius_factor=0.5, discretisation=(1, 1, 1), name="a")
    passes = [p1, p1.scaled(1.0, name="b"), p1.scaled(2.0, name="c")]

    def warm_mp():
        tiny = BlockModel(BlockModelDefinition((0, 0, 0), (10, 10, 5), (8, 8, 4), max_depth=3))
        estimate_attribute(tiny, samples, attribute="au", passes=passes)

    summ = timed(
        "estimate_5M_3pass",
        lambda: estimate_attribute(model, samples, attribute="au", passes=passes),
        warm=warm_mp,
    )
    if summ is not None:
        results["_au_checksum"] = float(np.nansum(model.get("au")))

    sphere = icosphere(3, radius=0.3 * float(ext.min()), center=tuple(ext / 2 + 1.37))
    resolver = DictResolver(solids={"s": sphere})
    steps = [
        {
            "attribute": "rock",
            "value": "ORE",
            "subblock": True,
            "max_level": 3,
            "constraint": SolidConstraint("s", "inside"),
        }
    ]
    if "rock" not in model.attributes:
        model.add_attribute("rock", "category", background="WASTE")
    timed("subblock_code_d3", lambda: code_model(model, steps, resolver, merge=True))
    results["n_blocks_after_subblock"] = int(model.n_blocks)

    tmp = Path(tempfile.mkdtemp())
    try:
        timed("save", lambda: save_model(model, tmp / "m"))
        timed("load", lambda: load_model(tmp / "m"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    print("\n step                     seconds   target")
    for k, v in results.items():
        if isinstance(v, dict):
            t = v["target"]
            flag = "" if t is None else ("OK" if v["seconds"] < t else "MISS")
            print(f"{k:>22s} {v['seconds']:9.2f} {t or '':>8} {flag}")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"scale": args.scale, "blocks": n_blocks, **results}, indent=2))


if __name__ == "__main__":
    main()
