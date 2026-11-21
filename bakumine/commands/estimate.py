"""``bakumine estimate``: run an estimation plan (JSON) on a saved block model.

Composites come from a CSV or Parquet table (as written by ``bakumine composite``).
The plan is written by hand, by the dialog, or by ``bakumine estimate-plan``
(:mod:`bakumine.commands.estimate_plan`).
Geometry named in the plan's block constraints (solids, DTMs) is resolved from
``--geometry-dir`` (``<name>_vertices.npy`` + ``<name>_faces.npy``).
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import time
from collections.abc import Callable
from pathlib import Path

from bakumine.commands import estimate_plan
from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Estimate an attribute into a block model from composites with a parameter file"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="block model folder")
    parser.add_argument("--composites", required=True, help="composites table (CSV or Parquet)")
    parser.add_argument("--plan", required=True, help="estimation parameter JSON")
    parser.add_argument("--geometry-dir", metavar="DIR",
                        help="solids / DTMs the plan names")  # fmt: skip
    parser.add_argument("--overwrite", action="store_true",
                        help="estimate every block of the plan's constraints again; without it "
                        "(and without \"overwrite\": true in the plan) blocks that already "
                        "have a value are kept, so a second run changes nothing")  # fmt: skip
    parser.add_argument("--out", help="model folder to write (default: update MODEL in place)")
    parser.add_argument("--summary", metavar="JSON", help="also write the summary as JSON")
    parser.add_argument("--chunk-size", type=int, help="blocks per chunk (memory control)")
    parser.set_defaults(func=_guarded(run))


def _read_table(path: str):
    import pandas as pd

    file = Path(path)
    if not file.is_file():
        raise CommandError(f"{file} not found")
    if file.suffix.lower() in (".parquet", ".pq"):
        return pd.read_parquet(file)
    return pd.read_csv(file, dtype={"hole_id": str})


def _format(summary) -> str:
    frame = summary.to_frame()
    lines = [
        f"estimation of {summary.attribute}: {summary.n_estimated} of {summary.n_targeted} "
        f"blocks estimated, {summary.n_unestimated} not"
    ]
    if frame.empty:
        return "\n".join(lines)
    cols = ["domain", "number", "method", "n_targeted", "n_estimated", "n_unestimated",
            "mean_samples", "mean_estimate", "nn_mean", "seconds"]  # fmt: skip
    frame = frame[[c for c in cols if c in frame.columns]]
    lines.append(frame.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    for d in summary.domains:
        lines.append(
            f"domain {d.name}: samples {d.n_samples} (missing {d.n_samples_missing}, capped "
            f"{d.n_capped}), sample mean {d.sample_mean:.3f}, estimate mean {d.mean_estimate:.3f}"
        )
    return "\n".join(lines)


def format_times(loading: float, estimating: float, saving: float, passes: float) -> str:
    """The run time by step. ``passes`` is the time inside the estimation passes alone
    (the "seconds" column): the rest of the estimation step selects the blocks of each
    domain (constraints against solids and surfaces), prepares the samples and runs the
    nearest-neighbour check."""
    total = loading + estimating + saving
    return (
        f"time: total {total:.2f} s = loading {loading:.2f} s + estimation {estimating:.2f} s "
        f"(of which the passes {passes:.2f} s) + saving {saving:.2f} s"
    )


def run(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.constraints import FileResolver
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.estimation.block_estimation import EstimationPlan

    t0 = time.perf_counter()
    model = load_model(args.model)
    table = _read_table(args.composites)
    plan = EstimationPlan.load(args.plan)
    if args.overwrite and not plan.overwrite:
        plan = dataclasses.replace(plan, overwrite=True)
    resolver = FileResolver(args.geometry_dir) if args.geometry_dir else None
    t1 = time.perf_counter()
    summary = plan.run(model, table, resolver=resolver, chunk_size=args.chunk_size)
    t2 = time.perf_counter()
    print(_format(summary))
    if summary.n_targeted == 0 and not plan.overwrite and plan.attribute in model:
        print(
            f"note: no block was estimated. {plan.attribute} is already in the model, and "
            "blocks that have a value are kept: run again with --overwrite to estimate them "
            "anew (or the plan's constraints select no block)"
        )
    header = save_model(model, args.out or args.model)
    t3 = time.perf_counter()
    print(format_times(t1 - t0, t2 - t1, t3 - t2, sum(p.seconds for d in summary.domains
                                                      for p in d.passes)))  # fmt: skip
    _wrote(header.path, f"{header.n_blocks} blocks, {len(header.attributes)} attributes")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary.to_dict(), indent=2), encoding="utf-8")
        _wrote(path, "estimation summary")
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "estimate": (HELP, configure),
    **estimate_plan.COMMANDS,
}
