"""`bakumine synth` subcommand. Kept import-light: the generator loads only when it runs."""

from __future__ import annotations

import argparse

HELP = "Generate the synthetic benchmark deposit with known truth"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, help="output folder, e.g. data/synthetic/default")
    parser.add_argument("--seed", type=int, default=42, help="random seed (default: 42)")
    parser.add_argument(
        "--preset",
        choices=("default", "small"),
        default="default",
        help="deposit size: 'default' is the benchmark, 'small' is a quick reduced deposit",
    )
    parser.add_argument(
        "--save-field",
        action="store_true",
        help="also write the simulated Gaussian lattice (truth_gauss_field.npy, tens of MB)",
    )
    parser.set_defaults(func=run)


def run(args: argparse.Namespace) -> int:
    from bakumine.synth.config import PRESETS
    from bakumine.synth.generate import generate

    deposit = generate(PRESETS[args.preset](), seed=args.seed)
    written = deposit.write(args.out, save_field=args.save_field)
    totals = deposit.truth["totals"]
    print(f"wrote {len(written)} files to {args.out}")
    print(
        f"holes {len(deposit.holes)}, ore volume {totals['ore_volume_m3']:.0f} m3, "
        f"tonnes {totals['ore_tonnes']:.0f}, mean grade {totals['ore_mean_grade_g_t']:.3f} g/t, "
        f"metal {totals['ore_metal_troy_oz']:.0f} oz"
    )
    return 0
