"""``bakumine intercepts``: mineralised intervals and significant intercepts from assays.

Runs :mod:`bakumine.core.drillholes.mineralised` on an interval table of a database
folder. The intercept table is written as CSV; with ``--save-as NAME`` the coded interval
table (MIN / WASTE) is stored in the database as interval table NAME, ready to be the
domain table of ``bakumine composite --domain-table NAME --domain-column code``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.drillholes import CommandError, _guarded, _wrote

HELP = "Flag mineralised intervals in an assay table and list the significant intercepts"

#: argparse destination -> InterceptParams field, for the options that override --params
_OPTIONS = (
    "cutoff",
    "min_thickness",
    "max_internal_waste",
    "max_total_waste",
    "gap_tolerance",
    "top_cut",
    "including_min_thickness",
    "body_dip",
    "body_dip_azimuth",
    "ore_code",
    "waste_code",
    "code_column",
    "mode",
    "window",
)


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("database", help="drillhole database folder")
    parser.add_argument("--table", default="assay", help="assay interval table (default: assay)")
    parser.add_argument("--grade", required=True, help="grade column")
    parser.add_argument("--params", metavar="JSON", help="parameter file (options override it)")
    parser.add_argument("--cutoff", type=float, help="cut-off grade (ore is grade >= cut-off)")
    parser.add_argument(
        "--mode",
        choices=("strict", "average", "moving-average"),
        help="how runs are built (default: strict). strict: every part of a run measured "
        "from either end meets the cut-off. average: the run starts and ends on ore and "
        "its grade including the internal waste meets the cut-off. moving-average: "
        "samples whose mean grade over --window metres meets the cut-off "
        "(--max-internal-waste is not used)",
    )
    parser.add_argument(
        "--window", type=float, metavar="M", help="window of --mode moving-average, metres"
    )
    parser.add_argument(
        "--min-thickness",
        type=float,
        help="minimum thickness (default: 0). Down-hole length, unless --body-dip and "
        "--body-dip-azimuth are given: then it is the TRUE thickness across the body, so a "
        "hole that runs along the body needs a longer run to qualify (the waste limits "
        "stay down-hole lengths)",
    )
    parser.add_argument(
        "--max-internal-waste", type=float, help="longest internal waste stretch (default: 0)"
    )
    parser.add_argument("--max-total-waste", type=float, help="largest total internal waste")
    parser.add_argument(
        "--accumulation-rule",
        action="store_true",
        default=None,
        help="accept thinner runs with grade x thickness >= cut-off x minimum thickness",
    )
    parser.add_argument(
        "--gap-tolerance", type=float, help="longest missing stretch that is bridged (default: 0)"
    )
    parser.add_argument("--top-cut", type=float, help="top cut for the grade_cut column")
    parser.add_argument(
        "--including", metavar="CUTOFFS", help="higher cut-offs for sub-intercepts, e.g. 1,5"
    )
    parser.add_argument(
        "--including-min-thickness", type=float, help="minimum thickness of sub-intercepts"
    )
    parser.add_argument(
        "--body-dip",
        type=float,
        help="dip of the body: with --body-dip-azimuth, the minimum thicknesses are "
        "measured as true thickness, and true_thickness is written per intercept",
    )
    parser.add_argument("--body-dip-azimuth", type=float, help="azimuth of its dip direction")
    parser.add_argument("--ore-code", help="code of mineralised intervals (default: MIN)")
    parser.add_argument("--waste-code", help="code of the other intervals (default: WASTE)")
    parser.add_argument("--code-column", help="name of the code column (default: code)")
    parser.add_argument(
        "--save-as", metavar="NAME", help="store the coded intervals in the database as NAME"
    )
    parser.add_argument("--overwrite", action="store_true", help="replace an existing table NAME")
    parser.add_argument(
        "--out", help="intercept CSV (default: <database>/intercepts_<NAME or table>.csv)"
    )
    parser.add_argument("--intervals-out", metavar="CSV", help="also write the coded intervals")
    parser.set_defaults(func=_guarded(run))


def params_from_args(args: argparse.Namespace) -> Any:
    from bakumine.core.drillholes.mineralised import InterceptParams

    data: dict[str, Any] = {}
    if args.params:
        file = Path(args.params)
        if not file.is_file():
            raise CommandError(f"{file} not found")
        data = json.loads(file.read_text(encoding="utf-8"))
    for name in _OPTIONS:
        if getattr(args, name) is not None:
            data[name] = getattr(args, name)
    if args.accumulation_rule:
        data["accumulation_rule"] = True
    if args.including:
        try:
            data["including_cutoffs"] = [float(v) for v in args.including.split(",") if v.strip()]
        except ValueError:
            raise CommandError(f"--including expects numbers, got {args.including!r}") from None
    if "cutoff" not in data:
        raise CommandError("give --cutoff (or a --params file with a cutoff)")
    return InterceptParams.from_dict(data)


def format_summary(summary: dict[str, Any]) -> str:
    grade = "-" if summary["mean_grade"] is None else f"{summary['mean_grade']:.4g}"
    return (
        f"{summary['grade']} >= {summary['cutoff']:g} ({summary.get('mode', 'strict')}): "
        f"{summary['n_intercepts']} intercepts "
        f"(+{summary['n_including']} including) in {summary['n_holes_mineralised']} of "
        f"{summary['n_holes']} holes, {summary['mineralised_length']:.2f} m at {grade}, "
        f"{summary['grade_x_m']:.4g} grade x m; other coded length {summary['waste_length']:.2f} m"
    )


def run(args: argparse.Namespace) -> int:
    from bakumine.core.drillholes.mineralised import flag_database_folder

    folder = Path(args.database)
    if not (folder / "database.json").is_file():
        raise CommandError(f"{folder} is not a drillhole database folder (no database.json)")
    params = params_from_args(args)
    result, path = flag_database_folder(
        folder,
        args.table,
        args.grade,
        params,
        save_as=args.save_as,
        overwrite=args.overwrite,
        out=args.out,
    )
    _wrote(path, f"{len(result.intercepts)} intercept rows")
    if args.save_as:
        _wrote(folder / f"interval_{args.save_as}.parquet", f"{len(result.intervals)} intervals")
    if args.intervals_out:
        out = Path(args.intervals_out)
        out.parent.mkdir(parents=True, exist_ok=True)
        result.intervals.to_csv(out, index=False)
        _wrote(out, f"{len(result.intervals)} intervals")
    print(format_summary(result.summary()))
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "intercepts": (HELP, configure),
}


def replay(entry: dict[str, Any]) -> list[dict[str, Any]]:
    """CLI steps of a logged ``intercepts`` job (``bakumine project replay``)."""
    from bakumine.commands.replay import ReplayError, _in_project, _run, _work, _write

    p = entry["params"]
    params = (entry.get("replay") or {}).get("params")
    if params is None:
        raise ReplayError(f"job {entry['id']}: the intercept parameters were not logged")
    file = _work(entry["id"], "intercepts.json")
    argv = ["intercepts", _in_project(entry["inputs"][0])]
    argv += ["--table", p.get("table") or "assay", "--grade", p["grade"], "--params", file]
    if p.get("save_as"):
        argv += ["--save-as", p["save_as"]]
        if p.get("overwrite"):
            argv.append("--overwrite")
    return [_write(file, json.dumps(params, indent=2)), _run(*argv)]
