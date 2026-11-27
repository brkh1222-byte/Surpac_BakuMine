"""``bakumine geology build``: the geology model of a build spec, as a folder.

    bakumine geology build [DB] --spec SPEC --out FOLDER [--overwrite] [--summary JSON]
        [--base DIR]

Runs :func:`bakumine.core.geology.spec.build_geology` and writes the model folder of
geology_contracts.md section 9 (``geology.json``, ``units.json``, ``spec.json``,
``provenance.json``, ``diagnostics.json``, grid, fields, surfaces, faults, meshes). ``DB``
replaces the database the spec names. Paths in the spec are relative to the spec file, or
to ``--base``. Prints the figures of the build and the three provenance tables (measured,
assumed, missing).

The same function runs behind the dialog (:mod:`bakumine.api.operations_geology`);
:data:`REPLAY` turns the job-log entries ``geology_build`` and ``geology_code`` back into
``bakumine geology build`` and ``bakumine geology code``.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Build the geology model of a build spec: surfaces, faults, rock units, provenance"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        HELP + ". Every default the build applies, every input it uses and every input "
        "that is absent is listed in provenance.json and printed as three tables."
    )
    parser.add_argument("database", nargs="?", metavar="DB",
                        help="drillhole database folder (default: the spec's)")  # fmt: skip
    parser.add_argument("--spec", required=True, metavar="JSON", help="geology build spec")
    parser.add_argument("--out", required=True, metavar="FOLDER", help="geology folder to write")
    parser.add_argument("--overwrite", action="store_true",
                        help="replace an existing geology folder")  # fmt: skip
    parser.add_argument("--summary", metavar="JSON", help="also write the figures of the build")
    parser.add_argument("--base", metavar="DIR",
                        help="folder the paths of the spec are relative to (default: the "
                             "folder of the spec file)")  # fmt: skip
    parser.set_defaults(func=_guarded(run))


def _number(value: Any, digits: int = 1) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def print_summary(summary: dict[str, Any]) -> None:
    """The figures of :func:`bakumine.core.geology.spec.build_summary` as text."""
    grid = summary["grid"]
    print(f"geology model {summary['name']!r}")
    print(
        f"  grid {grid['shape'][0]} x {grid['shape'][1]} x {grid['shape'][2]} nodes at "
        f"{grid['spacing'][0]:g} m"
        + (f" (coarsened from {grid['resolution_requested']:g} m)" if grid["coarsened"] else "")
    )
    inputs = summary["inputs"]
    print(
        f"  {inputs['holes']} holes ({inputs['holes_with_trace']} with a trace), "
        f"{inputs['lithology_intervals']} logged intervals, {inputs['rock_intervals']} of "
        "modelled rock units"
    )
    for layer, categories in summary["layers"].items():
        print(f"  {layer}: {', '.join(categories)}")
    for fault in summary["faults"]:
        blocks = " / ".join(fault["blocks"]) if fault["blocks"] else "blocks from the logs"
        print(
            f"  fault {fault['name']} ({fault['source']}): dip {fault['dip']:.0f}, dip azimuth "
            f"{fault['dip_azimuth']:03.0f}; {blocks}"
        )
    if summary["pooled_blocks"]:
        print(f"  fault blocks on the pooled rock model: {', '.join(summary['pooled_blocks'])}")
    for throw in summary["throw"]:
        print(
            f"  throw of {throw['fault']} on {throw['upper']} / {throw['lower']}: vertical "
            f"separation {_number(throw['throw'])} m ({throw['kind']}); {throw['note']}"
        )
    for surface in summary["surfaces"]:
        used = surface["used"] or {}
        print(
            f"  surface {surface['name']}: depth {_number(surface['depth_min'])} to "
            f"{_number(surface['depth_max'])} m (mean {_number(surface['depth_mean'])}); "
            f"{used.get('exact', 0)} exact picks, "
            f"{sum(used.get(k, 0) for k in ('lower', 'upper', 'between'))} bounds"
        )
    residuals = summary["residuals"]
    if residuals and residuals["points"]:
        print(
            f"  logged rock unit reproduced at {100.0 * residuals['reproduced']:.2f} % of "
            f"{residuals['points']} points along the holes (grid labels)"
        )
    volumes = summary["volumes"]
    print("\nVolume per unit (m3): whole box, inside the drilled hull, extrapolated")
    print(f"  {'unit':<12} {'box':>14} {'inside hull':>14} {'extrapolated':>14} {'share':>7}")
    for code, row in volumes["units"].items():
        share = row["extrapolated_share"]
        print(
            f"  {code:<12} {row['volume']:>14.0f} {_number(row['inside_hull'], 0):>14} "
            f"{_number(row['extrapolated'], 0):>14} "
            f"{'n/a' if share is None else f'{100.0 * share:.1f} %':>7}"
        )
    if not volumes["hull"]:
        print("  (no drilled hull: fewer than 4 hole points, or all in one plane)")
    timing = summary["timing"]
    print(f"\nbuilt in {timing['total_seconds']:.1f} s")


def run(args: argparse.Namespace) -> int:
    from bakumine.commands.geometry import _database
    from bakumine.core.geology.spec import build_geology, build_summary, load_spec, save_geology

    out = Path(args.out)
    if out.exists() and any(out.iterdir()) and not args.overwrite:
        raise CommandError(f"{out} exists; pass --overwrite to replace it")
    spec = load_spec(Path(args.spec), base_dir=Path(args.base) if args.base else None)
    database = _database(args.database) if args.database else None
    if database is None and spec.database is None:
        raise CommandError("give the database folder (DB) or name it in the spec ('database')")
    model = build_geology(spec, database=database)
    summary = build_summary(model)
    print_summary(summary)
    print("\n" + model.provenance.format_text() + "\n")
    folder = save_geology(model, out, overwrite=args.overwrite)
    kinds = summary["provenance"]
    _wrote(
        folder,
        f"geology model, {len(model.categories('lith')) - 1} units, {len(model.blocks)} fault "
        f"block(s); provenance: {kinds['measured']} measured, {kinds['assumed']} assumed, "
        f"{kinds['missing']} missing",
    )
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        _wrote(path, "figures of the build")
    return 0


# ------------------------------------------------------------------ job-log replay


def _replay_build(e: dict[str, Any]) -> list[dict[str, Any]]:
    """``geology_build``: the spec the job ran, with the project objects as paths relative
    to the new project (``--base``), then ``project add``."""
    from bakumine.commands.replay import PROJECT, ReplayError, _add, _run, _work, _write

    spec = (e.get("replay") or {}).get("spec")
    if spec is None:
        raise ReplayError(f"job {e['id']}: the build spec was not logged")
    out = e["outputs"][0]
    spec_file = _work(e["id"], "spec.json")
    folder = _work(e["id"], Path(out["path"]).name)
    return [
        _write(spec_file, json.dumps(spec, indent=2)),
        _run("geology", "build", "--spec", spec_file, "--base", PROJECT, "--out", folder),
        _add(out, folder),
    ]


def _replay_code(e: dict[str, Any]) -> list[dict[str, Any]]:
    """``geology_code``: ``bakumine geology code`` on the model and the geology folder."""
    from bakumine.commands.replay import _in_project, _run

    p = e["params"]
    model = next(o for o in e["inputs"] if o["kind"] == "model")
    geology = next(o for o in e["inputs"] if o["kind"] == "geology")
    argv = ["geology", "code", _in_project(model), "--geology", _in_project(geology)]
    if p.get("layers"):
        argv += ["--layers", ",".join(str(layer) for layer in p["layers"])]
    if p.get("subblock") is False:
        argv.append("--no-subblock")
    if p.get("max_level") is not None:
        argv += ["--max-level", str(int(p["max_level"]))]
    if p.get("density") is None and "density" in p:
        argv.append("--no-density")
    elif p.get("density"):
        argv += ["--density", str(p["density"])]
    for key, flag in (("distance", "--distance"), ("display", "--display"),
                      ("ore_attribute", "--ore-attribute")):  # fmt: skip
        if p.get(key):
            argv += [flag, str(p[key])]
    if p.get("merge"):
        argv.append("--merge")
    return [_run(*argv)]


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "geology_build": _replay_build,
    "geology_code": _replay_code,
}
