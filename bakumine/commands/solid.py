"""Solid subcommands: ``bakumine solid implicit``.

    bakumine solid implicit DB --column rock --code ORE --dip 70 --dip-azimuth 90 \\
        --ratios 4:3:1 --max-extrapolation 60 --topography topo.obj --project P

Builds a closed solid of the intervals logged with the given code(s) directly from the
drillholes (:func:`bakumine.core.geometry.implicit.implicit_solid`). The anisotropy must be
stated: a plane (dip, dip azimuth, optional pitch) or rotation angles with the ratios of
the body's extent, or ``--anisotropy auto | none``. Outputs as for ``bakumine surface``:
``--out``, ``--geometry-dir``, ``--project``.

The same function runs behind the dialog (:mod:`bakumine.api.operations_implicit`);
:data:`REPLAY` turns its job-log entries back into this command.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded
from bakumine.commands.geometry import (
    _add_outputs,
    _database,
    _emit,
    _finish,
    _floats,
    _need_output,
)

HELP_SOLID = "Solids: implicit solid of logged codes straight from the drillholes"


def configure_implicit(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("database", help="drillhole database folder")
    parser.add_argument("--table", default="geology", help="interval table (default: geology)")
    parser.add_argument("--column", required=True, help="column with the logged codes")
    parser.add_argument("--code", action="append", required=True,
                        help="code of the domain (repeat for several codes)")  # fmt: skip
    ani = parser.add_argument_group(
        "anisotropy (required: a plane, angles, or --anisotropy auto|none)",
        "A plane: --dip --dip-azimuth [--pitch] with --ratios. The major axis lies along "
        "strike (dip azimuth - 90), turned by the pitch towards the dip direction; the "
        "minor axis is the pole of the plane. Angles: --angles bearing,plunge,dip in the "
        "project's rotation convention with --ratios. 'auto' estimates the plane: for one "
        "body from one point per intercept; for stacked lenses (holes that cut the domain "
        "several times) as the plane along which the ore of neighbouring holes lines up "
        "best, with a confidence. When that is not well determined no plane is chosen: "
        "the solid is isotropic and a warning asks for the orientation.",
    )
    ani.add_argument("--dip", type=float, help="dip of the body, degrees from horizontal")
    ani.add_argument("--dip-azimuth", type=float, help="direction the body dips towards")
    ani.add_argument("--pitch", type=float, default=0.0,
                     help="rake of the major axis in the plane, from strike")  # fmt: skip
    ani.add_argument("--angles", metavar="BEARING,PLUNGE,DIP")
    ani.add_argument("--ratios", metavar="A:B:C",
                     help="extent along major : semi-major : minor axis (along strike : "
                          "down dip : across), for example 4:2:1")  # fmt: skip
    ani.add_argument("--anisotropy", choices=("auto", "none"))
    parser.add_argument("--max-extrapolation", type=float, metavar="M",
                        help="keep the solid within M metres of the nearest hole")  # fmt: skip
    parser.add_argument("--topography", metavar="FILE",
                        help="DTM file (or name in --geometry-dir): nothing above it")  # fmt: skip
    parser.add_argument("--resolution", type=float, metavar="M",
                        help="cell across the body (default: median intercept / 8)")  # fmt: skip
    parser.add_argument("--min-component-volume", type=float, default=0.0, metavar="M3",
                        help="remove shells smaller than this (default: keep all)")  # fmt: skip
    _add_outputs(parser, "solid")
    parser.set_defaults(func=_guarded(run_implicit_command))


def anisotropy_spec(args: argparse.Namespace) -> dict[str, Any]:
    plane = args.dip is not None or args.dip_azimuth is not None
    given = sum([plane, args.angles is not None, args.anisotropy is not None])
    if given == 0:
        raise CommandError(
            "give the orientation of the body: --dip and --dip-azimuth (or --angles) with "
            "--ratios, or --anisotropy auto | none"
        )
    if given > 1:
        raise CommandError("give one of: a plane (--dip, --dip-azimuth), --angles, --anisotropy")
    if args.anisotropy:
        return {"mode": args.anisotropy}
    if plane:
        if args.dip is None or args.dip_azimuth is None:
            raise CommandError("a plane needs both --dip and --dip-azimuth")
        return {"mode": "plane", "dip": args.dip, "dip_azimuth": args.dip_azimuth,
                "pitch": args.pitch, "ratios": args.ratios}  # fmt: skip
    return {"mode": "angles", "angles": _floats(args.angles, 3, "--angles"), "ratios": args.ratios}


def _topography(args: argparse.Namespace):
    from bakumine.core.geometry.dtm import DTM

    if not args.topography:
        return None
    path = Path(args.topography)
    if path.is_file():
        from bakumine.core.io.files import read_mesh

        return DTM(read_mesh(path))
    if args.geometry_dir:
        from bakumine.core.blockmodel.constraints import FileResolver

        return FileResolver(args.geometry_dir).dtm(args.topography)
    raise CommandError(f"{path} not found")


def format_axes(anisotropy: dict[str, Any]) -> str:
    confidence = ""
    if anisotropy.get("confidence") is not None:
        confidence = (
            f"; confidence {anisotropy['confidence']:.2f} (indicator correlation between "
            f"holes {anisotropy.get('correlation', float('nan')):.2f})"
        )
    if "major" not in anisotropy:
        candidate = anisotropy.get("candidate") or {}
        if candidate.get("dip") is not None:
            confidence += (
                f"; best plane, NOT used: dips {candidate['dip']:.0f} towards "
                f"{candidate['dip_azimuth']:03.0f}"
            )
        return f"anisotropy: {anisotropy['source']} (isotropic){confidence}"
    parts = [
        f"{label} {anisotropy[key]['azimuth']:05.1f}/{anisotropy[key]['plunge']:.1f}"
        for label, key in (("major", "major"), ("semi-major", "semi_major"), ("minor", "minor"))
    ]
    extent = ":".join(f"{r:.2f}" for r in anisotropy["ratios"])
    plane = anisotropy["plane"]
    return (
        f"anisotropy ({anisotropy['source']}): plane dips {plane['dip']:.1f} towards "
        f"{plane['dip_azimuth']:05.1f}; axes azimuth/plunge {', '.join(parts)}; "
        f"extent major:semi-major:minor {extent}{confidence}"
    )


def run_implicit_command(args: argparse.Namespace) -> int:
    from bakumine.core.geometry.implicit_request import run_implicit

    _need_output(args)
    request = {
        "table": args.table,
        "column": args.column,
        "codes": list(args.code),
        "anisotropy": anisotropy_spec(args),
        "max_extrapolation": args.max_extrapolation,
        "resolution": args.resolution,
        "min_component_volume": args.min_component_volume,
    }
    result, summary = run_implicit(_database(args.database), request, _topography(args))
    for message in summary["warnings"]:
        print(f"warning: {message}")
    print(format_axes(summary["anisotropy"]))
    cavities = ""
    if summary["n_cavities"]:
        cavities = f" of which {summary['n_cavities']} cavities (holes in the solid, wound inward)"
    print(
        f"solid: volume {summary['volume']:.1f} m3, "
        f"{'valid' if summary['valid'] else 'NOT VALID'}, {summary['n_components']} "
        f"component(s){cavities}, {summary['n_faces']} triangles, "
        f"resolution {summary['resolution']:.2f} m, "
        f"{summary['n_holes_intersecting']} of {summary['n_holes']} holes intersect"
    )
    for issue in summary["issues"]:
        print(f"  issue: {issue}")
    if summary["touches_grid_boundary"]:
        print("warning: the solid reaches the edge of the grid and is cut flat there")
    name = args.name or "-".join(str(c) for c in args.code)
    _emit(result.mesh, "solid", name, args, summary)
    return 0


def configure_solid(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="solid_command", metavar="<action>")
    configure_implicit(
        sub.add_parser(
            "implicit",
            help="solid of logged codes from the drillholes (RBF implicit model)",
            description="Build a closed solid of the intervals logged with the given "
            "code(s), directly from the drillholes.",
        )
    )
    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "solid": (HELP_SOLID, configure_solid),
}


# ------------------------------------------------------------------ job-log replay


def _replay_implicit(e: dict[str, Any]) -> list[dict[str, Any]]:
    from bakumine.commands.replay import _in_project

    p = e["params"]
    database = next(o for o in e["inputs"] if o["kind"] == "database")
    argv = ["solid", "implicit", _in_project(database), "--table", str(p.get("table") or "geology")]
    argv += ["--column", str(p["column"])]
    codes = p.get("codes", p.get("code"))
    for code in codes if isinstance(codes, list) else [codes]:
        argv.append(f"--code={code}")
    a = p.get("anisotropy")
    a = {"mode": a} if isinstance(a, str) else dict(a or {})
    ratios = a.get("ratios")
    if isinstance(ratios, list):
        ratios = ":".join(repr(float(r)) for r in ratios)
    if a.get("mode") == "plane":
        argv += [f"--dip={float(a['dip'])!r}", f"--dip-azimuth={float(a['dip_azimuth'])!r}"]
        argv += [f"--pitch={float(a.get('pitch') or 0.0)!r}", f"--ratios={ratios}"]
    elif a.get("mode") == "angles":
        argv.append("--angles=" + ",".join(repr(float(v)) for v in a["angles"]))
        argv.append(f"--ratios={ratios}")
    else:
        argv += ["--anisotropy", str(a.get("mode"))]
    for key, flag in (
        ("max_extrapolation", "--max-extrapolation"),
        ("resolution", "--resolution"),
        ("min_component_volume", "--min-component-volume"),
    ):
        if p.get(key) not in (None, ""):
            argv.append(f"{flag}={float(p[key])!r}")
    topo = next((o for o in e["inputs"] if o["kind"] == "dtm"), None)
    if topo is not None:
        argv += ["--topography", _in_project(topo)]
    return _finish(e, argv)


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "implicit_solid": _replay_implicit,
}
