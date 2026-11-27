"""Blast subcommands: ``bakumine blast demo | simulate | report | defaults``.

    bakumine blast defaults [--project P] > request.json
    bakumine blast demo --out result.json [--design design.json]
    bakumine blast simulate --params request.json [--project P [--topo ID] [--model ID]]
                            --out result.json [--design design.json]
    bakumine blast report result.json [--out report.txt]

``demo`` and ``simulate`` without a project run on a synthetic bench. With a project the
ground is the project's topography (the first ``dtm`` object, or ``--topo``), a free face
is cut in front of the block, and the damage is computed for the blocks of ``--model`` when
given. The result is the bundle of plan_blast.md section 3; the project is only read.

The same function runs behind ``POST /api/v1/blast/simulate``
(:func:`bakumine.blast.simulate.simulate`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Blast BakuMine: simulate a bench blast (design, fragmentation, vibration, throw)"


def _load_json(path: str, what: str) -> Any:
    file = Path(path)
    if not file.is_file():
        raise CommandError(f"{what} {file} does not exist")
    try:
        return json.loads(file.read_text(encoding="utf-8"))
    except ValueError as exc:
        raise CommandError(f"{file} is not valid JSON: {exc}") from None


def _write(bundle: dict[str, Any], args: argparse.Namespace) -> None:
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(bundle, allow_nan=False), encoding="utf-8")
    s = bundle["summary"]
    _wrote(out, f"blast result, {s['holes']} holes, {s['explosive_kg']:.0f} kg")
    if args.design:
        design = Path(args.design)
        design.parent.mkdir(parents=True, exist_ok=True)
        design.write_text(json.dumps(bundle["design"], indent=2, allow_nan=False), encoding="utf-8")
        _wrote(design, "blast design")


def _demo(args: argparse.Namespace) -> int:
    from bakumine.blast.simulate import defaults, simulate

    _write(simulate(defaults()), args)
    return 0


def _simulate(args: argparse.Namespace) -> int:
    from bakumine.blast.bench import project_site
    from bakumine.blast.simulate import simulate

    request = _load_json(args.params, "request file")
    if not args.project:
        if args.topo or args.model:
            raise CommandError("--topo and --model need --project")
        _write(simulate(request), args)
        return 0
    ground, blocks, ids = project_site(args.project, args.topo, args.model)
    print(f"topography {ids['topo']!r}" + (f", block model {ids['model']!r}" if blocks else ""))
    _write(simulate(request, ground=ground, blocks=blocks), args)
    return 0


def _defaults(args: argparse.Namespace) -> int:
    from bakumine.blast.simulate import defaults

    top = None
    if args.project:
        from bakumine.blast.bench import project_site

        top, _, _ = project_site(args.project, args.topo)
    print(json.dumps(defaults(top), indent=2, allow_nan=False))
    return 0


# ------------------------------------------------------------------------------ report


def _num(value: Any, digits: int = 2, unit: str = "") -> str:
    if value is None:
        return "n/a"
    if isinstance(value, bool):
        return "yes" if value else "no"
    text = f"{value:,.{digits}f}" if isinstance(value, float) else f"{value:,}"
    return f"{text} {unit}".rstrip()


def _verdict(ok: Any) -> str:
    return "no limit given" if ok is None else ("within limit" if ok else "ABOVE LIMIT")


def report_text(bundle: dict[str, Any]) -> str:
    """Plain-text report of a result bundle for the feasibility study."""
    from bakumine.blast.simulate import validate_bundle

    problems = validate_bundle(bundle)
    if problems:
        raise CommandError("not a valid blast result: " + "; ".join(problems[:3]))
    d, s = bundle["design"], bundle["summary"]
    lines: list[str] = []

    def title(text: str) -> None:
        lines.extend(["", text, "-" * len(text)])

    def row(label: str, value: str) -> None:
        lines.append(f"  {label:<34} {value}")

    lines.append(f"BLAST REPORT: {d['name']}")
    lines.append("=" * len(lines[0]))
    lines.append("Predictions of empirical engineering models; see the caveats at the end.")

    title("Design")
    row("Bench height", _num(d.get("bench_height"), 2, "m"))
    row("Floor elevation", _num(d.get("floor_z"), 2, "m"))
    row("Face azimuth (throw direction)", _num(d.get("face_azimuth"), 1, "deg"))
    row("Pattern", str(d.get("pattern")))
    row("Hole diameter", _num(1000.0 * s["diameter_m"], 0, "mm") if "diameter_m" in s else "n/a")
    row("Burden x spacing", f"{_num(d.get('burden'))} x {_num(d.get('spacing'))} m")
    row("Stemming", _num(s.get("stemming_m"), 2, "m"))
    row("Subdrill", _num(s.get("subdrill_m"), 2, "m"))
    row("Stiffness ratio H/B", _num(s.get("stiffness_ratio")))
    row("Holes / rows", f"{s['holes']} / {_num(s.get('rows'))}")
    row("Explosive", str(s.get("explosive", "n/a")))
    timing = d.get("notes", {}).get("timing", {})
    if timing:
        row(
            "Timing",
            f"{timing.get('scheme')}, {_num(timing.get('inter_hole_ms'), 0)} ms between holes, "
            f"{_num(timing.get('inter_row_ms'), 0)} ms between rows",
        )
    row("Duration of the initiation", _num(s["duration_ms"], 0, "ms"))
    row("Burden relief", _num(s.get("burden_relief_ms_m"), 1, "ms/m"))

    title("Consumption")
    row("In-situ volume", _num(s["volume_m3"], 0, "m3"))
    row("Tonnes", _num(s["tonnes"], 0, "t"))
    row("Drilled metres", _num(s["drilled_m"], 1, "m"))
    row("Specific drilling", _num(s.get("specific_drilling_m_m3"), 4, "m/m3"))
    row("Explosive", _num(s["explosive_kg"], 0, "kg"))
    row("Powder factor", _num(s["powder_factor_kg_m3"], 3, "kg/m3"))
    row("Powder factor", _num(s["powder_factor_kg_t"], 3, "kg/t"))
    row("Max. instantaneous charge (8 ms)", _num(s["mic_kg"], 1, "kg"))

    title("Fragmentation (Kuz-Ram)")
    row("x50 (median size)", _num(s["x50_m"], 3, "m"))
    row("P80", _num(s["p80_m"], 3, "m"))
    row("Uniformity index n", _num(s["uniformity_n"]))
    oversize = s["models"].get("fragmentation", {}).get("oversize_m")
    row(f"Oversize above {_num(oversize)} m", _num(100.0 * s["oversize_fraction"], 2, "%"))
    row("Rock factor A", _num(s["models"].get("fragmentation", {}).get("rock_factor")))

    title("Ground vibration at the monitors (scaled distance)")
    w = bundle["waves"]
    row("Site law", f"PPV = {_num(w['k'], 0)} (R / sqrt(W))^-{_num(w['beta'])} mm/s")
    if not bundle["monitors"]:
        lines.append("  no monitor points given")
    for m in bundle["monitors"]:
        limit = "" if m["limit_mm_s"] is None else f" (limit {_num(m['limit_mm_s'], 1)} mm/s)"
        row(
            m["name"],
            f"{_num(m['ppv_mm_s'], 2, 'mm/s')} at {_num(m['distance_m'], 0, 'm')}{limit}: "
            f"{_verdict(m['ok'])}",
        )

    title("Air overpressure (cube-root scaling)")
    for m in bundle["monitors"]:
        limit = m["airblast_limit_dbl"]
        text = "" if limit is None else f" (limit {_num(limit, 0)} dBL)"
        row(m["name"], f"{_num(m['airblast_dbl'], 1, 'dBL')}{text}: {_verdict(m['airblast_ok'])}")
    for ring in bundle["airblast"]["rings"]:
        row(
            f"at {_num(ring['radius_m'], 0)} m",
            f"{_num(ring['dbl'], 1, 'dBL')} ({_num(ring['pa'], 0, 'Pa')})",
        )

    f = bundle["flyrock"]
    title("Flyrock (Richards and Moore)")
    row("Face burst", _num(f["face_burst_m"], 0, "m"))
    row("Cratering", _num(f["cratering_m"], 0, "m"))
    row("Stemming ejection", _num(f["stemming_ejection_m"], 0, "m"))
    row("Maximum throw", _num(f["max_range_m"], 0, "m"))
    row(f"Clearance for people (x {_num(f['safety_factor'], 0)})", _num(f["clearance_m"], 0, "m"))

    title("Damage and muckpile")
    row("Back-break behind the last row", _num(s["backbreak_m"], 1, "m"))
    limits = ", ".join(_num(v, 0) for v in bundle["damage"]["thresholds_mm_s"])
    row("Damage thresholds (PPV)", f"{limits} mm/s")
    row("Throw of the centre of mass", _num(s.get("throw_com_m"), 1, "m"))
    row("Largest throw of the muck", _num(s.get("throw_max_m"), 1, "m"))
    row("Muckpile volume (swelled)", _num(s.get("muckpile_volume_m3"), 0, "m3"))
    row("Muckpile maximum thickness", _num(s.get("muckpile_max_height_m"), 1, "m"))

    title("Models and site constants")
    for name, info in s["models"].items():
        state = "calibrated" if info["calibrated"] else "uncalibrated"
        constants = ", ".join(
            f"{k} = {_num(v)}"
            for k, v in info.items()
            if isinstance(v, int | float) and not isinstance(v, bool)
        )
        lines.append(f"  {name:<14} {state:<13} {constants}".rstrip())

    title("Caveats")
    lines.extend(f"  - {note}" for note in s.get("notes", []))
    return "\n".join(lines) + "\n"


def _report(args: argparse.Namespace) -> int:
    text = report_text(_load_json(args.result, "result file"))
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        _wrote(out, "blast report")
    else:
        print(text, end="")
    return 0


# ------------------------------------------------------------------------------ parser


def _add_outputs(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", required=True, metavar="FILE", help="result bundle (JSON)")
    parser.add_argument("--design", metavar="FILE", help="also write the blast design (JSON)")


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="blast_command", metavar="<action>")

    demo = sub.add_parser("demo", help="simulate the default blast on a synthetic bench")
    _add_outputs(demo)
    demo.set_defaults(func=_guarded(_demo))

    sim = sub.add_parser("simulate", help="simulate a request, on a project's topography or not")
    sim.add_argument("--params", required=True, metavar="FILE", help="request JSON")
    sim.add_argument("--project", metavar="DIR", help="project folder (topography, block model)")
    sim.add_argument("--topo", metavar="ID", help="dtm object (default: the project's first)")
    sim.add_argument("--model", metavar="ID", help="block model whose blocks show the damage")
    _add_outputs(sim)
    sim.set_defaults(func=_guarded(_simulate))

    report = sub.add_parser("report", help="plain-text report of a result bundle")
    report.add_argument("result", metavar="RESULT", help="result bundle (JSON)")
    report.add_argument("--out", metavar="FILE", help="write the report here instead of printing")
    report.set_defaults(func=_guarded(_report))

    dflt = sub.add_parser("defaults", help="print the default request JSON")
    dflt.add_argument("--project", metavar="DIR", help="put the polygon on this project")
    dflt.add_argument("--topo", metavar="ID", help="dtm object (default: the project's first)")
    dflt.set_defaults(func=_guarded(_defaults))

    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS = {"blast": (HELP, configure)}
