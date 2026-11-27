"""``bakumine geology surface``: a cover or weathering base as depth below the topography.

    bakumine geology surface DB --units units.json --unit OX --threshold "assay:s_ppm:<1000"
        --surface base:OXIDE --topography topo.obj --rectangle X0,Y0,X1,Y1 --spacing 5
        --above base_Q4.obj --out base_OXIDE.obj --picks-out base_OXIDE_picks.csv
        --summary base_OXIDE.json

Picks come from the logged units (``--unit``, mapped with the dictionary), an extra intervals
file (``--extra-intervals``) and a rule on an assay column (``--threshold``), in that order of
priority. The depth below the topography is interpolated through the exact picks under the
bounds, the surface is the topography minus that depth, and ``--above`` keeps it from rising
above a shallower surface that is already on file (research.md 5.7.4).

The command prints the picks per source, what was left out (holes without information,
conflicts between sources, picks outside the topography) and the three provenance tables
(measured, assumed, missing).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote
from bakumine.commands.geometry import _NAME, _database, _floats

HELP = "Cover / weathering base surface: depth below the topography from picks and bounds"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("database", help="drillhole database folder")
    src = parser.add_argument_group("picks (give --unit, --threshold or both)")
    src.add_argument("--units", metavar="JSON", help="unit dictionary (needed with --unit)")
    src.add_argument("--unit", action="append", default=[], metavar="CODE",
                     help="unit code of the zone above the surface; repeatable")  # fmt: skip
    src.add_argument("--table", help="logged interval table (default: the dictionary's "
                                     "source, else geology)")  # fmt: skip
    src.add_argument("--column", help="its code column (default: the dictionary's source)")
    src.add_argument("--threshold", metavar="TABLE:COLUMN:<VALUE|>VALUE",
                     help="rule on a numeric column: the zone is where the value is below "
                          "(<) or above (>) VALUE, e.g. assay:s_ppm:<1000")  # fmt: skip
    src.add_argument("--extra-intervals", metavar="CSV",
                     help="extra logged intervals: hole_id, depth_from, depth_to, code "
                          "(codes of the dictionary; needs --unit)")  # fmt: skip
    src.add_argument("--topography", required=True, metavar="FILE",
                     help="ground surface (.obj, .dxf, .ply, .stl, .dtm)")  # fmt: skip
    src.add_argument("--surface", metavar="NAME",
                     help="surface name (default: <side>:<first --unit>: base:OX)")  # fmt: skip
    src.add_argument("--side", choices=("base", "top"), default="base")
    src.add_argument("--mode", choices=("deepest", "first_run"), default="deepest")
    src.add_argument("--absent", choices=("unknown", "zero"), default="unknown",
                     help="a hole with other units and never the zone: no information "
                          "(default) or the zone is absent there")  # fmt: skip
    src.add_argument("--keep-conflicts", action="store_true",
                     help="fit the holes where the sources contradict each other as well "
                          "(default: left out and listed)")  # fmt: skip
    src.add_argument("--conflict-tolerance", type=float, metavar="M",
                     help="disagreement along the hole above which sources conflict "
                          "(default: the median interval of the threshold table)")  # fmt: skip

    area = parser.add_argument_group("area (give one)")
    area.add_argument("--rectangle", metavar="X0,Y0,X1,Y1")
    area.add_argument("--extend-model", metavar="MODEL", help="block model folder: its plan box")
    area.add_argument("--extend-padding", type=float, metavar="M",
                      help="grow the model's plan box by M")  # fmt: skip

    fit = parser.add_argument_group("interpolation")
    fit.add_argument("--spacing", type=float, required=True, metavar="M", help="grid spacing")
    fit.add_argument("--method", choices=("kriging", "rbf"), default="kriging")
    fit.add_argument("--variogram", metavar="JSON",
                     help="variogram model file (default: spherical, nugget 0, sill = variance "
                          "of the exact depths, range = 3 x median pick spacing)")  # fmt: skip
    fit.add_argument("--drift", choices=("constant", "linear"), default="constant")
    fit.add_argument("--above", metavar="OBJ",
                     help="a shallower surface already on file: this one is pushed down "
                          "wherever it would rise above it")  # fmt: skip

    out = parser.add_argument_group("output")
    out.add_argument("--out", required=True, metavar="FILE.obj", help="the surface as a DTM")
    out.add_argument("--geometry-dir", metavar="DIR",
                     help="also write <name>_vertices.npy / <name>_faces.npy here")  # fmt: skip
    out.add_argument("--name", help="name in --geometry-dir (default: the surface name with "
                                    ": as _)")  # fmt: skip
    out.add_argument("--picks-out", metavar="CSV", help="the combined picks, one row per hole")
    out.add_argument("--summary", metavar="JSON",
                     help="fit, diagnostics and provenance as JSON")  # fmt: skip
    parser.set_defaults(func=_guarded(run))


def _table(database: Any, name: str, column: str | None):
    if name not in database.intervals:
        have = ", ".join(sorted(database.intervals)) or "none"
        raise CommandError(f"the database has no interval table {name!r} (tables: {have})")
    frame = database.intervals[name]
    if column is not None and column not in frame.columns:
        have = ", ".join(str(c) for c in frame.columns)
        raise CommandError(f"table {name!r} has no column {column!r} (columns: {have})")
    return frame


def _threshold(text: str) -> tuple[str, str, str, float]:
    parts = text.rsplit(":", 2)
    rule = parts[-1].strip() if len(parts) == 3 else ""
    try:
        value = float(rule[1:])
    except ValueError:
        value = None
    if len(parts) != 3 or not parts[0] or not parts[1] or rule[:1] not in "<>" or value is None:
        raise CommandError(
            f"--threshold expects TABLE:COLUMN:<VALUE or TABLE:COLUMN:>VALUE, got {text!r}"
        )
    return parts[0].strip(), parts[1].strip(), rule[0], value


def _dtm(path: str, option: str):
    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.geometry.mesh_import import import_mesh

    if not Path(path).is_file():
        raise CommandError(f"{option}: no file {path}")
    return DTM(import_mesh(path, "dtm", 1.0)[0])


def _rectangle(args: argparse.Namespace) -> list[float]:
    if bool(args.rectangle) == bool(args.extend_model):
        raise CommandError("give exactly one of --rectangle and --extend-model")
    if args.rectangle:
        if args.extend_padding is not None:
            raise CommandError("--extend-padding goes with --extend-model")
        return _floats(args.rectangle, 4, "--rectangle")
    from bakumine.core.blockmodel.storage import load_model
    from bakumine.core.geometry.surface_points import model_rectangle

    box = model_rectangle(load_model(args.extend_model).definition)
    pad = float(args.extend_padding or 0.0)
    if pad < 0.0:
        raise CommandError("--extend-padding must be >= 0")
    return [box[0] - pad, box[1] - pad, box[2] + pad, box[3] + pad]


def _counts_text(counts: dict[str, int]) -> str:
    return ", ".join(f"{counts[k]} {k}" for k in ("exact", "lower", "upper", "between", "none"))


def run(args: argparse.Namespace) -> int:
    import numpy as np

    from bakumine.core.estimation.variogram import VariogramModel
    from bakumine.core.geology import surfaces as sf
    from bakumine.core.geology.picks import read_extra_intervals
    from bakumine.core.geology.provenance import Provenance
    from bakumine.core.geology.units import UnitDictionary
    from bakumine.core.io.files import write_mesh
    from bakumine.core.io.obj import write_obj

    if not args.unit and not args.threshold:
        raise CommandError("give --unit CODE (logged units), --threshold, or both")
    if args.unit and not args.units:
        raise CommandError("--unit needs --units (the unit dictionary)")
    if args.extra_intervals and not args.unit:
        raise CommandError("--extra-intervals needs --unit (the codes of the zone)")
    name = args.surface or (f"{args.side}:{args.unit[0]}" if args.unit else None)
    if not name:
        raise CommandError("give --surface NAME (there is no --unit to name it after)")
    geometry_name = args.name or sf.surface_file_stem(name)
    if args.geometry_dir and (not _NAME.match(geometry_name) or ".." in geometry_name):
        raise CommandError(
            f"--geometry-dir needs a plain --name (letters, digits, _ - .), got {geometry_name!r}"
        )
    if not args.spacing > 0.0:
        raise CommandError("--spacing must be > 0")
    rectangle = _rectangle(args)
    variogram = None
    if args.variogram:
        if args.method == "rbf":
            raise CommandError("--variogram does not apply with --method rbf")
        variogram = VariogramModel.from_dict(
            json.loads(Path(args.variogram).read_text(encoding="utf-8"))
        )

    database = _database(args.database)
    traces = database.desurvey()
    topography = _dtm(args.topography, "--topography")
    sources = []
    if args.unit:
        units = UnitDictionary.load(args.units)
        table_name = args.table or units.source_table or "geology"
        column = args.column or units.source_column
        if not column:
            raise CommandError("the dictionary names no source column; give --column")
        sources.append(
            sf.log_source(
                _table(database, table_name, column),
                column,
                units,
                args.unit,
                description=f"table {table_name}, column {column}",
            )
        )
        if args.extra_intervals:
            sources.append(
                sf.log_source(
                    read_extra_intervals(args.extra_intervals),
                    "code",
                    units,
                    args.unit,
                    source="extra",
                    description=f"file {Path(args.extra_intervals).name}",
                )
            )
    if args.threshold:
        t_table, t_column, operator, value = _threshold(args.threshold)
        limit = {"below": value} if operator == "<" else {"above": value}
        sources.append(
            sf.threshold_source(
                _table(database, t_table, t_column),
                t_column,
                description=f"table {t_table}",
                **limit,
            )
        )

    picks = sf.surface_picks(
        sources,
        traces,
        topography,
        surface=name,
        side=args.side,
        mode=args.mode,
        absent=args.absent,
        conflict_tolerance=args.conflict_tolerance,
    )
    surface = sf.depth_surface(
        picks,
        topography,
        rectangle,
        args.spacing,
        name=name,
        method=args.method,
        variogram=variogram,
        drift=args.drift,
        keep_conflicts=args.keep_conflicts,
    )
    if args.above:
        surface = sf.nest_under(surface, _dtm(args.above, "--above"), above_name=args.above)

    d = surface.diagnostics
    print(f"surface {name}: {len(traces.hole_ids)} holes with a trace")
    for source, info in d["picks"]["sources"].items():
        print(f"  picks from {source} ({info['description']}): {_counts_text(info['counts'])}")
        for key, what in (
            ("holes_skipped", "in the table without a trace"),
            ("holes_overlapping_intervals", "with overlapping intervals"),
        ):
            if info[key]:
                listed = ", ".join(str(h) for h in info[key][:10])
                print(f"    {len(info[key])} hole(s) {what}: {listed}")
    print(f"  combined: {_counts_text(d['picks']['counts'])}")
    used = d["used"]
    print(
        f"  used in the fit: {used['exact']} exact, {used['lower']} lower, {used['upper']} upper, "
        f"{used['between']} between; {d['holes_without_information']} holes give no information"
    )
    if d["conflicts"]:
        action = "kept (--keep-conflicts)" if d["conflicts_kept"] else "left out of the fit"
        print(
            f"  {d['conflicts']} hole(s) where the sources contradict each other by more than "
            f"{d['picks']['conflict_tolerance']:g} m, {action}: {', '.join(d['holes_conflict'])}"
        )
    if d["holes_outside_topography"]:
        print(
            f"  {len(d['holes_outside_topography'])} hole(s) with a pick outside the topography, "
            f"left out: {', '.join(d['holes_outside_topography'])}"
        )
    shape = d["grid"]["shape"]
    print(
        f"  fit: {d['method']}, drift {d['drift']}, {d['iterations']} solve(s), "
        f"{d['active_lower']} lower and {d['active_upper']} upper bound(s) active"
    )
    print(
        f"  grid {shape[0]} x {shape[1]} nodes at {args.spacing:g} m; {d['nodes_valid']} under the "
        f"topography; depth {d['depth_min']:.2f} to {d['depth_max']:.2f} m "
        f"(mean {d['depth_mean']:.2f})"
    )
    if d["nodes_clamped"]:
        print(f"  {d['nodes_clamped']} node(s) with a fitted depth below 0 set to the topography")
    if args.above:
        print(f"  {surface.n_nested} node(s) moved down to {args.above}")
    print(
        f"  {100.0 * d['extrapolated_share']:.1f} % of the surface is extrapolated (farther than "
        "2 x the median pick spacing from any pick)"
    )

    mesh = surface.dtm.mesh
    what = f"dtm {name}, {mesh.n_vertices} vertices, {mesh.n_faces} triangles"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == ".obj":
        write_obj(out, mesh, decimals=None)
    else:
        write_mesh(out, mesh)
    _wrote(out, what)
    if args.geometry_dir:
        folder = Path(args.geometry_dir)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{geometry_name}_vertices.npy", mesh.vertices)
        np.save(folder / f"{geometry_name}_faces.npy", mesh.faces)
        _wrote(folder / f"{geometry_name}_vertices.npy", what)
        _wrote(folder / f"{geometry_name}_faces.npy", what)
    if args.picks_out:
        path = Path(args.picks_out)
        path.parent.mkdir(parents=True, exist_ok=True)
        surface.picks.to_csv(path, index=False)
        _wrote(path, f"picks, {len(surface.picks)} holes")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(surface.summary(), indent=2), encoding="utf-8")
        _wrote(path, "summary")

    record = Provenance(object_kind="surface", name=name)
    record.extend(surface.provenance)
    print("\n" + record.format_text())
    return 0
