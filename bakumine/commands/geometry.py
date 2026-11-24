"""Surface subcommands: ``bakumine surface from-points | from-raster | check | import``.

    bakumine surface from-points --database DB --extend-padding 200 --project P --name Topo
    bakumine surface from-raster dem.tif --database DB --model M --padding 100 --out topo.obj
    bakumine surface check topo.obj --database DB
    bakumine surface import pit.dxf --kind dtm --project P

Every command can write its mesh three ways, in any combination: ``--out FILE`` (``.obj``,
``.ply``, ``.dxf`` or ``.dtm``), ``--geometry-dir DIR`` (``<name>_vertices.npy`` and
``<name>_faces.npy``, what ``blockmodel code`` and ``estimate`` read) and ``--project DIR``
(an OBJ added to the project as a DTM or solid, which the viewer shows).

The same core functions run behind the dialogs (:mod:`bakumine.api.operations_geometry`);
:data:`REPLAY` turns their job-log entries back into these commands.
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP_SURFACE = "Surfaces: build a DTM from points (collars, tables, strings) or import a mesh"

_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")


# ------------------------------------------------------------------ shared


def _floats(text: str, count: int, option: str) -> list[float]:
    try:
        values = [float(v) for v in text.replace(";", ",").split(",") if v.strip()]
    except ValueError:
        values = []
    if len(values) != count:
        raise CommandError(f"{option} expects {count} numbers separated by commas, got {text!r}")
    return values


def _add_outputs(parser: argparse.ArgumentParser, kind: str) -> None:
    group = parser.add_argument_group("output (give at least one)")
    group.add_argument("--out", metavar="FILE", help="mesh file: .obj, .ply, .dxf or .dtm")
    group.add_argument("--geometry-dir", metavar="DIR",
                       help="write <name>_vertices.npy / <name>_faces.npy here")  # fmt: skip
    group.add_argument("--project", metavar="DIR", help=f"add the {kind} to this project")
    group.add_argument("--name", help="object name (default: from the command)")
    group.add_argument("--id", help="project object id (default: from the name)")
    group.add_argument("--summary", metavar="JSON", help="also write the summary as JSON")


def _need_output(args: argparse.Namespace) -> None:
    """Said before the work, not after it."""
    if not (args.out or args.geometry_dir or args.project):
        raise CommandError("give --out, --geometry-dir or --project (where the mesh goes)")


def _emit(mesh: Any, kind: str, name: str, args: argparse.Namespace, summary: dict) -> None:
    import numpy as np

    from bakumine.core.io.files import write_mesh
    from bakumine.core.io.obj import write_obj

    _need_output(args)
    what = f"{kind}, {mesh.n_vertices} vertices, {mesh.n_faces} triangles"
    if args.out:
        path = Path(args.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".obj":
            write_obj(path, mesh, decimals=None)
        else:
            write_mesh(path, mesh)
        _wrote(path, what)
    if args.geometry_dir:
        if not _NAME.match(name) or ".." in name:
            raise CommandError(
                f"--geometry-dir needs a plain --name (letters, digits, _ - .), got {name!r}"
            )
        folder = Path(args.geometry_dir)
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / f"{name}_vertices.npy", mesh.vertices)
        np.save(folder / f"{name}_faces.npy", mesh.faces)
        _wrote(folder / f"{name}_vertices.npy", what)
        _wrote(folder / f"{name}_faces.npy", what)
    if args.project:
        from bakumine.core.project import Project, _slug

        with tempfile.TemporaryDirectory() as tmp:
            file = Path(tmp) / f"{_slug(name)}.obj"
            write_obj(file, mesh, decimals=None)
            obj = Project.open(args.project).add(kind, file, name, args.id)
        print(f"added {obj.kind} {obj.id!r} as {obj.path}")
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _wrote(path, "summary")


def _database(folder: str):
    from bakumine.core.drillholes.importer import load_database

    path = Path(folder)
    if not (path / "database.json").is_file():
        raise CommandError(f"{path} is not a drillhole database folder (no database.json)")
    return load_database(path)


# ------------------------------------------------------------------ surface from-points


def configure_from_points(parser: argparse.ArgumentParser) -> None:
    src = parser.add_argument_group("sources (any combination)")
    src.add_argument("--database", action="append", default=[], metavar="DIR",
                     help="drillhole database folder: its collars (repeatable)")  # fmt: skip
    src.add_argument("--points", action="append", default=[], metavar="FILE",
                     help="CSV / XLSX table of points (repeatable)")  # fmt: skip
    src.add_argument("--columns", action="append", default=[], metavar="X,Y,Z",
                     help="x,y,z column names of --points (east, north, up); once for all "
                          "tables or once per table. Default: by header name")  # fmt: skip
    src.add_argument("--strings", action="append", default=[], metavar="FILE",
                     help=".str or .dxf string file: its points (repeatable)")  # fmt: skip
    src.add_argument("--breaklines", action="append", default=[], metavar="FILE",
                     help=".str / .dxf strings whose segments become triangle edges")  # fmt: skip
    ext = parser.add_argument_group(
        "extension",
        "Adds points on a rectangle: corners and points along the sides (those inside the "
        "hull of the data are left out). Their elevation is that of the nearest data point "
        "in plan (--extend-z nearest, default) or the given value.",
    )
    ext.add_argument("--extend-padding", type=float, metavar="M",
                     help="grow the rectangle (without one: the data box) by M")  # fmt: skip
    ext.add_argument("--extend-rectangle", metavar="XMIN,YMIN,XMAX,YMAX")
    ext.add_argument("--extend-model", metavar="DIR", help="block model folder: its plan box")
    ext.add_argument("--extend-z", default="nearest", metavar="nearest|Z")
    ext.add_argument("--extend-step", type=float, metavar="M",
                     help="spacing of the added points (default: the data spacing)")  # fmt: skip
    parser.add_argument("--duplicates", default="mean",
                        choices=("mean", "first", "last", "min", "max", "error"),
                        help="points with equal XY and different Z (default: mean)")  # fmt: skip
    _add_outputs(parser, "DTM")
    parser.set_defaults(func=_guarded(run_from_points))


def run_from_points(args: argparse.Namespace) -> int:
    from bakumine.core.geometry import surface_points as sp
    from bakumine.core.io.files import read_strings

    _need_output(args)
    if len(args.columns) not in (0, 1, len(args.points)):
        raise CommandError("give --columns once, or once per --points table")
    clouds, sources = [], []
    for folder in args.database:
        pts = sp.collar_points(_database(folder))
        clouds.append(pts)
        sources.append(f"{len(pts)} collars of {folder}")
    for i, file in enumerate(args.points):
        columns = None
        if args.columns:
            columns = [c.strip() for c in args.columns[min(i, len(args.columns) - 1)].split(",")]
        pts, info = sp.read_points_table(file, columns)
        clouds.append(pts)
        skipped = f", {info['skipped']} rows without coordinates skipped" if info["skipped"] else ""
        sources.append(
            f"{len(pts)} points of {file} (columns {', '.join(info['columns'])}{skipped})"
        )
    for file in args.strings:
        pts = sp.string_points(read_strings(file))
        clouds.append(pts)
        sources.append(f"{len(pts)} string points of {file}")
    breaklines = []
    for file in args.breaklines:
        lines = sp.string_lines(read_strings(file))
        breaklines += lines
        sources.append(f"{len(lines)} breaklines of {file}")
    if not clouds and not breaklines:
        raise CommandError("give at least one of --database, --points, --strings, --breaklines")

    rectangle = None
    if args.extend_rectangle and args.extend_model:
        raise CommandError("give --extend-rectangle or --extend-model, not both")
    if args.extend_rectangle:
        rectangle = _floats(args.extend_rectangle, 4, "--extend-rectangle")
    elif args.extend_model:
        from bakumine.core.blockmodel.storage import load_model

        rectangle = sp.model_rectangle(load_model(args.extend_model).definition)
    z: float | str = args.extend_z
    if z != "nearest":
        try:
            z = float(z)
        except ValueError:
            raise CommandError("--extend-z expects 'nearest' or a number") from None
    result = sp.surface_from_points(
        clouds,
        breaklines,
        rectangle=rectangle,
        padding=args.extend_padding or None,
        extend_z=z,
        step=args.extend_step,
        duplicates=args.duplicates,
    )
    info = result.info
    for line in sources:
        print(line)
    if info["n_duplicates"]:
        print(
            f"{info['n_duplicates']} duplicate points merged; {info['n_conflicts']} locations "
            f"with different Z resolved by '{info['duplicates']}'"
        )
    if info["rectangle"]:
        rule = (
            "the elevation of the nearest data point"
            if info["extend_z"] == "nearest"
            else f"elevation {info['extend_z']}"
        )
        print(
            f"extended to {', '.join(f'{v:.3f}' for v in info['rectangle'])} with "
            f"{info['n_extension_points']} boundary points at {rule}"
        )
    print(
        f"surface: {info['n_vertices']} vertices, {info['n_faces']} triangles, plan area "
        f"{info['plan_area']:.1f} m2, Z {info['bounds'][0][2]:.2f} to {info['bounds'][1][2]:.2f}"
    )
    _emit(result.dtm.mesh, "dtm", args.name or "topography", args, info)
    return 0


# ------------------------------------------------------------------ surface from-raster


def _collar_check_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--collar-tolerance",
        type=float,
        default=5.0,
        metavar="M",
        help="list collars whose height differs from the (shifted) surface by "
        "more than M metres after the median shift (default 5)",
    )
    parser.add_argument(
        "--residuals",
        metavar="CSV",
        help="write the per-collar comparison (hole_id, x, y, z, surface_z, dz)",
    )


def _report_collars(table: Any, summary: dict[str, Any], args: argparse.Namespace) -> None:
    n, on = summary["collars"], summary["on_surface"]
    if summary["median"] is None:
        print(f"collars: none of {n} lies on the surface")
    else:
        s = summary["spread"]
        print(
            f"collars: {on} of {n} on the surface; collar z - surface z: median "
            f"{summary['median']:+.2f} m, mean {summary['mean']:+.2f} m; after the median: "
            f"std {s['std']:.2f} m, 90 % within {s['p90_abs']:.2f} m, worst {s['max_abs']:.2f} m"
        )
    if summary["off_surface"]:
        names = ", ".join(summary["off_surface"][:10])
        more = " ..." if len(summary["off_surface"]) > 10 else ""
        print(f"  {len(summary['off_surface'])} collars outside the surface: {names}{more}")
    suspect = summary["suspect"]
    if suspect:
        print(
            f"  {len(suspect)} collars differ by more than {summary['tolerance']:g} m from the "
            "median (wrong collar height, a dump or pit made after the surface, or a surface "
            "error):"
        )
        for row in suspect[:15]:
            dz, off = row["dz"], row["off_median"]
            print(f"    {row['hole_id']}: dz {dz:+.2f} m ({off:+.2f} from the median)")
        if len(suspect) > 15:
            print(f"    ... {len(suspect) - 15} more (see --residuals)")
    if args.residuals:
        path = Path(args.residuals)
        path.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(path, index=False)
        _wrote(path, f"{len(table)} collars")


def configure_from_raster(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "files",
        nargs="+",
        metavar="FILE",
        help="GeoTIFF (.tif) or ESRI ASCII grid (.asc); several = adjacent tiles",
    )
    parser.add_argument(
        "--raster-crs",
        metavar="CRS",
        help="CRS of the raster files when they do not say it (EPSG:4326 ...)",
    )
    parser.add_argument(
        "--crs",
        metavar="CRS",
        help="projected CRS of the model, e.g. EPSG:32648 (WGS 84 / UTM 48N). "
        "Default: the CRS of --database, else that of a projected raster",
    )
    area = parser.add_argument_group(
        "area (default: the collars of --database)",
        "The surface covers this plan rectangle, grown by --padding.",
    )
    area.add_argument("--rectangle", metavar="XMIN,YMIN,XMAX,YMAX")
    area.add_argument("--model", metavar="DIR", help="block model folder: its plan box")
    area.add_argument("--padding", type=float, default=0.0, metavar="M")
    parser.add_argument(
        "--spacing",
        type=float,
        metavar="M",
        help="grid spacing of the surface (default: the raster cell size)",
    )
    tie = parser.add_argument_group(
        "collars",
        "With --database the surface is compared with the collars. A satellite DEM measures "
        "heights from another zero (a geoid) than a local survey: --z-shift median (default "
        "with --database) moves the surface by the median of collar z - surface z.",
    )
    tie.add_argument("--database", metavar="DIR", help="drillhole database folder")
    tie.add_argument(
        "--z-shift",
        metavar="median|none|M",
        help="vertical shift added to the raster heights (default: median with "
        "--database, else none)",
    )
    _collar_check_options(tie)
    _add_outputs(parser, "DTM")
    parser.set_defaults(func=_guarded(run_from_raster))


def run_from_raster(args: argparse.Namespace) -> int:
    import numpy as np

    from bakumine.core.geometry import raster_dem as rd
    from bakumine.core.geometry import surface_points as sp
    from bakumine.core.geometry.dtm import DTM, dtm_from_grid

    _need_output(args)
    rasters = []
    for file in args.files:
        if not Path(file).is_file():
            raise CommandError(f"no file {file}")
        r = rd.read_raster(file, args.raster_crs)
        rasters.append(r)
        nrow, ncol = r.shape
        print(
            f"{r.source}: {r.info['format']}, {ncol} x {nrow} cells of {abs(r.dx):g} x "
            f"{abs(r.dy):g}, CRS {r.crs or 'not given'}, heights {np.nanmin(r.z):.1f} to "
            f"{np.nanmax(r.z):.1f}"
        )
    db = _database(args.database) if args.database else None

    crs = args.crs
    if crs is None and db is not None and db.metadata.crs:
        crs = db.metadata.crs
    if crs is None:
        projected = [r.crs for r in rasters if r.crs and not r.geographic]
        if projected and not any(r.geographic for r in rasters):
            crs = projected[0]
    if crs is None and any(r.crs for r in rasters):
        raise CommandError(
            "give --crs, the projected CRS of the model (e.g. EPSG:32648 for WGS 84 / UTM 48N)"
        )
    if crs is not None:
        crs = rd._crs_text(crs)
        for r in rasters:
            if r.crs is None:
                r.crs = crs
                print(f"{r.source}: no CRS in the file, taken to be the model CRS {crs}")

    if args.rectangle and args.model:
        raise CommandError("give --rectangle or --model, not both")
    if args.rectangle:
        rectangle = _floats(args.rectangle, 4, "--rectangle")
    elif args.model:
        from bakumine.core.blockmodel.storage import load_model

        rectangle = sp.model_rectangle(load_model(args.model).definition)
    elif db is not None:
        xy = sp.collar_points(db)[:, :2]
        if len(xy) == 0:
            raise CommandError("the database has no collar with coordinates")
        rectangle = [*xy.min(axis=0), *xy.max(axis=0)]
    else:
        raise CommandError("give the area: --rectangle, --model or --database")
    pad = float(args.padding or 0.0)
    if pad < 0:
        raise CommandError("--padding must be >= 0")
    rectangle = [rectangle[0] - pad, rectangle[1] - pad, rectangle[2] + pad, rectangle[3] + pad]

    spacing = args.spacing
    if spacing is None:
        latitude = None
        if rasters[0].geographic and crs is not None:
            from pyproj import Transformer

            t = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
            latitude = t.transform(
                0.5 * (rectangle[0] + rectangle[2]), 0.5 * (rectangle[1] + rectangle[3])
            )[1]
        spacing = round(rd.native_spacing(rasters[0], latitude), 1) or rd.native_spacing(rasters[0])
    z, origin, grid = rd.grid_from_rasters(rasters, rectangle, spacing, crs)
    if grid["nodes_with_height"] == 0:
        raise CommandError(
            "no node of the area lies on the raster: check --crs / --raster-crs and the area"
        )
    print(
        f"grid: {grid['nodes'][0]} x {grid['nodes'][1]} nodes at {spacing:g} m over "
        f"{', '.join(f'{v:.1f}' for v in grid['rectangle'])}" + (f" in {crs}" if crs else "")
    )
    if grid["nodes_without_height"]:
        print(f"  {grid['nodes_without_height']} nodes without height (off the raster or no data)")

    shift_rule = args.z_shift or ("median" if db is not None else "none")
    shift = 0.0
    summary: dict[str, Any] = {"grid": grid, "crs": crs, "rasters": [r.source for r in rasters]}
    if shift_rule not in ("median", "none"):
        try:
            shift = float(shift_rule)
        except ValueError:
            raise CommandError("--z-shift expects median, none or a number of metres") from None
    if shift_rule == "median" and db is None:
        raise CommandError("--z-shift median needs --database (the collars to tie to)")
    raw = dtm_from_grid(z, origin, (spacing, spacing))
    if db is not None:
        if shift_rule == "median":
            first = rd.residual_summary(rd.collar_residuals(raw, db.collar), args.collar_tolerance)
            if first["median"] is not None:
                shift = first["median"]
                print(f"vertical shift: {shift:+.2f} m (median of collar z - raster z)")
        table = rd.collar_residuals(raw, db.collar, shift)
        check = rd.residual_summary(table, args.collar_tolerance)
        _report_collars(table, check, args)
        summary["collars"] = check
    summary["z_shift"] = shift
    mesh = (
        raw.mesh
        if not shift
        else DTM(dtm_from_grid(z + shift, origin, (spacing, spacing)).mesh).mesh
    )
    lo, hi = mesh.vertices.min(axis=0), mesh.vertices.max(axis=0)
    print(
        f"surface: {mesh.n_vertices} vertices, {mesh.n_faces} triangles, "
        f"Z {lo[2]:.2f} to {hi[2]:.2f}"
    )
    _emit(mesh, "dtm", args.name or "topography", args, summary)
    return 0


# ------------------------------------------------------------------ surface check


def configure_check(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file",
        nargs="?",
        help="surface file (.obj, .dxf, .ply, .stl, .dtm); or --geometry-dir + --name",
    )
    parser.add_argument("--geometry-dir", metavar="DIR")
    parser.add_argument("--name", help="surface name in --geometry-dir")
    parser.add_argument(
        "--database", required=True, metavar="DIR", help="drillhole database folder"
    )
    _collar_check_options(parser)
    parser.add_argument("--summary", metavar="JSON", help="also write the summary as JSON")
    parser.set_defaults(func=_guarded(run_check))


def run_check(args: argparse.Namespace) -> int:
    import numpy as np

    from bakumine.core.geometry import raster_dem as rd
    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.geometry.mesh import Mesh

    if args.file and args.geometry_dir:
        raise CommandError("give a surface file or --geometry-dir, not both")
    if args.file:
        from bakumine.core.geometry.mesh_import import import_mesh

        mesh, _ = import_mesh(args.file, "dtm", 1.0)
        what = args.file
    elif args.geometry_dir and args.name:
        folder = Path(args.geometry_dir)
        mesh = Mesh(
            np.load(folder / f"{args.name}_vertices.npy"),
            np.load(folder / f"{args.name}_faces.npy"),
        )
        what = f"{args.name} in {folder}"
    else:
        raise CommandError("give a surface file, or --geometry-dir and --name")
    table = rd.collar_residuals(DTM(mesh), _database(args.database).collar)
    summary = rd.residual_summary(table, args.collar_tolerance)
    print(f"surface {what}: {mesh.n_vertices} vertices, {mesh.n_faces} triangles")
    _report_collars(table, summary, args)
    if args.summary:
        path = Path(args.summary)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        _wrote(path, "summary")
    return 0


# ------------------------------------------------------------------ surface import


def configure_import(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "file", help=".dxf, .obj, .ply, .stl or Surpac text .dtm (its .str next to it)"
    )
    parser.add_argument("--kind", default="auto", choices=("auto", "dtm", "solid"),
                        help="auto: solid when the mesh is closed, else dtm")  # fmt: skip
    parser.add_argument("--scale", type=float, default=1.0, metavar="FACTOR",
                        help="multiply every coordinate by this before the mesh is checked "
                        "(0.3048 for a file in feet; default 1)")  # fmt: skip
    _add_outputs(parser, "mesh")
    parser.set_defaults(func=_guarded(run_import))


def run_import(args: argparse.Namespace) -> int:
    from bakumine.core.geometry.mesh_import import import_mesh

    _need_output(args)
    mesh, info = import_mesh(args.file, args.kind, args.scale)
    line = f"{info['kind']}: {info['n_vertices']} vertices, {info['n_faces']} triangles"
    if info["kind"] == "solid":
        line += f", {'valid' if info['valid'] else 'NOT VALID'}"
        if info["volume"] is not None:
            line += f", volume {info['volume']:.1f} m3"
    if "scale" in info:
        line += f" (coordinates x {info['scale']:g})"
    print(line)
    for issue in info.get("issues") or []:
        print(f"  issue: {issue}")
    for text in info.get("warnings") or []:
        print(f"warning: {text}")
    _emit(mesh, info["kind"], args.name or Path(args.file).stem, args, info)
    return 0


def configure_surface(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="surface_command", metavar="<action>")
    configure_from_points(
        sub.add_parser(
            "from-points",
            help="DTM from collars, point tables and strings",
            description="Triangulate points into a DTM (topography), optionally with "
            "breaklines and extended to a rectangle.",
        )
    )
    configure_from_raster(
        sub.add_parser(
            "from-raster",
            help="DTM from an elevation grid (GeoTIFF / ASCII grid: satellite DEM, drone, LiDAR)",
            description="Resample elevation rasters onto a regular grid in the model CRS over "
            "the model area, tie the heights to the collars and triangulate.",
        )
    )
    configure_check(
        sub.add_parser(
            "check",
            help="compare a surface with the collar heights of a database",
            description="Collar z - surface z for every collar: median (the vertical shift "
            "between them), spread and the collars that disagree.",
        )
    )
    configure_import(
        sub.add_parser(
            "import",
            help="read a surface or solid file (DXF / OBJ / PLY / STL / .dtm)",
            description="Read a mesh file, check it and write it as a project object, "
            "OBJ or --geometry-dir files.",
        )
    )
    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "surface": (HELP_SURFACE, configure_surface),
}


# ------------------------------------------------------------------ job-log replay


def _where(item: dict[str, str]) -> str:
    from bakumine.commands.replay import PROJECT, SOURCE

    return f"{PROJECT}/{item['path']}" if "path" in item else f"{SOURCE}/{item['file']}"


def _finish(e: dict[str, Any], argv: list[str]) -> list[dict[str, Any]]:
    from bakumine.commands.replay import _add, _run, _work

    out = e["outputs"][0]
    file = _work(e["id"], Path(out["path"]).name)
    return [_run(*argv, "--out", file), _add(out, file)]


def _replay_surface(e: dict[str, Any]) -> list[dict[str, Any]]:
    replay, p = e.get("replay") or {}, e["params"]
    argv = ["surface", "from-points"]
    for item in replay.get("databases", []):
        argv += ["--database", _where(item)]
    for item in replay.get("points", []):
        argv += ["--points", _where(item), "--columns", ",".join(item["columns"])]
    for item in replay.get("strings", []):
        argv += ["--strings", _where(item)]
    for item in replay.get("breaklines", []):
        argv += ["--breaklines", _where(item)]
    extend = p.get("extend") or {}
    mode = extend.get("mode") or "none"
    if mode != "none":
        if mode == "rectangle":
            argv.append(
                "--extend-rectangle=" + ",".join(repr(float(v)) for v in extend["rectangle"])
            )
        elif mode == "model":
            argv += ["--extend-model", _where(replay["model"])]
        if float(extend.get("padding") or 0.0):
            argv.append(f"--extend-padding={float(extend['padding'])!r}")
        z = extend.get("z")
        if z not in (None, "", "nearest"):
            argv.append(f"--extend-z={float(z)!r}")
        if extend.get("step") not in (None, ""):
            argv.append(f"--extend-step={float(extend['step'])!r}")
    if p.get("duplicates"):
        argv += ["--duplicates", str(p["duplicates"])]
    return _finish(e, argv)


def _replay_import(e: dict[str, Any]) -> list[dict[str, Any]]:
    replay = e.get("replay") or {}
    argv = ["surface", "import", _where(replay), "--kind", replay["kind"]]
    if replay.get("scale") not in (None, 1, 1.0):
        argv += ["--scale", repr(float(replay["scale"]))]
    return _finish(e, argv)


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "surface_from_points": _replay_surface,
    "import_mesh": _replay_import,
}
