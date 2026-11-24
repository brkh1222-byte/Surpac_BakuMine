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
    dem = parser.add_argument_group(
        "DEM around the survey",
        "With --raster the grid of an elevation raster (satellite DEM, drone surface) fills the "
        "area around the precise sources: over the extension rectangle (or the data box grown "
        "by --extend-padding) its nodes are added, except those within --drop-within metres "
        "in plan of a source point or breakline, and are moved up or down by the median of "
        "source z - DEM z. No boundary points are added then: the raster reaches the edge.",
    )
    dem.add_argument("--raster", action="append", default=[], metavar="FILE",
                     help="GeoTIFF / ASCII grid (repeatable: adjacent tiles)")  # fmt: skip
    dem.add_argument("--raster-crs", metavar="CRS",
                     help="CRS of the raster files when they do not say it")  # fmt: skip
    dem.add_argument("--crs", metavar="CRS",
                     help="projected CRS of the model (default: of the first --database, "
                          "else of a projected raster)")  # fmt: skip
    dem.add_argument("--raster-spacing", type=float, metavar="M",
                     help="grid spacing (default: the raster cell size)")  # fmt: skip
    dem.add_argument("--raster-z-shift", metavar="median|none|M",
                     help="vertical shift of the raster heights (default: median)")  # fmt: skip
    dem.add_argument("--drop-within", type=float, metavar="M",
                     help="leave out raster nodes this close to a source (default: the grid "
                          "spacing; 0 keeps them all)")  # fmt: skip
    _fetch_options(dem)
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
    padding = args.extend_padding or None
    dem_info = None
    if args.raster or args.fetch:
        if z != "nearest" or args.extend_step:
            raise CommandError("--extend-z and --extend-step do not apply with a raster")
        dem_info = _dem_around(args, clouds, breaklines, rectangle, padding)
        clouds.append(dem_info.pop("points"))
        rectangle = padding = None
    else:
        given = [
            option
            for option, value in (
                ("--raster-crs", args.raster_crs),
                ("--crs", args.crs),
                ("--raster-spacing", args.raster_spacing),
                ("--raster-z-shift", args.raster_z_shift),
                ("--drop-within", args.drop_within),
                ("--dem-cache", args.dem_cache),
            )
            if value is not None
        ]
        if given:
            raise CommandError(f"{', '.join(given)} need --raster or --fetch")
    result = sp.surface_from_points(
        clouds,
        breaklines,
        rectangle=rectangle,
        padding=padding,
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
    if dem_info is not None:
        info["dem"] = dem_info
    _emit(result.dtm.mesh, "dtm", args.name or "topography", args, info)
    return 0


def _dem_around(
    args: argparse.Namespace,
    clouds: list[Any],
    breaklines: list[Any],
    rectangle: list[float] | None,
    padding: float | None,
) -> dict[str, Any]:
    """Raster nodes around the precise sources (``points`` key) and what was done."""
    import numpy as np

    from bakumine.core.geometry import raster_dem as rd

    survey = np.concatenate(clouds) if clouds else np.zeros((0, 3))
    every = np.concatenate([survey, *breaklines]) if breaklines else survey
    if rectangle is None:
        rectangle = [*every[:, :2].min(axis=0), *every[:, :2].max(axis=0)]
    pad = float(padding or 0.0)
    if pad < 0:
        raise CommandError("--extend-padding must be >= 0")
    area = [rectangle[0] - pad, rectangle[1] - pad, rectangle[2] + pad, rectangle[3] + pad]
    if not (area[2] > area[0] and area[3] > area[1]):
        raise CommandError("the area of the raster is empty: give --extend-padding or a rectangle")
    db = _database(args.database[0]) if args.database else None
    files = [*args.raster, *_fetched(args, area, db)]
    rasters, crs = _load_rasters(files, args.raster_crs, args.crs, db)
    z, origin, spacing, grid = _raster_grid(rasters, crs, area, args.raster_spacing)
    rule: float | str = args.raster_z_shift or "median"
    if rule not in ("median", "none"):
        try:
            rule = float(rule)
        except ValueError:
            raise CommandError("--raster-z-shift expects median, none or metres") from None
    try:
        points, info = rd.dem_fill(
            survey, breaklines, z, origin, spacing, drop_within=args.drop_within, z_shift=rule
        )
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    tie = info["tie"]
    if tie is not None:
        print(
            f"raster vs sources at {tie['points']} points: median source z - raster z "
            f"{tie['median']:+.2f} m; after the shift 90 % within {tie['p90_abs']:.2f} m, "
            f"worst {tie['max_abs']:.2f} m"
        )
    print(
        f"raster: shift {info['z_shift']:+.2f} m; {info['kept']} of {info['nodes']} nodes kept, "
        f"{info['dropped']} within {info['drop_within']:g} m of a source left out"
    )
    return {"points": points, **info, "grid": grid, "crs": crs,
            "rasters": [r.source for r in rasters]}  # fmt: skip


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


def _load_rasters(files: list[str], raster_crs: str | None, crs: str | None, db: Any):
    """Read the rasters and settle the model CRS (given, of the database, of the rasters)."""
    import numpy as np

    from bakumine.core.geometry import raster_dem as rd

    rasters = []
    for file in files:
        if not Path(file).is_file():
            raise CommandError(f"no file {file}")
        r = rd.read_raster(file, raster_crs)
        rasters.append(r)
        nrow, ncol = r.shape
        print(
            f"{r.source}: {r.info['format']}, {ncol} x {nrow} cells of {abs(r.dx):g} x "
            f"{abs(r.dy):g}, CRS {r.crs or 'not given'}, heights {np.nanmin(r.z):.1f} to "
            f"{np.nanmax(r.z):.1f}"
        )
    try:
        crs, notes = rd.settle_crs(rasters, crs, db.metadata.crs if db is not None else None)
    except ValueError:
        raise CommandError(
            "give --crs, the projected CRS of the model (e.g. EPSG:32633 for WGS 84 / UTM 33N)"
        ) from None
    for note in notes:
        print(note)
    return rasters, crs


def _raster_grid(rasters: list[Any], crs: str | None, rectangle: list[float], spacing: Any):
    """Heights of the rasters on a grid over ``rectangle``; spacing default: the cell size."""
    from bakumine.core.geometry import raster_dem as rd

    if spacing is None:
        spacing = rd.default_spacing(rasters, crs, rectangle)
    elif not spacing > 0:
        raise CommandError("the grid spacing must be > 0")
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
    return z, origin, spacing, grid


_MAX_TILES = 16


def _fetch_options(group: Any) -> None:
    group.add_argument(
        "--fetch",
        choices=("copernicus30",),
        help="download the Copernicus GLO-30 tiles (30 m, 2011-2015) that cover the area. Off "
        "unless given: the tile names, so the location of the area to the nearest degree, are "
        "sent to the public AWS bucket copernicus-dem-30m",
    )
    group.add_argument(
        "--dem-cache",
        metavar="DIR",
        help="folder of downloaded tiles, reused on the next run (default ~/.cache/bakumine/dem)",
    )


def _fetched(args: argparse.Namespace, area: list[float], db: Any) -> list[str]:
    """Paths of the tiles of ``--fetch`` over ``area`` (none without the option)."""
    if not getattr(args, "fetch", None):
        if getattr(args, "dem_cache", None):
            raise CommandError("--dem-cache needs --fetch")
        return []
    from bakumine.core.geometry import dem_fetch as df
    from bakumine.core.geometry import raster_dem as rd

    crs = args.crs or (db.metadata.crs if db is not None else None)
    if not crs:
        raise CommandError("--fetch needs the model CRS: give --crs, e.g. EPSG:32633")
    try:
        names = df.copernicus_tiles(*df.lonlat_box(area, rd._crs_text(crs)))
    except ValueError as exc:
        raise CommandError(str(exc)) from None
    if len(names) > _MAX_TILES:
        raise CommandError(
            f"the area needs {len(names)} tiles (more than {_MAX_TILES}): check --crs and the area"
        )
    print(
        f"fetch copernicus30: {len(names)} tile(s) {', '.join(names)}; tile names go to "
        f"{df.COPERNICUS30_URL}"
    )
    got = df.fetch_tiles(names, args.dem_cache, progress=lambda line: print(f"  {line}"))
    if got["cached"]:
        print(f"  {len(got['cached'])} tile(s) already in the cache")
    if got["missing"]:
        print(f"  not in the bucket (sea or no data): {', '.join(got['missing'])}")
    if not got["files"]:
        raise CommandError("no Copernicus tile covers the area")
    return [str(p) for p in got["files"]]


def configure_from_raster(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "files",
        nargs="*",
        metavar="FILE",
        help="GeoTIFF (.tif) or ESRI ASCII grid (.asc); several = adjacent tiles. "
        "None with --fetch",
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
    _fetch_options(parser.add_argument_group("download"))
    _add_outputs(parser, "DTM")
    parser.set_defaults(func=_guarded(run_from_raster))


def run_from_raster(args: argparse.Namespace) -> int:
    from bakumine.core.geometry import raster_dem as rd
    from bakumine.core.geometry import surface_points as sp

    _need_output(args)
    if not args.files and not args.fetch:
        raise CommandError("give raster files, or --fetch copernicus30")
    db = _database(args.database) if args.database else None

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

    files = [*args.files, *_fetched(args, rectangle, db)]
    rasters, crs = _load_rasters(files, args.raster_crs, args.crs, db)
    z, origin, spacing, grid = _raster_grid(rasters, crs, rectangle, args.spacing)

    rule: float | str | None = args.z_shift
    if rule not in (None, "median", "none"):
        try:
            rule = float(rule)
        except ValueError:
            raise CommandError("--z-shift expects median, none or a number of metres") from None
    if rule == "median" and db is None:
        raise CommandError("--z-shift median needs --database (the collars to tie to)")
    collar = db.collar if db is not None else None
    dtm, tie, table = rd.raster_surface(
        z, origin, spacing, collar, z_shift=rule, tolerance=args.collar_tolerance
    )
    summary: dict[str, Any] = {"grid": grid, "crs": crs, "rasters": [r.source for r in rasters]}
    if tie["collars"] is not None:
        if (rule or "median") == "median":
            print(f"vertical shift: {tie['z_shift']:+.2f} m (median of collar z - raster z)")
        _report_collars(table, tie["collars"], args)
        summary["collars"] = tie["collars"]
    summary["z_shift"] = tie["z_shift"]
    mesh = dtm.mesh
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
    dem = replay.get("raster")
    if dem:
        for file in dem["files"]:
            argv += ["--raster", _where({"file": file})]
        argv += _raster_flags(dem, "--raster-spacing")
        argv.append(f"--raster-z-shift={_shift_text(dem['z_shift'])}")
        if dem.get("drop_within") is not None:
            argv.append(f"--drop-within={float(dem['drop_within'])!r}")
    return _finish(e, argv)


def _shift_text(value: Any) -> str:
    return value if isinstance(value, str) else repr(float(value))


def _raster_flags(item: dict[str, Any], spacing_option: str) -> list[str]:
    argv = []
    if item.get("raster_crs"):
        argv += ["--raster-crs", str(item["raster_crs"])]
    if item.get("crs"):
        argv += ["--crs", str(item["crs"])]
    argv.append(f"{spacing_option}={float(item['spacing'])!r}")
    return argv


def _replay_raster(e: dict[str, Any]) -> list[dict[str, Any]]:
    replay = e.get("replay") or {}
    argv = ["surface", "from-raster", *(_where({"file": f}) for f in replay["files"])]
    argv += _raster_flags(replay, "--spacing")
    if "database" in replay:
        argv += ["--database", _where(replay["database"])]
    if "model" in replay:
        argv += ["--model", _where(replay["model"])]
    if "rectangle" in replay:
        argv.append("--rectangle=" + ",".join(repr(float(v)) for v in replay["rectangle"]))
    if replay.get("padding"):
        argv.append(f"--padding={float(replay['padding'])!r}")
    if replay.get("z_shift") is not None:
        argv.append(f"--z-shift={_shift_text(replay['z_shift'])}")
    argv.append(f"--collar-tolerance={float(replay.get('collar_tolerance', 5.0))!r}")
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
    "surface_from_raster": _replay_raster,
    "import_mesh": _replay_import,
}
