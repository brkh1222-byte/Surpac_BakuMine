"""Surface subcommands: ``bakumine surface from-points | import``.

    bakumine surface from-points --database DB --extend-padding 200 --project P --name Topo
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


def _emit(mesh: Any, kind: str, name: str, args: argparse.Namespace, summary: dict) -> None:
    import numpy as np

    from bakumine.core.io.files import write_mesh
    from bakumine.core.io.obj import write_obj

    if not (args.out or args.geometry_dir or args.project):
        raise CommandError("give --out, --geometry-dir or --project")
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


# ------------------------------------------------------------------ surface import


def configure_import(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("file", help=".dxf, .obj, .ply or Surpac text .dtm (its .str next to it)")
    parser.add_argument("--kind", default="auto", choices=("auto", "dtm", "solid"),
                        help="auto: solid when the mesh is closed, else dtm")  # fmt: skip
    _add_outputs(parser, "mesh")
    parser.set_defaults(func=_guarded(run_import))


def run_import(args: argparse.Namespace) -> int:
    from bakumine.core.geometry.mesh_import import import_mesh

    mesh, info = import_mesh(args.file, args.kind)
    line = f"{info['kind']}: {info['n_vertices']} vertices, {info['n_faces']} triangles"
    if info["kind"] == "solid":
        line += f", {'valid' if info['valid'] else 'NOT VALID'}"
        if info["volume"] is not None:
            line += f", volume {info['volume']:.1f} m3"
    print(line)
    for issue in info.get("issues") or []:
        print(f"  issue: {issue}")
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
    configure_import(
        sub.add_parser(
            "import",
            help="read a surface or solid file (DXF / OBJ / PLY / .dtm)",
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
    return _finish(e, ["surface", "import", _where(replay), "--kind", replay["kind"]])


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {
    "surface_from_points": _replay_surface,
    "import_mesh": _replay_import,
}
