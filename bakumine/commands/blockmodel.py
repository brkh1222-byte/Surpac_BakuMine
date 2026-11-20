"""Block model subcommands: ``bakumine blockmodel info | export-csv | import-csv | report``.

A MODEL is a folder written by :func:`bakumine.core.blockmodel.storage.save_model`
(``model.json`` + ``blocks.parquet``). Heavy imports happen inside the handlers so
that ``bakumine --help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from pathlib import Path

HELP = "Block model: info, CSV export / import, volume-tonnes-grade report"
HELP_INFO = "Show the definition and attributes of a saved block model (reads the header only)"
HELP_EXPORT = "Export blocks to a block model CSV (centroid, size, attributes per block)"
HELP_IMPORT = "Import a block model CSV as a new model, or into an existing one"
HELP_CREATE = "Create an empty block model from origin / size / shape, or fitted to geometry"
HELP_CODE = "Code a block model with assignment steps from a JSON file (optional sub-blocking)"
HELP_REPORT = "Report volume, tonnes, grade and metal, grouped by attribute, bench and cut-off"

Handler = Callable[[argparse.Namespace], int]


class CommandError(Exception):
    """A user-facing error: printed as one line, exit code 2."""


def _guarded(handler: Handler) -> Handler:
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except (CommandError, ValueError, FileNotFoundError, KeyError) as error:
            text = error.args[0] if isinstance(error, KeyError) and error.args else error
            print(f"error: {text}", file=sys.stderr)
            return 2

    return run


def _wrote(path: Path, what: str) -> None:
    print(f"wrote {path} ({what})")


def _names(items: list[str] | None) -> list[str] | None:
    """``-a au,rock -a density`` -> ['au', 'rock', 'density']; None when not given."""
    if items is None:
        return None
    return [name.strip() for item in items for name in item.split(",") if name.strip()]


def _where(expression: str | None):
    if not expression:
        return None
    from bakumine.core.blockmodel.constraints import AttributeConstraint

    return AttributeConstraint(expression)


def _size(n_bytes: int) -> str:
    return f"{n_bytes / 1e6:.1f} MB" if n_bytes >= 1e5 else f"{n_bytes} bytes"


# ----------------------------------------------------------------------- info


def configure_info(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="block model folder")
    parser.set_defaults(func=_guarded(run_info))


def run_info(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import read_header

    header = read_header(args.model)
    d = header.definition
    print(f"block model {header.path} (format version {header.version})")
    print(f"origin (x, y, z):      {d.origin[0]:g}, {d.origin[1]:g}, {d.origin[2]:g}")
    print(f"block size (x, y, z):  {d.block_size[0]:g}, {d.block_size[1]:g}, {d.block_size[2]:g}")
    print(f"blocks (nx, ny, nz):   {d.nx}, {d.ny}, {d.nz}")
    print(f"extent (x, y, z):      {d.extent[0]:g}, {d.extent[1]:g}, {d.extent[2]:g}")
    if d.matrix is not None:
        print("rotation:              matrix " + json.dumps([list(row) for row in d.matrix]))
    elif d.is_rotated:
        print(f"rotation:              {d.angles} ({d.convention})")
    else:
        print("rotation:              none")
    smallest = ", ".join(f"{s:g}" for s in d.min_block_size)
    print(f"sub-block depth:       {d.max_depth} (smallest block {smallest})")
    print(f"parent blocks:         {d.n_parents}")
    print(f"blocks:                {header.n_blocks}")
    print(f"blocks.parquet:        {_size(header.blocks_bytes)} ({header.compression})")
    print(f"checksum (sha256):     {header.checksum[:16]}...")
    print(f"attributes:            {len(header.attributes)}")
    for spec in header.attributes:
        background = "missing" if spec.background is None else spec.background
        text = f"  {spec.name:<20} {spec.kind:<9} {spec.dtype!s:<8} background {background}"
        if spec.unit:
            text += f"  [{spec.unit}]"
        if spec.kind == "category":
            shown = ", ".join(spec.categories[:8]) + (", ..." if len(spec.categories) > 8 else "")
            text += f"  {len(spec.categories)} categories: {shown}"
        print(text)
    return 0


# ----------------------------------------------------------------- export-csv


def configure_export(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="block model folder")
    parser.add_argument("out", help="CSV file to write")
    parser.add_argument("--layout", default="surpac_like",
                        help="surpac_like (default; Y, X, Z, PROVISIONAL) or xyz")  # fmt: skip
    parser.add_argument("--attributes", "-a", action="append", metavar="NAMES",
                        help="attributes to write, comma-separated (default all)")  # fmt: skip
    parser.add_argument("--where", metavar="EXPR",
                        help="only blocks for which EXPR is true, e.g. 'au > 0.5'")  # fmt: skip
    parser.add_argument("--decimals", type=int, help="decimals of float attributes (default exact)")
    parser.add_argument("--coord-decimals", type=int,
                        help="decimals of centroids and sizes (default exact)")  # fmt: skip
    parser.add_argument("--missing", default="",
                        help="token written for missing values (default: empty field)")  # fmt: skip
    parser.set_defaults(func=_guarded(run_export))


def run_export(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import load_model
    from bakumine.core.io.block_csv import export_block_csv

    model = load_model(args.model)
    result = export_block_csv(
        model,
        args.out,
        layout=args.layout,
        constraint=_where(args.where),
        attributes=_names(args.attributes),
        decimals=args.decimals,
        coord_decimals=args.coord_decimals,
        missing=args.missing,
    )
    _wrote(result.path, f"{result.n_rows} of {model.n_blocks} blocks, layout {result.layout}")
    print("columns: " + ", ".join(result.columns))
    return 0


# ----------------------------------------------------------------- import-csv


def configure_import(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("csv", help="block model CSV")
    parser.add_argument("out", help="block model folder to write")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--definition", metavar="JSON",
                        help="definition of the new model: a JSON file (a definition, or the "
                             "model.json of a saved model); needed if rotated")  # fmt: skip
    source.add_argument("--infer", action="store_true",
                        help="infer the definition from the rows (unrotated; default)")  # fmt: skip
    source.add_argument("--into", metavar="MODEL",
                        help="import into this existing model (it is saved as OUT)")  # fmt: skip
    parser.add_argument("--layout", default=None,
                        help="column order to assume when the file has no column names: "
                             "surpac_like (Y, X, Z; default) or xyz")  # fmt: skip
    parser.add_argument("--attributes", "-a", action="append", metavar="NAMES",
                        help="attribute columns to import, comma-separated")  # fmt: skip
    parser.add_argument("--missing", action="append", metavar="TOKEN",
                        help="value that means missing, e.g. -99 (repeatable; empty fields "
                             "and nan always are)")  # fmt: skip
    parser.add_argument("--tolerance", type=float, default=0.01,
                        help="allowed misfit of sizes and centroids, as a fraction of the "
                             "block size (default 0.01)")  # fmt: skip
    parser.set_defaults(func=_guarded(run_import))


def _definition(path: str):
    from bakumine.core.blockmodel.definition import BlockModelDefinition

    file = Path(path)
    if file.is_dir():
        file = file / "model.json"
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
        return BlockModelDefinition.from_dict(data.get("definition", data))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError) as error:
        raise CommandError(f"{file} is not a block model definition ({error!r})") from None


def run_import(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.io.block_csv import DEFAULT_MISSING, import_block_csv

    model = load_model(args.into) if args.into else None
    definition = _definition(args.definition) if args.definition else None
    model, report = import_block_csv(
        args.csv,
        model,
        definition=definition,
        layout=args.layout,
        attributes=_names(args.attributes),
        missing=(*DEFAULT_MISSING, *(args.missing or ())),
        tolerance=args.tolerance,
    )
    print(report.summary())
    header = save_model(model, args.out)
    _wrote(header.path, f"{header.n_blocks} blocks, {len(header.attributes)} attributes")
    return 0


# --------------------------------------------------------------------- create


def _triple(text: str | None, what: str) -> tuple[float, ...] | None:
    if text is None:
        return None
    try:
        values = tuple(float(v) for v in text.replace(" ", "").split(",") if v)
    except ValueError:
        values = ()
    if len(values) != 3:
        raise CommandError(f"{what} expects three numbers like 10,10,5, got {text!r}")
    return values


def configure_create(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("out", help="block model folder to write")
    parser.add_argument("--size", required=True, metavar="DX,DY,DZ", help="parent block size")
    parser.add_argument("--origin", metavar="X,Y,Z", help="minimum corner (with --shape/--extent)")
    parser.add_argument("--shape", metavar="NX,NY,NZ", help="number of parent blocks")
    parser.add_argument("--extent", metavar="EX,EY,EZ", help="model extent (whole blocks)")
    parser.add_argument("--extent-from", action="append", metavar="NAME",
                        help="fit the model to a solid or DTM in --geometry-dir, or to the "
                             "holes of a drillhole database as 'drillholes:FOLDER' "
                             "(collars down to max depth); repeatable")  # fmt: skip
    parser.add_argument("--geometry-dir", metavar="DIR", help="folder with named solids / DTMs")
    parser.add_argument("--padding", default="0", metavar="P|PX,PY,PZ",
                        help="margin added on every side with --extent-from (metres)")  # fmt: skip
    parser.add_argument("--max-depth", type=int, default=0,
                        help="sub-blocking depth (smallest block = size / 2**depth)")  # fmt: skip
    parser.add_argument("--angles", metavar="BEARING,PLUNGE,DIP",
                        help="model rotation in degrees (default none)")  # fmt: skip
    parser.set_defaults(func=_guarded(run_create))


def _extent_points(items: list[str], geometry_dir: str | None):
    import numpy as np

    from bakumine.core.blockmodel.constraints import FileResolver

    resolver = FileResolver(geometry_dir) if geometry_dir else None
    clouds = []
    for item in items:
        if item.startswith("drillholes:"):
            from bakumine.core.drillholes.importer import load_database

            collar = load_database(item.split(":", 1)[1]).collar
            top = collar[["x", "y", "z"]].to_numpy(dtype=float)
            bottom = top.copy()
            bottom[:, 2] -= collar["max_depth"].to_numpy(dtype=float)
            clouds += [top, bottom]
            continue
        if resolver is None:
            raise CommandError("--extent-from NAME needs --geometry-dir")
        clouds.append(np.asarray(resolver.solid(item).vertices))
    return np.concatenate(clouds)


def run_create(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.blocks import BlockModel
    from bakumine.core.blockmodel.definition import BlockModelDefinition
    from bakumine.core.blockmodel.storage import save_model

    size = _triple(args.size, "--size")
    options: dict = {"max_depth": args.max_depth}
    angles = _triple(args.angles, "--angles")
    if angles:
        options["angles"] = angles
    if args.extent_from:
        padding = args.padding if "," in args.padding else ",".join([args.padding] * 3)
        points = _extent_points(args.extent_from, args.geometry_dir)
        definition = BlockModelDefinition.auto_extent(
            points, size, margin=_triple(padding, "--padding"), **options
        )
    else:
        origin = _triple(args.origin, "--origin")
        if origin is None or (args.shape is None) == (args.extent is None):
            raise CommandError("give --origin and exactly one of --shape / --extent, "
                               "or --extent-from")  # fmt: skip
        if args.shape:
            shape = tuple(int(v) for v in _triple(args.shape, "--shape"))
            definition = BlockModelDefinition(origin, size, shape, **options)
        else:
            definition = BlockModelDefinition.from_extent(
                origin, _triple(args.extent, "--extent"), size, **options
            )
    header = save_model(BlockModel(definition), args.out)
    d = definition
    print(f"origin {d.origin}, size {d.block_size}, blocks {d.nx} x {d.ny} x {d.nz}")
    _wrote(header.path, f"{header.n_blocks} blocks")
    return 0


# ------------------------------------------------------------------------ code


def configure_code(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="block model folder")
    parser.add_argument("--steps", required=True, metavar="JSON",
                        help='coding file: {"attributes": [{"name", "kind", "background", ...}], '
                             '"steps": [{"attribute", "value", "constraint": {...}, '
                             '"subblock": bool, "max_level": int}]}')  # fmt: skip
    parser.add_argument("--geometry-dir", metavar="DIR",
                        help="folder with the solids / DTMs the constraints name")  # fmt: skip
    parser.add_argument("--subblock", action="store_true",
                        help="sub-block every step (unless it sets subblock itself)")  # fmt: skip
    parser.add_argument("--merge", action="store_true",
                        help="merge uniform sibling sub-blocks after the last step")  # fmt: skip
    parser.add_argument("--out", help="model folder to write (default: update MODEL in place)")
    parser.set_defaults(func=_guarded(run_code))


def run_code(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.coding import code_model
    from bakumine.core.blockmodel.constraints import FileResolver, constraint_from_dict
    from bakumine.core.blockmodel.storage import load_model, save_model

    path = Path(args.steps)
    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise CommandError(f"{path} is not valid JSON ({error})") from None
    model = load_model(args.model)
    for item in spec.get("attributes", []):
        item = dict(item)
        if item.get("name") in model.attributes:
            continue
        model.add_attribute(item.pop("name"), item.pop("kind", "float"), **item)
    steps = []
    for item in spec.get("steps", []):
        item = dict(item)
        if item.get("constraint") is not None:
            item["constraint"] = constraint_from_dict(item["constraint"])
        if args.subblock:
            item.setdefault("subblock", True)
        steps.append(item)
    resolver = FileResolver(args.geometry_dir) if args.geometry_dir else None
    results = code_model(model, steps, resolver, merge=args.merge)
    for r in results:
        print(f"{r.attribute} = {r.value!r}: {r.n_assigned} blocks, volume {r.volume:.0f} m3, "
              f"split {r.n_split}, blocks now {r.n_blocks}")  # fmt: skip
    header = save_model(model, args.out or args.model)
    _wrote(header.path, f"{header.n_blocks} blocks, {len(header.attributes)} attributes")
    return 0


# --------------------------------------------------------------------- report


def configure_report(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("model", help="block model folder")
    parser.add_argument("--where", metavar="EXPR",
                        help="only blocks for which EXPR is true, e.g. 'z < 1000'")  # fmt: skip
    parser.add_argument("--group-by", "-g", action="append", metavar="NAMES",
                        help="category / integer attributes to group by")  # fmt: skip
    parser.add_argument("--cutoffs", metavar="VALUES",
                        help="grade cut-offs, comma-separated (grade >= cut-off)")  # fmt: skip
    parser.add_argument("--cutoff-attribute", metavar="NAME",
                        help="grade the cut-offs apply to (default: first --grade)")  # fmt: skip
    parser.add_argument("--grade", action="append", metavar="NAME[:UNIT]",
                        help="grade attribute(s), comma-separated; UNIT g/t, ppm or %% sets "
                             "the metal columns (default: the attribute's unit)")  # fmt: skip
    parser.add_argument("--density", metavar="NAME|VALUE",
                        help="density attribute, or a constant in t/m3")  # fmt: skip
    parser.add_argument("--partial", metavar="NAME",
                        help="attribute with the fraction (0..1) of each block")  # fmt: skip
    parser.add_argument("--bench-height", type=float, metavar="H",
                        help="group by benches of this height (by block centroid)")  # fmt: skip
    parser.add_argument("--bench-reference", type=float, default=0.0, metavar="Z",
                        help="elevation of a bench floor (default 0)")  # fmt: skip
    parser.add_argument("--out", metavar="CSV", help="also write the report to this CSV file")
    parser.set_defaults(func=_guarded(run_report))


def run_report(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import load_model
    from bakumine.core.report import format_table, report, to_csv

    grades: list[str] = []
    units: dict[str, str] = {}
    for item in _names(args.grade) or []:
        name, sep, unit = item.partition(":")
        grades.append(name)
        if sep:
            units[name] = unit
    density: str | float | None = args.density
    if density is not None:
        try:
            density = float(density)
        except ValueError:
            pass
    cutoffs = None
    if args.cutoffs:
        try:
            cutoffs = [float(v) for v in args.cutoffs.split(",") if v.strip()]
        except ValueError:
            raise CommandError(f"--cutoffs expects numbers, got {args.cutoffs!r}") from None
    model = load_model(args.model)
    table = report(
        model,
        constraint=_where(args.where),
        group_by=_names(args.group_by) or (),
        cutoffs=cutoffs,
        cutoff_attribute=args.cutoff_attribute,
        grade_attributes=grades,
        density=density,
        partial=args.partial,
        bench_height=args.bench_height,
        bench_reference=args.bench_reference,
        units=units,
    )
    print(format_table(table))
    for note in table.attrs["notes"]:
        print(f"note: {note}")
    if args.out:
        _wrote(to_csv(table, args.out), f"{len(table)} rows")
    return 0


# ------------------------------------------------------------------- registry

SUBCOMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "info": (HELP_INFO, configure_info),
    "export-csv": (HELP_EXPORT, configure_export),
    "import-csv": (HELP_IMPORT, configure_import),
    "create": (HELP_CREATE, configure_create),
    "code": (HELP_CODE, configure_code),
    "report": (HELP_REPORT, configure_report),
}


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="blockmodel_command", metavar="<action>")
    for name, (help_text, configure_one) in SUBCOMMANDS.items():
        configure_one(sub.add_parser(name, help=help_text, description=help_text))
    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "blockmodel": (HELP, configure),
}
