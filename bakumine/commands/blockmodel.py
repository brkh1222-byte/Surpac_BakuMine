"""Block model subcommands: ``bakumine blockmodel info | export-csv | import-csv``.

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

HELP = "Block model: info, CSV export / import"
HELP_INFO = "Show the definition and attributes of a saved block model (reads the header only)"
HELP_EXPORT = "Export blocks to a block model CSV (centroid, size, attributes per block)"
HELP_IMPORT = "Import a block model CSV as a new model, or into an existing one"

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


# ------------------------------------------------------------------- registry

SUBCOMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "info": (HELP_INFO, configure_info),
    "export-csv": (HELP_EXPORT, configure_export),
    "import-csv": (HELP_IMPORT, configure_import),
}


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="blockmodel_command", metavar="<action>")
    for name, (help_text, configure_one) in SUBCOMMANDS.items():
        configure_one(sub.add_parser(name, help=help_text, description=help_text))
    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "blockmodel": (HELP, configure),
}
