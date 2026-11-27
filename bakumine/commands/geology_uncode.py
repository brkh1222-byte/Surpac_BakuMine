"""``bakumine geology uncode``: take the geology coding off a block model.

    bakumine geology uncode MODEL [--attributes NAMES] [--merge | --no-merge]
        [--geology FOLDER ...]

Drops the attributes ``bakumine geology code`` wrote to the model: the ones its record
(``<model folder>.geology-coding.json``, beside the model folder) lists as created by the
coding. An attribute the model already had before the coding (a density of your own) is
kept and named; it still holds the coded values. ``--attributes`` names the attributes to
drop instead, for a model coded before records were kept. Sibling sub-blocks that are
equal in all remaining attributes are merged afterwards, which takes back the sub-blocking
at the geology boundaries (``--no-merge`` leaves the blocks as they are).
``--geology FOLDER`` also deletes that geology model's record of the coding
(``FOLDER/coding/<model name>.provenance.json``).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote

HELP = "Take the geology coding off a block model: drop the attributes the coding wrote"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = (
        HELP + ". An attribute the model had before the coding is kept. Sibling sub-blocks "
        "equal in all remaining attributes are merged afterwards."
    )
    parser.add_argument("model", help="block model folder")
    parser.add_argument("--attributes", metavar="NAMES",
                        help="drop these attributes (comma separated) instead of the ones "
                             "the coding record lists")  # fmt: skip
    merge = parser.add_mutually_exclusive_group()
    merge.add_argument("--merge", dest="merge", action="store_true", default=None,
                       help="merge sibling sub-blocks equal in all remaining attributes "
                            "(default)")  # fmt: skip
    merge.add_argument("--no-merge", dest="merge", action="store_false",
                       help="leave the block structure as it is")  # fmt: skip
    parser.add_argument("--geology", action="append", default=[], metavar="FOLDER",
                        help="geology model folder whose record of this coding is deleted "
                             "too (repeatable)")  # fmt: skip
    parser.set_defaults(func=_guarded(run))


def run(args: argparse.Namespace) -> int:
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.geology.coding import (
        coding_provenance_path,
        coding_record_path,
        read_coding_record,
        remove_coding,
        save_coding_record,
    )

    target = Path(args.model)
    names = None
    if args.attributes is not None:
        names = [n.strip() for n in args.attributes.split(",") if n.strip()]
        if not names:
            raise CommandError("--attributes names no attribute")
    record = read_coding_record(target)
    if record is None and names is None:
        raise CommandError(
            f"{target} has no geology coding to remove (no {coding_record_path(target).name} "
            "beside it); name the attributes to drop with --attributes"
        )
    model = load_model(target)
    try:
        removal = remove_coding(model, record, attributes=names, merge=args.merge)
    except ValueError as exc:
        raise CommandError(f"{target}: {exc}") from None
    header = save_model(model, target)
    save_coding_record(target, removal.record)
    print(
        f"removed {', '.join(removal.removed)}: {removal.n_merged} sibling sets merged, "
        f"{header.n_blocks} blocks"
    )
    for name, reason in removal.kept.items():
        print(f"kept {name}: {reason}")
    _wrote(header.path, f"{header.n_blocks} blocks, {len(header.attributes)} attributes")
    if removal.record is None:
        for folder in args.geology:
            file = coding_provenance_path(folder, target.resolve().name)
            if file.is_file():
                file.unlink()
                print(f"deleted {file}")
    return 0


def _replay(e: dict[str, Any]) -> list[dict[str, Any]]:
    """``geology_uncode``: drop exactly the attributes the job dropped."""
    from bakumine.commands.replay import ReplayError, _in_project, _run

    replay = e.get("replay") or {}
    if not replay.get("attributes"):
        raise ReplayError(f"job {e['id']}: the attributes dropped were not logged")
    model = next(o for o in e["inputs"] if o["kind"] == "model")
    argv = ["geology", "uncode", _in_project(model)]
    argv += ["--attributes", ",".join(replay["attributes"])]
    argv.append("--merge" if replay.get("merge") else "--no-merge")
    for obj in e["inputs"]:
        if obj["kind"] == "geology":
            argv += ["--geology", _in_project(obj)]
    return [_run(*argv)]


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {"geology_uncode": _replay}
