"""``bakumine geology remove``: take a geology model out of a project.

    bakumine geology remove PROJECT ID

Removes the geology object ``ID`` from the project's registry and deletes its folder
(grid, fields, surfaces, faults, meshes, provenance). Block models coded from it keep
their attributes; ``bakumine geology uncode`` drops those. The command line does not
keep an undo copy: the viewer's "Remove" does (job log, Undo).
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded

HELP = "Take a geology model and its files out of a project"


def configure(parser: argparse.ArgumentParser) -> None:
    parser.description = HELP + ". Block models coded from it keep their attributes."
    parser.add_argument("project", type=Path, help="project folder")
    parser.add_argument("id", help="id of the geology object (see `bakumine project list`)")
    parser.set_defaults(func=_guarded(run))


def run(args: argparse.Namespace) -> int:
    from bakumine.core.project import Project

    project = Project.open(args.project)
    geologies = project.list("geology")
    if not any(o.id == args.id for o in geologies):
        have = ", ".join(o.id for o in geologies) or "none"
        raise CommandError(f"no geology model {args.id!r} in the project (geology models: {have})")
    obj = project.remove(args.id)
    print(f"removed geology {obj.id!r} ({obj.path})")
    return 0


def _replay(e: dict[str, Any]) -> list[dict[str, Any]]:
    """``geology_remove``: ``bakumine geology remove`` on the new project."""
    from bakumine.commands.replay import PROJECT, _run

    return [_run("geology", "remove", PROJECT, obj["id"]) for obj in e.get("removes", [])]


#: job-log operation -> mapper to CLI steps (merged into ``replay._MAPPERS``)
REPLAY: dict[str, Callable[[dict[str, Any]], list[dict[str, Any]]]] = {"geology_remove": _replay}
