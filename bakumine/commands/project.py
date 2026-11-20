"""Project subcommands: ``bakumine project init | add | list``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from bakumine.commands import replay
from bakumine.core.project import KINDS, Project, ProjectError

HELP = "Project folder: init, add objects, list"

Handler = Callable[[argparse.Namespace], int]


def _guarded(handler: Handler) -> Handler:
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except (ProjectError, ValueError, FileNotFoundError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    return run


def _init(args: argparse.Namespace) -> int:
    project = Project.create(args.project, args.name)
    print(f"created project {project.name!r} in {project.folder}")
    return 0


def _add(args: argparse.Namespace) -> int:
    obj = Project.open(args.project).add(args.kind, args.source, args.name, args.id)
    print(f"added {obj.kind} {obj.id!r} as {obj.path}")
    return 0


def _list(args: argparse.Namespace) -> int:
    project = Project.open(args.project)
    print(f"project {project.name!r} ({project.folder})")
    for obj in project.list(args.kind):
        print(f"{obj.kind:<11} {obj.id:<20} {obj.name:<24} {obj.path}")
    return 0


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="project_command", metavar="<action>")
    init = sub.add_parser("init", help="create a project folder")
    init.add_argument("project", type=Path, help="project folder")
    init.add_argument("--name", help="project name (default: the folder name)")
    init.set_defaults(func=_guarded(_init))
    add = sub.add_parser("add", help="copy a file or folder into the project")
    add.add_argument("project", type=Path, help="project folder")
    add.add_argument("kind", choices=KINDS)
    add.add_argument("source", type=Path, help="file or folder to add")
    add.add_argument("--name", help="display name (default: from the source)")
    add.add_argument("--id", help="object id (default: from the name)")
    add.set_defaults(func=_guarded(_add))
    lst = sub.add_parser("list", help="list the project's objects")
    lst.add_argument("project", type=Path, help="project folder")
    lst.add_argument("--kind", choices=KINDS)
    lst.set_defaults(func=_guarded(_list))
    replay.configure(sub, _guarded)  # replay, export-log, export-geometry (P6.T7)
    parser.set_defaults(func=lambda args: parser.print_help() or 0)


COMMANDS = {"project": (HELP, configure)}
