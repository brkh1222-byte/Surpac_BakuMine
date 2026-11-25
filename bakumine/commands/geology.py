"""The ``bakumine geology`` command group (research.md 5.7, geology_contracts.md 12).

    bakumine geology units init DB --column lith --out units.json
    bakumine geology units check DB --units units.json

Each subcommand lives in its own module, listed in :data:`SUBCOMMANDS`. A module exports
``HELP`` (one line) and ``configure(parser)``, which adds its arguments and calls
``parser.set_defaults(func=handler)``. A task that adds a subcommand adds its line here only.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from importlib import import_module

HELP = "Geology model: unit dictionary, surfaces, rock units, faults, block coding"

SUBCOMMANDS: list[tuple[str, str]] = [
    ("units", "bakumine.commands.geology_units"),
    ("surface", "bakumine.commands.geology_surface"),
    ("cv", "bakumine.commands.geology_cv"),
]


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="geology_command", metavar="<action>")
    for name, module_name in SUBCOMMANDS:
        module = import_module(module_name)
        module.configure(sub.add_parser(name, help=module.HELP, description=module.HELP))

    def usage(_: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    parser.set_defaults(func=usage)


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "geology": (HELP, configure),
}
