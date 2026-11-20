"""`bakumine <command>` entry point.

Each pipeline step registers a subcommand so it can be run headless. A command module
calls :func:`register` with a name and a function that configures its sub-parser; that
function must call ``parser.set_defaults(func=handler)`` where ``handler(args) -> int``.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence

from bakumine import __version__

Configure = Callable[[argparse.ArgumentParser], None]

_COMMANDS: dict[str, tuple[str, Configure]] = {}


def register(name: str, help: str, configure: Configure) -> None:
    """Add a subcommand to the registry."""
    if name in _COMMANDS:
        raise ValueError(f"command already registered: {name}")
    _COMMANDS[name] = (help, configure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bakumine", description="BakuMine Block Modeller")
    parser.add_argument("--version", action="version", version=f"bakumine {__version__}")
    if _COMMANDS:
        subparsers = parser.add_subparsers(dest="command", metavar="<command>")
        for name, (help_text, configure) in sorted(_COMMANDS.items()):
            configure(subparsers.add_parser(name, help=help_text, description=help_text))
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    func = getattr(args, "func", None)
    if func is None:
        parser.print_help()
        return 0
    return int(func(args) or 0)


def _register_builtin() -> None:
    from bakumine.synth import command as synth

    register("synth", synth.HELP, synth.configure)

    from bakumine.commands import drillholes

    for name, (help_text, configure) in drillholes.COMMANDS.items():
        register(name, help_text, configure)

    from bakumine.commands import blockmodel

    for name, (help_text, configure) in blockmodel.COMMANDS.items():
        register(name, help_text, configure)

    from bakumine.commands import geometry, solid

    for module in (geometry, solid):
        for name, (help_text, configure) in module.COMMANDS.items():
            register(name, help_text, configure)

    from bakumine.commands import estimate

    for name, (help_text, configure) in estimate.COMMANDS.items():
        register(name, help_text, configure)

    from bakumine.commands import project

    for name, (help_text, configure) in project.COMMANDS.items():
        register(name, help_text, configure)

    from bakumine.commands import intercepts

    for name, (help_text, configure) in intercepts.COMMANDS.items():
        register(name, help_text, configure)

    from bakumine.commands import classify

    for name, (help_text, configure) in classify.COMMANDS.items():
        register(name, help_text, configure)


_register_builtin()


if __name__ == "__main__":
    raise SystemExit(main())
