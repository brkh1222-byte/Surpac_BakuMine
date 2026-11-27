"""``bakumine geology units init | check``: the unit dictionary of a deposit.

    bakumine geology units init DB --table geology --column lith --out units.json
    bakumine geology units check DB --units units.json [--density-table density
        --density-column sg] [--allow-unmapped]

``init`` writes a draft dictionary from the codes logged in a column: every code its own
unit of kind ``lithology``, largest first. The user then sets kinds, names, aliases and
densities. ``check`` reads the dictionary against the log and prints what each raw
spelling maps to, the metres per unit, and the three provenance tables (measured,
assumed, missing). It exits 1 when a logged code is not in the dictionary and the
dictionary's policy is ``"unmapped": "error"``, unless ``--allow-unmapped`` is given.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from bakumine.commands.blockmodel import CommandError, _guarded, _wrote
from bakumine.commands.geometry import _database

HELP = "Unit dictionary: draft it from the logged codes (init) and check it against the log"


def configure(parser: argparse.ArgumentParser) -> None:
    sub = parser.add_subparsers(dest="units_command", metavar="<action>")

    init = sub.add_parser(
        "init",
        help="write a draft dictionary from the codes logged in a column",
        description="Write a draft unit dictionary: every logged code (folded) becomes its "
        "own unit of kind lithology, ordered by logged metres. Edit the file afterwards.",
    )
    init.add_argument("database", help="drillhole database folder")
    init.add_argument("--table", default="geology", help="interval table (default: geology)")
    init.add_argument("--column", required=True, help="column with the logged codes")
    init.add_argument("--out", required=True, metavar="JSON", help="dictionary file to write")
    init.add_argument("--name", default="", help="name of the deposit, kept in the file")
    init.add_argument("--overwrite", action="store_true",
                      help="replace --out when it exists (an edited dictionary is not "
                           "replaced without this)")  # fmt: skip
    init.set_defaults(func=_guarded(run_init))

    check = sub.add_parser(
        "check",
        help="check a dictionary against the log; print metres per unit and provenance",
        description="Validate the dictionary, map the logged codes and print: every raw "
        "spelling with its unit, metres per unit, and the measured / assumed / missing "
        "tables. Exit code 1 when a code is unmapped and the dictionary's policy is "
        "'error'.",
    )
    check.add_argument("database", help="drillhole database folder")
    check.add_argument("--units", required=True, metavar="JSON", help="unit dictionary")
    check.add_argument("--table", help="interval table (default: the dictionary's source, "
                                       "else geology)")  # fmt: skip
    check.add_argument("--column", help="code column (default: the dictionary's source)")
    check.add_argument("--density-table", metavar="TABLE",
                       help="interval table with density samples, counted per unit")  # fmt: skip
    check.add_argument("--density-column", metavar="COLUMN", help="its density column (t/m3)")
    check.add_argument(
        "--allow-unmapped",
        action="store_true",
        help="exit 0 although codes are unmapped (they are still listed)",
    )
    check.set_defaults(func=_guarded(run_check))

    def usage(_: argparse.Namespace) -> int:
        parser.print_help()
        return 0

    parser.set_defaults(func=usage)


def _interval_table(database: Any, table: str, column: str | None = None):
    if table not in database.intervals:
        have = ", ".join(sorted(database.intervals)) or "none"
        raise CommandError(f"the database has no interval table {table!r} (tables: {have})")
    frame = database.intervals[table]
    if column is not None and column not in frame.columns:
        have = ", ".join(str(c) for c in frame.columns)
        raise CommandError(f"table {table!r} has no column {column!r} (columns: {have})")
    return frame


def _print_table(headers: list[str], rows: list[list[str]], right: set[str]) -> None:
    cells = [headers, *rows]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    for n, row in enumerate(cells):
        parts = [
            cell.rjust(width) if header in right else cell.ljust(width)
            for cell, width, header in zip(row, widths, headers, strict=True)
        ]
        print("  " + "  ".join(parts).rstrip())
        if n == 0:
            print("  " + "  ".join("-" * width for width in widths))


_NUMERIC = {"intervals", "metres", "holes", "fraction", "n", "mean", "min", "max"}


def _dash(value: object) -> str:
    return "-" if not isinstance(value, str) else value


# ----------------------------------------------------------------------- init


def run_init(args: argparse.Namespace) -> int:
    from bakumine.core.geology.units import UnitDictionary, code_report, draft_dictionary

    out = Path(args.out)
    if out.exists() and not args.overwrite:
        raise CommandError(f"{out} exists; pass --overwrite to replace it")
    table = _interval_table(_database(args.database), args.table, args.column)
    draft = draft_dictionary(table, args.column, table_name=args.table)
    if not draft.units:
        raise CommandError(f"table {args.table!r}, column {args.column!r} holds no codes")
    if args.name:
        from dataclasses import replace

        draft = replace(draft, name=args.name)
    draft.save(out)
    UnitDictionary.load(out)  # the file just written must read back

    report = code_report(table, args.column, draft)
    by_unit = report.groupby("unit")[["intervals", "metres"]].sum()
    spellings = report.groupby("unit")["raw"].apply(lambda raw: ", ".join(repr(r) for r in raw))
    rows = [
        [code, str(int(by_unit.loc[code, "intervals"])), f"{by_unit.loc[code, 'metres']:.2f}",
         spellings[code]]
        for code in draft.codes()
    ]  # fmt: skip
    print(f"table {args.table}, column {args.column}: {len(draft.units)} codes after folding, "
          f"{len(report)} raw spellings")  # fmt: skip
    _print_table(["unit", "intervals", "metres", "logged as"], rows, _NUMERIC)
    _wrote(out, f"draft unit dictionary, {len(draft.units)} units, all of kind lithology")
    print("next: set the kind of each unit (lithology, intrusion, cover, weathering, "
          "fault-zone, other, missing), merge spellings of one unit as aliases, add names, "
          f"densities and weathering zones; then: bakumine geology units check "
          f"{args.database} --units {out}")  # fmt: skip
    return 0


# ----------------------------------------------------------------------- check


def run_check(args: argparse.Namespace) -> int:
    import sys

    from bakumine.core.geology.provenance import Provenance
    from bakumine.core.geology.units import (
        UnitDictionary,
        code_report,
        density_samples_by_unit,
        unit_metres,
        units_provenance,
    )

    units = UnitDictionary.load(args.units)
    table_name = args.table or units.source_table or "geology"
    column = args.column or units.source_column
    if not column:
        raise CommandError("the dictionary names no source column; give --column")
    if bool(args.density_table) != bool(args.density_column):
        raise CommandError("--density-table and --density-column go together")
    database = _database(args.database)
    table = _interval_table(database, table_name, column)

    report = code_report(table, column, units)
    metres = unit_metres(table, column, units)
    coded = int(report["intervals"].sum())
    print(
        f"unit dictionary {args.units}: {units.name or '(no name)'}; {len(units.units)} units, "
        f"{len(units.weathering_zones)} weathering zones; unmapped codes: {units.unmapped}"
    )
    print(
        f"table {table_name}, column {column}: {coded} intervals with a code, "
        f"{float(report['metres'].sum()):.2f} m; {len(table) - coded} intervals without a code"
    )

    print("\nLogged codes (one row per raw spelling)")
    _print_table(
        ["raw", "folded", "unit", "kind", "intervals", "metres", "holes"],
        [
            [repr(r.raw), r.folded, _dash(r.unit) if r.mapped else "NOT MAPPED", _dash(r.kind),
             str(r.intervals), f"{r.metres:.2f}", str(r.holes)]
            for r in report.itertuples(index=False)
        ],
        _NUMERIC,
    )  # fmt: skip

    def print_metres(frame: Any, by: str) -> None:
        print(f"\nMetres by {by}")
        _print_table(
            [by, "kind", "intervals", "metres", "holes", "fraction"],
            [
                [getattr(r, by), r.kind, str(r.intervals), f"{r.metres:.2f}", str(r.holes),
                 f"{r.fraction:.4f}"]
                for r in frame.itertuples(index=False)
            ],
            _NUMERIC,
        )  # fmt: skip

    print_metres(metres, "unit")
    if any(unit.group for unit in units.units):
        print_metres(unit_metres(table, column, units, by="group"), "group")

    not_logged = [code for code in units.codes() if code not in set(metres["unit"])]
    if not_logged:
        print(f"\nUnits of the dictionary with no logged interval: {', '.join(not_logged)}")

    samples = None
    density_source = ""
    if args.density_table:
        density = _interval_table(database, args.density_table, args.density_column)
        samples = density_samples_by_unit(table, column, density, args.density_column, units)
        density_source = f"table {args.density_table}, column {args.density_column}"
        print(f"\nDensity samples by unit ({density_source}; the unit logged at the midpoint "
              "of each sample)")  # fmt: skip
        if samples.empty:
            print("  (no sample falls in a mapped interval)")
        else:
            _print_table(
                ["unit", "n", "mean", "min", "max"],
                [
                    [r.unit, str(r.n), f"{r.mean:.3f}", f"{r.min:.3f}", f"{r.max:.3f}"]
                    for r in samples.itertuples(index=False)
                ],
                _NUMERIC,
            )

    record = Provenance(object_kind="units", name=units.name)
    record.extend(
        units_provenance(
            table,
            column,
            units,
            density_samples=samples,
            density_source=density_source or "density table",
        )  # fmt: skip
    )
    print("\n" + record.format_text())

    unmapped = sorted(set(report.loc[~report["mapped"], "folded"]))
    if unmapped:
        listed = ", ".join(repr(code) for code in unmapped)
        if units.unmapped == "error" and not args.allow_unmapped:
            print(f"error: {len(unmapped)} logged code(s) are not in the dictionary: {listed}. "
                  'Add each as a unit or an alias, set "unmapped": "unknown", or pass '
                  "--allow-unmapped.", file=sys.stderr)  # fmt: skip
            return 1
        print(f"\nwarning: {len(unmapped)} logged code(s) are not in the dictionary and are "
              f"read as unknown: {listed}")  # fmt: skip
    return 0
