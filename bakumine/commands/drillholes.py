"""Drillhole subcommands: ``bakumine import``, ``validate``, ``desurvey``, ``composite``.

Each command reads or writes a database folder (``DrillholeDatabase.save``), prints every
file it wrote and a short summary. The heavy imports happen inside the handlers so that
``bakumine --help`` stays fast.
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections.abc import Callable
from pathlib import Path

HELP_IMPORT = "Import collar/survey/interval tables (CSV/XLSX) into a drillhole database folder"
HELP_VALIDATE = "Check a drillhole database and write the findings"
HELP_DESURVEY = "Desurvey a drillhole database: station table and sample midpoints"
HELP_COMPOSITE = "Composite an interval table (downhole, bench or within domains)"

Handler = Callable[[argparse.Namespace], int]


class CommandError(Exception):
    """A user-facing error: printed as one line, exit code 2."""


def _guarded(handler: Handler) -> Handler:
    def run(args: argparse.Namespace) -> int:
        try:
            return handler(args)
        except (CommandError, ValueError, FileNotFoundError) as error:
            print(f"error: {error}", file=sys.stderr)
            return 2

    return run


def _wrote(path: Path, what: str) -> None:
    print(f"wrote {path} ({what})")


def _load(folder: str):
    from bakumine.core.drillholes.importer import load_database

    path = Path(folder)
    if not (path / "database.json").is_file():
        raise CommandError(f"{path} is not a drillhole database folder (no database.json)")
    return load_database(path)


def _write_table(table, base: Path, formats: tuple[str, ...], what: str) -> None:
    base.parent.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        path = base.with_name(f"{base.name}.{fmt}")
        if fmt == "parquet":
            table.to_parquet(path, index=False)
        else:
            table.to_csv(path, index=False)
        _wrote(path, what)


def _formats(choice: str) -> tuple[str, ...]:
    return ("parquet", "csv") if choice == "both" else (choice,)


def _pairs(items: list[str] | None, option: str) -> dict[str, str]:
    """``NAME=VALUE`` options -> dict."""
    out: dict[str, str] = {}
    for item in items or []:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise CommandError(f"{option} expects NAME=VALUE, got {item!r}")
        if name in out:
            raise CommandError(f"{option} names {name!r} twice")
        out[name] = value
    return out


# --------------------------------------------------------------------------------------
# import
# --------------------------------------------------------------------------------------
def configure_import(parser: argparse.ArgumentParser) -> None:
    from bakumine.core.drillholes.importer import (
        ABOVE_DETECTION_RULES,
        BELOW_DETECTION_RULES,
        LENGTH_UNITS,
    )

    files = parser.add_argument_group(
        "source files",
        "CSV, XLSX or XLS. A table option given more than once merges the files into one "
        "table (the first file wins where they disagree; conflicts are reported). What "
        "differs from file to file is said with --source-option.",
    )
    files.add_argument("--collar", action="append", help="collar table")
    files.add_argument(
        "--survey",
        action="append",
        help="downhole survey table (without it: one station per hole from the collar "
        "table's azimuth and dip columns)",
    )
    files.add_argument("--assay", action="append", help="assay interval table ('assay')")
    files.add_argument("--geology", action="append", help="geology interval table ('geology')")
    files.add_argument(
        "--interval",
        action="append",
        metavar="NAME=FILE",
        help="any other interval table, stored under NAME (repeatable)",
    )
    files.add_argument(
        "--source-option",
        action="append",
        metavar="FILE:KEY=VALUE",
        help="how ONE source file is read, where it differs from the other files of its "
        "table (repeatable; one KEY=VALUE each). FILE is the path as given to --collar, "
        "--assay ... or just its file name. Keys: " + SOURCE_OPTION_HELP + ". Example: "
        "--collar drilling.csv --collar gis.xlsx --source-option gis.xlsx:sheet=Summary "
        "--source-option gis.xlsx:header-row=2",
    )
    files.add_argument(
        "--recipe",
        help="import recipe (JSON, as saved in <database>/import_recipe.json) instead of "
        "the source file options: replays an import, e.g. on updated files",
    )
    files.add_argument(
        "--base", help="folder the recipe's relative file paths start from (default: its own)"
    )
    parser.add_argument("--out", help="database folder to write")
    parser.add_argument(
        "--save-recipe", metavar="FILE", help="also write the recipe (with the column maps) here"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="read and check everything, print the summary, write no database",
    )
    parser.add_argument(
        "--map",
        dest="map_file",
        help="JSON or TOML file: {table: {source column: canonical name}} with tables "
        "'collar', 'survey' and each interval table name",
    )
    parser.add_argument(
        "--auto",
        action="store_true",
        help="propose the column map from the headers for every table the map file omits",
    )
    shape = parser.add_argument_group("layout of the files (TABLE = collar, survey, assay ...)")
    shape.add_argument(
        "--sheet",
        action="append",
        metavar="TABLE=SHEETS",
        help="workbook sheets of a table: names, 0-based indexes or wildcards, comma "
        "separated ('assay=DH*' stacks one sheet per hole; default: the first sheet). "
        "Applies to every workbook of the table (CSV files have no sheets); for one file: "
        "--source-option FILE:sheet=...",
    )
    shape.add_argument(
        "--header-row",
        action="append",
        metavar="TABLE=N",
        help="spreadsheet row of the header (1 = first, 0 = no header; default: detected)",
    )
    shape.add_argument(
        "--header-rows",
        action="append",
        metavar="TABLE=N",
        help="1, or 2 for a group header over sub-headers (default: detected)",
    )
    shape.add_argument(
        "--hole-from-sheet",
        action="append",
        default=[],
        metavar="TABLE",
        help="take the hole id of this table from the sheet name",
    )
    shape.add_argument(
        "--hole-id-pattern",
        metavar="REGEX",
        help="rewrite hole ids matching this pattern, in every table, as --hole-id-template",
    )
    shape.add_argument(
        "--hole-id-template",
        metavar="TEMPLATE",
        help="e.g. 'DH-{1:03d}' ({1} = first group of the pattern, {0} = the whole id)",
    )
    shape.add_argument(
        "--include-holes",
        action="append",
        metavar="HOLES",
        help="import only these holes, in every table: names or wildcard patterns "
        "separated by commas ('DH-01,DH-02', 'DH25*'), or a text file with one per line "
        "('#' starts a comment). Compared without regard to case, after "
        "--hole-id-pattern rewrote the ids. Repeatable",
    )
    shape.add_argument(
        "--exclude-holes",
        action="append",
        metavar="HOLES",
        help="leave these holes out (same forms as --include-holes; exclusion wins). Both "
        "are stored in the recipe; given with --recipe they replace the recipe's filter",
    )
    shape.add_argument(
        "--sample-depth",
        action="append",
        default=[],
        metavar="TABLE",
        help="where from/to are empty, read them from the sample number: to = the number "
        "left after the hole id, from = to - sample length (or an explicit 'a-b')",
    )
    shape.add_argument(
        "--sample-length", type=float, default=1.0, help="sample length for --sample-depth"
    )
    shape.add_argument(
        "--sample-pattern",
        action="append",
        metavar="REGEX",
        help="replace the default --sample-depth rule: regex with (?P<to>...) and optional "
        "(?P<from>...); {hole} and {holenum} stand for the row's hole (repeatable)",
    )
    shape.add_argument(
        "--qc",
        choices=("separate", "off"),
        default="separate",
        help="blank / standard / duplicate rows of interval tables: set aside in a QC "
        "table (default) or left among the samples",
    )
    shape.add_argument(
        "--merge-precedence",
        choices=("first", "last"),
        default="first",
        help="which file wins when several files of one table disagree (default: first)",
    )
    shape.add_argument(
        "--merge-tolerance", type=float, default=0.01, help="numbers closer than this agree"
    )
    shape.add_argument(
        "--merge-ignore",
        action="append",
        metavar="COLUMNS",
        help="columns (canonical names, comma separated; repeatable) whose differences "
        "between files are informational, not conflicts: they are counted apart and listed "
        "in conflicts_<table>.csv only. Default: sample_id; 'none' checks every column",
    )
    parser.add_argument(
        "--length-unit",
        choices=tuple(LENGTH_UNITS),
        default="m",
        help="unit of coordinates and depths in the files (default: m)",
    )
    parser.add_argument(
        "--to-metres", action="store_true", help="convert coordinates and depths to metres"
    )
    parser.add_argument(
        "--dip-convention",
        choices=("negative_down", "positive_down", "auto"),
        default="negative_down",
        help="sign of the dips in the files; auto = positive_down when no dip is negative "
        "(default: negative_down)",
    )
    parser.add_argument(
        "--normalise-dip", action="store_true", help="store dips as negative_down"
    )
    parser.add_argument(
        "--coordinate-order",
        default="xyz",
        help="true axes of the columns mapped to x, y, z: 'xyz' (default), or 'yxz' when the "
        "column mapped to x holds the northing (X of CIS / Mongolian reports). 'auto' swaps "
        "x and y when the values say so (agreement with longitude / latitude; else, for "
        "metres, x within 1e6..1e7 and y within 1e5..1e6 in every row: UTM) and reports it. "
        "With 'xyz' such a table only gets a warning",
    )
    parser.add_argument(
        "--crs",
        help="projected CRS for collars given as longitude / latitude, e.g. EPSG:32647 "
        "(default: the UTM zone of the data)",
    )
    parser.add_argument(
        "--coordinate-tolerance",
        type=float,
        default=30.0,
        help="report collars whose projected columns differ from their longitude / "
        "latitude by more than this (default: 30)",
    )
    parser.add_argument(
        "--below-detection",
        choices=BELOW_DETECTION_RULES,
        default="half",
        help="rule for '<limit' cells (default: half the limit)",
    )
    parser.add_argument(
        "--above-detection",
        choices=ABOVE_DETECTION_RULES,
        default="limit",
        help="rule for '>limit' cells (default: the limit)",
    )
    parser.add_argument(
        "--text-column", action="append", default=[], help="attribute forced to text"
    )
    parser.add_argument(
        "--numeric-column", action="append", default=[], help="attribute forced to numeric"
    )
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    parser.set_defaults(func=_guarded(run_import))


def _read_map_file(path: Path, flat: bool = False) -> dict:
    """``{table: {source column: canonical name}}`` of a --map file; ``flat``: the column
    map of one source, ``{source column: canonical name}``."""
    if not path.is_file():
        raise CommandError(f"column map file not found: {path}")
    if path.suffix.lower() == ".toml":
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise CommandError(f"column map file must be .json or .toml, got {path.name}")
    if flat:
        if not isinstance(document, dict) or any(isinstance(v, dict) for v in document.values()):
            raise CommandError(f"{path}: expected {{source column: canonical name}}")
        return {str(k): str(v) for k, v in document.items()}
    if not isinstance(document, dict) or not all(isinstance(v, dict) for v in document.values()):
        raise CommandError(f"{path}: expected {{table: {{source column: canonical name}}}}")
    return {str(t): {str(k): str(v) for k, v in m.items()} for t, m in document.items()}


#: --source-option keys that say how a file is laid out -> the key of a recipe source
_SOURCE_LAYOUT_KEYS = {
    "sheet": "sheets",
    "sheets": "sheets",
    "exclude-sheets": "exclude_sheets",
    "header-row": "header_row",
    "header-rows": "header_rows",
    "hole-from-sheet": "hole_id_from_sheet",
    "map": "map",
}
#: --source-option keys that are import options of one file -> the ImportOptions field
_SOURCE_VALUE_KEYS: dict[str, str] = {
    "dip-convention": "dip_convention",
    "length-unit": "length_unit",
    "coordinate-order": "coordinate_order",
    "crs": "crs",
    "geographic-crs": "geographic_crs",
}
SOURCE_OPTION_HELP = (
    "sheet (names, indexes or wildcards, comma separated), exclude-sheets, header-row, "
    "header-rows, hole-from-sheet (yes / no), map (a JSON or TOML file: {source column: "
    "canonical name}); and, as the options of the same name but for this file only, "
    "dip-convention, length-unit, coordinate-order, crs (the CRS of this file's "
    "coordinates; files in another CRS than --crs are reprojected), geographic-crs. "
    "Files of one table with different dip conventions are all stored negative_down, "
    "files with different length units are all converted to metres"
)


def _truth(text: str, what: str) -> bool:
    if text.strip().casefold() in ("1", "yes", "y", "true", "on"):
        return True
    if text.strip().casefold() in ("0", "no", "n", "false", "off"):
        return False
    raise CommandError(f"{what} expects yes or no, got {text!r}")


def _source_options(
    items: list[str] | None, files: dict[str, list[Path]]
) -> dict[tuple[str, int], dict]:
    """``--source-option FILE:KEY=VALUE`` -> ``{(table, index of the file): source keys}``."""
    import re

    keys = {**_SOURCE_LAYOUT_KEYS, **_SOURCE_VALUE_KEYS}
    names = "|".join(map(re.escape, keys))
    pattern = re.compile(rf"^(?P<file>.+?):(?P<key>{names})=(?P<value>.*)$")
    given = [(table, i, path) for table, paths in files.items() for i, path in enumerate(paths)]
    out: dict[tuple[str, int], dict] = {}
    for item in items or []:
        match = pattern.match(item)
        if match is None or not match.group("value").strip():
            raise CommandError(
                f"--source-option expects FILE:KEY=VALUE with KEY one of {sorted(keys)}, "
                f"got {item!r}"
            )
        name, key, value = match.group("file"), match.group("key"), match.group("value").strip()
        found = [(t, i) for t, i, p in given if str(p) == name or p == Path(name)]
        found = found or [(t, i) for t, i, p in given if p.name == name]
        if not found:
            raise CommandError(
                f"--source-option names {name!r}, which is not one of the source files "
                f"({', '.join(str(p) for _, _, p in given)})"
            )
        for where in found:  # one file may feed two tables: the option is the file's
            source = out.setdefault(where, {})
            if key in _SOURCE_VALUE_KEYS:
                field = _SOURCE_VALUE_KEYS[key]
                if field in source.setdefault("options", {}):
                    raise CommandError(f"--source-option gives {name}:{key} twice")
                source["options"][field] = value
                continue
            field = _SOURCE_LAYOUT_KEYS[key]
            if field in source:
                raise CommandError(f"--source-option gives {name}:{key} twice")
            if field in ("sheets", "exclude_sheets"):
                source[field] = _sheet_list(value)
                if field == "exclude_sheets":
                    source[field] = [str(s) for s in source[field]]
            elif field in ("header_row", "header_rows"):
                if not value.isdigit():
                    raise CommandError(f"--source-option {name}:{key} expects a number")
                source[field] = int(value)
            elif field == "hole_id_from_sheet":
                source[field] = _truth(value, f"--source-option {name}:{key}")
            else:  # map: a file with one table's column map
                document = _read_map_file(Path(value), flat=True)
                source[field] = document
    return out


def _hole_list(items: list[str] | None, option: str) -> list[str]:
    """``--include-holes`` / ``--exclude-holes`` values -> names and patterns. A value
    that is an existing file is read: one hole per line."""
    out: list[str] = []
    for item in items or []:
        path = Path(item)
        if path.is_file():
            lines = path.read_text(encoding="utf-8-sig").splitlines()
            found = [line.split("#", 1)[0].strip() for line in lines]
        else:
            found = [part.strip() for part in item.split(",")]
        found = [f for f in found if f]
        if not found:
            raise CommandError(f"{option} {item!r} names no hole")
        out += [f for f in found if f not in out]
    return out


def _merge_ignore(items: list[str] | None) -> list[str] | None:
    """``--merge-ignore`` values -> column names; None = the default (the sample id)."""
    if not items:
        return None
    names = [part.strip() for item in items for part in item.split(",") if part.strip()]
    return [] if "none" in names else names


def _hole_filter(args: argparse.Namespace) -> dict[str, list[str]] | None:
    chosen = {
        "include": _hole_list(args.include_holes, "--include-holes"),
        "exclude": _hole_list(args.exclude_holes, "--exclude-holes"),
    }
    return chosen if chosen["include"] or chosen["exclude"] else None


def _sheet_list(text: str) -> list[str | int]:
    return [int(s) if s.isdigit() else s for s in (p.strip() for p in text.split(",")) if s]


def _recipe_from_args(args: argparse.Namespace) -> tuple[dict, dict[str, str]]:
    """The recipe the source file options describe, and where each table's map is from."""
    from bakumine.core.drillholes.importer import ImportOptions
    from bakumine.core.drillholes.recipe import make_recipe

    if not args.collar:
        raise CommandError("give the collar table with --collar FILE (or a --recipe)")
    files: dict[str, list[Path]] = {"collar": [Path(p) for p in args.collar]}
    for name in ("survey", "assay", "geology"):
        if getattr(args, name):
            files[name] = [Path(p) for p in getattr(args, name)]
    for item in args.interval or []:
        name, sep, value = item.partition("=")
        if not sep or not name or not value:
            raise CommandError(f"--interval expects NAME=FILE, got {item!r}")
        if name in ("collar", "survey"):
            raise CommandError(f"table {name!r} is given twice")
        files.setdefault(name, []).append(Path(value))
    for name, paths in files.items():
        for path in paths:
            if not path.is_file():
                raise CommandError(f"{name} file not found: {path}")
    if not args.map_file and not args.auto:
        raise CommandError("give a column map with --map FILE, or use --auto")

    maps = _read_map_file(Path(args.map_file)) if args.map_file else {}
    unknown = sorted(set(maps) - set(files))
    if unknown:
        raise CommandError(f"the column map names tables that were not given: {unknown}")
    per_table = {
        "sheets": {t: _sheet_list(v) for t, v in _pairs(args.sheet, "--sheet").items()},
        "header_row": {t: int(v) for t, v in _pairs(args.header_row, "--header-row").items()},
        "header_rows": {t: int(v) for t, v in _pairs(args.header_rows, "--header-rows").items()},
    }
    named = {t for values in per_table.values() for t in values}
    named |= set(args.hole_from_sheet) | set(args.sample_depth)
    if named - set(files):
        raise CommandError(f"options name tables that were not given: {sorted(named - set(files))}")
    options = ImportOptions(
        length_unit=args.length_unit,
        convert_to_metres=args.to_metres,
        dip_convention=args.dip_convention,
        normalise_dip=args.normalise_dip,
        coordinate_order=args.coordinate_order,
        below_detection=args.below_detection,
        above_detection=args.above_detection,
        crs=args.crs,
        coordinate_tolerance=args.coordinate_tolerance,
        text_columns=tuple(args.text_column),
        numeric_columns=tuple(args.numeric_column),
        delimiter=args.delimiter,
    )
    per_file = _source_options(args.source_option, files)
    tables: dict[str, list[dict]] = {}
    origin: dict[str, str] = {}
    for name, paths in files.items():
        own = [per_file.get((name, i), {}) for i in range(len(paths))]
        if name not in maps and not args.auto and not all("map" in o for o in own):
            raise CommandError(f"the column map has no table {name!r} (add it or use --auto)")
        origin[name] = "map file" if name in maps else "auto"
        tables[name] = []
        for path, mine in zip(paths, own, strict=True):
            source: dict = {"file": str(path)}
            for key, values in per_table.items():
                if name in values:
                    source[key] = values[name]
            if name in maps:
                source["map"] = maps[name]
            if name in args.hole_from_sheet:
                source["hole_id_from_sheet"] = True
            source.update(mine)  # what --source-option says of this file wins
            if not source.get("hole_id_from_sheet"):
                source.pop("hole_id_from_sheet", None)
            if name not in ("collar", "survey"):
                if name in args.sample_depth:
                    source["sample_depth"] = {"length": args.sample_length}
                    if args.sample_pattern:
                        source["sample_depth"]["patterns"] = list(args.sample_pattern)
                if args.qc == "off":
                    source["qc"] = {"mode": "off"}
            tables[name].append(source)
    recipe = make_recipe(
        tables,
        options,
        hole_id={"pattern": args.hole_id_pattern, "template": args.hole_id_template},
        merge={
            "precedence": args.merge_precedence,
            "tolerance": args.merge_tolerance,
            "ignore_columns": _merge_ignore(args.merge_ignore),
        },
        holes=_hole_filter(args),
    )
    return recipe, origin


def _print_import_reports(db) -> None:
    reports = db.metadata.reports
    for name, info in reports.get("tables", {}).items():
        sheets = sum(len(s["sheets"]) for s in info["sources"])
        if len(info["sources"]) > 1 or sheets > 1:
            print(f"  {name}: {len(info['sources'])} file(s), {sheets} sheet(s) stacked")
    for name, counts in reports.get("qc", {}).items():
        text = ", ".join(f"{kind} {n}" for kind, n in counts.items())
        print(f"  {name} QC rows set aside ({sum(counts.values())}): {text}")
    for name, checks in reports.get("sample_depth", {}).items():
        for check in checks:
            print(
                f"  {name} depths from sample numbers: {check['rows']} rows in "
                f"{len(check['holes'])} holes; not understood {check['unmatched']}, "
                f"duplicates {check['duplicates']}, backwards {check['backwards']}, "
                f"gaps {check['gaps']}, overlaps {check['overlaps']}"
            )
    for name, merged in reports.get("conflicts", {}).items():
        print(
            f"  {name} sources merged ({merged['precedence']} file wins): {merged['merged']} "
            f"rows joined, {merged['filled']} cells filled, {merged['conflicts']} conflicts"
        )
        if merged.get("by_column"):
            by_column = ", ".join(f"{c} {n}" for c, n in merged["by_column"].items())
            print(f"    conflicts by column: {by_column}")
        for row in merged["rows"][:5]:
            key = ", ".join(str(row[k]) for k in merged["keys"])
            print(
                f"    conflict {key} {row['column']}: kept {row['kept']} "
                f"({row['kept_source']}), dropped {row['dropped']} ({row['dropped_source']})"
            )
        if merged.get("informational"):
            apart = ", ".join(f"{c} {n}" for c, n in merged["informational"].items())
            print(f"    differences not counted as conflicts (informational): {apart}")
        if merged.get("listing"):
            print(f"    every difference: {merged['listing']} in the database folder")
    for name, entries in reports.get("sources", {}).items():
        for entry in entries:
            said = ", ".join(f"{k.replace('_', ' ')} {v}" for k, v in entry.items() if k != "file")
            if said:
                print(f"  {name} {entry['file']}: {said}")
    chosen = reports.get("hole_filter")
    if chosen:
        rules = "; ".join(
            f"{what} {', '.join(chosen[what])}" for what in ("include", "exclude") if chosen[what]
        )
        kept = ", ".join(
            f"{name} {c['rows_kept']} of {c['rows_kept'] + c['rows_dropped']} rows"
            for name, c in chosen["tables"].items()
        )
        print(f"  hole filter ({rules}): kept {kept}")
        if chosen["unmatched"]:
            print(f"  WARNING: hole filter matched no hole: {', '.join(chosen['unmatched'])}")
    for name, gone in reports.get("dropped_rows", {}).items():
        parts = [
            f"{gone[key]} {text}"
            for key, text in (
                ("total_rows", "total / summary row(s)"),
                ("rows_without_hole_id", "without a hole id"),
                ("collars_without_coordinates", "collar(s) without coordinates"),
            )
            if gone.get(key)
        ]
        holes = gone.get("holes_without_coordinates") or []
        listed = f" ({', '.join(holes[:8])}{' ...' if len(holes) > 8 else ''})" if holes else ""
        print(f"  {name} rows dropped: {', '.join(parts)}{listed}")
    coordinates = reports.get("coordinates")
    if coordinates:
        print(
            f"  collar longitude / latitude -> {coordinates['crs']}: {coordinates['converted']} "
            f"converted, {coordinates['checked']} checked against the projected columns, "
            f"{coordinates['disagree']} differ by more than {coordinates['tolerance']:g}"
        )
        for row in coordinates["rows"][:5]:
            print(f"    {row['hole_id']}: differs by {row['difference']:g}")
    survey = reports.get("survey")
    if survey:
        print(f"  survey made from the collar table: {survey['generated_from_collar']} holes")


def run_import(args: argparse.Namespace) -> int:
    from bakumine.core.drillholes.recipe import (
        RECIPE_FILE,
        MissingColumns,
        load_recipe,
        run_recipe,
        save_recipe,
    )

    if not args.out and not args.dry_run:
        raise CommandError("give the database folder with --out (or use --dry-run)")
    if args.recipe:
        given = [o for o in ("collar", "survey", "assay", "geology", "interval", "map_file")
                 if getattr(args, o)]  # fmt: skip
        if given or args.auto:
            raise CommandError("--recipe replaces the source file options, --map and --auto")
        path = Path(args.recipe)
        if not path.is_file():
            raise CommandError(f"recipe not found: {path}")
        recipe = load_recipe(path)
        chosen = _hole_filter(args)
        if chosen:
            recipe["holes"] = chosen
        base = Path(args.base) if args.base else path.resolve().parent
        origin = {name: "recipe" for name in recipe["tables"]}
    else:
        recipe, origin = _recipe_from_args(args)
        base = None
    try:
        db = run_recipe(recipe, base)
    except MissingColumns as error:
        how = "--auto found" if not args.recipe else "found"
        raise CommandError(
            f"{how} no column for {error.missing} in the {error.table} table "
            f"(headers: {error.headers}); give a column map"
            + (" with --map" if not args.recipe else " in the recipe")
        ) from None

    if args.save_recipe:
        _wrote(save_recipe(db.metadata.recipe, args.save_recipe), "import recipe")
    if args.dry_run:
        print("dry run: no database written")
    else:
        folder = db.save(args.out)
        _wrote(folder / "database.json", "metadata and column maps")
        _wrote(folder / RECIPE_FILE, "import recipe; replay with `bakumine import --recipe`")
        for name, table in db.tables.items():
            file = (
                f"{name}.parquet" if name in ("collar", "survey") else f"interval_{name}.parquet"
            )
            _wrote(folder / file, f"{len(table)} rows")
        for name, table in db.qc.items():
            _wrote(folder / f"qc_{name}.parquet", f"{len(table)} QC rows")
        for name, table in db.listings.items():
            _wrote(folder / f"{name}.csv", f"{len(table)} rows, the full listing")
    print("column maps:")
    for name, sources in db.metadata.recipe["tables"].items():
        for source in sources:
            pairs = ", ".join(f"{s}->{t}" for s, t in source.get("map", {}).items())
            how = origin.get(name, "auto")
            if len(sources) > 1:  # say which file
                how += f", {Path(str(source['file'])).name}"
            print(f"  {name} ({how}): {pairs}")
    print(
        f"holes {db.collar['hole_id'].nunique()}, survey stations {len(db.survey)}, "
        + ", ".join(f"{name} intervals {len(table)}" for name, table in db.intervals.items())
    )
    source_unit = db.metadata.source_length_unit
    converted = "" if db.length_unit == source_unit else f" (converted from {source_unit})"
    print(
        f"length unit {db.length_unit}{converted}, dip convention {db.dip_convention}, "
        f"coordinate order {db.metadata.coordinate_order}"
    )
    for note in db.metadata.notes:
        if note["kind"] in ("x_looks_like_northing", "x_looks_like_easting"):
            print(
                f"WARNING: the collar column mapped to x {note['kind'][2:].replace('_', ' ')}"
                "s: check the axes, or import with --coordinate-order auto"
            )
    for name in db.intervals:
        table = db.intervals[name]
        units = db.metadata.reports.get("units", {}).get(name, {})
        columns = ", ".join(
            f"{c} [{table[c].dtype}{', ' + units[c] if c in units else ''}]"
            for c in db.attribute_columns(name)
        )
        print(f"  {name} attributes: {columns or '(none)'}")
    _print_import_reports(db)
    for note in db.metadata.notes[:10]:
        values = note.get("values")
        detail = (
            " (" + ", ".join(f"{k!r} x {v}" for k, v in list(values.items())[:6]) + ")"
            if values
            else ""
        )
        print(f"  note: {note['table']}.{note['column']}: {note['kind']} x {note['count']}{detail}")
    if len(db.metadata.notes) > 10:
        print(f"  ... {len(db.metadata.notes) - 10} more notes in database.json")
    return 0


# --------------------------------------------------------------------------------------
# validate
# --------------------------------------------------------------------------------------
def configure_validate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("database", help="database folder written by `bakumine import`")
    parser.add_argument(
        "--out", help="findings CSV (default: <database>/validation_findings.csv)"
    )
    parser.add_argument("--summary-out", help="also write the per-hole summary CSV here")
    parser.add_argument(
        "--allow-errors", action="store_true", help="exit with code 0 even when errors are found"
    )
    parser.add_argument(
        "--max-deviation",
        type=float,
        default=10.0,
        help="sudden-deviation threshold, degrees per 30 m (default: 10)",
    )
    parser.add_argument(
        "--depth-tolerance", type=float, default=1e-6, help="slack for depth comparisons"
    )
    parser.set_defaults(func=_guarded(run_validate))


def run_validate(args: argparse.Namespace) -> int:
    from bakumine.core.drillholes.validate import CHECKS, validate

    db = _load(args.database)
    report = validate(
        db, max_deviation_deg_per_30m=args.max_deviation, depth_tolerance=args.depth_tolerance
    )
    out = Path(args.out) if args.out else Path(args.database) / "validation_findings.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    report.findings.to_csv(out, index=False)
    _wrote(out, f"{len(report.findings)} findings")
    if args.summary_out:
        summary = Path(args.summary_out)
        summary.parent.mkdir(parents=True, exist_ok=True)
        report.summary.to_csv(summary, index=False)
        _wrote(summary, f"{len(report.summary)} holes")

    severity = report.findings["severity"].value_counts()
    errors = int(severity.get("error", 0))
    print(
        f"holes {len(report.summary)}, errors {errors}, "
        f"warnings {int(severity.get('warning', 0))}, infos {int(severity.get('info', 0))}"
    )
    with_errors = int((report.summary["errors"] > 0).sum())
    clean = len(report.summary) - with_errors
    print(f"holes with errors {with_errors}, holes without errors {clean}")
    for code, count in sorted(report.counts().items(), key=lambda item: (-item[1], item[0])):
        print(f"  {CHECKS[code][0]:<8}{code:<28}{count:>7}  {CHECKS[code][1]}")
    for _, row in report.errors.head(5).iterrows():
        print(f"  first errors: [{row['table']}] {row['hole_id']}: {row['message']}")
    if errors and not args.allow_errors:
        print("validation FAILED (use --allow-errors to continue anyway)")
        return 1
    print("validation passed" if not errors else "validation has errors (allowed)")
    return 0


# --------------------------------------------------------------------------------------
# desurvey
# --------------------------------------------------------------------------------------
def configure_desurvey(parser: argparse.ArgumentParser) -> None:
    from bakumine.core.drillholes.desurvey import METHODS

    parser.add_argument("database", help="database folder written by `bakumine import`")
    parser.add_argument("--out", help="output folder (default: the database folder)")
    parser.add_argument(
        "--method", choices=METHODS, default="minimum_curvature", help="desurvey method"
    )
    parser.add_argument(
        "--format", choices=("parquet", "csv", "both"), default="both", help="output format"
    )
    parser.add_argument(
        "--midpoints",
        action="append",
        default=[],
        metavar="TABLE",
        help="also write the XYZ of every interval midpoint of this table (repeatable)",
    )
    parser.set_defaults(func=_guarded(run_desurvey))


def run_desurvey(args: argparse.Namespace) -> int:
    import numpy as np

    db = _load(args.database)
    for name in args.midpoints:
        if name not in db.intervals:
            raise CommandError(f"no interval table {name!r}; database has {list(db.intervals)}")
    folder = Path(args.out) if args.out else Path(args.database)
    formats = _formats(args.format)
    traces = db.desurvey(method=args.method)
    stations = traces.stations()
    _write_table(stations, folder / "stations", formats, f"{len(stations)} stations")
    if traces.warnings:
        path = folder / "desurvey_warnings.csv"
        traces.warnings_frame().to_csv(path, index=False)
        _wrote(path, f"{len(traces.warnings)} warnings")

    located: dict[str, tuple[int, int]] = {}
    for name in args.midpoints:
        table = db.intervals[name]
        middle = 0.5 * (table["depth_from"] + table["depth_to"]).to_numpy(dtype=np.float64)
        xyz, direction = traces.xyz_at(table["hole_id"].to_numpy(dtype=object), middle)
        out = table[["hole_id", "depth_from", "depth_to"]].copy()
        out["depth_mid"] = middle
        for i, axis in enumerate("xyz"):
            out[axis] = xyz[:, i]
        for i, axis in enumerate("xyz"):
            out[f"d{axis}"] = direction[:, i]
        for column in db.attribute_columns(name):
            target = column if column not in out.columns else f"{name}_{column}"
            out[target] = table[column]
        found = int(np.isfinite(xyz[:, 0]).sum())
        located[name] = (found, len(out))
        _write_table(out, folder / f"midpoints_{name}", formats, f"{len(out)} midpoints")

    print(
        f"method {args.method}, dip convention {db.dip_convention}, length unit {db.length_unit}"
    )
    print(
        f"holes traced {len(traces)} of {db.collar['hole_id'].nunique()}, "
        f"stations {len(stations)}, "
        f"total length {float(np.sum(traces.max_depth)):.1f} {db.length_unit}"
    )
    if len(stations):
        low, high = stations[["x", "y", "z"]].min(), stations[["x", "y", "z"]].max()
        print(
            f"extent x {low['x']:.2f}..{high['x']:.2f}, y {low['y']:.2f}..{high['y']:.2f}, "
            f"z {low['z']:.2f}..{high['z']:.2f}"
        )
    codes = traces.warnings_frame()["code"].value_counts()
    print("warnings: " + (", ".join(f"{c} x {n}" for c, n in codes.items()) or "none"))
    for name, (found, total) in located.items():
        print(f"midpoints {name}: {found} of {total} located on a trace")
    return 0


# --------------------------------------------------------------------------------------
# composite
# --------------------------------------------------------------------------------------
def configure_composite(parser: argparse.ArgumentParser) -> None:
    from bakumine.core.drillholes.composite import MODES, SHORT_RULES
    from bakumine.core.drillholes.desurvey import METHODS

    parser.add_argument("database", help="database folder written by `bakumine import`")
    parser.add_argument("--table", default="assay", help="interval table (default: assay)")
    parser.add_argument(
        "--mode",
        choices=MODES,
        default="downhole",
        help="downhole: fixed length from the collar; bench: between elevation levels; "
        "domain: fixed length restarting at every domain contact",
    )
    parser.add_argument("--length", type=float, help="composite length (downhole, domain)")
    parser.add_argument("--bench-height", type=float, help="bench height (bench mode)")
    parser.add_argument(
        "--bench-reference",
        type=float,
        default=0.0,
        help="elevation of one bench level (default: 0)",
    )
    parser.add_argument(
        "--domain-column",
        help="categorical column with the domain code, e.g. rock. Composites never cross "
        "a change of this code (in any mode)",
    )
    parser.add_argument(
        "--domain-table",
        help="interval table holding --domain-column, e.g. geology (default: --table itself)",
    )
    parser.add_argument(
        "--best-fit",
        action="store_true",
        help="divide each run equally, nearest to --length, so there is no short remnant",
    )
    parser.add_argument("--attributes", help="comma-separated columns (default: all numeric)")
    parser.add_argument("--density", help="density column: weight by length x density")
    parser.add_argument(
        "--fill-missing",
        action="append",
        metavar="VALUE|ATTR=VALUE",
        help="value for missing samples and unsampled gaps (default: missing stays missing)",
    )
    parser.add_argument(
        "--min-length",
        type=float,
        default=0.0,
        help="composites below this fraction of the length are short (default: 0)",
    )
    parser.add_argument(
        "--short", choices=SHORT_RULES, default="keep", help="what to do with short composites"
    )
    parser.add_argument(
        "--min-coverage",
        type=float,
        default=0.0,
        help="attribute is NaN when its sampled fraction of the composite is below this",
    )
    parser.add_argument(
        "--keep-empty", action="store_true", help="keep composites without any sampled length"
    )
    parser.add_argument(
        "--drop-invalid", action="store_true", help="skip unusable interval rows instead of failing"
    )
    parser.add_argument(
        "--cap",
        action="append",
        metavar="ATTR=VALUE|ATTR:DOMAIN=VALUE",
        help="top-cut the samples before compositing; DOMAIN '*' is every other domain "
        "(repeatable). The uncut grade is composited too, as <ATTR>_uncut",
    )
    parser.add_argument(
        "--method", choices=METHODS, default="minimum_curvature", help="desurvey method"
    )
    parser.add_argument(
        "--out", help="output path without extension (default: <database>/composites_<table>)"
    )
    parser.set_defaults(func=_guarded(run_composite))


def _number(text: str, option: str) -> float:
    try:
        return float(text)
    except ValueError:
        raise CommandError(f"{option}: {text!r} is not a number") from None


def _parse_caps(items: list[str] | None) -> dict[str, float | dict[str, float]]:
    caps: dict[str, float | dict[str, float]] = {}
    for item in items or []:
        key, sep, value = item.rpartition("=")
        if not sep or not key:
            raise CommandError(f"--cap expects ATTR=VALUE or ATTR:DOMAIN=VALUE, got {item!r}")
        number = _number(value, "--cap")
        attribute, colon, code = key.partition(":")
        if not colon:
            if attribute in caps:
                raise CommandError(f"--cap gives {attribute!r} both a single and a per-domain cap")
            caps[attribute] = number
            continue
        existing = caps.setdefault(attribute, {})
        if not isinstance(existing, dict) or code in existing or not code:
            raise CommandError(f"--cap {item!r} conflicts with another --cap of {attribute!r}")
        existing[code] = number
    return caps


def _parse_fill(items: list[str] | None) -> float | dict[str, float] | None:
    if not items:
        return None
    if len(items) == 1 and "=" not in items[0]:
        return _number(items[0], "--fill-missing")
    fill: dict[str, float] = {}
    for item in items:
        name, sep, value = item.partition("=")
        if not sep or not name:
            raise CommandError("--fill-missing expects one VALUE, or ATTR=VALUE for each attribute")
        fill[name] = _number(value, "--fill-missing")
    return fill


def run_composite(args: argparse.Namespace) -> int:
    import warnings

    import numpy as np

    from bakumine.core.drillholes.capping import cap_grades
    from bakumine.core.drillholes.composite import (
        IntervalDomains,
        composite,
        coverage_column,
        split_at_domains,
    )

    db = _load(args.database)
    if args.table not in db.intervals:
        raise CommandError(f"no interval table {args.table!r}; database has {list(db.intervals)}")
    samples = db.intervals[args.table]
    on_invalid = "drop" if args.drop_invalid else "raise"
    caps = _parse_caps(args.cap)
    fill = _parse_fill(args.fill_missing)
    attributes = None
    if args.attributes:
        attributes = [a.strip() for a in args.attributes.split(",") if a.strip()]

    domain: str | None = None
    if args.domain_table and not args.domain_column:
        raise CommandError("--domain-table needs --domain-column")
    if args.domain_column:
        domain_table = args.domain_table or args.table
        if domain_table not in db.intervals:
            raise CommandError(
                f"no interval table {domain_table!r}; database has {list(db.intervals)}"
            )
        if args.domain_column not in db.intervals[domain_table].columns:
            raise CommandError(f"table {domain_table!r} has no column {args.domain_column!r}")
        if domain_table == args.table:
            domain = args.domain_column
        else:  # split the samples at the contacts; the code lands in a new column
            domain = "domain"
            while domain in samples.columns:
                domain = f"_{domain}"
            samples = split_at_domains(
                samples,
                IntervalDomains(db.intervals[domain_table], args.domain_column),
                name=domain,
                on_invalid=on_invalid,
            )
    if args.mode == "domain" and domain is None:
        raise CommandError("--mode domain needs --domain-column (and usually --domain-table)")

    cap_report = None
    if caps:
        per_domain = any(isinstance(c, dict) for c in caps.values())
        if per_domain and domain is None:
            raise CommandError("per-domain caps (ATTR:DOMAIN=VALUE) need --domain-column")
        samples, cap_report = cap_grades(samples, caps, domain=domain)
        if attributes is not None:
            attributes += [f"{a}_uncut" for a in caps if a in attributes]

    traces = db.desurvey(method=args.method)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        out = composite(
            samples,
            traces,
            length=args.length,
            mode=args.mode,
            attributes=attributes,
            bench_height=args.bench_height,
            bench_reference=args.bench_reference,
            domain=domain,
            best_fit=args.best_fit,
            density=args.density,
            fill_missing=fill,
            min_length=args.min_length,
            short=args.short,
            min_coverage=args.min_coverage,
            drop_empty=not args.keep_empty,
            on_invalid=on_invalid,
        )
    base = Path(args.out) if args.out else Path(args.database) / f"composites_{args.table}"
    _write_table(out, base, ("parquet", "csv"), f"{len(out)} composites")
    if cap_report is not None:
        path = base.with_name(f"{base.name}_capping.csv")
        cap_report.to_csv(path, index=False)
        _wrote(path, f"{len(cap_report)} capping report rows")

    size = (
        f"bench height {args.bench_height:g}, reference {args.bench_reference:g}"
        if args.mode == "bench"
        else f"length {args.length:g}" + (" (best fit)" if args.best_fit else "")
    )
    where = ""
    if domain:
        where = f", domains from {args.domain_table or args.table}.{args.domain_column}"
    print(f"table {args.table}, mode {args.mode}, {size}{where}, unit {db.length_unit}")
    for warning in caught:
        print(f"warning: {warning.message}")
    if not len(out):
        print("no composites")
        return 0
    print(
        f"composites {len(out)} in {out['hole_id'].nunique()} holes, length min "
        f"{out['length'].min():.3f} / mean {out['length'].mean():.3f} / max "
        f"{out['length'].max():.3f}, total {out['length'].sum():.1f}"
    )
    print(f"midpoints located on a trace: {int(out['x'].notna().sum())} of {len(out)}")
    if "domain" in out.columns:
        counts = out["domain"].fillna("(none)").value_counts()
        print("domains: " + ", ".join(f"{code} x {n}" for code, n in counts.items()))
    names = [c for c in out.columns if coverage_column(c) in out.columns]
    sample_length = (samples["depth_to"] - samples["depth_from"]).to_numpy(dtype=np.float64)
    print(
        f"  {'attribute':<16}{'values':>9}{'mean':>12}{'min':>12}{'max':>12}"
        f"{'accumulation kept':>20}"
    )
    for name in names[:12]:
        value = out[name].to_numpy(dtype=np.float64)
        cover = out[coverage_column(name)].to_numpy(dtype=np.float64)
        ok = np.isfinite(value)
        raw = samples[name].to_numpy(dtype=np.float64)
        metal_in = float(np.nansum(raw * sample_length))
        metal_out = float((value[ok] * cover[ok]).sum())
        kept = f"{100.0 * metal_out / metal_in:.4f} %" if metal_in else "n/a"
        if ok.any():
            mean = metal_out / cover[ok].sum() if cover[ok].sum() > 0 else float("nan")
            print(
                f"  {name:<16}{int(ok.sum()):>9}{mean:>12.5g}{value[ok].min():>12.5g}"
                f"{value[ok].max():>12.5g}{kept:>20}"
            )
        else:
            print(f"  {name:<16}{0:>9}{'-':>12}{'-':>12}{'-':>12}{kept:>20}")
    if len(names) > 12:
        print(f"  ... and {len(names) - 12} more attributes")
    if cap_report is not None:
        for _, row in cap_report.iterrows():
            scope = "" if row["domain"] is None else f" in {row['domain']}"
            print(
                f"cap {row['attribute']}{scope} at {row['cap']:g}: {row['n_cut']} of "
                f"{row['n_values']} values cut, metal removed {row['metal_removed_pct']:.2f} %"
            )
    return 0


#: name -> (help, configure); registered by ``bakumine.cli``.
COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "import": (HELP_IMPORT, configure_import),
    "validate": (HELP_VALIDATE, configure_validate),
    "desurvey": (HELP_DESURVEY, configure_desurvey),
    "composite": (HELP_COMPOSITE, configure_composite),
}
