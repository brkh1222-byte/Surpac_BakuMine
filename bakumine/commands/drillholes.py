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
    from bakumine.core.drillholes.importer import BELOW_DETECTION_RULES, LENGTH_UNITS

    parser.add_argument("--collar", required=True, help="collar table (CSV or XLSX)")
    parser.add_argument("--survey", required=True, help="downhole survey table")
    parser.add_argument("--assay", help="assay interval table (stored as 'assay')")
    parser.add_argument("--geology", help="geology interval table (stored as 'geology')")
    parser.add_argument(
        "--interval",
        action="append",
        metavar="NAME=FILE",
        help="any other interval table, stored under NAME (repeatable)",
    )
    parser.add_argument("--out", required=True, help="database folder to write")
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
        choices=("negative_down", "positive_down"),
        default="negative_down",
        help="sign of the dips in the survey file (default: negative_down)",
    )
    parser.add_argument(
        "--normalise-dip", action="store_true", help="store dips as negative_down"
    )
    parser.add_argument(
        "--coordinate-order",
        default="xyz",
        help="true axes of the columns mapped to x, y, z; 'yxz' when the column mapped "
        "to x holds the northing (default: xyz)",
    )
    parser.add_argument(
        "--below-detection",
        choices=BELOW_DETECTION_RULES,
        default="half",
        help="rule for '<limit' cells (default: half the limit)",
    )
    parser.add_argument(
        "--text-column", action="append", default=[], help="attribute forced to text"
    )
    parser.add_argument(
        "--numeric-column", action="append", default=[], help="attribute forced to numeric"
    )
    parser.add_argument("--delimiter", default=",", help="CSV delimiter (default: ',')")
    parser.add_argument(
        "--sheet", action="append", metavar="TABLE=SHEET", help="XLSX sheet of a table"
    )
    parser.set_defaults(func=_guarded(run_import))


def _read_map_file(path: Path) -> dict[str, dict[str, str]]:
    if not path.is_file():
        raise CommandError(f"column map file not found: {path}")
    if path.suffix.lower() == ".toml":
        document = tomllib.loads(path.read_text(encoding="utf-8"))
    elif path.suffix.lower() == ".json":
        document = json.loads(path.read_text(encoding="utf-8"))
    else:
        raise CommandError(f"column map file must be .json or .toml, got {path.name}")
    if not isinstance(document, dict) or not all(isinstance(v, dict) for v in document.values()):
        raise CommandError(f"{path}: expected {{table: {{source column: canonical name}}}}")
    return {str(t): {str(k): str(v) for k, v in m.items()} for t, m in document.items()}


def run_import(args: argparse.Namespace) -> int:
    from bakumine.core.drillholes.importer import (
        ImportOptions,
        TableSource,
        import_drillholes,
        read_raw_table,
        suggest_column_map,
    )

    files: dict[str, Path] = {"collar": Path(args.collar), "survey": Path(args.survey)}
    if args.assay:
        files["assay"] = Path(args.assay)
    if args.geology:
        files["geology"] = Path(args.geology)
    for name, value in _pairs(args.interval, "--interval").items():
        if name in files:
            raise CommandError(f"table {name!r} is given twice")
        files[name] = Path(value)
    for name, path in files.items():
        if not path.is_file():
            raise CommandError(f"{name} file not found: {path}")
    if not args.map_file and not args.auto:
        raise CommandError("give a column map with --map FILE, or use --auto")

    sheets: dict[str, str | int] = {
        name: int(sheet) if sheet.isdigit() else sheet
        for name, sheet in _pairs(args.sheet, "--sheet").items()
    }
    maps = _read_map_file(Path(args.map_file)) if args.map_file else {}
    unknown = sorted(set(maps) - set(files))
    if unknown:
        raise CommandError(f"the column map names tables that were not given: {unknown}")
    options = ImportOptions(
        length_unit=args.length_unit,
        convert_to_metres=args.to_metres,
        dip_convention=args.dip_convention,
        normalise_dip=args.normalise_dip,
        coordinate_order=args.coordinate_order,
        below_detection=args.below_detection,
        text_columns=tuple(args.text_column),
        numeric_columns=tuple(args.numeric_column),
        delimiter=args.delimiter,
    )
    origin: dict[str, str] = {}
    for name, path in files.items():
        if name in maps:
            origin[name] = "map file"
            continue
        if not args.auto:
            raise CommandError(f"the column map has no table {name!r} (add it or use --auto)")
        kind = name if name in ("collar", "survey") else "interval"
        raw = read_raw_table(
            path, sheet=sheets.get(name), delimiter=options.delimiter, encoding=options.encoding
        )
        suggestion = suggest_column_map(raw.columns, kind)  # type: ignore[arg-type]
        if suggestion.missing:
            raise CommandError(
                f"--auto found no column for {suggestion.missing} in the {name} table "
                f"(headers: {list(raw.columns)}); give a column map with --map"
            )
        maps[name], origin[name] = suggestion.mapping, "auto"

    sources = {
        name: TableSource(path, maps[name], sheet=sheets.get(name)) for name, path in files.items()
    }
    intervals = {n: s for n, s in sources.items() if n not in ("collar", "survey")}
    db = import_drillholes(sources["collar"], sources["survey"], intervals, options=options)
    folder = db.save(args.out)

    _wrote(folder / "database.json", "metadata and column maps")
    for name, table in db.tables.items():
        file = f"{name}.parquet" if name in ("collar", "survey") else f"interval_{name}.parquet"
        _wrote(folder / file, f"{len(table)} rows")
    print("column maps:")
    for name in files:
        pairs = ", ".join(f"{source}->{target}" for source, target in maps[name].items())
        print(f"  {name} ({origin[name]}): {pairs}")
    print(
        f"holes {db.collar['hole_id'].nunique()}, survey stations {len(db.survey)}, "
        + ", ".join(f"{name} intervals {len(table)}" for name, table in db.intervals.items())
    )
    converted = (
        "" if db.length_unit == args.length_unit else f" (converted from {args.length_unit})"
    )
    print(
        f"length unit {db.length_unit}{converted}, dip convention {db.dip_convention}, "
        f"coordinate order {args.coordinate_order}"
    )
    for name in db.intervals:
        table = db.intervals[name]
        columns = ", ".join(f"{c} [{table[c].dtype}]" for c in db.attribute_columns(name))
        print(f"  {name} attributes: {columns or '(none)'}")
    for note in db.metadata.notes[:10]:
        print(f"  note: {note['table']}.{note['column']}: {note['kind']} x {note['count']}")
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
