"""``bakumine extract-tables``: tables of a text PDF (a report appendix) as CSV.

    bakumine extract-tables REPORT.pdf --pages 6-229 --out DIR

writes ``DIR/<name>_1.csv`` ... (one per table found in the page range) and
``DIR/<name>_report.json``, and prints for each table its pages, rows and, when it has
hole / from / to columns, the rows per hole, the depth range and the holes where ``from``
goes backwards. The CSVs are ordinary inputs of ``bakumine import`` and the Import panel.
See :mod:`bakumine.core.io.pdf_tables`.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

HELP_EXTRACT = "Extract the tables of a text PDF (report appendix) as CSV files"
_MAX_HOLES_PRINTED = 12


def configure_extract(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("pdf", help="PDF file with text (not a scan)")
    parser.add_argument("--pages", help="pages to read, e.g. '6-229' or '3-5,9' (default: all)")
    parser.add_argument("--out", required=True, help="folder for the CSV files and the report")
    parser.add_argument(
        "--carry-down",
        action="append",
        metavar="COLUMN",
        help="column (name or 1-based position) whose empty cells take the value above, "
        "for a merged cell that names the hole once; repeatable; 'none' turns it off "
        "(default: the hole-id column, found by its header)",
    )
    parser.add_argument(
        "--one-table",
        action="store_true",
        help="the page range holds one table: later headers that differ are data",
    )
    parser.add_argument(
        "--strategy",
        choices=("lines", "text"),
        default="lines",
        help="lines: ruled tables (default); text: guess columns from the text alignment",
    )
    parser.add_argument("--name", help="prefix of the output files (default: the PDF's name)")
    parser.set_defaults(func=run_extract)


def run_extract(args: argparse.Namespace) -> int:
    from bakumine.core.io.pdf_tables import extract_tables, write_tables

    path = Path(args.pdf)
    try:
        if not path.is_file():
            raise FileNotFoundError(f"PDF not found: {path}")
        carry: object = "auto"
        if args.carry_down:
            carry = "none" if "none" in args.carry_down else list(args.carry_down)

        def progress(done: int, total: int) -> None:
            if done % 25 == 0 or done == total:
                print(f"  page {done} of {total}", file=sys.stderr, flush=True)

        tables = extract_tables(
            path,
            args.pages,
            carry_down=carry,  # type: ignore[arg-type]
            one_table=args.one_table,
            strategy=args.strategy,
            progress=progress,
        )
        if not tables:
            raise ValueError(
                f"no table found in {path.name} (pages {args.pages or 'all'}); a scanned "
                "PDF has no text, and tables without ruling lines need --strategy text"
            )
        stem = args.name or "".join(c if c.isalnum() or c in "-_" else "_" for c in path.stem)
        paths, report_path, reports = write_tables(tables, args.out, stem)
    except (ValueError, FileNotFoundError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
    except Exception as error:  # pdfminer raises its own types on broken files
        print(f"error: cannot read {path.name}: {error}", file=sys.stderr)
        return 2
    for csv_path, report in zip(paths, reports, strict=True):
        print(
            f"wrote {csv_path} ({report['rows']} rows, {len(report['columns'])} columns, "
            f"pages {report['first_page']}-{report['last_page']})"
        )
        print(f"  columns: {', '.join(report['columns'])}")
        dropped = (
            f"{report['repeated_headers']} repeated headers and {report['title_rows']} "
            f"title rows dropped"
        )
        carried = report["carried_down"]
        if carried["columns"]:
            dropped += f"; {carried['cells']} cells of {carried['columns']} carried down"
        print(f"  {dropped}")
        if "holes" not in report:
            continue
        print(
            f"  {report['n_holes']} holes; rows without a hole {report['rows_without_hole']}, "
            f"without depths {report['rows_without_depths']}"
        )
        for hole in report["holes"][:_MAX_HOLES_PRINTED]:
            print(
                f"    {hole['hole_id']}: {hole['rows']} rows, {hole['from']} - {hole['to']}"
                + (f", from goes backwards x {hole['from_backwards']}"
                   if hole["from_backwards"] else "")  # fmt: skip
            )
        if len(report["holes"]) > _MAX_HOLES_PRINTED:
            print(f"    ... {len(report['holes']) - _MAX_HOLES_PRINTED} more in the report")
        if report["holes_from_backwards"]:
            print(f"  from goes backwards in: {', '.join(report['holes_from_backwards'][:20])}")
        if report["holes_in_several_runs"]:
            print(f"  holes in several runs: {', '.join(report['holes_in_several_runs'][:20])}")
    print(f"wrote {report_path} (rows per hole, depth ranges, dropped titles)")
    return 0


COMMANDS: dict[str, tuple[str, Callable[[argparse.ArgumentParser], None]]] = {
    "extract-tables": (HELP_EXTRACT, configure_extract),
}
