"""A synthetic report appendix as a PDF: one assay table over 30+ pages (reportlab).

The table has what real appendices have: the header (two rows, a merged ``Interval`` over
``from`` / ``to``, and a row of column numbers) repeated on every page; the hole named
once, in a cell merged over the hole's rows on that page (and not again on the next page);
hole ids broken over two lines; empty grade cells; rows without the optional last value;
a section title across the table; one page drawn as two pieces side by side; and a small
unrelated table before it. ``write_appendix`` returns the rows it drew.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

ROWS_PER_PAGE = 40
HEADER = [
    ["No", "Hole", "Type", "Sample No", "Interval", "", "Length, m", "Au, g/t", "Ag, g/t"],
    ["", "", "", "", "from", "to", "", "", ""],
    [str(i) for i in range(1, 10)],
]
COLUMNS = ["No", "Hole", "Type", "Sample No", "from", "to", "Length, m", "Au, g/t", "Ag, g/t"]
WIDTHS = [28, 62, 40, 92, 44, 44, 50, 46, 46]


def appendix_rows(n_holes: int = 12, seed: int = 3) -> list[list[str]]:
    """The assay rows: No, Hole, Type, Sample No, from, to, Length, Au, Ag (text)."""
    rng = np.random.default_rng(seed)
    rows: list[list[str]] = []
    for h in range(1, n_holes + 1):
        hole = f"SYN25_{h:02d}"
        depth = float(rng.integers(1, 5))
        for k in range(int(rng.integers(95, 125))):
            length = float(rng.choice([1.0, 1.0, 1.0, 1.5, 0.7]))
            au = f"{rng.lognormal(-1.5, 1.0):.2f}"
            ag = f"{rng.lognormal(0.3, 0.6):.2f}" if rng.random() < 0.6 else "<1.0"
            if rng.random() < 0.04:
                au = ""  # an empty grade cell
            if rng.random() < 0.25:
                ag = ""  # the optional last value is absent
            rows.append(
                [str(len(rows) + 1), hole, "Core", f"{hole}-{240000 + len(rows)}",
                 f"{depth:.2f}", f"{depth + length:.2f}", f"{length:.2f}", au, ag]  # fmt: skip
            )
            depth += length
            _ = k
    return rows


def write_appendix(path: Path, rows: list[list[str]] | None = None) -> list[list[str]]:
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet
    from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table

    rows = rows if rows is not None else appendix_rows()
    styles = getSampleStyleSheet()
    everywhere = ((0, 0), (-1, -1))
    grid = [("GRID", *everywhere, 0.5, colors.black), ("FONTSIZE", *everywhere, 7),
            ("TOPPADDING", *everywhere, 1), ("BOTTOMPADDING", *everywhere, 1)]  # fmt: skip
    story: list = [Paragraph("Appendix 1. Drilling summary (synthetic)", styles["Heading4"])]
    summary = [["No", "Hole", "Coordinates", "", "Depth, m"], ["", "", "X", "Y", ""]]
    summary += [[str(i), f"SYN25_{i:02d}", f"{5204000 + 31 * i}.5", f"{412000 + 17 * i}.25",
                 f"{250 + i}.0"] for i in range(1, 6)]  # fmt: skip
    story.append(
        Table(summary, colWidths=[30, 70, 90, 90, 60],
              style=[*grid, ("SPAN", (2, 0), (3, 0)), ("SPAN", (0, 0), (0, 1)),
                     ("SPAN", (1, 0), (1, 1)), ("SPAN", (4, 0), (4, 1))])  # fmt: skip
    )
    story += [PageBreak(), Paragraph("Appendix 2. Core assays (synthetic)", styles["Heading4"])]
    header_spans = [("SPAN", (4, 0), (5, 0))]
    header_spans += [("SPAN", (c, 0), (c, 1)) for c in (0, 1, 2, 3, 6, 7, 8)]
    previous = None
    title_at = len(rows) // 2
    two_pieces = 2 if title_at // ROWS_PER_PAGE != 2 else 1  # not the page with the title
    for page, first in enumerate(range(0, len(rows), ROWS_PER_PAGE)):
        chunk = rows[first : first + ROWS_PER_PAGE]
        body, spans = [], []
        for offset, row in enumerate(chunk):
            line = list(row)
            if first + offset == title_at:  # a section title across the whole table
                body.append(["Infill drilling, second season"] + [""] * 8)
                spans.append(("SPAN", (0, len(body) + 2), (8, len(body) + 2)))
            if row[1] == previous:
                line[1] = ""  # the hole is named once
            else:
                previous = row[1]
                if int(row[1][-2:]) % 3 == 0:
                    line[1] = f"{row[1][:-1]}\n{row[1][-1]}"  # broken over two lines
                last = offset
                while last + 1 < len(chunk) and chunk[last + 1][1] == row[1]:
                    last += 1
                at = len(body) + 3
                # the title row may fall inside the hole: end the merged cell before it
                if first + offset < title_at <= first + last:
                    last = title_at - first - 1
                if last > offset:
                    spans.append(("SPAN", (1, at), (1, at + last - offset)))
            body.append(line)
        if page == two_pieces:  # drawn as two pieces side by side: the number column on its own
            heights = [11] * len(body)
            left = Table([[r[0]] for r in body], colWidths=WIDTHS[:1], rowHeights=heights,
                         style=grid, hAlign="LEFT")  # fmt: skip
            moved = [("SPAN", (a[0] - 1, a[1] - 3), (b[0] - 1, b[1] - 3)) for _, a, b in spans]
            right = Table([r[1:] for r in body], colWidths=WIDTHS[1:], rowHeights=heights,
                          style=[*grid, *moved], hAlign="LEFT")  # fmt: skip
            flush = [(name, (0, 0), (-1, -1), 0) for name in
                     ("LEFTPADDING", "RIGHTPADDING", "TOPPADDING", "BOTTOMPADDING")]  # fmt: skip
            story.append(
                Table([[left, right]], colWidths=[WIDTHS[0] + 6, sum(WIDTHS[1:])],
                      style=[*flush, ("VALIGN", (0, 0), (-1, -1), "TOP")])  # fmt: skip
            )
        else:
            story.append(
                Table(HEADER + body, colWidths=WIDTHS, style=[*grid, *header_spans, *spans])
            )
        story += [Spacer(1, 4), PageBreak()]
    SimpleDocTemplate(
        str(path), pagesize=A4, topMargin=30, bottomMargin=30, leftMargin=40, rightMargin=40
    ).build(story)
    return rows
