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


# --------------------------------------------------------------------------------------
# a collar table cut by page breaks the way report writers leave it
# --------------------------------------------------------------------------------------
COLLAR_HEADER = [
    ["No", "Hole", "Type", "Coordinates", "", "Elevation, m", "Azimuth, deg", "Dip, deg",
     "Depth, m"],
    ["", "", "", "X", "Y", "", "", "", ""],
    [str(i) for i in range(1, 10)],
]  # fmt: skip
COLLAR_COLUMNS = ["No", "Hole", "Type", "X", "Y", "Elevation, m", "Azimuth, deg", "Dip, deg",
                  "Depth, m"]  # fmt: skip
COLLAR_WIDTHS = [26, 62, 36, 70, 66, 58, 58, 46, 50]
#: rows per page; what each page starts with
COLLAR_PAGES = (
    (12, "header"),  # the header, two rows and the column numbers
    (18, "open"),  # no header, and no rule above the first row (cut by the page break)
    (15, "header"),  # the header again
    (14, "text"),  # no header; the first row holds text only (a hole not yet surveyed)
    (9, "open"),  # open again; the table ends with a total row
)


def collar_rows() -> list[list[str]]:
    """No, Hole, Type, X (northing), Y (easting), elevation, azimuth, dip, depth; then a
    total row. The first row of the fourth page has text only."""
    rng = np.random.default_rng(11)
    rows: list[list[str]] = []
    text_only = sum(n for n, _ in COLLAR_PAGES[:3])
    total = 0.0
    for i in range(sum(n for n, _ in COLLAR_PAGES) - 1):
        hole = f"SYN25_{i + 1:02d}"
        if i == text_only:
            rows.append(["", hole, "RC", "", "", "", "", "", ""])
            continue
        depth = float(rng.integers(120, 400)) + 0.5
        total += depth
        rows.append(
            [str(i + 1), hole, "Core" if i % 4 else "RC",
             f"{5204000 + rng.uniform(0, 900):.3f}", f"{412000 + rng.uniform(0, 900):.3f}",
             f"{1230 + rng.uniform(0, 20):.3f}", str(int(rng.choice([30, 210]))),
             str(int(rng.choice([-60, -70, -90]))), f"{depth:.1f}"]  # fmt: skip
        )
    rows.append(["Total", "", "", "", "", "", "", "", f"{total:.1f}"])
    return rows


def write_collar_report(path: Path) -> list[list[str]]:
    """Draw :func:`collar_rows` over five pages (see ``COLLAR_PAGES``), then a different
    small table on a page of its own. Returns the collar rows."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import PageBreak, SimpleDocTemplate, Table

    rows = collar_rows()
    everywhere = ((0, 0), (-1, -1))
    font = [("FONTSIZE", *everywhere, 7), ("TOPPADDING", *everywhere, 1),
            ("BOTTOMPADDING", *everywhere, 1)]  # fmt: skip
    closed = [*font, ("GRID", *everywhere, 0.5, colors.black)]
    # every rule but the one above the first row
    open_top = [*font, ("INNERGRID", *everywhere, 0.5, colors.black),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.black),
                ("LINEBEFORE", (0, 0), (0, -1), 0.5, colors.black),
                ("LINEAFTER", (-1, 0), (-1, -1), 0.5, colors.black)]  # fmt: skip
    spans = [("SPAN", (3, 0), (4, 0))] + [("SPAN", (c, 0), (c, 1)) for c in (0, 1, 2, 5, 6, 7, 8)]
    story: list = []
    first = 0
    for count, start in COLLAR_PAGES:
        body = rows[first : first + count]
        first += count
        if start == "header":
            story.append(Table(COLLAR_HEADER + body, colWidths=COLLAR_WIDTHS,
                               style=[*closed, *spans]))  # fmt: skip
        else:
            story.append(Table(body, colWidths=COLLAR_WIDTHS,
                               style=open_top if start == "open" else closed))  # fmt: skip
        story.append(PageBreak())
    other = [["Trench", "Length, m", "Samples"], ["TR-1", "40.0", "38"], ["TR-2", "55.5", "51"]]
    story.append(Table(other, colWidths=[80, 80, 80], style=closed))
    SimpleDocTemplate(
        str(path), pagesize=A4, topMargin=30, bottomMargin=30, leftMargin=40, rightMargin=40
    ).build(story)
    return rows
