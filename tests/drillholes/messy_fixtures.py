"""Synthetic drillhole data in the shapes exploration teams keep it in.

Everything here is invented (seeded random numbers, holes ``SYN-01`` ...); only the
*structure* follows real-world workbooks: one sheet per hole, two-row headers, depths in
the sample number, QC rows among the samples, special lab values, collars in degrees /
minutes / seconds. ``truth()`` is the same data as plain tables; every writer below
produces a messy file that must import to exactly that.

    uv run python tests/drillholes/messy_fixtures.py OUT_DIR

writes the fixture files of the browser test (``web/e2e/import-shapes.spec.ts``).
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

N_HOLES = 6
UTM = "EPSG:32633"  # the synthetic project lies in UTM zone 33 north
#: one spelling for the hole ids of every file (sheet names, collar table)
HOLE_ID = {"pattern": r"SYN[_-]?0*(\d+)", "template": "SYN-{1:02d}"}
#: options of every import here: the clean CSVs are ``;`` separated, dips are positive down
OPTIONS = {"delimiter": ";", "dip_convention": "auto"}
CLEAN_MAPS = {
    "collar": {"hole_id": "hole_id", "x": "x", "y": "y", "z": "z", "max_depth": "max_depth"},
    "survey": {"hole_id": "hole_id", "depth": "depth", "dip": "dip", "azimuth": "azimuth"},
    "assay": {"hole_id": "hole_id", "from": "depth_from", "to": "depth_to", "ag": "ag",
              "au": "au"},  # fmt: skip
}


def hole_name(i: int) -> str:
    return f"SYN-{i:02d}"


def sheet_name(i: int) -> str:
    """Sheet names as people type them: ``SYN_01``, ``SYN-002``, ``SYN_3`` ..."""
    return [f"SYN_{i:02d}", f"SYN-{i:03d}", f"SYN_{i}"][i % 3]


def truth(seed: int = 7) -> dict[str, pd.DataFrame]:
    """Clean collar / survey / assay tables. Assay cells are *text as the lab wrote it*
    (``<0.01``, ``>100``, ``LNR`` ...), 1 m samples from a start depth to the end of hole."""
    rng = np.random.default_rng(seed)
    collars, surveys, assays = [], [], []
    for i in range(1, N_HOLES + 1):
        depth = float(rng.integers(18, 31))
        east = 412_000.0 + float(rng.integers(0, 900))
        north = 5_204_000.0 + float(rng.integers(0, 900))
        collars.append(
            {"hole_id": hole_name(i), "x": east, "y": north, "z": 1200.0 + i, "max_depth": depth}
        )
        surveys.append(
            {"hole_id": hole_name(i), "depth": 0.0, "dip": 60.0 + 5 * (i % 3),
             "azimuth": float(30 * i)}  # fmt: skip
        )
        start = int(rng.integers(0, 4))
        for to in range(start + 1, int(depth) + 1):
            au = f"{rng.lognormal(-1.5, 1.0):.2f}"
            ag = f"{rng.lognormal(0.5, 0.8):.1f}"
            roll = rng.random()
            if roll < 0.12:
                au = "<0.01"
            elif roll < 0.15:
                au = ">100"
            elif roll < 0.18:
                au = "LNR"
            elif roll < 0.21:
                au = "-"
            elif roll < 0.25:
                au = au.replace(".", ",")
            if rng.random() < 0.3:
                ag = "<1.0" if rng.random() < 0.7 else "n/a"
            assays.append(
                {"hole_id": hole_name(i), "from": float(to - 1), "to": float(to),
                 "ag": ag, "au": au}  # fmt: skip
            )
    return {
        "collar": pd.DataFrame(collars),
        "survey": pd.DataFrame(surveys),
        "assay": pd.DataFrame(assays),
    }


def write_clean(folder: Path, tables: dict[str, pd.DataFrame] | None = None) -> dict[str, Path]:
    """The truth as three plain CSV files (``;`` separated: assay cells hold commas)."""
    tables = tables or truth()
    folder.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, table in tables.items():
        paths[name] = folder / f"{name}.csv"
        table.to_csv(paths[name], index=False, sep=";")
    return paths


def clean_import(paths: dict[str, Path]):
    """The reference: a straightforward import of the clean CSVs (plain reader, no recipe)."""
    from bakumine.core.drillholes.importer import ImportOptions, TableSource, import_drillholes

    return import_drillholes(
        TableSource(paths["collar"], CLEAN_MAPS["collar"]),
        TableSource(paths["survey"], CLEAN_MAPS["survey"]),
        {"assay": TableSource(paths["assay"], CLEAN_MAPS["assay"])},
        options=ImportOptions(**OPTIONS),
    )


def tables_fixture():
    """A module-scoped ``tables`` fixture: the truth."""

    @pytest.fixture(scope="module")
    def tables() -> dict[str, pd.DataFrame]:
        return truth()

    return tables


def clean_fixture():
    """A module-scoped ``clean`` fixture: ``(paths of the clean CSVs, their database)``."""

    @pytest.fixture(scope="module")
    def clean(tmp_path_factory, tables):
        paths = write_clean(tmp_path_factory.mktemp("clean"), tables)
        db = clean_import(paths)
        assert db.dip_convention == "positive_down"  # all dips positive: assumed and noted
        assert {"table": "survey", "column": "dip", "kind": "dip_assumed_positive_down",
                "count": N_HOLES} in db.metadata.notes  # fmt: skip
        return paths, db

    return clean


def same_assays(db, clean_db) -> None:
    """The assay table equals the clean import's, column for column."""
    columns = ["hole_id", "depth_from", "depth_to", "ag", "au"]
    pd.testing.assert_frame_equal(
        db.intervals["assay"][columns].reset_index(drop=True),
        clean_db.intervals["assay"][columns].reset_index(drop=True),
    )


def recipe_for(paths, assay_sources, **extra):
    """A recipe with the clean collar and survey and the given assay source(s)."""
    from bakumine.core.drillholes.recipe import make_recipe

    return make_recipe(
        {
            "collar": {"file": str(paths["collar"]), "map": CLEAN_MAPS["collar"]},
            "survey": {"file": str(paths["survey"]), "map": CLEAN_MAPS["survey"]},
            "assay": assay_sources,
        },
        OPTIONS,
        hole_id=HOLE_ID,
        **extra,
    )


def _number(text: str) -> object:
    """A cell as a spreadsheet would hold it: a number when it is one."""
    try:
        return float(text)
    except ValueError:
        return text


def write_sheet_per_hole(
    path: Path,
    assay: pd.DataFrame,
    *,
    depths: str = "columns",
    qc: bool = False,
    mongolian: bool = True,
    holes: list[int] | None = None,
) -> dict[str, int]:
    """A workbook with one sheet per hole (the sheet name is the only place the hole is
    named), a blank first column, a title line, a two-row header with merged group cells
    and a summary sheet that is not a hole.

    ``depths="columns"``  from / to are filled in.
    ``depths="sample"``   from / to are empty; the sample number ``SYN01-<to>`` carries the
                          depth (every seventh one as an explicit ``SYN01-<from>-<to>``).
    ``qc``                blanks, standards and duplicates are mixed in: the word sits in
                          the interval column or the duplicate's number ends in ``A``.
    Returns the number of QC rows written, by kind.
    """
    from openpyxl import Workbook

    book = Workbook()
    book.remove(book.active)
    lower = ("аас", "хүртэл") if mongolian else ("from", "to")
    counts = {"blank": 0, "standard": 0, "letter_suffix": 0}
    lab = 24000
    for i in holes or range(1, N_HOLES + 1):
        rows = assay[assay["hole_id"] == hole_name(i)]
        sheet = book.create_sheet(sheet_name(i))
        sheet.append([None, f"Assay results, hole {i}"])
        sheet.append([])
        top = [None, "№", "Лаб №" if mongolian else "Lab No", "Дээж №" if mongolian else
               "Sample No", "Интервал" if mongolian else "Interval", None,
               "Элемент, мг/кг" if mongolian else "Element, mg/kg", None]  # fmt: skip
        sheet.append(top)
        sheet.append([None, None, None, None, lower[0], lower[1], "Ag", "Au"])
        sheet.merge_cells(start_row=3, start_column=5, end_row=3, end_column=6)
        sheet.merge_cells(start_row=3, start_column=7, end_row=3, end_column=8)
        for k, row in enumerate(rows.itertuples(index=False), start=1):
            lab += 1
            to = int(row.to)
            sample = f"SYN{i:02d}-{to}"
            if depths == "sample" and k % 7 == 0:
                sample = f"SYN{i:02d}-{to - 1}-{to}"
            interval = [row[1], row.to] if depths == "columns" else [None, None]
            sheet.append([None, k, lab, sample, *interval, _number(row.ag), _number(row.au)])
            if qc and k % 6 == 0:
                lab += 1
                word = "Standart" if k % 12 == 0 else "Blank"
                counts["standard" if k % 12 == 0 else "blank"] += 1
                sheet.append([None, None, lab, f"SYN{i:02d}-Q{k}", word, None, 5.0, 1.05])
            if qc and k % 10 == 0:
                lab += 1
                counts["letter_suffix"] += 1
                sheet.append([None, None, lab, f"{sample}A", *interval, 1.0, 0.5])
    summary = book.create_sheet("Summary")
    summary.append(["Hole", "Samples"])
    for i in holes or range(1, N_HOLES + 1):
        summary.append([hole_name(i), int((assay["hole_id"] == hole_name(i)).sum())])
    book.save(path)
    return counts


def dms(value: float) -> tuple[int, int, float]:
    degrees = int(value)
    minutes = int((value - degrees) * 60)
    seconds = round((value - degrees - minutes / 60) * 3600, 3)
    return degrees, minutes, seconds


def write_collar_dms(
    path: Path,
    collar: pd.DataFrame,
    survey: pd.DataFrame,
    *,
    projected: bool = True,
    blank_projected: tuple[str, ...] = (),
    shifted: dict[str, float] | None = None,
    text: bool = False,
) -> None:
    """A collar workbook as a survey office writes it: a title, the header on the third
    row with ``Longitude`` and ``Latitude`` each merged over three columns (degrees,
    minutes, seconds), projected ``X`` = northing and ``Y`` = easting (CIS order),
    elevation, azimuth, dip (positive = down) and depth, in Mongolian.

    ``blank_projected`` holes have no X / Y; ``shifted`` moves a hole's X / Y by metres;
    ``text=True`` writes the angles as one text cell each (``15°50'33.1"E``)."""
    from openpyxl import Workbook
    from pyproj import Transformer

    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(
        collar["x"].to_numpy(), collar["y"].to_numpy()
    )
    book = Workbook()
    sheet = book.active
    sheet.title = "Нэгтгэл"
    sheet.append(["Drill collars (synthetic)"])
    sheet.append([])
    geographic = ["Уртраг", "Өргөрөг"] if text else ["Уртраг", None, None, "Өргөрөг", None, None]
    header = ["№", "Цооног", *geographic]
    if projected:
        header += ["X", "Y"]
    header += ["Өндөршил", "Азимут /градус/", "Цооногийн налуу /градус/", "Цооногийн гүн /м/"]
    sheet.append(header)
    if not text:
        sheet.merge_cells(start_row=3, start_column=3, end_row=3, end_column=5)
        sheet.merge_cells(start_row=3, start_column=6, end_row=3, end_column=8)
    stations = survey.set_index("hole_id")
    for k, row in enumerate(collar.itertuples(index=False)):
        a, b = dms(float(lon[k])), dms(float(lat[k]))
        if text:
            angles = [f"{a[0]}°{a[1]}'{a[2]}\"E", f"{b[0]}°{b[1]}'{b[2]}\"N"]
        else:
            angles = [*a, *b]
        line = [k + 1, row.hole_id.replace("-", "_"), *angles]
        if projected:
            shift = (shifted or {}).get(row.hole_id, 0.0)
            xy = [row.y + shift, row.x]  # X = northing, Y = easting
            line += [None, None] if row.hole_id in blank_projected else xy
        station = stations.loc[row.hole_id]
        line += [row.z, station["azimuth"], station["dip"], row.max_depth]
        sheet.append(line)
    book.save(path)


def write_e2e_fixtures(folder: Path) -> None:
    """The files the browser test uploads: an assay workbook with one sheet per hole,
    QC rows and depths in the sample numbers, a DMS collar workbook, and the truth."""
    tables = truth()
    folder.mkdir(parents=True, exist_ok=True)
    counts = write_sheet_per_hole(folder / "assays.xlsx", tables["assay"], depths="sample", qc=True)
    write_collar_dms(folder / "collars.xlsx", tables["collar"], tables["survey"])
    write_clean(folder / "clean", tables)
    summary = {
        "holes": N_HOLES,
        "assay_rows": int(len(tables["assay"])),
        "qc_rows": int(sum(counts.values())),
    }
    (folder / "expected.json").write_text(pd.Series(summary).to_json())


if __name__ == "__main__":
    write_e2e_fixtures(Path(sys.argv[1]))
