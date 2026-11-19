"""Shared drillhole test data: a small clean dataset and the Babbitt public dataset."""

from pathlib import Path

import pandas as pd

from bakumine.core.drillholes.importer import (
    DrillholeDatabase,
    ImportOptions,
    TableSource,
    import_drillholes,
)

REPO = Path(__file__).resolve().parents[2]
BABBITT_RAW = REPO / "data" / "public" / "babbitt" / "raw"

SMALL_MAPS = {
    "collar": {"HOLE": "hole_id", "EAST": "x", "NORTH": "y", "RL": "z", "EOH": "max_depth"},
    "survey": {"HOLE": "hole_id", "DEPTH": "depth", "DIP": "dip", "AZI": "azimuth"},
    "assay": {"HOLE": "hole_id", "FROM": "depth_from", "TO": "depth_to", "AU": "au", "CU": "cu"},
    "geology": {"HOLE": "hole_id", "FROM": "depth_from", "TO": "depth_to", "ROCK": "rock"},
}
BABBITT_MAPS = {
    "collar": {"BHID": "hole_id", "XCOLLAR": "x", "YCOLLAR": "y", "ZCOLLAR": "z"},
    "survey": {"BHID": "hole_id", "AT": "depth", "AZ": "azimuth", "DIP": "dip"},
    "assay": {
        "BHID": "hole_id",
        "FROM": "depth_from",
        "TO": "depth_to",
        "CU": "cu",
        "NI": "ni",
        "S": "s",
        "FE": "fe",
    },
}
#: Babbitt is in feet with positive-down dips (data/public/babbitt/SOURCE.md).
BABBITT_OPTIONS = ImportOptions(length_unit="ftUS", dip_convention="positive_down")


def small_raw_tables() -> dict[str, pd.DataFrame]:
    """A clean 6-hole dataset as source-style tables (source headers, text cells).

    Holes on a 2 x 3 grid, 50 m apart, each 100 m deep, dipping -60 towards 090 with a
    1 degree flattening per station; 10 m assays (one unassayed), three rock intervals.
    Hole ``1004`` has a numeric-looking ID.
    """
    holes = ["DH001", "DH002", "DH003", "1004", "DH005", "DH006"]
    collar, survey, assay, geology = [], [], [], []
    for i, hole in enumerate(holes):
        east, north = 1000.0 + 50.0 * (i % 3), 5000.0 + 50.0 * (i // 3)
        collar.append([hole, f"{east:.2f}", f"{north:.2f}", f"{200.0 + i:.1f}", "100"])
        for k, depth in enumerate((0, 50, 100)):
            survey.append([hole, str(depth), str(-60 + k), str(90 + k)])
        for k in range(10):
            au = "" if (i, k) == (1, 4) else f"{0.1 * (k + 1) + 0.01 * i:.2f}"
            assay.append([hole, str(10 * k), str(10 * k + 10), au, f"{0.5 + 0.05 * k:.2f}"])
        for start, end, rock in ((0, 15, "OVB"), (15, 62.5, "ORE"), (62.5, 100, "WST")):
            geology.append([hole, str(start), str(end), rock])
    return {
        "collar": pd.DataFrame(collar, columns=["HOLE", "EAST", "NORTH", "RL", "EOH"]),
        "survey": pd.DataFrame(survey, columns=["HOLE", "DEPTH", "DIP", "AZI"]),
        "assay": pd.DataFrame(assay, columns=["HOLE", "FROM", "TO", "AU", "CU"]),
        "geology": pd.DataFrame(geology, columns=["HOLE", "FROM", "TO", "ROCK"]),
    }


def import_small(
    tables: dict[str, pd.DataFrame] | None = None, options: ImportOptions | None = None
) -> DrillholeDatabase:
    tables = tables or small_raw_tables()
    return import_drillholes(
        TableSource(tables["collar"], SMALL_MAPS["collar"]),
        TableSource(tables["survey"], SMALL_MAPS["survey"]),
        {name: TableSource(tables[name], SMALL_MAPS[name]) for name in ("assay", "geology")},
        options=options,
    )


def import_babbitt(options: ImportOptions = BABBITT_OPTIONS) -> DrillholeDatabase:
    return import_drillholes(
        TableSource(BABBITT_RAW / "collar_BABBITT.csv", BABBITT_MAPS["collar"]),
        TableSource(BABBITT_RAW / "survey_BABBITT.csv", BABBITT_MAPS["survey"]),
        {"assay": TableSource(BABBITT_RAW / "assay_BABBITT.csv", BABBITT_MAPS["assay"])},
        options=options,
    )
