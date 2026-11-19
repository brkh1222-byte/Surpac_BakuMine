"""Build the tidy Babbitt tables (collar.csv, survey.csv, assay.csv) from raw/.

The raw files are already plain CSV; this script only
  * renames columns to lower-case descriptive names,
  * adds a `depth` column to the collar table (the raw collar has none), taken as the
    largest assay TO of each hole (every hole is logged from 0 with no gaps),
  * writes the rows in the raw order.

No value is rescaled, rounded or sign-flipped: every cell is copied as the exact text
found in the raw file, so units (feet, weight %) and the dip convention (positive =
downward) are those of the source. See SOURCE.md.

Run from anywhere:  python data/public/babbitt/convert.py
"""

from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"

COLLAR_COLS = {
    "BHID": "hole_id",
    "XCOLLAR": "easting",
    "YCOLLAR": "northing",
    "ZCOLLAR": "elevation",
}
SURVEY_COLS = {"BHID": "hole_id", "AT": "depth", "DIP": "dip", "AZ": "azimuth"}
ASSAY_COLS = {
    "BHID": "hole_id",
    "FROM": "from",
    "TO": "to",
    "CU": "cu",
    "NI": "ni",
    "S": "s",
    "FE": "fe",
}


def read_raw(name: str) -> pd.DataFrame:
    """Read a raw table with every cell kept as its original text."""
    return pd.read_csv(RAW / name, dtype=str, keep_default_na=False)


def main() -> None:
    collar = read_raw("collar_BABBITT.csv").rename(columns=COLLAR_COLS)
    survey = read_raw("survey_BABBITT.csv").rename(columns=SURVEY_COLS)
    assay = read_raw("assay_BABBITT.csv").rename(columns=ASSAY_COLS)

    # Hole depth = text of the deepest assay TO for that hole.
    deepest = assay.loc[assay["to"].astype(float).groupby(assay["hole_id"]).idxmax()]
    depth = deepest.set_index("hole_id")["to"]
    collar["depth"] = collar["hole_id"].map(depth)
    if collar["depth"].isna().any():
        raise SystemExit("collar hole without assay intervals; cannot derive depth")

    collar = collar[["hole_id", "easting", "northing", "elevation", "depth"]]
    survey = survey[["hole_id", "depth", "dip", "azimuth"]]
    assay = assay[["hole_id", "from", "to", "cu", "ni", "s", "fe"]]

    for name, table in (("collar", collar), ("survey", survey), ("assay", assay)):
        table.to_csv(HERE / f"{name}.csv", index=False, lineterminator="\n")
        print(f"{name}.csv: {len(table)} rows")


if __name__ == "__main__":
    main()
