"""Drillhole table import: CSV / XLSX files + a column map -> ``DrillholeDatabase``.

Background: ``research.md`` section 2 (what the tables are) and 2.5 (the database).

Canonical internal column names (``x = east, y = north, z = up``):

* collar   - ``hole_id, x, y, z, max_depth``
* survey   - ``hole_id, depth, dip, azimuth``
* interval - ``hole_id, depth_from, depth_to, <attributes...>`` (assay, geology, ...)

These are exactly the names ``bakumine.core.drillholes.desurvey.desurvey`` expects.

Import rules (all controlled by :class:`ImportOptions`, none by dataset special cases):

* Every cell is first read as text, so ``hole_id`` is always a stripped string
  (``34873`` stays ``"34873"``) and nothing is guessed by the file reader.
* Missing-value tokens (blank, ``-``, ``NA``, ``-99``, ``-999`` ..., user-configurable)
  become NaN, never 0. Text tokens apply to every column; numeric sentinels such as
  ``-99`` apply to attribute columns only (a coordinate or a dip of -99 is left alone and
  is the validator's business).
* Below-detection values such as ``<0.01`` follow one user-chosen rule: NaN, half the
  limit, or the limit. ``>100`` becomes the limit (or NaN), ``0,25`` is read as 0.25.
  Any other text in a mostly numeric column (``LNR``, ``IS`` ...) becomes NaN. Every
  kind is counted in the notes, by value.
* ``coordinate_order`` converts files whose coordinate columns are in another axis order
  (Surpac writes ``Y, X, Z``) at this boundary; inside, ``x`` is always east.
* ``length_unit`` records the unit of coordinates and depths. With
  ``convert_to_metres=True`` coordinates *and* depths are converted together.
* ``dip_convention`` is recorded; ``normalise_dip=True`` rewrites dips as negative-down.
  ``"auto"`` decides from the signs (all positive -> positive down) and says so.
* Collars given as longitude / latitude (decimal, text or degrees-minutes-seconds columns)
  are projected (``crs``, default the UTM zone of the data); see
  :mod:`bakumine.core.drillholes.coordinates`.
* Without a survey table, a collar table with ``azimuth`` and ``dip`` gives one station
  per hole at depth 0.
* A collar without ``max_depth`` gets it from the deepest interval of the hole.
* Numeric attributes are ``float64``; text attributes are strings.

Messy sources (one sheet per hole, two-row headers, depths in the sample number, QC rows,
several files per table) are described by a recipe: :mod:`bakumine.core.drillholes.recipe`.

The database is saved as a folder: one Parquet file per table + ``database.json``.
"""

from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .desurvey import DIP_CONVENTIONS, HoleTraces, desurvey

__all__ = [
    "ABOVE_DETECTION_RULES",
    "BELOW_DETECTION_RULES",
    "COLLAR_COLUMNS",
    "DEFAULT_MISSING_TOKENS",
    "INTERVAL_COLUMNS",
    "LENGTH_UNITS",
    "SURVEY_COLUMNS",
    "ColumnMapSuggestion",
    "DrillholeDatabase",
    "DrillholeMetadata",
    "ImportOptions",
    "TableSource",
    "import_drillholes",
    "import_table",
    "finish_database",
    "load_database",
    "read_raw_table",
    "suggest_column_map",
]

TableKind = Literal["collar", "survey", "interval"]
BelowDetection = Literal["nan", "half", "limit"]

COLLAR_COLUMNS: tuple[str, ...] = ("hole_id", "x", "y", "z", "max_depth")
SURVEY_COLUMNS: tuple[str, ...] = ("hole_id", "depth", "dip", "azimuth")
INTERVAL_COLUMNS: tuple[str, ...] = ("hole_id", "depth_from", "depth_to")

_CANONICAL: dict[str, tuple[str, ...]] = {
    "collar": COLLAR_COLUMNS,
    "survey": SURVEY_COLUMNS,
    "interval": INTERVAL_COLUMNS,
}
#: Columns that must be present after mapping (``max_depth`` can be derived).
_REQUIRED: dict[str, tuple[str, ...]] = {
    "collar": ("hole_id", "x", "y", "z"),
    "survey": SURVEY_COLUMNS,
    "interval": INTERVAL_COLUMNS,
}
#: Columns holding a length (converted together when the unit is converted).
_LENGTH_COLUMNS: dict[str, tuple[str, ...]] = {
    "collar": ("x", "y", "z", "max_depth"),
    "survey": ("depth",),
    "interval": ("depth_from", "depth_to"),
}

#: Metres per unit. ``ft`` is the international foot, ``ftUS`` the US survey foot.
LENGTH_UNITS: dict[str, float] = {"m": 1.0, "ft": 0.3048, "ftUS": 1200.0 / 3937.0}

BELOW_DETECTION_RULES: tuple[str, ...] = ("nan", "half", "limit")
ABOVE_DETECTION_RULES: tuple[str, ...] = ("limit", "nan")

#: Optional collar columns that give the position as longitude / latitude.
GEOGRAPHIC_COLUMNS: tuple[str, ...] = (
    "lon", "lat", "lon_deg", "lon_min", "lon_sec", "lat_deg", "lat_min", "lat_sec",
)  # fmt: skip
#: Attributes that are identifiers: always text, also when every cell is a number.
ID_COLUMNS: tuple[str, ...] = ("sample_id", "sample_type")

#: Default missing-value tokens. Text tokens are compared case-insensitively after
#: stripping; numeric tokens are compared by value on attribute columns only.
DEFAULT_MISSING_TOKENS: tuple[str | float, ...] = (
    "",
    "-",
    "--",
    "NA",
    "N/A",
    "#N/A",
    "NaN",
    "NULL",
    "None",
    -99,
    -999,
    -9999,
)

_FORMAT = "bakumine-drillholes"
_FORMAT_VERSION = 1
_METADATA_FILE = "database.json"
_TABLE_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_\-]*$")
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?"
_BELOW_DETECTION = re.compile(rf"^<\s*=?\s*({_NUMBER})$")
_ABOVE_DETECTION = re.compile(rf"^>\s*=?\s*({_NUMBER})$")
_DECIMAL_COMMA = re.compile(r"^([<>]?\s*=?\s*[+-]?\d+),(\d+)$")


# --------------------------------------------------------------------------------------
# options and containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ImportOptions:
    """How the source files are to be read. One set of options for all tables.

    ``length_unit``         unit of coordinates and depths in the files (``m``, ``ft``,
                            ``ftUS``).
    ``convert_to_metres``   convert coordinates and depths to metres on import.
    ``dip_convention``      sign convention of the dips in the files; ``"auto"`` takes
                            ``positive_down`` when no dip is negative (reported).
    ``normalise_dip``       store dips as ``negative_down`` (negates ``positive_down``).
    ``coordinate_order``    a permutation of ``"xyz"``. Letter *i* names the true axis held
                            by the column that the column map calls ``"xyz"[i]``. Use
                            ``"yxz"`` for a file whose first coordinate column (the one
                            mapped to ``x``) is the northing. ``"auto"`` swaps x and y
                            when the column mapped to ``x`` clearly holds northings
                            (or agrees with the longitude / latitude that way); reported.
    ``missing_tokens``      values meaning "no data" (see module docstring).
    ``below_detection``     rule for ``<limit`` cells: ``"nan"``, ``"half"`` or ``"limit"``.
    ``above_detection``     rule for ``>limit`` cells: ``"limit"`` or ``"nan"``.
    ``decimal_comma``       read ``0,25`` as 0.25.
    ``crs``                 projected CRS for collars given as longitude / latitude
                            (``"EPSG:32647"``); ``None`` or ``"utm"`` = the UTM zone of the
                            data. ``geographic_crs`` is the CRS of the longitudes.
    ``coordinate_tolerance``  metres; rows whose projected columns differ from their own
                            longitude / latitude by more are reported.
    ``derive_max_depth``    fill a missing ``max_depth`` from the hole's deepest interval.
    ``text_columns`` / ``numeric_columns``
                            canonical attribute names forced to text / to float64;
                            otherwise a column is numeric when every non-missing cell is.
    ``keep_unmapped``       keep source columns absent from the column map (under their
                            source header) instead of dropping them.
    ``delimiter``           CSV delimiter; ``None`` lets the reader detect it.
    """

    length_unit: str = "m"
    convert_to_metres: bool = False
    dip_convention: str = "negative_down"
    normalise_dip: bool = False
    coordinate_order: str = "xyz"
    missing_tokens: tuple[str | float, ...] = DEFAULT_MISSING_TOKENS
    below_detection: BelowDetection = "half"
    above_detection: str = "limit"
    decimal_comma: bool = True
    crs: str | None = None
    geographic_crs: str = "EPSG:4326"
    coordinate_tolerance: float = 30.0
    derive_max_depth: bool = True
    text_columns: tuple[str, ...] = ()
    numeric_columns: tuple[str, ...] = ()
    keep_unmapped: bool = True
    delimiter: str | None = ","
    encoding: str = "utf-8-sig"

    def __post_init__(self) -> None:
        if self.length_unit not in LENGTH_UNITS:
            raise ValueError(
                f"length_unit must be one of {tuple(LENGTH_UNITS)}, got {self.length_unit!r}"
            )
        if self.dip_convention not in (*DIP_CONVENTIONS, "auto"):
            raise ValueError(
                f"dip_convention must be one of {(*DIP_CONVENTIONS, 'auto')}, "
                f"got {self.dip_convention!r}"
            )
        if self.coordinate_order != "auto" and sorted(self.coordinate_order) != ["x", "y", "z"]:
            raise ValueError(
                "coordinate_order must be a permutation of 'xyz' or 'auto', "
                f"got {self.coordinate_order!r}"
            )
        if self.above_detection not in ABOVE_DETECTION_RULES:
            raise ValueError(
                f"above_detection must be one of {ABOVE_DETECTION_RULES}, "
                f"got {self.above_detection!r}"
            )
        if self.below_detection not in BELOW_DETECTION_RULES:
            raise ValueError(
                f"below_detection must be one of {BELOW_DETECTION_RULES}, "
                f"got {self.below_detection!r}"
            )
        both = set(self.text_columns) & set(self.numeric_columns)
        if both:
            raise ValueError(f"columns listed as both text and numeric: {sorted(both)}")


@dataclass(frozen=True)
class TableSource:
    """One source table: a CSV/XLSX path (or an already loaded DataFrame) + its column map.

    ``column_map`` maps *source column -> canonical name*. Source columns are matched by
    header (exact, then case-insensitive); with ``header=False`` they are 0-based column
    positions. ``sheet`` selects an XLSX sheet (name or 0-based index, default first).
    """

    source: str | Path | pd.DataFrame
    column_map: Mapping[str | int, str]
    sheet: str | int | None = None
    header: bool = True


@dataclass
class DrillholeMetadata:
    """Everything about a database that is not in the tables. JSON-serialisable."""

    length_unit: str = "m"
    dip_convention: str = "negative_down"
    source_length_unit: str = "m"
    source_dip_convention: str = "negative_down"
    coordinate_order: str = "xyz"
    #: table name -> {source column -> canonical name}
    column_maps: dict[str, dict[str, str]] = field(default_factory=dict)
    #: table name -> source file (or "<DataFrame>")
    sources: dict[str, str] = field(default_factory=dict)
    #: what the importer changed or could not read: dicts with table, column, kind, count
    notes: list[dict[str, Any]] = field(default_factory=list)
    #: projected CRS of the collars when they were converted from longitude / latitude
    crs: str | None = None
    #: the import recipe (see ``bakumine.core.drillholes.recipe``), for replay
    recipe: dict[str, Any] | None = None
    #: import reports: QC rows, sample-number depths, merge conflicts, coordinate check
    reports: dict[str, Any] = field(default_factory=dict)


@dataclass
class DrillholeDatabase:
    """Collar + survey + any number of named interval tables, with metadata."""

    collar: pd.DataFrame
    survey: pd.DataFrame
    intervals: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: DrillholeMetadata = field(default_factory=DrillholeMetadata)
    #: QC rows (blanks, standards, duplicates ...) set aside per interval table, as text
    qc: dict[str, pd.DataFrame] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(self.collar, COLLAR_COLUMNS, "collar")
        _require(self.survey, SURVEY_COLUMNS, "survey")
        for name, table in self.intervals.items():
            _check_table_name(name)
            _require(table, INTERVAL_COLUMNS, name)

    @property
    def length_unit(self) -> str:
        return self.metadata.length_unit

    @property
    def dip_convention(self) -> str:
        return self.metadata.dip_convention

    @property
    def tables(self) -> dict[str, pd.DataFrame]:
        """All tables by name: ``collar``, ``survey`` and every interval table."""
        return {"collar": self.collar, "survey": self.survey, **self.intervals}

    def attribute_columns(self, table: str) -> list[str]:
        """Columns of an interval table other than ``hole_id, depth_from, depth_to``."""
        return [c for c in self.intervals[table].columns if c not in INTERVAL_COLUMNS]

    def desurvey(self, *, method: str = "minimum_curvature") -> HoleTraces:
        """Desurvey with this database's dip convention."""
        return desurvey(
            self.collar, self.survey, method=method, dip_convention=self.dip_convention
        )

    def save(self, folder: str | Path) -> Path:
        """Write ``<folder>/collar.parquet``, ``survey.parquet``, ``interval_<name>.parquet``
        and ``database.json``. Returns the folder."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        files = {"collar": "collar.parquet", "survey": "survey.parquet"}
        files.update({name: f"interval_{name}.parquet" for name in self.intervals})
        for name, table in self.tables.items():
            table.to_parquet(folder / files[name], index=False)
        qc_files = {name: f"qc_{name}.parquet" for name in self.qc}
        for name, table in self.qc.items():
            table.to_parquet(folder / qc_files[name], index=False)
        document = {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "tables": {
                "collar": files["collar"],
                "survey": files["survey"],
                "intervals": {name: files[name] for name in self.intervals},
                "qc": qc_files,
            },
            "metadata": asdict(self.metadata),
        }
        (folder / _METADATA_FILE).write_text(
            json.dumps(document, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        return folder

    @classmethod
    def load(cls, folder: str | Path) -> DrillholeDatabase:
        """Read a folder written by :meth:`save`."""
        folder = Path(folder)
        document = json.loads((folder / _METADATA_FILE).read_text(encoding="utf-8"))
        if document.get("format") != _FORMAT:
            raise ValueError(f"{folder / _METADATA_FILE} is not a {_FORMAT} file")
        if document.get("version") != _FORMAT_VERSION:
            raise ValueError(f"unsupported {_FORMAT} version {document.get('version')!r}")
        tables = document["tables"]
        return cls(
            collar=pd.read_parquet(folder / tables["collar"]),
            survey=pd.read_parquet(folder / tables["survey"]),
            intervals={
                name: pd.read_parquet(folder / file) for name, file in tables["intervals"].items()
            },
            metadata=DrillholeMetadata(**document["metadata"]),
            qc={
                name: pd.read_parquet(folder / file)
                for name, file in tables.get("qc", {}).items()
            },
        )


def load_database(folder: str | Path) -> DrillholeDatabase:
    """Read a database folder written by :meth:`DrillholeDatabase.save`."""
    return DrillholeDatabase.load(folder)


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} table is missing columns {missing}; has {list(frame.columns)}")


#: collar columns besides the canonical ones that are read as numbers without sentinels
_COLLAR_NUMBERS: tuple[str, ...] = (*GEOGRAPHIC_COLUMNS, "azimuth", "dip")


def has_geographic(columns: Iterable[str]) -> bool:
    """True when the columns give longitude and latitude (decimal / text or as degrees ...)."""
    names = set(columns)
    return {"lon", "lat"} <= names or {"lon_deg", "lat_deg"} <= names


def missing_required(kind: str, present: Iterable[str]) -> list[str]:
    """Required canonical columns that ``present`` lacks. A collar table with longitude and
    latitude needs no ``x`` and ``y``."""
    names = set(present)
    missing = [c for c in _REQUIRED[kind] if c not in names]
    if kind == "collar" and has_geographic(names):
        missing = [c for c in missing if c not in ("x", "y")]
    return missing


def _check_table_name(name: str) -> None:
    if not _TABLE_NAME.match(name) or name in ("collar", "survey"):
        raise ValueError(
            f"invalid interval table name {name!r}: use letters, digits, '_' or '-', "
            "and not 'collar' or 'survey'"
        )


# --------------------------------------------------------------------------------------
# reading raw text tables
# --------------------------------------------------------------------------------------
def _cell_text(value: object) -> str:
    """Spreadsheet cell -> text. Integral numbers lose the ``.0`` so IDs stay clean."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, bool | np.bool_):
        return str(bool(value))
    if isinstance(value, int | np.integer):
        return str(int(value))
    if isinstance(value, float | np.floating):
        number = float(value)
        if np.isnan(number):
            return ""
        if number.is_integer() and abs(number) < 1e15:
            return str(int(number))
        return repr(number)
    return str(value)


def read_raw_table(
    path: str | Path,
    *,
    sheet: str | int | None = None,
    header: bool = True,
    delimiter: str | None = ",",
    encoding: str = "utf-8-sig",
) -> pd.DataFrame:
    """Read a CSV or XLSX / XLS file with every cell as text (blank cells are ``""``).

    Nothing is interpreted here: no NaN detection, no number parsing. Without a header
    the columns are named ``0, 1, 2, ...``. Rows that are entirely blank are dropped.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    head = 0 if header else None
    if suffix in (".xlsx", ".xlsm", ".xls"):
        frame = pd.read_excel(
            path,
            sheet_name=0 if sheet is None else sheet,
            header=head,
            dtype=object,
            na_filter=False,
            engine="xlrd" if suffix == ".xls" else "openpyxl",
        )
    elif suffix in (".csv", ".txt", ".tsv", ".dat"):
        frame = pd.read_csv(
            path,
            header=head,
            dtype=str,
            keep_default_na=False,
            na_filter=False,
            sep=delimiter,
            engine="python" if delimiter is None else "c",
            encoding=encoding,
        ).astype(object)
    else:
        raise ValueError(f"unsupported table file type {suffix!r}: {path}")
    return _clean_raw(frame, header)


def _clean_raw(frame: pd.DataFrame, header: bool) -> pd.DataFrame:
    frame = frame.copy()
    frame.columns = [str(c).strip() for c in frame.columns] if header else range(frame.shape[1])
    if header and len(set(frame.columns)) != frame.shape[1]:
        raise ValueError(f"duplicate column headers: {list(frame.columns)}")
    text = {c: [_cell_text(v) for v in frame[c].tolist()] for c in frame.columns}
    frame = pd.DataFrame(text, columns=frame.columns, dtype=object)
    blank = np.ones(len(frame), dtype=bool)
    for column in frame.columns:
        blank &= frame[column].str.strip().to_numpy(dtype=object) == ""
    return frame.loc[~blank].reset_index(drop=True)


# --------------------------------------------------------------------------------------
# parsing cells
# --------------------------------------------------------------------------------------
def _split_tokens(tokens: Iterable[str | float]) -> tuple[set[str], np.ndarray]:
    """Missing tokens -> (case-folded text tokens, numeric sentinel values)."""
    text: set[str] = set()
    numbers: list[float] = []
    for token in tokens:
        if isinstance(token, str):
            stripped = token.strip()
            try:
                number = float(stripped)
            except ValueError:
                text.add(stripped.casefold())
            else:
                if np.isnan(number):
                    text.add(stripped.casefold())
                else:
                    numbers.append(number)
        else:
            numbers.append(float(token))
    text.add("")
    return text, np.asarray(numbers, dtype=np.float64)


@dataclass
class _Parsed:
    values: np.ndarray  # float64
    unparseable: np.ndarray  # bool, cells that are neither missing nor a number
    below_detection: int
    sentinels: int
    above_detection: int = 0
    decimal_commas: int = 0
    #: non-blank missing tokens met ("-", "n/a" ...), by value
    missing: dict[str, int] = field(default_factory=dict)


def _parse_numeric(
    cells: pd.Series,
    text_tokens: set[str],
    sentinels: np.ndarray,
    rule: str,
    above: str = "limit",
    decimal_comma: bool = True,
) -> _Parsed:
    """Text cells -> float64. Missing -> NaN; ``<limit`` by ``rule``; ``>limit`` by ``above``;
    ``0,25`` -> 0.25; anything else -> NaN + flag."""
    stripped = cells.str.strip()
    commas = 0
    if decimal_comma:
        has = stripped.str.contains(",", regex=False).to_numpy(dtype=bool)
        if has.any():
            fixed = stripped[has].str.replace(_DECIMAL_COMMA, r"\1.\2", regex=True)
            commas = int((fixed != stripped[has]).sum())
            stripped = stripped.copy()
            stripped[has] = fixed
    missing = stripped.str.casefold().isin(text_tokens).to_numpy(dtype=bool)
    tokens = Counter(stripped[missing & (stripped != "").to_numpy(dtype=bool)].tolist())
    values = np.array(
        pd.to_numeric(stripped.where(~missing), errors="coerce"), dtype=np.float64, copy=True
    )
    values[missing] = np.nan
    candidates = np.flatnonzero(np.isnan(values) & ~missing)
    unparseable = np.zeros(len(cells), dtype=bool)
    below = over = 0
    for i in candidates:  # only the cells pandas could not read as a number
        text = stripped.iat[i]
        match = _BELOW_DETECTION.match(text)
        if match is not None:
            limit = float(match.group(1))
            below += 1
            values[i] = {"nan": np.nan, "half": 0.5 * limit, "limit": limit}[rule]
            continue
        match = _ABOVE_DETECTION.match(text)
        if match is not None:
            over += 1
            values[i] = float(match.group(1)) if above == "limit" else np.nan
            continue
        if text.casefold() not in ("nan", "+nan", "-nan"):
            unparseable[i] = True
    hit = np.isin(values, sentinels) if sentinels.size else np.zeros(len(cells), dtype=bool)
    values[hit] = np.nan
    return _Parsed(values, unparseable, below, int(hit.sum()), over, commas, dict(tokens))


def _parse_text(cells: pd.Series, text_tokens: set[str]) -> pd.Series:
    stripped = cells.str.strip()
    missing = stripped.str.casefold().isin(text_tokens)
    return stripped.where(~missing, None).astype("str")


def _resolve_column_map(
    headers: list[Any], column_map: Mapping[str | int, str], header: bool, name: str
) -> dict[Any, str]:
    """Match the user's source names to the real headers. Returns header -> canonical."""
    resolved: dict[Any, str] = {}
    folded = {str(h).strip().casefold(): h for h in headers}
    for source, target in column_map.items():
        if not header:
            try:
                position = int(source)
            except (TypeError, ValueError):
                raise ValueError(
                    f"{name}: without a header the column map keys must be 0-based "
                    f"positions, got {source!r}"
                ) from None
            if not 0 <= position < len(headers):
                raise ValueError(
                    f"{name}: column position {position} is outside 0..{len(headers) - 1}"
                )
            found = headers[position]
        elif source in headers:
            found = source
        elif str(source).strip().casefold() in folded:
            found = folded[str(source).strip().casefold()]
        else:
            raise ValueError(f"{name}: source column {source!r} not found; file has {headers}")
        if found in resolved:
            raise ValueError(f"{name}: source column {found!r} is mapped twice")
        resolved[found] = str(target)
    targets = list(resolved.values())
    duplicates = sorted({t for t in targets if targets.count(t) > 1})
    if duplicates:
        raise ValueError(f"{name}: several source columns are mapped to {duplicates}")
    return resolved


def import_table(
    source: str | Path | pd.DataFrame,
    kind: TableKind,
    column_map: Mapping[str | int, str],
    options: ImportOptions | None = None,
    *,
    sheet: str | int | None = None,
    header: bool = True,
    name: str | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Import one table. Returns ``(table, notes)``.

    Applies the column map, missing tokens, the below-detection rule and column typing.
    It does **not** apply ``coordinate_order``, unit conversion or dip normalisation; those
    act on the database as a whole in :func:`import_drillholes`.
    """
    options = options or ImportOptions()
    name = name or kind
    if kind not in _CANONICAL:
        raise ValueError(f"kind must be one of {tuple(_CANONICAL)}, got {kind!r}")
    if isinstance(source, pd.DataFrame):
        raw = _clean_raw(source, header)
    else:
        raw = read_raw_table(
            source,
            sheet=sheet,
            header=header,
            delimiter=options.delimiter,
            encoding=options.encoding,
        )
    headers = list(raw.columns)
    resolved = _resolve_column_map(headers, column_map, header, name)
    canonical = _CANONICAL[kind]
    present = set(resolved.values())
    missing = missing_required(kind, present)
    if missing:
        raise ValueError(
            f"{name}: the column map gives no source for {missing}; mapped: {sorted(present)}"
        )

    # Output column order: canonical first, then attributes in file order.
    names: dict[Any, str] = {}
    for column in headers:
        if column in resolved:
            names[column] = resolved[column]
        elif options.keep_unmapped:
            if str(column) in present or str(column) in canonical:
                raise ValueError(
                    f"{name}: unmapped source column {column!r} collides with a mapped name; "
                    "map it to another name or set keep_unmapped=False"
                )
            names[column] = str(column)
    by_target = {target: column for column, target in names.items()}
    order = [c for c in canonical if c in by_target]
    order += [t for t in names.values() if t not in canonical]

    text_tokens, sentinels = _split_tokens(options.missing_tokens)
    no_sentinels = np.empty(0, dtype=np.float64)
    notes: list[dict[str, Any]] = []

    def note(column: str, what: str, count: int, **extra: Any) -> None:
        if count:
            notes.append({"table": name, "column": column, "kind": what, "count": count, **extra})

    out: dict[str, Any] = {}
    for target in order:
        cells = raw[by_target[target]]
        if target == "hole_id":
            out[target] = cells.str.strip().astype("str")
            continue
        if target in canonical or (kind == "collar" and target in _COLLAR_NUMBERS):
            # coordinates, depths, angles: numeric, no sentinels
            parsed = _parse_numeric(
                cells, text_tokens, no_sentinels, "nan", "nan", options.decimal_comma
            )
            if kind == "collar" and target in ("lon", "lat") and parsed.unparseable.any():
                out[target] = _parse_text(cells, text_tokens)  # "98 47 11.0", parsed later
                continue
            note(
                target,
                "unparseable_set_to_nan",
                int(parsed.unparseable.sum()),
                examples=sorted(set(cells[parsed.unparseable].str.strip()))[:5],
            )
            note(
                target,
                "below_detection_set_to_nan",
                parsed.below_detection + parsed.above_detection,
            )
            note(target, "decimal_comma", parsed.decimal_commas)
            out[target] = parsed.values
            continue
        if target in options.text_columns or (
            target in ID_COLUMNS and target not in options.numeric_columns
        ):
            out[target] = _parse_text(cells, text_tokens)
            continue
        parsed = _parse_numeric(
            cells,
            text_tokens,
            sentinels,
            options.below_detection,
            options.above_detection,
            options.decimal_comma,
        )
        n_bad = int(parsed.unparseable.sum())
        n_numbers = (
            int(np.isfinite(parsed.values).sum())
            + parsed.sentinels
            + (parsed.below_detection if options.below_detection == "nan" else 0)
            + (parsed.above_detection if options.above_detection == "nan" else 0)
        )
        if n_bad and target not in options.numeric_columns and n_numbers <= n_bad:
            out[target] = _parse_text(cells, text_tokens)  # a text column
            continue
        # A numeric column: odd cells ("LNR", "IS") become NaN and are counted by value,
        # so that compositing and estimation still see numbers.
        odd = Counter(cells[parsed.unparseable].str.strip().tolist())
        note(
            target,
            "unparseable_set_to_nan",
            n_bad,
            examples=sorted(odd)[:5],
            **({"values": dict(odd.most_common(20))} if target not in options.numeric_columns
               else {}),
        )  # fmt: skip
        note(target, "below_detection", parsed.below_detection, rule=options.below_detection)
        note(target, "above_detection", parsed.above_detection, rule=options.above_detection)
        note(target, "decimal_comma", parsed.decimal_commas)
        note(target, "sentinel_set_to_nan", parsed.sentinels)
        note(
            target,
            "missing_token",
            sum(parsed.missing.values()),
            values=dict(sorted(parsed.missing.items())),
        )
        out[target] = parsed.values
    return pd.DataFrame(out, columns=order), notes


# --------------------------------------------------------------------------------------
# whole database
# --------------------------------------------------------------------------------------
def _source_label(source: object) -> str:
    return "<DataFrame>" if isinstance(source, pd.DataFrame) else str(source)


def import_drillholes(
    collar: TableSource,
    survey: TableSource,
    intervals: Mapping[str, TableSource] | None = None,
    *,
    options: ImportOptions | None = None,
) -> DrillholeDatabase:
    """Import collar, survey and any number of named interval tables into one database.

    ``intervals`` maps a table name (``"assay"``, ``"geology"``, ...) to its source.
    See :class:`ImportOptions` for units, axis order, dip convention, missing values and
    below-detection handling. Row order of every table is the file order.
    """
    options = options or ImportOptions()
    intervals = dict(intervals or {})
    for name in intervals:
        _check_table_name(name)

    metadata = DrillholeMetadata(
        length_unit=options.length_unit,
        source_length_unit=options.length_unit,
    )
    tables: dict[str, pd.DataFrame] = {}
    specs = {"collar": ("collar", collar), "survey": ("survey", survey)}
    specs.update({name: ("interval", spec) for name, spec in intervals.items()})
    for name, (kind, spec) in specs.items():
        table, notes = import_table(
            spec.source,
            kind,  # type: ignore[arg-type]
            spec.column_map,
            options,
            sheet=spec.sheet,
            header=spec.header,
            name=name,
        )
        tables[name] = table
        metadata.notes.extend(notes)
        metadata.column_maps[name] = {str(k): str(v) for k, v in spec.column_map.items()}
        metadata.sources[name] = _source_label(spec.source)

    collar_table = tables.pop("collar")
    survey_table = tables.pop("survey")
    return finish_database(collar_table, survey_table, tables, options, metadata)


def finish_database(
    collar_table: pd.DataFrame,
    survey_table: pd.DataFrame | None,
    tables: dict[str, pd.DataFrame],
    options: ImportOptions,
    metadata: DrillholeMetadata,
    qc: dict[str, pd.DataFrame] | None = None,
) -> DrillholeDatabase:
    """Typed tables -> database: everything that acts on the tables together.

    Longitude / latitude -> projected coordinates and the axis order
    (:mod:`.coordinates`), ``max_depth`` from the deepest interval, a survey from the
    collar's ``azimuth`` / ``dip`` when ``survey_table`` is None, the dip convention, and
    the unit conversion. What was decided or changed is added to ``metadata``.
    """
    from .coordinates import resolve_collar_coordinates

    collar_table, order = resolve_collar_coordinates(collar_table, options, metadata)
    metadata.coordinate_order = order
    # Axis order: the column mapped as "xyz"[i] holds the true axis coordinate_order[i].
    if order != "xyz":
        true_axis = dict(zip("xyz", order, strict=True))
        collar_table = collar_table.rename(columns=true_axis)

    # max_depth: fill what the collar does not give from the deepest interval.
    if "max_depth" not in collar_table.columns:
        collar_table["max_depth"] = np.nan
    lacking = collar_table["max_depth"].isna().to_numpy()
    if options.derive_max_depth and lacking.any() and tables:
        deepest = pd.concat(
            [t[["hole_id", "depth_to"]] for t in tables.values()], ignore_index=True
        )
        deepest = deepest.groupby("hole_id", sort=False)["depth_to"].max()
        derived = collar_table["hole_id"].map(deepest).to_numpy(dtype=np.float64)
        fill = lacking & np.isfinite(derived)
        collar_table.loc[fill, "max_depth"] = derived[fill]
        if fill.any():
            metadata.notes.append(
                {
                    "table": "collar",
                    "column": "max_depth",
                    "kind": "derived_from_deepest_interval",
                    "count": int(fill.sum()),
                }
            )
    extras = [c for c in collar_table.columns if c not in COLLAR_COLUMNS]
    collar_table = collar_table[[*COLLAR_COLUMNS, *extras]]

    if survey_table is None:
        survey_table = _survey_from_collar(collar_table, metadata)

    convention = options.dip_convention
    dips = survey_table["dip"].dropna().to_numpy(dtype=np.float64)
    all_down_positive = bool(dips.size and (dips >= 0).all() and (dips > 0).any())
    if convention == "auto":
        mixed = bool((dips > 0).any() and (dips < 0).any())
        convention = "positive_down" if all_down_positive else "negative_down"
        kind = "dip_signs_mixed_assumed_negative_down" if mixed else f"dip_assumed_{convention}"
        metadata.notes.append(
            {"table": "survey", "column": "dip", "kind": kind, "count": int(dips.size)}
        )
    elif convention == "negative_down" and all_down_positive:
        metadata.notes.append(
            {
                "table": "survey",
                "column": "dip",
                "kind": "all_dips_positive_but_convention_negative_down",
                "count": int(dips.size),
                "hint": "the holes point up; use dip_convention 'positive_down' or 'auto'",
            }
        )
    metadata.dip_convention = metadata.source_dip_convention = convention

    if options.convert_to_metres and options.length_unit != "m":
        factor = LENGTH_UNITS[options.length_unit]
        collar_table[list(_LENGTH_COLUMNS["collar"])] *= factor
        survey_table[list(_LENGTH_COLUMNS["survey"])] *= factor
        for table in tables.values():
            table[list(_LENGTH_COLUMNS["interval"])] *= factor
        metadata.length_unit = "m"

    if options.normalise_dip and convention != "negative_down":
        survey_table["dip"] = -survey_table["dip"]
        metadata.dip_convention = "negative_down"

    return DrillholeDatabase(
        collar=collar_table,
        survey=survey_table,
        intervals=tables,
        metadata=metadata,
        qc=dict(qc or {}),
    )


def _survey_from_collar(collar: pd.DataFrame, metadata: DrillholeMetadata) -> pd.DataFrame:
    """One station at depth 0 per hole from the collar's ``azimuth`` and ``dip``."""
    if not {"azimuth", "dip"} <= set(collar.columns):
        raise ValueError(
            "there is no survey table, and the collar table has no azimuth and dip columns "
            "to make one from"
        )
    known = collar[["azimuth", "dip"]].notna().all(axis=1).to_numpy()
    survey = pd.DataFrame(
        {
            "hole_id": collar["hole_id"].to_numpy()[known],
            "depth": 0.0,
            "dip": collar["dip"].to_numpy(dtype=np.float64)[known],
            "azimuth": collar["azimuth"].to_numpy(dtype=np.float64)[known],
        }
    )
    metadata.sources["survey"] = "<collar azimuth, dip>"
    metadata.notes.append(
        {"table": "survey", "column": "*", "kind": "generated_from_collar", "count": len(survey)}
    )
    without = collar["hole_id"].to_numpy()[~known].tolist()
    if without:
        metadata.notes.append(
            {
                "table": "survey",
                "column": "*",
                "kind": "collar_without_azimuth_or_dip",
                "count": len(without),
                "examples": without[:5],
            }
        )
    metadata.reports["survey"] = {
        "generated_from_collar": int(len(survey)),
        "holes_without_azimuth_or_dip": without[:100],
    }
    return survey


# --------------------------------------------------------------------------------------
# column-map suggestion
# --------------------------------------------------------------------------------------
#: Common header spellings, normalised (lower case, letters and digits only, unit words
#: removed; Latin look-alike letters inside Cyrillic words are folded). English, Russian and
#: Mongolian. Earlier entries win ties.
_HOLE = (
    "holeid", "bhid", "hole", "dhid", "holeno", "holenumber", "holename", "drillhole",
    "drillholeid", "ddh", "borehole", "boreholeid", "wellid", "well", "siteid", "id",
    "name", "numhole",
    "скважина", "скв", "номерскважины", "скважины", "nскважины", "выработка",
    "цооног", "цооногдугаар", "цооногийндугаар", "цооногийнномер", "цооногийннэр",
    "trench", "trenchid", "channel", "channelid", "канава", "суваг", "сувгийндугаар",
)  # fmt: skip
_AZIMUTH = (
    "azimuth", "azi", "az", "azim", "bearing", "brg", "azimuthgrid", "azimuthtrue",
    "gridazimuth", "trueazimuth", "direction", "азимут",
)  # fmt: skip
_DIP = (
    "dip", "inclination", "incl", "inc", "plunge",
    "налуу", "цооногийнналуу", "налуугийнөнцөг", "уголнаклона", "наклон", "уголпадения",
)  # fmt: skip
_SYNONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    "collar": {
        "hole_id": _HOLE,
        "x": (
            "x", "east", "easting", "eastings", "xcollar", "collarx", "xcoord", "xcoordinate",
            "collareast", "collareasting", "utme", "utmeast", "utmeasting", "e",
            "восток", "зүүн", "utmlongitude", "utmlon", "utmдолгота", "utmуртраг",
        ),
        "y": (
            "y", "north", "northing", "northings", "ycollar", "collary", "ycoord",
            "ycoordinate", "collarnorth", "collarnorthing", "utmn", "utmnorth", "utmnorthing",
            "n", "север", "хойд", "utmlatitude", "utmlat", "utmширота", "utmөргөрөг",
        ),
        "z": (
            "z", "rl", "elevation", "elev", "zcollar", "collarz", "collarrl", "zcoord",
            "zcoordinate", "collarelevation", "reducedlevel", "altitude", "level", "height",
            "h", "высота", "отметка", "абсотметка", "абсолютнаяотметка", "альтитуда",
            "өндөршил", "өндөр", "өндөрлөг",
        ),
        "max_depth": (
            "maxdepth", "depth", "totaldepth", "eoh", "enddepth", "finaldepth", "holedepth",
            "endofhole", "td", "length", "holelength", "totallength",
            "глубина", "глубинаскважины", "гүн", "цооногийнгүн",
        ),
        "lon": ("lon", "long", "longitude", "lng", "долгота", "уртраг"),
        "lat": ("lat", "latitude", "широта", "өргөрөг"),
        "azimuth": _AZIMUTH,
        "dip": _DIP,
    },
    "survey": {
        "hole_id": _HOLE,
        "depth": (
            "depth", "at", "surveydepth", "sdepth", "measureddepth", "md", "downholedepth",
            "distance", "station", "stationdepth", "глубина", "гүн",
        ),
        "dip": _DIP,
        "azimuth": _AZIMUTH,
    },
    "interval": {
        "hole_id": _HOLE,
        "depth_from": (
            "depthfrom", "from", "fromdepth", "sampfrom", "samplefrom", "intervalfrom",
            "startdepth", "start", "top", "topdepth",
            "от", "аас", "ээс", "оос", "өөс", "эхлэл", "интервалот", "интервалаас",
            "interval", "интервал",
        ),
        "depth_to": (
            "depthto", "to", "todepth", "sampto", "sampleto", "intervalto", "enddepth", "end",
            "bottom", "bottomdepth", "base",
            "до", "хүртэл", "төгсгөл", "интервалдо", "интервалхүртэл",
            "interval2", "интервал2",
        ),
        "sample_id": (
            "sampleid", "sample", "sampleno", "samplenumber", "samplenum", "sampid",
            "проба", "номерпробы", "nпробы", "пробы",
            "дээж", "дээжийндугаар", "сорьц", "сорьцдугаар", "сорьцындугаар", "сорьцийндугаар",
        ),
        "sample_type": (
            "sampletype", "samptype", "type", "qctype", "типпробы", "видпробы",
            "сорьцынтөрөл", "сорьцийнтөрөл", "дээжийнтөрөл", "төрөл",
        ),
        "sample_length": (
            "samplelength", "length", "len", "длина", "длинапробы", "мощность",
            "сорьцынурт", "сорьцийнурт", "дээжийнурт", "урт",
        ),
    },
}
_UNIT_WORDS = frozenset(
    {
        "m", "mtr", "metre", "metres", "meter", "meters", "ft", "feet", "deg", "degree",
        "degrees", "м", "т", "метр", "метров", "град", "градус", "градусов",
    }
)  # fmt: skip
_MIN_FUZZY_LENGTH = 4
_CYRILLIC = re.compile(r"[Ѐ-ӿ]")
#: Latin letters typed in place of the Cyrillic ones that look the same
_LOOKALIKES = str.maketrans("aceopxykmhbt", "асеорхукмнвт")
#: the longitude / latitude columns of a degrees-minutes-seconds triple
_DMS_PARTS = {"lon": ("lon_deg", "lon_min", "lon_sec"), "lat": ("lat_deg", "lat_min", "lat_sec")}


@dataclass
class ColumnMapSuggestion:
    """A proposed column map. ``mapping`` is a plain dict the user can edit and pass on.

    ``missing``    required canonical columns with no source column found.
    ``unmatched``  source columns left out of ``mapping`` (collar/survey only; for an
                   interval table every other column is proposed as an attribute).
    ``scores``     canonical name -> match score in 0..1 (1 = known spelling).
    """

    table: str
    mapping: dict[str, str]
    missing: list[str]
    unmatched: list[str]
    scores: dict[str, float]


def _words(header: object) -> list[str]:
    words = [w for w in re.split(r"[\W_]+", str(header).casefold()) if w]
    return [w.translate(_LOOKALIKES) if _CYRILLIC.search(w) else w for w in words]


def _normalise_header(header: object) -> str:
    words = _words(header)
    kept = [w for w in words if w not in _UNIT_WORDS] or words
    return "".join(kept)


def _attribute_name(header: object) -> str:
    name = re.sub(r"[\W_]+", "_", str(header).strip().casefold()).strip("_")
    return name or "column"


def suggest_column_map(
    headers: Iterable[object],
    table: TableKind = "collar",
    *,
    cutoff: float = 0.8,
) -> ColumnMapSuggestion:
    """Propose *source column -> canonical name* from the headers of one table.

    Known spellings (``BHID``, ``HOLEID``, ``EAST``, ``NORTH``, ``RL``, ``FROM``, ``TO``,
    ``AZI``, ``DIP``, ``скважина``, ``от``, ``до``, ``Цооног``, ``аас``, ``хүртэл`` ...) match
    exactly after normalisation; other headers of 4 or more characters match fuzzily when
    their similarity to a known spelling reaches ``cutoff``. Each canonical name gets at
    most one source column. A collar table may also give ``lon`` / ``lat`` (three columns in
    a row under one longitude header become ``lon_deg, lon_min, lon_sec``) and
    ``azimuth`` / ``dip``; an interval table ``sample_id``, ``sample_type`` and
    ``sample_length``. For an interval table the remaining columns are proposed as
    attributes under a lower-case name. The result is a proposal: check it, edit
    ``.mapping``, then pass it to :class:`TableSource`.
    """
    if table not in _SYNONYMS:
        raise ValueError(f"table must be one of {tuple(_SYNONYMS)}, got {table!r}")
    headers = [str(h).strip() for h in headers]
    candidates: list[tuple[float, int, int, str, str]] = []
    for position, header in enumerate(headers):
        normal = _normalise_header(header)
        if not normal:
            continue
        for canonical, spellings in _SYNONYMS[table].items():
            if normal in spellings:
                candidates.append((1.0, spellings.index(normal), position, header, canonical))
            elif len(normal) >= _MIN_FUZZY_LENGTH:
                ratios = [
                    difflib.SequenceMatcher(None, normal, s).ratio()
                    for s in spellings
                    if len(s) >= _MIN_FUZZY_LENGTH
                ]
                best = max(ratios, default=0.0)
                if best >= cutoff:
                    candidates.append((best, len(spellings), position, header, canonical))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))
    chosen: dict[str, str] = {}  # canonical -> header
    scores: dict[str, float] = {}
    used: set[str] = set()
    for score, _, _, header, canonical in candidates:
        if canonical in chosen or header in used:
            continue
        chosen[canonical] = header
        scores[canonical] = round(score, 3)
        used.add(header)

    by_header = {header: canonical for canonical, header in chosen.items()}
    # "Longitude", "Longitude 2", "Longitude 3": degrees, minutes, seconds
    for whole, parts in _DMS_PARTS.items():
        header = chosen.get(whole)
        if header is None:
            continue
        rest = [f"{header} 2", f"{header} 3"]
        if all(h in headers and h not in by_header for h in rest):
            for part, name in zip(parts, [header, *rest], strict=True):
                by_header[name] = part
            chosen[parts[0]] = chosen.pop(whole)
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    taken = set(_CANONICAL[table]) | set(_SYNONYMS[table])
    for header in headers:
        if header in by_header:
            mapping[header] = by_header[header]
        elif table == "interval" and header not in mapping:
            base = _attribute_name(header)
            name, n = base, 2
            while name in taken:
                name, n = f"{base}_{n}", n + 1
            taken.add(name)
            mapping[header] = name
        else:
            unmatched.append(header)
    missing = missing_required(table, by_header.values())
    return ColumnMapSuggestion(table, mapping, missing, unmatched, scores)
