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
  limit, or the limit.
* ``coordinate_order`` converts files whose coordinate columns are in another axis order
  (Surpac writes ``Y, X, Z``) at this boundary; inside, ``x`` is always east.
* ``length_unit`` records the unit of coordinates and depths. With
  ``convert_to_metres=True`` coordinates *and* depths are converted together.
* ``dip_convention`` is recorded; ``normalise_dip=True`` rewrites dips as negative-down.
* A collar without ``max_depth`` gets it from the deepest interval of the hole.
* Numeric attributes are ``float64``; text attributes are strings.

The database is saved as a folder: one Parquet file per table + ``database.json``.
"""

from __future__ import annotations

import difflib
import json
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd

from .desurvey import DIP_CONVENTIONS, HoleTraces, desurvey

__all__ = [
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


# --------------------------------------------------------------------------------------
# options and containers
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class ImportOptions:
    """How the source files are to be read. One set of options for all tables.

    ``length_unit``         unit of coordinates and depths in the files (``m``, ``ft``,
                            ``ftUS``).
    ``convert_to_metres``   convert coordinates and depths to metres on import.
    ``dip_convention``      sign convention of the dips in the files.
    ``normalise_dip``       store dips as ``negative_down`` (negates ``positive_down``).
    ``coordinate_order``    a permutation of ``"xyz"``. Letter *i* names the true axis held
                            by the column that the column map calls ``"xyz"[i]``. Use
                            ``"yxz"`` for a file whose first coordinate column (the one
                            mapped to ``x``) is the northing.
    ``missing_tokens``      values meaning "no data" (see module docstring).
    ``below_detection``     rule for ``<limit`` cells: ``"nan"``, ``"half"`` or ``"limit"``.
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
        if self.dip_convention not in DIP_CONVENTIONS:
            raise ValueError(
                f"dip_convention must be one of {DIP_CONVENTIONS}, got {self.dip_convention!r}"
            )
        if sorted(self.coordinate_order) != ["x", "y", "z"]:
            raise ValueError(
                f"coordinate_order must be a permutation of 'xyz', got {self.coordinate_order!r}"
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


@dataclass
class DrillholeDatabase:
    """Collar + survey + any number of named interval tables, with metadata."""

    collar: pd.DataFrame
    survey: pd.DataFrame
    intervals: dict[str, pd.DataFrame] = field(default_factory=dict)
    metadata: DrillholeMetadata = field(default_factory=DrillholeMetadata)

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
        document = {
            "format": _FORMAT,
            "version": _FORMAT_VERSION,
            "tables": {
                "collar": files["collar"],
                "survey": files["survey"],
                "intervals": {name: files[name] for name in self.intervals},
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
        )


def load_database(folder: str | Path) -> DrillholeDatabase:
    """Read a database folder written by :meth:`DrillholeDatabase.save`."""
    return DrillholeDatabase.load(folder)


def _require(frame: pd.DataFrame, columns: Iterable[str], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} table is missing columns {missing}; has {list(frame.columns)}")


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
    """Read a CSV or XLSX file with every cell as text (blank cells are ``""``).

    Nothing is interpreted here: no NaN detection, no number parsing. Without a header
    the columns are named ``0, 1, 2, ...``. Rows that are entirely blank are dropped.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    head = 0 if header else None
    if suffix in (".xlsx", ".xlsm"):
        frame = pd.read_excel(
            path,
            sheet_name=0 if sheet is None else sheet,
            header=head,
            dtype=object,
            na_filter=False,
            engine="openpyxl",
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


def _parse_numeric(
    cells: pd.Series,
    text_tokens: set[str],
    sentinels: np.ndarray,
    rule: str,
) -> _Parsed:
    """Text cells -> float64. Missing -> NaN; ``<limit`` by ``rule``; garbage -> NaN + flag."""
    stripped = cells.str.strip()
    missing = stripped.str.casefold().isin(text_tokens).to_numpy(dtype=bool)
    values = np.array(
        pd.to_numeric(stripped.where(~missing), errors="coerce"), dtype=np.float64, copy=True
    )
    values[missing] = np.nan
    candidates = np.flatnonzero(np.isnan(values) & ~missing)
    unparseable = np.zeros(len(cells), dtype=bool)
    below = 0
    for i in candidates:  # only the cells pandas could not read as a number
        match = _BELOW_DETECTION.match(stripped.iat[i])
        if match is None:
            unparseable[i] = True
            continue
        limit = float(match.group(1))
        below += 1
        values[i] = {"nan": np.nan, "half": 0.5 * limit, "limit": limit}[rule]
    hit = np.isin(values, sentinels) if sentinels.size else np.zeros(len(cells), dtype=bool)
    values[hit] = np.nan
    return _Parsed(values, unparseable, below, int(hit.sum()))


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
    missing = [c for c in _REQUIRED[kind] if c not in present]
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
        if target in canonical:  # coordinates, depths, angles: numeric, no sentinels
            parsed = _parse_numeric(cells, text_tokens, no_sentinels, "nan")
            note(target, "unparseable_set_to_nan", int(parsed.unparseable.sum()))
            note(target, "below_detection_set_to_nan", parsed.below_detection)
            out[target] = parsed.values
            continue
        if target in options.text_columns:
            out[target] = _parse_text(cells, text_tokens)
            continue
        parsed = _parse_numeric(cells, text_tokens, sentinels, options.below_detection)
        if parsed.unparseable.any() and target not in options.numeric_columns:
            out[target] = _parse_text(cells, text_tokens)
            continue
        note(
            target,
            "unparseable_set_to_nan",
            int(parsed.unparseable.sum()),
            examples=sorted(set(cells[parsed.unparseable].str.strip()))[:5],
        )
        note(target, "below_detection", parsed.below_detection, rule=options.below_detection)
        note(target, "sentinel_set_to_nan", parsed.sentinels)
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
        dip_convention=options.dip_convention,
        source_length_unit=options.length_unit,
        source_dip_convention=options.dip_convention,
        coordinate_order=options.coordinate_order,
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

    # Axis order: the column mapped as "xyz"[i] holds the true axis coordinate_order[i].
    if options.coordinate_order != "xyz":
        true_axis = dict(zip("xyz", options.coordinate_order, strict=True))
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

    if options.convert_to_metres and options.length_unit != "m":
        factor = LENGTH_UNITS[options.length_unit]
        collar_table[list(_LENGTH_COLUMNS["collar"])] *= factor
        survey_table[list(_LENGTH_COLUMNS["survey"])] *= factor
        for table in tables.values():
            table[list(_LENGTH_COLUMNS["interval"])] *= factor
        metadata.length_unit = "m"

    if options.normalise_dip and options.dip_convention != "negative_down":
        survey_table["dip"] = -survey_table["dip"]
        metadata.dip_convention = "negative_down"

    return DrillholeDatabase(
        collar=collar_table, survey=survey_table, intervals=tables, metadata=metadata
    )


# --------------------------------------------------------------------------------------
# column-map suggestion
# --------------------------------------------------------------------------------------
#: Common header spellings, normalised (lower case, letters and digits only, unit words
#: removed). Earlier entries win ties.
_SYNONYMS: dict[str, dict[str, tuple[str, ...]]] = {
    "collar": {
        "hole_id": (
            "holeid", "bhid", "hole", "dhid", "holeno", "holenumber", "holename", "drillhole",
            "drillholeid", "ddh", "borehole", "boreholeid", "wellid", "well", "siteid", "id",
            "name",
        ),
        "x": (
            "x", "east", "easting", "eastings", "xcollar", "collarx", "xcoord", "xcoordinate",
            "collareast", "collareasting", "utme", "utmeast", "utmeasting", "e",
        ),
        "y": (
            "y", "north", "northing", "northings", "ycollar", "collary", "ycoord",
            "ycoordinate", "collarnorth", "collarnorthing", "utmn", "utmnorth", "utmnorthing",
            "n",
        ),
        "z": (
            "z", "rl", "elevation", "elev", "zcollar", "collarz", "collarrl", "zcoord",
            "zcoordinate", "collarelevation", "reducedlevel", "altitude", "level", "height",
        ),
        "max_depth": (
            "maxdepth", "depth", "totaldepth", "eoh", "enddepth", "finaldepth", "holedepth",
            "endofhole", "td", "length", "holelength", "totallength",
        ),
    },
    "survey": {
        "hole_id": (
            "holeid", "bhid", "hole", "dhid", "holeno", "holenumber", "holename", "drillhole",
            "drillholeid", "ddh", "borehole", "boreholeid", "wellid", "well", "siteid", "id",
            "name",
        ),
        "depth": (
            "depth", "at", "surveydepth", "sdepth", "measureddepth", "md", "downholedepth",
            "distance", "station", "stationdepth",
        ),
        "dip": ("dip", "inclination", "incl", "inc", "plunge"),
        "azimuth": (
            "azimuth", "azi", "az", "azim", "bearing", "brg", "azimuthgrid", "azimuthtrue",
            "gridazimuth", "trueazimuth", "direction",
        ),
    },
    "interval": {
        "hole_id": (
            "holeid", "bhid", "hole", "dhid", "holeno", "holenumber", "holename", "drillhole",
            "drillholeid", "ddh", "borehole", "boreholeid", "wellid", "well", "siteid", "id",
            "name",
        ),
        "depth_from": (
            "depthfrom", "from", "fromdepth", "sampfrom", "samplefrom", "intervalfrom",
            "startdepth", "start", "top", "topdepth",
        ),
        "depth_to": (
            "depthto", "to", "todepth", "sampto", "sampleto", "intervalto", "enddepth", "end",
            "bottom", "bottomdepth", "base",
        ),
    },
}
_UNIT_WORDS = frozenset({"m", "mtr", "metre", "metres", "meter", "meters", "ft", "feet", "deg"})
_MIN_FUZZY_LENGTH = 4


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


def _normalise_header(header: object) -> str:
    words = [w for w in re.split(r"[^a-z0-9]+", str(header).casefold()) if w]
    kept = [w for w in words if w not in _UNIT_WORDS] or words
    return "".join(kept)


def _attribute_name(header: object) -> str:
    name = re.sub(r"[^a-z0-9]+", "_", str(header).strip().casefold()).strip("_")
    return name or "column"


def suggest_column_map(
    headers: Iterable[object],
    table: TableKind = "collar",
    *,
    cutoff: float = 0.8,
) -> ColumnMapSuggestion:
    """Propose *source column -> canonical name* from the headers of one table.

    Known spellings (``BHID``, ``HOLEID``, ``EAST``, ``NORTH``, ``RL``, ``FROM``, ``TO``,
    ``AZI``, ``DIP`` ...) match exactly after normalisation; other headers of 4 or more
    characters match fuzzily when their similarity to a known spelling reaches ``cutoff``.
    Each canonical name gets at most one source column. For an interval table the
    remaining columns are proposed as attributes under a lower-case name. The result is a
    proposal: check it, edit ``.mapping``, then pass it to :class:`TableSource`.
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
    mapping: dict[str, str] = {}
    unmatched: list[str] = []
    taken = set(_CANONICAL[table])
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
    missing = [c for c in _REQUIRED[table] if c not in chosen]
    return ColumnMapSuggestion(table, mapping, missing, unmatched, scores)
