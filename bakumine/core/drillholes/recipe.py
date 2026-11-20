"""Import recipes: how a set of source files becomes a drillhole database, as JSON.

A recipe is saved with the database (``import_recipe.json`` and ``database.json``) and in
the project's job log, so the same import can be run again on updated files with
``bakumine import --recipe FILE``. It is a plain dict::

    {
      "format": "bakumine-import-recipe", "version": 1,
      "options": {...},                      # ImportOptions fields (units, dip, CRS ...)
      "hole_id": {"pattern": "DH[_-]?0*(\\\\d+)", "template": "DH-{1:02d}"},
      "merge": {"precedence": "first", "tolerance": 0.01},
      "tables": {
        "collar": [SOURCE, ...],             # one table from one or more sources
        "survey": [SOURCE, ...],             # optional: else from the collar's azimuth, dip
        "assay":  [SOURCE, ...]              # any number of interval tables
      }
    }

``SOURCE`` describes one file:

``file``                path, relative to the recipe's base folder unless absolute.
``sheets``              workbook sheets to stack: a name, a 0-based index, a wildcard
                        (``"DH*"``) or a list; default the first sheet.
``exclude_sheets``      names / wildcards left out of the selection.
``header_row``          spreadsheet row of the header (1 = first row, 0 = no header);
                        default: found by the known column names.
``header_rows``         1 or 2 (group header over sub-headers); default: detected.
``map``                 source column -> canonical name; default: suggested from the
                        header. Besides the table's canonical columns: ``sample_id``,
                        ``sample_type``, ``sample_length`` (interval tables) and ``lon``,
                        ``lat`` or ``lon_deg`` ... ``lat_sec``, ``azimuth``, ``dip`` (collar).
``hole_id_from_sheet``  true: the sheet name is the hole id.
``hole_id``             ``{"pattern", "template"}`` for this source (else the recipe's).
``sample_depth``        interval tables: read depths from the sample number where the
                        ``from`` / ``to`` cells are empty. ``true`` or ``{"column":
                        "sample_id", "length": 1.0, "length_column": null, "patterns": [...],
                        "when": "missing" | "always"}`` (see
                        :func:`.tables.depths_from_sample_ids`).
``qc``                  interval tables: ``{"mode": "separate" | "off", "columns": [...],
                        "letter_suffix": true}``; default separate: QC rows leave the table
                        and are kept in the database's QC table.

Several sources of one table are stacked. Rows with the same key (``hole_id`` for the
collar; ``hole_id, depth`` for the survey; ``hole_id, depth_from, depth_to`` for an interval
table) coming from *different* sources are merged into one: the first source listed wins
(``"precedence": "last"`` reverses that), cells it lacks are filled from the others, and
cells that differ by more than ``tolerance`` (numbers) or at all (text) are reported as
conflicts. So a hole may take its coordinates from one file and azimuth, dip and depth
from another.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import pandas as pd

from .importer import (
    DrillholeDatabase,
    DrillholeMetadata,
    ImportOptions,
    _check_table_name,
    finish_database,
    import_table,
    missing_required,
)
from .tables import Prepared, _stack, prepare_source, read_grids

__all__ = [
    "RECIPE_FILE",
    "MissingColumns",
    "load_recipe",
    "make_recipe",
    "options_from_dict",
    "options_to_dict",
    "run_recipe",
    "save_recipe",
]

RECIPE_FORMAT = "bakumine-import-recipe"
RECIPE_VERSION = 1
RECIPE_FILE = "import_recipe.json"
_SOURCE = "__source__"
_SOURCE_KEYS = frozenset(
    {
        "file", "sheets", "exclude_sheets", "header_row", "header_rows", "map",
        "hole_id_from_sheet", "hole_id", "sample_depth", "qc",
    }
)  # fmt: skip
_KEYS: dict[str, tuple[str, ...]] = {
    "collar": ("hole_id",),
    "survey": ("hole_id", "depth"),
    "interval": ("hole_id", "depth_from", "depth_to"),
}
_MAX_LISTED = 200
_TUPLE_OPTIONS = ("missing_tokens", "text_columns", "numeric_columns")


class MissingColumns(ValueError):
    """No column was found for a required canonical column of a table whose column map
    was left to the suggester."""

    def __init__(self, table: str, missing: list[str], headers: list[str]) -> None:
        super().__init__(
            f"no column found for {missing} in the {table} table (headers: {headers}); "
            "give a column map"
        )
        self.table, self.missing, self.headers = table, missing, headers


# --------------------------------------------------------------------------------------
# recipe documents
# --------------------------------------------------------------------------------------
def options_to_dict(options: ImportOptions) -> dict[str, Any]:
    out = asdict(options)
    for key in _TUPLE_OPTIONS:
        out[key] = list(out[key])
    return out


def options_from_dict(values: Mapping[str, Any] | None) -> ImportOptions:
    """``ImportOptions`` from a recipe / request dict; None and "" mean the default."""
    known = {f.name for f in fields(ImportOptions)}
    cleaned: dict[str, Any] = {}
    for key, value in (values or {}).items():
        if key not in known:
            raise ValueError(f"unknown import option {key!r}; known: {sorted(known)}")
        if value is None or value == "":
            if key == "delimiter" and value is None:
                cleaned[key] = None  # None is meaningful: detect the delimiter
            continue
        if key in _TUPLE_OPTIONS:
            value = (value,) if isinstance(value, str | int | float) else tuple(value)
        cleaned[key] = value
    return ImportOptions(**cleaned)


def _kind(table: str) -> str:
    return table if table in ("collar", "survey") else "interval"


def make_recipe(
    tables: Mapping[str, Sequence[Mapping[str, Any]] | Mapping[str, Any]],
    options: ImportOptions | Mapping[str, Any] | None = None,
    *,
    hole_id: Mapping[str, Any] | None = None,
    merge: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """A checked recipe dict. ``tables`` maps a table name to one source or a list."""
    if not isinstance(options, ImportOptions):
        options = options_from_dict(options)
    if "collar" not in tables:
        raise ValueError("a recipe needs a collar table")
    checked: dict[str, list[dict[str, Any]]] = {}
    for name, sources in tables.items():
        if name not in ("collar", "survey"):
            _check_table_name(name)
        if isinstance(sources, Mapping):
            sources = [sources]
        if not sources:
            raise ValueError(f"table {name!r} has no source")
        checked[name] = []
        for source in sources:
            if not isinstance(source, Mapping) or not source.get("file"):
                raise ValueError(f"table {name!r}: every source needs a 'file'")
            unknown = sorted(set(source) - _SOURCE_KEYS)
            if unknown:
                raise ValueError(
                    f"table {name!r}: unknown source key(s) {unknown}; "
                    f"known: {sorted(_SOURCE_KEYS)}"
                )
            checked[name].append(
                {k: copy.deepcopy(v) for k, v in source.items() if v is not None and v != ""}
            )
    merge = {"precedence": "first", "tolerance": 0.01, **(merge or {})}
    if merge["precedence"] not in ("first", "last"):
        raise ValueError(
            f"merge precedence must be 'first' or 'last', got {merge['precedence']!r}"
        )
    merge["tolerance"] = float(merge["tolerance"])
    recipe: dict[str, Any] = {
        "format": RECIPE_FORMAT,
        "version": RECIPE_VERSION,
        "options": options_to_dict(options),
        "merge": merge,
        "tables": checked,
    }
    if hole_id and hole_id.get("pattern"):
        recipe["hole_id"] = {
            "pattern": str(hole_id["pattern"]),
            "template": str(hole_id.get("template") or "{0}"),
        }
    return recipe


def check_recipe(document: Any) -> dict[str, Any]:
    """Validate a recipe read from JSON and return it in the checked form."""
    if not isinstance(document, Mapping) or document.get("format") != RECIPE_FORMAT:
        raise ValueError(f"not a {RECIPE_FORMAT} document")
    if document.get("version") != RECIPE_VERSION:
        raise ValueError(f"unsupported {RECIPE_FORMAT} version {document.get('version')!r}")
    tables = document.get("tables")
    if not isinstance(tables, Mapping):
        raise ValueError("the recipe has no 'tables'")
    return make_recipe(
        tables,
        document.get("options"),
        hole_id=document.get("hole_id"),
        merge=document.get("merge"),
    )


def load_recipe(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: not JSON ({exc})") from None
    try:
        return check_recipe(document)
    except ValueError as exc:
        raise ValueError(f"{path}: {exc}") from None


def save_recipe(recipe: Mapping[str, Any], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(recipe, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


# --------------------------------------------------------------------------------------
# running a recipe
# --------------------------------------------------------------------------------------
def _resolve(file: str, base: Path | None) -> Path:
    path = Path(file)
    if not path.is_absolute() and base is not None:
        path = base / path
    if not path.is_file():
        raise FileNotFoundError(f"source file not found: {path}")
    return path


def prepare_table(
    name: str,
    sources: Sequence[Mapping[str, Any]],
    recipe: Mapping[str, Any],
    options: ImportOptions,
    base: Path | None,
    grids: dict[Path, dict[str, pd.DataFrame]],
) -> list[Prepared]:
    """The sources of one table after the row rules (text). ``grids`` caches files read."""
    prepared = []
    for source in sources:
        path = _resolve(str(source["file"]), base)
        if path not in grids:
            grids[path] = read_grids(path, delimiter=options.delimiter, encoding=options.encoding)
        try:
            prepared.append(
                prepare_source(
                    source,
                    _kind(name),
                    name=name,
                    grids=grids[path],
                    hole_id=recipe.get("hole_id"),
                    keep_unmapped=options.keep_unmapped,
                )
            )
        except ValueError as exc:
            raise ValueError(f"{Path(str(source['file'])).name}: {exc}") from None
    if any(not source.get("map") for source in sources):
        present = {t for part in prepared for t in part.info["targets"]}
        missing = missing_required(_kind(name), present)
        if missing:
            headers = [h for part in prepared for h in part.info["columns"]]
            raise MissingColumns(name, missing, list(dict.fromkeys(headers)))
    return prepared


def run_recipe(recipe: Mapping[str, Any], base: str | Path | None = None) -> DrillholeDatabase:
    """Import the files a recipe names. ``base`` is the folder its relative paths start
    from. The database's metadata holds the recipe with every column map filled in
    (``metadata.recipe``) and the reports (``metadata.reports``): per table the sheets read,
    QC rows set aside, depths taken from sample numbers, merge conflicts; the coordinate
    check; the survey made from the collar."""
    recipe = check_recipe(recipe)
    base = Path(base) if base is not None else None
    options = options_from_dict(recipe["options"])
    metadata = DrillholeMetadata(
        length_unit=options.length_unit, source_length_unit=options.length_unit
    )
    resolved = copy.deepcopy(recipe)
    grids: dict[Path, dict[str, pd.DataFrame]] = {}
    typed: dict[str, pd.DataFrame] = {}
    qc: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {"tables": {}, "qc": {}, "sample_depth": {}, "conflicts": {}}
    for name, sources in recipe["tables"].items():
        kind = _kind(name)
        prepared = prepare_table(name, sources, recipe, options, base, grids)
        labels = [Path(str(s["file"])).name for s in sources]
        for i, part in enumerate(prepared):
            resolved["tables"][name][i]["map"] = part.mapping
            metadata.notes.extend(part.notes)
        stacked = _stack([p.table.assign(**{_SOURCE: str(i)}) for i, p in enumerate(prepared)])
        identity = {c: c for c in stacked.columns}
        table, notes = import_table(stacked, kind, identity, options, name=name)  # type: ignore[arg-type]
        metadata.notes.extend(notes)
        table.pop(_SOURCE)  # several sources are stacked as they come
        typed[name] = table
        metadata.column_maps[name] = dict(prepared[0].mapping)
        metadata.sources[name] = "; ".join(str(s["file"]) for s in sources)
        reports["tables"][name] = {
            "rows": int(len(table)),
            "sources": [
                {"file": label, "rows": p.info["rows"], "sheets": p.info["sheets"]}
                for label, p in zip(labels, prepared, strict=True)
            ],
        }
        if kind == "interval":
            parts = [p.qc.assign(source=label) for label, p in zip(labels, prepared, strict=True)
                     if len(p.qc)]  # fmt: skip
            if parts:
                qc[name] = pd.concat(parts, ignore_index=True).fillna("").astype(str)
                reports["qc"][name] = {
                    str(k): int(v) for k, v in qc[name]["qc_type"].value_counts().items()
                }
            checks = [p.info["sample_depth"] for p in prepared if "sample_depth" in p.info]
            if checks:
                reports["sample_depth"][name] = checks
    metadata.reports.update({k: v for k, v in reports.items() if v})
    collar = typed.pop("collar")
    survey = typed.pop("survey", None)
    db = finish_database(collar, survey, typed, options, metadata, qc)
    db.metadata.recipe = resolved
    return db
