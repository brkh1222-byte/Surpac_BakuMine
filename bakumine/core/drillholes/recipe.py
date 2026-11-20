"""Import recipes: how a set of source files becomes a drillhole database, as JSON.

A recipe is saved with the database (``import_recipe.json`` and ``database.json``) and in
the project's job log, so the same import can be run again on updated files with
``bakumine import --recipe FILE``. It is a plain dict::

    {
      "format": "bakumine-import-recipe", "version": 1,
      "options": {...},                      # ImportOptions fields (units, dip, CRS ...)
      "hole_id": {"pattern": "DH[_-]?0*(\\\\d+)", "template": "DH-{1:02d}"},
      "holes": {"include": ["DH-*"], "exclude": ["DH-07", "*-T"]},   # optional
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
                        (``"DH*"``) or a list; default the first sheet. Ignored for a
                        CSV file, so that a table can mix a CSV and a workbook.
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

``holes`` keeps only some holes, in every table (QC rows too): ``include`` and ``exclude``
are lists of hole names or wildcard patterns (``*``, ``?``, ``[abc]``), compared without
regard to case with the hole ids *after* the ``hole_id`` rewriting. A hole is kept when it
matches ``include`` (or there is none) and does not match ``exclude``. The rows dropped
per table and the patterns that matched nothing are in ``reports["hole_filter"]``.

Rows that are not data are dropped and counted (``reports["dropped_rows"]`` and the
notes): total / summary rows (``Total``, ``Нийт``, ``Итого`` ... in the hole id cell or the
first filled cell), rows without a hole id, and, once the sources are merged, collars that
have neither projected coordinates nor longitude / latitude.
"""

from __future__ import annotations

import copy
import fnmatch
import json
from collections.abc import Mapping, Sequence
from dataclasses import asdict, fields
from pathlib import Path
from typing import Any

import numpy as np
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
    "HoleFilter",
    "MissingColumns",
    "load_recipe",
    "make_recipe",
    "merge_sources",
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


class HoleFilter:
    """Which holes an import keeps: names or wildcard patterns to include and to exclude
    (see the module documentation). Counts what it drops, per table."""

    def __init__(self, include: Sequence[str] = (), exclude: Sequence[str] = ()) -> None:
        self.include = self._patterns(include, "include")
        self.exclude = self._patterns(exclude, "exclude")
        self.matched: set[str] = set()
        self.tables: dict[str, dict[str, int]] = {}

    @staticmethod
    def _patterns(values: Any, what: str) -> list[str]:
        if values is None:
            return []
        if isinstance(values, str):
            values = [values]
        if not isinstance(values, Sequence):
            raise ValueError(f"holes.{what} must be a list of hole names or patterns")
        cleaned = [str(v).strip() for v in values]
        return list(dict.fromkeys(v for v in cleaned if v))

    @classmethod
    def from_dict(cls, values: Mapping[str, Any] | None) -> HoleFilter:
        if values is None:
            return cls()
        if not isinstance(values, Mapping):
            raise ValueError("holes must be {'include': [...], 'exclude': [...]}")
        unknown = sorted(set(values) - {"include", "exclude"})
        if unknown:
            raise ValueError(f"unknown holes key(s) {unknown}; known: ['exclude', 'include']")
        return cls(values.get("include") or (), values.get("exclude") or ())

    def to_dict(self) -> dict[str, list[str]]:
        out = {"include": list(self.include), "exclude": list(self.exclude)}
        return {k: v for k, v in out.items() if v}

    def __bool__(self) -> bool:
        return bool(self.include or self.exclude)

    def _hit(self, hole: str, patterns: Sequence[str]) -> bool:
        found = False
        for pattern in patterns:
            if fnmatch.fnmatchcase(hole.casefold(), pattern.casefold()):
                self.matched.add(pattern)
                found = True  # no break: every pattern that matches is marked as used
        return found

    def keep(self, hole_ids: pd.Series) -> np.ndarray:
        """True for the rows whose hole is kept."""
        ids = hole_ids.astype(str).str.strip()
        verdict = {}
        for hole in ids.unique():
            included = self._hit(hole, self.include) if self.include else True
            excluded = self._hit(hole, self.exclude)
            verdict[hole] = included and not excluded
        return ids.map(verdict).to_numpy(dtype=bool)

    def apply(self, table: pd.DataFrame, name: str, *extra: np.ndarray) -> tuple[Any, ...]:
        """``table`` without the rows of the holes left out (and ``extra`` row arrays cut
        the same way); counted under ``name``."""
        if not self or "hole_id" not in table.columns:
            return (table, *extra)
        keep = self.keep(table["hole_id"]) if len(table) else np.zeros(0, dtype=bool)
        count = self.tables.setdefault(
            name, {"rows_kept": 0, "rows_dropped": 0, "holes_dropped": 0}
        )
        count["rows_kept"] += int(keep.sum())
        count["rows_dropped"] += int((~keep).sum())
        count["holes_dropped"] += int(table["hole_id"][~keep].nunique())
        if keep.all():
            return (table, *extra)
        return (table.loc[keep].reset_index(drop=True), *(e[keep] for e in extra))

    def report(self) -> dict[str, Any]:
        return {
            **{"include": list(self.include), "exclude": list(self.exclude)},
            "tables": self.tables,
            "unmatched": [p for p in (*self.include, *self.exclude) if p not in self.matched],
        }


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
    holes: Mapping[str, Any] | None = None,
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
    chosen = HoleFilter.from_dict(holes)
    if chosen:
        recipe["holes"] = chosen.to_dict()
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
        holes=document.get("holes"),
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
# merging sources
# --------------------------------------------------------------------------------------
def _differs(a: Any, b: Any, tolerance: float) -> bool:
    if isinstance(a, int | float | np.integer | np.floating) and isinstance(
        b, int | float | np.integer | np.floating
    ):
        return bool(abs(float(a) - float(b)) > tolerance)
    return str(a) != str(b)


def merge_sources(
    table: pd.DataFrame,
    source: np.ndarray,
    keys: Sequence[str],
    *,
    precedence: str = "first",
    tolerance: float = 0.01,
    labels: Sequence[str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Merge rows with the same key that come from different sources (see module doc).

    ``source`` is the 0-based source index of each row. Returns the merged table (row
    order kept; the winning row stays where it was) and a report: ``merged`` rows
    removed, ``filled`` cells taken from a losing source, ``conflicts`` (count) and
    ``rows`` (the first conflicts: key, column, kept / dropped value and source).
    """
    source = np.asarray(source, dtype=np.int64)
    labels = list(labels) if labels is not None else [str(i) for i in range(source.max() + 1)]
    report: dict[str, Any] = {
        "precedence": precedence, "tolerance": tolerance, "keys": list(keys),
        "merged": 0, "filled": 0, "conflicts": 0, "rows": [],
    }  # fmt: skip
    if len(table) == 0 or len(np.unique(source)) < 2:
        return table, report
    key_frame = table[list(keys)].copy()
    for column in keys:
        if column != "hole_id":
            key_frame[column] = key_frame[column].round(6)
    groups = key_frame.groupby(list(keys), sort=False, dropna=True).indices
    table = table.copy()
    values = [c for c in table.columns if c not in keys]
    drop: list[int] = []
    for positions in groups.values():
        if len(positions) < 2 or len(set(source[positions].tolist())) < 2:
            continue
        sign = 1 if precedence == "first" else -1
        ranked = sorted(positions, key=lambda i: sign * source[i])
        winner_source = source[ranked[0]]
        winner = ranked[0]
        for other in ranked[1:]:
            if source[other] == winner_source:
                continue  # a duplicate inside the winning source: the validator's business
            drop.append(int(other))
            for column in values:
                at = table.columns.get_loc(column)
                kept, lost = table.iat[winner, at], table.iat[other, at]
                if pd.isna(lost):
                    continue
                if pd.isna(kept):
                    table.iat[winner, at] = lost
                    report["filled"] += 1
                elif _differs(kept, lost, tolerance):
                    report["conflicts"] += 1
                    if len(report["rows"]) < _MAX_LISTED:
                        row = {k: _plain(table[k].iat[winner]) for k in keys}
                        row.update(
                            column=column,
                            kept=_plain(kept),
                            dropped=_plain(lost),
                            kept_source=labels[winner_source],
                            dropped_source=labels[source[other]],
                        )
                        report["rows"].append(row)
    report["merged"] = len(drop)
    if drop:
        table = table.drop(index=table.index[drop]).reset_index(drop=True)
    return table, report


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    return None if isinstance(value, float) and np.isnan(value) else value


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


def _collars_with_coordinates(collar: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """The collars that have a position (projected x and y, or a readable longitude and
    latitude), and the hole ids of the rows that have none (a planned hole, a remark)."""
    from .coordinates import _geographic

    known = np.ones(len(collar), dtype=bool)
    for axis in ("x", "y"):
        if axis in collar.columns:
            known &= collar[axis].notna().to_numpy(dtype=bool)
        else:
            known[:] = False
    geographic = _geographic(collar)
    if geographic is not None:
        known |= np.isfinite(geographic[0]) & np.isfinite(geographic[1])
    without = collar["hole_id"].astype(str).to_numpy()[~known].tolist()
    return (collar.loc[known].reset_index(drop=True) if without else collar), without


def _units(
    name: str, per_source: Sequence[Mapping[str, str]], labels: Sequence[str],
    metadata: DrillholeMetadata,
) -> dict[str, str]:  # fmt: skip
    """Column -> unit read from the headers of a table's sources. Sources that give one
    column in different units are reported (nothing is converted)."""
    units: dict[str, str] = {}
    seen: dict[str, dict[str, str]] = {}
    for label, found in zip(labels, per_source, strict=True):
        for column, unit in found.items():
            units.setdefault(column, unit)
            seen.setdefault(column, {})[label] = unit
    for column, by_file in seen.items():
        if len(set(by_file.values())) > 1:
            metadata.notes.append(
                {"table": name, "column": column, "kind": "units_differ_between_sources",
                 "count": len(by_file), "values": dict(by_file),
                 "hint": "nothing was converted; import the files separately or rename"}
            )  # fmt: skip
    return units


def run_recipe(recipe: Mapping[str, Any], base: str | Path | None = None) -> DrillholeDatabase:
    """Import the files a recipe names. ``base`` is the folder its relative paths start
    from. The database's metadata holds the recipe with every column map filled in
    (``metadata.recipe``) and the reports (``metadata.reports``): per table the sheets read,
    QC rows set aside, depths taken from sample numbers, merge conflicts, the units read
    from element headers (``units``); the coordinate check; the survey made from the
    collar."""
    recipe = check_recipe(recipe)
    base = Path(base) if base is not None else None
    options = options_from_dict(recipe["options"])
    metadata = DrillholeMetadata(
        length_unit=options.length_unit, source_length_unit=options.length_unit
    )
    resolved = copy.deepcopy(recipe)
    holes = HoleFilter.from_dict(recipe.get("holes"))
    grids: dict[Path, dict[str, pd.DataFrame]] = {}
    typed: dict[str, pd.DataFrame] = {}
    qc: dict[str, pd.DataFrame] = {}
    reports: dict[str, Any] = {
        "tables": {}, "qc": {}, "sample_depth": {}, "conflicts": {}, "units": {},
        "dropped_rows": {},
    }  # fmt: skip
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
        source = table.pop(_SOURCE).to_numpy(dtype=np.int64)
        table, source = holes.apply(table, name, source)
        if len(sources) > 1:
            table, merged = merge_sources(
                table,
                source,
                _KEYS[kind],
                precedence=recipe["merge"]["precedence"],
                tolerance=recipe["merge"]["tolerance"],
                labels=labels,
            )
            reports["conflicts"][name] = merged
            for what in ("merged", "filled", "conflicts"):
                if merged[what]:
                    metadata.notes.append(
                        {"table": name, "column": "*", "kind": f"sources_{what}",
                         "count": merged[what]}  # fmt: skip
                    )
        gone: dict[str, Any] = {}
        for part in prepared:
            for reason, count in part.info.get("dropped", {}).items():
                gone[reason] = gone.get(reason, 0) + count
        if kind == "collar":
            table, without = _collars_with_coordinates(table)
            if without:
                gone["collars_without_coordinates"] = len(without)
                gone["holes_without_coordinates"] = without[:_MAX_LISTED]
                metadata.notes.append(
                    {"table": name, "column": "x,y", "count": len(without),
                     "kind": "collars_without_coordinates_dropped", "examples": without[:5]}
                )  # fmt: skip
        if gone:
            reports["dropped_rows"][name] = gone
        units = _units(name, [p.info.get("units", {}) for p in prepared], labels, metadata)
        if units:
            reports["units"][name] = {c: u for c, u in units.items() if c in table.columns}
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
                (qc[name],) = holes.apply(qc[name], f"{name} (QC)")
            if name in qc and len(qc[name]):
                reports["qc"][name] = {
                    str(k): int(v) for k, v in qc[name]["qc_type"].value_counts().items()
                }
            elif name in qc:
                del qc[name]
            checks = [p.info["sample_depth"] for p in prepared if "sample_depth" in p.info]
            if checks:
                reports["sample_depth"][name] = checks
    if holes:
        if not len(typed["collar"]):
            raise ValueError(
                f"the hole filter leaves no hole (include {holes.include or 'all'}, "
                f"exclude {holes.exclude or 'none'})"
            )
        reports["hole_filter"] = holes.report()
        for pattern in reports["hole_filter"]["unmatched"]:
            metadata.notes.append(
                {"table": "*", "column": "hole_id", "kind": "hole_filter_matched_nothing",
                 "count": 1, "examples": [pattern]}
            )  # fmt: skip
    metadata.reports.update({k: v for k, v in reports.items() if v})
    collar = typed.pop("collar")
    survey = typed.pop("survey", None)
    db = finish_database(collar, survey, typed, options, metadata, qc)
    db.metadata.recipe = resolved
    return db
