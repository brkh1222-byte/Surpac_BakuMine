"""Unit dictionary: logged codes -> units of a kind (research.md 5.7.1, geology_contracts.md 2).

Raw logged codes are *folded* (outer whitespace removed, inner whitespace runs made one
space, upper case) and mapped through a per-deposit dictionary ``units.json`` to units. The
kind of a unit decides what the modeller does with its intervals. A code that is not in the
dictionary is an error unless the dictionary says ``"unmapped": "unknown"``; it never
becomes a unit silently, and a missing code stays missing (NaN).

The reports (:func:`code_report`, :func:`unit_metres`, :func:`density_samples_by_unit`)
count what the logs hold for each unit; :func:`units_provenance` turns them into the
measured / assumed / missing lines of the provenance record.
"""

from __future__ import annotations

import json
import math
import numbers
import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.geology.provenance import ProvenanceEntry

FORMAT = "bakumine.units"
VERSION = 1

UNIT_KINDS = ("lithology", "intrusion", "cover", "weathering", "fault-zone", "other", "missing")
MODELLED_KINDS = ("lithology", "intrusion")
RESERVED_CODES = ("AIR", "UNK", "ORE")
UNMAPPED_POLICIES = ("error", "unknown")
DENSITY_SOURCES = ("measured", "assumed")
DENSITY_KINDS = ("lithology", "intrusion", "cover")  # kinds that become rock in the blocks

# Colours for units without one, by position in the dictionary (units, then zones).
PALETTE = (
    "#4e79a7", "#f28e2b", "#59a14f", "#e15759", "#b07aa1", "#76b7b2", "#edc948", "#9c755f",
    "#ff9da7", "#8c8c8c", "#2f5d8a", "#c26a12", "#3d7a37", "#b03a3c", "#7d5a8c", "#4f8f8a",
    "#b89b1e", "#6e5140", "#d9707d", "#5c5c5c",
)  # fmt: skip

_CODE = re.compile(r"^[A-Z][A-Z0-9_]{0,31}$")
_COLOUR = re.compile(r"^#[0-9a-fA-F]{6}$")
_INTERVAL_COLUMNS = ("hole_id", "depth_from", "depth_to")

_USED_FOR = {
    "lithology": "rock-unit fields",
    "intrusion": "intrusion fields (overprint the host units)",
    "cover": "base surface of the cover",
    "weathering": "base surface of the weathering zone",
    "fault-zone": "fault positions",
    "missing": "marks intervals without information (a bound, never a contact)",
}


class UnitsError(ValueError):
    """A unit dictionary, or a table used with one, that breaks the rules."""


class UnmappedCodeError(UnitsError):
    """Logged codes that the dictionary does not know (policy ``unmapped = "error"``)."""

    codes: tuple[str, ...]

    def __init__(self, codes: Iterable[str]) -> None:
        self.codes = tuple(sorted(set(codes)))
        listed = ", ".join(repr(code) for code in self.codes)
        super().__init__(
            f"{len(self.codes)} logged code(s) are not in the unit dictionary: {listed}. "
            'Add them as a unit or an alias, or set "unmapped": "unknown".'
        )


def fold_code(raw: object, *, case: bool = True, strip: bool = True) -> str | None:
    """A logged code as it is compared: ``None`` for a missing value or an empty text.

    ``strip`` removes leading and trailing whitespace and makes inner whitespace runs one
    space; ``case`` upper-cases. A whole number read as float (``5.0``) folds to ``"5"``.
    """
    if raw is None or raw is pd.NA or raw is pd.NaT:
        return None
    if isinstance(raw, numbers.Real) and not isinstance(raw, numbers.Integral):
        value = float(raw)
        if math.isnan(value):
            return None
        text = str(int(value)) if value.is_integer() else str(raw)
    else:
        text = raw if isinstance(raw, str) else str(raw)
    if strip:
        text = " ".join(text.split())
    if case:
        text = text.upper()
    return text or None


@dataclass(frozen=True)
class Unit:
    code: str
    kind: str
    name: str | None = None
    aliases: tuple[str, ...] = ()
    group: str | None = None
    age: int | None = None
    colour: str | None = None
    density: float | None = None
    density_source: str | None = None


@dataclass(frozen=True)
class WeatheringZone:
    code: str
    name: str | None = None
    colour: str | None = None
    density: float | None = None
    density_source: str | None = None


_UNIT_FIELDS = ("code", "name", "kind", "aliases", "group", "age", "colour", "density",
                "density_source")  # fmt: skip
_ZONE_FIELDS = ("code", "name", "colour", "density", "density_source")
_TOP_FIELDS = (
    "format",
    "version",
    "name",
    "source",
    "fold",
    "unmapped",
    "default_density",
    "default_density_source",
    "display_colours",
    "units",
    "weathering_zones",
)


def _text(value: object, what: str, *, optional: bool = True) -> str | None:
    if value is None and optional:
        return None
    if not isinstance(value, str):
        raise UnitsError(f"{what} must be text, not {value!r}")
    return value


def _colour(value: object, what: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not _COLOUR.match(value):
        raise UnitsError(f"{what}: colour {value!r} is not of the form #rrggbb")
    return value.lower()


def _density(value: object, source: object, what: str) -> tuple[float | None, str | None]:
    if value is None:
        if source is not None:
            raise UnitsError(f"{what}: density_source {source!r} is given without a density")
        return None, None
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise UnitsError(f"{what}: density {value!r} is not a number")
    number = float(value)
    if not math.isfinite(number) or number <= 0.0:
        raise UnitsError(f"{what}: density {value!r} must be greater than 0 (t/m3)")
    source = "assumed" if source is None else source
    if source not in DENSITY_SOURCES:
        raise UnitsError(
            f"{what}: density_source {source!r} is not one of {', '.join(DENSITY_SOURCES)}"
        )
    return number, source


def _check_code(code: object, what: str) -> str:
    if not isinstance(code, str) or not _CODE.match(code):
        raise UnitsError(
            f"{what}: code {code!r} must match ^[A-Z][A-Z0-9_]{{0,31}}$ "
            "(upper-case letter first, then letters, digits or _, at most 32 characters)"
        )
    if code in RESERVED_CODES:
        raise UnitsError(f"{what}: code {code!r} is reserved ({', '.join(RESERVED_CODES)})")
    return code


def _only_known(data: Mapping, known: Iterable[str], what: str) -> None:
    unknown = sorted(str(key) for key in set(data) - set(known))
    if unknown:
        raise UnitsError(f"{what}: unknown field(s) {', '.join(unknown)}")


@dataclass(frozen=True)
class UnitDictionary:
    """The units of one deposit and how logged codes map to them.

    Every instance is checked and normalised when it is made (also when built directly, not
    only through :meth:`from_dict`): aliases are stored folded, without the unit's own code
    and without repeats; colours lower case; a density without a source is ``assumed``.
    """

    units: tuple[Unit, ...]
    weathering_zones: tuple[WeatheringZone, ...] = ()
    name: str = ""
    source_table: str | None = None
    source_column: str | None = None
    fold_case: bool = True
    strip: bool = True
    unmapped: str = "error"
    default_density: float | None = None
    default_density_source: str = "assumed"
    display_colours: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("fold_case", "strip"):
            if not isinstance(getattr(self, name), bool):
                raise UnitsError(f"fold: '{name}' must be true or false")
        if self.unmapped not in UNMAPPED_POLICIES:
            raise UnitsError(
                f"unmapped: {self.unmapped!r} is not one of {', '.join(UNMAPPED_POLICIES)}"
            )
        if not isinstance(self.name, str):
            raise UnitsError(f"name must be text, not {self.name!r}")
        _text(self.source_table, "source table")
        _text(self.source_column, "source column")

        if self.default_density is None:
            default, default_source = None, "assumed"
        else:
            default, default_source = _density(
                self.default_density, self.default_density_source, "default_density"
            )
        object.__setattr__(self, "default_density", default)
        object.__setattr__(self, "default_density_source", default_source)

        units: list[Unit] = []
        owner: dict[str, str] = {}  # folded alias -> "units[i] (code)"
        seen: dict[str, str] = {}
        for i, unit in enumerate(self.units):
            if not isinstance(unit, Unit):
                raise UnitsError(f"units[{i}] is not a Unit: {unit!r}")
            what = f"units[{i}] (code {unit.code!r})"
            code = _check_code(unit.code, what)
            if code in seen:
                raise UnitsError(f"{what}: the code is already used by {seen[code]}")
            seen[code] = what
            if unit.kind not in UNIT_KINDS:
                raise UnitsError(
                    f"{what}: kind {unit.kind!r} is not one of {', '.join(UNIT_KINDS)}"
                )
            if unit.age is not None and (
                isinstance(unit.age, bool) or not isinstance(unit.age, numbers.Integral)
            ):
                raise UnitsError(f"{what}: age {unit.age!r} must be a whole number")
            density, source = _density(unit.density, unit.density_source, what)
            if isinstance(unit.aliases, str) or not isinstance(unit.aliases, Iterable):
                raise UnitsError(f"{what}: aliases must be a list")
            aliases: list[str] = []
            for raw in (code, *unit.aliases):
                if isinstance(raw, bool) or not isinstance(raw, str | numbers.Integral):
                    raise UnitsError(f"{what}: alias {raw!r} must be text")
                folded = fold_code(raw, case=self.fold_case, strip=self.strip)
                if folded is None:
                    raise UnitsError(f"{what}: an alias is empty")
                if folded in owner and owner[folded] != what:
                    raise UnitsError(
                        f"{what}: raw code {folded!r} already belongs to {owner[folded]}; "
                        "an alias may belong to one unit only"
                    )
                owner[folded] = what
                if folded != code and folded not in aliases:
                    aliases.append(folded)
            units.append(
                replace(
                    unit,
                    name=_text(unit.name, f"{what}: name"),
                    group=_text(unit.group, f"{what}: group"),
                    aliases=tuple(aliases),
                    age=None if unit.age is None else int(unit.age),
                    colour=_colour(unit.colour, what),
                    density=density,
                    density_source=source,
                )
            )
        object.__setattr__(self, "units", tuple(units))

        zones: list[WeatheringZone] = []
        for i, zone in enumerate(self.weathering_zones):
            if not isinstance(zone, WeatheringZone):
                raise UnitsError(f"weathering_zones[{i}] is not a WeatheringZone: {zone!r}")
            what = f"weathering_zones[{i}] (code {zone.code!r})"
            code = _check_code(zone.code, what)
            if code in seen:
                raise UnitsError(f"{what}: the code is already used by {seen[code]}")
            seen[code] = what
            density, source = _density(zone.density, zone.density_source, what)
            zones.append(
                replace(
                    zone,
                    name=_text(zone.name, f"{what}: name"),
                    colour=_colour(zone.colour, what),
                    density=density,
                    density_source=source,
                )
            )
        object.__setattr__(self, "weathering_zones", tuple(zones))

        if not isinstance(self.display_colours, Mapping):
            raise UnitsError("display_colours must be an object of code: colour")
        display: dict[str, str] = {}
        for key, value in self.display_colours.items():
            if not isinstance(key, str) or not key:
                raise UnitsError(f"display_colours: key {key!r} must be text")
            colour = _colour(value, f"display_colours[{key!r}]")
            if colour is None:
                raise UnitsError(f"display_colours[{key!r}]: a colour is required")
            display[key] = colour
        object.__setattr__(self, "display_colours", display)

    # ------------------------------------------------------------------ file

    @classmethod
    def from_dict(cls, data: Mapping) -> UnitDictionary:
        if not isinstance(data, Mapping) or data.get("format") != FORMAT:
            raise UnitsError(f"not a unit dictionary (format is not {FORMAT!r})")
        if data.get("version") != VERSION:
            raise UnitsError(
                f"unit dictionary version {data.get('version')!r} is not supported "
                f"(this build reads version {VERSION})"
            )
        _only_known(data, _TOP_FIELDS, "unit dictionary")
        source = data.get("source") or {}
        if not isinstance(source, Mapping):
            raise UnitsError("source must be an object with 'table' and 'column'")
        _only_known(source, ("table", "column"), "source")
        fold = data.get("fold") or {}
        if not isinstance(fold, Mapping):
            raise UnitsError("fold must be an object with 'case' and 'strip'")
        _only_known(fold, ("case", "strip"), "fold")

        raw_units = data.get("units")
        if not isinstance(raw_units, list):
            raise UnitsError("units must be a list")
        units: list[Unit] = []
        for i, entry in enumerate(raw_units):
            if not isinstance(entry, Mapping):
                raise UnitsError(f"units[{i}] must be an object")
            what = f"units[{i}] (code {entry.get('code')!r})"
            _only_known(entry, _UNIT_FIELDS, what)
            for required in ("code", "kind"):
                if required not in entry:
                    raise UnitsError(f"{what}: '{required}' is required")
            aliases = entry.get("aliases", [])
            if not isinstance(aliases, list):
                raise UnitsError(f"{what}: aliases must be a list")
            units.append(
                Unit(
                    code=entry["code"],
                    kind=entry["kind"],
                    name=entry.get("name"),
                    aliases=tuple(aliases),
                    group=entry.get("group"),
                    age=entry.get("age"),
                    colour=entry.get("colour"),
                    density=entry.get("density"),
                    density_source=entry.get("density_source"),
                )
            )

        raw_zones = data.get("weathering_zones", [])
        if not isinstance(raw_zones, list):
            raise UnitsError("weathering_zones must be a list")
        zones: list[WeatheringZone] = []
        for i, entry in enumerate(raw_zones):
            if not isinstance(entry, Mapping):
                raise UnitsError(f"weathering_zones[{i}] must be an object")
            what = f"weathering_zones[{i}] (code {entry.get('code')!r})"
            _only_known(entry, _ZONE_FIELDS, what)
            if "code" not in entry:
                raise UnitsError(f"{what}: 'code' is required")
            zones.append(WeatheringZone(**{name: entry.get(name) for name in _ZONE_FIELDS}))

        default_source = data.get("default_density_source")
        if data.get("default_density") is None and default_source is not None:
            raise UnitsError("default_density_source is given without a default_density")
        return cls(
            units=tuple(units),
            weathering_zones=tuple(zones),
            name=data.get("name", ""),
            source_table=source.get("table"),
            source_column=source.get("column"),
            fold_case=fold.get("case", True),
            strip=fold.get("strip", True),
            unmapped=data.get("unmapped", "error"),
            default_density=data.get("default_density"),
            default_density_source="assumed" if default_source is None else default_source,
            display_colours=data.get("display_colours") or {},
        )

    @classmethod
    def load(cls, path: str | Path) -> UnitDictionary:
        file = Path(path)
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise UnitsError(f"{file} is not valid JSON: {error}") from None
        return cls.from_dict(data)

    def to_dict(self) -> dict:
        out: dict[str, Any] = {"format": FORMAT, "version": VERSION, "name": self.name}
        source = {"table": self.source_table, "column": self.source_column}
        if any(value is not None for value in source.values()):
            out["source"] = {key: value for key, value in source.items() if value is not None}
        out["fold"] = {"case": self.fold_case, "strip": self.strip}
        out["unmapped"] = self.unmapped
        if self.default_density is not None:
            out["default_density"] = self.default_density
            out["default_density_source"] = self.default_density_source
        if self.display_colours:
            out["display_colours"] = dict(self.display_colours)

        def entry(item: Unit | WeatheringZone, names: tuple[str, ...]) -> dict[str, Any]:
            row: dict[str, Any] = {}
            for name in names:
                value = getattr(item, name)
                if name == "aliases":
                    value = list(value) or None
                if value is not None:
                    row[name] = value
            return row

        out["units"] = [entry(unit, _UNIT_FIELDS) for unit in self.units]
        out["weathering_zones"] = [entry(zone, _ZONE_FIELDS) for zone in self.weathering_zones]
        return out

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        target.write_text(text + "\n", encoding="utf-8")

    # ------------------------------------------------------------------ lookups

    def unit(self, code: str) -> Unit:
        for unit in self.units:
            if unit.code == code:
                return unit
        raise KeyError(f"no unit {code!r} in the unit dictionary")

    def codes(self, *kinds: str) -> tuple[str, ...]:
        """Unit codes in dictionary order; of the given kinds only, or all."""
        for kind in kinds:
            if kind not in UNIT_KINDS:
                raise UnitsError(f"kind {kind!r} is not one of {', '.join(UNIT_KINDS)}")
        return tuple(unit.code for unit in self.units if not kinds or unit.kind in kinds)

    def lookup(self) -> dict[str, str]:
        """Folded alias -> unit code. A unit's own code is one of its aliases."""
        table: dict[str, str] = {}
        for unit in self.units:
            for alias in (unit.code, *unit.aliases):
                table[self.fold(alias) or alias] = unit.code
        return table

    def fold(self, raw: object) -> str | None:
        """:func:`fold_code` with this dictionary's settings."""
        return fold_code(raw, case=self.fold_case, strip=self.strip)

    def _map(self, values: pd.Series) -> tuple[pd.Series, pd.Series]:
        """(folded code, unit code) per value, object dtype, NaN where there is none."""
        table = self.lookup()
        folded = [self.fold(value) for value in values]
        units = [table.get(code, np.nan) if code is not None else np.nan for code in folded]
        return (
            pd.Series([np.nan if c is None else c for c in folded], values.index, dtype=object),
            pd.Series(units, index=values.index, dtype=object),
        )

    def map_codes(self, values: pd.Series) -> pd.Series:
        """Unit code per logged value (object dtype, same index); NaN for a missing value.

        A code that is not in the dictionary raises :class:`UnmappedCodeError` (all of them
        listed) when ``unmapped`` is ``"error"``, and becomes NaN when it is ``"unknown"``.
        """
        folded, units = self._map(values)
        if self.unmapped == "error":
            unmapped = folded[folded.notna() & units.isna()]
            if len(unmapped):
                raise UnmappedCodeError(unmapped.unique())
        return units

    def kinds_of(self, units: pd.Series) -> pd.Series:
        """Kind per unit code; NaN stays NaN. An unknown unit code is an error."""
        kinds = {unit.code: unit.kind for unit in self.units}
        present = units.dropna()
        unknown = sorted({str(code) for code in present[~present.isin(list(kinds))]})
        if unknown:
            raise UnitsError(f"not unit codes of this dictionary: {', '.join(unknown)}")
        return units.map(kinds).astype(object)

    def mask_kinds(self, units: pd.Series, keep: Iterable[str]) -> pd.Series:
        """The unit codes, NaN where the kind of the unit is not in ``keep``."""
        wanted = set(self.codes(*keep)) if (keep := tuple(keep)) else set()
        return units.where(units.isin(list(wanted)), np.nan).astype(object)

    def colours(self) -> dict[str, str]:
        """Colour per unit, weathering zone and display code.

        An entry without a colour takes :data:`PALETTE` by its position in the dictionary
        (units, then zones), so giving one unit a colour does not move the others.
        """
        entries = (*self.units, *self.weathering_zones)
        out = {e.code: e.colour or PALETTE[i % len(PALETTE)] for i, e in enumerate(entries)}
        out.update(self.display_colours)
        return out

    def density(self, code: str) -> tuple[float | None, str]:
        """(t/m3, ``"measured"`` | ``"assumed"``) of a unit or a weathering zone.

        A unit without a density takes ``default_density`` and its source; ``(None,
        "assumed")`` when there is none either. A weathering zone without a density gives
        ``(None, "assumed")``: it does not override the rock densities.
        """
        for zone in self.weathering_zones:
            if zone.code == code:
                return zone.density, zone.density_source or "assumed"
        unit = self.unit(code)
        if unit.density is not None:
            return unit.density, unit.density_source or "assumed"
        return self.default_density, self.default_density_source


# ---------------------------------------------------------------------- draft


def _draft_code(folded: str, position: int, taken: set[str]) -> str:
    code = re.sub(r"[^A-Z0-9_]", "_", folded.upper())
    if not re.search(r"[A-Z0-9]", code):
        code = f"U{position}"
    elif not re.match(r"[A-Z]", code):
        code = "U" + code
    code = code[:32]
    if code in RESERVED_CODES:
        code += "_LOG"
    base, n = code, 1
    while code in taken:
        n += 1
        suffix = f"_{n}"
        code = base[: 32 - len(suffix)] + suffix
    return code


def draft_dictionary(
    table: pd.DataFrame,
    column: str,
    *,
    table_name: str = "geology",
    fold_case: bool = True,
    strip: bool = True,
) -> UnitDictionary:
    """A first dictionary from the logged codes: every folded code its own ``lithology`` unit.

    Units are ordered by logged metres, largest first (ties by code). The user then edits
    kinds, names and aliases. A folded code that is not a valid unit code (a space, a digit
    first, a reserved word, other letters than A-Z) gets a valid code made from it and keeps
    the folded spelling as an alias.
    """
    frame = _intervals(table, column)
    folded = pd.Series(
        [fold_code(value, case=fold_case, strip=strip) for value in frame[column]],
        index=frame.index,
        dtype=object,
    )
    metres = frame["length"].groupby(folded).sum()
    order = sorted(metres.index, key=lambda code: (-float(metres[code]), code))
    units: list[Unit] = []
    # A spelling that is a valid code keeps it; a made code must not be any logged spelling.
    taken: set[str] = set(order)
    for position, spelling in enumerate(order, start=1):
        if _CODE.match(spelling) and spelling not in RESERVED_CODES:
            code = spelling
        else:
            code = _draft_code(spelling, position, taken)
        taken.add(code)
        aliases = () if spelling == code else (spelling,)
        units.append(Unit(code=code, kind="lithology", aliases=aliases))
    return UnitDictionary(
        units=tuple(units),
        source_table=table_name,
        source_column=column,
        fold_case=fold_case,
        strip=strip,
    )


# ---------------------------------------------------------------------- reports


def _intervals(table: pd.DataFrame, column: str) -> pd.DataFrame:
    """The interval table with ``length = depth_to - depth_from`` (NaN stays NaN)."""
    missing = [name for name in (*_INTERVAL_COLUMNS, column) if name not in table.columns]
    if missing:
        raise UnitsError(
            f"the table has no column {', '.join(repr(m) for m in missing)} "
            f"(columns: {', '.join(str(c) for c in table.columns)})"
        )
    frame = table[[*_INTERVAL_COLUMNS, column]].reset_index(drop=True)
    start = pd.to_numeric(frame["depth_from"], errors="coerce").astype("float64")
    end = pd.to_numeric(frame["depth_to"], errors="coerce").astype("float64")
    return frame.assign(depth_from=start, depth_to=end, length=end - start)


def _logged(table: pd.DataFrame, column: str, units: UnitDictionary) -> pd.DataFrame:
    """Intervals with a code: columns hole_id, depth_from, depth_to, length, raw, folded, unit."""
    frame = _intervals(table, column)
    folded, mapped = units._map(frame[column])
    frame = frame.assign(folded=folded, unit=mapped)
    frame = frame[frame["folded"].notna()]
    raw = [value if isinstance(value, str) else str(value) for value in frame[column]]
    return frame.assign(raw=pd.Series(raw, index=frame.index, dtype=object))


def _counts(frame: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = frame.groupby(key, sort=False)
    return pd.DataFrame(
        {
            "intervals": grouped.size().astype("int64"),
            "metres": grouped["length"].sum().astype("float64"),
            "holes": grouped["hole_id"].nunique().astype("int64"),
        }
    )


def code_report(table: pd.DataFrame, column: str, units: UnitDictionary) -> pd.DataFrame:
    """What every raw spelling in the log maps to: one row per raw spelling.

    Columns ``raw, folded, unit, kind, intervals, metres, holes, mapped``; sorted by folded
    code, then raw spelling. ``unit`` and ``kind`` are NaN for a code that is not in the
    dictionary (``mapped`` False); this function never raises for those. Intervals without
    a code (NaN, empty) are not rows. ``metres`` is the sum of ``depth_to - depth_from``.
    """
    frame = _logged(table, column, units)
    out = _counts(frame, "raw")
    first = frame.drop_duplicates("raw").set_index("raw")
    kinds = {unit.code: unit.kind for unit in units.units}
    out.insert(0, "folded", first["folded"].reindex(out.index).astype(object))
    out.insert(1, "unit", first["unit"].reindex(out.index).astype(object))
    out.insert(2, "kind", out["unit"].map(kinds).astype(object))
    out["mapped"] = out["unit"].notna().astype(bool)
    out = out.rename_axis("raw").reset_index()
    out["raw"] = out["raw"].astype(object)
    return out.sort_values(["folded", "raw"], kind="stable").reset_index(drop=True)


def unit_metres(
    table: pd.DataFrame, column: str, units: UnitDictionary, *, by: str = "unit"
) -> pd.DataFrame:
    """Logged intervals, metres and holes per unit, or per reporting group.

    Columns ``<by>, kind, intervals, metres, holes, fraction``; dictionary order; only
    units that are logged. ``fraction`` is the share of the logged metres, which are the
    metres of every interval that has a code, mapped or not, so the fractions sum to less
    than 1 when codes are unmapped. With ``by="group"`` a unit without a group counts under
    its own code, and ``kind`` is ``"mixed"`` when the units of a group differ in kind.
    """
    if by not in ("unit", "group"):
        raise UnitsError(f"by must be 'unit' or 'group', not {by!r}")
    frame = _logged(table, column, units)
    total = float(frame["length"].sum())
    frame = frame[frame["unit"].notna()]
    key_of = {u.code: u.code if by == "unit" else (u.group or u.code) for u in units.units}
    kinds: dict[str, set[str]] = {}
    for unit in units.units:
        kinds.setdefault(key_of[unit.code], set()).add(unit.kind)
    out = _counts(frame.assign(key=frame["unit"].map(key_of)), "key")
    order = [key for key in dict.fromkeys(key_of.values()) if key in out.index]
    out = out.reindex(order)
    out.insert(0, "kind", [next(iter(kinds[k])) if len(kinds[k]) == 1 else "mixed" for k in order])
    out["fraction"] = out["metres"] / total if total > 0.0 else np.nan
    out = out.rename_axis(by).reset_index()
    out[by] = out[by].astype(object)
    out["kind"] = out["kind"].astype(object)
    return out


def density_samples_by_unit(
    lith_table: pd.DataFrame,
    lith_column: str,
    density_table: pd.DataFrame,
    density_column: str,
    units: UnitDictionary,
) -> pd.DataFrame:
    """Density samples per unit: the unit logged at the midpoint of each sample, same hole.

    Columns ``unit, n, mean, min, max``; dictionary order; units with a sample only. The
    density table holds intervals (``depth_from``, ``depth_to``) or points (``depth``). A
    sample counts for the log interval with ``depth_from <= midpoint < depth_to``; samples
    with no density, outside every coded interval, or in an unmapped code are left out.
    Overlapping log intervals are not resolved: the one that starts last is taken.
    """
    lith = _logged(lith_table, lith_column, units)
    lith = lith[lith["unit"].notna() & lith["depth_from"].notna() & lith["depth_to"].notna()]
    need = ["hole_id", density_column]
    absent = [name for name in need if name not in density_table.columns]
    if absent:
        raise UnitsError(f"the density table has no column {', '.join(map(repr, absent))}")
    if {"depth_from", "depth_to"} <= set(density_table.columns):
        start = pd.to_numeric(density_table["depth_from"], errors="coerce")
        end = pd.to_numeric(density_table["depth_to"], errors="coerce")
        mid = (start + end) / 2.0
    elif "depth" in density_table.columns:
        mid = pd.to_numeric(density_table["depth"], errors="coerce")
    else:
        raise UnitsError("the density table needs depth_from and depth_to, or depth")
    samples = pd.DataFrame(
        {
            "hole_id": density_table["hole_id"].to_numpy(),
            "mid": np.asarray(mid, dtype="float64"),
            "value": pd.to_numeric(density_table[density_column], errors="coerce").to_numpy(
                dtype="float64"
            ),
        }
    ).dropna(subset=["mid", "value"])

    logs = {hole: part.sort_values("depth_from") for hole, part in lith.groupby("hole_id")}
    found_units: list[str] = []
    found_values: list[float] = []
    for hole, part in samples.groupby("hole_id"):
        log = logs.get(hole)
        if log is None:
            continue
        starts = log["depth_from"].to_numpy()
        ends = log["depth_to"].to_numpy()
        codes = log["unit"].to_numpy(dtype=object)
        mids = part["mid"].to_numpy()
        index = np.searchsorted(starts, mids, side="right") - 1
        inside = index >= 0
        inside[inside] = mids[inside] < ends[index[inside]]
        found_units.extend(codes[index[inside]])
        found_values.extend(part["value"].to_numpy()[inside])

    found = pd.DataFrame({"unit": pd.Series(found_units, dtype=object), "value": found_values})
    grouped = found.groupby("unit", sort=False)["value"]
    out = pd.DataFrame(
        {
            "n": grouped.size().astype("int64"),
            "mean": grouped.mean().astype("float64"),
            "min": grouped.min().astype("float64"),
            "max": grouped.max().astype("float64"),
        }
    )
    out = out.reindex([code for code in units.codes() if code in out.index])
    out = out.rename_axis("unit").reset_index()
    out["unit"] = out["unit"].astype(object)
    return out


# ---------------------------------------------------------------------- provenance


def units_provenance(
    table: pd.DataFrame,
    column: str,
    units: UnitDictionary,
    *,
    source: str | None = None,
    density_samples: pd.DataFrame | None = None,
    density_source: str = "density table",
) -> list[ProvenanceEntry]:
    """The dictionary's lines of the provenance record (geology_contracts.md 3, row T3).

    - measured: the log of every unit that is used (intervals, metres, holes); density
      samples per unit when ``density_samples`` (from :func:`density_samples_by_unit`) is
      given; a ``default_density`` the dictionary states as measured;
    - assumed: units of kind ``other`` (logged but not modelled); an assumed
      ``default_density``;
    - missing: logged codes that are not in the dictionary; rock units (lithology,
      intrusion, cover) with no density of their own.
    """
    table_name = units.source_table or "geology"
    source = source or f"table {table_name}, column {column}"
    tag = {"stage": "units", "task": "P9.T3"}
    metres = unit_metres(table, column, units)
    report = code_report(table, column, units)
    entries: list[ProvenanceEntry] = []

    def counted(row: Any) -> str:
        return f"{int(row.intervals)} intervals, {row.metres:.2f} m, {int(row.holes)} holes"

    for row in metres.itertuples(index=False):
        if row.kind == "other":
            continue
        entries.append(
            ProvenanceEntry(
                f"Log of unit {row.unit}",
                "measured",
                source,
                _USED_FOR[row.kind],
                detail=counted(row),
                count=int(row.intervals),
                **tag,
            )  # fmt: skip
        )
    if density_samples is not None:
        for row in density_samples.itertuples(index=False):
            entries.append(
                ProvenanceEntry(
                    f"Density samples in unit {row.unit}",
                    "measured",
                    density_source,
                    "density of the unit",
                    detail=f"{int(row.n)} samples, mean {row.mean:.3f}, "
                    f"{row.min:.3f} to {row.max:.3f} t/m3",
                    count=int(row.n),
                    value=float(row.mean),
                    **tag,
                )  # fmt: skip
            )
    if units.default_density is not None:
        measured = units.default_density_source == "measured"
        entries.append(
            ProvenanceEntry(
                "Default density",
                "measured" if measured else "assumed",
                f"unit dictionary, default_density ({units.default_density_source})",
                "units without a density of their own",
                detail=f"{units.default_density:g} t/m3",
                value=units.default_density,
                **tag,
            )  # fmt: skip
        )
    for row in metres.itertuples(index=False):
        if row.kind != "other":
            continue
        entries.append(
            ProvenanceEntry(
                f"Unit {row.unit} is not modelled",
                "assumed",
                "unit dictionary (kind other)",
                "its intervals are unknown for the rock-unit fields",
                detail=counted(row),
                count=int(row.intervals),
                **tag,
            )  # fmt: skip
        )

    unmapped = report[~report["mapped"]]
    if len(unmapped):
        by_code = unmapped.groupby("folded")[["intervals", "metres"]].sum()
        listed = ", ".join(repr(code) for code in sorted(by_code.index))
        policy = (
            "read as unknown" if units.unmapped == "unknown" else "an error until they are mapped"
        )
        entries.append(
            ProvenanceEntry(
                f"Logged codes not in the unit dictionary: {listed}",
                "missing",
                "the geologist's legend of the log codes (a unit or an alias in units.json)",
                f"their intervals ({policy})",
                detail=f"{len(by_code)} codes, {int(by_code['intervals'].sum())} intervals, "
                f"{float(by_code['metres'].sum()):.2f} m",
                count=int(by_code["intervals"].sum()),
                **tag,
            )  # fmt: skip
        )
    logged = set(metres["unit"])
    sampled: dict[str, tuple[int, float]] = {}
    if density_samples is not None:
        sampled = {r.unit: (int(r.n), float(r.mean)) for r in density_samples.itertuples()}
    for unit in units.units:
        if unit.kind not in DENSITY_KINDS or unit.density is not None:
            continue
        if units.default_density is not None:
            detail = f"uses the default density {units.default_density:g} t/m3"
        else:
            detail = "no default density either: its blocks get no density"
        if unit.code in sampled:
            detail += (
                f"; {sampled[unit.code][0]} samples in the density table (mean "
                f"{sampled[unit.code][1]:.3f} t/m3) are not yet entered in the dictionary"
            )
        if unit.code not in logged:
            detail += "; the unit is not logged in this table"
        entries.append(
            ProvenanceEntry(
                f"Density of unit {unit.code}",
                "missing",
                "density measurements in this unit (a density table or laboratory results)",
                "tonnes of the unit",
                detail=detail,
                **tag,
            )  # fmt: skip
        )
    return entries
