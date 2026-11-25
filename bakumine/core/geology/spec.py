"""The build spec of a geology model and the build itself (P9.T14).

Contract: geology_contracts.md section 10 (spec), 9 (folder), 3 (provenance). Theory:
research.md 5.7.6 (order of the steps).

:func:`load_spec` validates a spec (a JSON file or a mapping) and resolves its paths;
every error names the field and says what is allowed. :func:`build_geology` runs the
steps in the order of research.md 5.7.6:

1. drillholes, unit dictionary, the logged units (cover, weathering, fault-zone, core-loss
   and ``other`` codes are unknown to the rock-unit fields);
2. the grid from the extent and the resolution (coarsened to ``max_nodes``);
3. cover base and weathering bases as depths below the topography, nested top to bottom;
4. the faults (imported or inferred from the fault-block codes) and their network;
5. the rock units: one stack per fault block (``fit_block_stacks``), or one stack ``ALL``;
   intrusions overprint in age order, the units younger than the faults last;
6. the model, sampled on the grid; throw of every fault; the logged units reproduced
   along the holes; volumes inside the drilled hull and outside it.

Nothing is defaulted silently. A value the spec leaves out is listed in
``GeologySpec.defaults`` and becomes an *assumed* line of the provenance record; an input
that is used is a *measured* line with its counts; an optional input that is absent (no
topography, no fault data, no density of a unit, cover or weathering not built) is a
*missing* line that says what to provide.

:func:`save_geology` writes the folder of section 9: ``GeologyModel.save`` plus
``spec.json`` (the spec as run, every default written out), the side files of the
surfaces (``surfaces/<name>.json``, ``<name>_picks.csv``) and ``diagnostics.json`` as
standard JSON (NaN is written as ``null``).

Spec keys beyond the example of section 10 (all optional):

``topography``            a path, or ``{"file": path, "source": "survey" | "dem" | ...}``
``lithology.events``      default: one ``argmax`` event of the logged lithology units and
                          one ``intrusion`` event per logged intrusion unit
``lithology.intrusion_order``  the intrusions from the oldest to the youngest; default:
                          the ``age`` of the dictionary when every intrusion has one,
                          else the dictionary order
``faults.post_fault``     units younger than every fault (default none)
``density_source``        ``{"table", "column"}``: density samples, counted per unit
``extra_intervals``       ``{"file" | "rows", "units_column", "surfaces"}``: more logged
                          intervals for the surfaces. An entry is used for a surface when
                          ``surfaces`` names it, or, without ``surfaces``, when it holds
                          an interval of the surface's own units.
``units``                 a path or the dictionary itself (an object)
``replace_with.points``   a file or a list of ``[x, y, z]``
``defaults``              written by the build into ``spec.json``: the fields the spec
                          left out, with the value used. Read back, they are still
                          reported as defaults.

Coordinates are x = east, y = north, z = up, metres.
"""

from __future__ import annotations

import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from bakumine.core.geology.lithology import EventSpec, FieldSettings
from bakumine.core.geology.provenance import Provenance, ProvenanceEntry
from bakumine.core.geology.units import MODELLED_KINDS, UnitDictionary, UnitsError

if TYPE_CHECKING:
    from bakumine.core.drillholes.importer import DrillholeDatabase
    from bakumine.core.geology.model import GeologyModel
    from bakumine.core.geometry.dtm import DTM

__all__ = [
    "FORMAT",
    "VERSION",
    "BuildRecord",
    "GeologySpec",
    "SpecError",
    "build_geology",
    "build_summary",
    "json_safe",
    "load_spec",
    "save_geology",
]

FORMAT = "bakumine.geology-spec"
VERSION = 1
DEFAULT_MAX_NODES = 20_000_000
DEFAULT_MIN_PICKS = 3
DEFAULT_MIN_HOLES = 5
#: step of the hole points the distance to data is measured from, metres
HOLE_POINT_STEP = 5.0
#: step of the points along the holes where the logged unit is compared with the model
RESIDUAL_STEP = 2.0
EVENT_MODES = ("argmax", "sequence", "intrusion")
KERNELS = ("linear", "cubic", "thin_plate")
_TASK = "P9.T14"
_NAME = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_\-]{0,63}\Z")


class SpecError(ValueError):
    """The build spec is not valid. The text names the field and what is allowed."""


# --------------------------------------------------------------------------
# reading values
# --------------------------------------------------------------------------


def _fail(where: str, text: str) -> SpecError:
    return SpecError(f"{where}: {text}")


def _mapping(value: Any, where: str, allowed: Sequence[str]) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise _fail(where, f"must be an object with the keys {', '.join(allowed)}")
    unknown = [str(k) for k in value if k not in allowed]
    if unknown:
        raise _fail(
            where, f"unknown key(s) {', '.join(sorted(unknown))}; allowed: {', '.join(allowed)}"
        )
    return dict(value)


def _number(
    value: Any, where: str, *, positive: bool = False, minimum: float | None = None
) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float) or not math.isfinite(value):
        raise _fail(where, f"must be a number, got {value!r}")
    if positive and not value > 0:
        raise _fail(where, f"must be > 0, got {value!r}")
    if minimum is not None and value < minimum:
        raise _fail(where, f"must be >= {minimum:g}, got {value!r}")
    return float(value)


def _integer(value: Any, where: str, *, minimum: int = 1) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise _fail(where, f"must be a whole number >= {minimum}, got {value!r}")
    return int(value)


def _text(value: Any, where: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise _fail(where, f"must be text, got {value!r}")
    return value.strip()


def _choice(value: Any, where: str, allowed: Sequence[str]) -> str:
    if value not in allowed:
        raise _fail(where, f"must be one of {', '.join(allowed)}; got {value!r}")
    return str(value)


def _codes(value: Any, where: str, known: Sequence[str], what: str) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence) or len(value) == 0:
        raise _fail(
            where,
            f"must be a list of {what}, for example [{known[0]!r}]"
            if known
            else f"must be a list of {what}",
        )
    out = []
    for item in value:
        if not isinstance(item, str) or item not in known:
            raise _fail(where, f"{item!r} is not one of the {what}: {', '.join(known) or 'none'}")
        out.append(item)
    if len(set(out)) != len(out):
        raise _fail(where, f"lists a code twice: {', '.join(out)}")
    return tuple(out)


def _path(value: Any, where: str, base: Path, *, must_exist: str = "file") -> Path:
    if isinstance(value, Path):
        value = str(value)
    text = _text(value, where)
    path = Path(text)
    path = path if path.is_absolute() else base / path
    exists = path.is_dir() if must_exist == "folder" else path.is_file()
    if not exists:
        raise _fail(where, f"no {must_exist} {path}")
    return path


def _table_column(value: Any, where: str, extra: Sequence[str] = ()) -> dict[str, Any]:
    out = _mapping(value, where, ("table", "column", *extra))
    for key in ("table", "column"):
        if key not in out:
            raise _fail(where, f"needs '{key}' (an interval table of the database and a column)")
        out[key] = _text(out[key], f"{where}.{key}")
    return out


# --------------------------------------------------------------------------
# the spec
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SurfaceSource:
    """One source of a base surface: logged units or a rule on a numeric column."""

    units: tuple[str, ...] = ()
    table: str = ""
    column: str = ""
    below: float | None = None
    above: float | None = None


@dataclass(frozen=True)
class SurfaceSpec:
    """A base surface: of the cover unit or of a weathering zone."""

    name: str  # "base:Q4", "base:OXIDE"
    code: str
    role: str  # "cover" | "weathering"
    sources: tuple[SurfaceSource, ...]
    mode: str
    absent: str


@dataclass(frozen=True)
class FaultSpec:
    name: str
    between: tuple[str, str] | None
    prior: tuple[float, float] | None
    #: ("file", Path) | ("points", Path | array) | ("plane", (point, dip, dip_azimuth)) | None
    replace_with: tuple[str, Any] | None
    terminates_on: tuple[str, int] | None


@dataclass(frozen=True)
class FaultsSpec:
    blocks: dict[str, str] | None  # table, column, pattern
    fault_zone_units: tuple[str, ...]
    min_picks: int
    min_holes_per_block: int
    post_fault: tuple[str, ...] | None
    surfaces: tuple[FaultSpec, ...]


@dataclass(frozen=True)
class ExtraIntervals:
    label: str
    file: Path | None
    rows: tuple[dict[str, Any], ...] | None
    units_column: str
    surfaces: tuple[str, ...] | None


@dataclass(frozen=True)
class GeologySpec:
    """A validated build spec with resolved paths. Make one with :func:`load_spec`."""

    name: str
    database: Path | None
    units: UnitDictionary
    units_text: str  # how the spec names the dictionary, for the provenance record
    table: str
    column: str
    topography: Path | None
    topography_source: str | None
    extent: dict[str, Any] | None  # {"rectangle", "z"} | {"model": Path, "padding"} | None
    resolution: float | None
    max_nodes: int
    faults: FaultsSpec | None
    events: tuple[EventSpec, ...] | None
    intrusion_order: tuple[str, ...] | None
    cap: float | None
    kernel: str
    max_extrapolation: float | None
    cover: SurfaceSpec | None
    weathering: tuple[SurfaceSpec, ...]
    interpolation: dict[str, Any]
    density_source: dict[str, str] | None
    extra_intervals: tuple[ExtraIntervals, ...]
    #: the spec with every default written out; paths as given
    document: dict[str, Any] = field(default_factory=dict)
    #: (field, value used, what it is used for) of every value the spec left out
    defaults: tuple[tuple[str, str, str], ...] = ()
    base_dir: Path | None = None

    def to_dict(self) -> dict[str, Any]:
        """The spec as run: JSON content of ``spec.json``."""
        return json.loads(json.dumps(self.document))


_TOP = (
    "format",
    "version",
    "name",
    "database",
    "units",
    "lithology_source",
    "topography",
    "extent",
    "resolution",
    "max_nodes",
    "faults",
    "lithology",
    "cover",
    "weathering",
    "interpolation",
    "density_source",
    "extra_intervals",
    "defaults",
)
_FAULTS = (
    "blocks",
    "fault_zone_units",
    "min_picks",
    "min_holes_per_block",
    "post_fault",
    "surfaces",
)
_FAULT = ("name", "between", "prior", "replace_with", "terminates_on")
_LITHOLOGY = ("events", "cap", "kernel", "max_extrapolation", "intrusion_order")
_EVENT = ("mode", "units", "anisotropy", "thickness")
_INTERPOLATION = ("method", "variogram", "drift", "spacing", "keep_conflicts")
_MODES = ("deepest", "first_run")
_ABSENT = ("unknown", "zero")


def load_spec(source: str | Path | Mapping, *, base_dir: Path | None = None) -> GeologySpec:
    """Validate a build spec. ``source`` is a JSON file or its content.

    Relative paths are taken from ``base_dir``; default: the folder of the spec file, or
    the working directory for a mapping. Raises :class:`SpecError` with the field and the
    allowed values. The files the spec names must exist; they are read by
    :func:`build_geology`, except the unit dictionary, which is read here.
    """
    if isinstance(source, str | Path):
        path = Path(source)
        if not path.is_file():
            raise SpecError(f"spec file {path} does not exist")
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            raise SpecError(f"{path}: not valid JSON ({error})") from None
        base = Path(base_dir) if base_dir is not None else path.parent
    else:
        data = source
        base = Path(base_dir) if base_dir is not None else Path.cwd()
    data = _mapping(data, "spec", _TOP)
    if data.get("format", FORMAT) != FORMAT:
        raise _fail("format", f"must be {FORMAT!r}, got {data.get('format')!r}")
    if data.get("version", VERSION) != VERSION:
        raise _fail("version", f"this build reads version {VERSION}, got {data.get('version')!r}")
    defaults: list[tuple[str, str, str]] = []
    doc: dict[str, Any] = {"format": FORMAT, "version": VERSION}

    if "name" not in data:
        raise _fail("name", "is required (the name of the geology model)")
    name = doc["name"] = _text(data["name"], "name")
    if len(name) > 80:
        raise _fail("name", f"at most 80 characters, got {len(name)}")

    database = None
    if data.get("database") is not None:
        database = _path(data["database"], "database", base, must_exist="folder")
        if not (database / "database.json").is_file():
            raise _fail("database", f"{database} is not a drillhole database (no database.json)")
        doc["database"] = str(data["database"])
    else:
        doc["database"] = None

    if "units" not in data or data["units"] is None:
        raise _fail("units", "is required: the path of units.json or the dictionary itself")
    try:
        if isinstance(data["units"], Mapping):
            units = UnitDictionary.from_dict(data["units"])
            units_text = "the dictionary written in the build spec"
            doc["units"] = units.to_dict()
        else:
            file = _path(data["units"], "units", base)
            units = UnitDictionary.load(file)
            units_text = f"file {file.name}"
            doc["units"] = str(data["units"])
    except UnitsError as error:
        raise _fail("units", str(error)) from None
    all_codes = units.codes()
    modelled = units.codes(*MODELLED_KINDS)

    if data.get("lithology_source") is not None:
        at = _mapping(data["lithology_source"], "lithology_source", ("table", "column"))
    else:
        at = {}
    table = at.get("table") or units.source_table
    column = at.get("column") or units.source_column
    if not table:
        table = "geology"
        defaults.append(("lithology_source.table", "geology", "the rock-type log"))
    if not column:
        raise _fail(
            "lithology_source.column",
            "is required: the dictionary names no source column either",
        )
    table, column = _text(table, "lithology_source.table"), _text(column, "lithology_source.column")
    doc["lithology_source"] = {"table": table, "column": column}

    topography, topography_source = None, None
    given = data.get("topography")
    if isinstance(given, Mapping):
        given = _mapping(given, "topography", ("file", "source"))
        if "file" not in given:
            raise _fail("topography", "an object needs 'file' (and optionally 'source')")
        topography = _path(given["file"], "topography.file", base)
        if given.get("source") is not None:
            topography_source = _text(given["source"], "topography.source")
        doc["topography"] = {"file": str(given["file"]), "source": topography_source}
    elif given is not None:
        topography = _path(given, "topography", base)
        doc["topography"] = str(given)
    else:
        doc["topography"] = None

    extent = None
    if data.get("extent") is not None:
        given = data["extent"]
        if isinstance(given, Mapping) and "model" in given:
            given = _mapping(given, "extent", ("model", "padding"))
            model = _path(given["model"], "extent.model", base, must_exist="folder")
            padding = _number(given.get("padding", 0.0), "extent.padding", minimum=0.0)
            extent = {"model": model, "padding": padding}
            doc["extent"] = {"model": str(given["model"]), "padding": padding}
        else:
            given = _mapping(given, "extent", ("rectangle", "z"))
            what = "needs 'rectangle' [x0, y0, x1, y1] and 'z' [z0, z1], or {'model', 'padding'}"
            if "rectangle" not in given or "z" not in given:
                raise _fail("extent", what)
            rect, z = given["rectangle"], given["z"]
            if not isinstance(rect, Sequence) or isinstance(rect, str) or len(rect) != 4:
                raise _fail("extent.rectangle", "must be four numbers: x0, y0, x1, y1")
            if not isinstance(z, Sequence) or isinstance(z, str) or len(z) != 2:
                raise _fail("extent.z", "must be two numbers: z0, z1")
            rect = [_number(v, "extent.rectangle") for v in rect]
            z = [_number(v, "extent.z") for v in z]
            if not (rect[2] > rect[0] and rect[3] > rect[1]):
                raise _fail("extent.rectangle", "needs x0 < x1 and y0 < y1")
            if not z[1] > z[0]:
                raise _fail("extent.z", "needs z0 < z1")
            extent = doc["extent"] = {"rectangle": rect, "z": z}
    else:
        doc["extent"] = None
        defaults.append(
            ("extent", "the box of the hole traces grown by 10 % on every side", "model extent")
        )

    resolution = None
    if data.get("resolution") is not None:
        resolution = _number(data["resolution"], "resolution", positive=True)
    else:
        defaults.append(
            ("resolution", "the shortest horizontal side of the extent / 80", "grid of the model")
        )
    doc["resolution"] = resolution
    if data.get("max_nodes") is not None:
        max_nodes = _integer(data["max_nodes"], "max_nodes", minimum=8)
    else:
        max_nodes = DEFAULT_MAX_NODES
        defaults.append(("max_nodes", str(max_nodes), "largest grid before it is coarsened"))
    doc["max_nodes"] = max_nodes

    # ------------------------------------------------------------------ lithology
    lithology = _mapping(data.get("lithology") or {}, "lithology", _LITHOLOGY)
    cap = None
    if lithology.get("cap") is not None:
        cap = _number(lithology["cap"], "lithology.cap", positive=True)
    if lithology.get("kernel") is not None:
        kernel = _choice(lithology["kernel"], "lithology.kernel", KERNELS)
    else:
        kernel = "linear"
        defaults.append(("lithology.kernel", "linear", "rock-unit fields"))
    max_extrapolation = None
    if lithology.get("max_extrapolation") is not None:
        max_extrapolation = _number(
            lithology["max_extrapolation"], "lithology.max_extrapolation", positive=True
        )
    events: list[EventSpec] | None = None
    event_docs: list[dict[str, Any]] | None = None
    if lithology.get("events") is not None:
        from bakumine.core.geometry.implicit_request import anisotropy_from_spec

        listed = lithology["events"]
        if not isinstance(listed, list) or not listed:
            raise _fail("lithology.events", "must be a list of at least one event")
        events, event_docs, seen = [], [], set()
        for i, entry in enumerate(listed):
            where = f"lithology.events[{i}]"
            entry = _mapping(entry, where, _EVENT)
            mode = _choice(entry.get("mode"), f"{where}.mode", EVENT_MODES)
            if "units" not in entry:
                raise _fail(f"{where}.units", "is required")
            codes = _codes(entry["units"], f"{where}.units", modelled, "modelled units")
            if mode == "intrusion" and len(codes) != 1:
                raise _fail(f"{where}.units", "an intrusion event has exactly one unit")
            if mode != "intrusion" and len(codes) < 2 and mode == "sequence":
                raise _fail(f"{where}.units", "a sequence needs at least two units, top to bottom")
            again = seen & set(codes)
            if again:
                raise _fail(f"{where}.units", f"{', '.join(sorted(again))} is in an earlier event")
            seen |= set(codes)
            anisotropy = None
            if entry.get("anisotropy") is not None:
                try:
                    anisotropy = anisotropy_from_spec(entry["anisotropy"])
                except (ValueError, TypeError) as error:
                    raise _fail(f"{where}.anisotropy", str(error)) from None
            thickness = None
            if entry.get("thickness") is not None:
                if mode != "sequence":
                    raise _fail(f"{where}.thickness", "applies to a sequence event only")
                given = _mapping(entry["thickness"], f"{where}.thickness", codes)
                thickness = {
                    u: _number(v, f"{where}.thickness.{u}", positive=True) for u, v in given.items()
                }
            settings = FieldSettings(anisotropy=anisotropy, cap=cap, kernel=kernel)
            events.append(EventSpec(mode, codes, settings, thickness))
            event_docs.append(
                {
                    "mode": mode,
                    "units": list(codes),
                    "anisotropy": entry.get("anisotropy"),
                    **({"thickness": thickness} if thickness else {}),
                }
            )
    intrusion_order = None
    if lithology.get("intrusion_order") is not None:
        intrusion_order = _codes(
            lithology["intrusion_order"],
            "lithology.intrusion_order",
            units.codes("intrusion"),
            "intrusion units",
        )
    doc["lithology"] = {
        "events": event_docs,
        "cap": cap,
        "kernel": kernel,
        "max_extrapolation": max_extrapolation,
        "intrusion_order": None if intrusion_order is None else list(intrusion_order),
    }

    # ------------------------------------------------------------------ faults
    faults = None
    doc["faults"] = None
    if data.get("faults") is not None:
        given = _mapping(data["faults"], "faults", _FAULTS)
        blocks = None
        if given.get("blocks") is not None:
            blocks = _table_column(given["blocks"], "faults.blocks", ("pattern",))
            if "pattern" not in blocks:
                raise _fail(
                    "faults.blocks.pattern",
                    "is required: a regular expression whose first group is the fault block, "
                    r"for example ^(FB\d)-",
                )
            try:
                compiled = re.compile(_text(blocks["pattern"], "faults.blocks.pattern"))
            except re.error as error:
                raise _fail(
                    "faults.blocks.pattern", f"not a regular expression ({error})"
                ) from None
            if compiled.groups < 1:
                raise _fail("faults.blocks.pattern", "needs one group (...) for the block name")
        zone_kind = units.codes("fault-zone")
        if given.get("fault_zone_units") is not None:
            zone_units = (
                ()
                if given["fault_zone_units"] == []
                else _codes(
                    given["fault_zone_units"], "faults.fault_zone_units", all_codes, "units"
                )
            )
        else:
            zone_units = zone_kind
            defaults.append(
                (
                    "faults.fault_zone_units",
                    ", ".join(zone_kind) or "none",
                    "logged fault zones that place a fault (the units of kind fault-zone)",
                )
            )
        if given.get("min_picks") is not None:
            min_picks = _integer(given["min_picks"], "faults.min_picks")
        else:
            min_picks = DEFAULT_MIN_PICKS
            defaults.append(
                ("faults.min_picks", str(min_picks), "crossings needed to fit a fault plane")
            )
        if given.get("min_holes_per_block") is not None:
            min_holes = _integer(given["min_holes_per_block"], "faults.min_holes_per_block")
        else:
            min_holes = DEFAULT_MIN_HOLES
            defaults.append(
                (
                    "faults.min_holes_per_block",
                    str(min_holes),
                    "holes a fault block needs for a rock model of its own",
                )
            )
        post_fault = None
        if given.get("post_fault") is not None:
            post_fault = (
                ()
                if given["post_fault"] == []
                else _codes(given["post_fault"], "faults.post_fault", modelled, "modelled units")
            )
        listed = given.get("surfaces")
        if not isinstance(listed, list) or not listed:
            raise _fail("faults.surfaces", "must list at least one fault (or set faults to null)")
        surfaces: list[FaultSpec] = []
        surface_docs = []
        for i, entry in enumerate(listed):
            where = f"faults.surfaces[{i}]"
            entry = _mapping(entry, where, _FAULT)
            fault_name = _text(entry.get("name"), f"{where}.name")
            if not _NAME.match(fault_name):
                raise _fail(f"{where}.name", "use letters, digits, '_' and '-' only")
            if fault_name in [s.name for s in surfaces]:
                raise _fail(f"{where}.name", f"{fault_name!r} is listed twice")
            between = None
            if entry.get("between") is not None:
                pair = entry["between"]
                ok = isinstance(pair, list) and len(pair) == 2
                if not ok or not all(isinstance(b, str) and b for b in pair) or pair[0] == pair[1]:
                    raise _fail(f"{where}.between", "must name two different fault blocks")
                between = (pair[0], pair[1])
            prior = None
            if entry.get("prior") is not None:
                p = _mapping(entry["prior"], f"{where}.prior", ("dip", "dip_azimuth"))
                if "dip" not in p or "dip_azimuth" not in p:
                    raise _fail(f"{where}.prior", "needs 'dip' and 'dip_azimuth' in degrees")
                prior = (
                    _number(p["dip"], f"{where}.prior.dip", minimum=0.0),
                    _number(p["dip_azimuth"], f"{where}.prior.dip_azimuth"),
                )
                if prior[0] > 90.0:
                    raise _fail(f"{where}.prior.dip", "must be from 0 to 90")
            replace, replace_doc = None, None
            if entry.get("replace_with") is not None:
                r = _mapping(
                    entry["replace_with"], f"{where}.replace_with", ("file", "points", "plane")
                )
                if len(r) != 1:
                    raise _fail(f"{where}.replace_with", "give exactly one of file, points, plane")
                kind, value = next(iter(r.items()))
                if kind == "file":
                    replace = ("file", _path(value, f"{where}.replace_with.file", base))
                    replace_doc = {"file": str(value)}
                elif kind == "points" and isinstance(value, list):
                    try:
                        xyz = np.asarray(value, dtype=np.float64)
                    except (TypeError, ValueError):
                        xyz = np.zeros(0)
                    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) < 3:
                        raise _fail(
                            f"{where}.replace_with.points",
                            "must be a file or a list of at least three [x, y, z]",
                        )
                    replace = ("points", xyz)
                    replace_doc = {"points": xyz.tolist()}
                elif kind == "points":
                    replace = ("points", _path(value, f"{where}.replace_with.points", base))
                    replace_doc = {"points": str(value)}
                else:
                    p = _mapping(
                        value, f"{where}.replace_with.plane", ("point", "dip", "dip_azimuth")
                    )
                    if set(p) != {"point", "dip", "dip_azimuth"}:
                        raise _fail(
                            f"{where}.replace_with.plane",
                            "needs 'point' [x, y, z], 'dip' and 'dip_azimuth'",
                        )
                    point = p["point"]
                    if not isinstance(point, list) or len(point) != 3:
                        raise _fail(f"{where}.replace_with.plane.point", "must be [x, y, z]")
                    point = [_number(v, f"{where}.replace_with.plane.point") for v in point]
                    dip = _number(p["dip"], f"{where}.replace_with.plane.dip", minimum=0.0)
                    azimuth = _number(p["dip_azimuth"], f"{where}.replace_with.plane.dip_azimuth")
                    replace = ("plane", (point, dip, azimuth))
                    replace_doc = {"plane": {"point": point, "dip": dip, "dip_azimuth": azimuth}}
            if replace is None and between is None:
                raise _fail(
                    f"{where}.between",
                    "is required for a fault inferred from the logs (the two blocks)",
                )
            if replace is None and blocks is None:
                raise _fail(
                    "faults.blocks",
                    f"is required: fault {fault_name} is inferred from the fault-block "
                    "codes (or give it a replace_with surface)",
                )
            ends = None
            if entry.get("terminates_on") is not None:
                t = _mapping(entry["terminates_on"], f"{where}.terminates_on", ("fault", "side"))
                if t.get("side") not in (1, -1) or not isinstance(t.get("fault"), str):
                    raise _fail(
                        f"{where}.terminates_on",
                        "needs 'fault' (a fault name) and 'side' (1 or -1)",
                    )
                ends = (t["fault"], int(t["side"]))
            surfaces.append(FaultSpec(fault_name, between, prior, replace, ends))
            surface_docs.append(
                {
                    "name": fault_name,
                    "between": None if between is None else list(between),
                    "prior": None if prior is None else {"dip": prior[0], "dip_azimuth": prior[1]},
                    "replace_with": replace_doc,
                    "terminates_on": None if ends is None else {"fault": ends[0], "side": ends[1]},
                }
            )
        names = [s.name for s in surfaces]
        for i, s in enumerate(surfaces):
            if s.terminates_on is not None and (
                s.terminates_on[0] not in names or s.terminates_on[0] == s.name
            ):
                raise _fail(
                    f"faults.surfaces[{i}].terminates_on.fault",
                    f"must be another fault of the spec: {', '.join(names)}",
                )
        faults = FaultsSpec(
            None if blocks is None else {k: str(v) for k, v in blocks.items()},
            zone_units,
            min_picks,
            min_holes,
            post_fault,
            tuple(surfaces),
        )
        doc["faults"] = {
            "blocks": faults.blocks,
            "fault_zone_units": list(zone_units),
            "min_picks": min_picks,
            "min_holes_per_block": min_holes,
            "post_fault": None if post_fault is None else list(post_fault),
            "surfaces": surface_docs,
        }

    # ------------------------------------------------------------------ surfaces
    def base_sources(listed: Any, where: str, default_mode: str) -> tuple[Any, str, str, list]:
        if not isinstance(listed, list) or not listed:
            raise _fail(
                where, "must be a list of sources: {'units': [...]} or {'threshold': {...}}"
            )
        sources, docs = [], []
        mode: str | None = None
        absent: str | None = None
        for i, entry in enumerate(listed):
            at = f"{where}[{i}]"
            entry = _mapping(entry, at, ("units", "mode", "absent", "threshold"))
            if ("units" in entry) == ("threshold" in entry):
                raise _fail(at, "give exactly one of 'units' (logged codes) and 'threshold'")
            if "units" in entry:
                codes = _codes(entry["units"], f"{at}.units", all_codes, "units")
                sources.append(SurfaceSource(units=codes))
                docs.append({"units": list(codes)})
                for key, allowed in (("mode", _MODES), ("absent", _ABSENT)):
                    if entry.get(key) is None:
                        continue
                    value = _choice(entry[key], f"{at}.{key}", allowed)
                    earlier = mode if key == "mode" else absent
                    if earlier is not None and earlier != value:
                        raise _fail(
                            f"{at}.{key}",
                            f"differs from an earlier source ({earlier}): one surface has one rule",
                        )
                    if key == "mode":
                        mode = value
                    else:
                        absent = value
            else:
                if entry.get("mode") is not None or entry.get("absent") is not None:
                    raise _fail(at, "'mode' and 'absent' go with a 'units' source")
                rule = _table_column(entry["threshold"], f"{at}.threshold", ("below", "above"))
                if (rule.get("below") is None) == (rule.get("above") is None):
                    raise _fail(f"{at}.threshold", "give exactly one of 'below' and 'above'")
                key = "below" if rule.get("below") is not None else "above"
                value = _number(rule[key], f"{at}.threshold.{key}")
                sources.append(
                    SurfaceSource(table=rule["table"], column=rule["column"], **{key: value})
                )
                docs.append(
                    {"threshold": {"table": rule["table"], "column": rule["column"], key: value}}
                )
        if mode is None:
            mode = default_mode
            if any(s.units for s in sources):
                label = "cover.mode" if where == "cover" else f"{where}[].mode"
                defaults.append((label, mode, "which logged run of the units gives the base"))
        if absent is None:
            absent = "unknown"
        for item in docs:
            if "units" in item:
                item.update(mode=mode, absent=absent)
        return tuple(sources), mode, absent, docs

    cover = None
    doc["cover"] = None
    if data.get("cover") is not None:
        given = _mapping(data["cover"], "cover", ("unit", "mode", "absent"))
        code = given.get("unit")
        if code not in units.codes("cover"):
            raise _fail(
                "cover.unit",
                f"must be a unit of kind cover: "
                f"{', '.join(units.codes('cover')) or 'the dictionary has none'}",
            )
        for key, allowed in (("mode", _MODES), ("absent", _ABSENT)):
            if given.get(key) is not None:
                _choice(given[key], f"cover.{key}", allowed)
        entry = {"units": [code], "mode": given.get("mode"), "absent": given.get("absent")}
        sources, mode, absent, _ = base_sources([entry], "cover", "first_run")
        cover = SurfaceSpec(f"base:{code}", code, "cover", sources, mode, absent)
        doc["cover"] = {"unit": code, "mode": mode, "absent": absent}
    zones = [zone.code for zone in units.weathering_zones]
    weathering: list[SurfaceSpec] = []
    doc["weathering"] = []
    if data.get("weathering"):
        listed = data["weathering"]
        if not isinstance(listed, list):
            raise _fail("weathering", "must be a list of {'zone', 'base'}")
        expected = zones[:-1]
        for i, entry in enumerate(listed):
            where = f"weathering[{i}]"
            entry = _mapping(entry, where, ("zone", "base"))
            if i >= len(expected) or entry.get("zone") != expected[i]:
                raise _fail(
                    f"{where}.zone",
                    "the zones are listed top to bottom as in the dictionary, the deepest one "
                    f"(no base) left out: {', '.join(expected) or 'the dictionary has none'}; "
                    f"got {entry.get('zone')!r}",
                )
            sources, mode, absent, docs = base_sources(
                entry.get("base"), f"{where}.base", "deepest"
            )
            zone = expected[i]
            weathering.append(
                SurfaceSpec(f"base:{zone}", zone, "weathering", sources, mode, absent)
            )
            doc["weathering"].append({"zone": zone, "base": docs})
        if len(weathering) != len(expected):
            raise _fail(
                "weathering",
                f"every zone but the deepest needs a base: "
                f"{', '.join(expected)}; got {', '.join(s.code for s in weathering)}",
            )
    if (cover is not None or weathering) and topography is None:
        raise _fail(
            "topography",
            "is required when cover or weathering is built: their bases "
            "are depths below the topography (a DTM file: OBJ, DXF, .dtm)",
        )

    given = _mapping(data.get("interpolation") or {}, "interpolation", _INTERPOLATION)
    interpolation: dict[str, Any] = {}
    surface_defaults = []
    if given.get("method") is not None:
        interpolation["method"] = _choice(
            given["method"], "interpolation.method", ("kriging", "rbf")
        )
    else:
        interpolation["method"] = "kriging"
        surface_defaults.append(("interpolation.method", "kriging", "base surfaces"))
    if given.get("drift") is not None:
        interpolation["drift"] = _choice(
            given["drift"], "interpolation.drift", ("constant", "linear")
        )
    else:
        interpolation["drift"] = "constant"
        surface_defaults.append(("interpolation.drift", "constant", "base surfaces"))
    interpolation["variogram"] = None
    if given.get("variogram") is not None:
        if interpolation["method"] == "rbf":
            raise _fail("interpolation.variogram", "does not apply with method rbf")
        from bakumine.core.estimation.variogram import VariogramModel

        try:
            VariogramModel.from_dict(given["variogram"])
        except (ValueError, TypeError, KeyError) as error:
            raise _fail("interpolation.variogram", f"not a variogram model ({error})") from None
        interpolation["variogram"] = dict(given["variogram"])
    interpolation["spacing"] = None
    if given.get("spacing") is not None:
        interpolation["spacing"] = _number(given["spacing"], "interpolation.spacing", positive=True)
    else:
        surface_defaults.append(
            ("interpolation.spacing", "2 x the grid resolution", "grid of the base surfaces")
        )
    interpolation["keep_conflicts"] = bool(given.get("keep_conflicts", False))
    if not isinstance(given.get("keep_conflicts", False), bool):
        raise _fail("interpolation.keep_conflicts", "must be true or false")
    if cover is not None or weathering:
        defaults.extend(surface_defaults)
    doc["interpolation"] = dict(interpolation)

    density_source = None
    if data.get("density_source") is not None:
        density_source = _table_column(data["density_source"], "density_source")
    doc["density_source"] = density_source

    extra: list[ExtraIntervals] = []
    doc["extra_intervals"] = []
    surface_names = [s.name for s in ([cover] if cover else []) + weathering]
    for i, entry in enumerate(data.get("extra_intervals") or []):
        where = f"extra_intervals[{i}]"
        entry = _mapping(entry, where, ("file", "rows", "units_column", "surfaces"))
        if ("file" in entry) == ("rows" in entry):
            raise _fail(where, "give exactly one of 'file' (a CSV) and 'rows'")
        units_column = _text(entry.get("units_column", "code"), f"{where}.units_column")
        only = None
        if entry.get("surfaces") is not None:
            only = entry["surfaces"]
            if not isinstance(only, list) or any(s not in surface_names for s in only):
                raise _fail(
                    f"{where}.surfaces",
                    "must list surfaces of this spec: "
                    f"{', '.join(surface_names) or 'none is built'}",
                )
            only = tuple(only)
        item = {"units_column": units_column, "surfaces": None if only is None else list(only)}
        if "file" in entry:
            file = _path(entry["file"], f"{where}.file", base)
            extra.append(ExtraIntervals(f"file {file.name}", file, None, units_column, only))
            doc["extra_intervals"].append({"file": str(entry["file"]), **item})
        else:
            rows = entry["rows"]
            needed = ("hole_id", "depth_from", "depth_to", units_column)
            if not isinstance(rows, list) or not all(
                isinstance(r, Mapping) and all(k in r for k in needed) for r in rows
            ):
                raise _fail(f"{where}.rows", f"must be a list of objects with {', '.join(needed)}")
            rows = tuple(dict(r) for r in rows)
            extra.append(
                ExtraIntervals(
                    f"rows of the build spec ({len(rows)})", None, rows, units_column, only
                )
            )
            doc["extra_intervals"].append({"rows": [dict(r) for r in rows], **item})

    # a spec as run (spec.json) has its defaults written out and lists them: they stay
    # defaults when it is run again, so the provenance record does not change
    listed = data.get("defaults") or []
    if not isinstance(listed, list) or not all(
        isinstance(d, Mapping)
        and all(isinstance(d.get(k), str) for k in ("field", "value", "used_for"))
        for d in listed
    ):
        raise _fail(
            "defaults",
            "must be a list of {'field', 'value', 'used_for'} (texts), as written in spec.json",
        )
    found = {name for name, _, _ in defaults}
    defaults.extend(
        (d["field"], d["value"], d["used_for"]) for d in listed if d["field"] not in found
    )
    doc["defaults"] = [
        {"field": name, "value": value, "used_for": used_for} for name, value, used_for in defaults
    ]

    return GeologySpec(
        name=name,
        database=database,
        units=units,
        units_text=units_text,
        table=table,
        column=column,
        topography=topography,
        topography_source=topography_source,
        extent=extent,
        resolution=resolution,
        max_nodes=max_nodes,
        faults=faults,
        events=None if events is None else tuple(events),
        intrusion_order=intrusion_order,
        cap=cap,
        kernel=kernel,
        max_extrapolation=max_extrapolation,
        cover=cover,
        weathering=tuple(weathering),
        interpolation=interpolation,
        density_source=density_source,
        extra_intervals=tuple(extra),
        document=doc,
        defaults=tuple(defaults),
        base_dir=base,
    )


# --------------------------------------------------------------------------
# JSON
# --------------------------------------------------------------------------


def json_safe(value: Any) -> Any:
    """``value`` as standard JSON data: numpy types converted, NaN and infinity ``None``."""
    if value is None or isinstance(value, bool | str):
        return value
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, int):
        return value
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return json_safe(value.tolist())
    if isinstance(value, pd.DataFrame):
        return json_safe(value.to_dict(orient="records"))
    if isinstance(value, list | tuple | set | frozenset):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def _write_json(path: Path, data: Any) -> None:
    text = json.dumps(json_safe(data), indent=2, allow_nan=False, ensure_ascii=False)
    path.write_text(text + "\n", encoding="utf-8")


# --------------------------------------------------------------------------
# the build
# --------------------------------------------------------------------------


@dataclass
class BuildRecord:
    """What :func:`build_geology` keeps on the model (``model.build``) for
    :func:`save_geology`: the spec and the fitted surfaces with their picks."""

    spec: GeologySpec
    surfaces: list[Any] = field(default_factory=list)


def _entry(item: str, kind: str, source: str, used_for: str, detail: str = "", **extra: Any):
    extra.setdefault("task", _TASK)
    return ProvenanceEntry(item, kind, source, used_for, detail=detail, **extra)


def _interval_table(database: Any, table: str, column: str, where: str) -> pd.DataFrame:
    if table not in database.intervals:
        have = ", ".join(sorted(database.intervals)) or "none"
        raise SpecError(f"{where}: the database has no interval table {table!r} (tables: {have})")
    frame = database.intervals[table]
    if column not in frame.columns:
        have = ", ".join(str(c) for c in frame.columns)
        raise SpecError(f"{where}: table {table!r} has no column {column!r} (columns: {have})")
    return frame


def _nice_up(value: float) -> float:
    """``value`` rounded up to two significant digits."""
    scale = 10.0 ** (math.floor(math.log10(value)) - 1)
    return math.ceil(value / scale - 1e-9) * scale


def _grid(lo: np.ndarray, hi: np.ndarray, resolution: float, max_nodes: int):
    from bakumine.core.geometry.isosurface import GridSpec

    h = float(resolution)
    for _ in range(64):
        shape = tuple(int(n) for n in np.maximum(np.ceil((hi - lo) / h - 1e-9), 1).astype(int) + 1)
        nodes = int(np.prod(shape, dtype=np.int64))
        if nodes <= max_nodes:
            return GridSpec(lo, (h, h, h), shape), h
        h = _nice_up(h * max((nodes / max_nodes) ** (1.0 / 3.0), 1.02))
    raise SpecError("max_nodes: no grid of the extent fits; raise max_nodes")


def _event_order(
    events: Sequence[EventSpec], order: Sequence[str], younger: Sequence[str]
) -> list[EventSpec]:
    """Oldest first: the other events as listed, then the intrusions in age order; within
    that, the events younger than the faults last."""
    age = {u: i for i, u in enumerate(order)}
    rank = []
    for i, event in enumerate(events):
        intrusion = event.mode == "intrusion"
        post = all(u in younger for u in event.units) and bool(younger)
        rank.append((post, intrusion, age[event.units[0]] if intrusion else i, i))
    ordered = [events[i] for *_, i in sorted(rank)]
    listed = [e.units[0] for e in ordered if e.mode == "intrusion"]
    if listed != [u for u in order if u in listed]:
        raise SpecError(
            "faults.post_fault: an intrusion younger than the faults cannot be older than one "
            f"the faults cut; age order {' < '.join(order)}, younger than the faults: "
            f"{', '.join(younger)}"
        )
    return ordered


def _hole_samples(table: pd.DataFrame, known: np.ndarray, traces: Any, step: float):
    """Points every ``<= step`` metres inside the known intervals: ``(xyz, row)``."""
    a = pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(dtype=np.float64)
    rows = np.flatnonzero(np.asarray(known, dtype=bool) & np.isfinite(a) & np.isfinite(b) & (b > a))
    if len(rows) == 0:
        return np.zeros((0, 3)), rows
    n = np.maximum(1, np.ceil((b[rows] - a[rows]) / step - 1e-9).astype(np.int64))
    rep = np.repeat(np.arange(len(rows)), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]])
    position = np.arange(len(rep)) - start[rep] + 0.5
    depth = a[rows][rep] + (b[rows][rep] - a[rows][rep]) * position / n[rep]
    hole = table["hole_id"].to_numpy(dtype=object)[rows][rep]
    xyz, _ = traces.xyz_at(hole, depth, depth_tolerance=1e-6)
    ok = np.isfinite(xyz).all(axis=1)
    return xyz[ok], rows[rep][ok]


def _residuals(model: Any, table: pd.DataFrame, rock: pd.Series, traces: Any) -> dict[str, Any]:
    """The logged unit against the model along the holes (grid labels)."""
    pts, row = _hole_samples(table, rock.notna().to_numpy(), traces, RESIDUAL_STEP)
    if len(pts) == 0:
        return {"points": 0, "reproduced": None, "units": {}}
    logged = rock.to_numpy(dtype=object)[row]
    got = model.decode(model.labels(pts, "lith"), "lith")
    same = got == logged
    out: dict[str, Any] = {
        "points": int(len(pts)),
        "step": RESIDUAL_STEP,
        "reproduced": float(same.mean()),
        "note": "logged rock unit against the grid labels of the model at points along the "
        "holes; the cover and the air are labelled by their surfaces and count as differences",
        "units": {},
    }
    for unit in dict.fromkeys(logged):
        of = logged == unit
        out["units"][str(unit)] = {"points": int(of.sum()), "reproduced": float(same[of].mean())}
    return out


def _volumes(model: Any) -> dict[str, Any]:
    """Volume per rock unit: whole grid, inside the drilled hull, and outside it.

    One point per grid cell, grid labels. The point of a cell is shifted in plan from
    level to level (``r = k mod 31``, shift ``((r + 0.5) / 31, ((12 r mod 31) + 0.5) / 31)``
    cells, the rule of the synthetic truth, geology_contracts.md 13): a plain lattice meets
    a steep sheet thinner than two cells at one phase and is then off by 10 to 20 %. The
    drilled hull is the convex hull of the logged hole points.
    """
    from scipy.spatial import Delaunay, QhullError

    grid = model.grid
    nx, ny, nz = (n - 1 for n in grid.shape)
    cell = float(np.prod(grid.spacing))
    categories = model.categories("lith")
    total = np.zeros(len(categories) + 1)
    inside = np.zeros(len(categories) + 1)
    hull = None
    if model.hole_points is not None and len(model.hole_points) >= 4:
        try:
            hull = Delaunay(model.hole_points)
        except QhullError:
            hull = None
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny), indexing="ij")
    for k in range(nz):
        r = k % 31
        x = grid.origin[0] + grid.spacing[0] * (ii.ravel() + (r + 0.5) / 31.0)
        y = grid.origin[1] + grid.spacing[1] * (jj.ravel() + ((12 * r) % 31 + 0.5) / 31.0)
        z = np.full(x.size, grid.origin[2] + grid.spacing[2] * (k + 0.5))
        pts = np.column_stack([x, y, z])
        labels = np.asarray(model.labels(pts, "lith")).astype(np.int64)
        labels = np.where(labels < 0, len(categories), labels)
        total += np.bincount(labels, minlength=len(categories) + 1)
        if hull is not None:
            in_hull = hull.find_simplex(pts) >= 0
            inside += np.bincount(labels[in_hull], minlength=len(categories) + 1)
    units = {}
    for i, code in enumerate((*categories, "(none)")):
        if total[i] == 0 and code == "(none)":
            continue
        units[code] = {
            "volume": float(total[i] * cell),
            "inside_hull": float(inside[i] * cell) if hull is not None else None,
            "extrapolated": float((total[i] - inside[i]) * cell) if hull is not None else None,
            "extrapolated_share": (
                float((total[i] - inside[i]) / total[i]) if hull is not None and total[i] else None
            ),
        }
    return {
        "method": "one point per grid cell, shifted in plan from level to level, grid labels; "
        "the drilled hull is the convex hull of the logged hole points",
        "cell_volume": cell,
        "hull": hull is not None,
        "units": units,
    }


def _load_fault(item: FaultSpec, have_points: bool):
    from bakumine.core.geology.faults import Plane, import_fault

    kind, value = item.replace_with  # type: ignore[misc]
    blocks = None if have_points else item.between
    if kind == "plane":
        point, dip, azimuth = value
        return import_fault(item.name, plane=Plane.from_dip(point, dip, azimuth), blocks=blocks)
    if kind == "file":
        from bakumine.core.io.files import read_mesh

        return import_fault(item.name, mesh=read_mesh(value), blocks=blocks)
    if isinstance(value, Path):
        from bakumine.core.geometry.surface_points import read_points_table

        value, _ = read_points_table(value)
    return import_fault(item.name, points=np.asarray(value, dtype=np.float64), blocks=blocks)


def _extra_table(item: ExtraIntervals) -> pd.DataFrame:
    if item.file is not None:
        from bakumine.core.geology.picks import read_extra_intervals

        return read_extra_intervals(item.file, code=item.units_column)
    frame = pd.DataFrame(list(item.rows or ()))
    frame = frame.rename(columns={item.units_column: "code"})
    for key in ("depth_from", "depth_to"):
        frame[key] = pd.to_numeric(frame[key], errors="coerce")
    return frame[["hole_id", "depth_from", "depth_to", "code"]]


def build_geology(
    spec: GeologySpec,
    *,
    database: DrillholeDatabase | None = None,
    topography: DTM | None = None,
    progress: Callable[[str, float], None] | None = None,
) -> GeologyModel:
    """Build the geology model of ``spec`` (the steps of the module text).

    ``database`` and ``topography`` replace the ones the spec names (objects already in
    memory). The model comes back sampled on its grid, with ``provenance`` (the merged
    record), ``diagnostics`` and ``build`` (a :class:`BuildRecord`) set; write it with
    :func:`save_geology`.
    """
    from bakumine.core.geology import surfaces as sf
    from bakumine.core.geology.lithology import LithologyStack, fit_stack, logged_hole_points
    from bakumine.core.geology.model import GeologyModel
    from bakumine.core.geology.sequence import mask_younger, order_intrusions
    from bakumine.core.geology.units import density_samples_by_unit, units_provenance
    from bakumine.core.geometry.dtm import DTM

    def tell(stage: str, fraction: float) -> None:
        if progress is not None:
            progress(stage, fraction)

    timing: dict[str, float] = {}
    t_start = clock = time.perf_counter()

    def lap(key: str) -> None:
        nonlocal clock
        now = time.perf_counter()
        timing[key] = now - clock
        clock = now

    units = spec.units
    record = Provenance(object_kind="geology", name=spec.name)
    diagnostics: dict[str, Any] = {"name": spec.name}

    # ---- 1. holes and logs
    tell("drillholes", 0.0)
    database_text = "the database handed to the build"
    if database is None:
        if spec.database is None:
            raise SpecError("database: is required (the drillhole database folder)")
        from bakumine.core.drillholes.importer import load_database

        database = load_database(spec.database)
        database_text = f"database {spec.database.name!r}"
    elif spec.database is not None:
        database_text = f"database {spec.database.name!r}"
    table = _interval_table(database, spec.table, spec.column, "lithology_source")
    traces = database.desurvey()
    mapped = units.map_codes(table[spec.column])
    rock = units.mask_kinds(mapped, MODELLED_KINDS)
    logged = [u for u in units.codes(*MODELLED_KINDS) if (rock == u).any()]
    n_holes = len(database.collar)
    n_traced = len(traces.hole_ids)
    logged_holes = int(table.loc[mapped.notna().to_numpy(), "hole_id"].nunique())
    tables = ", ".join(f"{name} {len(frame)}" for name, frame in database.intervals.items())
    record.measured(
        "Drillhole database",
        source=database_text,
        used_for="everything the model is built from",
        detail=f"{n_holes} holes, {n_traced} with a trace; intervals per table: {tables}",
        stage="extent",
        task=_TASK,
        count=n_holes,
    )
    record.measured(
        "Rock-type log",
        source=f"{database_text}, table {spec.table}, column {spec.column}",
        used_for="rock units, cover and weathering picks, fault zones",
        detail=f"{len(table)} intervals, {int(mapped.notna().sum())} with a unit of the "
        f"dictionary, {logged_holes} holes; unit dictionary: {spec.units_text}",
        stage="units",
        task=_TASK,
        count=len(table),
    )
    samples = None
    if spec.density_source is not None:
        d = spec.density_source
        density_table = _interval_table(database, d["table"], d["column"], "density_source")
        samples = density_samples_by_unit(table, spec.column, density_table, d["column"], units)
    record.extend(
        units_provenance(
            table,
            spec.column,
            units,
            source=f"table {spec.table}, column {spec.column}",
            density_samples=samples,
            density_source=(
                "density table"
                if spec.density_source is None
                else f"table {spec.density_source['table']}, column {spec.density_source['column']}"
            ),
        )
    )
    if spec.density_source is None:
        record.missing(
            "Density samples",
            source="a density table of the database (density_source: table and column)",
            used_for="check of the densities of the dictionary, unit by unit",
            detail="none named in the build spec: the densities are the dictionary's",
            stage="density",
            task=_TASK,
        )
    diagnostics["inputs"] = {
        "holes": n_holes,
        "holes_with_trace": n_traced,
        "lithology_intervals": len(table),
        "lithology_intervals_mapped": int(mapped.notna().sum()),
        "rock_intervals": int(rock.notna().sum()),
        "holes_logged": logged_holes,
        "interval_tables": {name: len(frame) for name, frame in database.intervals.items()},
    }

    # ---- events
    if spec.events is not None:
        events = list(spec.events)
    else:
        group = [u for u in units.codes("lithology") if u in logged]
        settings = FieldSettings(cap=spec.cap, kernel=spec.kernel)
        events = [EventSpec("argmax", tuple(group), settings)] if group else []
        events += [
            EventSpec("intrusion", (u,), settings) for u in units.codes("intrusion") if u in logged
        ]
        if not events:
            raise SpecError(
                "lithology.events: none given and no lithology or intrusion unit is logged"
            )
        record.assumed(
            "Rock-unit events",
            source="default (lithology.events not given in the build spec)",
            used_for="how the rock units are modelled",
            detail="; ".join(f"{e.mode}: {', '.join(e.units)}" for e in events)
            + "; no stratigraphic order is assumed (largest field wins)",
            stage="lithology",
            task=_TASK,
        )
    in_events = [u for e in events for u in e.units]
    absent = [u for u in in_events if u not in logged]
    if absent:
        raise SpecError(
            f"lithology.events: unit(s) {', '.join(absent)} are not logged in table "
            f"{spec.table}, column {spec.column}; logged: {', '.join(logged) or 'none'}"
        )
    left_out = [u for u in logged if u not in in_events]
    if left_out:
        record.assumed(
            "Logged rock units without an event",
            source="build spec (lithology.events does not list them)",
            used_for="rock-unit fields",
            detail=f"{', '.join(left_out)}: their intervals count as other rock and get no "
            "volume of their own",
            stage="lithology",
            task=_TASK,
            count=len(left_out),
        )
    intrusions = [e.units[0] for e in events if e.mode == "intrusion"]
    age_order: Sequence[str] | None = None
    if spec.intrusion_order is not None:
        if sorted(spec.intrusion_order) != sorted(intrusions):
            raise SpecError(
                "lithology.intrusion_order: must list every intrusion event once: "
                f"{', '.join(intrusions) or 'there is none'}"
            )
        age_order = spec.intrusion_order
    else:
        ages = [units.unit(u).age for u in intrusions]
        if len(intrusions) > 1 and all(a is not None for a in ages) and len(set(ages)) == len(ages):
            age_order = [u for _, u in sorted(zip(ages, intrusions, strict=True))]
    order, order_entries = order_intrusions(
        [u for u in units.codes("intrusion") if u in intrusions], age_order
    )
    if age_order is not None and spec.intrusion_order is None and order_entries:
        order_entries = [
            _entry(
                e.item,
                e.kind,
                "unit dictionary (the 'age' of the intrusions, larger = younger)",
                e.used_for,
                e.detail,
                stage=e.stage or "lithology",
                task=e.task,
            )
            for e in order_entries
        ]
    post_fault = spec.faults.post_fault if spec.faults is not None else None
    events = _event_order(events, order, post_fault or ())
    lap("read_seconds")

    # ---- 2. topography, extent, grid
    tell("grid", 0.05)
    if topography is None and spec.topography is not None:
        from bakumine.core.geometry.mesh_import import import_mesh

        topography = DTM(import_mesh(spec.topography, "dtm", 1.0)[0])
    if topography is not None:
        mesh = topography.mesh
        where = "handed to the build" if spec.topography is None else f"file {spec.topography.name}"
        origin = spec.topography_source or "origin not stated in the build spec (topography.source)"
        record.measured(
            "Topography",
            source=f"{where}; {origin}",
            used_for="air above the ground; depth of the cover and weathering bases",
            detail=f"{mesh.n_vertices} vertices, {mesh.n_faces} triangles",
            stage="topography",
            task=_TASK,
            count=int(mesh.n_vertices),
        )
    else:
        record.missing(
            "Topography",
            source="a survey or drone surface, a DEM (`bakumine surface from-raster`) or, "
            "failing that, a surface through the collars (`bakumine surface from-points`)",
            used_for="air above the ground; cover and weathering bases",
            detail="none given: the model has no AIR, no cover and no weathering zones",
            stage="topography",
            task=_TASK,
        )
    if spec.extent is None:
        xyz = traces.stations()[["x", "y", "z"]].to_numpy(dtype=np.float64)
        if len(xyz) == 0:
            raise SpecError("extent: is required (the database has no hole with a trace)")
        lo, hi = xyz.min(axis=0), xyz.max(axis=0)
        pad = 0.1 * np.maximum(hi - lo, 1.0)
        lo, hi = lo - pad, hi + pad
        extent_source = "default (the box of the hole traces grown by 10 % on every side)"
    elif "model" in spec.extent:
        from bakumine.core.blockmodel.storage import load_model

        box = np.asarray(load_model(spec.extent["model"]).definition.bounding_box(), dtype=float)
        lo, hi = box[0] - spec.extent["padding"], box[1] + spec.extent["padding"]
        extent_source = (
            f"build spec (the box of block model {spec.extent['model'].name!r}, "
            f"padding {spec.extent['padding']:g} m)"
        )
    else:
        r, z = spec.extent["rectangle"], spec.extent["z"]
        lo, hi = np.array([r[0], r[1], z[0]]), np.array([r[2], r[3], z[1]])
        extent_source = "build spec (extent: rectangle and z range)"
    if spec.resolution is not None:
        resolution, resolution_source = spec.resolution, "build spec (resolution)"
    else:
        resolution = _nice_up(float(min(hi[0] - lo[0], hi[1] - lo[1])) / 80.0)
        resolution_source = "default (the shortest horizontal side of the extent / 80)"
    grid, spacing = _grid(lo, hi, resolution, spec.max_nodes)
    top = grid.origin + grid.spacing * (np.array(grid.shape) - 1)
    record.assumed(
        "Model extent",
        source=extent_source,
        used_for="the volume the model is built and reported in",
        detail=f"x {lo[0]:.1f} to {top[0]:.1f}, y {lo[1]:.1f} to {top[1]:.1f}, "
        f"z {lo[2]:.1f} to {top[2]:.1f}",
        stage="extent",
        task=_TASK,
    )
    coarsened = spacing > resolution * (1.0 + 1e-9)
    record.assumed(
        "Grid resolution",
        source=resolution_source,
        used_for="grid labels of the model (block coding, meshes, volumes)",
        detail=f"{spacing:g} m, {grid.shape[0]} x {grid.shape[1]} x {grid.shape[2]} = "
        f"{grid.n_nodes} nodes"
        + (
            f"; coarsened from {resolution:g} m to stay within max_nodes = {spec.max_nodes}"
            if coarsened
            else ""
        )
        + "; a body thinner than the spacing is not resolved by the grid",
        stage="extent",
        task=_TASK,
        value=float(spacing),
    )
    diagnostics["grid"] = {
        "origin": grid.origin.tolist(),
        "spacing": grid.spacing.tolist(),
        "shape": list(grid.shape),
        "nodes": grid.n_nodes,
        "resolution_requested": resolution,
        "coarsened": bool(coarsened),
    }
    lap("grid_seconds")

    # ---- 3. cover and weathering bases
    tell("surfaces", 0.1)
    built: list[Any] = []
    wanted = ([spec.cover] if spec.cover is not None else []) + list(spec.weathering)
    if wanted:
        from bakumine.core.estimation.variogram import VariogramModel

        rectangle = (float(lo[0]), float(lo[1]), float(top[0]), float(top[1]))
        opts = spec.interpolation
        surface_spacing = opts["spacing"] or 2.0 * spacing
        variogram = None
        if opts["variogram"] is not None:
            variogram = VariogramModel.from_dict(opts["variogram"])
        extras = [(item, _extra_table(item)) for item in spec.extra_intervals]
        for item in wanted:
            sources = []
            for s in item.sources:
                if s.units:
                    sources.append(
                        sf.log_source(
                            table,
                            spec.column,
                            units,
                            s.units,
                            description=f"table {spec.table}, column {spec.column}",
                        )
                    )
                    for extra, frame in extras:
                        codes = set(units.map_codes(frame["code"]).dropna())
                        use = (
                            item.name in extra.surfaces
                            if extra.surfaces is not None
                            else bool(codes & set(s.units))
                        )
                        if use:
                            sources.append(
                                sf.log_source(
                                    frame,
                                    "code",
                                    units,
                                    s.units,
                                    source="extra",
                                    description=extra.label,
                                )
                            )
                else:
                    limit = {"below": s.below} if s.below is not None else {"above": s.above}
                    where = f"{item.role} base {item.code}: threshold"
                    frame = _interval_table(database, s.table, s.column, where)
                    sources.append(
                        sf.threshold_source(
                            frame, s.column, description=f"table {s.table}", **limit
                        )
                    )
            picks = sf.surface_picks(
                sources, traces, topography, surface=item.name, mode=item.mode, absent=item.absent
            )
            built.append(
                sf.depth_surface(
                    picks,
                    topography,
                    rectangle,
                    surface_spacing,
                    name=item.name,
                    method=opts["method"],
                    variogram=variogram,
                    drift=opts["drift"],
                    keep_conflicts=opts["keep_conflicts"],
                )
            )
        built = sf.nest_surfaces(built)
        for surface in built:
            record.extend(surface.provenance)
        diagnostics["surfaces"] = {s.name: s.diagnostics for s in built}
    by_name = {s.name: s for s in built}
    cover = None
    if spec.cover is not None:
        cover = (spec.cover.code, by_name[spec.cover.name].dtm)
    elif units.codes("cover") and (mapped.isin(units.codes("cover"))).any():
        record.missing(
            "Cover base",
            source="the build spec (cover: the cover unit) and a topography",
            used_for="the cover unit of the model",
            detail=f"cover is logged ({', '.join(units.codes('cover'))}) but not built: the "
            "model has rock up to the topography",
            stage="surfaces",
            task=_TASK,
        )
    zones = [zone.code for zone in units.weathering_zones]
    weathering: list[tuple[str, Any]] = []
    if spec.weathering:
        weathering = [(s.code, by_name[s.name].dtm) for s in spec.weathering]
        weathering.append((zones[-1], None))
    elif zones:
        record.missing(
            "Weathering bases",
            source="oxidation logging or sulphur assays, a topography, and the build spec "
            "(weathering: a base per zone)",
            used_for=f"weathering zones {', '.join(zones)} and their densities",
            detail="not built: the model has no weathering layer",
            stage="surfaces",
            task=_TASK,
        )
    lap("surfaces_seconds")

    # ---- 4. faults
    tell("faults", 0.2)
    network = None
    if spec.faults is not None:
        from bakumine.core.geology.faults import (
            FaultNetwork,
            block_codes,
            fault_picks,
            infer_fault,
            labelled_points,
        )

        f = spec.faults
        xyz, block, hole = np.zeros((0, 3)), np.zeros(0, dtype=object), np.zeros(0, dtype=object)
        picks = None
        if f.blocks is not None:
            block_table = _interval_table(
                database, f.blocks["table"], f.blocks["column"], "faults.blocks"
            )
            codes = block_codes(block_table[f.blocks["column"]], f.blocks["pattern"])
            xyz, block, hole = labelled_points(block_table, codes, traces)
            zone_rows = mapped.isin(list(f.fault_zone_units)).to_numpy()
            zone_table = table[zone_rows] if f.fault_zone_units else None
            picks = fault_picks(block_table, codes, traces, fault_zone=zone_table)
            found = sorted({str(b) for b in block})
            record.measured(
                "Fault-block codes",
                source=f"{database_text}, table {f.blocks['table']}, column "
                f"{f.blocks['column']}, pattern {f.blocks['pattern']}",
                used_for="fault blocks; position of the inferred faults",
                detail=f"{int(codes.notna().sum())} intervals with a block "
                f"({', '.join(found) or 'none'}), {len(set(hole))} holes; "
                f"{len(picks)} crossings between blocks; "
                f"{int(zone_rows.sum())} logged fault-zone intervals",
                stage="faults",
                task=_TASK,
                count=int(codes.notna().sum()),
            )
            diagnostics["fault_picks"] = {
                "block_intervals": int(codes.notna().sum()),
                "blocks": found,
                "crossings": int(len(picks)),
                "fault_zone_intervals": int(zone_rows.sum()),
            }
            for i, item in enumerate(f.surfaces):
                unknown = [b for b in item.between or () if b not in found]
                if unknown and item.replace_with is None:
                    raise SpecError(
                        f"faults.surfaces[{i}].between: block(s) {', '.join(unknown)} are not "
                        f"in the fault-block codes: {', '.join(found) or 'none found'}"
                    )
        fitted = []
        for item in f.surfaces:
            if item.replace_with is not None:
                fitted.append(_load_fault(item, len(xyz) > 0))
                continue
            a, b = item.between  # type: ignore[misc]
            in_a, in_b = block == a, block == b
            fitted.append(
                infer_fault(
                    item.name,
                    picks,
                    xyz[in_a],
                    xyz[in_b],
                    blocks=(a, b),
                    prior=item.prior,
                    min_picks=f.min_picks,
                    holes_a=hole[in_a],
                    holes_b=hole[in_b],
                )
            )
        ends = {s.name: s.terminates_on for s in f.surfaces if s.terminates_on is not None}
        network = FaultNetwork.fit(fitted, xyz, block, terminations=ends or None)
    else:
        record.missing(
            "Fault data",
            source="fault-block codes in a log column (faults.blocks), or the geologist's "
            "fault surfaces (DXF / OBJ / points, faults.surfaces[].replace_with)",
            used_for="fault blocks; rock units that step at a fault",
            detail="no faults in the build spec: the rock units are modelled as unfaulted",
            stage="faults",
            task=_TASK,
        )
    lap("faults_seconds")

    # ---- 5. rock units
    tell("rock units", 0.3)
    pooled: list[str] = []
    stack_entries: list[ProvenanceEntry] = []
    if network is not None:
        from bakumine.core.geology.faulted import fit_block_stacks

        stacks, pooled, stack_entries = fit_block_stacks(
            table,
            rock,
            traces,
            network,
            events,
            min_holes=spec.faults.min_holes_per_block,  # type: ignore[union-attr]
            post_fault=None if post_fault is None else list(post_fault),
        )
    else:
        fitted_events = []
        for event in events:
            known = mask_younger(rock, order, event.units[0]) if event.mode == "intrusion" else rock
            fitted_events.append(fit_stack(table, known, traces, [event]).events[0])
        stacks = {"ALL": LithologyStack("ALL", tuple(fitted_events))}
    lap("fit_seconds")

    # ---- 6. the model
    tell("grid values", 0.55)
    model = GeologyModel(
        name=spec.name,
        units=units,
        grid=grid,
        topography=topography,
        stacks=stacks,
        network=network,
        cover=cover,
        weathering=weathering,
        hole_points=logged_hole_points(table, traces, step=HOLE_POINT_STEP, known=rock.notna()),
        max_extrapolation=spec.max_extrapolation,
    )
    model.pooled_blocks = list(pooled)
    model.sample_grid()
    lap("sample_grid_seconds")

    tell("checks", 0.85)
    record.extend(model.provenance_entries())
    record.extend(stack_entries)
    record.extend(order_entries)
    throw: list[dict[str, Any]] = []
    if network is not None:
        from bakumine.core.geology.faulted import throw_diagnostics

        throw, throw_entries = throw_diagnostics(model, table=table, units=rock, traces=traces)
        record.extend(throw_entries)
    diagnostics["throw"] = throw
    lap("throw_seconds")
    diagnostics["residuals"] = _residuals(model, table, rock, traces)
    volumes = diagnostics["volumes"] = _volumes(model)
    rock_units = [u for u in model.categories("lith") if u in in_events]
    if volumes["hull"]:
        whole = sum(volumes["units"][u]["volume"] for u in rock_units)
        outside = sum(volumes["units"][u]["extrapolated"] for u in rock_units)
        worst = max(rock_units, key=lambda u: volumes["units"][u]["extrapolated_share"] or 0.0)
        record.assumed(
            "Rock volume outside the drilled hull",
            source="extrapolation of the rock-unit fields beyond the convex hull of the "
            "logged hole points",
            used_for="volumes of the rock units in the whole model box",
            detail=f"{100.0 * outside / whole if whole else 0.0:.1f} % of the rock volume of "
            f"the box; largest share: {worst} "
            f"{100.0 * (volumes['units'][worst]['extrapolated_share'] or 0.0):.1f} %. Report "
            "volumes inside the hull (diagnostics.json, volumes) separately",
            stage="extent",
            task=_TASK,
            value=float(outside / whole) if whole else 0.0,
        )
    diagnostics["cross_validation"] = {
        "run": False,
        "note": "not run by the build; `bakumine geology cv DB --spec SPEC --out cv.csv`",
    }
    diagnostics["pooled_blocks"] = list(pooled)
    for name, value, used_for in spec.defaults:
        record.assumed(
            f"Default for {name}",
            source="default (not given in the build spec)",
            used_for=used_for,
            detail=value,
            task=_TASK,
        )
    lap("checks_seconds")
    timing["total_seconds"] = time.perf_counter() - t_start
    model.diagnostics.setdefault("timing", {}).update(timing)
    model.diagnostics.update(diagnostics)
    model.provenance = record
    model.build = BuildRecord(spec, built)  # type: ignore[attr-defined]
    tell("done", 1.0)
    return model


# --------------------------------------------------------------------------
# folder and summary
# --------------------------------------------------------------------------


def save_geology(model: GeologyModel, folder: str | Path, *, overwrite: bool = False) -> Path:
    """Write the model folder of a build: ``GeologyModel.save`` plus ``spec.json``, the
    surface side files and ``diagnostics.json`` as standard JSON."""
    from bakumine.core.geology.surfaces import surface_file_stem

    folder = model.save(folder, overwrite=overwrite)
    build: BuildRecord | None = getattr(model, "build", None)
    if build is not None:
        _write_json(folder / "spec.json", build.spec.to_dict())
        for surface in build.surfaces:
            stem = surface_file_stem(surface.name)
            _write_json(folder / "surfaces" / f"{stem}.json", surface.summary())
            surface.picks.to_csv(folder / "surfaces" / f"{stem}_picks.csv", index=False)
    _write_json(folder / "diagnostics.json", model.diagnostics)  # NaN -> null
    return folder


def build_summary(model: GeologyModel) -> dict[str, Any]:
    """The figures of a build, for the CLI summary, ``--summary`` and the API result."""
    d = model.diagnostics
    record = model.provenance
    kinds = {"measured": 0, "assumed": 0, "missing": 0}
    for entry in record.entries if record is not None else ():
        kinds[entry.kind] += 1
    faults = []
    if model.network is not None:
        for fault in model.network.faults:
            faults.append(
                {
                    "name": fault.name,
                    "source": fault.source,
                    "dip": float(fault.plane.dip),
                    "dip_azimuth": float(fault.plane.dip_azimuth),
                    "blocks": list(fault.blocks) if fault.blocks else None,
                }
            )
    build: BuildRecord | None = getattr(model, "build", None)
    surfaces = [
        {
            "name": s.name,
            "depth_mean": s.diagnostics.get("depth_mean"),
            "depth_min": s.diagnostics.get("depth_min"),
            "depth_max": s.diagnostics.get("depth_max"),
            "used": s.diagnostics.get("used"),
            "n_nested": int(s.n_nested),
        }
        for s in (build.surfaces if build is not None else [])
    ]
    return json_safe(
        {
            "name": model.name,
            "grid": d.get("grid"),
            "layers": {
                layer: list(model.categories(layer))
                for layer in ("lith", "weathering", "fault_block")
            },
            "blocks": list(model.blocks),
            "pooled_blocks": list(model.pooled_blocks),
            "faults": faults,
            "surfaces": surfaces,
            "throw": [
                {
                    k: r.get(k)
                    for k in (
                        "fault",
                        "upper",
                        "lower",
                        "throw",
                        "spread",
                        "kind",
                        "throw_dip_slip",
                        "heave_dip_slip",
                        "note",
                    )
                }
                for r in d.get("throw", [])
            ],
            "residuals": d.get("residuals"),
            "volumes": d.get("volumes"),
            "inputs": d.get("inputs"),
            "timing": d.get("timing"),
            "provenance": kinds,
        }
    )
