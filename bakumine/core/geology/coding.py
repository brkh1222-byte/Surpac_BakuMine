"""Coding a block model from a geology model (research.md 5.7.6, geology_contracts.md 11).

A geology model gives three independent categorical layers. Each is coded into its own
block attribute, in this order:

``fault_block``  the fault blocks (only when the model has more than one)
``lith``         rock units, the cover unit, ``AIR`` above the topography
``weathering``   the weathering zones, ``AIR`` (only when the model has weathering surfaces)

Then, from the unit dictionary and the two coded attributes:

``density``      default density -> density of the rock unit -> density of the weathering
                 zone where the dictionary gives one -> NaN in ``AIR``

and optionally ``geo_distance`` (metres to the nearest logged hole) and a display
attribute (``geo``: the rock unit, then ``ORE`` where the ore attribute is ``ORE``). The
existing ore attribute and every other attribute of the model are not written to;
sub-blocking splits blocks, and the children inherit all values.

How the blocks are coded -- precisely
-------------------------------------
:func:`geology_steps` writes the whole coding as the ``{"attributes": [...], "steps":
[...]}`` document of ``bakumine blockmodel code --steps``, and :func:`code_geology` runs
exactly those steps through :func:`bakumine.core.blockmodel.coding.code_model`. Replaying
the document therefore gives the same model, block for block.

For one layer the steps are: set the attribute to missing everywhere; then one step per
category with a :class:`~bakumine.core.blockmodel.constraints.GeologyConstraint` (the
label of the block centroid is that category). The first category step sub-blocks, with
``"boundaries": true``: a block is split when the labels at its 8 corners and its centre
are not all equal, so one pass refines along every boundary of the layer. Points without
a label (-1: beyond ``max_extrapolation``, outside the plan extent of the topography for
the weathering) stay missing.

**The sub-blocking is not conservative.** Nine sample points can miss a body that is
thinner than the block: a dyke thinner than a parent block is refined only where a sample
point happens to fall inside it, and its coded volume is then too small or too large by
an amount that depends on how the lattice meets it. The centroid rule is still exact for
the blocks that exist; the weathering and cover surfaces can instead be coded exactly
with a ``DtmConstraint`` on the DTMs in the geology folder.

Density rules as coded
----------------------
The weathering-zone density replaces the rock density in the blocks of that zone below
the topography, **except in the blocks of a cover unit** (kind ``cover``): soil is not
oxidised rock, so those keep the cover unit's own density (or the default density when
the unit has none).

The distance attribute is written by :func:`code_geology` directly; it has no equivalent
in the steps document and is absent after a replay.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.coding import AssignResult, code_model
from bakumine.core.blockmodel.constraints import DictResolver, constraint_from_dict
from bakumine.core.blockmodel.subblocking import _level_limit, merge_uniform
from bakumine.core.geology.provenance import Provenance, ProvenanceEntry
from bakumine.core.geology.units import UnitDictionary

__all__ = [
    "AIR",
    "LAYERS",
    "GeologyCodingResult",
    "code_geology",
    "coding_provenance_path",
    "density_provenance",
    "density_steps",
    "geology_steps",
    "unit_tonnage",
    "write_coding_provenance",
]

LAYERS = ("lith", "weathering", "fault_block")
DEFAULT_LAYERS = ("fault_block", "lith", "weathering")
AIR = "AIR"
#: unit kinds whose density becomes the density of blocks
ROCK_KINDS = ("lithology", "intrusion", "cover")
_TASK = "P9.T12"


@dataclass
class GeologyCodingResult:
    """What :func:`code_geology` did.

    Attributes
    ----------
    results : one :class:`AssignResult` per step of ``steps``.
    steps : the ``blockmodel code --steps`` document that was run.
    provenance : the density and coding entries (measured / assumed / missing).
    volumes : columns ``layer, category, volume`` (m3): every category of every coded
        layer, and a row with category ``None`` for blocks without a label when there are
        any. The rows of a layer sum to the volume of the model.
    attributes : coded layer -> attribute name; a layer that was skipped is absent.
    skipped : layer -> why it was not coded.
    n_merged : sibling sets merged afterwards (``merge=True``).
    """

    results: list[AssignResult]
    steps: dict
    provenance: list[ProvenanceEntry]
    volumes: pd.DataFrame
    attributes: dict[str, str] = field(default_factory=dict)
    skipped: dict[str, str] = field(default_factory=dict)
    n_merged: int = 0


# --------------------------------------------------------------------------
# the steps document
# --------------------------------------------------------------------------


def _quoted(label: str) -> str:
    """A category label as a string literal of the expression language."""
    return json.dumps(str(label), ensure_ascii=True)


def _equals(attribute: str, label: str) -> dict:
    return {"type": "attribute", "expression": f"{attribute} == {_quoted(label)}"}


def _attribute_names(attributes: Mapping[str, str] | None) -> dict[str, str]:
    names = {layer: layer for layer in LAYERS}
    for layer, name in (attributes or {}).items():
        if layer not in LAYERS:
            raise ValueError(f"attributes: unknown layer {layer!r}; layers are {LAYERS}")
        names[layer] = str(name)
    return names


def _active_layers(geology: Any, layers: Sequence[str]) -> tuple[list[str], dict[str, str]]:
    """The layers to code, in the order given, and the reason for each one left out."""
    if isinstance(layers, str):
        layers = [layers]
    active: list[str] = []
    skipped: dict[str, str] = {}
    for layer in layers:
        if layer not in LAYERS:
            raise ValueError(f"unknown layer {layer!r}; layers are {LAYERS}")
        if layer in active or layer in skipped:
            raise ValueError(f"layer {layer!r} is listed twice")
        categories = [c for c in geology.categories(layer) if c != AIR]
        if layer == "fault_block" and len(categories) < 2:
            skipped[layer] = "the geology model has no faults (one block)"
        elif layer == "weathering" and (
            not categories or len(getattr(geology, "weathering", (None,))) == 0
        ):
            skipped[layer] = "the geology model has no weathering surfaces"
        elif not categories:
            skipped[layer] = "the layer has no categories"
        else:
            active.append(layer)
    return active, skipped


def density_steps(
    units: UnitDictionary,
    *,
    lith: str = "lith",
    weathering: str | None = "weathering",
    density: str = "density",
) -> list[dict]:
    """The density steps from a unit dictionary and the coded attributes.

    Order (later wins): ``default_density`` everywhere (NaN when the dictionary has
    none) -> the density of each rock unit that has one, where ``lith`` is that unit ->
    the density of each weathering zone that has one, where ``weathering`` is that zone
    and ``lith`` is not a unit of kind ``cover`` (the cover keeps its own density) -> NaN
    where ``lith`` is ``AIR``. ``weathering=None`` leaves the zone steps out.
    """
    not_cover = "".join(f" and {lith} != {_quoted(code)}" for code in units.codes("cover"))
    default = units.default_density
    steps: list[dict] = [
        {"attribute": density, "value": "nan" if default is None else float(default),
         "subblock": False}
    ]  # fmt: skip
    for unit in units.units:
        if unit.kind in ROCK_KINDS and unit.density is not None:
            steps.append(
                {"attribute": density, "value": float(unit.density),
                 "constraint": _equals(lith, unit.code), "subblock": False}
            )  # fmt: skip
    if weathering is not None:
        for zone in units.weathering_zones:
            if zone.density is not None:
                steps.append(
                    {"attribute": density, "value": float(zone.density),
                     "constraint": {"type": "attribute", "expression":
                                    f"{weathering} == {_quoted(zone.code)}{not_cover}"},
                     "subblock": False}
                )  # fmt: skip
    steps.append(
        {"attribute": density, "value": "nan", "constraint": _equals(lith, AIR),
         "subblock": False}
    )  # fmt: skip
    return steps


def geology_steps(
    geology_name: str,
    geology: Any,
    *,
    layers: Sequence[str] = DEFAULT_LAYERS,
    attributes: Mapping[str, str] | None = None,
    subblock: bool = True,
    max_level: int | None = None,
    density: str | None = "density",
    distance: str | None = None,
    display: str | None = None,
    ore_attribute: str | None = None,
    ore_value: str = "ORE",
    merge: bool = False,
) -> dict:
    """The coding of :func:`code_geology` as a ``blockmodel code --steps`` document.

    ``geology_name`` is the name the constraints carry; on replay the resolver turns it
    into the model (``--geometry-dir DIR`` reads ``DIR/<name>.geology/``). ``distance``
    and ``merge`` are accepted so that the same options can be passed to both functions:
    the distance attribute has no step, and merging is the ``--merge`` flag of the replay.
    """
    del distance, merge  # no step of their own; see the docstring
    names = _attribute_names(attributes)
    active, _ = _active_layers(geology, layers)
    out: dict[str, list] = {"attributes": [], "steps": []}
    for layer in active:
        name = names[layer]
        categories = [str(c) for c in geology.categories(layer)]
        out["attributes"].append(
            {"name": name, "kind": "category", "dtype": "int16", "categories": categories}
        )
        out["steps"].append({"attribute": name, "value": None, "subblock": False})
        for n, category in enumerate(categories):
            first = subblock and n == 0
            constraint: dict[str, Any] = {
                "type": "geology", "model": geology_name, "layer": layer, "in": [category],
                "side": "inside",
            }  # fmt: skip
            if first:
                constraint["boundaries"] = True
            step: dict[str, Any] = {
                "attribute": name, "value": category, "constraint": constraint, "subblock": first,
            }  # fmt: skip
            if first and max_level is not None:
                step["max_level"] = int(max_level)
            out["steps"].append(step)
    lith = names["lith"]
    if density is not None and "lith" in active:
        units = getattr(geology, "units", None)
        if not isinstance(units, UnitDictionary):
            raise ValueError("density needs the unit dictionary of the geology model (.units)")
        out["attributes"].append({"name": density, "kind": "float", "unit": "t/m3"})
        zones = names["weathering"] if "weathering" in active else None
        out["steps"].extend(density_steps(units, lith=lith, weathering=zones, density=density))
    if display is not None:
        if "lith" not in active:
            raise ValueError("the display attribute needs the layer 'lith' to be coded")
        categories = [str(c) for c in geology.categories("lith")]
        listed = categories + ([ore_value] if ore_attribute and ore_value not in categories else [])
        out["attributes"].append(
            {"name": display, "kind": "category", "dtype": "int16", "categories": listed}
        )
        out["steps"].append({"attribute": display, "value": None, "subblock": False})
        for category in categories:
            out["steps"].append(
                {"attribute": display, "value": category,
                 "constraint": _equals(lith, category), "subblock": False}
            )  # fmt: skip
        if ore_attribute:
            out["steps"].append(
                {"attribute": display, "value": ore_value,
                 "constraint": _equals(ore_attribute, ore_value), "subblock": False}
            )  # fmt: skip
    return out


# --------------------------------------------------------------------------
# coding
# --------------------------------------------------------------------------


class _CachedLabels:
    """A geology model that remembers the labels of the last point set of each layer.

    The per-category steps of one layer all ask for the labels of the same block
    centroids; the points are recognised by a digest of their bytes.
    """

    def __init__(self, geology: Any):
        self._geology = geology
        self._last: dict[str, tuple[bytes, np.ndarray]] = {}

    def categories(self, layer: str):
        return self._geology.categories(layer)

    def labels(self, points, layer: str = "lith", *, exact: bool = False) -> np.ndarray:
        pts = np.ascontiguousarray(points, dtype=np.float64)
        key = hashlib.blake2b(pts.tobytes(), digest_size=16).digest() + bytes([bool(exact)])
        held = self._last.get(layer)
        if held is not None and held[0] == key and len(held[1]) == len(pts):
            return held[1]
        labels = np.asarray(self._geology.labels(pts, layer, exact=exact))
        self._last[layer] = (key, labels)
        return labels


def _check_existing(model: BlockModel, spec: dict, ore_attribute: str | None) -> None:
    for item in spec["attributes"]:
        name = item["name"]
        if name in model.attributes and model.attribute(name).kind != item["kind"]:
            raise ValueError(
                f"the model already has an attribute {name!r} of kind "
                f"{model.attribute(name).kind}; geology coding needs kind {item['kind']}"
            )
    if ore_attribute is not None:
        if ore_attribute not in model.attributes:
            raise ValueError(f"the model has no ore attribute {ore_attribute!r}")
        if model.attribute(ore_attribute).kind != "category":
            raise ValueError(f"the ore attribute {ore_attribute!r} is not a category")
        if any(item["name"] == ore_attribute for item in spec["attributes"]):
            raise ValueError(
                f"the ore attribute {ore_attribute!r} would be overwritten by the geology coding"
            )
    names = [item["name"] for item in spec["attributes"]]
    if len(set(names)) != len(names):
        raise ValueError(f"two geology attributes share a name: {', '.join(names)}")


def _layer_volumes(model: BlockModel, coded: Mapping[str, str]) -> pd.DataFrame:
    volume = model.volumes()
    rows = []
    for layer, name in coded.items():
        spec = model.attribute(name)
        codes = np.asarray(model[name]).astype(np.int64)
        sums = np.bincount(codes + 1, weights=volume, minlength=len(spec.categories) + 1)
        rows.extend(
            (layer, category, float(sums[i + 1])) for i, category in enumerate(spec.categories)
        )
        if sums[0] > 0.0:
            rows.append((layer, None, float(sums[0])))
    return pd.DataFrame(
        {
            "layer": pd.Series([r[0] for r in rows], dtype=object),
            "category": pd.Series([r[1] for r in rows], dtype=object),  # None stays None
            "volume": pd.Series([r[2] for r in rows], dtype=np.float64),
        }
    )


def code_geology(
    model: BlockModel,
    geology: Any,
    *,
    geology_name: str,
    layers: Sequence[str] = DEFAULT_LAYERS,
    attributes: Mapping[str, str] | None = None,
    subblock: bool = True,
    max_level: int | None = None,
    density: str | None = "density",
    distance: str | None = None,
    display: str | None = None,
    ore_attribute: str | None = None,
    ore_value: str = "ORE",
    merge: bool = False,
) -> GeologyCodingResult:
    """Code the layers of a geology model, the density and the optional attributes.

    Parameters
    ----------
    model : the block model; modified in place.
    geology : a :class:`~bakumine.core.geology.model.GeologyModel`, or any object with
        ``labels(points, layer, exact=False)`` and ``categories(layer)`` (plus ``units``
        for the density and ``distance_to_data(points)`` for the distance).
    geology_name : the name written into the constraints of the steps document.
    layers : which layers, in coding order. ``fault_block`` is left out when the model
        has a single block and ``weathering`` when it has no weathering surfaces (see
        ``result.skipped``).
    attributes : layer -> attribute name, where it differs from the layer name.
    subblock, max_level : split the blocks cut by the boundaries of each layer, down to
        the minimum block size or ``max_level``. Not conservative: see the module text.
    density : name of the density attribute (float, t/m3), or ``None`` for none. Coded
        only together with the layer ``lith``.
    distance : name of a float attribute for the distance to the nearest logged hole.
    display, ore_attribute, ore_value : a category attribute that repeats the rock unit
        and, when ``ore_attribute`` is given, holds ``ore_value`` where that attribute
        equals ``ore_value``. The ore attribute itself is only read.
    merge : afterwards merge sibling sub-blocks that are identical in **all** attributes
        (nothing merges across a distance attribute, which differs from block to block).

    An attribute that already exists is coded again from scratch (it must be of the right
    kind). Everything is validated before the model is changed.
    """
    options = dict(
        layers=layers, attributes=attributes, subblock=subblock, max_level=max_level,
        density=density, display=display, ore_attribute=ore_attribute, ore_value=ore_value,
    )  # fmt: skip
    spec = geology_steps(geology_name, geology, **options)
    active, skipped = _active_layers(geology, layers)
    names = _attribute_names(attributes)
    coded = {layer: names[layer] for layer in active}
    if ore_attribute is not None and display is None:
        raise ValueError("ore_attribute is only used by the display attribute: give display too")
    _check_existing(model, spec, ore_attribute if display is not None else None)
    _level_limit(model, max_level)
    if distance is not None:
        if not callable(getattr(geology, "distance_to_data", None)):
            raise ValueError("the geology model cannot give a distance to the logged holes")
        if distance in model.attributes and model.attribute(distance).kind != "float":
            raise ValueError(f"the model already has a non-float attribute {distance!r}")
        if any(item["name"] == distance for item in spec["attributes"]):
            raise ValueError(f"the distance attribute {distance!r} is also a geology attribute")

    for item in spec["attributes"]:
        item = dict(item)
        if item["name"] not in model.attributes:
            model.add_attribute(item.pop("name"), item.pop("kind"), **item)
    steps = []
    for item in spec["steps"]:
        item = dict(item)
        if item.get("constraint") is not None:
            item["constraint"] = constraint_from_dict(item["constraint"])
        steps.append(item)
    resolver = DictResolver(geologies={geology_name: _CachedLabels(geology)})
    results = code_model(model, steps, resolver)
    if distance is not None:
        if distance not in model.attributes:
            model.add_attribute(distance, "float", unit="m")
        model.set(distance, geology.distance_to_data(model.centroids()))
    n_merged = merge_uniform(model) if merge else 0

    provenance: list[ProvenanceEntry] = []
    if subblock and coded:
        smallest = " x ".join(f"{s:g}" for s in model.definition.min_block_size)
        provenance.append(
            ProvenanceEntry(
                "Sub-blocking at the geology boundaries",
                "assumed",
                "labels at the 8 corners and the centre of each block (not conservative)",
                "block boundaries of " + ", ".join(coded.values()),
                detail=f"smallest block {smallest} m; a body thinner than a block can pass "
                "between the sample points and is then coded by block centroids only",
                stage="coding",
                task=_TASK,
            )
        )
    if density is not None and "lith" in coded:
        provenance.extend(
            density_provenance(
                model,
                geology.units,
                lith=coded["lith"],
                weathering=coded.get("weathering"),
                density=density,
            )  # fmt: skip
        )
    return GeologyCodingResult(
        results=results,
        steps=spec,
        provenance=provenance,
        volumes=_layer_volumes(model, coded),
        attributes=coded,
        skipped=skipped,
        n_merged=n_merged,
    )


# --------------------------------------------------------------------------
# tonnage and provenance
# --------------------------------------------------------------------------


def unit_tonnage(
    model: BlockModel, *, lith: str = "lith", density: str = "density"
) -> pd.DataFrame:
    """Volume and tonnes per rock unit of a coded model.

    Columns ``unit, blocks, volume`` (m3), ``tonnes`` and ``density`` (tonnes / volume
    of the blocks that have a density; NaN when none has). Blocks without a density
    (``AIR``) count in the volume and give no tonnes. One row per category of ``lith``,
    in its order, and a row with unit ``None`` for blocks without a rock unit, if any.
    """
    spec = model.attribute(lith)
    codes = np.asarray(model[lith]).astype(np.int64) + 1
    volume = model.volumes()
    n = len(spec.categories) + 1
    has = np.zeros(len(volume), dtype=bool)
    tonnes_of = np.zeros(len(volume))
    if density in model.attributes:
        rho = np.asarray(model[density], dtype=np.float64)
        has = np.isfinite(rho)
        tonnes_of = np.where(has, rho, 0.0) * volume
    blocks = np.bincount(codes, minlength=n)
    volumes = np.bincount(codes, weights=volume, minlength=n)
    tonnes = np.bincount(codes, weights=tonnes_of, minlength=n)
    with_density = np.bincount(codes, weights=np.where(has, volume, 0.0), minlength=n)
    rows = []
    for i, unit in [*((i + 1, c) for i, c in enumerate(spec.categories)), (0, None)]:
        if unit is None and blocks[0] == 0:
            continue
        mean = tonnes[i] / with_density[i] if with_density[i] > 0.0 else math.nan
        rows.append((unit, int(blocks[i]), float(volumes[i]), float(tonnes[i]), float(mean)))
    return pd.DataFrame(rows, columns=["unit", "blocks", "volume", "tonnes", "density"])


def density_provenance(
    model: BlockModel,
    units: UnitDictionary,
    *,
    lith: str = "lith",
    weathering: str | None = "weathering",
    density: str = "density",
) -> list[ProvenanceEntry]:
    """Where the density of the coded blocks comes from (contract section 3, row T12).

    One entry per rock unit and weathering zone with a density (``measured`` or
    ``assumed``, as the dictionary says), one for ``default_density`` when a block takes
    it, and a ``missing`` entry per rock unit of the model without a density. Every
    entry states the block volume and the tonnes that take the value, as shares of the
    volume and tonnes of all blocks with a density. The rule a block falls under is
    worked out here from the coded ``lith`` and ``weathering``, in the order of
    :func:`density_steps`.
    """
    volume = model.volumes()
    lith_labels = model.get(lith, decode=True)
    zone_labels = None
    if weathering is not None and weathering in model.attributes:
        zone_labels = model.get(weathering, decode=True)
    rho = np.asarray(model[density], dtype=np.float64)
    tonnes = np.where(np.isfinite(rho), rho, 0.0) * volume
    ground = lith_labels != AIR
    total_volume = float(volume[np.isfinite(rho)].sum())
    total_tonnes = float(tonnes.sum())

    rule = np.full(len(volume), "none", dtype=object)
    if units.default_density is not None:
        rule[ground] = "default"
    unit_density = {
        u.code: u for u in units.units if u.kind in ROCK_KINDS and u.density is not None
    }
    for code in unit_density:
        rule[lith_labels == code] = f"unit:{code}"
    zone_density = {z.code: z for z in units.weathering_zones if z.density is not None}
    if zone_labels is not None:
        rock = ground & ~np.isin(lith_labels, list(units.codes("cover")))  # cover keeps its own
        for code in zone_density:
            rule[(zone_labels == code) & rock] = f"zone:{code}"

    def share(part: float, whole: float) -> str:
        return f"{100.0 * part / whole:.1f} %" if whole > 0.0 else "0.0 %"

    def applies(mask: np.ndarray) -> tuple[str, float]:
        v, t = float(volume[mask].sum()), float(tonnes[mask].sum())
        text = (
            f"{v:.0f} m3 ({share(v, total_volume)} of the block volume with a density), "
            f"{t:.0f} t ({share(t, total_tonnes)} of the tonnes)"
        )
        return text, v

    tag = {"stage": "density", "task": _TASK}
    entries: list[ProvenanceEntry] = []
    present = [c for c in model.attribute(lith).categories if c != AIR]
    for code in present:
        if code not in unit_density:
            continue
        unit = unit_density[code]
        text, _ = applies(rule == f"unit:{code}")
        overridden = float(volume[(lith_labels == code) & (rule != f"unit:{code}")].sum())
        if overridden > 0.0:
            text += f"; {overridden:.0f} m3 of the unit take a weathering-zone density instead"
        entries.append(
            ProvenanceEntry(
                f"Density of unit {code}",
                unit.density_source or "assumed",
                f"unit dictionary: {unit.density:g} t/m3 ({unit.density_source or 'assumed'})",
                "block density and tonnes",
                detail=text,
                value=float(unit.density),
                **tag,
            )
        )
    if zone_labels is not None:
        for code, zone in zone_density.items():
            text, _ = applies(rule == f"zone:{code}")
            entries.append(
                ProvenanceEntry(
                    f"Density of weathering zone {code}",
                    zone.density_source or "assumed",
                    f"unit dictionary: {zone.density:g} t/m3 ({zone.density_source or 'assumed'}),"
                    " replaces the rock density in the zone (not in the cover)",
                    "block density and tonnes",
                    detail=text,
                    value=float(zone.density),
                    **tag,
                )
            )
    without = [c for c in present if c not in unit_density]
    by_default = rule == "default"
    if units.default_density is not None and (by_default.any() or without):
        text, _ = applies(by_default)
        users = list(without)
        if (by_default & (lith_labels == None)).any():  # noqa: E711  (object array)
            users.append("blocks without a rock unit")
        entries.append(
            ProvenanceEntry(
                "Default density",
                units.default_density_source,
                f"unit dictionary: default_density {units.default_density:g} t/m3 "
                f"({units.default_density_source})",
                "block density and tonnes of: " + (", ".join(users) or "no block"),
                detail=text,
                value=float(units.default_density),
                **tag,
            )
        )
    for code in without:
        own = lith_labels == code
        v = float(volume[own].sum())
        if units.default_density is not None:
            used = f"would replace the default density {units.default_density:g} t/m3"
        else:
            used = "its blocks have no density and no tonnes"
        entries.append(
            ProvenanceEntry(
                f"Density samples for unit {code}",
                "missing",
                "density measurements in this unit (a density table or laboratory results)",
                used,
                detail=f"{v:.0f} m3 of blocks ({share(v, float(volume[ground].sum()))} of the "
                "volume below the topography)",
                **tag,
            )
        )
    return entries


_PLAIN = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.\-]*\Z")


def coding_provenance_path(geology_folder: str | Path, model_name: str) -> Path:
    """``<geology folder>/coding/<model name>.provenance.json``."""
    if not isinstance(model_name, str) or not _PLAIN.match(model_name) or ".." in model_name:
        raise ValueError(
            f"invalid model name {model_name!r}: use letters, digits, '_', '-' and '.' only"
        )
    return Path(geology_folder) / "coding" / f"{model_name}.provenance.json"


def write_coding_provenance(
    geology_folder: str | Path, model_name: str, entries: Sequence[ProvenanceEntry]
) -> Path:
    """Write the record of one coding run into the geology folder; returns the path.

    The record is kept with the geology model, not in the block model folder:
    ``storage.save_model`` replaces that folder on every save.
    """
    folder = Path(geology_folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"{folder} is not a folder")
    path = coding_provenance_path(folder, model_name)
    record = Provenance(object_kind="geology coding", name=model_name)
    record.extend(entries)
    record.save(path)
    return path
