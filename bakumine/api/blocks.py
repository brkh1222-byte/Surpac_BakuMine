"""Block models for the viewer: the demo and synthetic models, selection, level of detail
and the "blocks" payload.

Wire format (``kind = "blocks"``), N blocks sent out of T in the model
--------------------------------------------------------------------
Nothing geometric is sent. The client rebuilds centroids and sizes from the model
definition and the structure columns exactly as :class:`BlockModel` does
(``web/src/blocks/structure.ts``; fixtures in ``tests/api/block_fixtures.py``)::

    row            u4 (N)      row of each block in the full table, increasing.
                               Omitted when every row is sent (meta.rows = "all").
    parent         u4 (N)      flat parent index (i * ny + j) * nz + k
    level          u1 (N)      number of splits below the parent
    sub            u1 (N, 3)   minimum corner inside the parent, in finest cells
                               parent / level / sub are omitted for a model without
                               sub-blocks (meta.structure = "dense"): there parent = row,
                               level = 0, sub = 0.
    value:<name>   f4 (N)      numeric attribute (float or int); NaN = missing
    code:<name>    i2 (N)      category attribute, index into its categories; -1 = missing
                               (i4 when there are more than 32 767 categories)

    meta = {object_id, definition, structure: "dense" | "octree", rows: "all" | "listed",
            n_total, n_selected, n_visible, n_blocks, selected_volume,
            lod: "full" | "shell", lod_requested, clipped, where, present,
            bytes_per_block,
            attributes: [{name, type: "numeric", array, unit} |
                         {name, type: "categorical", array, categories}]}

Bytes per block: 0 (dense, everything sent), 4 (dense, a selection) or 12 (sub-blocked:
4 + 4 + 1 + 3) of structure, plus 4 per numeric and 2 per category attribute requested.
For comparison, float64 centroids and sizes alone are 48 bytes per block.

Selection
---------
``where`` is a constraint: either the JSON of
:func:`bakumine.core.blockmodel.constraints.constraint_from_dict` (text starting with
``{``) or a short attribute expression (``au > 1 and rock == "ORE"``), which becomes an
:class:`AttributeConstraint`. It is evaluated by the constraint engine with the object's
geometry resolver (solids and surfaces by id); nothing is ever passed to ``eval``. The
rows returned are exactly ``np.flatnonzero(constraint.mask(model, resolver))``.

Blocks without a value
----------------------
The viewer normally hides the blocks that have no value of the colour attribute. Then
what can be seen is "selected AND has a value", and that is the set everything below
works on: ``present = <attribute>`` intersects the selection with the blocks where the
attribute is not missing before the level of detail or the clip is applied, and only
those blocks are sent. Without it a model whose grade exists in the ore alone would be
reduced to the shell of the whole box - air and waste, all without a value, all hidden -
and nothing would be drawn. ``n_selected`` and ``selected_volume`` still describe
``where`` alone; ``n_visible`` counts the selected blocks that have a value (equal to
``n_selected`` without ``present``).

Level of detail
---------------
``lod = "shell"`` sends only the *exposed* blocks of the selection: those that could
show a face. It is computed on the parent grid, conservatively. A parent cell is *full*
when every one of its leaf blocks is selected; a block is dropped only when its parent
cell and the six face-neighbouring parent cells are all full (cells outside the model
count as not full). So every dropped block has all six neighbours selected, whatever the
sub-blocking; blocks in a partly selected parent, and the layer of whole parents next to
them, are kept. A solid body of millions of blocks shrinks to its skin: O(n^(2/3)).

``lod = "auto"`` (default) applies the shell when the visible selection has more than
:data:`AUTO_SHELL_BLOCKS` blocks; ``lod = "full"`` never does.

What the shell does to slicing: the interior is not there, so a section through a shell
would look hollow. The viewer therefore passes the section slab as ``clip`` (a second
constraint, normally two planes): with ``clip`` the server sends every selected block
inside it - no shell - while ``n_selected`` and ``selected_volume`` still describe the
``where`` selection. The same goes for a shrink factor or an opacity below one, which
make the interior visible: the viewer then asks for ``lod = "full"``.

Limits of the shell: it is decided per parent cell, so in a sub-blocked model it keeps
more than the true skin; every exposed block is sent whole (all its faces); and it is a
property of the selection, not of the view - nothing is dropped for being far away or
small on screen, so a selection that is all skin (scattered blocks) is sent in full.

Besides the payload: ``block_object_entry`` is the manifest entry (definition, attribute
summaries, ``display`` hints for the first view, the solids and surfaces a constraint may
name) and ``block_details`` is one block with every attribute at full precision.
"""

from __future__ import annotations

import json
import re
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.api.binary import pack_arrays
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    AttributeConstraint,
    Constraint,
    DictResolver,
    DtmConstraint,
    SolidConstraint,
    constraint_from_dict,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.mesh import Mesh

#: ``lod=auto`` sends the exposed shell when more blocks than this are selected.
AUTO_SHELL_BLOCKS = 250_000
LODS = ("auto", "full", "shell")
#: Largest synthetic model served for the performance probe.
MAX_SYNTHETIC_BLOCKS = 8_000_000
_SYNTHETIC_ID = re.compile(r"^perf-(\d{1,8})$")
_PERCENTILES = (2.0, 25.0, 50.0, 75.0, 98.0)


class BlockQueryError(ValueError):
    """A request parameter is invalid; the message is shown to the user (HTTP 400)."""


@dataclass
class BlockData:
    """A block model with what the viewer needs around it."""

    model: BlockModel
    #: looks up the solids and surfaces that constraints name (by scene object id)
    resolver: DictResolver | None = None
    #: attribute -> unit label, where the attribute itself has none
    units: dict[str, str] = field(default_factory=dict)
    #: scene object id -> name of the closed solids / surfaces a constraint may name
    solids: dict[str, str] = field(default_factory=dict)
    surfaces: dict[str, str] = field(default_factory=dict)
    #: how the viewer should open the model: ``color_by`` (attribute name) and
    #: ``category_filter`` (``{"attribute", "values"}``: show only these categories)
    display: dict[str, Any] = field(default_factory=dict)
    #: category -> colour ``#rrggbb`` from the project's unit dictionaries; the manifest
    #: entry of a category attribute carries the ones of its own categories (``colours``)
    colours: dict[str, str] = field(default_factory=dict)


# ---------------------------------------------------------------------- demo model


def build_demo_block_model(
    folder: str | Path,
    solid: Mesh | None = None,
    topo: Mesh | None = None,
    levels: int = 2,
) -> BlockModel:
    """The synthetic deposit's truth grid as a block model, sub-blocked on the ore contact.

    Parents carry the truth table (``rock`` as a category; ``au``, ``au_ore``,
    ``ore_fraction``, ``density`` as float, NaN where the truth has none). With ``solid``
    the blocks cut by the ore solid are split ``levels`` times (each pass splits the
    blocks whose box the surface crosses; ``SolidConstraint.classify_boxes``) and the
    blocks created are coded by their own centroid: inside the solid they are ORE with
    the parent's ore grade, outside they are WASTE at the background grade, or AIR above
    ``topo``. This is display data for the viewer, not the sub-blocking workflow.
    """
    folder = Path(folder)
    truth = json.loads((folder / "truth.json").read_text())
    grid = truth["blocks"]
    definition = BlockModelDefinition(
        origin=tuple(grid["origin"]),
        block_size=tuple(grid["size"]),
        shape=tuple(grid["shape"]),
        max_depth=levels if solid is not None else 0,
    )
    table = pd.read_parquet(folder / "truth_blocks.parquet")
    if len(table) != definition.n_parents:
        raise ValueError("truth_blocks.parquet does not match the grid in truth.json")
    # The truth table runs i fastest; the model runs k fastest.
    order = np.empty(len(table), dtype=np.int64)
    ijk = table[["i", "j", "k"]].to_numpy(dtype=np.int64)
    order[definition.flat_index(ijk)] = np.arange(len(table))

    grade = truth.get("grade", {})
    unit = str(grade.get("units", ""))
    model = BlockModel(definition)
    rocks = ["AIR", "WASTE", "ORE"]
    model.add_attribute("rock", "category", categories=rocks, dtype=np.int16,
                        description="rock type of the larger part of the block")  # fmt: skip
    model.set("rock", table["rock"].to_numpy(dtype=object)[order])
    for name, description, attr_unit in (
        ("au", "block grade", unit),
        ("au_ore", "grade of the ore part", unit),
        ("ore_fraction", "volume fraction of ore", ""),
        ("density", "bulk density", "t/m3"),
    ):
        model.add_attribute(name, "float", dtype=np.float32, description=description,
                            unit=attr_unit)  # fmt: skip
        model.set(name, table[name].to_numpy(dtype=np.float64)[order])
    if solid is None:
        return model

    contact = SolidConstraint(solid)
    created = np.zeros(model.n_blocks, dtype=bool)
    for _ in range(levels):
        status = contact.classify_boxes(definition, *model.bounds_local())
        result = model.split(status == 0, strict=False)
        if result.n_split == 0:
            break
        created = created[result.source] | result.is_child
    if not created.any():
        return model
    inside = contact.mask(model)
    air = np.zeros(model.n_blocks, dtype=bool)
    if topo is not None:
        air = DtmConstraint(DTM(topo), "above").mask(model)
    ore = created & inside
    waste = created & ~inside & ~air
    sky = created & ~inside & air
    background = float(grade.get("waste_mean", np.nan))
    densities = truth.get("density_t_m3", {})
    ore_grade = np.where(np.isnan(model["au_ore"]), model["au"], model["au_ore"])
    model.set("rock", "ORE", ore)
    model.set("au", ore_grade[ore], ore)
    model.set("au_ore", ore_grade[ore], ore)
    model.set("ore_fraction", 1.0, ore)
    model.set("rock", "WASTE", waste)
    model.set("au", background, waste)
    model.set("rock", "AIR", sky)
    model.set("au", np.nan, sky)
    model.set("density", np.nan, sky)
    for selection in (waste, sky):
        model.set("au_ore", np.nan, selection)
        model.set("ore_fraction", 0.0, selection)
    if isinstance(densities, dict):
        if "ORE" in densities:
            model.set("density", float(densities["ORE"]), ore)
        if "WASTE" in densities:
            model.set("density", float(densities["WASTE"]), waste)
    return model


# ---------------------------------------------------------------------- synthetic model


def synthetic_block_model(n_blocks: int) -> BlockModel:
    """A dense model of about ``n_blocks`` blocks for the performance probe.

    ``nx = ny``, 50 levels (2 000 000 -> 200 x 200 x 50 exactly), 10 x 10 x 5 m blocks
    at mine coordinates. An ellipsoidal ore body with a smooth grade inside, waste
    around it and air above an undulating surface. Deterministic.
    """
    if not 1 <= n_blocks <= MAX_SYNTHETIC_BLOCKS:
        raise BlockQueryError(f"synthetic models have 1 to {MAX_SYNTHETIC_BLOCKS} blocks")
    nz = 50 if n_blocks >= 50_000 else max(1, round(n_blocks ** (1 / 3)))
    side = max(1, round(float(np.sqrt(n_blocks / nz))))
    definition = BlockModelDefinition(
        origin=(500_000.0, 5_300_000.0, 800.0), block_size=(10.0, 10.0, 5.0), shape=(side, side, nz)
    )
    model = BlockModel(definition)
    i, j, k = np.meshgrid(
        (np.arange(side, dtype=np.float32) + 0.5) / side,
        (np.arange(side, dtype=np.float32) + 0.5) / side,
        (np.arange(nz, dtype=np.float32) + 0.5) / nz,
        indexing="ij",
    )
    radius = np.sqrt(((i - 0.5) / 0.4) ** 2 + ((j - 0.5) / 0.33) ** 2 + ((k - 0.45) / 0.36) ** 2)
    surface = 0.88 + 0.05 * np.sin(i * 9.0) * np.cos(j * 7.0)
    air = k > surface
    ore = (radius < 1.0) & ~air
    ripple = 0.5 + 0.5 * np.sin(i * 40.0) * np.sin(j * 34.0 + 1.0) * np.cos(k * 25.0)
    au = np.where(ore, 0.3 + 4.0 * (1.0 - radius) ** 2 + 1.2 * ripple, 0.02 + 0.03 * ripple)
    au = np.where(air, np.nan, au).astype(np.float32)
    rock = np.where(air, 0, np.where(ore, 2, 1)).astype(np.int16)
    model.add_attribute("rock", "category", categories=["AIR", "WASTE", "ORE"], dtype=np.int16)
    model["rock"][:] = rock.reshape(-1)
    model.add_attribute("au", "float", dtype=np.float32, unit="g/t")
    model["au"][:] = au.reshape(-1)
    model.add_attribute("density", "float", dtype=np.float32, unit="t/m3")
    model["density"][:] = np.where(air, np.nan, np.where(ore, 2.85, 2.7)).reshape(-1)
    return model


_synthetic_lock = threading.Lock()
_synthetic: dict[int, BlockData] = {}


def synthetic_block_data(object_id: str) -> BlockData | None:
    """Block data of a ``perf-<n>`` object id (kept: one model at a time); else ``None``."""
    match = _SYNTHETIC_ID.match(object_id)
    if not match:
        return None
    n = int(match.group(1))
    with _synthetic_lock:
        if n not in _synthetic:
            data = BlockData(synthetic_block_model(n), display={"color_by": "au"})
            _synthetic.clear()
            _synthetic[n] = data
        return _synthetic[n]


# ---------------------------------------------------------------------- manifest


def _numeric_summary(values: np.ndarray) -> dict[str, Any]:
    finite = values[np.isfinite(values)].astype(np.float64)
    if finite.size == 0:
        return {"min": None, "max": None, "display_min": None, "display_max": None,
                "percentiles": None}  # fmt: skip
    p = np.percentile(finite, _PERCENTILES)
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        # A robust range for the default legend: grades are long-tailed.
        "display_min": float(p[0]),
        "display_max": float(p[-1]),
        "percentiles": {f"p{int(q):02d}": float(v) for q, v in zip(_PERCENTILES, p, strict=True)},
    }


def block_attribute_summaries(data: BlockData) -> list[dict[str, Any]]:
    """One manifest entry per attribute: type, unit, range and percentiles or categories."""
    model = data.model
    out: list[dict[str, Any]] = []
    for name, spec in model.attributes.items():
        missing = int(model.is_missing(name).sum())
        entry: dict[str, Any] = {
            "table": "blocks",
            "name": name,
            "count": model.n_blocks - missing,
            "missing": missing,
            "description": spec.description,
        }
        if spec.kind == "category":
            entry["type"] = "categorical"
            entry["categories"] = [str(c) for c in spec.categories]
            colours = {c: data.colours[c] for c in entry["categories"] if c in data.colours}
            if colours:
                entry["colours"] = colours
        else:
            entry["type"] = "numeric"
            entry["storage"] = spec.kind
            entry["unit"] = spec.unit or data.units.get(name) or None
            entry.update(_numeric_summary(model[name]))
        out.append(entry)
    return out


def definition_dict(definition: BlockModelDefinition) -> dict[str, Any]:
    """The definition as the viewer needs it: the rotation always as an explicit matrix."""
    return {
        "origin": list(definition.origin),
        "block_size": list(definition.block_size),
        "shape": list(definition.shape),
        "max_depth": list(definition.max_depth),
        # rows of R; its COLUMNS are the world directions of the model i, j, k axes
        "rotation": [[float(v) for v in row] for row in definition.rotation],
        "rotated": bool(definition.is_rotated),
    }


def block_object_entry(data: BlockData) -> dict[str, Any]:
    """The block-specific part of a manifest entry (``bounds``, ``definition``, ...)."""
    model = data.model
    box = model.definition.bounding_box()
    return {
        "bounds": {"min": box[0].tolist(), "max": box[1].tolist()},
        "definition": definition_dict(model.definition),
        "attributes": block_attribute_summaries(data),
        "stats": {
            "blocks": int(model.n_blocks),
            "parents": int(model.definition.n_parents),
            "dense": bool(model.is_dense),
            "max_level": int(model.level.max()) if model.n_blocks else 0,
            "volume": float(model.definition.volume),
        },
        "display": dict(data.display),
        "constraint_geometry": {
            "solids": [{"id": k, "name": v} for k, v in data.solids.items()],
            "surfaces": [{"id": k, "name": v} for k, v in data.surfaces.items()],
        },
    }


# ---------------------------------------------------------------------- first view

#: Suffixes of the attributes an estimation run writes next to its estimate
#: (``bakumine.core.estimation.block_estimation.OutputNames`` defaults).
DIAGNOSTIC_SUFFIXES = ("_pass", "_ns", "_dist", "_ndist", "_kv", "_slope", "_ke")
#: Float attributes that are not a grade: never the default colour while another exists.
_NOT_A_GRADE = {"density", "dens", "sg", "bd", "bulk_density", "specific_gravity", "tonnage",
                "tonnes", "volume"}  # fmt: skip
#: Category values that mean "nothing here": left out of the first view.
_AIR_LIKE = {"air", "sky", "atmosphere", "void", "above topo", "above_topo", "above topography"}


def default_display(model: BlockModel) -> dict[str, Any]:
    """How a model about which nothing else is known should open (``BlockData.display``).

    ``color_by``: an estimated grade - a float attribute with diagnostics next to it
    (``au`` with ``au_pass``, ...; of several, the first). Failing that, the float
    attribute with the most missing values that is neither a density nor a diagnostic and
    has a value somewhere (a grade exists in the ore only; a ready-made attribute is
    filled everywhere). Failing that a category attribute, then anything.

    ``category_filter``: when a category attribute has an air-like value ("AIR"), every
    other category of it. Nothing else is filtered: a grade that most blocks lack needs
    no filter, because blocks without a value of the colour attribute are not drawn
    (``present`` in the module docstring).
    """
    names = model.attribute_names
    kinds = {name: model.attribute(name).kind for name in names}

    def diagnostic(name: str) -> bool:
        return any(
            name.endswith(suffix) and kinds.get(name[: -len(suffix)]) == "float"
            for suffix in DIAGNOSTIC_SUFFIXES
        )

    floats = [n for n in names if kinds[n] == "float" and not diagnostic(n)]
    valued = {n: model.n_blocks - int(model.is_missing(n).sum()) for n in floats}
    floats = [n for n in floats if valued[n] > 0]
    estimated = [n for n in floats if any(n + suffix in kinds for suffix in DIAGNOSTIC_SUFFIXES)]
    grades = [n for n in floats if n.lower() not in _NOT_A_GRADE]
    categories = [n for n in names if kinds[n] == "category"]
    display: dict[str, Any] = {}
    if estimated:
        display["color_by"] = estimated[0]
    elif grades:
        display["color_by"] = min(grades, key=lambda n: valued[n])  # first of equals
    elif categories:
        display["color_by"] = categories[0]
    elif names:
        display["color_by"] = floats[0] if floats else names[0]
    for name in categories:
        values = [str(c) for c in model.attribute(name).categories]
        kept = [v for v in values if v.strip().lower() not in _AIR_LIKE]
        # (The filter travels in the request URL: not for a dictionary of hundreds.)
        if kept and len(kept) < len(values) and len(kept) <= 64:
            display["category_filter"] = {"attribute": name, "values": kept}
            break
    return display


# ---------------------------------------------------------------------- selection


def parse_constraint(text: str | None, what: str = "where") -> Constraint | None:
    """``where`` / ``clip`` text -> constraint; ``None`` for nothing. Never evaluates code."""
    if text is None or not text.strip():
        return None
    text = text.strip()
    try:
        if text.startswith("{"):
            try:
                document = json.loads(text)
            except RecursionError:
                raise BlockQueryError(f"{what}: JSON is nested too deeply") from None
            except ValueError as exc:
                raise BlockQueryError(f"{what}: not valid JSON ({exc})") from None
            return constraint_from_dict(document)
        return AttributeConstraint(text)
    except BlockQueryError:
        raise
    except (ValueError, TypeError) as exc:  # ConstraintError and ExpressionError included
        raise BlockQueryError(str(exc)) from None


def select(data: BlockData, constraint: Constraint | None) -> np.ndarray:
    """``constraint.mask(model, resolver)`` with user errors turned into ``BlockQueryError``."""
    model = data.model
    if constraint is None:
        return np.ones(model.n_blocks, dtype=bool)
    try:
        mask = np.asarray(constraint.mask(model, data.resolver))
    except (ValueError, TypeError, KeyError) as exc:
        raise BlockQueryError(str(exc)) from None
    if mask.dtype != bool or mask.shape != (model.n_blocks,):
        raise BlockQueryError("the constraint did not give one true / false per block")
    return mask


def exposed_blocks(model: BlockModel, mask: np.ndarray) -> np.ndarray:
    """Blocks of the selection ``mask`` that may show a face (see the module docstring).

    Returns a bool mask, a subset of ``mask``. Every selected block left out has all six
    face neighbours selected.
    """
    definition = model.definition
    if model.is_dense:
        full = mask.reshape(definition.shape)
    else:
        units = model.volume_units()
        selected = np.bincount(
            model.parent[mask], weights=units[mask], minlength=definition.n_parents
        )
        per_parent = float(1 << int(np.sum(definition.max_depth)))
        full = (selected == per_parent).reshape(definition.shape)
    interior = full.copy()
    for axis in range(3):
        low = [slice(None)] * 3
        high = [slice(None)] * 3
        low[axis] = slice(0, -1)
        high[axis] = slice(1, None)
        first = [slice(None)] * 3
        last = [slice(None)] * 3
        first[axis] = 0
        last[axis] = -1
        interior[tuple(high)] &= full[tuple(low)]
        interior[tuple(low)] &= full[tuple(high)]
        # The model's own faces are exposed.
        interior[tuple(first)] = False
        interior[tuple(last)] = False
    return mask & ~interior.reshape(-1)[model.parent]


def parse_attributes(model: BlockModel, text: str | None) -> list[str]:
    """Comma-separated attribute names; unknown names are a ``BlockQueryError``."""
    if text is None or not text.strip():
        return []
    names = list(dict.fromkeys(n.strip() for n in text.split(",") if n.strip()))
    unknown = [n for n in names if n not in model]
    if unknown:
        raise BlockQueryError(
            f"unknown attribute(s) {', '.join(unknown)}; "
            f"available: {', '.join(model.attribute_names)}"
        )
    return names


@dataclass(frozen=True)
class BlockSelection:
    """Which blocks a request sends. ``rows`` is increasing."""

    rows: np.ndarray
    n_selected: int
    selected_volume: float
    lod: str
    clipped: bool
    #: selected blocks that have a value of ``present``; ``n_selected`` without it
    n_visible: int = -1

    def __post_init__(self) -> None:
        if self.n_visible < 0:
            object.__setattr__(self, "n_visible", self.n_selected)


def select_blocks(
    data: BlockData,
    where: Constraint | None = None,
    lod: str = "auto",
    clip: Constraint | None = None,
    present: str | None = None,
) -> BlockSelection:
    """Evaluates ``where`` and applies the level of detail or the ``clip`` constraint.

    With ``present`` (an attribute name) only the selected blocks that have a value of it
    are visible: the shell and the clip are taken of those (module docstring).
    """
    if lod not in LODS:
        raise BlockQueryError(f"lod must be one of {', '.join(LODS)}")
    model = data.model
    mask = select(data, where)
    n_selected = int(mask.sum())
    volume = float(model.volumes()[mask].sum()) if n_selected else 0.0
    n_visible = n_selected
    if present is not None:
        if present not in model:
            raise BlockQueryError(
                f"unknown attribute {present}; available: {', '.join(model.attribute_names)}"
            )
        mask = mask & ~model.is_missing(present)
        n_visible = int(mask.sum())
    applied = "full"
    if clip is not None:
        send = mask & select(data, clip)
    elif lod == "shell" or (lod == "auto" and n_visible > AUTO_SHELL_BLOCKS):
        send = exposed_blocks(model, mask)
        applied = "shell"
    else:
        send = mask
    return BlockSelection(
        np.flatnonzero(send), n_selected, volume, applied, clip is not None, n_visible
    )


# ---------------------------------------------------------------------- payload


def block_payload(
    data: BlockData,
    object_id: str,
    attributes: list[str] | None = None,
    where: Constraint | None = None,
    lod: str = "auto",
    clip: Constraint | None = None,
    present: str | None = None,
) -> bytes:
    """The "blocks" BKMA payload (layout in the module docstring)."""
    model = data.model
    selection = select_blocks(data, where, lod, clip, present)
    rows = selection.rows
    everything = len(rows) == model.n_blocks
    take = slice(None) if everything else rows
    arrays: dict[str, np.ndarray] = {}
    if not everything:
        arrays["row"] = rows.astype(np.uint32)
    dense = model.is_dense
    if not dense:
        arrays["parent"] = model.parent[take].astype(np.uint32)
        arrays["level"] = model.level[take]
        arrays["sub"] = model.sub[take]
    described: list[dict[str, Any]] = []
    for name in attributes or []:
        spec = model.attribute(name)
        column = model[name][take]
        if spec.kind == "category":
            key = f"code:{name}"
            arrays[key] = column.astype(np.int16 if len(spec.categories) < 2**15 else np.int32)
            described.append({"name": name, "type": "categorical", "array": key,
                              "categories": [str(c) for c in spec.categories]})  # fmt: skip
        else:
            key = f"value:{name}"
            arrays[key] = column.astype(np.float32)
            described.append({"name": name, "type": "numeric", "array": key,
                              "unit": spec.unit or data.units.get(name) or None})  # fmt: skip
    n = len(rows)
    meta = {
        "object_id": object_id,
        "definition": definition_dict(model.definition),
        "structure": "dense" if dense else "octree",
        "rows": "all" if everything else "listed",
        "n_total": int(model.n_blocks),
        "n_selected": selection.n_selected,
        "n_visible": selection.n_visible,
        "n_blocks": int(n),
        "selected_volume": selection.selected_volume,
        "lod": selection.lod,
        "lod_requested": lod,
        "clipped": selection.clipped,
        "where": None if where is None else _describe(where),
        "present": present,
        "bytes_per_block": (sum(a.nbytes for a in arrays.values()) / n) if n else 0.0,
        "attributes": described,
    }
    return pack_arrays("blocks", meta, arrays)


def _describe(constraint: Constraint) -> Any:
    try:
        return constraint.to_dict()
    except ValueError:
        return repr(constraint)


def block_details(data: BlockData, row: int) -> dict[str, Any]:
    """Everything about one block, at full precision (the viewer's pick panel)."""
    model = data.model
    if not 0 <= row < model.n_blocks:
        raise KeyError(row)
    where = np.array([row])
    values: dict[str, Any] = {}
    for name, spec in model.attributes.items():
        if spec.kind == "category":
            text = model.get(name, where, decode=True)[0]
            values[name] = None if text is None else str(text)
        else:
            value = float(model.get(name, where)[0])
            values[name] = value if np.isfinite(value) else None
    size = model.sizes(where)[0]
    return {
        "row": int(row),
        # int64: beyond what JSON numbers hold exactly
        "block_id": str(int(model.block_ids(where)[0])),
        "ijk": [int(v) for v in model.ijk(where)[0]],
        "parent": int(model.parent[row]),
        "level": int(model.level[row]),
        "sub": [int(v) for v in model.sub[row]],
        "centroid": [float(v) for v in model.centroids(where)[0]],
        "size": [float(v) for v in size],
        "volume": float(size[0] * size[1] * size[2]),
        "attributes": values,
    }
