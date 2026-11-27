"""Build the synthetic layered deposit and write it to a folder (geology_contracts.md 13).

Input files (what a geologist would have): ``collar.csv``, ``survey.csv``, ``assay.csv``,
``geology.csv``, ``units.json``, ``topo.obj``. Everything named ``truth*`` and the ore solid
are the oracle and must not be fed to the pipeline under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.synth.generate import _write_csv, _write_obj
from bakumine.synth.layered import (
    CATEGORIES,
    FAULT_BLOCKS,
    LAYERED_PRESETS,
    LITH,
    ROCK,
    WEATHERING,
    LayeredConfig,
    LayeredTruth,
)

GENERATOR_VERSION = 1
TRUTH_LAYERS = ("lith", "weathering", "fault_block", "rock")
#: logged spellings by hole index ``h``: ``spellings[h % len(spellings)]``
SPELLINGS: dict[str, tuple[str, ...]] = {
    "SST": ("SST", "Sst", "sst "),
    "SLT": ("SLT", "Slt"),
    "AND": ("AND",),
    "DIO": ("DIO", "Dio", "HB DIO"),
    "GRD": ("GRD",),
    "Q4": ("SOIL", "SOL"),
    "OX": ("OZ",),
    "FLT": ("FLT", "FB"),
    "CL": ("CL",),
}
SOIL_LOGGED = (0, 1, 2)  # h % 5: the cover is logged as soil
OXIDE_LOGGED = (0, 3)  # h % 5: "OZ" is logged instead of the rock down to the oxide base
FAULT_NOT_LOGGED = 9  # h % 10: no fault-zone interval
# Volume lattice: one point per cell, shifted in plan by a rank-1 rule that changes with the
# level, so that a thin vertical body is not sampled at one fixed phase (see lattice_volumes).
_PHASES = 31
_PHASE_STEP = 12

UNITS_JSON: dict[str, Any] = {
    "format": "bakumine.units",
    "version": 1,
    "name": "synthetic layered deposit",
    "source": {"table": "geology", "column": "lith"},
    "fold": {"case": True, "strip": True},
    "unmapped": "error",
    "default_density": 2.70,
    "default_density_source": "assumed",
    "display_colours": {"ORE": "#e8c547"},
    "units": [
        {"code": "Q4", "name": "Soil and cover", "kind": "cover", "aliases": ["SOIL", "SOL"],
         "colour": "#d8c37a", "density": 1.90, "density_source": "measured"},
        {"code": "OX", "name": "Oxidised zone (logged)", "kind": "weathering", "aliases": ["OZ"]},
        {"code": "SST", "name": "Sandstone", "kind": "lithology", "group": "SED",
         "colour": "#c9b28f", "density": 2.55, "density_source": "measured"},
        {"code": "SLT", "name": "Siltstone", "kind": "lithology", "group": "SED",
         "colour": "#8f8a6e", "density": 2.65, "density_source": "measured"},
        {"code": "AND", "name": "Andesite", "kind": "lithology", "colour": "#6e7f5a",
         "density": 2.75, "density_source": "measured"},
        {"code": "DIO", "name": "Diorite", "kind": "intrusion", "aliases": ["HB DIO"], "age": 1,
         "colour": "#7a6a9c", "density": 2.80, "density_source": "measured"},
        {"code": "GRD", "name": "Granodiorite dyke", "kind": "intrusion", "age": 2,
         "colour": "#c2557a", "density": 2.70, "density_source": "measured"},
        {"code": "FLT", "name": "Fault zone", "kind": "fault-zone", "aliases": ["FB"]},
        {"code": "CL", "name": "Core loss", "kind": "missing"},
        {"code": "BX", "name": "Breccia, not modelled", "kind": "other", "aliases": ["BRECCIA"]},
    ],
    "weathering_zones": [
        {"code": "OXIDE", "name": "Oxidised", "colour": "#d98c3a", "density": 2.30,
         "density_source": "measured"},
        {"code": "TRANSITION", "name": "Transition", "colour": "#c7b35a"},
        {"code": "FRESH", "name": "Fresh", "colour": "#7d8c99"},
    ],
}  # fmt: skip

CSV_FORMATS: dict[str, dict[str, str]] = {
    "collar": {"hole_id": "{}", "x": "{:.3f}", "y": "{:.3f}", "z": "{:.6f}", "max_depth": "{:.2f}"},
    "survey": {"hole_id": "{}", "depth": "{:.2f}", "dip": "{:.2f}", "azimuth": "{:.2f}"},
    "assay": {
        "hole_id": "{}",
        "depth_from": "{:.2f}",
        "depth_to": "{:.2f}",
        "au": "{:.3f}",
        "s_ppm": "{:d}",
        "wf": "{}",
    },
    "geology": {"hole_id": "{}", "depth_from": "{:.2f}", "depth_to": "{:.2f}", "lith": "{}"},
    "truth_contacts": {
        "hole_id": "{}",
        "layer": "{}",
        "depth": "{:.6f}",
        "x": "{:.6f}",
        "y": "{:.6f}",
        "z": "{:.6f}",
        "above": "{}",
        "below": "{}",
    },
}
SURFACES = ("truth_cover_base", "truth_oxide_base", "truth_transition_base")


class SelfCheckError(ValueError):
    """A generator self-check of geology_contracts.md 13 is not met."""


@dataclass
class Holes:
    """Straight holes: position at depth ``d`` is ``collar + d * direction``."""

    hole_id: np.ndarray  # object
    collar: np.ndarray  # (n, 3)
    dip: np.ndarray
    azimuth: np.ndarray
    length: np.ndarray
    direction: np.ndarray  # (n, 3) unit

    def xyz(self, hole: np.ndarray, depth: np.ndarray) -> np.ndarray:
        return self.collar[hole] + np.asarray(depth)[:, None] * self.direction[hole]


@dataclass
class LayeredDeposit:
    config: LayeredConfig
    seed: int
    truth: LayeredTruth
    tables: dict[str, dict[str, np.ndarray]]  # collar, survey, assay, geology, truth_*
    truth_json: dict[str, Any]
    holes: Holes | None = None
    meshes: dict[str, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    units: dict[str, Any] = field(default_factory=dict)

    def write(self, out: str | Path) -> list[Path]:
        return _write(self, Path(out))


# --------------------------------------------------------------------------------------
# drilling
# --------------------------------------------------------------------------------------
def plan_holes(cfg: LayeredConfig, truth: LayeredTruth) -> Holes:
    s = cfg.hole_spacing
    nx = round((cfg.x_range[1] - cfg.x_range[0]) / s)
    ny = round((cfg.y_range[1] - cfg.y_range[0]) / s)
    j, i = (a.ravel() for a in np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij"))
    h = j * nx + i
    x = cfg.x_range[0] + 0.5 * s + s * i
    y = cfg.y_range[0] + 0.5 * s + s * j
    vertical = h % cfg.vertical_modulus == cfg.vertical_remainder
    dip = np.where(vertical, -90.0, cfg.hole_dip)
    azimuth = np.where(vertical, 0.0, cfg.hole_azimuth)
    length = np.where(vertical, cfg.vertical_length_factor * cfg.hole_length, cfg.hole_length)
    d, a = np.radians(dip), np.radians(azimuth)
    direction = np.column_stack([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)])
    direction[vertical] = (0.0, 0.0, -1.0)
    return Holes(
        hole_id=np.array([f"L{k + 1:03d}" for k in h], dtype=object),
        collar=np.column_stack([x, y, truth.topo_z(x, y)]),
        dip=dip,
        azimuth=azimuth,
        length=length,
        direction=direction,
    )


def label_changes(
    truth: LayeredTruth, holes: Holes, layer: str, step: float, tolerance: float
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray, np.ndarray]]]:
    """Every change of a label along every hole.

    Returns the code at each collar and, per hole, ``(depth, code above, code below)`` sorted
    by depth. The path is scanned every ``step`` metres and each change is bisected to
    ``tolerance``. Two changes inside one scan step are both found when they separate three
    different labels; a label that appears and disappears within one step is missed.
    """
    hole_parts, depth_parts = [], []
    for k, length in enumerate(holes.length):
        d = np.append(np.arange(0.0, length, step), length)
        hole_parts.append(np.full(len(d), k))
        depth_parts.append(d)
    hole, depth = np.concatenate(hole_parts), np.concatenate(depth_parts)
    code = truth.codes(holes.xyz(hole, depth), layer)
    first = np.flatnonzero(np.r_[True, hole[1:] != hole[:-1]])
    change = np.flatnonzero((hole[1:] == hole[:-1]) & (code[1:] != code[:-1]))
    b_hole, lo, stop = hole[change], depth[change], depth[change + 1]
    above, end = code[change], code[change + 1]
    found: list[tuple[np.ndarray, ...]] = []
    while len(b_hole):
        hi, below = stop.copy(), end.copy()
        while float(np.max(hi - lo)) > tolerance:
            mid = 0.5 * (lo + hi)
            c = truth.codes(holes.xyz(b_hole, mid), layer)
            left = c == above
            lo, hi = np.where(left, mid, lo), np.where(left, hi, mid)
            below = np.where(left, below, c)
        found.append((b_hole, 0.5 * (lo + hi), above, below))
        more = below != end  # another change between this one and the scan point
        b_hole, lo, stop, above, end = b_hole[more], hi[more], stop[more], below[more], end[more]
    if found:
        f_hole, f_depth, f_above, f_below = (
            np.concatenate(part) for part in zip(*found, strict=True)
        )
    else:
        f_hole = np.zeros(0, dtype=np.int64)
        f_depth = np.zeros(0)
        f_above = f_below = np.zeros(0, dtype=np.int8)
    order = np.lexsort((f_depth, f_hole))
    f_hole, f_depth, f_above, f_below = (a[order] for a in (f_hole, f_depth, f_above, f_below))
    per_hole = []
    for k in range(len(holes.length)):
        rows = slice(*np.searchsorted(f_hole, [k, k + 1]))
        per_hole.append((f_depth[rows], f_above[rows], f_below[rows]))
    return code[first], per_hole


def log_hole(
    cfg: LayeredConfig,
    h: int,
    length: float,
    bedrock: tuple[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    cover_base: float | None,
    oxide_base: float | None,
    fault_depths: np.ndarray,
    core_loss: list[tuple[float, float]],
) -> list[tuple[float, float, str, str]]:
    """The logged intervals of hole ``h``: ``(depth_from, depth_to, spelling, unit)``.

    Depths are whole centimetres. Layers are painted in the order of the contract; a later
    one replaces an earlier one: rock under the cover, soil, "OZ", fault zone, core loss.
    """
    scale = 10**cfg.depth_decimals

    def cm(depth: float) -> int:
        return round(float(depth) * scale)

    end = cm(length)
    start_code, (depths, _, below) = bedrock
    edges = [0, *[cm(d) for d in depths], end]
    units = [LITH[start_code], *[LITH[c] for c in below]]
    paints: list[tuple[int, int, str]] = [
        (a, b, unit) for a, b, unit in zip(edges[:-1], edges[1:], units, strict=True)
    ]
    soil_base = 0
    if h % 5 in SOIL_LOGGED and cover_base is not None:
        soil_base = cm(cover_base)
        paints.append((0, soil_base, "Q4"))
    if h % 5 in OXIDE_LOGGED and oxide_base is not None:
        paints.append((soil_base, cm(oxide_base), "OX"))
    if h % 10 != FAULT_NOT_LOGGED:
        half = cfg.fault_zone_half_length
        paints.extend((max(cm(d - half), 0), min(cm(d + half), end), "FLT") for d in fault_depths)
    paints.extend((cm(a), min(cm(b), end), "CL") for a, b in core_loss)

    cuts = sorted({0, end, *[p for a, b, _ in paints for p in (a, b) if 0 <= p <= end]})
    rows: list[tuple[float, float, str, str]] = []
    for a, b in zip(cuts[:-1], cuts[1:], strict=True):
        unit = next(u for pa, pb, u in reversed(paints) if pa <= a and b <= pb)
        spellings = SPELLINGS[unit]
        code = spellings[h % len(spellings)]
        if rows and rows[-1][2] == code:
            rows[-1] = (rows[-1][0], b / scale, code, unit)
        else:
            rows.append((a / scale, b / scale, code, unit))
    return rows


def _first_change(changes: tuple[np.ndarray, np.ndarray, np.ndarray], above: int) -> float | None:
    depths, codes_above, _ = changes
    hits = np.flatnonzero(codes_above == above)
    return float(depths[hits[0]]) if hits.size else None


# --------------------------------------------------------------------------------------
# truth volumes, blocks, meshes
# --------------------------------------------------------------------------------------
def lattice_volumes(truth: LayeredTruth, cfg: LayeredConfig, spacing: float) -> dict[str, float]:
    """Volume of every lith category in the box by counting lattice points (one per cell).

    Level ``k`` (elevation ``z_min + (k + 0.5) h``) holds the points
    ``(x_min + (i + a_r) h, y_min + (j + b_r) h)`` with ``r = k mod 31``,
    ``a_r = (r + 0.5) / 31`` and ``b_r = ((12 r) mod 31 + 0.5) / 31``. An unshifted lattice
    samples the 8 m dyke (strike 045) at one fixed phase and misses its volume by several per
    cent at any spacing of this size; the shifts average over the phases.
    """
    h = spacing
    nx = round((cfg.x_range[1] - cfg.x_range[0]) / h)
    ny = round((cfg.y_range[1] - cfg.y_range[0]) / h)
    nz = round((cfg.z_range[1] - cfg.z_range[0]) / h)
    counts = np.zeros(len(LITH), dtype=np.int64)
    for r in range(min(_PHASES, nz)):
        x = cfg.x_range[0] + (np.arange(nx) + (r + 0.5) / _PHASES) * h
        y = cfg.y_range[0] + (np.arange(ny) + ((_PHASE_STEP * r) % _PHASES + 0.5) / _PHASES) * h
        plan = truth.plan(x[:, None], y[None, :])
        for k in range(r, nz, _PHASES):
            codes = truth.lith_codes(plan, cfg.z_range[0] + (k + 0.5) * h)
            counts += np.bincount(codes.ravel(), minlength=len(LITH))
    return {unit: float(n) * h**3 for unit, n in zip(LITH, counts, strict=True)}


def build_blocks(truth: LayeredTruth, cfg: LayeredConfig) -> dict[str, np.ndarray]:
    """Truth blocks over the box, row ``i + nx (j + ny k)``."""
    size = np.array(cfg.block_size)
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]])
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]])
    nx, ny, nz = (int(n) for n in np.round((hi - lo) / size))
    cx, cy, cz = (lo[a] + (np.arange(n) + 0.5) * size[a] for a, n in enumerate((nx, ny, nz)))
    sx, sy, sz = cfg.block_lattice
    offsets = [((np.arange(n) + 0.5) / n - 0.5) * size[a] for a, n in enumerate((sx, sy, sz))]
    fine_x = (cx[:, None] + offsets[0][None, :]).ravel()
    fine_y = (cy[:, None] + offsets[1][None, :]).ravel()
    fine = truth.plan(fine_x[:, None], fine_y[None, :])
    centre = truth.plan(cx[:, None], cy[None, :])

    lith = np.empty((nz, ny, nx), dtype=np.int8)
    weathering = np.empty((nz, ny, nx), dtype=np.int8)
    counts = np.zeros((len(LITH), nz, ny, nx), dtype=np.int32)
    for k in range(nz):
        lith[k] = truth.lith_codes(centre, cz[k]).T
        weathering[k] = truth.weathering_codes(centre, cz[k]).T
        for dz in offsets[2]:
            codes = truth.lith_codes(fine, cz[k] + dz).reshape(nx, sx, ny, sy)
            for c in range(len(LITH)):
                counts[c, k] += (codes == c).sum(axis=(1, 3)).T
    zz, yy, xx = (a.ravel() for a in np.meshgrid(cz, cy, cx, indexing="ij"))
    points = np.column_stack([xx, yy, zz])
    columns: dict[str, np.ndarray] = {
        "x": xx,
        "y": yy,
        "z": zz,
        "lith": np.array(LITH, dtype=object)[lith.ravel()],
        "weathering": np.array(WEATHERING, dtype=object)[weathering.ravel()],
        "fault_block": np.array(FAULT_BLOCKS, dtype=object)[truth.codes(points, "fault_block")],
        "rock": np.array(ROCK, dtype=object)[truth.codes(points, "rock")],
    }
    per_block = float(sx * sy * sz)
    for c, unit in enumerate(LITH):
        columns[f"frac_{unit}"] = counts[c].ravel() / per_block
    return columns


def grid_surface(cfg: LayeredConfig, z_of: Any) -> tuple[np.ndarray, np.ndarray]:
    """Regular-grid DTM over the box; two triangles per cell, counter-clockwise in plan."""
    s = cfg.surface_spacing
    nx = round((cfg.x_range[1] - cfg.x_range[0]) / s) + 1
    ny = round((cfg.y_range[1] - cfg.y_range[0]) / s) + 1
    x = cfg.x_range[0] + s * np.arange(nx)
    y = cfg.y_range[0] + s * np.arange(ny)
    xx, yy = (a.ravel() for a in np.meshgrid(x, y, indexing="ij"))  # vertex = i * ny + j
    vertices = np.column_stack([xx, yy, z_of(xx, yy)])
    i, j = (a.ravel() for a in np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing="ij"))
    a, b, c, d = i * ny + j, (i + 1) * ny + j, (i + 1) * ny + j + 1, i * ny + j + 1
    faces = np.concatenate([np.column_stack([a, b, c]), np.column_stack([a, c, d])])
    return vertices, faces.astype(np.int64)


def fault_mesh(truth: LayeredTruth, cfg: LayeredConfig) -> tuple[np.ndarray, np.ndarray]:
    """The fault plane clipped to the box: a convex polygon as a triangle fan, normal up."""
    n, origin = truth.fault_normal, truth.fault_point
    along = np.cross([0.0, 0.0, 1.0], n)
    along /= np.linalg.norm(along)
    up_dip = np.cross(n, along)
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]]) - origin
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]]) - origin
    reach = 4.0 * float(np.linalg.norm(hi - lo) + np.linalg.norm(lo) + np.linalg.norm(hi))
    polygon = [reach * (a * along + b * up_dip) for a, b in ((-1, -1), (1, -1), (1, 1), (-1, 1))]
    for axis in range(3):
        for bound, sign in ((lo[axis], 1.0), (hi[axis], -1.0)):
            clipped: list[np.ndarray] = []
            for p, q in zip(polygon, polygon[1:] + polygon[:1], strict=True):
                dp, dq = sign * (p[axis] - bound), sign * (q[axis] - bound)
                if dp >= 0.0:
                    clipped.append(p)
                if (dp > 0.0 > dq) or (dp < 0.0 < dq):
                    cut = p + (q - p) * (dp / (dp - dq))
                    cut[axis] = bound
                    clipped.append(cut)
            polygon = clipped
    if len(polygon) < 3:
        raise ValueError("the fault plane does not cross the box")
    vertices = np.array(polygon) + origin
    faces = np.array([[0, k, k + 1] for k in range(1, len(polygon) - 1)], dtype=np.int64)
    a, b, c = (vertices[faces[0, k]] for k in range(3))
    if float(np.cross(b - a, c - a) @ n) < 0.0:
        faces = faces[:, ::-1].copy()
    return vertices, faces


# --------------------------------------------------------------------------------------
# the deposit
# --------------------------------------------------------------------------------------
def generate_layered(config: LayeredConfig | None = None, seed: int = 42) -> LayeredDeposit:
    """Generate the deposit. The same ``config`` and ``seed`` always give the same result."""
    cfg = config or LAYERED_PRESETS["layered"]
    truth = LayeredTruth(cfg)
    logging_seq, assay_seq, _reserve = np.random.SeedSequence(seed).spawn(3)
    rng_log, rng_assay = np.random.default_rng(logging_seq), np.random.default_rng(assay_seq)
    holes = plan_holes(cfg, truth)
    n_holes = len(holes.hole_id)

    step, tol = cfg.contact_scan_step, cfg.contact_tolerance
    changes = {
        layer: label_changes(truth, holes, layer, step, tol) for layer in (*TRUTH_LAYERS, "bedrock")
    }

    # logging
    geology: list[list[tuple[float, float, str, str]]] = []
    for h in range(n_holes):
        length = float(holes.length[h])
        n_loss = int(rng_log.poisson(cfg.core_loss_rate * length))
        loss_from = rng_log.uniform(0.0, length, n_loss)
        loss_length = rng_log.uniform(*cfg.core_loss_length, n_loss)
        geology.append(
            log_hole(
                cfg,
                h,
                length,
                (int(changes["bedrock"][0][h]), changes["bedrock"][1][h]),
                _first_change(changes["lith"][1][h], LITH.index("Q4")),
                _first_change(changes["weathering"][1][h], WEATHERING.index("OXIDE")),
                changes["fault_block"][1][h][0],
                sorted(zip(loss_from.tolist(), (loss_from + loss_length).tolist(), strict=True)),
            )
        )

    # assays
    assay: dict[str, list[np.ndarray]] = {k: [] for k in CSV_FORMATS["assay"]}
    for h in range(n_holes):
        length = float(holes.length[h])
        edges = np.append(np.arange(0.0, length, cfg.sample_length), length)
        n = len(edges) - 1
        mid = holes.xyz(np.full(n, h), 0.5 * (edges[:-1] + edges[1:]))
        ore = truth.codes(mid, "rock") == ROCK.index("ORE")
        zone = truth.codes(mid, "weathering")
        block = truth.codes(mid, "fault_block")
        e_au, e_s = rng_assay.standard_normal(n), rng_assay.standard_normal(n)
        au = np.where(
            ore,
            np.exp(np.log(cfg.ore_median) + cfg.ore_sigma * e_au),
            np.exp(np.log(cfg.waste_median) + cfg.waste_sigma * e_au),
        )
        base = np.array([*cfg.sulphur_ppm, np.nan])[zone]
        wf = np.where(ore, np.array([f"{b}-ORE" for b in FAULT_BLOCKS], dtype=object)[block], "WST")
        assay["hole_id"].append(np.full(n, holes.hole_id[h], dtype=object))
        assay["depth_from"].append(edges[:-1])
        assay["depth_to"].append(edges[1:])
        assay["au"].append(np.maximum(np.round(au, cfg.assay_decimals), cfg.assay_floor))
        assay["s_ppm"].append(np.round(base * np.exp(cfg.sulphur_sigma * e_s)).astype(np.int64))
        assay["wf"].append(wf.astype(object))

    contacts: dict[str, list[np.ndarray]] = {k: [] for k in CSV_FORMATS["truth_contacts"]}
    for layer in TRUTH_LAYERS:
        names = np.array(CATEGORIES[layer], dtype=object)
        for h in range(n_holes):
            depth, above, below = changes[layer][1][h]
            xyz = holes.xyz(np.full(len(depth), h), depth)
            contacts["hole_id"].append(np.full(len(depth), holes.hole_id[h], dtype=object))
            contacts["layer"].append(np.full(len(depth), layer, dtype=object))
            contacts["depth"].append(depth)
            for axis, name in enumerate("xyz"):
                contacts[name].append(xyz[:, axis])
            contacts["above"].append(names[above])
            contacts["below"].append(names[below])

    def per_hole(counts: list[int]) -> np.ndarray:
        return np.repeat(holes.hole_id, counts)

    tables: dict[str, dict[str, np.ndarray]] = {
        "collar": {
            "hole_id": holes.hole_id,
            "x": holes.collar[:, 0],
            "y": holes.collar[:, 1],
            "z": holes.collar[:, 2],
            "max_depth": holes.length,
        },
        "survey": {
            "hole_id": per_hole([2] * n_holes),
            "depth": np.column_stack([np.zeros(n_holes), holes.length]).ravel(),
            "dip": np.repeat(holes.dip, 2),
            "azimuth": np.repeat(holes.azimuth, 2),
        },
        "assay": {k: np.concatenate(v) for k, v in assay.items()},
        "geology": {
            "hole_id": per_hole([len(g) for g in geology]),
            "depth_from": np.array([row[0] for g in geology for row in g]),
            "depth_to": np.array([row[1] for g in geology for row in g]),
            "lith": np.array([row[2] for g in geology for row in g], dtype=object),
        },
        "truth_contacts": {k: np.concatenate(v) for k, v in contacts.items()},
        "truth_blocks": build_blocks(truth, cfg),
    }

    # meshes
    lens_vertices, lens_faces = truth.lens.mesh(cfg.lens.mesh_subdivisions)
    margin = cfg.topo.dtm_margin
    dtm_lo = np.array([cfg.x_range[0], cfg.y_range[0]]) - margin
    dtm_hi = np.array([cfg.x_range[1], cfg.y_range[1]]) + margin
    meshes = {
        "topo": truth.topo.dtm(dtm_lo, dtm_hi, holes.collar[:, :2]),
        "ore_solid": (lens_vertices, lens_faces),
        "truth_fault_F1": fault_mesh(truth, cfg),
        "truth_cover_base": grid_surface(
            cfg, lambda x, y: truth.topo_z(x, y) - truth.cover_depth(x, y)
        ),
        "truth_oxide_base": grid_surface(
            cfg, lambda x, y: truth.topo_z(x, y) - truth.oxide_depth(x, y)
        ),
        "truth_transition_base": grid_surface(
            cfg, lambda x, y: truth.topo_z(x, y) - truth.transition_depth(x, y)
        ),
    }

    # self-checks
    modelled = ("SST", "SLT", "AND", "DIO", "GRD")
    holes_per_unit = {
        unit: sum(any(row[3] == unit for row in g) for g in geology) for unit in SPELLINGS
    }
    fault_holes = sum(len(changes["fault_block"][1][h][0]) > 0 for h in range(n_holes))
    lens_cover = float(
        np.min(truth.topo_z(lens_vertices[:, 0], lens_vertices[:, 1]) - lens_vertices[:, 2])
    )
    coarse, fine = (lattice_volumes(truth, cfg, h) for h in cfg.volume_spacings)
    difference = {u: coarse[u] / fine[u] - 1.0 for u in LITH if fine[u] > 0.0}
    minimum = {unit: cfg.min_holes_per_unit for unit in modelled}
    minimum.update(dict(cfg.min_holes_exceptions))
    failures = [
        f"{unit} is logged in {holes_per_unit[unit]} holes (< {minimum[unit]})"
        for unit in modelled
        if holes_per_unit[unit] < minimum[unit]
    ]
    if fault_holes < cfg.min_fault_holes:
        failures.append(f"{fault_holes} holes cross F1 (< {cfg.min_fault_holes})")
    if lens_cover < cfg.min_lens_cover:
        failures.append(f"the ore lens comes within {lens_cover:.1f} m of the topography")
    failures.extend(
        f"volume of {unit} differs by {100 * d:.2f} % between the two lattices"
        for unit, d in difference.items()
        if unit != "AIR" and abs(d) > cfg.volume_tolerance
    )
    if failures:
        raise SelfCheckError("layered deposit self-check failed: " + "; ".join(failures))

    d_f = truth.fault_down_dip
    truth_json: dict[str, Any] = {
        "generator": {"name": "bakumine.synth.layered", "version": GENERATOR_VERSION},
        "preset": cfg.name,
        "seed": seed,
        "config": json.loads(json.dumps(cfg.to_dict())),
        "coordinate_system": {
            "axes": "x = east, y = north, z = up",
            "units": "metres",
            "notation": "X = x - origin_x, Y = y - origin_y; azimuths clockwise from north",
            "dip_convention": "negative downward (vertical hole = -90)",
        },
        "box": {"x": list(cfg.x_range), "y": list(cfg.y_range), "z": list(cfg.z_range)},
        "layers": {name: list(CATEGORIES[name]) for name in TRUTH_LAYERS},
        "formulas": {
            "order": "layers -> fault F1 -> DIO stock -> GRD dyke -> cover Q4 -> AIR; later wins",
            "topography": (
                "z_T = base + slope_x X + slope_y Y + sum amp sin(2 pi (X sin d + Y cos d) / "
                "wavelength + phase); rounded to 1e-6 m"
            ),
            "layers": (
                "s = n_b . (q - P0), u = t . (q - P0), s_BC = -(slt_thickness + slt_amplitude "
                "sin(2 pi u / slt_wavelength)); SST if s >= 0; SLT if s_BC <= s < 0; AND if "
                "s < s_BC; q = restored position"
            ),
            "bedding_normal": "n_b = (sin dip sin az, sin dip cos az, cos dip), dip azimuth az",
            "fault": (
                "hanging wall (FB2) where n_f . (p - Pf) > 0, footwall FB1 otherwise; "
                "q = p - D in the hanging wall, q = p in the footwall; D = slip d_f, "
                "d_f = (cos dip sin az, cos dip cos az, -sin dip), slip = throw / sin dip"
            ),
            "stock": (
                "DIO where (a / A)^2 + (b / B)^2 + (c / C)^2 < 1; a, b = horizontal offsets "
                "from the centre along stock_azimuth and stock_azimuth + 90, c = vertical"
            ),
            "dyke": "GRD where |m . (p - Q)| < dyke_half_thickness, m horizontal",
            "cover": (
                "Q4 where 0 <= z_T - z < c; c = cover_mean + cover_amplitude "
                "sin(2 pi X / wx) sin(2 pi Y / wy)"
            ),
            "air": "AIR where z > z_T (lith, weathering and rock layers)",
            "weathering": (
                "depth = z_T - z; OXIDE if depth < d_ox; TRANSITION if depth < d_ox + "
                "transition_thickness; else FRESH; d_ox = oxide_mean + oxide_amplitude_x "
                "sin(2 pi X / wx + phase) + oxide_amplitude_y cos(2 pi Y / wy)"
            ),
            "rock": "ORE inside the lens (not offset by the fault), else WASTE; AIR above z_T",
            "holes": "straight: p(d) = collar + d (cos dip sin az, cos dip cos az, sin dip)",
        },
        "bedding": {
            "point": truth.bedding_point.tolist(),
            "normal": truth.bedding_normal.tolist(),
            "strike_direction": truth.strike.tolist(),
            "dip": cfg.bedding_dip,
            "dip_azimuth": cfg.bedding_dip_azimuth,
        },
        "fault": {
            "name": "F1",
            "blocks": {"footwall": "FB1", "hanging_wall": "FB2"},
            "point": truth.fault_point.tolist(),
            "normal": truth.fault_normal.tolist(),
            "dip": cfg.fault_dip,
            "dip_azimuth": cfg.fault_dip_azimuth,
            "down_dip": d_f.tolist(),
            "slip": truth.slip,
            "slip_vector": truth.slip_vector.tolist(),
            "throw": float(-truth.slip_vector[2]),
            "heave": float(np.hypot(truth.slip_vector[0], truth.slip_vector[1])),
            "vertical_separation_SST_SLT": float(
                truth.contact_z(cfg.fault_point[0], cfg.fault_point[1], "SST/SLT", "FB1")
                - truth.contact_z(cfg.fault_point[0], cfg.fault_point[1], "SST/SLT", "FB2")
            ),
        },
        "ore_lens": {
            "implicit": "see bakumine.synth.solid (LensSolid); config.lens holds the parameters",
            "centre": truth.lens.centre.tolist(),
            "axes": truth.lens.axes.tolist(),
            "volume_m3": truth.lens.volume(),
            "min_cover_below_topography_m": lens_cover,
        },
        "volumes": {
            "method": (
                "lattice count in the box, one point per cell of the spacing; level k is "
                "shifted in plan by ((r + 0.5) / 31, ((12 r) mod 31 + 0.5) / 31) cells, "
                "r = k mod 31; AIR is the part of the box above the topography"
            ),
            "spacings_m": list(cfg.volume_spacings),
            "units": {
                unit: {
                    "volume_m3": fine[unit],
                    "volume_coarse_m3": coarse[unit],
                    "relative_difference": difference.get(unit),
                }
                for unit in LITH
            },
            "below_topography_m3": float(sum(v for u, v in fine.items() if u != "AIR")),
        },
        "blocks": {
            "size": list(cfg.block_size),
            "origin": [cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]],
            "shape": [
                round((hi - lo) / s)
                for (lo, hi), s in zip(
                    (cfg.x_range, cfg.y_range, cfg.z_range), cfg.block_size, strict=True
                )
            ],
            "lattice": list(cfg.block_lattice),
            "order": "row = i + nx * (j + ny * k); x, y, z are block centroids",
        },
        "drilling": {
            "holes": n_holes,
            "vertical_holes": int(np.sum(holes.dip == -90.0)),
            "metres": float(holes.length.sum()),
            "geology_intervals": int(len(tables["geology"]["lith"])),
            "assay_intervals": int(len(tables["assay"]["au"])),
            "holes_per_logged_unit": holes_per_unit,
            "holes_crossing_F1": int(fault_holes),
            "true_contacts": {
                layer: int(sum(len(changes[layer][1][h][0]) for h in range(n_holes)))
                for layer in TRUTH_LAYERS
            },
        },
        "logging": {
            "spellings": {unit: list(s) for unit, s in SPELLINGS.items()},
            "soil_logged_when_h_mod_5_in": list(SOIL_LOGGED),
            "oxide_logged_when_h_mod_5_in": list(OXIDE_LOGGED),
            "fault_zone_not_logged_when_h_mod_10_is": FAULT_NOT_LOGGED,
            "order": "rock under the cover, soil, OZ, fault zone, core loss; a later one wins",
            "hole_index": "h = j n + i from the south-west corner; hole id L{h+1:03d}",
        },
        "self_checks": {
            "min_holes_per_unit": minimum,
            "min_fault_holes": cfg.min_fault_holes,
            "min_lens_cover_m": cfg.min_lens_cover,
            "volume_tolerance": cfg.volume_tolerance,
        },
    }
    return LayeredDeposit(
        config=cfg,
        seed=seed,
        truth=truth,
        tables=tables,
        truth_json=truth_json,
        holes=holes,
        meshes=meshes,
        units=json.loads(json.dumps(UNITS_JSON)),
    )


MESH_COMMENTS = {
    "topo": "BakuMine synthetic topography DTM; normals up",
    "ore_solid": "BakuMine synthetic ore solid (truth); closed, outward normals",
    "truth_fault_F1": "BakuMine synthetic fault F1 (truth); plane clipped to the box, normal up",
    "truth_cover_base": "BakuMine synthetic base of the cover (truth); normals up",
    "truth_oxide_base": "BakuMine synthetic base of the oxide zone (truth); normals up",
    "truth_transition_base": "BakuMine synthetic base of the transition zone (truth); normals up",
}
NPY_PAIRS = ("topo", "ore_solid", *SURFACES)


def _write(deposit: LayeredDeposit, out: Path) -> list[Path]:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def target(name: str) -> Path:
        written.append(out / name)
        return written[-1]

    for name, formats in CSV_FORMATS.items():
        _write_csv(target(f"{name}.csv"), deposit.tables[name], formats)
    target("units.json").write_text(
        json.dumps(deposit.units, indent=2) + "\n", encoding="ascii", newline="\n"
    )
    for name, comment in MESH_COMMENTS.items():
        vertices, faces = deposit.meshes[name]
        _write_obj(target(f"{name}.obj"), vertices, faces, comment)
        if name in NPY_PAIRS:
            np.save(target(f"{name}_vertices.npy"), vertices)
            np.save(target(f"{name}_faces.npy"), faces)
    pd.DataFrame(deposit.tables["truth_blocks"]).to_parquet(
        target("truth_blocks.parquet"), index=False, compression="zstd"
    )
    truth = dict(deposit.truth_json)
    truth["files"] = sorted(p.name for p in written) + ["truth.json"]
    target("truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="ascii", newline="\n"
    )
    return written


def import_layered(folder: str | Path) -> Any:
    """The written CSV files as a ``DrillholeDatabase`` (tables ``assay`` and ``geology``)."""
    from bakumine.core.drillholes.importer import TableSource, import_drillholes

    folder = Path(folder)

    def source(name: str) -> TableSource:
        return TableSource(folder / f"{name}.csv", {c: c for c in CSV_FORMATS[name]})

    return import_drillholes(
        source("collar"),
        source("survey"),
        {"assay": source("assay"), "geology": source("geology")},
    )
