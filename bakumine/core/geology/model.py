"""The geology model: rock units, weathering zones and fault blocks at any point, and its folder.

Contract: geology_contracts.md section 9. Theory: research.md 5.7.2 (fast evaluation) and
5.7.6 (order of the steps).

A model holds one :class:`~bakumine.core.geology.lithology.LithologyStack` per fault block
(the single block ``ALL`` without faults), an optional topography, cover base and
weathering bases, and a regular world-aligned grid. :meth:`GeologyModel.sample_grid`
evaluates every event of every stack once at the grid nodes; labels are then taken from
those node values by interpolation on the Kuhn tetrahedra of the cells, which is what
makes coding millions of points affordable. ``exact=True`` evaluates the fields directly.

Layers and their categories:

``lith``         the units of the stacks in dictionary order, the cover unit (if any), ``AIR``
``weathering``   the weathering zones of the dictionary, ``AIR``
``fault_block``  the block names

Parts left to later tasks, marked in the code: the fault layer of the folder and the
meshes across fault blocks (P9.T13), ``spec.json`` and the surface side files (P9.T14).
"""

from __future__ import annotations

import json
import shutil
import time
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.spatial import cKDTree

from bakumine import __version__
from bakumine.core.geology.fields_io import load_field, load_field_inputs, save_field
from bakumine.core.geology.lithology import (
    LithologyStack,
    event_from_manifest,
    tetra_interpolate,
    unit_meshes,
)
from bakumine.core.geology.provenance import Provenance, ProvenanceEntry
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.isosurface import GridSpec
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io.obj import read_obj, write_obj

if TYPE_CHECKING:  # the fault network is P9.T8 / P9.T13
    from bakumine.core.geology.faults import FaultNetwork

__all__ = ["AIR", "FORMAT", "LAYERS", "VERSION", "GeologyModel"]

LAYERS = ("lith", "weathering", "fault_block")
AIR = "AIR"
FORMAT = "bakumine.geology"
VERSION = 1
MANIFEST = "geology.json"
_TASK = "P9.T7"
_CHUNK = 200_000
#: what ``save(overwrite=True)`` replaces; ``spec.json`` and ``coding/`` are left alone
_OWNED = (
    MANIFEST, "units.json", "provenance.json", "diagnostics.json", "topography.obj",
    "grid", "fields", "surfaces", "faults", "meshes",
)  # fmt: skip


def _surface_file(name: str) -> str:
    return f"surfaces/{name.replace(':', '_')}.obj"


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serialisable: {type(value).__name__}")


def _write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, indent=2, default=_json_default) + "\n", encoding="utf-8")


def _as_dtm(surface: DTM | Mesh | None) -> DTM | None:
    return DTM(surface) if isinstance(surface, Mesh) else surface


class GeologyModel:
    """Rock unit, weathering zone and fault block of any point. See the module text."""

    def __init__(
        self,
        *,
        name: str,
        units: UnitDictionary,
        grid: GridSpec,
        topography: DTM | None,
        stacks: Mapping[str, LithologyStack],
        network: FaultNetwork | None = None,
        cover: tuple[str, DTM] | None = None,
        weathering: Sequence[tuple[str, DTM | None]] = (),
        hole_points: np.ndarray | None = None,
        max_extrapolation: float | None = None,
        provenance: Provenance | None = None,
        diagnostics: dict | None = None,
    ) -> None:
        if grid.rotation is not None:
            raise ValueError("the geology grid must be world-aligned (rotation None)")
        if min(grid.shape) < 2:
            raise ValueError("the geology grid needs at least 2 nodes along every axis")
        self.name = str(name)
        self.units = units
        self.grid = grid
        self.topography = _as_dtm(topography)
        self.network = network
        self.stacks = dict(stacks)
        self.blocks: tuple[str, ...] = ("ALL",) if network is None else tuple(network.blocks)
        missing = [b for b in self.blocks if b not in self.stacks]
        if missing:
            raise ValueError(f"no lithology stack for block(s) {', '.join(missing)}")
        extra = [b for b in self.stacks if b not in self.blocks]
        if extra:
            raise ValueError(f"stack(s) for unknown block(s) {', '.join(extra)}")
        self.pooled_blocks: list[str] = []

        modelled = {c for stack in self.stacks.values() for c in stack.categories}
        known = units.codes()
        unknown = sorted(modelled - set(known))
        if unknown:
            raise ValueError(f"unit(s) not in the dictionary: {', '.join(unknown)}")
        lith = [code for code in known if code in modelled]
        self.cover: tuple[str, DTM] | None = None
        if cover is not None:
            code, base = cover
            if code not in known:
                raise ValueError(f"cover unit {code!r} is not in the dictionary")
            if code in modelled:
                raise ValueError(f"cover unit {code!r} is also modelled as a rock unit")
            self.cover = (str(code), _as_dtm(base))
            lith.append(str(code))
        zones = tuple(zone.code for zone in units.weathering_zones)
        self.weathering: tuple[tuple[str, DTM | None], ...] = tuple(
            (str(zone), _as_dtm(base)) for zone, base in weathering
        )
        for i, (zone, base) in enumerate(self.weathering):
            if zone not in zones:
                raise ValueError(f"weathering zone {zone!r} is not in the dictionary")
            if (base is None) != (i == len(self.weathering) - 1):
                raise ValueError(
                    "every weathering zone but the last (deepest) needs a base surface"
                )
        self._categories = {
            "lith": (*lith, AIR),
            "weathering": (*zones, AIR),
            "fault_block": self.blocks,
        }

        self.hole_points = None
        if hole_points is not None:
            self.hole_points = np.asarray(hole_points, dtype=np.float64).reshape(-1, 3)
        if max_extrapolation is not None:
            if not (np.isfinite(max_extrapolation) and max_extrapolation > 0.0):
                raise ValueError("max_extrapolation must be > 0")
            if self.hole_points is None or len(self.hole_points) == 0:
                raise ValueError("max_extrapolation needs the hole points")
        self.max_extrapolation = None if max_extrapolation is None else float(max_extrapolation)
        self.provenance = provenance
        self.diagnostics: dict = {} if diagnostics is None else dict(diagnostics)
        self._hole_tree: cKDTree | None = None
        self._grid_values: dict[tuple[str, int], np.ndarray] = {}

    # ------------------------------------------------------------------ categories

    def categories(self, layer: str) -> tuple[str, ...]:
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
        return self._categories[layer]

    def decode(self, codes, layer: str) -> np.ndarray:
        """Category names of integer labels (object array); None for -1."""
        names = np.array([*self.categories(layer), None], dtype=object)
        index = np.asarray(codes, dtype=np.int64)
        if ((index < -1) | (index >= len(names) - 1)).any():
            raise ValueError(f"labels outside the categories of layer {layer!r}")
        return names[index]

    # ------------------------------------------------------------------ geometry helpers

    def distance_to_data(self, points) -> np.ndarray:
        """Distance (3-D, metres) from each point to the nearest logged hole point."""
        if self.hole_points is None or len(self.hole_points) == 0:
            raise ValueError("the model has no hole points")
        if self._hole_tree is None:
            self._hole_tree = cKDTree(self.hole_points)
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        distance, _ = self._hole_tree.query(pts, workers=-1)
        return distance

    def _block_index(self, pts: np.ndarray) -> np.ndarray:
        if self.network is None:
            return np.zeros(len(pts), dtype=np.int16)
        return np.asarray(self.network.block_index(pts)).astype(np.int16)

    def _in_grid(self, pts: np.ndarray) -> np.ndarray:
        u = (pts - self.grid.origin) / self.grid.spacing
        with np.errstate(invalid="ignore"):
            return (np.isfinite(u) & (u >= 0.0) & (u <= np.array(self.grid.shape) - 1)).all(axis=1)

    # ------------------------------------------------------------------ the grid

    @property
    def sampled(self) -> bool:
        return all(
            (block, e) in self._grid_values
            for block, stack in self.stacks.items()
            for e in range(len(stack.events))
        )

    def sample_grid(self) -> None:
        """Node values of every event of every stack, float32 ``(k, nx, ny, nz)``."""
        t0 = time.perf_counter()
        nx, ny, nz = self.grid.shape
        slab = max(1, _CHUNK // (ny * nz))
        for block, stack in self.stacks.items():
            for e, event in enumerate(stack.events):
                out: np.ndarray | None = None
                for i0 in range(0, nx, slab):
                    i1 = min(i0 + slab, nx)
                    pts = self.grid.slab_points(i0, i1).reshape(-1, 3)
                    values = np.asarray(event.node_values(pts))
                    if out is None:
                        out = np.empty((values.shape[1], nx, ny, nz), dtype=np.float32)
                    out[:, i0:i1] = values.T.reshape(-1, i1 - i0, ny, nz)
                self._grid_values[(block, e)] = out
        self.diagnostics.setdefault("timing", {})["sample_grid_seconds"] = time.perf_counter() - t0
        self.diagnostics["grid_nodes"] = self.grid.n_nodes

    def grid_values(self, block: str, event: int) -> np.ndarray:
        """The sampled node values of one event (sampling the grid first if needed)."""
        if not self.sampled:
            self.sample_grid()
        return self._grid_values[(block, event)]

    # ------------------------------------------------------------------ labels

    def _stack_labels(self, block: str, pts: np.ndarray, exact: bool) -> np.ndarray:
        stack = self.stacks[block]
        if not stack.events:
            return np.full(len(pts), -1, dtype=np.int16)
        if exact:
            return stack.labels(pts)
        inside = self._in_grid(pts)
        out = np.full(len(pts), -1, dtype=np.int16)
        if inside.any():
            values = [
                tetra_interpolate(self.grid_values(block, e), self.grid, pts[inside])
                for e in range(len(stack.events))
            ]
            out[inside] = stack.labels_from_values(values)
        if not inside.all():
            finite = ~inside & np.isfinite(pts).all(axis=1)
            if finite.any():
                out[finite] = stack.labels(pts[finite])
        return out

    def _lith(self, pts: np.ndarray, exact: bool) -> np.ndarray:
        categories = self._categories["lith"]
        index = {c: i for i, c in enumerate(categories)}
        out = np.full(len(pts), -1, dtype=np.int16)
        block_index = self._block_index(pts)
        for b, block in enumerate(self.blocks):
            sel = block_index == b
            if not sel.any():
                continue
            stack = self.stacks[block]
            local = self._stack_labels(block, pts[sel], exact).astype(np.int64)
            lookup = np.array([index[c] for c in stack.categories] + [-1], dtype=np.int16)
            out[sel] = lookup[local]
        if self.max_extrapolation is not None:
            out[self.distance_to_data(pts) > self.max_extrapolation] = -1
        height = None if self.topography is None else self.topography.height_above(pts)
        if self.cover is not None:
            code, base = self.cover
            with np.errstate(invalid="ignore"):
                above = base.height_above(pts) > 0.0
            if height is not None:
                above &= np.isfinite(height)  # no cover outside the topography's plan extent
            out[above] = index[code]
        if height is not None:
            with np.errstate(invalid="ignore"):
                out[height > 0.0] = index[AIR]
        return out

    def _weathering(self, pts: np.ndarray) -> np.ndarray:
        categories = self._categories["weathering"]
        index = {c: i for i, c in enumerate(categories)}
        out = np.full(len(pts), -1, dtype=np.int16)
        if self.weathering:
            out[:] = index[self.weathering[-1][0]]
            for zone, base in reversed(self.weathering[:-1]):
                with np.errstate(invalid="ignore"):
                    out[base.height_above(pts) > 0.0] = index[zone]
        if self.topography is not None:
            height = self.topography.height_above(pts)
            with np.errstate(invalid="ignore"):
                out[height > 0.0] = index[AIR]
            out[~np.isfinite(height)] = -1
        out[~np.isfinite(pts).all(axis=1)] = -1
        return out

    def labels(self, points, layer: str = "lith", *, exact: bool = False) -> np.ndarray:
        """Labels at world points ``(n, 3)``: int16 index into ``categories(layer)``, -1 missing.

        ``exact=False`` interpolates the sampled grid (sampling it on first use); points
        outside the grid are evaluated exactly. ``exact=True`` evaluates the fields.
        """
        if layer not in LAYERS:
            raise ValueError(f"layer must be one of {LAYERS}, got {layer!r}")
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
        if layer == "lith":
            return self._lith(pts, exact)
        if layer == "weathering":
            return self._weathering(pts)
        out = self._block_index(pts)
        out[~np.isfinite(pts).all(axis=1)] = -1
        return out

    # ------------------------------------------------------------------ meshes

    def _stack_margins(self, block: str) -> tuple[tuple[str, ...], np.ndarray]:
        """A margin per category of the block's stack at the grid nodes, after the overprints.

        An event's margins ``m_u`` (positive inside its unit ``u``) overprint what came
        before where the event applies, i.e. where ``a = max_u m_u > 0``:
        ``G_c <- min(G_c, -a)`` for the earlier categories and ``G_u <- max(G_u, m_u)``.
        """
        stack = self.stacks[block]
        index = {c: i for i, c in enumerate(stack.categories)}
        combined = np.full((len(stack.categories), *self.grid.shape), -np.inf)
        for e, event in enumerate(stack.events):
            values = np.asarray(self.grid_values(block, e), dtype=np.float64)
            convert = getattr(event, "unit_margins", None)
            own = np.asarray(convert(values), dtype=np.float64) if convert else values
            if own.shape[0] != len(event.units):
                raise ValueError(
                    f"event {e} of block {block} ({event.mode}) gives no margin per unit: "
                    "it needs a unit_margins method"
                )
            applies = np.fmax.reduce(own, axis=0)
            applies = np.where(np.isnan(applies), -np.inf, applies)
            np.minimum(combined, -applies[None], out=combined)
            for u, unit in enumerate(event.units):
                i = index[str(unit)]
                combined[i] = np.fmax(combined[i], own[u])
        return stack.categories, combined

    def meshes(self, layer: str = "lith") -> dict[str, Mesh]:
        """Closed display mesh per rock unit, from the sampled grid.

        Nothing is kept above the topography, above the cover base or beyond
        ``max_extrapolation``; the meshes are capped flat at the grid boundary. Units
        without volume are left out. The cover itself has no mesh: it is usually thinner
        than a grid cell, which a grid isosurface cannot represent (a 4 m cover on a 5 m
        grid comes out at half its volume); its base surface is in ``surfaces/``.
        """
        if layer != "lith":
            raise ValueError(
                "meshes exist for the layer 'lith' only (the other layers are surfaces)"
            )
        if self.network is not None:
            raise NotImplementedError("unit meshes across fault blocks are part of P9.T13")
        categories, node_margins = self._stack_margins(self.blocks[0])
        nodes = self.grid.slab_points(0, self.grid.shape[0]).reshape(-1, 3)
        clip = np.full(len(nodes), np.inf)
        covered = np.ones(len(nodes), dtype=bool)
        if self.topography is not None:
            depth = -self.topography.height_above(nodes)
            covered = np.isfinite(depth)
            clip = np.where(covered, depth, clip)
        if self.cover is not None:
            below_base = -self.cover[1].height_above(nodes)
            clip = np.where(covered & np.isfinite(below_base), np.minimum(clip, below_base), clip)
        if self.max_extrapolation is not None:
            clip = np.minimum(clip, self.max_extrapolation - self.distance_to_data(nodes))
        out = unit_meshes(node_margins, self.grid, categories, clip=clip.reshape(self.grid.shape))
        return {c: out[c] for c in self._categories["lith"] if c in out}

    # ------------------------------------------------------------------ provenance

    def _extrapolation_entry(self) -> ProvenanceEntry | None:
        if self.hole_points is None or len(self.hole_points) == 0:
            return None
        if self.max_extrapolation is not None:
            limit, rule = self.max_extrapolation, "max_extrapolation"
        else:
            spacings = [
                event.hole_spacing
                for stack in self.stacks.values()
                for event in stack.events
                if getattr(event, "hole_spacing", None) is not None
            ]
            if not spacings:
                return None
            limit, rule = 2.0 * float(np.median(spacings)), "2 x the median hole spacing"
        nodes = self.grid.slab_points(0, self.grid.shape[0]).reshape(-1, 3)
        ground = np.ones(len(nodes), dtype=bool)
        if self.topography is not None:
            with np.errstate(invalid="ignore"):
                ground = ~(self.topography.height_above(nodes) > 0.0)
        if not ground.any():
            return None
        share = float((self.distance_to_data(nodes[ground]) > limit).mean())
        left = "left without a rock unit" if self.max_extrapolation is not None else "extrapolated"
        return ProvenanceEntry(
            item="Rock units beyond the drilling",
            kind="assumed",
            source=f"farther than {limit:.0f} m ({rule}) from the nearest logged hole",
            used_for="rock units away from the holes",
            detail=f"{100.0 * share:.1f} % of the grid nodes below the topography, {left}",
            stage="lithology",
            task=_TASK,
            value=share,
        )

    def provenance_entries(self) -> list[ProvenanceEntry]:
        """The entries of the stacks' events plus the extrapolated volume share."""
        entries: list[ProvenanceEntry] = []
        seen: set[int] = set()
        for stack in self.stacks.values():  # a pooled stack serves several blocks: list it once
            if id(stack) not in seen:
                seen.add(id(stack))
                entries.extend(stack.provenance)
        extrapolated = self._extrapolation_entry()
        if extrapolated is not None:
            entries.append(extrapolated)
        return entries

    # ------------------------------------------------------------------ folder

    def save(self, folder, *, overwrite: bool = False) -> Path:
        """Write the model folder (section 9). ``overwrite`` replaces an existing geology
        folder (``spec.json`` and ``coding/`` are kept); any other folder is refused."""
        if self.network is not None:
            raise NotImplementedError("saving the fault layer is part of P9.T13")
        folder = Path(folder)
        if not self.sampled:
            self.sample_grid()
        # values may be memory-mapped from this very folder: read them before removing files
        grids = {key: np.array(value) for key, value in self._grid_values.items()}
        stored: dict[tuple[str, int], dict] = {}
        for block, stack in self.stacks.items():
            for e, event in enumerate(stack.events):
                files = getattr(event, "field_files", None)
                if files is None:
                    raise ValueError(f"event {e} of block {block} ({event.mode}) cannot be stored")
                stored[(block, e)] = files()
        meshes = self.meshes("lith")

        if folder.exists():
            if not folder.is_dir():
                raise FileExistsError(f"{folder} exists and is not a folder")
            content = list(folder.iterdir())
            if content and not overwrite:
                raise FileExistsError(f"{folder} exists; pass overwrite=True to replace it")
            if content and not (folder / MANIFEST).is_file():
                raise FileExistsError(f"{folder} is not a geology folder: refusing to replace it")
            for name in _OWNED:
                target = folder / name
                if target.is_dir():
                    shutil.rmtree(target)
                elif target.exists():
                    target.unlink()
        for sub in ("grid", "fields", "meshes/lith"):
            (folder / sub).mkdir(parents=True, exist_ok=True)

        manifest: dict[str, Any] = {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "bakumine_version": __version__,
            "created": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "coordinates": "x = east, y = north, z = up, metres",
            "grid": {
                "origin": self.grid.origin.tolist(),
                "spacing": self.grid.spacing.tolist(),
                "shape": list(self.grid.shape),
            },
            "topography": None,
            "layers": {layer: list(self._categories[layer]) for layer in LAYERS},
            "blocks": list(self.blocks),
            "pooled_blocks": list(self.pooled_blocks),
            "stacks": {},
            "faults": [],  # P9.T13
            "fault_network": None,  # P9.T13
            "cover": None,
            "weathering": [],
            "max_extrapolation": self.max_extrapolation,
            "hole_points": None,
            "meshes": {"lith": {}},
        }
        self.units.save(folder / "units.json")
        if self.topography is not None:
            write_obj(folder / "topography.obj", self.topography.mesh, decimals=None)
            manifest["topography"] = "topography.obj"
        for block, stack in self.stacks.items():
            entries = []
            for e, event in enumerate(stack.events):
                values_file = f"grid/values_{block}_{e}.npy"
                np.save(folder / values_file, grids[(block, e)])
                field_files = []
                for suffix, (fitted, xyz, values) in stored[(block, e)].items():
                    field_file = f"fields/{block}_{e}_{suffix}.npz"
                    save_field(folder / field_file, fitted, xyz, values)
                    field_files.append(field_file)
                entry = {
                    "mode": event.mode,
                    "units": list(event.units),
                    "values": values_file,
                    "fields": field_files,
                }
                extra = getattr(event, "to_manifest", None)
                if extra is not None:
                    entry.update(extra())
                entries.append(entry)
            manifest["stacks"][block] = entries
        surfaces = []
        if self.cover is not None:
            code, base = self.cover
            manifest["cover"] = {"unit": code, "surface": _surface_file(f"base:{code}")}
            surfaces.append((manifest["cover"]["surface"], base))
        for zone, base in self.weathering:
            file = None if base is None else _surface_file(f"base:{zone}")
            manifest["weathering"].append({"zone": zone, "base": file})
            if base is not None:
                surfaces.append((file, base))
        if surfaces:
            (folder / "surfaces").mkdir(exist_ok=True)
            for file, base in surfaces:
                write_obj(folder / file, base.mesh, decimals=None)
        if self.hole_points is not None:
            np.save(folder / "grid/hole_points.npy", self.hole_points)
            manifest["hole_points"] = "grid/hole_points.npy"
        for code, mesh in meshes.items():
            file = f"meshes/lith/{code}.obj"
            write_obj(folder / file, mesh, decimals=3)
            manifest["meshes"]["lith"][code] = file

        record = self.provenance
        if record is None:
            record = Provenance(object_kind="geology", name=self.name)
            record.extend(self.provenance_entries())
        record.save(folder / "provenance.json")
        _write_json(folder / "diagnostics.json", self.diagnostics)
        _write_json(folder / MANIFEST, manifest)
        return folder

    @classmethod
    def load(cls, folder) -> GeologyModel:
        """Read a model folder. The grid arrays are opened with ``mmap_mode="r"``."""
        folder = Path(folder)
        path = folder / MANIFEST
        if not path.is_file():
            raise FileNotFoundError(f"{folder} is not a geology folder: no {MANIFEST}")
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("format") != FORMAT:
            raise ValueError(f"{path}: format is {manifest.get('format')!r}, expected {FORMAT!r}")
        if manifest.get("version") != VERSION:
            raise ValueError(f"{path}: version {manifest.get('version')!r} is not supported")
        if manifest.get("faults") or manifest.get("fault_network"):
            raise NotImplementedError("loading the fault layer is part of P9.T13")
        g = manifest["grid"]
        grid = GridSpec(g["origin"], g["spacing"], tuple(g["shape"]))
        topography = None
        if manifest.get("topography"):
            topography = DTM(read_obj(folder / manifest["topography"]))

        stacks: dict[str, LithologyStack] = {}
        grids: dict[tuple[str, int], np.ndarray] = {}
        for block, entries in manifest["stacks"].items():
            events = []
            for e, entry in enumerate(entries):
                prefix = f"{block}_{e}_"
                fields, inputs = {}, {}
                for file in entry["fields"]:
                    stem = Path(file).stem
                    if not stem.startswith(prefix):
                        raise ValueError(
                            f"{path}: field file {file!r} does not belong to {prefix}*"
                        )
                    fields[stem[len(prefix) :]] = load_field(folder / file)
                    inputs[stem[len(prefix) :]] = load_field_inputs(folder / file)
                events.append(event_from_manifest(entry, fields, inputs))
                grids[(block, e)] = np.load(folder / entry["values"], mmap_mode="r")
            stacks[block] = LithologyStack(block, tuple(events))

        cover = None
        if manifest.get("cover"):
            cover = (
                manifest["cover"]["unit"],
                DTM(read_obj(folder / manifest["cover"]["surface"])),
            )
        weathering = [
            (w["zone"], None if w["base"] is None else DTM(read_obj(folder / w["base"])))
            for w in manifest.get("weathering", [])
        ]
        hole_points = None
        if manifest.get("hole_points"):
            hole_points = np.load(folder / manifest["hole_points"])
        provenance = None
        if (folder / "provenance.json").is_file():
            provenance = Provenance.load(folder / "provenance.json")
        diagnostics = None
        if (folder / "diagnostics.json").is_file():
            diagnostics = json.loads((folder / "diagnostics.json").read_text(encoding="utf-8"))
        model = cls(
            name=manifest["name"],
            units=UnitDictionary.load(folder / "units.json"),
            grid=grid,
            topography=topography,
            stacks=stacks,
            cover=cover,
            weathering=weathering,
            hole_points=hole_points,
            max_extrapolation=manifest.get("max_extrapolation"),
            provenance=provenance,
            diagnostics=diagnostics,
        )
        model.pooled_blocks = list(manifest.get("pooled_blocks", []))
        if tuple(manifest["layers"]["lith"]) != model.categories("lith"):
            raise ValueError(f"{path}: the lith categories do not match units.json and the stacks")
        for key, values in grids.items():
            if tuple(values.shape[1:]) != grid.shape:
                raise ValueError(f"{path}: grid values of {key} are not on the grid")
            model._grid_values[key] = values
        return model
