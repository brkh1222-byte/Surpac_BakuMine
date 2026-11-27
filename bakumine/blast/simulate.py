"""Design + site -> result bundle (plan_blast.md section 3, ``bakumine-blast-result`` v1).

This module only **orchestrates** the finished models; it contains no blast physics:

1. polygon (from the request, or the demo bench when there is no ground);
2. pattern: :func:`bakumine.blast.design.suggest` (Konya) for what the request leaves
   open, :func:`bakumine.blast.design.layout` on the bench top;
3. timing (:mod:`bakumine.blast.timing`);
4. fragmentation (Kuz-Ram), ground vibration and air overpressure (scaled distance),
   flyrock (Richards-Moore), wave sources, near-field damage of blocks
   (Holmberg-Persson), back-break, throw and muckpile (ballistic particles).

Site
----
``ground`` is the surface before the blast (a level, a DTM, any object with ``z_at``).
Without one the blast is put on a synthetic bench (:func:`bakumine.blast.bench.demo_site`).
With one, the collars are taken from it and a free face is cut in front of the block
(:func:`bakumine.blast.bench.cut_bench`, a stated visualisation assumption); the bench floor
is the mean ground elevation over the polygon minus the bench height.

Blocks
------
The damage is computed for ``blocks = (centres, sizes)`` of a block model when given
(those that touch the blast box; subsampled above :data:`MAX_BLOCKS`), else for a regular
grid of cubes made here. Blocks whose centre is above the ground after the cut are air and
are left out.

The bundle holds plain Python lists, floats, ints, bools and None only; a value that is not
finite is written as None, so ``json.dumps(bundle, allow_nan=False)`` works.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

import numpy as np

from bakumine.blast import bench, damage, design, flyrock, fragmentation, throw, timing, waves
from bakumine.blast import vibration as vib
from bakumine.blast.model import ANFO, EMULSION, BlastDesign, Explosive, Monitor, Rock

__all__ = [
    "AIRBLAST_RADII",
    "FORMAT",
    "MAX_BLOCKS",
    "VERSION",
    "SimulationRequest",
    "defaults",
    "plain",
    "simulate",
    "validate_bundle",
]

FORMAT = "bakumine-blast-result"
VERSION = 1

#: Largest number of blocks in ``damage.blocks`` (the viewer draws every one).
MAX_BLOCKS = 6000
#: Largest number of holes a request may lay out.
MAX_HOLES = 3000
MAX_PARTICLES = 50000
#: Largest number of nodes along a side of the ``ppv_surface`` grid.
MAX_GRID_NODES = 150
#: Radii of the air overpressure rings, metres from the nearest collar.
AIRBLAST_RADII = (50.0, 100.0, 200.0, 500.0, 1000.0)
WINDOW_MS = 8.0

EXPLOSIVES: dict[str, Explosive] = {"anfo": ANFO, "emulsion": EMULSION}
_TIMING_TYPES = ("row_by_row", "echelon", "v_cut", "hole_by_hole")
_ROCK_FIELDS = tuple(f.name for f in fields(Rock))
_DEFAULT_TIMING = {"type": "row_by_row", "inter_hole_ms": 17.0, "inter_row_ms": 42.0}
_DEFAULT_ROCK = {
    "density": 2700.0,
    "ucs_mpa": 100.0,
    "tensile_mpa": 10.0,
    "youngs_gpa": 50.0,
    "poisson": 0.25,
    "rock_factor": None,
}


# ------------------------------------------------------------------------------ request


def _number(name: str, value: Any, *, positive: bool = False, optional: bool = False) -> Any:
    if value is None:
        if optional:
            return None
        raise ValueError(f"{name} is required")
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be a number, got {value!r}")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if positive and not value > 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _mapping(name: str, value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object, got {type(value).__name__}")
    return dict(value)


@dataclass
class SimulationRequest:
    """What ``POST /api/v1/blast/simulate`` and ``bakumine blast simulate`` take.

    Lengths in metres (``diameter`` too), delays in ms. ``None`` for ``burden``,
    ``spacing``, ``stemming`` or ``subdrill`` means "from the rules of thumb"
    (:func:`bakumine.blast.design.suggest`, method ``konya``; stemming 0.7 B, subdrill
    0.3 B). ``polygon=None`` means the demo bench, or a rectangle at the centre of the
    topography when there is one. ``site`` holds site constants:
    ``{"vibration": {"k", "beta"}, "flyrock_k", "hp": {"k", "alpha", "beta"},
    "airblast": {"k", "a"}}`` (``calibrated: true`` inside a group marks constants fitted
    to the site). Keys this version does not know are ignored.
    """

    name: str = "Blast"
    polygon: list[list[float]] | None = None
    bench_height: float = 10.0
    diameter: float = 0.165
    burden: float | None = None
    spacing: float | None = None
    pattern: str = "staggered"
    face_azimuth: float = 180.0
    explosive: str = "ANFO"
    stemming: float | None = None
    subdrill: float | None = None
    timing: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_TIMING))
    rock: dict[str, Any] = field(default_factory=lambda: dict(_DEFAULT_ROCK))
    monitors: list[dict[str, Any]] = field(default_factory=list)
    particles: int = 3000
    seed: int = 0
    grid_spacing: float | None = None
    oversize_m: float = 1.0
    site: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty text")
        if self.polygon is not None:
            try:
                poly = np.asarray(self.polygon, dtype=np.float64)
            except (TypeError, ValueError):
                raise ValueError("polygon must be a list of [x, y] vertices or null") from None
            if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
                raise ValueError("polygon needs at least three [x, y] vertices")
            if not np.isfinite(poly).all():
                raise ValueError("polygon has non-finite coordinates")
            self.polygon = [[float(p[0]), float(p[1])] for p in poly]
        self.bench_height = _number("bench_height", self.bench_height, positive=True)
        self.diameter = _number("diameter", self.diameter, positive=True)
        if self.diameter > 1.0:
            raise ValueError(
                f"diameter is in metres (e.g. 0.165), got {self.diameter:g}: is it millimetres?"
            )
        self.burden = _number("burden", self.burden, positive=True, optional=True)
        self.spacing = _number("spacing", self.spacing, positive=True, optional=True)
        self.stemming = _number("stemming", self.stemming, optional=True)
        self.subdrill = _number("subdrill", self.subdrill, optional=True)
        for key in ("stemming", "subdrill"):
            value = getattr(self, key)
            if value is not None and value < 0:
                raise ValueError(f"{key} must be >= 0, got {value!r}")
        if self.pattern not in ("staggered", "square"):
            raise ValueError(f"pattern must be 'staggered' or 'square', got {self.pattern!r}")
        self.face_azimuth = _number("face_azimuth", self.face_azimuth)
        if not isinstance(self.explosive, str) or self.explosive.lower() not in EXPLOSIVES:
            raise ValueError(f"explosive must be 'ANFO' or 'Emulsion', got {self.explosive!r}")
        self.explosive = EXPLOSIVES[self.explosive.lower()].name

        timing_in = {**_DEFAULT_TIMING, **_mapping("timing", self.timing)}
        if timing_in["type"] not in _TIMING_TYPES:
            raise ValueError(
                f"timing.type must be one of {_TIMING_TYPES}, got {timing_in['type']!r}"
            )
        self.timing = {
            "type": timing_in["type"],
            "inter_hole_ms": _number("timing.inter_hole_ms", timing_in["inter_hole_ms"]),
            "inter_row_ms": _number("timing.inter_row_ms", timing_in["inter_row_ms"]),
        }

        rock_in = {**_DEFAULT_ROCK, **_mapping("rock", self.rock)}
        self.rock = {k: v for k, v in rock_in.items() if k in _ROCK_FIELDS}
        self.to_rock()  # validates

        if not isinstance(self.monitors, list):
            raise ValueError("monitors must be a list")
        monitors = []
        for i, m in enumerate(self.monitors):
            what = f"monitors[{i}]"
            m = _mapping(what, m)
            if not m:
                raise ValueError(f"{what} must be an object with name and xyz")
            xyz = m.get("xyz")
            if not isinstance(xyz, list | tuple) or len(xyz) != 3:
                raise ValueError(f"{what}.xyz must be [x, y, z]")
            monitors.append(
                {
                    "name": str(m.get("name") or f"Monitor {i + 1}"),
                    "xyz": [_number(f"{what}.xyz", v) for v in xyz],
                    "limit_mm_s": _number(
                        f"{what}.limit_mm_s", m.get("limit_mm_s"), positive=True, optional=True
                    ),
                    "airblast_limit_dbl": _number(
                        f"{what}.airblast_limit_dbl",
                        m.get("airblast_limit_dbl"),
                        positive=True,
                        optional=True,
                    ),
                }
            )
        self.monitors = monitors

        for key in ("particles", "seed"):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
                raise ValueError(f"{key} must be a whole number, got {value!r}")
            setattr(self, key, int(value))
        if not 1 <= self.particles <= MAX_PARTICLES:
            raise ValueError(f"particles must be between 1 and {MAX_PARTICLES}")
        if self.seed < 0:
            raise ValueError("seed must be >= 0")
        self.grid_spacing = _number("grid_spacing", self.grid_spacing, positive=True, optional=True)
        self.oversize_m = _number("oversize_m", self.oversize_m, positive=True)
        self.site = _mapping("site", self.site)
        self.site_constants()  # validates

    # ---- conversions
    def to_rock(self) -> Rock:
        try:
            return Rock(**self.rock)
        except TypeError as exc:
            raise ValueError(f"rock: {exc}") from None

    def to_explosive(self) -> Explosive:
        return EXPLOSIVES[self.explosive.lower()]

    def site_constants(self) -> dict[str, Any]:
        """The site constants as model objects: ``law`` (vibration), ``hp``
        (Holmberg-Persson), ``flyrock_k``, ``flyrock_calibrated``, ``ka_kpa``, ``a``,
        ``airblast_calibrated``. Defaults are the published uncalibrated values."""
        site = self.site
        v = _mapping("site.vibration", site.get("vibration"))
        law = vib.AS2187_AVERAGE
        if "k" in v or "beta" in v:
            calibrated = bool(v.get("calibrated", False))
            law = vib.SiteLaw(
                k=_number("site.vibration.k", v.get("k", law.k), positive=True),
                beta=_number("site.vibration.beta", v.get("beta", law.beta), positive=True),
                calibrated=calibrated,
                source="site calibration" if calibrated else "given in the request (uncalibrated)",
            )
        h = _mapping("site.hp", site.get("hp"))
        base = damage.HPConstants()
        hp = damage.HPConstants(
            k=_number("site.hp.k", h.get("k", base.k), positive=True),
            alpha=_number("site.hp.alpha", h.get("alpha", base.alpha), positive=True),
            beta=_number("site.hp.beta", h.get("beta", base.beta), positive=True),
            calibrated=bool(h.get("calibrated", False)),
        )
        a = _mapping("site.airblast", site.get("airblast"))
        flyrock_k = site.get("flyrock_k")
        return {
            "law": law,
            "hp": hp,
            "flyrock_k": _number(
                "site.flyrock_k", flyrock.K_HARD if flyrock_k is None else flyrock_k, positive=True
            ),
            "flyrock_calibrated": bool(site.get("flyrock_calibrated", False)),
            "ka_kpa": _number(
                "site.airblast.k", a.get("k", vib.AIRBLAST_KA_CONFINED_HIGH), positive=True
            ),
            "a": _number("site.airblast.a", a.get("a", vib.AIRBLAST_EXPONENT), positive=True),
            "airblast_calibrated": bool(a.get("calibrated", False)),
        }

    # ---- I/O
    @classmethod
    def from_dict(cls, data: Any) -> SimulationRequest:
        if not isinstance(data, dict):
            raise ValueError("the request must be a JSON object")
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def defaults(top: Any = None) -> dict[str, Any]:
    """A request with sensible defaults, as a dict.

    ``top`` is the topography of the project (an object with ``bounds`` and ``z_at``) or
    ``None``. With one, the polygon is a 60 x 35 m rectangle at its centre
    (:func:`bakumine.blast.bench.default_polygon`); without, the polygon is ``None`` (the
    demo bench). One monitor is put 300 m behind the block with the AS 2187.2 "sensitive
    site, maximum" limits (10 mm/s, 120 dBL) as an example to edit.
    """
    request = SimulationRequest()
    if top is None:
        site = bench.demo_site(bench_height=request.bench_height, face_azimuth=request.face_azimuth)
        polygon, z = site.polygon, site.top_z
    else:
        polygon = bench.default_polygon(top, face_azimuth=request.face_azimuth)
        request.polygon = polygon
        z = math.nan
    centre = np.mean(np.asarray(polygon), axis=0)
    _, f = design.face_axes(request.face_azimuth)
    xy = centre - 300.0 * f
    if top is not None:
        for point in (xy, centre):
            z = float(bench.ground_z(top, point[0], point[1]))
            if math.isfinite(z):
                break
        else:
            raise ValueError("the topography has no elevation at its centre")
    request.monitors = [
        {
            "name": "Monitor 1",
            "xyz": [float(xy[0]), float(xy[1]), z],
            "limit_mm_s": vib.LIMITS["as2187_sensitive_max"].value,
            "airblast_limit_dbl": vib.AIRBLAST_LIMITS["as2187_sensitive_max"].value,
        }
    ]
    return request.to_dict()


# ------------------------------------------------------------------------------ plain JSON


def plain(value: Any) -> Any:
    """``value`` as plain JSON data: numpy arrays and scalars become lists and Python
    numbers, tuples lists, and every non-finite float ``None``."""
    if isinstance(value, dict):
        return {str(k): plain(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [plain(v) for v in value]
    if isinstance(value, np.ndarray):
        if value.dtype.kind == "f":
            finite = np.isfinite(value)
            if finite.all():
                return value.tolist()
            out = value.astype(object)
            out[~finite] = None
            return out.tolist()
        if value.dtype.kind in "iub":
            return value.tolist()
        return [plain(v) for v in value.tolist()]
    if isinstance(value, np.bool_ | bool):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float | np.floating):
        value = float(value)
        return value if math.isfinite(value) else None
    return value


# ------------------------------------------------------------------------------ steps


def _mean_top(top: Any, polygon: np.ndarray) -> float:
    """Mean elevation of ``top`` over the polygon (a 24 x 24 sample of its bounding box)."""
    lo, hi = polygon.min(axis=0), polygon.max(axis=0)
    gx = lo[0] + (np.arange(24) + 0.5) * (hi[0] - lo[0]) / 24
    gy = lo[1] + (np.arange(24) + 0.5) * (hi[1] - lo[1]) / 24
    xy = np.column_stack([np.repeat(gx, 24), np.tile(gy, 24)])
    xy = np.vstack([xy[damage.points_in_polygon(xy, polygon)], polygon])
    z = bench.ground_z(top, xy[:, 0], xy[:, 1])
    z = z[np.isfinite(z)]
    if z.size == 0:
        raise ValueError("the polygon lies outside the topography: no ground elevation there")
    return float(z.mean())


def _timed(layout: BlastDesign, spec: dict[str, Any]) -> BlastDesign:
    th, tr = spec["inter_hole_ms"], spec["inter_row_ms"]
    kind = spec["type"]
    if kind == "row_by_row":
        return timing.row_by_row(layout, tr, th)
    if kind == "echelon":
        return timing.echelon(layout, th, tr)
    if kind == "v_cut":
        return timing.v_cut(layout, th, tr)
    return timing.hole_by_hole(layout, th, tr)


def _grid(bounds: Any, spacing: float | None) -> tuple[tuple[float, float], float, tuple[int, int]]:
    """Origin, spacing and shape of the ``ppv_surface`` grid: about 2 m, at most
    :data:`MAX_GRID_NODES` nodes a side."""
    (x0, y0), (x1, y1) = bounds
    side = max(x1 - x0, y1 - y0)
    step = max(2.0 if spacing is None else spacing, side / (MAX_GRID_NODES - 1))
    shape = (int(math.floor((x1 - x0) / step)) + 1, int(math.floor((y1 - y0) / step)) + 1)
    return (float(x0), float(y0)), float(step), (max(shape[0], 2), max(shape[1], 2))


def _generated_blocks(box_lo: np.ndarray, box_hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Cubes that fill the box, 2.5 m or larger so that there are fewer than
    :data:`MAX_BLOCKS`; the top layer ends at the top of the box."""
    size = 2.5
    extent = box_hi - box_lo
    while np.prod(np.ceil(extent / size)) >= MAX_BLOCKS:
        size *= 1.1
    n = np.ceil(extent / size).astype(int)
    gx = box_lo[0] + (np.arange(n[0]) + 0.5) * size
    gy = box_lo[1] + (np.arange(n[1]) + 0.5) * size
    gz = box_hi[2] - (np.arange(n[2]) + 0.5) * size
    xx, yy, zz = np.meshgrid(gx, gy, gz, indexing="ij")
    centres = np.column_stack([xx.ravel(), yy.ravel(), zz.ravel()])
    return centres, np.full(centres.shape, size)


def _blocks(
    blocks: Any, laid: BlastDesign, surface: Any, subdrill: float, notes: list[str]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Blocks for the damage model: those of the block model in the blast box, else a
    generated grid. Air (centre above ``surface``) is left out."""
    poly = np.asarray(laid.polygon, dtype=np.float64)
    margin = 1.5 * laid.burden
    top = max(h.collar[2] for h in laid.holes)
    box_lo = np.array([*(poly.min(axis=0) - margin), laid.floor_z - subdrill - 2.0])
    box_hi = np.array([*(poly.max(axis=0) + margin), top])
    info: dict[str, Any] = {"source": "generated grid"}
    centres = sizes = None
    if blocks is not None:
        c = np.asarray(blocks[0], dtype=np.float64).reshape(-1, 3)
        s = np.asarray(blocks[1], dtype=np.float64).reshape(-1, 3)
        if c.shape != s.shape:
            raise ValueError("block centres and sizes must have the same shape")
        touch = np.all((c + 0.5 * s > box_lo) & (c - 0.5 * s < box_hi), axis=1)
        if touch.any():
            centres, sizes = c[touch], s[touch]
            info = {"source": "block model", "in_box": int(touch.sum())}
        else:
            notes.append(
                "The block model has no blocks around the blast: the damage is shown on a "
                "generated grid of blocks instead."
            )
    if centres is None:
        centres, sizes = _generated_blocks(box_lo, box_hi)
    z = bench.ground_z(surface, centres[:, 0], centres[:, 1])
    with np.errstate(invalid="ignore"):
        rock = ~(centres[:, 2] > z)
    if info["source"] == "block model" and not rock.any():
        notes.append(
            "All blocks of the block model around the blast are above the ground: the damage "
            "is shown on a generated grid of blocks instead."
        )
        info = {"source": "generated grid"}
        centres, sizes = _generated_blocks(box_lo, box_hi)
        z = bench.ground_z(surface, centres[:, 0], centres[:, 1])
        with np.errstate(invalid="ignore"):
            rock = ~(centres[:, 2] > z)
    centres, sizes = centres[rock], sizes[rock]
    if len(centres) > MAX_BLOCKS:
        keep = np.unique(np.linspace(0, len(centres) - 1, MAX_BLOCKS).round().astype(np.int64))
        notes.append(
            f"The block model has {len(centres)} blocks around the blast; {len(keep)} evenly "
            "picked ones are shown (the damage of the others is not in this result)."
        )
        info["subsampled_from"] = int(len(centres))
        centres, sizes = centres[keep], sizes[keep]
    info["blocks"] = int(len(centres))
    info["block_size_m"] = [float(v) for v in np.median(sizes, axis=0)] if len(sizes) else None
    return centres, sizes, info


# ------------------------------------------------------------------------------ simulate


def _with_cut(
    muckpile: dict[str, Any], ground: Any, polygon: Any, face_azimuth: float
) -> dict[str, Any]:
    """The muckpile grid with ``cut_m``: how far each node of ``z_before`` lies below the
    natural ``ground`` inside the excavated area of :func:`bakumine.blast.bench.cut_bench`
    (0 on untouched ground, NaN where the ground is unknown). Without a cut (``ground`` is
    None) the grid is returned as it is. The viewer draws cut ground as fresh rock, not as
    the satellite picture of the old surface.
    """
    if ground is None:
        return muckpile
    nx, ny = (int(v) for v in muckpile["shape"])
    step = float(muckpile["spacing"])
    natural = bench.sample_ground(ground, muckpile["origin"], step, (nx, ny)).z
    before = np.asarray(muckpile["z_before"], dtype=np.float64).reshape(nx, ny)
    xx, yy = np.meshgrid(
        muckpile["origin"][0] + step * np.arange(nx),
        muckpile["origin"][1] + step * np.arange(ny),
        indexing="ij",
    )
    front = bench.front_of_polygon(xx, yy, polygon, face_azimuth, 150.0)
    cut = np.where(front, np.maximum(natural - before, 0.0), 0.0)
    cut = np.where(np.isfinite(natural) & np.isfinite(before), cut, np.nan)
    return {**muckpile, "cut_m": cut.ravel()}


def simulate(
    request: SimulationRequest | dict[str, Any],
    ground: float | Any = None,
    blocks: tuple[Any, Any] | None = None,
    top: float | Any = None,
) -> dict[str, Any]:
    """Run every model on a request and return the result bundle.

    Parameters
    ----------
    request : SimulationRequest or dict
    ground : float or object with ``z_at(x, y)``, optional
        Surface before the blast. ``None`` = a synthetic bench (the request's polygon, or
        the demo polygon) with a vertical face and a floor; nothing is cut then.
    blocks : (centres, sizes), optional
        Blocks of a block model, each (n, 3). ``None`` = a generated grid.
    top : float or object with ``z_at(x, y)``, optional
        Bench top the collars are put on. Default: ``ground``; without a ground a number
        is the elevation of the synthetic bench top (default 1000 m).

    Returns
    -------
    dict
        The bundle of plan_blast.md section 3 (plus ``envelope``); see :func:`validate_bundle`.

    Raises
    ------
    ValueError
        The request is not valid or the models cannot be evaluated for it (the message
        says why).
    """
    req = request
    if not isinstance(req, SimulationRequest):
        req = SimulationRequest.from_dict(req)
    rock, explosive = req.to_rock(), req.to_explosive()
    site = req.site_constants()
    notes: list[str] = []

    # ---- site
    if ground is None and top is not None and hasattr(top, "z_at"):
        ground = top
    cut_depth = 0.0
    if ground is None:
        top_z = 1000.0 if top is None else float(top)
        if req.polygon is None:
            demo = bench.demo_site(top_z, req.bench_height, face_azimuth=req.face_azimuth)
        else:
            demo = bench.bench_for_polygon(req.polygon, top_z, req.bench_height, req.face_azimuth)
        polygon = np.asarray(demo.polygon)
        top_surface: Any = demo.top_z
        floor_z = demo.floor_z
        surface: Any = demo.ground
        bounds = demo.bounds
        site_kind = "synthetic bench"
        notes.append(
            "Synthetic bench: flat top, vertical free face and a level floor in front of "
            "the block; not the project's topography."
        )
    else:
        top_surface = ground if top is None else top
        polygon = np.asarray(
            req.polygon
            if req.polygon is not None
            else bench.default_polygon(ground, face_azimuth=req.face_azimuth)
            if hasattr(ground, "bounds")
            else bench.demo_site(face_azimuth=req.face_azimuth).polygon
        )
        floor_z = _mean_top(top_surface, polygon) - req.bench_height
        surface = bench.cut_bench(ground, polygon, floor_z, req.face_azimuth)
        sx, sy = surface.shape
        end = (
            surface.origin[0] + surface.spacing * (sx - 1),
            surface.origin[1] + surface.spacing * (sy - 1),
        )
        bounds = (surface.origin, end)
        before = bench.sample_ground(ground, surface.origin, surface.spacing, surface.shape).z
        lowered = before - surface.z
        cut_depth = float(np.nanmax(lowered)) if np.isfinite(lowered).any() else 0.0
        site_kind = "ground surface"
        if cut_depth > 0.05:
            notes.append(
                "Free face excavated in front of the block: the ground within the block's "
                f"width, up to 150 m in the throw direction, is lowered to the bench floor "
                f"({floor_z:.1f} m; up to {cut_depth:.1f} m of cut). This is an assumption "
                "made to give the blast a face and a floor, not a pit design."
            )
        if req.polygon is None:
            notes.append("No polygon given: a 60 x 35 m block at the centre of the ground is used.")

    # ---- pattern
    suggestion = design.suggest(
        req.diameter, req.bench_height, rock, explosive, req.pattern, method="konya"
    )
    burden = suggestion.burden if req.burden is None else req.burden
    spacing = design.spacing_ratio_for(req.pattern) * burden if req.spacing is None else req.spacing
    area = 0.5 * abs(
        float(
            np.dot(polygon[:, 0], np.roll(polygon[:, 1], -1))
            - np.dot(polygon[:, 1], np.roll(polygon[:, 0], -1))
        )
    )
    if area / (burden * spacing) > MAX_HOLES:
        raise ValueError(
            f"about {area / (burden * spacing):.0f} holes would be needed (burden "
            f"{burden:.2f} m, spacing {spacing:.2f} m on {area:.0f} m2); the limit is {MAX_HOLES}"
        )
    laid = design.layout(
        polygon,
        top_surface,
        floor_z,
        burden,
        spacing,
        req.pattern,
        req.face_azimuth,
        req.diameter,
        explosive,
        stemming=req.stemming,
        subdrill=req.subdrill,
        rock=rock,
        name=req.name,
    )
    laid.monitors = [
        Monitor(m["name"], tuple(m["xyz"]), m["limit_mm_s"], m["airblast_limit_dbl"])
        for m in req.monitors
    ]
    if not any(h.charge_mass > 0 for h in laid.holes):
        raise ValueError(
            "no hole has room for a charge: the stemming is not shorter than the holes "
            "(check stemming, bench height and the ground over the polygon)"
        )
    if any(h.charge_mass > 0 and not h.stemming > 0 for h in laid.holes):
        raise ValueError(
            "stemming must be > 0: the flyrock model has no bound for a hole charged to the collar"
        )
    timed = _timed(laid, req.timing)
    stemming_used = float(timed.notes["layout"]["stemming"])
    subdrill_used = float(timed.notes["layout"]["subdrill"])
    notes.extend(suggestion.warnings if req.burden is None else ())
    stiffness = design.stiffness_warning(timed.bench_height / burden)
    if req.burden is not None and stiffness:
        notes.append(stiffness)
    for key, text in (
        ("uncharged_holes", "hole(s) are no longer than their stemming and carry no charge"),
        ("dropped_holes", "hole position(s) dropped (no ground there, or not above the floor)"),
    ):
        if timed.notes.get(key):
            notes.append(f"{len(timed.notes[key])} {text}.")

    totals = design.summary(timed)
    relief = timing.burden_relief(timed)
    notes.extend(f for f in relief.flags if "within the guide range" not in f)

    # ---- fragmentation
    frag = fragmentation.predict(timed, oversize=req.oversize_m)

    # ---- vibration, air overpressure
    law: vib.SiteLaw = site["law"]
    groups, mic = vib.charge_per_delay(timed, WINDOW_MS)
    cp, cs = waves.wave_speeds(rock)
    sources = waves.sources(timed)
    origin, step, shape = _grid(bounds, req.grid_spacing)
    drape = bench.sample_ground(surface, origin, step, shape)
    gx = origin[0] + step * np.arange(shape[0])
    gy = origin[1] + step * np.arange(shape[1])
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    nodes = np.column_stack([xx.ravel(), yy.ravel(), drape.z.ravel()])
    surface_ppv = vib.ppv_at(nodes, timed, law, mode="mic", window_ms=WINDOW_MS).ppv_mm_s
    arrival = np.full(len(nodes), np.inf)
    for source in sources:
        d = np.linalg.norm(nodes - np.asarray(source["centre"]), axis=1)
        arrival = np.fmin(arrival, source["t_ms"] + d / cp * 1000.0)
    arrival = np.where(np.isfinite(nodes[:, 2]), arrival, np.nan)

    radii = np.asarray(AIRBLAST_RADII)
    ring_pa = vib.airblast_pressure(radii, mic, site["ka_kpa"], site["a"])
    ring_dbl = vib.pressure_to_dbl(ring_pa)
    monitors: list[dict[str, Any]] = []
    if timed.monitors:
        points = np.array([m.xyz for m in timed.monitors], dtype=np.float64)
        at = vib.ppv_at(points, timed, law, mode="mic", window_ms=WINDOW_MS)
        air = vib.airblast_at(points, timed, site["ka_kpa"], site["a"], WINDOW_MS)
        for i, m in enumerate(timed.monitors):
            value = float(at.ppv_mm_s[i])
            if not math.isfinite(value):
                raise ValueError(f"monitor {m.name!r} lies on a charge: no PPV can be given there")
            level = float(air.dbl[i])
            monitors.append(
                {
                    "name": m.name,
                    "xyz": list(m.xyz),
                    "distance_m": float(at.distance[i]),
                    "charge_kg": float(at.charge_kg[i]),
                    "scaled_distance": float(at.scaled_distance[i]),
                    "ppv_mm_s": value,
                    "limit_mm_s": m.limit_mm_s,
                    "ok": None if m.limit_mm_s is None else bool(vib.check(value, m.limit_mm_s)),
                    "airblast_dbl": level,
                    "airblast_limit_dbl": m.airblast_limit_dbl,
                    "airblast_ok": (
                        None
                        if m.airblast_limit_dbl is None
                        else bool(vib.check(level, m.airblast_limit_dbl))
                    ),
                }
            )

    # ---- flyrock
    fly = flyrock.predict(timed, k=site["flyrock_k"], calibrated=site["flyrock_calibrated"])

    # ---- damage
    hp: damage.HPConstants = site["hp"]
    thresholds = damage.thresholds_for(rock)
    centres, sizes, block_info = _blocks(blocks, timed, surface, subdrill_used, notes)
    dmg = damage.block_damage(centres, sizes, timed, hp, thresholds, cp=cp)
    backbreak = damage.backbreak(timed, hp, thresholds)

    # ---- throw and muckpile
    params = throw.ThrowParams()
    thrown = throw.simulate_throw(
        timed,
        surface,
        x50=frag.x50,
        uniformity=frag.n,
        n=req.particles,
        seed=req.seed,
        params=params,
        cp=cp,
    )
    pile = thrown["summary"]
    # A particle that starts under the gridded ground (beside the wall of the block, where
    # the heightmap turns the vertical wall into a ramp one cell wide) "lands at once": its
    # flight time is 0 and the throw model reports the ground height as its landing z. The
    # viewer needs p(t_land) = p_land, so such a particle keeps its start point as landing
    # point; it still comes to rest on the pile (p_rest).
    parts = thrown["throw"]["particles"]
    stuck = parts["t_land_ms"] <= parts["t0_ms"]
    if stuck.any():
        parts["p_land"] = np.where(stuck[:, None], parts["p0"], parts["p_land"])
    pile["stuck_particles"] = int(stuck.sum())
    if not pile["relax_converged"]:
        notes.append("The muckpile relaxation did not converge: slopes may exceed repose.")
    if pile["landed_outside_grid"]:
        notes.append(f"{pile['landed_outside_grid']} particle(s) landed outside the muckpile grid.")

    # ---- summary
    duration = timing.duration_ms(timed)
    t_end = max(duration, float(np.max(thrown["throw"]["particles"]["t_land_ms"])))
    notes.extend(
        [
            "Site constants are published typical values, not calibrated to this site: "
            "vibration K and beta, air overpressure K and a, flyrock k, Holmberg-Persson "
            "K, alpha and beta, the rock factor A unless given.",
            f"The launch velocity of the throw ({params.face_velocity:g} m/s at the face, "
            f"x {params.row_factor:g} per row) is assumed, not computed from the charge; the "
            "muckpile is a kinematic picture.",
            "The scaled-distance PPV is a far-field law: within a few burdens of the holes it "
            "overestimates; the block damage there comes from the Holmberg-Persson model.",
            "Limits shown are published guideline values, not a compliance statement.",
        ]
    )
    models = {
        "design": {
            "model": "Konya and Walter (1990) burden; S/B 1.15 staggered, 1.0 square; "
            "stemming 0.7 B, subdrill 0.3 B",
            "burden": "rule" if req.burden is None else "given",
            "spacing": "rule" if req.spacing is None else "given",
            "stemming": timed.notes["layout"]["stemming_rule"],
            "subdrill": timed.notes["layout"]["subdrill_rule"],
            "calibrated": False,
        },
        "fragmentation": {
            "model": frag.model,
            "rock_factor": frag.rock_factor,
            "oversize_m": req.oversize_m,
            "calibrated": bool(frag.rock_factor_calibrated),
        },
        "vibration": {
            "model": "PPV = k (R / sqrt(W))^-beta, W = charge per 8 ms delay",
            "k": law.k,
            "beta": law.beta,
            "window_ms": WINDOW_MS,
            "source": law.source,
            "calibrated": bool(law.calibrated),
        },
        "airblast": {
            "model": "P [kPa] = k (R / W^(1/3))^-a (AS 2187.2-2006 Appendix J)",
            "k": site["ka_kpa"],
            "a": site["a"],
            "calibrated": site["airblast_calibrated"],
        },
        "flyrock": {
            "model": fly.model,
            "k": fly.k,
            "safety_factor": fly.safety_factor_people,
            "governing_mechanism": fly.governing_mechanism,
            "calibrated": bool(fly.calibrated),
        },
        "damage": {
            **hp.to_dict(),
            "thresholds_mm_s": thresholds.as_list(),
            "thresholds_source": thresholds.source,
            "blocks": block_info,
        },
        "waves": {
            "model": "isotropic linear elastic wave speeds",
            "cp_m_s": cp,
            "cs_m_s": cs,
            "source": "measured p_wave" if rock.p_wave is not None else "elastic constants",
            "calibrated": rock.p_wave is not None,
        },
        "throw": {**params.to_dict(), "swell": rock.swell, "repose_deg": rock.repose_deg},
        "site": {"model": site_kind, "free_face_cut_m": cut_depth, "calibrated": False},
    }
    summary = {
        "volume_m3": totals["volume_m3"],
        "tonnes": totals["tonnes"],
        "explosive_kg": totals["explosive_kg"],
        "powder_factor_kg_m3": totals["powder_factor_kg_m3"],
        "powder_factor_kg_t": totals["powder_factor_kg_t"],
        "drilled_m": totals["drilled_m"],
        "holes": totals["holes"],
        "x50_m": frag.x50,
        "uniformity_n": frag.n,
        "p80_m": frag.p80,
        "oversize_fraction": frag.oversize_fraction,
        "mic_kg": mic,
        "duration_ms": duration,
        "flyrock_max_m": fly.max_range_m,
        "clearance_m": fly.clearance_people_m,
        "backbreak_m": backbreak,
        "models": models,
        "notes": notes,
        # further totals for the report
        "rows": totals["rows"],
        "stiffness_ratio": totals["stiffness_ratio"],
        "specific_drilling_m_m3": totals["specific_drilling_m_m3"],
        "burden_m": burden,
        "spacing_m": spacing,
        "stemming_m": stemming_used,
        "subdrill_m": subdrill_used,
        "diameter_m": req.diameter,
        "explosive": explosive.name,
        "burden_relief_ms_m": relief.ms_per_m,
        "throw_com_m": pile["throw_com_m"],
        "throw_max_m": pile["throw_max_m"],
        "muckpile_max_height_m": pile["muckpile_max_height_m"],
        "muckpile_volume_m3": pile["pile_volume_m3"],
        "blasted_volume_m3": pile["blasted_volume_m3"],
        "particles_not_thrown": pile["stuck_particles"],
    }
    bundle = {
        "format": FORMAT,
        "version": VERSION,
        "design": timed.to_dict(),
        "summary": summary,
        "timeline": {
            "t_end_ms": t_end,
            "holes": [
                {
                    "id": h.id,
                    "row": h.row,
                    "t_fire_ms": h.delay_ms,
                    "collar": list(h.collar),
                    "toe": list(h.toe),
                    "charge_top": h.point_at(h.charge_top),
                    "charge_bottom": h.point_at(h.charge_bottom),
                    "charge_kg": h.charge_mass,
                    "diameter_m": h.diameter,
                }
                for h in timed.holes
            ],
        },
        "waves": {"cp_m_s": cp, "cs_m_s": cs, "k": law.k, "beta": law.beta, "sources": sources},
        "ppv_surface": {
            "origin": list(origin),
            "spacing": step,
            "shape": list(shape),
            "z": drape.z.ravel(),
            "ppv_mm_s": surface_ppv,
            "arrival_ms": arrival,
        },
        "damage": {
            "classes": list(damage.CLASSES),
            "thresholds_mm_s": thresholds.as_list(),
            "blocks": dmg,
        },
        "fragmentation": frag.to_dict(),
        "throw": thrown["throw"],
        "muckpile": _with_cut(
            thrown["muckpile"], ground if cut_depth > 0.05 else None, polygon, req.face_azimuth
        ),
        "flyrock": fly.to_dict(),
        "airblast": {
            "k": site["ka_kpa"],
            "a": site["a"],
            "rings": [
                {"radius_m": float(r), "pa": float(p), "dbl": float(level)}
                for r, p, level in zip(radii, ring_pa, ring_dbl, strict=True)
            ],
        },
        "monitors": monitors,
        "envelope": {
            "window_ms": WINDOW_MS,
            "bins": [
                {"t_ms": g.t_ms, "kg": g.charge_kg, "holes": list(g.hole_ids)} for g in groups
            ],
            "mic_kg": mic,
        },
    }
    out = plain(bundle)
    problems = validate_bundle(out)
    if problems:
        raise ValueError("the result cannot be shown: " + "; ".join(problems[:5]))
    return out


# ------------------------------------------------------------------------------ validation

#: Keys of every section of the bundle (plan_blast.md section 3; web/src/blast/types.ts).
SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "summary": (
        "volume_m3", "tonnes", "explosive_kg", "powder_factor_kg_m3", "powder_factor_kg_t",
        "drilled_m", "holes", "x50_m", "uniformity_n", "p80_m", "oversize_fraction", "mic_kg",
        "duration_ms", "flyrock_max_m", "clearance_m", "backbreak_m", "models",
    ),
    "timeline": ("t_end_ms", "holes"),
    "waves": ("cp_m_s", "cs_m_s", "k", "beta", "sources"),
    "ppv_surface": ("origin", "spacing", "shape", "z", "ppv_mm_s", "arrival_ms"),
    "damage": ("classes", "thresholds_mm_s", "blocks"),
    "fragmentation": ("sizes_m", "passing", "x50_m", "n", "xc_m"),
    "throw": ("g", "particles"),
    "muckpile": ("origin", "spacing", "shape", "z_before", "z_after"),
    "flyrock": (
        "face_burst_m", "cratering_m", "stemming_ejection_m", "max_range_m", "clearance_m",
        "safety_factor",
    ),
    "airblast": ("k", "a", "rings"),
    "envelope": ("window_ms", "bins", "mic_kg"),
}  # fmt: skip
HOLE_KEYS = (
    "id", "row", "t_fire_ms", "collar", "toe", "charge_top", "charge_bottom", "charge_kg",
    "diameter_m",
)  # fmt: skip
SOURCE_KEYS = ("hole", "t_ms", "centre", "charge_kg")
BLOCK_KEYS = ("centre", "size", "ppv_mm_s", "cls", "t_break_ms", "blasted")
PARTICLE_KEYS = ("p0", "v0", "t0_ms", "t_land_ms", "p_land", "p_rest", "size_m", "kind")
MONITOR_KEYS = (
    "name", "xyz", "distance_m", "charge_kg", "scaled_distance", "ppv_mm_s", "limit_mm_s", "ok",
    "airblast_dbl", "airblast_limit_dbl", "airblast_ok",
)  # fmt: skip
RING_KEYS = ("radius_m", "pa", "dbl")
BIN_KEYS = ("t_ms", "kg", "holes")


def _is_number(value: Any) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool) and math.isfinite(value)


def validate_bundle(bundle: Any) -> list[str]:
    """Problems of a result bundle against plan_blast.md section 3; empty = valid.

    Checks the format and version, that every section and key is there, that the numbers
    the viewer needs are finite numbers (not None), the lengths of the parallel arrays and
    of the grids (``nx * ny``), that every wave source names a timeline hole, and that the
    whole bundle is plain JSON without NaN or infinity.
    """
    errors: list[str] = []
    if not isinstance(bundle, dict):
        return ["the bundle is not an object"]
    if bundle.get("format") != FORMAT:
        errors.append(f"format is {bundle.get('format')!r}, expected {FORMAT!r}")
    if bundle.get("version") != VERSION:
        errors.append(f"version is {bundle.get('version')!r}, expected {VERSION}")
    for name in ("design", *SECTION_KEYS):
        if not isinstance(bundle.get(name), dict):
            errors.append(f"{name!r} is missing")
    if not isinstance(bundle.get("monitors"), list):
        errors.append("'monitors' is missing")
    if errors:
        return errors
    for name, keys in SECTION_KEYS.items():
        errors.extend(f"{name}.{k} is missing" for k in keys if k not in bundle[name])
    if errors:
        return errors

    def numbers(what: str, obj: dict[str, Any], keys: Any) -> None:
        errors.extend(
            f"{what}.{k} is not a finite number" for k in keys if not _is_number(obj.get(k))
        )

    def records(what: str, items: Any, keys: tuple[str, ...]) -> list[dict[str, Any]]:
        if not isinstance(items, list):
            errors.append(f"{what} is not a list")
            return []
        good = []
        for i, item in enumerate(items):
            if not isinstance(item, dict):
                errors.append(f"{what}[{i}] is not an object")
                continue
            missing = [k for k in keys if k not in item]
            if missing:
                errors.append(f"{what}[{i}] lacks {missing}")
            else:
                good.append(item)
        return good

    def same(what: str, obj: Any, keys: tuple[str, ...], expected: int | None = None) -> None:
        if not isinstance(obj, dict):
            errors.append(f"{what} is missing")
            return
        n = expected
        for k in keys:
            if not isinstance(obj.get(k), list):
                errors.append(f"{what}.{k} is not a list")
                continue
            if n is None:
                n = len(obj[k])
            elif len(obj[k]) != n:
                errors.append(f"{what}.{k} has {len(obj[k])} entries, expected {n}")

    def grid(what: str, keys: tuple[str, ...]) -> None:
        g = bundle[what]
        origin, shape = g["origin"], g["shape"]
        if not (isinstance(origin, list) and len(origin) == 2 and all(map(_is_number, origin))):
            errors.append(f"{what}.origin must be [x0, y0]")
        if not (_is_number(g["spacing"]) and g["spacing"] > 0):
            errors.append(f"{what}.spacing must be > 0")
        if not (
            isinstance(shape, list)
            and len(shape) == 2
            and all(isinstance(n, int) and not isinstance(n, bool) and n >= 0 for n in shape)
        ):
            errors.append(f"{what}.shape must be [nx, ny]")
            return
        same(what, g, keys, shape[0] * shape[1])

    def xyz(value: Any) -> bool:
        return isinstance(value, list) and len(value) == 3 and all(map(_is_number, value))

    summary = bundle["summary"]
    numbers("summary", summary, [k for k in SECTION_KEYS["summary"] if k != "models"])
    if isinstance(summary["models"], dict) and summary["models"]:
        for name, info in summary["models"].items():
            if not isinstance(info, dict) or not isinstance(info.get("calibrated"), bool):
                errors.append(f"summary.models.{name}.calibrated must be true or false")
    else:
        errors.append("summary.models is empty")

    if not _is_number(bundle["timeline"]["t_end_ms"]):
        errors.append("timeline.t_end_ms is not a finite number")
    holes = records("timeline.holes", bundle["timeline"]["holes"], HOLE_KEYS)
    for h in holes:
        for k in ("collar", "toe", "charge_top", "charge_bottom"):
            if not xyz(h[k]):
                errors.append(f"timeline hole {h['id']}: {k} must be [x, y, z]")
        numbers(f"timeline hole {h['id']}", h, ("t_fire_ms", "charge_kg", "diameter_m"))
    ids = {h["id"] for h in holes}

    numbers("waves", bundle["waves"], ("cp_m_s", "cs_m_s", "k", "beta"))
    for s in records("waves.sources", bundle["waves"]["sources"], SOURCE_KEYS):
        if s["hole"] not in ids:
            errors.append(f"waves source {s['hole']!r} is not a timeline hole")
        if not xyz(s["centre"]):
            errors.append(f"waves source {s['hole']!r}: centre must be [x, y, z]")
        numbers(f"waves source {s['hole']!r}", s, ("t_ms", "charge_kg"))

    grid("ppv_surface", ("z", "ppv_mm_s", "arrival_ms"))
    grid("muckpile", ("z_before", "z_after"))

    dmg = bundle["damage"]
    same("damage.blocks", dmg["blocks"], BLOCK_KEYS)
    classes, limits = dmg["classes"], dmg["thresholds_mm_s"]
    if not (isinstance(classes, list) and isinstance(limits, list)):
        errors.append("damage.classes and damage.thresholds_mm_s must be lists")
    else:
        if len(limits) != len(classes) - 1:
            errors.append("damage.thresholds_mm_s needs one entry per class above the first")
        cls = dmg["blocks"].get("cls") if isinstance(dmg["blocks"], dict) else None
        if isinstance(cls, list) and any(
            not isinstance(c, int) or not 0 <= c < len(classes) for c in cls
        ):
            errors.append("damage.blocks.cls has entries that are not a class index")
        ppv = dmg["blocks"].get("ppv_mm_s") if isinstance(dmg["blocks"], dict) else None
        if isinstance(ppv, list) and not all(map(_is_number, ppv)):
            errors.append("damage.blocks.ppv_mm_s has entries that are not finite numbers")

    frag = bundle["fragmentation"]
    same("fragmentation", frag, ("sizes_m", "passing"))
    numbers("fragmentation", frag, ("x50_m", "n", "xc_m"))

    if not _is_number(bundle["throw"]["g"]):
        errors.append("throw.g is not a finite number")
    same("throw.particles", bundle["throw"]["particles"], PARTICLE_KEYS)

    numbers("flyrock", bundle["flyrock"], SECTION_KEYS["flyrock"])
    numbers("airblast", bundle["airblast"], ("k", "a"))
    for i, ring in enumerate(records("airblast.rings", bundle["airblast"]["rings"], RING_KEYS)):
        numbers(f"airblast.rings[{i}]", ring, RING_KEYS)

    for m in records("monitors", bundle["monitors"], MONITOR_KEYS):
        if not xyz(m["xyz"]):
            errors.append(f"monitor {m['name']!r}: xyz must be [x, y, z]")
        numbers(
            f"monitor {m['name']!r}", m, ("distance_m", "charge_kg", "scaled_distance", "ppv_mm_s")
        )

    env = bundle["envelope"]
    numbers("envelope", env, ("window_ms", "mic_kg"))
    for i, b in enumerate(records("envelope.bins", env["bins"], BIN_KEYS)):
        numbers(f"envelope.bins[{i}]", b, ("t_ms", "kg"))
        if not isinstance(b["holes"], list) or any(h not in ids for h in b["holes"]):
            errors.append(f"envelope.bins[{i}].holes must list timeline hole ids")

    def walk(value: Any, path: str) -> None:
        if len(errors) > 50:
            return
        if isinstance(value, dict):
            for k, v in value.items():
                walk(v, f"{path}.{k}")
        elif isinstance(value, list):
            for i, v in enumerate(value):
                if isinstance(v, dict | list):
                    walk(v, f"{path}[{i}]")
                elif isinstance(v, float) and not math.isfinite(v):
                    errors.append(f"{path}[{i}] is not finite")
                elif not isinstance(v, str | int | float | bool | type(None)):
                    errors.append(f"{path}[{i}] is not plain JSON ({type(v).__name__})")
        elif isinstance(value, float) and not math.isfinite(value):
            errors.append(f"{path} is not finite")
        elif not isinstance(value, str | int | float | bool | type(None)):
            errors.append(f"{path} is not plain JSON ({type(value).__name__})")

    walk(bundle, "bundle")
    return errors
