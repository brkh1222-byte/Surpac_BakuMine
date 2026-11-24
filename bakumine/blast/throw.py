"""Throw and muckpile: where the broken rock of a bench blast goes.

research_blast.md part 3, section 3. This is a **simplified kinematic model**, made to show
and to size the muckpile, not to predict the path of a single rock:

1. *Particles.* ``n`` points are drawn uniformly in the blasted volume (the design polygon
   between ``floor_z`` and the bench top, the collar elevation of the nearest hole). Each
   stands for the same in-situ volume and belongs to its nearest hole.
2. *Launch.* A particle starts when its hole has fired and the P wave has reached it
   (``t0 = delay + distance / cp``, plus an optional response time per metre). Its speed is
   ``face_velocity * row_factor^row`` with random scatter: the front row moves freely, the
   rows behind are more confined. The direction is the face azimuth, with an elevation
   angle that grows linearly from the toe to the crest, both with scatter. The launch
   speed is **not** computed from the charge: no burden-velocity formula with constants we
   could verify is implemented. The default 15 m/s lies in the 10-30 m/s range commonly
   reported from high-speed films of bench faces (Chiappetta et al. 1983, as quoted in
   Jimeno, Jimeno and Carcedo 1995); all launch parameters are uncalibrated.
3. *Fragment size.* Drawn from the Rosin-Rammler distribution of the fragmentation model,
   ``x = x50 (-ln(1 - u) / ln 2)^(1 / n)``. The size only slows the launch when
   ``size_exponent > 0``.
4. *Flight.* No air drag: ``p(t) = p0 + v0 t - g t^2 / 2 ez`` until the path first meets the
   ground. The ground is the surface with the blasted block taken out (``floor_z`` inside
   the polygon), held as a heightmap and interpolated bilinearly. Particles do not collide,
   bounce or roll.
5. *Muckpile.* Each landed particle puts its swelled volume
   (``blasted volume * swell / n``) into the grid cell it lands in. The heap is then relaxed
   until no slope between neighbouring cells (8 neighbours) is steeper than the angle of
   repose: for a pair that is too steep, a part of the excess height moves from the upper to
   the lower cell (talus / thermal-erosion relaxation, Musgrave, Kolb and Mace 1989). Only
   the loose rock moves, never the ground below it, and the volume is conserved exactly.

Heightmap layout: node ``(ix, iy)`` is at ``x = origin[0] + ix * spacing``,
``y = origin[1] + iy * spacing``; arrays are ``[ix, iy]`` and flattened row-major
(``ix * ny + iy``) in the result, as in the bundle's ``ppv_surface``.

Units: metres, seconds, kilograms inside; ``*_ms`` in milliseconds.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from bakumine.blast.damage import _segment_distance, bench_top, points_in_polygon
from bakumine.blast.model import BlastDesign
from bakumine.blast.waves import wave_speeds

__all__ = [
    "Heightmap",
    "ThrowParams",
    "deposit",
    "land",
    "max_slope_excess",
    "particles",
    "relax",
    "rosin_rammler_size",
    "simulate_throw",
]

Surface = Callable[[np.ndarray, np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ThrowParams:
    """Parameters of the launch model. All are assumptions until calibrated on a filmed blast.

    ``face_velocity``: speed of the front row burden, m/s (commonly observed 10-30 m/s).
    ``row_factor``: speed of row ``r`` is ``face_velocity * row_factor**r``.
    ``elevation_toe_deg`` / ``elevation_crest_deg``: launch angle above horizontal at the
    floor and at the bench top. ``velocity_scatter``: relative standard deviation of the
    speed. ``angle_scatter_deg``: standard deviation of elevation and azimuth.
    ``size_exponent``: fragments larger than x50 are slowed by ``(x50 / size)**exponent``
    (0 = size has no effect). ``response_ms_per_m``: extra delay of the start per metre from
    the charge (0 = start at the P-wave arrival).
    """

    face_velocity: float = 15.0
    row_factor: float = 0.8
    elevation_toe_deg: float = 5.0
    elevation_crest_deg: float = 25.0
    velocity_scatter: float = 0.15
    angle_scatter_deg: float = 5.0
    size_exponent: float = 0.0
    response_ms_per_m: float = 0.0
    g: float = 9.81
    calibrated: bool = False

    def __post_init__(self) -> None:
        if not (math.isfinite(self.face_velocity) and self.face_velocity > 0):
            raise ValueError("face_velocity must be > 0")
        if not 0.0 < self.row_factor <= 1.0:
            raise ValueError("row_factor must be in (0, 1]")
        if min(self.velocity_scatter, self.angle_scatter_deg, self.size_exponent) < 0:
            raise ValueError("scatter and size_exponent must be >= 0")
        if self.response_ms_per_m < 0 or not self.g > 0:
            raise ValueError("response_ms_per_m must be >= 0 and g > 0")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "parametrised ballistic throw", **asdict(self)}


def rosin_rammler_size(u: Any, x50: float, n: float) -> np.ndarray:
    """Inverse of the Rosin-Rammler distribution ``P(x) = 1 - exp(-ln 2 (x / x50)^n)``.

    ``u`` in [0, 1) is the passing fraction; the result is the size with that fraction.
    """
    if not (x50 > 0 and n > 0):
        raise ValueError("x50 and the uniformity index must be > 0")
    u = np.asarray(u, dtype=np.float64)
    return x50 * (-np.log1p(-u) / math.log(2.0)) ** (1.0 / n)


# ---------------------------------------------------------------------------------------
# Particles


def _blasted_volume(design: BlastDesign, cells: int = 200) -> float:
    """Volume of the polygon between the floor and the bench top (midpoint rule in plan)."""
    poly = np.asarray(design.polygon, dtype=np.float64)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    gx = lo[0] + (np.arange(cells) + 0.5) * (hi[0] - lo[0]) / cells
    gy = lo[1] + (np.arange(cells) + 0.5) * (hi[1] - lo[1]) / cells
    xy = np.column_stack([np.repeat(gx, cells), np.tile(gy, cells)])
    xy = xy[points_in_polygon(xy, poly)]
    height = bench_top(xy, design) - design.floor_z
    return design.plan_area * float(np.mean(np.maximum(height, 0.0)))


def particles(
    design: BlastDesign,
    n: int = 4000,
    seed: int = 0,
    *,
    x50: float,
    uniformity: float,
    params: ThrowParams | None = None,
    cp: float | None = None,
) -> dict[str, Any]:
    """Sample the blasted rock as particles with start time, size and launch velocity.

    Parameters
    ----------
    design : BlastDesign
        Needs a ``polygon`` and ``delay_ms`` on every hole.
    n : int
        Number of particles.
    seed : int
        Seed of the ``numpy`` generator: the same seed gives the same particles.
    x50, uniformity : float
        Median fragment size (m) and Rosin-Rammler uniformity index of the fragmentation.
    params : ThrowParams, optional
    cp : float, optional
        P-wave speed, m/s; default from the rock.

    Returns
    -------
    dict
        ``p0`` (n, 3), ``v0`` (n, 3), ``t0_ms`` (n,), ``size_m`` (n,), ``hole`` (n,) index
        into ``design.holes`` of the nearest hole, ``blasted_volume_m3`` (float).
    """
    params = params or ThrowParams()
    if n < 1:
        raise ValueError("need at least one particle")
    if len(design.polygon) < 3:
        raise ValueError("the throw model needs the blast polygon of the design")
    if not design.holes:
        raise ValueError("the design has no holes")
    for hole in design.holes:
        if not math.isfinite(hole.delay_ms):
            raise ValueError(f"hole {hole.id} has no delay_ms: time the design before the throw")
    if cp is None:
        cp, _ = wave_speeds(design.rock)
    rng = np.random.default_rng(seed)
    poly = np.asarray(design.polygon, dtype=np.float64)
    lo, hi = poly.min(axis=0), poly.max(axis=0)
    collars = np.array([h.collar for h in design.holes])
    height_max = float(collars[:, 2].max()) - design.floor_z
    if not height_max > 0:
        raise ValueError("the hole collars are not above floor_z")

    # uniform in the volume: uniform in plan, accepted in proportion to the bench height there
    xy = np.empty((0, 2))
    for _ in range(1000):
        cand = rng.uniform(lo, hi, size=(max(2 * n, 256), 2))
        cand = cand[points_in_polygon(cand, poly)]
        height = bench_top(cand, design) - design.floor_z
        cand = cand[rng.uniform(0.0, height_max, len(cand)) < height]
        xy = np.vstack([xy, cand])
        if len(xy) >= n:
            break
    else:
        raise ValueError("could not sample the blast polygon (degenerate polygon?)")
    xy = xy[:n]
    _, nearest = cKDTree(collars[:, :2]).query(xy)
    top = collars[nearest, 2]
    frac = rng.uniform(0.0, 1.0, n)  # 0 = toe, 1 = crest
    p0 = np.column_stack([xy, design.floor_z + frac * (top - design.floor_z)])

    size = rosin_rammler_size(rng.uniform(0.0, 1.0, n), x50, uniformity)

    distance = np.empty(n)
    for k, hole in enumerate(design.holes):
        mine = nearest == k
        if mine.any():
            a, b = hole.point_at(hole.charge_top), hole.point_at(hole.charge_bottom)
            distance[mine] = _segment_distance(p0[mine], a, b)
    delay = np.array([h.delay_ms for h in design.holes])[nearest]
    t0_ms = delay + distance / cp * 1000.0 + params.response_ms_per_m * distance

    row = np.array([h.row for h in design.holes], dtype=np.float64)[nearest]
    speed = params.face_velocity * params.row_factor**row
    speed = speed * np.clip(1.0 + params.velocity_scatter * rng.standard_normal(n), 0.1, None)
    if params.size_exponent > 0:
        speed = speed * np.minimum(1.0, (x50 / size) ** params.size_exponent)
    elevation = np.radians(
        params.elevation_toe_deg
        + (params.elevation_crest_deg - params.elevation_toe_deg) * frac
        + params.angle_scatter_deg * rng.standard_normal(n)
    )
    azimuth = np.radians(design.face_azimuth + params.angle_scatter_deg * rng.standard_normal(n))
    horizontal = speed * np.cos(elevation)
    v0 = np.column_stack(
        [horizontal * np.sin(azimuth), horizontal * np.cos(azimuth), speed * np.sin(elevation)]
    )
    return {
        "p0": p0,
        "v0": v0,
        "t0_ms": t0_ms,
        "size_m": size,
        "hole": nearest.astype(np.int64),
        "blasted_volume_m3": _blasted_volume(design),
    }


# ---------------------------------------------------------------------------------------
# Ground and flight


@dataclass
class Heightmap:
    """A regular grid of heights ``z[ix, iy]`` with bilinear interpolation.

    ``z_at`` clamps positions outside the grid to its edge; NaN nodes give NaN around them.
    """

    origin: tuple[float, float]
    spacing: float
    z: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape  # type: ignore[return-value]

    def z_at(self, x: Any, y: Any) -> np.ndarray:
        nx, ny = self.z.shape
        fx = np.clip((np.asarray(x, dtype=np.float64) - self.origin[0]) / self.spacing, 0, nx - 1)
        fy = np.clip((np.asarray(y, dtype=np.float64) - self.origin[1]) / self.spacing, 0, ny - 1)
        ix = np.minimum(fx.astype(np.int64), max(nx - 2, 0))
        iy = np.minimum(fy.astype(np.int64), max(ny - 2, 0))
        ix1, iy1 = np.minimum(ix + 1, nx - 1), np.minimum(iy + 1, ny - 1)
        tx, ty = fx - ix, fy - iy
        z = self.z
        return (
            z[ix, iy] * (1 - tx) * (1 - ty)
            + z[ix1, iy] * tx * (1 - ty)
            + z[ix, iy1] * (1 - tx) * ty
            + z[ix1, iy1] * tx * ty
        )

    def cell_of(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        """Index of the nearest node (clamped to the grid)."""
        nx, ny = self.z.shape
        ix = np.rint((np.asarray(x, dtype=np.float64) - self.origin[0]) / self.spacing)
        iy = np.rint((np.asarray(y, dtype=np.float64) - self.origin[1]) / self.spacing)
        return (
            np.clip(ix, 0, nx - 1).astype(np.int64),
            np.clip(iy, 0, ny - 1).astype(np.int64),
        )


def _ballistic_landing(
    p0: np.ndarray,
    v0: np.ndarray,
    surface: Surface,
    z_min: float,
    g: float = 9.81,
    steps: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """First meeting of drag-free trajectories with the ground.

    ``surface(x, y)`` is the ground height (vectorised); ``z_min`` an estimate of its lowest
    level, used for the first guess of the flight time (the guess is doubled until every
    particle is on or under the ground). The flight is sampled at ``steps`` equal time
    steps; the first step whose end is on or below the ground brackets the landing, which
    bisection then locates. A particle that starts on or below the ground lands at once. A
    ridge thinner than one time step can be flown through.

    Returns the flight times (n,) in seconds and the landing points (n, 3), whose z is the
    ground height at the landing position.
    """
    p0 = np.asarray(p0, dtype=np.float64)
    v0 = np.asarray(v0, dtype=np.float64)

    def height(t: np.ndarray) -> np.ndarray:  # above the ground; t is (n,) or (n, k)
        base, vel = (p0, v0) if t.ndim == 1 else (p0[:, None, :], v0[:, None, :])
        pos = base + vel * t[..., None]
        return pos[..., 2] - 0.5 * g * t**2 - surface(pos[..., 0], pos[..., 1])

    drop = np.maximum(p0[:, 2] - (z_min - 1.0), 0.0)
    t_end = (v0[:, 2] + np.sqrt(v0[:, 2] ** 2 + 2.0 * g * drop)) / g
    t_end = np.maximum(t_end, 1e-6)
    for _ in range(40):
        flying = ~(height(t_end) <= 0.0)
        if not flying.any():
            break
        t_end = np.where(flying, 2.0 * t_end, t_end)
    else:
        raise ValueError("particles never reach the ground (ground unknown along the flight?)")
    times = t_end[:, None] * np.linspace(0.0, 1.0, steps + 1)
    below = height(times) <= 0.0
    below[:, -1] = True
    first = np.argmax(below, axis=1)
    rows = np.arange(len(p0))
    hi = times[rows, first]
    lo = times[rows, np.maximum(first - 1, 0)]
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        under = height(mid) <= 0.0
        hi = np.where(under, mid, hi)
        lo = np.where(under, lo, mid)
    t = np.where(first == 0, 0.0, hi)
    xy = p0[:, :2] + v0[:, :2] * t[:, None]
    return t, np.column_stack([xy, surface(xy[:, 0], xy[:, 1])])


def land(
    p0: Any, v0: Any, ground: float | Any, g: float = 9.81, steps: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Flight time (s) and landing point of drag-free particles on ``ground``.

    ``ground`` is a level (float) or an object with ``z_at(x, y)`` that is finite wherever
    the particles fly (a :class:`Heightmap`, a DTM over its own area).
    """
    p0 = np.atleast_2d(np.asarray(p0, dtype=np.float64))
    v0 = np.atleast_2d(np.asarray(v0, dtype=np.float64))
    if isinstance(ground, int | float):
        level = float(ground)
        return _ballistic_landing(p0, v0, lambda x, y: np.full(np.shape(x), level), level, g, steps)
    z_min = float(np.nanmin(ground.z_at(p0[:, 0], p0[:, 1])))
    return _ballistic_landing(
        p0, v0, lambda x, y: np.asarray(ground.z_at(x, y), dtype=np.float64), z_min, g, steps
    )


# ---------------------------------------------------------------------------------------
# Muckpile

# (step in ix, step in iy) of the four neighbour directions; the other four are their mirror
_DIRECTIONS = ((1, 0), (0, 1), (1, 1), (1, -1))


def _pair_views(a: np.ndarray, di: int, dj: int, parity: int) -> tuple[np.ndarray, np.ndarray]:
    """Views of the two cells of disjoint neighbour pairs (each cell is in one pair)."""
    nx, ny = a.shape
    if (di, dj) == (1, 0):
        return a[parity : nx - 1 : 2, :], a[parity + 1 : nx : 2, :]
    if (di, dj) == (0, 1):
        return a[:, parity : ny - 1 : 2], a[:, parity + 1 : ny : 2]
    if (di, dj) == (1, 1):
        return a[parity : nx - 1 : 2, : ny - 1], a[parity + 1 : nx : 2, 1:]
    return a[parity : nx - 1 : 2, 1:], a[parity + 1 : nx : 2, : ny - 1]


def relax(
    z_base: np.ndarray,
    thickness: np.ndarray,
    spacing: float,
    tan_repose: float,
    tol: float = 1e-4,
    max_iter: int = 20000,
    damping: float = 0.25,
) -> tuple[np.ndarray, int, bool]:
    """Let loose rock slide until no slope exceeds the angle of repose.

    Parameters
    ----------
    z_base : (nx, ny) array
        Fixed ground (finite).
    thickness : (nx, ny) array
        Loose rock on it, metres (>= 0).
    spacing : float
        Grid spacing, metres.
    tan_repose : float
        Largest allowed slope of the loose rock.
    tol : float
        Stop when the largest transfer of one sweep is below ``tol`` metres of thickness.
    max_iter : int
        Largest number of sweeps.
    damping : float
        Share of a pair's excess height moved in one visit, in (0, 0.5]. 0.5 brings the pair
        exactly to the limit; smaller values spread a tall column more evenly around it.

    Returns
    -------
    thickness : (nx, ny) array
        Relaxed; ``thickness.sum()`` is unchanged (to rounding) and nothing is negative.
    iterations : int
    converged : bool

    Each sweep visits the pairs of the 8-neighbourhood in groups of disjoint pairs. For a
    pair whose surface difference exceeds ``tan_repose * distance``, ``damping`` times the
    excess (at most what the upper cell holds) moves to the lower cell. The grid edge is
    closed: nothing leaves. The resting shape is one of many stable ones and depends
    slightly on the order of the sweep.
    """
    t = np.array(thickness, dtype=np.float64)
    base = np.asarray(z_base, dtype=np.float64)
    if t.shape != base.shape or not np.all(np.isfinite(base)):
        raise ValueError("z_base must be finite and have the shape of thickness")
    if np.any(t < 0):
        raise ValueError("thickness must be >= 0")
    if not 0.0 < damping <= 0.5:
        raise ValueError("damping must be in (0, 0.5]")
    for iteration in range(1, max_iter + 1):
        largest = 0.0
        # the same sweep seen through the 8 symmetries of the grid in turn, so that the
        # order of the pairs does not push the heap to one side
        k = iteration % 8
        fx, fy = (1, -1)[k & 1], (1, -1)[(k >> 1) & 1]
        tv, bv = t[::fx, ::fy], base[::fx, ::fy]
        if k & 4:
            tv, bv = tv.T, bv.T
        for di, dj in _DIRECTIONS:
            limit = tan_repose * spacing * math.hypot(di, dj)
            for parity in (0, 1):
                ta, tb = _pair_views(tv, di, dj, parity)
                if ta.size == 0:
                    continue
                ba, bb = _pair_views(bv, di, dj, parity)
                diff = (ba + ta) - (bb + tb)
                move = np.minimum(np.maximum(damping * (diff - limit), 0.0), ta)
                move -= np.minimum(np.maximum(damping * (-diff - limit), 0.0), tb)
                ta -= move
                tb += move
                largest = max(largest, float(np.abs(move).max()))
        if largest < tol:
            return t, iteration, True
    return t, max_iter, False


def max_slope_excess(
    z_base: np.ndarray, thickness: np.ndarray, spacing: float, tan_repose: float
) -> float:
    """Largest slope above ``tan_repose`` between neighbours whose upper cell holds loose rock.

    0 when the pile is nowhere too steep. Slopes of bare ground are not counted.
    """
    surface = z_base + thickness
    worst = 0.0
    nx, ny = surface.shape
    for di, dj in _DIRECTIONS:
        sa = surface[: nx - di, max(-dj, 0) : ny - max(dj, 0)]
        sb = surface[di:, max(dj, 0) : ny - max(-dj, 0)]
        ta = thickness[: nx - di, max(-dj, 0) : ny - max(dj, 0)]
        tb = thickness[di:, max(dj, 0) : ny - max(-dj, 0)]
        if sa.size == 0:
            continue
        slope = (sa - sb) / (spacing * math.hypot(di, dj))
        loose = np.where(slope > 0, ta, tb) > 1e-12
        if loose.any():
            worst = max(worst, float(np.abs(slope[loose]).max()) - tan_repose)
    return max(worst, 0.0)


def deposit(grid: Heightmap, xy: np.ndarray, volume: Any) -> np.ndarray:
    """Thickness (m) of loose rock when each point puts ``volume`` (m3) into its nearest cell.

    Points outside the grid go to the nearest edge cell, so the volume is conserved.
    """
    ix, iy = grid.cell_of(xy[:, 0], xy[:, 1])
    thickness = np.zeros(grid.shape)
    np.add.at(thickness, (ix, iy), np.broadcast_to(volume, ix.shape) / grid.spacing**2)
    return thickness


def _sample_ground(ground: float | Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    if isinstance(ground, int | float):
        return np.full(x.shape, float(ground))
    return np.asarray(ground.z_at(x, y), dtype=np.float64).reshape(x.shape)


def simulate_throw(
    design: BlastDesign,
    ground: float | Any,
    *,
    x50: float,
    uniformity: float,
    n: int = 4000,
    seed: int = 0,
    params: ThrowParams | None = None,
    origin: tuple[float, float] | None = None,
    spacing: float | None = None,
    shape: tuple[int, int] | None = None,
    cp: float | None = None,
    relax_tol: float = 1e-4,
    max_iter: int = 20000,
) -> dict[str, Any]:
    """Throw the blasted rock and build the muckpile.

    Parameters
    ----------
    design : BlastDesign
        Timed design with a polygon.
    ground : float or object with ``z_at(x, y)``
        Surface before the blast: a level, or e.g. a ``DTM``. It is sampled once on the
        heightmap grid. Where it is unknown (NaN) the level ``floor_z`` is used for the
        flight and the pile, and ``z_before`` / ``z_after`` stay NaN there.
    x50, uniformity, n, seed, params, cp
        See :func:`particles`.
    origin, spacing, shape
        Heightmap grid. Default: the polygon's bounding box widened by the longest possible
        flight plus the run-out of a repose slope one bench high, with a spacing of 1/120 of
        the box plus flight (not below 0.5 m), which gives roughly 120 to 150 cells a side.

    Returns
    -------
    dict
        ``"throw"``: ``{"g", "particles": {p0, v0, t0_ms, t_land_ms, p_land, p_rest, size_m,
        kind}}`` (``kind`` 0 = muck; ``p_rest`` is the landing position in plan on the final
        pile surface); ``"muckpile"``: ``{origin, spacing, shape, z_before, z_after}`` with
        the two surfaces flattened row-major; ``"summary"``: throw of the centre of mass,
        largest throw, pile height, volumes and the swell check; ``"model"``: the launch
        parameters. Arrays are numpy arrays (NaN possible in ``z_before`` / ``z_after``).
    """
    params = params or ThrowParams()
    rock = design.rock
    parts = particles(design, n, seed, x50=x50, uniformity=uniformity, params=params, cp=cp)
    p0, v0 = parts["p0"], parts["v0"]
    volume = parts["blasted_volume_m3"]
    tan_repose = math.tan(math.radians(rock.repose_deg))
    poly = np.asarray(design.polygon, dtype=np.float64)

    if origin is None or spacing is None or shape is None:
        speed = float(np.sqrt((v0**2).sum(axis=1)).max())
        drop = float(p0[:, 2].max() - design.floor_z) + design.bench_height
        reach = speed / params.g * math.sqrt(speed**2 + 2.0 * params.g * drop)
        lo, hi = poly.min(axis=0), poly.max(axis=0)
        if spacing is None:
            spacing = max(0.5, float((hi - lo).max() + 2.0 * reach) / 120.0)
        margin = reach + (p0[:, 2].max() - design.floor_z) / tan_repose + 2.0 * spacing
        if origin is None:
            origin = (float(lo[0] - margin), float(lo[1] - margin))
        if shape is None:
            far = hi + margin
            shape = (
                int(math.ceil((far[0] - origin[0]) / spacing)) + 1,
                int(math.ceil((far[1] - origin[1]) / spacing)) + 1,
            )
    nx, ny = int(shape[0]), int(shape[1])
    if nx < 2 or ny < 2 or not spacing > 0:
        raise ValueError("the heightmap needs a spacing > 0 and at least 2 x 2 nodes")
    gx = origin[0] + spacing * np.arange(nx)
    gy = origin[1] + spacing * np.arange(ny)
    xx, yy = np.meshgrid(gx, gy, indexing="ij")
    z_before = _sample_ground(ground, xx, yy)
    known = np.isfinite(z_before)

    # ground the rock lands on: unknown -> floor level, blasted block taken out
    z_base = np.where(known, z_before, design.floor_z)
    in_block = points_in_polygon(np.column_stack([xx.ravel(), yy.ravel()]), poly).reshape(nx, ny)
    z_base[in_block] = np.minimum(z_base[in_block], design.floor_z)
    base = Heightmap((float(origin[0]), float(origin[1])), float(spacing), z_base)

    flight, p_land = _ballistic_landing(p0, v0, base.z_at, float(z_base.min()), params.g)
    t_land_ms = parts["t0_ms"] + flight * 1000.0

    pile_volume = volume * rock.swell
    raw = deposit(base, p_land[:, :2], pile_volume / n)
    thickness, iterations, converged = relax(
        z_base, raw, float(spacing), tan_repose, tol=relax_tol, max_iter=max_iter
    )
    z_after_full = z_base + thickness
    z_after = np.where(known | in_block, z_after_full, np.nan)
    final = Heightmap(base.origin, base.spacing, z_after_full)
    p_rest = p_land.copy()
    p_rest[:, 2] = np.minimum(final.z_at(p_land[:, 0], p_land[:, 1]), z_after_full.max())

    shift = p_land[:, :2] - p0[:, :2]
    inside = (
        (p_land[:, 0] >= gx[0])
        & (p_land[:, 0] <= gx[-1])
        & (p_land[:, 1] >= gy[0])
        & (p_land[:, 1] <= gy[-1])
    )
    cell = float(spacing) ** 2
    summary = {
        "particles": int(n),
        "blasted_volume_m3": float(volume),
        "pile_volume_m3": float(thickness.sum() * cell),
        "swell": float(rock.swell),
        "swell_check": float(thickness.sum() * cell / volume),
        "throw_com_m": float(np.linalg.norm(shift.mean(axis=0))),
        "throw_max_m": float(np.linalg.norm(shift, axis=1).max()),
        "muckpile_max_height_m": float(thickness.max()),
        "repose_deg": float(rock.repose_deg),
        "max_slope_excess": max_slope_excess(z_base, thickness, float(spacing), tan_repose),
        "relax_iterations": int(iterations),
        "relax_converged": bool(converged),
        "landed_outside_grid": int((~inside).sum()),
        "volume_on_unknown_ground_m3": float(thickness[~(known | in_block)].sum() * cell),
    }
    return {
        "throw": {
            "g": params.g,
            "particles": {
                "p0": p0,
                "v0": v0,
                "t0_ms": parts["t0_ms"],
                "t_land_ms": t_land_ms,
                "p_land": p_land,
                "p_rest": p_rest,
                "size_m": parts["size_m"],
                "kind": np.zeros(n, dtype=np.int64),
            },
        },
        "muckpile": {
            "origin": [float(origin[0]), float(origin[1])],
            "spacing": float(spacing),
            "shape": [nx, ny],
            "z_before": z_before.ravel(),
            "z_after": z_after.ravel(),
        },
        "summary": summary,
        "model": params.to_dict(),
    }
