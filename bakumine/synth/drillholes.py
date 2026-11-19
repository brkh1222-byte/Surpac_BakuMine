"""Synthetic drilling: hole planning, deviation, minimum-curvature paths, logging, sampling.

Survey convention (Surpac): ``dip`` is negative downward, ``azimuth`` is degrees clockwise
from north. The unit direction of travel is

    (east, north, up) = (cos(dip) sin(az), cos(dip) cos(az), sin(dip)).

Paths are integrated by minimum curvature (research.md section 3.3): a circular arc between
consecutive stations, a straight line below the last station.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import brentq, minimize_scalar

from bakumine.synth.config import DrillConfig, GradeConfig
from bakumine.synth.field import GaussianField, GradeModel
from bakumine.synth.solid import LensSolid
from bakumine.synth.topo import Topography

ORE, WASTE = "ORE", "WASTE"
_STRAIGHT = 1e-7  # dogleg (rad) below which a segment is treated as straight


def direction(dip: np.ndarray, azimuth: np.ndarray) -> np.ndarray:
    """Unit direction vectors (..., 3) in (east, north, up) from dip and azimuth in degrees."""
    d = np.radians(np.asarray(dip, dtype=np.float64))
    a = np.radians(np.asarray(azimuth, dtype=np.float64))
    return np.stack([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)], axis=-1)


class HolePath:
    """Minimum-curvature path through survey stations; ``depth[0]`` must be 0 (the collar)."""

    def __init__(
        self, collar: np.ndarray, depth: np.ndarray, dip: np.ndarray, azimuth: np.ndarray
    ) -> None:
        self.depth = np.asarray(depth, dtype=np.float64)
        if self.depth[0] != 0.0 or np.any(np.diff(self.depth) <= 0.0):
            raise ValueError("survey depths must start at 0 and increase")
        self.tangent = direction(dip, azimuth)
        t1, t2 = self.tangent[:-1], self.tangent[1:]
        self.length = np.diff(self.depth)
        # 2*asin(|t2 - t1| / 2) is accurate for the small doglegs where acos loses digits.
        chord = np.linalg.norm(t2 - t1, axis=1)
        self.dogleg = 2.0 * np.arcsin(np.clip(0.5 * chord, 0.0, 1.0))
        curved = self.dogleg > _STRAIGHT
        safe = np.where(curved, self.dogleg, 1.0)
        ratio = np.where(curved, np.tan(0.5 * safe) / (0.5 * safe), 1.0)
        steps = 0.5 * self.length[:, None] * (t1 + t2) * ratio[:, None]
        self.station = np.asarray(collar, dtype=np.float64) + np.concatenate(
            [np.zeros((1, 3)), np.cumsum(steps, axis=0)]
        )

    def xyz(self, depth: np.ndarray) -> np.ndarray:
        """Positions (n, 3) at measured depths, on the arcs (not the chords)."""
        d = np.atleast_1d(np.asarray(depth, dtype=np.float64))
        last = len(self.depth) - 1
        i = np.clip(np.searchsorted(self.depth, d, side="right") - 1, 0, last)
        s = d - self.depth[i]
        out = self.station[i] + s[:, None] * self.tangent[i]  # straight: also below the end
        seg = np.flatnonzero(i < last)
        if seg.size:
            j = i[seg]
            beta, length, sj = self.dogleg[j], self.length[j], s[seg]
            t1, t2 = self.tangent[j], self.tangent[j + 1]
            curved = beta > _STRAIGHT
            safe = np.where(curved, beta, 1.0)
            radius = length / safe
            angle = sj / radius
            normal = (t2 - t1 * np.cos(safe)[:, None]) / np.sin(safe)[:, None]
            arc = radius[:, None] * (
                np.sin(angle)[:, None] * t1 + (1.0 - np.cos(angle))[:, None] * normal
            )
            nearly = sj[:, None] * t1 + (0.5 * sj**2 / length)[:, None] * (t2 - t1)
            out[seg] = self.station[j] + np.where(curved[:, None], arc, nearly)
        return out


@dataclass(frozen=True)
class Hole:
    hole_id: str
    collar: np.ndarray
    max_depth: float
    depth: np.ndarray  # survey stations
    dip: np.ndarray
    azimuth: np.ndarray
    path: HolePath


def _survey(
    cfg: DrillConfig, vertical: bool, dip0: float, az0: float, max_depth: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # fmt: skip
    """Stations every ``station_spacing`` with a slow, mostly systematic drift."""
    n = max(1, int(np.ceil((max_depth - 5.0) / cfg.station_spacing)))
    depth = cfg.station_spacing * np.arange(n)
    lift = rng.normal(cfg.lift_mean, cfg.lift_sd)
    swing = rng.normal(cfg.swing_mean, cfg.swing_sd)
    d_dip = lift + rng.normal(0.0, cfg.wander_sd, n - 1)
    d_az = swing + rng.normal(0.0, cfg.wander_sd, n - 1)
    if vertical:
        d_dip = 0.5 * np.abs(d_dip)  # a vertical hole can only flatten
    dip = np.clip(dip0 + np.concatenate([[0.0], np.cumsum(d_dip)]), -90.0, -5.0)
    azimuth = (az0 + np.concatenate([[0.0], np.cumsum(d_az)])) % 360.0
    azimuth = np.round(azimuth, 2) % 360.0
    return depth, np.round(dip, 2), azimuth


def plan_holes(
    cfg: DrillConfig, solid: LensSolid, topo: Topography, rng: np.random.Generator
) -> list[Hole]:
    """Holes on a rough section grid, each aimed at a point of the lens mid-surface."""
    normal = solid.axes[:, 2]
    # Drill against the dip: horizontally opposite to the upward normal of the lens.
    section_azimuth = float(np.round(np.degrees(np.arctan2(-normal[0], -normal[1])) / 5.0) * 5.0)
    section_azimuth %= 360.0
    us = np.linspace(-cfg.coverage * solid.a, cfg.coverage * solid.a, cfg.n_sections)
    vs = np.linspace(-cfg.coverage * solid.b, cfg.coverage * solid.b, cfg.holes_per_section)
    holes: list[Hole] = []
    for u0 in us:
        for v0 in vs:
            u = u0 + rng.normal(0.0, cfg.target_jitter)
            v = v0 + rng.normal(0.0, cfg.target_jitter)
            target = solid.to_world(np.array([u, v, float(solid.bend(u, v))]))
            vertical = bool(rng.random() < cfg.vertical_fraction)
            dip_noise = rng.normal(0.0, cfg.dip_sd)
            az_noise = rng.normal(0.0, cfg.azimuth_sd)
            free_azimuth = rng.uniform(0.0, 360.0)
            jitter = rng.normal(0.0, cfg.collar_jitter, 2)
            overshoot = rng.uniform(*cfg.overshoot)

            planned = direction(-90.0 if vertical else cfg.dip, section_azimuth)

            def above_ground(t: float, target=target, planned=planned) -> float:
                p = target - t * planned
                return float(p[2] - topo.z(p[0], p[1]))

            reach = brentq(above_ground, 0.0, 5_000.0, xtol=1e-9)
            xy = np.round((target - reach * planned)[:2] + jitter, 3)
            collar = np.array([xy[0], xy[1], float(topo.z(xy[0], xy[1]))])
            max_depth = float(np.round(reach + overshoot, 1))
            dip0 = -90.0 if vertical else float(np.clip(cfg.dip + dip_noise, -89.0, -20.0))
            az0 = free_azimuth if vertical else section_azimuth + az_noise
            depth, dip, azimuth = _survey(cfg, vertical, dip0, az0, max_depth, rng)
            holes.append(
                Hole(
                    hole_id=f"BM{len(holes) + 1:03d}",
                    collar=collar,
                    max_depth=max_depth,
                    depth=depth,
                    dip=dip,
                    azimuth=azimuth,
                    path=HolePath(collar, depth, dip, azimuth),
                )
            )
    return holes


def log_geology(hole: Hole, solid: LensSolid, scan_step: float) -> list[tuple[float, float, str]]:
    """Geology intervals from the exact contacts of the path with the solid (to 0.01 m)."""
    n = int(np.ceil(hole.max_depth / scan_step))
    scan = np.linspace(0.0, hole.max_depth, n + 1)
    f = solid.implicit(hole.path.xyz(scan))

    def along(d: float) -> float:
        return float(solid.implicit(hole.path.xyz(d))[0])

    roots = [
        brentq(along, scan[i], scan[i + 1], xtol=1e-6)
        for i in np.flatnonzero(np.signbit(f[:-1]) != np.signbit(f[1:]))
    ]
    # A grazing pass can enter and leave between two scan points. Wherever |f| has a local
    # minimum without a sign change, look for the extremum and, if it crosses, both roots.
    g = np.abs(f)
    same = (np.signbit(f[:-2]) == np.signbit(f[1:-1])) & (np.signbit(f[1:-1]) == np.signbit(f[2:]))
    for i in np.flatnonzero(same & (g[1:-1] <= g[:-2]) & (g[1:-1] <= g[2:])) + 1:
        sign = -1.0 if f[i] < 0.0 else 1.0
        best = minimize_scalar(
            lambda d, sign=sign: sign * along(d),
            bounds=(scan[i - 1], scan[i + 1]),
            method="bounded",
            options={"xatol": 1e-6},
        )
        if best.fun < 0.0:
            roots.append(brentq(along, scan[i - 1], best.x, xtol=1e-6))
            roots.append(brentq(along, best.x, scan[i + 1], xtol=1e-6))
    edges = [0.0, *(round(r, 2) for r in sorted(roots)), hole.max_depth]
    rock = ORE if f[0] < 0.0 else WASTE
    intervals: list[tuple[float, float, str]] = []
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        if hi > lo:
            if intervals and intervals[-1][2] == rock:
                intervals[-1] = (intervals[-1][0], hi, rock)
            else:
                intervals.append((lo, hi, rock))
        rock = WASTE if rock == ORE else ORE
    return intervals


def sample_intervals(
    cfg: DrillConfig, max_depth: float, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    """Assay intervals of about 1 m (0.6 to 1.4 m), with a few unsampled gaps left out.

    Depths are built in whole centimetres, so they are exact to two decimals.
    """
    end = int(round(max_depth * 100.0))
    position = 10 * int(rng.integers(0, int(cfg.sample_start_max * 10) + 1))
    edges = [position]
    while position < end:
        regular = rng.random() < cfg.sample_regular_fraction
        odd = 10 * int(rng.integers(6, 15))
        position = min(position + (100 if regular else odd), end)
        edges.append(position)
    if len(edges) > 2 and edges[-1] - edges[-2] < 30:
        del edges[-2]  # merge a short remnant into the previous sample
    start, stop = np.array(edges[:-1]), np.array(edges[1:])
    keep = np.ones(len(start), dtype=bool)
    for _ in range(int(rng.choice(len(cfg.gap_probabilities), p=cfg.gap_probabilities))):
        length = int(rng.integers(cfg.gap_samples[0], cfg.gap_samples[1] + 1))
        if len(start) > length + 2:
            first = int(rng.integers(1, len(start) - length))
            keep[first : first + length] = False
    return start[keep] / 100.0, stop[keep] / 100.0


@dataclass(frozen=True)
class Samples:
    """Per-sample truth for one hole (arrays of equal length)."""

    depth_from: np.ndarray
    depth_to: np.ndarray
    xyz: np.ndarray  # true position of the interval midpoint
    ore_fraction: np.ndarray  # share of the interval inside the solid
    inside_mid: np.ndarray
    gauss: np.ndarray  # Gaussian value Y at the midpoint; NaN outside the solid
    au_true: np.ndarray  # before rounding to assay precision
    au: np.ndarray


def assay_hole(
    cfg: DrillConfig,
    grade_cfg: GradeConfig,
    hole: Hole,
    solid: LensSolid,
    field: GaussianField,
    grade: GradeModel,
    rng: np.random.Generator,
) -> Samples:
    """Grades of each interval: the mean of the truth at sub-points along its true path."""
    lo, hi = sample_intervals(cfg, hole.max_depth, rng)
    n, k = len(lo), cfg.sample_subpoints
    eps_ore = rng.standard_normal(n)  # nugget: one draw per sample
    eps_waste = rng.standard_normal(n)
    fractions = (np.arange(k) + 0.5) / k
    depths = lo[:, None] + (hi - lo)[:, None] * fractions
    points = hole.path.xyz(depths.ravel())
    in_ore = solid.inside(points)
    values = np.broadcast_to(grade.waste_point(eps_waste)[:, None], (n, k)).ravel().copy()
    eps = np.broadcast_to(eps_ore[:, None], (n, k)).ravel()
    values[in_ore] = grade.ore_point(field.at(points[in_ore]), eps[in_ore])
    au_true = values.reshape(n, k).mean(axis=1)

    mid = hole.path.xyz(0.5 * (lo + hi))
    inside_mid = solid.inside(mid)
    gauss = np.full(n, np.nan)
    gauss[inside_mid] = grade.gaussian(field.at(mid[inside_mid]), eps_ore[inside_mid])
    au = np.maximum(np.round(au_true, grade_cfg.assay_decimals), grade_cfg.assay_floor)
    return Samples(
        depth_from=lo,
        depth_to=hi,
        xyz=mid,
        ore_fraction=in_ore.reshape(n, k).mean(axis=1),
        inside_mid=inside_mid,
        gauss=gauss,
        au_true=au_true,
        au=au,
    )
