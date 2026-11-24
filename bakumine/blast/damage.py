"""Near-field blast damage: Holmberg-Persson PPV, damage classes, back-break, block attribute.

research_blast_physics.md section 2. Close to a blast hole the charge is not a point: the
vibration at a point is built up from the whole charge column. Holmberg and Persson (1978,
1979) take the far-field law ``PPV = K W^alpha / R^beta`` for every element ``q dx`` of the
column and add the elements up:

    PPV = K * ( q * I )^alpha,    I = integral dx / (r0^2 + (x - x0)^2)^(beta / (2 alpha))

over the charged length, with ``q`` the linear charge density (kg/m), ``r0`` the perpendicular
distance of the point from the hole axis and ``x0`` its position along the axis. PPV is in
mm/s, lengths in metres. The published constants for Swedish hard bedrock are ``K = 700``,
``alpha = 0.7``, ``beta = 1.5``; they are **site constants** and stay labelled uncalibrated
until near-field measurements of the site replace them.

The integral
------------
With ``x - x0 = r0 tan(theta)`` and ``p = beta / (2 alpha)``

    I = r0^(1 - 2p) * integral cos(theta)^(2p - 2) d theta,   theta = atan((x - x0) / r0)

which is smooth and is integrated by Gauss-Legendre quadrature. For ``beta = 2 alpha``
(``p = 1``) it is the closed form ``(atan(b) - atan(a)) / r0`` of the original papers. ``r0``
is not allowed below the hole radius: inside the hole the model has no meaning.

Known limits of the model (stated, not corrected): the elements are added as if their peaks
arrived together and pointed the same way (Blair and Minchinton 1996 discuss this), so the
PPV near a long column is an upper estimate. Several holes are **not** summed: the field is
the maximum over the holes (``combine="max"``), or, with ``combine="window"``, the holes
that fire inside one time window are treated as one charge (their ``q I`` are added).

Damage classes
--------------
A plane wave with particle velocity ``v`` strains the rock by ``v / cp``; the stress
``E v / cp`` reaches the tensile strength at the critical velocity

    PPV_c = sigma_t * cp / E

(Persson, Holmberg and Lee 1994). The classes are intact < damaged < fractured < crushed,
with "fractured" above ``PPV_c``, "damaged" above ``0.4 PPV_c`` and "crushed" above
``4 PPV_c``. The two multiples are the ratios of the steps in the damage table of Bauer and
Calder (1978): 250 / 635 / 2540 mm/s (10 / 25 / 100 in/s). Tying ``PPV_c`` to their 635 mm/s
step is a choice of this module, not a published rule; the thresholds carry
``calibrated=False``. :data:`BAUER_CALDER` holds the table itself.

Units: metres, seconds, kilograms inside; PPV in mm/s and times in ms where the name says so.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.spatial import cKDTree

from bakumine.blast.model import BlastDesign, Hole, Rock
from bakumine.blast.waves import wave_speeds

__all__ = [
    "BAUER_CALDER",
    "CLASSES",
    "DamageThresholds",
    "HPConstants",
    "backbreak",
    "bench_top",
    "block_damage",
    "classify",
    "critical_ppv",
    "hole_ppv",
    "hp_integral",
    "points_in_polygon",
    "ppv_field",
    "thresholds_for",
]

#: Names of the classes returned by :func:`classify` (index = class).
CLASSES = ("intact", "damaged", "fractured", "crushed")

_CHUNK = 32768  # points per chunk of the quadrature


@dataclass(frozen=True)
class HPConstants:
    """Site constants of the Holmberg-Persson law (PPV mm/s, charge kg, distance m).

    The defaults are the published values for Swedish hard bedrock (Holmberg and Persson
    1979); ``calibrated`` says whether they were fitted to measurements of this site.
    """

    k: float = 700.0
    alpha: float = 0.7
    beta: float = 1.5
    calibrated: bool = False

    def __post_init__(self) -> None:
        for key in ("k", "alpha", "beta"):
            value = getattr(self, key)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"Holmberg-Persson {key} must be > 0, got {value!r}")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "Holmberg-Persson", **asdict(self)}


@dataclass(frozen=True)
class DamageThresholds:
    """PPV limits of the damage classes, mm/s: ``damaged < fractured < crushed``."""

    damaged_mm_s: float
    fractured_mm_s: float
    crushed_mm_s: float
    calibrated: bool = False
    source: str = ""

    def __post_init__(self) -> None:
        values = self.as_list()
        if not all(math.isfinite(v) and v > 0 for v in values):
            raise ValueError(f"damage thresholds must be > 0, got {values}")
        if not values[0] < values[1] < values[2]:
            raise ValueError(f"damage thresholds must increase, got {values}")

    def as_list(self) -> list[float]:
        """``[damaged, fractured, crushed]`` (the bundle's ``damage.thresholds_mm_s``)."""
        return [self.damaged_mm_s, self.fractured_mm_s, self.crushed_mm_s]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


#: The rock-independent damage table of Bauer and Calder (1978): below 250 mm/s no fracturing
#: of intact rock, 250-635 minor tensile slabbing, 635-2540 strong tensile and some radial
#: cracking, above 2540 mm/s complete break-up of the rock mass.
BAUER_CALDER = DamageThresholds(250.0, 635.0, 2540.0, source="Bauer and Calder (1978)")


def critical_ppv(rock: Rock) -> float:
    """Critical particle velocity ``sigma_t cp / E`` of the rock, mm/s."""
    cp, _ = wave_speeds(rock)
    return rock.tensile_mpa * 1e6 * cp / (rock.youngs_gpa * 1e9) * 1000.0


def thresholds_for(
    rock: Rock,
    damaged_factor: float = 0.4,
    crushed_factor: float = 4.0,
    calibrated: bool = False,
) -> DamageThresholds:
    """Damage thresholds of a rock from its critical PPV.

    ``fractured = PPV_c``, ``damaged = damaged_factor * PPV_c``,
    ``crushed = crushed_factor * PPV_c``. The default factors are the ratios 250 / 635 and
    2540 / 635 of the Bauer and Calder (1978) table (module docstring).
    """
    if not 0.0 < damaged_factor < 1.0 < crushed_factor:
        raise ValueError("need 0 < damaged_factor < 1 < crushed_factor")
    ppv_c = critical_ppv(rock)
    return DamageThresholds(
        damaged_factor * ppv_c,
        ppv_c,
        crushed_factor * ppv_c,
        calibrated=calibrated,
        source="PPV_c = sigma_t cp / E; multiples after Bauer and Calder (1978)",
    )


# ---------------------------------------------------------------------------------------
# Holmberg-Persson


def _points(points: Any) -> np.ndarray:
    p = np.atleast_2d(np.asarray(points, dtype=np.float64))
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {p.shape}")
    return p


def _axis_coordinates(points: np.ndarray, hole: Hole) -> tuple[np.ndarray, np.ndarray]:
    """Depth along the hole axis and perpendicular distance from it (not below the radius)."""
    d = hole.direction
    rel = points - np.asarray(hole.collar)
    s = rel @ d
    r0 = np.linalg.norm(rel - s[:, None] * d, axis=1)
    return s, np.maximum(r0, 0.5 * hole.diameter)


def hp_integral(
    r0: Any, x0: Any, x_top: float, x_bottom: float, exponent: float, nodes: int = 32
) -> np.ndarray:
    """``integral_{x_top}^{x_bottom} dx / (r0^2 + (x - x0)^2)^exponent``.

    Parameters
    ----------
    r0, x0 : array_like
        Perpendicular distance from the axis (> 0) and position along the axis, metres.
    x_top, x_bottom : float
        Ends of the charge column along the axis.
    exponent : float
        ``beta / (2 alpha)``. Exactly 1 uses the arctan closed form.
    nodes : int
        Gauss-Legendre nodes for the general case.
    """
    r0 = np.asarray(r0, dtype=np.float64)
    x0 = np.asarray(x0, dtype=np.float64)
    if np.any(~(r0 > 0)):
        raise ValueError("r0 must be > 0")
    th1 = np.arctan((x_top - x0) / r0)
    th2 = np.arctan((x_bottom - x0) / r0)
    if exponent == 1.0:
        return (th2 - th1) / r0
    xk, wk = np.polynomial.legendre.leggauss(nodes)
    half = 0.5 * (th2 - th1)
    mid = 0.5 * (th2 + th1)
    theta = mid[..., None] + half[..., None] * xk
    integrand = np.cos(theta) ** (2.0 * exponent - 2.0)
    return r0 ** (1.0 - 2.0 * exponent) * half * (integrand @ wk)


def _hole_term(points: np.ndarray, hole: Hole, constants: HPConstants, nodes: int) -> np.ndarray:
    """``q * I`` of one hole at the points (PPV = K * term^alpha)."""
    out = np.zeros(len(points))
    if not hole.charge_length > 0:
        return out
    exponent = constants.beta / (2.0 * constants.alpha)
    for start in range(0, len(points), _CHUNK):
        chunk = points[start : start + _CHUNK]
        s, r0 = _axis_coordinates(chunk, hole)
        out[start : start + _CHUNK] = hole.linear_density * hp_integral(
            r0, s, hole.charge_top, hole.charge_bottom, exponent, nodes
        )
    return out


def hole_ppv(
    points: Any, hole: Hole, constants: HPConstants | None = None, nodes: int = 32
) -> np.ndarray:
    """Holmberg-Persson PPV of one hole at the points, mm/s.

    Works for any hole direction: the distances are taken from the hole axis
    (``Hole.direction``). A point closer to the axis than the hole radius is evaluated at
    the hole wall. A hole without charge gives 0.
    """
    constants = constants or HPConstants()
    return constants.k * _hole_term(_points(points), hole, constants, nodes) ** constants.alpha


def ppv_field(
    points: Any,
    design: BlastDesign,
    constants: HPConstants | None = None,
    combine: str = "max",
    window_ms: float = 8.0,
    nodes: int = 32,
) -> np.ndarray:
    """Near-field PPV of the whole blast at the points, mm/s.

    Parameters
    ----------
    points : (n, 3) array_like
    design : BlastDesign
    constants : HPConstants, optional
        Default: the published uncalibrated constants.
    combine : {"max", "window"}
        ``"max"``: the largest single-hole PPV; holes are never summed.
        ``"window"``: holes whose firing times lie within ``window_ms`` of an earlier hole
        (``t_i <= t < t_i + window_ms``) act as one charge, ``PPV = K (sum q I)^alpha``; the
        result is the largest such group. Needs ``delay_ms`` on every charged hole.
    window_ms : float
        Width of the time window for ``"window"``.
    nodes : int
        Gauss-Legendre nodes per hole.
    """
    constants = constants or HPConstants()
    p = _points(points)
    holes = [h for h in design.holes if h.charge_length > 0]
    best = np.zeros(len(p))
    if combine == "max":
        for hole in holes:
            np.maximum(best, _hole_term(p, hole, constants, nodes), out=best)
    elif combine == "window":
        if any(not math.isfinite(h.delay_ms) for h in holes):
            raise ValueError('combine="window" needs delay_ms on every charged hole')
        holes.sort(key=lambda h: h.delay_ms)
        cache: dict[int, np.ndarray] = {}
        last_end = 0
        for i, first in enumerate(holes):
            end = i + 1
            while end < len(holes) and holes[end].delay_ms - first.delay_ms < window_ms:
                end += 1
            cache.pop(i - 1, None)
            if end == last_end:  # a subset of the previous group
                continue
            last_end = end
            total = np.zeros(len(p))
            for j in range(i, end):
                if j not in cache:
                    cache[j] = _hole_term(p, holes[j], constants, nodes)
                total += cache[j]
            np.maximum(best, total, out=best)
    else:
        raise ValueError(f"unknown combine {combine!r}")
    return constants.k * best**constants.alpha


def classify(ppv: Any, thresholds: DamageThresholds) -> np.ndarray:
    """Damage class of each PPV value: 0 intact, 1 damaged, 2 fractured, 3 crushed.

    A class starts strictly above its threshold. NaN gives -1 (unknown).
    """
    v = np.asarray(ppv, dtype=np.float64)
    cls = np.zeros(v.shape, dtype=np.int64)
    with np.errstate(invalid="ignore"):
        for limit in thresholds.as_list():
            cls += v > limit
    cls[np.isnan(v)] = -1
    return cls


def backbreak(
    design: BlastDesign,
    constants: HPConstants | None = None,
    thresholds: DamageThresholds | None = None,
    tol: float = 1e-3,
) -> float:
    """Depth of fracturing behind the last row, metres.

    The PPV is followed from the middle of the charge of every hole of the last row (the
    largest ``row``) horizontally away from the free face (against ``design.face_azimuth``);
    the back-break is the distance at which the largest of these PPV falls to the
    "fractured" threshold, found by bisection. 0 when the PPV at the hole wall is already
    below the threshold; NaN when no hole is charged.

    This is the reach of the PPV criterion only. Joints, gas penetration and the confinement
    of the last row, which control real back-break, are not in the model.
    """
    constants = constants or HPConstants()
    thresholds = thresholds or thresholds_for(design.rock)
    charged = [h for h in design.holes if h.charge_length > 0]
    if not charged:
        return math.nan
    last = max(h.row for h in charged)
    centres = np.array([h.charge_centre for h in charged if h.row == last])
    radius = min(0.5 * h.diameter for h in charged if h.row == last)
    az = math.radians(design.face_azimuth)
    back = np.array([-math.sin(az), -math.cos(az), 0.0])
    limit = thresholds.fractured_mm_s

    def excess(distance: float) -> float:
        ppv = ppv_field(centres + distance * back, design, constants)
        return float(ppv.max()) - limit

    lo = radius
    if excess(lo) <= 0:
        return 0.0
    hi = max(2.0 * lo, 1.0)
    while excess(hi) > 0:
        lo, hi = hi, 2.0 * hi
        if hi > 1e5:
            raise RuntimeError("back-break search did not find the end of the fractured zone")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        if excess(mid) > 0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# ---------------------------------------------------------------------------------------
# Blocks


def points_in_polygon(xy: Any, polygon: Any) -> np.ndarray:
    """True for the plan points inside a polygon (even-odd rule, any orientation).

    Parameters
    ----------
    xy : (n, 2+) array_like
        Only the first two columns are used.
    polygon : (m, 2) array_like
        Vertices in order, open or closed.
    """
    pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or len(poly) < 3:
        raise ValueError("a polygon needs at least 3 vertices")
    x, y = pts[:, 0], pts[:, 1]
    inside = np.zeros(len(pts), dtype=bool)
    x1, y1 = poly[:, 0], poly[:, 1]
    x2, y2 = np.roll(x1, -1), np.roll(y1, -1)
    for ax, ay, bx, by in zip(x1, y1, x2, y2, strict=True):
        if ay == by:
            continue
        crosses = (ay > y) != (by > y)
        with np.errstate(invalid="ignore"):
            x_cross = ax + (y - ay) * (bx - ax) / (by - ay)
        inside ^= crosses & (x < x_cross)
    return inside


def _segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance of the points from the segment ``a``-``b``."""
    ab = b - a
    length2 = float(ab @ ab)
    rel = points - a
    t = np.clip(rel @ ab / length2, 0.0, 1.0) if length2 > 0 else np.zeros(len(points))
    return np.linalg.norm(rel - t[:, None] * ab, axis=1)


def bench_top(xy: Any, design: BlastDesign) -> np.ndarray:
    """Top of the bench at plan points: the collar elevation of the nearest hole in plan."""
    if not design.holes:
        raise ValueError("the design has no holes")
    collars = np.array([h.collar for h in design.holes])
    pts = np.atleast_2d(np.asarray(xy, dtype=np.float64))[:, :2]
    _, nearest = cKDTree(collars[:, :2]).query(pts)
    return collars[nearest, 2]


def block_damage(
    centres: Any,
    sizes: Any,
    design: BlastDesign,
    constants: HPConstants | None = None,
    thresholds: DamageThresholds | None = None,
    combine: str = "max",
    window_ms: float = 8.0,
    cp: float | None = None,
) -> dict[str, np.ndarray]:
    """Damage attributes of blocks: the ``damage.blocks`` part of the result bundle.

    Parameters
    ----------
    centres, sizes : (n, 3) array_like
        Block centres and block sizes (dx, dy, dz). The PPV is evaluated at the centre.
    design : BlastDesign
        Needs a ``polygon``. Holes without ``delay_ms`` give NaN break times.
    constants, thresholds
        Default: uncalibrated Holmberg-Persson constants and :func:`thresholds_for` the
        design's rock.
    combine, window_ms
        See :func:`ppv_field`.
    cp : float, optional
        P-wave speed, m/s; default from the rock.

    Returns
    -------
    dict of arrays
        ``centre`` (n, 3), ``size`` (n, 3), ``ppv_mm_s``, ``cls`` (int, see
        :func:`classify`), ``blasted`` (bool: the centre is inside the polygon in plan and
        between ``floor_z`` and the bench top, the collar elevation of the nearest hole) and
        ``t_break_ms``: firing time of the nearest charge column plus the P-wave travel time
        from it, for blocks that are blasted or at least fractured; NaN for the others.
    """
    c = _points(centres)
    s = _points(sizes)
    if s.shape != c.shape:
        raise ValueError("centres and sizes must have the same shape")
    if len(design.polygon) < 3:
        raise ValueError("block damage needs the blast polygon of the design")
    constants = constants or HPConstants()
    thresholds = thresholds or thresholds_for(design.rock)
    if cp is None:
        cp, _ = wave_speeds(design.rock)

    ppv = ppv_field(c, design, constants, combine=combine, window_ms=window_ms)
    cls = classify(ppv, thresholds)
    top = bench_top(c, design)
    blasted = points_in_polygon(c, design.polygon) & (c[:, 2] >= design.floor_z) & (c[:, 2] <= top)

    distance = np.full(len(c), np.inf)
    t_fire = np.full(len(c), np.nan)
    for hole in design.holes:
        if not hole.charge_length > 0:
            continue
        d = _segment_distance(c, hole.point_at(hole.charge_top), hole.point_at(hole.charge_bottom))
        nearer = d < distance
        distance[nearer] = d[nearer]
        t_fire[nearer] = hole.delay_ms
    broken = blasted | (cls >= 2)
    t_break = np.where(broken, t_fire + distance / cp * 1000.0, np.nan)
    return {
        "centre": c,
        "size": s,
        "ppv_mm_s": ppv,
        "cls": cls,
        "blasted": blasted,
        "t_break_ms": t_break,
    }
