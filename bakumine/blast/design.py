"""Bench blast design: rules of thumb for the pattern, hole layout on a bench polygon, totals.

research_blast_design.md sections 1 to 4. Everything is SI (metres, kilograms, kg/m3); the
published imperial forms are converted in the text of the research note and here only the SI
form is computed.

Rules
-----
* :func:`burden_konya` -- Konya and Walter (1990): ``B = 0.012 (2 SGe / SGr + 1.5) De`` with
  ``B`` in metres and ``De`` in millimetres (the imperial ``B[ft] = (2 SGe / SGr + 1.5)
  De[in]`` times ``0.3048 / 25.4 = 0.012``).
* :func:`burden_ash` -- Ash (1963): ``B = Kb d``, ``Kb`` 20 to 40 (30 for average rock with
  an explosive of average density).
* :func:`burden_from_powder_factor` -- the burden that gives a target powder factor for a
  given hole diameter, explosive and bench height (closed-form root of a quadratic).
* :func:`suggest` -- burden by one of the three, then spacing, stemming (0.7 B), subdrill
  (0.3 B), stiffness ratio H / B with its warning, hole and charge lengths, charge per hole
  and powder factor.

Layout
------
:func:`layout` puts the holes on a bench polygon. Plan axes: ``f`` is the unit vector of
``face_azimuth`` (the direction the free face looks and the rock is thrown) and ``u`` is
``f`` turned 90 degrees clockwise, i.e. *left to right for somebody who stands behind the
blast and looks out over the face*. The face edge of the polygon is its extreme in the
``f`` direction. Row 0 is one burden behind it, every later row one burden further back; a
row that falls exactly on the back edge is kept (the back row defines the back limit of the
blast). Along a row the holes sit at ``s_min + S / 2 + k S`` (``s_min`` = left extreme of
the polygon); in a staggered pattern the odd rows are shifted by ``S / 2``. A hole is kept
when its collar is inside the polygon or on its boundary, so an odd staggered row of a
rectangle ``m S`` wide has ``m + 1`` holes (one on each side edge) and an even row ``m``.

Holes are named by row letter and position, ``A1`` = front row, leftmost hole.

Inclined holes (``hole_dip`` < 90, the angle below the horizontal) lie in the vertical plane
of the face azimuth and are parallel to an inclined face: the toe is displaced *toward the
face* by ``L cos(dip)``. Burden and spacing are laid out in plan at the collars; subdrill
and stemming are lengths along the hole. The hole length is
``(collar z - floor z) / sin(dip) + subdrill``.

A hole no longer than its stemming is kept as an **uncharged** hole (charge length 0) and
listed in ``design.notes["uncharged_holes"]``; a collar that the ``top`` surface cannot give
a height for, or a collar at or below the floor, drops the hole, and it is listed in
``design.notes["dropped_holes"]``.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from bakumine.blast.model import ANFO, BlastDesign, Explosive, Hole, Rock

__all__ = [
    "ASH_KB_RANGE",
    "STEMMING_RATIO",
    "SUBDRILL_RATIO",
    "Suggestion",
    "burden_ash",
    "burden_from_powder_factor",
    "burden_konya",
    "layout",
    "points_in_polygon",
    "spacing_ratio_for",
    "stiffness_warning",
    "suggest",
    "summary",
]

#: Ash (1963) burden ratio range, B = Kb * d.
ASH_KB_RANGE = (20.0, 40.0)
#: Stemming T = 0.7 B (Konya and Walter 1990; Ash's Kt = 0.7). Published range 0.7 to 1.0 B.
STEMMING_RATIO = 0.7
#: Subdrill J = 0.3 B (Konya and Walter 1990; Ash's Kj = 0.3).
SUBDRILL_RATIO = 0.3
#: Staggered: equilateral triangle, S / B = 2 / sqrt(3) = 1.1547, quoted as 1.15.
SPACING_RATIO_STAGGERED = 1.15
#: Square pattern: S = B. Langefors and Kihlstrom (1963) use S = 1.25 B (rectangular).
SPACING_RATIO_SQUARE = 1.0
SPACING_RATIO_SQUARE_RANGE = (1.0, 1.25)

_WATER_DENSITY = 1000.0  # kg/m3, reference of a specific gravity
_PATTERNS = ("staggered", "square")


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not (math.isfinite(value) and value > 0):
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return value


def _check_pattern(pattern: str) -> str:
    if pattern not in _PATTERNS:
        raise ValueError(f"pattern must be one of {_PATTERNS}, got {pattern!r}")
    return pattern


# ---------------------------------------------------------------------------------- rules


def burden_konya(diameter: float, rock_density: float, explosive_density: float) -> float:
    """Burden by Konya and Walter (1990), SI form.

    ``B = 0.012 (2 SGe / SGr + 1.5) De`` with ``De`` the explosive (hole) diameter in
    millimetres and ``SGe``, ``SGr`` the specific gravities of explosive and rock.

    Parameters
    ----------
    diameter : float
        Diameter of the explosive column, metres.
    rock_density, explosive_density : float
        kg/m3.

    Returns
    -------
    float
        Burden in metres.
    """
    d_mm = _positive("diameter", diameter) * 1000.0
    sg_e = _positive("explosive_density", explosive_density) / _WATER_DENSITY
    sg_r = _positive("rock_density", rock_density) / _WATER_DENSITY
    return 0.012 * (2.0 * sg_e / sg_r + 1.5) * d_mm


def burden_ash(diameter: float, kb: float = 30.0) -> float:
    """Burden by Ash (1963): ``B = Kb d`` (same length unit for both).

    ``kb`` must lie in the published range 20 to 40 (:data:`ASH_KB_RANGE`); 30 is Ash's
    value for average rock (density about 2.7) and an explosive of density about 1.2-1.3,
    25 for a light explosive (ANFO) and 35 for a dense one in the same rock.
    """
    kb = float(kb)
    if not ASH_KB_RANGE[0] <= kb <= ASH_KB_RANGE[1]:
        raise ValueError(f"Ash burden ratio Kb must be in {ASH_KB_RANGE}, got {kb!r}")
    return kb * _positive("diameter", diameter)


def burden_from_powder_factor(
    powder_factor: float,
    bench_height: float,
    diameter: float,
    explosive: Explosive,
    spacing_ratio: float = SPACING_RATIO_STAGGERED,
    stemming_ratio: float = STEMMING_RATIO,
    subdrill_ratio: float = SUBDRILL_RATIO,
) -> float:
    """Burden that gives a target powder factor.

    With ``S = m B``, stemming ``T = kt B``, subdrill ``J = kj B`` and the linear charge
    density ``w = pi/4 d^2 rho_e`` the powder factor of one hole is

    ``q = w (H + (kj - kt) B) / (m B^2 H)``,

    so ``B`` is the positive root of ``q m H B^2 - w (kj - kt) B - w H = 0``.

    Parameters
    ----------
    powder_factor : float
        Target, kg of explosive per m3 of in-situ rock.
    bench_height, diameter : float
        Metres.

    Returns
    -------
    float
        Burden in metres.
    """
    q = _positive("powder_factor", powder_factor)
    h = _positive("bench_height", bench_height)
    m = _positive("spacing_ratio", spacing_ratio)
    w = math.pi / 4.0 * _positive("diameter", diameter) ** 2 * explosive.density
    k = float(subdrill_ratio) - float(stemming_ratio)
    a, b, c = q * m * h, -w * k, -w * h
    return (-b + math.sqrt(b * b - 4.0 * a * c)) / (2.0 * a)


def spacing_ratio_for(pattern: str) -> float:
    """Default S / B: 1.15 staggered (equilateral triangle), 1.0 square."""
    _check_pattern(pattern)
    return SPACING_RATIO_STAGGERED if pattern == "staggered" else SPACING_RATIO_SQUARE


def stiffness_warning(stiffness_ratio: float) -> str | None:
    """Text for a stiffness ratio H / B outside 2 to 4, else ``None``.

    Konya and Walter (1990): H / B = 1 poor (severe airblast, flyrock, vibration, coarse
    fragmentation; "redesign"), 2 fair, 3 good, 4 excellent; above 4 no further benefit.
    """
    if not math.isfinite(stiffness_ratio):
        return None
    if stiffness_ratio < 2.0:
        return (
            f"stiffness ratio H/B = {stiffness_ratio:.2f} < 2: stiff bench, poor fragmentation "
            "with more airblast, flyrock and vibration (Konya and Walter 1990); use a smaller "
            "hole diameter and burden or a higher bench"
        )
    if stiffness_ratio > 4.0:
        return (
            f"stiffness ratio H/B = {stiffness_ratio:.2f} > 4: no further benefit above 4 "
            "(Konya and Walter 1990); a larger hole diameter and burden would break the same "
            "bench with less drilling (check hole deviation at this height)"
        )
    return None


@dataclass(frozen=True)
class Suggestion:
    """Pattern proposed by :func:`suggest`. Lengths in metres, masses in kilograms."""

    method: str
    pattern: str
    diameter: float
    bench_height: float
    burden: float
    spacing: float
    stemming: float
    subdrill: float
    stiffness_ratio: float
    hole_length: float
    charge_length: float
    charge_per_hole: float
    powder_factor: float  # kg/m3
    powder_factor_kg_t: float
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["warnings"] = list(self.warnings)
        return out


def suggest(
    diameter: float,
    bench_height: float,
    rock: Rock,
    explosive: Explosive,
    pattern: str = "staggered",
    method: str = "konya",
    *,
    kb: float = 30.0,
    powder_factor: float | None = None,
    spacing_ratio: float | None = None,
    stemming_ratio: float = STEMMING_RATIO,
    subdrill_ratio: float = SUBDRILL_RATIO,
) -> Suggestion:
    """Propose a pattern for vertical holes from published rules of thumb.

    Parameters
    ----------
    diameter, bench_height : float
        Hole diameter and bench height, metres.
    pattern : {"staggered", "square"}
    method : {"konya", "ash", "powder_factor"}
        Burden rule: :func:`burden_konya`, :func:`burden_ash` (uses ``kb``) or
        :func:`burden_from_powder_factor` (needs ``powder_factor`` in kg/m3).
    spacing_ratio : float, optional
        S / B. Default 1.15 staggered, 1.0 square (published range for square or
        rectangular patterns 1.0 to 1.25).
    stemming_ratio, subdrill_ratio : float
        T / B (published 0.7 to 1.0, default 0.7) and J / B (default 0.3).

    Returns
    -------
    Suggestion
        ``warnings`` holds the stiffness-ratio text and a note when the stemming leaves no
        room for a charge.
    """
    d = _positive("diameter", diameter)
    h = _positive("bench_height", bench_height)
    _check_pattern(pattern)
    m = (
        spacing_ratio_for(pattern)
        if spacing_ratio is None
        else _positive("spacing_ratio", spacing_ratio)
    )
    if stemming_ratio < 0 or subdrill_ratio < 0:
        raise ValueError("stemming_ratio and subdrill_ratio must be >= 0")

    if method == "konya":
        burden = burden_konya(d, rock.density, explosive.density)
    elif method == "ash":
        burden = burden_ash(d, kb)
    elif method == "powder_factor":
        if powder_factor is None:
            raise ValueError("method 'powder_factor' needs powder_factor (kg/m3)")
        burden = burden_from_powder_factor(
            powder_factor, h, d, explosive, m, stemming_ratio, subdrill_ratio
        )
    else:
        raise ValueError(f"method must be 'konya', 'ash' or 'powder_factor', got {method!r}")

    spacing = m * burden
    stemming = stemming_ratio * burden
    subdrill = subdrill_ratio * burden
    hole_length = h + subdrill
    warnings: list[str] = []
    charge_length = hole_length - stemming
    if charge_length <= 0:
        warnings.append(
            f"stemming {stemming:.2f} m is not shorter than the hole ({hole_length:.2f} m): "
            "no room for a charge"
        )
        charge_length = 0.0
    charge = math.pi / 4.0 * d**2 * explosive.density * charge_length
    volume = burden * spacing * h
    stiffness = h / burden
    text = stiffness_warning(stiffness)
    if text:
        warnings.append(text)
    return Suggestion(
        method=method,
        pattern=pattern,
        diameter=d,
        bench_height=h,
        burden=burden,
        spacing=spacing,
        stemming=stemming,
        subdrill=subdrill,
        stiffness_ratio=stiffness,
        hole_length=hole_length,
        charge_length=charge_length,
        charge_per_hole=charge,
        powder_factor=charge / volume,
        powder_factor_kg_t=charge / (volume * rock.density / 1000.0),
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------------- layout


def points_in_polygon(
    x: np.ndarray, y: np.ndarray, polygon: Sequence[Sequence[float]], tol: float = 1e-9
) -> np.ndarray:
    """Which points lie inside a simple polygon or within ``tol`` of its boundary.

    Even-odd ray casting (a ray toward +x crosses the boundary an odd number of times for
    an inside point), plus a distance test against every edge so that boundary points count
    as inside whatever the rounding.
    """
    poly = np.asarray(polygon, dtype=np.float64)[:, :2]
    px = np.asarray(x, dtype=np.float64).ravel()
    py = np.asarray(y, dtype=np.float64).ravel()
    inside = np.zeros(px.shape, dtype=bool)
    on_edge = np.zeros(px.shape, dtype=bool)
    for (x1, y1), (x2, y2) in zip(poly, np.roll(poly, -1, axis=0), strict=True):
        if (y1 > y2) or (y1 == y2 and x1 > x2):
            x1, y1, x2, y2 = x2, y2, x1, y1
        dx, dy = x2 - x1, y2 - y1
        if dy != 0.0:
            crosses = (py >= y1) & (py < y2)
            x_cross = x1 + (py - y1) * dx / dy
            inside ^= crosses & (px < x_cross)
        length2 = dx * dx + dy * dy
        if length2 > 0.0:
            t = np.clip(((px - x1) * dx + (py - y1) * dy) / length2, 0.0, 1.0)
        else:
            t = np.zeros(px.shape)
        on_edge |= np.hypot(px - (x1 + t * dx), py - (y1 + t * dy)) <= tol
    return (inside | on_edge).reshape(np.shape(x))


def _row_letters(row: int) -> str:
    """0 -> A, 25 -> Z, 26 -> AA (spreadsheet column style)."""
    out = ""
    n = row + 1
    while n > 0:
        n, rem = divmod(n - 1, 26)
        out = chr(ord("A") + rem) + out
    return out


def face_axes(face_azimuth: float) -> tuple[np.ndarray, np.ndarray]:
    """Plan unit vectors ``(u, f)``: ``f`` along the face azimuth, ``u`` = ``f`` turned 90
    degrees clockwise (left to right when looking out over the face)."""
    az = math.radians(float(face_azimuth))
    f = np.array([math.sin(az), math.cos(az)])
    u = np.array([math.cos(az), -math.sin(az)])
    return u, f


def layout(
    polygon: Sequence[Sequence[float]],
    top: Any,
    floor_z: float,
    burden: float,
    spacing: float,
    pattern: str,
    face_azimuth: float,
    diameter: float,
    explosive: Explosive = ANFO,
    stemming: float | None = None,
    subdrill: float | None = None,
    rock: Rock | None = None,
    hole_dip: float = 90.0,
    name: str = "blast",
) -> BlastDesign:
    """Lay a drill pattern on a bench polygon (conventions in the module docstring).

    Parameters
    ----------
    polygon : sequence of (x, y)
        Plan outline of the block to blast, at least three vertices, not self-crossing.
    top : float or object with ``z_at(x, y)``
        Bench top: one elevation, or a surface (e.g. a
        :class:`~bakumine.core.geometry.dtm.DTM`) that gives the collar elevations.
    floor_z : float
        Bench floor (design grade).
    burden, spacing : float
        Metres, in plan.
    pattern : {"staggered", "square"}
    face_azimuth : float
        Direction the free face looks, degrees clockwise from north.
    diameter : float
        Hole diameter, metres.
    stemming, subdrill : float, optional
        Metres along the hole. ``None`` = 0.7 B and 0.3 B (recorded in the notes).
    hole_dip : float
        Angle of the holes below the horizontal, degrees; 90 = vertical.

    Returns
    -------
    BlastDesign
        ``bench_height`` is ``top - floor_z`` for a flat top, the mean of
        ``collar z - floor_z`` for a surface. ``notes["layout"]`` records the inputs,
        ``notes["uncharged_holes"]`` and ``notes["dropped_holes"]`` the exceptions.
    """
    b = _positive("burden", burden)
    s = _positive("spacing", spacing)
    d = _positive("diameter", diameter)
    _check_pattern(pattern)
    floor_z = float(floor_z)
    dip = float(hole_dip)
    if not 0.0 < dip <= 90.0:
        raise ValueError(f"hole_dip must be in (0, 90] degrees, got {hole_dip!r}")
    poly = np.asarray(polygon, dtype=np.float64)
    if poly.ndim != 2 or poly.shape[0] < 3 or poly.shape[1] < 2:
        raise ValueError("polygon needs at least three (x, y) vertices")
    poly = poly[:, :2]
    if not np.isfinite(poly).all():
        raise ValueError("polygon has non-finite coordinates")
    stem = STEMMING_RATIO * b if stemming is None else float(stemming)
    sub = SUBDRILL_RATIO * b if subdrill is None else float(subdrill)
    if stem < 0 or sub < 0:
        raise ValueError("stemming and subdrill must be >= 0")
    rock = Rock() if rock is None else rock

    u, f = face_axes(face_azimuth)
    ps, pt = poly @ u, poly @ f
    s_min, s_max, t_min, t_max = ps.min(), ps.max(), pt.min(), pt.max()
    tol = 1e-6
    n_rows = int(math.floor((t_max - t_min) / b + tol))

    cand: list[tuple[int, float, float]] = []  # row, s, t
    for row in range(n_rows):
        offset = 0.5 if (pattern == "staggered" and row % 2 == 1) else 0.0
        start = s_min + (0.5 + offset) * s
        k_lo = int(math.floor((s_min - start) / s)) - 1
        k_hi = int(math.ceil((s_max - start) / s)) + 1
        for k in range(k_lo, k_hi + 1):
            cand.append((row, start + k * s, t_max - (row + 1) * b))
    if not cand:
        raise ValueError("the polygon is shallower than one burden: no row fits")
    rows = np.array([c[0] for c in cand])
    cs = np.array([c[1] for c in cand])
    ct = np.array([c[2] for c in cand])
    keep = points_in_polygon(cs, ct, np.column_stack([ps, pt]), tol=tol)
    rows, cs, ct = rows[keep], cs[keep], ct[keep]
    x = cs * u[0] + ct * f[0]
    y = cs * u[1] + ct * f[1]

    flat = not hasattr(top, "z_at")
    if flat:
        zc = np.full(x.shape, float(top))
    else:
        zc = np.asarray(top.z_at(x, y), dtype=np.float64).reshape(x.shape)

    sin_d, cos_d = math.sin(math.radians(dip)), math.cos(math.radians(dip))
    if dip == 90.0:
        sin_d, cos_d = 1.0, 0.0
    holes: list[Hole] = []
    uncharged: list[str] = []
    dropped: list[dict[str, Any]] = []
    order = np.lexsort((cs, rows))
    counters: dict[int, int] = {}
    for i in order:
        row = int(rows[i])
        if not math.isfinite(zc[i]):
            dropped.append({"row": row, "xy": [float(x[i]), float(y[i])], "reason": "no top"})
            continue
        if zc[i] <= floor_z:
            dropped.append(
                {"row": row, "xy": [float(x[i]), float(y[i])], "reason": "collar not above floor"}
            )
            continue
        index = counters.get(row, 0)
        counters[row] = index + 1
        length = (zc[i] - floor_z) / sin_d + sub
        toe = (
            x[i] + length * cos_d * f[0],
            y[i] + length * cos_d * f[1],
            zc[i] - length * sin_d,
        )
        hole_id = f"{_row_letters(row)}{index + 1}"
        charge_top = min(stem, length)
        if stem >= length:
            uncharged.append(hole_id)
        holes.append(
            Hole(
                id=hole_id,
                collar=(x[i], y[i], zc[i]),
                toe=toe,
                diameter=d,
                charge_top=charge_top,
                charge_bottom=length,
                explosive=explosive,
                row=row,
                index=index,
                subdrill=sub,
                burden=b,
                spacing=s,
            )
        )
    if not holes:
        raise ValueError("no hole fits: check the polygon, burden, spacing and the top surface")

    if flat:
        bench_height = float(top) - floor_z
    else:
        bench_height = float(np.mean([h.collar[2] for h in holes])) - floor_z
    notes: dict[str, Any] = {
        "layout": {
            "stemming": stem,
            "subdrill": sub,
            "hole_dip": dip,
            "diameter": d,
            "explosive": explosive.name,
            "stemming_rule": "0.7 B" if stemming is None else "given",
            "subdrill_rule": "0.3 B" if subdrill is None else "given",
            "bench_height": "top - floor" if flat else "mean collar z - floor",
        }
    }
    if uncharged:
        notes["uncharged_holes"] = uncharged
    if dropped:
        notes["dropped_holes"] = dropped
    return BlastDesign(
        name=name,
        holes=holes,
        rock=rock,
        bench_height=bench_height,
        floor_z=floor_z,
        burden=b,
        spacing=s,
        pattern=pattern,
        face_azimuth=float(face_azimuth),
        polygon=[(float(px), float(py)) for px, py in poly],
        notes=notes,
    )


# -------------------------------------------------------------------------------- summary


def summary(design: BlastDesign) -> dict[str, Any]:
    """Totals of a design.

    ``volume_m3`` is the in-situ volume of :attr:`BlastDesign.volume` (polygon area x bench
    height), ``tonnes`` that volume times the rock density, ``powder_factor_kg_m3`` and
    ``powder_factor_kg_t`` the explosive mass per m3 and per tonne,
    ``specific_drilling_m_m3`` the drilled metres per m3 and ``stiffness_ratio`` the bench
    height over the design burden (NaN when the design has no burden).
    """
    volume = float(design.volume)
    tonnes = volume * design.rock.density / 1000.0
    explosive = design.explosive_mass
    drilled = design.drilled_length
    ok = math.isfinite(volume) and volume > 0
    burden = design.burden
    return {
        "holes": len(design.holes),
        "rows": len({h.row for h in design.holes}),
        "drilled_m": drilled,
        "explosive_kg": explosive,
        "volume_m3": volume,
        "tonnes": tonnes,
        "powder_factor_kg_m3": explosive / volume if ok else math.nan,
        "powder_factor_kg_t": explosive / tonnes if ok else math.nan,
        "specific_drilling_m_m3": drilled / volume if ok else math.nan,
        "stiffness_ratio": (
            design.bench_height / burden if math.isfinite(burden) and burden > 0 else math.nan
        ),
    }
