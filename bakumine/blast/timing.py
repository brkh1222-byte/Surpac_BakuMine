"""Initiation timing of a bench blast: delay patterns, firing order, charge per delay.

research_blast_design.md sections 5 to 7. Times are milliseconds after the first initiation
(``Hole.delay_ms``); the first hole of every pattern fires at 0.

Each pattern function returns a **new** :class:`~bakumine.blast.model.BlastDesign` (the
input is not changed) and records itself, with the id of the initiation hole, in
``notes["timing"]``. Rows are the ``Hole.row`` numbers (lowest = next to the free face),
``r`` = 0, 1, 2 ... from the face. The position of a hole along its row comes from its
collar: ``s = collar . u`` with ``u`` the left-to-right axis of
:func:`bakumine.blast.design.face_axes`. Every pattern is

``t = n Th + r Tr``

with ``Th`` the inter-hole and ``Tr`` the inter-row delay and ``n`` the number of hole
spacings ``S`` along the row between the hole and the initiation hole (a front-row hole at
``s0``):

* :func:`row_by_row` -- with ``Th = 0`` (the default) a whole row fires at once; with
  ``Th > 0`` it is the echelon from the left.
* :func:`echelon` -- initiation at a front corner (``start`` = ``"left"`` or ``"right"``).
  With ``d`` the distance from the corner hole into the pattern, ``n = ceil(d / S)``; for a
  hole outside the corner hole (``d < 0``, the edge hole of a staggered row, a bench that
  widens to the back) ``n = floor(|d| / S)``.
* :func:`v_cut` -- initiation at the centre of the front row, ``n = floor(|s - s0| / S)``,
  the same on both sides.
* :func:`hole_by_hole` -- the echelon from the left, but it must give every hole its own
  firing time; delays that make two holes coincide are refused.

In a square pattern ``n`` is the column and a hole fires ``Tr`` after the hole straight in
front of it. In a staggered pattern a hole has two holes in front, half a spacing to either
side: the rounding above makes the one nearer the initiation hole fire first for every
``Th`` and ``Tr > 0``, and both when ``Tr > Th``. A row never starts before the row in
front of it.

:func:`max_instantaneous_charge` is the largest explosive mass that fires within any
half-open window ``[t, t + 8 ms)`` -- the "charge per delay" of the scaled-distance
vibration laws (Duvall et al. 1963, USBM RI 6151; 30 CFR 816.67).
:func:`burden_relief` is the row-to-row delay per metre of burden, compared with the range
of Konya and Walter (1990).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from bakumine.blast.design import face_axes
from bakumine.blast.model import BlastDesign, Hole

__all__ = [
    "RELIEF_GUIDE_MS_PER_M",
    "BurdenRelief",
    "burden_relief",
    "duration_ms",
    "echelon",
    "firing_order",
    "hole_by_hole",
    "max_instantaneous_charge",
    "row_by_row",
    "v_cut",
]

#: Row-to-row delay per metre of burden, Konya and Walter (1990): 6.5 ms/m = violent,
#: excessive airblast and backbreak ... 16.5 ms/m = scattered pile, minimum backbreak.
RELIEF_GUIDE_MS_PER_M = (6.5, 16.5)


def _delay(name: str, value: float, *, strict: bool) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0 or (strict and value == 0):
        bound = "> 0" if strict else ">= 0"
        raise ValueError(f"{name} must be {bound}, got {value!r}")
    return value


def _along(design: BlastDesign) -> np.ndarray:
    """Along-row coordinate ``s`` of every hole collar."""
    u, _ = face_axes(design.face_azimuth)
    return np.array([h.collar[0] * u[0] + h.collar[1] * u[1] for h in design.holes])


def _row_numbers(design: BlastDesign) -> np.ndarray:
    """Row of every hole as 0, 1, 2 ... from the face (ranks of the distinct ``Hole.row``)."""
    rows = np.array([h.row for h in design.holes])
    return np.searchsorted(np.unique(rows), rows)


def _front_row(design: BlastDesign, s: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Indices of the front-row holes, left to right."""
    if not design.holes:
        raise ValueError("the design has no holes")
    front = np.flatnonzero(rows == 0)
    return front[np.argsort(s[front], kind="stable")]


def _spacing(design: BlastDesign) -> float:
    spacing = design.spacing
    if not (math.isfinite(spacing) and spacing > 0):
        raise ValueError("timing needs the design spacing")
    return spacing


def _sequence(
    design: BlastDesign, th: float, tr: float, start: str, scheme: dict[str, Any]
) -> BlastDesign:
    """``t = n Th + r Tr`` from the ``start`` hole ("left", "right" or "centre")."""
    s = _along(design)
    rows = _row_numbers(design)
    front = _front_row(design, s, rows)
    spacing = _spacing(design)
    eps = 1e-6
    if start == "centre":
        gap = np.abs(s[front] - 0.5 * (s[front[0]] + s[front[-1]]))
        first = front[int(np.flatnonzero(gap <= gap.min() + 1e-9)[0])]
        steps = np.floor(np.abs(s - s[first]) / spacing + eps)
    else:
        first = front[0] if start == "left" else front[-1]
        d = (s - s[first]) / spacing * (1.0 if start == "left" else -1.0)
        steps = np.where(d >= -eps, np.ceil(d - eps), np.floor(-d + eps))
    times = steps * th + rows * tr
    times = times - times.min()
    holes = [replace(h, delay_ms=float(t)) for h, t in zip(design.holes, times, strict=True)]
    scheme = {**scheme, "initiation_hole": design.holes[int(first)].id}
    return replace(design, holes=holes, notes={**design.notes, "timing": scheme})


def row_by_row(design: BlastDesign, inter_row_ms: float, inter_hole_ms: float = 0.0) -> BlastDesign:
    """Row after row: ``t = r * inter_row_ms + n * inter_hole_ms`` (n from the left).

    With the default ``inter_hole_ms = 0`` all holes of a row fire together.
    """
    tr = _delay("inter_row_ms", inter_row_ms, strict=True)
    th = _delay("inter_hole_ms", inter_hole_ms, strict=False)
    scheme = {"scheme": "row_by_row", "inter_row_ms": tr, "inter_hole_ms": th}
    return _sequence(design, th, tr, "left", scheme)


def echelon(
    design: BlastDesign, inter_hole_ms: float, inter_row_ms: float, start: str = "left"
) -> BlastDesign:
    """Echelon from a front corner: ``t = n * inter_hole_ms + r * inter_row_ms``.

    ``start`` is the corner of the front row that fires first, ``"left"`` or ``"right"``
    for somebody who stands behind the blast and looks out over the face; ``n`` is the
    number of spacings from that corner hole (module docstring).
    """
    if start not in ("left", "right"):
        raise ValueError(f"start must be 'left' or 'right', got {start!r}")
    th = _delay("inter_hole_ms", inter_hole_ms, strict=False)
    tr = _delay("inter_row_ms", inter_row_ms, strict=True)
    scheme = {"scheme": "echelon", "inter_hole_ms": th, "inter_row_ms": tr, "start": start}
    return _sequence(design, th, tr, start, scheme)


def v_cut(design: BlastDesign, inter_hole_ms: float, inter_row_ms: float) -> BlastDesign:
    """V pattern: initiation at the centre of the front row.

    The initiation hole is the front-row hole nearest the middle of the front row (the left
    one of two equally near). ``t = n * inter_hole_ms + r * inter_row_ms`` with
    ``n = floor(|s - s0| / S)``, ``S`` the design spacing.
    """
    th = _delay("inter_hole_ms", inter_hole_ms, strict=False)
    tr = _delay("inter_row_ms", inter_row_ms, strict=True)
    scheme = {"scheme": "v_cut", "inter_hole_ms": th, "inter_row_ms": tr}
    return _sequence(design, th, tr, "centre", scheme)


def hole_by_hole(design: BlastDesign, inter_hole_ms: float, inter_row_ms: float) -> BlastDesign:
    """Every hole on its own delay: the echelon from the left, refused with ``ValueError``
    when two holes would fire at the same time (e.g. ``inter_row_ms`` a small multiple of
    ``inter_hole_ms``)."""
    th = _delay("inter_hole_ms", inter_hole_ms, strict=True)
    tr = _delay("inter_row_ms", inter_row_ms, strict=True)
    scheme = {"scheme": "hole_by_hole", "inter_hole_ms": th, "inter_row_ms": tr}
    timed = _sequence(design, th, tr, "left", scheme)
    ordered = np.sort([h.delay_ms for h in timed.holes])
    if ordered.size > 1 and np.min(np.diff(ordered)) < 1e-9:
        raise ValueError(
            f"inter_hole_ms = {th:g} and inter_row_ms = {tr:g} fire two holes at the same "
            "time; choose delays such that n * inter_hole_ms never equals r * inter_row_ms "
            "within the pattern (e.g. 17 and 42 ms)"
        )
    return timed


# ------------------------------------------------------------------------------- analysis


def _times(design: BlastDesign) -> np.ndarray:
    if not design.holes:
        raise ValueError("the design has no holes")
    if not design.timed:
        raise ValueError("the design is not timed: some holes have no delay_ms")
    return np.array([h.delay_ms for h in design.holes], dtype=np.float64)


def firing_order(design: BlastDesign) -> list[str]:
    """Hole ids in firing order (ties by row, then by position along the row)."""
    _times(design)
    holes: list[Hole] = sorted(design.holes, key=lambda h: (h.delay_ms, h.row, h.index))
    return [h.id for h in holes]


def duration_ms(design: BlastDesign) -> float:
    """Time from the first to the last initiation, milliseconds."""
    times = _times(design)
    return float(times.max() - times.min())


def max_instantaneous_charge(
    design: BlastDesign, window_ms: float = 8.0
) -> tuple[float, list[str]]:
    """Largest explosive mass firing within any window ``[t, t + window_ms)``.

    Charges separated by less than 8 ms are taken to act together in the scaled-distance
    vibration laws; charges 8 ms or more apart are separate delays (hence the half-open
    window). ``window_ms = 0`` counts only holes with the same firing time.

    Returns
    -------
    (kg, ids)
        The mass and the holes of the heaviest window, in firing order (the earliest such
        window when several are equally heavy).
    """
    window = _delay("window_ms", window_ms, strict=False)
    times = _times(design)
    mass = np.array([h.charge_mass for h in design.holes])
    order = np.argsort(times, kind="stable")
    t, m = times[order], mass[order]
    cumulative = np.concatenate([[0.0], np.cumsum(m)])
    if window > 0:
        end = np.searchsorted(t, t + window, side="left")  # first index with t >= start + w
    else:
        end = np.searchsorted(t, t, side="right")
    sums = cumulative[end] - cumulative[np.arange(t.size)]
    best = int(np.argmax(sums))  # first maximum = earliest window
    ids = [design.holes[int(i)].id for i in order[best : end[best]]]
    return float(sums[best]), ids


@dataclass(frozen=True)
class BurdenRelief:
    """Row-to-row delay per metre of burden (:func:`burden_relief`).

    ``ms_per_m`` is the median over the pairs of consecutive rows (NaN for a single row or
    an unknown burden), ``per_row`` the value of each pair, ``guide_ms_per_m`` the published
    range and ``flags`` the text of the comparison.
    """

    ms_per_m: float
    per_row: tuple[float, ...]
    guide_ms_per_m: tuple[float, float]
    flags: tuple[str, ...]


def burden_relief(design: BlastDesign) -> BurdenRelief:
    """Burden relief: (first firing time of a row - that of the row in front) / burden.

    Compared with Konya and Walter (1990): below 6.5 ms/m the blast is violent with
    excessive airblast and backbreak, 8 ms/m piles high close to the face, 11.5 ms/m is
    average, 16.5 ms/m gives a scattered pile with minimum backbreak.
    """
    times = _times(design)
    rows = _row_numbers(design)
    lo, hi = RELIEF_GUIDE_MS_PER_M
    burden = design.burden
    if rows.max() == 0:
        return BurdenRelief(math.nan, (), (lo, hi), ("single row: no burden relief between rows",))
    if not (math.isfinite(burden) and burden > 0):
        return BurdenRelief(math.nan, (), (lo, hi), ("design burden unknown",))
    first = np.array([times[rows == r].min() for r in range(int(rows.max()) + 1)])
    per_row = np.diff(first) / burden
    value = float(np.median(per_row))
    flags: list[str] = []
    if np.any(per_row <= 0):
        flags.append("a row fires no later than the row in front of it: no relief")
    if value < lo:
        flags.append(
            f"burden relief {value:.1f} ms/m < {lo} ms/m: violent blast, excessive airblast "
            "and backbreak (Konya and Walter 1990)"
        )
    elif value > hi:
        flags.append(
            f"burden relief {value:.1f} ms/m > {hi} ms/m: longer than the guide range of "
            "Konya and Walter (1990) (16.5 ms/m = scattered pile, minimum backbreak)"
        )
    else:
        flags.append(
            f"burden relief {value:.1f} ms/m within the guide range {lo} to {hi} ms/m "
            "(Konya and Walter 1990)"
        )
    return BurdenRelief(value, tuple(float(v) for v in per_row), (lo, hi), tuple(flags))
