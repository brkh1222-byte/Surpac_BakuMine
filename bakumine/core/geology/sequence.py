"""Layer stacks as one stratigraphic field, and intrusions that overprint them.

Theory: research.md 5.7.3. Signatures: geology_contracts.md section 7.

**Sequence** (:func:`fit_sequence`). For a stack whose order is known (top to bottom
``U_0 .. U_{m-1}``) one scalar field, the stratigraphic position ``phi``, replaces the
``m`` signed-distance fields of the largest-field rule. ``phi`` grows downwards through the
stack; the contact ``U_k / U_{k+1}`` is the level ``T_k`` (``T_0 = 0``,
``T_k = t_1 + .. + t_k`` with ``t_k`` the median true thickness of ``U_k``). Layers cannot
cross or leave gaps. This is the "value constraint" form of a potential field (Lajaunie et
al. 1997; Calcagno et al. 2008; Hillier et al. 2014; references in research.md).

**Intrusions** (:func:`fit_intrusion`). An intrusion is its own field, positive inside, and
overprints what the earlier events gave. Two shapes:

``body``
    the binary signed-distance field of the unit against every other known unit
    (:func:`~bakumine.core.geometry.implicit.extract_constraints` and ``fit_field``).
    Right for stocks and plugs, whose intercepts are long compared with the hole spacing.

``sheet``
    for dykes, sills and veins: bodies much thinner than the hole spacing. A distance
    field between holes 50 m apart cannot connect 8 m intercepts and models them as blobs
    at the holes. The sheet is built from what the intercepts do determine:

    1. a *reference surface* through the midpoints of the complete intercepts (both
       contacts logged): the least-squares plane (principal components), plus the
       midpoints' residuals from it interpolated in the plane, so a gently curved sheet
       is followed;
    2. a *thickness* in the plane: each complete intercept gives the true thickness
       ``L |cos theta|`` (``L`` its length, ``theta`` the angle between the hole and the
       pole of the plane), interpolated in the plane and kept inside the measured range;
    3. an *extent* in the plane: the sheet continues ``reach`` metres beyond the nearest
       intercept and stops half-way towards a hole that crosses the reference surface in
       known rock without the unit.

    The field is ``f = min(thickness / 2 - |w - w_0(u, v)|, e(u, v))`` with ``w`` the
    distance from the plane, ``w_0`` the reference surface and ``e`` the extent margin.

    This is the construction commercial implicit modellers describe for veins (a
    reference surface through the midpoints of the vein intervals, hanging-wall and
    footwall offsets, pinch-out at holes without the vein: Seequent, Leapfrog Geo help,
    "Vein systems" **[unverified]**, written from memory; Cowan et al. 2003 for the
    implicit setting). True thickness from intercept length and the hole-to-pole angle is
    textbook practice (Sinclair & Blackwell 2002, **[unverified]** page). Stopping half-way
    to the next barren hole is the rule of sectional and polygonal interpretation (Popoff
    1966, USBM IC 8283 **[unverified]**). The numbers are this project's defaults and are
    written to the provenance record as assumptions: :data:`MIN_SHEET_HOLES`,
    :data:`SHEET_MAX_FLATNESS`, :data:`SHEET_MAX_THICKNESS_TO_SPACING`,
    :data:`SHEET_REACH_FACTOR`.

    Limits. One sheet per unit code: two dykes logged with one code do not lie near one
    plane, fail the flatness test and are modelled as a ``body``. The thickness is
    measured along the pole of the plane, not of the curved surface (exact for a planar
    sheet). On a sampled grid a sheet thinner than about one node spacing is lost.

``shape="auto"`` (the default) takes ``sheet`` when at least :data:`MIN_SHEET_HOLES` holes
have a complete intercept, the midpoints lie near one plane and the median true thickness
is at most :data:`SHEET_MAX_THICKNESS_TO_SPACING` of the hole spacing; otherwise ``body``.
A ``sheet`` that was asked for but is not determined falls back to ``body``; the
provenance says so.

Events are applied in age order, the oldest first, so a later one wins
(:class:`~bakumine.core.geology.lithology.LithologyStack`). :func:`order_intrusions` puts
the intrusions in the order given, or reports the age order as a missing input.

Coordinates are x = east, y = north, z = up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geology.lithology import (
    FieldSettings,
    _auto_anisotropy,
    _contacts_of,
    _hole_spacing,
    _runs_of,
    _unit_table,
)
from bakumine.core.geology.provenance import ProvenanceEntry
from bakumine.core.geometry.implicit import (
    KERNELS,
    Constraints,
    ImplicitField,
    estimate_intercept_anisotropy,
    extract_constraints,
    fit_field,
)

__all__ = [
    "INTRUSION_SHAPES",
    "MIN_SHEET_HOLES",
    "SHEET_MAX_FLATNESS",
    "SHEET_MAX_THICKNESS_TO_SPACING",
    "SHEET_REACH_FACTOR",
    "IntrusionField",
    "SequenceField",
    "SheetGeometry",
    "event_from_manifest",
    "fit_intrusion",
    "fit_sequence",
    "mask_younger",
    "order_intrusions",
]

INTRUSION_SHAPES = ("auto", "body", "sheet")
#: holes with a complete intercept (both contacts logged) a sheet needs: three points fix
#: a plane, the fourth checks it
MIN_SHEET_HOLES = 4
#: largest spread of the intercept midpoints across the plane, as a fraction of their
#: smaller spread in the plane, for the midpoints to count as lying on one surface
SHEET_MAX_FLATNESS = 0.2
#: ``shape="auto"`` takes a sheet when the median true thickness is at most this fraction
#: of the hole spacing (thicker bodies are resolved by a distance field)
SHEET_MAX_THICKNESS_TO_SPACING = 0.5
#: a sheet continues this many median distances between neighbouring intercepts beyond
#: the nearest intercept (the "2 x median spacing" extrapolation limit of the contract)
SHEET_REACH_FACTOR = 2.0
#: a point whose value comes from one contact only is kept this fraction of the unit's
#: thickness inside its unit, so that it is never placed on the far contact
_INSIDE_FRACTION = 0.95
_TASK = "P9.T10"
_UNIT_COLUMN = "__unit__"
_TOL = 1e-6
_DEFAULT_SETTINGS = FieldSettings()


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _points(points) -> np.ndarray:
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    return pts


def _dip_and_azimuth(pole: np.ndarray) -> tuple[float, float]:
    """Dip and dip azimuth (degrees) of the plane with the upward unit normal ``pole``."""
    dip = float(np.degrees(np.arccos(np.clip(pole[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(pole[0], pole[1])) % 360.0)
    return dip, azimuth


def _upward(vector) -> np.ndarray:
    v = np.asarray(vector, dtype=np.float64)
    v = v / np.linalg.norm(v)
    if v[2] < 0.0 or (v[2] == 0.0 and (v[0] < 0.0 or (v[0] == 0.0 and v[1] < 0.0))):
        v = -v
    return v


def _spacing_of(xyz: np.ndarray, hole_id: np.ndarray) -> float | None:
    return _hole_spacing([SimpleNamespace(xyz=xyz, hole_id=hole_id)])


def _anisotropy_dict(anisotropy: Ellipsoid | None) -> dict | None:
    return None if anisotropy is None else anisotropy.to_dict()


def _anisotropy_from(data) -> Ellipsoid | None:
    return None if data is None else Ellipsoid.from_dict(data)


def _entry(item, kind, source, used_for, detail="", **extra) -> ProvenanceEntry:
    return ProvenanceEntry(
        item=item,
        kind=kind,
        source=source,
        used_for=used_for,
        detail=detail,
        stage="lithology",
        task=_TASK,
        **extra,
    )


def _source(given, default: str) -> str:
    return "given in the settings" if given is not None else f"default ({default})"


# --------------------------------------------------------------------------
# the sequence field
# --------------------------------------------------------------------------


class SequenceField:
    """One stratigraphic field for a layer stack: mode ``sequence``.

    ``units`` are listed top to bottom; ``thresholds`` ``T_0 .. T_{m-2}`` are the values
    of ``phi`` on the contacts; ``thickness`` holds the true thickness used for every unit
    (for the top and the bottom unit: the cap of their one-sided values); ``pole`` is the
    upward unit normal of the layering; ``violations`` lists where a hole's units are out
    of the stated order (``hole_id, depth, found, expected``).
    """

    mode = "sequence"

    def __init__(
        self,
        units: Sequence[str],
        field: ImplicitField,
        thresholds,
        thickness: Mapping[str, float],
        pole,
        *,
        violations: pd.DataFrame | None = None,
        inputs: tuple[np.ndarray, np.ndarray] | None = None,
        anisotropy: Ellipsoid | None = None,
        kernel: str = "linear",
        sample_spacing: float = np.nan,
        far_spacing: float = np.nan,
        hole_spacing: float | None = None,
        settings: FieldSettings | None = None,
        provenance: Sequence[ProvenanceEntry] = (),
    ) -> None:
        self.units = tuple(str(u) for u in units)
        if len(self.units) < 2 or len(set(self.units)) != len(self.units):
            raise ValueError("a sequence needs at least two different units, top to bottom")
        self.field = field
        self.thresholds = np.asarray(thresholds, dtype=np.float64).reshape(-1)
        if len(self.thresholds) != len(self.units) - 1:
            raise ValueError("a sequence of m units has m - 1 thresholds")
        if (np.diff(self.thresholds) <= 0.0).any() or not np.isfinite(self.thresholds).all():
            raise ValueError("thresholds must be finite and increase from top to bottom")
        self.thickness = {str(k): float(v) for k, v in thickness.items()}
        self.pole = _upward(pole)
        if violations is None:
            violations = pd.DataFrame(columns=["hole_id", "depth", "found", "expected"])
        self.violations = violations
        self.inputs = None if inputs is None else (np.asarray(inputs[0]), np.asarray(inputs[1]))
        self.anisotropy = anisotropy
        self.kernel = kernel
        self.sample_spacing = float(sample_spacing)
        self.far_spacing = float(far_spacing)
        self.hole_spacing = None if hole_spacing is None else float(hole_spacing)
        self.settings = settings if settings is not None else FieldSettings()
        self.provenance = list(provenance)

    def phi(self, points) -> np.ndarray:
        """The stratigraphic position at world points ``(n, 3)``: ``(n,)``."""
        return self.field(_points(points), clipped=False)

    def node_values(self, points) -> np.ndarray:
        """``phi`` as a column: ``(n, 1)``."""
        return self.phi(points)[:, None]

    def labels_from_values(self, values) -> np.ndarray:
        """``U_0`` below ``T_0``, ``U_k`` on ``[T_{k-1}, T_k)``, ``U_{m-1}`` from
        ``T_{m-2}``; -1 for NaN. ``values`` is ``(n, 1)`` or ``(n,)``."""
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        if v.ndim != 1:
            raise ValueError(f"values must have shape (n, 1), got {np.shape(values)}")
        out = np.searchsorted(self.thresholds, v, side="right").astype(np.int16)
        out[np.isnan(v)] = -1
        return out

    def labels(self, points) -> np.ndarray:
        """Exact labels: int16 index into ``units``."""
        return self.labels_from_values(self.phi(points))

    def unit_margins(self, values) -> np.ndarray:
        """A margin per unit from node values ``(1, ...)``: ``min(phi - T_{k-1},
        T_k - phi)``, positive inside the unit. Shape ``(m, ...)``."""
        v = np.asarray(values, dtype=np.float64)
        if v.shape[0] != 1:
            raise ValueError("a sequence has one node value (phi)")
        phi = v[0]
        m = len(self.units)
        out = np.empty((m, *phi.shape))
        for k in range(m):
            if k == 0:
                out[k] = self.thresholds[0] - phi
            elif k == m - 1:
                out[k] = phi - self.thresholds[-1]
            else:
                out[k] = np.minimum(phi - self.thresholds[k - 1], self.thresholds[k] - phi)
        return out

    # persistence hooks ----------------------------------------------------

    def field_files(self) -> dict[str, tuple[ImplicitField, np.ndarray, np.ndarray]]:
        if self.inputs is None:
            raise ValueError("this field does not carry the inputs of its fit: cannot store")
        return {"phi": (self.field, self.inputs[0], self.inputs[1])}

    def to_manifest(self) -> dict[str, Any]:
        return {
            "thresholds": [float(t) for t in self.thresholds],
            "thickness": dict(self.thickness),
            "pole": [float(c) for c in self.pole],
            "anisotropy": _anisotropy_dict(self.anisotropy),
            "kernel": self.kernel,
            "sample_spacing": self.sample_spacing,
            "far_spacing": self.far_spacing,
            "hole_spacing": self.hole_spacing,
            "violations": [
                {
                    "hole_id": str(row.hole_id),
                    "depth": float(row.depth),
                    "found": str(row.found),
                    "expected": str(row.expected),
                }
                for row in self.violations.itertuples(index=False)
            ],
        }

    @classmethod
    def from_manifest(cls, entry, fields, inputs) -> SequenceField:
        rows = entry.get("violations", [])
        return cls(
            entry["units"],
            fields["phi"],
            entry["thresholds"],
            entry["thickness"],
            entry["pole"],
            violations=pd.DataFrame(rows, columns=["hole_id", "depth", "found", "expected"]),
            inputs=inputs["phi"],
            anisotropy=_anisotropy_from(entry.get("anisotropy")),
            kernel=entry.get("kernel", "linear"),
            sample_spacing=entry.get("sample_spacing", np.nan),
            far_spacing=entry.get("far_spacing", np.nan),
            hole_spacing=entry.get("hole_spacing"),
        )


@dataclass
class _Span:
    """The stretch of a hole in one unit: runs of that unit with only unknown between."""

    hole: object
    unit: int
    pieces: list[tuple[float, float]]
    top: tuple[float, int] | None = None  # (depth of the contact, unit index above)
    base: tuple[float, int] | None = None  # (depth of the contact, unit index below)

    def bounds(self) -> tuple[str | None, str | None]:
        """Which contact of its own unit the span's top and base are: 'upper' (towards
        ``U_{k-1}``, value ``T_{k-1}``) or 'lower' (value ``T_k``)."""
        top = None if self.top is None else ("upper" if self.top[1] < self.unit else "lower")
        base = None if self.base is None else ("lower" if self.base[1] > self.unit else "upper")
        return top, base

    @property
    def complete(self) -> bool:
        top, base = self.bounds()
        return top is not None and base is not None and top != base


def _spans_of(runs: pd.DataFrame, order: Sequence[str]) -> tuple[list[_Span], list[tuple]]:
    """Spans of every hole, and the order violations ``(hole, depth, found, expected)``."""
    index = {u: k for k, u in enumerate(order)}
    m = len(order)
    spans: list[_Span] = []
    violations: list[tuple] = []
    hole = runs["hole_id"].to_numpy(dtype=object)
    a = runs["depth_from"].to_numpy(dtype=np.float64)
    b = runs["depth_to"].to_numpy(dtype=np.float64)
    unit = np.array([index[u] for u in runs["unit"]], dtype=np.int64)
    for i in range(len(runs)):
        k = int(unit[i])
        same_hole = i > 0 and hole[i] == hole[i - 1]
        if not same_hole:
            spans.append(_Span(hole[i], k, [(a[i], b[i])]))
            continue
        j = int(unit[i - 1])
        abuts = abs(a[i] - b[i - 1]) <= _TOL
        if j == k and not abuts:
            spans[-1].pieces.append((a[i], b[i]))
            continue
        new = _Span(hole[i], k, [(a[i], b[i])])
        if abuts:
            spans[-1].base = (float(a[i]), k)
            new.top = (float(a[i]), j)
        if (abuts and k != j + 1) or (not abuts and k < j):
            expected = order[j + 1] if j + 1 < m else ""
            violations.append((hole[i], float(a[i]), order[k], expected))
        spans.append(new)
    return spans, violations


def _piece_depths(a: float, b: float, spacing: float) -> np.ndarray:
    n = max(1, int(np.ceil((b - a) / spacing - 1e-9)))
    return a + (np.arange(n) + 0.5) * ((b - a) / n)


def fit_sequence(
    table: pd.DataFrame,
    units: pd.Series,
    order: Sequence[str],
    traces: HoleTraces,
    *,
    settings: FieldSettings = _DEFAULT_SETTINGS,
    thickness: Mapping[str, float] | None = None,
) -> SequenceField:
    """One stratigraphic field ``phi`` for the stack ``order`` (top to bottom).

    ``table`` has ``hole_id, depth_from, depth_to``; ``units`` gives the unit of each row.
    Units outside ``order`` (and NaN) are unknown: no points, and no contact at their ends
    (an intrusion logged inside the stack hides what it replaced).

    Pole of the layering: the minor axis of ``settings.anisotropy`` when an ``Ellipsoid``
    is given; otherwise the direction of least spread of the contact points after each
    contact's own mean is removed (vertical when there are too few). The field is fitted
    in the anisotropy given, in the automatic one for ``"auto"``, isotropic for None.

    Thickness ``t_k`` of a unit: the value in ``thickness`` if given, else the median of
    ``|pole . (base - top)|`` over the spans of the unit that are entered through one of
    its contacts and left through the other. A unit without such a span takes the median
    of the others. The top and the bottom unit have one contact only: their values are
    capped at ``settings.cap``, their ``thickness`` entry, or the median of the ``t_k``.

    Values along a hole, for a span of ``U_k`` (its runs, joined across unknown
    intervals):

    * both contacts logged: linear in depth between the two contact values;
    * one contact logged (value ``T``): ``T +- |pole . (p - contact)|``, into the unit,
      at most 0.95 of the unit's thickness (the cap for the top and bottom unit);
    * no contact: the middle of the unit (top / bottom unit: one cap beyond the contact).

    Every value lies inside its own unit's range, so an exact fit reproduces the logged
    units at the points whatever the order of the units in a hole. Holes whose units are
    out of the stated order are listed in ``violations``; their points are still used.

    Point spacing: ``settings.sample_spacing`` (default: the smaller of a quarter of the
    median thickness and a tenth of the median logged hole length) where the value
    varies, ``settings.far_spacing`` (default: a tenth of the median logged hole length)
    where it is capped. ``settings.thin`` is not used.
    """
    order = tuple(str(u) for u in order)
    m = len(order)
    if m < 2:
        raise ValueError("a sequence needs at least two units, top to bottom")
    if len(set(order)) != m:
        raise ValueError(f"the sequence lists a unit twice: {order}")
    if settings.kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {settings.kernel!r}")
    given = {} if thickness is None else {str(k): float(v) for k, v in thickness.items()}
    for name, value in given.items():
        if name not in order:
            raise ValueError(f"thickness names the unit {name!r}, which is not in the sequence")
        if not (np.isfinite(value) and value > 0.0):
            raise ValueError(f"the thickness of {name} must be > 0, got {value!r}")
    frame = _unit_table(table, units, order)
    runs = _runs_of(frame, order, traces)
    absent = [u for u in order if not (runs["unit"] == u).any()]
    if absent:
        raise ValueError(f"no logged run of unit(s) {', '.join(absent)}: cannot fit the sequence")

    # pole and anisotropy
    contacts = _contacts_of(runs, traces)
    anisotropy = settings.anisotropy
    if isinstance(anisotropy, Ellipsoid):
        ellipsoid, pole = anisotropy, _upward(anisotropy.minor_axis)
        pole_source, how = "minor axis of the anisotropy given in the settings", "given"
        note = (
            f"ratios major/semi-major {anisotropy.semi_major_ratio:.2f}, "
            f"major/minor {anisotropy.minor_ratio:.2f}"
        )
    elif anisotropy is None or anisotropy == "auto":
        auto, note = _auto_anisotropy(contacts)
        if auto is None:
            pole, pole_source = np.array([0.0, 0.0, 1.0]), f"vertical ({note})"
        else:
            pole = _upward(auto.minor_axis)
            pole_source = "least spread of the contact points, each contact's mean removed"
        if anisotropy == "auto":
            ellipsoid, how = auto, "automatic"
        else:
            ellipsoid, how, note = None, "none", "isotropic field"
    else:
        raise TypeError("anisotropy must be an Ellipsoid, 'auto' or None")

    spans, violation_rows = _spans_of(runs, order)

    def xyz_at(holes, depths) -> np.ndarray:
        xyz, _ = traces.xyz_at(
            np.asarray(holes, dtype=object), np.asarray(depths, dtype=np.float64),
            depth_tolerance=_TOL,
        )  # fmt: skip
        return np.asarray(xyz, dtype=np.float64).reshape(-1, 3)

    top_xyz = xyz_at([s.hole for s in spans], [s.top[0] if s.top else np.nan for s in spans])
    base_xyz = xyz_at([s.hole for s in spans], [s.base[0] if s.base else np.nan for s in spans])

    # thickness of every unit
    measured: dict[str, list[float]] = {u: [] for u in order}
    for i, span in enumerate(spans):
        if span.complete:
            t = abs(float(pole @ (base_xyz[i] - top_xyz[i])))
            if np.isfinite(t) and t > 0.0:
                measured[order[span.unit]].append(t)
    used: dict[str, float] = {}
    how_thick: dict[str, str] = {}
    for u in order[1:-1]:
        if u in given:
            used[u], how_thick[u] = given[u], "given"
        elif measured[u]:
            used[u] = float(np.median(measured[u]))
            how_thick[u] = f"median true thickness of {len(measured[u])} complete intercepts"
    length = (runs["depth_to"] - runs["depth_from"]).to_numpy(dtype=np.float64)
    hole_length = float(np.median(pd.Series(length).groupby(runs["hole_id"].to_numpy()).sum()))
    if used:
        typical = float(np.median(list(used.values())))
        typical_how = "median of the units' thicknesses"
    else:
        ends = xyz_at(runs["hole_id"], runs["depth_to"]) - xyz_at(
            runs["hole_id"], runs["depth_from"]
        )
        true_length = np.abs(ends @ pole)
        typical = float(np.nanmedian(true_length[np.isfinite(true_length)]))
        typical_how = "median true length of the logged runs (no unit has both contacts)"
    for u in order[1:-1]:
        if u not in used:
            used[u], how_thick[u] = typical, f"no complete intercept: {typical_how}"
    for u in (order[0], order[-1]):
        if u in given:
            used[u], how_thick[u] = given[u], "cap of the one-sided values, given"
        elif settings.cap is not None:
            used[u], how_thick[u] = float(settings.cap), "cap given in the settings"
        else:
            used[u], how_thick[u] = typical, f"cap of the one-sided values: {typical_how}"
    for u, t in used.items():
        if not (np.isfinite(t) and t > 0.0):
            raise ValueError(f"the thickness of {u} is not a positive number ({t!r})")
    thresholds = np.concatenate([[0.0], np.cumsum([used[u] for u in order[1:-1]])])

    far = hole_length / 10.0 if settings.far_spacing is None else float(settings.far_spacing)
    if settings.sample_spacing is not None:
        near = float(settings.sample_spacing)
    else:
        near = min(float(np.median(list(used.values()))) / 4.0, hole_length / 10.0)
    for name, v in (("sample_spacing", near), ("far_spacing", far)):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be a positive finite number, got {v!r}")
    far = max(far, near)
    keep_every = max(1, int(round(far / near)))

    def level(k: int, which: str) -> float:
        return float(thresholds[k - 1] if which == "upper" else thresholds[k])

    # points along the spans
    holes, depths, owner = [], [], []
    for i, span in enumerate(spans):
        for a, b in span.pieces:
            d = _piece_depths(a, b, near)
            holes.append(np.full(len(d), span.hole, dtype=object))
            depths.append(d)
            owner.append(np.full(len(d), i, dtype=np.int64))
    holes, depths, owner = np.concatenate(holes), np.concatenate(depths), np.concatenate(owner)
    xyz = xyz_at(holes, depths)
    values = np.full(len(depths), np.nan)
    flat = np.zeros(len(depths), dtype=bool)
    start = np.searchsorted(owner, np.arange(len(spans) + 1))
    for i, span in enumerate(spans):
        sel = slice(start[i], start[i + 1])
        k = span.unit
        width = used[order[k]]
        end_unit = k in (0, m - 1)
        top, base = span.bounds()
        if span.complete:
            v_top, v_base = level(k, top), level(k, base)
            t = (depths[sel] - span.top[0]) / (span.base[0] - span.top[0])
            values[sel] = v_top + (v_base - v_top) * t
            continue
        known = [(which, p) for which, p in ((top, top_xyz[i]), (base, base_xyz[i])) if which]
        if not known:
            if k == 0:
                values[sel] = thresholds[0] - width
            elif k == m - 1:
                values[sel] = thresholds[-1] + width
            else:
                values[sel] = 0.5 * (thresholds[k - 1] + thresholds[k])
            flat[sel] = True
            continue
        which = known[0][0]  # with two contacts here, both are the same contact of the unit
        distance = np.min([np.abs((xyz[sel] - p) @ pole) for _, p in known], axis=0)
        limit = width if end_unit else _INSIDE_FRACTION * width
        sign = 1.0 if which == "upper" else -1.0
        values[sel] = level(k, which) + sign * np.minimum(distance, limit)
        flat[sel] = distance >= limit
    position = np.arange(len(depths)) - start[owner]
    keep = np.isfinite(values) & np.isfinite(xyz).all(axis=1)
    keep &= ~flat | (position % keep_every == keep_every // 2) | (keep_every == 1)
    # a span that is flat throughout keeps at least one point
    kept_per_span = np.bincount(owner[keep], minlength=len(spans))
    for i in np.flatnonzero(kept_per_span == 0):
        if start[i + 1] > start[i]:
            middle = (start[i] + start[i + 1]) // 2
            keep[middle] = np.isfinite(values[middle]) and np.isfinite(xyz[middle]).all()

    # contact points (the two spans of a contact may give different values where the
    # order is broken: fit_field averages coincident points)
    contact_xyz, contact_value = [], []
    for i, span in enumerate(spans):
        top, base = span.bounds()
        for which, p in ((top, top_xyz[i]), (base, base_xyz[i])):
            if which and np.isfinite(p).all():
                contact_xyz.append(p)
                contact_value.append(level(span.unit, which))
    fit_xyz = np.concatenate([xyz[keep], np.reshape(contact_xyz, (-1, 3))])
    fit_values = np.concatenate([values[keep], np.asarray(contact_value, dtype=np.float64)])
    fit_holes = np.concatenate([holes[keep], np.full(len(contact_value), None, dtype=object)])
    field = fit_field(
        fit_xyz, fit_values, anisotropy=ellipsoid, kernel=settings.kernel,
        neighbors=settings.neighbors,
    )  # fmt: skip

    violations = pd.DataFrame(violation_rows, columns=["hole_id", "depth", "found", "expected"])
    label = " > ".join(order)
    used_for = f"stratigraphic field of the sequence {label}"
    rows = frame[_UNIT_COLUMN]
    metres = pd.to_numeric(frame["depth_to"], errors="coerce") - pd.to_numeric(
        frame["depth_from"], errors="coerce"
    )
    provenance = []
    for u in order:
        mask = (rows == u).to_numpy()
        own = runs[runs["unit"] == u]
        provenance.append(
            _entry(
                f"Rock-type log: unit {u}",
                "measured",
                "drillhole log, intervals mapped to the unit",
                used_for,
                f"{int(mask.sum())} intervals, {float(metres[mask].sum()):.0f} m, "
                f"{own['hole_id'].nunique()} holes, {len(own)} runs",
                count=int(mask.sum()),
            )
        )
    known_rows = rows.notna().to_numpy()
    provenance.append(
        _entry(
            f"Rock-type log: sequence {label}",
            "measured",
            "drillhole log, intervals mapped to the units of the sequence",
            used_for,
            f"{int(known_rows.sum())} intervals, {float(metres[known_rows].sum()):.0f} m, "
            f"{runs['hole_id'].nunique()} holes; {len(fit_values)} points "
            f"({len(contact_value)} at contacts); {int((~known_rows).sum())} intervals of other "
            "codes are unknown (no points, no contacts)",
            count=int(known_rows.sum()),
        )
    )
    provenance.append(
        _entry(
            f"Modelling mode of the sequence {label}",
            "assumed",
            "one stratigraphic field; the order of the units (top to bottom) as given",
            used_for,
            f"kernel {settings.kernel}; "
            + ("global fit" if field.neighbors is None else "local fit")
            + "; contacts are the levels "
            + ", ".join(f"{t:.2f}" for t in thresholds),
        )
    )
    dip, azimuth = _dip_and_azimuth(pole)
    provenance.append(
        _entry(
            f"Pole of the layering, sequence {label}",
            "assumed",
            pole_source,
            "true thickness of the units and distances from the contacts",
            f"dip {dip:.0f}, dip azimuth {azimuth:03.0f}",
        )
    )
    provenance.append(_entry(f"Anisotropy of the sequence {label}", "assumed", how, used_for, note))
    for u in order:
        provenance.append(
            _entry(
                f"Thickness of {u} in the sequence {label}",
                "assumed",
                how_thick[u],
                "levels of the contacts in the stratigraphic field",
                f"{used[u]:.2f} m",
                value=float(used[u]),
            )
        )
    spacing_rule = "a tenth of the median logged hole length"
    for item, value, given_value, default in (
        ("Point spacing where the value varies", near, settings.sample_spacing,
         f"smaller of a quarter of the median thickness and {spacing_rule}"),
        ("Point spacing where the value is capped", far, settings.far_spacing, spacing_rule),
    ):  # fmt: skip
        provenance.append(
            _entry(
                f"{item}, sequence {label}",
                "assumed",
                _source(given_value, default),
                used_for,
                f"{value:.2f} m",
                value=float(value),
            )
        )
    if len(violations):
        provenance.append(
            _entry(
                f"Units out of order in the sequence {label}",
                "assumed",
                "holes where a unit is not followed by the next one of the stated order",
                used_for,
                f"{len(violations)} places in {violations['hole_id'].nunique()} holes; their "
                "points are used with the values their own contacts imply (a fault, an "
                "unconformity or a logging difference is not resolved here)",
                count=int(len(violations)),
            )
        )
    return SequenceField(
        order,
        field,
        thresholds,
        used,
        pole,
        violations=violations,
        inputs=(fit_xyz, fit_values),
        anisotropy=ellipsoid,
        kernel=settings.kernel,
        sample_spacing=near,
        far_spacing=far,
        hole_spacing=_spacing_of(fit_xyz[: int(keep.sum())], fit_holes[: int(keep.sum())]),
        settings=settings,
        provenance=provenance,
    )


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------


class SheetGeometry:
    """A thin sheet: a reference surface, a thickness and an extent, all in one plane.

    ``origin`` is a point of the plane, ``normal`` its upward unit pole, ``u`` (along
    strike) and ``v`` (up dip) the in-plane axes. ``surface`` gives the offset of the
    reference surface from the plane and ``thickness`` the true thickness, both fields of
    the in-plane position ``(u, v, 0)``. ``present`` ``(n, 2)`` are in-plane positions
    where a hole is in the unit, ``absent`` ``(k, 2)`` where a hole crosses the surface in
    known rock without it; ``reach`` is how far the sheet continues beyond ``present``.
    """

    def __init__(
        self,
        *,
        origin,
        normal,
        u,
        v,
        surface: ImplicitField,
        thickness: ImplicitField,
        thickness_range: tuple[float, float],
        present,
        absent,
        reach: float,
        surface_inputs: tuple[np.ndarray, np.ndarray],
        thickness_inputs: tuple[np.ndarray, np.ndarray],
        info: Mapping[str, Any] | None = None,
    ) -> None:
        self.origin = np.asarray(origin, dtype=np.float64).reshape(3)
        self.normal = np.asarray(normal, dtype=np.float64).reshape(3)
        self.u = np.asarray(u, dtype=np.float64).reshape(3)
        self.v = np.asarray(v, dtype=np.float64).reshape(3)
        self.surface = surface
        self.thickness = thickness
        self.thickness_range = (float(thickness_range[0]), float(thickness_range[1]))
        self.present = np.asarray(present, dtype=np.float64).reshape(-1, 2)
        self.absent = np.asarray(absent, dtype=np.float64).reshape(-1, 2)
        if len(self.present) == 0:
            raise ValueError("a sheet needs at least one position where the unit is present")
        self.reach = float(reach)
        self.surface_inputs = (np.asarray(surface_inputs[0]), np.asarray(surface_inputs[1]))
        self.thickness_inputs = (np.asarray(thickness_inputs[0]), np.asarray(thickness_inputs[1]))
        self.info = dict(info or {})
        self._present_tree = cKDTree(self.present)
        self._absent_tree = cKDTree(self.absent) if len(self.absent) else None

    @property
    def dip(self) -> float:
        return _dip_and_azimuth(self.normal)[0]

    @property
    def dip_azimuth(self) -> float:
        return _dip_and_azimuth(self.normal)[1]

    def coordinates(self, points) -> np.ndarray:
        """In-plane position and distance from the plane: ``(n, 3)`` columns u, v, w."""
        local = _points(points) - self.origin
        return np.column_stack([local @ self.u, local @ self.v, local @ self.normal])

    def _in_plane(self, uvw: np.ndarray) -> np.ndarray:
        return np.column_stack([uvw[:, 0], uvw[:, 1], np.zeros(len(uvw))])

    def offset(self, uvw: np.ndarray) -> np.ndarray:
        """Distance of the reference surface from the plane at the in-plane positions."""
        return self.surface(self._in_plane(uvw), clipped=False)

    def thickness_at(self, uvw: np.ndarray) -> np.ndarray:
        """True thickness at the in-plane positions, kept inside the measured range."""
        return np.clip(self.thickness(self._in_plane(uvw), clipped=False), *self.thickness_range)

    def extent_margin(self, uvw: np.ndarray) -> np.ndarray:
        """Positive where the sheet exists in the plane: ``reach`` minus the distance to
        the nearest position with the unit, and half the lead of that position over the
        nearest barren crossing."""
        d_present, _ = self._present_tree.query(uvw[:, :2])
        margin = self.reach - d_present
        if self._absent_tree is not None:
            d_absent, _ = self._absent_tree.query(uvw[:, :2])
            margin = np.minimum(margin, 0.5 * (d_absent - d_present))
        return margin

    def value(self, points) -> np.ndarray:
        """``min(thickness / 2 - |w - w_0|, extent margin)``: positive inside the sheet."""
        uvw = self.coordinates(points)
        across = 0.5 * self.thickness_at(uvw) - np.abs(uvw[:, 2] - self.offset(uvw))
        return np.minimum(across, self.extent_margin(uvw))

    def to_dict(self) -> dict[str, Any]:
        return {
            "origin": [float(c) for c in self.origin],
            "normal": [float(c) for c in self.normal],
            "u": [float(c) for c in self.u],
            "v": [float(c) for c in self.v],
            "thickness_range": [float(c) for c in self.thickness_range],
            "present": [[float(a), float(b)] for a, b in self.present],
            "absent": [[float(a), float(b)] for a, b in self.absent],
            "reach": self.reach,
            "info": self.info,
        }

    @classmethod
    def from_dict(cls, data, fields, inputs) -> SheetGeometry:
        return cls(
            origin=data["origin"],
            normal=data["normal"],
            u=data["u"],
            v=data["v"],
            surface=fields["surface"],
            thickness=fields["thickness"],
            thickness_range=tuple(data["thickness_range"]),
            present=data["present"],
            absent=data["absent"],
            reach=data["reach"],
            surface_inputs=inputs["surface"],
            thickness_inputs=inputs["thickness"],
            info=data.get("info"),
        )


def _intercepts_of(constraints: Constraints, traces: HoleTraces) -> pd.DataFrame:
    """The in-unit runs with their end points and whether both contacts are logged."""
    runs = constraints.intercepts.sort_values(["hole_id", "depth_from"], kind="stable")
    hole = runs["hole_id"].to_numpy(dtype=object)
    a = runs["depth_from"].to_numpy(dtype=np.float64)
    b = runs["depth_to"].to_numpy(dtype=np.float64)
    state = runs["state"].to_numpy()
    n = len(runs)
    above = np.zeros(n, dtype=bool)
    below = np.zeros(n, dtype=bool)
    if n > 1:
        touch = (hole[1:] == hole[:-1]) & (np.abs(a[1:] - b[:-1]) <= _TOL)
        touch &= state[1:] != state[:-1]
        above[1:], below[:-1] = touch, touch
    top, _ = traces.xyz_at(hole, a, depth_tolerance=_TOL)
    bottom, _ = traces.xyz_at(hole, b, depth_tolerance=_TOL)
    out = pd.DataFrame(
        {
            "hole_id": hole,
            "depth_from": a,
            "depth_to": b,
            "state": state,
            "complete": above & below,
        }
    )
    for name, xyz in (("top", top), ("bottom", bottom)):
        for c, axis in enumerate("xyz"):
            out[f"{name}_{axis}"] = xyz[:, c]
    ok = np.isfinite(top).all(axis=1) & np.isfinite(bottom).all(axis=1)
    return out[ok].reset_index(drop=True)


def _ends(rows: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    top = rows[["top_x", "top_y", "top_z"]].to_numpy(dtype=np.float64)
    bottom = rows[["bottom_x", "bottom_y", "bottom_z"]].to_numpy(dtype=np.float64)
    return top, bottom


def _fit_sheet(
    unit: str,
    constraints: Constraints,
    traces: HoleTraces,
    *,
    hole_spacing: float | None,
    require_thin: bool,
    reach: float | None,
    crossing_step: float = 5.0,
) -> tuple[SheetGeometry | None, str]:
    """The sheet of ``unit``, or None and the reason it is not determined."""
    runs = _intercepts_of(constraints, traces)
    inside = runs[runs["state"] > 0]
    complete = inside[inside["complete"]]
    n_holes = int(complete["hole_id"].nunique())
    if n_holes < MIN_SHEET_HOLES:
        return None, (
            f"{n_holes} holes with a complete intercept (both contacts logged), a sheet needs "
            f"{MIN_SHEET_HOLES}"
        )
    top, bottom = _ends(complete)
    mid = 0.5 * (top + bottom)
    origin = mid.mean(axis=0)
    q = mid - origin
    lam, vec = np.linalg.eigh(q.T @ q / len(q))  # ascending
    sd = np.sqrt(np.maximum(lam, 0.0))
    if not sd[2] > 0.0 or sd[1] < 0.02 * sd[2]:
        return None, "the intercept midpoints lie on one line: the plane is not determined"
    if sd[0] > SHEET_MAX_FLATNESS * sd[1]:
        return None, (
            f"the intercept midpoints do not lie near one plane (spread {sd[0]:.1f} m across, "
            f"{sd[1]:.1f} m and {sd[2]:.1f} m along; limit {SHEET_MAX_FLATNESS:g} of the smaller)"
        )
    normal = _upward(vec[:, 0])
    true_thickness = np.abs((bottom - top) @ normal)
    if not (true_thickness > 0.0).all():
        return None, "an intercept runs in the plane of the midpoints: no true thickness"
    median_thickness = float(np.median(true_thickness))
    if require_thin:
        if hole_spacing is None:
            return None, "the hole spacing is unknown (one hole): thin or not cannot be judged"
        if median_thickness > SHEET_MAX_THICKNESS_TO_SPACING * hole_spacing:
            return None, (
                f"median true thickness {median_thickness:.1f} m is more than "
                f"{SHEET_MAX_THICKNESS_TO_SPACING:g} of the hole spacing ({hole_spacing:.1f} m): "
                "the holes resolve the body"
            )
    strike = np.cross([0.0, 0.0, 1.0], normal)
    u = strike / np.linalg.norm(strike) if np.linalg.norm(strike) > 1e-9 else np.array([1.0, 0, 0])
    v = np.cross(normal, u)

    def uvw(points: np.ndarray) -> np.ndarray:
        local = points - origin
        return np.column_stack([local @ u, local @ v, local @ normal])

    mid_uvw = uvw(mid)
    plane_xyz = np.column_stack([mid_uvw[:, :2], np.zeros(len(mid_uvw))])
    surface = fit_field(plane_xyz, mid_uvw[:, 2], kernel="linear", degree=0, neighbors=None)
    thickness = fit_field(plane_xyz, true_thickness, kernel="linear", degree=0, neighbors=None)

    # where the unit is: the points of the holes inside it, and its contacts
    in_unit = constraints.value >= 0.0
    present = uvw(constraints.xyz[in_unit])[:, :2]

    # barren crossings: outside runs that pass through the reference surface
    absent: list[np.ndarray] = []
    outside = runs[runs["state"] < 0]
    for row in outside.itertuples(index=False):
        n = max(1, int(np.ceil((row.depth_to - row.depth_from) / crossing_step)))
        depth = np.linspace(row.depth_from, row.depth_to, n + 1)
        xyz, _ = traces.xyz_at(
            np.full(n + 1, row.hole_id, dtype=object), depth, depth_tolerance=_TOL
        )
        local = uvw(np.asarray(xyz, dtype=np.float64))
        side = local[:, 2] - surface(
            np.column_stack([local[:, :2], np.zeros(len(local))]), clipped=False
        )
        for k in np.flatnonzero(np.sign(side[1:]) * np.sign(side[:-1]) < 0):
            t = side[k] / (side[k] - side[k + 1])
            absent.append(local[k, :2] + t * (local[k + 1, :2] - local[k, :2]))
    absent_uv = np.reshape(absent, (-1, 2))

    # how far the sheet continues beyond the nearest intercept
    all_mid = uvw(0.5 * np.add(*_ends(inside)))[:, :2]
    neighbour = float("nan")
    if len(all_mid) >= 2:
        distance, _ = cKDTree(all_mid).query(all_mid, k=2)
        neighbour = float(np.median(distance[:, 1]))
    if reach is not None:
        used_reach, reach_source = float(reach), "max_extrapolation given in the settings"
    elif np.isfinite(neighbour) and neighbour > 0.0:
        used_reach = SHEET_REACH_FACTOR * neighbour
        reach_source = (
            f"default ({SHEET_REACH_FACTOR:g} x the median distance between neighbouring "
            f"intercepts in the plane, {neighbour:.1f} m)"
        )
    else:
        return None, "the distance between neighbouring intercepts is not defined"
    info = {
        "n_holes": n_holes,
        "n_intercepts": int(len(complete)),
        "n_incomplete": int(len(inside) - len(complete)),
        "n_barren_crossings": int(len(absent_uv)),
        "misfit_rms": float(sd[0]),
        "thickness_median": median_thickness,
        "thickness_min": float(true_thickness.min()),
        "thickness_max": float(true_thickness.max()),
        "neighbour_distance": neighbour,
        "reach_source": reach_source,
        "hole_spacing": hole_spacing,
    }
    sheet = SheetGeometry(
        origin=origin,
        normal=normal,
        u=u,
        v=v,
        surface=surface,
        thickness=thickness,
        thickness_range=(float(true_thickness.min()), float(true_thickness.max())),
        present=present,
        absent=absent_uv,
        reach=used_reach,
        surface_inputs=(plane_xyz, mid_uvw[:, 2]),
        thickness_inputs=(plane_xyz, true_thickness),
        info=info,
    )
    return sheet, "sheet"


# --------------------------------------------------------------------------
# intrusions
# --------------------------------------------------------------------------


class IntrusionField:
    """One intrusion: mode ``intrusion``. ``node_values`` is its field ``(n, 1)``; the
    label is 0 (the unit) where the field is positive and -1 (does not apply) elsewhere.

    ``shape`` is ``"body"`` (``field`` is the binary signed-distance field) or ``"sheet"``
    (``sheet`` holds the geometry and ``field`` is None).
    """

    mode = "intrusion"

    def __init__(
        self,
        unit: str,
        field: ImplicitField | None = None,
        *,
        sheet: SheetGeometry | None = None,
        constraints: Constraints | None = None,
        inputs: tuple[np.ndarray, np.ndarray] | None = None,
        cap: float = np.nan,
        sample_spacing: float = np.nan,
        far_spacing: float = np.nan,
        anisotropy: Ellipsoid | None = None,
        kernel: str = "linear",
        hole_spacing: float | None = None,
        settings: FieldSettings | None = None,
        provenance: Sequence[ProvenanceEntry] = (),
    ) -> None:
        if (field is None) == (sheet is None):
            raise ValueError("an intrusion is either a field (body) or a sheet")
        self.units = (str(unit),)
        self.field = field
        self.sheet = sheet
        self.shape = "body" if sheet is None else "sheet"
        self.constraints = constraints
        if inputs is None and constraints is not None:
            inputs = (constraints.xyz, constraints.value)
        self.inputs = None if inputs is None else (np.asarray(inputs[0]), np.asarray(inputs[1]))
        self.cap = float(cap)
        self.sample_spacing = float(sample_spacing)
        self.far_spacing = float(far_spacing)
        self.anisotropy = anisotropy
        self.kernel = kernel
        self.hole_spacing = None if hole_spacing is None else float(hole_spacing)
        self.settings = settings if settings is not None else FieldSettings()
        self.provenance = list(provenance)

    def values(self, points) -> np.ndarray:
        """The field at world points ``(n, 3)``: ``(n,)``, positive inside the intrusion."""
        pts = _points(points)
        if self.sheet is not None:
            return self.sheet.value(pts)
        return self.field(pts, clipped=False)

    def node_values(self, points) -> np.ndarray:
        return self.values(points)[:, None]

    def labels_from_values(self, values) -> np.ndarray:
        v = np.asarray(values, dtype=np.float64)
        if v.ndim == 2 and v.shape[1] == 1:
            v = v[:, 0]
        if v.ndim != 1:
            raise ValueError(f"values must have shape (n, 1), got {np.shape(values)}")
        with np.errstate(invalid="ignore"):
            return np.where(v > 0.0, 0, -1).astype(np.int16)

    def labels(self, points) -> np.ndarray:
        """Exact labels: 0 inside the intrusion, -1 elsewhere."""
        return self.labels_from_values(self.values(points))

    # persistence hooks ----------------------------------------------------

    def field_files(self) -> dict[str, tuple[ImplicitField, np.ndarray, np.ndarray]]:
        if self.sheet is not None:
            s = self.sheet
            return {
                "surface": (s.surface, *s.surface_inputs),
                "thickness": (s.thickness, *s.thickness_inputs),
            }
        if self.inputs is None:
            raise ValueError("this field does not carry the inputs of its fit: cannot store")
        return {self.units[0]: (self.field, self.inputs[0], self.inputs[1])}

    def to_manifest(self) -> dict[str, Any]:
        out: dict[str, Any] = {"shape": self.shape, "hole_spacing": self.hole_spacing}
        if self.sheet is not None:
            out["sheet"] = self.sheet.to_dict()
        else:
            out.update(
                cap=self.cap,
                sample_spacing=self.sample_spacing,
                far_spacing=self.far_spacing,
                kernel=self.kernel,
                anisotropy=_anisotropy_dict(self.anisotropy),
            )
        return out

    @classmethod
    def from_manifest(cls, entry, fields, inputs) -> IntrusionField:
        unit = entry["units"][0]
        if entry.get("shape", "body") == "sheet":
            sheet = SheetGeometry.from_dict(entry["sheet"], fields, inputs)
            return cls(unit, sheet=sheet, hole_spacing=entry.get("hole_spacing"))
        return cls(
            unit,
            fields[unit],
            inputs=inputs[unit],
            cap=entry.get("cap", np.nan),
            sample_spacing=entry.get("sample_spacing", np.nan),
            far_spacing=entry.get("far_spacing", np.nan),
            anisotropy=_anisotropy_from(entry.get("anisotropy")),
            kernel=entry.get("kernel", "linear"),
            hole_spacing=entry.get("hole_spacing"),
        )


def fit_intrusion(
    table: pd.DataFrame,
    units: pd.Series,
    unit: str,
    traces: HoleTraces,
    *,
    settings: FieldSettings = _DEFAULT_SETTINGS,
    shape: str = "auto",
) -> IntrusionField:
    """The field of the intrusion ``unit`` against every other known unit.

    ``units`` gives the unit of each row of ``table``; every value other than ``unit`` is
    known rock outside the intrusion, and NaN is unknown. The caller masks what must not
    count as outside: units that are not rock (cover, weathering codes) and intrusions
    younger than this one (:func:`mask_younger`), which hide what they replaced.

    ``shape``: ``"body"``, ``"sheet"`` or ``"auto"`` (see the module text). For a body,
    ``settings.cap``, ``sample_spacing`` and ``far_spacing`` default to those of
    ``extract_constraints`` (the cap is the median in-unit intercept length), and
    ``settings.anisotropy="auto"`` uses the intercept midpoints
    (``estimate_intercept_anisotropy``). For a sheet, ``settings.max_extrapolation`` is how
    far it continues beyond the nearest intercept (default :data:`SHEET_REACH_FACTOR` x
    the median distance between neighbouring intercepts); the other settings are not used.

    A unit without any logged run raises ``ValueError``.
    """
    unit = str(unit)
    if shape not in INTRUSION_SHAPES:
        raise ValueError(f"shape must be one of {INTRUSION_SHAPES}, got {shape!r}")
    if settings.kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {settings.kernel!r}")
    for name in ("hole_id", "depth_from", "depth_to"):
        if name not in table.columns:
            raise ValueError(f"interval table has no column {name!r}")
    if len(units) != len(table):
        raise ValueError(f"units has {len(units)} values, the table has {len(table)} rows")
    frame = table[["hole_id", "depth_from", "depth_to"]].copy()
    column = np.asarray(units.to_numpy(dtype=object) if isinstance(units, pd.Series) else units)
    frame[_UNIT_COLUMN] = pd.Series(column, index=frame.index, dtype=object)
    rows = frame[_UNIT_COLUMN]
    if not (rows == unit).any():
        raise ValueError(f"no logged run of unit {unit}: cannot fit its field")
    constraints = extract_constraints(
        frame, _UNIT_COLUMN, unit, traces,
        cap=settings.cap, sample_spacing=settings.sample_spacing,
        far_spacing=settings.far_spacing, thin=settings.thin,
    )  # fmt: skip
    if constraints.n_holes_intersecting == 0:
        raise ValueError(f"no logged run of unit {unit} on a desurveyed hole")
    hole_spacing = _hole_spacing([constraints])
    used_for = f"field of the intrusion {unit} (overprints the earlier units)"
    metres = pd.to_numeric(frame["depth_to"], errors="coerce") - pd.to_numeric(
        frame["depth_from"], errors="coerce"
    )
    own = (rows == unit).to_numpy()
    others = rows.notna().to_numpy() & ~own
    provenance = [
        _entry(
            f"Rock-type log: intrusion {unit}",
            "measured",
            "drillhole log, intervals mapped to the unit",
            used_for,
            f"{int(own.sum())} intervals, {float(metres[own].sum()):.0f} m, "
            f"{constraints.n_holes_intersecting} holes; against {int(others.sum())} intervals "
            f"({float(metres[others].sum()):.0f} m) of other known units; "
            f"{int(rows.isna().sum())} intervals unknown",
            count=int(own.sum()),
        )
    ]

    sheet, reason = None, ""
    if shape in ("auto", "sheet"):
        sheet, reason = _fit_sheet(
            unit,
            constraints,
            traces,
            hole_spacing=hole_spacing,
            require_thin=shape == "auto",
            reach=settings.max_extrapolation,
        )
    if sheet is not None:
        info = sheet.info
        asked = "asked for" if shape == "sheet" else "chosen automatically"
        provenance += [
            _entry(
                f"Shape of the intrusion {unit}",
                "assumed",
                f"sheet ({asked}): reference surface, thickness and extent in one plane",
                used_for,
                f"{info['n_holes']} holes with a complete intercept (needs {MIN_SHEET_HOLES}); "
                f"median true thickness {info['thickness_median']:.1f} m"
                + (
                    f" against a hole spacing of {hole_spacing:.1f} m"
                    if hole_spacing is not None
                    else ""
                )
                + f"; {info['n_incomplete']} incomplete intercepts used for the extent only",
            ),
            _entry(
                f"Orientation of the sheet {unit}",
                "assumed",
                "least-squares plane through the midpoints of the complete intercepts",
                used_for,
                f"dip {sheet.dip:.0f}, dip azimuth {sheet.dip_azimuth:03.0f}; "
                f"{info['n_intercepts']} midpoints, RMS distance from the plane "
                f"{info['misfit_rms']:.2f} m; between the holes the surface follows "
                "the midpoints' offsets from the plane (interpolated)",
            ),
            _entry(
                f"Thickness of the sheet {unit}",
                "assumed",
                "intercept length x |cos| of the angle between hole and pole, interpolated "
                "in the plane",
                used_for,
                f"median {info['thickness_median']:.2f} m, {info['thickness_min']:.2f} to "
                f"{info['thickness_max']:.2f} m; never outside this range between the holes",
                value=float(info["thickness_median"]),
            ),
            _entry(
                f"Continuation of the sheet {unit} beyond the last intercept",
                "assumed",
                info["reach_source"],
                used_for,
                f"{sheet.reach:.1f} m in the plane beyond the nearest intercept; stops half-way "
                f"towards a hole that crosses the surface in known rock without the unit "
                f"({info['n_barren_crossings']} such crossings)",
                value=float(sheet.reach),
            ),
        ]
        return IntrusionField(
            unit,
            sheet=sheet,
            constraints=constraints,
            cap=constraints.cap,
            sample_spacing=constraints.sample_spacing,
            far_spacing=constraints.far_spacing,
            hole_spacing=hole_spacing,
            settings=settings,
            provenance=provenance,
        )

    if shape == "sheet":
        shape_source = "body (signed-distance field); the sheet asked for is not determined"
    elif shape == "auto":
        shape_source = "body (signed-distance field), chosen automatically"
    else:
        shape_source = "body (signed-distance field), as asked"
    provenance.append(
        _entry(
            f"Shape of the intrusion {unit}",
            "assumed",
            shape_source,
            used_for,
            f"not a sheet: {reason}" if reason else "",
        )
    )
    anisotropy = settings.anisotropy
    if isinstance(anisotropy, Ellipsoid):
        ellipsoid, how = anisotropy, "given"
        note = (
            f"ratios major/semi-major {anisotropy.semi_major_ratio:.2f}, "
            f"major/minor {anisotropy.minor_ratio:.2f}"
        )
    elif anisotropy is None:
        ellipsoid, how, note = None, "none", "isotropic field"
    elif anisotropy == "auto":
        inside = _intercepts_of(constraints, traces)
        top, bottom = _ends(inside[inside["state"] > 0])
        try:
            ellipsoid, found = estimate_intercept_anisotropy(top, bottom)
            how = "automatic"
            note = (
                f"intercept midpoints, {found['n_intercepts']} intercepts: ratios major/semi-major "
                f"{ellipsoid.semi_major_ratio:.2f}, major/minor {ellipsoid.minor_ratio:.2f}"
                + "".join(f"; {w}" for w in found["warnings"])
            )
        except ValueError as error:
            ellipsoid, how, note = (
                None,
                "automatic",
                f"no automatic anisotropy ({error}): isotropic",
            )
    else:
        raise TypeError("anisotropy must be an Ellipsoid, 'auto' or None")
    field = fit_field(
        constraints.xyz, constraints.value, anisotropy=ellipsoid, kernel=settings.kernel,
        neighbors=settings.neighbors,
    )  # fmt: skip
    provenance.append(_entry(f"Anisotropy of the intrusion {unit}", "assumed", how, used_for, note))
    hole_rule = "a tenth of the median logged hole length"
    for item, value, given_value, default in (
        ("Cap of the signed distance", constraints.cap, settings.cap,
         "median length of the unit's intercepts"),
        ("Point spacing near contacts", constraints.sample_spacing, settings.sample_spacing,
         f"smaller of a quarter of the median intercept and {hole_rule}"),
        ("Point spacing away from contacts", constraints.far_spacing, settings.far_spacing,
         hole_rule),
    ):  # fmt: skip
        provenance.append(
            _entry(
                f"{item}, intrusion {unit}",
                "assumed",
                _source(given_value, default),
                used_for,
                f"{value:.2f} m; {len(constraints)} points ({constraints.n_contacts} contacts), "
                + ("global fit" if field.neighbors is None else "local fit"),
                value=float(value),
            )
        )
    return IntrusionField(
        unit,
        field,
        constraints=constraints,
        cap=constraints.cap,
        sample_spacing=constraints.sample_spacing,
        far_spacing=constraints.far_spacing,
        anisotropy=ellipsoid,
        kernel=settings.kernel,
        hole_spacing=hole_spacing,
        settings=settings,
        provenance=provenance,
    )


# --------------------------------------------------------------------------
# age order
# --------------------------------------------------------------------------


def order_intrusions(
    intrusions: Sequence[str], age_order: Sequence[str] | None = None
) -> tuple[tuple[str, ...], list[ProvenanceEntry]]:
    """The intrusions from the oldest to the youngest, and what the record says about it.

    With ``age_order`` (oldest first; it must list every intrusion once) that order is
    used. Without it the order of ``intrusions`` is kept (the order of the unit
    dictionary); with more than one intrusion this is recorded as an assumption, and the
    age order as a missing input.
    """
    listed = tuple(str(u) for u in intrusions)
    if len(set(listed)) != len(listed):
        raise ValueError(f"an intrusion is listed twice: {listed}")
    used_for = "order of the intrusion overprints (a later intrusion wins where two overlap)"
    if age_order is not None:
        order = tuple(str(u) for u in age_order)
        if sorted(order) != sorted(listed):
            raise ValueError(f"the age order {order} must list every intrusion once: {listed}")
        if len(order) < 2:
            return order, []
        entry = _entry(
            "Age order of intrusions",
            "assumed",
            "given by the user, oldest first",
            used_for,
            " < ".join(order),
        )
        return order, [entry]
    if len(listed) < 2:
        return listed, []
    text = " < ".join(listed)
    return listed, [
        _entry(
            "Age order of intrusions",
            "missing",
            "the geologist (cross-cutting relations seen in core, pits or on the map)",
            used_for,
            f"not given; {len(listed)} intrusions: {', '.join(listed)}",
        ),
        _entry(
            "Age order of intrusions used",
            "assumed",
            "default (the order of the unit dictionary, the last one the youngest)",
            used_for,
            text,
        ),
    ]


def mask_younger(units: pd.Series, order: Sequence[str], unit: str) -> pd.Series:
    """``units`` with the intrusions younger than ``unit`` made unknown (NaN).

    ``order`` lists the intrusions from the oldest to the youngest. What a younger
    intrusion replaced is not observed, so its intervals are neither inside nor outside
    the older one.
    """
    order = [str(u) for u in order]
    if str(unit) not in order:
        raise ValueError(f"{unit!r} is not in the age order {order}")
    younger = order[order.index(str(unit)) + 1 :]
    return units.where(~units.isin(younger), np.nan)


# --------------------------------------------------------------------------
# reading an event back
# --------------------------------------------------------------------------


def event_from_manifest(
    entry: Mapping[str, Any],
    fields: Mapping[str, ImplicitField],
    inputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> SequenceField | IntrusionField:
    """Rebuild a ``sequence`` or ``intrusion`` event from its ``geology.json`` entry and
    its stored fields (keyed by the last part of the field file name)."""
    mode = entry.get("mode")
    if mode == "sequence":
        return SequenceField.from_manifest(entry, fields, inputs)
    if mode == "intrusion":
        return IntrusionField.from_manifest(entry, fields, inputs)
    raise ValueError(f"event mode {mode!r} is not 'sequence' or 'intrusion'")
