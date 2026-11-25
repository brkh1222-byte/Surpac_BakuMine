"""Rock units from the logs: one signed-distance field per unit, the largest field wins.

Theory: research.md 5.7.2. Signatures: geology_contracts.md section 7.

For every unit ``i`` of a group, :func:`fit_unit_fields` takes the capped signed down-hole
distance to the nearest contact of ``i`` along the holes
(:func:`~bakumine.core.geometry.implicit.extract_constraints`) and interpolates it
(:func:`~bakumine.core.geometry.implicit.fit_field`): ``f_i``. A point takes the unit with
the largest field, ``L(p) = argmax_i f_i(p)`` (the first unit on ties), so the units
neither overlap nor leave gaps, whatever the fields. All fields of a group share the cap,
the two point spacings, the kernel and the anisotropy: with different caps the unit with
the larger one would win between the holes for no geological reason.

The boundary of unit ``i`` is the zero level of its margin
``g_i = f_i - max_{j != i} f_j`` (:func:`margins`). Margins sampled once on a regular grid
are interpolated linearly on the six Kuhn tetrahedra of each cell
(:func:`tetra_interpolate`), the subdivision the isosurface uses, so the labels from the
grid and the unit meshes (:func:`unit_meshes`) agree.

A :class:`LithologyStack` applies several events in order (a later one wins where it
applies). This module fits the ``argmax`` event; ``sequence`` and ``intrusion`` events
come from :mod:`bakumine.core.geology.sequence`.

Optional methods of an event, used by :class:`~bakumine.core.geology.model.GeologyModel`
when present:

``field_files() -> dict[suffix, (ImplicitField, xyz, values)]``
    the fields to store as ``fields/<block>_<event>_<suffix>.npz`` with the inputs of
    their fit;
``to_manifest() -> dict``
    extra keys of the event's entry in ``geology.json``;
``unit_margins(values) -> array (len(units), ...)``
    a margin per unit (positive inside the unit) from node values ``(k, ...)``, for the
    display meshes. Without it the node values themselves are used, which is right when
    there is one value per unit.

Coordinates are x = east, y = north, z = up.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geology.provenance import ProvenanceEntry
from bakumine.core.geometry.implicit import (
    KERNELS,
    Constraints,
    ImplicitField,
    estimate_anisotropy,
    extract_constraints,
    fit_field,
)
from bakumine.core.geometry.isosurface import GridSpec, marching_tetrahedra
from bakumine.core.geometry.mesh import Mesh

__all__ = [
    "AUTO_MAX_RATIO",
    "EVENT_MODES",
    "EventSpec",
    "FieldSettings",
    "LithologyEvent",
    "LithologyStack",
    "UnitFields",
    "event_from_manifest",
    "fit_stack",
    "fit_unit_fields",
    "group_contacts",
    "group_runs",
    "logged_hole_points",
    "margins",
    "tetra_interpolate",
    "unit_meshes",
]

EVENT_MODES = ("argmax", "sequence", "intrusion")
#: largest axis ratio of the automatic anisotropy of a group (the plane of its contacts)
AUTO_MAX_RATIO = 4.0
_TASK = "P9.T7"
_UNIT_COLUMN = "__unit__"
_CHUNK = 200_000


# --------------------------------------------------------------------------
# settings and protocols
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class FieldSettings:
    """How the fields of one event are fitted.

    ``anisotropy``: an ``Ellipsoid``, ``"auto"`` (the plane of the group's contact points,
    see :func:`fit_unit_fields`) or None (isotropic). ``cap``, ``sample_spacing`` and
    ``far_spacing`` are those of ``extract_constraints``; their defaults are computed once
    for the whole group. ``max_extrapolation`` is not applied to the fields (stored fields
    carry no clip): the geology model applies it.
    """

    anisotropy: Ellipsoid | str | None = None
    cap: float | None = None
    sample_spacing: float | None = None
    far_spacing: float | None = None
    kernel: str = "linear"
    neighbors: int | str | None = "auto"
    thin: float | None = None
    max_extrapolation: float | None = None


@runtime_checkable
class LithologyEvent(Protocol):
    """One modelling event of a stack. See the module text for the optional methods."""

    mode: str
    units: tuple[str, ...]
    provenance: list[ProvenanceEntry]

    def node_values(self, points) -> np.ndarray: ...

    def labels_from_values(self, values) -> np.ndarray: ...


@dataclass(frozen=True)
class EventSpec:
    """An event to fit: ``argmax`` (``units`` = the group), ``sequence`` (``units`` top to
    bottom) or ``intrusion`` (one unit)."""

    mode: str
    units: tuple[str, ...]
    settings: FieldSettings = FieldSettings()
    thickness: Mapping[str, float] | None = None

    def __post_init__(self) -> None:
        if self.mode not in EVENT_MODES:
            raise ValueError(f"mode must be one of {EVENT_MODES}, got {self.mode!r}")
        units = tuple(str(u) for u in self.units)
        if not units:
            raise ValueError("an event needs at least one unit")
        if len(set(units)) != len(units):
            raise ValueError(f"an event lists a unit twice: {units}")
        if self.mode == "intrusion" and len(units) != 1:
            raise ValueError("an intrusion event has exactly one unit")
        if self.thickness is not None and self.mode != "sequence":
            raise ValueError("thickness applies to a sequence event only")
        object.__setattr__(self, "units", units)


# --------------------------------------------------------------------------
# margins and grid interpolation
# --------------------------------------------------------------------------


def margins(values: np.ndarray) -> np.ndarray:
    """``g_i = f_i - max_{j != i} f_j`` along the last axis: ``(n, k) -> (n, k)``.

    ``g_i > 0`` exactly where unit ``i`` has the strictly largest field. With one column
    the margin is the value itself. A row with a NaN gives NaN.
    """
    v = np.asarray(values, dtype=np.float64)
    if v.ndim < 1:
        raise ValueError("values must have at least one axis")
    k = v.shape[-1]
    if k == 0:
        raise ValueError("values has no columns")
    if k == 1:
        return v.copy()
    flat = v.reshape(-1, k)
    bad = np.isnan(flat).any(axis=1)
    work = np.where(bad[:, None], 0.0, flat)
    rows = np.arange(len(work))
    first = np.argmax(work, axis=1)
    top = work[rows, first]
    rest = work.copy()
    rest[rows, first] = -np.inf
    second = rest.max(axis=1)
    out = work - top[:, None]
    out[rows, first] = top - second
    out[bad] = np.nan
    return out.reshape(v.shape)


def _argmax_labels(values) -> np.ndarray:
    v = np.asarray(values, dtype=np.float64)
    if v.ndim != 2:
        raise ValueError(f"values must have shape (n, k), got {v.shape}")
    bad = np.isnan(v).any(axis=1)
    out = np.argmax(np.where(bad[:, None], 0.0, v), axis=1).astype(np.int16)
    out[bad] = -1
    return out


def tetra_interpolate(values: np.ndarray, grid: GridSpec, points) -> np.ndarray:
    """Linear interpolation of node values on the six Kuhn tetrahedra of each cell.

    ``values`` has shape ``(k, nx, ny, nz)`` or ``(nx, ny, nz)`` on a world-aligned grid
    (``rotation`` None); the result is ``(n, k)`` or ``(n,)``. For the local coordinates
    ``f`` of a point in its cell, sorted so that ``f_a >= f_b >= f_c``, the point lies in
    the tetrahedron ``0, e_a, e_a + e_b, e_a + e_b + e_c`` with the weights
    ``1 - f_a, f_a - f_b, f_b - f_c, f_c`` (research.md 5.7.2). Points outside the grid
    (and NaN points) give NaN. The node values are reproduced exactly at the nodes and a
    function that is linear in x, y, z is reproduced everywhere.
    """
    if grid.rotation is not None:
        raise ValueError("tetra_interpolate needs a world-aligned grid (rotation None)")
    single = np.ndim(values) == 3
    if np.ndim(values) not in (3, 4):
        raise ValueError(
            f"values must have shape (k, nx, ny, nz) or (nx, ny, nz), got {np.shape(values)}"
        )
    shape = tuple(np.shape(values)[-3:])
    if shape != grid.shape:
        raise ValueError(f"values are on a grid of {shape} nodes, the grid has {grid.shape}")
    if min(shape) < 2:
        raise ValueError("the grid needs at least 2 nodes along every axis")
    array = values if isinstance(values, np.ndarray) else np.asarray(values)
    k = 1 if single else array.shape[0]
    flat_values = array.reshape(k, -1)
    pts = np.asarray(points, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
    n_axis = np.array(shape, dtype=np.int64)
    stride = np.array([shape[1] * shape[2], shape[2], 1], dtype=np.int64)
    out = np.full((len(pts), k), np.nan)
    for a in range(0, len(pts), _CHUNK):
        u = (pts[a : a + _CHUNK] - grid.origin) / grid.spacing
        with np.errstate(invalid="ignore"):
            good = (np.isfinite(u) & (u >= 0.0) & (u <= n_axis - 1)).all(axis=1)
        if not good.any():
            continue
        u = u[good]
        cell = np.minimum(np.floor(u).astype(np.int64), n_axis - 2)
        f = u - cell
        order = np.argsort(-f, axis=1, kind="stable")
        fs = np.take_along_axis(f, order, axis=1)
        weights = (1.0 - fs[:, 0], fs[:, 0] - fs[:, 1], fs[:, 1] - fs[:, 2], fs[:, 2])
        node = cell @ stride
        acc = np.zeros((k, len(u)))
        for step in range(4):
            if step:
                node = node + stride[order[:, step - 1]]
            # a zero weight must not turn a non-finite node value into NaN
            term = flat_values[:, node].astype(np.float64, copy=False)
            w = weights[step]
            with np.errstate(invalid="ignore"):
                acc += np.where(w[None, :] == 0.0, 0.0, w[None, :] * term)
        block = out[a : a + _CHUNK]
        block[good] = acc.T
    return out[:, 0] if single else out


# --------------------------------------------------------------------------
# runs, contacts and hole points of a group
# --------------------------------------------------------------------------


def _unit_table(table: pd.DataFrame, units: pd.Series, group: Sequence[str]) -> pd.DataFrame:
    for name in ("hole_id", "depth_from", "depth_to"):
        if name not in table.columns:
            raise ValueError(f"interval table has no column {name!r}")
    if len(units) != len(table):
        raise ValueError(f"units has {len(units)} values, the table has {len(table)} rows")
    if isinstance(units, pd.Series):
        if not units.index.equals(table.index):
            raise ValueError("units must have the index of the table")
        column = pd.Series(units.to_numpy(dtype=object))
    else:
        column = pd.Series(np.asarray(units, dtype=object))
    column = column.where(column.isin(list(group)), np.nan)
    frame = table[["hole_id", "depth_from", "depth_to"]].copy()
    frame[_UNIT_COLUMN] = column.to_numpy(dtype=object)
    return frame


def _runs_of(frame: pd.DataFrame, group: Sequence[str], traces: HoleTraces) -> pd.DataFrame:
    """Merged runs of every unit of the group: ``hole_id, depth_from, depth_to, unit``."""
    parts = []
    for unit in group:
        probe = extract_constraints(
            frame, _UNIT_COLUMN, unit, traces, cap=1.0, sample_spacing=1e12, far_spacing=1e12
        )
        own = probe.intercepts[probe.intercepts["state"] > 0].drop(columns="state")
        parts.append(own.assign(unit=unit))
    runs = pd.concat(parts, ignore_index=True)
    return runs.sort_values(["hole_id", "depth_from"], kind="stable").reset_index(drop=True)


def group_runs(
    table: pd.DataFrame, units: pd.Series, group: Sequence[str], traces: HoleTraces
) -> pd.DataFrame:
    """Merged logged runs of the group's units: ``hole_id, depth_from, depth_to, unit``.

    The runs are those ``extract_constraints`` builds (equal neighbours merged, cut at the
    end of the trace); intervals of other units are unknown and belong to no run.
    """
    group = tuple(str(u) for u in group)
    return _runs_of(_unit_table(table, units, group), group, traces)


def _contacts_of(runs: pd.DataFrame, traces: HoleTraces, tol: float = 1e-6) -> pd.DataFrame:
    if len(runs) < 2:
        return pd.DataFrame(columns=["hole_id", "depth", "above", "below", "x", "y", "z"])
    hole = runs["hole_id"].to_numpy(dtype=object)
    a = runs["depth_from"].to_numpy()
    b = runs["depth_to"].to_numpy()
    unit = runs["unit"].to_numpy(dtype=object)
    touch = (hole[1:] == hole[:-1]) & (np.abs(a[1:] - b[:-1]) <= tol) & (unit[1:] != unit[:-1])
    idx = np.flatnonzero(touch)
    xyz, _ = traces.xyz_at(hole[idx + 1], a[idx + 1], depth_tolerance=tol)
    out = pd.DataFrame(
        {
            "hole_id": hole[idx + 1],
            "depth": a[idx + 1],
            "above": unit[idx],
            "below": unit[idx + 1],
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }
    )
    return out[np.isfinite(xyz).all(axis=1)].reset_index(drop=True)


def group_contacts(
    table: pd.DataFrame, units: pd.Series, group: Sequence[str], traces: HoleTraces
) -> pd.DataFrame:
    """Contacts between two units of the group: ``hole_id, depth, above, below, x, y, z``.

    A contact exists where two runs of different units abut; an unknown interval between
    them hides it.
    """
    return _contacts_of(group_runs(table, units, group, traces), traces)


def logged_hole_points(
    table: pd.DataFrame,
    traces: HoleTraces,
    *,
    step: float = 5.0,
    known: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    """Points every ``<= step`` metres along the logged intervals of the holes, ``(n, 3)``.

    ``known`` (bool, one per row) keeps only those intervals. These are the points the
    geology model measures its distance to data from.
    """
    if not (np.isfinite(step) and step > 0.0):
        raise ValueError("step must be > 0")
    a = pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(dtype=np.float64)
    keep = np.isfinite(a) & np.isfinite(b) & (b > a)
    if known is not None:
        keep &= np.asarray(known, dtype=bool)
    hole = table["hole_id"].to_numpy(dtype=object)[keep]
    a, b = a[keep], b[keep]
    if len(a) == 0:
        return np.zeros((0, 3))
    n = np.maximum(1, np.ceil((b - a) / step - 1e-9).astype(np.int64))
    rep = np.repeat(np.arange(len(a)), n + 1)
    start = np.concatenate([[0], np.cumsum(n + 1)[:-1]])
    position = np.arange(len(rep)) - start[rep]
    depth = a[rep] + (b[rep] - a[rep]) * position / n[rep]
    xyz, _ = traces.xyz_at(hole[rep], depth, depth_tolerance=1e-6)
    xyz = xyz[np.isfinite(xyz).all(axis=1)]
    return np.unique(xyz, axis=0)


def _hole_spacing(constraints: Sequence[Constraints]) -> float | None:
    """Median over the holes of the typical distance to the nearest other hole.

    For one hole: the median, over its constraint points, of the 3-D distance to the
    nearest point of any other hole. For parallel holes this is their spacing.
    """
    xyz = np.concatenate([c.xyz for c in constraints])
    hole = np.concatenate([c.hole_id for c in constraints])
    codes, names = pd.factorize(hole)
    if len(names) < 2:
        return None
    per_hole = []
    for h in range(len(names)):
        own = codes == h
        distance, _ = cKDTree(xyz[~own]).query(xyz[own])
        per_hole.append(float(np.median(distance)))
    return float(np.median(per_hole))


def _auto_anisotropy(contacts: pd.DataFrame) -> tuple[Ellipsoid | None, str]:
    """The plane of the contacts: principal components after each contact's mean is removed."""
    parts = []
    for _, rows in contacts.groupby(["above", "below"], sort=False):
        if len(rows) >= 3:
            p = rows[["x", "y", "z"]].to_numpy(dtype=np.float64)
            parts.append(p - p.mean(axis=0))
    if not parts or sum(len(p) for p in parts) < 4:
        return None, "too few contact points for an automatic anisotropy: isotropic"
    try:
        ellipsoid = estimate_anisotropy(np.concatenate(parts), max_ratio=AUTO_MAX_RATIO)
    except ValueError as error:
        return None, f"no automatic anisotropy ({error}): isotropic"
    pole = np.asarray(ellipsoid.minor_axis, dtype=np.float64)
    pole = pole if pole[2] >= 0.0 else -pole
    dip = float(np.degrees(np.arccos(np.clip(pole[2], -1.0, 1.0))))
    azimuth = float(np.degrees(np.arctan2(pole[0], pole[1])) % 360.0)
    note = (
        f"plane of {sum(len(p) for p in parts)} contact points: dip {dip:.0f}, dip azimuth "
        f"{azimuth:03.0f}; ratios major/semi-major {ellipsoid.semi_major_ratio:.2f}, "
        f"major/minor {ellipsoid.minor_ratio:.2f} (limited to {AUTO_MAX_RATIO:g})"
    )
    return ellipsoid, note


# --------------------------------------------------------------------------
# the argmax event
# --------------------------------------------------------------------------


class UnitFields:
    """The fields of one group: mode ``argmax`` (the largest field wins).

    Attributes: ``units``, ``fields`` and ``constraints`` (one per unit; ``constraints``
    is empty for fields read from a folder), the shared ``cap``, ``sample_spacing``,
    ``far_spacing``, ``kernel`` and ``anisotropy``, ``inputs`` (the ``(xyz, values)`` each
    field was fitted to), ``hole_spacing`` (median plan distance between neighbouring
    holes, None if unknown), ``settings`` and ``provenance``.
    """

    mode = "argmax"

    def __init__(
        self,
        units: Sequence[str],
        fields: Sequence[ImplicitField],
        *,
        cap: float,
        sample_spacing: float,
        far_spacing: float,
        anisotropy: Ellipsoid | None = None,
        constraints: Sequence[Constraints] = (),
        inputs: Sequence[tuple[np.ndarray, np.ndarray]] | None = None,
        kernel: str = "linear",
        hole_spacing: float | None = None,
        settings: FieldSettings | None = None,
        provenance: Sequence[ProvenanceEntry] = (),
    ) -> None:
        self.units = tuple(str(u) for u in units)
        self.fields = tuple(fields)
        self.constraints = tuple(constraints)
        if not self.units or len(self.units) != len(self.fields):
            raise ValueError("one field per unit is needed, and at least one unit")
        if len(set(self.units)) != len(self.units):
            raise ValueError(f"a unit is listed twice: {self.units}")
        if self.constraints and len(self.constraints) != len(self.units):
            raise ValueError("constraints must be empty or one per unit")
        if inputs is None:
            inputs = [(c.xyz, c.value) for c in self.constraints]
        self.inputs = tuple((np.asarray(p), np.asarray(v)) for p, v in inputs)
        if self.inputs and len(self.inputs) != len(self.units):
            raise ValueError("inputs must be empty or one per unit")
        self.cap = float(cap)
        self.sample_spacing = float(sample_spacing)
        self.far_spacing = float(far_spacing)
        self.anisotropy = anisotropy
        self.kernel = kernel
        self.hole_spacing = None if hole_spacing is None else float(hole_spacing)
        self.settings = settings if settings is not None else FieldSettings()
        self.provenance = list(provenance)

    def values(self, points) -> np.ndarray:
        """Exact ``f_i`` at world points ``(n, 3)``: ``(n, k)``. No clip is applied."""
        pts = np.asarray(points, dtype=np.float64)
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {pts.shape}")
        return np.column_stack([f(pts, clipped=False) for f in self.fields])

    def node_values(self, points) -> np.ndarray:
        """Margins ``g_i = f_i - max_{j != i} f_j`` at the points: ``(n, k)``."""
        return margins(self.values(points))

    def labels_from_values(self, values) -> np.ndarray:
        """Index of the largest value of each row (the first on ties); -1 for a NaN row."""
        return _argmax_labels(values)

    def labels(self, points) -> np.ndarray:
        """Exact labels: int16 index into ``units``."""
        return _argmax_labels(self.values(points))

    # persistence hooks ----------------------------------------------------

    def field_files(self) -> dict[str, tuple[ImplicitField, np.ndarray, np.ndarray]]:
        if not self.inputs:
            raise ValueError("these fields do not carry the inputs of their fit: cannot store")
        return {
            u: (f, xyz, v)
            for u, f, (xyz, v) in zip(self.units, self.fields, self.inputs, strict=True)
        }

    def to_manifest(self) -> dict[str, Any]:
        return {
            "cap": self.cap,
            "anisotropy": None if self.anisotropy is None else self.anisotropy.to_dict(),
            "sample_spacing": self.sample_spacing,
            "far_spacing": self.far_spacing,
            "kernel": self.kernel,
            "hole_spacing": self.hole_spacing,
        }

    @classmethod
    def from_manifest(
        cls,
        entry: Mapping[str, Any],
        fields: Mapping[str, ImplicitField],
        inputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
    ) -> UnitFields:
        units = tuple(entry["units"])
        anisotropy = entry.get("anisotropy")
        return cls(
            units,
            [fields[u] for u in units],
            cap=entry["cap"],
            sample_spacing=entry.get("sample_spacing", np.nan),
            far_spacing=entry.get("far_spacing", np.nan),
            anisotropy=None if anisotropy is None else Ellipsoid.from_dict(anisotropy),
            inputs=[inputs[u] for u in units],
            kernel=entry.get("kernel", "linear"),
            hole_spacing=entry.get("hole_spacing"),
        )


def event_from_manifest(
    entry: Mapping[str, Any],
    fields: Mapping[str, ImplicitField],
    inputs: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> LithologyEvent:
    """Rebuild an event from its ``geology.json`` entry and its stored fields.

    ``fields`` and ``inputs`` are keyed by the last part of the field file name (the unit
    code, or ``phi``). ``sequence`` and ``intrusion`` events are rebuilt by
    ``bakumine.core.geology.sequence.event_from_manifest`` (same arguments).
    """
    mode = entry.get("mode")
    if mode == "argmax":
        return UnitFields.from_manifest(entry, fields, inputs)
    if mode in ("sequence", "intrusion"):
        from bakumine.core.geology import sequence

        return sequence.event_from_manifest(entry, fields, inputs)
    raise ValueError(f"unknown event mode {mode!r}; one of {EVENT_MODES}")


def _settings_source(given: float | None, default: str) -> str:
    return "given in the settings" if given is not None else f"default ({default})"


def fit_unit_fields(
    table: pd.DataFrame,
    units: pd.Series,
    group: Sequence[str],
    traces: HoleTraces,
    settings: FieldSettings = FieldSettings(),
) -> UnitFields:
    """One signed-distance field per unit of ``group``, on a common scale.

    ``table`` has ``hole_id, depth_from, depth_to``; ``units`` gives the unit of each row
    (same index). Units outside ``group`` (and NaN) are unknown: no points, and no contact
    at their ends. Defaults, the same for every unit: ``cap`` = the median length of all
    runs of the group's units; ``far_spacing`` = a tenth of the median logged hole length;
    ``sample_spacing`` = ``min(cap / 4, far_spacing default)``.

    ``anisotropy="auto"`` takes the plane of the contacts between the group's units: the
    contact points of each pair of units minus their mean, pooled, then principal
    components, with the ratios limited to :data:`AUTO_MAX_RATIO`. It is a default, not a
    measurement; with fewer than four usable contact points the fields are isotropic.

    A unit without any logged run raises ``ValueError``.
    """
    group = tuple(str(u) for u in group)
    if not group:
        raise ValueError("the group has no units")
    if len(set(group)) != len(group):
        raise ValueError(f"the group lists a unit twice: {group}")
    if settings.kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {settings.kernel!r}")
    frame = _unit_table(table, units, group)
    runs = _runs_of(frame, group, traces)
    absent = [u for u in group if not (runs["unit"] == u).any()]
    if absent:
        raise ValueError(f"no logged run of unit(s) {', '.join(absent)}: cannot fit their fields")

    length = (runs["depth_to"] - runs["depth_from"]).to_numpy(dtype=np.float64)
    hole_length = float(np.median(pd.Series(length).groupby(runs["hole_id"].to_numpy()).sum()))
    cap = float(settings.cap) if settings.cap is not None else float(np.median(length))
    far = hole_length / 10.0 if settings.far_spacing is None else float(settings.far_spacing)
    if settings.sample_spacing is not None:
        near = float(settings.sample_spacing)
    else:
        near = min(cap / 4.0, hole_length / 10.0)
    for name, v in (("cap", cap), ("sample_spacing", near), ("far_spacing", far)):
        if not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be a positive finite number, got {v!r}")
    far = max(far, near)

    anisotropy = settings.anisotropy
    if isinstance(anisotropy, str):
        if anisotropy != "auto":
            raise ValueError("anisotropy must be an Ellipsoid, 'auto' or None")
        ellipsoid, note = _auto_anisotropy(_contacts_of(runs, traces))
        how = "automatic"
    elif anisotropy is None:
        ellipsoid, how, note = None, "none", "isotropic fields"
    elif isinstance(anisotropy, Ellipsoid):
        ellipsoid, how = anisotropy, "given"
        note = (
            f"ratios major/semi-major {anisotropy.semi_major_ratio:.2f}, "
            f"major/minor {anisotropy.minor_ratio:.2f}"
        )
    else:
        raise TypeError("anisotropy must be an Ellipsoid, 'auto' or None")

    constraints, fields = [], []
    for unit in group:
        c = extract_constraints(
            frame, _UNIT_COLUMN, unit, traces,
            cap=cap, sample_spacing=near, far_spacing=far, thin=settings.thin,
        )  # fmt: skip
        constraints.append(c)
        fields.append(
            fit_field(
                c.xyz,
                c.value,
                anisotropy=ellipsoid,
                kernel=settings.kernel,
                neighbors=settings.neighbors,
            )  # fmt: skip
        )

    label = " + ".join(group)
    used_for = f"rock-unit fields of the group {label} (largest field)"
    rows = frame[_UNIT_COLUMN]
    metres_all = pd.to_numeric(frame["depth_to"], errors="coerce") - pd.to_numeric(
        frame["depth_from"], errors="coerce"
    )
    provenance = []
    for unit, c in zip(group, constraints, strict=True):
        own = runs[runs["unit"] == unit]
        mask = (rows == unit).to_numpy()
        provenance.append(
            ProvenanceEntry(
                item=f"Rock-type log: unit {unit}",
                kind="measured",
                source="drillhole log, intervals mapped to the unit",
                used_for=used_for,
                detail=(
                    f"{int(mask.sum())} intervals, {float(metres_all[mask].sum()):.0f} m, "
                    f"{own['hole_id'].nunique()} holes, {len(own)} runs; {len(c)} points "
                    f"({c.n_contacts} contacts)"
                ),
                stage="lithology",
                task=_TASK,
                count=int(mask.sum()),
            )
        )
    known = rows.notna().to_numpy()
    provenance.append(
        ProvenanceEntry(
            item=f"Rock-type log: group {label}",
            kind="measured",
            source="drillhole log, intervals mapped to the units of the group",
            used_for=used_for,
            detail=(
                f"{int(known.sum())} intervals, {float(metres_all[known].sum()):.0f} m, "
                f"{runs['hole_id'].nunique()} holes; {int((~known).sum())} intervals of other "
                "codes are unknown (no points, no contacts)"
            ),
            stage="lithology",
            task=_TASK,
            count=int(known.sum()),
        )
    )
    provenance.append(
        ProvenanceEntry(
            item=f"Modelling mode of the group {label}",
            kind="assumed",
            source="largest signed-distance field (argmax); no age order used",
            used_for=used_for,
            detail=(
                f"kernel {settings.kernel}; "
                + ("global fits" if all(f.neighbors is None for f in fields) else "local fits")
                + (f"; points thinned to one per {settings.thin:g} m" if settings.thin else "")
            ),
            stage="lithology",
            task=_TASK,
        )
    )
    provenance.append(
        ProvenanceEntry(
            item=f"Anisotropy of the group {label}",
            kind="assumed",
            source=how,
            used_for=used_for,
            detail=note,
            stage="lithology",
            task=_TASK,
        )
    )
    spacing_rule = "a tenth of the median logged hole length"
    for item, value, given, default in (
        ("Cap of the signed distance", cap, settings.cap, "median length of the group's runs"),
        ("Point spacing near contacts", near, settings.sample_spacing,
         f"smaller of cap / 4 and {spacing_rule}"),
        ("Point spacing away from contacts", far, settings.far_spacing, spacing_rule),
    ):  # fmt: skip
        provenance.append(
            ProvenanceEntry(
                item=f"{item}, group {label}",
                kind="assumed",
                source=_settings_source(given, default),
                used_for=used_for,
                detail=f"{value:.2f} m, the same for every unit of the group",
                stage="lithology",
                task=_TASK,
                value=float(value),
            )
        )
    return UnitFields(
        group,
        fields,
        cap=cap,
        sample_spacing=near,
        far_spacing=far,
        anisotropy=ellipsoid,
        constraints=constraints,
        kernel=settings.kernel,
        hole_spacing=_hole_spacing(constraints),
        settings=settings,
        provenance=provenance,
    )


# --------------------------------------------------------------------------
# a stack of events
# --------------------------------------------------------------------------


@dataclass
class LithologyStack:
    """The events of one fault block, applied in order: a later label that is not -1 wins.

    ``categories`` lists every event's units in order of first appearance.
    """

    block: str
    events: tuple[LithologyEvent, ...]
    categories: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        self.events = tuple(self.events)
        seen: dict[str, None] = {}
        for event in self.events:
            for unit in event.units:
                seen.setdefault(str(unit))
        self.categories = tuple(seen)

    @property
    def provenance(self) -> list[ProvenanceEntry]:
        return [entry for event in self.events for entry in event.provenance]

    def labels_from_values(self, values: Sequence[np.ndarray]) -> np.ndarray:
        """Labels from one ``(n, k)`` array of node values per event: int16 index into
        ``categories``, -1 where no event applies."""
        if len(values) != len(self.events):
            raise ValueError("one array of values per event is needed")
        index = {c: i for i, c in enumerate(self.categories)}
        out: np.ndarray | None = None
        for event, v in zip(self.events, values, strict=True):
            local = np.asarray(event.labels_from_values(v)).astype(np.int64)
            lookup = np.array([index[str(u)] for u in event.units] + [-1], dtype=np.int16)
            mapped = lookup[local]  # -1 picks the last entry
            if out is None:
                out = mapped.copy()
            else:
                out = np.where(mapped >= 0, mapped, out)
        if out is None:
            raise ValueError("the stack has no events")
        return out.astype(np.int16)

    def labels(self, points) -> np.ndarray:
        """Exact labels at world points ``(n, 3)``."""
        pts = np.asarray(points, dtype=np.float64)
        if not self.events:
            return np.full(len(pts), -1, dtype=np.int16)
        return self.labels_from_values([event.node_values(pts) for event in self.events])


def fit_stack(
    table: pd.DataFrame,
    units: pd.Series,
    traces: HoleTraces,
    events: Sequence[EventSpec],
    *,
    block: str = "ALL",
) -> LithologyStack:
    """Fit the events of one block in the order given."""
    fitted: list[LithologyEvent] = []
    for spec in events:
        if spec.mode == "argmax":
            fitted.append(fit_unit_fields(table, units, spec.units, traces, spec.settings))
        elif spec.mode == "sequence":
            from bakumine.core.geology.sequence import fit_sequence

            fitted.append(
                fit_sequence(
                    table,
                    units,
                    spec.units,
                    traces,
                    settings=spec.settings,
                    thickness=spec.thickness,
                )  # fmt: skip
            )
        elif spec.mode == "intrusion":
            from bakumine.core.geology.sequence import fit_intrusion

            fitted.append(
                fit_intrusion(table, units, spec.units[0], traces, settings=spec.settings)
            )
        else:
            raise ValueError(f"unknown event mode {spec.mode!r}; one of {EVENT_MODES}")
    return LithologyStack(block, tuple(fitted))


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------


def unit_meshes(
    node_margins: np.ndarray,
    grid: GridSpec,
    units: Sequence[str],
    *,
    clip: np.ndarray | None = None,
) -> dict[str, Mesh]:
    """One closed mesh per unit: the zero level of its margin on the grid.

    ``node_margins`` has shape ``(k, nx, ny, nz)``; ``clip`` ``(nx, ny, nz)`` is an upper
    limit applied to every margin (for example the depth below the topography, so nothing
    is kept above it). The values are padded with an outside layer, so each mesh is closed
    and capped flat at the grid boundary. On a grid edge where two units have the two
    largest fields at both ends their margins are opposite and the two meshes get the same
    vertex. A unit that wins at no node has no mesh and is left out.
    """
    units = tuple(str(u) for u in units)
    if np.ndim(node_margins) != 4 or np.shape(node_margins)[0] != len(units):
        raise ValueError(
            f"node_margins must have shape ({len(units)}, nx, ny, nz), got {np.shape(node_margins)}"
        )
    if tuple(np.shape(node_margins)[1:]) != grid.shape:
        raise ValueError("node_margins are not on the grid")
    if clip is not None and tuple(np.shape(clip)) != grid.shape:
        raise ValueError("clip is not on the grid")
    out: dict[str, Mesh] = {}
    for i, unit in enumerate(units):
        g = np.asarray(node_margins[i], dtype=np.float64)
        if clip is not None:
            g = np.fmin(g, clip)
        mesh = marching_tetrahedra(g, grid.origin, grid.spacing, grid.rotation, inside="above")
        if mesh.n_faces:
            out[unit] = mesh
    return out
