"""Cover and weathering base surfaces as a depth below the topography (research.md 5.7.4).

A weathering or cover base follows the ground, so the depth below the topography is
interpolated and the surface is ``z = topography - depth``:

1. :func:`surface_picks` reads the picks of one surface from one or several sources (the log,
   an extra intervals file, a rule on an assay column), combines them per hole and keeps what
   every source said. Codes of kind ``missing`` and ``fault-zone`` are unknown, so a fault zone
   across the contact gives a bracket, never a falsely exact pick.
2. :func:`depth_surface` fits the depth through the exact picks under the lower and upper
   bounds (:func:`bakumine.core.geometry.surface_interp.fit_surface`), evaluates it on a grid,
   keeps it at or below the topography (``depth >= 0``) and builds the DTM.
3. :func:`nest_surfaces` (several surfaces of one profile) or :func:`nest_under` (one surface
   under an existing DTM) pushes a deeper surface down wherever it would rise above the
   shallower one. This is a correction after the fact, not a joint estimate; the number of
   nodes moved is reported.

What is left out of a fit, and reported instead of used silently:

- holes with no information (kind ``none``): they change nothing, and are never zero depth;
- picks flagged ``conflict`` by :func:`bakumine.core.geology.picks.combine_picks` (the sources
  contradict each other), unless ``keep_conflicts`` is set;
- picks whose depth below the topography is not finite (outside the topography's extent).

Every result carries provenance entries (geology_contracts.md 3): *measured* = the picks and
bounds used, per source, and the threshold rule; *assumed* = the ``absent`` policy, the
default variogram, excluded conflicts, the clamp at the topography, nesting changes, and the
share of the surface farther than 2 x the median pick spacing from any pick; *missing* = the
holes in which the zone is not logged.
"""

from __future__ import annotations

import json
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.estimation.variogram import VariogramModel
from bakumine.core.geology.picks import (
    PICK_COLUMNS,
    PICK_KINDS,
    combine_picks,
    threshold_units,
    unit_picks,
)
from bakumine.core.geology.provenance import ProvenanceEntry
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geometry.dtm import DTM, dtm_from_grid
from bakumine.core.geometry.surface_interp import Surface2D, SurfaceFitError, fit_surface

FORMAT = "bakumine.depth_surface"
VERSION = 1
UNKNOWN_KINDS = ("missing", "fault-zone")  # unit kinds that say nothing about a base
EXTRAPOLATION_FACTOR = 2.0  # x the median pick spacing

_TASK = "P9.T9"
_STAGE = "surfaces"
_NESTING_ITEM = "Nesting of "
_INTERVAL_COLUMNS = ("hole_id", "depth_from", "depth_to")


def _plural(n: int, word: str) -> str:
    return f"{n} {word}" + ("" if n == 1 else "s")


def _entry(kind: str, item: str, source: str, surface: str, detail: str, **extra: Any):
    return ProvenanceEntry(
        item=item,
        kind=kind,
        source=source,
        used_for=f"surface {surface}",
        detail=detail,
        stage=_STAGE,
        task=_TASK,
        **extra,
    )


def surface_file_stem(name: str) -> str:
    """File stem of a surface: ``base:OXIDE`` -> ``base_OXIDE`` (geology_contracts.md 9)."""
    return str(name).replace(":", "_")


# ---------------------------------------------------------------------------- sources


@dataclass(frozen=True)
class PickSource:
    """Intervals of one source, classed for :func:`surface_picks`.

    ``units`` is aligned with ``table`` by position. ``target`` are the codes that make up
    the zone; ``unknown_codes`` say nothing (core loss, fault zones). ``threshold`` is the
    rule of a threshold source, kept here explicitly: the copy in ``units.attrs`` is lost
    whenever pandas copies the series.
    """

    source: str
    table: pd.DataFrame
    units: pd.Series
    target: tuple[str, ...]
    unknown_codes: tuple[str, ...] = ()
    threshold: Mapping[str, Any] | None = None
    description: str = ""


def _need_intervals(table: pd.DataFrame, what: str) -> None:
    lacking = [c for c in _INTERVAL_COLUMNS if c not in table.columns]
    if lacking:
        raise ValueError(f"{what} lacks the columns: {', '.join(lacking)}")


def log_source(
    table: pd.DataFrame,
    column: str,
    units: UnitDictionary,
    target: Collection[str],
    *,
    source: str = "log",
    description: str = "",
) -> PickSource:
    """A logged interval table (or an extra intervals file with ``source="extra"``).

    The codes of ``column`` are mapped with the dictionary. Units of kind ``missing`` and
    ``fault-zone`` are unknown: a core-loss or fault-zone interval across the contact makes
    the pick a bracket or a lower bound, not an exact depth.
    """
    _need_intervals(table, "the interval table")
    if column not in table.columns:
        raise ValueError(f"the interval table has no column {column!r}")
    if isinstance(target, str):
        target = (target,)
    target = tuple(dict.fromkeys(str(code) for code in target))
    if not target:
        raise ValueError("target must name at least one unit code")
    known = set(units.codes())
    absent = [code for code in target if code not in known]
    if absent:
        raise ValueError(f"not a unit of the dictionary: {', '.join(absent)}")
    unknown = tuple(units.codes(*UNKNOWN_KINDS))
    both = [code for code in target if code in unknown]
    if both:
        raise ValueError(
            f"units of kind {' / '.join(UNKNOWN_KINDS)} cannot be the target of a surface: "
            f"{', '.join(both)}"
        )
    return PickSource(
        source=source,
        table=table,
        units=units.map_codes(table[column]),
        target=target,
        unknown_codes=unknown,
        description=description or f"column {column}",
    )


def threshold_source(
    table: pd.DataFrame,
    column: str,
    *,
    below: float | None = None,
    above: float | None = None,
    description: str = "",
) -> PickSource:
    """A rule on a numeric column: the zone is where ``value < below`` (or ``> above``)."""
    _need_intervals(table, "the threshold table")
    series = threshold_units(table, column, below=below, above=above)
    rule = dict(series.attrs["threshold"])  # taken now: attrs do not survive a copy
    lengths = (
        pd.to_numeric(table["depth_to"], errors="coerce")
        - pd.to_numeric(table["depth_from"], errors="coerce")
    ).to_numpy(dtype=np.float64)
    lengths = lengths[np.isfinite(lengths) & (lengths > 0.0)]
    rule["median_interval"] = float(np.median(lengths)) if lengths.size else 0.0
    return PickSource(
        source="threshold",
        table=table,
        units=series,
        target=(rule["inside"],),
        threshold=rule,
        description=description or f"column {column}",
    )


@dataclass
class SurfacePicks:
    """The picks of one surface: per source and combined.

    ``table`` has :data:`PICK_COLUMNS` and ``conflict``, one row per hole.
    ``provenance`` holds what cannot be read from the table afterwards: the threshold rule,
    the ``absent`` policy and the conflict tolerance. :func:`depth_surface` adds the counts.
    """

    surface: str
    table: pd.DataFrame
    sources: dict[str, pd.DataFrame] = field(default_factory=dict)
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def surface_picks(
    sources: Sequence[PickSource],
    traces: HoleTraces,
    topography: DTM,
    *,
    surface: str,
    side: str = "base",
    mode: str = "deepest",
    absent: str = "unknown",
    conflict_tolerance: float | None = None,
) -> SurfacePicks:
    """Picks of ``surface`` from every source, combined per hole.

    ``sources`` are in order of priority (:func:`combine_picks`): the first exact pick wins,
    otherwise the tightest bounds. ``conflict_tolerance`` is the disagreement along the hole,
    in metres, above which two sources contradict each other. Default: the median interval
    length of the threshold tables (a rule on assays places the contact no finer than one
    sample), or 1e-6 without a threshold source.
    """
    sources = list(sources)
    if not sources:
        raise ValueError("give at least one source of picks")
    names = [s.source for s in sources]
    if len(set(names)) != len(names):
        raise ValueError(f"each source may be given once, got {', '.join(names)}")
    rules = [s.threshold for s in sources if s.threshold is not None]
    automatic = conflict_tolerance is None
    if automatic:
        conflict_tolerance = max([1e-6, *(float(r.get("median_interval", 0.0)) for r in rules)])
    conflict_tolerance = float(conflict_tolerance)
    if not (np.isfinite(conflict_tolerance) and conflict_tolerance >= 0.0):
        raise ValueError("conflict_tolerance must be >= 0")

    tables: dict[str, pd.DataFrame] = {}
    per_source: dict[str, dict] = {}
    for s in sources:
        result = unit_picks(
            s.table,
            s.units,
            s.target,
            traces,
            topography,
            surface=surface,
            side=side,
            mode=mode,
            absent=absent,
            unknown_codes=s.unknown_codes,
            source=s.source,
        )
        tables[s.source] = result.table
        d = result.diagnostics
        per_source[s.source] = {
            "counts": d["counts"],
            "description": s.description,
            "target": list(s.target),
            "unknown_codes": list(s.unknown_codes),
            "holes_skipped": d["holes_skipped"],
            "holes_outside_topography": d["holes_outside_topography"],
            "holes_absent_policy": d["holes_absent_policy"],
            "holes_overlapping_intervals": d["holes_overlapping_intervals"],
            "rows_dropped": d["rows_dropped"],
        }
    combined = combine_picks(*tables.values(), tolerance=conflict_tolerance)

    kinds = combined["kind"].to_numpy(dtype=object)
    n_none = int((kinds == "none").sum())
    policy_holes = sorted({str(h) for d in per_source.values() for h in d["holes_absent_policy"]})
    provenance: list[ProvenanceEntry] = []
    for s in sources:
        rule = s.threshold
        if rule is None:
            continue
        provenance.append(
            _entry(
                "measured",
                f"Threshold rule for {surface}",
                f"{s.description}: {rule['column']} {rule['operator']} {rule['value']:g}",
                surface,
                f"{_plural(int(rule['inside_count']), 'interval')} inside the rule, "
                f"{int(rule['outside_count'])} outside, {int(rule['missing_count'])} missing",
                count=int(rule["inside_count"]),
                value=float(rule["value"]),
            )
        )
    if absent == "zero":
        bound = "an upper" if side == "base" else "a lower"
        provenance.append(
            _entry(
                "assumed",
                f"Holes without the units of {surface}",
                "option absent = zero",
                surface,
                f"{_plural(len(policy_holes), 'hole')} read as the zone being absent: {bound} "
                "bound at the first other unit",
                count=len(policy_holes),
            )
        )
    else:
        provenance.append(
            _entry(
                "assumed",
                f"Holes without the units of {surface}",
                "default (absent = unknown)",
                surface,
                f"{_plural(n_none, 'hole')} give no information, not zero depth",
                count=n_none,
            )
        )
    if automatic and rules:
        provenance.append(
            _entry(
                "assumed",
                f"Conflict tolerance between the sources of {surface}",
                "default (median interval length of the threshold table)",
                surface,
                f"sources that differ by more than {conflict_tolerance:g} m along the hole "
                "are flagged as a conflict",
                value=conflict_tolerance,
            )
        )
    diagnostics = {
        "surface": surface,
        "side": side,
        "mode": mode,
        "absent": absent,
        "conflict_tolerance": conflict_tolerance,
        "sources": per_source,
        "thresholds": [dict(r) for r in rules],
        "holes": len(combined),
        "counts": {kind: int((kinds == kind).sum()) for kind in PICK_KINDS},
        "conflicts": int(combined["conflict"].sum()),
        "holes_absent_policy": policy_holes,
    }
    return SurfacePicks(
        surface=surface,
        table=combined,
        sources=tables,
        provenance=provenance,
        diagnostics=diagnostics,
    )


# ---------------------------------------------------------------------------- surface


@dataclass
class DepthSurface:
    """A base surface as a gridded depth below the topography.

    ``depth[i, j]`` is at ``origin + (i, j) * spacing``; it is ``>= 0`` and NaN where the
    topography is missing. ``dtm`` is ``topography_z - depth``. ``picks`` is the table handed
    in for this surface plus a ``used`` column (the rows that went into the fit).
    """

    name: str
    picks: pd.DataFrame
    fit: Surface2D
    origin: tuple[float, float]
    spacing: tuple[float, float]
    depth: np.ndarray
    topography_z: np.ndarray
    dtm: DTM
    n_nested: int = 0
    provenance: list[ProvenanceEntry] = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.depth.shape[0]), int(self.depth.shape[1]))

    def nodes(self) -> tuple[np.ndarray, np.ndarray]:
        """Plan coordinates ``(x, y)`` of the grid nodes, each of shape ``(nx, ny)``."""
        nx, ny = self.shape
        return np.meshgrid(
            self.origin[0] + self.spacing[0] * np.arange(nx),
            self.origin[1] + self.spacing[1] * np.arange(ny),
            indexing="ij",
        )

    def summary(self) -> dict[str, Any]:
        """``Surface2D.to_dict()`` plus the grid, the diagnostics and the provenance."""
        out = self.fit.to_dict()
        out.update(
            {
                "format": FORMAT,
                "version": VERSION,
                "name": self.name,
                "grid": {
                    "origin": [float(v) for v in self.origin],
                    "spacing": [float(v) for v in self.spacing],
                    "shape": list(self.shape),
                },
                "n_nested": int(self.n_nested),
                "diagnostics": self.diagnostics,
                "provenance": [entry.to_dict() for entry in self.provenance],
            }
        )
        return out

    def save(self, folder: str | Path) -> dict[str, Path]:
        """Write ``<stem>.obj``, ``<stem>.json`` and ``<stem>_picks.csv`` into ``folder``.

        The stem is the name with ``:`` replaced by ``_`` (the ``surfaces/`` files of a
        geology folder, geology_contracts.md 9). Returns the three paths by kind.
        """
        from bakumine.core.io.obj import write_obj

        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        stem = surface_file_stem(self.name)
        paths = {
            "obj": folder / f"{stem}.obj",
            "json": folder / f"{stem}.json",
            "picks": folder / f"{stem}_picks.csv",
        }
        write_obj(paths["obj"], self.dtm.mesh, decimals=None)
        paths["json"].write_text(json.dumps(self.summary(), indent=2), encoding="utf-8")
        self.picks.to_csv(paths["picks"], index=False)
        return paths


def _grid(rectangle: Any, spacing: float) -> tuple[tuple[float, float], tuple[int, int]]:
    try:
        x0, y0, x1, y1 = (float(v) for v in rectangle)
    except (TypeError, ValueError):
        raise ValueError("rectangle needs four numbers: xmin, ymin, xmax, ymax") from None
    if not (np.isfinite([x0, y0, x1, y1]).all() and x1 > x0 and y1 > y0):
        raise ValueError("rectangle needs xmin < xmax and ymin < ymax")
    spacing = float(spacing)
    if not (np.isfinite(spacing) and spacing > 0.0):
        raise ValueError("spacing must be > 0")
    origin = (float(np.floor(x0 / spacing) * spacing), float(np.floor(y0 / spacing) * spacing))
    # nodes up to the first one at or beyond the maximum (1e-9 cells of slack for rounding)
    shape = tuple(
        max(2, int(np.ceil((hi - lo) / spacing - 1e-9)) + 1)
        for lo, hi in ((origin[0], x1), (origin[1], y1))
    )
    return origin, shape


def _finite(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    return np.isfinite(frame[list(columns)].to_numpy(dtype=np.float64)).all(axis=1)


def _holes(frame: pd.DataFrame, mask: np.ndarray) -> list[str]:
    return [str(h) for h in frame.loc[mask, "hole_id"]]


def _pick_spacing(xy: np.ndarray) -> float:
    """Median distance from a pick to its nearest other pick (NaN with fewer than two)."""
    distinct = np.unique(xy, axis=0)
    if distinct.shape[0] < 2:
        return float("nan")
    return float(np.median(cKDTree(distinct).query(distinct, k=2)[0][:, 1]))


def depth_surface(
    picks: pd.DataFrame | SurfacePicks,
    topography: DTM,
    rectangle: tuple[float, float, float, float],
    spacing: float,
    *,
    name: str,
    method: str = "kriging",
    variogram: VariogramModel | None = None,
    drift: str = "constant",
    keep_conflicts: bool = False,
) -> DepthSurface:
    """Fit the depth below the topography of one surface and grid it.

    Parameters
    ----------
    picks : a picks table (:data:`PICK_COLUMNS`, optionally ``conflict``) or the
        :class:`SurfacePicks` of :func:`surface_picks`, whose provenance (threshold rule,
        ``absent`` policy) is then carried over. Rows of other surfaces are ignored.
    topography : the ground surface; nodes outside it get NaN.
    rectangle : ``(xmin, ymin, xmax, ymax)``. The grid origin is the minimum snapped down to
        a multiple of ``spacing``; the grid covers the rectangle.
    spacing : node spacing in metres (both directions).
    method, variogram, drift : see :func:`fit_surface`. Without a variogram the default
        model of the exact picks is used and reported as assumed.
    keep_conflicts : use the picks flagged ``conflict`` as well. By default they are left
        out of the fit, counted and reported.

    Exact picks are honoured by the fitted depth; lower and upper bounds limit it. A
    ``between`` pick is a lower bound at ``(x, y)`` and an upper bound at ``(x2, y2)``. The
    gridded depth is clamped at 0, so the surface never rises above the topography.

    Raises
    ------
    SurfaceFitError : no exact pick is left, or the picks are inconsistent.
    """
    inherited: list[ProvenanceEntry] = []
    picks_diagnostics: dict | None = None
    if isinstance(picks, SurfacePicks):
        inherited = list(picks.provenance)
        picks_diagnostics = picks.diagnostics
        table = picks.table
    else:
        table = picks
    lacking = [c for c in PICK_COLUMNS if c not in table.columns]
    if lacking:
        raise ValueError(f"the picks table lacks the columns: {', '.join(lacking)}")
    if not isinstance(name, str) or not name.strip():
        raise ValueError("name must be the surface name, e.g. 'base:OXIDE'")
    origin, shape = _grid(rectangle, spacing)
    spacing = float(spacing)

    mine = table[table["surface"].astype(str) == name].reset_index(drop=True).copy()
    if mine.empty:
        raise ValueError(f"the picks table has no row of surface {name!r}")
    bad = sorted(set(mine["kind"]) - set(PICK_KINDS))
    if bad:
        raise ValueError(f"unknown pick kinds: {', '.join(map(str, bad))}")
    kind = mine["kind"].to_numpy(dtype=object)
    if "conflict" in mine.columns:
        conflict = mine["conflict"].fillna(False).to_numpy(dtype=bool).copy()
    else:
        conflict = np.zeros(len(mine), dtype=bool)
        mine["conflict"] = conflict
    is_none = kind == "none"
    conflict = conflict & ~is_none
    excluded = conflict & (not keep_conflicts)

    need_lo = np.isin(kind, ("exact", "lower", "between"))
    located = np.ones(len(mine), dtype=bool)
    located[need_lo] &= _finite(mine, ("x", "y", "below_lo"))[need_lo]
    located[kind == "upper"] &= _finite(mine, ("x", "y", "below_hi"))[kind == "upper"]
    located[kind == "between"] &= _finite(mine, ("x2", "y2", "below_hi"))[kind == "between"]
    not_located = ~is_none & ~located
    used = ~is_none & located & ~excluded
    mine["used"] = used

    def rows(kinds: tuple[str, ...]) -> pd.DataFrame:
        return mine[used & np.isin(kind, kinds)]

    exact = rows(("exact",))
    lower = rows(("lower", "between"))
    upper_only = rows(("upper",))
    between = rows(("between",))
    exact_xy = exact[["x", "y"]].to_numpy(dtype=np.float64)
    exact_depth = exact["below_lo"].to_numpy(dtype=np.float64)
    lower_xy = lower[["x", "y"]].to_numpy(dtype=np.float64)
    lower_depth = lower["below_lo"].to_numpy(dtype=np.float64)
    upper_xy = np.vstack(
        [
            upper_only[["x", "y"]].to_numpy(dtype=np.float64),
            between[["x2", "y2"]].to_numpy(dtype=np.float64),
        ]
    )
    upper_depth = np.concatenate(
        [
            upper_only["below_hi"].to_numpy(dtype=np.float64),
            between["below_hi"].to_numpy(dtype=np.float64),
        ]
    )
    n_bounds = int((used & (kind != "exact")).sum())
    if not len(exact):
        raise SurfaceFitError(
            f"surface {name}: no exact pick to fit ({_plural(n_bounds, 'hole')} with bounds "
            f"only, {int(excluded.sum())} left out as conflicts, {int(not_located.sum())} "
            "outside the topography); a depth surface needs at least one exact pick"
        )
    try:
        fit = fit_surface(
            exact_xy,
            exact_depth,
            lower_xy=lower_xy if len(lower_xy) else None,
            lower=lower_depth if len(lower_xy) else None,
            upper_xy=upper_xy if len(upper_xy) else None,
            upper=upper_depth if len(upper_xy) else None,
            method=method,
            variogram=variogram,
            drift=drift,
        )
    except SurfaceFitError as error:
        raise SurfaceFitError(f"surface {name}: {error}") from error

    fitted = fit.grid(origin, (spacing, spacing), shape)
    gx, gy = np.meshgrid(
        origin[0] + spacing * np.arange(shape[0]),
        origin[1] + spacing * np.arange(shape[1]),
        indexing="ij",
    )
    topo_z = np.asarray(topography.z_at(gx, gy), dtype=np.float64)
    valid = np.isfinite(topo_z)
    if not valid.any():
        raise ValueError(f"surface {name}: the topography does not reach the rectangle")
    clamped = valid & (fitted < 0.0)
    depth = np.where(valid, np.maximum(fitted, 0.0), np.nan)
    try:
        dtm = dtm_from_grid(topo_z - depth, origin, (spacing, spacing))
    except ValueError as error:
        raise ValueError(f"surface {name}: {error} (topography against the grid)") from error

    # extrapolation: nodes farther than 2 x the median pick spacing from every pick
    pick_xy = mine.loc[used, ["x", "y"]].to_numpy(dtype=np.float64)
    pick_spacing = _pick_spacing(pick_xy)
    n_valid = int(valid.sum())
    if np.isfinite(pick_spacing):
        limit = EXTRAPOLATION_FACTOR * pick_spacing
        distance = cKDTree(pick_xy).query(np.column_stack([gx[valid], gy[valid]]))[0]
        n_far = int((distance > limit).sum())
    else:
        limit = float("nan")
        n_far = n_valid  # one pick: nothing to measure a spacing with
    share = n_far / n_valid

    # ------------------------------------------------------------------ provenance
    provenance = list(inherited)
    source = mine["source"].astype(str).to_numpy(dtype=object)
    for src in dict.fromkeys(source[used].tolist()):
        sel = used & (source == src)
        n_exact = int((sel & (kind == "exact")).sum())
        if n_exact:
            provenance.append(
                _entry(
                    "measured",
                    f"Exact picks of {name} ({src})",
                    f"picks table, source {src}",
                    name,
                    f"{_plural(n_exact, 'hole')} with the depth below the topography "
                    "honoured exactly",
                    count=n_exact,
                )
            )
        parts = [
            f"{int((sel & (kind == k)).sum())} {k}"
            for k in ("lower", "upper", "between")
            if (sel & (kind == k)).any()
        ]
        if parts:
            n_src = int((sel & (kind != "exact")).sum())
            provenance.append(
                _entry(
                    "measured",
                    f"Bounds of {name} ({src})",
                    f"picks table, source {src}",
                    name,
                    f"{_plural(n_src, 'hole')}: {', '.join(parts)}",
                    count=n_src,
                )
            )
    if not isinstance(picks, SurfacePicks):
        provenance.append(
            _entry(
                "assumed",
                f"Holes without the units of {name}",
                "policy not passed with the picks table (the picks default is absent = unknown)",
                name,
                f"{_plural(int(is_none.sum()), 'hole')} without a pick change nothing",
                count=int(is_none.sum()),
            )
        )
    n_conflicts = int(conflict.sum())
    if n_conflicts:
        holes_text = ", ".join(_holes(mine, conflict)[:10]) + (" ..." if n_conflicts > 10 else "")
        provenance.append(
            _entry(
                "assumed",
                f"Conflicting picks of {name}",
                "option keep_conflicts" if keep_conflicts else "default (conflicts left out)",
                name,
                f"{_plural(n_conflicts, 'hole')} where the sources contradict each other "
                f"{'kept in' if keep_conflicts else 'left out of'} the fit: {holes_text}",
                count=n_conflicts,
            )
        )
    if not_located.any():
        n_out = int(not_located.sum())
        holes_text = ", ".join(_holes(mine, not_located)[:10]) + (" ..." if n_out > 10 else "")
        provenance.append(
            _entry(
                "assumed",
                f"Picks of {name} outside the topography",
                "rule: a pick without a depth below the topography is left out",
                name,
                f"{_plural(n_out, 'hole')} left out: {holes_text}",
                count=n_out,
            )
        )
    if fit.method == "kriging":
        text = _variogram_text(fit.variogram)
        if variogram is None:
            provenance.append(
                _entry(
                    "assumed",
                    f"Variogram of {name}",
                    "default (spherical, nugget 0, sill = variance of the exact depths, "
                    "range = 3 x median pick spacing)",
                    name,
                    f"{text}; drift {fit.drift}",
                )
            )
        else:
            provenance.append(
                _entry(
                    "assumed",
                    f"Variogram of {name}",
                    "given by the user",
                    name,
                    f"{text}; drift {fit.drift}",
                )
            )
    else:
        provenance.append(
            _entry(
                "assumed",
                f"Interpolant of {name}",
                "option method = rbf",
                name,
                "thin-plate spline with a linear drift",
            )
        )
    if clamped.any():
        provenance.append(
            _entry(
                "assumed",
                f"{name} kept at or below the topography",
                "rule: depth >= 0 (research.md 5.7.4)",
                name,
                f"{_plural(int(clamped.sum()), 'grid node')} of {n_valid} with a fitted depth "
                f"below 0 set to 0 (lowest {float(fitted[clamped].min()):.2f} m)",
                count=int(clamped.sum()),
            )
        )
    if np.isfinite(limit):
        far_text = (
            f"{100.0 * share:.1f} % of the surface ({n_far} of {n_valid} grid nodes) is farther "
            f"than {limit:.1f} m (2 x the median pick spacing of {pick_spacing:.1f} m) from "
            "any pick"
        )
    else:
        far_text = "a single pick: the whole surface is extrapolated"
    provenance.append(
        _entry(
            "assumed",
            f"Extrapolated part of {name}",
            "interpolation beyond the picks",
            name,
            far_text,
            count=n_far,
            value=round(share, 6),
        )
    )
    if is_none.any():
        provenance.append(
            _entry(
                "missing",
                f"{name} not logged",
                "logs of the zone, or assays of an element that marks it (e.g. sulphur)",
                name,
                f"{_plural(int(is_none.sum()), 'hole')} of {len(mine)} give no pick and no bound",
                count=int(is_none.sum()),
            )
        )

    by_source = {
        src: {
            k: int((used & (source == src) & (kind == k)).sum())
            for k in ("exact", "lower", "upper", "between")
        }
        for src in dict.fromkeys(source[used].tolist())
    }
    diagnostics: dict[str, Any] = {
        "surface": name,
        "holes": len(mine),
        "used": {
            k: int((used & (kind == k)).sum()) for k in ("exact", "lower", "upper", "between")
        },
        "used_by_source": by_source,
        "holes_without_information": int(is_none.sum()),
        "conflicts": n_conflicts,
        "conflicts_kept": bool(keep_conflicts),
        "holes_conflict": _holes(mine, conflict),
        "holes_outside_topography": _holes(mine, not_located),
        "method": fit.method,
        "drift": fit.drift,
        "variogram": None if fit.variogram is None else fit.variogram.to_dict(),
        "variogram_default": bool(fit.method == "kriging" and variogram is None),
        "iterations": int(fit.iterations),
        "active_lower": int(fit.active_lower.size),
        "active_upper": int(fit.active_upper.size),
        "max_violation": float(fit.max_violation),
        "grid": {"origin": list(origin), "spacing": [spacing, spacing], "shape": list(shape)},
        "nodes": int(valid.size),
        "nodes_valid": n_valid,
        "nodes_clamped": int(clamped.sum()),
        "pick_spacing": None if not np.isfinite(pick_spacing) else pick_spacing,
        "extrapolation_distance": None if not np.isfinite(limit) else limit,
        "nodes_extrapolated": n_far,
        "extrapolated_share": share,
        "depth_min": float(np.nanmin(depth)),
        "depth_max": float(np.nanmax(depth)),
        "depth_mean": float(np.nanmean(depth)),
        "n_nested": 0,
    }
    if picks_diagnostics is not None:
        diagnostics["picks"] = picks_diagnostics
    return DepthSurface(
        name=name,
        picks=mine,
        fit=fit,
        origin=origin,
        spacing=(spacing, spacing),
        depth=depth,
        topography_z=topo_z,
        dtm=dtm,
        n_nested=0,
        provenance=provenance,
        diagnostics=diagnostics,
    )


def _variogram_text(model: VariogramModel) -> str:
    data = model.to_dict()
    parts = [f"nugget {float(data.get('nugget', 0.0)):g}"]
    for s in data.get("structures", []):
        ranges = s.get("ranges", s.get("range"))
        if isinstance(ranges, list | tuple):
            ranges = ranges[0]
        text = f"{s.get('type', s.get('model', 'structure'))} sill {float(s['sill']):.4g}"
        if ranges is not None:
            text += f" range {float(ranges):.1f} m"
        parts.append(text)
    return ", ".join(parts)


# ---------------------------------------------------------------------------- nesting


def _pushed_down(
    surface: DepthSurface, floor: np.ndarray, above: str
) -> tuple[DepthSurface, np.ndarray]:
    """``surface`` with its depth raised to ``floor`` where that is deeper; and the new depth.

    ``floor`` is the least depth allowed at each node (NaN: no limit there).
    """
    depth = surface.depth
    with np.errstate(invalid="ignore"):
        move = np.isfinite(depth) & np.isfinite(floor) & (floor > depth)
    new = np.where(move, floor, depth)
    n = int(move.sum())
    largest = float((floor - depth)[move].max()) if n else 0.0
    n_valid = int(np.isfinite(depth).sum())
    entry = _entry(
        "assumed",
        f"{_NESTING_ITEM}{surface.name}",
        f"rule: not above {above} (research.md 5.7.4); a correction after the fit, "
        "not a joint estimate",
        surface.name,
        f"{_plural(n, 'grid node')} of {n_valid} moved down to {above}"
        + (f" (largest move {largest:.2f} m); the zone between them vanishes there" if n else ""),
        count=n,
    )
    provenance = [e for e in surface.provenance if not e.item.startswith(_NESTING_ITEM)]
    provenance.append(entry)
    diagnostics = dict(surface.diagnostics)
    diagnostics.update(
        {
            "n_nested": n,
            "nested_under": above,
            "nesting_largest_move": largest,
            "depth_min": float(np.nanmin(new)),
            "depth_max": float(np.nanmax(new)),
            "depth_mean": float(np.nanmean(new)),
        }
    )
    dtm = surface.dtm
    if n:
        dtm = dtm_from_grid(surface.topography_z - new, surface.origin, surface.spacing)
    out = replace(
        surface, depth=new, dtm=dtm, n_nested=n, provenance=provenance, diagnostics=diagnostics
    )
    return out, new


def nest_surfaces(surfaces: Sequence[DepthSurface]) -> list[DepthSurface]:
    """Make the surfaces of one profile non-crossing: listed top to bottom, cumulative max.

    ``depth_1' = max(depth_1, 0)``, ``depth_k' = max(depth_k, depth_{k-1}')`` at every node.
    All surfaces must be on the same grid. The inputs are not changed; each result reports
    its number of moved nodes in ``n_nested`` and in an assumed provenance entry.
    """
    surfaces = list(surfaces)
    if not surfaces:
        return []
    first = surfaces[0]
    for s in surfaces[1:]:
        same = (
            s.shape == first.shape
            and np.allclose(s.origin, first.origin, rtol=0.0, atol=1e-9)
            and np.allclose(s.spacing, first.spacing, rtol=0.0, atol=1e-12)
        )
        if not same:
            raise ValueError(
                f"surfaces {first.name} and {s.name} are on different grids; nesting needs "
                "the same rectangle and spacing"
            )
    out: list[DepthSurface] = []
    floor = np.zeros(first.depth.shape)
    above = "the topography"
    for s in surfaces:
        nested, floor = _pushed_down(s, floor, above)
        out.append(nested)
        above = s.name
    return out


def nest_under(surface: DepthSurface, above: DTM, *, above_name: str = "") -> DepthSurface:
    """Push ``surface`` down wherever it is above the DTM ``above`` (an existing base).

    For a surface built on its own under one that is already on file (the oxide base under
    the cover base). Nodes outside ``above`` are not changed.
    """
    gx, gy = surface.nodes()
    floor = surface.topography_z - np.asarray(above.z_at(gx, gy), dtype=np.float64)
    floor = np.where(np.isfinite(floor), np.maximum(floor, 0.0), np.nan)
    return _pushed_down(surface, floor, above_name or "the surface above")[0]
