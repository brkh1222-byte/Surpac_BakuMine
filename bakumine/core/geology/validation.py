"""Hole-out cross-validation of the geology model (research.md 5.7, P9.T11).

On a real deposit there is no truth to compare the model with. What can be measured is how
well the model predicts a hole it has not seen: leave holes out, fit the model from the
others, and compare the prediction along the left-out holes with their logs.

Holes are left out whole, never single intervals: an interval left out on its own is
predicted by its neighbours in the same hole, a few metres away, which says nothing about
the model between the holes.

What is reported (:class:`CVResult`):

- the agreement by length: metres predicted as logged over metres logged;
- the confusion matrix in logged metres (rows: logged, columns: predicted);
- recall and precision per unit;
- the contact position error: the distance along the hole between a logged contact and the
  predicted contact of the same pair of units (above, below), matched one to one, nearest
  first;
- all of these by the distance from the point to the nearest logged point of a remaining
  hole, which shows how fast the model degrades away from data.

What it proves and what it does not. The figures measure the prediction *between the holes,
at about the drill spacing*: every test point lies on a drilled hole, so inside the drilled
volume, at most about one hole spacing from data. They say nothing about the model beyond
the drilling (deeper than the holes, outside their outline), and nothing about errors that
all the logs share (a unit logged wrongly everywhere is predicted "correctly"). With K folds
the model of each fold has fewer holes than the final model, so the figures lean to the
pessimistic side for points between the holes.

The model is built by a callable ``builder(table, labels, traces)`` that gets the intervals
of the remaining holes only and returns an object with ``labels(points)`` (integer codes
into its ``categories`` / ``units``, or names) or ``block_of(points)`` (a
:class:`~bakumine.core.geology.faults.FaultNetwork`). :func:`largest_field_builder` and
:func:`stack_builder` give the rock-unit builders, :func:`fault_network_builder` the fault
blocks; any other builder works the same way.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.geology.lithology import (
    EventSpec,
    FieldSettings,
    fit_stack,
    logged_hole_points,
)
from bakumine.core.geology.provenance import ProvenanceEntry

NONE_LABEL = "(none)"
"""Column of the confusion matrix for metres where the model gives no label."""

DEFAULT_FOLDS = 10
_TASK = "P9.T11"
_STAGES = {"lith": "lithology", "fault_block": "faults"}
_WHAT = {"lith": "rock units", "fault_block": "fault blocks"}
_TOL = 1e-6

ModelBuilder = Callable[[pd.DataFrame, pd.Series, HoleTraces], Any]

SCOPE_TEXT = (
    "What this measures: holes are left out whole, the model is fitted again from the "
    "remaining holes and compared with the logs of the left-out holes. The figures "
    "describe the prediction between the holes, at about the drill spacing, inside the "
    "drilled volume.\n"
    "What this does not prove: nothing is tested beyond the drilling (below the holes, "
    "outside their outline) or farther from data than the largest distance listed; an "
    "error shared by all logs is not seen; the figures compare the model with the logs, "
    "not with the ground."
)


# --------------------------------------------------------------------------
# builders
# --------------------------------------------------------------------------


class _Constant:
    """The model of a training set with one unit only: that unit everywhere."""

    def __init__(self, unit: str) -> None:
        self.categories = (unit,)

    def labels(self, points) -> np.ndarray:
        return np.zeros(len(np.asarray(points)), dtype=np.int16)


def stack_builder(events: Sequence[EventSpec]) -> ModelBuilder:
    """Builder of a rock-unit model from events (``fit_stack``), for :func:`cross_validate`.

    A unit with no interval in the remaining holes cannot be fitted: it is taken out of
    its ``argmax`` group for that fold (an ``intrusion`` event of such a unit is dropped),
    so its metres in the left-out holes count as wrong, which is what a unit known from
    the left-out holes only deserves.
    """
    events = tuple(events)
    if not events:
        raise ValueError("give at least one event")

    def build(table: pd.DataFrame, labels: pd.Series, traces: HoleTraces):
        present = {str(u) for u in pd.Series(labels).dropna().unique()}
        kept: list[EventSpec] = []
        for spec in events:
            if spec.mode == "sequence":
                kept.append(spec)
                continue
            units = tuple(u for u in spec.units if u in present)
            if units:
                kept.append(
                    spec if units == spec.units else EventSpec(spec.mode, units, spec.settings)
                )
        if not kept:
            raise ValueError("no unit of the model is logged in the remaining holes")
        if len(kept) == 1 and kept[0].mode == "argmax" and len(kept[0].units) == 1:
            return _Constant(kept[0].units[0])
        return fit_stack(table, labels, traces, kept)

    build.description = "; ".join(f"{e.mode} of {', '.join(e.units)}" for e in events)
    return build


def largest_field_builder(
    group: Sequence[str], settings: FieldSettings | None = None
) -> ModelBuilder:
    """The default rock-unit builder: one field per unit of ``group``, the largest wins."""
    settings = FieldSettings() if settings is None else settings
    return stack_builder([EventSpec("argmax", tuple(str(u) for u in group), settings)])


def fault_network_builder(
    surfaces: Sequence[Mapping[str, Any]],
    *,
    fault_zone: pd.DataFrame | None = None,
    min_picks: int = 3,
) -> ModelBuilder:
    """Builder of the fault blocks: every fault inferred from the remaining holes.

    ``surfaces``: one mapping per fault with ``name``, ``between`` (the two blocks) and
    optionally ``prior`` = ``(dip, dip_azimuth)``. ``fault_zone`` holds the logged
    fault-zone intervals of all holes; only those of holes that have an interval in the
    fold's table are used, so a left-out hole gives the fit nothing.
    """
    surfaces = [dict(s) for s in surfaces]
    if not surfaces:
        raise ValueError("give at least one fault")

    def build(table: pd.DataFrame, blocks: pd.Series, traces: HoleTraces):
        from bakumine.core.geology.faults import (
            FaultNetwork,
            fault_picks,
            infer_fault,
            labelled_points,
        )

        zone = fault_zone
        if zone is not None:
            zone = zone[zone["hole_id"].isin(set(table["hole_id"]))]
        xyz, block, hole = labelled_points(table, blocks, traces)
        picks = fault_picks(table, blocks, traces, fault_zone=zone)
        faults = []
        for surface in surfaces:
            a, b = (str(v) for v in surface["between"])
            in_a, in_b = block == a, block == b
            prior = surface.get("prior")
            faults.append(
                infer_fault(
                    str(surface["name"]),
                    picks,
                    xyz[in_a],
                    xyz[in_b],
                    blocks=(a, b),
                    prior=None if prior is None else (float(prior[0]), float(prior[1])),
                    min_picks=min_picks,
                    holes_a=hole[in_a],
                    holes_b=hole[in_b],
                )
            )
        return FaultNetwork.fit(faults, xyz, block)

    build.description = "faults inferred from the logs: " + ", ".join(
        str(s["name"]) for s in surfaces
    )
    return build


def predicted_names(model: Any, points: np.ndarray) -> np.ndarray:
    """Label names (object array, None where there is none) a model gives at ``points``."""
    if hasattr(model, "block_of"):
        raw = np.asarray(model.block_of(points), dtype=object)
    elif hasattr(model, "labels"):
        raw = np.asarray(model.labels(points))
    else:
        raise TypeError("the builder must return an object with labels(points) or block_of(points)")
    if raw.shape != (len(points),):
        raise ValueError(f"the model gave labels of shape {raw.shape} for {len(points)} points")
    if raw.dtype.kind in "iu":
        names = getattr(model, "categories", None)
        if names is None or callable(names):
            names = getattr(model, "units", None)
        if names is None or callable(names):
            raise TypeError(
                "a model with integer labels needs 'categories' or 'units' (a sequence of names)"
            )
        table = np.array([*(str(n) for n in names), None], dtype=object)
        codes = raw.astype(np.int64)
        if codes.size and (codes.min() < -1 or codes.max() >= len(table) - 1):
            raise ValueError("the model gave a label code outside its categories")
        return table[codes]
    out = np.empty(len(raw), dtype=object)
    for i, value in enumerate(raw):
        out[i] = (
            None if value is None or (isinstance(value, float) and value != value) else str(value)
        )
    return out


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------


def hole_folds(
    hole_ids: Sequence[object], folds: int | None = DEFAULT_FOLDS, seed: int = 0
) -> list[tuple[object, ...]]:
    """Split the holes into folds: every hole in exactly one fold.

    ``folds`` None, 0 or >= the number of holes: leave-one-out, one hole per fold in sorted
    order (no random numbers). Otherwise the sorted holes are shuffled with
    ``numpy.random.default_rng(seed)`` and cut into ``folds`` parts whose sizes differ by
    at most one. The result depends on the set of holes and the seed only, not on their
    order.
    """
    holes = sorted({h for h in hole_ids if not pd.isna(h)}, key=lambda h: (str(h), repr(h)))
    if len(holes) < 2:
        raise ValueError("cross-validation needs at least two holes")
    if folds is not None and (isinstance(folds, bool) or int(folds) != folds or folds < 0):
        raise ValueError("folds must be a whole number >= 2 (0 = leave-one-out)")
    if folds == 1:
        raise ValueError("one fold would leave no hole to fit from; give >= 2, or 0")
    if not folds or folds >= len(holes):
        return [(h,) for h in holes]
    order = np.random.default_rng(seed).permutation(len(holes))
    return [tuple(holes[i] for i in sorted(part.tolist())) for part in np.array_split(order, folds)]


# --------------------------------------------------------------------------
# intervals, samples and contacts along the holes
# --------------------------------------------------------------------------


def _scored_intervals(table: pd.DataFrame, labels: Any, scored: Sequence[str] | None):
    for name in ("hole_id", "depth_from", "depth_to"):
        if name not in table.columns:
            raise ValueError(f"interval table has no column {name!r}")
    if isinstance(labels, pd.Series):
        if not labels.index.equals(table.index):
            raise ValueError("labels must have the index of the table")
        values = labels.to_numpy(dtype=object)
    else:
        values = np.asarray(labels, dtype=object)
        if values.shape != (len(table),):
            raise ValueError("labels must have one value per row of the table")
    frame = pd.DataFrame(
        {
            "row": np.arange(len(table)),
            "hole_id": table["hole_id"].to_numpy(dtype=object),
            "depth_from": pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(float),
            "depth_to": pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(float),
            "label": values,
        }
    )
    keep = (
        frame["label"].notna()
        & frame["hole_id"].notna()
        & np.isfinite(frame["depth_from"])
        & np.isfinite(frame["depth_to"])
        & (frame["depth_to"] > frame["depth_from"])
    )
    frame = frame[keep].copy()
    frame["label"] = frame["label"].astype(str)
    if scored is None:
        categories = tuple(sorted(frame["label"].unique()))
    else:
        categories = tuple(str(u) for u in scored)
        if len(set(categories)) != len(categories):
            raise ValueError("a scored label is listed twice")
        frame = frame[frame["label"].isin(categories)]
    frame = frame.sort_values(["hole_id", "depth_from", "depth_to"], kind="stable")
    frame = frame.reset_index(drop=True)
    # a segment: intervals of one hole that abut (a gap or an unknown interval ends it)
    hole = frame["hole_id"].to_numpy(dtype=object)
    a = frame["depth_from"].to_numpy()
    b = frame["depth_to"].to_numpy()
    new = np.ones(len(frame), dtype=bool)
    if len(frame) > 1:
        new[1:] = (hole[1:] != hole[:-1]) | (np.abs(a[1:] - b[:-1]) > _TOL)
        if ((hole[1:] == hole[:-1]) & (a[1:] < b[:-1] - _TOL)).any():
            raise ValueError("the labelled intervals of a hole overlap; they must not")
    frame["segment"] = np.cumsum(new) - 1
    return frame, categories


def _samples(intervals: pd.DataFrame, traces: HoleTraces, step: float) -> pd.DataFrame:
    """Each interval cut into equal pieces no longer than ``step``; one row per piece, at
    its middle, with its length. The lengths of an interval's pieces sum to its length."""
    a = intervals["depth_from"].to_numpy()
    b = intervals["depth_to"].to_numpy()
    n = np.maximum(1, np.ceil((b - a) / step - 1e-9).astype(np.int64))
    rep = np.repeat(np.arange(len(intervals)), n)
    start = np.concatenate([[0], np.cumsum(n)[:-1]]) if len(n) else np.zeros(0, dtype=np.int64)
    position = np.arange(len(rep)) - start[rep]
    length = (b[rep] - a[rep]) / n[rep]
    depth = a[rep] + (position + 0.5) * length
    hole = intervals["hole_id"].to_numpy(dtype=object)[rep]
    xyz = traces.xyz_at(hole, depth, depth_tolerance=_TOL)[0] if len(rep) else np.zeros((0, 3))
    return pd.DataFrame(
        {
            "hole_id": hole,
            "segment": intervals["segment"].to_numpy()[rep],
            "depth": depth,
            "length": length,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "logged": intervals["label"].to_numpy(dtype=object)[rep],
        }
    )


def _logged_contacts(intervals: pd.DataFrame, traces: HoleTraces) -> pd.DataFrame:
    seg = intervals["segment"].to_numpy()
    label = intervals["label"].to_numpy(dtype=object)
    idx = np.flatnonzero((seg[1:] == seg[:-1]) & (label[1:] != label[:-1])) + 1
    hole = intervals["hole_id"].to_numpy(dtype=object)[idx]
    depth = intervals["depth_from"].to_numpy()[idx]
    xyz = traces.xyz_at(hole, depth, depth_tolerance=_TOL)[0] if len(idx) else np.zeros((0, 3))
    out = pd.DataFrame(
        {
            "hole_id": hole,
            "segment": seg[idx],
            "depth": depth,
            "above": label[idx - 1],
            "below": label[idx],
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
        }
    )
    return out[np.isfinite(xyz).all(axis=1)].reset_index(drop=True)


def _predicted_contacts(
    model: Any, samples: pd.DataFrame, traces: HoleTraces, tolerance: float
) -> pd.DataFrame:
    """Changes of the predicted label between neighbouring pieces of a segment, each
    located by bisection along the hole to ``tolerance``."""
    seg = samples["segment"].to_numpy()
    pred = samples["predicted"].to_numpy(dtype=object)
    depth = samples["depth"].to_numpy()
    known = ~pd.isna(pred)
    change = (seg[1:] == seg[:-1]) & (pred[1:] != pred[:-1]) & known[1:] & known[:-1]
    idx = np.flatnonzero(change)
    hole = samples["hole_id"].to_numpy(dtype=object)[idx]
    lo, hi = depth[idx].copy(), depth[idx + 1].copy()
    above, below = pred[idx], pred[idx + 1]
    if len(idx):
        steps = max(0, math.ceil(math.log2(max(float((hi - lo).max()) / tolerance, 1.0))))
        for _ in range(steps):
            mid = 0.5 * (lo + hi)
            xyz = traces.xyz_at(hole, mid, depth_tolerance=_TOL)[0]
            same = predicted_names(model, xyz) == above
            lo = np.where(same, mid, lo)
            hi = np.where(same, hi, mid)
    return pd.DataFrame(
        {
            "hole_id": hole,
            "segment": seg[idx],
            "depth": 0.5 * (lo + hi),
            "above": above,
            "below": below,
        }
    )


def match_contacts(
    logged: pd.DataFrame, predicted: pd.DataFrame, max_distance: float | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Match logged and predicted contacts of the same hole and the same pair of units
    (above, below), one to one, the closest pairs first.

    Both tables have ``hole_id, depth, above, below``. Returns the predicted depth per
    logged contact (NaN when unmatched) and a boolean per predicted contact (matched).
    """
    out = np.full(len(logged), np.nan)
    used = np.zeros(len(predicted), dtype=bool)
    if len(logged) == 0 or len(predicted) == 0:
        return out, used
    keys = ["hole_id", "above", "below"]
    groups = predicted.reset_index(drop=True).groupby(keys, sort=False).indices
    for key, rows in logged.reset_index(drop=True).groupby(keys, sort=False).indices.items():
        cand = groups.get(key)
        if cand is None:
            continue
        d_log = logged["depth"].to_numpy()[rows]
        d_pred = predicted["depth"].to_numpy()[cand]
        gap = np.abs(d_log[:, None] - d_pred[None, :])
        free_l = np.ones(len(rows), dtype=bool)
        free_p = np.ones(len(cand), dtype=bool)
        for flat in np.argsort(gap, axis=None, kind="stable"):
            i, j = divmod(int(flat), len(cand))
            if max_distance is not None and gap[i, j] > max_distance:
                break
            if free_l[i] and free_p[j]:
                free_l[i] = free_p[j] = False
                out[rows[i]] = d_pred[j]
                used[cand[j]] = True
    return out, used


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def confusion_matrix(samples: pd.DataFrame, categories: Sequence[str]) -> pd.DataFrame:
    """Logged metres by logged (rows) and predicted (columns) label.

    Rows: ``categories``. Columns: ``categories``, then predicted labels outside them,
    then :data:`NONE_LABEL` (no prediction). A row sums to the logged metres of its unit.
    """
    categories = [str(c) for c in categories]
    predicted = samples["predicted"].to_numpy(dtype=object)
    names = [NONE_LABEL if pd.isna(p) else str(p) for p in predicted]
    extra = sorted(set(names) - set(categories) - {NONE_LABEL})
    columns = [*categories, *extra, NONE_LABEL]
    row = {c: i for i, c in enumerate(categories)}
    col = {c: i for i, c in enumerate(columns)}
    matrix = np.zeros((len(categories), len(columns)))
    np.add.at(
        matrix,
        (
            np.array([row[v] for v in samples["logged"]], dtype=np.int64),
            np.array([col[v] for v in names], dtype=np.int64),
        ),
        samples["length"].to_numpy(dtype=np.float64),
    )
    out = pd.DataFrame(matrix, index=categories, columns=columns)
    out.index.name = "logged"
    out.columns.name = "predicted"
    return out


def unit_scores(confusion: pd.DataFrame) -> pd.DataFrame:
    """Per unit: ``logged_m``, ``predicted_m``, ``correct_m``, ``recall`` (correct over
    logged) and ``precision`` (correct over predicted); NaN where the divisor is zero."""
    rows = []
    for unit in confusion.index:
        logged = float(confusion.loc[unit].sum())
        predicted = float(confusion[unit].sum())
        correct = float(confusion.loc[unit, unit])
        rows.append(
            {
                "unit": unit,
                "logged_m": logged,
                "predicted_m": predicted,
                "correct_m": correct,
                "recall": correct / logged if logged > 0 else np.nan,
                "precision": correct / predicted if predicted > 0 else np.nan,
            }
        )
    return pd.DataFrame(
        rows, columns=["unit", "logged_m", "predicted_m", "correct_m", "recall", "precision"]
    )


def _agreement(samples: pd.DataFrame) -> tuple[float, float]:
    length = samples["length"].to_numpy(dtype=np.float64)
    same = samples["logged"].to_numpy(dtype=object) == samples["predicted"].to_numpy(dtype=object)
    total = float(length.sum())
    return (float(length[same].sum()) / total if total > 0 else np.nan), total


def contact_scores(contacts: pd.DataFrame, n_spurious: int = 0) -> dict[str, float | int]:
    """Counts and error figures of the contacts: ``error`` is predicted minus logged depth."""
    error = contacts["error"].to_numpy(dtype=np.float64) if len(contacts) else np.zeros(0)
    error = error[np.isfinite(error)]
    out: dict[str, float | int] = {
        "logged": len(contacts),
        "matched": len(error),
        "missed": len(contacts) - len(error),
        "spurious": int(n_spurious),
    }
    absolute = np.abs(error)
    for name, value in (
        ("mean_abs", absolute.mean() if len(error) else np.nan),
        ("median_abs", np.median(absolute) if len(error) else np.nan),
        ("rms", np.sqrt(np.mean(error**2)) if len(error) else np.nan),
        ("p90_abs", np.quantile(absolute, 0.9) if len(error) else np.nan),
        ("max_abs", absolute.max() if len(error) else np.nan),
        ("bias", error.mean() if len(error) else np.nan),
    ):
        out[name] = float(value)
    return out


def _bin_edges(distance: np.ndarray, weight: np.ndarray, bins: int | Sequence[float]) -> np.ndarray:
    """Inner edges of the distance classes: class i is ``edge[i-1] <= d < edge[i]``."""
    if isinstance(bins, (int, np.integer)) and not isinstance(bins, bool):
        if bins < 1:
            raise ValueError("distance_bins must be >= 1")
        good = np.isfinite(distance)
        if bins == 1 or not good.any():
            return np.zeros(0)
        order = np.argsort(distance[good], kind="stable")
        d, w = distance[good][order], weight[good][order]
        share = np.cumsum(w) / w.sum()
        edges = d[np.minimum(np.searchsorted(share, np.arange(1, bins) / bins), len(d) - 1)]
        return np.unique(edges[edges > d[0]])
    edges = np.asarray(list(bins), dtype=np.float64)
    if edges.ndim != 1 or not np.isfinite(edges).all() or (edges <= 0).any():
        raise ValueError("distance_bins must be a count or positive distances")
    if (np.diff(edges) <= 0).any():
        raise ValueError("distance_bins must increase")
    return edges


def by_distance(samples: pd.DataFrame, contacts: pd.DataFrame, edges: np.ndarray) -> pd.DataFrame:
    """The agreement and the contact error per class of distance to the nearest logged
    point of a remaining hole. Columns: ``distance_from, distance_to, metres, agreement,
    contacts, matched, mean_abs_error, median_abs_error``."""
    bounds = np.concatenate([[0.0], np.asarray(edges, dtype=np.float64), [np.inf]])
    s_bin = np.searchsorted(edges, samples["distance"].to_numpy(dtype=np.float64), side="right")
    c_bin = np.searchsorted(edges, contacts["distance"].to_numpy(dtype=np.float64), side="right")
    rows = []
    for i in range(len(bounds) - 1):
        agreement, metres = _agreement(samples[s_bin == i])
        scores = contact_scores(contacts[c_bin == i])
        rows.append(
            {
                "distance_from": bounds[i],
                "distance_to": bounds[i + 1],
                "metres": metres,
                "agreement": agreement,
                "contacts": scores["logged"],
                "matched": scores["matched"],
                "mean_abs_error": scores["mean_abs"],
                "median_abs_error": scores["median_abs"],
            }
        )
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# the result
# --------------------------------------------------------------------------


@dataclass
class CVResult:
    """What a hole-out cross-validation measured. Lengths in metres, agreement 0 to 1.

    ``samples``: one row per scored piece of a left-out hole: ``hole_id, fold, depth``
    (its middle), ``length, x, y, z, logged, predicted`` (None = no prediction),
    ``distance`` (to the nearest logged point of a remaining hole).
    ``contacts``: one row per logged contact: ``hole_id, fold, depth, above, below, x, y,
    z, predicted_depth, error`` (predicted minus logged depth; NaN when no predicted
    contact of the pair was found), ``distance``.
    ``spurious``: predicted contacts that match no logged one.
    """

    layer: str
    categories: tuple[str, ...]
    folds: list[tuple[object, ...]]
    leave_one_out: bool
    seed: int
    step: float
    samples: pd.DataFrame
    contacts: pd.DataFrame
    spurious: pd.DataFrame
    confusion: pd.DataFrame
    per_unit: pd.DataFrame
    distance_table: pd.DataFrame
    by_hole: pd.DataFrame
    by_fold: pd.DataFrame
    agreement: float
    metres: float
    contact: dict[str, float | int]
    seconds: float
    builder: str = ""
    skipped_holes: tuple[object, ...] = ()
    metres_without_trace: float = 0.0
    notes: list[str] = field(default_factory=list)

    @property
    def holes(self) -> tuple[object, ...]:
        return tuple(h for fold in self.folds for h in fold)

    @property
    def scheme(self) -> str:
        n = len(self.holes)
        if self.leave_one_out:
            return f"leave-one-out by hole, {n} holes"
        return f"{len(self.folds)} folds by hole, {n} holes, seed {self.seed}"

    @property
    def max_distance(self) -> float:
        d = self.samples["distance"].to_numpy(dtype=np.float64)
        return float(np.nanmax(d)) if len(d) and np.isfinite(d).any() else float("nan")

    @property
    def provenance(self) -> list[ProvenanceEntry]:
        """The quality figures as ``measured`` entries, and the limit of the method as an
        ``assumed`` one (geology_contracts.md section 3)."""
        what = _WHAT.get(self.layer, self.layer)
        stage = _STAGES.get(self.layer, "")
        source = f"hole-out cross-validation ({self.scheme}; the model fitted again without them)"
        used_for = f"quality of the {what} between the holes"
        entries = [
            ProvenanceEntry(
                item=f"Hole-out agreement, {what}",
                kind="measured",
                source=source,
                used_for=used_for,
                detail=(
                    f"{self.agreement:.3f} of {self.metres:.0f} m of left-out logs predicted as "
                    f"logged; test points up to {self.max_distance:.0f} m from a remaining hole"
                ),
                stage=stage,
                task=_TASK,
                count=len(self.holes),
                value=round(float(self.agreement), 4),
            )
        ]
        c = self.contact
        if c["logged"]:
            if c["matched"]:
                detail = (
                    f"{c['matched']} of {c['logged']} logged contacts found again; distance "
                    f"along the hole: median {c['median_abs']:.1f} m, mean {c['mean_abs']:.1f} m, "
                    f"90 % within {c['p90_abs']:.1f} m; {c['spurious']} predicted contacts "
                    "without a logged one"
                )
            else:
                detail = f"none of the {c['logged']} logged contacts was found again"
            entries.append(
                ProvenanceEntry(
                    item=f"Hole-out contact position error, {what}",
                    kind="measured",
                    source=source,
                    used_for=f"position error of the {what} contacts between the holes",
                    detail=detail,
                    stage=stage,
                    task=_TASK,
                    count=int(c["matched"]),
                    **({"value": round(float(c["median_abs"]), 3)} if c["matched"] else {}),
                )
            )
        if len(self.distance_table) > 1:
            parts = [
                f"{_range_text(r.distance_from, r.distance_to)}: {r.agreement:.3f} "
                f"({r.metres:.0f} m)"
                for r in self.distance_table.itertuples(index=False)
                if r.metres > 0
            ]
            entries.append(
                ProvenanceEntry(
                    item=f"Hole-out agreement by distance to data, {what}",
                    kind="measured",
                    source=source,
                    used_for=f"how fast the {what} degrade away from the holes",
                    detail="; ".join(parts),
                    stage=stage,
                    task=_TASK,
                )
            )
        entries.append(
            ProvenanceEntry(
                item=f"Quality of the {what} beyond the drilling",
                kind="assumed",
                source="not measured: every cross-validation test point lies on a drilled hole",
                used_for=f"{what} below the holes, outside their outline, or farther than "
                f"{self.max_distance:.0f} m from a hole",
                detail="the hole-out figures hold between the holes at the drill spacing only",
                stage=stage,
                task=_TASK,
            )
        )
        return entries

    def to_dict(self) -> dict[str, Any]:
        """The figures as JSON-ready data (no per-sample rows)."""
        return {
            "layer": self.layer,
            "scheme": self.scheme,
            "folds": len(self.folds),
            "leave_one_out": self.leave_one_out,
            "seed": self.seed,
            "step": self.step,
            "holes": len(self.holes),
            "builder": self.builder,
            "metres": self.metres,
            "agreement": _json(self.agreement),
            "max_distance": _json(self.max_distance),
            "confusion": {
                "logged": list(self.confusion.index),
                "predicted": list(self.confusion.columns),
                "metres": self.confusion.to_numpy().tolist(),
            },
            "per_unit": _records(self.per_unit),
            "contacts": {k: _json(v) for k, v in self.contact.items()},
            "by_distance": _records(self.distance_table),
            "by_fold": _records(self.by_fold),
            "skipped_holes": len(self.skipped_holes),
            "metres_without_trace": self.metres_without_trace,
            "seconds": self.seconds,
            "notes": list(self.notes),
            "scope": SCOPE_TEXT,
            "provenance": [entry.to_dict() for entry in self.provenance],
        }

    def format_text(self) -> str:
        """The report for the terminal: the figures, then what they do and do not prove."""
        what = _WHAT.get(self.layer, self.layer)
        lines = [
            f"Hole-out cross-validation of the {what}: {self.scheme}",
            f"  model: {self.builder}" if self.builder else "",
            f"  scored: {self.metres:.2f} m of logs in pieces of at most {self.step:g} m; "
            f"test points up to {self.max_distance:.1f} m from a remaining hole",
            f"  agreement by length: {self.agreement:.4f}",
            "",
            "Confusion matrix, metres (rows: logged, columns: predicted)",
            _table(
                ["logged", *self.confusion.columns],
                [[u, *(f"{v:.1f}" for v in self.confusion.loc[u])] for u in self.confusion.index],
            ),
            "",
            "Per unit",
            _table(
                ["unit", "logged m", "predicted m", "recall", "precision"],
                [
                    [
                        r.unit,
                        f"{r.logged_m:.1f}",
                        f"{r.predicted_m:.1f}",
                        _num(r.recall, 4),
                        _num(r.precision, 4),
                    ]
                    for r in self.per_unit.itertuples(index=False)
                ],
            ),  # fmt: skip
            "",
        ]
        c = self.contact
        lines.append(
            f"Contacts: {c['logged']} logged, {c['matched']} found again (same pair of units, "
            f"same hole), {c['missed']} missed, {c['spurious']} predicted without a logged one"
        )
        if c["matched"]:
            lines.append(
                f"  distance along the hole between logged and predicted contact: median "
                f"{c['median_abs']:.2f} m, mean {c['mean_abs']:.2f} m, RMS {c['rms']:.2f} m, "
                f"90 % within {c['p90_abs']:.2f} m, largest {c['max_abs']:.2f} m; mean "
                f"(predicted - logged depth) {c['bias']:+.2f} m"
            )
        lines += [
            "",
            "By distance to the nearest logged point of a remaining hole",
            _table(
                [
                    "distance m",
                    "metres",
                    "agreement",
                    "contacts",
                    "found",
                    "mean error m",
                    "median error m",
                ],
                [
                    [
                        _range_text(r.distance_from, r.distance_to),
                        f"{r.metres:.1f}",
                        _num(r.agreement, 4),
                        str(r.contacts),
                        str(r.matched),
                        _num(r.mean_abs_error, 2),
                        _num(r.median_abs_error, 2),
                    ]
                    for r in self.distance_table.itertuples(index=False)
                ],
            ),  # fmt: skip
        ]
        if self.skipped_holes:
            lines.append(
                f"\n{len(self.skipped_holes)} hole(s) with labels but no trace were left out "
                "of the test and of the fits."
            )
        if self.metres_without_trace > 0:
            lines.append(
                f"{self.metres_without_trace:.2f} m of logs lie beyond the hole traces and "
                "are not scored."
            )
        lines += [f"note: {note}" for note in self.notes]
        lines += ["", SCOPE_TEXT, f"\n{len(self.folds)} fits in {self.seconds:.1f} s"]
        return "\n".join(line for line in lines if line is not None)


def _json(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        return float(value) if math.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [{k: _json(v) for k, v in row.items()} for row in frame.to_dict("records")]


def _num(value: float, digits: int) -> str:
    return "-" if value is None or not math.isfinite(value) else f"{value:.{digits}f}"


def _range_text(lo: float, hi: float) -> str:
    return f">= {lo:.1f}" if not math.isfinite(hi) else f"{lo:.1f} - {hi:.1f}"


def _table(headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    cells = [[str(h) for h in headers], *[[str(c) for c in row] for row in rows]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(headers))]
    out = []
    for n, row in enumerate(cells):
        parts = [
            row[0].ljust(widths[0]),
            *(c.rjust(w) for c, w in zip(row[1:], widths[1:], strict=True)),
        ]
        out.append("  " + "  ".join(parts).rstrip())
        if n == 0:
            out.append("  " + "  ".join("-" * w for w in widths))
    return "\n".join(out)


# --------------------------------------------------------------------------
# the cross-validation
# --------------------------------------------------------------------------


def cross_validate(
    table: pd.DataFrame,
    labels: pd.Series,
    traces: HoleTraces,
    builder: ModelBuilder,
    *,
    layer: str = "lith",
    scored: Sequence[str] | None = None,
    folds: int | None = DEFAULT_FOLDS,
    seed: int = 0,
    step: float = 1.0,
    contact_tolerance: float = 0.01,
    match_distance: float | None = None,
    distance_bins: int | Sequence[float] = 4,
    progress: Callable[[str, float], None] | None = None,
) -> CVResult:
    """Leave holes out, fit the model from the others, score it along the left-out holes.

    ``table`` has ``hole_id, depth_from, depth_to``; ``labels`` the label of each row (same
    index; NaN = unknown). ``builder(table, labels, traces)`` gets the rows of the
    remaining holes (all their rows, with their labels as given) and returns the model.

    ``scored``: the labels that are tested (default: every label present). Intervals with
    another label are not scored and end a run of contacts, but the builder still sees them.
    ``folds``: see :func:`hole_folds` (0 or None = leave-one-out). ``step``: the longest
    piece an interval is cut into; the prediction is taken at the middle of each piece.
    ``contact_tolerance``: predicted contacts are located to this length along the hole.
    ``match_distance``: the largest distance at which a logged and a predicted contact are
    matched (default: any distance within the hole). ``distance_bins``: a number of
    distance classes with equal logged metres, or the class limits in metres.

    Holes that have labels but no trace are left out of everything and listed in
    ``skipped_holes``. Raises ``ValueError`` when a fold's model cannot be built.
    """
    if not (np.isfinite(step) and step > 0.0):
        raise ValueError("step must be > 0")
    if not (np.isfinite(contact_tolerance) and contact_tolerance > 0.0):
        raise ValueError("contact_tolerance must be > 0")
    t_start = time.perf_counter()
    intervals, categories = _scored_intervals(table, labels, scored)
    if intervals.empty:
        raise ValueError("no interval carries a scored label")
    traced = np.array([h in traces for h in intervals["hole_id"]], dtype=bool)
    skipped = tuple(sorted(set(intervals["hole_id"][~traced]), key=str))
    intervals = intervals[traced].reset_index(drop=True)
    parts = hole_folds(intervals["hole_id"].unique(), folds, seed)
    loo = all(len(p) == 1 for p in parts)

    table_hole = table["hole_id"].to_numpy(dtype=object)
    label_series = labels if isinstance(labels, pd.Series) else pd.Series(labels, index=table.index)
    known = np.zeros(len(table), dtype=bool)
    known[intervals["row"].to_numpy()] = True

    all_samples, all_contacts, all_spurious, fold_rows = [], [], [], []
    lost = 0.0
    for number, held in enumerate(parts):
        t0 = time.perf_counter()
        if progress is not None:
            progress(f"fold {number + 1} of {len(parts)}", number / len(parts))
        out = pd.Series(table_hole).isin(list(held)).to_numpy()
        train = table[~out]
        try:
            model = builder(train, label_series[~out], traces)
        except ValueError as exc:
            raise ValueError(
                f"fold {number + 1} of {len(parts)} ({len(held)} hole(s) left out): the model "
                f"could not be built from the remaining holes: {exc}"
            ) from exc
        data = logged_hole_points(table, traces, step=step, known=known & ~out)
        if len(data) == 0:
            raise ValueError(f"fold {number + 1}: the remaining holes have no logged point")
        tree = cKDTree(data)

        test = intervals[intervals["hole_id"].isin(list(held))]
        samples = _samples(test, traces, step)
        inside = np.isfinite(samples[["x", "y", "z"]].to_numpy()).all(axis=1)
        lost += float(samples["length"].to_numpy()[~inside].sum())
        samples = samples[inside].reset_index(drop=True)
        xyz = samples[["x", "y", "z"]].to_numpy()
        names = predicted_names(model, xyz) if len(xyz) else np.empty(0, dtype=object)
        # object dtype on purpose: a string dtype would turn None (no prediction) into NaN
        samples["predicted"] = pd.Series(names, index=samples.index, dtype=object)
        samples["distance"] = tree.query(xyz)[0] if len(xyz) else np.zeros(0)
        samples["fold"] = number

        logged = _logged_contacts(test, traces)
        predicted = _predicted_contacts(model, samples, traces, contact_tolerance)
        depth, used = match_contacts(logged, predicted, match_distance)
        logged["predicted_depth"] = depth
        logged["error"] = depth - logged["depth"].to_numpy()
        logged["distance"] = (
            tree.query(logged[["x", "y", "z"]].to_numpy())[0] if len(logged) else np.zeros(0)
        )
        logged["fold"] = number
        spurious = predicted[~used].assign(fold=number)

        all_samples.append(samples)
        all_contacts.append(logged)
        all_spurious.append(spurious)
        agreement, metres = _agreement(samples)
        fold_rows.append(
            {
                "fold": number,
                "holes": len(held),
                "metres": metres,
                "agreement": agreement,
                "contacts": len(logged),
                "matched": int(np.isfinite(depth).sum()),
                "seconds": time.perf_counter() - t0,
            }
        )
    if progress is not None:
        progress("done", 1.0)

    samples = pd.concat(all_samples, ignore_index=True)[
        ["hole_id", "fold", "depth", "length", "x", "y", "z", "logged", "predicted", "distance"]
    ]
    contacts = pd.concat(all_contacts, ignore_index=True)[
        ["hole_id", "fold", "depth", "above", "below", "x", "y", "z", "predicted_depth", "error",
         "distance"]
    ]  # fmt: skip
    spurious = pd.concat(all_spurious, ignore_index=True)[
        ["hole_id", "fold", "depth", "above", "below"]
    ]
    confusion = confusion_matrix(samples, categories)
    agreement, metres = _agreement(samples)
    edges = _bin_edges(
        samples["distance"].to_numpy(dtype=np.float64),
        samples["length"].to_numpy(dtype=np.float64),
        distance_bins,
    )

    hole_rows = []
    contact_groups = dict(tuple(contacts.groupby("hole_id", sort=False)))
    for hole, part in samples.groupby("hole_id", sort=False):
        a, m = _agreement(part)
        own = contact_groups.get(hole)
        scores = contact_scores(own) if own is not None else contact_scores(contacts.iloc[:0])
        hole_rows.append(
            {
                "hole_id": hole,
                "fold": int(part["fold"].iloc[0]),
                "metres": m,
                "agreement": a,
                "mean_distance": float(np.average(part["distance"], weights=part["length"])),
                "contacts": scores["logged"],
                "matched": scores["matched"],
                "mean_abs_error": scores["mean_abs"],
            }
        )

    return CVResult(
        layer=layer,
        categories=categories,
        folds=parts,
        leave_one_out=loo,
        seed=int(seed),
        step=float(step),
        samples=samples,
        contacts=contacts,
        spurious=spurious,
        confusion=confusion,
        per_unit=unit_scores(confusion),
        distance_table=by_distance(samples, contacts, edges),
        by_hole=pd.DataFrame(hole_rows),
        by_fold=pd.DataFrame(fold_rows),
        agreement=agreement,
        metres=metres,
        contact=contact_scores(contacts, len(spurious)),
        seconds=time.perf_counter() - t_start,
        builder=str(getattr(builder, "description", getattr(builder, "__name__", ""))),
        skipped_holes=skipped,
        metres_without_trace=lost,
    )


def cross_validate_units(
    table: pd.DataFrame,
    units: pd.Series,
    traces: HoleTraces,
    group: Sequence[str],
    *,
    settings: FieldSettings | None = None,
    builder: ModelBuilder | None = None,
    **options: Any,
) -> CVResult:
    """Hole-out cross-validation of the rock units of ``group``.

    ``builder`` defaults to the largest-field model of the group
    (:func:`largest_field_builder`) with ``settings``. Options: see :func:`cross_validate`.
    """
    group = tuple(str(u) for u in group)
    if builder is None:
        builder = largest_field_builder(group, settings)
    options.setdefault("scored", group)
    return cross_validate(table, units, traces, builder, layer="lith", **options)


def cross_validate_fault_blocks(
    table: pd.DataFrame,
    blocks: pd.Series,
    traces: HoleTraces,
    builder: ModelBuilder,
    **options: Any,
) -> CVResult:
    """Hole-out cross-validation of the fault blocks: ``blocks`` is the block of each row
    of ``table`` (NaN = unknown), ``builder`` gives a fault network (or anything with
    ``block_of(points)`` / ``labels(points)``) from the remaining holes, e.g.
    :func:`fault_network_builder`. Its "contacts" are the fault crossings."""
    return cross_validate(table, blocks, traces, builder, layer="fault_block", **options)
