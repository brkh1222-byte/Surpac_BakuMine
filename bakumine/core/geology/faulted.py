"""Rock units per fault block, and the throw a fault shows in the rock model (P9.T13).

Theory: research.md 5.7.5 ("Rock units per fault block", "Throw diagnostic"). Signatures:
geology_contracts.md section 7.

**One rock model per fault block.** A fault moves the rocks that existed when it slipped.
The fields of such units are fitted once per fault block, from the intervals of that block
only, and a point takes the model of its own block, ``L(p) = L_block(p)(p)``: this is what
lets a contact jump at the fault. Units younger than the faults (an intrusion that cuts
them) are not moved; they are fitted once from all holes and overprint every block. Which
units are younger is not something the logs say: ``post_fault`` names them. Without it
every unit is taken to be older than the faults (the rule of research.md), and the
provenance record says which reading was used.

An interval belongs to the block of its midpoint (:func:`assign_blocks`). An interval that
crosses a fault would put logged rock of one block into the fit of the other, so the
intervals are first cut where the block changes along the hole (:func:`split_at_faults`);
a cut is not a contact (the other block's part is unknown to the fit).

A block with fewer than ``min_holes`` holes (default 5) cannot carry a model of its own:
it uses the *pooled* model, fitted from all holes as if there were no fault. That is
written to the provenance record as an assumption: in such a block the units do not step
at the fault.

**Throw diagnostic** (:func:`throw_diagnostic`). Does the rock model show an offset where
the fault is? For one fault and one contact ``upper / lower``:

1. The modelled contact is located on vertical lines of a plan lattice (``spacing``): the
   label changes from ``upper`` (above) to ``lower`` (below), refined by bisection.
2. On each side of the fault the contact points between ``band[0]`` and ``band[1]``
   metres from the fault, inside the plan outline of the drilling (the convex hull of the
   hole points), give a least-squares plane ``z = a + b x + c y``. The points nearer than
   ``band[0]`` are left out: next to the fault each block's model is at the edge of its
   data. Outside the outline the contact is extrapolated and is no evidence.
3. The two planes are compared along the fault trace: the plan lattice points (half the
   spacing) where either plane meets the fault surface, inside the plan outline of the
   drilling. ``throw`` is the median of ``z_footwall - z_hangingwall`` there, the
   **vertical separation** of the contact at the fault; ``spread`` is half the range
   between its 16th and 84th percentiles along the trace (zero for parallel planes), and
   ``rms_footwall`` / ``rms_hangingwall`` are the residuals of the plane fits.

"Footwall" is the negative side of the fault and "hanging wall" its positive side (the
sign convention of research.md 5.7.5; for a vertical fault these are only names). A
positive throw means the positive side is down (normal sense for a dipping fault).

Vertical separation, throw and heave. The vertical separation equals the throw (the
vertical component of the slip) only when the contact is not tilted along the slip
direction. For a pure dip slip ``S`` on a fault of dip ``delta`` whose dip direction is the
horizontal unit vector ``h``, a contact with plan gradient ``g = (b, c)`` is separated
vertically by (derived here from "hanging-wall contact = footwall contact moved by the
slip vector", checked against an analytic case in the tests)::

    V = S (sin delta + cos delta (g . h))
    throw = S sin delta = V / (1 + (g . h) / tan delta),   heave = S cos delta

These are reported as ``throw_dip_slip``, ``heave_dip_slip`` and ``slip_dip_slip``. They
rest on the assumption of pure dip slip: one contact cannot show a strike-slip component.

What the diagnostic proves: an offset of the modelled contact across the fault, measured
20 to 60 m away from it on either side and carried to the fault along fitted planes. It is
``measured`` in the provenance record only when both blocks carry their own model and each
side has at least :data:`MIN_THROW_HOLES` holes that log the contact within ``band[1]`` of
the fault; otherwise it is ``assumed``, with the reason.

Coordinates are x = east, y = north, z = up.
"""

from __future__ import annotations

import dataclasses
import math
from collections.abc import Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd
from scipy.spatial import ConvexHull, Delaunay, QhullError

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.geology.lithology import (
    EventSpec,
    LithologyEvent,
    LithologyStack,
    fit_stack,
    group_contacts,
)
from bakumine.core.geology.provenance import ProvenanceEntry

if TYPE_CHECKING:
    from bakumine.core.geology.faults import FaultNetwork, FaultSurface
    from bakumine.core.geology.model import GeologyModel

__all__ = [
    "MIN_CONTACT_POINTS",
    "MIN_THROW_HOLES",
    "POOLED",
    "BlockStacks",
    "assign_blocks",
    "block_stack_builder",
    "fit_block_stacks",
    "logged_contacts",
    "split_at_faults",
    "throw_diagnostic",
    "throw_diagnostics",
    "throw_entry",
]

#: name of the stack fitted from all holes together (used by blocks with too few holes)
POOLED = "ALL"
#: modelled contact points a plane fit needs on each side of the fault
MIN_CONTACT_POINTS = 6
#: holes logging the contact within ``band[1]`` of the fault, on each side, for the throw
#: to count as measured (three points fix a plane)
MIN_THROW_HOLES = 3
_TASK = "P9.T13"
_STAGE = "faults"
_TOL = 1e-6
_CUT_TOLERANCE = 1e-4  # metres along the hole
_CHUNK = 400_000


def _entry(item: str, kind: str, source: str, used_for: str, detail: str = "", **extra: Any):
    return ProvenanceEntry(
        item=item, kind=kind, source=source, used_for=used_for, detail=detail,
        stage=_STAGE, task=_TASK, **extra,
    )  # fmt: skip


def _plural(n: int, one: str) -> str:
    return f"{n} {one}" if n == 1 else f"{n} {one}s"


def _depths(table: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    for name in ("hole_id", "depth_from", "depth_to"):
        if name not in table.columns:
            raise ValueError(f"interval table has no column {name!r}")
    hole = table["hole_id"].to_numpy(dtype=object)
    a = pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(dtype=np.float64)
    b = pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(dtype=np.float64)
    return hole, a, b


def _block_names(network: FaultNetwork, index: np.ndarray) -> np.ndarray:
    names = np.array([*network.blocks, np.nan], dtype=object)
    return names[np.asarray(index, dtype=np.int64)]


# --------------------------------------------------------------------------
# intervals and blocks
# --------------------------------------------------------------------------


def assign_blocks(table: pd.DataFrame, traces: HoleTraces, network: FaultNetwork) -> pd.Series:
    """Fault block of each interval: the block of its midpoint (NaN where the hole has no
    trace or the midpoint lies beyond it). Object series with the index of ``table``."""
    hole, a, b = _depths(table)
    if len(table) == 0:
        return pd.Series(np.empty(0, dtype=object), index=table.index, dtype=object)
    xyz, _ = traces.xyz_at(hole, 0.5 * (a + b), depth_tolerance=_TOL)
    index = np.asarray(network.block_index(xyz))
    return pd.Series(_block_names(network, index), index=table.index, dtype=object)


def split_at_faults(
    table: pd.DataFrame, traces: HoleTraces, network: FaultNetwork, *, step: float = 2.0
) -> pd.DataFrame:
    """The intervals cut where the fault block changes along the hole.

    The block is sampled every ``<= step`` metres along each interval; a change between
    two samples is located by bisection (to 0.1 mm of depth). Two changes closer together
    than ``step`` can be missed. Returns ``hole_id, depth_from, depth_to, row, block`` with
    a new index: ``row`` is the position of the source interval in ``table`` and ``block``
    the block of the piece's midpoint (:func:`assign_blocks`). Intervals without a valid
    trace are returned whole with block NaN.
    """
    if not (np.isfinite(step) and step > 0.0):
        raise ValueError("step must be > 0")
    hole, a, b = _depths(table)
    ok = np.isfinite(a) & np.isfinite(b) & (b > a)
    cuts: dict[int, list[float]] = {}
    rows = np.flatnonzero(ok)
    if len(rows):
        n = np.maximum(1, np.ceil((b[rows] - a[rows]) / step - 1e-9).astype(np.int64))
        rep = np.repeat(np.arange(len(rows)), n + 1)
        start = np.concatenate([[0], np.cumsum(n + 1)[:-1]])
        position = np.arange(len(rep)) - start[rep]
        depth = a[rows][rep] + (b[rows][rep] - a[rows][rep]) * position / n[rep]
        xyz, _ = traces.xyz_at(hole[rows][rep], depth, depth_tolerance=_TOL)
        index = np.asarray(network.block_index(xyz))
        change = (rep[1:] == rep[:-1]) & (index[1:] != index[:-1])
        change &= (index[1:] >= 0) & (index[:-1] >= 0)
        for k in np.flatnonzero(change):
            row = int(rows[rep[k]])
            lo, hi, first = float(depth[k]), float(depth[k + 1]), int(index[k])
            while hi - lo > _CUT_TOLERANCE:
                mid = 0.5 * (lo + hi)
                at, _ = traces.xyz_at(hole[row], np.array([mid]), depth_tolerance=_TOL)
                if int(np.asarray(network.block_index(at))[0]) == first:
                    lo = mid
                else:
                    hi = mid
            cuts.setdefault(row, []).append(0.5 * (lo + hi))
    out_row: list[int] = []
    out_a: list[float] = []
    out_b: list[float] = []
    for row in range(len(table)):
        edges = [a[row], *sorted(cuts.get(row, [])), b[row]]
        for lo, hi in zip(edges[:-1], edges[1:], strict=True):
            out_row.append(row)
            out_a.append(lo)
            out_b.append(hi)
    pieces = pd.DataFrame(
        {
            "hole_id": hole[np.array(out_row, dtype=np.int64)] if out_row else [],
            "depth_from": np.array(out_a, dtype=np.float64),
            "depth_to": np.array(out_b, dtype=np.float64),
            "row": np.array(out_row, dtype=np.int64),
        }
    )
    pieces["block"] = assign_blocks(pieces, traces, network).to_numpy(dtype=object)
    return pieces


# --------------------------------------------------------------------------
# one stack per block
# --------------------------------------------------------------------------


def _fit_event(
    table: pd.DataFrame,
    values: np.ndarray,
    traces: HoleTraces,
    spec: EventSpec,
    later_intrusions: Sequence[str],
) -> LithologyEvent:
    """One event from ``values`` (the unit of each row; NaN unknown). For an intrusion the
    younger intrusions are unknown: what they replaced is not observed."""
    series = pd.Series(values, index=table.index, dtype=object)
    if spec.mode == "intrusion" and later_intrusions:
        series = series.where(~series.isin(list(later_intrusions)), np.nan)
    return fit_stack(table, series, traces, [spec]).events[0]


def _tag(event: LithologyEvent, text: str) -> None:
    """Name the block in the event's provenance lines (the same items repeat per block)."""
    event.provenance = [
        dataclasses.replace(entry, item=f"{entry.item} ({text})") for entry in event.provenance
    ]


def fit_block_stacks(
    table: pd.DataFrame,
    units: pd.Series,
    traces: HoleTraces,
    network: FaultNetwork,
    events: Sequence[EventSpec],
    *,
    min_holes: int = 5,
    post_fault: Sequence[str] | None = None,
) -> tuple[dict[str, LithologyStack], list[str], list[ProvenanceEntry]]:
    """One stack per fault block; returns ``(stacks, pooled blocks, provenance entries)``.

    ``units`` gives the unit of each row of ``table`` (NaN = unknown; the caller masks
    cover and weathering codes, as for ``fit_stack``). ``events`` are listed from the
    oldest to the youngest.

    ``post_fault`` names the units younger than every fault of the network. An event
    whose units are all listed is fitted once from all intervals and shared by every
    block's stack (the same event object). The other events are older than the faults:
    each block fits its own from its own pieces of the intervals
    (:func:`split_at_faults`). The post-fault events must be the last ones of ``events``,
    and an event cannot mix older and younger units. Default: no unit is younger.

    A block with fewer than ``min_holes`` holes that log a unit of the pre-fault events,
    or whose own fit fails (a unit of the event is not logged in the block), takes the
    pooled events: the pre-fault events fitted from all intervals, ignoring the faults.
    All pooled blocks share one stack object named :data:`POOLED`. Every pooled block is
    an ``assumed`` provenance entry.

    Each intrusion event is fitted with the units of the later intrusion events made
    unknown (``mask_younger`` of the sequence module).
    """
    events = list(events)
    if not events:
        raise ValueError("give at least one event")
    if min_holes < 1:
        raise ValueError("min_holes must be >= 1")
    if len(units) != len(table):
        raise ValueError(f"units has {len(units)} values, the table has {len(table)} rows")
    column = np.asarray(
        units.to_numpy(dtype=object) if isinstance(units, pd.Series) else units, dtype=object
    )
    younger = {str(u) for u in (post_fault or ())}
    modelled = {u for spec in events for u in spec.units}
    if younger - modelled:
        raise ValueError(
            f"post_fault names unit(s) no event models: {', '.join(sorted(younger - modelled))}"
        )
    shared: list[bool] = []
    for spec in events:
        inside = [u in younger for u in spec.units]
        if any(inside) and not all(inside):
            raise ValueError(
                f"the event of {', '.join(spec.units)} mixes units older and younger than "
                "the faults: split it into two events"
            )
        shared.append(all(inside))
    if shared != sorted(shared):
        raise ValueError(
            "events are listed from the oldest to the youngest: a unit older than the "
            "faults cannot follow one that is younger"
        )
    later = [
        [u for s in events[i + 1 :] if s.mode == "intrusion" for u in s.units]
        for i in range(len(events))
    ]
    old = [i for i, is_shared in enumerate(shared) if not is_shared]
    old_units = {u for i in old for u in events[i].units}
    blocks = tuple(network.blocks)

    pieces = split_at_faults(table, traces, network)
    piece_units = column[pieces["row"].to_numpy()]
    piece_block = pieces["block"].to_numpy(dtype=object)
    piece_table = pieces[["hole_id", "depth_from", "depth_to"]]
    n_cut = int(len(pieces) - len(table))
    known = pd.Series(piece_units).isin(list(old_units)).to_numpy()
    holes = {
        b: int(pd.Series(piece_table["hole_id"].to_numpy()[known & (piece_block == b)]).nunique())
        for b in blocks
    }

    entries: list[ProvenanceEntry] = []
    stacks: dict[str, LithologyStack] = {}
    pooled: list[str] = []
    reasons: dict[str, str] = {}
    own: dict[str, list[LithologyEvent]] = {}
    for block in blocks if old else ():
        if holes[block] < min_holes:
            reasons[block] = (
                f"{_plural(holes[block], 'hole')} in the block, fewer than the "
                f"{min_holes} a model of its own needs"
            )
            continue
        values = np.where(piece_block == block, piece_units, np.nan).astype(object)
        try:
            fitted = [_fit_event(piece_table, values, traces, events[i], later[i]) for i in old]
        except ValueError as error:
            reasons[block] = f"the block's own fit failed ({error})"
            continue
        for event in fitted:
            _tag(event, f"fault block {block}")
        own[block] = fitted

    young = [
        _fit_event(table, column, traces, events[i], later[i])
        for i, is_shared in enumerate(shared)
        if is_shared
    ]
    pooled_stack: LithologyStack | None = None
    if reasons:
        pooled_old = [_fit_event(table, column, traces, events[i], later[i]) for i in old]
        for event in pooled_old:
            _tag(event, "all holes pooled, used by fault block " + ", ".join(reasons))
        pooled_stack = LithologyStack(POOLED, (*pooled_old, *young))
    for block in blocks:
        if block in own:
            stacks[block] = LithologyStack(block, (*own[block], *young))
        elif block in reasons:
            stacks[block] = pooled_stack
            pooled.append(block)
        else:  # no pre-fault event at all
            stacks[block] = LithologyStack(block, tuple(young))

    old_text = ", ".join(u for i in old for u in events[i].units) or "none"
    young_text = ", ".join(u for i, s in enumerate(shared) if s for u in events[i].units) or "none"
    faults = ", ".join(f.name for f in network.faults) or "none"
    entries.append(
        _entry(
            "Age of the faults relative to the rock units",
            "assumed",
            (
                "given (the units listed as younger than the faults)"
                if post_fault is not None
                else "default (every unit is older than the faults)"
            ),
            "which units step at a fault and which are continuous across it",
            f"faults: {faults}; offset by them, one model per fault block: {old_text}; "
            f"younger, one model across the blocks: {young_text}",
        )
    )
    if old:
        entries.append(
            _entry(
                "Rock-type log by fault block",
                "measured",
                "the intervals of each hole, by the fault block of their midpoint",
                "rock-unit fields of each fault block",
                "; ".join(f"{b}: {_plural(holes[b], 'hole')}" for b in blocks)
                + (
                    f"; {_plural(n_cut, 'interval')} crossing a fault cut at the fault"
                    if n_cut
                    else ""
                ),
                count=int(sum(holes.values())),
            )
        )
    for block in pooled:
        entries.append(
            _entry(
                f"Fault block {block}: pooled rock model",
                "assumed",
                reasons[block],
                f"rock units of fault block {block}",
                f"the fields fitted from all holes together are used ({old_text}): in this "
                "block the units do not step at the fault",
                count=holes[block],
            )
        )
    return stacks, pooled, entries


class BlockStacks:
    """The stacks of the fault blocks as one object: ``labels(points)`` are integer codes
    into ``categories`` (-1 = none), each point labelled by the stack of its block."""

    def __init__(self, network: FaultNetwork, stacks: Mapping[str, LithologyStack]) -> None:
        missing = [b for b in network.blocks if b not in stacks]
        if missing:
            raise ValueError(f"no lithology stack for block(s) {', '.join(missing)}")
        self.network = network
        self.stacks = dict(stacks)
        seen: dict[str, None] = {}
        for block in network.blocks:
            for unit in self.stacks[block].categories:
                seen.setdefault(str(unit))
        self.categories: tuple[str, ...] = tuple(seen)

    def labels(self, points) -> np.ndarray:
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        index = {c: i for i, c in enumerate(self.categories)}
        out = np.full(len(pts), -1, dtype=np.int16)
        block_index = np.asarray(self.network.block_index(pts))
        for b, block in enumerate(self.network.blocks):
            sel = block_index == b
            stack = self.stacks[block]
            if not sel.any() or not stack.events:
                continue
            lookup = np.array([index[c] for c in stack.categories] + [-1], dtype=np.int16)
            out[sel] = lookup[stack.labels(pts[sel]).astype(np.int64)]
        return out


def block_stack_builder(
    network: FaultNetwork,
    events: Sequence[EventSpec],
    *,
    min_holes: int = 5,
    post_fault: Sequence[str] | None = None,
):
    """Builder of the faulted rock model for ``validation.cross_validate``: the network is
    kept as it is (it is not inferred again in each fold), the stacks are fitted from the
    remaining holes with :func:`fit_block_stacks`."""
    events = tuple(events)

    def build(table: pd.DataFrame, labels: pd.Series, traces: HoleTraces) -> BlockStacks:
        stacks, _, _ = fit_block_stacks(
            table, labels, traces, network, events, min_holes=min_holes, post_fault=post_fault
        )
        return BlockStacks(network, stacks)

    build.description = (
        "one stack per fault block ("
        + ", ".join(network.blocks)
        + "): "
        + "; ".join(f"{e.mode} of {', '.join(e.units)}" for e in events)
    )
    return build


# --------------------------------------------------------------------------
# throw diagnostic
# --------------------------------------------------------------------------


def logged_contacts(
    table: pd.DataFrame, units: pd.Series, traces: HoleTraces, upper: str, lower: str
) -> pd.DataFrame:
    """The logged contacts between ``upper`` and ``lower`` (in either order down the hole):
    ``hole_id, depth, above, below, x, y, z``. A contact exists where runs of the two units
    abut; an unknown interval or another unit between them hides it."""
    column = pd.Series(
        np.asarray(
            units.to_numpy(dtype=object) if isinstance(units, pd.Series) else units, dtype=object
        )
    )
    group = tuple(sorted({str(u) for u in column.dropna().unique()}))
    if len(group) < 2:
        return pd.DataFrame(columns=["hole_id", "depth", "above", "below", "x", "y", "z"])
    contacts = group_contacts(table, units, group, traces)
    pair = {str(upper), str(lower)}
    keep = [
        {str(a), str(b)} == pair for a, b in zip(contacts["above"], contacts["below"], strict=True)
    ]
    return contacts[np.array(keep, dtype=bool)].reset_index(drop=True)


def _rock_categories(model: GeologyModel) -> np.ndarray:
    """Indices of the ``lith`` categories that come from the stacks (no cover, no AIR)."""
    modelled = {c for stack in model.stacks.values() for c in stack.categories}
    return np.array(
        [i for i, c in enumerate(model.categories("lith")) if c in modelled], dtype=np.int64
    )


def _contact_scan(model: GeologyModel, spacing: float) -> dict[str, np.ndarray]:
    """Every change of rock unit along vertical lines of a plan lattice.

    Lines every ``spacing`` metres over the plan extent of the model grid, labels (from
    the sampled grid) every half vertical node spacing, each change refined by bisection
    to about a millimetre. Returns ``xyz (m, 3)``, ``above (m,)`` and ``below (m,)`` (codes
    of the ``lith`` layer). Changes into or out of the cover, the air or "no unit" are not
    contacts.
    """
    if not (np.isfinite(spacing) and spacing > 0.0):
        raise ValueError("spacing must be > 0")
    grid = model.grid
    lo = grid.origin
    hi = lo + grid.spacing * (np.array(grid.shape) - 1)
    xs = np.arange(lo[0], hi[0] + 1e-9, spacing)
    ys = np.arange(lo[1], hi[1] + 1e-9, spacing)
    dz = 0.5 * float(grid.spacing[2])
    zs = np.arange(hi[2], lo[2] - 1e-9, -dz)  # top to bottom
    xy = np.stack(np.meshgrid(xs, ys, indexing="ij"), axis=-1).reshape(-1, 2)
    rock = np.zeros(len(model.categories("lith")) + 1, dtype=bool)  # last entry: label -1
    rock[_rock_categories(model)] = True
    found_xy, found_top, found_bottom, found_above, found_below = [], [], [], [], []
    per_chunk = max(1, _CHUNK // len(zs))
    for i0 in range(0, len(xy), per_chunk):
        part = xy[i0 : i0 + per_chunk]
        cols = np.column_stack([np.repeat(part, len(zs), axis=0), np.tile(zs, len(part))])
        lab = np.asarray(model.labels(cols), dtype=np.int64).reshape(len(part), len(zs))
        change = (lab[:, :-1] != lab[:, 1:]) & rock[lab[:, :-1]] & rock[lab[:, 1:]]
        i, k = np.nonzero(change)
        found_xy.append(part[i])
        found_top.append(zs[k])
        found_bottom.append(zs[k + 1])
        found_above.append(lab[i, k])
        found_below.append(lab[i, k + 1])
    xy = np.concatenate(found_xy)
    top, bottom = np.concatenate(found_top), np.concatenate(found_bottom)
    above, below = np.concatenate(found_above), np.concatenate(found_below)
    under = below.copy()  # the label at `bottom`, which moves
    for _ in range(12):
        if not len(xy):
            break
        mid = 0.5 * (top + bottom)
        got = np.asarray(model.labels(np.column_stack([xy, mid])), dtype=np.int64)
        same = got == above
        top = np.where(same, mid, top)
        bottom = np.where(same, bottom, mid)
        under = np.where(same, under, got)
    keep = under == below  # a third unit in between: not a contact of the pair
    return {
        "xyz": np.column_stack([xy, 0.5 * (top + bottom)])[keep],
        "above": above[keep],
        "below": below[keep],
    }


def _fit_z_plane(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Least-squares ``z = a0 + b (x - x0) + c (y - y0)``: ``((a0, b, c), (x0, y0), rms)``,
    or None when the points do not span a plane in plan."""
    centre = xyz[:, :2].mean(axis=0)
    design = np.column_stack([np.ones(len(xyz)), xyz[:, :2] - centre])
    scale = max(float(np.abs(design[:, 1:]).max()), 1e-12)
    design[:, 1:] /= scale
    coef, _, rank, _ = np.linalg.lstsq(design, xyz[:, 2], rcond=1e-9)
    if rank < 3:
        return None
    rms = float(np.sqrt(np.mean((design @ coef - xyz[:, 2]) ** 2)))
    coef = coef.copy()
    coef[1:] /= scale
    return coef, centre, rms


def _plane_z(plane: tuple[np.ndarray, np.ndarray, float], xy: np.ndarray) -> np.ndarray:
    coef, centre, _ = plane
    return coef[0] + (xy - centre) @ coef[1:]


def _world_plane(plane: tuple[np.ndarray, np.ndarray, float]) -> tuple[float, float, float]:
    coef, centre, _ = plane
    return (float(coef[0] - coef[1:] @ centre), float(coef[1]), float(coef[2]))


def _plan_outline(points: np.ndarray | None):
    """``inside(xy) -> bool`` for the plan convex hull of the hole points, or None."""
    if points is None or len(points) < 3:
        return None
    xy = np.unique(np.asarray(points, dtype=np.float64)[:, :2], axis=0)
    try:
        hull = Delaunay(xy[ConvexHull(xy).vertices])
    except (QhullError, ValueError):
        return None
    return lambda q: hull.find_simplex(q) >= 0


def _fault_of(model: GeologyModel, fault: str) -> FaultSurface:
    if model.network is None:
        raise ValueError("the model has no faults")
    for surface in model.network.faults:
        if surface.name == str(fault):
            return surface
    names = ", ".join(f.name for f in model.network.faults)
    raise ValueError(f"no fault {fault!r} in the model (faults: {names})")


def throw_diagnostic(
    model: GeologyModel,
    fault: str,
    upper: str,
    lower: str,
    *,
    band: tuple[float, float] = (20.0, 60.0),
    spacing: float = 10.0,
    logged: pd.DataFrame | None = None,
    scan: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Vertical separation of the modelled contact ``upper / lower`` across ``fault``.

    Method: the module text. ``logged``: the logged contacts of the pair
    (:func:`logged_contacts`, columns ``hole_id, x, y, z``), to count the holes that
    support each side; without it the holes are not counted and the result is ``assumed``.
    ``scan``: the output of an earlier contact scan of the same model and spacing.

    Keys of the result: ``fault, upper, lower, blocks`` (negative side, positive side),
    ``throw`` (vertical separation, metres, positive = positive side down; NaN when it
    cannot be measured), ``spread``, ``n_trace``, ``at`` (``"fault trace"`` or
    ``"band points"``), ``n_footwall, n_hangingwall`` (modelled contact points used),
    ``plane_footwall, plane_hangingwall`` (``(a, b, c)`` of ``z = a + b x + c y``, or None),
    ``rms_footwall, rms_hangingwall``, ``holes_footwall, holes_hangingwall`` (holes logging
    the contact within ``band[1]`` of the fault; None without ``logged``),
    ``fault_dip, fault_dip_azimuth, throw_dip_slip, heave_dip_slip, slip_dip_slip``,
    ``offset_seen`` (the separation exceeds twice its uncertainty, at least one vertical
    node spacing), ``pooled`` (pooled
    blocks among the two), ``kind`` (``measured`` / ``assumed``), ``band``, ``spacing``,
    ``note``.
    """
    near, far = float(band[0]), float(band[1])
    if not (0.0 <= near < far and math.isfinite(far)):
        raise ValueError("band must be (near, far) with 0 <= near < far")
    surface = _fault_of(model, fault)
    categories = model.categories("lith")
    for unit in (upper, lower):
        if unit not in categories:
            raise ValueError(f"{unit!r} is not a rock unit of the model")
    if scan is None:
        scan = _contact_scan(model, spacing)
    pair = (scan["above"] == categories.index(upper)) & (scan["below"] == categories.index(lower))
    xyz = scan["xyz"][pair]
    offset = surface.offset(xyz) if len(xyz) else np.zeros(0)
    in_band = (np.abs(offset) >= near) & (np.abs(offset) <= far)
    outline = _plan_outline(model.hole_points)
    if outline is not None and len(xyz):
        in_band &= outline(xyz[:, :2])  # beyond the drilling the contact is extrapolated
    block = np.empty(0, dtype=object)
    if len(xyz):
        block = _block_names(model.network, model.network.block_index(xyz))
    sides: dict[int, np.ndarray] = {}
    names: list[str | None] = list(surface.blocks) if surface.blocks is not None else [None, None]
    for k, sign in enumerate((-1, 1)):
        sel = in_band & ((offset > 0.0) == (sign > 0))
        if names[k] is None and sel.any():  # the fault does not name its blocks
            found, counts = np.unique(block[sel].astype(str), return_counts=True)
            names[k] = str(found[int(np.argmax(counts))])
        if names[k] is not None:
            sel &= block == names[k]
        sides[sign] = xyz[sel]
    pooled = [b for b in names if b is not None and b in model.pooled_blocks]
    reasons: list[str] = []
    if pooled:
        reasons.append(
            f"block {', '.join(pooled)} uses the pooled model, which has no step at the fault"
        )

    holes: dict[int, int | None] = {-1: None, 1: None}
    if logged is not None:
        at = logged[["x", "y", "z"]].to_numpy(dtype=np.float64).reshape(-1, 3)
        where = surface.offset(at) if len(at) else np.zeros(0)
        hole = logged["hole_id"].to_numpy(dtype=object)
        for sign in (-1, 1):
            sel = (np.abs(where) <= far) & ((where > 0.0) == (sign > 0))
            holes[sign] = int(pd.Series(hole[sel]).nunique())

    plane = surface.plane
    result: dict[str, Any] = {
        "fault": surface.name,
        "upper": str(upper),
        "lower": str(lower),
        "blocks": names,
        "throw": math.nan,
        "spread": math.nan,
        "n_trace": 0,
        "at": None,
        "n_footwall": int(len(sides[-1])),
        "n_hangingwall": int(len(sides[1])),
        "plane_footwall": None,
        "plane_hangingwall": None,
        "rms_footwall": math.nan,
        "rms_hangingwall": math.nan,
        "holes_footwall": holes[-1],
        "holes_hangingwall": holes[1],
        "fault_dip": float(plane.dip),
        "fault_dip_azimuth": float(plane.dip_azimuth),
        "throw_dip_slip": math.nan,
        "heave_dip_slip": math.nan,
        "slip_dip_slip": math.nan,
        "offset_seen": False,
        "pooled": pooled,
        "kind": "assumed",
        "band": [near, far],
        "spacing": float(spacing),
        "note": "",
    }
    fits = {
        sign: (_fit_z_plane(pts) if len(pts) >= MIN_CONTACT_POINTS else None)
        for sign, pts in sides.items()
    }
    if fits[-1] is None or fits[1] is None:
        short = [
            f"{name} ({len(sides[sign])} points)"
            for sign, name in ((-1, "footwall"), (1, "hanging wall"))
            if fits[sign] is None
        ]
        result["note"] = "; ".join(
            [
                f"not measured: the modelled contact {upper} / {lower} has too few points "
                f"{near:g} to {far:g} m from the fault, inside the drilled outline, in the "
                f"{' and the '.join(short)} ({MIN_CONTACT_POINTS} that span a plane are "
                "needed on each side)",
                *reasons,
            ]
        )
        return result
    result["plane_footwall"] = _world_plane(fits[-1])
    result["plane_hangingwall"] = _world_plane(fits[1])
    result["rms_footwall"], result["rms_hangingwall"] = fits[-1][2], fits[1][2]

    # where the two planes meet the fault, inside the plan outline of the drilling
    grid = model.grid
    lo = grid.origin
    hi = lo + grid.spacing * (np.array(grid.shape) - 1)
    step = 0.5 * float(spacing)
    plan = np.stack(
        np.meshgrid(
            np.arange(lo[0], hi[0] + 1e-9, step), np.arange(lo[1], hi[1] + 1e-9, step),
            indexing="ij",
        ),
        axis=-1,
    ).reshape(-1, 2)  # fmt: skip
    on_trace = np.zeros(len(plan), dtype=bool)
    for sign in (-1, 1):
        z = _plane_z(fits[sign], plan)
        distance = np.abs(surface.offset(np.column_stack([plan, z])))
        on_trace |= (distance <= 0.5 * step) & (z >= lo[2]) & (z <= hi[2])
    if outline is not None:
        on_trace &= outline(plan)
    where = plan[on_trace]
    result["at"] = "fault trace"
    notes: list[str] = []
    if len(where) == 0:
        where = np.concatenate([sides[-1][:, :2], sides[1][:, :2]])
        result["at"] = "band points"
        notes.append(
            "the contact does not meet the fault inside the drilled area: the planes are "
            "compared at the plan positions of the contact points instead"
        )
    separation = _plane_z(fits[-1], where) - _plane_z(fits[1], where)
    throw = float(np.median(separation))
    p16, p84 = np.percentile(separation, [16.0, 84.0])
    spread = 0.5 * float(p84 - p16)
    result.update(throw=throw, spread=spread, n_trace=int(len(where)))
    uncertainty = math.sqrt(spread**2 + fits[-1][2] ** 2 + fits[1][2] ** 2)
    # the model cannot show a step smaller than about a cell of its grid
    result["offset_seen"] = bool(abs(throw) > 2.0 * max(uncertainty, 0.5 * grid.spacing[2]))

    # pure dip slip: V = S (sin d + cos d (g . h))
    d, a = math.radians(plane.dip), math.radians(plane.dip_azimuth)
    gradient = 0.5 * (fits[-1][0][1:] + fits[1][0][1:])
    along = float(gradient @ np.array([math.sin(a), math.cos(a)]))
    factor = math.sin(d) + math.cos(d) * along
    if abs(factor) > 0.1:
        slip = throw / factor
        result.update(
            slip_dip_slip=slip,
            throw_dip_slip=slip * math.sin(d),
            heave_dip_slip=slip * math.cos(d),
        )
    else:
        notes.append(
            "the contact is nearly parallel to the slip direction: dip-slip throw not given"
        )

    if logged is None:
        reasons.append("the holes that log the contact near the fault were not counted")
    else:
        for sign, name in ((-1, "footwall"), (1, "hanging wall")):
            if holes[sign] < MIN_THROW_HOLES:
                reasons.append(
                    f"{_plural(holes[sign], 'hole')} with the logged contact within {far:g} m "
                    f"of the fault in the {name} (fewer than {MIN_THROW_HOLES}): the contact "
                    "there is extrapolated"
                )
    result["kind"] = "assumed" if reasons else "measured"
    seen = (
        "the rock model shows an offset at the fault"
        if result["offset_seen"]
        else "no offset beyond the uncertainty: the rock units do not support this fault"
    )
    result["note"] = "; ".join([seen, *reasons, *notes])
    return result


def throw_entry(result: Mapping[str, Any]) -> ProvenanceEntry:
    """The provenance line of one :func:`throw_diagnostic` result."""
    fault, upper, lower = result["fault"], result["upper"], result["lower"]
    near, far = result["band"]
    item = f"Throw of fault {fault} on the contact {upper} / {lower}"
    used_for = "check of the fault: does the rock model step where the fault is"
    throw = result["throw"]
    if not math.isfinite(throw):
        return _entry(item, "assumed", "not measured", used_for, result["note"])
    counted = result["holes_footwall"] is not None
    holes = (
        f"; holes logging the contact within {far:g} m of the fault: "
        f"{result['holes_footwall']} footwall, {result['holes_hangingwall']} hanging wall"
        if counted
        else ""
    )
    source = (
        f"modelled contact on both sides of the fault, {near:g} to {far:g} m from it "
        f"({result['n_footwall']} / {result['n_hangingwall']} contact points)" + holes
    )
    sense = "positive side down" if throw > 0 else "positive side up"
    detail = (
        f"vertical separation {throw:+.1f} m ({sense}), spread {result['spread']:.1f} m along "
        f"the {result['at']}; plane residuals {result['rms_footwall']:.1f} / "
        f"{result['rms_hangingwall']:.1f} m"
    )
    if math.isfinite(result["throw_dip_slip"]):
        detail += (
            f"; read as dip slip on a fault dipping {result['fault_dip']:.0f}: throw "
            f"{result['throw_dip_slip']:+.1f} m, heave {result['heave_dip_slip']:+.1f} m"
        )
    detail += f"; {result['note']}"
    return _entry(item, result["kind"], source, used_for, detail, value=float(throw))


def _default_pairs(model: GeologyModel, blocks: Sequence[str | None]) -> list[tuple[str, str]]:
    """Contacts to try: consecutive units of the sequence events of the two blocks."""
    pairs: dict[tuple[str, str], None] = {}
    for block in blocks:
        stack = model.stacks.get(block) if block is not None else None
        for event in stack.events if stack is not None else ():
            if event.mode == "sequence":
                for a, b in zip(event.units[:-1], event.units[1:], strict=True):
                    pairs.setdefault((str(a), str(b)))
    return list(pairs)


def throw_diagnostics(
    model: GeologyModel,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    table: pd.DataFrame | None = None,
    units: pd.Series | None = None,
    traces: HoleTraces | None = None,
    band: tuple[float, float] = (20.0, 60.0),
    spacing: float = 10.0,
) -> tuple[list[dict[str, Any]], list[ProvenanceEntry]]:
    """:func:`throw_diagnostic` for every fault and every contact, with the provenance.

    ``pairs``: ``(upper, lower)`` contacts. Default: the consecutive units of the
    ``sequence`` events of the two blocks of each fault, plus, for ``argmax`` groups
    (whose order is not known), every ordered pair of units that the model shows as a
    contact with enough points on both sides. With ``table``, ``units`` and ``traces`` the
    holes that log each contact near the fault are counted (needed for ``measured``).
    Returns ``(results, entries)``; the results go to ``diagnostics.json`` under
    ``"throw"``.
    """
    if model.network is None:
        return [], []
    scan = _contact_scan(model, spacing)
    categories = model.categories("lith")
    seen = {
        (categories[a], categories[b])
        for a, b in {(int(a), int(b)) for a, b in zip(scan["above"], scan["below"], strict=True)}
    }
    with_logs = table is not None and units is not None and traces is not None
    results: list[dict[str, Any]] = []
    for surface in model.network.faults:
        blocks = surface.blocks if surface.blocks is not None else model.blocks
        if pairs is not None:
            wanted, optional = [(str(a), str(b)) for a, b in pairs], []
        else:
            wanted = _default_pairs(model, blocks)
            grouped = {
                (str(a), str(b))
                for block in blocks
                for event in model.stacks[block].events
                if event.mode == "argmax"
                for a in event.units
                for b in event.units
                if a != b
            }
            optional = sorted(grouped & seen - set(wanted))
        for upper, lower in [*wanted, *optional]:
            logged = logged_contacts(table, units, traces, upper, lower) if with_logs else None
            result = throw_diagnostic(
                model, surface.name, upper, lower,
                band=band, spacing=spacing, logged=logged, scan=scan,
            )  # fmt: skip
            if (upper, lower) in optional and not math.isfinite(result["throw"]):
                continue
            results.append(result)
    return results, [throw_entry(result) for result in results]
