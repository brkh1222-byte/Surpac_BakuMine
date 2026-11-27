"""Fault surfaces and fault blocks (research.md 5.7.5, geology_contracts.md 8).

A fault is known from three kinds of evidence, in falling order of trust:

- an **imported surface** from the geologist (a mesh, points, or a point with dip and dip
  azimuth): :func:`import_fault`;
- **crossings along holes** (:func:`fault_picks`): where a hole passes from one fault block to
  another, or where a fault zone is logged. A plane is fitted through them (:func:`fit_plane`,
  weighted principal components);
- the **fault-block codes** alone (:func:`labelled_points`): the plane that best separates the
  interval midpoints of the two blocks (:func:`separating_plane`, a soft-margin linear support
  vector machine).

:func:`infer_fault` chooses between the last two. :class:`FaultNetwork` turns several faults
into fault blocks: the faults split space into cells by their sign vector, and each cell takes
the block of the labelled points in it.

Sign convention: the normal of a plane points up (``n_z >= 0``; for a vertical plane it points
to an azimuth in ``[0, 180)``). A point with ``n . (p - c) > 0`` is on the positive side, which
for a fault that is not vertical is the hanging wall.

Every inferred fault is an *assumption* and says so in its provenance entries, with the
method, dip, dip azimuth and misfit. :func:`faults_provenance` adds the *missing* line for the
geologist's fault surfaces as long as any fault is inferred.
"""

from __future__ import annotations

import itertools
import json
import math
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.geology.provenance import ProvenanceEntry
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io.obj import read_obj, write_obj

FORMAT = "bakumine.fault"
NETWORK_FORMAT = "bakumine.fault_network"
VERSION = 1

FAULT_SOURCES = (
    "inferred-plane",
    "inferred-separating",
    "imported-mesh",
    "imported-points",
    "imported-plane",
)
PICK_SOURCES = ("fault-zone", "gap-middle")
PICK_COLUMNS = (
    "hole_id",
    "block_above",
    "block_below",
    "gap_from",
    "gap_to",
    "depth",
    "x",
    "y",
    "z",
    "uncertainty",
    "weight",
    "source",
)

_STAGE = "faults"
_TASK = "P9.T8"
_INTERVAL_COLUMNS = ("hole_id", "depth_from", "depth_to")
_VERTICAL = 1e-12  # |n_z| below this is a vertical plane
_COLLINEAR = 1e-3  # second / first principal standard deviation below this is "on one line"
_SVM_TOLERANCE = 1e-8
_MAX_FAULTS = 16


class FaultError(ValueError):
    """Fault evidence that cannot give a fault surface or a fault block."""


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------


def _xyz(points: Any, what: str, *, allow_empty: bool = False) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.size == 0 and allow_empty:
        return np.zeros((0, 3))
    if p.ndim == 1 and p.shape == (3,):
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != 3:
        raise FaultError(f"{what} must have shape (n, 3), got {p.shape}")
    if not np.isfinite(p).all():
        raise FaultError(f"{what} contain NaN or infinite coordinates")
    return p


def _upward_normal(dip: float, dip_azimuth: float) -> np.ndarray:
    dip, azimuth = float(dip), float(dip_azimuth)
    if not (math.isfinite(dip) and math.isfinite(azimuth)):
        raise FaultError("dip and dip azimuth must be finite numbers")
    if not 0.0 <= dip <= 90.0:
        raise FaultError(f"dip must be between 0 and 90 degrees, got {dip:g}")
    d, a = math.radians(dip), math.radians(azimuth)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def _oriented(normal: np.ndarray) -> np.ndarray:
    """Unit normal turned up; a vertical one is turned to an azimuth in [0, 180)."""
    n = np.asarray(normal, dtype=np.float64).reshape(-1)
    if n.shape != (3,) or not np.isfinite(n).all():
        raise FaultError("a plane normal must be three finite numbers")
    length = float(np.linalg.norm(n))
    if length == 0.0:
        raise FaultError("a plane normal must not be the zero vector")
    n = n / length
    if abs(n[2]) < _VERTICAL:
        n[2] = 0.0
        if abs(n[0]) < _VERTICAL:
            n[0] = 0.0
            flip = n[1] < 0.0
        else:
            flip = n[0] < 0.0
        n = n / np.linalg.norm(n)
    else:
        flip = n[2] < 0.0
    return -n if flip else n + 0.0


def _text_angle(value: float, width: int = 0) -> str:
    return f"{round(value) % 360 if width else round(value):0{width}d}"


def _orientation_text(plane: Plane) -> str:
    return f"plane dip {_text_angle(plane.dip)}, dip azimuth {_text_angle(plane.dip_azimuth, 3)}"


def _plural(n: int, one: str, many: str | None = None) -> str:
    return f"{n} {one if n == 1 else (many or one + 's')}"


# --------------------------------------------------------------------------
# evidence from the logs
# --------------------------------------------------------------------------


def block_codes(values: pd.Series, pattern: str) -> pd.Series:
    """Fault-block code of each logged value: the first group of ``pattern``, NaN otherwise.

    ``pattern`` is a regular expression with at least one group, searched in the text of the
    value (anchor it with ``^`` when the code is a prefix), e.g. ``^(FB\\d)-``.
    """
    try:
        regex = re.compile(pattern)
    except re.error as exc:
        raise FaultError(
            f"fault-block pattern {pattern!r} is not a regular expression: {exc}"
        ) from None
    if regex.groups < 1:
        raise FaultError(f"fault-block pattern {pattern!r} needs one group, e.g. '^(FB\\d)-'")
    values = pd.Series(values)
    out: list[object] = []
    for value in values.to_numpy(dtype=object):
        code: object = np.nan
        if isinstance(value, str):
            match = regex.search(value)
            if match is not None and match.group(1):
                code = match.group(1)
        out.append(code)
    return pd.Series(out, index=values.index, dtype=object)


def _known_intervals(table: pd.DataFrame, blocks: Any) -> pd.DataFrame:
    """Intervals with a known block: ``hole_id, depth_from, depth_to, block``, sorted."""
    missing = [c for c in _INTERVAL_COLUMNS if c not in table.columns]
    if missing:
        raise FaultError(f"interval table lacks column(s) {', '.join(missing)}")
    if isinstance(blocks, pd.Series):
        block = blocks.reindex(table.index).to_numpy(dtype=object)
    else:
        block = np.asarray(blocks, dtype=object)
        if block.shape != (len(table),):
            raise FaultError("blocks must have one value per row of the interval table")
    frame = pd.DataFrame(
        {
            "hole_id": table["hole_id"].to_numpy(dtype=object),
            "depth_from": table["depth_from"].to_numpy(dtype=np.float64),
            "depth_to": table["depth_to"].to_numpy(dtype=np.float64),
            "block": block,
        }
    )
    keep = frame["block"].notna() & frame["depth_from"].notna() & frame["depth_to"].notna()
    frame = frame[keep & (frame["depth_to"] >= frame["depth_from"])]
    frame["block"] = frame["block"].astype(str)
    return frame.sort_values(["hole_id", "depth_from", "depth_to"], kind="stable").reset_index(
        drop=True
    )


def labelled_points(
    table: pd.DataFrame, blocks: pd.Series, traces: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Midpoints of the intervals with a known fault block.

    Returns ``xyz (n, 3)``, ``block (n,)`` and ``hole_id (n,)`` (object arrays). Intervals of
    holes without a trace, or deeper than the trace, are left out.
    """
    frame = _known_intervals(table, blocks)
    hole = frame["hole_id"].to_numpy(dtype=object)
    mid = 0.5 * (frame["depth_from"].to_numpy() + frame["depth_to"].to_numpy())
    if len(frame) == 0:
        return np.zeros((0, 3)), np.empty(0, dtype=object), np.empty(0, dtype=object)
    xyz, _ = traces.xyz_at(hole, mid)
    keep = np.isfinite(xyz).all(axis=1)
    return xyz[keep], frame["block"].to_numpy(dtype=object)[keep], hole[keep]


def _merged_zones(fault_zone: pd.DataFrame | None) -> dict[object, list[tuple[float, float]]]:
    """Fault-zone intervals per hole, touching ones merged."""
    if fault_zone is None or len(fault_zone) == 0:
        return {}
    missing = [c for c in _INTERVAL_COLUMNS if c not in fault_zone.columns]
    if missing:
        raise FaultError(f"fault_zone lacks column(s) {', '.join(missing)}")
    zones: dict[object, list[tuple[float, float]]] = {}
    frame = fault_zone.dropna(subset=list(_INTERVAL_COLUMNS)).sort_values(
        ["hole_id", "depth_from", "depth_to"], kind="stable"
    )
    for hole, lo, hi in zip(
        frame["hole_id"].to_numpy(dtype=object),
        frame["depth_from"].to_numpy(dtype=np.float64),
        frame["depth_to"].to_numpy(dtype=np.float64),
        strict=True,
    ):
        if hi < lo:
            continue
        runs = zones.setdefault(hole, [])
        if runs and lo <= runs[-1][1] + 1e-9:
            runs[-1] = (runs[-1][0], max(runs[-1][1], hi))
        else:
            runs.append((lo, hi))
    return zones


def fault_picks(
    table: pd.DataFrame,
    blocks: pd.Series,
    traces: Any,
    *,
    fault_zone: pd.DataFrame | None = None,
    min_uncertainty: float = 0.5,
    include_unlabelled: bool = True,
) -> pd.DataFrame:
    """Fault crossings along the holes (research.md 5.7.5).

    Where a hole passes from block ``a`` to block ``b`` the fault lies in the gap between the
    last ``a`` interval and the first ``b`` interval. The pick is the middle of a logged
    fault-zone interval that touches the gap (the one nearest to its middle), otherwise the
    middle of the gap; ``uncertainty`` is half the length of that interval or gap and
    ``weight = 1 / max(uncertainty, min_uncertainty)^2``.

    ``fault_zone`` holds the logged fault-zone intervals (``hole_id, depth_from, depth_to``).
    With ``include_unlabelled`` a fault-zone interval that belongs to no block change is a pick
    too (source ``"fault-zone"``): ``block_above`` / ``block_below`` are then the nearest known
    blocks up and down the hole, or None, and ``gap_from`` / ``gap_to`` the ends of those
    intervals, or NaN.

    Columns: ``hole_id, block_above, block_below, gap_from, gap_to, depth, x, y, z,
    uncertainty, weight, source``. Picks whose depth has no trace are left out.
    """
    if not (math.isfinite(min_uncertainty) and min_uncertainty > 0.0):
        raise FaultError("min_uncertainty must be a positive number")
    known = _known_intervals(table, blocks)
    zones = _merged_zones(fault_zone)
    rows: list[tuple] = []

    def add(hole, above, below, lo, hi, depth, half, source) -> None:
        rows.append((hole, above, below, lo, hi, depth, half, source))

    by_hole = {hole: part for hole, part in known.groupby("hole_id", sort=False)}
    for hole in sorted(set(by_hole) | set(zones), key=str):
        part = by_hole.get(hole)
        hole_zones = zones.get(hole, [])
        used = [False] * len(hole_zones)
        if part is not None:
            lo_all = part["depth_from"].to_numpy()
            hi_all = part["depth_to"].to_numpy()
            block = part["block"].to_numpy(dtype=object)
            for k in np.flatnonzero(block[1:] != block[:-1]):
                gap_from = float(min(hi_all[k], lo_all[k + 1]))
                gap_to = float(max(hi_all[k], lo_all[k + 1]))
                middle = 0.5 * (gap_from + gap_to)
                touching = [
                    i
                    for i, (lo, hi) in enumerate(hole_zones)
                    if lo <= gap_to + 1e-9 and hi >= gap_from - 1e-9
                ]
                if touching:
                    best = min(touching, key=lambda i: abs(0.5 * sum(hole_zones[i]) - middle))
                    for i in touching:
                        used[i] = True
                    lo, hi = hole_zones[best]
                    add(hole, block[k], block[k + 1], gap_from, gap_to, 0.5 * (lo + hi),
                        0.5 * (hi - lo), "fault-zone")  # fmt: skip
                else:
                    add(hole, block[k], block[k + 1], gap_from, gap_to, middle,
                        0.5 * (gap_to - gap_from), "gap-middle")  # fmt: skip
        if not include_unlabelled:
            continue
        for (lo, hi), taken in zip(hole_zones, used, strict=True):
            if taken:
                continue
            depth = 0.5 * (lo + hi)
            above = below = None
            gap_from = gap_to = math.nan
            if part is not None:
                mids = 0.5 * (part["depth_from"].to_numpy() + part["depth_to"].to_numpy())
                up = np.flatnonzero(mids <= depth)
                down = np.flatnonzero(mids > depth)
                if up.size:
                    above = part["block"].iloc[up[-1]]
                    gap_from = float(part["depth_to"].iloc[up[-1]])
                if down.size:
                    below = part["block"].iloc[down[0]]
                    gap_to = float(part["depth_from"].iloc[down[0]])
            add(hole, above, below, gap_from, gap_to, depth, 0.5 * (hi - lo), "fault-zone")

    if not rows:
        return pd.DataFrame({c: pd.Series(dtype=object) for c in PICK_COLUMNS}).astype(
            {
                c: np.float64
                for c in ("gap_from", "gap_to", "depth", "x", "y", "z", "uncertainty", "weight")
            }  # fmt: skip
        )
    hole = np.array([r[0] for r in rows], dtype=object)
    depth = np.array([r[5] for r in rows], dtype=np.float64)
    half = np.array([r[6] for r in rows], dtype=np.float64)
    xyz, _ = traces.xyz_at(hole, depth)
    picks = pd.DataFrame(
        {
            "hole_id": hole,
            "block_above": np.array([r[1] for r in rows], dtype=object),
            "block_below": np.array([r[2] for r in rows], dtype=object),
            "gap_from": np.array([r[3] for r in rows], dtype=np.float64),
            "gap_to": np.array([r[4] for r in rows], dtype=np.float64),
            "depth": depth,
            "x": xyz[:, 0],
            "y": xyz[:, 1],
            "z": xyz[:, 2],
            "uncertainty": half,
            "weight": 1.0 / np.maximum(half, min_uncertainty) ** 2,
            "source": np.array([r[7] for r in rows], dtype=object),
        }
    )
    keep = np.isfinite(xyz).all(axis=1)
    return picks[keep].reset_index(drop=True)


# --------------------------------------------------------------------------
# plane
# --------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class Plane:
    """A plane through ``point`` with unit ``normal`` turned up.

    The normal given is normalised and, if it points down, reversed, so ``n_z >= 0``. A
    vertical plane has its normal turned to an azimuth in ``[0, 180)``.
    """

    point: np.ndarray
    normal: np.ndarray

    def __post_init__(self) -> None:
        point = np.array(self.point, dtype=np.float64).reshape(-1)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise FaultError("a plane point must be three finite numbers")
        normal = _oriented(self.normal)
        point.flags.writeable = False
        normal.flags.writeable = False
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "normal", normal)

    @property
    def dip(self) -> float:
        """Angle from the horizontal in degrees: 0 (flat) to 90 (vertical)."""
        return math.degrees(math.acos(min(1.0, max(-1.0, float(self.normal[2])))))

    @property
    def dip_azimuth(self) -> float:
        """Azimuth of the horizontal part of the upward normal, degrees clockwise from north
        in ``[0, 360)``: the direction the plane dips towards. 0 for a horizontal plane."""
        nx, ny = float(self.normal[0]), float(self.normal[1])
        if math.hypot(nx, ny) < _VERTICAL:
            return 0.0
        return math.degrees(math.atan2(nx, ny)) % 360.0

    @classmethod
    def from_dip(cls, point: Any, dip: float, dip_azimuth: float) -> Plane:
        """Plane through ``point`` that dips ``dip`` degrees towards ``dip_azimuth``."""
        return cls(point, _upward_normal(dip, dip_azimuth))

    def signed_distance(self, points: Any) -> np.ndarray:
        """``n . (p - point)`` for points ``(..., 3)``: positive on the upper side."""
        p = np.asarray(points, dtype=np.float64)
        if p.shape[-1:] != (3,):
            raise FaultError(f"points must have shape (..., 3), got {p.shape}")
        return (p - self.point) @ self.normal

    def to_dict(self) -> dict[str, Any]:
        return {
            "point": [float(v) for v in self.point],
            "normal": [float(v) for v in self.normal],
            "dip": self.dip,
            "dip_azimuth": self.dip_azimuth,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Plane:
        return cls(data["point"], data["normal"])


def _as_normal(prior: Any) -> np.ndarray:
    """A prior orientation as an upward normal: a Plane, ``(dip, dip_azimuth)`` or a vector."""
    if isinstance(prior, Plane):
        return prior.normal
    values = np.asarray(prior, dtype=np.float64).reshape(-1)
    if values.shape == (2,):
        return _oriented(_upward_normal(values[0], values[1]))
    return _oriented(values)


def _principal(points: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, Any]:
    """Weighted centroid, eigenvalues (ascending) and eigenvectors of the scatter matrix."""
    shift = points.mean(axis=0)
    q = points - shift
    total = weights.sum()
    centre = (weights[:, None] * q).sum(axis=0) / total
    r = q - centre
    scatter = (weights[:, None] * r).T @ r / total
    values, vectors = np.linalg.eigh(scatter)
    return centre + shift, np.maximum(values, 0.0), vectors


def _on_one_line(values: np.ndarray) -> bool:
    return values[2] <= 0.0 or math.sqrt(values[1] / values[2]) < _COLLINEAR


def fit_plane(
    points: Any,
    weights: Any = None,
    *,
    prior: Plane | tuple[float, float] | None = None,
) -> Plane:
    """Weighted least-squares plane through points (orthogonal distances).

    The plane passes through the weighted centroid; its normal is the eigenvector of the
    weighted scatter matrix with the smallest eigenvalue. Needs three points that are not on
    one line, else :class:`FaultError`.

    With ``prior`` (a :class:`Plane` or ``(dip, dip_azimuth)``) the orientation is taken from
    it and only the position is fitted; one point is then enough.
    """
    p = _xyz(points, "points")
    if weights is None:
        w = np.ones(len(p))
    else:
        w = np.asarray(weights, dtype=np.float64).reshape(-1)
        if w.shape != (len(p),):
            raise FaultError("weights must have one value per point")
        if not (np.isfinite(w).all() and (w > 0.0).all()):
            raise FaultError("weights must be positive finite numbers")
    if prior is not None:
        centre, _, _ = _principal(p, w)
        return Plane(centre, _as_normal(prior))
    if len(p) < 3:
        raise FaultError(
            f"a plane needs at least 3 points, got {len(p)}; "
            "give a prior orientation or use the separating plane"
        )
    centre, values, vectors = _principal(p, w)
    if _on_one_line(values):
        raise FaultError(
            "the points lie on one line (or coincide): they do not fix a plane; "
            "give a prior orientation or use the separating plane"
        )
    return Plane(centre, vectors[:, 0])


# --------------------------------------------------------------------------
# separating plane
# --------------------------------------------------------------------------


def _svm(x: np.ndarray, y: np.ndarray, penalty: float) -> tuple[np.ndarray, float, int, bool]:
    """Soft-margin linear SVM by sequential minimal optimisation on the dual.

    minimise ``|w|^2 / 2 + penalty * sum(max(0, 1 - y (w . x + beta)))``. Two multipliers
    move at a time (the most violating one and the partner with the largest second-order
    gain, Fan, Chen & Lin 2005), which keeps ``sum(alpha y) = 0``; ``w = sum(alpha y x)`` is
    carried along, so one step costs O(n).
    """
    n = len(y)
    alpha = np.zeros(n)
    w = np.zeros(x.shape[1])
    positive = y > 0
    upper = lower = 0.0
    max_iter = max(20000, 400 * n)
    converged = False
    iteration = 0
    for iteration in range(max_iter):  # noqa: B007
        u = y - x @ w
        below_cap, above_zero = alpha < penalty, alpha > 0.0
        can_rise = np.where(positive, below_cap, above_zero)
        can_fall = np.where(positive, above_zero, below_cap)
        i = int(np.argmax(np.where(can_rise, u, -np.inf)))
        upper = float(u[i])
        lower = float(np.min(np.where(can_fall, u, np.inf)))
        if upper - lower < _SVM_TOLERANCE:
            converged = True
            break
        d = x[i] - x
        curvature = np.maximum(np.einsum("ij,ij->i", d, d), 1e-300)
        gain = np.where(can_fall & (u < upper), (upper - u) ** 2 / curvature, -np.inf)
        j = int(np.argmax(gain))
        room_i = penalty - alpha[i] if positive[i] else alpha[i]
        room_j = alpha[j] if positive[j] else penalty - alpha[j]
        step = (upper - u[j]) / curvature[j]
        if step >= room_i:
            step = room_i
            alpha[i] = penalty if positive[i] else 0.0
        else:
            alpha[i] += y[i] * step
        if step >= room_j:
            # the pair moves together: i takes the smaller step as well
            if step > room_j:
                alpha[i] -= y[i] * (step - room_j)
                step = room_j
            alpha[j] = 0.0 if positive[j] else penalty
        else:
            alpha[j] -= y[j] * step
        w = w + step * (x[i] - x[j])
    # w from the multipliers, free of the rounding that the running sum collects
    w = (alpha * y) @ x
    u = y - x @ w
    can_rise = np.where(positive, alpha < penalty, alpha > 0.0)
    can_fall = np.where(positive, alpha > 0.0, alpha < penalty)
    upper = float(np.max(np.where(can_rise, u, -np.inf)))
    lower = float(np.min(np.where(can_fall, u, np.inf)))
    return w, 0.5 * (upper + lower), iteration, converged


def _best_offset(t_neg: np.ndarray, t_pos: np.ndarray) -> tuple[float, int, float] | None:
    """Offset along a fixed normal with the fewest points on the wrong side.

    Returns ``(offset, n_wrong, width of the range of equally good offsets)``; the offset is
    the middle of that range (the widest one when several tie). None when all projections
    are equal.
    """
    values = np.unique(np.concatenate([t_neg, t_pos]))
    if len(values) < 2:
        return None
    neg, pos = np.sort(t_neg), np.sort(t_pos)
    # threshold between values[k] and values[k + 1]
    wrong = (len(neg) - np.searchsorted(neg, values[:-1], side="right")) + np.searchsorted(
        pos, values[:-1], side="right"
    )
    width = np.diff(values)
    best = wrong == wrong.min()
    k = int(np.argmax(np.where(best, width, -np.inf)))
    return 0.5 * (values[k] + values[k + 1]), int(wrong[k]), float(width[k])


def separating_plane(
    points_neg: Any,
    points_pos: Any,
    *,
    prior_normal: Any = None,
    penalty: float = 10.0,
) -> tuple[Plane, dict]:
    """The plane that separates two point clouds (research.md 5.7.5).

    Without a prior this is the soft-margin linear support vector machine

        minimise  |w|^2 / 2 + penalty * sum(xi)
        subject to  y_i (w . x_i + beta) >= 1 - xi_i,  xi_i >= 0

    with ``y = -1`` for ``points_neg``, ``+1`` for ``points_pos`` and ``x = (p - c) / L``
    (``c`` the mean of all points, ``L`` the largest side of their bounding box).

    With ``prior_normal`` (a vector, a :class:`Plane` or ``(dip, dip_azimuth)``) only the
    offset is free: the middle of the gap between the two clouds along the normal when they
    separate, otherwise the middle of the range of offsets with the fewest points on the
    wrong side. Both directions of the normal are tried.

    The plane's normal always points up, so ``points_pos`` may end on its negative side: the
    second value says so. Its keys: ``method`` (``"soft-margin"`` / ``"prior-normal"``),
    ``flipped`` (True when ``points_pos`` are on the negative side of the returned plane),
    ``n_wrong``, ``wrong_neg`` / ``wrong_pos`` (indices of the points on the wrong side),
    ``separable``, ``margin`` (metres from the plane to the nearest margin, or half the gap),
    and for the soft margin ``w``, ``beta``, ``centre``, ``scale``, ``objective``,
    ``iterations``, ``converged``.
    """
    a = _xyz(points_neg, "points_neg", allow_empty=True)
    b = _xyz(points_pos, "points_pos", allow_empty=True)
    if len(a) == 0 or len(b) == 0:
        raise FaultError(
            "a separating plane needs points of two fault blocks; "
            f"got {len(a)} and {len(b)} (one block only)"
        )
    if not (math.isfinite(penalty) and penalty > 0.0):
        raise FaultError("penalty must be a positive number")
    both = np.concatenate([a, b])
    centre = both.mean(axis=0)
    scale = float(np.ptp(both, axis=0).max())
    if scale <= 0.0:
        raise FaultError("all points of the two fault blocks coincide: no separating plane")

    if prior_normal is not None:
        n0 = _as_normal(prior_normal)
        t_a, t_b = (a - centre) @ n0, (b - centre) @ n0
        options = []
        for sign in (1.0, -1.0):
            found = _best_offset(sign * t_a, sign * t_b)
            if found is not None:
                options.append((found[1], -found[2], -sign, sign, found[0]))
        if not options:
            raise FaultError(
                "all points have the same position along the prior normal: no separating plane"
            )
        n_wrong, neg_width, _, sign, offset = min(options)
        normal = sign * n0  # points from the negative cloud to the positive cloud
        point = centre + offset * normal
        info: dict[str, Any] = {"method": "prior-normal", "margin": 0.5 * -neg_width}
    else:
        x = (both - centre) / scale
        y = np.concatenate([-np.ones(len(a)), np.ones(len(b))])
        w, beta, iterations, converged = _svm(x, y, float(penalty))
        length = float(np.linalg.norm(w))
        if length < 1e-9:
            raise FaultError(
                "the two fault blocks are not separated by any plane (their points are mixed)"
            )
        normal = w / length
        point = centre - (beta / length) * scale * normal
        slack = np.maximum(0.0, 1.0 - y * (x @ w + beta))
        info = {
            "method": "soft-margin",
            "margin": scale / length,
            "w": [float(v) for v in w],
            "beta": float(beta),
            "centre": [float(v) for v in centre],
            "scale": scale,
            "objective": float(0.5 * length**2 + penalty * slack.sum()),
            "iterations": int(iterations),
            "converged": bool(converged),
        }
    d_a, d_b = (a - point) @ normal, (b - point) @ normal
    wrong_neg, wrong_pos = np.flatnonzero(d_a > 0.0), np.flatnonzero(d_b <= 0.0)
    plane = Plane(point, normal)
    info.update(
        flipped=bool(plane.normal @ normal < 0.0),
        n_wrong=int(len(wrong_neg) + len(wrong_pos)),
        wrong_neg=[int(i) for i in wrong_neg],
        wrong_pos=[int(i) for i in wrong_pos],
        separable=bool(len(wrong_neg) + len(wrong_pos) == 0),
    )
    return plane, info


# --------------------------------------------------------------------------
# fault surface
# --------------------------------------------------------------------------


def _frame(normal: np.ndarray) -> np.ndarray:
    """Rows ``x', y', z'`` of a right-handed frame with ``z'`` along ``normal``."""
    across = np.cross([0.0, 0.0, 1.0], normal)
    if np.linalg.norm(across) < 1e-9:
        across = np.array([1.0, 0.0, 0.0])
    across = across / np.linalg.norm(across)
    return np.array([across, np.cross(normal, across), normal])


def _clip(polygon: np.ndarray, axis: int, bound: float, keep_below: bool) -> np.ndarray:
    """Convex polygon ``(m, 3)`` cut by ``p[axis] <= bound`` (or ``>=``)."""
    if len(polygon) == 0:
        return polygon
    side = (bound - polygon[:, axis]) if keep_below else (polygon[:, axis] - bound)
    out: list[np.ndarray] = []
    for k in range(len(polygon)):
        p, q = polygon[k], polygon[(k + 1) % len(polygon)]
        sp, sq = side[k], side[(k + 1) % len(polygon)]
        if sp >= 0.0:
            out.append(p)
        if (sp > 0.0 and sq < 0.0) or (sp < 0.0 and sq > 0.0):
            out.append(p + (q - p) * (sp / (sp - sq)))
    return np.array(out).reshape(-1, 3)


def _clip_box(polygon: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    for axis in range(3):
        polygon = _clip(polygon, axis, hi[axis], True)
        polygon = _clip(polygon, axis, lo[axis], False)
    return polygon


class FaultSurface:
    """One fault: a plane, or an imported mesh with its best-fit plane.

    ``blocks`` is ``(block on the negative side, block on the positive side)`` when known.
    ``misfit`` holds what the surface says about its own fit (``n_wrong``, ``holes_wrong``,
    ``pick_rms`` ...). ``provenance`` are its lines of the provenance record.
    """

    def __init__(
        self,
        *,
        name: str,
        source: str,
        plane: Plane,
        blocks: tuple[str, str] | None = None,
        mesh: Mesh | None = None,
        misfit: Mapping[str, Any] | None = None,
        provenance: Iterable[ProvenanceEntry] = (),
    ) -> None:
        if not isinstance(name, str) or not name.strip():
            raise FaultError("a fault needs a name")
        if source not in FAULT_SOURCES:
            raise FaultError(
                f"fault {name!r}: source {source!r} is not one of {', '.join(FAULT_SOURCES)}"
            )
        if (mesh is not None) != (source == "imported-mesh"):
            raise FaultError(f"fault {name!r}: a mesh goes with source 'imported-mesh' only")
        if blocks is not None:
            blocks = (str(blocks[0]), str(blocks[1]))
            if len(set(blocks)) != 2:
                raise FaultError(f"fault {name!r}: the two blocks must differ")
        self.name = name
        self.source = source
        self.plane = plane
        self.blocks = blocks
        self.mesh = mesh
        self.misfit: dict = dict(misfit or {})
        self.provenance: list[ProvenanceEntry] = list(provenance)
        self._rotation: np.ndarray | None = None
        self._dtm: DTM | None = None
        if mesh is not None:
            self._rotation, self._dtm = _mesh_frame(name, mesh, plane)

    def __repr__(self) -> str:
        return (
            f"FaultSurface({self.name!r}, {self.source}, dip {self.plane.dip:.1f}, "
            f"dip azimuth {self.plane.dip_azimuth:.1f}, blocks {self.blocks})"
        )

    @property
    def inferred(self) -> bool:
        return self.source.startswith("inferred")

    def offset(self, points: Any) -> np.ndarray:
        """Signed offset of points ``(..., 3)`` from the fault, positive on the upper side.

        A plane: the signed distance. A mesh: the height above the surface in the mesh's own
        frame (``z'`` along its mean normal); outside the footprint of the mesh in that
        frame, the signed distance to its best-fit plane.
        """
        p = np.asarray(points, dtype=np.float64)
        out = self.plane.signed_distance(p)
        if self._dtm is None:
            return out
        local = (p - self.plane.point) @ self._rotation.T
        height = self._dtm.height_above(local)
        return np.where(np.isnan(height), out, height)

    def side(self, points: Any) -> np.ndarray:
        """``+1`` where :meth:`offset` is positive, else ``-1`` (int8)."""
        return np.where(self.offset(points) > 0.0, 1, -1).astype(np.int8)

    def display_mesh(self, lo: Any, hi: Any, *, spacing: float | None = None) -> Mesh:
        """The surface inside the box ``lo .. hi`` for display.

        An imported mesh is returned as it is. A plane is clipped to the box: one polygon as
        a triangle fan, or with ``spacing`` a grid of that cell size in the plane.
        """
        if self.mesh is not None:
            return self.mesh
        lo = np.asarray(lo, dtype=np.float64).reshape(3)
        hi = np.asarray(hi, dtype=np.float64).reshape(3)
        if not (hi > lo).all():
            raise FaultError("display box: hi must be greater than lo on every axis")
        middle = 0.5 * (lo + hi)
        radius = 0.5 * float(np.linalg.norm(hi - lo))
        distance = float(self.plane.signed_distance(middle))
        if abs(distance) > radius:
            raise FaultError(f"fault {self.name!r} does not cross the display box")
        across, updip, normal = _frame(self.plane.normal)
        origin = middle - distance * normal
        if spacing is None:
            edges = np.array([-radius, radius])
        else:
            if not (math.isfinite(spacing) and spacing > 0.0):
                raise FaultError("spacing must be a positive number")
            n = max(1, math.ceil(2.0 * radius / spacing))
            edges = (np.arange(n + 1) - 0.5 * n) * spacing
        polygons: list[np.ndarray] = []
        for i, j in itertools.product(range(len(edges) - 1), repeat=2):
            corners = np.array(
                [
                    origin + edges[i + a] * across + edges[j + b] * updip
                    for a, b in ((0, 0), (1, 0), (1, 1), (0, 1))
                ]
            )
            if (corners < lo).all(axis=0).any() or (corners > hi).all(axis=0).any():
                continue
            if not ((corners >= lo).all() and (corners <= hi).all()):
                corners = _clip_box(corners, lo, hi)
            if len(corners) >= 3:
                polygons.append(corners)
        vertices: list[np.ndarray] = []
        faces: list[tuple[int, int, int]] = []
        for polygon in polygons:
            base = sum(len(v) for v in vertices)
            vertices.append(polygon)
            faces.extend((base, base + k, base + k + 1) for k in range(1, len(polygon) - 1))
        if not faces:
            raise FaultError(f"fault {self.name!r} does not cross the display box")
        mesh = Mesh(np.concatenate(vertices), np.array(faces))
        return mesh.weld(1e-6 * max(radius, 1.0)).remove_unreferenced_vertices()

    # ------------------------------------------------------------------ files

    def to_dict(self) -> dict:
        """JSON content of ``faults/<name>.json``. An imported mesh is named, not included:
        ``mesh`` is ``<name>_source.obj`` beside the JSON file (:meth:`save` writes both)."""
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "source": self.source,
            "plane": self.plane.to_dict(),
            "blocks": list(self.blocks) if self.blocks else None,
            "mesh": f"{self.name}_source.obj" if self.mesh is not None else None,
            "misfit": _jsonable(self.misfit),
            "provenance": [entry.to_dict() for entry in self.provenance],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], folder: str | Path | None = None) -> FaultSurface:
        """Inverse of :meth:`to_dict`; ``folder`` holds the mesh file the record names."""
        if not isinstance(data, Mapping) or data.get("format") != FORMAT:
            raise FaultError(f"not a fault record (format {FORMAT!r} expected)")
        if data.get("version") != VERSION:
            raise FaultError(f"fault record version {data.get('version')!r} is not supported")
        mesh = None
        if data.get("mesh"):
            if folder is None:
                raise FaultError(f"fault {data.get('name')!r}: a folder is needed for its mesh")
            mesh = read_obj(Path(folder) / data["mesh"])
        blocks = data.get("blocks")
        return cls(
            name=data["name"],
            source=data["source"],
            plane=Plane.from_dict(data["plane"]),
            blocks=tuple(blocks) if blocks else None,
            mesh=mesh,
            misfit=data.get("misfit") or {},
            provenance=[ProvenanceEntry.from_dict(e) for e in data.get("provenance", [])],
        )

    def save(self, folder: str | Path) -> Path:
        """Write ``<name>.json`` (and ``<name>_source.obj`` for an imported mesh) in ``folder``."""
        folder = Path(folder)
        folder.mkdir(parents=True, exist_ok=True)
        record = self.to_dict()
        if self.mesh is not None:
            write_obj(folder / record["mesh"], self.mesh, decimals=None)
        path = folder / f"{self.name}.json"
        path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> FaultSurface:
        path = Path(path)
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")), path.parent)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple | np.ndarray):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _mesh_plane(name: str, mesh: Mesh) -> Plane:
    """Best-fit plane of a mesh: its vertices weighted by the area of their faces."""
    if mesh.n_faces == 0:
        raise FaultError(f"fault {name!r}: the mesh has no faces")
    areas = mesh.face_areas
    if not areas.sum() > 0.0:
        raise FaultError(f"fault {name!r}: the mesh has no area")
    weight = np.zeros(mesh.n_vertices)
    np.add.at(weight, mesh.faces.ravel(), np.repeat(areas / 3.0, 3))
    used = weight > 0.0
    try:
        return fit_plane(mesh.vertices[used], weight[used])
    except FaultError as exc:
        raise FaultError(f"fault {name!r}: the mesh does not fix a plane ({exc})") from None


def _folds(mesh: Mesh, direction: np.ndarray) -> bool:
    """True when two faces that share an edge face opposite ways along ``direction``.

    The winding of the faces does not matter: two neighbours are compared after one of them
    is reversed when they run along their common edge in the same direction. Faces that
    share no vertex index with a neighbour (a triangle soup) are welded first.
    """
    extent = float(np.ptp(mesh.vertices, axis=0).max()) if mesh.n_vertices else 0.0
    welded = mesh.weld(1e-9 * max(extent, 1.0))
    if welded.n_faces == 0:
        return False
    tri = welded.triangles
    along = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]) @ direction
    edges = welded.edge_map
    pairs = edges.manifold_halfedge_pairs
    forward = edges.forward.reshape(-1)
    same_way = forward[pairs[:, 0]] == forward[pairs[:, 1]]
    product = along[pairs[:, 0] // 3] * along[pairs[:, 1] // 3]
    return bool((np.where(same_way, -product, product) < 0.0).any())


def _mesh_frame(name: str, mesh: Mesh, plane: Plane) -> tuple[np.ndarray, DTM]:
    """Rotation into the mesh's own frame (``z'`` = mean normal) and the surface there."""
    tri = mesh.triangles
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    cross = cross * np.where(cross @ plane.normal < 0.0, -1.0, 1.0)[:, None]
    mean = cross.sum(axis=0)
    if np.linalg.norm(mean) == 0.0:
        raise FaultError(f"fault {name!r}: the mesh has no area")
    mean = mean / np.linalg.norm(mean)
    rotation = _frame(mean)
    if _folds(mesh, mean):
        raise FaultError(
            f"fault {name!r}: the mesh folds over itself seen along its mean normal; "
            "it cannot be used as one fault surface"
        )
    area2 = np.linalg.norm(cross, axis=1)
    keep = area2 > 1e-12 * area2.max()
    local = Mesh((mesh.vertices - plane.point) @ rotation.T, mesh.faces[keep])
    try:
        return rotation, DTM(local)
    except ValueError as exc:
        raise FaultError(
            f"fault {name!r}: the mesh folds over itself seen along its mean normal ({exc}); "
            "it cannot be used as one fault surface"
        ) from None


# --------------------------------------------------------------------------
# inferred and imported faults
# --------------------------------------------------------------------------


def _usable_picks(picks: pd.DataFrame | None, blocks: tuple[str, str]) -> pd.DataFrame:
    """Picks that can belong to the fault between ``blocks``.

    A pick is kept when the blocks it names are among the two and it does not have the same
    block on both sides. A fault-zone pick that names no block is kept.
    """
    if picks is None or len(picks) == 0:
        return pd.DataFrame({c: [] for c in PICK_COLUMNS})
    missing = [c for c in ("x", "y", "z") if c not in picks.columns]
    if missing:
        raise FaultError(f"picks lack column(s) {', '.join(missing)}")
    keep = np.isfinite(picks[["x", "y", "z"]].to_numpy(dtype=np.float64)).all(axis=1)
    pair = set(blocks)
    if "block_above" in picks.columns and "block_below" in picks.columns:
        above = picks["block_above"].to_numpy(dtype=object)
        below = picks["block_below"].to_numpy(dtype=object)
        for k in range(len(picks)):
            named = [str(v) for v in (above[k], below[k]) if not pd.isna(v)]
            if any(v not in pair for v in named) or (len(named) == 2 and named[0] == named[1]):
                keep[k] = False
    return picks[keep].reset_index(drop=True)


def _hole_counts(holes: Any, index: np.ndarray, n: int) -> dict[str, int]:
    if holes is None or len(index) == 0:
        return {}
    holes = np.asarray(holes, dtype=object)
    if holes.shape != (n,):
        raise FaultError("hole ids must have one value per labelled point")
    names, counts = np.unique(holes[index].astype(str), return_counts=True)
    return {str(k): int(v) for k, v in zip(names, counts, strict=True)}


def infer_fault(
    name: str,
    picks: pd.DataFrame | None,
    points_a: Any,
    points_b: Any,
    *,
    blocks: tuple[str, str],
    prior: tuple[float, float] | None = None,
    min_picks: int = 3,
    holes_a: Any = None,
    holes_b: Any = None,
) -> FaultSurface:
    """Infer the fault between two fault blocks from the logs.

    ``picks`` are the crossings (:func:`fault_picks`); ``points_a`` / ``points_b`` the
    labelled interval midpoints of ``blocks[0]`` / ``blocks[1]`` (:func:`labelled_points`).

    - At least ``min_picks`` crossings that are not on one line: weighted plane through them
      (source ``"inferred-plane"``).
    - Otherwise the separating plane of the two point clouds (``"inferred-separating"``).
    - With ``prior = (dip, dip_azimuth)`` the orientation is the user's: the position comes
      from the crossings when there is at least one, else from the two clouds.

    The result's ``blocks`` are ordered (negative side, positive side) by the labelled points,
    and ``misfit`` counts the labelled points on the wrong side (``holes_wrong`` by hole when
    ``holes_a`` / ``holes_b`` are given). Raises :class:`FaultError` when the evidence cannot
    give a fault: no crossings and one block only, or no labelled point to tell the sides.
    """
    blocks = (str(blocks[0]), str(blocks[1]))
    if blocks[0] == blocks[1]:
        raise FaultError(f"fault {name!r}: the two blocks must differ")
    a = _xyz(points_a, "points_a", allow_empty=True)
    b = _xyz(points_b, "points_b", allow_empty=True)
    usable = _usable_picks(picks, blocks)
    xyz = usable[["x", "y", "z"]].to_numpy(dtype=np.float64) if len(usable) else np.zeros((0, 3))
    weights = (
        usable["weight"].to_numpy(dtype=np.float64)
        if len(usable) and "weight" in usable.columns
        else np.ones(len(xyz))
    )
    if len(a) + len(b) == 0:
        raise FaultError(
            f"fault {name!r}: no interval carries block {blocks[0]!r} or {blocks[1]!r}, "
            "so the sides of the fault cannot be told"
        )

    info: dict[str, Any] = {}
    enough = len(xyz) >= max(int(min_picks), 3) and not _on_one_line(_principal(xyz, weights)[1])
    if prior is not None and len(xyz) >= 1:
        plane, method = fit_plane(xyz, weights, prior=prior), "inferred-plane"
    elif prior is None and enough:
        plane, method = fit_plane(xyz, weights), "inferred-plane"
    else:
        try:
            plane, info = separating_plane(
                a, b, prior_normal=None if prior is None else _as_normal(prior)
            )
        except FaultError as exc:
            reason = (
                f"{_plural(len(xyz), 'usable crossing')} (at least {max(int(min_picks), 3)} "
                "not on one line are needed for a plane fit)"
            )
            raise FaultError(f"fault {name!r}: {reason}, and {exc}") from None
        method = "inferred-separating"

    d_a, d_b = plane.signed_distance(a), plane.signed_distance(b)
    wrong_ab = int((d_a > 0.0).sum() + (d_b <= 0.0).sum())  # a below, b above
    wrong_ba = int((d_a <= 0.0).sum() + (d_b > 0.0).sum())
    a_negative = wrong_ab <= wrong_ba
    wrong_a = np.flatnonzero(d_a > 0.0 if a_negative else d_a <= 0.0)
    wrong_b = np.flatnonzero(d_b <= 0.0 if a_negative else d_b > 0.0)
    holes_wrong: dict[str, int] = {}
    for part in (_hole_counts(holes_a, wrong_a, len(a)), _hole_counts(holes_b, wrong_b, len(b))):
        for hole, count in part.items():
            holes_wrong[hole] = holes_wrong.get(hole, 0) + count
    n_wrong = len(wrong_a) + len(wrong_b)
    misfit: dict[str, Any] = {
        "method": "plane fit" if method == "inferred-plane" else "separating plane",
        "n_points": int(len(a) + len(b)),
        "n_wrong": int(n_wrong),
        "wrong_by_block": {blocks[0]: int(len(wrong_a)), blocks[1]: int(len(wrong_b))},
        "holes_wrong": holes_wrong,
        "n_picks": int(len(xyz)),
        "prior": None if prior is None else [float(prior[0]), float(prior[1])],
    }
    if len(xyz):
        residual = plane.signed_distance(xyz)
        misfit["pick_rms"] = float(math.sqrt((weights * residual**2).sum() / weights.sum()))
        misfit["pick_max"] = float(np.abs(residual).max())
        if "source" in usable.columns:
            kinds = usable["source"].astype(str).value_counts()
            misfit["picks_by_source"] = {str(k): int(v) for k, v in kinds.items()}
    if info:
        misfit["margin"] = float(info["margin"])
        if "converged" in info:
            misfit["converged"] = bool(info["converged"])

    ordered = blocks if a_negative else (blocks[1], blocks[0])
    used_for = f"fault blocks {blocks[0]} / {blocks[1]}"
    wrong_text = f"{_plural(n_wrong, 'point')} on the wrong side"
    if holes_wrong:
        wrong_text += f" in {_plural(len(holes_wrong), 'hole')}"
    entries: list[ProvenanceEntry] = []
    if method == "inferred-plane":
        source = f"inferred from {_plural(len(xyz), 'fault crossing')} along holes (plane fit)"
        detail = (
            f"{_orientation_text(plane)}; {wrong_text}; crossings RMS {misfit['pick_rms']:.1f} m"
        )
        by_source = misfit.get("picks_by_source", {})
        entries.append(
            ProvenanceEntry(
                item=f"Fault picks of {name}",
                kind="measured",
                source="block changes and logged fault-zone intervals along holes",
                used_for=f"position of fault {name}",
                detail=(
                    f"{by_source.get('fault-zone', 0)} from fault-zone intervals, "
                    f"{by_source.get('gap-middle', 0)} from the middle of a gap between blocks"
                    if by_source
                    else ""
                ),
                stage=_STAGE,
                task=_TASK,
                count=int(len(xyz)),
            )
        )
    else:
        source = (
            f"inferred from the fault-block codes of {_plural(len(a) + len(b), 'interval')} "
            "(separating plane)"
        )
        detail = f"{_orientation_text(plane)}; {wrong_text}"
    if prior is not None:
        source += ", orientation given by the user"
        entries.append(
            ProvenanceEntry(
                item=f"Prior orientation of fault {name}",
                kind="assumed",
                source="given by the user",
                used_for=f"orientation of fault {name} (only its position is fitted)",
                detail=f"dip {float(prior[0]):g}, dip azimuth {float(prior[1]):g}",
                stage=_STAGE,
                task=_TASK,
            )
        )
    entries.insert(
        0,
        ProvenanceEntry(
            item=f"Fault {name}",
            kind="assumed",
            source=source,
            used_for=used_for,
            detail=detail,
            stage=_STAGE,
            task=_TASK,
            count=int(n_wrong),
        ),
    )
    return FaultSurface(
        name=name,
        source=method,
        plane=plane,
        blocks=ordered,
        misfit=misfit,
        provenance=entries,
    )


def import_fault(
    name: str,
    *,
    mesh: Mesh | None = None,
    points: Any = None,
    plane: Plane | None = None,
    blocks: tuple[str, str] | None = None,
) -> FaultSurface:
    """A fault from the geologist: exactly one of a mesh, points, or a plane.

    A mesh keeps its shape inside its footprint and continues as its best-fit plane outside.
    Points give their least-squares plane. ``blocks`` is (negative side, positive side) when
    the geologist names them; otherwise :meth:`FaultNetwork.fit` finds the blocks from the
    labelled points.
    """
    given = [v is not None for v in (mesh, points, plane)]
    if sum(given) != 1:
        raise FaultError(f"fault {name!r}: give exactly one of mesh, points or plane")
    misfit: dict[str, Any] = {}
    if mesh is not None:
        if not isinstance(mesh, Mesh):
            raise FaultError(f"fault {name!r}: mesh must be a Mesh")
        fitted = _mesh_plane(name, mesh)
        source = "imported-mesh"
        residual = fitted.signed_distance(mesh.vertices[np.unique(mesh.faces)])
        misfit = {
            "n_faces": int(mesh.n_faces),
            "area": float(mesh.area),
            "plane_rms": float(math.sqrt((residual**2).mean())),
            "plane_max": float(np.abs(residual).max()),
        }
        described = (
            f"imported mesh ({_plural(mesh.n_faces, 'triangle')}, {mesh.area:.0f} m2); "
            "outside its footprint its best-fit plane is used"
        )
    elif points is not None:
        p = _xyz(points, "points")
        fitted = fit_plane(p)
        source = "imported-points"
        residual = fitted.signed_distance(p)
        misfit = {
            "n_points": int(len(p)),
            "plane_rms": float(math.sqrt((residual**2).mean())),
            "plane_max": float(np.abs(residual).max()),
        }
        described = (
            f"imported points ({len(p)}), least-squares plane, "
            f"RMS distance {misfit['plane_rms']:.2f} m"
        )
    else:
        if not isinstance(plane, Plane):
            raise FaultError(f"fault {name!r}: plane must be a Plane")
        fitted = plane
        source = "imported-plane"
        described = "plane through a point with dip and dip azimuth, as given"
    entry = ProvenanceEntry(
        item=f"Fault {name}",
        kind="measured",
        source=f"the geologist's fault interpretation: {described}",
        used_for="fault blocks" + (f" {blocks[0]} / {blocks[1]}" if blocks else ""),
        detail=_orientation_text(fitted),
        stage=_STAGE,
        task=_TASK,
    )
    return FaultSurface(
        name=name, source=source, plane=fitted, blocks=blocks, mesh=mesh, misfit=misfit,
        provenance=[entry],
    )  # fmt: skip


def faults_provenance(faults: Iterable[FaultSurface]) -> list[ProvenanceEntry]:
    """The provenance lines of a set of faults.

    Every fault's own lines, then, while any fault is inferred, the *missing* line for the
    geologist's fault surfaces.
    """
    faults = list(faults)
    entries = [entry for fault in faults for entry in fault.provenance]
    inferred = [fault.name for fault in faults if fault.inferred]
    if inferred:
        entries.append(
            ProvenanceEntry(
                item="Fault surfaces",
                kind="missing",
                source="the geologist's fault interpretation (DXF / OBJ / points)",
                used_for="would replace the inferred planes",
                detail=f"inferred: {', '.join(inferred)}",
                stage=_STAGE,
                task=_TASK,
                count=len(inferred),
            )
        )
    return entries


# --------------------------------------------------------------------------
# fault network
# --------------------------------------------------------------------------


def _cell_text(cell: Sequence[int]) -> str:
    return ",".join(f"{int(s):+d}" if s else "0" for s in cell)


class FaultNetwork:
    """Several faults and the fault block of every cell they cut space into.

    ``cells`` maps a sign vector (one entry per fault: ``+1`` / ``-1`` side, ``0`` where the
    fault does not exist) to a block. ``terminations[f] = (g, s)`` says that fault ``f`` exists
    only on side ``s`` of fault ``g``; elsewhere its sign is 0 and the cells on either side of
    it are one cell. Where ``g`` itself does not exist, ``f`` does not exist either.
    """

    def __init__(
        self,
        faults: Iterable[FaultSurface],
        blocks: Iterable[str],
        cells: Mapping[tuple[int, ...], str],
        terminations: Mapping[str, tuple[str, int]] | None = None,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        self.faults: tuple[FaultSurface, ...] = tuple(faults)
        names = [fault.name for fault in self.faults]
        if len(set(names)) != len(names):
            raise FaultError("fault names must be unique in a network")
        if len(names) > _MAX_FAULTS:
            raise FaultError(f"a network holds at most {_MAX_FAULTS} faults, got {len(names)}")
        self.blocks: tuple[str, ...] = tuple(str(b) for b in blocks)
        self.terminations: dict[str, tuple[str, int]] = {}
        for name, (other, side) in (terminations or {}).items():
            if name not in names or other not in names or name == other:
                raise FaultError(
                    f"termination {name!r} on {other!r}: both must be different faults "
                    "of the network"
                )
            if side not in (1, -1):
                raise FaultError(f"termination of {name!r}: the side must be +1 or -1")
            self.terminations[name] = (other, int(side))
        self._order = self._termination_order(names)
        self.cells: dict[tuple[int, ...], str] = {}
        for cell, block in cells.items():
            cell = tuple(int(s) for s in cell)
            if len(cell) != len(names) or any(s not in (-1, 0, 1) for s in cell):
                raise FaultError(f"cell {cell}: one sign (-1, 0, +1) per fault is needed")
            if block not in self.blocks:
                raise FaultError(f"cell {cell}: block {block!r} is not a block of the network")
            self.cells[cell] = str(block)
        self.diagnostics: dict = dict(diagnostics or {})

    def _termination_order(self, names: list[str]) -> list[tuple[int, int, int]]:
        """``(fault, fault it ends on, side)`` as column indices, parents first."""
        index = {name: k for k, name in enumerate(names)}
        order: list[tuple[int, int, int]] = []
        done: set[str] = {name for name in names if name not in self.terminations}
        pending = dict(self.terminations)
        while pending:
            ready = [name for name, (other, _) in pending.items() if other in done]
            if not ready:
                raise FaultError(
                    "terminations form a loop: " + ", ".join(sorted(pending)) + " end on each other"
                )
            for name in ready:
                other, side = pending.pop(name)
                order.append((index[name], index[other], side))
                done.add(name)
        return order

    def _terminate(self, signs: np.ndarray) -> np.ndarray:
        for column, other, side in self._order:
            signs[signs[:, other] != side, column] = 0
        return signs

    @property
    def provenance(self) -> list[ProvenanceEntry]:
        """The provenance lines of the faults (:func:`faults_provenance`)."""
        return faults_provenance(self.faults)

    # ------------------------------------------------------------------ fit

    @classmethod
    def fit(
        cls,
        faults: Iterable[FaultSurface],
        xyz: Any,
        block: Any,
        *,
        terminations: Mapping[str, tuple[str, int]] | None = None,
    ) -> FaultNetwork:
        """Label the cells of the faults with the blocks of the labelled points.

        A cell with labelled points takes their majority block (a tie goes to the block that
        sorts first); the others are conflicts. A cell without labelled points takes the
        block of the labelled point nearest to it: the point that would have to move the
        least across the faults that separate it from the cell (the largest of its offsets
        from those faults).

        With one fault that names its ``blocks`` the labelled points may be empty. Labelled
        points of one block only cannot give fault blocks: :class:`FaultError`.
        """
        faults = tuple(faults)
        points = _xyz(xyz, "xyz", allow_empty=True)
        labels = np.asarray(block, dtype=object).reshape(-1)
        if labels.shape != (len(points),):
            raise FaultError("block must have one value per labelled point")
        if any(pd.isna(v) for v in labels):
            raise FaultError("labelled points must all have a block")
        labels = labels.astype(str)
        k = len(faults)

        if len(points) == 0:
            if k == 1 and faults[0].blocks is not None:
                low, high = faults[0].blocks
                diagnostics = {
                    _cell_text(cell): {"n_points": 0, "majority": None, "conflicts": 0,
                                       "filled_from": "the blocks named by the fault"}
                    for cell in ((-1,), (1,))
                }  # fmt: skip
                return cls(faults, sorted((low, high)), {(-1,): low, (1,): high}, None, diagnostics)
            raise FaultError("no labelled points: the fault blocks cannot be told")

        names = sorted(set(labels))
        if k >= 1 and len(names) < 2:
            raise FaultError(
                f"the labelled points are all in block {names[0]!r} (one block only): "
                "faults need intervals of at least two fault blocks"
            )
        blocks = sorted(set(names) | {b for f in faults if f.blocks for b in f.blocks})
        network = cls(faults, blocks, {}, terminations)
        offsets = (
            np.column_stack([fault.offset(points) for fault in faults])
            if k
            else np.zeros((len(points), 0))
        )
        raw = np.where(offsets > 0.0, 1, -1).astype(np.int8)
        signs = network._terminate(raw.copy())

        diagnostics: dict[str, Any] = {}
        cells: dict[tuple[int, ...], str] = {}
        found, inverse = np.unique(signs, axis=0, return_inverse=True)
        inverse = np.asarray(inverse).reshape(-1)
        for c, cell in enumerate(found):
            inside = labels[inverse == c]
            values, counts = np.unique(inside, return_counts=True)
            majority = str(values[int(np.argmax(counts))])
            key = tuple(int(s) for s in cell)
            cells[key] = majority
            diagnostics[_cell_text(key)] = {
                "n_points": int(len(inside)),
                "majority": majority,
                "conflicts": int(len(inside) - counts.max()),
                "counts": {str(v): int(n) for v, n in zip(values, counts, strict=True)},
            }
        if k:
            every = np.array(list(itertools.product((-1, 1), repeat=k)), dtype=np.int8)
            feasible = np.unique(network._terminate(every), axis=0)
            for cell in feasible:
                key = tuple(int(s) for s in cell)
                if key in cells:
                    continue
                differs = (raw != cell[None, :]) & (cell[None, :] != 0)
                cost = np.where(differs, np.abs(offsets), 0.0).max(axis=1)
                nearest = int(np.argmin(cost))
                cells[key] = str(labels[nearest])
                diagnostics[_cell_text(key)] = {
                    "n_points": 0,
                    "majority": None,
                    "conflicts": 0,
                    "filled_from": "nearest labelled point",
                    "distance": float(cost[nearest]),
                }
        network.cells = cells
        network.diagnostics = diagnostics
        return network

    # ------------------------------------------------------------------ queries

    def signs(self, points: Any) -> np.ndarray:
        """Side of every fault for points ``(n, 3)``: ``(n, k)`` int8, 0 where a fault that
        terminates on another does not exist."""
        p = np.asarray(points, dtype=np.float64)
        if p.ndim != 2 or p.shape[1] != 3:
            raise FaultError(f"points must have shape (n, 3), got {p.shape}")
        if not self.faults:
            return np.zeros((len(p), 0), dtype=np.int8)
        raw = np.column_stack([fault.side(p) for fault in self.faults]).astype(np.int8)
        return self._terminate(raw)

    def block_index(self, points: Any) -> np.ndarray:
        """Index into ``blocks`` for points ``(n, 3)`` (int16); -1 for a point with NaN."""
        p = np.asarray(points, dtype=np.float64)
        signs = self.signs(p)
        index = {name: i for i, name in enumerate(self.blocks)}
        out = np.full(len(p), -1, dtype=np.int16)
        if signs.shape[1] == 0:
            out[:] = index.get(self.cells.get((), ""), -1)
        else:
            powers = 3 ** np.arange(signs.shape[1], dtype=np.int64)
            keys = (signs.astype(np.int64) + 1) @ powers
            for cell, block in self.cells.items():
                out[keys == int((np.array(cell, dtype=np.int64) + 1) @ powers)] = index[block]
        out[~np.isfinite(p).all(axis=1)] = -1
        return out

    def block_of(self, points: Any) -> np.ndarray:
        """Block name per point (object); None where :meth:`block_index` is -1."""
        index = self.block_index(points)
        names = np.array([*self.blocks, None], dtype=object)
        return names[index]

    # ------------------------------------------------------------------ files

    def to_dict(self) -> dict:
        """The network without the surfaces: each fault is named with its file
        ``faults/<name>.json`` relative to the geology folder (:meth:`save` writes them)."""
        return {
            "format": NETWORK_FORMAT,
            "version": VERSION,
            "faults": [{"name": f.name, "file": f"faults/{f.name}.json"} for f in self.faults],
            "blocks": list(self.blocks),
            "cells": [
                {"signs": [int(s) for s in cell], "block": block}
                for cell, block in sorted(self.cells.items())
            ],
            "terminations": {k: [g, int(s)] for k, (g, s) in sorted(self.terminations.items())},
            "diagnostics": _jsonable(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], folder: str | Path) -> FaultNetwork:
        """Inverse of :meth:`to_dict`; ``folder`` is the geology folder the files are under."""
        if not isinstance(data, Mapping) or data.get("format") != NETWORK_FORMAT:
            raise FaultError(f"not a fault network (format {NETWORK_FORMAT!r} expected)")
        if data.get("version") != VERSION:
            raise FaultError(f"fault network version {data.get('version')!r} is not supported")
        folder = Path(folder)
        faults = [FaultSurface.load(folder / item["file"]) for item in data.get("faults", [])]
        cells = {tuple(item["signs"]): item["block"] for item in data.get("cells", [])}
        terminations = {k: (v[0], int(v[1])) for k, v in data.get("terminations", {}).items()}
        return cls(faults, data.get("blocks", []), cells, terminations, data.get("diagnostics"))

    def save(self, folder: str | Path) -> dict:
        """Write every fault under ``<folder>/faults`` and return :meth:`to_dict`."""
        for fault in self.faults:
            fault.save(Path(folder) / "faults")
        return self.to_dict()
