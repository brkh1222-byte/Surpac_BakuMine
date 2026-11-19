"""Desurvey: turn collar + downhole survey tables into 3D hole traces.

Theory: ``research.md`` section 3. Coordinates are ``x = east, y = north, z = up``.
Azimuth is in degrees clockwise from north. Dip is in degrees from horizontal; its sign
is a parameter (``"negative_down"`` is the Surpac-style default, ``"positive_down"`` the
GSLIB style). Depths and coordinates share one length unit, whatever the caller uses.

Path model between two consecutive stations (length ``L``, unit directions ``d1``, ``d2``):

* ``minimum_curvature`` - a circular arc tangent to ``d1`` at the upper station and to
  ``d2`` at the lower one. Points inside the interval lie on that arc (the direction is
  the spherical interpolation of ``d1`` and ``d2``), not on the chord.
* ``balanced_tangential`` - ``L/2`` along ``d1`` followed by ``L/2`` along ``d2``.
* ``tangent`` - ``L`` along ``d1`` (the upper station's direction).

Every trace runs from depth 0 to the hole's ``max_depth``:

* no station at depth 0 - the first station's direction is used back to the collar;
* last station above ``max_depth`` - straight extension on the last direction;
* a station below ``max_depth`` (for example a "straight to end of hole" sentinel at
  depth 90000) - the path towards that station is cut at ``max_depth``.

Each of these, and every other repair, is recorded in ``HoleTraces.warnings``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

__all__ = [
    "DIP_CONVENTIONS",
    "METHODS",
    "DesurveyWarning",
    "HoleTrace",
    "HoleTraces",
    "desurvey",
    "direction_vectors",
]

Method = Literal["minimum_curvature", "balanced_tangential", "tangent"]
DipConvention = Literal["negative_down", "positive_down"]
OnInvalid = Literal["nan", "raise"]

METHODS: tuple[str, ...] = ("minimum_curvature", "balanced_tangential", "tangent")
DIP_CONVENTIONS: tuple[str, ...] = ("negative_down", "positive_down")

#: Station kinds in the station table.
KIND_SURVEY = "survey"  # a measured survey station
KIND_COLLAR = "collar"  # added at depth 0 because the survey had no station there
KIND_END = "end"  # added (or cut) at max_depth
_KINDS = np.array([KIND_SURVEY, KIND_COLLAR, KIND_END], dtype=object)

#: A dogleg within this angle (radians) of 180 degrees has no defined arc plane.
_DEGENERATE_DOGLEG = 1e-6
#: Below this dogleg (radians) the interval is treated as exactly straight.
_STRAIGHT_DOGLEG = 1e-12


@dataclass(frozen=True)
class DesurveyWarning:
    """One repair or anomaly found while desurveying. ``code`` is machine-readable."""

    hole_id: object
    code: str
    message: str


@dataclass(frozen=True)
class HoleTrace:
    """Station table of one hole. ``xyz`` and ``direction`` have shape ``(n, 3)``."""

    hole_id: object
    max_depth: float
    depth: np.ndarray
    xyz: np.ndarray
    direction: np.ndarray
    kind: np.ndarray


def direction_vectors(
    dip: np.ndarray | float,
    azimuth: np.ndarray | float,
    dip_convention: DipConvention = "negative_down",
) -> np.ndarray:
    """Unit direction vectors ``(east, north, up)`` of the hole at survey readings.

    ``dip`` and ``azimuth`` are in degrees and broadcast together; the result has one
    more trailing axis of length 3.
    """
    if dip_convention not in DIP_CONVENTIONS:
        raise ValueError(f"dip_convention must be one of {DIP_CONVENTIONS}, got {dip_convention!r}")
    dip_rad = np.radians(np.asarray(dip, dtype=np.float64))
    if dip_convention == "positive_down":
        dip_rad = -dip_rad
    azi_rad = np.radians(np.asarray(azimuth, dtype=np.float64))
    dip_rad, azi_rad = np.broadcast_arrays(dip_rad, azi_rad)
    horizontal = np.cos(dip_rad)
    return np.stack(
        [horizontal * np.sin(azi_rad), horizontal * np.cos(azi_rad), np.sin(dip_rad)], axis=-1
    )


def _sinc(x: np.ndarray) -> np.ndarray:
    """``sin(x) / x`` with the limit 1 at 0 (numpy's sinc is the normalised one)."""
    return np.sinc(x / np.pi)


@dataclass
class HoleTraces:
    """Desurveyed traces of many holes, stored as flat arrays grouped by hole.

    Build with :func:`desurvey`. ``hole_ids`` lists the holes that have a trace, in
    collar order. ``warnings`` lists every repair made.
    """

    method: str
    hole_ids: np.ndarray
    max_depth: np.ndarray
    warnings: list[DesurveyWarning] = field(default_factory=list)
    # Internal path nodes, sorted by (hole, depth). Node i starts the interval to node
    # i + 1; the last node of a hole starts a zero-length interval.
    _offsets: np.ndarray = field(default=None, repr=False)  # (n_holes + 1,)
    _md: np.ndarray = field(default=None, repr=False)
    _xyz: np.ndarray = field(default=None, repr=False)
    _dir: np.ndarray = field(default=None, repr=False)
    _dir_next: np.ndarray = field(default=None, repr=False)
    _seg_len: np.ndarray = field(default=None, repr=False)
    _kappa: np.ndarray = field(default=None, repr=False)  # curvature, rad per unit length
    _perp: np.ndarray = field(default=None, repr=False)  # unit normal of the arc, in its plane
    # Public station table (a node below max_depth is replaced by the point at max_depth).
    _st_md: np.ndarray = field(default=None, repr=False)
    _st_xyz: np.ndarray = field(default=None, repr=False)
    _st_dir: np.ndarray = field(default=None, repr=False)
    _st_kind: np.ndarray = field(default=None, repr=False)
    _index: pd.Index = field(default=None, repr=False)

    def __len__(self) -> int:
        return len(self.hole_ids)

    def __contains__(self, hole_id: object) -> bool:
        return hole_id in self._index

    # ------------------------------------------------------------------ station tables

    def stations(self) -> pd.DataFrame:
        """All stations: ``hole_id, depth, x, y, z, dx, dy, dz, kind``.

        ``(dx, dy, dz)`` is the unit direction of the hole at the station. ``kind`` is
        ``"survey"``, ``"collar"`` (added at depth 0) or ``"end"`` (added or cut at
        ``max_depth``).
        """
        counts = np.diff(self._offsets)
        return pd.DataFrame(
            {
                "hole_id": np.repeat(self.hole_ids, counts),
                "depth": self._st_md,
                "x": self._st_xyz[:, 0],
                "y": self._st_xyz[:, 1],
                "z": self._st_xyz[:, 2],
                "dx": self._st_dir[:, 0],
                "dy": self._st_dir[:, 1],
                "dz": self._st_dir[:, 2],
                "kind": _KINDS[self._st_kind],
            }
        )

    def hole(self, hole_id: object) -> HoleTrace:
        """Station table of one hole. Raises ``KeyError`` for a hole without a trace."""
        i = self._index.get_loc(hole_id)
        a, b = self._offsets[i], self._offsets[i + 1]
        return HoleTrace(
            hole_id=self.hole_ids[i],
            max_depth=float(self.max_depth[i]),
            depth=self._st_md[a:b].copy(),
            xyz=self._st_xyz[a:b].copy(),
            direction=self._st_dir[a:b].copy(),
            kind=_KINDS[self._st_kind[a:b]],
        )

    def warnings_frame(self) -> pd.DataFrame:
        """``warnings`` as a table with columns ``hole_id, code, message``."""
        return pd.DataFrame(
            [(w.hole_id, w.code, w.message) for w in self.warnings],
            columns=["hole_id", "code", "message"],
        )

    # ------------------------------------------------------------------ interpolation

    def xyz_at(
        self,
        hole_id: object,
        depths: np.ndarray | float,
        *,
        on_invalid: OnInvalid = "nan",
        depth_tolerance: float = 1e-9,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Coordinates and unit direction of the hole path at downhole depths.

        ``hole_id`` is one id or an array of ids; it broadcasts against ``depths``.
        Returns ``(xyz, direction)``, each of shape ``broadcast_shape + (3,)``.

        A query is invalid when the hole has no trace, the depth is NaN, or the depth
        lies outside ``[0, max_depth]`` by more than ``depth_tolerance`` (depths within
        the tolerance are snapped to the nearest end). Invalid queries give NaN rows
        with ``on_invalid="nan"`` and raise ``ValueError`` with ``on_invalid="raise"``.
        """
        if on_invalid not in ("nan", "raise"):
            raise ValueError(f"on_invalid must be 'nan' or 'raise', got {on_invalid!r}")
        q = np.asarray(depths, dtype=np.float64)
        ids = np.asarray(hole_id, dtype=object) if not np.isscalar(hole_id) else None
        if ids is None or ids.ndim == 0:
            key = hole_id if ids is None else ids.item()
            h_scalar = self._index.get_loc(key) if key in self._index else -1
            shape = q.shape
            q = q.ravel()
            h = np.full(q.shape, h_scalar, dtype=np.int64)
            single = True
        else:
            shape = np.broadcast_shapes(ids.shape, q.shape)
            h = self._index.get_indexer(ids.ravel()).astype(np.int64).reshape(ids.shape)
            h = np.broadcast_to(h, shape).ravel()
            q = np.broadcast_to(q, shape).ravel()
            single = False

        known = h >= 0
        h_safe = np.where(known, h, 0)
        if len(self.hole_ids) == 0:
            limit = np.full(q.shape, np.nan)
        else:
            limit = self.max_depth[h_safe]
        with np.errstate(invalid="ignore"):
            valid = known & (q >= -depth_tolerance) & (q <= limit + depth_tolerance)
        if on_invalid == "raise" and not valid.all():
            bad = np.flatnonzero(~valid)
            first = bad[0]
            first_id = hole_id if single else np.broadcast_to(ids, shape).ravel()[first]
            reason = (
                "hole has no trace"
                if not known[first]
                else f"depth outside [0, {limit[first]:g}]"
            )
            raise ValueError(
                f"{bad.size} invalid xyz_at queries; first: hole {first_id!r}, "
                f"depth {q[first]!r} ({reason})"
            )

        xyz = np.full((q.size, 3), np.nan)
        direction = np.full((q.size, 3), np.nan)
        sel = np.flatnonzero(valid)
        if sel.size:
            hs = h[sel]
            qs = np.clip(q[sel], 0.0, limit[sel])
            node = self._locate(hs, qs, single)
            xyz[sel], direction[sel] = self._evaluate(node, qs)
        return xyz.reshape(shape + (3,)), direction.reshape(shape + (3,))

    def _locate(self, h: np.ndarray, q: np.ndarray, single: bool) -> np.ndarray:
        """Index of the node that starts the interval containing each depth."""
        offsets, md = self._offsets, self._md
        if single:
            a, b = offsets[h[0]], offsets[h[0] + 1]
            return a + np.clip(np.searchsorted(md[a:b], q, side="right") - 1, 0, b - a - 1)
        order = np.argsort(h, kind="stable")
        h_sorted = h[order]
        q_sorted = q[order]
        cuts = np.flatnonzero(h_sorted[1:] != h_sorted[:-1]) + 1
        starts = np.concatenate(([0], cuts))
        ends = np.concatenate((cuts, [h_sorted.size]))
        node_sorted = np.empty(h.size, dtype=np.int64)
        for s, e in zip(starts.tolist(), ends.tolist(), strict=True):
            hole = h_sorted[s]
            a, b = offsets[hole], offsets[hole + 1]
            local = np.searchsorted(md[a:b], q_sorted[s:e], side="right") - 1
            node_sorted[s:e] = a + np.clip(local, 0, b - a - 1)
        node = np.empty(h.size, dtype=np.int64)
        node[order] = node_sorted
        return node

    def _evaluate(self, node: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Position and direction at depth ``q`` inside the interval starting at ``node``."""
        s = (q - self._md[node])[:, None]
        d1 = self._dir[node]
        if self.method == "balanced_tangential":
            half = 0.5 * self._seg_len[node][:, None]
            d2 = self._dir_next[node]
            xyz = self._xyz[node] + np.minimum(s, half) * d1 + np.maximum(s - half, 0.0) * d2
            direction = np.where(s < half, d1, d2)
            # The last node of a hole has no second half: keep its own direction.
            direction = np.where(half > 0.0, direction, d1)
            return xyz, direction
        # Circular arc of curvature kappa in the plane (d1, perp); kappa = 0 is a line.
        # Position is the integral of the unit direction cos(theta) d1 + sin(theta) perp,
        # written with sinc so that it has no division and no cancellation at theta -> 0.
        theta = s * self._kappa[node][:, None]
        perp = self._perp[node]
        half_theta = 0.5 * theta
        xyz = (
            self._xyz[node]
            + s * _sinc(theta) * d1
            + s * np.sin(half_theta) * _sinc(half_theta) * perp
        )
        direction = np.cos(theta) * d1 + np.sin(theta) * perp
        return xyz, direction


# ---------------------------------------------------------------------- construction


def _require_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [c for c in columns if c not in frame.columns]
    if missing:
        raise ValueError(f"{name} table is missing columns {missing}; has {list(frame.columns)}")


def _group_bounds(hole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """First and last row index of each run in a sorted hole-index array."""
    if hole.size == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty
    cuts = np.flatnonzero(hole[1:] != hole[:-1]) + 1
    return np.concatenate(([0], cuts)), np.concatenate((cuts - 1, [hole.size - 1]))


def _sort_nodes(hole, md, direction, kind):
    order = np.lexsort((md, hole))
    return hole[order], md[order], direction[order], kind[order]


def desurvey(
    collar: pd.DataFrame,
    survey: pd.DataFrame,
    *,
    method: Method = "minimum_curvature",
    dip_convention: DipConvention = "negative_down",
) -> HoleTraces:
    """Desurvey all holes.

    ``collar`` needs columns ``hole_id, x, y, z, max_depth`` (x east, y north, z up) and
    ``survey`` needs ``hole_id, depth, dip, azimuth`` (degrees). Rows may be in any order.

    Nothing is dropped or repaired silently; see ``HoleTraces.warnings``. Warning codes:

    ``survey_without_collar``, ``invalid_station`` (non-finite value or negative depth,
    row ignored), ``duplicate_station`` (same hole and depth, first row kept),
    ``invalid_collar`` (non-finite collar coordinate, hole skipped), ``no_survey`` (hole
    skipped), ``no_max_depth`` (missing or negative ``max_depth``; the deepest station is
    used), ``no_station_at_zero``, ``station_beyond_max_depth`` and ``dogleg_180``
    (consecutive directions are opposite, so no arc is defined; minimum curvature runs
    that interval straight on the upper station's direction).
    """
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    _require_columns(collar, ("hole_id", "x", "y", "z", "max_depth"), "collar")
    _require_columns(survey, ("hole_id", "depth", "dip", "azimuth"), "survey")

    warnings: list[DesurveyWarning] = []

    def warn(hole_id: object, code: str, message: str) -> None:
        warnings.append(DesurveyWarning(hole_id, code, message))

    collar_ids = collar["hole_id"].to_numpy(dtype=object)
    collar_index = pd.Index(collar_ids)
    if not collar_index.is_unique:
        dup = collar_index[collar_index.duplicated()].unique().tolist()
        raise ValueError(f"duplicate hole_id in collar table: {dup[:10]}")
    n_collar = len(collar_ids)
    collar_xyz = collar[["x", "y", "z"]].to_numpy(dtype=np.float64)
    collar_depth = collar["max_depth"].to_numpy(dtype=np.float64).copy()
    collar_ok = np.isfinite(collar_xyz).all(axis=1)

    # --- survey rows: map to collar, drop unusable rows, sort, drop duplicates --------
    survey_ids = survey["hole_id"].to_numpy(dtype=object)
    hole = collar_index.get_indexer(survey_ids).astype(np.int64)
    md = survey["depth"].to_numpy(dtype=np.float64)
    dip = survey["dip"].to_numpy(dtype=np.float64)
    azimuth = survey["azimuth"].to_numpy(dtype=np.float64)

    orphan = hole < 0
    if orphan.any():
        for hid, count in zip(*np.unique(survey_ids[orphan].astype(str), return_counts=True),
                              strict=True):
            warn(hid, "survey_without_collar", f"{count} survey rows have no collar; ignored")
    with np.errstate(invalid="ignore"):
        bad = ~orphan & ~(np.isfinite(md) & np.isfinite(dip) & np.isfinite(azimuth) & (md >= 0))
    for i in np.flatnonzero(bad):
        warn(
            collar_ids[hole[i]],
            "invalid_station",
            f"survey row ignored: depth={md[i]!r}, dip={dip[i]!r}, azimuth={azimuth[i]!r}",
        )
    keep = ~orphan & ~bad
    hole, md = hole[keep], md[keep]
    direction = direction_vectors(dip[keep], azimuth[keep], dip_convention).reshape(-1, 3)
    kind = np.zeros(hole.size, dtype=np.int8)
    hole, md, direction, kind = _sort_nodes(hole, md, direction, kind)  # stable: file order

    duplicate = np.zeros(hole.size, dtype=bool)
    duplicate[1:] = (hole[1:] == hole[:-1]) & (md[1:] == md[:-1])
    for i in np.flatnonzero(duplicate):
        conflict = "" if np.allclose(direction[i], direction[i - 1]) else " (readings differ)"
        warn(
            collar_ids[hole[i]],
            "duplicate_station",
            f"duplicate survey station at depth {md[i]:g}{conflict}; first row kept",
        )
    hole, md, direction, kind = hole[~duplicate], md[~duplicate], direction[~duplicate], kind[
        ~duplicate
    ]

    # --- which holes get a trace ------------------------------------------------------
    has_survey = np.zeros(n_collar, dtype=bool)
    has_survey[hole] = True
    for i in np.flatnonzero(~collar_ok):
        warn(collar_ids[i], "invalid_collar", "collar coordinate is not finite; hole skipped")
    for i in np.flatnonzero(collar_ok & ~has_survey):
        warn(collar_ids[i], "no_survey", "hole has no usable survey station; hole skipped")
    traced = collar_ok & has_survey
    usable = traced[hole]
    hole, md, direction, kind = hole[usable], md[usable], direction[usable], kind[usable]

    first, last = _group_bounds(hole)

    with np.errstate(invalid="ignore"):
        no_depth = traced & ~(np.isfinite(collar_depth) & (collar_depth >= 0))
    if no_depth.any():
        deepest = np.zeros(n_collar)
        deepest[hole[last]] = md[last]
        for i in np.flatnonzero(no_depth):
            warn(
                collar_ids[i],
                "no_max_depth",
                f"max_depth is {collar_depth[i]!r}; using the deepest station {deepest[i]:g}",
            )
        collar_depth[no_depth] = deepest[no_depth]

    # --- node at depth 0 --------------------------------------------------------------
    need_zero = first[md[first] > 0]
    for i in need_zero:
        warn(
            collar_ids[hole[i]],
            "no_station_at_zero",
            f"no survey at depth 0; orientation of the first station ({md[i]:g}) used from "
            "the collar",
        )
    if need_zero.size:
        hole = np.concatenate((hole, hole[need_zero]))
        md = np.concatenate((md, np.zeros(need_zero.size)))
        direction = np.concatenate((direction, direction[need_zero]))
        kind = np.concatenate((kind, np.full(need_zero.size, 1, dtype=np.int8)))
        hole, md, direction, kind = _sort_nodes(hole, md, direction, kind)

    # --- stations below max_depth: keep only the first, as the far end of the last arc -
    limit = collar_depth[hole]
    over = md > limit
    if over.any():
        prev_over = np.zeros(hole.size, dtype=bool)
        prev_over[1:] = over[:-1] & (hole[1:] == hole[:-1])
        prev_at_limit = np.zeros(hole.size, dtype=bool)
        prev_at_limit[1:] = md[:-1] == limit[1:]  # previous node is in the same hole here
        first_over = over & ~prev_over
        over_count = np.bincount(hole[over], minlength=n_collar)
        for i in np.flatnonzero(first_over):
            warn(
                collar_ids[hole[i]],
                "station_beyond_max_depth",
                f"{over_count[hole[i]]} survey station(s) deeper than max_depth {limit[i]:g} "
                f"(first at {md[i]:g}); trace cut at max_depth",
            )
        keep = ~over | (first_over & ~prev_at_limit)
        hole, md, direction, kind = hole[keep], md[keep], direction[keep], kind[keep]

    # --- node at max_depth: straight extension on the last direction ------------------
    first, last = _group_bounds(hole)
    need_end = last[md[last] < collar_depth[hole[last]]]
    if need_end.size:
        hole = np.concatenate((hole, hole[need_end]))
        md = np.concatenate((md, collar_depth[hole[need_end]]))
        direction = np.concatenate((direction, direction[need_end]))
        kind = np.concatenate((kind, np.full(need_end.size, 2, dtype=np.int8)))
        hole, md, direction, kind = _sort_nodes(hole, md, direction, kind)
        first, last = _group_bounds(hole)

    # --- intervals --------------------------------------------------------------------
    n = hole.size
    is_last = np.zeros(n, dtype=bool)
    is_last[last] = True
    nxt = np.minimum(np.arange(n) + 1, max(n - 1, 0))
    nxt[is_last] = np.flatnonzero(is_last)  # the last node pairs with itself
    seg_len = md[nxt] - md
    d1, d2 = direction, direction[nxt]
    cross = np.cross(d1, d2)
    sin_beta = np.linalg.norm(cross, axis=1)
    cos_beta = np.einsum("ij,ij->i", d1, d2)
    beta = np.arctan2(sin_beta, cos_beta)  # dogleg angle, accurate near 0 and near pi

    degenerate = beta > np.pi - _DEGENERATE_DOGLEG
    fallback = {
        "minimum_curvature": "interval run straight on the upper station's direction",
        "balanced_tangential": "the two half-steps cancel",
        "tangent": "no effect on the tangent method",
    }[method]
    for i in np.flatnonzero(degenerate):
        warn(
            collar_ids[hole[i]],
            "dogleg_180",
            f"directions at depths {md[i]:g} and {md[nxt[i]]:g} are opposite (dogleg 180 "
            f"degrees); {fallback}",
        )

    kappa = np.zeros(n)
    perp = np.zeros((n, 3))
    if method == "minimum_curvature":
        curved = (beta > _STRAIGHT_DOGLEG) & ~degenerate & (seg_len > 0)
        in_plane = d2[curved] - cos_beta[curved, None] * d1[curved]
        perp[curved] = in_plane / np.linalg.norm(in_plane, axis=1)[:, None]
        kappa[curved] = beta[curved] / seg_len[curved]

    hole_ids = collar_ids[traced]
    renumber = np.cumsum(traced) - 1
    hole_new = renumber[hole]
    counts = np.bincount(hole_new, minlength=hole_ids.size)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)

    traces = HoleTraces(
        method=method,
        hole_ids=hole_ids,
        max_depth=collar_depth[traced],
        warnings=warnings,
        _offsets=offsets,
        _md=md,
        _xyz=np.zeros((n, 3)),
        _dir=direction,
        _dir_next=d2,
        _seg_len=seg_len,
        _kappa=kappa,
        _perp=perp,
        _index=pd.Index(hole_ids),
    )

    # Step of each interval = the path evaluated at its far end (zero-origin), then a
    # running sum per hole. The global cumulative sum loses at most eps * (total length
    # of all holes), i.e. about 1e-8 for 50 000 km of drilling.
    nodes = np.arange(n)
    step, _ = traces._evaluate(nodes, md + seg_len)
    before = np.zeros((n + 1, 3))
    np.cumsum(step, axis=0, out=before[1:])
    before = before[:-1]
    first_of_node = offsets[:-1][hole_new] if n else np.zeros(0, dtype=np.int64)
    traces._xyz = collar_xyz[hole] + (before - before[first_of_node])

    # Public station table: a node below max_depth becomes the path point at max_depth.
    st_md, st_xyz, st_dir, st_kind = md.copy(), traces._xyz.copy(), direction.copy(), kind.copy()
    beyond = np.flatnonzero(md > collar_depth[hole])
    if beyond.size:
        cut_depth = collar_depth[hole[beyond]]
        st_xyz[beyond], st_dir[beyond] = traces._evaluate(beyond - 1, cut_depth)
        st_md[beyond] = cut_depth
        st_kind[beyond] = 2
    traces._st_md, traces._st_xyz, traces._st_dir, traces._st_kind = st_md, st_xyz, st_dir, st_kind
    return traces
