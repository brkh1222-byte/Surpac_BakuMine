"""Orientation of stacked tabular bodies from the ore / waste logging of neighbouring holes.

The midpoint estimate of :func:`bakumine.core.geometry.implicit.estimate_intercept_anisotropy`
works for **one** tabular body: its intercept midpoints lie on one surface. A stack of
parallel lenses drilled over a volume gives a cloud of midpoints whose shape is that of
the drilled volume, and the principal components then describe the drilling pattern.

Here the plane is found by a **direct search over the pole** ``n`` (unit normal) of the
plane, maximising the **indicator correlation between neighbouring holes**:

1. Every hole is sampled at a constant down-hole step with the indicator ``I`` (1 in the
   domain, 0 outside, unknown where nothing is logged).
2. For a sample point ``p`` of hole A and a neighbouring hole B (one of the nearest holes
   within the search distance), the point of B *on the plane through ``p`` with pole
   ``n``* is ``q + t d`` with ``t = ((p - q) . n) / (d . n)``, where ``q`` is the sample
   of B nearest to ``p`` and ``d`` the direction of B there (the hole is taken as
   straight between ``q`` and that point). If the bodies follow the plane, the indicator
   of B at that point equals the indicator of A at ``p``.
3. A pair is used when the matched point is logged, the distance ``r`` between the two
   points (it lies in the plane) is at most twice the search distance, and both holes
   cross the plane at more than :data:`MIN_CROSSING` (``|d . n|``). Its weight is
   ``1 / (1 + (r / spacing)^2)``: near pairs count most.
4. The pairs are grouped by (hole A, hole B) and the two indicators are centred on their
   weighted means **within each group**, so that a hole with much ore next to a hole
   with little does not count as continuity. The groups are sorted into
   :data:`N_SECTORS` sectors by the direction from hole A to hole B (seen along the mean
   hole direction). The correlation of a sector is the pooled within-group correlation
   ``sum(cov) / sqrt(sum(var_A) sum(var_B))``; a sector in which fewer than 15 % of its
   pairs can be compared counts as 0. The **score** of ``n`` is the mean over the sectors
   that hold at least :data:`MIN_SECTOR_SHARE` of the pairs. Averaging by sector is what
   makes the strike count as much as the dip: on a 25 x 50 m grid most near pairs lie
   along the section, and every plane with the right apparent dip in the section matches
   those. This is the normalised form of "the plane along which the indicator variogram
   is smallest": for indicators ``gamma = p (1 - p) (1 - rho)``.
5. The poles tried are a regular set on the upper hemisphere (about
   :data:`COARSE_STEP_DEG` apart), then a fine set (1 degree) around the best one.

Confidence
----------
``correlation`` is the best score; ``runner_up`` the best score among the poles more than
:data:`EXCLUSION_DEG` from the best one; ``confidence = 1 - runner_up / correlation``
(0 when the correlation is not positive). ``stability`` is the share of
:data:`STABILITY_RESAMPLES` bootstrap resamplings of the holes (hole A of every group drawn
with replacement, fixed seed) whose best plane among the poles tried is within
:data:`EXCLUSION_DEG` of the chosen one. The confidence compares two maxima of a noisy
score: with thin, short lenses against the hole spacing, a spike of the score can stand
well above the next plane (one synthetic case: correlation 0.48, confidence 0.23, the plane
45 degrees off); the stability shows it, as that plane does not come back when the holes
are resampled. The orientation is **determined** when ``correlation >= MIN_CORRELATION``,
``confidence >= MIN_CONFIDENCE``, ``stability >= MIN_STABILITY`` and enough pairs were
compared. A single fence of holes, holes drilled along the bodies, or intercepts
without continuity between holes give a flat score: several very different planes explain
the logging equally well, the confidence is low, and no plane is chosen.

Limits: one orientation for the whole data set (no folds); planes that contain the hole
direction cannot be seen (the holes do not cross them) and are not tried; the continuity
*within* the plane (strike against dip) is not estimated.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = [
    "COARSE_STEP_DEG",
    "EXCLUSION_DEG",
    "MIN_CONFIDENCE",
    "MIN_CORRELATION",
    "MIN_CROSSING",
    "MIN_STABILITY",
    "PlaneEstimate",
    "estimate_plane_from_holes",
    "pole_of_plane",
]

#: spacing of the poles of the first search, degrees
COARSE_STEP_DEG = 4.0
#: a second maximum closer than this to the best pole is the same maximum
EXCLUSION_DEG = 25.0
#: smallest best correlation and smallest confidence of a determined orientation
MIN_CORRELATION = 0.25
MIN_CONFIDENCE = 0.15
#: hole resamplings of the stability check, and the smallest share of them whose best plane
#: must be within EXCLUSION_DEG of the chosen one
STABILITY_RESAMPLES = 100
MIN_STABILITY = 0.8
#: smallest ``|hole direction . pole|`` of a usable pair (holes at < 11.5 degrees to the plane)
MIN_CROSSING = 0.2
#: fewest compared pairs (sample point, neighbouring hole) of a usable pole
MIN_PAIRS = 50
#: sectors of the direction from a hole to its neighbour (180 degrees / N_SECTORS each)
N_SECTORS = 6
#: a sector is used when it holds at least this share of the pairs
MIN_SECTOR_SHARE = 0.05
MIN_SECTOR_PAIRS = 10
MIN_GROUP_PAIRS = 5
_MAX_SAMPLES = 20_000
_MAX_PAIRS = 40_000
_NEIGHBOUR_HOLES = 6


def pole_of_plane(dip: float, dip_azimuth: float) -> np.ndarray:
    """Upward unit normal (east, north, up) of a plane dipping ``dip`` towards ``dip_azimuth``."""
    d, a = math.radians(dip), math.radians(dip_azimuth)
    return np.array([math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)])


def _plane_of_pole(pole: np.ndarray) -> tuple[float, float]:
    n = pole if pole[2] >= 0 else -pole
    dip = math.degrees(math.acos(min(1.0, float(n[2]))))
    azimuth = math.degrees(math.atan2(n[0], n[1])) % 360.0 if dip > 1e-6 else 0.0
    return dip, azimuth


@dataclass
class PlaneEstimate:
    """Result of :func:`estimate_plane_from_holes`; see the module docstring."""

    pole: np.ndarray
    dip: float
    dip_azimuth: float
    correlation: float
    runner_up: float
    confidence: float
    determined: bool
    n_pairs: int
    n_holes: int
    search_distance: float
    spacing: float
    #: median in-plane distance of the compared pairs and mean true thickness of the intercepts
    pair_distance: float = float("nan")
    thickness: float = float("nan")
    #: angle between the pole and the mean hole direction, degrees
    hole_angle_deg: float = float("nan")
    #: share of the hole resamplings whose best plane is within EXCLUSION_DEG of this one
    stability: float = float("nan")
    reason: str = ""
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": "indicator correlation between holes",
            "dip": self.dip,
            "dip_azimuth": self.dip_azimuth,
            "correlation": self.correlation,
            "runner_up": self.runner_up,
            "confidence": self.confidence,
            "determined": self.determined,
            "stability": self.stability,
            "n_pairs": self.n_pairs,
            "n_holes": self.n_holes,
            "search_distance": self.search_distance,
            "hole_angle_deg": self.hole_angle_deg,
            "thickness": self.thickness,
            "warnings": list(self.warnings),
        }


def _hemisphere(step_deg: float) -> np.ndarray:
    """Unit vectors on the upper hemisphere about ``step_deg`` apart (rings of colatitude)."""
    out = [np.array([[0.0, 0.0, 1.0]])]
    step = math.radians(step_deg)
    for i in range(1, int(round(0.5 * math.pi / step)) + 1):
        theta = i * step
        m = max(1, int(round(2.0 * math.pi * math.sin(theta) / step)))
        if abs(theta - 0.5 * math.pi) < 1e-9:
            m = max(1, m // 2)  # an axis on the equator is counted once
            phi = np.arange(m) * (math.pi / m)
        else:
            phi = np.arange(m) * (2.0 * math.pi / m)
        out.append(
            np.column_stack(
                [np.sin(theta) * np.sin(phi), np.sin(theta) * np.cos(phi), np.cos(theta) + 0 * phi]
            )
        )
    return np.vstack(out)


def _around(pole: np.ndarray, radius_deg: float, step_deg: float) -> np.ndarray:
    """Unit vectors within ``radius_deg`` of ``pole`` about ``step_deg`` apart."""
    helper = np.array([1.0, 0.0, 0.0]) if abs(pole[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(pole, helper)
    u /= np.linalg.norm(u)
    v = np.cross(pole, u)
    out = [pole[None, :]]
    for i in range(1, int(round(radius_deg / step_deg)) + 1):
        theta = math.radians(i * step_deg)
        m = max(6, int(round(2.0 * math.pi * i)))
        phi = np.arange(m) * (2.0 * math.pi / m)
        out.append(
            math.cos(theta) * pole[None, :]
            + math.sin(theta) * (np.cos(phi)[:, None] * u + np.sin(phi)[:, None] * v)
        )
    return np.vstack(out)


class _Pairs:
    """The (sample point, neighbouring hole) pairs and their score for a set of poles."""

    def __init__(self, xyz, direction, hole, state, value, step, search_distance, spacing):
        from scipy.spatial import cKDTree

        self.state = state
        self.value = value
        self.step = step
        self.spacing = spacing
        self.max_distance = 2.0 * search_distance
        n = len(xyz)
        # first / one-past-last sample of every hole (the samples are grouped by hole)
        change = np.flatnonzero(np.diff(hole)) + 1
        starts = np.concatenate([[0], change])
        ends = np.concatenate([change, [n]])
        tree = cKDTree(xyz)
        pairs = tree.query_pairs(search_distance, output_type="ndarray")
        if len(pairs):
            pairs = np.vstack([pairs, pairs[:, ::-1]])
            pairs = pairs[(hole[pairs[:, 0]] != hole[pairs[:, 1]]) & (state[pairs[:, 0]] >= 0)]
        if len(pairs):
            dist = np.linalg.norm(xyz[pairs[:, 0]] - xyz[pairs[:, 1]], axis=1)
            # the nearest sample of each other hole, then the nearest holes of each sample
            order = np.lexsort((dist, hole[pairs[:, 1]], pairs[:, 0]))
            pairs, dist = pairs[order], dist[order]
            first = np.ones(len(pairs), dtype=bool)
            first[1:] = (pairs[1:, 0] != pairs[:-1, 0]) | (
                hole[pairs[1:, 1]] != hole[pairs[:-1, 1]]
            )
            pairs, dist = pairs[first], dist[first]
            order = np.lexsort((dist, pairs[:, 0]))
            pairs, dist = pairs[order], dist[order]
            rank = np.arange(len(pairs)) - np.searchsorted(pairs[:, 0], pairs[:, 0])
            pairs = pairs[rank < _NEIGHBOUR_HOLES]
        if len(pairs) > _MAX_PAIRS:
            pairs = pairs[:: int(math.ceil(len(pairs) / _MAX_PAIRS))]
        a, b = (pairs[:, 0], pairs[:, 1]) if len(pairs) else (np.empty(0, int), np.empty(0, int))
        self.n = len(a)
        self.a_value = value[a]
        self.a_dir = direction[a]
        self.b_index = b
        self.b_dir = direction[b]
        self.b_lo = starts[hole[b]] if self.n else b
        self.b_hi = ends[hole[b]] if self.n else b
        self.v = xyz[a] - xyz[b]
        self.vv = (self.v**2).sum(axis=1)
        self.vd = (self.v * self.b_dir).sum(axis=1)
        # direction of every pair (hole to hole, across the mean hole direction) by sector
        self.sector = np.zeros(self.n, dtype=np.int64)
        self.sectors = np.empty(0, dtype=np.int64)
        self.sector_size = np.zeros(N_SECTORS, dtype=np.int64)
        self.fence_ratio = 0.0
        if self.n:
            vec = np.linalg.eigh(direction.T @ direction)[1]
            e = np.column_stack([self.v @ vec[:, 0], self.v @ vec[:, 1]])
            length = np.linalg.norm(e, axis=1)
            good = length > 1e-9
            e[good] /= length[good, None]
            lam = np.linalg.eigvalsh(e[good].T @ e[good])
            self.fence_ratio = float(lam[0] / lam[1]) if lam[1] > 0 else 0.0
            angle = np.arctan2(e[:, 1], e[:, 0]) % np.pi
            self.sector = np.minimum((angle / (np.pi / N_SECTORS)).astype(np.int64), N_SECTORS - 1)
            share = np.bincount(self.sector[good], minlength=N_SECTORS) / max(int(good.sum()), 1)
            self.sectors = np.flatnonzero(share >= MIN_SECTOR_SHARE)
            self.sector_size = np.bincount(self.sector, minlength=N_SECTORS)
        # groups: the pairs of one hole with one neighbouring hole in one sector
        key = (hole[a] * (hole.max() + 1 if n else 1) + hole[b]) * N_SECTORS + self.sector
        _, first, self.group = np.unique(key, return_index=True, return_inverse=True)
        self.group_sector = self.sector[first]
        #: hole A of every group (the unit resampled by the stability check)
        self.group_hole = hole[a][first] if self.n else np.empty(0, dtype=np.int64)

    def score(self, poles: np.ndarray, chunk: int = 48) -> tuple[np.ndarray, np.ndarray]:
        """(score, number of pairs compared) for each pole; see the module docstring."""
        rho = np.full(len(poles), np.nan)
        count = np.zeros(len(poles), dtype=np.int64)
        if self.n == 0 or len(self.sectors) == 0:
            return rho, count
        ones = np.ones((1, len(self.group_sector)))
        for c0 in range(0, len(poles), chunk):
            sums = self._group_sums(poles[c0 : c0 + chunk])
            rho[c0 : c0 + chunk] = self._combine(sums, ones)[0]
            count[c0 : c0 + chunk] = sums[4]
        return rho, count

    def resampled_scores(
        self, poles: np.ndarray, group_weights: np.ndarray, chunk: int = 48
    ) -> np.ndarray:
        """Score of each pole (columns) for each weighting of the groups (rows): the
        stability check weights every group by how often its hole A was drawn."""
        out = np.full((len(group_weights), len(poles)), np.nan)
        if self.n == 0 or len(self.sectors) == 0:
            return out
        for c0 in range(0, len(poles), chunk):
            out[:, c0 : c0 + chunk] = self._combine(
                self._group_sums(poles[c0 : c0 + chunk]), group_weights
            )
        return out

    def _group_sums(self, p: np.ndarray) -> tuple[np.ndarray, ...]:
        """Per group and pole: centred co-variance, the two variances and the pairs compared
        (0 for groups with too few pairs), and per pole the number of pairs compared."""
        from scipy.sparse import csr_matrix

        ia = self.a_value[:, None]
        to_group = csr_matrix(
            (np.ones(self.n), (self.group, np.arange(self.n))),
            shape=(len(self.group_sector), self.n),
        )
        dn = self.b_dir @ p.T
        ok = (np.abs(dn) > MIN_CROSSING) & (np.abs(self.a_dir @ p.T) > MIN_CROSSING)
        with np.errstate(divide="ignore", invalid="ignore"):
            t = np.where(ok, (self.v @ p.T) / dn, 0.0)
        index = self.b_index[:, None] + np.rint(t / self.step).astype(np.int64)
        ok &= (index >= self.b_lo[:, None]) & (index < self.b_hi[:, None])
        index = np.where(ok, index, 0)
        ok &= self.state[index] >= 0
        ib = self.value[index]
        r2 = self.vv[:, None] - 2.0 * t * self.vd[:, None] + t * t
        ok &= r2 <= self.max_distance**2
        w = np.where(ok, 1.0 / (1.0 + np.maximum(r2, 0.0) / self.spacing**2), 0.0)
        # weighted sums of each group, centred on the means of the group
        sw = to_group @ w
        n_ok = to_group @ ok.astype(np.float64)
        sa, sb = to_group @ (w * ia), to_group @ (w * ib)
        with np.errstate(divide="ignore", invalid="ignore"):
            cab = to_group @ (w * ia * ib) - sa * sb / sw
            va = to_group @ (w * ia * ia) - sa * sa / sw
            vb = to_group @ (w * ib * ib) - sb * sb / sw
        use = (n_ok >= MIN_GROUP_PAIRS) & (sw > 0)
        cab, va, vb, n_used = (np.where(use, x, 0.0) for x in (cab, va, vb, n_ok))
        return cab, va, vb, n_used, ok.sum(axis=0)

    def _combine(self, sums: tuple[np.ndarray, ...], group_weights: np.ndarray) -> np.ndarray:
        """Mean over the used sectors of the pooled correlation, for each row of weights."""
        cab, va, vb, n_used, _ = sums
        r, s = len(group_weights), len(self.sectors)
        in_sector = (self.group_sector[None, :] == self.sectors[:, None]).astype(np.float64)
        m = (group_weights[:, None, :] * in_sector[None, :, :]).reshape(r * s, -1)
        with np.errstate(divide="ignore", invalid="ignore"):
            value = (m @ cab) / np.sqrt((m @ va) * (m @ vb))
        need = np.maximum(MIN_SECTOR_PAIRS, 0.15 * self.sector_size[self.sectors])
        value[~np.isfinite(value) | (m @ n_used < np.tile(need, r)[:, None])] = 0.0
        return value.reshape(r, s, -1).mean(axis=1)

    def in_plane_distance(self, pole: np.ndarray) -> float:
        dn = self.b_dir @ pole
        ok = np.abs(dn) > MIN_CROSSING
        t = (self.v[ok] @ pole) / dn[ok]
        r2 = self.vv[ok] - 2.0 * t * self.vd[ok] + t * t
        r2 = r2[r2 <= self.max_distance**2]
        return float(np.sqrt(np.median(r2))) if len(r2) else float("nan")


def _angle(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle in degrees between the axes ``a`` (n, 3) and ``b`` (3,)."""
    return np.degrees(np.arccos(np.clip(np.abs(a @ b), 0.0, 1.0)))


def estimate_plane_from_holes(
    runs: Any,
    traces: Any,
    *,
    search_distance: float | None = None,
    step: float | None = None,
    smoothing: float = 0.0,
) -> PlaneEstimate:
    """The plane of stacked tabular bodies from logged runs; see the module docstring.

    ``runs`` is a table ``hole_id, depth_from, depth_to, state`` (state > 0 in the domain,
    < 0 outside; depth that no run covers is unknown), as ``Constraints.intercepts``;
    ``traces`` the desurveyed holes. ``search_distance`` (default 2.5 x the median distance
    from a hole to its nearest hole) limits the neighbouring holes; ``step`` is the
    down-hole sampling step (default a quarter of the median in-domain run, at least
    0.5 m, coarser when there would be more than 20 000 samples). ``smoothing`` is the
    length in metres of a moving average of the indicator down the hole (0: none).

    Always returns a :class:`PlaneEstimate`; ``determined`` is False (with ``reason``)
    when the logging does not fix one plane. The caller must not use the orientation then.
    """
    from scipy.spatial import cKDTree

    hole_ids = runs["hole_id"].to_numpy(dtype=object)
    frm = runs["depth_from"].to_numpy(dtype=np.float64)
    to = runs["depth_to"].to_numpy(dtype=np.float64)
    inside = runs["state"].to_numpy(dtype=np.float64) > 0

    def failed(reason: str, n_holes: int = 0) -> PlaneEstimate:
        nan = float("nan")
        return PlaneEstimate(
            np.array([0.0, 0.0, 1.0]), nan, nan, nan, nan, 0.0, False, 0, n_holes, nan, nan,
            reason=reason, warnings=[reason],
        )  # fmt: skip

    if not inside.any():
        return failed("there is no in-domain intercept")
    ore_lengths = (to - frm)[inside]
    total = float((to - frm).sum())
    if step is None:
        step = max(0.25 * float(np.median(ore_lengths)), 0.5)
        step = max(step, total / _MAX_SAMPLES)
    if not step > 0:
        raise ValueError("step must be > 0")

    # samples: every hole from its first to its last logged depth at the step
    codes, uniques = _factorize(hole_ids)
    xyz_parts, dir_parts, hole_parts, state_parts = [], [], [], []
    for h in range(len(uniques)):
        rows = np.flatnonzero(codes == h)
        order = rows[np.argsort(frm[rows], kind="stable")]
        start, end = float(frm[order].min()), float(to[order].max())
        m = int(math.floor((end - start) / step))
        if m < 1:
            continue
        depth = start + (np.arange(m) + 0.5) * step
        which = np.searchsorted(frm[order], depth, side="right") - 1
        covered = (which >= 0) & (depth <= to[order][np.maximum(which, 0)])
        state = np.where(covered, inside[order][np.maximum(which, 0)].astype(np.int8), -1)
        xyz, direction = traces.xyz_at(uniques[h], depth)
        good = np.isfinite(xyz).all(axis=1) & np.isfinite(direction).all(axis=1)
        if not good.all():
            if not good.any():
                continue
            state = np.where(good, state, -1)
            xyz = np.where(good[:, None], xyz, xyz[good][0])
            direction = np.where(good[:, None], direction, direction[good][0])
        xyz_parts.append(xyz)
        dir_parts.append(direction)
        hole_parts.append(np.full(m, len(xyz_parts) - 1, dtype=np.int64))
        state_parts.append(state.astype(np.int8))
    n_holes = len(xyz_parts)
    if n_holes < 3:
        return failed("fewer than three holes: the plane is not determined", n_holes)
    xyz = np.vstack(xyz_parts)
    direction = np.vstack(dir_parts)
    hole = np.concatenate(hole_parts)
    state = np.concatenate(state_parts)

    # hole spacing: median over holes of the median distance to the nearest other hole
    nearest = np.full(len(xyz), np.nan)
    for h in range(n_holes):
        mine = hole == h
        other = cKDTree(xyz[~mine])
        nearest[mine] = other.query(xyz[mine])[0]
    spacing = float(np.median([np.median(nearest[hole == h]) for h in range(n_holes)]))
    if search_distance is None:
        search_distance = 2.5 * spacing
    if not (np.isfinite(search_distance) and search_distance > 0):
        raise ValueError("search_distance must be > 0")

    value = _smooth(state, hole, n_holes, int(round(smoothing / step)))
    pairs = _Pairs(
        xyz, direction, hole, state, value, step, float(search_distance), max(spacing, step)
    )
    if pairs.n and pairs.fence_ratio < 0.01:
        return failed(
            "the neighbouring holes lie on one line (a single fence): the plane of the "
            "bodies is not determined by the drilling",
            n_holes,
        )
    coarse = _hemisphere(COARSE_STEP_DEG)
    rho, count = pairs.score(coarse)
    enough = max(MIN_PAIRS, int(0.25 * count.max())) if count.size else MIN_PAIRS
    usable = np.isfinite(rho) & (count >= enough)
    if not usable.any():
        return failed(
            "too few neighbouring holes with logged intervals to compare "
            f"(search distance {search_distance:.0f} m)",
            n_holes,
        )
    rho_usable = np.where(usable, rho, -np.inf)
    best = coarse[int(np.argmax(rho_usable))]
    fine = _around(best, 1.5 * COARSE_STEP_DEG, 1.0)
    rho_fine, count_fine = pairs.score(fine)
    rho_fine = np.where(np.isfinite(rho_fine) & (count_fine >= enough), rho_fine, -np.inf)
    k = int(np.argmax(rho_fine))
    if rho_fine[k] >= rho_usable.max():
        pole, correlation, n_pairs = fine[k], float(rho_fine[k]), int(count_fine[k])
    else:
        k = int(np.argmax(rho_usable))
        pole, correlation, n_pairs = best, float(rho_usable[k]), int(count[k])
    if pole[2] < 0:
        pole = -pole
    away = usable & (_angle(coarse, pole) > EXCLUSION_DEG)
    runner_up = float(rho_usable[away].max()) if away.any() else 0.0
    confidence = 0.0
    if correlation > 0:
        confidence = float(np.clip(1.0 - max(runner_up, 0.0) / correlation, 0.0, 1.0))
    dip, dip_azimuth = _plane_of_pole(pole)
    candidates = np.vstack([coarse[usable], fine[np.isfinite(rho_fine)]])
    stability = _stability(pairs, candidates, pole, n_holes)

    in_dirs = direction[state > 0]
    mean_dir = np.linalg.eigh(in_dirs.T @ in_dirs)[1][:, 2]
    hole_angle = float(_angle(pole[None, :], mean_dir)[0])
    # mean true thickness of the intercepts
    mid_xyz, mid_dir = traces.xyz_at(hole_ids[inside], 0.5 * (frm + to)[inside])
    ok = np.isfinite(mid_dir).all(axis=1)
    thickness = float(np.mean(ore_lengths[ok] * np.abs(mid_dir[ok] @ pole))) if ok.any() else np.nan

    reason = ""
    if correlation < MIN_CORRELATION:
        reason = (
            f"the intercepts of neighbouring holes do not line up on any plane (best "
            f"indicator correlation {correlation:.2f}, needed {MIN_CORRELATION:.2f})"
        )
    elif confidence < MIN_CONFIDENCE:
        reason = (
            f"several different planes explain the intercepts almost equally well "
            f"(confidence {confidence:.2f}, needed {MIN_CONFIDENCE:.2f}; best plane dips "
            f"{dip:.0f} towards {dip_azimuth:03.0f})"
        )
    elif stability < MIN_STABILITY:
        reason = (
            f"the best plane (dips {dip:.0f} towards {dip_azimuth:03.0f}) is not stable: "
            f"with the holes resampled it stays within {EXCLUSION_DEG:.0f} degrees in only "
            f"{100 * stability:.0f} % of {STABILITY_RESAMPLES} draws (needed "
            f"{100 * MIN_STABILITY:.0f} %); the intercepts do not line up clearly enough "
            "between holes"
        )
    if reason and hole_angle < 15.0:
        reason += (
            "; that plane is perpendicular to the holes, which is what equal depths in "
            "neighbouring holes give when the bodies are not followed from hole to hole"
        )
    return PlaneEstimate(
        pole=pole,
        dip=dip,
        dip_azimuth=dip_azimuth,
        correlation=correlation,
        runner_up=runner_up,
        confidence=confidence,
        determined=not reason,
        n_pairs=n_pairs,
        n_holes=n_holes,
        search_distance=float(search_distance),
        spacing=spacing,
        pair_distance=pairs.in_plane_distance(pole),
        thickness=thickness,
        hole_angle_deg=hole_angle,
        stability=stability,
        reason=reason,
        warnings=[reason] if reason else [],
    )


def _stability(pairs: _Pairs, candidates: np.ndarray, pole: np.ndarray, n_holes: int) -> float:
    """Share of bootstrap resamplings of the holes whose best candidate pole is within
    EXCLUSION_DEG of ``pole``. A hole is drawn with replacement; each draw counts its
    groups (the hole and one neighbour, in one sector) once more. Fixed seed: the result
    is repeatable."""
    if len(candidates) == 0 or len(pairs.group_hole) == 0:
        return 0.0
    rng = np.random.default_rng(0)
    drawn = rng.integers(0, n_holes, size=(STABILITY_RESAMPLES, n_holes))
    per_hole = np.stack([np.bincount(d, minlength=n_holes) for d in drawn]).astype(np.float64)
    scores = pairs.resampled_scores(candidates, per_hole[:, pairs.group_hole])
    best = candidates[np.argmax(np.where(np.isfinite(scores), scores, -np.inf), axis=1)]
    return float(np.mean(_angle(best, pole) <= EXCLUSION_DEG))


def _smooth(state: np.ndarray, hole: np.ndarray, n_holes: int, width: int) -> np.ndarray:
    """Moving average of the indicator down each hole over ``width`` samples (unknown skipped)."""
    value = np.where(state > 0, 1.0, 0.0)
    if width < 2:
        return value
    known = (state >= 0).astype(np.float64)
    kernel = np.ones(width if width % 2 else width + 1)
    out = value.copy()
    for h in range(n_holes):
        mine = hole == h
        total = np.convolve(value[mine] * known[mine], kernel, mode="same")
        weight = np.convolve(known[mine], kernel, mode="same")
        out[mine] = np.where(weight > 0, total / np.maximum(weight, 1e-12), 0.0)
    return out


def _factorize(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    import pandas as pd

    codes, uniques = pd.factorize(values, sort=True)
    return codes, np.asarray(uniques, dtype=object)
