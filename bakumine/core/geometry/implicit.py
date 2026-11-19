"""Implicit solid of a logged domain, built directly from drillholes.

The published idea (Carr et al. 2001, "Reconstruction and representation of 3D
objects with radial basis functions"; Cowan et al. 2003, "Practical implicit
geological modelling"): describe the domain by a scalar field that is positive
inside it, negative outside and zero on its boundary; fix the field at points
along the drillholes; interpolate it everywhere with a radial basis function;
triangulate the zero level.

Steps, each usable on its own:

1. :func:`extract_constraints` - contact points (value 0) where the logged code
   changes between the target and anything else, and sample points along every
   hole valued by the signed down-hole distance to the nearest contact of that
   hole (positive in the domain, capped). Holes that never meet the domain give
   negative points only; they are what bounds the solid sideways.
2. :func:`fit_field` - exact (or smoothed) RBF interpolation in the *isotropic
   space* of an :class:`~bakumine.core.estimation.rotation.Ellipsoid`, i.e.
   after rotating into the anisotropy axes and stretching the short axes.
   :func:`estimate_anisotropy` proposes the ellipsoid from the data.
3. Evaluation on a regular grid aligned with the anisotropy axes, in bounded
   chunks, optionally clipped by a topography DTM and by a maximum distance
   from the nearest hole.
4. Marching tetrahedra (:mod:`bakumine.core.geometry.isosurface`) at level 0.

:func:`implicit_solid` runs all four.

Design choices
--------------
*Kernel.* The default is the linear kernel ``phi(r) = r`` (the biharmonic
spline in 3-D). It is the usual choice for signed-distance data: the
interpolant has no free shape parameter, does not overshoot between points
the way the cubic kernel does (which creates spurious shells), and for data
that is a distance function it reproduces the natural unit gradient. Cubic
``r^3`` and thin-plate ``r^2 log r`` are available; they need a linear drift.

*Grid frame.* The grid is aligned with the anisotropy axes, not the world
axes. A thin dipping body needs metre-scale resolution across it and can be
much coarser along it; only a grid whose third axis is the minor axis can have
such cells. The cell is ``resolution`` across the minor axis and longer along
the other two by the anisotropy ratios, limited by ``max_cell_stretch``. With
no anisotropy the grid is world-aligned with cubic cells.

*Numerics.* Everything is computed relative to a local origin (the centroid
of the constraints) and scaled to unit size before the linear solve.

Coordinates are x = east, y = north, z = up.
"""

from __future__ import annotations

import time
import warnings
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numba import njit, prange
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

from bakumine.core.drillholes.desurvey import HoleTraces, desurvey
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.isosurface import (
    GridSpec,
    marching_tetrahedra,
    mesh_components,
    remove_small_components,
)
from bakumine.core.geometry.mesh import Mesh

__all__ = [
    "KERNELS",
    "Constraints",
    "ImplicitField",
    "ImplicitResult",
    "estimate_anisotropy",
    "extract_constraints",
    "fit_field",
    "implicit_solid",
]

KERNELS = ("linear", "cubic", "thin_plate")
_KERNEL_ID = {"linear": 0, "cubic": 1, "thin_plate": 2}
_MIN_DEGREE = {"linear": 0, "cubic": 1, "thin_plate": 1}
_SCIPY_KERNEL = {"linear": "linear", "cubic": "cubic", "thin_plate": "thin_plate_spline"}

CONTACT = 0
INSIDE = 1
OUTSIDE = -1


# --------------------------------------------------------------------------
# 1. constraints
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Constraints:
    """Interpolation constraints taken from the holes.

    Attributes
    ----------
    xyz : (n, 3) world coordinates.
    value : (n,) signed distance value: 0 at contacts, > 0 in the domain,
        < 0 outside, ``|value| <= cap``.
    kind : (n,) int8: ``CONTACT`` (0), ``INSIDE`` (1) or ``OUTSIDE`` (-1).
    hole_id : (n,) object.
    depth : (n,) down-hole depth of the point.
    cap : the cap applied to ``|value|``.
    sample_spacing : the spacing of the off-contact points within ``cap`` of a contact.
    far_spacing : the spacing of the points further from every contact.
    n_holes : holes that contributed at least one point.
    n_holes_intersecting : holes with at least one in-domain interval.
    skipped_holes : hole ids in the table that have no desurveyed trace.
    intercepts : table ``hole_id, depth_from, depth_to, state`` of the merged
        runs (state 1 = in the domain, -1 = outside).
    """

    xyz: np.ndarray
    value: np.ndarray
    kind: np.ndarray
    hole_id: np.ndarray
    depth: np.ndarray
    cap: float
    sample_spacing: float
    far_spacing: float
    n_holes: int
    n_holes_intersecting: int
    skipped_holes: tuple = ()
    intercepts: pd.DataFrame = field(default=None, repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.value)

    @property
    def n_contacts(self) -> int:
        return int((self.kind == CONTACT).sum())

    @property
    def n_inside(self) -> int:
        return int((self.kind == INSIDE).sum())

    @property
    def n_outside(self) -> int:
        return int((self.kind == OUTSIDE).sum())

    def frame(self) -> pd.DataFrame:
        """The constraints as a table: ``hole_id, depth, x, y, z, value, kind``."""
        names = np.array(["outside", "contact", "inside"], dtype=object)
        return pd.DataFrame(
            {
                "hole_id": self.hole_id,
                "depth": self.depth,
                "x": self.xyz[:, 0],
                "y": self.xyz[:, 1],
                "z": self.xyz[:, 2],
                "value": self.value,
                "kind": names[self.kind.astype(np.int64) + 1],
            }
        )


def _codes(code) -> set:
    if isinstance(code, (str, bytes)) or not hasattr(code, "__iter__"):
        return {code}
    return set(code)


def _hole_runs(depth_from, depth_to, state, max_depth, tol):
    """Merge one hole's intervals (sorted by depth_from) into runs of constant state.

    Unknown intervals (state 0) and unlogged gaps separate runs. Returns a list
    of ``[from, to, state]`` and, for each run but the first, whether it abuts
    the previous one.
    """
    runs: list[list[float]] = []
    abuts: list[bool] = []
    reach = -np.inf  # deepest depth already covered, known or not
    for a, b, s in zip(depth_from, depth_to, state, strict=True):
        a = max(float(a), 0.0, reach)  # an overlap keeps the interval logged first
        b = min(float(b), max_depth)
        if not b - a > tol:
            continue
        touching = a - reach <= tol
        reach = b
        if s == 0:
            continue
        if runs and touching and runs[-1][1] >= a - tol and runs[-1][2] == s:
            runs[-1][1] = b
            continue
        abuts.append(bool(runs) and touching and runs[-1][1] >= a - tol)
        runs.append([a, b, float(s)])
    return runs, abuts


def _sample_depths(a, b, contacts, cap, near_spacing, far_spacing) -> np.ndarray:
    """Depths of the off-contact points of the run ``[a, b]``.

    The run is cut where the distance to the nearest contact of the hole
    equals ``cap``; a stretch nearer than that is sampled at ``near_spacing``,
    a stretch further away at ``far_spacing``.
    """
    cuts = [a, b]
    if len(contacts):
        marks = np.concatenate([contacts - cap, contacts + cap])
        cuts += marks[(marks > a) & (marks < b)].tolist()
    cuts = np.unique(cuts)
    out = []
    for lo, hi in zip(cuts[:-1], cuts[1:], strict=True):
        if not hi - lo > 1e-9:
            continue
        near = len(contacts) and np.abs(contacts - 0.5 * (lo + hi)).min() < cap
        spacing = near_spacing if near else far_spacing
        n = max(1, int(np.ceil((hi - lo) / spacing - 1e-9)))
        out.append(lo + (np.arange(n) + 0.5) * ((hi - lo) / n))
    return np.concatenate(out) if out else np.array([0.5 * (a + b)])


def extract_constraints(
    table: pd.DataFrame,
    column: str,
    code,
    traces: HoleTraces,
    *,
    sample_spacing: float | None = None,
    far_spacing: float | None = None,
    cap: float | None = None,
    thin: float | None = None,
    depth_tolerance: float = 1e-6,
) -> Constraints:
    """Contacts and signed-distance points of one domain along the holes.

    Parameters
    ----------
    table : interval table with ``hole_id, depth_from, depth_to`` and ``column``.
    column : the categorical column.
    code : the target code, or a collection of codes that together form the domain.
    traces : desurveyed holes (``DrillholeDatabase.desurvey()``).
    sample_spacing : spacing of the off-contact points along a hole where the
        value varies, i.e. within ``cap`` (down-hole) of a contact. Each such
        stretch is cut into equal pieces no longer than this, with one point
        at the middle of each piece, so even the thinnest intercept gets a
        point. Default: a quarter of the median in-domain intercept length,
        but at most a tenth of the median logged hole length. The points
        along a hole must be this dense because the anisotropy stretches the
        space across a thin body: two points 10 m apart down a hole that
        crosses a body with a 16:1 anisotropy are as far apart, for the
        interpolant, as two holes 160 m apart.
    far_spacing : spacing further than ``cap`` from every contact, where the
        value is the constant ``+-cap`` (this includes barren holes). Default:
        a tenth of the median logged hole length; never below ``sample_spacing``.
    cap : largest ``|value|``. Default: the median in-domain intercept
        length. Inside the domain the value cannot exceed half an intercept,
        so this keeps the two sides of a contact on the same scale; without
        it the large negative values of distant waste outweigh a thin domain
        and pinch it out between the holes (down-hole distance is not a true
        distance to the contact anyway once it exceeds the body's thickness).
        It is also the value given to holes without any contact.
    thin : optional decimation: keep at most one off-contact point per cube of
        this side (the one closest to a contact). Contacts are always kept.
    depth_tolerance : two intervals abut when the gap between them is at most this.

    Rules
    -----
    * A contact is created only where an in-domain run and an out-of-domain
      run abut. The collar and the end of a hole are not contacts: a hole that
      starts or ends in the domain leaves the domain open there.
    * A missing value in ``column`` and an unlogged gap are unknown: no points
      are placed there and no contact is created at their ends.
    * The value of a point is the down-hole distance to the nearest contact of
      its own hole, signed and capped. A hole without contacts gives ``-cap``
      (barren) or ``+cap`` (entirely in the domain) at every point.
    * Holes without a trace are skipped and listed in ``skipped_holes``.
    """
    for name in ("hole_id", "depth_from", "depth_to", column):
        if name not in table.columns:
            raise ValueError(f"interval table has no column {name!r}")
    limits = (
        ("sample_spacing", sample_spacing),
        ("far_spacing", far_spacing),
        ("cap", cap),
        ("thin", thin),
    )
    for name, v in limits:
        if v is not None and not (np.isfinite(v) and v > 0.0):
            raise ValueError(f"{name} must be a positive finite number, got {v!r}")
    codes = _codes(code)
    values = table[column]
    known = values.notna().to_numpy()
    state_all = np.where(values.isin(codes).to_numpy(), 1, -1) * known
    d_from = pd.to_numeric(table["depth_from"], errors="coerce").to_numpy(dtype=np.float64)
    d_to = pd.to_numeric(table["depth_to"], errors="coerce").to_numpy(dtype=np.float64)
    ok = np.isfinite(d_from) & np.isfinite(d_to)
    holes = table["hole_id"].to_numpy(dtype=object)
    max_depth = dict(zip(traces.hole_ids.tolist(), traces.max_depth.tolist(), strict=True))

    per_hole: list[tuple[object, list, list]] = []
    skipped = []
    groups = pd.Series(np.arange(len(table))[ok]).groupby(holes[ok], sort=False)
    for hole, rows in groups:
        if hole not in max_depth:
            skipped.append(hole)
            continue
        idx = rows.to_numpy()
        idx = idx[np.argsort(d_from[idx], kind="stable")]
        runs, abuts = _hole_runs(
            d_from[idx], d_to[idx], state_all[idx], max_depth[hole], depth_tolerance
        )
        if runs:
            per_hole.append((hole, runs, abuts))

    in_lengths = [r[1] - r[0] for _, runs, _ in per_hole for r in runs if r[2] > 0]
    logged = [sum(r[1] - r[0] for r in runs) for _, runs, _ in per_hole]
    if sample_spacing is None:
        if not per_hole:
            sample_spacing = 1.0
        else:
            sample_spacing = float(np.median(logged)) / 10.0
            if in_lengths:
                sample_spacing = min(sample_spacing, 0.25 * float(np.median(in_lengths)))
    if cap is None:
        cap = float(np.median(in_lengths)) if in_lengths else 5.0 * sample_spacing
    if far_spacing is None:
        far_spacing = float(np.median(logged)) / 10.0 if per_hole else sample_spacing
    far_spacing = max(float(far_spacing), float(sample_spacing))

    out_hole, out_depth, out_value, out_kind = [], [], [], []
    run_rows = []
    for hole, runs, abuts in per_hole:
        contacts = np.array(
            [runs[i][0] for i in range(1, len(runs)) if abuts[i] and runs[i][2] != runs[i - 1][2]]
        )
        for a, b, s in runs:
            run_rows.append((hole, a, b, int(s)))
            depth = _sample_depths(a, b, contacts, cap, sample_spacing, far_spacing)
            n = len(depth)
            if contacts.size:
                dist = np.abs(depth[:, None] - contacts[None, :]).min(axis=1)
                value = s * np.minimum(dist, cap)
            else:
                value = np.full(n, s * cap)
            out_hole.append(np.full(n, hole, dtype=object))
            out_depth.append(depth)
            out_value.append(value)
            out_kind.append(np.full(n, INSIDE if s > 0 else OUTSIDE, dtype=np.int8))
        if contacts.size:
            out_hole.append(np.full(contacts.size, hole, dtype=object))
            out_depth.append(contacts)
            out_value.append(np.zeros(contacts.size))
            out_kind.append(np.full(contacts.size, CONTACT, dtype=np.int8))

    if out_hole:
        hole_id = np.concatenate(out_hole)
        depth = np.concatenate(out_depth)
        value = np.concatenate(out_value)
        kind = np.concatenate(out_kind)
        xyz, _ = traces.xyz_at(hole_id, depth, depth_tolerance=max(depth_tolerance, 1e-9))
        good = np.isfinite(xyz).all(axis=1)
        hole_id, depth, value, kind, xyz = (
            hole_id[good],
            depth[good],
            value[good],
            kind[good],
            xyz[good],
        )
    else:
        hole_id = np.zeros(0, dtype=object)
        depth, value, kind, xyz = np.zeros(0), np.zeros(0), np.zeros(0, np.int8), np.zeros((0, 3))

    if thin is not None and len(value):
        keep = kind == CONTACT
        off = np.flatnonzero(~keep)
        if off.size:
            cell = np.floor((xyz[off] - xyz[off].min(axis=0)) / thin).astype(np.int64)
            order = np.lexsort((np.abs(value[off]), cell[:, 2], cell[:, 1], cell[:, 0]))
            sorted_cell = cell[order]
            first = np.ones(off.size, dtype=bool)
            first[1:] = (sorted_cell[1:] != sorted_cell[:-1]).any(axis=1)
            keep[off[order[first]]] = True
        hole_id, depth, value, kind, xyz = (
            hole_id[keep],
            depth[keep],
            value[keep],
            kind[keep],
            xyz[keep],
        )

    intercepts = pd.DataFrame(run_rows, columns=["hole_id", "depth_from", "depth_to", "state"])
    return Constraints(
        xyz=xyz,
        value=value,
        kind=kind,
        hole_id=hole_id,
        depth=depth,
        cap=float(cap),
        sample_spacing=float(sample_spacing),
        far_spacing=float(far_spacing),
        n_holes=len(per_hole),
        n_holes_intersecting=sum(any(r[2] > 0 for r in runs) for _, runs, _ in per_hole),
        skipped_holes=tuple(skipped),
        intercepts=intercepts,
    )


# --------------------------------------------------------------------------
# 2. anisotropy
# --------------------------------------------------------------------------


def estimate_anisotropy(
    points, weights=None, *, max_ratio: float = 20.0, min_points: int = 4
) -> Ellipsoid:
    """Orientation and axis ratios of a point cloud by principal components.

    The eigenvectors of the (weighted) covariance of ``points`` give the
    major, semi-major and minor axes in order of decreasing variance; the
    ratios are the ratios of the standard deviations, limited to
    ``[1, max_ratio]``. For points spread uniformly through an ellipsoid the
    standard deviation along an axis is ``semi-axis / sqrt(5)``, so the ratios
    are those of the body and ``major_radius`` is set to ``sqrt(5) * sd``.

    Used with the in-domain constraint points this is a serviceable default
    for a single tabular or lens-shaped body. It under-estimates the
    major/minor ratio when the body is curved (the sag adds to the variance
    across it) and is meaningless for a folded or multi-limbed domain: supply
    the anisotropy in those cases.
    """
    p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(p) < min_points:
        raise ValueError(f"need at least {min_points} points to estimate an anisotropy")
    w = np.ones(len(p)) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != (len(p),) or (w < 0).any() or not w.sum() > 0:
        raise ValueError("weights must be non-negative, one per point, not all zero")
    centre = (p * w[:, None]).sum(axis=0) / w.sum()
    q = p - centre
    cov = (q * w[:, None]).T @ q / w.sum()
    lam, vec = np.linalg.eigh(cov)  # ascending
    lam = np.maximum(lam, 0.0)
    if not lam[2] > 0.0:
        raise ValueError("points are coincident: no anisotropy can be estimated")
    sd = np.sqrt(lam)
    floor = sd[2] / max_ratio
    semi_ratio = float(sd[2] / max(sd[1], floor))
    minor_ratio = float(sd[2] / max(sd[0], floor))
    return Ellipsoid.from_axes(
        vec[:, 2],
        vec[:, 1],
        major_radius=float(np.sqrt(5.0) * sd[2]),
        semi_major_ratio=max(semi_ratio, 1.0),
        minor_ratio=max(minor_ratio, 1.0),
    )


# --------------------------------------------------------------------------
# 3. RBF field
# --------------------------------------------------------------------------


@njit(parallel=True, cache=True)
def _rbf_sum(query, nodes, weights, poly, kernel):
    """sum_j w_j phi(|q - x_j|) + poly(q); poly = (c0, cx, cy, cz)."""
    n = query.shape[0]
    m = nodes.shape[0]
    out = np.empty(n)
    for i in prange(n):
        qx, qy, qz = query[i, 0], query[i, 1], query[i, 2]
        total = poly[0] + poly[1] * qx + poly[2] * qy + poly[3] * qz
        for j in range(m):
            dx = qx - nodes[j, 0]
            dy = qy - nodes[j, 1]
            dz = qz - nodes[j, 2]
            r2 = dx * dx + dy * dy + dz * dz
            if kernel == 0:
                total += weights[j] * np.sqrt(r2)
            elif kernel == 1:
                total += weights[j] * r2 * np.sqrt(r2)
            elif r2 > 0.0:
                total += weights[j] * 0.5 * r2 * np.log(r2)
        out[i] = total
    return out


def _kernel_matrix(nodes: np.ndarray, kernel: str) -> np.ndarray:
    r = cdist(nodes, nodes)
    if kernel == "linear":
        return r
    if kernel == "cubic":
        return r**3
    with np.errstate(divide="ignore", invalid="ignore"):
        k = r * r * np.log(r)
    k[r == 0.0] = 0.0
    return k


class ImplicitField:
    """The fitted signed-distance field: positive inside the domain.

    Call it with world points ``(..., 3)`` to get the field; :meth:`inside`
    gives the boolean (usable to code block centroids directly). With
    ``clipped=True`` (default) the topography and extrapolation limits used
    for the solid are applied, so ``inside`` agrees with the mesh up to the
    grid resolution.
    """

    def __init__(
        self,
        *,
        centre: np.ndarray,
        anisotropy: Ellipsoid | None,
        length: float,
        nodes: np.ndarray,
        kernel: str,
        degree: int,
        smoothing: float,
        weights: np.ndarray | None,
        poly: np.ndarray | None,
        local: RBFInterpolator | None,
        neighbors: int | None,
        topography: DTM | None = None,
        max_extrapolation: float | None = None,
        hole_points: np.ndarray | None = None,
    ):
        self.centre = centre
        self.anisotropy = anisotropy
        self.length = length
        self.nodes = nodes
        self.kernel = kernel
        self.degree = degree
        self.smoothing = smoothing
        self.weights = weights
        self.poly = poly
        self.neighbors = neighbors
        self.topography = topography
        self.max_extrapolation = max_extrapolation
        self._local = local
        self._hole_tree = None
        if max_extrapolation is not None:
            if hole_points is None or len(hole_points) == 0:
                raise ValueError("max_extrapolation needs the hole points")
            self._hole_tree = cKDTree(np.asarray(hole_points, dtype=np.float64) - centre)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    def _to_model(self, points: np.ndarray) -> np.ndarray:
        """World points (n, 3) -> scaled isotropic coordinates."""
        local = points - self.centre
        if self.anisotropy is not None:
            local = self.anisotropy.to_isotropic(local)
        return np.ascontiguousarray(local / self.length)

    def _raw(self, flat: np.ndarray, chunk: int) -> np.ndarray:
        out = np.empty(len(flat))
        bounds = [(a, min(a + chunk, len(flat))) for a in range(0, len(flat), chunk)]
        if self._local is None:
            kid = _KERNEL_ID[self.kernel]
            for a, b in bounds:
                out[a:b] = _rbf_sum(
                    self._to_model(flat[a:b]), self.nodes, self.weights, self.poly, kid
                )
            return out

        def work(ab):
            a, b = ab
            out[a:b] = self._local(self._to_model(flat[a:b]))

        if len(bounds) == 1:
            work(bounds[0])
        else:
            with ThreadPoolExecutor() as pool:
                list(pool.map(work, bounds))
        return out

    def __call__(self, points, *, clipped: bool = True, chunk: int = 200_000) -> np.ndarray:
        """Field at world points ``(..., 3)``. NaN coordinates give NaN.

        Work is done ``chunk`` points at a time (each chunk in parallel), so
        memory does not grow with the number of constraints.
        """
        pts = np.asarray(points, dtype=np.float64)
        if pts.shape[-1:] != (3,):
            raise ValueError(f"points must have shape (..., 3), got {pts.shape}")
        flat = pts.reshape(-1, 3)
        good = np.isfinite(flat).all(axis=1)
        out = np.full(len(flat), np.nan)
        if good.any():
            sel = flat[good]
            value = self._raw(sel, int(chunk))
            if clipped:
                value = self._clip(sel, value)
            out[good] = value
        return out.reshape(pts.shape[:-1])

    def _clip(self, flat: np.ndarray, value: np.ndarray) -> np.ndarray:
        if self.topography is not None:
            depth = -self.topography.height_above(flat)  # positive below the surface
            value = np.where(np.isfinite(depth), np.minimum(value, depth), value)
        if self._hole_tree is not None:
            dist, _ = self._hole_tree.query(flat - self.centre, workers=-1)
            value = np.minimum(value, self.max_extrapolation - dist)
        return value

    def inside(self, points, *, clipped: bool = True) -> np.ndarray:
        """True where the field is positive (NaN points are outside)."""
        with np.errstate(invalid="ignore"):
            return self(points, clipped=clipped) > 0.0


def fit_field(
    xyz,
    values,
    *,
    anisotropy: Ellipsoid | None = None,
    kernel: str = "linear",
    degree: int | None = None,
    smoothing: float = 0.0,
    neighbors: int | None | str = "auto",
    global_limit: int = 6000,
    topography: DTM | None = None,
    max_extrapolation: float | None = None,
    hole_points=None,
) -> ImplicitField:
    """Interpolate scattered values with an RBF in anisotropic space.

    Parameters
    ----------
    xyz, values : (n, 3) world points and their values.
    anisotropy : distances are measured after ``Ellipsoid.to_isotropic``; only
        the orientation and the two ratios matter. None = isotropic.
    kernel : "linear" (``r``, default), "cubic" (``r^3``) or "thin_plate"
        (``r^2 log r``).
    degree : polynomial drift, 0 (constant) or 1 (linear). Default 1. The
        cubic and thin-plate kernels need 1.
    smoothing : 0 interpolates exactly; a positive value is added to the
        diagonal of the system and lets the surface miss the points (the unit
        is that of the kernel in coordinates scaled to the data extent).
    neighbors : None solves one global system (exact, O(n^3)); an integer
        uses, for every evaluation point, only its nearest ``neighbors``
        constraints (``scipy.interpolate.RBFInterpolator``), which stays fast
        for many points but is only piecewise continuous. "auto" (default):
        global up to ``global_limit`` points, 100 neighbours beyond.
    topography, max_extrapolation, hole_points : the clips applied by
        ``ImplicitField(..., clipped=True)``; see :func:`implicit_solid`.

    Coincident points are merged (their values averaged).
    """
    if kernel not in KERNELS:
        raise ValueError(f"kernel must be one of {KERNELS}, got {kernel!r}")
    if degree is None:
        degree = 1
    if degree not in (0, 1):
        raise ValueError("degree must be 0 or 1")
    if degree < _MIN_DEGREE[kernel]:
        raise ValueError(f"the {kernel} kernel needs a drift of degree {_MIN_DEGREE[kernel]}")
    if not (np.isfinite(smoothing) and smoothing >= 0.0):
        raise ValueError("smoothing must be >= 0")
    p = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if len(p) != len(v):
        raise ValueError("xyz and values differ in length")
    if not (np.isfinite(p).all() and np.isfinite(v).all()):
        raise ValueError("constraints contain non-finite coordinates or values")
    if len(p) < degree * 3 + 1:
        raise ValueError(f"need at least {degree * 3 + 1} constraint points, got {len(p)}")

    centre = p.mean(axis=0)
    local = p - centre
    iso = local if anisotropy is None else anisotropy.to_isotropic(local)
    length = float(np.abs(iso).max()) or 1.0
    nodes = iso / length
    nodes, inverse = np.unique(nodes, axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    # np.unique sorts the nodes: always carry the values over (mean of coincident points)
    v = np.bincount(inverse, weights=v) / np.bincount(inverse)
    nodes = np.ascontiguousarray(nodes)
    n = len(nodes)

    if neighbors == "auto":
        neighbors = None if n <= global_limit else 100
    if neighbors is not None:
        neighbors = int(min(neighbors, n))

    common = dict(
        centre=centre, anisotropy=anisotropy, length=length, nodes=nodes, kernel=kernel,
        degree=degree, smoothing=float(smoothing), neighbors=neighbors, topography=topography,
        max_extrapolation=max_extrapolation, hole_points=hole_points,
    )  # fmt: skip
    if neighbors is not None:
        local_rbf = RBFInterpolator(
            nodes, v, neighbors=neighbors, smoothing=smoothing,
            kernel=_SCIPY_KERNEL[kernel], degree=degree,
        )  # fmt: skip
        return ImplicitField(weights=None, poly=None, local=local_rbf, **common)

    n_poly = 1 + 3 * degree
    system = np.zeros((n + n_poly, n + n_poly))
    system[:n, :n] = _kernel_matrix(nodes, kernel)
    if smoothing:
        system[:n, :n] += smoothing * np.eye(n)
    system[:n, n] = system[n, :n] = 1.0
    if degree == 1:
        system[:n, n + 1 :] = nodes
        system[n + 1 :, :n] = nodes.T
    rhs = np.zeros(n + n_poly)
    rhs[:n] = v
    try:
        solution = np.linalg.solve(system, rhs)
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "the RBF system is singular: the constraint points are degenerate "
            "(for a linear drift they must not all lie in one plane)"
        ) from exc
    poly = np.zeros(4)
    poly[:n_poly] = solution[n:]
    return ImplicitField(weights=solution[:n].copy(), poly=poly, local=None, **common)


# --------------------------------------------------------------------------
# 4. the whole pipeline
# --------------------------------------------------------------------------


@dataclass
class ImplicitResult:
    """Output of :func:`implicit_solid`.

    ``mesh`` is the solid; ``field`` evaluates the fitted function anywhere
    (``result.evaluate(points)`` / ``result.inside(points)``); ``grid`` is the
    evaluation grid; ``grid_values`` its node values when ``keep_grid=True``.
    """

    mesh: Mesh
    constraints: Constraints
    anisotropy: Ellipsoid | None
    anisotropy_source: str
    grid: GridSpec
    field: ImplicitField
    diagnostics: dict[str, Any]
    grid_values: np.ndarray | None = None

    def evaluate(self, points, *, clipped: bool = True) -> np.ndarray:
        """Signed-distance field at world points (positive inside)."""
        return self.field(points, clipped=clipped)

    def inside(self, points, *, clipped: bool = True) -> np.ndarray:
        return self.field.inside(points, clipped=clipped)


def _resolve_source(source, table, traces):
    """-> (interval DataFrame, HoleTraces)."""
    if isinstance(table, pd.DataFrame):
        frame = table
    elif hasattr(source, "intervals"):
        if table not in source.intervals:
            raise KeyError(f"no interval table {table!r}; available: {sorted(source.intervals)}")
        frame = source.intervals[table]
    elif isinstance(source, Mapping):
        frame = source[table]
    else:
        raise TypeError("source must be a DrillholeDatabase or a mapping of tables")
    if traces is None:
        if hasattr(source, "desurvey"):
            traces = source.desurvey()
        elif isinstance(source, Mapping) and "collar" in source and "survey" in source:
            traces = desurvey(source["collar"], source["survey"])
        else:
            raise ValueError("pass traces=, or a source with collar and survey tables")
    return frame, traces


def _hole_points(constraints: Constraints, traces: HoleTraces, step: float) -> np.ndarray:
    """Points every ``<= step`` along the logged part of every hole."""
    runs = constraints.intercepts
    if len(runs) == 0:
        return np.zeros((0, 3))
    span = runs.groupby("hole_id", sort=False).agg(a=("depth_from", "min"), b=("depth_to", "max"))
    ids, depths = [], []
    for hole, a, b in zip(span.index, span["a"], span["b"], strict=True):
        n = max(1, int(np.ceil((b - a) / step)))
        depths.append(np.linspace(a, b, n + 1))
        ids.append(np.full(n + 1, hole, dtype=object))
    xyz, _ = traces.xyz_at(np.concatenate(ids), np.concatenate(depths))
    return xyz[np.isfinite(xyz).all(axis=1)]


def _typical_spacing(points: np.ndarray) -> float:
    """Median distance from each point to its nearest neighbour."""
    if len(points) < 2:
        return 0.0
    d, _ = cKDTree(points).query(points, k=2)
    return float(np.median(d[:, 1]))


def implicit_solid(
    source,
    table: str | pd.DataFrame = "geology",
    column: str = "rock",
    code=None,
    *,
    traces: HoleTraces | None = None,
    anisotropy: Ellipsoid | str | None = "auto",
    resolution: float | None = None,
    max_cell_stretch: float = 4.0,
    kernel: str = "linear",
    degree: int | None = None,
    smoothing: float = 0.0,
    neighbors: int | None | str = "auto",
    sample_spacing: float | None = None,
    far_spacing: float | None = None,
    cap: float | None = None,
    thin: float | None = None,
    max_extrapolation: float | None = None,
    margin: float | None = None,
    topography: DTM | Mesh | None = None,
    min_component_volume: float = 0.0,
    max_nodes: int = 4_000_000,
    chunk: int = 200_000,
    validate: bool = True,
    keep_grid: bool = False,
) -> ImplicitResult:
    """Closed solid of a logged domain from the drillholes alone.

    Parameters
    ----------
    source : a ``DrillholeDatabase``, or a mapping with ``collar``, ``survey``
        and the interval table. Ignored for the tables when ``table`` is a
        DataFrame and ``traces`` is given.
    table, column, code : the interval table (name or DataFrame), its
        categorical column and the target code (or a collection of codes).
    traces : desurveyed holes; default ``source.desurvey()``.
    anisotropy : an ``Ellipsoid`` (orientation and ratios; the radius is not
        used), ``"auto"`` (default: :func:`estimate_anisotropy` of the
        in-domain points) or ``None`` (isotropic). The most important
        parameter for a thin or elongated body: supply it when the
        orientation is known.
    resolution : grid cell size across the minor axis, in metres. Default: an
        eighth of the median in-domain intercept length. Coarsened
        automatically (with a warning) if the grid would exceed ``max_nodes``.
    max_cell_stretch : the cells are longer along the major and semi-major
        axes by the anisotropy ratios, but by no more than this factor.
    kernel, degree, smoothing, neighbors : see :func:`fit_field`.
    sample_spacing, far_spacing, cap, thin : see :func:`extract_constraints`.
    max_extrapolation : clip the solid to within this distance (3-D, metres)
        of the nearest logged hole, so it cannot balloon into undrilled
        ground. Default None: no clip. Like the topography clip it is applied
        to the node values, so the cut surface is accurate to about one grid
        cell (use a smaller ``max_cell_stretch`` for a sharper cut).
    margin : how far the grid extends beyond the box of the in-domain points
        (measured along the grid axes). Default ``max_extrapolation`` when
        given, else twice the typical spacing between ore intercepts. A solid
        that reaches the grid boundary is capped flat there and
        ``diagnostics["touches_grid_boundary"]`` is True.
    topography : a ``DTM`` (or its mesh); nothing above it is kept. Outside
        the plan extent of the surface there is no clipping.
    min_component_volume : remove disconnected shells (and cavities) smaller
        than this volume; the largest shell is always kept. Default 0: keep all.
    max_nodes, chunk : limits on the grid size and on the number of points
        evaluated at once.
    validate : run the full mesh validation (including self-intersection).
    keep_grid : keep the node values in the result.

    Returns
    -------
    ImplicitResult
    """
    t0 = time.perf_counter()
    if code is None:
        raise ValueError("code (the target domain) is required")
    frame, traces = _resolve_source(source, table, traces)
    if isinstance(topography, Mesh):
        topography = DTM(topography)
    constraints = extract_constraints(
        frame,
        column,
        code,
        traces,
        sample_spacing=sample_spacing,
        far_spacing=far_spacing,
        cap=cap,
        thin=thin,
    )
    domain = constraints.kind != OUTSIDE
    if constraints.n_inside == 0:
        raise ValueError(f"no interval of {column!r} has the code {code!r}: nothing to model")
    if constraints.n_outside == 0:
        raise ValueError("every logged interval is in the domain: there is no boundary to model")
    t1 = time.perf_counter()

    # anisotropy ---------------------------------------------------------
    if isinstance(anisotropy, str):
        if anisotropy != "auto":
            raise ValueError("anisotropy must be an Ellipsoid, 'auto' or None")
        source_name = "auto"
        if int(domain.sum()) >= 4:
            ellipsoid = estimate_anisotropy(constraints.xyz[domain])
        else:
            ellipsoid = None
            source_name = "none (too few in-domain points)"
    elif anisotropy is None:
        ellipsoid, source_name = None, "none"
    elif isinstance(anisotropy, Ellipsoid):
        ellipsoid, source_name = anisotropy, "user"
    else:
        raise TypeError("anisotropy must be an Ellipsoid, 'auto' or None")

    # field ----------------------------------------------------------------
    if max_extrapolation is not None and not (max_extrapolation > 0):
        raise ValueError("max_extrapolation must be > 0")
    hole_points = None
    if max_extrapolation is not None:
        step = min(constraints.sample_spacing, max_extrapolation / 4.0)
        hole_points = _hole_points(constraints, traces, step)
    fitted = fit_field(
        constraints.xyz,
        constraints.value,
        anisotropy=ellipsoid,
        kernel=kernel,
        degree=degree,
        smoothing=smoothing,
        neighbors=neighbors,
        topography=topography,
        max_extrapolation=max_extrapolation,
        hole_points=hole_points,
    )
    t2 = time.perf_counter()

    # grid -------------------------------------------------------------------
    in_lengths = constraints.intercepts.query("state > 0")
    in_lengths = (in_lengths["depth_to"] - in_lengths["depth_from"]).to_numpy()
    if resolution is None:
        resolution = float(np.median(in_lengths)) / 8.0
    if not (np.isfinite(resolution) and resolution > 0):
        raise ValueError("resolution must be > 0")
    if not max_cell_stretch >= 1.0:
        raise ValueError("max_cell_stretch must be >= 1")
    if ellipsoid is None:
        rotation = None
        axes = np.eye(3)
        stretch = np.ones(3)
    else:
        rotation = np.array(ellipsoid.rotation)
        axes = rotation
        stretch = np.array(
            [
                min(max(ellipsoid.minor_ratio / ellipsoid.semi_major_ratio, 1.0), max_cell_stretch),
                min(max(ellipsoid.minor_ratio, 1.0), max_cell_stretch),
                1.0,
            ]
        )
    centre = fitted.centre
    local = (constraints.xyz[domain] - centre) @ axes
    if margin is None:
        if max_extrapolation is not None:
            margin = float(max_extrapolation)
        else:
            mids = constraints.intercepts.query("state > 0")
            mid_xyz, _ = traces.xyz_at(
                mids["hole_id"].to_numpy(dtype=object),
                0.5 * (mids["depth_from"] + mids["depth_to"]).to_numpy(),
            )
            margin = 2.0 * _typical_spacing(mid_xyz[np.isfinite(mid_xyz).all(axis=1)])
            margin = max(margin, 4.0 * resolution, 2.0 * constraints.sample_spacing)
    if not (np.isfinite(margin) and margin >= 0):
        raise ValueError("margin must be >= 0")
    lo = local.min(axis=0) - margin
    hi = local.max(axis=0) + margin
    spacing = resolution * stretch
    shape = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
    coarsened = 1.0
    if int(np.prod(shape)) > max_nodes:
        base = spacing
        while int(np.prod(shape)) > max_nodes:
            coarsened *= max(1.02, float((np.prod(shape) / max_nodes) ** (1.0 / 3.0)))
            spacing = base * coarsened
            shape = np.ceil((hi - lo) / spacing).astype(np.int64) + 1
        warnings.warn(
            f"grid coarsened by {coarsened:.2f} to stay under max_nodes={max_nodes}; "
            f"resolution is now {spacing[2]:.3g}",
            stacklevel=2,
        )
    lo = 0.5 * (lo + hi) - 0.5 * (shape - 1) * spacing  # centre the grid on the box
    grid = GridSpec(centre + axes @ lo, spacing, tuple(int(s) for s in shape), rotation)

    # evaluation, slab by slab -------------------------------------------------
    nx, ny, nz = grid.shape
    values = np.empty(grid.shape)
    slab = max(1, int(chunk) // (ny * nz))
    for i0 in range(0, nx, slab):
        i1 = min(i0 + slab, nx)
        pts = grid.slab_points(i0, i1).reshape(-1, 3)
        values[i0:i1] = fitted(pts, clipped=True, chunk=chunk).reshape(i1 - i0, ny, nz)
    t3 = time.perf_counter()

    boundary = np.zeros(grid.shape, dtype=bool)
    boundary[[0, -1], :, :] = boundary[:, [0, -1], :] = boundary[:, :, [0, -1]] = True
    touches = bool((values[boundary] > 0.0).any())

    # isosurface ---------------------------------------------------------------
    mesh = marching_tetrahedra(values, grid.origin, grid.spacing, grid.rotation, inside="above")
    n_removed = 0
    if min_component_volume > 0.0 and mesh.n_faces:
        mesh, n_removed = remove_small_components(mesh, min_component_volume)
        mesh = mesh.remove_unreferenced_vertices()
    n_components, _, component_volume = mesh_components(mesh)
    t4 = time.perf_counter()

    # diagnostics --------------------------------------------------------------
    is_contact = constraints.kind == CONTACT
    if is_contact.any():
        residual = fitted(constraints.xyz[is_contact], clipped=False)
        contact_max = float(np.abs(residual).max())
        contact_rms = float(np.sqrt(np.mean(residual**2)))
    else:
        contact_max = contact_rms = float("nan")
    all_residual = fitted(constraints.xyz, clipped=False) - constraints.value
    report = None
    if validate and mesh.n_faces:
        report = mesh.validate()
    t5 = time.perf_counter()
    diagnostics: dict[str, Any] = {
        "n_constraints": len(constraints),
        "n_contacts": constraints.n_contacts,
        "n_inside_points": constraints.n_inside,
        "n_outside_points": constraints.n_outside,
        "n_rbf_nodes": fitted.n_nodes,
        "n_holes": constraints.n_holes,
        "n_holes_intersecting": constraints.n_holes_intersecting,
        "skipped_holes": list(constraints.skipped_holes),
        "sample_spacing": constraints.sample_spacing,
        "far_spacing": constraints.far_spacing,
        "cap": constraints.cap,
        "kernel": kernel,
        "degree": fitted.degree,
        "smoothing": float(smoothing),
        "neighbors": fitted.neighbors,
        "anisotropy_source": source_name,
        "anisotropy": None if ellipsoid is None else ellipsoid.to_dict(),
        "contact_residual_max": contact_max,
        "contact_residual_rms": contact_rms,
        "constraint_residual_max": float(np.abs(all_residual).max()),
        "resolution": float(spacing[2]),
        "grid": grid.to_dict(),
        "grid_nodes": grid.n_nodes,
        "grid_coarsening": coarsened,
        "margin": float(margin),
        "max_extrapolation": max_extrapolation,
        "touches_grid_boundary": touches,
        "n_vertices": mesh.n_vertices,
        "n_faces": mesh.n_faces,
        "n_components": n_components,
        "component_volumes": np.sort(component_volume)[::-1].tolist(),
        "n_components_removed": n_removed,
        "volume": float(mesh.volume) if mesh.n_faces else 0.0,
        "mesh_valid": None if report is None else bool(report.is_valid),
        "mesh_outward": None if report is None else bool(report.is_outward),
        "mesh_issues": None if report is None else report.issues(),
        "seconds": {
            "constraints": t1 - t0,
            "fit": t2 - t1,
            "grid_evaluation": t3 - t2,
            "isosurface": t4 - t3,
            "diagnostics": t5 - t4,
            "total": t5 - t0,
        },
    }
    return ImplicitResult(
        mesh=mesh,
        constraints=constraints,
        anisotropy=ellipsoid,
        anisotropy_source=source_name,
        grid=grid,
        field=fitted,
        diagnostics=diagnostics,
        grid_values=values if keep_grid else None,
    )
