"""2-D interpolation with exact values and inequality data (research.md 5.7.4).

Interpolant
-----------
One global solve gives a function of the plan position ``x = (east, north)``::

    f(x) = sum_j b_j k(x, x_j) + sum_a c_a p_a(u),        u = (x - origin) / scale

``x_j`` are the exact points and the *active* bound points. The drift monomials
``p`` are ``[1]`` (``drift="constant"``) or ``[1, u_x, u_y]`` (``"linear"``).
They are written in the shifted and scaled coordinates ``u``, so
``drift_coefficients`` are in value units per unit of ``u``, not per metre
(``origin`` = mean of the exact points, ``scale`` = largest coordinate offset of
an exact point from it, 1 for a single point; both are stored on the result).

``method="kriging"``: ``k(x, x_j) = C(x - x_j)``, the covariance of the
variogram model on the plan lag ``(dx, dy, 0)`` in metres. This is ordinary
kriging (constant drift) or universal kriging (linear drift) with a global
neighbourhood, in dual form.

``method="rbf"``: thin-plate spline, ``k = phi(|u - u_j|)`` with
``phi(r) = r^2 log r`` and ``phi(0) = 0``; the drift is always linear.

Coefficients ``(b, c)`` solve the bordered system::

    [ K    P ] [ b ]   [ t ]        K[i, j] = k(x_i, x_j),  P[i, a] = p_a(u_i)
    [ P^T  0 ] [ c ] = [ 0 ]        t = exact values, then the active bounds

Inequality data
---------------
With lower bounds ``f(x_l) >= l`` and upper bounds ``f(x_u) <= u`` the surface
is the minimiser of ``(1/2) b^T K b`` under the data, the bounds and
``P^T b = 0`` (a convex quadratic programme). At the solution the coefficient
of a bound point is its Lagrange multiplier: ``>= 0`` at an active lower bound,
``<= 0`` at an active upper bound, 0 at an inactive bound. Inactive bound points
therefore do not appear in the result.

The programme is solved by a primal active-set method (Nocedal & Wright 2006,
algorithm 16.3) on the values ``v`` of the surface at the bound points:

1. Solve with the exact data only. Bounds that this surface violates start in
   the working set; ``v`` starts as that surface clipped into the bounds.
2. Solve with the working set as equalities. The values ``v*`` of this surface
   at the free bound points are the minimiser for the working set.
3. If ``v*`` violates a free bound, move from ``v`` towards ``v*`` up to the
   first bound met, add that bound to the working set, and go to 2.
4. Otherwise take ``v = v*``. If a multiplier has the wrong sign, release the
   bound with the worst one and go to 2. Otherwise stop.

Every iterate is feasible and the objective never increases, which is what
makes the method finite. ``max_iterations`` still caps the number of linear
solves; reaching it raises :class:`SurfaceFitError`.

Data rules
----------
* Points closer than ``COINCIDENT_DISTANCE`` are one location.
* Coincident exact values are averaged.
* Several lower bounds at one location: the largest counts; several upper
  bounds: the smallest. A lower bound above the upper bound at one location is
  an error. Equal lower and upper bounds pin the value there.
* A bound at the location of an exact value is dropped when the exact value
  satisfies it and is an error otherwise.
* Non-finite coordinates, values or bounds are an error.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from scipy.special import xlogy

from bakumine.core.estimation.variogram import Structure, VariogramModel

__all__ = [
    "COINCIDENT_DISTANCE",
    "SCHEMA",
    "Surface2D",
    "SurfaceFitError",
    "default_variogram",
    "fit_surface",
]

SCHEMA = "bakumine.surface2d/1"

#: points closer than this (metres) are the same location
COINCIDENT_DISTANCE = 1e-6

_METHODS = ("kriging", "rbf")
_DRIFTS = ("constant", "linear")
#: kernel evaluations per chunk (bounds the memory of the lag arrays)
_CHUNK = 1 << 20
#: smallest accepted ratio of singular values of the drift matrix of the exact points
_DRIFT_RANK_TOLERANCE = 1e-6
#: a multiplier counts as wrong-signed beyond this fraction of the largest coefficient
_MULTIPLIER_TOLERANCE = 1e-10
#: accepted relative residual of a linear solve
_RESIDUAL_TOLERANCE = 1e-6


class SurfaceFitError(ValueError):
    """The data cannot be fitted: too few or inconsistent data, a singular
    system, or the iteration cap was reached."""


# ------------------------------------------------------------------ variogram
def default_variogram(xy: Any, values: Any) -> VariogramModel:
    """Default model for :func:`fit_surface`.

    Spherical, nugget 0, sill = variance of ``values`` (population variance;
    1.0 when it is 0), isotropic range = 3 x the median distance from a point
    to its nearest neighbour (over distinct points; 1.0 with fewer than two).
    """
    pts = _as_xy(xy, "xy")
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size != pts.shape[0]:
        raise ValueError("xy and values must have the same length")
    if vals.size == 0 or not np.isfinite(vals).all():
        raise SurfaceFitError("default_variogram needs at least one finite value")
    sill = float(np.var(vals))
    if not sill > 0.0:
        sill = 1.0
    distinct = np.unique(pts, axis=0)
    span = 1.0
    if distinct.shape[0] >= 2:
        nearest = cKDTree(distinct).query(distinct, k=2)[0][:, 1]
        median = float(np.median(nearest))
        if median > 0.0:
            span = 3.0 * median
    return VariogramModel(0.0, (Structure("spherical", sill, span),))


# -------------------------------------------------------------------- helpers
def _as_xy(xy: Any, name: str) -> np.ndarray:
    arr = np.asarray(xy, dtype=np.float64)
    if arr.ndim == 1 and arr.size == 2:
        arr = arr.reshape(1, 2)
    if arr.size == 0:
        return np.zeros((0, 2))
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError(f"{name} must have shape (n, 2)")
    return arr


def _kernel(
    method: str,
    variogram: VariogramModel | None,
    scale: float,
    a: np.ndarray,
    b: np.ndarray,
) -> np.ndarray:
    """(len(a), len(b)) kernel matrix between plan points (metres)."""
    out = np.empty((a.shape[0], b.shape[0]))
    if b.shape[0] == 0 or a.shape[0] == 0:
        return out
    step = max(1, _CHUNK // b.shape[0])
    for start in range(0, a.shape[0], step):
        part = a[start : start + step]
        d = part[:, None, :] - b[None, :, :]
        if method == "rbf":
            r2 = np.einsum("ijk,ijk->ij", d, d) / (scale * scale)
            out[start : start + step] = 0.5 * xlogy(r2, r2)
        else:
            lag = np.zeros(d.shape[:2] + (3,))
            lag[..., :2] = d
            out[start : start + step] = variogram.covariance(lag)
    return out


def _drift_matrix(drift: str, origin: np.ndarray, scale: float, xy: np.ndarray) -> np.ndarray:
    if drift == "constant":
        return np.ones((xy.shape[0], 1))
    return np.column_stack([np.ones(xy.shape[0]), (xy - origin) / scale])


def _solve(k: np.ndarray, p: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve the bordered system; returns ``(b, c)``."""
    m, q = p.shape
    a = np.zeros((m + q, m + q))
    a[:m, :m] = k
    a[:m, m:] = p
    a[m:, :m] = p.T
    rhs = np.zeros(m + q)
    rhs[:m] = target
    try:
        sol = np.linalg.solve(a, rhs)
    except np.linalg.LinAlgError as exc:
        raise SurfaceFitError(f"the interpolation system is singular: {exc}") from exc
    residual = a @ sol - rhs
    limit = _RESIDUAL_TOLERANCE * (1.0 + float(np.abs(rhs).max(initial=0.0)))
    if not np.isfinite(sol).all() or float(np.abs(residual).max()) > limit:
        raise SurfaceFitError(
            "the interpolation system is singular or too ill-conditioned to solve "
            "(check the variogram model: zero sill, or a gaussian model without nugget)"
        )
    return sol[:m], sol[m:]


@dataclass
class _Data:
    exact_xy: np.ndarray  # (ne, 2)
    exact: np.ndarray  # (ne,)
    bound_xy: np.ndarray  # (nb, 2) distinct bound locations without an exact value
    lo: np.ndarray  # (nb,) -inf where there is no lower bound
    hi: np.ndarray  # (nb,) +inf where there is no upper bound
    lo_index: np.ndarray  # (nb,) row of the lower-bound input that sets lo, or -1
    hi_index: np.ndarray  # (nb,)
    pinned: np.ndarray  # (nb,) bool: lower == upper
    dropped_violation: float  # worst violation among bounds dropped at exact points
    n_bounds: int  # bounds given (before any merging)


def _bound_input(xy: Any, values: Any, name: str) -> tuple[np.ndarray, np.ndarray]:
    if xy is None and values is None:
        return np.zeros((0, 2)), np.zeros(0)
    if xy is None or values is None:
        raise ValueError(f"{name}_xy and {name} must be given together")
    pts = _as_xy(xy, f"{name}_xy")
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    if vals.size != pts.shape[0]:
        raise ValueError(f"{name}_xy and {name} must have the same length")
    if not (np.isfinite(pts).all() and np.isfinite(vals).all()):
        raise SurfaceFitError(f"{name} bounds must be finite (coordinates and values)")
    return pts, vals


def _prepare(xy, values, lower_xy, lower, upper_xy, upper, tolerance: float) -> _Data:
    """Apply the data rules of the module docstring."""
    e_xy = _as_xy(xy, "xy")
    e_val = np.asarray(values, dtype=np.float64).reshape(-1)
    if e_val.size != e_xy.shape[0]:
        raise ValueError("xy and values must have the same length")
    if e_val.size == 0:
        raise SurfaceFitError("at least one exact value is needed")
    if not (np.isfinite(e_xy).all() and np.isfinite(e_val).all()):
        raise SurfaceFitError("exact data must be finite (coordinates and values)")
    l_xy, l_val = _bound_input(lower_xy, lower, "lower")
    u_xy, u_val = _bound_input(upper_xy, upper, "upper")
    ne, nl, nu = e_val.size, l_val.size, u_val.size

    # locations: connected groups of points closer than COINCIDENT_DISTANCE
    pts = np.vstack([e_xy, l_xy, u_xy])
    n = pts.shape[0]
    pairs = cKDTree(pts).query_pairs(COINCIDENT_DISTANCE, output_type="ndarray")
    if pairs.shape[0]:
        graph = coo_matrix((np.ones(pairs.shape[0]), (pairs[:, 0], pairs[:, 1])), shape=(n, n))
        n_loc, label = connected_components(graph, directed=False)
    else:
        n_loc, label = n, np.arange(n)

    # exact values: mean per location, at the first point of the location
    e_label = label[:ne]
    count = np.bincount(e_label, minlength=n_loc)
    mean = np.bincount(e_label, weights=e_val, minlength=n_loc) / np.maximum(count, 1)
    has_exact = count > 0
    _, first = np.unique(e_label, return_index=True)
    first.sort()
    exact_xy = e_xy[first]
    exact = mean[e_label[first]]

    # bounds: tightest per location
    lo = np.full(n_loc, -np.inf)
    hi = np.full(n_loc, np.inf)
    lo_index = np.full(n_loc, -1, dtype=np.int64)
    hi_index = np.full(n_loc, -1, dtype=np.int64)
    where = np.full(n_loc, -1, dtype=np.int64)  # first bound point of the location
    dropped = 0.0
    for i in range(nl):
        loc = label[ne + i]
        if has_exact[loc]:
            excess = l_val[i] - mean[loc]
            if excess > tolerance:
                raise SurfaceFitError(
                    f"inconsistent data: lower bound {i} is above the exact value at its location"
                )
            dropped = max(dropped, excess)
            continue
        if where[loc] < 0:
            where[loc] = ne + i
        if l_val[i] > lo[loc]:
            lo[loc], lo_index[loc] = l_val[i], i
    for i in range(nu):
        loc = label[ne + nl + i]
        if has_exact[loc]:
            excess = mean[loc] - u_val[i]
            if excess > tolerance:
                raise SurfaceFitError(
                    f"inconsistent data: upper bound {i} is below the exact value at its location"
                )
            dropped = max(dropped, excess)
            continue
        if where[loc] < 0:
            where[loc] = ne + nl + i
        if u_val[i] < hi[loc]:
            hi[loc], hi_index[loc] = u_val[i], i
    keep = np.flatnonzero(where >= 0)
    keep = keep[np.argsort(where[keep])]
    lo, hi, lo_index, hi_index = lo[keep], hi[keep], lo_index[keep], hi_index[keep]
    crossed = np.flatnonzero(lo > hi + tolerance)
    if crossed.size:
        j = crossed[0]
        raise SurfaceFitError(
            f"inconsistent data: lower bound {lo_index[j]} is above upper bound {hi_index[j]} "
            "at the same location"
        )
    pinned = lo >= hi
    middle = 0.5 * (lo[pinned] + hi[pinned])
    lo[pinned] = middle
    hi[pinned] = middle
    return _Data(
        exact_xy=exact_xy,
        exact=exact,
        bound_xy=pts[where[keep]],
        lo=lo,
        hi=hi,
        lo_index=lo_index,
        hi_index=hi_index,
        pinned=pinned,
        dropped_violation=float(dropped),
        n_bounds=nl + nu,
    )


# --------------------------------------------------------------------- result
@dataclass
class Surface2D:
    """A fitted surface ``f(x, y)``; see the module docstring for the form.

    Attributes
    ----------
    method : ``"kriging"`` or ``"rbf"``.
    variogram : the model used (``None`` for ``"rbf"``).
    drift : ``"constant"`` or ``"linear"``.
    points : (m, 2) exact points (coincident ones merged), then the active
        bound points.
    coefficients : (m,) dual weights ``b``.
    drift_coefficients : (1,) or (3,) ``c`` for the monomials ``[1, u_x, u_y]``
        of ``u = (xy - origin) / scale``.
    active_lower, active_upper : rows of the lower / upper bound input that
        are active (met with equality), ascending.
    iterations : number of linear solves (1 when no bound is active).
    max_violation : largest violation over all bounds given, after the fit.
        Free bounds are within ``tolerance`` by construction; at an active
        bound this is the rounding residual of the linear solve (about 1e-13
        for some hundreds of points, up to about 1e-8 for thousands of
        thin-plate points).
    origin, scale : the frame of the drift monomials (and of the thin-plate
        kernel).
    """

    method: str
    variogram: VariogramModel | None
    drift: str
    points: np.ndarray
    coefficients: np.ndarray
    drift_coefficients: np.ndarray
    active_lower: np.ndarray
    active_upper: np.ndarray
    iterations: int
    max_violation: float
    origin: np.ndarray
    scale: float

    def __call__(self, xy: Any) -> np.ndarray:
        """Surface values, shape (n,), at plan points ``xy`` (n, 2)."""
        targets = _as_xy(xy, "xy")
        out = _drift_matrix(self.drift, self.origin, self.scale, targets) @ self.drift_coefficients
        m = self.points.shape[0]
        step = max(1, _CHUNK // max(m, 1))
        for start in range(0, targets.shape[0], step):
            part = targets[start : start + step]
            k = _kernel(self.method, self.variogram, self.scale, part, self.points)
            out[start : start + step] += k @ self.coefficients
        return out

    def grid(self, origin_xy: Any, spacing_xy: Any, shape: Any) -> np.ndarray:
        """Values on a regular grid: ``z[i, j]`` at ``origin + (i dx, j dy)``,
        shape ``(nx, ny)``."""
        ox, oy = (float(v) for v in origin_xy)
        dx, dy = (float(v) for v in spacing_xy)
        nx, ny = (int(v) for v in shape)
        if nx < 0 or ny < 0:
            raise ValueError("shape must be non-negative")
        gx, gy = np.meshgrid(ox + dx * np.arange(nx), oy + dy * np.arange(ny), indexing="ij")
        return self(np.column_stack([gx.ravel(), gy.ravel()])).reshape(nx, ny)

    def to_dict(self) -> dict[str, Any]:
        """JSON-serialisable dict; :meth:`from_dict` restores the surface."""
        return {
            "schema": SCHEMA,
            "method": self.method,
            "variogram": None if self.variogram is None else self.variogram.to_dict(),
            "drift": self.drift,
            "points": self.points.tolist(),
            "coefficients": self.coefficients.tolist(),
            "drift_coefficients": self.drift_coefficients.tolist(),
            "active_lower": self.active_lower.tolist(),
            "active_upper": self.active_upper.tolist(),
            "iterations": int(self.iterations),
            "max_violation": float(self.max_violation),
            "origin": self.origin.tolist(),
            "scale": float(self.scale),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Surface2D:
        schema = data.get("schema", SCHEMA)
        if schema != SCHEMA:
            raise ValueError(f"unsupported surface schema {schema!r}, expected {SCHEMA!r}")
        method, drift = data["method"], data["drift"]
        if method not in _METHODS or drift not in _DRIFTS:
            raise ValueError(f"unknown method {method!r} or drift {drift!r}")
        variogram = data.get("variogram")
        if method == "kriging" and variogram is None:
            raise ValueError("a kriging surface needs its variogram")
        points = np.asarray(data["points"], dtype=np.float64).reshape(-1, 2)
        coefficients = np.asarray(data["coefficients"], dtype=np.float64).reshape(-1)
        drift_coefficients = np.asarray(data["drift_coefficients"], dtype=np.float64).reshape(-1)
        if coefficients.size != points.shape[0]:
            raise ValueError("points and coefficients must have the same length")
        if drift_coefficients.size != (1 if drift == "constant" else 3):
            raise ValueError("drift_coefficients do not match the drift")
        return cls(
            method=method,
            variogram=None if variogram is None else VariogramModel.from_dict(variogram),
            drift=drift,
            points=points,
            coefficients=coefficients,
            drift_coefficients=drift_coefficients,
            active_lower=np.asarray(data.get("active_lower", ()), dtype=np.int64).reshape(-1),
            active_upper=np.asarray(data.get("active_upper", ()), dtype=np.int64).reshape(-1),
            iterations=int(data.get("iterations", 0)),
            max_violation=float(data.get("max_violation", 0.0)),
            origin=np.asarray(data["origin"], dtype=np.float64).reshape(2),
            scale=float(data["scale"]),
        )


# ------------------------------------------------------------------------ fit
def fit_surface(
    xy: Any,
    values: Any,
    *,
    lower_xy: Any = None,
    lower: Any = None,
    upper_xy: Any = None,
    upper: Any = None,
    method: str = "kriging",
    variogram: VariogramModel | None = None,
    drift: str = "constant",
    tolerance: float = 1e-9,
    max_iterations: int | None = None,
) -> Surface2D:
    """Fit a surface through exact values that honours lower and upper bounds.

    Parameters
    ----------
    xy, values : (n, 2) plan points and (n,) exact values; at least one.
    lower_xy, lower : points and values of lower bounds ``f >= lower``.
    upper_xy, upper : points and values of upper bounds ``f <= upper``. A
        "between" pick is one lower and one upper bound at the same point.
    method : ``"kriging"`` (dual ordinary / universal kriging) or ``"rbf"``
        (thin-plate spline; ``drift`` is forced to ``"linear"`` and
        ``variogram`` is not used).
    variogram : covariance model for ``"kriging"``; default
        :func:`default_variogram` of the exact data.
    drift : ``"constant"`` or ``"linear"``. A linear drift needs three exact
        points that are not on one line.
    tolerance : absolute tolerance, in value units, on the bounds: a free
        bound is violated beyond it. ``max_violation`` on the result is at most
        this, apart from the rounding residual of the solve at active bounds
        (see :class:`Surface2D`).
    max_iterations : cap on the linear solves; default
        ``10 x (number of bounds) + 10``. Reaching it raises.

    Raises
    ------
    SurfaceFitError : no exact value, too few points for the drift,
        inconsistent data, a singular system, or the iteration cap.
    ValueError : malformed arguments.
    """
    if method not in _METHODS:
        raise ValueError(f"method must be one of {_METHODS}, got {method!r}")
    if drift not in _DRIFTS:
        raise ValueError(f"drift must be one of {_DRIFTS}, got {drift!r}")
    if not (np.isfinite(tolerance) and tolerance >= 0.0):
        raise ValueError("tolerance must be a non-negative finite number")
    data = _prepare(xy, values, lower_xy, lower, upper_xy, upper, float(tolerance))
    if max_iterations is None:
        max_iterations = 10 * data.n_bounds + 10
    if int(max_iterations) < 1:
        raise ValueError("max_iterations must be >= 1")
    if method == "rbf":
        drift = "linear"
        variogram = None
    else:
        if variogram is None:
            variogram = default_variogram(data.exact_xy, data.exact)
        elif not isinstance(variogram, VariogramModel):
            raise TypeError("variogram must be a VariogramModel")

    ne, nb = data.exact.size, data.lo.size
    pts = np.vstack([data.exact_xy, data.bound_xy])
    origin = data.exact_xy.mean(axis=0)
    scale = float(np.abs(data.exact_xy - origin).max())
    if not scale > 0.0:
        scale = 1.0
    p_all = _drift_matrix(drift, origin, scale, pts)
    if drift == "linear":
        sv = np.linalg.svd(p_all[:ne], compute_uv=False)
        if ne < 3 or sv[-1] <= _DRIFT_RANK_TOLERANCE * sv[0]:
            raise SurfaceFitError(
                "a linear drift needs at least three exact points that are not on one line"
            )
    k_all = _kernel(method, variogram, scale, pts, pts)

    # values are shifted by their mean for the solve (the drift absorbs it)
    shift = float(data.exact.mean())
    exact = data.exact - shift
    lo, hi = data.lo - shift, data.hi - shift

    def solve(active: np.ndarray, state: np.ndarray):
        idx = np.concatenate([np.arange(ne), ne + active])
        target = np.concatenate([exact, np.where(state[active] < 0, lo[active], hi[active])])
        b, c = _solve(k_all[np.ix_(idx, idx)], p_all[idx], target)
        return idx, b, c

    def at_bounds(rows: np.ndarray, idx: np.ndarray, b: np.ndarray, c: np.ndarray):
        return k_all[np.ix_(ne + rows, idx)] @ b + p_all[ne + rows] @ c

    state = np.zeros(nb, dtype=np.int8)  # -1 at the lower bound, +1 at the upper, 0 free
    active = np.zeros(0, dtype=np.int64)
    idx, b, c = solve(active, state)
    iterations = 1
    if nb:
        every = np.arange(nb)
        v = at_bounds(every, idx, b, c)
        state[v < lo - tolerance] = -1
        state[v > hi + tolerance] = 1
        state[data.pinned] = -1
        v = np.clip(v, lo, hi)
        while state.any():
            if iterations >= max_iterations:
                raise SurfaceFitError(
                    f"the active-set iteration did not finish in {max_iterations} solves"
                )
            active = np.flatnonzero(state)
            idx, b, c = solve(active, state)
            iterations += 1
            free = np.flatnonzero(state == 0)
            v_new = at_bounds(free, idx, b, c)
            step = v_new - v[free]
            below = v_new < lo[free] - tolerance
            above = v_new > hi[free] + tolerance
            if below.any() or above.any():
                alpha = np.full(free.size, np.inf)
                alpha[below] = (lo[free][below] - v[free][below]) / step[below]
                alpha[above] = (hi[free][above] - v[free][above]) / step[above]
                j = int(np.argmin(alpha))
                a = min(max(float(alpha[j]), 0.0), 1.0)
                v[free] = np.clip(v[free] + a * step, lo[free], hi[free])
                blocker = free[j]
                state[blocker] = -1 if below[j] else 1
                v[blocker] = lo[blocker] if below[j] else hi[blocker]
                continue
            v[free] = v_new
            # multipliers: b >= 0 at a lower bound, b <= 0 at an upper bound
            wrong = state[active] * b[ne:]
            wrong[data.pinned[active]] = -np.inf
            worst = int(np.argmax(wrong))
            if wrong[worst] > _MULTIPLIER_TOLERANCE * float(np.abs(b).max()):
                state[active[worst]] = 0
                continue
            break
        active = np.flatnonzero(state)

    violation = data.dropped_violation
    if nb:
        f = at_bounds(np.arange(nb), idx, b, c)
        violation = max(violation, float(np.max(lo - f)), float(np.max(f - hi)))
    violation = max(violation, 0.0)
    at_lower = active[(state[active] < 0) | data.pinned[active]]
    at_upper = active[(state[active] > 0) | data.pinned[active]]
    c = c.copy()
    c[0] += shift
    return Surface2D(
        method=method,
        variogram=variogram,
        drift=drift,
        points=pts[idx].copy(),
        coefficients=b,
        drift_coefficients=c,
        active_lower=np.sort(data.lo_index[at_lower]),
        active_upper=np.sort(data.hi_index[at_upper]),
        iterations=iterations,
        max_violation=violation,
        origin=origin,
        scale=scale,
    )
