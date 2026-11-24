"""Flyrock: maximum throw by mechanism (Richards and Moore 2004) and ballistic flight.

research_blast.md part 2, section 3.

Richards and Moore (2004)
-------------------------
Three mechanisms, each with a launch velocity ``v0 = k * (sqrt(m) / c)**1.3`` where ``m`` is
the explosive mass per metre of hole (kg/m), ``c`` the confinement (burden ``B`` or stemming
height ``SH``, in metres) and ``k`` a site constant:

* face burst, ``c = B``:        ``L = (k**2 / g) * (sqrt(m) / B)**2.6``
* cratering, ``c = SH``:        ``L = (k**2 / g) * (sqrt(m) / SH)**2.6``
* stemming ejection, ``c = SH``: ``L = (k**2 / g) * (sqrt(m) / SH)**2.6 * sin(2 * theta)``

``L`` is the maximum horizontal range on level ground. The first two are the ballistic
maximum ``v0**2 / g`` (launch at 45 degrees); stemming ejection ("rifling") leaves along the
hole axis, so ``theta`` is the inclination of the hole **from the horizontal** (90 degrees
for a vertical hole, which gives no horizontal range: the ejecta go straight up).

``k`` is 13.5 for soft to 27 for hard rock in the paper; the default 27 is the conservative
end and is uncalibrated. The paper's safety factors on the maximum throw are 2 for
equipment and 4 for people.

Ballistics
----------
:func:`range_on_level` is the analytic range without drag; :func:`trajectory` and
:func:`land` integrate the flight of many particles, optionally with quadratic air drag on a
sphere (:class:`Drag`), to the first contact with the ground (a level or a surface with
``z_at(x, y)``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from bakumine.blast.model import BlastDesign

__all__ = [
    "G",
    "K_HARD",
    "K_SOFT",
    "Drag",
    "FlyrockResult",
    "Ground",
    "cratering_range",
    "face_burst_range",
    "land",
    "launch_velocity",
    "predict",
    "range_on_level",
    "stemming_ejection_range",
    "trajectory",
]

#: Standard gravity, m/s2.
G = 9.81
#: Richards and Moore (2004) site constant: soft rock and hard rock.
K_SOFT = 13.5
K_HARD = 27.0


# --------------------------------------------------------------------------- Richards & Moore


def launch_velocity(
    linear_density: float | np.ndarray, confinement: float | np.ndarray, k: float = K_HARD
) -> np.ndarray:
    """Launch velocity ``v0 = k * (sqrt(m) / c)**1.3`` in m/s.

    ``linear_density`` m in kg/m, ``confinement`` c (burden or stemming height) in m.
    Infinite for zero confinement; NaN for a negative one.
    """
    m = np.asarray(linear_density, dtype=np.float64)
    c = np.asarray(confinement, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return k * (np.sqrt(m) / np.where(c >= 0, c, np.nan)) ** 1.3


def face_burst_range(
    linear_density: float | np.ndarray,
    burden: float | np.ndarray,
    k: float = K_HARD,
    g: float = G,
) -> np.ndarray:
    """Maximum throw in front of the face, ``L = (k**2 / g) * (sqrt(m) / B)**2.6`` (m)."""
    return launch_velocity(linear_density, burden, k) ** 2 / g


def cratering_range(
    linear_density: float | np.ndarray,
    stemming: float | np.ndarray,
    k: float = K_HARD,
    g: float = G,
) -> np.ndarray:
    """Maximum throw from cratering of the collar, ``L = (k**2 / g) * (sqrt(m) / SH)**2.6``."""
    return launch_velocity(linear_density, stemming, k) ** 2 / g


def stemming_ejection_range(
    linear_density: float | np.ndarray,
    stemming: float | np.ndarray,
    angle: float | np.ndarray,
    k: float = K_HARD,
    g: float = G,
) -> np.ndarray:
    """Maximum throw of ejected stemming,
    ``L = (k**2 / g) * (sqrt(m) / SH)**2.6 * sin(2 * angle)``.

    ``angle`` is the launch angle in **radians from the horizontal**, i.e. the inclination
    of the hole (pi/2 for a vertical hole gives 0).
    """
    s = np.clip(np.sin(2.0 * np.asarray(angle, dtype=np.float64)), 0.0, None)
    return launch_velocity(linear_density, stemming, k) ** 2 / g * s


@dataclass
class FlyrockResult:
    """Richards-Moore maximum throws of a design (metres; NaN = mechanism not evaluated).

    ``governing_mechanism`` / ``governing_hole`` name the largest throw; the clearance
    distances are ``max_range_m`` times the safety factors. ``holes`` maps each mechanism to
    the id of its governing hole.
    """

    face_burst_m: float
    cratering_m: float
    stemming_ejection_m: float
    max_range_m: float
    governing_mechanism: str
    governing_hole: str
    holes: dict[str, str]
    max_launch_velocity: float
    clearance_equipment_m: float
    clearance_people_m: float
    k: float
    safety_factor_equipment: float
    safety_factor_people: float
    calibrated: bool = False
    model: str = "Richards and Moore (2004)"

    def to_dict(self) -> dict[str, Any]:
        """The ``flyrock`` block of the result bundle (plan_blast.md section 3); the
        clearance is the one for people."""
        return {
            "face_burst_m": self.face_burst_m,
            "cratering_m": self.cratering_m,
            "stemming_ejection_m": self.stemming_ejection_m,
            "max_range_m": self.max_range_m,
            "clearance_m": self.clearance_people_m,
            "safety_factor": self.safety_factor_people,
        }


def predict(
    design: BlastDesign,
    k: float = K_HARD,
    safety_factor_equipment: float = 2.0,
    safety_factor_people: float = 4.0,
    g: float = G,
    calibrated: bool = False,
) -> FlyrockResult:
    """Maximum flyrock throw of ``design`` by mechanism, over its charged holes.

    * Face burst: the holes of row 0 (next to the free face), with the hole's burden or,
      when the hole has none, the design burden. NaN when no burden is known.
    * Cratering and stemming ejection: every charged hole, with its stemming length
      (``Hole.stemming``, measured along the hole). A hole charged to the collar gives an
      infinite throw: the formula has no bound without stemming, and that is reported as is.
    * The launch angle of stemming ejection is the hole inclination from the horizontal.

    ``m`` is ``Hole.linear_density`` (fully coupled charge).
    """
    holes = [h for h in design.holes if h.charge_mass > 0]
    if not holes:
        raise ValueError("the design has no charged holes")
    best: dict[str, tuple[float, str]] = {}
    v_max = 0.0

    def offer(mechanism: str, value: float, hole_id: str) -> None:
        if math.isnan(value):
            return
        if mechanism not in best or value > best[mechanism][0]:
            best[mechanism] = (value, hole_id)

    for h in holes:
        m = h.linear_density
        if h.row == 0:
            burden = h.burden if math.isfinite(h.burden) else design.burden
            if math.isfinite(burden) and burden > 0:
                offer("face_burst", float(face_burst_range(m, burden, k, g)), h.id)
                v_max = max(v_max, float(launch_velocity(m, burden, k)))
        inclination = math.asin(min(1.0, max(-1.0, -float(h.direction[2]))))
        offer("cratering", float(cratering_range(m, h.stemming, k, g)), h.id)
        offer(
            "stemming_ejection",
            float(stemming_ejection_range(m, h.stemming, inclination, k, g)),
            h.id,
        )
        v_max = max(v_max, float(launch_velocity(m, h.stemming, k)))

    def value(mechanism: str) -> float:
        return best[mechanism][0] if mechanism in best else math.nan

    governing = max(best, key=lambda name: best[name][0])
    max_range = best[governing][0]
    return FlyrockResult(
        face_burst_m=value("face_burst"),
        cratering_m=value("cratering"),
        stemming_ejection_m=value("stemming_ejection"),
        max_range_m=max_range,
        governing_mechanism=governing,
        governing_hole=best[governing][1],
        holes={name: hole_id for name, (_, hole_id) in best.items()},
        max_launch_velocity=v_max,
        clearance_equipment_m=safety_factor_equipment * max_range,
        clearance_people_m=safety_factor_people * max_range,
        k=k,
        safety_factor_equipment=safety_factor_equipment,
        safety_factor_people=safety_factor_people,
        calibrated=calibrated,
    )


# --------------------------------------------------------------------------- ballistics


class Ground(Protocol):
    """A surface that gives its height at plan positions (arrays in, array out; NaN where
    the surface is not defined)."""

    def z_at(self, x: np.ndarray, y: np.ndarray) -> np.ndarray: ...


@dataclass(frozen=True)
class Drag:
    """Quadratic air drag on a sphere: ``a = -c * |v| * v`` with
    ``c = 3 * rho_air * cd / (4 * rho * d)`` (drag force ``0.5 * rho_air * cd * A * v**2``
    over the mass of a sphere of diameter ``d`` and density ``rho``).

    ``size`` (m) and ``density`` (kg/m3) may be arrays with one value per particle.
    ``cd = 0.47`` is the textbook value for a smooth sphere at Reynolds numbers of about
    1e3 to 2e5; real fragments are angular and have a higher, shape-dependent coefficient.
    """

    size: float | np.ndarray
    density: float | np.ndarray = 2700.0
    cd: float = 0.47
    air_density: float = 1.225

    @property
    def coefficient(self) -> np.ndarray:
        """``c`` in 1/m."""
        size = np.asarray(self.size, dtype=np.float64)
        density = np.asarray(self.density, dtype=np.float64)
        return 3.0 * self.air_density * self.cd / (4.0 * density * size)


def range_on_level(
    v0: float | np.ndarray, angle: float | np.ndarray, h0: float | np.ndarray = 0.0, g: float = G
) -> np.ndarray:
    """Horizontal range without drag of a launch at speed ``v0`` (m/s) and ``angle``
    (**radians** above the horizontal) from height ``h0`` (m) above the landing level:

    ``R = v0 cos(a) / g * (v0 sin(a) + sqrt(v0**2 sin(a)**2 + 2 g h0))``

    which is ``v0**2 sin(2a) / g`` for ``h0 = 0``. NaN when the particle starts below the
    level and cannot reach it.
    """
    v0 = np.asarray(v0, dtype=np.float64)
    angle = np.asarray(angle, dtype=np.float64)
    vz = v0 * np.sin(angle)
    with np.errstate(invalid="ignore"):
        root = np.sqrt(vz**2 + 2.0 * g * np.asarray(h0, dtype=np.float64))
    return v0 * np.cos(angle) / g * (vz + root)


def _acceleration(v: np.ndarray, g: float, c: np.ndarray | None) -> np.ndarray:
    a = np.zeros_like(v)
    a[:, 2] = -g
    if c is not None:
        a -= (c * np.linalg.norm(v, axis=1))[:, None] * v
    return a


def _step(
    p: np.ndarray, v: np.ndarray, dt: float | np.ndarray, g: float, c: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Advance ``p, v`` (n, 3) by ``dt`` (scalar or one per particle): exact without drag,
    one classical Runge-Kutta step with it."""
    h = np.broadcast_to(np.asarray(dt, dtype=np.float64), (p.shape[0],))[:, None]
    if c is None:
        a = _acceleration(v, g, None)
        return p + v * h + 0.5 * a * h**2, v + a * h
    k1v = _acceleration(v, g, c)
    k1p = v
    k2v = _acceleration(v + 0.5 * h * k1v, g, c)
    k2p = v + 0.5 * h * k1v
    k3v = _acceleration(v + 0.5 * h * k2v, g, c)
    k3p = v + 0.5 * h * k2v
    k4v = _acceleration(v + h * k3v, g, c)
    k4p = v + h * k3v
    return (
        p + h / 6.0 * (k1p + 2.0 * k2p + 2.0 * k3p + k4p),
        v + h / 6.0 * (k1v + 2.0 * k2v + 2.0 * k3v + k4v),
    )


def _particles(p0: np.ndarray, v0: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    p = np.asarray(p0, dtype=np.float64)
    v = np.asarray(v0, dtype=np.float64)
    single = p.ndim == 1 and v.ndim == 1
    p, v = np.atleast_2d(p), np.atleast_2d(v)
    if p.shape[1] != 3 or v.shape[1] != 3:
        raise ValueError("p0 and v0 must be (n, 3) or (3,)")
    n = max(p.shape[0], v.shape[0])
    return (
        np.array(np.broadcast_to(p, (n, 3))),
        np.array(np.broadcast_to(v, (n, 3))),
        single,
    )


def _drag_coefficient(drag: Drag | None, n: int) -> np.ndarray | None:
    if drag is None:
        return None
    return np.array(np.broadcast_to(drag.coefficient, (n,)), dtype=np.float64)


def trajectory(
    p0: np.ndarray,
    v0: np.ndarray,
    g: float = G,
    drag: Drag | None = None,
    dt: float = 0.02,
    t_end: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Flight path of particles launched at ``p0`` (n, 3) with velocity ``v0`` (n, 3).

    Returns ``(t, p)``: times ``t`` (m,) from 0 in steps of ``dt`` and positions ``p``
    (m, n, 3), or (m, 3) for a single particle given as two (3,) vectors. ``t_end`` defaults
    to the longest no-drag flight time back to the launch height (``2 vz / g``), at least
    one step. The path is not stopped at the ground; use :func:`land` for the contact.
    """
    if not dt > 0:
        raise ValueError("dt must be > 0")
    p, v, single = _particles(p0, v0)
    c = _drag_coefficient(drag, p.shape[0])
    if t_end is None:
        vz = v[:, 2][np.isfinite(v[:, 2])]
        t_end = max(dt, 2.0 * float(vz.max(initial=0.0)) / g)
    steps = max(1, math.ceil(t_end / dt - 1e-9))
    t = np.arange(steps + 1) * dt
    out = np.empty((steps + 1, p.shape[0], 3))
    out[0] = p
    if c is None:
        a = _acceleration(v, g, None)
        out[1:] = p[None] + v[None] * t[1:, None, None] + 0.5 * a[None] * t[1:, None, None] ** 2
    else:
        for i in range(steps):
            p, v = _step(p, v, dt, g, c)
            out[i + 1] = p
    return t, (out[:, 0, :] if single else out)


def land(
    p0: np.ndarray,
    v0: np.ndarray,
    ground: float | Ground,
    g: float = G,
    drag: Drag | None = None,
    dt: float = 0.02,
    t_max: float = 120.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Time and point of the first ground contact of each particle.

    ``ground`` is a level (float z) or an object with ``z_at(x, y)``. Returns
    ``(t_land (n,), p_land (n, 3))``; NaN for a particle that does not land within ``t_max``
    seconds (or that starts below the ground and never comes out of it).

    A level ground without drag is solved in closed form. Otherwise the flight is stepped
    by ``dt`` (exact parabola without drag, RK4 with it), the first step in which the
    height above the ground changes from >= 0 to < 0 is bracketed, and the contact is found
    in it by bisection. A particle that starts on or below the ground is allowed to rise
    out of it first. A ridge thinner than the distance flown in one ``dt`` can be jumped;
    where the surface is undefined (``z_at`` gives NaN) the particle keeps flying.
    """
    if not dt > 0:
        raise ValueError("dt must be > 0")
    p, v, _ = _particles(p0, v0)
    n = p.shape[0]
    c = _drag_coefficient(drag, n)
    t_land = np.full(n, np.nan)
    p_land = np.full((n, 3), np.nan)

    level = None if hasattr(ground, "z_at") else float(ground)  # type: ignore[arg-type]
    if level is not None and c is None:
        with np.errstate(invalid="ignore"):
            t = (v[:, 2] + np.sqrt(v[:, 2] ** 2 + 2.0 * g * (p[:, 2] - level))) / g
        ok = np.isfinite(t) & (t >= 0) & (t <= t_max)
        ok &= np.isfinite(p).all(axis=1) & np.isfinite(v).all(axis=1)
        t_land[ok] = t[ok]
        p_land[ok] = p[ok] + v[ok] * t[ok, None]
        p_land[ok, 2] = level
        return t_land, p_land

    def height(q: np.ndarray) -> np.ndarray:
        if level is not None:
            return q[:, 2] - level
        z = np.asarray(ground.z_at(q[:, 0], q[:, 1]), dtype=np.float64)  # type: ignore[union-attr]
        return q[:, 2] - z

    active = np.isfinite(p).all(axis=1) & np.isfinite(v).all(axis=1)
    h_prev = height(p)
    t_now = 0.0
    for _ in range(math.ceil(t_max / dt)):
        if not active.any():
            break
        idx = np.flatnonzero(active)
        ci = None if c is None else c[idx]
        p_new, v_new = _step(p[idx], v[idx], dt, g, ci)
        h_new = height(p_new)
        with np.errstate(invalid="ignore"):
            cross = (h_prev[idx] >= 0) & (h_new < 0)
        if cross.any():
            j = idx[cross]
            cj = None if c is None else c[j]
            lo = np.zeros(j.size)
            hi = np.full(j.size, dt)
            for _ in range(48):
                mid = 0.5 * (lo + hi)
                with np.errstate(invalid="ignore"):
                    below = height(_step(p[j], v[j], mid, g, cj)[0]) < 0
                hi = np.where(below, mid, hi)
                lo = np.where(below, lo, mid)
            t_land[j] = t_now + lo
            p_land[j] = _step(p[j], v[j], lo, g, cj)[0]
            active[j] = False
        p[idx], v[idx] = p_new, v_new
        h_prev[idx] = h_new
        t_now += dt
    return t_land, p_land
