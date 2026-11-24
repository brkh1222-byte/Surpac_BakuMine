"""Elastic wave speeds, explosive pressures and wave sources of a blast.

research_blast.md part 3, section 1. These are the textbook relations of an isotropic linear
elastic solid and of an ideal detonation; they feed the viewer (expanding wavefronts) and
the near-field damage model (:mod:`bakumine.blast.damage`).

Wave speeds
-----------
With Young's modulus ``E``, Poisson's ratio ``nu`` and density ``rho``

* ``cp = sqrt(E (1 - nu) / (rho (1 + nu) (1 - 2 nu)))`` (longitudinal, P),
* ``cs = sqrt(E / (2 rho (1 + nu)))`` (shear, S),
* ``cs / cp = sqrt((1 - 2 nu) / (2 (1 - nu)))``.

A measured ``Rock.p_wave`` replaces the computed ``cp``; ``cs`` then follows from the ratio.

Pressures
---------
* Detonation (Chapman-Jouguet) pressure ``P_d = rho_e VOD^2 / 4`` (adiabatic exponent 3).
* Borehole (explosion) pressure of a fully coupled charge ``P_b = P_d / 2``.

Both are the usual engineering approximations (Cook 1958; Persson, Holmberg and Lee 1994).

Crushed zone
------------
:func:`crushed_zone_radius` gives the radius of the crushed annulus around a hole by one of
two published simple models (Esen, Onederra and Bilgin 2003; the pressure / strength ratio
used by Kanchibotla, Valery and Morrell 1999). They disagree by a factor of several, which
is the honest state of knowledge; the result names its model.

Units: SI (m, kg, s, Pa). Times are seconds except the ``t_ms`` of :func:`sources`.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from bakumine.blast.model import BlastDesign, Explosive, Hole, Rock

__all__ = [
    "arrival_time",
    "borehole_pressure",
    "crushed_zone_radius",
    "detonation_pressure",
    "sources",
    "wave_speeds",
]

#: Constants of the crushing-zone model of Esen, Onederra and Bilgin (2003):
#: ``r_c = 0.812 r_0 CZI^0.219``.
ESEN_COEFFICIENT = 0.812
ESEN_EXPONENT = 0.219


def wave_speeds(rock: Rock) -> tuple[float, float]:
    """P- and S-wave speed of the rock, m/s.

    Parameters
    ----------
    rock : Rock
        ``youngs_gpa``, ``poisson`` and ``density`` give the speeds; a ``p_wave`` that is not
        None is taken as the measured P-wave speed, and the S-wave speed then follows from
        the Poisson ratio.

    Returns
    -------
    cp, cs : float
    """
    nu = rock.poisson
    ratio = math.sqrt((1.0 - 2.0 * nu) / (2.0 * (1.0 - nu)))  # cs / cp
    if rock.p_wave is not None:
        cp = float(rock.p_wave)
        if not (math.isfinite(cp) and cp > 0):
            raise ValueError(f"rock p_wave must be > 0, got {rock.p_wave!r}")
        return cp, cp * ratio
    e = rock.youngs_gpa * 1e9
    cp = math.sqrt(e * (1.0 - nu) / (rock.density * (1.0 + nu) * (1.0 - 2.0 * nu)))
    cs = math.sqrt(e / (2.0 * rock.density * (1.0 + nu)))
    return cp, cs


def detonation_pressure(explosive: Explosive) -> float:
    """Detonation pressure ``rho VOD^2 / 4`` in pascals (ideal detonation, gamma = 3)."""
    return explosive.density * explosive.vod**2 / 4.0


def borehole_pressure(explosive: Explosive) -> float:
    """Pressure on the wall of a fully coupled hole, pascals: half the detonation pressure.

    Decoupled charges (a gap between explosive and wall) are not modelled.
    """
    return 0.5 * detonation_pressure(explosive)


def crushed_zone_radius(hole: Hole, rock: Rock, model: str = "esen") -> float:
    """Radius of the crushed zone around a hole, metres from the hole axis.

    Parameters
    ----------
    hole : Hole
        Its diameter and explosive are used; the charge is taken as fully coupled.
    rock : Rock
        ``ucs_mpa``, ``youngs_gpa`` and ``poisson``.
    model : {"esen", "pressure_ratio"}
        ``"esen"``: Esen, Onederra and Bilgin (2003), ``r_c = 0.812 r_0 CZI^0.219`` with the
        crushing zone index ``CZI = P_b^3 / (K sigma_c^2)`` and the rock stiffness
        ``K = E / (1 + nu)`` (the paper uses the dynamic modulus and Poisson ratio; the
        values of ``rock`` are used as they are).
        ``"pressure_ratio"``: ``r_c = r_0 sqrt(P_d / sigma_c)`` with the detonation pressure
        (Kanchibotla, Valery and Morrell 1999); it gives a much larger zone.

    Returns
    -------
    float
        Never smaller than the hole radius.
    """
    r0 = 0.5 * hole.diameter
    sigma_c = rock.ucs_mpa * 1e6
    if model == "esen":
        stiffness = rock.youngs_gpa * 1e9 / (1.0 + rock.poisson)
        czi = borehole_pressure(hole.explosive) ** 3 / (stiffness * sigma_c**2)
        radius = ESEN_COEFFICIENT * r0 * czi**ESEN_EXPONENT
    elif model == "pressure_ratio":
        radius = r0 * math.sqrt(detonation_pressure(hole.explosive) / sigma_c)
    else:
        raise ValueError(f"unknown crushed zone model {model!r}")
    return max(radius, r0)


def arrival_time(points: Any, source: Any, t0: float, cp: float) -> np.ndarray:
    """Arrival time of the P wave: ``t0 + |point - source| / cp``, seconds.

    Parameters
    ----------
    points : (n, 3) or (3,) array_like
    source : (3,) array_like
    t0 : float
        Time the source fires, seconds.
    cp : float
        Wave speed, m/s.
    """
    if not (math.isfinite(cp) and cp > 0):
        raise ValueError(f"wave speed must be > 0, got {cp!r}")
    p = np.asarray(points, dtype=np.float64)
    s = np.asarray(source, dtype=np.float64)
    return t0 + np.linalg.norm(p - s, axis=-1) / cp


def sources(design: BlastDesign) -> list[dict[str, Any]]:
    """Wave sources of the result bundle (``waves.sources``): one per charged hole.

    Each is ``{"hole": id, "t_ms": firing time, "centre": [x, y, z] of the charge column,
    "charge_kg": mass}``, ordered by firing time (then by the order of the holes). Holes
    without charge are left out.

    Raises
    ------
    ValueError
        A charged hole has no ``delay_ms`` (the design is not timed).
    """
    out: list[dict[str, Any]] = []
    for hole in design.holes:
        if not hole.charge_mass > 0:
            continue
        if not math.isfinite(hole.delay_ms):
            raise ValueError(
                f"hole {hole.id} has no delay_ms: time the design before asking for wave sources"
            )
        out.append(
            {
                "hole": hole.id,
                "t_ms": float(hole.delay_ms),
                "centre": [float(v) for v in hole.charge_centre],
                "charge_kg": float(hole.charge_mass),
            }
        )
    out.sort(key=lambda s: s["t_ms"])
    return out
