"""Ground vibration and air overpressure of a blast: scaled-distance laws.

research_blast.md part 2, section 2.

Ground vibration
----------------
``PPV [mm/s] = K * (R / sqrt(W))**-beta``

with ``R`` the distance in metres, ``W`` the explosive mass fired per delay in kg, and
``R / sqrt(W)`` the square-root scaled distance (Duvall and Fogelson 1962; Nicholls, Johnson
and Duvall 1971, USBM Bulletin 656; Siskind et al. 1980, USBM RI 8507; Dowding 1985). ``K``
and ``beta`` are **site constants**: they come from a regression of monitored blasts
(:func:`fit_site_law`). The defaults are published typical values and are labelled
uncalibrated (:data:`AS2187_AVERAGE`, :data:`ISEE_TYPICAL`, :data:`ISEE_UPPER_BOUND`).

The charge per delay follows the usual 8 ms convention: charges firing less than
``window_ms`` apart are counted together (:func:`charge_per_delay`); the largest such sum is
the maximum instantaneous charge (MIC).

The law is a far-field law. Close to the holes (a few burdens) it overestimates without
bound; the near field is the subject of ``damage.py`` (Holmberg-Persson).

Air overpressure
----------------
``P = K_a * (R / W**(1/3))**-a`` with cube-root scaling (AS 2187.2-2006 Appendix J:
``a = 1.45`` and, with ``P`` in kPa, ``K_a`` = 10 to 100 for confined blasthole charges),
and the level ``dBL = 20 * log10(P / 20e-6 Pa)``. Pressures are in pascals in this module;
``k_a`` is entered in the published kPa form (argument ``ka_kpa``).

Limits
------
:data:`LIMITS` (PPV, mm/s) and :data:`AIRBLAST_LIMITS` (dBL) are published guideline values
with their source. They are not a compliance statement: most are frequency dependent and the
applicable rule is set by the regulator of the site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from bakumine.blast.model import BlastDesign, Hole

__all__ = [
    "AIRBLAST_EXPONENT",
    "AIRBLAST_KA_CONFINED_HIGH",
    "AIRBLAST_KA_CONFINED_LOW",
    "AIRBLAST_LIMITS",
    "AS2187_AVERAGE",
    "ISEE_TYPICAL",
    "ISEE_UPPER_BOUND",
    "LIMITS",
    "P_REF",
    "AirblastResult",
    "DelayGroup",
    "Limit",
    "PpvResult",
    "SiteFit",
    "SiteLaw",
    "airblast_at",
    "airblast_pressure",
    "charge_per_delay",
    "check",
    "dbl_to_pressure",
    "fit_site_law",
    "ppv",
    "ppv_at",
    "pressure_to_dbl",
    "scaled_distance",
]

#: Reference sound pressure of the decibel scale, Pa.
P_REF = 20e-6

_FT_PER_M = 1.0 / 0.3048
_LB_PER_KG = 1.0 / 0.45359237
_MM_PER_IN = 25.4


@dataclass(frozen=True)
class SiteLaw:
    """``PPV [mm/s] = k * (R [m] / sqrt(W [kg]))**-beta``.

    ``calibrated`` is True only for constants fitted to monitored blasts of the site.
    """

    k: float = 1140.0
    beta: float = 1.6
    calibrated: bool = False
    source: str = "AS 2187.2-2006 Appendix J, average conditions, free face (uncalibrated)"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.k) and self.k > 0):
            raise ValueError(f"site law k must be > 0, got {self.k!r}")
        if not (math.isfinite(self.beta) and self.beta > 0):
            raise ValueError(f"site law beta must be > 0, got {self.beta!r}")

    def ppv(self, distance: float | np.ndarray, charge: float | np.ndarray) -> np.ndarray:
        """PPV in mm/s at ``distance`` (m) from ``charge`` (kg per delay)."""
        return ppv(distance, charge, self)

    def distance_for(self, ppv_mm_s: float, charge: float) -> float:
        """Distance (m) at which ``charge`` (kg) gives ``ppv_mm_s``."""
        return math.sqrt(charge) * (self.k / ppv_mm_s) ** (1.0 / self.beta)

    def charge_for(self, ppv_mm_s: float, distance: float) -> float:
        """Largest charge per delay (kg) that keeps the PPV at ``distance`` (m) to
        ``ppv_mm_s``."""
        return (distance / (self.k / ppv_mm_s) ** (1.0 / self.beta)) ** 2


def _imperial_k_to_metric(k_in_s: float, beta: float) -> float:
    """Convert ``PPV [in/s] = k * (D [ft] / sqrt(W [lb]))**-beta`` to mm/s, m, kg."""
    sd_ratio = _FT_PER_M / math.sqrt(_LB_PER_KG)  # (ft/lb^0.5) per (m/kg^0.5)
    return k_in_s * _MM_PER_IN * sd_ratio**-beta


#: AS 2187.2-2006 Appendix J: K = 1140, B = 1.6 for "average conditions", blasting to a free
#: face; the standard gives it as the 50 % exceedance estimate. Uncalibrated.
AS2187_AVERAGE = SiteLaw()
#: ISEE Blasters' Handbook: PPV [in/s] = 160 (D/sqrt(W))**-1.6 for typical bench blasting
#: (D in ft, W in lb), converted here to mm/s, m, kg (K about 1143). Uncalibrated.
ISEE_TYPICAL = SiteLaw(
    k=_imperial_k_to_metric(160.0, 1.6),
    beta=1.6,
    source="ISEE Blasters' Handbook, typical: 160 (ft/lb^0.5)^-1.6 in/s (uncalibrated)",
)
#: ISEE Blasters' Handbook (after Oriard): the upper bound of typical data,
#: PPV [in/s] = 242 (D/sqrt(W))**-1.6, converted (K about 1729). Uncalibrated.
ISEE_UPPER_BOUND = SiteLaw(
    k=_imperial_k_to_metric(242.0, 1.6),
    beta=1.6,
    source="ISEE Blasters' Handbook, upper bound: 242 (ft/lb^0.5)^-1.6 in/s (uncalibrated)",
)


def scaled_distance(
    distance: float | np.ndarray, charge: float | np.ndarray, exponent: float = 0.5
) -> np.ndarray:
    """``R / W**exponent``: square-root (0.5, ground vibration) or cube-root (1/3, airblast)
    scaled distance, m/kg^exponent. NaN where the charge is not positive."""
    r = np.asarray(distance, dtype=np.float64)
    w = np.asarray(charge, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return r / np.where(w > 0, w, np.nan) ** exponent


def ppv(
    distance: float | np.ndarray, charge: float | np.ndarray, law: SiteLaw = AS2187_AVERAGE
) -> np.ndarray:
    """``PPV [mm/s] = k * (R / sqrt(W))**-beta``. Infinite at ``R = 0`` (far-field law)."""
    sd = scaled_distance(distance, charge, 0.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        return law.k * sd**-law.beta


# --------------------------------------------------------------------------- charge per delay


@dataclass(frozen=True)
class DelayGroup:
    """Holes that fire within one delay window.

    ``t_ms`` is the firing time of the first hole of the group, ``charge_kg`` the summed
    explosive mass and ``centroid`` the charge-weighted mean of the hole charge centres.
    """

    t_ms: float
    charge_kg: float
    hole_ids: tuple[str, ...]
    centroid: tuple[float, float, float]


def _charged(design: BlastDesign) -> list[Hole]:
    holes = [h for h in design.holes if h.charge_mass > 0]
    if not holes:
        raise ValueError("the design has no charged holes")
    return holes


def charge_per_delay(design: BlastDesign, window_ms: float = 8.0) -> tuple[list[DelayGroup], float]:
    """Delay groups of the charged holes and the maximum instantaneous charge (kg).

    One group per window ``[t, t + window_ms)`` that starts at a firing time ``t`` (windows
    that only repeat part of the previous one are dropped). Two holes exactly ``window_ms``
    apart are separate delays. Groups overlap when the firing times are closer than the
    window; the MIC is the largest group charge, which is the largest mass that fires in
    any window of that length.

    Every charged hole must be timed (``delay_ms`` finite); an untimed hole is an error, not
    a guess.
    """
    if not window_ms >= 0:
        raise ValueError("window_ms must be >= 0")
    holes = _charged(design)
    times = np.array([h.delay_ms for h in holes], dtype=np.float64)
    if not np.all(np.isfinite(times)):
        missing = [h.id for h, t in zip(holes, times, strict=True) if not math.isfinite(t)]
        raise ValueError(f"holes without a firing time (delay_ms): {missing[:10]}")
    order = np.argsort(times, kind="stable")
    times = times[order]
    holes = [holes[i] for i in order]
    masses = np.array([h.charge_mass for h in holes])
    centres = np.array([h.charge_centre for h in holes])

    eps = 1e-9
    groups: list[DelayGroup] = []
    last_end = 0
    starts = np.flatnonzero(np.r_[True, np.diff(times) > eps])
    for i in starts:
        j = int(np.searchsorted(times, times[i] + max(window_ms - eps, eps), side="right"))
        j = max(j, i + 1)
        if j <= last_end:
            continue  # a subset of the previous window
        last_end = j
        w = masses[i:j]
        total = float(w.sum())
        c = (centres[i:j] * w[:, None]).sum(axis=0) / total
        groups.append(
            DelayGroup(
                t_ms=float(times[i]),
                charge_kg=total,
                hole_ids=tuple(h.id for h in holes[i:j]),
                centroid=(float(c[0]), float(c[1]), float(c[2])),
            )
        )
    return groups, max(g.charge_kg for g in groups)


# --------------------------------------------------------------------------- PPV at points


@dataclass
class PpvResult:
    """PPV at points. Arrays have one entry per point.

    ``group`` is the index into ``groups`` of the governing delay group (-1 where the PPV is
    NaN), ``distance`` and ``charge_kg`` the R and W of the governing prediction.
    """

    ppv_mm_s: np.ndarray
    group: np.ndarray
    scaled_distance: np.ndarray
    distance: np.ndarray
    charge_kg: np.ndarray
    groups: list[DelayGroup]
    law: SiteLaw
    mode: str


def _points(points: np.ndarray) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64)
    if p.ndim == 1:
        p = p[None, :]
    if p.ndim != 2 or p.shape[1] != 3:
        raise ValueError(f"points must be (n, 3), got {p.shape}")
    return p


def ppv_at(
    points: np.ndarray,
    design: BlastDesign,
    law: SiteLaw = AS2187_AVERAGE,
    mode: str = "mic",
    window_ms: float = 8.0,
) -> PpvResult:
    """Predicted PPV (mm/s) at ``points`` (n, 3).

    ``mode="mic"``
        The classical single-number prediction: ``W`` = the maximum instantaneous charge and
        ``R`` = the 3D distance to the nearest charge centre of any charged hole.
    ``mode="per_hole"``
        The envelope over the delay groups: ``max_g k * (R_g / sqrt(W_g))**-beta`` with
        ``R_g`` the distance from the point to the charge centroid of group ``g``. Lower
        than "mic" where the largest group is far from the point.

    Points with NaN coordinates give NaN.
    """
    p = _points(points)
    groups, mic = charge_per_delay(design, window_ms)
    with np.errstate(invalid="ignore", divide="ignore"):
        if mode == "mic":
            centres = np.array([h.charge_centre for h in _charged(design)])
            dist = np.full(p.shape[0], np.inf)
            for start in range(0, centres.shape[0], 512):  # bounded memory
                block = centres[start : start + 512]
                d = np.linalg.norm(p[:, None, :] - block[None, :, :], axis=2)
                dist = np.fmin(dist, d.min(axis=1))
            dist = np.where(np.isnan(p).any(axis=1), np.nan, dist)
            charge = np.full(p.shape[0], mic)
            governing = int(np.argmax([g.charge_kg for g in groups]))
            group = np.full(p.shape[0], governing, dtype=np.int64)
        elif mode == "per_hole":
            cents = np.array([g.centroid for g in groups])
            w = np.array([g.charge_kg for g in groups])
            d = np.linalg.norm(p[:, None, :] - cents[None, :, :], axis=2)  # (n, g)
            v = law.k * (d / np.sqrt(w)[None, :]) ** -law.beta
            group = np.argmax(np.where(np.isnan(v), -np.inf, v), axis=1)
            rows = np.arange(p.shape[0])
            dist = d[rows, group]
            charge = w[group]
        else:
            raise ValueError(f"unknown mode {mode!r} (use 'mic' or 'per_hole')")
        sd = dist / np.sqrt(charge)
        value = law.k * sd**-law.beta
    bad = np.isnan(value)
    return PpvResult(
        ppv_mm_s=value,
        group=np.where(bad, -1, group),
        scaled_distance=sd,
        distance=dist,
        charge_kg=np.where(bad, np.nan, charge),
        groups=groups,
        law=law,
        mode=mode,
    )


# --------------------------------------------------------------------------- site calibration


@dataclass(frozen=True)
class SiteFit:
    """Regression of monitored blasts, ``log10 PPV = log10 K - beta * log10(R / sqrt(W))``.

    ``law`` is the best-fit (50 %) line. ``upper_law`` has the same slope and
    ``K_95 = K * 10**(1.645 * s)`` with ``s`` the standard error of the residuals in log10
    units: the line below which 95 % of the data are expected for normal residuals (the
    usual design line, Dowding 1985). ``r2`` is the coefficient of determination in log-log
    space, ``n`` the number of records used.
    """

    law: SiteLaw
    upper_law: SiteLaw
    r2: float
    se_log10: float
    n: int

    @property
    def k_95(self) -> float:
        return self.upper_law.k


def fit_site_law(
    distances: np.ndarray, charges: np.ndarray, ppvs: np.ndarray, source: str = "site fit"
) -> SiteFit:
    """Fit ``K`` and ``beta`` to monitoring records by least squares in log-log space.

    ``distances`` in m, ``charges`` in kg per delay, ``ppvs`` in mm/s. Records with a missing
    or non-positive value are left out. At least 3 records with 2 distinct scaled distances
    are needed.
    """
    r = np.asarray(distances, dtype=np.float64).ravel()
    w = np.asarray(charges, dtype=np.float64).ravel()
    v = np.asarray(ppvs, dtype=np.float64).ravel()
    if not (r.size == w.size == v.size):
        raise ValueError("distances, charges and ppvs must have the same length")
    ok = np.isfinite(r) & np.isfinite(w) & np.isfinite(v) & (r > 0) & (w > 0) & (v > 0)
    n = int(ok.sum())
    if n < 3:
        raise ValueError(f"need at least 3 valid records, got {n}")
    x = np.log10(r[ok] / np.sqrt(w[ok]))
    y = np.log10(v[ok])
    xm, ym = x.mean(), y.mean()
    sxx = float(np.sum((x - xm) ** 2))
    if sxx <= 0:
        raise ValueError("all records have the same scaled distance")
    slope = float(np.sum((x - xm) * (y - ym))) / sxx
    intercept = float(ym - slope * xm)
    if slope >= 0:
        raise ValueError("the PPV does not decrease with scaled distance in these records")
    resid = y - (intercept + slope * x)
    sse = float(np.sum(resid**2))
    sst = float(np.sum((y - ym) ** 2))
    r2 = 1.0 - sse / sst if sst > 0 else math.nan
    se = math.sqrt(sse / (n - 2))
    law = SiteLaw(k=10.0**intercept, beta=-slope, calibrated=True, source=source)
    upper = replace(law, k=law.k * 10.0 ** (1.645 * se), source=f"{source}, 95 % upper bound")
    return SiteFit(law=law, upper_law=upper, r2=r2, se_log10=se, n=n)


# --------------------------------------------------------------------------- limits


@dataclass(frozen=True)
class Limit:
    """A published guideline limit: ``value`` in ``unit`` ("mm/s" or "dBL")."""

    name: str
    value: float
    unit: str
    source: str
    note: str = ""


_RI8507 = (
    "Siskind, Stagg, Kopp and Dowding 1980, USBM RI 8507, Structure response and damage "
    "produced by ground vibration from surface mine blasting"
)
_OSM = "US Office of Surface Mining, 30 CFR 816.67(d) (1983)"
_DIN = "DIN 4150-3:1999, Table 1 (short-term vibration, at the foundation)"
_AS = "AS 2187.2-2006, Appendix J"

#: Ground vibration guideline limits, PPV in mm/s. Frequency-dependent rules are given by
#: their value for the frequency band in the name.
LIMITS: dict[str, Limit] = {
    "usbm_ri8507_plaster_low_freq": Limit(
        "USBM RI 8507, plaster, below 40 Hz", 12.7, "mm/s", _RI8507, "0.50 in/s"
    ),
    "usbm_ri8507_drywall_low_freq": Limit(
        "USBM RI 8507, drywall, below 40 Hz", 19.05, "mm/s", _RI8507, "0.75 in/s"
    ),
    "usbm_ri8507_above_40hz": Limit("USBM RI 8507, above 40 Hz", 50.8, "mm/s", _RI8507, "2.0 in/s"),
    "osm_0_300ft": Limit("OSM, 0-300 ft (0-91 m)", 31.75, "mm/s", _OSM, "1.25 in/s"),
    "osm_301_5000ft": Limit("OSM, 301-5000 ft (92-1524 m)", 25.4, "mm/s", _OSM, "1.00 in/s"),
    "osm_beyond_5000ft": Limit("OSM, beyond 5000 ft (1524 m)", 19.05, "mm/s", _OSM, "0.75 in/s"),
    "din4150_industrial": Limit(
        "DIN 4150-3 line 1, commercial / industrial buildings, 1-10 Hz",
        20.0,
        "mm/s",
        _DIN,
        "20-40 mm/s at 10-50 Hz, 40-50 mm/s at 50-100 Hz",
    ),
    "din4150_residential": Limit(
        "DIN 4150-3 line 2, dwellings, 1-10 Hz",
        5.0,
        "mm/s",
        _DIN,
        "5-15 mm/s at 10-50 Hz, 15-20 mm/s at 50-100 Hz",
    ),
    "din4150_sensitive": Limit(
        "DIN 4150-3 line 3, sensitive / heritage structures, 1-10 Hz",
        3.0,
        "mm/s",
        _DIN,
        "3-8 mm/s at 10-50 Hz, 8-10 mm/s at 50-100 Hz",
    ),
    "as2187_sensitive_95pct": Limit(
        "AS 2187.2 human comfort, sensitive site, 95 % of blasts per year",
        5.0,
        "mm/s",
        _AS,
        "operations longer than 12 months",
    ),
    "as2187_sensitive_max": Limit(
        "AS 2187.2 human comfort, sensitive site, maximum", 10.0, "mm/s", _AS
    ),
    "as2187_commercial": Limit(
        "AS 2187.2 human comfort, occupied non-sensitive site (factory, commercial)",
        25.0,
        "mm/s",
        _AS,
    ),
}

_RI8485 = (
    "Siskind, Stachura, Stagg and Kopp 1980, USBM RI 8485, Structure response and damage "
    "produced by airblast from surface mining"
)

#: Air overpressure guideline limits, dBL (linear, peak).
AIRBLAST_LIMITS: dict[str, Limit] = {
    "as2187_sensitive_95pct": Limit(
        "AS 2187.2 human comfort, sensitive site, 95 % of blasts per year", 115.0, "dBL", _AS
    ),
    "as2187_sensitive_max": Limit(
        "AS 2187.2 human comfort, sensitive site, maximum", 120.0, "dBL", _AS
    ),
    "as2187_commercial": Limit(
        "AS 2187.2 human comfort, occupied non-sensitive site", 125.0, "dBL", _AS
    ),
    "as2187_damage": Limit("AS 2187.2 structural damage limit", 133.0, "dBL", _AS),
    "usbm_ri8485_2hz": Limit(
        "USBM RI 8485 / OSM, 2 Hz high-pass system", 133.0, "dBL", _RI8485, "30 CFR 816.67(b)"
    ),
    "usbm_ri8485_0_1hz": Limit(
        "USBM RI 8485 / OSM, 0.1 Hz high-pass system", 134.0, "dBL", _RI8485, "30 CFR 816.67(b)"
    ),
    "usbm_ri8485_6hz": Limit(
        "USBM RI 8485 / OSM, 6 Hz high-pass system", 129.0, "dBL", _RI8485, "30 CFR 816.67(b)"
    ),
}


def check(value: float | np.ndarray, limit: Limit | float) -> np.ndarray:
    """True where ``value`` is at or below the limit. A NaN value gives False (unknown is
    not a pass); test ``np.isnan`` to tell the two apart."""
    lim = limit.value if isinstance(limit, Limit) else float(limit)
    with np.errstate(invalid="ignore"):
        return np.asarray(value, dtype=np.float64) <= lim


# --------------------------------------------------------------------------- air overpressure

#: AS 2187.2-2006 Appendix J: site exponent of the cube-root law.
AIRBLAST_EXPONENT = 1.45
#: AS 2187.2-2006 Appendix J: K_a (P in kPa) for confined blasthole charges is 10 to 100.
AIRBLAST_KA_CONFINED_LOW = 10.0
AIRBLAST_KA_CONFINED_HIGH = 100.0


def pressure_to_dbl(pressure_pa: float | np.ndarray) -> np.ndarray:
    """``dBL = 20 * log10(P / 20e-6 Pa)``. NaN where the pressure is not positive."""
    p = np.asarray(pressure_pa, dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 20.0 * np.log10(np.where(p > 0, p, np.nan) / P_REF)


def dbl_to_pressure(dbl: float | np.ndarray) -> np.ndarray:
    """Peak pressure in Pa of a level in dBL."""
    return P_REF * 10.0 ** (np.asarray(dbl, dtype=np.float64) / 20.0)


def airblast_pressure(
    distance: float | np.ndarray,
    charge: float | np.ndarray,
    ka_kpa: float = AIRBLAST_KA_CONFINED_HIGH,
    a: float = AIRBLAST_EXPONENT,
) -> np.ndarray:
    """Peak air overpressure in **Pa**: ``P [kPa] = ka_kpa * (R / W**(1/3))**-a``.

    ``distance`` in m, ``charge`` in kg per delay. The default ``ka_kpa = 100`` is the upper
    end of the AS 2187.2 range for confined blasthole charges (the conservative end);
    uncalibrated.
    """
    sd = scaled_distance(distance, charge, 1.0 / 3.0)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1000.0 * ka_kpa * sd**-a


@dataclass
class AirblastResult:
    """Air overpressure at points: pressure (Pa), level (dBL), and the R, W used."""

    pressure_pa: np.ndarray
    dbl: np.ndarray
    scaled_distance: np.ndarray
    distance: np.ndarray
    charge_kg: float
    ka_kpa: float
    a: float
    calibrated: bool = False


def airblast_at(
    points: np.ndarray,
    design: BlastDesign,
    ka_kpa: float = AIRBLAST_KA_CONFINED_HIGH,
    a: float = AIRBLAST_EXPONENT,
    window_ms: float = 8.0,
    calibrated: bool = False,
) -> AirblastResult:
    """Air overpressure at ``points`` (n, 3): ``W`` = the maximum instantaneous charge,
    ``R`` = the 3D distance to the nearest collar of a charged hole (the air wave leaves at
    the surface). Wind, temperature inversions and the topography are not modelled."""
    p = _points(points)
    _, mic = charge_per_delay(design, window_ms)
    collars = np.array([h.collar for h in _charged(design)])
    dist = np.full(p.shape[0], np.inf)
    with np.errstate(invalid="ignore"):
        for start in range(0, collars.shape[0], 512):
            block = collars[start : start + 512]
            d = np.linalg.norm(p[:, None, :] - block[None, :, :], axis=2)
            dist = np.fmin(dist, d.min(axis=1))
    dist = np.where(np.isnan(p).any(axis=1), np.nan, dist)
    pressure = airblast_pressure(dist, mic, ka_kpa, a)
    return AirblastResult(
        pressure_pa=pressure,
        dbl=pressure_to_dbl(pressure),
        scaled_distance=scaled_distance(dist, mic, 1.0 / 3.0),
        distance=dist,
        charge_kg=mic,
        ka_kpa=ka_kpa,
        a=a,
        calibrated=calibrated,
    )
