"""Fragment size distribution of a bench blast: the Kuz-Ram model.

research_blast.md part 2, section 1. Three published pieces are combined (Cunningham 1983,
1987, 2005):

Mean size (Kuznetsov 1973, in Cunningham's relative-weight-strength form)
-------------------------------------------------------------------------
``x50 [cm] = A * K**-0.8 * Q**(1/6) * (115 / RWS)**(19/20)``

with ``A`` the rock factor, ``K`` the powder factor in kg of explosive per m3 of rock, ``Q``
the explosive mass per hole in kg and ``RWS`` the relative weight strength of the explosive
(ANFO = 100, TNT = 115). The original is in centimetres; :func:`kuznetsov_x50` returns
**metres**.

Rock factor (Lilly 1986 blastability index as used by Cunningham 1987)
----------------------------------------------------------------------
``A = 0.06 * (RMD + RDI + HF)``, with ``RMD = JF = JPS + JPA`` for a vertically jointed rock
mass, see :func:`rock_factor`.

Uniformity index (Cunningham 1987)
----------------------------------
``n = (2.2 - 14 B/d) * sqrt((1 + S/B) / 2) * (1 - W/B) * (|BCL - CCL| / L + 0.1)**0.1 * L/H``

with burden ``B`` and spacing ``S`` in metres, hole diameter ``d`` in **millimetres** in the
original (this module takes metres and converts), ``W`` the standard deviation of drilling
accuracy in metres, ``BCL`` / ``CCL`` the bottom and column charge lengths, ``L`` the total
charge length and ``H`` the bench height. A staggered pattern multiplies ``n`` by 1.1.

Size distribution (Rosin and Rammler 1933)
------------------------------------------
``P(x) = 1 - exp(-ln 2 * (x / x50)**n)``: the fraction passing a mesh of size ``x``. The
literature writes the constant as 0.693; this module uses ``ln 2`` so that ``P(x50) = 0.5``
exactly. The characteristic size ``xc = x50 / (ln 2)**(1/n)`` has ``P(xc) = 1 - 1/e``.

An alternative curve, the Swebrec function (Ouchterlony 2005), is given as :func:`swebrec`.

The model is empirical. The rock factor is a site constant: values from
:func:`rock_factor_from_rock` are estimates and stay "uncalibrated" until fitted to measured
fragmentation. Kuz-Ram is known to underestimate fines (Cunningham 2005; Ouchterlony 2005).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from bakumine.blast.model import BlastDesign, Rock

__all__ = [
    "LN2",
    "STAGGERED_FACTOR",
    "FragmentationResult",
    "RosinRammler",
    "kuznetsov_x50",
    "predict",
    "rock_factor",
    "rock_factor_from_rock",
    "swebrec",
    "uniformity_index",
]

#: The Rosin-Rammler constant that makes ``x50`` the median (0.693 in the literature).
LN2 = math.log(2.0)
#: Cunningham (1987): the uniformity index of a staggered pattern is 10 % higher.
STAGGERED_FACTOR = 1.1


def kuznetsov_x50(
    rock_factor: float, powder_factor: float, charge_mass: float, rws: float = 100.0
) -> float:
    """Mean (median) fragment size in **metres**.

    ``x50 [cm] = A * K**-0.8 * Q**(1/6) * (115 / RWS)**(19/20)`` with ``powder_factor`` K in
    kg/m3, ``charge_mass`` Q in kg per hole and ``rws`` the relative weight strength
    (ANFO = 100). The centimetre result of the published formula is divided by 100.
    """
    for name, value in (
        ("rock_factor", rock_factor),
        ("powder_factor", powder_factor),
        ("charge_mass", charge_mass),
        ("rws", rws),
    ):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be > 0, got {value!r}")
    x50_cm = rock_factor * powder_factor**-0.8 * charge_mass ** (1.0 / 6.0) * (115.0 / rws) ** 0.95
    return x50_cm / 100.0


def rock_factor(
    density: float,
    *,
    rmd: float | None = None,
    jps: float | None = None,
    jpa: float | None = None,
    youngs_gpa: float | None = None,
    ucs_mpa: float | None = None,
) -> float:
    """Rock factor ``A = 0.06 * (RMD + RDI + HF)`` (Cunningham 1987 after Lilly 1986).

    Parameters
    ----------
    density
        Rock density in kg/m3. ``RDI = 25 * rho - 50`` with ``rho`` in t/m3.
    rmd
        Rock mass description: 10 powdery / friable, 50 massive. For a vertically jointed
        rock mass leave it ``None`` and give ``jps`` and ``jpa``: then ``RMD = JF``.
    jps, jpa
        Joint factor ``JF = JPS + JPA``. Vertical joint spacing JPS: 10 for < 0.1 m, 20 for
        0.1 m to the oversize limit, 50 for oversize to the drill pattern size. Joint plane
        angle JPA: 20 dip out of the face, 30 strike perpendicular to the face, 40 dip into
        the face.
    youngs_gpa, ucs_mpa
        Hardness factor: ``HF = E / 3`` when ``E < 50 GPa``, else ``HF = UCS / 5`` (UCS in
        MPa). With only one of the two given, that one is used.

    In Cunningham's papers the joint factor takes the place of RMD for jointed rock; it is
    not added to a separate RMD (that would be Lilly's original index, which has other
    class values). The published range of ``A`` is about 0.8 (very soft) to 22 (very hard,
    massive); the value is returned as computed, not clipped.
    """
    if rmd is None:
        if jps is None or jpa is None:
            raise ValueError("rock_factor needs rmd, or jps and jpa for a jointed rock mass")
        rmd = jps + jpa
    elif jps is not None or jpa is not None:
        raise ValueError("give rmd (friable 10, massive 50) or jps and jpa, not both")
    if youngs_gpa is None and ucs_mpa is None:
        raise ValueError("rock_factor needs youngs_gpa or ucs_mpa for the hardness factor")
    if youngs_gpa is not None and (youngs_gpa < 50.0 or ucs_mpa is None):
        hf = youngs_gpa / 3.0
    else:
        assert ucs_mpa is not None
        hf = ucs_mpa / 5.0
    rdi = 25.0 * (density / 1000.0) - 50.0
    return 0.06 * (rmd + rdi + hf)


def rock_factor_from_rock(
    rock: Rock,
    *,
    oversize: float = 1.0,
    pattern_size: float | None = None,
    jpa: float = 30.0,
    friable: bool = False,
) -> float:
    """Rock factor of a :class:`~bakumine.blast.model.Rock`; ``rock.rock_factor`` wins when set.

    Otherwise an **estimate** from the fields the model has (uncalibrated; state it as such):

    * RMD = 10 when ``friable``; 50 (massive) when ``pattern_size`` is given and
      ``rock.joint_spacing`` is at least that large; else the joint factor ``JPS + JPA``
      with JPS from ``rock.joint_spacing`` (10 below 0.1 m, 20 below ``oversize`` in metres,
      else 50) and JPA = ``jpa`` (default 30, strike perpendicular to the face: the middle
      class, since the model does not carry the joint orientation);
    * RDI from ``rock.density``; HF from ``rock.youngs_gpa`` / ``rock.ucs_mpa``.
    """
    if rock.rock_factor is not None:
        return float(rock.rock_factor)
    kwargs = {"youngs_gpa": rock.youngs_gpa, "ucs_mpa": rock.ucs_mpa}
    if friable:
        return rock_factor(rock.density, rmd=10.0, **kwargs)
    if pattern_size is not None and rock.joint_spacing >= pattern_size:
        return rock_factor(rock.density, rmd=50.0, **kwargs)
    if rock.joint_spacing < 0.1:
        jps = 10.0
    elif rock.joint_spacing < oversize:
        jps = 20.0
    else:
        jps = 50.0
    return rock_factor(rock.density, jps=jps, jpa=jpa, **kwargs)


def uniformity_index(
    burden: float,
    spacing: float,
    diameter: float,
    charge_length: float,
    bench_height: float,
    *,
    drill_deviation: float = 0.0,
    bottom_charge_length: float | None = None,
    column_charge_length: float | None = None,
    staggered: bool = False,
) -> float:
    """Cunningham (1987) uniformity index ``n`` of the Rosin-Rammler curve.

    ``n = (2.2 - 14 B/d) * sqrt((1 + S/B)/2) * (1 - W/B) * (|BCL - CCL|/L + 0.1)**0.1 * L/H``

    All lengths in **metres**, including ``diameter`` (the published formula has ``d`` in mm
    and ``B`` in m; the conversion is done here). ``drill_deviation`` is ``W``, the standard
    deviation of drilling accuracy. ``bottom_charge_length`` / ``column_charge_length``
    default to a single column charge (``BCL = 0``, ``CCL = L``). ``staggered`` multiplies by
    1.1.

    Returns NaN when the formula gives ``n <= 0`` (burden very large against the diameter,
    or deviation >= burden): the model does not apply there.
    """
    for name, value in (
        ("burden", burden),
        ("spacing", spacing),
        ("diameter", diameter),
        ("charge_length", charge_length),
        ("bench_height", bench_height),
    ):
        if not (math.isfinite(value) and value > 0):
            raise ValueError(f"{name} must be > 0, got {value!r}")
    if not (math.isfinite(drill_deviation) and drill_deviation >= 0):
        raise ValueError(f"drill_deviation must be >= 0, got {drill_deviation!r}")
    if bottom_charge_length is None and column_charge_length is None:
        bcl, ccl = 0.0, charge_length
    elif bottom_charge_length is None:
        ccl = float(column_charge_length)  # type: ignore[arg-type]
        bcl = charge_length - ccl
    elif column_charge_length is None:
        bcl = float(bottom_charge_length)
        ccl = charge_length - bcl
    else:
        bcl, ccl = float(bottom_charge_length), float(column_charge_length)
    d_mm = diameter * 1000.0
    n = (
        (2.2 - 14.0 * burden / d_mm)
        * math.sqrt((1.0 + spacing / burden) / 2.0)
        * (1.0 - drill_deviation / burden)
        * (abs(bcl - ccl) / charge_length + 0.1) ** 0.1
        * (charge_length / bench_height)
    )
    if staggered:
        n *= STAGGERED_FACTOR
    return n if n > 0 else math.nan


@dataclass(frozen=True)
class RosinRammler:
    """Rosin-Rammler distribution ``P(x) = 1 - exp(-ln 2 * (x / x50)**n)``; sizes in metres."""

    x50: float
    n: float

    def __post_init__(self) -> None:
        if not (math.isfinite(self.x50) and self.x50 > 0):
            raise ValueError(f"x50 must be > 0, got {self.x50!r}")
        if not (math.isfinite(self.n) and self.n > 0):
            raise ValueError(f"uniformity index n must be > 0, got {self.n!r}")

    @property
    def xc(self) -> float:
        """Characteristic size: ``P(xc) = 1 - 1/e`` (63.2 % passing)."""
        return self.x50 / LN2 ** (1.0 / self.n)

    def passing(self, x: float | np.ndarray) -> np.ndarray:
        """Fraction (0..1) finer than size ``x``. NaN stays NaN; ``x <= 0`` gives 0."""
        x = np.asarray(x, dtype=np.float64)
        ratio = np.where(x > 0, x, 0.0) / self.x50
        out = -np.expm1(-LN2 * ratio**self.n)
        return np.where(np.isnan(x), np.nan, out)

    def size_at(self, p: float | np.ndarray) -> np.ndarray:
        """Size with fraction ``p`` passing (``size_at(0.8)`` is P80). NaN outside [0, 1]."""
        p = np.asarray(p, dtype=np.float64)
        ok = (p >= 0.0) & (p <= 1.0)
        with np.errstate(divide="ignore", invalid="ignore"):
            x = self.x50 * (-np.log1p(-np.where(ok, p, np.nan)) / LN2) ** (1.0 / self.n)
        return x

    def curve(
        self, n_points: int = 60, p_min: float = 0.001, p_max: float = 0.999
    ) -> tuple[np.ndarray, np.ndarray]:
        """``(sizes, passing)``: ``n_points`` log-spaced sizes from ``size_at(p_min)`` to
        ``size_at(p_max)`` and the fraction passing each."""
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        if not 0.0 < p_min < p_max < 1.0:
            raise ValueError("need 0 < p_min < p_max < 1")
        lo, hi = float(self.size_at(p_min)), float(self.size_at(p_max))
        sizes = np.geomspace(lo, hi, n_points)
        return sizes, self.passing(sizes)

    def fraction_above(self, oversize: float | np.ndarray) -> np.ndarray:
        """Fraction coarser than ``oversize``."""
        x = np.asarray(oversize, dtype=np.float64)
        ratio = np.where(x > 0, x, 0.0) / self.x50
        return np.where(np.isnan(x), np.nan, np.exp(-LN2 * ratio**self.n))

    def fraction_below(self, fines: float | np.ndarray) -> np.ndarray:
        """Fraction finer than ``fines`` (the same as :meth:`passing`)."""
        return self.passing(fines)


def swebrec(x: float | np.ndarray, x50: float, xmax: float, b: float) -> np.ndarray:
    """Swebrec function (Ouchterlony 2005): fraction passing size ``x``.

    ``P(x) = 1 / (1 + (ln(xmax / x) / ln(xmax / x50))**b)`` for ``0 < x <= xmax``; 0 at
    ``x <= 0`` and 1 at ``x >= xmax``. ``xmax`` is the largest fragment and ``b`` the curve
    undulation parameter. ``P(x50) = 0.5`` and ``P(xmax) = 1`` by construction.
    """
    if not (0.0 < x50 < xmax):
        raise ValueError("need 0 < x50 < xmax")
    if not b > 0:
        raise ValueError("b must be > 0")
    x = np.asarray(x, dtype=np.float64)
    inside = (x > 0) & (x < xmax)
    safe = np.where(inside, x, x50)
    f = (np.log(xmax / safe) / math.log(xmax / x50)) ** b
    out = np.where(inside, 1.0 / (1.0 + f), np.where(x >= xmax, 1.0, 0.0))
    return np.where(np.isnan(x), np.nan, out)


@dataclass
class FragmentationResult:
    """Kuz-Ram prediction for a design. Sizes in metres, fractions 0..1.

    ``oversize_fraction`` / ``fines_fraction`` are NaN when the limit was not given.
    ``inputs`` holds every number that went into the formulas.
    """

    x50: float
    n: float
    xc: float
    p80: float
    p20: float
    sizes: np.ndarray
    passing: np.ndarray
    rock_factor: float
    rock_factor_calibrated: bool
    oversize_fraction: float = math.nan
    fines_fraction: float = math.nan
    inputs: dict[str, Any] = field(default_factory=dict)
    model: str = "Kuz-Ram (Kuznetsov 1973; Cunningham 1987; Rosin-Rammler)"

    @property
    def distribution(self) -> RosinRammler:
        return RosinRammler(self.x50, self.n)

    def to_dict(self) -> dict[str, Any]:
        """The ``fragmentation`` block of the result bundle (plan_blast.md section 3)."""
        return {
            "sizes_m": self.sizes.tolist(),
            "passing": self.passing.tolist(),
            "x50_m": self.x50,
            "n": self.n,
            "xc_m": self.xc,
        }


def _design_length(design_value: float, hole_values: list[float], name: str) -> float:
    if math.isfinite(design_value) and design_value > 0:
        return float(design_value)
    finite = [v for v in hole_values if math.isfinite(v) and v > 0]
    if not finite:
        raise ValueError(f"the design has no {name} (neither on the design nor on the holes)")
    return float(np.mean(finite))


def predict(
    design: BlastDesign,
    drill_deviation: float = 0.0,
    oversize: float | None = None,
    fines: float | None = None,
    *,
    n_points: int = 60,
) -> FragmentationResult:
    """Kuz-Ram fragmentation of ``design`` from the mean values of its charged holes.

    * ``K`` = ``design.powder_factor`` (total explosive / in-situ volume);
    * ``Q`` = mean charge mass per hole; ``RWS`` = charge-mass-weighted mean;
    * ``B``, ``S`` = the design burden and spacing (else the mean of the holes');
    * ``d``, ``L`` = mean diameter and charge length; one explosive per hole, so the charge
      is a single column (``BCL = 0``, ``CCL = L``); ``H`` = ``design.bench_height``;
    * ``A`` = ``design.rock.rock_factor`` when set (taken as calibrated), else the estimate
      of :func:`rock_factor_from_rock` with ``oversize`` (default 1 m) and the pattern size
      ``max(B, S)``.

    ``oversize`` and ``fines`` are sizes in metres for the reported fractions.
    """
    holes = [h for h in design.holes if h.charge_mass > 0]
    if not holes:
        raise ValueError("the design has no charged holes")
    burden = _design_length(design.burden, [h.burden for h in holes], "burden")
    spacing = _design_length(design.spacing, [h.spacing for h in holes], "spacing")
    powder_factor = design.powder_factor
    if not (math.isfinite(powder_factor) and powder_factor > 0):
        raise ValueError("the design has no powder factor (volume unknown or zero)")
    masses = np.array([h.charge_mass for h in holes])
    charge_mass = float(masses.mean())
    rws = float(np.sum(masses * [h.explosive.rws for h in holes]) / masses.sum())
    diameter = float(np.mean([h.diameter for h in holes]))
    charge_length = float(np.mean([h.charge_length for h in holes]))
    staggered = design.pattern == "staggered"

    calibrated = design.rock.rock_factor is not None
    a = rock_factor_from_rock(
        design.rock,
        oversize=1.0 if oversize is None else float(oversize),
        pattern_size=max(burden, spacing),
    )
    x50 = kuznetsov_x50(a, powder_factor, charge_mass, rws)
    n = uniformity_index(
        burden,
        spacing,
        diameter,
        charge_length,
        design.bench_height,
        drill_deviation=drill_deviation,
        staggered=staggered,
    )
    if not math.isfinite(n):
        raise ValueError(
            "Cunningham's uniformity index is not positive for this design "
            f"(B={burden:g} m, d={diameter:g} m, W={drill_deviation:g} m): outside the model"
        )
    rr = RosinRammler(x50, n)
    sizes, passing = rr.curve(n_points)
    return FragmentationResult(
        x50=x50,
        n=n,
        xc=rr.xc,
        p80=float(rr.size_at(0.8)),
        p20=float(rr.size_at(0.2)),
        sizes=sizes,
        passing=passing,
        rock_factor=a,
        rock_factor_calibrated=calibrated,
        oversize_fraction=math.nan if oversize is None else float(rr.fraction_above(oversize)),
        fines_fraction=math.nan if fines is None else float(rr.fraction_below(fines)),
        inputs={
            "powder_factor_kg_m3": powder_factor,
            "charge_per_hole_kg": charge_mass,
            "rws": rws,
            "burden_m": burden,
            "spacing_m": spacing,
            "diameter_m": diameter,
            "charge_length_m": charge_length,
            "bench_height_m": design.bench_height,
            "drill_deviation_m": drill_deviation,
            "staggered": staggered,
            "oversize_m": oversize,
            "fines_m": fines,
        },
    )
