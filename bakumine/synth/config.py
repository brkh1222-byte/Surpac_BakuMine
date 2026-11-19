"""Parameters of the synthetic benchmark deposit.

Everything that shapes the generated data lives here so that ``truth.json`` can record it
verbatim. Angles are degrees, lengths metres, grades g/t, densities t/m3.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any


@dataclass(frozen=True)
class LensConfig:
    """Ore solid: a bent ellipsoidal lens with a thickness neck (the pinch)."""

    centre: tuple[float, float, float] = (500_600.0, 5_300_500.0, 1_000.0)
    bearing: float = 30.0  # azimuth of the long axis, clockwise from north
    plunge: float = 15.0  # of the long axis, positive downward
    dip: float = 55.0  # rotation of the down-dip axis about the long axis
    half_length: float = 260.0  # along the long (plunging) axis
    half_width: float = 140.0  # down dip
    half_thickness: float = 16.0  # across the lens, before the pinch
    pinch_position: float = 70.0  # along the long axis, from the centre
    pinch_width: float = 45.0
    pinch_depth: float = 0.75  # thickness is multiplied by (1 - depth) at the pinch
    bend_major: float = 18.0  # mid-surface offset at the ends of the long axis
    bend_semi: float = -10.0  # mid-surface offset at the up/down-dip edges
    mesh_subdivisions: int = 5  # icosphere level: 20 * 4**n faces


@dataclass(frozen=True)
class VariogramConfig:
    """Variogram of the Gaussian (normal-score) field. Total sill is 1."""

    nugget: float = 0.15
    ranges: tuple[float, float, float] = (120.0, 70.0, 15.0)  # major, semi-major, minor
    bearing: float = 35.0
    plunge: float = 10.0
    dip: float = 50.0
    cell: tuple[float, float, float] = (4.0, 2.5, 0.5)  # simulation lattice, variogram axes


@dataclass(frozen=True)
class GradeConfig:
    ore_median: float = 1.6
    ore_sigma: float = 0.8  # standard deviation of ln(grade) in ore
    waste_median: float = 0.02
    waste_sigma: float = 0.5
    ore_density: float = 2.85
    waste_density: float = 2.70
    assay_decimals: int = 3
    assay_floor: float = 0.001


@dataclass(frozen=True)
class TopoConfig:
    origin: tuple[float, float] = (500_000.0, 5_300_000.0)
    base: float = 1_200.0
    slope: tuple[float, float] = (0.02, -0.012)
    # (amplitude m, wavelength m, direction deg clockwise from north, phase rad)
    waves: tuple[tuple[float, float, float, float], ...] = (
        (14.0, 900.0, 70.0, 0.4),
        (8.0, 420.0, 340.0, 1.3),
        (3.0, 160.0, 30.0, 2.1),
    )
    dtm_spacing: float = 25.0
    dtm_margin: float = 100.0
    min_cover: float = 10.0  # the ore solid must stay this far below the surface


@dataclass(frozen=True)
class DrillConfig:
    n_sections: int = 10
    holes_per_section: int = 8
    coverage: float = 0.92  # targets span this fraction of the lens half-axes
    target_jitter: float = 4.0
    collar_jitter: float = 2.0
    dip: float = -60.0  # planned dip, negative downward
    dip_sd: float = 1.5
    azimuth_sd: float = 2.0
    vertical_fraction: float = 0.08
    station_spacing: float = 30.0
    lift_mean: float = 0.45  # degrees of flattening per station
    lift_sd: float = 0.20
    swing_mean: float = 0.50  # degrees of azimuth drift per station
    swing_sd: float = 0.35
    wander_sd: float = 0.15  # random walk per station, both angles
    overshoot: tuple[float, float] = (40.0, 90.0)  # drilled past the target
    sample_regular_fraction: float = 0.65  # share of samples that are exactly 1.0 m
    sample_start_max: float = 6.0  # unsampled length below the collar, up to
    gap_probabilities: tuple[float, float, float] = (0.50, 0.35, 0.15)  # 0, 1, 2 gaps
    gap_samples: tuple[int, int] = (2, 8)
    sample_subpoints: int = 4
    contact_scan_step: float = 0.25


@dataclass(frozen=True)
class BlockConfig:
    size: tuple[float, float, float] = (10.0, 10.0, 5.0)
    discretisation: tuple[int, int, int] = (5, 5, 5)
    margin: float = 40.0


@dataclass(frozen=True)
class SynthConfig:
    lens: LensConfig = field(default_factory=LensConfig)
    variogram: VariogramConfig = field(default_factory=VariogramConfig)
    grade: GradeConfig = field(default_factory=GradeConfig)
    topo: TopoConfig = field(default_factory=TopoConfig)
    drill: DrillConfig = field(default_factory=DrillConfig)
    blocks: BlockConfig = field(default_factory=BlockConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def default_config() -> SynthConfig:
    return SynthConfig()


def small_config() -> SynthConfig:
    """A reduced deposit that generates in about a second. Used by the test suite."""
    base = SynthConfig()
    return replace(
        base,
        lens=replace(
            base.lens,
            centre=(500_400.0, 5_300_350.0, 1_080.0),
            half_length=150.0,
            half_width=80.0,
            half_thickness=12.0,
            pinch_position=40.0,
            pinch_width=28.0,
            bend_major=10.0,
            bend_semi=-6.0,
            mesh_subdivisions=4,
        ),
        variogram=replace(base.variogram, ranges=(60.0, 35.0, 10.0), cell=(3.0, 2.0, 0.75)),
        blocks=replace(base.blocks, discretisation=(3, 3, 3), margin=20.0),
    )


PRESETS = {"default": default_config, "small": small_config}
