"""The synthetic layered deposit: an exact truth for the geology model (geology_contracts.md 13).

Every point of space gets one rock unit, one weathering zone and one fault block from closed
formulas, so the truth does not depend on any mesh or grid. Notation: ``X = x - 500000``,
``Y = y - 5300000``; azimuths are degrees clockwise from north; ``x = east, y = north, z = up``.

Rock units, applied in this order (a later one wins):

1. Layers, at the restored position ``q``: ``s = n_b . (q - P0)``, ``u = t . (q - P0)``,
   ``s_BC(u) = -(T + A sin(2 pi u / W))``; ``SST`` if ``s >= 0``, ``SLT`` if
   ``s_BC <= s < 0``, ``AND`` if ``s < s_BC``.
2. Fault F1: hanging wall (``FB2``) where ``n_f . (p - Pf) > 0``; there ``q = p - D`` with
   ``D`` the slip vector (down dip), elsewhere ``q = p``.
3. ``DIO`` stock: an ellipsoid, not offset by the fault.
4. ``GRD`` dyke: a vertical slab, not offset by the fault.
5. Cover ``Q4``: depth below the topography smaller than ``c(x, y)``.
6. ``AIR`` above the topography.

The weathering zones depend only on the depth below the topography. The ore lens is a separate
attribute (``rock``), not a rock unit.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from typing import Any

import numpy as np

from bakumine.synth.config import LensConfig, TopoConfig
from bakumine.synth.solid import LensSolid
from bakumine.synth.topo import Topography

LITH = ("SST", "SLT", "AND", "DIO", "GRD", "Q4", "AIR")
WEATHERING = ("OXIDE", "TRANSITION", "FRESH", "AIR")
FAULT_BLOCKS = ("FB1", "FB2")
ROCK = ("WASTE", "ORE", "AIR")
#: label layers of :meth:`LayeredTruth.labels`; ``bedrock`` is ``lith`` without the cover
CATEGORIES: dict[str, tuple[str, ...]] = {
    "lith": LITH,
    "bedrock": LITH,
    "weathering": WEATHERING,
    "fault_block": FAULT_BLOCKS,
    "rock": ROCK,
}
CONTACTS = ("SST/SLT", "SLT/AND")

_SST, _SLT, _AND, _DIO, _GRD, _Q4, _AIR = range(7)


def _layered_lens() -> LensConfig:
    return LensConfig(
        centre=(500_450.0, 5_300_400.0, 1_095.0),
        bearing=70.0,
        plunge=0.0,
        dip=20.0,
        half_length=120.0,
        half_width=60.0,
        half_thickness=8.0,
        pinch_position=30.0,
        pinch_width=25.0,
        pinch_depth=0.5,
        bend_major=6.0,
        bend_semi=-4.0,
        mesh_subdivisions=4,
    )


@dataclass(frozen=True)
class LayeredConfig:
    """Every number of the layered deposit. Angles in degrees, lengths in metres."""

    name: str = "layered"
    origin: tuple[float, float] = (500_000.0, 5_300_000.0)  # of X and Y in the formulas
    # box and drilling
    x_range: tuple[float, float] = (500_000.0, 500_800.0)
    y_range: tuple[float, float] = (5_300_000.0, 5_300_800.0)
    z_range: tuple[float, float] = (880.0, 1_260.0)
    hole_spacing: float = 50.0
    hole_length: float = 200.0
    hole_dip: float = -60.0  # negative downward
    hole_azimuth: float = 160.0
    vertical_modulus: int = 10  # hole h is vertical when h % modulus == remainder
    vertical_remainder: int = 4
    vertical_length_factor: float = 0.8
    topo: TopoConfig = field(default_factory=TopoConfig)
    # layers
    bedding_dip: float = 20.0
    bedding_dip_azimuth: float = 340.0
    bedding_point: tuple[float, float, float] = (500_400.0, 5_300_400.0, 1_110.0)  # P0
    bedding_strike_azimuth: float = 70.0  # direction t
    slt_thickness: float = 40.0
    slt_amplitude: float = 10.0
    slt_wavelength: float = 300.0
    # fault F1
    fault_point: tuple[float, float, float] = (500_450.0, 5_300_400.0, 1_100.0)  # Pf
    fault_dip: float = 65.0
    fault_dip_azimuth: float = 70.0
    fault_throw: float = 30.0  # vertical component of the slip, down in the hanging wall
    # DIO stock
    stock_centre: tuple[float, float, float] = (500_300.0, 5_300_480.0, 1_060.0)
    stock_semi_axes: tuple[float, float, float] = (110.0, 80.0, 150.0)  # along, across, vertical
    stock_azimuth: float = 45.0  # of the first horizontal semi-axis
    # GRD dyke
    dyke_point: tuple[float, float, float] = (500_480.0, 5_300_300.0, 0.0)  # Q
    dyke_normal_azimuth: float = 135.0  # m; the strike is 045
    dyke_half_thickness: float = 4.0
    # cover: c = mean + amplitude sin(2 pi X / wx) sin(2 pi Y / wy)
    cover_mean: float = 4.0
    cover_amplitude: float = 2.0
    cover_wavelengths: tuple[float, float] = (300.0, 250.0)
    # oxide base: d_ox = mean + ax sin(2 pi X / wx + phase) + ay cos(2 pi Y / wy)
    oxide_mean: float = 22.5
    oxide_amplitude_x: float = 5.0
    oxide_wavelength_x: float = 500.0
    oxide_phase_x: float = 0.7
    oxide_amplitude_y: float = 2.5
    oxide_wavelength_y: float = 350.0
    transition_thickness: float = 5.0
    lens: LensConfig = field(default_factory=_layered_lens)
    # logging
    contact_scan_step: float = 0.25
    contact_tolerance: float = 1e-6
    depth_decimals: int = 2
    fault_zone_half_length: float = 1.0
    core_loss_rate: float = 0.02  # per metre (Poisson)
    core_loss_length: tuple[float, float] = (0.5, 2.0)
    # assays
    sample_length: float = 1.0
    ore_median: float = 1.5
    ore_sigma: float = 0.6
    waste_median: float = 0.02
    waste_sigma: float = 0.5
    assay_decimals: int = 3
    assay_floor: float = 0.001
    sulphur_ppm: tuple[float, float, float] = (300.0, 1_600.0, 6_000.0)  # OXIDE, TRANSITION, FRESH
    sulphur_sigma: float = 0.1
    # truth outputs
    block_size: tuple[float, float, float] = (10.0, 10.0, 5.0)
    block_lattice: tuple[int, int, int] = (4, 4, 4)
    surface_spacing: float = 10.0
    volume_spacings: tuple[float, float] = (2.0, 1.0)
    # self-checks of the generator
    min_holes_per_unit: int = 20
    # The stock is cut by at most 14 holes of the 50 m grid whatever the hole length (its
    # shadow along the holes is about 16 grid cells), so 20 cannot be met for DIO.
    min_holes_exceptions: tuple[tuple[str, int], ...] = (("DIO", 12),)
    min_fault_holes: int = 15
    min_lens_cover: float = 10.0
    volume_tolerance: float = 0.01

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


LAYERED_PRESETS: dict[str, LayeredConfig] = {
    "layered": LayeredConfig(),
    "layered-small": replace(
        LayeredConfig(),
        name="layered-small",
        x_range=(500_200.0, 500_600.0),
        y_range=(5_300_200.0, 5_300_600.0),
        z_range=(940.0, 1_250.0),
        hole_length=180.0,
        min_holes_per_unit=8,
        min_holes_exceptions=(),
        min_fault_holes=6,
    ),
}


def _horizontal(azimuth: float) -> np.ndarray:
    a = np.radians(azimuth)
    return np.array([np.sin(a), np.cos(a), 0.0])


def _upward_normal(dip: float, dip_azimuth: float) -> np.ndarray:
    d, a = np.radians(dip), np.radians(dip_azimuth)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


@dataclass(frozen=True)
class ColumnPlan:
    """Everything that depends on (x, y) only, so that many elevations are cheap to label."""

    ground: np.ndarray  # topography
    cover: np.ndarray  # cover depth c
    oxide: np.ndarray  # oxide depth d_ox
    bedding: np.ndarray  # horizontal part of n_b . (p - P0)
    fault: np.ndarray  # horizontal part of n_f . (p - Pf)
    base_fw: np.ndarray  # s_BC in the footwall
    base_hw: np.ndarray  # s_BC in the hanging wall (at the restored position)
    stock: np.ndarray  # horizontal part of the ellipsoid equation
    dyke: np.ndarray  # bool


class LayeredTruth:
    """Exact labels of the layered deposit at any point."""

    def __init__(self, cfg: LayeredConfig) -> None:
        self.cfg = cfg
        self.topo = Topography(cfg.topo)
        self.lens = LensSolid.from_config(cfg.lens)
        self.bedding_normal = _upward_normal(cfg.bedding_dip, cfg.bedding_dip_azimuth)
        self.bedding_point = np.array(cfg.bedding_point, dtype=np.float64)
        self.strike = _horizontal(cfg.bedding_strike_azimuth)
        self.fault_point = np.array(cfg.fault_point, dtype=np.float64)
        self.fault_normal = _upward_normal(cfg.fault_dip, cfg.fault_dip_azimuth)
        d, a = np.radians(cfg.fault_dip), np.radians(cfg.fault_dip_azimuth)
        self.fault_down_dip = np.array([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), -np.sin(d)])
        self.slip = cfg.fault_throw / float(np.sin(d))
        self.slip_vector = self.slip * self.fault_down_dip
        self.stock_centre = np.array(cfg.stock_centre, dtype=np.float64)
        self._stock_axes = (_horizontal(cfg.stock_azimuth), _horizontal(cfg.stock_azimuth + 90.0))
        self.dyke_point = np.array(cfg.dyke_point, dtype=np.float64)
        self.dyke_normal = _horizontal(cfg.dyke_normal_azimuth)

    # ------------------------------------------------------------------ surfaces

    def topo_z(self, x, y) -> np.ndarray:
        return self.topo.z(x, y)

    def cover_depth(self, x, y) -> np.ndarray:
        cfg = self.cfg
        dx = np.asarray(x, dtype=np.float64) - cfg.origin[0]
        dy = np.asarray(y, dtype=np.float64) - cfg.origin[1]
        wx, wy = cfg.cover_wavelengths
        return cfg.cover_mean + cfg.cover_amplitude * np.sin(2.0 * np.pi * dx / wx) * np.sin(
            2.0 * np.pi * dy / wy
        )

    def oxide_depth(self, x, y) -> np.ndarray:
        cfg = self.cfg
        dx = np.asarray(x, dtype=np.float64) - cfg.origin[0]
        dy = np.asarray(y, dtype=np.float64) - cfg.origin[1]
        return (
            cfg.oxide_mean
            + cfg.oxide_amplitude_x
            * np.sin(2.0 * np.pi * dx / cfg.oxide_wavelength_x + cfg.oxide_phase_x)
            + cfg.oxide_amplitude_y * np.cos(2.0 * np.pi * dy / cfg.oxide_wavelength_y)
        )

    def transition_depth(self, x, y) -> np.ndarray:
        return self.oxide_depth(x, y) + self.cfg.transition_thickness

    # ------------------------------------------------------------------ fault

    def hanging_wall(self, points) -> np.ndarray:
        p = np.asarray(points, dtype=np.float64)
        return (p - self.fault_point) @ self.fault_normal > 0.0

    def restored(self, points) -> np.ndarray:
        """Position before the fault moved: ``p - D`` in the hanging wall, ``p`` elsewhere."""
        p = np.asarray(points, dtype=np.float64)
        return p - self.hanging_wall(p)[..., None] * self.slip_vector

    def _layer_base(self, u: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        return -(
            cfg.slt_thickness + cfg.slt_amplitude * np.sin(2.0 * np.pi * u / cfg.slt_wavelength)
        )

    def contact_z(self, x, y, contact: str = "SST/SLT", block: str = "FB1") -> np.ndarray:
        """Elevation of a layer contact at (x, y) as if the whole column were in ``block``."""
        if contact not in CONTACTS:
            raise ValueError(f"contact must be one of {CONTACTS}, not {contact!r}")
        if block not in FAULT_BLOCKS:
            raise ValueError(f"block must be one of {FAULT_BLOCKS}, not {block!r}")
        shift = self.slip_vector if block == "FB2" else np.zeros(3)
        qx = np.asarray(x, dtype=np.float64) - shift[0] - self.bedding_point[0]
        qy = np.asarray(y, dtype=np.float64) - shift[1] - self.bedding_point[1]
        n = self.bedding_normal
        s = 0.0
        if contact == "SLT/AND":
            s = self._layer_base(self.strike[0] * qx + self.strike[1] * qy)
        return self.bedding_point[2] + (s - n[0] * qx - n[1] * qy) / n[2] + shift[2]

    # ------------------------------------------------------------------ labels

    def plan(self, x, y) -> ColumnPlan:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        bx, by = x - self.bedding_point[0], y - self.bedding_point[1]
        u = self.strike[0] * bx + self.strike[1] * by
        sx, sy = x - self.stock_centre[0], y - self.stock_centre[1]
        e1, e2 = self._stock_axes
        a, b, _ = self.cfg.stock_semi_axes
        m, q = self.dyke_normal, self.dyke_point
        return ColumnPlan(
            ground=self.topo.z(x, y),
            cover=self.cover_depth(x, y),
            oxide=self.oxide_depth(x, y),
            bedding=self.bedding_normal[0] * bx + self.bedding_normal[1] * by,
            fault=self.fault_normal[0] * (x - self.fault_point[0])
            + self.fault_normal[1] * (y - self.fault_point[1]),
            base_fw=self._layer_base(u),
            base_hw=self._layer_base(u - float(self.strike @ self.slip_vector)),
            stock=((e1[0] * sx + e1[1] * sy) / a) ** 2 + ((e2[0] * sx + e2[1] * sy) / b) ** 2,
            dyke=np.abs(m[0] * (x - q[0]) + m[1] * (y - q[1])) < self.cfg.dyke_half_thickness,
        )

    def lith_codes(self, plan: ColumnPlan, z, *, cover: bool = True) -> np.ndarray:
        """Index into ``LITH`` at elevations ``z`` (broadcast against the plan)."""
        z = np.asarray(z, dtype=np.float64)
        hw = plan.fault + self.fault_normal[2] * (z - self.fault_point[2]) > 0.0
        s = plan.bedding + self.bedding_normal[2] * (z - self.bedding_point[2])
        s = s - hw * float(self.bedding_normal @ self.slip_vector)
        base = np.where(hw, plan.base_hw, plan.base_fw)
        code = np.where(s >= 0.0, _SST, np.where(s >= base, _SLT, _AND)).astype(np.int8)
        vertical = (z - self.stock_centre[2]) / self.cfg.stock_semi_axes[2]
        code[plan.stock + vertical**2 < 1.0] = _DIO
        code[np.broadcast_to(plan.dyke, code.shape)] = _GRD
        depth = plan.ground - z
        if cover:
            code[depth < plan.cover] = _Q4
        code[depth < 0.0] = _AIR
        return code

    def weathering_codes(self, plan: ColumnPlan, z) -> np.ndarray:
        depth = plan.ground - np.asarray(z, dtype=np.float64)
        code = np.where(
            depth < plan.oxide,
            0,
            np.where(depth < plan.oxide + self.cfg.transition_thickness, 1, 2),
        ).astype(np.int8)
        code[depth < 0.0] = 3
        return code

    def codes(self, points, layer: str = "lith") -> np.ndarray:
        """Integer labels (index into ``CATEGORIES[layer]``) at points (n, 3)."""
        p = np.atleast_2d(np.asarray(points, dtype=np.float64))
        if layer == "fault_block":
            return self.hanging_wall(p).astype(np.int8)
        if layer == "rock":
            code = self.lens.inside(p).astype(np.int8)
            code[p[:, 2] > self.topo.z(p[:, 0], p[:, 1])] = 2
            return code
        plan = self.plan(p[:, 0], p[:, 1])
        if layer == "weathering":
            return self.weathering_codes(plan, p[:, 2])
        if layer in ("lith", "bedrock"):
            return self.lith_codes(plan, p[:, 2], cover=layer == "lith")
        raise ValueError(f"unknown layer {layer!r}; one of {tuple(CATEGORIES)}")

    def labels(self, points, layer: str = "lith") -> np.ndarray:
        """Labels (object array) at points (n, 3): ``lith | weathering | fault_block | rock``.

        ``bedrock`` is also accepted: the rock unit under the cover (``lith`` without step 5).
        """
        codes = self.codes(points, layer)
        return np.array(CATEGORIES[layer], dtype=object)[codes]
