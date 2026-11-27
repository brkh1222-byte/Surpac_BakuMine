"""Data model of a bench blast (plan_blast.md, research_blast.md).

SI units throughout: metres, kilograms, seconds, pascals. The only exceptions are the fields
whose names end in ``_ms`` (milliseconds) and ``_mpa`` / ``_gpa`` (rock strength and stiffness
as they are quoted in practice). Coordinates are x east, y north, z up.

A hole is a straight segment from ``collar`` (on the bench surface) to ``toe``. Depths along
the hole are measured from the collar. The explosive column runs from ``charge_top`` to
``charge_bottom`` (depths along the hole); above it is stemming, the part of the hole below
the bench floor is subdrill.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "ANFO",
    "EMULSION",
    "BlastDesign",
    "Explosive",
    "Hole",
    "Monitor",
    "Rock",
]

FORMAT = "bakumine-blast-design"
VERSION = 1


@dataclass(frozen=True)
class Explosive:
    """A bulk explosive as loaded in the hole.

    ``rws`` is the relative weight strength with ANFO = 100; ``vod`` the velocity of
    detonation; ``energy`` the heat of explosion per kilogram.
    """

    name: str
    density: float  # kg/m3 in the hole
    vod: float  # m/s
    rws: float = 100.0
    energy: float = 3.7e6  # J/kg

    def __post_init__(self) -> None:
        for key in ("density", "vod", "rws", "energy"):
            value = getattr(self, key)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"explosive {key} must be > 0, got {value!r}")


#: Typical published values (Persson, Holmberg and Lee 1994; ISEE Blasters' Handbook).
ANFO = Explosive("ANFO", density=850.0, vod=4000.0, rws=100.0, energy=3.7e6)
EMULSION = Explosive("Emulsion", density=1150.0, vod=5500.0, rws=115.0, energy=3.2e6)


@dataclass(frozen=True)
class Rock:
    """Intact-rock and rock-mass properties the blast models need.

    ``rock_factor`` is the Kuz-Ram rock factor *A* (about 0.8 for very soft to 22 for very
    hard massive rock); ``None`` means "derive it from the blastability inputs".
    """

    name: str = "rock"
    density: float = 2700.0  # kg/m3
    ucs_mpa: float = 100.0  # uniaxial compressive strength
    tensile_mpa: float = 10.0  # tensile strength
    youngs_gpa: float = 50.0  # Young's modulus
    poisson: float = 0.25
    p_wave: float | None = None  # m/s; None = from the elastic constants
    rock_factor: float | None = None
    joint_spacing: float = 1.0  # m, mean in-situ block size
    swell: float = 1.3  # broken volume / in-situ volume
    repose_deg: float = 37.0  # angle of repose of the broken rock

    def __post_init__(self) -> None:
        for key in ("density", "ucs_mpa", "tensile_mpa", "youngs_gpa", "joint_spacing"):
            value = getattr(self, key)
            if not (math.isfinite(value) and value > 0):
                raise ValueError(f"rock {key} must be > 0, got {value!r}")
        if not 0.0 <= self.poisson < 0.5:
            raise ValueError("rock poisson must be in [0, 0.5)")
        if not self.swell >= 1.0:
            raise ValueError("rock swell must be >= 1")
        if not 0.0 < self.repose_deg < 90.0:
            raise ValueError("rock repose_deg must be between 0 and 90")


@dataclass
class Hole:
    """One blast hole. See the module docstring for the depth convention."""

    id: str
    collar: tuple[float, float, float]
    toe: tuple[float, float, float]
    diameter: float  # m
    charge_top: float  # m along the hole from the collar (= stemming length)
    charge_bottom: float  # m along the hole from the collar
    explosive: Explosive = ANFO
    row: int = 0  # 0 = the row next to the free face
    index: int = 0  # position along the row
    subdrill: float = 0.0  # m of the hole below the bench floor
    burden: float = math.nan  # m, design burden of this hole
    spacing: float = math.nan  # m, design spacing of this hole
    delay_ms: float = math.nan  # firing time after the first initiation; NaN = not timed

    def __post_init__(self) -> None:
        self.collar = tuple(float(v) for v in self.collar)  # type: ignore[assignment]
        self.toe = tuple(float(v) for v in self.toe)  # type: ignore[assignment]
        if not (math.isfinite(self.diameter) and self.diameter > 0):
            raise ValueError(f"hole {self.id}: diameter must be > 0")
        length = self.length
        if not length > 0:
            raise ValueError(f"hole {self.id}: collar and toe coincide")
        if not (0.0 <= self.charge_top <= self.charge_bottom <= length + 1e-9):
            raise ValueError(
                f"hole {self.id}: need 0 <= charge_top <= charge_bottom <= length "
                f"({self.charge_top}, {self.charge_bottom}, {length})"
            )

    # ---- geometry
    @property
    def length(self) -> float:
        return float(np.linalg.norm(np.subtract(self.toe, self.collar)))

    @property
    def direction(self) -> np.ndarray:
        """Unit vector from collar to toe."""
        return np.subtract(self.toe, self.collar) / self.length

    def point_at(self, depth: float) -> np.ndarray:
        """World coordinates at ``depth`` metres along the hole."""
        return np.asarray(self.collar) + self.direction * float(depth)

    # ---- charge
    @property
    def stemming(self) -> float:
        return self.charge_top

    @property
    def charge_length(self) -> float:
        return self.charge_bottom - self.charge_top

    @property
    def linear_density(self) -> float:
        """Kilograms of explosive per metre of charged hole (fully coupled)."""
        return math.pi / 4.0 * self.diameter**2 * self.explosive.density

    @property
    def charge_mass(self) -> float:
        return self.linear_density * self.charge_length

    @property
    def charge_centre(self) -> np.ndarray:
        return self.point_at(0.5 * (self.charge_top + self.charge_bottom))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["collar"], out["toe"] = list(self.collar), list(self.toe)
        for key in ("burden", "spacing", "delay_ms"):
            if isinstance(out[key], float) and math.isnan(out[key]):
                out[key] = None
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Hole:
        data = dict(data)
        data["explosive"] = Explosive(**data["explosive"])
        for key in ("burden", "spacing", "delay_ms"):
            if data.get(key) is None:
                data[key] = math.nan
        return cls(**data)


@dataclass(frozen=True)
class Monitor:
    """A point where vibration and air overpressure are reported (a house, a wall, a plant)."""

    name: str
    xyz: tuple[float, float, float]
    limit_mm_s: float | None = None
    airblast_limit_dbl: float | None = None


@dataclass
class BlastDesign:
    """A bench blast: the holes and the facts they share.

    ``face_azimuth`` is the direction the free face looks, i.e. the direction the rock is
    thrown, in degrees clockwise from north. ``floor_z`` is the bench floor (design grade);
    ``polygon`` the plan outline of the blasted block.
    """

    name: str
    holes: list[Hole]
    rock: Rock = field(default_factory=Rock)
    bench_height: float = 10.0
    floor_z: float = 0.0
    burden: float = math.nan
    spacing: float = math.nan
    pattern: str = "staggered"  # "staggered" | "square"
    face_azimuth: float = 0.0
    polygon: list[tuple[float, float]] = field(default_factory=list)
    monitors: list[Monitor] = field(default_factory=list)
    notes: dict[str, Any] = field(default_factory=dict)

    # ---- totals
    @property
    def explosive_mass(self) -> float:
        return float(sum(h.charge_mass for h in self.holes))

    @property
    def drilled_length(self) -> float:
        return float(sum(h.length for h in self.holes))

    @property
    def plan_area(self) -> float:
        """Shoelace area of ``polygon`` (0 when none is given)."""
        if len(self.polygon) < 3:
            return 0.0
        p = np.asarray(self.polygon, dtype=np.float64)
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))

    @property
    def volume(self) -> float:
        """In-situ volume to break: polygon area x bench height when a polygon is given,
        else holes x burden x spacing x bench height."""
        area = self.plan_area
        if area > 0:
            return area * self.bench_height
        return len(self.holes) * self.burden * self.spacing * self.bench_height

    @property
    def powder_factor(self) -> float:
        """Kilograms of explosive per cubic metre of in-situ rock."""
        return self.explosive_mass / self.volume

    @property
    def timed(self) -> bool:
        return all(math.isfinite(h.delay_ms) for h in self.holes)

    # ---- I/O
    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "name": self.name,
            "rock": asdict(self.rock),
            "bench_height": self.bench_height,
            "floor_z": self.floor_z,
            "burden": None if math.isnan(self.burden) else self.burden,
            "spacing": None if math.isnan(self.spacing) else self.spacing,
            "pattern": self.pattern,
            "face_azimuth": self.face_azimuth,
            "polygon": [list(p) for p in self.polygon],
            "monitors": [asdict(m) | {"xyz": list(m.xyz)} for m in self.monitors],
            "notes": self.notes,
            "holes": [h.to_dict() for h in self.holes],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> BlastDesign:
        if data.get("format") != FORMAT:
            raise ValueError(f"not a blast design (format {data.get('format')!r})")
        if data.get("version") != VERSION:
            raise ValueError(f"unsupported blast design version {data.get('version')!r}")
        return cls(
            name=data["name"],
            holes=[Hole.from_dict(h) for h in data["holes"]],
            rock=Rock(**data["rock"]),
            bench_height=float(data["bench_height"]),
            floor_z=float(data["floor_z"]),
            burden=math.nan if data.get("burden") is None else float(data["burden"]),
            spacing=math.nan if data.get("spacing") is None else float(data["spacing"]),
            pattern=data.get("pattern", "staggered"),
            face_azimuth=float(data.get("face_azimuth", 0.0)),
            polygon=[tuple(p) for p in data.get("polygon", [])],
            monitors=[
                Monitor(
                    m["name"], tuple(m["xyz"]), m.get("limit_mm_s"), m.get("airblast_limit_dbl")
                )
                for m in data.get("monitors", [])
            ],
            notes=dict(data.get("notes", {})),
        )

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> BlastDesign:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
