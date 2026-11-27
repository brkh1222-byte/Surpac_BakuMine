"""The synthetic layered deposit (geology_contracts.md 13): truth labels, files, self-checks.

The expected values are worked out here from the contract's numbers with plain ``math`` /
numpy formulas, not taken from the generator: the topography, the cover and oxide depths, the
fault plane and the hole paths are all re-stated in this file.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine import cli
from bakumine.synth.layered import LAYERED_PRESETS, LITH, LayeredConfig, LayeredTruth
from bakumine.synth.layered_generate import (
    LayeredDeposit,
    SelfCheckError,
    generate_layered,
    import_layered,
)

X0, Y0 = 500_000.0, 5_300_000.0
SEED = 42
EXPECTED_FILES = {
    "collar.csv", "survey.csv", "assay.csv", "geology.csv", "units.json",
    "topo.obj", "topo_vertices.npy", "topo_faces.npy",
    "ore_solid.obj", "ore_solid_vertices.npy", "ore_solid_faces.npy",
    "truth_fault_F1.obj",
    "truth_cover_base.obj", "truth_cover_base_vertices.npy", "truth_cover_base_faces.npy",
    "truth_oxide_base.obj", "truth_oxide_base_vertices.npy", "truth_oxide_base_faces.npy",
    "truth_transition_base.obj", "truth_transition_base_vertices.npy",
    "truth_transition_base_faces.npy",
    "truth_blocks.parquet", "truth_contacts.csv", "truth.json",
}  # fmt: skip


# ---------------------------------------------------------------------------------------
# the contract's formulas, written out by hand
# ---------------------------------------------------------------------------------------
def rad(degrees: float) -> float:
    return math.radians(degrees)


N_F = np.array([math.sin(rad(65)) * math.sin(rad(70)), math.sin(rad(65)) * math.cos(rad(70)),
                math.cos(rad(65))])  # fmt: skip
P_F = np.array([500_450.0, 5_300_400.0, 1_100.0])
SLIP = 30.0 / math.sin(rad(65))
D = SLIP * np.array([math.cos(rad(65)) * math.sin(rad(70)), math.cos(rad(65)) * math.cos(rad(70)),
                     -math.sin(rad(65))])  # fmt: skip
N_B = np.array([math.sin(rad(20)) * math.sin(rad(340)), math.sin(rad(20)) * math.cos(rad(340)),
                math.cos(rad(20))])  # fmt: skip
P_0 = np.array([500_400.0, 5_300_400.0, 1_110.0])


def topo_hand(x, y):
    dx, dy = np.asarray(x) - X0, np.asarray(y) - Y0
    z = 1200.0 + 0.02 * dx - 0.012 * dy
    for amplitude, wavelength, direction, phase in (
        (14.0, 900.0, 70.0, 0.4),
        (8.0, 420.0, 340.0, 1.3),
        (3.0, 160.0, 30.0, 2.1),
    ):
        along = dx * math.sin(rad(direction)) + dy * math.cos(rad(direction))
        z = z + amplitude * np.sin(2 * np.pi * along / wavelength + phase)
    return z


def cover_hand(x, y):
    dx, dy = np.asarray(x) - X0, np.asarray(y) - Y0
    return 4.0 + 2.0 * np.sin(2 * np.pi * dx / 300.0) * np.sin(2 * np.pi * dy / 250.0)


def oxide_hand(x, y):
    dx, dy = np.asarray(x) - X0, np.asarray(y) - Y0
    return 22.5 + 5.0 * np.sin(2 * np.pi * dx / 500.0 + 0.7) + 2.5 * np.cos(2 * np.pi * dy / 350.0)


def hanging_wall_hand(points) -> np.ndarray:
    return (np.asarray(points) - P_F) @ N_F > 0.0


def fold(code: str) -> str:
    return " ".join(str(code).split()).upper()


# ---------------------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Layered:
    deposit: LayeredDeposit
    path: Path

    @property
    def truth(self) -> LayeredTruth:
        return self.deposit.truth

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path / f"{name}.csv", dtype={"hole_id": str})

    def json(self, name: str) -> dict:
        return json.loads((self.path / f"{name}.json").read_text())

    def holes(self) -> pd.DataFrame:
        """Collar + direction of every hole, from the two input files only."""
        collar = self.csv("collar").set_index("hole_id")
        survey = self.csv("survey").groupby("hole_id").first()
        dip, az = np.radians(survey["dip"]), np.radians(survey["azimuth"])
        collar["ux"] = np.cos(dip) * np.sin(az)
        collar["uy"] = np.cos(dip) * np.cos(az)
        collar["uz"] = np.sin(dip)
        collar["h"] = [int(name[1:]) - 1 for name in collar.index]
        return collar

    def xyz(self, hole_id, depth) -> np.ndarray:
        """Points on straight holes: collar + depth * direction."""
        holes = self.holes().loc[np.asarray(hole_id)]
        start = holes[["x", "y", "z"]].to_numpy()
        return (
            start + np.asarray(depth, dtype=float)[:, None] * holes[["ux", "uy", "uz"]].to_numpy()
        )

    def unit_of(self) -> dict[str, tuple[str, str]]:
        """Folded spelling -> (unit code, kind), from units.json alone."""
        lookup = {}
        for unit in self.json("units")["units"]:
            for alias in (unit["code"], *unit.get("aliases", ())):
                lookup[fold(alias)] = (unit["code"], unit["kind"])
        return lookup


@pytest.fixture(scope="module")
def small(tmp_path_factory) -> Layered:
    deposit = generate_layered(LAYERED_PRESETS["layered-small"], seed=SEED)
    path = tmp_path_factory.mktemp("layered_small")
    deposit.write(path)
    return Layered(deposit, path)


@pytest.fixture(scope="module")
def truth() -> LayeredTruth:
    return LayeredTruth(LAYERED_PRESETS["layered-small"])


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    out = tmp_path_factory.mktemp("layered_cli")
    code = cli.main(["synth", "--preset", "layered-small", "--out", str(out), "--seed", str(SEED)])
    return code, out


def at(dx: float, dy: float, z: float) -> list[float]:
    return [X0 + dx, Y0 + dy, z]


def below(dx: float, dy: float, depth: float) -> list[float]:
    """A point ``depth`` metres under the topography."""
    return [X0 + dx, Y0 + dy, float(topo_hand(X0 + dx, Y0 + dy)) - depth]


# ---------------------------------------------------------------------------------------
# truth labels at hand-computed points
# ---------------------------------------------------------------------------------------
# Bedding through P0 = (400, 400, 1110): directly above and below P0, s = cos20 * dz.
# SLT base in the footwall at (400, 400): u = 0, s_BC = -40, z = 1110 - 40 / cos20 = 1067.43.
# Hanging wall at (550, 400): restored q = p - D = (536.854, 395.215, z + 30);
#   n_b . (q - P0) = 0 at z = 1098.67 (30 m below the footwall contact, 1128.67);
#   u = 126.96, s_BC = -44.64, so the SLT base is at z = 1051.17.
# Stock: centre (300, 480, 1060), semi-axes 110 (along 045), 80 (along 135), 150 (vertical).
# Dyke: |m . (p - Q)| < 4 with m = (sin135, cos135, 0), Q = (480, 300).
S45 = math.sin(rad(45))
LITH_POINTS = [
    (at(400, 400, 1112), "SST"),
    (at(400, 400, 1108), "SLT"),
    (at(400, 400, 1069), "SLT"),
    (at(400, 400, 1066), "AND"),
    (at(550, 400, 1100), "SST"),  # SLT without the fault: n_b . (p - P0) = -26.9
    (at(550, 400, 1097), "SLT"),
    (at(550, 400, 1053), "SLT"),
    (at(550, 400, 1049), "AND"),
    (at(300, 480, 1060), "DIO"),
    (at(300 + 109 * S45, 480 + 109 * S45, 1060), "DIO"),
    (at(300 + 79 * S45, 480 - 79 * S45, 1060), "DIO"),
    (at(300 - 79 * S45, 480 + 79 * S45, 1060), "DIO"),
    (at(300, 480, 1060 - 149), "DIO"),
    (at(480, 300, 1100), "GRD"),
    (at(480 + 3.9 * S45, 300 - 3.9 * S45, 1100), "GRD"),
    (at(480 - 3.9 * S45, 300 + 3.9 * S45, 1100), "GRD"),
    (at(480 + 100 * S45, 300 + 100 * S45, 1100), "GRD"),  # along strike, in the hanging wall
    (at(480 - 100 * S45, 300 - 100 * S45, 1100), "GRD"),  # along strike, in the footwall
    (below(75, 62.5, 5.9), "Q4"),  # c = 4 + 2 * 1 * 1 = 6
    (below(225, 62.5, 1.9), "Q4"),  # c = 4 + 2 * (-1) * 1 = 2
    (below(400, 400, 0.001), "Q4"),
    (below(400, 400, -0.5), "AIR"),
    (below(550, 400, -40.0), "AIR"),
]
NOT_LITH_POINTS = [
    (at(300 + 111 * S45, 480 + 111 * S45, 1060), "DIO"),
    (at(300 + 81 * S45, 480 - 81 * S45, 1060), "DIO"),
    (at(300 - 81 * S45, 480 + 81 * S45, 1060), "DIO"),
    (at(300, 480, 1060 - 151), "DIO"),
    (at(480 + 4.1 * S45, 300 - 4.1 * S45, 1100), "GRD"),
    (at(480 - 4.1 * S45, 300 + 4.1 * S45, 1100), "GRD"),
    (below(75, 62.5, 6.1), "Q4"),
    (below(225, 62.5, 2.1), "Q4"),
    (below(75, 62.5, 6.1), "AIR"),
]


@pytest.mark.parametrize(("point", "expected"), LITH_POINTS)
def test_lith_label_at_hand_computed_point(truth, point, expected):
    assert truth.labels([point], "lith").tolist() == [expected]


@pytest.mark.parametrize(("point", "excluded"), NOT_LITH_POINTS)
def test_lith_label_just_outside_a_body(truth, point, excluded):
    assert truth.labels([point], "lith")[0] != excluded


def test_labels_do_not_depend_on_the_preset(truth):
    full = LayeredTruth(LAYERED_PRESETS["layered"])
    points = np.array([p for p, _ in LITH_POINTS])
    for layer in ("lith", "weathering", "fault_block", "rock"):
        assert truth.labels(points, layer).tolist() == full.labels(points, layer).tolist()


def test_fault_block_labels(truth):
    # n_f . (p - Pf) at the level of Pf is 0.8517 * (x - 450) + 0.3100 * (y - 400).
    points = [at(400, 400, 1100), at(550, 400, 1100), at(449, 400, 1100), at(451, 400, 1100),
              at(450, 400, 1101), at(450, 400, 1099), at(550, 400, 1300)]  # fmt: skip
    expected = ["FB1", "FB2", "FB1", "FB2", "FB2", "FB1", "FB2"]
    assert truth.labels(points, "fault_block").tolist() == expected
    assert truth.hanging_wall(points).tolist() == [name == "FB2" for name in expected]
    cloud = np.random.default_rng(0).uniform([X0, Y0, 900], [X0 + 800, Y0 + 800, 1250], (5000, 3))
    assert np.array_equal(truth.hanging_wall(cloud), hanging_wall_hand(cloud))


def test_weathering_labels(truth):
    # d_ox = 22.5 + 5 sin(2 pi X / 500 + 0.7) + 2.5 cos(2 pi Y / 350)
    x_max = (math.pi / 2 - 0.7) * 500 / (2 * math.pi)  # the sine is +1: 69.30
    x_min = (3 * math.pi / 2 - 0.7) * 500 / (2 * math.pi)  # the sine is -1: 319.30
    cases = [
        (below(x_max, 0, 29.9), "OXIDE"),  # d_ox = 30
        (below(x_max, 0, 30.1), "TRANSITION"),
        (below(x_max, 0, 34.9), "TRANSITION"),
        (below(x_max, 0, 35.1), "FRESH"),
        (below(x_min, 175, 14.9), "OXIDE"),  # d_ox = 15
        (below(x_min, 175, 15.1), "TRANSITION"),
        (below(x_min, 175, 20.1), "FRESH"),
        (below(x_max, 0, 0.001), "OXIDE"),
        (below(x_max, 0, -0.1), "AIR"),
    ]
    points = [p for p, _ in cases]
    assert truth.labels(points, "weathering").tolist() == [label for _, label in cases]


def test_rock_labels(truth):
    centre = [500_450.0, 5_300_400.0, 1_095.0]
    points = [centre, at(450, 400, 1095 + 30), at(450, 400, 1095 - 30), below(450, 400, -1.0)]
    assert truth.labels(points, "rock").tolist() == ["ORE", "WASTE", "WASTE", "AIR"]
    with pytest.raises(ValueError, match="unknown layer"):
        truth.labels([centre], "nonsense")


def test_cover_and_oxide_depths_are_within_their_ranges(truth):
    x, y = (
        a.ravel() for a in np.meshgrid(X0 + np.arange(0, 800.5, 2.5), Y0 + np.arange(0, 800.5, 2.5))
    )
    cover, oxide = truth.cover_depth(x, y), truth.oxide_depth(x, y)
    np.testing.assert_allclose(cover, cover_hand(x, y), rtol=0, atol=1e-9)
    np.testing.assert_allclose(oxide, oxide_hand(x, y), rtol=0, atol=1e-9)
    np.testing.assert_allclose(truth.transition_depth(x, y), oxide + 5.0, rtol=0, atol=1e-12)
    np.testing.assert_allclose(truth.topo_z(x, y), topo_hand(x, y), rtol=0, atol=1e-6)
    assert 2.0 <= cover.min() < 2.01 and 5.99 < cover.max() <= 6.0
    assert 15.0 <= oxide.min() < 15.01 and 29.99 < oxide.max() <= 30.0


# ---------------------------------------------------------------------------------------
# the fault
# ---------------------------------------------------------------------------------------
def test_slip_vector(truth):
    np.testing.assert_allclose(truth.fault_normal, N_F, atol=1e-12)
    np.testing.assert_allclose(truth.fault_point, P_F, atol=0)
    np.testing.assert_allclose(truth.slip_vector, D, atol=1e-12)
    assert truth.slip_vector[2] == pytest.approx(-30.0, abs=1e-12)  # throw
    assert np.linalg.norm(truth.slip_vector) == pytest.approx(33.101, abs=5e-4)
    assert np.hypot(*truth.slip_vector[:2]) == pytest.approx(13.989, abs=5e-4)  # heave
    assert truth.slip_vector @ truth.fault_normal == pytest.approx(0.0, abs=1e-12)  # in the plane
    azimuth = math.degrees(math.atan2(truth.slip_vector[0], truth.slip_vector[1]))
    assert azimuth == pytest.approx(70.0, abs=1e-9)


def test_sst_slt_contact_is_separated_30_m_vertically(truth):
    x, y = (
        a.ravel() for a in np.meshgrid(X0 + np.arange(0, 801, 40.0), Y0 + np.arange(0, 801, 40.0))
    )
    footwall = truth.contact_z(x, y, "SST/SLT", "FB1")
    hanging = truth.contact_z(x, y, "SST/SLT", "FB2")
    np.testing.assert_allclose(footwall - hanging, 30.0, rtol=0, atol=1e-8)
    # the footwall contact is the plane through P0 with normal n_b
    by_hand = P_0[2] - (N_B[0] * (x - P_0[0]) + N_B[1] * (y - P_0[1])) / N_B[2]
    np.testing.assert_allclose(footwall, by_hand, rtol=0, atol=1e-8)
    assert truth.contact_z(X0 + 400, Y0 + 400) == pytest.approx(1110.0, abs=1e-9)
    assert truth.contact_z(X0 + 400, Y0 + 400, "SLT/AND") == pytest.approx(
        1110.0 - 40.0 / math.cos(rad(20)), abs=1e-9
    )
    assert truth.contact_z(X0 + 550, Y0 + 400, "SST/SLT", "FB2") == pytest.approx(
        1098.673, abs=1e-3
    )
    assert truth.contact_z(X0 + 550, Y0 + 400, "SLT/AND", "FB2") == pytest.approx(1051.17, abs=1e-2)
    # the SLT base is also offset by the slip: 30 m lower at the point moved by the heave
    base_fw = truth.contact_z(x, y, "SLT/AND", "FB1")
    base_hw = truth.contact_z(x + D[0], y + D[1], "SLT/AND", "FB2")
    np.testing.assert_allclose(base_fw - base_hw, 30.0, rtol=0, atol=1e-8)
    with pytest.raises(ValueError):
        truth.contact_z(x, y, "AND/DIO")
    with pytest.raises(ValueError):
        truth.contact_z(x, y, "SST/SLT", "FB3")


@pytest.mark.parametrize(
    ("contact", "upper", "lower"), [("SST/SLT", "SST", "SLT"), ("SLT/AND", "SLT", "AND")]
)
@pytest.mark.parametrize("block", ["FB1", "FB2"])
def test_labels_change_at_the_analytic_contact(truth, contact, upper, lower, block):
    """Just above / below contact_z the labels are the two units, on the block's own side."""
    x, y = (
        a.ravel()
        for a in np.meshgrid(X0 + np.arange(205, 600, 10.0), Y0 + np.arange(205, 600, 10.0))
    )
    z = truth.contact_z(x, y, contact, block)
    above, under = np.column_stack([x, y, z + 0.25]), np.column_stack([x, y, z - 0.25])
    side = hanging_wall_hand(above) if block == "FB2" else ~hanging_wall_hand(above)
    same_side = side & (hanging_wall_hand(above) == hanging_wall_hand(under))
    top, bottom = truth.labels(above, "lith"), truth.labels(under, "lith")
    layered = np.isin(top, ["SST", "SLT", "AND"]) & np.isin(bottom, ["SST", "SLT", "AND"])
    use = same_side & layered
    assert use.sum() > 100
    assert set(top[use]) == {upper} and set(bottom[use]) == {lower}


def test_vertical_separation_from_the_labels_alone(truth):
    """Bisect the SST/SLT change on vertical lines either side of the fault trace."""

    def contact(x: float, y: float, lo: float, hi: float) -> float:
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if truth.labels([[x, y, mid]], "lith")[0] == "SST":
                hi = mid
            else:
                lo = mid
        return 0.5 * (lo + hi)

    # two columns on a line along the bedding strike (070): the footwall contact has the same
    # elevation at both, so the difference is the fault's vertical separation.
    strike = np.array([math.sin(rad(70)), math.cos(rad(70))])
    a = np.array([X0 + 430.0, Y0 + 380.0])
    b = a + 80.0 * strike
    za = contact(*a, 1090.0, 1130.0)
    zb = contact(*b, 1050.0, 1100.0)
    assert not hanging_wall_hand([*a, za]) and hanging_wall_hand([*b, zb])
    assert za - zb == pytest.approx(30.0, abs=1e-6)


def test_restored_is_the_inverse_displacement(truth):
    rng = np.random.default_rng(1)
    points = rng.uniform([X0 + 200, Y0 + 200, 950], [X0 + 600, Y0 + 600, 1250], (4000, 3))
    hw = hanging_wall_hand(points)
    assert hw.sum() > 500 and (~hw).sum() > 500
    restored = truth.restored(points)
    np.testing.assert_array_equal(restored[~hw], points[~hw])
    np.testing.assert_allclose(restored[hw], points[hw] - D, rtol=0, atol=1e-9)
    # displacing a pre-fault position q by D lands in the hanging wall when q was on that
    # side, and restoring brings it back
    moved = points[hw] + D
    assert hanging_wall_hand(moved).all()  # D lies in the plane, the side does not change
    np.testing.assert_allclose(truth.restored(moved), points[hw], rtol=0, atol=1e-9)
    # the layered units at p are the unfaulted layers at restored(p)
    q = restored - P_0
    s = q @ N_B
    u = q[:, 0] * math.sin(rad(70)) + q[:, 1] * math.cos(rad(70))
    base = -(40.0 + 10.0 * np.sin(2 * np.pi * u / 300.0))
    by_hand = np.where(s >= 0, "SST", np.where(s >= base, "SLT", "AND"))
    labels = truth.labels(points, "bedrock")
    layered = np.isin(labels, ["SST", "SLT", "AND"])
    assert layered.sum() > 2000
    assert (labels[layered] == by_hand[layered]).all()


# ---------------------------------------------------------------------------------------
# drilling and logging
# ---------------------------------------------------------------------------------------
def test_holes_follow_the_contract(small):
    collar, survey = small.csv("collar"), small.csv("survey")
    assert len(collar) == 64
    assert collar["hole_id"].tolist() == [f"L{k + 1:03d}" for k in range(64)]
    h = np.arange(64)
    np.testing.assert_allclose(collar["x"], X0 + 200 + 25 + 50 * (h % 8), atol=1e-9)
    np.testing.assert_allclose(collar["y"], Y0 + 200 + 25 + 50 * (h // 8), atol=1e-9)
    np.testing.assert_allclose(collar["z"], topo_hand(collar["x"], collar["y"]), atol=1e-6)
    vertical = h % 10 == 4
    np.testing.assert_allclose(collar["max_depth"], np.where(vertical, 144.0, 180.0))
    assert len(survey) == 128
    assert survey["depth"].tolist() == [d for length in collar["max_depth"] for d in (0.0, length)]
    np.testing.assert_allclose(survey["dip"], np.repeat(np.where(vertical, -90.0, -60.0), 2))
    assert (survey["azimuth"][~np.repeat(vertical, 2)] == 160.0).all()


def test_units_json(small):
    units = small.json("units")
    assert units["format"] == "bakumine.units" and units["version"] == 1
    assert units["source"] == {"table": "geology", "column": "lith"}
    assert units["default_density"] == 2.70 and units["default_density_source"] == "assumed"
    by_code = {u["code"]: u for u in units["units"]}
    density = {"SST": 2.55, "SLT": 2.65, "AND": 2.75, "DIO": 2.80, "GRD": 2.70, "Q4": 1.90}
    for code, value in density.items():
        assert by_code[code]["density"] == value
        assert by_code[code]["density_source"] == "measured"
    kinds = {code: unit["kind"] for code, unit in by_code.items()}
    assert kinds == {
        "Q4": "cover", "OX": "weathering", "SST": "lithology", "SLT": "lithology",
        "AND": "lithology", "DIO": "intrusion", "GRD": "intrusion", "FLT": "fault-zone",
        "CL": "missing", "BX": "other",
    }  # fmt: skip
    zones = units["weathering_zones"]
    assert [z["code"] for z in zones] == ["OXIDE", "TRANSITION", "FRESH"]
    assert zones[0]["density"] == 2.30 and zones[0]["density_source"] == "measured"
    # every logged spelling maps to exactly one unit after folding
    lookup = small.unit_of()
    logged = set(small.csv("geology")["lith"])
    assert {"sst ", "Sst", "HB DIO", "Dio", "Slt", "SOL", "SOIL", "OZ", "FB", "FLT", "CL"} <= logged
    assert {fold(code) for code in logged} <= set(lookup)


def test_geology_intervals_tile_every_hole(small):
    geology, collar = small.csv("geology"), small.csv("collar").set_index("hole_id")
    assert list(geology.columns) == ["hole_id", "depth_from", "depth_to", "lith"]
    assert set(geology["hole_id"]) == set(collar.index)
    for hole_id, rows in geology.groupby("hole_id"):
        assert rows["depth_from"].iloc[0] == 0.0
        assert rows["depth_to"].iloc[-1] == collar.loc[hole_id, "max_depth"]
        assert (rows["depth_from"].to_numpy()[1:] == rows["depth_to"].to_numpy()[:-1]).all()
        assert (rows["depth_to"] > rows["depth_from"]).all()
        codes = rows["lith"].tolist()
        assert all(a != b for a, b in zip(codes[:-1], codes[1:], strict=True))  # merged


def test_logged_rock_agrees_with_the_truth_labels(small):
    """Inside every interval logged as a rock unit the truth has that unit.

    The override codes (soil, OZ, fault zone, core loss) are excluded. A hole that does not log
    soil logs the rock under the cover instead, so there the truth may be Q4 inside the cover.
    """
    geology, lookup, holes = small.csv("geology"), small.unit_of(), small.holes()
    units = geology["lith"].map(lambda code: lookup[fold(code)][0])
    kinds = geology["lith"].map(lambda code: lookup[fold(code)][1])
    rock = kinds.isin(["lithology", "intrusion"]) & (
        geology["depth_to"] - geology["depth_from"] > 0.05
    )
    rows = geology[rock]
    assert len(rows) > 300 and set(units[rock]) == {"SST", "SLT", "AND", "DIO", "GRD"}
    checked = in_cover = 0
    for fraction in (0.0, 0.5, 1.0):
        # 0.011 m inside the ends: the logged depths are rounded to 0.01 m
        lo, hi = rows["depth_from"] + 0.011, rows["depth_to"] - 0.011
        depth = lo + fraction * (hi - lo)
        points = small.xyz(rows["hole_id"], depth)
        labels = small.truth.labels(points, "lith")
        expected = units[rock].to_numpy()
        differs = labels != expected
        soil_logged = holes.loc[rows["hole_id"], "h"].to_numpy() % 5 <= 2
        depth_below = topo_hand(points[:, 0], points[:, 1]) - points[:, 2]
        covered = depth_below < cover_hand(points[:, 0], points[:, 1])
        assert (labels[differs] == "Q4").all()
        assert (covered[differs] & ~soil_logged[differs]).all()
        checked += len(labels)
        in_cover += int(differs.sum())
    assert checked > 900 and in_cover > 0


def test_logged_soil_and_oxide_bases_match_the_true_depths(small):
    geology, lookup, holes = small.csv("geology"), small.unit_of(), small.holes()
    unit = geology["lith"].map(lambda code: lookup[fold(code)][0])
    same_hole = geology["hole_id"] == geology["hole_id"].shift(-1)
    next_kind = (
        geology["lith"].shift(-1).map(lambda c: lookup[fold(c)][1] if isinstance(c, str) else "")
    )
    under_rock = same_hole & next_kind.isin(["lithology", "intrusion"])
    h = holes.loc[geology["hole_id"], "h"].to_numpy()
    for code, depth_of, limits, allowed in (
        ("Q4", cover_hand, (2.0, 6.0), {0, 1, 2}),
        ("OX", oxide_hand, (15.0, 30.0), {0, 3}),
    ):
        logged = (unit == code).to_numpy()
        assert set(h[logged] % 5) == allowed  # logged in these holes only
        rows = geology[logged & under_rock.to_numpy()]
        assert len(rows) >= 15
        base = small.xyz(rows["hole_id"], rows["depth_to"])
        depth_below = topo_hand(base[:, 0], base[:, 1]) - base[:, 2]
        np.testing.assert_allclose(depth_below, depth_of(base[:, 0], base[:, 1]), rtol=0, atol=0.02)
        assert depth_below.min() >= limits[0] - 0.02 and depth_below.max() <= limits[1] + 0.02
    # soil starts at the collar; OZ starts at the soil base or at the collar
    first = geology.groupby("hole_id").first()
    first_unit = first["lith"].map(lambda code: lookup[fold(code)][0])
    first_h = holes.loc[first.index, "h"] % 5
    assert set(first_unit[first_h.isin([0, 1, 2])]) <= {"Q4", "CL", "FLT"}
    assert set(first_unit[first_h == 3]) <= {"OX", "CL", "FLT"}


def test_fault_zone_and_core_loss_intervals(small):
    geology, lookup, holes = small.csv("geology"), small.unit_of(), small.holes()
    unit = geology["lith"].map(lambda code: lookup[fold(code)][0])
    # analytic crossing depth of each straight hole with the fault plane
    start = holes[["x", "y", "z"]].to_numpy()
    direction = holes[["ux", "uy", "uz"]].to_numpy()
    depth = -((start - P_F) @ N_F) / (direction @ N_F)
    crossing = (depth > 0) & (depth < holes["max_depth"].to_numpy())
    assert crossing.sum() >= 6
    zones = geology[unit == "FLT"]
    assert set(zones["hole_id"]) <= set(holes.index[crossing])
    assert not (holes.loc[zones["hole_id"], "h"] % 10 == 9).any()
    expected = holes.index[crossing & (holes["h"].to_numpy() % 10 != 9)]
    assert len(expected) >= 5
    cross_depth = pd.Series(depth, index=holes.index)
    for hole_id in expected:
        rows = zones[zones["hole_id"] == hole_id]
        d = cross_depth[hole_id]
        assert len(rows) >= 1  # core loss may split or shorten it, never remove all 2 m
        assert rows["depth_from"].min() >= d - 1.0 - 0.006
        assert rows["depth_to"].max() <= d + 1.0 + 0.006
    for row in zones.itertuples():
        spelling = ["FLT", "FB"][holes.loc[row.hole_id, "h"] % 2]
        assert row.lith == spelling
    loss = geology[unit == "CL"]
    metres = holes["max_depth"].sum()
    assert 0.6 * 0.02 * metres < len(loss) < 1.2 * 0.02 * metres  # Poisson rate, merged overlaps
    assert (loss["depth_to"] - loss["depth_from"]).median() == pytest.approx(1.25, abs=0.2)


def test_truth_contacts_are_label_changes(small):
    contacts = small.csv("truth_contacts")
    assert list(contacts.columns) == ["hole_id", "layer", "depth", "x", "y", "z", "above", "below"]
    assert set(contacts["layer"]) == {"lith", "weathering", "fault_block", "rock"}
    points = small.xyz(contacts["hole_id"], contacts["depth"])
    np.testing.assert_allclose(points, contacts[["x", "y", "z"]].to_numpy(), rtol=0, atol=2e-6)
    for layer, rows in contacts.groupby("layer"):
        assert (rows["above"] != rows["below"]).all()
        before = small.xyz(rows["hole_id"], rows["depth"] - 1e-4)
        after = small.xyz(rows["hole_id"], rows["depth"] + 1e-4)
        assert small.truth.labels(before, layer).tolist() == rows["above"].tolist()
        assert small.truth.labels(after, layer).tolist() == rows["below"].tolist()
    # fault crossings: the analytic intersection of each hole with the plane
    holes = small.holes()
    fault = contacts[contacts["layer"] == "fault_block"].set_index("hole_id")
    start = holes[["x", "y", "z"]].to_numpy()
    direction = holes[["ux", "uy", "uz"]].to_numpy()
    depth = pd.Series(-((start - P_F) @ N_F) / (direction @ N_F), index=holes.index)
    crossing = (depth > 0) & (depth < holes["max_depth"])
    assert sorted(fault.index) == sorted(holes.index[crossing])
    np.testing.assert_allclose(fault["depth"], depth[fault.index], rtol=0, atol=2e-6)
    # every hole starts in the cover and leaves the oxide zone once
    lith = contacts[contacts["layer"] == "lith"]
    assert (lith.groupby("hole_id").first()["above"] == "Q4").all()
    weathering = contacts[contacts["layer"] == "weathering"]
    assert (
        weathering.groupby("hole_id")["above"]
        .apply(list)
        .map(lambda v: v == ["OXIDE", "TRANSITION"])
        .all()
    )


def test_logged_contacts_are_the_true_contacts_rounded(small):
    """A logged boundary between two rock units is a true lith contact rounded to 0.01 m."""
    geology, lookup = small.csv("geology"), small.unit_of()
    kind = geology["lith"].map(lambda code: lookup[fold(code)][1])
    unit = geology["lith"].map(lambda code: lookup[fold(code)][0])
    rock = kind.isin(["lithology", "intrusion"])
    pair = (
        rock
        & rock.shift(-1, fill_value=False)
        & (geology["hole_id"] == geology["hole_id"].shift(-1))
    )
    logged = pd.DataFrame(
        {
            "hole_id": geology["hole_id"][pair],
            "depth": geology["depth_to"][pair],
            "above": unit[pair],
            "below": unit.shift(-1)[pair],
        }
    )
    truth_rows = small.csv("truth_contacts")
    truth_rows = truth_rows[truth_rows["layer"] == "lith"].assign(
        depth=lambda t: t["depth"].round(2)
    )
    merged = logged.merge(
        truth_rows, on=["hole_id", "depth", "above", "below"], how="left", indicator=True
    )
    assert len(logged) > 80
    assert (merged["_merge"] == "both").all()


def test_assays(small):
    assay, holes = small.csv("assay"), small.holes()
    assert list(assay.columns) == ["hole_id", "depth_from", "depth_to", "au", "s_ppm", "wf"]
    assert len(assay) == int(holes["max_depth"].sum())  # 1 m samples, whole-metre holes
    for hole_id, rows in assay.groupby("hole_id"):
        n = len(rows)
        np.testing.assert_array_equal(rows["depth_from"], np.arange(n, dtype=float))
        np.testing.assert_array_equal(rows["depth_to"], np.arange(1, n + 1, dtype=float))
        assert n == holes.loc[hole_id, "max_depth"]
    mid = small.xyz(assay["hole_id"], 0.5 * (assay["depth_from"] + assay["depth_to"]))
    ore = small.truth.lens.inside(mid)
    block = np.where(hanging_wall_hand(mid), "FB2", "FB1")
    expected = np.where(ore, np.char.add(block.astype(str), "-ORE"), "WST")
    assert assay["wf"].tolist() == expected.tolist()
    assert {"FB1-ORE", "FB2-ORE"} <= set(assay["wf"])  # the lens straddles F1
    assert assay["au"].notna().all() and (assay["au"] >= 0.001).all()
    assert np.median(assay["au"][~ore]) == pytest.approx(0.02, rel=0.03)
    assert np.std(np.log(assay["au"][~ore])) == pytest.approx(0.5, rel=0.05)
    assert ore.sum() > 60
    assert np.median(assay["au"][ore]) == pytest.approx(1.5, rel=0.25)
    depth_below = topo_hand(mid[:, 0], mid[:, 1]) - mid[:, 2]
    oxide = oxide_hand(mid[:, 0], mid[:, 1])
    base = np.where(depth_below < oxide, 300.0, np.where(depth_below < oxide + 5.0, 1600.0, 6000.0))
    assert set(np.unique(base)) == {300.0, 1600.0, 6000.0}
    assert (assay["s_ppm"] == assay["s_ppm"].round()).all()
    ratio = np.log(assay["s_ppm"] / base)
    assert np.abs(ratio).max() < 0.6  # no sample takes another zone's level (ln(1600/300) = 1.67)
    assert np.std(ratio) == pytest.approx(0.1, rel=0.05)
    assert np.mean(ratio) == pytest.approx(0.0, abs=0.005)


def test_database_import_and_traces(small):
    db = import_layered(small.path)
    assert len(db.collar) == 64 and set(db.intervals) == {"assay", "geology"}
    assert len(db.intervals["geology"]) == len(small.csv("geology"))
    assert len(db.intervals["assay"]) == len(small.csv("assay"))
    assert {"lith"} <= set(db.intervals["geology"].columns)
    assert {"au", "s_ppm", "wf"} <= set(db.intervals["assay"].columns)
    traces = db.desurvey()
    assert len(traces) == 64 and not traces.warnings
    for hole_id in ("L001", "L005", "L064"):  # L005 is vertical (h = 4)
        depths = np.array([0.0, 37.5, 144.0])
        xyz, _ = traces.xyz_at(hole_id, depths)
        np.testing.assert_allclose(xyz, small.xyz([hole_id] * 3, depths), rtol=0, atol=1e-6)


# ---------------------------------------------------------------------------------------
# truth surfaces, blocks, volumes, self-checks
# ---------------------------------------------------------------------------------------
def read_obj(path: Path) -> tuple[np.ndarray, np.ndarray]:
    lines = path.read_text().splitlines()
    vertices = np.array([line.split()[1:] for line in lines if line.startswith("v ")], dtype=float)
    faces = np.array([line.split()[1:] for line in lines if line.startswith("f ")], dtype=int) - 1
    return vertices, faces


def up_component(vertices: np.ndarray, faces: np.ndarray, normal=(0.0, 0.0, 1.0)) -> np.ndarray:
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    return np.cross(b - a, c - a) @ np.asarray(normal)


@pytest.mark.parametrize(
    ("name", "depth_of"),
    [
        ("truth_cover_base", cover_hand),
        ("truth_oxide_base", oxide_hand),
        ("truth_transition_base", lambda x, y: oxide_hand(x, y) + 5.0),
    ],
)
def test_truth_surfaces(small, name, depth_of):
    vertices, faces = (
        np.load(small.path / f"{name}_vertices.npy"),
        np.load(small.path / f"{name}_faces.npy"),
    )
    assert len(vertices) == 41 * 41 and len(faces) == 2 * 40 * 40  # 10 m grid over 400 x 400 m
    grid = {
        (x, y) for x in X0 + 200 + 10.0 * np.arange(41) for y in Y0 + 200 + 10.0 * np.arange(41)
    }
    assert {(x, y) for x, y in vertices[:, :2].tolist()} == grid
    expected = topo_hand(vertices[:, 0], vertices[:, 1]) - depth_of(vertices[:, 0], vertices[:, 1])
    np.testing.assert_allclose(vertices[:, 2], expected, rtol=0, atol=2e-6)
    assert (up_component(vertices, faces) > 0).all()
    obj_vertices, obj_faces = read_obj(small.path / f"{name}.obj")
    np.testing.assert_allclose(obj_vertices, vertices, rtol=0, atol=1e-6)
    assert np.array_equal(obj_faces, faces)


def test_topography_and_ore_solid_files(small):
    vertices, faces = (
        np.load(small.path / "topo_vertices.npy"),
        np.load(small.path / "topo_faces.npy"),
    )
    np.testing.assert_allclose(vertices[:, 2], topo_hand(vertices[:, 0], vertices[:, 1]), atol=2e-6)
    assert (up_component(vertices, faces) > 0).all()
    collar = small.csv("collar")
    assert {(x, y) for x, y in collar[["x", "y"]].to_numpy().tolist()} <= {
        (x, y) for x, y in vertices[:, :2].tolist()
    }
    # the DTM covers the box and the whole of every hole path
    ends = small.xyz(collar["hole_id"], collar["max_depth"])
    assert vertices[:, 0].min() <= min(X0 + 200, ends[:, 0].min())
    assert vertices[:, 1].min() <= min(Y0 + 200, ends[:, 1].min())
    assert vertices[:, 0].max() >= max(X0 + 600, ends[:, 0].max())
    assert vertices[:, 1].max() >= Y0 + 600
    ore = np.load(small.path / "ore_solid_vertices.npy")
    np.testing.assert_allclose(small.truth.lens.implicit(ore), 0.0, atol=1e-9)
    assert (topo_hand(ore[:, 0], ore[:, 1]) - ore[:, 2]).min() >= 10.0  # self-check


def test_fault_mesh_is_the_plane_clipped_to_the_box(small):
    vertices, faces = read_obj(small.path / "truth_fault_F1.obj")
    np.testing.assert_allclose((vertices - P_F) @ N_F, 0.0, atol=5e-6)
    lo, hi = np.array([X0 + 200, Y0 + 200, 940.0]), np.array([X0 + 600, Y0 + 600, 1250.0])
    assert (vertices >= lo - 1e-6).all() and (vertices <= hi + 1e-6).all()
    on_boundary = (np.abs(vertices - lo) < 1e-6) | (np.abs(vertices - hi) < 1e-6)
    assert on_boundary.any(axis=1).all()  # every corner of the polygon is on a box face
    assert (up_component(vertices, faces, N_F) > 0).all()
    # area against the plane's trace through the box, integrated numerically in plan:
    # area = (plan area where the plane is inside the z range) / n_z
    x, y = (
        a.ravel()
        for a in np.meshgrid(
            X0 + 200 + np.arange(0.25, 400, 0.5), Y0 + 200 + np.arange(0.25, 400, 0.5)
        )
    )
    z = P_F[2] - (N_F[0] * (x - P_F[0]) + N_F[1] * (y - P_F[1])) / N_F[2]
    plan_area = 0.25 * np.count_nonzero((z > 940.0) & (z < 1250.0))
    a, b, c = (vertices[faces[:, k]] for k in range(3))
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1).sum()
    assert area == pytest.approx(plan_area / N_F[2], rel=2e-3)


def test_truth_blocks(small):
    blocks = pd.read_parquet(small.path / "truth_blocks.parquet")
    nx, ny, nz = 40, 40, 62  # 400 x 400 x 310 m in 10 x 10 x 5 m blocks
    assert len(blocks) == nx * ny * nz
    fractions = [f"frac_{code}" for code in LITH]
    assert list(blocks.columns) == [
        "x",
        "y",
        "z",
        "lith",
        "weathering",
        "fault_block",
        "rock",
        *fractions,
    ]
    row = np.arange(len(blocks))
    i, j, k = row % nx, (row // nx) % ny, row // (nx * ny)
    np.testing.assert_array_equal(blocks["x"], X0 + 200 + 5.0 + 10.0 * i)
    np.testing.assert_array_equal(blocks["y"], Y0 + 200 + 5.0 + 10.0 * j)
    np.testing.assert_array_equal(blocks["z"], 940.0 + 2.5 + 5.0 * k)
    centroids = blocks[["x", "y", "z"]].to_numpy()
    for layer in ("lith", "weathering", "fault_block", "rock"):
        assert blocks[layer].tolist() == small.truth.labels(centroids, layer).tolist()
    assert set(blocks["lith"]) == set(LITH)
    assert set(blocks["rock"]) == {"AIR", "ORE", "WASTE"}
    # hand rules at the centroids
    depth = topo_hand(blocks["x"], blocks["y"]) - blocks["z"]
    assert ((blocks["lith"] == "AIR") == (depth < 0)).all()
    assert (
        (blocks["lith"] == "Q4") == ((depth >= 0) & (depth < cover_hand(blocks["x"], blocks["y"])))
    ).all()
    assert ((blocks["fault_block"] == "FB2") == hanging_wall_hand(centroids)).all()
    np.testing.assert_allclose(blocks[fractions].sum(axis=1), 1.0, atol=1e-12)
    # fractions of a sample of blocks from a 4 x 4 x 4 lattice built here
    pick = np.random.default_rng(3).choice(len(blocks), 300, replace=False)
    pick = np.concatenate(
        [pick, np.flatnonzero((blocks["frac_GRD"] > 0) & (blocks["frac_GRD"] < 1))[:100]]
    )
    offsets = (np.arange(4) + 0.5) / 4 - 0.5
    lattice = np.array(
        [[a * 10.0, b * 10.0, c * 5.0] for a in offsets for b in offsets for c in offsets]
    )
    for code in LITH:
        labels = small.truth.labels(
            (centroids[pick][:, None, :] + lattice[None]).reshape(-1, 3), "lith"
        )
        fraction = (labels.reshape(len(pick), 64) == code).mean(axis=1)
        np.testing.assert_allclose(blocks[f"frac_{code}"].to_numpy()[pick], fraction, atol=1e-12)


def test_volumes_against_monte_carlo(small):
    """truth.json volumes against an independent estimate: uniform random points in the box."""
    volumes = small.json("truth")["volumes"]
    box = 400.0 * 400.0 * 310.0
    total = sum(entry["volume_m3"] for entry in volumes["units"].values())
    assert total == pytest.approx(box, rel=1e-12)
    n = 2_000_000
    rng = np.random.default_rng(11)
    points = rng.uniform([X0 + 200, Y0 + 200, 940.0], [X0 + 600, Y0 + 600, 1250.0], (n, 3))
    labels = small.truth.labels(points, "lith")
    for code in LITH:
        share = float(np.mean(labels == code))
        sigma = math.sqrt((1.0 - share) / (share * n))  # relative standard error
        assert volumes["units"][code]["volume_m3"] == pytest.approx(share * box, rel=4.0 * sigma), (
            code
        )
    below = sum(v["volume_m3"] for code, v in volumes["units"].items() if code != "AIR")
    assert volumes["below_topography_m3"] == pytest.approx(below, rel=1e-12)


def test_dyke_volume_against_its_plan_area(small):
    """The 8 m dyke: volume = sum over plan of its thickness below the topography.

    Outside the stock nothing but the cover and the air replaces the dyke, so in plan cells
    away from the stock's footprint its volume is (strip area) x (rock height). The plan
    integral uses 0.25 m cells rotated into the dyke's own frame, where the strip is exact.
    """
    volumes = small.json("truth")["volumes"]["units"]
    s = np.arange(-400.0, 400.0, 0.25) + 0.125  # along strike 045
    w = np.arange(-4.0, 4.0, 0.25) + 0.125  # across, |w| < 4
    ss, ww = (a.ravel() for a in np.meshgrid(s, w))
    x = X0 + 480.0 + ss * S45 + ww * S45
    y = Y0 + 300.0 + ss * S45 - ww * S45
    inside = (x > X0 + 200) & (x < X0 + 600) & (y > Y0 + 200) & (y < Y0 + 600)
    x, y = x[inside], y[inside]
    top = topo_hand(x, y) - cover_hand(x, y)
    # the stock never reaches the dyke: nearest approach in plan is far more than its semi-axes
    along = (x - (X0 + 300)) * S45 + (y - (Y0 + 480)) * S45
    across = (x - (X0 + 300)) * S45 - (y - (Y0 + 480)) * S45
    assert ((along / 110.0) ** 2 + (across / 80.0) ** 2 > 1.0).all()
    by_hand = float(np.sum(top - 940.0) * 0.25 * 0.25)
    assert volumes["GRD"]["volume_m3"] == pytest.approx(by_hand, rel=5e-3)
    assert volumes["GRD"]["volume_coarse_m3"] == pytest.approx(by_hand, rel=1e-2)


def test_generator_self_checks_hold_for_layered_small(small):
    truth_json = small.json("truth")
    geology, lookup = small.csv("geology"), small.unit_of()
    unit = geology["lith"].map(lambda code: lookup[fold(code)][0])
    holes_per_unit = geology.assign(unit=unit).groupby("unit")["hole_id"].nunique()
    for code in ("SST", "SLT", "AND", "DIO", "GRD"):
        assert holes_per_unit[code] >= 8, code
    assert holes_per_unit.to_dict() == truth_json["drilling"]["holes_per_logged_unit"]
    holes = small.holes()
    ends = small.xyz(holes.index, holes["max_depth"])
    crossing = hanging_wall_hand(holes[["x", "y", "z"]].to_numpy()) != hanging_wall_hand(ends)
    assert crossing.sum() >= 6
    assert truth_json["drilling"]["holes_crossing_F1"] == crossing.sum()
    assert truth_json["ore_lens"]["min_cover_below_topography_m"] >= 10.0
    for code, entry in truth_json["volumes"]["units"].items():
        assert entry["relative_difference"] == pytest.approx(
            entry["volume_coarse_m3"] / entry["volume_m3"] - 1.0, abs=1e-12
        )
        if code != "AIR":
            assert abs(entry["relative_difference"]) <= 0.01, code
    assert truth_json["volumes"]["spacings_m"] == [2.0, 1.0]


def test_a_failed_self_check_raises():
    from dataclasses import replace

    config = replace(LAYERED_PRESETS["layered-small"], min_fault_holes=1000)
    with pytest.raises(SelfCheckError, match="holes cross F1"):
        generate_layered(config, seed=SEED)


def test_truth_json(small):
    truth_json = small.json("truth")
    fault = truth_json["fault"]
    assert fault["dip"] == 65.0 and fault["dip_azimuth"] == 70.0
    np.testing.assert_allclose(fault["normal"], N_F, atol=1e-12)
    np.testing.assert_allclose(fault["point"], P_F)
    assert fault["slip"] == pytest.approx(33.101, abs=5e-4)
    assert fault["throw"] == pytest.approx(30.0, abs=1e-9)
    assert fault["heave"] == pytest.approx(13.989, abs=5e-4)
    assert fault["vertical_separation_SST_SLT"] == pytest.approx(30.0, abs=1e-8)
    assert truth_json["seed"] == SEED and truth_json["preset"] == "layered-small"
    assert truth_json["config"] == json.loads(
        json.dumps(LAYERED_PRESETS["layered-small"].to_dict())
    )
    assert truth_json["drilling"]["holes"] == 64
    assert truth_json["drilling"]["geology_intervals"] == len(small.csv("geology"))
    assert truth_json["drilling"]["assay_intervals"] == len(small.csv("assay"))
    assert set(truth_json["files"]) == EXPECTED_FILES
    assert {"layers", "fault", "stock", "dyke", "cover", "weathering", "topography"} <= set(
        truth_json["formulas"]
    )


def test_presets_carry_the_contract_numbers():
    full, reduced = LAYERED_PRESETS["layered"], LAYERED_PRESETS["layered-small"]
    assert isinstance(full, LayeredConfig) and set(LAYERED_PRESETS) == {"layered", "layered-small"}
    assert (full.x_range, full.y_range, full.z_range) == (
        (500000, 500800),
        (5300000, 5300800),
        (880, 1260),
    )
    assert (reduced.x_range, reduced.y_range, reduced.z_range) == (
        (500200, 500600),
        (5300200, 5300600),
        (940, 1250),
    )
    assert (full.hole_length, reduced.hole_length) == (200.0, 180.0)
    assert (full.hole_spacing, full.hole_dip, full.hole_azimuth) == (50.0, -60.0, 160.0)
    assert (full.min_holes_per_unit, full.min_fault_holes) == (20, 15)
    assert (reduced.min_holes_per_unit, reduced.min_fault_holes) == (8, 6)
    lens = full.lens
    assert lens.centre == (500450.0, 5300400.0, 1095.0)
    assert (lens.bearing, lens.plunge, lens.dip) == (70.0, 0.0, 20.0)
    assert (lens.half_length, lens.half_width, lens.half_thickness) == (120.0, 60.0, 8.0)
    assert (lens.pinch_position, lens.pinch_width, lens.pinch_depth) == (30.0, 25.0, 0.5)
    assert (lens.bend_major, lens.bend_semi, lens.mesh_subdivisions) == (6.0, -4.0, 4)
    assert reduced.lens == lens and reduced.topo == full.topo


# ---------------------------------------------------------------------------------------
# determinism and the CLI
# ---------------------------------------------------------------------------------------
def test_cli_writes_every_file(cli_run, capsys):
    code, out = cli_run
    assert code == 0
    assert {p.name for p in out.iterdir()} == EXPECTED_FILES
    assert set(json.loads((out / "truth.json").read_text())["files"]) == EXPECTED_FILES


def test_same_seed_is_byte_identical(small, cli_run):
    _, out = cli_run
    assert {p.name for p in small.path.iterdir()} == EXPECTED_FILES
    for name in sorted(EXPECTED_FILES):
        assert (small.path / name).read_bytes() == (out / name).read_bytes(), name


def test_different_seed_changes_only_the_random_parts(small, tmp_path):
    other = generate_layered(LAYERED_PRESETS["layered-small"], seed=SEED + 1)
    other.write(tmp_path)
    for name in ("assay.csv", "geology.csv", "truth.json"):
        assert (small.path / name).read_bytes() != (tmp_path / name).read_bytes(), name
    fixed = EXPECTED_FILES - {"assay.csv", "geology.csv", "truth.json"}
    for name in sorted(fixed):
        assert (small.path / name).read_bytes() == (tmp_path / name).read_bytes(), name
    # the seed moves the core loss and the assay noise, never a contact or a zone
    a, b = small.csv("assay"), pd.read_csv(tmp_path / "assay.csv")
    assert a["wf"].tolist() == b["wf"].tolist() and not np.array_equal(a["au"], b["au"])


def test_default_seed_and_config():
    import inspect

    signature = inspect.signature(generate_layered)
    assert signature.parameters["seed"].default == 42
    assert signature.parameters["config"].default is None


def test_synth_presets_on_the_command_line():
    """The old presets are still there and ``default`` is still the default."""
    parser = cli.build_parser()
    assert parser.parse_args(["synth", "--out", "x"]).preset == "default"
    assert parser.parse_args(["synth", "--out", "x"]).seed == 42
    for preset in ("default", "small", "layered", "layered-small"):
        assert parser.parse_args(["synth", "--out", "x", "--preset", preset]).preset == preset
    with pytest.raises(SystemExit):
        parser.parse_args(["synth", "--out", "x", "--preset", "nonsense"])


# ---------------------------------------------------------------------------------------
# the full preset
# ---------------------------------------------------------------------------------------
@pytest.mark.slow
def test_full_preset(tmp_path):
    deposit = generate_layered(LAYERED_PRESETS["layered"], seed=SEED)
    written = deposit.write(tmp_path)
    assert {p.name for p in written} == EXPECTED_FILES
    full = Layered(deposit, tmp_path)
    collar = full.csv("collar")
    h = np.arange(256)
    assert collar["hole_id"].tolist() == [f"L{k + 1:03d}" for k in range(256)]
    np.testing.assert_allclose(collar["x"], X0 + 25 + 50 * (h % 16))
    np.testing.assert_allclose(collar["y"], Y0 + 25 + 50 * (h // 16))
    np.testing.assert_allclose(collar["max_depth"], np.where(h % 10 == 4, 160.0, 200.0))
    assert len(pd.read_parquet(tmp_path / "truth_blocks.parquet")) == 80 * 80 * 76
    geology, lookup = full.csv("geology"), full.unit_of()
    unit = geology["lith"].map(lambda code: lookup[fold(code)][0])
    holes_per_unit = geology.assign(unit=unit).groupby("unit")["hole_id"].nunique()
    for code in ("SST", "SLT", "AND", "GRD"):
        assert holes_per_unit[code] >= 20, code
    # Deviation from the contract (>= 20 for every unit): the stock cannot be cut by more than
    # 14 holes of the 50 m grid at any hole length, so its minimum is 12.
    assert holes_per_unit["DIO"] >= 12
    holes = full.holes()
    ends = full.xyz(holes.index, holes["max_depth"])
    crossing = hanging_wall_hand(holes[["x", "y", "z"]].to_numpy()) != hanging_wall_hand(ends)
    assert crossing.sum() >= 15
    truth_json = full.json("truth")
    for code, entry in truth_json["volumes"]["units"].items():
        if code != "AIR":
            assert abs(entry["relative_difference"]) <= 0.01, code
    total = sum(entry["volume_m3"] for entry in truth_json["volumes"]["units"].values())
    assert total == pytest.approx(800.0 * 800.0 * 380.0, rel=1e-12)
    vertices = np.load(tmp_path / "truth_oxide_base_vertices.npy")
    assert len(vertices) == 81 * 81
