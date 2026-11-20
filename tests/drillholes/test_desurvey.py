"""Acceptance tests for P1.T3 desurvey (research.md section 3)."""

import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import wellpathpy as wp  # independent oracle, dev dependency only
from hypothesis import given, settings
from hypothesis import strategies as st

from bakumine.core.drillholes.desurvey import (
    METHODS,
    HoleTraces,
    desurvey,
    direction_vectors,
)

BABBITT = Path(__file__).resolve().parents[2] / "data" / "public" / "babbitt"


def one_hole(depth, dip, azimuth, *, max_depth=None, collar=(0.0, 0.0, 0.0), hole_id="H1"):
    depth = np.asarray(depth, dtype=float)
    if max_depth is None:
        max_depth = float(depth.max())
    collar_df = pd.DataFrame(
        {
            "hole_id": [hole_id],
            "x": [collar[0]],
            "y": [collar[1]],
            "z": [collar[2]],
            "max_depth": [max_depth],
        }
    )
    survey_df = pd.DataFrame({"hole_id": hole_id, "depth": depth, "dip": dip, "azimuth": azimuth})
    return collar_df, survey_df


def codes(traces: HoleTraces) -> list[str]:
    return [w.code for w in traces.warnings]


# --------------------------------------------------------------------- exact cases


@pytest.mark.parametrize("method", METHODS)
def test_vertical_hole_exact(method):
    collar, survey = one_hole([0, 50, 120], [-90, -90, -90], [0, 0, 0], collar=(1000, 2000, 300))
    traces = desurvey(collar, survey, method=method)
    hole = traces.hole("H1")
    np.testing.assert_allclose(hole.depth, [0, 50, 120])
    np.testing.assert_allclose(
        hole.xyz, [[1000, 2000, 300], [1000, 2000, 250], [1000, 2000, 180]], atol=1e-11
    )
    xyz, direction = traces.xyz_at("H1", [0.0, 33.3, 120.0])
    np.testing.assert_allclose(xyz, [[1000, 2000, 300], [1000, 2000, 266.7], [1000, 2000, 180]],
                               atol=1e-11)
    np.testing.assert_allclose(direction, np.tile([0, 0, -1], (3, 1)), atol=1e-15)
    assert traces.warnings == []


@pytest.mark.parametrize("method", METHODS)
def test_straight_inclined_hole_exact(method):
    # Dip -60 towards azimuth 030: horizontal reach = L cos60 = L/2, drop = L sin60.
    # East = L/2 sin30 = L/4, north = L/2 cos30 = L sqrt(3)/4.
    collar, survey = one_hole([0, 100, 200], [-60, -60, -60], [30, 30, 30], collar=(500, 600, 70))
    traces = desurvey(collar, survey, method=method)
    r3 = math.sqrt(3.0)
    expected = np.array(
        [
            [500, 600, 70],
            [500 + 25, 600 + 25 * r3, 70 - 50 * r3],
            [500 + 50, 600 + 50 * r3, 70 - 100 * r3],
        ]
    )
    np.testing.assert_allclose(traces.hole("H1").xyz, expected, atol=1e-10)
    xyz, direction = traces.xyz_at("H1", 80.0)
    np.testing.assert_allclose(xyz, [500 + 20, 600 + 20 * r3, 70 - 40 * r3], atol=1e-10)
    np.testing.assert_allclose(direction, [0.25, r3 / 4, -r3 / 2], atol=1e-15)


def test_direction_vectors_axes():
    np.testing.assert_allclose(direction_vectors(-90, 0), [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(direction_vectors(0, 0), [0, 1, 0], atol=1e-15)  # north
    np.testing.assert_allclose(direction_vectors(0, 90), [1, 0, 0], atol=1e-15)  # east
    np.testing.assert_allclose(direction_vectors(90, 0, "positive_down"), [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(direction_vectors(45, 0), [0, math.sqrt(0.5), math.sqrt(0.5)])


# --------------------------------------------------------------------- quarter circle


@pytest.mark.parametrize("n_stations", [2, 3, 10])
def test_quarter_circle_within_1mm(n_stations):
    # Hole starts vertical and builds to horizontal heading east on a circle of radius R.
    # Centre of the circle is R east of the collar: x = R (1 - cos phi), z = -R sin phi.
    radius = 250.0
    length = radius * math.pi / 2
    depth = np.linspace(0.0, length, n_stations)
    phi = depth / radius
    collar, survey = one_hole(depth, -90.0 + np.degrees(phi), 90.0, collar=(10.0, 20.0, 30.0))
    traces = desurvey(collar, survey)

    def analytic(s):
        a = np.asarray(s) / radius
        xyz = np.stack(
            [10.0 + radius * (1 - np.cos(a)), np.full_like(a, 20.0), 30.0 - radius * np.sin(a)],
            axis=-1,
        )
        tangent = np.stack([np.sin(a), np.zeros_like(a), -np.cos(a)], axis=-1)
        return xyz, tangent

    hole = traces.hole("H1")
    assert np.abs(hole.xyz - analytic(hole.depth)[0]).max() < 1e-3
    queries = np.linspace(0.0, length, 1001)
    xyz, direction = traces.xyz_at("H1", queries)
    exact_xyz, exact_dir = analytic(queries)
    assert np.abs(xyz - exact_xyz).max() < 1e-3
    assert np.abs(xyz - exact_xyz).max() < 1e-9  # in fact exact to rounding
    assert np.abs(direction - exact_dir).max() < 1e-12
    # Chord interpolation would miss by far more than 1 mm: the test can tell them apart.
    chord_mid = 0.5 * (hole.xyz[0] + hole.xyz[1])
    arc_mid = traces.xyz_at("H1", 0.5 * hole.depth[1])[0]
    assert np.linalg.norm(chord_mid - arc_mid) > 0.1


# --------------------------------------------------------------------- oracle


def random_deviated_holes(seed, n_holes=50):
    rng = np.random.default_rng(seed)
    collars, surveys, per_hole = [], [], {}
    for i in range(n_holes):
        n = int(rng.integers(5, 40))
        depth = np.concatenate(([0.0], np.cumsum(rng.uniform(5.0, 60.0, n - 1))))
        dip = np.clip(rng.uniform(-90, -30) + np.cumsum(rng.normal(0, 2.5, n)), -89.5, 10.0)
        azimuth = (rng.uniform(0, 360) + np.cumsum(rng.normal(0, 6.0, n))) % 360.0
        xyz = rng.uniform(-5000, 5000, 3)
        hid = f"DH{i:03d}"
        collars.append((hid, *xyz, depth[-1]))
        surveys.append(pd.DataFrame({"hole_id": hid, "depth": depth, "dip": dip,
                                     "azimuth": azimuth}))
        per_hole[hid] = (xyz, depth, dip, azimuth)
    collar = pd.DataFrame(collars, columns=["hole_id", "x", "y", "z", "max_depth"])
    return collar, pd.concat(surveys, ignore_index=True), per_hole


def test_matches_wellpathpy_on_50_random_holes():
    collar, survey, per_hole = random_deviated_holes(seed=20261119)
    # Shuffle rows: input order must not matter.
    survey = survey.sample(frac=1.0, random_state=1).reset_index(drop=True)
    traces = desurvey(collar, survey, method="minimum_curvature", dip_convention="negative_down")
    assert traces.warnings == []
    assert len(traces) == 50
    worst = 0.0
    for hid, (xyz0, depth, dip, azimuth) in per_hole.items():
        # wellpathpy: inclination from vertical (0 = straight down), TVD positive down.
        oracle = wp.deviation(md=depth, inc=90.0 + dip, azi=azimuth).minimum_curvature()
        expected = np.column_stack(
            [xyz0[0] + oracle.easting, xyz0[1] + oracle.northing, xyz0[2] - oracle.depth]
        )
        hole = traces.hole(hid)
        np.testing.assert_array_equal(hole.depth, depth)
        assert set(hole.kind) == {"survey"}
        worst = max(worst, np.abs(hole.xyz - expected).max())
        # xyz_at at station depths returns the stations.
        at_stations, _ = traces.xyz_at(hid, depth)
        worst = max(worst, np.abs(at_stations - expected).max())
    assert worst < 1e-6


def test_positive_down_matches_wellpathpy():
    collar, survey, per_hole = random_deviated_holes(seed=7, n_holes=5)
    survey = survey.assign(dip=-survey["dip"])
    traces = desurvey(collar, survey, dip_convention="positive_down")
    for hid, (xyz0, depth, dip, azimuth) in per_hole.items():
        oracle = wp.deviation(md=depth, inc=90.0 + dip, azi=azimuth).minimum_curvature()
        expected = np.column_stack(
            [xyz0[0] + oracle.easting, xyz0[1] + oracle.northing, xyz0[2] - oracle.depth]
        )
        assert np.abs(traces.hole(hid).xyz - expected).max() < 1e-6


# --------------------------------------------------------------------- other methods


def two_station_example():
    # Station 1: depth 0, vertical down (dip -90).  Station 2: depth 100, dip -30, azimuth 090.
    # d1 = (0, 0, -1);  d2 = (cos30, 0, -sin30) = (sqrt3/2, 0, -1/2).
    return one_hole([0, 100], [-90, -30], [90, 90])


def test_tangent_two_station_hand_computed():
    traces = desurvey(*two_station_example(), method="tangent")
    # Whole interval on the upper direction: (0, 0, -100).
    np.testing.assert_allclose(traces.hole("H1").xyz[1], [0, 0, -100], atol=1e-12)
    xyz, direction = traces.xyz_at("H1", [40.0])
    np.testing.assert_allclose(xyz, [[0, 0, -40]], atol=1e-12)
    np.testing.assert_allclose(direction, [[0, 0, -1]], atol=1e-15)


def test_balanced_tangential_two_station_hand_computed():
    traces = desurvey(*two_station_example(), method="balanced_tangential")
    # 50 * d1 + 50 * d2 = (25 sqrt3, 0, -50 - 25) = (43.30127..., 0, -75).
    np.testing.assert_allclose(traces.hole("H1").xyz[1], [25 * math.sqrt(3), 0, -75], atol=1e-12)
    # Inside the interval: first half on d1, second half on d2.
    xyz, direction = traces.xyz_at("H1", [20.0, 80.0])
    np.testing.assert_allclose(xyz[0], [0, 0, -20], atol=1e-12)
    np.testing.assert_allclose(xyz[1], [30 * math.sqrt(3) / 2, 0, -50 - 15], atol=1e-12)
    np.testing.assert_allclose(direction[0], [0, 0, -1], atol=1e-15)
    np.testing.assert_allclose(direction[1], [math.sqrt(3) / 2, 0, -0.5], atol=1e-15)


def test_minimum_curvature_two_station_hand_computed():
    traces = desurvey(*two_station_example(), method="minimum_curvature")
    # Dogleg 60 deg = pi/3; RF = (2/beta) tan(beta/2) = (6/pi) tan(30 deg) = 1.10265779...
    rf = (6 / math.pi) * math.tan(math.pi / 6)
    np.testing.assert_allclose(
        traces.hole("H1").xyz[1], [25 * math.sqrt(3) * rf, 0, -75 * rf], atol=1e-12
    )


def test_unknown_method_and_convention_rejected():
    collar, survey = two_station_example()
    with pytest.raises(ValueError, match="method"):
        desurvey(collar, survey, method="average_angle")
    with pytest.raises(ValueError, match="dip_convention"):
        desurvey(collar, survey, dip_convention="up")
    with pytest.raises(ValueError, match="missing columns"):
        desurvey(collar.drop(columns="max_depth"), survey)
    with pytest.raises(ValueError, match="duplicate hole_id"):
        desurvey(pd.concat([collar, collar]), survey)


# --------------------------------------------------------------------- dip convention


@pytest.mark.parametrize("method", METHODS)
def test_dip_conventions_agree_for_sign_flipped_input(method):
    collar, survey, _ = random_deviated_holes(seed=3, n_holes=8)
    negative = desurvey(collar, survey, method=method, dip_convention="negative_down")
    positive = desurvey(
        collar, survey.assign(dip=-survey["dip"]), method=method, dip_convention="positive_down"
    )
    a, b = negative.stations(), positive.stations()
    np.testing.assert_array_equal(a[["x", "y", "z"]].to_numpy(), b[["x", "y", "z"]].to_numpy())
    assert (a["z"].to_numpy()[1] < a["z"].to_numpy()[0]) and a["dz"].iloc[0] < 0  # goes down
    rng = np.random.default_rng(0)
    ids = collar["hole_id"].to_numpy()[rng.integers(0, 8, 200)]
    depths = rng.uniform(0, 1, 200) * collar.set_index("hole_id").loc[ids, "max_depth"].to_numpy()
    for got_a, got_b in zip(negative.xyz_at(ids, depths), positive.xyz_at(ids, depths),
                            strict=True):
        np.testing.assert_array_equal(got_a, got_b)


# --------------------------------------------------------------------- edge handling


def test_no_survey_at_depth_zero_uses_first_station_and_warns():
    collar, survey = one_hole([30, 60], [-60, -60], [90, 90], collar=(0, 0, 0))
    traces = desurvey(collar, survey)
    assert codes(traces) == ["no_station_at_zero"]
    assert traces.warnings[0].hole_id == "H1"
    hole = traces.hole("H1")
    np.testing.assert_allclose(hole.depth, [0, 30, 60])
    assert list(hole.kind) == ["collar", "survey", "survey"]
    np.testing.assert_allclose(hole.xyz[0], [0, 0, 0])
    np.testing.assert_allclose(hole.xyz[1], [15, 0, -15 * math.sqrt(3)], atol=1e-12)
    xyz, _ = traces.xyz_at("H1", 10.0)
    np.testing.assert_allclose(xyz, [5, 0, -5 * math.sqrt(3)], atol=1e-12)


@pytest.mark.parametrize("method", METHODS)
def test_single_station_hole_is_straight(method):
    collar, survey = one_hole([0], [-45], [180], max_depth=100.0)
    traces = desurvey(collar, survey, method=method)
    assert traces.warnings == []
    hole = traces.hole("H1")
    assert list(hole.kind) == ["survey", "end"]
    h = 100 * math.sqrt(0.5)
    np.testing.assert_allclose(hole.xyz[1], [0, -h, -h], atol=1e-12)
    xyz, direction = traces.xyz_at("H1", [0.0, 50.0, 100.0])
    np.testing.assert_allclose(xyz, [[0, 0, 0], [0, -h / 2, -h / 2], [0, -h, -h]], atol=1e-12)
    np.testing.assert_allclose(direction, np.tile([0, -math.sqrt(0.5), -math.sqrt(0.5)], (3, 1)),
                               atol=1e-15)


def test_single_station_not_at_zero():
    collar, survey = one_hole([25], [-90], [0], max_depth=80.0)
    traces = desurvey(collar, survey)
    assert codes(traces) == ["no_station_at_zero"]
    assert list(traces.hole("H1").kind) == ["collar", "survey", "end"]
    np.testing.assert_allclose(traces.xyz_at("H1", 80.0)[0], [0, 0, -80], atol=1e-12)


def test_beyond_last_station_extends_straight_then_nan_or_error():
    collar, survey = one_hole([0, 100], [-90, -30], [90, 90], max_depth=160.0)
    traces = desurvey(collar, survey)
    hole = traces.hole("H1")
    assert list(hole.kind) == ["survey", "survey", "end"]
    last_dir = np.array([math.sqrt(3) / 2, 0, -0.5])
    np.testing.assert_allclose(hole.xyz[2], hole.xyz[1] + 60 * last_dir, atol=1e-12)
    xyz, direction = traces.xyz_at("H1", [130.0, 160.0, 160.0 + 1e-12, 160.5, -0.5, np.nan])
    np.testing.assert_allclose(xyz[0], hole.xyz[1] + 30 * last_dir, atol=1e-12)
    np.testing.assert_allclose(direction[0], last_dir, atol=1e-15)
    np.testing.assert_allclose(xyz[1], hole.xyz[2], atol=1e-12)
    np.testing.assert_allclose(xyz[2], hole.xyz[2], atol=1e-12)  # within depth_tolerance
    assert np.isnan(xyz[3:]).all() and np.isnan(direction[3:]).all()
    with pytest.raises(ValueError, match="depth outside"):
        traces.xyz_at("H1", [10.0, 160.5], on_invalid="raise")
    with pytest.raises(ValueError, match="depth outside"):
        traces.xyz_at("H1", -0.5, on_invalid="raise")
    # Unknown hole: NaN or error by the same option.
    assert np.isnan(traces.xyz_at("NOPE", 10.0)[0]).all()
    assert np.isnan(traces.xyz_at(["H1", "NOPE"], 10.0)[0][1]).all()
    with pytest.raises(ValueError, match="no trace"):
        traces.xyz_at(["H1", "NOPE"], 10.0, on_invalid="raise")
    with pytest.raises(KeyError):
        traces.hole("NOPE")


@pytest.mark.parametrize("method", METHODS)
def test_sentinel_station_deeper_than_max_depth_is_clamped(method):
    # Babbitt style: stations at 0 and 90000 with the same reading = straight to end of hole.
    collar, survey = one_hole([0, 90000], [-60, -60], [0, 0], max_depth=500.0)
    traces = desurvey(collar, survey, method=method)
    assert codes(traces) == ["station_beyond_max_depth"]
    hole = traces.hole("H1")
    np.testing.assert_array_equal(hole.depth, [0, 500])
    assert list(hole.kind) == ["survey", "end"]
    np.testing.assert_allclose(hole.xyz[1], [0, 250, -250 * math.sqrt(3)], atol=1e-9)
    xyz, _ = traces.xyz_at("H1", [250.0, 500.0, 501.0])
    np.testing.assert_allclose(xyz[0], [0, 125, -125 * math.sqrt(3)], atol=1e-9)
    np.testing.assert_allclose(xyz[1], hole.xyz[1], atol=1e-12)
    assert np.isnan(xyz[2]).all()


def test_curved_station_deeper_than_max_depth_cuts_the_same_arc():
    # A quarter circle surveyed at 0 and at its end; the hole stops half way along it.
    radius = 100.0
    length = radius * math.pi / 2
    full = desurvey(*one_hole([0, length], [-90, 0], [90, 90]))
    cut = desurvey(*one_hole([0, length, 2 * length], [-90, 0, 0], [90, 90, 90],
                             max_depth=length / 2))
    assert codes(cut) == ["station_beyond_max_depth"]
    hole = cut.hole("H1")
    np.testing.assert_allclose(hole.depth, [0, length / 2])
    r = math.sqrt(0.5)
    np.testing.assert_allclose(hole.xyz[1], [radius * (1 - r), 0, -radius * r], atol=1e-10)
    np.testing.assert_allclose(hole.direction[1], [r, 0, -r], atol=1e-12)
    q = np.linspace(0, length / 2, 20)
    np.testing.assert_allclose(cut.xyz_at("H1", q)[0], full.xyz_at("H1", q)[0], atol=1e-12)
    assert np.isnan(cut.xyz_at("H1", length * 0.75)[0]).all()


def test_station_exactly_at_max_depth_then_deeper_ones_are_dropped():
    collar, survey = one_hole([0, 100, 150, 200], [-90, -90, -10, -10], [0, 0, 0, 0],
                              max_depth=100.0)
    traces = desurvey(collar, survey)
    assert codes(traces) == ["station_beyond_max_depth"]
    hole = traces.hole("H1")
    np.testing.assert_array_equal(hole.depth, [0, 100])
    np.testing.assert_allclose(hole.xyz[1], [0, 0, -100], atol=1e-12)
    np.testing.assert_allclose(traces.xyz_at("H1", 100.0)[1], [0, 0, -1], atol=1e-15)


@pytest.mark.parametrize("method", METHODS)
def test_duplicate_stations_keep_first_and_warn(method):
    collar, survey = one_hole([0, 50, 50, 100], [-90, -60, -20, -60], [0, 90, 90, 90])
    clean_collar, clean_survey = one_hole([0, 50, 100], [-90, -60, -60], [0, 90, 90])
    traces = desurvey(collar, survey, method=method)
    clean = desurvey(clean_collar, clean_survey, method=method)
    assert codes(traces) == ["duplicate_station"]
    assert "readings differ" in traces.warnings[0].message
    np.testing.assert_array_equal(traces.hole("H1").xyz, clean.hole("H1").xyz)
    assert np.isfinite(traces.xyz_at("H1", np.linspace(0, 100, 11))[0]).all()


def test_identical_consecutive_directions_have_ratio_factor_one():
    # Dogleg 0 between stations 2 and 3 (and nearly 0 between 3 and 4): no division by zero.
    collar, survey = one_hole(
        [0, 40, 90, 140, 200], [-90, -70, -70, -70 + 1e-11, -60], [10, 45, 45, 45, 50]
    )
    with np.errstate(all="raise"):
        traces = desurvey(collar, survey)
        xyz, direction = traces.xyz_at("H1", np.linspace(0, 200, 401))
    hole = traces.hole("H1")
    assert np.isfinite(hole.xyz).all() and np.isfinite(xyz).all() and np.isfinite(direction).all()
    d = direction_vectors(-70, 45)
    np.testing.assert_allclose(hole.xyz[2] - hole.xyz[1], 50 * d, atol=1e-12)  # RF = 1
    np.testing.assert_allclose(hole.xyz[3] - hole.xyz[2], 50 * d, atol=1e-9)
    oracle = wp.deviation(
        md=survey["depth"].to_numpy(), inc=90 + survey["dip"].to_numpy(),
        azi=survey["azimuth"].to_numpy(),
    ).minimum_curvature()
    np.testing.assert_allclose(hole.xyz[:, 0], oracle.easting, atol=1e-6)
    np.testing.assert_allclose(hole.xyz[:, 2], -oracle.depth, atol=1e-6)


@pytest.mark.parametrize("method", METHODS)
def test_180_degree_dogleg_is_reported_and_finite(method):
    # Straight down, then straight up: no circular arc is defined between them.
    collar, survey = one_hole([0, 100, 200], [-90, 90, 90], [0, 0, 0])
    traces = desurvey(collar, survey, method=method)
    assert codes(traces) == ["dogleg_180"]
    hole = traces.hole("H1")
    assert np.isfinite(hole.xyz).all()
    xyz, direction = traces.xyz_at("H1", np.linspace(0, 200, 41))
    assert np.isfinite(xyz).all() and np.isfinite(direction).all()
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
    first_leg = {"minimum_curvature": -100.0, "tangent": -100.0, "balanced_tangential": 0.0}
    np.testing.assert_allclose(hole.xyz[1], [0, 0, first_leg[method]], atol=1e-9)

    # Nearly opposite but not degenerate: still an arc, no warning, still finite.
    collar, survey = one_hole([0, 100], [-90, 89.9], [0, 0])
    near = desurvey(collar, survey, method="minimum_curvature")
    assert near.warnings == []
    assert np.isfinite(near.xyz_at("H1", np.linspace(0, 100, 11))[0]).all()


def test_bad_rows_and_holes_are_reported_not_silently_used():
    collar = pd.DataFrame(
        {
            "hole_id": ["A", "B", "C", "D"],
            "x": [0.0, 10.0, np.nan, 30.0],
            "y": [0.0, 0.0, 0.0, 0.0],
            "z": [0.0, 0.0, 0.0, 0.0],
            "max_depth": [50.0, 50.0, 50.0, np.nan],
        }
    )
    survey = pd.DataFrame(
        {
            "hole_id": ["A", "A", "C", "D", "D", "X"],
            "depth": [0.0, np.nan, 0.0, 0.0, 70.0, 0.0],
            "dip": [-90.0, -90.0, -90.0, -90.0, -90.0, -90.0],
            "azimuth": [0.0] * 6,
        }
    )
    traces = desurvey(collar, survey)
    assert sorted(codes(traces)) == sorted(
        ["survey_without_collar", "invalid_station", "invalid_collar", "no_survey", "no_max_depth"]
    )
    by_code = {w.code: w.hole_id for w in traces.warnings}
    assert by_code == {
        "survey_without_collar": "X",
        "invalid_station": "A",
        "invalid_collar": "C",
        "no_survey": "B",
        "no_max_depth": "D",
    }
    assert list(traces.hole_ids) == ["A", "D"]
    assert "A" in traces and "B" not in traces
    assert list(traces.warnings_frame().columns) == ["hole_id", "code", "message"]
    np.testing.assert_allclose(traces.hole("D").xyz[-1], [30, 0, -70], atol=1e-12)
    assert traces.hole("D").max_depth == 70.0


def test_zero_length_hole():
    collar, survey = one_hole([0], [-90], [0], max_depth=0.0, collar=(1, 2, 3))
    traces = desurvey(collar, survey)
    xyz, direction = traces.xyz_at("H1", 0.0)
    np.testing.assert_allclose(xyz, [1, 2, 3])
    np.testing.assert_allclose(direction, [0, 0, -1], atol=1e-15)


def test_empty_tables():
    collar = pd.DataFrame({"hole_id": [], "x": [], "y": [], "z": [], "max_depth": []})
    survey = pd.DataFrame({"hole_id": [], "depth": [], "dip": [], "azimuth": []})
    traces = desurvey(collar, survey)
    assert len(traces) == 0 and len(traces.stations()) == 0
    assert np.isnan(traces.xyz_at(["A"], [1.0])[0]).all()


# --------------------------------------------------------------------- xyz_at mechanics


def test_xyz_at_many_holes_matches_per_hole_and_broadcasts():
    collar, survey, _ = random_deviated_holes(seed=11, n_holes=12)
    traces = desurvey(collar, survey)
    rng = np.random.default_rng(5)
    max_depth = collar.set_index("hole_id")["max_depth"]
    ids = collar["hole_id"].to_numpy()[rng.integers(0, 12, 500)]
    depths = rng.uniform(0, 1, 500) * max_depth.loc[ids].to_numpy()
    xyz, direction = traces.xyz_at(ids, depths)
    assert xyz.shape == (500, 3) and direction.shape == (500, 3)
    for hid in np.unique(ids):
        mask = ids == hid
        one_xyz, one_dir = traces.xyz_at(hid, depths[mask])
        np.testing.assert_array_equal(xyz[mask], one_xyz)
        np.testing.assert_array_equal(direction[mask], one_dir)
    np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
    # Scalar depth gives shape (3,); 2D depths keep their shape.
    assert traces.xyz_at(ids[0], 1.0)[0].shape == (3,)
    assert traces.xyz_at(ids[0], np.ones((4, 5)))[0].shape == (4, 5, 3)
    assert traces.xyz_at(ids[:4, None], np.ones((4, 5)))[1].shape == (4, 5, 3)
    # The station table is consistent with xyz_at.
    table = traces.stations()
    at, _ = traces.xyz_at(table["hole_id"].to_numpy(), table["depth"].to_numpy())
    np.testing.assert_allclose(at, table[["x", "y", "z"]].to_numpy(), atol=1e-9)


def test_integer_hole_ids():
    collar, survey = one_hole([0, 10], [-90, -90], [0, 0], hole_id=34873)
    traces = desurvey(collar, survey)
    np.testing.assert_allclose(traces.xyz_at(34873, 5.0)[0], [0, 0, -5], atol=1e-12)
    np.testing.assert_allclose(traces.xyz_at(np.array([34873]), 5.0)[0], [[0, 0, -5]], atol=1e-12)


def test_minimum_curvature_path_is_smooth_across_stations():
    collar, survey, _ = random_deviated_holes(seed=2, n_holes=3)
    traces = desurvey(collar, survey)
    table = traces.stations()
    inner = table[(table["depth"] > 0)]
    ids, depth = inner["hole_id"].to_numpy(), inner["depth"].to_numpy()
    eps = 1e-7
    below_xyz, below_dir = traces.xyz_at(ids, depth - eps)
    at_xyz, at_dir = traces.xyz_at(ids, depth)
    assert np.abs(below_xyz - at_xyz).max() < 2 * eps
    assert np.abs(below_dir - at_dir).max() < 1e-6


# --------------------------------------------------------------------- real dataset


def load_babbitt():
    collar = pd.read_csv(BABBITT / "collar.csv", dtype={"hole_id": str}).rename(
        columns={"easting": "x", "northing": "y", "elevation": "z", "depth": "max_depth"}
    )
    survey = pd.read_csv(BABBITT / "survey.csv", dtype={"hole_id": str})
    return collar, survey


@pytest.mark.parametrize("method", METHODS)
def test_babbitt_dataset(method):
    collar, survey = load_babbitt()
    traces = desurvey(collar, survey, method=method, dip_convention="positive_down")
    assert len(traces) == 399
    table = traces.stations()
    assert np.isfinite(table[["depth", "x", "y", "z", "dx", "dy", "dz"]].to_numpy()).all()
    assert set(codes(traces)) == {"station_beyond_max_depth"}

    collar = collar.set_index("hole_id")
    ends = table.groupby("hole_id", sort=False).last()
    np.testing.assert_array_equal(ends["depth"].to_numpy(), collar.loc[ends.index, "max_depth"])
    start = collar.loc[ends.index, ["x", "y", "z"]].to_numpy()
    reach = np.linalg.norm(ends[["x", "y", "z"]].to_numpy() - start, axis=1)
    max_depth = collar.loc[ends.index, "max_depth"].to_numpy()
    assert (reach <= max_depth * (1 + 1e-12)).all()
    assert (reach > 0.5 * max_depth).all()
    # Positive-down input: every hole must go down, not up.
    assert (ends["z"].to_numpy() < start[:, 2]).all()

    # Holes with the 90000 sentinel are straight on the direction of their first station.
    sentinel = survey.loc[survey["depth"] == 90000, "hole_id"].unique()
    assert len(sentinel) == 70
    assert sorted(w.hole_id for w in traces.warnings) == sorted(sentinel)
    first = survey[survey["hole_id"].isin(sentinel)].groupby("hole_id").first()
    for hid in sentinel:
        hole = traces.hole(hid)
        assert hole.depth.tolist() == [0.0, collar.loc[hid, "max_depth"]]
        d = direction_vectors(first.loc[hid, "dip"], first.loc[hid, "azimuth"], "positive_down")
        q = np.linspace(0, hole.max_depth, 7)
        xyz, direction = traces.xyz_at(hid, q)
        np.testing.assert_allclose(xyz, hole.xyz[0] + q[:, None] * d, atol=1e-8)
        np.testing.assert_allclose(direction, np.tile(d, (7, 1)), atol=1e-12)

    # Sample midpoints of every assay interval resolve without NaN.
    assay = pd.read_csv(BABBITT / "assay.csv", dtype={"hole_id": str})
    mid = 0.5 * (assay["from"].to_numpy() + assay["to"].to_numpy())
    xyz, direction = traces.xyz_at(assay["hole_id"].to_numpy(), mid, on_invalid="raise")
    assert np.isfinite(xyz).all() and np.isfinite(direction).all()


# --------------------------------------------------------------------- property test

reading = st.tuples(
    st.floats(0.5, 80.0),  # interval length
    st.floats(-90.0, 90.0),  # dip
    st.floats(0.0, 360.0),  # azimuth
)


@settings(max_examples=150, deadline=None)
@given(
    first=st.tuples(st.floats(-90.0, 90.0), st.floats(0.0, 360.0)),
    rest=st.lists(reading, min_size=1, max_size=8),
    sign=st.sampled_from(["negative_down", "positive_down"]),
)
def test_arc_length_equals_measured_depth_difference(first, rest, sign):
    depth = np.concatenate(([0.0], np.cumsum([r[0] for r in rest])))
    dip = np.array([first[0]] + [r[1] for r in rest])
    azimuth = np.array([first[1]] + [r[2] for r in rest])
    collar, survey = one_hole(depth, dip, azimuth)
    traces = desurvey(collar, survey, dip_convention=sign)
    hole = traces.hole("H1")
    assert np.isfinite(hole.xyz).all()
    degenerate = {w.code for w in traces.warnings} == {"dogleg_180"}
    assert degenerate or traces.warnings == []

    n_sub = 64
    for i in range(len(depth) - 1):
        d_md = depth[i + 1] - depth[i]
        chord = np.linalg.norm(hole.xyz[i + 1] - hole.xyz[i])
        assert chord <= d_md * (1 + 1e-12)
        if degenerate:
            # A 180-degree reversal is desurveyed as a straight line whose direction flips
            # at the end station (warning dogleg_180), so the arc-from-directions measure
            # below does not apply: it would read the flip as a half-turn in the last piece.
            continue

        # Walk the reconstructed path through xyz_at. Each small piece is measured as a
        # circular arc from its end points and end directions only:
        # arc = chord * (theta / 2) / sin(theta / 2), theta = turn of the direction.
        q = np.linspace(depth[i], depth[i + 1], n_sub + 1)
        xyz, direction = traces.xyz_at("H1", q)
        assert np.isfinite(xyz).all() and np.isfinite(direction).all()
        np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1.0, atol=1e-12)
        pieces = np.linalg.norm(np.diff(xyz, axis=0), axis=1)
        cos_turn = np.clip(np.einsum("ij,ij->i", direction[:-1], direction[1:]), -1.0, 1.0)
        sin_turn = np.linalg.norm(np.cross(direction[:-1], direction[1:]), axis=1)
        half = 0.5 * np.arctan2(sin_turn, cos_turn)
        factor = np.where(half > 1e-8, half / np.where(half > 1e-8, np.sin(half), 1.0), 1.0)
        path = float(np.sum(pieces * factor))
        assert abs(path - d_md) < 1e-6
        # The plain polyline is never longer than the measured depth and converges to it.
        assert pieces.sum() <= d_md * (1 + 1e-12)
        assert d_md - pieces.sum() < 1e-3 * d_md + 1e-9


# --------------------------------------------------------------------- performance


def test_performance_5000_holes_and_1m_queries():
    rng = np.random.default_rng(42)
    n_holes, n_stations = 5000, 100
    ids = np.array([f"P{i:04d}" for i in range(n_holes)], dtype=object)
    collar = pd.DataFrame(
        {
            "hole_id": ids,
            "x": rng.uniform(0, 10_000, n_holes),
            "y": rng.uniform(0, 10_000, n_holes),
            "z": rng.uniform(0, 500, n_holes),
            "max_depth": 1000.0,
        }
    )
    dip = -65 + np.cumsum(rng.normal(0, 0.5, (n_holes, n_stations)), axis=1)
    azimuth = rng.uniform(0, 360, (n_holes, 1)) + np.cumsum(
        rng.normal(0, 1.0, (n_holes, n_stations)), axis=1
    )
    survey = pd.DataFrame(
        {
            "hole_id": np.repeat(ids, n_stations),
            "depth": np.tile(np.arange(n_stations) * 10.0, n_holes),
            "dip": dip.ravel(),
            "azimuth": azimuth.ravel() % 360.0,
        }
    )
    query_ids = ids[rng.integers(0, n_holes, 1_000_000)]
    query_depths = rng.uniform(0, 1000, 1_000_000)

    start = time.perf_counter()
    traces = desurvey(collar, survey)
    xyz, direction = traces.xyz_at(query_ids, query_depths)
    elapsed = time.perf_counter() - start

    assert len(traces.stations()) == n_holes * (n_stations + 1)  # + the end station at 1000
    assert np.isfinite(xyz).all() and np.isfinite(direction).all()
    assert elapsed < 10.0, f"desurvey + 1M xyz_at took {elapsed:.2f} s"
