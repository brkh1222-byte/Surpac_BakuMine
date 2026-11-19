"""Drillhole tables: schema, conventions, interval rules, agreement with the truth solid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.synth.config import default_config
from bakumine.synth.drillholes import HolePath, direction


def _paths(dataset) -> dict[str, HolePath]:
    """Paths rebuilt from the CSV files only (what a user of the data has)."""
    collar = dataset.csv("collar").set_index("hole_id")
    paths = {}
    for hole_id, s in dataset.csv("survey").groupby("hole_id", sort=False):
        c = collar.loc[hole_id]
        paths[hole_id] = HolePath(
            np.array([c.x, c.y, c.z]), s.depth.to_numpy(), s.dip.to_numpy(), s.azimuth.to_numpy()
        )
    return paths


def test_schemas(dataset):
    assert list(dataset.csv("collar").columns) == ["hole_id", "x", "y", "z", "max_depth"]
    assert list(dataset.csv("survey").columns) == ["hole_id", "depth", "dip", "azimuth"]
    assert list(dataset.csv("assay").columns) == ["hole_id", "depth_from", "depth_to", "au"]
    assert list(dataset.csv("geology").columns) == ["hole_id", "depth_from", "depth_to", "rock"]
    assert list(dataset.csv("truth_samples").columns) == [
        "hole_id", "depth_from", "depth_to", "x", "y", "z", "rock", "ore_fraction", "gauss",
        "au_true",
    ]  # fmt: skip
    assert list(dataset.csv("truth_stations").columns) == ["hole_id", "depth", "x", "y", "z"]
    assert set(dataset.csv("geology").rock) == {"ORE", "WASTE"}
    for name in ("collar", "survey", "assay", "geology"):
        assert not dataset.csv(name).isna().any().any()  # gaps are absent rows, not blanks


def test_hole_count_and_mine_grid_coordinates(dataset):
    collar = dataset.csv("collar")
    assert 60 <= len(collar) <= 100
    assert collar.hole_id.is_unique
    assert collar.x.between(499_000, 502_000).all()
    assert collar.y.between(5_299_000, 5_302_000).all()
    assert collar.z.between(1_100, 1_300).all()
    cfg = default_config().drill
    assert 60 <= cfg.n_sections * cfg.holes_per_section <= 100


def test_collars_lie_on_the_topography(dataset):
    collar = dataset.csv("collar")
    dz = collar.z.to_numpy() - dataset.deposit.topo.z(collar.x.to_numpy(), collar.y.to_numpy())
    assert np.abs(dz).max() < 1e-6
    # ... and on the DTM itself: every collar is one of its vertices.
    dtm = dataset.npy("topo_vertices")
    lookup = {(round(x, 3), round(y, 3)): z for x, y, z in dtm.tolist()}
    for row in collar.itertuples():
        assert abs(lookup[(round(row.x, 3), round(row.y, 3))] - row.z) < 1e-6


def test_survey_conventions_and_gradual_deviation(dataset):
    survey = dataset.csv("survey")
    collar = dataset.csv("collar").set_index("hole_id")
    assert set(survey.hole_id) == set(collar.index)
    assert survey.dip.between(-90.0, -20.0).all()  # negative downward
    assert survey.azimuth.between(0.0, 360.0, inclusive="left").all()
    first = survey.groupby("hole_id", sort=False).first()
    assert (first.depth == 0.0).all()  # a station at the collar for every hole
    inclined = (first.dip > -85.0).mean()
    assert 0.8 <= inclined < 1.0  # mostly inclined, a few vertical
    bent = 0
    for hole_id, s in survey.groupby("hole_id", sort=False):
        assert np.all(np.diff(s.depth) > 0)
        assert np.allclose(np.diff(s.depth), 30.0)
        assert s.depth.max() < collar.loc[hole_id, "max_depth"]
        t = direction(s.dip.to_numpy(), s.azimuth.to_numpy())
        dogleg = np.degrees(np.arccos(np.clip(np.sum(t[:-1] * t[1:], axis=1), -1, 1)))
        assert dogleg.max() < 3.0  # well under the 10 deg / 30 m "suspect" threshold
        total = np.degrees(np.arccos(np.clip(np.dot(t[0], t[-1]), -1, 1)))
        bent += total > 1.0
    assert bent > 0.9 * len(collar)  # the holes really deviate


@pytest.mark.parametrize("table", ["assay", "geology"])
def test_intervals_are_ordered_non_overlapping_and_within_depth(dataset, table):
    data = dataset.csv(table)
    max_depth = dataset.csv("collar").set_index("hole_id").max_depth
    assert set(data.hole_id) == set(max_depth.index)
    for hole_id, rows in data.groupby("hole_id", sort=False):
        lo, hi = rows.depth_from.to_numpy(), rows.depth_to.to_numpy()
        assert lo[0] >= 0.0
        assert np.all(hi > lo)
        assert np.all(lo[1:] >= hi[:-1])  # sorted, no overlap
        assert hi[-1] <= max_depth[hole_id]
        if table == "geology":  # logged from collar to end of hole without gaps
            assert lo[0] == 0.0 and hi[-1] == max_depth[hole_id]
            assert np.array_equal(lo[1:], hi[:-1])
            assert np.all(rows.rock.to_numpy()[1:] != rows.rock.to_numpy()[:-1])


def test_assay_samples_are_irregular_with_gaps(dataset):
    assay = dataset.csv("assay")
    length = (assay.depth_to - assay.depth_from).round(2)
    assert length.between(0.3, 1.7).all()
    assert 0.5 < (length == 1.0).mean() < 0.8
    assert length.nunique() >= 8
    gaps = 0
    for _, rows in assay.groupby("hole_id", sort=False):
        gaps += int(np.sum(rows.depth_from.to_numpy()[1:] > rows.depth_to.to_numpy()[:-1]))
    assert 20 <= gaps <= 2 * assay.hole_id.nunique()
    assert (assay.au > 0).all()


def test_truth_positions_follow_from_the_input_tables(dataset):
    """The oracle XYZ is the minimum-curvature position computed from collar.csv + survey.csv."""
    truth = dataset.csv("truth_samples")
    assay = dataset.csv("assay")
    assert truth.hole_id.tolist() == assay.hole_id.tolist()
    assert np.array_equal(truth.depth_from, assay.depth_from)
    assert np.array_equal(truth.depth_to, assay.depth_to)
    paths = _paths(dataset)
    for hole_id, rows in truth.groupby("hole_id", sort=False):
        mid = 0.5 * (rows.depth_from.to_numpy() + rows.depth_to.to_numpy())
        xyz = paths[hole_id].xyz(mid)
        assert np.abs(xyz - rows[["x", "y", "z"]].to_numpy()).max() < 2e-6
    stations = dataset.csv("truth_stations")
    for hole_id, rows in stations.groupby("hole_id", sort=False):
        assert np.abs(paths[hole_id].station - rows[["x", "y", "z"]].to_numpy()).max() < 2e-6


def test_geology_log_agrees_with_implicit_solid(dataset):
    solid = dataset.deposit.solid
    truth = dataset.csv("truth_samples")
    geology = dataset.csv("geology")
    inside = solid.inside(truth[["x", "y", "z"]].to_numpy())
    assert np.array_equal(inside, truth.rock.to_numpy() == "ORE")

    checked = straddling = 0
    for hole_id, rows in truth.groupby("hole_id", sort=False):
        g = geology[geology.hole_id == hole_id]
        g_from, g_to = g.depth_from.to_numpy(), g.depth_to.to_numpy()
        mid = 0.5 * (rows.depth_from.to_numpy() + rows.depth_to.to_numpy())
        k = np.searchsorted(g_to, mid, side="right")  # geology interval holding the midpoint
        logged_ore = g.rock.to_numpy()[k] == "ORE"
        whole = (rows.depth_from.to_numpy() >= g_from[k]) & (rows.depth_to.to_numpy() <= g_to[k])
        hole_inside = inside[rows.index.to_numpy()]
        # Samples wholly within one geology interval: no exception allowed.
        assert np.array_equal(hole_inside[whole], logged_ore[whole]), hole_id
        fraction = rows.ore_fraction.to_numpy()
        assert np.all(fraction[whole] == logged_ore[whole])
        # Boundary-straddling samples may disagree only within the 0.01 m contact rounding.
        contact_distance = np.minimum(np.abs(mid - g_from[k]), np.abs(mid - g_to[k]))
        wrong = ~whole & (hole_inside != logged_ore)
        assert np.all(contact_distance[wrong] <= 0.0051), hole_id
        checked += int(whole.sum())
        straddling += int((~whole).sum())
    n_contacts = len(geology) - geology.hole_id.nunique()
    assert straddling <= n_contacts  # at most one straddling sample per contact
    assert checked > 0.98 * len(truth)
    assert (geology.rock == "ORE").sum() >= 40  # most holes hit the lens


def test_ore_hits_are_true_crossings_of_the_solid(dataset):
    """Both ends of every logged ORE interval lie on the surface f = 0 (to the cm rounding)."""
    solid = dataset.deposit.solid
    geology = dataset.csv("geology")
    max_depth = dataset.csv("collar").set_index("hole_id").max_depth
    paths = _paths(dataset)
    for row in geology[geology.rock == "ORE"].itertuples():
        path = paths[row.hole_id]
        for depth in (row.depth_from, row.depth_to):
            if depth in (0.0, max_depth[row.hole_id]):
                continue
            f = solid.implicit(path.xyz(np.array([depth - 0.02, depth + 0.02])))
            assert f[0] * f[1] < 0, (row.hole_id, depth)


def test_assays_come_from_the_truth_grade(dataset):
    truth = dataset.csv("truth_samples")
    assay = dataset.csv("assay")
    grade = dataset.deposit.grade
    cfg = dataset.deposit.config.grade
    exact = dataset.deposit.tables["truth_samples"]["au_true"]
    expected = np.maximum(np.round(exact, cfg.assay_decimals), cfg.assay_floor)
    assert np.array_equal(assay.au.to_numpy(), expected)
    assert np.abs(truth.au_true - exact).max() < 1e-6  # the truth file keeps 6 decimals
    ore = truth.ore_fraction == 1.0
    waste = truth.ore_fraction == 0.0
    assert truth.gauss[waste].isna().all() and truth.gauss[truth.rock == "ORE"].notna().all()
    # Ore: positive and right-skewed around the stated lognormal; waste: low background.
    au_ore = truth.au_true[ore]
    assert au_ore.mean() > au_ore.median() > 0
    assert 0.6 * grade.ore_ensemble_mean < au_ore.mean() < 1.5 * grade.ore_ensemble_mean
    assert abs(truth.au_true[waste].mean() / grade.waste_mean - 1.0) < 0.05
    assert truth.au_true[waste].max() < 0.25
    # The midpoint Gaussian value explains the sample grade (sub-points share the nugget).
    predicted = np.exp(grade.mu + grade.sigma * truth.gauss[ore])
    assert np.corrcoef(np.log(predicted), np.log(au_ore))[0, 1] > 0.97


# --- minimum curvature: checks against analytic cases and an independent integration -------


def test_direction_convention():
    assert np.allclose(direction(-90.0, 0.0), [0, 0, -1])
    assert np.allclose(direction(0.0, 0.0), [0, 1, 0])  # horizontal, north
    assert np.allclose(direction(0.0, 90.0), [1, 0, 0])  # horizontal, east
    assert np.allclose(direction(-30.0, 90.0), [np.cos(np.pi / 6), 0, -0.5])


def test_vertical_and_straight_inclined_holes_are_exact():
    collar = np.array([500_000.0, 5_300_000.0, 1_200.0])
    vertical = HolePath(collar, [0.0, 50.0, 100.0], [-90.0] * 3, [0.0, 120.0, 240.0])
    expect = [[0, 0, 0], [0, 0, -37.5], [0, 0, -100], [0, 0, -160]]
    assert np.allclose(vertical.xyz([0.0, 37.5, 100.0, 160.0]) - collar, expect, atol=1e-9)
    inclined = HolePath(collar, [0.0, 80.0], [-60.0, -60.0], [90.0, 90.0])
    depth = np.array([0.0, 10.0, 80.0, 200.0])  # the last one is below the last station
    expect = depth[:, None] * [np.cos(np.radians(60)), 0.0, -np.sin(np.radians(60))]
    assert np.allclose(inclined.xyz(depth) - collar, expect, atol=1e-9)


def test_quarter_circle_matches_the_analytic_arc():
    radius = 200.0
    length = 0.5 * np.pi * radius
    path = HolePath(np.zeros(3), [0.0, length], [0.0, -90.0], [0.0, 0.0])  # north, turning down
    s = np.linspace(0.0, length, 41)
    expect = np.column_stack([np.zeros_like(s), radius * np.sin(s / radius),
                              -radius * (1.0 - np.cos(s / radius))])  # fmt: skip
    assert np.abs(path.xyz(s) - expect).max() < 1e-9


def test_stations_match_the_research_md_formulas(dataset):
    """research.md 3.3 with I = 90 + dip, typed out here independently of the package."""
    survey = dataset.csv("survey")
    stations = dataset.csv("truth_stations")
    collar = dataset.csv("collar").set_index("hole_id")
    worst = 0.0
    for hole_id, s in survey.groupby("hole_id", sort=False):
        inc = np.radians(90.0 + s.dip.to_numpy())
        azi = np.radians(s.azimuth.to_numpy())
        md = s.depth.to_numpy()
        pos = [collar.loc[hole_id, ["x", "y", "z"]].to_numpy(dtype=float)]
        for i in range(len(md) - 1):
            i1, i2, a1, a2 = inc[i], inc[i + 1], azi[i], azi[i + 1]
            cos_beta = np.cos(i2 - i1) - np.sin(i1) * np.sin(i2) * (1.0 - np.cos(a2 - a1))
            beta = np.arccos(np.clip(cos_beta, -1.0, 1.0))
            rf = 1.0 if beta < 1e-7 else 2.0 / beta * np.tan(beta / 2.0)
            half = 0.5 * (md[i + 1] - md[i])
            d_north = half * (np.sin(i1) * np.cos(a1) + np.sin(i2) * np.cos(a2)) * rf
            d_east = half * (np.sin(i1) * np.sin(a1) + np.sin(i2) * np.sin(a2)) * rf
            d_up = -half * (np.cos(i1) + np.cos(i2)) * rf
            pos.append(pos[-1] + [d_east, d_north, d_up])
        truth = stations[stations.hole_id == hole_id][["x", "y", "z"]].to_numpy()
        worst = max(worst, np.abs(np.array(pos) - truth).max())
    assert worst < 5e-6  # acos loses ~1e-8 rad on small doglegs; files hold 1e-6 m


def test_positions_between_stations_match_numerical_integration(dataset):
    """Integrate the slerped direction numerically and compare with the closed-form arc."""
    hole = max(dataset.deposit.holes, key=lambda h: h.max_depth)
    t = direction(hole.dip, hole.azimuth)
    rng = np.random.default_rng(3)
    for depth in rng.uniform(0.0, hole.depth[-1], 12):
        position = hole.collar.copy()
        for i in range(len(hole.depth) - 1):
            top, bottom = hole.depth[i], min(hole.depth[i + 1], depth)
            if bottom <= top:
                break
            beta = np.arccos(np.clip(np.dot(t[i], t[i + 1]), -1, 1))
            n = 2000
            f = (top + (np.arange(n) + 0.5) * (bottom - top) / n - top) / (
                hole.depth[i + 1] - top
            )
            if beta < 1e-9:
                tangents = np.repeat(t[i][None], n, axis=0)
            else:
                tangents = (
                    np.sin((1 - f) * beta)[:, None] * t[i] + np.sin(f * beta)[:, None] * t[i + 1]
                ) / np.sin(beta)
            position = position + tangents.sum(axis=0) * (bottom - top) / n
        assert np.abs(hole.path.xyz(depth)[0] - position).max() < 1e-6


def test_truth_xyz_is_not_in_the_input_tables(dataset):
    for name in ("assay", "geology", "survey"):
        assert not {"x", "y", "z"} & set(pd.read_csv(dataset.path / f"{name}.csv").columns)
