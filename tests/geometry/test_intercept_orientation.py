"""Automatic orientation of stacked lenses: indicator correlation between neighbouring holes.

The case that failed on a real project: many sub-parallel lenses drilled on a grid. The
intercept midpoints fill the drilled volume, so their principal components describe the
drilling pattern (here the midpoint pole is 45-85 degrees off). Truth is known here by
construction: the lenses dip 69 degrees towards 216.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from stacked_lenses import DIP, POLE, stacked_lenses
from test_implicit_anisotropy import NORMAL, tabular_drilling

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geometry.implicit import (
    UndeterminedOrientation,
    estimate_intercept_anisotropy,
    estimate_stacked_anisotropy,
    extract_constraints,
    implicit_solid,
)
from bakumine.core.geometry.implicit_request import run_implicit
from bakumine.core.geometry.intercept_orientation import (
    MIN_CONFIDENCE,
    MIN_CORRELATION,
    estimate_plane_from_holes,
    pole_of_plane,
)


def angle(a, b) -> float:
    c = abs(float(np.dot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def estimate(tables, **options):
    traces = desurvey(tables["collar"], tables["survey"])
    constraints = extract_constraints(tables["geology"], "rock", "LODE", traces)
    return estimate_plane_from_holes(constraints.intercepts, traces, **options), constraints, traces


def midpoint_error(constraints, traces, truth) -> float:
    runs = constraints.intercepts.query("state > 0")
    ids = runs["hole_id"].to_numpy(dtype=object)
    top, _ = traces.xyz_at(ids, runs["depth_from"].to_numpy())
    bottom, _ = traces.xyz_at(ids, runs["depth_to"].to_numpy())
    return angle(estimate_intercept_anisotropy(top, bottom)[0].minor_axis, truth)


def test_pole_of_plane():
    np.testing.assert_allclose(pole_of_plane(0.0, 0.0), [0, 0, 1], atol=1e-12)
    np.testing.assert_allclose(pole_of_plane(90.0, 90.0), [1, 0, 0], atol=1e-12)
    # 69 -> 216: the pole plunges 21 degrees towards 036 (taken at its lower end)
    down = -pole_of_plane(69.0, 216.0)
    assert np.degrees(np.arcsin(-down[2])) == pytest.approx(21.0)
    assert np.degrees(np.arctan2(down[0], down[1])) % 360 == pytest.approx(36.0)


def test_one_tabular_body():
    """(a) a single thin body: the pair estimate agrees with the truth (and the midpoints)."""
    result, constraints, traces = estimate(tabular_drilling(10.0))
    print(f"\ntabular: {angle(result.pole, NORMAL):.1f} deg off, "
          f"correlation {result.correlation:.2f}, confidence {result.confidence:.2f}")  # fmt: skip
    assert angle(result.pole, NORMAL) < 15.0
    assert result.determined and result.confidence > 0.5
    assert midpoint_error(constraints, traces, NORMAL) < 15.0


def test_stacked_lenses():
    """(b) five parallel lenses: the pair estimate finds the plane, the midpoints do not."""
    result, constraints, traces = estimate(stacked_lenses())
    error = angle(result.pole, POLE)
    print(f"\nstacked: {result.dip:.1f} -> {result.dip_azimuth:.1f}, {error:.1f} deg off, "
          f"correlation {result.correlation:.2f}, confidence {result.confidence:.2f}")  # fmt: skip
    assert error < 15.0
    assert result.dip == pytest.approx(DIP, abs=10.0)
    assert result.determined
    assert result.correlation > MIN_CORRELATION and result.confidence > MIN_CONFIDENCE
    assert result.warnings == [] and result.reason == ""
    assert result.thickness == pytest.approx(20.0, rel=0.15)  # mean of 10..30 m
    assert midpoint_error(constraints, traces, POLE) > 30.0  # what "auto" did before


@pytest.mark.parametrize("seed", [0, 1, 2, 3, 4])
def test_stacked_lenses_with_missing_intercepts_and_noise(seed):
    """(c) 20 % of the intercepts missing, contacts moved by +-3 m, thickness +-20 %."""
    result, _, _ = estimate(stacked_lenses(missing=0.2, noise=3.0, seed=seed))
    print(f"\nseed {seed}: {result.dip:.1f} -> {result.dip_azimuth:.1f}, "
          f"{angle(result.pole, POLE):.1f} deg off, correlation {result.correlation:.2f}, "
          f"confidence {result.confidence:.2f}")  # fmt: skip
    assert angle(result.pole, POLE) < 15.0
    assert result.determined
    assert result.to_dict()["confidence"] == result.confidence >= MIN_CONFIDENCE


def test_not_determined_cases_are_refused():
    # one section of holes: every plane with the right apparent dip matches
    fence, _, _ = estimate(stacked_lenses(n_sections=1))
    assert not fence.determined and fence.warnings
    assert "one line" in fence.reason or "equally well" in fence.reason
    # intercepts without any continuity between holes
    tables = stacked_lenses()
    rng = np.random.default_rng(0)
    rows = []
    for hole in tables["collar"]["hole_id"]:
        depth = 0.0
        for a, b in np.sort(rng.uniform(5.0, 315.0, size=8)).reshape(-1, 2):
            rows += [(hole, depth, a, "WASTE"), (hole, a, b, "LODE")]
            depth = b
        rows.append((hole, depth, 320.0, "WASTE"))
    tables["geology"] = pd.DataFrame(rows, columns=tables["geology"].columns)
    noise, constraints, traces = estimate(tables)
    assert not noise.determined and noise.correlation < MIN_CORRELATION
    assert "do not line up" in noise.reason
    with pytest.raises(UndeterminedOrientation, match="do not line up") as caught:
        estimate_stacked_anisotropy(constraints.intercepts, traces)
    assert caught.value.info["determined"] is False
    # two holes, and no intercept at all
    two = stacked_lenses(n_along=2, n_sections=1)
    assert not estimate(two)[0].determined
    runs = constraints.intercepts.assign(state=-1.0)
    assert not estimate_plane_from_holes(runs, traces).determined


def test_implicit_solid_auto_on_stacked_lenses():
    tables = stacked_lenses(missing=0.2, noise=3.0, seed=1)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a determined orientation gives no warning
        result = implicit_solid(tables, "geology", "rock", "LODE", resolution=8.0, validate=False)
    d = result.diagnostics
    assert d["anisotropy_source"] == "auto"
    assert d["anisotropy_method"] == "indicator correlation between holes"
    assert angle(result.anisotropy.minor_axis, POLE) < 15.0
    assert d["anisotropy_confidence"] >= MIN_CONFIDENCE and d["anisotropy_candidate"] is None
    # in-plane continuity is not estimated: major axis along strike, equal extent down dip
    assert abs(result.anisotropy.major_axis[2]) < 1e-9
    assert result.anisotropy.semi_major_ratio == 1.0 and result.anisotropy.minor_ratio > 2.0


def test_implicit_solid_auto_refuses_an_undetermined_orientation():
    tables = stacked_lenses(n_sections=1)

    class Database:
        intervals = {"geology": tables["geology"]}

        @staticmethod
        def desurvey():
            return desurvey(tables["collar"], tables["survey"])

    request = {"table": "geology", "column": "rock", "codes": ["LODE"], "resolution": 10.0,
               "anisotropy": "auto"}  # fmt: skip
    result, summary = run_implicit(Database, request)
    assert result.anisotropy is None
    assert summary["anisotropy"]["source"] == "none (auto failed)"
    assert "plane" not in summary["anisotropy"]  # nothing was chosen
    assert any("give the orientation" in w for w in summary["warnings"])
