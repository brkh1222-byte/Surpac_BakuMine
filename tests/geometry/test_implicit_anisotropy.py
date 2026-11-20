"""Automatic anisotropy of an implicit solid must follow the body, not the drilling.

The case that failed on a real project: a tabular body drilled by parallel inclined holes.
Each intercept is a line of sample points down the hole, so a principal-component analysis
of the sample points (the old ``"auto"``) sees the hole direction as an axis of the body.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geometry.implicit import (
    OUTSIDE,
    estimate_anisotropy,
    estimate_intercept_anisotropy,
    extract_constraints,
    implicit_solid,
)
from bakumine.core.geometry.implicit_request import anisotropy_from_spec, run_implicit

DIP = 70.0  # the body dips 70 degrees towards east (+x); strike is north-south
HOLE_DIP = -60.0  # holes drilled towards west, perpendicular to strike
SPACING = 50.0
NORMAL = np.array([np.sin(np.radians(DIP)), 0.0, np.cos(np.radians(DIP))])
STRIKE = np.array([0.0, 1.0, 0.0])
HOLE = np.array([-np.cos(np.radians(60.0)), 0.0, -np.sin(np.radians(60.0))])


def angle(a, b) -> float:
    c = abs(float(np.dot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def tabular_drilling(thickness: float, nx: int = 3, ny: int = 3, seed: int = 5):
    """Parallel inclined holes on a 50 m grid through a planar body of ``thickness`` (true).

    The mid-plane passes through (0, 0, -150). The surface is rough (+-4 m) and the
    thickness varies by +-20 % from hole to hole, as in real logging.
    """
    rng = np.random.default_rng(seed)
    centre = np.array([0.0, 0.0, -150.0])
    collars, logs = [], []
    for i in range(nx):
        for j in range(ny):
            collar = np.array(
                [(i - (nx - 1) / 2) * SPACING + 110.0, (j - (ny - 1) / 2) * SPACING, 0.0]
            )
            mid = ((centre - collar) @ NORMAL) / (HOLE @ NORMAL) + rng.uniform(-4.0, 4.0)
            half = 0.5 * thickness * rng.uniform(0.8, 1.2) / abs(HOLE @ NORMAL)
            hole = f"T{i}{j}"
            collars.append((hole, *collar, 420.0))
            logs += [
                (hole, 0.0, mid - half, "WASTE"),
                (hole, mid - half, mid + half, "LODE"),
                (hole, mid + half, 420.0, "WASTE"),
            ]
    collar = pd.DataFrame(collars, columns=["hole_id", "x", "y", "z", "max_depth"])
    survey = pd.DataFrame(
        {"hole_id": collar["hole_id"], "depth": 0.0, "dip": HOLE_DIP, "azimuth": 270.0}
    )
    geology = pd.DataFrame(logs, columns=["hole_id", "depth_from", "depth_to", "rock"])
    return {"collar": collar, "survey": survey, "geology": geology}


def both_estimates(tables):
    traces = desurvey(tables["collar"], tables["survey"])
    constraints = extract_constraints(tables["geology"], "rock", "LODE", traces)
    old = estimate_anisotropy(constraints.xyz[constraints.kind != OUTSIDE])
    runs = constraints.intercepts.query("state > 0")
    ids = runs["hole_id"].to_numpy(dtype=object)
    top, _ = traces.xyz_at(ids, runs["depth_from"].to_numpy())
    bottom, _ = traces.xyz_at(ids, runs["depth_to"].to_numpy())
    new, info = estimate_intercept_anisotropy(top, bottom)
    return old, new, info


def test_tabular_body_auto_anisotropy_follows_the_body_not_the_drilling():
    print("\nthickness  old: normal off / major vs holes   new: normal off / major vs holes")
    rows = {}
    for thickness in (10.0, 60.0, 100.0, 120.0):
        old, new, info = both_estimates(tabular_drilling(thickness))
        rows[thickness] = (
            angle(old.minor_axis, NORMAL),
            angle(old.major_axis, HOLE),
            angle(new.minor_axis, NORMAL),
            info["hole_angle_deg"],
        )
        print(
            f"{thickness:7.0f} m  {rows[thickness][0]:5.1f} deg / {rows[thickness][1]:5.1f} deg"
            f"          {rows[thickness][2]:5.1f} deg / {rows[thickness][3]:5.1f} deg"
        )
        # the acceptance criterion: the plane normal within 15 degrees of the truth
        assert rows[thickness][2] < 15.0
        # and the major axis lies in the plane of the body
        assert angle(new.major_axis, NORMAL) > 75.0
    # the sample-point estimate fails as the intercepts get long against the drilled area:
    # its normal tilts towards the holes and its major axis ends up along the drilling
    assert rows[100.0][0] > 15.0
    assert rows[120.0][0] > 25.0 and rows[120.0][1] < 25.0
    assert rows[120.0][3] > 40.0  # the intercept estimate stays 50 degrees from the holes


def test_implicit_solid_auto_uses_the_intercept_estimate():
    tables = tabular_drilling(100.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = implicit_solid(tables, "geology", "rock", "LODE", resolution=10.0)
    d = result.diagnostics
    assert d["anisotropy_source"] == "auto"
    assert d["anisotropy_method"] == "intercept midpoints"
    assert angle(result.anisotropy.minor_axis, NORMAL) < 15.0
    assert d["anisotropy_hole_angle_deg"] > 40.0
    assert d["mesh_valid"]
    # the solid is a slab along the true plane: its thinnest direction is the true normal
    v = result.mesh.vertices - result.mesh.vertices.mean(axis=0)
    thinnest = np.linalg.svd(v, full_matrices=False)[2][2]
    assert angle(thinnest, NORMAL) < 15.0
    given = implicit_solid(
        tables, "geology", "rock", "LODE", resolution=10.0,
        anisotropy=Ellipsoid.from_axes(STRIKE, np.cross(STRIKE, NORMAL), NORMAL, 1.0, 1.0, 3.0),
    )  # fmt: skip
    assert given.diagnostics["anisotropy_method"] == "given"
    assert given.diagnostics["anisotropy_warnings"] == []


def test_ratios_of_an_evenly_drilled_ellipsoid():
    """Vertical holes on a fine grid through an ellipsoid: the ratios are those of the half-axes."""
    a, b, c = 200.0, 100.0, 20.0
    xs, ys = np.meshgrid(np.arange(-195.0, 200.0, 10.0), np.arange(-95.0, 100.0, 10.0))
    inside = (xs / a) ** 2 + (ys / b) ** 2 < 1.0
    x, y = xs[inside], ys[inside]
    half = c * np.sqrt(1.0 - (x / a) ** 2 - (y / b) ** 2)
    top = np.column_stack([x, y, half])
    bottom = np.column_stack([x, y, -half])
    ellipsoid, info = estimate_intercept_anisotropy(top, bottom)
    assert angle(ellipsoid.major_axis, [1, 0, 0]) < 0.5
    assert angle(ellipsoid.minor_axis, [0, 0, 1]) < 1e-6
    assert ellipsoid.semi_major_ratio == pytest.approx(a / b, rel=0.03)
    assert ellipsoid.minor_ratio == pytest.approx(a / c, rel=0.03)
    assert ellipsoid.major_radius == pytest.approx(a, rel=0.03)
    assert info["warnings"] == [] and info["n_intercepts"] == int(inside.sum())


def test_warnings_when_the_estimate_follows_the_drilling():
    # intercept midpoints strung out along the holes: holes drilled down the dip of a body,
    # two fences 30 m apart; the "plane" of the midpoints contains the hole direction
    direction = np.array([0.5, 0.0, -np.sqrt(0.75)])
    mids = np.array([s * direction + [0.0, y, 0.0] for s in (50, 120, 200, 260) for y in (0, 30)])
    top, bottom = mids - 4.0 * direction, mids + 4.0 * direction
    ellipsoid, info = estimate_intercept_anisotropy(top, bottom)
    assert info["hole_angle_deg"] < 20.0
    assert any("mean hole direction" in w for w in info["warnings"])
    # long intercepts on a small area: poorly determined
    _, _, info = both_estimates(tabular_drilling(150.0, nx=2, ny=3))
    assert any("thicker than the drilled area" in w for w in info["warnings"])
    # no plane at all: one fence with the midpoints on a line
    line = np.array([[x, 0.0, -x] for x in (0.0, 50.0, 100.0, 150.0)])
    with pytest.raises(ValueError, match="on one line"):
        estimate_intercept_anisotropy(line + [0, 0, 5.0], line - [0, 0, 5.0])
    with pytest.raises(ValueError, match="at least 4"):
        estimate_intercept_anisotropy(line[:3] + [0, 0, 5.0], line[:3] - [0, 0, 5.0])


def test_auto_falls_back_to_isotropic_with_a_warning():
    tables = tabular_drilling(20.0, nx=3, ny=1)  # one fence: three intercepts
    with pytest.warns(UserWarning, match="no automatic anisotropy"):
        result = implicit_solid(tables, "geology", "rock", "LODE", resolution=5.0)
    assert result.anisotropy is None
    assert result.anisotropy_source == "none (auto failed)"
    assert result.diagnostics["anisotropy_warnings"]


def test_request_anisotropy_and_summary():
    plane = anisotropy_from_spec(
        {"mode": "plane", "dip": DIP, "dip_azimuth": 90.0, "ratios": "4:3:1"}
    )
    assert isinstance(plane, Ellipsoid)
    assert angle(plane.minor_axis, NORMAL) < 1e-6 and angle(plane.major_axis, STRIKE) < 1e-6
    assert plane.semi_major_ratio == pytest.approx(4 / 3) and plane.minor_ratio == 4.0
    angles = anisotropy_from_spec({"mode": "angles", "angles": [0, 0, -DIP], "ratios": [4, 3, 1]})
    np.testing.assert_allclose(angles.rotation, plane.rotation, atol=1e-9)
    assert anisotropy_from_spec("auto") == "auto" and anisotropy_from_spec({"mode": "none"}) is None
    for bad in (
        None,
        {},
        {"mode": "plane", "dip": 70, "dip_azimuth": 90},
        {"mode": "plane", "dip": 70, "ratios": "1:1:1"},
        {"mode": "angles", "angles": [1, 2], "ratios": "1:1:1"},
        {"mode": "pca"},
    ):
        with pytest.raises(ValueError):
            anisotropy_from_spec(bad)

    tables = tabular_drilling(30.0)

    class Database:
        intervals = {"geology": tables["geology"]}

        @staticmethod
        def desurvey():
            return desurvey(tables["collar"], tables["survey"])

    request = {
        "table": "geology", "column": "rock", "codes": ["LODE"], "resolution": 6.0,
        "anisotropy": {"mode": "plane", "dip": DIP, "dip_azimuth": 90.0, "ratios": "4:3:1"},
    }  # fmt: skip
    result, summary = run_implicit(Database, request)
    assert summary["valid"] and summary["volume"] == pytest.approx(result.mesh.volume)
    assert summary["anisotropy"]["source"] == "user"
    assert summary["anisotropy"]["plane"] == {"dip": DIP, "dip_azimuth": 90.0}
    assert summary["anisotropy"]["major"]["plunge"] == 0.0
    assert summary["anisotropy"]["ratios"] == [1.0, 0.75, 0.25]
    auto, auto_summary = run_implicit(Database, {**request, "anisotropy": "auto"})
    assert auto_summary["anisotropy"]["source"] == "auto"
    assert auto_summary["anisotropy"]["plane"]["dip"] == pytest.approx(DIP, abs=15.0)
    for bad in ({"column": None}, {"codes": ["NOPE"]}, {"table": "assay"}, {"column": "au"},
                {"resolution": "fine"}, {"anisotropy": None}):  # fmt: skip
        with pytest.raises(ValueError):
            run_implicit(Database, {**request, **bad})
