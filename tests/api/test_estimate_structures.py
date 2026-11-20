"""Nested variogram structures with their own ranges and orientation, and dip / dip azimuth
/ pitch input: the plan the endpoint builds equals a plan built in Python.

The endpoint's plan is read from the job log (``replay.plan``, what was actually run) and
compared with ``EstimationPlan(...)`` made from ``VariogramModel`` / ``Structure``; the
estimated grades are compared with the Python plan run directly and with ``bakumine
estimate`` on the plan saved from Python.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.operations import OperationError, build_plan
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import set_scene_source
from bakumine.cli import main
from bakumine.core import joblog
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.estimation.block_estimation import (
    EstimationDomain,
    EstimationPass,
    EstimationPlan,
)
from bakumine.core.estimation.orientation import angles_from_plane, plane_axes
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel
from bakumine.core.project import Project

OPS = "/api/v1/ops"

#: the dialog's request: search ellipsoid as a plane, three structures with their own
#: ranges; one follows the search orientation, one has angles, one a plane with pitch
SPEC = {
    "attribute": "au",
    "method": "ok",
    "ellipsoid": {"radii": [150, 100, 40], "plane": {"dip": 69, "dip_azimuth": 216}},
    "variogram": {
        "nugget": 0.1,
        "structures": [
            {"type": "spherical", "sill": 0.4, "ranges": [30, 20, 8]},
            {"type": "exponential", "sill": 0.3, "ranges": [90, 60, 15], "angles": [20, -5, 30]},
            {"type": "gaussian", "sill": 0.2, "ranges": [200, 120, 40],
             "plane": {"dip": 60, "dip_azimuth": 200, "pitch": 35}},
        ],
    },
    "discretisation": [2, 2, 2],
    "passes": [
        {"name": "near", "radius_factor": 0.5, "min_samples": 4, "max_samples": 12,
         "max_per_hole": 3},
        {"name": "far", "radius_factor": 1.0, "min_samples": 1, "max_samples": 12},
    ],
}  # fmt: skip


def python_plan() -> EstimationPlan:
    """The same plan written with the estimation classes, angles worked out by hand."""
    search_angles = (126.0, 0.0, -69.0)  # plane 69 -> 216: strike 126, horizontal, dip -69
    ellipsoid = Ellipsoid.from_radii((150.0, 100.0, 40.0), search_angles)
    variogram = VariogramModel(
        0.1,
        (
            Structure("spherical", 0.4, (30.0, 20.0, 8.0), search_angles),
            Structure("exponential", 0.3, (90.0, 60.0, 15.0), (20.0, -5.0, 30.0)),
            Structure("gaussian", 0.2, (200.0, 120.0, 40.0), angles_from_plane(60.0, 200.0, 35.0)),
        ),
    )
    passes = [
        EstimationPass("ok", ellipsoid, SearchParams(max_samples=12, min_samples=4, max_per_hole=3),
                       radius_factor=0.5, variogram=variogram, discretisation=(2, 2, 2),
                       name="near"),
        EstimationPass("ok", ellipsoid, SearchParams(max_samples=12, min_samples=1),
                       radius_factor=1.0, variogram=variogram, discretisation=(2, 2, 2),
                       name="far"),
    ]  # fmt: skip
    return EstimationPlan("au", [EstimationDomain("all", passes)])


def test_build_plan_equals_the_python_plan():
    built = build_plan(SPEC)
    assert built.to_dict() == python_plan().to_dict()
    structures = built.domains[0].passes[0].variogram.structures
    assert [s.type for s in structures] == ["spherical", "exponential", "gaussian"]
    assert [s.ranges for s in structures] == [(30, 20, 8), (90, 60, 15), (200, 120, 40)]
    # the third structure's axes are those of its plane: major pitching 35 from strike
    major, semi, minor = plane_axes(60.0, 200.0, 35.0)
    rotation = Ellipsoid.from_radii(structures[2].ranges, structures[2].angles).rotation
    np.testing.assert_allclose(rotation, np.column_stack([semi, major, minor]), atol=1e-8)
    # the search ellipsoid: major along strike 126 / 306, minor = pole of the plane
    e = built.domains[0].passes[0].ellipsoid
    np.testing.assert_allclose(np.abs(e.major_axis @ plane_axes(69.0, 216.0)[0]), 1.0)
    np.testing.assert_allclose(np.abs(e.minor_axis @ plane_axes(69.0, 216.0)[2]), 1.0)


def test_old_request_shape_still_works():
    """Structures without ranges / orientation take both from the search ellipsoid."""
    old = {
        "attribute": "au", "method": "ok",
        "ellipsoid": {"radii": [120.0, 70.0, 15.0], "angles": [35.0, -10.0, 50.0]},
        "variogram": {"nugget": 0.15, "structures": [{"type": "spherical", "sill": 0.85}]},
    }  # fmt: skip
    plan = build_plan(old)
    ellipsoid = Ellipsoid.from_radii((120.0, 70.0, 15.0), (35.0, -10.0, 50.0))
    model = VariogramModel(
        0.15, (Structure("spherical", 0.85, (120.0, 70.0, 15.0), (35.0, -10.0, 50.0)),)
    )
    expected = EstimationPlan(
        "au", [EstimationDomain("all", [EstimationPass("ok", ellipsoid, SearchParams(
            max_samples=16, min_samples=1), variogram=model, name="pass 1")])]
    )  # fmt: skip
    assert plan.to_dict() == expected.to_dict()
    # a plane for the search ellipsoid alone moves the structures with it
    planar = build_plan(
        {**old, "ellipsoid": {"radii": [120, 70, 15], "plane": {"dip": 69, "dip_azimuth": 216}}}
    )
    assert planar.domains[0].passes[0].variogram.structures[0].angles == (126.0, 0.0, -69.0)


def test_invalid_structures_are_reported():
    def with_structures(structures, **ellipsoid):
        return {**SPEC, "ellipsoid": {**SPEC["ellipsoid"], **ellipsoid},
                "variogram": {"nugget": 0.0, "structures": structures}}  # fmt: skip

    one = {"type": "spherical", "sill": 1.0}
    for bad in (
        with_structures([one] * 4),  # more than three
        with_structures([{**one, "ranges": [10, 5]}]),
        with_structures([{**one, "ranges": [10, 5, 0]}]),
        with_structures([{**one, "type": "cubic"}]),
        with_structures([{**one, "plane": {"dip": 30}}]),
        with_structures([{**one, "plane": {"dip": 30, "dip_azimuth": 10}, "angles": [1, 2, 3]}]),
        with_structures([one], angles=[1, 2, 3]),  # ellipsoid: plane and angles
    ):
        with pytest.raises(OperationError):
            build_plan(bad)


# ------------------------------------------------------------------ through the endpoint


def samples() -> pd.DataFrame:
    rng = np.random.default_rng(4)
    n = 240
    xyz = rng.uniform([0.0, 0.0, 0.0], [200.0, 200.0, 60.0], size=(n, 3))
    au = (
        1.0
        + 0.5 * np.sin(xyz[:, 0] / 30.0)
        + 0.3 * np.cos(xyz[:, 1] / 45.0)
        + rng.normal(0, 0.1, n)
    )
    return pd.DataFrame(
        {"hole_id": [f"H{i // 8:02d}" for i in range(n)], "x": xyz[:, 0], "y": xyz[:, 1],
         "z": xyz[:, 2], "length": 2.0, "au": au, "au_coverage": 1.0}
    )  # fmt: skip


@pytest.fixture()
def project(tmp_path):
    folder = tmp_path / "project"
    proj = Project.create(folder, "structures")
    samples().to_parquet(tmp_path / "comps.parquet")
    proj.add("composite", tmp_path / "comps.parquet", "Composites", "comps")
    definition = BlockModelDefinition((0.0, 0.0, 0.0), (20.0, 20.0, 10.0), (10, 10, 6))
    save_model(BlockModel(definition), tmp_path / "model")
    proj.add("model", tmp_path / "model", "Model", "model")
    set_scene_source(ProjectSceneSource(folder))
    yield TestClient(app), folder, definition
    set_scene_source(None)


def test_endpoint_plan_and_grades_equal_python_and_cli(project, tmp_path, capsys):
    client, folder, definition = project
    started = client.post(f"{OPS}/estimate", json={"model": "model", "composites": "comps",
                                                   "plan": SPEC})  # fmt: skip
    assert started.status_code == 200, started.text
    end = time.time() + 300
    while True:
        job = client.get(f"{OPS}/jobs/{started.json()['id']}").json()
        if job["status"] not in ("queued", "running"):
            break
        assert time.time() < end
        time.sleep(0.1)
    assert job["status"] == "done", job
    assert job["summary"]["n_estimated"] == definition.n_parents

    plan = python_plan()
    entry = next(e for e in joblog.jobs(folder) if e["operation"] == "estimate")
    assert entry["replay"]["plan"] == plan.to_dict()
    assert EstimationPlan.from_dict(entry["replay"]["plan"]).to_dict() == plan.to_dict()

    api_model = load_model(Project.open(folder).path_of("model"))
    direct = BlockModel(definition)
    plan.run(direct, samples())
    np.testing.assert_array_equal(api_model.get("au"), direct.get("au"))
    assert np.isfinite(direct.get("au")).all()

    # the CLI on the plan saved from Python
    save_model(BlockModel(definition), tmp_path / "cli_model")
    plan.save(tmp_path / "plan.json")
    assert main(["estimate", str(tmp_path / "cli_model"), "--composites",
                 str(Path(folder) / "composites" / "comps.parquet"),
                 "--plan", str(tmp_path / "plan.json")]) == 0  # fmt: skip
    capsys.readouterr()
    cli_model = load_model(tmp_path / "cli_model")
    np.testing.assert_array_equal(cli_model.get("au"), api_model.get("au"))

    # the nested model matters: one structure with the search ranges gives other grades
    single = build_plan({**SPEC, "variogram": {"nugget": 0.1, "structures": [
        {"type": "spherical", "sill": 0.9}]}})  # fmt: skip
    other = BlockModel(definition)
    single.run(other, samples())
    assert np.abs(other.get("au") - direct.get("au")).max() > 1e-3

    bad = client.post(
        f"{OPS}/estimate",
        json={
            "model": "model",
            "composites": "comps",
            "plan": {
                **SPEC,
                "variogram": {"nugget": 0, "structures": [{"type": "spherical", "sill": 1}] * 4},
            },
        },
    )
    assert bad.status_code == 400 and "at most 3" in bad.json()["detail"]
