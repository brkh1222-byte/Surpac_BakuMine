"""``bakumine estimate-plan`` writes the plan a user would build in Python, and it runs."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.commands.estimate_plan import (
    build_plan,
    composite_variance,
    parse_pass,
    parse_sample_filter,
)
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import AttributeConstraint
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model
from bakumine.core.estimation.block_estimation import (
    EstimationDomain,
    EstimationPass,
    EstimationPlan,
)
from bakumine.core.estimation.orientation import angles_from_plane
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.variogram import Structure, VariogramModel

DIP, DIP_AZIMUTH, RANGES, TOP_CUT = 69.0, 216.0, (100.0, 55.0, 10.0), 31.0


@pytest.fixture(scope="module")
def composites(tmp_path_factory):
    """600 composites in a 120 x 120 x 60 m box; a third are waste; a few very high grades."""
    rng = np.random.default_rng(7)
    n = 600
    table = pd.DataFrame(
        {
            "hole_id": [f"H{i % 30:02d}" for i in range(n)],
            "x": rng.uniform(0.0, 120.0, n),
            "y": rng.uniform(0.0, 120.0, n),
            "z": rng.uniform(0.0, 60.0, n),
            "au": rng.lognormal(0.0, 1.2, n),
            "domain": np.where(rng.random(n) < 0.33, "WASTE", "MIN"),
            "zone": rng.integers(1, 4, n),
        }
    )
    table.loc[::50, "au"] = 80.0  # above the top cut
    table.loc[5, "au"] = np.nan
    path = tmp_path_factory.mktemp("plan") / "composites.parquet"
    table.to_parquet(path)
    return path, table


def flags(path, out, *extra):
    return [
        "estimate-plan", "--out", str(out), "--attribute", "au", "--method", "ok",
        "--domain", "ORE", "--constraint", 'rock == "ORE"', "--sample-filter", "domain=MIN",
        "--dip", str(DIP), "--dip-azimuth", str(DIP_AZIMUTH), "--ranges", "100,55,10",
        "--composites", str(path), "--top-cut", str(TOP_CUT),
        "--nugget-fraction", "0.25", "--sill-fraction", "0.75", "--structure", "spherical",
        "--pass", "0.8:4:10:3", "--pass", "1.2:4:10:3", "--pass", "1.6:4:10:3",
        "--pass", "4:1:10", "--discretisation", "3,3,3", *extra,
    ]  # fmt: skip


def python_plan(table) -> EstimationPlan:
    """The same plan written with the classes, the way a script would."""
    mins = table[(table["domain"] == "MIN") & table["au"].notna()]
    variance = float(np.var(np.minimum(mins["au"].to_numpy(), TOP_CUT), ddof=1))
    angles = angles_from_plane(DIP, DIP_AZIMUTH)
    ellipsoid = Ellipsoid.from_radii(RANGES, angles)
    variogram = VariogramModel(
        0.25 * variance, (Structure("spherical", 0.75 * variance, RANGES, angles),)
    )
    passes = [
        EstimationPass(
            "ok", ellipsoid, SearchParams(max_samples=10, min_samples=4, max_per_hole=3),
            radius_factor=factor, variogram=variogram, discretisation=(3, 3, 3), name=f"p{i}",
        )
        for i, factor in enumerate((0.8, 1.2, 1.6), start=1)
    ]  # fmt: skip
    passes.append(
        EstimationPass(
            "ok", ellipsoid, SearchParams(max_samples=10, min_samples=1),
            radius_factor=4.0, variogram=variogram, discretisation=(3, 3, 3), name="p4",
        )
    )  # fmt: skip
    domain = EstimationDomain(
        "ORE", passes, block_constraint=AttributeConstraint('rock == "ORE"'),
        sample_filter=("domain", "MIN"), capping=TOP_CUT,
    )  # fmt: skip
    return EstimationPlan("au", [domain], value_column="au")


def test_plan_from_flags_equals_the_plan_built_in_python(composites, tmp_path, capsys):
    path, table = composites
    out = tmp_path / "plan.json"
    assert main(flags(path, out)) == 0
    text = capsys.readouterr().out
    expected = python_plan(table)
    assert json.loads(out.read_text()) == expected.to_dict()
    assert EstimationPlan.load(out).to_dict() == expected.to_dict()
    # what was derived is printed: the variance after the top cut, of the MIN composites
    mins = table[(table["domain"] == "MIN") & table["au"].notna()]
    variance = float(np.minimum(mins["au"], TOP_CUT).var())  # pandas: n - 1
    assert f"composite variance {variance:.6g} from {len(mins)} composites" in text
    assert f"nugget {0.25 * variance:.6g}, sill(s) {0.75 * variance:.6g}" in text
    # the orientation is the plane: major axis along strike (126), minor axis the pole
    ellipsoid = expected.domains[0].passes[0].ellipsoid
    az, dip = np.radians(DIP_AZIMUTH), np.radians(DIP)
    strike = np.array([np.sin(az - np.pi / 2), np.cos(az - np.pi / 2), 0.0])
    down_dip = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), -np.sin(dip)])
    np.testing.assert_allclose(ellipsoid.major_axis, strike, atol=1e-9)
    np.testing.assert_allclose(ellipsoid.semi_major_axis, down_dip, atol=1e-9)
    assert ellipsoid.radii == pytest.approx(RANGES)


def test_variance_helper_and_parsers(composites):
    _, table = composites
    variance, n = composite_variance(table, "au", {"column": "domain", "equals": "MIN"}, TOP_CUT)
    mins = table[table["domain"] == "MIN"]["au"].dropna()
    assert n == len(mins) and variance == pytest.approx(np.minimum(mins, TOP_CUT).var())
    assert composite_variance(table, "au")[0] == pytest.approx(table["au"].var())
    with pytest.raises(ValueError, match="no column"):
        composite_variance(table, "cu")
    with pytest.raises(ValueError, match="fewer than two"):
        composite_variance(table, "au", {"column": "domain", "equals": "NONE"})
    assert parse_sample_filter("domain=MIN") == {"column": "domain", "equals": "MIN"}
    assert parse_sample_filter(" zone = 1, 2 ") == {"column": "zone", "in": [1, 2]}
    assert parse_sample_filter('code="1"') == {"column": "code", "equals": "1"}
    assert parse_pass("0.8:4:10:3") == {
        "factor": 0.8, "min_samples": 4, "max_samples": 10, "max_per_hole": 3,
    }  # fmt: skip
    assert parse_pass("4:1:10") == {"factor": 4.0, "min_samples": 1, "max_samples": 10}
    for bad in ("domain", "=MIN", "domain="):
        with pytest.raises(ValueError):
            parse_sample_filter(bad)
    for bad in ("1:2", "a:1:2", "1:2:3:4:5", "1:2.5:3"):
        with pytest.raises(ValueError):
            parse_pass(bad)


def test_absolute_sills_nested_structures_angles_and_other_methods(tmp_path):
    out = tmp_path / "plan.json"
    argv = [
        "estimate-plan", "--out", str(out), "--attribute", "cu", "--value-column", "cu_pct",
        "--method", "sk", "--mean", "0.4", "--angles", "30,10,-50", "--ranges", "80,40,20",
        "--search-ranges", "120,60,30", "--sample-filter", "zone=1,2",
        "--structure", "spherical", "--structure", "exponential",
        "--structure-ranges", "20,10,5", "--structure-ranges", "80,40,20",
        "--nugget", "0.1", "--sill", "0.3,0.6", "--pass", "1:2:16",
    ]  # fmt: skip
    assert main(argv) == 0
    angles = (30.0, 10.0, -50.0)
    variogram = VariogramModel(
        0.1,
        (
            Structure("spherical", 0.3, (20.0, 10.0, 5.0), angles),
            Structure("exponential", 0.6, (80.0, 40.0, 20.0), angles),
        ),
    )
    expected = EstimationPlan(
        "cu",
        [
            EstimationDomain(
                "domain",
                [
                    EstimationPass(
                        "sk", Ellipsoid.from_radii((120.0, 60.0, 30.0), angles),
                        SearchParams(max_samples=16, min_samples=2), variogram=variogram,
                        mean=0.4, name="p1",
                    )
                ],
                sample_filter={"column": "zone", "in": [1, 2]},
            )
        ],
        value_column="cu_pct",
    )  # fmt: skip
    assert json.loads(out.read_text()) == expected.to_dict()
    # inverse distance: a power, no variogram; isotropic ranges need no orientation
    assert main(["estimate-plan", "--out", str(out), "--attribute", "au", "--method", "idw",
                 "--power", "3", "--ranges", "50,50,50", "--pass", "1:1:8:2",
                 "--pass", "2:1:8"]) == 0  # fmt: skip
    sphere = Ellipsoid.from_radii((50.0, 50.0, 50.0))
    idw = EstimationPlan(
        "au",
        [
            EstimationDomain(
                "domain",
                [
                    EstimationPass("idw", sphere, SearchParams(8, 1, max_per_hole=2), power=3.0,
                                   name="p1"),
                    EstimationPass("idw", sphere, SearchParams(8, 1), radius_factor=2.0,
                                   power=3.0, name="p2"),
                ],
            )
        ],
        value_column="au",
    )  # fmt: skip
    assert json.loads(out.read_text()) == idw.to_dict()
    plan, info = build_plan({"attribute": "au", "method": "nn", "ranges": "30,30,30",
                             "passes": ["1:1:1"]})  # fmt: skip
    assert plan.domains[0].passes[0].method == "nn" and info["variance"] is None


def test_errors_are_reported_and_nothing_is_written(composites, tmp_path, capsys):
    path, _ = composites
    out = tmp_path / "plan.json"
    base = ["estimate-plan", "--out", str(out), "--attribute", "au", "--ranges", "100,55,10"]
    plane = ["--dip", "69", "--dip-azimuth", "216"]
    sills = ["--nugget", "0.1", "--sill", "0.9"]
    for argv, message in (
        ([*base, *sills, "--pass", "1:1:8"], "orientation"),
        ([*base, *plane, "--pass", "1:1:8"], "nugget and the sill"),
        ([*base, *plane, *sills, "--nugget-fraction", "0.2", "--pass", "1:1:8"], "either"),
        ([*base, *plane, "--nugget-fraction", "0.2", "--sill-fraction", "0.8", "--pass",
          "1:1:8"], "need the composites"),
        ([*base, *plane, "--nugget", "0.1", "--sill", "0.5,0.4", "--pass", "1:1:8"], "sill"),
        ([*base, *plane, *sills, "--pass", "1:8"], "FACTOR:MIN:MAX"),
        ([*base, *plane, *sills, "--pass", "1:9:8"], "min_samples"),
        ([*base, "--dip", "69", *sills, "--pass", "1:1:8"], "both the dip"),
        ([*base, *plane, "--angles", "1,2,3", *sills, "--pass", "1:1:8"], "not both"),
        ([*base, *plane, *sills, "--pass", "1:1:8", "--constraint", "rock =="], ""),
        ([*base, *plane, "--method", "idw", "--nugget", "0.1", "--pass", "1:1:8"],
         "no variogram"),
        ([*base, *plane, *sills, "--method", "sk", "--pass", "1:1:8"], "mean"),
        ([*base, *plane, "--nugget-fraction", "0.2", "--sill-fraction", "0.8", "--pass",
          "1:1:8", "--composites", str(path), "--value-column", "ag"], "no column"),
    ):  # fmt: skip
        assert main(argv) == 2, argv
        assert message in capsys.readouterr().err
        assert not out.exists()


def test_the_written_plan_runs(composites, tmp_path, capsys):
    path, table = composites
    model = BlockModel(BlockModelDefinition((0, 0, 0), (20, 20, 20), (6, 6, 3)))
    model.add_attribute("rock", "category", categories=["WASTE", "ORE"])
    model["rock"][:] = (model.centroids()[:, 0] < 80.0).astype(np.int64)
    folder = tmp_path / "model"
    save_model(model, folder)
    plan = tmp_path / "plan.json"
    assert main(flags(path, plan)) == 0
    capsys.readouterr()
    assert main(["estimate", str(folder), "--composites", str(path), "--plan", str(plan)]) == 0
    estimated = load_model(folder)
    ore = estimated["rock"] == 1
    assert np.isfinite(estimated["au"][ore]).all() and np.isnan(estimated["au"][~ore]).all()
    assert estimated["au"][ore].max() <= TOP_CUT  # the top cut was applied to the samples
