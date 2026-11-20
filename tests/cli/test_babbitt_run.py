"""P7.T2: scripts/run_babbitt.py on the real Babbitt Cu-Ni drillholes (about 15 s).

Acceptance (plan.md P7.T2, research.md section 9): the global mean of the ordinary-kriging
model is within 5 % of both declustered references (nearest neighbour into the same blocks,
cell-declustered composites), and the swath curves of blocks and composites correlate (> 0.5)
along X, Y and Z.  The choices behind the run are documented in the script's docstring.
"""

import importlib.util
import json
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def metrics(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "run_babbitt", REPO / "scripts" / "run_babbitt.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("babbitt") / "run"
    result = module.run(out, cross_validation=False)
    assert json.loads((out / "metrics.json").read_text())["global_mean"] == result["global_mean"]
    for name in ("model/model.json", "report.csv", "estimation_summary_ok.json", "coding.json",
                 "geometry/top_vertices.npy", "geometry/base_faces.npy"):  # fmt: skip
        assert (out / name).is_file(), name
    return result


def test_units_converted(metrics):
    # feet -> metres: composites are 10 ft = 3.048 m
    assert metrics["choices"]["composite_length_m"] == pytest.approx(3.048)
    assert metrics["composites"]["n_with_cu"] > 20000


def test_domain(metrics):
    d = metrics["domain"]
    # the composites below the basal contact are footwall: barren
    assert d["footwall_cu_median"] < 0.05
    zones = d["composites_by_zone"]
    assert zones["1"] > 5000 and zones["2"] > 2000
    b = metrics["blocks"]
    assert b["by_zone"]["basal"] > 3000 and b["by_zone"]["upper"] > 1000
    # every domain block is estimated, and almost all within twice the variogram range
    assert b["estimated"] == b["in_domain"]
    assert b["by_pass"]["3"] < 0.05 * b["in_domain"]


def test_variogram_fitted(metrics):
    for name in ("basal", "upper"):
        v = metrics["variogram"][name]
        assert v["r_squared"] >= 0.8 and not v["at_bounds"], v


def test_global_mean_within_5_percent(metrics):
    """Research.md section 9: OK against the declustered references, overall and per zone."""
    g = metrics["global_mean"]
    assert g["n_blocks"] == metrics["blocks"]["in_domain"]
    for scope in (g, *g["by_zone"].values()):
        assert abs(scope["ok_vs_nn"]) < 0.05, scope
        assert abs(scope["ok_vs_declustered"]) < 0.05, scope


def test_swath_correlation(metrics):
    for axis, s in metrics["swath"].items():
        assert s["slices_used"] >= 10, (axis, s["slices_used"])
        assert s["correlation"] > 0.5, (axis, s["correlation"])
        assert s["correlation_naive"] > 0.5, (axis, s["correlation_naive"])
        for row in s["table"]:
            assert row["sample_count"] >= 30 and row["block_count"] >= 10
