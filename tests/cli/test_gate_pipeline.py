"""Gate A and Gate B through the CLI: scripts/gate_synthetic.py on the synthetic deposit."""

import importlib.util
import json
from pathlib import Path

import pytest

from bakumine.cli import build_parser

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def metrics(tmp_path_factory):
    spec = importlib.util.spec_from_file_location(
        "gate_synthetic", REPO / "scripts" / "gate_synthetic.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    out = tmp_path_factory.mktemp("gate")
    result = module.run(out / "run")
    assert json.loads((out / "run" / "metrics.json").read_text()) == result
    for name in ("model/model.json", "estimation_summary.json", "report.csv"):
        assert (out / "run" / name).is_file()
    return result


def test_commands_registered():
    text = build_parser().format_help()
    assert "estimate" in text and "blockmodel" in text


def test_gate_a_coded_ore_volume(metrics):
    assert abs(metrics["volume_error"]) < 0.01, metrics


def test_gate_b_estimation(metrics):
    assert metrics["ore_blocks_unestimated"] == 0
    assert metrics["correlation"] > 0.75, metrics
    assert abs(metrics["tonnes_error"]) < 0.05, metrics
    assert abs(metrics["metal_error"]) < 0.05, metrics
