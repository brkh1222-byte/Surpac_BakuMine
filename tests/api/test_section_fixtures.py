"""The committed web section fixture still matches its generator."""

from __future__ import annotations

import json

import numpy as np
import pytest
from section_fixtures import DEMO, FIXTURE, build_fixture, plane_normal

pytestmark = pytest.mark.skipif(
    not (DEMO / "ore_solid_vertices.npy").is_file(),
    reason="data/synthetic/default has not been generated",
)


def test_plane_normal_convention():
    # W-E section: the viewer looks north, so the normal points south.
    assert plane_normal(90.0, 90.0) == [0.0, -1.0, 0.0]
    # S-N section, looking west; plan, looking down.
    assert plane_normal(0.0, 90.0) == [1.0, 0.0, 0.0]
    assert plane_normal(90.0, 0.0) == [0.0, 0.0, 1.0]
    assert np.linalg.norm(plane_normal(40.0, 65.0)) == pytest.approx(1.0)


def test_committed_section_fixture_matches_the_generator():
    """Regenerate with `uv run python tests/api/section_fixtures.py` if this fails."""
    built = json.loads(build_fixture())
    committed = json.loads(FIXTURE.read_text())
    assert [c["name"] for c in committed["cases"]] == [c["name"] for c in built["cases"]]
    for mine, theirs in zip(built["cases"], committed["cases"], strict=True):
        assert mine["closed"] == theirs["closed"], mine["name"]
        assert mine["origin"] == theirs["origin"]
        assert mine["normal"] == pytest.approx(theirs["normal"], abs=1e-15)
        assert mine["area"] == pytest.approx(theirs["area"], rel=1e-9)
        assert len(mine["polylines"]) == len(theirs["polylines"])
        for a, b in zip(mine["polylines"], theirs["polylines"], strict=True):
            np.testing.assert_allclose(np.array(a), np.array(b), atol=1e-5, rtol=0)


def test_fixture_sections_are_what_the_viewer_test_expects():
    cases = {c["name"]: c for c in json.loads(FIXTURE.read_text())["cases"]}
    # One closed loop through the ore solid, one open profile across the topography.
    assert cases["ore W-E through the middle"]["closed"] == [True]
    assert cases["topography W-E through the middle"]["closed"] == [False]
    assert cases["ore W-E through the middle"]["area"] > 1000.0
