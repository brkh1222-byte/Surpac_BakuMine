"""Shared fixtures of the geology tests: the small synthetic layered deposit, generated once."""

from __future__ import annotations

from pathlib import Path

import pytest

from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.drillholes.importer import DrillholeDatabase
from bakumine.synth.layered import LAYERED_PRESETS
from bakumine.synth.layered_generate import LayeredDeposit, generate_layered, import_layered

LAYERED_SEED = 42


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "slow: needs the full synthetic layered deposit")


@pytest.fixture(scope="session")
def layered_small() -> LayeredDeposit:
    return generate_layered(LAYERED_PRESETS["layered-small"], seed=LAYERED_SEED)


@pytest.fixture(scope="session")
def layered_small_dir(layered_small, tmp_path_factory) -> Path:
    """The written folder. Files named ``truth*`` and ``ore_solid*`` are the oracle."""
    folder = tmp_path_factory.mktemp("layered_small")
    layered_small.write(folder)
    return folder


@pytest.fixture(scope="session")
def layered_small_db(layered_small_dir) -> DrillholeDatabase:
    return import_layered(layered_small_dir)


@pytest.fixture(scope="session")
def layered_small_traces(layered_small_db) -> HoleTraces:
    return layered_small_db.desurvey()


@pytest.fixture(scope="session")
def layered_units_path(layered_small_dir) -> Path:
    return layered_small_dir / "units.json"
