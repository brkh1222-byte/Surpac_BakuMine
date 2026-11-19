"""Fixtures for the drillhole tests."""

import pandas as pd
import pytest
from drillhole_samples import import_babbitt, import_small, small_raw_tables

from bakumine.core.drillholes.importer import DrillholeDatabase


@pytest.fixture
def small_tables() -> dict[str, pd.DataFrame]:
    return small_raw_tables()


@pytest.fixture
def small_db() -> DrillholeDatabase:
    return import_small()


@pytest.fixture(scope="session")
def babbitt_db() -> DrillholeDatabase:
    """Babbitt imported from its raw files. Session-scoped: treat as read-only."""
    return import_babbitt()
