"""Shared fixtures: each preset is generated once per test session and written to a temp folder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.synth.config import PRESETS
from bakumine.synth.generate import SynthDeposit, generate

SEEDS = {"small": 7, "default": 42}


@dataclass(frozen=True)
class Dataset:
    preset: str
    seed: int
    deposit: SynthDeposit
    path: Path

    def csv(self, name: str) -> pd.DataFrame:
        return pd.read_csv(self.path / f"{name}.csv", dtype={"hole_id": str})

    def npy(self, name: str) -> np.ndarray:
        return np.load(self.path / f"{name}.npy")


@pytest.fixture(scope="session", params=["small", "default"])
def dataset(request, tmp_path_factory) -> Dataset:
    preset = request.param
    deposit = generate(PRESETS[preset](), seed=SEEDS[preset])
    path = tmp_path_factory.mktemp(f"synth_{preset}")
    deposit.write(path)
    return Dataset(preset=preset, seed=SEEDS[preset], deposit=deposit, path=path)


@pytest.fixture(scope="session")
def small(tmp_path_factory) -> Dataset:
    """The reduced deposit only, for tests that regenerate it."""
    deposit = generate(PRESETS["small"](), seed=SEEDS["small"])
    path = tmp_path_factory.mktemp("synth_small_again")
    deposit.write(path, save_field=True)
    return Dataset(preset="small", seed=SEEDS["small"], deposit=deposit, path=path)
