"""Same seed -> identical files and arrays; different seed -> different data. Also the CLI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine import cli
from bakumine.synth.config import PRESETS, default_config
from bakumine.synth.generate import CSV_FORMATS, GENERATOR_VERSION, generate

REPO = Path(__file__).resolve().parents[2]
EXPECTED_FILES = {
    "collar.csv", "survey.csv", "assay.csv", "geology.csv",
    "truth_samples.csv", "truth_stations.csv", "truth_blocks.parquet", "truth.json",
    "ore_solid.obj", "ore_solid_vertices.npy", "ore_solid_faces.npy",
    "topo.obj", "topo_vertices.npy", "topo_faces.npy",
}  # fmt: skip


@pytest.fixture(scope="module")
def cli_run(tmp_path_factory):
    """`bakumine synth` with the same preset and seed as the ``small`` fixture."""
    out = tmp_path_factory.mktemp("synth_cli")
    args = ["synth", "--out", str(out), "--seed", "7", "--preset", "small", "--save-field"]
    code = cli.main(args)
    return code, out


def test_cli_writes_all_files(cli_run, capsys):
    code, out = cli_run
    assert code == 0
    assert {p.name for p in out.iterdir()} == EXPECTED_FILES | {"truth_gauss_field.npy"}
    truth = json.loads((out / "truth.json").read_text())
    assert set(truth["files"]) == EXPECTED_FILES | {"truth_gauss_field.npy"}


def test_cli_lists_synth_command():
    assert "synth" in cli.build_parser().format_help()


def test_same_seed_is_byte_identical(small, cli_run):
    _, out = cli_run
    names = sorted(p.name for p in small.path.iterdir())
    assert names == sorted(p.name for p in out.iterdir())
    for name in names:
        assert (small.path / name).read_bytes() == (out / name).read_bytes(), name


def test_same_seed_gives_identical_arrays(small, dataset):
    if dataset.preset != "small":
        pytest.skip("the regenerated deposit is the small one")
    a, b = small.deposit, dataset.deposit
    assert np.array_equal(a.field.values, b.field.values)
    assert np.array_equal(a.solid_vertices, b.solid_vertices)
    assert np.array_equal(a.solid_faces, b.solid_faces)
    assert np.array_equal(a.topo_vertices, b.topo_vertices)
    assert np.array_equal(a.topo_faces, b.topo_faces)
    for table in a.tables:
        for column, values in a.tables[table].items():
            other = b.tables[table][column]
            if values.dtype == object:
                assert values.tolist() == other.tolist()
            else:
                assert np.array_equal(values, other, equal_nan=True), (table, column)
    for column, values in a.blocks.columns.items():
        other = b.blocks.columns[column]
        if values.dtype == object:
            assert values.tolist() == other.tolist()
        else:
            assert np.array_equal(values, other, equal_nan=True), column


def test_different_seed_gives_different_data(small, tmp_path):
    other = generate(PRESETS["small"](), seed=8)
    other.write(tmp_path)
    for name in ("collar", "survey", "assay", "geology", "truth_samples"):
        assert (small.path / f"{name}.csv").read_bytes() != (tmp_path / f"{name}.csv").read_bytes()
    assert not np.array_equal(small.deposit.field.values, other.field.values)
    assert not np.allclose(
        small.deposit.blocks.columns["au_ore"], other.blocks.columns["au_ore"], equal_nan=True
    )
    # The ore solid is fixed by the configuration, not by the seed.
    assert np.array_equal(small.deposit.solid_vertices, other.solid_vertices)
    assert other.truth["seed"] == 8


def test_committed_default_dataset_is_current(dataset):
    """The dataset in data/synthetic/default must be what the generator produces today."""
    folder = REPO / "data" / "synthetic" / "default"
    if dataset.preset != "default":
        pytest.skip("compares the default preset only")
    if not (folder / "truth.json").exists():
        pytest.skip("no committed default dataset")
    truth = json.loads((folder / "truth.json").read_text())
    assert truth["seed"] == dataset.seed
    assert truth["generator"]["version"] == GENERATOR_VERSION
    assert truth["config"] == json.loads(json.dumps(default_config().to_dict()))
    assert {p.name for p in folder.iterdir()} == EXPECTED_FILES
    assert sum(p.stat().st_size for p in folder.iterdir()) < 15 * 1024 * 1024
    for name in CSV_FORMATS:
        committed = pd.read_csv(folder / f"{name}.csv")
        fresh = pd.read_csv(dataset.path / f"{name}.csv")
        assert list(committed.columns) == list(fresh.columns)
        assert len(committed) == len(fresh), name
        for column in committed.columns:
            if not pd.api.types.is_numeric_dtype(committed[column]):
                assert committed[column].tolist() == fresh[column].tolist()
            else:
                # Tolerance, not bytes: another platform's libm may differ in the last digit.
                np.testing.assert_allclose(
                    committed[column], fresh[column], rtol=0, atol=2e-3, equal_nan=True
                )
    np.testing.assert_allclose(
        np.load(folder / "ore_solid_vertices.npy"), dataset.deposit.solid_vertices, atol=1e-6
    )
    assert np.array_equal(np.load(folder / "ore_solid_faces.npy"), dataset.deposit.solid_faces)
