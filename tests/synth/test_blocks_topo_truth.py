"""Truth block grid, topography DTM and truth.json."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

BLOCK_COLUMNS = [
    "i", "j", "k", "x", "y", "z", "dx", "dy", "dz", "rock",
    "ore_fraction", "waste_fraction", "air_fraction", "density", "au", "au_ore",
]  # fmt: skip


def _truth(dataset) -> dict:
    return json.loads((dataset.path / "truth.json").read_text())


def test_block_grid_layout(dataset):
    blocks = pd.read_parquet(dataset.path / "truth_blocks.parquet")
    info = _truth(dataset)["blocks"]
    nx, ny, nz = info["shape"]
    assert list(blocks.columns) == BLOCK_COLUMNS
    assert len(blocks) == nx * ny * nz
    assert info["size"] == [10.0, 10.0, 5.0]
    row = blocks.i + nx * (blocks.j + ny * blocks.k)
    assert np.array_equal(row, np.arange(len(blocks)))
    for axis, (index, size) in enumerate(zip("ijk", ("dx", "dy", "dz"), strict=True)):
        centre = info["origin"][axis] + (blocks[index] + 0.5) * info["size"][axis]
        assert np.array_equal(blocks["xyz"[axis]], centre)
        assert (blocks[size] == info["size"][axis]).all()
    # The grid encloses the solid and reaches above the topography.
    vertices = dataset.deposit.solid_vertices
    lo = np.array(info["origin"])
    hi = lo + np.array(info["shape"]) * np.array(info["size"])
    assert np.all(vertices.min(axis=0) > lo) and np.all(vertices.max(axis=0) < hi)
    top = blocks[blocks.k == nz - 1]
    assert (top.air_fraction == 1.0).all()


def test_block_rock_codes_and_fractions(dataset):
    blocks = pd.read_parquet(dataset.path / "truth_blocks.parquet")
    solid, topo = dataset.deposit.solid, dataset.deposit.topo
    assert set(blocks.rock) == {"AIR", "ORE", "WASTE"}
    total = blocks.ore_fraction + blocks.waste_fraction + blocks.air_fraction
    assert np.allclose(total, 1.0, atol=1e-12)
    for name in ("ore_fraction", "waste_fraction", "air_fraction"):
        assert blocks[name].between(0.0, 1.0).all()
    centre = blocks[["x", "y", "z"]].to_numpy()
    air = centre[:, 2] > topo.z(centre[:, 0], centre[:, 1])
    inside = solid.inside(centre)
    assert not np.any(air & inside)
    expected = np.where(air, "AIR", np.where(inside, "ORE", "WASTE"))
    assert np.array_equal(blocks.rock.to_numpy(), expected)
    # Blocks whose eight corners are all inside (and far from the pinch) are wholly ore.
    half = 0.5 * blocks[["dx", "dy", "dz"]].to_numpy()
    corners_inside = np.ones(len(blocks), dtype=bool)
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                corners_inside &= solid.implicit(centre + half * [sx, sy, sz]) < -0.3
    assert corners_inside.sum() > 20
    assert (blocks.ore_fraction[corners_inside] > 0.95).all()
    assert (blocks.ore_fraction[blocks.air_fraction == 1.0] == 0.0).all()


def test_block_totals_match_the_truth(dataset):
    blocks = pd.read_parquet(dataset.path / "truth_blocks.parquet")
    truth = _truth(dataset)
    grade = dataset.deposit.grade
    cfg = dataset.deposit.config.grade
    volume = blocks.dx * blocks.dy * blocks.dz
    ore_volume = (blocks.ore_fraction * volume).sum()
    assert abs(ore_volume / truth["totals"]["ore_volume_m3"] - 1.0) < 0.01
    metal = (blocks.ore_fraction * volume * cfg.ore_density * blocks.au_ore.fillna(0.0)).sum()
    assert abs(metal / truth["totals"]["ore_metal_g"] - 1.0) < 0.02
    # Grades: missing only where there is no rock / no ore; whole-block grade is the
    # mass-weighted mix of the ore part and the waste mean.
    assert blocks.au_ore.isna().equals(blocks.ore_fraction == 0.0)
    assert blocks.au.isna().equals(blocks.air_fraction == 1.0)
    assert blocks.density.isna().equals(blocks.air_fraction == 1.0)
    ore_mass = cfg.ore_density * blocks.ore_fraction
    waste_mass = cfg.waste_density * blocks.waste_fraction
    rock = blocks.air_fraction < 1.0
    mixed = (ore_mass * blocks.au_ore.fillna(0.0) + waste_mass * grade.waste_mean) / (
        ore_mass + waste_mass
    )
    assert np.allclose(blocks.au[rock], mixed[rock], rtol=1e-12)
    assert np.allclose(blocks.au[rock & (blocks.ore_fraction == 0)], grade.waste_mean)
    assert blocks.density[rock].between(cfg.waste_density - 1e-9, cfg.ore_density + 1e-9).all()


def test_block_grades_against_a_finer_independent_average(dataset):
    """Re-average a few whole-ore blocks with a denser, differently placed point set."""
    blocks = pd.read_parquet(dataset.path / "truth_blocks.parquet")
    field, grade = dataset.deposit.field, dataset.deposit.grade
    whole = blocks[blocks.ore_fraction == 1.0]
    assert len(whole) >= 10
    rng = np.random.default_rng(2)
    picked = whole.iloc[rng.choice(len(whole), 10, replace=False)]
    ratio = []
    for row in picked.itertuples():
        points = [row.x, row.y, row.z] + (rng.random((4000, 3)) - 0.5) * [row.dx, row.dy, row.dz]
        ratio.append(grade.ore_expected(field.at(points)).mean() / row.au_ore)
    assert np.abs(np.array(ratio) - 1.0).max() < 0.08
    assert abs(np.mean(ratio) - 1.0) < 0.03
    # Block averages are smoother than samples: lower variance than the point distribution.
    assert whole.au_ore.std() < grade.ore_ensemble_mean * np.sqrt(np.exp(grade.sigma**2) - 1.0)


def test_dtm_is_a_valid_surface_through_the_generating_function(dataset):
    vertices, faces = dataset.npy("topo_vertices"), dataset.npy("topo_faces")
    topo = dataset.deposit.topo
    extent = _truth(dataset)["topography"]["dtm_extent"]
    assert np.abs(vertices[:, 2] - topo.z(vertices[:, 0], vertices[:, 1])).max() < 1e-9
    assert len(np.unique(vertices[:, :2], axis=0)) == len(vertices)  # one z per (x, y)
    assert len(np.unique(faces)) == len(vertices)
    a, b, c = (vertices[faces[:, i], :2] for i in range(3))
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    assert cross.min() > 0  # counter-clockwise in plan: normals up
    lo, hi = np.array(extent["min"]), np.array(extent["max"])
    assert np.isclose(0.5 * cross.sum(), np.prod(hi - lo), rtol=1e-12)  # no holes, no overlap
    assert np.allclose(vertices[:, :2].min(axis=0), lo) and np.allclose(
        vertices[:, :2].max(axis=0), hi
    )
    # The surface undulates and stays above the whole solid.
    assert np.ptp(vertices[:, 2]) > 10.0
    solid_vertices = dataset.deposit.solid_vertices
    cover = topo.z(solid_vertices[:, 0], solid_vertices[:, 1]) - solid_vertices[:, 2]
    assert cover.min() > 10.0
    assert np.all(solid_vertices[:, :2].min(axis=0) > lo) and np.all(
        solid_vertices[:, :2].max(axis=0) < hi
    )


def test_obj_files_match_the_arrays(dataset):
    for name in ("ore_solid", "topo"):
        vertices, faces = [], []
        for line in (dataset.path / f"{name}.obj").read_text().splitlines():
            parts = line.split()
            if parts[0] == "v":
                vertices.append([float(p) for p in parts[1:]])
            elif parts[0] == "f":
                faces.append([int(p) - 1 for p in parts[1:]])
        assert np.abs(np.array(vertices) - dataset.npy(f"{name}_vertices")).max() < 1e-6
        assert np.array_equal(np.array(faces), dataset.npy(f"{name}_faces"))


def test_truth_json_records_everything(dataset):
    truth = _truth(dataset)
    cfg = dataset.deposit.config
    assert truth["seed"] == dataset.seed
    assert truth["config"] == json.loads(json.dumps(cfg.to_dict()))
    assert truth["coordinate_system"]["axes"] == "x = east, y = north, z = up"
    solid = truth["solid"]
    assert 499_000 < solid["centre"][0] < 502_000 and 5_299_000 < solid["centre"][1] < 5_302_000
    assert np.allclose(np.array(solid["axes"]), dataset.deposit.solid.axes)
    variogram = truth["variogram"]
    assert variogram["model"] == "spherical"
    assert variogram["nugget"] == cfg.variogram.nugget
    assert np.isclose(variogram["nugget"] + variogram["sill"], variogram["total_sill"])
    assert variogram["ranges_m"] == list(cfg.variogram.ranges)
    axes = np.array(variogram["axes"])
    assert np.allclose(axes.T @ axes, np.eye(3), atol=1e-12)
    # Roughly, not exactly, aligned with the lens: major axes within 15 degrees.
    angle = np.degrees(np.arccos(axes[:, 0] @ dataset.deposit.solid.axes[:, 0]))
    assert 1.0 < angle < 15.0
    totals = truth["totals"]
    assert totals["ore_volume_m3"] == solid["volume_m3"]
    assert np.isclose(totals["ore_tonnes"], totals["ore_volume_m3"] * cfg.grade.ore_density)
    assert np.isclose(totals["ore_metal_g"], totals["ore_tonnes"] * totals["ore_mean_grade_g_t"])
    assert np.isclose(totals["ore_metal_troy_oz"] * 31.1034768, totals["ore_metal_g"])
    assert truth["density_t_m3"] == {"ORE": cfg.grade.ore_density, "WASTE": cfg.grade.waste_density}
    assert 0.5 < totals["ore_mean_grade_g_t"] < 5.0
    assert truth["drilling"]["holes"] == len(dataset.csv("collar"))
    assert sorted(truth["files"]) == sorted(p.name for p in dataset.path.iterdir())
