"""P3.T6 acceptance: partial percentages (fraction of a block inside a constraint)."""

import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.constraints import (
    STRADDLE,
    AttributeConstraint,
    ConstraintError,
    DtmConstraint,
    FileResolver,
    PlaneConstraint,
    PolygonConstraint,
    RangeConstraint,
    SolidConstraint,
    combine,
)
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.partial import (
    assign_partial,
    constraint_volume,
    partial_fraction,
)
from bakumine.core.blockmodel.subblocking import subblock_to_constraint
from bakumine.core.geometry.dtm import dtm_from_grid
from bakumine.core.geometry.primitives import box, icosphere

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
TRUTH = json.loads((DATA / "truth.json").read_text())
GRID = TRUTH["blocks"]
MESH_VOLUME = TRUTH["solid"]["mesh"]["volume_m3"]
EXACT_VOLUME = TRUTH["solid"]["volume_m3"]
ORE = SolidConstraint("ore_solid")
AIR = DtmConstraint("topo", "above")
ROT_Z30 = (
    (math.cos(math.radians(30.0)), -math.sin(math.radians(30.0)), 0.0),
    (math.sin(math.radians(30.0)), math.cos(math.radians(30.0)), 0.0),
    (0.0, 0.0, 1.0),
)
TILTED = {"angles": (30.0, 20.0, 10.0)}


@pytest.fixture(scope="module")
def resolver():
    return FileResolver(DATA)


@pytest.fixture(scope="module")
def truth():
    """(model on the truth grid, truth ore_fraction, truth air_fraction) in model row order."""
    table = pd.read_parquet(DATA / "truth_blocks.parquet")
    d = BlockModelDefinition(GRID["origin"], GRID["size"], GRID["shape"], max_depth=2)
    rows = d.flat_index(table[["i", "j", "k"]].to_numpy())
    ore = np.empty(len(table))
    air = np.empty(len(table))
    ore[rows] = table["ore_fraction"].to_numpy()
    air[rows] = table["air_fraction"].to_numpy()
    return BlockModel(d), ore, air


def total(model, fraction):
    return float((fraction * model.volumes()).sum())


# ------------------------------------------------------------------ the plan's criterion


def test_sphere_volume_by_partials(capsys):
    radius, centre = 60.0, (151.0, 148.5, 77.0)
    sphere = icosphere(5, radius, centre)
    analytic = 4.0 / 3.0 * math.pi * radius**3
    constraint = SolidConstraint(sphere)
    model = BlockModel(BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (30, 30, 30)))
    errors = {}
    lines = []
    for method in ("lattice", "column"):
        for n in (1, 2, 3, 4, 5, 8, 10):
            fraction = partial_fraction(model, constraint, samples=n, method=method)
            assert fraction.shape == (model.n_blocks,)
            assert fraction.min() >= 0.0 and fraction.max() <= 1.0
            volume = total(model, fraction)
            errors[method, n] = (volume / sphere.volume - 1.0, volume / analytic - 1.0)
        cells = [
            f"  n={n}: {100 * errors[method, n][0]:+.4f}% / {100 * errors[method, n][1]:+.4f}%"
            for n in (1, 2, 3, 4, 5, 8, 10)
        ]
        lines.append(f"  {method:8s}" + "".join(cells))
    # the plan: within 0.5%. Stated n: 4 x 4 x 4 points per cut block, or 4 x 4 lines.
    for method in ("lattice", "column"):
        for n in (2, 3, 4, 5, 8, 10):
            assert abs(errors[method, n][0]) < 0.005  # vs the volume of the mesh
            assert abs(errors[method, n][1]) < 0.005  # vs 4/3 pi r^3
        assert abs(errors[method, 4][0]) < 0.001 and abs(errors[method, 10][0]) < 1e-4
    with capsys.disabled():
        print(
            f"\n[P3.T6 sphere] r = 60 m (20 480 faces, mesh volume "
            f"{100 * (sphere.volume / analytic - 1):+.4f}% of 4/3 pi r^3) in 10 x 10 x 5 m "
            "blocks; error of sum(fraction * volume) vs mesh / vs analytic:\n" + "\n".join(lines)
        )


def test_ore_fraction_agrees_with_truth(truth, resolver, capsys):
    model, ore_truth, _ = truth
    d = model.definition
    lo, hi = model.bounds_local()
    cut = ORE.classify_boxes(d, lo, hi, resolver) == STRADDLE
    lines = []
    t0 = time.perf_counter()
    lattice = partial_fraction(model, ORE, resolver, samples=(5, 5, 5))
    t_lattice = time.perf_counter() - t0
    t0 = time.perf_counter()
    column = partial_fraction(model, ORE, resolver, samples=5, method="column")
    t_column = time.perf_counter() - t0
    for name, fraction, mean_tol, max_tol in (
        ("lattice 5x5x5", lattice, 0.003, 0.05),
        ("column 5x5", column, 0.008, 0.06),
    ):
        # exactly 0 / 1 away from the boundary ...
        assert np.isin(fraction[~cut], (0.0, 1.0)).all()
        assert ((fraction > 0.0) & (fraction < 1.0)).sum() <= cut.sum()
        # ... where the truth (5 x 5 x 5 samples of the implicit solid) agrees, except for a
        # handful of blocks that the implicit surface grazes and its mesh does not
        off = np.abs(fraction - ore_truth)[~cut]
        assert (off > 0).sum() <= 5 and off.max() <= 2 / 125 + 1e-12
        diff = np.abs(fraction - ore_truth)[cut]
        assert diff.mean() < mean_tol and diff.max() < max_tol
        volume = total(model, fraction)
        lines.append(
            f"  {name}: |fraction - truth ore_fraction| over the {cut.sum():,} cut blocks: mean "
            f"{diff.mean():.4f}, max {diff.max():.4f}; sum(fraction * volume) = {volume:,.0f} m3: "
            f"{100 * (volume / EXACT_VOLUME - 1):+.4f}% vs exact, "
            f"{100 * (volume / MESH_VOLUME - 1):+.4f}% vs mesh"
        )
    # the same sample points as the truth: only mesh-vs-implicit differences remain
    assert (np.abs(lattice - ore_truth) > 1e-12).mean() < 0.005
    # the plan's volume criterion with the column method: within 0.5% of the exact volume
    assert total(model, column) == pytest.approx(EXACT_VOLUME, rel=0.005)
    # in fact it reproduces the mesh volume to 0.01%; the rest is the mesh's own -0.057%
    assert total(model, column) == pytest.approx(MESH_VOLUME, rel=1e-4)
    assert total(model, lattice) == pytest.approx(MESH_VOLUME, rel=2e-4)
    with capsys.disabled():
        print(
            f"\n[P3.T6 ore] truth grid, {d.n_parents:,} blocks (lattice {t_lattice:.2f} s, "
            f"column {t_column:.2f} s):\n" + "\n".join(lines)
        )


def test_air_fraction_agrees_with_truth(truth, resolver, capsys):
    model, _, air_truth = truth
    d = model.definition
    lo, hi = model.bounds_local()
    cut = AIR.classify_boxes(d, lo, hi, resolver) == STRADDLE
    truth_volume = total(model, air_truth)
    lines = []
    for name, kw in (("lattice 5x5x5", {"samples": 5}),
                     ("column 5x5", {"samples": 5, "method": "column"})):  # fmt: skip
        fraction = partial_fraction(model, AIR, resolver, **kw)
        assert np.isin(fraction[~cut], (0.0, 1.0)).all()
        # the DTM is the analytic surface sampled every 25 m: up to ~0.75 m of chord error,
        # i.e. 0.15 of a 5 m block right under a crest; elsewhere the two agree closely
        diff = np.abs(fraction - air_truth)
        assert diff[~cut].max() <= 0.12 and (diff[~cut] > 0).mean() < 2e-4
        assert diff[cut].mean() < 0.04 and diff[cut].max() < 0.25
        volume = total(model, fraction)
        assert volume == pytest.approx(truth_volume, rel=5e-4)
        lines.append(
            f"  {name}: |fraction - truth air_fraction| over the {cut.sum():,} cut blocks: mean "
            f"{diff[cut].mean():.4f}, max {diff[cut].max():.4f}; air volume {volume:,.0f} m3, "
            f"{100 * (volume / truth_volume - 1):+.4f}% vs truth"
        )
    # below = 1 - above, block by block
    below = partial_fraction(model, ~AIR, resolver, samples=5, method="column")
    np.testing.assert_allclose(below, 1.0 - fraction, rtol=0, atol=1e-12)
    with capsys.disabled():
        print("\n[P3.T6 air] DTM (25 m grid) vs analytic topography:\n" + "\n".join(lines))


# ------------------------------------------------------------------ what each method computes


@pytest.mark.parametrize("rotation", [{}, {"matrix": ROT_Z30}, TILTED])
def test_lattice_equals_centroid_coding_of_uniform_sub_blocks(rotation, resolver):
    """An n = 4 lattice is exactly the level-2 sub-block centroids."""
    mesh = resolver.solid("ore_solid")
    d = BlockModelDefinition.auto_extent(mesh, GRID["size"], margin=15.0, max_depth=2, **rotation)
    model = BlockModel(d)
    lo, hi = model.bounds_local()
    cut = np.flatnonzero(ORE.classify_boxes(d, lo, hi, resolver) == STRADDLE)[::7]
    fraction = partial_fraction(model, ORE, resolver, samples=4, where=cut)
    assert fraction.shape == (len(cut),)
    parents = model.parent[cut]
    model.split(cut)
    model.split(np.isin(model.parent, parents))
    children = np.isin(model.parent, parents)
    assert children.sum() == 64 * len(cut)
    selected = ORE.mask(model, resolver)[children].reshape(len(cut), 64)
    np.testing.assert_array_equal(fraction, selected.sum(axis=1) / 64.0)
    assert 0.2 < fraction.mean() < 0.8


def test_column_method_is_exact_along_k():
    d = BlockModelDefinition((100.0, 200.0, 50.0), (10.0, 10.0, 5.0), (6, 5, 8))
    model = BlockModel(d)
    lo, hi = model.bounds_local()
    z_lo, z_hi = lo[:, 2] + 50.0, hi[:, 2] + 50.0
    # a horizontal surface inside a bench: exact with one line per block
    flat = dtm_from_grid(np.full((3, 3), 63.2), (90.0, 190.0), (40.0, 40.0))
    below = partial_fraction(model, DtmConstraint(flat, "below"), samples=1, method="column")
    np.testing.assert_allclose(below, np.clip((63.2 - z_lo) / 5.0, 0.0, 1.0), rtol=0, atol=1e-12)
    assert ((below > 0) & (below < 1)).sum() == 30
    # the lattice can only round that to its layers: 0.64 becomes 0.6 (n = 5) or 0.5 (n = 2)
    for n, value in ((5, 0.6), (2, 0.5), (1, 1.0)):
        lattice = partial_fraction(model, DtmConstraint(flat, "below"), samples=n)
        np.testing.assert_allclose(lattice[(below > 0) & (below < 1)], value)
    # an inclined plane, as a DTM and as a plane constraint: the volume below it is exact
    # (the midpoint rule integrates a linear height exactly), block fractions agree
    gx, gy = np.meshgrid(90.0 + 20.0 * np.arange(5), 190.0 + 20.0 * np.arange(5), indexing="ij")
    height = 61.0 + 0.11 * (gx - 100.0) + 0.07 * (gy - 200.0)
    slope = dtm_from_grid(height, (90.0, 190.0), (20.0, 20.0))
    normal = np.array([-0.11, -0.07, 1.0])
    plane = PlaneConstraint((100.0, 200.0, 61.0), normal, "below")
    exact = 0.0
    for i in range(6):
        for j in range(5):
            exact += 100.0 * (61.0 + 0.11 * (10.0 * i + 5.0) + 0.07 * (10.0 * j + 5.0) - 50.0)
    by_dtm = partial_fraction(model, DtmConstraint(slope, "below"), samples=2, method="column")
    by_plane = partial_fraction(model, plane, samples=2, method="column")
    np.testing.assert_allclose(by_dtm, by_plane, rtol=0, atol=1e-9)
    assert total(model, by_plane) == pytest.approx(exact, rel=1e-12)
    assert total(model, by_dtm) == pytest.approx(exact, rel=1e-9)
    assert ((by_plane > 0) & (by_plane < 1)).sum() > 30
    # ranges: a bench between two elevations, and a vertical limit sampled in plan
    bench = partial_fraction(model, RangeConstraint.elevation(57.5, 71.0), samples=1,
                             method="column")  # fmt: skip
    expected = np.clip((np.minimum(z_hi, 71.0) - np.maximum(z_lo, 57.5)) / 5.0, 0.0, 1.0)
    np.testing.assert_allclose(bench, expected, rtol=0, atol=1e-12)
    east = partial_fraction(model, RangeConstraint(x=(None, 127.5)), samples=4, method="column")
    np.testing.assert_array_equal(np.unique(east), [0.0, 0.75, 1.0])
    index = partial_fraction(model, RangeConstraint(k=(2, 3), x=(None, 127.5)), samples=4,
                             method="column")  # fmt: skip
    k = model.ijk()[:, 2]
    np.testing.assert_array_equal(index, np.where((k >= 2) & (k <= 3), east, 0.0))


@pytest.mark.parametrize("rotation", [{}, {"matrix": ROT_Z30}, TILTED])
def test_box_solid_fractions_are_exact_with_the_column_method(rotation):
    """A box in the model frame, off the block faces: the true fraction is a product of overlaps."""
    d = BlockModelDefinition((500.0, 900.0, 40.0), (10.0, 10.0, 5.0), (9, 8, 7), **rotation)
    model = BlockModel(d)
    a, b = np.array([21.25, 13.75, 6.3]), np.array([63.75, 58.75, 27.1])
    local = box(a, b)
    solid = SolidConstraint(type(local)(d.local_to_world(local.vertices), local.faces))
    lo, hi = model.bounds_local()
    overlap = np.clip(np.minimum(hi, b) - np.maximum(lo, a), 0.0, None) / (hi - lo)
    exact = overlap.prod(axis=1)
    # the plan lattice hits the plan faces exactly at n = 8 (1.25 m cells); k is exact anyway
    column = partial_fraction(model, solid, samples=8, method="column")
    np.testing.assert_allclose(column, exact, rtol=0, atol=1e-9)
    assert total(model, column) == pytest.approx(42.5 * 45.0 * 20.8, rel=1e-9)
    # the lattice rounds the k overlap to its layers
    lattice = partial_fraction(model, solid, samples=8)
    assert np.abs(lattice - exact).max() > 0.02
    assert np.abs(lattice - exact).max() < 1.0 / 16.0 + 1e-9


@pytest.mark.parametrize("rotation", [{"matrix": ROT_Z30}, TILTED])
def test_rotated_model(rotation, resolver, capsys):
    mesh = resolver.solid("ore_solid")
    d = BlockModelDefinition.auto_extent(mesh, GRID["size"], margin=(20.0, 20.0, 150.0),
                                         **rotation)  # fmt: skip
    model = BlockModel(d)
    lattice = partial_fraction(model, ORE, resolver, samples=5)
    column = partial_fraction(model, ORE, resolver, samples=5, method="column")
    fine = partial_fraction(model, ORE, resolver, samples=10)
    for fraction in (lattice, column, fine):
        assert total(model, fraction) == pytest.approx(MESH_VOLUME, rel=3e-4)
        assert total(model, fraction) == pytest.approx(EXACT_VOLUME, rel=0.005)
    cut = (fine > 0) & (fine < 1)
    assert np.abs(column - fine)[cut].mean() < 0.006 and np.abs(lattice - fine)[cut].mean() < 0.008
    # topography cutting through the rotated model (both methods; column lines follow k)
    air_lattice = partial_fraction(model, AIR, resolver, samples=10)
    air_column = partial_fraction(model, AIR, resolver, samples=5, method="column")
    assert ((air_lattice > 0) & (air_lattice < 1)).sum() > 1000
    assert total(model, air_column) == pytest.approx(total(model, air_lattice), rel=1e-4)
    assert np.abs(air_column - air_lattice).max() < 0.08
    with capsys.disabled():
        print(
            f"\n[P3.T6 rotated {'tilted' if rotation is TILTED else '30 deg about Z'}] ore volume "
            f"vs mesh: lattice 5 {100 * (total(model, lattice) / MESH_VOLUME - 1):+.4f}%, column 5 "
            f"{100 * (total(model, column) / MESH_VOLUME - 1):+.4f}%, lattice 10 "
            f"{100 * (total(model, fine) / MESH_VOLUME - 1):+.4f}%; air: column 5 vs lattice 10 "
            f"{100 * (total(model, air_column) / total(model, air_lattice) - 1):+.4f}%"
        )


# ------------------------------------------------------------------ combinations, options


def test_combinations(truth, resolver):
    model, _, _ = truth
    d = model.definition
    centre = np.array(d.local_to_world(0.5 * np.array(d.extent)))
    plane = PlaneConstraint(centre, (0.3, -0.2, 0.9), "below")
    both = combine("ore and footwall", {"ore": ORE, "footwall": plane})
    either = ORE | plane
    for method, n, tol in (("lattice", 4, 1e-12), ("column", 4, 1e-9)):
        kw = {"samples": n, "method": method}
        f_ore = partial_fraction(model, ORE, resolver, **kw)
        f_plane = partial_fraction(model, plane, resolver, **kw)
        f_and = partial_fraction(model, both, resolver, **kw)
        f_or = partial_fraction(model, either, resolver, **kw)
        f_not = partial_fraction(model, ~both, resolver, **kw)
        # inclusion-exclusion and complement hold sample by sample, hence block by block
        np.testing.assert_allclose(f_and + f_or, f_ore + f_plane, rtol=0, atol=tol)
        np.testing.assert_allclose(f_not, 1.0 - f_and, rtol=0, atol=tol)
        assert (f_and <= np.minimum(f_ore, f_plane) + tol).all()
        assert 0.3 * MESH_VOLUME < total(model, f_and) < 0.7 * MESH_VOLUME
    # the two methods agree on the combined volume
    lattice = total(model, partial_fraction(model, both, resolver, samples=6))
    column = total(model, partial_fraction(model, both, resolver, samples=6, method="column"))
    assert column == pytest.approx(lattice, rel=5e-4)
    # ore below the topography is all of the ore (min cover 64 m); in the air there is none
    in_rock = partial_fraction(model, ORE & ~AIR, resolver, samples=5, method="column")
    np.testing.assert_array_equal(
        in_rock, partial_fraction(model, ORE, resolver, samples=5, method="column")
    )
    assert not partial_fraction(model, ORE & AIR, resolver).any()


def test_polygon_by_lattice_only():
    d = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (6, 6, 2))
    model = BlockModel(d)
    polygon = PolygonConstraint([(15.0, 15.0), (45.0, 15.0), (45.0, 40.0), (15.0, 40.0)])
    fraction = partial_fraction(model, polygon, samples=(2, 2, 1))
    lo, hi = model.bounds_local()
    a, b = np.array([15.0, 15.0]), np.array([45.0, 40.0])
    overlap = np.clip(np.minimum(hi[:, :2], b) - np.maximum(lo[:, :2], a), 0.0, None) / 10.0
    np.testing.assert_array_equal(fraction, overlap.prod(axis=1))
    assert total(model, fraction) == 30.0 * 25.0 * 10.0
    with pytest.raises(ConstraintError, match="column method"):
        partial_fraction(model, polygon, method="column")
    with pytest.raises(ConstraintError, match="column method"):
        partial_fraction(model, polygon & RangeConstraint(z=(0.0, 5.0)), method="column")


def test_where_chunks_sub_blocks_and_errors(truth, resolver):
    model, _, _ = truth
    full = partial_fraction(model, ORE, resolver, samples=3)
    for method in ("lattice", "column"):
        reference = partial_fraction(model, ORE, resolver, samples=3, method=method)
        small = partial_fraction(model, ORE, resolver, samples=3, method=method, chunk_size=1000)
        np.testing.assert_array_equal(small, reference)
    rows = np.flatnonzero((full > 0) & (full < 1))[::-3]
    np.testing.assert_array_equal(
        partial_fraction(model, ORE, resolver, samples=3, where=rows), full[rows]
    )
    mask = model.centroids()[:, 0] > GRID["origin"][0] + 190.0
    np.testing.assert_array_equal(
        partial_fraction(model, ORE, resolver, samples=3, where=mask), full[mask]
    )
    assert partial_fraction(model, ORE, resolver, where=np.zeros(0, dtype=np.int64)).shape == (0,)
    # on a sub-blocked model: per-leaf fractions; the total is unchanged in kind, and summing
    # sub-block fractions (column) gives the same volume as parent fractions at the same lines
    sub = BlockModel(model.definition)
    subblock_to_constraint(sub, ORE, resolver, max_level=1)
    parent_fraction = partial_fraction(model, ORE, resolver, samples=4, method="column")
    leaf_fraction = partial_fraction(sub, ORE, resolver, samples=2, method="column")
    assert leaf_fraction.shape == (sub.n_blocks,)
    assert total(sub, leaf_fraction) == pytest.approx(total(model, parent_fraction), rel=1e-12)
    # errors
    with pytest.raises(ConstraintError, match="geometric"):
        partial_fraction(model, AttributeConstraint("z > 900"), resolver)
    with pytest.raises(ValueError, match="method"):
        partial_fraction(model, ORE, resolver, method="exact")
    for bad in (0, (5, 5), (2, 2, 0), 2.5, "many"):
        with pytest.raises(ValueError, match="samples"):
            partial_fraction(model, ORE, resolver, samples=bad)
    assert partial_fraction(model, ORE, resolver, samples=(3, 3), method="column").shape == (
        model.n_blocks,
    )
    with pytest.raises(ValueError, match="chunk_size"):
        partial_fraction(model, ORE, resolver, chunk_size=0)
    open_mesh = resolver.dtm("topo").mesh
    with pytest.raises(ValueError, match="closed"):
        partial_fraction(model, SolidConstraint(open_mesh), method="column")


def test_assign_partial_and_constraint_volume(resolver):
    model = BlockModel(BlockModelDefinition(GRID["origin"], GRID["size"], GRID["shape"]))
    fraction = assign_partial(model, "ore_pct", ORE, resolver, samples=5, method="column")
    spec = model.attribute("ore_pct")
    assert spec.kind == "float" and spec.background == 0.0
    np.testing.assert_array_equal(model["ore_pct"], fraction)
    # volume under a constraint: centroid rule, or weighted by the partial fraction
    assert constraint_volume(model) == model.definition.volume
    by_centroid = constraint_volume(model, ORE, resolver)
    assert by_centroid == 1000.0 * 500.0 * ORE.mask(model, resolver).sum() / 1000.0
    by_partial = constraint_volume(model, partial="ore_pct")
    assert by_partial == pytest.approx(float((fraction * 500.0).sum()), rel=1e-12)
    assert abs(by_partial / MESH_VOLUME - 1.0) < 1e-4 < abs(by_centroid / MESH_VOLUME - 1.0)
    assert constraint_volume(model, partial=fraction) == by_partial
    # both: ore above the 1000 m level (bench limit on block faces)
    upper = RangeConstraint.elevation(1000.0, None)
    assert constraint_volume(model, upper, partial="ore_pct") == pytest.approx(
        float((fraction * 500.0)[model.centroids()[:, 2] >= 1000.0].sum()), rel=1e-12
    )
    assert constraint_volume(model, AttributeConstraint("ore_pct >= 0.5"), partial=None) > 0
    # percent, overwrite, errors
    assign_partial(model, "air_pct", AIR, resolver, scale=100.0, samples=2)
    assert model.attribute("air_pct").unit == "%" and model["air_pct"].max() == 100.0
    assign_partial(model, "ore_pct", AIR, resolver, samples=2)
    np.testing.assert_array_equal(model["ore_pct"] * 100.0, model["air_pct"])
    model.add_attribute("code", "int")
    with pytest.raises(ValueError, match="float"):
        assign_partial(model, "code", ORE, resolver)
    with pytest.raises(TypeError, match="where"):
        assign_partial(model, "ore_pct", ORE, resolver, where=np.arange(3))
    with pytest.raises(ValueError, match="one value per block"):
        constraint_volume(model, partial=np.ones(3))
    with pytest.raises(KeyError):
        constraint_volume(model, partial="nope")
