"""P3.T1 acceptance: block model definition and indexing."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.core.blockmodel.definition import BlockModelDefinition, cell_index
from bakumine.core.estimation.rotation import rotation_matrix
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.mesh import Mesh

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
ORIGIN = (500_400.0, 5_300_250.0, 825.0)

ROTATIONS = [
    {},
    {"angles": (30.0, 0.0, 0.0)},
    {"angles": (-75.0, 20.0, 0.0)},
    {"angles": (140.0, -35.0, 25.0)},
    {"angles": (33.0, 12.0, -48.0), "convention": "gslib"},
    {"matrix": tuple(map(tuple, rotation_matrix(211.0, 5.0, 60.0)))},
]


def make(**kw):
    return BlockModelDefinition(ORIGIN, (10.0, 12.5, 5.0), (39, 52, 84), **kw)


# ------------------------------------------------------------------ validation


def test_validation_and_basic_properties():
    d = make(max_depth=3)
    assert d.max_depth == (3, 3, 3)
    assert d.extent == (390.0, 650.0, 420.0)
    assert d.n_parents == 39 * 52 * 84
    assert d.min_block_size == (1.25, 1.5625, 0.625)
    assert not d.is_rotated
    assert np.array_equal(d.rotation, np.eye(3))
    assert hash(d) == hash(make(max_depth=(3, 3, 3)))
    for bad in (
        dict(origin=(0, 0)),
        dict(block_size=(1, 0, 1)),
        dict(shape=(1, 2, 0)),
        dict(shape=(1.5, 2, 3)),
        dict(max_depth=9),
        dict(max_depth=(1, -1, 0)),
        dict(convention="nope"),
        dict(matrix=((1, 0, 0), (0, 1, 0), (0, 0, -1))),
        dict(matrix=((1, 0, 0), (0, 2, 0), (0, 0, 1))),
        dict(origin=(np.nan, 0, 0)),
    ):
        args = dict(origin=(0, 0, 0), block_size=(1, 1, 1), shape=(2, 2, 2)) | bad
        with pytest.raises(ValueError):
            BlockModelDefinition(**args)


def test_from_extent_validates_divisibility():
    d = BlockModelDefinition.from_extent(ORIGIN, (390.0, 520.0, 420.0), (10.0, 10.0, 5.0))
    assert d.shape == (39, 52, 84)
    d = BlockModelDefinition.from_extent((0, 0, 0), (1.2, 0.3, 0.7), (0.1, 0.1, 0.1))
    assert d.shape == (12, 3, 7)
    with pytest.raises(ValueError, match="whole number of blocks.*along y"):
        BlockModelDefinition.from_extent(ORIGIN, (390.0, 525.0, 420.0), (10.0, 10.0, 5.0))
    with pytest.raises(ValueError):
        BlockModelDefinition.from_extent(ORIGIN, (390.0, 520.0, 0.0), (10.0, 10.0, 5.0))


def test_rotation_columns_are_axis_directions():
    d = make(angles=(90.0, 0.0, 0.0))  # bearing 90: j axis points east, i axis south
    np.testing.assert_allclose(d.rotation[:, 1], (1, 0, 0), atol=1e-15)
    np.testing.assert_allclose(d.rotation[:, 0], (0, -1, 0), atol=1e-15)
    np.testing.assert_array_equal(d.rotation, rotation_matrix(90.0, 0.0, 0.0))
    # the origin is the pivot: local (0, 0, 0) is the origin whatever the angles
    np.testing.assert_array_equal(d.local_to_world((0.0, 0.0, 0.0)), ORIGIN)
    step = d.ijk_to_world((0, 1, 0)) - d.ijk_to_world((0, 0, 0))
    np.testing.assert_allclose(step, (12.5, 0.0, 0.0), atol=1e-9)
    # plunge lifts the j axis
    up = make(angles=(0.0, 30.0, 0.0)).rotation[:, 1]
    np.testing.assert_allclose(up, (0.0, np.cos(np.pi / 6), 0.5), atol=1e-15)
    assert not d.rotation.flags.writeable


# ------------------------------------------------------------------ round trips


@pytest.mark.parametrize("rot", ROTATIONS)
def test_ijk_world_round_trip(rot):
    d = make(**rot)
    rng = np.random.default_rng(1)
    # fractional ijk -> world -> local -> fractional ijk
    frac = rng.uniform(0.0, 1.0, (20_000, 3)) * np.array(d.shape)
    world = d.local_to_world(frac * np.array(d.block_size))
    back = d.world_to_local(world)
    assert np.abs(back - frac * np.array(d.block_size)).max() < 1e-6
    assert np.abs(d.local_to_world(back) - world).max() < 1e-6
    # centroids of random blocks map back to their own ijk
    ijk = rng.integers(0, d.shape, (20_000, 3))
    centroid = d.ijk_to_world(ijk)
    found, inside = d.world_to_ijk(centroid)
    assert inside.all()
    np.testing.assert_array_equal(found, ijk)
    # independent matrix formula
    expected = np.array(d.origin) + ((ijk + 0.5) * np.array(d.block_size)) @ d.rotation.T
    assert np.abs(centroid - expected).max() < 1e-6
    # random world points: the reported block really contains the point
    lo, hi = d.bounding_box()
    pts = rng.uniform(lo - 30.0, hi + 30.0, (50_000, 3))
    ijk, inside = d.world_to_ijk(pts)
    assert 0 < inside.sum() < len(pts) or not d.is_rotated
    local = (pts - np.array(d.origin)) @ d.rotation
    brute_inside = ((local >= 0) & (local < np.array(d.extent))).all(axis=1)
    near_face = (np.abs(local / np.array(d.block_size) % 1.0 - 0.5) > 0.5 - 1e-6).any(axis=1)
    np.testing.assert_array_equal(inside[~near_face], brute_inside[~near_face])
    offset = local[inside] - d.ijk_to_local(ijk[inside])
    assert (np.abs(offset) <= 0.5 * np.array(d.block_size) + 1e-6).all()


def test_flat_index_round_trip_and_order():
    d = make()
    flat = np.arange(d.n_parents)
    ijk = d.unflat_index(flat)
    np.testing.assert_array_equal(d.flat_index(ijk), flat)
    # documented C order, k fastest
    np.testing.assert_array_equal(ijk[:3], [(0, 0, 0), (0, 0, 1), (0, 0, 2)])
    np.testing.assert_array_equal(ijk[d.nz], (0, 1, 0))
    np.testing.assert_array_equal(ijk[d.ny * d.nz], (1, 0, 0))
    grid = np.stack(np.meshgrid(*(np.arange(n) for n in d.shape), indexing="ij"), axis=-1)
    np.testing.assert_array_equal(grid.reshape(-1, 3), ijk)
    np.testing.assert_array_equal(d.parent_ijk(), ijk)
    f = d.flat_index_f(ijk)
    np.testing.assert_array_equal(f, ijk[:, 0] + d.nx * (ijk[:, 1] + d.ny * ijk[:, 2]))


# ------------------------------------------------------------------ containing block


def test_containing_block_half_open_rule_on_faces():
    d = BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (39, 52, 84))
    size = np.array(d.block_size)
    ijk = np.array([(0, 0, 0), (5, 7, 9), (38, 51, 83)])
    lo = np.array(ORIGIN) + ijk * size
    hi = lo + size
    # the minimum corner and minimum faces belong to the block
    found, inside = d.world_to_ijk(lo)
    np.testing.assert_array_equal(found, ijk)
    assert inside.all()
    # a point on a maximum face belongs to the next block along that axis
    for a in range(3):
        p = lo + 0.5 * size
        p[:, a] = hi[:, a]
        found, inside = d.world_to_ijk(p)
        expect = ijk.copy()
        expect[:, a] += 1
        np.testing.assert_array_equal(found, expect)
        np.testing.assert_array_equal(inside, [True, True, False])  # model max face is outside
        # just below the face: still this block
        p[:, a] = np.nextafter(hi[:, a], -np.inf)
        found, inside = d.world_to_ijk(p)
        np.testing.assert_array_equal(found, ijk)
        assert inside.all()
    # below the model minimum
    found, inside = d.world_to_ijk(np.nextafter(np.array(ORIGIN), -np.inf))
    np.testing.assert_array_equal(found, (-1, -1, -1))
    assert not inside
    found, inside = d.world_to_ijk((np.nan, 0.0, 0.0))
    assert not inside and found[0] == -1
    found, inside = d.world_to_ijk((1e300, -1e300, ORIGIN[2]))
    np.testing.assert_array_equal(found, (39, -1, 0))


@pytest.mark.parametrize("size", [0.1, 0.3, 1.0 / 3.0, 7.7, 1e-3])
def test_cell_index_agrees_with_face_comparisons(size):
    """index n <=> fl(n * size) <= x < fl((n + 1) * size), also where x / size rounds badly."""
    n = np.arange(0, 2000)
    faces = n * size
    probes = np.concatenate([faces, np.nextafter(faces, np.inf), np.nextafter(faces, -np.inf)])
    idx, inside = cell_index(probes, size, 2000)
    ok = inside
    assert ((idx[ok] * size <= probes[ok]) & (probes[ok] < (idx[ok] + 1) * size)).all()
    np.testing.assert_array_equal(idx[: len(n)], n)  # a face belongs to the block above
    np.testing.assert_array_equal(idx[len(n) : 2 * len(n)], n)
    np.testing.assert_array_equal(idx[2 * len(n) :], n - 1)


@pytest.mark.parametrize("rot", ROTATIONS)
def test_containing_block_consistent_with_centroid_half_size(rot):
    d = make(**rot)
    rng = np.random.default_rng(7)
    size = np.array(d.block_size)
    ijk = rng.integers(0, d.shape, (5000, 3))
    # points strictly inside each block (99.99% of the half size) and its corners
    offs = rng.uniform(-0.4999, 0.4999, (5000, 3)) * size
    pts = d.local_to_world(d.ijk_to_local(ijk) + offs)
    found, inside = d.world_to_ijk(pts)
    assert inside.all()
    np.testing.assert_array_equal(found, ijk)
    corners = d.block_corners(ijk)
    assert corners.shape == (5000, 8, 3)
    centre = corners.mean(axis=1)
    assert np.abs(centre - d.ijk_to_world(ijk)).max() < 1e-6
    local = d.world_to_local(corners)
    np.testing.assert_allclose(local[:, 0], ijk * size, atol=1e-6)
    np.testing.assert_allclose(local[:, 7], (ijk + 1) * size, atol=1e-6)
    np.testing.assert_allclose(local[:, 1] - local[:, 0], np.tile((0, 0, size[2]), (5000, 1)),
                               atol=1e-6)  # fmt: skip
    np.testing.assert_allclose(local[:, 4] - local[:, 0], np.tile((size[0], 0, 0), (5000, 1)),
                               atol=1e-6)  # fmt: skip


def test_bounding_box():
    d = make()
    np.testing.assert_array_equal(d.bounding_box(), [ORIGIN, np.add(ORIGIN, d.extent)])
    r = make(angles=(90.0, 0.0, 0.0))  # j east, i south
    lo, hi = r.bounding_box()
    np.testing.assert_allclose(lo, (ORIGIN[0], ORIGIN[1] - 390.0, ORIGIN[2]), atol=1e-6)
    np.testing.assert_allclose(hi, (ORIGIN[0] + 650.0, ORIGIN[1], ORIGIN[2] + 420.0), atol=1e-6)
    for rot in ROTATIONS:
        d = make(**rot)
        lo, hi = d.bounding_box()
        c = d.parent_centroids()
        assert (c > lo).all() and (c < hi).all()


# ------------------------------------------------------------------ auto extent


def _synthetic_geometry():
    solid = Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))
    topo = DTM(Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy")))
    return solid, topo


def test_auto_extent_covers_synthetic_solid_and_topo():
    solid, topo = _synthetic_geometry()
    size = np.array((10.0, 10.0, 5.0))
    margin = 40.0
    d = BlockModelDefinition.auto_extent([solid, topo], size, margin=margin)
    pts = np.concatenate([solid.vertices, topo.mesh.vertices])
    lo, hi = d.bounding_box()
    assert (lo <= pts.min(axis=0) - margin).all()
    assert (hi >= pts.max(axis=0) + margin).all()
    # tight: no more than one block beyond the margin on each side
    assert (lo > pts.min(axis=0) - margin - size).all()
    assert (hi < pts.max(axis=0) + margin + size).all()
    # snapped: the origin is a multiple of the block size
    np.testing.assert_array_equal(np.array(d.origin) % size, 0.0)
    # every geometry point is inside the model
    assert d.world_to_ijk(pts)[1].all()

    unsnapped = BlockModelDefinition.auto_extent([solid, topo], size, margin=margin, snap=False)
    np.testing.assert_allclose(unsnapped.origin, pts.min(axis=0) - margin)
    assert unsnapped.world_to_ijk(pts)[1].all()

    # the solid alone with the truth margin reproduces the truth origin and plan shape
    truth = json.loads((DATA / "truth.json").read_text())["blocks"]
    d = BlockModelDefinition.auto_extent(solid, truth["size"], margin=40.0)
    assert list(d.origin) == truth["origin"]
    assert list(d.shape[:2]) == truth["shape"][:2]

    # per-axis margin, plain points, a single mesh, max_depth passed through
    d = BlockModelDefinition.auto_extent(pts, size, margin=(0.0, 20.0, 5.0), max_depth=2)
    assert d.max_depth == (2, 2, 2)
    lo, hi = d.bounding_box()  # with zero margin the extreme points may lie on the max faces
    assert (lo <= pts.min(axis=0) - (0.0, 20.0, 5.0)).all()
    assert (hi >= pts.max(axis=0) + (0.0, 20.0, 5.0)).all()
    # geometry that is an exact number of blocks gets an exact fit
    d = BlockModelDefinition.auto_extent([(0.0, 0.0, 0.0), (30.0, 20.0, 10.0)], (10.0, 10.0, 5.0))
    assert d.origin == (0.0, 0.0, 0.0) and d.shape == (3, 2, 2)
    with pytest.raises(ValueError):
        BlockModelDefinition.auto_extent(np.zeros((0, 3)), size)
    with pytest.raises(ValueError):
        BlockModelDefinition.auto_extent(pts, size, margin=-1.0)


def test_auto_extent_rotated_covers_geometry():
    solid, _ = _synthetic_geometry()
    size = np.array((10.0, 20.0, 5.0))
    d = BlockModelDefinition.auto_extent(solid, size, margin=15.0, angles=(30.0, 0.0, 55.0))
    local = d.world_to_local(solid.vertices)
    assert (local.min(axis=0) >= 15.0 - 1e-6).all()
    assert (local.max(axis=0) <= np.array(d.extent) - 15.0 + 1e-6).all()
    assert (local.min(axis=0) < 15.0 + size + 1e-6).all()
    assert (np.array(d.extent) - local.max(axis=0) < 15.0 + size + 1e-6).all()
    # a model aligned with the lens needs far fewer blocks than a world-aligned one
    plain = BlockModelDefinition.auto_extent(solid, size, margin=15.0)
    assert d.n_parents < plain.n_parents


# ------------------------------------------------------------------ truth


def test_reproduces_truth_block_centroids_exactly():
    truth = json.loads((DATA / "truth.json").read_text())["blocks"]
    table = pd.read_parquet(DATA / "truth_blocks.parquet")
    d = BlockModelDefinition(truth["origin"], truth["size"], truth["shape"])
    assert d.n_parents == len(table)
    ijk = table[["i", "j", "k"]].to_numpy()
    xyz = table[["x", "y", "z"]].to_numpy()
    # their row order is the documented Fortran-order number
    np.testing.assert_array_equal(d.flat_index_f(ijk), np.arange(len(table)))
    np.testing.assert_array_equal(d.ijk_to_world(ijk), xyz)  # exact
    # our flat order mapped to theirs
    ours = d.parent_centroids()
    np.testing.assert_array_equal(ours[d.flat_index(ijk)], xyz)
    np.testing.assert_array_equal(ours, xyz[d.flat_index_f(d.parent_ijk())])
    found, inside = d.world_to_ijk(xyz)
    assert inside.all()
    np.testing.assert_array_equal(found, ijk)
    np.testing.assert_array_equal(table[["dx", "dy", "dz"]].to_numpy()[0], d.block_size)


# ------------------------------------------------------------------ JSON


@pytest.mark.parametrize("rot", ROTATIONS)
def test_json_round_trip(rot):
    d = BlockModelDefinition((500_400.1, 5_300_250.7, 825.3), (10.0, 0.1, 1 / 3), (39, 5, 8),
                             max_depth=(3, 2, 0), **rot)  # fmt: skip
    text = d.to_json()
    back = BlockModelDefinition.from_json(text)
    assert back == d
    assert hash(back) == hash(d)
    np.testing.assert_array_equal(back.rotation, d.rotation)
    data = json.loads(text)
    assert set(data) == {"origin", "block_size", "shape", "max_depth", "rotation"}
    assert ("matrix" in data["rotation"]) == ("matrix" in rot)
    assert BlockModelDefinition.from_dict(d.to_dict()) == d
