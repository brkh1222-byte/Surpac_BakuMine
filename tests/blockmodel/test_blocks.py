"""P3.T2 acceptance: block table with octree sub-blocks."""

import time
from fractions import Fraction

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.estimation.rotation import rotation_matrix

ORIGIN = (500_400.0, 5_300_250.0, 825.0)
DEPTHS = [(1, 1, 1), (2, 2, 2), (3, 3, 3), (2, 2, 0), (3, 1, 2), (0, 0, 2), (1, 3, 0)]


def make(shape=(3, 2, 2), size=(10.0, 12.0, 5.0), depth=2, **kw):
    return BlockModel(BlockModelDefinition(ORIGIN, size, shape, max_depth=depth, **kw))


def raster_cover(model):
    """Independent tiling check: paint every block on the finest grid, count the paint."""
    d = model.definition
    depth = np.array(d.max_depth)
    cover = np.zeros(np.array(d.shape) << depth, dtype=np.int32)
    lo = (model.ijk() << depth) + model.sub
    units = 1 << (depth - np.minimum(model.level[:, None].astype(int), depth))
    for (a, b, c), (ua, ub, uc) in zip(lo.tolist(), units.tolist(), strict=True):
        cover[a : a + ua, b : b + ub, c : c + uc] += 1
    return cover


def brute_locate(model, points):
    """(n_points, n_blocks) bool: which half-open block boxes contain each point."""
    lo, hi = model.bounds_local()
    local = model.definition.world_to_local(points)
    return ((local[:, None, :] >= lo[None]) & (local[:, None, :] < hi[None])).all(axis=2)


def table(model):
    cols = [model.parent, model.level, model.sub[:, 0], model.sub[:, 1], model.sub[:, 2]]
    return np.column_stack([c.astype(np.int64) for c in cols])


# ------------------------------------------------------------------ creation


def test_full_model_of_parents():
    m = make()
    d = m.definition
    assert m.n_blocks == len(m) == 12 and m.is_dense
    assert m.parent.dtype == np.int32 and m.level.dtype == np.uint8 and m.sub.dtype == np.uint8
    np.testing.assert_array_equal(m.parent, np.arange(12))
    np.testing.assert_array_equal(m.centroids(), d.parent_centroids())
    np.testing.assert_array_equal(m.sizes(), np.tile(d.block_size, (12, 1)))
    np.testing.assert_array_equal(m.volumes(), 600.0)
    np.testing.assert_array_equal(m.ijk(), d.parent_ijk())
    m.check_invariants()
    assert not m.parent.flags.writeable
    assert m.nbytes == 12 * 8
    with pytest.raises(ValueError):
        m.parent[0] = 3


# ------------------------------------------------------------------ split


def test_split_makes_eight_children_in_documented_order():
    m = make(depth=2)
    m.add_attribute("au", background=0.5)
    m.add_attribute("code", "int", background=7)
    m["au"][:] = np.arange(12) * 1.5
    result = m.split([5])
    assert result.n_split == 1 and m.n_blocks == 19 and not m.is_dense
    np.testing.assert_array_equal(result.source, [0, 1, 2, 3, 4, *[5] * 8, 6, 7, 8, 9, 10, 11])
    np.testing.assert_array_equal(np.flatnonzero(result.is_child), np.arange(5, 13))
    kids = slice(5, 13)
    np.testing.assert_array_equal(m.level[kids], 1)
    np.testing.assert_array_equal(m.parent[kids], 5)
    # children in (i, j, k) order, k fastest; sub counted in finest cells (quarter blocks)
    expect = [(i, j, k) for i in (0, 2) for j in (0, 2) for k in (0, 2)]
    np.testing.assert_array_equal(m.sub[kids], expect)
    np.testing.assert_array_equal(m.sizes()[kids], np.tile((5.0, 6.0, 2.5), (8, 1)))
    # all attributes inherited
    np.testing.assert_array_equal(m["au"][kids], 7.5)
    np.testing.assert_array_equal(m["code"][kids], 7)
    np.testing.assert_array_equal(m["au"], (np.arange(12) * 1.5)[result.source])
    # centroids of the children: parent centroid +- a quarter block
    parent_c = m.definition.ijk_to_world(m.definition.unflat_index(5))
    offs = np.array([(i, j, k) for i in (-1, 1) for j in (-1, 1) for k in (-1, 1)])
    np.testing.assert_array_equal(m.centroids()[kids], parent_c + offs * (2.5, 3.0, 1.25))
    m.check_invariants()

    # split one child again, by mask
    mask = np.zeros(m.n_blocks, dtype=bool)
    mask[6] = True
    m.split(mask)
    assert m.n_blocks == 26
    np.testing.assert_array_equal(m.level[6:14], 2)
    expect = [(i, j, 2 + k) for i in (0, 1) for j in (0, 1) for k in (0, 1)]
    np.testing.assert_array_equal(m.sub[6:14], expect)
    m.check_invariants()
    # deepest level cannot be split
    with pytest.raises(ValueError, match="deepest level"):
        m.split([6])
    assert m.split([6], strict=False).n_split == 0
    assert m.split([0, 6], strict=False).n_split == 1
    assert m.split([]).n_split == 0
    with pytest.raises(IndexError):
        m.split([10_000])
    with pytest.raises(ValueError):
        m.split(np.zeros(3, dtype=bool))


def test_anisotropic_depth_never_splits_shallow_axis():
    m = make(depth=(2, 1, 0))
    r = m.split([0])
    assert r.n_split == 1 and m.n_blocks == 12 + 3  # x and y halved: 4 children
    np.testing.assert_array_equal(m.sizes()[:4], np.tile((5.0, 6.0, 5.0), (4, 1)))
    np.testing.assert_array_equal(m.sub[:4], [(0, 0, 0), (0, 1, 0), (2, 0, 0), (2, 1, 0)])
    m.split([0])  # level 1 -> 2: only x is halved: 2 children
    assert m.n_blocks == 16
    np.testing.assert_array_equal(m.sizes()[:2], np.tile((2.5, 6.0, 5.0), (2, 1)))
    np.testing.assert_array_equal(m.sub[:2], [(0, 0, 0), (1, 0, 0)])
    m.check_invariants()
    assert (raster_cover(m) == 1).all()
    assert m.merge() == 2 and m.is_dense
    # a model without sub-blocking cannot split at all
    flat = make(depth=0)
    with pytest.raises(ValueError):
        flat.split([0])
    assert flat.merge() == 0


# ------------------------------------------------------------------ merge


def test_merge_only_when_all_siblings_exist_with_identical_values():
    m = make(depth=2)
    m.add_attribute("au", background=1.0)
    m.add_attribute("rock", "int", background=0)
    before = table(m)
    m.split([2, 7])
    m.split([3])  # a grandchild set inside parent 2
    assert m.n_blocks == 12 + 7 + 7 + 7
    # identical values: everything merges back, recursively
    assert m.merge() == 3
    np.testing.assert_array_equal(table(m), before)
    np.testing.assert_array_equal(m["au"], 1.0)

    m.split([2, 7])
    m["au"][3] = 2.0  # one child of parent 2 differs
    assert m.merge() == 1  # only parent 7's children merge
    assert m.n_blocks == 19 and (m.parent[m.level == 1] == 2).all()
    assert m.merge(["rock"]) == 1  # merging on "rock" only: allowed, first child's au kept
    assert m.is_dense and m["au"][2] == 1.0

    # NaN equals NaN for merging; a NaN next to a number does not
    m.split([4])
    m["au"][4:12] = np.nan
    assert m.merge() == 1
    assert np.isnan(m["au"][4])
    m.split([4])
    m["au"][5] = 3.0
    assert m.merge() == 0

    # incomplete sibling set (one sibling split further) does not merge until it is whole
    m = make(depth=2)
    m.add_attribute("au", background=1.0)
    m.split([0])
    m.split([3])
    m["au"][3] = 9.0  # one grandchild differs
    assert m.merge() == 0
    m["au"][3] = 1.0
    assert m.merge(recursive=False) == 1 and m.n_blocks == 19
    assert m.merge(recursive=False) == 1 and m.is_dense

    # where: only fully selected sets
    m.split([0, 1])
    sel = np.zeros(m.n_blocks, dtype=bool)
    sel[:9] = True  # children of 0 and the first child of 1
    assert m.merge(where=sel) == 1
    assert m.n_blocks == 12 + 7 and (m.parent[m.level == 1] == 1).all()
    assert m.merge([]) == 1
    with pytest.raises(KeyError):
        m.merge(["nope"])


# ------------------------------------------------------------------ exactness


@pytest.mark.parametrize("depth", [(3, 3, 3), (3, 1, 2), (8, 8, 8)])
def test_centroids_and_sizes_are_exact(depth):
    size = (0.1, 1.0 / 3.0, 7.3)  # not representable: any accumulation would show
    m = make(shape=(7, 5, 3), size=size, depth=depth)
    rng = np.random.default_rng(3)
    for _ in range(max(depth)):
        pick = rng.random(m.n_blocks) < min(0.3, 3000 / m.n_blocks)
        m.split(pick, strict=False)
    assert m.level.max() == max(depth)
    rows = rng.choice(m.n_blocks, 400, replace=False)
    local = m.centroids_local()
    sizes = m.sizes()
    lo, hi = m.bounds_local()
    ijk = m.ijk()
    for r in rows.tolist():
        for a in range(3):
            d = Fraction(size[a])
            lev = min(int(m.level[r]), depth[a])
            exact_size = d / 2**lev
            exact_lo = int(ijk[r, a]) * d + int(m.sub[r, a]) * d / 2 ** depth[a]
            assert Fraction(float(sizes[r, a])) == exact_size
            # correctly rounded value of the exact rational
            assert local[r, a] == float(exact_lo + exact_size / 2)
            assert lo[r, a] == float(exact_lo)
            assert hi[r, a] == float(exact_lo + exact_size)
    np.testing.assert_array_equal(m.centroids(), m.definition.local_to_world(local))
    np.testing.assert_array_equal(m.centroids(rows), m.centroids()[rows])
    np.testing.assert_array_equal(m.volumes(), sizes.prod(axis=1))
    # relative-to-first-centroid coordinates of parents are exactly i * d
    parents = np.flatnonzero(m.level == 0)
    np.testing.assert_array_equal(m.centroids_grid(parents), m.ijk(parents) * np.array(size))
    m.check_invariants()


def test_corners_match_bounds():
    m = make(depth=1, angles=(30.0, 10.0, -20.0))
    m.split([3])
    corners = m.corners()
    assert corners.shape == (m.n_blocks, 8, 3)
    lo, hi = m.bounds_local()
    np.testing.assert_allclose(m.definition.world_to_local(corners[:, 0]), lo, atol=1e-7)
    np.testing.assert_allclose(m.definition.world_to_local(corners[:, 7]), hi, atol=1e-7)
    np.testing.assert_allclose(corners.mean(axis=1), m.centroids(), atol=1e-7)


# ------------------------------------------------------------------ ordering / ids


def test_ordering_is_deterministic_and_history_independent():
    a = make(depth=3)
    b = make(depth=3)
    ids = a.block_ids().copy()
    # a: split parents 1, 4, 9, then two of the children
    a.split(a.rows_of(ids[[1, 4, 9]]))
    grandparents = a.block_ids()[[2, 30]].copy()
    a.split(a.rows_of(grandparents))
    # b: the same refinement in another order, with a detour (split parent 0, merge it back)
    b.add_attribute("tag", "int")
    b.split(b.rows_of(ids[[9]]))
    b.split(b.rows_of(ids[[0]]))
    b.split(b.rows_of(ids[[4]]))
    assert b.rows_of(grandparents[[0]])[0] == -1  # its parent (1) is not split yet
    b.split(b.rows_of(grandparents[[1]]))
    b.split(b.rows_of(ids[[1]]))
    b.split(b.rows_of(grandparents[[0]]))
    b.set("tag", 1, b.parent != 0)
    b.merge(where=b.parent == 0)
    np.testing.assert_array_equal(table(a), table(b))
    np.testing.assert_array_equal(a.block_ids(), b.block_ids())
    np.testing.assert_array_equal(a.centroids(), b.centroids())
    # rows are sorted by key = (parent, Morton); keys strictly increase
    keys = a.keys()
    assert (np.diff(keys) > 0).all() and not keys.flags.writeable
    assert (np.diff(a.parent.astype(int)) >= 0).all()
    # ids are stable for blocks that were not touched, and rows_of finds them
    survivors = np.isin(ids, a.block_ids())
    assert survivors.sum() == 12 - 3
    rows = a.rows_of(ids)
    assert (rows[survivors] >= 0).all() and (rows[~survivors] == -1).all()
    np.testing.assert_array_equal(a.block_ids(rows[survivors]), ids[survivors])
    np.testing.assert_array_equal(a.rows_of(a.block_ids()), np.arange(a.n_blocks))
    # from_arrays accepts the table and rejects a shuffled or incomplete one
    c = BlockModel.from_arrays(a.definition, a.parent, a.level, a.sub)
    np.testing.assert_array_equal(table(c), table(a))
    perm = np.arange(a.n_blocks)
    perm[[0, 1]] = [1, 0]
    with pytest.raises(ValueError, match="order"):
        BlockModel.from_arrays(a.definition, a.parent[perm], a.level[perm], a.sub[perm])
    with pytest.raises(ValueError, match="fill"):
        BlockModel.from_arrays(a.definition, a.parent[1:], a.level[1:], a.sub[1:])
    bad_level = a.level.copy()
    child = np.flatnonzero(bad_level == 1)[0]
    bad_level[child] = 0  # a child claiming to be a whole parent: overlaps its siblings
    with pytest.raises(ValueError):
        BlockModel.from_arrays(a.definition, a.parent, bad_level, a.sub)


# ------------------------------------------------------------------ locate


@pytest.mark.parametrize("rot", [{}, {"angles": (40.0, -15.0, 25.0)}])
@pytest.mark.parametrize("depth", [(2, 2, 2), (3, 1, 0)])
def test_locate_equals_brute_force(rot, depth):
    m = make(shape=(4, 3, 3), depth=depth, **rot)
    rng = np.random.default_rng(11)
    for _ in range(max(depth)):
        m.split(rng.random(m.n_blocks) < 0.35, strict=False)
    lo, hi = m.definition.bounding_box()
    pts = rng.uniform(lo - 3.0, hi + 3.0, (4000, 3))
    # add points exactly on block corners / faces (exact in the unrotated case)
    blo, bhi = m.bounds_local()
    face_pts = m.definition.local_to_world(np.concatenate([blo, bhi, 0.5 * (blo + bhi)]))
    pts = np.concatenate([pts, face_pts])
    rows = m.locate(pts)
    inside = brute_locate(m, pts)
    count = inside.sum(axis=1)
    in_model = m.definition.world_to_ijk(pts)[1]
    np.testing.assert_array_equal(count, in_model.astype(int))  # exactly one leaf, or none
    np.testing.assert_array_equal(rows >= 0, in_model)
    np.testing.assert_array_equal(rows[in_model], inside[in_model].argmax(axis=1))
    assert 0.2 < in_model.mean() < 1.0
    # each block's centroid is in that block; scalar input works
    np.testing.assert_array_equal(m.locate(m.centroids()), np.arange(m.n_blocks))
    assert m.locate(m.centroids()[5]) == 5
    assert m.locate((0.0, 0.0, 0.0)) == -1


def test_locate_dense_fast_path():
    m = make(shape=(5, 4, 3), depth=2)
    np.testing.assert_array_equal(m.locate(m.centroids()), np.arange(60))
    d = m.definition
    pts = np.random.default_rng(0).uniform(*d.bounding_box(), (1000, 3))
    ijk, _ = d.world_to_ijk(pts)
    np.testing.assert_array_equal(m.locate(pts), d.flat_index(ijk))


# ------------------------------------------------------------------ property test


@st.composite
def operations(draw):
    depth = draw(st.sampled_from(DEPTHS))
    shape = draw(st.tuples(st.integers(1, 3), st.integers(1, 3), st.integers(1, 2)))
    ops = draw(
        st.lists(
            st.tuples(
                st.sampled_from(["split", "split", "merge", "merge_subset", "assign"]),
                st.integers(0, 2**31 - 1),
                st.floats(0.05, 0.9),
            ),
            min_size=1,
            max_size=8,
        )
    )
    rotated = draw(st.booleans())
    return depth, shape, ops, rotated


@settings(max_examples=120, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(operations())
def test_any_split_merge_sequence_keeps_an_exact_tiling(case):
    depth, shape, ops, rotated = case
    rot = {"matrix": tuple(map(tuple, rotation_matrix(25.0, 10.0, -30.0)))} if rotated else {}
    m = make(shape=shape, size=(10.0, 0.3, 2.5), depth=depth, **rot)
    m.add_attribute("au", background=1.0)
    m.add_attribute("rock", "int", background=1, dtype=np.int8)
    d = m.definition
    total_units = d.n_parents << sum(depth)
    total_volume = d.volume
    metal = float((m["au"] * m.volumes()).sum())
    for kind, seed, frac in ops:
        rng = np.random.default_rng(seed)
        if m.n_blocks > 3000:
            kind = "merge"
        before_n = m.n_blocks
        if kind == "split":
            pick = rng.random(m.n_blocks) < frac
            au, rock, ids = m["au"].copy(), m["rock"].copy(), m.block_ids().copy()
            result = m.split(pick, strict=False)
            # children inherit every attribute; untouched blocks keep id and values
            np.testing.assert_array_equal(m["au"], au[result.source])
            np.testing.assert_array_equal(m["rock"], rock[result.source])
            same = ~result.is_child
            np.testing.assert_array_equal(m.block_ids()[same], ids[result.source[same]])
            assert m.n_blocks >= before_n
        elif kind == "assign":
            pick = rng.random(m.n_blocks) < frac
            m.set("au", rng.integers(1, 4, int(pick.sum())).astype(float), pick)
            m.set("rock", rng.integers(1, 3, int(pick.sum())), pick)
            metal = float((m["au"] * m.volumes()).sum())
        else:
            names = ["rock"] if kind == "merge_subset" else None
            au, rock = m["au"].copy(), m["rock"].copy()
            lo_before, hi_before = m.bounds_local()
            merged = m.merge(names, recursive=bool(seed & 1))
            assert (merged == 0) == (m.n_blocks == before_n)
            # the old blocks inside every surviving or merged block had identical values
            lo, hi = m.bounds_local()
            centre = 0.5 * (lo_before + hi_before)
            contains = (centre[:, None, :] >= lo[None]) & (centre[:, None, :] < hi[None])
            owner = contains.all(axis=2).argmax(axis=1)
            assert (rock == m["rock"][owner]).all()
            if names is None:
                np.testing.assert_array_equal(au, m["au"][owner])
            else:
                metal = float((m["au"] * m.volumes()).sum())
        # --- invariants after every operation
        m.check_invariants()
        assert int(m.volume_units().sum()) == total_units  # exact integer accounting
        assert float(m.volumes().sum()) == pytest.approx(total_volume, rel=1e-12)
        cover = raster_cover(m)
        assert cover.min() == 1 and cover.max() == 1  # no gaps, no overlaps
        assert (np.diff(m.keys()) > 0).all()
        assert len(m["au"]) == len(m["rock"]) == m.n_blocks
        # splits and all-attribute merges preserve the volume-weighted sum
        assert float((m["au"] * m.volumes()).sum()) == pytest.approx(metal, rel=1e-12)
    # random points: exactly one leaf contains each, and locate finds it
    rng = np.random.default_rng(len(ops))
    local = rng.uniform(0.0, 1.0, (300, 3)) * np.array(d.extent)
    pts = d.local_to_world(local)
    inside = brute_locate(m, pts)
    in_model = d.world_to_ijk(pts)[1]
    np.testing.assert_array_equal(inside.sum(axis=1), in_model.astype(int))
    rows = m.locate(pts)
    np.testing.assert_array_equal(rows[in_model], inside[in_model].argmax(axis=1))
    # merging on structure alone always returns to the parents
    m.merge([])
    assert m.is_dense
    np.testing.assert_array_equal(m.parent, np.arange(d.n_parents))


# ------------------------------------------------------------------ dense fast path


def test_dense_view_shares_memory():
    m = make(shape=(4, 3, 5), depth=1)
    m.add_attribute("au", background=0.0)
    view = m.dense_view("au")
    assert view.shape == (4, 3, 5)
    assert np.shares_memory(view, m["au"]) and view.base is not None
    view[2, 1, 3] = 9.0
    flat = int(m.definition.flat_index((2, 1, 3)))
    assert m["au"][flat] == 9.0 and m.get("au")[flat] == 9.0
    m["au"][0] = 4.0
    assert view[0, 0, 0] == 4.0
    # view[i, j, k] is the block with parent ijk (i, j, k)
    i, j, k = np.unravel_index(np.arange(60), (4, 3, 5))
    np.testing.assert_array_equal(np.column_stack([i, j, k]), m.ijk())
    m.split([0])
    with pytest.raises(ValueError, match="without sub-blocks"):
        m.dense_view("au")
    m.merge()
    assert np.shares_memory(m.dense_view("au"), m["au"])


# ------------------------------------------------------------------ memory


def test_five_million_blocks_fit_in_two_gigabytes(capsys):
    """~5M blocks (parents + two levels of sub-blocks) with 6 float64 attributes."""
    t0 = time.perf_counter()
    m = BlockModel(BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (150, 150, 100), max_depth=2))
    t_create = time.perf_counter() - t0
    for n in range(6):
        m.add_attribute(f"a{n}", background=float(n))
    rng = np.random.default_rng(0)
    t0 = time.perf_counter()
    m.split(rng.random(m.n_blocks) < 0.12)
    m.split((m.level == 1) & (rng.random(m.n_blocks) < 0.12))
    t_split = time.perf_counter() - t0
    n_blocks = m.n_blocks
    assert n_blocks >= 5_000_000
    assert set(np.unique(m.level).tolist()) == {0, 1, 2}
    t0 = time.perf_counter()
    centroids = m.centroids()
    t_centroids = time.perf_counter() - t0
    t0 = time.perf_counter()
    rows = m.locate(centroids[::50])
    t_locate = time.perf_counter() - t0
    np.testing.assert_array_equal(rows, np.arange(n_blocks)[::50])
    t0 = time.perf_counter()
    m.check_invariants()
    t_check = time.perf_counter() - t0
    structure = m.parent.nbytes + m.level.nbytes + m.sub.nbytes
    columns = m.nbytes
    total = columns + m.cache_nbytes  # + cached keys and world centroids
    assert structure == 8 * n_blocks
    assert columns == (8 + 6 * 8) * n_blocks
    assert total < 2 * 1024**3
    t0 = time.perf_counter()
    merged = m.merge()
    t_merge = time.perf_counter() - t0
    assert m.is_dense and merged > 0
    with capsys.disabled():
        print(
            f"\n[P3.T2 memory] {n_blocks:,} blocks, 6 float64 attributes: columns "
            f"{columns / 1e6:.0f} MB (structure {structure / 1e6:.0f} MB), with cached keys + "
            f"centroids {total / 1e6:.0f} MB; create {t_create:.2f} s, two splits "
            f"{t_split:.2f} s, centroids {t_centroids:.2f} s, locate {len(rows):,} points "
            f"{t_locate:.2f} s, check_invariants {t_check:.2f} s, merge all {t_merge:.2f} s"
        )
