"""Acceptance tests for P4.T2 anisotropic search (research.md 7.5, 7.6).

Every option combination is compared with ``BruteForce`` below: a plain
per-target Python implementation of the selection rule that shares no code
with ``search.py`` (explicit axis projections and the research.md 7.5
distance formula, full sort, Python greedy loop; no KD-tree, no isotropic
pre-transform).
"""

import time

import numpy as np
import pytest

from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SampleSearch, SearchParams, SearchResult


# ----------------------------------------------------------------- reference
class BruteForce:
    """Reference search. Candidate lists (all samples within the ellipsoid's
    major radius, ordered by (distance, sample index)) are computed once per
    target; ``run`` applies the options with a Python loop per target."""

    def __init__(self, samples, holes, targets, ellipsoid, chunk=500):
        self.samples, self.holes, self.targets = samples, holes, targets
        self.radius = ellipsoid.major_radius
        major, semi, minor = ellipsoid.major_axis, ellipsoid.semi_major_axis, ellipsoid.minor_axis
        r_semi, r_minor = ellipsoid.semi_major_ratio, ellipsoid.minor_ratio
        self.lists = []
        for start in range(0, len(targets), chunk):
            diff = samples[None, :, :] - targets[start : start + chunk, None, :]
            u = diff @ major  # along the major axis
            v = diff @ semi  # along the semi-major axis
            w = diff @ minor  # along the minor axis
            d = np.sqrt(u**2 + (v * r_semi) ** 2 + (w * r_minor) ** 2)
            dz = np.abs(diff[:, :, 2])
            for t in range(d.shape[0]):
                cand = np.flatnonzero(d[t] <= self.radius)
                order = np.lexsort((cand, d[t, cand]))  # by distance, then index
                cand = cand[order]
                self.lists.append(
                    (
                        cand,
                        d[t, cand],
                        dz[t, cand],
                        v[t, cand] >= 0,  # x' = semi-major
                        u[t, cand] >= 0,  # y' = major
                        w[t, cand] >= 0,  # z' = minor
                    )
                )

    def run(
        self,
        max_samples=24,
        min_samples=1,
        radius=None,
        max_per_hole=None,
        sectors=None,
        max_per_sector=None,
        min_sectors=None,
        max_vertical=None,
        exclude_zero_distance=False,
    ):
        m = len(self.targets)
        radius = self.radius if radius is None else radius
        assert radius <= self.radius
        idx = np.full((m, max_samples), -1, dtype=np.int64)
        dist = np.full((m, max_samples), np.inf)
        counts = np.zeros(m, dtype=np.int32)
        informed = np.zeros(m, dtype=np.int8)
        for t, (cand, d, dz, xpos, ypos, zpos) in enumerate(self.lists):
            if sectors == "octant":
                sector = xpos + 2 * ypos + 4 * zpos
            elif sectors == "quadrant":
                sector = xpos + 2 * ypos
            else:
                sector = np.zeros(len(cand), dtype=int)
            per_hole, per_sector, n = {}, {}, 0
            for j in range(len(cand)):
                if d[j] > radius:
                    break
                if exclude_zero_distance and d[j] == 0.0:
                    continue
                if max_vertical is not None and dz[j] > max_vertical:
                    continue
                s = int(sector[j])
                if max_per_sector is not None and per_sector.get(s, 0) >= max_per_sector:
                    continue
                h = self.holes[cand[j]]
                if max_per_hole is not None and per_hole.get(h, 0) >= max_per_hole:
                    continue
                idx[t, n], dist[t, n] = cand[j], d[j]
                per_hole[h] = per_hole.get(h, 0) + 1
                per_sector[s] = per_sector.get(s, 0) + 1
                n += 1
                if n == max_samples:
                    break
            counts[t] = n
            informed[t] = len(per_sector) if sectors else 0
        estimated = counts >= min_samples
        if min_sectors is not None:
            estimated &= informed >= min_sectors
        return idx, dist, counts, informed, estimated


def assert_same(result, reference):
    idx, dist, counts, informed, estimated = reference
    assert isinstance(result, SearchResult)
    assert result.indices.dtype == np.int64 and result.distances.dtype == np.float64
    np.testing.assert_array_equal(result.counts, counts)
    np.testing.assert_array_equal(result.indices, idx)
    np.testing.assert_allclose(result.distances, dist, rtol=1e-9, atol=1e-9)
    np.testing.assert_array_equal(result.sectors_informed, informed)
    np.testing.assert_array_equal(result.estimated, estimated)


# -------------------------------------------------------------------- dataset
def make_holes(rng, n_holes=80, per_hole=50, spacing=2.0):
    """Clustered samples: points strung down inclined 'holes'."""
    collars = np.column_stack(
        [rng.uniform(0, 300, n_holes), rng.uniform(0, 300, n_holes), rng.uniform(140, 150, n_holes)]
    )
    azimuth = rng.uniform(0, 2 * np.pi, n_holes)
    dip = np.radians(rng.uniform(50, 90, n_holes))
    direction = np.column_stack(
        [np.sin(azimuth) * np.cos(dip), np.cos(azimuth) * np.cos(dip), -np.sin(dip)]
    )
    # Irregular intervals: with equal spacing, a target placed on a sample has
    # its up-hole and down-hole neighbours at mathematically equal distance,
    # and the comparison would then hinge on last-bit rounding. Exact ties
    # are tested separately on a lattice.
    depth = np.cumsum(rng.uniform(0.75, 1.25, (n_holes, per_hole)) * spacing, axis=1)
    samples = (collars[:, None, :] + depth[:, :, None] * direction[:, None, :]).reshape(-1, 3)
    holes = np.repeat([f"DH{k:03d}" for k in range(n_holes)], per_hole)
    perm = rng.permutation(len(samples))  # sample order unrelated to holes
    return samples[perm], holes[perm]


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(20240)
    samples, holes = make_holes(rng)
    targets = np.column_stack(
        [rng.uniform(-20, 320, 10_000), rng.uniform(-20, 320, 10_000), rng.uniform(30, 160, 10_000)]
    )
    # 500 of the 10k targets sit exactly on samples (zero distance cases)
    targets[:500] = samples[rng.choice(len(samples), 500, replace=False)]
    ellipsoid = Ellipsoid(60.0, 1.5, 3.0, angles=(30.0, 20.0, 10.0))
    search = SampleSearch(samples, ellipsoid, hole_ids=holes)
    reference = BruteForce(samples, holes, targets, ellipsoid)
    return search, reference, targets


COMBINATIONS = {
    "plain": dict(max_samples=12),
    "plain_default_24": dict(),
    "plain_max1": dict(max_samples=1),
    "plain_more_than_available": dict(max_samples=150, min_samples=20),
    "radius": dict(max_samples=12, radius=35.0),
    "min_samples": dict(max_samples=12, min_samples=6, radius=30.0),
    "max_per_hole": dict(max_samples=12, max_per_hole=2),
    "max_per_hole_1": dict(max_samples=8, max_per_hole=1, min_samples=3),
    "octant": dict(max_samples=12, sectors="octant", max_per_sector=2),
    "octant_min_sectors": dict(max_samples=16, sectors="octant", max_per_sector=2, min_sectors=5),
    "octant_only_min_sectors": dict(max_samples=10, sectors="octant", min_sectors=4),
    "quadrant": dict(max_samples=12, sectors="quadrant", max_per_sector=3),
    "quadrant_min_sectors": dict(
        max_samples=8, sectors="quadrant", max_per_sector=2, min_sectors=3
    ),
    "max_vertical": dict(max_samples=12, max_vertical=6.0),
    "exclude_zero": dict(max_samples=12, exclude_zero_distance=True),
    "hole+octant": dict(max_samples=12, max_per_hole=2, sectors="octant", max_per_sector=2),
    "hole+vertical+min": dict(max_samples=10, max_per_hole=3, max_vertical=10.0, min_samples=4),
    "quadrant+vertical+radius": dict(
        max_samples=12, sectors="quadrant", max_per_sector=4, max_vertical=12.0, radius=45.0
    ),
    "everything": dict(
        max_samples=14,
        min_samples=4,
        radius=50.0,
        max_per_hole=3,
        sectors="octant",
        max_per_sector=3,
        min_sectors=3,
        max_vertical=15.0,
        exclude_zero_distance=True,
    ),
    "everything_quadrant": dict(
        max_samples=6,
        min_samples=2,
        max_per_hole=1,
        sectors="quadrant",
        max_per_sector=2,
        min_sectors=2,
        max_vertical=20.0,
        exclude_zero_distance=True,
    ),
}


@pytest.mark.parametrize("name", list(COMBINATIONS))
def test_equals_brute_force(case, name):
    search, reference, targets = case
    options = COMBINATIONS[name]
    assert len(targets) == 10_000
    result = search.query(targets, SearchParams(**options))
    expected = reference.run(**options)
    assert_same(result, expected)
    # the case must actually exercise the option (not all empty / all full)
    assert 0 < result.estimated.sum()
    assert (result.counts == 0).any() and (result.counts > 0).any()


def test_case_exercises_the_constraints(case):
    """Guard against vacuous comparisons: each option changes the answer."""
    search, _, targets = case
    plain = search.query(targets, max_samples=12)
    for name in ("max_per_hole", "octant", "quadrant", "max_vertical", "exclude_zero", "radius"):
        other = search.query(targets, **COMBINATIONS[name])
        changed = (other.indices != plain.indices).any(axis=1).mean()
        assert changed > 0.04, (name, changed)
    assert not search.query(targets, **COMBINATIONS["min_samples"]).estimated.all()
    informed = search.query(targets, **COMBINATIONS["octant_min_sectors"])
    assert 0.02 < informed.estimated.mean() < 0.98


def test_result_layout_and_invariants(case):
    search, _, targets = case
    p = SearchParams(max_samples=10, max_per_hole=2, sectors="octant", max_per_sector=2)
    r = search.query(targets, p)
    m = len(targets)
    assert r.indices.shape == r.distances.shape == (m, 10)
    assert r.counts.shape == r.sectors_informed.shape == r.estimated.shape == (m,)
    cols = np.arange(10)[None, :]
    valid = cols < r.counts[:, None]
    assert (r.indices[valid] >= 0).all() and (r.indices[~valid] == -1).all()
    assert np.isfinite(r.distances[valid]).all() and np.isinf(r.distances[~valid]).all()
    assert (r.distances[valid] <= search.ellipsoid.major_radius).all()
    padded = np.where(valid, r.distances, np.inf)
    assert (padded[:, 1:] >= padded[:, :-1]).all()  # nearest first
    # distances are the ellipsoid's anisotropic distances
    rows, slots = np.nonzero(valid)
    np.testing.assert_allclose(
        r.distances[rows, slots],
        search.ellipsoid.distance(search.samples[r.indices[rows, slots]], targets[rows]),
        rtol=1e-9,
        atol=1e-9,
    )
    # no hole contributes more than 2 samples to any target
    codes = np.where(valid, search.hole_codes[r.indices], -1 - cols)
    codes.sort(axis=1)
    run2 = (codes[:, 2:] == codes[:, :-2]) & (codes[:, 2:] >= 0)
    assert not run2.any()


def test_chunking_and_workers_do_not_change_results(case):
    search, _, targets = case
    p = SearchParams(max_samples=9, max_per_hole=2, sectors="quadrant", max_per_sector=3)
    base = search.query(targets[:3000], p)
    runs = [
        (300, dict(chunk_size=1)),
        (3000, dict(chunk_size=3)),
        (3000, dict(chunk_size=777, workers=1)),
        (3000, dict(chunk_size=10**7)),
    ]
    for k, kwargs in runs:
        other = search.query(targets[:k], p, **kwargs)
        np.testing.assert_array_equal(other.indices, base.indices[:k])
        np.testing.assert_array_equal(other.distances, base.distances[:k])
        np.testing.assert_array_equal(other.estimated, base.estimated[:k])


# ----------------------------------------------------------------------- ties
@pytest.mark.parametrize(
    "ellipsoid",
    [Ellipsoid(2.5), Ellipsoid(4.0, 2.0, 4.0)],  # scale factors exact in binary
    ids=["isotropic", "anisotropic"],
)
@pytest.mark.parametrize(
    "options",
    [
        dict(max_samples=4),
        dict(max_samples=1),
        dict(max_samples=7),
        dict(max_samples=30),
        dict(max_samples=5, exclude_zero_distance=True),
        dict(max_samples=6, max_per_hole=1),
        dict(max_samples=8, sectors="octant", max_per_sector=1),
        dict(max_samples=6, sectors="quadrant", max_per_sector=2, min_sectors=4),
        dict(max_samples=5, max_vertical=0.5, max_per_hole=2),
    ],
)
def test_distance_ties_broken_by_sample_index(ellipsoid, options):
    """Tie-break: equal distances are ordered by ascending sample index.

    Samples on an integer lattice and targets on lattice nodes, edge
    midpoints and cell centres give many EXACT ties (6 neighbours at 1, 12 at
    sqrt 2, 8 at sqrt(3)/2, ...), including ties that straddle the
    ``max_samples`` cut-off, sector boundaries (offsets of exactly 0) and
    zero distances.
    """
    rng = np.random.default_rng(5)
    g = np.arange(7, dtype=float)
    samples = np.array(np.meshgrid(g, g, g, indexing="ij")).reshape(3, -1).T
    samples = samples[rng.permutation(len(samples))]
    holes = (samples[:, 0] * 7 + samples[:, 1]).astype(int)  # vertical "holes"
    targets = np.vstack(
        [
            samples[:60],
            samples[60:120] + [0.5, 0.0, 0.0],
            samples[120:180] + [0.5, 0.5, 0.5],
            samples[180:240] + [0.0, 0.5, 0.5],
        ]
    )
    search = SampleSearch(samples, ellipsoid, hole_ids=holes)
    result = search.query(targets, **options)
    expected = BruteForce(samples, holes, targets, ellipsoid).run(**options)
    np.testing.assert_array_equal(result.indices, expected[0])
    np.testing.assert_array_equal(result.distances, expected[1])  # exact
    np.testing.assert_array_equal(result.counts, expected[2])
    np.testing.assert_array_equal(result.sectors_informed, expected[3])
    np.testing.assert_array_equal(result.estimated, expected[4])
    # ties really occur, and inside each tie group indices ascend
    d, i = result.distances, result.indices
    tie = (d[:, 1:] == d[:, :-1]) & np.isfinite(d[:, 1:])
    if options["max_samples"] > 1:
        assert tie.any()
    assert (i[:, 1:][tie] > i[:, :-1][tie]).all()
    # deterministic: identical on repeat, single-threaded and re-chunked
    again = search.query(targets, **options, workers=1, chunk_size=7)
    np.testing.assert_array_equal(again.indices, result.indices)


def test_explicit_tie_example():
    # four samples at distance exactly 1 from the origin, given out of order
    samples = np.array(
        [[0.0, -1.0, 0.0], [5.0, 5.0, 5.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]]
    )
    search = SampleSearch(samples, Ellipsoid(2.0))
    r = search.query(np.zeros((1, 3)), max_samples=2)
    np.testing.assert_array_equal(r.indices, [[0, 2]])  # lowest indices win
    np.testing.assert_array_equal(r.distances, [[1.0, 1.0]])
    r = search.query(np.zeros((1, 3)), max_samples=6)
    np.testing.assert_array_equal(r.indices, [[0, 2, 3, 4, -1, -1]])
    assert r.counts[0] == 4


# ---------------------------------------------------------------- small cases
def test_radius_is_inclusive_and_anisotropic():
    # ellipsoid radii 10 (north), 5 (east), 2 (up)
    e = Ellipsoid(10.0, 2.0, 5.0)
    samples = np.array(
        [[0, 10.0, 0], [0, 10.001, 0], [5.0, 0, 0], [5.001, 0, 0], [0, 0, 2.0], [0, 0, -2.001]]
    )
    r = SampleSearch(samples, e).query(np.zeros((1, 3)), max_samples=6)
    assert sorted(r.indices[0, : r.counts[0]]) == [0, 2, 4]
    np.testing.assert_allclose(r.distances[0, :3], 10.0)
    # rotated: bearing 90 puts the long axis east
    e90 = Ellipsoid(10.0, 2.0, 5.0, angles=(90.0, 0.0, 0.0))
    samples = np.array([[9.0, 0, 0], [0, 9.0, 0], [0, 4.0, 0]])
    r = SampleSearch(samples, e90).query(np.zeros((1, 3)), max_samples=3)
    np.testing.assert_array_equal(r.indices, [[2, 0, -1]])
    np.testing.assert_allclose(r.distances[0, :2], [8.0, 9.0])


def test_max_vertical_uses_world_z_not_ellipsoid_axes():
    # steeply plunging ellipsoid: world dz is not the minor-axis offset
    e = Ellipsoid(50.0, 1.0, 1.0, angles=(0.0, 60.0, 0.0))
    samples = np.array([[0, 0, 3.0], [0, 0, -4.0], [3.5, 0, 0], [0, 0, 2.0]])
    r = SampleSearch(samples, e).query(np.zeros((1, 3)), max_samples=4, max_vertical=3.0)
    np.testing.assert_array_equal(r.indices, [[3, 0, 2, -1]])


def test_min_samples_flags_but_keeps_neighbours():
    samples = np.array([[1.0, 0, 0], [2.0, 0, 0], [50.0, 0, 0]])
    search = SampleSearch(samples, Ellipsoid(5.0))
    targets = np.array([[0.0, 0, 0], [49.0, 0, 0], [200.0, 0, 0]])
    r = search.query(targets, max_samples=4, min_samples=2)
    np.testing.assert_array_equal(r.counts, [2, 1, 0])
    np.testing.assert_array_equal(r.estimated, [True, False, False])
    np.testing.assert_array_equal(r.indices[1], [2, -1, -1, -1])
    r0 = search.query(targets, max_samples=4, min_samples=0)
    assert r0.estimated.all()


def test_cross_validation_excludes_the_sample_itself(case):
    search, _, _ = case
    pts = search.samples[:2000]
    r = search.query(pts, max_samples=8, exclude_zero_distance=True)
    assert (r.indices != np.arange(2000)[:, None]).all()
    assert (r.distances[:, 0] > 0).all()
    keep = search.query(pts, max_samples=8)
    np.testing.assert_array_equal(keep.indices[:, 0], np.arange(2000))
    np.testing.assert_array_equal(keep.distances[:, 0], 0.0)
    np.testing.assert_array_equal(keep.indices[:, 1:], r.indices[:, :7])


def test_hole_ids_of_any_dtype_and_empty_inputs():
    samples = np.array([[1.0, 0, 0], [2.0, 0, 0], [3.0, 0, 0], [0, 4.0, 0]])
    target = np.zeros((1, 3))
    for holes in (["a", "a", "a", "b"], [7, 7, 7, 3], np.array([1.5, 1.5, 1.5, 0.5])):
        r = SampleSearch(samples, Ellipsoid(10.0), hole_ids=holes).query(
            target, max_samples=4, max_per_hole=2
        )
        np.testing.assert_array_equal(r.indices, [[0, 1, 3, -1]])
    empty = SampleSearch(np.zeros((0, 3)), Ellipsoid(10.0)).query(target, max_samples=3)
    assert empty.counts[0] == 0 and not empty.estimated[0]
    assert empty.indices.shape == (1, 3)
    none = SampleSearch(samples, Ellipsoid(10.0)).query(np.zeros((0, 3)), max_samples=3)
    assert none.indices.shape == (0, 3) and none.counts.shape == (0,)


def test_validation():
    samples = np.zeros((3, 3))
    search = SampleSearch(samples, Ellipsoid(10.0))
    target = np.zeros((1, 3))
    with pytest.raises(ValueError, match="hole_ids"):
        search.query(target, max_per_hole=2)
    with pytest.raises(ValueError, match="max_per_sector needs"):
        SearchParams(max_per_sector=2)
    with pytest.raises(ValueError, match="min_sectors needs"):
        SearchParams(min_sectors=2)
    with pytest.raises(ValueError, match="min_sectors must"):
        SearchParams(sectors="quadrant", min_sectors=5)
    with pytest.raises(ValueError, match="sectors must"):
        SearchParams(sectors="hexant")
    with pytest.raises(ValueError):
        SearchParams(max_samples=0)
    with pytest.raises(ValueError):
        SearchParams(max_samples=4, min_samples=5)
    with pytest.raises(ValueError):
        SearchParams(radius=-1.0)
    with pytest.raises(ValueError):
        SearchParams(max_vertical=-1.0)
    with pytest.raises(TypeError):
        search.query(target, SearchParams(), max_samples=3)
    with pytest.raises(ValueError, match="shape"):
        search.query(np.zeros((4, 2)))
    with pytest.raises(ValueError, match="non-finite"):
        search.query(np.array([[0.0, np.nan, 0.0]]))
    with pytest.raises(ValueError, match="non-finite"):
        SampleSearch(np.array([[0.0, np.inf, 0.0]]), Ellipsoid(1.0))
    with pytest.raises(ValueError, match="one entry per sample"):
        SampleSearch(samples, Ellipsoid(1.0), hole_ids=[1, 2])
    assert SearchParams(sectors="octant").n_sectors == 8
    assert SearchParams(sectors="quadrant").n_sectors == 4
    assert SearchParams().n_sectors == 0


# ---------------------------------------------------------------- performance
def test_performance_plain_search():
    """Reduced-size version of the target (1M targets x 20k samples, 24
    neighbours, well under a minute): 200k targets must finish in 20 s, a
    generous bound (about 1 s is typical on a multi-core laptop)."""
    rng = np.random.default_rng(1)
    samples, holes = make_holes(rng, n_holes=200, per_hole=100, spacing=1.5)
    assert len(samples) == 20_000
    targets = np.column_stack(
        [rng.uniform(0, 300, 200_000), rng.uniform(0, 300, 200_000), rng.uniform(0, 150, 200_000)]
    )
    search = SampleSearch(samples, Ellipsoid(80.0, 1.5, 3.0, angles=(30.0, 20.0, 10.0)), holes)
    search.query(targets[:100], max_samples=24)  # compile outside the timing
    start = time.perf_counter()
    result = search.query(targets, max_samples=24)
    elapsed = time.perf_counter() - start
    assert result.indices.shape == (200_000, 24)
    assert result.counts.mean() > 12
    assert elapsed < 20.0, f"plain search took {elapsed:.1f} s"
