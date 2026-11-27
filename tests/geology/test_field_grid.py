"""Interpolation of sampled fields on the Kuhn tetrahedra of a grid (P9.T7): research.md 5.7.2.

Independent checks: an explicit barycentric solve over the six tetrahedra of a cell,
linear and quadratic functions with their analytic error bound, and direct evaluation of
fitted RBF fields.

Error bounds stated and tested, for a cell with edges ``h = (hx, hy, hz)`` and
``R = |h| / 2`` (every Kuhn tetrahedron has its corners on the sphere around the cell):

* a linear function is reproduced exactly;
* a quadratic function with Hessian ``H``: ``|error| <= R^2 |H|_2 / 2``;
* a function with Lipschitz constant ``L`` (an RBF field with the linear kernel, whose
  second derivative is unbounded at the data points): ``|error| <= L |h|``.
"""

from __future__ import annotations

import time
from itertools import permutations

import numpy as np
import pytest

from bakumine.core.geology.lithology import LithologyStack, UnitFields, margins, tetra_interpolate
from bakumine.core.geology.model import GeologyModel
from bakumine.core.geology.units import Unit, UnitDictionary
from bakumine.core.geometry.implicit import fit_field
from bakumine.core.geometry.isosurface import GridSpec

GRID = GridSpec((100.0, -50.0, 20.0), (2.0, 3.0, 1.5), (9, 7, 11))
LO = GRID.origin
HI = GRID.origin + (np.array(GRID.shape) - 1) * GRID.spacing


def nodes_of(grid: GridSpec) -> np.ndarray:
    return grid.slab_points(0, grid.shape[0])


def random_points(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(LO, HI, size=(n, 3))


def barycentric_reference(values: np.ndarray, grid: GridSpec, point: np.ndarray) -> float:
    """Value at one point by solving for the barycentric coordinates in each of the six
    tetrahedra ``0, e_a, e_a + e_b, e_a + e_b + e_c`` of its cell (no sorting involved)."""
    u = (point - grid.origin) / grid.spacing
    cell = np.minimum(np.floor(u).astype(int), np.array(grid.shape) - 2)
    best = None
    for order in permutations(range(3)):
        corners = [cell.copy()]
        for axis in order:
            step = corners[-1].copy()
            step[axis] += 1
            corners.append(step)
        corners = np.array(corners, dtype=float)
        system = np.vstack([corners.T, np.ones(4)])
        weights = np.linalg.solve(system, np.append(u, 1.0))
        if weights.min() >= -1e-9 and (best is None or weights.min() > best[0]):
            best = (
                weights.min(),
                float(
                    sum(
                        w * values[tuple(c.astype(int))]
                        for w, c in zip(weights, corners, strict=True)
                    )
                ),
            )
    assert best is not None
    return best[1]


def test_linear_function_is_reproduced():
    coefficient = np.array([0.7, -1.3, 2.1])
    nodes = nodes_of(GRID)
    values = nodes @ coefficient + 5.0
    pts = np.vstack([random_points(5000, 1), LO, HI, [HI[0], LO[1], HI[2]]])
    got = tetra_interpolate(values, GRID, pts)
    assert got.shape == (len(pts),)
    np.testing.assert_allclose(got, pts @ coefficient + 5.0, rtol=0, atol=1e-10)


def test_node_values_are_reproduced_and_shapes():
    rng = np.random.default_rng(2)
    values = rng.normal(size=(3, *GRID.shape))
    nodes = nodes_of(GRID).reshape(-1, 3)
    got = tetra_interpolate(values, GRID, nodes)
    assert got.shape == (GRID.n_nodes, 3)
    np.testing.assert_allclose(got, values.reshape(3, -1).T, rtol=0, atol=1e-12)
    single = tetra_interpolate(values[1], GRID, nodes)
    np.testing.assert_allclose(single, values[1].reshape(-1), rtol=0, atol=1e-12)


def test_against_explicit_barycentric_solve():
    rng = np.random.default_rng(3)
    values = rng.normal(size=GRID.shape)
    pts = random_points(400, 4)
    # also points on cell faces and edges, where two or more tetrahedra meet
    on_face = pts[:100].copy()
    on_face[:, 0] = LO[0] + GRID.spacing[0] * np.round((on_face[:, 0] - LO[0]) / GRID.spacing[0])
    on_edge = on_face[:50].copy()
    on_edge[:, 2] = LO[2] + GRID.spacing[2] * np.round((on_edge[:, 2] - LO[2]) / GRID.spacing[2])
    pts = np.vstack([pts, on_face, on_edge])
    got = tetra_interpolate(values, GRID, pts)
    expected = np.array([barycentric_reference(values, GRID, p) for p in pts])
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-10)


def test_quadratic_error_bound():
    hessian = np.array([[2.0, 0.6, -0.4], [0.6, -3.0, 0.8], [-0.4, 0.8, 1.2]])
    centre = 0.5 * (LO + HI)

    def f(p):
        q = p - centre
        return 0.5 * np.einsum("...i,ij,...j->...", q, hessian, q) + q @ np.array([1.0, 2.0, -1.0])

    pts = random_points(40000, 5)
    for refine in (1, 2):
        spacing = GRID.spacing / refine
        shape = tuple((np.array(GRID.shape) - 1) * refine + 1)
        grid = GridSpec(LO, spacing, shape)
        error = np.abs(tetra_interpolate(f(nodes_of(grid)), grid, pts) - f(pts))
        bound = (
            0.5 * (0.5 * np.linalg.norm(spacing)) ** 2 * np.abs(np.linalg.eigvalsh(hessian)).max()
        )
        assert error.max() <= bound * (1 + 1e-9)
        assert error.max() >= 0.3 * bound  # the bound is not vacuous
        if refine == 1:
            coarse = error.max()
    assert error.max() == pytest.approx(coarse / 4.0, rel=0.15)  # second order


def test_outside_nan_and_errors(tmp_path):
    values = np.arange(GRID.n_nodes, dtype=np.float64).reshape(GRID.shape)
    pts = np.array(
        [
            LO - [1e-6, 0.0, 0.0],
            HI + [0.0, 0.0, 1e-6],
            [np.nan, 0.0, 30.0],
            0.5 * (LO + HI),
            [np.inf, 0.0, 30.0],
        ]
    )
    got = tetra_interpolate(values, GRID, pts)
    assert np.isnan(got[[0, 1, 2, 4]]).all() and np.isfinite(got[3])
    assert tetra_interpolate(values, GRID, np.zeros((0, 3))).shape == (0,)
    assert tetra_interpolate(np.stack([values, values]), GRID, np.zeros((0, 3))).shape == (0, 2)
    # a float32 array opened read-only from disk (how the model folder is read)
    np.save(tmp_path / "v.npy", np.stack([values, -values]).astype(np.float32))
    mapped = np.load(tmp_path / "v.npy", mmap_mode="r")
    inside = random_points(1000, 6)
    np.testing.assert_allclose(
        tetra_interpolate(mapped, GRID, inside)[:, 0],
        tetra_interpolate(values, GRID, inside),
        rtol=1e-6,
    )
    # an infinite node value spreads only into the cells that touch it
    spiked = values.copy()
    spiked[0, 0, 0] = -np.inf
    far = tetra_interpolate(spiked, GRID, np.array([HI - 0.1 * GRID.spacing, LO]))
    assert np.isfinite(far[0]) and far[1] == -np.inf
    with pytest.raises(ValueError, match="world-aligned"):
        turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
        tetra_interpolate(values, GridSpec(LO, GRID.spacing, GRID.shape, turned), inside)
    with pytest.raises(ValueError, match="grid has"):
        tetra_interpolate(values[:-1], GRID, inside)
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        tetra_interpolate(values, GRID, inside[:, :2])
    with pytest.raises(ValueError, match="shape"):
        tetra_interpolate(values[0], GRID, inside)


# --------------------------------------------------------------------------
# against direct evaluation of fitted fields
# --------------------------------------------------------------------------

BOX_LO = np.array([0.0, 0.0, 0.0])
BOX_HI = np.array([120.0, 120.0, 90.0])
CAP = 15.0
NORMAL = np.array([0.2, -0.1, 1.0]) / np.linalg.norm([0.2, -0.1, 1.0])
TOPS = (55.0, 35.0)  # along NORMAL: unit T above 55, M between, B below 35


def signed_distances(points: np.ndarray) -> np.ndarray:
    s = points @ NORMAL
    top = s - TOPS[0]
    bottom = TOPS[1] - s
    middle = np.minimum(-top, -bottom)
    return np.clip(np.column_stack([top, middle, bottom]), -CAP, CAP)


@pytest.fixture(scope="module")
def fitted() -> UnitFields:
    points = np.random.default_rng(7).uniform(BOX_LO, BOX_HI, size=(600, 3))
    data = signed_distances(points)
    fields = [fit_field(points, data[:, i]) for i in range(3)]
    return UnitFields(
        ("T", "M", "B"),
        fields,
        cap=CAP,
        sample_spacing=5.0,
        far_spacing=10.0,
        inputs=[(points, data[:, i]) for i in range(3)],
    )


def lipschitz(field) -> float:
    """An upper bound of |grad f| for the isotropic linear kernel: every kernel term has a
    gradient of length 1 in model coordinates, which are world coordinates / ``length``."""
    assert field.kernel == "linear" and field.anisotropy is None and field.weights is not None
    return float((np.abs(field.weights).sum() + np.linalg.norm(field.poly[1:])) / field.length)


def test_grid_interpolation_against_direct_evaluation(fitted):
    """|interpolated - direct| <= L |h| for every field (rigorous, loose); the measured
    error is printed and must fall when the grid is refined."""
    pts = np.random.default_rng(8).uniform(BOX_LO, BOX_HI, size=(20000, 3))
    t0 = time.perf_counter()
    direct = fitted.values(pts)
    seconds_direct = time.perf_counter() - t0
    worst = {}
    for h in (6.0, 3.0):
        shape = tuple(int(n) for n in np.round((BOX_HI - BOX_LO) / h) + 1)
        grid = GridSpec(BOX_LO, (h, h, h), shape)
        nodes = nodes_of(grid).reshape(-1, 3)
        node_values = fitted.values(nodes).T.reshape(3, *shape)
        t0 = time.perf_counter()
        error = np.abs(tetra_interpolate(node_values, grid, pts) - direct)
        seconds = time.perf_counter() - t0
        for i, field in enumerate(fitted.fields):
            assert error[:, i].max() <= lipschitz(field) * h * np.sqrt(3.0)
        worst[h] = (float(error.max()), float(error.mean()))
        print(
            f"\ngrid {h:g} m: max |error| {error.max():.3f} m, mean {error.mean():.4f} m "
            f"(field range +-{CAP:g} m; bound L|h| = "
            f"{max(lipschitz(f) for f in fitted.fields) * h * np.sqrt(3.0):.1f} m); "
            f"interpolation {seconds * 1e3:.1f} ms, direct {seconds_direct * 1e3:.1f} ms"
        )
    assert worst[3.0][1] < 0.6 * worst[6.0][1]
    assert worst[3.0][0] < worst[6.0][0]
    assert worst[3.0][0] < 0.1 * CAP  # measured: the stated practical accuracy at 3 m


def test_labels_from_grid_margins_against_exact(fitted):
    """Labels from interpolated node margins against exact labels: they differ only next
    to a boundary, within the interpolation error of the margins."""
    h = 3.0
    shape = tuple(int(n) for n in np.round((BOX_HI - BOX_LO) / h) + 1)
    grid = GridSpec(BOX_LO, (h, h, h), shape)
    nodes = nodes_of(grid).reshape(-1, 3)
    node_margins = fitted.node_values(nodes).T.reshape(3, *shape)
    np.testing.assert_allclose(
        node_margins.reshape(3, -1).T, margins(fitted.values(nodes)), rtol=0, atol=1e-12
    )
    pts = np.random.default_rng(9).uniform(BOX_LO, BOX_HI, size=(20000, 3))
    exact = fitted.labels(pts)
    fast = fitted.labels_from_values(tetra_interpolate(node_margins, grid, pts))
    agreement = float((exact == fast).mean())
    print(f"\nlabels from a {h:g} m grid equal the exact labels at {agreement:.4%} of the points")
    assert agreement >= 0.995
    wrong = exact != fast
    exact_margin = np.abs(
        np.sort(fitted.values(pts[wrong]), axis=1)[:, -1]
        - np.sort(fitted.values(pts[wrong]), axis=1)[:, -2]
    )
    assert exact_margin.max() < 1.0  # only where the two largest fields are within 1 m


def test_model_grid_labels_and_fallback_outside_the_grid(fitted):
    units = UnitDictionary(units=tuple(Unit(code=c, kind="lithology") for c in ("T", "M", "B")))
    inner_lo, h = np.array([30.0, 30.0, 15.0]), 3.0
    grid = GridSpec(inner_lo, (h, h, h), (21, 21, 21))
    model = GeologyModel(
        name="grid",
        units=units,
        grid=grid,
        topography=None,
        stacks={"ALL": LithologyStack("ALL", (fitted,))},
    )
    assert not model.sampled
    pts = np.random.default_rng(10).uniform(BOX_LO, BOX_HI, size=(20000, 3))
    fast = model.labels(pts)  # samples the grid on first use
    assert model.sampled and model.diagnostics["grid_nodes"] == 21**3
    values = model.grid_values("ALL", 0)
    assert values.dtype == np.float32 and values.shape == (3, 21, 21, 21)
    nodes = nodes_of(grid).reshape(-1, 3)
    np.testing.assert_allclose(
        values.reshape(3, -1).T, fitted.node_values(nodes), rtol=1e-6, atol=1e-5
    )
    exact = model.labels(pts, exact=True)
    np.testing.assert_array_equal(exact, fitted.labels(pts))
    inside = ((pts >= inner_lo) & (pts <= inner_lo + 60.0)).all(axis=1)
    assert 0.05 < inside.mean() < 0.5
    np.testing.assert_array_equal(fast[~inside], exact[~inside])  # exact outside the grid
    assert (fast[inside] == exact[inside]).mean() >= 0.995
    expected = fitted.labels_from_values(tetra_interpolate(values, grid, pts[inside]))
    np.testing.assert_array_equal(fast[inside], expected)
    assert model.labels(np.array([[np.nan, 0.0, 0.0]]))[0] == -1
