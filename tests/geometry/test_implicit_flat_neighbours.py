"""Local RBF evaluation where the nearest constraints are coplanar (holes of one section).

The check is a hand solution of the constant-drift system with numpy, not scipy's
interpolator: for the linear kernel phi(r) = -r,

    [ K   1 ] [w]   [v]
    [ 1^T 0 ] [c] = [0],     s(x) = sum_i w_i phi(|x - x_i|) + c
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.interpolate import RBFInterpolator

from bakumine.core.geometry.implicit import fit_field

K = 20


def by_hand(nodes: np.ndarray, values: np.ndarray, x: np.ndarray) -> float:
    n = len(nodes)
    system = np.zeros((n + 1, n + 1))
    system[:n, :n] = -np.linalg.norm(nodes[:, None, :] - nodes[None, :, :], axis=2)
    system[:n, n] = 1.0
    system[n, :n] = 1.0
    solution = np.linalg.solve(system, np.append(values, 0.0))
    return float(-np.linalg.norm(nodes - x, axis=1) @ solution[:n] + solution[n])


@pytest.fixture(scope="module")
def case():
    rng = np.random.default_rng(7)
    # 60 constraints in the plane y = 0 (one drilling section) and 60 scattered ones far away.
    section = np.column_stack([rng.uniform(0, 100, 60), np.zeros(60), rng.uniform(0, 100, 60)])
    cloud = rng.uniform(0, 100, (60, 3)) + [1000.0, 0.0, 0.0]
    xyz = np.vstack([section, cloud])
    values = np.concatenate([
        2.0 + 0.03 * section[:, 0] - 0.05 * section[:, 2],
        np.sin(cloud[:, 0] / 20.0) + 0.02 * cloud[:, 1],
    ])  # fmt: skip
    field = fit_field(xyz, values, kernel="linear", degree=1, neighbors=K)
    beside = np.column_stack(
        [rng.uniform(20, 80, 25), rng.uniform(2, 8, 25), rng.uniform(20, 80, 25)]
    )
    inside = rng.uniform(20, 80, (25, 3)) + [1000.0, 0.0, 0.0]
    return xyz, values, field, beside, inside


def test_plain_scipy_fails_on_a_section(case):
    xyz, values, _, beside, _ = case
    plain = RBFInterpolator(xyz, values, neighbors=K, kernel="linear", degree=1)
    with pytest.raises(np.linalg.LinAlgError, match="Singular"):
        plain(beside)


def test_coplanar_neighbours_get_a_constant_drift(case):
    xyz, values, field, beside, inside = case
    field.flat_points = 0
    got = field(np.vstack([beside, inside]), clipped=False)
    assert np.isfinite(got).all()
    assert field.flat_points == len(beside)
    for x, value in zip(beside, got[: len(beside)], strict=True):
        near = np.argsort(np.linalg.norm(xyz - x, axis=1))[:K]
        assert (xyz[near, 1] == 0.0).all()  # all its neighbours are in the section
        assert value == pytest.approx(by_hand(xyz[near], values[near], x), abs=1e-8)
    # Points with neighbours in general position are evaluated as before.
    plain = RBFInterpolator(xyz, values, neighbors=K, kernel="linear", degree=1)
    np.testing.assert_allclose(got[len(beside) :], plain(inside), rtol=0, atol=1e-9)


def test_constraints_in_the_section_are_reproduced(case):
    xyz, values, field, _, _ = case
    np.testing.assert_allclose(field(xyz[:60], clipped=False), values[:60], atol=1e-8)


def test_many_points_and_other_kernels(case):
    xyz, values, field, beside, inside = case
    # More than one bisection level, mixed order.
    rng = np.random.default_rng(3)
    many = np.vstack([beside, inside] * 8)
    many = many[rng.permutation(len(many))]
    got = field(many, clipped=False)
    again = np.array([field(p[None, :], clipped=False)[0] for p in many[:40]])
    np.testing.assert_allclose(got[:40], again, atol=1e-9)
    # A kernel that needs the linear drift cannot fall back: the error is kept.
    cubic = fit_field(xyz, values, kernel="cubic", degree=1, neighbors=K)
    with pytest.raises(np.linalg.LinAlgError):
        cubic(beside, clipped=False)
