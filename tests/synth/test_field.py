"""The simulated Gaussian field has the stated variogram; the grade transform is as stated."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.optimize import curve_fit

from bakumine.synth.field import GradeModel, spherical_correlation


def _spherical(h, sill, a):
    r = np.minimum(h / a, 1.0)
    return sill * (1.5 * r - 0.5 * r**3)


def _lattice_variogram(values: np.ndarray, axis: int, lags: np.ndarray) -> np.ndarray:
    """Experimental semivariogram along one lattice axis: 0.5 * mean squared increment."""
    v = np.moveaxis(values, axis, 0)
    return np.array([0.5 * np.mean((v[k:] - v[:-k]) ** 2) for k in lags])


def test_spherical_correlation_values():
    assert np.allclose(spherical_correlation(np.array([0.0, 0.5, 1.0, 3.0])), [1, 0.3125, 0, 0])


@pytest.mark.parametrize("axis", [0, 1, 2], ids=["major", "semi-major", "minor"])
def test_experimental_variogram_reaches_the_sill_at_the_stated_range(dataset, axis):
    field = dataset.deposit.field
    stated = dataset.deposit.truth["variogram"]["ranges_m"][axis]
    cell = field.cell[axis]
    n = field.values.shape[axis]
    lags = np.arange(1, min(int(2.0 * stated / cell), n // 2) + 1)
    h = lags * cell
    assert h[-1] >= 1.3 * stated  # the lattice is long enough to see the sill
    gamma = _lattice_variogram(field.values, axis, lags)

    (sill, fitted), _ = curve_fit(_spherical, h, gamma, p0=[1.0, 0.5 * h[-1]])
    assert abs(fitted / stated - 1.0) < 0.25
    assert abs(sill - 1.0) < 0.20
    # Shape, without a fit: about 69% of the sill at half the range, the sill at the range.
    at = lambda d: np.interp(d, h, gamma)  # noqa: E731
    assert 0.50 < at(0.5 * stated) < 0.85
    assert 0.80 < at(stated) < 1.20
    assert 0.80 < at(1.3 * stated) < 1.20
    # The whole curve stays near the model.
    assert np.abs(gamma - _spherical(h, 1.0, stated)).max() < 0.15


def test_variogram_is_anisotropic_as_stated(dataset):
    """At one common distance the three axes sit at clearly different variogram levels."""
    field = dataset.deposit.field
    ranges = dataset.deposit.truth["variogram"]["ranges_m"]
    distance = 0.8 * ranges[2]
    level = []
    for axis in range(3):
        k = max(1, int(round(distance / field.cell[axis])))
        gamma = _lattice_variogram(field.values, axis, np.array([k]))[0]
        model = _spherical(k * field.cell[axis], 1.0, ranges[axis])
        assert abs(gamma - model) < 0.1
        level.append(gamma)
    assert level[0] < level[1] < level[2]


def test_field_is_standard_normal(dataset):
    values = dataset.deposit.field.values
    assert abs(values.mean()) < 0.15
    assert abs(values.std() - 1.0) < 0.1
    quantiles = np.quantile(values, [0.025, 0.16, 0.5, 0.84, 0.975])
    assert np.allclose(quantiles, [-1.96, -0.994, 0.0, 0.994, 1.96], atol=0.2)


def test_field_lookup_returns_node_values_and_interpolates(dataset):
    field = dataset.deposit.field
    rng = np.random.default_rng(5)
    index = np.column_stack([rng.integers(0, n - 1, 200) for n in field.values.shape])
    world = field.centre + (field.lower + index * field.cell) @ field.axes.T
    assert np.allclose(field.at(world), field.values[tuple(index.T)], atol=1e-9)
    step = np.array([1, 0, 0])
    halfway = field.centre + (field.lower + (index + 0.5 * step) * field.cell) @ field.axes.T
    mean = 0.5 * (field.values[tuple(index.T)] + field.values[tuple((index + step).T)])
    assert np.allclose(field.at(halfway), mean, atol=1e-9)


def test_field_covers_the_solid(dataset):
    field, vertices = dataset.deposit.field, dataset.deposit.solid_vertices
    index = ((vertices - field.centre) @ field.axes - field.lower) / field.cell
    assert index.min() >= 2.0
    assert np.all(index.max(axis=0) <= np.array(field.values.shape) - 3.0)


def test_sample_gaussian_values_have_unit_sill_and_a_nugget(dataset):
    """Down-hole pairs one sample apart: gamma lies between the nugget and the sill."""
    truth = dataset.csv("truth_samples")
    nugget = dataset.deposit.truth["variogram"]["nugget"]
    ore = truth[truth.gauss.notna()]
    assert abs(ore.gauss.var() - 1.0) < 0.3
    squared = []
    for _, rows in ore.groupby("hole_id", sort=False):
        adjacent = rows.depth_from.to_numpy()[1:] == rows.depth_to.to_numpy()[:-1]
        squared.append(np.diff(rows.gauss.to_numpy())[adjacent] ** 2)
    gamma = 0.5 * np.mean(np.concatenate(squared))
    assert nugget * 0.7 < gamma < 0.6


def test_grade_model_moments():
    model = GradeModel(mu=np.log(1.6), sigma=0.8, nugget=0.15, waste_mu=np.log(0.02),
                       waste_sigma=0.5)  # fmt: skip
    rng = np.random.default_rng(11)
    structured, eps = rng.standard_normal((2, 2_000_000))
    point = model.ore_point(structured, eps)
    assert abs(point.mean() / model.ore_ensemble_mean - 1.0) < 0.01
    assert abs(np.median(point) / 1.6 - 1.0) < 0.01
    # Averaging the nugget analytically is unbiased for the point grade.
    assert abs(model.ore_expected(structured).mean() / point.mean() - 1.0) < 0.01
    assert abs(model.waste_point(eps).mean() / model.waste_mean - 1.0) < 0.01
    assert np.isclose(model.gaussian(structured, eps).var(), 1.0, atol=0.01)
