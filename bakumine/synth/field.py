"""Unconditional Gaussian simulation with a known anisotropic spherical variogram.

Method: FFT moving average (Le Ravalec, Noetinger and Hu, 2000). On a periodic lattice the
covariance matrix is circulant, so its eigenvalues are the FFT of the covariance sampled on
the lattice. White noise ``w`` is filtered as ``z = IFFT(sqrt(FFT(C)) * FFT(w))``, which gives
``Cov(z) = C`` exactly at every lattice lag. The lattice is aligned with the variogram axes
and extended by one range per axis; the extension is discarded so that no pair of retained
nodes is correlated through the periodic wrap.

The simulated array is the *structured* part ``Ys`` with unit variance and covariance
``sph(h)``. The full Gaussian variable of the variogram ``nugget + (1 - nugget) * sph`` is

    Y = sqrt(1 - nugget) * Ys + sqrt(nugget) * eps,     eps ~ N(0, 1) independent per sample.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import fft, ndimage

from bakumine.synth.config import GradeConfig, VariogramConfig


def spherical_correlation(h: np.ndarray) -> np.ndarray:
    """Spherical correlogram for normalised lag ``h = |lag| / range``: ``1 - 1.5h + 0.5h^3``."""
    h = np.minimum(np.abs(h), 1.0)
    return 1.0 - 1.5 * h + 0.5 * h**3


def simulate_fftma(
    shape: tuple[int, int, int],
    cell: tuple[float, float, float],
    ranges: tuple[float, float, float],
    rng: np.random.Generator,
) -> np.ndarray:
    """Standard-normal field on a lattice with spherical covariance of the given axis ranges.

    Returns ``shape`` nodes. Internally a larger periodic lattice is simulated (padded by one
    range per axis and rounded up to an FFT-friendly size).
    """
    full = tuple(
        fft.next_fast_len(n + int(np.ceil(a / d)) + 1, real=True)
        for n, d, a in zip(shape, cell, ranges, strict=True)
    )
    lag2 = np.zeros(full)
    for axis, (n, d, a) in enumerate(zip(full, cell, ranges, strict=True)):
        if n * d < 2.0 * a:
            raise ValueError("simulation lattice must span at least two ranges on every axis")
        k = np.arange(n)
        h = np.minimum(k, n - k) * d / a
        lag2 = lag2 + (h**2).reshape([-1 if i == axis else 1 for i in range(3)])
    spectrum = fft.rfftn(spherical_correlation(np.sqrt(lag2))).real
    del lag2
    np.maximum(spectrum, 0.0, out=spectrum)  # round-off only: the periodic sum is PSD
    noise = rng.standard_normal(full)
    z = fft.irfftn(np.sqrt(spectrum) * fft.rfftn(noise), s=full)
    return np.ascontiguousarray(z[: shape[0], : shape[1], : shape[2]])


@dataclass(frozen=True)
class GaussianField:
    """Structured Gaussian field on a lattice in the variogram frame."""

    centre: np.ndarray  # world origin of the variogram frame
    axes: np.ndarray  # columns: major, semi-major, minor (world)
    lower: np.ndarray  # frame coordinates of node [0, 0, 0]
    cell: np.ndarray
    values: np.ndarray

    @classmethod
    def simulate(
        cls,
        cfg: VariogramConfig,
        centre: np.ndarray,
        axes: np.ndarray,
        cover_points: np.ndarray,
        rng: np.random.Generator,
    ) -> GaussianField:
        """Simulate a field whose lattice covers ``cover_points`` (world) with a 3-cell margin."""
        cell = np.array(cfg.cell, dtype=np.float64)
        local = (cover_points - centre) @ axes
        lower = np.floor(local.min(axis=0) / cell) * cell - 3.0 * cell
        upper = local.max(axis=0) + 3.0 * cell
        shape = tuple(int(n) for n in np.ceil((upper - lower) / cell).astype(int) + 1)
        values = simulate_fftma(shape, cfg.cell, cfg.ranges, rng)
        return cls(centre=centre, axes=axes, lower=lower, cell=cell, values=values)

    def node_coordinates(self, axis: int) -> np.ndarray:
        return self.lower[axis] + self.cell[axis] * np.arange(self.values.shape[axis])

    def at(self, points: np.ndarray) -> np.ndarray:
        """Structured value at world points (n, 3) by trilinear interpolation between nodes."""
        index = ((np.asarray(points, dtype=np.float64) - self.centre) @ self.axes - self.lower)
        index = index / self.cell
        return ndimage.map_coordinates(self.values, index.T, order=1, mode="nearest")


@dataclass(frozen=True)
class GradeModel:
    """Lognormal grade in ore driven by the Gaussian field; uncorrelated low grade in waste."""

    mu: float
    sigma: float
    nugget: float
    waste_mu: float
    waste_sigma: float

    @classmethod
    def from_config(cls, grade: GradeConfig, variogram: VariogramConfig) -> GradeModel:
        return cls(
            mu=float(np.log(grade.ore_median)),
            sigma=grade.ore_sigma,
            nugget=variogram.nugget,
            waste_mu=float(np.log(grade.waste_median)),
            waste_sigma=grade.waste_sigma,
        )

    def gaussian(self, structured: np.ndarray, eps: np.ndarray) -> np.ndarray:
        return np.sqrt(1.0 - self.nugget) * structured + np.sqrt(self.nugget) * eps

    def ore_point(self, structured: np.ndarray, eps: np.ndarray) -> np.ndarray:
        """Sample-support ore grade: ``exp(mu + sigma * Y)``."""
        return np.exp(self.mu + self.sigma * self.gaussian(structured, eps))

    def ore_expected(self, structured: np.ndarray) -> np.ndarray:
        """Ore grade averaged over the nugget component (exact for any volume average)."""
        scale = self.sigma * np.sqrt(1.0 - self.nugget)
        return np.exp(self.mu + scale * structured + 0.5 * self.sigma**2 * self.nugget)

    def waste_point(self, eps: np.ndarray) -> np.ndarray:
        return np.exp(self.waste_mu + self.waste_sigma * eps)

    @property
    def waste_mean(self) -> float:
        return float(np.exp(self.waste_mu + 0.5 * self.waste_sigma**2))

    @property
    def ore_ensemble_mean(self) -> float:
        return float(np.exp(self.mu + 0.5 * self.sigma**2))
