"""Topography: an analytic undulating surface and its triangulated DTM.

    X = x - origin_x,  Y = y - origin_y
    z(x, y) = base + slope_x * X + slope_y * Y
              + sum_i amp_i * sin(2 pi (X sin(dir_i) + Y cos(dir_i)) / wavelength_i + phase_i)

``dir_i`` is an azimuth (clockwise from north). Elevations are rounded to 1e-6 m so that the
value written to a file is exactly the value used everywhere else.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import Delaunay

from bakumine.synth.config import TopoConfig

Z_DECIMALS = 6


@dataclass(frozen=True)
class Topography:
    cfg: TopoConfig

    def z(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        dx = np.asarray(x, dtype=np.float64) - cfg.origin[0]
        dy = np.asarray(y, dtype=np.float64) - cfg.origin[1]
        z = cfg.base + cfg.slope[0] * dx + cfg.slope[1] * dy
        for amplitude, wavelength, direction, phase in cfg.waves:
            d = np.radians(direction)
            along = dx * np.sin(d) + dy * np.cos(d)
            z = z + amplitude * np.sin(2.0 * np.pi * along / wavelength + phase)
        return np.round(z, Z_DECIMALS)

    def upper_bound(self, lo: np.ndarray, hi: np.ndarray) -> float:
        """A value no lower than the surface anywhere in the rectangle ``lo..hi`` (x, y)."""
        cfg = self.cfg
        plane = [
            cfg.base + cfg.slope[0] * (x - cfg.origin[0]) + cfg.slope[1] * (y - cfg.origin[1])
            for x in (lo[0], hi[0])
            for y in (lo[1], hi[1])
        ]
        return float(max(plane) + sum(abs(wave[0]) for wave in cfg.waves))

    def dtm(
        self, lo: np.ndarray, hi: np.ndarray, extra_xy: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """Triangulated surface over ``lo..hi`` on a regular grid plus ``extra_xy`` points.

        The extra points (drillhole collars) become vertices, so the DTM passes exactly
        through them. Faces are counter-clockwise in plan (normals up).
        """
        s = self.cfg.dtm_spacing
        gx = np.arange(lo[0], hi[0] + 0.5 * s, s)
        gy = np.arange(lo[1], hi[1] + 0.5 * s, s)
        grid = np.column_stack([a.ravel() for a in np.meshgrid(gx, gy, indexing="ij")])
        xy = np.unique(np.concatenate([grid, np.asarray(extra_xy, dtype=np.float64)]), axis=0)
        # Triangulate in coordinates relative to the corner: full precision for the predicates.
        faces = Delaunay(xy - lo).simplices.astype(np.int64)
        p = xy - lo
        a, b, c = p[faces[:, 0]], p[faces[:, 1]], p[faces[:, 2]]
        ab, ac = b - a, c - a
        cross = ab[:, 0] * ac[:, 1] - ab[:, 1] * ac[:, 0]
        faces[cross < 0.0] = faces[cross < 0.0][:, ::-1]
        faces = faces[np.abs(cross) > 1e-9]
        # Canonical order, so the output does not depend on Qhull's facet order.
        start = faces.argmin(axis=1)
        faces = np.take_along_axis(faces, (start[:, None] + np.arange(3)) % 3, axis=1)
        faces = faces[np.lexsort(faces.T[::-1])]
        vertices = np.column_stack([xy, self.z(xy[:, 0], xy[:, 1])])
        return vertices, faces
