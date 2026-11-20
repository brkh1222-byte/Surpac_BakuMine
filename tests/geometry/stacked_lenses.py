"""Synthetic drilling of stacked parallel lenses (test data for the automatic orientation)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from bakumine.core.geometry.intercept_orientation import pole_of_plane

DIP, DIP_AZIMUTH = 69.0, 216.0
HOLE_AZIMUTH, HOLE_DIP = 36.0, -60.0
THICKNESS = (10.0, 30.0, 15.0, 25.0, 20.0)  # true thickness of the lenses, metres
LENS_SPACING = 40.0  # between the mid-planes of consecutive lenses, metres
POLE = pole_of_plane(DIP, DIP_AZIMUTH)
HOLE_DEPTH = 320.0


def stacked_lenses(
    *,
    missing: float = 0.0,
    noise: float = 0.0,
    n_along: int = 9,
    n_sections: int = 5,
    seed: int = 11,
) -> dict[str, pd.DataFrame]:
    """Parallel holes (azimuth 036, dip -60) on a 25 x 50 m grid through five lenses.

    The lenses are parallel planes dipping 69 degrees towards 216, 10-30 m thick, their
    mid-planes 40 m apart; each is an ellipse 260 m along strike by 200 m down dip whose
    centre is shifted from lens to lens. Holes are 25 m apart along the section (towards
    036) and the sections 50 m apart. ``missing`` is the fraction of intercepts logged as
    waste; ``noise`` moves the middle of every intercept (uniform, +- ``noise`` metres
    across the lens) and changes its thickness by up to +- 20 %.
    """
    rng = np.random.default_rng(seed)
    az, dip = np.radians(HOLE_AZIMUTH), np.radians(-HOLE_DIP)
    hole = np.array([np.sin(az) * np.cos(dip), np.cos(az) * np.cos(dip), -np.sin(dip)])
    along = np.array([np.sin(az), np.cos(az), 0.0])
    across = np.array([np.cos(az), -np.sin(az), 0.0])  # strike of the lenses (azimuth 126)
    down_dip = np.cross(POLE, across)
    centre = 150.0 * hole  # the middle of the stack is 150 m down the central hole
    shifts = rng.uniform(-30.0, 30.0, size=(len(THICKNESS), 2))
    collars, logs = [], []
    for i in range(n_along):
        for j in range(n_sections):
            collar = (i - (n_along - 1) / 2) * 25.0 * along
            collar = collar + (j - (n_sections - 1) / 2) * 50.0 * across
            name = f"S{j}{i:02d}"
            collars.append((name, *collar, HOLE_DEPTH))
            ore = []
            for k, thickness in enumerate(THICKNESS):
                offset = (k - (len(THICKNESS) - 1) / 2) * LENS_SPACING
                t = ((centre - collar) @ POLE + offset) / (hole @ POLE)
                point = collar + t * hole - centre
                u = point @ across - shifts[k, 0]
                v = point @ down_dip - shifts[k, 1]
                if (u / 130.0) ** 2 + (v / 100.0) ** 2 > 1.0:
                    continue
                if rng.random() < missing:
                    continue
                true = thickness * (rng.uniform(0.8, 1.2) if noise else 1.0)
                t += rng.uniform(-noise, noise) / abs(hole @ POLE) if noise else 0.0
                half = 0.5 * true / abs(hole @ POLE)
                if 0.0 < t - half and t + half < HOLE_DEPTH:
                    ore.append((t - half, t + half))
            depth = 0.0
            for a, b in sorted(ore):
                if a <= depth:
                    continue
                logs += [(name, depth, a, "WASTE"), (name, a, b, "LODE")]
                depth = b
            logs.append((name, depth, HOLE_DEPTH, "WASTE"))
    collar = pd.DataFrame(collars, columns=["hole_id", "x", "y", "z", "max_depth"])
    survey = pd.DataFrame(
        {"hole_id": collar["hole_id"], "depth": 0.0, "dip": HOLE_DIP, "azimuth": HOLE_AZIMUTH}
    )
    geology = pd.DataFrame(logs, columns=["hole_id", "depth_from", "depth_to", "rock"])
    return {"collar": collar, "survey": survey, "geology": geology}
