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


def fanned_lenses(
    *,
    dip: float = 77.0,
    dip_azimuth: float = 340.0,
    n_lenses: int = 10,
    extent: tuple[float, float] = (35.0, 25.0),
    missing: float = 0.2,
    seed: int = 1,
) -> tuple[dict[str, pd.DataFrame], np.ndarray]:
    """Thin steep lenses drilled from both sides of the sections, and the true pole.

    Holes dip 60 degrees, towards 170 and 350 alternately, 25 m apart on north-south
    sections 45 m apart (12 sections of 6 holes, 200 m deep). The lenses are parallel
    planes dipping ``dip`` towards ``dip_azimuth``, 2-6 m thick, about 20 m apart, each an
    ellipse with semi-axes ``extent`` (along strike, down dip) whose centre is shifted from
    lens to lens; ``missing`` is the fraction of intercepts logged as waste. Short lenses
    (35 x 25 m) against 45 m between sections give little continuity between holes.
    """
    rng = np.random.default_rng(seed)
    pole = pole_of_plane(dip, dip_azimuth)
    s = np.radians(dip_azimuth - 90.0)
    strike = np.array([np.sin(s), np.cos(s), 0.0])
    down = np.cross(pole, strike)
    section_azimuth, hole_dip, depth = 170.0, 60.0, 200.0
    sa = np.radians(section_azimuth)
    along = np.array([np.sin(sa), np.cos(sa), 0.0])
    between = np.array([np.cos(sa), -np.sin(sa), 0.0])
    centre = np.array([0.0, 0.0, -90.0])
    offset = (np.arange(n_lenses) - (n_lenses - 1) / 2) * 20.0 + rng.uniform(-5, 5, n_lenses)
    shift = rng.uniform(-150, 150, size=(n_lenses, 2)) * [1.0, 0.3]
    thickness = rng.uniform(2.0, 6.0, size=n_lenses)
    collars, surveys, logs = [], [], []
    n_sections, n_along = 12, 6
    for j in range(n_sections):
        for i in range(n_along):
            collar = (j - (n_sections - 1) / 2) * 45.0 * between
            collar = collar + (i - (n_along - 1) / 2) * 25.0 * along
            azimuth = section_azimuth if (i + j) % 2 == 0 else (section_azimuth + 180.0) % 360
            a, d = np.radians(azimuth), np.radians(hole_dip)
            hole = np.array([np.sin(a) * np.cos(d), np.cos(a) * np.cos(d), -np.sin(d)])
            name = f"DH-{j:02d}{i:02d}"
            collars.append((name, *collar, depth))
            surveys.append((name, 0.0, -hole_dip, azimuth))
            ore = []
            for k in range(n_lenses):
                crossing = hole @ pole
                if abs(crossing) < 0.1:
                    continue
                t = ((centre - collar) @ pole + offset[k]) / crossing
                point = collar + t * hole - centre
                u = point @ strike - shift[k, 0]
                v = point @ down - shift[k, 1]
                if (u / extent[0]) ** 2 + (v / extent[1]) ** 2 > 1:
                    continue
                if rng.random() < missing:
                    continue
                half = 0.5 * thickness[k] / abs(crossing)
                if 0 < t - half and t + half < depth:
                    ore.append((t - half, t + half))
            last = 0.0
            for a0, b0 in sorted(ore):
                if a0 <= last:
                    continue
                logs += [(name, last, a0, "WASTE"), (name, a0, b0, "LODE")]
                last = b0
            logs.append((name, last, depth, "WASTE"))
    collar = pd.DataFrame(collars, columns=["hole_id", "x", "y", "z", "max_depth"])
    survey = pd.DataFrame(surveys, columns=["hole_id", "depth", "dip", "azimuth"])
    geology = pd.DataFrame(logs, columns=["hole_id", "depth_from", "depth_to", "rock"])
    return {"collar": collar, "survey": survey, "geology": geology}, pole
