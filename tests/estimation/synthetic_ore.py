"""Shared set-up for the estimation tests on the synthetic deposit.

Not a test module. Builds the ORE composites the way a user would (import the
raw CSVs, desurvey, composite 2 m inside the ``rock`` domains of the geology
table) and derives the variogram of their normal scores, so that the
experimental-variogram test (P4.T8) and the validation test (P4.T9) use the
same fitted model.
"""

from __future__ import annotations

import json
from functools import cache, lru_cache
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

from bakumine.core.drillholes.composite import IntervalDomains, composite
from bakumine.core.drillholes.importer import TableSource, import_drillholes
from bakumine.core.estimation.experimental import (
    ExperimentalVariogram,
    VariogramFit,
    downhole_variogram,
    fit_variogram,
    normal_score,
    principal_variograms,
)
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.variogram import Structure, VariogramModel

SYNTH = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
COMPOSITE_LENGTH = 2.0


@lru_cache(maxsize=1)
def truth() -> dict[str, Any]:
    return json.loads((SYNTH / "truth.json").read_text())


def true_range_ellipsoid(major_radius: float | None = None) -> Ellipsoid:
    """The true variogram's range ellipsoid (truth.json axes are the columns
    major, semi-major, minor), optionally rescaled to another major radius."""
    v = truth()["variogram"]
    axes = np.array(v["axes"])
    a_major, a_semi, a_minor = v["ranges_m"]
    return Ellipsoid.from_axes(
        axes[:, 0],
        axes[:, 1],
        axes[:, 2],
        a_major if major_radius is None else major_radius,
        a_major / a_semi,
        a_major / a_minor,
    )


def true_model() -> VariogramModel:
    v = truth()["variogram"]
    assert v["model"] == "spherical"
    return VariogramModel(
        v["nugget"], (Structure.from_ellipsoid("spherical", v["sill"], true_range_ellipsoid()),)
    )


@lru_cache(maxsize=1)
def ore_composites() -> pd.DataFrame:
    """2 m ORE composites: columns hole_id, depth_from, depth_to, x, y, z, au, ..."""
    identity = {
        "collar": ["hole_id", "x", "y", "z", "max_depth"],
        "survey": ["hole_id", "depth", "dip", "azimuth"],
        "assay": ["hole_id", "depth_from", "depth_to", "au"],
        "geology": ["hole_id", "depth_from", "depth_to", "rock"],
    }
    maps = {name: {c: c for c in columns} for name, columns in identity.items()}
    db = import_drillholes(
        TableSource(SYNTH / "collar.csv", maps["collar"]),
        TableSource(SYNTH / "survey.csv", maps["survey"]),
        {
            "assay": TableSource(SYNTH / "assay.csv", maps["assay"]),
            "geology": TableSource(SYNTH / "geology.csv", maps["geology"]),
        },
    )
    comps = composite(
        db.intervals["assay"],
        db.desurvey(),
        length=COMPOSITE_LENGTH,
        mode="domain",
        domain=IntervalDomains(db.intervals["geology"], "rock"),
        attributes=["au"],
    )
    ore = comps[(comps["domain"] == "ORE") & comps["au"].notna()].reset_index(drop=True)
    assert np.isfinite(ore[["x", "y", "z"]].to_numpy()).all()
    return ore


def ore_samples() -> pd.DataFrame:
    """ORE assay intervals at their original (about 1 m) support: columns
    hole_id, x, y, z (interval midpoints of truth_samples.csv) and au (the
    rounded assays of assay.csv, what a user would have)."""
    smp = pd.read_csv(SYNTH / "truth_samples.csv", dtype={"hole_id": str})
    assay = pd.read_csv(SYNTH / "assay.csv", dtype={"hole_id": str})
    key = ["hole_id", "depth_from", "depth_to"]
    assert (smp[key].to_numpy() == assay[key].to_numpy()).all()  # row-aligned
    ore = (smp["rock"] == "ORE").to_numpy()
    out = smp.loc[ore, ["hole_id", "x", "y", "z"]].copy()
    out["au"] = assay.loc[ore, "au"].to_numpy()
    return out.reset_index(drop=True)


class OreVariography(NamedTuple):
    downhole: ExperimentalVariogram
    downhole_fit: VariogramFit
    directional: tuple[ExperimentalVariogram, ExperimentalVariogram, ExperimentalVariogram]
    fit: VariogramFit


#: support -> (table builder, nominal sample length = down-hole lag)
SUPPORTS = {"composites": (ore_composites, COMPOSITE_LENGTH), "samples": (ore_samples, 1.0)}


@cache
def ore_variography(
    support: str = "composites", angular_tolerance: float = 10.0
) -> OreVariography:
    """Variogram of the normal scores of the ORE data, true orientation.

    Steps (the routine a user would follow):

    1. normal-score transform of ``au`` (the truth variogram is that of the
       Gaussian variable behind the lognormal grades);
    2. down-hole variogram at the sample length, fitted on its own
       (isotropic spherical, total sill fixed at 1) to read the nugget;
    3. directional variograms along the three true axes. The anisotropy is
       strong (120 / 70 / 15 m), so the major and semi-major directions use a
       narrow cone (``angular_tolerance``, default 10 degrees) and stay
       within 3 m of the plane of the two long axes (a fifth of the minor
       range); the minor direction uses a 20 degree cone;
    4. joint fit of one spherical structure with the nugget fixed at the
       down-hole value and the total sill fixed at 1 (normal scores).
    """
    builder, length = SUPPORTS[support]
    data = builder()
    xyz = data[["x", "y", "z"]].to_numpy()
    scores = normal_score(data["au"].to_numpy())
    holes = data["hole_id"].to_numpy()
    downhole = downhole_variogram(xyz, scores, holes, lag_size=length, n_lags=12)
    downhole_fit = fit_variogram(downhole, sill=1.0, min_pairs=50)
    directional = principal_variograms(
        xyz,
        scores,
        true_range_ellipsoid(),
        lag_size=(20.0, 15.0, length),
        n_lags=(10, 8, 12),
        angular_tolerance=(angular_tolerance, angular_tolerance, 20.0),
        bandwidths=(None, None, 3.0),
    )
    fit = fit_variogram(
        directional,
        ellipsoid=true_range_ellipsoid(),
        nugget=downhole_fit.nugget,
        sill=1.0,
        min_pairs=50,
    )
    return OreVariography(downhole, downhole_fit, directional, fit)
