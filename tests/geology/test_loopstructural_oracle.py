"""The sequence field against LoopStructural (P9.T10): a second implementation.

LoopStructural (Grose et al. 2021, MIT) is a development dependency used here as an
oracle only; nothing under ``bakumine/`` imports it. Both programs get the same value
data: the points and stratigraphic values that :func:`fit_sequence` takes from the holes.
Ours interpolates them exactly with a radial basis function; LoopStructural fits a
piecewise-linear field on a tetrahedral mesh by regularised least squares (its "PLI"
interpolator) or finite differences on a grid ("FDI"). The units are then cut at the same
levels, and the labels compared on a lattice inside the drilled volume.

Section 14 of geology_contracts.md: agreement at >= 95 % of the grid points on the
unfaulted sequence. A faulted comparison is not made here: the faulted rock model is
P9.T13.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geology.sequence import SequenceField, fit_sequence
from bakumine.core.geology.units import UnitDictionary

LoopStructural = pytest.importorskip(
    "LoopStructural", reason="LoopStructural is not installed (dev dependency, oracle only)"
)

AGREEMENT = 0.95


def loop_field(seq: SequenceField, lo, hi, *, interpolator: str, elements: int):
    """LoopStructural's field for the value data of ``seq`` in the box ``lo .. hi``:
    a function ``points -> values``."""
    logging.getLogger("LoopStructural").setLevel(logging.ERROR)
    logging.getLogger("loop_interpolation").setLevel(logging.ERROR)
    xyz, values = seq.inputs
    data = pd.DataFrame(
        {"X": xyz[:, 0], "Y": xyz[:, 1], "Z": xyz[:, 2], "val": values, "feature_name": "strat"}
    )
    model = LoopStructural.GeologicalModel(np.asarray(lo, float), np.asarray(hi, float))
    model.data = data
    model.create_and_add_foliation("strat", interpolatortype=interpolator, nelements=elements)
    model.update()
    return lambda points: np.asarray(model.evaluate_feature_value("strat", points))


def compare(seq: SequenceField, other, points, truth):
    ours = seq.labels(points)
    theirs = seq.labels_from_values(other(points))
    ok = theirs >= 0
    units = np.array(seq.units, dtype=object)
    return {
        "agreement": float((ours[ok] == theirs[ok]).mean()),
        "evaluated": float(ok.mean()),
        "ours": float((units[ours] == truth).mean()),
        "theirs": float((units[theirs[ok]] == truth[ok]).mean()),
    }


# --------------------------------------------------------------------------
# an analytic stack with a gentle fold
# --------------------------------------------------------------------------

FOLD_UNITS = ("A", "B", "C", "D")
FOLD_LEVELS = (-50.0, -80.0, -125.0)


def fold_position(points):
    p = np.asarray(points)
    return 0.94 * p[:, 2] + 0.30 * p[:, 1] - 0.15 * p[:, 0] + 12.0 * np.sin(p[:, 0] / 90.0)


def fold_truth(points):
    s = fold_position(points)
    k = (s < FOLD_LEVELS[0]).astype(int) + (s < FOLD_LEVELS[1]) + (s < FOLD_LEVELS[2])
    return np.array(FOLD_UNITS, dtype=object)[k]


def drill_vertical(label, spacing=40.0, half=120.0, length=220.0, step=0.25):
    axis = np.arange(-half, half + 1e-9, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    ids = [f"V{i + 1:03d}" for i in range(x.size)]
    collar = pd.DataFrame({"hole_id": ids, "x": x.ravel(), "y": y.ravel(), "z": 0.0})
    collar["max_depth"] = length
    survey = pd.DataFrame(
        {"hole_id": np.repeat(ids, 2), "depth": np.tile([0.0, length], len(ids)), "dip": -90.0}
    ).assign(azimuth=0.0)
    depth = np.arange(0.0, length + 1e-9, step)
    rows = []
    for hole, cx, cy in zip(ids, x.ravel(), y.ravel(), strict=True):
        codes = label(np.column_stack([np.full(len(depth), cx), np.full(len(depth), cy), -depth]))
        cut = np.flatnonzero(codes[1:] != codes[:-1])
        edges = np.concatenate([[0.0], 0.5 * (depth[cut] + depth[cut + 1]), [length]])
        rows += [
            (hole, a, b, c)
            for a, b, c in zip(edges[:-1], edges[1:], codes[np.r_[0, cut + 1]], strict=True)
        ]
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])
    return table, desurvey(collar, survey)


@pytest.mark.parametrize("interpolator", ["PLI", "FDI"])
def test_folded_stack_against_loopstructural(interpolator):
    """Four folded layers, vertical holes 40 m apart: the two programs give the same unit
    at >= 95 % of 4 m lattice points inside the drilled volume."""
    table, traces = drill_vertical(fold_truth)
    seq = fit_sequence(table, table["unit"], FOLD_UNITS, traces)
    lo, hi = np.array([-140.0, -140.0, -240.0]), np.array([140.0, 140.0, 20.0])
    other = loop_field(seq, lo, hi, interpolator=interpolator, elements=30000)
    axes = [np.arange(-118.0, 119.0, 4.0), np.arange(-118.0, 119.0, 4.0), np.arange(-215, -4, 4.0)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    result = compare(seq, other, points, fold_truth(points))
    print(
        f"\nfolded stack, LoopStructural {LoopStructural.__version__} {interpolator}: same unit "
        f"at {result['agreement']:.4f} of {len(points)} points; accuracy against the truth: "
        f"ours {result['ours']:.4f}, LoopStructural {result['theirs']:.4f}"
    )
    assert result["evaluated"] > 0.999
    assert result["agreement"] >= AGREEMENT
    assert result["ours"] >= 0.97 and result["theirs"] >= 0.95


# --------------------------------------------------------------------------
# the layered units of the synthetic deposit, without fault handling
# --------------------------------------------------------------------------


def test_layered_sequence_against_loopstructural(
    layered_small, layered_small_dir, layered_small_db, layered_small_traces
):
    """SST > SLT > AND of layered-small as one unfaulted sequence (the intrusions unknown):
    the same unit at >= 95 % of 5 m lattice points inside the drilled hull."""
    units = UnitDictionary.load(layered_small_dir / "units.json")
    table = layered_small_db.intervals["geology"]
    mapped = units.map_codes(table["lith"])
    order = ("SST", "SLT", "AND")
    seq = fit_sequence(table, mapped, order, layered_small_traces)
    cfg = layered_small.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    other = loop_field(seq, lo, hi, interpolator="PLI", elements=40000)
    axes = [np.arange(a + 2.5, b, 5.0) for a, b in zip(lo, hi, strict=True)]
    points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
    points = points[Delaunay(seq.inputs[0]).find_simplex(points) >= 0]
    # compared where the truth is one of the three layers (not inside an intrusion)
    labels = layered_small.truth.labels(points, "bedrock")
    layered = np.isin(labels, order)
    result = compare(seq, other, points[layered], labels[layered])
    print(
        f"\nlayered-small sequence, LoopStructural {LoopStructural.__version__} PLI: same unit "
        f"at {result['agreement']:.4f} of {int(layered.sum())} points in the hull; accuracy "
        f"against the truth: ours {result['ours']:.4f}, LoopStructural {result['theirs']:.4f}"
    )
    assert result["evaluated"] > 0.999
    assert result["agreement"] >= AGREEMENT
