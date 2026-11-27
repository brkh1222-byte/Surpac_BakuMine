"""Sequence field and intrusion overprints (P9.T10): research.md 5.7.3.

Independent checks: analytic bodies drilled by synthetic straight holes (parallel dipping
layers with known thicknesses, overlapping spheres, a vertical sheet of known thickness,
a curved sheet that pinches out), lattice counts against analytic volumes, the
largest-field model of P9.T7 on the same data, and the exact truth of the synthetic
layered deposit. Measured numbers are printed (``pytest -s``).
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geology.lithology import (
    EventSpec,
    FieldSettings,
    LithologyEvent,
    LithologyStack,
    fit_stack,
    fit_unit_fields,
    logged_hole_points,
)
from bakumine.core.geology.model import GeologyModel
from bakumine.core.geology.sequence import (
    MIN_SHEET_HOLES,
    IntrusionField,
    SequenceField,
    event_from_manifest,
    fit_intrusion,
    fit_sequence,
    mask_younger,
    order_intrusions,
)
from bakumine.core.geology.units import MODELLED_KINDS, UnitDictionary
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.isosurface import GridSpec
from bakumine.core.io.obj import read_obj

# --------------------------------------------------------------------------
# synthetic drilling of an analytic body
# --------------------------------------------------------------------------


def drill(label, collars, *, dip=-90.0, azimuth=0.0, length=200.0, scan=0.5):
    """Straight holes logged from ``label(points) -> codes``; contacts by bisection.

    Returns ``(table, traces)``: ``hole_id, depth_from, depth_to, unit`` and the traces.
    """
    collars = np.asarray(collars, dtype=np.float64)
    n = len(collars)
    dip = np.broadcast_to(np.asarray(dip, dtype=np.float64), n)
    azimuth = np.broadcast_to(np.asarray(azimuth, dtype=np.float64), n)
    d, a = np.radians(dip), np.radians(azimuth)
    direction = np.column_stack([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)])
    ids = [f"H{i + 1:03d}" for i in range(n)]
    collar = pd.DataFrame(
        {"hole_id": ids, "x": collars[:, 0], "y": collars[:, 1], "z": collars[:, 2]}
    ).assign(max_depth=length)
    survey = pd.DataFrame(
        {
            "hole_id": np.repeat(ids, 2),
            "depth": np.tile([0.0, length], n),
            "dip": np.repeat(dip, 2),
            "azimuth": np.repeat(azimuth, 2),
        }
    )
    depths = np.arange(0.0, length + 1e-9, scan)
    rows = []
    for i in range(n):

        def at(depth, i=i):
            return label(collars[i] + np.atleast_1d(depth)[:, None] * direction[i])

        codes = at(depths)
        start = 0.0
        for k in np.flatnonzero(codes[1:] != codes[:-1]):
            lo, hi = depths[k], depths[k + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if at(mid)[0] == codes[k]:
                    lo = mid
                else:
                    hi = mid
            rows.append((ids[i], start, hi, codes[k]))
            start = hi
        rows.append((ids[i], start, length, codes[-1]))
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])
    return table, desurvey(collar, survey)


def collar_grid(lo, hi, spacing, z=0.0):
    axis = np.arange(lo, hi + 1e-9, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def lattice(lo, hi, spacing):
    axes = [np.arange(a + spacing / 2.0, b, spacing) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def normal_of(dip, dip_azimuth):
    d, a = np.radians(dip), np.radians(dip_azimuth)
    return np.array([np.sin(d) * np.sin(a), np.sin(d) * np.cos(a), np.cos(d)])


def angle_between(a, b):
    c = abs(float(np.dot(a, b))) / (np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(c, 1.0))))


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def interval_points(table, known, traces, step=2.0):
    """Points every ``<= step`` m along the known intervals: ``(xyz, row index)``."""
    a = table["depth_from"].to_numpy(dtype=float)[known]
    b = table["depth_to"].to_numpy(dtype=float)[known]
    hole = table["hole_id"].to_numpy(dtype=object)[known]
    row = np.flatnonzero(known)
    n = np.maximum(1, np.ceil((b - a) / step).astype(int))
    rep = np.repeat(np.arange(len(a)), n)
    first = np.concatenate([[0], np.cumsum(n)[:-1]])
    depth = a[rep] + (np.arange(len(rep)) - first[rep] + 0.5) * ((b - a) / n)[rep]
    xyz, _ = traces.xyz_at(hole[rep], depth)
    return xyz, row[rep]


# --------------------------------------------------------------------------
# parallel dipping layers: thicknesses 30 m and 45 m between two thick units
# --------------------------------------------------------------------------

STACK = ("A", "B", "C", "D")
STACK_POLE = normal_of(25.0, 110.0)
STACK_TOPS = (-50.0, -80.0, -125.0)  # pole . p of the contacts A/B, B/C, C/D


def stack_truth(points):
    s = np.asarray(points) @ STACK_POLE
    k = (s < STACK_TOPS[0]).astype(int) + (s < STACK_TOPS[1]) + (s < STACK_TOPS[2])
    return np.array(STACK, dtype=object)[k]


@pytest.fixture(scope="module")
def stack():
    collars = collar_grid(-100.0, 100.0, 40.0)
    dip = np.where(np.arange(len(collars)) % 4 == 1, -65.0, -90.0)
    table, traces = drill(stack_truth, collars, dip=dip, azimuth=200.0, length=220.0)
    return table, traces, fit_sequence(table, table["unit"], STACK, traces)


def test_sequence_thickness_pole_and_levels(stack):
    """True thicknesses 30 m and 45 m within 0.5 %, the pole within 0.5 degrees, the
    levels ``T = 0, 30, 75``; the end units are capped at the median of the two."""
    table, traces, seq = stack
    assert isinstance(seq, SequenceField) and isinstance(seq, LithologyEvent)
    assert seq.mode == "sequence" and seq.units == STACK
    assert seq.thickness["B"] == pytest.approx(30.0, rel=5e-3)
    assert seq.thickness["C"] == pytest.approx(45.0, rel=5e-3)
    assert seq.thickness["A"] == seq.thickness["D"] == pytest.approx(37.5, rel=5e-3)
    np.testing.assert_allclose(seq.thresholds, [0.0, 30.0, 75.0], atol=0.2)
    assert angle_between(seq.pole, STACK_POLE) < 0.5 and seq.pole[2] > 0.0
    assert len(seq.violations) == 0
    assert seq.field.neighbors is None and seq.anisotropy is None
    print(
        f"\nstack: thickness B {seq.thickness['B']:.3f} (30), C {seq.thickness['C']:.3f} (45); "
        f"pole {angle_between(seq.pole, STACK_POLE):.3f} deg from the truth; "
        f"{seq.field.n_nodes} points"
    )


def test_sequence_contacts_recovered(stack):
    """Each contact, as the level ``phi = T_k`` on vertical lines 10 m apart inside the
    drilled area: RMS distance to the analytic plane <= 1 m, largest <= 3 m. (The field
    is not exactly linear: the values of the top and bottom unit are capped.)"""
    table, traces, seq = stack
    xy = lattice((-90.0, -90.0), (90.0, 90.0), 10.0)
    z = np.arange(-200.0, -5.0 + 1e-9, 1.0)
    pts = np.column_stack([np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))])
    phi = seq.phi(pts).reshape(len(xy), len(z))
    for k, level in enumerate(seq.thresholds):
        z_true = (STACK_TOPS[k] - xy @ STACK_POLE[:2]) / STACK_POLE[2]
        found = np.full(len(xy), np.nan)
        for i, row in enumerate(phi):
            j = np.flatnonzero((row[:-1] >= level) & (row[1:] < level))  # phi falls upwards
            if len(j) == 1:
                t = (row[j[0]] - level) / (row[j[0]] - row[j[0] + 1])
                found[i] = z[j[0]] + t
        inside = (z_true > -195.0) & (z_true < -10.0)
        assert np.isfinite(found[inside]).all()
        distance = (found[inside] - z_true[inside]) * STACK_POLE[2]
        rms = float(np.sqrt(np.mean(distance**2)))
        print(
            f"\nstack: contact {STACK[k]}/{STACK[k + 1]}: RMS {rms:.3f} m, largest "
            f"{np.abs(distance).max():.3f} m over {inside.sum()} lines"
        )
        assert rms <= 1.0 and np.abs(distance).max() <= 3.0


def test_sequence_against_largest_field(stack):
    """The same logs through both models: label accuracy against the truth on random
    points of the drilled box. Sequence >= 0.99 and not worse than the largest-field
    model by more than 0.5 %; the two agree at >= 98 % of the points."""
    table, traces, seq = stack
    pts = np.random.default_rng(3).uniform([-90, -90, -200], [90, 90, -10], size=(40000, 3))
    truth = stack_truth(pts)
    by_sequence = np.array(STACK, dtype=object)[seq.labels(pts)]
    largest = fit_unit_fields(table, table["unit"], STACK, traces)
    by_largest = np.array(STACK, dtype=object)[largest.labels(pts)]
    a, b = float((by_sequence == truth).mean()), float((by_largest == truth).mean())
    agree = float((by_sequence == by_largest).mean())
    print(f"\nstack: accuracy sequence {a:.4f}, largest field {b:.4f}; they agree at {agree:.4f}")
    assert a >= 0.99 and a >= b - 0.005 and agree >= 0.98


def test_sequence_reproduces_logs_and_margins(stack):
    """The logged unit is reproduced at every point of the fit that is not on a contact
    (exact global fit), along the holes at 2 m, and the unit margins are positive exactly
    where the label is the unit."""
    table, traces, seq = stack
    xyz, values = seq.inputs
    off_contact = ~np.isin(np.round(values, 9), np.round(seq.thresholds, 9))
    assert off_contact.sum() > 500
    expected = np.searchsorted(seq.thresholds, values[off_contact], side="right")
    np.testing.assert_array_equal(seq.labels(xyz[off_contact]), expected)
    pts, row = interval_points(table, np.ones(len(table), dtype=bool), traces)
    got = np.array(STACK, dtype=object)[seq.labels(pts)]
    logged = table["unit"].to_numpy(dtype=object)[row]
    share = float((got == logged).mean())
    print(f"\nstack: logged unit reproduced at {share:.4%} of {len(pts)} points along the holes")
    assert share >= 0.999

    phi = np.random.default_rng(4).uniform(-60.0, 130.0, size=(1, 7, 5, 3))
    margin = seq.unit_margins(phi)
    assert margin.shape == (4, 7, 5, 3)
    label = seq.labels_from_values(phi.reshape(-1, 1)).reshape(7, 5, 3)
    np.testing.assert_array_equal(np.argmax(margin, axis=0), label)
    assert ((margin > 0.0).sum(axis=0) == 1).all()
    t1 = seq.thresholds[1]
    edge = seq.labels_from_values(np.array([[np.nan], [-1.0], [0.0], [t1 - 0.01], [t1], [1e3]]))
    np.testing.assert_array_equal(edge, [-1, 0, 1, 1, 2, 3])
    np.testing.assert_array_equal(seq.node_values(pts[:5])[:, 0], seq.phi(pts[:5]))


def test_sequence_options_violations_and_errors(stack):
    table, traces, seq = stack
    given = fit_sequence(
        table, table["unit"], STACK, traces, thickness={"B": 20.0, "A": 60.0},
        settings=FieldSettings(sample_spacing=4.0, far_spacing=25.0, anisotropy="auto"),
    )  # fmt: skip
    assert given.thickness["B"] == 20.0 and given.thickness["A"] == 60.0
    np.testing.assert_allclose(given.thresholds, [0.0, 20.0, 20.0 + given.thickness["C"]])
    assert (given.sample_spacing, given.far_spacing) == (4.0, 25.0)
    assert isinstance(given.anisotropy, Ellipsoid)
    assert angle_between(given.anisotropy.minor_axis, STACK_POLE) < 0.5
    pts = np.random.default_rng(5).uniform([-90, -90, -200], [90, 90, -10], size=(5000, 3))
    # other levels, the same contacts: the labels hardly change
    assert (given.labels(pts) == seq.labels(pts)).mean() > 0.98
    poled = fit_sequence(
        table, table["unit"], STACK, traces,
        settings=FieldSettings(anisotropy=Ellipsoid.from_axes([1, 0, 0], [0, 1, 0])),
    )  # fmt: skip
    np.testing.assert_allclose(poled.pole, [0.0, 0.0, 1.0], atol=1e-12)

    # a hole that logs C above B is reported; its points keep their own unit
    rows = []
    for hole in ("N1", "N2", "N3"):
        rows += [(hole, 0.0, 50.0, "A"), (hole, 50.0, 80.0, "B"), (hole, 80.0, 125.0, "C")]
        rows += [(hole, 125.0, 200.0, "D")]
    rows += [("ODD", 0.0, 40.0, "A"), ("ODD", 40.0, 70.0, "C"), ("ODD", 70.0, 120.0, "B")]
    rows += [("ODD", 120.0, 150.0, "C"), ("ODD", 150.0, 200.0, "D")]
    odd = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])
    ids = ["N1", "N2", "N3", "ODD"]
    collar = pd.DataFrame(
        {"hole_id": ids, "x": [0.0, 100.0, 0.0, 60.0], "y": [0.0, 0.0, 100.0, 60.0], "z": 0.0}
    ).assign(max_depth=200.0)
    survey = pd.DataFrame(
        {"hole_id": np.repeat(ids, 2), "depth": np.tile([0.0, 200.0], 4), "dip": -90.0}
    ).assign(azimuth=0.0)
    broken = fit_sequence(odd, odd["unit"], STACK, desurvey(collar, survey))
    assert broken.violations.to_dict("records") == [
        {"hole_id": "ODD", "depth": 40.0, "found": "C", "expected": "B"},
        {"hole_id": "ODD", "depth": 70.0, "found": "B", "expected": "D"},
    ]
    assert any("out of order" in e.item for e in broken.provenance)
    mids = np.array([[60.0, 60.0, -z] for z in (20.0, 55.0, 95.0, 135.0, 175.0)])
    got = [STACK[i] for i in broken.labels(mids)]
    assert got == ["A", "C", "B", "C", "D"]

    with pytest.raises(ValueError, match="at least two"):
        fit_sequence(table, table["unit"], ["A"], traces)
    with pytest.raises(ValueError, match="twice"):
        fit_sequence(table, table["unit"], ["A", "B", "A"], traces)
    with pytest.raises(ValueError, match="no logged run"):
        fit_sequence(table, table["unit"], ["A", "B", "Z"], traces)
    with pytest.raises(ValueError, match="not in the sequence"):
        fit_sequence(table, table["unit"], STACK, traces, thickness={"Z": 3.0})
    with pytest.raises(ValueError, match="> 0"):
        fit_sequence(table, table["unit"], STACK, traces, thickness={"B": 0.0})
    with pytest.raises(TypeError, match="anisotropy"):
        fit_sequence(table, table["unit"], STACK, traces, settings=FieldSettings(anisotropy=3))


def test_sequence_provenance(stack):
    table, traces, seq = stack
    kinds = {e.kind for e in seq.provenance}
    assert kinds == {"measured", "assumed"}
    assert all(e.stage == "lithology" and e.task == "P9.T10" for e in seq.provenance)
    assumed = {e.item: e for e in seq.provenance if e.kind == "assumed"}
    label = "A > B > C > D"
    assert f"Modelling mode of the sequence {label}" in assumed
    assert "dip 25" in assumed[f"Pole of the layering, sequence {label}"].detail
    assert assumed[f"Thickness of B in the sequence {label}"].value == pytest.approx(30.0, rel=5e-3)
    assert "2 complete" not in assumed[f"Thickness of A in the sequence {label}"].source
    measured = {e.item: e for e in seq.provenance if e.kind == "measured"}
    assert measured["Rock-type log: unit B"].count == int((table["unit"] == "B").sum())


# --------------------------------------------------------------------------
# intrusions: overprint order on two overlapping spheres
# --------------------------------------------------------------------------

X_CENTRE, X_RADIUS = np.array([-15.0, 0.0, -100.0]), 45.0
Y_CENTRE, Y_RADIUS = np.array([30.0, 0.0, -90.0]), 35.0


def in_x(points):
    return np.linalg.norm(np.asarray(points) - X_CENTRE, axis=1) < X_RADIUS


def in_y(points):
    return np.linalg.norm(np.asarray(points) - Y_CENTRE, axis=1) < Y_RADIUS


def two_spheres(points):
    """Y is the younger: it is what the log shows where the two overlap."""
    points = np.asarray(points)
    host = np.where(points[:, 2] >= -95.0 + 0.2 * points[:, 0], "UP", "LOW")
    return np.where(in_y(points), "Y", np.where(in_x(points), "X", host)).astype(object)


@pytest.fixture(scope="module")
def overprint():
    table, traces = drill(two_spheres, collar_grid(-100.0, 100.0, 20.0), length=200.0)
    units = table["unit"]
    host = fit_sequence(table, units, ["UP", "LOW"], traces)
    x = fit_intrusion(table, mask_younger(units, ["X", "Y"], "X"), "X", traces)
    y = fit_intrusion(table, units, "Y", traces)
    return table, traces, host, x, y


def test_overprint_later_event_wins(overprint):
    """Hand case: where both intrusion fields are positive the later event gives the
    label, so swapping the order changes the labels exactly there; everywhere else the
    two stacks agree. With the true order (Y last) the accuracy is >= 0.93 and higher than
    with the wrong order. The spheres (r = 45 m and 35 m, holes 20 m apart) come out
    too small as signed-distance bodies (measured IoU 0.80 and 0.65), so in the true
    overlap only a loose share is asserted: Y at > 60 % with Y last, < 35 % with X last."""
    table, traces, host, x, y = overprint
    assert x.shape == y.shape == "body" and x.mode == "intrusion" and x.units == ("X",)
    assert isinstance(x, LithologyEvent)
    pts = lattice((-90.0, -90.0, -190.0), (90.0, 90.0, -10.0), 4.0)
    fx, fy = x.values(pts), y.values(pts)
    both = (fx > 0.0) & (fy > 0.0)
    assert both.sum() > 200
    young_last = LithologyStack("ALL", (host, x, y))
    old_last = LithologyStack("ALL", (host, y, x))
    a = np.array(young_last.categories, dtype=object)[young_last.labels(pts)]
    b = np.array(old_last.categories, dtype=object)[old_last.labels(pts)]
    assert (a[both] == "Y").all() and (b[both] == "X").all()
    np.testing.assert_array_equal(a[~both], b[~both])
    only_x, only_y, neither = (fx > 0) & ~both, (fy > 0) & ~both, (fx <= 0) & (fy <= 0)
    assert (a[only_x] == "X").all() and (a[only_y] == "Y").all()
    assert np.isin(a[neither], ["UP", "LOW"]).all()
    truth = two_spheres(pts)
    right, wrong = float((a == truth).mean()), float((b == truth).mean())
    overlap = in_x(pts) & in_y(pts)
    print(
        f"\ntwo spheres: accuracy with Y last {right:.4f}, with X last {wrong:.4f}; in the true "
        f"overlap ({overlap.sum()} points) Y last gives Y at {(a[overlap] == 'Y').mean():.3f}, "
        f"X last at {(b[overlap] == 'Y').mean():.3f}; IoU X {iou(a == 'X', truth == 'X'):.3f}, "
        f"Y {iou(a == 'Y', truth == 'Y'):.3f}"
    )
    assert right >= 0.93 and right > wrong
    assert (a[overlap] == "Y").mean() > 0.60 and (b[overlap] == "Y").mean() < 0.35


def test_younger_intrusion_is_unknown_for_the_older(overprint):
    """With Y made unknown, X's field has no outside points in Y's intervals: the older
    sphere is not dented by the younger one. X's volume against the full sphere (r = 45 m,
    holes 20 m apart): within 20 % (measured -17 %); without the mask it is smaller (-24 %)."""
    table, traces, host, x, y = overprint
    units = table["unit"]
    masked = mask_younger(units, ["X", "Y"], "X")
    assert masked.isna().sum() == (units == "Y").sum() and (masked.dropna() != "Y").all()
    np.testing.assert_array_equal(mask_younger(units, ["X", "Y"], "Y"), units)
    with pytest.raises(ValueError, match="not in the age order"):
        mask_younger(units, ["X", "Y"], "Q")
    dented = fit_intrusion(table, units, "X", traces)
    pts = lattice(X_CENTRE - 60.0, X_CENTRE + 60.0, 3.0)
    analytic = 4.0 / 3.0 * np.pi * X_RADIUS**3
    full = (x.values(pts) > 0).sum() * 27.0
    cut = (dented.values(pts) > 0).sum() * 27.0
    print(
        f"\nsphere X: volume {full / analytic - 1:+.2%} with Y unknown, "
        f"{cut / analytic - 1:+.2%} without"
    )
    assert full == pytest.approx(analytic, rel=0.20) and cut < full


def test_order_intrusions():
    order, entries = order_intrusions(["DIO", "GRD"])
    assert order == ("DIO", "GRD")
    assert [(e.kind, e.item) for e in entries] == [
        ("missing", "Age order of intrusions"),
        ("assumed", "Age order of intrusions used"),
    ]
    assert "geologist" in entries[0].source and "DIO, GRD" in entries[0].detail
    assert entries[1].detail == "DIO < GRD"
    order, entries = order_intrusions(["DIO", "GRD"], age_order=["GRD", "DIO"])
    assert order == ("GRD", "DIO")
    assert [(e.kind, e.source) for e in entries] == [("assumed", "given by the user, oldest first")]
    assert order_intrusions(["DIO"]) == (("DIO",), [])
    assert order_intrusions([]) == ((), [])
    with pytest.raises(ValueError, match="every intrusion once"):
        order_intrusions(["DIO", "GRD"], age_order=["DIO"])
    with pytest.raises(ValueError, match="twice"):
        order_intrusions(["DIO", "DIO"])


# --------------------------------------------------------------------------
# sheets
# --------------------------------------------------------------------------

SHEET_NORMAL = normal_of(90.0, 120.0)  # vertical, strike 030
SHEET_THICKNESS = 6.0


def vertical_sheet(points):
    w = np.asarray(points) @ SHEET_NORMAL
    return np.where(np.abs(w) < SHEET_THICKNESS / 2.0, "DYKE", "HOST").astype(object)


def sheet_holes(n):
    """Holes inclined 55 degrees towards the sheet from different distances and places
    along strike, so that the intercepts are spread over the plane."""
    strike = np.array([SHEET_NORMAL[1], -SHEET_NORMAL[0], 0.0])
    along = np.array([-75.0, -40.0, -5.0, 30.0, 65.0, 95.0, -60.0, 10.0])[:n]
    back = np.array([35.0, 80.0, 50.0, 95.0, 30.0, 70.0, 110.0, 20.0])[:n]
    return along[:, None] * strike - back[:, None] * SHEET_NORMAL


@pytest.fixture(scope="module")
def dyke():
    table, traces = drill(vertical_sheet, sheet_holes(6), dip=-55.0, azimuth=120.0, length=220.0)
    return table, traces, fit_intrusion(table, table["unit"], "DYKE", traces)


def test_vertical_sheet_thickness_orientation_volume(dyke):
    """A vertical sheet 6 m thick, strike 030, cut by six holes inclined 55 degrees.

    Tolerances: pole within 0.5 degrees of the truth, true thickness within 1 %, volume
    inside the area of the intercepts within 3 % of thickness x area (lattice count at
    1 m), IoU >= 0.95. The signed-distance body on the same holes is printed for
    comparison: it does not connect the intercepts."""
    table, traces, sheet = dyke
    assert sheet.shape == "sheet" and sheet.field is None and sheet.mode == "intrusion"
    geometry = sheet.sheet
    assert geometry.info["n_holes"] == 6 and geometry.info["n_barren_crossings"] == 0
    assert angle_between(geometry.normal, SHEET_NORMAL) < 0.5
    assert geometry.dip == pytest.approx(90.0, abs=0.5)
    assert min(abs(geometry.dip_azimuth - a) for a in (120.0, 300.0)) < 0.5
    assert geometry.info["thickness_median"] == pytest.approx(SHEET_THICKNESS, rel=0.01)
    assert geometry.info["misfit_rms"] < 0.05

    # a box around the intercepts, in sheet coordinates: 150 m along strike, 90 m high
    strike = np.array([SHEET_NORMAL[1], -SHEET_NORMAL[0], 0.0])
    local = lattice((-70.0, -110.0, -15.0), (80.0, -20.0, 15.0), 1.0)
    pts = local[:, [0]] * strike + local[:, [2]] * SHEET_NORMAL + local[:, [1]] * [0.0, 0.0, 1.0]
    truth = vertical_sheet(pts) == "DYKE"
    got = sheet.labels(pts) == 0
    analytic = SHEET_THICKNESS * 150.0 * 90.0
    volume = float(got.sum())
    body = fit_intrusion(table, table["unit"], "DYKE", traces, shape="body")
    blob = body.labels(pts) == 0
    print(
        f"\nvertical sheet: pole {angle_between(geometry.normal, SHEET_NORMAL):.3f} deg off, "
        f"thickness {geometry.info['thickness_median']:.3f} m (6), "
        f"volume {volume / analytic - 1:+.2%}, IoU {iou(got, truth):.3f}; "
        f"as a signed-distance body: volume {blob.sum() / analytic - 1:+.2%}, "
        f"IoU {iou(blob, truth):.3f}"
    )
    assert truth.sum() == pytest.approx(analytic, rel=0.01)
    assert volume == pytest.approx(analytic, rel=0.03)
    assert iou(got, truth) >= 0.95
    assert iou(blob, truth) < iou(got, truth)


def test_sheet_provenance_and_reach(dyke):
    """Orientation, thickness and continuation are recorded as assumptions, and the sheet
    ends ``reach`` beyond the nearest intercept (``max_extrapolation`` when given)."""
    table, traces, sheet = dyke
    assumed = {e.item: e for e in sheet.provenance if e.kind == "assumed"}
    assert set(assumed) == {
        "Shape of the intrusion DYKE",
        "Orientation of the sheet DYKE",
        "Thickness of the sheet DYKE",
        "Continuation of the sheet DYKE beyond the last intercept",
    }
    assert "chosen automatically" in assumed["Shape of the intrusion DYKE"].source
    assert "dip 90" in assumed["Orientation of the sheet DYKE"].detail
    assert assumed["Thickness of the sheet DYKE"].value == pytest.approx(6.0, rel=0.01)
    reach = assumed["Continuation of the sheet DYKE beyond the last intercept"]
    geometry = sheet.sheet
    assert reach.value == pytest.approx(geometry.reach)
    assert geometry.reach == pytest.approx(2.0 * geometry.info["neighbour_distance"])
    assert [e.kind for e in sheet.provenance].count("measured") == 1

    short = fit_intrusion(
        table, table["unit"], "DYKE", traces, settings=FieldSettings(max_extrapolation=25.0)
    )
    assert short.sheet.reach == 25.0
    assert "max_extrapolation" in short.provenance[-1].source
    # on the plane, far along strike: inside within the reach of an intercept, not beyond
    strike = np.array([SHEET_NORMAL[1], -SHEET_NORMAL[0], 0.0])
    uv = short.sheet.present
    distance = np.array([20.0, 30.0])
    edge = uv[np.argmax(uv[:, 0])]
    probe = (
        short.sheet.origin + (edge[0] + distance)[:, None] * short.sheet.u + edge[1] * short.sheet.v
    )
    np.testing.assert_array_equal(short.labels(probe), [0, -1])
    assert abs(float(short.sheet.u @ strike)) == pytest.approx(1.0, abs=1e-4)


def test_sheet_needs_enough_holes_else_body():
    """Three holes do not make a sheet (needs MIN_SHEET_HOLES = 4): the intrusion falls
    back to the signed-distance body and the provenance says why, also when a sheet was
    asked for. A thick body is not taken for a sheet automatically."""
    assert MIN_SHEET_HOLES == 4
    table, traces = drill(vertical_sheet, sheet_holes(3), dip=-55.0, azimuth=120.0, length=220.0)
    for shape, text in (("auto", "chosen automatically"), ("sheet", "is not determined")):
        fallback = fit_intrusion(table, table["unit"], "DYKE", traces, shape=shape)
        assert fallback.shape == "body" and fallback.sheet is None
        entry = next(e for e in fallback.provenance if e.item == "Shape of the intrusion DYKE")
        assert entry.kind == "assumed" and text in entry.source
        assert "3 holes with a complete intercept" in entry.detail and "needs 4" in entry.detail
    asked = fit_intrusion(table, table["unit"], "DYKE", traces, shape="body")
    assert "as asked" in asked.provenance[1].source and asked.provenance[1].detail == ""

    def slab(points):  # 60 m thick: resolved by holes 40 m apart
        return np.where(np.abs(np.asarray(points) @ SHEET_NORMAL) < 30.0, "DYKE", "HOST").astype(
            object
        )

    table, traces = drill(slab, sheet_holes(8), dip=-55.0, azimuth=120.0, length=320.0)
    thick = fit_intrusion(table, table["unit"], "DYKE", traces)
    assert thick.shape == "body"
    assert "holes resolve the body" in thick.provenance[1].detail
    forced = fit_intrusion(table, table["unit"], "DYKE", traces, shape="sheet")
    assert forced.shape == "sheet"
    assert forced.sheet.info["thickness_median"] == pytest.approx(60.0, rel=0.01)

    with pytest.raises(ValueError, match="shape"):
        fit_intrusion(table, table["unit"], "DYKE", traces, shape="vein")
    with pytest.raises(ValueError, match="no logged run"):
        fit_intrusion(table, table["unit"], "NONE", traces)


CURVED_NORMAL = normal_of(70.0, 90.0)  # dips 70 degrees to the east


def curved_sheet(points):
    """A gently curved sheet, 4 to 8 m thick, that ends at 60 m north (it pinches out)."""
    p = np.asarray(points)
    w = p @ CURVED_NORMAL - 6.0 * np.sin(p[:, 1] / 70.0) - 0.0004 * (p[:, 2] + 80.0) ** 2
    half = 3.0 + np.cos(p[:, 1] / 60.0)
    return np.where((np.abs(w) < half) & (p[:, 1] < 60.0), "DYKE", "HOST").astype(object)


def test_curved_sheet_that_pinches_out():
    """A curved sheet of varying thickness that stops inside the drilling, holes 40 m
    apart inclined 60 degrees against the dip. Not an exact case for the method (the
    surface between the holes is interpolated, the end is placed half-way to the first
    barren hole): IoU >= 0.60 and volume within 20 % are asserted; the body model on
    the same holes is printed for comparison."""
    axis = np.arange(-100.0, 100.0 + 1e-9, 40.0)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    collars = np.column_stack([x.ravel() + 40.0, y.ravel(), np.zeros(x.size)])
    table, traces = drill(curved_sheet, collars, dip=-60.0, azimuth=270.0, length=220.0)
    sheet = fit_intrusion(table, table["unit"], "DYKE", traces)
    assert sheet.shape == "sheet"
    geometry = sheet.sheet
    assert geometry.info["n_barren_crossings"] > 0
    assert angle_between(geometry.normal, CURVED_NORMAL) < 8.0
    assert 4.0 < geometry.info["thickness_median"] < 8.0
    assert geometry.info["misfit_rms"] > 0.5  # not a plane: the offsets are interpolated
    pts = lattice((-120.0, -100.0, -170.0), (140.0, 100.0, -10.0), 2.0)
    truth = curved_sheet(pts) == "DYKE"
    got = sheet.labels(pts) == 0
    body = fit_intrusion(table, table["unit"], "DYKE", traces, shape="body").labels(pts) == 0
    ratio = got.sum() / truth.sum()
    info = geometry.info
    print(
        f"\ncurved sheet: {info['n_holes']} holes, {info['n_barren_crossings']} barren "
        f"crossings, pole {angle_between(geometry.normal, CURVED_NORMAL):.1f} deg off, "
        f"thickness {info['thickness_min']:.1f} to {info['thickness_max']:.1f} m; "
        f"IoU {iou(got, truth):.3f}, volume {ratio - 1:+.1%}; "
        f"as a body: IoU {iou(body, truth):.3f}, volume {body.sum() / truth.sum() - 1:+.1%}"
    )
    assert iou(got, truth) >= 0.60
    assert ratio == pytest.approx(1.0, abs=0.20)
    north = pts[:, 1] > 85.0  # beyond the first barren row: nothing
    assert got[north].sum() == 0


# --------------------------------------------------------------------------
# the synthetic layered deposit: sequence, then the stock, then the dyke
# (still without fault handling, P9.T13)
# --------------------------------------------------------------------------

LAYERS = ("SST", "SLT", "AND")
INTRUSIONS = ("DIO", "GRD")  # oldest first
GROUP = LAYERS + INTRUSIONS
#: geology_contracts.md section 14
TARGET_LAYER, TARGET_STOCK, TARGET_DYKE = 0.85, 0.80, 0.50


@pytest.fixture(scope="module")
def layered(layered_small, layered_small_dir, layered_small_db, layered_small_traces):
    units = UnitDictionary.load(layered_small_dir / "units.json")
    assert units.codes(*MODELLED_KINDS) == GROUP
    table = layered_small_db.intervals["geology"]
    mapped = units.map_codes(table["lith"])
    rock = mapped.where(mapped.isin(GROUP), np.nan)  # cover and weathering codes: unknown
    traces = layered_small_traces
    order, order_entries = order_intrusions(INTRUSIONS, age_order=INTRUSIONS)
    events = [fit_sequence(table, rock, LAYERS, traces)]
    events += [fit_intrusion(table, mask_younger(rock, order, u), u, traces) for u in order]
    cfg = layered_small.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    shape = tuple(int(n) for n in np.round((hi - lo) / 5.0) + 1)
    model = GeologyModel(
        name="layered-small, sequence and intrusions, no faults",
        units=units,
        grid=GridSpec(lo, (5.0, 5.0, 5.0), shape),
        topography=DTM(read_obj(layered_small_dir / "topo.obj")),
        stacks={"ALL": LithologyStack("ALL", tuple(events))},
        hole_points=logged_hole_points(table, traces, step=5.0, known=rock.notna()),
    )
    model.sample_grid()
    return model, events, (lo, hi), table, rock, traces


def test_layered_shapes_and_dyke_geometry(layered, layered_small):
    """The stock is a body, the dyke a sheet chosen automatically; the dyke's plane and
    thickness against the truth: pole within 1 degree, thickness within 2 % of 8 m."""
    model, (seq, dio, grd), _, _, _, _ = layered
    assert (seq.mode, dio.shape, grd.shape) == ("sequence", "body", "sheet")
    truth_normal = normal_of(90.0, 135.0)
    geometry = grd.sheet
    print(
        f"\nlayered-small: SLT thickness {seq.thickness['SLT']:.1f} m (truth 40 +/- 10), "
        f"{len(seq.violations)} order violations; dyke from {geometry.info['n_holes']} holes: pole "
        f"{angle_between(geometry.normal, truth_normal):.3f} deg off, thickness "
        f"{geometry.info['thickness_median']:.2f} m (8), reach {geometry.reach:.0f} m; stock cap "
        f"{dio.cap:.1f} m, {dio.field.n_nodes} points"
    )
    assert angle_between(geometry.normal, truth_normal) < 1.0
    assert geometry.info["thickness_median"] == pytest.approx(8.0, rel=0.02)
    assert 30.0 < seq.thickness["SLT"] < 50.0


def test_layered_iou_and_volumes(layered, layered_small):
    """IoU per unit against the exact ``bedrock`` truth on the 2.5 m lattice inside the
    drilled hull, below the topography (the lattice of P9.T7), with the grid labels.

    Section 14 targets, asserted because they are met: the layered units >= 0.85, the
    stock >= 0.80, the dyke >= 0.50. The synthetic dyke is an exact plane of constant
    thickness, which is the very model fitted, so its IoU is a best case.

    Volumes: uniform random points in the whole box, against the truth at the same
    points and against ``truth.json`` (which counts the cover separately; this model has
    no cover surface, P9.T9). Outside the drilled hull the model extrapolates: only the
    figures inside the hull are held to 12 %."""
    model, _, (lo, hi), _, _, _ = layered
    pts = lattice(lo, hi, 2.5)
    pts = pts[Delaunay(model.hole_points).find_simplex(pts) >= 0]
    truth = layered_small.truth.labels(pts, "bedrock")
    got = model.decode(model.labels(pts), "lith")
    ground = (truth != "AIR") & (got != "AIR")
    got, truth = got[ground], truth[ground]
    assert (got != None).all()  # noqa: E711
    scores = {u: iou(got == u, truth == u) for u in GROUP}
    inside = {u: float((got == u).sum() / (truth == u).sum()) for u in GROUP}
    accuracy = float((got == truth).mean())
    print(
        f"\nlayered-small, sequence + stock + dyke, no fault handling, {len(got)} lattice points "
        f"in the hull: accuracy {accuracy:.4f}; "
        + ", ".join(f"IoU {u} {v:.3f}" for u, v in scores.items())
        + "; volume model / truth in the hull: "
        + ", ".join(f"{u} {v:.3f}" for u, v in inside.items())
    )
    for u in LAYERS:
        assert scores[u] >= TARGET_LAYER, (u, scores[u])
    assert scores["DIO"] >= TARGET_STOCK
    assert scores["GRD"] >= TARGET_DYKE
    assert accuracy >= 0.95
    for u in GROUP:
        assert inside[u] == pytest.approx(1.0, abs=0.12), (u, inside[u])

    rng = np.random.default_rng(10)
    box = rng.uniform(lo, hi, size=(1_500_000, 3))
    box_volume = float(np.prod(hi - lo))
    mine = model.decode(model.labels(box), "lith")
    bedrock = layered_small.truth.labels(box, "bedrock")
    with_cover = layered_small.truth.labels(box, "lith")
    stated = layered_small.truth_json["volumes"]["units"]
    lines = []
    for u in GROUP:
        v = float((mine == u).mean()) * box_volume
        same = float((bedrock == u).mean()) * box_volume
        listed = stated[u]["volume_m3"]
        lines.append(
            f"{u} {v / 1e6:.3f} Mm3 (truth without cover at the same points {same / 1e6:.3f}: "
            f"{v / same - 1:+.1%}; truth.json {listed / 1e6:.3f}: {v / listed - 1:+.1%})"
        )
        # the random points reproduce the volume stated in truth.json (which has the cover)
        assert float((with_cover == u).mean()) * box_volume == pytest.approx(listed, rel=0.02)
        assert v == pytest.approx(same, rel=0.35)  # regression bound; extrapolation included
    print("\nlayered-small, whole box below the topography: " + "; ".join(lines))


def test_layered_logs_reproduced(layered):
    """The logged unit along the holes (points every 2 m of every interval of a modelled
    unit): exact evaluation >= 99.5 %, grid labels >= 98 %."""
    model, _, _, table, rock, traces = layered
    pts, row = interval_points(table, rock.notna().to_numpy(), traces)
    logged = rock.to_numpy(dtype=object)[row]
    exact = float((model.decode(model.labels(pts, exact=True), "lith") == logged).mean())
    fast = float((model.decode(model.labels(pts), "lith") == logged).mean())
    print(
        f"\nlayered-small: logged unit reproduced at {exact:.3%} (exact) and {fast:.3%} (5 m grid) "
        f"of {len(pts)} points along the holes"
    )
    assert exact >= 0.995 and fast >= 0.98


def test_layered_grid_labels_and_meshes(layered, layered_small):
    """Grid labels against exact labels (section 14: >= 99.5 %), and a closed mesh for
    every unit from the events' margins; the dyke's mesh volume within 10 % of the truth
    (the whole box; the sheet reaches its edges here)."""
    model, _, (lo, hi), _, _, _ = layered
    pts = np.random.default_rng(11).uniform(lo, hi, size=(20000, 3))
    agreement = float((model.labels(pts, exact=True) == model.labels(pts)).mean())
    print(f"\nlayered-small: grid labels equal exact labels at {agreement:.3%} of 20000 points")
    assert agreement >= 0.995
    meshes = model.meshes("lith")
    assert set(meshes) == set(GROUP)
    assert all(mesh.is_closed for mesh in meshes.values())
    stated = layered_small.truth_json["volumes"]["units"]["GRD"]["volume_m3"]
    print(f"\nlayered-small: dyke mesh {meshes['GRD'].volume:.0f} m3, truth.json {stated:.0f} m3")
    assert meshes["GRD"].volume == pytest.approx(stated, rel=0.10)


def test_layered_provenance(layered):
    model, events, _, _, _, _ = layered
    entries = model.provenance_entries()
    assumed = {e.item for e in entries if e.kind == "assumed"}
    assert {
        "Modelling mode of the sequence SST > SLT > AND",
        "Shape of the intrusion DIO",
        "Shape of the intrusion GRD",
        "Orientation of the sheet GRD",
        "Thickness of the sheet GRD",
        "Continuation of the sheet GRD beyond the last intercept",
        "Cap of the signed distance, intrusion DIO",
    } <= assumed
    measured = {e.item for e in entries if e.kind == "measured"}
    assert {"Rock-type log: intrusion DIO", "Rock-type log: intrusion GRD"} <= measured
    assert all(e.hole_spacing is not None for e in events)


def test_layered_save_load_round_trip(layered, tmp_path):
    """A model with a sequence, a body and a sheet: saved and loaded, the labels are
    identical (exact and grid), and so are the events' values."""
    model, events, (lo, hi), _, _, _ = layered
    folder = model.save(tmp_path / "layered.geology")
    manifest = json.loads((folder / "geology.json").read_text())
    entries = manifest["stacks"]["ALL"]
    assert [e["mode"] for e in entries] == ["sequence", "intrusion", "intrusion"]
    assert entries[0]["fields"] == ["fields/ALL_0_phi.npz"]
    assert entries[1]["fields"] == ["fields/ALL_1_DIO.npz"] and entries[1]["shape"] == "body"
    assert entries[2]["fields"] == ["fields/ALL_2_surface.npz", "fields/ALL_2_thickness.npz"]
    assert entries[2]["shape"] == "sheet"
    loaded = GeologyModel.load(folder)
    pts = np.random.default_rng(12).uniform(lo, hi, size=(20000, 3))
    np.testing.assert_array_equal(loaded.labels(pts), model.labels(pts))
    np.testing.assert_array_equal(loaded.labels(pts, exact=True), model.labels(pts, exact=True))
    again = loaded.stacks["ALL"].events
    assert [type(e) for e in again] == [SequenceField, IntrusionField, IntrusionField]
    for old, new in zip(events, again, strict=True):
        np.testing.assert_array_equal(new.node_values(pts[:2000]), old.node_values(pts[:2000]))
    np.testing.assert_array_equal(again[0].thresholds, events[0].thresholds)
    assert again[0].thickness == events[0].thickness and again[2].shape == "sheet"
    with pytest.raises(ValueError, match="mode"):
        event_from_manifest({"mode": "argmax", "units": ["A"]}, {}, {})


def test_fit_stack_dispatches_to_this_module(layered):
    """``fit_stack`` with sequence and intrusion specs gives the same labels as the
    events fitted directly (the same units table for every event)."""
    model, events, (lo, hi), table, rock, traces = layered
    stack = fit_stack(
        table,
        rock,
        traces,
        [
            EventSpec("sequence", LAYERS, thickness={"SLT": 40.0}),
            EventSpec("intrusion", ("DIO",)),
            EventSpec("intrusion", ("GRD",)),
        ],
    )
    assert stack.categories == GROUP
    assert [e.mode for e in stack.events] == ["sequence", "intrusion", "intrusion"]
    assert stack.events[0].thickness["SLT"] == 40.0 and stack.events[2].shape == "sheet"
    pts = np.random.default_rng(13).uniform(lo, hi, size=(5000, 3))
    direct = LithologyStack("ALL", tuple(events)).labels(pts)
    assert (stack.labels(pts) == direct).mean() > 0.98
