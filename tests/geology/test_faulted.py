"""Rock units per fault block and the throw diagnostic (P9.T13): contracts 7, 9, 14.

Independent checks: analytic truths (flat layers stepped by a vertical fault; tilted
layers moved by a known dip slip on a dipping fault), the same data fitted without the
fault, analytic volumes and trimesh for the meshes, bit-identical labels after a save /
load round trip, and the exact truth of the synthetic layered deposit. All other data are
made up.
"""

from __future__ import annotations

import json
import math
import time

import numpy as np
import pandas as pd
import pytest
import trimesh
from scipy.spatial import Delaunay

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geology.faulted import (
    MIN_THROW_HOLES,
    POOLED,
    BlockStacks,
    assign_blocks,
    block_stack_builder,
    fit_block_stacks,
    logged_contacts,
    split_at_faults,
    throw_diagnostic,
    throw_diagnostics,
    throw_entry,
)
from bakumine.core.geology.faults import (
    FaultNetwork,
    Plane,
    block_codes,
    fault_picks,
    import_fault,
    infer_fault,
    labelled_points,
)
from bakumine.core.geology.lithology import (
    EventSpec,
    LithologyStack,
    fit_stack,
    logged_hole_points,
)
from bakumine.core.geology.model import LAYERS, GeologyModel
from bakumine.core.geology.sequence import mask_younger
from bakumine.core.geology.units import MODELLED_KINDS, Unit, UnitDictionary
from bakumine.core.geology.validation import cross_validate, stack_builder
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.isosurface import GridSpec
from bakumine.core.io.obj import read_obj

# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def drill(label, collars, *, length=200.0, scan=0.5):
    """Vertical holes logged from ``label(points) -> codes``; contacts by bisection.
    Returns ``(table, traces)`` with columns ``hole_id, depth_from, depth_to, unit``."""
    collars = np.asarray(collars, dtype=np.float64)
    n = len(collars)
    down = np.array([0.0, 0.0, -1.0])
    ids = [f"H{i + 1:03d}" for i in range(n)]
    collar = pd.DataFrame(
        {"hole_id": ids, "x": collars[:, 0], "y": collars[:, 1], "z": collars[:, 2]}
    ).assign(max_depth=length)
    survey = pd.DataFrame(
        {
            "hole_id": np.repeat(ids, 2),
            "depth": np.tile([0.0, length], n),
            "dip": -90.0,
            "azimuth": 0.0,
        }
    )
    depths = np.arange(0.0, length + 1e-9, scan)
    rows = []
    for i in range(n):

        def at(depth, i=i):
            return label(collars[i] + np.atleast_1d(depth)[:, None] * down)

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


def collar_grid(lo, hi, spacing):
    axis = np.arange(lo, hi + 1e-9, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), np.zeros(x.size)])


def lattice(lo, hi, spacing):
    axes = [np.arange(a + spacing / 2.0, b, spacing) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


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
# analytic case 1: flat layers stepped 25 m by a vertical fault
# --------------------------------------------------------------------------

STACK = ("A", "B", "C")
UNITS = UnitDictionary(
    name="made-up faulted layers",
    units=(*(Unit(code=c, kind="lithology") for c in STACK), Unit(code="P", kind="intrusion")),
)
BOX_LO = np.array([-150.0, -150.0, -200.0])
BOX_HI = np.array([150.0, 150.0, 0.0])
GRID = GridSpec(BOX_LO, (5.0, 5.0, 5.0), (61, 61, 41))
TOPS = (-60.0, -100.0)  # contacts A/B and B/C west of the fault
THROW = 25.0  # the east side is down
EVENTS = [EventSpec("sequence", STACK)]


def flat_truth(points):
    p = np.asarray(points, dtype=np.float64)
    z = p[:, 2] + THROW * (p[:, 0] > 0.0)
    return np.array(STACK, dtype=object)[(z < TOPS[0]).astype(int) + (z < TOPS[1])]


def vertical_network() -> FaultNetwork:
    fault = import_fault(
        "F", plane=Plane.from_dip((0.0, 0.0, -100.0), 90.0, 90.0), blocks=("W", "E")
    )
    return FaultNetwork.fit([fault], np.zeros((0, 3)), [])


def build(table, traces, network, events=EVENTS, **options):
    stacks, pooled, entries = fit_block_stacks(
        table, table["unit"], traces, network, events, **options
    )
    model = GeologyModel(
        name="made-up faulted layers",
        units=UNITS,
        grid=GRID,
        topography=None,
        stacks=stacks,
        network=network,
        hole_points=logged_hole_points(table, traces, step=5.0),
    )
    model.pooled_blocks = pooled
    model.sample_grid()
    return model, entries


def unfaulted(table, traces, events=EVENTS) -> GeologyModel:
    model = GeologyModel(
        name="no fault handling",
        units=UNITS,
        grid=GRID,
        topography=None,
        stacks={"ALL": fit_stack(table, table["unit"], traces, events)},
        hole_points=logged_hole_points(table, traces, step=5.0),
    )
    model.sample_grid()
    return model


@pytest.fixture(scope="module")
def flat():
    table, traces = drill(flat_truth, collar_grid(-135.0, 135.0, 30.0))
    network = vertical_network()
    model, entries = build(table, traces, network)
    return model, entries, table, traces, network


def test_assign_blocks_and_split_at_the_fault():
    """Midpoint rule by hand, and an interval that crosses a dipping fault is cut at the
    crossing (analytic depth) without changing what is logged."""
    collars = np.array([[-40.0, 0.0, 0.0], [-10.0, 0.0, 0.0], [30.0, 0.0, 0.0]])
    table, traces = drill(lambda p: np.full(len(p), "A", dtype=object), collars)
    # dip 45 to the east through (0, 0, -100): a vertical hole at x meets it at depth 100 + x
    fault = import_fault(
        "F", plane=Plane.from_dip((0.0, 0.0, -100.0), 45.0, 90.0), blocks=("W", "E")
    )
    network = FaultNetwork.fit([fault], np.zeros((0, 3)), [])
    # midpoints at depth 100: the footwall (W) where 100 > 100 + x
    assert assign_blocks(table, traces, network).tolist() == ["W", "W", "E"]
    pieces = split_at_faults(table, traces, network)
    assert pieces["row"].tolist() == [0, 0, 1, 1, 2, 2]
    np.testing.assert_allclose(
        pieces["depth_to"].to_numpy()[[0, 2, 4]], [60.0, 90.0, 130.0], atol=1e-3
    )
    assert pieces["block"].tolist() == ["E", "W", "E", "W", "E", "W"]
    assert pieces["depth_from"].iloc[0] == 0.0 and pieces["depth_to"].iloc[-1] == 200.0
    # a hole without a trace: no block, and the interval is kept whole
    extra = pd.concat(
        [table, pd.DataFrame({"hole_id": ["NOPE"], "depth_from": [0.0], "depth_to": [10.0]})],
        ignore_index=True,
    )
    assert pd.isna(assign_blocks(extra, traces, network).iloc[-1])
    assert len(split_at_faults(extra, traces, network)) == 7
    with pytest.raises(ValueError, match="step"):
        split_at_faults(table, traces, network, step=0.0)


def test_flat_layers_labels_against_the_truth(flat):
    """Grid labels against the analytic truth on a 4 m lattice inside the drilled outline.
    Tolerance: >= 99 % overall and >= 98 % within 30 m of the fault. The same data fitted
    without the fault is wrong at >= 3 % of the points within 30 m of the fault (measured
    3.7 %: a 25 m step on two contacts in 200 m of depth, holes 30 m apart)."""
    model, _, table, traces, network = flat
    pts = lattice(BOX_LO + [15.0, 15.0, 0.0], BOX_HI - [15.0, 15.0, 0.0], 4.0)
    truth = flat_truth(pts)
    got = model.decode(model.labels(pts), "lith")
    plain = unfaulted(table, traces)
    other = plain.decode(plain.labels(pts), "lith")
    near = np.abs(pts[:, 0]) < 30.0
    accuracy, accuracy_near = float((got == truth).mean()), float((got[near] == truth[near]).mean())
    plain_near = float((other[near] == truth[near]).mean())
    print(
        f"\nflat layers, vertical fault, throw {THROW:g} m: accuracy {accuracy:.4f} "
        f"({len(pts)} points), within 30 m of the fault {accuracy_near:.4f}; without fault "
        f"handling {float((other == truth).mean()):.4f} and {plain_near:.4f}"
    )
    assert accuracy >= 0.99 and accuracy_near >= 0.98
    assert plain_near <= accuracy_near - 0.03
    exact = model.decode(model.labels(pts[::5], exact=True), "lith")
    assert (exact == truth[::5]).mean() >= 0.99
    east = pts[:, 0] > 0.0
    np.testing.assert_array_equal(
        model.decode(model.labels(pts, "fault_block"), "fault_block"), np.where(east, "E", "W")
    )
    # the stacks as one object give the same exact labels as the model
    both = BlockStacks(network, model.stacks)
    assert both.categories == STACK
    np.testing.assert_array_equal(
        np.array(both.categories, dtype=object)[both.labels(pts[::5])], exact
    )


def test_flat_layers_throw_recovered(flat):
    """The throw diagnostic against the known 25 m: within 1 m on both contacts, no heave
    on a vertical fault, planes flat (gradients < 0.01) at the analytic elevations within
    1 m; measured with the logged contacts, assumed without them."""
    model, _, table, traces, _ = flat
    for (upper, lower), top in zip((("A", "B"), ("B", "C")), TOPS, strict=True):
        logged = logged_contacts(table, table["unit"], traces, upper, lower)
        assert logged["hole_id"].nunique() == 100
        got = throw_diagnostic(model, "F", upper, lower, logged=logged)
        print(
            f"\nflat layers {upper}/{lower}: throw {got['throw']:.3f} m (truth {THROW:g}), spread "
            f"{got['spread']:.3f}, planes from {got['n_footwall']} / {got['n_hangingwall']} "
            f"points, rms {got['rms_footwall']:.3f} / {got['rms_hangingwall']:.3f} m"
        )
        assert got["throw"] == pytest.approx(THROW, abs=1.0)
        assert got["throw_dip_slip"] == pytest.approx(THROW, abs=1.0)
        assert got["heave_dip_slip"] == pytest.approx(0.0, abs=1e-6)
        assert got["at"] == "fault trace" and got["n_trace"] > 0
        assert got["blocks"] == ["W", "E"] and got["pooled"] == []
        for plane, level in ((got["plane_footwall"], top), (got["plane_hangingwall"], top - THROW)):
            assert plane[0] == pytest.approx(level, abs=1.0)
            assert abs(plane[1]) < 0.01 and abs(plane[2]) < 0.01
        # the holes 15 and 45 m from the fault on each side: 2 rows of 10
        assert (got["holes_footwall"], got["holes_hangingwall"]) == (20, 20)
        assert got["kind"] == "measured" and got["offset_seen"]
        entry = throw_entry(got)
        assert entry.kind == "measured" and entry.value == pytest.approx(got["throw"])
        assert "20 footwall, 20 hanging wall" in entry.source and entry.task == "P9.T13"
    bare = throw_diagnostic(model, "F", "A", "B")
    assert bare["throw"] == pytest.approx(THROW, abs=1.0)
    assert bare["kind"] == "assumed" and "not counted" in bare["note"]
    assert bare["holes_footwall"] is None
    # a band without contact points, and bad input
    empty = throw_diagnostic(model, "F", "A", "B", band=(400.0, 500.0))
    assert math.isnan(empty["throw"]) and "too few points" in empty["note"]
    assert throw_entry(empty).kind == "assumed"
    with pytest.raises(ValueError, match="no fault"):
        throw_diagnostic(model, "F9", "A", "B")
    with pytest.raises(ValueError, match="rock unit"):
        throw_diagnostic(model, "F", "A", "Z")
    with pytest.raises(ValueError, match="band"):
        throw_diagnostic(model, "F", "A", "B", band=(60.0, 20.0))
    # every contact of the sequence, with the holes counted
    results, entries = throw_diagnostics(model, table=table, units=table["unit"], traces=traces)
    assert [(r["upper"], r["lower"]) for r in results] == [("A", "B"), ("B", "C")]
    assert [e.kind for e in entries] == ["measured", "measured"]
    assert throw_diagnostics(unfaulted(table, traces)) == ([], [])


def test_flat_layers_meshes_meet_at_the_fault(flat):
    """Closed meshes, cut at the fault. Analytic volumes (no topography): each unit within
    2 %, and the three sum to the box within 0.5 % (no gaps, no overlaps). The step at the
    fault is in the mesh: unit A reaches 25 m deeper east of the fault than west of it."""
    model, _, _, _, _ = flat
    meshes = model.meshes("lith")
    assert list(meshes) == list(STACK)
    area = 150.0 * 300.0
    analytic = {
        "A": area * (-TOPS[0] + -TOPS[0] + THROW),
        "B": area * 2.0 * (TOPS[0] - TOPS[1]),
        "C": area * ((200.0 + TOPS[1]) + (200.0 + TOPS[1] - THROW)),
    }
    total = 0.0
    for code, mesh in meshes.items():
        other = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        assert mesh.is_closed and other.is_watertight, code
        assert mesh.volume == pytest.approx(analytic[code], rel=0.02), code
        total += mesh.volume
    box = float(np.prod(BOX_HI - BOX_LO))
    print(f"\nflat layers: mesh volumes sum to {total / box:.5f} of the box")
    assert total == pytest.approx(box, rel=0.005)
    v = meshes["A"].vertices
    west, east = v[v[:, 0] < -10.0], v[v[:, 0] > 10.0]
    assert west[:, 2].min() == pytest.approx(TOPS[0], abs=1.5)
    assert east[:, 2].min() == pytest.approx(TOPS[0] - THROW, abs=1.5)
    grid = model.fault_block_grid()
    assert grid.dtype == np.int16 and grid.shape == GRID.shape
    assert model.blocks == ("E", "W")  # sorted names; the index follows that order
    assert (grid[:30] == 1).all() and (grid[31:] == 0).all()  # node plane 30 is on the fault


def test_block_with_too_few_holes_uses_the_pooled_model():
    """Four holes east of the fault: the east block takes the pooled stack and the record
    says so; the west block keeps its own. With every block pooled the labels are those of
    a fit without faults, and the throw diagnostic is 'assumed'."""
    collars = collar_grid(-135.0, 135.0, 30.0)
    keep = (collars[:, 0] < 0.0) | ((collars[:, 0] < 20.0) & (np.abs(collars[:, 1]) < 60.0))
    table, traces = drill(flat_truth, collars[keep])
    network = vertical_network()
    stacks, pooled, entries = fit_block_stacks(table, table["unit"], traces, network, EVENTS)
    assert pooled == ["E"]
    assert stacks["E"].block == POOLED and stacks["W"].block == "W"
    assert stacks["E"] is not stacks["W"]
    (entry,) = [e for e in entries if "pooled" in e.item]
    assert entry.kind == "assumed" and entry.item == "Fault block E: pooled rock model"
    assert "4 holes" in entry.source and "fewer than the 5" in entry.source and entry.count == 4
    assert "do not step at the fault" in entry.detail
    counts = next(e for e in entries if e.item == "Rock-type log by fault block")
    assert counts.kind == "measured" and "E: 4 holes; W: 50 holes" in counts.detail
    age = next(e for e in entries if e.item.startswith("Age of the faults"))
    assert age.kind == "assumed" and "default" in age.source
    assert any("pooled" in e.item for e in stacks["E"].provenance)
    # with four holes allowed the block's own fit is tried; these four holes are on one line,
    # which does not give the pole of the layering: the fit fails and the block is pooled
    _, again, entries = fit_block_stacks(table, table["unit"], traces, network, EVENTS, min_holes=4)
    assert again == ["E"]
    (entry,) = [e for e in entries if "pooled" in e.item]
    assert entry.kind == "assumed" and "own fit failed" in entry.source

    model, _ = build(table, traces, network, min_holes=1000)
    assert model.pooled_blocks == ["E", "W"]
    assert model.stacks["W"] is model.stacks["E"]
    pts = lattice(BOX_LO, BOX_HI, 9.0)
    plain = unfaulted(table, traces)
    np.testing.assert_array_equal(model.labels(pts, exact=True), plain.labels(pts, exact=True))
    logged = logged_contacts(table, table["unit"], traces, "A", "B")
    got = throw_diagnostic(model, "F", "A", "B", logged=logged)
    assert got["kind"] == "assumed" and got["pooled"] == ["W", "E"]
    assert "pooled model" in got["note"]
    # the pooled stack is listed once in the model's provenance
    items = [e.item for e in model.provenance_entries()]
    assert len(items) == len(set(items))
    with pytest.raises(ValueError, match="min_holes"):
        fit_block_stacks(table, table["unit"], traces, network, EVENTS, min_holes=0)
    with pytest.raises(ValueError, match="at least one event"):
        fit_block_stacks(table, table["unit"], traces, network, [])


# --------------------------------------------------------------------------
# analytic case 2: tilted layers, a fault dipping 60 degrees, a plug younger than it
# --------------------------------------------------------------------------

DIP = 60.0
SLIP_THROW = 20.0  # vertical component of the slip
SLIP = SLIP_THROW / math.sin(math.radians(DIP))
HEAVE = SLIP * math.cos(math.radians(DIP))
SLOPE = -0.1  # dz/dx of the layers: they dip east, along the slip
FAULT_POINT = np.array([0.0, 0.0, -100.0])
FAULT_NORMAL = np.array([math.sin(math.radians(DIP)), 0.0, math.cos(math.radians(DIP))])
PLUG_CENTRE = np.array([10.0, 0.0, -95.0])
PLUG_RADIUS = 45.0
#: vertical separation of a contact with gradient g along the slip: S (sin d + cos d g)
SEPARATION = SLIP * (math.sin(math.radians(DIP)) + math.cos(math.radians(DIP)) * SLOPE)


def dipping_truth(points, plug=True):
    p = np.asarray(points, dtype=np.float64)
    hanging = (p - FAULT_POINT) @ FAULT_NORMAL > 0.0
    q = p - hanging[:, None] * np.array([HEAVE, 0.0, -SLIP_THROW])  # restored position
    s = q[:, 2] - SLOPE * q[:, 0]
    out = np.array(STACK, dtype=object)[(s < TOPS[0]).astype(int) + (s < TOPS[1])]
    if plug:
        out[np.linalg.norm(p - PLUG_CENTRE, axis=1) < PLUG_RADIUS] = "P"
    return out


@pytest.fixture(scope="module")
def dipping():
    table, traces = drill(dipping_truth, collar_grid(-135.0, 135.0, 30.0))
    fault = import_fault("F", plane=Plane.from_dip(FAULT_POINT, DIP, 90.0), blocks=("FW", "HW"))
    network = FaultNetwork.fit([fault], np.zeros((0, 3)), [])
    events = [*EVENTS, EventSpec("intrusion", ("P",))]
    model, entries = build(table, traces, network, events, post_fault=("P",))
    return model, entries, table, traces, network, events


def test_dipping_fault_labels_and_shared_intrusion(dipping):
    """Tilted layers moved by a known dip slip, and a plug that is not moved. Tolerance:
    the layered units >= 97 % against the truth on a 4 m lattice inside the drilled
    outline (outside the plug), >= 97 % within 30 m of the fault, and without fault
    handling at least 1 point of accuracy worse there (measured 0.985 against 0.970; the
    vertical holes are cut by the dipping fault before they reach the contacts, see
    ``test_dipping_fault_separation_throw_and_heave``). The plug is one event shared by
    both blocks."""
    model, entries, table, traces, _, events = dipping
    assert model.pooled_blocks == []
    plug = model.stacks["FW"].events[1]
    assert plug is model.stacks["HW"].events[1] and plug.units == ("P",)
    assert model.stacks["FW"].events[0] is not model.stacks["HW"].events[0]
    assert model.grid_values("FW", 1) is model.grid_values("HW", 1)  # sampled once
    age = next(e for e in entries if e.item.startswith("Age of the faults"))
    assert "given" in age.source
    assert "one model per fault block: A, B, C" in age.detail
    assert "one model across the blocks: P" in age.detail
    counts = next(e for e in entries if e.item == "Rock-type log by fault block")
    assert "crossing a fault cut at the fault" in counts.detail
    items = [e.item for e in model.provenance_entries()]
    assert len(items) == len(set(items))  # the shared plug is listed once
    assert any(i.endswith("(fault block FW)") for i in items)

    pts = lattice(BOX_LO + [15.0, 15.0, 0.0], BOX_HI - [15.0, 15.0, 0.0], 4.0)
    truth = dipping_truth(pts)
    got = model.decode(model.labels(pts), "lith")
    plain = unfaulted(table, traces, events)
    other = plain.decode(plain.labels(pts), "lith")
    rock = (truth != "P") & (got != "P") & (other != "P")
    near = rock & (np.abs((pts - FAULT_POINT) @ FAULT_NORMAL) < 30.0)
    accuracy, accuracy_near = float((got == truth)[rock].mean()), float((got == truth)[near].mean())
    plain_near = float((other == truth)[near].mean())
    print(
        f"\ntilted layers, fault dipping {DIP:g}: layered units {accuracy:.4f}, within 30 m of "
        f"the fault {accuracy_near:.4f}; without fault handling "
        f"{float((other == truth)[rock].mean()):.4f} and {plain_near:.4f}; plug IoU "
        f"{iou(got == 'P', truth == 'P'):.3f} (no fault handling "
        f"{iou(other == 'P', truth == 'P'):.3f})"
    )
    assert accuracy >= 0.97 and accuracy_near >= 0.97
    assert plain_near <= accuracy_near - 0.01
    # the plug is younger than the fault: the same event with and without fault handling
    assert iou(got == "P", truth == "P") == pytest.approx(iou(other == "P", truth == "P"), abs=0.02)
    hanging = (pts - FAULT_POINT) @ FAULT_NORMAL > 0.0
    np.testing.assert_array_equal(
        model.decode(model.labels(pts, "fault_block"), "fault_block"),
        np.where(hanging, "HW", "FW"),
    )


def test_dipping_fault_separation_throw_and_heave(dipping):
    """The fitted model on the tilted layers: vertical separation within 5 m of the
    analytic 18.845 m, and the dip-slip throw within 5 m of 20 m.

    The tolerance is wide because the fitted model, not the diagnostic, is off here
    (measured 14.5 m): the vertical holes next to the dipping fault are cut by it before
    they reach the contact, the sequence fit gives a run without a contact a constant
    value one cap away from the contact, and the hanging-wall contact is bent within about
    40 m of the fault (mean error -8 m within 20 m, +1.8 m at 20 to 40 m). The diagnostic
    on exact contacts is ``test_throw_diagnostic_on_exact_contacts``."""
    model, _, table, traces, _, _ = dipping
    separation = SEPARATION
    logged = logged_contacts(table, table["unit"], traces, "A", "B")
    got = throw_diagnostic(model, "F", "A", "B", logged=logged)
    print(
        f"\ntilted layers A/B: vertical separation {got['throw']:.2f} m (analytic "
        f"{separation:.2f}), dip-slip throw {got['throw_dip_slip']:.2f} (20), heave "
        f"{got['heave_dip_slip']:.2f} ({HEAVE:.2f}), slip {got['slip_dip_slip']:.2f} ({SLIP:.2f}); "
        f"gradients {got['plane_footwall'][1]:.3f} / {got['plane_hangingwall'][1]:.3f} ({SLOPE})"
    )
    assert got["throw"] == pytest.approx(separation, abs=5.0)
    assert got["throw_dip_slip"] == pytest.approx(SLIP_THROW, abs=5.0)
    assert got["heave_dip_slip"] == pytest.approx(HEAVE, abs=3.0)
    assert got["offset_seen"]
    assert got["fault_dip"] == pytest.approx(DIP) and got["fault_dip_azimuth"] == pytest.approx(
        90.0
    )
    assert got["plane_footwall"][1] == pytest.approx(SLOPE, abs=0.01)
    assert got["plane_hangingwall"][1] == pytest.approx(SLOPE, abs=0.08)
    assert got["kind"] == "measured"
    assert min(got["holes_footwall"], got["holes_hangingwall"]) >= MIN_THROW_HOLES


class ExactLayers:
    """The tilted layers as an exact event: ``phi`` grows downwards, 0 on the contact A/B,
    in the position restored by ``shift`` (what a perfect fit of one block would give)."""

    mode = "sequence"
    units = STACK

    def __init__(self, shift):
        self.shift, self.provenance = np.asarray(shift, dtype=np.float64), []

    def node_values(self, points):
        q = np.asarray(points, dtype=np.float64) - self.shift
        return (TOPS[0] - (q[:, 2] - SLOPE * q[:, 0]))[:, None]

    def labels_from_values(self, values):
        phi = np.asarray(values)[:, 0]
        return ((phi > 0.0).astype(np.int16) + (phi > TOPS[0] - TOPS[1])).astype(np.int16)

    def unit_margins(self, values):
        phi, thick = np.asarray(values, dtype=np.float64)[0], TOPS[0] - TOPS[1]
        return np.stack([-phi, np.minimum(phi, thick - phi), phi - thick])


def test_throw_diagnostic_on_exact_contacts(dipping):
    """The diagnostic alone, on a model whose two blocks carry the exact tilted layers
    (hanging wall = footwall moved by the slip vector): the hand-derived figures, vertical
    separation 18.845 m, throw 20 m, heave 11.547 m, slip 23.094 m, each within 0.05 m, on
    both contacts, with zero spread and plane gradients equal to the truth."""
    assert SEPARATION == pytest.approx(18.845, abs=1e-3)
    assert (SLIP, HEAVE) == (pytest.approx(23.094, abs=1e-3), pytest.approx(11.547, abs=1e-3))
    real, _, _, _, network, _ = dipping
    model = GeologyModel(
        name="exact tilted layers",
        units=UNITS,
        grid=GRID,
        topography=None,
        stacks={
            "FW": LithologyStack("FW", (ExactLayers([0.0, 0.0, 0.0]),)),
            "HW": LithologyStack("HW", (ExactLayers([HEAVE, 0.0, -SLIP_THROW]),)),
        },
        network=network,
        hole_points=real.hole_points,
    )
    pts = lattice(BOX_LO, BOX_HI, 7.0)
    assert (
        model.decode(model.labels(pts), "lith") == dipping_truth(pts, plug=False)
    ).mean() > 0.999
    for (upper, lower), top in zip((("A", "B"), ("B", "C")), TOPS, strict=True):
        got = throw_diagnostic(model, "F", upper, lower)
        print(
            f"\nexact tilted layers {upper}/{lower}: vertical separation {got['throw']:.4f} m "
            f"({SEPARATION:.4f}), dip-slip throw {got['throw_dip_slip']:.4f} (20), heave "
            f"{got['heave_dip_slip']:.4f} ({HEAVE:.4f}), slip {got['slip_dip_slip']:.4f} "
            f"({SLIP:.4f})"
        )
        assert got["throw"] == pytest.approx(SEPARATION, abs=0.05)
        assert got["throw_dip_slip"] == pytest.approx(SLIP_THROW, abs=0.05)
        assert got["heave_dip_slip"] == pytest.approx(HEAVE, abs=0.05)
        assert got["slip_dip_slip"] == pytest.approx(SLIP, abs=0.05)
        assert got["spread"] < 0.01 and max(got["rms_footwall"], got["rms_hangingwall"]) < 0.01
        a, b, c = got["plane_footwall"]
        assert (a, b, c) == (pytest.approx(top, abs=0.01), pytest.approx(SLOPE, abs=1e-4),
                             pytest.approx(0.0, abs=1e-4))  # fmt: skip
        a, b, c = got["plane_hangingwall"]
        assert a == pytest.approx(top - SLIP_THROW - SLOPE * HEAVE, abs=0.01)
        assert b == pytest.approx(SLOPE, abs=1e-4)
    # reverse movement: the same model with the blocks' layers exchanged gives the negative
    swapped = GeologyModel(
        name="exact tilted layers, reverse",
        units=UNITS,
        grid=GRID,
        topography=None,
        stacks={"FW": model.stacks["HW"], "HW": model.stacks["FW"]},
        network=network,
    )
    got = throw_diagnostic(swapped, "F", "A", "B")
    assert got["throw"] == pytest.approx(-SEPARATION, abs=0.05)
    assert "positive side up" in throw_entry(got).detail


def test_two_faults_with_a_termination():
    """Three blocks from two vertical faults, the second ending on the first (it exists
    east of it only), each block with the exact layers lowered by its own amount. The
    meshes are cut at both faults and fill the box (within 1 %; each within 3 % of its
    label volume on a 2 m lattice), and each fault shows the step between its two blocks:
    20 m on F1 (W against S), 15 m on F2 (S against N), within 0.05 m."""
    f1 = import_fault("F1", plane=Plane.from_dip((2.0, 0.0, -100.0), 90.0, 90.0), blocks=("W", "S"))
    f2 = import_fault("F2", plane=Plane.from_dip((0.0, 2.0, -100.0), 90.0, 0.0), blocks=("S", "N"))
    network = FaultNetwork(
        [f1, f2],
        ["N", "S", "W"],
        {(-1, 0): "W", (1, -1): "S", (1, 1): "N"},
        {"F2": ("F1", 1)},
    )
    drop = {"W": 0.0, "S": 20.0, "N": 35.0}
    model = GeologyModel(
        name="three blocks",
        units=UNITS,
        grid=GRID,
        topography=None,
        stacks={b: LithologyStack(b, (ExactLayers([0.0, 0.0, -d]),)) for b, d in drop.items()},
        network=network,
    )
    pts = lattice(BOX_LO, BOX_HI, 2.0)
    block = np.where(pts[:, 0] <= 2.0, "W", np.where(pts[:, 1] <= 2.0, "S", "N"))
    np.testing.assert_array_equal(
        model.decode(model.labels(pts, "fault_block"), "fault_block"), block
    )
    labels = model.decode(model.labels(pts), "lith")
    lowered = np.select([block == "W", block == "S"], [drop["W"], drop["S"]], drop["N"])
    s = pts[:, 2] - SLOPE * pts[:, 0] + lowered
    truth = np.array(STACK, dtype=object)[(s < TOPS[0]).astype(int) + (s < TOPS[1])]
    assert (labels == truth).mean() > 0.999
    meshes = model.meshes("lith")
    box = float(np.prod(BOX_HI - BOX_LO))
    total = 0.0
    for code in STACK:
        assert meshes[code].is_closed, code
        assert meshes[code].volume == pytest.approx(float((truth == code).mean()) * box, rel=0.03)
        total += meshes[code].volume
    print(f"\nthree blocks: mesh volumes sum to {total / box:.5f} of the box")
    assert total == pytest.approx(box, rel=0.01)
    results, entries = throw_diagnostics(model)
    found = {(r["fault"], r["upper"]): r["throw"] for r in results}
    assert found[("F1", "A")] == pytest.approx(20.0, abs=0.05)
    assert found[("F1", "B")] == pytest.approx(20.0, abs=0.05)
    assert found[("F2", "A")] == pytest.approx(15.0, abs=0.05)
    assert found[("F2", "B")] == pytest.approx(15.0, abs=0.05)
    assert {e.kind for e in entries} == {"assumed"}  # no logs given: the holes are not counted


def test_dipping_fault_meshes_fill_the_box(dipping):
    """Four closed meshes, cut at the dipping fault; their volumes sum to the box within
    1 % (no gaps, no overlaps), and each is within 3 % of the volume of the same label of
    the model counted on a 2 m lattice (the layered units also within 4 % of the truth)."""
    model, _, _, _, _, _ = dipping
    meshes = model.meshes("lith")
    assert list(meshes) == [*STACK, "P"]
    pts = lattice(BOX_LO, BOX_HI, 2.0)
    truth = dipping_truth(pts)
    labels = model.decode(model.labels(pts), "lith")
    box = float(np.prod(BOX_HI - BOX_LO))
    total = 0.0
    for code, mesh in meshes.items():
        assert mesh.is_closed, code
        assert trimesh.Trimesh(mesh.vertices, mesh.faces, process=False).is_watertight, code
        assert mesh.volume == pytest.approx(float((labels == code).mean()) * box, rel=0.03), code
        if code != "P":
            assert mesh.volume == pytest.approx(float((truth == code).mean()) * box, rel=0.04)
        total += mesh.volume
    print(f"\ntilted layers: mesh volumes sum to {total / box:.5f} of the box")
    assert total == pytest.approx(box, rel=0.01)


def test_post_fault_rules(dipping):
    _, _, table, traces, network, events = dipping
    units = table["unit"]
    with pytest.raises(ValueError, match="no event models"):
        fit_block_stacks(table, units, traces, network, events, post_fault=("Q",))
    with pytest.raises(ValueError, match="mixes units"):
        fit_block_stacks(table, units, traces, network, events, post_fault=("A",))
    with pytest.raises(ValueError, match="oldest to the youngest"):
        fit_block_stacks(table, units, traces, network, events[::-1], post_fault=("P",))
    with pytest.raises(ValueError, match="units has"):
        fit_block_stacks(table, units.iloc[:-1], traces, network, events)
    # every unit younger than the fault: one shared stack content, no block is pooled
    stacks, pooled, entries = fit_block_stacks(
        table, units, traces, network, events, post_fault=("A", "B", "C", "P")
    )
    assert pooled == []
    assert all(a is b for a, b in zip(stacks["FW"].events, stacks["HW"].events, strict=True))
    assert not any(e.item == "Rock-type log by fault block" for e in entries)


def test_save_load_round_trip_with_shared_events(dipping, tmp_path):
    """The folder of a faulted model: the fault layer files, the shared plug stored once,
    and identical labels (grid and exact) for every layer after loading."""
    model, _, table, traces, _, _ = dipping
    results, _ = throw_diagnostics(model, table=table, units=table["unit"], traces=traces)
    model.diagnostics["throw"] = results
    folder = model.save(tmp_path / "dipping.geology")
    manifest = json.loads((folder / "geology.json").read_text())
    assert manifest["blocks"] == ["FW", "HW"] and manifest["pooled_blocks"] == []
    assert manifest["layers"]["fault_block"] == ["FW", "HW"]
    assert manifest["faults"] == [{"name": "F", "file": "faults/F.json", "mesh": "faults/F.obj"}]
    assert manifest["fault_network"] == {
        "cells": [{"signs": [-1], "block": "FW"}, {"signs": [1], "block": "HW"}],
        "terminations": {},
    }
    fw, hw = manifest["stacks"]["FW"], manifest["stacks"]["HW"]
    assert fw[0]["values"] == "grid/values_FW_0.npy" and hw[0]["values"] == "grid/values_HW_0.npy"
    assert fw[1]["values"] == hw[1]["values"] == "grid/values_FW_1.npy"
    assert fw[1]["fields"] == hw[1]["fields"] == ["fields/FW_1_P.npz"]
    assert not (folder / "grid/values_HW_1.npy").exists()
    blocks = np.load(folder / "grid/fault_block.npy")
    assert blocks.dtype == np.int16 and blocks.shape == GRID.shape
    np.testing.assert_array_equal(blocks, model.fault_block_grid())
    display = read_obj(folder / "faults/F.obj")
    assert np.abs((display.vertices - FAULT_POINT) @ FAULT_NORMAL).max() < 1e-2
    stored = json.loads((folder / "diagnostics.json").read_text())
    assert [r["upper"] for r in stored["throw"]] == ["A", "B"]
    assert set(stored["fault_network"]) == {"-1", "+1"}

    loaded = GeologyModel.load(folder)
    assert loaded.blocks == ("FW", "HW") and loaded.network is not None
    assert loaded.network.cells == model.network.cells
    assert loaded.network.faults[0].blocks == ("FW", "HW")
    assert loaded.stacks["FW"].events[1] is loaded.stacks["HW"].events[1]
    assert loaded.grid_values("FW", 1) is loaded.grid_values("HW", 1)
    pts = np.random.default_rng(3).uniform(BOX_LO - 10.0, BOX_HI + 10.0, size=(20000, 3))
    for layer in LAYERS:
        np.testing.assert_array_equal(loaded.labels(pts, layer), model.labels(pts, layer))
    np.testing.assert_array_equal(
        loaded.labels(pts[:5000], exact=True), model.labels(pts[:5000], exact=True)
    )
    again = loaded.meshes("lith")
    for code, mesh in model.meshes("lith").items():
        assert again[code].volume == pytest.approx(mesh.volume, rel=1e-6)
    # saving the loaded model over its own folder (memory-mapped values) works
    loaded.save(folder, overwrite=True)
    once_more = GeologyModel.load(folder)
    np.testing.assert_array_equal(once_more.labels(pts), model.labels(pts))


def test_cross_validation_builder(flat):
    """The per-block builder in hole-out cross-validation (5 folds): it predicts the
    left-out holes better than one stack for all blocks, by more than 1 % of the logged
    metres (measured 0.998 against 0.981)."""
    _, _, table, traces, network = flat
    faulted = cross_validate(
        table, table["unit"], traces, block_stack_builder(network, EVENTS), folds=5, step=2.0
    )
    pooled = cross_validate(table, table["unit"], traces, stack_builder(EVENTS), folds=5, step=2.0)
    print(
        f"\nflat layers, hole-out agreement: per fault block {faulted.agreement:.4f}, "
        f"one stack {pooled.agreement:.4f}"
    )
    assert "one stack per fault block (E, W)" in faulted.builder
    assert faulted.agreement >= 0.99
    assert faulted.agreement > pooled.agreement + 0.01


# --------------------------------------------------------------------------
# the synthetic layered deposit (geology_contracts.md 13, 14)
# --------------------------------------------------------------------------

LAYERED = ("SST", "SLT", "AND")
INTRUSIONS = ("DIO", "GRD")  # oldest first; both younger than F1
GROUP = LAYERED + INTRUSIONS
LAYERED_EVENTS = [
    EventSpec("sequence", LAYERED),
    EventSpec("intrusion", ("DIO",)),
    EventSpec("intrusion", ("GRD",)),
]
#: geology_contracts.md section 14
TARGET_LAYER, TARGET_STOCK, TARGET_DYKE = 0.85, 0.80, 0.50
TARGET_CONTACT_RMS = 3.0
TRUE_THROW, THROW_TOLERANCE = 30.0, 6.0
TARGET_BLOCKS = {"imported": 0.99, "inferred": 0.97}
#: the unfaulted model of P9.T10 on the same lattice (measured again in the fixture)
T10 = {"SST": 0.941, "SLT": 0.883, "AND": 0.953, "DIO": 0.856, "GRD": 1.000}
T10_CONTACT_RMS = 3.33


def contact_rms(model, truth, lo, hi):
    """SST/SLT contact against the analytic contact inside the drilled hull, on vertical
    lines 5 m apart where the model and the truth both have exactly one such contact (the
    measure of P9.T7): ``(columns, RMS distance to the plane, largest distance)``."""
    xy = lattice(lo[:2], hi[:2], 5.0)
    z = np.arange(lo[2], hi[2] + 1e-9, 0.5)
    cols = np.column_stack([np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))])
    got = model.labels(cols).reshape(len(xy), len(z))
    true = truth.labels(cols, "bedrock").reshape(len(xy), len(z))
    sst, slt = model.categories("lith").index("SST"), model.categories("lith").index("SLT")
    hull = Delaunay(model.hole_points)
    separation = []
    for i, (x, y) in enumerate(xy):
        k = np.flatnonzero((got[i, 1:] == sst) & (got[i, :-1] == slt))
        j = np.flatnonzero((true[i, 1:] == "SST") & (true[i, :-1] == "SLT"))
        if len(k) != 1 or len(j) != 1:
            continue
        candidates = [float(truth.contact_z(x, y, "SST/SLT", b)) for b in ("FB1", "FB2")]
        z_true = min(candidates, key=lambda c: abs(c - (z[j[0]] + 0.25)))
        if abs(z_true - (z[j[0]] + 0.25)) > 0.3:
            continue  # SST lies on SLT across the fault here, not across the bedding contact
        if hull.find_simplex(np.array([x, y, z_true])) >= 0:
            separation.append(z[k[0]] + 0.25 - z_true)
    distance = np.array(separation) * truth.bedding_normal[2]
    return len(distance), float(np.sqrt(np.mean(distance**2))), float(np.abs(distance).max())


@pytest.fixture(scope="module")
def layered_data(layered_small, layered_small_dir, layered_small_db, layered_small_traces):
    """The logs, the two versions of F1, and the unfaulted model of P9.T10 as the baseline."""
    units = UnitDictionary.load(layered_small_dir / "units.json")
    assert units.codes(*MODELLED_KINDS) == GROUP
    table = layered_small_db.intervals["geology"]
    assay = layered_small_db.intervals["assay"]
    traces = layered_small_traces
    mapped = units.map_codes(table["lith"])
    rock = mapped.where(mapped.isin(GROUP), np.nan)  # cover and weathering codes: unknown
    zones = table[(units.kinds_of(mapped) == "fault-zone").to_numpy()]
    blocks = block_codes(assay["wf"], r"^(FB\d)-")
    xyz, block, hole = labelled_points(assay, blocks, traces)
    a, b = block == "FB1", block == "FB2"
    picks = fault_picks(assay, blocks, traces, fault_zone=zones)
    cfg = layered_small.config
    faults = {
        "imported": import_fault(
            "F1", plane=Plane.from_dip(cfg.fault_point, cfg.fault_dip, cfg.fault_dip_azimuth)
        ),
        "inferred": infer_fault(
            "F1", picks, xyz[a], xyz[b], blocks=("FB1", "FB2"), holes_a=hole[a], holes_b=hole[b]
        ),
    }
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    common = dict(
        units=units,
        grid=GridSpec(lo, (5.0, 5.0, 5.0), tuple(int(n) for n in np.round((hi - lo) / 5.0) + 1)),
        topography=DTM(read_obj(layered_small_dir / "topo.obj")),
        hole_points=logged_hole_points(table, traces, step=5.0, known=rock.notna()),
    )
    events = [fit_stack(table, rock, traces, LAYERED_EVENTS[:1]).events[0]]
    events += [
        fit_stack(table, mask_younger(rock, INTRUSIONS, u), traces, [spec]).events[0]
        for u, spec in zip(INTRUSIONS, LAYERED_EVENTS[1:], strict=True)
    ]
    baseline = GeologyModel(
        name="no fault handling", stacks={"ALL": LithologyStack("ALL", tuple(events))}, **common
    )
    baseline.sample_grid()
    pts = lattice(lo, hi, 2.5)
    pts = pts[Delaunay(common["hole_points"]).find_simplex(pts) >= 0]
    return {
        "table": table, "rock": rock, "traces": traces, "faults": faults, "common": common,
        "labelled": (xyz, block), "box": (lo, hi), "baseline": baseline, "lattice": pts,
        "truth": layered_small.truth, "truth_lattice": layered_small.truth.labels(pts, "bedrock"),
    }  # fmt: skip


@pytest.fixture(scope="module", params=["imported", "inferred"])
def layered(request, layered_data):
    data = layered_data
    t0 = time.perf_counter()
    network = FaultNetwork.fit([data["faults"][request.param]], *data["labelled"])
    stacks, pooled, entries = fit_block_stacks(
        data["table"], data["rock"], data["traces"], network, LAYERED_EVENTS,
        post_fault=INTRUSIONS,
    )  # fmt: skip
    t1 = time.perf_counter()
    model = GeologyModel(
        name=f"layered-small, F1 {request.param}", stacks=stacks, network=network, **data["common"]
    )
    model.pooled_blocks = pooled
    model.sample_grid()
    t2 = time.perf_counter()
    print(f"\nlayered-small, F1 {request.param}: fit {t1 - t0:.2f} s, grid {t2 - t1:.2f} s")
    return request.param, model, entries, data


def scores(model, pts, truth):
    got = model.decode(model.labels(pts), "lith")
    ground = (truth != "AIR") & (got != "AIR")
    got, truth = got[ground], truth[ground]
    assert (got != None).all()  # noqa: E711
    return (
        {u: iou(got == u, truth == u) for u in GROUP},
        {u: float((got == u).sum() / (truth == u).sum()) for u in GROUP},
        float((got == truth).mean()),
    )


def test_layered_iou_volumes_and_blocks(layered):
    """IoU per unit against the exact ``bedrock`` truth on the 2.5 m lattice inside the
    drilled hull, below the topography (the lattice of P9.T7 / P9.T10), grid labels.

    Section 14, asserted because met: layered units >= 0.85, DIO >= 0.80, GRD >= 0.50,
    fault blocks >= 0.99 (imported plane) / >= 0.97 (inferred), here on lattice points.
    Against the unfaulted model on the same points: every layered unit better, SLT (the
    unit both of whose contacts step at the fault) by at least 0.03; the intrusions are
    younger than the fault and fitted once, so their figures do not change.
    Volumes inside the hull against the truth: layered units and the dyke within 5 %; the
    stock within 12 % (a regression bound: it was 0.911 before this task and is not
    touched by it)."""
    how, model, _, data = layered
    pts, truth = data["lattice"], data["truth_lattice"]
    before_iou, before_volume, before_accuracy = scores(data["baseline"], pts, truth)
    after_iou, after_volume, accuracy = scores(model, pts, truth)
    blocks = model.decode(model.labels(pts, "fault_block"), "fault_block")
    block_accuracy = float((blocks == data["truth"].labels(pts, "fault_block")).mean())
    print(
        f"\nlayered-small, F1 {how}, {len(pts)} lattice points in the hull\n"
        f"  no fault handling: accuracy {before_accuracy:.4f}; IoU "
        + ", ".join(f"{u} {v:.3f}" for u, v in before_iou.items())
        + "; volume model / truth "
        + ", ".join(f"{u} {v:.3f}" for u, v in before_volume.items())
        + f"\n  per fault block:    accuracy {accuracy:.4f}; IoU "
        + ", ".join(f"{u} {v:.3f}" for u, v in after_iou.items())
        + "; volume model / truth "
        + ", ".join(f"{u} {v:.3f}" for u, v in after_volume.items())
        + f"\n  fault blocks right at {block_accuracy:.5f} of the points"
    )
    for u in GROUP:  # the baseline is the model of P9.T10
        assert before_iou[u] == pytest.approx(T10[u], abs=0.002), u
    for u in LAYERED:
        assert after_iou[u] >= TARGET_LAYER, (u, after_iou[u])
        assert after_iou[u] > before_iou[u], u
        assert after_volume[u] == pytest.approx(1.0, abs=0.05), u
    assert after_iou["SLT"] >= before_iou["SLT"] + 0.03
    assert accuracy > before_accuracy + 0.005
    assert after_iou["DIO"] >= TARGET_STOCK and after_iou["GRD"] >= TARGET_DYKE
    assert after_iou["DIO"] == pytest.approx(before_iou["DIO"], abs=1e-9)
    assert after_iou["GRD"] == pytest.approx(before_iou["GRD"], abs=1e-9)
    assert after_volume["GRD"] == pytest.approx(1.0, abs=0.05)
    assert after_volume["DIO"] == pytest.approx(1.0, abs=0.12)
    assert block_accuracy >= TARGET_BLOCKS[how]
    assert model.pooled_blocks == []


def test_layered_contact_rms(layered):
    """Section 14: SST/SLT contact RMS distance to the analytic contact inside the drilled
    hull <= 3 m. Met with the fault; the unfaulted model misses it (3.33 m)."""
    how, model, _, data = layered
    lo, hi = data["box"]
    n0, before, before_max = contact_rms(data["baseline"], data["truth"], lo, hi)
    n1, after, after_max = contact_rms(model, data["truth"], lo, hi)
    print(
        f"\nlayered-small, F1 {how}: SST/SLT contact RMS {after:.2f} m (largest {after_max:.1f}, "
        f"{n1} columns); no fault handling {before:.2f} m (largest {before_max:.1f}, {n0} columns)"
    )
    assert before == pytest.approx(T10_CONTACT_RMS, abs=0.02) and before > TARGET_CONTACT_RMS
    assert after <= TARGET_CONTACT_RMS
    assert n1 > 1000


def test_layered_throw(layered):
    """Section 14: throw (vertical separation of SST/SLT) 30 +/- 6 m, measured, with the
    holes on each side. The fault dips 65 degrees and the beds do not dip along the slip,
    so the dip-slip throw is the same figure, and the heave is throw / tan 65 (truth
    13.99 m). SLT/AND is reported; it is a wavy contact fitted by planes and only held to
    30 +/- 8 m (a regression bound, not a contract target)."""
    how, model, _, data = layered
    t0 = time.perf_counter()
    results, entries = throw_diagnostics(
        model, table=data["table"], units=data["rock"], traces=data["traces"]
    )
    seconds = time.perf_counter() - t0
    assert [(r["upper"], r["lower"]) for r in results] == [("SST", "SLT"), ("SLT", "AND")]
    for r in results:
        print(
            f"\nlayered-small, F1 {how}, {r['upper']}/{r['lower']}: vertical separation "
            f"{r['throw']:.2f} m (truth 30), spread {r['spread']:.2f} m over {r['n_trace']} trace "
            f"points; planes from {r['n_footwall']} / {r['n_hangingwall']} points, rms "
            f"{r['rms_footwall']:.2f} / {r['rms_hangingwall']:.2f} m; holes "
            f"{r['holes_footwall']} / {r['holes_hangingwall']}; dip-slip throw "
            f"{r['throw_dip_slip']:.2f}, heave {r['heave_dip_slip']:.2f} (13.99); {r['kind']} "
            f"({seconds:.2f} s for both)"
        )
    top, base = results
    assert top["throw"] == pytest.approx(TRUE_THROW, abs=THROW_TOLERANCE)
    assert top["throw_dip_slip"] == pytest.approx(TRUE_THROW, abs=THROW_TOLERANCE)
    assert top["heave_dip_slip"] == pytest.approx(13.989, abs=3.0)
    assert top["kind"] == "measured" and top["offset_seen"] and top["at"] == "fault trace"
    assert top["blocks"] == ["FB1", "FB2"]
    assert min(top["holes_footwall"], top["holes_hangingwall"]) >= MIN_THROW_HOLES
    assert base["throw"] == pytest.approx(TRUE_THROW, abs=8.0)
    assert [e.kind for e in entries] == ["measured", "measured"]
    assert entries[0].item == "Throw of fault F1 on the contact SST / SLT"
    assert "footwall" in entries[0].source and "vertical separation" in entries[0].detail


def test_layered_logs_and_grid_labels(layered):
    """The logged unit along the holes (points every 2 m): exact >= 99.5 %, grid >= 98 %
    (the bounds of P9.T10). Grid labels against exact labels >= 99.5 % of random points
    (section 14): the block of a point is always taken exactly from the fault surfaces,
    and each block's node values are interpolated on their own, so the fault does not
    blur the grid labels."""
    how, model, _, data = layered
    table, rock, traces = data["table"], data["rock"], data["traces"]
    pts, row = interval_points(table, rock.notna().to_numpy(), traces)
    logged = rock.to_numpy(dtype=object)[row]
    exact = float((model.decode(model.labels(pts, exact=True), "lith") == logged).mean())
    fast = float((model.decode(model.labels(pts), "lith") == logged).mean())
    lo, hi = data["box"]
    some = np.random.default_rng(11).uniform(lo, hi, size=(20000, 3))
    agreement = float((model.labels(some, exact=True) == model.labels(some)).mean())
    plain = data["baseline"]
    before = float((plain.decode(plain.labels(pts, exact=True), "lith") == logged).mean())
    print(
        f"\nlayered-small, F1 {how}: logged unit reproduced at {exact:.3%} (exact) and {fast:.3%} "
        f"(5 m grid) of {len(pts)} points along the holes (no fault handling {before:.3%}); grid "
        f"labels equal exact labels at {agreement:.3%} of 20000 points"
    )
    assert exact >= 0.995 and fast >= 0.98
    assert agreement >= 0.995


def test_layered_meshes(layered):
    """A closed mesh per unit, the layered units cut at F1. Their volumes sum to the
    volume of the grid nodes below the topography within 1 %."""
    how, model, _, _ = layered
    t0 = time.perf_counter()
    meshes = model.meshes("lith")
    seconds = time.perf_counter() - t0
    assert set(meshes) == set(GROUP)
    assert all(mesh.is_closed for mesh in meshes.values())
    nodes = model.grid.slab_points(0, model.grid.shape[0]).reshape(-1, 3)
    weight = np.ones(model.grid.shape)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            weight[tuple(index)] *= 0.5
    below = model.topography.height_above(nodes) <= 0.0
    ground = float(weight.reshape(-1)[below].sum()) * 125.0
    total = sum(mesh.volume for mesh in meshes.values())
    print(
        f"\nlayered-small, F1 {how}: mesh volumes sum to {total / ground:.4f} of the ground "
        f"({seconds:.2f} s)"
    )
    assert total == pytest.approx(ground, rel=0.01)


def test_layered_provenance(layered):
    how, model, entries, _ = layered
    record = [*model.provenance_entries(), *entries]
    items = {(e.kind, e.item) for e in record}
    assert ("assumed", "Age of the faults relative to the rock units") in items
    assert ("measured", "Rock-type log by fault block") in items
    assert not any("pooled" in e.item for e in record)
    assert any(e.item.endswith("(fault block FB1)") for e in record)
    assert any(e.item.endswith("(fault block FB2)") for e in record)
    fault_lines = [e for e in record if e.task == "P9.T8"]
    assert fault_lines and (fault_lines[0].kind == "assumed") == (how == "inferred")
    assert ("missing", "Fault surfaces") in items if how == "inferred" else True
    names = [e.item for e in model.provenance_entries()]
    assert len(names) == len(set(names))


def test_layered_save_load_round_trip(layered, tmp_path):
    """Saved and loaded: identical labels for every layer (grid and exact), the shared
    intrusions stored once, the fault layer in the manifest."""
    how, model, _, data = layered
    t0 = time.perf_counter()
    folder = model.save(tmp_path / "layered.geology")
    t1 = time.perf_counter()
    loaded = GeologyModel.load(folder)
    t2 = time.perf_counter()
    print(f"\nlayered-small, F1 {how}: save {t1 - t0:.2f} s, load {t2 - t1:.2f} s")
    manifest = json.loads((folder / "geology.json").read_text())
    assert [f["name"] for f in manifest["faults"]] == ["F1"]
    assert manifest["fault_network"]["cells"] == [
        {"signs": [-1], "block": "FB1"}, {"signs": [1], "block": "FB2"},
    ]  # fmt: skip
    assert [e["values"] for e in manifest["stacks"]["FB2"]] == [
        "grid/values_FB2_0.npy", "grid/values_FB1_1.npy", "grid/values_FB1_2.npy",
    ]  # fmt: skip
    assert (folder / "faults/F1.json").is_file() and (folder / "faults/F1.obj").is_file()
    assert loaded.network.faults[0].source == model.network.faults[0].source
    lo, hi = data["box"]
    pts = np.random.default_rng(12).uniform(lo - 20.0, hi + 20.0, size=(20000, 3))
    for layer in LAYERS:
        np.testing.assert_array_equal(loaded.labels(pts, layer), model.labels(pts, layer))
    np.testing.assert_array_equal(
        loaded.labels(pts[:5000], exact=True), model.labels(pts[:5000], exact=True)
    )
    assert [e.item for e in loaded.provenance.entries] == [
        e.item for e in model.provenance_entries()
    ]
