"""Initiation timing (bakumine.blast.timing), research_blast.md part 1, sections 5 to 7.

Firing times of a 3 x 4 pattern are worked by hand; the maximum instantaneous charge is
checked against a brute-force scan of every window.
"""

from __future__ import annotations

import math
from dataclasses import replace

import numpy as np
import pytest

from bakumine.blast import design as bd
from bakumine.blast import timing as bt
from bakumine.blast.model import ANFO, BlastDesign, Rock


def pattern(kind: str = "square", width: float = 14.0, depth: float = 9.0) -> BlastDesign:
    """3 rows x 4 holes (square) or 4 / 5 / 4 (staggered); B = 3 m, S = 3.5 m."""
    return bd.layout(
        [(0.0, 0.0), (width, 0.0), (width, depth), (0.0, depth)],
        top=110.0,
        floor_z=100.0,
        burden=3.0,
        spacing=3.5,
        pattern=kind,
        face_azimuth=0.0,
        diameter=0.1,
        explosive=ANFO,
        stemming=2.1,
        subdrill=0.9,
        rock=Rock(),
    )


def delays(design: BlastDesign) -> dict[str, float]:
    return {h.id: h.delay_ms for h in design.holes}


Q = math.pi / 4.0 * 0.1**2 * 850.0 * 8.8  # kg per hole of `pattern`, by hand = 58.74778


def test_row_by_row_by_hand():
    base = pattern()
    timed = bt.row_by_row(base, 42.0)
    assert delays(timed) == {f"{r}{i}": 42.0 * n for n, r in enumerate("ABC") for i in range(1, 5)}
    assert not base.timed and timed.timed  # a new design; the input is untouched
    assert timed.notes["timing"] == {
        "scheme": "row_by_row",
        "inter_row_ms": 42.0,
        "inter_hole_ms": 0.0,
        "initiation_hole": "A1",
    }
    assert "timing" not in base.notes
    assert bt.duration_ms(timed) == 84.0
    assert bt.firing_order(timed) == [f"{r}{i}" for r in "ABC" for i in range(1, 5)]
    with_hole = bt.row_by_row(base, 42.0, 17.0)
    assert delays(with_hole) == delays(bt.echelon(base, 17.0, 42.0, start="left"))


def test_echelon_by_hand():
    """t = 17 k + 42 r:  A: 0 17 34 51 | B: 42 59 76 93 | C: 84 101 118 135."""
    left = delays(bt.echelon(pattern(), 17.0, 42.0))
    assert [left[f"A{i}"] for i in range(1, 5)] == [0.0, 17.0, 34.0, 51.0]
    assert [left[f"B{i}"] for i in range(1, 5)] == [42.0, 59.0, 76.0, 93.0]
    assert [left[f"C{i}"] for i in range(1, 5)] == [84.0, 101.0, 118.0, 135.0]
    right = bt.echelon(pattern(), 17.0, 42.0, start="right")
    assert [delays(right)[f"B{i}"] for i in range(1, 5)] == [93.0, 76.0, 59.0, 42.0]
    assert bt.firing_order(right)[:3] == ["A4", "A3", "A2"]
    assert bt.duration_ms(right) == 135.0
    with pytest.raises(ValueError, match="start"):
        bt.echelon(pattern(), 17.0, 42.0, start="middle")


def test_v_cut_by_hand():
    """Front row at s = 1.75, 5.25, 8.75, 12.25, middle 7.0: A2 and A3 are equally near,
    the left one (A2) initiates. Spacings from A2: 1 0 1 2 -> A: 25 0 25 50,
    B: 75 50 75 100, C: 125 100 125 150 (Th = 25, Tr = 50)."""
    timed = bt.v_cut(pattern(), 25.0, 50.0)
    got = delays(timed)
    assert [got[f"A{i}"] for i in range(1, 5)] == [25.0, 0.0, 25.0, 50.0]
    assert [got[f"B{i}"] for i in range(1, 5)] == [75.0, 50.0, 75.0, 100.0]
    assert [got[f"C{i}"] for i in range(1, 5)] == [125.0, 100.0, 125.0, 150.0]
    assert timed.notes["timing"]["initiation_hole"] == "A2"
    assert bt.firing_order(timed)[0] == "A2"


def test_v_cut_staggered_and_odd_front_row():
    """Five holes in front (17.5 m wide): A3 at the centre, symmetric V. Row B of the
    staggered pattern sits half a spacing off: floor(0.5) = 0, floor(1.5) = 1, floor(2.5) = 2."""
    timed = bt.v_cut(pattern("staggered", width=17.5), 10.0, 100.0)
    got = delays(timed)
    assert [got[f"A{i}"] for i in range(1, 6)] == [20.0, 10.0, 0.0, 10.0, 20.0]
    assert [got[f"B{i}"] for i in range(1, 7)] == [120.0, 110.0, 100.0, 100.0, 110.0, 120.0]
    assert [got[f"C{i}"] for i in range(1, 6)] == [220.0, 210.0, 200.0, 210.0, 220.0]


def test_hole_by_hole_unique_or_refused():
    timed = bt.hole_by_hole(pattern(), 17.0, 42.0)
    times = sorted(delays(timed).values())
    assert times[0] == 0.0 and len(set(times)) == 12
    assert delays(timed) == delays(bt.echelon(pattern(), 17.0, 42.0))
    with pytest.raises(ValueError, match="same"):
        bt.hole_by_hole(pattern(), 25.0, 50.0)  # A3 (2 * 25) and B1 (50) coincide


def test_bad_delays_and_untimed():
    base = pattern()
    for call in (
        lambda: bt.row_by_row(base, 0.0),
        lambda: bt.echelon(base, -1.0, 42.0),
        lambda: bt.v_cut(base, 17.0, math.nan),
        lambda: bt.hole_by_hole(base, 0.0, 42.0),
    ):
        with pytest.raises(ValueError):
            call()
    for fn in (bt.firing_order, bt.duration_ms, bt.max_instantaneous_charge, bt.burden_relief):
        with pytest.raises(ValueError, match="not timed"):
            fn(base)


SCHEMES = [
    lambda d: bt.row_by_row(d, 42.0),
    lambda d: bt.row_by_row(d, 42.0, 9.0),
    lambda d: bt.echelon(d, 17.0, 42.0),
    lambda d: bt.echelon(d, 30.0, 5.0, start="right"),
    lambda d: bt.v_cut(d, 25.0, 50.0),
    lambda d: bt.v_cut(d, 40.0, 3.0),
    lambda d: bt.hole_by_hole(d, 17.0, 42.0),
]


@pytest.mark.parametrize("scheme", SCHEMES)
@pytest.mark.parametrize("kind", ["square", "staggered"])
def test_rows_never_fire_out_of_order(scheme, kind):
    """Property: of the holes of the row in front that shield a hole (straight ahead in a
    square pattern, half a spacing to either side in a staggered one) the one nearer the
    initiation hole always fires first, and all of them when Tr > Th or the pattern is
    square; no row starts before the row in front; the first hole fires at 0."""
    base = bd.layout(
        [(0.0, 30.0), (45.0, 30.0), (38.0, 0.0), (6.0, 4.0)],  # irregular quadrilateral
        110.0, 100.0, 3.0, 3.5, kind, 0.0, 0.1,
    )  # fmt: skip
    timed = scheme(base)
    assert min(h.delay_ms for h in timed.holes) == 0.0
    assert all(math.isfinite(h.delay_ms) for h in timed.holes)
    start = next(g for g in timed.holes if g.id == timed.notes["timing"]["initiation_hole"])
    th = timed.notes["timing"]["inter_hole_ms"]
    tr = timed.notes["timing"]["inter_row_ms"]
    assert start.row == 0
    checked = 0
    for h in timed.holes:
        if h.row == 0:
            continue
        ahead = [
            g for g in timed.holes
            if g.row == h.row - 1 and abs(g.collar[0] - h.collar[0]) <= 3.5 * 0.51
        ]  # fmt: skip
        if not ahead:
            continue
        checked += 1
        nearest = min(ahead, key=lambda g: abs(g.collar[0] - start.collar[0]))
        assert nearest.delay_ms < h.delay_ms, h.id
        if tr > th or kind == "square":
            assert all(g.delay_ms < h.delay_ms for g in ahead), h.id
    assert checked > 20
    first = [min(g.delay_ms for g in timed.holes if g.row == r) for r in range(3)]
    assert first == sorted(first) and first[0] < first[1] < first[2]


# ---------------------------------------------------------------------------------- MIC


def brute_mic(design: BlastDesign, window: float) -> float:
    """Oracle: start a window at every firing time and add up, hole by hole."""
    best = 0.0
    for a in design.holes:
        total = 0.0
        for b in design.holes:
            if a.delay_ms <= b.delay_ms < a.delay_ms + window:
                total += b.charge_mass
        best = max(best, total)
    return best


def test_mic_by_hand():
    """Rows of 4 holes of 58.748 kg. Rows 42 ms apart: 4 holes. Rows exactly 8 ms apart are
    separate delays (half-open window): still 4. Rows 7.9 ms apart: two rows = 8 holes;
    3.9 ms apart: all three rows = 12 holes."""
    base = pattern()
    kg, ids = bt.max_instantaneous_charge(bt.row_by_row(base, 42.0))
    assert kg == pytest.approx(4 * Q) and ids == ["A1", "A2", "A3", "A4"]
    assert Q == pytest.approx(58.74778, abs=1e-5)
    assert bt.max_instantaneous_charge(bt.row_by_row(base, 8.0))[0] == pytest.approx(4 * Q)
    kg, ids = bt.max_instantaneous_charge(bt.row_by_row(base, 7.9))
    assert kg == pytest.approx(8 * Q) and len(ids) == 8
    assert bt.max_instantaneous_charge(bt.row_by_row(base, 3.9))[0] == pytest.approx(12 * Q)
    # hole by hole, 17 / 42 ms: closest pairs are 8 ms apart (B1 at 42, A4 at 51; ...) -> 1 hole
    hbh = bt.hole_by_hole(base, 17.0, 42.0)
    assert bt.max_instantaneous_charge(hbh)[0] == pytest.approx(Q)
    assert bt.max_instantaneous_charge(hbh, window_ms=10.0)[0] == pytest.approx(2 * Q)
    assert bt.max_instantaneous_charge(bt.row_by_row(base, 42.0), 0.0)[0] == pytest.approx(4 * Q)


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("window", [8.0, 25.0, 0.0])
def test_mic_against_brute_force(seed, window):
    """Random firing times (many ties and near-ties) and unequal charges."""
    rng = np.random.default_rng(seed)
    base = pattern("staggered", width=35.0, depth=15.0)
    holes = [
        replace(
            h,
            delay_ms=float(rng.integers(0, 40)) * rng.choice([1.0, 4.0, 8.0]),
            charge_top=float(rng.uniform(1.0, 6.0)),
        )
        for h in base.holes
    ]
    timed = replace(base, holes=holes)
    kg, ids = bt.max_instantaneous_charge(timed, window)
    assert kg == pytest.approx(brute_mic(timed, window) if window else max(
        sum(b.charge_mass for b in holes if b.delay_ms == a.delay_ms) for a in holes
    ))  # fmt: skip
    by_id = {h.id: h for h in holes}
    assert sum(by_id[i].charge_mass for i in ids) == pytest.approx(kg)
    span = [by_id[i].delay_ms for i in ids]
    assert span == sorted(span) and (max(span) - min(span) < window or window == 0.0)


# ------------------------------------------------------------------------ burden relief


def test_burden_relief_by_hand():
    """B = 3 m. 42 ms between rows = 14 ms/m (inside 6.5 to 16.5); 9 ms = 3 ms/m (violent);
    90 ms = 30 ms/m (beyond the table). The echelon and the V give the same row-to-row value."""
    base = pattern()
    r = bt.burden_relief(bt.row_by_row(base, 42.0))
    assert r.ms_per_m == pytest.approx(14.0) and r.per_row == pytest.approx((14.0, 14.0))
    assert r.guide_ms_per_m == (6.5, 16.5)
    assert len(r.flags) == 1 and "within" in r.flags[0]
    assert bt.burden_relief(bt.echelon(base, 17.0, 42.0)).ms_per_m == pytest.approx(14.0)
    assert bt.burden_relief(bt.v_cut(base, 17.0, 42.0)).ms_per_m == pytest.approx(14.0)
    fast = bt.burden_relief(bt.row_by_row(base, 9.0))
    assert fast.ms_per_m == pytest.approx(3.0) and "< 6.5" in fast.flags[0]
    slow = bt.burden_relief(bt.row_by_row(base, 90.0))
    assert slow.ms_per_m == pytest.approx(30.0) and "> 16.5" in slow.flags[0]
    one_row = bt.row_by_row(pattern(depth=3.0), 42.0)
    single = bt.burden_relief(one_row)
    assert math.isnan(single.ms_per_m) and "single row" in single.flags[0]
