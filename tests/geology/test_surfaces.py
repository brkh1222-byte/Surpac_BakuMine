"""Cover / weathering base surfaces as depth below the topography (P9.T9, research.md 5.7.4).

Independent checks: the analytic truth of the synthetic layered deposit (``LayeredTruth``),
planes that the interpolant must reproduce, picks and counts recomputed here with pandas
from the raw CSV files, and distances by brute force.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geology import surfaces as sf
from bakumine.core.geology.picks import PICK_COLUMNS, unit_picks
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geometry.dtm import DTM, dtm_from_grid
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.surface_interp import Surface2D, SurfaceFitError

RECTANGLE = (500200.0, 5300200.0, 500600.0, 5300600.0)
SPACING = 5.0
OXIDE_TARGET_RMS = 2.5  # geology_contracts.md 14, fixed before any tuning
UNKNOWN_RAW = {"CL", "FLT", "FB"}  # core loss and fault-zone spellings of the synthetic log
ERRORS: dict[tuple[str, str], tuple[float, float]] = {}


# ---------------------------------------------------------------------------- hand cases


def plane_topography(a: float = 100.0, bx: float = 0.02, by: float = -0.01) -> DTM:
    x = np.arange(0.0, 201.0, 10.0)
    gx, gy = np.meshgrid(x, x, indexing="ij")
    return dtm_from_grid(a + bx * gx + by * gy, (0.0, 0.0), (10.0, 10.0))


def topo_z(x, y, a: float = 100.0, bx: float = 0.02, by: float = -0.01):
    return a + bx * np.asarray(x) + by * np.asarray(y)


def hand_picks(rows: list[dict], surface: str = "base:A") -> pd.DataFrame:
    """A picks table of vertical holes: ``kind, x, y`` and ``lo`` / ``hi`` (depth below topo)."""
    out = []
    for n, row in enumerate(rows):
        kind = row["kind"]
        record = dict.fromkeys(PICK_COLUMNS, np.nan)
        record.update(
            hole_id=row.get("hole_id", f"H{n + 1}"),
            surface=row.get("surface", surface),
            kind=kind,
            source=row.get("source", "log"),
        )
        if kind != "none":
            x, y = row["x"], row["y"]
            lo, hi = row.get("lo", np.nan), row.get("hi", np.nan)
            if kind == "exact":
                hi = lo
            record.update(x=x, y=y, topo_z=float(topo_z(x, y)), below_lo=lo, below_hi=hi)
            record["depth"] = lo if kind != "upper" else hi
            record["z"] = record["topo_z"] - record["depth"]
            if kind == "between":
                record.update(
                    x2=x, y2=y, topo_z2=record["topo_z"], depth_max=hi, z2=topo_z(x, y) - hi
                )
        frame_row = {c: record[c] for c in PICK_COLUMNS}
        if "conflict" in row:
            frame_row["conflict"] = row["conflict"]
        out.append(frame_row)
    frame = pd.DataFrame(out)
    if "conflict" in frame.columns:
        frame["conflict"] = frame["conflict"].fillna(False).astype(bool)
    return frame


def lattice(function, step: float = 40.0, lo: float = 20.0, hi: float = 180.0) -> list[dict]:
    xs = np.arange(lo, hi + 1e-9, step)
    return [
        {"kind": "exact", "x": float(x), "y": float(y), "lo": float(function(x, y))}
        for x in xs
        for y in xs
    ]


HAND_RECTANGLE = (0.0, 0.0, 200.0, 200.0)


def entries(surface, kind: str, start: str = ""):
    return [e for e in surface.provenance if e.kind == kind and e.item.startswith(start)]


def test_constant_depth_under_a_planar_topography_is_reproduced_exactly():
    topo = plane_topography()
    surface = sf.depth_surface(
        hand_picks(lattice(lambda x, y: 12.5)), topo, HAND_RECTANGLE, 10.0, name="base:A"
    )
    assert surface.origin == (0.0, 0.0) and surface.shape == (21, 21)
    np.testing.assert_allclose(surface.depth, 12.5, rtol=0, atol=1e-9)
    gx, gy = np.meshgrid(np.arange(1.0, 200.0, 7.0), np.arange(1.0, 200.0, 7.0), indexing="ij")
    np.testing.assert_allclose(surface.dtm.z_at(gx, gy), topo_z(gx, gy) - 12.5, atol=1e-9)
    assert surface.n_nested == 0 and surface.diagnostics["nodes_clamped"] == 0


@pytest.mark.parametrize("options", [{"drift": "linear"}, {"method": "rbf"}])
def test_plane_depth_field_is_reproduced(options):
    def depth(x, y):
        return 8.0 + 0.05 * x - 0.02 * y

    surface = sf.depth_surface(
        hand_picks(lattice(depth)),
        plane_topography(),
        HAND_RECTANGLE,
        5.0,
        name="base:A",
        **options,
    )
    gx, gy = surface.nodes()
    np.testing.assert_allclose(surface.depth, depth(gx, gy), rtol=0, atol=1e-7)
    np.testing.assert_allclose(surface.topography_z, topo_z(gx, gy), atol=1e-9)
    fx, fy = np.meshgrid(np.arange(0.5, 200.0, 3.3), np.arange(0.5, 200.0, 3.3), indexing="ij")
    np.testing.assert_allclose(surface.dtm.z_at(fx, fy), topo_z(fx, fy) - depth(fx, fy), atol=1e-7)


def test_grid_origin_snaps_down_and_covers_the_rectangle():
    surface = sf.depth_surface(
        hand_picks(lattice(lambda x, y: 5.0)), plane_topography(), (12.0, 33.0, 151.0, 160.0),
        10.0, name="base:A",
    )  # fmt: skip
    assert surface.origin == (10.0, 30.0)
    gx, gy = surface.nodes()
    assert gx.max() >= 151.0 and gy.max() >= 160.0 and gx.max() < 161.0 and gy.max() < 170.0
    with pytest.raises(ValueError, match="xmin < xmax"):
        sf.depth_surface(hand_picks(lattice(lambda x, y: 5.0)), plane_topography(),
                         (10.0, 0.0, 10.0, 5.0), 10.0, name="base:A")  # fmt: skip


def test_bounds_are_honoured_at_their_holes_and_inactive_bounds_change_nothing():
    base = lattice(lambda x, y: 10.0)
    bounds = [
        {"kind": "lower", "x": 40.0, "y": 40.0, "lo": 16.0},  # active: pushes down
        {"kind": "upper", "x": 120.0, "y": 120.0, "hi": 6.0},  # active: pulls up
        {"kind": "between", "x": 80.0, "y": 160.0, "lo": 12.0, "hi": 13.0},  # active (lower)
        {"kind": "lower", "x": 160.0, "y": 40.0, "lo": 3.0},  # inactive
        {"kind": "upper", "x": 40.0, "y": 120.0, "hi": 25.0},  # inactive
        {"kind": "between", "x": 160.0, "y": 160.0, "lo": 2.0, "hi": 30.0},  # inactive
    ]
    topo = plane_topography()
    free = sf.depth_surface(hand_picks(base), topo, HAND_RECTANGLE, 10.0, name="base:A")
    active = sf.depth_surface(
        hand_picks(base + bounds[:3]), topo, HAND_RECTANGLE, 10.0, name="base:A"
    )
    full = sf.depth_surface(hand_picks(base + bounds), topo, HAND_RECTANGLE, 10.0, name="base:A")

    def at(surface, x, y):  # the holes are on grid nodes
        return surface.depth[int(round(x / 10.0)), int(round(y / 10.0))]

    assert at(free, 40.0, 40.0) == pytest.approx(10.0, abs=1e-9)
    assert at(full, 40.0, 40.0) == pytest.approx(16.0, abs=1e-7)
    assert at(full, 120.0, 120.0) == pytest.approx(6.0, abs=1e-7)
    assert 12.0 - 1e-7 <= at(full, 80.0, 160.0) <= 13.0 + 1e-7
    assert at(full, 160.0, 40.0) >= 3.0 and at(full, 40.0, 120.0) <= 25.0
    assert 2.0 <= at(full, 160.0, 160.0) <= 30.0
    # the same through the surface itself: z = topography - depth
    assert full.dtm.z_at(40.0, 40.0) == pytest.approx(topo_z(40.0, 40.0) - 16.0, abs=1e-7)
    assert full.dtm.z_at(120.0, 120.0) == pytest.approx(topo_z(120.0, 120.0) - 6.0, abs=1e-7)
    # exact picks stay exact
    for row in base:
        assert at(full, row["x"], row["y"]) == pytest.approx(10.0, abs=1e-7)
    np.testing.assert_allclose(full.depth, active.depth, rtol=0, atol=1e-9)
    assert not np.allclose(full.depth, free.depth, atol=1e-3)
    assert full.diagnostics["used"] == {"exact": 25, "lower": 2, "upper": 2, "between": 2}
    assert full.diagnostics["active_lower"] == 2 and full.diagnostics["active_upper"] == 1
    assert full.fit.max_violation <= 1e-7
    assert [e.count for e in entries(full, "measured", "Bounds")] == [6]
    assert [e.count for e in entries(full, "measured", "Exact")] == [25]


def test_holes_without_information_change_nothing_hand_case():
    rows = lattice(lambda x, y: 10.0 + 0.03 * x)
    rows += [{"kind": "lower", "x": 80.0, "y": 80.0, "lo": 17.0}]
    nothing = [{"kind": "none"} for _ in range(7)]
    topo = plane_topography()
    with_none = sf.depth_surface(
        hand_picks(rows + nothing), topo, HAND_RECTANGLE, 10.0, name="base:A"
    )
    without = sf.depth_surface(hand_picks(rows), topo, HAND_RECTANGLE, 10.0, name="base:A")
    np.testing.assert_array_equal(with_none.depth, without.depth)
    np.testing.assert_array_equal(with_none.dtm.mesh.vertices, without.dtm.mesh.vertices)
    assert with_none.diagnostics["holes_without_information"] == 7
    assert [e.count for e in entries(with_none, "missing")] == [7]
    assert entries(without, "missing") == []
    # a bare table carries no policy: said so, never silently missing
    assert len(entries(with_none, "assumed", "Holes without the units")) == 1


def test_rows_of_other_surfaces_are_ignored():
    rows = lattice(lambda x, y: 10.0)
    other = [dict(r, surface="base:B", lo=40.0) for r in rows]
    topo = plane_topography()
    surface = sf.depth_surface(hand_picks(rows + other), topo, HAND_RECTANGLE, 10.0, name="base:A")
    np.testing.assert_allclose(surface.depth, 10.0, atol=1e-9)
    assert len(surface.picks) == len(rows)
    with pytest.raises(ValueError, match="no row of surface"):
        sf.depth_surface(hand_picks(rows), topo, HAND_RECTANGLE, 10.0, name="base:C")


def test_conflicts_are_left_out_by_default_counted_and_kept_on_request():
    rows = lattice(lambda x, y: 10.0)
    for row in rows:
        row["conflict"] = False
    bad = [
        {"kind": "exact", "x": 80.0, "y": 80.0, "lo": 25.0, "conflict": True, "hole_id": "BAD1"},
        {"kind": "lower", "x": 120.0, "y": 80.0, "lo": 22.0, "conflict": True, "hole_id": "BAD2"},
        {"kind": "none", "conflict": False, "hole_id": "EMPTY"},
    ]
    topo = plane_topography()
    clean = sf.depth_surface(hand_picks(rows), topo, HAND_RECTANGLE, 10.0, name="base:A")
    default = sf.depth_surface(hand_picks(rows + bad), topo, HAND_RECTANGLE, 10.0, name="base:A")
    kept = sf.depth_surface(
        hand_picks(rows + bad), topo, HAND_RECTANGLE, 10.0, name="base:A", keep_conflicts=True
    )
    np.testing.assert_array_equal(default.depth, clean.depth)
    assert default.diagnostics["conflicts"] == 2 and not default.diagnostics["conflicts_kept"]
    assert default.diagnostics["holes_conflict"] == ["BAD1", "BAD2"]
    assert default.diagnostics["used"]["exact"] == 25 and default.diagnostics["used"]["lower"] == 0
    assert not default.picks.loc[default.picks["hole_id"].isin(["BAD1", "BAD2"]), "used"].any()
    (entry,) = entries(default, "assumed", "Conflicting picks")
    assert entry.count == 2 and "left out" in entry.detail and "BAD1" in entry.detail
    assert kept.depth[8, 8] == pytest.approx(25.0, abs=1e-7) and kept.depth[12, 8] >= 22.0 - 1e-7
    assert kept.diagnostics["used"]["exact"] == 26 and kept.diagnostics["used"]["lower"] == 1
    (entry,) = entries(kept, "assumed", "Conflicting picks")
    assert entry.count == 2 and "kept in" in entry.detail
    assert entries(clean, "assumed", "Conflicting picks") == []


def test_picks_outside_the_topography_are_dropped_and_reported():
    rows = lattice(lambda x, y: 10.0)
    table = hand_picks(
        rows
        + [
            {"kind": "exact", "x": 100.0, "y": 100.0, "lo": 40.0, "hole_id": "OUT1"},
            {"kind": "upper", "x": 60.0, "y": 100.0, "hi": 2.0, "hole_id": "OUT2"},
            {"kind": "between", "x": 60.0, "y": 140.0, "lo": 1.0, "hi": 3.0, "hole_id": "OUT3"},
        ]
    )
    table.loc[table["hole_id"] == "OUT1", ["below_lo", "below_hi", "topo_z"]] = np.nan
    table.loc[table["hole_id"] == "OUT2", ["below_hi", "topo_z"]] = np.nan
    table.loc[table["hole_id"] == "OUT3", ["below_hi", "topo_z2"]] = np.nan
    topo = plane_topography()
    surface = sf.depth_surface(table, topo, HAND_RECTANGLE, 10.0, name="base:A")
    np.testing.assert_allclose(surface.depth, 10.0, atol=1e-9)
    assert surface.diagnostics["holes_outside_topography"] == ["OUT1", "OUT2", "OUT3"]
    (entry,) = entries(surface, "assumed", "Picks of base:A outside")
    assert entry.count == 3 and "OUT2" in entry.detail


def test_no_exact_pick_is_an_error_not_a_guess():
    rows = [{"kind": "lower", "x": 50.0, "y": 50.0, "lo": 5.0}, {"kind": "none"}]
    with pytest.raises(SurfaceFitError, match="no exact pick"):
        sf.depth_surface(hand_picks(rows), plane_topography(), HAND_RECTANGLE, 10.0, name="base:A")


def test_surface_never_rises_above_the_topography():
    def depth(x, y):
        return 0.1 * (x - 45.0)  # negative west of x = 45: extrapolated there

    rows = lattice(depth, lo=60.0)
    topo = plane_topography()
    surface = sf.depth_surface(
        hand_picks(rows), topo, HAND_RECTANGLE, 10.0, name="base:A", drift="linear"
    )
    fitted = surface.fit.grid(surface.origin, surface.spacing, surface.shape)
    expected = int((fitted < 0.0).sum())
    assert expected == 5 * 21  # the columns x = 0 .. 40
    assert surface.diagnostics["nodes_clamped"] == expected
    assert np.nanmin(surface.depth) == 0.0
    np.testing.assert_allclose(surface.depth, np.maximum(fitted, 0.0), atol=0)
    fx, fy = np.meshgrid(np.arange(0.5, 200.0, 2.1), np.arange(0.5, 200.0, 2.1), indexing="ij")
    assert (surface.dtm.z_at(fx, fy) <= topo.z_at(fx, fy) + 1e-9).all()
    (entry,) = entries(surface, "assumed", "base:A kept at or below")
    assert entry.count == expected


def crossing_pair():
    """Two fitted planes that cross at x = 100: A (listed first) is deeper east of it."""
    topo = plane_topography()
    a = sf.depth_surface(hand_picks(lattice(lambda x, y: 5.0 + 0.1 * x), "base:A"), topo,
                         HAND_RECTANGLE, 5.0, name="base:A", drift="linear")  # fmt: skip
    b = sf.depth_surface(hand_picks(lattice(lambda x, y: 15.0 + 0.0 * x), "base:B"), topo,
                         HAND_RECTANGLE, 5.0, name="base:B", drift="linear")  # fmt: skip
    return topo, a, b


def test_nested_surfaces_never_cross_hand_case_built_to_cross():
    topo, a, b = crossing_pair()
    crossed = a.depth > b.depth
    assert crossed.sum() == 20 * 41  # x = 105 .. 200: the case does cross before correction
    na, nb = sf.nest_surfaces([a, b])
    assert na.n_nested == 0 and nb.n_nested == int(crossed.sum())
    np.testing.assert_array_equal(na.depth, a.depth)
    np.testing.assert_array_equal(nb.depth, np.maximum(a.depth, b.depth))
    np.testing.assert_array_equal(b.depth[crossed] < a.depth[crossed], True)  # input untouched
    assert b.n_nested == 0
    fx, fy = np.meshgrid(np.arange(0.0, 200.01, 0.9), np.arange(0.0, 200.01, 0.9), indexing="ij")
    za, zb, zt = na.dtm.z_at(fx, fy), nb.dtm.z_at(fx, fy), topo.z_at(fx, fy)
    assert (zt >= za - 1e-9).all() and (za >= zb - 1e-9).all()
    assert (b.dtm.z_at(fx, fy) > a.dtm.z_at(fx, fy) + 1e-6).any()  # before: B above A
    west = fx < 99.0
    np.testing.assert_allclose(zb[west], zt[west] - 15.0, atol=1e-7)  # unchanged where no crossing
    (entry,) = entries(nb, "assumed", "Nesting of")
    assert entry.count == int(crossed.sum()) and "base:A" in entry.source
    assert nb.diagnostics["nesting_largest_move"] == pytest.approx(10.0, abs=1e-6)
    (entry,) = entries(na, "assumed", "Nesting of")
    assert entry.count == 0
    # nesting again changes nothing and does not duplicate the entry
    again = sf.nest_surfaces([na, nb])
    np.testing.assert_array_equal(again[1].depth, nb.depth)
    assert again[1].n_nested == 0 and len(entries(again[1], "assumed", "Nesting of")) == 1


def test_nest_under_a_dtm_equals_nesting_the_pair_and_grids_must_match():
    topo, a, b = crossing_pair()
    nb = sf.nest_surfaces([a, b])[1]
    under = sf.nest_under(b, a.dtm, above_name="base:A")
    np.testing.assert_allclose(under.depth, nb.depth, rtol=0, atol=1e-9)
    assert under.n_nested == nb.n_nested
    coarse = sf.depth_surface(hand_picks(lattice(lambda x, y: 15.0), "base:B"), topo,
                              HAND_RECTANGLE, 10.0, name="base:B")  # fmt: skip
    with pytest.raises(ValueError, match="different grids"):
        sf.nest_surfaces([a, coarse])
    assert sf.nest_surfaces([]) == []


def test_extrapolated_share_against_brute_force():
    rows = lattice(lambda x, y: 10.0, step=20.0, lo=20.0, hi=80.0)  # picks in one corner only
    surface = sf.depth_surface(
        hand_picks(rows), plane_topography(), HAND_RECTANGLE, 10.0, name="base:A"
    )
    pts = np.array([[r["x"], r["y"]] for r in rows])
    gx, gy = surface.nodes()
    nodes = np.column_stack([gx.ravel(), gy.ravel()])
    nearest = np.sqrt(((nodes[:, None, :] - pts[None, :, :]) ** 2).sum(axis=2)).min(axis=1)
    expected = int((nearest > 2.0 * 20.0).sum())
    assert surface.diagnostics["pick_spacing"] == pytest.approx(20.0)
    assert surface.diagnostics["nodes_extrapolated"] == expected > 0
    (entry,) = entries(surface, "assumed", "Extrapolated part")
    assert entry.count == expected and entry.value == pytest.approx(
        expected / nodes.shape[0], abs=1e-6
    )


def test_save_writes_the_three_files_of_the_contract(tmp_path):
    surface = sf.depth_surface(
        hand_picks(lattice(lambda x, y: 10.0 + 0.02 * x)), plane_topography(), HAND_RECTANGLE,
        10.0, name="base:A",
    )  # fmt: skip
    paths = surface.save(tmp_path / "surfaces")
    assert {p.name for p in paths.values()} == {"base_A.obj", "base_A.json", "base_A_picks.csv"}
    data = json.loads(paths["json"].read_text(encoding="utf-8"))
    again = Surface2D.from_dict(data)
    xy = np.array([[33.0, 71.0], [150.0, 12.0]])
    np.testing.assert_allclose(again(xy), surface.fit(xy), atol=1e-12)
    assert data["diagnostics"]["used"]["exact"] == 25 and data["name"] == "base:A"
    assert {e["kind"] for e in data["provenance"]} <= {"measured", "assumed", "missing"}
    table = pd.read_csv(paths["picks"])
    assert list(table.columns[: len(PICK_COLUMNS)]) == list(PICK_COLUMNS) and len(table) == 25
    from bakumine.core.io.obj import read_obj

    mesh = read_obj(paths["obj"])
    np.testing.assert_allclose(DTM(mesh).z_at(55.0, 45.0), surface.dtm.z_at(55.0, 45.0), atol=1e-9)


# ---------------------------------------------------------------------------- fault zones


def test_fault_zone_across_the_contact_is_a_bracket_not_an_exact_pick(layered_units_path):
    units = UnitDictionary.load(layered_units_path)
    xs = np.array([50.0, 100.0, 150.0])
    collar = pd.DataFrame({"hole_id": ["A", "B", "C"], "x": xs, "y": 100.0,
                           "z": topo_z(xs, 100.0), "max_depth": 40.0})  # fmt: skip
    survey = pd.DataFrame({"hole_id": ["A", "B", "C"], "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    traces = desurvey(collar, survey)
    log = pd.DataFrame(
        [("A", 0.0, 10.0, "OZ"), ("A", 10.0, 12.0, "FLT"), ("A", 12.0, 40.0, "SST"),
         ("B", 0.0, 8.0, "OZ"), ("B", 8.0, 40.0, "SST"),
         ("C", 0.0, 9.0, "OZ"), ("C", 9.0, 11.0, "CL"), ("C", 11.0, 40.0, "FB")],
        columns=["hole_id", "depth_from", "depth_to", "lith"],
    )  # fmt: skip
    topo = plane_topography()
    source = sf.log_source(log, "lith", units, ["OX"])
    assert set(source.unknown_codes) == set(units.codes("missing", "fault-zone"))
    picks = sf.surface_picks([source], traces, topo, surface="base:OXIDE")
    table = picks.table.set_index("hole_id")
    assert table.loc["A", "kind"] == "between"
    assert (table.loc["A", "below_lo"], table.loc["A", "below_hi"]) == pytest.approx((10.0, 12.0))
    assert table.loc["B", "kind"] == "exact" and table.loc["B", "below_lo"] == pytest.approx(8.0)
    assert table.loc["C", "kind"] == "lower" and table.loc["C", "below_lo"] == pytest.approx(9.0)
    # what the hand-over warned about: with the fault zone read as a logged rock, A is "exact"
    naive = unit_picks(log, units.map_codes(log["lith"]), ["OX"], traces, topo,
                       surface="base:OXIDE", unknown_codes=units.codes("missing"))  # fmt: skip
    naive = naive.table
    assert naive.set_index("hole_id").loc["A", "kind"] == "exact"
    with pytest.raises(ValueError, match="cannot be the target"):
        sf.log_source(log, "lith", units, ["FLT"])
    with pytest.raises(ValueError, match="not a unit"):
        sf.log_source(log, "lith", units, ["NOPE"])


# ---------------------------------------------------------------------------- layered-small


@pytest.fixture(scope="module")
def topography(layered_small_dir) -> DTM:
    return DTM(Mesh(np.load(layered_small_dir / "topo_vertices.npy"),
                    np.load(layered_small_dir / "topo_faces.npy")))  # fmt: skip


@pytest.fixture(scope="module")
def sources(layered_small_db, layered_units_path) -> dict[str, sf.PickSource]:
    units = UnitDictionary.load(layered_units_path)
    geology, assay = layered_small_db.intervals["geology"], layered_small_db.intervals["assay"]
    return {
        "cover_log": sf.log_source(geology, "lith", units, ["Q4"]),
        "oxide_log": sf.log_source(geology, "lith", units, ["OX"]),
        "oxide_threshold": sf.threshold_source(assay, "s_ppm", below=1000.0),
        "transition_threshold": sf.threshold_source(assay, "s_ppm", below=3000.0),
    }


@pytest.fixture(scope="module")
def built(sources, layered_small_traces, topography) -> dict[tuple[str, str], sf.DepthSurface]:
    """Every surface and source combination on the same grid."""
    cases = {
        ("base:Q4", "log"): ["cover_log"],
        ("base:OXIDE", "log"): ["oxide_log"],
        ("base:OXIDE", "threshold"): ["oxide_threshold"],
        ("base:OXIDE", "combined"): ["oxide_log", "oxide_threshold"],
        ("base:TRANSITION", "threshold"): ["transition_threshold"],
    }
    out = {}
    for (name, label), keys in cases.items():
        picks = sf.surface_picks(
            [sources[k] for k in keys], layered_small_traces, topography, surface=name
        )
        out[name, label] = sf.depth_surface(picks, topography, RECTANGLE, SPACING, name=name)
    return out


def truth_depth(truth, name: str, x, y):
    return {
        "base:Q4": truth.cover_depth,
        "base:OXIDE": truth.oxide_depth,
        "base:TRANSITION": truth.transition_depth,
    }[name](x, y)


def drilled_grid(step: float = 5.0):
    """Points inside the collar grid (25 m inside the box), not on the surface's own nodes."""
    x = np.arange(RECTANGLE[0] + 26.0, RECTANGLE[2] - 25.0, step)
    y = np.arange(RECTANGLE[1] + 26.0, RECTANGLE[3] - 25.0, step)
    return np.meshgrid(x, y, indexing="ij")


def errors(surface, truth, name: str) -> tuple[float, float]:
    gx, gy = drilled_grid()
    depth = truth.topo_z(gx, gy) - surface.dtm.z_at(gx, gy)
    e = depth - truth_depth(truth, name, gx, gy)
    assert np.isfinite(e).all()
    return float(np.sqrt(np.mean(e**2))), float(np.abs(e).max())


# regression floors (RMS, max) in metres, set from the first run with ~50 % margin;
# the section 14 target (oxide base <= 2.5 m RMS) is asserted separately
FLOORS = {
    ("base:Q4", "log"): (0.50, 1.7),
    ("base:OXIDE", "log"): (1.05, 3.7),
    ("base:OXIDE", "threshold"): (0.48, 1.4),
    ("base:OXIDE", "combined"): (0.40, 1.4),
    ("base:TRANSITION", "threshold"): (0.40, 1.6),
}


@pytest.mark.parametrize("case", list(FLOORS))
def test_surface_against_the_analytic_truth(case, built, layered_small):
    name, label = case
    rms, worst = errors(built[case], layered_small.truth, name)
    ERRORS[case] = (rms, worst)
    print(f"\n{name} ({label}): RMS {rms:.3f} m, max {worst:.3f} m")
    assert rms <= FLOORS[case][0] and worst <= FLOORS[case][1]
    if name == "base:OXIDE":
        assert rms <= OXIDE_TARGET_RMS  # geology_contracts.md 14: met by every source


def test_sources_available_for_each_surface_are_what_the_deposit_logs(built):
    """Cover has no assay rule and the transition zone is not logged: one source each."""
    assert built["base:Q4", "log"].diagnostics["used"]["exact"] > 30
    assert built["base:TRANSITION", "threshold"].diagnostics["used"]["exact"] == 64
    combined = built["base:OXIDE", "combined"].diagnostics["used_by_source"]
    assert set(combined) == {"log", "threshold"}
    assert combined["log"]["exact"] + combined["threshold"]["exact"] == 64


def test_exact_picks_and_bounds_are_honoured_on_the_deposit(built):
    seen_bound = False
    for surface in built.values():
        p = surface.picks[surface.picks["used"]]
        exact = p[p["kind"] == "exact"]
        np.testing.assert_allclose(
            surface.fit(exact[["x", "y"]].to_numpy()), exact["below_lo"].to_numpy(), atol=1e-7
        )
        for row in p[p["kind"].isin(["lower", "between"])].itertuples():
            assert surface.fit(np.array([[row.x, row.y]]))[0] >= row.below_lo - 1e-7
            seen_bound = True
        for row in p[p["kind"] == "between"].itertuples():
            assert surface.fit(np.array([[row.x2, row.y2]]))[0] <= row.below_hi + 1e-7
        for row in p[p["kind"] == "upper"].itertuples():
            assert surface.fit(np.array([[row.x, row.y]]))[0] <= row.below_hi + 1e-7
    assert seen_bound  # the cover log has a bracket (core loss or a fault zone at the contact)


def test_holes_without_information_change_nothing_on_the_deposit(built, topography):
    surface = built["base:OXIDE", "log"]
    table = surface.picks[list(PICK_COLUMNS) + ["conflict"]]
    n_none = int((table["kind"] == "none").sum())
    assert n_none > 20
    removed = sf.depth_surface(table[table["kind"] != "none"], topography, RECTANGLE, SPACING,
                               name="base:OXIDE")  # fmt: skip
    np.testing.assert_array_equal(removed.depth, surface.depth)
    # read as zero depth instead, the surface would be pulled up to the topography
    assert float(np.nanmean(surface.depth)) > 15.0


def test_nested_surfaces_never_cross_on_a_fine_grid(built, topography):
    chain = [built["base:Q4", "log"], built["base:OXIDE", "combined"],
             built["base:TRANSITION", "threshold"]]  # fmt: skip
    nested = sf.nest_surfaces(chain)
    for before, after in zip(chain, nested, strict=True):
        moved = int((after.depth != before.depth).sum())
        assert after.n_nested == moved
        assert [e.count for e in entries(after, "assumed", "Nesting of")] == [moved]
    x = np.arange(RECTANGLE[0], RECTANGLE[2] + 1e-9, 1.3)
    y = np.arange(RECTANGLE[1], RECTANGLE[3] + 1e-9, 1.3)
    fx, fy = np.meshgrid(x, y, indexing="ij")
    z = [topography.z_at(fx, fy)] + [s.dtm.z_at(fx, fy) for s in nested]
    assert all(np.isfinite(layer).all() for layer in z)
    for upper, lower in zip(z[:-1], z[1:], strict=True):
        assert (upper >= lower - 1e-9).all()
    for upper, lower in zip(nested[:-1], nested[1:], strict=True):
        assert (lower.depth >= upper.depth).all() and (upper.depth >= 0.0).all()


# ---------------------------------------------------------------------------- independent counts


def raw_log_picks(folder, target: set[str]) -> pd.DataFrame:
    """Pick kind and depths per hole from ``geology.csv`` with pandas only."""
    log = pd.read_csv(folder / "geology.csv", dtype={"lith": str}, keep_default_na=False)
    log["code"] = log["lith"].str.strip().str.upper()
    rows = []
    for hole_id, group in log.sort_values(["hole_id", "depth_from"]).groupby("hole_id"):
        codes = group["code"].tolist()
        tops, bottoms = group["depth_from"].tolist(), group["depth_to"].tolist()
        assert np.allclose(tops[1:], bottoms[:-1])  # contiguous logging
        inside = [i for i, c in enumerate(codes) if c in target]
        if not inside:
            rows.append((hole_id, "none", np.nan, np.nan))
            continue
        k = inside[-1]
        j = k + 1
        while j < len(codes) and codes[j] in UNKNOWN_RAW:
            j += 1
        if j == len(codes):
            rows.append((hole_id, "lower", bottoms[k], np.nan))
        elif j == k + 1:
            rows.append((hole_id, "exact", bottoms[k], np.nan))
        else:
            rows.append((hole_id, "between", bottoms[k], tops[j]))
    return pd.DataFrame(rows, columns=["hole_id", "kind", "depth", "depth_max"]).set_index(
        "hole_id"
    )


def raw_threshold_picks(folder, limit: float) -> pd.DataFrame:
    assay = pd.read_csv(folder / "assay.csv")
    rows = []
    for hole_id, group in assay.sort_values(["hole_id", "depth_from"]).groupby("hole_id"):
        below = np.flatnonzero(group["s_ppm"].to_numpy() < limit)
        if below.size == 0:
            rows.append((hole_id, "none", np.nan))
        elif below[-1] == len(group) - 1:
            rows.append((hole_id, "lower", group["depth_to"].iloc[-1]))
        else:
            rows.append((hole_id, "exact", group["depth_to"].iloc[below[-1]]))
    return pd.DataFrame(rows, columns=["hole_id", "kind", "depth"]).set_index("hole_id")


def count_of(surface, kind: str, start: str) -> int:
    found = entries(surface, kind, start)
    return sum(e.count for e in found) if found else 0


def test_provenance_counts_equal_counts_made_independently(built, layered_small_dir):
    raw_cover = raw_log_picks(layered_small_dir, {"SOIL", "SOL"})
    raw_oxide = raw_log_picks(layered_small_dir, {"OZ"})
    raw_thr = raw_threshold_picks(layered_small_dir, 1000.0)
    assert len(raw_oxide) == 64

    for raw, surface in (
        (raw_cover, built["base:Q4", "log"]),
        (raw_oxide, built["base:OXIDE", "log"]),
    ):
        name = surface.name
        kinds = raw["kind"].value_counts()
        mine = surface.picks.set_index("hole_id")
        assert (mine["kind"].reindex(raw.index) == raw["kind"]).all()
        np.testing.assert_allclose(mine["depth"].reindex(raw.index), raw["depth"], atol=1e-6)
        assert count_of(surface, "measured", f"Exact picks of {name} (log)") == kinds.get(
            "exact", 0
        )
        n_bounds = int(kinds.get("lower", 0) + kinds.get("between", 0))
        assert count_of(surface, "measured", f"Bounds of {name} (log)") == n_bounds
        assert count_of(surface, "missing", f"{name} not logged") == kinds.get("none", 0) > 0
        (policy,) = entries(surface, "assumed", "Holes without the units")
        assert policy.count == kinds.get("none", 0) and "absent = unknown" in policy.source
        assert len(entries(surface, "assumed", "Variogram of")) == 1
        assert "default" in entries(surface, "assumed", "Variogram of")[0].source
        assert len(entries(surface, "assumed", "Extrapolated part")) == 1

    threshold = built["base:OXIDE", "threshold"]
    kinds = raw_thr["kind"].value_counts()
    assert (
        count_of(threshold, "measured", "Exact picks of base:OXIDE (threshold)") == kinds["exact"]
    )
    assert count_of(threshold, "missing", "") == kinds.get("none", 0) == 0
    (rule,) = entries(threshold, "measured", "Threshold rule")
    assay = pd.read_csv(layered_small_dir / "assay.csv")
    assert rule.count == int((assay["s_ppm"] < 1000.0).sum()) and rule.value == 1000.0
    assert "s_ppm < 1000" in rule.source

    # combined: the log's exact pick wins, the rule fills the holes the log says nothing about
    combined = built["base:OXIDE", "combined"]
    log_exact = set(raw_oxide.index[raw_oxide["kind"] == "exact"])
    thr_exact = set(raw_thr.index[raw_thr["kind"] == "exact"])
    assert count_of(combined, "measured", "Exact picks of base:OXIDE (log)") == len(log_exact)
    assert count_of(combined, "measured", "Exact picks of base:OXIDE (threshold)") == len(
        thr_exact - log_exact
    )
    assert len(entries(combined, "measured", "Threshold rule")) == 1
    assert len(entries(combined, "assumed", "Conflict tolerance")) == 1


def test_conflicts_between_log_and_threshold_counted_independently(
    sources, layered_small_traces, topography, layered_small_dir, built
):
    """With a tolerance far below the sample length, log and sulphur rule disagree in holes
    that have both; those holes are left out, counted and listed."""
    tolerance = 0.25
    raw_log = raw_log_picks(layered_small_dir, {"OZ"})
    raw_thr = raw_threshold_picks(layered_small_dir, 1000.0)
    expected = set()
    for hole_id, row in raw_log.iterrows():
        other = raw_thr.loc[hole_id]
        if row["kind"] == "none" or other["kind"] != "exact":
            continue
        d = other["depth"]
        if row["kind"] == "exact" and abs(row["depth"] - d) > tolerance:
            expected.add(hole_id)
        if row["kind"] in ("lower", "between") and row["depth"] > d + tolerance:
            expected.add(hole_id)
        if row["kind"] == "between" and row["depth_max"] < d - tolerance:
            expected.add(hole_id)
    assert expected  # the case exists on this deposit

    picks = sf.surface_picks(
        [sources["oxide_log"], sources["oxide_threshold"]], layered_small_traces, topography,
        surface="base:OXIDE", conflict_tolerance=tolerance,
    )  # fmt: skip
    assert set(picks.table.loc[picks.table["conflict"], "hole_id"]) == expected
    assert entries(picks, "assumed", "Conflict tolerance") == []  # given, not assumed
    default = sf.depth_surface(picks, topography, RECTANGLE, SPACING, name="base:OXIDE")
    kept = sf.depth_surface(picks, topography, RECTANGLE, SPACING, name="base:OXIDE",
                            keep_conflicts=True)  # fmt: skip
    assert default.diagnostics["conflicts"] == kept.diagnostics["conflicts"] == len(expected)
    assert set(default.diagnostics["holes_conflict"]) == expected
    assert sum(default.diagnostics["used"].values()) == 64 - len(expected)
    assert sum(kept.diagnostics["used"].values()) == 64
    (entry,) = entries(default, "assumed", "Conflicting picks")
    assert entry.count == len(expected)
    # with the default tolerance (one assay interval) the two sources agree everywhere
    assert built["base:OXIDE", "combined"].diagnostics["conflicts"] == 0
    assert built["base:OXIDE", "combined"].diagnostics["picks"]["conflict_tolerance"] == 1.0


def test_absent_zero_policy_is_recorded(sources, layered_small_traces, topography):
    picks = sf.surface_picks([sources["oxide_log"]], layered_small_traces, topography,
                             surface="base:OXIDE", absent="zero")  # fmt: skip
    (policy,) = entries(picks, "assumed", "Holes without the units")
    n_upper = int((picks.table["kind"] == "upper").sum())
    assert policy.source == "option absent = zero" and policy.count == n_upper > 0
    with pytest.raises(ValueError, match="at least one source"):
        sf.surface_picks([], layered_small_traces, topography, surface="base:OXIDE")
