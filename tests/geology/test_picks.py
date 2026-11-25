"""Picks of cover and weathering bases with bounds (P9.T6, geology_contracts.md 4).

Independent checks: the analytic truth of the synthetic layered deposit (the contact depth
along each hole is found here with scipy ``brentq`` on straight hole paths built from the
collar and survey tables by hand, not with the desurvey), closed-form geometry of an inclined
hole under a sloping plane, and counts computed with pandas.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import brentq

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geology.picks import (
    PICK_COLUMNS,
    PickResult,
    combine_picks,
    read_extra_intervals,
    threshold_units,
    unit_picks,
)
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geometry.dtm import DTM, dtm_from_grid
from bakumine.core.geometry.mesh import Mesh

# ----------------------------------------------------------------------------------------
# hand-built case: a plane z = 100 + 0.1 x, one vertical and one inclined hole
# ----------------------------------------------------------------------------------------

SLOPE = 0.1
DIP = 60.0
#: vertical depth below the plane per metre along the inclined hole (dip 60 towards east)
INCLINED_FACTOR = math.sin(math.radians(DIP)) + SLOPE * math.cos(math.radians(DIP))


@pytest.fixture(scope="module")
def plane() -> DTM:
    x = np.arange(-20.0, 121.0, 20.0)
    y = np.arange(-20.0, 121.0, 20.0)
    z = 100.0 + SLOPE * x[:, None] + 0.0 * y[None, :]
    return dtm_from_grid(z, (x[0], y[0]), (20.0, 20.0))


@pytest.fixture(scope="module")
def traces():
    collar = pd.DataFrame(
        {
            "hole_id": ["V1", "I1", "V2", "FAR"],
            "x": [0.0, 0.0, 40.0, 500.0],
            "y": [0.0, 50.0, 0.0, 0.0],
            "z": [100.0, 100.0, 104.0, 100.0],
            "max_depth": [50.0, 50.0, 50.0, 50.0],
        }
    )
    survey = pd.DataFrame(
        {
            "hole_id": ["V1", "I1", "V2", "FAR"],
            "depth": [0.0, 0.0, 0.0, 0.0],
            "dip": [-90.0, -DIP, -90.0, -90.0],
            "azimuth": [0.0, 90.0, 0.0, 0.0],
        }
    )
    return desurvey(collar, survey)


def intervals(*rows) -> tuple[pd.DataFrame, pd.Series]:
    """``(hole, from, to, code)`` rows as a table and its unit column (None = not logged)."""
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "code"])
    units = table["code"].astype(object).where(table["code"].notna(), np.nan)
    return table[["hole_id", "depth_from", "depth_to"]], units


def pick(traces, plane, *rows, hole: str = "V1", **options) -> pd.Series:
    table, units = intervals(*rows)
    options.setdefault("surface", "base:OXIDE")
    options.setdefault("unknown_codes", ("CL",))
    result = unit_picks(table, units, {"OX"}, traces, plane, **options)
    assert tuple(result.table.columns) == PICK_COLUMNS
    assert len(result.table) == len(traces)
    row = result.table.set_index("hole_id").loc[hole]
    return row


def all_nan(row: pd.Series, *names: str) -> bool:
    return all(np.isnan(row[name]) for name in names)


def test_exact_pick_where_another_unit_abuts(traces, plane):
    row = pick(traces, plane, ("V1", 0.0, 10.0, "OX"), ("V1", 10.0, 30.0, "ROCK"))
    assert row["kind"] == "exact"
    assert row["source"] == "log"
    assert row["surface"] == "base:OXIDE"
    assert row["depth"] == 10.0
    assert (row["x"], row["y"], row["z"]) == pytest.approx((0.0, 0.0, 90.0), abs=1e-9)
    assert row["topo_z"] == pytest.approx(100.0, abs=1e-9)
    assert row["below_lo"] == pytest.approx(10.0, abs=1e-9)
    assert row["below_hi"] == pytest.approx(10.0, abs=1e-9)
    assert all_nan(row, "depth_max", "x2", "y2", "z2", "topo_z2")


def test_inclined_hole_depth_along_hole_is_not_the_vertical_depth(traces, plane):
    row = pick(traces, plane, ("I1", 0.0, 10.0, "OX"), ("I1", 10.0, 30.0, "ROCK"), hole="I1")
    assert row["kind"] == "exact"
    assert row["depth"] == 10.0
    half = math.cos(math.radians(DIP))
    assert row["x"] == pytest.approx(10.0 * half, abs=1e-9)
    assert row["y"] == pytest.approx(50.0, abs=1e-9)
    assert row["z"] == pytest.approx(100.0 - 10.0 * math.sin(math.radians(DIP)), abs=1e-9)
    # the topography is taken at the pick, not at the collar (100.0)
    assert row["topo_z"] == pytest.approx(100.0 + SLOPE * 10.0 * half, abs=1e-9)
    assert row["below_lo"] == pytest.approx(10.0 * INCLINED_FACTOR, abs=1e-9)
    assert row["below_lo"] < 10.0 - 0.5  # 9.16 m below the surface for 10 m of hole
    assert row["below_hi"] == row["below_lo"]


def test_hole_that_ends_inside_the_unit_gives_a_lower_bound(traces, plane):
    row = pick(traces, plane, ("V1", 0.0, 4.0, "ROCK"), ("V1", 4.0, 18.0, "OX"))
    assert row["kind"] == "lower"
    assert row["depth"] == 18.0
    assert row["below_lo"] == pytest.approx(18.0, abs=1e-9)
    assert all_nan(row, "below_hi", "depth_max", "x2", "topo_z2")


@pytest.mark.parametrize("code", ["CL", None])
def test_unknown_to_the_end_gives_a_lower_bound(traces, plane, code):
    row = pick(traces, plane, ("V1", 0.0, 12.0, "OX"), ("V1", 12.0, 30.0, code))
    assert row["kind"] == "lower"
    assert row["depth"] == 12.0
    assert np.isnan(row["below_hi"])


def test_core_loss_across_the_contact_gives_bounds_either_side(traces, plane):
    rows = (("I1", 0.0, 10.0, "OX"), ("I1", 10.0, 12.5, "CL"), ("I1", 12.5, 30.0, "ROCK"))
    row = pick(traces, plane, *rows, hole="I1")
    assert row["kind"] == "between"
    assert (row["depth"], row["depth_max"]) == (10.0, 12.5)
    assert row["below_lo"] == pytest.approx(10.0 * INCLINED_FACTOR, abs=1e-9)
    assert row["below_hi"] == pytest.approx(12.5 * INCLINED_FACTOR, abs=1e-9)
    half = math.cos(math.radians(DIP))
    assert (row["x"], row["x2"]) == pytest.approx((10.0 * half, 12.5 * half), abs=1e-9)
    assert row["topo_z2"] == pytest.approx(100.0 + SLOPE * 12.5 * half, abs=1e-9)
    assert row["z2"] < row["z"]


def test_unlogged_gap_across_the_contact_is_a_bracket_and_a_tiny_gap_is_not(traces, plane):
    row = pick(traces, plane, ("V1", 0.0, 10.0, "OX"), ("V1", 11.0, 30.0, "ROCK"))
    assert (row["kind"], row["depth"], row["depth_max"]) == ("between", 10.0, 11.0)
    row = pick(traces, plane, ("V1", 0.0, 10.0, "OX"), ("V1", 10.0 + 5e-7, 30.0, "ROCK"))
    assert row["kind"] == "exact"
    assert row["depth"] == pytest.approx(10.0, abs=1e-6)
    row = pick(
        traces,
        plane,
        ("V1", 0.0, 10.0, "OX"),
        ("V1", 10.004, 30.0, "ROCK"),
        depth_tolerance=0.01,
    )
    assert row["kind"] == "exact"


def test_unit_absent_with_rock_from_the_collar(traces, plane):
    rows = (("V1", 0.0, 20.0, "ROCK"), ("V1", 20.0, 30.0, "OTHER"))
    row = pick(traces, plane, *rows)
    assert row["kind"] == "none"  # default: the zone was simply not logged
    assert all_nan(row, *PICK_COLUMNS[4:])  # unknown is NaN, never a zero depth

    row = pick(traces, plane, *rows, absent="zero")
    assert row["kind"] == "upper"
    assert row["depth"] == 0.0
    assert row["below_hi"] == pytest.approx(0.0, abs=1e-9)
    assert np.isnan(row["below_lo"])


def test_absent_zero_bound_sits_at_the_first_logged_unit(traces, plane):
    rows = (("I1", 0.0, 3.0, "CL"), ("I1", 3.0, 20.0, "ROCK"))
    row = pick(traces, plane, *rows, hole="I1", absent="zero")
    assert (row["kind"], row["depth"]) == ("upper", 3.0)
    assert row["below_hi"] == pytest.approx(3.0 * INCLINED_FACTOR, abs=1e-9)
    # nothing usable logged: no information even with absent = zero
    row = pick(traces, plane, ("V1", 0.0, 20.0, "CL"), ("V1", 20.0, 30.0, None), absent="zero")
    assert row["kind"] == "none"


def test_holes_without_intervals_and_without_a_trace(traces, plane):
    table, units = intervals(
        ("V1", 0.0, 10.0, "OX"), ("V1", 10.0, 30.0, "ROCK"), ("GHOST", 0.0, 5.0, "OX")
    )
    result = unit_picks(table, units, {"OX"}, traces, plane, surface="base:OXIDE")
    assert isinstance(result, PickResult)
    out = result.table.set_index("hole_id")
    assert list(out.index) == ["V1", "I1", "V2", "FAR"]  # one row per hole with a trace
    assert list(out["kind"]) == ["exact", "none", "none", "none"]
    assert out.loc[["I1", "V2", "FAR"], list(PICK_COLUMNS[4:])].isna().all().all()
    assert result.diagnostics["holes_skipped"] == ["GHOST"]
    assert result.diagnostics["holes_without_intervals"] == ["I1", "V2", "FAR"]
    assert result.diagnostics["counts"] == {
        "exact": 1,
        "lower": 0,
        "upper": 0,
        "between": 0,
        "none": 3,
    }


def test_deepest_and_first_run(traces, plane):
    rows = (
        ("V1", 0.0, 5.0, "OX"),
        ("V1", 5.0, 8.0, "ROCK"),
        ("V1", 8.0, 12.0, "OX"),
        ("V1", 12.0, 30.0, "ROCK"),
    )
    assert pick(traces, plane, *rows)["depth"] == 12.0
    assert pick(traces, plane, *rows, mode="first_run")["depth"] == 5.0
    # core loss inside the unit does not end the run
    rows = (
        ("V1", 0.0, 5.0, "OX"),
        ("V1", 5.0, 6.0, "CL"),
        ("V1", 6.0, 9.0, "OX"),
        ("V1", 9.0, 30.0, "ROCK"),
    )
    for mode in ("deepest", "first_run"):
        row = pick(traces, plane, *rows, mode=mode)
        assert (row["kind"], row["depth"]) == ("exact", 9.0)


def test_other_kinds_of_units_are_out_and_listed_codes_are_unknown(traces, plane):
    rows = (("V1", 0.0, 10.0, "OX"), ("V1", 10.0, 12.0, "BX"), ("V1", 12.0, 30.0, "ROCK"))
    assert pick(traces, plane, *rows)["kind"] == "exact"
    row = pick(traces, plane, *rows, unknown_codes=("CL", "BX"))
    assert (row["kind"], row["depth"], row["depth_max"]) == ("between", 10.0, 12.0)


def test_rows_in_any_order_and_split_intervals(traces, plane):
    rows = (
        ("V1", 14.0, 30.0, "ROCK"),
        ("V1", 6.0, 14.0, "OX"),
        ("V1", 0.0, 6.0, "OX"),
        ("V1", 30.0, 30.0, "OX"),  # zero length: dropped
    )
    table, units = intervals(*rows)
    result = unit_picks(table, units, ["OX"], traces, plane, surface="base:OXIDE")
    row = result.table.set_index("hole_id").loc["V1"]
    assert (row["kind"], row["depth"]) == ("exact", 14.0)
    assert result.diagnostics["rows_dropped"] == 1


def test_top_side_mirrors_the_rules(traces, plane):
    options = {"side": "top", "surface": "top:SULPHIDE"}
    row = pick(traces, plane, ("V1", 0.0, 7.0, "ROCK"), ("V1", 7.0, 30.0, "OX"), **options)
    assert (row["kind"], row["depth"], row["surface"]) == ("exact", 7.0, "top:SULPHIDE")
    # the unit starts at the top of the logging: its top is not deeper than that
    row = pick(traces, plane, ("V1", 2.0, 30.0, "OX"), **options)
    assert (row["kind"], row["depth"]) == ("upper", 2.0)
    assert row["below_hi"] == pytest.approx(2.0, abs=1e-9)
    assert np.isnan(row["below_lo"])
    rows = (("V1", 0.0, 7.0, "ROCK"), ("V1", 7.0, 9.0, "CL"), ("V1", 9.0, 30.0, "OX"))
    row = pick(traces, plane, *rows, **options)
    assert (row["kind"], row["depth"], row["depth_max"]) == ("between", 7.0, 9.0)
    assert (row["below_lo"], row["below_hi"]) == pytest.approx((7.0, 9.0), abs=1e-9)
    # absent = zero: the top of an absent unit is not above the end of the logging
    row = pick(traces, plane, ("V1", 0.0, 25.0, "ROCK"), absent="zero", **options)
    assert (row["kind"], row["depth"]) == ("lower", 25.0)


def test_pick_outside_the_topography_keeps_its_depth_and_has_no_vertical_depth(traces, plane):
    table, units = intervals(("FAR", 0.0, 10.0, "OX"), ("FAR", 10.0, 30.0, "ROCK"))
    result = unit_picks(table, units, {"OX"}, traces, plane, surface="base:OXIDE")
    row = result.table.set_index("hole_id").loc["FAR"]
    assert (row["kind"], row["depth"]) == ("exact", 10.0)
    assert all_nan(row, "topo_z", "below_lo", "below_hi")
    assert result.diagnostics["holes_outside_topography"] == ["FAR"]


def test_invalid_arguments(traces, plane):
    table, units = intervals(("V1", 0.0, 10.0, "OX"))
    call = lambda **kw: unit_picks(  # noqa: E731
        table, units, kw.pop("target", {"OX"}), traces, plane, **{"surface": "base:OXIDE", **kw}
    )
    for bad in (
        {"side": "left"},
        {"mode": "shallowest"},
        {"absent": "no"},
        {"source": "guess"},
        {"surface": " "},
        {"target": ()},
        {"unknown_codes": ("OX",)},
    ):
        with pytest.raises(ValueError):
            call(**bad)
    with pytest.raises(ValueError, match="units has"):
        unit_picks(table, units.iloc[:0], {"OX"}, traces, plane, surface="base:OXIDE")
    with pytest.raises(ValueError, match="depth_to"):
        unit_picks(table[["hole_id", "depth_from"]], units, {"OX"}, traces, plane, surface="s")


# ----------------------------------------------------------------------------------------
# threshold rule and the extra intervals file
# ----------------------------------------------------------------------------------------


def test_threshold_units_classes_and_keeps_missing():
    table = pd.DataFrame({"s": [200.0, 999.9, 1000.0, np.nan, 4000.0]}, index=[5, 6, 7, 8, 9])
    below = threshold_units(table, "s", below=1000.0)
    assert list(below.index) == [5, 6, 7, 8, 9]
    assert list(below.iloc[[0, 1, 2, 4]]) == ["IN", "IN", "OUT", "OUT"]
    assert pd.isna(below.iloc[3])  # missing stays missing
    above = threshold_units(table, "s", above=1000.0, inside="FRESH", outside="WEATHERED")
    assert list(above.iloc[[0, 1, 2, 4]]) == ["WEATHERED", "WEATHERED", "WEATHERED", "FRESH"]
    assert pd.isna(above.iloc[3])
    for bad in ({}, {"below": 1.0, "above": 2.0}, {"below": np.nan}):
        with pytest.raises(ValueError):
            threshold_units(table, "s", **bad)
    with pytest.raises(ValueError):
        threshold_units(table, "missing_column", below=1.0)
    with pytest.raises(ValueError):
        threshold_units(table, "s", below=1.0, inside="A", outside="A")


def test_threshold_picks_and_their_provenance(traces, plane):
    assay = pd.DataFrame(
        {
            "hole_id": ["V1"] * 5 + ["I1"] * 4 + ["V2"] * 3,
            "depth_from": [0.0, 2.0, 4.0, 6.0, 8.0, 0.0, 2.0, 4.0, 6.0, 0.0, 2.0, 4.0],
            "depth_to": [2.0, 4.0, 6.0, 8.0, 10.0, 2.0, 4.0, 6.0, 8.0, 2.0, 4.0, 6.0],
            # V1: weathered to 6 m; I1: to 4 m, no assay 4 - 6 m; V2: fresh from the collar
            "s": [150.0, 300.0, 250.0, 5000.0, 6000.0]
            + [100.0, 200.0, np.nan, 7000.0]
            + [4000.0, 5000.0, 6000.0],
        }
    )  # fmt: skip
    classes = threshold_units(assay, "s", below=1000.0)
    result = unit_picks(
        assay, classes, {"IN"}, traces, plane, surface="base:OXIDE", source="threshold",
        absent="zero",
    )  # fmt: skip
    out = result.table.set_index("hole_id")
    assert set(out["source"]) == {"threshold"}
    assert (out.loc["V1", "kind"], out.loc["V1", "depth"]) == ("exact", 6.0)
    assert out.loc["I1", ["kind", "depth", "depth_max"]].tolist() == ["between", 4.0, 6.0]
    assert out.loc["I1", "below_hi"] == pytest.approx(6.0 * INCLINED_FACTOR, abs=1e-9)
    assert (out.loc["V2", "kind"], out.loc["V2", "depth"]) == ("upper", 0.0)
    assert out.loc["FAR", "kind"] == "none"

    by_item = {entry.item: entry for entry in result.provenance}
    rule = by_item["Threshold rule for base:OXIDE"]
    assert rule.kind == "measured"
    assert rule.source == "column s < 1000"
    assert rule.value == 1000.0
    assert rule.count == int((assay["s"] < 1000.0).sum()) == 5
    assert by_item["Exact picks of base:OXIDE (threshold)"].count == 1
    assert (
        by_item["Bounds of base:OXIDE (threshold)"].count == 1
    )  # the bracket; not the policy bound
    policy = by_item["Holes without the units of base:OXIDE (threshold)"]
    assert (policy.kind, policy.source, policy.count) == ("assumed", "option absent = zero", 1)
    assert by_item["base:OXIDE not logged (threshold)"].count == 1
    assert {entry.stage for entry in result.provenance} == {"picks"}
    assert {entry.task for entry in result.provenance} == {"P9.T6"}


def test_read_extra_intervals(tmp_path, traces, plane):
    path = tmp_path / "weathering.csv"
    path.write_text(
        "Hole, From,To ,Zone,remark\n"
        " V1 ,0,8.5, ox ,a\n"
        "V1,8.5,30,FR,\n"
        "I1,0,12,ox,\n"
        "I1,12,14,,not logged\n"
        "I1,14,30,FR,\n",
        encoding="utf-8",
    )
    extra = read_extra_intervals(
        path, hole_id="Hole", depth_from="From", depth_to="To", code="Zone"
    )
    assert list(extra.columns) == ["hole_id", "depth_from", "depth_to", "code"]
    assert extra["hole_id"].tolist() == ["V1", "V1", "I1", "I1", "I1"]
    assert extra["depth_to"].tolist() == [8.5, 30.0, 12.0, 14.0, 30.0]
    assert extra["depth_from"].dtype == np.float64
    assert extra["code"].tolist()[:3] == ["ox", "FR", "ox"]
    assert pd.isna(extra["code"].iloc[3])  # an empty code is unknown, not a unit

    units = extra["code"].str.upper()
    result = unit_picks(extra, units, {"OX"}, traces, plane, surface="base:OXIDE", source="extra")
    out = result.table.set_index("hole_id")
    assert set(out["source"]) == {"extra"}
    assert (out.loc["V1", "kind"], out.loc["V1", "depth"]) == ("exact", 8.5)
    assert out.loc["I1", ["kind", "depth", "depth_max"]].tolist() == ["between", 12.0, 14.0]
    assert "extra intervals file" in result.provenance[0].source


def test_read_extra_intervals_rejects_bad_files(tmp_path):
    def write(text: str):
        path = tmp_path / "bad.csv"
        path.write_text(text, encoding="utf-8")
        return path

    with pytest.raises(ValueError, match="missing column"):
        read_extra_intervals(write("hole_id,depth_from,depth_to\nA,0,1\n"))
    with pytest.raises(ValueError, match="line 3.*not a number"):
        read_extra_intervals(write("hole_id,depth_from,depth_to,code\nA,0,1,OX\nA,x,2,OX\n"))
    with pytest.raises(ValueError, match="line 2.*depth_to"):
        read_extra_intervals(write("hole_id,depth_from,depth_to,code\nA,5,5,OX\n"))
    with pytest.raises(ValueError, match="empty hole id"):
        read_extra_intervals(write("hole_id,depth_from,depth_to,code\n ,0,1,OX\n"))


# ----------------------------------------------------------------------------------------
# combining sources
# ----------------------------------------------------------------------------------------


def source_picks(traces, plane, source: str, *rows, **options) -> pd.DataFrame:
    table, units = intervals(*rows)
    return unit_picks(
        table, units, {"OX"}, traces, plane, surface="base:OXIDE", source=source,
        unknown_codes=("CL",), **options,
    ).table  # fmt: skip


def test_combine_first_exact_wins_and_bounds_are_tightened(traces, plane):
    log = source_picks(
        traces, plane, "log",
        ("V1", 0.0, 9.0, "OX"), ("V1", 9.0, 30.0, "ROCK"),      # exact 9
        ("I1", 0.0, 6.0, "OX"),                                 # lower 6
        ("V2", 0.0, 4.0, "OX"), ("V2", 4.0, 9.0, "CL"), ("V2", 9.0, 30.0, "ROCK"),  # 4 .. 9
    )  # fmt: skip
    extra = source_picks(
        traces, plane, "extra",
        ("V1", 0.0, 9.0, "OX"), ("V1", 9.0, 30.0, "ROCK"),      # the same exact pick
        ("I1", 0.0, 8.0, "OX"), ("I1", 8.0, 11.0, "CL"), ("I1", 11.0, 30.0, "ROCK"),  # 8 .. 11
        ("V2", 0.0, 5.0, "OX"), ("V2", 5.0, 7.0, "CL"), ("V2", 7.0, 30.0, "ROCK"),    # 5 .. 7
        ("FAR", 0.0, 3.0, "OX"), ("FAR", 3.0, 9.0, "ROCK"),
    )  # fmt: skip
    out = combine_picks(log, extra)
    assert list(out.columns) == [*PICK_COLUMNS, "conflict"]
    assert out["conflict"].dtype == bool
    assert not out["conflict"].any()
    assert len(out) == len(traces)
    out = out.set_index("hole_id")

    assert out.loc["V1", ["kind", "source", "depth"]].tolist() == ["exact", "log", 9.0]
    # lower 6 (log) and bracket 8 .. 11 (extra): the deepest lower bound, the only upper bound
    row = out.loc["I1"]
    assert row[["kind", "source", "depth", "depth_max"]].tolist() == ["between", "extra", 8.0, 11.0]
    assert row["below_lo"] == pytest.approx(8.0 * INCLINED_FACTOR, abs=1e-9)
    assert row["below_hi"] == pytest.approx(11.0 * INCLINED_FACTOR, abs=1e-9)
    # brackets 4 .. 9 and 5 .. 7: the tightest pair
    assert out.loc["V2", ["kind", "depth", "depth_max"]].tolist() == ["between", 5.0, 7.0]
    assert out.loc["FAR", ["kind", "source"]].tolist() == ["exact", "extra"]

    # argument order decides between exact picks, and between sources that only add a bound
    assert combine_picks(extra, log).set_index("hole_id").loc["V1", "source"] == "extra"


def test_combine_joins_bounds_of_different_sources(traces, plane):
    log = source_picks(traces, plane, "log", ("V1", 0.0, 6.0, "OX"), ("I1", 0.0, 5.0, "OX"))
    threshold = source_picks(
        traces, plane, "threshold", ("V1", 0.0, 8.0, "ROCK"), ("I1", 5.0, 9.0, "ROCK"),
        absent="zero",
    )  # fmt: skip
    # the policy bound of V1 is at the collar (0 m), above the logged lower bound at 6 m
    out = combine_picks(log, threshold).set_index("hole_id")
    assert bool(out.loc["V1", "conflict"])
    assert out.loc["V1", ["kind", "source", "depth"]].tolist() == ["lower", "log", 6.0]
    # I1: lower 5 (log) and upper 5 (threshold) meet, so the contact is known
    row = out.loc["I1"]
    assert row[["kind", "source", "depth"]].tolist() == ["exact", "log+threshold", 5.0]
    assert row["below_lo"] == row["below_hi"] == pytest.approx(5.0 * INCLINED_FACTOR, abs=1e-9)
    assert not row["conflict"]
    assert np.isnan(row["depth_max"])
    assert out.loc["V2", "kind"] == "none" and not out.loc["V2", "conflict"]


def test_combine_flags_conflicting_sources(traces, plane):
    log = source_picks(
        traces, plane, "log",
        ("V1", 0.0, 9.0, "OX"), ("V1", 9.0, 30.0, "ROCK"),      # exact 9
        ("I1", 0.0, 15.0, "OX"),                                # lower 15
        ("V2", 0.0, 9.0, "OX"), ("V2", 9.0, 30.0, "ROCK"),      # exact 9
    )  # fmt: skip
    extra = source_picks(
        traces, plane, "extra",
        ("V1", 0.0, 12.0, "OX"), ("V1", 12.0, 30.0, "ROCK"),    # exact 12: differs
        ("I1", 0.0, 4.0, "OX"), ("I1", 4.0, 6.0, "CL"), ("I1", 6.0, 30.0, "ROCK"),  # 4 .. 6
        ("V2", 0.0, 12.0, "OX"),                                # lower 12 below the exact 9
    )  # fmt: skip
    out = combine_picks(log, extra).set_index("hole_id")
    assert out["conflict"].tolist() == [True, True, True, False]
    assert out.loc["V1", ["kind", "source", "depth"]].tolist() == ["exact", "log", 9.0]
    # lower bound (15) below the upper bound (6): the first table is kept, the row is flagged
    assert out.loc["I1", ["kind", "source", "depth"]].tolist() == ["lower", "log", 15.0]
    assert np.isnan(out.loc["I1", "below_hi"])
    assert out.loc["V2", ["kind", "depth"]].tolist() == ["exact", 9.0]
    # a tolerance wider than the disagreement of the exact picks removes that flag only
    relaxed = combine_picks(log, extra, tolerance=3.5).set_index("hole_id")
    assert relaxed["conflict"].tolist() == [False, True, False, False]


def test_combine_keeps_surfaces_apart_and_handles_empty_input(traces, plane):
    table, units = intervals(("V1", 0.0, 3.0, "Q4"), ("V1", 3.0, 9.0, "OX"), ("V1", 9.0, 30.0, "R"))
    cover = unit_picks(table, units, {"Q4"}, traces, plane, surface="base:Q4").table
    oxide = unit_picks(table, units, {"OX"}, traces, plane, surface="base:OXIDE").table
    out = combine_picks(cover, oxide)
    assert len(out) == 2 * len(traces)
    v1 = out[out["hole_id"] == "V1"].set_index("surface")
    assert v1["depth"].to_dict() == {"base:Q4": 3.0, "base:OXIDE": 9.0}
    empty = combine_picks()
    assert list(empty.columns) == [*PICK_COLUMNS, "conflict"] and len(empty) == 0
    with pytest.raises(ValueError, match="lacks the columns"):
        combine_picks(cover.drop(columns="below_hi"))


# ----------------------------------------------------------------------------------------
# synthetic layered deposit: picks against the analytic truth
# ----------------------------------------------------------------------------------------

LOG_ROUNDING = 0.005  # contacts are logged to 0.01 m
#: allowance on vertical depths: rounding along the hole, the 1 m topography grid (measured
#: below at under 2 mm) and the bisection of the test
VERTICAL_TOL = 0.01


@pytest.fixture(scope="module")
def synthetic(layered_small, layered_small_db, layered_small_traces, layered_units_path):
    truth = layered_small.truth
    collar = layered_small_db.collar.set_index("hole_id")
    xs = np.arange(collar["x"].min() - 150.0, collar["x"].max() + 151.0, 1.0)
    ys = np.arange(collar["y"].min() - 150.0, collar["y"].max() + 151.0, 1.0)
    grid = truth.topo_z(*np.meshgrid(xs, ys, indexing="ij"))
    topography = dtm_from_grid(grid, (xs[0], ys[0]), (1.0, 1.0))

    # straight hole paths by hand: collar + depth * direction from the first survey row
    survey = layered_small_db.survey.sort_values(["hole_id", "depth"]).groupby("hole_id").first()
    assert layered_small_db.dip_convention == "negative_down"
    dip = np.radians(survey["dip"].to_numpy(dtype=float))
    azimuth = np.radians(survey["azimuth"].to_numpy(dtype=float))
    direction = pd.DataFrame(
        {
            "dx": np.cos(dip) * np.sin(azimuth),
            "dy": np.cos(dip) * np.cos(azimuth),
            "dz": np.sin(dip),
        },
        index=survey.index,
    )

    def path(hole_id, depth):
        c = collar.loc[hole_id, ["x", "y", "z"]].to_numpy(dtype=float)
        return c + depth * direction.loc[hole_id].to_numpy(dtype=float)

    def truth_depth(hole_id, depth_of) -> float:
        """Depth along the hole where the hole crosses the analytic surface."""

        def above(d: float) -> float:
            x, y, z = path(hole_id, d)
            return float(truth.topo_z(x, y) - z - depth_of(x, y))

        return brentq(above, 0.0, 80.0, xtol=1e-10)

    units = UnitDictionary.load(layered_units_path)
    geology = layered_small_db.intervals["geology"]
    return {
        "truth": truth,
        "topography": topography,
        "traces": layered_small_traces,
        "units": units,
        "geology": geology,
        "mapped": units.map_codes(geology["lith"]),
        "unknown": (*units.codes("missing"), *units.codes("fault-zone")),
        "assay": layered_small_db.intervals["assay"],
        "path": path,
        "truth_depth": truth_depth,
        "written_topography": DTM(Mesh(*layered_small.meshes["topo"])),
    }


def log_picks(s, target: str, surface: str, **options) -> PickResult:
    return unit_picks(
        options.pop("table", s["geology"]),
        options.pop("mapped", s["mapped"]),
        {target},
        s["traces"],
        s["topography"],
        surface=surface,
        unknown_codes=s["unknown"],
        **options,
    )


def expected_log_counts(s, target: str) -> dict[str, int]:
    """Counts per kind from the logged table with pandas alone (no run logic of the module)."""
    table = s["geology"].assign(unit=s["mapped"]).sort_values(["hole_id", "depth_from"])
    table = table[~table["unit"].isin(list(s["unknown"]))]  # what is left is logged rock or zone
    counts = {"exact": 0, "lower": 0, "upper": 0, "between": 0, "none": 0}
    for _, hole in table.groupby("hole_id"):
        inside = np.flatnonzero((hole["unit"] == target).to_numpy())
        if not inside.size:
            counts["none"] += 1
            continue
        last = inside[-1]
        if last + 1 == len(hole):
            counts["lower"] += 1
        elif hole["depth_from"].iloc[last + 1] == hole["depth_to"].iloc[last]:
            counts["exact"] += 1
        else:
            counts["between"] += 1
    counts["none"] += len(s["traces"]) - table["hole_id"].nunique()
    return counts


def check_against_truth(s, table: pd.DataFrame, depth_of, *, along: float, vertical: float):
    """Exact picks at the truth; every bound on the right side of it. Returns the kinds seen."""
    truth = s["truth"]
    for row in table.itertuples(index=False):
        if row.kind == "none":
            assert all(np.isnan(getattr(row, c)) for c in PICK_COLUMNS[4:])
            continue
        d_true = s["truth_depth"](row.hole_id, depth_of)
        # the position and the vertical depth of the pick, from the hand-built straight path
        x, y, z = s["path"](row.hole_id, row.depth)
        assert (row.x, row.y, row.z) == pytest.approx((x, y, z), abs=1e-6)
        below = float(truth.topo_z(x, y) - z)
        here = float(depth_of(x, y))  # the true depth of the surface under the pick
        if row.kind == "exact":
            assert abs(row.depth - d_true) <= along
            assert row.below_lo == row.below_hi == pytest.approx(below, abs=2e-3)
            assert abs(row.below_lo - here) <= vertical
        if row.kind in ("lower", "between"):
            assert row.depth <= d_true + along
            assert row.below_lo == pytest.approx(below, abs=2e-3)
            assert row.below_lo <= here + vertical
        if row.kind == "upper":
            assert row.depth >= d_true - along
            assert row.below_hi == pytest.approx(below, abs=2e-3)
            assert row.below_hi >= here - vertical
        if row.kind == "between":
            assert row.depth_max >= d_true - along
            x2, y2, z2 = s["path"](row.hole_id, row.depth_max)
            assert (row.x2, row.y2, row.z2) == pytest.approx((x2, y2, z2), abs=1e-6)
            assert row.below_hi == pytest.approx(float(truth.topo_z(x2, y2) - z2), abs=2e-3)
            assert row.below_hi >= float(depth_of(x2, y2)) - vertical
    return set(table["kind"])


def test_synthetic_topography_of_the_test_is_the_analytic_one(synthetic):
    s = synthetic
    rng = np.random.default_rng(7)
    lo, hi = s["topography"].bounds[0, :2] + 1.0, s["topography"].bounds[1, :2] - 1.0
    xy = rng.uniform(lo, hi, size=(4000, 2))
    error = np.abs(s["topography"].z_at(xy) - s["truth"].topo_z(xy[:, 0], xy[:, 1]))
    assert error.max() < 2e-3


def test_synthetic_oxide_picks_agree_with_the_truth(synthetic):
    s = synthetic
    result = log_picks(s, "OX", "base:OXIDE")
    table = result.table
    assert tuple(table.columns) == PICK_COLUMNS
    assert table["hole_id"].tolist() == list(s["traces"].hole_ids)
    kinds = check_against_truth(
        s, table, s["truth"].oxide_depth, along=LOG_ROUNDING + 1e-6, vertical=VERTICAL_TOL
    )
    assert "exact" in kinds and "none" in kinds

    expected = expected_log_counts(s, "OX")
    assert result.diagnostics["counts"] == expected
    assert table["kind"].value_counts().to_dict() == {k: v for k, v in expected.items() if v}
    # the oxide zone is logged in two holes of five: the others say nothing, and nothing is zero
    logged = s["geology"].loc[(s["mapped"] == "OX").to_numpy(), "hole_id"].nunique()
    assert expected["none"] == len(s["traces"]) - logged > 20
    silent = table[table["kind"] == "none"]
    assert silent[list(PICK_COLUMNS[4:])].isna().all().all()
    assert (table.loc[table["kind"] != "none", "below_lo"] > 10.0).all()
    assert result.diagnostics["holes_skipped"] == []
    assert result.diagnostics["holes_outside_topography"] == []


def test_synthetic_cover_picks_agree_with_the_truth(synthetic):
    s = synthetic
    result = log_picks(s, "Q4", "base:Q4")
    kinds = check_against_truth(
        s, result.table, s["truth"].cover_depth, along=LOG_ROUNDING + 1e-6, vertical=VERTICAL_TOL
    )
    assert "exact" in kinds
    assert result.diagnostics["counts"] == expected_log_counts(s, "Q4")
    exact = result.table[result.table["kind"] == "exact"]
    assert exact["below_lo"].between(2.0 - VERTICAL_TOL, 6.0 + VERTICAL_TOL).all()


def test_synthetic_inclined_and_vertical_holes(synthetic):
    # the vertical holes of the deposit log neither soil nor the oxide zone, so the sulphur
    # rule is used here: it gives a pick in every hole
    s = synthetic
    classes = threshold_units(s["assay"], "s_ppm", below=1000.0)
    table = unit_picks(
        s["assay"], classes, {"IN"}, s["traces"], s["topography"], surface="base:OXIDE",
        source="threshold",
    ).table  # fmt: skip
    exact = table[table["kind"] == "exact"]
    collar = s["traces"].stations().groupby("hole_id").first()
    steep = collar.loc[exact["hole_id"], "dz"].to_numpy() < -0.999
    assert steep.any() and (~steep).any()
    ratio = (exact["below_lo"] / exact["depth"]).to_numpy()
    # vertical holes: the depth along the hole is the vertical depth
    assert ratio[steep] == pytest.approx(1.0, abs=1e-3)
    # holes at 60 degrees: the vertical depth is not the depth along the hole, and not
    # depth x sin(60) either, because the ground is not level between the collar and the pick
    # (the exact value is checked against the hand-built path in check_against_truth)
    assert (ratio[~steep] < 0.99).all()
    assert np.abs(ratio[~steep] - math.sin(math.radians(60.0))).max() > 0.05


@pytest.mark.parametrize("cut", [8.0, 12.0, 20.0, 26.0])
def test_synthetic_truncated_holes_give_lower_bounds_consistent_with_the_truth(synthetic, cut):
    s = synthetic
    keep = (s["geology"]["depth_from"] < cut).to_numpy()
    table = s["geology"][keep].copy()
    table["depth_to"] = table["depth_to"].clip(upper=cut)
    mapped = s["mapped"][keep]
    result = log_picks(s, "OX", "base:OXIDE", table=table, mapped=mapped)
    kinds = check_against_truth(
        s, result.table, s["truth"].oxide_depth, along=LOG_ROUNDING + 1e-6, vertical=VERTICAL_TOL
    )
    assert "lower" in kinds
    lower = result.table[result.table["kind"] == "lower"]
    assert (lower["depth"] <= cut).all()
    assert lower["below_hi"].isna().all()
    if cut <= 12.0:  # every hole stops above the oxide base (at least 15 m deep)
        assert "exact" not in kinds
    bounds = [e for e in result.provenance if e.item == "Bounds of base:OXIDE (log)"]
    assert bounds[0].count == int((result.table["kind"] == "lower").sum())


def test_synthetic_unlogged_window_over_the_contact_gives_brackets(synthetic):
    s = synthetic
    geology = s["geology"].assign(unit=s["mapped"])
    rows = []
    windows = {}
    for hole_id, hole in geology.groupby("hole_id", sort=False):
        if not (hole["unit"] == "OX").any():
            rows.append(hole)
            continue
        d_true = s["truth_depth"](hole_id, s["truth"].oxide_depth)
        a, b = round(d_true - 1.3, 2), round(d_true + 2.1, 2)
        windows[hole_id] = (a, b)
        upper = hole[hole["depth_from"] < a].copy()
        upper["depth_to"] = upper["depth_to"].clip(upper=a)
        lower = hole[hole["depth_to"] > b].copy()
        lower["depth_from"] = lower["depth_from"].clip(lower=b)
        rows.extend([upper, lower])
    table = pd.concat(rows, ignore_index=True)
    result = log_picks(s, "OX", "base:OXIDE", table=table, mapped=table["unit"])
    kinds = check_against_truth(
        s, result.table, s["truth"].oxide_depth, along=LOG_ROUNDING + 1e-6, vertical=VERTICAL_TOL
    )
    assert "between" in kinds and "exact" not in kinds
    between = result.table[result.table["kind"] == "between"].set_index("hole_id")
    assert len(between) >= 20
    for hole_id, row in between.iterrows():
        a, b = windows[hole_id]
        # the bracket holds the window (it is wider where core loss or a fault zone adjoins)
        assert row["depth"] <= a + 1e-9 and row["depth_max"] >= b - 1e-9
        assert row["below_lo"] < row["below_hi"]


def test_synthetic_threshold_picks_follow_the_sample_length(synthetic):
    s = synthetic
    assay = s["assay"]
    classes = threshold_units(assay, "s_ppm", below=1000.0)
    result = unit_picks(
        assay, classes, {"IN"}, s["traces"], s["topography"], surface="base:OXIDE",
        source="threshold", absent="zero",
    )  # fmt: skip
    table = result.table
    # 1 m samples labelled at their midpoint: the contact is within half a sample of the pick
    kinds = check_against_truth(s, table, s["truth"].oxide_depth, along=0.5 + 1e-6, vertical=0.6)
    assert kinds == {"exact"}  # sulphur is assayed in every hole, the logged zone in 2 of 5
    assert (table["source"] == "threshold").all()
    assert (table["depth"] == table["depth"].round()).all()

    by_item = {entry.item: entry for entry in result.provenance}
    rule = by_item["Threshold rule for base:OXIDE"]
    assert rule.source == "column s_ppm < 1000"
    assert rule.count == int((assay["s_ppm"] < 1000.0).sum())
    assert by_item["Exact picks of base:OXIDE (threshold)"].count == len(s["traces"])
    assert "base:OXIDE not logged (threshold)" not in by_item


def test_synthetic_provenance_counts_equal_pandas_counts(synthetic):
    s = synthetic
    for target, surface in (("OX", "base:OXIDE"), ("Q4", "base:Q4")):
        result = log_picks(s, target, surface)
        table = result.table
        by_item = {entry.item: entry for entry in result.provenance}
        assert len(by_item) == len(result.provenance)
        counts = table["kind"].value_counts()
        n_exact = int(counts.get("exact", 0))
        n_bounds = int(counts.reindex(["lower", "upper", "between"]).fillna(0).sum())
        n_none = int(counts.get("none", 0))

        exact = by_item[f"Exact picks of {surface} (log)"]
        assert (exact.kind, exact.count) == ("measured", n_exact)
        assert f"units {target}" in exact.source
        if n_bounds:
            assert by_item[f"Bounds of {surface} (log)"].count == n_bounds
        else:
            assert f"Bounds of {surface} (log)" not in by_item

        # holes with logged intervals and none of the target unit, straight from the table
        logged = s["geology"].assign(unit=s["mapped"])
        logged = logged[logged["unit"].notna() & ~logged["unit"].isin(list(s["unknown"]))]
        with_target = set(logged.loc[logged["unit"] == target, "hole_id"])
        not_logged = logged["hole_id"].nunique() - len(with_target)
        policy = by_item[f"Holes without the units of {surface} (log)"]
        assert (policy.kind, policy.source) == ("assumed", "default (absent = unknown)")
        assert policy.count == not_logged
        missing = by_item[f"{surface} not logged (log)"]
        assert (missing.kind, missing.count) == ("missing", n_none)
        assert n_exact + n_bounds + n_none == len(s["traces"]) == 64
        for entry in result.provenance:
            assert entry.stage == "picks" and entry.task == "P9.T6"
            assert entry.used_for == f"surface {surface}"


def test_synthetic_absent_zero_bounds_are_wrong_for_an_unlogged_zone(synthetic):
    """Why ``absent = unknown`` is the default: read as absence, a zone that was not logged
    gives upper bounds at the collar, far above the true oxide base."""
    s = synthetic
    result = log_picks(s, "OX", "base:OXIDE", absent="zero")
    upper = result.table[result.table["kind"] == "upper"]
    assert len(upper) == expected_log_counts(s, "OX")["none"]
    assert (upper["below_hi"] < 10.0).all()  # the truth is 15 to 30 m
    policy = [e for e in result.provenance if e.source == "option absent = zero"]
    assert len(policy) == 1 and policy[0].kind == "assumed" and policy[0].count == len(upper)
    assert not [e for e in result.provenance if e.item.startswith("Bounds of")]


def test_synthetic_log_and_threshold_combined(synthetic):
    s = synthetic
    log = log_picks(s, "OX", "base:OXIDE").table
    classes = threshold_units(s["assay"], "s_ppm", below=1000.0)
    threshold = unit_picks(
        s["assay"], classes, {"IN"}, s["traces"], s["topography"], surface="base:OXIDE",
        source="threshold",
    ).table  # fmt: skip
    combined = combine_picks(log, threshold, tolerance=0.5 + 2 * LOG_ROUNDING)
    assert len(combined) == len(s["traces"])
    assert not combined["conflict"].any()  # the sources agree within the sample length
    assert (combined["kind"] == "exact").all()

    merged = combined.set_index("hole_id")
    log_exact = set(log.loc[log["kind"] == "exact", "hole_id"])
    assert set(merged.index[merged["source"] == "log"]) == log_exact
    assert int((merged["source"] == "threshold").sum()) == len(s["traces"]) - len(log_exact)
    pd.testing.assert_frame_equal(
        merged.loc[sorted(log_exact), list(PICK_COLUMNS[1:])],
        log.set_index("hole_id").loc[sorted(log_exact), list(PICK_COLUMNS[1:])],
    )
    # with a zero tolerance the half-sample differences of the two sources are reported
    strict = combine_picks(log, threshold)
    assert strict["conflict"].sum() > 0
    assert set(strict.loc[strict["conflict"], "hole_id"]) <= log_exact


def test_synthetic_picks_on_the_written_topography_file(synthetic):
    """The same picks on the triangulated ``topo`` mesh of the dataset: depths along the hole
    are unchanged, vertical depths move only by the error of that mesh."""
    s = synthetic
    fine = log_picks(s, "OX", "base:OXIDE").table
    written = unit_picks(
        s["geology"], s["mapped"], {"OX"}, s["traces"], s["written_topography"],
        surface="base:OXIDE", unknown_codes=s["unknown"],
    ).table  # fmt: skip
    pd.testing.assert_frame_equal(fine[list(PICK_COLUMNS[:12])], written[list(PICK_COLUMNS[:12])])
    picked = fine["kind"] != "none"
    shift = (written.loc[picked, "below_lo"] - fine.loc[picked, "below_lo"]).abs()
    mesh_error = (written.loc[picked, "topo_z"] - fine.loc[picked, "topo_z"]).abs()
    assert np.allclose(shift, mesh_error, atol=1e-9)
    assert shift.max() < 1.0
