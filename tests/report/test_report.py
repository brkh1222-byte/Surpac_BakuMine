"""Volume / tonnes / grade report (P3.T9)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from bakumine.core.blockmodel import AttributeConstraint, BlockModel, BlockModelDefinition
from bakumine.core.report import TROY_OUNCE_G, ReportError, format_table, report, to_csv

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "data" / "synthetic" / "default"

EXACT = pytest.approx  # used with rel=1e-12: sums of a few exactly representable numbers


def hand_model() -> BlockModel:
    """3 x 3 x 3 blocks of 10 x 5 x 5 m: every block is 250 m3.

    rock     i = 0 -> OX (9 blocks), i = 1, 2 -> SU (18 blocks)
    density  OX 2.0, SU 3.0 t/m3
    au       j = 0 -> 1.0, j = 1 -> 2.0, j = 2 -> 4.0 g/t; block (2, 2, 2) has NO grade
    partial  1.0, except block (0, 0, 0) = 0.5 and block (1, 1, 2) = 0.2
    benches  10 m from z = 0: k = 0, 1 (centroids 2.5, 7.5) lower; k = 2 (12.5) upper
    """
    definition = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 5.0, 5.0), (3, 3, 3), max_depth=1)
    model = BlockModel(definition)
    i, j, k = model.ijk().T
    model.add_attribute("rock", "category")
    model.set("rock", np.where(i == 0, "OX", "SU").astype(object))
    model.add_attribute("density")
    model["density"][:] = np.where(i == 0, 2.0, 3.0)
    model.add_attribute("au", unit="g/t")
    model["au"][:] = np.array([1.0, 2.0, 4.0])[j]
    model["au"][(i == 2) & (j == 2) & (k == 2)] = np.nan
    model.add_attribute("part", background=1.0)
    model["part"][(i == 0) & (j == 0) & (k == 0)] = 0.5
    model["part"][(i == 1) & (j == 1) & (k == 2)] = 0.2
    model.add_attribute("zone", "int")
    model["zone"][:] = k
    return model


def rows_by(frame: pd.DataFrame, **keys) -> pd.Series:
    mask = np.ones(len(frame), dtype=bool)
    for name, value in keys.items():
        mask &= (frame[name] == value).to_numpy()
    assert mask.sum() == 1, (keys, int(mask.sum()))
    return frame[mask].iloc[0]


def test_hand_computed_3x3x3_by_rock():
    """V = 250 m3 per block.

    OX  9 blocks, one at 0.5:  volume 8 * 250 + 125 = 2125;  tonnes 2125 * 2 = 4250
        j=0: (125 + 250 + 250) * 2 = 1250 t @ 1 -> 1250 g
        j=1: 750 * 2 = 1500 t @ 2 -> 3000 g
        j=2: 750 * 2 = 1500 t @ 4 -> 6000 g
        metal 10250 g; grade 10250 / 4250 = 41 / 17
    SU  18 blocks, one at 0.2: volume 17 * 250 + 50 = 4300;  tonnes 4300 * 3 = 12900
        block (2,2,2) has no grade: 250 * 3 = 750 t unestimated; estimated 12150 t
        j=0: 6 * 250 * 3 = 4500 t @ 1 -> 4500 g
        j=1: (5 * 250 + 50) * 3 = 3900 t @ 2 -> 7800 g
        j=2: 5 * 250 * 3 = 3750 t @ 4 -> 15000 g     (the sixth block is the NaN)
        metal 27300 g; grade 27300 / 12150 = 182 / 81
    all volume 6425; tonnes 17150; unestimated 750; metal 37550 g; grade 37550 / 16400
    """
    model = hand_model()
    table = report(model, group_by="rock", grade_attributes="au", density="density",
                   partial="part")  # fmt: skip
    assert list(table.columns) == [
        "rock", "row", "blocks", "volume", "tonnes", "volume_without_density",
        "au_grade", "au_metal_g", "au_metal_oz", "au_unestimated_tonnes",
    ]  # fmt: skip
    assert table["rock"].tolist() == ["OX", "SU", None] and table["row"].tolist() == [
        "group", "group", "total"]  # fmt: skip
    ox, su, total = (table.iloc[r] for r in range(3))
    assert (ox.blocks, su.blocks, total.blocks) == (9, 18, 27)
    assert (ox.volume, su.volume, total.volume) == (2125.0, 4300.0, 6425.0)
    assert (ox.tonnes, su.tonnes, total.tonnes) == (4250.0, 12900.0, 17150.0)
    assert (ox.au_unestimated_tonnes, su.au_unestimated_tonnes) == (0.0, 750.0)
    assert total.au_unestimated_tonnes == 750.0
    assert (ox.au_metal_g, su.au_metal_g, total.au_metal_g) == (10250.0, 27300.0, 37550.0)
    assert ox.au_grade == EXACT(41 / 17, rel=1e-12)
    assert su.au_grade == EXACT(182 / 81, rel=1e-12)  # NOT 27300 / 12900: NaN is not zero
    assert total.au_grade == EXACT(37550 / 16400, rel=1e-12)
    assert su.au_grade != pytest.approx(27300 / 12900, rel=1e-3)
    assert total.au_metal_oz == EXACT(37550 / 31.1034768, rel=1e-12)
    assert (table["volume_without_density"] == 0.0).all()
    # metal = estimated tonnes x grade
    estimated = table["tonnes"] - table["au_unestimated_tonnes"]
    assert np.allclose(estimated * table["au_grade"], table["au_metal_g"], rtol=1e-12)
    assert table.dtypes["blocks"] == np.int64 and table.dtypes["tonnes"] == np.float64


def test_hand_computed_3x3x3_by_bench_and_rock():
    """Benches of 10 m: lower [0, 10) holds k = 0, 1; upper [10, 20) holds k = 2.

    lower OX  6 blocks: volume 5 * 250 + 125 = 1375; t 2750
              metal (125 + 250) * 2 * 1 + 500 * 2 * 2 + 500 * 2 * 4 = 750 + 2000 + 4000 = 6750
    lower SU  12 blocks: volume 3000; t 9000; metal 3000 * (1 + 2 + 4) = 21000
    upper OX  3 blocks: volume 750; t 1500; metal 500 * (1 + 2 + 4) = 3500
    upper SU  6 blocks: volume 5 * 250 + 50 = 1300; t 3900, of which 750 unestimated
              metal 2*250*3 * 1 + (250 + 50)*3 * 2 + 250*3 * 4 = 1500 + 1800 + 3000 = 6300
              grade 6300 / 3150 = 2
    """
    model = hand_model()
    table = report(model, group_by=["rock"], grade_attributes=["au"], density="density",
                   partial="part", bench_height=10.0)  # fmt: skip
    assert list(table.columns[:4]) == ["rock", "bench_from", "bench_to", "row"]
    assert table["rock"].tolist() == ["OX", "OX", "SU", "SU", None]
    assert table["bench_from"].tolist()[:4] == [0.0, 10.0, 0.0, 10.0]
    assert table["bench_to"].tolist()[:4] == [10.0, 20.0, 10.0, 20.0]
    expected = {  # (rock, bench_from): blocks, volume, tonnes, unestimated, metal
        ("OX", 0.0): (6, 1375.0, 2750.0, 0.0, 6750.0),
        ("OX", 10.0): (3, 750.0, 1500.0, 0.0, 3500.0),
        ("SU", 0.0): (12, 3000.0, 9000.0, 0.0, 21000.0),
        ("SU", 10.0): (6, 1300.0, 3900.0, 750.0, 6300.0),
    }
    for (rock, bench), (blocks, volume, tonnes, unestimated, metal) in expected.items():
        row = rows_by(table, rock=rock, bench_from=bench)
        assert row.blocks == blocks and row.volume == EXACT(volume, rel=1e-12)
        assert row.tonnes == EXACT(tonnes, rel=1e-12)
        assert row.au_unestimated_tonnes == EXACT(unestimated, abs=1e-9)
        assert row.au_metal_g == EXACT(metal, rel=1e-12)
        assert row.au_grade == EXACT(metal / (tonnes - unestimated), rel=1e-12)
    assert rows_by(table, rock="SU", bench_from=10.0).au_grade == EXACT(2.0, rel=1e-12)
    total = table.iloc[-1]
    assert total.row == "total" and np.isnan(total.bench_from)
    assert (total.blocks, total.volume, total.tonnes) == (27, 6425.0, 17150.0)

    # benches only: lower 4375 m3, 11750 t, 27750 g; upper 2050 m3, 5400 t, 9800 g
    table = report(model, grade_attributes=["au"], density="density", partial="part",
                   bench_height=10.0)  # fmt: skip
    lower, upper = table.iloc[0], table.iloc[1]
    assert (lower.volume, lower.tonnes, lower.au_metal_g) == (4375.0, 11750.0, 27750.0)
    assert (upper.volume, upper.tonnes, upper.au_metal_g) == (2050.0, 5400.0, 9800.0)
    assert upper.au_grade == EXACT(9800 / 4650, rel=1e-12)
    # the group rows add up to the total row
    groups = table[table["row"] == "group"]
    for column in ("blocks", "volume", "tonnes", "au_metal_g", "au_unestimated_tonnes"):
        assert groups[column].sum() == EXACT(table.iloc[-1][column], rel=1e-12)


def test_bench_rule_is_by_centroid_with_floors_belonging_to_the_bench_above():
    model = hand_model()  # centroids at z = 2.5, 7.5, 12.5
    # floors at 2.5, 7.5, 12.5: every centroid is exactly on a floor -> the bench above it
    table = report(model, bench_height=5.0, bench_reference=2.5, totals=False)
    assert table["bench_from"].tolist() == [2.5, 7.5, 12.5]
    assert table["blocks"].tolist() == [9, 9, 9]
    # 4 m benches cut through the 5 m blocks: each block is counted whole where its centroid is
    table = report(model, bench_height=4.0, totals=False)
    assert table["bench_from"].tolist() == [0.0, 4.0, 12.0]  # 2.5 | 7.5 | 12.5
    assert table["volume"].tolist() == [2250.0, 2250.0, 2250.0]
    # a negative reference and elevations below it work the same way
    table = report(model, bench_height=10.0, bench_reference=-5.0, totals=False)
    assert table["bench_from"].tolist() == [-5.0, 5.0] and table["blocks"].tolist() == [9, 18]
    assert "counted whole in the bench that contains its centroid" in " ".join(table.attrs["notes"])


def test_hand_computed_3x3x3_grade_tonnage_table():
    """Cut-offs 1.5 and 3.0 on au (grade >= cut-off), by rock.

    >= 1.5 (j = 1, 2):  OX 6 blocks, 1500 m3, 3000 t, 3000 + 6000 = 9000 g, grade 3
                        SU 6 + 5 = 11 blocks, 1300 + 1250 = 2550 m3, 3900 + 3750 = 7650 t,
                           7800 + 15000 = 22800 g
                        all 17 blocks, 4050 m3, 10650 t, 31800 g
    >= 3.0 (j = 2):     OX 3 blocks, 750 m3, 1500 t, 6000 g
                        SU 5 blocks, 1250 m3, 3750 t, 15000 g
                        all 8 blocks, 2000 m3, 5250 t, 21000 g, grade 4
    the block without grade (750 t, SU) is in no row: unclassified
    """
    model = hand_model()
    table = report(model, group_by="rock", grade_attributes="au", density="density",
                   partial="part", cutoffs=[3.0, 1.5])  # fmt: skip
    assert table["rock"].tolist() == ["OX", "OX", "SU", "SU", None, None]
    assert table["cutoff"].tolist() == [1.5, 3.0, 1.5, 3.0, 1.5, 3.0]
    assert table["row"].tolist() == ["group"] * 4 + ["total"] * 2
    assert table["blocks"].tolist() == [6, 3, 11, 5, 17, 8]
    assert table["volume"].tolist() == [1500.0, 750.0, 2550.0, 1250.0, 4050.0, 2000.0]
    assert table["tonnes"].tolist() == [3000.0, 1500.0, 7650.0, 3750.0, 10650.0, 5250.0]
    assert table["au_metal_g"].tolist() == [9000.0, 6000.0, 22800.0, 15000.0, 31800.0, 21000.0]
    assert table["au_grade"].tolist() == EXACT(
        [3.0, 4.0, 22800 / 7650, 4.0, 31800 / 10650, 4.0], rel=1e-12)  # fmt: skip
    assert table["unclassified_tonnes"].tolist() == [0.0, 0.0, 750.0, 750.0, 750.0, 750.0]
    assert (table["au_unestimated_tonnes"] == 0.0).all()  # every block in a row has a grade
    # a cut-off above every grade gives an empty row, not an error; grade is NaN, metal 0
    table = report(model, grade_attributes="au", density="density", cutoffs=[0.0, 100.0])
    assert table["blocks"].tolist() == [26, 0] and table["tonnes"].tolist()[1] == 0.0
    assert np.isnan(table["au_grade"][1]) and table["au_metal_g"][1] == 0.0
    # cut-offs on one grade, reporting another attribute
    model.add_attribute("ag")
    model["ag"][:] = 10.0
    table = report(model, grade_attributes=["ag"], density=1.0, cutoffs=[3.0],
                   cutoff_attribute="au")  # fmt: skip
    assert table["blocks"].tolist() == [8] and table["ag_grade"].tolist() == [10.0]
    assert "ag_metal" in table.columns


def test_constraint_mask_and_integer_groups():
    model = hand_model()
    su = AttributeConstraint('rock == "SU"')
    table = report(model, constraint=su, group_by="zone", density=3.0)
    assert table["zone"].tolist() == [0, 1, 2, None] and table["blocks"].tolist() == [6, 6, 6, 18]
    assert table["tonnes"].tolist() == [4500.0, 4500.0, 4500.0, 13500.0]
    upper = model.centroids()[:, 2] > 10.0
    table = report(model, constraint=su, where=upper, group_by=["rock", "zone"])
    assert list(table.columns) == ["rock", "zone", "row", "blocks", "volume"]  # no density
    assert table["blocks"].tolist() == [6, 6] and table["volume"].tolist() == [1500.0, 1500.0]
    # nothing selected: a total row of zeros
    table = report(model, where=np.zeros(model.n_blocks, bool), group_by="rock", density=2.0,
                   grade_attributes="au")  # fmt: skip
    assert len(table) == 1 and table["row"][0] == "total" and table["tonnes"][0] == 0.0
    assert np.isnan(table["au_grade"][0]) and table["au_metal_g"][0] == 0.0
    # a missing category is its own group
    model["rock"][:3] = -1
    table = report(model, group_by="rock", totals=False)
    assert table["rock"].tolist() == [None, "OX", "SU"] and table["blocks"].tolist() == [3, 6, 18]


def test_nan_grades_are_never_zero():
    model = hand_model()
    model["au"][:] = np.nan
    table = report(model, group_by="rock", grade_attributes="au", density="density")
    assert np.isnan(table["au_grade"]).all()
    assert np.isnan(table["au_metal_g"]).all()  # unknown, not 0
    assert table["au_unestimated_tonnes"].tolist() == table["tonnes"].tolist()
    model["au"][0] = 2.0  # one OX block of 250 m3 * 2.0 = 500 t
    table = report(model, group_by="rock", grade_attributes="au", density="density")
    assert table["au_grade"][0] == 2.0 and table["au_metal_g"][0] == 1000.0
    assert table["au_unestimated_tonnes"][0] == 4500.0 - 500.0
    assert np.isnan(table["au_grade"][1]) and table["au_grade"][2] == 2.0
    # density missing: no tonnes for those blocks, their volume is reported
    model["density"][:9] = np.nan  # all of OX
    table = report(model, group_by="rock", grade_attributes="au", density="density")
    assert table["tonnes"].tolist() == [0.0, 13500.0, 13500.0]
    assert table["volume_without_density"].tolist() == [2250.0, 0.0, 2250.0]
    assert table["volume"].tolist() == [2250.0, 4500.0, 6750.0]


def test_units():
    model = hand_model()
    model.add_attribute("cu", unit="%")
    model["cu"][:] = 0.5
    model.add_attribute("x1")  # no unit
    model["x1"][:] = 3.0
    table = report(model, grade_attributes=["au", "cu", "x1"], density=2.0)
    tonnes = 27 * 250 * 2.0
    assert table.attrs["units"] == {"au": "g/t", "cu": "%", "x1": ""}
    assert table["cu_metal_t"][0] == EXACT(tonnes * 0.5 / 100, rel=1e-12)  # tonnes of copper
    assert table["x1_metal"][0] == EXACT(tonnes * 3.0, rel=1e-12)
    grams = table["au_metal_g"][0]
    assert table["au_metal_oz"][0] == EXACT(grams / 31.1034768, rel=1e-14)
    assert TROY_OUNCE_G == 31.1034768
    # 1 t at 31.1034768 g/t holds exactly one troy ounce
    one = BlockModel(BlockModelDefinition((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (1, 1, 1)))
    one.add_attribute("au")
    one["au"][:] = 31.1034768
    table = report(one, grade_attributes="au", density=1.0, units={"au": "ppm"})
    assert table["au_metal_g"][0] == 31.1034768 and table["au_metal_oz"][0] == 1.0
    # overriding the attribute's unit; "" switches the conversion off
    table = report(model, grade_attributes="au", density=2.0, units={"au": "%"})
    assert "au_metal_t" in table.columns and "au_metal_g" not in table.columns
    table = report(model, grade_attributes="au", density=2.0, units={"au": ""})
    assert "au_metal" in table.columns
    with pytest.raises(ReportError, match="not one of g/t, ppm, %"):
        report(model, grade_attributes="au", density=2.0, units={"au": "oz/st"})


def test_errors():
    model = hand_model()
    with pytest.raises(ReportError, match="give the density"):
        report(model, grade_attributes="au")
    with pytest.raises(KeyError, match="no attribute 'cu'"):
        report(model, grade_attributes="cu", density=1.0)
    with pytest.raises(ReportError, match="cannot group by the float attribute"):
        report(model, group_by="au")
    with pytest.raises(ReportError, match="must be a float attribute"):
        report(model, grade_attributes="zone", density=1.0)
    with pytest.raises(ReportError, match="density must be positive"):
        report(model, density=-1.0)
    with pytest.raises(ReportError, match="bench_height must be"):
        report(model, bench_height=0.0)
    with pytest.raises(ReportError, match="cut-offs need a grade"):
        report(model, cutoffs=[1.0], density=1.0)
    with pytest.raises(ReportError, match="without cutoffs"):
        report(model, cutoff_attribute="au")
    model["part"][3] = np.nan
    model["part"][4] = 1.5
    with pytest.raises(ReportError, match=r"'part': 2 selected block\(s\).*\(1 missing\)"):
        report(model, partial="part")
    # ... unless those blocks are not selected
    keep = np.ones(model.n_blocks, bool)
    keep[3:5] = False
    assert report(model, partial="part", where=keep)["blocks"][0] == 25


# ---------------------------------------------------------------- sub-blocks


@st.composite
def split_plans(draw):
    return draw(st.lists(st.integers(0, 2**32 - 1), min_size=1, max_size=4))


def full_report(model: BlockModel) -> pd.DataFrame:
    # benches aligned with the parent blocks (5 m from z = 0), so no sub-block changes bench
    return report(model, group_by=["rock", "zone"], grade_attributes=["au", "cu"],
                  density="density", partial="part", cutoffs=[0.5, 1.5, 3.0], bench_height=5.0,
                  units={"cu": "%"})  # fmt: skip


@settings(max_examples=40, deadline=None)
@given(split_plans(), st.sampled_from([(1, 1, 1), (2, 2, 1), (2, 1, 0), (0, 0, 2)]))
def test_report_is_unchanged_by_splitting_blocks(seeds, depth):
    definition = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 5.0, 5.0), (3, 3, 3),
                                      max_depth=depth)  # fmt: skip
    model = BlockModel(definition)
    rng = np.random.default_rng(12345)
    n = model.n_blocks
    model.add_attribute("rock", "category", categories=["OX", "SU"])
    model["rock"][:] = rng.integers(-1, 2, n)
    model.add_attribute("zone", "int")
    model["zone"][:] = rng.integers(0, 3, n)
    model.add_attribute("density")
    model["density"][:] = rng.uniform(2.0, 3.0, n)
    model["density"][rng.random(n) < 0.1] = np.nan
    model.add_attribute("part")
    model["part"][:] = rng.random(n)
    for name in ("au", "cu"):
        model.add_attribute(name)
        model[name][:] = rng.lognormal(size=n)
        model[name][rng.random(n) < 0.2] = np.nan
    before = full_report(model)
    for seed in seeds:
        split_rng = np.random.default_rng(seed)
        can = model.level < model.max_level
        model.split(can & (split_rng.random(model.n_blocks) < 0.4))
    after = full_report(model)
    assert model.n_blocks >= n
    keys = ["rock", "zone", "bench_from", "bench_to", "cutoff", "row"]
    assert before[keys].astype(str).equals(after[keys].astype(str))
    for column in before.columns:
        if column in keys or column == "blocks":
            continue
        assert np.allclose(before[column], after[column], rtol=1e-11, atol=1e-9,
                           equal_nan=True), column  # fmt: skip
    assert (after["blocks"] >= before["blocks"]).all()
    # volumes are exact: every block volume is a power-of-two fraction of 250 m3
    plain_before = report(model, group_by="rock")
    model.split(model.level < model.max_level)
    assert report(model, group_by="rock")["volume"].tolist() == plain_before["volume"].tolist()


# --------------------------------------------------------------- the truth grid


def truth_model() -> BlockModel:
    frame = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    definition = BlockModelDefinition((500400.0, 5300250.0, 825.0), (10.0, 10.0, 5.0),
                                      (39, 52, 84))  # fmt: skip
    model = BlockModel(definition)
    flat = definition.flat_index(frame[["i", "j", "k"]].to_numpy())
    for name in ("au", "au_ore", "density", "ore_fraction"):
        model.add_attribute(name, unit="g/t" if name.startswith("au") else "")
        model[name][flat] = frame[name].to_numpy()
    model.add_attribute("rock", "category")
    model.set("rock", frame["rock"].to_numpy(dtype=object), flat)
    return model


def test_ore_report_of_the_truth_grid_against_truth_json():
    """Ore = blocks weighted by ore_fraction, density 2.85, grade au_ore.

    Against an independent pandas computation on the same table the report is exact
    (1e-12). Against truth.json it is NOT exact, for reasons that lie in the data:

    * tonnes / volume: -2.96e-5 relative. The block table holds the ore fraction on a
      5 x 5 x 5 point lattice per block (truth.json blocks.ore_volume_relative_error says
      the same -2.955e-5); truth.json totals use the exact analytic volume of the solid.
    * grade: -1.14 %, metal -1.15 %. truth.json's mean grade is the average over the
      simulation lattice NODES; the block grades are averages of the field interpolated
      trilinearly between nodes at the 125 points of each block. Interpolation smooths
      the Gaussian variable, and the mean of exp(mu + sigma Y) falls with the variance
      of Y, so the block table is systematically about 1 % lower. The report reproduces
      the table; it cannot recover what the table does not hold.
    """
    truth = json.loads((SYNTH / "truth.json").read_text())["totals"]
    frame = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    model = truth_model()
    table = report(model, density=2.85, partial="ore_fraction", grade_attributes=["au_ore"])
    assert len(table) == 1
    row = table.iloc[0]

    # independent arithmetic on the raw table
    volume = (frame.dx * frame.dy * frame.dz * frame.ore_fraction).to_numpy()
    tonnes = volume * 2.85
    has = frame.au_ore.notna().to_numpy()
    metal = float((tonnes[has] * frame.au_ore.to_numpy()[has]).sum())
    assert row.volume == pytest.approx(volume.sum(), rel=1e-12)
    assert row.tonnes == pytest.approx(tonnes.sum(), rel=1e-12)
    assert row.au_ore_metal_g == pytest.approx(metal, rel=1e-12)
    assert row.au_ore_grade == pytest.approx(metal / tonnes[has].sum(), rel=1e-12)
    assert row.au_ore_unestimated_tonnes == 0.0  # blocks without au_ore hold no ore

    # truth.json: 5 857 904 t at 2.010 g/t, 11 774 712 g
    assert row.volume == pytest.approx(truth["ore_volume_m3"], rel=5e-5)
    assert row.tonnes == pytest.approx(truth["ore_tonnes"], rel=5e-5)
    assert row.tonnes / truth["ore_tonnes"] - 1 == pytest.approx(-2.955e-5, abs=1e-7)
    assert row.au_ore_grade == pytest.approx(truth["ore_mean_grade_g_t"], rel=0.015)
    assert row.au_ore_metal_g == pytest.approx(truth["ore_metal_g"], rel=0.015)
    assert row.au_ore_metal_oz == pytest.approx(truth["ore_metal_troy_oz"], rel=0.015)
    assert row.au_ore_grade < truth["ore_mean_grade_g_t"]  # the known direction of the bias
    # the conversion itself agrees with truth.json's own grams -> ounces to rounding
    assert truth["ore_metal_g"] / TROY_OUNCE_G == pytest.approx(truth["ore_metal_troy_oz"],
                                                                rel=1e-12)  # fmt: skip


def test_grade_tonnage_table_is_monotone():
    model = truth_model()
    cutoffs = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
    table = report(model, constraint=AttributeConstraint('rock == "ORE"'), density="density",
                   grade_attributes="au", cutoffs=cutoffs)  # fmt: skip
    assert table["cutoff"].tolist() == cutoffs
    for column in ("blocks", "volume", "tonnes", "au_metal_g", "au_metal_oz"):
        assert (np.diff(table[column]) <= 0).all(), column  # less material above a higher cut
    grade = table["au_grade"].to_numpy()
    assert (np.diff(grade[~np.isnan(grade)]) >= 0).all()  # and richer
    assert (grade[~np.isnan(grade)] >= np.array(cutoffs)[~np.isnan(grade)]).all()
    assert table["tonnes"][0] > table["tonnes"].iloc[-1] > 0
    assert table["unclassified_tonnes"].nunique() == 1


# --------------------------------------------------------------------- output


def test_to_csv_and_text_table(tmp_path):
    model = hand_model()
    table = report(model, group_by="rock", grade_attributes="au", density="density",
                   partial="part", bench_height=10.0)  # fmt: skip
    path = to_csv(table, tmp_path / "out" / "report.csv")
    back = pd.read_csv(path)
    assert list(back.columns) == list(table.columns) and len(back) == len(table)
    assert back["tonnes"].tolist() == table["tonnes"].tolist()
    assert back["au_grade"].tolist() == table["au_grade"].tolist()  # full precision
    assert path.read_text().splitlines()[-1].startswith(",,,total,27,6425.0,17150.0")

    text = format_table(table)
    lines = text.splitlines()
    assert lines[0].split() == [c for c in table.columns if c != "row"]
    assert set(lines[1]) == {"-", " "}
    assert lines[2].split() == ["OX", "0.0", "10.0", "6", "1,375", "2,750", "0", "2.455",
                                "6,750", "217", "0"]  # fmt: skip
    assert lines[-1].split()[:6] == ["TOTAL", "-", "-", "27", "6,425", "17,150"]
    assert len({len(line) for line in lines[:2]}) == 1
    text = format_table(table, decimals={"grade": 1, "quantity": 2, "au_metal_oz": 4})
    assert text.splitlines()[2].split()[4:10] == ["1,375.00", "2,750.00", "0.00", "2.5",
                                                  "6,750.00", "217.0175"]  # fmt: skip
    model["au"][:] = np.nan
    text = format_table(report(model, grade_attributes="au", density=1.0))
    assert text.splitlines()[2].split() == ["27", "6,750", "6,750", "0", "-", "-", "-", "6,750"]
