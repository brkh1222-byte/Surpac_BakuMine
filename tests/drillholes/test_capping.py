"""Grade capping (P1.T6): caps, per-domain caps and the report numbers, worked by hand."""

import numpy as np
import pandas as pd
import pytest

from bakumine.core.drillholes.capping import REPORT_COLUMNS, cap_grades
from bakumine.core.drillholes.composite import composite

NAN = float("nan")


def samples() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "hole_id": ["A"] * 6,
            "depth_from": [0.0, 1.0, 3.0, 4.0, 6.0, 7.0],
            "depth_to": [1.0, 3.0, 4.0, 6.0, 7.0, 10.0],  # lengths 1, 2, 1, 2, 1, 3
            "au": [1.0, 50.0, NAN, 8.0, 300.0, 2.0],
            "cu": [0.5, 0.6, 0.7, 3.0, 0.2, 0.1],
            "rock": ["ORE", "ORE", "ORE", "WASTE", "WASTE", None],
        }
    )


def test_single_cap_and_report_numbers():
    table = samples()
    before = table.copy()
    capped, report = cap_grades(table, {"au": 20.0})
    pd.testing.assert_frame_equal(table, before)  # the input is not changed
    np.testing.assert_array_equal(capped["au"], [1.0, 20.0, NAN, 8.0, 20.0, 2.0])
    np.testing.assert_array_equal(capped["au_uncut"], before["au"])
    assert list(capped.columns) == [
        "hole_id", "depth_from", "depth_to", "au", "au_uncut", "cu", "rock",
    ]
    assert list(report.columns) == list(REPORT_COLUMNS)
    assert len(report) == 1
    row = report.iloc[0]
    # Weights are the interval lengths 1, 2, (NaN sample skipped), 2, 1, 3 = 9 m.
    # metal uncut = 1*1 + 50*2 + 8*2 + 300*1 + 2*3 = 423
    # metal cut   = 1*1 + 20*2 + 8*2 +  20*1 + 2*3 = 83
    assert row["attribute"] == "au"
    assert row["domain"] is None
    assert row["cap"] == 20.0
    assert row["n_values"] == 5
    assert row["n_cut"] == 2
    assert row["max_uncut"] == 300.0
    assert row["metal_uncut"] == pytest.approx(423.0)
    assert row["metal_cut"] == pytest.approx(83.0)
    assert row["metal_removed"] == pytest.approx(340.0)
    assert row["metal_removed_pct"] == pytest.approx(100.0 * 340.0 / 423.0)
    assert row["mean_uncut"] == pytest.approx(423.0 / 9.0)
    assert row["mean_cut"] == pytest.approx(83.0 / 9.0)


def test_value_equal_to_cap_is_not_cut_and_several_attributes():
    capped, report = cap_grades(samples(), {"au": 50.0, "cu": 1.0})
    np.testing.assert_array_equal(capped["au"], [1.0, 50.0, NAN, 8.0, 50.0, 2.0])
    np.testing.assert_array_equal(capped["cu"], [0.5, 0.6, 0.7, 1.0, 0.2, 0.1])
    assert report["attribute"].tolist() == ["au", "cu"]
    assert report["n_cut"].tolist() == [1, 1]  # 50 itself is not above the cap
    # cu: only the 2 m at 3.0 is cut to 1.0 -> removed 2 * 2 = 4
    assert report["metal_removed"].tolist() == pytest.approx([250.0, 4.0])
    assert {"au_uncut", "cu_uncut"} <= set(capped.columns)


def test_per_domain_caps():
    capped, report = cap_grades(samples(), {"au": {"ORE": 30.0, "WASTE": 10.0}}, domain="rock")
    # ORE rows 0-2 capped at 30; WASTE rows 3-4 at 10; the row without a domain is not capped.
    np.testing.assert_array_equal(capped["au"], [1.0, 30.0, NAN, 8.0, 10.0, 2.0])
    assert report["domain"].tolist() == ["ORE", "WASTE"]
    assert report["cap"].tolist() == [30.0, 10.0]
    assert report["n_values"].tolist() == [2, 2]
    assert report["n_cut"].tolist() == [1, 1]
    # ORE: (50 - 30) * 2 m = 40 of 1 + 100 = 101.  WASTE: (300 - 10) * 1 m = 290 of 16 + 300.
    assert report["metal_uncut"].tolist() == pytest.approx([101.0, 316.0])
    assert report["metal_removed"].tolist() == pytest.approx([40.0, 290.0])


def test_per_domain_default_cap_covers_unlisted_domains():
    table = samples()
    table.loc[5, "au"] = 99.0
    capped, report = cap_grades(table, {"au": {"ORE": 30.0, "*": 5.0}}, domain="rock")
    np.testing.assert_array_equal(capped["au"], [1.0, 30.0, NAN, 5.0, 5.0, 5.0])
    assert report["domain"].tolist() == ["ORE", "*"]
    assert report["n_cut"].tolist() == [1, 3]
    # others: (8-5)*2 + (300-5)*1 + (99-5)*3 = 6 + 295 + 282 = 583
    assert report["metal_removed"].iloc[1] == pytest.approx(583.0)
    # A listed domain that does not occur is reported with nothing cut.
    _, report = cap_grades(table, {"au": {"DYKE": 1.0}}, domain="rock")
    assert report[["n_values", "n_cut", "metal_removed"]].iloc[0].tolist() == [0, 0, 0.0]
    assert np.isnan(report["max_uncut"].iloc[0])


def test_recapping_starts_from_the_uncut_values():
    once, _ = cap_grades(samples(), {"au": 20.0})
    twice, report = cap_grades(once, {"au": 100.0})
    np.testing.assert_array_equal(twice["au"], [1.0, 50.0, NAN, 8.0, 100.0, 2.0])
    np.testing.assert_array_equal(twice["au_uncut"], samples()["au"])
    assert report["n_cut"].iloc[0] == 1
    assert report["metal_removed"].iloc[0] == pytest.approx(200.0)
    other, _ = cap_grades(samples(), {"au": 20.0}, suffix="_raw")
    assert "au_raw" in other.columns


def test_weights_explicit_column_and_unweighted_table():
    table = samples().assign(tonnes=[10.0, 10.0, 10.0, 10.0, NAN, 10.0])
    _, report = cap_grades(table, {"au": 20.0}, weight="tonnes")
    # The 300 g/t row has no weight: metal removed = (50 - 20) * 10.
    assert report["metal_removed"].iloc[0] == pytest.approx(300.0)
    assert report["n_cut"].iloc[0] == 2
    points = pd.DataFrame({"au": [1.0, 5.0, 9.0]})
    capped, report = cap_grades(points, {"au": 4.0})
    assert capped["au"].tolist() == [1.0, 4.0, 4.0]
    assert report["metal_removed"].iloc[0] == pytest.approx(6.0)  # weight 1 each


def test_works_on_composites_with_coverage_weights():
    table = samples()
    comps = composite(table, length=5.0, attributes=["au"])
    # 0-5 : (1*1 + 50*2 + 8*1) / 4 = 27.25 on 4 m (NaN metre skipped)
    # 5-10: (8*1 + 300*1 + 2*3) / 5 = 62.8 on 5 m
    np.testing.assert_allclose(comps["au"], [27.25, 62.8])
    capped, report = cap_grades(comps, {"au": 40.0})
    np.testing.assert_allclose(capped["au"], [27.25, 40.0])
    np.testing.assert_allclose(capped["au_uncut"], [27.25, 62.8])
    row = report.iloc[0]
    assert row["n_cut"] == 1
    # coverage weights 4 and 5: removed (62.8 - 40) * 5 = 114 of 27.25*4 + 62.8*5 = 423
    assert row["metal_uncut"] == pytest.approx(423.0)
    assert row["metal_removed"] == pytest.approx(114.0)
    # Capping the samples first and compositing both columns keeps the uncut grade alongside.
    capped_samples, _ = cap_grades(table, {"au": 20.0})
    both = composite(capped_samples, length=5.0, attributes=["au", "au_uncut"])
    np.testing.assert_allclose(both["au_uncut"], [27.25, 62.8])
    # 0-5: (1 + 20*2 + 8) / 4 = 12.25;  5-10: (8 + 20 + 6) / 5 = 6.8
    np.testing.assert_allclose(both["au"], [12.25, 6.8])


def test_errors():
    with pytest.raises(ValueError, match="not in the table"):
        cap_grades(samples(), {"ag": 1.0})
    with pytest.raises(ValueError, match="not numeric"):
        cap_grades(samples(), {"rock": 1.0})
    with pytest.raises(ValueError, match="need the domain column"):
        cap_grades(samples(), {"au": {"ORE": 1.0}})
    with pytest.raises(ValueError, match="domain column"):
        cap_grades(samples(), {"au": {"ORE": 1.0}}, domain="zone")
    with pytest.raises(ValueError, match="finite"):
        cap_grades(samples(), {"au": NAN})
    with pytest.raises(ValueError, match="must be a number"):
        cap_grades(samples(), {"au": "high"})
    with pytest.raises(ValueError, match="weight column"):
        cap_grades(samples(), {"au": 1.0}, weight="tonnes")
