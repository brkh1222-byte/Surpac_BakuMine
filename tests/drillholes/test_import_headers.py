"""Column recognition for tables that come out of reports (PDF appendices): element
headers with units, and the Mongolian headers of a drilling table. Synthetic values."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.core.drillholes.importer import (
    element_attribute,
    load_database,
    suggest_column_map,
)
from bakumine.core.drillholes.recipe import make_recipe, run_recipe


@pytest.mark.parametrize(
    ("header", "expected"),
    [
        ("Au (г/т)", ("au", "g/t")),
        ("Au /г/т/", ("au", "g/t")),
        ("Au, g/t", ("au", "g/t")),
        ("Au-г.т", ("au", "g/t")),
        ("дүн Au- г.т", ("au", "g/t")),
        ("Ag (г/т)", ("ag", "g/t")),
        ("FAA303 Au г/т", ("au", "g/t")),
        ("Шинжилгээний үр дүн Au", ("au", None)),
        ("Element, mg/kg Au", ("au", "mg/kg")),
        ("Element, мг/кг Pb", ("pb", "mg/kg")),
        ("Cu, %", ("cu", "%")),
        ("Mo ppm", ("mo", "ppm")),
        ("As_ppm", ("as", "ppm")),
        ("AU_PPB", ("au", "ppb")),
        ("zn_pct", ("zn", "%")),
        ("W, ppm", ("w", "ppm")),
        ("Содержание Sb, г/т", ("sb", "g/t")),
        ("Au", ("au", None)),
        ("U", ("u", None)),
        # not elements
        ("Recovery, %", None),
        ("Fe2O3 %", None),
        ("Au Ag", None),
        ("in %", None),
        ("No", None),
        ("Sample No", None),
        ("Length, m", None),
        ("result S", None),
        ("Lab", None),
    ],
)
def test_element_headers(header, expected):
    assert element_attribute(header) == expected


def test_every_element_symbol_is_recognised_with_a_unit():
    from bakumine.core.drillholes.importer import ELEMENT_SYMBOLS

    assert len(ELEMENT_SYMBOLS) == 92
    for symbol in ELEMENT_SYMBOLS:
        assert element_attribute(f"{symbol} (г/т)") == (symbol.lower(), "g/t")
        assert element_attribute(f"{symbol}, ppm") == (symbol.lower(), "ppm")


def test_assay_headers_of_a_report_become_elements_with_units():
    headers = ["Д/д", "Цооног №", "Сорьцы н төрөл", "Сорьцын дугаар", "интервал -аас",
               "- хүртэл", "Сорьцы н урт /м/", "дүн Au- г.т", "дүн Ag- г.т", "Cu, %",
               "Тайлбар"]  # fmt: skip
    found = suggest_column_map(headers, "interval")
    assert found.missing == []
    assert found.mapping == {
        "Д/д": "д_д", "Цооног №": "hole_id", "Сорьцы н төрөл": "sample_type",
        "Сорьцын дугаар": "sample_id", "интервал -аас": "depth_from", "- хүртэл": "depth_to",
        "Сорьцы н урт /м/": "sample_length", "дүн Au- г.т": "au", "дүн Ag- г.т": "ag",
        "Cu, %": "cu", "Тайлбар": "тайлбар",
    }  # fmt: skip
    assert found.units == {"au": "g/t", "ag": "g/t", "cu": "%"}
    # two columns of one element keep both, numbered
    twice = suggest_column_map(["Hole", "from", "to", "Au (г/т)", "Au /г/т/"], "interval")
    assert [twice.mapping[h] for h in ("Au (г/т)", "Au /г/т/")] == ["au", "au_2"]
    assert twice.units == {"au": "g/t", "au_2": "g/t"}


def test_mongolian_drilling_table_headers():
    headers = ["№", "Цооногийн дугаар", "X", "Y", "Өндөр /м/", "Азимут /градус/",
               "Налуугийн өнцөг /градус/", "Гүн /т.м/"]  # fmt: skip
    found = suggest_column_map(headers, "collar")
    assert found.missing == [] and found.unmatched == ["№"]
    assert found.mapping == {
        "Цооногийн дугаар": "hole_id", "X": "x", "Y": "y", "Өндөр /м/": "z",
        "Азимут /градус/": "azimuth", "Налуугийн өнцөг /градус/": "dip",
        "Гүн /т.м/": "max_depth",
    }  # fmt: skip
    # the group header flattened in front of the axis, Cyrillic letters for X and Y
    grouped = suggest_column_map(
        ["Цооногийн дугаар", "Солбицол /UTM/ Х", "Солбицол /UTM/ У", "Солбицол /UTM/ H", "Гүн"],
        "collar",
    )
    assert list(grouped.mapping.values()) == ["hole_id", "x", "y", "z", "max_depth"]
    # geodetic B / L (latitude / longitude) only as a pair
    both = suggest_column_map(["Цооногийн дугаар", "X", "Y", "B° ′ ″", "L° ′ ″", "Өндөр"], "collar")
    assert (both.mapping["B° ′ ″"], both.mapping["L° ′ ″"]) == ("lat", "lon")
    alone = suggest_column_map(["Hole", "X", "Y", "Z", "L"], "collar")
    assert "L" not in alone.mapping and alone.unmatched == ["L"]


@pytest.fixture
def report_tables(tmp_path):
    """A drilling table as a Mongolian report gives it (X = northing) and two assay
    tables whose gold column is spelled differently."""
    rng = np.random.default_rng(5)
    holes = [f"SYN-{i:02d}" for i in range(1, 9)]
    east = 512000 + rng.uniform(0, 500, len(holes)).round(3)
    north = 5310000 + rng.uniform(0, 500, len(holes)).round(3)
    collar = pd.DataFrame(
        {
            "№": range(1, len(holes) + 1),
            "Цооногийн дугаар": holes,
            "Солбицол /UTM/ X": north,
            "Солбицол /UTM/ Y": east,
            "Өндөр /м/": 1200 + rng.uniform(0, 30, len(holes)).round(2),
            "Азимут /градус/": 30,
            "Налуугийн өнцөг /градус/": -60,
            "Гүн /т.м/": 20.0,
        }
    )
    collar.to_csv(tmp_path / "drilling.csv", index=False)

    def assays(names, gold, silver):
        rows = [
            {"Цооног №": hole, "интервал -аас": float(k), "-хүртэл": float(k + 1),
             gold: round(float(rng.lognormal(-1, 1)), 2),
             silver: round(float(rng.uniform(1, 9)), 1)}
            for hole in names for k in range(20)
        ]  # fmt: skip
        return pd.DataFrame(rows)

    first = assays(holes[:4], "Au-г.т", "Ag-г.т")
    second = assays(holes[4:], "дүн Au- г.т", "Ag (ppm)")
    first.to_csv(tmp_path / "assay_a.csv", index=False)
    second.to_csv(tmp_path / "assay_b.csv", index=False)
    return tmp_path, collar, east, north, first, second


def test_report_tables_import_without_a_column_map(report_tables, capsys):
    folder, collar, east, north, first, second = report_tables
    out = folder / "db"
    code = main(
        ["import", "--collar", str(folder / "drilling.csv"), "--assay", str(folder / "assay_a.csv"),
         "--assay", str(folder / "assay_b.csv"), "--auto", "--coordinate-order", "auto",
         "--out", str(out)]
    )  # fmt: skip
    printed = capsys.readouterr().out
    assert code == 0
    db = load_database(out)
    # X of the report is the northing: decided from the values, and said
    assert db.metadata.coordinate_order == "yxz"
    np.testing.assert_allclose(db.collar["x"], east)
    np.testing.assert_allclose(db.collar["y"], north)
    assert "coordinate order yxz" in printed
    assert any(n["kind"] == "coordinate_order_assumed_yxz" for n in db.metadata.notes)
    np.testing.assert_allclose(db.collar["max_depth"], 20.0)
    assert len(db.survey) == len(collar) and set(db.survey["dip"]) == {-60.0}
    # both spellings of the gold column are one attribute
    assay = db.intervals["assay"]
    assert list(assay.columns) == ["hole_id", "depth_from", "depth_to", "au", "ag"]
    gold = np.r_[first["Au-г.т"].to_numpy(), second["дүн Au- г.т"].to_numpy()]
    np.testing.assert_allclose(assay["au"], gold)
    assert not assay["au"].isna().any()
    assert db.metadata.reports["units"] == {"assay": {"au": "g/t", "ag": "g/t"}}
    assert "au [float64, g/t]" in printed
    # silver is g/t in one file and ppm in the other: reported, not converted
    differ = [n for n in db.metadata.notes if n["kind"] == "units_differ_between_sources"]
    assert [(n["column"], n["values"]) for n in differ] == [
        ("ag", {"assay_a.csv": "g/t", "assay_b.csv": "ppm"})
    ]


def test_coordinate_order_stays_as_mapped_when_asked_or_undecidable(report_tables, capsys):
    folder, collar, east, north, *_ = report_tables
    recipe = make_recipe(
        {"collar": [{"file": "drilling.csv"}]}, {"coordinate_order": "xyz"}
    )
    db = run_recipe(recipe, folder)
    np.testing.assert_allclose(db.collar["x"], north)  # as mapped, with a warning note
    assert any(n["kind"] == "x_looks_like_northing" for n in db.metadata.notes)
    code = main(["import", "--collar", str(folder / "drilling.csv"), "--auto", "--dry-run"])
    assert code == 0
    assert "WARNING: the collar column mapped to x looks like northings" in capsys.readouterr().out
    # feet: the magnitudes of UTM metres mean nothing
    feet = run_recipe(
        make_recipe({"collar": [{"file": "drilling.csv"}]},
                    {"coordinate_order": "auto", "length_unit": "ft"}),
        folder,
    )  # fmt: skip
    assert feet.metadata.coordinate_order == "xyz"
    # a local grid tells nothing: kept as mapped
    local = collar.assign(
        **{"Солбицол /UTM/ X": east - 512000, "Солбицол /UTM/ Y": north - 5310000}
    )
    local.to_csv(folder / "local.csv", index=False)
    db = run_recipe(make_recipe({"collar": [{"file": "local.csv"}]}, {"coordinate_order": "auto"}),
                    folder)  # fmt: skip
    assert db.metadata.coordinate_order == "xyz"
    np.testing.assert_allclose(db.collar["x"], east - 512000)


def test_element_named_like_a_derived_column_gets_another_name():
    """Yttrium (``Y_ppm``) as ``y`` would clash with the y that desurvey and compositing add."""
    headers = ["HoleID", "From", "To", "Au_ppm", "Y_ppm", "Y", "Zr_ppm"]
    found = suggest_column_map(headers, "interval")
    assert found.mapping["Y_ppm"] == "y_ppm"
    assert found.mapping["Y"] == "y_2"
    assert found.mapping["Au_ppm"] == "au" and found.mapping["Zr_ppm"] == "zr"
    assert found.units["y_ppm"] == "ppm"
    names = set(found.mapping.values())
    assert not names & {"length", "depth_mid", "x", "y", "z", "dx", "dy", "dz", "domain"}


def test_multi_element_assays_with_yttrium_composite_with_every_attribute(tmp_path, capsys):
    """A multi-element sheet (Au ... Y ... Zr) imports, desurveys and composites without
    naming the attributes; the yttrium values stay with their samples."""
    collar = pd.DataFrame({"HoleID": ["DH-01", "DH-02"], "X": [1000.0, 1050.0],
                           "Y": [2000.0, 2000.0], "Elev": [500.0, 502.0], "Depth": [10.0, 10.0],
                           "Azimuth": [0.0, 0.0], "Dip": [-90.0, -90.0]})  # fmt: skip
    rows = [(h, f, f + 2.0) for h in ("DH-01", "DH-02") for f in np.arange(0.0, 10.0, 2.0)]
    assay = pd.DataFrame(rows, columns=["HoleID", "From", "To"])
    assay["Au_ppm"] = np.linspace(0.1, 1.0, len(assay))
    assay["Y_ppm"] = np.arange(len(assay), dtype=float) + 10.0
    assay["Zr_ppm"] = 100.0
    collar.to_csv(tmp_path / "collar.csv", index=False)
    assay.to_csv(tmp_path / "assay.csv", index=False)
    db = tmp_path / "db"
    argv = ["import", "--collar", str(tmp_path / "collar.csv"),
            "--assay", str(tmp_path / "assay.csv"), "--auto", "--out", str(db)]  # fmt: skip
    assert main(argv) == 0
    assert main(["desurvey", str(db)]) == 0
    assert main(["composite", str(db), "--length", "2"]) == 0
    capsys.readouterr()
    comps = pd.read_parquet(db / "composites_assay.parquet")
    # the composite y is the northing of the collar, the yttrium is y_ppm
    np.testing.assert_allclose(comps["y"], 2000.0)
    np.testing.assert_allclose(comps["y_ppm"], assay["Y_ppm"])
    np.testing.assert_allclose(comps["au"], assay["Au_ppm"])
