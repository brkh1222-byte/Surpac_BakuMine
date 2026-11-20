"""Collars in degrees / minutes / seconds, the axis order, surveys made from the collar
table, and one table from several sources (with a conflict report).

The fixtures are synthetic (``messy_fixtures``); the reference is always
``import_drillholes`` on three plain CSV files, which reads them with the plain reader
(``read_raw_table``) and knows nothing of sheets, header rows, rules or recipes.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from messy_fixtures import (
    CLEAN_MAPS,
    HOLE_ID,
    N_HOLES,
    OPTIONS,
    UTM,
    clean_fixture,
    dms,
    hole_name,
    recipe_for,
    same_assays,
    tables_fixture,
    write_collar_dms,
)

from bakumine.core.drillholes.coordinates import dms_to_decimal, parse_angle, project, utm_crs
from bakumine.core.drillholes.recipe import (
    make_recipe,
    merge_sources,
    run_recipe,
)

tables = tables_fixture()
clean = clean_fixture()


def kruger_utm(lon: float, lat: float, zone: int) -> tuple[float, float]:
    """Transverse Mercator by the Krüger series (Karney 2011, to n^3), WGS 84, UTM north:
    a second implementation to check pyproj's use against."""
    a, f = 6378137.0, 1 / 298.257223563
    n = f / (2 - f)
    big_a = a / (1 + n) * (1 + n**2 / 4 + n**4 / 64)
    alpha = [n / 2 - 2 * n**2 / 3 + 5 * n**3 / 16, 13 * n**2 / 48 - 3 * n**3 / 5,
             61 * n**3 / 240]  # fmt: skip
    phi, lam = math.radians(lat), math.radians(lon - (zone * 6 - 183))
    c = 2 * math.sqrt(n) / (1 + n)
    t = math.sinh(math.atanh(math.sin(phi)) - c * math.atanh(c * math.sin(phi)))
    xi = math.atan2(t, math.cos(lam))
    eta = math.atanh(math.sin(lam) / math.sqrt(1 + t * t))
    east = eta + sum(a_j * math.cos(2 * j * xi) * math.sinh(2 * j * eta)
                     for j, a_j in enumerate(alpha, start=1))  # fmt: skip
    north = xi + sum(a_j * math.sin(2 * j * xi) * math.cosh(2 * j * eta)
                     for j, a_j in enumerate(alpha, start=1))  # fmt: skip
    return 500000 + 0.9996 * big_a * east, 0.9996 * big_a * north


@pytest.mark.parametrize(
    ("lon", "lat"), [(15.8425, 46.9911), (98.7864, 43.9017), (-70.5, 61.25), (3.0, 0.0)]
)
def test_projection_matches_an_independent_series(lon, lat):
    crs = utm_crs(lon, lat)
    zone = int(crs[-2:])
    assert crs == f"EPSG:326{zone:02d}" and zone == int((lon + 180) // 6) + 1
    east, north = project(np.array([lon]), np.array([lat]), crs)
    want = kruger_utm(lon, lat, zone)
    assert east[0] == pytest.approx(want[0], abs=1e-3)
    assert north[0] == pytest.approx(want[1], abs=1e-3)
    if lon == 3.0:  # on the central meridian at the equator: the false origin
        assert (east[0], north[0]) == pytest.approx((500000.0, 0.0), abs=1e-6)
    assert utm_crs(lon, -lat - 1) == f"EPSG:327{zone:02d}"


def test_angles_from_text_and_parts():
    assert dms_to_decimal(98, 47, 11.0) == pytest.approx(98 + 47 / 60 + 11 / 3600)
    assert dms_to_decimal(-0.0, 30, 0) == -0.5 and dms_to_decimal(-12, 30, np.nan) == -12.5
    for text in ["98 47 11.0", "98°47'11.0\"E", "98° 47′ 11,0″", "E 98 47 11"]:
        assert parse_angle(text) == pytest.approx(98.786388888, abs=1e-8)
    assert parse_angle("12°30'S") == -12.5 and parse_angle("-12.5") == -12.5
    assert parse_angle(43.9) == 43.9
    assert math.isnan(parse_angle("n/a")) and math.isnan(parse_angle("12 75 00"))
    assert dms(12.5) == (12, 30, 0.0)


def collar_recipe(paths, book, **options):
    return make_recipe(
        {
            "collar": {"file": str(book)},
            "assay": {"file": str(paths["assay"]), "map": CLEAN_MAPS["assay"]},
        },
        {**OPTIONS, "coordinate_order": "auto", **options},
        hole_id=HOLE_ID,
    )


@pytest.mark.parametrize("text", [False, True])
def test_collar_in_degrees_minutes_seconds_and_survey_from_collar(tmp_path, tables, clean, text):
    paths, clean_db = clean
    book = tmp_path / "collars.xlsx"
    write_collar_dms(book, tables["collar"], tables["survey"], projected=False, text=text)
    db = run_recipe(collar_recipe(paths, book))
    found = db.metadata.recipe["tables"]["collar"][0]["map"]
    if text:
        assert found["Уртраг"] == "lon" and found["Өргөрөг"] == "lat"
    else:
        assert [found[k] for k in ("Уртраг", "Уртраг 2", "Уртраг 3")] == [
            "lon_deg", "lon_min", "lon_sec"]  # fmt: skip
    assert db.metadata.crs == UTM  # the zone was found from the data
    # seconds are written to 0.001": a few centimetres
    np.testing.assert_allclose(db.collar["x"], clean_db.collar["x"], atol=0.05)
    np.testing.assert_allclose(db.collar["y"], clean_db.collar["y"], atol=0.05)
    assert (db.collar["x"] != clean_db.collar["x"]).any()  # converted, not copied
    for column in ("hole_id", "z", "max_depth"):
        pd.testing.assert_series_equal(db.collar[column], clean_db.collar[column])
    # no survey table: one station per hole from the collar's azimuth and dip
    pd.testing.assert_frame_equal(db.survey, clean_db.survey)
    assert db.dip_convention == "positive_down"
    assert db.metadata.reports["survey"]["generated_from_collar"] == N_HOLES
    report = db.metadata.reports["coordinates"]
    assert (report["converted"], report["checked"], report["disagree"]) == (N_HOLES, 0, 0)
    same_assays(db, clean_db)
    # a CRS can be named; a wrong zone is a different place
    other = run_recipe(collar_recipe(paths, book, crs="EPSG:32634"))
    assert other.metadata.crs == "EPSG:32634"
    assert (np.abs(other.collar["x"] - clean_db.collar["x"]) > 1000).all()
    with pytest.raises(ValueError, match="cannot project"):
        run_recipe(collar_recipe(paths, book, crs="EPSG:nonsense"))


def test_projected_columns_are_checked_against_their_degrees(tmp_path, tables, clean):
    paths, clean_db = clean
    book = tmp_path / "collars.xlsx"
    write_collar_dms(
        book,
        tables["collar"],
        tables["survey"],
        blank_projected=(hole_name(2),),
        shifted={hole_name(4): 120.0},
    )
    db = run_recipe(collar_recipe(paths, book))
    # X = northing, Y = easting in the file: found from the agreement with the degrees
    assert db.metadata.coordinate_order == "yxz"
    report = db.metadata.reports["coordinates"]
    assert (report["converted"], report["checked"], report["disagree"]) == (1, N_HOLES - 1, 1)
    (row,) = report["rows"]
    assert row["hole_id"] == hole_name(4) and row["difference"] == pytest.approx(120.0, abs=0.1)
    assert "projected columns of the table are kept" in report["precedence"]
    exact = ~clean_db.collar["hole_id"].isin([hole_name(2), hole_name(4)])
    pd.testing.assert_frame_equal(
        db.collar.loc[exact, ["hole_id", "x", "y", "z", "max_depth"]], clean_db.collar[exact]
    )
    filled = db.collar.set_index("hole_id").loc[hole_name(2)]
    want = clean_db.collar.set_index("hole_id").loc[hole_name(2)]
    assert (filled["x"], filled["y"]) == pytest.approx((want["x"], want["y"]), abs=0.05)
    kept = db.collar.set_index("hole_id").loc[hole_name(4)]
    assert kept["y"] == clean_db.collar.set_index("hole_id").loc[hole_name(4), "y"] + 120.0
    # a wider tolerance accepts the shifted hole
    wide = run_recipe(collar_recipe(paths, book, coordinate_tolerance=150.0))
    assert wide.metadata.reports["coordinates"]["disagree"] == 0


def test_decimal_degrees_and_axis_order_by_magnitude(tmp_path, tables, clean):
    paths, clean_db = clean
    from pyproj import Transformer

    collar = tables["collar"]
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(
        collar["x"].to_numpy(), collar["y"].to_numpy()
    )
    decimal = tmp_path / "collar_deg.csv"
    collar.assign(Longitude=lon, Latitude=lat).drop(columns=["x", "y"]).to_csv(
        decimal, index=False, sep=";"
    )
    recipe = make_recipe(
        {"collar": {"file": str(decimal)}, "survey": {"file": str(paths["survey"])}}, OPTIONS
    )
    db = run_recipe(recipe)
    np.testing.assert_allclose(db.collar[["x", "y"]], clean_db.collar[["x", "y"]], atol=1e-3)
    # northing first, no degrees to compare with: decided by the magnitudes, and said
    swapped = tmp_path / "collar_cis.csv"
    collar.rename(columns={"x": "Y", "y": "X"}).to_csv(swapped, index=False, sep=";")
    recipe = make_recipe(
        {"collar": {"file": str(swapped)}, "survey": {"file": str(paths["survey"])}},
        {**OPTIONS, "coordinate_order": "auto"},
    )
    db = run_recipe(recipe)
    assert db.metadata.coordinate_order == "yxz"
    pd.testing.assert_frame_equal(db.collar, clean_db.collar)
    kinds = [n["kind"] for n in db.metadata.notes]
    assert "coordinate_order_assumed_yxz" in kinds
    recipe["options"]["coordinate_order"] = "xyz"
    db = run_recipe(recipe)  # as mapped, but the importer says what it sees
    assert "x_looks_like_northing" in [n["kind"] for n in db.metadata.notes]
    # no survey and no azimuth / dip in the collar table
    with pytest.raises(ValueError, match="no survey table"):
        run_recipe(make_recipe({"collar": {"file": str(paths["collar"])}}, OPTIONS))


def test_assays_from_several_files_with_a_conflict(tmp_path, tables, clean):
    paths, clean_db = clean
    assay = tables["assay"]
    early = assay[assay["hole_id"] <= hole_name(3)]
    late = assay[assay["hole_id"] >= hole_name(3)].copy()  # hole 3 is in both files
    changed = late.index[2]
    assert late.loc[changed, "hole_id"] == hole_name(3)
    late.loc[changed, "au"] = "9.99"
    first, second = tmp_path / "assay_a.csv", tmp_path / "assay_b.csv"
    early.to_csv(first, index=False, sep=";")
    late.to_csv(second, index=False, sep=";")
    recipe = recipe_for(paths, [{"file": str(first)}, {"file": str(second)}])
    db = run_recipe(recipe)
    same_assays(db, clean_db)  # the first file wins
    report = db.metadata.reports["conflicts"]["assay"]
    n_shared = int((assay["hole_id"] == hole_name(3)).sum())
    assert report["precedence"] == "first" and report["merged"] == n_shared
    assert report["conflicts"] == 1
    (row,) = report["rows"]
    kept = clean_db.intervals["assay"].loc[changed, "au"]
    assert row["hole_id"] == hole_name(3) and row["column"] == "au"
    assert (row["kept_source"], row["dropped_source"]) == ("assay_a.csv", "assay_b.csv")
    assert row["dropped"] == 9.99
    assert row["kept"] == (None if np.isnan(kept) else kept) or np.isnan(kept)
    last = run_recipe(recipe_for(paths, recipe["tables"]["assay"], merge={"precedence": "last"}))
    got = last.intervals["assay"].set_index(["hole_id", "depth_to"])["au"]
    assert got.loc[(hole_name(3), assay.loc[changed, "to"])] == 9.99
    assert len(last.intervals["assay"]) == len(clean_db.intervals["assay"])


def test_one_hole_from_two_tables(tmp_path, tables, clean):
    """Coordinates from one table, azimuth / dip / depth from another."""
    paths, clean_db = clean
    collar, survey = tables["collar"], tables["survey"]
    surveyed = tmp_path / "surveyed.csv"
    collar[["hole_id", "x", "y", "z"]].to_csv(surveyed, index=False, sep=";")
    drilled = tmp_path / "drilled.csv"
    drill = collar[["hole_id", "z", "max_depth"]].merge(survey[["hole_id", "azimuth", "dip"]])
    drill["hole_id"] = drill["hole_id"].str.replace("-", "_")
    drill.loc[1, "z"] += 3.0  # the planned elevation of one hole differs from the surveyed
    drill.iloc[:-1].to_csv(drilled, index=False, sep=";")  # and the last hole was not drilled
    recipe = make_recipe(
        {
            "collar": [{"file": str(surveyed)}, {"file": str(drilled)}],
            "assay": {"file": str(paths["assay"]), "map": CLEAN_MAPS["assay"]},
        },
        OPTIONS,
        hole_id=HOLE_ID,
    )
    db = run_recipe(recipe)
    pd.testing.assert_frame_equal(
        db.collar[["hole_id", "x", "y", "z"]], clean_db.collar[["hole_id", "x", "y", "z"]]
    )
    # the hole without drilling data: depth from its deepest assay, no survey station
    deepest = tables["assay"].groupby("hole_id")["to"].max()
    want_depth = clean_db.collar["max_depth"].copy()
    want_depth.iloc[-1] = deepest[hole_name(N_HOLES)]
    np.testing.assert_array_equal(db.collar["max_depth"], want_depth)
    pd.testing.assert_frame_equal(db.survey, clean_db.survey.iloc[:-1])
    assert db.metadata.reports["survey"]["holes_without_azimuth_or_dip"] == [hole_name(N_HOLES)]
    report = db.metadata.reports["conflicts"]["collar"]
    assert (report["merged"], report["conflicts"]) == (N_HOLES - 1, 1)
    assert report["filled"] == 3 * (N_HOLES - 1)  # max_depth, azimuth, dip
    (row,) = report["rows"]
    assert row == {
        "hole_id": hole_name(2), "column": "z", "kept": collar.loc[1, "z"],
        "dropped": collar.loc[1, "z"] + 3.0, "kept_source": "surveyed.csv",
        "dropped_source": "drilled.csv",
    }  # fmt: skip


def test_merge_sources_keeps_duplicates_inside_one_source():
    table = pd.DataFrame(
        {"hole_id": ["A", "A", "A", "B"], "depth": [0.0, 0.0, 0.0, 0.0],
         "dip": [-60.0, -60.0, -60.004, np.nan], "azimuth": [10.0, 10.0, np.nan, 5.0]}  # fmt: skip
    )
    merged, report = merge_sources(table, np.array([0, 0, 1, 1]), ("hole_id", "depth"))
    assert len(merged) == 3 and report["merged"] == 1 and report["conflicts"] == 0
    merged, report = merge_sources(
        table, np.array([0, 0, 1, 1]), ("hole_id", "depth"), tolerance=0.001
    )
    assert report["conflicts"] == 1 and report["rows"][0]["kept"] == -60.0
