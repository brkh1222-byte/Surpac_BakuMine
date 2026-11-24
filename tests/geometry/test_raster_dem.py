"""Topography from elevation rasters (GeoTIFF / ASCII grid), datum tie and collar check."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import tifffile
from pyproj import Transformer
from scipy.interpolate import RegularGridInterpolator

from bakumine.cli import main
from bakumine.core.drillholes.importer import DrillholeDatabase, DrillholeMetadata
from bakumine.core.geometry.dtm import dtm_from_grid
from bakumine.core.geometry.raster_dem import (
    collar_residuals,
    grid_from_rasters,
    native_spacing,
    read_raster,
    residual_summary,
)

UTM = "EPSG:32633"


def write_geotiff(
    path, z, x_tie, y_tie, sx, sy, epsg, *, geographic, pixel_is_point=False, nodata=None
):
    """A GeoTIFF with the tags a GIS writes: pixel scale, one tie point, geo keys, nodata."""
    model_type = 2 if geographic else 1
    cs_key = 2048 if geographic else 3072
    keys = [
        1,
        1,
        0,
        3,
        1024,
        0,
        1,
        model_type,
        1025,
        0,
        1,
        2 if pixel_is_point else 1,
        cs_key,
        0,
        1,
        epsg,
    ]
    extratags = [
        (33550, "d", 3, (sx, sy, 0.0), False),
        (33922, "d", 6, (0.0, 0.0, 0.0, x_tie, y_tie, 0.0), False),
        (34735, "H", len(keys), keys, False),
    ]
    if nodata is not None:
        extratags.append((42113, "s", 0, f"{nodata:g}", False))
    tifffile.imwrite(path, np.asarray(z, dtype=np.float32), extratags=extratags)


def plane(e, n):
    return 300.0 + 0.02 * (e - 500_000.0) - 0.015 * (n - 5_540_000.0)


def test_projected_geotiff_reproduces_a_plane_exactly(tmp_path):
    # cells of 10 m, rows from north to south, tie point at the corner of the first cell
    nrow, ncol, cell = 40, 60, 10.0
    x_corner, y_corner = 499_800.0, 5_540_400.0
    xc = x_corner + cell * (np.arange(ncol) + 0.5)
    yc = y_corner - cell * (np.arange(nrow) + 0.5)
    z = plane(xc[None, :], yc[:, None])
    write_geotiff(tmp_path / "dem.tif", z, x_corner, y_corner, cell, cell, 32633, geographic=False)
    r = read_raster(tmp_path / "dem.tif")
    assert r.crs == UTM and r.info["pixel_is"] == "area"
    assert (r.x0, r.y0) == (xc[0], yc[0])
    assert native_spacing(r) == cell
    g, origin, info = grid_from_rasters(
        [r], (499_900.0, 5_540_100.0, 500_300.0, 5_540_300.0), 7.0, UTM
    )
    gx = origin[0] + 7.0 * np.arange(g.shape[0])[:, None]
    gy = origin[1] + 7.0 * np.arange(g.shape[1])[None, :]
    assert info["nodes_without_height"] == 0
    np.testing.assert_allclose(g, plane(gx, gy), atol=1e-3)  # float32 storage
    dtm = dtm_from_grid(g, origin, (7.0, 7.0))
    p = np.random.default_rng(1).uniform([499_910, 5_540_110], [500_290, 5_540_290], (200, 2))
    np.testing.assert_allclose(dtm.z_at(p[:, 0], p[:, 1]), plane(p[:, 0], p[:, 1]), atol=1e-3)


def test_pixel_is_point_puts_the_tie_point_on_the_cell_centre(tmp_path):
    z = np.arange(12, dtype=float).reshape(3, 4)
    write_geotiff(tmp_path / "a.tif", z, 100.0, 200.0, 2.0, 2.0, 32633, geographic=False)
    write_geotiff(
        tmp_path / "p.tif", z, 100.0, 200.0, 2.0, 2.0, 32633, geographic=False, pixel_is_point=True
    )
    area, point = read_raster(tmp_path / "a.tif"), read_raster(tmp_path / "p.tif")
    assert (area.x0, area.y0) == (101.0, 199.0)
    assert (point.x0, point.y0) == (100.0, 200.0)
    assert (area.dx, area.dy) == (2.0, -2.0)


def test_geographic_dem_matches_scipy_after_projection(tmp_path):
    """A lon/lat raster (Copernicus layout: pixel is point, 1 arc second) resampled to UTM."""
    step = 1.0 / 3600.0
    lon0, lat0 = 15.0, 50.01  # north-west centre
    ncol, nrow = 120, 90
    lons = lon0 + step * np.arange(ncol)
    lats = lat0 - step * np.arange(nrow)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    e, n = to_utm.transform(*np.meshgrid(lons, lats))
    z = plane(e, n) + 3.0 * np.sin(e / 40.0)  # some relief between the cells
    write_geotiff(
        tmp_path / "geo.tif", z, lon0, lat0, step, step, 4326, geographic=True, pixel_is_point=True
    )
    r = read_raster(tmp_path / "geo.tif")
    assert r.geographic and r.crs == "EPSG:4326"
    assert 19.0 < native_spacing(r) < 20.5  # one arc second of longitude at 50 N
    emin, nmin = to_utm.transform(lons[10], lats[-10])
    emax, nmax = to_utm.transform(lons[-10], lats[10])
    g, origin, info = grid_from_rasters([r], (emin, nmin, emax, nmax), 15.0, UTM)
    assert info["nodes_without_height"] == 0
    gx, gy = np.meshgrid(
        origin[0] + 15.0 * np.arange(g.shape[0]),
        origin[1] + 15.0 * np.arange(g.shape[1]),
        indexing="ij",
    )
    lon, lat = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True).transform(gx, gy)
    oracle = RegularGridInterpolator(
        (lats[::-1], lons), np.asarray(z, dtype=np.float32)[::-1].astype(float)
    )
    expected = oracle(np.column_stack([lat.ravel(), lon.ravel()])).reshape(g.shape)
    np.testing.assert_allclose(g, expected, atol=1e-6)


def test_ascii_grid_nodata_and_corner_header(tmp_path):
    text = (
        "ncols 4\nnrows 3\nxllcorner 1000\nyllcorner 2000\ncellsize 5\nNODATA_value -9999\n"
        "10 11 12 13\n20 21 -9999 23\n30 31 32 33\n"
    )
    (tmp_path / "g.asc").write_text(text)
    r = read_raster(tmp_path / "g.asc")
    assert (r.x0, r.y0, r.dx, r.dy) == (1002.5, 2012.5, 5.0, -5.0)  # first row is the north one
    assert np.isnan(r.z[1, 2]) and r.z[2, 0] == 30.0
    # centre of the south-west cell, and half way between the two west cells of the top rows
    np.testing.assert_allclose(
        r.sample(np.array([1002.5, 1002.5]), np.array([2002.5, 2010.0])), [30.0, 15.0]
    )
    # next to the missing cell: NaN, unless the gap-filling pass rescales the known weights
    assert np.isnan(r.sample(np.array([1015.0]), np.array([2007.5])))[0]
    np.testing.assert_allclose(
        r.sample(np.array([1015.0]), np.array([2007.5]), edge=True),
        [(12 + 13 + 32 + 33 + 23) / 5],
        rtol=0.2,
    )
    with pytest.raises(ValueError, match="values for"):
        (tmp_path / "bad.asc").write_text(text.replace("30 31 32 33\n", ""))
        read_raster(tmp_path / "bad.asc")


def test_adjacent_tiles_have_no_gap_at_the_seam(tmp_path):
    """Two tiles whose outermost centres are one cell apart: the strip between them is filled."""
    cell = 10.0
    xc = 5.0 + cell * np.arange(20)
    yc = 195.0 - cell * np.arange(20)
    z = 100.0 + 0.1 * xc[None, :] + 0.05 * yc[:, None]
    write_geotiff(tmp_path / "w.tif", z[:, :10], 0.0, 200.0, cell, cell, 32633, geographic=False)
    write_geotiff(tmp_path / "e.tif", z[:, 10:], 100.0, 200.0, cell, cell, 32633, geographic=False)
    rasters = [read_raster(tmp_path / "w.tif"), read_raster(tmp_path / "e.tif")]
    g, origin, info = grid_from_rasters(rasters, (20.0, 20.0, 180.0, 180.0), 2.5, UTM)
    assert info["nodes_without_height"] == 0 and info["nodes_from_edges"] > 0
    gx = origin[0] + 2.5 * np.arange(g.shape[0])[:, None]
    gy = origin[1] + 2.5 * np.arange(g.shape[1])[None, :]
    gx, gy = np.broadcast_arrays(gx, gy)
    exact = 100.0 + 0.1 * gx + 0.05 * gy
    # a plane stays a plane across the strip between the tiles (blended linearly)
    np.testing.assert_allclose(g, exact, atol=1e-3)
    assert info["nodes_from_edges"] == int(((gx > 95.0) & (gx < 105.0)).sum())


def test_collar_residuals_find_the_datum_shift_and_the_bad_collar():
    xs, ys = np.arange(0, 101, 10.0), np.arange(0, 101, 10.0)
    z = 500.0 + 0.1 * xs[:, None] + 0.2 * ys[None, :]
    dtm = dtm_from_grid(z, (0.0, 0.0), (10.0, 10.0))
    rng = np.random.default_rng(4)
    x, y = rng.uniform(5, 95, 30), rng.uniform(5, 95, 30)
    collar = pd.DataFrame(
        {
            "hole_id": [f"H{i}" for i in range(30)],
            "x": x,
            "y": y,
            "z": 500.0 + 0.1 * x + 0.2 * y + 2.5,
        }
    )
    collar.loc[7, "z"] += 12.0  # a wrong collar
    collar.loc[30] = ["OUT", 500.0, 500.0, 600.0]  # off the surface
    table = collar_residuals(dtm, collar)
    s = residual_summary(table, tolerance=5.0)
    assert s["median"] == pytest.approx(2.5)
    assert s["on_surface"] == 30 and s["off_surface"] == ["OUT"]
    assert [row["hole_id"] for row in s["suspect"]] == ["H7"]
    assert s["suspect"][0]["off_median"] == pytest.approx(12.0)
    shifted = residual_summary(collar_residuals(dtm, collar, shift=2.5))
    assert shifted["median"] == pytest.approx(0.0)


def _database(folder, collar: pd.DataFrame, crs: str | None) -> None:
    survey = pd.DataFrame({"hole_id": collar.hole_id, "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    DrillholeDatabase(
        collar.assign(max_depth=50.0), survey, metadata=DrillholeMetadata(crs=crs)
    ).save(folder)


def test_cli_from_raster_ties_to_the_collars_and_check_agrees(tmp_path, capsys):
    step = 1.0 / 3600.0
    lon0, lat0, ncol, nrow = 15.0, 50.01, 100, 80
    lons, lats = lon0 + step * np.arange(ncol), lat0 - step * np.arange(nrow)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    e, n = to_utm.transform(*np.meshgrid(lons, lats))
    write_geotiff(
        tmp_path / "dem.tif",
        plane(e, n),
        lon0,
        lat0,
        step,
        step,
        4326,
        geographic=True,
        pixel_is_point=True,
    )
    rng = np.random.default_rng(2)
    ce = rng.uniform(e.min() + 300, e.max() - 300, 25)
    cn = rng.uniform(n.min() + 300, n.max() - 300, 25)
    collar = pd.DataFrame(
        {"hole_id": [f"D{i:02d}" for i in range(25)], "x": ce, "y": cn, "z": plane(ce, cn) - 1.7}
    )  # local heights 1.7 m below the DEM
    collar.loc[3, "z"] += 9.0
    _database(tmp_path / "db", collar, UTM)
    code = main(
        [
            "surface",
            "from-raster",
            str(tmp_path / "dem.tif"),
            "--database",
            str(tmp_path / "db"),
            "--padding",
            "100",
            "--out",
            str(tmp_path / "topo.obj"),
            "--residuals",
            str(tmp_path / "res.csv"),
            "--summary",
            str(tmp_path / "s.json"),
        ]
    )
    out = capsys.readouterr().out
    assert code == 0, out
    assert "vertical shift: -1.70 m" in out and "D03" in out
    summary = json.loads((tmp_path / "s.json").read_text())
    assert summary["z_shift"] == pytest.approx(-1.7, abs=1e-3)
    assert summary["crs"] == UTM
    assert [r["hole_id"] for r in summary["collars"]["suspect"]] == ["D03"]
    res = pd.read_csv(tmp_path / "res.csv")
    assert np.abs(res.dz.drop(index=3)).max() < 0.01  # the shifted surface meets the collars
    # the written surface, checked on its own, gives the same picture with no further shift
    code = main(
        [
            "surface",
            "check",
            str(tmp_path / "topo.obj"),
            "--database",
            str(tmp_path / "db"),
            "--summary",
            str(tmp_path / "c.json"),
        ]
    )
    assert code == 0
    check = json.loads((tmp_path / "c.json").read_text())
    assert abs(check["median"]) < 0.01 and [r["hole_id"] for r in check["suspect"]] == ["D03"]


def test_cli_needs_a_crs_for_a_geographic_raster(tmp_path, capsys):
    write_geotiff(
        tmp_path / "dem.tif", np.ones((5, 5)), 15.0, 50.0, 0.001, 0.001, 4326, geographic=True
    )
    collar = pd.DataFrame({"hole_id": ["A"], "x": [500_000.0], "y": [5_540_000.0], "z": [1.0]})
    _database(tmp_path / "db", collar, None)
    code = main(
        [
            "surface",
            "from-raster",
            str(tmp_path / "dem.tif"),
            "--database",
            str(tmp_path / "db"),
            "--out",
            str(tmp_path / "t.obj"),
        ]
    )
    assert code != 0
    assert "--crs" in capsys.readouterr().err
