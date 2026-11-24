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
    dem_fill,
    grid_from_rasters,
    native_spacing,
    read_raster,
    residual_summary,
)
from bakumine.core.geometry.surface_points import surface_from_points

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


# ------------------------------------------------------------------ survey + DEM


def _segment_distance(p, a, b):
    """Plan distance from every point of ``p`` (m, 2) to the segment ``a``-``b``."""
    ab = b - a
    t = np.clip(((p - a) @ ab) / (ab @ ab), 0.0, 1.0)
    return np.hypot(*(p - (a + t[:, None] * ab)).T)


def _dem_grid(f, offset, spacing=10.0):
    xs = 500_000.0 + spacing * np.arange(61)
    ys = 5_540_000.0 + spacing * np.arange(51)
    return f(*np.meshgrid(xs, ys, indexing="ij")) + offset, (xs[0], ys[0])


def test_dem_fill_drops_nodes_near_the_survey_and_ties_the_heights():
    z, origin = _dem_grid(plane, 4.0)  # the DEM reads 4 m high
    gx, gy = np.meshgrid(
        np.arange(500_200.0, 500_300.1, 5.0), np.arange(5_540_150.0, 5_540_250.1, 5.0)
    )
    survey = np.column_stack([gx.ravel(), gy.ravel(), plane(gx, gy).ravel()])
    a, b = np.array([500_050.0, 5_540_420.0]), np.array([500_550.0, 5_540_380.0])
    line = np.column_stack([[a[0], b[0]], [a[1], b[1]], plane(*np.array([a, b]).T)])
    points, info = dem_fill(survey, [line], z, origin, 10.0, drop_within=25.0)
    assert info["z_shift"] == pytest.approx(-4.0) and info["tie"]["max_abs"] < 1e-6
    # independent count: exact plan distance to every survey point and to the segment
    nx, ny = z.shape
    nodes = np.column_stack(
        [a.ravel() for a in np.meshgrid(origin[0] + 10.0 * np.arange(nx),
                                        origin[1] + 10.0 * np.arange(ny), indexing="ij")]
    )  # fmt: skip
    d_points = np.min(np.hypot(*(nodes[:, None, :] - survey[None, :, :2]).transpose(2, 0, 1)), 1)
    d_line = _segment_distance(nodes, a, b)
    near = np.minimum(d_points, d_line)
    kept_d = np.minimum(
        np.min(np.hypot(*(points[:, None, :2] - survey[None, :, :2]).transpose(2, 0, 1)), 1),
        _segment_distance(points[:, :2], a, b),
    )
    assert kept_d.min() > 25.0 * 0.99  # nothing kept inside the drop distance
    assert info["kept"] >= int((near > 25.0).sum())  # nothing beyond it dropped
    assert info["kept"] <= int((near > 25.0 * 0.99).sum())
    assert info["kept"] + info["dropped"] == info["nodes"] == nx * ny
    assert np.allclose(points[:, 2], plane(points[:, 0], points[:, 1]))  # shifted onto the plane


def test_survey_and_dem_together_follow_each_where_it_is_right():
    def hill(e, n):
        return 300.0 + 40.0 * np.exp(-((e - 500_300.0) ** 2 + (n - 5_540_250.0) ** 2) / 150.0**2)

    def survey_truth(e, n):  # the survey sees a cut the DEM does not
        return hill(e, n) - np.where(np.hypot(e - 500_300.0, n - 5_540_250.0) < 40.0, 6.0, 0.0)

    z, origin = _dem_grid(hill, -2.0, spacing=10.0)
    gx, gy = np.meshgrid(
        np.arange(500_220.0, 500_380.1, 4.0), np.arange(5_540_170.0, 5_540_330.1, 4.0)
    )
    survey = np.column_stack([gx.ravel(), gy.ravel(), survey_truth(gx, gy).ravel()])
    fill, info = dem_fill(survey, None, z, origin, 10.0)
    assert info["z_shift"] == pytest.approx(2.0, abs=0.1)  # the cut is a minority: median holds
    surface = surface_from_points([survey, fill]).dtm
    inside = np.array([[500_300.0, 5_540_250.0], [500_250.0, 5_540_300.0]])
    assert np.allclose(surface.z_at(*inside.T), survey_truth(*inside.T), atol=0.2)
    outside = np.array(
        [[500_080.0, 5_540_050.0], [500_550.0, 5_540_420.0], [500_120.0, 5_540_450.0]]
    )
    assert np.allclose(surface.z_at(*outside.T), hill(*outside.T), atol=0.5)


def test_dem_fill_needs_a_shift_when_the_survey_is_off_the_dem():
    z, origin = _dem_grid(plane, 0.0)
    far = np.array([[400_000.0, 5_000_000.0, 1.0]])
    with pytest.raises(ValueError, match="cannot be tied"):
        dem_fill(far, None, z, origin, 10.0)
    points, info = dem_fill(far, None, z, origin, 10.0, z_shift="none")
    assert info["tie"] is None and len(points) == z.size


def test_cli_from_points_with_raster_merges_survey_collars_and_dem(tmp_path, capsys):
    cell, nrow, ncol = 10.0, 60, 80
    x_corner, y_corner = 499_800.0, 5_540_500.0
    xc = x_corner + cell * (np.arange(ncol) + 0.5)
    yc = y_corner - cell * (np.arange(nrow) + 0.5)
    write_geotiff(tmp_path / "dem.tif", plane(xc[None, :], yc[:, None]) + 3.0,
                  x_corner, y_corner, cell, cell, 32633, geographic=False)  # fmt: skip
    gx, gy = np.meshgrid(
        np.arange(500_050.0, 500_150.1, 5.0), np.arange(5_540_100.0, 5_540_200.1, 5.0)
    )
    pd.DataFrame({"E": gx.ravel(), "N": gy.ravel(), "RL": plane(gx, gy).ravel()}).to_csv(
        tmp_path / "survey.csv", index=False
    )
    ce, cn = (
        np.array([500_000.0, 500_250.0, 500_300.0]),
        np.array([5_540_000.0, 5_540_050.0, 5_540_250.0]),
    )
    collar = pd.DataFrame({"hole_id": ["A", "B", "C"], "x": ce, "y": cn, "z": plane(ce, cn)})
    _database(tmp_path / "db", collar, UTM)
    code = main(
        [
            "surface", "from-points",
            "--database", str(tmp_path / "db"),
            "--points", str(tmp_path / "survey.csv"),
            "--raster", str(tmp_path / "dem.tif"),
            "--extend-padding", "120",
            "--drop-within", "15",
            "--out", str(tmp_path / "topo.obj"),
            "--summary", str(tmp_path / "s.json"),
        ]
    )  # fmt: skip
    out = capsys.readouterr().out
    assert code == 0, out
    assert "shift -3.00 m" in out
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["dem"]["z_shift"] == pytest.approx(-3.0) and s["dem"]["crs"] == UTM
    assert s["dem"]["dropped"] > 0 and s["n_extension_points"] == 0
    from bakumine.core.geometry.dtm import DTM
    from bakumine.core.geometry.mesh_import import import_mesh

    surface = DTM(import_mesh(tmp_path / "topo.obj", "dtm", 1.0)[0])
    rng = np.random.default_rng(5)
    e = rng.uniform(499_900.0, 500_400.0, 200)
    n = rng.uniform(5_539_910.0, 5_540_350.0, 200)  # inside the outermost cell centres
    assert np.allclose(surface.z_at(e, n), plane(e, n), atol=1e-4)  # one plane (float32 cells)


def test_cli_raster_options_need_a_raster(tmp_path, capsys):
    collar = pd.DataFrame({"hole_id": list("ABC"), "x": [0.0, 10.0, 0.0], "y": [0.0, 0.0, 10.0],
                           "z": [1.0, 1.0, 1.0]})  # fmt: skip
    _database(tmp_path / "db", collar, None)
    code = main(["surface", "from-points", "--database", str(tmp_path / "db"),
                 "--drop-within", "10", "--out", str(tmp_path / "t.obj")])  # fmt: skip
    assert code != 0 and "--drop-within need --raster" in capsys.readouterr().err
