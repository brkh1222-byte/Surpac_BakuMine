"""Copernicus GLO-30 tiles for an area: names, longitude / latitude box, cached download.

No network: the bucket is replaced by a function that serves files written by the test.
"""

from __future__ import annotations

import io
import json
from urllib.error import HTTPError

import numpy as np
import pandas as pd
import pytest
from pyproj import Transformer

from bakumine.cli import main
from bakumine.core.geometry import dem_fetch as df
from tests.geometry.test_raster_dem import UTM, _database, plane, write_geotiff


def test_tile_names_follow_the_south_west_corner():
    assert df.copernicus_name(15, 50) == "Copernicus_DSM_COG_10_N50_00_E015_00_DEM"
    assert df.copernicus_name(-71, -33) == "Copernicus_DSM_COG_10_S33_00_W071_00_DEM"
    assert df.copernicus_name(0, 0) == "Copernicus_DSM_COG_10_N00_00_E000_00_DEM"
    assert df.copernicus_name(-1, -1) == "Copernicus_DSM_COG_10_S01_00_W001_00_DEM"


def test_tiles_of_a_box():
    assert df.copernicus_tiles(15.2, 50.1, 15.8, 50.9) == [df.copernicus_name(15, 50)]
    # across the 0 meridian and the equator, south to north then west to east
    assert df.copernicus_tiles(-0.5, -0.5, 0.5, 0.5) == [
        df.copernicus_name(-1, -1),
        df.copernicus_name(0, -1),
        df.copernicus_name(-1, 0),
        df.copernicus_name(0, 0),
    ]
    # an edge on a whole degree does not take the next tile
    assert df.copernicus_tiles(15.0, 50.0, 16.0, 51.0) == [df.copernicus_name(15, 50)]
    with pytest.raises(ValueError):
        df.copernicus_tiles(10.0, 0.0, 9.0, 1.0)


def test_lonlat_box_holds_the_whole_rectangle():
    rect = (480_000.0, 5_530_000.0, 520_000.0, 5_560_000.0)
    lon0, lat0, lon1, lat1 = df.lonlat_box(rect, UTM)
    back = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    xx, yy = np.meshgrid(np.linspace(rect[0], rect[2], 21), np.linspace(rect[1], rect[3], 21))
    lon, lat = back.transform(xx, yy)
    assert lon.min() >= lon0 - 1e-9 and lon.max() <= lon1 + 1e-9
    assert lat.min() >= lat0 - 1e-9 and lat.max() <= lat1 + 1e-9
    # the top edge is curved: its middle (on the central meridian) is further north than its
    # corners, so the box from the corners alone would be too small
    assert lat1 > max(back.transform(rect[0], rect[3])[1], back.transform(rect[2], rect[3])[1])


class _Bucket:
    """Serves ``files[name]`` for ``.../<name>/<name>.tif``, 404 for any other tile."""

    def __init__(self, files):
        self.files, self.urls = files, []

    def __call__(self, url):
        self.urls.append(url)
        name = url.rsplit("/", 1)[1][: -len(".tif")]
        assert url == f"{df.COPERNICUS30_URL}/{name}/{name}.tif"
        if name not in self.files:
            raise HTTPError(url, 404, "Not Found", {}, None)
        return io.BytesIO(self.files[name])


def test_fetch_downloads_once_and_reports_missing_tiles(tmp_path):
    land, sea = df.copernicus_name(15, 50), df.copernicus_name(16, 50)
    bucket = _Bucket({land: b"tile bytes"})
    got = df.fetch_tiles([land, sea], tmp_path / "cache", opener=bucket)
    assert got["downloaded"] == [land] and got["missing"] == [sea] and not got["cached"]
    assert got["files"] == [tmp_path / "cache" / f"{land}.tif"]
    assert got["files"][0].read_bytes() == b"tile bytes"
    assert not list((tmp_path / "cache").glob(".*.part"))
    again = df.fetch_tiles([land], tmp_path / "cache", opener=bucket)
    assert again["cached"] == [land] and len(bucket.urls) == 2  # not asked again


def test_fetch_leaves_no_partial_file_when_the_download_breaks(tmp_path):
    class Broken(io.BytesIO):
        def read(self, *_):
            raise OSError("connection reset")

    name = df.copernicus_name(15, 50)
    with pytest.raises(OSError):
        df.fetch_tiles([name], tmp_path, opener=lambda url: Broken())
    assert list(tmp_path.iterdir()) == []


def _tile_bytes(tmp_path):
    """A 1 x 1 degree tile at 15-16 E, 50-51 N (pixel is point), heights of ``plane`` + 2."""
    step = 0.01
    lons = 15.0 + step * np.arange(101)
    lats = 51.0 - step * np.arange(101)
    to_utm = Transformer.from_crs("EPSG:4326", UTM, always_xy=True)
    e, n = to_utm.transform(*np.meshgrid(lons, lats))
    path = tmp_path / "tile.tif"
    write_geotiff(path, plane(e, n) + 2.0, 15.0, 51.0, step, step, 4326,
                  geographic=True, pixel_is_point=True)  # fmt: skip
    return path.read_bytes()


def test_cli_from_raster_fetch_ties_the_fetched_tile(tmp_path, capsys, monkeypatch):
    bucket = _Bucket({df.copernicus_name(15, 50): _tile_bytes(tmp_path)})
    monkeypatch.setattr(df, "_open", bucket)
    ce = np.array([501_800.0, 502_200.0, 502_000.0])  # east of 15 E: one tile
    cn = np.array([5_541_000.0, 5_541_300.0, 5_541_600.0])
    collar = pd.DataFrame({"hole_id": ["A", "B", "C"], "x": ce, "y": cn, "z": plane(ce, cn)})
    _database(tmp_path / "db", collar, UTM)
    argv = ["surface", "from-raster", "--fetch", "copernicus30",
            "--dem-cache", str(tmp_path / "cache"), "--database", str(tmp_path / "db"),
            "--padding", "100", "--spacing", "50", "--out", str(tmp_path / "t.obj"),
            "--summary", str(tmp_path / "s.json")]  # fmt: skip
    assert main(argv) == 0
    out = capsys.readouterr().out
    assert "tile names go to" in out and "downloading" in out
    s = json.loads((tmp_path / "s.json").read_text())
    assert s["z_shift"] == pytest.approx(-2.0, abs=0.05)  # bilinear in lon/lat on a plane
    assert len(bucket.urls) == 1
    assert main(argv) == 0  # second run: from the cache
    assert "already in the cache" in capsys.readouterr().out and len(bucket.urls) == 1


def test_fetch_needs_a_crs_and_is_off_by_default(tmp_path, capsys, monkeypatch):
    def no_network(url):
        raise AssertionError("must not download")

    monkeypatch.setattr(df, "_open", no_network)
    xyz = {"x": [500_000.0, 500_100.0, 500_000.0], "y": [5_541_000.0, 5_541_000.0, 5_541_100.0]}
    collar = pd.DataFrame({"hole_id": list("ABC"), **xyz, "z": [1.0, 1.0, 1.0]})
    _database(tmp_path / "db", collar, None)
    code = main(["surface", "from-raster", "--fetch", "copernicus30", "--database",
                 str(tmp_path / "db"), "--out", str(tmp_path / "t.obj")])  # fmt: skip
    assert code != 0 and "--crs" in capsys.readouterr().err
    code = main(["surface", "from-raster", "--database", str(tmp_path / "db"),
                 "--out", str(tmp_path / "t.obj")])  # fmt: skip
    assert code != 0 and "--fetch" in capsys.readouterr().err
    # from-points without --fetch never touches the network
    assert main(["surface", "from-points", "--database", str(tmp_path / "db"),
                 "--out", str(tmp_path / "t.obj")]) == 0  # fmt: skip
