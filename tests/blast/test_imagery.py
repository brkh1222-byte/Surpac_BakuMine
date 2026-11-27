"""Satellite tiles for the blast view: tile numbers against the published formula and a
second projection route, texture coordinates against exact per-point values, the download
with a stand-in server, and the API (which never fetches unless asked)."""

from __future__ import annotations

import io
import math
from urllib.error import HTTPError

import numpy as np
import pytest
from fastapi.testclient import TestClient
from pyproj import Transformer

from bakumine.api import app
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.scene import EmptySceneSource, set_scene_source
from bakumine.blast import imagery
from bakumine.core.geometry.dtm import dtm_from_grid
from bakumine.core.io.obj import write_obj
from bakumine.core.project import Project

CRS = "EPSG:32633"  # UTM zone 33N
RECT = [500000.0, 5540000.0, 503000.0, 5542000.0]
API = "/api/v1/blast"


def test_tile_numbers_follow_the_slippy_map_formula():
    # Hand values: zoom 0 holds the world in one tile; Greenwich / equator is its centre.
    x, y = imagery.tile_xy(0.0, 0.0, 0)
    assert (float(x), float(y)) == pytest.approx((0.5, 0.5))
    # West and north edges of the Mercator square.
    x, y = imagery.tile_xy(-180.0, imagery.MAX_LATITUDE, 3)
    assert float(x) == pytest.approx(0.0)
    assert float(y) == pytest.approx(0.0, abs=1e-4)
    # A point by the formula written with log / tan / sec instead of asinh.
    lon, lat, z = 15.3, 50.02, 14
    r = math.radians(lat)
    want_y = (1 - math.log(math.tan(r) + 1 / math.cos(r)) / math.pi) / 2 * 2**z
    x, y = imagery.tile_xy(lon, lat, z)
    assert float(x) == pytest.approx((lon + 180) / 360 * 2**z)
    assert float(y) == pytest.approx(want_y)


def test_tile_numbers_agree_with_web_mercator_metres():
    """Second route: EPSG:3857 metres from pyproj, scaled to tiles."""
    lon = np.array([-70.5, 15.0, 106.9])
    lat = np.array([-33.4, 50.0, 47.9])
    z = 12
    mx, my = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lon, lat)
    half = math.pi * 6378137.0
    x, y = imagery.tile_xy(lon, lat, z)
    np.testing.assert_allclose(x, (mx + half) / (2 * half) * 2**z, rtol=1e-12)
    np.testing.assert_allclose(y, (half - my) / (2 * half) * 2**z, rtol=1e-12)


def test_plan_covers_the_rectangle_within_the_tile_budget():
    p = imagery.plan(RECT, CRS, max_tiles=100)
    block = p["tiles"]
    assert block["nx"] * block["ny"] <= 100
    u, v = np.array(p["grid"]["u"]), np.array(p["grid"]["v"])
    assert u.min() >= 0 and u.max() <= 1 and v.min() >= 0 and v.max() <= 1
    # One zoom deeper would break the budget (or the cap is reached).
    deeper = imagery.plan(RECT, CRS, max_tiles=10_000, max_zoom=p["zoom"] + 1)
    assert p["zoom"] == imagery.MAX_ZOOM or deeper["tiles"]["nx"] * deeper["tiles"]["ny"] > 100
    # u grows to the east, v to the south.
    n = p["grid"]["shape"][0]
    assert u[(n - 1) * n] > u[0]  # ix = n-1 against ix = 0, at iy = 0
    assert v[n - 1] < v[0]  # iy = n-1 (north) against iy = 0 (south)
    # Pixel size: Web Mercator ground resolution at this latitude.
    lat = 50.0
    want = 156543.03392 * math.cos(math.radians(lat)) / 2 ** p["zoom"]
    assert p["metres_per_pixel"] == pytest.approx(want, rel=2e-3)


def test_bilinear_use_of_the_grid_is_exact_to_a_fraction_of_a_pixel():
    """What the viewer does (bilinear interpolation of the grid) against the exact tile
    coordinates of random points in the rectangle."""
    p = imagery.plan(RECT, CRS, max_tiles=100)
    g = p["grid"]
    n = g["shape"][0]
    u = np.array(g["u"]).reshape(n, n)
    v = np.array(g["v"]).reshape(n, n)
    rng = np.random.default_rng(3)
    x = rng.uniform(RECT[0], RECT[2], 500)
    y = rng.uniform(RECT[1], RECT[3], 500)
    fx = (x - g["origin"][0]) / g["step"][0]
    fy = (y - g["origin"][1]) / g["step"][1]
    ix = np.clip(np.floor(fx).astype(int), 0, n - 2)
    iy = np.clip(np.floor(fy).astype(int), 0, n - 2)
    tx, ty = fx - ix, fy - iy

    def lerp(a):
        return (
            a[ix, iy] * (1 - tx) * (1 - ty)
            + a[ix + 1, iy] * tx * (1 - ty)
            + a[ix, iy + 1] * (1 - tx) * ty
            + a[ix + 1, iy + 1] * tx * ty
        )

    lon, lat = Transformer.from_crs(CRS, "EPSG:4326", always_xy=True).transform(x, y)
    ex, ey = imagery.tile_xy(lon, lat, p["zoom"])
    block = p["tiles"]
    pixels_u = (lerp(u) - (ex - block["x0"]) / block["nx"]) * block["nx"] * 256
    pixels_v = (lerp(v) - (ey - block["y0"]) / block["ny"]) * block["ny"] * 256
    assert np.abs(pixels_u).max() < 0.05
    assert np.abs(pixels_v).max() < 0.05


@pytest.mark.parametrize(
    "rectangle, crs, words",
    [
        ([0, 0, 0, 10], CRS, "positive size"),
        (RECT, "", "coordinate system"),
        (RECT, "EPSG:999999", "not known"),
    ],
)
def test_plan_refuses_bad_input(rectangle, crs, words):
    with pytest.raises(ValueError, match=words):
        imagery.plan(rectangle, crs)


class _Answer(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_fetch_downloads_once_and_reports_missing_tiles(tmp_path):
    p = imagery.plan(RECT, CRS, max_tiles=6)
    block = p["tiles"]
    total = block["nx"] * block["ny"]
    asked: list[str] = []
    gone = f"/{p['zoom']}/{block['y0']}/{block['x0']}"

    def opener(url):
        asked.append(url)
        if url.endswith(gone):
            raise HTTPError(url, 404, "Not Found", None, None)
        return _Answer(b"jpeg-bytes")

    first = imagery.fetch(p, tmp_path, opener=opener)
    assert first == {"tiles": total, "downloaded": total - 1, "cached": 0, "missing": 1}
    assert len(asked) == total
    assert all(u.startswith("https://server.arcgisonline.com/") for u in asked)
    kept = imagery.tile_path(tmp_path, p["zoom"], block["x0"] + block["nx"] - 1, block["y0"])
    assert kept.read_bytes() == b"jpeg-bytes"
    again = imagery.fetch(p, tmp_path, opener=opener)
    assert again["cached"] == total - 1 and again["downloaded"] == 0
    assert len(asked) == total + 1  # only the missing tile is asked for again

    def broken(url):
        raise HTTPError(url, 500, "Server Error", None, None)

    with pytest.raises(ValueError, match="500"):
        imagery.fetch(p, tmp_path / "other", opener=broken)


def test_fetch_best_steps_down_past_zooms_without_imagery(tmp_path):
    top = imagery.plan(RECT, CRS, max_tiles=40)["zoom"]
    asked: list[int] = []

    def opener(url):
        zoom = int(url.split("/tile/")[1].split("/")[0])
        asked.append(zoom)
        # The two highest zooms answer every tile with one and the same picture.
        return _Answer(b"no data" if zoom > top - 2 else b"ground " + url.encode())

    p, counts = imagery.fetch_best(RECT, CRS, tmp_path, max_tiles=40, opener=opener)
    assert p["zoom"] == top - 2 and p["skipped_zooms"] == [top, top - 1]
    assert counts["downloaded"] == p["tiles"]["nx"] * p["tiles"]["ny"] and counts["missing"] == 0
    assert not imagery.is_placeholder(p, tmp_path)
    assert imagery.is_placeholder(imagery.plan(RECT, CRS, max_tiles=40), tmp_path)
    assert sorted(set(asked), reverse=True) == [top, top - 1, top - 2]

    # One tile cannot be judged: it is kept.
    one = imagery.plan(RECT, CRS, max_tiles=1)
    imagery.fetch(one, tmp_path / "one", opener=lambda url: _Answer(b"no data"))
    assert one["tiles"]["nx"] * one["tiles"]["ny"] == 1
    assert not imagery.is_placeholder(one, tmp_path / "one")


@pytest.fixture
def served(tmp_path):
    i = np.arange(31)
    z = 300.0 + 0.0 * i[None, :] + np.zeros((31, 1))
    write_obj(
        tmp_path / "topo.obj",
        dtm_from_grid(z, (RECT[0], RECT[1]), (100.0, 100.0)).mesh,
        decimals=None,
    )
    folder = tmp_path / "project"
    Project.create(folder, "blast").add("dtm", tmp_path / "topo.obj", "Topography", "topo")
    set_scene_source(ProjectSceneSource(folder))
    yield TestClient(app), folder
    set_scene_source(None)


def test_api_fetches_only_when_asked(served, monkeypatch):
    client, folder = served
    calls: list[str] = []

    def opener(url):
        calls.append(url)
        return _Answer(b"jpeg-bytes " + url.encode())  # every tile its own picture

    monkeypatch.setattr(imagery, "_open", opener)

    state = client.get(f"{API}/imagery").json()
    assert state == {"available": False, "plan": None, "crs": None, "project": True}
    assert client.get(f"{API}/imagery/tile/10/1/1").status_code == 404
    assert calls == []  # reading the state and asking for a tile never goes out

    # No coordinate system anywhere: refused, still nothing sent.
    refused = client.post(f"{API}/imagery", json={})
    assert refused.status_code == 400 and "coordinate system" in refused.json()["detail"]
    assert calls == []

    done = client.post(f"{API}/imagery", json={"crs": CRS, "max_tiles": 12})
    assert done.status_code == 200, done.text
    plan = done.json()["plan"]
    block = plan["tiles"]
    assert plan["rectangle"] == [RECT[0], RECT[1], RECT[0] + 3000.0, RECT[1] + 3000.0]
    assert plan["fetched"]["downloaded"] == block["nx"] * block["ny"] == len(calls)
    assert (folder / "blast" / "imagery" / "plan.json").is_file()

    tile = client.get(f"{API}/imagery/tile/{plan['zoom']}/{block['y0']}/{block['x0']}")
    assert tile.status_code == 200 and tile.content.startswith(b"jpeg-bytes ")
    assert plan["skipped_zooms"] == []
    assert tile.headers["content-type"] == "image/jpeg"
    assert client.get(f"{API}/imagery").json()["plan"] == plan
    assert client.post(f"{API}/imagery", json={"crs": CRS, "max_tiles": 0}).status_code == 400


def test_api_without_a_project_has_no_imagery():
    set_scene_source(EmptySceneSource())
    try:
        client = TestClient(app)
        assert client.get(f"{API}/imagery").json()["project"] is False
        assert client.post(f"{API}/imagery", json={"crs": CRS}).status_code == 400
    finally:
        set_scene_source(None)
