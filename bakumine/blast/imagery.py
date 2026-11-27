"""Satellite imagery for the blast view: web-map tiles over a plan rectangle of the model.

The viewer drapes the tiles on the terrain. This module does the geometry and the download;
it never decodes an image.

* Tiles are the usual 256-pixel "XYZ" tiles in spherical Web Mercator (EPSG:3857): at zoom
  ``z`` the world is ``2**z`` tiles wide, tile ``(0, 0)`` is the north-west corner, and for
  longitude ``lon`` / latitude ``lat`` (degrees) the fractional tile coordinates are

      x = (lon + 180) / 360 * 2**z
      y = (1 - asinh(tan(lat)) / pi) / 2 * 2**z

  (OpenStreetMap wiki, "Slippy map tilenames").
* :func:`plan` picks the highest zoom whose tile block over the rectangle stays within
  ``max_tiles`` and returns, on a regular grid of model points, the texture coordinates
  ``(u, v)`` in that tile block (``u`` to the east, ``v`` to the south, both 0..1). The
  viewer interpolates them bilinearly: a projected CRS against Web Mercator is smooth over
  a mine site, so a 33 x 33 grid is exact to well under a pixel.
* :func:`fetch` downloads the tiles into a cache folder. Nothing is requested unless it is
  called: the tile numbers tell the imagery provider where the site is (to about the size
  of one tile), so fetching is an explicit action of the user.

* :func:`fetch_best` steps down in zoom while the server answers with its "no data at this
  zoom" picture: such a block is recognised by all its tiles being the same bytes
  (:func:`is_placeholder`), which a photograph of ground never is.

The default source is Esri World Imagery; its terms ask for the attribution in
:data:`ESRI_ATTRIBUTION` to be shown with the picture.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

__all__ = [
    "ESRI_ATTRIBUTION",
    "ESRI_WORLD_IMAGERY_URL",
    "fetch",
    "fetch_best",
    "is_placeholder",
    "plan",
    "tile_path",
    "tile_xy",
]

ESRI_WORLD_IMAGERY_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
ESRI_ATTRIBUTION = "Imagery: Esri, Maxar, Earthstar Geographics, and the GIS User Community"
SOURCE = "esri-world-imagery"
TILE_SIZE = 256
MAX_ZOOM = 19
MAX_LATITUDE = 85.0511  # the edge of the Web Mercator square
GRID = 33
_TIMEOUT = 30.0
_WORKERS = 8


def tile_xy(lon: Any, lat: Any, zoom: int) -> tuple[np.ndarray, np.ndarray]:
    """Fractional tile coordinates of longitude / latitude (degrees) at ``zoom``."""
    lon = np.asarray(lon, dtype=np.float64)
    lat = np.radians(np.asarray(lat, dtype=np.float64))
    n = 2.0**zoom
    x = (lon + 180.0) / 360.0 * n
    y = (1.0 - np.arcsinh(np.tan(lat)) / math.pi) / 2.0 * n
    return x, y


def plan(
    rectangle: Sequence[float],
    crs: str,
    *,
    max_tiles: int = 100,
    max_zoom: int = MAX_ZOOM,
    grid: int = GRID,
) -> dict[str, Any]:
    """Tile block and texture coordinates for the plan ``rectangle`` (xmin, ymin, xmax, ymax)
    given in the projected ``crs``.

    Returns a JSON-ready dict::

        {"source", "attribution", "crs", "rectangle", "zoom", "tile_size",
         "tiles": {"x0", "y0", "nx", "ny"},          # block of nx x ny tiles from (x0, y0)
         "metres_per_pixel",                         # ground size of a pixel at the centre
         "grid": {"origin": [xmin, ymin], "step": [dx, dy], "shape": [n, n],
                  "u": [...], "v": [...]}}           # row-major [ix * n + iy]

    Raises ``ValueError`` for an empty rectangle, a rectangle that cannot be carried to
    longitude / latitude, or one outside the Web Mercator latitudes.
    """
    from bakumine.core.drillholes.coordinates import reproject

    xmin, ymin, xmax, ymax = (float(v) for v in rectangle)
    if not (math.isfinite(xmin + ymin + xmax + ymax) and xmax > xmin and ymax > ymin):
        raise ValueError(
            "the imagery rectangle must be xmin, ymin, xmax, ymax with a positive size"
        )
    if not isinstance(crs, str) or not crs.strip():
        raise ValueError("a coordinate system (for example EPSG:32633) is needed for imagery")
    if grid < 2:
        raise ValueError("grid must be at least 2")
    if max_tiles < 1:
        raise ValueError("max_tiles must be at least 1")

    xs = np.linspace(xmin, xmax, grid)
    ys = np.linspace(ymin, ymax, grid)
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    try:
        lon, lat = reproject(gx.ravel(), gy.ravel(), crs.strip(), "EPSG:4326")
    except Exception as exc:  # pyproj raises its own CRSError for an unknown code
        raise ValueError(f"the coordinate system {crs!r} is not known: {exc}") from None
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"the rectangle cannot be carried from {crs} to longitude / latitude")
    if np.abs(lat).max() > MAX_LATITUDE:
        raise ValueError("the rectangle is outside the latitudes web-map imagery covers")

    zoom = 0
    for z in range(max_zoom, -1, -1):
        tx, ty = tile_xy(lon, lat, z)
        nx = int(math.floor(tx.max())) - int(math.floor(tx.min())) + 1
        ny = int(math.floor(ty.max())) - int(math.floor(ty.min())) + 1
        if nx * ny <= max_tiles:
            zoom = z
            break
    tx, ty = tile_xy(lon, lat, zoom)
    x0, y0 = int(math.floor(tx.min())), int(math.floor(ty.min()))
    nx = int(math.floor(tx.max())) - x0 + 1
    ny = int(math.floor(ty.max())) - y0 + 1
    u = (tx - x0) / nx
    v = (ty - y0) / ny
    lat0 = float(np.radians(lat.mean()))
    metres_per_pixel = 2.0 * math.pi * 6378137.0 * math.cos(lat0) / (TILE_SIZE * 2.0**zoom)
    return {
        "source": SOURCE,
        "attribution": ESRI_ATTRIBUTION,
        "crs": crs.strip(),
        "rectangle": [xmin, ymin, xmax, ymax],
        "zoom": zoom,
        "tile_size": TILE_SIZE,
        "tiles": {"x0": x0, "y0": y0, "nx": nx, "ny": ny},
        "metres_per_pixel": metres_per_pixel,
        "grid": {
            "origin": [xmin, ymin],
            "step": [(xmax - xmin) / (grid - 1), (ymax - ymin) / (grid - 1)],
            "shape": [grid, grid],
            "u": [float(a) for a in u],
            "v": [float(a) for a in v],
        },
    }


def tile_path(cache: str | Path, zoom: int, x: int, y: int) -> Path:
    """Where tile ``(zoom, x, y)`` is kept in the cache folder."""
    return Path(cache) / str(int(zoom)) / str(int(y)) / f"{int(x)}.jpg"


def _open(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "tile-client/1.0"})  # noqa: S310
    return urlopen(request, timeout=_TIMEOUT)  # noqa: S310 - fixed https URL


def fetch(
    tile_plan: dict[str, Any],
    cache: str | Path,
    *,
    url: str = ESRI_WORLD_IMAGERY_URL,
    opener: Callable[[str], Any] | None = None,
) -> dict[str, int]:
    """Download the tiles of ``tile_plan`` (from :func:`plan`) into ``cache``.

    Tiles already there are kept. Returns ``{"tiles", "downloaded", "cached", "missing"}``;
    ``missing`` counts tiles the server does not have (HTTP 404). Other errors are raised
    as ``ValueError`` with the server's words.
    """
    block = tile_plan["tiles"]
    zoom = int(tile_plan["zoom"])
    wanted = [
        (x, y)
        for x in range(block["x0"], block["x0"] + block["nx"])
        for y in range(block["y0"], block["y0"] + block["ny"])
    ]
    opener = opener or _open

    def one(xy: tuple[int, int]) -> str:
        x, y = xy
        path = tile_path(cache, zoom, x, y)
        if path.is_file() and path.stat().st_size > 0:
            return "cached"
        try:
            with opener(url.format(z=zoom, x=x, y=y)) as response:
                data = response.read()
        except HTTPError as exc:
            if exc.code == 404:
                return "missing"
            raise ValueError(f"the imagery server answered {exc.code} {exc.reason}") from None
        except OSError as exc:
            raise ValueError(f"the imagery server cannot be reached: {exc}") from None
        if not data:
            return "missing"
        path.parent.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as file:
                file.write(data)
            os.replace(temporary, path)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
        return "downloaded"

    with ThreadPoolExecutor(max_workers=_WORKERS) as pool:
        states = list(pool.map(one, wanted))
    return {
        "tiles": len(wanted),
        "downloaded": states.count("downloaded"),
        "cached": states.count("cached"),
        "missing": states.count("missing"),
    }


def is_placeholder(tile_plan: dict[str, Any], cache: str | Path) -> bool:
    """True when the block of ``tile_plan`` holds two or more tiles in ``cache`` and all of
    them are the same bytes: the server's picture for "no imagery at this zoom", not ground.
    A block of one tile cannot be judged and gives False.
    """
    block = tile_plan["tiles"]
    zoom = int(tile_plan["zoom"])
    first: bytes | None = None
    seen = 0
    for x in range(block["x0"], block["x0"] + block["nx"]):
        for y in range(block["y0"], block["y0"] + block["ny"]):
            path = tile_path(cache, zoom, x, y)
            if not path.is_file():
                continue
            data = path.read_bytes()
            if first is None:
                first = data
            elif data != first:
                return False
            seen += 1
    return seen >= 2


def fetch_best(
    rectangle: Sequence[float],
    crs: str,
    cache: str | Path,
    *,
    max_tiles: int = 100,
    max_zoom: int = MAX_ZOOM,
    url: str = ESRI_WORLD_IMAGERY_URL,
    opener: Callable[[str], Any] | None = None,
) -> tuple[dict[str, Any], dict[str, int]]:
    """:func:`plan` and :func:`fetch` at the highest zoom that has imagery.

    A zoom whose block is a placeholder (:func:`is_placeholder`) is skipped and the next lower
    one is tried. Returns the plan, with ``"skipped_zooms"`` listing the zooms without imagery,
    and the fetch counts of that plan.
    """
    skipped: list[int] = []
    zoom = max_zoom
    while True:
        tile_plan = plan(rectangle, crs, max_tiles=max_tiles, max_zoom=zoom)
        counts = fetch(tile_plan, cache, url=url, opener=opener)
        if tile_plan["zoom"] == 0 or not is_placeholder(tile_plan, cache):
            break
        skipped.append(int(tile_plan["zoom"]))
        zoom = int(tile_plan["zoom"]) - 1
    tile_plan["skipped_zooms"] = skipped
    return tile_plan, counts
