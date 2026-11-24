"""Download the Copernicus GLO-30 DEM tiles that cover an area (research.md section 5.2.1).

Nothing here runs unless asked for (``--fetch copernicus30``): a download sends the tile
names, and with them the location of the area to the nearest degree, to the public bucket
``copernicus-dem-30m`` on Amazon Web Services (open data, no account).

Tiles are one degree by one degree, named after their south-west corner:
``Copernicus_DSM_COG_10_N50_00_E015_00_DEM`` covers latitude 50 to 51 north, longitude 15
to 16 east (``S`` / ``W`` for south and west, latitude two digits, longitude three). Each is
a Cloud Optimised GeoTIFF, about 30-40 MB, in EPSG:4326 with heights above EGM2008. There
is no tile over open sea; the bucket then answers 404 and the tile is reported missing.

Downloaded tiles are kept in a cache folder and never fetched again.
"""

from __future__ import annotations

import math
import os
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np

__all__ = [
    "COPERNICUS30_URL",
    "copernicus_name",
    "copernicus_tiles",
    "default_cache",
    "fetch_tiles",
    "lonlat_box",
]

COPERNICUS30_URL = "https://copernicus-dem-30m.s3.amazonaws.com"
_TIMEOUT = 120.0


def default_cache() -> Path:
    """``~/.cache/bakumine/dem``."""
    return Path.home() / ".cache" / "bakumine" / "dem"


def copernicus_name(lon: int, lat: int) -> str:
    """Name of the tile whose south-west corner is at (``lon``, ``lat``) whole degrees."""
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat):02d}_00_{ew}{abs(lon):03d}_00_DEM"


def copernicus_tiles(lon_min: float, lat_min: float, lon_max: float, lat_max: float) -> list[str]:
    """Names of the tiles that cover the longitude / latitude box, south to north, west to east.

    A box edge on a whole degree does not take the tile beyond it.
    """
    if not (np.isfinite([lon_min, lat_min, lon_max, lat_max]).all()):
        raise ValueError("the box must be finite")
    if not (lon_max >= lon_min and lat_max >= lat_min):
        raise ValueError("the box needs lon_min <= lon_max and lat_min <= lat_max")
    if not (-180 <= lon_min and lon_max <= 180 and -90 <= lat_min and lat_max <= 90):
        raise ValueError("longitude must be within -180..180 and latitude within -90..90")

    def span(lo: float, hi: float) -> range:
        first = math.floor(lo)
        last = max(first, math.ceil(hi) - 1)
        return range(first, last + 1)

    return [
        copernicus_name(lon, lat)
        for lat in span(lat_min, lat_max)
        for lon in span(lon_min, lon_max)
    ]


def lonlat_box(rectangle: Sequence[float], crs: str) -> tuple[float, float, float, float]:
    """Longitude / latitude box of a plan rectangle in a projected ``crs``.

    The edges are sampled (a straight edge in UTM is curved in longitude / latitude), so the
    box holds the whole rectangle.
    """
    from bakumine.core.drillholes.coordinates import reproject

    xmin, ymin, xmax, ymax = (float(v) for v in rectangle)
    t = np.linspace(0.0, 1.0, 33)
    xs = np.concatenate([xmin + t * (xmax - xmin), np.full(33, xmax),
                         xmax - t * (xmax - xmin), np.full(33, xmin)])  # fmt: skip
    ys = np.concatenate([np.full(33, ymin), ymin + t * (ymax - ymin),
                         np.full(33, ymax), ymax - t * (ymax - ymin)])  # fmt: skip
    lon, lat = reproject(xs, ys, crs, "EPSG:4326")
    if not (np.isfinite(lon).all() and np.isfinite(lat).all()):
        raise ValueError(f"the rectangle cannot be carried from {crs} to longitude / latitude")
    return float(lon.min()), float(lat.min()), float(lon.max()), float(lat.max())


def _open(url: str) -> Any:
    request = Request(url, headers={"User-Agent": "bakumine"})
    return urlopen(request, timeout=_TIMEOUT)  # noqa: S310 - fixed https URL


def fetch_tiles(
    names: Sequence[str],
    cache: str | Path | None = None,
    *,
    base_url: str = COPERNICUS30_URL,
    opener: Callable[[str], Any] | None = None,
    progress: Callable[[str], None] | None = None,
) -> dict[str, list]:
    """Tiles ``names`` as local files, downloaded into ``cache`` when not there yet.

    Returns ``{"files": [paths], "downloaded": [names], "cached": [names],
    "missing": [names]}``; ``missing`` are the tiles the bucket does not have (sea).
    """
    folder = Path(cache) if cache is not None else default_cache()
    folder.mkdir(parents=True, exist_ok=True)
    opener = opener or _open
    out: dict[str, list] = {"files": [], "downloaded": [], "cached": [], "missing": []}
    for name in names:
        path = folder / f"{name}.tif"
        if path.is_file() and path.stat().st_size > 0:
            out["files"].append(path)
            out["cached"].append(name)
            continue
        url = f"{base_url}/{name}/{name}.tif"
        if progress:
            progress(f"downloading {url}")
        try:
            response = opener(url)
        except HTTPError as exc:
            if exc.code in (403, 404):  # the bucket answers 403 or 404 for a tile it lacks
                out["missing"].append(name)
                continue
            raise
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=f".{name}.", suffix=".part")
        try:
            with os.fdopen(fd, "wb") as handle, response:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
            os.replace(tmp, path)
        except BaseException:
            Path(tmp).unlink(missing_ok=True)
            raise
        out["files"].append(path)
        out["downloaded"].append(name)
    return out
