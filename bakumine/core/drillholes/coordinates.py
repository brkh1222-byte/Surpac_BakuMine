"""Collar coordinates given as longitude / latitude, and the axis order of X and Y.

Exploration collar tables often carry the position as degrees-minutes-seconds (three
columns each, or text such as ``98°47'11.0"E``), as decimal degrees, as projected
coordinates, or as several of these at once. This module turns what the column map calls

* ``lon_deg, lon_min, lon_sec, lat_deg, lat_min, lat_sec`` (numbers), or
* ``lon, lat`` (decimal degrees or text)

into projected ``x = east, y = north`` with ``pyproj`` (transverse Mercator; the UTM zone
of the data unless ``ImportOptions.crs`` names a CRS), and checks them against the
projected columns of the same table when there are any:

* rows where both exist and differ by more than ``coordinate_tolerance`` are reported;
  the projected columns of the table are kept (stated precedence: table over conversion);
* rows without projected coordinates are filled from the conversion;
* with ``coordinate_order="auto"`` the order of the projected columns (``X`` = northing
  in CIS-style tables) is the one that agrees with the longitude / latitude. Without
  longitude / latitude the order is decided by magnitude when it is unambiguous
  (a UTM easting is below 1 000 000 m, a northing away from the equator is above).
"""

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .importer import DrillholeMetadata, ImportOptions

__all__ = [
    "dms_to_decimal",
    "parse_angle",
    "project",
    "resolve_collar_coordinates",
    "utm_crs",
]

_NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
_MAX_LISTED = 100


def dms_to_decimal(degrees: Any, minutes: Any = 0.0, seconds: Any = 0.0) -> np.ndarray:
    """Degrees, minutes, seconds -> decimal degrees. The sign is that of ``degrees``
    (``-0`` counts as negative); missing minutes or seconds count as 0."""
    d = np.asarray(degrees, dtype=np.float64)
    m = np.nan_to_num(np.asarray(minutes, dtype=np.float64), nan=0.0)
    s = np.nan_to_num(np.asarray(seconds, dtype=np.float64), nan=0.0)
    sign = np.where(np.signbit(d), -1.0, 1.0)
    return sign * (np.abs(d) + np.abs(m) / 60.0 + np.abs(s) / 3600.0)


def parse_angle(text: object) -> float:
    """``"98 47 11.0"``, ``98°47'11"E``, ``"43.9017"``, ``"S 12 30"`` -> decimal degrees
    (NaN when unreadable). ``W`` / ``S`` or a leading minus make it negative."""
    if text is None:
        return math.nan
    if isinstance(text, int | float | np.integer | np.floating):
        return float(text)
    raw = str(text).strip()
    parts = [float(p.replace(",", ".")) for p in _NUMBER.findall(raw)]
    if not parts or len(parts) > 3:
        return math.nan
    parts += [0.0] * (3 - len(parts))
    if parts[1] >= 60.0 or parts[2] >= 60.0:
        return math.nan
    value = parts[0] + parts[1] / 60.0 + parts[2] / 3600.0
    letters = re.sub(r"[^a-zа-я]", "", raw.casefold())
    negative = raw.startswith("-") or letters in ("w", "s", "з", "ю")
    return -value if negative else value


def utm_crs(lon: float, lat: float) -> str:
    """EPSG code of the WGS 84 / UTM zone holding the point."""
    zone = min(60, max(1, int(math.floor((lon + 180.0) / 6.0)) + 1))
    return f"EPSG:{(32600 if lat >= 0 else 32700) + zone}"


def project(
    lon: np.ndarray, lat: np.ndarray, crs: str, geographic_crs: str = "EPSG:4326"
) -> tuple[np.ndarray, np.ndarray]:
    """Longitude, latitude (degrees) -> easting, northing in ``crs`` (NaN stays NaN)."""
    from pyproj import Transformer

    lon = np.asarray(lon, dtype=np.float64)
    lat = np.asarray(lat, dtype=np.float64)
    east = np.full(lon.shape, np.nan)
    north = np.full(lon.shape, np.nan)
    known = np.isfinite(lon) & np.isfinite(lat)
    if known.any():
        transformer = Transformer.from_crs(geographic_crs, crs, always_xy=True)
        east[known], north[known] = transformer.transform(lon[known], lat[known])
    return east, north


def _geographic(collar: pd.DataFrame) -> tuple[np.ndarray, np.ndarray] | None:
    """Decimal longitude and latitude of the collar rows, or None when not given."""
    columns = set(collar.columns)
    if {"lon_deg", "lat_deg"} <= columns:
        def part(name: str) -> Any:
            return collar[name].to_numpy(dtype=np.float64) if name in columns else 0.0

        return (
            dms_to_decimal(part("lon_deg"), part("lon_min"), part("lon_sec")),
            dms_to_decimal(part("lat_deg"), part("lat_min"), part("lat_sec")),
        )
    if {"lon", "lat"} <= columns:
        return (
            np.array([parse_angle(v) for v in collar["lon"].tolist()], dtype=np.float64),
            np.array([parse_angle(v) for v in collar["lat"].tolist()], dtype=np.float64),
        )
    return None


def _order_by_magnitude(x: np.ndarray, y: np.ndarray) -> str | None:
    """``"yxz"`` when x holds northings and y eastings beyond doubt, ``"xyz"`` for the
    reverse, None when the magnitudes do not tell (local grids, the equator)."""
    both = np.isfinite(x) & np.isfinite(y)
    if not both.any():
        return None
    x, y = x[both], y[both]

    def easting(v: np.ndarray) -> bool:
        return bool(((v >= 1.0e5) & (v < 1.0e6)).all())

    def northing(v: np.ndarray) -> bool:
        return bool(((v >= 1.0e6) & (v <= 1.0e7)).all())

    if northing(x) and easting(y):
        return "yxz"
    if easting(x) and northing(y):
        return "xyz"
    return None


def resolve_collar_coordinates(
    collar: pd.DataFrame, options: ImportOptions, metadata: DrillholeMetadata
) -> tuple[pd.DataFrame, str]:
    """Fill / check the collar's projected columns and decide the axis order.

    Returns ``(collar, coordinate_order)`` where the order is a permutation of ``"xyz"``
    still to be applied by the caller (the collar keeps its columns as mapped). Notes and
    the ``coordinates`` report are added to ``metadata``.
    """
    collar = collar.copy()
    order = options.coordinate_order
    notes = metadata.notes

    def note(kind: str, count: int, **extra: Any) -> None:
        if count:
            notes.append(
                {"table": "collar", "column": "x,y", "kind": kind, "count": count, **extra}
            )

    for axis in ("x", "y"):
        if axis not in collar.columns:
            collar[axis] = np.nan
    x = collar["x"].to_numpy(dtype=np.float64)
    y = collar["y"].to_numpy(dtype=np.float64)
    geographic = _geographic(collar)

    if geographic is None:
        guess = _order_by_magnitude(x, y)
        if order == "auto":
            order = guess or "xyz"
            note(f"coordinate_order_assumed_{order}", int(len(collar)),
                 reason="magnitude" if guess else "not decidable; kept as mapped")  # fmt: skip
        elif guess is not None and guess[:2] != order[:2] and order in ("xyz", "yxz"):
            note("x_looks_like_northing" if guess == "yxz" else "x_looks_like_easting",
                 int(len(collar)), hint="check coordinate_order (or use 'auto')")  # fmt: skip
        both = np.isfinite(x) & np.isfinite(y)
        if both.any() and (np.abs(x[both]) <= 180).all() and (np.abs(y[both]) <= 180).all():
            if not np.allclose(x[both], np.round(x[both])):
                note("x_y_look_like_degrees", int(both.sum()),
                     hint="map the columns to lon / lat to project them")  # fmt: skip
        return collar, order

    lon, lat = geographic
    known = np.isfinite(lon) & np.isfinite(lat)
    note("longitude_latitude_unreadable", int((~known).sum()))
    crs = options.crs
    if crs is None or crs.casefold() == "utm":
        if not known.any():
            raise ValueError("collar: no readable longitude / latitude to choose a UTM zone from")
        crs = utm_crs(float(np.median(lon[known])), float(np.median(lat[known])))
    try:
        east, north = project(lon, lat, crs, options.geographic_crs)
    except Exception as exc:  # pyproj.exceptions.CRSError and friends
        raise ValueError(f"collar: cannot project to {crs!r}: {exc}") from None
    metadata.crs = crs

    have = np.isfinite(x) & np.isfinite(y)
    check = have & known
    if order == "auto":
        order = "xyz"
        if check.any():
            as_mapped = np.hypot(x[check] - east[check], y[check] - north[check])
            swapped = np.hypot(y[check] - east[check], x[check] - north[check])
            order = "yxz" if np.median(swapped) < np.median(as_mapped) else "xyz"
            note(f"coordinate_order_assumed_{order}", int(check.sum()),
                 reason="agreement with longitude / latitude")  # fmt: skip
    # east/north in the columns' own order, so that the caller's renaming puts them right
    as_x, as_y = (north, east) if order[:2] == "yx" else (east, north)
    if options.length_unit != "m":  # projected metres -> the unit of the file
        from .importer import LENGTH_UNITS

        factor = LENGTH_UNITS[options.length_unit]
        as_x, as_y = as_x / factor, as_y / factor
    distance = np.hypot(x - as_x, y - as_y)
    tolerance = float(options.coordinate_tolerance)
    disagree = check & (distance > tolerance)
    fill = ~have & known
    collar.loc[fill, "x"] = as_x[fill]
    collar.loc[fill, "y"] = as_y[fill]
    note("converted_from_longitude_latitude", int(fill.sum()), crs=crs)
    examples = collar["hole_id"].to_numpy()[disagree][:5].tolist()
    note("projected_differs_from_longitude_latitude", int(disagree.sum()),
         tolerance=tolerance, examples=examples)  # fmt: skip
    metadata.reports["coordinates"] = {
        "crs": crs,
        "geographic_crs": options.geographic_crs,
        "order": order,
        "tolerance": tolerance,
        "converted": int(fill.sum()),
        "checked": int(check.sum()),
        "disagree": int(disagree.sum()),
        "max_difference": float(distance[check].max()) if check.any() else None,
        "precedence": "projected columns of the table are kept; rows without them are "
        "filled from longitude / latitude",
        "rows": [
            {
                "hole_id": str(collar["hole_id"].iat[i]),
                "difference": round(float(distance[i]), 3),
                "table": [float(x[i]), float(y[i])],
                "from_longitude_latitude": [round(float(as_x[i]), 3), round(float(as_y[i]), 3)],
            }
            for i in np.flatnonzero(disagree)[:_MAX_LISTED]
        ],
    }
    return collar, order
