"""Topography from an elevation raster (DEM), and checks of a surface against the collars.

research.md section 5.2.1. A DEM is a regular grid of heights: a satellite model
(Copernicus GLO-30, ALOS AW3D30, SRTM), a drone or LiDAR surface exported as a grid, or a
surveyor's grid.

Read
----
:func:`read_raster` reads

* GeoTIFF (``.tif`` / ``.tiff``, also Cloud Optimised GeoTIFF tiles), with its georeferencing
  tags: ``ModelPixelScale`` + ``ModelTiepoint`` (or a ``ModelTransformation`` without
  rotation), ``GTRasterTypeGeoKey`` (pixel is area: the tie point is the corner of the pixel;
  pixel is point: its centre), the EPSG code of ``ProjectedCSTypeGeoKey`` or
  ``GeographicTypeGeoKey``, and the ``GDAL_NODATA`` value;
* ESRI ASCII grid (``.asc``): ``ncols, nrows, xllcorner | xllcenter, yllcorner | yllcenter,
  cellsize`` (or ``dx, dy``), ``nodata_value``, then the rows from north to south.

The result is a :class:`Raster`: heights ``z[row, col]`` (NaN where there is no data) and
the position of the centre of each cell, ``x = x0 + col * dx``, ``y = y0 + row * dy``
(``dy`` is negative when the rows run from north to south).

Resample
--------
:func:`grid_from_rasters` lays a regular grid in the target (projected) CRS over a plan
rectangle, carries every node to the CRS of the rasters with ``pyproj`` and interpolates
there **bilinearly** between the four surrounding cell centres. With several rasters
(adjacent tiles) the first raster that knows a node gives its height. A second pass fills
the nodes left: those up to one cell beyond the outermost cell centres (the strip between two
tiles, the outer half cell of a raster) and those next to a cell without height take the
bilinear weights of the known cells, rescaled to sum to one, and the heights of the rasters
near the node are blended by their distance (:meth:`Raster.sample_edge`), so the surface is
continuous across a seam (linear across the strip). Nodes still unknown stay NaN
and their triangles are left out. :func:`~bakumine.core.geometry.dtm.dtm_from_grid` then
triangulates the grid.

Datum and collars
-----------------
Global DEMs give heights above a geoid model (EGM2008 for Copernicus), local surveys use
their own national height system, and a DEM may be shifted by a metre or more against the
survey. :func:`collar_residuals` compares a surface with the collars
(``dz = collar z - surface z``); the **median** of ``dz`` is the vertical shift that ties the
surface to the surveyed collars (the median, so that a few wrong collars do not move it).
Collars that still differ from the shifted surface by more than a tolerance are listed: a
wrong collar elevation, a collar on a waste dump or in a pit dug after the DEM was made, or
an error in the DEM.

Survey and DEM together
-----------------------
A precise survey (points, contour strings, collars) is right where it was measured and
usually covers only the deposit; a DEM covers everything but is coarse. :func:`dem_fill`
keeps the DEM grid nodes farther than ``drop_within`` metres in plan from every survey point
and from every breakline segment, shifted by the median of ``survey z - DEM z`` (the
survey is the height reference), so that triangulating survey and kept nodes together
gives the survey near the deposit and the DEM around it.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

__all__ = [
    "Raster",
    "collar_residuals",
    "dem_fill",
    "grid_from_rasters",
    "native_spacing",
    "read_raster",
    "residual_summary",
]

#: GeoTIFF key values
_PIXEL_IS_AREA, _PIXEL_IS_POINT = 1, 2
_USER_DEFINED = 32767


@dataclass
class Raster:
    """A regular grid of heights.

    ``z[row, col]`` is the height of the cell whose centre is at ``(x0 + col * dx,
    y0 + row * dy)`` in ``crs`` (``None`` when the file does not say). NaN marks no data.
    """

    z: np.ndarray
    x0: float
    y0: float
    dx: float
    dy: float
    crs: str | None = None
    source: str = ""
    info: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.z.shape  # type: ignore[return-value]

    @property
    def geographic(self) -> bool:
        """True when the CRS is longitude / latitude."""
        if self.crs is None:
            return False
        from pyproj import CRS

        return bool(CRS.from_user_input(self.crs).is_geographic)

    def extent(self) -> tuple[float, float, float, float]:
        """``(xmin, ymin, xmax, ymax)`` of the cell centres."""
        nrow, ncol = self.shape
        xs = (self.x0, self.x0 + (ncol - 1) * self.dx)
        ys = (self.y0, self.y0 + (nrow - 1) * self.dy)
        return min(xs), min(ys), max(xs), max(ys)

    def sample(self, x: np.ndarray, y: np.ndarray, edge: bool = False) -> np.ndarray:
        """Bilinear height at ``(x, y)`` given in the raster CRS.

        NaN outside the cell centres or when one of the four cells has no height. With
        ``edge=True`` see :meth:`sample_edge` (its value, without the weight).
        """
        return self._sample(x, y, edge)[0]

    def sample_edge(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Gap-filling sample: ``(value, weight)``.

        A point up to one cell beyond the outermost cell centres is moved onto them, and the
        bilinear weights of the cells that have a height are rescaled to sum to one. ``weight``
        is ``(1 - a) * (1 - b)`` for a point ``a`` columns and ``b`` rows beyond the centres
        (1 inside, 0 one cell out), so that heights from two tiles either side of a seam can be
        blended linearly across the strip between them.
        """
        return self._sample(x, y, True)

    def _sample(self, x, y, edge: bool) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        nrow, ncol = self.shape
        col = (x - self.x0) / self.dx
        row = (y - self.y0) / self.dy
        out = np.full(x.shape, np.nan)
        weight = np.zeros(x.shape)
        reach = 1.0 if edge else 0.0
        inside = np.isfinite(col) & np.isfinite(row)
        inside &= (col >= -reach) & (col <= ncol - 1 + reach)
        inside &= (row >= -reach) & (row <= nrow - 1 + reach)
        if not inside.any() or nrow < 2 or ncol < 2:
            return out, weight
        c = np.clip(col[inside], 0.0, ncol - 1)
        r = np.clip(row[inside], 0.0, nrow - 1)
        # a point exactly on the last row / column is interpolated in the cell before it
        c0 = np.minimum(np.floor(c).astype(np.int64), ncol - 2)
        r0 = np.minimum(np.floor(r).astype(np.int64), nrow - 2)
        fc, fr = c - c0, r - r0
        z = self.z
        corners = np.stack([z[r0, c0], z[r0, c0 + 1], z[r0 + 1, c0], z[r0 + 1, c0 + 1]])
        weights = np.stack([(1 - fc) * (1 - fr), fc * (1 - fr), (1 - fc) * fr, fc * fr])
        if edge:
            known = np.isfinite(corners)
            weights = np.where(known, weights, 0.0)
            total = weights.sum(axis=0)
            with np.errstate(invalid="ignore", divide="ignore"):
                value = (np.where(known, corners, 0.0) * weights).sum(axis=0) / total
            value[total <= 1e-12] = np.nan
        else:
            value = (corners * weights).sum(axis=0)
        out[inside] = value
        beyond_c = np.abs(col[inside] - c)
        beyond_r = np.abs(row[inside] - r)
        weight[inside] = np.where(np.isfinite(value), (1 - beyond_c) * (1 - beyond_r), 0.0)
        return out, weight


# ------------------------------------------------------------------ reading


def read_raster(path: str | Path, crs: str | None = None) -> Raster:
    """Read a GeoTIFF or ESRI ASCII grid; ``crs`` overrides (or supplies) the file's CRS."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".tif", ".tiff"):
        raster = _read_geotiff(path)
    elif suffix == ".asc":
        raster = _read_ascii_grid(path)
    else:
        raise ValueError(f"{path.name}: elevation rasters are .tif / .tiff (GeoTIFF) or .asc")
    if crs:
        raster.crs = _crs_text(crs)
    if not np.isfinite(raster.z).any():
        raise ValueError(f"{path.name}: the grid holds no height")
    return raster


def _crs_text(crs: str) -> str:
    from pyproj import CRS

    try:
        parsed = CRS.from_user_input(crs)
    except Exception as exc:  # pyproj.exceptions.CRSError
        raise ValueError(f"unknown CRS {crs!r}: {exc}") from None
    code = parsed.to_epsg()
    return f"EPSG:{code}" if code else parsed.to_string()


def _key(meta: dict[str, Any], name: str) -> int | None:
    value = meta.get(name)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_geotiff(path: Path) -> Raster:
    import tifffile

    with tifffile.TiffFile(path) as tif:
        page = tif.pages[0]
        meta = tif.geotiff_metadata or {}
        try:
            data = page.asarray()
        except ValueError as exc:  # a codec that is not installed
            raise ValueError(f"{path.name}: cannot decode the image ({exc})") from None
        nodata_tag = page.tags.get("GDAL_NODATA")
        nodata = None
        if nodata_tag is not None:
            try:
                nodata = float(str(nodata_tag.value).strip().strip("\x00"))
            except ValueError:
                nodata = None
    if data.ndim == 3:  # several samples per pixel: the first band
        data = data[..., 0] if data.shape[-1] <= 4 else data[0]
    if data.ndim != 2:
        raise ValueError(f"{path.name}: expected a single-band grid, got shape {data.shape}")
    z = np.asarray(data, dtype=np.float64)
    if nodata is not None and np.isfinite(nodata):
        z[z == nodata] = np.nan

    if "ModelTransformation" in meta:
        m = np.asarray(meta["ModelTransformation"], dtype=np.float64).reshape(4, 4)
        if abs(m[0, 1]) > 1e-12 or abs(m[1, 0]) > 1e-12:
            raise ValueError(f"{path.name}: rotated rasters are not supported")
        sx, sy, tx, ty = m[0, 0], m[1, 1], m[0, 3], m[1, 3]  # x = sx * col + tx, y = sy * row + ty
    elif "ModelPixelScale" in meta and "ModelTiepoint" in meta:
        scale = np.asarray(meta["ModelPixelScale"], dtype=np.float64).ravel()
        tie = np.asarray(meta["ModelTiepoint"], dtype=np.float64).ravel()
        if len(tie) > 6:
            raise ValueError(f"{path.name}: rasters with several tie points are not supported")
        i, j, x, y = tie[0], tie[1], tie[3], tie[4]
        sx, sy = scale[0], -scale[1]
        tx, ty = x - i * sx, y - j * sy
    else:
        raise ValueError(f"{path.name}: no georeferencing (ModelPixelScale / ModelTiepoint)")
    if not (sx and sy):
        raise ValueError(f"{path.name}: the pixel size is zero")
    raster_type = _key(meta, "GTRasterTypeGeoKey") or _PIXEL_IS_AREA
    half = 0.5 if raster_type == _PIXEL_IS_AREA else 0.0  # tie point at the corner of a cell
    x0, y0 = tx + half * sx, ty + half * sy

    crs = None
    for name in ("ProjectedCSTypeGeoKey", "GeographicTypeGeoKey"):
        code = _key(meta, name)
        if code and code != _USER_DEFINED:
            crs = f"EPSG:{code}"
            break
    vertical = _key(meta, "VerticalCSTypeGeoKey")
    info = {
        "format": "GeoTIFF",
        "pixel_is": "point" if raster_type == _PIXEL_IS_POINT else "area",
        "nodata": nodata,
        "vertical_crs": f"EPSG:{vertical}" if vertical and vertical != _USER_DEFINED else None,
    }
    return Raster(z, float(x0), float(y0), float(sx), float(sy), crs, path.name, info)


def _read_ascii_grid(path: Path) -> Raster:
    header: dict[str, float] = {}
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    k = 0
    while k < len(lines):
        parts = lines[k].split()
        if len(parts) == 2 and parts[0][:1].isalpha():
            try:
                header[parts[0].lower()] = float(parts[1])
            except ValueError:
                raise ValueError(f"{path.name}: header line {k + 1} is not a number") from None
            k += 1
        elif not parts:
            k += 1
        else:
            break
    for need in ("ncols", "nrows"):
        if need not in header:
            raise ValueError(f"{path.name}: ASCII grid header has no {need}")
    ncols, nrows = int(header["ncols"]), int(header["nrows"])
    if "cellsize" in header:
        dx = dy = header["cellsize"]
    elif "dx" in header and "dy" in header:
        dx, dy = header["dx"], header["dy"]
    else:
        raise ValueError(f"{path.name}: ASCII grid header has no cellsize")
    if "xllcenter" in header:
        xc = header["xllcenter"]
    elif "xllcorner" in header:
        xc = header["xllcorner"] + 0.5 * dx
    else:
        raise ValueError(f"{path.name}: ASCII grid header has no xllcorner / xllcenter")
    if "yllcenter" in header:
        yc = header["yllcenter"]
    elif "yllcorner" in header:
        yc = header["yllcorner"] + 0.5 * dy
    else:
        raise ValueError(f"{path.name}: ASCII grid header has no yllcorner / yllcenter")
    values = np.array(" ".join(lines[k:]).split(), dtype=np.float64)
    if len(values) != ncols * nrows:
        raise ValueError(f"{path.name}: {len(values)} values for {nrows} rows x {ncols} columns")
    z = values.reshape(nrows, ncols)
    nodata = header.get("nodata_value")
    if nodata is not None:
        z[z == nodata] = np.nan
    # first row is the northernmost: its centre is (nrows - 1) cells above the lower-left one
    y0 = yc + (nrows - 1) * dy
    info = {"format": "ESRI ASCII grid", "pixel_is": "area", "nodata": nodata, "vertical_crs": None}
    return Raster(z, float(xc), float(y0), float(dx), -float(dy), None, path.name, info)


# ------------------------------------------------------------------ resampling


def native_spacing(raster: Raster, latitude: float | None = None) -> float:
    """Cell size in metres: the smaller of the two sides (geographic: at ``latitude``)."""
    dx, dy = abs(raster.dx), abs(raster.dy)
    if raster.geographic:
        if latitude is None:
            nrow = raster.shape[0]
            latitude = raster.y0 + 0.5 * (nrow - 1) * raster.dy
        metres = 111_320.0
        dx *= metres * math.cos(math.radians(latitude))
        dy *= metres
    return float(min(dx, dy))


def grid_from_rasters(
    rasters: Sequence[Raster],
    rectangle: Sequence[float],
    spacing: float,
    crs: str | None,
) -> tuple[np.ndarray, tuple[float, float], dict[str, Any]]:
    """Heights on a regular grid over ``rectangle`` in ``crs`` from one or more rasters.

    Returns ``(z, origin, info)`` with ``z[i, j]`` the height at ``origin + (i, j) * spacing``
    (the layout of :func:`dtm_from_grid`). ``crs=None`` means the rasters' own coordinates
    (no transformation; every raster must then be without a CRS or share one).
    """
    if not rasters:
        raise ValueError("no raster given")
    xmin, ymin, xmax, ymax = (float(v) for v in rectangle)
    if not (xmax > xmin and ymax > ymin):
        raise ValueError("rectangle needs xmin < xmax and ymin < ymax")
    if not (np.isfinite(spacing) and spacing > 0):
        raise ValueError("spacing must be > 0")
    nx = int(math.floor((xmax - xmin) / spacing + 1e-9)) + 1
    ny = int(math.floor((ymax - ymin) / spacing + 1e-9)) + 1
    if nx < 2 or ny < 2:
        raise ValueError(f"the rectangle holds fewer than 2 x 2 nodes at spacing {spacing:g} m")
    if nx * ny > 25_000_000:
        raise ValueError(
            f"{nx} x {ny} nodes is too many: give a smaller rectangle or a larger spacing"
        )
    xs = xmin + np.arange(nx) * spacing
    ys = ymin + np.arange(ny) * spacing
    gx, gy = np.meshgrid(xs, ys, indexing="ij")
    z = np.full(gx.shape, np.nan)
    used = {r.source: 0 for r in rasters}
    edge_filled = 0

    def in_raster_crs(raster: Raster, px: np.ndarray, py: np.ndarray):
        if crs is not None and raster.crs is None:
            raise ValueError(
                f"{raster.source}: the raster has no CRS; give the CRS of the raster file"
            )
        if crs is not None and raster.crs != crs:
            from bakumine.core.drillholes.coordinates import reproject

            return reproject(px, py, crs, raster.crs)
        return px, py

    for raster in rasters:  # interior: the first raster that knows a node
        todo = ~np.isfinite(z)
        if not todo.any():
            break
        values = raster.sample(*in_raster_crs(raster, gx[todo], gy[todo]))
        z[todo] = values
        used[raster.source] += int(np.isfinite(values).sum())
    todo = ~np.isfinite(z)
    if todo.any():  # seams, outer half cells, cells next to no data: blend every raster
        total = np.zeros(int(todo.sum()))
        weights = np.zeros(int(todo.sum()))
        for raster in rasters:
            values, w = raster.sample_edge(*in_raster_crs(raster, gx[todo], gy[todo]))
            total += np.where(w > 0, values, 0.0) * w
            weights += w
            used[raster.source] += int((w > 0).sum())
        with np.errstate(invalid="ignore", divide="ignore"):
            z[todo] = np.where(weights > 0, total / weights, np.nan)
        edge_filled = int((weights > 0).sum())
    filled = int(np.isfinite(z).sum())
    info = {
        "nodes": [nx, ny],
        "spacing": float(spacing),
        "rectangle": [xmin, ymin, float(xs[-1]), float(ys[-1])],
        "nodes_with_height": filled,
        "nodes_without_height": int(z.size - filled),
        "nodes_from_edges": edge_filled,
        "rasters": [{"source": s, "nodes": n} for s, n in used.items()],
    }
    return z, (xmin, ymin), info


# ------------------------------------------------------------------ collars


def collar_residuals(surface: Any, collar: Any, shift: float = 0.0) -> Any:
    """Per collar: ``surface_z`` (after ``shift``) and ``dz = collar z - surface_z``.

    ``surface`` is a :class:`~bakumine.core.geometry.dtm.DTM`; ``collar`` a table with
    ``hole_id, x, y, z``. Returns a DataFrame; ``dz`` is NaN for collars off the surface.
    """
    import pandas as pd

    table = pd.DataFrame(
        {
            "hole_id": collar["hole_id"].astype(str).to_numpy(),
            "x": pd.to_numeric(collar["x"], errors="coerce").to_numpy(dtype=np.float64),
            "y": pd.to_numeric(collar["y"], errors="coerce").to_numpy(dtype=np.float64),
            "z": pd.to_numeric(collar["z"], errors="coerce").to_numpy(dtype=np.float64),
        }
    )
    surface_z = np.full(len(table), np.nan)
    known = np.isfinite(table[["x", "y"]].to_numpy()).all(axis=1)
    if known.any():
        surface_z[known] = surface.z_at(table.x.to_numpy()[known], table.y.to_numpy()[known])
    table["surface_z"] = surface_z + shift
    table["dz"] = table["z"] - table["surface_z"]
    return table


def residual_summary(table: Any, tolerance: float = 5.0) -> dict[str, Any]:
    """Statistics of ``dz`` and the collars off by more than ``tolerance`` from the median.

    ``median`` is the vertical shift that ties the surface to the collars. ``spread`` gives
    the residuals after that shift (``dz - median``): standard deviation, 90th percentile and
    maximum of their absolute value. ``suspect`` lists the collars beyond ``tolerance``,
    largest first.
    """
    dz = table["dz"].to_numpy(dtype=np.float64)
    ok = np.isfinite(dz)
    summary: dict[str, Any] = {
        "collars": int(len(dz)),
        "on_surface": int(ok.sum()),
        "off_surface": [str(h) for h in table["hole_id"][~ok]],
        "tolerance": float(tolerance),
    }
    if not ok.any():
        summary.update(median=None, mean=None, spread=None, suspect=[])
        return summary
    median = float(np.median(dz[ok]))
    r = dz[ok] - median
    order = np.argsort(-np.abs(r))
    ids = table["hole_id"].to_numpy()[ok]
    suspect = [
        {"hole_id": str(ids[i]), "dz": float(dz[ok][i]), "off_median": float(r[i])}
        for i in order
        if abs(r[i]) > tolerance
    ]
    summary.update(
        median=median,
        mean=float(dz[ok].mean()),
        spread={
            "std": float(r.std(ddof=1)) if len(r) > 1 else 0.0,
            "p90_abs": float(np.quantile(np.abs(r), 0.9)),
            "max_abs": float(np.abs(r).max()),
        },
        suspect=suspect,
    )
    return summary


# ------------------------------------------------------------------ survey + DEM


def _densify(lines: Sequence[np.ndarray], step: float) -> np.ndarray:
    """Points along every segment of ``lines`` no farther apart than ``step``."""
    out = []
    for line in lines:
        xy = np.asarray(line, dtype=np.float64).reshape(-1, 3)[:, :2]
        out.append(xy)
        for a, b in zip(xy[:-1], xy[1:], strict=True):
            n = int(math.ceil(float(np.hypot(*(b - a))) / step))
            if n > 1:
                t = np.arange(1, n)[:, None] / n
                out.append(a + t * (b - a))
    return np.concatenate(out) if out else np.zeros((0, 2))


def dem_fill(
    survey: np.ndarray,
    breaklines: Sequence[np.ndarray] | None,
    z: np.ndarray,
    origin: Sequence[float],
    spacing: float,
    *,
    drop_within: float | None = None,
    z_shift: float | str = "median",
) -> tuple[np.ndarray, dict[str, Any]]:
    """The DEM grid nodes to triangulate with a survey; see the module docstring.

    ``z, origin, spacing`` are the grid of :func:`grid_from_rasters`. ``survey`` holds the
    precise points ``(n, 3)``; ``breaklines`` their strings. ``drop_within`` (default: the
    grid spacing) is the plan distance inside which DEM nodes give way to the survey
    (breakline segments are sampled every ``drop_within / 4``, so a node within about 1 %
    of ``drop_within`` of a segment may stay).
    ``z_shift`` is ``"median"`` (tie to the survey), ``"none"`` or metres.

    Returns ``(points, info)``: the kept nodes (shifted) and ``info`` with ``nodes`` (grid
    nodes with a height), ``kept``, ``dropped``, ``drop_within``, ``z_shift`` and ``tie``
    (``points`` compared, ``median``, ``p90_abs`` and ``max_abs`` of ``survey z - DEM z``
    after the shift; ``None`` without a comparison).
    """
    from scipy.spatial import cKDTree

    from bakumine.core.geometry.dtm import dtm_from_grid

    survey = np.asarray(survey, dtype=np.float64).reshape(-1, 3)
    lines = [np.asarray(b, dtype=np.float64).reshape(-1, 3) for b in breaklines or ()]
    reference = np.concatenate([survey, *lines]) if lines else survey
    if not np.isfinite(reference).all():
        raise ValueError("survey points must be finite")
    z = np.asarray(z, dtype=np.float64)
    drop = float(spacing if drop_within is None else drop_within)
    if not (np.isfinite(drop) and drop >= 0):
        raise ValueError("drop_within must be >= 0")
    nx, ny = z.shape
    gx, gy = np.meshgrid(
        origin[0] + np.arange(nx) * spacing, origin[1] + np.arange(ny) * spacing, indexing="ij"
    )
    known = np.isfinite(z)
    nodes = np.column_stack([gx[known], gy[known], z[known]])

    tie = None
    if len(reference) and len(nodes) >= 3:
        try:
            dem_z = dtm_from_grid(z, origin, (spacing, spacing)).z_at(
                reference[:, 0], reference[:, 1]
            )
        except ValueError:  # no grid cell with four corners
            dem_z = np.full(len(reference), np.nan)
        dz = reference[:, 2] - dem_z
        dz = dz[np.isfinite(dz)]
        if len(dz):
            tie = {"points": int(len(dz)), "median": float(np.median(dz))}
    if isinstance(z_shift, str):
        if z_shift == "none":
            shift = 0.0
        elif z_shift == "median":
            if tie is None:
                raise ValueError(
                    "no survey point lies on the DEM, so the DEM cannot be tied to it: "
                    "give the vertical shift in metres, or none"
                )
            shift = tie["median"]
        else:
            raise ValueError("z_shift must be 'median', 'none' or a number")
    else:
        shift = float(z_shift)
        if not np.isfinite(shift):
            raise ValueError("z_shift must be finite")
    if tie is not None:
        r = np.abs(dz - shift)
        tie.update(p90_abs=float(np.quantile(r, 0.9)), max_abs=float(r.max()))

    keep = np.ones(len(nodes), dtype=bool)
    if drop > 0 and len(nodes) and len(reference):
        near = reference[:, :2]
        if lines:
            near = np.concatenate([near, _densify(lines, max(drop / 4.0, 1e-6))])
        distance = cKDTree(near).query(nodes[:, :2], distance_upper_bound=drop)[0]
        keep = ~(distance <= drop)
    points = nodes[keep].copy()
    points[:, 2] += shift
    info = {
        "nodes": int(len(nodes)),
        "kept": int(keep.sum()),
        "dropped": int((~keep).sum()),
        "drop_within": drop,
        "z_shift": shift,
        "tie": tie,
    }
    return points, info
