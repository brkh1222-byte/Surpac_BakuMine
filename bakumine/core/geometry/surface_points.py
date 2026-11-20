"""A topography (or any single-valued surface) from scattered points.

Sources of points, each a plain ``(n, 3)`` array of (east, north, up):

* :func:`collar_points` - the collars of a drillhole database;
* :func:`read_points_table` - a CSV / XLSX table with x, y, z columns;
* :func:`string_points` / :func:`string_lines` - the points of a string file, or its
  segments as breaklines.

:func:`surface_from_points` joins them, optionally adds boundary points on a rectangle,
and triangulates with :func:`bakumine.core.geometry.dtm.dtm_from_points`.

Extension to a rectangle
------------------------
Collars cover only the drilled area; a block model usually reaches further. With
``rectangle=(xmin, ymin, xmax, ymax)`` and / or ``padding`` (metres added on every side
of the rectangle, or of the box of the data when no rectangle is given) points are added
on the rectangle: its four corners and points along each
side at equal steps no longer than ``step`` (default: the median distance from a data
point to its nearest neighbour in plan, but at least a hundredth of the longer side).
A boundary point that falls inside the convex hull of the data is left out, so the
rectangle never changes the surface where there are data. The elevation of a boundary
point is

* ``extend_z="nearest"`` (default): the elevation of the data point nearest to it in
  plan - a flat continuation of the edge of the data, no extrapolated slope;
* a number: that elevation everywhere on the rectangle.

Data outside the rectangle are kept; the surface then reaches beyond it there.

Duplicates
----------
Points with the same XY and the same Z are merged silently. Points with the same XY and
different Z follow ``duplicates`` (``mean`` default, ``first``, ``last``, ``min``, ``max``
or ``error``), as in :func:`dtm_from_points`; their number is reported.
"""

from __future__ import annotations

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.core.geometry.dtm import DTM, DuplicatePointWarning, dtm_from_points

__all__ = [
    "POINT_COLUMN_NAMES",
    "SurfaceResult",
    "collar_points",
    "extension_points",
    "model_rectangle",
    "read_points_table",
    "string_lines",
    "string_points",
    "surface_from_points",
]

#: accepted (lower-case) header names of the coordinate columns of a points table
POINT_COLUMN_NAMES: dict[str, tuple[str, ...]] = {
    "x": ("x", "east", "easting", "e", "xcoord", "x_coord", "xcollar"),
    "y": ("y", "north", "northing", "n", "ycoord", "y_coord", "ycollar"),
    "z": ("z", "rl", "elev", "elevation", "level", "height", "zcoord", "z_coord", "zcollar"),
}


@dataclass
class SurfaceResult:
    """The surface and what went into it."""

    dtm: DTM
    info: dict[str, Any] = field(default_factory=dict)


def collar_points(database: Any) -> np.ndarray:
    """Collar coordinates of a drillhole database, rows with a missing value left out."""
    xyz = database.collar[["x", "y", "z"]].to_numpy(dtype=np.float64)
    return xyz[np.isfinite(xyz).all(axis=1)]


def read_points_table(
    path: str | Path,
    columns: Sequence[str] | None = None,
    *,
    sheet: str | int | None = None,
    delimiter: str | None = ",",
) -> tuple[np.ndarray, dict[str, Any]]:
    """Points of a CSV / XLSX table.

    ``columns`` names the x, y and z columns (east, north, up - give them in that order
    whatever the order in the file). Default: found by header name
    (:data:`POINT_COLUMN_NAMES`, case-insensitive). Rows where a coordinate is blank or
    not a number are left out and counted.

    Returns ``(points, info)`` with ``info = {columns, rows, skipped}``.
    """
    import pandas as pd

    from bakumine.core.drillholes.importer import read_raw_table

    raw = read_raw_table(path, sheet=sheet, delimiter=delimiter)
    names = [str(c) for c in raw.columns]
    if columns is None:
        lower = {n.strip().lower(): n for n in reversed(names)}
        chosen = []
        for axis, accepted in POINT_COLUMN_NAMES.items():
            found = next((lower[a] for a in accepted if a in lower), None)
            if found is None:
                raise ValueError(
                    f"{Path(path).name}: no {axis} column found among {names}; name the "
                    "x, y, z columns"
                )
            chosen.append(found)
    else:
        chosen = [str(c) for c in columns]
        if len(chosen) != 3:
            raise ValueError("columns needs three names: x, y, z")
        missing = [c for c in chosen if c not in names]
        if missing:
            raise ValueError(f"{Path(path).name}: no column {missing}; the file has {names}")
    raw.columns = names
    values = np.column_stack(
        [pd.to_numeric(raw[c].replace("", np.nan), errors="coerce") for c in chosen]
    ).astype(np.float64)
    keep = np.isfinite(values).all(axis=1)
    info = {"columns": chosen, "rows": int(len(values)), "skipped": int((~keep).sum())}
    return values[keep], info


def string_points(strings: Any) -> np.ndarray:
    """All points of a :class:`~bakumine.core.io.strings.StringFile`."""
    if not len(strings.segments):
        return np.zeros((0, 3))
    return np.asarray(strings.all_points(), dtype=np.float64).reshape(-1, 3)


def string_lines(strings: Any) -> list[np.ndarray]:
    """The segments of a string file with two or more points, for use as breaklines."""
    lines = []
    for segment in strings.segments:
        points = np.asarray(segment.points, dtype=np.float64).reshape(-1, 3)
        if len(points) >= 2:
            lines.append(points)
    return lines


def model_rectangle(definition: Any) -> list[float]:
    """Plan box ``[xmin, ymin, xmax, ymax]`` of a block model definition."""
    box = np.asarray(definition.bounding_box(), dtype=np.float64)
    return [float(box[0, 0]), float(box[0, 1]), float(box[1, 0]), float(box[1, 1])]


def _rectangle(value: Any) -> tuple[float, float, float, float]:
    try:
        xmin, ymin, xmax, ymax = (float(v) for v in value)
    except (TypeError, ValueError):
        raise ValueError("rectangle needs four numbers: xmin, ymin, xmax, ymax") from None
    if not (np.isfinite([xmin, ymin, xmax, ymax]).all() and xmax > xmin and ymax > ymin):
        raise ValueError("rectangle needs xmin < xmax and ymin < ymax")
    return xmin, ymin, xmax, ymax


def extension_points(
    points: np.ndarray,
    rectangle: Sequence[float],
    extend_z: float | str = "nearest",
    step: float | None = None,
) -> np.ndarray:
    """Boundary points on ``rectangle`` for the data ``points``; see the module docstring."""
    from scipy.spatial import Delaunay, QhullError, cKDTree

    xmin, ymin, xmax, ymax = _rectangle(rectangle)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points) == 0:
        raise ValueError("no data points to extend")
    xy = np.unique(points[:, :2], axis=0)
    tree = cKDTree(points[:, :2])
    longest = max(xmax - xmin, ymax - ymin)
    if step is None:
        if len(xy) > 1:
            step = float(np.median(cKDTree(xy).query(xy, k=2)[0][:, 1]))
        else:
            step = longest
    if not (np.isfinite(step) and step > 0):
        raise ValueError("step must be > 0")
    step = max(float(step), longest / 100.0)
    nx = max(1, int(np.ceil((xmax - xmin) / step)))
    ny = max(1, int(np.ceil((ymax - ymin) / step)))
    xs = np.linspace(xmin, xmax, nx + 1)
    ys = np.linspace(ymin, ymax, ny + 1)
    ring = np.concatenate(
        [
            np.column_stack([xs, np.full(nx + 1, ymin)]),
            np.column_stack([xs, np.full(nx + 1, ymax)]),
            np.column_stack([np.full(ny - 1, xmin), ys[1:-1]]),
            np.column_stack([np.full(ny - 1, xmax), ys[1:-1]]),
        ]
    )
    if len(xy) >= 3:
        try:
            centre = xy.mean(axis=0)
            hull = Delaunay(xy - centre)
            ring = ring[hull.find_simplex(ring - centre) < 0]
        except QhullError:  # collinear data: no hull, nothing is inside it
            pass
    if isinstance(extend_z, str):
        if extend_z != "nearest":
            raise ValueError("extend_z must be a number or 'nearest'")
        z = points[tree.query(ring)[1], 2] if len(ring) else np.zeros(0)
    else:
        if not np.isfinite(float(extend_z)):
            raise ValueError("extend_z must be a finite number or 'nearest'")
        z = np.full(len(ring), float(extend_z))
    return np.column_stack([ring, z])


def surface_from_points(
    points: Any,
    breaklines: Sequence[Any] | None = None,
    *,
    rectangle: Sequence[float] | None = None,
    padding: float | None = None,
    extend_z: float | str = "nearest",
    step: float | None = None,
    duplicates: str = "mean",
) -> SurfaceResult:
    """Triangulate ``points`` (and ``breaklines``) into a DTM, optionally extended.

    Parameters
    ----------
    points : (n, 3) array_like, or a list of such arrays (several sources).
    breaklines : sequence of (k, 3) arrays; their segments become triangle edges.
    rectangle : ``(xmin, ymin, xmax, ymax)`` to extend the surface to (for example the
        plan box of a block model).
    padding : metres added on every side of ``rectangle``; without a rectangle, of the
        box of the data.
    extend_z, step : elevation rule and spacing of the added boundary points.
    duplicates : rule for points with equal XY and different Z.

    ``result.info`` has ``n_points`` (given), ``n_breaklines``, ``n_duplicates`` (points
    merged because another point has the same XY), ``n_conflicts`` (XY locations whose
    points disagree in Z), ``duplicates`` (the rule), ``n_extension_points``,
    ``rectangle``, ``extend_z``, ``n_vertices``, ``n_faces``, ``bounds``, ``plan_area``.
    """
    if isinstance(points, list | tuple) and len(points) and np.ndim(points[0]) == 2:
        parts = [np.asarray(p, dtype=np.float64).reshape(-1, 3) for p in points]
        data = np.concatenate(parts) if parts else np.zeros((0, 3))
    else:
        data = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(data).all():
        raise ValueError("points must be finite")
    lines = [np.asarray(b, dtype=np.float64).reshape(-1, 3) for b in breaklines or ()]
    everything = np.concatenate([data, *lines]) if lines else data
    if len(everything) < 3:
        raise ValueError(f"a surface needs at least 3 points, got {len(everything)}")
    if padding is not None and not (np.isfinite(padding) and padding >= 0):
        raise ValueError("padding must be >= 0")
    if rectangle is not None:
        xmin, ymin, xmax, ymax = _rectangle(rectangle)
        pad = float(padding or 0.0)
        rectangle = (xmin - pad, ymin - pad, xmax + pad, ymax + pad)
    elif padding is not None:
        if not padding > 0:
            raise ValueError("padding must be > 0 when no rectangle is given")
        lo, hi = everything[:, :2].min(axis=0), everything[:, :2].max(axis=0)
        rectangle = (lo[0] - padding, lo[1] - padding, hi[0] + padding, hi[1] + padding)
    extra = np.zeros((0, 3))
    if rectangle is not None:
        extra = extension_points(everything, rectangle, extend_z, step)

    # duplicate statistics (the merge itself is done by dtm_from_points)
    _, inverse, counts = np.unique(
        everything[:, :2], axis=0, return_inverse=True, return_counts=True
    )
    inverse = np.asarray(inverse).reshape(-1)
    z_min = np.full(len(counts), np.inf)
    z_max = np.full(len(counts), -np.inf)
    np.minimum.at(z_min, inverse, everything[:, 2])
    np.maximum.at(z_max, inverse, everything[:, 2])
    n_conflicts = int((z_max > z_min).sum())

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DuplicatePointWarning)
        dtm = dtm_from_points(
            np.concatenate([data, extra]) if len(extra) else data,
            lines or None,
            duplicates=duplicates,
        )
    bounds = dtm.mesh.vertices.min(axis=0), dtm.mesh.vertices.max(axis=0)
    info = {
        "n_points": int(len(data)),
        "n_breaklines": len(lines),
        "n_breakline_points": int(sum(len(b) for b in lines)),
        "n_duplicates": int(len(everything) - len(counts)),
        "n_conflicts": n_conflicts,
        "duplicates": duplicates,
        "n_extension_points": int(len(extra)),
        "rectangle": None if rectangle is None else [float(v) for v in rectangle],
        "extend_z": extend_z if isinstance(extend_z, str) else float(extend_z),
        "n_vertices": int(dtm.mesh.n_vertices),
        "n_faces": int(dtm.mesh.n_faces),
        "bounds": [[float(v) for v in bounds[0]], [float(v) for v in bounds[1]]],
        "plan_area": float(dtm.plan_area),
    }
    if rectangle is None:
        info["extend_z"] = None
    return SurfaceResult(dtm, info)
