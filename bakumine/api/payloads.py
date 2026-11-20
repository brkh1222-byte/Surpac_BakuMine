"""Builds the scene manifest and the BKMA payloads from core objects.

Coordinates are WORLD float64 (east, north, up; metres) in every payload. They are not
reduced to an origin + float32 offsets on the wire: the viewer subtracts its own scene
origin in float64 (``web/src/scene/origin.ts``), so the precision is lost exactly once,
and the payloads can be checked bit for bit against ``HoleTraces.xyz_at`` and the source
arrays. gzip removes most of the redundancy of the shared high-order bytes.

Drillhole payload (``kind = "drillholes"``), H holes, P polyline points, S segments::

    hole_offsets   u4 (H + 1)   points of hole h are rows hole_offsets[h]:hole_offsets[h+1]
    point_depth    f8 (P)       downhole depth of each polyline point
    point_xyz      f8 (P, 3)    desurveyed position at that depth (xyz_at)
    collar_xyz     f8 (H, 3)    collar position
    max_depth      f8 (H)
    -- only with ?table=<interval table> --
    seg_hole       u4 (S)       index into meta.hole_ids
    seg_row        u4 (S)       row of the interval in the source table (file order)
    seg_from       f8 (S)       depth_from
    seg_to         f8 (S)       depth_to
    seg_start      f8 (S, 3)    xyz_at(hole, depth_from)
    seg_end        f8 (S, 3)    xyz_at(hole, depth_to)
    value:<name>   f8 (S)       numeric attribute; NaN = missing
    code:<name>    i4 (S)       categorical attribute, index into its categories; -1 = missing

    meta = {object_id, hole_ids: [str], n_holes, n_points, tolerance, coordinates,
            table, n_segments, dropped_segments,
            attributes: [{name, type: "numeric", array, unit} |
                         {name, type: "categorical", array, categories: [str]}]}

Mesh payload (``kind = "mesh"``)::

    vertices       f8 (V, 3)
    faces          u4 (F, 3)
    normals        f4 (V, 3)    area-weighted unit vertex normals; only with ?normals=1

    meta = {object_id, n_vertices, n_faces, closed, coordinates}

Block payload (``kind = "blocks"``): see :mod:`bakumine.api.blocks`. It carries no
coordinates at all: the client rebuilds them from the model definition.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from bakumine.api.binary import pack_arrays
from bakumine.api.scene import DrillholeData, ObjectInfo, SceneSource
from bakumine.core.drillholes.desurvey import HoleTraces
from bakumine.core.geometry.mesh import Mesh

COORDINATES = "world float64: x = east, y = north, z = up; metres"
#: Largest distance between the polyline and the true arc, metres (default).
DEFAULT_TOLERANCE = 0.05
_MAX_SUBDIVISIONS = 256


# ---------------------------------------------------------------------- drillholes


def densify_traces(
    traces: HoleTraces, tolerance: float = DEFAULT_TOLERANCE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Polyline of every hole, dense enough that curved holes look curved.

    Each interval between two stations is cut into ``n`` equal parts so that the chord
    stays within ``tolerance`` of a circular arc: an arc of length ``L`` turning by
    ``theta`` has sagitta ``L * theta / 8``, hence ``n = ceil(sqrt(L * theta / (8 tol)))``.
    Straight intervals are not subdivided. Every station is a polyline point.

    Returns ``(hole_offsets (H + 1,), depth (P,), xyz (P, 3))``; positions come from
    ``traces.xyz_at`` at those depths.
    """
    if not tolerance > 0:
        raise ValueError("tolerance must be positive")
    stations = traces.stations()
    n_holes = len(traces.hole_ids)
    if len(stations) == 0:
        return np.zeros(n_holes + 1, dtype=np.uint32), np.zeros(0), np.zeros((0, 3))
    hole = pd.Index(traces.hole_ids).get_indexer(stations["hole_id"].to_numpy(dtype=object))
    depth = stations["depth"].to_numpy(dtype=np.float64)
    direction = stations[["dx", "dy", "dz"]].to_numpy(dtype=np.float64)

    # Interval i runs from station i to station i + 1 when both belong to the same hole.
    same = hole[1:] == hole[:-1]
    length = np.where(same, depth[1:] - depth[:-1], 0.0)
    cosine = np.clip(np.einsum("ij,ij->i", direction[1:], direction[:-1]), -1.0, 1.0)
    theta = np.where(same, np.arccos(cosine), 0.0)
    parts = np.ceil(np.sqrt(np.maximum(length, 0.0) * theta / (8.0 * tolerance)))
    parts = np.clip(parts, 1, _MAX_SUBDIVISIONS).astype(np.int64)
    # Station i emits itself plus the interior points of its interval; the last station
    # of a hole (and zero-length intervals) emit only themselves.
    emits = np.ones(len(depth), dtype=np.int64)
    emits[:-1] = np.where(same & (length > 0), parts, 1)

    first = np.repeat(np.arange(len(depth)), emits)
    step = np.arange(len(first)) - np.repeat(np.cumsum(emits) - emits, emits)
    span = np.zeros(len(depth))
    span[:-1] = length
    point_depth = depth[first] + span[first] * step / emits[first]
    point_hole = hole[first]
    xyz, _ = traces.xyz_at(traces.hole_ids[point_hole], point_depth, on_invalid="raise")
    offsets = np.searchsorted(point_hole, np.arange(n_holes + 1)).astype(np.uint32)
    return offsets, point_depth, xyz


def _is_numeric(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def _categories(series: pd.Series) -> list[str]:
    return sorted({str(v) for v in series.dropna().unique()})


def attribute_summaries(data: DrillholeData) -> list[dict[str, Any]]:
    """Manifest entries for every attribute column of every interval table."""
    out: list[dict[str, Any]] = []
    for table, frame in data.database.intervals.items():
        for name in data.database.attribute_columns(table):
            series = frame[name]
            entry: dict[str, Any] = {
                "table": table,
                "name": name,
                "count": int(series.notna().sum()),
                "missing": int(series.isna().sum()),
            }
            if _is_numeric(series):
                values = series.to_numpy(dtype=np.float64)
                values = values[np.isfinite(values)]
                entry["type"] = "numeric"
                entry["unit"] = data.units.get(name)
                if values.size:
                    low, high = np.percentile(values, [2.0, 98.0])
                    entry["min"] = float(values.min())
                    entry["max"] = float(values.max())
                    # A robust range for the default legend: grades are long-tailed.
                    entry["display_min"] = float(low)
                    entry["display_max"] = float(high)
                else:
                    entry.update(min=None, max=None, display_min=None, display_max=None)
            else:
                entry["type"] = "categorical"
                entry["categories"] = _categories(series)
            out.append(entry)
    return out


def drillhole_payload(
    data: DrillholeData,
    object_id: str,
    table: str | None = None,
    tolerance: float = DEFAULT_TOLERANCE,
) -> bytes:
    """Traces and collars of every hole and, with ``table``, that table's intervals."""
    database, traces = data.database, data.traces
    if table is not None and table not in database.intervals:
        raise KeyError(table)
    offsets, point_depth, point_xyz = densify_traces(traces, tolerance)
    hole_ids = np.asarray(traces.hole_ids, dtype=object)
    collar = database.collar.set_index("hole_id").loc[hole_ids, ["x", "y", "z"]]
    arrays: dict[str, np.ndarray] = {
        "hole_offsets": offsets,
        "point_depth": point_depth,
        "point_xyz": point_xyz,
        "collar_xyz": collar.to_numpy(dtype=np.float64),
        "max_depth": np.asarray(traces.max_depth, dtype=np.float64),
    }
    meta: dict[str, Any] = {
        "object_id": object_id,
        "hole_ids": [str(h) for h in hole_ids],
        "n_holes": int(len(hole_ids)),
        "n_points": int(len(point_depth)),
        "tolerance": float(tolerance),
        "coordinates": COORDINATES,
        "table": table,
    }
    if table is not None:
        frame = database.intervals[table]
        ids = frame["hole_id"].to_numpy(dtype=object)
        depth_from = frame["depth_from"].to_numpy(dtype=np.float64)
        depth_to = frame["depth_to"].to_numpy(dtype=np.float64)
        hole = pd.Index(hole_ids).get_indexer(ids)
        start, _ = traces.xyz_at(ids, depth_from)
        end, _ = traces.xyz_at(ids, depth_to)
        # Intervals of holes without a trace, or outside [0, max_depth], cannot be drawn.
        keep = (hole >= 0) & np.isfinite(start).all(axis=1) & np.isfinite(end).all(axis=1)
        arrays.update(
            seg_hole=hole[keep].astype(np.uint32),
            seg_row=np.flatnonzero(keep).astype(np.uint32),
            seg_from=depth_from[keep],
            seg_to=depth_to[keep],
            seg_start=start[keep],
            seg_end=end[keep],
        )
        attributes: list[dict[str, Any]] = []
        for name in database.attribute_columns(table):
            series = frame[name]
            if _is_numeric(series):
                key = f"value:{name}"
                arrays[key] = series.to_numpy(dtype=np.float64, na_value=np.nan)[keep]
                attributes.append(
                    {"name": name, "type": "numeric", "array": key, "unit": data.units.get(name)}
                )
            else:
                key = f"code:{name}"
                categories = _categories(series)
                lookup = {c: i for i, c in enumerate(categories)}
                codes = np.array(
                    [-1 if pd.isna(v) else lookup[str(v)] for v in series.to_numpy(dtype=object)],
                    dtype=np.int32,
                )
                arrays[key] = codes[keep]
                attributes.append(
                    {"name": name, "type": "categorical", "array": key, "categories": categories}
                )
        meta.update(
            n_segments=int(keep.sum()),
            dropped_segments=int((~keep).sum()),
            attributes=attributes,
        )
    return pack_arrays("drillholes", meta, arrays)


def _drillhole_bounds(data: DrillholeData) -> dict[str, list[float]] | None:
    xyz = data.traces.stations()[["x", "y", "z"]].to_numpy(dtype=np.float64)
    if len(xyz) == 0:
        return None
    return {"min": xyz.min(axis=0).tolist(), "max": xyz.max(axis=0).tolist()}


# ---------------------------------------------------------------------- meshes


def mesh_payload(mesh: Mesh, object_id: str, normals: bool = False) -> bytes:
    """Vertices, faces and (optionally) per-vertex normals of one mesh."""
    if mesh.n_vertices > np.iinfo(np.uint32).max:
        raise ValueError("mesh has too many vertices for uint32 face indices")
    arrays: dict[str, np.ndarray] = {
        "vertices": mesh.vertices,
        "faces": mesh.faces.astype(np.uint32),
    }
    if normals:
        arrays["normals"] = mesh.vertex_normals.astype(np.float32)
    meta = {
        "object_id": object_id,
        "n_vertices": int(mesh.n_vertices),
        "n_faces": int(mesh.n_faces),
        "closed": bool(mesh.is_closed),
        "coordinates": COORDINATES,
    }
    return pack_arrays("mesh", meta, arrays)


# ---------------------------------------------------------------------- manifest


def _object_entry(source: SceneSource, info: ObjectInfo) -> dict[str, Any]:
    base = f"/api/v1/scene/objects/{info.id}"
    entry: dict[str, Any] = {
        "id": info.id,
        "name": info.name,
        "category": info.category,
        "kind": info.kind,
        "attributes": [],
    }
    if info.kind == "drillholes":
        data = source.drillholes(info.id)
        entry["href"] = f"{base}/drillholes"
        entry["bounds"] = _drillhole_bounds(data)
        entry["attributes"] = attribute_summaries(data)
        entry["stats"] = {
            "holes": int(len(data.traces.hole_ids)),
            "tables": {name: int(len(t)) for name, t in data.database.intervals.items()},
            "length_unit": data.database.length_unit,
        }
    elif info.kind == "mesh":
        mesh = source.mesh(info.id)
        entry["href"] = f"{base}/mesh"
        low, high = (mesh.vertices.min(axis=0), mesh.vertices.max(axis=0))
        entry["bounds"] = {"min": low.tolist(), "max": high.tolist()} if mesh.n_vertices else None
        entry["stats"] = {
            "vertices": int(mesh.n_vertices),
            "faces": int(mesh.n_faces),
            "closed": bool(mesh.is_closed),
        }
    else:
        from bakumine.api.blocks import block_object_entry

        entry["href"] = f"{base}/blocks"
        entry.update(block_object_entry(source.blocks(info.id)))
    return entry


def scene_manifest(source: SceneSource) -> dict[str, Any]:
    """JSON description of the scene: the objects and where to fetch their data."""
    return {
        "api_version": 1,
        "project": {"id": source.project_id, "name": source.project_name},
        "version": source.version,
        "coordinates": COORDINATES,
        "objects": [_object_entry(source, info) for info in source.objects()],
    }
