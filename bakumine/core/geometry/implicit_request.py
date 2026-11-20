"""``implicit_solid`` driven by a plain request dict (the CLI and the API share it).

Request keys::

    table, column              interval table and its categorical column
    codes                      one code or a list of codes that form the domain
    anisotropy                 {"mode": "plane", "dip", "dip_azimuth", "pitch"?, "ratios"}
                               {"mode": "angles", "angles": [bearing, plunge, dip], "ratios"}
                               {"mode": "auto"} | {"mode": "none"}
    max_extrapolation          metres from the nearest hole (optional)
    resolution                 grid cell across the body, metres (optional)
    min_component_volume       m3; smaller shells are removed (default 0)

``ratios`` is the extent along the major : semi-major : minor axis, for a plane without
pitch "along strike : down dip : across" (``"4:2:1"`` or three numbers). ``angles`` are in
the project's rotation convention (:mod:`bakumine.core.estimation.rotation`); a plane is
converted with :mod:`bakumine.core.estimation.orientation`.
"""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from bakumine.core.estimation.orientation import (
    angles_from_plane,
    describe_axes,
    parse_ratios,
)
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geometry.implicit import ImplicitResult, implicit_solid

__all__ = ["ANISOTROPY_MODES", "anisotropy_from_spec", "implicit_summary", "run_implicit"]

ANISOTROPY_MODES = ("plane", "angles", "auto", "none")


def anisotropy_from_spec(spec: Mapping[str, Any] | str | None) -> Ellipsoid | str | None:
    """The ``anisotropy`` argument of :func:`implicit_solid` from a request."""
    if spec is None:
        raise ValueError(
            "give the anisotropy: a plane (dip, dip azimuth), angles, 'auto' or 'none'"
        )
    if isinstance(spec, str):
        spec = {"mode": spec}
    if not isinstance(spec, Mapping):
        raise ValueError("anisotropy must be an object")
    mode = str(spec.get("mode") or "").lower()
    if mode not in ANISOTROPY_MODES:
        raise ValueError(f"anisotropy mode must be one of {', '.join(ANISOTROPY_MODES)}")
    if mode == "auto":
        return "auto"
    if mode == "none":
        return None
    if spec.get("ratios") in (None, "", []):
        raise ValueError("give the ratios major:semi-major:minor, for example 4:2:1")
    major, semi, minor = parse_ratios(spec["ratios"])
    if mode == "plane":
        if spec.get("dip") in (None, "") or spec.get("dip_azimuth") in (None, ""):
            raise ValueError("a plane needs dip and dip_azimuth")
        angles = angles_from_plane(
            float(spec["dip"]), float(spec["dip_azimuth"]), float(spec.get("pitch") or 0.0)
        )
    else:
        try:
            angles = tuple(float(v) for v in spec.get("angles") or ())
        except (TypeError, ValueError):
            angles = ()
        if len(angles) != 3:
            raise ValueError("angles needs three numbers (bearing, plunge, dip)")
    return Ellipsoid(1.0, major / semi, major / minor, angles)  # type: ignore[arg-type]


def _codes(value: Any) -> list[Any]:
    if value is None or value == "" or value == []:
        raise ValueError("give the code (or codes) of the domain")
    return list(value) if isinstance(value, list | tuple | set) else [value]


def implicit_summary(result: ImplicitResult, messages: list[str]) -> dict[str, Any]:
    """What the CLI prints and the dialog shows."""
    d = result.diagnostics
    anisotropy: dict[str, Any] = {
        "source": d["anisotropy_source"],
        "method": d["anisotropy_method"],
    }
    described = describe_axes(result.anisotropy)
    if described:
        anisotropy.update(described)
    if d.get("anisotropy_hole_angle_deg") is not None:
        anisotropy["hole_angle_deg"] = d["anisotropy_hole_angle_deg"]
    for key in ("confidence", "correlation", "candidate"):
        if d.get(f"anisotropy_{key}") is not None:
            anisotropy[key] = d[f"anisotropy_{key}"]
    return {
        "volume": float(d["volume"]),
        "valid": bool(d["mesh_valid"]),
        "outward": bool(d["mesh_outward"]),
        "issues": list(d["mesh_issues"] or []),
        "n_vertices": int(d["n_vertices"]),
        "n_faces": int(d["n_faces"]),
        "n_components": int(d["n_components"]),
        "n_components_removed": int(d["n_components_removed"]),
        "n_holes": int(d["n_holes"]),
        "n_holes_intersecting": int(d["n_holes_intersecting"]),
        "resolution": float(d["resolution"]),
        "max_extrapolation": d["max_extrapolation"],
        "touches_grid_boundary": bool(d["touches_grid_boundary"]),
        "anisotropy": anisotropy,
        "warnings": messages,
    }


def run_implicit(
    database: Any, request: Mapping[str, Any], topography: Any = None
) -> tuple[ImplicitResult, dict[str, Any]]:
    """Build the solid of ``request`` from ``database``; returns it with its summary.

    ``topography`` is a ``DTM`` or its mesh (the caller resolves the request's name).
    Warnings raised on the way (automatic anisotropy, grid coarsening) are returned in
    ``summary["warnings"]``. Raises ``ValueError`` for a bad request and when no solid
    comes out.
    """

    def number(key: str, default: float | None = None) -> float | None:
        value = request.get(key)
        if value in (None, ""):
            return default
        try:
            return float(value)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be a number") from None

    table = str(request.get("table") or "geology")
    column = request.get("column")
    if not column:
        raise ValueError("give the column that holds the codes")
    if table not in database.intervals:
        raise ValueError(f"no interval table {table!r}; database has {sorted(database.intervals)}")
    frame = database.intervals[table]
    if column not in frame.columns:
        raise ValueError(f"table {table!r} has no column {column!r}")
    codes = _codes(request.get("codes", request.get("code")))
    present = set(frame[column].dropna().astype(str))
    unknown = [c for c in codes if str(c) not in present]
    if unknown:
        raise ValueError(f"column {column!r} has no code {unknown}; it has {sorted(present)[:20]}")
    anisotropy = anisotropy_from_spec(request.get("anisotropy"))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = implicit_solid(
            database,
            table,
            str(column),
            codes if len(codes) > 1 else codes[0],
            anisotropy=anisotropy,
            resolution=number("resolution"),
            max_extrapolation=number("max_extrapolation"),
            topography=topography,
            min_component_volume=number("min_component_volume", 0.0) or 0.0,
        )
    messages = list(
        dict.fromkeys(str(w.message) for w in caught if issubclass(w.category, UserWarning))
    )
    if result.mesh.n_faces == 0:
        raise ValueError("the solid is empty (clipped away or below the grid resolution)")
    return result, implicit_summary(result, messages)
