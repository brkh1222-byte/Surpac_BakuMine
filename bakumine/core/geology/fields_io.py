"""Store and reload a fitted field (geology_contracts.md section 7).

One ``.npz`` per field with the inputs of ``fit_field`` and its settings:

``xyz`` (n, 3), ``values`` (n,)
    the points and values the field was fitted to;
``kernel``, ``degree``, ``smoothing``, ``neighbors`` (-1 = one global system),
``anisotropy`` (JSON text of ``Ellipsoid.to_dict()``, or empty);
``centre``, ``length``, ``nodes``, ``weights``, ``poly``
    the solved system, for a global fit only.

A global field is rebuilt from the stored solution, so no system is solved again and the
values are identical. A local field (nearest-neighbour fits) is fitted again from ``xyz``
and ``values``, which is deterministic. Stored fields carry no clip (no topography, no
``max_extrapolation``): the geology model applies the clips.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geometry.implicit import ImplicitField, fit_field

__all__ = ["load_field", "load_field_inputs", "save_field"]

_GLOBAL_KEYS = ("centre", "length", "nodes", "weights", "poly")
_COMMON_KEYS = ("xyz", "values", "kernel", "degree", "smoothing", "neighbors", "anisotropy")


def save_field(path, field: ImplicitField, xyz: np.ndarray, values: np.ndarray) -> None:
    """Write ``field`` and the inputs it was fitted to (``xyz`` (n, 3), ``values`` (n,))."""
    xyz = np.asarray(xyz, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(xyz) != len(values):
        raise ValueError("xyz must have shape (n, 3) and values shape (n,)")
    anisotropy = "" if field.anisotropy is None else json.dumps(field.anisotropy.to_dict())
    data = {
        "xyz": xyz,
        "values": values,
        "kernel": np.array(field.kernel),
        "degree": np.array(int(field.degree), dtype=np.int64),
        "smoothing": np.array(float(field.smoothing), dtype=np.float64),
        "neighbors": np.array(
            -1 if field.neighbors is None else int(field.neighbors), dtype=np.int64
        ),
        "anisotropy": np.array(anisotropy),
    }
    if field.neighbors is None:
        if field.weights is None or field.poly is None:
            raise ValueError("a global field needs its weights and drift coefficients")
        data.update(
            centre=np.asarray(field.centre, dtype=np.float64),
            length=np.array(float(field.length), dtype=np.float64),
            nodes=np.asarray(field.nodes, dtype=np.float64),
            weights=np.asarray(field.weights, dtype=np.float64),
            poly=np.asarray(field.poly, dtype=np.float64),
        )
    path = Path(path)
    with path.open("wb") as handle:  # np.savez would append ".npz" to another suffix
        np.savez(handle, **data)


def _open(path) -> dict[str, np.ndarray]:
    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    missing = [key for key in _COMMON_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path}: not a stored field, missing {', '.join(missing)}")
    return data


def load_field_inputs(path) -> tuple[np.ndarray, np.ndarray]:
    """The ``(xyz, values)`` the stored field was fitted to."""
    data = _open(path)
    return data["xyz"], data["values"]


def load_field(path) -> ImplicitField:
    """Rebuild the field written by :func:`save_field`."""
    data = _open(path)
    text = str(data["anisotropy"])
    anisotropy = Ellipsoid.from_dict(json.loads(text)) if text else None
    kernel = str(data["kernel"])
    degree = int(data["degree"])
    smoothing = float(data["smoothing"])
    neighbors = int(data["neighbors"])
    if neighbors >= 0:
        return fit_field(
            data["xyz"], data["values"], anisotropy=anisotropy, kernel=kernel, degree=degree,
            smoothing=smoothing, neighbors=neighbors,
        )  # fmt: skip
    missing = [key for key in _GLOBAL_KEYS if key not in data]
    if missing:
        raise ValueError(f"{path}: a global field without {', '.join(missing)}")
    return ImplicitField(
        centre=data["centre"],
        anisotropy=anisotropy,
        length=float(data["length"]),
        nodes=np.ascontiguousarray(data["nodes"]),
        kernel=kernel,
        degree=degree,
        smoothing=smoothing,
        weights=np.ascontiguousarray(data["weights"]),
        poly=np.ascontiguousarray(data["poly"]),
        local=None,
        neighbors=None,
    )
