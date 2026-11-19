"""A tiny in-memory scene and the web decoder fixtures generated from it.

``uv run python tests/api/scene_fixtures.py`` rewrites ``web/src/api/fixtures/``:
``drillholes.bkma`` and ``mesh.bkma`` (what the endpoints return for the tiny scene) and
``expected.json`` (the same content as plain numbers, for the vitest round trip).
``test_scene_api.py`` checks that the committed files still match this generator.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.api.binary import unpack_arrays
from bakumine.api.payloads import drillhole_payload, mesh_payload
from bakumine.api.scene import DrillholeData, ObjectInfo
from bakumine.core.drillholes.importer import TableSource, import_drillholes
from bakumine.core.geometry.mesh import Mesh

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "web" / "src" / "api" / "fixtures"

_IDENTITY = {
    "collar": ["hole_id", "x", "y", "z", "max_depth"],
    "survey": ["hole_id", "depth", "dip", "azimuth"],
    "assay": ["hole_id", "depth_from", "depth_to", "au", "rock"],
}


def tiny_tables() -> dict[str, pd.DataFrame]:
    """Three holes at mine coordinates: one vertical, one curved, one inclined."""
    collar = pd.DataFrame(
        [
            ("T1", 500100.0, 5300200.0, 1210.5, 40.0),
            ("T2", 500150.25, 5300210.5, 1208.0, 90.0),
            ("T3", 500200.0, 5300190.0, 1205.25, 30.0),
        ],
        columns=_IDENTITY["collar"],
    )
    survey = pd.DataFrame(
        [
            ("T1", 0.0, -90.0, 0.0),
            ("T2", 0.0, -60.0, 90.0),
            ("T2", 30.0, -50.0, 100.0),
            ("T2", 60.0, -38.0, 112.0),
            ("T3", 0.0, -70.0, 270.0),
        ],
        columns=_IDENTITY["survey"],
    )
    assay = pd.DataFrame(
        [
            # Not in hole order on purpose; one missing grade, one missing rock code,
            # a gap in T2 between 20 and 45 m.
            ("T2", 0.0, 20.0, 0.25, "WASTE"),
            ("T1", 0.0, 10.0, 0.031, "WASTE"),
            ("T1", 10.0, 25.0, None, "ORE"),
            ("T2", 45.0, 70.0, 3.5, "ORE"),
            ("T1", 25.0, 40.0, 1.75, None),
            ("T2", 70.0, 90.0, 0.4, "DYKE"),
            ("T3", 5.0, 30.0, 12.125, "ORE"),
        ],
        columns=_IDENTITY["assay"],
    )
    return {"collar": collar, "survey": survey, "assay": assay}


def tiny_drillholes() -> DrillholeData:
    tables = tiny_tables()
    source = {name: TableSource(tables[name], {c: c for c in cols}) for name, cols in
              _IDENTITY.items()}
    database = import_drillholes(source["collar"], source["survey"], {"assay": source["assay"]})
    return DrillholeData(database, database.desurvey(), {"au": "g/t"})


def tiny_mesh() -> Mesh:
    """A tetrahedron at mine coordinates (closed, outward winding)."""
    vertices = np.array(
        [
            [500100.0, 5300200.0, 1000.0],
            [500110.0, 5300200.0, 1000.0],
            [500100.0, 5300212.5, 1000.0],
            [500100.0, 5300200.0, 1007.25],
        ]
    )
    faces = np.array([[0, 2, 1], [0, 1, 3], [1, 2, 3], [0, 3, 2]])
    return Mesh(vertices, faces)


class TinySceneSource:
    """In-memory :class:`bakumine.api.scene.SceneSource` used by the endpoint tests."""

    project_id = "tiny"
    project_name = "Tiny test project"
    version = "1"

    def __init__(self) -> None:
        self._drillholes = tiny_drillholes()
        self._mesh = tiny_mesh()

    def objects(self) -> list[ObjectInfo]:
        return [
            ObjectInfo("tetra", "Tetrahedron", "solids", "mesh"),
            ObjectInfo("holes", "Holes", "drillholes", "drillholes"),
        ]

    def drillholes(self, object_id: str) -> DrillholeData:
        if object_id != "holes":
            raise KeyError(object_id)
        return self._drillholes

    def mesh(self, object_id: str) -> Mesh:
        if object_id != "tetra":
            raise KeyError(object_id)
        return self._mesh


def _plain(arrays: dict[str, np.ndarray]) -> dict[str, Any]:
    """Arrays as nested lists; NaN becomes null (JSON has no NaN)."""
    out: dict[str, Any] = {}
    for name, array in arrays.items():
        values = array.astype(object)
        if array.dtype.kind == "f":
            values[np.isnan(array)] = None
        out[name] = {"dtype": array.dtype.kind + str(array.dtype.itemsize), "data": values.tolist()}
    return out


def build_fixtures() -> dict[str, bytes]:
    """File name -> content of every web fixture."""
    source = TinySceneSource()
    drillholes = drillhole_payload(source.drillholes("holes"), "holes", "assay")
    mesh = mesh_payload(source.mesh("tetra"), "tetra", normals=True)
    expected = {}
    for name, payload in (("drillholes", drillholes), ("mesh", mesh)):
        header, arrays = unpack_arrays(payload)
        expected[name] = {"kind": header["kind"], "meta": header["meta"], "arrays": _plain(arrays)}
    return {
        "drillholes.bkma": drillholes,
        "mesh.bkma": mesh,
        "expected.json": (json.dumps(expected, indent=1) + "\n").encode(),
    }


if __name__ == "__main__":
    FIXTURES.mkdir(parents=True, exist_ok=True)
    for file_name, content in build_fixtures().items():
        (FIXTURES / file_name).write_bytes(content)
        print(f"{file_name}: {len(content)} bytes")
