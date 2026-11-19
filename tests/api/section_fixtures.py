"""Reference sections of the synthetic deposit for the web viewer's outline code.

``uv run python tests/api/section_fixtures.py`` rewrites
``web/src/section/fixtures/sections.json``: polylines cut by
:func:`bakumine.core.geometry.section.section` from the demo meshes
(``data/synthetic/default``) on a few planes. The viewer computes the same outlines in
the browser (``web/src/section/outline.ts``); its vitest compares against this file, and
the Playwright section test projects the W-E outline of the ore solid onto the screen.
``test_section_fixtures.py`` checks that the committed file still matches this generator.

Planes are given the way the viewer defines a section: a centre, the azimuth of the
section line (degrees clockwise from north) and the dip of the plane (90 = vertical).
The normal points towards the viewer: ``u = (sin az, cos az, 0)`` is the section line,
``n = cos(t) (cos az, -sin az, 0) + sin(t) z`` with ``t = 90 - dip``.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.section import section

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "data" / "synthetic" / "default"
FIXTURE = REPO / "web" / "src" / "section" / "fixtures" / "sections.json"

#: name, mesh, azimuth, dip, offset of the centre from the middle of the ore solid (m)
CASES = [
    ("ore W-E through the middle", "ore_solid", 90.0, 90.0, (0.0, 0.0, 0.0)),
    ("ore W-E 100 m north", "ore_solid", 90.0, 90.0, (0.0, 100.0, 0.0)),
    ("ore S-N through the middle", "ore_solid", 0.0, 90.0, (0.0, 0.0, 0.0)),
    ("ore plan through the middle", "ore_solid", 90.0, 0.0, (0.0, 0.0, 0.0)),
    ("ore oblique, dipping", "ore_solid", 40.0, 65.0, (15.0, -20.0, 10.0)),
    ("topography W-E through the middle", "topo", 90.0, 90.0, (0.0, 0.0, 0.0)),
    ("topography oblique", "topo", 127.0, 90.0, (0.0, 0.0, 0.0)),
]


def _trig(degrees: float) -> tuple[float, float]:
    """Sine and cosine, exact at multiples of 90 degrees (as the viewer computes them)."""
    d = degrees % 360.0
    exact = {0.0: (0.0, 1.0), 90.0: (1.0, 0.0), 180.0: (0.0, -1.0), 270.0: (-1.0, 0.0)}
    return exact.get(d, (math.sin(math.radians(d)), math.cos(math.radians(d))))


def plane_normal(azimuth: float, dip: float) -> list[float]:
    sa, ca = _trig(azimuth)
    st, ct = _trig(90.0 - dip)
    return [ct * ca, -ct * sa, st]


def load_mesh(name: str) -> Mesh:
    return Mesh(np.load(DEMO / f"{name}_vertices.npy"), np.load(DEMO / f"{name}_faces.npy"))


def build_fixture() -> str:
    meshes = {name: load_mesh(name) for name in {case[1] for case in CASES}}
    lo, hi = meshes["ore_solid"].bounds
    middle = np.round((lo + hi) / 2.0)
    cases = []
    for name, mesh_name, azimuth, dip, offset in CASES:
        origin = middle + np.array(offset)
        normal = plane_normal(azimuth, dip)
        cut = section(meshes[mesh_name], origin, normal)
        cases.append(
            {
                "name": name,
                "mesh": mesh_name,
                "azimuth": azimuth,
                "dip": dip,
                "origin": origin.tolist(),
                "normal": normal,
                "closed": list(cut.closed),
                "area": cut.area,
                "length": cut.length,
                "polylines": [np.round(p, 6).tolist() for p in cut.polylines],
            }
        )
    return json.dumps({"cases": cases}, separators=(",", ":")) + "\n"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    text = build_fixture()
    FIXTURE.write_text(text)
    print(f"{FIXTURE.relative_to(REPO)}: {len(text)} bytes")
