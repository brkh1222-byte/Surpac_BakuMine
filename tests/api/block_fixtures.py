"""Small block models and what the web client must reconstruct from them.

``uv run python tests/api/block_fixtures.py`` rewrites
``web/src/blocks/fixtures/blocks.json``: for a dense, a sub-blocked (unequal depths) and a
rotated sub-blocked model, the definition and structure columns as the API sends them,
the centroids / sizes / ids ``BlockModel`` computes, a "blocks" payload (base64) with its
decoded content, and rays with the block they hit first (brute force over all boxes).
``test_blocks_api.py`` checks that the committed file still matches this generator.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.api.binary import unpack_arrays
from bakumine.api.blocks import BlockData, block_payload, definition_dict, parse_constraint
from bakumine.core.blockmodel import BlockModel, BlockModelDefinition

REPO = Path(__file__).resolve().parents[2]
FIXTURE = REPO / "web" / "src" / "blocks" / "fixtures" / "blocks.json"

ORIGIN = (500400.0, 5300250.0, 825.0)


def _with_attributes(model: BlockModel) -> BlockModel:
    """``au`` (float32, some NaN) and ``rock`` (category, some missing), deterministic."""
    n = model.n_blocks
    index = np.arange(n)
    model.add_attribute("au", "float", dtype=np.float32, unit="g/t")
    model.set("au", np.where(index % 7 == 3, np.nan, 0.125 * (index % 23) + 0.05))
    model.add_attribute("rock", "category", categories=["AIR", "WASTE", "ORE"], dtype=np.int16)
    model["rock"][:] = (index % 4) - 1
    return model


def dense_model() -> BlockModel:
    return _with_attributes(BlockModel(BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (3, 4, 2))))


def octree_model() -> BlockModel:
    """Depths (2, 1, 0): first split halves x and y, the second halves x only."""
    model = BlockModel(
        BlockModelDefinition(ORIGIN, (10.0, 10.0, 5.0), (3, 2, 2), max_depth=(2, 1, 0))
    )
    model.split([1, 4, 10])
    model.split(np.flatnonzero(model.level == 1)[::3])
    return _with_attributes(model)


def rotated_model() -> BlockModel:
    """Bearing 30, plunge 10, dip 5 about the minimum corner; one octree level."""
    model = BlockModel(
        BlockModelDefinition(
            ORIGIN, (12.5, 7.5, 4.0), (3, 3, 2), angles=(30.0, 10.0, 5.0), max_depth=1
        )
    )
    model.split([0, 7, 17])
    return _with_attributes(model)


MODELS = {"dense": dense_model, "octree": octree_model, "rotated": rotated_model}


def first_hits(
    model: BlockModel, rows: np.ndarray, origins: np.ndarray, directions: np.ndarray, shrink: float
) -> list[dict[str, Any]]:
    """Brute force: nearest box of ``rows`` each ray enters (slab test in the model frame)."""
    definition = model.definition
    lo, hi = model.bounds_local(rows)
    centre, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * shrink
    lo, hi = centre - half, centre + half
    rotation = definition.rotation
    out = []
    for origin, direction in zip(origins, directions, strict=True):
        o = definition.world_to_local(origin)
        d = rotation.T @ direction
        with np.errstate(divide="ignore", invalid="ignore"):
            t1, t2 = (lo - o) / d, (hi - o) / d
        near = np.nanmax(np.minimum(t1, t2), axis=1)
        far = np.nanmin(np.maximum(t1, t2), axis=1)
        hit = (near <= far) & (far >= 0.0)
        if not hit.any():
            out.append({"origin": origin.tolist(), "direction": direction.tolist(), "row": -1,
                        "t": None})  # fmt: skip
            continue
        entry = np.where(hit, np.maximum(near, 0.0), np.inf)
        best = int(np.argmin(entry))
        out.append({"origin": origin.tolist(), "direction": direction.tolist(),
                    "row": int(rows[best]), "t": float(entry[best])})  # fmt: skip
    return out


def _rays(model: BlockModel, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rays from outside the model towards chosen block centroids, and two that miss."""
    box = model.definition.bounding_box()
    centre, reach = box.mean(axis=0), float(np.linalg.norm(box[1] - box[0]))
    targets = model.centroids(rows[:: max(1, len(rows) // 6)])
    rng = np.random.default_rng(11)
    origins, directions = [], []
    for k, target in enumerate(targets):
        away = rng.normal(size=3)
        away[2] = abs(away[2]) + 0.2  # from above, like a camera
        origin = target + away / np.linalg.norm(away) * reach * (1.5 + 0.1 * k)
        origins.append(origin)
        directions.append((target - origin) / np.linalg.norm(target - origin))
    for offset in ((3.0, 0.0, 0.0), (0.0, -3.0, 0.5)):
        origins.append(centre + np.array(offset) * reach)
        directions.append(np.array([0.0, 0.0, 1.0]))
    return np.array(origins), np.array(directions)


def build_case(name: str) -> dict[str, Any]:
    model = MODELS[name]()
    data = BlockData(model)
    where = "rock != \"AIR\" and not isnan(rock)"
    shown = np.flatnonzero(parse_constraint(where).mask(model))
    payload = block_payload(data, name, ["au", "rock"], parse_constraint(where), "full")
    header, arrays = unpack_arrays(payload)
    origins, directions = _rays(model, shown)
    return {
        "name": name,
        "definition": definition_dict(model.definition),
        "n_blocks": model.n_blocks,
        "parent": model.parent.tolist(),
        "level": model.level.tolist(),
        "sub": model.sub.reshape(-1).tolist(),
        "centroids": model.centroids().reshape(-1).tolist(),
        "centroids_local": model.centroids_local().reshape(-1).tolist(),
        "sizes": model.sizes().reshape(-1).tolist(),
        "volumes": model.volumes().tolist(),
        "ijk": model.ijk().reshape(-1).tolist(),
        "block_ids": [str(int(v)) for v in model.block_ids()],
        "where": where,
        "shown_rows": shown.tolist(),
        "payload_base64": base64.b64encode(payload).decode("ascii"),
        "payload_meta": header["meta"],
        "payload_arrays": {
            key: {
                "dtype": a.dtype.kind + str(a.dtype.itemsize),
                "data": [None if isinstance(v, float) and np.isnan(v) else v
                         for v in a.reshape(-1).tolist()],
            }
            for key, a in arrays.items()
        },  # fmt: skip
        "rays": {
            "1": first_hits(model, shown, origins, directions, 1.0),
            "0.8": first_hits(model, shown, origins, directions, 0.8),
        },
    }


def build_fixture() -> str:
    return json.dumps({"cases": [build_case(name) for name in MODELS]}, indent=1) + "\n"


if __name__ == "__main__":
    FIXTURE.parent.mkdir(parents=True, exist_ok=True)
    FIXTURE.write_text(build_fixture())
    print(f"{FIXTURE.name}: {FIXTURE.stat().st_size} bytes")
