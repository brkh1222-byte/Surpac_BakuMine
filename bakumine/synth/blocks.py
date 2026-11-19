"""Truth block grid: true ore fraction, rock code and block-average grade of every parent block.

Each block is discretised by a regular lattice of points. At a point the rock is AIR above
the topography, ORE inside the solid, otherwise WASTE; its grade is the truth averaged over
the nugget component (``GradeModel.ore_expected``) in ore and the waste mean in waste.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bakumine.synth.config import BlockConfig, GradeConfig
from bakumine.synth.field import GaussianField, GradeModel
from bakumine.synth.solid import LensSolid
from bakumine.synth.topo import Topography

AIR, WASTE, ORE = "AIR", "WASTE", "ORE"


@dataclass(frozen=True)
class BlockGrid:
    origin: np.ndarray  # minimum corner (x, y, z)
    size: np.ndarray
    shape: tuple[int, int, int]  # blocks along x, y, z
    columns: dict[str, np.ndarray]  # flat, index = i + nx * (j + ny * k)


def build_blocks(
    cfg: BlockConfig,
    grade_cfg: GradeConfig,
    solid: LensSolid,
    solid_vertices: np.ndarray,
    topo: Topography,
    field: GaussianField,
    grade: GradeModel,
) -> BlockGrid:
    size = np.array(cfg.size, dtype=np.float64)
    lo = solid_vertices.min(axis=0) - cfg.margin
    hi = solid_vertices.max(axis=0) + cfg.margin
    hi[2] = topo.upper_bound(lo[:2], hi[:2]) + size[2]
    origin = np.floor(lo / size) * size
    shape = tuple(int(n) for n in np.ceil((hi - origin) / size).astype(int))
    nx, ny, nz = shape
    ore_z = (solid_vertices[:, 2].min() - size[2], solid_vertices[:, 2].max() + size[2])

    offsets = [
        ((np.arange(n) + 0.5) / n - 0.5) * s
        for n, s in zip(cfg.discretisation, size, strict=True)
    ]
    sub = np.stack(np.meshgrid(*offsets, indexing="ij"), axis=-1).reshape(-1, 3)
    cx = origin[0] + (np.arange(nx) + 0.5) * size[0]
    cy = origin[1] + (np.arange(ny) + 0.5) * size[1]
    cz = origin[2] + (np.arange(nz) + 0.5) * size[2]
    # One layer: x varies fastest, matching the flat index.
    layer_x, layer_y = np.tile(cx, ny), np.repeat(cy, nx)
    px = layer_x[:, None] + sub[:, 0]
    py = layer_y[:, None] + sub[:, 1]
    ground = topo.z(px, py)
    ground_centre = topo.z(layer_x, layer_y)

    per_layer = nx * ny
    ore_fraction = np.zeros(per_layer * nz)
    air_fraction = np.zeros(per_layer * nz)
    au_ore = np.full(per_layer * nz, np.nan)
    rock = np.empty(per_layer * nz, dtype=object)
    for k in range(nz):
        rows = slice(k * per_layer, (k + 1) * per_layer)
        pz = cz[k] + np.broadcast_to(sub[:, 2], px.shape)
        air = pz > ground
        air_fraction[rows] = air.mean(axis=1)
        code = np.where(cz[k] > ground_centre, AIR, WASTE).astype(object)
        if ore_z[0] <= cz[k] <= ore_z[1]:
            points = np.stack([px, py, pz], axis=-1).reshape(-1, 3)
            inside = solid.inside(points)
            if inside.any():
                values = np.zeros(len(points))
                values[inside] = grade.ore_expected(field.at(points[inside]))
                inside = inside.reshape(px.shape)
                count = inside.sum(axis=1)
                ore_fraction[rows] = count / px.shape[1]
                total = values.reshape(px.shape).sum(axis=1)
                au_ore[rows] = np.where(count > 0, total / np.maximum(count, 1), np.nan)
                centre = np.column_stack([layer_x, layer_y, np.full(per_layer, cz[k])])
                code[solid.inside(centre)] = ORE
        rock[rows] = code

    waste_fraction = np.clip(1.0 - ore_fraction - air_fraction, 0.0, 1.0)
    ore_mass = grade_cfg.ore_density * ore_fraction
    waste_mass = grade_cfg.waste_density * waste_fraction
    solid_fraction = ore_fraction + waste_fraction
    has_rock = solid_fraction > 0.0
    with np.errstate(invalid="ignore", divide="ignore"):
        density = np.where(has_rock, (ore_mass + waste_mass) / solid_fraction, np.nan)
        metal = ore_mass * np.nan_to_num(au_ore) + waste_mass * grade.waste_mean
        au = np.where(has_rock, metal / (ore_mass + waste_mass), np.nan)

    k_index, j_index, i_index = np.meshgrid(
        np.arange(nz, dtype=np.int32),
        np.arange(ny, dtype=np.int32),
        np.arange(nx, dtype=np.int32),
        indexing="ij",
    )
    columns = {
        "i": i_index.ravel(),
        "j": j_index.ravel(),
        "k": k_index.ravel(),
        "x": cx[i_index.ravel()],
        "y": cy[j_index.ravel()],
        "z": cz[k_index.ravel()],
        "dx": np.full(per_layer * nz, size[0]),
        "dy": np.full(per_layer * nz, size[1]),
        "dz": np.full(per_layer * nz, size[2]),
        "rock": rock,
        "ore_fraction": ore_fraction,
        "waste_fraction": waste_fraction,
        "air_fraction": air_fraction,
        "density": density,
        "au": au,
        "au_ore": au_ore,
    }
    return BlockGrid(origin=origin, size=size, shape=shape, columns=columns)
