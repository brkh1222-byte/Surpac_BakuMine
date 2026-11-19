"""Build the synthetic benchmark deposit and write it to a folder.

Input files (what a geologist would have): ``collar.csv``, ``survey.csv``, ``assay.csv``,
``geology.csv``, ``topo.obj``. Everything named ``truth*`` and the ore solid are the oracle
and must not be fed to the pipeline under test.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from bakumine.synth.blocks import BlockGrid, build_blocks
from bakumine.synth.config import SynthConfig, default_config
from bakumine.synth.drillholes import ORE, WASTE, Hole, assay_hole, log_geology, plan_holes
from bakumine.synth.field import GaussianField, GradeModel
from bakumine.synth.solid import LensSolid, axes_from_angles, mesh_volume
from bakumine.synth.topo import Topography

GENERATOR_VERSION = 1
MESH_VOLUME_BOUND = 0.01  # stated bound on |mesh volume - true volume| / true volume
GRAMS_PER_TROY_OUNCE = 31.1034768


@dataclass(frozen=True)
class SynthDeposit:
    config: SynthConfig
    seed: int
    solid: LensSolid
    topo: Topography
    field: GaussianField
    grade: GradeModel
    holes: list[Hole]
    solid_vertices: np.ndarray
    solid_faces: np.ndarray
    topo_vertices: np.ndarray
    topo_faces: np.ndarray
    tables: dict[str, dict[str, np.ndarray]]  # collar, survey, assay, geology, truth_*
    blocks: BlockGrid
    truth: dict[str, Any]

    def write(self, out_dir: str | Path, save_field: bool = False) -> list[Path]:
        return _write(self, Path(out_dir), save_field)


def _integrate_ore(
    solid: LensSolid, field: GaussianField, grade: GradeModel
) -> tuple[float, float]:
    """Voxel count on the simulation lattice: (ore volume, volume-integrated grade)."""
    u, v, w = (field.node_coordinates(axis) for axis in range(3))
    cell_volume = float(np.prod(field.cell))
    volume = 0.0
    grade_volume = 0.0
    vw = np.stack(np.meshgrid(v, w, indexing="ij"), axis=-1).reshape(-1, 2)
    for i, ui in enumerate(u):
        local = np.column_stack([np.full(len(vw), ui), vw])
        inside = solid.inside(field.centre + local @ field.axes.T)
        volume += float(inside.sum()) * cell_volume
        values = grade.ore_expected(field.values[i].reshape(-1)[inside])
        grade_volume += float(values.sum()) * cell_volume
    return volume, grade_volume


def generate(config: SynthConfig | None = None, seed: int = 42) -> SynthDeposit:
    """Generate the deposit. The same ``config`` and ``seed`` always give the same result."""
    cfg = config or default_config()
    n_holes = cfg.drill.n_sections * cfg.drill.holes_per_section
    field_seq, hole_seq, sample_seq = np.random.SeedSequence(seed).spawn(3)
    rng_field, rng_holes = np.random.default_rng(field_seq), np.random.default_rng(hole_seq)

    solid = LensSolid.from_config(cfg.lens)
    vertices, faces = solid.mesh(cfg.lens.mesh_subdivisions)
    topo = Topography(cfg.topo)
    cover = float(np.min(topo.z(vertices[:, 0], vertices[:, 1]) - vertices[:, 2]))
    if cover < cfg.topo.min_cover:
        raise ValueError(f"ore solid comes within {cover:.1f} m of the topography")

    vario_axes = axes_from_angles(cfg.variogram.bearing, cfg.variogram.plunge, cfg.variogram.dip)
    field = GaussianField.simulate(cfg.variogram, solid.centre, vario_axes, vertices, rng_field)
    grade = GradeModel.from_config(cfg.grade, cfg.variogram)

    holes = plan_holes(cfg.drill, solid, topo, rng_holes)
    sample_rngs = [np.random.default_rng(s) for s in sample_seq.spawn(n_holes)]
    geology = [log_geology(h, solid, cfg.drill.contact_scan_step) for h in holes]
    samples = [
        assay_hole(cfg.drill, cfg.grade, h, solid, field, grade, rng)
        for h, rng in zip(holes, sample_rngs, strict=True)
    ]

    def per_hole(counts: list[int]) -> np.ndarray:
        return np.repeat(np.array([h.hole_id for h in holes], dtype=object), counts)

    n_stations = [len(h.depth) for h in holes]
    n_samples = [len(s.au) for s in samples]
    collars = np.array([h.collar for h in holes])
    tables: dict[str, dict[str, np.ndarray]] = {
        "collar": {
            "hole_id": per_hole([1] * n_holes),
            "x": collars[:, 0],
            "y": collars[:, 1],
            "z": collars[:, 2],
            "max_depth": np.array([h.max_depth for h in holes]),
        },
        "survey": {
            "hole_id": per_hole(n_stations),
            "depth": np.concatenate([h.depth for h in holes]),
            "dip": np.concatenate([h.dip for h in holes]),
            "azimuth": np.concatenate([h.azimuth for h in holes]),
        },
        "assay": {
            "hole_id": per_hole(n_samples),
            "depth_from": np.concatenate([s.depth_from for s in samples]),
            "depth_to": np.concatenate([s.depth_to for s in samples]),
            "au": np.concatenate([s.au for s in samples]),
        },
        "geology": {
            "hole_id": per_hole([len(g) for g in geology]),
            "depth_from": np.array([row[0] for g in geology for row in g]),
            "depth_to": np.array([row[1] for g in geology for row in g]),
            "rock": np.array([row[2] for g in geology for row in g], dtype=object),
        },
        "truth_samples": {
            "hole_id": per_hole(n_samples),
            "depth_from": np.concatenate([s.depth_from for s in samples]),
            "depth_to": np.concatenate([s.depth_to for s in samples]),
            "x": np.concatenate([s.xyz[:, 0] for s in samples]),
            "y": np.concatenate([s.xyz[:, 1] for s in samples]),
            "z": np.concatenate([s.xyz[:, 2] for s in samples]),
            "rock": np.where(np.concatenate([s.inside_mid for s in samples]), ORE, WASTE).astype(
                object
            ),
            "ore_fraction": np.concatenate([s.ore_fraction for s in samples]),
            "gauss": np.concatenate([s.gauss for s in samples]),
            "au_true": np.concatenate([s.au_true for s in samples]),
        },
        "truth_stations": {
            "hole_id": per_hole(n_stations),
            "depth": np.concatenate([h.depth for h in holes]),
            "x": np.concatenate([h.path.station[:, 0] for h in holes]),
            "y": np.concatenate([h.path.station[:, 1] for h in holes]),
            "z": np.concatenate([h.path.station[:, 2] for h in holes]),
        },
    }

    spacing = cfg.topo.dtm_spacing
    xy = np.concatenate([vertices[:, :2], collars[:, :2]])
    dtm_lo = np.floor((xy.min(axis=0) - cfg.topo.dtm_margin) / spacing) * spacing
    dtm_hi = np.ceil((xy.max(axis=0) + cfg.topo.dtm_margin) / spacing) * spacing
    topo_vertices, topo_faces = topo.dtm(dtm_lo, dtm_hi, collars[:, :2])

    blocks = build_blocks(cfg.blocks, cfg.grade, solid, vertices, topo, field, grade)

    volume = solid.volume()
    volume_mesh = mesh_volume(vertices, faces)
    volume_voxel, grade_volume = _integrate_ore(solid, field, grade)
    mean_grade = grade_volume / volume_voxel
    tonnes = volume * cfg.grade.ore_density
    metal_g = tonnes * mean_grade
    block_volume = float(np.prod(blocks.size))
    block_ore_volume = float(blocks.columns["ore_fraction"].sum() * block_volume)
    if abs(volume_mesh / volume - 1.0) > MESH_VOLUME_BOUND:
        raise ValueError("mesh volume is outside the stated bound; raise mesh_subdivisions")

    truth: dict[str, Any] = {
        "generator": {"name": "bakumine.synth", "version": GENERATOR_VERSION},
        "seed": seed,
        "config": cfg.to_dict(),
        "coordinate_system": {
            "axes": "x = east, y = north, z = up",
            "units": "metres",
            "description": "local mine grid; all files use x, y, z order",
            "dip_convention": "negative downward (vertical hole = -90)",
            "azimuth_convention": "degrees clockwise from grid north",
        },
        "rotation_convention": (
            "axes matrix columns = unit vectors (east, north, up) of the major, semi-major and "
            "minor axes. major = (sin b cos p, cos b cos p, -sin p) for bearing b, plunge p; "
            "with dip 0 the semi-major axis is horizontal, 90 deg anticlockwise of the bearing; "
            "dip rotates it downward about the major axis. local = (world - centre) @ axes."
        ),
        "solid": {
            "implicit": (
                "inside when (u/a)^2 + (v/b)^2 + ((w - m)/(c t))^2 < 1, with "
                "t = 1 - pinch_depth exp(-((u - pinch_position)/pinch_width)^2), "
                "m = bend_major (u/a)^2 + bend_semi (v/b)^2, (u, v, w) = local coordinates"
            ),
            "centre": solid.centre.tolist(),
            "axes": solid.axes.tolist(),
            "half_axes": [solid.a, solid.b, solid.c],
            "volume_m3": volume,
            "volume_method": (
                "exact: V = pi b c integral_{-a}^{a} t(u) (1 - u^2/a^2) du, adaptive quadrature "
                "to 1e-13 relative"
            ),
            "mesh": {
                "vertices": int(len(vertices)),
                "faces": int(len(faces)),
                "volume_m3": volume_mesh,
                "relative_error": volume_mesh / volume - 1.0,
                "relative_error_bound": MESH_VOLUME_BOUND,
            },
            "voxel_check": {
                "cell_m": field.cell.tolist(),
                "frame": "variogram axes",
                "volume_m3": volume_voxel,
                "relative_error": volume_voxel / volume - 1.0,
            },
            "min_cover_below_topography_m": cover,
        },
        "variogram": {
            "space": "Gaussian variable Y (grade = exp(mu + sigma Y) in ore)",
            "model": "spherical",
            "formula": "gamma(h) = nugget + sill (1.5 h/a - 0.5 (h/a)^3) for h < a, else total",
            "nugget": cfg.variogram.nugget,
            "sill": 1.0 - cfg.variogram.nugget,
            "total_sill": 1.0,
            "ranges_m": list(cfg.variogram.ranges),
            "angles_deg": {
                "bearing": cfg.variogram.bearing,
                "plunge": cfg.variogram.plunge,
                "dip": cfg.variogram.dip,
            },
            "axes": vario_axes.tolist(),
            "simulation": {
                "method": "FFT moving average on a periodic lattice aligned with the axes",
                "cell_m": field.cell.tolist(),
                "shape": list(field.values.shape),
                "lower_corner_local_m": field.lower.tolist(),
                "between_nodes": "trilinear interpolation of the structured component",
                "nugget": "independent standard normal per assay sample",
            },
        },
        "grade": {
            "element": "au",
            "units": "g/t",
            "ore": "exp(mu + sigma Y), Y = sqrt(1 - nugget) Ys + sqrt(nugget) eps",
            "ore_mu": grade.mu,
            "ore_sigma": grade.sigma,
            "ore_ensemble_mean": grade.ore_ensemble_mean,
            "waste": "exp(waste_mu + waste_sigma eps), spatially uncorrelated",
            "waste_mu": grade.waste_mu,
            "waste_sigma": grade.waste_sigma,
            "waste_mean": grade.waste_mean,
            "sample_support": (
                "mean of the point grade at sample_subpoints positions along the interval; "
                "one nugget draw per sample; rounded to assay_decimals, floored at assay_floor"
            ),
        },
        "density_t_m3": {ORE: cfg.grade.ore_density, WASTE: cfg.grade.waste_density},
        "topography": {
            "formula": (
                "z = base + slope_x X + slope_y Y + sum amp sin(2 pi (X sin d + Y cos d) / "
                "wavelength + phase), X = x - origin_x, Y = y - origin_y, d = direction "
                "(azimuth); rounded to 1e-6 m"
            ),
            "dtm_extent": {"min": dtm_lo.tolist(), "max": dtm_hi.tolist()},
            "dtm_vertices": int(len(topo_vertices)),
            "dtm_faces": int(len(topo_faces)),
            "dtm_note": "regular grid plus every collar as a vertex; faces counter-clockwise",
        },
        "drilling": {
            "holes": n_holes,
            "survey_stations": int(sum(n_stations)),
            "assay_intervals": int(sum(n_samples)),
            "desurvey": "minimum curvature; straight below the last station",
        },
        "blocks": {
            "origin": blocks.origin.tolist(),
            "size": blocks.size.tolist(),
            "shape": list(blocks.shape),
            "discretisation": list(cfg.blocks.discretisation),
            "order": "row = i + nx * (j + ny * k); x, y, z are block centroids",
            "ore_volume_m3": block_ore_volume,
            "ore_volume_relative_error": block_ore_volume / volume - 1.0,
        },
        "totals": {
            "ore_volume_m3": volume,
            "ore_tonnes": tonnes,
            "ore_mean_grade_g_t": mean_grade,
            "ore_metal_g": metal_g,
            "ore_metal_troy_oz": metal_g / GRAMS_PER_TROY_OUNCE,
            "mean_grade_method": (
                "volume average of the nugget-averaged grade over the simulation lattice nodes "
                "inside the solid (see solid.voxel_check for the resolution)"
            ),
        },
    }
    return SynthDeposit(
        config=cfg,
        seed=seed,
        solid=solid,
        topo=topo,
        field=field,
        grade=grade,
        holes=holes,
        solid_vertices=vertices,
        solid_faces=faces,
        topo_vertices=topo_vertices,
        topo_faces=topo_faces,
        tables=tables,
        blocks=blocks,
        truth=truth,
    )


# Column formats of the CSV files. Depths are whole centimetres; collar z keeps 1e-6 m.
CSV_FORMATS: dict[str, dict[str, str]] = {
    "collar": {"hole_id": "{}", "x": "{:.3f}", "y": "{:.3f}", "z": "{:.6f}", "max_depth": "{:.2f}"},
    "survey": {"hole_id": "{}", "depth": "{:.2f}", "dip": "{:.2f}", "azimuth": "{:.2f}"},
    "assay": {"hole_id": "{}", "depth_from": "{:.2f}", "depth_to": "{:.2f}", "au": "{:.3f}"},
    "geology": {"hole_id": "{}", "depth_from": "{:.2f}", "depth_to": "{:.2f}", "rock": "{}"},
    "truth_samples": {
        "hole_id": "{}",
        "depth_from": "{:.2f}",
        "depth_to": "{:.2f}",
        "x": "{:.6f}",
        "y": "{:.6f}",
        "z": "{:.6f}",
        "rock": "{}",
        "ore_fraction": "{:.2f}",
        "gauss": "{:.6f}",
        "au_true": "{:.6f}",
    },
    "truth_stations": {
        "hole_id": "{}",
        "depth": "{:.2f}",
        "x": "{:.6f}",
        "y": "{:.6f}",
        "z": "{:.6f}",
    },
}


def _write_csv(path: Path, table: dict[str, np.ndarray], formats: dict[str, str]) -> None:
    columns = [[formats[name].format(value) for value in table[name].tolist()] for name in formats]
    lines = [",".join(formats)]
    lines.extend(",".join(row) for row in zip(*columns, strict=True))
    # A missing value is an empty field ("nan" never reaches a file).
    path.write_text("\n".join(lines).replace(",nan", ",") + "\n", encoding="ascii", newline="\n")


def _write_obj(path: Path, vertices: np.ndarray, faces: np.ndarray, comment: str) -> None:
    lines = [f"# {comment}", "# x = east, y = north, z = up (metres)"]
    lines.extend(f"v {x:.6f} {y:.6f} {z:.6f}" for x, y, z in vertices.tolist())
    lines.extend(f"f {a + 1} {b + 1} {c + 1}" for a, b, c in faces.tolist())
    path.write_text("\n".join(lines) + "\n", encoding="ascii", newline="\n")


def _write(deposit: SynthDeposit, out: Path, save_field: bool) -> list[Path]:
    import pandas as pd

    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    def target(name: str) -> Path:
        written.append(out / name)
        return written[-1]

    for name, formats in CSV_FORMATS.items():
        _write_csv(target(f"{name}.csv"), deposit.tables[name], formats)
    _write_obj(
        target("ore_solid.obj"), deposit.solid_vertices, deposit.solid_faces,
        "BakuMine synthetic ore solid (truth); closed, outward normals",
    )  # fmt: skip
    np.save(target("ore_solid_vertices.npy"), deposit.solid_vertices)
    np.save(target("ore_solid_faces.npy"), deposit.solid_faces)
    _write_obj(
        target("topo.obj"), deposit.topo_vertices, deposit.topo_faces,
        "BakuMine synthetic topography DTM; normals up",
    )  # fmt: skip
    np.save(target("topo_vertices.npy"), deposit.topo_vertices)
    np.save(target("topo_faces.npy"), deposit.topo_faces)
    pd.DataFrame(deposit.blocks.columns).to_parquet(
        target("truth_blocks.parquet"), index=False, compression="zstd"
    )
    if save_field:
        np.save(target("truth_gauss_field.npy"), deposit.field.values)
    truth = dict(deposit.truth)
    truth["files"] = sorted(p.name for p in written) + ["truth.json"]
    target("truth.json").write_text(
        json.dumps(truth, indent=2, sort_keys=True) + "\n", encoding="ascii", newline="\n"
    )
    return written
