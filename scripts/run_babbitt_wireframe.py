"""P8.T12: the Babbitt Cu-Ni data with the public domain wireframe, against published numbers.

    uv run python scripts/run_babbitt_wireframe.py [OUT_DIR]

The PyGSLIB tutorial "Resource estimation with PyGSLIB" (Adrian Martinez Vargas;
https://opengeostat.github.io/pygslib/Tutorial.html, source ``doc/source/Tutorial.md`` of the
commit pinned in ``data/public/babbitt/SOURCE.md``) estimates Cu inside the wireframe
``Mpz.stl`` with a modified GSLIB ``kt3d`` and prints the numbers in ``PUBLISHED`` below.
This script repeats the tutorial with the ``bakumine`` CLI and compares.  Unlike
``run_babbitt.py`` (our own choices, no wireframe) every parameter here is the tutorial's,
including the ones we would not choose.

Tutorial parameters (feet; converted to metres, 1 ft = 0.3048 m, at import)
---------------------------------------------------------------------------
* Unassayed Cu set to 0 before compositing (``composite --fill-missing 0``).  This departs
  from "missing stays missing" on purpose: it is what the published numbers are made of.
  The run is repeated with missing kept missing (``variant_missing_stays_missing`` in
  ``metrics.json``) to show what that choice does.
* Composites: 10 ft down the hole from the collar; ``minlen=-1`` means half the length in
  PyGSLIB, so a last composite shorter than 5 ft is dropped (``--short discard --min-length
  0.5``).
* Domain: composites whose midpoint is inside ``Mpz.stl``.
* Block model: origin (2288230, 415200, -1000) ft, 100 x 100 x 30 ft blocks, 160 x 100 x 90.
  Blocks with any part inside the wireframe are estimated (PyGSLIB ``fillwireframe``, fraction
  ``D1 > 0``); the published means are plain means over those blocks, not weighted by ``D1``.
* Search: radii 850 / 850 / 250 ft, GSLIB angles (-28, 34, 7), 4 to 12 samples, at most 3 per
  hole, no octants.
* Variogram: nugget 0.35, exponential 0.41 with range 96 ft (isotropic), exponential 0.23
  with ranges 1117 / 1117 / 300 ft, same angles; fractions of the variance 0.109758094158.
  GSLIB's exponential uses the practical range, as ``bakumine`` does.
* Ordinary block kriging, 5 x 5 x 3 discretisation.  The tutorial's ``kt3d`` also returns, from
  the same neighbourhood, an inverse-distance-squared estimate (point, at the block centre,
  ISOTROPIC distances) and the nearest sample (first of the neighbourhood, i.e. nearest in the
  anisotropic search metric); both only where kriging had its 4 samples.
* Declustering: GSLIB ``declus``, cells 1000 x 1000 x 100 ft, 8 origin offsets.

What differs here, by construction (measured in ``metrics.json`` ``diagnostics``)
-------------------------------------------------------------------------------
* ``Mpz.stl`` is not a closed solid as published: one triangle is missing and three
  zero-thickness fins (pairs of coincident, opposite triangles) hang on its surface.
  ``repair_wireframe`` removes the fins and closes the hole (asserting that this is all there
  is); the tutorial's VTK point-in-solid test tolerates the defects.
* ``bakumine`` inverse distance weights by the anisotropic distance of the search ellipsoid;
  the tutorial's by the isotropic distance.  The plan estimate is ours; the tutorial's variant
  is recomputed in ``diagnostics`` from the same neighbourhoods to size the effect.
* ``bakumine`` nearest neighbour has no minimum number of samples; its mean is taken over the
  blocks kriging estimated, as in the tutorial.
* Composite midpoints: ours is the midpoint of the composite's actual length, PyGSLIB's the
  midpoint of the nominal 10 ft interval (they differ for the last composite of a hole).
  Desurvey methods may differ; 348 of 399 holes are vertical.
* The fraction of a block inside the wireframe (``d1``) is computed here with
  ``bakumine.core.blockmodel.partial`` (column method) and the inside flag of the composites
  with ``points_in_solid``, because the repaired wireframe exists only in memory. The CLI
  commands for a solid file are ``bakumine blockmodel fraction`` and ``bakumine composite
  --flag-solid`` (added after this run was written).

Density 2.7 t/m3 is the tutorial's figure (its tonnage formula; it prints no tonnage).
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "data" / "public" / "babbitt"
DEFAULT_OUT = REPO / "data" / "reference" / "babbitt_wireframe_out"
FT = 0.3048

# Printed by the tutorial (doc/source/Tutorial.md at the pinned commit; the same text is on
# https://opengeostat.github.io/pygslib/Tutorial.html, read 2026-11-21).
PUBLISHED = {
    "ok_mean": 0.21166003,
    "id2_mean": 0.20810474,
    "nn_mean": 0.20687895,
    "data_mean": 0.24149141734342264,
    "declustered_mean": 0.2251903672467954,
    "declustered_variance": 0.109758094158,  # "Raw Variance" of the weighted anamorphosis
    "data_cv_declustered": 1.4711890014945388,
    "data_max": 15.4000001,
    "n_composites": 54189,
}

ORIGIN_FT = (2288230.0, 415200.0, -1000.0)
BLOCK_FT = (100.0, 100.0, 30.0)
SHAPE = (160, 100, 90)
SEARCH_RADII_FT = (850.0, 850.0, 250.0)  # GSLIB radius, radius1, radius2
ANGLES_GSLIB = (-28.0, 34.0, 7.0)
MAX_SAMPLES, MIN_SAMPLES, MAX_PER_HOLE = 12, 4, 3
DISCRETISATION = (5, 5, 3)
VARIANCE = PUBLISHED["declustered_variance"]
NUGGET_FRACTION = 0.35
STRUCTURES = (("exponential", 0.41, (96.0, 96.0, 96.0)),
              ("exponential", 0.23, (1117.0, 1117.0, 300.0)))  # fmt: skip
DECLUSTER_CELL_FT = (1000.0, 1000.0, 100.0)
DENSITY = 2.7
SOLID = "mpz"
COINCIDENT = 1e-6  # m: samples closer than this are one place (twin holes; krige's default)


def cli(*argv: object) -> None:
    from bakumine.cli import main

    code = main([str(a) for a in argv])
    if code != 0:
        raise SystemExit(f"bakumine {' '.join(map(str, argv[:2]))} failed with exit code {code}")


def metres(values) -> tuple[float, ...]:
    return tuple(float(v) * FT for v in values)


# -------------------------------------------------------------------------- wireframe
def repair_wireframe(mesh):
    """Close ``Mpz.stl``: drop the zero-thickness fins, fill the one triangular hole.

    Returns the closed mesh and what was done.  Anything else wrong with the mesh (another
    hole, a hole that is not a triangle) stops the run: this is not a general repair.
    """
    from bakumine.core.geometry.mesh import Mesh

    faces = mesh.faces
    key = np.sort(faces, axis=1)
    _, inverse, counts = np.unique(key, axis=0, return_inverse=True, return_counts=True)
    fin = counts[np.asarray(inverse).reshape(-1)] > 1  # both triangles of every coincident pair
    kept = mesh.submesh(~fin)
    edges = kept.edges[kept.edge_map.boundary_edges]
    hole = np.unique(edges)
    if len(edges) != 3 or len(hole) != 3 or len(kept.edge_map.nonmanifold_edges):
        raise SystemExit(f"Mpz.stl: unexpected defects ({len(edges)} open edges after the fins)")
    # wind the new triangle against the neighbour across one of its edges
    a, b = edges[0]
    c = int(hole[(hole != a) & (hole != b)][0])
    owner = kept.faces[kept.edge_map.faces_of(int(kept.edge_map.boundary_edges[0]))[0]]
    i = int(np.where(owner == a)[0][0])
    patch = (b, a, c) if owner[(i + 1) % 3] == b else (a, b, c)
    closed = Mesh(kept.vertices, np.vstack([kept.faces, patch]))
    info = {"triangles_in_file": int(mesh.n_faces), "fin_triangles_removed": int(fin.sum()),
            "hole_triangles_added": 1,
            "hole_area_m2": float(closed.face_areas[-1])}  # fmt: skip
    return closed, info


def import_wireframe(out: Path, geometry: Path) -> dict:
    from bakumine.core.io.files import read_mesh, write_mesh

    raw = read_mesh(SRC / "raw" / "Mpz.stl", scale=FT)  # feet -> metres
    closed, info = repair_wireframe(raw)
    write_mesh(out / "mpz_closed.obj", closed, decimals=4)
    cli("surface", "import", out / "mpz_closed.obj", "--kind", "solid", "--geometry-dir",
        geometry, "--name", SOLID, "--summary", out / "mpz_import.json")  # fmt: skip
    summary = json.loads((out / "mpz_import.json").read_text())
    return info | {k: summary[k] for k in ("valid", "closed", "outward", "volume", "issues",
                                           "n_vertices", "n_faces", "bounds")}  # fmt: skip


# ------------------------------------------------------------------------------ plans
def ellipsoid(radii_ft):
    from bakumine.core.estimation.rotation import Ellipsoid

    # GSLIB radius / radius1 / radius2 = principal, horizontal-minor, vertical-minor
    # = our major, semi-major, minor (rotation.py, "gslib")
    return Ellipsoid.from_radii(metres(radii_ft), ANGLES_GSLIB, "gslib")


def variogram(variance: float = VARIANCE):
    from bakumine.core.estimation.variogram import Structure, VariogramModel

    return VariogramModel(
        NUGGET_FRACTION * variance,
        [Structure(kind, fraction * variance, metres(ranges), ANGLES_GSLIB, "gslib")
         for kind, fraction, ranges in STRUCTURES],
    )  # fmt: skip


def write_plans(out: Path) -> dict[str, Path]:
    from bakumine.core.estimation.block_estimation import (
        EstimationDomain,
        EstimationPass,
        EstimationPlan,
        OutputNames,
    )
    from bakumine.core.estimation.search import SearchParams

    ell = ellipsoid(SEARCH_RADII_FT)
    search = SearchParams(max_samples=MAX_SAMPLES, min_samples=MIN_SAMPLES,
                          max_per_hole=MAX_PER_HOLE)  # fmt: skip
    passes = {
        "ok": EstimationPass("ok", ell, search, variogram=variogram(),
                             discretisation=DISCRETISATION, name="ok"),
        "id2": EstimationPass("idw", ell, search, power=2.0, discretisation=(1, 1, 1),
                              name="id2"),
        "nn": EstimationPass("nn", ell, SearchParams(max_samples=1), name="nn"),
    }  # fmt: skip
    block = {"type": "attribute", "expression": "d1 > 0"}
    samples = {"column": "domain", "equals": 1}
    paths = {}
    for name, p in passes.items():
        attribute = "cu" if name == "ok" else f"cu_{name}"
        outputs = None if name == "ok" else OutputNames.only()
        plan = EstimationPlan(attribute, [EstimationDomain(SOLID, [p], block, samples)],
                              value_column="cu", outputs=outputs)  # fmt: skip
        paths[name] = out / f"plan_{name}.json"
        plan.save(paths[name])
    return paths


# ---------------------------------------------------------------------- diagnostics
def neighbourhood_diagnostics(dom: pd.DataFrame, centroids: np.ndarray):
    """From our neighbourhoods of ``centroids``: the tutorial's inverse distance (isotropic
    distances, point estimate), and which neighbourhoods hold two samples at one place.

    GSLIB ``kt3d`` solves without merging duplicates: two coincident samples make the kriging
    matrix singular and the block gets no kriged value (while its ID2 and NN values are kept),
    so such blocks are missing from the published kriging mean.
    """
    from bakumine.core.estimation.search import SampleSearch, SearchParams

    xyz, cu = dom[["x", "y", "z"]].to_numpy(), dom["cu"].to_numpy()
    search = SampleSearch(xyz, ellipsoid(SEARCH_RADII_FT), hole_ids=dom["hole_id"].to_numpy())
    r = search.query(centroids, SearchParams(max_samples=MAX_SAMPLES, min_samples=MIN_SAMPLES,
                                             max_per_hole=MAX_PER_HOLE))  # fmt: skip
    idx = np.where(r.indices >= 0, r.indices, 0)
    used = r.indices >= 0
    d = np.linalg.norm(xyz[idx] - centroids[:, None, :], axis=2)
    w = np.where(used, 1.0 / np.maximum(d, 1e-10) ** 2, 0.0)
    est = (w * cu[idx]).sum(axis=1) / np.where(w.sum(axis=1) > 0, w.sum(axis=1), np.nan)
    p = np.where(used[:, :, None], xyz[idx], np.nan)
    coincident = np.zeros(len(centroids), dtype=bool)
    for a in range(MAX_SAMPLES):
        for b in range(a + 1, MAX_SAMPLES):
            coincident |= np.linalg.norm(p[:, a] - p[:, b], axis=1) <= COINCIDENT
    return np.where(r.estimated, est, np.nan), coincident


def relative(ours: float, published: float) -> float:
    return ours / published - 1.0


def run(out: Path) -> dict:
    from bakumine.core.blockmodel.constraints import FileResolver, SolidConstraint
    from bakumine.core.blockmodel.partial import partial_fraction
    from bakumine.core.blockmodel.storage import load_model, save_model
    from bakumine.core.estimation.validation import cell_declustering_weights
    from bakumine.core.geometry.inside import points_in_solid

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    db, model_dir, geometry = out / "db", out / "model", out / "geometry"

    cli("import", "--collar", SRC / "collar.csv", "--survey", SRC / "survey.csv",
        "--assay", SRC / "assay.csv", "--out", db, "--auto", "--length-unit", "ft",
        "--to-metres", "--dip-convention", "positive_down", "--normalise-dip")  # fmt: skip
    cli("desurvey", db)
    composite = ("composite", db, "--table", "assay", "--mode", "downhole",
                 "--length", 10 * FT, "--attributes", "cu",
                 "--short", "discard", "--min-length", 0.5)  # fmt: skip
    cli(*composite, "--fill-missing", 0, "--out", db / "composites_zero")
    cli(*composite, "--out", db / "composites_nan")

    wire = import_wireframe(out, geometry)
    resolver = FileResolver(geometry)
    solid = resolver.solid(SOLID)

    def flag(name: str) -> tuple[pd.DataFrame, Path]:
        comp = pd.read_parquet(db / f"{name}.parquet")
        inside = points_in_solid(solid, comp[["x", "y", "z"]].to_numpy())
        comp = comp.assign(domain=inside.astype(np.int64))
        path = out / f"{name}_domain.parquet"
        comp.to_parquet(path)
        return comp, path

    comp, comp_path = flag("composites_zero")
    comp_nan, comp_nan_path = flag("composites_nan")
    dom = comp[(comp["domain"] == 1) & comp["cu"].notna()].reset_index(drop=True)
    weights = cell_declustering_weights(dom[["x", "y", "z"]].to_numpy(),
                                        metres(DECLUSTER_CELL_FT), n_offsets=8)  # fmt: skip
    cu = dom["cu"].to_numpy()
    decl_mean = float(np.average(cu, weights=weights))
    decl_var = float(np.average((cu - decl_mean) ** 2, weights=weights))

    cli("blockmodel", "create", "--overwrite", model_dir,
        "--size", ",".join(map(str, metres(BLOCK_FT))),
        "--origin", ",".join(map(str, metres(ORIGIN_FT))),
        "--shape", ",".join(map(str, SHAPE)), "--max-depth", 0)  # fmt: skip
    coding = {"attributes": [{"name": "zone", "kind": "int", "background": 0,
                              "description": "1 = block centroid inside Mpz"}],
              "steps": [{"attribute": "zone", "value": 1,
                         "constraint": {"type": "solid", "solid": SOLID}}]}  # fmt: skip
    (out / "coding.json").write_text(json.dumps(coding, indent=2))
    cli("blockmodel", "code", model_dir, "--steps", out / "coding.json",
        "--geometry-dir", geometry)  # fmt: skip
    m = load_model(model_dir)
    m.add_attribute("d1", "float", background=0.0,
                    description="fraction of the block inside Mpz (column, 5 x 5)")  # fmt: skip
    m.set("d1", partial_fraction(m, SolidConstraint(SOLID), resolver, samples=5,
                                 method="column"))  # fmt: skip
    save_model(m, model_dir)

    plans = write_plans(out)
    for name, plan in plans.items():
        cli("estimate", model_dir, "--composites", comp_path, "--plan", plan,
            "--summary", out / f"estimation_summary_{name}.json")  # fmt: skip
    cli("blockmodel", "report", model_dir, "--where", "d1 > 0 and cu >= 0", "--grade", "cu:%",
        "--density", DENSITY, "--partial", "d1", "--cutoffs", "0,0.1,0.2,0.3,0.4,0.5",
        "--out", out / "report.csv")  # fmt: skip

    m = load_model(model_dir)
    vol, d1, zone = m.volumes(), m.get("d1"), m.get("zone")
    ok, id2, nn = m.get("cu"), m.get("cu_id2"), m.get("cu_nn")
    touched = d1 > 0
    est = touched & np.isfinite(ok)
    ours = {
        "ok_mean": float(ok[est].mean()),
        "id2_mean": float(id2[est].mean()),
        "nn_mean": float(nn[est].mean()),
        "data_mean": float(cu.mean()),
        "declustered_mean": decl_mean,
        "declustered_variance": decl_var,
        "data_cv_declustered": float(np.sqrt(decl_var) / decl_mean),
        "data_max": float(cu.max()),
        "n_composites": int(len(comp)),
    }
    iso, coincident = neighbourhood_diagnostics(dom, m.centroids()[est])
    twins = dom.groupby(["x", "y", "z"])["hole_id"].agg(["nunique", "unique"])
    twins = twins[twins["nunique"] > 1]

    # the wireframe reaches 54 ft south of the model: measure that part on one more row
    strip_dir = out / "model_south_strip"
    cli("blockmodel", "create", "--overwrite", strip_dir,
        "--size", ",".join(map(str, metres(BLOCK_FT))),
        "--origin", ",".join(map(str, metres((ORIGIN_FT[0], ORIGIN_FT[1] - BLOCK_FT[1],
                                              ORIGIN_FT[2])))),
        "--shape", f"{SHAPE[0]},1,{SHAPE[2]}", "--max-depth", 0)  # fmt: skip
    strip = load_model(strip_dir)
    south = float((partial_fraction(strip, SolidConstraint(SOLID), resolver, samples=5,
                                    method="column") * strip.volumes()).sum())  # fmt: skip
    block_volume = float(np.prod(metres(BLOCK_FT)))
    lo = np.array(metres(ORIGIN_FT))
    hi = lo + np.array(metres(BLOCK_FT)) * np.array(SHAPE)
    report = pd.read_csv(out / "report.csv")

    # same run with unassayed intervals left missing (bakumine's default)
    variant_model = out / "model_missing"
    shutil.copytree(model_dir, variant_model)
    for name in ("ok", "nn"):
        overwritten = out / f"plan_{name}_overwrite.json"
        overwritten.write_text(json.dumps(json.loads(plans[name].read_text())
                                          | {"overwrite": True}))  # fmt: skip
        cli("estimate", variant_model, "--composites", comp_nan_path, "--plan", overwritten)
    v = load_model(variant_model)
    v_ok, v_nn = v.get("cu"), v.get("cu_nn")
    v_est = touched & np.isfinite(v_ok)
    dom_nan = comp_nan[(comp_nan["domain"] == 1) & comp_nan["cu"].notna()]

    metrics = {
        "published": PUBLISHED,
        "ours": ours,
        "relative_difference": {k: relative(ours[k], PUBLISHED[k]) for k in PUBLISHED},
        "wireframe": wire,
        "domain": {
            "composites_inside": int((comp["domain"] == 1).sum()),
            "composites_inside_originally_assayed": int(len(dom_nan)),
            "holes_with_composites_inside": int(dom["hole_id"].nunique()),
            "mean_cu_inside": float(cu.mean()),
            "mean_cu_outside": float(comp.loc[comp["domain"] == 0, "cu"].mean()),
            "share_of_cu_metal_inside": float(
                (dom["cu"] * dom["length"]).sum() / (comp["cu"] * comp["length"]).sum()),
        },
        "blocks": {
            "n_model": int(m.n_blocks),
            "touched_d1_gt_0": int(touched.sum()),
            "centroid_inside": int((zone == 1).sum()),
            "estimated": int(est.sum()),
            "not_estimated": int((touched & ~np.isfinite(ok)).sum()),
            "nn_estimated": int((touched & np.isfinite(nn)).sum()),
        },
        "volume": {
            "mesh_m3": wire["volume"],
            "mesh_south_of_model_m3": south,
            "mesh_inside_model_m3": wire["volume"] - south,
            "model_box_m": [lo.tolist(), hi.tolist()],
            "partial_m3": float((d1 * vol).sum()),
            "centroid_m3": float(vol[zone == 1].sum()),
            "touched_m3": float(touched.sum() * block_volume),
            "partial_vs_mesh_inside_model": relative(float((d1 * vol).sum()),
                                                     wire["volume"] - south),
            "centroid_vs_mesh_inside_model": relative(float(vol[zone == 1].sum()),
                                                      wire["volume"] - south),
        },
        "tonnage": {
            "density_t_m3": DENSITY,
            "estimated_partial_Mt": float((d1 * vol)[est].sum() * DENSITY / 1e6),
            "report_rows": report.to_dict("records"),
        },
        "diagnostics": {
            "coincident_sample_pairs_in_domain": int(len(twins)),
            "twin_holes": sorted({" = ".join(sorted(u)) for u in twins["unique"]}),
            "blocks_with_coincident_samples": int(coincident.sum()),
            "ok_mean_in_those_blocks": float(ok[est][coincident].mean()),
            "ok_mean_without_those_blocks": float(ok[est][~coincident].mean()),
            "ok_without_those_blocks_vs_published": relative(
                float(ok[est][~coincident].mean()), PUBLISHED["ok_mean"]),
            "id2_isotropic_weights_mean": float(np.nanmean(iso)),
            "id2_isotropic_vs_published": relative(float(np.nanmean(iso)), PUBLISHED["id2_mean"]),
            "ok_mean_weighted_by_d1": float(np.average(ok[est], weights=d1[est])),
            "nn_mean_weighted_by_d1": float(np.average(nn[est], weights=d1[est])),
            "ok_mean_centroid_blocks": float(np.nanmean(ok[zone == 1])),
            "nn_mean_all_touched_blocks": float(np.nanmean(nn[touched])),
        },
        "variant_missing_stays_missing": {
            "composites_inside_with_cu": int(len(dom_nan)),
            "data_mean": float(dom_nan["cu"].mean()),
            "ok_mean": float(v_ok[v_est].mean()),
            "nn_mean": float(v_nn[v_est].mean()),
            "estimated": int(v_est.sum()),
        },
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", nargs="?", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args(argv)
    m = run(args.out)
    print(f"{'':24}{'published':>14}{'ours':>14}{'difference':>12}")
    for key, pub in m["published"].items():
        print(f"{key:24}{pub:14.6g}{m['ours'][key]:14.6g}{m['relative_difference'][key]:+12.2%}")
    for key in ("wireframe", "domain", "blocks", "volume", "diagnostics",
                "variant_missing_stays_missing"):  # fmt: skip
        print(key, json.dumps(m[key], indent=1))
    print("tonnage", json.dumps({k: v for k, v in m["tonnage"].items() if k != "report_rows"}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
