"""The build spec and ``build_geology`` (P9.T14): geology_contracts.md 10, 9, 3, 14.

Independent checks: the exact truth of the synthetic layered deposit (labels on a 2.5 m
lattice, the analytic SST/SLT contact, the oxide depth function, the true fault plane and
throw); counts of the provenance record against pandas and ``re`` on the CSV files; the
volumes of the diagnostics against a lattice count on other points; a strict JSON parser
for the written files; identical labels after a save / load round trip. The build is only
compared with itself where that is the claim (round trip).
"""

from __future__ import annotations

import json
import math
import re
import time

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import Delaunay

from bakumine.core.geology.faults import Plane, block_codes, fault_picks
from bakumine.core.geology.lithology import EventSpec
from bakumine.core.geology.model import LAYERS, GeologyModel
from bakumine.core.geology.provenance import PROVENANCE_KINDS, Provenance
from bakumine.core.geology.spec import (
    GeologySpec,
    SpecError,
    _event_order,
    build_geology,
    build_summary,
    json_safe,
    load_spec,
    save_geology,
)
from bakumine.core.geology.units import UnitDictionary

LAYERED = ("SST", "SLT", "AND")
GROUP = (*LAYERED, "DIO", "GRD")
#: geology_contracts.md section 14
TARGET_LAYER, TARGET_STOCK, TARGET_DYKE = 0.85, 0.80, 0.50
TARGET_GRID_LABELS = 0.995
TARGET_CONTACT_RMS = 3.0
TARGET_OXIDE_RMS = 2.5
TARGET_FAULT = (5.0, 10.0, 5.0)  # dip, dip azimuth (degrees), position (m)
TRUE_THROW, THROW_TOLERANCE = 30.0, 6.0
TARGET_BLOCKS_INFERRED = 0.97


def strict_json(path):
    """Parse a file as standard JSON: NaN / Infinity are errors."""

    def refuse(token):
        raise ValueError(f"{path.name}: non-standard JSON token {token}")

    return json.loads(path.read_text(encoding="utf-8"), parse_constant=refuse)


def lattice(lo, hi, spacing):
    axes = [np.arange(a + spacing / 2.0, b, spacing) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


def iou(a, b):
    return float((a & b).sum() / max((a | b).sum(), 1))


def base_spec(cfg, *, faults: bool) -> dict:
    spec = {
        "format": "bakumine.geology-spec", "version": 1, "name": "layered small",
        "database": "db", "units": "units.json",
        "lithology_source": {"table": "geology", "column": "lith"},
        "topography": {"file": "topo.obj", "source": "synthetic survey"},
        "extent": {"rectangle": [cfg.x_range[0], cfg.y_range[0], cfg.x_range[1], cfg.y_range[1]],
                   "z": list(cfg.z_range)},
        "resolution": 5.0,
        "faults": None,
        "lithology": {
            "events": [
                {"mode": "sequence", "units": list(LAYERED)},
                {"mode": "intrusion", "units": ["GRD"]},  # listed out of age order on purpose
                {"mode": "intrusion", "units": ["DIO"]},
            ]
        },
        "cover": {"unit": "Q4"},
        "weathering": [
            {"zone": "OXIDE",
             "base": [{"units": ["OX"]},
                      {"threshold": {"table": "assay", "column": "s_ppm", "below": 1000}}]},
            {"zone": "TRANSITION",
             "base": [{"threshold": {"table": "assay", "column": "s_ppm", "below": 3000}}]},
        ],
        "interpolation": {"spacing": 10.0},
    }  # fmt: skip
    if faults:
        spec["faults"] = {
            "blocks": {"table": "assay", "column": "wf", "pattern": r"^(FB\d)-"},
            "fault_zone_units": ["FLT"],
            "post_fault": ["DIO", "GRD"],
            "surfaces": [{"name": "F1", "between": ["FB1", "FB2"]}],
        }
    return spec


@pytest.fixture(scope="module")
def workdir(layered_small_dir, layered_small_db, tmp_path_factory):
    """A folder with the database, the dictionary and the topography of the small deposit."""
    folder = tmp_path_factory.mktemp("geology_build")
    layered_small_db.save(folder / "db")
    for name in ("units.json", "topo.obj"):
        (folder / name).write_bytes((layered_small_dir / name).read_bytes())
    return folder


@pytest.fixture(scope="module", params=["faulted", "unfaulted"])
def built(request, workdir, layered_small):
    spec_file = workdir / f"{request.param}.json"
    document = base_spec(layered_small.config, faults=request.param == "faulted")
    spec_file.write_text(json.dumps(document), encoding="utf-8")
    spec = load_spec(spec_file)
    stages: list[tuple[str, float]] = []
    t0 = time.perf_counter()
    model = build_geology(spec, progress=lambda stage, f: stages.append((stage, f)))
    seconds = time.perf_counter() - t0
    folder = save_geology(model, workdir / f"{request.param}.geology")
    cfg = layered_small.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    pts = lattice(lo, hi, 2.5)
    pts = pts[Delaunay(model.hole_points).find_simplex(pts) >= 0]
    print(f"\nlayered-small, {request.param}: build {seconds:.2f} s")
    return {
        "how": request.param, "model": model, "folder": folder, "spec": spec, "stages": stages,
        "document": document, "box": (lo, hi), "lattice": pts, "truth": layered_small.truth,
        "seconds": seconds,
    }  # fmt: skip


# --------------------------------------------------------------------------
# validation
# --------------------------------------------------------------------------


def change(document: dict, path: str, value) -> dict:
    """A copy of ``document`` with the key at ``path`` (a.b.0.c) set, or removed (...)."""
    out = json.loads(json.dumps(document))
    keys = [int(k) if k.isdigit() else k for k in path.split(".")]
    at = out
    for key in keys[:-1]:
        at = at[key]
    if value is ...:
        del at[keys[-1]]
    else:
        at[keys[-1]] = value
    return out


CASES = [
    ("name", ..., "name", "required"),
    ("colour", "red", "spec", "allowed: format, version, name"),
    ("format", "bakumine.units", "format", "bakumine.geology-spec"),
    ("version", 2, "version", "version 1"),
    ("units", "nope.json", "units", "no file"),
    ("database", "nope", "database", "no folder"),
    ("topography", "nope.obj", "topography", "no file"),
    ("topography", None, "topography", "required when cover or weathering is built"),
    ("resolution", 0, "resolution", "> 0"),
    ("resolution", "5", "resolution", "must be a number"),
    ("max_nodes", 2.5, "max_nodes", "whole number"),
    ("extent.rectangle", [0, 0, 10], "extent.rectangle", "four numbers"),
    ("extent.rectangle", [10, 0, 0, 10], "extent.rectangle", "x0 < x1"),
    ("extent.z", [5, 5], "extent.z", "z0 < z1"),
    ("extent.box", 1, "extent", "allowed: rectangle, z"),
    ("lithology.kernel", "gaussian", "lithology.kernel", "linear, cubic, thin_plate"),
    ("lithology.cap", -1, "lithology.cap", "> 0"),
    ("lithology.events", [], "lithology.events", "at least one event"),
    ("lithology.events.0.mode", "layers", "lithology.events[0].mode",
     "argmax, sequence, intrusion"),
    ("lithology.events.0.units", ["SST", "XXX"], "lithology.events[0].units",
     "SST, SLT, AND, DIO, GRD"),
    ("lithology.events.0.units", ["SST", "Q4"], "lithology.events[0].units", "modelled units"),
    ("lithology.events.1.units", ["GRD", "DIO"], "lithology.events[1].units", "exactly one"),
    ("lithology.events.1.units", ["SST"], "lithology.events[1].units", "earlier event"),
    ("lithology.events.0.anisotropy", {"mode": "plane", "dip": 20},
     "lithology.events[0].anisotropy", "ratios"),
    ("lithology.events.1.thickness", {"GRD": 8}, "lithology.events[1].thickness", "sequence"),
    ("lithology.intrusion_order", ["DIO", "SST"], "lithology.intrusion_order",
     "intrusion units: DIO, GRD"),
    ("cover.unit", "SST", "cover.unit", "kind cover: Q4"),
    ("cover.mode", "shallowest", "cover.mode", "deepest, first_run"),
    ("weathering.0.zone", "TRANSITION", "weathering[0].zone", "OXIDE, TRANSITION"),
    ("weathering.1", ..., "weathering", "every zone but the deepest"),
    ("weathering.0.base", [], "weathering[0].base", "list of sources"),
    ("weathering.0.base.0.absent", "none", "weathering[0].base[0].absent", "unknown, zero"),
    ("weathering.0.base.1.threshold.above", 5, "weathering[0].base[1].threshold",
     "exactly one of 'below' and 'above'"),
    ("weathering.0.base.1.threshold.column", ..., "weathering[0].base[1].threshold", "'column'"),
    ("weathering.0.base.0.threshold", {"table": "assay", "column": "s_ppm", "below": 1},
     "weathering[0].base[0]", "exactly one of 'units'"),
    ("interpolation.method", "idw", "interpolation.method", "kriging, rbf"),
    ("interpolation.drift", "quadratic", "interpolation.drift", "constant, linear"),
    ("interpolation.spacing", -2, "interpolation.spacing", "> 0"),
    ("faults.blocks.pattern", "FB", "faults.blocks.pattern", "one group"),
    ("faults.blocks.pattern", "(", "faults.blocks.pattern", "regular expression"),
    ("faults.blocks", None, "faults.blocks", "inferred from the fault-block codes"),
    ("faults.surfaces", [], "faults.surfaces", "at least one fault"),
    ("faults.surfaces.0.name", "F 1", "faults.surfaces[0].name", "letters, digits"),
    ("faults.surfaces.0.between", ["FB1"], "faults.surfaces[0].between", "two different"),
    ("faults.surfaces.0.prior", {"dip": 100, "dip_azimuth": 70}, "faults.surfaces[0].prior.dip",
     "0 to 90"),
    ("faults.surfaces.0.replace_with", {"plane": {"dip": 65}},
     "faults.surfaces[0].replace_with.plane", "'point' [x, y, z], 'dip' and 'dip_azimuth'"),
    ("faults.surfaces.0.replace_with", {"file": "f.obj", "points": "p.csv"},
     "faults.surfaces[0].replace_with", "exactly one of file, points, plane"),
    ("faults.surfaces.0.terminates_on", {"fault": "F9", "side": 1},
     "faults.surfaces[0].terminates_on.fault", "another fault of the spec"),
    ("faults.surfaces.0.terminates_on", {"fault": "F2", "side": 0},
     "faults.surfaces[0].terminates_on", "'side' (1 or -1)"),
    ("faults.min_picks", 0, "faults.min_picks", ">= 1"),
    ("faults.post_fault", ["Q4"], "faults.post_fault", "modelled units"),
    ("density_source", {"table": "assay"}, "density_source", "'column'"),
    ("extra_intervals", [{"file": "a.csv", "rows": []}], "extra_intervals[0]", "exactly one"),
    ("extra_intervals", [{"rows": [{"hole_id": "A"}]}], "extra_intervals[0].rows", "depth_from"),
]  # fmt: skip


@pytest.mark.parametrize(("path", "value", "where", "text"), CASES, ids=[c[0] for c in CASES])
def test_spec_errors_name_the_field_and_what_is_allowed(
    path, value, where, text, workdir, layered_small
):
    document = change(base_spec(layered_small.config, faults=True), path, value)
    with pytest.raises(SpecError) as error:
        load_spec(document, base_dir=workdir)
    message = str(error.value)
    assert message.startswith(f"{where}:"), message
    assert text in message, message


def test_spec_file_errors(tmp_path):
    with pytest.raises(SpecError, match="does not exist"):
        load_spec(tmp_path / "none.json")
    (tmp_path / "bad.json").write_text("{", encoding="utf-8")
    with pytest.raises(SpecError, match="not valid JSON"):
        load_spec(tmp_path / "bad.json")
    with pytest.raises(SpecError, match="spec: must be an object"):
        load_spec([1, 2])  # type: ignore[arg-type]


def test_spec_is_read_with_defaults_written_out(workdir, layered_small):
    """What the spec leaves out is in ``defaults`` and written out in ``to_dict``; paths
    are resolved against the spec's folder; a dictionary may be written in the spec."""
    document = base_spec(layered_small.config, faults=True)
    del document["faults"]["fault_zone_units"], document["resolution"]
    document["units"] = json.loads((workdir / "units.json").read_text())
    spec = load_spec(document, base_dir=workdir)
    assert isinstance(spec, GeologySpec)
    assert spec.database == workdir / "db" and spec.topography == workdir / "topo.obj"
    assert spec.units.codes() == UnitDictionary.load(workdir / "units.json").codes()
    assert spec.faults.fault_zone_units == ("FLT",) and spec.faults.min_picks == 3
    assert spec.faults.min_holes_per_block == 5 and spec.faults.post_fault == ("DIO", "GRD")
    assert spec.cover.mode == "first_run" and spec.weathering[0].mode == "deepest"
    assert [e.mode for e in spec.events] == ["sequence", "intrusion", "intrusion"]
    names = [name for name, _, _ in spec.defaults]
    assert names == [
        "resolution", "max_nodes", "lithology.kernel", "faults.fault_zone_units",
        "faults.min_picks", "faults.min_holes_per_block", "cover.mode",
        "weathering[0].base[].mode", "interpolation.method", "interpolation.drift",
    ]  # fmt: skip
    written = spec.to_dict()
    assert written["faults"]["min_picks"] == 3 and written["max_nodes"] == 20_000_000
    assert written["faults"]["fault_zone_units"] == ["FLT"]
    assert written["cover"] == {"unit": "Q4", "mode": "first_run", "absent": "unknown"}
    assert written["interpolation"]["method"] == "kriging"
    assert written["units"]["units"][0]["code"] == "Q4"
    assert [d["field"] for d in written["defaults"]] == names
    again = load_spec(written, base_dir=workdir)  # the spec as run is a valid spec ...
    assert again.to_dict() == written
    assert again.defaults == spec.defaults  # ... and its defaults are still defaults
    with pytest.raises(SpecError, match="defaults: must be a list"):
        load_spec(dict(written, defaults=["resolution"]), base_dir=workdir)


def test_event_order_and_json_safe():
    """Intrusions go last in age order, the units younger than the faults after the
    others; an order the faults contradict is refused. NaN and numpy values become
    standard JSON."""
    events = [EventSpec("intrusion", ("B",)), EventSpec("argmax", ("X", "Y")),
              EventSpec("intrusion", ("A",)), EventSpec("intrusion", ("C",))]  # fmt: skip
    plain = _event_order(events, ("A", "B", "C"), ())
    assert [e.units for e in plain] == [("X", "Y"), ("A",), ("B",), ("C",)]
    young = _event_order(events, ("A", "B", "C"), ("B", "C"))
    assert [e.units for e in young] == [("X", "Y"), ("A",), ("B",), ("C",)]
    with pytest.raises(SpecError, match=r"faults\.post_fault"):
        _event_order(events, ("A", "B", "C"), ("A",))

    raw = {"a": float("nan"), "b": np.float64("inf"), "c": np.int64(3), 4: np.array([1.5, np.nan]),
           "d": (1, "x", None, True), "e": pd.DataFrame({"v": [np.nan, 2.0]})}  # fmt: skip
    safe = json_safe(raw)
    assert safe == {"a": None, "b": None, "c": 3, "4": [1.5, None], "d": [1, "x", None, True],
                    "e": [{"v": None}, {"v": 2.0}]}  # fmt: skip
    json.dumps(safe, allow_nan=False)


# --------------------------------------------------------------------------
# the build against the truth (section 14)
# --------------------------------------------------------------------------


def test_build_runs_the_steps_and_orders_the_events(built):
    model, how = built["model"], built["how"]
    stages = [s for s, _ in built["stages"]]
    assert stages == ["drillholes", "grid", "surfaces", "faults", "rock units", "grid values",
                      "checks", "done"]  # fmt: skip
    assert [f for _, f in built["stages"]] == sorted(f for _, f in built["stages"])
    assert model.categories("lith") == ("SST", "SLT", "AND", "DIO", "GRD", "Q4", "AIR")
    assert model.categories("weathering") == ("OXIDE", "TRANSITION", "FRESH", "AIR")
    assert model.blocks == (("FB1", "FB2") if how == "faulted" else ("ALL",))
    assert model.sampled and model.pooled_blocks == []
    for stack in model.stacks.values():  # the spec lists GRD before DIO; DIO is the older
        assert [(e.mode, tuple(e.units)) for e in stack.events] == [
            ("sequence", LAYERED), ("intrusion", ("DIO",)), ("intrusion", ("GRD",)),
        ]  # fmt: skip
    if how == "faulted":  # the intrusions are younger than F1: one object in both stacks
        a, b = model.stacks["FB1"].events, model.stacks["FB2"].events
        assert a[1] is b[1] and a[2] is b[2] and a[0] is not b[0]
    assert tuple(model.grid.shape) == (81, 81, 63) and model.grid.spacing.tolist() == [5.0] * 3
    assert built["seconds"] < 60.0


def test_build_labels_against_the_truth(built):
    """IoU per unit on the 2.5 m lattice inside the drilled hull, below the topography,
    grid labels against the exact ``lith`` truth (cover included on both sides).

    Section 14: layered units >= 0.85, DIO >= 0.80, GRD >= 0.50; fault blocks >= 0.97 with
    the inferred fault. The unfaulted build is the same data without the fault handling:
    its targets are asserted too (it meets them; SLT is the weakest), and the faulted
    build must beat it on SLT, the unit both of whose contacts step at the fault."""
    model, pts, truth = built["model"], built["lattice"], built["truth"]
    true = truth.labels(pts, "lith")
    got = model.decode(model.labels(pts), "lith")
    ground = (true != "AIR") & (got != "AIR")
    assert (got[ground] != None).all()  # noqa: E711
    scores = {u: iou(got[ground] == u, true[ground] == u) for u in (*GROUP, "Q4")}
    ratio = {u: float((got[ground] == u).sum() / (true[ground] == u).sum()) for u in GROUP}
    accuracy = float((got[ground] == true[ground]).mean())
    print(
        f"\nlayered-small build, {built['how']}, {int(ground.sum())} lattice points in the "
        f"hull: accuracy {accuracy:.4f}; IoU "
        + ", ".join(f"{u} {v:.3f}" for u, v in scores.items())
        + "; volume model / truth in the hull "
        + ", ".join(f"{u} {v:.3f}" for u, v in ratio.items())
    )
    for u in LAYERED:
        assert scores[u] >= TARGET_LAYER, (u, scores[u])
    assert scores["DIO"] >= TARGET_STOCK and scores["GRD"] >= TARGET_DYKE
    built["scores"] = scores
    if built["how"] == "faulted":
        blocks = model.decode(model.labels(pts, "fault_block"), "fault_block")
        block_accuracy = float((blocks == truth.labels(pts, "fault_block")).mean())
        print(f"  fault blocks right at {block_accuracy:.5f} of the points")
        assert block_accuracy >= TARGET_BLOCKS_INFERRED
        assert scores["SLT"] >= 0.90  # 0.883 without the fault (P9.T10, P9.T13)
    air = float(((got == "AIR") == (true == "AIR")).mean())
    assert air >= 0.999


def test_build_grid_labels_equal_exact_labels(built):
    """Section 14: labels with ``exact=False`` against ``exact=True`` at >= 99.5 % of
    random points of the box, for every layer the model has."""
    model = built["model"]
    lo, hi = built["box"]
    pts = np.random.default_rng(14).uniform(lo, hi, size=(20000, 3))
    for layer in LAYERS:
        if layer == "fault_block" and model.network is None:
            continue
        same = float((model.labels(pts, layer) == model.labels(pts, layer, exact=True)).mean())
        print(f"\nlayered-small build, {built['how']}, {layer}: grid = exact at {same:.4%}")
        assert same >= TARGET_GRID_LABELS, layer


def contact_rms(model, truth, lo, hi):
    """SST/SLT contact against the analytic contact inside the drilled hull, on vertical
    lines 5 m apart where the model and the truth both have exactly one such contact:
    ``(columns, RMS distance to the plane)`` (the measure of P9.T7 and P9.T13)."""
    axes = [np.arange(a + 2.5, b, 5.0) for a, b in zip(lo[:2], hi[:2], strict=True)]
    xy = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 2)
    z = np.arange(lo[2], hi[2] + 1e-9, 0.5)
    cols = np.column_stack([np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))])
    got = model.labels(cols).reshape(len(xy), len(z))
    true = truth.labels(cols, "bedrock").reshape(len(xy), len(z))
    sst, slt = model.categories("lith").index("SST"), model.categories("lith").index("SLT")
    hull = Delaunay(model.hole_points)
    separation = []
    for i, (x, y) in enumerate(xy):
        k = np.flatnonzero((got[i, 1:] == sst) & (got[i, :-1] == slt))
        j = np.flatnonzero((true[i, 1:] == "SST") & (true[i, :-1] == "SLT"))
        if len(k) != 1 or len(j) != 1:
            continue
        candidates = [float(truth.contact_z(x, y, "SST/SLT", b)) for b in ("FB1", "FB2")]
        z_true = min(candidates, key=lambda c: abs(c - (z[j[0]] + 0.25)))
        if abs(z_true - (z[j[0]] + 0.25)) > 0.3:
            continue  # SST on SLT across the fault, not across the bedding contact
        if hull.find_simplex(np.array([x, y, z_true])) >= 0:
            separation.append(z[k[0]] + 0.25 - z_true)
    distance = np.array(separation) * truth.bedding_normal[2]
    return len(distance), float(np.sqrt(np.mean(distance**2)))


def test_build_contact_surfaces_fault_and_throw(built, layered_small, layered_small_db,
                                                layered_small_traces):  # fmt: skip
    """Section 14, each against the analytic truth:

    - SST/SLT contact RMS <= 3 m inside the hull: asserted for the faulted build. The
      unfaulted build misses it (3.3 m, as P9.T10 and P9.T13 measured); that is reported,
      and asserted only as the known miss so a change is noticed;
    - oxide base RMS <= 2.5 m inside the collar grid (and the other two bases reported);
    - F1 within 5 deg dip, 10 deg dip azimuth and 5 m at the centroid of the fault picks;
    - throw (vertical separation of SST/SLT) 30 +/- 6 m, a measured entry."""
    model, truth, how = built["model"], built["truth"], built["how"]
    lo, hi = built["box"]
    n, rms = contact_rms(model, truth, lo, hi)
    print(f"\nlayered-small build, {how}: SST/SLT contact RMS {rms:.2f} m over {n} columns")
    assert n > 1000
    if how == "faulted":
        assert rms <= TARGET_CONTACT_RMS
    else:
        assert TARGET_CONTACT_RMS < rms < 3.6  # a miss of the target, known since P9.T10

    gx, gy = np.meshgrid(np.arange(lo[0] + 26.0, hi[0] - 25.0, 5.0),
                         np.arange(lo[1] + 26.0, hi[1] - 25.0, 5.0), indexing="ij")  # fmt: skip
    depth_of = {"Q4": truth.cover_depth, "OXIDE": truth.oxide_depth,
                "TRANSITION": truth.transition_depth}  # fmt: skip
    bases = {model.cover[0]: model.cover[1], **{z: b for z, b in model.weathering if b}}
    errors = {}
    for code, dtm in bases.items():
        e = truth.topo_z(gx, gy) - dtm.z_at(gx, gy) - depth_of[code](gx, gy)
        assert np.isfinite(e).all()
        errors[code] = float(np.sqrt(np.mean(e**2)))
    print("  base surfaces, RMS against the truth: "
          + ", ".join(f"{c} {v:.2f} m" for c, v in errors.items()))  # fmt: skip
    assert errors["OXIDE"] <= TARGET_OXIDE_RMS
    assert errors["Q4"] <= 1.0 and errors["TRANSITION"] <= TARGET_OXIDE_RMS
    # nested: cover base above oxide base above transition base, everywhere on the grid
    z = [bases[c].z_at(gx, gy) for c in ("Q4", "OXIDE", "TRANSITION")]
    assert (z[0] >= z[1] - 1e-9).all() and (z[1] >= z[2] - 1e-9).all()

    if how == "unfaulted":
        assert model.network is None and model.diagnostics["throw"] == []
        return
    cfg = layered_small.config
    true_plane = Plane.from_dip(cfg.fault_point, cfg.fault_dip, cfg.fault_dip_azimuth)
    fault = model.network.faults[0]
    assay, log = layered_small_db.intervals["assay"], layered_small_db.intervals["geology"]
    zones = log[log["lith"].astype(str).str.strip().str.upper().isin(["FLT", "FB"])]
    picks = fault_picks(assay, block_codes(assay["wf"], r"^(FB\d)-"), layered_small_traces,
                        fault_zone=zones)  # fmt: skip
    centroid = picks[["x", "y", "z"]].to_numpy(dtype=float).mean(axis=0)
    on_plane = centroid - fault.plane.signed_distance(centroid) * fault.plane.normal
    dip = abs(fault.plane.dip - true_plane.dip)
    azimuth = abs((fault.plane.dip_azimuth - true_plane.dip_azimuth + 180.0) % 360.0 - 180.0)
    position = abs(float(true_plane.signed_distance(on_plane)))
    throw = model.diagnostics["throw"]
    print(
        f"  F1 ({fault.source}): dip error {dip:.3f} deg, dip azimuth error {azimuth:.3f} deg, "
        f"{position:.3f} m at the centroid of {len(picks)} picks; throw SST/SLT "
        f"{throw[0]['throw']:.2f} m, SLT/AND {throw[1]['throw']:.2f} m (truth 30)"
    )
    assert fault.source == "inferred-plane" and fault.blocks == ("FB1", "FB2")
    assert dip <= TARGET_FAULT[0] and azimuth <= TARGET_FAULT[1] and position <= TARGET_FAULT[2]
    assert [(r["upper"], r["lower"]) for r in throw] == [("SST", "SLT"), ("SLT", "AND")]
    assert throw[0]["throw"] == pytest.approx(TRUE_THROW, abs=THROW_TOLERANCE)
    assert throw[0]["kind"] == "measured"
    entries = [e for e in model.provenance.entries if e.item.startswith("Throw of fault F1")]
    assert [e.kind for e in entries] == ["measured", "measured"]
    assert entries[0].value == pytest.approx(throw[0]["throw"])


def test_build_diagnostics_volumes_inside_and_outside_the_hull(built, layered_small_dir):
    """The volumes of the diagnostics: the parts add up to the box (analytic); the part
    inside the drilled hull agrees with a Monte Carlo count on random points within 3 %;
    and the reason they are reported apart is visible against ``truth.json``: the stock in
    the whole box is far from the truth while the layered units are not."""
    model, d = built["model"], built["model"].diagnostics
    volumes = d["volumes"]
    lo, hi = built["box"]
    assert volumes["hull"] and volumes["cell_volume"] == 125.0
    total = sum(v["volume"] for v in volumes["units"].values())
    assert total == pytest.approx(float(np.prod(hi - lo)), rel=1e-12)
    for v in volumes["units"].values():
        assert v["inside_hull"] + v["extrapolated"] == pytest.approx(v["volume"])
    # independent count: uniform random points of the box (Monte Carlo), inside the hull
    n = 3_000_000
    pts = np.random.default_rng(21).uniform(lo, hi, size=(n, 3))
    pts = pts[Delaunay(model.hole_points).find_simplex(pts) >= 0]
    got = model.decode(model.labels(pts), "lith")
    box_volume = float(np.prod(hi - lo))
    counted = {u: float((got == u).sum()) / n * box_volume for u in GROUP}
    for u in GROUP:  # standard error of the count: 0.2 % (SST) to 0.9 % (GRD)
        assert volumes["units"][u]["inside_hull"] == pytest.approx(counted[u], rel=0.03), u
    truth = json.loads((layered_small_dir / "truth.json").read_text())["volumes"]["units"]
    box = {u: volumes["units"][u]["volume"] / truth[u]["volume_m3"] for u in GROUP}
    share = {u: volumes["units"][u]["extrapolated_share"] for u in GROUP}
    print(
        f"\nlayered-small build, {built['how']}: volume in the whole box, model / truth: "
        + ", ".join(f"{u} {v:.3f}" for u, v in box.items())
        + "; extrapolated share: "
        + ", ".join(f"{u} {v:.2f}" for u, v in share.items())
    )
    assert box["DIO"] > 1.15  # the extrapolation below the holes; the reason for the split
    for u in LAYERED:
        assert box[u] == pytest.approx(1.0, abs=0.08), u
    entry = next(e for e in model.provenance.entries if e.item.startswith("Rock volume outside"))
    assert entry.kind == "assumed" and 0.3 < entry.value < 0.7
    res = d["residuals"]
    assert res["points"] > 5000 and res["reproduced"] >= 0.98
    assert set(res["units"]) == set(GROUP)
    assert d["cross_validation"]["run"] is False
    assert d["timing"]["total_seconds"] > 0 and d["grid"]["coarsened"] is False


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_has_the_three_kinds_and_its_counts_match_the_data(
    built, layered_small_dir, layered_small_db
):
    """Counts of the record against pandas / ``re`` on the CSV files of the deposit (the
    spellings of each unit are taken from ``truth.json``, not from the dictionary)."""
    model, how = built["model"], built["how"]
    record = model.provenance
    by_kind = {k: [e for e in record.entries if e.kind == k] for k in PROVENANCE_KINDS}
    assert all(by_kind.values())
    item = {e.item: e for e in record.entries}
    assert len(item) == len(record.entries)  # no line twice

    collar = pd.read_csv(layered_small_dir / "collar.csv")
    log = pd.read_csv(layered_small_dir / "geology.csv")
    assay = pd.read_csv(layered_small_dir / "assay.csv")
    assert item["Drillhole database"].kind == "measured"
    assert item["Drillhole database"].count == len(collar) == len(layered_small_db.collar)
    assert f"assay {len(assay)}" in item["Drillhole database"].detail
    assert item["Rock-type log"].count == len(log)
    spellings = json.loads((layered_small_dir / "truth.json").read_text())["logging"]["spellings"]
    written = log["lith"].astype(str).str.strip()
    for unit, forms in spellings.items():
        rows = written.isin([f.strip() for f in forms])
        entry = item[f"Log of unit {unit}"]
        assert entry.kind == "measured" and entry.count == int(rows.sum()), unit
        assert f"{log.loc[rows, 'hole_id'].nunique()} holes" in entry.detail
    assert item["Topography"].kind == "measured" and "synthetic survey" in item["Topography"].source
    vertices = sum(line.startswith("v ") for line in open(layered_small_dir / "topo.obj"))
    assert item["Topography"].count == vertices

    # picks: every hole gives the oxide base an exact depth from one of the two sources
    exact = [e.count for e in by_kind["measured"] if e.item.startswith("Exact picks of base:OXIDE")]
    assert sum(exact) == len(collar)
    below = int((assay["s_ppm"] < 1000).sum())
    assert f"{below} intervals inside" in item["Threshold rule for base:OXIDE"].detail

    assumed = {e.item for e in by_kind["assumed"]}
    assert {"Model extent", "Grid resolution", "Default density", "Age order of intrusions",
            "Rock volume outside the drilled hull", "Rock units beyond the drilling",
            "Default for max_nodes", "Default for lithology.kernel", "Default for cover.mode",
            "Default for interpolation.method", "Variogram of base:OXIDE"} <= assumed  # fmt: skip
    assert "unit dictionary" in item["Age order of intrusions"].source
    assert item["Age order of intrusions"].detail == "DIO < GRD"
    assert "5 m, 81 x 81 x 63 = 413343 nodes" in item["Grid resolution"].detail
    missing = {e.item for e in by_kind["missing"]}
    assert "Density samples" in missing and "base:Q4 not logged" in missing
    if how == "faulted":
        blocks = assay["wf"].astype(str).map(lambda v: bool(re.match(r"^(FB\d)-", v)))
        assert item["Fault-block codes"].count == int(blocks.sum())
        assert item["Fault F1"].kind == "assumed" and "Fault surfaces" in missing
        assert item["Rock-type log by fault block"].kind == "measured"
        assert "given" in item["Age of the faults relative to the rock units"].source
        assert "Fault data" not in missing
        assert "Default for faults.min_picks" in assumed
    else:
        assert item["Fault data"].kind == "missing" and "faults.blocks" in item["Fault data"].source
        assert not any(e.stage == "faults" and e.kind != "missing" for e in record.entries)
    counts = build_summary(model)["provenance"]
    assert counts == {k: len(v) for k, v in by_kind.items()}
    text = record.format_text()
    assert all(title in text for title in ("Measured", "Assumed", "Missing"))


def test_build_without_topography_with_defaults_and_a_coarsened_grid(workdir, layered_small):
    """No topography, no cover, no weathering, no events, no extent, no resolution: every
    gap is a line of the record, and ``max_nodes`` coarsens the grid."""
    spec = load_spec(
        {"name": "bare", "database": "db", "units": "units.json", "max_nodes": 20000},
        base_dir=workdir,
    )
    model = build_geology(spec)
    record = model.provenance
    item = {e.item: e for e in record.entries}
    for name in ("Topography", "Cover base", "Weathering bases", "Fault data", "Density samples"):
        assert item[name].kind == "missing" and item[name].source, name
    assert "survey" in item["Topography"].source and "DEM" in item["Topography"].source
    assert item["Rock-unit events"].kind == "assumed"
    assert item["Rock-unit events"].detail.startswith(
        "argmax: SST, SLT, AND; intrusion: DIO; intrusion: GRD"
    )
    assert "default" in item["Model extent"].source and "default" in item["Grid resolution"].source
    assert "coarsened from" in item["Grid resolution"].detail
    assert model.grid.n_nodes <= 20000 and model.diagnostics["grid"]["coarsened"] is True
    assert {"Default for extent", "Default for resolution"} <= set(item)
    assert model.topography is None and model.cover is None and model.weathering == ()
    assert model.categories("lith") == (*GROUP, "AIR")
    # the default box: the hole traces grown by 10 % on every side
    xyz = np.vstack([layered_small.truth.cfg.x_range, layered_small.truth.cfg.y_range]).T
    top = model.grid.origin + model.grid.spacing * (np.array(model.grid.shape) - 1)
    assert (model.grid.origin[:2] < xyz[0]).all() and (top[:2] > xyz[1] - 60.0).all()
    pts = np.random.default_rng(3).uniform(model.grid.origin, top, size=(2000, 3))
    labels = model.decode(model.labels(pts), "lith")
    assert set(labels) <= set(GROUP)  # no AIR without a topography
    assert (model.labels(pts, "weathering") == -1).all()


def test_build_errors_name_the_field(workdir, layered_small):
    document = base_spec(layered_small.config, faults=True)
    cases = [
        ("lithology_source.column", "rock", "lithology_source", "no column 'rock'"),
        ("lithology_source.table", "logs", "lithology_source", "tables: assay, geology"),
        ("faults.blocks.column", "zone", "faults.blocks", "no column 'zone'"),
        ("faults.surfaces.0.between", ["FB1", "FB9"], "faults.surfaces[0].between", "FB1, FB2"),
        ("weathering.1.base.0.threshold.column", "cu", "weathering base TRANSITION", "'cu'"),
        ("lithology.intrusion_order", ["DIO"], "lithology.intrusion_order", "every intrusion"),
        ("faults.post_fault", ["DIO"], "faults.post_fault", "cannot be older"),
    ]
    for path, value, where, text in cases:
        with pytest.raises(SpecError) as error:
            build_geology(load_spec(change(document, path, value), base_dir=workdir))
        assert str(error.value).startswith(f"{where}:") and text in str(error.value), error.value
    with pytest.raises(SpecError, match="database: is required"):
        build_geology(load_spec(change(document, "database", None), base_dir=workdir))


# --------------------------------------------------------------------------
# folder
# --------------------------------------------------------------------------


def test_folder_round_trip_and_standard_json(built):
    """The folder of section 9: every file there, every JSON file standard JSON (a NaN in
    the diagnostics is written as null), and the loaded model gives identical labels."""
    model, folder, how = built["model"], built["folder"], built["how"]
    for name in ("geology.json", "units.json", "spec.json", "provenance.json", "diagnostics.json",
                 "topography.obj", "grid/hole_points.npy", "meshes/lith/SST.obj"):  # fmt: skip
        assert (folder / name).is_file(), name
    for stem in ("base_Q4", "base_OXIDE", "base_TRANSITION"):
        for suffix in (".obj", ".json", "_picks.csv"):
            assert (folder / "surfaces" / f"{stem}{suffix}").is_file()
    documents = {p.relative_to(folder).as_posix(): strict_json(p) for p in folder.rglob("*.json")}
    assert len(documents) >= 8
    assert documents["spec.json"] == built["spec"].to_dict()
    assert documents["spec.json"]["lithology"]["kernel"] == "linear"  # a default, written out
    side = documents["surfaces/base_OXIDE.json"]
    assert side["name"] == "base:OXIDE" and side["diagnostics"]["used"]["exact"] == 64
    picks = pd.read_csv(folder / "surfaces" / "base_OXIDE_picks.csv")
    assert len(picks) == 64 and set(picks["source"]) == {"log", "threshold"}
    manifest = documents["geology.json"]
    assert manifest["cover"] == {"unit": "Q4", "surface": "surfaces/base_Q4.obj"}
    assert [w["zone"] for w in manifest["weathering"]] == ["OXIDE", "TRANSITION", "FRESH"]
    d = documents["diagnostics.json"]
    assert {"timing", "grid", "inputs", "surfaces", "throw", "residuals", "volumes"} <= set(d)
    if how == "faulted":
        assert "fault_network" in d and (folder / "faults/F1.obj").is_file()
        assert [f["name"] for f in manifest["faults"]] == ["F1"]

    loaded = GeologyModel.load(folder)
    lo, hi = built["box"]
    pts = np.random.default_rng(12).uniform(lo - 20.0, hi + 20.0, size=(20000, 3))
    for layer in LAYERS:
        np.testing.assert_array_equal(loaded.labels(pts, layer), model.labels(pts, layer))
    np.testing.assert_array_equal(
        loaded.labels(pts[:4000], exact=True), model.labels(pts[:4000], exact=True)
    )
    saved = Provenance.load(folder / "provenance.json")
    assert [e.to_dict() for e in saved.entries] == [e.to_dict() for e in model.provenance.entries]

    # a NaN in the diagnostics does not reach the file; a second save needs overwrite
    model.diagnostics["probe"] = {"value": math.nan, "array": np.array([1.0, np.inf])}
    with pytest.raises(FileExistsError):
        save_geology(model, folder)
    save_geology(model, folder, overwrite=True)
    assert strict_json(folder / "diagnostics.json")["probe"] == {
        "value": None,
        "array": [1.0, None],
    }
    del model.diagnostics["probe"]
    again = GeologyModel.load(folder)
    np.testing.assert_array_equal(again.labels(pts, "lith"), model.labels(pts, "lith"))


# --------------------------------------------------------------------------
# imported faults, extra intervals, density samples
# --------------------------------------------------------------------------


@pytest.mark.parametrize("how", ["plane", "file", "points"])
def test_build_with_an_imported_fault(how, workdir, layered_small, layered_small_dir):
    """The geologist's fault replaces the inferred one. Section 14: fault blocks >= 99 %
    with the true plane, here given as a plane, as a mesh file and as points on it."""
    cfg, truth = layered_small.config, layered_small.truth
    true_plane = Plane.from_dip(cfg.fault_point, cfg.fault_dip, cfg.fault_dip_azimuth)
    document = base_spec(cfg, faults=True)
    document.update(resolution=10.0, cover=None, weathering=[])
    if how == "plane":
        replace = {"plane": {"point": list(cfg.fault_point), "dip": cfg.fault_dip,
                             "dip_azimuth": cfg.fault_dip_azimuth}}  # fmt: skip
    elif how == "file":
        (workdir / "fault.obj").write_bytes((layered_small_dir / "truth_fault_F1.obj").read_bytes())
        replace = {"file": "fault.obj"}
    else:
        strike = np.cross(true_plane.normal, [0.0, 0.0, 1.0])
        dip_line = np.cross(true_plane.normal, strike)
        point = np.asarray(cfg.fault_point, dtype=float)
        replace = {"points": [list(point + a * strike + b * dip_line)
                              for a, b in ((0, 0), (80, 0), (0, 80), (-60, -40))]}  # fmt: skip
    document["faults"]["surfaces"][0]["replace_with"] = replace
    model = build_geology(load_spec(document, base_dir=workdir))
    fault = model.network.faults[0]
    assert fault.source == f"imported-{'mesh' if how == 'file' else how}"
    assert fault.plane.dip == pytest.approx(cfg.fault_dip, abs=0.01)
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    pts = np.random.default_rng(4).uniform(lo, hi, size=(50000, 3))
    blocks = model.decode(model.labels(pts, "fault_block"), "fault_block")
    accuracy = float((blocks == truth.labels(pts, "fault_block")).mean())
    print(f"\nimported fault ({how}): fault blocks right at {accuracy:.5f} of 50000 points")
    assert accuracy >= 0.99
    items = {e.item: e for e in model.provenance.entries}
    assert "Fault surfaces" not in items and "Fault data" not in items
    assert not any(e.kind == "assumed" and e.item == "Fault F1" for e in model.provenance.entries)
    throw = model.diagnostics["throw"][0]
    assert throw["throw"] == pytest.approx(TRUE_THROW, abs=THROW_TOLERANCE)


def test_extra_intervals_and_density_samples(workdir, layered_small, layered_small_db):
    """Intervals written in the spec give a hole without cover logging an exact pick, and
    the surface passes through it. ``density_source`` counts the samples per unit (the
    sulphur column stands in for a density column: only the counting is tested)."""
    cfg = layered_small.config
    log = layered_small_db.intervals["geology"]
    cover = log["lith"].astype(str).str.strip().str.upper().isin(["SOIL", "SOL"])
    collar = layered_small_db.collar
    hole = next(h for h in collar["hole_id"] if h not in set(log.loc[cover, "hole_id"]))
    rows = [{"hole_id": hole, "depth_from": 0.0, "depth_to": 3.0, "code": "SOIL"},
            {"hole_id": hole, "depth_from": 3.0, "depth_to": 9.0, "code": "SST"}]  # fmt: skip
    document = base_spec(cfg, faults=False)
    document.update(resolution=10.0, weathering=[],
                    extra_intervals=[{"rows": rows, "units_column": "code"}],
                    density_source={"table": "assay", "column": "s_ppm"})  # fmt: skip
    model = build_geology(load_spec(document, base_dir=workdir))
    surface = model.build.surfaces[0]
    assert surface.name == "base:Q4"
    pick = surface.picks.set_index("hole_id").loc[hole]
    assert (pick["kind"], pick["source"]) == ("exact", "extra")
    # 3 m along a hole that dips 60 degrees under a sloping ground: the depth below the
    # topography is taken at the pick's own (x, y), from the analytic topography here
    assert pick["depth"] == pytest.approx(3.0)
    below = float(layered_small.truth.topo_z(pick["x"], pick["y"])) - pick["z"]
    assert 2.0 < below < 3.0 and pick["below_lo"] == pytest.approx(below, abs=0.3)
    z = model.cover[1].z_at(np.array([pick["x"]]), np.array([pick["y"]]))
    assert float(z[0]) == pytest.approx(pick["z"], abs=0.3)  # the base passes through the pick
    items = {e.item: e for e in model.provenance.entries}
    assert "Exact picks of base:Q4 (extra)" in items
    assert "Density samples" not in items
    counted = [e for e in model.provenance.entries if e.item.startswith("Density samples in unit")]
    assert counted and all(e.kind == "measured" and "table assay" in e.source for e in counted)
    assert sum(e.count for e in counted) <= len(layered_small_db.intervals["assay"])
    assert {e.item.rsplit(" ", 1)[1] for e in counted} >= set(GROUP)


def test_max_extrapolation_reaches_the_model(workdir):
    """``lithology.max_extrapolation``: no rock unit farther than that from the logged
    holes (distance checked here with scipy's k-d tree on the model's hole points)."""
    from scipy.spatial import cKDTree

    spec = load_spec(
        {"name": "limited", "database": "db", "units": "units.json", "max_nodes": 20000,
         "lithology": {"max_extrapolation": 30.0}},
        base_dir=workdir,
    )  # fmt: skip
    model = build_geology(spec)
    assert model.max_extrapolation == 30.0
    top = model.grid.origin + model.grid.spacing * (np.array(model.grid.shape) - 1)
    pts = np.random.default_rng(6).uniform(model.grid.origin, top, size=(20000, 3))
    distance, _ = cKDTree(model.hole_points).query(pts)
    labels = model.labels(pts, "lith", exact=True)
    assert (labels[distance > 30.0] == -1).all() and (labels[distance < 29.9] >= 0).all()
    assert (distance > 30.0).mean() > 0.2
    entry = next(e for e in model.provenance.entries if e.item == "Rock units beyond the drilling")
    assert "max_extrapolation" in entry.source and "left without a rock unit" in entry.detail
    assert "(none)" in model.diagnostics["volumes"]["units"]
