"""Geology model container, its folder and the stored fields (P9.T7): contracts 7 and 9.

Independent checks: the label rules re-implemented here from the surfaces, brute-force
distances, node counts for the mesh volumes, trimesh for closedness, and bit-identical
labels after a save / load round trip. All data are made up.
"""

from __future__ import annotations

import json
import time

import numpy as np
import pytest
import trimesh

from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geology.fields_io import load_field, load_field_inputs, save_field
from bakumine.core.geology.lithology import LithologyStack, UnitFields
from bakumine.core.geology.model import AIR, LAYERS, GeologyModel
from bakumine.core.geology.provenance import Provenance
from bakumine.core.geology.units import Unit, UnitDictionary, WeatheringZone
from bakumine.core.geometry.dtm import dtm_from_grid
from bakumine.core.geometry.implicit import fit_field
from bakumine.core.geometry.isosurface import GridSpec

LO = np.array([1000.0, 2000.0, 300.0])
SIZE = np.array([120.0, 100.0, 90.0])
H = 5.0
SHAPE = tuple(int(n) for n in SIZE / H + 1)
CAP = 15.0
NORMAL = np.array([0.25, -0.1, 1.0]) / np.linalg.norm([0.25, -0.1, 1.0])
TOPS = (NORMAL @ (LO + [60.0, 50.0, 55.0]), NORMAL @ (LO + [60.0, 50.0, 30.0]))
TOPO_EDGE = LO[0] + 90.0  # the topography ends here: beyond it, no AIR and no cover

UNITS = UnitDictionary(
    name="made-up deposit",
    units=(
        Unit(code="QC", kind="cover", name="Cover"),
        Unit(code="BOT", kind="lithology"),
        Unit(code="TOP", kind="lithology"),
        Unit(code="MID", kind="lithology"),
        Unit(code="PLUG", kind="intrusion"),
        Unit(code="CL", kind="missing"),
    ),
    weathering_zones=(
        WeatheringZone(code="OX"),
        WeatheringZone(code="TR"),
        WeatheringZone(code="FR"),
    ),
)


def ground(x, y):
    return LO[2] + 75.0 + 6.0 * np.sin((x - LO[0]) / 30.0) + 4.0 * np.cos((y - LO[1]) / 25.0)


def surface(depth: float):
    """The topography lowered by ``depth``, on a 5 m grid from the box edge to TOPO_EDGE."""
    x = np.arange(LO[0] - 10.0, TOPO_EDGE + 1e-9, 5.0)
    y = np.arange(LO[1] - 10.0, LO[1] + SIZE[1] + 10.0 + 1e-9, 5.0)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    return dtm_from_grid(ground(xx, yy) - depth, (x[0], y[0]), (5.0, 5.0))


def signed_distances(points):
    s = points @ NORMAL
    top = s - TOPS[0]
    bottom = TOPS[1] - s
    return np.clip(np.column_stack([top, np.minimum(-top, -bottom), bottom]), -CAP, CAP)


@pytest.fixture(scope="module")
def host() -> UnitFields:
    rng = np.random.default_rng(11)
    holes = rng.uniform(LO[:2], LO[:2] + SIZE[:2], size=(40, 2))
    z = np.arange(LO[2] + 2.5, LO[2] + SIZE[2], 5.0)
    points = np.column_stack([np.repeat(holes, len(z), axis=0), np.tile(z, len(holes))])
    data = signed_distances(points)
    fields = [fit_field(points, data[:, i]) for i in range(3)]
    return UnitFields(
        ("TOP", "MID", "BOT"),  # not the dictionary order
        fields,
        cap=CAP,
        sample_spacing=5.0,
        far_spacing=5.0,
        inputs=[(points, data[:, i]) for i in range(3)],
        hole_spacing=18.0,
    )


def make_model(host, **changes) -> GeologyModel:
    settings = dict(
        name="made-up geology",
        units=UNITS,
        grid=GridSpec(LO, (H, H, H), SHAPE),
        topography=surface(0.0),
        stacks={"ALL": LithologyStack("ALL", (host,))},
        cover=("QC", surface(4.0)),
        weathering=[("OX", surface(12.0)), ("TR", surface(20.0)), ("FR", None)],
        hole_points=host.inputs[0][0],
    )
    settings.update(changes)
    return GeologyModel(**settings)


@pytest.fixture(scope="module")
def model(host) -> GeologyModel:
    made = make_model(host)
    made.sample_grid()
    return made


def box_points(n: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).uniform(LO, LO + SIZE, size=(n, 3))


# --------------------------------------------------------------------------
# categories and label rules
# --------------------------------------------------------------------------


def test_categories(model):
    assert LAYERS == ("lith", "weathering", "fault_block") and AIR == "AIR"
    assert model.categories("lith") == ("BOT", "TOP", "MID", "QC", "AIR")  # dictionary order
    assert model.categories("weathering") == ("OX", "TR", "FR", "AIR")
    assert model.categories("fault_block") == ("ALL",) and model.blocks == ("ALL",)
    with pytest.raises(ValueError, match="layer"):
        model.categories("rock")
    codes = np.array([0, 4, -1, 3], dtype=np.int16)
    assert model.decode(codes, "lith").tolist() == ["BOT", "AIR", None, "QC"]
    with pytest.raises(ValueError, match="outside"):
        model.decode([5], "lith")


@pytest.mark.parametrize("exact", [True, False])
def test_lith_rules(model, host, exact):
    pts = box_points(30000, 12)
    got = model.decode(model.labels(pts, "lith", exact=exact), "lith")
    x, y, z = pts.T
    covered = x < TOPO_EDGE  # plan extent of the topography (y is covered everywhere)
    surface_z = ground(x, y)  # the DTM is a 5 m triangulation of this: compare away from it
    clear = np.abs(z - surface_z) > 0.5
    air = covered & (z > surface_z) & clear
    assert (got[air] == "AIR").all() and air.sum() > 1000
    cover = (
        covered
        & (z < surface_z)
        & (z > surface_z - 4.0)
        & clear
        & (np.abs(z - surface_z + 4.0) > 0.5)
    )
    assert (got[cover] == "QC").all() and cover.sum() > 300
    assert not np.isin(got[~covered], ["AIR", "QC"]).any()  # outside the topography's extent
    rock = (~covered | (z < surface_z - 4.5)) & clear
    assert np.isin(got[rock], ["TOP", "MID", "BOT"]).all()
    if exact:
        expected = np.array(host.units, dtype=object)[host.labels(pts[rock])]
        assert (got[rock] == expected).all()
    else:
        exact_labels = model.labels(pts, "lith", exact=True)
        assert (model.labels(pts, "lith") == exact_labels).mean() >= 0.995
    assert (got != None).all()  # noqa: E711


def test_weathering_rules(model):
    pts = box_points(30000, 13)
    got = model.decode(model.labels(pts, "weathering"), "weathering")
    x, y, z = pts.T
    depth = ground(x, y) - z
    covered = x < TOPO_EDGE
    assert (got[~covered] == None).all()  # noqa: E711
    clear = covered & (np.abs(depth[:, None] - np.array([0.0, 12.0, 20.0])).min(axis=1) > 0.5)
    expected = np.where(
        depth < 0, "AIR", np.where(depth < 12, "OX", np.where(depth < 20, "TR", "FR"))
    )
    assert (got[clear] == expected[clear]).all()
    assert set(got[clear]) == {"AIR", "OX", "TR", "FR"}
    assert (model.labels(pts, "fault_block") == 0).all()
    with pytest.raises(ValueError, match="layer"):
        model.labels(pts, "rock")
    with pytest.raises(ValueError, match=r"\(n, 3\)"):
        model.labels(pts[:, :2])


def test_distance_to_data_and_max_extrapolation(host):
    limited = make_model(host, max_extrapolation=12.0, cover=None)
    pts = box_points(4000, 14)
    brute = np.sqrt(((pts[:, None, :] - limited.hole_points[None, :, :]) ** 2).sum(axis=2)).min(
        axis=1
    )
    np.testing.assert_allclose(limited.distance_to_data(pts), brute, rtol=0, atol=1e-9)
    got = limited.decode(limited.labels(pts, exact=True), "lith")
    below = (pts[:, 0] >= TOPO_EDGE) | (ground(pts[:, 0], pts[:, 1]) - pts[:, 2] > 0.5)
    far = brute > 12.0
    assert (got[below & far] == None).all() and (below & far).sum() > 100  # noqa: E711
    assert (got[below & ~far] != None).all()  # noqa: E711
    above = (pts[:, 0] < TOPO_EDGE) & (ground(pts[:, 0], pts[:, 1]) - pts[:, 2] < -0.5)
    assert (got[above] == "AIR").all()  # AIR also far from the holes
    with pytest.raises(ValueError, match="hole points"):
        make_model(host, hole_points=None, max_extrapolation=10.0)
    with pytest.raises(ValueError, match="no hole points"):
        make_model(host, hole_points=None).distance_to_data(pts)


def test_constructor_checks(host):
    stack = LithologyStack("ALL", (host,))
    turned = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="world-aligned"):
        make_model(host, grid=GridSpec(LO, (H, H, H), SHAPE, turned))
    with pytest.raises(ValueError, match="no lithology stack"):
        make_model(host, stacks={})
    with pytest.raises(ValueError, match="unknown block"):
        make_model(host, stacks={"ALL": stack, "FB9": stack})
    with pytest.raises(ValueError, match="not in the dictionary"):
        make_model(host, units=UnitDictionary(units=(Unit(code="TOP", kind="lithology"),)))
    with pytest.raises(ValueError, match="cover unit"):
        make_model(host, cover=("ZZ", surface(4.0)))
    with pytest.raises(ValueError, match="also modelled"):
        make_model(host, cover=("TOP", surface(4.0)))
    with pytest.raises(ValueError, match="weathering zone"):
        make_model(host, weathering=[("QC", None)])
    with pytest.raises(ValueError, match="base surface"):
        make_model(host, weathering=[("OX", None), ("FR", None)])


# --------------------------------------------------------------------------
# fault hook: a stack per block (the fault layer itself: the end of this file, P9.T13)
# --------------------------------------------------------------------------


class TwoBlocks:
    """A stand-in for a FaultNetwork: block 1 east of a vertical plane."""

    blocks = ("W", "E")

    def block_index(self, points):
        return (np.asarray(points)[:, 0] > LO[0] + 60.0).astype(np.int16)


class Shifted:
    """The host fields moved down by 15 m, as the other side of a fault."""

    mode = "argmax"

    def __init__(self, host):
        self.host, self.units, self.provenance = host, host.units, []

    def node_values(self, points):
        return self.host.node_values(np.asarray(points) + [0.0, 0.0, 15.0])

    def labels_from_values(self, values):
        return self.host.labels_from_values(values)


@pytest.mark.parametrize("exact", [True, False])
def test_labels_take_the_stack_of_the_block(host, exact):
    shifted = Shifted(host)
    faulted = make_model(
        host,
        network=TwoBlocks(),
        stacks={"W": LithologyStack("W", (host,)), "E": LithologyStack("E", (shifted,))},
        topography=None,
        cover=None,
        weathering=[],
    )
    assert faulted.blocks == ("W", "E") and faulted.categories("fault_block") == ("W", "E")
    pts = box_points(8000, 15)
    east = pts[:, 0] > LO[0] + 60.0
    np.testing.assert_array_equal(faulted.labels(pts, "fault_block"), east.astype(np.int16))
    got = faulted.decode(faulted.labels(pts, exact=exact), "lith")
    names = np.array(host.units, dtype=object)
    expected = np.where(east, names[host.labels(pts + [0.0, 0.0, 15.0])], names[host.labels(pts)])
    assert (got == expected).mean() >= (1.0 if exact else 0.995)
    assert (got[east] != names[host.labels(pts)][east]).mean() > 0.1  # the blocks do differ


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------


def node_weights(shape) -> np.ndarray:
    """Trapezoid weights: the share of a cell volume that belongs to each node."""
    weight = np.ones(shape)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            weight[tuple(index)] *= 0.5
    return weight.reshape(-1)


def test_meshes_are_closed_and_fill_the_ground(model):
    """One closed mesh per rock unit, cut at the cover base. Tolerances: each mesh volume
    within 3 % of the volume of the nodes with that label; their sum within 1 % of the
    volume of all rock nodes. The cover has no mesh (thinner than a cell)."""
    t0 = time.perf_counter()
    meshes = model.meshes("lith")
    seconds = time.perf_counter() - t0
    assert list(meshes) == ["BOT", "TOP", "MID"]
    nodes = model.grid.slab_points(0, SHAPE[0]).reshape(-1, 3)
    labels = model.decode(model.labels(nodes), "lith")
    weight = node_weights(SHAPE) * H**3
    total = 0.0
    for code, mesh in meshes.items():
        other = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        assert mesh.is_closed and other.is_watertight, code
        assert mesh.volume == pytest.approx(float(other.volume), rel=1e-9)
        counted = float(weight[labels == code].sum())
        assert mesh.volume == pytest.approx(counted, rel=0.03), code
        total += mesh.volume
    solid = float(weight[np.isin(labels, ["BOT", "TOP", "MID"])].sum())
    print(f"\nmeshes: volumes sum to {total / solid:.4f} of the rock volume ({seconds:.2f} s)")
    assert total == pytest.approx(solid, rel=0.01)
    # nothing above the cover base where the topography exists
    for mesh in meshes.values():
        v = mesh.vertices[mesh.vertices[:, 0] < TOPO_EDGE - H]
        assert (v[:, 2] <= ground(v[:, 0], v[:, 1]) - 4.0 + 0.5).all()
    with pytest.raises(ValueError, match="lith"):
        model.meshes("weathering")


class Plug:
    """An overprinting event: one unit inside a sphere (what an intrusion event does)."""

    mode = "intrusion"
    units = ("PLUG",)
    centre = LO + [50.0, 50.0, 40.0]
    radius = 22.0

    def __init__(self):
        self.provenance = []

    def node_values(self, points):
        return (self.radius - np.linalg.norm(np.asarray(points) - self.centre, axis=1))[:, None]

    def labels_from_values(self, values):
        return np.where(np.asarray(values)[:, 0] > 0.0, 0, -1).astype(np.int16)


def test_meshes_with_an_overprint(host):
    """A later event cuts its unit out of the earlier ones: the sphere's mesh volume is
    within 4 % of 4/3 pi r^3 (r = 22 m on a 5 m grid; the mesh is inscribed) and the
    meshes still fill the box (0.5 %)."""
    stacked = make_model(
        host,
        stacks={"ALL": LithologyStack("ALL", (host, Plug()))},
        topography=None,
        cover=None,
        weathering=[],
    )
    assert stacked.categories("lith") == ("BOT", "TOP", "MID", "PLUG", "AIR")
    meshes = stacked.meshes()
    assert set(meshes) == {"BOT", "TOP", "MID", "PLUG"}
    sphere = 4.0 / 3.0 * np.pi * Plug.radius**3
    assert meshes["PLUG"].volume == pytest.approx(sphere, rel=0.04)
    assert meshes["PLUG"].volume < sphere
    assert sum(m.volume for m in meshes.values()) == pytest.approx(float(np.prod(SIZE)), rel=5e-3)
    without = make_model(host, topography=None, cover=None, weathering=[]).meshes()
    lost = sum(without[c].volume - meshes[c].volume for c in ("BOT", "TOP", "MID"))
    # where the sphere meets a host contact (three units in one cell) the host meshes
    # stop short of it by a sliver smaller than a cell (research.md 5.7.2): 10 %
    assert lost == pytest.approx(meshes["PLUG"].volume, rel=0.10)
    assert lost >= meshes["PLUG"].volume  # slivers are gaps, never overlaps, here
    pts = box_points(5000, 16)
    got = stacked.decode(stacked.labels(pts, exact=True), "lith")
    inside = np.linalg.norm(pts - Plug.centre, axis=1) < Plug.radius
    assert (got[inside] == "PLUG").all() and (got[~inside] != "PLUG").all()
    with pytest.raises(ValueError, match="cannot be stored"):
        stacked.save("unused")


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_extrapolation_share_in_the_provenance(model, host):
    entries = model.provenance_entries()
    entry = entries[-1]
    assert entry.item == "Rock units beyond the drilling" and entry.kind == "assumed"
    assert "36 m" in entry.source and "2 x the median hole spacing" in entry.source
    nodes = model.grid.slab_points(0, SHAPE[0]).reshape(-1, 3)
    below = ~(model.topography.height_above(nodes) > 0.0)
    distance = np.sqrt(
        ((nodes[below][:, None, :] - model.hole_points[None, ::1, :]) ** 2).sum(axis=2).min(axis=1)
    )
    assert entry.value == pytest.approx(float((distance > 36.0).mean()), abs=1e-12)
    assert 0.0 <= entry.value < 0.2
    limited = make_model(host, max_extrapolation=10.0)
    limited_entry = limited.provenance_entries()[-1]
    assert (
        "max_extrapolation" in limited_entry.source
        and "without a rock unit" in limited_entry.detail
    )
    assert limited_entry.value > entry.value
    assert make_model(host, hole_points=None).provenance_entries() == []


# --------------------------------------------------------------------------
# stored fields
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "options",
    [
        {},
        {"kernel": "cubic", "smoothing": 1e-6},
        {"degree": 0},
        {"neighbors": 30},
        {"anisotropy": Ellipsoid.from_axes((1.0, 0.2, 0.0), (-0.2, 1.0, 0.1), None, 1.0, 1.5, 3.0)},
        {
            "neighbors": 25,
            "anisotropy": Ellipsoid.from_axes(
                (1.0, 0.0, 0.0), (0.0, 1.0, 0.0), None, 1.0, 2.0, 4.0
            ),
        },
    ],
    ids=["global", "cubic-smoothed", "degree0", "local", "anisotropic", "local-anisotropic"],
)
def test_field_round_trip_is_identical(tmp_path, options):
    rng = np.random.default_rng(17)
    xyz = rng.uniform(LO, LO + SIZE, size=(300, 3))
    values = signed_distances(xyz)[:, 1]
    field = fit_field(xyz, values, **options)
    path = tmp_path / "ALL_0_MID.npz"
    save_field(path, field, xyz, values)
    with np.load(path) as archive:
        keys = set(archive.files)
        assert int(archive["neighbors"]) == (-1 if field.neighbors is None else field.neighbors)
    common = {"xyz", "values", "kernel", "degree", "smoothing", "neighbors", "anisotropy"}
    solved = {"centre", "length", "nodes", "weights", "poly"}
    assert keys == (common | solved if field.neighbors is None else common)
    again = load_field(path)
    pts = box_points(3000, 18)
    np.testing.assert_array_equal(again(pts), field(pts))  # bit-identical
    assert (again.kernel, again.degree, again.smoothing, again.neighbors) == (
        field.kernel,
        field.degree,
        field.smoothing,
        field.neighbors,
    )
    assert (again.anisotropy is None) == (field.anisotropy is None)
    assert again.topography is None and again.max_extrapolation is None  # stored without clips
    stored_xyz, stored_values = load_field_inputs(path)
    np.testing.assert_array_equal(stored_xyz, xyz)
    np.testing.assert_array_equal(stored_values, values)


def test_field_io_errors(tmp_path):
    xyz = np.random.default_rng(19).uniform(0.0, 10.0, size=(20, 3))
    field = fit_field(xyz, xyz[:, 2])
    with pytest.raises(ValueError, match="shape"):
        save_field(tmp_path / "f.npz", field, xyz, xyz[:5, 2])
    np.savez(tmp_path / "other.npz", xyz=xyz)
    with pytest.raises(ValueError, match="not a stored field"):
        load_field(tmp_path / "other.npz")
    save_field(tmp_path / "f.npz", field, xyz, xyz[:, 2])
    with np.load(tmp_path / "f.npz") as archive:
        kept = {k: archive[k] for k in archive.files if k != "weights"}
    np.savez(tmp_path / "cut.npz", **kept)
    with pytest.raises(ValueError, match="without weights"):
        load_field(tmp_path / "cut.npz")


# --------------------------------------------------------------------------
# the folder
# --------------------------------------------------------------------------

EXPECTED_FILES = {
    "geology.json",
    "units.json",
    "provenance.json",
    "diagnostics.json",
    "topography.obj",
    "grid/values_ALL_0.npy",
    "grid/hole_points.npy",
    "fields/ALL_0_TOP.npz",
    "fields/ALL_0_MID.npz",
    "fields/ALL_0_BOT.npz",
    "surfaces/base_QC.obj",
    "surfaces/base_OX.obj",
    "surfaces/base_TR.obj",
    "meshes/lith/BOT.obj",
    "meshes/lith/TOP.obj",
    "meshes/lith/MID.obj",
}


def files_of(folder) -> set[str]:
    return {p.relative_to(folder).as_posix() for p in folder.rglob("*") if p.is_file()}


@pytest.fixture(scope="module")
def saved(model, tmp_path_factory):
    folder = tmp_path_factory.mktemp("model") / "made-up.geology"
    t0 = time.perf_counter()
    out = model.save(folder)
    t1 = time.perf_counter()
    loaded = GeologyModel.load(folder)
    t2 = time.perf_counter()
    print(f"\nmodel folder: save {t1 - t0:.2f} s, load {t2 - t1:.3f} s")
    assert out == folder
    return folder, loaded


def test_folder_layout_and_manifest(model, saved):
    folder, _ = saved
    assert files_of(folder) == EXPECTED_FILES
    manifest = json.loads((folder / "geology.json").read_text())
    assert manifest["format"] == "bakumine.geology" and manifest["version"] == 1
    assert manifest["name"] == "made-up geology"
    assert manifest["coordinates"] == "x = east, y = north, z = up, metres"
    assert manifest["grid"] == {"origin": LO.tolist(), "spacing": [H, H, H], "shape": list(SHAPE)}
    assert manifest["topography"] == "topography.obj"
    assert manifest["layers"] == {
        "lith": ["BOT", "TOP", "MID", "QC", "AIR"],
        "weathering": ["OX", "TR", "FR", "AIR"],
        "fault_block": ["ALL"],
    }
    assert manifest["blocks"] == ["ALL"] and manifest["pooled_blocks"] == []
    assert manifest["faults"] == [] and manifest["fault_network"] is None
    (event,) = manifest["stacks"]["ALL"]
    assert event["mode"] == "argmax" and event["units"] == ["TOP", "MID", "BOT"]
    assert event["values"] == "grid/values_ALL_0.npy"
    assert event["fields"] == [f"fields/ALL_0_{u}.npz" for u in ("TOP", "MID", "BOT")]
    assert event["cap"] == CAP and event["anisotropy"] is None
    assert manifest["cover"] == {"unit": "QC", "surface": "surfaces/base_QC.obj"}
    assert manifest["weathering"] == [
        {"zone": "OX", "base": "surfaces/base_OX.obj"},
        {"zone": "TR", "base": "surfaces/base_TR.obj"},
        {"zone": "FR", "base": None},
    ]
    assert manifest["max_extrapolation"] is None
    assert manifest["hole_points"] == "grid/hole_points.npy"
    assert manifest["meshes"] == {
        "lith": {c: f"meshes/lith/{c}.obj" for c in ("BOT", "TOP", "MID")}
    }
    values = np.load(folder / "grid/values_ALL_0.npy")
    assert values.dtype == np.float32 and values.shape == (3, *SHAPE)
    points = np.load(folder / "grid/hole_points.npy")
    assert points.dtype == np.float64 and points.shape == model.hole_points.shape
    assert UnitDictionary.load(folder / "units.json") == UNITS
    record = Provenance.load(folder / "provenance.json")
    assert [e.item for e in record.entries] == [e.item for e in model.provenance_entries()]
    assert "timing" in json.loads((folder / "diagnostics.json").read_text())


def test_round_trip_gives_identical_labels(model, saved):
    _, loaded = saved
    assert loaded.name == model.name and loaded.blocks == model.blocks
    assert loaded.grid.shape == model.grid.shape
    np.testing.assert_array_equal(loaded.grid.origin, model.grid.origin)
    assert isinstance(loaded.grid_values("ALL", 0), np.memmap)
    assert not loaded.grid_values("ALL", 0).flags.writeable
    for layer in LAYERS:
        assert loaded.categories(layer) == model.categories(layer)
    grid = GridSpec(LO - 6.0, (4.0, 4.0, 4.0), tuple(int(n) for n in (SIZE + 12.0) / 4.0 + 1))
    pts = grid.slab_points(0, grid.shape[0]).reshape(-1, 3)  # reaches outside the model grid
    for layer in LAYERS:
        np.testing.assert_array_equal(loaded.labels(pts, layer), model.labels(pts, layer))
    some = pts[::7]
    np.testing.assert_array_equal(loaded.labels(some, exact=True), model.labels(some, exact=True))
    event, original = loaded.stacks["ALL"].events[0], model.stacks["ALL"].events[0]
    np.testing.assert_array_equal(event.values(some), original.values(some))  # bit-identical fields
    assert (event.cap, event.hole_spacing, event.units) == (CAP, 18.0, original.units)
    np.testing.assert_array_equal(loaded.hole_points, model.hole_points)
    np.testing.assert_array_equal(loaded.distance_to_data(some), model.distance_to_data(some))
    assert [e.item for e in loaded.provenance.entries] == [
        e.item for e in model.provenance_entries()
    ]


def test_overwrite_rules(model, saved, tmp_path):
    folder, loaded = saved
    with pytest.raises(FileExistsError, match="overwrite"):
        model.save(folder)
    other = tmp_path / "not-geology"
    other.mkdir()
    (other / "notes.txt").write_text("keep me")
    with pytest.raises(FileExistsError, match="not a geology folder"):
        model.save(other, overwrite=True)
    assert files_of(other) == {"notes.txt"}
    with pytest.raises(FileNotFoundError, match="geology.json"):
        GeologyModel.load(other)

    # a loaded model saved over its own folder: its grid values are mapped from the files
    # being replaced; files of later tasks (spec.json, coding/) survive
    (folder / "spec.json").write_text("{}")
    (folder / "coding").mkdir()
    (folder / "coding" / "bm.provenance.json").write_text("{}")
    (folder / "meshes" / "lith" / "STALE.obj").write_text("")
    pts = box_points(5000, 20)
    before = {layer: loaded.labels(pts, layer) for layer in LAYERS}
    loaded.save(folder, overwrite=True)
    assert files_of(folder) == EXPECTED_FILES | {"spec.json", "coding/bm.provenance.json"}
    again = GeologyModel.load(folder)
    for layer in LAYERS:
        np.testing.assert_array_equal(again.labels(pts, layer), before[layer])
    (folder / "spec.json").unlink()
    (folder / "coding" / "bm.provenance.json").unlink()
    (folder / "coding").rmdir()


def test_save_samples_the_grid_and_minimal_model(host, tmp_path):
    """No topography, cover, weathering or hole points: the optional files are absent."""
    plain = make_model(host, topography=None, cover=None, weathering=[], hole_points=None)
    assert not plain.sampled
    folder = plain.save(tmp_path / "plain.geology")
    assert plain.sampled
    assert files_of(folder) == {
        f for f in EXPECTED_FILES if not f.startswith(("surfaces/", "topography", "grid/hole"))
    }
    manifest = json.loads((folder / "geology.json").read_text())
    assert manifest["topography"] is None and manifest["cover"] is None
    assert manifest["weathering"] == [] and manifest["hole_points"] is None
    assert manifest["layers"]["lith"] == ["BOT", "TOP", "MID", "AIR"]
    loaded = GeologyModel.load(folder)
    pts = box_points(5000, 21)
    np.testing.assert_array_equal(loaded.labels(pts), plain.labels(pts))
    assert (loaded.labels(pts, "weathering") == -1).all()
    assert loaded.topography is None and loaded.hole_points is None


def test_load_refuses_other_files(saved, tmp_path):
    folder, _ = saved
    manifest = json.loads((folder / "geology.json").read_text())
    for change, message in (
        ({"format": "bakumine.units"}, "format"),
        ({"version": 2}, "version"),
    ):
        copy = tmp_path / message
        copy.mkdir()
        (copy / "geology.json").write_text(json.dumps({**manifest, **change}))
        with pytest.raises(ValueError, match=message):
            GeologyModel.load(copy)


# --------------------------------------------------------------------------
# the fault layer of the folder (P9.T13)
# --------------------------------------------------------------------------

FAULT_X = LO[0] + 61.0  # between the node planes of the grid and of the test lattices
DROP = 15.0  # the east block's contacts are this much lower


def shifted_host(host: UnitFields) -> UnitFields:
    """The host fields fitted again from the same points with the contacts 15 m lower."""
    points = host.inputs[0][0]
    data = signed_distances(points + [0.0, 0.0, DROP])
    return UnitFields(
        host.units,
        [fit_field(points, data[:, i]) for i in range(3)],
        cap=CAP,
        sample_spacing=5.0,
        far_spacing=5.0,
        inputs=[(points, data[:, i]) for i in range(3)],
        hole_spacing=18.0,
    )


def fault_surface(kind: str):
    """The vertical fault x = FAULT_X as a plane or as an imported mesh (two triangles)."""
    from bakumine.core.geology.faults import Plane, import_fault
    from bakumine.core.geometry.mesh import Mesh

    if kind == "plane":
        point = (FAULT_X, LO[1] + 50.0, LO[2] + 45.0)
        return import_fault("F1", plane=Plane.from_dip(point, 90.0, 90.0), blocks=("W", "E"))
    y0, y1 = LO[1] - 20.0, LO[1] + SIZE[1] + 20.0
    z0, z1 = LO[2] - 20.0, LO[2] + SIZE[2] + 20.0
    vertices = np.array(
        [[FAULT_X, y0, z0], [FAULT_X, y1, z0], [FAULT_X, y1, z1], [FAULT_X, y0, z1]]
    )
    faces = np.array([[0, 1, 2], [0, 2, 3]])
    return import_fault("F1", mesh=Mesh(vertices, faces), blocks=("W", "E"))


def faulted_model(host, kind: str, *, pooled: bool = False) -> GeologyModel:
    from bakumine.core.geology.faults import FaultNetwork

    network = FaultNetwork.fit([fault_surface(kind)], np.zeros((0, 3)), [])
    west = LithologyStack("ALL" if pooled else "W", (host,))
    east = west if pooled else LithologyStack("E", (shifted_host(host),))
    made = make_model(host, network=network, stacks={"W": west, "E": east})
    made.pooled_blocks = ["E", "W"] if pooled else []
    return made


@pytest.mark.parametrize("kind", ["plane", "mesh"])
def test_fault_layer_round_trip(host, tmp_path, kind):
    """A model with a fault network: the folder holds the fault layer, and the loaded
    model gives identical labels for every layer, from the grid and exactly."""
    model = faulted_model(host, kind)
    assert model.blocks == ("E", "W")
    folder = model.save(tmp_path / "faulted.geology")
    fault_files = {"faults/F1.json", "faults/F1.obj", "grid/fault_block.npy"}
    if kind == "mesh":
        fault_files.add("faults/F1_source.obj")
    assert files_of(folder) == (
        EXPECTED_FILES
        - {"grid/values_ALL_0.npy"}
        - {f"fields/ALL_0_{u}.npz" for u in ("TOP", "MID", "BOT")}
        | {f"grid/values_{b}_0.npy" for b in ("E", "W")}
        | {f"fields/{b}_0_{u}.npz" for b in ("E", "W") for u in ("TOP", "MID", "BOT")}
        | fault_files
    )
    manifest = json.loads((folder / "geology.json").read_text())
    assert manifest["blocks"] == ["E", "W"] and manifest["layers"]["fault_block"] == ["E", "W"]
    assert manifest["faults"] == [{"name": "F1", "file": "faults/F1.json", "mesh": "faults/F1.obj"}]
    assert manifest["fault_network"] == {
        "cells": [{"signs": [-1], "block": "W"}, {"signs": [1], "block": "E"}],
        "terminations": {},
    }
    assert [e["values"] for e in manifest["stacks"]["E"]] == ["grid/values_E_0.npy"]
    blocks = np.load(folder / "grid/fault_block.npy")
    assert blocks.dtype == np.int16 and blocks.shape == SHAPE
    east_nodes = LO[0] + H * np.arange(SHAPE[0]) > FAULT_X
    np.testing.assert_array_equal(blocks[:, 0, 0], np.where(east_nodes, 0, 1))
    record = Provenance.load(folder / "provenance.json")
    assert any(e.stage == "faults" for e in record.entries)  # the fault's own lines are kept
    assert "fault_network" in json.loads((folder / "diagnostics.json").read_text())

    loaded = GeologyModel.load(folder)
    assert loaded.blocks == model.blocks and loaded.network.cells == model.network.cells
    fault = loaded.network.faults[0]
    assert fault.source == ("imported-plane" if kind == "plane" else "imported-mesh")
    assert (fault.mesh is not None) == (kind == "mesh") and fault.blocks == ("W", "E")
    grid = GridSpec(LO - 6.0, (4.0, 4.0, 4.0), tuple(int(n) for n in (SIZE + 12.0) / 4.0 + 1))
    pts = grid.slab_points(0, grid.shape[0]).reshape(-1, 3)  # reaches outside the model grid
    for layer in LAYERS:
        np.testing.assert_array_equal(loaded.labels(pts, layer), model.labels(pts, layer))
    some = pts[::7]
    np.testing.assert_array_equal(loaded.labels(some, exact=True), model.labels(some, exact=True))
    # the fault block is the side of the fault, and the two blocks do differ in rock units
    east = pts[:, 0] > FAULT_X
    np.testing.assert_array_equal(
        loaded.decode(loaded.labels(pts, "fault_block"), "fault_block"), np.where(east, "E", "W")
    )
    names = np.array(model.categories("lith"), dtype=object)
    inside = east & (pts[:, 0] < TOPO_EDGE - H) & (pts[:, 2] < LO[2] + 60.0)
    lowered = names[make_model(host).labels(pts[inside] + [0.0, 0.0, DROP], exact=True)]
    # (two separate fits of the same layers, so not identical)
    assert (names[loaded.labels(pts[inside], exact=True)] == lowered).mean() > 0.95
    # saving the loaded model again (values memory-mapped from the folder) changes nothing
    loaded.save(folder, overwrite=True)
    again = GeologyModel.load(folder)
    np.testing.assert_array_equal(again.labels(pts), model.labels(pts))
    assert (folder / "faults/F1_source.obj").is_file() == (kind == "mesh")


def test_fault_layer_meshes_are_cut_at_the_fault(host):
    """Unit meshes across two blocks: closed, each within 3 % of the volume of the nodes
    with that label, their sum within 1 % of all rock nodes (no gaps, no overlaps), and
    the step at the fault is in the mesh."""
    model = faulted_model(host, "plane")
    meshes = model.meshes("lith")
    assert list(meshes) == ["BOT", "TOP", "MID"]
    nodes = model.grid.slab_points(0, SHAPE[0]).reshape(-1, 3)
    labels = model.decode(model.labels(nodes), "lith")
    weight = node_weights(SHAPE) * H**3
    total = 0.0
    for code, mesh in meshes.items():
        other = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        assert mesh.is_closed and other.is_watertight, code
        assert mesh.volume == pytest.approx(float(weight[labels == code].sum()), rel=0.03), code
        total += mesh.volume
    solid = float(weight[np.isin(labels, ["BOT", "TOP", "MID"])].sum())
    print(f"\nfaulted meshes: volumes sum to {total / solid:.4f} of the rock volume")
    assert total == pytest.approx(solid, rel=0.01)
    # BOT's top is DROP lower east of the fault than west of it (same y, 10 m either side)
    v = meshes["BOT"].vertices
    row = np.abs(v[:, 1] - (LO[1] + 50.0)) < 0.01
    west = v[row & (np.abs(v[:, 0] - (FAULT_X - 11.0)) < 0.01), 2].max()
    east = v[row & (np.abs(v[:, 0] - (FAULT_X + 14.0)) < 0.01), 2].max()
    slope = -25.0 * NORMAL[0] / NORMAL[2]  # the contact's own change over the 25 m between them
    assert west - east == pytest.approx(DROP - slope, abs=2.0)


def test_pooled_stack_is_stored_once(host, tmp_path):
    """Two blocks that share one stack (pooled): one set of files, named after the first
    block of ``stacks`` that uses it; after loading both blocks hold the same event and the
    same grid array, and the labels are those of the model without faults."""
    model = faulted_model(host, "plane", pooled=True)
    folder = model.save(tmp_path / "pooled.geology")
    manifest = json.loads((folder / "geology.json").read_text())
    assert manifest["pooled_blocks"] == ["E", "W"]
    assert manifest["stacks"]["E"] == manifest["stacks"]["W"]
    assert manifest["stacks"]["E"][0]["values"] == "grid/values_W_0.npy"
    assert not (folder / "grid/values_E_0.npy").exists()
    assert sorted(p.name for p in (folder / "fields").iterdir()) == sorted(
        f"W_0_{u}.npz" for u in ("TOP", "MID", "BOT")
    )
    loaded = GeologyModel.load(folder)
    assert loaded.pooled_blocks == ["E", "W"]
    assert loaded.stacks["E"].events[0] is loaded.stacks["W"].events[0]
    assert loaded.grid_values("E", 0) is loaded.grid_values("W", 0)
    pts = box_points(8000, 21)
    plain = make_model(host)
    for exact in (False, True):
        np.testing.assert_array_equal(
            loaded.labels(pts, exact=exact), plain.labels(pts, exact=exact)
        )
    record = [e.item for e in model.provenance_entries()]
    assert len(record) == len(set(record))
