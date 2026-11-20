"""Block model transport (P5.T8) and server-side constraint filtering (P5.T5)."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pytest
from block_fixtures import FIXTURE, MODELS, build_fixture
from fastapi.testclient import TestClient

from bakumine.api import app
from bakumine.api.binary import MEDIA_TYPE, unpack_arrays
from bakumine.api.blocks import (
    AUTO_SHELL_BLOCKS,
    BlockData,
    BlockQueryError,
    block_payload,
    build_demo_block_model,
    exposed_blocks,
    parse_constraint,
    select_blocks,
    synthetic_block_model,
)
from bakumine.api.routes_scene import payload_cache
from bakumine.api.scene import DemoSceneSource, get_scene_source
from bakumine.core.blockmodel import (
    AttributeConstraint,
    BlockModel,
    DtmConstraint,
    ExpressionError,
    PlaneConstraint,
    RangeConstraint,
    SolidConstraint,
    combine,
    constraint_from_dict,
)
from bakumine.core.blockmodel.expressions import parse

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "data" / "synthetic" / "default"
BLOCKS = "/api/v1/scene/objects/block-model/blocks"

pytestmark = pytest.mark.skipif(
    not (DEMO / "truth_blocks.parquet").is_file(),
    reason="data/synthetic/default has not been generated",
)


@pytest.fixture(scope="module")
def demo() -> DemoSceneSource:
    return DemoSceneSource(DEMO)


@pytest.fixture(scope="module")
def data(demo) -> BlockData:
    return demo.blocks("block-model")


@pytest.fixture(scope="module")
def dense() -> BlockData:
    """The truth grid without sub-blocks."""
    return BlockData(build_demo_block_model(DEMO))


@pytest.fixture
def client(demo):
    app.dependency_overrides[get_scene_source] = lambda: demo
    payload_cache.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


def fetch(client, **params):
    response = client.get(BLOCKS, params=params)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == MEDIA_TYPE
    header, arrays = unpack_arrays(response.content)
    assert header["kind"] == "blocks"
    return header["meta"], arrays


# ---------------------------------------------------------------------- demo model


def test_demo_model_is_the_truth_grid_sub_blocked_on_the_ore_contact(data, dense):
    model = data.model
    truth = json.loads((DEMO / "truth.json").read_text())["blocks"]
    definition = model.definition
    assert definition.origin == tuple(truth["origin"])
    assert definition.shape == tuple(truth["shape"]) == (39, 52, 84)
    assert definition.block_size == tuple(truth["size"])
    assert definition.max_depth == (2, 2, 2)
    model.check_invariants()
    assert not model.is_dense and dense.model.is_dense
    assert set(np.unique(model.level)) == {0, 1, 2}
    assert model.volumes().sum() == pytest.approx(definition.volume)

    # Parents that were not split keep the truth table, row for row.
    import pandas as pd

    table = pd.read_parquet(DEMO / "truth_blocks.parquet")
    ijk = table[["i", "j", "k"]].to_numpy()
    flat = definition.flat_index(ijk)
    np.testing.assert_allclose(dense.model.centroids()[flat], table[["x", "y", "z"]].to_numpy())
    np.testing.assert_array_equal(
        dense.model.get("rock", flat, decode=True), table["rock"].to_numpy(dtype=object)
    )
    np.testing.assert_array_equal(dense.model["au"][flat], table["au"].to_numpy(np.float32))
    whole = model.level == 0
    np.testing.assert_array_equal(model["au"][whole], dense.model["au"][model.parent[whole]])

    # Sub-blocks are coded by their own centroid.
    solid = data.resolver.solid("ore-solid")
    inside = SolidConstraint(solid).mask(model)
    children = model.level > 0
    ore = model["rock"] == model.attribute("rock").code_of("ORE")
    np.testing.assert_array_equal(ore[children], inside[children])
    assert not np.isnan(model["au"][children & ore]).any()
    # The ore volume of the blocks approaches the solid's (truth: block-discretised).
    ore_volume = model.volumes()[ore].sum()
    assert ore_volume == pytest.approx(solid.volume, rel=0.01)
    # Missing stays missing: AIR has no grade.
    air = model["rock"] == model.attribute("rock").code_of("AIR")
    assert np.isnan(model["au"][air & children]).all()


# ---------------------------------------------------------------------- manifest


def test_manifest_entry_describes_the_model(client, data):
    manifest = client.get("/api/v1/scene").json()
    entry = next(o for o in manifest["objects"] if o["id"] == "block-model")
    assert entry["kind"] == "blocks" and entry["category"] == "blockModels"
    assert entry["name"] == "Demo block model" and entry["href"] == BLOCKS
    model = data.model
    definition = entry["definition"]
    assert definition["origin"] == list(model.definition.origin)
    assert definition["block_size"] == [10.0, 10.0, 5.0]
    assert definition["shape"] == [39, 52, 84]
    assert definition["max_depth"] == [2, 2, 2]
    assert definition["rotation"] == np.eye(3).tolist() and definition["rotated"] is False
    assert entry["stats"]["blocks"] == model.n_blocks
    assert entry["stats"]["parents"] == 170_352
    assert entry["stats"]["dense"] is False and entry["stats"]["max_level"] == 2
    box = model.definition.bounding_box()
    assert entry["bounds"] == {"min": box[0].tolist(), "max": box[1].tolist()}
    assert entry["constraint_geometry"] == {
        "solids": [{"id": "ore-solid", "name": "Ore solid"}],
        "surfaces": [{"id": "topo", "name": "Topography"}],
    }

    attributes = {a["name"]: a for a in entry["attributes"]}
    assert list(attributes) == ["rock", "au", "au_ore", "ore_fraction", "density"]
    rock = attributes["rock"]
    assert rock["type"] == "categorical" and rock["categories"] == ["AIR", "WASTE", "ORE"]
    assert rock["missing"] == 0 and rock["count"] == model.n_blocks
    au = attributes["au"]
    values = model["au"][np.isfinite(model["au"])].astype(np.float64)
    assert au["type"] == "numeric" and au["unit"] == "g/t"
    assert au["min"] == values.min() and au["max"] == values.max()
    assert au["missing"] == int(np.isnan(model["au"]).sum()) > 0
    assert au["count"] + au["missing"] == model.n_blocks
    assert au["percentiles"]["p50"] == pytest.approx(np.percentile(values, 50))
    assert au["display_min"] == au["percentiles"]["p02"]
    assert au["display_max"] == au["percentiles"]["p98"] < au["max"]
    assert attributes["density"]["unit"] == "t/m3"

    # The same entry on its own, and for a generated model.
    info = client.get(f"{BLOCKS}/info").json()
    assert info == entry
    perf = client.get("/api/v1/scene/objects/perf-1000/blocks/info").json()
    assert perf["stats"]["dense"] is True and perf["kind"] == "blocks"
    assert client.get("/api/v1/scene/objects/nope/blocks/info").status_code == 404
    assert client.get("/api/v1/scene/objects/topo/blocks").status_code == 404
    assert client.get("/api/v1/scene/objects/perf-99999999/blocks").status_code == 400


def test_rotated_definition_is_sent_as_a_matrix():
    model = MODELS["rotated"]()
    payload = block_payload(BlockData(model), "r")
    meta = unpack_arrays(payload)[0]["meta"]
    np.testing.assert_array_equal(meta["definition"]["rotation"], model.definition.rotation)
    assert meta["definition"]["rotated"] is True


# ---------------------------------------------------------------------- payload


def _decode_structure(meta, arrays, model: BlockModel):
    """What the client reconstructs: (rows, parent, level, sub)."""
    n = meta["n_blocks"]
    rows = np.arange(n) if meta["rows"] == "all" else arrays["row"].astype(np.int64)
    if meta["structure"] == "dense":
        assert "parent" not in arrays and "level" not in arrays and "sub" not in arrays
        return rows, rows, np.zeros(n, np.uint8), np.zeros((n, 3), np.uint8)
    return rows, arrays["parent"].astype(np.int64), arrays["level"], arrays["sub"]


@pytest.mark.parametrize("which", ["dense", "octree"])
@pytest.mark.parametrize("where", [None, 'rock == "ORE"', "au > 1 and z < 1100"])
def test_payload_decodes_to_the_model_columns(which, where, data, dense):
    block_data = dense if which == "dense" else data
    model = block_data.model
    names = ["au", "rock", "density"]
    payload = block_payload(block_data, "m", names, parse_constraint(where), "full")
    header, arrays = unpack_arrays(payload)
    meta = header["meta"]
    expected = (
        np.arange(model.n_blocks)
        if where is None
        else np.flatnonzero(AttributeConstraint(where).mask(model))
    )
    rows, parent, level, sub = _decode_structure(meta, arrays, model)
    np.testing.assert_array_equal(rows, expected)
    assert meta["n_total"] == model.n_blocks
    assert meta["n_selected"] == meta["n_blocks"] == len(expected)
    assert meta["rows"] == ("all" if where is None else "listed")
    assert meta["structure"] == which and meta["lod"] == "full"
    np.testing.assert_array_equal(parent, model.parent[expected])
    np.testing.assert_array_equal(level, model.level[expected])
    np.testing.assert_array_equal(sub, model.sub[expected])
    assert meta["selected_volume"] == pytest.approx(model.volumes()[expected].sum())

    # Attribute columns: f4 with NaN = missing, i2 dictionary codes.
    by_name = {a["name"]: a for a in meta["attributes"]}
    assert list(by_name) == names
    assert arrays["value:au"].dtype == np.float32 and arrays["code:rock"].dtype == np.int16
    np.testing.assert_array_equal(arrays["value:au"], model["au"][expected].astype(np.float32))
    np.testing.assert_array_equal(arrays["value:density"], model["density"][expected])
    np.testing.assert_array_equal(arrays["code:rock"], model["rock"][expected])
    assert by_name["rock"]["categories"] == ["AIR", "WASTE", "ORE"]
    assert by_name["au"]["unit"] == "g/t"
    if where is None:
        assert np.isnan(arrays["value:au"]).sum() == np.isnan(model["au"]).sum() > 0

    # The definition and the structure give back BlockModel's geometry exactly.
    again = BlockModel.from_arrays(model.definition, model.parent, model.level, model.sub)
    np.testing.assert_array_equal(again.centroids(expected), model.centroids(expected))

    # No coordinates on the wire, and the documented bytes per block.
    assert not any(a.dtype == np.float64 for a in arrays.values())
    structure = 0 if (which == "dense" and where is None) else 4 if which == "dense" else 12
    if which == "octree" and where is None:
        structure = 8
    if len(expected):
        assert meta["bytes_per_block"] == pytest.approx(structure + 4 + 4 + 2, abs=0.01)


def test_payload_without_attributes_and_with_integer_attribute():
    model = MODELS["octree"]()
    model.add_attribute("bench", "int")
    model["bench"][:] = np.arange(model.n_blocks) % 5
    header, arrays = unpack_arrays(block_payload(BlockData(model), "m"))
    assert set(arrays) == {"parent", "level", "sub"} and header["meta"]["attributes"] == []
    header, arrays = unpack_arrays(block_payload(BlockData(model), "m", ["bench"]))
    assert arrays["value:bench"].dtype == np.float32
    np.testing.assert_array_equal(arrays["value:bench"], model["bench"])


# ---------------------------------------------------------------------- where


def _constraints(data: BlockData):
    solid = SolidConstraint("ore-solid")
    return {
        "attribute expression": AttributeConstraint('au > 1 and rock == "ORE"'),
        "numeric range": AttributeConstraint("0.5 <= au < 2"),
        "inside solid": solid,
        "outside solid": SolidConstraint("ore-solid", "outside"),
        "below surface": DtmConstraint("topo", "below"),
        "above surface": DtmConstraint("topo", "above"),
        "bench": RangeConstraint.elevation(1000.0, 1010.0),
        "index slice": RangeConstraint(i=(10, 12), k=(20, 60)),
        "plane": PlaneConstraint((500600.0, 5300500.0, 1000.0), (0.3, -1.0, 0.1), "above"),
        "and": solid & RangeConstraint(j=(20, 30)) & AttributeConstraint("au > 0.8"),
        "expression": combine(
            "a and (b or not c)",
            {
                "a": DtmConstraint("topo", "below"),
                "b": AttributeConstraint("au >= 2"),
                "c": RangeConstraint(z=(None, 1100.0)),
            },
        ),
    }


def test_rows_equal_the_python_mask_for_every_constraint(client, data, capsys):
    """The viewer's selection is `constraint.mask(model)`, row for row (plan: P5.T5)."""
    model = data.model
    counts = {}
    for name, constraint in _constraints(data).items():
        expected = np.flatnonzero(constraint.mask(model, data.resolver))
        meta, arrays = fetch(
            client, where=json.dumps(constraint.to_dict()), attributes="au", lod="full"
        )
        rows = arrays["row"] if meta["rows"] == "listed" else np.arange(meta["n_blocks"])
        np.testing.assert_array_equal(rows, expected, err_msg=name)
        assert meta["n_selected"] == len(expected), name
        assert 0 < len(expected) < model.n_blocks, name
        np.testing.assert_array_equal(arrays["value:au"], model["au"][expected], err_msg=name)
        # (a plane's normal is re-normalised when read back: equal to an ulp)
        assert meta["where"] == constraint_from_dict(constraint.to_dict()).to_dict()
        counts[name] = len(expected)
    with capsys.disabled():
        print("\n  blocks selected (server rows == constraint.mask):")
        for name, count in counts.items():
            print(f"    {name:22s} {count:>8,d}")


def test_a_short_expression_is_an_attribute_constraint(client, data):
    model = data.model
    for text in ('rock == "ORE"', "au > 1", "au > 1 or isnan(au)", "density * volume > 1000"):
        meta, arrays = fetch(client, where=text, lod="full")
        expected = np.flatnonzero(AttributeConstraint(text).mask(model))
        np.testing.assert_array_equal(arrays["row"], expected)
        assert meta["where"] == {"type": "attribute", "expression": parse(text).source}
    # No `where`, or a blank one: everything.
    assert fetch(client, where="  ", lod="full")[0]["n_blocks"] == model.n_blocks


def test_invalid_constraints_are_400_with_the_engine_message(client, monkeypatch):
    ran = []
    monkeypatch.setattr("os.system", lambda *a, **k: ran.append(a))
    cases = [
        "__import__('os').system('echo pwned')",
        "__import__('os')",
        "au >",
        "au.real > 1",
        "(lambda: 1)()",
        "open('/etc/passwd')",
        "nosuch > 1",
        'au == "ORE"',
        "9 ^ 2",
    ]
    for text in cases:
        response = client.get(BLOCKS, params={"where": text})
        assert response.status_code == 400, text
        try:
            parse(text)
            message = None  # parses; the error comes from evaluation
        except ExpressionError as exc:
            message = str(exc)
        detail = response.json()["detail"]
        assert isinstance(detail, str) and detail
        if message is not None:
            assert detail == message, text
    assert not ran
    assert "__import__" in client.get(BLOCKS, params={"where": cases[1]}).json()["detail"]
    assert "nosuch" in client.get(BLOCKS, params={"where": "nosuch > 1"}).json()["detail"]

    for where, fragment in [
        ("{not json", "not valid JSON"),
        ('{"type": "teleport"}', "unknown constraint type"),
        ('{"type": "solid", "solid": "nowhere"}', "unknown solid"),
        ('{"type": "dtm", "surface": "nowhere"}', "unknown dtm"),
        ('{"type": "range"}', "at least one range"),
        ('{"type": "attribute", "expression": "import os"}', ""),
        ('{"type": "expression", "expression": "a and b", "constraints": {}}', "undefined name"),
        ('{"expression": "au > 1"}', "'type'"),
    ]:
        response = client.get(BLOCKS, params={"where": where})
        assert response.status_code == 400, where
        assert fragment in response.json()["detail"], where
    assert client.get(BLOCKS, params={"attributes": "au,nosuch"}).status_code == 400
    assert "lod" in client.get(BLOCKS, params={"lod": "tiny"}).json()["detail"]
    assert client.get(BLOCKS, params={"clip": "au >"}).status_code == 400
    with pytest.raises(BlockQueryError):
        parse_constraint("exec('1')")


# ---------------------------------------------------------------------- level of detail


def _neighbours_selected_dense(mask3: np.ndarray) -> np.ndarray:
    """Brute force on a regular grid: all six face neighbours selected (outside = no)."""
    padded = np.pad(mask3, 1, constant_values=False)
    core = (slice(1, -1),) * 3
    out = np.ones_like(mask3)
    for axis in range(3):
        for shift in (-1, 1):
            out &= np.roll(padded, shift, axis=axis)[core]
    return out


def _truly_exposed(model: BlockModel, mask: np.ndarray) -> np.ndarray:
    """Brute force for sub-blocks, on the grid of finest cells: a block is exposed when
    one of its cells touches a cell that is not selected (or the outside of the model)."""
    definition = model.definition
    depth = np.array(definition.max_depth)
    fine_shape = tuple(np.array(definition.shape) << depth)
    owner = np.empty(fine_shape, dtype=np.int32)
    corner = (model.ijk() << depth) + model.sub
    rows = np.arange(model.n_blocks)
    for level in np.unique(model.level):
        chosen = model.level == level
        units = 1 << (depth - np.minimum(level, depth))
        for a in range(units[0]):
            for b in range(units[1]):
                for c in range(units[2]):
                    cell = corner[chosen] + (a, b, c)
                    owner[cell[:, 0], cell[:, 1], cell[:, 2]] = rows[chosen]
    selected = mask[owner]
    touches = ~_neighbours_selected_dense(selected) & selected
    exposed = np.zeros(model.n_blocks, dtype=bool)
    exposed[np.unique(owner[touches])] = True
    return exposed


SELECTIONS = [None, 'rock == "ORE"', 'rock != "AIR"', "au > 0.5", "z < 1000 or au > 2"]


@pytest.mark.parametrize("where", SELECTIONS)
def test_shell_omits_only_strictly_interior_blocks_dense(where, dense):
    model = dense.model
    mask = (
        np.ones(model.n_blocks, bool) if where is None else AttributeConstraint(where).mask(model)
    )
    shell = exposed_blocks(model, mask)
    assert not (shell & ~mask).any()
    omitted = mask & ~shell
    interior = _neighbours_selected_dense(mask.reshape(model.definition.shape)).reshape(-1)
    assert interior[omitted].all()
    # On a regular grid the shell is exact: nothing interior is kept either.
    np.testing.assert_array_equal(shell, mask & ~interior)


@pytest.mark.parametrize("where", SELECTIONS)
def test_shell_omits_only_strictly_interior_blocks_sub_blocked(where, data):
    model = data.model
    mask = (
        np.ones(model.n_blocks, bool) if where is None else AttributeConstraint(where).mask(model)
    )
    shell = exposed_blocks(model, mask)
    assert not (shell & ~mask).any()
    truly = _truly_exposed(model, mask)
    omitted = mask & ~shell
    assert not truly[omitted].any(), "an exposed block was dropped"
    # Conservative, not exact: it may keep blocks that are in fact hidden.
    assert shell.sum() >= (truly & mask).sum()


def test_shell_cuts_a_solid_body_to_its_skin(client, data, capsys):
    model = data.model
    everything, _ = fetch(client, lod="shell", attributes="au")
    assert everything["lod"] == "shell" and everything["n_selected"] == model.n_blocks
    assert everything["n_blocks"] < 0.12 * model.n_blocks
    below = DtmConstraint("topo", "below")
    meta, arrays = fetch(client, where=json.dumps(below.to_dict()), lod="shell")
    selected = below.mask(model, data.resolver)
    assert meta["n_selected"] == selected.sum() and meta["n_blocks"] == len(arrays["row"])
    assert selected[arrays["row"]].all()
    assert meta["selected_volume"] == pytest.approx(model.volumes()[selected].sum())
    # auto: full below the threshold, shell above it.
    assert model.n_blocks > AUTO_SHELL_BLOCKS
    assert fetch(client)[0]["lod"] == "shell"
    ore = fetch(client, where='rock == "ORE"')[0]
    assert ore["lod"] == "full" and ore["n_blocks"] == ore["n_selected"] < AUTO_SHELL_BLOCKS
    big = BlockData(synthetic_block_model(1_000_000))
    assert select_blocks(big).lod == "shell" and select_blocks(big, lod="full").lod == "full"
    with capsys.disabled():
        print(
            f"\n  shell LOD: {everything['n_blocks']:,d} of {model.n_blocks:,d} blocks "
            f"(everything), {meta['n_blocks']:,d} of {meta['n_selected']:,d} (below topography)"
        )


def test_clip_sends_the_interior_of_the_slab(client, data):
    """A section needs the blocks inside the slab, not the shell (see bakumine.api.blocks)."""
    model = data.model
    where = AttributeConstraint('rock == "ORE"')
    front = PlaneConstraint((500600.0, 5300520.0, 1000.0), (0.0, 1.0, 0.0), "below")
    back = PlaneConstraint((500600.0, 5300480.0, 1000.0), (0.0, 1.0, 0.0), "above")
    clip = front & back
    meta, arrays = fetch(
        client,
        where=json.dumps(where.to_dict()),
        clip=json.dumps(clip.to_dict()),
        lod="shell",
        attributes="au",
    )
    expected = np.flatnonzero(where.mask(model) & clip.mask(model))
    np.testing.assert_array_equal(arrays["row"], expected)
    assert meta["clipped"] is True and meta["lod"] == "full"
    # The counts still describe the `where` selection.
    assert meta["n_selected"] == where.mask(model).sum() > meta["n_blocks"] > 0
    y = model.centroids(expected)[:, 1]
    assert y.min() > 5300480.0 and y.max() <= 5300520.0


# ---------------------------------------------------------------------- one block, caching


def test_block_detail_gives_every_attribute_at_full_precision(client, data):
    model = data.model
    for row in (0, 100_000, model.n_blocks - 1, int(np.flatnonzero(model.level == 1)[5])):
        detail = client.get(f"{BLOCKS}/detail", params={"row": row}).json()
        assert detail["row"] == row
        assert detail["block_id"] == str(int(model.block_ids([row])[0]))
        assert detail["ijk"] == model.ijk([row])[0].tolist()
        assert detail["level"] == int(model.level[row])
        assert detail["sub"] == model.sub[row].tolist()
        assert detail["centroid"] == model.centroids([row])[0].tolist()
        assert detail["size"] == model.sizes([row])[0].tolist()
        assert detail["volume"] == float(model.volumes([row])[0])
        assert list(detail["attributes"]) == model.attribute_names
        assert detail["attributes"]["rock"] == model.get("rock", [row], decode=True)[0]
        au = float(model["au"][row])
        assert detail["attributes"]["au"] == (None if np.isnan(au) else au)
        by_id = client.get(f"{BLOCKS}/detail", params={"block_id": detail["block_id"]}).json()
        assert by_id == detail
    assert client.get(f"{BLOCKS}/detail", params={"row": model.n_blocks}).status_code == 404
    assert client.get(f"{BLOCKS}/detail", params={"block_id": "7"}).status_code == 404
    assert client.get(f"{BLOCKS}/detail", params={"block_id": "x"}).status_code == 400
    assert client.get(f"{BLOCKS}/detail").status_code == 400


def test_block_payloads_have_etags_and_gzip(client):
    params = {"where": 'rock == "ORE"', "attributes": "au"}
    first = client.get(BLOCKS, params=params)
    etag = first.headers["etag"]
    assert first.headers.get("content-encoding") == "gzip"
    again = client.get(BLOCKS, params=params, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    other = client.get(BLOCKS, params={"where": "au > 1", "attributes": "au"})
    assert other.headers["etag"] != etag


# ---------------------------------------------------------------------- size and speed


def test_payload_size_per_block_and_two_million_block_build_time(data, dense, capsys):
    lines = []

    def measure(label, block_data, names, where, lod):
        start = time.perf_counter()
        payload = block_payload(block_data, "m", names, parse_constraint(where), lod)
        elapsed = time.perf_counter() - start
        meta = unpack_arrays(payload)[0]["meta"]
        packed = len(gzip.compress(payload, 4))
        n = max(1, meta["n_blocks"])
        lines.append(
            f"    {label:44s} {meta['n_blocks']:>9,d} blocks  {len(payload):>10,d} B "
            f"({len(payload) / n:5.2f} B/block; gzip {packed / n:5.2f})  {elapsed * 1000:6.0f} ms"
        )
        return meta, len(payload), elapsed

    ore = 'rock == "ORE"'
    meta, size, _ = measure("demo, sub-blocked, ORE, au", data, ["au"], ore, "full")
    assert size < meta["n_blocks"] * 16 + 4096  # 12 structure + 4 attribute
    meta, size, _ = measure("demo, sub-blocked, all, au + rock", data, ["au", "rock"], None, "full")
    assert size < meta["n_blocks"] * 14 + 4096  # 8 structure (no row) + 4 + 2
    meta, size, _ = measure("truth grid, dense, all, au", dense, ["au"], None, "full")
    assert size < meta["n_blocks"] * 4 + 4096  # implicit ordering: attributes only
    meta, size, _ = measure("truth grid, dense, ORE, au", dense, ["au"], ore, "full")
    assert size < meta["n_blocks"] * 8 + 4096

    start = time.perf_counter()
    big = BlockData(synthetic_block_model(2_000_000))
    built = time.perf_counter() - start
    assert big.model.n_blocks == 2_000_000 and big.model.is_dense
    meta, size, full_time = measure("2M synthetic, all, au (lod=full)", big, ["au"], None, "full")
    assert meta["n_blocks"] == 2_000_000 and size < 2_000_000 * 4 + 4096
    meta, size, shell_time = measure("2M synthetic, all, au (lod=auto)", big, ["au"], None, "auto")
    assert meta["lod"] == "shell" and meta["n_blocks"] < 150_000
    meta, _, ore_time = measure("2M synthetic, ORE, au (lod=auto)", big, ["au"], ore, "auto")
    assert meta["lod"] == "shell" and meta["n_blocks"] < 0.25 * meta["n_selected"]
    measure("2M synthetic, au > 1, au + rock (lod=full)", big, ["au", "rock"], "au > 1", "full")
    with capsys.disabled():
        print(f"\n  block payloads (2M synthetic model generated in {built * 1000:.0f} ms):")
        print("\n".join(lines))
    # Loose bounds: the plan allows 5 s for transfer AND display.
    assert max(full_time, shell_time, ore_time) < 2.0
    assert built < 5.0


# ---------------------------------------------------------------------- web fixtures


def test_committed_block_fixture_matches_the_generator():
    """Regenerate with `uv run python tests/api/block_fixtures.py` if this fails."""
    assert FIXTURE.is_file(), f"{FIXTURE} is missing"
    committed = json.loads(FIXTURE.read_text())
    built = json.loads(build_fixture())
    assert [c["name"] for c in committed["cases"]] == ["dense", "octree", "rotated"]
    for have, want in zip(committed["cases"], built["cases"], strict=True):
        assert set(have) == set(want)
        for key in ("definition", "parent", "level", "sub", "block_ids", "shown_rows", "ijk"):
            assert have[key] == want[key], (want["name"], key)
        for key in ("centroids", "centroids_local", "sizes", "volumes"):
            np.testing.assert_allclose(have[key], want[key], rtol=0, atol=1e-9)
        assert have["payload_meta"] == want["payload_meta"]
        for shrink in ("1", "0.8"):
            assert [r["row"] for r in have["rays"][shrink]] == [
                r["row"] for r in want["rays"][shrink]
            ]
    rotated = committed["cases"][2]
    assert rotated["definition"]["rotated"] is True
    assert any(r["row"] >= 0 for r in rotated["rays"]["1"])
