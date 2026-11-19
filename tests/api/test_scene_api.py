"""Scene endpoints (P5.T8): manifest, drillhole and mesh payloads, caching, fixtures."""

from __future__ import annotations

import gzip
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from scene_fixtures import FIXTURES, TinySceneSource, build_fixtures, tiny_tables

from bakumine.api import app
from bakumine.api.binary import MAGIC, MEDIA_TYPE, pack_arrays, unpack_arrays
from bakumine.api.payloads import (
    DEFAULT_TOLERANCE,
    densify_traces,
    drillhole_payload,
    mesh_payload,
    scene_manifest,
)
from bakumine.api.routes_scene import payload_cache
from bakumine.api.scene import (
    DemoSceneSource,
    DrillholeData,
    EmptySceneSource,
    get_scene_source,
)
from bakumine.core.drillholes.importer import TableSource, import_drillholes

REPO = Path(__file__).resolve().parents[2]
DEMO = REPO / "data" / "synthetic" / "default"

pytestmark = pytest.mark.skipif(
    not (DEMO / "collar.csv").is_file(), reason="data/synthetic/default has not been generated"
)


@pytest.fixture(scope="module")
def demo() -> DemoSceneSource:
    return DemoSceneSource(DEMO)


@pytest.fixture
def client(demo):
    app.dependency_overrides[get_scene_source] = lambda: demo
    payload_cache.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


@pytest.fixture
def tiny_client():
    source = TinySceneSource()
    app.dependency_overrides[get_scene_source] = lambda: source
    payload_cache.clear()
    yield TestClient(app)
    app.dependency_overrides.clear()


def _get_arrays(client: TestClient, url: str):
    response = client.get(url)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"] == MEDIA_TYPE
    return unpack_arrays(response.content)


# ---------------------------------------------------------------------- container


def test_pack_unpack_round_trip_and_alignment():
    arrays = {
        "a": np.arange(5, dtype=np.float64),
        "b": np.arange(7, dtype=np.uint8),  # 7 bytes: the next array must be padded
        "c": np.arange(6, dtype=np.int32).reshape(2, 3) - 3,
        "d": np.zeros((0, 3), dtype=np.float32),
        "e": np.array([1.5, np.nan], dtype=">f8"),  # big-endian input is converted
    }
    payload = pack_arrays("test", {"names": ["x", "y"], "n": 2}, arrays)
    assert payload[:4] == MAGIC
    header, out = unpack_arrays(payload)
    assert header["kind"] == "test"
    assert header["meta"] == {"names": ["x", "y"], "n": 2}
    header_len = int.from_bytes(payload[8:12], "little")
    assert (12 + header_len) % 8 == 0
    assert [e["offset"] % 8 for e in header["arrays"]] == [0] * 5
    assert list(out) == list(arrays)
    for name, array in arrays.items():
        assert out[name].shape == array.shape
        assert out[name].dtype == array.dtype.newbyteorder("<")
        np.testing.assert_array_equal(out[name], array)


def test_pack_rejects_unsupported_dtypes_and_unpack_rejects_garbage():
    with pytest.raises(TypeError):
        pack_arrays("test", {}, {"a": np.arange(3, dtype=np.int64)})
    with pytest.raises(ValueError):
        pack_arrays("test", {"bad": float("nan")}, {})
    with pytest.raises(ValueError):
        unpack_arrays(b"nope")
    good = pack_arrays("test", {}, {"a": np.arange(4, dtype=np.float64)})
    with pytest.raises(ValueError):
        unpack_arrays(good[:-8])


# ---------------------------------------------------------------------- manifest


def test_manifest_lists_demo_objects(client, demo):
    response = client.get("/api/v1/scene")
    assert response.status_code == 200
    manifest = response.json()
    assert manifest["api_version"] == 1
    assert manifest["project"]["id"] == "demo"
    objects = {o["id"]: o for o in manifest["objects"]}
    assert set(objects) == {"topo", "ore-solid", "drillholes"}
    assert objects["topo"]["category"] == "surfaces" and objects["topo"]["kind"] == "mesh"
    assert objects["ore-solid"]["category"] == "solids"
    assert objects["drillholes"]["category"] == "drillholes"
    for entry in objects.values():
        assert entry["href"].startswith("/api/v1/scene/objects/")
        assert client.get(entry["href"]).status_code == 200

    for object_id, stem in (("topo", "topo"), ("ore-solid", "ore_solid")):
        vertices = np.load(DEMO / f"{stem}_vertices.npy")
        np.testing.assert_array_equal(objects[object_id]["bounds"]["min"], vertices.min(axis=0))
        np.testing.assert_array_equal(objects[object_id]["bounds"]["max"], vertices.max(axis=0))
        assert objects[object_id]["stats"]["vertices"] == len(vertices)
    assert objects["topo"]["stats"]["closed"] is False
    assert objects["ore-solid"]["stats"]["closed"] is True

    holes = objects["drillholes"]
    collar = pd.read_csv(DEMO / "collar.csv")
    assert holes["stats"]["holes"] == len(collar)
    # Bounds enclose every collar and reach below the deepest collar.
    assert holes["bounds"]["min"][0] <= collar["x"].min()
    assert holes["bounds"]["max"][2] == pytest.approx(collar["z"].max())
    assert holes["bounds"]["min"][2] < collar["z"].min() - 50

    attributes = {(a["table"], a["name"]): a for a in holes["attributes"]}
    assay = pd.read_csv(DEMO / "assay.csv")
    au = attributes[("assay", "au")]
    assert au["type"] == "numeric" and au["unit"] == "g/t"
    assert au["min"] == assay["au"].min() and au["max"] == assay["au"].max()
    assert au["min"] <= au["display_min"] < au["display_max"] <= au["max"]
    assert au["count"] == assay["au"].notna().sum() and au["missing"] == 0
    rock = attributes[("geology", "rock")]
    assert rock["type"] == "categorical"
    assert rock["categories"] == sorted(pd.read_csv(DEMO / "geology.csv")["rock"].unique())


def test_empty_source_gives_an_empty_manifest():
    assert scene_manifest(EmptySceneSource())["objects"] == []
    assert DemoSceneSource(REPO / "no-such-folder").objects() == []


# ---------------------------------------------------------------------- drillholes


def test_demo_polylines_follow_the_desurveyed_path(client, demo):
    header, a = _get_arrays(client, "/api/v1/scene/objects/drillholes/drillholes")
    meta = header["meta"]
    traces = demo.drillholes("drillholes").traces
    assert header["kind"] == "drillholes"
    assert meta["hole_ids"] == [str(h) for h in traces.hole_ids]
    assert meta["table"] is None and "seg_hole" not in a
    offsets = a["hole_offsets"]
    assert offsets.dtype == np.uint32 and a["point_xyz"].dtype == np.float64
    assert offsets[0] == 0 and offsets[-1] == len(a["point_depth"]) == meta["n_points"]

    hole_of_point = np.repeat(np.arange(len(traces.hole_ids)), np.diff(offsets))
    ids = traces.hole_ids[hole_of_point]
    xyz, _ = traces.xyz_at(ids, a["point_depth"], on_invalid="raise")
    np.testing.assert_array_equal(a["point_xyz"], xyz)

    collar = pd.read_csv(DEMO / "collar.csv").set_index("hole_id").loc[meta["hole_ids"]]
    np.testing.assert_array_equal(a["collar_xyz"], collar[["x", "y", "z"]].to_numpy())
    np.testing.assert_array_equal(a["max_depth"], collar["max_depth"].to_numpy())

    stations = traces.stations()
    for h in range(len(traces.hole_ids)):
        depth = a["point_depth"][offsets[h] : offsets[h + 1]]
        assert depth[0] == 0.0 and depth[-1] == a["max_depth"][h]
        assert (np.diff(depth) > 0).all()
        own = stations.loc[stations["hole_id"] == traces.hole_ids[h], "depth"].to_numpy()
        assert np.isin(own, depth).all()  # every station is a polyline point

    # Curved holes look curved: no chord strays more than the tolerance from the path.
    same = hole_of_point[1:] == hole_of_point[:-1]
    mid_depth = 0.5 * (a["point_depth"][1:] + a["point_depth"][:-1])[same]
    chord_mid = 0.5 * (a["point_xyz"][1:] + a["point_xyz"][:-1])[same]
    true_mid, _ = traces.xyz_at(ids[1:][same], mid_depth, on_invalid="raise")
    deviation = np.linalg.norm(chord_mid - true_mid, axis=1)
    assert deviation.max() <= DEFAULT_TOLERANCE * 1.01
    # ... and the densification did something: more points than stations.
    assert len(a["point_depth"]) > len(stations)


def test_coarser_tolerance_gives_fewer_points(demo):
    traces = demo.drillholes("drillholes").traces
    fine = densify_traces(traces, 0.01)[1]
    coarse = densify_traces(traces, 5.0)[1]
    assert len(coarse) < len(fine)
    assert len(coarse) == len(traces.stations())  # stations only
    with pytest.raises(ValueError):
        densify_traces(traces, 0.0)


def test_demo_assay_segments_equal_source_csv(client, demo):
    header, a = _get_arrays(client, "/api/v1/scene/objects/drillholes/drillholes?table=assay")
    meta = header["meta"]
    traces = demo.drillholes("drillholes").traces
    csv = pd.read_csv(DEMO / "assay.csv")
    assert meta["table"] == "assay"
    assert meta["n_segments"] == len(csv) and meta["dropped_segments"] == 0
    assert meta["attributes"] == [
        {"name": "au", "type": "numeric", "array": "value:au", "unit": "g/t"}
    ]
    np.testing.assert_array_equal(a["seg_row"], np.arange(len(csv)))
    ids = np.array(meta["hole_ids"], dtype=object)[a["seg_hole"]]
    assert list(ids) == list(csv["hole_id"])
    np.testing.assert_array_equal(a["seg_from"], csv["depth_from"].to_numpy())
    np.testing.assert_array_equal(a["seg_to"], csv["depth_to"].to_numpy())
    np.testing.assert_array_equal(a["value:au"], csv["au"].to_numpy())

    start, _ = traces.xyz_at(ids, a["seg_from"], on_invalid="raise")
    end, _ = traces.xyz_at(ids, a["seg_to"], on_invalid="raise")
    np.testing.assert_array_equal(a["seg_start"], start)
    np.testing.assert_array_equal(a["seg_end"], end)
    # Independent check of the geometry: a segment is as long as its interval, or a
    # little shorter where the hole bends.
    chord = np.linalg.norm(a["seg_end"] - a["seg_start"], axis=1)
    length = a["seg_to"] - a["seg_from"]
    assert (chord <= length + 1e-9).all() and (chord >= length * 0.999).all()


def test_demo_geology_segments_are_dictionary_encoded(client):
    header, a = _get_arrays(client, "/api/v1/scene/objects/drillholes/drillholes?table=geology")
    csv = pd.read_csv(DEMO / "geology.csv")
    (attribute,) = header["meta"]["attributes"]
    assert attribute["type"] == "categorical" and attribute["array"] == "code:rock"
    categories = np.array(attribute["categories"], dtype=object)
    assert a["code:rock"].dtype == np.int32
    assert list(categories[a["code:rock"]]) == list(csv["rock"])


def test_missing_values_stay_missing(tiny_client):
    header, a = _get_arrays(tiny_client, "/api/v1/scene/objects/holes/drillholes?table=assay")
    table = tiny_tables()["assay"]
    meta = header["meta"]
    assert meta["hole_ids"] == ["T1", "T2", "T3"]
    np.testing.assert_array_equal(a["seg_hole"], [1, 0, 0, 1, 0, 1, 2])
    au = a["value:au"]
    assert np.isnan(au[2]) and np.isfinite(np.delete(au, 2)).all()
    np.testing.assert_array_equal(np.delete(au, 2), table["au"].dropna().to_numpy(dtype=float))
    by_name = {attr["name"]: attr for attr in meta["attributes"]}
    assert by_name["rock"]["categories"] == ["DYKE", "ORE", "WASTE"]
    np.testing.assert_array_equal(a["code:rock"], [2, 2, 1, 1, -1, 0, 1])

    manifest = tiny_client.get("/api/v1/scene").json()
    attributes = {x["name"]: x for x in manifest["objects"][1]["attributes"]}
    assert attributes["au"]["missing"] == 1 and attributes["au"]["min"] == 0.031
    assert attributes["rock"]["missing"] == 1


def test_intervals_that_cannot_be_located_are_dropped_and_counted():
    tables = tiny_tables()
    extra = pd.DataFrame(
        [("T3", 30.0, 45.0, 1.0, "ORE"), ("NOPE", 0.0, 5.0, 1.0, "ORE")],
        columns=tables["assay"].columns,
    )
    tables["assay"] = pd.concat([tables["assay"], extra], ignore_index=True)
    database = import_drillholes(
        TableSource(tables["collar"], {c: c for c in tables["collar"].columns}),
        TableSource(tables["survey"], {c: c for c in tables["survey"].columns}),
        {"assay": TableSource(tables["assay"], {c: c for c in tables["assay"].columns})},
    )
    data = DrillholeData(database, database.desurvey())
    header, a = unpack_arrays(drillhole_payload(data, "holes", "assay"))
    assert header["meta"]["dropped_segments"] == 2 and header["meta"]["n_segments"] == 7
    np.testing.assert_array_equal(a["seg_row"], np.arange(7))
    assert np.isfinite(a["seg_start"]).all() and np.isfinite(a["seg_end"]).all()
    with pytest.raises(KeyError):
        drillhole_payload(data, "holes", "nope")


# ---------------------------------------------------------------------- meshes


@pytest.mark.parametrize(("object_id", "stem"), [("topo", "topo"), ("ore-solid", "ore_solid")])
def test_demo_mesh_arrays_equal_the_npy_files(client, demo, object_id, stem):
    header, a = _get_arrays(client, f"/api/v1/scene/objects/{object_id}/mesh")
    vertices = np.load(DEMO / f"{stem}_vertices.npy")
    faces = np.load(DEMO / f"{stem}_faces.npy")
    assert header["kind"] == "mesh" and set(a) == {"vertices", "faces"}
    assert a["vertices"].dtype == np.float64 and a["faces"].dtype == np.uint32
    np.testing.assert_array_equal(a["vertices"], vertices)
    np.testing.assert_array_equal(a["faces"], faces)
    assert header["meta"]["closed"] is (object_id == "ore-solid")

    header, a = _get_arrays(client, f"/api/v1/scene/objects/{object_id}/mesh?normals=1")
    normals = a["normals"]
    assert normals.dtype == np.float32 and normals.shape == vertices.shape
    np.testing.assert_allclose(np.linalg.norm(normals, axis=1), 1.0, atol=1e-6)
    # Independent: the area-weighted sum of the adjacent face normals, by accumulation.
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    acc = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(acc, faces[:, corner], cross)
    acc /= np.linalg.norm(acc, axis=1, keepdims=True)
    np.testing.assert_allclose(normals, acc, atol=1e-6)
    if object_id == "topo":
        assert (normals[:, 2] > 0).all()  # a DTM faces up


# ---------------------------------------------------------------------- HTTP behaviour


def test_unknown_objects_and_tables_are_404(client):
    assert client.get("/api/v1/scene/objects/nope/mesh").status_code == 404
    assert client.get("/api/v1/scene/objects/nope/drillholes").status_code == 404
    assert client.get("/api/v1/scene/objects/topo/drillholes").status_code == 404
    assert client.get("/api/v1/scene/objects/drillholes/mesh").status_code == 404
    assert client.get("/api/v1/scene/objects/drillholes/drillholes?table=x").status_code == 404
    url = "/api/v1/scene/objects/drillholes/drillholes?tolerance=0"
    assert client.get(url).status_code == 422


def test_etag_revalidation_and_gzip(client):
    url = "/api/v1/scene/objects/drillholes/drillholes?table=assay"
    first = client.get(url, headers={"Accept-Encoding": "gzip"})
    assert first.status_code == 200
    assert first.headers["content-encoding"] == "gzip"
    assert first.headers["cache-control"] == "private, no-cache"
    etag = first.headers["etag"]
    assert etag.startswith('"') and etag.endswith('"')

    again = client.get(url, headers={"If-None-Match": etag})
    assert again.status_code == 304 and again.content == b""
    assert again.headers["etag"] == etag
    assert client.get(url, headers={"If-None-Match": '"other"'}).status_code == 200
    other = client.get("/api/v1/scene/objects/drillholes/drillholes?table=geology")
    assert other.headers["etag"] != etag

    # Uncompressed on request; same bytes either way.
    plain = client.get(url, headers={"Accept-Encoding": "identity"})
    assert "content-encoding" not in plain.headers
    assert plain.content == first.content


def test_demo_payloads_are_fast_and_compact(capsys):
    """Cold start (CSV import + desurvey + mesh load) to all payloads in under 2 s."""
    start = time.perf_counter()
    source = DemoSceneSource(DEMO)
    manifest = scene_manifest(source)
    data = source.drillholes("drillholes")
    payloads = {
        "drillholes (traces only)": drillhole_payload(data, "drillholes"),
        "drillholes + assay": drillhole_payload(data, "drillholes", "assay"),
        "drillholes + geology": drillhole_payload(data, "drillholes", "geology"),
        "topo mesh + normals": mesh_payload(source.mesh("topo"), "topo", True),
        "ore solid mesh + normals": mesh_payload(source.mesh("ore-solid"), "ore-solid", True),
    }
    elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"demo payloads took {elapsed:.2f} s"

    sizes = {name: (len(p), len(gzip.compress(p, 4))) for name, p in payloads.items()}
    with capsys.disabled():
        print(f"\n  demo scene built in {elapsed * 1000:.0f} ms; payload sizes (raw / gzip -4):")
        print(f"    {'manifest (JSON)':28s} {len(json.dumps(manifest)):>9,d}")
        for name, (raw, packed) in sizes.items():
            print(f"    {name:28s} {raw:>9,d} / {packed:>9,d} bytes")

    # Compact: fixed bytes per row, nothing like JSON text.
    n_assay = len(data.database.intervals["assay"])
    raw, packed = sizes["drillholes + assay"]
    assert raw < n_assay * 80 + 200_000  # 80 bytes per interval + traces and header
    assert packed < 0.6 * raw
    # Clearly smaller than the same arrays written as JSON numbers (about 1.9x here).
    _, arrays = unpack_arrays(payloads["drillholes + assay"])
    as_json = sum(len(json.dumps(a.tolist())) for a in arrays.values())
    assert raw < 0.6 * as_json
    mesh = source.mesh("ore-solid")
    assert sizes["ore solid mesh + normals"][0] < mesh.n_vertices * 36 + mesh.n_faces * 12 + 1024


# ---------------------------------------------------------------------- web fixtures


def test_committed_web_fixtures_match_the_generator():
    """Regenerate with `uv run python tests/api/scene_fixtures.py` if this fails."""
    built = build_fixtures()
    for name in ("drillholes.bkma", "mesh.bkma"):
        path = FIXTURES / name
        assert path.is_file(), f"{path} is missing"
        header, arrays = unpack_arrays(path.read_bytes())
        want_header, want = unpack_arrays(built[name])
        assert header == want_header
        for key, array in want.items():
            np.testing.assert_allclose(arrays[key], array, rtol=0, atol=1e-9, equal_nan=True)
    expected = json.loads((FIXTURES / "expected.json").read_text())
    assert expected["drillholes"]["meta"]["hole_ids"] == ["T1", "T2", "T3"]
    assert set(expected) == {"drillholes", "mesh"}
