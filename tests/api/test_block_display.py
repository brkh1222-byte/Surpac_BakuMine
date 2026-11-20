"""What the viewer draws of a block model: blocks without a value of the colour attribute
(``present``) and the defaults a project model opens with (``display`` hints)."""

from __future__ import annotations

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

from bakumine.api import app, blocks
from bakumine.api.binary import unpack_arrays
from bakumine.api.blocks import (
    BlockData,
    BlockQueryError,
    block_payload,
    default_display,
    exposed_blocks,
    select_blocks,
)
from bakumine.api.project_source import ProjectSceneSource
from bakumine.api.routes_scene import payload_cache
from bakumine.api.scene import get_scene_source
from bakumine.core.blockmodel import AttributeConstraint, BlockModel, PlaneConstraint
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.blockmodel.storage import save_model
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.io.obj import write_obj
from bakumine.core.project import Project

SHAPE = (14, 12, 10)


def ore_model(max_depth: int = 0) -> BlockModel:
    """AIR on top, WASTE around, an ORE ellipsoid strictly inside; ``au`` in the ORE only."""
    definition = BlockModelDefinition(
        origin=(1000.0, 2000.0, 300.0), block_size=(10.0, 10.0, 5.0), shape=SHAPE,
        max_depth=max_depth,
    )  # fmt: skip
    model = BlockModel(definition)
    if max_depth:
        # Sub-blocks across the ore and well outside it, so that the contact is mixed.
        ijk = model.ijk()
        model.split((ijk[:, 0] >= 5) & (ijk[:, 0] <= 8) & (ijk[:, 1] % 2 == 0))
    model.add_attribute("rock", "category", categories=["WASTE", "AIR", "ORE"], dtype=np.int16)
    model.add_attribute("density", "float", unit="t/m3")
    model.add_attribute("au", "float", unit="g/t")
    model.add_attribute("au_pass", "int", dtype=np.int8)
    model.add_attribute("au_dist", "float")
    xyz = model.centroids()
    centre = np.array([1070.0, 2060.0, 322.0])
    radius = np.sqrt((((xyz - centre) / (38.0, 33.0, 14.0)) ** 2).sum(axis=1))
    air = xyz[:, 2] > 340.0
    ore = (radius < 1.0) & ~air
    model.set("rock", "WASTE")
    model.set("rock", "AIR", air)
    model.set("rock", "ORE", ore)
    model.set("density", 2.7, ~air)
    model.set("au", 0.5 + 3.0 * (1.0 - radius[ore]), ore)
    model.set("au_pass", 1, ore)
    model.set("au_dist", 20.0 * radius[ore], ore)
    return model


def neighbours_all(mask3: np.ndarray) -> np.ndarray:
    """Brute force on a regular grid: all six face neighbours selected (outside = no)."""
    padded = np.pad(mask3, 1, constant_values=False)
    out = np.ones_like(mask3)
    for axis in range(3):
        for shift in (-1, 1):
            out &= np.roll(padded, shift, axis=axis)[(slice(1, -1),) * 3]
    return out


def touching(model: BlockModel, mask: np.ndarray) -> np.ndarray:
    """Brute force with sub-blocks, on the grid of finest cells: the selected blocks one of
    whose cells touches a cell that is not selected (or the outside of the model)."""
    depth = np.array(model.definition.max_depth)
    owner = np.empty(tuple(np.array(model.definition.shape) << depth), dtype=np.int64)
    corner = (model.ijk() << depth) + model.sub
    for row in range(model.n_blocks):
        lo = corner[row]
        hi = lo + (1 << (depth - model.level[row]))
        owner[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = row
    selected = mask[owner]
    out = np.zeros(model.n_blocks, dtype=bool)
    out[np.unique(owner[selected & ~neighbours_all(selected)])] = True
    return out


# ---------------------------------------------------------------------- present


def test_shell_of_the_valued_blocks_when_missing_values_are_hidden():
    """The reported failure: colour by a grade that exists strictly inside the model."""
    model = ore_model()
    data = BlockData(model)
    valued = ~np.isnan(model["au"])
    grid = valued.reshape(SHAPE)
    assert valued.sum() > 50 and neighbours_all(grid).sum() > 5  # a body with an inside
    faces = np.zeros(SHAPE, dtype=bool)
    faces[[0, -1], :, :] = faces[:, [0, -1], :] = faces[:, :, [0, -1]] = True
    assert not (grid & faces).any(), "the valued blocks must be strictly interior"

    # Without `present`: the shell of the whole box, in which no block has a value.
    box = select_blocks(data, lod="shell")
    assert box.lod == "shell" and len(box.rows) > 0
    assert not valued[box.rows].any()

    # With it: the shell of the valued set, equal to brute force.
    shell = select_blocks(data, lod="shell", present="au")
    expected = np.flatnonzero(valued & ~neighbours_all(grid).reshape(-1))
    np.testing.assert_array_equal(shell.rows, expected)
    assert 0 < len(shell.rows) < valued.sum()
    assert shell.n_selected == model.n_blocks and shell.n_visible == valued.sum()
    assert shell.selected_volume == pytest.approx(model.definition.volume)

    # Together with a filter, and for a category attribute (code -1 = missing).
    where = AttributeConstraint("au > 1.5")
    high = where.mask(model)
    both = select_blocks(data, where, "shell", present="au")
    np.testing.assert_array_equal(
        both.rows, np.flatnonzero(high & ~neighbours_all(high.reshape(SHAPE)).reshape(-1))
    )
    assert both.n_selected == both.n_visible == high.sum()
    model.set("rock", None, model.ijk()[:, 0] < 3)
    coded = ~model.is_missing("rock")
    rock = select_blocks(data, lod="shell", present="rock")
    np.testing.assert_array_equal(
        rock.rows, np.flatnonzero(coded & ~neighbours_all(coded.reshape(SHAPE)).reshape(-1))
    )
    # An int attribute has no missing marker: nothing is left out.
    counted = select_blocks(data, lod="shell", present="au_pass")
    np.testing.assert_array_equal(counted.rows, box.rows)
    with pytest.raises(BlockQueryError, match="unknown attribute gold"):
        select_blocks(data, present="gold")


def test_shell_of_the_valued_blocks_sub_blocked():
    model = ore_model(max_depth=1)
    assert not model.is_dense
    valued = ~np.isnan(model["au"])
    shell = np.zeros(model.n_blocks, dtype=bool)
    shell[select_blocks(BlockData(model), lod="shell", present="au").rows] = True
    np.testing.assert_array_equal(shell, exposed_blocks(model, valued))
    truly = touching(model, valued)
    assert not (shell & ~valued).any()
    assert not (truly & ~shell).any(), "an exposed valued block was dropped"
    assert 0 < shell.sum() < valued.sum()


def test_auto_lod_and_clip_follow_the_valued_blocks(monkeypatch):
    model = ore_model()
    data = BlockData(model)
    valued = ~np.isnan(model["au"])
    n = int(valued.sum())
    # auto: decided by what is visible, not by what is selected.
    monkeypatch.setattr(blocks, "AUTO_SHELL_BLOCKS", n)
    assert select_blocks(data).lod == "shell"
    full = select_blocks(data, present="au")
    assert full.lod == "full"
    np.testing.assert_array_equal(full.rows, np.flatnonzero(valued))
    monkeypatch.setattr(blocks, "AUTO_SHELL_BLOCKS", n - 1)
    auto = select_blocks(data, present="au")
    assert auto.lod == "shell"
    np.testing.assert_array_equal(auto.rows, select_blocks(data, lod="shell", present="au").rows)
    # A section slab: every valued block in it, the interior included, none without a value.
    y = 2060.0
    slab = PlaneConstraint((0.0, y + 10, 0.0), (0, 1, 0), "below") & PlaneConstraint(
        (0.0, y - 10, 0.0), (0, 1, 0), "above"
    )
    cut = select_blocks(data, clip=slab, present="au")
    in_slab = slab.mask(model)
    np.testing.assert_array_equal(cut.rows, np.flatnonzero(valued & in_slab))
    assert cut.clipped and cut.n_visible == n and len(cut.rows) > len(
        np.intersect1d(cut.rows, auto.rows)
    )


def test_payload_and_endpoint_take_present(tmp_path):
    model = ore_model()
    data = BlockData(model)
    valued = ~np.isnan(model["au"])
    header, arrays = unpack_arrays(block_payload(data, "m", ["au"], None, "shell", None, "au"))
    meta = header["meta"]
    assert meta["present"] == "au" and meta["lod"] == "shell"
    assert meta["n_selected"] == meta["n_total"] == model.n_blocks
    assert meta["n_visible"] == valued.sum() and meta["n_blocks"] == len(arrays["row"])
    assert not np.isnan(arrays["value:au"]).any()
    np.testing.assert_array_equal(arrays["value:au"], model["au"][arrays["row"]].astype("f4"))
    plain = unpack_arrays(block_payload(data, "m", ["au"]))[0]["meta"]
    assert plain["present"] is None and plain["n_visible"] == plain["n_selected"]

    client, _ = _project_client(tmp_path, model)
    try:
        url = "/api/v1/scene/objects/model/blocks"
        response = client.get(url, params={"attributes": "au", "lod": "shell", "present": "au"})
        assert response.status_code == 200
        got = unpack_arrays(response.content)
        np.testing.assert_array_equal(got[1]["row"], arrays["row"])
        assert got[0]["meta"]["n_visible"] == meta["n_visible"]
        # A different answer from the same request without it (not the cached one).
        other = unpack_arrays(client.get(url, params={"attributes": "au", "lod": "shell"}).content)
        assert np.isnan(other[1]["value:au"]).all()
        bad = client.get(url, params={"present": "gold"})
        assert bad.status_code == 400 and "unknown attribute gold" in bad.json()["detail"]
    finally:
        app.dependency_overrides.clear()


# ---------------------------------------------------------------------- first view


def _project_client(tmp_path, model: BlockModel) -> tuple[TestClient, Project]:
    save_model(model, tmp_path / "model-src")
    box = model.definition.bounding_box()
    lo, hi = box[0] + 20.0, box[1] - 20.0
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                        for z in (lo[2], hi[2])])  # fmt: skip
    faces = np.array([
        [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5], [0, 4, 5], [0, 5, 1],
        [2, 3, 7], [2, 7, 6], [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
    ])  # fmt: skip
    write_obj(tmp_path / "ore.obj", Mesh(corners, faces))
    project = Project.create(tmp_path / "project", "Display project")
    project.add("solid", tmp_path / "ore.obj", name="Ore solid")
    project.add("model", tmp_path / "model-src", name="Model")
    source = ProjectSceneSource(project.folder)
    app.dependency_overrides[get_scene_source] = lambda: source
    payload_cache.clear()
    return TestClient(app), project


def test_project_model_opens_on_the_estimated_grade_without_air(tmp_path):
    model = ore_model(max_depth=1)
    client, _ = _project_client(tmp_path, model)
    try:
        manifest = client.get("/api/v1/scene").json()
        entry = next(o for o in manifest["objects"] if o["kind"] == "blocks")
        assert entry["display"] == {
            "color_by": "au",
            "category_filter": {"attribute": "rock", "values": ["WASTE", "ORE"]},
        }
        assert client.get("/api/v1/scene/objects/model/blocks/info").json()["display"] == (
            entry["display"]
        )
        # The solids of the project are the model's constraint geometry: the viewer opens
        # those unticked and see-through (web/src/scene/ProjectScene.tsx).
        assert entry["constraint_geometry"]["solids"] == [{"id": "ore-solid", "name": "Ore solid"}]

        # What the viewer then asks for is on screen: the ORE blocks, each with a grade.
        hint = entry["display"]["category_filter"]
        where = " or ".join(f'{hint["attribute"]} == {json.dumps(v)}' for v in hint["values"])
        response = client.get(
            "/api/v1/scene/objects/model/blocks",
            params={
                "attributes": "au",
                "where": json.dumps({"type": "attribute", "expression": where}),
                "present": "au",
            },
        )
        header, arrays = unpack_arrays(response.content)
        ore = model.get("rock", decode=True) == "ORE"
        np.testing.assert_array_equal(arrays["row"], np.flatnonzero(ore))
        air = model.get("rock", decode=True) == "AIR"
        assert header["meta"]["n_selected"] == (~air).sum() > header["meta"]["n_visible"]
        assert header["meta"]["n_visible"] == ore.sum()
    finally:
        app.dependency_overrides.clear()


def test_default_display_rules():
    model = ore_model()
    assert default_display(model)["color_by"] == "au"

    # No diagnostics: the float attribute with the most missing values that is not a
    # density (nor a diagnostic of another attribute, however empty).
    plain = BlockModel(model.definition)
    plain.add_attribute("density", "float")
    plain.add_attribute("cu", "float")
    plain.add_attribute("ag", "float")
    plain.add_attribute("ag_kv", "float")
    plain.add_attribute("empty", "float")
    plain.add_attribute("lith", "category", categories=["air", "OX", "SULPH"])
    k = plain.ijk()[:, 2]
    plain.set("cu", 1.0, k < 6)
    plain.set("ag", 5.0, k < 3)
    plain.set("ag_kv", 0.2, k < 1)
    assert default_display(plain) == {
        "color_by": "ag",
        "category_filter": {"attribute": "lith", "values": ["OX", "SULPH"]},
    }

    # Nothing grade-like: a category attribute; and no air, no filter.
    coded = BlockModel(model.definition)
    coded.add_attribute("density", "float", background=2.7)
    coded.add_attribute("zone", "category", categories=["N", "S"])
    assert default_display(coded) == {"color_by": "zone"}
    only = BlockModel(model.definition)
    only.add_attribute("density", "float", background=2.7)
    assert default_display(only) == {"color_by": "density"}
    assert default_display(BlockModel(model.definition)) == {}
    # A category attribute that is all air is not filtered down to nothing.
    sky = BlockModel(model.definition)
    sky.add_attribute("rock", "category", categories=["AIR"])
    assert "category_filter" not in default_display(sky)
