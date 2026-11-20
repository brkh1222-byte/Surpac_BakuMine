"""Surfaces from points: sources, extension rule, duplicates; mesh import."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from scipy.spatial import ConvexHull, Delaunay

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.mesh_import import import_mesh
from bakumine.core.geometry.primitives import box
from bakumine.core.geometry.surface_points import (
    extension_points,
    read_points_table,
    string_lines,
    string_points,
    surface_from_points,
)
from bakumine.core.io.files import write_mesh, write_strings
from bakumine.core.io.strings import StringFile, StringSegment


def scattered(n: int = 60, seed: int = 3) -> np.ndarray:
    rng = np.random.default_rng(seed)
    xy = rng.uniform([1000.0, 5000.0], [1400.0, 5300.0], size=(n, 2))
    z = 200.0 + 0.05 * (xy[:, 0] - 1000.0) + 10.0 * np.sin(xy[:, 1] / 40.0)
    return np.column_stack([xy, z])


def test_surface_honours_the_points_and_matches_scipy():
    points = scattered()
    result = surface_from_points(points)
    dtm, info = result.dtm, result.info
    np.testing.assert_allclose(dtm.z_at(points[:, 0], points[:, 1]), points[:, 2], atol=1e-9)
    # independent triangulation: same number of triangles, same area (the convex hull)
    reference = Delaunay(points[:, :2])
    assert info["n_faces"] == len(reference.simplices)
    assert info["n_vertices"] == len(points)
    assert info["plan_area"] == pytest.approx(ConvexHull(points[:, :2]).volume, rel=1e-12)
    assert info["n_extension_points"] == 0 and info["rectangle"] is None
    assert info["n_duplicates"] == 0 and info["n_conflicts"] == 0
    # several sources are joined
    joined = surface_from_points([points[:20], points[20:]])
    assert joined.info["n_faces"] == info["n_faces"]
    with pytest.raises(ValueError, match="at least 3"):
        surface_from_points(points[:2])


def test_extension_rule_nearest_and_constant():
    points = scattered()
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    result = surface_from_points(points, padding=150.0)
    rectangle = result.info["rectangle"]
    assert rectangle == pytest.approx([lo[0] - 150, lo[1] - 150, hi[0] + 150, hi[1] + 150])
    width, height = rectangle[2] - rectangle[0], rectangle[3] - rectangle[1]
    assert result.info["plan_area"] == pytest.approx(width * height, rel=1e-12)
    assert result.info["extend_z"] == "nearest"
    # the data are untouched
    np.testing.assert_allclose(result.dtm.z_at(points[:, 0], points[:, 1]), points[:, 2], atol=1e-9)
    # every added point has the elevation of the nearest data point in plan (brute force)
    extra = extension_points(points, rectangle)
    assert len(extra) == result.info["n_extension_points"]
    distance = np.linalg.norm(extra[:, None, :2] - points[None, :, :2], axis=2)
    np.testing.assert_array_equal(extra[:, 2], points[distance.argmin(axis=1), 2])
    np.testing.assert_allclose(result.dtm.z_at(extra[:, 0], extra[:, 1]), extra[:, 2], atol=1e-9)
    # the four corners are there and all points are on the rectangle
    for corner in [(0, 1), (2, 1), (0, 3), (2, 3)]:
        target = [rectangle[corner[0]], rectangle[corner[1]]]
        assert (np.abs(extra[:, :2] - target).max(axis=1) < 1e-9).any()
    on_x = np.isclose(extra[:, 0], rectangle[0]) | np.isclose(extra[:, 0], rectangle[2])
    on_y = np.isclose(extra[:, 1], rectangle[1]) | np.isclose(extra[:, 1], rectangle[3])
    assert (on_x | on_y).all()
    # equal steps along a side, no longer than the step asked for
    stepped = extension_points(points, rectangle, step=100.0)
    bottom = np.sort(stepped[np.isclose(stepped[:, 1], rectangle[1]), 0])
    assert np.diff(bottom).max() <= 100.0 + 1e-9
    assert np.allclose(np.diff(bottom), np.diff(bottom)[0])

    flat = surface_from_points(points, padding=150.0, extend_z=180.0)
    assert flat.info["extend_z"] == 180.0
    assert flat.dtm.z_at(rectangle[0], rectangle[1]) == pytest.approx(180.0)


def test_extension_rectangle_inside_the_data_changes_nothing_there():
    points = scattered()
    centre = points[:, :2].mean(axis=0)
    inner = (centre[0] - 20, centre[1] - 20, centre[0] + 20, centre[1] + 20)
    assert len(extension_points(points, inner)) == 0  # all inside the hull: left out
    plain = surface_from_points(points)
    inside = surface_from_points(points, rectangle=inner)
    assert inside.info["n_extension_points"] == 0
    np.testing.assert_array_equal(inside.dtm.mesh.vertices, plain.dtm.mesh.vertices)
    # a rectangle can be grown by a padding (block model extent plus a margin)
    lo, hi = points[:, :2].min(axis=0), points[:, :2].max(axis=0)
    grown = surface_from_points(points, rectangle=(*lo, *hi), padding=50.0)
    assert grown.info["rectangle"] == pytest.approx(
        [lo[0] - 50, lo[1] - 50, hi[0] + 50, hi[1] + 50]
    )
    for bad in ({"rectangle": (0, 0, 0, 1)}, {"rectangle": (0, 0, 1)}, {"padding": -1.0},
                {"padding": 10.0, "extend_z": "lowest"}):  # fmt: skip
        with pytest.raises(ValueError):
            surface_from_points(points, **bad)


def test_duplicate_handling():
    points = scattered(30)
    doubled = np.concatenate([points, points[:5], points[5:8] + [0.0, 0.0, 2.0]])
    result = surface_from_points(doubled)
    assert result.info["n_points"] == 38
    assert result.info["n_duplicates"] == 8  # 5 exact copies + 3 conflicting
    assert result.info["n_conflicts"] == 3
    assert result.info["n_vertices"] == 30
    z = result.dtm.z_at(points[5:8, 0], points[5:8, 1])
    np.testing.assert_allclose(z, points[5:8, 2] + 1.0, atol=1e-9)  # mean
    for rule, offset in (("min", 0.0), ("max", 2.0), ("first", 0.0), ("last", 2.0)):
        z = surface_from_points(doubled, duplicates=rule).dtm.z_at(points[5:8, 0], points[5:8, 1])
        np.testing.assert_allclose(z, points[5:8, 2] + offset, atol=1e-9)
    with pytest.raises(ValueError):
        surface_from_points(doubled, duplicates="error")
    assert surface_from_points(np.concatenate([points, points]), duplicates="error").info[
        "n_duplicates"
    ] == len(points)


def test_points_table_csv_and_xlsx(tmp_path):
    points = scattered(12)
    frame = pd.DataFrame(
        {"Northing": points[:, 1], "Easting": points[:, 0], "RL": points[:, 2], "note": "x"}
    )
    frame.loc[3, "RL"] = np.nan
    frame.to_csv(tmp_path / "survey.csv", index=False)
    frame.to_excel(tmp_path / "survey.xlsx", index=False)
    for name in ("survey.csv", "survey.xlsx"):
        got, info = read_points_table(tmp_path / name)
        assert info == {"columns": ["Easting", "Northing", "RL"], "rows": 12, "skipped": 1}
        np.testing.assert_allclose(got, np.delete(points, 3, axis=0), rtol=0, atol=1e-9)
    # explicit columns (here: a Surpac-style Y, X, Z file read with the right names)
    got, _ = read_points_table(tmp_path / "survey.csv", ["Easting", "Northing", "RL"])
    np.testing.assert_allclose(got[:, 0], np.delete(points[:, 0], 3), atol=1e-9)
    pd.DataFrame({"a": [1], "b": [2], "c": [3]}).to_csv(tmp_path / "bare.csv", index=False)
    with pytest.raises(ValueError, match="no x column"):
        read_points_table(tmp_path / "bare.csv")
    got, _ = read_points_table(tmp_path / "bare.csv", ["a", "b", "c"])
    assert got.tolist() == [[1.0, 2.0, 3.0]]
    with pytest.raises(ValueError, match="no column"):
        read_points_table(tmp_path / "bare.csv", ["a", "b", "zz"])


def test_strings_as_points_and_breaklines(tmp_path):
    points = scattered(40)
    # a ridge line crossing the data, as a string
    ridge = np.array([[1050.0, 5020.0, 260.0], [1200.0, 5150.0, 262.0], [1350.0, 5280.0, 261.0]])
    contour = np.array([[1010.0, 5010.0, 150.0], [1390.0, 5010.0, 150.0]])
    strings = StringFile([StringSegment(1, ridge), StringSegment(2, contour)])
    for name in ("lines.str", "lines.dxf"):
        write_strings(tmp_path / name, strings)
        from bakumine.core.io.files import read_strings

        back = read_strings(tmp_path / name)
        np.testing.assert_allclose(string_points(back), np.concatenate([ridge, contour]), atol=1e-3)
        lines = string_lines(back)
        assert [len(line) for line in lines] == [3, 2]
    result = surface_from_points(points, string_lines(strings))
    assert result.info["n_breaklines"] == 2 and result.info["n_breakline_points"] == 5
    mesh = result.dtm.mesh
    # every breakline segment is an edge of the triangulation
    edges = {
        tuple(sorted((int(a), int(b))))
        for face in mesh.faces
        for a, b in zip(face, np.roll(face, -1), strict=True)
    }

    def vertex(p) -> int:
        d = np.linalg.norm(mesh.vertices - p, axis=1)
        assert d.min() < 1e-9
        return int(d.argmin())

    for line in (ridge, contour):
        for a, b in zip(line[:-1], line[1:], strict=True):
            assert tuple(sorted((vertex(a), vertex(b)))) in edges
    # along the ridge the surface follows the string exactly
    mid = 0.5 * (ridge[0] + ridge[1])
    assert result.dtm.z_at(mid[0], mid[1]) == pytest.approx(mid[2])


def test_import_mesh_kinds_and_formats(tmp_path):
    solid = box((0.0, 0.0, 0.0), (10.0, 20.0, 30.0))
    surface = surface_from_points(scattered(25)).dtm.mesh
    for ext in (".obj", ".ply", ".dxf", ".dtm"):
        write_mesh(tmp_path / f"solid{ext}", solid)
        mesh, info = import_mesh(tmp_path / f"solid{ext}")
        assert info["kind"] == "solid" and info["closed"] and info["valid"], (ext, info)
        assert info["volume"] == pytest.approx(6000.0, rel=1e-6)
        assert mesh.n_vertices == 8 and mesh.n_faces == 12  # welded (DXF repeats corners)
        write_mesh(tmp_path / f"surface{ext}", surface)
        mesh, info = import_mesh(tmp_path / f"surface{ext}")
        assert info["kind"] == "dtm" and not info["closed"]
        assert mesh.n_faces == surface.n_faces and mesh.n_vertices == surface.n_vertices
        assert mesh.face_normals[:, 2].min() > 0.0
    # a clockwise surface is turned normals-up; a closed box is not a DTM
    write_mesh(tmp_path / "flipped.obj", Mesh(surface.vertices, surface.faces[:, ::-1]))
    mesh, _ = import_mesh(tmp_path / "flipped.obj", "dtm")
    assert mesh.face_normals[:, 2].min() > 0.0
    with pytest.raises(ValueError, match="not a surface"):
        import_mesh(tmp_path / "solid.obj", "dtm")
    # an open surface imported as a solid is reported invalid, not refused
    _, info = import_mesh(tmp_path / "surface.obj", "solid")
    assert info["kind"] == "solid" and not info["valid"] and info["volume"] is None
    with pytest.raises(FileNotFoundError):
        import_mesh(tmp_path / "missing.obj")
    with pytest.raises(ValueError):
        import_mesh(tmp_path / "solid.obj", "pit")
