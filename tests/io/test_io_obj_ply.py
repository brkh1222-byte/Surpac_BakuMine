"""P2.T8 acceptance: OBJ and ASCII PLY read/write, and the extension dispatcher."""

import warnings
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import cube, icosphere
from bakumine.core.io._text import BinaryFileError
from bakumine.core.io.files import read_mesh, read_meshes, read_strings, write_mesh, write_strings
from bakumine.core.io.obj import ObjFormatError, format_obj, parse_obj, read_obj, write_obj
from bakumine.core.io.ply import PlyFormatError, format_ply, parse_ply, read_ply, write_ply
from bakumine.core.io.strings import StringFile, StringSegment

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
UTM = np.array([500_000.123, 5_300_000.456, 900.0])


def load_synthetic(name: str) -> Mesh:
    return Mesh(np.load(DATA / f"{name}_vertices.npy"), np.load(DATA / f"{name}_faces.npy"))


def assert_same_mesh(a: Mesh, b: Mesh, atol: float = 0.0) -> None:
    np.testing.assert_array_equal(a.faces, b.faces)
    if atol == 0.0:
        np.testing.assert_array_equal(a.vertices, b.vertices)
    else:
        np.testing.assert_allclose(a.vertices, b.vertices, rtol=0, atol=atol)


# --------------------------------------------------------------------- OBJ


@pytest.mark.parametrize("name", ["ore_solid", "topo"])
def test_existing_obj_files_equal_npy_arrays(name):
    mesh = read_obj(DATA / f"{name}.obj")
    expected = load_synthetic(name)
    np.testing.assert_array_equal(mesh.faces, expected.faces)
    assert mesh.vertices.shape == expected.vertices.shape
    # the .obj was written with 6 decimals
    np.testing.assert_allclose(mesh.vertices, expected.vertices, rtol=0, atol=5.0e-7 + 1e-9)
    assert mesh.vertices.dtype == np.float64 and mesh.faces.dtype == np.int64
    if name == "ore_solid":
        assert mesh.is_closed
        assert mesh.signed_volume == pytest.approx(expected.signed_volume, rel=1e-8)


@pytest.mark.parametrize("fmt", ["obj", "ply"])
@pytest.mark.parametrize("name", ["cube", "icosphere", "ore_solid", "topo"])
def test_round_trip(tmp_path, fmt, name):
    if name == "cube":
        original = cube(40.0, center=UTM)
    elif name == "icosphere":
        original = icosphere(3, radius=125.5, center=UTM)
    else:
        original = load_synthetic(name)
    write, read = (write_obj, read_obj) if fmt == "obj" else (write_ply, read_ply)

    exact = tmp_path / f"exact.{fmt}"
    write(exact, original, decimals=None)
    assert_same_mesh(read(exact), original)  # bit-for-bit

    fixed = tmp_path / f"fixed.{fmt}"
    write(fixed, original)  # default: 6 decimals
    back = read(fixed)
    assert_same_mesh(back, original, atol=5.0e-7 + 1e-9)
    if original.is_closed:
        assert back.volume == pytest.approx(original.volume, rel=1e-8)
        assert back.validate().is_valid

    # independent reader agrees with what we wrote
    tm = trimesh.load(exact, process=False, force="mesh")
    np.testing.assert_array_equal(np.asarray(tm.faces), original.faces)
    np.testing.assert_allclose(np.asarray(tm.vertices), original.vertices, rtol=0, atol=1e-6)


def test_obj_ngon_fan_negative_indices_and_ignored_records():
    text = """\
# comment line
mtllib scene.mtl
o pentagon
v 0 0 0
v 2 0 0   # trailing comment
v 3 1 0 1.0
v 1 3 0
v -1 1 0
vt 0.5 0.5
vn 0 0 1
usemtl rock
s off
g group1
f 1/1/1 2/1/1 3/1/1 4/1/1 5/1/1
v 0 0 5
f -1 1//1 2//1
f -1 -2 \\
  -3
l 1 2
"""
    mesh = parse_obj(text)
    assert mesh.n_vertices == 6
    np.testing.assert_array_equal(
        mesh.faces,
        [
            [0, 1, 2],  # fan of the pentagon from its first corner
            [0, 2, 3],
            [0, 3, 4],
            [5, 0, 1],  # -1 is the vertex defined just before the face
            [5, 4, 3],  # continuation line; -2 and -3 count back from 6 vertices
        ],
    )
    np.testing.assert_array_equal(mesh.vertices[2], [3.0, 1.0, 0.0])
    # the fan covers the pentagon: area by the shoelace formula
    x, y = mesh.vertices[:5, 0], mesh.vertices[:5, 1]
    shoelace = 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    assert mesh.submesh([0, 1, 2]).area == pytest.approx(shoelace)


def test_obj_quads_from_another_writer(tmp_path):
    box = trimesh.creation.box(extents=(2, 3, 4))
    quads = "".join(f"v {x} {y} {z}\n" for x, y, z in box.vertices.tolist())
    quads += "f 1 2 4 3\nf 5 7 8 6\n"
    mesh = parse_obj(quads)
    assert mesh.n_faces == 4
    assert mesh.area == pytest.approx(2 * 3 * 4)  # two faces of 3 x 4


def test_obj_errors():
    with pytest.raises(ObjFormatError, match="line 2"):
        parse_obj("v 0 0 0\nf 1 2 3\n")  # vertices 2 and 3 do not exist
    with pytest.raises(ObjFormatError, match="index 0"):
        parse_obj("v 0 0 0\nv 1 0 0\nv 0 1 0\nf 0 1 2\n")
    with pytest.raises(ObjFormatError):
        parse_obj("v 0 zero 0\n")
    with pytest.raises(ObjFormatError):
        parse_obj("v 0 0\n")


def test_obj_format_and_swap(tmp_path):
    mesh = Mesh([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], [[0, 1, 2]])
    assert format_obj(mesh, decimals=2, comment="two\nlines") == (
        "# two\n# lines\nv 1.00 2.00 3.00\nv 4.00 5.00 6.00\nv 7.00 8.00 10.00\nf 1 2 3\n"
    )
    assert format_obj(mesh, decimals=1, swap_xy=True).startswith("v 2.0 1.0 3.0\n")
    assert_same_mesh(parse_obj(format_obj(mesh, swap_xy=True), swap_xy=True), mesh)
    np.testing.assert_array_equal(
        parse_obj(format_obj(mesh, swap_xy=True)).vertices[0], [2.0, 1.0, 3.0]
    )
    assert parse_obj("").n_faces == 0


# --------------------------------------------------------------------- PLY


def test_ply_hand_written_with_extra_properties_and_quad():
    text = """\
ply
format ascii 1.0
comment made by hand
element vertex 5
property float nx
property float x
property float y
property float z
property uchar red
element edge 1
property int vertex1
property int vertex2
element face 2
property uchar flag
property list uchar int vertex_index
end_header
0 0 0 0 255
0 1e0 0 0 255
0 1 1 0 255
0 0 1 0 255
0 0.5 0.5 2.5 0
0 1
7 4 0 1 2 3
7 3 0 1 4
"""
    mesh = parse_ply(text)
    np.testing.assert_array_equal(
        mesh.vertices, [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0.5, 0.5, 2.5]]
    )
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3], [0, 1, 4]])


def test_ply_format_and_swap():
    mesh = Mesh([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]], [[0, 1, 2]])
    text = format_ply(mesh, decimals=1, comment="c")
    assert text == (
        "ply\nformat ascii 1.0\ncomment c\nelement vertex 3\nproperty double x\n"
        "property double y\nproperty double z\nelement face 1\n"
        "property list uchar int vertex_indices\nend_header\n"
        "1.0 2.0 3.0\n4.0 5.0 6.0\n7.0 8.0 10.0\n3 0 1 2\n"
    )
    assert_same_mesh(parse_ply(format_ply(mesh, swap_xy=True), swap_xy=True), mesh)
    np.testing.assert_array_equal(
        parse_ply(format_ply(mesh, swap_xy=True)).vertices[0], [2.0, 1.0, 3.0]
    )


def test_ply_errors(tmp_path):
    with pytest.raises(PlyFormatError, match="not a PLY"):
        parse_ply("v 0 0 0\n")
    with pytest.raises(PlyFormatError, match="end_header"):
        parse_ply("ply\nformat ascii 1.0\nelement vertex 1\n")
    with pytest.raises(PlyFormatError, match="rows"):
        parse_ply("ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\n"
                  "property float y\nproperty float z\nend_header\n0 0 0\n")  # fmt: skip
    with pytest.raises(PlyFormatError):
        parse_ply("ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\n"
                  "property float y\nend_header\n0 0\n")  # fmt: skip
    with pytest.raises(BinaryFileError, match="binary PLY"):
        parse_ply("ply\nformat binary_little_endian 1.0\nend_header\n")
    path = tmp_path / "binary.ply"
    trimesh.creation.box().export(path, encoding="binary")
    with pytest.raises(BinaryFileError, match="binary PLY"):
        read_ply(path)


# -------------------------------------------------------------- dispatcher


@pytest.mark.parametrize("ext", [".obj", ".ply", ".dxf", ".dtm"])
def test_dispatch_mesh_round_trip(tmp_path, ext):
    original = icosphere(2, radius=50.0, center=UTM)
    path = tmp_path / f"solid{ext.upper()}"  # extension matching ignores case
    options = {"validated": True} if ext == ".dtm" else {}
    write_mesh(path, original, **options)
    back = read_mesh(path)
    assert back.n_faces == original.n_faces and back.n_vertices == original.n_vertices
    assert back.volume == pytest.approx(original.volume, rel=1e-5)  # .dtm: mm rounding
    assert back.is_closed and back.signed_volume > 0
    assert len(read_meshes(path)) == 1


@pytest.mark.parametrize("ext", [".obj", ".ply", ".dxf", ".dtm"])
def test_dispatch_several_meshes(tmp_path, ext):
    meshes = {"A": cube(2.0), "B": cube(4.0, center=(10.0, 0.0, 0.0))}
    path = tmp_path / f"two{ext}"
    options = {"validated": True} if ext == ".dtm" else {}
    write_mesh(path, meshes, **options)
    named = read_meshes(path)
    expected_names = {".dxf": ["A", "B"], ".dtm": ["1.1", "2.1"]}.get(ext, ["two"])
    assert list(named) == expected_names
    joined = read_mesh(path)
    assert joined.n_faces == 24
    assert joined.volume == pytest.approx(8.0 + 64.0)


@pytest.mark.parametrize("ext", [".str", ".dxf"])
def test_dispatch_strings_round_trip(tmp_path, ext):
    sf = StringFile(
        segments=[
            StringSegment(2, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.5], [1.0, 2.0, 3.0]]),
            StringSegment(5, [[10.0, 20.0, 30.0], [40.0, 50.0, 60.125]]),
        ]
    )
    path = tmp_path / f"lines{ext}"
    write_strings(path, sf)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_strings(path)
    assert [s.number for s in back] == [2, 5]
    assert [s.is_closed for s in back] == [True, False]
    for a, b in zip(sf.segments, back.segments, strict=True):
        np.testing.assert_array_equal(a.points, b.points)


def test_dispatch_unknown_extension(tmp_path):
    with pytest.raises(ValueError, match=r"'\.3ds'.*supported"):
        read_mesh(tmp_path / "a.3ds")
    with pytest.raises(ValueError, match="supported"):
        write_mesh(tmp_path / "a.mdl", cube(1.0))
    with pytest.raises(ValueError, match="supported"):
        read_strings(tmp_path / "a.obj")
    with pytest.raises(ValueError, match="supported"):
        write_strings(tmp_path / "a.txt", StringFile())
