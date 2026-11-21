"""P8.T12: STL reader (ASCII and binary) and its place in the extension dispatcher.

Checked against a hand-made tetrahedron (volume 1/6 by construction), against trimesh as an
independent reader / writer (dev dependency only), and on the public Babbitt wireframe.
"""

import struct
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bakumine.core.geometry.primitives import cube, icosphere
from bakumine.core.io.files import read_mesh, read_meshes, write_mesh
from bakumine.core.io.stl import StlFormatError, parse_stl, read_stl

MPZ = Path(__file__).resolve().parents[2] / "data" / "public" / "babbitt" / "raw" / "Mpz.stl"

# unit right tetrahedron, outward winding: volume 1/6, area 1.5 + sqrt(3)/2
CORNERS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
TRIANGLES = [(0, 2, 1), (0, 1, 3), (0, 3, 2), (1, 2, 3)]


def ascii_stl(offset=(0.0, 0.0, 0.0), name="tetra") -> str:
    lines = [f"solid {name}"]
    for tri in TRIANGLES:
        lines += ["  facet normal 0 0 0", "    outer loop"]
        lines += ["      vertex {:.6e} {:.6e} {:.6e}".format(*(CORNERS[i] + offset)) for i in tri]
        lines += ["    endloop", "  endfacet"]
    lines.append(f"endsolid {name}")
    return "\n".join(lines) + "\n"


def binary_stl(header: bytes = b"hand made") -> bytes:
    data = header.ljust(80, b"\0") + struct.pack("<I", len(TRIANGLES))
    for tri in TRIANGLES:
        data += struct.pack("<3f", 0.0, 0.0, 0.0)
        for i in tri:
            data += struct.pack("<3f", *CORNERS[i])
        data += struct.pack("<H", 0)
    return data


def assert_tetrahedron(mesh, scale: float = 1.0) -> None:
    assert (mesh.n_vertices, mesh.n_faces) == (4, 4)
    assert mesh.vertices.dtype == np.float64 and mesh.faces.dtype == np.int64
    # vertices in order of first appearance in the file: corners 0, 2, 1, 3
    np.testing.assert_array_equal(mesh.vertices, CORNERS[[0, 2, 1, 3]] * scale)
    np.testing.assert_array_equal(mesh.faces, [[0, 1, 2], [0, 2, 3], [0, 3, 1], [2, 1, 3]])
    assert mesh.is_closed and mesh.is_winding_consistent
    assert mesh.signed_volume == pytest.approx(scale**3 / 6.0, rel=1e-12)  # outward
    assert mesh.area == pytest.approx(scale**2 * (1.5 + np.sqrt(3.0) / 2.0), rel=1e-12)


def test_ascii_tetrahedron(tmp_path):
    path = tmp_path / "t.stl"
    path.write_text(ascii_stl())
    assert_tetrahedron(read_stl(path))
    assert_tetrahedron(parse_stl(ascii_stl().upper().replace("\n", "\r\n").encode()))


def test_binary_tetrahedron(tmp_path):
    path = tmp_path / "t.stl"
    path.write_bytes(binary_stl())
    assert_tetrahedron(read_stl(path))
    # a binary file may start with "solid": the length decides, not the first bytes
    assert_tetrahedron(parse_stl(binary_stl(b"solid made by some exporter")))


def test_scale_and_swap_xy():
    assert_tetrahedron(parse_stl(binary_stl(), scale=0.3048), scale=0.3048)
    swapped = parse_stl(ascii_stl().encode(), swap_xy=True)
    np.testing.assert_array_equal(swapped.vertices, CORNERS[[0, 2, 1, 3]][:, [1, 0, 2]])
    with pytest.raises(ValueError, match="scale"):
        parse_stl(binary_stl(), scale=0.0)


def test_several_solids_are_joined():
    text = ascii_stl(name="a") + ascii_stl(offset=(5.0, 0.0, 0.0), name="b")
    mesh = parse_stl(text.encode())
    assert (mesh.n_vertices, mesh.n_faces) == (8, 8)
    assert mesh.volume == pytest.approx(2.0 / 6.0, rel=1e-12)


def test_empty_solid():
    mesh = parse_stl(b"solid empty\nendsolid empty\n")
    assert (mesh.n_vertices, mesh.n_faces) == (0, 0)


@pytest.mark.parametrize(
    "data, message",
    [
        (b"hello world\n", "not an STL"),
        (binary_stl()[:-7], "not an STL"),  # truncated binary: neither length nor text fits
        (ascii_stl().replace("endloop", "", 1).encode(), "inside a loop"),
        (ascii_stl().replace("vertex 1.0", "vertex abc", 1).encode(), "bad vertex"),
        (ascii_stl().rsplit("endloop", 1)[0].encode(), "truncated"),
        (ascii_stl().replace("facet normal", "polygon normal", 1).encode(), "unknown STL record"),
        (ascii_stl().replace("vertex 0.000000e+00 0", "vertex nan 0", 1).encode(), "NaN"),
    ],
)
def test_bad_files_are_refused(data, message):
    with pytest.raises(StlFormatError, match=message):
        parse_stl(data)


@pytest.mark.parametrize("file_type", ["stl", "stl_ascii"])
def test_reads_what_trimesh_writes(tmp_path, file_type):
    ours = icosphere(2, radius=12.5, center=(100.0, 200.0, 50.0))
    path = tmp_path / "sphere.stl"
    trimesh.Trimesh(ours.vertices, ours.faces, process=False).export(path, file_type=file_type)
    back = read_stl(path)
    assert (back.n_vertices, back.n_faces) == (ours.n_vertices, ours.n_faces)
    assert back.is_closed and back.is_winding_consistent
    # binary STL is float32; trimesh's ASCII writer keeps about as many digits
    assert back.signed_volume == pytest.approx(ours.signed_volume, rel=1e-5)
    oracle = trimesh.load(path)
    assert back.signed_volume == pytest.approx(oracle.volume, rel=1e-9)
    assert back.area == pytest.approx(oracle.area, rel=1e-9)


def test_dispatcher(tmp_path):
    path = tmp_path / "t.STL"
    path.write_bytes(binary_stl())
    assert_tetrahedron(read_mesh(path))
    assert list(read_meshes(path)) == ["t"]
    assert_tetrahedron(read_mesh(path, scale=2.0), scale=2.0)
    with pytest.raises(ValueError, match="supported"):  # read only: single precision
        write_mesh(tmp_path / "out.stl", cube(1.0))


def test_babbitt_wireframe_against_trimesh():
    """The real file: binary, 63 749 triangles in feet, float32 coordinates of ~2.3e6."""
    assert MPZ.stat().st_size == 84 + 50 * 63749
    mesh = read_stl(MPZ)
    oracle = trimesh.load(MPZ)  # merges identical vertices too
    assert (mesh.n_vertices, mesh.n_faces) == (len(oracle.vertices), len(oracle.faces))
    assert (mesh.n_vertices, mesh.n_faces) == (31873, 63749)
    np.testing.assert_array_equal(mesh.bounds, oracle.bounds)
    assert mesh.area == pytest.approx(oracle.area, rel=1e-9)
    # as published the surface is not closed (one triangle missing, three double-sided
    # fins), so the volume depends slightly on the reference point: 1e-5, not 1e-9
    assert not mesh.is_closed and not oracle.is_watertight
    assert len(mesh.edge_map.boundary_edges) == 3
    assert mesh.signed_volume == pytest.approx(oracle.volume, rel=1e-5)
    metres = read_stl(MPZ, scale=0.3048)
    np.testing.assert_allclose(metres.vertices, mesh.vertices * 0.3048, rtol=1e-15)
