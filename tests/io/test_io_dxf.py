"""P2.T8 acceptance: ASCII DXF read/write of strings and meshes."""

from pathlib import Path

import numpy as np
import pytest

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import cube, icosphere
from bakumine.core.io._text import BinaryFileError
from bakumine.core.io.dxf import DxfFormatError, format_dxf, parse_dxf, read_dxf, write_dxf
from bakumine.core.io.strings import StringFile, StringSegment

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
UTM = np.array([500_000.123, 5_300_000.456, 900.0])


def canonical_triangles(mesh: Mesh) -> np.ndarray:
    """Triangles as coordinates, independent of vertex numbering and face order
    (winding kept: each triangle is rotated to start at its smallest corner)."""
    tri = mesh.triangles
    keys = np.array([np.lexsort(t.T[::-1])[0] for t in tri])
    rolled = np.array([np.roll(t, -k, axis=0) for t, k in zip(tri, keys, strict=True)])
    flat = rolled.reshape(len(tri), 9)
    return flat[np.lexsort(flat.T[::-1])]


def dxf(*pairs) -> str:
    """DXF text from (code, value) pairs."""
    return "".join(f"{code}\n{value}\n" for code, value in pairs)


def entity(name, layer, *pairs):
    return [(0, name), (8, layer), *pairs]


def xyz(x, y, z, index=0):
    return [(10 + index, x), (20 + index, y), (30 + index, z)]


# -------------------------------------------------------------- round trip


def test_round_trip_strings_by_layer(tmp_path):
    rng = np.random.default_rng(1)
    ring = rng.uniform(-50, 50, size=(6, 3)) + UTM
    ring[-1] = ring[0]
    strings = {
        "ORE OUTLINE": [StringSegment(1, ring), StringSegment(1, rng.normal(size=(4, 3)) + UTM)],
        "12": [StringSegment(12, rng.normal(size=(2, 3)) * 1e3)],
        "topo_crest-A": [StringSegment(1, rng.normal(size=(1, 3)))],
    }
    path = tmp_path / "strings.dxf"
    write_dxf(path, strings=strings)
    doc = read_dxf(path)

    assert list(doc.strings) == list(strings)  # layer names and order preserved
    for layer, segments in strings.items():
        assert len(doc.strings[layer]) == len(segments)
        for a, b in zip(segments, doc.strings[layer], strict=True):
            np.testing.assert_array_equal(b.points, a.points)  # lossless
            assert b.is_closed == a.is_closed
            assert b.number == a.number
    assert doc.strings["ORE OUTLINE"][0].is_closed
    assert doc.n_skipped == 0 and not doc.meshes and not doc.points
    assert doc.skipped_sections == {"TABLES": 1}  # our own layer table


def test_round_trip_string_file_uses_string_numbers_as_layers(tmp_path):
    sf = StringFile.from_arrays(
        [np.arange(9.0).reshape(3, 3), np.arange(12.0).reshape(4, 3) + 0.5, np.ones((2, 3))],
        numbers=[7, 3, 7],
    )
    path = tmp_path / "numbers.dxf"
    write_dxf(path, strings=sf)
    doc = read_dxf(path)
    assert list(doc.strings) == ["7", "3"]
    back = doc.string_file()
    assert [s.number for s in back] == [7, 7, 3]
    np.testing.assert_array_equal(back.segments[0].points, sf.segments[0].points)
    np.testing.assert_array_equal(back.segments[1].points, sf.segments[2].points)
    np.testing.assert_array_equal(back.segments[2].points, sf.segments[1].points)


@pytest.mark.parametrize("polyface", [False, True])
def test_round_trip_meshes_by_layer(tmp_path, polyface):
    meshes = {
        "ORE": icosphere(2, radius=80.0, center=UTM),
        "WASTE DUMP 1": cube(25.0, center=UTM + [300.0, 0.0, 0.0]),
        "TOPO": Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy")),
    }
    path = tmp_path / "meshes.dxf"
    write_dxf(path, meshes=meshes, polyface=polyface)
    doc = read_dxf(path)

    assert list(doc.meshes) == list(meshes)
    for layer, original in meshes.items():
        back = doc.meshes[layer]
        assert back.n_faces == original.n_faces
        assert back.n_vertices == original.n_vertices
        # same vertices as a set, same faces up to index remapping (lossless)
        np.testing.assert_array_equal(
            np.unique(back.vertices, axis=0), np.unique(original.vertices, axis=0)
        )
        np.testing.assert_array_equal(canonical_triangles(back), canonical_triangles(original))
        assert back.is_closed == original.is_closed
        if original.is_closed:
            assert back.signed_volume == pytest.approx(original.signed_volume, rel=1e-12)
            assert back.validate().is_valid
    text = path.read_text()
    assert ("3DFACE" in text) is not polyface
    assert doc.n_skipped == 0


def test_round_trip_ore_solid_and_strings_together(tmp_path):
    ore = Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))
    ring = np.array([[0, 0, 0], [5, 0, 0], [5, 5, 1], [0, 0, 0]], dtype=float) + UTM
    path = tmp_path / "both.dxf"
    write_dxf(
        path,
        meshes={"ORE": ore},
        strings={"ORE": [StringSegment(1, ring)]},
        points={"COLLARS": ring[:2]},
    )
    doc = read_dxf(path)
    assert doc.layers == ["ORE", "COLLARS"]
    back = doc.meshes["ORE"]
    np.testing.assert_array_equal(canonical_triangles(back), canonical_triangles(ore))
    assert back.volume == pytest.approx(ore.volume, rel=1e-12)
    assert back.validate().is_valid
    np.testing.assert_array_equal(doc.strings["ORE"][0].points, ring)
    np.testing.assert_array_equal(doc.points["COLLARS"], ring[:2])
    assert doc.mesh().n_faces == ore.n_faces


def test_fixed_decimals_and_mesh_arguments(tmp_path):
    text = format_dxf(meshes=cube(1.0, center=(0.12345, 0, 0)), decimals=2)
    assert "0.62" in text.splitlines() and "0.62345" not in text
    assert "-0.00" not in text.splitlines()
    doc = parse_dxf(format_dxf(meshes=[cube(1.0), cube(2.0)]))
    assert list(doc.meshes) == ["MESH_1", "MESH_2"]
    assert list(parse_dxf(format_dxf(meshes=cube(1.0))).meshes) == ["0"]
    with pytest.raises(DxfFormatError, match="layer name"):
        format_dxf(meshes={"bad\nname": cube(1.0)})
    big = Mesh(np.zeros((40000, 3)), [[0, 1, 2]])
    with pytest.raises(DxfFormatError, match="polyface"):
        format_dxf(meshes=big, polyface=True)


def test_swap_xy(tmp_path):
    m = cube(2.0, center=(10.0, 20.0, 30.0))
    s = [StringSegment(1, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])]
    path = tmp_path / "swapped.dxf"
    pts = {"P": [[7.0, 8.0, 9.0]]}
    write_dxf(path, meshes={"M": m}, strings={"S": s}, points=pts, swap_xy=True)
    plain = read_dxf(path)  # read without the option: columns are exchanged
    np.testing.assert_array_equal(plain.strings["S"][0].points, [[2.0, 1.0, 3.0], [5.0, 4.0, 6.0]])
    np.testing.assert_array_equal(plain.points["P"], [[8.0, 7.0, 9.0]])
    np.testing.assert_allclose(plain.meshes["M"].area_centroid, [20.0, 10.0, 30.0])
    back = read_dxf(path, swap_xy=True)
    np.testing.assert_array_equal(back.strings["S"][0].points, s[0].points)
    np.testing.assert_array_equal(back.points["P"], [[7.0, 8.0, 9.0]])
    np.testing.assert_array_equal(canonical_triangles(back.meshes["M"]), canonical_triangles(m))


# ------------------------------------------------------------ hand-written

HAND = dxf(
    (999, "hand written test file"),
    (0, "SECTION"), (2, "HEADER"), (9, "$ACADVER"), (1, "AC1009"), (0, "ENDSEC"),
    (0, "SECTION"), (2, "TABLES"),
    (0, "TABLE"), (2, "LAYER"), (70, 1), (0, "LAYER"), (2, "PIT"), (70, 0), (62, 3),
    (0, "ENDTAB"), (0, "ENDSEC"),
    (0, "SECTION"), (2, "BLOCKS"),
    (0, "BLOCK"), (8, "0"), (2, "MARKER"),
    *entity("LINE", "IN_BLOCK", *xyz(0, 0, 0), *xyz(1, 1, 1, 1)),
    (0, "ENDBLK"), (0, "ENDSEC"),
    (0, "SECTION"), (2, "ENTITIES"),
    # a triangle: fourth corner repeats the third
    *entity("3DFACE", "PIT",
            *xyz(0, 0, 0), *xyz(10, 0, 0, 1), *xyz(10, 10, 0, 2), *xyz(10, 10, 0, 3)),
    # a quad: must become two triangles
    *entity("3DFACE", "PIT",
            *xyz(0, 0, 5), *xyz(10, 0, 5, 1), *xyz(10, 10, 5, 2), *xyz(0, 10, 5, 3)),
    # a triangle with only three corners given, on another layer
    *entity("3DFACE", "DUMP", *xyz(0, 0, 0), *xyz(1, 0, 0, 1), *xyz(0, 1, 0, 2)),
    *entity("LINE", "AXIS", *xyz(500000.5, 5300000.25, 100), *xyz(500100.5, 5300000.25, 110, 1)),
    *entity("POINT", "COLLARS", *xyz(1, 2, 3)),
    *entity("POINT", "COLLARS", *xyz("4.0E+00", "5.0e0", "6")),
    # closed LWPOLYLINE at elevation 250
    *entity("LWPOLYLINE", "CONTOUR", (100, "AcDbPolyline"), (90, 3), (70, 1), (38, 250.0),
            (10, 0), (20, 0), (10, 100), (20, 0), (10, 100), (20, 50)),
    # open LWPOLYLINE without elevation, with a bulge
    *entity("LWPOLYLINE", "CONTOUR", (90, 2), (70, 0),
            (10, 1), (20, 2), (42, 0.5), (10, 3), (20, 4)),
    # open 3D polyline
    *entity("POLYLINE", "CREST", (66, 1), *xyz(0, 0, 0), (70, 8)),
    *entity("VERTEX", "CREST", *xyz(1, 1, 1), (70, 32)),
    *entity("VERTEX", "CREST", *xyz(2, 2, 2), (70, 32)),
    *entity("VERTEX", "CREST", *xyz(3, 2, 1), (70, 32)),
    *entity("SEQEND", "CREST"),
    # closed 2D polyline with elevation 42 on the POLYLINE record
    *entity("POLYLINE", "TOE", (66, 1), *xyz(0, 0, 42), (70, 1)),
    *entity("VERTEX", "TOE", (10, 0), (20, 0)),
    *entity("VERTEX", "TOE", (10, 4), (20, 0)),
    *entity("VERTEX", "TOE", (10, 4), (20, 3)),
    *entity("SEQEND", "TOE"),
    # polyface mesh: 4 vertices, one triangle and one quad with a hidden edge
    *entity("POLYLINE", "SHELL", (66, 1), *xyz(0, 0, 0), (70, 64), (71, 5), (72, 2)),
    *entity("VERTEX", "SHELL", *xyz(0, 0, 0), (70, 192)),
    *entity("VERTEX", "SHELL", *xyz(1, 0, 0), (70, 192)),
    *entity("VERTEX", "SHELL", *xyz(1, 1, 0), (70, 192)),
    *entity("VERTEX", "SHELL", *xyz(0, 1, 0), (70, 192)),
    *entity("VERTEX", "SHELL", *xyz(0, 0, 1), (70, 192)),
    *entity("VERTEX", "SHELL", *xyz(0, 0, 0), (70, 128), (71, 1), (72, 2), (73, 5)),
    *entity("VERTEX", "SHELL", *xyz(0, 0, 0), (70, 128), (71, 1), (72, -2), (73, 3), (74, 4)),
    *entity("SEQEND", "SHELL"),
    # 2 x 3 polygon mesh
    *entity("POLYLINE", "GRID", (66, 1), *xyz(0, 0, 0), (70, 16), (71, 2), (72, 3)),
    *[p for i in range(2) for j in range(3)
      for p in entity("VERTEX", "GRID", *xyz(i, j, 0), (70, 64))],
    *entity("SEQEND", "GRID"),
    # things we do not read
    *entity("CIRCLE", "PIT", *xyz(0, 0, 0), (40, 5.0)),
    *entity("TEXT", "PIT", *xyz(0, 0, 0), (40, 1.0), (1, "label")),
    *entity("CIRCLE", "PIT", *xyz(1, 1, 1), (40, 2.0)),
    *entity("INSERT", "PIT", (2, "MARKER"), *xyz(0, 0, 0)),
    (0, "ENDSEC"),
    (0, "SECTION"), (2, "OBJECTS"), (0, "DICTIONARY"), (5, "C"), (0, "ENDSEC"),
    (0, "EOF"),
)  # fmt: skip


def test_hand_written_dxf_parses(tmp_path):
    path = tmp_path / "hand.dxf"
    path.write_text(HAND.replace("\n", "\r\n"))  # Windows line endings
    doc = read_dxf(path)

    # LINE
    (axis,) = doc.strings["AXIS"]
    np.testing.assert_array_equal(
        axis.points, [[500000.5, 5300000.25, 100.0], [500100.5, 5300000.25, 110.0]]
    )
    assert not axis.is_closed
    # POINT, scientific notation
    np.testing.assert_array_equal(doc.points["COLLARS"], [[1, 2, 3], [4, 5, 6]])
    # LWPOLYLINE: closed one gets its first point repeated, elevation in z
    closed, opened = doc.strings["CONTOUR"]
    np.testing.assert_array_equal(
        closed.points, [[0, 0, 250], [100, 0, 250], [100, 50, 250], [0, 0, 250]]
    )
    assert closed.is_closed
    np.testing.assert_array_equal(opened.points, [[1, 2, 0], [3, 4, 0]])
    # 3D POLYLINE
    (crest,) = doc.strings["CREST"]
    np.testing.assert_array_equal(crest.points, [[1, 1, 1], [2, 2, 2], [3, 2, 1]])
    # closed 2D POLYLINE with elevation
    (toe,) = doc.strings["TOE"]
    np.testing.assert_array_equal(toe.points, [[0, 0, 42], [4, 0, 42], [4, 3, 42], [0, 0, 42]])
    assert toe.is_closed
    # polyface mesh: triangle + quad -> 3 triangles, hidden-edge sign ignored
    shell = doc.meshes["SHELL"]
    assert shell.n_faces == 3 and shell.n_vertices == 5
    np.testing.assert_array_equal(
        canonical_triangles(shell),
        canonical_triangles(
            Mesh(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1]],
                [[0, 1, 4], [0, 1, 2], [0, 2, 3]],
            )
        ),
    )
    # polygon mesh 2 x 3 -> 2 cells -> 4 triangles covering area 2
    grid = doc.meshes["GRID"]
    assert grid.n_faces == 4 and grid.n_vertices == 6
    assert grid.area == pytest.approx(2.0)
    assert doc.meshes["DUMP"].n_faces == 1
    assert doc.layers == [
        *["AXIS", "CONTOUR", "CREST", "TOE"],  # strings
        *["PIT", "DUMP", "SHELL", "GRID"],  # meshes
        "COLLARS",  # points
    ]


def test_3dface_quad_is_split_into_two_triangles():
    doc = parse_dxf(HAND)
    pit = doc.meshes["PIT"]
    # one triangle (z = 0) plus one quad (z = 5) -> 3 triangles
    assert pit.n_faces == 3
    quad = pit.submesh(pit.triangles[:, :, 2].min(axis=1) == 5.0)
    assert quad.n_faces == 2
    assert quad.area == pytest.approx(100.0)  # the two halves tile the 10 x 10 quad
    np.testing.assert_array_equal(
        canonical_triangles(quad),
        canonical_triangles(
            Mesh([[0, 0, 5], [10, 0, 5], [10, 10, 5], [0, 10, 5]], [[0, 1, 2], [0, 2, 3]])
        ),
    )
    # the shared diagonal is welded: 4 quad corners + 3 triangle corners
    assert pit.n_vertices == 7
    np.testing.assert_allclose(quad.face_normals, [[0, 0, 1], [0, 0, 1]])  # winding kept


def test_unknown_entities_and_sections_are_skipped_and_counted():
    doc = parse_dxf(HAND)
    assert doc.skipped_entities == {"CIRCLE": 2, "TEXT": 1, "INSERT": 1}
    assert doc.n_skipped == 4
    assert doc.skipped_sections == {"HEADER": 1, "TABLES": 1, "BLOCKS": 1, "OBJECTS": 1}
    assert "IN_BLOCK" not in doc.strings  # block definitions are not model geometry
    assert doc.notes["bulge ignored (arc read as chord)"] == 1


def test_entities_only_file_and_default_layer():
    text = dxf((0, "3DFACE"), *xyz(0, 0, 0), *xyz(1, 0, 0, 1), *xyz(0, 1, 0, 2), (0, "EOF"))
    doc = parse_dxf(text)
    assert list(doc.meshes) == ["0"] and doc.meshes["0"].n_faces == 1


def test_bad_input(tmp_path):
    with pytest.raises(DxfFormatError, match="group code"):
        parse_dxf("this is\nnot a dxf\nfile at all\nreally\n")
    with pytest.raises(DxfFormatError):
        parse_dxf("")
    path = tmp_path / "binary.dxf"
    path.write_bytes(b"AutoCAD Binary DXF\r\n\x1a\x00" + bytes(range(256)))
    with pytest.raises(BinaryFileError, match="binary DXF is not supported"):
        read_dxf(path)
