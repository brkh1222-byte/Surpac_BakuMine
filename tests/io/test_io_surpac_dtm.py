"""P2.T8 acceptance: Surpac .dtm + .str text read/write.

The layout follows the published "Appendix D - File Formats" (Surpac, 1996; see
the module docstring of ``bakumine.core.io.surpac_dtm``): vertex numbers count
the null records of the ``.str``, triangles are clockwise seen from outside, no
record of zeros ends a trisolation. The pairs below are hand-written to those
rules (the appendix's own example is not copied here; it was read once with this
reader: closed, 8 vertices, volume -> +100^3, neighbours as the edge map gives).
Not checked against a file exported by a current Surpac.
"""

import warnings
from pathlib import Path

import numpy as np
import pytest
import trimesh

from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import cube, icosphere, tetrahedron
from bakumine.core.io._text import BinaryFileError, FormatWarning
from bakumine.core.io.surpac_dtm import (
    NEIGHBOUR_EDGES,
    SurpacFormatError,
    SurpacTrisolation,
    describe_surpac_file,
    mesh_neighbours,
    read_surpac_dtm,
    write_surpac_dtm,
)
from bakumine.core.io.surpac_str import read_surpac_str

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
UTM = np.array([500_000.0, 5_300_000.0, 900.0])


def load_synthetic(name: str) -> Mesh:
    return Mesh(np.load(DATA / f"{name}_vertices.npy"), np.load(DATA / f"{name}_faces.npy"))


def canonical_triangles(mesh: Mesh) -> np.ndarray:
    """Triangles as coordinates, independent of vertex numbering and face order.

    Each triangle is rotated (winding kept) so that its smallest corner comes
    first, then the triangles are sorted.
    """
    tri = mesh.triangles
    keys = np.array([np.lexsort(t.T[::-1])[0] for t in tri])  # first corner in x, y, z order
    rolled = np.array([np.roll(t, -k, axis=0) for t, k in zip(tri, keys, strict=True)])
    flat = rolled.reshape(len(tri), 9)
    return flat[np.lexsort(flat.T[::-1])]


def written(vertices, decimals=3):
    rows = np.asarray(vertices).tolist()
    return np.array([[float(f"{c:.{decimals}f}") for c in v] for v in rows])


def check_neighbours(mesh: Mesh, neighbours: np.ndarray) -> None:
    """Neighbour columns agree with the mesh edge map and with trimesh."""
    faces = mesh.faces
    em = mesh.edge_map
    assert neighbours.shape == (mesh.n_faces, 3)
    for k, (a, b) in enumerate(NEIGHBOUR_EDGES):
        nb = neighbours[:, k]
        edge = em.face_edges[:, a]  # NEIGHBOUR_EDGES[k] = (k, k + 1): half-edge slot k
        assert (a, b) == (k, (k + 1) % 3)
        has = nb >= 0
        # an edge with two faces has a neighbour, any other edge has none
        np.testing.assert_array_equal(has, em.counts[edge] == 2)
        # the neighbour really contains both end points of that edge
        va, vb = faces[has, a], faces[has, b]
        other = faces[nb[has]]
        assert ((other == va[:, None]).any(axis=1) & (other == vb[:, None]).any(axis=1)).all()
        # and it is not the face itself
        assert (nb[has] != np.flatnonzero(has)).all()
    # symmetric: f is a neighbour of each of its neighbours
    for f in range(0, mesh.n_faces, max(1, mesh.n_faces // 500)):
        for g in neighbours[f]:
            if g >= 0:
                assert f in neighbours[g]
    # independent oracle: same set of adjacent face pairs as trimesh
    rows = enumerate(neighbours.tolist())
    ours = {(min(f, g), max(f, g)) for f, row in rows for g in row if g >= 0}
    tm = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
    theirs = {(min(f, g), max(f, g)) for f, g in tm.face_adjacency.tolist()}
    assert ours == theirs


def round_trip(tmp_path, meshes, **kwargs):
    path = tmp_path / "model.dtm"
    write_surpac_dtm(path, meshes, **kwargs)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        return read_surpac_dtm(path)


# -------------------------------------------------------------- round trips


@pytest.mark.parametrize(
    "name",
    ["cube", "icosphere", "ore_solid", "topo"],
)
def test_round_trip(tmp_path, name):
    if name == "cube":
        original = cube(40.0, center=UTM + 0.0004)
    elif name == "icosphere":
        original = icosphere(3, radius=125.5, center=UTM)
    else:
        original = load_synthetic(name)
    is_solid = name != "topo"

    result = round_trip(tmp_path, original, validated=is_solid)
    assert (tmp_path / "model.str").exists()  # the pair is written
    assert result.ids == [(1, 1)]
    tri = result.trisolations[0]
    back = tri.mesh

    # same vertices to the written precision, in the same order
    np.testing.assert_array_equal(back.vertices, written(original.vertices))
    assert np.abs(back.vertices - original.vertices).max() <= 0.5e-3 + 1e-9
    # same faces: identical here, and equal up to index remapping by construction
    np.testing.assert_array_equal(back.faces, original.faces)
    rounded = Mesh(written(original.vertices), original.faces)
    np.testing.assert_array_equal(canonical_triangles(back), canonical_triangles(rounded))
    # flags
    assert tri.closed is is_solid and tri.closed == original.is_closed
    assert tri.validated is is_solid
    assert tri.has_neighbours is True
    assert tri.flags == {
        "neighbours": "yes",
        "validated": "true" if is_solid else "false",
        "closed": "yes" if is_solid else "no",
        **({"direction": "solid"} if is_solid else {}),
    }
    # neighbours as written equal those of the mesh edge map
    np.testing.assert_array_equal(tri.neighbours, mesh_neighbours(original))
    check_neighbours(back, tri.neighbours)
    assert back.area == pytest.approx(original.area, rel=1e-5)

    if is_solid:
        assert back.signed_volume == pytest.approx(original.signed_volume, rel=1e-5)
        assert back.signed_volume > 0  # winding kept
        report = back.validate()
        assert report.is_valid, report.summary()
        assert (tri.neighbours >= 0).all()
    else:
        assert (tri.neighbours < 0).sum() == len(original.edge_map.boundary_edges)
        assert not back.is_closed


def test_volume_matches_analytic_after_round_trip(tmp_path):
    back = round_trip(tmp_path, cube(40.0, center=UTM), validated=True).meshes[0]
    assert back.volume == pytest.approx(64_000.0, rel=1e-12)


def test_validated_flag_is_written_only_when_given(tmp_path):
    """The published layout has no ``validated`` value: none is made up."""
    solid = cube(10.0, center=UTM)
    open_mesh = solid.submesh(np.arange(solid.n_faces - 2))
    result = round_trip(tmp_path, [solid, open_mesh])
    assert [t.validated for t in result] == [None, None]
    assert [t.closed for t in result] == [True, False]
    assert [t.flags for t in result] == [
        {"neighbours": "yes", "closed": "yes", "direction": "solid"},
        {"neighbours": "yes", "closed": "no"},
    ]
    lines = (tmp_path / "model.dtm").read_text().splitlines()
    assert lines[:4] == [
        "model.str,",
        "0, 0.000, 0.000, 0.000, END",
        "OBJECT, 1,",
        "TRISOLATION, 1, neighbours=yes,closed=yes,direction=solid",
    ]
    assert lines[-1] == "END" and not lines[-2].startswith("0,")  # no record of zeros


def test_multiple_objects_and_trisolations(tmp_path):
    a = cube(10.0, center=UTM)
    b = icosphere(1, radius=7.0, center=UTM + [100.0, 0.0, 0.0])
    c = tetrahedron(5.0).translated(UTM + [0.0, 200.0, 0.0])
    d = load_synthetic("topo")
    meshes = [a, b, c, d]
    result = round_trip(
        tmp_path,
        meshes,
        object_ids=[1, 1, 2, 7],
        trisolation_ids=[1, 2, 1, 3],
        validated=[True, True, True, False],
    )
    assert result.ids == [(1, 1), (1, 2), (2, 1), (7, 3)]
    assert len(result) == 4 and len(result.meshes) == 4
    for original, tri in zip(meshes, result.trisolations, strict=True):
        np.testing.assert_array_equal(tri.mesh.vertices, written(original.vertices))
        np.testing.assert_array_equal(tri.mesh.faces, original.faces)
        assert tri.mesh.volume == pytest.approx(original.volume, rel=1e-4)
        check_neighbours(tri.mesh, tri.neighbours)
    assert result.get(7, 3).closed is False
    assert result.get(2).mesh.n_faces == 4
    with pytest.raises(KeyError):
        result.get(3)

    # one OBJECT record per object, one TRISOLATION per mesh, indices global
    text = (tmp_path / "model.dtm").read_text()
    assert text.count("OBJECT, ") == 3 and text.count("TRISOLATION, ") == 4
    strings = read_surpac_str(tmp_path / "model.str")
    assert strings.n_segments == 4
    assert strings.n_points == sum(m.n_vertices for m in meshes)

    # a SurpacTrisolation written again keeps its ids and flags
    path2 = tmp_path / "again.dtm"
    write_surpac_dtm(path2, result.trisolations)
    again = read_surpac_dtm(path2)
    assert again.ids == result.ids
    assert [t.flags for t in again] == [t.flags for t in result]
    assert path2.read_text().split("\n", 1)[1] == text.split("\n", 1)[1]


def test_writer_rejects_bad_ids_and_empty_meshes(tmp_path):
    a = cube(1.0)
    with pytest.raises(SurpacFormatError, match="duplicate"):
        write_surpac_dtm(tmp_path / "x.dtm", [a, a], object_ids=[1, 1], validated=True)
    with pytest.raises(SurpacFormatError, match="not adjacent"):
        write_surpac_dtm(
            tmp_path / "x.dtm",
            [a, a, a],
            object_ids=[1, 2, 1],
            trisolation_ids=[1, 1, 2],
            validated=True,
        )
    with pytest.raises(SurpacFormatError, match="no faces"):
        write_surpac_dtm(tmp_path / "x.dtm", Mesh(a.vertices, np.zeros((0, 3), dtype=int)))
    with pytest.raises(ValueError, match="one entry per mesh"):
        write_surpac_dtm(tmp_path / "x.dtm", [a], object_ids=[1, 2], validated=True)


def test_without_neighbours(tmp_path):
    result = round_trip(tmp_path, cube(2.0), write_neighbours=False, validated=True)
    tri = result.trisolations[0]
    assert tri.has_neighbours is False and tri.flags["neighbours"] == "no"
    assert (tri.neighbours == -1).all()
    assert tri.mesh.is_closed


# ------------------------------------------------------------ hand-written

# A tetrahedron. The points sit in TWO segments (and two string numbers) of the
# .str so that the numbering is pinned down: v counts the records after the two
# header lines, 1-based, null records INCLUDED: P1 = 1, P2 = 2, (null = 3),
# P3 = 4, P4 = 5. Triangles are clockwise seen from outside; n1 is across v1-v2.
HAND_STR = """\
tetra.str, 19-Nov-26, hand written, minimal pair
0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
1, 5300000.000, 500000.000, 100.000,
1, 5300000.000, 500010.000, 100.000,
0, 0.000, 0.000, 0.000,
2, 5300010.000, 500000.000, 100.000,
2, 5300000.000, 500000.000, 110.000,
0, 0.000, 0.000, 0.000,
0, 0.000, 0.000, 0.000, END
"""
HAND_DTM = """\
tetra.str,
0, 0.000, 0.000, 0.000, END
OBJECT, 1,
TRISOLATION, 1, neighbours=yes,closed=yes,direction=solid
1, 1, 2, 4, 2, 4, 3,
2, 1, 5, 2, 3, 4, 1,
3, 1, 4, 5, 1, 4, 2,
4, 2, 5, 4, 2, 3, 1,
END
"""
# what the writer gives for the same mesh: all four points in ONE segment (1..4)
WRITTEN_DTM = """\
tetra.str,
0, 0.000, 0.000, 0.000, END
OBJECT, 1,
TRISOLATION, 1, neighbours=yes,closed=yes,direction=solid
1, 1, 2, 3, 2, 4, 3,
2, 1, 4, 2, 3, 4, 1,
3, 1, 3, 4, 1, 4, 2,
4, 2, 4, 3, 2, 3, 1,
END
"""
# the same tetrahedron as this module's provisional writer wrote it before 2026-11-21:
# null records not counted, counter-clockwise, a record of zeros, no second line
LEGACY_DTM = """\
tetra.str, 19-Nov-26, hand written, minimal pair
OBJECT, 1,
TRISOLATION, 1, neighbours=yes,validated=true,closed=yes
1, 1, 3, 2, 3, 4, 2,
2, 1, 2, 4, 1, 4, 3,
3, 1, 4, 3, 2, 4, 1,
4, 2, 3, 4, 1, 3, 2,
0, 0, 0, 0, 0, 0, 0,
END
"""
# points in internal (east, north, up): P1 origin, P2 +10 east, P3 +10 north, P4 +10 up
HAND_VERTICES = np.array(
    [
        [500000.0, 5300000.0, 100.0],
        [500010.0, 5300000.0, 100.0],
        [500000.0, 5300010.0, 100.0],
        [500000.0, 5300000.0, 110.0],
    ]
)
HAND_FACES = np.array([[0, 2, 1], [0, 1, 3], [0, 3, 2], [1, 2, 3]])
HAND_NEIGHBOURS = np.array([[2, 3, 1], [0, 3, 2], [1, 3, 0], [0, 2, 1]])


def write_hand_pair(tmp_path, dtm_text=HAND_DTM, str_text=HAND_STR, stem="tetra"):
    (tmp_path / f"{stem}.str").write_text(str_text)
    path = tmp_path / f"{stem}.dtm"
    path.write_text(dtm_text)
    return path


def test_hand_written_minimal_pair(tmp_path):
    path = write_hand_pair(tmp_path)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = read_surpac_dtm(path, strict=True)
    assert result.ids == [(1, 1)]
    tri = result.trisolations[0]
    np.testing.assert_array_equal(tri.mesh.vertices, HAND_VERTICES)
    np.testing.assert_array_equal(tri.mesh.faces, HAND_FACES)
    np.testing.assert_array_equal(tri.neighbours, HAND_NEIGHBOURS)
    assert (tri.has_neighbours, tri.validated, tri.closed) == (True, None, True)
    assert tri.flags["direction"] == "solid"
    # geometry: right-angled tetrahedron with 10 m legs, outward winding
    assert tri.mesh.signed_volume == pytest.approx(1000.0 / 6.0, rel=1e-12)
    assert tri.mesh.validate().is_valid
    # the hand-written neighbours are what the edge map gives
    np.testing.assert_array_equal(mesh_neighbours(tri.mesh), HAND_NEIGHBOURS)
    assert result.header.startswith("tetra.str")
    assert result.strings.n_segments == 2


def test_writer_reproduces_hand_written_dtm(tmp_path):
    """The writer's output for the same mesh is the hand-written topology file."""
    path = tmp_path / "tetra.dtm"
    write_surpac_dtm(path, Mesh(HAND_VERTICES, HAND_FACES))
    assert path.read_text() == WRITTEN_DTM
    lines = (tmp_path / "tetra.str").read_text().splitlines()
    assert lines[2] == "1, 5300000.000, 500000.000, 100.000,"  # north first
    assert lines[3] == "1, 5300000.000, 500010.000, 100.000,"
    assert lines[-1] == "0, 0.000, 0.000, 0.000, END"


def test_file_of_the_provisional_writer_is_still_read(tmp_path):
    path = write_hand_pair(tmp_path, dtm_text=LEGACY_DTM)
    with pytest.warns(FormatWarning, match="provisional BakuMine writer"):
        result = read_surpac_dtm(path)
    tri = result.trisolations[0]
    np.testing.assert_array_equal(tri.mesh.vertices, HAND_VERTICES)
    np.testing.assert_array_equal(tri.mesh.faces, HAND_FACES)
    np.testing.assert_array_equal(tri.neighbours, HAND_NEIGHBOURS)
    assert tri.mesh.signed_volume == pytest.approx(1000.0 / 6.0, rel=1e-12)
    # said to follow the published rule, its number 3 is the null record
    with pytest.raises(SurpacFormatError, match="null record"):
        read_surpac_dtm(path, indexing="records")
    with pytest.raises(ValueError, match="indexing"):
        read_surpac_dtm(path, indexing="other")


def test_rings_that_repeat_their_first_point(tmp_path):
    """A triangular prism digitised as two closed rings, as a Surpac solid between two
    sections is: each ring repeats its first point, a null record ends it, and the vertex
    numbers count all of them (ring 2 starts at 6). Analytic volume 30 * 40 / 2 * 10."""
    strings = """\
prism.str, 21-Nov-26, hand written,
0, 0.000, 0.000, 0.000, 0.000, 0.000, 0.000
1, 0.000, 0.000, 0.000,
1, 0.000, 30.000, 0.000,
1, 40.000, 0.000, 0.000,
1, 0.000, 0.000, 0.000,
0, 0.000, 0.000, 0.000,
2, 0.000, 0.000, 10.000,
2, 0.000, 30.000, 10.000,
2, 40.000, 0.000, 10.000,
2, 0.000, 0.000, 10.000,
0, 0.000, 0.000, 0.000,
0, 0.000, 0.000, 0.000, END
"""
    a, b, c, d, e, f = 1, 2, 3, 6, 7, 8  # record numbers of the six corners
    outward = [(a, c, b), (d, e, f), (a, b, e), (a, e, d), (b, c, f), (b, f, e),
               (c, a, d), (c, d, f)]  # counter-clockwise seen from outside  # fmt: skip
    records = [f"{k}, {u}, {w}, {v}, 0, 0, 0," for k, (u, v, w) in enumerate(outward, 1)]
    topology = "\n".join(
        ["prism.str,", "0, 0.000, 0.000, 0.000, END", "OBJECT, 1,",
         "TRISOLATION, 1, neighbours=no,closed=yes,direction=solid", *records, "END"]
    )  # fmt: skip
    path = write_hand_pair(tmp_path, dtm_text=topology + "\n", str_text=strings, stem="prism")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mesh = read_surpac_dtm(path, strict=True).meshes[0]
    assert mesh.n_vertices == 6 and mesh.is_closed
    assert mesh.signed_volume == pytest.approx(6000.0, rel=1e-12)
    assert mesh.validate().is_valid
    np.testing.assert_allclose(mesh.vertices.max(axis=0), [30.0, 40.0, 10.0])  # east, north, up
    text = describe_surpac_file(path)
    assert "10 record(s) with the null records" in text
    assert "(published rule) 0 vertex number(s) name no point" in text


def test_explicit_str_path_and_missing_str(tmp_path):
    path = write_hand_pair(tmp_path)
    other = tmp_path / "points_elsewhere.str"
    (tmp_path / "tetra.str").rename(other)
    with pytest.raises(FileNotFoundError, match="paired"):
        read_surpac_dtm(path)
    result = read_surpac_dtm(path, other)
    np.testing.assert_array_equal(result.meshes[0].faces, HAND_FACES)


def test_tolerant_variants(tmp_path):
    variant = (
        "tetra.str,some-checksum,extra,header,fields\r\n"
        "0, 0.000, 0.000, 0.000, END\r\n"  # string-style second line, kept as preamble
        "\r\n"
        "object,1\r\n"
        "Trisolation , 1 , Neighbors=NO , closed=TRUE, validated=Yes, colour=red\r\n"
        "1,1,2,4\r\n"  # no neighbour columns
        "  2 1 5 2 0 0 0\r\n"  # whitespace separated
        "3, 1, 4, 5, 0, 0, 0,\r\n"
        "4, 2, 5, 4, 0, 0, 0, 99, 98,\r\n"  # extra columns
        "0, 0, 0, 0, 0, 0, 0,\r\n"  # a record of zeros is tolerated
        "end\r\n"
    )
    path = write_hand_pair(tmp_path, dtm_text=variant.replace("\r\n", "\n"))
    path.write_bytes(variant.encode())
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        result = read_surpac_dtm(path)
    assert result.warnings == [] and result.skipped == []
    assert result.preamble == ["0, 0.000, 0.000, 0.000, END"]
    tri = result.trisolations[0]
    np.testing.assert_array_equal(tri.mesh.vertices, HAND_VERTICES)
    np.testing.assert_array_equal(tri.mesh.faces, HAND_FACES)
    assert (tri.has_neighbours, tri.validated, tri.closed) == (False, True, True)
    assert tri.flags == {"neighbours": "NO", "closed": "TRUE", "validated": "Yes", "colour": "red"}
    assert (tri.neighbours == -1).all()


def test_tolerant_structure_deviations(tmp_path):
    # no OBJECT record, no flags, no END, and a junk line
    text = (
        "header\n"
        "TRISOLATION, 4\n"
        "1, 1, 2, 4, 2, 4, 3,\n"
        "2, 1, 5, 2, 3, 4, 1,\n"
        "what is this\n"
        "3, 1, 4, 5, 1, 4, 2,\n"
        "4, 2, 5, 4, 77, 3, 1,\n"
    )
    path = write_hand_pair(tmp_path, dtm_text=text)
    with pytest.warns(FormatWarning):
        result = read_surpac_dtm(path)
    tri = result.trisolations[0]
    assert (tri.object_id, tri.trisolation_id) == (1, 4)
    np.testing.assert_array_equal(tri.mesh.faces, HAND_FACES)
    assert (tri.has_neighbours, tri.validated, tri.closed) == (None, None, None)
    assert result.skipped == [(5, "what is this")]
    joined = " | ".join(result.warnings)
    assert "before any OBJECT" in joined
    assert "no END record" in joined
    assert "skipped" in joined
    assert "neighbour id(s) name no triangle" in joined  # the 77
    assert tri.neighbours[3].tolist() == [0, 2, -1]
    with pytest.raises(SurpacFormatError):
        read_surpac_dtm(path, strict=True)


def test_vertex_index_out_of_range_is_an_error(tmp_path):
    path = write_hand_pair(tmp_path, dtm_text=HAND_DTM.replace("4, 2, 5, 4,", "4, 2, 5, 7,"))
    with pytest.raises(SurpacFormatError, match=r"point numbers 1..7.*first: 7.*6 records"):
        read_surpac_dtm(path)
    # a number that names a null record (3) is not read as the next point
    path = write_hand_pair(tmp_path, dtm_text=HAND_DTM.replace("1, 1, 2, 4,", "1, 1, 2, 3,"))
    with pytest.raises(SurpacFormatError, match="first: 3.*null record"):
        read_surpac_dtm(path)
    # 0-based files are caught as well, instead of being read shifted by one
    zero_based = HAND_DTM.replace("1, 1, 2, 4,", "1, 0, 1, 3,")
    path = write_hand_pair(tmp_path, dtm_text=zero_based)
    with pytest.raises(SurpacFormatError, match="describe_surpac_file"):
        read_surpac_dtm(path)


def test_unreferenced_points_are_dropped(tmp_path):
    extra = HAND_STR.replace(
        "1, 5300000.000, 500000.000, 100.000,\n",
        "9, 1.000, 2.000, 3.000, unused\n0, 0.000, 0.000, 0.000,\n"
        "1, 5300000.000, 500000.000, 100.000,\n",
    )

    def shift(line: str) -> str:
        """Add 2 (the unused point and its null record) to the vertex numbers."""
        if not line[0].isdigit() or line.startswith("0"):
            return line
        t = [int(x) for x in line.rstrip(",").split(", ")]
        return ", ".join(str(x) for x in [t[0], t[1] + 2, t[2] + 2, t[3] + 2, *t[4:]]) + ","

    shifted = "\n".join(shift(ln) for ln in HAND_DTM.splitlines())
    path = write_hand_pair(tmp_path, dtm_text=shifted + "\n", str_text=extra)
    result = read_surpac_dtm(path)
    np.testing.assert_array_equal(result.meshes[0].vertices, HAND_VERTICES)
    np.testing.assert_array_equal(result.meshes[0].faces, HAND_FACES)


# ------------------------------------------------------------------ binary


def test_binary_files_are_rejected(tmp_path):
    blob = b"\x00\x01\x02\x03BINARYDTM" + bytes(range(256)) * 8
    path = write_hand_pair(tmp_path)
    path.write_bytes(blob)
    with pytest.raises(BinaryFileError, match=r"binary \.dtm is not supported; re-save as text"):
        read_surpac_dtm(path)
    # text .dtm with a binary .str
    path = write_hand_pair(tmp_path)
    (tmp_path / "tetra.str").write_bytes(blob)
    with pytest.raises(BinaryFileError, match=r"binary \.str is not supported; re-save as text"):
        read_surpac_dtm(path)


# ---------------------------------------------------------------- describe


def test_describe_surpac_file(tmp_path):
    path = write_hand_pair(tmp_path)
    text = describe_surpac_file(path)
    assert "Surpac DTM file" in text and "1996 format appendix" in text
    assert "objects: 1 [1]; trisolations: 1" in text
    assert "4 triangle(s)" in text
    assert "'neighbours': 'yes'" in text
    assert "vertex indices 1..5" in text  # the null record is number 3
    assert "[1, 1, 2, 4, 2, 4, 3]" in text  # first record as parsed
    assert "END record found: True" in text
    assert "4 point record(s)" in text and "6 record(s) with the null records" in text
    assert "(published rule) 0 vertex number(s) name no point" in text

    text = describe_surpac_file(tmp_path / "tetra.str")
    assert "Surpac string file" in text
    assert "segments: 2; point records: 4" in text
    assert "string 1: 1 segment(s), 2 point(s)" in text
    assert "east 500000.000..500010.000, north 5300000.000..5300010.000" in text
    assert "location='tetra.str'" in text
    assert "warnings: 0" in text and "skipped lines: 0" in text


def test_describe_reports_problems_without_raising(tmp_path):
    bad_dtm = HAND_DTM.replace("4, 2, 5, 4,", "4, 2, 5, 9,").removesuffix("END\n") + "junk line\n"
    path = write_hand_pair(tmp_path, dtm_text=bad_dtm)
    text = describe_surpac_file(path, max_records=2)
    assert "(published rule) 1 vertex number(s) name no point" in text
    assert "no END record" in text
    assert "skipped lines: 1" in text and "'junk line'" in text

    (tmp_path / "tetra.str").unlink()
    assert "NOT FOUND" in describe_surpac_file(path)

    path.write_bytes(b"\x00\x00\x07\x08" * 100)
    text = describe_surpac_file(path)
    assert "BINARY" in text and "00 00 07 08" in text

    messy = tmp_path / "messy.str"
    messy.write_text("h\n0,0,0,0,0,0,0\n1, 2, 1, 3\nnot a record\n")
    text = describe_surpac_file(messy)
    assert "skipped lines: 1" in text and "no END record" in text
    assert "1, 1.0, 2.0, 3.0, []" in text  # east, north, up after the swap


def test_trisolation_dataclass_defaults():
    t = SurpacTrisolation(cube(1.0))
    assert (t.object_id, t.trisolation_id, t.validated) == (1, 1, None)
