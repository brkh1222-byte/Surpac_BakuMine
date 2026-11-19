"""P2.T6 acceptance: solid from sections.

Oracles: analytic volumes (box, prisms, ellipsoid, elliptic frusta), the
convex hull (scipy) as the minimal band between convex rings, the original
mesh for re-sectioned solids (volume and point-in-solid agreement), and the
P2.T2 validation for closedness, manifoldness and self-intersection.
"""

import math
import time
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial import ConvexHull

from bakumine.core.geometry.inside import points_in_solid
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.primitives import box, icosphere, torus
from bakumine.core.geometry.section import Section, plane_frame
from bakumine.core.geometry.sections_to_solid import (
    METHODS,
    SectionSolidError,
    densify_ring,
    resample_ring,
    sections_from_mesh,
    solid_from_sections,
    solid_from_strings,
    triangulate_between,
    triangulate_cap,
)
from bakumine.core.io.strings import StringFile
from bakumine.core.io.surpac_str import format_surpac_str, parse_surpac_str

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"
OFF = np.array([5.0e5, 5.3e6, 1000.0])  # mine coordinates


def ellipse(y, rx, rz, n, phase=0.0, centre=(0.0, 0.0), offset=OFF):
    """Ellipse on the vertical section at northing ``y`` (normal = +y)."""
    t = phase + 2 * np.pi * np.arange(n) / n
    ring = np.column_stack(
        [centre[0] + rx * np.cos(t), np.full(n, float(y)), centre[1] + rz * np.sin(t)]
    )
    return ring + offset


def polygon_on_section(xz, y, offset=OFF):
    xz = np.asarray(xz, dtype=float)
    return np.column_stack([xz[:, 0], np.full(len(xz), float(y)), xz[:, 1]]) + offset


def assert_valid(mesh: Mesh):
    report = mesh.validate()
    assert report.is_valid, report.summary()
    assert report.is_outward
    assert mesh.volume > 0


def canonical_triangles(mesh: Mesh) -> np.ndarray:
    """Faces as sorted coordinate triples, independent of vertex and face order."""
    tri = mesh.triangles.reshape(-1, 3, 3)
    rows = []
    for t in tri:
        rows.append(np.array(sorted(map(tuple, t))).reshape(-1))
    rows = np.array(rows)
    return rows[np.lexsort(rows.T[::-1])]


def frustum_volume(h, rx0, rz0, rx1, rz1):
    """Solid between two ellipses whose semi-axes vary linearly (Simpson is exact)."""
    mid = math.pi * 0.25 * (rx0 + rx1) * (rz0 + rz1)
    return h / 6.0 * (math.pi * rx0 * rz0 + 4 * mid + math.pi * rx1 * rz1)


# ------------------------------------------------------------------ bands


def test_two_squares_make_an_exact_box():
    sq = np.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0.0]]) + OFF
    mesh = solid_from_sections([sq, sq + [0, 0, 5.0]])
    assert_valid(mesh)
    assert mesh.n_faces == 12 and mesh.n_vertices == 8
    assert mesh.volume == pytest.approx(30.0, rel=1e-12)
    assert mesh.area == pytest.approx(2 * (6 + 10 + 15), rel=1e-12)
    # input vertices are kept bit-for-bit
    assert {tuple(v) for v in mesh.vertices} == {tuple(v) for v in np.vstack([sq, sq + [0, 0, 5]])}


@pytest.mark.parametrize("method", METHODS)
def test_band_without_twist_for_different_counts_starts_and_windings(method):
    r, h = 50.0, 30.0
    a = ellipse(0.0, r, r, 37)
    b = np.roll(ellipse(h, r, r, 61, phase=0.3), 23, axis=0)[::-1]  # other start, other winding
    band = triangulate_between(a, b, method=method)
    assert band.n_faces == 37 + 61
    assert np.array_equal(band.vertices, np.vstack([a, b]))
    assert band.is_winding_consistent
    assert len(band.edge_map.boundary_edges) == 37 + 61
    # Minimal band between two convex rings = side surface of their convex hull.
    hull = ConvexHull(np.vstack([a, b]) - OFF)
    cap = lambda n: 0.5 * n * r * r * math.sin(2 * math.pi / n)  # noqa: E731
    minimal = hull.area - cap(37) - cap(61)
    assert minimal * (1 - 1e-9) <= band.area <= 1.02 * minimal

    solid = solid_from_sections([a, b], method=method)
    assert_valid(solid)
    assert solid.volume == pytest.approx(hull.volume, rel=0.01)
    assert solid.volume == pytest.approx(math.pi * r * r * h, rel=0.02)


def test_band_handles_repeated_closing_point_and_nonplanar_rings():
    rng = np.random.default_rng(3)
    a = ellipse(0.0, 40, 20, 30)
    b = ellipse(20.0, 35, 22, 44)
    a[:, 1] += rng.uniform(-2, 2, len(a))  # digitised off the section plane
    b[:, 1] += rng.uniform(-2, 2, len(b))
    band = triangulate_between(np.vstack([a, a[:1]]), np.vstack([b, b[:1]]))
    assert band.n_vertices == 74 and band.n_faces == 74
    solid = solid_from_sections([a, b], normal=[0, 1, 0])
    assert_valid(solid)


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("angle", [30.0, 45.0, 60.0])
def test_rotated_ring_gives_valid_band(method, angle):
    """One ring rotated about the section normal: still no self-intersection."""
    t = 2 * np.pi * np.arange(60) / 60
    x, z = 60 * np.cos(t), 20 * np.sin(t)
    ca, sa = math.cos(math.radians(angle)), math.sin(math.radians(angle))
    rotated = np.column_stack([ca * x - sa * z, np.full(60, 30.0), sa * x + ca * z]) + OFF
    solid = solid_from_sections([ellipse(0.0, 60, 20, 80), rotated], method=method)
    assert_valid(solid)
    full = math.pi * 60 * 20 * 30
    assert 0.85 * full < solid.volume < 1.2 * full

    square = ellipse(0.0, 50, 50, 4)
    turned = ellipse(40.0, 50, 50, 4, phase=math.radians(angle))
    assert_valid(solid_from_sections([square, turned], method=method))


def test_tie_lines_are_honoured_in_a_band():
    a = ellipse(0.0, 50, 50, 40)
    b = ellipse(30.0, 50, 50, 40)
    free = triangulate_between(a, b)
    assert free.edge_map.find(0, 40) >= 0  # untwisted: vertex 0 joins vertex 0
    ties = [(0, 5), (20, 27)]
    for i, j in ties:
        assert free.edge_map.find(i, 40 + j) < 0  # not there unless asked for
    band = triangulate_between(a, b, tie_lines=ties)
    for i, j in ties:
        assert band.edge_map.find(i, 40 + j) >= 0
    assert band.n_faces == 80 and band.is_winding_consistent
    # opposite winding of the second ring: same vertices must still be joined
    band = triangulate_between(a, b[::-1], tie_lines=[(0, 39 - 5), (20, 39 - 27)])
    assert band.edge_map.find(0, 40 + 34) >= 0 and band.edge_map.find(20, 40 + 12) >= 0
    with pytest.raises(SectionSolidError, match="cross"):
        triangulate_between(a, b, tie_lines=[(0, 0), (10, 30), (20, 20)])
    with pytest.raises(ValueError, match="out of range"):
        triangulate_between(a, b, tie_lines=[(0, 40)])


def test_tie_lines_are_honoured_in_a_solid():
    secs = [ellipse(y, 50, 30, 40) for y in (0.0, 25.0, 50.0)]
    ties = np.array([[secs[0][0], secs[1][4]], [secs[2][10], secs[1][12]]])
    solid = solid_from_sections(secs, tie_lines=ties)
    assert_valid(solid)
    tree = {tuple(v): i for i, v in enumerate(solid.vertices)}
    for p, q in ties:
        assert solid.edge_map.find(*sorted((tree[tuple(p)], tree[tuple(q)]))) >= 0
    plain = solid_from_sections(secs)
    assert plain.edge_map.find(*sorted((tree[tuple(ties[0][0])], tree[tuple(ties[0][1])]))) < 0
    with pytest.raises(SectionSolidError, match="tie line 0"):
        solid_from_sections(secs, tie_lines=[[secs[0][0] + 1.0, secs[1][0]]])
    with pytest.raises(SectionSolidError, match="tie line 0 does not join"):
        solid_from_sections(secs, tie_lines=[[secs[0][0], secs[2][0]]])


def test_densify_and_resample():
    sq = np.array([[0, 0, 0], [10, 0, 0], [10, 10, 0], [0, 10, 0.0]])
    d = densify_ring(sq, 3.0)
    assert len(d) == 16
    assert np.linalg.norm(np.roll(d, -1, axis=0) - d, axis=1).max() <= 3.0
    assert all((d == p).all(axis=1).any() for p in sq)
    r = resample_ring(sq, 8)
    assert np.allclose(np.linalg.norm(np.roll(r, -1, axis=0) - r, axis=1), 5.0)
    # densified solid: same volume, more faces, tie indices refer to input vertices
    plain = solid_from_sections([sq, sq + [0, 0, 5.0]])
    dense = solid_from_sections([sq, sq + [0, 0, 5.0]], max_edge=3.0)
    assert_valid(dense)
    assert dense.n_faces > plain.n_faces
    assert dense.volume == pytest.approx(500.0, rel=1e-12)
    band = triangulate_between(sq, sq + [0, 0, 5.0], max_edge=3.0, tie_lines=[(1, 1)])
    assert band.n_vertices == 32
    assert band.edge_map.find(4, 16 + 4) >= 0


# ------------------------------------------------------------------- caps

L_SHAPE = [(0, 0), (40, 0), (40, 15), (15, 15), (15, 50), (0, 50)]
U_SHAPE = [(0, 0), (60, 0), (60, 50), (45, 50), (45, 12), (15, 12), (15, 50), (0, 50)]


def in_l(x, z):
    return (x > 0) & (z > 0) & (((x < 40) & (z < 15)) | ((x < 15) & (z < 50)))


def in_u(x, z):
    return (x > 0) & (x < 60) & (z > 0) & (z < 50) & ~((x > 15) & (x < 45) & (z > 12))


@pytest.mark.parametrize(
    "shape, inside, area",
    [(L_SHAPE, in_l, 40 * 15 + 15 * 35), (U_SHAPE, in_u, 60 * 50 - 30 * 38)],
    ids=["L", "U"],
)
def test_concave_outlines(shape, inside, area):
    rng = np.random.default_rng(11)
    ys = [0.0, 20.0, 40.0, 60.0]
    secs = [densify_ring(polygon_on_section(shape, y), 7.0) for y in ys]
    secs[1] = np.roll(secs[1], 5, axis=0)[::-1]
    secs[2] = polygon_on_section(shape, ys[2])  # fewer vertices on this section
    solid = solid_from_sections(secs)
    assert_valid(solid)
    assert solid.volume == pytest.approx(area * 60.0, rel=1e-9)
    pts = rng.uniform([-10, -10, -10], [70, 70, 60], (200_000, 3))
    truth = inside(pts[:, 0], pts[:, 2]) & (pts[:, 1] > 0) & (pts[:, 1] < 60)
    got = points_in_solid(solid, pts + OFF)
    assert np.array_equal(got, truth)  # no triangle outside the shape, none missing

    # the cap on its own: triangles inside the outline, areas add up
    ring = polygon_on_section(shape, 0.0)
    ring[:, 1] += rng.uniform(-0.5, 0.5, len(ring))  # not planar
    tris = triangulate_cap(ring)
    assert tris.shape == (len(ring) - 2, 3)
    flat = (ring - OFF)[:, [0, 2]]
    cen = flat[tris].mean(axis=1)
    assert inside(cen[:, 0], cen[:, 1]).all()
    e1, e2 = flat[tris[:, 1]] - flat[tris[:, 0]], flat[tris[:, 2]] - flat[tris[:, 0]]
    assert 0.5 * np.abs(e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).sum() == pytest.approx(area)


def test_cap_rejects_self_crossing_outline():
    bow = polygon_on_section([(0, 0), (10, 10), (10, 0), (0, 10), (-5, 5)], 0.0)
    with pytest.raises(SectionSolidError, match="cannot be capped"):
        triangulate_cap(bow)


def test_tapered_ends():
    r, h = 20.0, 10.0
    n = 180
    secs = [ellipse(y, r, r, n) for y in (0.0, h)]
    poly = 0.5 * n * r * r * math.sin(2 * math.pi / n)  # area of the n-gon
    flat = solid_from_sections(secs)
    point = solid_from_sections(secs, ends="point")
    taper = solid_from_sections(secs, ends="taper", taper_scale=0.5, end_distance=4.0)
    mixed = solid_from_sections(secs, ends=("flat", "point"))
    for m in (flat, point, taper, mixed):
        assert_valid(m)
    assert flat.volume == pytest.approx(poly * h, rel=1e-9)
    # default distance: half the section spacing; a cone holds 1/3 base x height
    assert point.volume == pytest.approx(poly * (h + 2 * (h / 2) / 3), rel=1e-9)
    assert mixed.volume == pytest.approx(poly * (h + (h / 2) / 3), rel=1e-9)
    assert taper.volume == pytest.approx(poly * (h + 2 * 4.0 * (1 + 0.5 + 0.25) / 3), rel=1e-9)
    lo, hi = point.bounds
    assert lo[1] == pytest.approx(OFF[1] - h / 2) and hi[1] == pytest.approx(OFF[1] + 1.5 * h)
    with pytest.raises(ValueError, match="ends"):
        solid_from_sections(secs, ends="round")


# -------------------------------------------------------------- ellipsoid

A, B, C = 100.0, 40.0, 25.0
V_ELLIPSOID = 4.0 / 3.0 * math.pi * A * B * C


def ellipsoid_sections(n_sections, n_vertices=72):
    """Analytic outlines at the mid-points of ``n_sections`` equal slabs along x."""
    h = 2 * A / n_sections
    out = []
    for x in -A + (np.arange(n_sections) + 0.5) * h:
        f = math.sqrt(1 - x * x / (A * A))
        t = 2 * np.pi * np.arange(n_vertices) / n_vertices
        ring = np.column_stack([np.full(n_vertices, x), B * f * np.cos(t), C * f * np.sin(t)])
        out.append(ring + OFF)
    return out


def test_ellipsoid_from_analytic_sections():
    err = {}
    for n in (5, 10, 20, 40):
        secs = ellipsoid_sections(n)
        for ends in ("flat", "point", "taper"):
            solid = solid_from_sections(secs, ends=ends)
            assert_valid(solid)
            err[n, ends] = solid.volume / V_ELLIPSOID - 1
    print("\nellipsoid, analytic sections, relative volume error (%):")
    for n in (5, 10, 20, 40):
        print(
            f"  N={n:3d} "
            + "  ".join(f"{e}: {100 * err[n, e]:+.3f}" for e in ("flat", "point", "taper"))
        )
    # the plan's criterion: within 2 % for a reasonable number of sections
    assert abs(err[20, "flat"]) < 0.02
    assert abs(err[20, "point"]) < 0.01
    assert abs(err[10, "taper"]) < 0.02
    # convergence, for every kind of end
    for ends in ("flat", "point", "taper"):
        e = [abs(err[n, ends]) for n in (5, 10, 20, 40)]
        assert e[0] > e[1] > e[2] > e[3]
        assert e[3] < 0.004
    # Systematic end effect of flat caps: the two pieces beyond the last
    # sections are missing. Their analytic volume, as a fraction of the solid:
    for n in (10, 20, 40):
        d = A / n  # half a section spacing
        missing = 2 * math.pi * B * C * (d * d / A - d**3 / (3 * A * A)) / V_ELLIPSOID
        assert err[n, "flat"] < -missing  # chords between sections lose a little more
        assert err[n, "flat"] + missing > -0.6 * missing - 0.012 * (10 / n)
        # tapered ends recover part of it, without overshooting
        assert err[n, "flat"] < err[n, "point"] < err[n, "taper"] < 0


def test_ellipsoid_from_mesh_sections():
    mesh = icosphere(4).transformed(np.diag([A, B, C, 1.0])).translated(OFF)
    assert mesh.volume == pytest.approx(V_ELLIPSOID, rel=0.003)
    print("\nellipsoid, sections of an icosphere mesh, relative volume error (%):")
    for spacing, shift in ((10.0, 0.0), (10.0, 3.7), (5.0, 1.85)):
        secs = sections_from_mesh(mesh, OFF + [shift, 0, 0], [1, 0, 0], spacing)
        assert len(secs) in (int(2 * A / spacing) - 1, int(2 * A / spacing))
        solid = solid_from_sections(secs, ends="point")
        assert_valid(solid)
        err = solid.volume / V_ELLIPSOID - 1
        print(f"  spacing {spacing:g} m, shift {shift:g}: {len(secs)} sections, {100 * err:+.3f}")
        assert abs(err) < 0.02
        assert abs(solid.volume / mesh.volume - 1) < 0.02


def test_sections_from_mesh_positions():
    mesh = icosphere(2, radius=10.0, center=OFF)
    secs = sections_from_mesh(mesh, OFF, [0, 0, 2.0], 4.0)
    assert [round(float((s.origin - OFF)[2]), 9) for s in secs] == [-8.0, -4.0, 0.0, 4.0, 8.0]
    assert all(isinstance(s, Section) and all(s.closed) for s in secs)
    assert (
        sections_from_mesh(Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)), OFF, [0, 0, 1], 1.0)
        == []
    )
    with pytest.raises(ValueError):
        sections_from_mesh(mesh, OFF, [0, 0, 1], 0.0)


# ---------------------------------------------------------- synthetic ore


@pytest.fixture(scope="module")
def ore():
    return Mesh(np.load(DATA / "ore_solid_vertices.npy"), np.load(DATA / "ore_solid_faces.npy"))


@pytest.mark.parametrize(
    "spacing, max_volume_error, min_iou", [(25.0, 0.02, 0.97), (50.0, 0.05, 0.92)]
)
def test_synthetic_ore_solid_rebuilds_from_sections(ore, spacing, max_volume_error, min_iou):
    strike = np.array([math.sin(math.radians(30.0)), math.cos(math.radians(30.0)), 0.0])
    secs = sections_from_mesh(ore, [500600.0, 5300500.0, 1000.0], strike, spacing)
    assert len(secs) >= 10
    rng = np.random.default_rng(1)
    lo, hi = ore.bounds
    pts = rng.uniform(lo - 5, hi + 5, (300_000, 3))
    truth = points_in_solid(ore, pts)
    print(f"\nsynthetic ore solid, vertical sections every {spacing:g} m ({len(secs)} sections):")
    for ends in ("flat", "point"):
        solid = solid_from_sections(secs, ends=ends)
        assert_valid(solid)
        err = solid.volume / ore.volume - 1
        got = points_in_solid(solid, pts)
        iou = (got & truth).sum() / (got | truth).sum()
        sym = (got ^ truth).sum() / truth.sum()
        print(
            f"  ends={ends}: volume error {100 * err:+.3f} %, IoU {iou:.4f}, "
            f"symmetric difference {100 * sym:.2f} % of the ore volume, {solid.n_faces} faces"
        )
        assert abs(err) < max_volume_error
        assert iou > min_iou
        assert sym < 2 * (1 - min_iou)


# ------------------------------------------------------------ bifurcation


def y_body():
    return [
        ellipse(0.0, 60, 20, 80),
        ellipse(25.0, 62, 20, 64),
        [ellipse(50.0, 25, 18, 40, centre=(-35, 0)), ellipse(50.0, 25, 18, 48, centre=(35, 3))],
        [ellipse(75.0, 22, 15, 40, centre=(-45, 0)), ellipse(75.0, 20, 15, 36, centre=(45, 5))],
    ]


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("lift", [0.0, 0.5, 1.0])
def test_bifurcation_y_body(method, lift):
    solid = solid_from_sections(y_body(), method=method, bifurcation_lift=lift)
    assert_valid(solid)
    assert solid.validate().n_components == 1
    stem = frustum_volume(25, 60, 20, 62, 20)
    limbs = frustum_volume(25, 25, 18, 22, 15) + frustum_volume(25, 25, 18, 20, 15)
    # The transition slab (25 m thick) holds at least the pyramids over the
    # two child rings and at most the prism over the parent ring.
    children = 2 * math.pi * 25 * 18
    lower = stem + limbs + children * 25 / 3
    upper = stem + limbs + math.pi * 62 * 20 * 25
    assert 0.98 * lower < solid.volume < upper


def test_bifurcation_volume_grows_with_lift():
    v = [solid_from_sections(y_body(), bifurcation_lift=f).volume for f in (0.0, 0.5, 1.0)]
    assert v[0] < v[1] < v[2]


def test_limb_that_ends_early_is_capped_and_merge_is_the_mirror_case():
    secs = [
        ellipse(0.0, 60, 20, 80),
        [ellipse(25.0, 25, 18, 40, centre=(-35, 0)), ellipse(25.0, 25, 18, 48, centre=(35, 3))],
        [ellipse(50.0, 22, 15, 40, centre=(-45, 0))],
        ellipse(75.0, 20, 12, 30, centre=(-50, 0)),
    ]
    solid = solid_from_sections(secs)
    assert_valid(solid)
    # the short limb stops with a flat cap on section 1: nothing of it beyond y = 25
    right = solid.vertices[solid.vertices[:, 0] > OFF[0] + 5]
    assert right[:, 1].max() == pytest.approx(OFF[1] + 25.0)
    pts = np.array([[35.0, 24.0, 3.0], [35.0, 26.0, 3.0], [-45.0, 60.0, 0.0]]) + OFF
    assert points_in_solid(solid, pts).tolist() == [True, False, True]
    # with a pointed end the limb reaches half a spacing further
    pointed = solid_from_sections(secs, ends="point")
    assert_valid(pointed)
    right = pointed.vertices[pointed.vertices[:, 0] > OFF[0] + 5]
    assert right[:, 1].max() == pytest.approx(OFF[1] + 37.5)
    assert pointed.volume > solid.volume
    # two rings merging into one: the same sections in the opposite order
    merged = solid_from_sections(secs[::-1])
    assert_valid(merged)
    assert merged.volume == pytest.approx(solid.volume, rel=1e-9)
    # split and merge again
    diamond = solid_from_sections([secs[0], secs[1], ellipse(50.0, 60, 20, 80)])
    assert_valid(diamond)


def test_separate_bodies_stay_separate():
    secs = [
        [ellipse(y, 10, 8, 24, centre=(-40, 0)), ellipse(y, 12, 6, 30, centre=(40, 0))]
        for y in (0.0, 10.0, 20.0)
    ]
    solid = solid_from_sections(secs)
    assert_valid(solid)
    assert solid.validate().n_components == 2


def test_one_to_three_is_reported():
    kids = [ellipse(20.0, 12, 10, 24, centre=(c, 0)) for c in (-45, 0, 45)]
    with pytest.raises(SectionSolidError, match="more than two") as info:
        solid_from_sections([ellipse(0.0, 70, 20, 80), kids])
    assert info.value.section_pairs == [(0, 1)]


def test_torus_sections_split_and_merge_again():
    """Planes parallel to the axis of a torus: 1 ring, then 2, then 1 again."""
    mesh = torus().translated(OFF)  # major radius 2, minor radius 0.5
    for spacing in (0.25, 0.5, 1.0):
        secs = sections_from_mesh(mesh, OFF + 0.013, [1, 0, 0], spacing)
        counts = [len(s) for s in secs]
        assert counts[0] == counts[-1] == 1 and max(counts) == 2
        solid = solid_from_sections(secs, ends="point")
        assert_valid(solid)
        assert solid.validate().n_components == 1
        # genus 1: V - E + F = 0
        assert solid.n_vertices - len(solid.edges) + solid.n_faces == 0
        assert 0.9 < solid.volume / mesh.volume < 1.02


def test_box_sections_with_collinear_points():
    """Sections of a box mesh carry collinear points (on the face diagonals)."""
    mesh = box(OFF, OFF + [10.0, 20.0, 30.0])
    secs = sections_from_mesh(mesh, OFF + [0.5, 0, 0], [1, 0, 0], 1.0)
    assert len(secs) == 10 and len(secs[0].polylines[0]) > 4
    solid = solid_from_sections(secs)
    assert_valid(solid)
    assert solid.volume == pytest.approx(9.0 * 20.0 * 30.0, rel=1e-9)
    for normal in ([1, 1, 0.3], [0.2, 0.1, 1.0]):  # triangles, quadrilaterals, hexagons
        secs = sections_from_mesh(mesh, OFF + [0.5, 0, 0], normal, 1.5)
        solid = solid_from_sections(secs, ends="point")
        assert_valid(solid)
        assert solid.volume == pytest.approx(mesh.volume, rel=0.02)


def test_plunging_body_without_overlap_between_sections():
    def secs(shift):
        return [ellipse(20.0 * k, 8, 3, 24, centre=(shift * k, 0)) for k in range(5)]

    prism = 0.5 * 24 * 8 * 3 * math.sin(2 * math.pi / 24) * 80.0
    # no overlap, but within the sum of the ring radii: joined by proximity
    solid = solid_from_sections(secs(12.0))
    assert_valid(solid)
    assert solid.volume == pytest.approx(prism, rel=1e-9)
    # further apart: not joined by default, and that is reported, not capped twice
    with pytest.raises(SectionSolidError, match="no partner") as info:
        solid_from_sections(secs(18.0))
    assert info.value.section_pairs == [(0, 0)]
    solid = solid_from_sections(secs(18.0), match_distance=30.0)
    assert_valid(solid)
    assert solid.validate().n_components == 1
    assert solid.volume == pytest.approx(prism, rel=1e-9)


# ------------------------------------------------------ errors, diagnostics


def test_hole_in_ring_raises_not_implemented():
    outer = [ellipse(y, 50, 30, 60) for y in (0.0, 20.0)]
    hole = ellipse(20.0, 10, 8, 20)
    with pytest.raises(NotImplementedError, match="inside ring"):
        solid_from_sections([outer[0], [outer[1], hole]])


def test_open_strings_raise_a_clear_error():
    closed = [np.vstack([r, r[:1]]) for r in (ellipse(0.0, 30, 20, 20), ellipse(10.0, 30, 20, 20))]
    strings = StringFile.from_arrays([closed[0], closed[1][:-1]], numbers=[1, 7])
    with pytest.raises(SectionSolidError, match=r"segment 1 \(string 7, 20 points\) is open"):
        solid_from_strings(strings)
    frame = plane_frame([0, 1, 0])
    open_section = Section(OFF, *frame, (ellipse(0.0, 30, 20, 20),), (False,))
    with pytest.raises(SectionSolidError, match="section 1 has an open polyline"):
        solid_from_sections([closed[1], open_section])
    with pytest.raises(SectionSolidError, match="at least two"):
        solid_from_sections([closed[0]])
    with pytest.raises(SectionSolidError, match="same position"):
        solid_from_sections([closed[0], closed[0]], normal=[0, 1, 0])


def test_failure_names_the_section_pair():
    square = [(-20, -20), (20, -20), (20, 20), (-20, 20)]
    # Section 2 is digitised wrongly: the outline crosses itself.
    crossed = [(-20, -20), (20, -20), (-10, 20), (20, 20), (20, 5), (-20, 15)]
    secs = [polygon_on_section(square, y) for y in (0.0, 10.0, 20.0, 30.0, 40.0)]
    secs[2] = polygon_on_section(crossed, 20.0)
    with pytest.raises(SectionSolidError) as info:
        solid_from_sections(secs)
    err = info.value
    assert err.report is not None and not err.report.is_valid
    assert err.mesh is not None
    assert err.problems
    assert all(p.kind == "self_intersection" for p in err.problems)
    # every problem is at or next to the faulty section 2, and the bands are named
    assert set(err.section_pairs) <= {(1, 2), (2, 2), (2, 3)}
    assert {(1, 2), (2, 3)} & set(err.section_pairs)
    assert "between sections" in str(err)
    for p in err.problems:
        assert p.count > 0 and p.positions[0] <= p.positions[1]
    # indices are those of the caller's list, whatever its order
    shuffled = [secs[4], secs[2], secs[0], secs[3], secs[1]]
    with pytest.raises(SectionSolidError) as info:
        solid_from_sections(shuffled)
    assert set(info.value.section_pairs) <= {(4, 1), (1, 1), (1, 3)}
    assert {(4, 1), (1, 3)} & set(info.value.section_pairs)
    # the same input without validation is returned as it is
    bad = solid_from_sections(secs, validate=False)
    assert bad.is_closed and not bad.validate().is_valid
    # skipping the self-intersection test hides this defect (documented trade-off)
    assert solid_from_sections(secs, check_self_intersection=False).is_closed

    # a figure of eight has no net area: the band step fails and names the pair
    secs[2] = polygon_on_section([(-20, -20), (20, 20), (20, -20), (-20, 20)], 20.0)
    with pytest.raises(SectionSolidError, match="between sections 1 and 2") as info:
        solid_from_sections(secs)
    assert info.value.problems[0].kind == "band"
    assert info.value.section_pairs == [(1, 2)]


def test_sections_are_ordered_automatically():
    secs = ellipsoid_sections(12)
    ordered = solid_from_sections(secs)
    rng = np.random.default_rng(5)
    shuffled = [secs[i] for i in rng.permutation(len(secs))]
    again = solid_from_sections(shuffled)
    assert np.array_equal(canonical_triangles(ordered), canonical_triangles(again))


# ---------------------------------------------------------------- strings


def test_round_trip_through_strings():
    secs = y_body()
    direct = solid_from_sections(secs)
    rings = [r for s in secs for r in (s if isinstance(s, list) else [s])]
    numbers = [1, 1, 1, 2, 1, 2]
    strings = StringFile.from_arrays([np.vstack([r, r[:1]]) for r in rings], numbers=numbers)
    from_strings = solid_from_strings(strings)
    assert_valid(from_strings)
    assert np.array_equal(canonical_triangles(direct), canonical_triangles(from_strings))
    # in any segment order
    mixed = StringFile(segments=strings.segments[::-1])
    assert np.array_equal(
        canonical_triangles(direct), canonical_triangles(solid_from_strings(mixed))
    )
    # through the Surpac text format (millimetre precision)
    text = format_surpac_str(strings, decimals=3)
    back = solid_from_strings(parse_surpac_str(text))
    assert_valid(back)
    assert back.n_faces == direct.n_faces
    assert back.volume == pytest.approx(direct.volume, rel=1e-4)
    # only one string number: the second limb is left out
    one = solid_from_strings(strings, numbers=[1])
    assert_valid(one)
    assert one.volume < direct.volume


def test_strings_on_inclined_sections_are_grouped_by_plane(ore):
    normal = np.array([0.5, 0.7, 0.3])
    secs = sections_from_mesh(ore, ore.centroid, normal, 40.0)
    direct = solid_from_sections(secs)
    rings = [np.vstack([p, p[:1]]) for s in secs for p in s.polylines]
    solid = solid_from_strings(StringFile.from_arrays(rings[::-1]))  # normal is estimated
    assert_valid(solid)
    assert solid.n_faces == direct.n_faces
    assert solid.volume == pytest.approx(direct.volume, rel=1e-9)


# ------------------------------------------------------------ performance


def test_performance_100_sections_of_500_vertices():
    secs = [ellipse(10.0 * k, 50 + 5 * math.sin(k), 30, 500, phase=0.1 * k) for k in range(100)]
    solid_from_sections(secs[:2])  # compile / load the numba kernels
    t0 = time.perf_counter()
    solid = solid_from_sections(secs, check_self_intersection=False)
    t_build = time.perf_counter() - t0
    t0 = time.perf_counter()
    checked = solid_from_sections(secs)
    t_checked = time.perf_counter() - t0
    t0 = time.perf_counter()
    greedy = solid_from_sections(secs, method="greedy", validate=False)
    t_greedy = time.perf_counter() - t0
    print(
        f"\n100 sections x 500 vertices ({solid.n_faces} faces): min_area {t_build:.2f} s, "
        f"with self-intersection check {t_checked:.2f} s, greedy without validation "
        f"{t_greedy:.2f} s"
    )
    assert solid.n_faces == checked.n_faces == greedy.n_faces == 99 * 1000 + 2 * 498
    assert solid.is_closed
    assert t_build < 15.0
    assert t_checked < 30.0


def test_large_rings():
    a = ellipse(0.0, 300, 200, 3000)
    b = np.roll(ellipse(40.0, 280, 210, 2500), 700, axis=0)
    t0 = time.perf_counter()
    solid = solid_from_sections([a, b], check_self_intersection=False)
    assert time.perf_counter() - t0 < 30.0
    assert solid.is_closed
    assert solid.volume == pytest.approx(frustum_volume(40, 300, 200, 280, 210), rel=1e-3)
