"""P2.T9: implicit solid from drillholes.

Independent checks: analytic volumes for the isosurface extractor, a
hand-computed three-hole example for the constraints, scipy's
``RBFInterpolator`` as a second implementation of the interpolant, and the
exact implicit function of the synthetic lens for the recovered solid.
Measured numbers are printed (run with ``-s``).
"""

from __future__ import annotations

import json
import time
from functools import cache, lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.interpolate import RBFInterpolator
from scipy.spatial import cKDTree

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.drillholes.importer import TableSource, import_drillholes
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.implicit import (
    CONTACT,
    INSIDE,
    OUTSIDE,
    estimate_anisotropy,
    extract_constraints,
    fit_field,
    implicit_solid,
)
from bakumine.core.geometry.inside import grid_in_solid, grid_points, points_in_solid
from bakumine.core.geometry.isosurface import (
    marching_tetrahedra,
    mesh_components,
    remove_small_components,
)
from bakumine.core.geometry.mesh import Mesh
from bakumine.synth.config import LensConfig
from bakumine.synth.solid import LensSolid

DATA = Path(__file__).resolve().parents[2] / "data" / "synthetic" / "default"


# ---------------------------------------------------------------------------
# isosurface extractor on analytic fields
# ---------------------------------------------------------------------------


def lattice(lo, hi, n, offset=(0.0, 0.0, 0.0)):
    """Cubic lattice of n^3 nodes on [lo, hi]^3: (X, Y, Z, origin, spacing)."""
    ax = np.linspace(lo, hi, n)
    x, y, z = np.meshgrid(ax, ax, ax, indexing="ij")
    h = ax[1] - ax[0]
    return x, y, z, np.array([lo, lo, lo]) + np.asarray(offset), (h, h, h)


def assert_solid(mesh: Mesh, components: int = 1):
    report = mesh.validate()
    assert report.is_valid, report.issues()
    assert report.is_closed and report.is_manifold and report.is_winding_consistent
    assert report.is_outward
    assert report.n_components == components
    return report


def sphere(x, y, z):
    return np.sqrt(x * x + y * y + z * z) - 1.0


def ellipsoid(x, y, z):
    return np.sqrt((x / 1.4) ** 2 + (y / 0.9) ** 2 + (z / 0.5) ** 2) - 1.0


def torus(x, y, z):
    return (np.sqrt(x * x + y * y) - 1.0) ** 2 + z * z - 0.3**2


def two_spheres(x, y, z):
    a = np.sqrt((x - 0.8) ** 2 + y * y + z * z) - 0.5
    b = np.sqrt((x + 0.8) ** 2 + y * y + z * z) - 0.4
    return np.minimum(a, b)


ANALYTIC = {
    "sphere": (sphere, 4.0 / 3.0 * np.pi, 1),
    "ellipsoid": (ellipsoid, 4.0 / 3.0 * np.pi * 1.4 * 0.9 * 0.5, 1),
    "torus": (torus, 2.0 * np.pi**2 * 1.0 * 0.3**2, 1),
    "two_spheres": (two_spheres, 4.0 / 3.0 * np.pi * (0.5**3 + 0.4**3), 2),
}


@pytest.mark.parametrize("name", list(ANALYTIC))
def test_isosurface_is_a_valid_solid_and_volume_converges(name):
    f, exact, components = ANALYTIC[name]
    errors = []
    for n in (21, 41, 81):
        x, y, z, origin, spacing = lattice(-1.6, 1.6, n)
        mesh = marching_tetrahedra(f(x, y, z), origin, spacing, inside="below")
        assert_solid(mesh, components)
        errors.append(mesh.volume / exact - 1.0)
    print(
        f"{name}: relative volume error at n = 21, 41, 81: "
        + ", ".join(f"{e:+.5f}" for e in errors)
    )
    # second-order convergence: halving the cell divides the error by about 4
    assert abs(errors[0]) < 0.12
    assert abs(errors[1]) < abs(errors[0]) / 3.0
    assert abs(errors[2]) < abs(errors[1]) / 3.0
    assert abs(errors[2]) < 0.008  # worst: the torus, tube radius 7.5 cells
    # Euler characteristic: sphere-like 2 per component, torus 0
    x, y, z, origin, spacing = lattice(-1.6, 1.6, 41)
    mesh = marching_tetrahedra(f(x, y, z), origin, spacing, inside="below")
    euler = mesh.n_vertices - len(mesh.edges) + mesh.n_faces
    assert euler == (0 if name == "torus" else 2 * components)


def test_isosurface_inside_above_is_the_mirror_convention():
    x, y, z, origin, spacing = lattice(-1.6, 1.6, 33)
    below = marching_tetrahedra(sphere(x, y, z), origin, spacing, inside="below")
    above = marching_tetrahedra(-sphere(x, y, z), origin, spacing, inside="above")
    np.testing.assert_array_equal(below.faces, above.faces)
    np.testing.assert_allclose(below.vertices, above.vertices, atol=1e-12)
    shifted = marching_tetrahedra(sphere(x, y, z) + 7.0, origin, spacing, level=7.0, inside="below")
    np.testing.assert_allclose(shifted.vertices, below.vertices, atol=1e-9)


def test_isosurface_at_large_coordinates():
    offset = (5.0e5, 5.3e6, 1.0e3)
    x, y, z, origin, spacing = lattice(-160.0, 160.0, 41)
    values = np.sqrt(x * x + y * y + z * z) - 100.0
    near = marching_tetrahedra(values, origin, spacing, inside="below")
    far = marching_tetrahedra(values, origin + np.array(offset), spacing, inside="below")
    assert_solid(far)
    np.testing.assert_array_equal(near.faces, far.faces)
    np.testing.assert_allclose(far.vertices - np.array(offset), near.vertices, atol=2e-9)
    assert far.volume == pytest.approx(near.volume, rel=1e-9)
    assert far.volume == pytest.approx(4.0 / 3.0 * np.pi * 100.0**3, rel=0.01)


def test_isosurface_rotated_anisotropic_grid():
    rotation = Ellipsoid(angles=(30.0, 15.0, 55.0)).rotation
    spacing = np.array([0.08, 0.1, 0.04])
    shape = (41, 33, 81)
    origin = np.array([100.0, 200.0, 300.0]) - rotation @ (0.5 * (np.array(shape) - 1) * spacing)
    points = grid_points(origin, spacing, shape, rotation)
    values = np.linalg.norm(points - np.array([100.0, 200.0, 300.0]), axis=-1) - 1.0
    mesh = marching_tetrahedra(values, origin, spacing, rotation, inside="below")
    assert_solid(mesh)
    assert mesh.volume == pytest.approx(4.0 / 3.0 * np.pi, rel=0.01)
    np.testing.assert_allclose(mesh.centroid, [100.0, 200.0, 300.0], atol=2e-3)
    radius = np.linalg.norm(mesh.vertices - np.array([100.0, 200.0, 300.0]), axis=1)
    # linear interpolation along an edge of length L misses by about L^2 / 8R; L <= 0.134
    assert np.abs(radius - 1.0).max() < 3e-3


def test_isosurface_closed_where_the_field_reaches_the_grid_boundary():
    # the unit sphere on a grid that only spans [-0.8, 0.8]^3: six flat caps
    x, y, z, origin, spacing = lattice(-0.8, 0.8, 33)
    mesh = marching_tetrahedra(sphere(x, y, z), origin, spacing, inside="below")
    assert_solid(mesh)
    lo, hi = mesh.bounds
    np.testing.assert_allclose(lo, -0.8, atol=1e-3 * spacing[0] * 1.01)
    np.testing.assert_allclose(hi, 0.8, atol=1e-3 * spacing[0] * 1.01)
    # volume of the cube minus what the sphere leaves out: Monte Carlo free check
    rng = np.random.default_rng(1)
    p = rng.uniform(-0.8, 0.8, size=(400_000, 3))
    exact = (np.linalg.norm(p, axis=1) < 1.0).mean() * 1.6**3
    assert mesh.volume == pytest.approx(exact, rel=0.01)
    # a field that is inside everywhere gives the whole box
    box = marching_tetrahedra(np.ones((5, 6, 7)), (0, 0, 0), (1, 1, 1))
    assert_solid(box)
    assert box.volume == pytest.approx(4 * 5 * 6, rel=1e-2)
    # without padding the surface is left open at the boundary
    open_mesh = marching_tetrahedra(sphere(x, y, z), origin, spacing, inside="below", pad=False)
    assert not open_mesh.is_closed


def test_isosurface_degenerate_inputs():
    x, y, z, origin, spacing = lattice(-1.6, 1.6, 17)
    values = sphere(x, y, z)
    assert marching_tetrahedra(values + 10.0, origin, spacing, inside="below").n_faces == 0
    # nodes exactly on the level and NaN nodes are outside; the mesh stays valid
    ax = np.linspace(-2, 2, 21)  # the sphere of radius 1 passes exactly through nodes
    gx, gy, gz = np.meshgrid(ax, ax, ax, indexing="ij")
    exact_nodes = sphere(gx, gy, gz)
    assert (exact_nodes == 0.0).any()
    assert_solid(marching_tetrahedra(exact_nodes, (-2, -2, -2), (0.2, 0.2, 0.2), inside="below"))
    holed = values.copy()
    holed[8, 8, 8] = np.nan  # the centre node: a cavity appears
    mesh = marching_tetrahedra(holed, origin, spacing, inside="below")
    report = mesh.validate()
    assert report.is_valid and report.n_components == 2
    with pytest.raises(ValueError):
        marching_tetrahedra(values[0], origin, spacing)
    with pytest.raises(ValueError):
        marching_tetrahedra(values, origin, spacing, inside="left")


def test_remove_small_components():
    x, y, z, origin, spacing = lattice(-1.6, 1.6, 65)
    mesh = marching_tetrahedra(two_spheres(x, y, z), origin, spacing, inside="below")
    n, _, volume = mesh_components(mesh)
    assert n == 2
    np.testing.assert_allclose(
        np.sort(volume), 4.0 / 3.0 * np.pi * np.array([0.4**3, 0.5**3]), rtol=0.01
    )
    kept, removed = remove_small_components(mesh, 0.4)
    assert removed == 1
    assert_solid(kept.remove_unreferenced_vertices())
    assert kept.volume == pytest.approx(4.0 / 3.0 * np.pi * 0.5**3, rel=0.01)
    assert remove_small_components(mesh, 0.01)[1] == 0
    # the largest shell survives any threshold
    assert remove_small_components(mesh, 100.0)[0].volume == pytest.approx(kept.volume)


# ---------------------------------------------------------------------------
# constraint extraction: three vertical holes, by hand
# ---------------------------------------------------------------------------


def three_holes():
    collar = pd.DataFrame(
        {
            "hole_id": ["A", "B", "C"],
            "x": [1000.0, 1050.0, 1100.0],
            "y": [2000.0, 2000.0, 2000.0],
            "z": [500.0, 505.0, 510.0],
            "max_depth": [100.0, 100.0, 100.0],
        }
    )
    survey = pd.DataFrame({"hole_id": ["A", "B", "C"], "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    geology = pd.DataFrame(
        [
            ("A", 0.0, 25.0, "WASTE"),
            ("A", 25.0, 40.0, "WASTE"),  # same state: merged with the row above
            ("A", 40.0, 52.0, "ORE"),
            ("A", 52.0, 60.0, "ORE"),
            ("A", 60.0, 100.0, "WASTE"),
            ("B", 0.0, 100.0, "WASTE"),  # barren
            ("C", 0.0, 70.0, "DYKE"),  # any other code is outside
            ("C", 70.0, 100.0, "ORE"),  # ends in ore
        ],
        columns=["hole_id", "depth_from", "depth_to", "rock"],
    )
    return collar, survey, geology


def test_constraints_three_hole_example():
    collar, survey, geology = three_holes()
    traces = desurvey(collar, survey)
    c = extract_constraints(
        geology, "rock", "ORE", traces, sample_spacing=10.0, far_spacing=10.0, cap=20.0
    )
    table = c.frame()
    assert c.n_holes == 3 and c.n_holes_intersecting == 2 and c.skipped_holes == ()

    contacts = table[table["kind"] == "contact"].sort_values(["hole_id", "depth"])
    assert contacts["hole_id"].tolist() == ["A", "A", "C"]
    np.testing.assert_allclose(contacts["depth"], [40.0, 60.0, 70.0])
    np.testing.assert_allclose(contacts["x"], [1000.0, 1000.0, 1100.0])
    np.testing.assert_allclose(contacts["y"], 2000.0)
    np.testing.assert_allclose(contacts["z"], [460.0, 440.0, 440.0])
    assert (contacts["value"] == 0.0).all()

    a = table[(table["hole_id"] == "A") & (table["kind"] != "contact")].sort_values("depth")
    # runs 0-40 (4 pieces), 40-60 (2 pieces), 60-100 (4 pieces): midpoints
    np.testing.assert_allclose(a["depth"], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    np.testing.assert_allclose(a["value"], [-20, -20, -15, -5, 5, 5, -5, -15, -20, -20])
    np.testing.assert_allclose(a["z"], 500.0 - a["depth"])
    assert a["kind"].tolist() == ["outside"] * 4 + ["inside"] * 2 + ["outside"] * 4

    b = table[table["hole_id"] == "B"]
    assert len(b) == 10 and (b["kind"] == "outside").all()
    np.testing.assert_allclose(b["value"], -20.0)  # no contact in the hole: -cap
    np.testing.assert_allclose(np.sort(b["z"]), 505.0 - np.arange(95.0, 0.0, -10.0))

    cc = table[(table["hole_id"] == "C") & (table["kind"] != "contact")].sort_values("depth")
    np.testing.assert_allclose(cc["depth"], [5, 15, 25, 35, 45, 55, 65, 75, 85, 95])
    # the end of the hole (100 m, in ore) is not a contact: values keep growing to the cap
    np.testing.assert_allclose(cc["value"], [-20, -20, -20, -20, -20, -15, -5, 5, 15, 20])
    assert not np.isclose(table["depth"], 100.0).any() and not np.isclose(table["depth"], 0.0).any()

    assert set(np.unique(c.kind)) == {CONTACT, INSIDE, OUTSIDE}
    assert (c.value[c.kind == INSIDE] > 0).all() and (c.value[c.kind == OUTSIDE] < 0).all()
    assert np.abs(c.value).max() <= c.cap == 20.0
    runs = c.intercepts
    assert runs[runs["hole_id"] == "A"][["depth_from", "depth_to", "state"]].values.tolist() == [
        [0.0, 40.0, -1],
        [40.0, 60.0, 1],
        [60.0, 100.0, -1],
    ]


def test_constraints_unknown_gaps_thinning_and_code_sets():
    collar, survey, geology = three_holes()
    traces = desurvey(collar, survey)
    # an unlogged gap and a missing code next to ore: no contact is invented there
    gappy = pd.DataFrame(
        [
            ("A", 0.0, 30.0, "WASTE"),
            ("A", 35.0, 50.0, "ORE"),  # gap 30-35
            ("A", 50.0, 60.0, None),  # not logged
            ("A", 60.0, 80.0, "WASTE"),
            ("A", 80.0, 90.0, "ORE"),
            ("Z", 0.0, 10.0, "ORE"),  # no collar
        ],
        columns=["hole_id", "depth_from", "depth_to", "rock"],
    )
    c = extract_constraints(gappy, "rock", "ORE", traces, sample_spacing=10.0, cap=50.0)
    np.testing.assert_allclose(c.depth[c.kind == CONTACT], [80.0])
    assert c.skipped_holes == ("Z",)
    assert not ((c.depth > 30.0) & (c.depth < 35.0)).any()
    assert not ((c.depth > 50.0) & (c.depth < 60.0)).any()
    inside = c.kind == INSIDE
    np.testing.assert_allclose(np.sort(c.depth[inside]), [38.75, 46.25, 85.0])
    np.testing.assert_allclose(np.sort(c.value[inside]), [5.0, 33.75, 41.25])

    # a set of codes forms one domain: the ORE/DYKE boundary disappears
    both = extract_constraints(geology, "rock", {"ORE", "DYKE"}, traces, sample_spacing=10.0)
    assert both.n_contacts == 2 and set(both.hole_id[both.kind == CONTACT]) == {"A"}
    assert (both.value[both.hole_id == "C"] == both.cap).all()

    # thinning keeps the contacts and at most one point per cube
    dense = extract_constraints(
        geology, "rock", "ORE", traces, sample_spacing=1.0, far_spacing=1.0, cap=25.0
    )
    thin = extract_constraints(
        geology, "rock", "ORE", traces, sample_spacing=1.0, far_spacing=1.0, cap=25.0, thin=10.0
    )
    assert thin.n_contacts == dense.n_contacts == 3
    assert len(dense) == 303 and len(thin) < 40
    off = thin.kind != CONTACT
    cells = np.floor((thin.xyz[off] - dense.xyz[dense.kind != CONTACT].min(axis=0)) / 10.0)
    assert len(np.unique(cells, axis=0)) == off.sum()
    # default spacing and cap are derived from the data
    auto = extract_constraints(geology, "rock", "ORE", traces)
    assert auto.sample_spacing == pytest.approx(6.25)  # median ore intercept / 4
    assert auto.far_spacing == pytest.approx(10.0)  # hole length 100 / 10
    assert auto.cap == pytest.approx(25.0)  # median ore intercept (20 and 30 m)
    with pytest.raises(ValueError):
        extract_constraints(geology, "nope", "ORE", traces)


# ---------------------------------------------------------------------------
# RBF interpolant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kernel", ["linear", "cubic", "thin_plate"])
def test_rbf_reproduces_constraints_and_matches_scipy(kernel):
    rng = np.random.default_rng(3)
    centre = np.array([5.0e5, 5.3e6, 900.0])
    points = centre + rng.uniform(-200.0, 200.0, size=(300, 3))
    values = np.sin(points[:, 0] / 90.0) * 20.0 + (points[:, 2] - 900.0) * 0.1
    aniso = Ellipsoid(1.0, 2.0, 5.0, angles=(40.0, 10.0, 30.0))
    f = fit_field(points, values, anisotropy=aniso, kernel=kernel)
    assert f.neighbors is None
    residual = f(points) - values
    print(f"{kernel}: max |residual| at the constraints = {np.abs(residual).max():.2e}")
    assert np.abs(residual).max() < 1e-6

    # scipy as the second implementation, in the same isotropic coordinates
    iso = aniso.to_isotropic(points - centre)
    scipy_kernel = {"thin_plate": "thin_plate_spline"}.get(kernel, kernel)
    oracle = RBFInterpolator(iso, values, kernel=scipy_kernel, degree=1)
    query = centre + rng.uniform(-250.0, 250.0, size=(500, 3))
    expected = oracle(aniso.to_isotropic(query - centre))
    np.testing.assert_allclose(f(query), expected, atol=1e-6 * np.abs(values).max())
    # any leading shape; NaN in gives NaN out
    assert f(query.reshape(50, 10, 3)).shape == (50, 10)
    assert np.isnan(f(np.array([np.nan, 0.0, 0.0])))

    # the local solver agrees closely where it uses all the points
    local = fit_field(points, values, anisotropy=aniso, kernel=kernel, neighbors=300)
    np.testing.assert_allclose(local(query), expected, atol=1e-5 * np.abs(values).max())


def test_rbf_options():
    rng = np.random.default_rng(4)
    points = rng.uniform(-50.0, 50.0, size=(120, 3))
    values = rng.normal(size=120)
    exact = fit_field(points, values)
    assert np.abs(exact(points) - values).max() < 1e-8
    smooth = fit_field(points, values, smoothing=0.05)
    rough = np.abs(smooth(points) - values)
    assert rough.max() > 1e-3  # no longer interpolating
    assert rough.max() < np.abs(values).max() * 2
    # degree 0 with the linear kernel: scipy agrees
    const = fit_field(points, values, degree=0)
    oracle = RBFInterpolator(points - points.mean(axis=0), values, kernel="linear", degree=0)
    q = rng.uniform(-60.0, 60.0, size=(50, 3))
    np.testing.assert_allclose(const(q), oracle(q - points.mean(axis=0)), atol=1e-8)
    # a linear field is reproduced exactly by the drift, also far outside the data
    plane = fit_field(points, 2.0 * points[:, 0] - points[:, 2] + 3.0)
    far = np.array([[500.0, 0.0, -300.0]])
    np.testing.assert_allclose(plane(far), [1303.0], atol=1e-6)
    # coincident points are merged
    doubled = fit_field(np.vstack([points, points[:5]]), np.concatenate([values, values[:5]]))
    assert doubled.n_nodes == 120
    np.testing.assert_allclose(doubled(q), exact(q), atol=1e-8)
    # many points switch to the local solver automatically
    assert fit_field(points, values, global_limit=100).neighbors == 100
    for bad in (dict(kernel="gauss"), dict(kernel="cubic", degree=0), dict(degree=2)):
        with pytest.raises(ValueError):
            fit_field(points, values, **bad)
    with pytest.raises(ValueError, match="singular"):
        flat = points.copy()
        flat[:, 2] = 0.0
        fit_field(flat, values)


def test_field_clipping_by_topography_and_distance():
    rng = np.random.default_rng(5)
    points = rng.uniform(-50.0, 50.0, size=(60, 3))
    values = np.full(60, 10.0) + rng.normal(scale=0.1, size=60)  # positive everywhere
    square = np.array([[-500.0, -500.0, 20.0], [500.0, -500.0, 20.0], [500.0, 500.0, 20.0],
                       [-500.0, 500.0, 20.0]])  # fmt: skip
    topo = DTM(Mesh(square, [[0, 1, 2], [0, 2, 3]]))
    f = fit_field(
        points, values, topography=topo, max_extrapolation=30.0, hole_points=np.zeros((1, 3))
    )
    q = np.array([[0.0, 0.0, 10.0], [0.0, 0.0, 25.0], [29.0, 0.0, 0.0], [31.0, 0.0, 0.0],
                  [0.0, 0.0, 19.5]])  # fmt: skip
    assert f.inside(q).tolist() == [True, False, True, False, True]
    assert f.inside(q, clipped=False).all()
    np.testing.assert_allclose(f(q)[[1, 3, 4]], [-5.0, -1.0, 0.5], atol=1e-9)


# ---------------------------------------------------------------------------
# anisotropy: a thin planar vein cut by a grid of vertical holes
# ---------------------------------------------------------------------------

VEIN_DIP = 40.0  # degrees, dipping towards +x; strike along y
VEIN_THICKNESS = 6.0  # true thickness, m
VEIN_NORMAL = np.array([np.sin(np.radians(VEIN_DIP)), 0.0, np.cos(np.radians(VEIN_DIP))])
VEIN_POINT = np.array([0.0, 0.0, -150.0])


def vein_drilling():
    """6 x 6 vertical holes 40 m apart; the vein drops 33.6 m from one hole to the next."""
    xs = np.arange(6) * 40.0 - 100.0
    ys = np.arange(6) * 40.0 - 100.0
    collars, logs = [], []
    half = 0.5 * VEIN_THICKNESS / VEIN_NORMAL[2]  # vertical half thickness
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            hole = f"V{i}{j}"
            collars.append((hole, x, y, 0.0, 300.0))
            centre = -(VEIN_POINT[2] - x * VEIN_NORMAL[0] / VEIN_NORMAL[2])  # depth of the vein
            logs += [
                (hole, 0.0, centre - half, "W"),
                (hole, centre - half, centre + half, "VEIN"),
                (hole, centre + half, 300.0, "W"),
            ]
    collar = pd.DataFrame(collars, columns=["hole_id", "x", "y", "z", "max_depth"])
    survey = pd.DataFrame(
        {"hole_id": collar["hole_id"], "depth": 0.0, "dip": -90.0, "azimuth": 0.0}
    )
    geology = pd.DataFrame(logs, columns=["hole_id", "depth_from", "depth_to", "rock"])
    return {"collar": collar, "survey": survey, "geology": geology}


def vein_plane_points():
    """Points on the mid-plane of the vein half way between the holes."""
    x, y = np.meshgrid(np.arange(5) * 40.0 - 80.0, np.arange(5) * 40.0 - 80.0, indexing="ij")
    z = VEIN_POINT[2] - x * VEIN_NORMAL[0] / VEIN_NORMAL[2]
    return np.column_stack([x.ravel(), y.ravel(), z.ravel()])


def test_planar_vein_needs_aligned_anisotropy():
    tables = vein_drilling()
    strike = np.array([0.0, 1.0, 0.0])
    down_dip = np.cross(strike, VEIN_NORMAL)
    aligned = Ellipsoid.from_axes(strike, down_dip, VEIN_NORMAL, 1.0, 1.0, 10.0)
    between = vein_plane_points()
    results = {}
    for name, aniso in (("aligned", aligned), ("isotropic", None)):
        r = implicit_solid(
            tables, "geology", "rock", "VEIN", anisotropy=aniso, resolution=2.0, margin=30.0
        )
        assert r.diagnostics["mesh_valid"], r.diagnostics["mesh_issues"]
        results[name] = r
        v = r.mesh.vertices - r.mesh.vertices.mean(axis=0)
        normal = np.linalg.svd(v, full_matrices=False)[2][2]
        tilt = np.degrees(np.arccos(min(1.0, abs(float(normal @ VEIN_NORMAL)))))
        off_plane = np.abs((r.mesh.vertices - VEIN_POINT) @ VEIN_NORMAL)
        print(
            f"vein, {name}: components {r.diagnostics['n_components']}, "
            f"mid-points inside {points_in_solid(r.mesh, between).mean():.2f}, "
            f"normal off by {tilt:.2f} deg, max vertex distance from the mid-plane "
            f"{off_plane.max():.1f} m, volume {r.mesh.volume:.0f}"
        )
        results[name + "_tilt"] = tilt
        results[name + "_off"] = off_plane.max()

    # aligned: one continuous sheet through all 36 intercepts, with the right dip
    a = results["aligned"]
    assert a.diagnostics["n_components"] == 1
    assert points_in_solid(a.mesh, between).all()
    assert results["aligned_tilt"] < 1.0
    assert results["aligned_off"] < VEIN_THICKNESS  # stays a sheet (half thickness is 3 m)
    sheet = 200.0 * (200.0 / np.cos(np.radians(VEIN_DIP))) * VEIN_THICKNESS  # between the holes
    assert sheet < a.mesh.volume < 2.2 * sheet  # plus the 30 m margin all round

    # isotropic: the intercepts do not join across the 40 m between the sections
    iso = results["isotropic"]
    assert iso.diagnostics["n_components"] >= 6
    assert points_in_solid(iso.mesh, between).mean() < 0.5
    assert not points_in_solid(
        iso.mesh, between[::5] + [20.0, 0.0, -20.0 * np.tan(np.radians(VEIN_DIP))]
    ).any()

    # the automatic estimate finds the plane from the intercepts alone
    auto = implicit_solid(tables, "geology", "rock", "VEIN", resolution=2.0, margin=30.0)
    assert auto.anisotropy_source == "auto"
    tilt = np.degrees(np.arccos(abs(float(auto.anisotropy.minor_axis @ VEIN_NORMAL))))
    print(
        f"vein, auto: minor axis off by {tilt:.2f} deg, ratios "
        f"{auto.anisotropy.semi_major_ratio:.2f} / {auto.anisotropy.minor_ratio:.2f}"
    )
    assert tilt < 1.0
    assert auto.diagnostics["n_components"] == 1
    assert points_in_solid(auto.mesh, between).all()


def test_estimate_anisotropy_recovers_a_uniform_ellipsoid():
    rng = np.random.default_rng(7)
    truth = Ellipsoid(100.0, 2.5, 8.0, angles=(30.0, 15.0, 55.0))
    u = rng.normal(size=(40_000, 3))
    u *= (rng.uniform(size=(40_000, 1)) ** (1.0 / 3.0)) / np.linalg.norm(u, axis=1, keepdims=True)
    points = [5.0e5, 5.3e6, 1.0e3] + (u * [40.0, 100.0, 12.5]) @ truth.rotation.T
    est = estimate_anisotropy(points)
    for mine, theirs in (
        (est.major_axis, truth.major_axis),
        (est.semi_major_axis, truth.semi_major_axis),
        (est.minor_axis, truth.minor_axis),
    ):
        assert abs(float(mine @ theirs)) > np.cos(np.radians(1.5))
    assert est.semi_major_ratio == pytest.approx(2.5, rel=0.03)
    assert est.minor_ratio == pytest.approx(8.0, rel=0.03)
    assert est.major_radius == pytest.approx(100.0, rel=0.03)
    assert estimate_anisotropy(points, max_ratio=4.0).minor_ratio == pytest.approx(4.0)
    with pytest.raises(ValueError):
        estimate_anisotropy(points[:3])


# ---------------------------------------------------------------------------
# the synthetic deposit: 80 holes on about 50 m sections, a 32 m thick lens
# ---------------------------------------------------------------------------

# IoU is estimated on a 4 m world-aligned grid of 117 x 150 x 125 = 2 193 750
# points covering the truth block model plus 40 m all round: the true solid by
# its exact implicit function, the recovered solid by grid_in_solid on the mesh.
IOU_SPACING = (4.0, 4.0, 4.0)
IOU_PAD = 40.0


@lru_cache(maxsize=1)
def synthetic():
    identity = {
        "collar": ["hole_id", "x", "y", "z", "max_depth"],
        "survey": ["hole_id", "depth", "dip", "azimuth"],
        "geology": ["hole_id", "depth_from", "depth_to", "rock"],
    }
    maps = {name: {c: c for c in columns} for name, columns in identity.items()}
    db = import_drillholes(
        TableSource(DATA / "collar.csv", maps["collar"]),
        TableSource(DATA / "survey.csv", maps["survey"]),
        {"geology": TableSource(DATA / "geology.csv", maps["geology"])},
    )
    truth = json.loads((DATA / "truth.json").read_text())
    solid = LensSolid.from_config(LensConfig())
    assert solid.volume() == pytest.approx(truth["solid"]["volume_m3"], rel=1e-9)
    origin = np.array(truth["blocks"]["origin"]) - IOU_PAD
    extent = np.array(truth["blocks"]["shape"]) * np.array(truth["blocks"]["size"]) + 2 * IOU_PAD
    shape = tuple(int(n) for n in np.round(extent / IOU_SPACING))
    true_mask = solid.inside(grid_points(origin, IOU_SPACING, shape))
    assert true_mask.sum() * np.prod(IOU_SPACING) == pytest.approx(solid.volume(), rel=2e-3)
    assert not true_mask[[0, -1]].any() and not true_mask[:, [0, -1]].any()
    return db, truth, solid, (origin, shape, true_mask)


def true_anisotropy(truth) -> Ellipsoid:
    """Orientation from truth.json (columns major, semi-major, minor); ratios of the half-axes."""
    axes = np.array(truth["solid"]["axes"])
    a, b, c = truth["solid"]["half_axes"]
    return Ellipsoid.from_axes(axes[:, 0], axes[:, 1], axes[:, 2], a, a / b, a / c)


@lru_cache(maxsize=1)
def synthetic_solid_warmup():
    """Compile the numba kernels on a tiny case so the timings measure the work."""
    tables = vein_drilling()
    implicit_solid(tables, "geology", "rock", "VEIN", resolution=8.0, margin=10.0)


@cache
def synthetic_solid(setting: str):
    db, truth, solid, (origin, shape, true_mask) = synthetic()
    anisotropy = {"true": true_anisotropy(truth), "auto": "auto", "none": None}[setting]
    synthetic_solid_warmup()
    start = time.perf_counter()
    # everything else is the default; without anisotropy the default 2.7 m grid would be
    # 143 x 193 x 137 world-aligned nodes, so that case alone is run at 4 m
    result = implicit_solid(
        db, "geology", "rock", "ORE", anisotropy=anisotropy,
        resolution=4.0 if anisotropy is None else None,
    )  # fmt: skip
    seconds = time.perf_counter() - start
    mask = grid_in_solid(result.mesh, origin, IOU_SPACING, shape)
    iou = (mask & true_mask).sum() / (mask | true_mask).sum()
    return result, float(iou), result.mesh.volume / solid.volume(), seconds


def test_synthetic_solid_with_true_orientation_meets_the_plan_criterion():
    result, iou, volume_ratio, seconds = synthetic_solid("true")
    d = result.diagnostics
    print(
        f"synthetic, true orientation: IoU {iou:.3f}, volume ratio {volume_ratio:.3f}, "
        f"{seconds:.2f} s, {d['n_constraints']} constraints ({d['n_contacts']} contacts), "
        f"grid {d['grid']['shape']} = {d['grid_nodes']} nodes, cell "
        f"{np.round(d['grid']['spacing'], 2).tolist()} m, {d['n_faces']} faces, "
        f"{d['n_components']} component(s), contact residual {d['contact_residual_max']:.1e}"
    )
    assert d["mesh_valid"], d["mesh_issues"]
    assert d["mesh_outward"]
    assert_solid(result.mesh, d["n_components"])
    assert not d["touches_grid_boundary"]
    assert d["n_holes"] == 80
    assert d["contact_residual_max"] < 1e-6
    assert d["constraint_residual_max"] < 1e-6
    assert d["volume"] == pytest.approx(result.mesh.volume)
    assert iou > 0.85
    assert 0.9 < volume_ratio < 1.1


def test_synthetic_solid_anisotropy_settings_report():
    db, truth, solid, _ = synthetic()
    rows = {name: synthetic_solid(name) for name in ("true", "auto", "none")}
    print("\nsetting  IoU    volume/truth  seconds  grid nodes  components")
    for name, (r, iou, ratio, seconds) in rows.items():
        d = r.diagnostics
        print(f"{name:7s}  {iou:.3f}  {ratio:.3f}         {seconds:6.2f}  {d['grid_nodes']:10d}"
              f"  {d['n_components']}")  # fmt: skip
    # how close the automatic estimate gets to the true frame
    auto = rows["auto"][0].anisotropy
    axes = np.array(truth["solid"]["axes"])
    angles = [
        float(np.degrees(np.arccos(min(1.0, abs(float(mine @ axes[:, k]))))))
        for k, mine in enumerate((auto.major_axis, auto.semi_major_axis, auto.minor_axis))
    ]
    a, b, c = truth["solid"]["half_axes"]
    print(
        f"auto anisotropy: axes off by {angles[0]:.1f} / {angles[1]:.1f} / {angles[2]:.1f} deg "
        f"(major / semi-major / minor); ratios {auto.semi_major_ratio:.2f} and "
        f"{auto.minor_ratio:.2f} against {a / b:.2f} and {a / c:.2f} of the true half-axes"
    )
    assert max(angles) < 5.0
    assert auto.semi_major_ratio == pytest.approx(a / b, rel=0.15)
    assert auto.minor_ratio == pytest.approx(a / c, rel=0.35)
    for name in rows:
        assert rows[name][0].diagnostics["mesh_valid"]
    # the automatic frame does as well as the true one; no anisotropy is far worse
    assert rows["auto"][1] > 0.85
    assert rows["none"][1] < rows["true"][1] - 0.1


@pytest.mark.parametrize("setting", ["true", "auto"])
def test_synthetic_solid_honours_the_drilling(setting):
    db, _, _, _ = synthetic()
    result, *_ = synthetic_solid(setting)
    traces = db.desurvey()
    resolution = result.diagnostics["resolution"]

    # 1. the midpoint of every logged run
    runs = result.constraints.intercepts
    mid = 0.5 * (runs["depth_from"] + runs["depth_to"]).to_numpy()
    half = 0.5 * (runs["depth_to"] - runs["depth_from"]).to_numpy()
    xyz, _ = traces.xyz_at(runs["hole_id"].to_numpy(dtype=object), mid)
    inside = points_in_solid(result.mesh, xyz)
    ore = runs["state"].to_numpy() > 0
    print(
        f"{setting}: {ore.sum()} ORE runs (shortest {2 * half[ore].min():.2f} m), "
        f"{(~ore).sum()} WASTE runs; ORE midpoints outside the solid: {(~inside[ore]).sum()}, "
        f"WASTE midpoints inside: {inside[~ore].sum()}"
    )
    # the fitted function honours every run; the mesh every run that is longer than a grid
    # cell (the grid cannot be relied on to see an intercept thinner than its resolution:
    # here one 1.94 m intercept against a 2.67 m cell)
    values = result.evaluate(xyz)
    assert (values[ore] > 0.0).all() and (values[~ore] < 0.0).all()
    assert inside[ore & (2 * half > resolution)].all()
    assert (~inside[ore]).sum() <= 1
    assert not inside[~ore].any()

    # 2. every metre of every hole, except within one grid cell (down-hole) of a contact
    geology = db.intervals["geology"]
    ids, depths, is_ore = [], [], []
    for row in geology.itertuples(index=False):
        n = max(1, int(row.depth_to - row.depth_from))
        d = row.depth_from + (np.arange(n) + 0.5) * (row.depth_to - row.depth_from) / n
        ids.append(np.full(n, row.hole_id, dtype=object))
        depths.append(d)
        is_ore.append(np.full(n, row.rock == "ORE"))
    ids, depths, is_ore = np.concatenate(ids), np.concatenate(depths), np.concatenate(is_ore)
    contacts = result.constraints.frame().query("kind == 'contact'")
    by_hole = {h: g["depth"].to_numpy() for h, g in contacts.groupby("hole_id")}
    to_contact = np.array(
        [np.abs(by_hole[h] - d).min() if h in by_hole else np.inf
         for h, d in zip(ids, depths, strict=True)]
    )  # fmt: skip
    xyz, _ = traces.xyz_at(ids, depths)
    inside = points_in_solid(result.mesh, xyz)
    wrong = inside != is_ore
    print(
        f"{setting}: {len(ids)} points at 1 m along the holes, {wrong.sum()} on the wrong side, "
        f"the furthest {to_contact[wrong].max() if wrong.any() else 0.0:.2f} m (down-hole) "
        f"from a contact; resolution {resolution:.2f} m"
    )
    assert not wrong[to_contact > resolution].any()


def test_synthetic_field_codes_points_like_the_mesh():
    _, _, _, (origin, shape, _) = synthetic()
    result, *_ = synthetic_solid("true")
    coarse = tuple(n // 2 for n in shape)
    spacing = tuple(2 * s for s in IOU_SPACING)
    by_mesh = grid_in_solid(result.mesh, origin, spacing, coarse)
    by_field = result.inside(grid_points(origin, spacing, coarse))
    differ = (by_mesh != by_field).sum()
    print(f"field vs mesh on {by_mesh.size} points: {differ} differ, "
          f"{by_mesh.sum()} inside the mesh, {by_field.sum()} by the field")  # fmt: skip
    # they differ only in the shell between the triangles and the true zero level
    assert differ < 0.03 * by_mesh.sum()
    assert by_field.sum() == pytest.approx(by_mesh.sum(), rel=0.02)


def test_synthetic_clipping_by_topography_and_extrapolation_distance():
    db, truth, _, _ = synthetic()
    topo = DTM(Mesh(np.load(DATA / "topo_vertices.npy"), np.load(DATA / "topo_faces.npy")))
    aniso = true_anisotropy(truth)
    free = synthetic_solid("true")[0]
    # the lens is at least 64 m below the surface: clipping by the real topography changes nothing
    clipped = implicit_solid(db, "geology", "rock", "ORE", anisotropy=aniso, topography=topo)
    assert clipped.mesh.volume == pytest.approx(free.mesh.volume, rel=1e-12)
    # a surface lowered by 150 m cuts the top off
    low = DTM(topo.mesh.translated([0.0, 0.0, -150.0]))
    cut = implicit_solid(db, "geology", "rock", "ORE", anisotropy=aniso, topography=low)
    assert cut.diagnostics["mesh_valid"], cut.diagnostics["mesh_issues"]
    height = low.height_above(cut.mesh.vertices)
    print(f"lowered topography: highest vertex {np.nanmax(height):.3f} m above it, volume "
          f"{cut.mesh.volume / free.mesh.volume:.2f} of the unclipped solid")  # fmt: skip
    # the cut is the zero level of min(field, depth below the surface), linear inside each
    # 10.7 x 10.7 x 2.7 m cell, while the surface is a 25 m triangulation: decimetres
    assert np.nanmax(height) < 0.5
    assert 0.2 * free.mesh.volume < cut.mesh.volume < 0.95 * free.mesh.volume
    # distance limit: no vertex further than the limit from a hole, to within a grid cell
    # (cubic 2.7 m cells here; the distance to the holes is far from linear inside the
    # default 10.7 m long cells)
    limit = 20.0
    near = implicit_solid(
        db, "geology", "rock", "ORE", anisotropy=aniso, max_extrapolation=limit,
        max_cell_stretch=1.0, validate=False,
    )  # fmt: skip
    traces = db.desurvey()
    ids, depths = [], []
    for hole, depth in zip(traces.hole_ids, traces.max_depth, strict=True):
        d = np.arange(0.0, depth, 0.5)
        ids.append(np.full(len(d), hole, dtype=object))
        depths.append(d)
    hole_xyz, _ = traces.xyz_at(np.concatenate(ids), np.concatenate(depths))
    distance, _ = cKDTree(hole_xyz).query(near.mesh.vertices)
    print(f"max_extrapolation {limit} m: furthest vertex {distance.max():.2f} m from a hole, "
          f"volume {near.mesh.volume / free.mesh.volume:.2f} of the unclipped solid")  # fmt: skip
    assert distance.max() < limit + np.linalg.norm(near.grid.spacing)
    assert near.mesh.is_closed
    assert near.mesh.volume < 0.9 * free.mesh.volume


def test_synthetic_performance():
    result, _, _, seconds = synthetic_solid("true")
    d = result.diagnostics
    stages = {k: round(v, 3) for k, v in d["seconds"].items()}
    print(f"synthetic end to end: {seconds:.2f} s for {d['n_constraints']} constraints and "
          f"{d['grid_nodes']} grid nodes; stages {stages}")  # fmt: skip
    assert seconds < 20.0


def test_implicit_solid_argument_errors():
    tables = vein_drilling()
    with pytest.raises(ValueError, match="code"):
        implicit_solid(tables, "geology", "rock")
    with pytest.raises(ValueError, match="nothing to model"):
        implicit_solid(tables, "geology", "rock", "GOLD")
    with pytest.raises(ValueError, match="no boundary"):
        implicit_solid(tables, "geology", "rock", {"VEIN", "W"})
    with pytest.raises(ValueError):
        implicit_solid(tables, "geology", "rock", "VEIN", anisotropy="pca")
    with pytest.warns(UserWarning, match="coarsened"):
        r = implicit_solid(
            tables, "geology", "rock", "VEIN", resolution=1.0, max_nodes=50_000, validate=False
        )
    assert r.diagnostics["grid_nodes"] <= 50_000 and r.diagnostics["grid_coarsening"] > 1.0
    # a table and traces can be passed directly; keep_grid returns the node values
    traces = desurvey(tables["collar"], tables["survey"])
    r = implicit_solid(
        None, tables["geology"], "rock", "VEIN", traces=traces, resolution=4.0, margin=20.0,
        keep_grid=True, min_component_volume=1.0,
    )  # fmt: skip
    assert r.grid_values.shape == r.grid.shape and r.mesh.n_faces > 0
