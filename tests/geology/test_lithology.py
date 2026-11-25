"""Rock units by the largest signed-distance field (P9.T7): research.md 5.7.2.

Independent checks: analytic bodies drilled by synthetic straight holes (a dipping plane,
three parallel layers, a sphere of known volume), plain pandas for the group defaults,
loops for the margins, trimesh for the mesh volumes, and the exact truth of the synthetic
layered deposit. The numbers measured on the layered deposit are printed (``pytest -s``).
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
import trimesh
from scipy.spatial import Delaunay, cKDTree

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.geology.lithology import (
    AUTO_MAX_RATIO,
    EventSpec,
    FieldSettings,
    LithologyEvent,
    LithologyStack,
    UnitFields,
    fit_stack,
    fit_unit_fields,
    group_contacts,
    group_runs,
    logged_hole_points,
    margins,
    unit_meshes,
)
from bakumine.core.geology.model import GeologyModel
from bakumine.core.geology.units import MODELLED_KINDS, UnitDictionary
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.implicit import CONTACT, INSIDE, OUTSIDE
from bakumine.core.geometry.isosurface import GridSpec
from bakumine.core.io.obj import read_obj

# --------------------------------------------------------------------------
# synthetic drilling of an analytic body
# --------------------------------------------------------------------------


def drill(label, collars, *, dip=-90.0, azimuth=0.0, length=200.0, scan=0.5):
    """Straight holes logged from ``label(points) -> codes``; contacts by bisection.

    Returns ``(table, traces)``: ``hole_id, depth_from, depth_to, unit`` and the traces.
    """
    collars = np.asarray(collars, dtype=np.float64)
    n = len(collars)
    dip = np.broadcast_to(np.asarray(dip, dtype=np.float64), n)
    azimuth = np.broadcast_to(np.asarray(azimuth, dtype=np.float64), n)
    d, a = np.radians(dip), np.radians(azimuth)
    direction = np.column_stack([np.cos(d) * np.sin(a), np.cos(d) * np.cos(a), np.sin(d)])
    ids = [f"H{i + 1:03d}" for i in range(n)]
    collar = pd.DataFrame(
        {
            "hole_id": ids,
            "x": collars[:, 0],
            "y": collars[:, 1],
            "z": collars[:, 2],
            "max_depth": length,
        }
    )
    survey = pd.DataFrame(
        {
            "hole_id": np.repeat(ids, 2),
            "depth": np.tile([0.0, length], n),
            "dip": np.repeat(dip, 2),
            "azimuth": np.repeat(azimuth, 2),
        }
    )
    depths = np.arange(0.0, length + 1e-9, scan)
    rows = []
    for i in range(n):

        def at(depth, i=i):
            return label(collars[i] + np.atleast_1d(depth)[:, None] * direction[i])

        codes = at(depths)
        start = 0.0
        for k in np.flatnonzero(codes[1:] != codes[:-1]):
            lo, hi = depths[k], depths[k + 1]
            for _ in range(40):
                mid = 0.5 * (lo + hi)
                if at(mid)[0] == codes[k]:
                    lo = mid
                else:
                    hi = mid
            rows.append((ids[i], start, hi, codes[k]))
            start = hi
        rows.append((ids[i], start, length, codes[-1]))
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])
    return table, desurvey(collar, survey)


def collar_grid(lo, hi, spacing, z=0.0, offset=0.0):
    axis = np.arange(lo + offset, hi + 1e-9, spacing)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    return np.column_stack([x.ravel(), y.ravel(), np.full(x.size, z)])


def lattice(lo, hi, spacing):
    axes = [np.arange(a + spacing / 2.0, b, spacing) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, len(axes))


def contact_z(fields: UnitFields, upper: int, lower: int, xy, z_lo, z_hi, step=1.0):
    """Elevation of the contact ``lower`` below ``upper`` along vertical lines (one per xy):
    where the label changes, placed by linear interpolation of ``f_upper - f_lower``."""
    z = np.arange(z_lo, z_hi + 1e-9, step)
    out = np.full(len(xy), np.nan)
    pts = np.column_stack([np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))])
    v = fields.values(pts)
    label = np.argmax(v, axis=1).reshape(len(xy), len(z))
    diff = (v[:, upper] - v[:, lower]).reshape(len(xy), len(z))
    for i, row in enumerate(diff):
        k = np.flatnonzero((label[i, :-1] == lower) & (label[i, 1:] == upper))
        if len(k) == 1:
            t = -row[k[0]] / (row[k[0] + 1] - row[k[0]])
            out[i] = z[k[0]] + t * step
    return out


# a plane dipping 25 degrees towards 110, through (0, 0, -90)
PLANE_NORMAL = np.array(
    [
        np.sin(np.radians(25)) * np.sin(np.radians(110)),
        np.sin(np.radians(25)) * np.cos(np.radians(110)),
        np.cos(np.radians(25)),
    ]
)
PLANE_POINT = np.array([0.0, 0.0, -90.0])


def half_spaces(points):
    s = (np.asarray(points) - PLANE_POINT) @ PLANE_NORMAL
    return np.where(s >= 0.0, "UP", "LOW").astype(object)


@pytest.fixture(scope="module")
def two_units():
    collars = collar_grid(-100.0, 100.0, 40.0)
    dip = np.where(np.arange(len(collars)) % 3 == 0, -60.0, -90.0)
    table, traces = drill(half_spaces, collars, dip=dip, azimuth=290.0, length=200.0)
    fields = fit_unit_fields(table, table["unit"], ["UP", "LOW"], traces)
    return table, traces, fields


LAYER_TOPS = (-60.0, -105.0)  # s = n . (p - P) of the two contacts, along the pole below


def three_layers(points):
    s = np.asarray(points) @ PLANE_NORMAL
    return np.where(s >= LAYER_TOPS[0], "A", np.where(s >= LAYER_TOPS[1], "B", "C")).astype(object)


@pytest.fixture(scope="module")
def layers():
    collars = collar_grid(-100.0, 100.0, 40.0)
    dip = np.where(np.arange(len(collars)) % 4 == 1, -65.0, -90.0)
    table, traces = drill(three_layers, collars, dip=dip, azimuth=200.0, length=200.0)
    fields = fit_unit_fields(table, table["unit"], ["A", "B", "C"], traces)
    return table, traces, fields


SPHERE_CENTRE = np.array([3.0, -4.0, -100.0])
SPHERE_RADIUS = 60.0


def sphere(points):
    r = np.linalg.norm(np.asarray(points) - SPHERE_CENTRE, axis=1)
    return np.where(r < SPHERE_RADIUS, "BALL", "HOST").astype(object)


@pytest.fixture(scope="module")
def ball():
    collars = collar_grid(-100.0, 100.0, 25.0)
    table, traces = drill(sphere, collars, length=200.0)
    fields = fit_unit_fields(table, table["unit"], ["HOST", "BALL"], traces)
    return table, traces, fields


# --------------------------------------------------------------------------
# margins
# --------------------------------------------------------------------------


def test_margins_against_loops():
    rng = np.random.default_rng(1)
    v = rng.normal(size=(200, 5))
    v[10, 2] = v[10, 4] = 9.0  # a tie for the largest
    expected = np.empty_like(v)
    for r in range(v.shape[0]):
        for i in range(v.shape[1]):
            expected[r, i] = v[r, i] - max(v[r, j] for j in range(v.shape[1]) if j != i)
    got = margins(v)
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-14)
    # exactly one positive margin per row, at the argmax, except on a tie (two zeros)
    assert ((got > 0).sum(axis=1) == 1)[np.arange(200) != 10].all()
    assert (got[10] > 0).sum() == 0 and got[10, 2] == 0.0 and got[10, 4] == 0.0
    assert (np.argmax(got, axis=1) == np.argmax(v, axis=1)).all()


def test_margins_edge_cases():
    one = np.array([[1.5], [-2.0]])
    np.testing.assert_array_equal(margins(one), one)
    v = np.array([[1.0, np.nan, 0.0], [3.0, 1.0, 2.0]])
    got = margins(v)
    assert np.isnan(got[0]).all()
    np.testing.assert_allclose(got[1], [1.0, -2.0, -1.0])
    two = np.array([[0.3, -0.7]])
    np.testing.assert_allclose(margins(two), [[1.0, -1.0]])


# --------------------------------------------------------------------------
# analytic case 1: two half-spaces
# --------------------------------------------------------------------------


def test_two_units_reduce_to_one_solid(two_units):
    """With two units the constraint points coincide and d_1 = -d_2, so f_1 = -f_2."""
    _, _, fields = two_units
    up, low = fields.constraints
    np.testing.assert_allclose(up.xyz, low.xyz, atol=1e-9)
    np.testing.assert_allclose(up.value, -low.value, atol=1e-9)
    pts = np.random.default_rng(2).uniform([-120, -120, -220], [120, 120, 20], size=(4000, 3))
    v = fields.values(pts)
    assert np.abs(v[:, 0] + v[:, 1]).max() < 1e-7 * fields.cap


def test_half_space_contact_plane_recovered(two_units):
    """Tolerance: RMS distance to the true plane <= 0.5 m, largest <= 2 m, inside the
    drilled area (holes 40 m apart, plane dipping 25 degrees)."""
    _, _, fields = two_units
    xy = lattice([-90.0, -90.0], [90.0, 90.0], 10.0)
    z = contact_z(fields, 0, 1, xy, -200.0, 0.0)
    assert np.isfinite(z).all()
    points = np.column_stack([xy, z])
    distance = (points - PLANE_POINT) @ PLANE_NORMAL
    rms = float(np.sqrt(np.mean(distance**2)))
    print(f"\nhalf-spaces: contact RMS {rms:.3f} m, max {np.abs(distance).max():.3f} m")
    assert rms <= 0.5
    assert np.abs(distance).max() <= 2.0
    # and the labels agree with the truth away from the contact
    pts = np.random.default_rng(3).uniform([-90, -90, -190], [90, 90, -10], size=(20000, 3))
    truth = half_spaces(pts)
    got = np.array(fields.units, dtype=object)[fields.labels(pts)]
    far = np.abs((pts - PLANE_POINT) @ PLANE_NORMAL) > 2.0
    assert (got[far] == truth[far]).all()
    assert (got == truth).mean() >= 0.995


# --------------------------------------------------------------------------
# analytic case 2: three parallel layers
# --------------------------------------------------------------------------


def test_three_parallel_layers(layers):
    """Tolerance: each contact within 1.0 m RMS of its plane; >= 98.5 % of random points
    in the drilled volume get the true layer (the middle layer is 45 m thick)."""
    _, _, fields = layers
    xy = lattice([-90.0, -90.0], [90.0, 90.0], 10.0)
    for (upper, lower), s_true in zip(((0, 1), (1, 2)), LAYER_TOPS, strict=True):
        z = contact_z(fields, upper, lower, xy, -200.0, 0.0)
        assert np.isfinite(z).all()
        distance = np.column_stack([xy, z]) @ PLANE_NORMAL - s_true
        rms = float(np.sqrt(np.mean(distance**2)))
        print(
            f"\nthree layers: contact {fields.units[upper]}/{fields.units[lower]} RMS {rms:.3f} m"
        )
        assert rms <= 1.0
    pts = np.random.default_rng(4).uniform([-90, -90, -190], [90, 90, -10], size=(30000, 3))
    got = np.array(fields.units, dtype=object)[fields.labels(pts)]
    accuracy = float((got == three_layers(pts)).mean())
    print(f"three layers: label accuracy {accuracy:.4f}")
    assert accuracy >= 0.985


# --------------------------------------------------------------------------
# analytic case 3: a sphere in a host
# --------------------------------------------------------------------------


def test_sphere_volume(ball):
    """A sphere of r = 60 m in a host, vertical holes 25 m apart.

    Tolerance, with a cap of 30 m and points 5 m apart: volume within 8 % of 4/3 pi r^3,
    by lattice count of exact labels and by the closed mesh, and IoU >= 0.88.

    With the group defaults the cap is the median run length, here the 200 m of the barren
    holes, and the body is distorted (the large negative values of the host outweigh it,
    see ``extract_constraints``): only a loose 15 % / IoU 0.70 is asserted, and the
    figures are printed.
    """
    table, traces, default = ball
    analytic = 4.0 / 3.0 * np.pi * SPHERE_RADIUS**3
    lo, hi = SPHERE_CENTRE - 80.0, SPHERE_CENTRE + 80.0
    pts = lattice(lo, hi, 2.5)
    truth = sphere(pts) == "BALL"
    capped = fit_unit_fields(
        table, table["unit"], ["HOST", "BALL"], traces, FieldSettings(cap=30.0, sample_spacing=5.0)
    )
    grid = GridSpec(lo, (4.0, 4.0, 4.0), (41, 41, 41))
    nodes = grid.slab_points(0, 41).reshape(-1, 3)
    for name, fields, tolerance, floor in (
        ("cap 30 m", capped, 0.08, 0.88),
        ("defaults", default, 0.15, 0.70),
    ):
        inside = fields.labels(pts) == 1
        counted = inside.sum() * 2.5**3
        node_margins = fields.node_values(nodes).T.reshape(2, 41, 41, 41)
        mesh = unit_meshes(node_margins, grid, fields.units)["BALL"]
        iou = (inside & truth).sum() / (inside | truth).sum()
        print(
            f"\nsphere, {name} (cap {fields.cap:.0f} m): analytic {analytic:.0f} m3, lattice "
            f"{counted / analytic - 1:+.2%}, mesh {mesh.volume / analytic - 1:+.2%}, IoU {iou:.3f}"
        )
        assert counted == pytest.approx(analytic, rel=tolerance)
        assert mesh.volume == pytest.approx(analytic, rel=tolerance)
        assert mesh.volume == pytest.approx(counted, rel=0.02)  # mesh and labels agree
        assert mesh.is_closed
        assert iou >= floor


# --------------------------------------------------------------------------
# properties of the largest-field rule
# --------------------------------------------------------------------------


def reproduction(fields: UnitFields) -> tuple[int, int]:
    """(points whose label is the logged unit, points) over the off-contact constraints."""
    right = total = 0
    for i, c in enumerate(fields.constraints):
        inside = c.kind == INSIDE
        outside = c.kind == OUTSIDE
        assert not (c.kind == CONTACT).all()
        right += int((fields.labels(c.xyz[inside]) == i).sum())
        right += int((fields.labels(c.xyz[outside]) != i).sum())
        total += int(inside.sum() + outside.sum())
    return right, total


@pytest.mark.parametrize("case", ["two_units", "layers", "ball"])
def test_logs_reproduced_at_constraint_points(case, request):
    _, _, fields = request.getfixturevalue(case)
    assert all(f.neighbors is None for f in fields.fields)  # global fits are exact
    right, total = reproduction(fields)
    assert right == total and total > 500


def test_partition_on_a_grid_and_shared_mesh_boundaries(layers):
    """No gaps, no overlaps: the unit volumes on a grid sum to the grid volume, the closed
    unit meshes sum to the box volume (tolerance 0.2 %: slivers at tie nodes and the
    1/1000-cell offset of the caps), and neighbouring meshes share their vertices."""
    _, _, fields = layers
    lo = np.array([-90.0, -90.0, -190.0])
    grid = GridSpec(lo, (5.0, 5.0, 5.0), (37, 37, 37))
    nodes = grid.slab_points(0, 37).reshape(-1, 3)
    values = fields.values(nodes)
    labels = fields.labels_from_values(fields.node_values(nodes))
    assert (labels == np.argmax(values, axis=1)).all()
    counts = np.bincount(labels, minlength=3)
    assert counts.sum() == grid.n_nodes and (labels >= 0).all()
    cell = 5.0**3
    assert counts.sum() * cell == pytest.approx(grid.n_nodes * cell)

    node_margins = margins(values).T.reshape(3, 37, 37, 37)
    meshes = unit_meshes(node_margins, grid, fields.units)
    assert set(meshes) == {"A", "B", "C"}
    box = float(np.prod((np.array(grid.shape) - 1) * grid.spacing))
    total = 0.0
    for unit, mesh in meshes.items():
        report = mesh.validate()
        assert report.is_valid and report.is_outward, unit
        other = trimesh.Trimesh(mesh.vertices, mesh.faces, process=False)
        assert other.is_watertight
        assert mesh.volume == pytest.approx(float(other.volume), rel=1e-9)
        total += mesh.volume
    print(f"\npartition: mesh volumes sum to {total / box:.5f} of the box")
    assert total == pytest.approx(box, rel=2e-3)
    # each mesh volume against the count of nodes it wins (trapezoid weights)
    weight = np.ones(grid.shape)
    for axis in range(3):
        index = [slice(None)] * 3
        for end in (0, -1):
            index[axis] = end
            weight[tuple(index)] *= 0.5
    for i, unit in enumerate(fields.units):
        counted = float((weight.reshape(-1) * (labels == i)).sum() * cell)
        assert meshes[unit].volume == pytest.approx(counted, rel=0.01)

    # A and C never touch here; every vertex of A away from the box faces is a vertex of B
    def interior(mesh):
        v = mesh.vertices
        inner = ((v > lo + 0.01) & (v < lo + 180.0 - 0.01)).all(axis=1)
        return {tuple(row) for row in np.round(v[inner], 9)}

    a, b, c = interior(meshes["A"]), interior(meshes["B"]), interior(meshes["C"])
    assert a and c and a <= b and c <= b and not (a & c)
    assert b == a | c


def test_unit_meshes_clip_and_empty_unit():
    """A clip at a plane: the two half-box volumes are analytic."""
    grid = GridSpec((0.0, 0.0, 0.0), (1.0, 1.0, 1.0), (11, 11, 11))
    z = grid.slab_points(0, 11)[..., 2]
    f = np.stack([z - 4.5, 4.5 - z, np.full(z.shape, -50.0)])  # unit 0 above z = 4.5
    meshes = unit_meshes(
        margins(np.moveaxis(f, 0, -1)).transpose(3, 0, 1, 2), grid, ["UP", "LOW", "NONE"]
    )
    assert set(meshes) == {"UP", "LOW"}  # a unit that wins nowhere has no mesh
    assert meshes["UP"].volume == pytest.approx(100 * 5.5, rel=2e-3)
    assert meshes["LOW"].volume == pytest.approx(100 * 4.5, rel=2e-3)
    clipped = unit_meshes(
        margins(np.moveaxis(f, 0, -1)).transpose(3, 0, 1, 2),
        grid,
        ["UP", "LOW", "NONE"],
        clip=7.25 - z,
    )
    assert clipped["UP"].volume == pytest.approx(100 * 2.75, rel=2e-3)  # between 4.5 and 7.25
    assert clipped["LOW"].volume == pytest.approx(100 * 4.5, rel=2e-3)
    with pytest.raises(ValueError, match="shape"):
        unit_meshes(f[:2], grid, ["UP", "LOW", "NONE"])


# --------------------------------------------------------------------------
# the group: common scale, unknown units, errors
# --------------------------------------------------------------------------


def test_group_defaults_against_pandas(layers):
    """Cap = median run length, far spacing = median logged hole length / 10, near spacing
    = min(cap / 4, that): computed here from the table (the synthetic log has one interval
    per run), and identical for every unit."""
    table, traces, fields = layers
    length = table["depth_to"] - table["depth_from"]
    cap = float(length.median())
    far = float(length.groupby(table["hole_id"]).sum().median()) / 10.0
    assert fields.cap == pytest.approx(cap, rel=1e-12)
    assert fields.far_spacing == pytest.approx(far, rel=1e-12)
    assert fields.sample_spacing == pytest.approx(min(cap / 4.0, far), rel=1e-12)
    for c in fields.constraints:
        assert (c.cap, c.sample_spacing, c.far_spacing) == (
            fields.cap,
            fields.sample_spacing,
            fields.far_spacing,
        )
        assert np.abs(c.value).max() <= fields.cap + 1e-12
    assert fields.anisotropy is None and fields.kernel == "linear"
    assert 20.0 < fields.hole_spacing <= 40.0  # collars 40 m apart, a quarter of the holes inclined
    runs = group_runs(table, table["unit"], ["A", "B", "C"], traces)
    assert len(runs) == len(table)
    np.testing.assert_allclose(
        np.sort((runs["depth_to"] - runs["depth_from"]).to_numpy()), np.sort(length.to_numpy())
    )
    contacts = group_contacts(table, table["unit"], ["A", "B", "C"], traces)
    assert len(contacts) == len(table) - table["hole_id"].nunique()
    s = contacts[["x", "y", "z"]].to_numpy() @ PLANE_NORMAL
    expected = np.where(contacts["above"] == "A", LAYER_TOPS[0], LAYER_TOPS[1])
    np.testing.assert_allclose(s, expected, atol=1e-6)


def test_hole_spacing_of_parallel_holes(ball):
    _, _, fields = ball
    assert fields.hole_spacing == pytest.approx(25.0, abs=2.5)  # vertical holes 25 m apart


def test_settings_are_used_and_shared(layers):
    table, traces, _ = layers
    settings = FieldSettings(cap=12.0, sample_spacing=3.0, far_spacing=30.0, thin=None)
    fields = fit_unit_fields(table, table["unit"], ["A", "B", "C"], traces, settings)
    assert (fields.cap, fields.sample_spacing, fields.far_spacing) == (12.0, 3.0, 30.0)
    assert all(np.abs(c.value).max() == pytest.approx(12.0) for c in fields.constraints)
    assert fields.settings is settings
    with pytest.raises(ValueError, match="cap"):
        fit_unit_fields(table, table["unit"], ["A", "B", "C"], traces, FieldSettings(cap=-1.0))
    with pytest.raises(ValueError, match="kernel"):
        fit_unit_fields(table, table["unit"], ["A", "B", "C"], traces, FieldSettings(kernel="x"))
    with pytest.raises(ValueError, match="anisotropy"):
        fit_unit_fields(
            table, table["unit"], ["A", "B", "C"], traces, FieldSettings(anisotropy="plane")
        )


def test_units_outside_the_group_are_unknown(layers):
    """B left out of the group: its intervals give no points and hide the A/B and B/C
    contacts, so no contact exists at all and both fields are constant +-cap along holes."""
    table, traces, _ = layers
    fields = fit_unit_fields(table, table["unit"], ["A", "C"], traces)
    assert fields.units == ("A", "C")
    for c in fields.constraints:
        assert c.n_contacts == 0
        assert set(np.round(np.abs(c.value), 9)) == {round(fields.cap, 9)}
    middle = table["unit"] == "B"
    hole = table.loc[middle, "hole_id"].iloc[0]
    row = table[middle & (table["hole_id"] == hole)].iloc[0]
    xyz, _ = traces.xyz_at(hole, 0.5 * (row["depth_from"] + row["depth_to"]))
    near = [np.linalg.norm(c.xyz - xyz, axis=1).min() for c in fields.constraints]
    assert min(near) > 0.25 * (row["depth_to"] - row["depth_from"])
    # NaN is unknown as well, and the index of the units must be the table's
    units = table["unit"].where(table["unit"] != "B", np.nan)
    again = fit_unit_fields(table, units, ["A", "C"], traces)
    np.testing.assert_array_equal(again.constraints[0].xyz, fields.constraints[0].xyz)
    with pytest.raises(ValueError, match="index"):
        fit_unit_fields(table, table["unit"].reset_index(drop=True).iloc[::-1], ["A", "C"], traces)


def test_unit_without_a_run_raises(layers):
    table, traces, _ = layers
    with pytest.raises(ValueError, match="no logged run of unit.*ZZ"):
        fit_unit_fields(table, table["unit"], ["A", "B", "ZZ"], traces)
    with pytest.raises(ValueError, match="twice"):
        fit_unit_fields(table, table["unit"], ["A", "A"], traces)
    with pytest.raises(ValueError, match="no units"):
        fit_unit_fields(table, table["unit"], [], traces)


def test_anisotropy_given_and_automatic(layers):
    table, traces, isotropic = layers
    group = ["A", "B", "C"]
    auto = fit_unit_fields(table, table["unit"], group, traces, FieldSettings(anisotropy="auto"))
    assert isinstance(auto.anisotropy, Ellipsoid)
    pole = np.asarray(auto.anisotropy.minor_axis)
    angle = np.degrees(np.arccos(min(1.0, abs(float(pole @ PLANE_NORMAL)))))
    assert angle < 1.0  # the contacts are exact planes: the pole is recovered
    assert auto.anisotropy.minor_ratio == pytest.approx(AUTO_MAX_RATIO)
    assert all(f.anisotropy is auto.anisotropy for f in auto.fields)
    given = fit_unit_fields(
        table, table["unit"], group, traces, FieldSettings(anisotropy=auto.anisotropy)
    )
    pts = np.random.default_rng(5).uniform([-90, -90, -190], [90, 90, -10], size=(3000, 3))
    np.testing.assert_allclose(given.values(pts), auto.values(pts), atol=1e-9)
    truth = three_layers(pts)
    accuracy = [
        float((np.array(group, dtype=object)[f.labels(pts)] == truth).mean())
        for f in (isotropic, auto)
    ]
    print(
        f"\nthree layers: accuracy isotropic {accuracy[0]:.4f}, automatic plane {accuracy[1]:.4f}"
    )
    assert accuracy[1] >= accuracy[0] - 0.002  # the true plane does not make it worse
    sources = {
        e.source for f in (isotropic, auto, given) for e in f.provenance if "Anisotropy" in e.item
    }
    assert sources == {"none", "automatic", "given"}


def test_logged_hole_points(layers):
    table, traces, _ = layers
    points = logged_hole_points(table, traces, step=5.0)
    # every interval end is a point, and every interval midpoint is within step / 2 of one
    ends, _ = traces.xyz_at(table["hole_id"].to_numpy(), table["depth_to"].to_numpy())
    assert cKDTree(points).query(ends)[0].max() < 1e-9
    middle = (0.5 * (table["depth_from"] + table["depth_to"])).to_numpy()
    mids, _ = traces.xyz_at(table["hole_id"].to_numpy(), middle)
    assert cKDTree(points).query(mids)[0].max() <= 2.5 + 1e-9
    # only the intervals asked for: points of the B intervals lie in B or on its contacts
    only_b = logged_hole_points(table, traces, step=5.0, known=table["unit"] == "B")
    assert 0 < len(only_b) < len(points)
    s = only_b @ PLANE_NORMAL
    assert ((s <= LAYER_TOPS[0] + 1e-6) & (s >= LAYER_TOPS[1] - 1e-6)).all()
    with pytest.raises(ValueError, match="step"):
        logged_hole_points(table, traces, step=0.0)


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------


def test_provenance_entries(layers):
    table, _, fields = layers
    by_item = {e.item: e for e in fields.provenance}
    assert {e.kind for e in fields.provenance} == {"measured", "assumed"}
    assert all(e.stage == "lithology" and e.task == "P9.T7" for e in fields.provenance)
    length = table["depth_to"] - table["depth_from"]
    for unit in fields.units:
        entry = by_item[f"Rock-type log: unit {unit}"]
        mask = table["unit"] == unit
        assert entry.kind == "measured" and entry.count == int(mask.sum())
        assert f"{length[mask].sum():.0f} m" in entry.detail
        assert f"{table.loc[mask, 'hole_id'].nunique()} holes" in entry.detail
    group = by_item["Rock-type log: group A + B + C"]
    assert group.kind == "measured" and group.count == len(table)
    assumed = {e.item: e for e in fields.provenance if e.kind == "assumed"}
    assert assumed["Cap of the signed distance, group A + B + C"].value == pytest.approx(fields.cap)
    assert assumed["Cap of the signed distance, group A + B + C"].source.startswith("default")
    assert assumed["Point spacing near contacts, group A + B + C"].value == pytest.approx(
        fields.sample_spacing
    )
    assert assumed["Point spacing away from contacts, group A + B + C"].value == pytest.approx(
        fields.far_spacing
    )
    assert "argmax" in assumed["Modelling mode of the group A + B + C"].source
    assert assumed["Anisotropy of the group A + B + C"].source == "none"


# --------------------------------------------------------------------------
# stacks
# --------------------------------------------------------------------------


class Overprint:
    """A stand-in for an intrusion event: one unit inside a sphere."""

    mode = "intrusion"

    def __init__(self, unit, centre, radius):
        self.units = (unit,)
        self.provenance = []
        self.centre, self.radius = np.asarray(centre, dtype=float), float(radius)

    def node_values(self, points):
        return (self.radius - np.linalg.norm(np.asarray(points) - self.centre, axis=1))[:, None]

    def labels_from_values(self, values):
        v = np.asarray(values)[:, 0]
        return np.where(v > 0.0, 0, -1).astype(np.int16)


def test_stack_later_event_wins(layers):
    table, traces, fields = layers
    assert isinstance(fields, LithologyEvent)
    centre = np.array([0.0, 0.0, -100.0])
    stack = LithologyStack(
        "ALL", (fields, Overprint("X", centre, 40.0), Overprint("Y", centre, 15.0))
    )
    assert stack.categories == ("A", "B", "C", "X", "Y")
    assert stack.provenance == fields.provenance
    pts = np.random.default_rng(6).uniform([-90, -90, -190], [90, 90, -10], size=(5000, 3))
    got = np.array(stack.categories, dtype=object)[stack.labels(pts)]
    r = np.linalg.norm(pts - centre, axis=1)
    host = np.array(fields.units, dtype=object)[fields.labels(pts)]
    expected = np.where(r < 15.0, "Y", np.where(r < 40.0, "X", host))
    assert (got == expected).all()
    assert {"X", "Y"} <= set(got)
    with pytest.raises(ValueError, match="one array of values per event"):
        stack.labels_from_values([fields.node_values(pts)])


def test_fit_stack_dispatch_and_event_spec(layers):
    table, traces, fields = layers
    stack = fit_stack(
        table, table["unit"], traces, [EventSpec("argmax", ("A", "B", "C"))], block="B1"
    )
    assert stack.block == "B1" and stack.categories == ("A", "B", "C")
    pts = np.random.default_rng(7).uniform([-90, -90, -190], [90, 90, -10], size=(500, 3))
    np.testing.assert_array_equal(stack.labels(pts), fields.labels(pts))
    empty = LithologyStack("ALL", ())
    assert (empty.labels(pts) == -1).all()
    with pytest.raises(ValueError, match="mode"):
        EventSpec("largest", ("A",))
    with pytest.raises(ValueError, match="exactly one unit"):
        EventSpec("intrusion", ("A", "B"))
    with pytest.raises(ValueError, match="thickness"):
        EventSpec("argmax", ("A", "B"), thickness={"A": 1.0})
    with pytest.raises(ValueError, match="twice"):
        EventSpec("argmax", ("A", "A"))


# --------------------------------------------------------------------------
# the synthetic layered deposit, without fault handling (P9.T13) and with the
# intrusions as plain members of the group (overprints are P9.T10)
# --------------------------------------------------------------------------

GROUP = ("SST", "SLT", "AND", "DIO", "GRD")
#: loose regression floors, not the targets of geology_contracts.md section 14
IOU_FLOOR = {"SST": 0.85, "SLT": 0.80, "AND": 0.85, "DIO": 0.75}


@pytest.fixture(scope="module")
def layered_model(layered_small, layered_small_dir, layered_small_db, layered_small_traces):
    units = UnitDictionary.load(layered_small_dir / "units.json")
    assert units.codes(*MODELLED_KINDS) == GROUP
    table = layered_small_db.intervals["geology"]
    mapped = units.map_codes(table["lith"])
    t0 = time.perf_counter()
    fields = fit_unit_fields(table, mapped, GROUP, layered_small_traces)
    t1 = time.perf_counter()
    cfg = layered_small.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]], dtype=float)
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]], dtype=float)
    shape = tuple(int(n) for n in np.round((hi - lo) / 5.0) + 1)
    hole_points = logged_hole_points(
        table, layered_small_traces, step=5.0, known=mapped.isin(GROUP)
    )
    model = GeologyModel(
        name="layered-small, no faults",
        units=units,
        grid=GridSpec(lo, (5.0, 5.0, 5.0), shape),
        topography=DTM(read_obj(layered_small_dir / "topo.obj")),
        stacks={"ALL": LithologyStack("ALL", (fields,))},
        hole_points=hole_points,
    )
    t2 = time.perf_counter()
    model.sample_grid()
    t3 = time.perf_counter()
    seconds = {"fit": t1 - t0, "sample_grid": t3 - t2}
    return model, fields, (lo, hi), seconds


def test_layered_logs_reproduced(layered_model):
    """Section 14: logged units reproduced at the constraint points, 100 % (global fits)."""
    _, fields, _, _ = layered_model
    assert all(f.neighbors is None for f in fields.fields)
    right, total = reproduction(fields)
    print(f"\nlayered-small: logged units reproduced at {right} of {total} constraint points")
    assert right == total


def test_layered_grid_labels_against_exact(layered_model):
    """Section 14: labels with exact=False against exact=True, >= 99.5 % of points."""
    model, _, (lo, hi), _ = layered_model
    pts = np.random.default_rng(8).uniform(lo, hi, size=(20000, 3))
    t0 = time.perf_counter()
    exact = model.labels(pts, exact=True)
    t1 = time.perf_counter()
    fast = model.labels(pts)
    t2 = time.perf_counter()
    agreement = float((exact == fast).mean())
    print(
        f"\nlayered-small: grid labels equal exact labels at {agreement:.4%} of 20000 points "
        f"(exact {t1 - t0:.2f} s, grid {t2 - t1:.3f} s)"
    )
    assert agreement >= 0.995


def test_layered_iou_against_truth(layered_small, layered_model):
    """IoU per unit against the exact ``bedrock`` truth on the 2.5 m lattice inside the
    drilled hull, below the topography. Only loose floors are asserted here: the fault
    (P9.T13) and the intrusion overprints (P9.T10) are not handled yet, and the 8 m dyke
    is not expected to survive an isotropic largest-field model."""
    model, _, (lo, hi), seconds = layered_model
    pts = lattice(lo, hi, 2.5)
    hull = Delaunay(model.hole_points)
    pts = pts[hull.find_simplex(pts) >= 0]
    truth = layered_small.truth.labels(pts, "bedrock")
    t0 = time.perf_counter()
    got = model.decode(model.labels(pts), "lith")
    seconds["labels"] = time.perf_counter() - t0
    # the model's AIR is the triangulated topography, the truth's the analytic one
    assert ((got == "AIR") == (truth == "AIR")).mean() > 0.999
    ground = (truth != "AIR") & (got != "AIR")
    got, truth = got[ground], truth[ground]
    assert (got != None).all()  # noqa: E711  (every point below the topography has a unit)
    iou = {}
    for unit in GROUP:
        a, b = got == unit, truth == unit
        iou[unit] = float((a & b).sum() / (a | b).sum())
    accuracy = float((got == truth).mean())
    print(
        f"\nlayered-small, no fault handling, {len(got)} lattice points: accuracy {accuracy:.4f}; "
        + ", ".join(f"IoU {u} {v:.3f}" for u, v in iou.items())
        + f"; seconds { ({k: round(v, 2) for k, v in seconds.items()}) }"
    )
    for unit, floor in IOU_FLOOR.items():
        assert iou[unit] >= floor, (unit, iou[unit])
    assert accuracy >= 0.90


def test_layered_contact_distance(layered_small, layered_model):
    """SST/SLT contact against the analytic contact inside the drilled hull, on vertical
    lines 5 m apart where both the model and the truth have exactly one such contact.
    Reported; the section 14 target (<= 3 m RMS) is for the faulted model, so only a loose
    bound is asserted here."""
    model, _, (lo, hi), _ = layered_model
    truth = layered_small.truth
    xy = lattice(lo[:2], hi[:2], 5.0)
    z = np.arange(lo[2], hi[2] + 1e-9, 0.5)
    cols = np.column_stack([np.repeat(xy, len(z), axis=0), np.tile(z, len(xy))])
    got = model.labels(cols).reshape(len(xy), len(z))
    true = truth.labels(cols, "bedrock").reshape(len(xy), len(z))
    sst, slt = model.categories("lith").index("SST"), model.categories("lith").index("SLT")
    hull = Delaunay(model.hole_points)
    separation = []
    for i, (x, y) in enumerate(xy):
        k = np.flatnonzero((got[i, 1:] == sst) & (got[i, :-1] == slt))
        j = np.flatnonzero((true[i, 1:] == "SST") & (true[i, :-1] == "SLT"))
        if len(k) != 1 or len(j) != 1:
            continue
        # the analytic elevation, in the fault block the true contact lies in
        candidates = [float(truth.contact_z(x, y, "SST/SLT", b)) for b in ("FB1", "FB2")]
        z_true = min(candidates, key=lambda c: abs(c - (z[j[0]] + 0.25)))
        if abs(z_true - (z[j[0]] + 0.25)) > 0.3:
            continue  # SST lies on SLT across the fault here, not across the bedding contact
        if hull.find_simplex(np.array([x, y, z_true])) >= 0:
            separation.append(z[k[0]] + 0.25 - z_true)
    separation = np.array(separation)
    assert len(separation) > 1000
    distance = separation * truth.bedding_normal[2]  # vertical separation -> distance to the plane
    rms = float(np.sqrt(np.mean(distance**2)))
    print(
        f"\nlayered-small: SST/SLT contact, {len(distance)} columns: RMS distance {rms:.2f} m "
        f"(vertical {np.sqrt(np.mean(separation**2)):.2f} m, "
        f"largest {np.abs(distance).max():.1f} m), no fault handling"
    )
    assert rms <= 6.0
