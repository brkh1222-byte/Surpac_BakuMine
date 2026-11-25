"""Faults (P9.T8): planes, separating plane, imported surfaces, fault network.

Independent checks: plane fit against numpy SVD and a least-squares solve; the soft-margin
separating plane against scipy's trust-constr on the same objective and against hand cases; dip and
dip azimuth against hand-computed normals; the small layered deposit against its exact truth.
"""

import json
import math

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import Bounds, LinearConstraint, minimize

from bakumine.core.geology.faults import (
    FAULT_SOURCES,
    PICK_COLUMNS,
    FaultError,
    FaultNetwork,
    FaultSurface,
    Plane,
    block_codes,
    fault_picks,
    faults_provenance,
    fit_plane,
    import_fault,
    infer_fault,
    labelled_points,
    separating_plane,
)
from bakumine.core.geology.provenance import PROVENANCE_KINDS, Provenance
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geometry.mesh import Mesh

# geology_contracts.md section 14
DIP_TARGET = 5.0
AZIMUTH_TARGET = 10.0
POSITION_TARGET = 5.0


def azimuth_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def angle_between(n1, n2) -> float:
    return math.degrees(math.acos(min(1.0, abs(float(np.dot(n1, n2))))))


def plane_points(plane: Plane, rng, n: int, extent: float = 100.0, noise: float = 0.0):
    """Random points on a plane, with optional noise along its normal."""
    n3 = plane.normal
    u = np.cross(n3, [0.3, -0.5, 0.8])
    u /= np.linalg.norm(u)
    v = np.cross(n3, u)
    ab = rng.uniform(-extent, extent, (n, 2))
    points = plane.point + ab[:, :1] * u + ab[:, 1:] * v
    return points + noise * rng.standard_normal(n)[:, None] * n3


# --------------------------------------------------------------------------
# plane: dip, dip azimuth, sign convention
# --------------------------------------------------------------------------


def test_fault_sources_are_the_contract_list():
    assert FAULT_SOURCES == (
        "inferred-plane",
        "inferred-separating",
        "imported-mesh",
        "imported-points",
        "imported-plane",
    )


@pytest.mark.parametrize(
    ("normal", "dip", "azimuth"),
    [
        ((0.0, 0.0, 1.0), 0.0, 0.0),  # horizontal: azimuth is 0 by convention
        ((1.0, 0.0, 1.0), 45.0, 90.0),  # dips 45 to the east
        ((0.0, 1.0, 1.0), 45.0, 0.0),  # to the north
        ((-1.0, 0.0, 1.0), 45.0, 270.0),  # to the west
        ((0.0, -1.0, math.sqrt(3.0)), 30.0, 180.0),  # 30 to the south
        ((1.0, 1.0, math.sqrt(2.0)), 45.0, 45.0),
        ((0.0, -math.sqrt(3.0), 1.0), 60.0, 180.0),
    ],
)
def test_dip_and_dip_azimuth_from_hand_normals(normal, dip, azimuth):
    plane = Plane((10.0, 20.0, 30.0), normal)
    assert plane.dip == pytest.approx(dip, abs=1e-9)
    assert plane.dip_azimuth == pytest.approx(azimuth, abs=1e-9)
    assert np.linalg.norm(plane.normal) == pytest.approx(1.0, abs=1e-15)
    again = Plane.from_dip(plane.point, dip, azimuth)
    np.testing.assert_allclose(again.normal, plane.normal, atol=1e-12)


def test_normal_is_turned_up():
    down = Plane((0.0, 0.0, 0.0), (-1.0, 0.0, -1.0))  # the same plane as normal (1, 0, 1)
    np.testing.assert_allclose(down.normal, np.array([1.0, 0.0, 1.0]) / math.sqrt(2.0))
    assert down.dip == pytest.approx(45.0)
    assert down.dip_azimuth == pytest.approx(90.0)
    # the positive side is the upper side (hanging wall)
    assert down.signed_distance([0.0, 0.0, 5.0]) > 0.0
    assert down.signed_distance([0.0, 0.0, -5.0]) < 0.0


@pytest.mark.parametrize(
    ("normal", "expected", "azimuth"),
    [
        ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90.0),
        ((-1.0, 0.0, 0.0), (1.0, 0.0, 0.0), 90.0),  # azimuth 270 becomes 90
        ((0.0, -1.0, 0.0), (0.0, 1.0, 0.0), 0.0),  # azimuth 180 becomes 0
        ((0.0, 1.0, 0.0), (0.0, 1.0, 0.0), 0.0),
        ((-1.0, -1.0, 0.0), (math.sqrt(0.5), math.sqrt(0.5), 0.0), 45.0),
        ((-1.0, 1.0, 0.0), (math.sqrt(0.5), -math.sqrt(0.5), 0.0), 135.0),
    ],
)
def test_vertical_plane_normal_points_to_azimuth_below_180(normal, expected, azimuth):
    plane = Plane((0.0, 0.0, 0.0), normal)
    np.testing.assert_allclose(plane.normal, expected, atol=1e-15)
    assert plane.dip == pytest.approx(90.0)
    assert plane.dip_azimuth == pytest.approx(azimuth)
    assert 0.0 <= plane.dip_azimuth < 180.0


def test_from_dip_vertical_and_horizontal():
    vertical = Plane.from_dip((0.0, 0.0, 0.0), 90.0, 250.0)  # turned to azimuth 70
    assert vertical.dip == pytest.approx(90.0)
    assert vertical.dip_azimuth == pytest.approx(70.0)
    assert vertical.normal[2] == 0.0
    flat = Plane.from_dip((0.0, 0.0, 7.0), 0.0, 123.0)
    np.testing.assert_allclose(flat.normal, [0.0, 0.0, 1.0], atol=1e-15)
    assert flat.dip_azimuth == 0.0
    assert flat.signed_distance([[5.0, 5.0, 9.0], [1.0, 1.0, 4.0]]) == pytest.approx([2.0, -3.0])


def test_from_dip_matches_the_formula_of_the_contract():
    # n = (sin dip sin az, sin dip cos az, cos dip), geology_contracts.md 13
    plane = Plane.from_dip((1.0, 2.0, 3.0), 65.0, 70.0)
    d, a = math.radians(65.0), math.radians(70.0)
    expected = [math.sin(d) * math.sin(a), math.sin(d) * math.cos(a), math.cos(d)]
    np.testing.assert_allclose(plane.normal, expected, atol=1e-15)
    assert plane.dip == pytest.approx(65.0, abs=1e-10)
    assert plane.dip_azimuth == pytest.approx(70.0, abs=1e-10)


def test_plane_rejects_bad_input():
    with pytest.raises(FaultError):
        Plane((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    with pytest.raises(FaultError):
        Plane((0.0, np.nan, 0.0), (0.0, 0.0, 1.0))
    with pytest.raises(FaultError):
        Plane.from_dip((0.0, 0.0, 0.0), 95.0, 10.0)


# --------------------------------------------------------------------------
# plane fit
# --------------------------------------------------------------------------


def test_fit_plane_through_exact_points():
    truth = Plane.from_dip((500_000.0, 5_300_000.0, 1_000.0), 65.0, 70.0)
    points = plane_points(truth, np.random.default_rng(1), 25)
    fitted = fit_plane(points)
    assert np.abs(fitted.signed_distance(points)).max() < 1e-9
    assert np.abs(truth.signed_distance(fitted.point)) < 1e-9
    assert angle_between(fitted.normal, truth.normal) < 1e-9
    # second implementation: z = a + b x + c y by least squares (local coordinates)
    q = points - points.mean(axis=0)
    coefficients, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(q)), q[:, :2]]), q[:, 2])
    normal = np.array([-coefficients[1], -coefficients[2], 1.0])
    assert angle_between(fitted.normal, normal / np.linalg.norm(normal)) < 1e-9


def test_fit_plane_three_points_by_hand():
    fitted = fit_plane([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 10.0]])  # z = y
    np.testing.assert_allclose(fitted.normal, [0.0, -math.sqrt(0.5), math.sqrt(0.5)], atol=1e-12)
    assert fitted.dip == pytest.approx(45.0)
    assert fitted.dip_azimuth == pytest.approx(180.0)


@pytest.mark.parametrize("weighted", [False, True])
def test_fit_plane_noisy_against_numpy_svd(weighted):
    rng = np.random.default_rng(7)
    truth = Plane.from_dip((500_300.0, 5_300_200.0, 1_050.0), 55.0, 310.0)
    points = plane_points(truth, rng, 200, noise=2.0)
    weights = rng.uniform(0.2, 4.0, len(points)) if weighted else None
    fitted = fit_plane(points, weights)

    w = np.ones(len(points)) if weights is None else weights
    centre = (w[:, None] * points).sum(axis=0) / w.sum()
    _, singular, vt = np.linalg.svd(np.sqrt(w)[:, None] * (points - centre), full_matrices=False)
    assert angle_between(fitted.normal, vt[-1]) < 1e-7
    np.testing.assert_allclose(fitted.point, centre, atol=1e-6)
    # the weighted sum of squared distances is the smallest singular value squared
    cost = (w * fitted.signed_distance(points) ** 2).sum()
    assert cost == pytest.approx(singular[-1] ** 2, rel=1e-9)
    # and close to the truth
    assert angle_between(fitted.normal, truth.normal) < 0.5


def test_fit_plane_with_prior_fits_the_position_only():
    rng = np.random.default_rng(3)
    truth = Plane.from_dip((100.0, 200.0, 50.0), 40.0, 120.0)
    points = plane_points(truth, rng, 30, noise=1.0)
    weights = rng.uniform(0.5, 2.0, 30)
    for prior in ((50.0, 100.0), Plane.from_dip((0.0, 0.0, 0.0), 50.0, 100.0)):
        fitted = fit_plane(points, weights, prior=prior)
        assert fitted.dip == pytest.approx(50.0)
        assert fitted.dip_azimuth == pytest.approx(100.0)
        # c . n0 = sum w (p . n0) / sum w
        expected = (weights * (points @ fitted.normal)).sum() / weights.sum()
        assert fitted.point @ fitted.normal == pytest.approx(expected, abs=1e-9)
    one = fit_plane([[1.0, 2.0, 3.0]], prior=(30.0, 0.0))
    assert one.signed_distance([1.0, 2.0, 3.0]) == pytest.approx(0.0)


def test_fit_plane_degenerate_inputs_raise():
    with pytest.raises(FaultError, match="at least 3 points"):
        fit_plane([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    line = np.outer(np.linspace(0.0, 100.0, 6), [1.0, 2.0, -0.5]) + [500_000.0, 5_300_000.0, 900.0]
    with pytest.raises(FaultError, match="one line"):
        fit_plane(line)
    with pytest.raises(FaultError, match="one line"):
        fit_plane(np.tile([1.0, 2.0, 3.0], (5, 1)))
    with pytest.raises(FaultError):
        fit_plane([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, np.nan]])
    with pytest.raises(FaultError, match="weights"):
        fit_plane(np.eye(3), [1.0, 0.0, 1.0])


# --------------------------------------------------------------------------
# separating plane
# --------------------------------------------------------------------------


def svm_objective(w, beta, x, y, penalty):
    return 0.5 * float(w @ w) + penalty * float(np.maximum(0.0, 1.0 - y * (x @ w + beta)).sum())


def scipy_soft_margin(x, y, penalty):
    """The same problem with slack variables, by scipy trust-constr: variables (w, beta, xi)."""
    n = len(y)
    hessian = np.zeros((n + 4, n + 4))
    hessian[:3, :3] = np.eye(3)
    linear = np.concatenate([np.zeros(4), np.full(n, penalty)])
    result = minimize(
        lambda v: 0.5 * float(v[:3] @ v[:3]) + penalty * float(v[4:].sum()),
        np.concatenate([np.zeros(4), np.full(n, 2.0)]),
        jac=lambda v: hessian @ v + linear,
        hess=lambda v: hessian,
        method="trust-constr",
        constraints=[
            LinearConstraint(np.column_stack([y[:, None] * x, y, np.eye(n)]), 1.0, np.inf)
        ],
        bounds=Bounds(np.concatenate([np.full(4, -np.inf), np.zeros(n)]), np.full(n + 4, np.inf)),
        options={"gtol": 1e-10, "xtol": 1e-14, "barrier_tol": 1e-12, "maxiter": 5000},
    )
    assert result.status in (1, 2), result.message
    return result.x[:3], result.x[3], result.fun


def test_separating_plane_two_points_by_hand():
    plane, info = separating_plane([[0.0, 0.0, 0.0]], [[2.0, 0.0, 0.0]])
    np.testing.assert_allclose(plane.normal, [1.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(plane.point, [1.0, 0.0, 0.0], atol=1e-8)
    assert info["margin"] == pytest.approx(1.0, abs=1e-7)
    assert info["separable"] and info["n_wrong"] == 0 and not info["flipped"]
    assert info["converged"]


def test_separating_plane_hand_case_with_three_support_points():
    # a triangle in the plane x = 0 against the point (4, 2, 2), which is 4 m from the inside
    # of the triangle: the nearest points of the two hulls are (0, 2, 2) and (4, 2, 2), so the
    # plane is x = 2 with a margin of 2 m, and three points of the triangle support it
    neg = [
        [0.0, 0.0, 0.0],
        [0.0, 10.0, 0.0],
        [0.0, 0.0, 10.0],
        [-3.0, 4.0, -2.0],
        [-8.0, -6.0, 1.0],
    ]
    pos = [[4.0, 2.0, 2.0], [9.0, 0.0, 0.0], [6.0, 8.0, -3.0]]
    # a large penalty makes the margin hard; with the default 10 (in coordinates divided by
    # the extent) a gap this narrow against the extent is traded for a few margin violations
    plane, info = separating_plane(neg, pos, penalty=1e6)
    assert info["converged"] and info["separable"]
    assert angle_between(plane.normal, [1.0, 0.0, 0.0]) < 1e-5
    assert plane.signed_distance([2.0, 0.0, 0.0]) == pytest.approx(0.0, abs=1e-5)
    assert info["margin"] == pytest.approx(2.0, abs=1e-5)
    soft, soft_info = separating_plane(neg, pos)
    assert soft_info["separable"] and soft_info["margin"] > 2.0


def test_separating_plane_reports_a_flip_when_the_positive_cloud_is_below():
    low = [[0.0, 0.0, 0.0], [5.0, 5.0, 1.0]]
    high = [[0.0, 0.0, 10.0], [5.0, 5.0, 11.0]]
    plane, info = separating_plane(high, low)  # the "positive" points are the lower ones
    assert plane.normal[2] > 0.0
    assert info["flipped"]
    assert (plane.signed_distance(low) < 0.0).all() and (plane.signed_distance(high) > 0.0).all()
    assert info["n_wrong"] == 0
    _, other = separating_plane(low, high)
    assert not other["flipped"]


def test_separating_plane_of_clouds_symmetric_about_a_known_plane():
    rng = np.random.default_rng(11)
    truth = Plane.from_dip((500_400.0, 5_300_400.0, 1_100.0), 65.0, 70.0)
    n = truth.normal
    below = plane_points(truth, rng, 80, extent=150.0) - rng.uniform(5.0, 60.0, 80)[:, None] * n
    above = below + 2.0 * (-truth.signed_distance(below))[:, None] * n  # mirror images
    for penalty in (10.0, 1e6):
        plane, info = separating_plane(below, above, penalty=penalty)
        assert info["converged"] and info["separable"]
        assert angle_between(plane.normal, n) < 1e-4
        assert abs(truth.signed_distance(plane.point)) < 1e-3
    # hard margin: the plane sits half-way between the nearest mirror pair
    assert info["margin"] == pytest.approx(np.abs(truth.signed_distance(below)).min(), rel=1e-5)


@pytest.mark.parametrize(("overlap", "penalty"), [(False, 10.0), (True, 10.0), (True, 100.0)])
def test_separating_plane_against_scipy_on_the_same_objective(overlap, penalty):
    rng = np.random.default_rng(5 if overlap else 6)
    truth = Plane.from_dip((500_000.0, 5_300_000.0, 1_000.0), 50.0, 200.0)
    base = plane_points(truth, rng, 60, extent=80.0)
    distance = rng.uniform(-6.0, 40.0, 60) if overlap else rng.uniform(3.0, 40.0, 60)
    neg = base[:30] - distance[:30, None] * truth.normal
    pos = base[30:] + distance[30:, None] * truth.normal
    plane, info = separating_plane(neg, pos, penalty=penalty)
    assert info["converged"]

    both = np.concatenate([neg, pos])
    centre, scale = both.mean(axis=0), np.ptp(both, axis=0).max()
    np.testing.assert_allclose(info["centre"], centre)
    assert info["scale"] == pytest.approx(scale)
    x = (both - centre) / scale
    y = np.concatenate([-np.ones(30), np.ones(30)])
    w, beta = np.array(info["w"]), info["beta"]
    assert info["objective"] == pytest.approx(svm_objective(w, beta, x, y, penalty), rel=1e-12)

    w_ref, beta_ref, cost_ref = scipy_soft_margin(x, y, penalty)
    assert info["objective"] == pytest.approx(cost_ref, rel=1e-5)
    assert info["objective"] <= cost_ref * (1.0 + 1e-9)  # the interior-point value is from above
    np.testing.assert_allclose(w, w_ref, atol=1e-4 * np.linalg.norm(w_ref))
    assert beta == pytest.approx(beta_ref, abs=1e-4)
    reference = Plane(centre - beta_ref * scale * w_ref / (w_ref @ w_ref), w_ref)
    assert angle_between(plane.normal, reference.normal) < 0.02
    assert abs(reference.signed_distance(plane.point)) < 0.02
    if overlap:
        assert info["n_wrong"] > 0 and not info["separable"]
        wrong = int(
            (plane.signed_distance(neg) > 0).sum() + (plane.signed_distance(pos) <= 0).sum()
        )
        assert info["n_wrong"] == wrong == len(info["wrong_neg"]) + len(info["wrong_pos"])


def test_separating_plane_with_prior_normal_takes_the_middle_of_the_gap():
    normal = Plane.from_dip((0.0, 0.0, 0.0), 60.0, 90.0).normal
    rng = np.random.default_rng(2)
    side = rng.uniform(-50.0, 50.0, (40, 3))
    side -= np.outer(side @ normal, normal)
    neg = side[:20] + np.outer(rng.uniform(-30.0, -4.0, 20), normal)
    pos = side[20:] + np.outer(rng.uniform(6.0, 30.0, 20), normal)
    neg[0] = side[0] - 4.0 * normal
    pos[0] = side[20] + 6.0 * normal
    plane, info = separating_plane(neg, pos, prior_normal=normal)
    np.testing.assert_allclose(plane.normal, normal, atol=1e-12)
    assert plane.point @ normal == pytest.approx(1.0, abs=1e-9)  # middle of -4 and +6
    assert info["method"] == "prior-normal" and info["separable"]
    assert info["margin"] == pytest.approx(5.0)
    # the prior as (dip, dip azimuth), and the clouds given the other way round
    swapped, other = separating_plane(pos, neg, prior_normal=(60.0, 90.0))
    assert swapped.point @ normal == pytest.approx(1.0, abs=1e-9)
    assert other["flipped"] and other["n_wrong"] == 0


def test_separating_plane_with_prior_normal_fewest_wrong_points():
    normal = np.array([0.0, 0.0, 1.0])
    neg = np.array([[0.0, 0.0, z] for z in (-9.0, -7.0, -5.0, -3.0, 2.0, 5.0)])
    pos = np.array([[1.0, 1.0, z] for z in (4.0, 6.0, 8.0, -4.0)])
    plane, info = separating_plane(neg, pos, prior_normal=normal)
    # brute force over the offset
    offsets = np.linspace(-10.0, 10.0, 4001)
    wrong = [(neg[:, 2] > t).sum() + (pos[:, 2] <= t).sum() for t in offsets]
    assert info["n_wrong"] == min(wrong) == 2
    best = offsets[np.array(wrong) == min(wrong)]
    # two wrong points is the least, reached on (2, 4) and on (5, 6): the wider range is
    # taken, and its middle is 3
    assert best.min() == pytest.approx(2.0, abs=0.01) and best.max() == pytest.approx(6.0, abs=0.01)
    assert not ((best > 4.0) & (best < 5.0)).any()
    assert plane.point[2] == pytest.approx(3.0)
    assert info["wrong_neg"] == [5] and info["wrong_pos"] == [3]


def test_separating_plane_degenerate_inputs_raise():
    with pytest.raises(FaultError, match="one block only"):
        separating_plane(np.zeros((0, 3)), [[1.0, 2.0, 3.0]])
    with pytest.raises(FaultError, match="one block only"):
        separating_plane([[1.0, 2.0, 3.0]], [])
    with pytest.raises(FaultError, match="coincide"):
        separating_plane([[1.0, 2.0, 3.0]], [[1.0, 2.0, 3.0]])
    with pytest.raises(FaultError, match="prior normal"):
        separating_plane([[0.0, 0.0, 1.0]], [[5.0, 0.0, 1.0]], prior_normal=(0.0, 0.0, 1.0))
    same = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    with pytest.raises(FaultError, match="mixed"):
        separating_plane(same, same)


# --------------------------------------------------------------------------
# logs: block codes, labelled points, picks (hand-made straight holes)
# --------------------------------------------------------------------------


class StraightHoles:
    """Vertical holes: ``xyz_at`` as HoleTraces gives it, for hand-made tables."""

    def __init__(self, collars: dict[str, tuple[float, float, float]], length: float = 200.0):
        self.collars, self.length = collars, length

    def xyz_at(self, hole_id, depths):
        holes = np.asarray(hole_id, dtype=object).reshape(-1)
        depths = np.asarray(depths, dtype=np.float64).reshape(-1)
        xyz = np.full((len(depths), 3), np.nan)
        for k, (hole, depth) in enumerate(zip(holes, depths, strict=True)):
            if hole in self.collars and 0.0 <= depth <= self.length:
                x, y, z = self.collars[hole]
                xyz[k] = (x, y, z - depth)
        direction = np.tile([0.0, 0.0, -1.0], (len(depths), 1))
        return xyz, direction


def interval_table(rows):
    return pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "code"])


def test_block_codes_takes_the_first_group():
    values = pd.Series(
        ["FB1-ORE", "FB2-ORE", "WST", None, np.nan, "XFB1-ORE", 7], index=list("abcdefg")
    )
    out = block_codes(values, r"^(FB\d)-")
    assert list(out.index) == list("abcdefg")
    assert out.tolist()[:2] == ["FB1", "FB2"]
    assert out.iloc[2:].isna().all()
    with pytest.raises(FaultError, match="one group"):
        block_codes(values, r"^FB\d-")
    with pytest.raises(FaultError, match="regular expression"):
        block_codes(values, r"^(FB")


def test_labelled_points_are_interval_midpoints():
    traces = StraightHoles({"A": (0.0, 0.0, 100.0), "B": (50.0, 0.0, 100.0)})
    table = interval_table(
        [("A", 0.0, 10.0, "N-x"), ("A", 10.0, 20.0, "zz"), ("B", 30.0, 40.0, "S-x"),
         ("B", 190.0, 300.0, "S-x"), ("C", 0.0, 5.0, "N-x")]
    )  # fmt: skip
    xyz, block, hole = labelled_points(table, block_codes(table["code"], r"^(\w)-"), traces)
    np.testing.assert_allclose(xyz, [[0.0, 0.0, 95.0], [50.0, 0.0, 65.0]])
    assert block.tolist() == ["N", "S"] and hole.tolist() == ["A", "B"]


def test_fault_picks_gap_middle_and_fault_zone():
    traces = StraightHoles(
        {"A": (0.0, 0.0, 100.0), "B": (50.0, 0.0, 100.0), "C": (90.0, 0.0, 100.0)}
    )
    table = interval_table(
        [
            ("A", 0.0, 10.0, "N-x"), ("A", 10.0, 20.0, "N-x"), ("A", 20.0, 26.0, "w"),
            ("A", 26.0, 30.0, "S-x"),                      # gap 20 - 26: middle 23, s = 3
            ("B", 5.0, 6.0, "N-x"), ("B", 6.0, 7.0, "S-x"),  # gap of zero length at 6
            ("B", 50.0, 60.0, "S-x"),
            ("C", 0.0, 10.0, "N-x"), ("C", 40.0, 50.0, "S-x"),  # gap 10 - 40
        ]
    )  # fmt: skip
    blocks = block_codes(table["code"], r"^(\w)-")

    plain = fault_picks(table, blocks, traces)
    assert tuple(plain.columns) == PICK_COLUMNS
    assert plain["hole_id"].tolist() == ["A", "B", "C"]
    assert plain["source"].tolist() == ["gap-middle"] * 3
    assert plain["depth"].tolist() == [23.0, 6.0, 25.0]
    assert plain["uncertainty"].tolist() == [3.0, 0.0, 15.0]
    # w = 1 / max(s, s_min)^2 with s_min = 0.5
    np.testing.assert_allclose(plain["weight"], [1.0 / 9.0, 4.0, 1.0 / 225.0])
    np.testing.assert_allclose(plain[["x", "y", "z"]], [[0, 0, 77.0], [50, 0, 94.0], [90, 0, 75.0]])
    assert plain["block_above"].tolist() == ["N"] * 3 and plain["block_below"].tolist() == ["S"] * 3
    assert plain["gap_from"].tolist() == [20.0, 6.0, 10.0]
    assert plain["gap_to"].tolist() == [26.0, 6.0, 40.0]

    zones = pd.DataFrame(
        [("C", 30.0, 34.0), ("C", 34.0, 36.0), ("C", 70.0, 72.0), ("D", 5.0, 7.0), ("B", 5.5, 6.5)],
        columns=["hole_id", "depth_from", "depth_to"],
    )
    picks = fault_picks(table, blocks, traces, fault_zone=zones, min_uncertainty=1.0)
    by = {(r.hole_id, r.depth): r for r in picks.itertuples()}
    assert set(by) == {("A", 23.0), ("B", 6.0), ("C", 33.0), ("C", 71.0)}  # D has no trace
    # the two touching zone intervals of C are one zone 30 - 36 inside the gap
    assert by[("C", 33.0)].source == "fault-zone" and by[("C", 33.0)].uncertainty == 3.0
    assert by[("C", 33.0)].weight == pytest.approx(1.0 / 9.0)
    assert by[("B", 6.0)].source == "fault-zone" and by[("B", 6.0)].uncertainty == 0.5
    assert by[("B", 6.0)].weight == pytest.approx(1.0)  # s_min = 1
    assert by[("A", 23.0)].source == "gap-middle"
    # a fault zone that belongs to no block change: the nearest known blocks are named
    lone = by[("C", 71.0)]
    assert lone.source == "fault-zone" and lone.block_above == "S" and pd.isna(lone.block_below)
    assert lone.gap_from == 50.0 and math.isnan(lone.gap_to)

    strict = fault_picks(table, blocks, traces, fault_zone=zones, include_unlabelled=False)
    assert sorted(zip(strict["hole_id"], strict["depth"], strict=True)) == [
        ("A", 23.0), ("B", 6.0), ("C", 33.0)
    ]  # fmt: skip
    empty = fault_picks(table.iloc[:2], blocks.iloc[:2], traces)
    assert len(empty) == 0 and tuple(empty.columns) == PICK_COLUMNS


# --------------------------------------------------------------------------
# infer_fault on hand cases
# --------------------------------------------------------------------------


def picks_frame(points, weights=None, above=None, below=None):
    points = np.asarray(points, dtype=np.float64)
    n = len(points)
    return pd.DataFrame(
        {
            "hole_id": [f"H{k}" for k in range(n)],
            "block_above": above if above is not None else [None] * n,
            "block_below": below if below is not None else [None] * n,
            "x": points[:, 0],
            "y": points[:, 1],
            "z": points[:, 2],
            "weight": np.ones(n) if weights is None else weights,
            "source": ["fault-zone"] * n,
        }
    )


def two_clouds(truth: Plane, rng, n: int = 40):
    base = plane_points(truth, rng, 2 * n, extent=120.0)
    distance = rng.uniform(5.0, 50.0, 2 * n)
    return (
        base[:n] - distance[:n, None] * truth.normal,
        base[n:] + distance[n:, None] * truth.normal,
    )


def test_infer_fault_from_picks_orders_the_blocks_by_side():
    rng = np.random.default_rng(21)
    truth = Plane.from_dip((1000.0, 2000.0, 300.0), 70.0, 250.0)
    below, above = two_clouds(truth, rng)
    picks = picks_frame(plane_points(truth, rng, 8))
    fault = infer_fault("F9", picks, above, below, blocks=("UP", "DOWN"))
    assert fault.source == "inferred-plane" and fault.inferred
    assert fault.blocks == ("DOWN", "UP")  # (negative side, positive side)
    assert angle_between(fault.plane.normal, truth.normal) < 1e-6
    assert fault.misfit["n_wrong"] == 0 and fault.misfit["n_picks"] == 8
    assert fault.misfit["pick_rms"] < 1e-9
    assert (fault.side(above) == 1).all() and (fault.side(below) == -1).all()
    assert fault.side(above).dtype == np.int8
    np.testing.assert_allclose(fault.offset(above), truth.signed_distance(above), atol=1e-6)


def test_infer_fault_ignores_picks_of_other_blocks():
    rng = np.random.default_rng(22)
    truth = Plane.from_dip((0.0, 0.0, 0.0), 30.0, 10.0)
    below, above = two_clouds(truth, rng)
    good = plane_points(truth, rng, 6)
    stray = good[:3] + 500.0
    picks = picks_frame(
        np.concatenate([good, stray]),
        above=["A", None, "B", None, "A", None, "A", "C", "B"],
        below=["B", "B", "A", None, None, None, "A", None, "C"],
    )
    fault = infer_fault("F1", picks, below, above, blocks=("A", "B"))
    assert fault.misfit["n_picks"] == 6
    assert angle_between(fault.plane.normal, truth.normal) < 1e-6
    assert fault.blocks == ("A", "B")


def test_infer_fault_falls_back_to_the_separating_plane():
    rng = np.random.default_rng(23)
    truth = Plane.from_dip((0.0, 0.0, 0.0), 80.0, 90.0)
    below, above = two_clouds(truth, rng, 60)
    two = picks_frame(plane_points(truth, rng, 2))
    line = picks_frame(np.outer(np.linspace(-50.0, 50.0, 5), [0.0, 1.0, 0.0]))  # on one line
    for picks in (None, two, line):
        fault = infer_fault("F1", picks, below, above, blocks=("A", "B"))
        assert fault.source == "inferred-separating"
        assert fault.misfit["method"] == "separating plane"
        assert fault.misfit["n_wrong"] == 0 and fault.blocks == ("A", "B")
        reference, _ = separating_plane(below, above)
        np.testing.assert_allclose(fault.plane.normal, reference.normal, atol=1e-12)


def test_infer_fault_with_a_prior_orientation():
    rng = np.random.default_rng(24)
    truth = Plane.from_dip((10.0, 20.0, 30.0), 60.0, 135.0)
    below, above = two_clouds(truth, rng)
    one_pick = picks_frame(plane_points(truth, rng, 1))
    with_pick = infer_fault("F1", one_pick, below, above, blocks=("A", "B"), prior=(60.0, 135.0))
    assert with_pick.source == "inferred-plane"
    assert abs(truth.signed_distance(with_pick.plane.point)) < 1e-9
    without = infer_fault("F1", None, below, above, blocks=("A", "B"), prior=(60.0, 135.0))
    assert without.source == "inferred-separating"
    gap_middle = 0.5 * (truth.signed_distance(below).max() + truth.signed_distance(above).min())
    assert truth.signed_distance(without.plane.point) == pytest.approx(gap_middle, abs=1e-9)
    for fault in (with_pick, without):
        assert fault.plane.dip == pytest.approx(60.0) and fault.plane.dip_azimuth == pytest.approx(
            135.0
        )
        kinds = {(e.item, e.kind) for e in fault.provenance}
        assert ("Prior orientation of fault F1", "assumed") in kinds
        assert ("Fault F1", "assumed") in kinds


def test_infer_fault_misfit_by_hole():
    truth = Plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
    below = np.array([[0.0, 0.0, -5.0], [10.0, 0.0, -6.0], [20.0, 0.0, 3.0], [20.0, 5.0, 4.0]])
    above = np.array([[0.0, 5.0, 5.0], [10.0, 5.0, 6.0], [30.0, 0.0, -2.0]])
    picks = picks_frame([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 10.0, 0.0]])
    fault = infer_fault(
        "F1", picks, below, above, blocks=("LOW", "HIGH"),
        holes_a=["h1", "h2", "h3", "h3"], holes_b=["h1", "h2", "h4"],
    )  # fmt: skip
    assert angle_between(fault.plane.normal, truth.normal) < 1e-9
    assert fault.misfit["n_wrong"] == 3
    assert fault.misfit["holes_wrong"] == {"h3": 2, "h4": 1}
    assert fault.misfit["wrong_by_block"] == {"LOW": 2, "HIGH": 1}
    entry = fault.provenance[0]
    assert "3 points on the wrong side in 2 holes" in entry.detail
    assert "dip 0" in entry.detail


def test_infer_fault_degenerate_inputs_raise():
    rng = np.random.default_rng(25)
    truth = Plane.from_dip((0.0, 0.0, 0.0), 45.0, 0.0)
    below, above = two_clouds(truth, rng, 10)
    none = np.zeros((0, 3))
    two = picks_frame(plane_points(truth, rng, 2))
    line = picks_frame(np.outer(np.linspace(0.0, 90.0, 4), [1.0, 1.0, 0.0]))
    # one block only: no separating plane, and too few / collinear crossings for a plane fit
    for picks, text in (
        (None, "0 usable crossings"),
        (two, "2 usable crossings"),
        (line, "4 usable"),
    ):
        with pytest.raises(FaultError, match="one block only") as caught:
            infer_fault("F1", picks, below, none, blocks=("A", "B"))
        assert text in str(caught.value)
    # no labelled point at all: the sides cannot be told, even with good picks
    good = picks_frame(plane_points(truth, rng, 6))
    with pytest.raises(FaultError, match="sides of the fault"):
        infer_fault("F1", good, none, none, blocks=("A", "B"))
    with pytest.raises(FaultError, match="must differ"):
        infer_fault("F1", good, below, above, blocks=("A", "A"))
    # one block only is enough once the crossings fix the plane
    fault = infer_fault("F1", good, below, none, blocks=("A", "B"))
    assert fault.blocks == ("A", "B") and fault.source == "inferred-plane"


# --------------------------------------------------------------------------
# imported faults, mesh frame, display mesh, files
# --------------------------------------------------------------------------


def rectangle_mesh(plane: Plane, half: float, n: int = 4, bump=None) -> Mesh:
    """A square grid mesh in a plane, optionally bent by ``bump(a, b)`` along the normal."""
    normal = plane.normal
    u = np.cross([0.0, 0.0, 1.0], normal)
    u = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u) < 1e-9 else u / np.linalg.norm(u)
    v = np.cross(normal, u)
    ticks = np.linspace(-half, half, n + 1)
    a, b = np.meshgrid(ticks, ticks, indexing="ij")
    height = np.zeros_like(a) if bump is None else bump(a, b)
    vertices = plane.point + a[..., None] * u + b[..., None] * v + height[..., None] * normal
    index = np.arange((n + 1) ** 2).reshape(n + 1, n + 1)
    quads = np.stack([index[:-1, :-1], index[1:, :-1], index[1:, 1:], index[:-1, 1:]], axis=-1)
    quads = quads.reshape(-1, 4)
    faces = np.concatenate([quads[:, [0, 1, 2]], quads[:, [0, 2, 3]]])
    return Mesh(vertices.reshape(-1, 3), faces)


def test_import_fault_takes_exactly_one_source():
    plane = Plane.from_dip((0.0, 0.0, 0.0), 45.0, 90.0)
    with pytest.raises(FaultError, match="exactly one"):
        import_fault("F1")
    with pytest.raises(FaultError, match="exactly one"):
        import_fault("F1", plane=plane, points=np.eye(3))
    fault = import_fault("F1", plane=plane, blocks=("W", "E"))
    assert fault.source == "imported-plane" and not fault.inferred
    assert fault.blocks == ("W", "E")
    assert [e.kind for e in fault.provenance] == ["measured"]
    assert faults_provenance([fault]) == fault.provenance  # nothing is missing


def test_import_fault_from_points_is_the_least_squares_plane():
    rng = np.random.default_rng(31)
    truth = Plane.from_dip((500_100.0, 5_300_100.0, 950.0), 75.0, 20.0)
    points = plane_points(truth, rng, 50, noise=0.5)
    fault = import_fault("F2", points=points)
    assert fault.source == "imported-points"
    reference = fit_plane(points)
    np.testing.assert_allclose(fault.plane.normal, reference.normal, atol=1e-15)
    assert fault.misfit["n_points"] == 50
    rms = math.sqrt((reference.signed_distance(points) ** 2).mean())
    assert fault.misfit["plane_rms"] == pytest.approx(rms)
    with pytest.raises(FaultError, match="one line"):
        import_fault("F2", points=np.outer(np.arange(5.0), [1.0, 1.0, 1.0]))


@pytest.mark.parametrize(
    ("dip", "azimuth"), [(65.0, 70.0), (90.0, 30.0), (0.0, 0.0), (20.0, 300.0)]
)
def test_planar_mesh_gives_the_plane_side_inside_and_outside_its_footprint(dip, azimuth):
    rng = np.random.default_rng(32)
    truth = Plane.from_dip((500_450.0, 5_300_400.0, 1_100.0), dip, azimuth)
    mesh = rectangle_mesh(truth, 60.0)
    # reverse some triangles: the winding of an imported mesh must not matter
    faces = mesh.faces.copy()
    faces[::3] = faces[::3, ::-1]
    fault = import_fault("F1", mesh=Mesh(mesh.vertices, faces))
    assert fault.source == "imported-mesh"
    assert angle_between(fault.plane.normal, truth.normal) < 1e-8
    assert fault.misfit["plane_max"] < 1e-7
    points = truth.point + rng.uniform(-300.0, 300.0, (5000, 3))
    distance = truth.signed_distance(points)
    inside = np.abs(plane_coordinates(truth, points)).max(axis=1) < 60.0
    assert inside.sum() > 50 and (~inside).sum() > 1000
    clear = np.abs(distance) > 1e-6
    assert clear.all()
    np.testing.assert_array_equal(fault.side(points), np.where(distance > 0.0, 1, -1))
    np.testing.assert_allclose(fault.offset(points), distance, atol=1e-6)


def plane_coordinates(plane: Plane, points) -> np.ndarray:
    u = np.cross([0.0, 0.0, 1.0], plane.normal)
    u = np.array([1.0, 0.0, 0.0]) if np.linalg.norm(u) < 1e-9 else u / np.linalg.norm(u)
    v = np.cross(plane.normal, u)
    q = np.asarray(points) - plane.point
    return np.column_stack([q @ u, q @ v])


def test_curved_mesh_is_followed_inside_and_its_plane_outside():
    truth = Plane.from_dip((1000.0, 2000.0, 500.0), 50.0, 140.0)

    def bump(a, b):  # symmetric, mean zero over the square: the best-fit plane stays `truth`
        return 6.0 * np.cos(np.pi * a / 80.0) * np.cos(np.pi * b / 80.0)

    fault = import_fault("F3", mesh=rectangle_mesh(truth, 80.0, n=64, bump=bump))
    assert angle_between(fault.plane.normal, truth.normal) < 0.05
    rng = np.random.default_rng(33)
    points = truth.point + rng.uniform(-200.0, 200.0, (4000, 3))
    near = plane_points(truth, rng, 2000, extent=75.0)  # close to the fault, inside the mesh
    points = np.concatenate([points, near + rng.uniform(-8.0, 8.0, 2000)[:, None] * truth.normal])
    ab = plane_coordinates(truth, points)
    height = truth.signed_distance(points)
    inside = np.abs(ab).max(axis=1) < 78.0
    outside = np.abs(ab).max(axis=1) > 82.0
    surface = bump(ab[:, 0], ab[:, 1])
    # inside the footprint: the height above the bent surface along the mean normal
    np.testing.assert_allclose(fault.offset(points)[inside], (height - surface)[inside], atol=0.05)
    between = (
        inside & (np.abs(height - surface) > 0.1) & (np.sign(height) != np.sign(height - surface))
    )
    assert between.sum() > 100  # points between the plane and the surface: the mesh decides
    np.testing.assert_array_equal(
        fault.side(points)[between], np.where(height - surface > 0, 1, -1)[between]
    )
    # outside: the best-fit plane
    np.testing.assert_allclose(
        fault.offset(points)[outside], fault.plane.signed_distance(points)[outside], atol=1e-12
    )


def test_folded_mesh_is_refused():
    vertices = [
        [0, 0, 0],
        [10, 0, 0],
        [10, 10, 0],
        [0, 10, 0],
        [0, 0, 5],
        [0, 10, 5],
        [5, 0, 5],
        [5, 10, 5],
    ]
    faces = [[0, 1, 2], [0, 2, 3], [0, 3, 5], [0, 5, 4], [4, 5, 7], [4, 7, 6]]  # a "C" in section
    with pytest.raises(FaultError, match="folds over"):
        import_fault("F1", mesh=Mesh(np.array(vertices, dtype=float), faces))
    with pytest.raises(FaultError, match="no faces"):
        import_fault("F1", mesh=Mesh(np.zeros((0, 3)), np.zeros((0, 3), dtype=int)))


def test_display_mesh_of_a_plane_is_clipped_to_the_box():
    lo, hi = np.array([0.0, 0.0, 0.0]), np.array([100.0, 60.0, 40.0])
    flat = import_fault("H", plane=Plane((7.0, 7.0, 10.0), (0.0, 0.0, 1.0)))
    for spacing in (None, 13.0):
        mesh = flat.display_mesh(lo, hi, spacing=spacing)
        assert mesh.area == pytest.approx(100.0 * 60.0, rel=1e-9)
        np.testing.assert_allclose(mesh.vertices[:, 2], 10.0, atol=1e-9)
        assert (mesh.face_normals[:, 2] > 0.99).all()
    # x + z = 100 through the box: a rectangle 60 wide and 40 sqrt(2) long
    tilted = import_fault("T", plane=Plane((100.0, 0.0, 0.0), (1.0, 0.0, 1.0)))
    for spacing in (None, 7.0):
        mesh = tilted.display_mesh(lo, hi, spacing=spacing)
        assert mesh.area == pytest.approx(60.0 * 40.0 * math.sqrt(2.0), rel=1e-9)
        assert np.abs(tilted.plane.signed_distance(mesh.vertices)).max() < 1e-9
        assert (mesh.vertices >= lo - 1e-9).all() and (mesh.vertices <= hi + 1e-9).all()
        assert (mesh.face_normals @ tilted.plane.normal > 0.99).all()
    fine = tilted.display_mesh(lo, hi, spacing=7.0)
    assert fine.n_faces > tilted.display_mesh(lo, hi).n_faces
    with pytest.raises(FaultError, match="does not cross"):
        import_fault("M", plane=Plane((0.0, 0.0, 500.0), (0.0, 0.0, 1.0))).display_mesh(lo, hi)
    square = rectangle_mesh(Plane((0.0, 0.0, 0.0), (0.0, 0.0, 1.0)), 5.0)
    assert import_fault("S", mesh=square).display_mesh(lo, hi) is square


def test_fault_surface_round_trip(tmp_path):
    rng = np.random.default_rng(34)
    truth = Plane.from_dip((500_450.0, 5_300_400.0, 1_100.0), 65.0, 70.0)
    below, above = two_clouds(truth, rng)
    inferred = infer_fault(
        "F1", picks_frame(plane_points(truth, rng, 6)), below, above, blocks=("FB1", "FB2")
    )
    imported = import_fault("F2", mesh=rectangle_mesh(truth, 50.0), blocks=("A", "B"))
    points = truth.point + rng.uniform(-200.0, 200.0, (500, 3))
    for fault in (inferred, imported):
        path = fault.save(tmp_path)
        record = json.loads(path.read_text())
        assert record == json.loads(json.dumps(fault.to_dict()))
        assert record["plane"]["dip"] == pytest.approx(65.0, abs=1e-6)
        back = FaultSurface.load(path)
        assert (back.name, back.source, back.blocks) == (fault.name, fault.source, fault.blocks)
        assert back.misfit == json.loads(json.dumps(fault.misfit))
        assert back.provenance == fault.provenance
        np.testing.assert_array_equal(back.offset(points), fault.offset(points))
        assert FaultSurface.from_dict(fault.to_dict(), tmp_path).source == fault.source
    assert (tmp_path / "F2_source.obj").is_file() and not (tmp_path / "F1_source.obj").exists()
    assert imported.to_dict()["mesh"] == "F2_source.obj" and inferred.to_dict()["mesh"] is None
    with pytest.raises(FaultError, match="not a fault record"):
        FaultSurface.from_dict({"format": "other"}, tmp_path)
    with pytest.raises(FaultError, match="not one of"):
        FaultSurface(name="F", source="guessed", plane=truth)


# --------------------------------------------------------------------------
# fault network
# --------------------------------------------------------------------------


def grid_points(lo, hi, step):
    axes = [np.arange(a + 0.5 * step, b, step) for a, b in zip(lo, hi, strict=True)]
    return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)


@pytest.fixture
def two_faults():
    f1 = import_fault("F1", plane=Plane((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)))  # the plane x = 0
    f2 = import_fault("F2", plane=Plane((0.0, 0.0, 0.0), (0.0, 1.0, 0.0)))  # the plane y = 0
    return f1, f2


def test_two_fault_network_with_a_termination_by_hand(two_faults):
    """F1 is the plane x = 0 and cuts everything. F2 is the plane y = 0 and exists only on the
    + side of F1 (x > 0). By hand: WEST for x < 0 at any y; SOUTH for x > 0, y < 0; NORTH for
    x > 0, y > 0."""
    xyz = np.array(
        [
            [-10, -10, 0],
            [-10, 10, 0],
            [-30, 5, 2],
            [10, -10, 0],
            [20, -5, 1],
            [10, 10, 0],
            [5, 30, -3],
        ],
        dtype=float,
    )
    block = ["WEST", "WEST", "WEST", "SOUTH", "SOUTH", "NORTH", "NORTH"]
    network = FaultNetwork.fit(two_faults, xyz, block, terminations={"F2": ("F1", 1)})
    assert network.blocks == ("NORTH", "SOUTH", "WEST")
    assert network.terminations == {"F2": ("F1", 1)}
    assert network.cells == {(-1, 0): "WEST", (1, -1): "SOUTH", (1, 1): "NORTH"}
    query = np.array(
        [[-5, -50, 9], [-5, 50, 9], [3, -1, 0], [3, 1, 0], [np.nan, 0, 0]], dtype=float
    )
    np.testing.assert_array_equal(network.signs(query[:4]), [[-1, 0], [-1, 0], [1, -1], [1, 1]])
    assert network.signs(query).dtype == np.int8
    index = network.block_index(query)
    assert index.dtype == np.int16
    assert index.tolist() == [2, 2, 1, 0, -1]
    assert network.block_of(query).tolist() == ["WEST", "WEST", "SOUTH", "NORTH", None]
    west = network.diagnostics["-1,0"]
    assert (west["n_points"], west["majority"], west["conflicts"]) == (3, "WEST", 0)
    # against the rule written out on a grid
    grid = grid_points((-40, -40, -10), (40, 40, 10), 4.0)
    expected = np.where(grid[:, 0] < 0, "WEST", np.where(grid[:, 1] < 0, "SOUTH", "NORTH"))
    np.testing.assert_array_equal(network.block_of(grid), expected)


def test_two_fault_network_without_termination_reports_conflicts(two_faults):
    xyz = np.array(
        [[-10, -10, 0], [-12, -10, 0], [-9, -8, 0], [-10, 10, 0], [10, -10, 0], [10, 10, 0]],
        dtype=float,
    )
    block = ["A", "A", "B", "A", "B", "C"]
    network = FaultNetwork.fit(two_faults, xyz, block)
    assert network.cells == {(-1, -1): "A", (-1, 1): "A", (1, -1): "B", (1, 1): "C"}
    cell = network.diagnostics["-1,-1"]
    assert cell == {"n_points": 3, "majority": "A", "conflicts": 1, "counts": {"A": 2, "B": 1}}
    assert sum(c["conflicts"] for c in network.diagnostics.values()) == 1


def test_a_cell_without_points_takes_the_nearest_labelled_point(two_faults):
    # nothing is labelled in the cell x > 0, y > 0. The point (2, -1) has to move 1 m across
    # F2 to reach it, the point (-3, 40) has to move 3 m across F1: the first is nearer.
    xyz = np.array([[-3, 40, 0], [-20, -20, 0], [2, -1, 0]], dtype=float)
    network = FaultNetwork.fit(two_faults, xyz, ["A", "B", "C"])
    assert network.cells == {(-1, 1): "A", (-1, -1): "B", (1, -1): "C", (1, 1): "C"}
    filled = network.diagnostics["+1,+1"]
    assert filled["n_points"] == 0 and filled["filled_from"] == "nearest labelled point"
    assert filled["distance"] == pytest.approx(1.0)
    assert network.block_of([[5.0, 5.0, 0.0]]).tolist() == ["C"]


def test_network_round_trip(two_faults, tmp_path):
    xyz = np.array([[-10, 0, 0], [10, -10, 0], [10, 10, 0]], dtype=float)
    network = FaultNetwork.fit(two_faults, xyz, ["W", "S", "N"], terminations={"F2": ("F1", 1)})
    record = network.save(tmp_path)
    assert (tmp_path / "faults" / "F1.json").is_file() and (
        tmp_path / "faults" / "F2.json"
    ).is_file()
    assert record["cells"] == [
        {"signs": [-1, 0], "block": "W"},
        {"signs": [1, -1], "block": "S"},
        {"signs": [1, 1], "block": "N"},
    ]
    assert record["terminations"] == {"F2": ["F1", 1]}
    back = FaultNetwork.from_dict(json.loads(json.dumps(record)), tmp_path)
    assert back.blocks == network.blocks and back.cells == network.cells
    assert back.terminations == network.terminations
    assert [f.name for f in back.faults] == ["F1", "F2"]
    grid = grid_points((-20, -20, -5), (20, 20, 5), 5.0)
    np.testing.assert_array_equal(back.block_index(grid), network.block_index(grid))


def test_network_degenerate_inputs_raise(two_faults):
    f1, f2 = two_faults
    xyz = np.array([[-10, 0, 0], [10, 0, 0]], dtype=float)
    with pytest.raises(FaultError, match="one block only"):
        FaultNetwork.fit([f1], xyz, ["A", "A"])
    with pytest.raises(FaultError, match="no labelled points"):
        FaultNetwork.fit([f1], np.zeros((0, 3)), [])
    with pytest.raises(FaultError, match="one value per"):
        FaultNetwork.fit([f1], xyz, ["A"])
    with pytest.raises(FaultError, match="loop"):
        FaultNetwork.fit([f1, f2], xyz, ["A", "B"], terminations={"F1": ("F2", 1), "F2": ("F1", 1)})
    with pytest.raises(FaultError, match="different faults"):
        FaultNetwork.fit([f1, f2], xyz, ["A", "B"], terminations={"F1": ("F7", 1)})
    with pytest.raises(FaultError, match=r"\+1 or -1"):
        FaultNetwork.fit([f1, f2], xyz, ["A", "B"], terminations={"F2": ("F1", 0)})
    with pytest.raises(FaultError, match="unique"):
        FaultNetwork.fit([f1, f1], xyz, ["A", "B"])
    # one fault that names its blocks needs no labelled points
    named = import_fault("F1", plane=f1.plane, blocks=("LEFT", "RIGHT"))
    network = FaultNetwork.fit([named], np.zeros((0, 3)), [])
    assert network.block_of([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]).tolist() == ["LEFT", "RIGHT"]


# --------------------------------------------------------------------------
# the small layered deposit (geology_contracts.md 13, 14)
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def layered_faults(layered_small, layered_small_db, layered_small_traces, layered_units_path):
    """What the logs of the deposit give: labelled points, fault-zone intervals, picks."""
    assay = layered_small_db.intervals["assay"]
    geology = layered_small_db.intervals["geology"]
    units = UnitDictionary.load(layered_units_path)
    kinds = units.kinds_of(units.map_codes(geology["lith"]))
    zones = geology[(kinds == "fault-zone").to_numpy()]
    blocks = block_codes(assay["wf"], r"^(FB\d)-")
    xyz, block, hole = labelled_points(assay, blocks, layered_small_traces)
    truth = layered_small.truth
    return {
        "assay": assay,
        "blocks": blocks,
        "zones": zones,
        "xyz": xyz,
        "block": block,
        "hole": hole,
        "truth": truth,
        "truth_plane": Plane(truth.fault_point, truth.fault_normal),
        "traces": layered_small_traces,
    }


def infer_f1(data, picks, **extra) -> FaultSurface:
    a, b = data["block"] == "FB1", data["block"] == "FB2"
    return infer_fault(
        "F1", picks, data["xyz"][a], data["xyz"][b], blocks=("FB1", "FB2"),
        holes_a=data["hole"][a], holes_b=data["hole"][b], **extra,
    )  # fmt: skip


def recovery(fault: FaultSurface, truth: Plane, at) -> tuple[float, float, float]:
    """Dip error, dip-azimuth error (degrees), and the distance between the inferred and the
    true plane at the centroid ``at``: the centroid is put on the inferred plane and its
    distance to the true plane is taken."""
    centroid = np.asarray(at, dtype=np.float64).mean(axis=0)
    on_plane = centroid - fault.plane.signed_distance(centroid) * fault.plane.normal
    return (
        abs(fault.plane.dip - truth.dip),
        azimuth_difference(fault.plane.dip_azimuth, truth.dip_azimuth),
        abs(float(truth.signed_distance(on_plane))),
    )


def box_grid(cfg, step=10.0):
    lo = (cfg.x_range[0], cfg.y_range[0], cfg.z_range[0])
    hi = (cfg.x_range[1], cfg.y_range[1], cfg.z_range[1])
    return grid_points(lo, hi, step)


def test_layered_labelled_points_agree_with_the_truth(layered_faults):
    data = layered_faults
    assert set(data["block"]) == {"FB1", "FB2"}
    np.testing.assert_array_equal(data["truth"].labels(data["xyz"], "fault_block"), data["block"])
    # the midpoints are the midpoints of the coded assay intervals
    coded = data["assay"][data["blocks"].notna().to_numpy()]
    assert len(coded) == len(data["xyz"])


def test_layered_fault_picks_are_on_the_true_fault(layered_faults, layered_small):
    data = layered_faults
    picks = fault_picks(data["assay"], data["blocks"], data["traces"], fault_zone=data["zones"])
    assert len(picks) >= layered_small.config.min_fault_holes
    assert set(picks["source"]) <= {"fault-zone", "gap-middle"}
    xyz = picks[["x", "y", "z"]].to_numpy()
    # a logged zone is 1 m either side of the crossing; core loss may cut one end off
    assert np.abs(data["truth_plane"].signed_distance(xyz)).max() < 1.0
    np.testing.assert_allclose(picks["weight"], 1.0 / np.maximum(picks["uncertainty"], 0.5) ** 2)
    assert picks["hole_id"].is_unique


def test_layered_f1_inferred_within_the_targets(layered_faults, layered_small):
    data = layered_faults
    picks = fault_picks(data["assay"], data["blocks"], data["traces"], fault_zone=data["zones"])
    fault = infer_f1(data, picks)
    dip, azimuth, position = recovery(fault, data["truth_plane"], picks[["x", "y", "z"]].to_numpy())
    print(
        f"\nF1 from {len(picks)} picks ({fault.source}): dip {fault.plane.dip:.4f}, "
        f"dip azimuth {fault.plane.dip_azimuth:.4f}; errors: dip {dip:.5f} deg, "
        f"dip azimuth {azimuth:.5f} deg, position at the picks' centroid {position:.5f} m; "
        f"{fault.misfit['n_wrong']} of {fault.misfit['n_points']} labelled points on the wrong side"
    )
    assert fault.source == "inferred-plane"
    assert fault.blocks == ("FB1", "FB2")  # FB2 is the hanging wall: the positive side
    assert dip <= DIP_TARGET
    assert azimuth <= AZIMUTH_TARGET
    assert position <= POSITION_TARGET
    assert fault.misfit["n_wrong"] == 0 and fault.misfit["holes_wrong"] == {}

    # fault blocks on a 10 m grid over the box against the truth
    network = FaultNetwork.fit([fault], data["xyz"], data["block"])
    assert network.blocks == ("FB1", "FB2")
    assert network.cells == {(-1,): "FB1", (1,): "FB2"}
    grid = box_grid(layered_small.config)
    expected = data["truth"].labels(grid, "fault_block")
    got = network.block_of(grid)
    wrong = got != expected
    agreement = 100.0 * (1.0 - wrong.mean())
    worst = np.abs(data["truth_plane"].signed_distance(grid[wrong])).max() if wrong.any() else 0.0
    print(
        f"fault-block agreement on {len(grid)} grid points (10 m): {agreement:.4f} %; "
        f"{int(wrong.sum())} wrong, all within {worst:.3f} m of the true fault"
    )
    assert agreement >= 97.0  # section 14, inferred (the block check proper is T13's)
    assert worst < 5.0


@pytest.mark.xfail(
    strict=True,
    reason="section 14 target missed without fault-zone intervals: the block codes are logged "
    "only in the thin ore lens, which does not fix the dip of the separating plane",
)
def test_layered_f1_from_block_codes_alone_meets_the_targets(layered_faults):
    data = layered_faults
    picks = fault_picks(data["assay"], data["blocks"], data["traces"])  # no fault-zone intervals
    fault = infer_f1(data, picks)
    dip, azimuth, position = recovery(fault, data["truth_plane"], data["xyz"])
    assert dip <= DIP_TARGET and azimuth <= AZIMUTH_TARGET and position <= POSITION_TARGET


def test_layered_f1_from_block_codes_alone_is_reported(layered_faults, layered_small):
    """No hole changes block inside the ore lens, so without the logged fault zones there is
    no crossing and the fault is the separating plane of the coded midpoints. It separates
    them all; its orientation misses the section 14 targets (the test above)."""
    data = layered_faults
    picks = fault_picks(data["assay"], data["blocks"], data["traces"])
    assert len(picks) == 0
    fault = infer_f1(data, picks)
    assert fault.source == "inferred-separating"
    assert fault.misfit["n_wrong"] == 0 and fault.misfit["converged"]
    dip, azimuth, position = recovery(fault, data["truth_plane"], data["xyz"])
    network = FaultNetwork.fit([fault], data["xyz"], data["block"])
    grid = box_grid(layered_small.config)
    agreement = 100.0 * (network.block_of(grid) == data["truth"].labels(grid, "fault_block")).mean()
    print(
        f"\nF1 from block codes alone (separating plane, {fault.misfit['n_points']} midpoints): "
        f"errors: dip {dip:.2f} deg, dip azimuth {azimuth:.2f} deg, position at the centroid of "
        f"the midpoints {position:.2f} m; fault-block agreement on the 10 m grid {agreement:.2f} %"
    )
    # with the user's orientation only the position is inferred: the middle of the gap
    prior = infer_f1(data, picks, prior=(65.0, 70.0))
    assert prior.source == "inferred-separating"
    assert prior.plane.dip == pytest.approx(65.0) and prior.plane.dip_azimuth == pytest.approx(70.0)
    truth_distance = data["truth_plane"].signed_distance(data["xyz"])
    low, high = truth_distance[truth_distance <= 0].max(), truth_distance[truth_distance > 0].min()
    assert data["truth_plane"].signed_distance(prior.plane.point) == pytest.approx(
        0.5 * (low + high)
    )
    print(
        f"with the prior orientation (65, 070): position error "
        f"{recovery(prior, data['truth_plane'], data['xyz'])[2]:.2f} m "
        f"(gap between the blocks along the normal: {high - low:.2f} m)"
    )


def test_layered_imported_true_plane_mesh_gives_the_blocks_exactly(layered_faults, layered_small):
    data = layered_faults
    vertices, faces = layered_small.meshes["truth_fault_F1"]  # the plane clipped to the box
    fault = import_fault("F1", mesh=Mesh(vertices, faces))
    assert fault.source == "imported-mesh"
    assert angle_between(fault.plane.normal, data["truth_plane"].normal) < 1e-7
    network = FaultNetwork.fit([fault], data["xyz"], data["block"])
    assert network.cells == {(-1,): "FB1", (1,): "FB2"}
    assert sum(cell["conflicts"] for cell in network.diagnostics.values()) == 0

    cfg = layered_small.config
    lo = np.array([cfg.x_range[0], cfg.y_range[0], cfg.z_range[0]]) - 150.0
    hi = np.array([cfg.x_range[1], cfg.y_range[1], cfg.z_range[1]]) + 150.0
    grid = grid_points(lo, hi, 10.0)
    in_box = ((grid >= lo + 150.0) & (grid <= hi - 150.0)).all(axis=1)
    assert (~in_box).sum() > in_box.sum()  # most of the grid is beyond the mesh
    distance = data["truth_plane"].signed_distance(grid)
    print(
        f"\nimported plane mesh: {len(grid)} grid points, {int((~in_box).sum())} outside the box; "
        f"nearest to the fault {np.abs(distance).min():.2e} m"
    )
    expected = data["truth"].labels(grid, "fault_block")
    np.testing.assert_array_equal(network.block_of(grid), expected)
    # the same through an imported plane and through imported points
    for other in (
        import_fault(
            "F1", plane=Plane.from_dip(cfg.fault_point, cfg.fault_dip, cfg.fault_dip_azimuth)
        ),
        import_fault("F1", points=vertices),
    ):
        again = FaultNetwork.fit([other], data["xyz"], data["block"])
        np.testing.assert_array_equal(again.block_of(grid), expected)


def test_layered_provenance_entries(layered_faults):
    data = layered_faults
    picks = fault_picks(data["assay"], data["blocks"], data["traces"], fault_zone=data["zones"])
    fault = infer_f1(data, picks)
    entries = faults_provenance([fault])
    assert {e.kind for e in entries} == set(PROVENANCE_KINDS)
    assert all(e.stage == "faults" and e.task == "P9.T8" for e in entries)
    assumed = next(e for e in entries if e.kind == "assumed")
    assert assumed.item == "Fault F1"
    assert assumed.source == f"inferred from {len(picks)} fault crossings along holes (plane fit)"
    assert assumed.used_for == "fault blocks FB1 / FB2"
    assert assumed.detail.startswith("plane dip 65, dip azimuth 070; 0 points on the wrong side")
    measured = next(e for e in entries if e.kind == "measured")
    assert measured.item == "Fault picks of F1" and measured.count == len(picks)
    missing = entries[-1]
    assert (missing.item, missing.kind) == ("Fault surfaces", "missing")
    assert missing.source == "the geologist's fault interpretation (DXF / OBJ / points)"
    assert missing.used_for == "would replace the inferred planes"

    # the separating plane says how it was found
    codes_only = infer_f1(data, None)
    line = codes_only.provenance[0]
    assert line.kind == "assumed" and "separating plane" in line.source
    assert "dip azimuth" in line.detail and "on the wrong side" in line.detail

    # once the geologist's surface is imported nothing is missing
    imported = import_fault("F1", plane=data["truth_plane"])
    assert [e.kind for e in faults_provenance([imported])] == ["measured"]
    network = FaultNetwork.fit([fault], data["xyz"], data["block"])
    assert network.provenance == entries
    record = Provenance(name="test")
    record.extend(entries)
    assert [len(table) for table in record.tables().values()] == [1, 1, 1]
