"""Hole-out cross-validation (P9.T11): hand cases with a known answer, the confusion matrix
against pandas, the folds, a deliberately wrong builder, and the synthetic layered deposit,
where the cross-validated figure is put next to the true accuracy of the model."""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import pytest

from bakumine.core.drillholes.desurvey import desurvey
from bakumine.core.geology.faults import block_codes
from bakumine.core.geology.lithology import EventSpec, fit_stack, fit_unit_fields
from bakumine.core.geology.provenance import Provenance
from bakumine.core.geology.units import UnitDictionary
from bakumine.core.geology.validation import (
    NONE_LABEL,
    SCOPE_TEXT,
    confusion_matrix,
    cross_validate,
    cross_validate_fault_blocks,
    cross_validate_units,
    fault_network_builder,
    hole_folds,
    largest_field_builder,
    match_contacts,
    predicted_names,
)

LENGTH = 100.0

# --------------------------------------------------------------------------
# made-up drilling
# --------------------------------------------------------------------------


def vertical_holes(collars, length=LENGTH):
    collars = np.asarray(collars, dtype=np.float64)
    ids = [f"V{i + 1:02d}" for i in range(len(collars))]
    collar = pd.DataFrame(
        {"hole_id": ids, "x": collars[:, 0], "y": collars[:, 1], "z": collars[:, 2],
         "max_depth": length}
    )  # fmt: skip
    survey = pd.DataFrame(
        {"hole_id": np.repeat(ids, 2), "depth": np.tile([0.0, length], len(ids)),
         "dip": -90.0, "azimuth": 0.0}
    )  # fmt: skip
    return ids, desurvey(collar, survey)


def two_unit_log(ids, contact_depth, length=LENGTH):
    """``UP`` from the collar to the contact, ``LOW`` below it."""
    rows = []
    for hole, depth in zip(ids, contact_depth, strict=True):
        rows += [(hole, 0.0, float(depth), "UP"), (hole, float(depth), length, "LOW")]
    return pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])


class NearestHole:
    """A model with a known answer: the log of the nearest hole it was given, at the same
    elevation. Ties go to the hole with the smaller x. Brute force, vertical holes."""

    def __init__(self, table, labels, collars):
        self.table = table.assign(unit=labels.to_numpy())
        self.collars = {h: collars[h] for h in table["hole_id"].unique()}

    def labels(self, points):
        out = np.empty(len(points), dtype=object)
        holes = sorted(self.collars, key=lambda h: self.collars[h][0])
        for n, (x, y, z) in enumerate(np.asarray(points)):
            best = min(holes, key=lambda h: round(np.hypot(*(self.collars[h][:2] - (x, y))), 9))
            log = self.table[self.table["hole_id"] == best]
            depth = self.collars[best][2] - z
            hit = log[(log["depth_from"] <= depth) & (depth < log["depth_to"])]
            out[n] = hit["unit"].iloc[0] if len(hit) else None
        return out


def nearest_hole_builder(collars):
    return lambda table, labels, traces: NearestHole(table, labels, collars)


# --------------------------------------------------------------------------
# folds
# --------------------------------------------------------------------------


@pytest.mark.parametrize("n, k", [(64, 10), (7, 3), (10, 10), (5, 2), (23, 4)])
def test_folds_partition_the_holes(n, k):
    holes = [f"H{i:03d}" for i in range(n)]
    folds = hole_folds(holes, k, seed=3)
    assert len(folds) == min(k, n)
    flat = [h for fold in folds for h in fold]
    assert len(flat) == len(set(flat)) == n  # no hole in two folds
    assert set(flat) == set(holes)  # none missing
    sizes = [len(fold) for fold in folds]
    assert max(sizes) - min(sizes) <= 1


def test_folds_are_deterministic_for_a_seed():
    holes = [f"H{i:03d}" for i in range(40)]
    first = hole_folds(holes, 5, seed=11)
    assert hole_folds(holes, 5, seed=11) == first
    # the order and repetition of the input do not matter, only the set of holes
    shuffled = list(np.random.default_rng(0).permutation(holes)) + holes[:5]
    assert hole_folds(shuffled, 5, seed=11) == first
    assert hole_folds(holes, 5, seed=12) != first


def test_leave_one_out_and_bad_fold_counts():
    holes = ["b", "a", "c"]
    assert hole_folds(holes, 0) == [("a",), ("b",), ("c",)]
    assert hole_folds(holes, None) == hole_folds(holes, 99) == hole_folds(holes, 0)
    with pytest.raises(ValueError, match="one fold"):
        hole_folds(holes, 1)
    with pytest.raises(ValueError, match="folds"):
        hole_folds(holes, -2)
    with pytest.raises(ValueError, match="two holes"):
        hole_folds(["a"], 0)


# --------------------------------------------------------------------------
# two half-spaces, a planar contact, regular holes: the answer is known
# --------------------------------------------------------------------------

SLOPE = 0.25
SPACING = 20.0


def line_case(first_contact):
    """Six vertical holes on a line; the contact dips so that it is 5 m deeper per hole."""
    x = SPACING * np.arange(6)
    collars = np.column_stack([x, np.zeros(6), np.zeros(6)])
    ids, traces = vertical_holes(collars)
    table = two_unit_log(ids, first_contact + SLOPE * x)
    return table, traces, dict(zip(ids, collars, strict=True))


def test_planar_contact_nearest_hole_model_analytic():
    """Leave-one-out with the nearest-hole model. The first hole is predicted by the second
    (contact 5 m deeper), every other hole by its neighbour on the low-x side (contact 5 m
    shallower). So: contact error exactly +5 m once and -5 m five times, 5 m of every 100 m
    hole wrong, agreement 0.95, and the wrong metres are LOW->UP 5 m, UP->LOW 25 m (logged:
    UP 40 + 45 + ... + 65 = 315 m, LOW 285 m)."""
    table, traces, collars = line_case(40.0)
    result = cross_validate(
        table, table["unit"], traces, nearest_hole_builder(collars), folds=0, step=1.0
    )
    assert result.leave_one_out and len(result.folds) == 6
    assert result.metres == pytest.approx(600.0)
    assert result.agreement == pytest.approx(0.95, abs=1e-12)
    np.testing.assert_allclose(result.by_hole["agreement"], 0.95, atol=1e-12)
    expected = pd.DataFrame(
        [[280.0, 5.0, 0.0], [25.0, 290.0, 0.0]],
        index=["LOW", "UP"], columns=["LOW", "UP", NONE_LABEL],
    )  # fmt: skip
    np.testing.assert_allclose(result.confusion.loc[expected.index, expected.columns], expected)

    contacts = result.contacts.sort_values("hole_id")
    assert len(contacts) == 6 and contacts["error"].notna().all()
    np.testing.assert_allclose(contacts["error"], [5.0, -5.0, -5.0, -5.0, -5.0, -5.0], atol=0.01)
    assert result.contact["matched"] == 6 and result.contact["spurious"] == 0
    assert result.contact["mean_abs"] == pytest.approx(SLOPE * SPACING, abs=0.01)
    assert result.contact["bias"] == pytest.approx(-20.0 / 6.0, abs=0.01)
    # the nearest remaining hole is one spacing away (data points every 1 m along it)
    assert result.samples["distance"].between(SPACING, np.hypot(SPACING, 0.5) + 1e-9).all()
    assert result.contacts["distance"].between(SPACING, np.hypot(SPACING, 0.5) + 1e-9).all()

    recall = result.per_unit.set_index("unit")["recall"]
    precision = result.per_unit.set_index("unit")["precision"]
    assert recall["LOW"] == pytest.approx(280.0 / 285.0)
    assert recall["UP"] == pytest.approx(290.0 / 315.0)
    assert precision["LOW"] == pytest.approx(280.0 / 305.0)
    assert precision["UP"] == pytest.approx(290.0 / 295.0)


def test_planar_contact_not_on_the_piece_ends():
    """The same with contacts that do not fall on the ends of the scored pieces: the contact
    error stays exact (bisection), the agreement is right to one piece per hole."""
    table, traces, collars = line_case(40.37)
    result = cross_validate(
        table, table["unit"], traces, nearest_hole_builder(collars), folds=0, step=0.7
    )
    np.testing.assert_allclose(result.contacts["error"].abs(), 5.0, atol=0.01)
    assert result.metres == pytest.approx(600.0)
    assert np.abs(result.by_hole["agreement"] - 0.95).max() <= 0.7 / LENGTH
    # two folds of three holes: still every hole predicted once, from the other fold
    halves = cross_validate(
        table, table["unit"], traces, nearest_hole_builder(collars), folds=2, seed=1, step=0.7
    )
    assert sorted(len(f) for f in halves.folds) == [3, 3]
    assert halves.metres == pytest.approx(600.0)
    assert sorted(halves.by_hole["hole_id"]) == sorted(collars)


def grid_case(contact):
    axis = 25.0 * np.arange(3)
    x, y = np.meshgrid(axis, axis, indexing="ij")
    collars = np.column_stack([x.ravel(), y.ravel(), np.zeros(9)])
    ids, traces = vertical_holes(collars)
    return two_unit_log(ids, contact(collars[:, 0], collars[:, 1])), traces, ids, collars


def test_horizontal_contact_default_builder_by_symmetry():
    """A horizontal contact at mid depth, the default (largest-field) model. The data are
    antisymmetric about the contact plane, so every refitted model puts the contact on that
    plane: agreement 1 and contact error 0, whatever hole is left out."""
    table, traces, _, _ = grid_case(lambda x, y: np.full(len(x), 50.0))
    result = cross_validate_units(table, table["unit"], traces, ("UP", "LOW"), folds=0, step=1.0)
    assert result.agreement == pytest.approx(1.0, abs=1e-12)
    assert result.contact["logged"] == result.contact["matched"] == 9
    assert result.contact["max_abs"] <= 0.02
    assert result.contact["spurious"] == 0
    assert result.categories == ("UP", "LOW")


def test_dipping_contact_default_builder_against_brute_force():
    """A dipping contact, the default model, leave-one-out. The test repeats every fold by
    hand: fit the T7 model without the hole, scan the hole every 0.01 m, take the agreement
    and the contact from the scan; the cross-validation must give the same."""
    table, traces, ids, collars = grid_case(lambda x, y: 38.0 + 0.3 * x + 0.12 * y)
    step = 0.5
    result = cross_validate_units(table, table["unit"], traces, ("UP", "LOW"), folds=0, step=step)
    depth = np.arange(0.005, LENGTH, 0.01)
    by_hole = result.by_hole.set_index("hole_id")
    contacts = result.contacts.set_index("hole_id")
    errors = []
    for hole, collar in zip(ids, collars, strict=True):
        rest = table[table["hole_id"] != hole]
        fields = fit_unit_fields(rest, rest["unit"], ("UP", "LOW"), traces)
        points = np.column_stack(
            [np.full(len(depth), collar[0]), np.full(len(depth), collar[1]), -depth]
        )
        got = np.array(fields.units, dtype=object)[fields.labels(points)]
        logged_contact = float(table.loc[table["hole_id"] == hole, "depth_to"].iloc[0])
        truth = np.where(depth < logged_contact, "UP", "LOW")
        agreement = float((got == truth).mean())
        change = np.flatnonzero(got[1:] != got[:-1])
        assert len(change) == 1  # one predicted contact
        predicted_contact = depth[change[0]] + 0.005
        errors.append(predicted_contact - logged_contact)
        assert by_hole.loc[hole, "agreement"] == pytest.approx(agreement, abs=step / LENGTH)
        assert contacts.loc[hole, "predicted_depth"] == pytest.approx(predicted_contact, abs=0.02)
    errors = np.array(errors)
    assert result.contact["mean_abs"] == pytest.approx(np.abs(errors).mean(), abs=0.02)
    assert result.agreement == pytest.approx(1.0 - np.abs(errors).sum() / 900.0, abs=step / LENGTH)
    print(
        f"\ndipping plane, 9 holes at 25 m, leave-one-out: agreement {result.agreement:.4f}, "
        f"mean contact error {result.contact['mean_abs']:.2f} m (brute force "
        f"{np.abs(errors).mean():.2f} m)"
    )


# --------------------------------------------------------------------------
# a random log: the confusion matrix against pandas, a wrong builder
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def random_log():
    rng = np.random.default_rng(5)
    collars = np.column_stack([30.0 * np.arange(8), 10.0 * (np.arange(8) % 3), np.zeros(8)])
    ids, traces = vertical_holes(collars)
    rows = []
    for hole in ids:
        depth = 0.0
        while depth < LENGTH - 1e-9:
            end = min(LENGTH, round(depth + float(rng.uniform(0.4, 9.0)), 2))
            code = ["A", "B", "C", None, "X"][int(rng.choice(5, p=[0.4, 0.25, 0.2, 0.1, 0.05]))]
            rows.append((hole, depth, end, code))
            depth = end
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "unit"])
    return table, traces, dict(zip(ids, collars, strict=True))


class Always:
    def __init__(self, unit):
        self.unit = unit

    def labels(self, points):
        return np.full(len(points), self.unit, dtype=object)


def logged_metres(table, units):
    """Metres per unit with pandas only."""
    length = table["depth_to"] - table["depth_from"]
    return length.groupby(table["unit"]).sum().reindex(list(units)).fillna(0.0)


def logged_contact_count(table, units):
    """Contacts between two scored units: abutting rows of one hole with different units."""
    n = 0
    for _, log in table.groupby("hole_id"):
        log = log.sort_values("depth_from")
        unit = log["unit"].to_numpy(dtype=object)
        for a, b in zip(unit[:-1], unit[1:], strict=True):
            n += int(a in units and b in units and a != b)
    return n


def test_wrong_builder_gives_the_predictable_numbers(random_log):
    """A model that always says ``A``: the agreement is the share of ``A`` in the logs, the
    whole matrix sits in column ``A``, no contact is found."""
    table, traces, _ = random_log
    scored = ("A", "B", "C")
    result = cross_validate(
        table, table["unit"], traces, lambda *_: Always("A"), scored=scored, folds=4, step=1.0
    )
    metres = logged_metres(table, scored)
    assert result.metres == pytest.approx(metres.sum(), rel=1e-12)
    assert result.agreement == pytest.approx(metres["A"] / metres.sum(), rel=1e-12)
    np.testing.assert_allclose(result.confusion["A"], metres, rtol=1e-12)
    assert (result.confusion.drop(columns="A").to_numpy() == 0.0).all()
    scores = result.per_unit.set_index("unit")
    assert scores.loc["A", "recall"] == 1.0
    assert scores.loc["B", "recall"] == scores.loc["C", "recall"] == 0.0
    assert scores.loc["A", "precision"] == pytest.approx(metres["A"] / metres.sum())
    assert np.isnan(scores.loc["B", "precision"]) and np.isnan(scores.loc["C", "precision"])
    n = logged_contact_count(table, scored)
    assert n > 20
    assert result.contact["logged"] == n
    assert result.contact["matched"] == 0 and result.contact["missed"] == n
    assert result.contact["spurious"] == 0 and np.isnan(result.contact["mean_abs"])
    # by distance: the classes hold all the metres, each with the same kind of answer
    assert result.distance_table["metres"].sum() == pytest.approx(metres.sum())
    assert result.distance_table["contacts"].sum() == n
    assert len(result.distance_table) == 4
    np.testing.assert_allclose(
        result.distance_table["metres"], metres.sum() / 4, rtol=0.05
    )  # equal-metre classes


def test_confusion_rows_sum_to_the_logged_metres(random_log):
    table, traces, collars = random_log
    scored = ("A", "B", "C")
    result = cross_validate(
        table, table["unit"], traces, nearest_hole_builder(collars), scored=scored, folds=3,
        seed=4, step=0.8,
    )  # fmt: skip
    metres = logged_metres(table, scored)
    np.testing.assert_allclose(result.confusion.sum(axis=1), metres, rtol=1e-12)
    np.testing.assert_allclose(result.per_unit["logged_m"], metres, rtol=1e-12)
    # the nearest-hole model predicts X and unknown stretches too: extra columns, no loss
    assert list(result.confusion.columns) == ["A", "B", "C", "X", NONE_LABEL]
    assert result.confusion["X"].sum() > 0 and result.confusion[NONE_LABEL].sum() > 0
    diagonal = sum(result.confusion.loc[u, u] for u in scored)
    assert result.agreement == pytest.approx(diagonal / metres.sum(), rel=1e-12)
    # the same matrix from the per-piece table with pandas
    pieces = result.samples.assign(predicted=result.samples["predicted"].fillna(NONE_LABEL))
    pivot = pieces.pivot_table(
        index="logged", columns="predicted", values="length", aggfunc="sum", fill_value=0.0
    )
    np.testing.assert_allclose(
        result.confusion.loc[pivot.index, pivot.columns], pivot.to_numpy(), rtol=1e-12
    )
    np.testing.assert_allclose(confusion_matrix(result.samples, scored), result.confusion)
    # distance classes given as limits
    limits = cross_validate(
        table, table["unit"], traces, lambda *_: Always("B"), scored=scored, folds=3, seed=4,
        distance_bins=[30.5, 31.0],
    )  # fmt: skip
    assert list(limits.distance_table["distance_from"]) == [0.0, 30.5, 31.0]
    assert limits.distance_table["metres"].sum() == pytest.approx(metres.sum())


def test_builder_sees_the_remaining_holes_only(random_log):
    table, traces, _ = random_log
    seen = []

    def builder(part, labels, _traces):
        assert labels.index.equals(part.index)
        seen.append(set(part["hole_id"]))
        return Always("A")

    result = cross_validate(table, table["unit"], traces, builder, folds=3, seed=9)
    everything = set(table["hole_id"])
    assert len(seen) == 3
    for fold, holes in zip(result.folds, seen, strict=True):
        assert holes == everything - set(fold)
    # every hole is scored in the fold it was left out of
    assert dict(zip(result.by_hole["hole_id"], result.by_hole["fold"], strict=True)) == {
        h: n for n, fold in enumerate(result.folds) for h in fold
    }
    # the same seed gives the same result, another seed other folds
    again = cross_validate(table, table["unit"], traces, builder, folds=3, seed=9)
    assert again.folds == result.folds
    pd.testing.assert_frame_equal(again.samples, result.samples)


def test_a_unit_known_from_the_left_out_hole_only_counts_as_wrong():
    table, traces, ids, _ = grid_case(lambda x, y: np.full(len(x), 50.0))
    rare = pd.DataFrame(
        [(ids[4], 0.0, 30.0, "UP"), (ids[4], 30.0, 40.0, "RARE"), (ids[4], 40.0, 50.0, "UP"),
         (ids[4], 50.0, LENGTH, "LOW")],
        columns=table.columns,
    )  # fmt: skip
    table = pd.concat([table[table["hole_id"] != ids[4]], rare], ignore_index=True)
    builder = largest_field_builder(("UP", "LOW", "RARE"))
    result = cross_validate(table, table["unit"], traces, builder, folds=0)
    scores = result.per_unit.set_index("unit")
    assert scores.loc["RARE", "logged_m"] == pytest.approx(10.0)
    assert scores.loc["RARE", "recall"] == 0.0


def test_inputs_are_checked(random_log):
    table, traces, _ = random_log
    with pytest.raises(ValueError, match="step"):
        cross_validate(table, table["unit"], traces, lambda *_: Always("A"), step=0.0)
    with pytest.raises(ValueError, match="index"):
        cross_validate(table, table["unit"].iloc[::-1], traces, lambda *_: Always("A"))
    with pytest.raises(ValueError, match="no interval"):
        cross_validate(table, table["unit"], traces, lambda *_: Always("A"), scored=("ZZ",))

    def failing(*_):
        raise ValueError("nothing to fit")

    with pytest.raises(ValueError, match="fold 1 of 8.*nothing to fit"):
        cross_validate(table, table["unit"], traces, failing, folds=0)
    with pytest.raises(TypeError, match="labels"):
        cross_validate(table, table["unit"], traces, lambda *_: object(), folds=0)


def test_predicted_names_reads_codes_and_names():
    class Codes:
        categories = ("P", "Q")

        def labels(self, points):
            return np.array([0, 1, -1], dtype=np.int16)

    points = np.zeros((3, 3))
    assert list(predicted_names(Codes(), points)) == ["P", "Q", None]
    names = type("N", (), {"labels": lambda self, p: np.array(["P", None, np.nan], dtype=object)})
    assert list(predicted_names(names(), points)) == ["P", None, None]


def test_match_contacts_one_to_one_nearest_first():
    logged = pd.DataFrame(
        {"hole_id": ["h", "h", "h", "g"], "depth": [10.0, 20.0, 50.0, 10.0],
         "above": ["A", "A", "B", "A"], "below": ["B", "B", "A", "B"]}
    )  # fmt: skip
    predicted = pd.DataFrame(
        {"hole_id": ["h", "h", "h"], "depth": [18.0, 49.0, 80.0],
         "above": ["A", "B", "B"], "below": ["B", "A", "A"]}
    )  # fmt: skip
    depth, used = match_contacts(logged, predicted)
    # A/B at 18 goes to the logged contact at 20 (2 m), not to the one at 10 (8 m)
    np.testing.assert_array_equal(depth, [np.nan, 18.0, 49.0, np.nan])
    assert list(used) == [True, True, False]
    depth, used = match_contacts(logged, predicted, max_distance=1.5)
    np.testing.assert_array_equal(depth, [np.nan, np.nan, 49.0, np.nan])
    assert list(used) == [False, True, False]


# --------------------------------------------------------------------------
# fault blocks on a hand case
# --------------------------------------------------------------------------


def test_fault_blocks_across_a_known_plane():
    """Vertical holes through the plane x + z = 0 (dip 45): block ``E`` above depth x,
    ``W`` below. The crossings lie on the plane exactly, so the fault inferred without any
    one hole is the plane itself: agreement 1, crossing error 0."""
    xs = np.arange(10.0, 91.0, 10.0)
    collars = np.array([(x, y, 0.0) for y in (0.0, 30.0, 60.0) for x in xs])
    ids, traces = vertical_holes(collars)
    rows = []
    for hole, (x, _, _) in zip(ids, collars, strict=True):
        for a in np.arange(0.0, LENGTH, 2.0):
            rows.append((hole, a, a + 2.0, "E-zone" if a + 1.0 < x else "W-zone"))
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "code"])
    blocks = block_codes(table["code"], r"^(\w)-")
    builder = fault_network_builder([{"name": "F", "between": ("E", "W")}])
    result = cross_validate_fault_blocks(table, blocks, traces, builder, folds=0, step=1.0)
    assert result.layer == "fault_block"
    assert result.metres == pytest.approx(len(ids) * LENGTH)
    assert result.agreement == pytest.approx(1.0, abs=1e-12)
    assert result.contact["logged"] == result.contact["matched"] == len(ids)
    assert result.contact["max_abs"] <= 0.02
    assert {e.stage for e in result.provenance} == {"faults"}
    # a wrong prior orientation (vertical fault) must show as a loss
    wrong = fault_network_builder([{"name": "F", "between": ("E", "W"), "prior": (90.0, 90.0)}])
    tilted = cross_validate_fault_blocks(table, blocks, traces, wrong, folds=0, step=1.0)
    assert tilted.agreement < 0.9


# --------------------------------------------------------------------------
# provenance and text
# --------------------------------------------------------------------------


def test_provenance_entries_and_text(random_log):
    table, traces, collars = random_log
    result = cross_validate(
        table, table["unit"], traces, nearest_hole_builder(collars), scored=("A", "B", "C"),
        folds=4,
    )  # fmt: skip
    entries = result.provenance
    measured = [e for e in entries if e.kind == "measured"]
    assumed = [e for e in entries if e.kind == "assumed"]
    assert len(measured) == 3 and len(assumed) == 1
    assert all(e.stage == "lithology" and e.task == "P9.T11" for e in entries)
    assert measured[0].value == pytest.approx(result.agreement, abs=1e-4)
    assert measured[0].count == 8
    assert "4 folds by hole, 8 holes, seed 0" in measured[0].source
    assert "beyond the drilling" in assumed[0].item
    record = Provenance(name="made up")
    record.extend(entries)
    assert Provenance.from_dict(record.to_dict()).entries == record.entries
    assert set(record.tables()) == {"measured", "assumed", "missing"}

    text = result.format_text()
    assert SCOPE_TEXT in text
    assert "between the holes, at about the drill spacing" in text
    assert "nothing is tested beyond the drilling" in text
    assert f"agreement by length: {result.agreement:.4f}" in text
    data = json.loads(json.dumps(result.to_dict()))
    assert data["agreement"] == pytest.approx(result.agreement)
    assert np.allclose(np.sum(data["confusion"]["metres"]), result.metres)
    assert data["scope"] == SCOPE_TEXT


# --------------------------------------------------------------------------
# the synthetic layered deposit: is the cross-validated figure honest?
# --------------------------------------------------------------------------

GROUP = ("SST", "SLT", "AND", "DIO", "GRD")
#: fixed before the first run: geology_contracts.md section 14 and the closeness to the truth
CV_TARGET = 0.85
CV_TO_TRUTH = 0.05


@pytest.fixture(scope="module")
def layered_logs(layered_small_db, layered_small_traces, layered_units_path):
    units = UnitDictionary.load(layered_units_path)
    table = layered_small_db.intervals["geology"]
    return table, units.map_codes(table["lith"]), layered_small_traces, units


@pytest.fixture(scope="module")
def layered_cv(layered_logs):
    table, mapped, traces, _ = layered_logs
    return cross_validate_units(table, mapped, traces, GROUP, folds=10, seed=0, step=1.0)


def weighted(same, length) -> float:
    return float(length[same].sum() / length.sum())


def test_layered_rows_against_pandas_and_run_time(layered_cv, layered_logs):
    table, mapped, _, _ = layered_logs
    result = layered_cv
    length = (table["depth_to"] - table["depth_from"]).groupby(mapped).sum()
    assert result.metres_without_trace == 0.0 and result.skipped_holes == ()
    np.testing.assert_allclose(result.confusion.sum(axis=1), length.reindex(list(GROUP)), rtol=1e-9)
    assert len(result.folds) == 10 and len(result.holes) == 64
    assert result.samples["hole_id"].nunique() == 64
    print(f"\nlayered-small, 10 folds: {result.seconds:.1f} s")
    assert result.seconds < 30.0  # the default must stay well under a minute


def test_layered_cv_against_the_truth(layered_cv, layered_logs, layered_small):
    """The cross-validated agreement next to what the truth says.

    (a) CV: fold models against the logs of the left-out holes.
    (b) the same predictions against the true unit at the same points.
    (c) the model of all holes against the truth along the same holes (it has seen them).
    (d) the model of all holes against the truth along the same holes moved 25 m east and
        25 m north, to the middle between the drill lines: the true accuracy between holes,
        which is what (a) tries to estimate without a truth.
    """
    table, mapped, traces, _ = layered_logs
    truth = layered_small.truth
    result = layered_cv
    samples = result.samples
    xyz = samples[["x", "y", "z"]].to_numpy()
    length = samples["length"].to_numpy()
    logged = samples["logged"].to_numpy(dtype=object)
    predicted = samples["predicted"].to_numpy(dtype=object)

    t0 = time.perf_counter()
    full = fit_stack(table, mapped, traces, [EventSpec("argmax", GROUP)])
    fit_seconds = time.perf_counter() - t0
    true_here = truth.labels(xyz, "bedrock")
    a = result.agreement
    b = weighted(predicted == true_here, length)
    c = weighted(predicted_names(full, xyz) == true_here, length)

    moved = xyz + np.array([25.0, 25.0, 0.0])
    cfg = layered_small.config
    inside = (
        (moved[:, 0] <= cfg.x_range[1] - 25.0)
        & (moved[:, 1] <= cfg.y_range[1] - 25.0)
        & (moved[:, 2] < truth.topo_z(moved[:, 0], moved[:, 1]))
    )
    true_moved = truth.labels(moved[inside], "bedrock")
    assert (true_moved != "AIR").all()
    d = weighted(predicted_names(full, moved[inside]) == true_moved, length[inside])
    per_unit = {
        u: weighted((predicted_names(full, moved[inside]) == u)[true_moved == u],
                    length[inside][true_moved == u])
        for u in GROUP
    }  # fmt: skip
    recall = result.per_unit.set_index("unit")["recall"]
    c_stats = result.contact
    print(
        f"\nlayered-small, {result.scheme} ({result.seconds:.1f} s; one fit of all holes "
        f"{fit_seconds:.1f} s), {result.metres:.0f} m scored:\n"
        f"  (a) hole-out agreement against the logs            {a:.4f}\n"
        f"  (b) the same predictions against the truth         {b:.4f}\n"
        f"  (c) full model against the truth, same holes       {c:.4f}\n"
        f"  (d) full model against the truth, between holes    {d:.4f} "
        f"({length[inside].sum():.0f} m)\n"
        f"  (a) - (d) = {a - d:+.4f}\n"
        "  recall per unit, hole-out / truth between holes: "
        + ", ".join(f"{u} {recall[u]:.3f} / {per_unit[u]:.3f}" for u in GROUP)
        + f"\n  contacts: {c_stats['matched']} of {c_stats['logged']} found, median "
        f"{c_stats['median_abs']:.2f} m, mean {c_stats['mean_abs']:.2f} m; "
        f"{c_stats['spurious']} spurious\n"
        + "\n".join(
            f"  {r.distance_from:5.1f} - {r.distance_to:5.1f} m: agreement {r.agreement:.4f}"
            for r in result.distance_table.itertuples(index=False)
        )
    )
    # the logs are the truth here (unknown intervals are not scored), so (a) = (b)
    assert weighted(logged == true_here, length) > 0.999
    assert abs(a - b) < 0.002
    assert c > 0.999  # a model scored on the holes it has seen says nothing: (c) is no estimate
    assert a >= CV_TARGET  # geology_contracts.md section 14
    assert abs(a - d) <= CV_TO_TRUTH  # the honest estimator check
    assert a <= d + 0.01  # hole-out leans to the pessimistic side, not the optimistic one


def test_layered_fault_blocks(layered_logs, layered_small_db, layered_small):
    """Fault blocks of the layered deposit (logged in the ore lens only): the hole-out
    agreement against the logs and against the true block at the same points."""
    table, mapped, traces, units = layered_logs
    assay = layered_small_db.intervals["assay"]
    blocks = block_codes(assay["wf"], r"^(FB\d)-")
    zones = table[(units.kinds_of(mapped) == "fault-zone").to_numpy()]
    builder = fault_network_builder([{"name": "F1", "between": ("FB1", "FB2")}], fault_zone=zones)
    result = cross_validate_fault_blocks(assay, blocks, traces, builder, folds=0)
    xyz = result.samples[["x", "y", "z"]].to_numpy()
    length = result.samples["length"].to_numpy()
    true = layered_small.truth.labels(xyz, "fault_block")
    against_truth = weighted(result.samples["predicted"].to_numpy(dtype=object) == true, length)
    coded = (assay["depth_to"] - assay["depth_from"])[blocks.notna().to_numpy()].sum()
    print(
        f"\nlayered-small fault blocks, {result.scheme}: {result.metres:.0f} m coded; hole-out "
        f"agreement {result.agreement:.4f} against the logs, {against_truth:.4f} against the "
        f"truth; {result.contact['logged']} logged crossings; {result.seconds:.1f} s"
    )
    assert result.metres == pytest.approx(coded)
    assert result.agreement == pytest.approx(against_truth, abs=1e-12)
    assert result.agreement >= 0.97


@pytest.mark.slow
def test_layered_leave_one_out(layered_logs, layered_cv):
    table, mapped, traces, _ = layered_logs
    result = cross_validate_units(table, mapped, traces, GROUP, folds=0)
    print(
        f"\nlayered-small leave-one-out: agreement {result.agreement:.4f} "
        f"(10 folds: {layered_cv.agreement:.4f}), {result.seconds:.1f} s"
    )
    assert len(result.folds) == 64
    assert result.metres == pytest.approx(layered_cv.metres)
    assert abs(result.agreement - layered_cv.agreement) < 0.03
