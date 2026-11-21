"""Resource classification: hand examples, rule order, drillhole support, smoothing, report."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition
from bakumine.core.estimation.classification import (
    ClassificationParams,
    ClassRule,
    HoleSupport,
    Smoothing,
    add_hole_support,
    classify,
    hole_support,
    smooth_classes,
)
from bakumine.core.report import report

JORC = [
    {"label": "Measured",
     "criteria": {"max_pass": 1, "min_samples": 8, "max_nearest_distance": 25.0}},
    {"label": "Indicated", "criteria": {"max_pass": 2, "min_samples": 4}},
    {"label": "Inferred"},
]  # fmt: skip


def row_model(n: int, **columns) -> BlockModel:
    """``n`` blocks of 10 m in a row along x, with float / int attributes from arrays."""
    model = BlockModel(BlockModelDefinition((0, 0, 0), (10, 10, 10), (n, 1, 1)))
    for name, values in columns.items():
        values = np.asarray(values)
        model.add_attribute(name, "int" if values.dtype.kind == "i" else "float")
        model.set(name, values)
    return model


def hand_model() -> BlockModel:
    nan = np.nan
    return row_model(
        7,
        #         0     1     2     3     4     5     6
        au=      [1.0,  2.0,  3.0,  4.0,  5.0,  nan,  6.0],
        au_pass= [1,    1,    2,    3,    1,    1,    1],
        au_ns=   [8,    7,    4,    1,    12,   12,   12],
        au_ndist=[25.0, 10.0, 60.0, 90.0, 25.1, 5.0,  nan],
    )  # fmt: skip


def labels(model: BlockModel, name: str = "res_class") -> list:
    return list(model.get(name, decode=True))


def params(categories=JORC, **more) -> ClassificationParams:
    return ClassificationParams.from_dict({"attribute": "au", "categories": categories, **more})


# -------------------------------------------------------------------------- the rules


def test_hand_example():
    model = hand_model()
    summary = classify(model, params())
    assert labels(model) == [
        "Measured",   # pass 1, 8 samples, 25 m: every limit met exactly
        "Indicated",  # 7 samples < 8
        "Indicated",  # pass 2, 4 samples
        "Inferred",   # pass 3
        "Indicated",  # 25.1 m > 25
        None,         # not estimated, whatever its diagnostics say
        "Indicated",  # nearest distance missing: cannot be Measured
    ]  # fmt: skip
    assert model.attribute("res_class").categories == ["Measured", "Indicated", "Inferred"]
    assert summary.to_dict() == {
        "attribute": "au",
        "output": "res_class",
        "n_blocks": 7,
        "n_estimated": 6,
        "n_classified": 6,
        "n_unclassified": 0,
        "n_not_estimated": 1,
        "n_smoothed": 0,
        "classes": [
            {"label": "Measured", "blocks": 1, "volume": 1000.0},
            {"label": "Indicated", "blocks": 4, "volume": 4000.0},
            {"label": "Inferred", "blocks": 1, "volume": 1000.0},
        ],
        "written": [],
    }
    # the class is an ordinary category attribute of the expression language
    assert model.evaluate('res_class == "Indicated"').sum() == 4


def test_rules_are_applied_in_order_and_first_match_wins():
    model = hand_model()
    # every estimated block meets "Any"; put first, it takes them all
    classify(model, params([{"label": "Any"}, *JORC]))
    assert labels(model) == ["Any"] * 5 + [None, "Any"]
    # Indicated before Measured: the Measured block meets Indicated first
    classify(model, params([JORC[1], JORC[0], JORC[2]]))
    assert labels(model)[0] == "Indicated" and "Measured" not in labels(model)
    assert model.attribute("res_class").categories == ["Indicated", "Measured", "Inferred"]


def test_estimated_blocks_that_meet_no_rule_stay_unclassified():
    model = hand_model()
    summary = classify(model, params(JORC[:2]))
    assert labels(model)[3] is None  # pass 3: neither Measured nor Indicated
    assert summary.n_unclassified == 1 and summary.n_classified == 5


def test_pass_zero_and_missing_values_meet_nothing():
    model = row_model(3, au=[1.0, 1.0, 1.0], au_pass=[0, 1, 2], au_kv=[0.1, np.nan, 0.1],
                      au_slope=[0.9, 0.9, np.nan])  # fmt: skip
    classify(model, params([{"label": "A", "criteria": {"max_pass": 2}}]))
    assert labels(model) == [None, "A", "A"]  # pass 0 is "no pass"
    classify(model, params([{"label": "A", "criteria": {"max_kriging_variance": 0.5}}]))
    assert labels(model) == ["A", None, "A"]
    classify(model, params([{"label": "A", "criteria": {"min_slope": 0.9}}]))
    assert labels(model) == ["A", "A", None]


def test_every_criterion_and_the_expression():
    model = row_model(
        2, au=[1.0, 1.0], au_pass=[1, 2], au_ns=[9, 3], au_dist=[10.0, 50.0],
        au_ndist=[5.0, 30.0], au_kv=[0.2, 0.8], au_slope=[0.9, 0.4], au_ke=[0.7, 0.1],
        au_nholes=[4, 1], au_dhole=[20.0, 80.0], samples=[9, 9],
    )  # fmt: skip
    limits = {"max_pass": 1, "min_samples": 8, "max_mean_distance": 20, "min_slope": 0.8,
              "max_nearest_distance": 10, "max_kriging_variance": 0.5, "min_efficiency": 0.5,
              "min_holes": 3, "max_hole_distance": 40}  # fmt: skip
    for criterion, value in limits.items():
        classify(model, params([{"label": "A", "criteria": {criterion: value}}]))
        assert labels(model) == ["A", None], criterion
    classify(model, params([{"label": "A", "where": "x > 10"}]))
    assert labels(model) == [None, "A"]
    # another attribute for a criterion
    classify(model, params([{"label": "A", "criteria": {"min_samples": 8}}],
                           inputs={"n_samples": "samples"}))  # fmt: skip
    assert labels(model) == ["A", "A"]


def test_free_text_labels_and_output_name():
    model = hand_model()
    cis = [{"label": "B", "criteria": {"max_pass": 1, "min_samples": 8}},
           {"label": "C1", "criteria": {"max_pass": 2}}, {"label": "Р1 (прогноз)"}]  # fmt: skip
    classify(model, params(cis, output="category_mn"))
    assert labels(model, "category_mn")[:4] == ["B", "C1", "C1", "Р1 (прогноз)"]
    assert "res_class" not in model


def test_bad_parameters_leave_the_model_unchanged():
    model = hand_model()
    before = list(model.attribute_names)
    for bad in (
        params(attribute="cu"),
        params([{"label": "A", "criteria": {"max_kriging_variance": 1}}]),  # no au_kv
        params([{"label": "A", "criteria": {"min_holes": 2}}]),  # no support, no composites
        params([{"label": "A", "where": "zz > 1"}]),  # unknown name in the expression
    ):
        with pytest.raises((ValueError, KeyError)):
            classify(model, bad)
        assert model.attribute_names == before
    model.add_attribute("flag", "int")
    with pytest.raises(ValueError, match="not a category"):
        classify(model, params(output="flag"))
    for bad in (
        {"attribute": "au", "categories": []},
        {"attribute": "au", "categories": [{"label": "A"}, {"label": "A"}]},
        {"attribute": "au", "categories": [{"label": ""}]},
        {"attribute": "au", "categories": [{"label": "A", "criteria": {"max_passes": 1}}]},
        {"attribute": "au", "categories": [{"label": "A", "criteria": {"max_pass": "x"}}]},
        {"attribute": "au", "categories": [{"label": "A"}], "output": "au"},
        {"attribute": "au", "categories": [{"label": "A"}], "smoothing": {"min_agree": 7}},
        {"attribute": "au", "categories": [{"label": "A"}], "hole_support": {"nth": 3}},
        {"attribute": "au", "categories": [{"label": "A"}], "extra": 1},
        {"categories": [{"label": "A"}]},
    ):
        with pytest.raises(ValueError):
            ClassificationParams.from_dict(bad)


def test_parameter_file_round_trip(tmp_path):
    full = ClassificationParams(
        "au",
        (
            ClassRule("Measured", {"max_pass": 1, "min_holes": 3}, 'rock == "ORE"'),
            ClassRule("Р1"),
        ),
        output="cls",
        hole_support=HoleSupport(50.0, 2, "bhid", "au"),
        smoothing=Smoothing(2, 5, 1),
        inputs={"n_samples": "n"},
    )
    path = full.save(tmp_path / "classes.json")
    assert ClassificationParams.load(path) == full
    assert json.loads(path.read_text(encoding="utf-8"))["bakumine_classification"] == 1
    assert "Р1" in path.read_text(encoding="utf-8")
    minimal = params()
    assert ClassificationParams.from_dict(minimal.to_dict()) == minimal
    assert minimal.smoothing is None and minimal.hole_support is None


# ------------------------------------------------------------------- drillhole support


def brute_force(points, xyz, holes, radius, nth):
    """Distance from each point to each hole = to its nearest composite; plain loops."""
    count, distance = [], []
    for p in points:
        nearest = {}
        for q, h in zip(xyz, holes, strict=True):
            d = float(np.sqrt(((p - q) ** 2).sum()))
            nearest[h] = min(d, nearest.get(h, np.inf))
        ordered = sorted(nearest.values())
        count.append(sum(d <= radius for d in ordered))
        distance.append(ordered[nth - 1] if len(ordered) >= nth else np.nan)
    return np.array(count), np.array(distance)


@pytest.mark.parametrize(
    ("n_holes", "per_hole", "radius", "nth"),
    [(12, 30, 40.0, 3), (2, 50, 500.0, 3), (40, 3, 15.0, 1), (5, 1, 1e6, 5), (1, 1, 10.0, 1),
     # a radius beyond everything and an nth beyond the holes: the hole-by-hole path
     (6, 100, 1e6, 3), (3, 400, 1e12, 2), (8, 60, 60.0, 50), (8, 60, 60.0, 8)],
)  # fmt: skip
def test_hole_support_against_brute_force(n_holes, per_hole, radius, nth):
    rng = np.random.default_rng(n_holes * 1000 + nth)
    collars = rng.uniform(0, 200, (n_holes, 2))
    rows = [
        (f"H{h}", collars[h, 0] + rng.normal(0, 1), collars[h, 1] + rng.normal(0, 1), -3.0 * i)
        for h in range(n_holes)
        for i in range(per_hole)
    ]
    composites = pd.DataFrame(rows, columns=["hole_id", "x", "y", "z"]).sample(
        frac=1, random_state=1
    )
    points = rng.uniform(-20, 220, (60, 3)) * [1, 1, -0.4]
    count, distance = hole_support(points, composites, radius=radius, nth=nth)
    expect_count, expect_distance = brute_force(
        points, composites[["x", "y", "z"]].to_numpy(), composites["hole_id"].to_numpy(),
        radius, nth,
    )  # fmt: skip
    np.testing.assert_array_equal(count, expect_count)
    np.testing.assert_allclose(distance, expect_distance, rtol=1e-12, equal_nan=True)


def test_hole_support_hand_example_and_missing_rows():
    # three vertical "holes" at x = 0, 30, 100; the point is at x = 10
    composites = pd.DataFrame(
        {
            "hole_id": ["A", "A", "B", "C", "C", None, "D"],
            "x": [0.0, 0.0, 30.0, 100.0, 100.0, 10.0, np.nan],
            "y": 0.0,
            "z": [0.0, -10.0, 0.0, 0.0, -50.0, 0.0, 0.0],
            "au": [1.0, 1.0, 1.0, np.nan, 1.0, 1.0, 1.0],
        }
    )
    point = [[10.0, 0.0, 0.0]]
    count, distance = hole_support(point, composites, radius=25.0, nth=3)
    assert count.tolist() == [2] and distance.tolist() == [90.0]  # A at 10, B at 20, C at 90
    count, distance = hole_support(point, composites, radius=25.0, nth=2)
    assert distance.tolist() == [20.0]
    # only composites with a grade: C's nearest is then the one 50 m down
    _, distance = hole_support(point, composites, radius=25.0, nth=3, value_column="au")
    assert distance[0] == pytest.approx(np.hypot(90.0, 50.0))
    count, distance = hole_support(point, composites, radius=25.0, nth=4)
    assert count.tolist() == [2] and np.isnan(distance[0])
    empty, none = hole_support(point, composites.iloc[:0], radius=25.0)
    assert empty.tolist() == [0] and np.isnan(none[0])
    with pytest.raises(ValueError, match="no column"):
        hole_support(point, composites.drop(columns="x"), radius=1.0)


def test_classify_with_hole_support():
    model = row_model(4, au=[1.0, 1.0, 1.0, np.nan], au_pass=[1, 1, 1, 1])
    # block centroids at x = 5, 15, 25, 35 (y = z = 5)
    composites = pd.DataFrame(
        {"hole_id": ["A", "B", "C"], "x": [5.0, 5.0, 25.0], "y": [5.0, 15.0, 5.0], "z": 5.0}
    )
    spec = params(
        [{"label": "M", "criteria": {"min_holes": 2, "max_hole_distance": 10.0}},
         {"label": "I", "criteria": {"min_holes": 2}}, {"label": "X"}],
        hole_support={"radius": 12.0, "nth": 2},
    )  # fmt: skip
    summary = classify(model, spec, composites)
    assert summary.written == ["au_nholes", "au_dhole"]
    # block 0: A at 0, B at 10 (C at 20)            -> 2 holes, 2nd at 10
    # block 1: A at 10, C at 10, B at sqrt(200)     -> 2 holes, 2nd at 10
    # block 2: C at 0, A at 20, B at sqrt(500)      -> 1 hole,  2nd at 20
    assert model["au_nholes"].tolist() == [2, 2, 1, 0]
    np.testing.assert_allclose(model["au_dhole"], [10.0, 10.0, 20.0, np.nan])
    assert labels(model) == ["M", "M", "X", None]
    # without composites the stored support is used as it is
    model.set("au_nholes", [0, 0, 5, 5])
    classify(model, spec)
    assert labels(model) == ["X", "X", "I", None]
    # the helper alone
    names = add_hole_support(model, composites, radius=12.0, nth=2,
                             count_name="au_nholes", distance_name="au_dhole")  # fmt: skip
    assert names == ("au_nholes", "au_dhole") and model["au_nholes"].tolist() == [2, 2, 1, 1]


def test_hole_support_sample_filter():
    from bakumine.core.estimation.block_estimation import ColumnFilter

    composites = pd.DataFrame(
        {
            "hole_id": ["A", "A", "B", "C"],
            "x": [0.0, 0.0, 30.0, 100.0],
            "y": 0.0,
            "z": [0.0, -40.0, 0.0, 0.0],
            "domain": ["WASTE", "MIN", "MIN", "OX"],
        }
    )
    point = [[10.0, 0.0, 0.0]]
    count, distance = hole_support(point, composites, radius=25.0, nth=2)
    assert count.tolist() == [2] and distance.tolist() == [20.0]  # A at 10, B at 20
    # only MIN composites: A's nearest is the one 40 m down, sqrt(10^2 + 40^2) away
    only = ColumnFilter("domain", ("MIN",))
    count, distance = hole_support(point, composites, radius=25.0, nth=2, sample_filter=only)
    assert count.tolist() == [1] and distance[0] == pytest.approx(np.hypot(10.0, 40.0))
    both = ColumnFilter("domain", ("MIN", "OX"))
    _, distance = hole_support(point, composites, radius=25.0, nth=3, sample_filter=both)
    assert distance.tolist() == [90.0]
    with pytest.raises(ValueError, match="not in the sample table"):
        hole_support(point, composites, radius=1.0, sample_filter=ColumnFilter("zone", ("A",)))


def test_hole_support_per_class_and_sample_filter():
    """B: 2 holes within 12 m. C: the 2nd hole within 21 m (its own radius and n-th hole)."""
    model = row_model(4, au=[1.0, 1.0, 1.0, np.nan], au_pass=[1, 1, 1, 1])
    # block centroids at x = 5, 15, 25, 35 (y = z = 5)
    composites = pd.DataFrame(
        {
            "hole_id": ["A", "B", "C", "W"],
            "x": [5.0, 5.0, 25.0, 25.0],
            "y": [5.0, 15.0, 5.0, 6.0],
            "z": 5.0,
            "domain": ["MIN", "MIN", "MIN", "WASTE"],
        }
    )
    spec = params(
        [
            {"label": "B", "criteria": {"min_holes": 2}},
            {"label": "C", "criteria": {"max_hole_distance": 21.0, "min_holes": 3},
             "hole_support": {"radius": 30.0, "nth": 2}},
            {"label": "P"},
        ],
        hole_support={"radius": 12.0, "nth": 3,
                      "sample_filter": {"column": "domain", "equals": "MIN"}},
    )  # fmt: skip
    # the class inherits the filter and overrides radius and nth
    own = spec.support_of(1)
    assert (own.radius, own.nth) == (30.0, 2)
    assert own.sample_filter == spec.hole_support.sample_filter
    assert spec.support_of(0) is spec.hole_support and spec.support_of(2) is spec.hole_support
    assert spec.support_names(1) == ("au_nholes_2", "au_dhole_2")
    written = ["au_nholes", "au_dhole", "au_nholes_2", "au_dhole_2"]
    assert spec.written_names() == ["res_class", *written]
    summary = classify(model, spec, composites)
    assert summary.written == written
    # MIN holes only (W is filtered out):
    # block 0 (x 5):  A 0, B 10, C 20              -> 12 m: 2 holes; 30 m: 3 holes, 2nd at 10
    # block 1 (x 15): A 10, C 10, B sqrt(200)      -> 12 m: 2 holes; 30 m: 3 holes, 2nd at 10
    # block 2 (x 25): C 0, A 20, B sqrt(500)=22.4  -> 12 m: 1 hole;  30 m: 3 holes, 2nd at 20
    assert model["au_nholes"].tolist() == [2, 2, 1, 0]
    np.testing.assert_allclose(model["au_dhole"], [20.0, np.sqrt(200.0), np.sqrt(500.0), np.nan])
    assert model["au_nholes_2"].tolist() == [3, 3, 3, 0]
    np.testing.assert_allclose(model["au_dhole_2"], [10.0, 10.0, 20.0, np.nan])
    assert labels(model) == ["B", "B", "C", None]
    # without the filter W is a 2nd hole within 12 m of block 2: it becomes B
    unfiltered = params(
        [c.to_dict() for c in spec.categories], hole_support={"radius": 12.0, "nth": 3}
    )
    other = row_model(4, au=[1.0, 1.0, 1.0, np.nan], au_pass=[1, 1, 1, 1])
    classify(other, unfiltered, composites)
    assert labels(other) == ["B", "B", "B", None]
    # a class alone may carry the support (no run-level hole_support)
    alone = params(
        [{"label": "C", "criteria": {"max_hole_distance": 15.0},
          "hole_support": {"radius": 30.0, "nth": 2,
                           "sample_filter": {"column": "domain", "in": ["MIN"]}}},
         {"label": "P"}]
    )  # fmt: skip
    third = row_model(4, au=[1.0, 1.0, 1.0, np.nan], au_pass=[1, 1, 1, 1])
    assert classify(third, alone, composites).written == ["au_nholes_1", "au_dhole_1"]
    assert labels(third) == ["C", "C", "P", None]
    # the stored per-class support is used when no composites are given
    third.set("au_dhole_1", [99.0, 1.0, 1.0, 1.0])
    classify(third, alone)
    assert labels(third) == ["P", "C", "C", None]
    # round trip, and old parameter files (no filter, no per-class support) are unchanged
    again = ClassificationParams.from_dict(json.loads(json.dumps(spec.to_dict())))
    assert again == spec and again.to_dict() == spec.to_dict()
    old = {"bakumine_classification": 1, "attribute": "au", "output": "res_class",
           "hole_support": {"radius": 50.0, "nth": 2},
           "categories": [{"label": "B", "criteria": {"max_pass": 1, "min_holes": 3}},
                          {"label": "P1"}], "smoothing": None}  # fmt: skip
    loaded = ClassificationParams.from_dict(old)
    assert loaded.hole_support == HoleSupport(50.0, 2) and loaded.hole_support.sample_filter is None
    assert loaded.categories[0].hole_support is None
    assert "sample_filter" not in loaded.to_dict()["hole_support"]
    assert "hole_support" not in loaded.to_dict()["categories"][0]
    assert loaded.written_names() == ["res_class", "au_nholes", "au_dhole"]
    for bad in (
        {"hole_support": {"radius": 10, "sample_filter": {"column": "domain"}}},
        {"hole_support": {"radius": 10, "sample_filter": "MIN"}},
        {"categories": [{"label": "B", "hole_support": {"radius": 10, "range": 3}}]},
        {"categories": [{"label": "B", "hole_support": {"nth": 2}}]},  # no radius anywhere
        {"categories": [{"label": "B", "hole_support": {"radius": -1}}]},
    ):
        with pytest.raises(ValueError):
            params(**{"categories": JORC, **bad}).support_of(0)


# --------------------------------------------------------------------------- smoothing


def cube(codes: np.ndarray, max_depth: int = 0) -> BlockModel:
    return BlockModel(
        BlockModelDefinition((0, 0, 0), (10, 10, 10), codes.shape, max_depth=max_depth)
    )


def grid_codes(model: BlockModel, codes: np.ndarray) -> np.ndarray:
    ijk = model.ijk()
    return codes[ijk[:, 0], ijk[:, 1], ijk[:, 2]]


def test_smoothing_replaces_an_isolated_block():
    grid = np.zeros((3, 3, 3), dtype=np.int64)
    grid[1, 1, 1] = 1  # one "Indicated" block inside "Measured": 6 neighbours of class 0
    model = cube(grid)
    codes = grid_codes(model, grid)
    out = smooth_classes(model, codes, 2, Smoothing())
    assert (out == 0).all()
    # a corner block has 3 neighbours: fewer than min_agree 4, so it stays
    grid[1, 1, 1], grid[0, 0, 0] = 0, 1
    codes = grid_codes(model, grid)
    np.testing.assert_array_equal(smooth_classes(model, codes, 2, Smoothing()), codes)
    assert (smooth_classes(model, codes, 2, Smoothing(min_agree=3)) == 0).all()


def test_smoothing_keeps_blocks_that_have_company_or_no_class():
    grid = np.zeros((3, 3, 3), dtype=np.int64)
    grid[1, 1, 1] = grid[1, 1, 2] = 1  # a pair: each has one neighbour of its own class
    model = cube(grid)
    codes = grid_codes(model, grid)
    np.testing.assert_array_equal(smooth_classes(model, codes, 2, Smoothing()), codes)
    # with max_same 1 the centre (5 others) goes; the face block (4 others + centre) too
    assert (smooth_classes(model, codes, 2, Smoothing(max_same=1)) == 0).all()
    # an unclassified block stays unclassified and does not vote
    grid[1, 1, 2] = 0
    grid[1, 1, 1] = -1
    codes = grid_codes(model, grid)
    np.testing.assert_array_equal(smooth_classes(model, codes, 2, Smoothing()), codes)
    grid[1, 1, 1] = 1
    grid[0, 1, 1] = grid[2, 1, 1] = grid[1, 0, 1] = -1  # only 3 classified neighbours left
    codes = grid_codes(model, grid)
    np.testing.assert_array_equal(smooth_classes(model, codes, 2, Smoothing()), codes)


def test_smoothing_ties_go_to_the_later_class_and_all_cells_use_the_old_state():
    # centre 0; neighbours: three of class 1, three of class 2 -> min_agree 3: class 2
    grid = np.full((3, 3, 3), -1, dtype=np.int64)
    grid[1, 1, 1] = 0
    grid[0, 1, 1] = grid[1, 0, 1] = grid[1, 1, 0] = 1
    grid[2, 1, 1] = grid[1, 2, 1] = grid[1, 1, 2] = 2
    model = cube(grid)
    out = smooth_classes(model, grid_codes(model, grid), 3, Smoothing(min_agree=3))
    assert out[model.definition.flat_index((1, 1, 1))] == 2
    # a row 0 1 0 1 0 along x with min_agree 2: both 1s flip, and the middle 0 flips too
    # because it is judged on the old state (its two neighbours were 1)
    row = np.array([0, 1, 0, 1, 0]).reshape(5, 1, 1)
    model = cube(row)
    out = smooth_classes(model, row.ravel(), 2, Smoothing(min_agree=2))
    assert out.tolist() == [0, 0, 1, 0, 0]
    out = smooth_classes(model, row.ravel(), 2, Smoothing(iterations=2, min_agree=2))
    assert out.tolist() == [0, 0, 0, 0, 0]


def test_smoothing_works_on_parent_cells_of_a_sub_blocked_model():
    grid = np.zeros((3, 3, 3), dtype=np.int64)
    model = cube(grid, max_depth=1)
    centre = int(model.definition.flat_index((1, 1, 1)))
    corner = int(model.definition.flat_index((0, 0, 0)))
    model.split(np.isin(model.parent, [centre, corner]))
    codes = np.zeros(model.n_blocks, dtype=np.int64)
    in_centre = np.flatnonzero(model.parent == centre)
    in_corner = np.flatnonzero(model.parent == corner)
    assert len(in_centre) == 8 and len(in_corner) == 8
    # centre cell: 5 sub-blocks of class 1, 2 of class 0, 1 unclassified -> cell class 1,
    # isolated among six class-0 cells -> its classified sub-blocks all become 0
    codes[in_centre[:5]] = 1
    codes[in_centre[7]] = -1
    # corner cell: 3 sub-blocks of class 1, 5 of class 0 -> cell class 0: nothing changes
    codes[in_corner[:3]] = 1
    out = smooth_classes(model, codes, 2, Smoothing())
    assert out[in_centre].tolist() == [0] * 7 + [-1]
    np.testing.assert_array_equal(out[in_corner], codes[in_corner])
    assert (np.delete(out, in_centre) == np.delete(codes, in_centre)).all()


def test_smoothing_is_optional_in_classify():
    n = 27
    grid = np.ones((3, 3, 3), dtype=np.int64)
    model = cube(grid)
    model.add_attribute("au", "float")
    model.set("au", np.ones(n))
    model.add_attribute("au_pass", "int")
    passes = np.ones(n, dtype=np.int64)
    centre = int(model.definition.flat_index((1, 1, 1)))
    passes[centre] = 2
    model.set("au_pass", passes)
    rules = [{"label": "M", "criteria": {"max_pass": 1}}, {"label": "I"}]
    plain = classify(model, params(rules))
    assert labels(model)[centre] == "I" and plain.n_smoothed == 0
    smoothed = classify(model, params(rules, smoothing={"iterations": 1}))
    assert labels(model) == ["M"] * n and smoothed.n_smoothed == 1
    assert [c["blocks"] for c in smoothed.classes] == [27, 0]


# ------------------------------------------------------------------------------ report


def test_report_by_class_sums_to_the_total_without_classes():
    rng = np.random.default_rng(7)
    model = BlockModel(BlockModelDefinition((0, 0, 0), (10, 10, 5), (8, 6, 4), max_depth=1))
    model.split(rng.random(model.n_blocks) < 0.2)
    n = model.n_blocks
    au = rng.lognormal(0, 1, n)
    au[rng.random(n) < 0.3] = np.nan  # not estimated
    model.add_attribute("au", "float", unit="g/t")
    model.set("au", au)
    model.add_attribute("au_pass", "int")
    model.set("au_pass", np.where(np.isnan(au), 0, rng.integers(1, 4, n)))
    model.add_attribute("au_ns", "int")
    model.set("au_ns", rng.integers(1, 16, n))
    model.add_attribute("au_ndist", "float")
    model.set("au_ndist", rng.uniform(0, 50, n))
    summary = classify(model, params(JORC[:2]))  # pass 3 stays unclassified
    assert summary.n_unclassified > 0 and summary.n_classified > 0

    options = {"grade_attributes": "au", "density": 2.7}
    grouped = report(model, group_by="res_class", **options)
    plain = report(model, **options)
    groups = grouped[grouped["row"] == "group"]
    total = plain[plain["row"] == "total"].iloc[0]
    numeric = [c for c in plain.columns if c not in ("row", "au_grade")]
    for column in numeric:
        assert groups[column].sum() == pytest.approx(total[column], rel=1e-12), column
    assert set(groups["res_class"]) == {"Measured", "Indicated", None}
    # the class rows hold exactly the classified blocks; the rest is the missing group
    by_class = groups.set_index(groups["res_class"].fillna("(none)"))
    for entry in summary.classes[:2]:
        assert by_class.loc[entry["label"], "blocks"] == entry["blocks"]
        assert by_class.loc[entry["label"], "volume"] == pytest.approx(entry["volume"])
    assert by_class.loc["(none)", "blocks"] == n - summary.n_classified
    # the summed columns include tonnes and the metal of au
    assert "tonnes" in numeric and any(c.startswith("au_") for c in numeric), numeric
