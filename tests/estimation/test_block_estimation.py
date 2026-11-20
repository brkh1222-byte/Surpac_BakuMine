"""Multi-pass estimation per domain on the block model (P4.T7).

The synthetic deposit supplies everything: a block model on the truth grid
(10 x 10 x 5 m), coded AIR / ORE / WASTE by centroid from the topography DTM
and the ore solid, and 2 m composites made inside the ``rock`` domains of the
geology table (column ``domain``). ORE grades are about 2 g/t, WASTE about
0.02 g/t, so a sample crossing the boundary would show.

The last section is the full synthetic run of Gate B; it prints its numbers.
"""

from __future__ import annotations

import dataclasses
import json
import time

import numpy as np
import pandas as pd
import pytest
import synthetic_ore as so

from bakumine.core.blockmodel import (
    AttributeConstraint,
    BlockModel,
    BlockModelDefinition,
    DtmConstraint,
    FileResolver,
    SolidConstraint,
)
from bakumine.core.drillholes.composite import IntervalDomains, composite
from bakumine.core.drillholes.importer import TableSource, import_drillholes
from bakumine.core.estimation.block_estimation import (
    ColumnFilter,
    EstimationDomain,
    EstimationPass,
    EstimationPlan,
    OutputNames,
    estimate_attribute,
    estimate_domains,
    load_passes,
    save_passes,
)
from bakumine.core.estimation.idw import inverse_distance
from bakumine.core.estimation.kriging import krige
from bakumine.core.estimation.nearest import nearest_neighbour
from bakumine.core.estimation.rotation import Ellipsoid
from bakumine.core.estimation.search import SearchParams
from bakumine.core.estimation.validation import compare_to_truth, global_comparison, swath_table

SYNTH = so.SYNTH
ORE = AttributeConstraint('rock == "ORE"')
WASTE = AttributeConstraint('rock == "WASTE"')
ORE_DENSITY = 2.85
FLOAT_KEYS = ("mean_distance", "nearest_distance", "variance", "slope", "efficiency")
ALL_SUFFIXES = ("", "_pass", "_ns", "_dist", "_ndist", "_kv", "_slope", "_ke")


# ------------------------------------------------------------------ set-up


@pytest.fixture(scope="module")
def resolver():
    return FileResolver(SYNTH)


@pytest.fixture(scope="module")
def composites() -> pd.DataFrame:
    """2 m composites of every domain: hole_id, x, y, z, domain, au, ..."""
    identity = {
        "collar": ["hole_id", "x", "y", "z", "max_depth"],
        "survey": ["hole_id", "depth", "dip", "azimuth"],
        "assay": ["hole_id", "depth_from", "depth_to", "au"],
        "geology": ["hole_id", "depth_from", "depth_to", "rock"],
    }
    maps = {name: {c: c for c in columns} for name, columns in identity.items()}
    db = import_drillholes(
        TableSource(SYNTH / "collar.csv", maps["collar"]),
        TableSource(SYNTH / "survey.csv", maps["survey"]),
        {
            "assay": TableSource(SYNTH / "assay.csv", maps["assay"]),
            "geology": TableSource(SYNTH / "geology.csv", maps["geology"]),
        },
    )
    comps = composite(
        db.intervals["assay"],
        db.desurvey(),
        length=so.COMPOSITE_LENGTH,
        mode="domain",
        domain=IntervalDomains(db.intervals["geology"], "rock"),
        attributes=["au"],
    )
    assert set(comps["domain"].dropna()) == {"ORE", "WASTE"}
    return comps


def coded_model(resolver, max_depth=0) -> BlockModel:
    """Truth grid, ``rock`` coded by centroid: AIR above the DTM, ORE inside the solid."""
    info = so.truth()["blocks"]
    d = BlockModelDefinition(info["origin"], info["size"], info["shape"], max_depth=max_depth)
    m = BlockModel(d)
    m.add_attribute("rock", "category", background="WASTE")
    m.set("rock", "AIR", DtmConstraint("topo", "above", outside=True).mask(m, resolver))
    m.set("rock", "ORE", SolidConstraint("ore_solid").mask(m, resolver))
    return m


@pytest.fixture(scope="module")
def coded(resolver):
    """A coded model that tests copy (never modify it)."""
    return coded_model(resolver)


def fresh(model: BlockModel) -> BlockModel:
    """Independent copy of a model: structure and attribute values."""
    m = BlockModel.from_arrays(model.definition, model.parent, model.level, model.sub)
    for name, spec in model.attributes.items():
        m.add_attribute(
            name, spec.kind, background=spec.background, dtype=spec.dtype,
            categories=spec.categories,
        )  # fmt: skip
        m[name][:] = model[name]
    return m


@pytest.fixture(scope="module")
def truth_columns(coded):
    """Truth table columns in the model's row order."""
    table = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    rows = coded.definition.flat_index(table[["i", "j", "k"]].to_numpy())
    np.testing.assert_allclose(coded.centroids()[rows], table[["x", "y", "z"]].to_numpy())
    out = {}
    for name in ("rock", "ore_fraction", "au_ore", "au"):
        column = np.empty(len(table), dtype=table[name].to_numpy().dtype)
        column[rows] = table[name].to_numpy()
        out[name] = column
    return out


def base_pass(method: str, factor: float = 1.0, **search) -> EstimationPass:
    options = {"max_samples": 16, "min_samples": 1, **search}
    kw = {}
    if method in ("ok", "sk"):
        kw["variogram"] = so.true_model()
    if method == "sk":
        kw["mean"] = 2.0
    return EstimationPass(
        method, so.true_range_ellipsoid(), SearchParams(**options), radius_factor=factor, **kw
    )


def three_passes(method: str = "ok") -> list[EstimationPass]:
    """Search = the true range ellipsoid x 0.25, x 0.5, x 1.5 (30, 60, 180 m major radius)."""
    p1 = dataclasses.replace(base_pass(method, 0.25, min_samples=6, max_per_hole=4), name="tight")
    return [
        p1,
        p1.scaled(0.5, search_min_samples=3, name="medium"),
        p1.scaled(1.5, search_min_samples=1, search_max_per_hole=None, name="wide"),
    ]


def array_estimate(p: EstimationPass, samples: pd.DataFrame, xyz, sizes, grid):
    """The array API called directly with the parameters of pass ``p``."""
    s = samples[np.isfinite(samples["au"].to_numpy())]
    coords = s[["x", "y", "z"]].to_numpy()
    values = s["au"].to_numpy()
    holes = s["hole_id"].to_numpy()
    if p.method == "nn":
        return nearest_neighbour(
            coords, values, xyz, p.search_ellipsoid,
            radius=p.search_params.radius, max_vertical=p.search_params.max_vertical,
        )  # fmt: skip
    block = dict(
        hole_ids=holes, block_size=sizes, discretisation=p.block_discretisation,
        grid_rotation=grid,
    )  # fmt: skip
    if p.method == "idw":
        return inverse_distance(
            coords, values, xyz, p.search_ellipsoid, p.search_params, power=p.power, **block
        )
    return krige(
        coords, values, xyz, p.variogram, p.search_ellipsoid, p.search_params,
        method="ordinary" if p.method == "ok" else "simple", mean=p.mean, **block,
    )  # fmt: skip


def assert_matches_array(model, attribute, rows, result, p, *, exact=True):
    """Model values of ``rows`` equal the array result (estimate and all diagnostics)."""
    ok = result.estimated & np.isfinite(result.estimate)
    compare = np.testing.assert_array_equal if exact else _close

    compare(model.get(attribute, rows), np.where(ok, result.estimate, np.nan))
    np.testing.assert_array_equal(model.get(attribute + "_pass", rows) > 0, ok)
    np.testing.assert_array_equal(
        model.get(attribute + "_ns", rows), np.where(ok, result.n_samples, 0)
    )
    names = {"mean_distance": "_dist", "nearest_distance": "_ndist"}
    if p.is_kriging:
        names |= {"variance": "_kv", "slope": "_slope", "efficiency": "_ke"}
    for key, suffix in names.items():
        compare(model.get(attribute + suffix, rows), np.where(ok, getattr(result, key), np.nan))


def _close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12, equal_nan=True)


def snapshot(model: BlockModel):
    return (
        model.parent.copy(), model.level.copy(), model.sub.copy(), model.attribute_names,
        {name: model[name].copy() for name in model.attribute_names},
        {name: dataclasses.asdict(spec) for name, spec in model.attributes.items()},
    )  # fmt: skip


def assert_same_model(model: BlockModel, before) -> None:
    parent, level, sub, names, columns, specs = before
    np.testing.assert_array_equal(model.parent, parent)
    np.testing.assert_array_equal(model.level, level)
    np.testing.assert_array_equal(model.sub, sub)
    assert model.attribute_names == names
    for name in names:
        assert model[name].dtype == columns[name].dtype
        assert model[name].tobytes() == columns[name].tobytes(), name
    now = {n: dataclasses.asdict(s) for n, s in model.attributes.items()}
    assert now.keys() == specs.keys()
    for name, spec in specs.items():
        for key, value in spec.items():
            other = now[name][key]
            both_nan = isinstance(value, float) and value != value and other != other
            assert both_nan or other == value, (name, key)


# ------------------------------------------------------------------ parameters and files


def test_pass_validation_and_scaling():
    ell = Ellipsoid(100.0, 2.0, 4.0, (30.0, 10.0, 0.0))
    with pytest.raises(ValueError, match="method"):
        EstimationPass("kriging", ell)
    with pytest.raises(ValueError, match="variogram"):
        EstimationPass("ok", ell)
    with pytest.raises(ValueError, match="mean"):
        EstimationPass("sk", ell, variogram=so.true_model())
    with pytest.raises(ValueError, match="radius_factor"):
        EstimationPass("nn", ell, radius_factor=0.0)
    with pytest.raises(ValueError, match="discretisation"):
        EstimationPass("idw", ell, discretisation=(2, 0, 1))

    p = EstimationPass("idw", ell, SearchParams(12, 4, radius=80.0), radius_factor=2.0)
    assert p.search_ellipsoid.major_radius == 200.0
    assert p.search_ellipsoid.radii == (200.0, 100.0, 50.0)
    np.testing.assert_array_equal(p.search_ellipsoid.rotation, ell.rotation)
    assert p.search_params.radius == 160.0 and p.search_params.max_samples == 12
    assert p.block_discretisation == (4, 4, 2)
    assert EstimationPass("nn", ell, discretisation=(3, 3, 3)).block_discretisation == (1, 1, 1)
    q = p.scaled(3.0, search_min_samples=1, power=3.0)
    assert (q.radius_factor, q.search.min_samples, q.search.max_samples, q.power) == (3, 1, 12, 3)
    assert q.ellipsoid == p.ellipsoid


@pytest.mark.parametrize("method", ["nn", "idw", "ok", "sk"])
def test_pass_json_round_trip(method, tmp_path):
    p = base_pass(method, 1.5, min_samples=3, max_per_hole=2, sectors="octant", max_per_sector=4,
                  max_vertical=40.0)  # fmt: skip
    p = dataclasses.replace(p, discretisation=(3, 3, 2), name="p", power=2.5)
    text = json.dumps(p.to_dict())
    assert EstimationPass.from_dict(json.loads(text)) == p
    passes = [p, p.scaled(2.0, search_min_samples=1)]
    save_passes(passes, tmp_path / "passes.json")
    assert load_passes(tmp_path / "passes.json") == passes
    with pytest.raises(ValueError, match="unknown pass field"):
        EstimationPass.from_dict({**p.to_dict(), "radius": 3})
    with pytest.raises(ValueError, match="unknown search option"):
        EstimationPass.from_dict({**p.to_dict(), "search": {"max_smaples": 3}})


def test_sample_filter_forms(coded, composites):
    domain = composites["domain"].to_numpy()
    forms = [
        ("domain", "ORE"),
        {"column": "domain", "equals": "ORE"},
        {"column": "domain", "in": ["ORE"]},
        ColumnFilter("domain", ("ORE",)),
        domain == "ORE",
        lambda table: (table["domain"] == "ORE").to_numpy(),
    ]
    results = []
    for sample_filter in forms:
        m = fresh(coded)
        s = estimate_attribute(m, composites, attribute="au", passes=[base_pass("idw")],
                               block_constraint=ORE, sample_filter=sample_filter)  # fmt: skip
        assert s.domains[0].n_samples == int(((domain == "ORE") & composites["au"].notna()).sum())
        results.append(m.get("au").copy())
    for other in results[1:]:
        np.testing.assert_array_equal(other, results[0])
    with pytest.raises(ValueError, match="not in the sample table"):
        estimate_attribute(fresh(coded), composites, attribute="au", passes=[base_pass("nn")],
                           sample_filter=("zone", 1))  # fmt: skip
    with pytest.raises(ValueError, match="one boolean per row"):
        estimate_attribute(fresh(coded), composites, attribute="au", passes=[base_pass("nn")],
                           sample_filter=np.ones(3, dtype=bool))  # fmt: skip
    with pytest.raises(ValueError, match="no column"):
        estimate_attribute(fresh(coded), composites, attribute="grade",
                           passes=[base_pass("nn")])  # fmt: skip


def test_plan_json_round_trip_reproduces_estimates(resolver, coded, composites, tmp_path):
    """A parameter file with constraints by name (solid, DTM, attribute) reruns identically."""
    ore_by_geometry = SolidConstraint("ore_solid") & DtmConstraint("topo", "below")
    waste_passes = [base_pass("idw", 0.5, min_samples=4), base_pass("nn", 2.0)]
    plan = EstimationPlan(
        attribute="au",
        value_column="au",
        domains=[
            EstimationDomain("ORE", three_passes("ok"), ore_by_geometry, ("domain", "ORE"), 12.0),
            EstimationDomain(
                "WASTE", waste_passes,
                AttributeConstraint('rock == "WASTE" and z > 1100'), ("domain", "WASTE"),
            ),
        ],
        outputs=OutputNames(variance="_var", slope=None),
        clip_negative=True,
        capping=1.0,
    )  # fmt: skip
    path = tmp_path / "estimate.json"
    plan.save(path)
    document = json.loads(path.read_text())
    assert document["bakumine_estimation"] == 1
    assert document["domains"][0]["block_constraint"]["type"] == "and"
    assert document["domains"][0]["sample_filter"] == {"column": "domain", "equals": "ORE"}
    assert document["domains"][0]["passes"][0]["variogram"] == so.true_model().to_dict()
    loaded = EstimationPlan.load(path)
    assert loaded.to_dict() == plan.to_dict()
    assert loaded.domains[0].passes == plan.domains[0].passes

    a, b = fresh(coded), fresh(coded)
    sa = plan.run(a, composites, resolver=resolver)
    sb = loaded.run(b, composites, resolver=resolver)
    assert a.attribute_names == b.attribute_names
    assert "au_var" in a and "au_slope" not in a and "au_ke" in a
    for name in a.attribute_names:
        assert a[name].tobytes() == b[name].tobytes(), name
    da, db = sa.to_dict(), sb.to_dict()
    for d in (da, db):
        d.pop("seconds")
        for dom in d["domains"]:
            dom.pop("seconds")
            for p in dom["passes"]:
                p.pop("seconds")
    assert da == db
    json.dumps(sa.to_dict())  # the summary itself is JSON-serialisable
    assert sa.domain("ORE").n_capped > 0 and sa.domain("WASTE").n_capped > 0
    assert sa.domain("ORE").sample_max == 12.0 and sa.domain("WASTE").sample_max == 1.0
    assert sa.domain("ORE").n_estimated > 4000 and sa.domain("WASTE").n_estimated > 10_000

    # masks and callables cannot be written to a parameter file
    with pytest.raises(TypeError, match="Constraint"):
        EstimationDomain("x", [base_pass("nn")], np.ones(coded.n_blocks, dtype=bool)).to_dict()
    with pytest.raises(TypeError, match="sample filter"):
        EstimationDomain("x", [base_pass("nn")], None, lambda t: t["au"] > 1).to_dict()


# ------------------------------------------------------------------ domains (plan criterion 1)


@pytest.fixture(scope="module")
def two_domains():
    def build(method):
        return [
            EstimationDomain("ORE", [base_pass(method, 1.5)], ORE, ("domain", "ORE")),
            EstimationDomain("WASTE", [base_pass(method, 1.5)], WASTE, ("domain", "WASTE")),
        ]

    return build


@pytest.mark.parametrize("method", ["nn", "idw"])
def test_estimates_stay_inside_the_range_of_their_own_domain(
    method, coded, composites, two_domains
):
    """NN and IDW weights are convex: an estimate cannot leave the range of its samples.

    The ORE and WASTE sample ranges overlap (a few WASTE composites above
    the lowest ORE one), so the test also checks the two means, which no
    mixing could leave three orders of magnitude apart.
    """
    m = fresh(coded)
    summary = estimate_domains(m, composites, attribute="au", domains=two_domains(method))
    au = m.get("au")
    for name, constraint in (("ORE", ORE), ("WASTE", WASTE)):
        values = composites.loc[composites["domain"] == name, "au"].dropna().to_numpy()
        block = au[constraint.mask(m)]
        block = block[np.isfinite(block)]
        d = summary.domain(name)
        assert d.n_estimated == len(block) > 0.5 * d.n_constraint
        tolerance = 1e-9 * values.max()
        assert block.min() >= values.min() - tolerance
        assert block.max() <= values.max() + tolerance
        assert (d.sample_min, d.sample_max) == (values.min(), values.max())
    ore_values = composites.loc[composites["domain"] == "ORE", "au"]
    waste_values = composites.loc[composites["domain"] == "WASTE", "au"]
    assert ore_values.mean() > 50 * waste_values.mean()
    assert summary.domain("ORE").mean_estimate > 1.5
    assert summary.domain("WASTE").mean_estimate < 0.05
    # nothing outside the two constraints (AIR) was touched
    air = ~(ORE.mask(m) | WASTE.mask(m))
    assert air.sum() > 10_000 and np.isnan(au[air]).all()
    assert (m.get("au_pass")[air] == 0).all()


@pytest.mark.parametrize("method", ["nn", "idw", "ok", "sk"])
def test_other_domains_samples_have_no_influence(method, coded, composites, two_domains):
    """Move and rescale every sample of the other domain: this domain's blocks do not change.

    Every output is compared bit for bit, so not even a neighbour count or a
    distance depends on a sample of the other domain.
    """
    reference = fresh(coded)
    estimate_domains(reference, composites, attribute="au", domains=two_domains(method))
    rng = np.random.default_rng(7)
    for own, constraint in (("ORE", ORE), ("WASTE", WASTE)):
        other = (composites["domain"] != own).to_numpy()
        perturbed = composites.copy()
        perturbed.loc[other, "au"] = perturbed.loc[other, "au"] * 1000.0 + 500.0
        own_xyz = composites.loc[~other, ["x", "y", "z"]].to_numpy()
        # put the other domain's samples right on top of this domain's
        take = rng.integers(0, len(own_xyz), int(other.sum()))
        perturbed.loc[other, ["x", "y", "z"]] = own_xyz[take] + rng.normal(0, 0.5, (len(take), 3))
        m = fresh(coded)
        estimate_domains(m, perturbed, attribute="au", domains=two_domains(method))
        rows = constraint.mask(m)
        assert np.isfinite(reference.get("au", rows)).sum() > 4000
        for name in reference.attribute_names:
            assert m.get(name, rows).tobytes() == reference.get(name, rows).tobytes(), name
        # ... while the perturbed domain itself did change (the test has teeth)
        other_rows = (WASTE if own == "ORE" else ORE).mask(m)
        assert not np.array_equal(
            m.get("au", other_rows), reference.get("au", other_rows), equal_nan=True
        )


def test_overlapping_domains_first_one_wins(coded, composites):
    m = fresh(coded)
    everything_below_topo = AttributeConstraint('rock != "AIR"')
    summary = estimate_domains(
        m, composites, attribute="au",
        domains=[
            {"name": "ORE", "passes": [base_pass("idw", 1.5)], "block_constraint": ORE,
             "sample_filter": ("domain", "ORE")},
            {"name": "REST", "passes": [base_pass("nn", 1.5)],
             "block_constraint": everything_below_topo, "sample_filter": ("domain", "WASTE")},
        ],
    )  # fmt: skip
    ore = ORE.mask(m)
    assert summary.domain("ORE").n_estimated == ore.sum()
    rest = summary.domain("REST")
    assert rest.n_constraint == everything_below_topo.mask(m).sum()
    assert rest.n_targeted == rest.n_constraint - ore.sum()
    assert m.get("au", ore).min() > 0.2  # still ORE grades: not re-estimated from WASTE


# ------------------------------------------------------------------ passes (plan criterion 2)


@pytest.fixture(scope="module")
def pass_runs(coded, composites):
    """ORE estimated with the first 1, 2 and 3 passes of the plan (three models)."""
    passes = three_passes("ok")
    out = []
    for n in (1, 2, 3):
        m = fresh(coded)
        s = estimate_attribute(m, composites, attribute="au", passes=passes[:n],
                               block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
        out.append((m, s))
    return passes, out


def test_pass_attribute_is_consistent(pass_runs, composites):
    passes, runs = pass_runs
    previous = None
    for n, (m, summary) in enumerate(runs, start=1):
        ore = ORE.mask(m)
        au, number = m.get("au"), m.get("au_pass")
        assert m.attribute("au_pass").kind == "int"
        # a pass number exactly where there is an estimate, 0 (background) elsewhere
        np.testing.assert_array_equal(number > 0, np.isfinite(au))
        assert set(np.unique(number)) <= set(range(n + 1))
        # blocks outside the constraint: every output still at its background
        for suffix in ALL_SUFFIXES:
            column = m.get("au" + suffix)[~ore]
            spec = m.attribute("au" + suffix)
            assert (np.isnan(column) if spec.kind == "float" else column == 0).all(), suffix
        # diagnostics exist exactly where the estimate does
        for suffix in ALL_SUFFIXES[2:]:
            column = m.get("au" + suffix)
            has = np.isfinite(column) if column.dtype.kind == "f" else column > 0
            np.testing.assert_array_equal(has, np.isfinite(au), err_msg=suffix)
        # the summary counts what the attribute says
        d = summary.domains[0]
        assert d.n_constraint == d.n_targeted == ore.sum()
        assert [p.number for p in d.passes] == list(range(1, n + 1))
        assert [p.n_estimated for p in d.passes] == [
            int((number == k).sum()) for k in range(1, n + 1)
        ]
        for a, b in zip(d.passes, d.passes[1:], strict=False):
            assert b.n_targeted == a.n_unestimated
        assert d.passes[-1].n_unestimated == d.n_unestimated == int((ore & (number == 0)).sum())
        if previous is not None:
            before_au, before_number = previous
            done = before_number > 0
            # adding a pass changes no block estimated before, in any output
            for suffix in ALL_SUFFIXES:
                was = runs[n - 2][0].get("au" + suffix)[done]
                assert m.get("au" + suffix)[done].tobytes() == was.tobytes(), suffix
            # ... and estimates a superset
            assert (number > 0)[done].all()
            assert (number > 0).sum() > done.sum()
            assert (number[~done & (number > 0)] == n).all()
        previous = (au.copy(), number.copy())
    m, summary = runs[-1]
    counts = [p.n_estimated for p in summary.domains[0].passes]
    assert all(c > 0 for c in counts), counts  # every pass of the plan does something


def test_each_pass_equals_the_array_api_with_that_pass(pass_runs, composites):
    """Blocks with pass number p hold exactly what pass p's parameters give."""
    passes, runs = pass_runs
    m, _ = runs[-1]
    ore_samples = composites[composites["domain"] == "ORE"]
    rows = np.flatnonzero(ORE.mask(m))
    size = m.definition.block_size
    for number, p in enumerate(passes, start=1):
        result = array_estimate(p, ore_samples, m.centroids(rows), size, m.definition.rotation)
        mine = m.get("au_pass", rows) == number
        assert mine.sum() > 0 and result.estimated[mine].all()
        np.testing.assert_array_equal(m.get("au", rows)[mine], result.estimate[mine])
        np.testing.assert_array_equal(m.get("au_kv", rows)[mine], result.variance[mine])
        np.testing.assert_array_equal(m.get("au_ns", rows)[mine], result.n_samples[mine])
        # a block left to a later pass was one this pass could not estimate
        later = m.get("au_pass", rows) > number
        assert not result.estimated[later].any()


def test_second_call_fills_only_missing_blocks_and_overwrite_redoes_all(coded, composites):
    passes = three_passes("idw")
    kw = dict(attribute="au", block_constraint=ORE, sample_filter=("domain", "ORE"))
    m = fresh(coded)
    estimate_attribute(m, composites, passes=passes[:1], **kw)
    first = {s: m.get("au" + s).copy() for s in ALL_SUFFIXES[:5]}
    done = first["_pass"] > 0
    ore = ORE.mask(m)
    # a second call with a wider pass keeps everything already estimated
    s2 = estimate_attribute(m, composites, passes=passes[2:], **kw)
    assert s2.domains[0].n_targeted == int((ore & ~done).sum())
    for suffix, was in first.items():
        assert m.get("au" + suffix)[done].tobytes() == was[done].tobytes()
    assert np.isfinite(m.get("au", ore)).all()
    assert (m.get("au_pass")[ore & ~done] == 1).all()  # numbered within its own call
    # overwrite: everything in the constraint is redone with the new plan
    s3 = estimate_attribute(m, composites, passes=passes[:1], overwrite=True, **kw)
    assert s3.domains[0].n_targeted == ore.sum()
    for suffix, was in first.items():
        assert m.get("au" + suffix).tobytes() == was.tobytes(), suffix
    # values that were there but cannot be estimated now are reset, not left stale
    assert np.isnan(m.get("au", ore & ~done)).all() and (ore & ~done).sum() > 0


# ------------------------------------------------------------------ equality with the array API


@pytest.mark.parametrize("method", ["nn", "idw", "ok", "sk"])
def test_single_pass_equals_array_api(method, coded, composites):
    p = base_pass(method, 0.5, min_samples=4, max_per_hole=3)
    m = fresh(coded)
    summary = estimate_attribute(m, composites, attribute="au", passes=[p],
                                 block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
    rows = np.flatnonzero(ORE.mask(m))
    result = array_estimate(
        p, composites[composites["domain"] == "ORE"], m.centroids(rows),
        m.definition.block_size, m.definition.rotation,
    )  # fmt: skip
    ok = result.estimated
    assert 0.5 * len(rows) < ok.sum() and summary.n_estimated == ok.sum()
    if method != "nn":
        assert ok.sum() < len(rows)  # some blocks are left unestimated: both paths agree where
    assert_matches_array(m, "au", rows, result, p)
    assert ("au_kv" in m) == p.is_kriging
    d = summary.domains[0]
    assert d.mean_estimate == pytest.approx(result.estimate[ok].mean(), rel=1e-12)
    assert d.passes[0].mean_samples == pytest.approx(result.n_samples[ok].mean(), rel=1e-12)
    assert (d.min_estimate, d.max_estimate) == (result.estimate[ok].min(),
                                                result.estimate[ok].max())  # fmt: skip


def test_declustered_mean_is_the_nn_mean_of_the_estimated_blocks(coded, composites):
    m = fresh(coded)
    p = base_pass("ok", 1.5)
    summary = estimate_attribute(m, composites, attribute="au", passes=[p],
                                 block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
    rows = np.flatnonzero(ORE.mask(m))
    ore = composites[(composites["domain"] == "ORE") & composites["au"].notna()]
    nn = nearest_neighbour(ore[["x", "y", "z"]].to_numpy(), ore["au"].to_numpy(),
                           m.centroids(rows), p.ellipsoid, radius=1e9)  # fmt: skip
    d = summary.domains[0]
    assert d.n_estimated == len(rows)
    assert d.nn_mean == pytest.approx(nn.estimate.mean(), rel=1e-12)
    check = global_comparison(m.get("au", rows), nn.estimate, ore["au"].to_numpy())
    assert d.estimate_vs_nn == pytest.approx(check.block_vs_nn, rel=1e-9)
    assert d.sample_mean == pytest.approx(check.naive_mean, rel=1e-12)
    off = estimate_attribute(fresh(coded), composites, attribute="au", passes=[p],
                             block_constraint=ORE, sample_filter=("domain", "ORE"),
                             declustered_check=False)  # fmt: skip
    assert np.isnan(off.domains[0].nn_mean) and off.to_dict()["domains"][0]["nn_mean"] is None


# ------------------------------------------------------------------ sub-blocks and rotation


@pytest.fixture(scope="module")
def sub_blocked(resolver):
    """Truth grid with max_depth 2; the parents cut by the ore boundary split by hand."""
    m = coded_model(resolver, max_depth=2)
    solid = SolidConstraint("ore_solid")
    d = m.definition
    # parents whose 26-neighbourhood is not all on one side of the boundary
    inside = solid.mask(m, resolver).reshape(d.shape)
    grown = np.zeros_like(inside)
    shrunk = np.ones_like(inside)
    pad = np.pad(inside, 1, mode="edge")
    for di in range(3):
        for dj in range(3):
            for dk in range(3):
                view = pad[di : di + d.shape[0], dj : dj + d.shape[1], dk : dk + d.shape[2]]
                grown |= view
                shrunk &= view
    boundary = (grown & ~shrunk).reshape(-1)
    m.split(boundary)
    m.split(solid.mask(m, resolver) & (m.level == 1) & (m.sub[:, 0] == 0))  # some go deeper
    below = m.get("rock", decode=True) != "AIR"
    m.set("rock", "WASTE", below)
    m.set("rock", "ORE", solid.mask(m, resolver) & below)
    assert set(np.unique(m.level)) == {0, 1, 2}
    return m


def test_parent_support_assigns_one_value_per_parent(sub_blocked, composites):
    m = fresh(sub_blocked)
    p = base_pass("ok", 1.5)
    summary = estimate_attribute(m, composites, attribute="au", passes=[p],
                                 block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
    assert summary.domains[0].support == "parent"  # the default for a sub-blocked model
    ore = ORE.mask(m)
    rows = np.flatnonzero(ore)
    parents, inverse = np.unique(m.parent[rows], return_inverse=True)
    d = m.definition
    centroids = d.ijk_to_world(d.unflat_index(parents))
    result = array_estimate(p, composites[composites["domain"] == "ORE"], centroids,
                            d.block_size, d.rotation)  # fmt: skip
    assert result.estimated.all()
    # every ORE block holds its parent's value, estimated at parent support
    np.testing.assert_array_equal(m.get("au", rows), result.estimate[inverse])
    np.testing.assert_array_equal(m.get("au_kv", rows), result.variance[inverse])
    np.testing.assert_array_equal(m.get("au_ns", rows), result.n_samples[inverse])
    np.testing.assert_array_equal(m.get("au_dist", rows), result.mean_distance[inverse])
    # parents with several ORE children, of different sizes, and with non-ORE children too
    children = np.bincount(inverse)
    assert (children > 1).sum() > 500
    mixed = np.isin(m.parent, parents) & ~ore
    assert mixed.sum() > 500
    for suffix in ALL_SUFFIXES:
        column = m.get("au" + suffix)[~ore]
        assert (np.isnan(column) if column.dtype.kind == "f" else column == 0).all()
    one = summary.domains[0].passes[0]
    assert (one.n_estimated, one.n_units_estimated) == (len(rows), len(parents))
    assert len(parents) < len(rows)
    # the mean is volume-weighted over the blocks written
    v = m.volumes(rows)
    assert summary.domains[0].mean_estimate == pytest.approx(
        (m.get("au", rows) * v).sum() / v.sum(), rel=1e-12
    )


@pytest.mark.parametrize("method", ["idw", "ok", "nn"])
def test_block_support_uses_each_sub_blocks_own_size(method, sub_blocked, composites):
    m = fresh(sub_blocked)
    p = base_pass(method, 1.5)
    summary = estimate_attribute(m, composites, attribute="au", passes=[p], support="block",
                                 block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
    rows = np.flatnonzero(ORE.mask(m))
    sizes = m.sizes(rows)
    assert len(np.unique(sizes, axis=0)) == 3  # parents, half and quarter blocks
    ore_samples = composites[composites["domain"] == "ORE"]
    result = array_estimate(p, ore_samples, m.centroids(rows), sizes, m.definition.rotation)
    assert_matches_array(m, "au", rows, result, p)
    assert summary.domains[0].n_estimated == len(rows)
    if method != "nn":
        # ... and the size matters: the parent size for everything gives other numbers
        wrong = array_estimate(p, ore_samples, m.centroids(rows), m.definition.block_size,
                               m.definition.rotation)  # fmt: skip
        small = (sizes != np.array(m.definition.block_size)).any(axis=1)
        assert (wrong.estimate[small] != result.estimate[small]).mean() > 0.9
        np.testing.assert_array_equal(wrong.estimate[~small], result.estimate[~small])


@pytest.mark.parametrize("method", ["idw", "ok"])
def test_rotated_model_passes_its_rotation_as_grid_rotation(method, composites):
    ore = composites[(composites["domain"] == "ORE") & composites["au"].notna()]
    centre = ore[["x", "y", "z"]].to_numpy().mean(axis=0)
    probe = BlockModelDefinition((0, 0, 0), (40.0, 10.0, 5.0), (8, 20, 16),
                                 angles=(40.0, 0.0, 25.0), max_depth=1)  # fmt: skip
    origin = centre - probe.rotation @ (np.array(probe.extent) / 2)
    d = dataclasses.replace(probe, origin=tuple(origin))
    assert d.is_rotated
    p = base_pass(method, 1.5)

    m = BlockModel(d)
    estimate_attribute(m, ore, attribute="au", passes=[p])
    xyz = m.centroids()
    result = array_estimate(p, ore, xyz, d.block_size, d.rotation)
    assert 0.3 < result.estimated.mean() < 1.0  # some blocks are beyond the search
    assert_matches_array(m, "au", np.arange(m.n_blocks), result, p)
    # the rotation is not a formality: unrotated discretisation gives other values
    unrotated = array_estimate(p, ore, xyz, d.block_size, None)
    both = result.estimated
    assert (unrotated.estimate[both] != result.estimate[both]).mean() > 0.9

    # sub-blocked and rotated, own support
    m = BlockModel(d)
    m.split(np.arange(0, m.n_blocks, 3))
    estimate_attribute(m, ore, attribute="au", passes=[p], support="block")
    result = array_estimate(p, ore, m.centroids(), m.sizes(), d.rotation)
    assert_matches_array(m, "au", np.arange(m.n_blocks), result, p)
    # ... and parent support: children share the value of the rotated parent block
    estimate_attribute(m, ore, attribute="au", passes=[p], overwrite=True)
    parents = d.ijk_to_world(d.unflat_index(np.arange(d.n_parents)))
    result = array_estimate(p, ore, parents, d.block_size, d.rotation)
    np.testing.assert_array_equal(
        m.get("au"), np.where(result.estimated, result.estimate, np.nan)[m.parent]
    )


# ------------------------------------------------------------------ cancel, progress, chunks


def test_cancel_leaves_the_model_bit_identical(coded, composites):
    m = fresh(coded)
    # a model that already holds results, so that "unchanged" is not trivially "empty"
    estimate_attribute(m, composites, attribute="au", passes=three_passes("ok")[:1],
                       block_constraint=ORE, sample_filter=("domain", "ORE"))  # fmt: skip
    before = snapshot(m)
    domains = [
        EstimationDomain("ORE", three_passes("ok"), ORE, ("domain", "ORE")),
        EstimationDomain("WASTE", [base_pass("idw", 1.5)], WASTE, ("domain", "WASTE")),
    ]
    kw = dict(attribute="au", domains=domains, chunk_size=1000, overwrite=True,
              outputs=OutputNames(variance="_variance"))  # fmt: skip

    polls = []
    full = fresh(m)
    estimate_domains(full, composites, cancel=lambda: bool(polls.append(1)), **kw)
    n_polls = len(polls)
    assert n_polls > 100  # polled between chunks
    assert "au_variance" in full and not np.array_equal(full["au"], m["au"], equal_nan=True)

    for stop_at in (1, 2, 5, n_polls // 2, n_polls - 1, n_polls):
        count = [0]
        fractions = []

        def cancel(count=count, stop_at=stop_at):
            count[0] += 1
            return count[0] >= stop_at

        def progress(fraction, _, fractions=fractions):
            fractions.append(fraction)

        summary = estimate_domains(m, composites, cancel=cancel, progress=progress, **kw)
        assert summary.cancelled
        assert count[0] == stop_at  # stops at the first true answer
        assert_same_model(m, before)
        assert "au_variance" not in m  # not even a new attribute is left behind
        assert all(f < 1.0 for f in fractions)
    # the same call, not cancelled, does write
    summary = estimate_domains(m, composites, cancel=lambda: False, **kw)
    assert not summary.cancelled
    for name in full.attribute_names:
        assert m[name].tobytes() == full[name].tobytes(), name


def test_exception_during_a_run_leaves_the_model_unchanged(coded, composites):
    m = fresh(coded)
    before = snapshot(m)
    calls = []

    def progress(fraction, message):
        calls.append(fraction)
        if "WASTE" in message:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        estimate_domains(
            m, composites, attribute="au", progress=progress,
            domains=[
                EstimationDomain("ORE", [base_pass("nn", 1.5)], ORE, ("domain", "ORE")),
                EstimationDomain("WASTE", [base_pass("nn", 1.5)], WASTE, ("domain", "WASTE")),
            ],
        )  # fmt: skip
    assert len(calls) > 2
    assert_same_model(m, before)
    # an int attribute cannot receive estimates; an output of the wrong kind is refused up front
    with pytest.raises(ValueError, match="float"):
        estimate_attribute(m, composites, attribute="rock", value_column="au",
                           passes=[base_pass("nn")])  # fmt: skip
    m.add_attribute("au_ns", "float")
    with pytest.raises(ValueError, match="au_ns"):
        estimate_attribute(m, composites, attribute="au", passes=[base_pass("nn")])
    assert "au" not in m


def test_progress_is_monotone_and_ends_at_one(coded, composites):
    m = fresh(coded)
    calls = []
    summary = estimate_domains(
        m, composites, attribute="au", chunk_size=700,
        progress=lambda fraction, message: calls.append((fraction, message)),
        domains=[
            EstimationDomain("ORE", three_passes("ok"), ORE, ("domain", "ORE")),
            EstimationDomain("WASTE", [base_pass("nn", 1.5)], WASTE, ("domain", "WASTE")),
        ],
    )  # fmt: skip
    fractions = np.array([f for f, _ in calls])
    assert len(fractions) > 50
    assert (np.diff(fractions) >= 0).all()
    assert fractions[0] >= 0.0 and fractions[-1] == 1.0
    assert (fractions[:-1] < 1.0).all()  # 1.0 is reported once, after the model is written
    assert len(np.unique(fractions)) > 0.5 * len(fractions)
    messages = [text for _, text in calls]
    assert any("ORE: pass 2/3 (medium)" in text for text in messages)
    assert any(text.startswith("WASTE") for text in messages)
    assert not summary.cancelled and summary.seconds > 0


@pytest.mark.parametrize("support", ["block", "parent"])
def test_chunk_size_does_not_change_results(support, sub_blocked, composites):
    results = []
    for chunk_size in (250_000, 997, 64):
        m = fresh(sub_blocked)
        s = estimate_attribute(m, composites, attribute="au", passes=three_passes("ok"),
                               block_constraint=ORE, sample_filter=("domain", "ORE"),
                               support=support, chunk_size=chunk_size)  # fmt: skip
        d = s.to_dict()["domains"][0]
        counts = [(p["n_targeted"], p["n_estimated"], p["n_negative"]) for p in d["passes"]]
        results.append(({n: m[n].copy() for n in m.attribute_names}, counts, d["mean_estimate"]))
    reference, counts, mean = results[0]
    assert sum(c[1] for c in counts) > 4000
    for columns, other_counts, other_mean in results[1:]:
        assert other_counts == counts
        assert other_mean == pytest.approx(mean, rel=1e-12)
        for name, column in reference.items():
            # block support mixes sizes inside a chunk; a chunk of equal sizes takes the
            # shared-size path of the estimator, which may differ in the last bits
            if support == "block" and column.dtype.kind == "f":
                _close(columns[name], column)
            else:
                assert columns[name].tobytes() == column.tobytes(), name


# ------------------------------------------------------------------ options


def test_capping_missing_values_and_negative_clipping(coded, composites):
    ore = composites[composites["domain"] == "ORE"].copy()
    ore.loc[ore.index[::7], "au"] = np.nan  # missing assays stay missing: never used as 0
    table = ore.copy()
    kw = dict(attribute="au", block_constraint=ORE)
    m = fresh(coded)
    p = base_pass("ok", 1.5)
    s = estimate_attribute(m, table, passes=[p], capping=5.0, **kw)
    pd.testing.assert_frame_equal(table, ore)  # the caller's table is not modified
    d = s.domains[0]
    valid = ore["au"].notna()
    assert (d.n_samples, d.n_samples_missing) == (int(valid.sum()), int((~valid).sum()))
    assert d.n_capped == int((ore["au"] > 5.0).sum()) > 0
    assert d.sample_max == 5.0 and d.sample_min == ore["au"].min()
    capped = ore[valid].assign(au=np.minimum(ore.loc[valid, "au"], 5.0))
    rows = np.flatnonzero(ORE.mask(m))
    result = array_estimate(p, capped, m.centroids(rows), m.definition.block_size, None)
    np.testing.assert_array_equal(m.get("au", rows), result.estimate)
    # per-domain caps through the existing capping function's mapping form
    both = composites[composites["au"].notna()]
    s = estimate_domains(
        fresh(coded), both, attribute="au", capping={"ORE": 4.0, "*": 0.5},
        capping_domain="domain",
        domains=[
            EstimationDomain("ORE", [base_pass("nn")], ORE, ("domain", "ORE")),
            EstimationDomain("WASTE", [base_pass("nn")], WASTE, ("domain", "WASTE")),
            EstimationDomain("WASTE2", [base_pass("nn")], WASTE, ("domain", "WASTE"), 0.1),
        ],
    )  # fmt: skip
    assert [d.sample_max for d in s.domains] == [4.0, 0.5, 0.1]
    assert s.domains[2].n_targeted == s.domains[1].n_unestimated  # the rest of WASTE

    # negative estimates: counted, clipped only on request. A strongly continuous
    # variogram with a wide search gives negative kriging weights (screen effect).
    smooth = dataclasses.replace(
        so.true_model(), nugget=0.0,
        structures=(dataclasses.replace(so.true_model().structures[0], type="gaussian"),),
    )  # fmt: skip
    shifted = ore[valid].assign(au=lambda t: t["au"] - 1.0)
    p = dataclasses.replace(base_pass("ok", 1.5), variogram=smooth)
    a, b = fresh(coded), fresh(coded)
    sa = estimate_attribute(a, shifted, passes=[p], **kw)
    sb = estimate_attribute(b, shifted, passes=[p], clip_negative=True, **kw)
    negative = a.get("au") < 0
    assert negative.sum() > 0
    assert sa.domains[0].n_negative == sb.domains[0].n_negative == negative.sum()
    assert sa.domains[0].passes[0].n_negative == negative.sum()
    assert (b.get("au")[negative] == 0.0).all() and sb.domains[0].min_estimate == 0.0
    np.testing.assert_array_equal(b.get("au")[~negative], a.get("au")[~negative])
    assert sa.domains[0].min_estimate < 0 and not sa.clip_negative and sb.clip_negative


def test_output_naming_and_selection(coded, composites):
    kw = dict(attribute="grade", value_column="au", block_constraint=ORE,
              sample_filter=("domain", "ORE"))  # fmt: skip
    m = fresh(coded)
    s = estimate_attribute(m, composites, passes=[base_pass("idw")], **kw)
    assert m.attribute_names == ["rock", "grade", "grade_pass", "grade_ns", "grade_dist",
                                 "grade_ndist"]  # fmt: skip
    assert s.outputs == {"estimate": "grade", "pass_number": "grade_pass",
                         "n_samples": "grade_ns", "mean_distance": "grade_dist",
                         "nearest_distance": "grade_ndist"}  # fmt: skip
    assert (m.attribute("grade_pass").dtype, m.attribute("grade_pass").background) == (np.int8, 0)
    assert np.isnan(m.attribute("grade").background)
    m = fresh(coded)
    estimate_attribute(m, composites, passes=[base_pass("ok")], outputs=["pass_number"], **kw)
    assert m.attribute_names == ["rock", "grade", "grade_pass"]
    m = fresh(coded)
    estimate_attribute(m, composites, passes=[base_pass("ok")], outputs=[], **kw)
    assert m.attribute_names == ["rock", "grade"]
    m = fresh(coded)
    estimate_attribute(m, composites, passes=[base_pass("ok")],
                       outputs={"pass_number": "_p", "n_samples": None, "variance": "_var"},
                       **kw)  # fmt: skip
    assert m.attribute_names == ["rock", "grade", "grade_p", "grade_dist", "grade_ndist",
                                 "grade_var", "grade_slope", "grade_ke"]  # fmt: skip
    with pytest.raises(ValueError, match="unknown output"):
        estimate_attribute(m, composites, passes=[base_pass("ok")], outputs=["weights"], **kw)
    with pytest.raises(ValueError, match="distinct"):
        OutputNames(variance="_x", slope="_x")


def test_empty_cases(coded, composites):
    m = fresh(coded)
    nothing = AttributeConstraint("z > 1e9")
    s = estimate_attribute(m, composites, attribute="au", passes=three_passes("ok"),
                           block_constraint=nothing)  # fmt: skip
    assert (s.n_targeted, s.n_estimated, s.n_unestimated) == (0, 0, 0)
    assert np.isnan(m.get("au")).all()  # the attribute is created all the same
    s = estimate_attribute(m, composites, attribute="au", passes=three_passes("ok"),
                           block_constraint=ORE, sample_filter=("domain", "NONE"))  # fmt: skip
    d = s.domains[0]
    assert (d.n_samples, d.n_estimated, d.n_unestimated) == (0, 0, d.n_targeted)
    assert d.n_targeted == ORE.mask(m).sum() and np.isnan(d.sample_mean)
    assert np.isnan(m.get("au")).all() and (m.get("au_pass") == 0).all()
    with pytest.raises(ValueError, match="hole_id"):
        estimate_attribute(m, composites.drop(columns="hole_id"), attribute="au",
                           passes=[base_pass("idw", max_per_hole=2)])  # fmt: skip
    with pytest.raises(ValueError, match="at least one"):
        estimate_attribute(m, composites, attribute="au", passes=[])


# ------------------------------------------------------------------ full synthetic run (Gate B)


def test_full_synthetic_run(resolver, composites, truth_columns, capsys):
    """Model on the truth grid, coded by centroid, ORE estimated by a three-pass OK plan with
    the true variogram; compared with the truth and with IDW2 and NN through the same code."""
    truth = so.truth()
    totals = truth["totals"]
    lines = []
    t_start = time.perf_counter()
    m = coded_model(resolver)
    t_model = time.perf_counter() - t_start
    ore = ORE.mask(m)
    rows = np.flatnonzero(ore)
    agree = float((m.get("rock", decode=True) == truth_columns["rock"]).mean())
    assert agree > 0.999

    summaries = {}
    seconds = {}
    for method in ("ok", "idw", "nn"):
        t = time.perf_counter()
        summaries[method] = estimate_attribute(
            m, composites, attribute=f"au_{method}", value_column="au",
            passes=three_passes(method), block_constraint=ORE, sample_filter=("domain", "ORE"),
        )  # fmt: skip
        seconds[method] = time.perf_counter() - t
    t_total = time.perf_counter() - t_start

    ok = summaries["ok"].domains[0]
    lines.append(
        f"model {m.n_blocks:,} blocks ({m.definition.shape}, 10 x 10 x 5 m), rock coded by "
        f"centroid: {100 * agree:.3f} % of blocks equal truth rock; ORE blocks {len(rows)} "
        f"(truth {int((truth_columns['rock'] == 'ORE').sum())})"
    )
    lines.append(
        f"ORE composites {ok.n_samples} (2 m): mean {ok.sample_mean:.4f}, "
        f"min {ok.sample_min:.4f}, max {ok.sample_max:.4f} g/t"
    )
    lines.append("pass  search(major m)  min/max smp  targeted  estimated  left   mean    NN mean")
    for method in ("ok", "idw", "nn"):
        for p, spec in zip(summaries[method].domains[0].passes, three_passes(method), strict=True):
            lines.append(
                f"{method:>3} {p.number}  {spec.search_ellipsoid.major_radius:>8.0f}        "
                f"{spec.search.min_samples:>2}/{spec.search.max_samples:<2}       "
                f"{p.n_targeted:>6}   {p.n_estimated:>6}   {p.n_unestimated:>4}  "
                f"{p.mean_estimate:.4f}  {p.nn_mean:.4f}"
            )

    # (1) every ORE block has an estimate after the last pass
    for method in ("ok", "idw", "nn"):
        d = summaries[method].domains[0]
        assert d.n_targeted == len(rows)
        assert d.n_unestimated == 0, (method, d.n_unestimated)
        assert np.isfinite(m.get(f"au_{method}", rows)).all()
        assert np.isnan(m.get(f"au_{method}")[~ore]).all()
    assert ok.n_negative == 0
    assert all(p.n_estimated > 0 for p in ok.passes)

    # (2) against the true block grades of the ore part of each block
    true = truth_columns["au_ore"][rows]
    has_truth = np.isfinite(true)  # centroid-coded ORE blocks that hold no ore at all: none
    assert has_truth.all()
    scores = {
        method: compare_to_truth(m.get(f"au_{method}", rows), true)
        for method in ("ok", "idw", "nn")
    }
    lines.append("method   n      mean     bias%    RMSE    corr   slope(truth|est)  seconds")
    for method, c in scores.items():
        lines.append(
            f"{method:>4}  {c.n:>5}  {c.mean_estimate:.4f}  {100 * c.relative_bias:+6.2f}  "
            f"{c.rmse:.4f}  {c.correlation:.4f}  {c.slope:.3f}             {seconds[method]:.2f}"
        )
    lines.append(f"truth mean of au_ore over the same blocks {scores['ok'].mean_truth:.4f}")
    assert scores["ok"].correlation > 0.75
    assert scores["ok"].rmse <= scores["idw"].rmse <= scores["nn"].rmse

    # (3) global means: truth, and the NN (declustered) mean
    est = m.get("au_ok", rows)
    nn = m.get("au_nn", rows)
    comparison = global_comparison(est, nn, composites.loc[composites["domain"] == "ORE", "au"])
    assert est.mean() == pytest.approx(true.mean(), rel=0.05)
    assert est.mean() == pytest.approx(totals["ore_mean_grade_g_t"], rel=0.05)
    assert abs(comparison.block_vs_nn) < 0.05
    assert abs(ok.estimate_vs_nn) < 0.05
    assert ok.nn_mean == pytest.approx(comparison.nn_mean, rel=1e-9)
    lines.append(
        f"means: OK {est.mean():.4f}; truth blocks {true.mean():.4f} "
        f"({100 * (est.mean() / true.mean() - 1):+.2f} %); truth.json ore mean "
        f"{totals['ore_mean_grade_g_t']:.4f} "
        f"({100 * (est.mean() / totals['ore_mean_grade_g_t'] - 1):+.2f} %); NN "
        f"{comparison.nn_mean:.4f} ({100 * comparison.block_vs_nn:+.2f} %); naive composites "
        f"{comparison.naive_mean:.4f}"
    )

    # (4) tonnes and metal of the ORE domain
    volume = m.volumes(rows)
    tonnes = float((volume * ORE_DENSITY).sum())
    metal = float((volume * ORE_DENSITY * est).sum())
    e_tonnes = tonnes / totals["ore_tonnes"] - 1
    e_metal = metal / totals["ore_metal_g"] - 1
    assert truth["density_t_m3"]["ORE"] == ORE_DENSITY
    assert abs(e_tonnes) < 0.05 and abs(e_metal) < 0.05
    lines.append(
        f"centroid-coded ORE: {tonnes:,.0f} t ({100 * e_tonnes:+.2f} % of truth "
        f"{totals['ore_tonnes']:,.0f}); metal {metal:,.0f} g ({100 * e_metal:+.2f} % of truth "
        f"{totals['ore_metal_g']:,.0f})"
    )
    # the same with the true ore fraction of every block as a partial percentage; blocks
    # that hold some ore but whose centroid is outside have no estimate and are reported
    fraction = truth_columns["ore_fraction"]
    full_volume = m.volumes()
    p_tonnes = float((full_volume * fraction * ORE_DENSITY).sum())
    estimate_all = m.get("au_ok")
    covered = np.isfinite(estimate_all)
    p_tonnes_est = float((full_volume * fraction * ORE_DENSITY)[covered].sum())
    p_metal = float((full_volume * fraction * ORE_DENSITY * estimate_all)[covered].sum())
    lines.append(
        f"with truth ore_fraction as partial: {p_tonnes:,.0f} t "
        f"({100 * (p_tonnes / totals['ore_tonnes'] - 1):+.2f} %); of which in estimated blocks "
        f"{p_tonnes_est:,.0f} t ({100 * (p_tonnes_est / totals['ore_tonnes'] - 1):+.2f} %), "
        f"metal {p_metal:,.0f} g ({100 * (p_metal / totals['ore_metal_g'] - 1):+.2f} %); "
        f"{int(((fraction > 0) & ~covered).sum())} blocks with some ore have no estimate "
        f"(centroid outside the solid)"
    )
    assert p_tonnes == pytest.approx(totals["ore_tonnes"], rel=0.001)

    # (5) swaths: block means follow composite means along x, y, z
    ore_comps = composites[(composites["domain"] == "ORE") & composites["au"].notna()]
    centroids = m.centroids(rows)
    swath = {}
    for axis, width in (("x", 20.0), ("y", 20.0), ("z", 10.0)):
        table = swath_table(centroids, est, ore_comps[["x", "y", "z"]].to_numpy(),
                            ore_comps["au"].to_numpy(), width=width, axis=axis)  # fmt: skip
        informed = table[(table["sample_count"] >= 10) & (table["block_count"] >= 10)]
        r = float(np.corrcoef(informed["block_mean"], informed["sample_mean"])[0, 1])
        ratio = float(informed["block_mean"].std() / informed["sample_mean"].std())
        swath[axis] = (r, len(informed), len(table), ratio)
        # for scale: the same slices against the true block means (not asserted)
        by_slice = compare_to_truth(est, true, coords=centroids, axis=axis, width=width).by_slice
        by_slice = by_slice[by_slice["count"] >= 10]
        r_truth = float(np.corrcoef(by_slice["estimate_mean"], by_slice["truth_mean"])[0, 1])
        lines.append(
            f"swath {axis} ({width:.0f} m): correlation with composite means {r:.3f} over "
            f"{len(informed)} of {len(table)} slices with >= 10 composites and blocks; "
            f"sd(block means) / sd(composite means) {ratio:.2f}; correlation with TRUE block "
            f"means per slice {r_truth:.3f} ({len(by_slice)} slices)"
        )
        assert r > 0.6, (axis, r)
    lines.append(
        f"time: model + coding {t_model:.2f} s; OK {seconds['ok']:.2f} s, IDW "
        f"{seconds['idw']:.2f} s, NN {seconds['nn']:.2f} s; total {t_total:.2f} s "
        f"(includes JIT compilation on a cold cache)"
    )
    assert t_total < 120.0
    with capsys.disabled():
        print("\n[P4.T7 full synthetic run]\n  " + "\n  ".join(lines))


# ------------------------------------------------------------------ performance


def test_block_layer_adds_little_to_the_array_api(composites, capsys):
    """Single-pass OK of 216 000 blocks: this layer against ``krige`` called directly."""
    ore = composites[(composites["domain"] == "ORE") & composites["au"].notna()]
    xyz = ore[["x", "y", "z"]].to_numpy()
    lo = xyz.min(axis=0)
    n = 60
    size = tuple(float(v) for v in (xyz.max(axis=0) - lo) / n)
    d = BlockModelDefinition(tuple(lo), size, (n, n, n))
    p = EstimationPass("ok", so.true_range_ellipsoid(3000.0),
                       SearchParams(max_samples=16, min_samples=1, max_per_hole=4),
                       variogram=so.true_model())  # fmt: skip
    warm = BlockModel(BlockModelDefinition(tuple(lo), size, (3, 3, 3)))
    estimate_attribute(warm, ore, attribute="au", passes=[p])  # JIT

    def direct():
        m = BlockModel(d)
        return array_estimate(p, ore, m.centroids(), size, d.rotation)

    def layer(**options):
        m = BlockModel(d)
        return m, estimate_attribute(m, ore, attribute="au", passes=[p], **options)

    def timed(function, **options):
        best, value = np.inf, None
        for _ in range(2):
            t = time.perf_counter()
            value = function(**options)
            best = min(best, time.perf_counter() - t)
        return best, value

    t_array, result = timed(direct)
    t_plain, (m, summary) = timed(layer, declustered_check=False)
    t_check, _ = timed(layer)
    assert summary.n_estimated == d.n_parents == result.estimated.sum()
    np.testing.assert_array_equal(m.get("au"), result.estimate)
    np.testing.assert_array_equal(m.get("au_kv"), result.variance)
    with capsys.disabled():
        print(
            f"\n[P4.T7 overhead] OK, {d.n_parents:,} blocks, {len(ore)} composites, max 16 "
            f"samples, 4x4x2: array API {t_array:.2f} s; block layer {t_plain:.2f} s "
            f"({100 * (t_plain / t_array - 1):+.1f} %); with the NN declustering check "
            f"{t_check:.2f} s ({100 * (t_check / t_array - 1):+.1f} %)"
        )
    assert t_plain < 1.5 * t_array + 2.0
    assert t_check < 1.5 * t_array + 3.0
