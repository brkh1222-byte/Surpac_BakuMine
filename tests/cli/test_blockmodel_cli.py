"""Block model CLI: info, export-csv, import-csv (P3.T8c) and report (P3.T9)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import build_parser, main
from bakumine.core.blockmodel import BlockModel, BlockModelDefinition
from bakumine.core.blockmodel.storage import load_model, save_model

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "data" / "synthetic" / "default"


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def truth_model() -> BlockModel:
    frame = pd.read_parquet(SYNTH / "truth_blocks.parquet")
    definition = BlockModelDefinition((500400.0, 5300250.0, 825.0), (10.0, 10.0, 5.0),
                                      (39, 52, 84))  # fmt: skip
    model = BlockModel(definition)
    flat = definition.flat_index(frame[["i", "j", "k"]].to_numpy())
    for name, unit in (("au", "g/t"), ("au_ore", "g/t"), ("density", "t/m3"), ("ore_fraction", "")):
        model.add_attribute(name, unit=unit)
        model[name][flat] = frame[name].to_numpy()
    model.add_attribute("rock", "category")
    model.set("rock", frame["rock"].to_numpy(dtype=object), flat)
    return model


@pytest.fixture(scope="module")
def saved(tmp_path_factory) -> tuple[Path, BlockModel]:
    model = truth_model()
    path = tmp_path_factory.mktemp("cli") / "truth_model"
    save_model(model, path)
    return path, model


def test_blockmodel_command_is_registered():
    text = build_parser().format_help()
    assert "blockmodel" in text


def test_info(saved, capsys):
    path, model = saved
    code, out, err = run(capsys, "blockmodel", "info", path)
    assert code == 0, err
    assert "origin (x, y, z):      500400, 5.30025e+06, 825" in out
    assert "blocks (nx, ny, nz):   39, 52, 84" in out
    assert "blocks:                170352" in out
    assert "rotation:              none" in out
    for name in model.attribute_names:
        assert f"  {name} " in out
    assert "3 categories: " in out and "[g/t]" in out
    code, out, err = run(capsys, "blockmodel", "info", path.parent / "nothing")
    assert code == 2 and "not a block model folder" in err


def test_export_then_import_reproduces_the_model(saved, capsys, tmp_path):
    path, model = saved
    csv = tmp_path / "blocks.csv"
    code, out, err = run(capsys, "blockmodel", "export-csv", path, csv)
    assert code == 0, err
    assert f"wrote {csv} (170352 of 170352 blocks, layout surpac_like)" in out
    assert "columns: ycentre, xcentre, zcentre, ysize, xsize, zsize, au, au_ore" in out

    out_model = tmp_path / "reimported"
    code, out, err = run(capsys, "blockmodel", "import-csv", csv, out_model, "--infer")
    assert code == 0, err
    assert "rows 170352, assigned 170352, unmatched 0" in out
    assert "assumed: definition inferred from the rows" in out
    assert f"wrote {out_model} (170352 blocks, 5 attributes)" in out
    loaded = load_model(out_model)
    assert loaded.definition == model.definition
    assert np.array_equal(loaded.parent, model.parent) and loaded.is_dense
    for name in ("au", "au_ore", "density", "ore_fraction"):
        assert np.array_equal(loaded[name], model[name], equal_nan=True)
    assert np.array_equal(loaded.get("rock", decode=True), model.get("rock", decode=True))


def test_export_subset_and_import_with_definition_or_into(saved, capsys, tmp_path):
    path, model = saved
    csv = tmp_path / "ore.csv"
    code, out, err = run(
        capsys, "blockmodel", "export-csv", path, csv, "--layout", "xyz", "--where",
        'rock == "ORE" and au_ore > 2', "-a", "au_ore,rock", "--decimals", 3,
        "--missing", "-99",
    )  # fmt: skip
    assert code == 0, err
    mask = (model.get("rock", decode=True) == "ORE") & (model["au_ore"] > 2)
    assert f"({int(mask.sum())} of 170352 blocks, layout xyz)" in out
    frame = pd.read_csv(csv)
    assert list(frame.columns) == ["x", "y", "z", "dx", "dy", "dz", "au_ore", "rock"]
    assert len(frame) == mask.sum() and (frame["rock"] == "ORE").all()
    assert np.allclose(frame["au_ore"], model["au_ore"][mask], atol=5e-4, rtol=0)

    # a definition file: either a bare definition or a saved model's model.json
    definition = tmp_path / "definition.json"
    definition.write_text(json.dumps(model.definition.to_dict()))
    for source in (definition, path / "model.json", path):
        target = tmp_path / f"new_{source.name}"
        code, out, err = run(capsys, "blockmodel", "import-csv", csv, target,
                             "--definition", source, "--missing", "-99")  # fmt: skip
        assert code == 0, err
        loaded = load_model(target)
        assert loaded.definition == model.definition
        assert np.array_equal(~np.isnan(loaded["au_ore"]), mask)

    # into the existing model: other attributes and blocks are kept
    target = tmp_path / "merged"
    code, out, err = run(capsys, "blockmodel", "import-csv", csv, target, "--into", path,
                         "-a", "au_ore")  # fmt: skip
    assert code == 0, err
    assert "updated: au_ore" in out
    merged = load_model(target)
    assert merged.attribute_names == model.attribute_names
    assert np.allclose(merged["au_ore"][mask], model["au_ore"][mask], atol=5e-4, rtol=0)
    assert np.array_equal(merged["au_ore"][~mask], model["au_ore"][~mask], equal_nan=True)
    assert np.array_equal(merged["au"], model["au"], equal_nan=True)


def test_errors_are_one_line_and_exit_code_2(saved, capsys, tmp_path):
    path, _ = saved
    code, out, err = run(capsys, "blockmodel", "export-csv", path, tmp_path / "x.csv",
                         "--where", "cu > 1")  # fmt: skip
    assert code == 2 and err.startswith("error: ") and "cu" in err and len(err.splitlines()) == 1
    code, out, err = run(capsys, "blockmodel", "export-csv", path, tmp_path / "x.csv", "-a", "cu")
    assert code == 2 and "no attribute 'cu'" in err
    code, out, err = run(capsys, "blockmodel", "export-csv", path, tmp_path / "x.csv",
                         "--layout", "vulcan")  # fmt: skip
    assert code == 2 and "unknown block CSV layout" in err
    bad = tmp_path / "bad.csv"
    bad.write_text("hello\n")
    code, out, err = run(capsys, "blockmodel", "import-csv", bad, tmp_path / "m")
    assert code == 2 and "no data found" in err and not (tmp_path / "m").exists()
    code, out, err = run(capsys, "blockmodel", "import-csv", bad, tmp_path / "m",
                         "--definition", bad)  # fmt: skip
    assert code == 2 and "not a block model definition" in err
    code, out, err = run(capsys, "blockmodel")
    assert code == 0 and "export-csv" in out


# --------------------------------------------------------------------- report


def test_report_reproduces_the_truth_totals(saved, capsys, tmp_path):
    path, model = saved
    out_csv = tmp_path / "ore.csv"
    code, out, err = run(
        capsys, "blockmodel", "report", path, "--density", 2.85, "--partial", "ore_fraction",
        "--grade", "au_ore", "--out", out_csv,
    )  # fmt: skip
    assert code == 0, err
    lines = out.splitlines()
    assert lines[0].split() == ["blocks", "volume", "tonnes", "volume_without_density",
                                "au_ore_grade", "au_ore_metal_g", "au_ore_metal_oz",
                                "au_ore_unestimated_tonnes"]  # fmt: skip
    assert lines[2].split()[:4] == ["170,352", "2,055,344", "5,857,730", "0"]
    assert "note: constant density 2.85 t/m3" in out and f"wrote {out_csv} (1 rows)" in out
    table = pd.read_csv(out_csv)
    truth = json.loads((SYNTH / "truth.json").read_text())["totals"]
    # tolerances and the reasons for them: tests/report/test_report.py
    assert table["tonnes"][0] == pytest.approx(truth["ore_tonnes"], rel=5e-5)
    assert table["au_ore_grade"][0] == pytest.approx(truth["ore_mean_grade_g_t"], rel=0.015)
    assert table["au_ore_metal_g"][0] == pytest.approx(truth["ore_metal_g"], rel=0.015)


def test_report_groups_cutoffs_benches_and_where(saved, capsys, tmp_path):
    from bakumine.core.blockmodel import AttributeConstraint
    from bakumine.core.report import report

    path, model = saved
    out_csv = tmp_path / "gt.csv"
    code, out, err = run(
        capsys, "blockmodel", "report", path, "--where", "z < 1100", "--group-by", "rock",
        "--cutoffs", "0.5,1,2", "--grade", "au:g/t", "--density", "density",
        "--bench-height", 100, "--bench-reference", 800, "--out", out_csv,
    )  # fmt: skip
    assert code == 0, err
    assert "note: benches of 100 m from elevation 800" in out and "TOTAL" in out
    table = pd.read_csv(out_csv)
    expected = report(
        model, constraint=AttributeConstraint("z < 1100"), group_by="rock",
        cutoffs=[0.5, 1.0, 2.0], grade_attributes="au", density="density", bench_height=100.0,
        bench_reference=800.0,
    )  # fmt: skip
    assert list(table.columns) == list(expected.columns) and len(table) == len(expected)
    for column in ("blocks", "tonnes", "au_grade", "au_metal_oz", "cutoff", "bench_from"):
        assert np.allclose(table[column], expected[column], rtol=1e-12, equal_nan=True), column
    assert table["bench_to"].max() == 1100.0 and set(table["rock"].dropna()) <= {"ORE", "WASTE",
                                                                                 "AIR"}  # fmt: skip
    code, out, err = run(capsys, "blockmodel", "report", path, "--grade", "au")
    assert code == 2 and "give the density" in err
    code, out, err = run(capsys, "blockmodel", "report", path, "--cutoffs", "a,b",
                         "--density", 2.7)  # fmt: skip
    assert code == 2 and "--cutoffs expects numbers" in err


def test_fraction_of_blocks_inside_a_solid(tmp_path, capsys):
    """`blockmodel fraction` against the analytic volume of a box that cuts blocks, and the
    report weighted by it; a mesh that is not closed is refused."""
    from bakumine.core.geometry.primitives import cube
    from bakumine.core.io.files import write_mesh

    model_dir = tmp_path / "model"
    save_model(BlockModel(BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 10.0), (4, 4, 4))),
               model_dir)  # fmt: skip
    # 8 .. 32 on every axis: cuts blocks, between the 5 sample lines of a block (1, 3, .. 9),
    # so that the sampled fraction is the exact one
    solid = cube(24.0, center=(20.0, 20.0, 20.0))
    write_mesh(tmp_path / "box.obj", solid)
    code, out, _ = run(capsys, "blockmodel", "fraction", model_dir, "--solid",
                       tmp_path / "box.obj", "--attribute", "in_box")  # fmt: skip
    assert code == 0, out
    assert "8 blocks entirely inside, 56 cut by the solid, 0 entirely on the other side" in out
    model = load_model(model_dir)
    fraction = model["in_box"]
    assert float((fraction * model.volumes()).sum()) == pytest.approx(24.0**3, rel=1e-9)
    assert fraction.min() > 0.0 and fraction.max() == 1.0
    # corner block 0..10: the box takes 2 m of each axis
    assert fraction[0] == pytest.approx((2.0 / 10.0) ** 3, rel=1e-9)
    code, out, _ = run(capsys, "blockmodel", "report", model_dir, "--partial", "in_box",
                       "--density", 2.0)  # fmt: skip
    assert code == 0 and f"{24**3 * 2:,.0f}" in out  # tonnes of the box
    # the other side, as percent, into another folder
    code, out, _ = run(capsys, "blockmodel", "fraction", model_dir, "--solid", tmp_path / "box.obj",
                       "--attribute", "out_pct", "--side", "outside", "--percent",
                       "--method", "lattice", "--samples", 5, "--out", tmp_path / "m2")  # fmt: skip
    assert code == 0, out
    other = load_model(tmp_path / "m2")
    np.testing.assert_allclose(other["out_pct"] + 100.0 * fraction, 100.0, atol=1e-9)
    assert "out_pct" not in load_model(model_dir)
    # an open mesh has no inside
    write_mesh(tmp_path / "open.obj", solid.submesh(np.arange(solid.n_faces - 2)))
    code, _, err = run(capsys, "blockmodel", "fraction", model_dir, "--solid",
                       tmp_path / "open.obj", "--attribute", "x")  # fmt: skip
    assert code == 2 and "not a closed solid" in err
