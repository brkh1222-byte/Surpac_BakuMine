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
