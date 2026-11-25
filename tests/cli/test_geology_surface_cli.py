"""`bakumine geology surface` (P9.T9), end to end on the small synthetic layered deposit."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import main
from bakumine.commands import geology
from bakumine.core.geology.picks import PICK_COLUMNS
from bakumine.core.geology.provenance import TABLE_TITLES
from bakumine.core.geometry.dtm import DTM
from bakumine.core.geometry.mesh import Mesh
from bakumine.core.geometry.surface_interp import Surface2D
from bakumine.core.io.obj import read_obj
from bakumine.synth.layered import LAYERED_PRESETS
from bakumine.synth.layered_generate import generate_layered, import_layered

RECTANGLE = "500200,5300200,500600,5300600"


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture(scope="module")
def deposit(tmp_path_factory):
    folder = tmp_path_factory.mktemp("layered_small_cli")
    generated = generate_layered(LAYERED_PRESETS["layered-small"], seed=42)
    generated.write(folder)
    import_layered(folder).save(folder / "db")
    return generated, folder


def inner_grid():
    x = np.arange(500226.0, 500575.0, 5.0)
    return np.meshgrid(x, x + 4800000.0, indexing="ij")


def rms(path: Path, truth, depth) -> float:
    gx, gy = inner_grid()
    z = DTM(read_obj(path)).z_at(gx, gy)
    return float(np.sqrt(np.mean((truth.topo_z(gx, gy) - z - depth(gx, gy)) ** 2)))


def cover_kinds(folder: Path) -> pd.Series:
    """Pick kind per hole for the soil codes, from the raw log with pandas only."""
    log = pd.read_csv(folder / "geology.csv", dtype={"lith": str}, keep_default_na=False)
    log["code"] = log["lith"].str.strip().str.upper()
    kinds = {}
    for hole_id, group in log.sort_values(["hole_id", "depth_from"]).groupby("hole_id"):
        codes = group["code"].tolist()
        inside = [i for i, c in enumerate(codes) if c in {"SOIL", "SOL"}]
        if not inside:
            kinds[hole_id] = "none"
            continue
        j = inside[-1] + 1
        while j < len(codes) and codes[j] in {"CL", "FLT", "FB"}:
            j += 1
        kinds[hole_id] = (
            "lower" if j == len(codes) else "exact" if j == inside[-1] + 1 else "between"
        )
    return pd.Series(kinds)


def test_surface_is_registered():
    assert ("surface", "bakumine.commands.geology_surface") in geology.SUBCOMMANDS
    assert geology.SUBCOMMANDS[0] == ("units", "bakumine.commands.geology_units")


def test_cover_then_oxide_then_transition_end_to_end(deposit, tmp_path, capsys):
    generated, folder = deposit
    truth = generated.truth
    common = (
        "geology", "surface", folder / "db", "--topography", folder / "topo.obj",
        "--spacing", 5,
    )  # fmt: skip

    # ---- cover base from the log
    cover = tmp_path / "out" / "base_Q4.obj"
    code, out, err = run(
        capsys, *common, "--units", folder / "units.json", "--unit", "Q4",
        "--rectangle", RECTANGLE, "--out", cover,
        "--picks-out", tmp_path / "out" / "base_Q4_picks.csv",
        "--summary", tmp_path / "out" / "base_Q4.json",
        "--geometry-dir", tmp_path / "geometry",
    )  # fmt: skip
    assert code == 0, err
    for title in TABLE_TITLES.values():
        assert title in out
    assert "surface base:Q4: 64 holes with a trace" in out
    kinds = cover_kinds(folder).value_counts()
    n = {k: int(kinds.get(k, 0)) for k in ("exact", "lower", "upper", "between", "none")}
    assert n["none"] > 0 and n["exact"] > 0
    assert (
        f"used in the fit: {n['exact']} exact, {n['lower']} lower, {n['upper']} upper, "
        f"{n['between']} between; {n['none']} holes give no information"
    ) in out
    assert f"{n['none']} holes of 64 give no pick and no bound" in out
    assert "default (absent = unknown)" in out and "Variogram of base:Q4" in out
    assert "Extrapolated part of base:Q4" in out

    picks = pd.read_csv(tmp_path / "out" / "base_Q4_picks.csv")
    assert list(picks.columns) == [*PICK_COLUMNS, "conflict", "used"] and len(picks) == 64
    assert picks["kind"].value_counts().to_dict() == {k: v for k, v in n.items() if v}
    assert int(picks["used"].sum()) == 64 - n["none"]
    summary = json.loads((tmp_path / "out" / "base_Q4.json").read_text(encoding="utf-8"))
    assert summary["name"] == "base:Q4" and summary["diagnostics"]["used"]["exact"] == n["exact"]
    by_item = {e["item"]: e for e in summary["provenance"]}
    assert by_item["Exact picks of base:Q4 (log)"]["count"] == n["exact"]
    assert by_item["base:Q4 not logged"]["count"] == n["none"]
    assert by_item["base:Q4 not logged"]["kind"] == "missing"
    fit = Surface2D.from_dict(summary)
    exact = picks[picks["kind"] == "exact"]
    np.testing.assert_allclose(fit(exact[["x", "y"]].to_numpy()), exact["below_lo"], atol=1e-7)
    mesh = Mesh(np.load(tmp_path / "geometry" / "base_Q4_vertices.npy"),
                np.load(tmp_path / "geometry" / "base_Q4_faces.npy"))  # fmt: skip
    assert mesh.n_faces == read_obj(cover).n_faces
    assert rms(cover, truth, truth.cover_depth) < 0.5

    # ---- oxide base from the log and the sulphur rule, under the cover base
    oxide = tmp_path / "out" / "base_OXIDE.obj"
    code, out, err = run(
        capsys, *common, "--units", folder / "units.json", "--unit", "OX",
        "--threshold", "assay:s_ppm:<1000", "--surface", "base:OXIDE", "--above", cover,
        "--rectangle", RECTANGLE, "--out", oxide, "--summary", tmp_path / "out" / "base_OXIDE.json",
    )  # fmt: skip
    assert code == 0, err
    assert "picks from log" in out and "picks from threshold" in out
    assert "Threshold rule for base:OXIDE" in out and "s_ppm < 1000" in out
    assert "Nesting of base:OXIDE" in out and "0 node(s) moved down" in out
    summary = json.loads((tmp_path / "out" / "base_OXIDE.json").read_text(encoding="utf-8"))
    assay = pd.read_csv(folder / "assay.csv")
    rule = next(e for e in summary["provenance"] if e["item"].startswith("Threshold rule"))
    assert rule["count"] == int((assay["s_ppm"] < 1000).sum())
    used = summary["diagnostics"]["used_by_source"]
    assert used["log"]["exact"] + used["threshold"]["exact"] == 64
    assert rms(oxide, truth, truth.oxide_depth) <= 2.5  # geology_contracts.md 14

    # ---- transition base from the rule alone, area from a block model, under the oxide base
    model = tmp_path / "model"
    code, _, err = run(capsys, "blockmodel", "create", model, "--origin", "500250,5300250,900",
                       "--size", "20,20,10", "--shape", "15,15,4")  # fmt: skip
    assert code == 0, err
    transition = tmp_path / "out" / "base_TRANSITION.obj"
    code, out, err = run(
        capsys, *common, "--threshold", "assay:s_ppm:<3000", "--surface", "base:TRANSITION",
        "--above", oxide, "--extend-model", model, "--extend-padding", 50,
        "--method", "rbf", "--out", transition,
    )  # fmt: skip
    assert code == 0, err
    assert "Interpolant of base:TRANSITION" in out and TABLE_TITLES["missing"] in out
    assert rms(transition, truth, truth.transition_depth) < 0.5
    bounds = read_obj(transition).bounds
    np.testing.assert_allclose(bounds[:, :2], [[500200.0, 5300200.0], [500600.0, 5300600.0]])

    # the three files never cross, and none is above the topography
    gx, gy = inner_grid()
    layers = [
        DTM(read_obj(p)).z_at(gx, gy) for p in (folder / "topo.obj", cover, oxide, transition)
    ]
    for upper, lower in zip(layers[:-1], layers[1:], strict=True):
        assert (upper >= lower - 1e-9).all()


def test_above_pushes_a_surface_down_and_says_how_many_nodes(deposit, tmp_path, capsys):
    """Built to cross: the cover base asked to lie under the oxide base."""
    _, folder = deposit
    common = ("geology", "surface", folder / "db", "--topography", folder / "topo.obj",
              "--spacing", 10, "--rectangle", RECTANGLE)  # fmt: skip
    oxide = tmp_path / "oxide.obj"
    code, _, err = run(capsys, *common, "--threshold", "assay:s_ppm:<1000", "--surface",
                       "base:OXIDE", "--out", oxide)  # fmt: skip
    assert code == 0, err
    cover = tmp_path / "cover.obj"
    code, out, err = run(capsys, *common, "--units", folder / "units.json", "--unit", "Q4",
                         "--above", oxide, "--out", cover)  # fmt: skip
    assert code == 0, err
    assert "1681 node(s) moved down" in out and "1681 grid nodes of 1681 moved down" in out
    gx, gy = inner_grid()
    np.testing.assert_allclose(DTM(read_obj(cover)).z_at(gx, gy), DTM(read_obj(oxide)).z_at(gx, gy),
                               atol=1e-9)  # fmt: skip


@pytest.mark.parametrize(
    ("extra", "message"),
    [
        (["--rectangle", RECTANGLE], "give --unit CODE"),
        (["--unit", "Q4", "--rectangle", RECTANGLE], "--unit needs --units"),
        (["--threshold", "assay:s_ppm:<1000", "--rectangle", RECTANGLE], "give --surface NAME"),
        (["--threshold", "assay:s_ppm:1000", "--surface", "base:X", "--rectangle", RECTANGLE],
         "--threshold expects"),
        (["--threshold", "assay:nope:<5", "--surface", "base:X", "--rectangle", RECTANGLE],
         "no column 'nope'"),
        (["--threshold", "assay:s_ppm:<1000", "--surface", "base:X"], "exactly one of"),
        (["--threshold", "assay:s_ppm:<1", "--surface", "base:X", "--rectangle", RECTANGLE],
         "no exact pick"),
    ],
)  # fmt: skip
def test_errors_are_one_line_and_exit_2(deposit, tmp_path, capsys, extra, message):
    _, folder = deposit
    out_file = tmp_path / "x.obj"
    code, _, err = run(capsys, "geology", "surface", folder / "db", "--topography",
                       folder / "topo.obj", "--spacing", 10, "--out", out_file, *extra)  # fmt: skip
    assert code == 2 and message in err and err.startswith("error:")
    assert not out_file.exists()
