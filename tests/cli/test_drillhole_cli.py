"""Drillhole CLI (P1.T7): import -> validate -> desurvey -> composite, end to end."""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import build_parser, main
from bakumine.core.drillholes.importer import load_database

REPO = Path(__file__).resolve().parents[2]
SYNTH = REPO / "data" / "synthetic" / "default"
BABBITT_RAW = REPO / "data" / "public" / "babbitt" / "raw"


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


def import_synth(capsys, db: Path, *extra: object) -> tuple[int, str, str]:
    return run(
        capsys, "import", "--collar", SYNTH / "collar.csv", "--survey", SYNTH / "survey.csv",
        "--assay", SYNTH / "assay.csv", "--geology", SYNTH / "geology.csv", "--out", db, *extra,
    )


def test_commands_are_registered():
    text = build_parser().format_help()
    for name in ("import", "validate", "desurvey", "composite", "synth"):
        assert name in text


def test_end_to_end_on_synthetic_csvs(tmp_path, capsys):
    db = tmp_path / "db"
    raw = {n: pd.read_csv(SYNTH / f"{n}.csv") for n in ("collar", "survey", "assay", "geology")}

    # ---- import ----------------------------------------------------------------------
    code, out, err = import_synth(capsys, db, "--auto")
    assert code == 0, err
    for file in ("database.json", "collar.parquet", "survey.parquet", "interval_assay.parquet",
                 "interval_geology.parquet"):
        assert (db / file).is_file()
        assert f"wrote {db / file}" in out
    assert "holes 80" in out and len(out.splitlines()) < 40
    loaded = load_database(db)
    assert len(loaded.collar) == len(raw["collar"]) == 80
    assert len(loaded.survey) == len(raw["survey"])
    assert len(loaded.intervals["assay"]) == len(raw["assay"])
    assert len(loaded.intervals["geology"]) == len(raw["geology"])
    assert set(loaded.intervals["geology"]["rock"]) == {"ORE", "WASTE"}
    np.testing.assert_array_equal(loaded.intervals["assay"]["au"], raw["assay"]["au"])

    # ---- validate --------------------------------------------------------------------
    code, out, err = run(capsys, "validate", db, "--summary-out", tmp_path / "holes.csv")
    assert code == 0, err
    findings = pd.read_csv(db / "validation_findings.csv")
    assert list(findings.columns) == ["severity", "code", "table", "hole_id", "row", "message"]
    assert (findings["severity"] != "error").all()
    assert f"wrote {db / 'validation_findings.csv'} ({len(findings)} findings)" in out
    assert "errors 0" in out and "validation passed" in out
    assert len(pd.read_csv(tmp_path / "holes.csv")) == 80

    # ---- desurvey --------------------------------------------------------------------
    code, out, err = run(capsys, "desurvey", db, "--midpoints", "assay")
    assert code == 0, err
    assert "holes traced 80 of 80" in out
    stations = pd.read_parquet(db / "stations.parquet")
    assert list(stations.columns) == ["hole_id", "depth", "x", "y", "z", "dx", "dy", "dz", "kind"]
    assert stations["hole_id"].nunique() == 80
    assert len(stations) >= len(raw["survey"])
    pd.testing.assert_frame_equal(
        pd.read_csv(db / "stations.csv", dtype={"hole_id": "str"}), stations, check_exact=False,
        rtol=1e-12,
    )
    # The measured stations agree with the generator's truth.
    truth_stations = pd.read_csv(SYNTH / "truth_stations.csv")
    merged = truth_stations.merge(stations, on=["hole_id", "depth"], suffixes=("_t", ""))
    assert len(merged) == len(truth_stations)
    for axis in "xyz":
        assert np.abs(merged[axis] - merged[f"{axis}_t"]).max() < 1e-5
    # Sample midpoints match the truth XYZ of every assay interval within 1e-5 m.
    truth = pd.read_csv(SYNTH / "truth_samples.csv")
    for file in ("midpoints_assay.parquet", "midpoints_assay.csv"):
        path = db / file
        mid = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
        assert len(mid) == len(truth) == len(raw["assay"])
        assert (mid["hole_id"].to_numpy() == truth["hole_id"].to_numpy()).all()
        np.testing.assert_array_equal(mid["depth_from"], truth["depth_from"])
        error = np.abs(mid[["x", "y", "z"]].to_numpy() - truth[["x", "y", "z"]].to_numpy())
        assert error.max() < 1e-5, error.max()
        np.testing.assert_array_equal(mid["au"], raw["assay"]["au"])

    # ---- composite: within the rock domains of the geology table -----------------------
    code, out, err = run(
        capsys, "composite", db, "--table", "assay", "--mode", "domain", "--length", 2,
        "--domain-table", "geology", "--domain-column", "rock",
    )
    assert code == 0, err
    comps = pd.read_parquet(db / "composites_assay.parquet")
    from_csv = pd.read_csv(db / "composites_assay.csv")
    assert len(from_csv) == len(comps)
    np.testing.assert_allclose(from_csv["au"], comps["au"], rtol=1e-12)
    assert list(comps.columns) == [
        "hole_id", "depth_from", "depth_to", "length", "x", "y", "z", "dx", "dy", "dz", "domain",
        "au", "au_coverage",
    ]
    assay = raw["assay"]
    total = float((assay["depth_to"] - assay["depth_from"]).sum())
    assert total / 2.0 <= len(comps) <= total / 2.0 + 2 * len(raw["geology"])  # ~11 400
    assert comps["hole_id"].nunique() == 80
    assert set(comps["domain"]) == {"ORE", "WASTE"}
    assert (comps["length"] <= 2.0 + 1e-9).all()
    assert np.isfinite(comps[["x", "y", "z", "au"]].to_numpy()).all()
    metal = float((assay["au"] * (assay["depth_to"] - assay["depth_from"])).sum())
    assert float((comps["au"] * comps["au_coverage"]).sum()) == pytest.approx(metal, rel=1e-9)
    ore, waste = (comps.loc[comps["domain"] == d, "au"].mean() for d in ("ORE", "WASTE"))
    assert ore > 5 * waste  # the domains really separate ore from waste
    assert f"composites {len(comps)} in 80 holes" in out
    assert "domains from geology.rock" in out and len(out.splitlines()) < 25


def test_composite_options_and_capping(tmp_path, capsys):
    db = tmp_path / "db"
    maps = {
        "collar": {"hole_id": "hole_id", "x": "x", "y": "y", "z": "z", "max_depth": "max_depth"},
        "survey": {"hole_id": "hole_id", "depth": "depth", "dip": "dip", "azimuth": "azimuth"},
        "assay": {"hole_id": "hole_id", "depth_from": "depth_from", "depth_to": "depth_to",
                  "au": "gold"},
        "geology": {"hole_id": "hole_id", "depth_from": "depth_from", "depth_to": "depth_to",
                    "rock": "rock"},
    }
    (tmp_path / "maps.json").write_text(json.dumps(maps))
    code, out, err = import_synth(capsys, db, "--map", tmp_path / "maps.json")
    assert code == 0, err
    assert "assay (map file)" in out
    assert "gold" in load_database(db).intervals["assay"].columns

    # Per-domain caps, merge of short ends, explicit output path.
    base = tmp_path / "out" / "comps"
    code, out, err = run(
        capsys, "composite", db, "--mode", "domain", "--length", 3, "--domain-table", "geology",
        "--domain-column", "rock", "--cap", "gold:ORE=5", "--cap", "gold:*=0.5",
        "--min-length", 0.5, "--short", "merge", "--out", base,
    )
    assert code == 0, err
    comps = pd.read_parquet(base.with_name("comps.parquet"))
    assert base.with_name("comps.csv").is_file()
    report = pd.read_csv(base.with_name("comps_capping.csv"))
    assert report["domain"].tolist() == ["ORE", "*"]
    assert report["cap"].tolist() == [5.0, 0.5]
    assert (report["n_cut"] > 0).all()
    assert {"gold", "gold_uncut", "gold_coverage", "gold_uncut_coverage"} <= set(comps.columns)
    assert (comps["gold"] <= comps["gold_uncut"] + 1e-12).all()
    assert comps.loc[comps["domain"] == "ORE", "gold"].max() <= 5.0 + 1e-12
    assert comps.loc[comps["domain"] == "WASTE", "gold"].max() <= 0.5 + 1e-12
    removed = float(((comps["gold_uncut"] - comps["gold"]) * comps["gold_coverage"]).sum())
    assert removed == pytest.approx(report["metal_removed"].sum(), rel=1e-9)
    assert "values cut" in out
    # Only a run that is a single short composite may stay below the threshold.
    short = comps[comps["length"] < 1.5 - 1e-9]
    assert len(short) < 0.01 * len(comps)

    # Bench mode with a single cap and a fill value; a TOML map gives the same database.
    code, out, err = run(
        capsys, "composite", db, "--mode", "bench", "--bench-height", 10, "--bench-reference", 5,
        "--cap", "gold=3", "--fill-missing", "gold=0", "--min-coverage", 0.5,
        "--out", tmp_path / "bench",
    )
    assert code == 0, err
    bench = pd.read_parquet(tmp_path / "bench.parquet")
    assert 1500 < len(bench) < 6000 and "domain" not in bench.columns
    assert bench["gold"].max() <= 3.0 + 1e-12
    toml = "\n".join(
        f"[{table}]\n" + "\n".join(f'{k} = "{v}"' for k, v in mapping.items())
        for table, mapping in maps.items()
    )
    (tmp_path / "maps.toml").write_text(toml)
    code, _, err = import_synth(capsys, tmp_path / "db2", "--map", tmp_path / "maps.toml")
    assert code == 0, err
    pd.testing.assert_frame_equal(
        load_database(tmp_path / "db2").intervals["assay"], load_database(db).intervals["assay"]
    )


def test_end_to_end_on_babbitt_raw_files(tmp_path, capsys):
    """Raw public files, column maps guessed from the headers, feet, positive-down dips."""
    db = tmp_path / "babbitt"
    code, out, err = run(
        capsys, "import", "--collar", BABBITT_RAW / "collar_BABBITT.csv",
        "--survey", BABBITT_RAW / "survey_BABBITT.csv",
        "--assay", BABBITT_RAW / "assay_BABBITT.csv", "--out", db, "--auto",
        "--length-unit", "ft", "--to-metres", "--dip-convention", "positive_down",
    )
    assert code == 0, err
    assert "BHID->hole_id" in out and "XCOLLAR->x" in out and "AT->depth" in out
    assert "length unit m (converted from ft)" in out
    assert "dip convention positive_down" in out
    loaded = load_database(db)
    raw_collar = pd.read_csv(BABBITT_RAW / "collar_BABBITT.csv")
    raw_assay = pd.read_csv(BABBITT_RAW / "assay_BABBITT.csv")
    assert len(loaded.collar) == len(raw_collar) == 399
    assert len(loaded.intervals["assay"]) == len(raw_assay)
    assert loaded.length_unit == "m" and loaded.dip_convention == "positive_down"
    np.testing.assert_allclose(loaded.collar["x"], raw_collar["XCOLLAR"] * 0.3048, rtol=1e-14)
    np.testing.assert_allclose(
        loaded.intervals["assay"]["depth_to"], raw_assay["TO"] * 0.3048, rtol=1e-14
    )
    assert (loaded.survey["dip"] > 0).all()  # source sign kept; the convention is recorded

    code, out, err = run(capsys, "validate", db, "--out", tmp_path / "findings.csv")
    assert code == 0, err
    findings = pd.read_csv(tmp_path / "findings.csv")
    assert len(findings) > 0 and (findings["severity"] != "error").all()

    code, out, err = run(capsys, "desurvey", db, "--format", "parquet", "--midpoints", "assay")
    assert code == 0, err
    assert not (db / "stations.csv").exists()
    stations = pd.read_parquet(db / "stations.parquet")
    assert stations["hole_id"].nunique() == 399
    # Positive-down dips: every hole goes down from its collar.
    first = stations.groupby("hole_id").first()
    last = stations.groupby("hole_id").last()
    assert (last["z"] < first["z"]).all()
    assert (stations["dz"] < 0).all()
    # A vertical hole of 2954 ft ends 2954 * 0.3048 m below its collar.
    hole = stations[stations["hole_id"] == "34873"]
    assert hole["z"].iloc[0] - hole["z"].iloc[-1] == pytest.approx(2954 * 0.3048, rel=1e-12)
    midpoints = pd.read_parquet(db / "midpoints_assay.parquet")
    assert len(midpoints) == len(raw_assay)
    assert np.isfinite(midpoints[["x", "y", "z"]].to_numpy()).all()

    code, out, err = run(capsys, "composite", db, "--length", 3, "--cap", "cu=5")
    assert code == 0, err
    comps = pd.read_parquet(db / "composites_assay.parquet")
    assert len(pd.read_csv(db / "composites_assay.csv")) == len(comps)
    assert {"cu", "cu_uncut", "ni", "s", "fe", "cu_coverage"} <= set(comps.columns)
    sampled = raw_assay.dropna(subset=["CU", "NI", "S", "FE"], how="all")
    sampled_m = float((sampled["TO"] - sampled["FROM"]).sum()) * 0.3048
    assert sampled_m / 3.0 <= len(comps) <= sampled_m / 3.0 + 2 * len(sampled)
    assert comps["cu"].max() <= 5.0 + 1e-12 < comps["cu_uncut"].max()
    length_m = (raw_assay["TO"] - raw_assay["FROM"]) * 0.3048
    for raw_name, name in (("CU", "cu_uncut"), ("NI", "ni"), ("S", "s")):
        metal = float((raw_assay[raw_name] * length_m).sum())
        got = float((comps[name] * comps[f"{name}_coverage"]).sum())
        assert got == pytest.approx(metal, rel=1e-9)
    assert np.isfinite(comps[["x", "y", "z"]].to_numpy()).all()


def test_validate_exit_code_and_allow_errors(tmp_path, capsys):
    bad = tmp_path / "assay.csv"
    assay = pd.read_csv(SYNTH / "assay.csv")
    assay.loc[10, "depth_to"] = assay.loc[10, "depth_from"] - 1.0  # from >= to
    assay.loc[len(assay)] = ["NOCOLLAR", 0.0, 1.0, 0.5]
    assay.to_csv(bad, index=False)
    db = tmp_path / "db"
    code, _, err = run(
        capsys, "import", "--collar", SYNTH / "collar.csv", "--survey", SYNTH / "survey.csv",
        "--assay", bad, "--out", db, "--auto",
    )
    assert code == 0, err
    code, out, _ = run(capsys, "validate", db)
    assert code == 1
    assert "validation FAILED" in out and "from_ge_to" in out and "missing_collar" in out
    findings = pd.read_csv(db / "validation_findings.csv")
    assert (findings["severity"] == "error").sum() >= 2
    code, out, _ = run(capsys, "validate", db, "--allow-errors")
    assert code == 0
    assert "errors (allowed)" in out
    # Compositing refuses the unusable row, and skips it on request.
    code, _, err = run(capsys, "composite", db, "--length", 2)
    assert code == 2 and "cannot be composited" in err
    code, _, err = run(capsys, "composite", db, "--length", 2, "--drop-invalid")
    assert code == 0, err


def test_usage_errors_are_reported_not_raised(tmp_path, capsys):
    db = tmp_path / "db"
    code, _, err = import_synth(capsys, db)
    assert code == 2 and "--map FILE, or use --auto" in err
    code, _, err = run(
        capsys, "import", "--collar", tmp_path / "nope.csv", "--survey", SYNTH / "survey.csv",
        "--out", db, "--auto",
    )
    assert code == 2 and "not found" in err
    headerless = tmp_path / "collar.csv"
    headerless.write_text("a,b,c\n1,2,3\n")
    code, _, err = run(
        capsys, "import", "--collar", headerless, "--survey", SYNTH / "survey.csv", "--out", db,
        "--auto",
    )
    assert code == 2 and "--auto found no column" in err
    assert not db.exists()
    for command in ("validate", "desurvey", "composite"):
        code, _, err = run(capsys, command, tmp_path / "missing")
        assert code == 2 and "not a drillhole database folder" in err
    assert import_synth(capsys, db, "--auto")[0] == 0
    cases = {
        ("--mode", "domain", "--length", 2): "needs --domain-column",
        ("--table", "lith", "--length", 2): "no interval table 'lith'",
        ("--length", 2, "--cap", "au:ORE=5"): "need --domain-column",
        ("--length", 2, "--cap", "au=high"): "not a number",
        ("--mode", "bench"): "bench_height",
        (): "needs length",
        ("--length", 2, "--domain-table", "geology", "--domain-column", "lith"): "no column",
    }
    for extra, message in cases.items():
        code, _, err = run(capsys, "composite", db, *extra)
        assert code == 2 and message in err, (extra, err)
    code, _, err = run(capsys, "desurvey", db, "--midpoints", "lith")
    assert code == 2 and "no interval table" in err


def test_runs_from_the_shell(tmp_path):
    """The same entry point as a real process (`python -m bakumine.cli ...`)."""
    db = tmp_path / "db"
    steps = [
        ["import", "--collar", SYNTH / "collar.csv", "--survey", SYNTH / "survey.csv",
         "--assay", SYNTH / "assay.csv", "--geology", SYNTH / "geology.csv", "--out", db,
         "--auto"],
        ["validate", db],
        ["desurvey", db, "--format", "csv"],
        ["composite", db, "--mode", "domain", "--length", "2", "--domain-table", "geology",
         "--domain-column", "rock", "--best-fit"],
    ]
    for step in steps:
        done = subprocess.run(
            [sys.executable, "-m", "bakumine.cli", *map(str, step)],
            capture_output=True, text=True, cwd=tmp_path, check=False,
        )
        assert done.returncode == 0, done.stderr
        assert "wrote " in done.stdout
    assert len(pd.read_csv(db / "stations.csv")) > 800
    assert len(pd.read_parquet(db / "composites_assay.parquet")) > 10000
