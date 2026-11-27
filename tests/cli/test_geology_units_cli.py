"""`bakumine geology units init | check` (P9.T3), end to end on a small made-up log."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from bakumine.cli import build_parser, main
from bakumine.commands import geology
from bakumine.core.geology.units import UnitDictionary

HOLES = 6
SPELLINGS = ["SST", "sst", "Sst", "soil", "SOIL", "hb  dio", "HB DIO", "and", "breccia", "cl",
             "qz", "Qz", "xx", ""]  # fmt: skip

DICTIONARY = {
    "format": "bakumine.units",
    "version": 1,
    "name": "made-up deposit",
    "source": {"table": "geology", "column": "lith"},
    "default_density": 2.7,
    "units": [
        {"code": "Q4", "kind": "cover", "aliases": ["SOIL"], "density": 1.9,
         "density_source": "measured"},
        {"code": "SST", "kind": "lithology", "group": "SED"},
        {"code": "AND", "kind": "lithology"},
        {"code": "DIO", "kind": "intrusion", "aliases": ["HB DIO"]},
        {"code": "GRD", "kind": "intrusion"},
        {"code": "CL", "kind": "missing"},
        {"code": "BX", "kind": "other", "aliases": ["BRECCIA"]},
    ],
    "weathering_zones": [{"code": "OXIDE"}, {"code": "FRESH"}],
}  # fmt: skip
TO_UNIT = {"SST": "SST", "SOIL": "Q4", "HB DIO": "DIO", "AND": "AND", "BRECCIA": "BX", "CL": "CL"}


def run(capsys, *argv: object) -> tuple[int, str, str]:
    code = main([str(a) for a in argv])
    captured = capsys.readouterr()
    return code, captured.out, captured.err


@pytest.fixture
def log() -> pd.DataFrame:
    """Vertical holes on a line, contiguous intervals, codes in many spellings."""
    rng = np.random.default_rng(8)
    rows = []
    for h in range(HOLES):
        depth = 0.0
        for _ in range(int(rng.integers(9, 15))):
            length = float(np.round(rng.uniform(0.5, 8.0), 1))
            code = SPELLINGS[int(rng.integers(len(SPELLINGS)))]
            rows.append((f"H{h + 1}", round(depth, 1), round(depth + length, 1), code))
            depth += length
    return pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "lith"])


@pytest.fixture
def database(tmp_path, capsys, log) -> Path:
    ends = log.groupby("hole_id")["depth_to"].max()
    collar = pd.DataFrame({"hole_id": ends.index, "x": 100.0 + 50.0 * np.arange(HOLES),
                           "y": 200.0, "z": 50.0, "max_depth": ends.to_numpy()})  # fmt: skip
    survey = pd.DataFrame({"hole_id": ends.index, "depth": 0.0, "dip": -90.0, "azimuth": 0.0})
    rng = np.random.default_rng(2)
    density = pd.DataFrame({
        "hole_id": np.repeat(ends.index.to_numpy(), 6),
        # six samples per hole that do not overlap one another
        "depth_from": np.round(np.tile(1.0 + 6.0 * np.arange(6), HOLES)
                               + rng.uniform(0.0, 5.0, 6 * HOLES), 1),
    })  # fmt: skip
    density["depth_to"] = density["depth_from"] + 0.2
    density["sg"] = np.round(rng.uniform(2.0, 3.0, len(density)), 2)
    files = {"collar": collar, "survey": survey, "geology": log, "density": density}
    for name, frame in files.items():
        frame.to_csv(tmp_path / f"{name}.csv", index=False)
    db = tmp_path / "db"
    code, _, err = run(
        capsys, "import", "--collar", tmp_path / "collar.csv", "--survey", tmp_path / "survey.csv",
        "--geology", tmp_path / "geology.csv", "--interval", f"density={tmp_path / 'density.csv'}",
        "--out", db, "--auto",
    )  # fmt: skip
    assert code == 0, err
    return db


def counted(log: pd.DataFrame) -> pd.DataFrame:
    """The log with folded code, unit and length: pandas only, nothing from bakumine."""
    folded = log["lith"].str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    out = log.assign(folded=folded.mask(folded == ""), length=log["depth_to"] - log["depth_from"])
    out["unit"] = out["folded"].map(TO_UNIT)
    return out


def section(out: str, title: str) -> list[list[str]]:
    """The rows of one printed table, split on whitespace."""
    lines = out.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(title))
    rows = []
    for line in lines[start + 3 :]:
        if not line.startswith("  "):
            break
        rows.append(line.split())
    return rows


def write_dictionary(path: Path, **top) -> Path:
    path.write_text(json.dumps(DICTIONARY | top), encoding="utf-8")
    return path


# ---------------------------------------------------------------------- group


def test_group_is_registered_with_its_subcommand_list(capsys):
    assert "geology" in build_parser().format_help()
    assert geology.SUBCOMMANDS[0] == ("units", "bakumine.commands.geology_units")
    code, out, _ = run(capsys, "geology")
    assert code == 0 and "units" in out
    code, out, _ = run(capsys, "geology", "units")
    assert code == 0 and "init" in out and "check" in out


# ---------------------------------------------------------------------- init


def test_init_writes_a_draft_from_the_logged_codes(tmp_path, capsys, database, log):
    out_file = tmp_path / "units.json"
    code, out, err = run(
        capsys,
        "geology",
        "units",
        "init",
        database,
        "--table",
        "geology",
        "--column",
        "lith",
        "--out",
        out_file,
        "--name",
        "made-up deposit",
    )
    assert code == 0, err
    assert f"wrote {out_file}" in out
    hand = counted(log)
    metres = hand.groupby("folded")["length"].sum()
    intervals = hand.groupby("folded").size()

    draft = UnitDictionary.load(out_file)
    assert draft.name == "made-up deposit"
    assert (draft.source_table, draft.source_column) == ("geology", "lith")
    assert len(draft.units) == hand["folded"].nunique() == 8
    assert {u.kind for u in draft.units} == {"lithology"}
    assert set(draft.lookup()) >= set(metres.index)
    assert draft.unit("HB_DIO").aliases == ("HB DIO",)
    assert f"{len(draft.units)} codes after folding" in out

    rows = {row[0]: row for row in section(out, "table geology, column lith")}
    assert list(rows) == list(draft.codes())  # largest first, as in the file
    lookup = draft.lookup()
    for folded in metres.index:
        row = rows[lookup[folded]]
        assert int(row[1]) == intervals[folded]
        assert float(row[2]) == pytest.approx(metres[folded], abs=0.006)
    assert [float(r[2]) for r in rows.values()] == sorted((float(r[2]) for r in rows.values()),
                                                          reverse=True)  # fmt: skip

    # the draft maps everything, so the check passes
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", out_file)
    assert code == 0, err
    assert "NOT MAPPED" not in out

    # an existing (possibly edited) dictionary is not replaced silently
    before = out_file.read_text(encoding="utf-8")
    code, _, err = run(capsys, "geology", "units", "init", database, "--column", "lith",
                       "--out", out_file)  # fmt: skip
    assert code == 2 and "exists" in err and out_file.read_text(encoding="utf-8") == before
    code, _, _ = run(capsys, "geology", "units", "init", database, "--column", "lith",
                     "--out", out_file, "--overwrite")  # fmt: skip
    assert code == 0 and UnitDictionary.load(out_file).name == ""


def test_init_errors(tmp_path, capsys, database):
    code, _, err = run(capsys, "geology", "units", "init", database, "--column", "rock",
                       "--out", tmp_path / "u.json")  # fmt: skip
    assert code == 2 and "no column 'rock'" in err and "lith" in err
    code, _, err = run(capsys, "geology", "units", "init", database, "--table", "logs",
                       "--column", "lith", "--out", tmp_path / "u.json")  # fmt: skip
    assert code == 2 and "no interval table 'logs'" in err
    code, _, err = run(capsys, "geology", "units", "init", tmp_path / "nothing", "--column",
                       "lith", "--out", tmp_path / "u.json")  # fmt: skip
    assert code == 2 and "not a drillhole database" in err
    assert not (tmp_path / "u.json").exists()


# ---------------------------------------------------------------------- check


def test_check_counts_match_pandas_and_unmapped_is_an_error(tmp_path, capsys, database, log):
    units_file = write_dictionary(tmp_path / "units.json")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", units_file)
    hand = counted(log)
    coded = hand[hand["folded"].notna()]
    total = coded["length"].sum()

    # unmapped codes: exit 1, every one named
    assert code == 1
    assert "2 logged code(s) are not in the dictionary: 'QZ', 'XX'" in err
    assert (f"{len(coded)} intervals with a code, {total:.2f} m; "
            f"{len(hand) - len(coded)} intervals without a code") in out  # fmt: skip
    assert len(hand) - len(coded) > 0

    # metres by unit against a pandas groupby
    expected = coded.groupby("unit").agg(
        intervals=("length", "size"), metres=("length", "sum"), holes=("hole_id", "nunique")
    )
    rows = {row[0]: row for row in section(out, "Metres by unit")}
    assert list(rows) == [c for c in ("Q4", "SST", "AND", "DIO", "CL", "BX") if c in rows]
    assert set(rows) == set(expected.index) and len(rows) == 6
    for unit, row in expected.iterrows():
        printed = rows[unit]
        assert int(printed[2]) == row["intervals"]
        assert float(printed[3]) == pytest.approx(row["metres"], abs=0.006)
        assert int(printed[4]) == row["holes"]
        assert float(printed[5]) == pytest.approx(row["metres"] / total, abs=6e-5)
    assert rows["Q4"][1] == "cover" and rows["BX"][1] == "other"

    # every raw spelling is listed with what it maps to
    spellings = coded.groupby("lith").agg(intervals=("length", "size"), metres=("length", "sum"),
                                          holes=("hole_id", "nunique"))  # fmt: skip
    printed = section(out, "Logged codes")
    assert len(printed) == len(spellings)
    lines = {line.split("'")[1]: line for line in out.splitlines() if line.startswith("  '")}
    assert set(lines) == set(spellings.index)
    for raw, row in spellings.iterrows():
        tail = lines[raw].split()[-3:]
        assert int(tail[0]) == row["intervals"] and int(tail[2]) == row["holes"]
        assert float(tail[1]) == pytest.approx(row["metres"], abs=0.006)
    assert (
        "NOT MAPPED" in lines["qz"] and "NOT MAPPED" in lines["Qz"] and "NOT MAPPED" in lines["xx"]
    )
    assert " DIO " in lines["hb  dio"] and " Q4 " in lines["soil"]

    # by group, and the unit that is in the dictionary but not in the log
    groups = {row[0]: row for row in section(out, "Metres by group")}
    assert int(groups["SED"][2]) == expected.loc["SST", "intervals"]
    assert "Units of the dictionary with no logged interval: GRD" in out

    # the three provenance tables
    titles = ["Measured: real site data used", "Assumed: chosen by the software",
              "Missing: inputs still needed"]  # fmt: skip
    at = [out.index(title) for title in titles]
    assert at == sorted(at)
    measured, assumed, missing = out[at[0] : at[1]], out[at[1] : at[2]], out[at[2] :]
    for unit in ("Q4", "SST", "AND", "DIO", "CL"):
        row = expected.loc[unit]
        assert f"Log of unit {unit}" in measured
        assert (f"{int(row['intervals'])} intervals, {row['metres']:.2f} m, "
                f"{int(row['holes'])} holes") in measured  # fmt: skip
    assert "Log of unit BX" not in measured
    assert "Unit BX is not modelled" in assumed
    assert (
        f"{expected.loc['BX', 'intervals']} intervals, {expected.loc['BX', 'metres']:.2f} m"
        in assumed
    )
    assert "Default density" in assumed and "2.7 t/m3" in assumed
    unmapped = coded[coded["unit"].isna()]
    assert "Logged codes not in the unit dictionary: 'QZ', 'XX'" in missing
    assert f"2 codes, {len(unmapped)} intervals, {unmapped['length'].sum():.2f} m" in missing
    for unit in ("SST", "AND", "DIO", "GRD"):
        assert f"Density of unit {unit}" in missing
    assert "Density of unit Q4" not in missing  # it has a measured density


def test_check_exit_code_follows_the_policy(tmp_path, capsys, database):
    strict = write_dictionary(tmp_path / "strict.json")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", strict,
                         "--allow-unmapped")  # fmt: skip
    assert code == 0 and err == "" and "'QZ', 'XX'" in out

    lenient = write_dictionary(tmp_path / "lenient.json", unmapped="unknown")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", lenient)
    assert code == 0 and err == ""
    assert "read as unknown: 'QZ', 'XX'" in out
    assert "Logged codes not in the unit dictionary: 'QZ', 'XX'" in out

    data = json.loads(json.dumps(DICTIONARY))
    data["units"].append({"code": "QZ", "kind": "other", "aliases": ["xx"]})
    full = tmp_path / "full.json"
    full.write_text(json.dumps(data), encoding="utf-8")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", full)
    assert code == 0 and err == ""
    assert "NOT MAPPED" not in out and "Logged codes not in" not in out
    assert "Unit QZ is not modelled" in out


def test_check_counts_density_samples_per_unit(tmp_path, capsys, database, log):
    units_file = write_dictionary(tmp_path / "units.json", unmapped="unknown")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", units_file,
                         "--density-table", "density", "--density-column", "sg")  # fmt: skip
    assert code == 0, err
    hand = counted(log)
    density = pd.read_csv(tmp_path / "density.csv")
    values: dict[str, list[float]] = {}
    for sample in density.itertuples(index=False):
        mid = (sample.depth_from + sample.depth_to) / 2
        for row in hand[hand["hole_id"] == sample.hole_id].itertuples(index=False):
            if row.depth_from <= mid < row.depth_to and isinstance(row.unit, str):
                values.setdefault(row.unit, []).append(sample.sg)
    rows = {row[0]: row for row in section(out, "Density samples by unit")}
    assert set(rows) == set(values) and len(rows) >= 3
    for unit, row in rows.items():
        assert int(row[1]) == len(values[unit])
        assert float(row[2]) == pytest.approx(np.mean(values[unit]), abs=6e-4)
        assert f"Density samples in unit {unit}" in out

    code, _, err = run(capsys, "geology", "units", "check", database, "--units", units_file,
                       "--density-table", "density")  # fmt: skip
    assert code == 2 and "go together" in err


def test_check_reports_an_invalid_dictionary(tmp_path, capsys, database):
    data = json.loads(json.dumps(DICTIONARY))
    data["units"][2]["kind"] = "rock"
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    code, out, err = run(capsys, "geology", "units", "check", database, "--units", bad)
    assert code == 2 and out == ""
    assert "units[2] (code 'AND')" in err and "kind 'rock'" in err

    data = json.loads(json.dumps(DICTIONARY))
    data["units"][3]["aliases"] = ["soil"]
    bad.write_text(json.dumps(data), encoding="utf-8")
    code, _, err = run(capsys, "geology", "units", "check", database, "--units", bad)
    assert code == 2 and "'SOIL' already belongs to units[0] (code 'Q4')" in err

    code, _, err = run(capsys, "geology", "units", "check", database, "--units",
                       tmp_path / "missing.json")  # fmt: skip
    assert code == 2 and "missing.json" in err

    nameless = dict(DICTIONARY)
    del nameless["source"]
    bad.write_text(json.dumps(nameless), encoding="utf-8")
    code, _, err = run(capsys, "geology", "units", "check", database, "--units", bad)
    assert code == 2 and "--column" in err
    code, _, _ = run(capsys, "geology", "units", "check", database, "--units", bad,
                     "--column", "lith", "--allow-unmapped")  # fmt: skip
    assert code == 0
