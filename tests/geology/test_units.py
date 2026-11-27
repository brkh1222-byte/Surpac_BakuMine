"""Unit dictionary (P9.T3): schema, folding, mapping, reports against plain pandas."""

import copy
import json

import numpy as np
import pandas as pd
import pytest

from bakumine.core.geology.units import (
    MODELLED_KINDS,
    PALETTE,
    RESERVED_CODES,
    UNIT_KINDS,
    Unit,
    UnitDictionary,
    UnitsError,
    UnmappedCodeError,
    WeatheringZone,
    code_report,
    density_samples_by_unit,
    draft_dictionary,
    fold_code,
    unit_metres,
    units_provenance,
)

# The example of geology_contracts.md section 2 (made-up deposit).
EXAMPLE = {
    "format": "bakumine.units",
    "version": 1,
    "name": "demo deposit",
    "source": {"table": "geology", "column": "lith"},
    "fold": {"case": True, "strip": True},
    "unmapped": "error",
    "default_density": 2.70,
    "default_density_source": "assumed",
    "display_colours": {"ORE": "#e8c547"},
    "units": [
        {"code": "Q4", "name": "Soil and cover", "kind": "cover", "aliases": ["SOIL", "SOL"],
         "colour": "#d8c37a", "density": 1.90, "density_source": "measured"},
        {"code": "OX", "name": "Oxidised zone (logged)", "kind": "weathering", "aliases": ["OZ"]},
        {"code": "SST", "name": "Sandstone", "kind": "lithology", "group": "SED",
         "colour": "#c9b28f", "density": 2.55, "density_source": "measured"},
        {"code": "SLT", "name": "Siltstone", "kind": "lithology", "group": "SED",
         "colour": "#8f8a6e"},
        {"code": "AND", "name": "Andesite", "kind": "lithology", "colour": "#6e7f5a"},
        {"code": "DIO", "name": "Diorite", "kind": "intrusion", "aliases": ["HB DIO"], "age": 1,
         "colour": "#7a6a9c"},
        {"code": "GRD", "name": "Granodiorite dyke", "kind": "intrusion", "age": 2,
         "colour": "#c2557a"},
        {"code": "FLT", "name": "Fault zone", "kind": "fault-zone", "aliases": ["FB"]},
        {"code": "CL", "name": "Core loss", "kind": "missing"},
        {"code": "BX", "name": "Breccia, not modelled", "kind": "other", "aliases": ["BRECCIA"]},
    ],
    "weathering_zones": [
        {"code": "OXIDE", "name": "Oxidised", "colour": "#d98c3a", "density": 2.30,
         "density_source": "measured"},
        {"code": "TRANSITION", "name": "Transition", "colour": "#c7b35a"},
        {"code": "FRESH", "name": "Fresh", "colour": "#7d8c99"},
    ],
}  # fmt: skip


def example(**top) -> dict:
    data = copy.deepcopy(EXAMPLE)
    data.update(top)
    return data


def with_unit(index: int, **fields) -> dict:
    data = example()
    data["units"][index].update(fields)
    return data


def with_zone(index: int, **fields) -> dict:
    data = example()
    data["weathering_zones"][index].update(fields)
    return data


# ---------------------------------------------------------------------- schema


def test_constants():
    assert UNIT_KINDS == ("lithology", "intrusion", "cover", "weathering", "fault-zone", "other",
                          "missing")  # fmt: skip
    assert MODELLED_KINDS == ("lithology", "intrusion")
    assert RESERVED_CODES == ("AIR", "UNK", "ORE")
    assert len(PALETTE) == len(set(PALETTE)) == 20


def test_contract_example_loads_and_writes_back_exactly():
    units = UnitDictionary.from_dict(EXAMPLE)
    assert units.name == "demo deposit"
    assert (units.source_table, units.source_column) == ("geology", "lith")
    assert units.fold_case and units.strip and units.unmapped == "error"
    assert units.default_density == 2.7 and units.default_density_source == "assumed"
    assert len(units.units) == 10 and len(units.weathering_zones) == 3
    assert units.unit("DIO") == Unit(code="DIO", kind="intrusion", name="Diorite",
                                     aliases=("HB DIO",), age=1, colour="#7a6a9c")  # fmt: skip
    assert units.to_dict() == EXAMPLE
    assert UnitDictionary.from_dict(units.to_dict()) == units


def test_save_and_load(tmp_path):
    units = UnitDictionary.from_dict(EXAMPLE)
    path = tmp_path / "sub" / "units.json"
    units.save(path)
    assert json.loads(path.read_text(encoding="utf-8")) == EXAMPLE
    assert UnitDictionary.load(path) == units


def test_minimal_dictionary_and_defaults(tmp_path):
    data = {"format": "bakumine.units", "version": 1, "units": [{"code": "A", "kind": "other"}]}
    units = UnitDictionary.from_dict(data)
    assert units.unmapped == "error" and units.fold_case and units.strip
    assert units.default_density is None and units.weathering_zones == ()
    assert units.unit("A").aliases == () and units.unit("A").name is None
    units.save(tmp_path / "u.json")
    assert UnitDictionary.load(tmp_path / "u.json") == units


@pytest.mark.parametrize("kind", ["rock", "Lithology", "fault_zone", "", None])
def test_unknown_kind(kind):
    with pytest.raises(UnitsError, match=r"units\[4\] \(code 'AND'\).*kind"):
        UnitDictionary.from_dict(with_unit(4, kind=kind))


@pytest.mark.parametrize("code", ["sst", "Sst", "1A", "_A", "S ST", "S-T", "", "A" * 33, 5, None])
def test_bad_code_pattern(code):
    with pytest.raises(UnitsError, match=r"units\[2\].*code"):
        UnitDictionary.from_dict(with_unit(2, code=code))
    with pytest.raises(UnitsError, match=r"weathering_zones\[1\].*code"):
        UnitDictionary.from_dict(with_zone(1, code=code))


def test_longest_and_shortest_codes_are_accepted():
    data = with_unit(2, code="A" * 32)
    data["units"][3]["code"] = "B"
    data["units"][4]["code"] = "C_9"
    assert UnitDictionary.from_dict(data).codes()[2:5] == ("A" * 32, "B", "C_9")


@pytest.mark.parametrize("code", RESERVED_CODES)
def test_reserved_codes(code):
    with pytest.raises(UnitsError, match=f"units\\[0\\].*{code}.*reserved"):
        UnitDictionary.from_dict(with_unit(0, code=code))
    with pytest.raises(UnitsError, match=f"weathering_zones\\[2\\].*{code}.*reserved"):
        UnitDictionary.from_dict(with_zone(2, code=code))


def test_reserved_word_as_an_alias_is_allowed():
    units = UnitDictionary.from_dict(with_unit(4, aliases=["ore"]))
    assert units.lookup()["ORE"] == "AND"


def test_duplicate_codes():
    with pytest.raises(UnitsError, match=r"units\[3\] \(code 'SST'\).*already used by units\[2\]"):
        UnitDictionary.from_dict(with_unit(3, code="SST"))
    with pytest.raises(UnitsError, match=r"weathering_zones\[0\] \(code 'OX'\).*already used"):
        UnitDictionary.from_dict(with_zone(0, code="OX"))
    with pytest.raises(UnitsError, match=r"weathering_zones\[2\].*already used"):
        UnitDictionary.from_dict(with_zone(2, code="OXIDE"))


@pytest.mark.parametrize(
    "index, aliases",
    [
        (4, ["SOIL"]),  # the alias of another unit
        (4, ["  soil "]),  # the same after folding
        (4, ["hb   dio"]),  # inner whitespace run and case
        (4, ["SST"]),  # the code of another unit
        (0, ["and"]),  # ... of a later unit
    ],
)
def test_duplicate_raw_codes(index, aliases):
    with pytest.raises(UnitsError, match="already belongs to .*one unit only"):
        UnitDictionary.from_dict(with_unit(index, aliases=aliases))


def test_duplicate_raw_codes_name_both_entries():
    with pytest.raises(UnitsError) as error:
        UnitDictionary.from_dict(with_unit(4, aliases=["oz"]))
    assert "units[4] (code 'AND')" in str(error.value)
    assert "'OZ'" in str(error.value) and "units[1] (code 'OX')" in str(error.value)


def test_repeats_inside_one_unit_are_merged():
    units = UnitDictionary.from_dict(with_unit(0, aliases=["soil", "SOIL ", "q4", "Sol"]))
    assert units.unit("Q4").aliases == ("SOIL", "SOL")


@pytest.mark.parametrize(
    "data, word",
    [
        (example(format="bakumine.other"), "format"),
        (example(version=2), "version"),
        (example(unmapped="ignore"), "unmapped"),
        (example(default_density=0), "default_density"),
        (example(default_density_source="guessed"), "default_density"),
        (example(default_density=None), "default_density_source"),
        (example(display_colours={"ORE": "yellow"}), "display_colours"),
        (example(fold={"case": "yes"}), "fold"),
        (example(fold={"upper": True}), "fold"),
        (example(units={}), "units"),
        (example(colour="#000000"), "unknown field"),
        (with_unit(2, colour="c9b28f"), "colour"),
        (with_unit(2, colour="#c9b28"), "colour"),
        (with_unit(2, density=-1.0), "density"),
        (with_unit(2, density="heavy"), "density"),
        (with_unit(2, density=float("nan")), "density"),
        (with_unit(2, density_source="lab"), "density_source"),
        (with_unit(3, density_source="measured"), "without a density"),
        (with_unit(5, age=1.5), "age"),
        (with_unit(5, age="old"), "age"),
        (with_unit(5, aliases="HB DIO"), "aliases"),
        (with_unit(5, aliases=[""]), "alias"),
        (with_unit(5, alias=["X"]), "unknown field"),
        (with_zone(0, density=0.0), "density"),
        (with_zone(0, kind="weathering"), "unknown field"),
    ],
)
def test_other_schema_errors(data, word):
    with pytest.raises(UnitsError, match=word):
        UnitDictionary.from_dict(data)


def test_missing_required_fields():
    data = example()
    del data["units"][1]["kind"]
    with pytest.raises(UnitsError, match=r"units\[1\].*'kind' is required"):
        UnitDictionary.from_dict(data)
    data = example()
    del data["weathering_zones"][0]["code"]
    with pytest.raises(UnitsError, match=r"weathering_zones\[0\].*'code' is required"):
        UnitDictionary.from_dict(data)


def test_direct_construction_is_checked_too():
    with pytest.raises(UnitsError, match="kind"):
        UnitDictionary(units=(Unit("A", "rock"),))
    with pytest.raises(UnitsError, match="reserved"):
        UnitDictionary(units=(Unit("A", "other"),), weathering_zones=(WeatheringZone("AIR"),))
    units = UnitDictionary(units=(Unit("A", "other", aliases=("a b", "A"), colour="#ABCDEF",
                                       density=2.5),))  # fmt: skip
    assert units.unit("A") == Unit("A", "other", aliases=("A B",), colour="#abcdef",
                                   density=2.5, density_source="assumed")  # fmt: skip


def test_colour_case_and_density_source_default():
    units = UnitDictionary.from_dict(with_unit(3, colour="#8F8A6E", density=2.4))
    assert units.unit("SLT").colour == "#8f8a6e"
    assert units.unit("SLT").density_source == "assumed"
    assert units.to_dict()["units"][3]["density_source"] == "assumed"


def test_load_reports_bad_json(tmp_path):
    path = tmp_path / "units.json"
    path.write_text("{not json", encoding="utf-8")
    with pytest.raises(UnitsError, match="not valid JSON"):
        UnitDictionary.load(path)


# ---------------------------------------------------------------------- folding


@pytest.mark.parametrize(
    "raw, folded",
    [
        ("SST", "SST"),
        ("sst", "SST"),
        ("  Sst\t", "SST"),
        ("hb   dio", "HB DIO"),
        (" hb \t dio\n", "HB DIO"),
        ("", None),
        ("   ", None),
        (None, None),
        (float("nan"), None),
        (np.nan, None),
        (pd.NA, None),
        (7, "7"),
        (7.0, "7"),
        (np.float64(12.0), "12"),
        (2.5, "2.5"),
    ],
)
def test_fold_code(raw, folded):
    assert fold_code(raw) == folded


def test_fold_code_switches():
    assert fold_code("  hb   Dio ", case=False) == "hb Dio"
    assert fold_code("  hb   Dio ", strip=False) == "  HB   DIO "
    assert fold_code("  hb   Dio ", case=False, strip=False) == "  hb   Dio "
    assert fold_code("", case=False, strip=False) is None


def test_lookup_holds_codes_and_aliases():
    lookup = UnitDictionary.from_dict(EXAMPLE).lookup()
    assert lookup["SST"] == "SST" and lookup["Q4"] == "Q4"
    assert lookup["SOIL"] == "Q4" and lookup["SOL"] == "Q4" and lookup["HB DIO"] == "DIO"
    assert len(lookup) == 10 + 6
    assert "OXIDE" not in lookup  # weathering zones are not logged codes


def test_map_codes_folds_case_and_whitespace():
    units = UnitDictionary.from_dict(EXAMPLE)
    values = pd.Series(
        ["sst", " SST ", "Sst", "soil", "SOL", "q4", "hb  dio", " Hb\tDio", "breccia", "cl",
         None, np.nan, "", "  "],
        index=np.arange(100, 114),
    )  # fmt: skip
    mapped = units.map_codes(values)
    assert mapped.dtype == object and mapped.index.equals(values.index)
    assert mapped.iloc[:10].tolist() == ["SST", "SST", "SST", "Q4", "Q4", "Q4", "DIO", "DIO",
                                         "BX", "CL"]  # fmt: skip
    assert mapped.iloc[10:].isna().all()  # missing stays missing, also under "error"


def test_unmapped_codes_are_an_error_by_default():
    units = UnitDictionary.from_dict(EXAMPLE)
    values = pd.Series(["sst", "qz", "XX ", "qz", " Qz", None, "a  b"])
    with pytest.raises(UnmappedCodeError) as error:
        units.map_codes(values)
    assert error.value.codes == ("A B", "QZ", "XX")
    assert isinstance(error.value, UnitsError)
    for code in ("'A B'", "'QZ'", "'XX'"):
        assert code in str(error.value)


def test_unmapped_codes_become_nan_when_the_policy_says_unknown():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    mapped = units.map_codes(pd.Series(["sst", "qz", None, "XX", "and"]))
    assert mapped.tolist()[0] == "SST" and mapped.tolist()[4] == "AND"
    assert mapped.isna().tolist() == [False, True, True, True, False]


def test_folding_can_be_switched_off_in_the_dictionary():
    exact = UnitDictionary.from_dict(example(fold={"case": False, "strip": True},
                                             unmapped="unknown"))  # fmt: skip
    mapped = exact.map_codes(pd.Series(["SST", "sst", " SST ", "HB  DIO", "hb dio"]))
    assert mapped.isna().tolist() == [False, True, False, False, True]
    literal = UnitDictionary.from_dict(example(fold={"case": True, "strip": False},
                                               unmapped="unknown"))  # fmt: skip
    mapped = literal.map_codes(pd.Series(["SST", "sst", " SST ", "HB  DIO", "hb dio"]))
    assert mapped.isna().tolist() == [False, False, True, True, False]


# ---------------------------------------------------------------------- order, kinds, colours


def test_codes_keep_dictionary_order():
    units = UnitDictionary.from_dict(EXAMPLE)
    assert units.codes() == ("Q4", "OX", "SST", "SLT", "AND", "DIO", "GRD", "FLT", "CL", "BX")
    assert units.codes("lithology") == ("SST", "SLT", "AND")
    assert units.codes("intrusion") == ("DIO", "GRD")
    # several kinds: still dictionary order, not the order of the kinds asked for
    assert units.codes("intrusion", "cover", "lithology") == ("Q4", "SST", "SLT", "AND", "DIO",
                                                              "GRD")  # fmt: skip
    assert units.codes(*MODELLED_KINDS) == ("SST", "SLT", "AND", "DIO", "GRD")
    with pytest.raises(UnitsError):
        units.codes("rock")
    with pytest.raises(KeyError):
        units.unit("NOPE")


def test_order_follows_the_file_and_zones_run_top_to_bottom():
    data = example()
    data["units"].reverse()
    data["weathering_zones"] = [data["weathering_zones"][i] for i in (2, 0, 1)]
    units = UnitDictionary.from_dict(data)
    assert units.codes() == ("BX", "CL", "FLT", "GRD", "DIO", "AND", "SLT", "SST", "OX", "Q4")
    assert units.codes("lithology") == ("AND", "SLT", "SST")
    assert [z.code for z in units.weathering_zones] == ["FRESH", "OXIDE", "TRANSITION"]
    assert [z["code"] for z in units.to_dict()["weathering_zones"]] == ["FRESH", "OXIDE",
                                                                         "TRANSITION"]  # fmt: skip
    zones = UnitDictionary.from_dict(EXAMPLE).weathering_zones
    assert [z.code for z in zones] == ["OXIDE", "TRANSITION", "FRESH"]  # last = deepest


def test_kinds_of_and_mask_kinds():
    units = UnitDictionary.from_dict(EXAMPLE)
    codes = pd.Series(["SST", "DIO", np.nan, "Q4", "BX", "CL", "OX"], index=list("abcdefg"))
    kinds = units.kinds_of(codes)
    assert kinds.index.equals(codes.index)
    assert kinds.tolist()[:2] == ["lithology", "intrusion"] and pd.isna(kinds["c"])
    assert kinds.tolist()[3:] == ["cover", "other", "missing", "weathering"]
    kept = units.mask_kinds(codes, MODELLED_KINDS)
    assert kept.index.equals(codes.index)
    assert kept.tolist()[:2] == ["SST", "DIO"] and kept.iloc[2:].isna().all()
    assert units.mask_kinds(codes, ["cover"]).dropna().tolist() == ["Q4"]
    assert units.mask_kinds(codes, []).isna().all()
    with pytest.raises(UnitsError, match="ZZ"):
        units.kinds_of(pd.Series(["SST", "ZZ"]))


def test_colours_explicit_palette_by_position_and_display():
    units = UnitDictionary.from_dict(EXAMPLE)
    colours = units.colours()
    assert list(colours) == [*units.codes(), "OXIDE", "TRANSITION", "FRESH", "ORE"]
    assert colours["SST"] == "#c9b28f" and colours["OXIDE"] == "#d98c3a"
    assert colours["ORE"] == "#e8c547"
    # no colour in the file: the palette entry of the position (units, then zones)
    assert colours["OX"] == PALETTE[1] and colours["FLT"] == PALETTE[7]
    assert colours["CL"] == PALETTE[8] and colours["BX"] == PALETTE[9]
    data = with_zone(1)
    del data["weathering_zones"][1]["colour"]
    assert UnitDictionary.from_dict(data).colours()["TRANSITION"] == PALETTE[11]
    # a display colour wins over a unit colour of the same code
    shown = UnitDictionary.from_dict(example(display_colours={"SST": "#FFFFFF"})).colours()
    assert shown["SST"] == "#ffffff"
    assert all(len(c) == 7 and c == c.lower() for c in colours.values())


def test_palette_wraps_after_twenty_entries():
    many = UnitDictionary(units=tuple(Unit(f"U{i}", "lithology") for i in range(23)))
    colours = many.colours()
    assert colours["U20"] == PALETTE[0] and colours["U22"] == PALETTE[2]


def test_density_of_units_and_zones():
    units = UnitDictionary.from_dict(EXAMPLE)
    assert units.density("SST") == (2.55, "measured")
    assert units.density("Q4") == (1.9, "measured")
    assert units.density("SLT") == (2.7, "assumed")  # the default density
    assert units.density("OXIDE") == (2.3, "measured")
    assert units.density("FRESH") == (None, "assumed")  # a zone without one does not override
    with pytest.raises(KeyError):
        units.density("NOPE")
    data = example()
    del data["default_density"], data["default_density_source"]
    assert UnitDictionary.from_dict(data).density("SLT") == (None, "assumed")
    stated = UnitDictionary.from_dict(example(default_density_source="measured"))
    assert stated.density("AND") == (2.7, "measured")


# ---------------------------------------------------------------------- tables against pandas

SPELLINGS = {
    "SST": ["SST", "sst", " Sst "],
    "Q4": ["soil", "SOIL", "Sol", "q4"],
    "DIO": ["hb  dio", "HB DIO", "dio"],
    "AND": ["and"],
    "SLT": ["slt", "SLT "],
    "BX": ["breccia", "BX"],
    "CL": ["cl"],
    None: ["qz", " QZ", "xx"],  # not in the dictionary
}


def logged_table(seed: int = 3, holes: int = 9) -> pd.DataFrame:
    """A made-up log: contiguous intervals with many spellings and some missing codes."""
    rng = np.random.default_rng(seed)
    pool = [s for group in SPELLINGS.values() for s in group] + [None, np.nan, ""]
    rows = []
    for h in range(holes):
        depth = 0.0
        for _ in range(int(rng.integers(8, 16))):
            length = float(np.round(rng.uniform(0.4, 9.0), 2))
            rows.append((f"H{h:02d}", depth, depth + length, pool[int(rng.integers(len(pool)))]))
            depth += length
    table = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "lith"])
    return table.sample(frac=1.0, random_state=seed).reset_index(drop=True)


def by_hand(table: pd.DataFrame) -> pd.DataFrame:
    """The log with folded code, unit and length, made with pandas string methods only."""
    text = table["lith"].astype("string")
    folded = text.str.strip().str.replace(r"\s+", " ", regex=True).str.upper()
    folded = folded.mask(folded == "")
    to_unit = {s.strip().upper().replace("  ", " "): unit
               for unit, group in SPELLINGS.items() if unit for s in group}  # fmt: skip
    out = table.assign(folded=folded.astype(object), length=table["depth_to"] - table["depth_from"])
    out["unit"] = out["folded"].map(to_unit)
    return out[out["folded"].notna()]


def test_code_report_matches_a_pandas_groupby():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    table = logged_table()
    report = code_report(table, "lith", units)
    assert list(report.columns) == ["raw", "folded", "unit", "kind", "intervals", "metres",
                                    "holes", "mapped"]  # fmt: skip
    hand = by_hand(table)
    expected = hand.groupby("lith").agg(
        intervals=("length", "size"), metres=("length", "sum"), holes=("hole_id", "nunique"),
        folded=("folded", "first"),
    )  # fmt: skip
    assert len(report) == len(expected) == hand["lith"].nunique()
    assert report["raw"].is_unique
    got = report.set_index("raw")
    for raw, row in expected.iterrows():
        assert got.loc[raw, "intervals"] == row["intervals"]
        assert got.loc[raw, "holes"] == row["holes"]
        assert got.loc[raw, "metres"] == pytest.approx(row["metres"], rel=1e-12)
        assert got.loc[raw, "folded"] == row["folded"]
    assert got.loc[" Sst ", "unit"] == "SST" and got.loc["hb  dio", "unit"] == "DIO"
    assert got.loc["Sol", "unit"] == "Q4" and got.loc["Sol", "kind"] == "cover"
    assert got.loc["breccia", "kind"] == "other"
    assert report["mapped"].dtype == bool
    unmapped = report[~report["mapped"]]
    assert sorted(unmapped["raw"]) == [" QZ", "qz", "xx"]
    assert unmapped["unit"].isna().all() and unmapped["kind"].isna().all()
    # totals: every interval with a code is counted once
    assert report["intervals"].sum() == len(hand)
    assert report["metres"].sum() == pytest.approx(hand["length"].sum(), rel=1e-12)
    # the same table under the "error" policy is still reported, not raised
    strict = code_report(table, "lith", UnitDictionary.from_dict(EXAMPLE))
    pd.testing.assert_frame_equal(strict, report)


def test_unit_metres_matches_a_pandas_groupby():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    table = logged_table(seed=11)
    metres = unit_metres(table, "lith", units)
    assert list(metres.columns) == ["unit", "kind", "intervals", "metres", "holes", "fraction"]
    hand = by_hand(table)
    expected = hand.groupby("unit").agg(
        intervals=("length", "size"), metres=("length", "sum"), holes=("hole_id", "nunique")
    )
    assert sorted(metres["unit"]) == sorted(expected.index)
    assert metres["unit"].tolist() == [c for c in units.codes() if c in expected.index]
    got = metres.set_index("unit")
    total = hand["length"].sum()
    for unit, row in expected.iterrows():
        assert got.loc[unit, "intervals"] == row["intervals"]
        assert got.loc[unit, "holes"] == row["holes"]
        assert got.loc[unit, "metres"] == pytest.approx(row["metres"], rel=1e-12)
        assert got.loc[unit, "fraction"] == pytest.approx(row["metres"] / total, rel=1e-12)
        assert got.loc[unit, "kind"] == units.unit(unit).kind
    unmapped_metres = hand.loc[hand["unit"].isna(), "length"].sum()
    assert unmapped_metres > 0
    assert metres["fraction"].sum() == pytest.approx(1.0 - unmapped_metres / total, rel=1e-12)


def test_unit_metres_by_group():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    table = logged_table(seed=5)
    groups = unit_metres(table, "lith", units, by="group")
    assert list(groups.columns) == ["group", "kind", "intervals", "metres", "holes", "fraction"]
    hand = by_hand(table)
    hand = hand[hand["unit"].notna()]
    key = hand["unit"].replace({"SST": "SED", "SLT": "SED"})
    expected = hand.groupby(key).agg(
        intervals=("length", "size"), metres=("length", "sum"), holes=("hole_id", "nunique")
    )
    assert groups["group"].tolist() == ["Q4", "SED", "AND", "DIO", "CL", "BX"]
    got = groups.set_index("group")
    for name, row in expected.iterrows():
        assert got.loc[name, "intervals"] == row["intervals"]
        assert got.loc[name, "holes"] == row["holes"]
        assert got.loc[name, "metres"] == pytest.approx(row["metres"], rel=1e-12)
    assert got.loc["SED", "kind"] == "lithology"
    mixed = with_unit(5, group="SED")
    mixed["unmapped"] = "unknown"
    by_group = unit_metres(table, "lith", UnitDictionary.from_dict(mixed), by="group")
    assert by_group.set_index("group").loc["SED", "kind"] == "mixed"
    with pytest.raises(UnitsError):
        unit_metres(table, "lith", units, by="kind")


def test_reports_need_the_interval_columns():
    units = UnitDictionary.from_dict(EXAMPLE)
    table = logged_table().drop(columns="depth_to")
    with pytest.raises(UnitsError, match="depth_to"):
        code_report(table, "lith", units)
    with pytest.raises(UnitsError, match="rock"):
        unit_metres(logged_table(), "rock", units)


def test_a_missing_depth_does_not_become_a_length():
    units = UnitDictionary.from_dict(EXAMPLE)
    table = pd.DataFrame(
        {
            "hole_id": ["H1", "H1", "H2"],
            "depth_from": [0.0, 4.0, 0.0],
            "depth_to": [4.0, np.nan, 2.5],
            "lith": ["sst", "sst", "SST"],
        }
    )
    metres = unit_metres(table, "lith", units).set_index("unit")
    assert metres.loc["SST", "intervals"] == 3 and metres.loc["SST", "holes"] == 2
    assert metres.loc["SST", "metres"] == pytest.approx(6.5)


# ---------------------------------------------------------------------- draft


def test_draft_dictionary_has_one_lithology_unit_per_folded_code():
    table = logged_table(seed=7)
    draft = draft_dictionary(table, "lith", table_name="logs")
    hand = by_hand(table)
    metres = hand.groupby("folded")["length"].sum()
    assert len(draft.units) == hand["folded"].nunique()
    assert {u.kind for u in draft.units} == {"lithology"}
    assert (draft.source_table, draft.source_column) == ("logs", "lith")
    assert draft.unmapped == "error" and draft.weathering_zones == ()
    lookup = draft.lookup()
    assert set(metres.index) <= set(lookup)
    # largest first
    ordered = [float(metres[u.aliases[0] if u.aliases else u.code]) for u in draft.units]
    assert ordered == sorted(ordered, reverse=True)
    # every logged spelling maps, and the draft is a valid file
    mapped = draft.map_codes(table["lith"])
    assert mapped.notna().sum() == len(hand)
    assert UnitDictionary.from_dict(draft.to_dict()) == draft
    report = unit_metres(table, "lith", draft)
    assert report["fraction"].sum() == pytest.approx(1.0)


def test_draft_codes_are_made_valid():
    table = pd.DataFrame({
        "hole_id": ["H1"] * 8,
        "depth_from": np.arange(8.0),
        "depth_to": np.arange(8.0) + [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3],
        "lith": ["hb  dio", "ore", 7.0, "a-b", "a b", "air", "x" * 40, "??"],
    })  # fmt: skip
    draft = draft_dictionary(table, "lith")
    by_alias = {(u.aliases[0] if u.aliases else u.code): u.code for u in draft.units}
    assert by_alias["HB DIO"] == "HB_DIO"
    assert by_alias["ORE"] == "ORE_LOG" and by_alias["AIR"] == "AIR_LOG"
    assert by_alias["7"] == "U7"
    assert {by_alias["A-B"], by_alias["A B"]} == {"A_B", "A_B_2"}
    assert by_alias["X" * 40] == "X" * 32
    assert by_alias["??"] == "U8"
    assert draft.map_codes(table["lith"]).notna().all()


def test_draft_without_case_folding_keeps_spellings_apart():
    table = pd.DataFrame({"hole_id": ["H1"] * 3, "depth_from": [0.0, 1.0, 3.0],
                          "depth_to": [1.0, 3.0, 6.0], "lith": ["sst", "SST", "Sst"]})  # fmt: skip
    assert len(draft_dictionary(table, "lith").units) == 1
    apart = draft_dictionary(table, "lith", fold_case=False)
    assert len(apart.units) == 3 and not apart.fold_case
    assert apart.map_codes(table["lith"]).nunique() == 3


# ---------------------------------------------------------------------- density samples


def test_density_samples_by_unit_matches_a_direct_loop():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    lith = logged_table(seed=21)
    rng = np.random.default_rng(4)
    rows = []
    for hole in [*sorted(lith["hole_id"].unique()), "H99"]:  # H99 has no log
        for _ in range(25):
            start = float(np.round(rng.uniform(0.0, 80.0), 2))
            rows.append((hole, start, start + 0.2, float(np.round(rng.uniform(1.8, 3.2), 3))))
    density = pd.DataFrame(rows, columns=["hole_id", "depth_from", "depth_to", "sg"])
    density.loc[::17, "sg"] = np.nan

    hand = by_hand(lith)
    values: dict[str, list[float]] = {}
    for sample in density.itertuples(index=False):
        if np.isnan(sample.sg):
            continue
        mid = (sample.depth_from + sample.depth_to) / 2
        for row in hand[hand["hole_id"] == sample.hole_id].itertuples(index=False):
            if row.depth_from <= mid < row.depth_to and isinstance(row.unit, str):
                values.setdefault(row.unit, []).append(sample.sg)
    assert len(values) >= 5

    got = density_samples_by_unit(lith, "lith", density, "sg", units)
    assert list(got.columns) == ["unit", "n", "mean", "min", "max"]
    assert got["unit"].tolist() == [c for c in units.codes() if c in values]
    for row in got.itertuples(index=False):
        assert row.n == len(values[row.unit])
        assert row.mean == pytest.approx(np.mean(values[row.unit]), rel=1e-12)
        assert row.min == min(values[row.unit]) and row.max == max(values[row.unit])


def test_density_samples_at_points_and_on_a_contact():
    units = UnitDictionary.from_dict(EXAMPLE)
    lith = pd.DataFrame({"hole_id": ["H1", "H1", "H2"], "depth_from": [0.0, 10.0, 0.0],
                         "depth_to": [10.0, 20.0, 5.0], "lith": ["sst", "and", "SST"]})  # fmt: skip
    points = pd.DataFrame({"hole_id": ["H1", "H1", "H1", "H2", "H2"],
                           "depth": [5.0, 10.0, 20.0, 1.0, 7.0],
                           "sg": [2.5, 2.9, 3.5, 2.7, 9.9]})  # fmt: skip
    got = density_samples_by_unit(lith, "lith", points, "sg", units).set_index("unit")
    # 10.0 is the top of the andesite; 20.0 is the end of the hole; 7.0 is below the log
    assert got.loc["SST", ["n", "mean", "min", "max"]].tolist() == [2, 2.6, 2.5, 2.7]
    assert got.loc["AND", ["n", "mean"]].tolist() == [1, 2.9]
    with pytest.raises(UnitsError, match="depth"):
        density_samples_by_unit(lith, "lith", points.drop(columns="depth"), "sg", units)


# ---------------------------------------------------------------------- provenance


def test_units_provenance_lists_measured_assumed_missing():
    units = UnitDictionary.from_dict(example(unmapped="unknown"))
    table = logged_table(seed=3)
    entries = units_provenance(table, "lith", units)
    assert {e.stage for e in entries} == {"units"} and {e.task for e in entries} == {"P9.T3"}
    by_kind = {k: [e for e in entries if e.kind == k] for k in ("measured", "assumed", "missing")}
    metres = unit_metres(table, "lith", units).set_index("unit")

    # measured: one line per logged unit that is used, with its counts
    logs = {e.item: e for e in by_kind["measured"]}
    assert set(logs) == {f"Log of unit {u}" for u in metres.index if u != "BX"}
    sst = logs["Log of unit SST"]
    assert sst.count == metres.loc["SST", "intervals"]
    assert sst.detail == (f"{metres.loc['SST', 'intervals']} intervals, "
                          f"{metres.loc['SST', 'metres']:.2f} m, "
                          f"{metres.loc['SST', 'holes']} holes")  # fmt: skip
    assert sst.source == "table geology, column lith" and sst.used_for == "rock-unit fields"
    assert logs["Log of unit Q4"].used_for == "base surface of the cover"

    # assumed: the unit of kind other, and the default density
    assumed = {e.item: e for e in by_kind["assumed"]}
    assert set(assumed) == {"Unit BX is not modelled", "Default density"}
    assert assumed["Unit BX is not modelled"].count == metres.loc["BX", "intervals"]
    assert assumed["Default density"].value == 2.7

    # missing: the unmapped codes, and rock units without a density of their own
    missing = {e.item: e for e in by_kind["missing"]}
    unmapped = [item for item in missing if item.startswith("Logged codes not in")]
    assert len(unmapped) == 1 and "'QZ'" in unmapped[0] and "'XX'" in unmapped[0]
    hand = by_hand(table)
    assert missing[unmapped[0]].count == int(hand["unit"].isna().sum())
    assert f"{hand.loc[hand['unit'].isna(), 'length'].sum():.2f} m" in missing[unmapped[0]].detail
    no_density = {item for item in missing if item.startswith("Density of unit")}
    assert no_density == {f"Density of unit {u}" for u in ("SLT", "AND", "DIO", "GRD")}
    assert "default density 2.7" in missing["Density of unit SLT"].detail
    assert "not logged" in missing["Density of unit GRD"].detail


def test_units_provenance_with_density_samples_and_a_measured_default():
    data = example(default_density_source="measured")
    units = UnitDictionary.from_dict(data)
    lith = pd.DataFrame({"hole_id": ["H1", "H1"], "depth_from": [0.0, 10.0],
                         "depth_to": [10.0, 20.0], "lith": ["sst", "and"]})  # fmt: skip
    density = pd.DataFrame({"hole_id": ["H1", "H1", "H1"], "depth_from": [1.0, 12.0, 14.0],
                            "depth_to": [2.0, 13.0, 15.0], "sg": [2.5, 2.8, 3.0]})  # fmt: skip
    samples = density_samples_by_unit(lith, "lith", density, "sg", units)
    entries = units_provenance(lith, "lith", units, density_samples=samples,
                               density_source="table density, column sg")  # fmt: skip
    items = {e.item: e for e in entries}
    assert items["Density samples in unit AND"].kind == "measured"
    assert items["Density samples in unit AND"].count == 2
    assert items["Density samples in unit AND"].value == pytest.approx(2.9)
    assert items["Density samples in unit AND"].source == "table density, column sg"
    assert items["Default density"].kind == "measured"
    assert "2 samples in the density table" in items["Density of unit AND"].detail
    assert not any(e.item.startswith("Logged codes not in") for e in entries)
    assert not any(e.kind == "assumed" for e in entries)
