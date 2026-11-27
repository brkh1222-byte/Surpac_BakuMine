"""Provenance record (P9.T3): entries, the file round trip and the three tables."""

import json

import pandas as pd
import pytest

from bakumine.core.geology.provenance import (
    PROVENANCE_KINDS,
    Provenance,
    ProvenanceEntry,
    ProvenanceError,
)

# The example of geology_contracts.md section 3 (made-up values).
EXAMPLE = {
    "format": "bakumine.provenance",
    "version": 1,
    "object": {"kind": "geology", "name": "demo geology", "created": "2026-11-25T10:00:00Z",
               "bakumine_version": "0.1.0"},
    "entries": [
        {"item": "Rock-type log", "kind": "measured",
         "source": "database 'drillholes', table geology, column lith",
         "used_for": "rock-unit fields", "detail": "3412 intervals, 18940 m, 64 holes",
         "stage": "lithology", "task": "P9.T7", "count": 3412},
        {"item": "Holes without oxidation logging", "kind": "assumed",
         "source": "default (absent = unknown)", "used_for": "oxide base surface",
         "detail": "38 holes give no information, not zero depth",
         "stage": "surfaces", "task": "P9.T9", "count": 38},
        {"item": "Fault F1", "kind": "assumed",
         "source": "inferred from 14 block changes along holes",
         "used_for": "fault blocks FB1 / FB2",
         "detail": "plane dip 63, dip azimuth 072; 2 points on the wrong side",
         "stage": "faults", "task": "P9.T8"},
        {"item": "Fault surfaces", "kind": "missing",
         "source": "the geologist's fault interpretation (DXF / OBJ / points)",
         "used_for": "would replace the inferred planes", "detail": "", "stage": "faults",
         "task": "P9.T8"},
    ],
}  # fmt: skip


def test_kinds():
    assert PROVENANCE_KINDS == ("measured", "assumed", "missing")


def test_contract_example_round_trips_exactly():
    record = Provenance.from_dict(EXAMPLE)
    assert record.object_kind == "geology" and record.name == "demo geology"
    assert record.created == "2026-11-25T10:00:00Z"
    assert [e.kind for e in record.entries] == ["measured", "assumed", "assumed", "missing"]
    assert record.entries[0].count == 3412 and record.entries[2].count is None
    assert record.to_dict() == EXAMPLE


def test_save_and_load(tmp_path):
    record = Provenance(name="demo")
    record.measured("Log of unit AAA", source="table geology", used_for="rock-unit fields",
                    detail="5 intervals", stage="units", task="P9.T3", count=5)  # fmt: skip
    record.assumed("Default density", source="unit dictionary", used_for="tonnes", value=2.7)
    record.missing("Fault surfaces", source="the geologist", used_for="fault blocks",
                   value="DXF")  # fmt: skip
    path = tmp_path / "sub" / "provenance.json"
    record.save(path)
    data = json.loads(path.read_text(encoding="utf-8"))
    assert data["format"] == "bakumine.provenance" and data["version"] == 1
    assert set(data["object"]) == {"kind", "name", "created", "bakumine_version"}
    loaded = Provenance.load(path)
    assert loaded.entries == record.entries
    assert loaded.to_dict() == record.to_dict()
    assert loaded.entries[1].value == 2.7 and loaded.entries[2].value == "DXF"


def test_add_extend_and_helpers_keep_order():
    record = Provenance()
    assert record.entries == ()
    record.missing("c", source="s", used_for="u")
    record.add(ProvenanceEntry("a", "measured", "s", "u"))
    record.extend([ProvenanceEntry("b", "assumed", "s", "u"),
                   ProvenanceEntry("d", "measured", "s", "u")])  # fmt: skip
    assert [(e.item, e.kind) for e in record.entries] == [
        ("c", "missing"), ("a", "measured"), ("b", "assumed"), ("d", "measured")]  # fmt: skip
    assert isinstance(record.entries, tuple)
    with pytest.raises(ProvenanceError):
        record.add({"item": "x"})


def test_three_tables():
    record = Provenance.from_dict(EXAMPLE)
    tables = record.tables()
    assert list(tables) == ["measured", "assumed", "missing"]
    for table in tables.values():
        assert isinstance(table, pd.DataFrame)
        assert list(table.columns) == ["item", "source", "used_for", "detail", "stage"]
    assert tables["measured"]["item"].tolist() == ["Rock-type log"]
    assert tables["assumed"]["item"].tolist() == ["Holes without oxidation logging", "Fault F1"]
    assert tables["missing"]["item"].tolist() == ["Fault surfaces"]
    assert tables["missing"]["source"].iloc[0].startswith("the geologist's fault")
    assert tables["assumed"]["stage"].tolist() == ["surfaces", "faults"]
    # every entry is in exactly one table
    assert sum(len(t) for t in tables.values()) == len(record.entries)


def test_empty_record_still_has_three_tables():
    tables = Provenance().tables()
    assert list(tables) == ["measured", "assumed", "missing"]
    assert all(t.empty and len(t.columns) == 5 for t in tables.values())
    text = Provenance().format_text()
    assert text.count("(none)") == 3


def test_format_text_has_three_titled_tables_in_order():
    text = Provenance.from_dict(EXAMPLE).format_text()
    titles = ["Measured: real site data used (1)", "Assumed: chosen by the software (2)",
              "Missing: inputs still needed (1)"]  # fmt: skip
    positions = [text.index(title) for title in titles]
    assert positions == sorted(positions)
    blocks = text.split("\n\n")
    assert len(blocks) == 3
    assert "Rock-type log" in blocks[0] and "3412 intervals, 18940 m, 64 holes" in blocks[0]
    assert "Fault F1" in blocks[1] and "Holes without oxidation logging" in blocks[1]
    assert "Fault surfaces" in blocks[2] and "Fault F1" not in blocks[2]
    header = blocks[0].splitlines()[1].split()
    assert header == ["item", "source", "used_for", "detail", "stage"]


@pytest.mark.parametrize(
    "change, word",
    [
        ({"kind": "guessed"}, "kind"),
        ({"item": ""}, "item"),
        ({"source": "  "}, "source"),
        ({"used_for": None}, "used_for"),
        ({"stage": "somewhere"}, "stage"),
        ({"count": 2.5}, "count"),
        ({"count": True}, "count"),
        ({"value": float("nan")}, "value"),
        ({"value": [1]}, "value"),
        ({"detail": 5}, "detail"),
    ],
)
def test_invalid_entries(change, word):
    fields = {"item": "x", "kind": "measured", "source": "s", "used_for": "u"} | change
    with pytest.raises(ProvenanceError, match=word):
        ProvenanceEntry(**fields)


def test_invalid_files():
    with pytest.raises(ProvenanceError, match="format"):
        Provenance.from_dict({"format": "something", "version": 1, "entries": []})
    with pytest.raises(ProvenanceError, match="version"):
        Provenance.from_dict(EXAMPLE | {"version": 2})
    with pytest.raises(ProvenanceError, match="used_for"):
        Provenance.from_dict(EXAMPLE | {"entries": [{"item": "x", "kind": "missing",
                                                     "source": "s"}]})  # fmt: skip
    with pytest.raises(ProvenanceError, match="colour"):
        Provenance.from_dict(EXAMPLE | {"entries": [{"item": "x", "kind": "missing",
                                                     "source": "s", "used_for": "u",
                                                     "colour": "red"}]})  # fmt: skip


def test_whole_numbers_from_numpy_are_plain_ints():
    import numpy as np

    entry = ProvenanceEntry("x", "measured", "s", "u", count=np.int64(7), value=np.float64(2.5))
    assert type(entry.count) is int and type(entry.value) is float
    json.dumps(entry.to_dict())
