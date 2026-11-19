"""Block model persistence (P3.T8a): lossless, atomic, checksummed folder format."""

import json
import os
import struct
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from bakumine.core.blockmodel import BlockModel, BlockModelDefinition, storage
from bakumine.core.blockmodel.storage import (
    BLOCKS_FILE,
    HEADER_FILE,
    StorageError,
    load_model,
    read_header,
    save_model,
)

#: a quiet NaN with a non-default payload, and a negative NaN
NAN_PAYLOAD = struct.unpack("<d", struct.pack("<Q", 0x7FF8_0000_0000_BEEF))[0]
NAN_NEGATIVE = struct.unpack("<d", struct.pack("<Q", 0xFFF8_0000_0000_0001))[0]


def make_model(seed: int = 0, *, rotation: str = "angles") -> BlockModel:
    """Sub-blocked on three levels with anisotropic depth, all attribute kinds, awkward values."""
    rng = np.random.default_rng(seed)
    kwargs = {"max_depth": (3, 2, 1)}
    if rotation == "angles":
        kwargs["angles"] = (30.0, 10.0, 5.0)
    elif rotation == "matrix":
        c, s = np.cos(0.4), np.sin(0.4)
        kwargs["matrix"] = ((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0))
    definition = BlockModelDefinition((500400.5, 5300250.25, 825.0), (10.0, 12.5, 5.0), (5, 4, 3),
                                      **kwargs)  # fmt: skip
    model = BlockModel(definition)
    for _ in range(3):
        can = model.level < model.max_level
        model.split(can & (rng.random(model.n_blocks) < 0.3))
    n = model.n_blocks
    assert set(np.unique(model.level)) == {0, 1, 2, 3}

    model.add_attribute("au", unit="g/t", decimals=3, description="gold")
    values = rng.lognormal(size=n)
    values[::7] = np.nan
    values[1], values[2], values[3], values[4] = NAN_PAYLOAD, NAN_NEGATIVE, -0.0, np.inf
    values[5] = 5e-324  # smallest subnormal
    model["au"][:] = values
    model.add_attribute("density", background=2.7, dtype=np.float32)
    model["density"][:] = rng.uniform(2.5, 3.0, n).astype(np.float32)
    model["density"][::5] = np.nan
    model.add_attribute("pass_no", "int", dtype=np.int8, background=-1)
    model["pass_no"][:] = rng.integers(-1, 4, n)
    model.add_attribute("big", "int", dtype=np.int64)
    model["big"][:] = rng.integers(-(2**62), 2**62, n)
    # dictionary deliberately not alphabetical, with an unused entry and missing values
    model.add_attribute("rock", "category", dtype=np.int8, categories=["WASTE", "ORE", "UNUSED"])
    model["rock"][:] = rng.choice(np.array([-1, 0, 1], dtype=np.int8), n)
    model.add_attribute("zone", "category", background="outside")
    model.set("zone", "é zone, with comma", where=rng.random(n) < 0.3)
    model.add_attribute("empty")  # all background (NaN)
    return model


def assert_identical(a: BlockModel, b: BlockModel, names=None) -> None:
    assert a.definition == b.definition
    assert np.array_equal(a.definition.rotation, b.definition.rotation)
    for column in ("parent", "level", "sub"):
        x, y = getattr(a, column), getattr(b, column)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert np.array_equal(x, y)
    assert np.array_equal(a.block_ids(), b.block_ids())
    names = a.attribute_names if names is None else names
    assert b.attribute_names == names
    for name in names:
        assert a.attribute(name).to_dict() == b.attribute(name).to_dict()
        assert a.attribute(name).dtype == b.attribute(name).dtype
        assert a[name].dtype == b[name].dtype
        # bit-identical: NaN payloads, signed zero, subnormals
        assert a[name].tobytes() == b[name].tobytes(), name


@pytest.mark.parametrize("rotation", ["angles", "matrix", "none"])
@pytest.mark.parametrize("compression", ["zstd", "snappy", "none"])
def test_round_trip_is_lossless(tmp_path, rotation, compression):
    model = make_model(1, rotation=rotation)
    assert model.definition.is_rotated == (rotation != "none")
    header = save_model(model, tmp_path / "m", compression=compression)
    assert sorted(p.name for p in (tmp_path / "m").iterdir()) == [BLOCKS_FILE, HEADER_FILE]
    assert header.n_blocks == model.n_blocks and header.compression == compression
    loaded = load_model(tmp_path / "m")
    assert_identical(model, loaded)
    loaded.check_invariants()
    # the awkward values really went through
    bits = loaded["au"].view(np.uint64)
    assert bits[1] == 0x7FF8_0000_0000_BEEF and bits[2] == 0xFFF8_0000_0000_0001
    assert np.signbit(loaded["au"][3]) and loaded["au"][3] == 0.0
    assert loaded.attribute("rock").categories == ["WASTE", "ORE", "UNUSED"]
    assert loaded.attribute("zone").background == "outside"
    assert np.isnan(loaded.attribute("empty").background) and np.isnan(loaded["empty"]).all()
    assert (loaded["rock"] == -1).any()
    assert np.array_equal(loaded.get("zone", decode=True), model.get("zone", decode=True))
    assert np.array_equal(loaded.centroids(), model.centroids())


def test_dense_model_and_model_without_attributes(tmp_path):
    model = BlockModel(BlockModelDefinition((0.0, 0.0, 0.0), (1.0, 2.0, 3.0), (3, 4, 5)))
    save_model(model, tmp_path / "bare")
    loaded = load_model(tmp_path / "bare")
    assert loaded.is_dense and loaded.attribute_names == []
    assert_identical(model, loaded)


def test_parquet_columns_are_plain_and_documented(tmp_path):
    model = make_model(2)
    save_model(model, tmp_path / "m")
    table = pq.read_table(tmp_path / "m" / BLOCKS_FILE)
    assert table.column_names[:5] == list(storage.STRUCTURE_COLUMNS)
    assert table.column_names[5:] == model.attribute_names
    assert table.schema.field("_parent").type == pa.int32()
    assert table.schema.field("_level").type == pa.uint8()
    assert table.schema.field("rock").type == pa.int8()  # codes; dictionary is in model.json
    assert table.schema.field("density").type == pa.float32()
    assert all(table.column(name).null_count == 0 for name in table.column_names)
    raw = json.loads((tmp_path / "m" / HEADER_FILE).read_text())
    assert raw["format"] == "bakumine.blockmodel" and raw["version"] == 1
    rock = next(a for a in raw["attributes"] if a["name"] == "rock")
    assert rock["categories"] == ["WASTE", "ORE", "UNUSED"] and rock["dtype"] == "int8"
    assert raw["n_blocks"] == model.n_blocks
    assert set(raw["checksum"]["columns"]) == set(table.column_names)


def test_column_subset_on_load(tmp_path):
    model = make_model(3)
    save_model(model, tmp_path / "m")
    loaded = load_model(tmp_path / "m", attributes=["rock", "au"])
    assert_identical(model, loaded, names=["au", "rock"])  # stored order is kept
    assert load_model(tmp_path / "m", attributes=[]).attribute_names == []
    with pytest.raises(StorageError, match="no attribute.*'nope'"):
        load_model(tmp_path / "m", attributes=["au", "nope"])


def test_read_header_does_not_read_blocks(tmp_path, monkeypatch):
    model = make_model(4)
    save_model(model, tmp_path / "m")

    def forbidden(*args, **kwargs):
        raise AssertionError("blocks were read")

    monkeypatch.setattr(storage.pq, "read_table", forbidden)
    (tmp_path / "m" / BLOCKS_FILE).unlink()  # and it does not even need the file
    header = read_header(tmp_path / "m")
    assert header.definition == model.definition
    assert header.n_blocks == model.n_blocks
    assert header.attribute_names == model.attribute_names
    assert header.attributes[4].categories == ["WASTE", "ORE", "UNUSED"]
    assert header.attributes[1].background == pytest.approx(2.7)
    assert len(header.checksum) == 64


def test_checksum_detects_changed_values(tmp_path):
    model = make_model(5)
    save_model(model, tmp_path / "m")
    blocks = tmp_path / "m" / BLOCKS_FILE
    table = pq.read_table(blocks)
    values = table.column("au").to_numpy().copy()
    values[10] = np.nextafter(values[10], np.inf)  # one bit of one value
    changed = table.set_column(table.column_names.index("au"), "au", pa.array(values))
    pq.write_table(changed, blocks)
    with pytest.raises(StorageError, match="checksum mismatch in column.*au"):
        load_model(tmp_path / "m")
    # other columns still verify; and verification can be switched off
    assert load_model(tmp_path / "m", attributes=["rock"]).attribute_names == ["rock"]
    assert load_model(tmp_path / "m", verify=False)["au"][10] == values[10]


def test_corrupt_files_are_reported(tmp_path):
    model = make_model(6)
    save_model(model, tmp_path / "m")
    blocks = tmp_path / "m" / BLOCKS_FILE
    data = bytearray(blocks.read_bytes())
    original = bytes(data)
    middle = len(data) // 3
    data[middle : middle + 64] = bytes(64)
    blocks.write_bytes(data)
    with pytest.raises(StorageError):
        load_model(tmp_path / "m")
    blocks.write_bytes(original[: len(original) // 2])  # truncated
    with pytest.raises(StorageError, match="cannot be read"):
        load_model(tmp_path / "m")
    blocks.write_bytes(original)
    load_model(tmp_path / "m")

    header = tmp_path / "m" / HEADER_FILE
    text = header.read_text()
    header.write_text(text.replace('"WASTE"', '"WASTF"'))  # a category renamed by hand
    with pytest.raises(StorageError, match="header does not match its checksum"):
        load_model(tmp_path / "m")
    header.write_text(text[: len(text) // 2])
    with pytest.raises(StorageError, match="not valid JSON"):
        read_header(tmp_path / "m")
    header.write_text(json.dumps({"format": "something else"}))
    with pytest.raises(StorageError, match="not a bakumine.blockmodel header"):
        read_header(tmp_path / "m")
    with pytest.raises(StorageError, match="not a block model folder"):
        read_header(tmp_path / "missing")


def test_newer_version_is_refused(tmp_path):
    save_model(make_model(7), tmp_path / "m")
    header = tmp_path / "m" / HEADER_FILE
    raw = json.loads(header.read_text())
    raw["version"] = 99
    header.write_text(json.dumps(raw))
    with pytest.raises(StorageError, match="newer"):
        read_header(tmp_path / "m")


def hidden(folder: Path) -> list[str]:
    return sorted(p.name for p in folder.iterdir() if p.name.startswith("."))


def test_failure_mid_write_leaves_the_old_model_intact(tmp_path, monkeypatch):
    old = make_model(8)
    target = tmp_path / "m"
    save_model(old, target)
    before = {p.name: p.read_bytes() for p in target.iterdir()}
    new = make_model(9)
    real = storage._write_blocks

    def crash(columns, path, compression):
        real(columns, path, compression)
        with open(path, "r+b") as handle:  # a half-written file ...
            handle.truncate(os.path.getsize(path) // 2)
        raise OSError("disk full")  # ... and then the failure

    monkeypatch.setattr(storage, "_write_blocks", crash)
    with pytest.raises(OSError, match="disk full"):
        save_model(new, target)
    assert {p.name: p.read_bytes() for p in target.iterdir()} == before
    assert_identical(old, load_model(target))
    assert hidden(tmp_path) == []  # no temporary folder left behind

    # failure in the final rename: the old model is put back
    monkeypatch.setattr(storage, "_write_blocks", real)
    real_rename = os.rename

    def rename(src, dst):
        if ".tmp-" in str(src):
            raise OSError("rename failed")
        real_rename(src, dst)

    monkeypatch.setattr(storage.os, "rename", rename)
    with pytest.raises(OSError, match="rename failed"):
        save_model(new, target)
    monkeypatch.setattr(storage.os, "rename", real_rename)
    assert_identical(old, load_model(target))
    assert hidden(tmp_path) == []

    # a failed first save leaves nothing at all
    monkeypatch.setattr(storage, "_write_blocks", crash)
    with pytest.raises(OSError):
        save_model(new, tmp_path / "fresh")
    assert not (tmp_path / "fresh").exists() and hidden(tmp_path) == []

    # and a successful save replaces the model completely
    monkeypatch.setattr(storage, "_write_blocks", real)
    save_model(new, target)
    assert_identical(new, load_model(target))
    assert hidden(tmp_path) == []


def test_crash_between_the_two_renames_is_recoverable(tmp_path):
    old = make_model(10)
    target = tmp_path / "m"
    save_model(old, target)
    os.rename(target, tmp_path / ".m.old-deadbeef")  # the state after such a crash
    with pytest.raises(StorageError, match="interrupted save left the previous model"):
        load_model(target)
    new = make_model(11)
    with pytest.raises(FileExistsError):  # the next save restores the old model first
        save_model(new, target, overwrite=False)
    assert_identical(old, load_model(target))


def test_overwrite_rules(tmp_path):
    model = make_model(12)
    save_model(model, tmp_path / "m")
    with pytest.raises(FileExistsError):
        save_model(model, tmp_path / "m", overwrite=False)
    other = tmp_path / "documents"
    other.mkdir()
    (other / "thesis.txt").write_text("precious")
    with pytest.raises(StorageError, match="not a block model folder"):
        save_model(model, other)
    assert (other / "thesis.txt").read_text() == "precious"
    (tmp_path / "file").write_text("x")
    with pytest.raises(StorageError, match="not a folder"):
        save_model(model, tmp_path / "file")
    (tmp_path / "empty").mkdir()
    save_model(model, tmp_path / "empty")  # an empty folder may be used
    assert_identical(model, load_model(tmp_path / "empty"))
    with pytest.raises(ValueError, match="compression"):
        save_model(model, tmp_path / "m", compression="lzma")


def test_save_and_load_speed_at_reduced_size(tmp_path):
    """Plan criterion: 5 M blocks save < 10 s. Here 0.56 M blocks, 6 attributes, bound 5 s.

    Measured at full size (5.64 M blocks, 4 float64 + int32 + category, zstd): save 0.84 s,
    load with verification 0.46 s, 123 MB on disk, on the development machine.
    """
    rng = np.random.default_rng(0)
    definition = BlockModelDefinition((0.0, 0.0, 0.0), (10.0, 10.0, 5.0), (70, 70, 46),
                                      max_depth=(2, 2, 1))  # fmt: skip
    model = BlockModel(definition)
    model.split(rng.random(model.n_blocks) < 0.08)
    model.split((model.level == 1) & (rng.random(model.n_blocks) < 0.5))
    n = model.n_blocks
    assert n > 500_000
    for name in ("au", "ag", "density", "kv"):
        model.add_attribute(name)
        model[name][:] = rng.random(n)
    model["au"][rng.random(n) < 0.3] = np.nan
    model.add_attribute("pass_no", "int")
    model["pass_no"][:] = rng.integers(0, 4, n)
    model.add_attribute("rock", "category", categories=["ORE", "AIR", "WASTE"])
    model["rock"][:] = rng.integers(-1, 3, n)
    start = time.perf_counter()
    save_model(model, tmp_path / "m")
    saved = time.perf_counter() - start
    start = time.perf_counter()
    loaded = load_model(tmp_path / "m")
    took = time.perf_counter() - start
    assert_identical(model, loaded)
    assert saved < 5.0 and took < 5.0, (saved, took)
