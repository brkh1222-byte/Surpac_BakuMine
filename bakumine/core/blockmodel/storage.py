"""Block model persistence: a folder with ``model.json`` and ``blocks.parquet``.

On-disk format (version 1)
--------------------------
``model.json`` -- the header, small, human-readable::

    {
      "format": "bakumine.blockmodel",
      "version": 1,
      "bakumine_version": "...",
      "definition": {...},                 BlockModelDefinition.to_dict()
      "n_blocks": 123,                     rows of the block table
      "n_parents": 100,
      "attributes": [{...}, ...],          Attribute.to_dict(), in creation order;
                                           a category attribute carries its dictionary
                                           ("categories", in code order) and every
                                           attribute its background value
      "blocks": {"file": "blocks.parquet", "compression": "zstd",
                 "structure_columns": ["_parent", "_level", "_sub_i", "_sub_j", "_sub_k"],
                 "bytes": 4567},
      "checksum": {"algorithm": "sha256",
                   "columns": {"_parent": "...", ..., "au": "..."},
                   "content": "..."}
    }

``blocks.parquet`` -- one row per leaf block, **in table order** (parent, Morton
code; see :mod:`bakumine.core.blockmodel.blocks`):

* the structure columns ``_parent`` (int32), ``_level`` (uint8) and ``_sub_i``,
  ``_sub_j``, ``_sub_k`` (uint8; the three columns of ``sub``). Attribute names
  cannot start with an underscore, so these never collide with an attribute;
* one column per attribute, named as the attribute, in the attribute's own
  dtype. **A category attribute is stored as its integer codes** (-1 =
  missing); the dictionary is in ``model.json``. This was chosen over Arrow
  dictionary columns because the codes *are* the model's column: unused
  categories, the order of the dictionary and the code dtype survive without
  any translation, and a reader that only wants numbers needs no dictionary.

No Parquet nulls are written: NaN is an ordinary float value and missing
categories are code -1. Float columns are written without dictionary encoding
(byte-stream-split + the codec), so every bit pattern, including NaN payloads
and the sign of zero, is preserved; integer and code columns use dictionary
encoding. Geometry (centroids, sizes) is not stored: it follows exactly from
the structure columns and the definition.

Compression: ``zstd`` (level 1) by default; ``snappy`` and ``none`` can be
chosen. Measured on a 5.64 M block model with sub-blocks, 4 float64, 1 int32
and 1 category attribute, 271 MB in memory (Apple M-series, pyarrow 26;
save / load with verification / file size)::

    zstd    0.84 s   0.46 s   123 MB      <- default: smallest, speed comparable
    snappy  0.56 s   0.39 s   142 MB
    none    0.51 s   0.39 s   205 MB

Checksum
--------
``checksum.columns`` holds a SHA-256 per column over the dtype string and the
raw little-endian bytes of the column *as held in memory*, not over the file:
it is independent of compression and encoding, and it verifies the decoded
values, so it also catches a decoding fault. ``checksum.content`` is the
SHA-256 over the definition JSON, the attribute specifications and the column
digests, i.e. over everything that defines the model. :func:`load_model`
verifies the columns it loads (all of them by default) and the content digest.

Atomic write
------------
:func:`save_model` writes both files into a hidden temporary folder beside the
target (``.<name>.tmp-<id>``), flushes them, and only then renames the folder
into place. A failure while writing leaves the existing model untouched and
removes the temporary folder. Replacing an existing model takes two renames
(old -> ``.<name>.old-<id>``, new -> target) because a directory cannot be
renamed onto a non-empty directory; a crash exactly between the two leaves no
folder at the target but the complete previous model in the ``.old`` folder
(:func:`load_model` says so, and the next :func:`save_model` restores it
first). At no point is a half-written model visible at the target path.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from bakumine import __version__
from bakumine.core.blockmodel.attributes import Attribute
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import BlockModelDefinition

__all__ = [
    "BLOCKS_FILE",
    "FORMAT",
    "HEADER_FILE",
    "STRUCTURE_COLUMNS",
    "VERSION",
    "ModelHeader",
    "StorageError",
    "is_model_folder",
    "load_model",
    "read_header",
    "save_model",
]

FORMAT = "bakumine.blockmodel"
VERSION = 1
HEADER_FILE = "model.json"
BLOCKS_FILE = "blocks.parquet"
STRUCTURE_COLUMNS = ("_parent", "_level", "_sub_i", "_sub_j", "_sub_k")
COMPRESSIONS = ("zstd", "snappy", "none")

_ROW_GROUP = 1 << 20


class StorageError(ValueError):
    """A model folder is missing, damaged, or of an unsupported version."""


@dataclass(frozen=True)
class ModelHeader:
    """Contents of ``model.json`` (see the module docstring); no block data.

    Attributes
    ----------
    path : the model folder
    version : format version
    definition : BlockModelDefinition
    attributes : tuple of :class:`Attribute` (with categories and backgrounds)
    n_blocks : number of rows of the block table
    compression : codec of ``blocks.parquet``
    blocks_bytes : size of ``blocks.parquet`` when it was written
    checksum : the content digest (hex)
    raw : the parsed JSON
    """

    path: Path
    version: int
    definition: BlockModelDefinition
    attributes: tuple[Attribute, ...]
    n_blocks: int
    compression: str
    blocks_bytes: int
    checksum: str
    raw: dict[str, Any]

    @property
    def attribute_names(self) -> list[str]:
        return [a.name for a in self.attributes]


# ------------------------------------------------------------------ checksums


def _digest(column: np.ndarray) -> str:
    column = np.ascontiguousarray(column)
    if column.dtype.byteorder == ">":  # pragma: no cover - never produced by BlockModel
        column = column.astype(column.dtype.newbyteorder("<"))
    h = hashlib.sha256()
    h.update(column.dtype.str.encode())
    h.update(memoryview(column).cast("B"))
    return h.hexdigest()


def _digests(columns: dict[str, np.ndarray]) -> dict[str, str]:
    """SHA-256 per column; hashing releases the GIL, so columns run in parallel."""
    with ThreadPoolExecutor(max_workers=min(8, max(1, len(columns)))) as pool:
        return dict(zip(columns, pool.map(_digest, columns.values()), strict=True))


def _content_digest(definition: dict, attributes: list[dict], columns: dict[str, str]) -> str:
    text = json.dumps(
        {"definition": definition, "attributes": attributes, "columns": columns}, sort_keys=True
    )
    return hashlib.sha256(text.encode()).hexdigest()


# ----------------------------------------------------------------------- save


def _stored_columns(model: BlockModel) -> dict[str, np.ndarray]:
    sub = model.sub
    columns: dict[str, np.ndarray] = {
        "_parent": model.parent,
        "_level": model.level,
        "_sub_i": np.ascontiguousarray(sub[:, 0]),
        "_sub_j": np.ascontiguousarray(sub[:, 1]),
        "_sub_k": np.ascontiguousarray(sub[:, 2]),
    }
    for name in model.attribute_names:
        columns[name] = np.ascontiguousarray(model[name])
    return columns


def _write_blocks(columns: dict[str, np.ndarray], path: Path, compression: str) -> None:
    table = pa.table({name: pa.array(col) for name, col in columns.items()})
    floats = [name for name, col in columns.items() if col.dtype.kind == "f"]
    others = [name for name in columns if name not in floats]
    pq.write_table(
        table,
        path,
        compression=compression,
        compression_level=1 if compression == "zstd" else None,
        use_dictionary=others,
        column_encoding={name: "BYTE_STREAM_SPLIT" for name in floats} or None,
        write_statistics=False,
        row_group_size=_ROW_GROUP,
    )


def _fsync(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def is_model_folder(path: str | Path) -> bool:
    """True when ``path`` is a folder containing ``model.json``."""
    return (Path(path) / HEADER_FILE).is_file()


def _stale(path: Path, kind: str) -> list[Path]:
    return sorted(path.parent.glob(f".{path.name}.{kind}-*"))


def save_model(
    model: BlockModel,
    path: str | Path,
    *,
    compression: str = "zstd",
    overwrite: bool = True,
) -> ModelHeader:
    """Save a block model as the folder ``path`` (atomically; see the module docstring).

    Parameters
    ----------
    compression : ``"zstd"`` (default), ``"snappy"`` or ``"none"``.
    overwrite : replace an existing model folder (default). An existing folder
        that is not a model folder and is not empty is never replaced.

    Returns the header that was written.
    """
    if compression not in COMPRESSIONS:
        raise ValueError(f"compression must be one of {COMPRESSIONS}, got {compression!r}")
    path = Path(path)
    parent_dir = path.parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        old = _stale(path, "old")
        if old:  # a previous save was interrupted between its two renames
            os.rename(old[-1], path)
    if path.exists():
        if not path.is_dir():
            raise StorageError(f"{path} exists and is not a folder")
        if not is_model_folder(path) and any(path.iterdir()):
            raise StorageError(f"{path} exists and is not a block model folder; not replaced")
        if not overwrite and is_model_folder(path):
            raise FileExistsError(f"{path} already holds a block model (overwrite=False)")

    columns = _stored_columns(model)
    definition = model.definition.to_dict()
    attributes = [model.attribute(name).to_dict() for name in model.attribute_names]
    digests = _digests(columns)
    token = uuid.uuid4().hex[:8]
    tmp = parent_dir / f".{path.name}.tmp-{token}"
    tmp.mkdir()
    try:
        blocks_path = tmp / BLOCKS_FILE
        _write_blocks(columns, blocks_path, compression)
        _fsync(blocks_path)
        header = {
            "format": FORMAT,
            "version": VERSION,
            "bakumine_version": __version__,
            "definition": definition,
            "n_blocks": model.n_blocks,
            "n_parents": model.definition.n_parents,
            "attributes": attributes,
            "blocks": {
                "file": BLOCKS_FILE,
                "compression": compression,
                "structure_columns": list(STRUCTURE_COLUMNS),
                "bytes": blocks_path.stat().st_size,
            },
            "checksum": {
                "algorithm": "sha256",
                "columns": digests,
                "content": _content_digest(definition, attributes, digests),
            },
        }
        header_path = tmp / HEADER_FILE
        header_path.write_text(json.dumps(header, indent=1) + "\n", encoding="utf-8")
        _fsync(header_path)
        if path.exists():
            backup = parent_dir / f".{path.name}.old-{token}"
            os.rename(path, backup)
            try:
                os.rename(tmp, path)
            except BaseException:
                os.rename(backup, path)
                raise
            shutil.rmtree(backup, ignore_errors=True)
        else:
            os.rename(tmp, path)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return _parse_header(path, header)


# ----------------------------------------------------------------------- load


def _parse_header(path: Path, raw: dict[str, Any]) -> ModelHeader:
    try:
        if raw.get("format") != FORMAT:
            raise StorageError(f"{path / HEADER_FILE}: not a {FORMAT} header")
        version = int(raw["version"])
        if version > VERSION:
            raise StorageError(
                f"{path}: format version {version} is newer than this program reads ({VERSION})"
            )
        return ModelHeader(
            path=path,
            version=version,
            definition=BlockModelDefinition.from_dict(raw["definition"]),
            attributes=tuple(Attribute.from_dict(a) for a in raw["attributes"]),
            n_blocks=int(raw["n_blocks"]),
            compression=str(raw["blocks"]["compression"]),
            blocks_bytes=int(raw["blocks"]["bytes"]),
            checksum=str(raw["checksum"]["content"]),
            raw=raw,
        )
    except (KeyError, TypeError) as error:
        raise StorageError(f"{path / HEADER_FILE}: incomplete header ({error!r})") from None


def read_header(path: str | Path) -> ModelHeader:
    """Read ``model.json`` only: definition, attributes, block count, checksum.

    Does not open ``blocks.parquet``.
    """
    path = Path(path)
    file = path / HEADER_FILE
    if not file.is_file():
        old = _stale(path, "old") if not path.exists() else []
        if old:
            raise StorageError(
                f"{path} is missing, but an interrupted save left the previous model at "
                f"{old[-1]}; rename it back to {path.name}"
            )
        raise StorageError(f"{path} is not a block model folder (no {HEADER_FILE})")
    try:
        raw = json.loads(file.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise StorageError(f"{file}: not valid JSON ({error})") from None
    if not isinstance(raw, dict):
        raise StorageError(f"{file}: not a {FORMAT} header")
    return _parse_header(path, raw)


def load_model(
    path: str | Path,
    *,
    attributes: Iterable[str] | None = None,
    verify: bool = True,
) -> BlockModel:
    """Load a block model saved by :func:`save_model`.

    Parameters
    ----------
    attributes : names of the attributes to load; default all. The structure
        columns are always loaded. The loaded attributes keep their stored
        order. Unknown names raise :class:`StorageError`.
    verify : check the SHA-256 of every loaded column and the content digest
        (default). A mismatch raises :class:`StorageError`.

    The result is identical to the saved model: same block order and ids,
    dtypes, values bit for bit (NaN payloads included), categories in the
    same order, backgrounds and metadata.
    """
    header = read_header(path)
    path = Path(path)
    specs = list(header.attributes)
    if attributes is not None:
        wanted = list(attributes)
        unknown = [n for n in wanted if n not in header.attribute_names]
        if unknown:
            raise StorageError(
                f"{path}: no attribute(s) {', '.join(map(repr, unknown))}; "
                f"stored: {', '.join(header.attribute_names) or '(none)'}"
            )
        specs = [s for s in specs if s.name in set(wanted)]
    names = [*STRUCTURE_COLUMNS, *(s.name for s in specs)]
    blocks_path = path / header.raw["blocks"].get("file", BLOCKS_FILE)
    try:
        table = pq.read_table(blocks_path, columns=names, use_threads=True)
    except (pa.ArrowException, OSError, KeyError) as error:
        raise StorageError(f"{blocks_path}: cannot be read ({error})") from None
    if table.num_rows != header.n_blocks:
        raise StorageError(
            f"{blocks_path}: {table.num_rows} rows, but the header says {header.n_blocks}"
        )
    columns: dict[str, np.ndarray] = {}
    for name in names:
        chunked = table.column(name)
        if chunked.null_count:
            raise StorageError(f"{blocks_path}: column {name!r} contains nulls")
        columns[name] = chunked.to_numpy()
    del table
    expected = {"_parent": np.int32, "_level": np.uint8, "_sub_i": np.uint8,
                "_sub_j": np.uint8, "_sub_k": np.uint8}  # fmt: skip
    expected.update({s.name: s.dtype for s in specs})
    for name, dtype in expected.items():
        if columns[name].dtype != dtype:
            raise StorageError(
                f"{blocks_path}: column {name!r} is {columns[name].dtype}, expected {dtype}"
            )
    if verify:
        stored = header.raw["checksum"]
        digests = _digests(columns)
        bad = [name for name in names if digests[name] != stored["columns"].get(name)]
        if bad:
            raise StorageError(
                f"{path}: checksum mismatch in column(s) {', '.join(bad)}; the model is damaged"
            )
        content = _content_digest(
            header.raw["definition"], header.raw["attributes"], stored["columns"]
        )
        if content != stored["content"]:
            raise StorageError(f"{path}: the header does not match its checksum; it was altered")
    sub = np.stack([columns["_sub_i"], columns["_sub_j"], columns["_sub_k"]], axis=1)
    try:
        model = BlockModel.from_arrays(
            header.definition, columns["_parent"], columns["_level"], sub
        )
    except ValueError as error:
        raise StorageError(f"{path}: invalid block structure ({error})") from None
    for spec in specs:
        model.add_attribute(
            spec.name,
            spec.kind,
            background=spec.background,
            dtype=spec.dtype,
            description=spec.description,
            unit=spec.unit,
            decimals=spec.decimals,
            categories=spec.categories,
        )
        model[spec.name][...] = columns.pop(spec.name)
    return model
