"""BKMA: the binary container used by every bulk-data endpoint.

One response carries several named arrays of different lengths (a drillhole payload has
per-hole, per-point and per-interval arrays), which is why this is a small container of
raw arrays and not an Arrow IPC stream (one schema, equal-length columns; several tables
would need several requests or nested lists). The browser reads it with zero copies and
zero dependencies: every array is a typed-array view on the response buffer.

Layout (all integers little-endian)::

    offset  size  content
    0       4     magic  b"BKMA"
    4       4     uint32 format version (1)
    8       4     uint32 H = length of the header in bytes
    12      H     header: UTF-8 JSON, padded with spaces so that 12 + H is a multiple of 8
    12 + H  ...   array data; every array starts at a multiple of 8 bytes

    header = {
      "kind":   str,                  # what the payload is ("drillholes", "mesh", ...)
      "meta":   {...},                # small non-bulk data: names, categories, counts
      "arrays": [{"name": str, "dtype": str, "shape": [int, ...],
                  "offset": int,      # from the start of the data section
                  "nbytes": int}, ...]
    }

``dtype`` is one of ``f8 f4 i4 u4 i2 u2 i1 u1`` (little-endian, C order). There is no
64-bit integer type: JavaScript typed arrays of int64 are BigInt arrays. Bulk numbers
never go in the JSON header.

``web/src/api/bkma.ts`` is the one decoder on the web side.
"""

from __future__ import annotations

import json
import struct
from collections.abc import Mapping
from typing import Any

import numpy as np

MAGIC = b"BKMA"
VERSION = 1
MEDIA_TYPE = "application/vnd.bakumine.arrays"
DTYPES = ("f8", "f4", "i4", "u4", "i2", "u2", "i1", "u1")
_ALIGN = 8


def _pad(n: int) -> int:
    return (-n) % _ALIGN


def pack_arrays(kind: str, meta: Mapping[str, Any], arrays: Mapping[str, np.ndarray]) -> bytes:
    """Serialise named arrays and a small JSON ``meta`` object to one BKMA payload."""
    entries = []
    chunks: list[bytes] = []
    offset = 0
    for name, array in arrays.items():
        a = np.asarray(array)
        code = a.dtype.kind + str(a.dtype.itemsize)
        if code not in DTYPES:
            raise TypeError(f"array {name!r}: dtype {a.dtype} is not one of {DTYPES}")
        data = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder("<")).tobytes()
        entries.append(
            {
                "name": name,
                "dtype": code,
                "shape": list(a.shape),
                "offset": offset,
                "nbytes": len(data),
            }
        )
        chunks.append(data)
        chunks.append(b"\0" * _pad(len(data)))
        offset += len(data) + _pad(len(data))
    header = json.dumps(
        {"kind": kind, "meta": dict(meta), "arrays": entries},
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    header += b" " * _pad(12 + len(header))
    return b"".join([MAGIC, struct.pack("<II", VERSION, len(header)), header, *chunks])


def unpack_arrays(payload: bytes) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Inverse of :func:`pack_arrays`: ``(header, {name: array})``. Arrays are read-only."""
    if len(payload) < 12 or payload[:4] != MAGIC:
        raise ValueError("not a BKMA payload")
    version, header_len = struct.unpack_from("<II", payload, 4)
    if version != VERSION:
        raise ValueError(f"unsupported BKMA version {version}")
    data_start = 12 + header_len
    if data_start > len(payload):
        raise ValueError("BKMA header is truncated")
    header = json.loads(payload[12:data_start].decode("utf-8"))
    arrays: dict[str, np.ndarray] = {}
    for entry in header["arrays"]:
        if entry["dtype"] not in DTYPES:
            raise ValueError(f"unknown dtype {entry['dtype']!r}")
        dtype = np.dtype("<" + entry["dtype"])
        count = int(np.prod(entry["shape"], dtype=np.int64))
        start = data_start + entry["offset"]
        if count * dtype.itemsize != entry["nbytes"] or start + entry["nbytes"] > len(payload):
            raise ValueError(f"array {entry['name']!r} does not fit the payload")
        arrays[entry["name"]] = np.frombuffer(
            payload, dtype=dtype, count=count, offset=start
        ).reshape(entry["shape"])
    return header, arrays
