"""Shared helpers for the text-format readers: loading, binary detection, tokens."""

from __future__ import annotations

from pathlib import Path

__all__ = ["BinaryFileError", "FormatWarning", "looks_binary", "read_text", "parse_float"]


class BinaryFileError(ValueError):
    """The file is binary, but only the text variant of the format is supported."""


class FormatWarning(UserWarning):
    """A reader accepted a file that deviates from the documented layout."""


_TEXT_CONTROL = frozenset(b"\t\n\r\f\x1a")  # 0x1A: DOS end-of-file mark


def looks_binary(data: bytes, sample: int = 8192) -> bool:
    """True when the first ``sample`` bytes cannot be a text file.

    A NUL byte, or more than 5 % control characters other than tab / newline /
    carriage return / form feed, is taken as binary.
    """
    head = data[:sample]
    if not head:
        return False
    if b"\x00" in head:
        return True
    control = sum(1 for b in head if b < 32 and b not in _TEXT_CONTROL)
    return control > 0.05 * len(head)


def read_text(path, *, format_name: str) -> str:
    """Read a text file, refusing binary content with a clear message.

    Decoding is UTF-8 (a byte-order mark is dropped) with a fall-back to
    Windows-1252 / Latin-1, so legacy files with accented memo text still load.
    Line endings are left as they are; callers use ``str.splitlines``.
    """
    data = Path(path).read_bytes()
    if looks_binary(data):
        raise BinaryFileError(
            f"{path}: binary {format_name} is not supported; re-save as text "
            f"(first bytes: {data[:16].hex(' ')})"
        )
    try:
        return data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return data.decode("latin-1")


def parse_float(token: str) -> float:
    """``float`` that also accepts Fortran ``D`` exponents (``1.5D+03``)."""
    try:
        return float(token)
    except ValueError:
        return float(token.replace("D", "E").replace("d", "e"))
