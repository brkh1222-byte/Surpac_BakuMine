"""Provenance record of a geology result (research.md 5.7.7, geology_contracts.md 3).

Every result says where it comes from in three lists:

- ``measured``: real site data that was used, and what for;
- ``assumed``: what the software chose on its own (defaults, inferred faults, stand-in
  densities, extrapolation beyond the drilling);
- ``missing``: an input that is still needed; ``source`` says where it would come from.

:class:`Provenance` collects the entries, writes ``provenance.json`` and gives the three
tables (:meth:`Provenance.tables`, :meth:`Provenance.format_text`).
"""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from bakumine import __version__

FORMAT = "bakumine.provenance"
VERSION = 1

PROVENANCE_KINDS = ("measured", "assumed", "missing")
PROVENANCE_STAGES = (
    "units",
    "picks",
    "surfaces",
    "lithology",
    "faults",
    "coding",
    "density",
    "topography",
    "extent",
)
TABLE_COLUMNS = ("item", "source", "used_for", "detail", "stage")
TABLE_TITLES = {
    "measured": "Measured: real site data used",
    "assumed": "Assumed: chosen by the software",
    "missing": "Missing: inputs still needed",
}


class ProvenanceError(ValueError):
    """An entry or a provenance file that breaks the rules of the record."""


@dataclass(frozen=True)
class ProvenanceEntry:
    """One line of the record. ``item``, ``kind``, ``source`` and ``used_for`` are required."""

    item: str
    kind: str
    source: str
    used_for: str
    detail: str = ""
    stage: str = ""
    task: str = ""
    count: int | None = None
    value: float | str | None = None

    def __post_init__(self) -> None:
        label = f"provenance entry {self.item!r}"
        for name in ("item", "kind", "source", "used_for"):
            text = getattr(self, name)
            if not isinstance(text, str) or not text.strip():
                raise ProvenanceError(f"{label}: '{name}' is required and must be text")
        if self.kind not in PROVENANCE_KINDS:
            raise ProvenanceError(
                f"{label}: kind {self.kind!r} is not one of {', '.join(PROVENANCE_KINDS)}"
            )
        for name in ("detail", "stage", "task"):
            if not isinstance(getattr(self, name), str):
                raise ProvenanceError(f"{label}: '{name}' must be text")
        if self.stage and self.stage not in PROVENANCE_STAGES:
            raise ProvenanceError(
                f"{label}: stage {self.stage!r} is not one of {', '.join(PROVENANCE_STAGES)}"
            )
        if self.count is not None:
            if isinstance(self.count, bool) or not isinstance(self.count, numbers.Integral):
                raise ProvenanceError(f"{label}: 'count' must be a whole number")
            object.__setattr__(self, "count", int(self.count))
        if self.value is not None and not isinstance(self.value, str):
            if isinstance(self.value, bool) or not isinstance(self.value, numbers.Real):
                raise ProvenanceError(f"{label}: 'value' must be a number or text")
            if not math.isfinite(self.value):
                raise ProvenanceError(f"{label}: 'value' must be finite")
            whole = isinstance(self.value, numbers.Integral)
            object.__setattr__(self, "value", int(self.value) if whole else float(self.value))

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "item": self.item,
            "kind": self.kind,
            "source": self.source,
            "used_for": self.used_for,
            "detail": self.detail,
        }
        if self.stage:
            out["stage"] = self.stage
        if self.task:
            out["task"] = self.task
        if self.count is not None:
            out["count"] = self.count
        if self.value is not None:
            out["value"] = self.value
        return out

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> ProvenanceEntry:
        if not isinstance(data, Mapping):
            raise ProvenanceError("a provenance entry must be an object")
        known = {f.name for f in fields(cls)}
        unknown = sorted(set(data) - known)
        if unknown:
            raise ProvenanceError(
                f"provenance entry {data.get('item')!r}: unknown field(s) {', '.join(unknown)}"
            )
        missing = [n for n in ("item", "kind", "source", "used_for") if n not in data]
        if missing:
            raise ProvenanceError(
                f"provenance entry {data.get('item')!r}: missing field(s) {', '.join(missing)}"
            )
        return cls(**{k: v for k, v in data.items() if v is not None})


def _now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class Provenance:
    """The record of one object: an ordered list of :class:`ProvenanceEntry`."""

    def __init__(self, *, object_kind: str = "geology", name: str = "") -> None:
        self.object_kind = object_kind
        self.name = name
        self.created = _now()
        self.bakumine_version = __version__
        self._entries: list[ProvenanceEntry] = []

    @property
    def entries(self) -> tuple[ProvenanceEntry, ...]:
        return tuple(self._entries)

    def add(self, entry: ProvenanceEntry) -> None:
        if not isinstance(entry, ProvenanceEntry):
            raise ProvenanceError(f"not a ProvenanceEntry: {entry!r}")
        self._entries.append(entry)

    def measured(self, item: str, *, source: str, used_for: str, **extra: Any) -> None:
        """Real site data that was used."""
        self.add(ProvenanceEntry(item, "measured", source, used_for, **extra))

    def assumed(self, item: str, *, source: str, used_for: str, **extra: Any) -> None:
        """Something the software chose on its own."""
        self.add(ProvenanceEntry(item, "assumed", source, used_for, **extra))

    def missing(self, item: str, *, source: str, used_for: str, **extra: Any) -> None:
        """An input still needed; ``source`` says where it would come from."""
        self.add(ProvenanceEntry(item, "missing", source, used_for, **extra))

    def extend(self, entries: Iterable[ProvenanceEntry]) -> None:
        for entry in entries:
            self.add(entry)

    # ------------------------------------------------------------------ file

    def to_dict(self) -> dict[str, Any]:
        return {
            "format": FORMAT,
            "version": VERSION,
            "object": {
                "kind": self.object_kind,
                "name": self.name,
                "created": self.created,
                "bakumine_version": self.bakumine_version,
            },
            "entries": [entry.to_dict() for entry in self._entries],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Provenance:
        if not isinstance(data, Mapping) or data.get("format") != FORMAT:
            raise ProvenanceError(f"not a provenance record (format is not {FORMAT!r})")
        if data.get("version") != VERSION:
            raise ProvenanceError(
                f"provenance record version {data.get('version')!r} is not supported "
                f"(this build reads version {VERSION})"
            )
        about = data.get("object") or {}
        if not isinstance(about, Mapping):
            raise ProvenanceError("provenance record: 'object' must be an object")
        entries = data.get("entries", [])
        if not isinstance(entries, list):
            raise ProvenanceError("provenance record: 'entries' must be a list")
        record = cls(object_kind=str(about.get("kind", "geology")), name=str(about.get("name", "")))
        record.created = str(about.get("created", record.created))
        record.bakumine_version = str(about.get("bakumine_version", record.bakumine_version))
        record.extend(ProvenanceEntry.from_dict(entry) for entry in entries)
        return record

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        text = json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        target.write_text(text + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> Provenance:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    # ------------------------------------------------------------------ tables

    def tables(self) -> dict[str, pd.DataFrame]:
        """``measured`` / ``assumed`` / ``missing``, each in the order the entries were added."""
        out: dict[str, pd.DataFrame] = {}
        for kind in PROVENANCE_KINDS:
            rows = [
                [getattr(entry, column) for column in TABLE_COLUMNS]
                for entry in self._entries
                if entry.kind == kind
            ]
            out[kind] = pd.DataFrame(rows, columns=list(TABLE_COLUMNS), dtype=object)
        return out

    def format_text(self) -> str:
        """The three tables as plain text, each under its title with its number of rows."""
        blocks: list[str] = []
        for kind, table in self.tables().items():
            lines = [f"{TABLE_TITLES[kind]} ({len(table)})"]
            if table.empty:
                lines.append("  (none)")
            else:
                columns = [c for c in TABLE_COLUMNS if c == "item" or (table[c] != "").any()]
                rows = [columns, *table[columns].astype(str).to_numpy().tolist()]
                widths = [max(len(row[i]) for row in rows) for i in range(len(columns))]
                for n, row in enumerate(rows):
                    cells = (cell.ljust(width) for cell, width in zip(row, widths, strict=True))
                    lines.append("  " + "  ".join(cells).rstrip())
                    if n == 0:
                        lines.append("  " + "  ".join("-" * width for width in widths))
            blocks.append("\n".join(lines))
        return "\n\n".join(blocks)
