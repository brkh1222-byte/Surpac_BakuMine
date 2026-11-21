"""Data model for string files: segments of 3D points with a string number.

A *string* (research.md section 5.1) is an ordered list of 3D points that
carries a string number. One contiguous run of points is a *segment*; a file
may hold several segments with the same string number.

Everything here is in INTERNAL coordinates: ``x = east, y = north, z = up``,
metres, float64. File formats that use another column order (Surpac writes
``Y, X, Z``) convert in their reader/writer and nowhere else.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field

import numpy as np

__all__ = ["AxisRecord", "StringFile", "StringHeader", "StringSegment", "Strings"]


def _normalise_descriptions(descriptions, n_points: int) -> tuple[tuple[str, ...], ...]:
    """One tuple of stripped strings per point, without trailing empty fields."""
    if descriptions is None:
        return ((),) * n_points
    rows = []
    for row in descriptions:
        if isinstance(row, str):
            row = (row,)
        fields = [str(d).strip() for d in row]
        while fields and fields[-1] == "":
            fields.pop()
        rows.append(tuple(fields))
    if len(rows) != n_points:
        raise ValueError(f"descriptions has {len(rows)} rows for {n_points} points")
    return tuple(rows)


@dataclass(eq=False)
class StringSegment:
    """One segment: a string number and its points.

    Parameters
    ----------
    number : int
        String number (>= 1; 0 is reserved for separators in Surpac files).
    points : (n, 3) array_like
        Point coordinates ``(east, north, up)``; stored read-only as float64.
    descriptions : sequence of sequences of str, optional
        Per-point description fields ``(d1, d2, ...)``. Fields are stripped of
        surrounding whitespace and trailing empty fields are dropped, because
        a text file cannot tell ``"a", ""`` from ``"a"``; empty fields in the
        middle (``"a", "", "c"``) are kept.
    """

    number: int
    points: np.ndarray
    descriptions: tuple[tuple[str, ...], ...] | None = None

    def __post_init__(self) -> None:
        self.number = int(self.number)
        p = np.array(self.points, dtype=np.float64, copy=True)
        if p.size == 0:
            p = p.reshape(0, 3)
        if p.ndim != 2 or p.shape[1] != 3:
            raise ValueError(f"points must have shape (n, 3), got {p.shape}")
        p.flags.writeable = False
        self.points = p
        self.descriptions = _normalise_descriptions(self.descriptions, len(p))

    @property
    def n_points(self) -> int:
        return len(self.points)

    def closed(self, tolerance: float = 1e-9) -> bool:
        """True when the last point repeats the first within ``tolerance`` (metres).

        Segments with fewer than three points are always open.
        """
        if self.n_points < 3:
            return False
        return bool(np.linalg.norm(self.points[-1] - self.points[0]) <= tolerance)

    @property
    def is_closed(self) -> bool:
        """:meth:`closed` with the default tolerance of 1e-9 m."""
        return self.closed()

    @property
    def has_descriptions(self) -> bool:
        return any(self.descriptions)

    def description_column(self, index: int) -> list[str]:
        """Field ``d<index + 1>`` of every point ('' where absent)."""
        return [row[index] if index < len(row) else "" for row in self.descriptions]

    def __eq__(self, other) -> bool:
        if not isinstance(other, StringSegment):
            return NotImplemented
        return (
            self.number == other.number
            and self.points.shape == other.points.shape
            and bool(np.array_equal(self.points, other.points))
            and self.descriptions == other.descriptions
        )

    def __repr__(self) -> str:
        kind = "closed" if self.is_closed else "open"
        return f"StringSegment(number={self.number}, n_points={self.n_points}, {kind})"


@dataclass
class StringHeader:
    """First line of a Surpac string file."""

    location: str = ""
    date: str = ""
    purpose: str = ""
    memo: str = ""


@dataclass(eq=False)
class AxisRecord:
    """Section axis of a string file: two points, internal ``(east, north, up)``."""

    start: np.ndarray = field(default_factory=lambda: np.zeros(3))
    end: np.ndarray = field(default_factory=lambda: np.zeros(3))

    def __post_init__(self) -> None:
        self.start = np.asarray(self.start, dtype=np.float64).reshape(3)
        self.end = np.asarray(self.end, dtype=np.float64).reshape(3)

    def __eq__(self, other) -> bool:
        if not isinstance(other, AxisRecord):
            return NotImplemented
        return bool(
            np.array_equal(self.start, other.start) and np.array_equal(self.end, other.end)
        )


@dataclass
class StringFile:
    """A set of string segments plus the header metadata of the file.

    ``warnings`` and ``skipped`` are filled by tolerant readers: human-readable
    notes about deviations from the layout, and ``(line_number, text)`` of
    every line the reader could not use.
    """

    segments: list[StringSegment] = field(default_factory=list)
    header: StringHeader = field(default_factory=StringHeader)
    axis: AxisRecord | None = None
    warnings: list[str] = field(default_factory=list, compare=False)
    skipped: list[tuple[int, str]] = field(default_factory=list, compare=False)
    #: filled by the Surpac reader: for every record after the header and the axis record,
    #: in file order, the row of :meth:`all_points` it holds, or -1 for a null (separator)
    #: record or an unreadable line. A ``.dtm`` numbers its vertices by these records.
    record_rows: list[int] = field(default_factory=list, compare=False, repr=False)

    @classmethod
    def from_arrays(
        cls, arrays: Iterable, numbers: int | Sequence[int] = 1, **kwargs
    ) -> StringFile:
        """Build from point arrays; ``numbers`` is one string number or one per array."""
        arrays = list(arrays)
        if isinstance(numbers, int):
            numbers = [numbers] * len(arrays)
        if len(numbers) != len(arrays):
            raise ValueError("numbers must have one entry per array")
        segments = [StringSegment(n, a) for n, a in zip(numbers, arrays, strict=True)]
        return cls(segments=segments, **kwargs)

    @property
    def n_segments(self) -> int:
        return len(self.segments)

    @property
    def n_points(self) -> int:
        return sum(s.n_points for s in self.segments)

    @property
    def string_numbers(self) -> list[int]:
        """Distinct string numbers, ascending."""
        return sorted({s.number for s in self.segments})

    def segments_of(self, number: int) -> list[StringSegment]:
        """Segments with the given string number, in file order."""
        return [s for s in self.segments if s.number == number]

    def all_points(self) -> np.ndarray:
        """(n, 3) points of every segment concatenated in order."""
        if not self.segments:
            return np.zeros((0, 3))
        return np.concatenate([s.points for s in self.segments])

    def __iter__(self):
        return iter(self.segments)

    def __len__(self) -> int:
        return len(self.segments)


Strings = StringFile
