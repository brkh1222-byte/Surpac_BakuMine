"""Block model CSV interchange (research.md sections 5.6 and 6).

One row per (sub-)block: centroid coordinates, block dimensions, attribute
values. This is the text route between this program and other packages; the
binary Surpac ``.mdl`` is never read (clean-room rule).

PROVISIONAL -- every assumption below is unverified **[verify]**
----------------------------------------------------------------
No real Surpac block model CSV export was available when this was written.
The default layout ``"surpac_like"`` is built from the one-line description in
research.md 5.6 ("centroid Y, X, Z, block sizes, attributes") and from general
recollection of the documentation. Assumed, and to be checked against a real
export:

1. **Column order**: centroid northing, easting, elevation (Y, X, Z), then the
   block dimensions along Y, X, Z, then the attributes. (Northing first is
   Surpac's coordinate order everywhere; that the export follows it is
   documented, the exact columns are not verified.)
2. **Column names**: ``ycentre, xcentre, zcentre, ysize, xsize, zsize``. The
   real header names are unknown. The reader therefore recognises many
   synonyms (see :data:`CENTROID_SYNONYMS`, :data:`SIZE_SYNONYMS`).
3. **Header lines**: the default writes two free-text lines and one line of
   column names. Whether Surpac writes none, one or several header lines is
   unknown; the reader skips any leading lines that are not data.
4. **Delimiter**: comma. The reader also accepts semicolon, tab and
   whitespace.
5. **Dimensions are the block's own size** (sub-block size for a sub-block),
   along the *model* axes, not a world bounding box. For a rotated model the
   centroids are assumed to be world coordinates and the sizes to be along
   the rotated model axes.
6. **One row per leaf block**, sub-blocks as separate rows; there are no rows
   for parents that have been split.
7. **Missing values**: Surpac's token for an unassigned value is unknown (the
   attribute's background value is the likely candidate). The token is an
   option on export (default: empty field) and on import.
8. **Row order** is not relied upon; ours is the block table order.
9. **Text attributes** are assumed to be written bare, quoted only when they
   contain the delimiter.

Because of this, nothing is hard-wired: a :class:`BlockCsvLayout` holds column
names, order, header lines and delimiter. Two layouts are provided,
``"surpac_like"`` (default) and ``"xyz"`` (``x, y, z, dx, dy, dz``, attributes;
one header line), and any other can be passed.

Coordinates
-----------
Inside the program blocks are ``x = east, y = north, z = up``. The swap to
Y, X, Z happens here and only here. Sizes are along the model i (x), j (y) and
k (z) axes.

Export
------
:func:`export_block_csv` writes the blocks selected by a constraint and / or a
mask, any subset of attributes, with a chosen number of decimals and a chosen
missing-value token. NaN and missing categories are written as that token,
never as 0. Without ``decimals`` floats are written with the shortest text
that reads back to the same number, so the round trip is exact. The file is
written in chunks to a temporary file and renamed when complete.

Import
------
:func:`import_block_csv` reads a CSV in chunks, works out what the columns are
(and reports every guess in ``report.assumptions``) and

* **into an existing model**: finds the block of each row from its centroid
  and size, splitting blocks where the CSV is finer than the model so that
  the sub-block structure matches, and assigns the attribute values. Row
  sizes must be ``parent size / 2**n`` and the centroid must sit on the
  corresponding grid; rows that do not are an error that lists them. Rows
  outside the model are reported as unmatched. A row coarser than the model's
  blocks at that place fills all the blocks it covers.
* **as a new model**: with ``definition=`` (required for a rotated model), or
  with the definition inferred from the rows of an **unrotated** model
  (:func:`infer_definition`): parent size = the largest block size present,
  origin = the minimum block corner, counts from the maximum corner,
  sub-block depth from the smallest size present.
"""

from __future__ import annotations

import csv
import os
import re
import uuid
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from bakumine.core.blockmodel.attributes import Attribute, check_name
from bakumine.core.blockmodel.blocks import BlockModel
from bakumine.core.blockmodel.definition import MAX_DEPTH, BlockModelDefinition

__all__ = [
    "CENTROID_SYNONYMS",
    "DEFAULT_MISSING",
    "LAYOUTS",
    "SIZE_SYNONYMS",
    "SURPAC_LIKE",
    "XYZ",
    "BlockCsvError",
    "BlockCsvExport",
    "BlockCsvFormat",
    "BlockCsvImport",
    "BlockCsvLayout",
    "BlockCsvTable",
    "export_block_csv",
    "get_layout",
    "import_block_csv",
    "infer_definition",
    "read_block_csv",
    "sniff_block_csv",
]

AXES = ("x", "y", "z")
_AXIS_LABEL = {"x": "easting (X)", "y": "northing (Y)", "z": "elevation (Z)"}
_DELIMITERS = {",": ",", ";": ";", "\t": "\t", "tab": "\t", " ": " ", "whitespace": " "}
_SNIFF_LINES = 200
_MAX_LISTED = 10

#: tokens read as "missing" by default (compared as text, after stripping)
DEFAULT_MISSING = ("", "nan", "NaN", "NAN")


class BlockCsvError(ValueError):
    """A block CSV cannot be written, understood or placed in the model."""


def _synonyms(axis: str, kind: str) -> tuple[str, ...]:
    a = axis
    if kind == "centroid":
        names = [a, f"{a}c", f"{a}centre", f"{a}center", f"{a}centroid", f"centroid{a}",
                 f"centre{a}", f"center{a}", f"{a}mid", f"mid{a}", f"{a}coord"]  # fmt: skip
        names += {"x": ["east", "easting"], "y": ["north", "northing"],
                  "z": ["elev", "elevation", "rl"]}[a]  # fmt: skip
    else:
        names = [f"d{a}", f"{a}inc", f"{a}size", f"size{a}", f"{a}dim", f"dim{a}", f"{a}length",
                 f"{a}len", f"length{a}", f"{a}extent", f"{a}width", f"block{a}", f"{a}block",
                 f"blocksize{a}", f"{a}blocksize"]  # fmt: skip
    return tuple(names)


#: recognised column names of the centroid, per axis (compared case-insensitively
#: with everything but letters and digits removed: ``Centroid_Y`` -> ``centroidy``)
CENTROID_SYNONYMS = {a: _synonyms(a, "centroid") for a in AXES}
#: recognised column names of the block dimension, per axis
SIZE_SYNONYMS = {a: _synonyms(a, "size") for a in AXES}


@dataclass(frozen=True)
class BlockCsvLayout:
    """Column layout of a block CSV.

    Parameters
    ----------
    name : label used in messages
    axis_order : order in which the three centroid columns (and the three
        size columns) are written, e.g. ``("y", "x", "z")`` = northing first.
        Also the order assumed when a file has no column names.
    centroid_names : column names of the centroid, **in ``axis_order``**
    size_names : column names of the block dimensions, in ``axis_order``;
        ``None`` writes no size columns (and a file without column names is
        then assumed to have none)
    delimiter : ``","``, ``";"``, ``"\\t"`` or ``"whitespace"`` (written as one space)
    preamble : free-text lines written before the column names. ``{layout}``,
        ``{n_blocks}``, ``{attributes}`` and ``{missing}`` are filled in.
    column_header : write the line of column names
    """

    name: str
    axis_order: tuple[str, str, str] = ("y", "x", "z")
    centroid_names: tuple[str, str, str] = ("ycentre", "xcentre", "zcentre")
    size_names: tuple[str, str, str] | None = ("ysize", "xsize", "zsize")
    delimiter: str = ","
    preamble: tuple[str, ...] = ()
    column_header: bool = True

    def __post_init__(self) -> None:
        order = tuple(str(a).lower() for a in self.axis_order)
        if sorted(order) != ["x", "y", "z"]:
            raise BlockCsvError(f"axis_order must be a permutation of x, y, z, got {order}")
        object.__setattr__(self, "axis_order", order)
        for label in ("centroid_names", "size_names"):
            names = getattr(self, label)
            if names is None and label == "size_names":
                continue
            names = tuple(str(n) for n in names)
            if len(names) != 3 or len(set(names)) != 3:
                raise BlockCsvError(f"{label} must be three distinct names")
            object.__setattr__(self, label, names)
        if self.delimiter not in _DELIMITERS:
            raise BlockCsvError(
                f"delimiter must be ',', ';', tab or 'whitespace', got {self.delimiter!r}"
            )
        object.__setattr__(self, "delimiter", _DELIMITERS[self.delimiter])
        object.__setattr__(self, "preamble", tuple(str(line) for line in self.preamble))

    @property
    def order_text(self) -> str:
        return ", ".join(a.upper() for a in self.axis_order)


#: default layout; PROVISIONAL, see the module docstring
SURPAC_LIKE = BlockCsvLayout(
    name="surpac_like",
    preamble=(
        "block model export; layout {layout} [provisional]; {n_blocks} blocks",
        "columns: centroid northing (Y) easting (X) elevation (Z); block size along Y X Z; "
        "attributes: {attributes}; missing value: '{missing}'",
    ),
)
#: plain layout in internal order: x, y, z, dx, dy, dz, attributes; one header line
XYZ = BlockCsvLayout(
    name="xyz",
    axis_order=("x", "y", "z"),
    centroid_names=("x", "y", "z"),
    size_names=("dx", "dy", "dz"),
)
LAYOUTS = {SURPAC_LIKE.name: SURPAC_LIKE, XYZ.name: XYZ}


def get_layout(layout: str | BlockCsvLayout | None) -> BlockCsvLayout:
    """A layout by name (``"surpac_like"``, ``"xyz"``); ``None`` gives the default."""
    if layout is None:
        return SURPAC_LIKE
    if isinstance(layout, BlockCsvLayout):
        return layout
    try:
        return LAYOUTS[layout]
    except KeyError:
        raise BlockCsvError(
            f"unknown block CSV layout {layout!r}; available: {', '.join(LAYOUTS)}"
        ) from None


# ===================================================================== export


@dataclass(frozen=True)
class BlockCsvExport:
    """What :func:`export_block_csv` wrote."""

    path: Path
    layout: str
    n_rows: int
    columns: tuple[str, ...]
    attributes: tuple[str, ...]
    missing: str


def _fixed(values: np.ndarray, decimals: int, missing: str) -> np.ndarray:
    fmt = f"%.{decimals}f"
    out = np.array([fmt % v for v in values.tolist()], dtype=object)
    nan = np.isnan(values)
    if nan.any():
        out[nan] = missing
    return out


def _token_as_number(token: str) -> float | None:
    try:
        value = float(token)
    except ValueError:
        return None
    return None if value != value else value


def export_block_csv(
    model: BlockModel,
    path: str | Path,
    *,
    layout: str | BlockCsvLayout | None = None,
    constraint: Any = None,
    resolver: Any = None,
    where: Any = None,
    attributes: Iterable[str] | None = None,
    decimals: int | Mapping[str, int] | None = None,
    coord_decimals: int | None = None,
    missing: str = "",
    chunk_size: int = 200_000,
) -> BlockCsvExport:
    """Write blocks to a CSV file, one row per (sub-)block, in table order.

    Parameters
    ----------
    layout : a layout name or :class:`BlockCsvLayout`; default ``"surpac_like"``
        (PROVISIONAL, see the module docstring).
    constraint, resolver : optional :class:`~bakumine.core.blockmodel.constraints.Constraint`
        selecting the blocks to write.
    where : optional bool mask over the block table, combined (AND) with the constraint.
    attributes : names to write, in this order; default all.
    decimals : decimals of float attributes: one number for all, or a mapping
        per attribute. Where nothing is given the attribute's own ``decimals``
        is used, and if that is unset the shortest exact text.
    coord_decimals : decimals of the centroid and size columns; default exact.
    missing : token written for NaN and for a missing category (default: an
        empty field). Never 0 unless you ask for ``"0"``. It is an error if
        an exported value equals the token, since it could not be told apart.
    chunk_size : rows formatted at a time (memory, not output, depends on it).
    """
    lay = get_layout(layout)
    sep = lay.delimiter
    missing = str(missing)
    unusable = any(ch in missing for ch in (sep, "\n", "\r", '"'))
    if unusable or (sep == " " and missing.strip() == ""):
        raise BlockCsvError(
            f"missing-value token {missing!r} cannot be used with delimiter {sep!r} "
            "(it must be non-empty for whitespace-separated files and must not contain "
            "the delimiter)"
        )
    names = model.attribute_names if attributes is None else list(attributes)
    specs = [model.attribute(name) for name in names]
    if len(set(names)) != len(names):
        raise BlockCsvError("an attribute is listed twice")
    geometry = list(lay.centroid_names) + list(lay.size_names or ())
    clash = [n for n in names if n in geometry]
    if clash:
        raise BlockCsvError(f"attribute name(s) {clash} collide with the layout's column names")
    per_attr: dict[str, int | None] = {}
    for spec in specs:
        if isinstance(decimals, Mapping):
            per_attr[spec.name] = decimals.get(spec.name, spec.decimals)
        else:
            per_attr[spec.name] = spec.decimals if decimals is None else int(decimals)

    mask = np.ones(model.n_blocks, dtype=bool)
    if constraint is not None:
        mask &= np.asarray(constraint.mask(model, resolver), dtype=bool)
    if where is not None:
        sel = np.asarray(where)
        if sel.dtype != bool or sel.shape != mask.shape:
            raise BlockCsvError("where must be a boolean mask with one value per block")
        mask &= sel
    rows = np.flatnonzero(mask)

    # a value equal to the missing token would be read back as missing
    token_number = _token_as_number(missing)
    tables: dict[str, np.ndarray] = {}
    for spec in specs:
        column = model.get(spec.name, rows)
        if spec.kind == "category":
            used = np.unique(column[column >= 0])
            if any(spec.categories[c] == missing for c in used):
                raise BlockCsvError(
                    f"attribute {spec.name!r} has the category {missing!r}, which is also the "
                    "missing-value token; choose another token"
                )
            tables[spec.name] = np.array([*spec.categories, missing], dtype=object)
        elif token_number is not None:
            # the reader turns the declared token into missing whether or not the
            # column has NaNs, so any equal value (also after rounding) is lost
            values = column.astype(np.float64)
            places = per_attr[spec.name] if spec.kind == "float" else None
            hit = values == token_number
            if places is not None:
                hit |= np.round(values, places) == np.round(token_number, places)
            if hit.any():
                raise BlockCsvError(
                    f"attribute {spec.name!r} contains the value {missing}, which is also the "
                    "missing-value token; choose another token"
                )
        del column

    axis_index = [AXES.index(a) for a in lay.axis_order]
    columns = [*geometry, *names]
    text = {
        "layout": lay.name,
        "n_blocks": len(rows),
        "attributes": " ".join(names) or "(none)",
        "missing": missing,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp-{uuid.uuid4().hex[:8]}")
    chunk_size = max(1, int(chunk_size))
    try:
        with open(tmp, "w", encoding="utf-8", newline="") as handle:
            for line in lay.preamble:
                try:
                    handle.write(line.format(**text) + "\n")
                except (KeyError, IndexError, ValueError) as error:
                    raise BlockCsvError(f"preamble line {line!r}: {error!r}") from None
            if lay.column_header:
                writer = csv.writer(handle, delimiter=sep, lineterminator="\n")
                writer.writerow(columns)
            for start in range(0, len(rows), chunk_size):
                part = rows[start : start + chunk_size]
                frame: dict[str, Any] = {}
                centroid = model.centroids(part)
                for name, a in zip(lay.centroid_names, axis_index, strict=True):
                    col = centroid[:, a]
                    frame[name] = col if coord_decimals is None else _fixed(col, coord_decimals, "")
                if lay.size_names is not None:
                    size = model.sizes(part)
                    for name, a in zip(lay.size_names, axis_index, strict=True):
                        col = size[:, a]
                        frame[name] = (
                            col if coord_decimals is None else _fixed(col, coord_decimals, "")
                        )
                for spec in specs:
                    col = model.get(spec.name, part)
                    if spec.kind == "category":
                        col = tables[spec.name][col]  # code -1 -> last entry, the token
                    elif spec.kind == "float" and per_attr[spec.name] is not None:
                        col = _fixed(col, per_attr[spec.name], missing)
                    frame[spec.name] = col
                pd.DataFrame(frame, copy=False).to_csv(
                    handle, sep=sep, header=False, index=False, na_rep=missing,
                    lineterminator="\n",
                )  # fmt: skip
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return BlockCsvExport(path, lay.name, len(rows), tuple(columns), tuple(names), missing)


# ====================================================================== sniff


@dataclass
class BlockCsvFormat:
    """What the reader decided about a file (:func:`sniff_block_csv`).

    Attributes
    ----------
    delimiter : ``","``, ``";"``, ``"\\t"`` or ``" "`` (any whitespace)
    skiprows : physical lines before the first data line
    has_column_header : a line of column names was found
    names : raw column names (given, read, or invented)
    centroid : axis -> column index
    size : axis -> column index, or None when the file has no size columns
    attributes : attribute name (made valid) -> column index
    assumptions : every guess, in words
    """

    delimiter: str
    skiprows: int
    has_column_header: bool
    names: list[str]
    centroid: dict[str, int]
    size: dict[str, int] | None
    attributes: dict[str, int]
    assumptions: list[str] = field(default_factory=list)


def _normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())


def _tokens(line: str, delimiter: str) -> list[str]:
    if delimiter == " ":
        return next(csv.reader([line.strip()], delimiter=" ", skipinitialspace=True), [])
    return [t.strip() for t in next(csv.reader([line], delimiter=delimiter), [])]


def _guess_delimiter(line: str) -> str:
    for candidate in (",", ";", "\t"):
        if candidate in line:
            return candidate
    return " "


def _is_number(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def _match_columns(names: Sequence[str]) -> tuple[dict[str, int], dict[str, int]]:
    normal = [_normalise(n) for n in names]
    centroid: dict[str, int] = {}
    size: dict[str, int] = {}
    for found, table in ((centroid, CENTROID_SYNONYMS), (size, SIZE_SYNONYMS)):
        for axis in AXES:
            for synonym in table[axis]:
                if synonym in normal:
                    found[axis] = normal.index(synonym)
                    break
    return centroid, size


def _attribute_name(raw: str, index: int, taken: set[str]) -> str:
    name = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip())
    name = re.sub(r"^[^A-Za-z]+", "", name).rstrip("_")[:60]
    if not name:
        name = f"column{index + 1}"
    try:
        check_name(name)
    except ValueError:
        name = f"{name}_attr"
    base, n = name, 2
    while name in taken:
        name = f"{base}_{n}"
        n += 1
    return name


def sniff_block_csv(
    path: str | Path,
    *,
    layout: str | BlockCsvLayout | None = None,
    columns: Sequence[str] | None = None,
    delimiter: str | None = None,
) -> BlockCsvFormat:
    """Work out the delimiter, header lines and columns of a block CSV.

    Parameters
    ----------
    layout : used only when the file has **no column names**: the centroid
        (and size) columns are then assumed to be in the layout's order.
        Default ``"surpac_like"`` = Y, X, Z, i.e. **northing first**.
    columns : column names to use instead of what the file says (for a file
        without names, or with wrong ones). One per column.
    delimiter : skip the guess (``","``, ``";"``, ``"\\t"``, ``"whitespace"``).

    Detection: leading lines are skipped until one whose first three fields
    are numbers (the first data line). Among the skipped lines, the last one
    in which a centroid column name for each of x, y and z is recognised is
    the column header (the data line is then the first line with numbers in
    those columns, wherever they are). A line of text right above the data
    with one word per column is also taken as the column header, so that
    unrecognised names are an error rather than silently replaced by the
    positional assumption. The delimiter is the first of ``, ; tab`` present
    in the data line, else whitespace.
    """
    lay = get_layout(layout)
    path = Path(path)
    if delimiter is not None:
        if delimiter not in _DELIMITERS:
            raise BlockCsvError(f"unknown delimiter {delimiter!r}")
        delimiter = _DELIMITERS[delimiter]
    assumptions: list[str] = []
    header_line: str | None = None
    data_line: str | None = None
    skipped: list[str] = []
    skiprows = 0
    with open(path, encoding="utf-8-sig", errors="replace", newline=None) as handle:
        for number, line in enumerate(handle):
            if number >= _SNIFF_LINES:
                break
            if not line.strip():
                continue
            sep = delimiter or _guess_delimiter(line)
            tokens = _tokens(line, sep)
            if columns is not None:
                where = sorted(_match_columns(list(columns))[0].values())
            elif header_line is not None:
                where = sorted(_match_columns(_tokens(header_line, sep))[0].values())
            else:
                where = [0, 1, 2]
            if len(where) == 3 and len(tokens) > where[-1]:
                if all(_is_number(tokens[i]) for i in where):
                    data_line, skiprows = line, number
                    delimiter = sep
                    break
            centroid, _ = _match_columns(tokens)
            if len(centroid) == 3:
                header_line = line
            skipped.append(line.rstrip("\n"))
    if data_line is None:
        raise BlockCsvError(
            f"{path}: no data found: no line in the first {_SNIFF_LINES} starts with three numbers"
        )
    sep_name = {",": "comma", ";": "semicolon", "\t": "tab", " ": "whitespace"}[delimiter]
    n_columns = len(_tokens(data_line, delimiter))
    if header_line is None and columns is None and skipped:
        # a line of names right above the data that we do not understand must not be
        # silently ignored in favour of the positional assumption
        above = _tokens(skipped[-1], delimiter)
        if len(above) == n_columns and not any(_is_number(t) for t in above):
            header_line = skipped[-1]
    n_free = len(skipped) - (header_line is not None)
    if n_free:
        assumptions.append(f"skipped {n_free} header line(s) that are not column names")

    if columns is not None:
        names = [str(c) for c in columns]
        assumptions.append("column names given by the caller")
    elif header_line is not None:
        names = _tokens(header_line, delimiter)
        while len(names) > n_columns and names[-1] == "":
            names.pop()
    else:
        names = []
    if names and len(names) != n_columns:
        raise BlockCsvError(
            f"{path}: {len(names)} column names but {n_columns} columns in the first data line "
            f"({sep_name}-separated)"
        )
    if names:
        centroid, size = _match_columns(names)
        lacking = [a for a in AXES if a not in centroid]
        if lacking:
            raise BlockCsvError(
                f"{path}: no centroid column recognised for {', '.join(lacking)} among "
                f"{names}; pass columns=[...] with names such as "
                f"{', '.join(lay.centroid_names)}"
            )
        if size and len(size) != 3:
            raise BlockCsvError(
                f"{path}: block size columns found for {sorted(size)} only; need all of x, y, z"
            )
        shown = ", ".join(f"{names[centroid[a]]!r} = {_AXIS_LABEL[a]}" for a in AXES)
        assumptions.append(f"centroid columns by name: {shown}")
        if size:
            assumptions.append(
                "block size columns by name: "
                + ", ".join(f"{names[size[a]]!r} along {a.upper()}" for a in AXES)
            )
    else:
        if n_columns < 3:  # pragma: no cover - the data-line test needs three numbers
            raise BlockCsvError(f"{path}: fewer than three columns")
        order = lay.axis_order
        centroid = {a: i for i, a in enumerate(order)}
        size = {}
        names = list(lay.centroid_names)
        assumptions.append(
            f"no column names in the file: assumed columns 1-3 are the centroid in "
            f"{lay.order_text} order (layout {lay.name!r})"
        )
        if lay.size_names is not None and n_columns >= 6:
            size = {a: 3 + i for i, a in enumerate(order)}
            names += list(lay.size_names)
            assumptions.append(
                f"no column names in the file: assumed columns 4-6 are the block size along "
                f"{lay.order_text}"
            )
        names += [f"attr{i + 1}" for i in range(n_columns - len(names))]
        if n_columns > len(centroid) + len(size):
            assumptions.append(
                "no column names in the file: attributes named attr1, attr2, ... in column order"
            )
    if not size:
        assumptions.append("no block size columns: every row is taken as a parent block")
    used = set(centroid.values()) | set(size.values())
    attributes: dict[str, int] = {}
    for index, raw in enumerate(names):
        if index in used:
            continue
        name = _attribute_name(raw, index, set(attributes))
        if name != raw:
            assumptions.append(f"column {raw!r} imported as attribute {name!r}")
        attributes[name] = index
    assumptions.insert(0, f"{sep_name}-separated")
    return BlockCsvFormat(
        delimiter=delimiter,
        skiprows=skiprows,
        has_column_header=header_line is not None,
        names=names,
        centroid=centroid,
        size=size or None,
        attributes=attributes,
        assumptions=assumptions,
    )


# ======================================================================= read


@dataclass
class BlockCsvTable:
    """A block CSV in memory (:func:`read_block_csv`), in internal conventions.

    Attributes
    ----------
    centroids : (n, 3) float64, world ``x, y, z`` (east, north, up)
    sizes : (n, 3) float64 along the model x, y, z axes, or None
    values : attribute name -> float64 / int64 values, or int32 codes of a category
    kinds : attribute name -> ``"float"``, ``"int"`` or ``"category"``
    categories : attribute name -> strings, in order of first appearance (code order)
    format : the :class:`BlockCsvFormat` that was used
    """

    centroids: np.ndarray
    sizes: np.ndarray | None
    values: dict[str, np.ndarray]
    kinds: dict[str, str]
    categories: dict[str, list[str]]
    format: BlockCsvFormat

    @property
    def n_rows(self) -> int:
        return len(self.centroids)


class _Restart(Exception):
    def __init__(self, names: set[str]):
        self.names = names


def _read_pass(path, fmt, wanted, as_text, text_missing, chunk_size):
    geometry = [fmt.centroid[a] for a in AXES]
    if fmt.size is not None:
        geometry += [fmt.size[a] for a in AXES]
    usecols = sorted({*geometry, *wanted.values()})
    dtype: dict[int, Any] = {i: np.float64 for i in geometry}
    dtype.update({wanted[name]: str for name in as_text})
    reader = pd.read_csv(
        path,
        sep=r"\s+" if fmt.delimiter == " " else fmt.delimiter,
        header=None,
        skiprows=fmt.skiprows,
        usecols=usecols,
        dtype=dtype,
        na_values=list(text_missing),
        keep_default_na=False,
        skipinitialspace=True,
        chunksize=chunk_size,
        encoding="utf-8-sig",
    )
    geo_parts: list[np.ndarray] = []
    parts: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    is_text: dict[str, bool | None] = {name: None for name in wanted}
    dictionary: dict[str, dict[str, int]] = {name: {} for name in wanted}
    conflict: set[str] = set()
    for chunk in reader:
        geo_parts.append(chunk[geometry].to_numpy(dtype=np.float64))
        for name, index in wanted.items():
            series = chunk[index]
            if pd.api.types.is_bool_dtype(series.dtype):
                series = series.astype(str)
            numeric = pd.api.types.is_numeric_dtype(series.dtype)
            if numeric and series.dtype.kind == "f" and bool(series.isna().all()):
                # a chunk without any value fits a text column as well as a numeric one
                parts[name].append(np.full(len(series), np.nan))
                continue
            if is_text[name] is None:
                is_text[name] = not numeric
            elif is_text[name] == numeric:
                conflict.add(name)
                continue
            if numeric:
                parts[name].append(series.to_numpy())
            else:
                codes, uniques = pd.factorize(series)
                table = dictionary[name]
                lookup = np.array(
                    [table.setdefault(str(u), len(table)) for u in uniques] + [-1], dtype=np.int32
                )
                parts[name].append(lookup[codes])
    for name in wanted:
        if is_text[name]:  # all-missing chunks of a text column: code -1
            parts[name] = [
                np.full(len(p), -1, dtype=np.int32) if p.dtype.kind == "f" else p
                for p in parts[name]
            ]
    if conflict:
        raise _Restart(conflict)
    return geo_parts, parts, is_text, dictionary


def read_block_csv(
    path: str | Path,
    *,
    layout: str | BlockCsvLayout | None = None,
    columns: Sequence[str] | None = None,
    delimiter: str | None = None,
    attributes: Iterable[str] | None = None,
    kinds: Mapping[str, Any] | None = None,
    missing: Iterable[str | float] = DEFAULT_MISSING,
    chunk_size: int = 200_000,
) -> BlockCsvTable:
    """Read a block CSV into arrays, in chunks.

    Parameters
    ----------
    layout, columns, delimiter : see :func:`sniff_block_csv`.
    attributes : names of the attribute columns to read; default all.
    kinds : attribute name -> ``"float"``, ``"int"``, ``"category"`` (or an
        :class:`Attribute`, whose kind is used) to overrule the guess. The
        guess: a column whose values are all whole numbers is ``int``, all
        numbers ``float``, anything else ``category``.
    missing : tokens that mean "no value" (default: empty field, ``nan``).
        Text tokens are compared as text; a token that is a number, e.g.
        ``-99``, is also compared by value (``-99.000`` matches) in numeric
        attribute columns. Missing numbers become NaN (the column is then
        float); missing categories become missing.
    chunk_size : rows parsed at a time.

    The centroid and size columns must be numbers in every row.
    """
    path = Path(path)
    fmt = sniff_block_csv(path, layout=layout, columns=columns, delimiter=delimiter)
    wanted = dict(fmt.attributes)
    if attributes is not None:
        names = list(attributes)
        unknown = [n for n in names if n not in wanted]
        if unknown:
            raise BlockCsvError(
                f"{path}: no attribute column(s) {unknown}; found: {list(wanted) or '(none)'}"
            )
        wanted = {n: wanted[n] for n in names}
    forced: dict[str, str] = {}
    for name, kind in (kinds or {}).items():
        kind = kind.kind if isinstance(kind, Attribute) else str(kind)
        if kind not in ("float", "int", "category"):
            raise BlockCsvError(f"kind of {name!r} must be float, int or category, got {kind!r}")
        if name in wanted:
            forced[name] = kind
    tokens = [str(t) if not isinstance(t, str) else t for t in missing]
    text_missing = sorted({t.strip() for t in tokens})
    numbers = sorted({v for v in map(_token_as_number, text_missing) if v is not None})
    if text_missing != sorted(DEFAULT_MISSING):
        fmt.assumptions.append(f"values read as missing: {text_missing}")

    as_text = {n for n, k in forced.items() if k == "category"}
    chunk_size = max(1, int(chunk_size))
    while True:
        try:
            geo_parts, parts, is_text, dictionary = _read_pass(
                path, fmt, wanted, as_text, text_missing, chunk_size
            )
            break
        except _Restart as restart:  # a column is text in one chunk and numbers in another
            as_text |= restart.names
        except (ValueError, pd.errors.ParserError) as error:
            raise BlockCsvError(f"{path}: cannot parse ({error})") from None
    geometry = np.concatenate(geo_parts) if geo_parts else np.zeros((0, 6))
    del geo_parts
    if len(geometry) == 0:
        raise BlockCsvError(f"{path}: no data rows")
    bad = np.flatnonzero(~np.isfinite(geometry).all(axis=1))
    if len(bad):
        raise BlockCsvError(
            f"{path}: {len(bad)} row(s) have a missing or non-finite centroid or size: "
            f"data row(s) {_listed(bad + 1)}"
        )
    centroids = np.ascontiguousarray(geometry[:, :3])
    sizes = np.ascontiguousarray(geometry[:, 3:6]) if fmt.size is not None else None
    del geometry

    values: dict[str, np.ndarray] = {}
    out_kinds: dict[str, str] = {}
    categories: dict[str, list[str]] = {}
    for name in wanted:
        if is_text[name]:
            values[name] = np.concatenate(parts[name]).astype(np.int32)
            out_kinds[name] = "category"
            categories[name] = list(dictionary[name])
            if forced.get(name, "category") != "category":
                raise BlockCsvError(
                    f"{path}: column {name!r} holds text and cannot be read as {forced[name]}"
                )
            continue
        column = np.concatenate(parts[name]) if parts[name] else np.zeros(0)
        for number in numbers:
            hit = column == number
            if hit.any():
                column = column.astype(np.float64)
                column[hit] = np.nan
        kind = "int" if column.dtype.kind in "iu" else "float"
        want = forced.get(name, kind)
        if want == "int" and kind == "float":
            if np.isnan(column).any():
                raise BlockCsvError(
                    f"{path}: column {name!r} has missing values and cannot be an integer "
                    "attribute (integers have no missing marker)"
                )
            if not (np.isfinite(column).all() and (column == np.rint(column)).all()):
                raise BlockCsvError(f"{path}: column {name!r} is not whole numbers")
            column = column.astype(np.int64)
        elif want == "float":
            column = column.astype(np.float64)
        else:
            column = column.astype(np.int64)
        values[name] = column
        out_kinds[name] = want
        parts[name] = []
    return BlockCsvTable(centroids, sizes, values, out_kinds, categories, fmt)


def _plain(values: Any) -> tuple[float, ...]:
    return tuple(float(v) for v in values)


def _listed(rows: np.ndarray) -> str:
    text = ", ".join(str(int(r)) for r in rows[:_MAX_LISTED])
    return text + (f", ... ({len(rows)} in all)" if len(rows) > _MAX_LISTED else "")


# ================================================================== placement


def _depths(
    sizes: np.ndarray, parent_size: np.ndarray, tolerance: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per row and axis, ``n`` with ``size = parent / 2**n``; and the rows where no ``n`` fits."""
    with np.errstate(divide="ignore", invalid="ignore"):
        n = np.rint(np.log2(parent_size / sizes))
        n = np.where(np.isfinite(n), n, -1.0)
        exact = parent_size / np.exp2(np.clip(n, 0, 60))
        bad = (sizes <= 0) | (n < 0) | (n > 60) | (np.abs(sizes - exact) > tolerance * exact)
    return n.astype(np.int64), bad.any(axis=1)


def _row_text(rows: np.ndarray, centroids: np.ndarray, sizes: np.ndarray | None) -> str:
    lines = []
    for r in rows[:_MAX_LISTED]:
        x, y, z = _plain(centroids[r])
        text = f"  data row {r + 1}: centroid Y {y!r}, X {x!r}, Z {z!r}"
        if sizes is not None:
            sx, sy, sz = _plain(sizes[r])
            text += f"; size Y {sy!r}, X {sx!r}, Z {sz!r}"
        lines.append(text)
    if len(rows) > _MAX_LISTED:
        lines.append(f"  ... {len(rows)} rows in all")
    return "\n".join(lines)


def _place(
    definition: BlockModelDefinition,
    centroids: np.ndarray,
    sizes: np.ndarray | None,
    tolerance: float,
):
    """Integer position of every row in the model: ``(index, units, level, inside, deviation)``.

    ``index`` is the minimum corner in finest cells along each axis, ``units``
    the block size in finest cells. Raises :class:`BlockCsvError` listing the
    rows whose size is not ``parent / 2**n`` (or not a combination the model
    can hold) or whose centroid is not on the grid of blocks of that size.
    """
    n_rows = len(centroids)
    depth = np.array(definition.max_depth, dtype=np.int64)
    parent = np.array(definition.block_size)
    fine = parent / (1 << depth)
    if sizes is None:
        level = np.zeros(n_rows, dtype=np.int64)
        n_axis = np.zeros((n_rows, 3), dtype=np.int64)
    else:
        n_axis, bad = _depths(sizes, parent, tolerance)
        if bad.any():
            rows = np.flatnonzero(bad)
            raise BlockCsvError(
                f"{len(rows)} row(s) have a block size that is not the parent size "
                f"{_plain(parent)} (x, y, z) divided by a power of two:\n"
                + _row_text(rows, centroids, sizes)
            )
        level = n_axis.max(axis=1)
        # a block of level L has size parent / 2**min(L, max_depth) on every axis
        bad = (np.minimum(level[:, None], depth[None, :]) != n_axis).any(axis=1)
        if bad.any():
            rows = np.flatnonzero(bad)
            raise BlockCsvError(
                f"{len(rows)} row(s) have a block size the model cannot hold: with "
                f"max_depth {definition.max_depth} (x, y, z) the smallest block is "
                f"{definition.min_block_size}, and every split halves all axes that are "
                "not yet at their smallest size:\n" + _row_text(rows, centroids, sizes)
            )
    units = np.int64(1) << (depth[None, :] - n_axis)
    size = units * fine
    local = definition.world_to_local(centroids)
    q = np.rint((local - 0.5 * size) / size)
    deviation = np.abs(local - (q + 0.5) * size) / size
    bad = (deviation > tolerance).any(axis=1)
    if bad.any():
        rows = np.flatnonzero(bad)
        raise BlockCsvError(
            f"{len(rows)} row(s) have a centroid that is not on the model's grid for a block "
            f"of that size (off by more than {tolerance:g} of the block size; model origin "
            f"{definition.origin}, parent size {definition.block_size}"
            + (", rotated" if definition.is_rotated else "")
            + "):\n"
            + _row_text(rows, centroids, sizes)
        )
    limit = np.array(definition.shape, dtype=np.float64) * (1 << depth)
    lo = q * units
    inside = ((lo >= 0) & (lo + units <= limit)).all(axis=1)
    index = np.where(inside[:, None], lo, 0).astype(np.int64)
    return index, units, level, inside, float(deviation.max()) if n_rows else 0.0


def infer_definition(
    centroids: np.ndarray, sizes: np.ndarray | None, *, tolerance: float = 0.01
) -> tuple[BlockModelDefinition, list[str]]:
    """Definition of an **unrotated** model from block centroids and sizes.

    Returns ``(definition, assumptions)``.

    * parent block size = the largest size present on each axis (if no parent
      block is left unsplit anywhere along an axis, the inferred parent is
      smaller than the original; the blocks are the same);
    * sub-block depth per axis from the smallest size present;
    * origin = the smallest ``centroid - size / 2``; counts from the largest
      ``centroid + size / 2``. Blocks missing from the file at the edges
      make the model smaller than the original.

    Without sizes every row is a parent block and the parent size is the
    smallest spacing between distinct centroid coordinates on each axis
    (an error if an axis has a single coordinate).
    """
    centroids = np.asarray(centroids, dtype=np.float64)
    notes = ["definition inferred from the rows, assuming an unrotated model"]
    if sizes is None:
        parent = np.empty(3)
        for a in range(3):
            unique = np.unique(centroids[:, a])
            if len(unique) < 2:
                raise BlockCsvError(
                    f"cannot infer the block size along {AXES[a]}: the file has no size columns "
                    "and a single centroid coordinate on that axis; supply a definition"
                )
            parent[a] = np.diff(unique).min()
        notes.append(
            f"parent block size {_plain(parent)} (x, y, z) taken from the smallest centroid spacing"
        )
        exact = np.broadcast_to(parent, centroids.shape)
        depth = np.zeros(3, dtype=np.int64)
    else:
        sizes = np.asarray(sizes, dtype=np.float64)
        if not (np.isfinite(sizes).all() and (sizes > 0).all()):
            raise BlockCsvError("block sizes must be positive numbers")
        parent = sizes.max(axis=0)
        n_axis, bad = _depths(sizes, parent, tolerance)
        if bad.any():
            rows = np.flatnonzero(bad)
            raise BlockCsvError(
                f"{len(rows)} row(s) have a block size that is not the largest size "
                f"{_plain(parent)} (x, y, z) divided by a power of two:\n"
                + _row_text(rows, centroids, sizes)
            )
        depth = n_axis.max(axis=0)
        if (depth > MAX_DEPTH).any():
            raise BlockCsvError(
                f"sub-blocks {1 << int(depth.max())} times smaller than the largest block: "
                f"deeper than the supported {MAX_DEPTH} levels"
            )
        exact = parent / np.exp2(n_axis)
        notes.append(f"parent block size {_plain(parent)} (x, y, z) = the largest size present")
    corner = centroids - 0.5 * exact
    origin = corner.min(axis=0)
    # Coordinates written with few decimals: the minimum corner carries one row's rounding
    # error. Use all rows (median offset from the grid of their own size), then the roundest
    # number within a quarter of the scatter.
    offset = corner - origin
    offset -= np.rint(offset / exact) * exact
    scatter = np.abs(offset).max(axis=0)
    for a in range(3):
        if scatter[a] > 1e-9 * parent[a]:
            estimate = origin[a] + float(np.median(offset[:, a]))
            for digits in range(0, 12):
                if abs(round(estimate, digits) - estimate) <= 0.25 * scatter[a]:
                    estimate = round(estimate, digits)
                    break
            origin[a] = estimate
            notes.append(
                f"centroids are off the grid by up to {scatter[a]:.3g} m along {AXES[a]} "
                f"(rounded coordinates?): origin estimated as {estimate!r} from all rows"
            )
    top = (centroids + 0.5 * exact).max(axis=0)
    counts = np.rint((top - origin) / parent)
    if (counts < 1).any() or (np.abs(counts * parent - (top - origin)) > tolerance * parent).any():
        raise BlockCsvError(
            f"the blocks do not span a whole number of parent blocks {_plain(parent)}: "
            f"extent {_plain(top - origin)} from origin {_plain(origin)}; "
            "is the model rotated? then supply its definition"
        )
    definition = BlockModelDefinition(
        tuple(origin), tuple(parent), tuple(int(c) for c in counts),
        max_depth=tuple(int(d) for d in depth),
    )  # fmt: skip
    notes.append(
        f"origin {definition.origin}, shape {definition.shape}, max_depth {definition.max_depth}"
    )
    return definition, notes


# ===================================================================== import


@dataclass
class BlockCsvImport:
    """What :func:`import_block_csv` did.

    Attributes
    ----------
    n_rows : data rows in the file
    n_assigned : rows whose values were put into the model
    n_unmatched : rows outside the model (not assigned)
    unmatched : up to 20 ``(data row number, reason)``
    n_split : blocks split so that the structure matches the file
    n_coarser : rows coarser than the model's blocks there (each filled all
        the blocks it covers)
    n_duplicates : rows that target a block another row also targets (the
        last one wins)
    created, updated : attribute names
    max_deviation : largest centroid offset from the exact block centroid, as
        a fraction of the block size
    definition_inferred : the definition was inferred from the rows
    assumptions : every guess made about the file and the model, in words
    """

    n_rows: int
    n_assigned: int
    n_unmatched: int
    unmatched: list[tuple[int, str]]
    n_split: int
    n_coarser: int
    n_duplicates: int
    created: list[str]
    updated: list[str]
    max_deviation: float
    definition_inferred: bool
    assumptions: list[str]

    def summary(self) -> str:
        lines = [
            f"rows {self.n_rows}, assigned {self.n_assigned}, unmatched {self.n_unmatched}, "
            f"blocks split {self.n_split}",
        ]
        if self.n_coarser or self.n_duplicates:
            lines.append(
                f"rows coarser than the model {self.n_coarser}, duplicate rows {self.n_duplicates}"
            )
        lines.append(
            f"attributes created: {', '.join(self.created) or '(none)'}; "
            f"updated: {', '.join(self.updated) or '(none)'}"
        )
        lines += [f"unmatched data row {row}: {reason}" for row, reason in self.unmatched]
        lines += [f"assumed: {text}" for text in self.assumptions]
        return "\n".join(lines)


def import_block_csv(
    path: str | Path,
    model: BlockModel | None = None,
    *,
    definition: BlockModelDefinition | None = None,
    layout: str | BlockCsvLayout | None = None,
    columns: Sequence[str] | None = None,
    delimiter: str | None = None,
    attributes: Iterable[str] | None = None,
    kinds: Mapping[str, Any] | None = None,
    missing: Iterable[str | float] = DEFAULT_MISSING,
    tolerance: float = 0.01,
    chunk_size: int = 200_000,
) -> tuple[BlockModel, BlockCsvImport]:
    """Import a block CSV into an existing model or as a new one.

    Parameters
    ----------
    model : the model to import into (modified in place and returned). Without
        it a new model is made from ``definition`` or, if that is not given
        either, from a definition inferred from the rows
        (:func:`infer_definition`; unrotated models only).
    definition : definition of the new model. Required for a rotated model:
        centroids are world coordinates, sizes are along the model axes.
    layout, columns, delimiter, attributes, kinds, missing, chunk_size :
        see :func:`read_block_csv`. ``kinds`` may map a name to an
        :class:`Attribute`, which is then used to create the attribute
        (dtype, background, category order, unit, decimals).
    tolerance : how far, as a fraction of the block size, a size may be from
        ``parent / 2**n`` and a centroid from its exact position (default
        0.01; raise it for files written with few decimals).

    New attributes are created as float (background NaN), int (int32, or
    int64 when needed; background 0) or category (background missing);
    blocks without a row keep the background. For an existing attribute the
    column must suit its kind.

    Returns ``(model, report)``; see :class:`BlockCsvImport`.
    """
    if model is not None and definition is not None:
        raise BlockCsvError("give either a model to import into or a definition, not both")
    if not 0 < tolerance < 0.25:
        raise BlockCsvError("tolerance must be between 0 and 0.25 of the block size")
    kinds = dict(kinds or {})
    if model is not None:
        for name in model.attribute_names:
            kinds.setdefault(name, model.attribute(name).kind)
    table = read_block_csv(
        path, layout=layout, columns=columns, delimiter=delimiter, attributes=attributes,
        kinds=kinds, missing=missing, chunk_size=chunk_size,
    )  # fmt: skip
    assumptions = list(table.format.assumptions)
    inferred = False
    if model is None:
        if definition is None:
            definition, notes = infer_definition(
                table.centroids, table.sizes, tolerance=tolerance
            )
            assumptions += notes
            inferred = True
        model = BlockModel(definition)
    definition = model.definition
    index, units, level, inside, deviation = _place(
        definition, table.centroids, table.sizes, tolerance
    )
    n_rows = table.n_rows
    outside = np.flatnonzero(~inside)
    unmatched = [(int(r) + 1, "outside the model") for r in outside[:20]]
    source = np.flatnonzero(inside)
    index, units, level = index[source], units[source], level[source]

    depth = np.array(definition.max_depth, dtype=np.int64)
    half_fine = np.array(definition.block_size) / (1 << (depth + 1))
    points = definition.local_to_world((2 * index + units) * half_fine)
    n_split = 0
    while True:
        rows = model.locate(points)
        finer = model.level[rows] < level
        if not finer.any():
            break
        n_split += model.split(np.unique(rows[finer])).n_split
    coarse = model.level[rows] > level
    n_coarser = int(coarse.sum())
    exact_rows = rows[~coarse]
    n_duplicates = len(exact_rows) - len(np.unique(exact_rows))
    if n_coarser:
        # all leaves inside the CSV block: a contiguous key range starting at its first leaf
        first = model.locate(definition.local_to_world((2 * index[coarse] + 1) * half_fine))
        keys = model.keys()
        span = np.int64(1) << (3 * (model.max_level - level[coarse]))
        last = np.searchsorted(keys, keys[first] + span)
        count = last - first
        offset = np.arange(int(count.sum())) - np.repeat(np.cumsum(count) - count, count)
        rows = np.concatenate([exact_rows, np.repeat(first, count) + offset])
        source = np.concatenate([source[~coarse], np.repeat(source[coarse], count)])

    created: list[str] = []
    updated: list[str] = []
    for name, column in table.values.items():
        kind = table.kinds[name]
        if name in model:
            spec = model.attribute(name)
            updated.append(name)
        else:
            given = kinds.get(name)
            if isinstance(given, Attribute):
                spec = model.add_attribute(
                    name, given.kind, background=given.background, dtype=given.dtype,
                    description=given.description, unit=given.unit, decimals=given.decimals,
                    categories=given.categories,
                )  # fmt: skip
            elif kind == "int":
                info = np.iinfo(np.int32)
                small = len(column) == 0 or (column.min() >= info.min and column.max() <= info.max)
                spec = model.add_attribute(name, "int", dtype=np.int32 if small else np.int64)
            else:
                spec = model.add_attribute(name, kind)
            created.append(name)
        if spec.kind == "category":
            if kind != "category":
                raise BlockCsvError(f"column {name!r} was not read as text")  # pragma: no cover
            lookup = np.array(
                [spec.code_of(c, add=True) for c in table.categories[name]] + [-1], dtype=np.int64
            )
            model.set(name, lookup[column[source]], rows)
        elif kind == "category":
            raise BlockCsvError(
                f"attribute {name!r} is {spec.kind} but the column holds text"
            )
        else:
            selected = column[source]
            if spec.kind == "int" and selected.dtype.kind == "f":
                if np.isnan(selected).any():
                    raise BlockCsvError(
                        f"attribute {name!r} is an integer attribute but the column has missing "
                        "values (integers have no missing marker)"
                    )
            try:
                model.set(name, selected, rows)
            except (ValueError, TypeError) as error:
                raise BlockCsvError(str(error)) from None
    report = BlockCsvImport(
        n_rows=n_rows,
        n_assigned=n_rows - len(outside),
        n_unmatched=len(outside),
        unmatched=unmatched,
        n_split=n_split,
        n_coarser=n_coarser,
        n_duplicates=n_duplicates,
        created=created,
        updated=updated,
        max_deviation=deviation,
        definition_inferred=inferred,
        assumptions=assumptions,
    )
    return model, report
