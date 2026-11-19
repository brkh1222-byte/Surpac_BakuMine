"""Typed block attributes (research.md section 6.3).

An attribute is a named column of the block table. Three kinds:

``"float"``
    float64 (default) or float32. **NaN is the missing marker.** Default
    background NaN.
``"int"``
    int8 / int16 / int32 (default) / int64. Integers have no missing marker;
    choose a background value that means "not assigned". Default background 0.
``"category"``
    Strings, dictionary-encoded: the column holds integer codes (int32 by
    default; int8 / int16 to save memory) and :attr:`Attribute.categories`
    lists the strings, code ``c`` standing for ``categories[c]``. **Code -1
    (:data:`MISSING_CODE`) is the missing marker** and decodes to ``None``.
    New strings are added to the dictionary when they are first assigned;
    codes of existing strings never change. Default background missing.

Every attribute has a **background value**: the value of every block when the
attribute is created, and the value to which blocks are reset. (Sub-blocks
created by a split inherit their parent's values, which are the background
wherever nothing was assigned.)

Names are identifiers (letters, digits, underscore; not starting with a digit
or an underscore), so that they can be used in expressions, and must not
collide with the built-in variables and functions of the expression language
(:data:`RESERVED_NAMES`) or with a Python keyword: expressions are parsed
with Python's grammar, so e.g. lowercase ``as``, ``in`` and ``is`` cannot be
names (``As``, ``AS``, ``In`` are fine; names are case-sensitive).
"""

from __future__ import annotations

import keyword
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

__all__ = ["BUILTIN_VARIABLES", "KINDS", "MISSING_CODE", "RESERVED_NAMES", "Attribute"]

#: code of a missing categorical value
MISSING_CODE = -1

KINDS = ("float", "int", "category")

#: variables every block has, usable in expressions (world centroid, size, volume)
BUILTIN_VARIABLES = ("x", "y", "z", "dx", "dy", "dz", "volume")

#: functions and constants of the expression language
EXPRESSION_FUNCTIONS = (
    "min", "max", "abs", "sqrt", "log", "log10", "exp", "where", "iif", "isnan",
    "floor", "ceil", "round",
)  # fmt: skip
EXPRESSION_CONSTANTS = ("nan", "pi", "true", "false")

#: names an attribute cannot have
RESERVED_NAMES = frozenset((*BUILTIN_VARIABLES, *EXPRESSION_FUNCTIONS, *EXPRESSION_CONSTANTS))

_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_]*\Z")
_MAX_NAME = 64

_DTYPES = {
    "float": (np.dtype(np.float64), (np.dtype(np.float64), np.dtype(np.float32))),
    "int": (
        np.dtype(np.int32),
        (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32), np.dtype(np.int64)),
    ),
    "category": (
        np.dtype(np.int32),
        (np.dtype(np.int8), np.dtype(np.int16), np.dtype(np.int32)),
    ),
}


def check_name(name: Any) -> str:
    """Validate an attribute name; returns it or raises ``ValueError``."""
    if not isinstance(name, str) or not _NAME.match(name) or len(name) > _MAX_NAME:
        raise ValueError(
            f"invalid attribute name {name!r}: use letters, digits and underscores, "
            f"starting with a letter, at most {_MAX_NAME} characters"
        )
    if (
        name in RESERVED_NAMES
        or keyword.iskeyword(name)
        or name.lower() in ("and", "or", "not")
    ):
        raise ValueError(f"attribute name {name!r} is reserved by the expression language")
    return name


@dataclass
class Attribute:
    """Description of one attribute column.

    Create attributes through :meth:`BlockModel.add_attribute`; the fields
    ``description``, ``unit`` and ``decimals`` may be edited freely afterwards.

    Attributes
    ----------
    name : str
    kind : ``"float"``, ``"int"`` or ``"category"``
    dtype : numpy dtype of the stored column (codes for a category)
    background : float, int, or for a category a string or ``None`` (missing)
    description, unit : free text
    decimals : number of decimals for display and export, or None
    categories : list of str -- the dictionary of a category attribute
    """

    name: str
    kind: str = "float"
    dtype: Any = None
    background: Any = None
    description: str = ""
    unit: str = ""
    decimals: int | None = None
    categories: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        check_name(self.name)
        if self.kind not in KINDS:
            raise ValueError(f"attribute kind must be one of {KINDS}, got {self.kind!r}")
        default, allowed = _DTYPES[self.kind]
        self.dtype = default if self.dtype is None else np.dtype(self.dtype)
        if self.dtype not in allowed:
            raise ValueError(
                f"dtype {self.dtype} is not allowed for a {self.kind} attribute "
                f"(allowed: {', '.join(str(d) for d in allowed)})"
            )
        if self.decimals is not None:
            if int(self.decimals) != self.decimals or self.decimals < 0:
                raise ValueError("decimals must be a non-negative integer or None")
            self.decimals = int(self.decimals)
        self.categories = list(self.categories)
        if self.kind != "category" and self.categories:
            raise ValueError("only category attributes have categories")
        if any(not isinstance(c, str) for c in self.categories) or len(
            set(self.categories)
        ) != len(self.categories):
            raise ValueError("categories must be distinct strings")
        if self.kind == "float":
            self.background = float("nan") if self.background is None else float(self.background)
        elif self.kind == "int":
            bg = 0 if self.background is None else self.background
            if isinstance(bg, bool) or int(bg) != bg:
                raise ValueError(f"background of an int attribute must be an integer, got {bg!r}")
            info = np.iinfo(self.dtype)
            if not info.min <= int(bg) <= info.max:
                raise ValueError(f"background {bg} does not fit in {self.dtype}")
            self.background = int(bg)
        else:
            if self.background is not None and not isinstance(self.background, str):
                raise ValueError("background of a category attribute must be a string or None")
            if self.background is not None:
                self.code_of(self.background, add=True)

    # ------------------------------------------------------------ categories

    def code_of(self, value: str | None, *, add: bool = False) -> int:
        """Code of a string (``None`` -> -1). Unknown strings: added if ``add``, else -2.

        -2 never occurs in a column, so comparing codes with it is False everywhere.
        """
        self._require_category()
        if value is None:
            return MISSING_CODE
        if not isinstance(value, str):
            raise TypeError(f"category values must be strings or None, got {type(value).__name__}")
        try:
            return self.categories.index(value)
        except ValueError:
            if not add:
                return -2
        if len(self.categories) > np.iinfo(self.dtype).max:
            raise ValueError(
                f"attribute {self.name!r}: more categories than {self.dtype} codes can hold"
            )
        self.categories.append(value)
        return len(self.categories) - 1

    def encode(self, values: Any) -> np.ndarray:
        """Strings (or ``None``) -> codes of this attribute's dtype; new strings are added."""
        self._require_category()
        if values is None or isinstance(values, str):
            return np.array(self.code_of(values, add=True), dtype=self.dtype)
        arr = np.asarray(values, dtype=object)
        flat = arr.reshape(-1)
        missing = np.array([v is None for v in flat], dtype=bool)
        codes = np.full(len(flat), MISSING_CODE, dtype=self.dtype)
        if not missing.all():
            present = flat[~missing]
            if any(not isinstance(v, str) for v in present):
                raise TypeError("category values must be strings or None")
            uniques, inverse = np.unique(present.astype(str), return_inverse=True)
            lookup = np.array([self.code_of(str(u), add=True) for u in uniques], dtype=self.dtype)
            codes[~missing] = lookup[inverse]
        return codes.reshape(arr.shape)

    def decode(self, codes: Any) -> np.ndarray:
        """Codes -> object array of strings, ``None`` where missing."""
        self._require_category()
        codes = np.asarray(codes)
        table = np.array([*self.categories, None], dtype=object)  # code -1 -> last entry = None
        if codes.size and (codes.min() < MISSING_CODE or codes.max() >= len(self.categories)):
            raise ValueError(f"attribute {self.name!r}: code outside the category dictionary")
        return table[codes]

    def _require_category(self) -> None:
        if self.kind != "category":
            raise TypeError(f"attribute {self.name!r} is not a category attribute")

    # ---------------------------------------------------------------- values

    @property
    def background_stored(self) -> Any:
        """The background as stored in the column (the code for a category)."""
        if self.kind == "category":
            return self.code_of(self.background)
        return self.background

    def to_stored(self, values: Any) -> np.ndarray:
        """Convert user values to the stored dtype (encoding category strings)."""
        if self.kind == "category":
            arr = np.asarray(values) if not isinstance(values, str) and values is not None else None
            if arr is not None and arr.dtype.kind in "iu":  # already codes
                if arr.size and (arr.min() < MISSING_CODE or arr.max() >= len(self.categories)):
                    raise ValueError(
                        f"attribute {self.name!r}: code outside the category dictionary"
                    )
                return arr.astype(self.dtype)
            return self.encode(values)
        arr = np.asarray(values)
        if arr.dtype.kind not in "fiub":
            raise TypeError(
                f"attribute {self.name!r} is {self.kind}; cannot store values of dtype {arr.dtype}"
            )
        if self.kind == "int":
            if arr.dtype.kind == "f":
                if not (np.isfinite(arr).all() and (arr == np.rint(arr)).all()):
                    raise ValueError(
                        f"attribute {self.name!r} is an integer attribute; "
                        "values must be whole finite numbers"
                    )
            info = np.iinfo(self.dtype)
            if arr.size and (arr.min() < info.min or arr.max() > info.max):
                raise ValueError(f"attribute {self.name!r}: value does not fit in {self.dtype}")
        return arr.astype(self.dtype)

    # --------------------------------------------------------- serialisation

    def to_dict(self) -> dict[str, Any]:
        """JSON-ready dict (NaN background is written as ``None``)."""
        bg = self.background
        if self.kind == "float" and bg != bg:
            bg = None
        out: dict[str, Any] = {
            "name": self.name,
            "kind": self.kind,
            "dtype": str(self.dtype),
            "background": bg,
            "description": self.description,
            "unit": self.unit,
            "decimals": self.decimals,
        }
        if self.kind == "category":
            out["categories"] = list(self.categories)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Attribute:
        return cls(
            name=data["name"],
            kind=data.get("kind", "float"),
            dtype=data.get("dtype"),
            background=data.get("background"),
            description=data.get("description", ""),
            unit=data.get("unit", ""),
            decimals=data.get("decimals"),
            categories=list(data.get("categories", [])),
        )
