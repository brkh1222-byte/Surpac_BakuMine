"""Safe expressions over block attributes (research.md sections 6.3 and 6.4).

Used for calculated attributes (``"au * density * volume"``) and for attribute
constraints (``'au > 0.5 and rock == "ORE"'``).

Safety
------
An expression is parsed with Python's :mod:`ast` (which only builds a syntax
tree and runs nothing) and then checked against a strict whitelist. It is
evaluated by walking that tree with numpy; ``eval`` / ``exec`` / ``compile``
are never used. Anything not listed below is rejected with
:class:`ExpressionError` before any evaluation: attribute access (``a.b``),
subscripts (``a[0]``), lambdas, comprehensions, f-strings, assignments,
keyword or starred arguments, calls of anything but the listed functions,
names starting with an underscore, bytes / complex / ``None`` literals.
Size is bounded: at most :data:`MAX_LENGTH` characters, :data:`MAX_NODES`
syntax nodes and :data:`MAX_NESTING` levels of nesting. All numbers are
float64, so ``9 ** 9 ** 9`` is ``inf``, not a long computation.

Language
--------
* numbers: ``1``, ``2.5``, ``1e-3``; ``True`` / ``False``; constants ``nan``,
  ``pi``, ``true``, ``false``;
* names: attributes of the model and the built-in variables ``x``, ``y``,
  ``z`` (world centroid), ``dx``, ``dy``, ``dz`` (block size along the model
  axes), ``volume``;
* arithmetic ``+ - * /`` and power ``**``, unary ``-`` and ``+``, parentheses.
  ``^`` is rejected (in Python's grammar it is not power and binds loosely);
* comparisons ``== != < <= > >=``, also chained (``0.5 <= au < 2``). A single
  ``=`` is accepted as ``==``;
* logic ``and``, ``or``, ``not`` (any letter case);
* strings (``"ORE"`` or ``'ORE'``) only on one side of ``==`` / ``!=`` with a
  category attribute on the other: ``rock == "ORE"``;
* functions: ``min(a, b, ...)``, ``max(a, b, ...)`` (element-wise, two or more
  arguments), ``abs``, ``sqrt``, ``log`` (natural), ``log10``, ``exp``,
  ``floor``, ``ceil``, ``round(a)`` / ``round(a, digits)``,
  ``where(cond, a, b)`` = ``iif(cond, a, b)``, ``isnan(a)``.

Semantics
---------
Evaluation is element-wise over blocks. Missing values are NaN and propagate
through arithmetic. Every comparison with NaN is False, except ``!=`` which
is True (IEEE rules); use ``isnan`` to treat missing values explicitly. In
``and`` / ``or`` / ``not`` / ``where``
a number counts as true when it is non-zero and not NaN. For a category
attribute ``isnan(rock)`` is True where it is missing; ``rock == "X"`` is
False and ``rock != "X"`` True where it is missing; a string that is not in
the attribute's dictionary simply matches nothing.

Division by zero, ``log`` of a non-positive number, ``sqrt`` of a negative
number and overflow do not raise: the result is ``inf`` or NaN as IEEE
arithmetic defines it, and the number of elements where an operation turned
finite inputs into a non-finite result is reported as
:attr:`Evaluation.n_invalid`.
"""

from __future__ import annotations

import ast
import io
import tokenize
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, NamedTuple

import numpy as np

from bakumine.core.blockmodel.attributes import EXPRESSION_CONSTANTS, EXPRESSION_FUNCTIONS

__all__ = [
    "MAX_LENGTH",
    "MAX_NESTING",
    "MAX_NODES",
    "CategoryValues",
    "Evaluation",
    "Expression",
    "ExpressionError",
    "ExpressionWarning",
    "parse",
]

MAX_LENGTH = 1000
MAX_NODES = 250
MAX_NESTING = 100


class ExpressionError(ValueError):
    """The expression is not valid in the expression language (or names something unknown)."""


class ExpressionWarning(UserWarning):
    """An expression produced inf / NaN from finite inputs (division by zero, log of <= 0...)."""


@dataclass(frozen=True)
class CategoryValues:
    """A category attribute as an expression operand: integer codes + the dictionary."""

    codes: np.ndarray
    categories: tuple[str, ...]


class Evaluation(NamedTuple):
    """Result of :meth:`Expression.evaluate`.

    Attributes
    ----------
    values : numpy array (bool for a logical expression, else float64), or a
        numpy scalar when the expression uses no array variable.
    n_invalid : number of elements where an operation produced inf / NaN from
        finite operands.
    """

    values: Any
    n_invalid: int


_CONSTANTS = {"nan": np.float64(np.nan), "pi": np.float64(np.pi), "true": True, "false": False}
assert set(_CONSTANTS) == set(EXPRESSION_CONSTANTS)

_BINARY = {ast.Add: np.add, ast.Sub: np.subtract, ast.Mult: np.multiply,
           ast.Div: np.divide, ast.Pow: np.power}  # fmt: skip
_COMPARE = {ast.Eq: np.equal, ast.NotEq: np.not_equal, ast.Lt: np.less, ast.LtE: np.less_equal,
            ast.Gt: np.greater, ast.GtE: np.greater_equal}  # fmt: skip
_COMPARE_SYMBOL = {ast.Eq: "==", ast.NotEq: "!=", ast.Lt: "<", ast.LtE: "<=", ast.Gt: ">",
                   ast.GtE: ">="}  # fmt: skip
_UNARY_MATH = {"abs": np.abs, "sqrt": np.sqrt, "log": np.log, "log10": np.log10, "exp": np.exp,
               "floor": np.floor, "ceil": np.ceil}  # fmt: skip
_ARITY = {"min": (2, None), "max": (2, None), "where": (3, 3), "iif": (3, 3), "isnan": (1, 1),
          "round": (1, 2), **{name: (1, 1) for name in _UNARY_MATH}}  # fmt: skip
assert set(_ARITY) == set(EXPRESSION_FUNCTIONS)

_LEAVES = (ast.Load, ast.operator, ast.unaryop, ast.boolop, ast.cmpop)


def _normalise(source: str) -> str:
    """Accept ``AND`` / ``OR`` / ``NOT`` in any case and a single ``=`` as equality."""
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(source).readline))
    except (tokenize.TokenError, SyntaxError, IndentationError) as exc:
        raise ExpressionError(f"cannot read expression {source!r}: {exc}") from None
    out = source
    for tok in reversed(tokens):
        replacement = None
        if tok.type == tokenize.NAME and tok.string.lower() in ("and", "or", "not"):
            replacement = tok.string.lower()
        elif tok.type == tokenize.OP and tok.string == "=":
            replacement = "=="
        if replacement is not None and replacement != tok.string:
            out = out[: tok.start[1]] + replacement + out[tok.end[1] :]
    return out


def _check_size(tree: ast.AST, source: str) -> None:
    """Bound the node count and nesting without recursion."""
    stack = [(tree, 0)]
    count = 0
    while stack:
        node, depth = stack.pop()
        if isinstance(node, _LEAVES):
            continue
        count += 1
        if depth > MAX_NESTING:
            raise ExpressionError(f"expression is nested deeper than {MAX_NESTING} levels")
        if count > MAX_NODES:
            raise ExpressionError(f"expression has more than {MAX_NODES} terms")
        stack.extend((child, depth + 1) for child in ast.iter_child_nodes(node))


def _reject(node: ast.AST, what: str) -> ExpressionError:
    return ExpressionError(f"{what} is not allowed in an expression")


def _validate(node: ast.AST, names: set[str], in_compare: bool = False) -> None:
    """Whitelist check; collects the variable names used."""
    if isinstance(node, ast.Expression):
        _validate(node.body, names)
    elif isinstance(node, ast.Constant):
        value = node.value
        if isinstance(value, str):
            if not in_compare:
                raise ExpressionError(
                    f"the string {value!r} can only be compared with a category attribute "
                    "using == or !="
                )
        elif not isinstance(value, bool | int | float):
            raise _reject(node, f"the literal {value!r}")
    elif isinstance(node, ast.Name):
        if not isinstance(node.ctx, ast.Load):
            raise _reject(node, "assignment")
        if node.id.startswith("_"):
            raise ExpressionError(f"the name {node.id!r} is not allowed (leading underscore)")
        if node.id in _ARITY:
            raise ExpressionError(f"{node.id!r} is a function; call it as {node.id}(...)")
        if node.id not in _CONSTANTS:
            names.add(node.id)
    elif isinstance(node, ast.BinOp):
        if type(node.op) not in _BINARY:
            hint = "; use ** for power" if isinstance(node.op, ast.BitXor) else ""
            raise _reject(node, f"the operator {type(node.op).__name__}{hint}")
        _validate(node.left, names)
        _validate(node.right, names)
    elif isinstance(node, ast.UnaryOp):
        if not isinstance(node.op, ast.USub | ast.UAdd | ast.Not):
            raise _reject(node, f"the operator {type(node.op).__name__}")
        _validate(node.operand, names)
    elif isinstance(node, ast.BoolOp):
        for value in node.values:
            _validate(value, names)
    elif isinstance(node, ast.Compare):
        for op in node.ops:
            if type(op) not in _COMPARE:
                raise _reject(node, f"the comparison {type(op).__name__}")
        operands = [node.left, *node.comparators]
        for n, operand in enumerate(operands):
            is_str = isinstance(operand, ast.Constant) and isinstance(operand.value, str)
            if is_str:
                ops = node.ops[max(n - 1, 0) : n + 1]
                if not all(isinstance(op, ast.Eq | ast.NotEq) for op in ops):
                    raise ExpressionError("strings can only be compared with == or !=")
            _validate(operand, names, in_compare=True)
    elif isinstance(node, ast.Call):
        if not isinstance(node.func, ast.Name):
            raise _reject(node, "calling anything but a named function")
        name = node.func.id
        if name not in _ARITY:
            raise ExpressionError(
                f"unknown function {name!r}; available: {', '.join(sorted(_ARITY))}"
            )
        if node.keywords or any(isinstance(a, ast.Starred) for a in node.args):
            raise _reject(node, "keyword or starred arguments")
        lo, hi = _ARITY[name]
        if len(node.args) < lo or (hi is not None and len(node.args) > hi):
            wanted = f"{lo}" if lo == hi else (f"at least {lo}" if hi is None else f"{lo} to {hi}")
            raise ExpressionError(f"{name}() takes {wanted} argument(s), got {len(node.args)}")
        for arg in node.args:
            _validate(arg, names)
    else:
        kinds = {
            ast.Attribute: "attribute access (a.b)",
            ast.Subscript: "subscripting (a[b])",
            ast.Lambda: "lambda",
            ast.IfExp: "if-else (use where(cond, a, b))",
            ast.NamedExpr: "assignment",
        }
        what = kinds.get(type(node), type(node).__name__)
        raise _reject(node, what)


def parse(source: str, names: Iterable[str] | None = None) -> Expression:
    """Parse and validate an expression.

    Parameters
    ----------
    source : the expression text.
    names : optional collection of the variable names that exist; when given,
        an expression that uses any other name is rejected here rather than
        at evaluation.

    Raises :class:`ExpressionError` (a ``ValueError``) for anything outside
    the language described in the module docstring.
    """
    if isinstance(source, Expression):
        expression = source
    else:
        if not isinstance(source, str):
            raise ExpressionError("an expression must be a string")
        if len(source) > MAX_LENGTH:
            raise ExpressionError(f"expression is longer than {MAX_LENGTH} characters")
        if not source.strip():
            raise ExpressionError("the expression is empty")
        if "\n" in source or "\r" in source or "\x00" in source:
            raise ExpressionError("an expression must be a single line")
        text = _normalise(source.strip())
        try:
            tree = ast.parse(text, mode="eval")
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:
            detail = exc.msg if isinstance(exc, SyntaxError) else type(exc).__name__
            raise ExpressionError(f"cannot parse expression {source!r}: {detail}") from None
        _check_size(tree, source)
        used: set[str] = set()
        _validate(tree, used)
        expression = Expression(source, tree, frozenset(used))
    if names is not None:
        unknown = sorted(expression.names - set(names))
        if unknown:
            raise ExpressionError(
                f"unknown name(s) {', '.join(map(repr, unknown))} in expression "
                f"{expression.source!r}"
            )
    return expression


class _Evaluator:
    def __init__(self, lookup: Callable[[str], Any]):
        self.lookup = lookup
        self.n_invalid = 0
        self.n_invalid_scalar = 0  # invalid results of constant sub-expressions
        self.cache: dict[str, Any] = {}

    # -- operand coercion
    def number(self, value: Any, what: str) -> Any:
        if isinstance(value, CategoryValues):
            raise ExpressionError(
                f"a category attribute cannot be used in {what}; compare it with a string "
                "using == or !="
            )
        if isinstance(value, str):
            raise ExpressionError(f"a string cannot be used in {what}")
        return np.asarray(value).astype(np.float64, copy=False)

    def truth(self, value: Any, what: str) -> Any:
        if _is_bool(value):
            return np.asarray(value)
        v = self.number(value, what)
        return (v != 0.0) & ~np.isnan(v)

    def count_invalid(self, out: Any, *inputs: Any) -> None:
        bad = ~np.isfinite(out)
        if bad.any():
            for value in inputs:
                bad = bad & np.isfinite(value)
            if np.ndim(out) == 0:
                self.n_invalid_scalar += int(bad)
            else:
                self.n_invalid += int(np.count_nonzero(bad))

    # -- nodes
    def visit(self, node: ast.AST) -> Any:
        if isinstance(node, ast.Constant):
            v = node.value
            if isinstance(v, bool | str):
                return v
            try:
                return np.float64(v)
            except OverflowError:  # an integer literal beyond the float range
                return np.float64(np.inf if v > 0 else -np.inf)
        if isinstance(node, ast.Name):
            if node.id in _CONSTANTS:
                return _CONSTANTS[node.id]
            if node.id not in self.cache:
                self.cache[node.id] = self.lookup(node.id)
            return self.cache[node.id]
        if isinstance(node, ast.BinOp):
            symbol = type(node.op).__name__
            a = self.number(self.visit(node.left), f"arithmetic ({symbol})")
            b = self.number(self.visit(node.right), f"arithmetic ({symbol})")
            out = _BINARY[type(node.op)](a, b)
            self.count_invalid(out, a, b)
            return out
        if isinstance(node, ast.UnaryOp):
            value = self.visit(node.operand)
            if isinstance(node.op, ast.Not):
                return ~self.truth(value, "not")
            v = self.number(value, "arithmetic")
            return -v if isinstance(node.op, ast.USub) else v
        if isinstance(node, ast.BoolOp):
            word = "and" if isinstance(node.op, ast.And) else "or"
            combine = np.logical_and if word == "and" else np.logical_or
            out = self.truth(self.visit(node.values[0]), word)
            for value in node.values[1:]:
                out = combine(out, self.truth(self.visit(value), word))
            return out
        if isinstance(node, ast.Compare):
            operands = [self.visit(n) for n in (node.left, *node.comparators)]
            out = None
            for op, a, b in zip(node.ops, operands[:-1], operands[1:], strict=True):
                part = self.compare(op, a, b)
                out = part if out is None else np.logical_and(out, part)
            return out
        if isinstance(node, ast.Call):
            return self.call(node.func.id, [self.visit(a) for a in node.args])
        raise ExpressionError(f"{type(node).__name__} is not allowed in an expression")

    def compare(self, op: ast.cmpop, a: Any, b: Any) -> Any:
        symbol = _COMPARE_SYMBOL[type(op)]
        a_special = isinstance(a, CategoryValues | str)
        b_special = isinstance(b, CategoryValues | str)
        if a_special or b_special:
            if isinstance(a, str) and isinstance(b, CategoryValues):
                a, b = b, a
            if not (isinstance(a, CategoryValues) and isinstance(b, str)):
                raise ExpressionError(
                    f"cannot compare these operands with {symbol}: a category attribute is "
                    "compared with a quoted string, numbers with numbers"
                )
            if not isinstance(op, ast.Eq | ast.NotEq):
                raise ExpressionError("strings can only be compared with == or !=")
            # -2 is never stored, so an unknown string matches nothing
            code = a.categories.index(b) if b in a.categories else -2
            return _COMPARE[type(op)](a.codes, code)
        with np.errstate(invalid="ignore"):
            return _COMPARE[type(op)](
                self.number(a, f"a comparison ({symbol})"),
                self.number(b, f"a comparison ({symbol})"),
            )

    def call(self, name: str, args: list[Any]) -> Any:
        if name == "isnan":
            if isinstance(args[0], CategoryValues):
                return args[0].codes < 0
            return np.isnan(self.number(args[0], "isnan()"))
        if name in ("where", "iif"):
            cond = self.truth(args[0], f"{name}()")
            return np.where(cond, self.number(args[1], f"{name}()"),
                            self.number(args[2], f"{name}()"))  # fmt: skip
        values = [self.number(a, f"{name}()") for a in args]
        if name in ("min", "max"):
            func = np.minimum if name == "min" else np.maximum
            out = values[0]
            for v in values[1:]:
                out = func(out, v)
            return out
        if name == "round":
            digits = 0
            if len(values) == 2:
                if np.ndim(values[1]) != 0 or values[1] != np.rint(values[1]):
                    raise ExpressionError("round(a, digits): digits must be a whole constant")
                digits = int(values[1])
            return np.round(values[0], digits)
        out = _UNARY_MATH[name](values[0])
        self.count_invalid(out, values[0])
        return out


def _is_bool(value: Any) -> bool:
    return isinstance(value, bool | np.bool_) or (
        isinstance(value, np.ndarray) and value.dtype == bool
    )


@dataclass(frozen=True, eq=False)
class Expression:
    """A validated expression. Create with :func:`parse`.

    Attributes
    ----------
    source : the text as given.
    names : frozenset of the variable names it uses (attributes and built-in
        variables; functions and constants excluded).
    """

    source: str
    _tree: ast.Expression
    names: frozenset[str]

    def __repr__(self) -> str:
        return f"Expression({self.source!r})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Expression) and other.source == self.source

    def __hash__(self) -> int:
        return hash(self.source)

    def evaluate(self, variables: Mapping[str, Any] | Callable[[str], Any]) -> Evaluation:
        """Evaluate with numpy.

        Parameters
        ----------
        variables : mapping (or function) from name to value: a number, a
            numpy array (all arrays of the same length), or a
            :class:`CategoryValues` for a category attribute. Only the names
            in :attr:`names` are looked up, each once.

        Returns
        -------
        :class:`Evaluation` ``(values, n_invalid)``. Unknown names raise
        :class:`ExpressionError`.
        """
        if callable(variables):
            lookup = variables
        else:

            def lookup(name: str) -> Any:
                try:
                    return variables[name]
                except KeyError:
                    known = ", ".join(sorted(map(str, variables))) or "(none)"
                    raise ExpressionError(
                        f"unknown name {name!r} in expression {self.source!r}; available: {known}"
                    ) from None

        evaluator = _Evaluator(lookup)
        with np.errstate(all="ignore"):
            values = evaluator.visit(self._tree.body)
        if isinstance(values, CategoryValues | str):
            raise ExpressionError(
                f"expression {self.source!r} yields a category, not a number or a condition"
            )
        # a constant sub-expression that is invalid is invalid for every element
        n_invalid = evaluator.n_invalid + evaluator.n_invalid_scalar * int(np.size(values))
        return Evaluation(values, n_invalid)
