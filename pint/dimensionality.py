"""
pint.dimensionality
~~~~~~~~~~~~~~~~~~~

Symbolic dimensional constraint solving.

Given labelled algebraic equations relating named symbols, the known
dimensionality (or units) of some of them, and the names of the symbols whose
dimensionality should be inferred, this module builds and solves an *exact*
rational linear system over the exponents of a fixed set of base dimensions.

Design constraints enforced here:

* Numeric literals are always dimensionless; no unit is ever attached to a
  number automatically.
* Only dimensionalities are inferred, never concrete units (``meter`` versus
  ``centimeter``) or scaling factors.
* Expressions are parsed with a dedicated restricted grammar (parentheses,
  unary ``+``/``-``, ``+``, ``-``, ``*``, ``/`` and rational constant powers).
  Neither :func:`eval` nor any function call or attribute access is used.
* Every coefficient and exponent is an exact :class:`fractions.Fraction`.
  Consistency is decided with exact zero tests, never with floating point
  tolerances.
* A symbol appearing several times always shares the same unknowns: each
  occurrence contributes to the same columns of the linear system instead of
  being inferred independently.
* The space of unknown dimensionalities is restricted to the base dimensions
  found in the supplied known dimensionalities and to the additional base
  dimensions explicitly declared by the caller. No new physical dimension can
  appear in a solution.

:copyright: 2024 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import re
from collections.abc import Hashable, Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal
from fractions import Fraction
from typing import TYPE_CHECKING, Any, Literal

from .errors import (
    DefinitionSyntaxError,
    UndefinedUnitError,
    is_dim,
    is_valid_dimension_name,
)
from .util import UnitsContainer

if TYPE_CHECKING:
    from .registry import UnitRegistry

#: Maximum nesting depth accepted by the expression parser. It stays well
#: below CPython's recursion limit while being far above any legitimate
#: dimensional equation.
_MAX_PARSE_DEPTH = 100

#: Statuses reported by :class:`DimensionalitySolution`.
_UNIQUE = "unique"
_UNDERDETERMINED = "underdetermined"
_INCONSISTENT = "inconsistent"

_TOKEN_RE = re.compile(
    r"""
      (?P<NUMBER>\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)
    | (?P<NAME>[^\W\d]\w*)
    | (?P<POW>\*\*)
    | (?P<OP>[()=+\-*/\[\]])
    | (?P<SKIP>\s+)
    """,
    re.VERBOSE | re.UNICODE,
)


# ---------------------------------------------------------------------------
# Tokenizer and parser
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Token:
    """A single lexical token.

    ``kind`` is one of ``"NUMBER"``, ``"NAME"``, ``"POW"``, ``"OP"`` or
    ``"END"``; ``value`` holds the raw source text and ``pos`` its offset.
    """

    kind: str
    value: str
    pos: int


def _tokenize(text: str) -> list[_Token]:
    """Split *text* into tokens using the restricted grammar.

    Any character outside numbers, identifier names, whitespace, the
    operators ``( ) [ ] = + - * / **`` (notably ``.``, ``,`` and ``@``)
    raises :class:`DefinitionSyntaxError`, which also blocks attribute access
    and every construct outside the supported grammar.
    """
    tokens: list[_Token] = []
    pos = 0
    size = len(text)
    while pos < size:
        match = _TOKEN_RE.match(text, pos)
        if match is None:
            raise DefinitionSyntaxError(
                f"unexpected character {text[pos]!r} at position {pos} "
                f"in expression {text!r}"
            )
        kind = match.lastgroup
        assert kind is not None
        value = match.group()
        pos = match.end()
        if kind != "SKIP":
            tokens.append(_Token(kind, value, match.start()))
    tokens.append(_Token("END", "", pos))
    return tokens


class _Parser:
    """Recursive descent parser producing a small tuple-based AST.

    AST nodes are one of::

        ("num", Fraction)
        ("name", str)
        ("neg", node)
        ("add" | "sub" | "mul" | "div", left, right)
        ("pow", base_node, Fraction)
    """

    def __init__(self, text: str) -> None:
        self.text = text
        self.tokens = _tokenize(text)
        self.index = 0
        self.depth = 0

    # -- helpers -----------------------------------------------------------

    @property
    def current(self) -> _Token:
        return self.tokens[self.index]

    def _advance(self) -> _Token:
        token = self.tokens[self.index]
        self.index += 1
        return token

    def _enter(self) -> None:
        self.depth += 1
        if self.depth > _MAX_PARSE_DEPTH:
            raise DefinitionSyntaxError(
                f"maximum nesting depth ({_MAX_PARSE_DEPTH}) exceeded "
                f"in expression {self.text!r}"
            )

    def _exit(self) -> None:
        self.depth -= 1

    def _expect_op(self, op: str) -> None:
        token = self.current
        if token.kind != "OP" or token.value != op:
            raise DefinitionSyntaxError(
                f"expected {op!r} at position {token.pos} in expression "
                f"{self.text!r}, found {token.value!r}"
            )
        self._advance()

    def _expect_end(self) -> None:
        token = self.current
        if token.kind != "END":
            raise DefinitionSyntaxError(
                f"unexpected token {token.value!r} at position {token.pos} "
                f"in expression {self.text!r}"
            )

    # -- entry point -------------------------------------------------------

    def parse_equation(self) -> tuple[str, Any, Any]:
        left = self.parse_expr()
        self._expect_op("=")
        right = self.parse_expr()
        self._expect_end()
        return ("eq", left, right)

    # -- expression grammar ------------------------------------------------

    def parse_expr(self) -> Any:
        self._enter()
        try:
            node = self.parse_term()
            while self.current.kind == "OP" and self.current.value in ("+", "-"):
                op = self._advance().value
                right = self.parse_term()
                node = ("add" if op == "+" else "sub", node, right)
            return node
        finally:
            self._exit()

    def parse_term(self) -> Any:
        self._enter()
        try:
            node = self.parse_factor()
            while self.current.kind == "OP" and self.current.value in ("*", "/"):
                op = self._advance().value
                right = self.parse_factor()
                node = ("mul" if op == "*" else "div", node, right)
            return node
        finally:
            self._exit()

    def parse_factor(self) -> Any:
        self._enter()
        try:
            token = self.current
            if token.kind == "OP" and token.value in ("+", "-"):
                self._advance()
                operand = self.parse_factor()
                if token.value == "-":
                    return ("neg", operand)
                return operand
            return self.parse_power()
        finally:
            self._exit()

    def parse_power(self) -> Any:
        base = self.parse_atom()
        if self.current.kind == "POW":
            self._advance()
            exponent = self.parse_power_exponent()
            return ("pow", base, exponent)
        return base

    def parse_power_exponent(self) -> Fraction:
        """Parse a bounded rational constant exponent.

        Only a signed numeric literal (``x ** -2``, ``x ** 0.5``) or a
        parenthesized constant expression (``x ** (2 / 3)``) is accepted, so
        the exponent cannot swallow following additive terms such as the
        ``+ 2 * a * s`` part of ``u ** 2 + 2 * a * s``.
        """
        self._enter()
        try:
            sign = 1
            token = self.current
            if token.kind == "OP" and token.value in ("+", "-"):
                self._advance()
                sign = -1 if token.value == "-" else 1
                token = self.current
            if token.kind == "NUMBER":
                self._advance()
                return sign * Fraction(token.value)
            if token.kind == "OP" and token.value == "(":
                self._advance()
                value = self.parse_constant_expr()
                self._expect_op(")")
                return sign * value
            raise DefinitionSyntaxError(
                f"power exponent must be a signed rational number or a "
                f"parenthesized rational constant expression at position "
                f"{token.pos} in expression {self.text!r}, found "
                f"{token.value!r}"
            )
        finally:
            self._exit()

    def parse_atom(self) -> Any:
        token = self.current
        if token.kind == "NUMBER":
            self._advance()
            # Fraction parses decimal and exponent strings exactly.
            return ("num", Fraction(token.value))
        if token.kind == "NAME":
            # A name immediately followed by "(" would be a function call.
            if not token.value.isidentifier():
                raise DefinitionSyntaxError(
                    f"{token.value!r} is not a valid symbol name "
                    f"at position {token.pos} in expression {self.text!r}"
                )
            self._advance()
            if self.current.kind == "OP" and self.current.value == "(":
                raise DefinitionSyntaxError(
                    f"function calls are not allowed in dimensional "
                    f"expressions: {token.value!r}(...) at position "
                    f"{token.pos} in expression {self.text!r}"
                )
            return ("name", token.value)
        if token.kind == "OP" and token.value == "(":
            self._advance()
            node = self.parse_expr()
            self._expect_op(")")
            self._reject_implicit_multiplication()
            return node
        if token.kind == "OP" and token.value == "[":
            # Bracket atoms are dimension names such as ``[length]`` (or the
            # dimensionless marker ``[]``), resolved through the registry like
            # any other undeclared name.
            self._advance()
            inner = self.current
            if inner.kind == "OP" and inner.value == "]":
                self._advance()
                dim_name = "[]"
            else:
                if inner.kind != "NAME":
                    raise DefinitionSyntaxError(
                        f"expected a dimension name inside '[' at position "
                        f"{inner.pos} in expression {self.text!r}, found "
                        f"{inner.value!r}"
                    )
                self._advance()
                self._expect_op("]")
                dim_name = f"[{inner.value}]"
                if not is_valid_dimension_name(dim_name):
                    raise DefinitionSyntaxError(
                        f"invalid dimension name {dim_name!r} in expression "
                        f"{self.text!r}"
                    )
            self._reject_implicit_multiplication()
            return ("name", dim_name)
        raise DefinitionSyntaxError(
            f"expected a number, symbol or parenthesized expression at "
            f"position {token.pos} in expression {self.text!r}, "
            f"found {token.value!r}"
        )

    def _reject_implicit_multiplication(self) -> None:
        # Implicit multiplication such as ``a (b)`` is not part of the
        # grammar: every product must use an explicit ``*``.
        if (
            self.current.kind in ("NAME", "NUMBER")
            or self.current.kind == "OP"
            and self.current.value in ("(", "[")
        ):
            raise DefinitionSyntaxError(
                f"implicit multiplication is not allowed; use '*' "
                f"explicitly at position {self.current.pos} in "
                f"expression {self.text!r}"
            )

    # -- constant-only grammar used for power exponents --------------------

    def parse_constant_expr(self) -> Fraction:
        self._enter()
        try:
            value = self._constant_term()
            while self.current.kind == "OP" and self.current.value in ("+", "-"):
                op = self._advance().value
                right = self._constant_term()
                value = value + right if op == "+" else value - right
            return value
        finally:
            self._exit()

    def _constant_term(self) -> Fraction:
        value = self._constant_factor()
        while self.current.kind == "OP" and self.current.value in ("*", "/"):
            op = self._advance().value
            right = self._constant_factor()
            value = value * right if op == "*" else value / right
        return value

    def _constant_factor(self) -> Fraction:
        token = self.current
        if token.kind == "OP" and token.value in ("+", "-"):
            self._advance()
            operand = self._constant_factor()
            return operand if token.value == "+" else -operand
        return self._constant_power()

    def _constant_power(self) -> Fraction:
        value = self._constant_atom()
        if self.current.kind == "POW":
            self._advance()
            # Right associative, like Python: 2 ** 3 ** 2 == 2 ** (3 ** 2).
            exponent = self._constant_factor()
            return value**exponent
        return value

    def _constant_atom(self) -> Fraction:
        token = self.current
        if token.kind == "NUMBER":
            self._advance()
            return Fraction(token.value)
        if token.kind == "OP" and token.value == "(":
            self._advance()
            value = self.parse_constant_expr()
            self._expect_op(")")
            return value
        if token.kind == "NAME":
            raise DefinitionSyntaxError(
                f"power exponents must be rational constants, symbol "
                f"{token.value!r} is not allowed at position {token.pos} "
                f"in expression {self.text!r}"
            )
        raise DefinitionSyntaxError(
            f"expected a rational constant exponent at position "
            f"{token.pos} in expression {self.text!r}, found "
            f"{token.value!r}"
        )


# ---------------------------------------------------------------------------
# Exact linear algebra
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Lin:
    """An affine linear expression :class:`Fraction` * const + sum(terms)."""

    terms: Mapping[int, Fraction] = field(default_factory=dict)
    const: Fraction = Fraction(0)


def _lin_zero() -> _Lin:
    return _Lin()


def _lin_variable(column: int) -> _Lin:
    return _Lin({column: Fraction(1)})


def _lin_add(left: _Lin, right: _Lin) -> _Lin:
    terms = dict(left.terms)
    for column, coefficient in right.terms.items():
        new = terms.get(column, Fraction(0)) + coefficient
        if new == 0:
            terms.pop(column, None)
        else:
            terms[column] = new
    return _Lin(terms, left.const + right.const)


def _lin_sub(left: _Lin, right: _Lin) -> _Lin:
    terms = dict(left.terms)
    for column, coefficient in right.terms.items():
        new = terms.get(column, Fraction(0)) - coefficient
        if new == 0:
            terms.pop(column, None)
        else:
            terms[column] = new
    return _Lin(terms, left.const - right.const)


def _lin_scale(expr: _Lin, factor: Fraction) -> _Lin:
    if factor == 0:
        return _Lin()
    return _Lin(
        {column: coefficient * factor for column, coefficient in expr.terms.items()},
        expr.const * factor,
    )


def _to_fraction(value: Any) -> Fraction:
    """Convert an exponent coming from a container to an exact Fraction."""
    if isinstance(value, Fraction):
        return value
    if isinstance(value, int):
        return Fraction(value)
    if isinstance(value, Decimal):
        return Fraction(value)
    if isinstance(value, float):
        # Keep the exact value held by the registry; the solver itself never
        # applies any tolerance.
        return Fraction(value)
    raise TypeError(
        f"dimensionality exponents must be numeric, found {type(value)!r}: {value!r}"
    )


def _container_to_fraction_map(
    container: Mapping[str, Any],
) -> dict[str, Fraction]:
    return {
        name: _to_fraction(value) for name, value in container.items() if value != 0
    }


def _rref(matrix: list[list[Fraction]], provenance: list[set[Hashable]]) -> list[int]:
    """Bring the augmented *matrix* to reduced row echelon form in place.

    All arithmetic is exact :class:`Fraction` arithmetic and pivot tests use
    exact comparisons. *provenance* tracks the source equation labels of each
    row and is transformed alongside the matrix; it is returned implicitly
    through the mutated list.

    Returns the ordered list of pivot (variable) columns.
    """
    if not matrix:
        return []
    n_rows = len(matrix)
    n_variables = len(matrix[0]) - 1
    pivot_columns: list[int] = []
    row = 0
    for column in range(n_variables):
        pivot = next(
            (
                candidate
                for candidate in range(row, n_rows)
                if matrix[candidate][column] != 0
            ),
            None,
        )
        if pivot is None:
            continue
        matrix[row], matrix[pivot] = matrix[pivot], matrix[row]
        provenance[row], provenance[pivot] = provenance[pivot], provenance[row]

        pivot_value = matrix[row][column]
        matrix[row] = [entry / pivot_value for entry in matrix[row]]

        for other in range(n_rows):
            if other == row:
                continue
            factor = matrix[other][column]
            if factor == 0:
                continue
            matrix[other] = [
                entry - factor * pivot_entry
                for entry, pivot_entry in zip(matrix[other], matrix[row])
            ]
            provenance[other] = provenance[other] | provenance[row]

        pivot_columns.append(column)
        row += 1
        if row == n_rows:
            break
    return pivot_columns


# ---------------------------------------------------------------------------
# Public result object
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DimensionalitySolution:
    """Result of :func:`solve_dimensionalities`.

    Attributes
    ----------
    status
        ``"unique"`` when every unknown dimensionality is uniquely determined,
        ``"underdetermined"`` when free degrees of freedom remain and
        ``"inconsistent"`` when the constraints contradict each other.
    base_dimensions
        Ordered tuple of the base dimensions spanning the solution space.
        Inference is restricted to these dimensions: they are the base
        dimensions of the supplied known dimensionalities plus the extra base
        dimensions explicitly declared by the caller; no undeclared physical
        dimension can ever appear in a solution.
    particular_solution
        Mapping of every unknown symbol to one valid dimensionality as a
        :class:`~pint.util.UnitsContainer` of base dimension exponents. It is
        the unique solution when ``status == "unique"`` and one particular
        solution (free variables set to zero) when underdetermined. Empty for
        inconsistent systems.
    nullspace_basis
        Basis of the homogeneous solution space as a tuple of mappings, each
        assigning one :class:`~pint.util.UnitsContainer` per unknown symbol.
        Every solution of an underdetermined system is the particular solution
        plus an exact rational linear combination of these basis vectors.
        Empty for unique and inconsistent systems.
    free_degrees
        Number of undetermined degrees of freedom, i.e. the number of vectors
        in ``nullspace_basis``.
    conflicting_equation_labels
        Labels of a set of original equations participating in a contradiction
        when ``status == "inconsistent"`` (not guaranteed minimal), otherwise
        empty.
    """

    status: Literal["unique", "underdetermined", "inconsistent"]
    base_dimensions: tuple[str, ...]
    particular_solution: dict[str, UnitsContainer]
    nullspace_basis: tuple[dict[str, UnitsContainer], ...]
    free_degrees: int
    conflicting_equation_labels: tuple[Hashable, ...]

    @property
    def is_unique(self) -> bool:
        return self.status == _UNIQUE

    @property
    def is_underdetermined(self) -> bool:
        return self.status == _UNDERDETERMINED

    @property
    def is_inconsistent(self) -> bool:
        return self.status == _INCONSISTENT

    def __repr__(self) -> str:
        if self.is_inconsistent:
            detail = f"conflicting_equation_labels={self.conflicting_equation_labels!r}"
        else:
            detail = (
                f"particular_solution={self.particular_solution!r}, "
                f"free_degrees={self.free_degrees}"
            )
        return (
            f"DimensionalitySolution(status={self.status!r}, "
            f"base_dimensions={self.base_dimensions!r}, {detail})"
        )


# ---------------------------------------------------------------------------
# Analyzer: AST -> exact linear constraints
# ---------------------------------------------------------------------------


class _Analyzer:
    def __init__(
        self,
        known: Mapping[str, Mapping[str, Fraction]],
        unknowns: tuple[str, ...],
        base_dimensions: tuple[str, ...],
        registry: UnitRegistry | None,
        label_order: Mapping[Hashable, int],
        container_factory,
    ) -> None:
        self._known = known
        self._unknowns = unknowns
        self._dims = base_dimensions
        self._registry = registry
        self._label_order = label_order
        self._make_container = container_factory

        # One variable column per (symbol, base dimension) pair. A symbol
        # appearing in several leaves therefore shares the same columns.
        self._columns: dict[tuple[str, str], int] = {}
        for symbol in unknowns:
            for dimension in base_dimensions:
                self._columns[(symbol, dimension)] = len(self._columns)

        self._dimension_index = {
            dimension: index for index, dimension in enumerate(base_dimensions)
        }
        self._name_cache: dict[str, list[_Lin] | None] = {}
        self.rows: list[dict[int, Fraction]] = []
        self.right_hand_sides: list[Fraction] = []
        self.provenance: list[set[Hashable]] = []

    # -- name resolution ---------------------------------------------------

    def _resolve_name(self, name: str) -> list[_Lin] | None:
        """Return the dimension vector of *name* or ``None`` if undeclared."""
        if name in self._name_cache:
            return self._name_cache[name]

        vector: list[_Lin] | None
        if name in self._known:
            vector = self._constant_vector(self._known[name])
        elif name in self._unknowns:
            vector = []
            for dimension in self._dims:
                column = self._columns[(name, dimension)]
                vector.append(_lin_variable(column))
        elif is_dim(name):
            # Bracket dimension names are self-describing and need no
            # registry: ``[length]`` is a unit exponent on itself, while the
            # dimensionless marker ``[]`` contributes nothing.
            vector = self._constant_vector({} if name == "[]" else {name: Fraction(1)})
        elif self._registry is not None:
            try:
                dimensionality = self._registry.get_dimensionality(name)
            except (UndefinedUnitError, DefinitionSyntaxError, KeyError):
                vector = None
            else:
                vector = self._constant_vector(
                    _container_to_fraction_map(dimensionality)
                )
        else:
            vector = None

        self._name_cache[name] = vector
        return vector

    def _constant_vector(
        self, dimension_exponents: Mapping[str, Fraction]
    ) -> list[_Lin]:
        vector: list[_Lin] = []
        for dimension in self._dims:
            exponent = dimension_exponents.get(dimension, Fraction(0))
            vector.append(_Lin(const=exponent))
        return vector

    # -- constraint collection ---------------------------------------------

    def _add_constraint(self, expr: _Lin, label: Hashable) -> None:
        """Record the homogeneous constraint ``expr == 0``."""
        if not expr.terms:
            if expr.const == 0:
                # Tautology such as ``x - x``; no information added.
                return
            # 0 == nonzero: record it as an augmented row with no variable
            # columns; RREF will flag the contradiction and keep provenance.
            self.rows.append({})
            self.right_hand_sides.append(-expr.const)
            self.provenance.append({label})
            return
        self.rows.append(dict(expr.terms))
        self.right_hand_sides.append(-expr.const)
        self.provenance.append({label})

    def analyze_equation(self, node: tuple[str, Any, Any], label: Hashable) -> None:
        _, left, right = node
        left_vector = self._walk(left, label)
        right_vector = self._walk(right, label)
        for left_lin, right_lin in zip(left_vector, right_vector):
            self._add_constraint(_lin_sub(left_lin, right_lin), label)

    def _walk(self, node: tuple, label: Hashable) -> list[_Lin]:
        kind = node[0]
        if kind == "num":
            # Numeric literals are always dimensionless.
            return [_lin_zero() for _ in self._dims]
        if kind == "name":
            vector = self._resolve_name(node[1])
            if vector is None:
                raise ValueError(
                    f"symbol {node[1]!r} is neither a declared known/unknown "
                    f"symbol nor a unit or dimension known to the registry"
                )
            return vector
        if kind == "neg":
            # Sign does not change dimensionality.
            return self._walk(node[1], label)
        if kind == "mul":
            return [
                _lin_add(left, right)
                for left, right in zip(
                    self._walk(node[1], label), self._walk(node[2], label)
                )
            ]
        if kind == "div":
            return [
                _lin_sub(left, right)
                for left, right in zip(
                    self._walk(node[1], label), self._walk(node[2], label)
                )
            ]
        if kind == "pow":
            base = self._walk(node[1], label)
            exponent: Fraction = node[2]
            return [_lin_scale(expr, exponent) for expr in base]
        if kind in ("add", "sub"):
            left_vector = self._walk(node[1], label)
            right_vector = self._walk(node[2], label)
            for left_lin, right_lin in zip(left_vector, right_vector):
                # Addends must share the same dimensionality.
                self._add_constraint(_lin_sub(left_lin, right_lin), label)
            return left_vector
        raise DefinitionSyntaxError(
            f"unsupported expression node {kind!r}"
        )  # pragma: no cover

    # -- solving and result assembly ---------------------------------------

    def solve(self) -> DimensionalitySolution:
        n_variables = len(self._columns)
        matrix: list[list[Fraction]] = []
        for coefficients, rhs in zip(self.rows, self.right_hand_sides):
            row = [Fraction(0)] * n_variables
            for column, coefficient in coefficients.items():
                row[column] = coefficient
            row.append(rhs)
            matrix.append(row)

        pivot_columns = _rref(matrix, self.provenance)
        # Pivots are recorded sequentially: RREF row i carries the pivot of
        # pivot_columns[i]. Pivot rows may additionally have nonzero entries in
        # free-variable columns, which is what makes the null space nonzero.
        pivot_rows: dict[int, int] = {
            column: row_index for row_index, column in enumerate(pivot_columns)
        }

        # A zero coefficient row with a nonzero augmented entry is a
        # contradiction; its provenance points at the offending equations.
        for row_index, row in enumerate(matrix):
            if all(entry == 0 for entry in row[:n_variables]) and row[-1] != 0:
                labels = {label for label in self.provenance[row_index]}
                ordered = tuple(
                    sorted(labels, key=lambda label: self._label_order[label])
                )
                return DimensionalitySolution(
                    _INCONSISTENT,
                    self._dims,
                    {},
                    (),
                    0,
                    ordered,
                )

        free_columns = [
            column for column in range(n_variables) if column not in pivot_columns
        ]

        particular = [Fraction(0)] * n_variables
        for column in pivot_columns:
            particular[column] = matrix[pivot_rows[column]][-1]

        basis_vectors: list[list[Fraction]] = []
        for free_column in free_columns:
            vector = [Fraction(0)] * n_variables
            vector[free_column] = Fraction(1)
            for column in pivot_columns:
                vector[column] = -matrix[pivot_rows[column]][free_column]
            basis_vectors.append(vector)

        status = _UNIQUE if not free_columns else _UNDERDETERMINED
        return DimensionalitySolution(
            status,
            self._dims,
            self._assemble(particular),
            tuple(self._assemble(vector) for vector in basis_vectors),
            len(free_columns),
            (),
        )

    def _assemble(self, vector: list[Fraction]) -> dict[str, UnitsContainer]:
        result: dict[str, UnitsContainer] = {}
        for symbol in self._unknowns:
            exponents: dict[str, Fraction | int] = {}
            for dimension in self._dims:
                value = vector[self._columns[(symbol, dimension)]]
                if value == 0:
                    continue
                # Prefer plain ints whenever the Fraction is integral so the
                # resulting containers match ordinary Pint containers.
                exponents[dimension] = (
                    value.numerator // value.denominator
                    if value.denominator == 1
                    else value
                )
            result[symbol] = self._make_container(exponents)
        return result


# ---------------------------------------------------------------------------
# Input normalization and public API
# ---------------------------------------------------------------------------


def _normalize_dimension_name(name: str) -> str:
    if is_dim(name):
        if not is_valid_dimension_name(name):
            raise ValueError(f"invalid base dimension name: {name!r}")
        return name
    if name.isidentifier():
        return f"[{name}]"
    raise ValueError(f"invalid base dimension name: {name!r}")


def _resolve_known(
    known: Mapping[str, Any], registry: UnitRegistry | None
) -> dict[str, dict[str, Fraction]]:
    resolved: dict[str, dict[str, Fraction]] = {}
    for symbol, value in known.items():
        if not isinstance(symbol, str) or not symbol.isidentifier():
            raise ValueError(f"known symbol names must be identifiers: {symbol!r}")
        if value is None:
            resolved[symbol] = {}
            continue
        if hasattr(value, "dimensionality"):
            # PlainUnit / PlainQuantity expose the base dimension container.
            resolved[symbol] = _container_to_fraction_map(value.dimensionality)
            continue
        if isinstance(value, Mapping):
            mapping = dict(value)
            if not mapping or all(is_dim(key) for key in mapping):
                resolved[symbol] = _container_to_fraction_map(mapping)
                continue
            if registry is None:
                raise ValueError(
                    f"a registry is required to resolve the units of known "
                    f"symbol {symbol!r}: {value!r}"
                )
            container = registry.UnitsContainer(mapping)
            resolved[symbol] = _container_to_fraction_map(
                registry._get_dimensionality(container)
            )
            continue
        if isinstance(value, str):
            if value == "dimensionless":
                resolved[symbol] = {}
                continue
            if registry is None:
                raise ValueError(
                    f"a registry is required to resolve the unit expression "
                    f"of known symbol {symbol!r}: {value!r}"
                )
            resolved[symbol] = _container_to_fraction_map(
                registry.get_dimensionality(value)
            )
            continue
        raise TypeError(
            f"cannot interpret known dimensionality for {symbol!r}: "
            f"{value!r} ({type(value)!r})"
        )
    return resolved


def solve_dimensionalities(
    equations: Iterable[tuple[Hashable, str]],
    known: Mapping[str, Any],
    unknowns: Iterable[str],
    registry: UnitRegistry | None = None,
    extra_base_dimensions: Iterable[str] = (),
) -> DimensionalitySolution:
    """Infer the base-dimension exponents of unknown symbols.

    Parameters
    ----------
    equations
        Labelled algebraic equations as an iterable of ``(label, expression)``
        pairs. Every expression must contain exactly one ``=`` sign and follow
        the restricted grammar: parentheses, unary ``+``/``-``, binary
        ``+``/``-``/``*``/``/`` and rational constant powers (``x ** (2 / 3)``,
        ``x ** 0.5``). Numeric literals are dimensionless, function calls and
        attribute access are rejected, and :func:`eval` is never used.
    known
        Mapping of symbol names to their known dimensionality. Values may be a
        :class:`~pint.Unit`, :class:`~pint.Quantity`, unit string, dimension
        container or mapping of base dimension names to exponents. Declared
        symbols take precedence over registry units sharing their name.
    unknowns
        Names of the symbols whose dimensionality should be inferred. All
        occurrences of one name share the same unknowns. An empty iterable is
        accepted and turns the call into a pure consistency check of the
        supplied equations and known dimensionalities.
    registry
        Optional registry used to resolve unit strings and names that are not
        declared symbols. It is never modified: no pseudo-unit is defined in
        it to perform inference.
    extra_base_dimensions
        Additional base dimensions (e.g. ``"[substance]"`` or
        ``"substance"``) allowed in the solution space.

    Returns
    -------
    DimensionalitySolution
        The exact rational solution, a particular solution with the null-space
        basis and the number of free degrees when underdetermined, or the
        labels of equations participating in a contradiction. The
        ``base_dimensions`` attribute states the boundary of the inference
        space.

    Raises
    ------
    DefinitionSyntaxError
        If an equation violates the restricted grammar.
    ValueError
        If an equation uses an undeclared name that the registry cannot
        resolve, or if an argument is otherwise invalid.
    """
    equation_list = list(equations)
    label_order: dict[Hashable, int] = {}
    parsed_equations: list[tuple[Hashable, tuple]] = []
    for label, expression in equation_list:
        if label in label_order:
            raise ValueError(f"duplicated equation label: {label!r}")
        label_order[label] = len(label_order)
        parsed_equations.append((label, _Parser(expression).parse_equation()))

    unknown_tuple = tuple(unknowns)
    if len(set(unknown_tuple)) != len(unknown_tuple):
        raise ValueError(f"unknown symbols must not be repeated: {unknown_tuple!r}")
    for symbol in unknown_tuple:
        if not isinstance(symbol, str) or not symbol.isidentifier():
            raise ValueError(f"unknown symbol names must be identifiers: {symbol!r}")

    overlap = set(known).intersection(unknown_tuple)
    if overlap:
        raise ValueError(
            f"symbols cannot be both known and unknown: {sorted(overlap)!r}"
        )

    resolved_known = _resolve_known(dict(known), registry)

    # Build the ordered base-dimension universe: caller-declared extras first,
    # then dimensions reached through known dimensionalities, then dimensions
    # reached through registry names used in the equations.
    dimension_universe: dict[str, None] = {}
    for dimension in extra_base_dimensions:
        normalized = _normalize_dimension_name(dimension)
        dimension_universe.setdefault(normalized, None)
    for exponents in resolved_known.values():
        for dimension in exponents:
            dimension_universe.setdefault(dimension, None)

    declared_names = set(resolved_known).union(unknown_tuple)

    def _collect_names(node: tuple, sink: set[str]) -> None:
        kind = node[0]
        if kind == "name":
            sink.add(node[1])
        elif kind == "num":
            return
        elif kind == "neg":
            _collect_names(node[1], sink)
        elif kind == "pow":
            _collect_names(node[1], sink)
        else:
            _collect_names(node[1], sink)
            _collect_names(node[2], sink)

    used_names: set[str] = set()
    for _, equation in parsed_equations:
        _collect_names(equation, used_names)

    undeclared_names = used_names - declared_names
    registry_less_names: list[str] = []
    for name in sorted(undeclared_names):
        if is_dim(name):
            # Self-describing bracket dimension: valid without a registry.
            if name != "[]":
                dimension_universe.setdefault(name, None)
            continue
        registry_less_names.append(name)
        if registry is None:
            continue
        try:
            dimensionality = registry.get_dimensionality(name)
        except (UndefinedUnitError, DefinitionSyntaxError, KeyError) as exc:
            raise ValueError(
                f"symbol {name!r} is neither a declared known/unknown "
                f"symbol nor a unit or dimension known to the registry"
            ) from exc
        for dimension in dimensionality:
            dimension_universe.setdefault(dimension, None)

    if registry is None and registry_less_names:
        raise ValueError(
            f"no registry provided and the following equation symbols "
            f"are not declared: {registry_less_names!r}"
        )

    base_dimensions = tuple(dimension_universe)

    # Output containers always hold exact Fraction exponents, independently of
    # the registry magnitude type (which may be float).
    def container_factory(exponents: Mapping[str, Any]) -> UnitsContainer:
        return UnitsContainer(dict(exponents), non_int_type=Fraction)

    analyzer = _Analyzer(
        resolved_known,
        unknown_tuple,
        base_dimensions,
        registry,
        label_order,
        container_factory,
    )
    for label, equation in parsed_equations:
        analyzer.analyze_equation(equation, label)
    return analyzer.solve()
