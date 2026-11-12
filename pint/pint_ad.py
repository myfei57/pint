"""
pint.pint_ad
~~~~~~~~~~~~

First-order forward-mode automatic differentiation of algebraic expressions.

The module reuses the restricted arithmetic parse tree built by
:mod:`pint.pint_eval` (``EvalTreeNode`` / ``build_eval_tree``) but never calls
Python's ``eval`` and never executes user supplied callables. A single
post-order traversal propagates, for every subexpression, both its value
(primal) and its differential (tangent) using the chain rule.

Every tangent is a dimensional :class:`~pint.Quantity`: the differentiation
variable is seeded with a unit increment in its own coordinate units while
every other variable, unit and constant is seeded with a *zero quantity that
keeps its own units*. Bare ``0`` is therefore never used to hide incompatible
terms: adding two tangents with mismatched dimensionalities raises the usual
:class:`~pint.errors.DimensionalityError`.

Only the restricted algebraic grammar is supported: parentheses, unary
``+``/``-``, binary ``+``/``-``/``*``/``/``, implicit multiplication and
constant dimensionless powers. Uncertainty (``+/-``), modulo (``%``), floor
division (``//``), variable or dimensional exponents and any non algebraic
construct are rejected.

:copyright: 2024 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import numbers
from decimal import Decimal
from fractions import Fraction
from tokenize import TokenInfo
from typing import TYPE_CHECKING, Callable, NamedTuple

from .compat import is_duck_array
from .errors import ExpressionEvaluationError
from .pint_eval import EvalTreeNode

if TYPE_CHECKING:
    from .facets.plain.quantity import PlainQuantity

#: Binary operators accepted by the first order differentiator.
_SUPPORTED_BINARY_OPS = frozenset({"", "*", "/", "+", "-", "**", "^"})


class DifferentiationResult(NamedTuple):
    """Result of :meth:`~pint.UnitRegistry.differentiate`.

    Attributes
    ----------
    value
        Quantity obtained by evaluating the expression at the bindings.
    derivative
        First derivative with respect to the differentiation variable, with
        units equivalent to ``value.units / variable.units``.
    """

    value: PlainQuantity
    derivative: PlainQuantity


class _Dual:
    """Primal/tangent pair carried through the single tree traversal.

    ``tangent`` always has the same dimensionality as ``value``: it represents
    the differential of the subexpression, not a bare derivative coefficient.
    ``has_variable`` tracks whether a bound variable appears in the subtree,
    which is used to require constant exponents in power operations.
    """

    __slots__ = ("value", "tangent", "has_variable")

    def __init__(
        self,
        value: PlainQuantity,
        tangent: PlainQuantity,
        has_variable: bool,
    ) -> None:
        self.value = value
        self.tangent = tangent
        self.has_variable = has_variable


def _token_location(
    token: TokenInfo, expression: str
) -> tuple[str, int | None, int | None, str | None]:
    """Return ``(expression, lineno, column, near)`` for an error message.

    ``tokenize`` reports 1-based lines and 0-based columns; columns are
    converted to 1-based for human readable messages.
    """
    lineno = token.start[0]
    col = token.start[1] + 1
    return expression, lineno, col, token.string


def _domain_error(
    msg: str, token: TokenInfo, expression: str
) -> ExpressionEvaluationError:
    return ExpressionEvaluationError(msg, *_token_location(token, expression))


def _zero_like(quantity: PlainQuantity) -> PlainQuantity:
    """Return a zero quantity sharing the operand's registry and units."""
    return quantity.__class__(0, quantity.units)


def _scalar_magnitude_is_zero(quantity: PlainQuantity) -> bool:
    """True for a scalar zero magnitude (never for array magnitudes)."""
    magnitude = quantity.magnitude
    if is_duck_array(magnitude):
        return False
    return magnitude == 0


def _is_integral_number(number: object) -> bool:
    """Whether a real numeric magnitude represents an integer value."""
    if isinstance(number, numbers.Integral):
        return True
    if isinstance(number, Fraction):
        return number.denominator == 1
    if isinstance(number, float):
        return number.is_integer()
    if isinstance(number, Decimal):
        return number == number.to_integral_value()
    return False


def _dual_add(left: _Dual, right: _Dual, has_variable: bool) -> _Dual:
    # Tangents share the dimensionality of the operands; pint rejects any
    # incompatible addition instead of silently accepting a bare zero.
    return _Dual(
        left.value + right.value,
        left.tangent + right.tangent,
        has_variable,
    )


def _dual_sub(left: _Dual, right: _Dual, has_variable: bool) -> _Dual:
    return _Dual(
        left.value - right.value,
        left.tangent - right.tangent,
        has_variable,
    )


def _dual_mul(left: _Dual, right: _Dual, has_variable: bool) -> _Dual:
    # d(a * b) = da * b + a * db
    return _Dual(
        left.value * right.value,
        left.tangent * right.value + left.value * right.tangent,
        has_variable,
    )


def _dual_truediv(
    left: _Dual, right: _Dual, operator_token: TokenInfo, expression: str
) -> _Dual:
    # Detect the domain error explicitly: a float denominator equal to zero
    # would otherwise raise ZeroDivisionError while some magnitude types could
    # propagate inf/nan. The whole evaluation aborts (no partial result).
    if _scalar_magnitude_is_zero(right.value):
        raise _domain_error("Division by zero", operator_token, expression)
    has_variable = left.has_variable or right.has_variable
    # d(a / b) = (da * b - a * db) / b ** 2
    denominator = right.value**2
    return _Dual(
        left.value / right.value,
        (left.tangent * right.value - left.value * right.tangent) / denominator,
        has_variable,
    )


def _dual_pow(
    base: _Dual,
    exponent: _Dual,
    operator_token: TokenInfo,
    expression: str,
) -> _Dual:
    """Power rule for a constant, dimensionless, real numeric exponent."""
    if exponent.has_variable:
        raise _domain_error(
            "Only constant exponents are supported in automatic differentiation",
            operator_token,
            expression,
        )
    if not exponent.value.dimensionless:
        raise _domain_error(
            "The exponent must be dimensionless", operator_token, expression
        )

    power = exponent.value.magnitude
    if (
        isinstance(power, bool)
        or not isinstance(power, numbers.Real)
        or is_duck_array(power)
    ):
        raise _domain_error(
            "The exponent must be a real numeric constant",
            operator_token,
            expression,
        )

    base_magnitude = base.value.magnitude

    # Domain validation happens before any arithmetic so that a fractional
    # power of a negative value cannot silently produce a complex magnitude.
    if _scalar_magnitude_is_zero(base.value) and power < 0:
        raise _domain_error(
            "Zero cannot be raised to a negative power",
            operator_token,
            expression,
        )
    if base_magnitude < 0 and not _is_integral_number(power):
        raise _domain_error(
            "Fractional power of a negative value is not defined",
            operator_token,
            expression,
        )

    value = base.value**power

    if power == 0:
        # d(a ** 0) = 0. Returning directly avoids evaluating ``a ** -1`` at
        # ``a == 0`` (and any other unnecessary reciprocal).
        return _Dual(value, _zero_like(value), base.has_variable)

    if power == 1:
        # d(a ** 1) = da; no extra factor is needed.
        return _Dual(value, base.tangent, base.has_variable)

    # d(a ** n) = n * a ** (n - 1) * da, with n constant.
    if _scalar_magnitude_is_zero(base.value) and (power - 1) < 0:
        # The primal is defined (e.g. ``0 ** 0.5 == 0``) but the derivative
        # diverges through a negative power of zero.
        raise _domain_error(
            "The derivative of a fractional power is not defined at zero",
            operator_token,
            expression,
        )
    tangent = power * (base.value ** (power - 1)) * base.tangent
    return _Dual(value, tangent, base.has_variable)


def evaluate_dual_tree(
    node: EvalTreeNode,
    resolve_leaf: Callable[[TokenInfo], _Dual],
    expression: str,
) -> _Dual:
    """Walk an :class:`EvalTreeNode` once, propagating value and tangent.

    Parameters
    ----------
    node
        Root of the restricted arithmetic parse tree produced by
        :func:`~pint.pint_eval.build_eval_tree`.
    resolve_leaf
        Callable mapping a leaf :class:`~tokenize.TokenInfo` to its seeded
        :class:`_Dual`. The registry owns name resolution and tangent seeding.
    expression
        Tokenized expression text, embedded in domain error messages.

    Raises
    ------
    ExpressionEvaluationError
        For unsupported operators or value-domain errors. The error carries
        the offending token position and aborts the whole traversal.
    """
    if node.right is not None:
        # Binary node (an implicit multiplication has ``operator is None``).
        assert isinstance(node.left, EvalTreeNode)
        left = evaluate_dual_tree(node.left, resolve_leaf, expression)
        right = evaluate_dual_tree(node.right, resolve_leaf, expression)
        operator_token = node.operator
        op_text = operator_token.string if operator_token is not None else ""

        if op_text not in _SUPPORTED_BINARY_OPS:
            raise _domain_error(
                f"Operator '{op_text}' is not supported by automatic differentiation",
                operator_token,
                expression,
            )

        has_variable = left.has_variable or right.has_variable

        if op_text in ("", "*"):
            return _dual_mul(left, right, has_variable)
        if op_text == "/":
            return _dual_truediv(left, right, operator_token, expression)
        if op_text == "+":
            return _dual_add(left, right, has_variable)
        if op_text == "-":
            return _dual_sub(left, right, has_variable)
        # ``^`` is normalized to ``**`` by the string preprocessor but both
        # spellings are accepted defensively.
        return _dual_pow(left, right, operator_token, expression)

    if node.operator is not None:
        # Unary node.
        assert isinstance(node.left, EvalTreeNode)
        inner = evaluate_dual_tree(node.left, resolve_leaf, expression)
        op_token = node.operator
        if op_token.string == "-":
            return _Dual(-inner.value, -inner.tangent, inner.has_variable)
        if op_token.string == "+":
            return _Dual(+inner.value, +inner.tangent, inner.has_variable)
        raise _domain_error(
            f"Unary operator '{op_token.string}' is not supported by "
            "automatic differentiation",
            op_token,
            expression,
        )

    # Leaf node: the registry seeded the primal and the tangent.
    assert isinstance(node.left, TokenInfo)
    return resolve_leaf(node.left)


def collect_leaf_tokens(node: EvalTreeNode) -> list[TokenInfo]:
    """Return all leaf tokens of an evaluation tree, left to right."""
    leaves: list[TokenInfo] = []

    def _walk(current: EvalTreeNode) -> None:
        if current.right is not None:
            assert isinstance(current.left, EvalTreeNode)
            _walk(current.left)
            _walk(current.right)
        elif current.operator is not None:
            assert isinstance(current.left, EvalTreeNode)
            _walk(current.left)
        else:
            assert isinstance(current.left, TokenInfo)
            leaves.append(current.left)

    _walk(node)
    return leaves
