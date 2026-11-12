"""Tests for first-order forward-mode automatic differentiation.

The analytical derivatives below are computed by hand and are the source of
truth. Finite differences are used only as a lenient auxiliary cross-check,
never as the implementation strategy or the unique reference value.
"""

from __future__ import annotations

import pytest

from pint import Context, DimensionalityError, UnitRegistry
from pint.errors import ExpressionEvaluationError


@pytest.fixture
def ureg():
    return UnitRegistry()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _finite_difference_slope(ureg, expression, wrt, bindings, eps_fraction=1e-6):
    """Auxiliary, tolerant finite-difference slope (NOT a source of truth).

    Re-evaluates the expression at a slightly perturbed binding and divides the
    value change by the coordinate increment. It only sanity-checks the
    analytical automatic-differentiation result within a loose tolerance.
    """
    point = bindings[wrt]
    increment = point.__class__(point.magnitude * eps_fraction, point.units)
    perturbed = {**bindings, wrt: point + increment}
    value_0 = ureg.differentiate(expression, wrt, bindings).value
    value_1 = ureg.differentiate(expression, wrt, perturbed).value
    return (value_1 - value_0) / increment


def _bind(ureg, spec):
    """Build ``{name: Quantity(magnitude, units)}`` from plain test data.

    ``units is None`` denotes a dimensionless quantity. Keeping the parametrize
    decorators free of live registry objects lets a function-scoped registry
    fixture build every quantity.
    """
    return {
        name: ureg.Quantity(magnitude, units)
        if units is not None
        else ureg.Quantity(magnitude)
        for name, (magnitude, units) in spec.items()
    }


# ---------------------------------------------------------------------------
# Analytical correctness (hand-derived reference values)
# ---------------------------------------------------------------------------


def test_kinetic_energy_derivative(ureg):
    # E = m * v ** 2 / 2  ->  dE/dv = m * v
    mass = ureg.Quantity(2, "kg")
    velocity = ureg.Quantity(3, "m/s")
    result = ureg.differentiate("m * v**2 / 2", "v", {"m": mass, "v": velocity})

    assert result.value == ureg.Quantity(9, "kg * m**2 / s**2")
    # Analytical derivative m * v = 2 kg * 3 m/s = 6 kg m / s.
    assert result.derivative == ureg.Quantity(6, "kg * m / s")
    # The primal value matches the ordinary evaluator exactly.
    assert result.value == ureg.parse_expression("m * v**2 / 2", m=mass, v=velocity)


@pytest.mark.parametrize(
    ("expression", "wrt", "spec"),
    [
        ("m * v**2 / 2", "v", {"m": (2.0, "kg"), "v": (3.0, "m/s")}),
        ("x**3", "x", {"x": (2, "m")}),
        ("1 / x", "x", {"x": (2, "m")}),
        ("(x + 2) / (x - 1)", "x", {"x": (4, None)}),
        (
            "a * x + b",
            "x",
            {"a": (3, "Hz"), "b": (5, None), "x": (2, "s")},
        ),
    ],
)
def test_derivative_units_equal_value_units_over_variable_units(
    ureg, expression, wrt, spec
):
    bindings = _bind(ureg, spec)
    result = ureg.differentiate(expression, wrt, bindings)
    expected_dimensionality = (result.value.units / bindings[wrt].units).dimensionality
    assert result.derivative.dimensionality == expected_dimensionality


def test_repeated_variable_shares_identity(ureg):
    # Every occurrence of x is seeded with the same unit tangent.
    x = ureg.Quantity(5, "m")
    assert ureg.differentiate("x + x", "x", {"x": x}).derivative.magnitude == 2
    assert ureg.differentiate("x - x", "x", {"x": x}).derivative.magnitude == 0
    # d(x * x)/dx = 2 x = 10 m.
    product = ureg.differentiate("x * x", "x", {"x": x})
    assert product.derivative == ureg.Quantity(10, "m")


def test_multiple_variables_cancel(ureg):
    # d(x * y - y * x)/dx = y - y = 0 (still carries the units of y).
    x = ureg.Quantity(2, "m")
    y = ureg.Quantity(4, "s")
    result = ureg.differentiate("x * y - y * x", "x", {"x": x, "y": y})
    assert result.value.magnitude == 0
    assert result.derivative.magnitude == 0
    assert result.derivative.dimensionality == y.dimensionality

    # Equal coefficients cancel: d(a*x - a*x)/dx = a - a = 0, keeping the
    # units of a (the value units Hz*m divided by the x units m gives Hz).
    a = ureg.Quantity(3, "Hz")
    cancel = ureg.differentiate("a * x - a * x", "x", {"a": a, "x": x})
    assert cancel.derivative.magnitude == 0
    assert cancel.derivative.dimensionality == a.dimensionality


def test_independent_variable_keeps_dimensional_zero(ureg):
    # f does not depend on x at all: the derivative is a dimensional zero,
    # never a bare float 0 (units are s / m here).
    x = ureg.Quantity(2, "m")
    y = ureg.Quantity(3, "s")
    result = ureg.differentiate("y", "x", {"x": x, "y": y})
    assert result.value == y
    assert result.derivative.magnitude == 0
    assert result.derivative.dimensionality == (y.units / x.units).dimensionality
    # The zero is a real Quantity, not a Python float.
    assert isinstance(result.derivative, ureg.Quantity)


def test_dimensional_zero_tangent_does_not_mask_incompatibility(ureg):
    # Adding metre and second is invalid for the value and must stay invalid;
    # dimensional zero tangents must not hide the mismatch.
    with pytest.raises(DimensionalityError):
        ureg.differentiate(
            "x + y",
            "x",
            {"x": ureg.Quantity(1, "m"), "y": ureg.Quantity(1, "s")},
        )


def test_unit_bearing_zero_value_and_derivative(ureg):
    # f(x) = x ** 2 evaluated at x = 0: value 0 m**2, derivative 0 m.
    result = ureg.differentiate("x**2", "x", {"x": ureg.Quantity(0, "m")})
    assert result.value == ureg.Quantity(0, "m**2")
    assert result.derivative == ureg.Quantity(0, "m")
    assert isinstance(result.derivative, ureg.Quantity)


def test_zero_power_avoids_negative_power_at_zero(ureg):
    # f(x) = x ** 0 = 1; the derivative is zero and must never evaluate
    # x ** -1, even at x = 0 (which would otherwise raise ZeroDivisionError).
    result = ureg.differentiate("x**0", "x", {"x": ureg.Quantity(0, "m")})
    assert result.value.magnitude == 1
    assert result.value.dimensionless
    assert result.derivative.magnitude == 0
    assert result.derivative.dimensionality == (1 / ureg.m).dimensionality


def test_nested_quotient_rule(ureg):
    # Dimensionless: f = (x + 2)/(x - 1), f' = -3/(x - 1)**2.
    dimensionless = ureg.differentiate(
        "(x + 2) / (x - 1)", "x", {"x": ureg.Quantity(4)}
    )
    assert dimensionless.value.magnitude == 2
    assert dimensionless.derivative.magnitude == pytest.approx(-1 / 3)
    assert dimensionless.derivative.dimensionless

    # Dimensional: f = x/(x + 1 m), f' = 1 m / (x + 1 m)**2 = 1/9 per metre.
    dimensional = ureg.differentiate(
        "x / (x + 1 * meter)", "x", {"x": ureg.Quantity(2, "m")}
    )
    assert dimensional.derivative.magnitude == pytest.approx(1 / 9)
    assert dimensional.derivative.dimensionality == (1 / ureg.m).dimensionality


def test_negative_powers(ureg):
    x = ureg.Quantity(2, "m")
    inverse = ureg.differentiate("x**-1", "x", {"x": x})
    # d(x**-1)/dx = -x**-2 = -1/4 m**-2.
    assert inverse.derivative.magnitude == pytest.approx(-0.25)
    assert inverse.derivative.dimensionality == (1 / ureg.m**2).dimensionality

    squared_inverse = ureg.differentiate("x**-2", "x", {"x": x})
    # d(x**-2)/dx = -2 x**-3 = -2/8 = -1/4 m**-3.
    assert squared_inverse.derivative.magnitude == pytest.approx(-0.25)
    assert squared_inverse.derivative.dimensionality == (1 / ureg.m**3).dimensionality


def test_fractional_powers(ureg):
    x = ureg.Quantity(4, "m")
    for expression in ("x**0.5", "x**(1/2)"):
        result = ureg.differentiate(expression, "x", {"x": x})
        # d(sqrt(x))/dx = 1/(2 sqrt(x)) = 1/4 per sqrt(metre).
        assert result.derivative.magnitude == pytest.approx(0.25)
        assert result.derivative.dimensionality == (1 / ureg.m**0.5).dimensionality

    inverse_root = ureg.differentiate("x**-0.5", "x", {"x": x})
    # d(x**-0.5)/dx = -0.5 x**-1.5 = -0.5 / 8 = -1/16.
    assert inverse_root.derivative.magnitude == pytest.approx(-1 / 16)


def test_unary_parentheses_and_implicit_multiplication(ureg):
    x = ureg.Quantity(3, "m")
    assert ureg.differentiate("-x", "x", {"x": x}).derivative.magnitude == -1
    assert ureg.differentiate("+x", "x", {"x": x}).derivative.magnitude == 1
    assert ureg.differentiate("3 x", "x", {"x": x}).derivative.magnitude == 3
    nested = ureg.differentiate("-(x + 1) * 2", "x", {"x": ureg.Quantity(2)})
    # d(-2(x+1))/dx = -2.
    assert nested.derivative.magnitude == -2


def test_metre_centimetre_coordinate_change(ureg):
    # f(x) = x ** 2 around the same physical point expressed in m and cm.
    in_metres = ureg.differentiate("x**2", "x", {"x": ureg.Quantity(2, "m")})
    in_centimetres = ureg.differentiate("x**2", "x", {"x": ureg.Quantity(200, "cm")})

    # Numerical slope follows the coordinate unit (4 m vs 400 cm).
    assert in_metres.derivative.magnitude == 4
    assert in_centimetres.derivative.magnitude == 400
    # But the physical derivatives are identical after conversion.
    assert in_metres.derivative.to("cm") == in_centimetres.derivative
    # The values represent the same physical quantity.
    assert in_metres.value.to("cm**2") == in_centimetres.value

    # Multiplying each coordinate derivative by the SAME physical increment
    # (1 cm = 0.01 m) must predict the same physical change.
    physical_increment_cm = ureg.Quantity(1, "cm")
    physical_increment_m = physical_increment_cm.to("m")
    change_from_cm = (in_centimetres.derivative * physical_increment_cm).to("cm**2")
    change_from_m = (in_metres.derivative * physical_increment_m).to("cm**2")
    assert change_from_cm.magnitude == pytest.approx(change_from_m.magnitude)


# ---------------------------------------------------------------------------
# Finite differences: lenient auxiliary check only (not the truth)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("expression", "wrt", "spec"),
    [
        ("x**2", "x", {"x": (3, "m")}),
        ("x**3 + 2 * x", "x", {"x": (1.5, None)}),
        ("1 / x", "x", {"x": (2, "m")}),
        ("(x + 2) / (x - 1)", "x", {"x": (4, None)}),
        ("x**0.5", "x", {"x": (4, "m")}),
    ],
)
def test_finite_difference_is_only_a_lenient_auxiliary_check(
    ureg, expression, wrt, spec
):
    bindings = _bind(ureg, spec)
    analytical = ureg.differentiate(expression, wrt, bindings).derivative
    finite_difference = _finite_difference_slope(ureg, expression, wrt, bindings).to(
        analytical.units
    )
    # Loose tolerance: finite differences only corroborate the analytical result.
    assert finite_difference.magnitude == pytest.approx(analytical.magnitude, rel=1e-3)


# ---------------------------------------------------------------------------
# Domain errors: positioned and aborting (no partial result)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("expression", ["1 / 0", "x / 0"])
def test_division_by_zero_reports_position(ureg, expression):
    with pytest.raises(ExpressionEvaluationError) as exc_info:
        ureg.differentiate(expression, "x", {"x": ureg.Quantity(2, "m")})
    error = exc_info.value
    assert error.lineno is not None and error.col is not None
    assert error.near == "/"
    assert expression in str(error)


@pytest.mark.parametrize("expression", ["(-1) ** 0.5", "x ** 0.5"])
def test_negative_fractional_power_reports_position(ureg, expression):
    with pytest.raises(ExpressionEvaluationError) as exc_info:
        ureg.differentiate(expression, "x", {"x": ureg.Quantity(-2, "m")})
    error = exc_info.value
    assert error.lineno is not None and error.col is not None
    assert error.near == "**"
    assert "negative" in str(error).lower()


def test_zero_to_negative_power_reports_position(ureg):
    with pytest.raises(ExpressionEvaluationError) as exc_info:
        ureg.differentiate("0 ** -1", "x", {"x": ureg.Quantity(2, "m")})
    assert exc_info.value.near == "**"
    assert exc_info.value.lineno is not None


def test_fractional_derivative_undefined_at_zero_reports_position(ureg):
    # Primitive 0 ** 0.5 is defined (0) but the derivative diverges.
    with pytest.raises(ExpressionEvaluationError):
        ureg.differentiate("x**0.5", "x", {"x": ureg.Quantity(0, "m")})


# ---------------------------------------------------------------------------
# Pre-evaluation validation
# ---------------------------------------------------------------------------


def test_differentiation_variable_must_be_bound(ureg):
    with pytest.raises(ExpressionEvaluationError, match="not bound"):
        ureg.differentiate("x", "y", {"x": ureg.Quantity(1, "m")})


def test_unknown_symbol_fails_before_evaluation(ureg):
    with pytest.raises(ExpressionEvaluationError) as exc_info:
        ureg.differentiate("zz + 1", "x", {"x": ureg.Quantity(1)})
    assert "zz" in str(exc_info.value)
    assert exc_info.value.lineno is not None


def test_duplicate_binding_definition_rejected(ureg):
    bindings = [("x", ureg.Quantity(1, "m")), ("x", ureg.Quantity(2, "m"))]
    with pytest.raises(ExpressionEvaluationError, match="Duplicate"):
        ureg.differentiate("x", "x", bindings)


def test_offset_unit_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError, match="offset|Absolute"):
        ureg.differentiate("x", "x", {"x": ureg.Quantity(1, "degC")})
    with pytest.raises(ExpressionEvaluationError):
        ureg.differentiate("degree_Celsius", "x", {"x": ureg.Quantity(1)})


def test_logarithmic_unit_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError, match="Logarithmic"):
        ureg.differentiate("x", "x", {"x": ureg.Quantity(1, "dB")})


def test_active_context_rejected(ureg):
    ureg.add_context(Context("ad_empty"))
    with ureg.context("ad_empty"):
        with pytest.raises(ExpressionEvaluationError, match="context"):
            ureg.differentiate("x**2", "x", {"x": ureg.Quantity(2, "m")})


def test_floor_division_operator_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError, match="not supported"):
        ureg.differentiate("x // 2", "x", {"x": ureg.Quantity(3, "m")})


def test_variable_exponent_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError, match="constant exponent"):
        ureg.differentiate(
            "x ** y", "x", {"x": ureg.Quantity(2, "m"), "y": ureg.Quantity(3)}
        )


def test_dimensional_exponent_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError, match="dimensionless"):
        ureg.differentiate("x ** (2 * meter)", "x", {"x": ureg.Quantity(2, "m")})


def test_cross_registry_binding_rejected():
    registry_a = UnitRegistry()
    registry_b = UnitRegistry()
    foreign = registry_b.Quantity(1, "m")
    with pytest.raises(ExpressionEvaluationError, match="different unit registry"):
        registry_a.differentiate("x", "x", {"x": foreign})


def test_empty_expression_rejected(ureg):
    with pytest.raises(ExpressionEvaluationError):
        ureg.differentiate("   ", "x", {"x": ureg.Quantity(1, "m")})


def test_percent_follows_preprocessing_and_matches_parse_expression(ureg):
    # Pint rewrites '%' as the percent unit, i.e. implicit multiplication,
    # which stays within the supported algebraic grammar.
    x = ureg.Quantity(3, "m")
    result = ureg.differentiate("x % 2", "x", {"x": x})
    assert result.value == ureg.parse_expression("x % 2", x=x)
    assert result.derivative == ureg.parse_expression("2 percent", x=x)


def test_parse_expression_default_semantics_unchanged(ureg):
    # Guard against accidental changes to the existing evaluator.
    assert ureg.parse_expression("m * s ** -2").units == ureg.m / ureg.s**2
    assert ureg.parse_expression("x * y", x=2, y=3) == 6
    assert ureg.parse_expression("") == ureg.Quantity(1)
