"""Tests for :class:`pint.QuantityInterval`.

The containment tests use :mod:`fractions` and :mod:`decimal` as high
precision oracles: every float64 endpoint is an exact rational, so the exact
real result of an operation on sampled points can be computed without any
float rounding and then checked against the outwardly enveloped interval.
"""

from __future__ import annotations

import math
import random
from decimal import Decimal, localcontext
from fractions import Fraction

import numpy as np
import pytest

import pint
from pint import (
    IntervalDomainError,
    IntervalError,
    IntervalOverflowError,
    IntervalUndefinedError,
    PintError,
    QuantityInterval,
)
from pint.testsuite import helpers

# ---------------------------------------------------------------------------
# High precision reference helpers
# ---------------------------------------------------------------------------


def fraction_chain_factor(ureg, units, exponent=1):
    """Exact Fraction factor of ``units`` against the registry root units.

    This is a test-side re-walk of the unit definitions (not of the code
    under test) treating every float scale as its exact rational value.
    """
    factor = Fraction(1)
    for key in units:
        name = ureg.get_name(key)
        if name == "":
            continue
        definition = ureg._units[name]
        power = exponent * units[key]
        if not definition.is_base:
            factor *= Fraction(definition.converter.scale) ** power
            if definition.reference is not None:
                factor *= fraction_chain_factor(ureg, definition.reference, power)
    return factor


def decimal_chain_factor(ureg, units, exponent=1):
    """Independent Decimal (60 digit) version of :func:`fraction_chain_factor`."""
    factor = Decimal(1)
    for key in units:
        name = ureg.get_name(key)
        if name == "":
            continue
        definition = ureg._units[name]
        power = exponent * units[key]
        if not definition.is_base:
            factor *= Decimal(str(definition.converter.scale)) ** power
            if definition.reference is not None:
                factor *= decimal_chain_factor(ureg, definition.reference, power)
    return factor


def conversion_factor(ureg, src, dst):
    """Exact rational factor of the ``src -> dst`` conversion."""
    from pint.util import to_units_container

    src_uc = to_units_container(src, ureg)
    dst_uc = to_units_container(dst, ureg)
    return fraction_chain_factor(ureg, src_uc / dst_uc)


def assert_brackets(interval, exact):
    """Assert that the exact rational ``exact`` lies inside ``interval``."""
    lower = interval.lower.magnitude
    upper = interval.upper.magnitude
    assert math.isfinite(lower) and math.isfinite(upper)
    assert lower <= upper
    assert Fraction(lower) <= exact <= Fraction(upper), (
        f"{exact} not bracketed by [{lower}, {upper}]"
    )


def assert_contains_float(interval, value_float):
    """Assert that an ordinary float result lies inside ``interval``."""
    lower = interval.lower.magnitude
    upper = interval.upper.magnitude
    assert lower <= value_float <= upper, (
        f"{value_float} not contained in [{lower}, {upper}]"
    )


def random_interval(rng, low_exp=-4, high_exp=4):
    """Return a sorted pair of random finite floats (may be negative)."""
    values = [
        math.ldexp(rng.uniform(-1.0, 1.0), rng.randint(low_exp, high_exp))
        for _ in range(2)
    ]
    values.sort()
    return values[0], values[1]


def sample_point(rng, lower, upper):
    """Pick a sample float in [lower, upper], favoring endpoints sometimes."""
    if rng.random() < 0.15:
        return rng.choice((lower, upper))
    return rng.uniform(lower, upper)


# ---------------------------------------------------------------------------
# Construction and accessors
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_basic(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "cm"), ureg.Quantity(2.0, "cm"))
        assert iv.lower.magnitude == 1.0
        assert iv.upper.magnitude == 2.0
        assert str(iv.units) == "centimeter"

    def test_common_unit_conversion_in_constructor(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(
            ureg.Quantity(100.0, "cm"),
            ureg.Quantity(1.0, "m"),
            units="m",
        )
        assert iv.lower.magnitude == pytest.approx(1.0)
        assert iv.upper.magnitude == pytest.approx(1.0)
        assert iv.lower <= iv.upper
        # The merged envelope must contain both exact rational conversions,
        # which are not strictly equal because float 0.01 is not 1/100.
        assert_brackets(iv, Fraction(100) * conversion_factor(ureg, "cm", "m"))
        assert_brackets(iv, Fraction(1))

    def test_lower_greater_than_upper_rejected(self, sess_registry):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(2.0, "m"), ureg.Quantity(1.0, "m"))

    def test_inputs_are_not_modified(self, sess_registry):
        ureg = sess_registry
        q1 = ureg.Quantity(1.0, "cm")
        q2 = ureg.Quantity(2.0, "cm")
        before = (q1.magnitude, str(q1.units), q2.magnitude, str(q2.units))
        iv = QuantityInterval(q1, q2)
        iv.to("m")
        iv + iv
        assert before == (
            q1.magnitude,
            str(q1.units),
            q2.magnitude,
            str(q2.units),
        )

    def test_endpoints_do_not_expose_internal_state(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "cm"), ureg.Quantity(2.0, "cm"))
        lower = iv.lower
        lower._magnitude = 999.0
        assert iv.lower.magnitude == 1.0
        assert iv.lower is not iv.lower

    def test_width_and_repr(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(3.5, "m"))
        assert iv.width.magnitude == 2.5
        assert "QuantityInterval" in repr(iv)
        assert "meter" in str(iv)

    def test_equality_and_hash(self, sess_registry):
        ureg = sess_registry
        iv1 = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        iv2 = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        assert iv1 == iv2
        assert hash(iv1) == hash(iv2)
        assert iv1 != 42


# ---------------------------------------------------------------------------
# Rejected inputs
# ---------------------------------------------------------------------------


class TestRejections:
    def test_incompatible_dimensions(self, sess_registry):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "s"))

    def test_incompatible_common_unit(self, sess_registry):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(
                ureg.Quantity(1.0, "m"),
                ureg.Quantity(2.0, "m"),
                units="s",
            )

    def test_different_registries(self):
        ureg1 = pint.UnitRegistry()
        ureg2 = pint.UnitRegistry()
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg1.Quantity(1.0, "m"), ureg2.Quantity(2.0, "m"))

    def test_non_quantity_bounds(self, sess_registry):
        with pytest.raises(IntervalDomainError):
            QuantityInterval(1.0, 2.0)

    @pytest.mark.parametrize("unit", ["degC", "degF"])
    def test_offset_units_rejected(self, sess_registry, unit):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(1.0, unit), ureg.Quantity(2.0, unit))

    def test_delta_temperature_units_are_linear_and_accepted(self, sess_registry):
        # Delta units are pure scale (temperature differences), not absolute
        # temperature scales, so they are legitimate multiplicative units.
        ureg = sess_registry
        iv = QuantityInterval(
            ureg.Quantity(1.0, "delta_degC"),
            ureg.Quantity(2.0, "delta_degC"),
        )
        converted = iv.to("delta_degF")
        assert converted.lower.magnitude <= 1.8 <= converted.upper.magnitude
        assert converted.upper.magnitude >= 3.6

    @pytest.mark.parametrize("unit", ["dB", "dBm"])
    def test_logarithmic_units_rejected(self, sess_registry, unit):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(1.0, unit), ureg.Quantity(2.0, unit))

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
    def test_non_finite_magnitudes_rejected(self, sess_registry, value):
        ureg = sess_registry
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(value, "m"), ureg.Quantity(2.0, "m"))

    def test_ndarray_magnitude_rejected(self, sess_registry):
        ureg = sess_registry
        array_q = ureg.Quantity(np.array([1.0, 2.0]), "m")
        with pytest.raises(IntervalDomainError):
            QuantityInterval(array_q, ureg.Quantity(2.0, "m"))

    def test_active_context_rejected(self, sess_registry):
        ureg = sess_registry
        with ureg.context("sp"):
            with pytest.raises(IntervalDomainError):
                QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))

    def test_non_integer_power_rejected(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        with pytest.raises(IntervalDomainError):
            iv**1.5

    @helpers.requires_uncertainties()
    def test_measurement_rejected(self, sess_registry):
        ureg = sess_registry
        measurement = ureg.Measurement(1.0, 0.1, "m")
        with pytest.raises(IntervalDomainError):
            QuantityInterval(measurement, ureg.Quantity(2.0, "m"))

    def test_error_hierarchy(self):
        assert issubclass(IntervalDomainError, IntervalError)
        assert issubclass(IntervalUndefinedError, IntervalError)
        assert issubclass(IntervalOverflowError, OverflowError)
        assert issubclass(IntervalOverflowError, PintError)


# ---------------------------------------------------------------------------
# Unit conversion
# ---------------------------------------------------------------------------


class TestConversion:
    def test_cm_to_m_exact_bounds(self, sess_registry):
        ureg = sess_registry
        point = QuantityInterval(ureg.Quantity(1.0, "cm"), ureg.Quantity(1.0, "cm")).to(
            "m"
        )
        exact = Fraction(1) * conversion_factor(ureg, "cm", "m")
        assert_brackets(point, exact)
        # The centimeter scale is the float 0.01, not exactly 1/100.
        assert conversion_factor(ureg, "cm", "m") == Fraction(
            ureg._units["centimeter"].converter.scale
        )
        assert conversion_factor(ureg, "cm", "m") != Fraction(1, 100)

    def test_decimal_oracle_agrees_on_chain_factor(self, sess_registry):
        ureg = sess_registry
        # inch is defined through yard (1/36 float) and meter (0.9144), so the
        # chain exercises several float scales. A local context avoids
        # changing the process-wide Decimal precision (default 28).
        with localcontext() as ctx:
            ctx.prec = 60
            container = ureg.UnitsContainer({"inch": 1, "meter": -1})
            fraction_factor = fraction_chain_factor(ureg, container)
            decimal_factor = decimal_chain_factor(ureg, container)
            fraction_as_decimal = Decimal(fraction_factor.numerator) / Decimal(
                fraction_factor.denominator
            )
            # Exact-rational and decimal-repr oracles differ by at most one
            # float ULP on each scale (both derive from the same float64s).
            relative_gap = abs(fraction_as_decimal - decimal_factor) / decimal_factor
            assert relative_gap < Decimal("1e-16")
            # Both agree with the nominal physical ratio to float precision.
            assert abs(decimal_factor - Decimal("0.0254")) < Decimal("1e-17")

    @pytest.mark.parametrize("seed", range(20))
    def test_random_conversion_containment(self, sess_registry, seed):
        ureg = sess_registry
        rng = random.Random(seed)
        lo, hi = random_interval(rng)
        iv = QuantityInterval(ureg.Quantity(lo, "cm"), ureg.Quantity(hi, "cm"))
        converted = iv.to("m")
        exact_factor = conversion_factor(ureg, "cm", "m")
        for _ in range(25):
            sample = sample_point(rng, lo, hi)
            exact = Fraction(sample) * exact_factor
            assert_brackets(converted, exact)
            # Ordinary pint float conversion must be contained too.
            pint_value = ureg.Quantity(sample, "cm").to("m").magnitude
            assert_contains_float(converted, pint_value)

    @pytest.mark.parametrize("seed", range(10))
    def test_random_reverse_conversion_containment(self, sess_registry, seed):
        ureg = sess_registry
        rng = random.Random(1000 + seed)
        lo, hi = random_interval(rng)
        iv = QuantityInterval(ureg.Quantity(lo, "m"), ureg.Quantity(hi, "m"))
        converted = iv.to("cm")
        exact_factor = conversion_factor(ureg, "m", "cm")
        for _ in range(25):
            sample = sample_point(rng, lo, hi)
            assert_brackets(converted, Fraction(sample) * exact_factor)
            assert_contains_float(
                converted, ureg.Quantity(sample, "m").to("cm").magnitude
            )

    def test_round_trip_conversion_contains_original(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(
            ureg.Quantity(1.23456789, "cm"),
            ureg.Quantity(9.87654321, "cm"),
        )
        round_trip = iv.to("m").to("cm")
        assert round_trip.lower.magnitude <= 1.23456789
        assert round_trip.upper.magnitude >= 9.87654321

    def test_incompatible_conversion_rejected(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        with pytest.raises(IntervalDomainError):
            iv.to("s")

    def test_conversion_overflow(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1e308, "m"), ureg.Quantity(1e308, "m"))
        with pytest.raises(IntervalOverflowError):
            iv.to("cm")


# ---------------------------------------------------------------------------
# Arithmetic
# ---------------------------------------------------------------------------


class TestAddSub:
    @pytest.mark.parametrize("seed", range(20))
    def test_random_add_sub_containment(self, sess_registry, seed):
        ureg = sess_registry
        rng = random.Random(2000 + seed)
        a_lo, a_hi = random_interval(rng)
        b_lo, b_hi = random_interval(rng)
        a = QuantityInterval(ureg.Quantity(a_lo, "m"), ureg.Quantity(a_hi, "m"))
        b = QuantityInterval(ureg.Quantity(b_lo, "m"), ureg.Quantity(b_hi, "m"))
        added = a + b
        subtracted = a - b
        for _ in range(25):
            x = sample_point(rng, a_lo, a_hi)
            y = sample_point(rng, b_lo, b_hi)
            assert_brackets(added, Fraction(x) + Fraction(y))
            assert_brackets(subtracted, Fraction(x) - Fraction(y))
            assert_contains_float(
                added, (ureg.Quantity(x, "m") + ureg.Quantity(y, "m")).magnitude
            )
            assert_contains_float(
                subtracted,
                (ureg.Quantity(x, "m") - ureg.Quantity(y, "m")).magnitude,
            )

    def test_cross_unit_addition(self, sess_registry):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        b = QuantityInterval(ureg.Quantity(100.0, "cm"), ureg.Quantity(200.0, "cm"))
        result = a + b
        assert str(result.units) == "meter"
        # 1 m + 100 cm = 2 m and 2 m + 200 cm = 4 m at the extreme corners.
        assert_brackets(result, Fraction(2))
        assert_brackets(result, Fraction(4))

    def test_subtraction_of_interval_from_itself_not_exact_zero(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.3, "m"), ureg.Quantity(2.7, "m"))
        difference = iv - iv
        # No correlation is assumed: zero is inside but the interval is not
        # the exact point zero.
        assert ureg.Quantity(0.0, "m") in difference
        assert difference.lower.magnitude < 0.0 < difference.upper.magnitude
        assert difference.width.magnitude > 0.0

    def test_dimension_mismatch_rejected(self, sess_registry):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        b = QuantityInterval(ureg.Quantity(1.0, "s"), ureg.Quantity(2.0, "s"))
        with pytest.raises(IntervalDomainError):
            a + b
        with pytest.raises(IntervalDomainError):
            a - b

    def test_bare_number_and_reflected_operations(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, ""), ureg.Quantity(2.0, ""))
        result = iv + 0.5
        assert result.lower.magnitude <= 1.5 <= result.upper.magnitude
        reflected = 0.5 + iv
        assert reflected == result
        assert (2.0 - iv).upper.magnitude >= 1.0
        with pytest.raises(IntervalDomainError):
            QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m")) + 1.0


class TestMulDiv:
    def test_negative_interval_multiplication_corners(self, sess_registry):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(-2.0, "m"), ureg.Quantity(-1.0, "m"))
        b = QuantityInterval(ureg.Quantity(-3.0, "m"), ureg.Quantity(-2.0, "m"))
        result = a * b
        assert result.lower.magnitude <= 2.0
        assert result.upper.magnitude >= 6.0
        assert "meter ** 2" in str(result.units)

    def test_mixed_sign_multiplication_corners(self, sess_registry):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(-2.0, "m"), ureg.Quantity(3.0, "m"))
        b = QuantityInterval(ureg.Quantity(4.0, "m"), ureg.Quantity(5.0, "m"))
        result = a * b
        # All four endpoint combinations are considered.
        assert result.lower.magnitude <= -10.0
        assert result.upper.magnitude >= 15.0

    @pytest.mark.parametrize("seed", range(20))
    def test_random_multiplication_containment(self, sess_registry, seed):
        ureg = sess_registry
        rng = random.Random(3000 + seed)
        a_lo, a_hi = random_interval(rng, low_exp=-3, high_exp=3)
        b_lo, b_hi = random_interval(rng, low_exp=-3, high_exp=3)
        a = QuantityInterval(ureg.Quantity(a_lo, "m"), ureg.Quantity(a_hi, "m"))
        b = QuantityInterval(ureg.Quantity(b_lo, "s"), ureg.Quantity(b_hi, "s"))
        result = a * b
        for _ in range(25):
            x = sample_point(rng, a_lo, a_hi)
            y = sample_point(rng, b_lo, b_hi)
            assert_brackets(result, Fraction(x) * Fraction(y))
            assert_contains_float(
                result,
                (ureg.Quantity(x, "m") * ureg.Quantity(y, "s")).magnitude,
            )

    @pytest.mark.parametrize("seed", range(20))
    def test_random_division_containment(self, sess_registry, seed):
        ureg = sess_registry
        rng = random.Random(4000 + seed)
        a_lo, a_hi = random_interval(rng, low_exp=-3, high_exp=3)
        # Keep the divisor strictly on one side of zero.
        sign = rng.choice((-1.0, 1.0))
        b_lo, b_hi = sorted(
            sign * abs(random_interval(rng, low_exp=-3, high_exp=0)[1])
            for _ in range(2)
        )
        a = QuantityInterval(ureg.Quantity(a_lo, "m"), ureg.Quantity(a_hi, "m"))
        b = QuantityInterval(ureg.Quantity(b_lo, "s"), ureg.Quantity(b_hi, "s"))
        result = a / b
        for _ in range(25):
            x = sample_point(rng, a_lo, a_hi)
            y = sample_point(rng, b_lo, b_hi)
            assert_brackets(result, Fraction(x) / Fraction(y))
            assert_contains_float(
                result,
                (ureg.Quantity(x, "m") / ureg.Quantity(y, "s")).magnitude,
            )
        assert "meter / second" in str(result.units)

    @pytest.mark.parametrize(
        "bounds",
        [(-1.0, 1.0), (-2.0, 0.0), (0.0, 3.0)],
    )
    def test_division_by_zero_containing_interval_undefined(
        self, sess_registry, bounds
    ):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        b = QuantityInterval(
            ureg.Quantity(bounds[0], "s"), ureg.Quantity(bounds[1], "s")
        )
        with pytest.raises(IntervalUndefinedError):
            a / b

    def test_division_units_follow_quantity_algebra(self, sess_registry):
        ureg = sess_registry
        a = QuantityInterval(ureg.Quantity(4.0, "m"), ureg.Quantity(6.0, "m"))
        b = QuantityInterval(ureg.Quantity(2.0, "s"), ureg.Quantity(2.0, "s"))
        result = a / b
        assert result.lower.magnitude <= 2.0 <= result.upper.magnitude
        assert "meter / second" in str(result.units)

    def test_reflected_division(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(2.0, ""), ureg.Quantity(4.0, ""))
        result = 1.0 / iv
        assert result.lower.magnitude <= 0.25
        assert result.upper.magnitude >= 0.5


# ---------------------------------------------------------------------------
# Integer powers
# ---------------------------------------------------------------------------


class TestPower:
    def test_even_power_crossing_zero(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(-2.0, "m"), ureg.Quantity(3.0, "m"))
        result = iv**2
        assert result.lower.magnitude == 0.0
        assert result.upper.magnitude >= 9.0
        assert "meter ** 2" in str(result.units)

    def test_even_power_negative_interval(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(-3.0, "m"), ureg.Quantity(-2.0, "m"))
        result = iv**2
        assert result.lower.magnitude <= 4.0
        assert result.upper.magnitude >= 9.0

    def test_odd_power_crossing_zero(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(-2.0, "m"), ureg.Quantity(3.0, "m"))
        result = iv**3
        assert result.lower.magnitude <= -8.0
        assert result.upper.magnitude >= 27.0

    @pytest.mark.parametrize("exponent", [-2, -1, 1, 2, 3, 4])
    def test_random_power_containment(self, sess_registry, exponent):
        ureg = sess_registry
        rng = random.Random(5000 + exponent)
        lo, hi = random_interval(rng, low_exp=-2, high_exp=2)
        if exponent < 0:
            # Keep zero out of the base interval for negative exponents.
            lo = abs(lo) + 1e-4
            hi = abs(hi) + 1.0
        iv = QuantityInterval(ureg.Quantity(lo, "m"), ureg.Quantity(hi, "m"))
        result = iv**exponent
        for _ in range(25):
            x = sample_point(rng, lo, hi)
            assert_brackets(result, Fraction(x) ** exponent)

    def test_negative_exponent_with_zero_undefined(self, sess_registry):
        ureg = sess_registry
        for bounds in ((-1.0, 1.0), (0.0, 2.0), (-2.0, 0.0)):
            iv = QuantityInterval(
                ureg.Quantity(bounds[0], "m"),
                ureg.Quantity(bounds[1], "m"),
            )
            with pytest.raises(IntervalUndefinedError):
                iv**-1

    def test_zero_power_convention(self, sess_registry):
        ureg = sess_registry
        cases = [
            QuantityInterval(ureg.Quantity(-2.0, "m"), ureg.Quantity(3.0, "m")),
            QuantityInterval(ureg.Quantity(0.0, "m"), ureg.Quantity(0.0, "m")),
        ]
        for iv in cases:
            result = iv**0
            assert result.lower.magnitude == 1.0
            assert result.upper.magnitude == 1.0
            assert len(result.units._units) == 0
            assert not result.dimensionality

    def test_power_overflow(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1e200, "m"), ureg.Quantity(2e200, "m"))
        with pytest.raises(IntervalOverflowError):
            iv**2


# ---------------------------------------------------------------------------
# Float64 extremes
# ---------------------------------------------------------------------------


class TestFloatExtremes:
    def test_smallest_subnormal_squared(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(5e-324, ""), ureg.Quantity(5e-324, ""))
        result = iv**2
        # Exact result (~2.5e-648) underflows: the envelope must reach down
        # to zero and up to the smallest positive subnormal.
        assert result.lower.magnitude == 0.0
        assert result.upper.magnitude == 5e-324
        assert result.lower.magnitude <= result.upper.magnitude

    def test_subnormal_division(self, sess_registry):
        ureg = sess_registry
        numerator = QuantityInterval(
            ureg.Quantity(5e-324, ""), ureg.Quantity(5e-324, "")
        )
        denominator = QuantityInterval(
            ureg.Quantity(1e300, ""), ureg.Quantity(1e300, "")
        )
        result = numerator / denominator
        assert math.isfinite(result.lower.magnitude)
        assert math.isfinite(result.upper.magnitude)
        assert result.lower.magnitude <= result.upper.magnitude
        assert result.lower.magnitude == 0.0

    def test_near_overflow_multiplication(self, sess_registry):
        ureg = sess_registry
        big = QuantityInterval(ureg.Quantity(1e300, ""), ureg.Quantity(2e300, ""))
        with pytest.raises(IntervalOverflowError):
            big * big
        # The operands themselves remain valid, nothing is clamped.
        assert big.lower.magnitude == 1e300
        assert big.upper.magnitude == 2e300

    def test_no_reversed_bounds_on_overflow(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1e308, ""), ureg.Quantity(1e308, ""))
        with pytest.raises(IntervalOverflowError):
            iv**2
        with pytest.raises(IntervalOverflowError):
            iv * iv

    def test_large_but_representable_interval(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1e150, ""), ureg.Quantity(2e150, ""))
        result = iv * iv
        assert result.upper.magnitude >= 4e300
        assert math.isfinite(result.lower.magnitude)


# ---------------------------------------------------------------------------
# Membership, signs and misc
# ---------------------------------------------------------------------------


class TestMisc:
    def test_membership(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "cm"), ureg.Quantity(2.0, "cm"))
        assert ureg.Quantity(15.0, "mm") in iv
        assert ureg.Quantity(0.5, "cm") not in iv
        with pytest.raises(TypeError):
            1.5 in iv

    def test_membership_incompatible_dimensions(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        assert ureg.Quantity(1.5, "s") not in iv

    def test_neg_and_pos(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        negated = -iv
        assert negated.lower.magnitude <= -2.0
        assert negated.upper.magnitude >= -1.0
        assert +iv == iv

    def test_dimensionality(self, sess_registry):
        ureg = sess_registry
        iv = QuantityInterval(ureg.Quantity(1.0, "m"), ureg.Quantity(2.0, "m"))
        assert iv.dimensionality == ureg.Quantity(1.0, "m").dimensionality

    def test_quantity_interval_exported(self):
        assert pint.QuantityInterval is QuantityInterval
        assert pint.IntervalUndefinedError is IntervalUndefinedError
        assert all(
            name in pint.__all__
            for name in (
                "QuantityInterval",
                "IntervalError",
                "IntervalDomainError",
                "IntervalUndefinedError",
                "IntervalOverflowError",
            )
        )
