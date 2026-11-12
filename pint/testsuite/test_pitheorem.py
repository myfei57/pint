from __future__ import annotations

import itertools
import logging
from fractions import Fraction

import pytest

from pint import UnitRegistry, pi_theorem, pi_theorem_repeating
from pint.testsuite import QuantityTestCase


# TODO: do not subclass from QuantityTestCase
class TestPiTheorem(QuantityTestCase):
    def test_simple(self, caplog):
        # simple movement
        with caplog.at_level(logging.DEBUG):
            assert pi_theorem({"V": "m/s", "T": "s", "L": "m"}) == [
                {"V": 1, "T": 1, "L": -1}
            ]

            # pendulum
            assert pi_theorem({"T": "s", "M": "grams", "L": "m", "g": "m/s**2"}) == [
                {"g": 1, "T": 2, "L": -1}
            ]
        assert len(caplog.records) == 7

    def test_inputs(self):
        V = "km/hour"
        T = "ms"
        L = "cm"

        f1 = lambda x: x
        f2 = lambda x: self.Q_(1, x)
        f3 = lambda x: self.Q_(1, x).units
        f4 = lambda x: self.Q_(1, x).dimensionality

        fs = f1, f2, f3, f4
        for fv, ft, fl in itertools.product(fs, fs, fs):
            qv = fv(V)
            qt = ft(T)
            ql = ft(L)
            assert self.ureg.pi_theorem({"V": qv, "T": qt, "L": ql}) == [
                {"V": 1.0, "T": 1.0, "L": -1.0}
            ]


# TODO: do not subclass from QuantityTestCase
class TestPiTheoremRepeating(QuantityTestCase):
    """Entry point with user-specified repeating variables."""

    FLUID_UNITS = {
        "rho": "kg/m**3",
        "v": "m/s",
        "L": "m",
        "p": "Pa",
        "mu": "poise",
    }
    REPEATING = ["rho", "v", "L"]

    # p / (rho v**2): Euler number; mu / (rho v L): inverse Reynolds number
    EXPECTED = [
        {"p": Fraction(1), "rho": Fraction(-1), "v": Fraction(-2)},
        {
            "mu": Fraction(1),
            "rho": Fraction(-1),
            "v": Fraction(-1),
            "L": Fraction(-1),
        },
    ]

    def test_fluid_groups_exact_rational_exponents(self):
        groups = self.ureg.pi_theorem_repeating(
            self.FLUID_UNITS, self.REPEATING
        )
        assert groups == self.EXPECTED

        # Every exponent is an exact rational number, never a float.
        for group in groups:
            assert all(isinstance(exponent, Fraction) for exponent in group.values())
            assert not any(
                isinstance(exponent, float) for exponent in group.values()
            )

        # Each non-repeating variable appears once, normalized to exponent 1,
        # and no other non-repeating variable participates in its product.
        non_repeating = ["p", "mu"]
        for group, name in zip(groups, non_repeating):
            assert group[name] == Fraction(1)
            assert not (set(group) & (set(non_repeating) - {name}))

        # The zero exponent of L in the Euler group is omitted rather than
        # rendered as a floating point zero.
        assert "L" not in groups[0]

        # Output keeps the declaration order of the non-repeating variables.
        assert [next(iter(group)) for group in groups] == non_repeating

    def test_input_forms(self):
        f1 = lambda x: x
        f2 = lambda x: self.Q_(1, x)
        f3 = lambda x: self.Q_(1, x).units
        f4 = lambda x: self.Q_(1, x).dimensionality

        for form in (f1, f2, f3, f4):
            quantities = {name: form(unit) for name, unit in self.FLUID_UNITS.items()}
            assert (
                self.ureg.pi_theorem_repeating(quantities, self.REPEATING)
                == self.EXPECTED
            )

    def test_quantity_dimension_and_numeric_consistency(self):
        groups = {
            next(iter(group)): group
            for group in self.ureg.pi_theorem_repeating(
                self.FLUID_UNITS, self.REPEATING
            )
        }

        def evaluate(group, values):
            result = self.Q_(1, "dimensionless")
            for name, exponent in group.items():
                # Rational exponents are accepted directly by Quantity.
                result = result * values[name] ** exponent
            return result.to_base_units()

        si_values = {
            "rho": self.Q_(1000, "kg/m**3"),
            "v": self.Q_(2, "m/s"),
            "L": self.Q_(0.05, "m"),
            "p": self.Q_(4000, "Pa"),
            "mu": self.Q_(1e-3, "Pa s"),
        }
        # The same physical situation expressed with different units.
        alt_values = {
            "rho": si_values["rho"].to("g/cm**3"),
            "v": si_values["v"].to("km/hour"),
            "L": si_values["L"].to("cm"),
            "p": si_values["p"].to("kPa"),
            "mu": si_values["mu"].to("poise"),
        }

        for name, expected in (("p", 1.0), ("mu", 1e-5)):
            for values in (si_values, alt_values):
                product = evaluate(groups[name], values)
                # The product is genuinely dimensionless.
                assert product.dimensionality == {}
                assert product.units == self.ureg.Unit("dimensionless")
                assert abs(float(product.magnitude) - expected) < 1e-12

    def test_fractional_dimensions_float_registry(self):
        # Dimensionality exponents arrive as floats; they must be converted to
        # exact Fractions instead of propagating floating point arithmetic.
        groups = self.ureg.pi_theorem_repeating(
            {"y": "m", "z": "m**0.5", "w": "m**1.5"}, ["y"]
        )
        assert groups == [
            {"z": Fraction(1), "y": Fraction(-1, 2)},
            {"w": Fraction(1), "y": Fraction(-3, 2)},
        ]
        assert all(
            isinstance(exponent, Fraction)
            for group in groups
            for exponent in group.values()
        )

    def test_fractional_dimensions_fraction_registry(self):
        fraction_registry = UnitRegistry(non_int_type=Fraction)
        groups = fraction_registry.pi_theorem_repeating(
            {"y": "m", "z": "m**(1/3)", "w": "m**(2/3)"}, ["y"]
        )
        assert groups == [
            {"z": Fraction(1), "y": Fraction(-1, 3)},
            {"w": Fraction(1), "y": Fraction(-2, 3)},
        ]

    def test_dimensionless_non_repeating_variable(self):
        quantities = {
            "rho": "kg/m**3",
            "v": "m/s",
            "L": "m",
            "theta": "degree",
        }
        groups = self.ureg.pi_theorem_repeating(quantities, self.REPEATING)
        assert groups == [{"theta": Fraction(1)}]

        # A dimensionless product built only from dimensionless variables
        # needs no repeating variables.
        groups = self.ureg.pi_theorem_repeating(
            {"a": "degree", "b": "dimensionless"}, []
        )
        assert groups == [{"a": Fraction(1)}, {"b": Fraction(1)}]

    def test_missing_base_dimension_exponent_is_zero(self):
        # [length] and [time] are independent; the solved product needs both
        # and the absent base dimension of each variable is treated as zero.
        groups = self.ureg.pi_theorem_repeating(
            {"a": "m", "b": "s", "c": "m/s"}, ["a", "b"]
        )
        assert groups == [
            {"c": Fraction(1), "a": Fraction(-1), "b": Fraction(1)}
        ]

    def test_derived_units_match_expanded_base_dimensions(self):
        expanded = self.ureg.pi_theorem_repeating(
            {
                "rho": "kg/m**3",
                "v": "m/s",
                "L": "m",
                "p": "kg/(m s**2)",
                "mu": "g/(cm s)",
            },
            self.REPEATING,
        )
        assert expanded == self.EXPECTED

    def test_order_invariance(self):
        names = list(self.FLUID_UNITS)
        baseline = {
            next(iter(group)): group
            for group in self.ureg.pi_theorem_repeating(
                self.FLUID_UNITS, self.REPEATING
            )
        }
        for permutation in itertools.permutations(names):
            quantities = {name: self.FLUID_UNITS[name] for name in permutation}
            # Also permute the repeating variable order: the exponents attached
            # to each name must remain identical.
            groups = self.ureg.pi_theorem_repeating(
                quantities, ["L", "rho", "v"]
            )
            assert {next(iter(group)): group for group in groups} == baseline

        # Only the conventional output order follows the declaration order.
        groups = self.ureg.pi_theorem_repeating(
            {
                "mu": "poise",
                "rho": "kg/m**3",
                "p": "Pa",
                "v": "m/s",
                "L": "m",
            },
            self.REPEATING,
        )
        assert [next(iter(group)) for group in groups] == ["mu", "p"]

    def test_top_level_with_dimension_strings(self):
        groups = pi_theorem_repeating(
            {
                "rho": "[mass]/[length]**3",
                "v": "[length]/[time]",
                "L": "[length]",
                "p": "[mass]/[length]/[time]**2",
            },
            self.REPEATING,
        )
        assert groups == [self.EXPECTED[0]]

    def test_unknown_repeating_name(self):
        with pytest.raises(ValueError, match="nope"):
            self.ureg.pi_theorem_repeating(
                {"a": "m", "b": "kg"}, ["a", "nope"]
            )

    def test_duplicated_repeating_name(self):
        with pytest.raises(ValueError, match="'a'"):
            self.ureg.pi_theorem_repeating(
                {"a": "m", "b": "kg"}, ["a", "a"]
            )

    def test_dimensionless_repeating_variable(self):
        with pytest.raises(ValueError, match="theta"):
            self.ureg.pi_theorem_repeating(
                {"a": "m", "b": "kg", "theta": "degree"}, ["theta"]
            )

    def test_dependent_repeating_columns(self):
        # Two length-like columns cannot both be repeating variables.
        with pytest.raises(ValueError, match="linearly independent.*'b'"):
            self.ureg.pi_theorem_repeating(
                {"a": "m", "b": "cm", "c": "kg"}, ["a", "b"]
            )

        # A fourth repeating variable in an already spanned length dimension.
        with pytest.raises(ValueError, match="'D'"):
            self.ureg.pi_theorem_repeating(
                {
                    "rho": "kg/m**3",
                    "v": "m/s",
                    "L": "m",
                    "D": "cm",
                    "p": "Pa",
                },
                ["rho", "v", "L", "D"],
            )

    def test_too_few_repeating_variables_names_uncovered_dimension(self):
        with pytest.raises(ValueError) as exc_info:
            self.ureg.pi_theorem_repeating(
                {"a": "m", "b": "kg"}, ["a"]
            )
        message = str(exc_info.value)
        assert "at least 2" in message
        assert "'b'" in message
        assert "[mass]" in message

    def test_uncovered_extra_base_dimension(self):
        with pytest.raises(ValueError) as exc_info:
            self.ureg.pi_theorem_repeating(
                {
                    "rho": "kg/m**3",
                    "v": "m/s",
                    "L": "m",
                    "T": "kelvin",
                },
                self.REPEATING,
            )
        message = str(exc_info.value)
        assert "'T'" in message
        assert "[temperature]" in message

    def test_does_not_filter_automatic_null_space_basis(self):
        # The automatic null-space basis of pi_theorem is a different
        # parametrization: it normalizes repeating variables and leaves the
        # non-repeating variable with exponent -1. The new entry point must
        # independently enforce the fixed +1 normalization, so none of its
        # groups can be a row picked out of the automatic basis.
        quantities = {
            "rho": "kg/m**3",
            "v": "m/s",
            "L": "m",
            "p": "Pa",
            "mu": "poise",
        }
        groups = self.ureg.pi_theorem_repeating(quantities, self.REPEATING)
        basis = self.ureg.pi_theorem(quantities)
        assert groups == self.EXPECTED
        for group, name in zip(groups, ("p", "mu")):
            assert group[name] == Fraction(1)
            assert not any(
                set(basis_row) == set(group)
                and all(basis_row[key] == exponent for key, exponent in group.items())
                for basis_row in basis
            )
