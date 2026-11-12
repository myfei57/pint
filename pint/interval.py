"""
pint.interval
~~~~~~~~~~~~~

Deterministic closed intervals over scalar float64 quantities.

The module provides :class:`QuantityInterval`, a closed interval ``[lower,
upper]`` whose bounds are :class:`~pint.Quantity` objects from a *single*
unit registry.

Design constraints of this first version
-----------------------------------------

* only finite Python ``float`` (float64) scalar magnitudes are accepted;
* only multiplicative units are supported. Absolute (offset) temperature
  scales, logarithmic units, non-linear contexts and NaN unit scales are
  rejected;
* every elementary floating point operation is outwardly enveloped. The
  exact real result is computed with :class:`fractions.Fraction` (every
  float is an exact rational) and then rounded down/up to the surrounding
  float64 values, so the returned interval provably contains the true
  result and the result of ordinary float arithmetic;
* unit conversion does **not** rely on a single ``Quantity.to`` call plus a
  blind ULP expansion. The actual float scale used by the registry is
  treated as a concrete parameter, together with the exact rational factor
  reconstructed from the registry's definition chain, and both are
  incorporated into the outward envelope (see :func:`_convert_bounds`);
* no correlation analysis is performed. ``x - x`` is therefore generally
  a non-degenerate interval containing zero instead of an exact zero;
* :class:`~pint.Measurement` standard deviations are not turned into
  deterministic intervals and first-order error propagation formulas are
  never used;
* results outside the finite float64 range raise
  :class:`~pint.errors.IntervalOverflowError` instead of being clamped or
  returned with reversed bounds;
* ndarray magnitudes, trigonometric functions, three-valued interval
  comparison logic and automatic dependency tracking are out of scope.

:copyright: 2026 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
from fractions import Fraction
from typing import TYPE_CHECKING, Any

from .errors import (
    IntervalDomainError,
    IntervalOverflowError,
    IntervalUndefinedError,
)
from .util import SharedRegistryObject, UnitsContainer, to_units_container

if TYPE_CHECKING:
    from .facets.plain.quantity import PlainQuantity

# Float64 boundary constants.
_INF = math.inf
_NEG_INF = -math.inf


# ---------------------------------------------------------------------------
# Outward rounding primitives
# ---------------------------------------------------------------------------


def _outward_round(exact: Fraction) -> tuple[float, float]:
    """Return the smallest closed float64 interval containing ``exact``.

    ``exact`` is an exact rational number. The lower bound is the greatest
    float64 value less than or equal to it and the upper bound is the
    smallest float64 value greater than or equal to it. Subnormal values and
    signed zero are handled by :func:`math.nextafter`.
    """
    try:
        rounded = float(exact)
    except OverflowError:
        raise IntervalOverflowError(
            f"Interval result {exact.numerator}/{exact.denominator} is outside "
            "the finite float64 range"
        ) from None

    rounded_exact = Fraction(rounded)
    if rounded_exact == exact:
        return rounded, rounded
    if rounded_exact > exact:
        # Round-to-nearest went up: the lower neighbor is below ``exact``.
        lower = math.nextafter(rounded, _NEG_INF)
        upper = rounded
    else:
        lower = rounded
        upper = math.nextafter(rounded, _INF)

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise IntervalOverflowError(
            "Interval result cannot be enclosed with finite float64 bounds "
            f"(exact value: {exact.numerator}/{exact.denominator})"
        )
    return lower, upper


def _envelope(candidates: list[Fraction]) -> tuple[float, float]:
    """Outwardly envelope a set of exact rational candidates."""
    lower = _INF
    upper = _NEG_INF
    for candidate in candidates:
        cand_lower, cand_upper = _outward_round(candidate)
        if cand_lower < lower:
            lower = cand_lower
        if cand_upper > upper:
            upper = cand_upper
    return float(lower), float(upper)


def _interval_add(
    a_lower: float, a_upper: float, b_lower: float, b_upper: float
) -> tuple[float, float]:
    return _envelope(
        [
            Fraction(a_lower) + Fraction(b_lower),
            Fraction(a_upper) + Fraction(b_upper),
        ]
    )


def _interval_sub(
    a_lower: float, a_upper: float, b_lower: float, b_upper: float
) -> tuple[float, float]:
    return _envelope(
        [
            Fraction(a_lower) - Fraction(b_upper),
            Fraction(a_upper) - Fraction(b_lower),
        ]
    )


def _interval_mul(
    a_lower: float, a_upper: float, b_lower: float, b_upper: float
) -> tuple[float, float]:
    a = (Fraction(a_lower), Fraction(a_upper))
    b = (Fraction(b_lower), Fraction(b_upper))
    return _envelope([x * y for x in a for y in b])


def _interval_div(
    a_lower: float, a_upper: float, b_lower: float, b_upper: float
) -> tuple[float, float]:
    # The caller guarantees that the divisor interval does not contain zero.
    a = (Fraction(a_lower), Fraction(a_upper))
    b = (Fraction(b_lower), Fraction(b_upper))
    return _envelope([x / y for x in a for y in b])


def _interval_pow(lower: float, upper: float, exponent: int) -> tuple[float, float]:
    """Outwardly envelope ``x ** exponent`` for an integer ``exponent``."""
    candidates = [Fraction(lower) ** exponent, Fraction(upper) ** exponent]
    # For even powers the extremum over a zero-crossing interval is zero.
    if exponent % 2 == 0 and lower <= 0.0 <= upper:
        candidates.append(Fraction(0))
    return _envelope(candidates)


# ---------------------------------------------------------------------------
# Registry conversion helpers
# ---------------------------------------------------------------------------


def _exact_chain_factor(
    registry: Any, units: UnitsContainer, exponent: int = 1
) -> Fraction:
    """Exact rational conversion factor implied by the registry definitions.

    The walk mirrors ``GenericPlainRegistry._get_root_units_recurse`` but
    combines every float scale as the exact rational it represents, so the
    rounding accumulated along the scale chain is not hidden.
    """
    factor = Fraction(1)
    for key in units:
        name = registry.get_name(key)
        if name == "":
            continue
        definition = registry._units[name]
        power = exponent * units[key]
        if not definition.is_base:
            # Base units reference a dimension (e.g. ``[length]``), not other
            # units, and carry no scale. Only derived definitions recurse.
            scale = definition.converter.scale
            factor *= Fraction(scale) ** power
            if definition.reference is not None:
                factor *= _exact_chain_factor(registry, definition.reference, power)
    return factor


def _conversion_factors(
    registry: Any, src: UnitsContainer, dst: UnitsContainer
) -> tuple[float, Fraction]:
    """Return ``(float factor, exact rational factor)`` for ``src -> dst``.

    The float factor is the concrete value actually used by
    :meth:`UnitRegistry.convert`; the rational factor additionally bounds the
    rounding accumulated while that float was built from the definition
    chain.
    """
    factor = registry._get_conversion_factor(src, dst)
    if isinstance(factor, BaseException):
        # ``_get_conversion_factor`` returns a DimensionalityError instance.
        raise IntervalDomainError(
            f"Cannot convert interval between incompatible units {src!s} and {dst!s}"
        ) from factor
    float_factor = float(factor)
    if not math.isfinite(float_factor):
        raise IntervalDomainError(
            f"Conversion factor between {src!s} and {dst!s} is not finite"
        )
    exact_factor = _exact_chain_factor(registry, src / dst)
    return float_factor, exact_factor


def _convert_point(
    registry: Any, value: float, src: UnitsContainer, dst: UnitsContainer
) -> tuple[float, float, Fraction]:
    """Convert one float64 point, returning ``(lower, upper, exact)``.

    ``lower``/``upper`` outwardly envelope the converted point while
    ``exact`` is its exact rational value (needed to validate bound
    ordering without relying on rounded floats).
    """
    if src == dst:
        return value, value, Fraction(value)

    float_factor, exact_factor = _conversion_factors(registry, src, dst)
    exact = Fraction(value) * exact_factor
    lower, upper = _outward_round(exact)

    # Verify that the registry's actual float materialization is contained
    # and step outward only if the chain rounding requires it.
    materialized = value * float_factor
    while materialized < lower:
        lower = math.nextafter(lower, _NEG_INF)
    while materialized > upper:
        upper = math.nextafter(upper, _INF)

    if not math.isfinite(lower) or not math.isfinite(upper):
        raise IntervalOverflowError(
            f"Converting interval to {dst!s} exceeds the finite float64 range"
        )
    return lower, upper, exact


def _convert_bounds(
    registry: Any,
    lower: float,
    upper: float,
    src: UnitsContainer,
    dst: UnitsContainer,
) -> tuple[float, float]:
    """Convert a float64 interval from ``src`` to ``dst`` units, outwardly.

    Each endpoint is enveloped against both the exact rational conversion
    (definition scales treated as exact rationals) and the concrete float
    scale that the registry really multiplies with. The latter check makes
    the result contain ordinary ``Quantity.to`` output without relying on a
    blind fixed-ULP inflation.
    """
    if src == dst:
        return lower, upper

    lower_lo, _, _ = _convert_point(registry, lower, src, dst)
    _, upper_hi, _ = _convert_point(registry, upper, src, dst)
    return lower_lo, upper_hi


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def _assert_supported_units(registry: Any, units: UnitsContainer, context: str) -> None:
    """Reject offset, logarithmic and NaN-scaled units."""
    for key in units:
        name = registry.get_name(key)
        if name == "":
            raise IntervalDomainError(f"Unknown unit {key!r} while {context}")
        definition = registry._units[name]
        converter = definition.converter
        if not converter.is_multiplicative:
            raise IntervalDomainError(
                f"Unit {name!r} is an offset/non-multiplicative unit; "
                "QuantityInterval only supports multiplicative units"
            )
        if converter.is_logarithmic:
            raise IntervalDomainError(
                f"Unit {name!r} is logarithmic; logarithmic units are not "
                "supported by QuantityInterval"
            )
        scale = getattr(converter, "scale", 1.0)
        if not math.isfinite(float(scale)):
            raise IntervalDomainError(
                f"Unit {name!r} has a non-finite conversion scale"
            )


def _finite_float_magnitude(quantity: PlainQuantity, role: str) -> float:
    """Validate and return a finite float64 scalar magnitude."""
    magnitude = quantity.magnitude
    if hasattr(magnitude, "nominal_value"):
        raise IntervalDomainError(
            "Measurement objects carry a standard deviation, not a "
            "deterministic interval; build a QuantityInterval from explicit "
            "bounds instead"
        )
    if isinstance(magnitude, bool) or not isinstance(magnitude, (int, float)):
        raise IntervalDomainError(
            f"QuantityInterval only accepts finite float64 scalar magnitudes; "
            f"the {role} bound has magnitude of type {type(magnitude).__name__}"
        )
    value = float(magnitude)
    if not math.isfinite(value):
        raise IntervalDomainError(
            f"QuantityInterval bounds must be finite; the {role} bound is {value!r}"
        )
    return value


# ---------------------------------------------------------------------------
# QuantityInterval
# ---------------------------------------------------------------------------


class QuantityInterval:
    """A closed deterministic interval between two scalar float64 quantities.

    Parameters
    ----------
    lower : pint.Quantity
        Lower bound of the interval. Must be a finite scalar float64
        quantity using multiplicative units.
    upper : pint.Quantity
        Upper bound of the interval, from the same registry as ``lower``.
    units : str, pint.Unit or pint.UnitsContainer, optional
        Common unit in which both bounds are stored. When omitted, the units
        of ``lower`` are used. Both bounds are converted to this unit before
        checking that ``lower <= upper``.

    Raises
    ------
    IntervalDomainError
        If the inputs do not satisfy the first-version constraints
        (incompatible dimensions, different registries, non-multiplicative
        units, non-finite or non-scalar magnitudes, ...).
    IntervalOverflowError
        If a converted bound cannot be represented as a finite float64.
    """

    __slots__ = ("_registry", "_units", "_lower", "_upper")

    def __init__(
        self,
        lower: PlainQuantity,
        upper: PlainQuantity,
        units: Any = None,
    ) -> None:
        registry = self._shared_registry(lower, upper)
        self._registry = registry

        if getattr(registry, "_active_ctx", None):
            raise IntervalDomainError(
                "QuantityInterval cannot be used while a (potentially "
                "non-linear) context is active"
            )

        lower_mag = _finite_float_magnitude(lower, "lower")
        upper_mag = _finite_float_magnitude(upper, "upper")

        if units is None:
            target = lower._units.copy()
        else:
            target = to_units_container(units, registry).copy()

        for quantity, role in ((lower, "lower"), (upper, "upper")):
            _assert_supported_units(
                registry, quantity._units, f"validating the {role} bound"
            )
        _assert_supported_units(registry, target, "resolving the common unit")

        if registry._get_dimensionality(lower._units) != registry._get_dimensionality(
            upper._units
        ):
            raise IntervalDomainError(
                f"Cannot build an interval from bounds of different "
                f"dimensions: {lower.units!s} and {upper.units!s}"
            )
        if registry._get_dimensionality(lower._units) != registry._get_dimensionality(
            target
        ):
            raise IntervalDomainError(
                f"Common unit {target!s} is incompatible with the interval "
                f"dimensions of {lower.units!s}"
            )

        lower_lo, lower_hi, lower_exact = _convert_point(
            registry, lower_mag, lower._units, target
        )
        upper_lo, upper_hi, upper_exact = _convert_point(
            registry, upper_mag, upper._units, target
        )

        # Reject only when the outward envelopes are strictly disjoint,
        # i.e. no float64 closed interval can contain both points. When they
        # merely overlap (e.g. ``100 cm`` and ``1 m`` are equal after float
        # rounding but not under the exact definition scales), merge them
        # into the smallest enclosing float64 interval.
        if lower_lo > upper_hi:
            raise IntervalDomainError(
                "The lower bound must be less than or equal to the upper "
                "bound after conversion; got exact values "
                f"{lower_exact} > {upper_exact} in {target!s}"
            )

        self._lower = min(lower_lo, upper_lo)
        self._upper = max(lower_hi, upper_hi)
        self._units = target

    @staticmethod
    def _shared_registry(lower: Any, upper: Any) -> Any:
        if not isinstance(lower, SharedRegistryObject) or not isinstance(
            upper, SharedRegistryObject
        ):
            raise IntervalDomainError(
                "QuantityInterval bounds must both be pint Quantity objects"
            )
        if lower._REGISTRY is not upper._REGISTRY:
            raise IntervalDomainError(
                "Cannot build an interval from quantities of different unit registries"
            )
        return lower._REGISTRY

    @classmethod
    def _from_bounds(
        cls,
        registry: Any,
        lower: float,
        upper: float,
        units: UnitsContainer,
    ) -> QuantityInterval:
        """Internal constructor trusting already validated finite bounds."""
        if not (math.isfinite(lower) and math.isfinite(upper)):
            raise IntervalOverflowError(
                "Cannot build an interval with non-finite bounds"
            )
        if lower > upper:
            raise IntervalOverflowError(
                "Refusing to build an interval with reversed bounds"
            )
        instance = object.__new__(cls)
        instance._registry = registry
        instance._units = units.copy()
        instance._lower = lower
        instance._upper = upper
        return instance

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def lower(self) -> PlainQuantity:
        """Lower bound as a new Quantity (internal state is not exposed)."""
        return self._registry.Quantity(self._lower, self._units)

    @property
    def upper(self) -> PlainQuantity:
        """Upper bound as a new Quantity (internal state is not exposed)."""
        return self._registry.Quantity(self._upper, self._units)

    @property
    def units(self) -> Any:
        """Units in which the interval is stored."""
        return self._registry.Unit(self._units)

    @property
    def registry(self) -> Any:
        """Unit registry the interval belongs to."""
        return self._registry

    @property
    def dimensionality(self) -> UnitsContainer:
        """Dimensionality of the interval."""
        return self._registry._get_dimensionality(self._units)

    @property
    def width(self) -> PlainQuantity:
        """Exact float width of the interval (upper - lower)."""
        return self._registry.Quantity(self._upper - self._lower, self._units)

    def __contains__(self, item: Any) -> bool:
        if not isinstance(item, SharedRegistryObject):
            raise TypeError("QuantityInterval membership requires a pint Quantity")
        if item._REGISTRY is not self._registry:
            raise IntervalDomainError(
                "Cannot test membership of a quantity from another registry"
            )
        _assert_supported_units(
            self._registry, item._units, "testing interval membership"
        )
        if self._registry._get_dimensionality(
            item._units
        ) != self._registry._get_dimensionality(self._units):
            return False
        magnitude = item.to(self._units).magnitude
        return self._lower <= magnitude <= self._upper

    # ------------------------------------------------------------------
    # Conversion
    # ------------------------------------------------------------------

    def to(self, units: Any) -> QuantityInterval:
        """Return the interval expressed in other (compatible) units."""
        target = to_units_container(units, self._registry)
        _assert_supported_units(self._registry, target, "converting the interval")
        if self._registry._get_dimensionality(
            target
        ) != self._registry._get_dimensionality(self._units):
            raise IntervalDomainError(
                f"Cannot convert interval to incompatible units {target!s}"
            )
        lower, upper = _convert_bounds(
            self._registry, self._lower, self._upper, self._units, target
        )
        return self._from_bounds(self._registry, lower, upper, target)

    # ------------------------------------------------------------------
    # Operand normalization
    # ------------------------------------------------------------------

    def _as_interval(self, other: Any) -> QuantityInterval:
        if isinstance(other, QuantityInterval):
            if other._registry is not self._registry:
                raise IntervalDomainError(
                    "Cannot combine intervals from different registries"
                )
            return other
        if isinstance(other, SharedRegistryObject):
            return QuantityInterval(other, other)
        if isinstance(other, bool) or not isinstance(other, (int, float)):
            return NotImplemented
        value = float(other)
        if not math.isfinite(value):
            raise IntervalDomainError("Interval arithmetic requires finite operands")
        point = self._registry.Quantity(value, self._registry.UnitsContainer())
        return QuantityInterval(point, point)

    def _check_same_dimension(self, other: QuantityInterval, operation: str) -> None:
        if self.dimensionality != other.dimensionality:
            raise IntervalDomainError(
                f"Cannot {operation} intervals of different dimensions: "
                f"{self.units!s} and {other.units!s}"
            )

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------

    def __add__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        self._check_same_dimension(operand, "add")
        other_lower, other_upper = _convert_bounds(
            self._registry,
            operand._lower,
            operand._upper,
            operand._units,
            self._units,
        )
        lower, upper = _interval_add(self._lower, self._upper, other_lower, other_upper)
        return self._from_bounds(self._registry, lower, upper, self._units)

    def __radd__(self, other: Any) -> QuantityInterval:
        return self.__add__(other)

    def __sub__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        self._check_same_dimension(operand, "subtract")
        other_lower, other_upper = _convert_bounds(
            self._registry,
            operand._lower,
            operand._upper,
            operand._units,
            self._units,
        )
        lower, upper = _interval_sub(self._lower, self._upper, other_lower, other_upper)
        return self._from_bounds(self._registry, lower, upper, self._units)

    def __rsub__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        self._check_same_dimension(operand, "subtract")
        return operand.__sub__(self)

    def __mul__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        lower, upper = _interval_mul(
            self._lower,
            self._upper,
            operand._lower,
            operand._upper,
        )
        units = self._units * operand._units
        return self._from_bounds(self._registry, lower, upper, units)

    def __rmul__(self, other: Any) -> QuantityInterval:
        return self.__mul__(other)

    def __truediv__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        if operand._lower <= 0.0 <= operand._upper:
            raise IntervalUndefinedError(
                "Cannot divide by an interval containing zero: "
                f"[{operand._lower!r}, {operand._upper!r}]"
            )
        lower, upper = _interval_div(
            self._lower,
            self._upper,
            operand._lower,
            operand._upper,
        )
        units = self._units / operand._units
        return self._from_bounds(self._registry, lower, upper, units)

    def __rtruediv__(self, other: Any) -> QuantityInterval:
        operand = self._as_interval(other)
        if operand is NotImplemented:
            return NotImplemented
        return operand.__truediv__(self)

    def __pow__(self, exponent: Any) -> QuantityInterval:
        if isinstance(exponent, bool) or not isinstance(exponent, int):
            raise IntervalDomainError(
                "QuantityInterval only supports integer powers; got "
                f"{exponent!r} of type {type(exponent).__name__}"
            )
        if exponent == 0:
            # Convention: any finite interval to the power zero is the
            # dimensionless point interval [1, 1], including zero bases
            # (mirroring Python's ``0.0 ** 0 == 1.0``).
            return self._from_bounds(
                self._registry,
                1.0,
                1.0,
                self._registry.UnitsContainer(),
            )
        if exponent < 0 and self._lower <= 0.0 <= self._upper:
            raise IntervalUndefinedError(
                "Cannot raise an interval containing zero to a negative "
                f"integer power (exponent {exponent})"
            )
        lower, upper = _interval_pow(self._lower, self._upper, exponent)
        units = self._units**exponent
        return self._from_bounds(self._registry, lower, upper, units)

    def __neg__(self) -> QuantityInterval:
        lower, upper = _envelope([-Fraction(self._lower), -Fraction(self._upper)])
        return self._from_bounds(self._registry, lower, upper, self._units)

    def __pos__(self) -> QuantityInterval:
        return self._from_bounds(self._registry, self._lower, self._upper, self._units)

    # ------------------------------------------------------------------
    # Representation and structural equality
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return f'QuantityInterval({self._lower!r}, {self._upper!r}, "{self._units!s}")'

    def __str__(self) -> str:
        return f"[{self._lower}, {self._upper}] {self._units!s}"

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, QuantityInterval):
            return NotImplemented
        return (
            self._registry is other._registry
            and self._units == other._units
            and self._lower == other._lower
            and self._upper == other._upper
        )

    def __hash__(self) -> int:
        return hash((id(self._registry), self._units, self._lower, self._upper))
