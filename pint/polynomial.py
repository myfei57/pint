"""
pint.polynomial
~~~~~~~~~~~~~~~

Polynomial model for one-dimensional :class:`~pint.Quantity` samples.

The model keeps the fitted polynomial in a normalized, dimensionless
variable ``t = (x - origin) / scale``.  The origin and the (non-zero) scale
are stored as :class:`~pint.Quantity` objects, so evaluation always happens
in a well-conditioned domain even when the samples are located far from the
zero of the input unit.  Coefficients expressed in the original ``x`` power
basis are available on demand through :attr:`QuantityPolynomial.coefficients`
and carry the proper per-power unit ``y_unit / x_unit ** k`` instead of a
single unit for the whole coefficient array.

Only multiplicative units (no absolute temperature scales such as ``degC``),
one-dimensional arrays of finite real numbers and polynomial degrees from
0 to :data:`MAX_DEGREE` are supported in this first version.

:copyright: 2024 by Pint Authors, see AUTHORS for more details.
:license: BSD, see LICENSE for more details.
"""

from __future__ import annotations

import math
from typing import Any

from .compat import HAS_NUMPY, np
from .errors import DimensionalityError, PintError
from .util import SharedRegistryObject

#: Highest polynomial degree accepted by :meth:`QuantityPolynomial.fit`.
MAX_DEGREE = 8


class PolynomialFitError(ValueError, PintError):
    """Raised when a :class:`QuantityPolynomial` cannot be built from the
    provided samples (invalid inputs, not enough effective samples,
    identical abscissae or a rank-deficient design matrix).
    """

    def __init__(self, msg: str):
        self.msg = msg

    def __str__(self) -> str:
        return self.msg

    def __reduce__(self):
        return self.__class__, (self.msg,)


def _require_numpy() -> None:
    if not HAS_NUMPY:
        raise ImportError(
            "QuantityPolynomial requires NumPy. "
            "Install it by running:\npip install numpy"
        )


def _is_pint_object(obj: Any) -> bool:
    """Whether ``obj`` is a registry-bound Quantity or Unit.

    Kept as a function (instead of ``isinstance`` at call sites) so that
    static type checkers do not narrow away Quantity-specific attributes.
    """
    return isinstance(obj, SharedRegistryObject)


def _finite_real_1d(quantity: Any, name: str) -> np.ndarray:
    """Return a private float64 copy of a one-dimensional finite real magnitude."""
    mag = np.asarray(quantity.magnitude)
    if mag.ndim != 1:
        raise PolynomialFitError(
            f"{name} must be a one-dimensional array, got {mag.ndim} dimensions."
        )
    # Integer and floating point arrays are accepted; booleans, complex,
    # strings and object arrays are not "finite real numbers".
    if mag.dtype.kind not in "fi":
        raise PolynomialFitError(
            f"{name} must contain finite real numbers, got dtype {mag.dtype!r}."
        )
    out = np.array(mag, dtype=np.float64, copy=True)
    if not np.all(np.isfinite(out)):
        raise PolynomialFitError(
            f"{name} contains non-finite values (NaN or infinity)."
        )
    return out


def _ensure_multiplicative(quantity: Any, name: str) -> None:
    """Reject offset/logarithmic units (e.g. absolute temperature scales)."""
    non_mult = quantity._get_non_multiplicative_units()
    if non_mult:
        raise PolynomialFitError(
            f"{name} uses the non-multiplicative unit(s) {tuple(non_mult)}. "
            "Only multiplicative units are supported (use e.g. 'delta_degC' "
            "instead of 'degC' for temperature differences)."
        )


def _same_dimensionality(quantity: Any, unit: Any, name: str) -> None:
    if quantity.dimensionality != unit.dimensionality:
        raise DimensionalityError(
            quantity.units,
            unit,
            quantity.dimensionality,
            unit.dimensionality,
            extra_msg=f": {name} is not compatible with the model unit.",
        )


def _horner(coeffs: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Evaluate ``sum(coeffs[k] * t**k)`` with Horner's scheme."""
    out = np.zeros_like(t)
    for ck in coeffs[::-1]:
        out = out * t + ck
    return out


class QuantityPolynomial:
    """Degree-``n`` polynomial fitted to one-dimensional Quantity samples.

    The model is stored as ``y = sum(c_k * t**k)`` with the dimensionless
    normalized abscissa ``t = (x - origin) / scale``.  The normalized
    coefficients ``c_k`` all share ``y_unit`` because ``t`` is dimensionless,
    while the original power-basis coefficients exposed by
    :attr:`coefficients` carry ``y_unit / x_unit ** k``.

    Instances are normally created with :meth:`fit`.  Direct construction
    expects the *normalized* representation together with the Quantity
    origin and scale used for the normalization.

    Parameters
    ----------
    coefficients
        One-dimensional array of normalized coefficients, expressed as
        magnitudes in ``y_unit``. A private copy is stored.
    origin
        Scalar Quantity with the same dimensionality as ``x_unit``.
    scale
        Non-zero scalar Quantity with the same dimensionality as ``x_unit``.
    x_unit
        Display unit of the independent variable (str, Unit or UnitsContainer
        of ``registry``).
    y_unit
        Display unit of the dependent variable.
    registry
        The unit registry the model is bound to.
    """

    #: Highest polynomial degree accepted by :meth:`fit`.
    MAX_DEGREE = MAX_DEGREE

    def __init__(
        self,
        coefficients: Any,
        origin: Any,
        scale: Any,
        x_unit: Any,
        y_unit: Any,
        registry: Any,
    ) -> None:
        _require_numpy()

        if not hasattr(registry, "Quantity") or not hasattr(registry, "Unit"):
            raise TypeError("registry must be a pint UnitRegistry.")
        self._reg = registry

        xu = registry.Unit(x_unit)
        yu = registry.Unit(y_unit)
        _ensure_multiplicative(registry.Quantity(1, xu), "x_unit")
        _ensure_multiplicative(registry.Quantity(1, yu), "y_unit")

        if (
            not _is_pint_object(origin)
            or getattr(origin, "_REGISTRY", None) is not registry
        ):
            raise TypeError(
                "origin must be a Quantity from the same registry as the model."
            )
        if (
            not _is_pint_object(scale)
            or getattr(scale, "_REGISTRY", None) is not registry
        ):
            raise TypeError(
                "scale must be a Quantity from the same registry as the model."
            )
        _same_dimensionality(origin, xu, "origin")
        _same_dimensionality(scale, xu, "scale")

        origin_mag = np.asarray(origin.m_as(xu), dtype=np.float64)
        scale_mag = np.asarray(scale.m_as(xu), dtype=np.float64)
        if origin_mag.shape != () or scale_mag.shape != ():
            raise PolynomialFitError("origin and scale must be scalar Quantities.")
        if not np.isfinite(origin_mag) or not np.isfinite(scale_mag):
            raise PolynomialFitError("origin and scale must be finite Quantities.")
        if float(scale_mag) == 0.0:
            raise PolynomialFitError("scale must be a non-zero Quantity.")

        coeffs = np.array(coefficients, dtype=np.float64, copy=True)
        if coeffs.ndim != 1 or coeffs.size == 0:
            raise PolynomialFitError(
                "coefficients must be a non-empty one-dimensional array."
            )
        if not np.all(np.isfinite(coeffs)):
            raise PolynomialFitError("coefficients must all be finite.")

        self._x_unit = xu
        self._y_unit = yu
        self._origin_mag = float(origin_mag)
        self._scale_mag = float(scale_mag)
        self._coeffs = coeffs

    # ------------------------------------------------------------------
    # Fitting entry point
    # ------------------------------------------------------------------
    @classmethod
    def fit(
        cls,
        x: Any,
        y: Any,
        degree: int = 1,
        weights: Any = None,
    ) -> QuantityPolynomial:
        """Fit a polynomial to weighted one-dimensional Quantity samples.

        The abscissae are mapped to dimensionless values
        ``t = (x - origin) / scale`` using a weighted mean as origin and the
        weighted standard deviation (computed from the positive-weight
        samples) as scale.  A weighted least squares problem with explicit
        rank detection is then solved in the normalized domain.

        Parameters
        ----------
        x
            One-dimensional Quantity of finite real abscissae with
            multiplicative units.
        y
            One-dimensional Quantity of finite real ordinates, same length
            and same registry as ``x``.
        degree
            Polynomial degree, an integer between 0 and
            :data:`MAX_DEGREE` (8). The degree is never silently reduced.
        weights
            Optional one-dimensional array of dimensionless, finite,
            non-negative weights, given as a plain array or as a
            dimensionless Quantity. Points with weight zero are kept out of
            the fit, including the rank determination.

        Returns
        -------
        QuantityPolynomial

        Raises
        ------
        PolynomialFitError
            For invalid samples or units, negative weights, fewer
            positive-weight samples than coefficients, identical effective
            abscissae or a rank-deficient design matrix.
        DimensionalityError
            If dimensionful weights are supplied or units are incompatible.
        ValueError
            If ``x`` and ``y`` do not belong to the same registry.
        """
        _require_numpy()

        ureg = cls._validate_samples(x, y)

        if isinstance(degree, bool) or not isinstance(degree, (int, np.integer)):
            raise TypeError("degree must be an integer between 0 and %d." % MAX_DEGREE)
        degree = int(degree)
        if not 0 <= degree <= MAX_DEGREE:
            raise PolynomialFitError(
                f"degree must be between 0 and {MAX_DEGREE}, got {degree}."
            )

        _ensure_multiplicative(x, "x")
        _ensure_multiplicative(y, "y")

        x_mag = _finite_real_1d(x, "x")
        y_mag = _finite_real_1d(y, "y")
        if x_mag.shape != y_mag.shape:
            raise PolynomialFitError(
                f"x and y must have the same length, got {x_mag.size} and {y_mag.size}."
            )
        n_samples = x_mag.size

        w_mag = cls._validate_weights(weights, ureg, n_samples)

        n_coeffs = degree + 1
        effective = w_mag > 0.0
        n_effective = int(effective.sum())
        if n_effective < n_coeffs:
            raise PolynomialFitError(
                f"degree {degree} requires at least {n_coeffs} "
                f"positive-weight samples, got {n_effective}."
            )

        w_eff = w_mag[effective]
        x_eff = ureg.Quantity(x_mag[effective].copy(), x.units)
        y_eff = y_mag[effective].copy()
        w_sum = float(w_eff.sum())

        # Normalization quantities are built with Quantity arithmetic and
        # therefore carry the units of x throughout.
        origin = (x_eff * ureg.Quantity(w_eff, "")).sum() / w_sum
        variance = (((x_eff - origin) ** 2) * ureg.Quantity(w_eff, "")).sum() / w_sum
        scale = variance**0.5
        scale_mag = float(scale.magnitude)
        if not math.isfinite(scale_mag) or scale_mag == 0.0:
            raise PolynomialFitError(
                "all positive-weight x samples are identical; a non-zero "
                "scale is required to define the normalized abscissa."
            )

        # Dimensionless abscissae and ordinates expressed in y's own unit.
        t = ((x_eff - origin) / scale).m_as("")
        sqrt_w = np.sqrt(w_eff)

        design = np.vander(t, n_coeffs, increasing=True) * sqrt_w[:, None]
        rhs = y_eff * sqrt_w

        # Explicit, documented rank rule: a singular value counts as
        # non-zero when strictly above eps * max(n_rows, n_cols) * s_max.
        singular = np.linalg.svd(design, compute_uv=False)
        rcond = np.finfo(np.float64).eps * max(design.shape)
        rank = int(np.count_nonzero(singular > rcond * singular[0]))
        if rank < n_coeffs:
            raise PolynomialFitError(
                f"rank-deficient design matrix: rank {rank} < {n_coeffs} "
                f"required coefficients for degree {degree}. Reduce the "
                "degree or provide better distributed samples; the degree "
                "is not reduced automatically."
            )

        coeffs, *_ = np.linalg.lstsq(design, rhs, rcond=rcond)

        return cls(
            coeffs,
            origin,
            scale,
            x.units,
            y.units,
            ureg,
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _validate_samples(x: Any, y: Any) -> Any:
        if not _is_pint_object(x) or not hasattr(x, "magnitude"):
            raise TypeError("x must be a one-dimensional pint Quantity.")
        if not _is_pint_object(y) or not hasattr(y, "magnitude"):
            raise TypeError("y must be a one-dimensional pint Quantity.")
        if x._REGISTRY is not y._REGISTRY:
            raise ValueError("Cannot fit x and y Quantities from different registries.")
        return x._REGISTRY

    @staticmethod
    def _validate_weights(weights: Any, ureg: Any, n_samples: int) -> np.ndarray:
        if weights is None:
            return np.ones(n_samples, dtype=np.float64)

        if _is_pint_object(weights):
            if weights._REGISTRY is not ureg:
                raise ValueError(
                    "weights Quantity must belong to the same registry as x and y."
                )
            if not weights.dimensionless:
                raise DimensionalityError(
                    weights.units,
                    ureg.UnitsContainer(),
                    weights.dimensionality,
                    ureg.UnitsContainer(),
                    extra_msg=": weights must be dimensionless.",
                )
            raw = weights.magnitude
        else:
            raw = weights

        w = np.asarray(raw)
        if w.ndim != 1 or w.dtype.kind not in "fi":
            raise PolynomialFitError(
                "weights must be a one-dimensional array of finite real numbers."
            )
        w = np.array(w, dtype=np.float64, copy=True)
        if w.shape != (n_samples,):
            raise PolynomialFitError(
                f"weights must have length {n_samples}, got {w.size}."
            )
        if not np.all(np.isfinite(w)):
            raise PolynomialFitError(
                "weights contain non-finite values (NaN or infinity)."
            )
        if np.any(w < 0.0):
            raise PolynomialFitError("negative weights are not allowed.")
        return w

    def _canonical_unit(self, unit: Any, reference: Any, name: str) -> Any:
        """Validate a display unit: same registry, multiplicative, same dim."""
        canonical = self._reg.Unit(unit)
        _ensure_multiplicative(self._reg.Quantity(1, canonical), name)
        if canonical.dimensionality != reference.dimensionality:
            raise DimensionalityError(
                canonical,
                reference,
                canonical.dimensionality,
                reference.dimensionality,
                extra_msg=f": {name} must keep the original dimensionality.",
            )
        return canonical

    def _conversion_factor(self, src_unit: Any, dst_unit: Any) -> float:
        # Same dimensionality plus multiplicative units guarantees a pure
        # multiplicative factor; active contexts only connect different
        # dimensionalities and therefore cannot alter the conversion.
        return float(self._reg.convert(1.0, src_unit, dst_unit))

    def _prepare_x(self, x: Any) -> tuple[np.ndarray, bool]:
        if not _is_pint_object(x) or not hasattr(x, "magnitude"):
            raise TypeError("x must be a pint Quantity.")
        if x._REGISTRY is not self._reg:
            raise ValueError(
                "Cannot evaluate a QuantityPolynomial on a Quantity from "
                "another registry."
            )
        _ensure_multiplicative(x, "x")
        _same_dimensionality(x, self._x_unit, "x")

        mag = np.asarray(x.m_as(self._x_unit), dtype=np.float64)
        scalar = mag.ndim == 0
        if scalar:
            mag = mag.reshape(1)
        if mag.ndim != 1:
            raise PolynomialFitError(
                "evaluation supports scalar or one-dimensional Quantities only."
            )
        if not np.all(np.isfinite(mag)):
            raise PolynomialFitError("x contains non-finite values.")
        return mag, scalar

    # ------------------------------------------------------------------
    # Evaluation and derivative
    # ------------------------------------------------------------------
    def __call__(self, x: Any) -> Any:
        """Evaluate the model at ``x`` (scalar or one-dimensional Quantity).

        Evaluation always uses the normalized representation to avoid the
        cancellation that would result from expanding a large origin into
        high-order power-basis coefficients.
        """
        x_mag, scalar = self._prepare_x(x)
        t = (x_mag - self._origin_mag) / self._scale_mag
        y_mag = _horner(self._coeffs, t)
        if scalar:
            return self._reg.Quantity(float(y_mag[0]), self._y_unit)
        return self._reg.Quantity(y_mag, self._y_unit)

    def derivative(self) -> QuantityPolynomial:
        """Return the first derivative as another :class:`QuantityPolynomial`.

        The derivative model keeps the same origin, scale and x unit while
        its y unit becomes ``y_unit / x_unit``. For a constant model the
        result is a zero polynomial that still carries
        ``y_unit / x_unit``.
        """
        if self.degree == 0:
            d_coeffs = np.zeros(1, dtype=np.float64)
        else:
            powers = np.arange(1, self.degree + 1, dtype=np.float64)
            # c_k carries y_unit and the scale magnitude is expressed in
            # x_unit, so d_j is naturally expressed in y_unit / x_unit.
            d_coeffs = powers * self._coeffs[1:] / self._scale_mag

        derivative_y_unit = self._y_unit / self._x_unit
        origin = self._reg.Quantity(self._origin_mag, self._x_unit)
        scale = self._reg.Quantity(self._scale_mag, self._x_unit)
        return QuantityPolynomial(
            d_coeffs,
            origin,
            scale,
            self._x_unit,
            derivative_y_unit,
            self._reg,
        )

    # ------------------------------------------------------------------
    # Representation changes
    # ------------------------------------------------------------------
    def to(self, x_unit: Any = None, y_unit: Any = None) -> QuantityPolynomial:
        """Return an equivalent model using other display units.

        Both units must belong to the same registry, be multiplicative and
        keep the original dimensionality (no absolute temperature scale and
        no implicit context transformation).
        """
        new_x_unit = (
            self._x_unit
            if x_unit is None
            else self._canonical_unit(x_unit, self._x_unit, "x_unit")
        )
        new_y_unit = (
            self._y_unit
            if y_unit is None
            else self._canonical_unit(y_unit, self._y_unit, "y_unit")
        )

        x_factor = self._conversion_factor(self._x_unit, new_x_unit)
        y_factor = self._conversion_factor(self._y_unit, new_y_unit)

        new_coeffs = self._coeffs * y_factor
        new_origin = self._reg.Quantity(self._origin_mag * x_factor, new_x_unit)
        new_scale = self._reg.Quantity(self._scale_mag * x_factor, new_x_unit)

        return QuantityPolynomial(
            new_coeffs,
            new_origin,
            new_scale,
            new_x_unit,
            new_y_unit,
            self._reg,
        )

    # ------------------------------------------------------------------
    # Coefficient and parameter access
    # ------------------------------------------------------------------
    @property
    def degree(self) -> int:
        """Degree of the fitted polynomial."""
        return int(self._coeffs.size - 1)

    @property
    def x_unit(self) -> Any:
        """Display unit of the independent variable."""
        return self._x_unit

    @property
    def y_unit(self) -> Any:
        """Display unit of the dependent variable."""
        return self._y_unit

    @property
    def origin(self) -> Any:
        """Saved Quantity origin of the normalized abscissa (a fresh copy)."""
        return self._reg.Quantity(self._origin_mag, self._x_unit)

    @property
    def scale(self) -> Any:
        """Saved non-zero Quantity scale of the normalized abscissa."""
        return self._reg.Quantity(self._scale_mag, self._x_unit)

    @property
    def normalized_coefficients(self) -> Any:
        """Coefficients ``c_k`` of the ``t`` basis as a Quantity array in
        ``y_unit`` (a private copy; mutating it does not affect the model).
        """
        return self._reg.Quantity(self._coeffs.copy(), self._y_unit)

    @property
    def coefficients(self) -> tuple[Any, ...]:
        """Coefficients of the original ``x`` power basis.

        The returned tuple has ``degree + 1`` scalar Quantities, where
        element ``k`` multiplies ``x ** k`` and carries the unit
        ``y_unit / x_unit ** k``. These coefficients are computed on demand
        from the normalized representation and are never used internally for
        evaluation, because expanding a large origin would reintroduce
        severe numerical cancellation.
        """
        magnitudes = self._power_basis_magnitudes()
        return tuple(
            self._reg.Quantity(
                float(magnitudes[k]),
                self._y_unit / self._x_unit**k,
            )
            for k in range(self.degree + 1)
        )

    def _power_basis_magnitudes(self) -> np.ndarray:
        """Expand ``sum c_k ((x - o)/s)**k`` into ``sum a_j x**j``."""
        n = self.degree
        expanded = np.zeros(n + 1, dtype=np.float64)
        origin = self._origin_mag
        scale = self._scale_mag
        for k in range(n + 1):
            c_k = self._coeffs[k]
            scale_k = scale**k
            for j in range(k + 1):
                expanded[j] += c_k * math.comb(k, j) * ((-origin) ** (k - j)) / scale_k
        return expanded

    def __repr__(self) -> str:
        return (
            f"<QuantityPolynomial(degree={self.degree}, "
            f"x_unit='{self._x_unit}', y_unit='{self._y_unit}', "
            f"origin={self._origin_mag} {self._x_unit}, "
            f"scale={self._scale_mag} {self._x_unit})>"
        )

    def __getstate__(self) -> dict[str, Any]:
        # Registries are not picklable, so only the unit containers and the
        # numerical state are serialized.
        return {
            "coeffs": self._coeffs.copy(),
            "origin_mag": self._origin_mag,
            "scale_mag": self._scale_mag,
            "x_units": self._x_unit._units,
            "y_units": self._y_unit._units,
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        # Mirror Quantity unpickling: rebind to the application registry,
        # which must define all units referenced by the model.
        from . import application_registry

        reg: Any = application_registry.get()
        self.__init__(
            state["coeffs"],
            reg.Quantity(state["origin_mag"], state["x_units"]),
            reg.Quantity(state["scale_mag"], state["x_units"]),
            state["x_units"],
            state["y_units"],
            reg,
        )
