"""Tests for the dimensional QuantityPolynomial model and its fitting entry.

Expected values are always derived from independent analytic formulas or
exact small rational examples, never by stripping units and reusing the
implementation under test.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

import pint
from pint import (
    MAX_POLYNOMIAL_DEGREE,
    DimensionalityError,
    PolynomialFitError,
    QuantityPolynomial,
)
from pint.testsuite import helpers


@pytest.fixture(scope="module")
def ureg():
    return pint.UnitRegistry()


@pytest.fixture(scope="module")
def Q(ureg):
    return ureg.Quantity


# ---------------------------------------------------------------------------
# Exact, analytically known examples
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_exact_quadratic_three_points(Q):
    # Hand-solved interpolation through (0, 1), (1, 3), (2, 9):
    # c = 1; a + b = 2; 2a + b = 4  ->  y = 2 x**2 + 1 (b = 0).
    x = Q([0.0, 1.0, 2.0], "m")
    y = Q([1.0, 3.0, 9.0], "J")

    model = QuantityPolynomial.fit(x, y, 2)

    assert model.degree == 2
    coeffs = model.coefficients
    assert len(coeffs) == 3
    np.testing.assert_allclose(
        [c.magnitude for c in coeffs], [1.0, 0.0, 2.0], rtol=1e-12
    )
    # Every power carries its own unit: y_unit / x_unit ** k.
    assert coeffs[0].units == Q(1, "J").units
    assert coeffs[1].units == Q(1, "J/m").units
    assert coeffs[2].units == Q(1, "J/m**2").units
    assert coeffs[2].dimensionality == Q(1, "J/m**2").dimensionality


@helpers.requires_numpy
def test_known_dimensional_quadratic_predictions_and_derivative(Q):
    # y(x) = 2 J/m^2 * x^2 + 3 J/m * x + 5 J, sampled on a non-symmetric grid.
    x_data = Q(np.linspace(-2.0, 3.0, 7), "m")
    y_data = (
        2.0 * Q(1, "J/m**2") * x_data**2 + 3.0 * Q(1, "J/m") * x_data + 5.0 * Q(1, "J")
    )
    model = QuantityPolynomial.fit(x_data, y_data, 2)

    # Recovered power-basis coefficients match the analytic ones.
    expected_magnitudes = [5.0, 3.0, 2.0]
    np.testing.assert_allclose(
        [c.m_as("J/m**%d" % k) for k, c in enumerate(model.coefficients)],
        expected_magnitudes,
        rtol=1e-10,
    )

    # Fresh evaluation points, compared against the independent analytic
    # Quantity expression (both scalar and array evaluation).
    x_eval = Q(np.array([-1.7, -0.25, 0.5, 1.33, 2.8]), "m")
    y_expected = (
        2.0 * Q(1, "J/m**2") * x_eval**2 + 3.0 * Q(1, "J/m") * x_eval + 5.0 * Q(1, "J")
    )
    np.testing.assert_allclose(
        model(x_eval).m_as("J"), y_expected.m_as("J"), rtol=1e-12
    )

    scalar = model(Q(0.5, "m"))
    assert np.ndim(scalar.magnitude) == 0
    assert np.isclose(float(scalar.magnitude), 7.0, rtol=1e-12)
    assert scalar.units == Q(1, "J").units

    # First derivative: dy/dx = 4 J/m^2 * x + 3 J/m, unit y_unit / x_unit.
    derivative = model.derivative()
    assert derivative.y_unit == Q(1, "J/m").units
    d_expected = 4.0 * Q(1, "J/m**2") * x_eval + 3.0 * Q(1, "J/m")
    np.testing.assert_allclose(
        derivative(x_eval).m_as("J/m"), d_expected.m_as("J/m"), rtol=1e-12
    )
    assert derivative(Q(0.5, "m")).units == Q(1, "J/m").units

    # Derivative power-basis coefficients carry J/m**(k+1).
    d_coeffs = derivative.coefficients
    np.testing.assert_allclose([c.magnitude for c in d_coeffs], [3.0, 4.0], rtol=1e-10)
    assert d_coeffs[0].units == Q(1, "J/m").units
    assert d_coeffs[1].units == Q(1, "J/m**2").units


@helpers.requires_numpy
def test_normalized_representation_is_dimensionless(Q):
    x_data = Q(np.linspace(1.0, 4.0, 5), "m")
    y_data = 2.0 * Q(1, "J/m") * x_data + 1.0 * Q(1, "J")
    model = QuantityPolynomial.fit(x_data, y_data, 1)

    # Origin and scale are Quantities in the x unit, scale is non-zero.
    assert model.origin.units == Q(1, "m").units
    assert model.scale.units == Q(1, "m").units
    assert float(model.scale.magnitude) > 0.0

    # All normalized coefficients share the single y unit (t is dimensionless).
    nc = model.normalized_coefficients
    assert nc.units == Q(1, "J").units
    assert nc.magnitude.shape == (2,)

    # The stored origin equals the (unweighted) mean of the samples.
    assert np.isclose(float(model.origin.magnitude), 2.5, rtol=1e-14)


@helpers.requires_numpy
def test_constant_model_and_zero_derivative_unit(Q):
    x_data = Q(np.linspace(0.0, 10.0, 6), "m")
    y_data = Q(np.full(6, 2.5), "J")
    model = QuantityPolynomial.fit(x_data, y_data, 0)

    assert model.degree == 0
    assert np.isclose(float(model.coefficients[0].magnitude), 2.5, rtol=1e-14)
    assert model.coefficients[0].units == Q(1, "J").units

    derivative = model.derivative()
    zero = derivative(Q(3.0, "m"))
    assert float(zero.magnitude) == 0.0
    # The zero derivative keeps the proper y_unit / x_unit.
    assert zero.units == Q(1, "J/m").units
    assert derivative.y_unit == Q(1, "J/m").units
    np.testing.assert_array_equal(derivative(Q([0.0, 5.0], "m")).magnitude, [0.0, 0.0])


# ---------------------------------------------------------------------------
# Weighted fitting, verified against independent weighted formulas
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_weighted_fit_matches_independent_weighted_formulas(Q):
    # Exact line y = 2 x + 1 with unequal weights; derive the weighted least
    # squares answer directly from the definitions instead of the model.
    x_mag = np.array([0.0, 1.0, 2.0, 3.0])
    y_mag = 2.0 * x_mag + 1.0
    w = np.array([1.0, 2.0, 1.0, 2.0])
    x = Q(x_mag, "m")
    y = Q(y_mag, "J")

    model = QuantityPolynomial.fit(x, y, 1, weights=w)

    w_sum = w.sum()
    x_bar = (w * x_mag).sum() / w_sum
    y_bar = (w * y_mag).sum() / w_sum
    slope = (w * (x_mag - x_bar) * (y_mag - y_bar)).sum() / (
        w * (x_mag - x_bar) ** 2
    ).sum()
    intercept = y_bar - slope * x_bar

    np.testing.assert_allclose(model.coefficients[0].m_as("J"), intercept, rtol=1e-12)
    np.testing.assert_allclose(model.coefficients[1].m_as("J/m"), slope, rtol=1e-12)

    # A dimensionless Quantity of weights must behave identically.
    model_q = QuantityPolynomial.fit(x, y, 1, weights=Q(w, ""))
    np.testing.assert_allclose(
        model_q.normalized_coefficients.magnitude,
        model.normalized_coefficients.magnitude,
        rtol=1e-14,
    )


@helpers.requires_numpy
def test_zero_weight_outlier_is_ignored(Q):
    # Truth: y = 2 x**2 + 3 x + 5; the last point is a wild outlier with
    # zero weight, so it must not influence the fit or the rank check.
    x = Q(np.array([0.0, 1.0, 2.0, 1.5]), "m")
    y = Q(np.array([5.0, 10.0, 19.0, 9999.0]), "J")
    weights = np.array([1.0, 1.0, 1.0, 0.0])

    model = QuantityPolynomial.fit(x, y, 2, weights=weights)
    clean = QuantityPolynomial.fit(x[:3], y[:3], 2)

    np.testing.assert_allclose(
        model.normalized_coefficients.magnitude,
        clean.normalized_coefficients.magnitude,
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        [c.magnitude for c in model.coefficients],
        [5.0, 3.0, 2.0],
        rtol=1e-10,
    )

    # Zero-weight points do not count toward the effective sample size:
    # degree 2 with exactly three positive-weight samples is valid.
    x_eval = Q(1.5, "m")
    assert np.isclose(float(model(x_eval).magnitude), 14.0, rtol=1e-10)


@helpers.requires_numpy
def test_zero_weights_rank_and_count_rules(Q):
    x = Q([0.0, 1.0, 2.0, 3.0], "m")
    y = Q([1.0, 2.0, 4.0, 8.0], "J")

    # Four samples but only two positive weights: not enough for degree 2.
    with pytest.raises(PolynomialFitError, match="positive-weight samples"):
        QuantityPolynomial.fit(x, y, 2, weights=np.array([1.0, 0.0, 1.0, 0.0]))

    # The same two weights still allow a degree-1 fit.
    model = QuantityPolynomial.fit(x, y, 1, weights=np.array([1.0, 0.0, 1.0, 0.0]))
    assert model.degree == 1


# ---------------------------------------------------------------------------
# Numerical stability with a large origin
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_large_origin_small_range_stays_accurate(Q):
    offset = 1.0e9
    local = np.linspace(-1.0, 1.0, 9)
    x_data = Q(offset + local, "m")
    y_data = (
        2.0 * Q(1, "J/m**2") * x_data**2 + 3.0 * Q(1, "J/m") * x_data + 5.0 * Q(1, "J")
    )
    model = QuantityPolynomial.fit(x_data, y_data, 2)

    x_eval = Q(offset + np.array([-0.37, 0.11, 0.83]), "m")
    y_expected = (
        2.0 * Q(1, "J/m**2") * x_eval**2 + 3.0 * Q(1, "J/m") * x_eval + 5.0 * Q(1, "J")
    )
    got = model(x_eval).m_as("J")
    rel_err = np.abs(got - y_expected.m_as("J")) / np.abs(y_expected.m_as("J"))
    assert rel_err.max() < 1e-13

    # A naive evaluation using the expanded power-basis coefficients in
    # meters suffers from catastrophic cancellation at this origin; the
    # model's normalized evaluation must clearly outperform it.
    a = np.array([c.m_as("J/m**%d" % k) for k, c in enumerate(model.coefficients)])
    x_raw = x_eval.m_as("m")
    naive = a[0] + a[1] * x_raw + a[2] * x_raw**2
    naive_rel = np.abs(naive - y_expected.m_as("J")) / np.abs(y_expected.m_as("J"))
    assert naive_rel.max() > rel_err.max()


# ---------------------------------------------------------------------------
# Display unit changes
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_display_unit_changes_keep_predictions_and_derivatives_consistent(Q):
    x_data = Q(np.linspace(0.0, 5.0, 6), "s")
    y_data = (
        9.81 * Q(1, "m/s**2") * x_data**2 + 2.0 * Q(1, "m/s") * x_data + 0.5 * Q(1, "m")
    )
    model = QuantityPolynomial.fit(x_data, y_data, 2)
    x_eval = Q(np.array([0.4, 1.3, 2.1]), "s")
    y_expected = (
        9.81 * Q(1, "m/s**2") * x_eval**2 + 2.0 * Q(1, "m/s") * x_eval + 0.5 * Q(1, "m")
    )
    d_expected = 2.0 * 9.81 * Q(1, "m/s**2") * x_eval + 2.0 * Q(1, "m/s")

    variants = [
        model.to("ms"),
        model.to(y_unit="km"),
        model.to("ms", "km"),
    ]
    for converted in variants:
        xu = str(converted.x_unit)
        yu = str(converted.y_unit)
        predictions = converted(x_eval.to(xu))
        np.testing.assert_allclose(
            predictions.m_as(yu), y_expected.m_as(yu), rtol=1e-11
        )
        derivative_values = converted.derivative()(x_eval.to(xu))
        d_unit = converted.y_unit / converted.x_unit
        np.testing.assert_allclose(
            derivative_values.m_as(d_unit),
            d_expected.m_as(d_unit),
            rtol=1e-11,
        )
        # Power-basis coefficients keep per-power units in the new display.
        for k, c in enumerate(converted.coefficients):
            assert c.units == converted.y_unit / converted.x_unit**k


@helpers.requires_numpy
def test_km_and_meter_coefficient_equivalence(Q):
    # Fit a length-in-length polynomial and compare coefficients physically
    # after switching the x display unit from meters to kilometers.
    x_data = Q(np.array([1000.0, 2000.0, 3000.0]), "m")
    y_data = Q(np.array([10.0, 40.0, 90.0]), "m")  # y = x_km**2 + ... see below
    model_m = QuantityPolynomial.fit(x_data, y_data, 2)
    model_km = model_m.to("km")

    # In kilometers the abscissae are 1, 2, 3 and the polynomial is
    # y[m] = 10 * x_km**2, so the km-basis quadratic coefficient is 10 m/km^2.
    np.testing.assert_allclose(
        model_km.coefficients[2].m_as("m/km**2"), 10.0, rtol=1e-10
    )
    # The lower-order coefficients vanish in the km representation.
    assert abs(model_km.coefficients[0].m_as("m")) < 1e-10
    assert abs(model_km.coefficients[1].m_as("m/km")) < 1e-10
    # Physical coefficient values agree when converted back to the same unit.
    for k in range(3):
        common = "m/m**%d" % k
        np.testing.assert_allclose(
            model_km.coefficients[k].m_as(common),
            model_m.coefficients[k].m_as(common),
            rtol=1e-9,
            atol=1e-12,
        )


@helpers.requires_numpy
def test_to_rejects_wrong_dimensionality_and_offset_units(Q):
    x_data = Q([0.0, 1.0, 2.0], "m")
    y_data = Q([1.0, 2.0, 3.0], "J")
    model = QuantityPolynomial.fit(x_data, y_data, 1)

    with pytest.raises(DimensionalityError):
        model.to("s")
    with pytest.raises(DimensionalityError):
        model.to(y_unit="m")
    with pytest.raises(PolynomialFitError, match="non-multiplicative"):
        model.to(y_unit="degC")


@helpers.requires_numpy
def test_no_implicit_context_conversion(Q, ureg):
    x_data = Q([0.0, 1.0, 2.0], "m")
    y_data = Q([1.0, 2.0, 3.0], "J")
    model = QuantityPolynomial.fit(x_data, y_data, 1)

    # Even with a context that relates energy and frequency active, switching
    # to an incompatible dimensionality must fail rather than transform.
    with ureg.context("spectroscopy"):
        with pytest.raises(DimensionalityError):
            model.to(y_unit="hertz")
        # Same-dimension conversion remains an ordinary factor conversion.
        converted = model.to(y_unit="kJ")
        np.testing.assert_allclose(
            converted(x_data.to("m")).m_as("kJ"),
            y_data.m_as("kJ"),
            rtol=1e-12,
        )


# ---------------------------------------------------------------------------
# Input validation and explicit error rules
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_degree_rules(Q):
    x = Q([0.0, 1.0, 2.0], "m")
    y = Q([1.0, 2.0, 3.0], "J")

    QuantityPolynomial.fit(x, y, 0)
    # Degree 8 needs nine positive-weight samples.
    QuantityPolynomial.fit(
        Q(np.linspace(0.0, 1.0, 9), "m"),
        Q(np.linspace(0.0, 1.0, 9), "J"),
        MAX_POLYNOMIAL_DEGREE,
    )
    assert QuantityPolynomial.MAX_DEGREE == MAX_POLYNOMIAL_DEGREE == 8

    with pytest.raises(PolynomialFitError, match="degree must be between"):
        QuantityPolynomial.fit(x, y, 9)
    with pytest.raises(PolynomialFitError):
        QuantityPolynomial.fit(x, y, -1)
    with pytest.raises(TypeError):
        QuantityPolynomial.fit(x, y, 1.0)
    with pytest.raises(TypeError):
        QuantityPolynomial.fit(x, y, True)


@helpers.requires_numpy
def test_negative_weights_rejected(Q):
    x = Q([0.0, 1.0, 2.0], "m")
    y = Q([1.0, 2.0, 3.0], "J")
    with pytest.raises(PolynomialFitError, match="negative weights"):
        QuantityPolynomial.fit(x, y, 1, weights=np.array([1.0, -1.0, 1.0]))


@helpers.requires_numpy
def test_insufficient_samples_rejected(Q):
    x = Q([0.0, 1.0], "m")
    y = Q([1.0, 2.0], "J")
    with pytest.raises(PolynomialFitError, match="positive-weight samples"):
        QuantityPolynomial.fit(x, y, 2)


@helpers.requires_numpy
def test_identical_x_rejected(Q):
    x = Q([2.0, 2.0, 2.0], "m")
    y = Q([1.0, 2.0, 3.0], "J")
    with pytest.raises(PolynomialFitError, match="identical"):
        QuantityPolynomial.fit(x, y, 1)


@helpers.requires_numpy
def test_rank_deficiency_reported_without_silent_reduction(Q):
    # 1 + 1e-16 rounds to 1.0 in float64: three samples but only two
    # distinguishable abscissae, so the quadratic design is rank 2 < 3.
    x = Q([0.0, 1.0, 1.0 + 1.0e-16], "m")
    y = Q([0.0, 1.0, 2.0], "J")
    with pytest.raises(PolynomialFitError, match="rank-deficient"):
        QuantityPolynomial.fit(x, y, 2)


@helpers.requires_numpy
def test_nonmultiplicative_units_rejected_but_delta_units_accepted(Q):
    with pytest.raises(PolynomialFitError, match="non-multiplicative"):
        QuantityPolynomial.fit(Q([1.0, 2.0, 3.0], "degC"), Q([1.0, 2.0, 3.0], "m"), 1)
    with pytest.raises(PolynomialFitError, match="non-multiplicative"):
        QuantityPolynomial.fit(Q([1.0, 2.0, 3.0], "m"), Q([1.0, 2.0, 3.0], "degF"), 1)

    # Temperature differences (delta units) are multiplicative and valid.
    x = Q([0.0, 1.0, 2.0], "delta_degC")
    y = Q([1.0, 3.0, 6.0], "delta_degC")
    model = QuantityPolynomial.fit(x, y, 2)
    assert model(x).units == Q(1, "delta_degC").units


@helpers.requires_numpy
def test_registry_mismatch_rejected(Q):
    other = pint.UnitRegistry()
    x = Q([0.0, 1.0, 2.0], "m")
    y_other = other.Quantity([1.0, 2.0, 3.0], "joule")
    with pytest.raises(ValueError, match="different registries"):
        QuantityPolynomial.fit(x, y_other, 1)

    model = QuantityPolynomial.fit(x, Q([1.0, 2.0, 3.0], "J"), 1)
    with pytest.raises(ValueError, match="another registry"):
        model(other.Quantity(1.0, "m"))

    with pytest.raises(ValueError, match="same registry"):
        QuantityPolynomial.fit(
            x, Q([1.0, 2.0, 3.0], "J"), 1, weights=other.Quantity([1.0, 1.0, 1.0], "")
        )


@helpers.requires_numpy
def test_invalid_sample_shapes_and_values(Q):
    good_x = Q([0.0, 1.0, 2.0], "m")
    good_y = Q([1.0, 2.0, 3.0], "J")

    with pytest.raises(TypeError):
        QuantityPolynomial.fit(np.array([0.0, 1.0, 2.0]), good_y, 1)
    with pytest.raises(TypeError):
        QuantityPolynomial.fit(good_x, np.array([1.0, 2.0, 3.0]), 1)

    with pytest.raises(PolynomialFitError, match="one-dimensional"):
        QuantityPolynomial.fit(
            Q([[0.0, 1.0], [2.0, 3.0]], "m"),
            Q([1.0, 2.0, 3.0, 4.0], "J"),
            1,
        )
    with pytest.raises(PolynomialFitError, match="non-finite"):
        QuantityPolynomial.fit(Q([0.0, np.nan, 2.0], "m"), good_y, 1)
    with pytest.raises(PolynomialFitError, match="non-finite"):
        QuantityPolynomial.fit(good_x, Q([1.0, np.inf, 3.0], "J"), 1)
    with pytest.raises(PolynomialFitError, match="real"):
        QuantityPolynomial.fit(
            Q([0.0, 1.0, 2.0], "m"),
            Q(np.array([1.0 + 0j, 2.0, 3.0]), "J"),
            1,
        )
    with pytest.raises(PolynomialFitError, match="same length"):
        QuantityPolynomial.fit(Q([0.0, 1.0, 2.0], "m"), Q([1.0, 2.0], "J"), 1)


@helpers.requires_numpy
def test_weight_validation(Q):
    x = Q([0.0, 1.0, 2.0], "m")
    y = Q([1.0, 2.0, 3.0], "J")

    with pytest.raises(DimensionalityError):
        QuantityPolynomial.fit(x, y, 1, weights=Q([1.0, 1.0, 1.0], "m"))
    with pytest.raises(PolynomialFitError, match="length"):
        QuantityPolynomial.fit(x, y, 1, weights=np.array([1.0, 1.0]))
    with pytest.raises(PolynomialFitError, match="non-finite"):
        QuantityPolynomial.fit(x, y, 1, weights=np.array([1.0, np.nan, 1.0]))
    with pytest.raises(PolynomialFitError, match="one-dimensional"):
        QuantityPolynomial.fit(x, y, 1, weights=np.ones((3, 1)))


@helpers.requires_numpy
def test_evaluation_dimensionality_check(Q):
    x = Q([0.0, 1.0, 2.0], "m")
    y = Q([1.0, 2.0, 3.0], "J")
    model = QuantityPolynomial.fit(x, y, 1)

    with pytest.raises(DimensionalityError):
        model(Q(1.0, "s"))
    with pytest.raises(TypeError):
        model(1.0)


# ---------------------------------------------------------------------------
# Immutability / defensive copies
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_external_mutation_does_not_change_model(Q):
    x_mag = np.linspace(-2.0, 3.0, 7)
    y_mag = 2.0 * x_mag**2 + 5.0
    w_mag = np.ones_like(x_mag)

    x = Q(x_mag, "m")
    y = Q(y_mag, "J")
    weights = Q(w_mag, "")
    model = QuantityPolynomial.fit(x, y, 2, weights=weights)

    before = float(model(Q(0.7, "m")).magnitude)
    x_mag[:] = 999.0
    x.magnitude[:] = 999.0
    y.magnitude[:] = 999.0
    weights.magnitude[:] = 0.0
    after = float(model(Q(0.7, "m")).magnitude)
    assert before == after

    # Returned coefficient/parameter objects are independent copies.
    nc = model.normalized_coefficients
    nc.magnitude[:] = 0.0
    assert np.all(model.normalized_coefficients.magnitude != 0.0)

    origin = model.origin
    origin.ito("km")
    assert model.origin.units == Q(1, "m").units
    scale = model.scale
    scale.ito("km")
    assert model.scale.units == Q(1, "m").units


# ---------------------------------------------------------------------------
# Coverage of remaining supported behavior
# ---------------------------------------------------------------------------


@helpers.requires_numpy
def test_degree_eight_supported(Q):
    x_mag = np.linspace(-1.0, 1.0, 11)
    # A degree-5 truth is representable by a degree-8 fit.
    y_mag = 1.0 + 0.5 * x_mag - 0.25 * x_mag**3 + x_mag**5
    model = QuantityPolynomial.fit(Q(x_mag, "m"), Q(y_mag, "J"), 8)
    assert model.degree == 8
    np.testing.assert_allclose(
        model(Q(x_mag, "m")).m_as("J"), y_mag, rtol=1e-9, atol=1e-10
    )


@helpers.requires_numpy
def test_integer_inputs_promoted_and_fit(Q):
    model = QuantityPolynomial.fit(Q([0, 1, 2, 3], "m"), Q([1, 3, 7, 13], "J"), 2)
    # Truth is y = x**2 + x + 1 on these integer samples.
    np.testing.assert_allclose(
        [c.magnitude for c in model.coefficients],
        [1.0, 1.0, 1.0],
        rtol=1e-10,
    )
    assert model.normalized_coefficients.magnitude.dtype == np.float64


@helpers.requires_numpy
def test_pickle_roundtrip(ureg):
    # Unpickling rebinds to the application registry, like Quantity does.
    previous = pint.get_application_registry()
    pint.set_application_registry(ureg)
    try:
        Q = ureg.Quantity
        x = Q(np.linspace(0.0, 2.0, 5), "m")
        y = 2.0 * Q(1, "J/m**2") * x**2 + 3.0 * Q(1, "J/m") * x + 5.0 * Q(1, "J")
        model = QuantityPolynomial.fit(x, y, 2)

        restored = pickle.loads(pickle.dumps(model))
        x_eval = Q(np.array([0.3, 1.4]), "m")
        np.testing.assert_allclose(
            restored(x_eval).m_as("J"), model(x_eval).m_as("J"), rtol=1e-14
        )
        assert restored.x_unit == model.x_unit
        assert restored.y_unit == model.y_unit
    finally:
        pint.set_application_registry(previous)


@helpers.requires_numpy
def test_dimensionless_quantities_are_supported(Q):
    # Independent least-squares solution for (0, 2), (1, 4), (2, 8):
    # slope 3, intercept 5/3, so y(3) = 32/3 and y(4) = 41/3.
    x = Q([0.0, 1.0, 2.0], "")
    y = Q([2.0, 4.0, 8.0], "")
    model = QuantityPolynomial.fit(x, y, 1)
    assert model(Q(3.0, "")).units == Q(1, "").units
    np.testing.assert_allclose(
        model(Q([3.0, 4.0], "")).magnitude,
        [32.0 / 3.0, 41.0 / 3.0],
        rtol=1e-12,
    )
    assert model.derivative().y_unit.dimensionless
