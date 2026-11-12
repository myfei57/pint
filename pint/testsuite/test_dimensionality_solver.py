"""Tests for the symbolic dimensionality constraint solver.

The solver infers base-dimension exponents of unknown symbols from labelled
algebraic equations using an exact rational linear system. These tests cover
the physical textbook chains, under-determination with a null-space basis,
exact contradiction detection, the restricted grammar and the boundary of the
inference space.
"""

from __future__ import annotations

from fractions import Fraction
from itertools import permutations

import pytest

from pint import DimensionalitySolution, UnitRegistry, solve_dimensionalities
from pint.dimensionality import _Parser
from pint.errors import DefinitionSyntaxError
from pint.util import UnitsContainer

# ---------------------------------------------------------------------------
# Independent re-substitution machinery
# ---------------------------------------------------------------------------


def _fraction_map(container) -> dict[str, Fraction]:
    return {str(dim): Fraction(value) for dim, value in container.items()}


def _known_environment(known, registry):
    env = {}
    for name, value in known.items():
        if hasattr(value, "dimensionality"):
            env[name] = _fraction_map(value.dimensionality)
        elif isinstance(value, dict) and (
            not value or all(str(key).startswith("[") for key in value)
        ):
            env[name] = {dim: Fraction(exp) for dim, exp in value.items()}
        else:
            env[name] = _fraction_map(registry.get_dimensionality(value))
    return env


def _merge(target: dict[str, Fraction], other: dict[str, Fraction], sign: int):
    for dim, exponent in other.items():
        target[dim] = target.get(dim, Fraction(0)) + sign * exponent
    return {dim: exp for dim, exp in target.items() if exp != 0}


def _evaluate(node, env: dict[str, dict[str, Fraction]]) -> dict[str, Fraction]:
    """Independently evaluate an AST against an exponent environment."""
    kind = node[0]
    if kind == "num":
        # Numeric literals are always dimensionless.
        return {}
    if kind == "name":
        name = node[1]
        if name.startswith("["):
            # Bracket dimension atoms are self-describing constants.
            return {} if name == "[]" else {name: Fraction(1)}
        return dict(env[name])
    if kind == "neg":
        return _evaluate(node[1], env)
    if kind == "mul":
        return _merge(_evaluate(node[1], env), _evaluate(node[2], env), 1)
    if kind == "div":
        return _merge(_evaluate(node[1], env), _evaluate(node[2], env), -1)
    if kind == "pow":
        base = _evaluate(node[1], env)
        exponent: Fraction = node[2]
        return {
            dim: value * exponent for dim, value in base.items() if value * exponent
        }
    if kind in ("add", "sub"):
        left = _evaluate(node[1], env)
        right = _evaluate(node[2], env)
        # Addends must be dimensionally identical, exactly.
        assert left == right
        return left
    raise AssertionError(f"unexpected node {kind!r}")  # pragma: no cover


def _resubstitute(equations, env, registry) -> None:
    """Re-evaluate every original equation and assert exact zero residual."""
    for _label, expression in equations:
        node = _Parser(expression).parse_equation()
        _, left, right = node
        left_map = _evaluate(left, env)
        right_map = _evaluate(right, env)
        assert left_map == right_map


def _solution_environment(
    solution: DimensionalitySolution, known, unknowns, registry, coefficients=()
):
    env = _known_environment(known, registry)
    for symbol in unknowns:
        combined: dict[str, Fraction] = _fraction_map(
            solution.particular_solution[symbol]
        )
        for coefficient, basis_vector in zip(coefficients, solution.nullspace_basis):
            for dim, exponent in basis_vector[symbol].items():
                combined[dim] = combined.get(dim, Fraction(0)) + (
                    coefficient * Fraction(exponent)
                )
        env[symbol] = {dim: exp for dim, exp in combined.items() if exp != 0}
    return env


def _assert_exact_containers(solution: DimensionalitySolution, unknowns) -> None:
    containers = [solution.particular_solution[name] for name in unknowns]
    containers += [
        vector[name] for vector in solution.nullspace_basis for name in unknowns
    ]
    for container in containers:
        assert isinstance(container, UnitsContainer)
        for exponent in container.values():
            assert isinstance(exponent, (int, Fraction))
            assert not isinstance(exponent, float)


# ---------------------------------------------------------------------------
# Textbook physical chains
# ---------------------------------------------------------------------------


def test_force_mass_acceleration_chain(sess_registry):
    equations = [("force", "F = m * a"), ("kinematics", "a = v / t")]
    known = {
        "F": "newton",
        "a": "meter / second**2",
        "t": "second",
    }
    solution = solve_dimensionalities(
        equations, known, ("m", "v"), registry=sess_registry
    )

    assert solution.is_unique
    assert solution.free_degrees == 0
    assert solution.nullspace_basis == ()
    assert solution.conflicting_equation_labels == ()
    assert solution.base_dimensions == ("[mass]", "[length]", "[time]")

    assert dict(solution.particular_solution["m"]) == {"[mass]": 1}
    assert dict(solution.particular_solution["v"]) == {
        "[length]": 1,
        "[time]": -1,
    }
    _assert_exact_containers(solution, ("m", "v"))

    env = _solution_environment(solution, known, ("m", "v"), sess_registry)
    _resubstitute(equations, env, sess_registry)


def test_infer_mass_alone_with_velocity_known(sess_registry):
    equations = [("e1", "F = m * a"), ("e2", "a = v / t")]
    known = {
        "F": sess_registry.newton,
        "a": sess_registry.meter / sess_registry.second**2,
        "t": sess_registry.second,
        "v": sess_registry.meter / sess_registry.second,
    }
    solution = solve_dimensionalities(equations, known, ("m",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["m"]) == {"[mass]": 1}
    env = _solution_environment(solution, known, ("m",), sess_registry)
    _resubstitute(equations, env, sess_registry)


def test_repeated_symbol_shares_one_unknown(sess_registry):
    # If every leaf got its own unknown this would be under-determined; the
    # shared variable makes the exponent uniquely equal to 1/2.
    equations = [("e1", "x = y * y")]
    known = {"x": UnitsContainer({"[length]": 1})}
    solution = solve_dimensionalities(equations, known, ("y",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["y"]) == {"[length]": Fraction(1, 2)}
    env = _solution_environment(solution, known, ("y",), sess_registry)
    _resubstitute(equations, env, sess_registry)


# ---------------------------------------------------------------------------
# Under-determination: particular solution, basis and free degrees
# ---------------------------------------------------------------------------


def test_cancellation_leaves_unknown_free(sess_registry):
    equations = [("e1", "y = x / x")]
    known = {"y": "dimensionless"}
    solution = solve_dimensionalities(
        equations,
        known,
        ("x",),
        registry=sess_registry,
        extra_base_dimensions=("[length]",),
    )

    assert solution.is_underdetermined
    assert solution.free_degrees == 1
    assert solution.base_dimensions == ("[length]",)
    assert dict(solution.particular_solution["x"]) == {}
    assert len(solution.nullspace_basis) == 1
    assert dict(solution.nullspace_basis[0]["x"]) == {"[length]": 1}

    for coefficient in (Fraction(0), Fraction(1), Fraction(-7, 3), Fraction(12)):
        env = _solution_environment(
            solution, known, ("x",), sess_registry, (coefficient,)
        )
        _resubstitute(equations, env, sess_registry)


def test_two_free_degrees_with_basis(sess_registry):
    # a = b with no known dimensionality: two symbols over two dimensions,
    # rank two, hence two free degrees of freedom.
    equations = [("e1", "a = b")]
    solution = solve_dimensionalities(
        equations,
        {},
        ("a", "b"),
        registry=sess_registry,
        extra_base_dimensions=("[length]", "[time]"),
    )
    assert solution.is_underdetermined
    assert solution.free_degrees == 2
    assert len(solution.nullspace_basis) == 2

    # The basis vectors must be linearly independent.
    basis_maps = [
        {name: _fraction_map(vector[name]) for name in ("a", "b")}
        for vector in solution.nullspace_basis
    ]
    assert basis_maps[0] != basis_maps[1]

    coefficients = (Fraction(2, 5), Fraction(-4))
    env = _solution_environment(solution, {}, ("a", "b"), sess_registry, coefficients)
    _resubstitute(equations, env, sess_registry)


def test_particular_plus_basis_always_satisfies(sess_registry):
    equations = [
        ("e1", "z = x * y"),
        ("e2", "w = x * y"),
    ]
    known = {
        "z": UnitsContainer({"[length]": 1}),
        "w": UnitsContainer({"[length]": 1}),
    }
    solution = solve_dimensionalities(
        equations, known, ("x", "y"), extra_base_dimensions=("[length]",)
    )
    assert solution.is_underdetermined
    assert solution.free_degrees == 1

    # Homogeneous check: each basis vector must satisfy the equations with the
    # known symbols made dimensionless.
    for basis_vector in solution.nullspace_basis:
        homogeneous_env = {name: {} for name in known}
        for name in ("x", "y"):
            homogeneous_env[name] = _fraction_map(basis_vector[name])
        _resubstitute(equations, homogeneous_env, sess_registry)

    for coefficient in (Fraction(0), Fraction(1), Fraction(9, 2), Fraction(-3, 7)):
        env = _solution_environment(
            solution, known, ("x", "y"), sess_registry, (coefficient,)
        )
        _resubstitute(equations, env, sess_registry)


# ---------------------------------------------------------------------------
# Rational constant powers
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression, expected",
    [
        ("y = x ** (1 / 2)", Fraction(1, 2)),
        ("y = x ** 0.5", Fraction(1, 2)),
        ("y = x ** 0.25", Fraction(1, 4)),
        ("y = x ** (-2)", -2),
        ("y = x ** ( -3 / 4 )", Fraction(-3, 4)),
    ],
)
def test_rational_powers_are_exact_fractions(sess_registry, expression, expected):
    known = {"x": UnitsContainer({"[length]": 1})}
    solution = solve_dimensionalities(
        [("e1", expression)], known, ("y",), registry=sess_registry
    )
    assert solution.is_unique
    assert dict(solution.particular_solution["y"]) == {"[length]": expected}
    for exponent in solution.particular_solution["y"].values():
        assert isinstance(exponent, (int, Fraction))
        assert not isinstance(exponent, float)


def test_square_root_chain(sess_registry):
    # v**2 = u**2 + 2*a*s  (all addends share the [length]**2/[time]**2 dim)
    equations = [("e1", "v ** 2 = u ** 2 + 2 * a * s")]
    known = {
        "u": "meter / second",
        "a": "meter / second**2",
        "s": "meter",
    }
    solution = solve_dimensionalities(equations, known, ("v",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["v"]) == {
        "[length]": 1,
        "[time]": -1,
    }
    env = _solution_environment(solution, known, ("v",), sess_registry)
    _resubstitute(equations, env, sess_registry)


def test_exact_rational_contradiction_not_hidden_by_tolerance(sess_registry):
    # 1/3 and 333/1000 are close as floats but distinct as Fractions.
    equations = [("e1", "y = x ** (1 / 3)")]
    known = {
        "x": UnitsContainer({"[length]": 1}),
        "y": UnitsContainer({"[length]": Fraction(333, 1000)}),
    }
    solution = solve_dimensionalities(equations, known, (), registry=sess_registry)
    assert solution.is_inconsistent
    assert solution.conflicting_equation_labels == ("e1",)


# ---------------------------------------------------------------------------
# Contradictions
# ---------------------------------------------------------------------------


def test_addition_contradiction_reports_label(sess_registry):
    equations = [("bad-add", "x + y = z")]
    known = {"x": "meter", "y": "second", "z": "meter"}
    solution = solve_dimensionalities(equations, known, (), registry=sess_registry)
    assert solution.is_inconsistent
    assert solution.conflicting_equation_labels == ("bad-add",)
    assert solution.particular_solution == {}
    assert solution.nullspace_basis == ()
    assert solution.free_degrees == 0


def test_cross_equation_contradiction_reports_both_labels(sess_registry):
    equations = [
        ("e1", "x = y * z"),
        ("e2", "x = y / z"),
    ]
    known = {"x": "meter", "y": "second"}
    solution = solve_dimensionalities(equations, known, ("z",), registry=sess_registry)
    assert solution.is_inconsistent
    assert set(solution.conflicting_equation_labels) == {"e1", "e2"}


def test_integer_labels_are_preserved(sess_registry):
    equations = [(7, "x + y = z")]
    known = {"x": "meter", "y": "second", "z": "meter"}
    solution = solve_dimensionalities(equations, known, (), registry=sess_registry)
    assert solution.conflicting_equation_labels == (7,)


# ---------------------------------------------------------------------------
# Dimensionless cases and numeric literals
# ---------------------------------------------------------------------------


def test_all_dimensionless(sess_registry):
    equations = [
        ("e1", "x = 2 * 3 + 4"),
        ("e2", "y = x - 1"),
    ]
    solution = solve_dimensionalities(equations, {}, ("x", "y"), registry=sess_registry)
    assert solution.is_unique
    assert solution.base_dimensions == ()
    assert dict(solution.particular_solution["x"]) == {}
    assert dict(solution.particular_solution["y"]) == {}
    env = _solution_environment(solution, {}, ("x", "y"), sess_registry)
    _resubstitute(equations, env, sess_registry)


def test_numeric_literals_never_get_units(sess_registry):
    equations = [("e1", "x = 2 * y"), ("e2", "z = 3.14 * y + y")]
    known = {"y": "meter"}
    solution = solve_dimensionalities(
        equations, known, ("x", "z"), registry=sess_registry
    )
    assert solution.is_unique
    assert dict(solution.particular_solution["x"]) == {"[length]": 1}
    assert dict(solution.particular_solution["z"]) == {"[length]": 1}


# ---------------------------------------------------------------------------
# Name resolution priority and registry invariance
# ---------------------------------------------------------------------------


def test_variable_symbol_shadows_registry_unit(sess_registry):
    # "m" is the metre symbol in the registry; as a declared variable it must
    # be solved as mass instead.
    assert sess_registry.get_name("m") == "meter"
    equations = [("e1", "F = m * a")]
    known = {
        "F": "newton",
        "a": "meter / second**2",
    }
    solution = solve_dimensionalities(equations, known, ("m",), registry=sess_registry)
    assert dict(solution.particular_solution["m"]) == {"[mass]": 1}


def test_undeclared_name_falls_back_to_registry_unit(sess_registry):
    equations = [("e1", "x = meter / second")]
    solution = solve_dimensionalities(equations, {}, ("x",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["x"]) == {
        "[length]": 1,
        "[time]": -1,
    }


def test_bracket_dimension_names_in_equations(sess_registry):
    equations = [("e1", "x = y * [length] / [time]")]
    known = {"y": UnitsContainer({"[time]": -1})}
    solution = solve_dimensionalities(equations, known, ("x",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["x"]) == {
        "[length]": 1,
        "[time]": -2,
    }


def test_dimensionless_marker_in_equation(sess_registry):
    equations = [("e1", "x = y * []")]
    known = {"y": "meter"}
    solution = solve_dimensionalities(equations, known, ("x",), registry=sess_registry)
    assert solution.is_unique
    assert dict(solution.particular_solution["x"]) == {"[length]": 1}


def test_bracket_dimensions_work_without_registry():
    equations = [("e1", "x = y * [length] / [time]")]
    known = {"y": {"[time]": -1}}
    solution = solve_dimensionalities(equations, known, ("x",))
    assert solution.is_unique
    assert solution.base_dimensions == ("[time]", "[length]")
    assert dict(solution.particular_solution["x"]) == {
        "[length]": 1,
        "[time]": -2,
    }
    env = _solution_environment(solution, known, ("x",), None)
    _resubstitute(equations, env, None)


@pytest.mark.parametrize(
    "permutation",
    list(permutations(range(2))),
)
def test_equation_order_and_aliases_do_not_change_solution(sess_registry, permutation):
    base = [
        ("e1", "F = m * a"),
        ("e2", "a = v / t"),
    ]
    variants = [
        (
            {"F": "N", "a": "m/s**2", "t": "s"},
            ("m", "v"),
        ),
        (
            {"F": "newton", "a": "metre / second**2", "t": "second"},
            ("v", "m"),
        ),
    ]
    results = []
    for known, unknowns in variants:
        ordered = [base[index] for index in permutation]
        results.append(
            solve_dimensionalities(ordered, known, unknowns, registry=sess_registry)
        )
    assert results[0].particular_solution == results[1].particular_solution
    assert results[0].base_dimensions == results[1].base_dimensions


def test_registry_is_not_modified(sess_registry):
    units_before = set(sess_registry._units)
    dimensions_before = set(sess_registry._dimensions)
    equations = [("e1", "F = m * a"), ("e2", "a = v / t")]
    known = {"F": "newton", "a": "meter / second**2", "t": "second"}
    solve_dimensionalities(equations, known, ("m", "v"), registry=sess_registry)
    assert set(sess_registry._units) == units_before
    assert set(sess_registry._dimensions) == dimensions_before


# ---------------------------------------------------------------------------
# Inference-space boundary
# ---------------------------------------------------------------------------


def test_extra_base_dimensions_are_the_boundary(sess_registry):
    equations = [("e1", "x = y")]
    solution = solve_dimensionalities(
        equations,
        {},
        ("x", "y"),
        registry=sess_registry,
        extra_base_dimensions=("[substance]",),
    )
    assert solution.base_dimensions == ("[substance]",)
    for container in solution.particular_solution.values():
        assert all(dim == "[substance]" for dim in container)
    for vector in solution.nullspace_basis:
        for container in vector.values():
            assert all(dim == "[substance]" for dim in container)


def test_bare_extra_dimension_names_are_bracketed(sess_registry):
    solution = solve_dimensionalities(
        [("e1", "x = y")],
        {},
        ("x", "y"),
        registry=sess_registry,
        extra_base_dimensions=("substance",),
    )
    assert solution.base_dimensions == ("[substance]",)


def test_no_new_dimension_is_invented(sess_registry):
    # Only [length] is declared; the free unknown can carry [length] exponents
    # but never any other physical dimension.
    solution = solve_dimensionalities(
        [("e1", "y = x / x")],
        {"y": "dimensionless"},
        ("x",),
        registry=sess_registry,
        extra_base_dimensions=("[length]",),
    )
    available = set(solution.base_dimensions)
    for vector in (solution.particular_solution, *solution.nullspace_basis):
        for container in vector.values():
            assert set(container).issubset(available)


# ---------------------------------------------------------------------------
# Input forms
# ---------------------------------------------------------------------------


def test_dimension_mapping_input_without_registry():
    equations = [("e1", "F = m * a")]
    known = {"a": {"[length]": 1, "[time]": -2}}
    solution = solve_dimensionalities(
        equations,
        known,
        ("F", "m"),
        extra_base_dimensions=("[length]", "[time]", "[mass]"),
    )
    assert solution.is_underdetermined
    # Two symbols over three dimensions linked by F - m = a: three shared
    # shifts (one per base dimension) remain undetermined.
    assert solution.free_degrees == 3
    assert len(solution.nullspace_basis) == 3
    env = _solution_environment(
        solution, known, ("F", "m"), None, (Fraction(2), Fraction(-1), Fraction(5))
    )
    _resubstitute(equations, env, None)


def test_quantity_and_unit_known_inputs(sess_registry):
    equations = [("e1", "x = v * t")]
    known = {
        "v": sess_registry.Quantity(1, "meter / second"),
        "t": sess_registry.Unit("second"),
    }
    solution = solve_dimensionalities(equations, known, ("x",), registry=sess_registry)
    assert dict(solution.particular_solution["x"]) == {"[length]": 1}


def test_fraction_registry_keeps_exact_types():
    registry = UnitRegistry(non_int_type=Fraction)
    equations = [("e1", "y = x ** (1 / 2)")]
    known = {"x": registry.Quantity(Fraction(1), "meter")}
    solution = solve_dimensionalities(equations, known, ("y",), registry=registry)
    assert solution.is_unique
    exponent = next(iter(solution.particular_solution["y"].values()))
    assert exponent == Fraction(1, 2)
    assert isinstance(exponent, Fraction)


# ---------------------------------------------------------------------------
# Restricted grammar and validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expression",
    [
        "x = y.__class__",
        "x = y.foo",
        "x = foo(y)",
        "x = eval('1')",
        "x = @y",
        "x = y, z",
        "x = y # comment",
        "x = y 2",
        "x = (y",
        "x = y)",
        "x = y ** z",
        "x = y ** (1 + z)",
        "x = = y",
        "x = y = z",
        "x = y + + +",
        "x = * y",
        "x = [length",
        "x = y * [123]",
    ],
)
def test_unsafe_or_malformed_expressions_are_rejected(expression):
    with pytest.raises(DefinitionSyntaxError):
        solve_dimensionalities([("e", expression)], {}, ("x", "y", "z"))


@pytest.mark.parametrize(
    "expression",
    [
        "x = -y",
        "x = +y",
        "x = -(y + z) * 2",
        "x = y * -z",
        "x = (y + z) / w",
        "x = y ** 2",
        "x = y ** (2 / 3)",
        "x = y ** -1",
        "x = 1e-3 * y",
    ],
)
def test_restricted_grammar_accepts_valid_expressions(expression):
    solution = solve_dimensionalities(
        [("e", expression)],
        {"y": "meter", "z": "meter", "w": "meter"},
        ("x",),
        registry=UnitRegistry(),
    )
    assert not solution.is_inconsistent


def test_undeclared_symbol_raises(sess_registry):
    with pytest.raises(ValueError, match="neither a declared"):
        solve_dimensionalities(
            [("e", "x = y * z")],
            {"y": "meter"},
            ("x",),
            registry=sess_registry,
        )


def test_missing_registry_raises_for_unit_strings():
    with pytest.raises(ValueError, match="registry is required"):
        solve_dimensionalities([("e", "x = y")], {"y": "meter"}, ("x",))


def test_duplicated_label_raises():
    with pytest.raises(ValueError, match="duplicated equation label"):
        solve_dimensionalities([("e", "x = y"), ("e", "y = z")], {}, ("x", "y", "z"))


def test_known_and_unknown_overlap_raises():
    with pytest.raises(ValueError, match="cannot be both known and unknown"):
        solve_dimensionalities([("e", "x = y")], {"x": "meter"}, ("x",))


def test_repeated_unknown_raises():
    with pytest.raises(ValueError, match="must not be repeated"):
        solve_dimensionalities([("e", "x = y")], {}, ("x", "x"))


def test_invalid_extra_dimension_raises():
    with pytest.raises(ValueError, match="invalid base dimension name"):
        solve_dimensionalities(
            [("e", "x = y")], {}, ("x",), extra_base_dimensions=("not a dim",)
        )


# ---------------------------------------------------------------------------
# Result object behaviour
# ---------------------------------------------------------------------------


def test_solution_repr_documents_boundary(sess_registry):
    solution = solve_dimensionalities(
        [("e1", "F = m * a")],
        {"F": "newton", "a": "meter / second**2"},
        ("m",),
        registry=sess_registry,
    )
    text = repr(solution)
    assert "unique" in text
    assert "[mass]" in text


def test_status_flags(sess_registry):
    unique = solve_dimensionalities(
        [("e1", "x = y")], {"y": "meter"}, ("x",), registry=sess_registry
    )
    assert unique.is_unique
    assert not unique.is_underdetermined
    assert not unique.is_inconsistent
    assert isinstance(unique, DimensionalitySolution)

    inconsistent = solve_dimensionalities(
        [("e1", "x + y = z")],
        {"x": "meter", "y": "second", "z": "meter"},
        (),
        registry=sess_registry,
    )
    assert inconsistent.is_inconsistent
    assert not inconsistent.is_unique
