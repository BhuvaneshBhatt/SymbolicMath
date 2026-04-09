import sympy as sp
from sympy import atan, exp, gamma, oo, pi, Rational, sqrt

from multiple_integrate import BallRegion, multiple_integrate
from tests.helpers import assert_eq, run_with_soft_budget


def test_transformed_linear_reduction_returns_prompt_unevaluated_integral():
    x1, x2 = sp.symbols('x1 x2', positive=True)
    expr = exp(-exp(x1 + x2))
    result, _ = run_with_soft_budget(
        lambda: multiple_integrate(expr, (x1, 0, oo), (x2, 0, oo)),
        4.0,
        label='transformed linear layer-cake reduction should not hang',
    )
    assert isinstance(result, sp.Integral)
    assert len(result.limits) == 1


def test_nested_quadratic_reference_example_has_soft_budget():
    x1, x2 = sp.symbols('x1 x2', real=True)
    expr = 1 / (1 + (x1**2 + 2 * x2**2 + 2 * (x1 + 2 * x2)) ** 2)
    result, _ = run_with_soft_budget(
        lambda: multiple_integrate(expr, (x1, -oo, oo), (x2, -oo, oo)),
        4.0,
        label='nested quadratic reference example',
    )
    expected = pi * (pi / 2 + atan(sp.Integer(3))) / sqrt(2)
    assert_eq(result, expected, 'nested quadratic reference example')


def test_superellipse_reference_example_has_soft_budget():
    x1, x2 = sp.symbols('x1 x2', real=True)
    expr = 1 / (1 + (x1**2 + x2**4) ** 3)
    result, _ = run_with_soft_budget(
        lambda: multiple_integrate(expr, (x1, 0, oo), (x2, 0, oo)),
        2.0,
        label='superellipse reference example',
    )
    expected = -(pi ** Rational(5, 2)) / (4 * gamma(Rational(-1, 4)) * gamma(Rational(7, 4)))
    assert abs(complex(result.evalf(20)).real - float(expected.evalf())) < 1e-9


def test_notebook_rectangular_double_integral_has_soft_budget():
    x, y = sp.symbols('x y', positive=True)
    expr = x**2 * y / (x + y)
    result, _ = run_with_soft_budget(
        lambda: multiple_integrate(expr, (x, 0, 3), (y, 1, 2)),
        2.0,
        label='notebook rectangular double integral',
    )
    expected = sp.Rational(39, 8) + 36 * sp.log(2) - sp.Rational(65, 4) * sp.log(5)
    assert sp.simplify(result - expected) == 0


def test_notebook_ball_moment_has_soft_budget():
    x, y, z = sp.symbols('x y z', real=True)
    region = BallRegion(
        (
            (x, -1, 1),
            (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
            (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        ),
        radius=1,
        dimension=3,
    )
    result, _ = run_with_soft_budget(
        lambda: multiple_integrate(x**2 + y**2 + z**2, region),
        2.0,
        label='notebook ball moment integral',
    )
    assert sp.simplify(result - sp.Rational(4, 5) * sp.pi) == 0
