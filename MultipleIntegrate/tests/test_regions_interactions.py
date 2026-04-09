import sympy as sp

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq


def test_constant_over_simplex():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(2, (y, 0, 1 - x), (x, 0, 1)), 1, 'constant over simplex')


def test_zero_over_disk():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(0, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), 0, 'zero over disk')


def test_sum_of_radial_and_nonradial_terms_on_disk():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate((x**2 + y**2) + x, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.pi / 2, 'sum of radial and odd terms on disk')


def test_graph_region_polynomial_in_inner_variable():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(y**2, (y, x, 1), (x, 0, 1))
    assert_eq(result, sp.Rational(1, 4), 'inner-variable polynomial on graph region')


def test_simplex_region_not_misread_as_box_symmetry():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(y, (y, 0, 1 - x), (x, 0, 1))
    assert_eq(result, sp.Rational(1, 6), 'simplex should not use symmetric-zero shortcut')
