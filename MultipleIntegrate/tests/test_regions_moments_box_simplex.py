import sympy as sp

from multiple_integrate import multiple_integrate, region_from_ranges, BoxRegion, SimplexRegion
from tests.helpers import assert_eq


def test_box_moment_2d():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1))
    assert_eq(result, sp.Rational(1, 12), 'box x^2 y^3 moment')


def test_box_odd_moment_vanishes():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(x * y**2, (x, -1, 1), (y, -1, 1))
    assert_eq(result, 0, 'odd moment on symmetric box')


def test_box_moment_3d():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(x**2 * y * z, (x, 0, 1), (y, 0, 2), (z, 0, 3))
    assert_eq(result, 3, '3D box monomial moment')


def test_simplex_constant_moment_against_region_method():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert_eq(region.polynomial_moment(1), sp.Rational(1, 2), 'simplex area from region method')
    assert_eq(multiple_integrate(1, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 2), 'simplex area')


def test_simplex_first_moments():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(x, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 6), 'simplex x moment')
    assert_eq(multiple_integrate(y, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 6), 'simplex y moment')


def test_simplex_mixed_moment():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(x * y, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 24), 'simplex xy moment')


def test_simplex_3d_constant_volume():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([(z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert_eq(region.constant_volume(), sp.Rational(1, 6), '3D simplex volume')


def test_simplex_3d_linear_moment():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(x, (z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1))
    assert_eq(result, sp.Rational(1, 24), '3D simplex x moment')
