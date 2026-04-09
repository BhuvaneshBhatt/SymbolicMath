import sympy as sp

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq


def test_disk_radial_exponential():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(sp.exp(-(x**2 + y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.pi * (1 - sp.exp(-1)), 'disk radial exponential')


def test_ball_radial_exponential():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(sp.exp(-(x**2 + y**2 + z**2)), (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    expected = sp.pi * (-2 * sp.exp(-1) + sp.sqrt(sp.pi) * sp.erf(1))
    assert_eq(sp.simplify(result), sp.simplify(expected), 'ball radial exponential')


def test_nonradial_disk_integrand_still_correct():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(x**2 + y, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.pi / 4, 'nonradial disk integrand should not misfire')
