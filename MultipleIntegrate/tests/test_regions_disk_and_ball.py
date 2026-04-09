
import sympy as sp

from multiple_integrate import BallRegion, DiskRegion, multiple_integrate, region_from_ranges
from tests.helpers import assert_eq


def test_disk_region_recognition():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    assert isinstance(region, DiskRegion)
    assert_eq(region.constant_volume(), sp.pi, 'unit disk area')


def test_disk_polynomial_moment():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(x**2, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.pi / 4, 'disk x^2 moment')


def test_disk_radial_integral():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(sp.exp(-(x**2 + y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.pi * (1 - sp.exp(-1)), 'radial disk integral')


def test_ball_region_recognition():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([
        (x, -1, 1),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
    ])
    assert isinstance(region, BallRegion)
    assert_eq(region.constant_volume(), sp.Rational(4, 3) * sp.pi, 'unit ball volume')


def test_ball_polynomial_moment():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(x**2, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, sp.Rational(4, 15) * sp.pi, 'ball x^2 moment')


def test_ball_radial_integral():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(sp.exp(-(x**2 + y**2 + z**2)), (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    expected = sp.pi * (-2 * sp.exp(-1) + sp.sqrt(sp.pi) * sp.erf(1))
    assert_eq(sp.simplify(result), sp.simplify(expected), 'radial ball integral')
