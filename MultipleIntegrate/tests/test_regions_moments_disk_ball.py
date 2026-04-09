import sympy as sp

from multiple_integrate import multiple_integrate, region_from_ranges, BallRegion, DiskRegion
from tests.helpers import assert_eq


def test_disk_area_and_region_volume():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    assert isinstance(region, DiskRegion)
    assert_eq(region.constant_volume(), sp.pi, 'disk area from region object')
    assert_eq(multiple_integrate(1, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.pi, 'unit disk area')


def test_disk_even_moments():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(x**2, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.pi / 4, 'disk x^2 moment')
    assert_eq(multiple_integrate(x**2 + y**2, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.pi / 2, 'disk radial quadratic moment')


def test_disk_odd_moment_vanishes():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(x, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, 0, 'disk odd moment')


def test_ball_volume_and_even_moments():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    ])
    assert isinstance(region, BallRegion)
    assert_eq(region.constant_volume(), sp.Rational(4, 3) * sp.pi, 'ball volume from region object')
    assert_eq(multiple_integrate(1, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.Rational(4, 3) * sp.pi, 'unit ball volume')
    assert_eq(multiple_integrate(x**2, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.Rational(4, 15) * sp.pi, 'ball x^2 moment')
    assert_eq(multiple_integrate(x**2 + y**2 + z**2, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), sp.Rational(4, 5) * sp.pi, 'ball radial quadratic moment')


def test_ball_odd_moment_vanishes():
    x, y, z = sp.symbols('x y z', real=True)
    result = multiple_integrate(x * z, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, 0, 'ball odd moment')
