import sympy as sp

from multiple_integrate import BoxRegion, DiskRegion, BallRegion, SimplexRegion, multiple_integrate, region_from_ranges
from tests.helpers import assert_eq


def test_box_symmetric_range_detected():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, -2, 2), (y, 0, 1)])
    assert isinstance(region, BoxRegion)
    assert region.is_reflection_invariant(x)
    assert region.symmetric_range(x) == (-2, 2)
    assert not region.is_reflection_invariant(y)


def test_simplex_not_reflection_invariant():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert not region.is_reflection_invariant(x)
    assert not region.is_reflection_invariant(y)


def test_disk_region_recognized_and_odd_integral_vanishes():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    assert isinstance(region, DiskRegion)
    result = multiple_integrate(x, (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, 0, 'disk odd integral')


def test_ball_region_recognized_and_odd_integral_vanishes():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    ])
    assert isinstance(region, BallRegion)
    result = multiple_integrate(x, (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)), (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1))
    assert_eq(result, 0, 'ball odd integral')
