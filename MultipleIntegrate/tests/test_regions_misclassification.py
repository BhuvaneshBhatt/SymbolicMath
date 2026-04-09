import sympy as sp

from multiple_integrate import BallRegion, DiskRegion, GraphRegion, IteratedRegion, SimplexRegion, region_from_ranges


def test_shifted_disk_not_classified_as_disk():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, 0, 2), (y, -sp.sqrt(1 - (x - 1)**2), sp.sqrt(1 - (x - 1)**2))])
    assert not isinstance(region, DiskRegion)


def test_annulus_like_bounds_not_classified_as_disk():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, -1, 1), (y, sp.sqrt(1 - x**2) / 2, sp.sqrt(1 - x**2))])
    assert not isinstance(region, DiskRegion)


def test_ellipsoid_like_bounds_not_classified_as_ball():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([
        (x, -1, 1),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (z, -sp.sqrt(4 - x**2 - y**2), sp.sqrt(4 - x**2 - y**2)),
    ])
    assert not isinstance(region, BallRegion)


def test_nonlinear_triangle_not_classified_as_simplex():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, 0, 1), (y, 0, 1 - x**2)])
    assert not isinstance(region, SimplexRegion)


def test_nonlinear_graph_falls_back_to_iterated():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, 0, 1), (y, x**2, 1)])
    assert isinstance(region, IteratedRegion)
    assert not isinstance(region, GraphRegion)
