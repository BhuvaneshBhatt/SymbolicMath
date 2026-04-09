import sympy as sp

from multiple_integrate import BallRegion, DiskRegion, GraphRegion, SimplexRegion, region_from_ranges


def test_region_from_ranges_uses_sympy_inner_first_simplex():
    x, y, z = sp.symbols('x y z')
    region = region_from_ranges([(z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)


def test_region_from_ranges_uses_sympy_inner_first_graph():
    x, y = sp.symbols('x y')
    region = region_from_ranges([(y, x, 1 - x), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    assert region.inner_var == y
    assert region.outer_var == x


def test_region_from_ranges_uses_sympy_inner_first_disk_ball():
    x, y, z = sp.symbols('x y z')
    disk = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    ball = region_from_ranges([
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    ])
    assert isinstance(disk, DiskRegion)
    assert isinstance(ball, BallRegion)
