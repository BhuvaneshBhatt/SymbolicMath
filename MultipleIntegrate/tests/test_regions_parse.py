import sympy as sp

from multiple_integrate import (
    BallRegion,
    BoxRegion,
    DiskRegion,
    GraphRegion,
    IteratedRegion,
    SimplexRegion,
    region_from_ranges,
)


def test_parse_box_region_2d():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(x, 0, 1), (y, -2, 3)])
    assert isinstance(region, BoxRegion)


def test_parse_box_region_3d():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([(x, -1, 1), (y, 0, 2), (z, 3, 5)])
    assert isinstance(region, BoxRegion)


def test_parse_graph_region_generic_affine_strip():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1 + x), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    assert isinstance(region, IteratedRegion)


def test_parse_simplex_region_2d():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert region.dimension == 2


def test_parse_simplex_region_3d():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([(z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert region.dimension == 3


def test_parse_graph_region_affine():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    assert region.outer_var == x
    assert region.inner_var == y


def test_parse_disk_region_standard_unit():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    assert isinstance(region, DiskRegion)
    assert sp.simplify(region.radius - 1) == 0


def test_parse_ball_region_standard_unit():
    x, y, z = sp.symbols('x y z', real=True)
    region = region_from_ranges([
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    ])
    assert isinstance(region, BallRegion)
    assert sp.simplify(region.radius - 1) == 0
    assert region.dimension == 3
