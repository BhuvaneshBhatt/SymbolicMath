import sympy as sp

from multiple_integrate import (
    AffineSimplexRegion,
    BallRegion,
    DiskRegion,
    GraphRegion,
    SimplexRegion,
    region_from_ranges,
)
from multiple_integrate.regions import (
    BoxRegion,
    EllipsoidRegion,
    match_affine_simplex,
    match_graph_region,
)


def test_graph_region_priority_over_affine_simplex_for_variable_lower_bound():
    x, y = sp.symbols('x y', real=True)
    ranges = [(x, 0, 1), (y, x, 1)]
    assert match_affine_simplex(ranges) is None
    assert match_graph_region(ranges) is not None
    region = region_from_ranges(ranges)
    assert isinstance(region, GraphRegion)


def test_affine_simplex_priority_over_graph_when_both_match():
    x, y = sp.symbols('x y', real=True)
    ranges = [(x, 2, 5), (y, 1, 1 + 2 * (1 - (x - 2) / 3))]
    assert match_affine_simplex(ranges) is not None
    assert match_graph_region(ranges) is not None
    region = region_from_ranges(ranges)
    assert isinstance(region, AffineSimplexRegion)
    assert not isinstance(region, GraphRegion)


def test_standard_simplex_priority_over_graph():
    x, y = sp.symbols('x y', real=True)
    ranges = [(y, 0, 1 - x), (x, 0, 1)]
    assert match_affine_simplex(ranges) is None
    assert match_graph_region(ranges) is not None
    region = region_from_ranges(ranges)
    assert isinstance(region, SimplexRegion)
    assert not isinstance(region, GraphRegion)


def test_standard_disk_priority_over_box_fallback():
    x, y = sp.symbols('x y', real=True)
    ranges = [(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)]
    region = region_from_ranges(ranges)
    assert isinstance(region, DiskRegion)
    assert not isinstance(region, BoxRegion)
    assert type(region).__name__ == 'DiskRegion'


def test_standard_ball_priority_over_iterated_fallback():
    x, y, z = sp.symbols('x y z', real=True)
    ranges = [
        (x, -1, 1),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
    ]
    region = region_from_ranges(ranges)
    assert isinstance(region, BallRegion)
    assert type(region).__name__ == 'BallRegion'


def test_standard_ellipsoid_priority_over_iterated_fallback():
    x, y = sp.symbols('x y', real=True)
    ranges = [(x, -2, 2), (y, -3 * sp.sqrt(1 - x**2 / 4), 3 * sp.sqrt(1 - x**2 / 4))]
    region = region_from_ranges(ranges)
    assert isinstance(region, EllipsoidRegion)
    assert type(region).__name__ == 'EllipsoidRegion'
