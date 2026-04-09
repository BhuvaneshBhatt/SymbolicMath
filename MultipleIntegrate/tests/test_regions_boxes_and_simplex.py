import sympy as sp

from multiple_integrate import BoxRegion, IteratedRegion, SimplexRegion, region_from_ranges


def test_box_region_recognition():
    x, y = sp.symbols("x y", real=True)
    region = region_from_ranges([(x, 0, 1), (y, -2, 2)])
    assert isinstance(region, BoxRegion)
    assert sp.simplify(region.constant_volume() - 4) == 0
    assert region.is_reflection_invariant(y)


def test_iterated_region_recognition():
    x, y = sp.symbols("x y", real=True)
    region = region_from_ranges([(x, 0, 1), (y, 0, x)])
    assert isinstance(region, IteratedRegion)
    assert not isinstance(region, BoxRegion)
    assert not region.is_reflection_invariant(y)


def test_standard_simplex_recognition():
    x, y, z = sp.symbols("x y z", real=True)
    region = region_from_ranges([(z, 0, 1 - x - y), (y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert region.dimension == 3


def test_reflection_invariance_rejects_later_dependency():
    x, y, z = sp.symbols("x y z", real=True)
    region = region_from_ranges([(x, -1, 1), (y, -x, x), (z, 0, y)])
    assert not region.is_reflection_invariant(y)
