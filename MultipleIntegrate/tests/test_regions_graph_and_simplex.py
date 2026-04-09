import sympy as sp

from multiple_integrate import GraphRegion, SimplexRegion, multiple_integrate, region_from_ranges
from tests.helpers import assert_eq


def test_simplex_constant_moment():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, 0, 1 - x), (x, 0, 1)])
    assert isinstance(region, SimplexRegion)
    assert_eq(multiple_integrate(1, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 2), 'simplex area')


def test_simplex_polynomial_moment():
    x, y = sp.symbols('x y', real=True)
    assert_eq(multiple_integrate(x * y, (y, 0, 1 - x), (x, 0, 1)), sp.Rational(1, 24), 'simplex xy moment')


def test_graph_region_recognition():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    pieces = region.reversed_pieces()
    assert pieces == [[(x, 0, y), (y, 0, 1)]]


def test_graph_region_reversal_for_inner_only_integrand():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(sp.exp(-y), (y, x, 1), (x, 0, 1))
    assert_eq(result, 1 - 2 / sp.E, 'graph reversal should reduce inner-only integrand')


def test_graph_region_cell_split():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1 - x), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    pieces = region.reversed_pieces()
    assert pieces is not None and len(pieces) == 2
    # Check total geometric area via reversed pieces.
    total = sp.Integer(0)
    for (_, inner_lo, inner_hi), (outer, lo, hi) in pieces:
        total += sp.integrate(sp.simplify(inner_hi - inner_lo), (outer, lo, hi))
    assert_eq(sp.simplify(total), sp.Rational(1, 4), 'split graph region area')
