import sympy as sp

from multiple_integrate import GraphRegion, multiple_integrate, region_from_ranges
from tests.helpers import assert_eq


def test_graph_reversal_without_split_structure():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    assert region.reversed_pieces() == [[(x, 0, y), (y, 0, 1)]]


def test_graph_reversal_without_split_integral_equivalence():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(y, (y, x, 1), (x, 0, 1))
    assert_eq(result, sp.Rational(1, 3), 'graph reversal without split')


def test_graph_reversal_with_split_structure():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, x, 1 - x), (x, 0, 1)])
    assert isinstance(region, GraphRegion)
    pieces = region.reversed_pieces()
    assert pieces is not None and len(pieces) == 2
    assert pieces[0][0] == (x, 0, y)
    assert pieces[1][0] == (x, 0, 1 - y)


def test_graph_reversal_with_split_integral_equivalence():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(y, (y, x, 1 - x), (x, 0, 1))
    assert_eq(result, sp.Rational(1, 8), 'graph reversal with split')


def test_graph_reversal_inner_only_integrand_exp():
    x, y = sp.symbols('x y', real=True)
    result = multiple_integrate(sp.exp(-y), (y, x, 1), (x, 0, 1))
    assert_eq(result, 1 - 2 / sp.E, 'graph reversal inner-only exponential')
