
import sympy as sp

from multiple_integrate import boole, multiple_integrate


class TestBooleSupport:
    def test_boole_interval_restriction(self):
        x = sp.symbols("x", real=True)
        result = multiple_integrate(x * boole(x >= 0), (x, -1, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_boole_graph_region_from_box(self):
        x, y = sp.symbols("x y", real=True)
        result = multiple_integrate((x + y) * boole(y <= 1 - x), (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 3)) == 0

    def test_boole_disk_restriction_from_square(self):
        x, y = sp.symbols("x y", real=True)
        result = multiple_integrate(boole(x**2 + y**2 <= 1), (x, -1, 1), (y, -1, 1))
        assert sp.simplify(result - sp.pi) == 0


class TestPiecewiseSupport:
    def test_piecewise_univariate(self):
        x = sp.symbols("x", real=True)
        expr = sp.Piecewise((x, x < 1), (x**2, True))
        result = multiple_integrate(expr, (x, 0, 2))
        assert sp.simplify(result - sp.Rational(17, 6)) == 0

    def test_piecewise_indicator_on_box(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((1, y <= 1 - x), (0, True))
        result = multiple_integrate(expr, (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_piecewise_graph_branches(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((y, y <= 1 - x), (x, True))
        result = multiple_integrate(expr, (x, 0, 1), (y, 0, 1))
        assert sp.simplify(result - sp.Rational(1, 2)) == 0

    def test_piecewise_constant_outside_supported_restriction_falls_back(self):
        x, y = sp.symbols("x y", real=True)
        expr = sp.Piecewise((1, x*y < 1), (2, True))
        result = multiple_integrate(expr, (x, 0, 1), (y, 0, 1))
        # On the unit square x*y < 1 except at the point (1,1), which has measure zero.
        assert sp.simplify(result - 1) == 0
