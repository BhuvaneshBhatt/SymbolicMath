import sympy as sp
from sympy import Integer, exp, oo, pi, sqrt

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq

x, y = sp.symbols("x y", real=True)


class TestRegressionSymmetryAndInner1D:
    def test_odd_function_on_symmetric_box_is_zero(self):
        result = multiple_integrate(x * exp(-(x**2 + y**2)), (x, -1, 1), (y, -1, 1))
        assert_eq(result, Integer(0), "Odd-in-x integrand on symmetric box should vanish")

    def test_nonsymmetric_interval_does_not_trigger_odd_shortcut(self):
        result = multiple_integrate(x, (x, -1, 2))
        assert_eq(result, sp.Rational(3, 2), "Odd shortcut must not fire on nonsymmetric interval")

    def test_standard_inner_rational_integral(self):
        result = multiple_integrate(1 / (x**2 + 1), (x, -oo, oo))
        assert_eq(result, pi, "Standard full-line rational integral")

    def test_standard_inner_gaussian_moment(self):
        result = multiple_integrate(x**2 * exp(-x**2), (x, -oo, oo))
        assert_eq(result, sqrt(pi) / 2, "Standard Gaussian moment")
