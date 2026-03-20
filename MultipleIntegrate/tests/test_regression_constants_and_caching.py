import sympy as sp
from sympy import E, Integer, Rational, exp

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq, sym_eq

x, y = sp.symbols("x y", real=True)


class TestRegressionConstantsAndCaching:
    def test_python_int_constant_integrand(self):
        result = multiple_integrate(1, (x, 0, 1), (y, 0, 1))
        assert_eq(result, Integer(1), "Python int constant should integrate exactly")

    def test_python_float_constant_integrand(self):
        result = multiple_integrate(2.0, (x, 0, 1), (y, 0, 1))
        assert_eq(result, sp.Float(2.0), "Python float constant should integrate correctly")

    def test_zero_integrand_nontrivial_bounds(self):
        result = multiple_integrate(0, (x, -2, 3), (y, Rational(-1, 2), Rational(5, 2)))
        assert_eq(result, Integer(0), "Zero integrand should short-circuit cleanly")

    def test_distinct_calls_do_not_reuse_wrong_cached_value(self):
        first = multiple_integrate(exp(-x), (x, 0, 1))
        second = multiple_integrate(exp(-x), (x, 0, 2))
        assert_eq(first, 1 - exp(-1), "First cached result")
        assert_eq(second, 1 - exp(-2), "Second cached result with different bounds")
        assert not sym_eq(first, second)

    def test_one_variable_log_integral_exact(self):
        result = multiple_integrate(sp.log(x), (x, 1, E))
        assert_eq(result, Integer(1), "1-D log integral should stay exact")
