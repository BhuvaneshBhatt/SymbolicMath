import sympy as sp
from sympy import Integer, Rational, exp, log, oo, pi, sqrt

from multiple_integrate import multiple_integrate
from tests.helpers import assert_diverges, assert_eq

x, y = sp.symbols("x y", real=True)


class TestSingularButConvergentCases:
    def test_log_endpoint_integral(self):
        result = multiple_integrate(log(x), (x, 0, 1))
        assert_eq(result, -1, "log(x) should be integrable on (0, 1)")

    def test_log_one_minus_x_endpoint_integral(self):
        result = multiple_integrate(log(1 - x), (x, 0, 1))
        assert_eq(result, -1, "log(1-x) should be integrable on (0, 1)")

    def test_inverse_sqrt_endpoint_integral(self):
        result = multiple_integrate(x ** Rational(-1, 2), (x, 0, 1))
        assert_eq(result, 2, "x^(-1/2) should be integrable on (0, 1)")

    def test_integrable_endpoint_singularity(self):
        result = multiple_integrate(x ** Rational(-1, 2), (x, 0, 1))
        assert_eq(result, Integer(2), "Integrable endpoint singularity should succeed")


class TestDivergenceRegressions:
    def test_nonintegrable_endpoint_singularity_diverges(self):
        result = multiple_integrate(1 / x, (x, 0, 1))
        assert_diverges(result)

    def test_upper_endpoint_log_divergence(self):
        result = multiple_integrate(1 / (1 - x), (x, 0, 1))
        assert_diverges(result)

    def test_upper_endpoint_sqrt_is_still_integrable(self):
        result = multiple_integrate(1 / sqrt(1 - x), (x, 0, 1))
        assert_eq(result, Integer(2), "Borderline square-root endpoint should converge")

    def test_tail_harmonic_diverges(self):
        result = multiple_integrate(1 / x, (x, 1, oo))
        assert_diverges(result)

    def test_tail_p_two_converges(self):
        result = multiple_integrate(1 / x**2, (x, 1, oo))
        assert_eq(result, Integer(1), "p=2 tail should converge")

    def test_wrong_sign_gaussian_diverges(self):
        result = multiple_integrate(exp(x**2), (x, -oo, oo))
        assert_diverges(result)

    def test_right_sign_gaussian_converges(self):
        result = multiple_integrate(exp(-x**2), (x, -oo, oo))
        assert_eq(result, sqrt(pi), "Standard Gaussian should converge")

    def test_two_dimensional_kernel_can_still_converge(self):
        result = multiple_integrate(1 / (x + y), (x, 0, 1), (y, 0, 1))
        assert_eq(result, 2 * log(2), "Mild 2-D singular kernel should converge")

    def test_two_dimensional_kernel_can_diverge(self):
        result = multiple_integrate(1 / (x + y) ** 2, (x, 0, 1), (y, 0, 1))
        assert_diverges(result)

    def test_nonobvious_two_dimensional_convergent_case(self):
        result = multiple_integrate(1 / (x + y**2), (x, 0, 1), (y, 0, 1))
        assert_eq(result, log(2) + pi / 2, "Nonobvious mixed singularity should converge")

    def test_product_singularity_diverges(self):
        result = multiple_integrate(1 / (x * y), (x, 0, 1), (y, 0, 1))
        assert_diverges(result)
