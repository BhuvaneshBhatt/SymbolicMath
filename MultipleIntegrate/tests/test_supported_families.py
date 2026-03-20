import pytest
import sympy as sp
from sympy import Rational, exp, oo, pi, sqrt

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq

x, y = sp.symbols("x y", real=True)
r, theta = sp.symbols("r theta", real=True)


class TestSpecialFunctionAndTransformCases:
    def test_quartic_rational_full_line(self):
        result = multiple_integrate(1 / (x**4 + 1), (x, -oo, oo))
        assert_eq(result, pi / sqrt(2), "Quartic rational full-line integral")

    def test_damped_cosine_transform(self):
        result = multiple_integrate(exp(-x) * sp.cos(x), (x, 0, oo))
        assert_eq(result, Rational(1, 2), "Laplace/Fourier-style transform case")

    def test_beta_polynomial_case(self):
        result = multiple_integrate(x**2 * (1 - x) ** 3, (x, 0, 1))
        assert_eq(result, Rational(1, 60), "Beta-type polynomial integral")

    def test_arctangent_kernel_case(self):
        result = multiple_integrate(1 / (1 + x**2), (x, 0, 1))
        assert_eq(result, pi / 4, "Arctangent kernel should integrate exactly")


class TestGaussianAndRadialCases:
    def test_two_dimensional_gaussian_mass(self):
        result = multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi, "2-D Gaussian mass")

    def test_radial_disk_moment_in_polar_coordinates(self):
        result = multiple_integrate(r**3 * sp.cos(theta) ** 2, (r, 0, 1), (theta, 0, 2 * pi))
        assert_eq(result, pi / 4, "Polar-coordinate disk moment")


class TestTrigExponentialSeparableCases:
    def test_separable_trigonometric_product(self):
        result = multiple_integrate(sp.sin(x) * sp.sin(y), (x, 0, pi), (y, 0, pi))
        assert_eq(result, 4, "Separable trigonometric product on [0, pi]^2")


class TestSupportedFamiliesMatrix:
    def test_box_moment_polynomial(self):
        result = multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1))
        assert_eq(result, Rational(1, 12), "Box moment recognizer should stay stable")

    def test_product_rational_full_line_integral(self):
        result = multiple_integrate(1 / ((x**2 + 1) * (y**2 + 1)), (x, -oo, oo), (y, -oo, oo))
        assert_eq(result, pi**2, "Repeated rational full-line integral")

    def test_separable_trig_transform(self):
        result = multiple_integrate(sp.sin(x) * sp.sin(y), (x, 0, pi), (y, 0, pi))
        assert_eq(result, 4, "Separable trig case should remain stable")

    def test_gaussian_moment_family(self):
        result = multiple_integrate(x**2 * exp(-x**2), (x, -oo, oo))
        assert_eq(result, sqrt(pi) / 2, "Gaussian moment family")

    def test_constant_family(self):
        result = multiple_integrate(1, (x, 0, 1), (y, 0, 1))
        assert_eq(result, 1, "Constant integrands on boxes")


@pytest.mark.xfail(reason="General dependent-bound simplex support is still incomplete")
def test_simplex_area_with_dependent_bounds():
    result = multiple_integrate(1, (x, 0, 1), (y, 0, 1 - x))
    assert_eq(result, Rational(1, 2), "Triangle area over a simplex domain")


@pytest.mark.xfail(reason="General dependent-bound simplex support is still incomplete")
def test_simplex_xy_moment_with_dependent_bounds():
    result = multiple_integrate(x * y, (x, 0, 1), (y, 0, 1 - x))
    assert_eq(result, Rational(1, 24), "Triangle xy moment over a simplex domain")
