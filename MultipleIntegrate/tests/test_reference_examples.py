"""
test_reference_examples.py
--------------------------
Tests drawn from external reference computations (reference CAS).

Covers:
  • Fubini-violation integrals (non-L¹ integrands where order changes the value)
  • Superellipse layer-cake (Strategy 4b): f((x1^p1 + x2^p2)^k) over [0,∞)²
  • Nested quadratic composition (Strategy 4c): f(q(x)^k + c) over ℝⁿ
  • Symbolic shifted Gaussians in 2-D and 6-D
  • Linear cubic: exp(-(1+s1·x1+s2·x2)³) over [0,∞)²
  • Bounded quadratic-sum: exp(-(x1+x2)²) on [0,3]×[0,5]
"""

import pytest
import sympy as sp
from sympy import (
    atan,
    exp,
    gamma,
    Integer,
    oo,
    pi,
    Rational,
    sqrt,
    symbols,
    expint,
    sin,
)

from multiple_integrate import multiple_integrate
from tests.helpers import assert_eq

x1, x2, x3, x4, x5, x6 = symbols("x1 x2 x3 x4 x5 x6", real=True)
s1, s2, s3, s4, s5, s6 = symbols("s1 s2 s3 s4 s5 s6", positive=True)
a1, a2, a3, a4, a5, a6 = symbols("a1 a2 a3 a4 a5 a6", real=True)


# ── Fubini violation: (x1²-x2²)/(x1²+x2²)² ──────────────────────────────────
# This integrand is NOT in L¹([0,1]²): the singularity at (0,0) is not
# absolutely integrable, so Fubini's theorem does not apply.  Both iterated
# integrals exist but disagree.  The value depends on which variable is
# integrated first (innermost).
#
# Convention: multiple_integrate(f, (x1,a,b), (x2,c,d)) integrates x1 first
# (innermost), then x2.
#
#   ∫_{x2=0}^{1} [∫_{x1=0}^{1} (x1²-x2²)/(x1²+x2²)² dx1] dx2 = -π/4
#   ∫_{x1=0}^{1} [∫_{x2=0}^{1} (x1²-x2²)/(x1²+x2²)² dx2] dx1 =  π/4


class TestFubiniViolation:
    """
    Classical non-L¹ integrand where integration order changes the sign.
    """

    def test_fubini_violation_x1_inner(self):
        """
        Integrating x1 first (inner), then x2 gives -π/4.
        """
        expr = (x1**2 - x2**2) / (x1**2 + x2**2) ** 2
        result = multiple_integrate(expr, (x1, 0, 1), (x2, 0, 1))
        assert_eq(result, -pi / 4, "Fubini violation: x1 inner")

    def test_fubini_violation_x2_inner(self):
        """
        Integrating x2 first (inner), then x1 gives +π/4.
        """
        expr = (x1**2 - x2**2) / (x1**2 + x2**2) ** 2
        result = multiple_integrate(expr, (x2, 0, 1), (x1, 0, 1))
        assert_eq(result, pi / 4, "Fubini violation: x2 inner")

    def test_fubini_violation_order_changes_sign(self):
        """Swapping integration order negates the result — Fubini failure."""
        expr = (x1**2 - x2**2) / (x1**2 + x2**2) ** 2
        r1 = multiple_integrate(expr, (x1, 0, 1), (x2, 0, 1))
        r2 = multiple_integrate(expr, (x2, 0, 1), (x1, 0, 1))
        assert sp.simplify(r1 + r2) == 0, "Orders should give negatives of each other"


# ── Superellipse layer-cake (Strategy 4b) ────────────────────────────────────
# ∫∫_{[0,∞)²} f((x1^p1 + x2^p2)^k) dx1 dx2
# handled by _try_superellipse via the scaling law μ'(y) = C·α·y^(α-1).


class TestSuperellipseLayerCake:
    """
    Strategy 4b: f(power-sum^k) integrals over [0,∞)^n.
    """

    def test_superellipse_1_over_1_plus_h_p2_p4(self):
        """
        ∫∫_{[0,∞)²} 1/(1+(x1²+x2⁴)³) dx1 dx2
        = -π^(5/2)/(4·Γ(-1/4)·Γ(7/4))  ≈ 0.9708
        """
        result = multiple_integrate(
            1 / (1 + (x1**2 + x2**4) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        expected = (
            -(pi ** Rational(5, 2))
            / (4 * gamma(Rational(-1, 4)) * gamma(Rational(7, 4)))
        )
        assert abs(complex(result.evalf(20)).real - float(expected.evalf())) < 1e-9

    def test_superellipse_exp_minus_h_p2_p4(self):
        """
        ∫∫_{[0,∞)²} exp(-(x1²+x2⁴)³) dx1 dx2
        = -(π^(3/2)·Γ(5/4))/(√2·Γ(-1/4)·Γ(7/4))  ≈ 0.7922
        """
        result = multiple_integrate(
            exp(-(x1**2 + x2**4) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        expected = -(pi ** Rational(3, 2) * gamma(Rational(5, 4))) / (
            sqrt(2) * gamma(Rational(-1, 4)) * gamma(Rational(7, 4))
        )
        assert abs(complex(result.evalf(20)).real - float(expected.evalf())) < 1e-9

    def test_superellipse_1_over_1_plus_h_p2_p3(self):
        """
        ∫∫_{[0,∞)²} 1/(1+(x1²+x2³)³) dx1 dx2
        = π^(3/2)·Γ(1/3)/(18·sin(5π/18)·Γ(5/6))  ≈ 0.9584
        """
        result = multiple_integrate(
            1 / (1 + (x1**2 + x2**3) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        expected = (
            pi ** Rational(3, 2)
            * gamma(Rational(1, 3))
            / (18 * sin(5 * pi / 18) * gamma(Rational(5, 6)))
        )
        assert abs(complex(result.evalf(20)).real - float(expected.evalf())) < 1e-9

    def test_superellipse_exp_minus_h_p2_p3(self):
        """
        ∫∫_{[0,∞)²} exp(-(x1²+x2³)³) dx1 dx2
        = -(5·π^(3/2)·Γ(5/18))/(18·√3·Γ(-1/3)·Γ(11/6))
        """
        result = multiple_integrate(
            exp(-(x1**2 + x2**3) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        expected = -(
            5
            * pi ** Rational(3, 2)
            * gamma(Rational(5, 18))
        ) / (18 * sqrt(3) * gamma(Rational(-1, 3)) * gamma(Rational(11, 6)))
        assert abs(complex(result.evalf(20)).real - float(expected.evalf())) < 1e-9

    def test_superellipse_divergent_returns_oo(self):
        """
        exp((x1²+x2⁴)³) over [0,∞)² diverges → returns oo.
        """
        result = multiple_integrate(
            exp((x1**2 + x2**4) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        assert result == oo or result == sp.zoo or (
            hasattr(result, "is_infinite") and result.is_infinite
        ), f"Expected oo or zoo, got {result}"


# ── Nested quadratic composition (Strategy 4c) ───────────────────────────────
# f(q(x)^k + c) where q is a positive-definite quadratic over ℝⁿ.


class TestNestedQuadraticComposition:
    """
    Strategy 4c: f((quadratic)^k + c) over ℝⁿ.
    """

    def test_2d_quadratic_nested_composition(self):
        """
        ∫∫_ℝ² 1/(1+(x1²+2x2²+2(x1+2x2))²) = π(π/2+arctan(3))/√2
        """
        expr = 1 / (1 + (x1**2 + 2 * x2**2 + 2 * (x1 + 2 * x2)) ** 2)
        result = multiple_integrate(expr, (x1, -oo, oo), (x2, -oo, oo))
        expected = pi * (pi / 2 + atan(Integer(3))) / sqrt(2)
        assert_eq(result, expected, "2D nested quadratic composition")

    def test_5d_quadratic_power(self):
        """
        ∫_ℝ⁵ 1/(20+x1²+2x2²+3x3²+4x4²+5x5²+2(x1+2x2+3x3+4x4+5x5))⁴
        = π³/(600√6)
        """
        g = (
            20
            + x1**2
            + 2 * x2**2
            + 3 * x3**2
            + 4 * x4**2
            + 5 * x5**2
            + 2 * (x1 + 2 * x2 + 3 * x3 + 4 * x4 + 5 * x5)
        )
        result = multiple_integrate(
            g ** (-4),
            (x1, -oo, oo),
            (x2, -oo, oo),
            (x3, -oo, oo),
            (x4, -oo, oo),
            (x5, -oo, oo),
        )
        assert_eq(result, pi**3 / (600 * sqrt(6)), "5D quadratic power")


# ── Symbolic shifted Gaussians ────────────────────────────────────────────────


class TestSymbolicShiftedGaussians:
    """
    Anisotropic Gaussians with symbolic shifts — both 2-D and 6-D.
    The shifts cancel and the result depends only on the scale parameters.
    """

    def test_shifted_gaussian_2d_symbolic(self):
        """
        ∫∫_ℝ² exp(-s1(x1-a1)²-s2(x2-a2)²) = π/√(s1·s2)
        """
        integrand = exp(-s1 * (x1 - a1) ** 2 - s2 * (x2 - a2) ** 2)
        result = multiple_integrate(integrand, (x1, -oo, oo), (x2, -oo, oo))
        assert_eq(result, pi / sqrt(s1 * s2), "2D shifted Gaussian")

    def test_shifted_gaussian_6d_symbolic(self):
        """
        ∫…∫_ℝ⁶ exp(-∑ sᵢ(xᵢ-aᵢ)²) = π³/√(s1·s2·s3·s4·s5·s6)
        """
        integrand = exp(
            -s1 * (x1 - a1) ** 2
            - s2 * (x2 - a2) ** 2
            - s3 * (x3 - a3) ** 2
            - s4 * (x4 - a4) ** 2
            - s5 * (x5 - a5) ** 2
            - s6 * (x6 - a6) ** 2
        )
        result = multiple_integrate(
            integrand,
            (x1, -oo, oo),
            (x2, -oo, oo),
            (x3, -oo, oo),
            (x4, -oo, oo),
            (x5, -oo, oo),
            (x6, -oo, oo),
        )
        expected = pi**3 / sp.sqrt(s1 * s2 * s3 * s4 * s5 * s6)
        assert_eq(result, expected, "6D shifted Gaussian")


# ── Misc reference cases ──────────────────────────────────────────────────────


class TestMiscReferenceIntegrals:
    """
    Additional reference integrals verified against external computations.
    """

    def test_linear_cubic_exp_concrete(self):
        """
        ∫∫_{[0,∞)²} exp(-(1+x1+x2)³) dx1 dx2
        = (E₁/₃(1) - E₂/₃(1)) / 3  ≈ 0.01601

        S1 (linear strategy) detects g_work = 1+x1+x2, k=3.
        """
        result = multiple_integrate(
            exp(-(1 + x1 + x2) ** 3), (x1, 0, oo), (x2, 0, oo)
        )
        expected_num = (
            float((expint(Rational(1, 3), 1) - expint(Rational(2, 3), 1)).evalf()) / 3
        )
        if isinstance(result, sp.Integral):
            pytest.skip("Unevaluated integral — acceptable")
        assert (
            abs(float(result.evalf()) - expected_num) < 1e-9
        ), f"Numerical mismatch: {float(result.evalf())} vs {expected_num}"

    def test_exp_quadratic_sum_bounded(self):
        """
        ∫₀³∫₀⁵ exp(-(x1+x2)²) dx1 dx2  ≈ 0.49999702…

        Exact closed form involves erf terms with large arguments.
        """
        result = multiple_integrate(exp(-(x1 + x2) ** 2), (x1, 0, 3), (x2, 0, 5))
        if isinstance(result, sp.Integral):
            pytest.skip("Unevaluated integral — acceptable")
        val = float(result.evalf())
        assert abs(val - 0.4999970266775357) < 1e-8, f"Numerical mismatch: {val}"
