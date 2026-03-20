# Testing Guide

The test suite lives in `test_multiple_integrate.py` and uses **pytest**.
It contains 138 tests across 18 classes, covering all nine strategies plus
cross-cutting mathematical properties.

---

## Running the tests

```bash
# Run all tests
pytest test_multiple_integrate.py -v

# Stop at first failure
pytest test_multiple_integrate.py -v -x

# Run one class
pytest test_multiple_integrate.py::TestStrategy6Monotone -v

# Run one test
pytest test_multiple_integrate.py::TestStrategy6Monotone::test_log_monotone_increasing -v

# Show timing
pytest test_multiple_integrate.py -v --durations=20

# Suppress SymPy warnings (common in test output)
pytest test_multiple_integrate.py -v -W ignore::UserWarning
```

### Path setup

The test file adds `../outputs` to `sys.path` to import `multiple_integrate`.
If your directory layout differs, adjust the `sys.path.insert` line near the top:

```python
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'outputs'))
```

---

## Test infrastructure

### `sym_eq(result, expected, tol=1e-12)`

Returns `True` if `result == expected` symbolically. Procedure:

1. Try `sympy.simplify(sympy.expand(result - expected)) == 0`.
2. If that fails (SymPy couldn't simplify the difference), fall back to a
   floating-point evaluation: `abs(complex(diff.evalf(30))) < tol`.

This handles cases where SymPy produces different-but-equivalent forms
(e.g. `log(2)/2` vs `log(sqrt(2))`).

### `assert_eq(result, expected, label="")`

Asserts `sym_eq(result, expected)` with a descriptive failure message showing
both the result and the expected value.

### `assert_diverges(result)`

Accepts `oo`, `-oo`, `zoo` (complex infinity), or an unevaluated `sympy.Integral`
(since SymPy sometimes returns unevaluated forms for divergent integrals).

### `_num_check(result, expected_float, tol=1e-8)`

Used in `TestNumericalSpotCheck`. Evaluates `result` to 20 significant figures
and compares to a Python float. Skips the test (`pytest.skip`) if `result` is
an unevaluated `Integral`.

---

## Test classes

### §A — `TestDecompose` (10 tests)

Unit tests for `_decompose()` directly. Tests all five detection branches:

| Test | Branch | Expression |
|---|---|---|
| `test_polynomial_univariate` | 1 | `x³ + 2x` |
| `test_polynomial_multivariate` | 1 | `x²y + y³` |
| `test_single_arg_composite_exp` | 2 | `exp(x²+y)` |
| `test_single_arg_composite_sin` | 2 | `sin(x+y)` |
| `test_single_arg_composite_log` | 2 | `log(x)` |
| `test_power_constant_exponent` | 3 | `(x²+1)^(3/2)` |
| `test_constant_factor_peeled` | 4 | `3·sin(x)` |
| `test_constant_addend_peeled` | 4 | `sin(x)+2` |
| `test_single_active_variable` | 5 | `exp(-x)` in `[x,y]` |
| `test_undecomposable_returns_none` | — | `sin(x)·cos(y)` |

### §B — `TestStrategy1Linear` (7 tests)

Tests for S1. Verifies the simplex measure formula and that S1 is bypassed
when the domain is bounded.

Key tests: 1-D, 2-D, 3-D linear exponential; non-unit $b$ coefficients;
divergent polynomial $f$ (should return `oo`); bypass test on bounded domain.

### §C — `TestStrategy2QuadraticInfinite` (6 tests)

Tests for S2. Verifies 2-D and 3-D Gaussians, shifted Gaussians, anisotropic
Gaussians. Includes a divergence test (`exp(+x²+y²)`) and a weighted Gaussian.

### §D — `TestStrategy3QuadraticEvenHalf` (4 tests)

Tests for S3. Half-line Gaussian ($\sqrt{\pi}/2$), quarter-plane ($\pi/4$),
mixed full/half ($\pi/2$), and an evenness-bypass test (`x·exp(-x²)` → S6).

### §E — `TestStrategy4GeneralPolynomial` (8 tests)

Tests for S4. Cubic and quintic on `[0,1]`; double integral on unit square;
triangle domain; triple integral; non-analytic $f$ (`Abs(x)`, `sign(x)`) with
polynomial $g$.

### §F — `TestStrategy5Separable` (6 tests)

Tests for S5. Additive trig and exponential; 3-D sum; logarithm sum; Heaviside
of a linear sum (discontinuous $g$). Verifies convolution of marginal densities.

### §G — `TestStrategy6Monotone` (12 tests)

Tests for S6. Exponential (increasing and decreasing); logarithm on `[1,e]` and
`[0,1]`; arctan; sqrt; power law; half-Gaussian; monotone with a free dimension;
non-analytic $f$ with monotone $g$ (`Abs(exp(x)-e/2)`); two divergence tests.

### §H — `TestStrategy7PiecewiseMonotone` (10 tests)

Tests for S7. `sin(x)` on `[0,π]` and `[0,2π]`; `cos(x)` on `[0,2π]`;
`Abs(x)` on `[-1,1]`; `x²` on `[-1,1]`; `sin²(x)` on `[0,π]`;
`cos(x)·exp(-x)` on `[0,π]`; `sin(x)` with a free dimension; `Abs(cos(x))`;
`Abs(sin(x))` over a full period.

### §I — `TestStrategy8GeneralNonpoly` (3 tests)

Tests for S8. Product integrands `sin(x)·cos(y)`, `exp(-x·y)`, `cos(x·y)`.
Uses `sym_eq` or checks for an unevaluated `Integral` (S8 may fall through to S9
for some of these).

### §J — `TestStrategy9Fallback` (6 tests)

Tests for S9. Product `xy`; variable-limit triangle; `exp(y/x)`;
`x·sin(x)` (integration by parts); `x·log(x)`; `sin(x)·cos(y)`.

### §K — `TestConvergenceDivergence` (11 tests)

5 convergent integrals and 6 divergent integrals. Tests the p-test at both
infinity ($\int_1^\infty x^p$) and the origin ($\int_0^1 x^p$), Gaussian
convergence, and multi-dimensional divergence.

### §L — `TestAnalyticVsNonAnalytic` (9 tests)

2 smooth functions, 4 Lipschitz / non-C¹ functions (`Abs`, ReLU), and 3
discontinuous functions (`Heaviside`, `sign`, `Piecewise`).

### §M — `TestContinuousVsDiscontinuousG` (6 tests)

Varies the continuity of the **inner function** $g$ rather than $f$:
smooth $g$ (sin), non-differentiable $g$ (`Abs(x)`), discontinuous $g$
(`Heaviside`, `sign`), and composite `exp(Heaviside(·))`.

### §N — `TestMonotoneVsNonMonotone` (8 tests)

4 strictly monotone functions (exp, exp(-x), log, rational) and 4 non-monotone
functions (cos, x², sin², |x³|).

### §O — `TestPolynomialVsNonPolynomial` (11 tests)

Degree-1, 2, 3 polynomials vs. non-polynomial: exp, sin, log, 1/(1+x²), sqrt,
and `exp(sin(x))` (transcendental composite).

### §P — `TestMixedProperties` (9 tests)

Combinations of hard properties: `|sin(x)|`, `|cos(x)|`,
`Heaviside(sin(x)-1/2)`, `|x-y|` (2-D), `exp(-|x|)`, `sin(x)·Heaviside`,
`|x|` on ℝ (divergent), 2-D Gaussian with polynomial factor, 3-D separable mixed.

### §Q — `TestEdgeCases` (7 tests)

Zero integrand, constant integrand, point domain (`[0,0]`), reversed limits,
constant-1 integrand, `x^100`, invalid range (raises `ValueError`).

### §R — `TestNumericalSpotCheck` (5 tests)

Float verification of Gaussian 2-D, `sin(x)`, arctan, separable cos, triangle.
Guards against symbolic simplification producing wrong-looking-but-right forms.

---

## Coverage map

The table below maps each mathematical property to the test classes that cover it.

| Property | Primary class(es) |
|---|---|
| S1 preconditions and formula | `TestStrategy1Linear` |
| S2 preconditions and formula | `TestStrategy2QuadraticInfinite` |
| S3 preconditions and formula | `TestStrategy3QuadraticEvenHalf` |
| S4 general polynomial | `TestStrategy4GeneralPolynomial` |
| S5 separable | `TestStrategy5Separable` |
| S6 monotone substitution | `TestStrategy6Monotone` |
| S7 piecewise-monotone | `TestStrategy7PiecewiseMonotone` |
| S8 general non-poly | `TestStrategy8GeneralNonpoly` |
| S9 fallback | `TestStrategy9Fallback` |
| Decomposition branches 1–5 | `TestDecompose` |
| Convergent improper integrals | `TestConvergenceDivergence`, `TestStrategy6Monotone` |
| Divergent integrals | `TestConvergenceDivergence`, `TestStrategy1Linear`, `TestMixedProperties` |
| Analytic smooth $f$ | `TestAnalyticVsNonAnalytic` |
| Lipschitz / non-C¹ $f$ | `TestAnalyticVsNonAnalytic`, `TestStrategy7PiecewiseMonotone` |
| Discontinuous $f$ (Heaviside, sign) | `TestAnalyticVsNonAnalytic`, `TestMixedProperties` |
| Continuous but kinked $g$ (`Abs`) | `TestContinuousVsDiscontinuousG`, `TestStrategy7PiecewiseMonotone` |
| Discontinuous $g$ (Heaviside, sign) | `TestContinuousVsDiscontinuousG`, `TestStrategy5Separable` |
| Strictly monotone $g$ | `TestMonotoneVsNonMonotone`, `TestStrategy6Monotone` |
| Non-monotone $g$ | `TestMonotoneVsNonMonotone`, `TestStrategy7PiecewiseMonotone` |
| Polynomial $g$ | `TestPolynomialVsNonPolynomial`, `TestStrategy1Linear`–`TestStrategy4` |
| Non-polynomial $g$ | `TestPolynomialVsNonPolynomial`, `TestStrategy5Separable`–`TestStrategy8` |
| Numerical float verification | `TestNumericalSpotCheck` |
| Edge cases (zero, constant, reversed) | `TestEdgeCases` |

---

## Writing new tests

### Choosing a test class

Add tests to the most specific existing class if the new test fits cleanly.
If not, create a new class following the naming convention `TestXxx`.

### Using `assert_eq`

Always use `assert_eq` rather than `==` for SymPy results — raw equality
comparison is unreliable because `simplify` is not always called automatically:

```python
# Wrong
assert multiple_integrate(x**2, (x, 0, 1)) == Rational(1, 3)

# Correct
assert_eq(multiple_integrate(x**2, (x, 0, 1)), Rational(1, 3))
```

### Verifying which strategy fires

The test suite does not assert strategy routing directly (strategies are private
functions). To test that a specific strategy is used, call the strategy function
directly in `TestDecompose` or add a unit test:

```python
from multiple_integrate import _try_linear
from sympy import Lambda, Dummy, exp, oo

t = Dummy('t')
result = _try_linear(
    Lambda(t, exp(-t)),   # f_outer
    x + y,                # g
    [x, y],               # vars_
    [(x, 0, oo), (y, 0, oo)],  # ranges
    {}                    # opts
)
assert_eq(result, 1)
```

### Divergence tests

When expecting divergence, use `assert_diverges(result)`:

```python
def test_my_divergent_integral(self):
    result = multiple_integrate(1/x**2, (x, 0, 1))  # non-integrable at 0... wait, this converges
    # ∫_0^1 x^{-2} dx diverges (p = -2 < -1 at 0)
    result = multiple_integrate(x**(-2), (x, 0, 1))
    assert_diverges(result)
```

### Slow tests

SymPy can be slow for complex integrals. Mark expected slow tests with:

```python
@pytest.mark.slow
def test_complex_heaviside(self):
    ...
```

Then run fast tests only:

```bash
pytest test_multiple_integrate.py -v -m "not slow"
```

---

## Known limitations

- **SymPy timeout:** Some integrals cause SymPy's simplification to run for a
  very long time. Strategies that call `sympy.integrate` internally may appear
  to hang. Add a `pytest.mark.timeout(30)` decorator if needed.
- **Non-elementary results:** Some convergent integrals have non-elementary
  closed forms (Ei, erfi, etc.). SymPy returns these as unevaluated `Integral`
  in some versions; the tests for such cases use the float fallback or check
  `not result.has(sp.Integral)` rather than asserting an exact value.
- **Conditional results:** When `generate_conditions=True`, results may be
  wrapped in `ConditionalExpression`. The test suite uses `generate_conditions=False`
  (the default) throughout.
