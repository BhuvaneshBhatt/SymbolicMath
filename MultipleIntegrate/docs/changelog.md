# Changelog

---

## v2.0.0

### Summary

Major extension of the integration engine to support non-polynomial inner
functions $g(\mathbf{x})$. The original implementation only handled $g$ 
that were polynomials in the integration variables. This release adds 
four new strategies (S5–S8) and a completely rewritten decomposition layer.

### New features

#### Strategies 5–8 for non-polynomial $g$

**Strategy 5 — Separable $g$**  
Handles $g(\mathbf{x}) = h_1(x_1) + h_2(x_2) + \cdots + h_n(x_n)$ where each
term depends on exactly one variable. The layer-cake density is computed as the
convolution of the marginal Lebesgue pushforward densities $\nu_i$.

Examples now handled: `cos(x+y)`, `exp(-(x+y))`, `sin(x)+sin(y)`, `log(x+y)`.

**Strategy 6 — Monotone substitution**  
For single-variable $g$ with no interior critical points on the integration
interval. Inverts $g$ analytically via `sympy.solve`, computes the Jacobian
$|dx/dy|$, and reduces to a 1-D integral.

Examples now handled: `exp(x)`, `log(x)`, `1/(1+x²)`, `sqrt(x)`, `x^(1/3)`.

**Strategy 7 — Piecewise-monotone substitution**  
Extends S6 to $g$ with interior critical points by splitting the domain at each
critical point. Also detects non-differentiable kinks (zeros of `Abs` arguments
and `sqrt` radicands).

Examples now handled: `sin(x)` on `[0,π]`, `cos(x)` on `[0,2π]`, `Abs(x)`,
`Abs(sin(x))`.

**Strategy 8 — General non-polynomial Heaviside layer-cake**  
For multi-variable non-polynomial $g$ where none of S5–S7 applies. Uses the
same Heaviside-integral algorithm as S4 but without the polynomial restriction.

Examples now handled: `sin(x)·cos(y)`, non-separable product integrands.

#### Rewritten decomposition layer

The original `_find_polynomial_in_variables` function has been replaced by
`_decompose`, which returns a `Decomposition` dataclass with fields
`(f_outer, g_inner, is_polynomial)`.

`_decompose` now recognises five structural patterns instead of one:

1. Polynomial in `vars_` (unchanged)
2. Single-argument composite: `exp(g)`, `sin(g)`, `log(g)`, etc.
3. Power with constant exponent: `base**c` where `c` is free of `vars_`
4. Constant factor/addend peeling: `c * h(x)`, `h(x) + c`
5. Single-active-variable fallback: any expression in one variable

#### Critical point detection

`_real_critical_points` detects both:
- Stationary points (zeros of $g'$)
- Non-differentiable kinks (zeros of `Abs` arguments and `sqrt` radicands)

#### Bounds estimation

`_bounds_of_g` estimates the range $[g_{\min}, g_{\max}]$ for multivariate $g$
by evaluating $g$ at all corners of the bounding box and at critical points along
each axis.

### Breaking changes

None. The public API (`multiple_integrate`) is unchanged. Existing code that
worked in v1.x continues to work in v2.0.

### Internal changes

- `_find_polynomial_in_variables` removed; replaced by `_decompose` + `Decomposition`
- `_coefficient_arrays` now raises `ValueError` for degree > 2 (previously silently
  failed)
- Strategy routing in `multiple_integrate`: polynomial strategies S1–S4 are now
  only attempted when `is_polynomial = True`; non-polynomial strategies S5–S8 are
  always attempted

---

## v1.0.0

### Summary

Initial implementation. Supports integrands of the form $f(g(\mathbf{x}))$ where $g$ is
a polynomial in the integration variables.

### Features

**Strategy 1 — Linear polynomial over $[0,\infty)^n$**  
Implements the simplex measure formula:

$$\int_{[0,\infty)^n} f(\mathbf{b}\cdot\mathbf{x}+c)\,d\mathbf{x}
= \frac{1}{\prod b_i\,(n-1)!}\int_c^\infty (y-c)^{n-1}\,f(y)\,dy$$

**Strategy 2 — Quadratic doubly-infinite**  
Implements the ellipsoid surface-area formula for Gaussian-type integrals over
$\mathbb{R}^n$.

**Strategy 3 — Quadratic even / half-infinite**  
Exploits symmetry to halve each half-infinite dimension.

**Strategy 4 — General polynomial, Heaviside layer-cake**  
Computes $\mu(y) = \int_\Omega \Theta(y - g(\mathbf{x}))\,d\mathbf{x}$ symbolically
for any polynomial $g$ on any domain that SymPy can handle.

**Strategy 9 — Fallback**  
Plain iterated `sympy.integrate`.

### Known limitations in v1.0

- Non-polynomial $g$ is not supported; such integrals always fall through to
  the iterated fallback.
- Single-argument composites like `exp(x²+y)` are recognised (Branch 2 of
  `_find_polynomial_in_variables`) but only when the argument is polynomial.
- No detection of kinks or non-differentiable points in $g$.
- `_find_polynomial_in_variables` is a flat function with no recurse-on-peel
  capability; `3*sin(x)` is not recognisable as $f \circ g$.
