# Strategies Reference

This page gives the complete specification for each of the nine strategies:
exact preconditions, algorithm steps, closed-form formula, routing examples,
and bypass conditions.

Strategies are tried in the order listed. The first one whose preconditions are
satisfied and whose computation succeeds returns the result; the remainder are
skipped. A strategy "fails" by returning `None`, which triggers the next one.

---

## Decision flow

```
multiple_integrate(f, *ranges)
│
├─ _decompose(f, vars)
│   ├─ None  →  fallback (S9)
│   └─ Decomposition(f_outer, g_inner, is_polynomial)
│
├─ if is_polynomial:
│   ├─ S1  _try_linear
│   ├─ S2  _try_quadratic_infinite
│   ├─ S3  _try_quadratic_even_half_infinite
│   └─ S4  _try_general_polynomial
│
├─ S5  _try_separable
├─ S6  _try_monotone_substitution
├─ S7  _try_piecewise_monotone
├─ S8  _try_general_nonpolynomial
└─ S9  _iterated_integrate  (fallback)
```

Strategies 5–8 are attempted for **all** integrands (including polynomial ones
that slipped through S1–S4), so they act as a safety net for polynomial $g$ on
unusual domains as well.

---

## Strategy 1 — Linear polynomial

### Preconditions

1. `is_polynomial = True` (inner function $g$ is a polynomial in the integration variables)
2. $g = \mathbf{b}\cdot\mathbf{x} + c$ — **purely linear** (the quadratic coefficient matrix $A = 0$)
3. All integration ranges are $[0, \infty)$
4. All linear coefficients $b_i \ne 0$

### Formula

$$\int_{[0,\infty)^n} f(\mathbf{b}\cdot\mathbf{x} + c)\,d\mathbf{x}
= \frac{1}{\left(\prod_{i=1}^n b_i\right)(n-1)!}
\int_c^\infty (y - c)^{n-1}\,f(y)\,dy$$

### Algorithm

1. Extract coefficients $c$ (constant), $\mathbf{b}$ (linear) using `_coefficient_arrays`.
2. Verify $A = 0$ and all $b_i \ne 0$.
3. Form the 1-D integral above with a fresh dummy variable $y$.
4. Delegate to `sympy.integrate`; return `None` if unevaluated.

### Examples

```python
# n=1: ∫_0^∞ exp(-(2x+1)) dx = exp(-1)/2
multiple_integrate(exp(-(2*x + 1)), (x, 0, oo))           # exp(-1)/2

# n=2: ∫∫_[0,∞)² exp(-(x+y)) dx dy = 1
multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo)) # 1

# n=3: ∫∫∫_[0,∞)³ exp(-(x+y+z)) dV = 1
multiple_integrate(exp(-(x+y+z)), (x,0,oo),(y,0,oo),(z,0,oo))  # 1
```

### Bypass conditions

- Domain is $[0, 1]^n$ → S4 handles it
- $g$ is quadratic → S2 or S3
- $b_i = 0$ for any $i$ → strategy returns `None`

---

## Strategy 2 — Quadratic doubly-infinite

### Preconditions

1. `is_polynomial = True`
2. $g = \mathbf{x}^\top A\mathbf{x} + \mathbf{b}\cdot\mathbf{x} + c$ with $A \ne 0$
3. All integration ranges are $(-\infty, \infty)$
4. $A$ is positive definite (checked via eigenvalues)

### Formula

Let $y_{\min} = c - \frac{1}{4}\mathbf{b}^\top A^{-1}\mathbf{b}$. Then:

$$\int_{\mathbb{R}^n} f(\mathbf{x}^\top A\mathbf{x}+\mathbf{b}\cdot\mathbf{x}+c)\,d\mathbf{x}
= \frac{\pi^{n/2}}{\sqrt{\det A}\,\Gamma(n/2+1)}
\int_{y_{\min}}^\infty \frac{n}{2}(y - y_{\min})^{n/2-1}\,f(y)\,dy$$

### Algorithm

1. Extract $c$, $\mathbf{b}$, $A$ via `_coefficient_arrays`.
2. Compute $A^{-1}$, $\det A$, and $y_{\min}$.
3. Check positive-definiteness of $A$.
4. Form the 1-D integral and delegate to `sympy.integrate`.

### Examples

```python
# 2-D Gaussian
multiple_integrate(exp(-(x**2 + y**2)), (x,-oo,oo),(y,-oo,oo))        # π

# 3-D Gaussian
multiple_integrate(exp(-(x**2+y**2+z**2)), (x,-oo,oo),(y,-oo,oo),(z,-oo,oo))  # π^(3/2)

# Anisotropic: ∫∫ exp(-(2x²+3y²)) = π/√6
multiple_integrate(exp(-(2*x**2 + 3*y**2)), (x,-oo,oo),(y,-oo,oo))    # π/√6

# With linear term: ∫∫ exp(-((x-1)²+(y+2)²)) = π  (shift doesn't change value)
multiple_integrate(exp(-((x-1)**2+(y+2)**2)), (x,-oo,oo),(y,-oo,oo))  # π
```

### Bypass conditions

- $A$ not positive definite (e.g. $g = x^2 - y^2$) → returns `None`
- Bounded domain → S4
- Mixed infinite/half-infinite → S3

---

## Strategy 3 — Quadratic even / half-infinite

### Preconditions

1. `is_polynomial = True`
2. $g$ is quadratic with positive-definite $A$
3. Each integration range is either $(-\infty, \infty)$ or $[0, \infty)$
4. $f(g(\mathbf{x}))$ is **even** in every variable whose range is $[0, \infty)$

### Formula

If $k$ dimensions have range $[0, \infty)$ and the remaining $n-k$ have $(-\infty, \infty)$:

$$\text{result} = \frac{1}{2^k} \times \text{(Strategy 2 result on all of } \mathbb{R}^n\text{)}$$

### Algorithm

1. Count half-infinite dimensions $k$ and full-infinite dimensions.
2. For each half-infinite variable, verify evenness of $f(g(\mathbf{x}))$ via
   `_is_even_function`.
3. Call `_qs_integrate` as in S2 and divide by $2^k$.

### Examples

```python
# Quarter-plane Gaussian: ∫∫_{[0,∞)²} exp(-(x²+y²)) = π/4
multiple_integrate(exp(-(x**2+y**2)), (x,0,oo),(y,0,oo))              # π/4

# Half-line Gaussian: ∫_0^∞ exp(-x²) dx = √π/2
multiple_integrate(exp(-x**2), (x, 0, oo))                             # √π/2

# Mixed: ∫_{-∞}^∞∫_0^∞ exp(-(x²+y²)) dy dx = π/2
multiple_integrate(exp(-(x**2+y**2)), (x,-oo,oo),(y,0,oo))            # π/2
```

### Bypass conditions

- `f(g(x))` is not even in a half-infinite variable → `None`
- Domain is fully bounded → S4

---

## Strategy 4 — General polynomial, Heaviside layer-cake

### Preconditions

1. `is_polynomial = True`
2. Domain is bounded or semi-infinite (any shape)
3. SymPy can integrate $\Theta(y - g(\mathbf{x}))$ dimension by dimension

### Algorithm

1. Form $\Theta(y - g(\mathbf{x}))$ symbolically.
2. Integrate over each dimension in order; abort with `None` if any step produces
   an unevaluated `Integral`.
3. Differentiate with respect to $y$ to get the density $\mu'(y)$.
4. Determine $y_{\min}$, $y_{\max}$ by evaluating $g$ at the corners of the domain.
5. Integrate $f(y)\,\mu'(y)$ over $[y_{\min}, y_{\max}]$.

### Examples

```python
# ∫_0^1 x³ dx = 1/4
multiple_integrate(x**3, (x, 0, 1))

# ∫∫_{[0,1]²} (x+y)³ dx dy
multiple_integrate((x+y)**3, (x, 0, 1), (y, 0, 1))

# Triangle domain: ∫∫ x²y over {0≤y≤1-x, 0≤x≤1} = 1/60
multiple_integrate(x**2 * y, (y, 0, 1-x), (x, 0, 1))

# ∫∫∫_{[0,1]³} xyz dV = 1/8
multiple_integrate(x*y*z, (x,0,1),(y,0,1),(z,0,1))
```

### Notes

- Handles **non-rectangular domains** via variable limits in the integration ranges.
- For high-degree polynomials the Heaviside integral can become complex; SymPy
  may time out, in which case `None` is returned and the fallback is used.

---

## Strategy 5 — Separable inner function

### Preconditions

1. $g(\mathbf{x}) = h_1(x_1) + h_2(x_2) + \cdots + h_n(x_n)$ — every term in $g$
   depends on **exactly one** variable
2. At least two variables are active in $g$
3. Each marginal Heaviside integral $\int \Theta(y - h_i(x_i))\,dx_i$ is
   computable by SymPy

### Algorithm

1. Parse $g$ into single-variable components $h_i$ and a constant residual $c_0$.
2. For each $h_i$: compute $\nu_i(y) = \frac{d}{dy}\int_{a_i}^{b_i}\Theta(y-h_i(x))\,dx$.
3. Convolve $\nu_1, \nu_2, \ldots, \nu_n$ iteratively.
4. Integrate $f(y + c_0)\cdot(\nu_1 * \cdots * \nu_n)(y)$.

### Examples

```python
# Additive separable trig: ∫∫ cos(x+y) dx dy = 0
multiple_integrate(cos(x+y), (x, 0, pi), (y, 0, pi))

# Additive exp: ∫∫ exp(-(x+y)) dx dy = 1
multiple_integrate(exp(-(x+y)), (x, 0, oo), (y, 0, oo))

# Additive trig squared: ∫∫ (sin(x)+sin(y))² dx dy
multiple_integrate((sin(x)+sin(y))**2, (x, 0, 1), (y, 0, 1))

# 3-D: ∫∫∫ sin(x+y+z) dV
multiple_integrate(sin(x+y+z), (x,0,1),(y,0,1),(z,0,1))
```

### Notes

- The separability check is structural (based on `expr.args`), not semantic.
  `sin(x)*cos(y)` is **not** separable under this strategy because it is a
  product, not a sum. S8 or S9 handles it instead.
- Variables missing from $g$ entirely contribute a volume factor.

---

## Strategy 6 — Monotone substitution

### Preconditions

1. $g$ depends on **exactly one** integration variable $x_i$
2. $g$ has **no interior critical points** on $[a_i, b_i]$ (monotone)
3. $g(x_i) = y$ has a real analytic solution $x_i = g^{-1}(y)$

### Algorithm

1. Find the active variable and its range.
2. Call `_real_critical_points`; abort with `None` if any are found.
3. Solve $g(x) = y$ for $x$; select the branch consistent with $[a_i, b_i]$.
4. Compute the Jacobian $|dx/dy| = |\,d(g^{-1})/dy\,|$.
5. Compute $g(a_i)$ and $g(b_i)$ (using limits for infinite endpoints).
6. Integrate $f(y)\cdot|dx/dy|\cdot V_{\text{other}}$ over $[g_{\min}, g_{\max}]$.

### Examples

```python
# ∫_0^1 exp(x) dx = e - 1
multiple_integrate(exp(x), (x, 0, 1))

# ∫_1^e log(x) dx = 1
multiple_integrate(log(x), (x, 1, E))

# ∫_0^1 1/(1+x²) dx = π/4
multiple_integrate(1/(1+x**2), (x, 0, 1))

# ∫_0^∞ exp(-x²) dx = √π/2
multiple_integrate(exp(-x**2), (x, 0, oo))

# ∫_0^4 √x dx = 16/3
multiple_integrate(sqrt(x), (x, 0, 4))
```

### Branch selection

When `sympy.solve` returns multiple solutions (e.g. $g = x^2$ gives $\pm\sqrt{y}$),
the code selects the branch whose midpoint value $g^{-1}((g_{\min}+g_{\max})/2)$
lies within $[a_i, b_i]$.

### Bypass conditions

- Interior critical points exist → S7
- $g$ depends on more than one variable → `None` (unless separable, handled by S5)
- `sympy.solve` returns no real solutions or more than one valid branch → `None`

---

## Strategy 7 — Piecewise-monotone substitution

### Preconditions

1. $g$ depends on exactly one variable $x_i$
2. $g$ has at least one real interior critical point on $[a_i, b_i]$
3. Strategy 6 can be applied to each sub-interval

### Algorithm

1. Collect critical points via `_real_critical_points` (stationary points plus
   points where $g$ is non-differentiable, e.g. kinks of $|x|$).
2. Sort them to form sub-intervals.
3. Apply `_try_monotone_substitution` to each sub-interval.
4. If S6 fails on a piece, fall back to direct SymPy integration on that piece.
5. Sum contributions from all pieces.

### Critical point detection

`_real_critical_points` finds:

- **Stationary points**: solutions of $g'(x) = 0$
- **Non-differentiable kinks**: zeros of arguments inside `Abs(·)` or `sqrt(·)`

### Examples

```python
# ∫_0^π sin(x) dx = 2  (critical point at π/2)
multiple_integrate(sin(x), (x, 0, pi))

# ∫_0^{2π} cos(x) dx = 0  (critical points at π)
multiple_integrate(cos(x), (x, 0, 2*pi))

# ∫_{-1}^1 |x| dx = 1  (kink at 0)
multiple_integrate(Abs(x), (x, -1, 1))

# ∫_0^π sin²(x) dx = π/2
multiple_integrate(sin(x)**2, (x, 0, pi))
```

### Notes on `Abs`

`Abs(x)` is handled specially: the kink at $x = 0$ is detected by solving the
argument for zero, then the domain is split there and S6 is applied on each half.

---

## Strategy 8 — General non-polynomial, Heaviside layer-cake

### Preconditions

1. $g$ is non-polynomial (transcendental or algebraic non-polynomial)
2. SymPy can integrate $\Theta(y - g(\mathbf{x}))$ dimension by dimension in
   closed form

### Algorithm

Identical to Strategy 4, but there is no polynomial restriction on $g$.

1. Compute $\mu(y) = \int_\Omega \Theta(y - g(\mathbf{x}))\,d\mathbf{x}$ dimension by dimension.
2. Differentiate: $\mu'(y) = d\mu/dy$.
3. Determine $y$-bounds using `_bounds_of_g` (evaluates $g$ at corners and critical points).
4. Integrate $f(y)\,\mu'(y)$.

### `_bounds_of_g`

Evaluates $g$ at all corners of the bounding box and at critical points found
by scanning each axis, takes the symbolic `Min`/`Max`.

### Examples

```python
# ∫∫ sin(x)·cos(y) dx dy  (product, not a sum — S5 doesn't apply)
multiple_integrate(sin(x)*cos(y), (x, 0, pi/2), (y, 0, pi/2))   # 1

# Non-separable non-polynomial 2-D g
multiple_integrate(exp(-x*y), (x, 0, 1), (y, 0, 1))
```

### When S8 returns `None`

SymPy cannot always integrate $\Theta(y - g)$ in closed form for transcendental
$g$ (e.g. $g = e^{xy}$). In that case S8 returns `None` and S9 handles it.

---

## Strategy 9 — Fallback (plain iterated integration)

### When it fires

- `_decompose` returned `None` (integrand not recognisable as $f(g(\mathbf{x}))$)
- All of S1–S8 returned `None` or raised exceptions

### Algorithm

Plain iterated SymPy integration in reverse order (innermost variable last in
the range list, integrated first):

```python
for r in reversed(ranges):
    result = sympy.integrate(result, (r[0], r[1], r[2]))
```

### Examples

```python
# Product integrand not of f(g) form
multiple_integrate(x * sin(x), (x, 0, pi))      # π  (integration by parts)

# Variable limits
multiple_integrate(x + y, (y, 0, 1-x), (x, 0, 1))   # 1/3

# Mixed poly/trig product
multiple_integrate(x * log(x), (x, 0, 1))        # -1/4
```

---

## Strategy comparison table

| Property | S1 | S2 | S3 | S4 | S5 | S6 | S7 | S8 | S9 |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| Polynomial $g$ required | ✓ | ✓ | ✓ | ✓ | — | — | — | — | — |
| Non-polynomial $g$ | — | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ |
| Infinite domain required | ✓ | ✓ | ✓ | — | — | — | — | — | — |
| Bounded domain | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Variable limits | — | — | — | ✓ | — | — | — | — | ✓ |
| Multi-variable $g$ | ✓ | ✓ | ✓ | ✓ | ✓ | — | — | ✓ | ✓ |
| Single-variable $g$ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| Separable $g = \sum h_i$ | — | — | — | — | ✓ | — | — | — | ✓ |
| Non-monotone $g$ | — | — | — | — | — | — | ✓ | ✓ | ✓ |
| Reduces to 1-D integral | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ | — |

`—` means "not required / not applicable" (the strategy still handles it).
