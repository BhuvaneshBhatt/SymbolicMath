# API Reference

This page documents every public symbol and every internal function in
`multiple_integrate.py`, including their signatures, parameters, return types,
and precise behaviour.

---

## Public API

### `multiple_integrate`

```python
def multiple_integrate(
    f: sympy.Expr,
    *ranges,
    assumptions=None,
    generate_conditions: bool = False,
    principal_value: bool = False,
) -> sympy.Expr
```

Symbolically evaluate the multiple integral $\int_\Omega f(\mathbf{x})\,d\mathbf{x}$.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `f` | `sympy.Expr` | The integrand, expressed as a SymPy expression in the integration variables. Must be a SymPy expression, not a Python callable. |
| `*ranges` | `tuple(symbol, lower, upper)` | One 3-tuple per integration variable. Each tuple contains the integration variable (a SymPy `Symbol`), the lower limit, and the upper limit. Limits may be finite expressions, `sympy.oo`, or `-sympy.oo`. |
| `assumptions` | `dict`, optional | Extra assumptions forwarded to SymPy's `integrate` (e.g. `{'positive': True}`). Default `None`. |
| `generate_conditions` | `bool` | If `True`, ask SymPy to emit `ConditionalExpression` when results depend on parameter signs. Default `False`. |
| `principal_value` | `bool` | Use the Cauchy principal value for improper integrals. Default `False`. |

**Returns**

`sympy.Expr` — The closed-form result. If no strategy yields a closed form,
an unevaluated `sympy.Integral` is returned.

**Raises**

`ValueError` — If any range tuple does not have exactly three elements.

**Examples**

```python
from sympy import symbols, exp, sin, cos, sqrt, pi, oo, log, Abs, E
from multiple_integrate import multiple_integrate

x, y, z = symbols('x y z', real=True)

# Polynomial
multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 2))
# → 2/3

# Gaussian
multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
# → pi

# Separable trig
multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))
# → 0

# Monotone substitution
multiple_integrate(1/(1 + x**2), (x, 0, 1))
# → pi/4

# Piecewise-monotone
multiple_integrate(sin(x), (x, 0, pi))
# → 2

# Non-analytic f
multiple_integrate(Abs(x), (x, -1, 1))
# → 1

# Variable limits (triangle)
multiple_integrate(x + y, (y, 0, 1 - x), (x, 0, 1))
# → 1/3

# Divergent (returns oo)
multiple_integrate(1/x, (x, 0, 1))
# → oo
```

**Notes**

- Variables are ordered as listed in `ranges`. The fallback strategy (S9)
  integrates in reverse order (innermost first), which matches standard
  mathematical convention.
- The function does not pre-validate convergence; divergent integrals either
  return `oo`/`-oo` or an unevaluated `Integral`.

---

## `Decomposition` class

```python
class Decomposition:
    f_outer: Callable
    g_inner: sympy.Expr
    is_polynomial: bool
```

A lightweight container (using `__slots__`) that records the result of decomposing
an integrand $F(\mathbf{x})$ into $f \circ g$.

**Attributes**

| Attribute | Type | Description |
|---|---|---|
| `f_outer` | `Callable` | Univariate function. Given a SymPy expression `t`, returns `f(t)`. Typically a `sympy.Lambda`. |
| `g_inner` | `sympy.Expr` | The inner expression in the integration variables, i.e. $g(\mathbf{x})$. |
| `is_polynomial` | `bool` | `True` if `g_inner` is a polynomial in the integration variables. Determines whether strategies 1–4 are attempted. |

**Constructor**

```python
Decomposition(f_outer: Callable, g_inner: sympy.Expr, is_polynomial: bool)
```

---

## Internal functions

### `_decompose`

```python
def _decompose(
    expr: sympy.Expr,
    vars_: list[sympy.Symbol]
) -> Decomposition | None
```

Attempts to write `expr` as $f(g(\mathbf{x}))$ where $g$ depends on `vars_`
and $f$ is univariate. Returns a `Decomposition` on success, `None` on failure.

**Detection branches (in order)**

1. **Polynomial** — `sympy.Poly(expr, *vars_)` succeeds → `f = identity`, `g = expr`
2. **Single-argument composite** — `len(expr.args) == 1` and the argument depends
   on `vars_` → `f = head`, `g = argument` (catches `exp`, `sin`, `cos`, `log`, etc.)
3. **Power with constant exponent** — `expr = base**exp` where `exp` is free of
   `vars_` → `f = (·)**exp`, `g = base`
4. **Constant factor / addend** — `expr = c * h(x)` or `c + h(x)` where `c` is
   free of `vars_` → peel `c`, recurse on `h(x)`
5. **Single active variable** — `expr` depends on exactly one element of `vars_`
   → `f = identity`, `g = expr` (enables S6 and S7 for any single-variable expression)

Returns `None` if none of the five branches succeed.

---

### `_is_polynomial`

```python
def _is_polynomial(expr: sympy.Expr, vars_: list[sympy.Symbol]) -> bool
```

Returns `True` if `expr` is a polynomial in the symbols listed in `vars_`,
using `sympy.Poly` for detection. Returns `False` on `PolynomialError`.

---

### `_coefficient_arrays`

```python
def _coefficient_arrays(
    poly: sympy.Expr,
    vars_: list[sympy.Symbol]
) -> tuple[sympy.Expr, sympy.Matrix, sympy.Matrix]
```

Extracts $(c, \mathbf{b}, A)$ from a degree-≤2 polynomial:

$$\text{poly}(\mathbf{x}) = \mathbf{x}^\top A \mathbf{x} + \mathbf{b}^\top \mathbf{x} + c$$

**Returns** `(c, b, A)` where:
- `c` — constant term (SymPy expression)
- `b` — $n \times 1$ `sympy.Matrix` of linear coefficients
- `A` — symmetric $n \times n$ `sympy.Matrix` of quadratic coefficients

**Raises** `ValueError` if `poly` has degree > 2.

---

### `_is_even_function`

```python
def _is_even_function(expr: sympy.Expr, var: sympy.Symbol) -> bool
```

Returns `True` if `sympy.simplify(expr.subs(var, -var) - expr) == 0`.
Used by S3 to verify that $f(g(\mathbf{x}))$ is even in each half-infinite variable.

---

### `_real_critical_points`

```python
def _real_critical_points(
    g: sympy.Expr,
    var: sympy.Symbol,
    lo: sympy.Expr,
    hi: sympy.Expr
) -> list[sympy.Expr]
```

Returns a sorted, deduplicated list of real critical points of `g` (as a function
of `var`) that lie strictly inside `(lo, hi)`.

Detects:

- **Stationary points**: solutions of $g'(x) = 0$
- **Kinks from `Abs`**: zeros of the argument of any `Abs(·)` subexpression
- **Kinks from `sqrt`**: zeros of the radicand of any `sqrt(·)` subexpression

**Sorting** is by numerical value when points are numeric; symbolic points are
placed last.

---

### `_g_range_on_interval`

```python
def _g_range_on_interval(
    g: sympy.Expr,
    var: sympy.Symbol,
    lo: sympy.Expr,
    hi: sympy.Expr
) -> tuple[sympy.Expr, sympy.Expr]
```

Returns `(g_min, g_max)` of `g` over `[lo, hi]` by evaluating at the endpoints
and at all critical points returned by `_real_critical_points`. Falls back to
`(-oo, oo)` if no finite candidates are found.

---

### `_bounds_of_g`

```python
def _bounds_of_g(
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple]
) -> tuple[sympy.Expr, sympy.Expr]
```

Estimates `(g_min, g_max)` for a multivariate `g` on the box $\prod_i [a_i, b_i]$
by evaluating `g` at all corners of the box (excluding infinite endpoints) and at
critical points along each axis. Returns symbolic `Min`/`Max` of all candidates.

---

### `_try_linear`

```python
def _try_linear(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 1. Returns `None` if preconditions are not met or if the resulting
1-D integral is unevaluated.

---

### `_qs_integrate`

```python
def _qs_integrate(
    f_outer: Callable,
    A_mat: sympy.Matrix,
    b_vec: sympy.Matrix,
    c_val: sympy.Expr,
    n: int,
    opts: dict
) -> sympy.Expr | None
```

Core engine for Strategies 2 and 3. Computes the quadratic Gaussian-type integral
using the ellipsoid surface-area formula. Requires `A_mat` positive definite.
Returns `None` if `A_mat` is not invertible, has non-positive eigenvalues, or
the 1-D integral is unevaluated.

---

### `_try_quadratic_infinite`

```python
def _try_quadratic_infinite(
    f_outer, g, vars_, ranges, opts
) -> sympy.Expr | None
```

Strategy 2. Checks all ranges are $(-\infty, \infty)$, extracts $(c, \mathbf{b}, A)$,
delegates to `_qs_integrate`.

---

### `_try_quadratic_even_half_infinite`

```python
def _try_quadratic_even_half_infinite(
    f_outer, g, vars_, ranges, opts
) -> sympy.Expr | None
```

Strategy 3. Checks the mix of $(-\infty, \infty)$ and $[0, \infty)$ ranges,
verifies evenness, calls `_qs_integrate`, divides by $2^k$.

---

### `_try_general_polynomial`

```python
def _try_general_polynomial(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 4. Integrates `Heaviside(y - g)` over all dimensions, differentiates,
then integrates against `f_outer`.

---

### `_try_separable`

```python
def _try_separable(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 5. Checks additive separability of `g`, computes marginal densities via
Heaviside integrals, convolves them iteratively, then integrates against `f_outer`.

---

### `_try_monotone_substitution`

```python
def _try_monotone_substitution(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 6. Requires exactly one active variable in `g` and no interior critical
points. Inverts `g` via `sympy.solve`, selects the correct branch, computes the
Jacobian $|dx/dy|$, integrates.

---

### `_try_piecewise_monotone`

```python
def _try_piecewise_monotone(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 7. Requires exactly one active variable in `g` and at least one interior
critical point. Splits at critical points, calls `_try_monotone_substitution` on
each piece, falls back to direct integration for individual pieces when needed.

---

### `_try_general_nonpolynomial`

```python
def _try_general_nonpolynomial(
    f_outer: Callable,
    g: sympy.Expr,
    vars_: list[sympy.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr | None
```

Strategy 8. Same algorithm as `_try_general_polynomial` but with no polynomial
restriction on `g`.

---

### `_iterated_integrate`

```python
def _iterated_integrate(
    expr: sympy.Expr,
    ranges: list[tuple],
    opts: dict
) -> sympy.Expr
```

Strategy 9 (fallback). Integrates `expr` over each variable in reverse order
(innermost first). Always returns a SymPy expression; never raises. If SymPy
cannot evaluate a step, an unevaluated `sympy.Integral` is propagated.

---

## Type conventions

| Concept | Python type |
|---|---|
| Integration variable | `sympy.Symbol` declared with `real=True` |
| Integration limit | `sympy.Expr`, `sympy.oo`, or `-sympy.oo` |
| Range tuple | `tuple[sympy.Symbol, sympy.Expr, sympy.Expr]` |
| Internal options | `dict` passed as keyword args to `sympy.integrate` |
| Strategy return | `sympy.Expr` on success, `None` on failure |

## Error handling

All strategy functions wrap their computation in `try/except` blocks and return
`None` on any exception, including `sympy.PolynomialError`,
`sympy.SolveFailed`, `NotImplementedError`, and generic `Exception`.
Only `_iterated_integrate` and `multiple_integrate` itself may propagate
exceptions (`ValueError` for malformed ranges).
