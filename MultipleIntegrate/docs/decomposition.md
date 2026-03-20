# Decomposition

The decomposition layer is the first thing `multiple_integrate` does: it tries to
recognise the integrand $F(\mathbf{x})$ as a composition $f(g(\mathbf{x}))$ where
$f$ is univariate and $g$ maps the integration variables to a scalar.

This is the **gating mechanism** for the whole strategy cascade. A `Decomposition`
object carries `(f_outer, g_inner, is_polynomial)` forward to every strategy;
`is_polynomial` determines which strategies are even attempted.

---

## Why decomposition is necessary

The layer-cake formula requires an explicit $f$ and $g$:

$$\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x} = \int f(y)\,\mu'(y)\,dy$$

Without knowing what $f$ is, we cannot form the final 1-D integral $\int f(y)\,\mu'(y)\,dy$.
Without knowing what $g$ is, we cannot compute $\mu(y) = \int_\Omega \Theta(y - g)\,d\mathbf{x}$.

The decomposition step extracts both from the SymPy expression tree.

---

## The `_decompose` function

```python
_decompose(expr: sympy.Expr, vars_: list[sympy.Symbol]) -> Decomposition | None
```

Five detection branches are tried in order. The first successful branch returns
a `Decomposition`; all others are skipped.

---

### Branch 1 — Polynomial

**Condition:** `sympy.Poly(expr, *vars_)` succeeds without raising `PolynomialError`.

**Result:** `f = identity` (i.e. $f(t) = t$), `g = expr`, `is_polynomial = True`.

**Rationale:** A polynomial in $\mathbf{x}$ is the simplest case. The "inner function"
is the entire expression and the "outer function" is the identity. This gates
Strategies 1–4.

**Examples**

| Expression | `g_inner` | `is_polynomial` |
|---|---|---|
| `x**3 + 2*x` | `x**3 + 2*x` | `True` |
| `x**2 * y + y**3` | `x**2*y + y**3` | `True` |
| `x + y + z` | `x + y + z` | `True` |
| `3` (constant) | `3` | `True` |

---

### Branch 2 — Single-argument composite

**Condition:** `len(expr.args) == 1` **and** `expr.args[0]` depends on `vars_`.

This matches any SymPy function with a single argument: `exp`, `sin`, `cos`,
`tan`, `asin`, `log`, `sinh`, `cosh`, and so on.

**Result:** `f = head(·)` (the outer function), `g = expr.args[0]` (the argument),
`is_polynomial = _is_polynomial(g, vars_)`.

**How `f_outer` is built:**

```python
t = Dummy('t')
outer = sympy.Lambda(t, expr.subs(inner, t))
```

This replaces the inner argument with a fresh dummy variable, producing a
`Lambda` that reproduces the original function when applied to any expression.

**Examples**

| Expression | `f_outer(t)` | `g_inner` | `is_polynomial` |
|---|---|---|---|
| `exp(x**2 + y)` | `exp(t)` | `x**2 + y` | `True` |
| `sin(x - y)` | `sin(t)` | `x - y` | `True` |
| `log(x*y)` | `log(t)` | `x*y` | `True` |
| `cos(exp(x))` | `cos(t)` | `exp(x)` | `False` |
| `sqrt(x**2 + 1)` | — | — | (caught by Branch 3) |

**Edge case:** `sqrt(expr)` in SymPy is `Pow(expr, 1/2)`, which has two arguments
and is not caught by Branch 2. It is caught by Branch 3 instead.

---

### Branch 3 — Power with constant exponent

**Condition:** `expr.is_Pow` **and** the exponent is free of `vars_`.

**Result:** `f = (·)**exp`, `g = base`, `is_polynomial = _is_polynomial(base, vars_)`.

**Examples**

| Expression | `f_outer(t)` | `g_inner` | `is_polynomial` |
|---|---|---|---|
| `(x**2 + 1)**(3/2)` | `t**(3/2)` | `x**2 + 1` | `True` |
| `sqrt(x**2 + y**2)` | `sqrt(t)` | `x**2 + y**2` | `True` |
| `sin(x)**2` | `t**2` | `sin(x)` | `False` |
| `exp(-x**2)` | — | — | (is `Pow(E, -x**2)`, caught here) |

**Note:** `exp(-x**2)` in SymPy is actually `E**(-x**2)`, a `Pow` with `E` as
base and `-x**2` as exponent. This is caught by Branch 3 only if the exponent
depends on `vars_` and the base is free of them. In practice, `exp(expr)` has
a single argument in SymPy's representation and is caught by Branch 2 first.

---

### Branch 4 — Constant factor or addend peeling

**Condition (multiplicative):** `expr.is_Mul` and at least one factor is free of
`vars_` and at least one factor depends on `vars_`.

**Condition (additive):** `expr.is_Add` and at least one term is free of `vars_`
and at least one term depends on `vars_`.

**Algorithm:** Split into `const_part` (free of `vars_`) and `var_part` (depends
on `vars_`), recurse on `var_part`, then adjust `f_outer` to account for the
constant.

For a multiplicative peel:
```
f_outer_new(t) = const_part * f_outer_from_recursion(t)
```

For an additive peel:
```
f_outer_new(t) = f_outer_from_recursion(t) + const_part
```

**Examples**

| Expression | `const_part` | `var_part` | `f_outer(t)` | `g_inner` |
|---|---|---|---|---|
| `3 * sin(x)` | `3` | `sin(x)` | `3*t` | `x` |
| `2 * exp(-x**2)` | `2` | `exp(-x**2)` | `2*exp(t)` | `-x**2` |
| `sin(x) + 2` | `2` | `sin(x)` | `t + 2` | `x` |
| `-exp(-x)` | `-1` | `exp(-x)` | `-exp(t)` | `-x` |

**Recursion depth:** The recursion can be nested (e.g. `3 * (sin(x) + 2)` would
peel the `3`, recurse on `sin(x) + 2`, which then peels the `+ 2`).

---

### Branch 5 — Single active variable fallback

**Condition:** Exactly one element of `vars_` appears in `expr.free_symbols`.

**Result:** `f = identity`, `g = expr`, `is_polynomial = _is_polynomial(expr, vars_)`.

**Rationale:** Any expression in a single variable can be treated as $g$ with the
identity outer function. This enables Strategies 6 and 7 for arbitrary
single-variable integrands: `tan(x)`, `1/x`, `x * log(x)`, etc.

**Examples**

| Expression | `g_inner` | `is_polynomial` |
|---|---|---|
| `tan(x)` | `tan(x)` | `False` |
| `1/x` | `1/x` | `False` |
| `x * log(x)` | `x * log(x)` | `False` |
| `exp(-x) * sin(x)` | `exp(-x)*sin(x)` | `False` |

---

### When `_decompose` returns `None`

Decomposition fails and returns `None` when none of the five branches match.
In practice this only happens for multi-variable expressions that:

1. Are not polynomials
2. Don't have a single-argument structure
3. Don't have a constant-exponent power structure
4. Cannot have their constant parts peeled
5. Depend on more than one variable

**Example:** `sin(x) * cos(y)` — two active variables, not a sum, not a product
of a constant and something with one variable. `_decompose` returns `None` and
the fallback (S9) is used immediately.

However, `sin(x) * cos(y)` in SymPy is `Mul(sin(x), cos(y))`. The `Mul`-peeling
branch (Branch 4) requires one factor to be *free of vars_*, which is not the case
here. So `_decompose` returns `None`.

---

## The `is_polynomial` flag

The `is_polynomial` attribute of the returned `Decomposition` controls which
strategies are attempted:

```
is_polynomial = True   →  try S1, S2, S3, S4, then S5, S6, S7, S8, S9
is_polynomial = False  →  skip S1–S4, try S5, S6, S7, S8, S9
```

This is a significant optimisation: Strategies 1–4 involve polynomial coefficient
extraction (`_coefficient_arrays`) and the quadratic positive-definiteness check,
both of which would fail or be meaningless for non-polynomial `g`. Skipping them
avoids wasted computation and spurious errors.

---

## Interaction with SymPy's expression tree

SymPy represents expressions as trees. Key structural properties used by `_decompose`:

| SymPy property | Meaning | Used in branch |
|---|---|---|
| `expr.args` | Immediate children of the expression node | 2, 3, 4 |
| `len(expr.args) == 1` | Single-child node | 2 |
| `expr.is_Pow` | Expression is a power `base**exp` | 3 |
| `expr.is_Mul` | Expression is a product | 4 |
| `expr.is_Add` | Expression is a sum | 4 |
| `expr.free_symbols` | Set of symbols appearing in expr | All |

**Gotcha:** SymPy normalises some expressions. For example, `exp(x)**2` becomes
`exp(2*x)` (a single-argument composite). This means Branch 2 catches it even
though the "original" form looked like Branch 3.

---

## Extending the decomposition

To handle a new class of expressions, add a new branch to `_decompose` before the
final `return None`. The branch should:

1. Check whether `expr` matches the pattern structurally
2. Extract `g_inner` and build a `sympy.Lambda` for `f_outer`
3. Determine `is_polynomial` via `_is_polynomial(g_inner, vars_)`
4. Return a `Decomposition`; return `None` if the pattern doesn't match

See the [Contributing](contributing.md) guide for a full example.
