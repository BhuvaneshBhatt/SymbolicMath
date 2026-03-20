# Contributing

This guide explains how to add new strategies, extend the decomposition layer,
and maintain the quality of the codebase.

---

## Repository layout

```
multiple_integrate.py          # Library source
test_multiple_integrate.py     # Test suite (pytest)
multiple_integration.ipynb     # Tutorial notebook
docs/
  index.md
  theory.md
  strategies.md
  api.md
  decomposition.md
  examples.md
  testing.md
  contributing.md
  changelog.md
```

---

## Internal data flow

Understanding the data flow through `multiple_integrate` is essential before
modifying it.

```
multiple_integrate(f, *ranges)
│
│  Parse and normalise ranges
│  vars_ = [r[0] for r in parsed_ranges]
│
├─ _decompose(f_expr, vars_)
│  │
│  │  Returns Decomposition(f_outer, g_inner, is_polynomial)
│  │  or None → fall directly to _iterated_integrate
│  │
│  f_outer : Callable  (Lambda  t  →  f(t))
│  g_inner : Expr      (the inner function g(x₁,…,xₙ))
│  is_polynomial : bool
│
├─ if is_polynomial:
│   _try_linear(f_outer, g, vars_, ranges, opts)          → Expr | None
│   _try_quadratic_infinite(...)                           → Expr | None
│   _try_quadratic_even_half_infinite(...)                 → Expr | None
│   _try_general_polynomial(...)                           → Expr | None
│
├─ _try_separable(f_outer, g, vars_, ranges, opts)         → Expr | None
├─ _try_monotone_substitution(...)                         → Expr | None
├─ _try_piecewise_monotone(...)                            → Expr | None
├─ _try_general_nonpolynomial(...)                         → Expr | None
│
└─ _iterated_integrate(f_expr, ranges, opts)               → Expr (always)
```

Each strategy function receives:

| Argument | Type | Meaning |
|---|---|---|
| `f_outer` | `Callable` | The outer function: `f_outer(y)` gives `f(y)` |
| `g` | `sympy.Expr` | The inner expression in the integration variables |
| `vars_` | `list[Symbol]` | Integration variables in declaration order |
| `ranges` | `list[tuple]` | `[(var, lo, hi), ...]` |
| `opts` | `dict` | Keyword args forwarded to `sympy.integrate` |

And returns either a SymPy expression (success) or `None` (strategy inapplicable
or failed).

---

## Adding a new strategy

### Step 1 — Decide where it fits in the cascade

Strategies are tried in order. A new strategy should be placed:

- **Before S5** if it requires `is_polynomial = True` (add to the `if is_poly` block)
- **After S4 and before S9** if it handles non-polynomial $g$
- **Before S9 always** — S9 is the unconditional fallback

### Step 2 — Write the function

Follow the existing signature exactly:

```python
def _try_mymethod(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict
) -> sp.Expr | None:
    """
    One-line summary.

    Preconditions
    -------------
    • Condition 1
    • Condition 2

    Formula
    -------
    ∫_Ω f(g(x)) dx = ...

    Returns None if any precondition fails or the result is unevaluated.
    """
    # 1. Check preconditions — return None immediately if not met
    if <not applicable>:
        return None

    # 2. Compute μ'(y) or the 1-D integrand
    y = sp.Dummy('y_mymethod')
    ...

    # 3. Integrate — always check for unevaluated result
    try:
        result = integrate(f_outer(y) * density, (y, y_lo, y_hi), **opts)
        if result.has(sp.Integral):
            return None
        return simplify(result)
    except Exception:
        return None
```

**Key rules:**
- Always return `None` (not raise) when inapplicable or failed.
- Wrap all `sympy` calls in `try/except Exception: return None`.
- Use `Dummy` variables (never reuse user variable names inside the function).
- Check `result.has(sp.Integral)` before returning.

### Step 3 — Register the strategy

Add a call in `multiple_integrate`:

```python
# For non-polynomial g (after S4):
for strategy in (_try_separable,
                 _try_monotone_substitution,
                 _try_piecewise_monotone,
                 _try_general_nonpolynomial,
                 _try_mymethod,            # ← add here
                 ):
    res = strategy(f_outer, g, vars_, parsed_ranges, opts)
    if res is not None:
        return res
```

For polynomial-only strategies, add it inside the `if is_poly:` block.

### Step 4 — Update `_decompose` if needed

If your strategy requires a different decomposition (e.g. recognising a new
function form), add a new branch to `_decompose` **before** the `return None`:

```python
# ── N. My new pattern ──────────────────────────────────────────────────
if <condition on expr>:
    inner = <extract g from expr>
    t = Dummy('t')
    outer = sp.Lambda(t, <reconstruct f(t) from expr and inner>)
    is_poly = _is_polynomial(inner, vars_)
    return Decomposition(outer, inner, is_polynomial=is_poly)
```

### Step 5 — Write tests

Add a new test class:

```python
class TestStrategyNMymethod:
    """
    SN fires when: <preconditions>.
    """

    def test_basic_case(self):
        result = multiple_integrate(<expr>, (<var>, <lo>, <hi>))
        assert_eq(result, <expected>)

    def test_bypass_condition(self):
        # Test an integral that looks similar but should NOT use SN
        result = multiple_integrate(<similar_expr>, ...)
        assert_eq(result, <expected>)  # still correct, different strategy

    def test_strategy_directly(self):
        # Unit-test the strategy function directly
        from multiple_integrate import _try_mymethod
        t = Dummy('t')
        result = _try_mymethod(
            Lambda(t, <f(t)>), <g_expr>, [<vars>], [<ranges>], {}
        )
        assert_eq(result, <expected>)
```

---

## Extending `_decompose`

### Worked example: recognising `f(g)` where `g = x·sin(x)`

Suppose you want to handle `exp(x * sin(x))`. This is not caught by any current
branch because `exp` has a single argument `x*sin(x)`, but that argument is not
itself recognisable as a polynomial or simple form. However, it **is** caught by
Branch 2: `len(expr.args) == 1` and the argument `x*sin(x)` depends on `x`.
So no extension is needed — Branch 2 handles it with `g = x*sin(x)`.

### Worked example: product form `g = h₁(x₁) · h₂(x₂)`

To handle multiplicative separability (e.g. `f(sin(x)·exp(y))`), you would:

1. Add a new branch to `_decompose` that detects `expr.is_Mul` with each factor
   depending on a single variable.
2. Write a `_try_separable_product` strategy that uses a log transform:
   $\log(h_1 \cdot h_2) = \log h_1 + \log h_2$, reduces to the additive case, then
   transforms back.

---

## Coding conventions

| Convention | Detail |
|---|---|
| **Imports** | All SymPy names imported at module top; no `from sympy import *` |
| **Dummy variables** | Always `Dummy('descriptive_name')` inside functions; never reuse integration variable names |
| **Type hints** | `sp.Expr \| None` for strategy returns; `list[sp.Symbol]` for variable lists |
| **Exception handling** | Broad `except Exception: return None` in strategies; let `ValueError` propagate from the public API |
| **Simplification** | Call `simplify()` on intermediate results (density, Jacobian) but not on the final integral — let SymPy's `integrate` return the canonical form |
| **Testing** | Every new strategy needs at least: one basic test, one bypass test, one test via the private function directly |
| **Docstrings** | Strategy functions: preconditions, formula, `Returns None if ...` |
| **Section markers** | Use `# ═══════ §N Title ═══════` for top-level sections |

---

## Design principles

### Strategies are independent

Each `_try_*` function must be completely self-contained. It may call shared
utilities (`_real_critical_points`, `_g_range_on_interval`, etc.) but must not
modify any global state or call other strategy functions (except S7 calling S6
for sub-intervals, which is the one designed exception).

### Fail fast, fail silently

Strategies should check their preconditions at the top and return `None` immediately
rather than attempting the computation and failing mid-way. This keeps the strategy
cascade efficient.

### Preserve SymPy's exact arithmetic

Never convert SymPy expressions to floats inside the library. All intermediate
calculations use SymPy exact arithmetic. The only floating-point operations
permitted are in the test suite's `_num_check` helper.

### The layer-cake formula is the master abstraction

Every strategy should be derivable from:

$$\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x} = \int f(y)\,\mu'(y)\,dy$$

The strategies differ only in how they compute $\mu'(y)$. If a new strategy does
not fit this framework, reconsider whether it belongs in this library or should
be a preprocessing step.

---

## Common pitfalls

### Dummy variable collision

Never use a symbol from `vars_` as a dummy inside a strategy function. Always
create fresh dummies with `Dummy('descriptive_name')`.

```python
# Wrong — reuses user's 'y' symbol
y = symbols('y')
result = integrate(f_outer(y), (y, 0, 1))

# Correct
y = Dummy('y_strategy6')
result = integrate(f_outer(y), (y, 0, 1))
```

### Forgetting `has(sp.Integral)` check

Every `integrate` call inside a strategy must be followed by a check:

```python
result = integrate(expr, (var, lo, hi))
if result.has(sp.Integral):
    return None   # SymPy couldn't evaluate it
```

Without this check, an unevaluated `Integral` object will be returned as if
it were a closed-form result.

### Eigenvalue check for non-positive-definite matrices

The quadratic strategies require `A` to be positive definite. SymPy's eigenvalue
computation can be slow or inconclusive for symbolic matrices. The current code
uses `try/except` around the eigenvalue check and proceeds symbolically if it
fails — this means the computation may produce incorrect results for
indefinite matrices. If your strategy involves matrix definiteness, be explicit:

```python
try:
    evs = list(A.eigenvals().keys())
    if any(sp.ask(sp.Q.negative(ev)) for ev in evs):
        return None
except Exception:
    pass  # can't determine — proceed and let SymPy fail gracefully
```
