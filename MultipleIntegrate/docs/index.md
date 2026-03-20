# MultipleIntegrate

**Symbolic multiple integration for Python**

MultipleIntegrate is a pure-Python library built on [SymPy](https://www.sympy.org) that evaluates
$n$-dimensional integrals of the form

$$\int_\Omega f\!\left(g(\mathbf{x})\right)\,d\mathbf{x}$$

in closed symbolic form using the **layer-cake / co-area identity**

$$\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x} \;=\; \int f(y)\,\mu'(y)\,dy$$

which holds for *any* measurable $g$. The library implements nine strategies that each 
compute the pushforward density $\mu'(y)$ in a different way suited to the structure of $g$.

---

## Quick start

```python
from sympy import symbols, exp, sin, cos, pi, oo, sqrt
from multiple_integrate import multiple_integrate

x, y, z = symbols('x y z', real=True)

# Polynomial integrand
multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 2))          # → 2/3

# 2-D Gaussian (Strategy 2)
multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))   # → π

# Separable non-polynomial g (Strategy 5)
multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))       # → 0

# Monotone substitution (Strategy 6)
multiple_integrate(1/(1 + x**2), (x, 0, 1))                   # → π/4

# Piecewise-monotone (Strategy 7)
multiple_integrate(sin(x), (x, 0, pi))                         # → 2

# Triple integral
multiple_integrate(exp(-(x + y + z)), (x, 0, oo), (y, 0, oo), (z, 0, oo))  # → 1
```

---

## Installation

MultipleIntegrate requires Python ≥ 3.10 and SymPy ≥ 1.12.

```bash
pip install sympy
# then place multiple_integrate.py in your project directory or on PYTHONPATH
```

No other dependencies are needed.

---

## Project files

| File | Description |
|---|---|
| `multiple_integrate.py` | Library source — all strategies and public API |
| `test_multiple_integrate.py` | pytest suite — 138 tests across 18 classes |
| `multiple_integration.ipynb` | Tutorial notebook with theory and worked examples |
| `docs/` | This documentation |

---

## Strategy overview

Nine strategies are tried in order. The first one whose preconditions are satisfied
returns the result; the rest are skipped.

| # | Name | When it fires | Key formula |
|---|---|---|---|
| 1 | **Linear** | Polynomial $g = \mathbf{b}\cdot\mathbf{x}+c$, domain $[0,\infty)^n$ | Univariate reduction via simplex measure |
| 2 | **Quadratic doubly-infinite** | Polynomial $g = \mathbf{x}^\top A\mathbf{x}+\cdots$, domain $\mathbb{R}^n$ | Ellipsoid surface-area formula |
| 3 | **Quadratic even/half-infinite** | Same as 2, mixed $\mathbb{R}/[0,\infty)$ ranges, $f\circ g$ even | Halving by symmetry |
| 4 | **General polynomial** | Any polynomial $g$, bounded/semi-infinite domain | Heaviside $\mu(y)$ computed by SymPy |
| 5 | **Separable** | $g = h_1(x_1)+\cdots+h_n(x_n)$, one variable per term | Convolution of marginal densities |
| 6 | **Monotone substitution** | Single-variable $g$, no interior critical points | Analytic inversion $x = g^{-1}(y)$ |
| 7 | **Piecewise-monotone** | Single-variable $g$, has critical points | Domain split at critical points, sum branches |
| 8 | **General non-polynomial** | Multi-variable non-polynomial $g$ | Heaviside $\mu(y)$ computed by SymPy |
| 9 | **Fallback** | Anything else | Plain iterated `sympy.integrate` |

Full mathematical derivations for each strategy are in [Theory](theory.md).  
Precise firing conditions and worked examples are in [Strategies](strategies.md).

---

## Documentation contents

- [**Theory**](theory.md) — Mathematical foundations: Riemann integrals, Fubini's theorem,
  the layer-cake formula, the co-area formula, and the derivation of all nine strategies.
- [**Strategies**](strategies.md) — Complete reference for each strategy: preconditions,
  algorithm, formula, examples, and bypass conditions.
- [**API Reference**](api.md) — Full docstring-level reference for `multiple_integrate`,
  `Decomposition`, and all internal functions.
- [**Decomposition**](decomposition.md) — Deep dive into `_decompose`: how the library
  recognises $f(g(\mathbf{x}))$ structure in a SymPy expression tree.
- [**Examples**](examples.md) — Worked examples grouped by integrand type: polynomial,
  trigonometric, exponential, mixed, non-analytic, divergent.
- [**Testing**](testing.md) — Test suite architecture, how to run tests, coverage map,
  and guidance on writing new tests.
- [**Contributing**](contributing.md) — How to add a new strategy, coding conventions,
  and the internal data flow.
- [**Changelog**](changelog.md) — Version history and migration notes.

---

## Design philosophy

> *The polynomial restriction was a limitation of the decomposition logic,
> not of the mathematics.*

The original implementation only handled polynomial $g$, but the layer-cake formula

$$\mu(y) = \int_\Omega \mathbf{1}[g(\mathbf{x}) \le y]\,d\mathbf{x},
\qquad \mu'(y) = \frac{d\mu}{dy}$$

is valid for any measurable $g$. The nine strategies are simply nine different ways 
of computing $\mu'(y)$ efficiently for different classes of $g$.

The code is intentionally written as a cascade of independent, testable functions
(`_try_linear`, `_try_separable`, …) rather than a monolithic routine, so that
new strategies can be added without touching existing ones.
