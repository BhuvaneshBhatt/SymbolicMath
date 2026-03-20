# MultipleIntegrate

**Symbolic multiple integration for Python**

MultipleIntegrate is a pure-Python library built on [SymPy](https://www.sympy.org)
that evaluates $n$-dimensional integrals of the form

$$\int_\Omega f\left(g(\mathbf{x})\right) d\mathbf{x}$$

in closed symbolic form. Nine specialised strategies are tried in order; the first
one whose preconditions are satisfied returns an exact result. A plain iterated
fallback handles all remaining cases.

The key insight unifying all strategies is the **layer-cake / co-area identity**:

$$\int_\Omega f(g(\mathbf{x})) d\mathbf{x} = \int f(y)\ \mu'(y) dy,
\qquad \mu(y) = \lambda\left(\{\mathbf{x}\in\Omega : g(\mathbf{x})\le y\}\right)$$

This identity holds for *any* measurable $g$ — polynomial, transcendental, or
discontinuous. The nine strategies differ only in how they compute $\mu'(y)$.

---

## Quick start

```python
from sympy import symbols, exp, sin, cos, sqrt, log, pi, oo, Abs
from multiple_integrate import multiple_integrate

x, y, z = symbols("x y z", real=True)

# Polynomial — Strategy 4 (general polynomial Heaviside)
multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 2))
# → 2/3

# 2-D Gaussian — Strategy 2 (quadratic doubly-infinite)
multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
# → pi

# Half-line Gaussian — Strategy 3 (quadratic even/half-infinite)
multiple_integrate(exp(-x**2), (x, 0, oo))
# → sqrt(pi)/2

# Separable non-polynomial — Strategy 5 (separable)
multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))
# → 0

# Monotone substitution — Strategy 6
multiple_integrate(1 / (1 + x**2), (x, 0, 1))
# → pi/4

# Piecewise-monotone — Strategy 7
multiple_integrate(sin(x), (x, 0, pi))
# → 2

# Non-analytic f — Strategy 7 (kink detection)
multiple_integrate(Abs(x), (x, -1, 1))
# → 1

# Triple integral — Strategy 1 (linear)
multiple_integrate(exp(-(x + y + z)), (x, 0, oo), (y, 0, oo), (z, 0, oo))
# → 1

# Variable limits — Strategy 9 (fallback)
multiple_integrate(x + y, (y, 0, 1 - x), (x, 0, 1))
# → 1/3
```

---

## Installation

### From source (recommended while the package is not on PyPI)

```bash
git clone https://github.com/BhuvaneshBhatt/SymbolicMath/MultipleIntegrate.git
cd MultipleIntegrate
pip install -e .
```

### With optional dependencies

```bash
# For development (pytest, ruff, mypy)
pip install -e ".[dev]"

# For the tutorial notebook
pip install -e ".[notebook]"

# For building the documentation
pip install -e ".[docs]"

# Everything
pip install -e ".[all]"
```

### Requirements

| Dependency | Minimum version | Purpose |
|---|---|---|
| Python | 3.10 | `X \| Y` union types in annotations |
| SymPy | 1.12 | Symbolic integration engine |
| pytest | 7.4 | Running the test suite *(dev only)* |

---

## Project layout

```
multiple-integrate/
├── src/
│   └── multiple_integrate/
│       ├── __init__.py          # Public API re-exports
│       └── core.py              # All nine strategies + decomposition
├── tests/
│   └── test_multiple_integrate.py   # 138 tests across 18 classes
├── notebooks/
│   └── multiple_integration.ipynb   # Tutorial notebook
├── docs/
│   ├── index.md                 # Project overview
│   ├── theory.md                # Mathematical foundations
│   ├── strategies.md            # Strategy reference
│   ├── api.md                   # Full API reference
│   ├── decomposition.md         # _decompose() deep dive
│   ├── examples.md              # 60+ worked examples
│   ├── testing.md               # Test suite guide
│   ├── contributing.md          # Developer guide
│   └── changelog.md             # Version history
├── pyproject.toml               # Build + tool configuration
└── README.md
```

---

## Strategies

Nine strategies are tried in order. The first whose preconditions are met returns
the result; the rest are skipped.

| # | Strategy | When it fires |
|---|---|---|
| 1 | **Linear polynomial** | Poly $g = \mathbf{b}\cdot\mathbf{x}+c$, domain $[0,\infty)^n$ |
| 2 | **Quadratic doubly-infinite** | Poly $g = \mathbf{x}^\top A\mathbf{x}+\cdots$, domain $\mathbb{R}^n$, $A\succ 0$ |
| 3 | **Quadratic even/half-infinite** | Same as 2, mixed $\mathbb{R}\,/\,[0,\infty)$ ranges, $f\circ g$ even |
| 4 | **General polynomial** | Any polynomial $g$, any bounded/semi-infinite domain |
| 5 | **Separable** | $g = h_1(x_1)+\cdots+h_n(x_n)$, one variable per term |
| 6 | **Monotone substitution** | Single-variable $g$, no interior critical points |
| 7 | **Piecewise-monotone** | Single-variable $g$, has interior critical points |
| 8 | **General non-polynomial** | Multi-variable non-polynomial $g$, SymPy evaluates $\Theta(y-g)$ |
| 9 | **Fallback** | Anything else — plain iterated `sympy.integrate` |

---

## Mathematical background

### The layer-cake formula

For any measurable $g : \Omega \to \mathbb{R}$ and integrable $f$:

$$\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x}
= \int_{y_{\min}}^{y_{\max}} f(y)\,\mu'(y)\,dy$$

where $\mu(y) = \lambda(\{\mathbf{x}\in\Omega : g(\mathbf{x})\le y\})$ is the
Lebesgue measure of the sublevel set, and $\mu'(y)$ is its density. This is a
consequence of Fubini's theorem and the layer-cake representation
(Folland, *Real Analysis*, Proposition 6.16).

### Co-area formula

The co-area formula (Federer, 1959) provides a geometric view of $\mu'(y)$:

$$\mu'(y) = \int_{g^{-1}(y)} \frac{1}{|\nabla g(\mathbf{x})|}\,d\mathcal{H}^{n-1}(\mathbf{x})$$

For a single-variable monotone $g$ this collapses to $\mu'(y) = 1/|g'(g^{-1}(y))|$,
which is the Jacobian used in Strategies 6 and 7.

### Key formulas

**Strategy 1 — simplex measure:**

$$\int_{[0,\infty)^n} f(\mathbf{b}\cdot\mathbf{x}+c)\,d\mathbf{x}
= \frac{1}{\prod_i b_i\,(n-1)!}\int_c^\infty (y-c)^{n-1}\,f(y)\,dy$$

**Strategy 2 — ellipsoid surface area:**

$$\int_{\mathbb{R}^n} f(\mathbf{x}^\top A\mathbf{x}+\mathbf{b}\cdot\mathbf{x}+c)\,d\mathbf{x}
= \frac{\pi^{n/2}}{\sqrt{\det A}\,\Gamma(n/2+1)}
\int_{y_{\min}}^\infty \tfrac{n}{2}(y-y_{\min})^{n/2-1}\,f(y)\,dy$$

**Strategy 5 — marginal convolution:**

$$\mu'_{h_1+\cdots+h_n}(y) = (\nu_1 * \nu_2 * \cdots * \nu_n)(y)$$

where $\nu_i$ is the Lebesgue pushforward density of $h_i(x_i)$ on $[a_i,b_i]$.

---

## Examples

### 1. Polynomial integrals

```python
# ∫∫∫_{[0,1]³} xyz dV = 1/8
multiple_integrate(x * y * z, (x, 0, 1), (y, 0, 1), (z, 0, 1))   # 1/8

# ∫∫ x²y on triangle {y ≤ 1-x} = 1/60
multiple_integrate(x**2 * y, (y, 0, 1 - x), (x, 0, 1))            # 1/60
```

### 2. Gaussian integrals

```python
# Anisotropic 2-D Gaussian
multiple_integrate(exp(-(2*x**2 + 3*y**2)), (x, -oo, oo), (y, -oo, oo))
# → pi/sqrt(6)

# Quarter-plane
multiple_integrate(exp(-(x**2 + y**2)), (x, 0, oo), (y, 0, oo))
# → pi/4
```

### 3. Separable non-polynomial

```python
# Convolution of marginal densities
multiple_integrate(exp(-(x + y + z)), (x, 0, oo), (y, 0, oo), (z, 0, oo))
# → 1

multiple_integrate((sin(x) + sin(y))**2, (x, 0, 1), (y, 0, 1))
# → sympy closed form
```

### 4. Monotone substitution

```python
multiple_integrate(log(x), (x, 1, exp(1)))   # → 1
multiple_integrate(sqrt(x), (x, 0, 4))        # → 16/3
multiple_integrate(x**Rational(1, 3), (x, 0, 1))  # → 3/4
```

### 5. Piecewise-monotone and non-analytic f

```python
multiple_integrate(Abs(sin(x)), (x, 0, 2 * pi))   # → 4
multiple_integrate(Abs(x - y), (x, 0, 1), (y, 0, 1))  # → 1/3

# Heaviside of a non-polynomial argument
multiple_integrate(Heaviside(sin(x) - Rational(1, 2)), (x, 0, pi))
# → 2*pi/3
```

### 6. Convergent and divergent improper integrals

```python
# Convergent
multiple_integrate(x**(-2), (x, 1, oo))           # → 1
multiple_integrate(log(x), (x, 0, 1))             # → -1  (integrable singularity)

# Divergent — returns oo
multiple_integrate(1 / x, (x, 1, oo))             # → oo
multiple_integrate(exp(x**2), (x, -oo, oo))        # → oo
```

---

## Running the tests

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run all 138 tests
pytest

# Run a specific strategy class
pytest tests/ -k TestStrategy6Monotone -v

# Run only fast tests (exclude long SymPy computations)
pytest tests/ -m "not slow" -v

# Show the 10 slowest tests
pytest tests/ --durations=10
```

The test suite covers all nine strategies plus cross-cutting properties:
convergence/divergence, analytic vs. non-analytic $f$, continuous vs. discontinuous
$g$, monotone vs. non-monotone $g$, and polynomial vs. non-polynomial $g$.

---

## Documentation

The `docs/` directory contains nine Markdown files:

| File | Contents |
|---|---|
| `index.md` | Overview, quick-start, strategy table |
| `theory.md` | Riemann integrals, Fubini, layer-cake, co-area, all strategy derivations |
| `strategies.md` | Preconditions, formula, algorithm, examples for each of S1–S9 |
| `api.md` | Full reference for every public and internal function |
| `decomposition.md` | `_decompose()` deep dive: all five detection branches |
| `examples.md` | 60+ worked examples by integrand type |
| `testing.md` | How to run tests, test class descriptions, coverage map |
| `contributing.md` | How to add a strategy, coding conventions, common pitfalls |
| `changelog.md` | v1.0 → v2.0 migration notes |

To build the HTML documentation with MkDocs:

```bash
pip install -e ".[docs]"
mkdocs serve       # live-preview at http://127.0.0.1:8000
mkdocs build       # static site in site/
```

---

## Contributing

Contributions are welcome. The architecture is designed so that **new strategies
can be added without touching existing ones**:

1. Write a `_try_mystrategy(f_outer, g, vars_, ranges, opts)` function that
   returns a SymPy expression on success or `None` on failure.
2. Insert it into the cascade in `multiple_integrate()` at the appropriate position.
3. Add a test class `TestStrategyNMyStrategy` in `tests/test_multiple_integrate.py`.

See [`docs/contributing.md`](docs/contributing.md) for the full guide including
the internal data-flow diagram, coding conventions, and common pitfalls.

---

## License

GNU General Public License v3.0 or later. See `LICENSE` for the full text.

---

## References

1. Folland, G. B. (1999). *Real Analysis: Modern Techniques and Their Applications*
   (2nd ed.). Wiley. — Layer-cake representation: Proposition 6.16.
2. Federer, H. (1969). *Geometric Measure Theory*. Springer. — Co-area formula.
3. Evans, L. C. & Gariepy, R. F. (2015). *Measure Theory and Fine Properties of
   Functions* (revised ed.). CRC Press. — Co-area formula: Theorem 3.11.
4. Rudin, W. (1987). *Real and Complex Analysis* (3rd ed.). McGraw-Hill.
   — Fubini–Tonelli: Chapter 8.
5. Risch, R. H. (1969). The problem of integration in finite terms.
   *Trans. Amer. Math. Soc.*, 139, 167–189.
6. SymPy Development Team (2023). *SymPy: Python library for symbolic mathematics.*
   https://www.sympy.org
