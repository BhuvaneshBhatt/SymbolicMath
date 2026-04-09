# MultipleIntegrate

**Symbolic multiple integration for Python**

MultipleIntegrate is a pure-Python library built on [SymPy](https://www.sympy.org) for **structured exact definite and multiple integration**. It combines:

- region recognition for boxes, graphs, simplices, disks, balls, ellipsoids, annuli, and shells
- direct exact formulas for selected families, especially **Dirichlet/simplex** integrals
- coordinate-change methods such as **polar**, **spherical**, and selected **affine** transformations
- symmetry and separability heuristics
- controlled fallback to `sympy.integrate`

The package is strongest when an integral has recognizable geometric or algebraic structure.

---

## Quick start

```python
from sympy import symbols, exp, cos, pi, oo, sqrt
from multiple_integrate import multiple_integrate

x, y, z = symbols('x y z', real=True)

# Box / product region
multiple_integrate(x**2 * y, (y, 0, 2), (x, 0, 1))

# Triangle / simplex
multiple_integrate(1, (y, 0, 1 - x), (x, 0, 1))

# Fractional-power simplex / Dirichlet family
multiple_integrate(
    x**sp.Rational(1, 2) * y**sp.Rational(1, 3) * (1 - x - y)**sp.Rational(1, 4),
    (y, 0, 1 - x),
    (x, 0, 1),
)

# Full-space Gaussian
multiple_integrate(exp(-(x**2 + y**2)), (y, -oo, oo), (x, -oo, oo))

# Disk integral handled by polar coordinates
multiple_integrate(
    x**2 * y**2 / sqrt(1 - x**2 - y**2),
    (y, -sqrt(1 - x**2), sqrt(1 - x**2)),
    (x, -1, 1),
)
```

---

## Range convention

`multiple_integrate` follows the same public convention as `sympy.integrate`:
range tuples are given in **inner-first iterated order**.

So

```python
multiple_integrate(expr, (y, 0, 1 - x), (x, 0, 1))
```

means

\[
\int_0^1 \left(\int_0^{1-x} \text{expr} \, dy\right) dx.
\]

The same convention is used by `region_from_ranges(...)`.

---

## Installation

MultipleIntegrate requires Python >= 3.10 and SymPy >= 1.12.

```bash
pip install -e .
```

For development:

```bash
pip install -e .[dev]
```

---

## Repository layout

| Path | Description |
|---|---|
| `src/multiple_integrate/` | Library source |
| `tests/` | Pytest suite |
| `notebooks/multiple_integration.ipynb` | Tutorial notebook |
| `docs/` | Documentation |

---

## What the library currently does well

### Region-aware exact methods

The package can recognize several standard domains directly from iterated bounds, including:

- boxes / product regions
- graph regions
- simplices and affine simplices
- disks and balls
- ellipsoids
- annuli and spherical shells

That lets it use geometric formulas instead of blindly nesting 1D antiderivatives.

### Exact simplex / Dirichlet evaluation

A dedicated simplex engine handles many integrals of the form

\[
\int_\Delta x_1^{a_1-1}\cdots x_n^{a_n-1}(1-x_1-\cdots-x_n)^{a_{n+1}-1}\,dx,
\]

including fractional exponents when the convergence conditions are clear.

### Coordinate changes

The package includes structured coordinate changes for selected families:

- polar coordinates on disks and annuli
- spherical coordinates on balls and spherical shells
- selected affine normalizations, such as ellipsoids
- selected quadratic Gaussian reductions on full space

### Assumption-aware structured paths

For some exact structured methods, the library now performs basic safety checks before applying a closed form. This is not a complete general convergence engine, but it helps avoid some branch and sign errors on recognized families.

---

## High-level solving flow

The current solver can be thought of as a layered pipeline:

1. Parse ranges using the SymPy convention.
2. Try to recognize a structured region.
3. Try exact structured methods for that region.
4. Try decomposition- and symmetry-based heuristics.
5. Try selected coordinate changes.
6. Fall back to `sympy.integrate` when needed.

This is broader than the older "nine strategy" description: the library still uses decomposition-based heuristics internally, but region recognition and coordinate transforms are now equally important parts of the architecture.

---

## Documentation contents

- [**Theory**](theory.md) — Mathematical background and the main exact families.
- [**Strategies**](strategies.md) — How the solver chooses between region formulas, transforms, decomposition methods, and fallback.
- [**API Reference**](api.md) — Public API and major internal data structures.
- [**Decomposition**](decomposition.md) — Notes on the integrand decomposition logic.
- [**Examples**](examples.md) — Worked examples by family.
- [**Testing**](testing.md) — Test-suite structure and running tests.
- [**Contributing**](contributing.md) — Project layout and development guidance.
- [**Changelog**](changelog.md) — Version history.
