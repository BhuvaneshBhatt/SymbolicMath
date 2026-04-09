# MultipleIntegrate

**MultipleIntegrate** is a symbolic definite-integration package for exact evaluation of many multiple integrals.

It is designed for problems where direct antiderivatives are not the best approach. Instead, it combines structural recognition, simplfication, region-aware dispatch, and exact fallback methods for families such as:

- product-region multiple integrals
- exact Dirichlet/simplex integrals with rational exponents
- polynomial moments on boxes, simplices, disks, and balls
- polar, spherical, and affine coordinate-change reductions for selected regions
- selected simplex-like and level-set / layer-cake reductions
- Gaussian integrals and Gaussian moments
- rational full-line integrals
- trigonometric and exponential transform-friendly integrals
- beta/gamma-type exact integrals
- selected dependent-bound graph regions
- basic convergence and assumptions checks on structured exact paths

---

## Installation

```bash
pip install -e .
```

For development and tests:

```bash
pip install -e .[dev]
```

---

## Quick start

```python
from sympy import symbols, sin, cos, exp, pi, oo
from multiple_integrate import multiple_integrate

x, y = symbols("x y", real=True)

print(multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1)))
# 1/12

print(multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo)))
# pi

print(multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi)))
# -4
```

---


## Range convention

`multiple_integrate` follows **exactly the same range convention as `sympy.integrate`**:

- range tuples are interpreted in **inner-first iterated order**
- the **first** tuple is the innermost integral
- the **last** tuple is the outermost integral

So a triangular integral should be written as

```python
multiple_integrate(1, (y, 0, 1 - x), (x, 0, 1))
```

not with the structural outer-to-inner ordering used, for example, by Mathematica. The same convention is used by
`region_from_ranges(...)` when regions are recognized from dependent bounds.

This matters especially for triangular, disk, and ball examples with dependent bounds: write the tuples in the same order you would pass them to `sympy.integrate`.

---

## Representative multiple-integral examples

### Box moments

```python
import sympy as sp
from sympy import symbols
from multiple_integrate import multiple_integrate

x, y = symbols("x y", real=True)

multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1))
# 1/12
```

### Simplex / triangle moments

```python
from sympy import symbols, Rational
from multiple_integrate import multiple_integrate

x, y = symbols("x y", real=True)

multiple_integrate(1, (y, 0, 1 - x), (x, 0, 1))
# 1/2

multiple_integrate(x * y, (y, 0, 1 - x), (x, 0, 1))
# 1/24

multiple_integrate(
    x**Rational(1, 2) * y**Rational(3, 2) * (1 - x - y)**Rational(1, 2),
    (y, 0, 1 - x),
    (x, 0, 1),
)
# gamma(3/2)*gamma(5/2)*gamma(3/2)/gamma(11/2)
```

### Disk and ball moments

```python
from sympy import symbols, sqrt, exp, oo
from multiple_integrate import multiple_integrate

x, y, z = symbols("x y z", real=True)

multiple_integrate(1, (y, -sqrt(1 - x**2), sqrt(1 - x**2)), (x, -1, 1))
# pi

multiple_integrate(
    x**2 * y**2 / sqrt(1 - x**2 - y**2),
    (y, -sqrt(1 - x**2), sqrt(1 - x**2)),
    (x, -1, 1),
)
# pi/24

multiple_integrate(
    1,
    (z, -sqrt(1 - x**2 - y**2), sqrt(1 - x**2 - y**2)),
    (y, -sqrt(1 - x**2), sqrt(1 - x**2)),
    (x, -1, 1),
)
# 4*pi/3

multiple_integrate(
    (x**2 + y**2 + z**2) * exp(-(x**2 + y**2 + z**2)),
    (z, -oo, oo),
    (y, -oo, oo),
    (x, -oo, oo),
)
# 3*pi**(3/2)/2
```

### Gaussian moments

```python
from sympy import symbols, exp, oo
from multiple_integrate import multiple_integrate

x, y = symbols("x y", real=True)

multiple_integrate(exp(-x**2), (x, -oo, oo))
# sqrt(pi)

multiple_integrate(x**2 * exp(-x**2), (x, -oo, oo))
# sqrt(pi)/2

multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
# pi
```

### Rational full-line integrals

```python
from sympy import symbols, oo
from multiple_integrate import multiple_integrate

x = symbols("x", real=True)

multiple_integrate(1 / (x**2 + 1), (x, -oo, oo))
# pi

multiple_integrate(1 / (x**4 + 1), (x, -oo, oo))
# pi/sqrt(2)
```

### Trigonometric and exponential transform-friendly cases

```python
from sympy import symbols, sin, cos, exp, pi, oo
from multiple_integrate import multiple_integrate

x, y = symbols("x y", real=True)

multiple_integrate(sin(x) * sin(y), (x, 0, pi), (y, 0, pi))
# 4

multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi))
# -4

multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo))
# 1
```

### Level-set / layer-cake example

One of the package strategies rewrites suitable integrals using level sets of an inner function. A simple example is

```python
from sympy import symbols, exp, oo
from multiple_integrate import multiple_integrate

x, y = symbols("x y", nonnegative=True)

multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo))
# 1
```

Here the inner function is `g(x, y) = x + y`. Its level sets in the first quadrant are line segments, so the integral can be reduced to a one-dimensional pushforward density instead of treated only as a plain iterated antiderivative.

---

## Region model

The solver now normalizes input bounds into explicit region objects before applying several structural shortcuts.

Current region support includes:

- `BoxRegion` for product domains with independent bounds
- `IteratedRegion` for general nested bounds
- `SimplexRegion` for standard simplex-style regions
- `AffineSimplexRegion` for affine images of standard simplices
- `GraphRegion` for simple affine graph-bounded 2D regions
- `DiskRegion` for standard centered disks
- `BallRegion` for standard centered balls
- `EllipsoidRegion` for axis-aligned centered ellipsoids
- `AnnulusRegion` for centered annuli
- `SphericalShellRegion` for centered spherical shells
- `UnionRegion` for finite unions of supported regions

This improves:

- symmetry detection
- exact Dirichlet / simplex evaluation
- moment formulas
- dependent-bound handling
- safe order reversal for simple graph regions
- polar /spherical / affine coordinate-change shortcuts
- convergence-aware structured dispatch on several exact families

---

## Main strategy families

The solver uses a dispatcher with exact strategies and simplification passes such as:

- constant and zero fast paths
- separability detection
- region-aware symmetry shortcuts
- polynomial-in-one-variable reduction
- moment-based evaluation on recognized families
- Gaussian-family recognition
- rational full-line recognition
- trigonometric / exponential rewrites
- level-set / layer-cake style reductions for suitable inner functions
- graph-region order reversal for simple affine dependent bounds
- exact symbolic fallback when no specialized strategy applies

---

## Dependent bounds

The package is not limited to product regions. It has **structured** support for some dependent-bound multiple integrals, especially:

- standard simplex / triangle regions
- affine simplex variants
- simple affine graph regions
- standard disk / ball / ellipsoid regions written in nested-bounds form
- explicit annulus, spherical shell, and union regions

However, it is still **not** a full symbolic region engine. In particular, it does **not** yet provide:

- general geometric region rewriting
- automatic order reversal for arbitrary dependent bounds
- a full region algebra comparable to symbolic `Region` objects
- unrestricted automatic polar/spherical coordinate changes
- arbitrary semialgebraic cell decomposition

---

## Testing

Run the test suite with:

```bash
pytest -q
```

The tests cover:

- region parsing and classification
- symmetry behavior
- box / simplex / disk / ball moments
- radial-region shortcuts
- graph-region reversal
- singular-but-convergent cases
- divergence checks
- rational full-line integrals
- representative supported families

---

## Repository layout

```text
MultipleIntegrate/
├── multiple_integrate/
├── tests/
├── docs/
├── notebooks/
├── pyproject.toml
└── README.md
```

---

## Author

**Bhuvanesh Bhatt**

---

## License

GPL-3.0-or-later


---

## Current limitations

Recent additions include exact simplex / Dirichlet formulas and the coordinate-change layer for selected disks, balls, shells, and ellipsoids, butthe package still does **not** attempt completely general geometric rewriting or arbitrary symbolic substitutions. It is best viewed as a recognition-driven exact integrator for structured families.
