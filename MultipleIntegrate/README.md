# MultipleIntegrate

**MultipleIntegrate** is a symbolic definite-integration package for evaluation of many multiple integrals.

It is designed for problems where direct antiderivatives are not the best approach, and instead uses a collection of structural strategies such as:

- separability detection
- symmetry shortcuts
- moment-based evaluation
- Gaussian-family recognition
- rational full-line integration
- trigonometric / exponential rewrites
- polynomial-in-one-variable reduction
- special-function simplification
- fallback to exact symbolic integration where appropriate

The package focuses on **definite integrals**, especially multiple integrals, and aims to return **exact symbolic results** whenever possible.

---

## What the package can do

The solver is strongest on integrals that fit one or more structured families.

### Supported capabilities include

- exact evaluation of many **multiple integrals over product regions**
- support for selected **dependent-bound iterated integrals** in the order given
- recognition of several important exact families:
  - constants and zero integrands
  - polynomial box moments
  - selected simplex-like moment cases
  - Gaussian integrals and Gaussian moments
  - rational full-line integrals
  - radial / polar-coordinate-friendly families
  - trigonometric and exponential transform-friendly forms
  - beta/gamma-type integrals
- fast simplifications such as:
  - constant-factor pullout
  - dimension peeling for separable factors
  - early odd/even symmetry elimination on symmetric domains
  - aggressive sum splitting
  - postprocessing into cleaner special-function forms
- lightweight divergence screening for obvious singular and tail-divergent cases
- memoization of normalized subproblems to reduce repeated symbolic work

---

## Installation

Clone the repository and install in editable mode:

```bash
pip install -e .
```

Or install dependencies first if needed:

```bash
pip install sympy pytest
```

---

## Quick start

```python
from sympy import symbols, sin, cos, exp, pi, oo
from multiple_integrate.core import multiple_integrate

x, y = symbols("x y", real=True)

# Basic 1D definite integral
print(multiple_integrate(sin(x), (x, 0, pi)))                # 2

# Product-region multiple integral
print(multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1))) # 1/12

# Gaussian integral
print(multiple_integrate(exp(-x**2), (x, -oo, oo)))          # sqrt(pi)

# Rational full-line integral
print(multiple_integrate(1 / (x**2 + 1), (x, -oo, oo)))      # pi

# Trig sum on a rectangle
print(multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi)))  # -4
```

---

## Mathematical viewpoint

A multiple integral can be viewed either as:

- an integral over a region $begin:math:text$R \\subset \\mathbb\{R\}\^n$end:math:text$, or
- an iterated integral with nested bounds.

This package currently works best when the problem matches a **recognized structured form** rather than attempting fully general geometric region analysis.

In particular, it is well suited to exact evaluation of integrals such as:

- polynomial moments on boxes
- Gaussian moments on full space
- rational kernels over the real line
- some radial integrals
- some transform-friendly oscillatory or exponential integrals

---

## Examples by family

### 1. Constants and zero integrands

```python
from sympy import symbols
from multiple_integrate.core import multiple_integrate

x, y = symbols("x y", real=True)

print(multiple_integrate(1, (x, 0, 1), (y, 0, 2)))   # 2
print(multiple_integrate(0, (x, -1, 3), (y, 2, 5)))  # 0
```

### 2. Basic exact 1D integrals

```python
from sympy import symbols, sin, exp, pi
from multiple_integrate.core import multiple_integrate

x = symbols("x", real=True)

print(multiple_integrate(x**3, (x, 0, 2)))     # 4
print(multiple_integrate(sin(x), (x, 0, pi)))  # 2
print(multiple_integrate(exp(x), (x, 0, 1)))   # E - 1
```

### 3. Singular but convergent integrals

```python
from sympy import symbols, log, sqrt
from multiple_integrate.core import multiple_integrate

x = symbols("x", positive=True)

print(multiple_integrate(x**(-1/2), (x, 0, 1)))      # 2
print(multiple_integrate(log(x), (x, 0, 1)))         # -1
print(multiple_integrate(1 / sqrt(1 - x), (x, 0, 1)))  # 2
```

### 4. Rational full-line integrals

```python
from sympy import symbols, oo
from multiple_integrate.core import multiple_integrate

x = symbols("x", real=True)

print(multiple_integrate(1 / (x**2 + 1), (x, -oo, oo)))  # pi
print(multiple_integrate(1 / (x**4 + 1), (x, -oo, oo)))  # pi/sqrt(2)
```

### 5. Box moments

```python
from sympy import symbols
from multiple_integrate.core import multiple_integrate

x, y = symbols("x y", real=True)

print(multiple_integrate(x**2 * y**3, (x, 0, 1), (y, 0, 1)))    # 1/12
print(multiple_integrate(x**2 * y**2, (x, -1, 1), (y, -1, 1)))  # 4/9
```

### 6. Gaussian moments

```python
from sympy import symbols, exp, oo
from multiple_integrate.core import multiple_integrate

x, y = symbols("x y", real=True)

print(multiple_integrate(exp(-x**2), (x, -oo, oo)))                     # sqrt(pi)
print(multiple_integrate(x**2 * exp(-x**2), (x, -oo, oo)))              # sqrt(pi)/2
print(multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo)))  # pi
```

### 7. Trigonometric / exponential transform-friendly cases

```python
from sympy import symbols, sin, cos, exp, pi, oo
from multiple_integrate.core import multiple_integrate

x, y = symbols("x y", real=True)

print(multiple_integrate(sin(x) * sin(y), (x, 0, pi), (y, 0, pi)))   # 4
print(multiple_integrate(cos(x + y), (x, 0, pi), (y, 0, pi)))        # -4
print(multiple_integrate(exp(-(x + y)), (x, 0, oo), (y, 0, oo)))     # 1
print(multiple_integrate(exp(-x) * cos(x), (x, 0, oo)))              # 1/2
```

### 8. Special-function outputs

```python
from sympy import symbols, sqrt, exp
from multiple_integrate.core import multiple_integrate

x = symbols("x", positive=True)

print(multiple_integrate(1 / (1 + x), (x, 0, 1)))          # log(2)
print(multiple_integrate(1 / (1 + x**2), (x, 0, 1)))       # pi/4
print(multiple_integrate(x**2 * (1 - x)**3, (x, 0, 1)))    # 1/60
print(multiple_integrate(sqrt(x) * exp(-x), (x, 0, oo)))   # sqrt(pi)/2
```

---

## Main strategy classes

Internally, the solver uses a dispatcher with multiple exact strategies and simplification passes. These include:

### Structural simplifications
- constant-integrand fast paths
- constant-factor extraction
- normalization of simple ranges
- sum splitting
- separability detection
- polynomial-in-one-variable reduction

### Symmetry and parity
- odd/even detection on symmetric domains
- zero shortcuts for odd integrands where justified

### Family recognizers
- box moments
- selected simplex-like moments
- Gaussian integrals and moments
- radial/domain-recognized cases
- rational full-line kernels

### Transform-style methods
- trigonometric rewrites
- exponential separability exposure
- special-function post-simplification
- selected hypergeometric/beta/gamma recognition

### Fallback behavior
If no specialized strategy succeeds, the package falls back to exact symbolic integration attempts where possible.

---

## Testing

The project includes a growing test suite covering:

- constants and zero integrands
- exact 1D definite integrals
- box moments
- Gaussian moments
- rational full-line integrals
- singular-but-convergent cases
- divergent cases
- symmetry shortcuts
- caching safety
- transform-friendly trig/exponential families
- supported-family smoke tests

Representative categories include:

- convergent singular integrals such as `∫_0^1 x^(-1/2) dx`
- divergent singular integrals such as `∫_0^1 1/x dx`
- Gaussian sign checks distinguishing `exp(-x^2)` from `exp(x^2)`
- subtle multidimensional kernels such as `1/(x+y)` and `1/(x+y)^2`

Run the tests with:

```bash
pytest -q
```

---

## Current limitations

The package is intentionally **not** a full general-purpose geometric region engine.

### In particular, it does not yet fully support:
- arbitrary reordering of dependent-bound multiple integrals
- generic piecewise geometric decomposition of arbitrary semialgebraic regions
- unrestricted automatic polar/spherical coordinate changes
- full general multivariate residue calculus
- general holonomic / creative-telescoping multiple integration

### Dependent bounds
Some iterated integrals with dependent inner bounds can be computed successfully **in the given order**, especially when they match a supported family or simplify easily. However, dependent-bound regions are **not yet treated as fully supported geometric regions** for general rewriting and order reversal.

So the package should currently be viewed as:

- strong on many exact structured families,
- partial on dependent-bound iterated cases,
- not yet a complete symbolic region calculus system.

---

## Design philosophy

The package aims to be:

- **exact-first**, not numeric-first
- **strategy-driven**, not purely antiderivative-driven
- **practical**, focusing on families that give many exact results for modest implementation complexity
- **incrementally extensible**, so new region recognizers and integrand families can be added cleanly

---

## Suggested future directions

Possible future extensions include:

- structured region classes such as `BoxRegion`, `SimplexRegion`, `DiskRegion`, `BallRegion`, and `IteratedRegion`
- better supported dependent-bound region handling
- limited exact change-of-variables support
- stronger radial-family recognition
- more robust divergence analysis
- piecewise-branch cleanup using assumptions
- higher-dimensional polytope moments
- principal value and regularized integrals

---

## Author

**Bhuvanesh Bhatt**

---

## License

Use the license included in the repository.
