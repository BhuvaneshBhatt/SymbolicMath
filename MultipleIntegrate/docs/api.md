# API Reference

This page documents the public API and the main internal data structures used by the current `src/multiple_integrate/` package.

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

Symbolically evaluate a definite or multiple integral.

**Parameters**

| Name | Type | Description |
|---|---|---|
| `f` | `sympy.Expr` | Integrand as a SymPy expression. |
| `*ranges` | `tuple(symbol, lower, upper)` | Integration ranges in **inner-first iterated order**, matching `sympy.integrate`. |
| `assumptions` | `dict`, optional | Extra assumptions forwarded to selected SymPy calls. |
| `generate_conditions` | `bool` | Request conditional output from SymPy in supported fallback calls. |
| `principal_value` | `bool` | Request Cauchy principal value in supported fallback calls. |

**Returns**

A `sympy.Expr`. On some unsupported inputs the result may remain an unevaluated `sympy.Integral`.

**Notes**

- The first range tuple is the **innermost** integral.
- The solver may use region-aware exact formulas, coordinate changes, symmetry reductions, decomposition-based methods, or raw SymPy fallback.
- The package performs some structured-path safety checks, but it does **not** attempt complete general convergence analysis for arbitrary integrals.

**Example**

```python
from sympy import symbols, sqrt
from multiple_integrate import multiple_integrate

x, y = symbols('x y', real=True)

multiple_integrate(
    x**2 * y**2 / sqrt(1 - x**2 - y**2),
    (y, -sqrt(1 - x**2), sqrt(1 - x**2)),
    (x, -1, 1),
)
```

---

### `region_from_ranges`

```python
def region_from_ranges(ranges) -> Region
```

Parse a list of range tuples and try to classify the domain as a structured region.

The input uses the same public convention as `multiple_integrate`: **inner-first iterated order**.

Typical return values include `BoxRegion`, `GraphRegion`, `SimplexRegion`, `DiskRegion`, `BallRegion`, `EllipsoidRegion`, `AnnulusRegion`, `SphericalShellRegion`, `AffineSimplexRegion`, or `IteratedRegion` when no more structured type is recognized.

---

## Region classes

All region objects inherit from `Region` in `multiple_integrate.regions`.

### `Region`

Abstract base class for structured integration domains.

Important methods include:

- `variables`
- `normalized_ranges()`
- `constant_volume()`
- `polynomial_moment(expr)`
- `radial_integral(expr)`
- `is_reflection_invariant(var)`
- `symmetric_range(var)`

Most subclasses implement only the methods that make sense for that region type.

### Main concrete region types

- `BoxRegion`
- `GraphRegion`
- `IteratedRegion`
- `SimplexRegion`
- `AffineSimplexRegion`
- `DiskRegion`
- `AnnulusRegion`
- `BallRegion`
- `SphericalShellRegion`
- `EllipsoidRegion`
- `UnionRegion`

These live in `src/multiple_integrate/regions.py`.

---

## Decomposition support

### `Decomposition`

```python
class Decomposition:
    f_outer: Callable
    g_inner: sympy.Expr
    is_polynomial: bool
```

A lightweight container describing an integrand decomposition of the form

\[
F(x_1,\dots,x_n) = f(g(x_1,\dots,x_n)).
\]

This still matters internally for several non-region-specific heuristics, but it is no longer the only organizing idea in the solver.

---

## Coordinate transforms

### `CoordinateTransform`

```python
@dataclass(frozen=True)
class CoordinateTransform:
    source_vars: tuple[sp.Symbol, ...]
    target_vars: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    target_ranges: tuple[tuple, ...]
```

Internal container used by structured change-of-variables paths.

It represents a map from target variables into the original integration variables, together with the Jacobian factor and the transformed target ranges.

This is currently used for selected coordinate changes such as:

- polar coordinates
- spherical coordinates
- selected affine normalizations
- selected quadratic Gaussian reductions

---

## Internal modules

### `multiple_integrate.core`

Contains:

- `multiple_integrate`
- `Decomposition`
- `CoordinateTransform`
- decomposition helpers
- planner / fallback logic
- selected exact family solvers

### `multiple_integrate.regions`

Contains:

- region classes
- range parsing and region classification
- region-specific formulas such as moments and selected radial / transformed integrals

---

## About fallback

The public convention is always SymPy-style inner-first ordering.

Internally, the solver may normalize or reinterpret ranges for region matching, but that is an implementation detail. From the API point of view, users should write ranges exactly as they would for `sympy.integrate`.
