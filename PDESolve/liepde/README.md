# liepde

`liepde` is a small symbolic toolkit for Lie-symmetry based PDE reduction built on top of SymPy.

The primary solver is:

```python
import liepde as lp
import sympy as sp
x, t = sp.symbols('x t')
u = sp.Function('u')
result = lp.liepde(eq, u, (x, t))
```

The function `liepde(...)` analyzes a scalar PDE, builds the corresponding jet-space representation, searches for polynomial Lie-symmetry generators, and attempts a symmetry-based reduction.

## Features

- **scalar jet-space PDE parsing**
  - builds scalar jet-space representations from SymPy PDEs

- **automatic PDE-order inference**
  - infers PDE order directly from SymPy expressions before jet-space construction

- **solved-form principal derivative selection**
  - identifies a suitable principal derivative and rewrites the PDE into solved form when possible

- **determining-equation construction for Lie point symmetries**
  - builds determining systems for scalar Lie point symmetry analysis

- **polynomial ansatz solving for infinitesimals**
  - solves for infinitesimal generators using polynomial ansätze of configurable degree

- **symbolic search for reducible symmetry combinations**
  - searches for symmetry generators and linear combinations that admit useful reductions

- **Frobenius-chart based reductions**
  - constructs invariant coordinates and reduced equations using Frobenius-style chart methods

- **repeated reduction workflow utilities**
  - supports multi-step reduction workflows for reducing PDEs to lower-dimensional PDEs or ODEs

- **reduced-equation solving**
  - attempts symbolic solution of reduced equations once a reduction has been found

- **lifting reduced solutions back to the original variables**
  - maps similarity or reduced solutions back into the original dependent-variable form when possible

- **default best-available-result API**
  - full PDE solution if available,
  - otherwise a lifted similarity/reduced solution,
  - otherwise a reduced equation

- **configurable result levels**
  - supports solution-only (default), reduction-level, and full-details output modes

- **public PDE analysis and classification utilities**
  - provides lightweight structural analysis for PDEs without requiring a full solve attempt

- **analysis-only mode**
  - can inspect PDE structure and solver setup without running the full reduction/solve pipeline

- **direct fallback solving with SymPy**
  - uses SymPy solvers where appropriate when symmetry reduction alone does not produce a usable final result

- **pattern-based fallback solutions for common PDE families**
  - includes built-in fallback handling for several standard PDEs, including transport equations, heat/diffusion equations, advection-diffusion equations, and wave equations

- **verification helpers and diagnostics**
  - verifies reductions and lifted solutions where possible, and records structured diagnostics and warnings

- **configurable failure reporting**
  - supports silent failure, readable failure messages, or structured status-style reporting (might want to add optional messages for intermediate steps in the future)

- **tests, examples, and benchmark coverage**
  - includes regression tests, example workflows, and a benchmark suite of common PDEs amenable to Lie-symmetry reduction

## Installation

```bash
python -m pip install -e .
```

## Quick start

```python
import sympy as sp
import liepde as lp

x, t = sp.symbols("x t")
u = sp.Function("u")

eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
sol = lp.liepde(eq, u, (x, t))
print(sol)
# Eq(u(x, t), F(-t + x))
```

## Result levels

Use `result_level` to control how much is returned:

```python
# Best lifted solution only (default)
lp.liepde(eq, u, (x, t), result_level="solution")

# Return a reduced equation or reduced solution when a full lift is unavailable
lp.liepde(eq, u, (x, t), result_level="reduction")

# Return the full internal diagnostics object
res = lp.liepde(eq, u, (x, t), result_level="details")
```

## Examples

### Transport equation

```python
eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
lp.liepde(eq, u, (x, t))
```

### Heat equation

```python
eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
lp.liepde(eq, u, (x, t))
# Eq(u(x, t), C1 + C2*erf(x/(2*sqrt(t))))
```

### Wave equation

```python
eq = sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0)
lp.liepde(eq, u, (x, t))
```

### Analyze only

```python
analysis = lp.liepde(eq, u, (x, t), analyze_only=True)
print(analysis.order, analysis.is_linear, analysis.is_homogeneous)
```

### Failure reporting without full details

```python
status = lp.liepde(eq, u, (x, t), failure_mode="status")
```

## Public API

- `liepde`
- `classify_pde`
- `build_equation_object`
- `compute_polynomial_symmetries`
- `search_reductions_from_symmetries`
- `solve_reduced_equation`
- `infer_sympy_pde_order`

## Benchmark

A small benchmark suite covering about twenty common PDEs is included in:

```text
benchmarks/benchmark_common_pdes.py
```

It records whether each example produced a full solution, reduced result, or no result, along with runtime.

## Future development

- **Improve PDE classification and pre-analysis**
  - Detect order, linearity, quasilinearity, coefficient structure, conservation form, autonomous variables, and scaling weights.
  - Use that analysis to guide symmetry search and reduction strategy.

- **Expand symmetry ansatz families**
  - Go beyond low-degree polynomial point symmetries.
  - Add affine, weighted-homogeneous, rational, separable, exponential/logarithmic, and coefficient-aware ansätz.
  - Support user-supplied ansatz templates.

- **Strengthen reduced-equation solving**
  - Add better classification of reduced ODEs/PDEs.
  - Improve solving for autonomous, separable, exact, linear, Riccati/Bernoulli-type, and conservation-form reductions.
  - Better recognize canonical reduced forms before falling back to generic solvers.

- **Improve subalgebra and reduction selection**
  - Strengthen optimal-system/adjoint-equivalence machinery.
  - Reduce redundant searches over symmetry combinations.
  - Use cleaner, more canonical generators for reduction.

- **Add multi-step reduction planning**
  - Support repeated symmetry reduction, especially for reducing PDE → lower-dimensional PDE → ODE.
  - Add planning heuristics for choosing the most promising reduction sequence.

- **Improve symbolic performance**
  - Reduce expensive `simplify(...)` calls in hot loops.
  - Add stronger pruning in symmetry and reduction search.
  - Improve caching and memoization for prolongation, invariants, and chart construction.

- **Strengthen diagnostics and verification**
  - Replace more broad exception catches with structured diagnostics.
  - Improve verification of reductions and lifted solutions against the original PDE.
  - Add clearer reporting for failed symmetry searches and failed reduction attempts.

- **Add conservation-law and potential-symmetry support**
  - Detect conservation laws for suitable PDE classes.
  - Introduce potential variables automatically where appropriate.
  - Search for and reduce using potential symmetries.

- **Add equivalence transformations and normalization**
  - Normalize PDEs by scaling, affine changes of variables, and dependent-variable rescaling.
  - Map variable-coefficient PDEs into cleaner canonical forms before symmetry analysis.

- **Add nonclassical/conditional symmetry methods**
  - Support invariant-surface conditions and nonclassical determining systems.
  - Use conditional symmetries to obtain reductions unavailable from classical point symmetries.

- **Extend beyond classical point symmetries**
  - Add limited support for contact, evolutionary, and generalized symmetries.
  - Build toward broader jet-dependent symmetry representations.

- **Expand test and benchmark coverage**
  - Add more higher-order, variable-coefficient, nonlinear, and multi-variable PDE examples.
  - Add regression tests for failed reductions, ambiguous reduced equations, and verification edge cases.
  - Add benchmark problems to track solver coverage and performance over time.

- **Broaden examples and documentation**
  - Add more worked examples across transport, diffusion, wave, reaction–diffusion, Burgers-type, and variable-coefficient PDEs.
  - Expand the demo notebook with reduction trees, diagnostics, and side-by-side symmetry workflows.
  - Give more of a background on PDEs (history, theory, applications, symbolic+numeric solving methods, et cetera).
