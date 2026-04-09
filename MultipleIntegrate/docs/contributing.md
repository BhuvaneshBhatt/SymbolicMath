# Contributing

This guide explains the current project layout and the main ideas to keep in mind when extending MultipleIntegrate.

---

## Repository layout

```text
src/multiple_integrate/
  __init__.py
  core.py
  regions.py
tests/
notebooks/
  multiple_integration.ipynb
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

## Current architectural picture

The package is no longer just a single decomposition-first engine. The current flow is closer to:

```text
multiple_integrate(expr, *ranges)
│
├─ parse ranges using the SymPy convention
├─ classify the region if possible
├─ try exact structured region methods
├─ try symmetry / separability reductions
├─ try selected coordinate changes
├─ try selected decomposition-based methods
└─ fall back to sympy.integrate
```

Two source files currently hold most of this logic:

- `core.py` — planner, decomposition helpers, fallback logic, selected exact families, coordinate transforms
- `regions.py` — region classes, region recognition, region-specific formulas

---

## Range convention

Use the same public convention everywhere:

- range tuples are in **inner-first iterated order**, matching `sympy.integrate`

When writing tests, docs, examples, or new code paths, do not switch back to outer-first structural order at the public interface.

---

## Main extension areas

### 1. Region recognition

If you add a new standard domain, it usually belongs in `regions.py`.

Typical tasks:

- define or extend a region class
- teach `region_from_ranges(...)` how to recognize it
- add any useful exact methods, such as `constant_volume`, `polynomial_moment`, or `radial_integral`

### 2. Exact structured families

If a class of integrals has a stable closed form, prefer implementing that directly rather than relying on generic SymPy definite integration.

Current examples include:

- simplex / Dirichlet formulas
- selected disk / ball / shell formulas
- selected Gaussian structured formulas

### 3. Coordinate changes

Selected coordinate transforms are handled internally through structured transform objects and helper logic in `core.py`.

When extending this area, focus on transformations that are:

- mathematically safe
- easy to validate
- easy to map to a standard transformed region

### 4. Decomposition-based methods

`Decomposition` and its helpers still matter for monotone, piecewise-monotone, separable, and selected layer-cake style methods.

This is a useful extension area, but it should now be thought of as one family inside the broader solver.

---

## Adding a new capability

A good pattern is:

1. decide whether the idea is primarily about **region structure**, **integrand structure**, or a **coordinate change**
2. add the smallest coherent implementation in `regions.py` or `core.py`
3. add focused tests in `tests/`
4. update docs if the capability changes public expectations

---

## Coding guidance

- prefer exact formulas for recognized families
- avoid unnecessary delegation to raw SymPy for structured definite integrals
- keep public examples in SymPy-style inner-first order
- add comments for genuinely tricky code paths, especially region reversal and coordinate splitting
- keep helper names descriptive and stable

---

## Documentation updates

When the public behavior changes, update at least:

- `docs/index.md`
- `docs/api.md`
- `docs/strategies.md`
- `docs/theory.md`
- `docs/testing.md` if the test layout changed

The notebook and README are maintained separately.
