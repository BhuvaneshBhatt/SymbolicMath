# branchcuts

A prototype Python package for symbolic branch-cut computations in SymPy.

It currently includes:
- a symbolic jump registry for many elementary and special functions,
- recursive jump propagation through sums, products, powers, unary compositions, and selected derivatives,
- a numeric seam sampler for representative cuts,
- a post-simplifier aimed at dominant branch-condition patterns such as affine real mappings, interval intersection, duplicate merging, and selected log-side simplifications.

## Clean repo layout

```
branchcuts/
├── src/branchcuts/
│   ├── __init__.py
│   ├── core.py
│   ├── computation.py
│   ├── post_simplify.py
│   ├── classification.py
│   └── inventory.py
├── tests/
├── examples/
├── docs/
├── pyproject.toml
└── README.md
```

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

## Quick start

```python
from sympy import symbols, log
from branchcuts import branch_jump, branch_cut_jumps

z = symbols('z', complex=True)

print(branch_jump(log(z), z))
print(branch_cut_jumps(log(1 - z), z, mode="piecewise"))
```

## Main modules

### `core.py`
Core symbolic objects and explicit branch-jump rules.

### `computation.py`
Recursive branch-cut propagation and numeric seam sampling.

### `post_simplify.py`
Targeted post-simplification for common branch-condition patterns.

### `inventory.py`
Function inventories for SymPy, mpmath, and SciPy special functions.

### `classification.py`
Coverage and classification scaffold for discovered functions.

## Current strengths

The project already handles many practical symbolic computations involving:
- `log`, `sqrt`, constant-exponent powers,
- inverse trigonometric and inverse hyperbolic functions,
- several Bessel, Hankel, hypergeometric, beta/incomplete-beta, elliptic, and logarithmic-integral families,
- composite expressions where cuts come from both outer functions and transformed inner arguments.

## Current limitations

This is still a prototype. It does not yet implement a full condition algebra or quantifier-elimination-based simplification. The post-simplifier is intentionally targeted rather than complete.

## Examples

See `examples/` for simple scripts.
