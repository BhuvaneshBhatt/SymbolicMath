
# Developer Guide

## 1. Architecture Overview

The core computation proceeds via recursive traversal of the expression tree.

Main entry:
- branch_cut_jumps(expr, z)

Core engine:
- branch_cut_jumps_pairs(expr, z)

Key recursion function:
- go(e)

Each expression is decomposed and combined via:
- combine_general_capped
- cross_diff_pairs

---

## 2. Data Model

### Pair format
(expr, condition)

- expr: SymPy expression
- condition: Boolean predicate (SymPy)

### Association format
{
  "Plus": [...],
  "Minus": [...],
  "Jump": [...]
}

---

## 3. Core Invariants

- All conditions must be Boolean
- No symbolic conditions should appear in arithmetic expressions
- Outputs must be normalized
- Every branch result should include a neutral element when appropriate

---

## 4. RestrictedExpr Semantics

RestrictedExpr(expr, cond) acts as a domain filter.

Propagation:
- Compute inner branch structure
- Intersect all conditions with cond
- Remove impossible branches

---

## 5. Computation Pipeline

1. Decompose expression
2. Compute Plus / Minus expansions
3. Combine via algebra rules
4. Compute Jump = Plus - Minus
5. Simplify pairs
6. Merge duplicates

---

## 6. Condition Algebra

Conditions include:
- inequalities (z < 1)
- Contains(z, Reals)
- logical combinations (And, Or)

Simplification includes:
- interval normalization
- affine transformation handling
- intersection

---

## 7. Fail-Fast Design

All public functions validate inputs.

Examples:
- post_simplify_pairs requires list of pairs
- invalid mode raises ValueError

This prevents silent logical errors.

---

## 8. Extending the System

To add a new function:

1. Add rule in branch table
2. Define jump behavior
3. Ensure condition correctness
4. Add tests

---

## 9. Numeric Layer

numeric_branch_cut_jumps approximates jumps via sampling.

Limitations:
- heuristic
- depends on resolution
- not guaranteed complete

---

## 10. Known Limitations

- transcendental cut sets not fully simplified
- multivariate cuts not fully supported
- no full CAD / QE engine

---

## 11. Testing Guidelines

Tests should cover:
- simple functions
- compositions
- restricted expressions
- failure cases

---

## 12. Debugging Tips

- inspect raw pairs via mode="pairs"
- check conditions explicitly
- isolate subexpressions
