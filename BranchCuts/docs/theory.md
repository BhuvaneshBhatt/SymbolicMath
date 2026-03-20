
# Theory: Branch Cuts, Branch Points, and Analytic Continuation

## 1. Multivalued Functions

A complex function is *multivalued* if analytic continuation along different paths yields different values.

Example:
log(z) = log|z| + i(arg z + 2πk)

This arises because arg(z) is only defined modulo 2π.

## 2. Branch Points

A branch point is a point such that analytic continuation around a loop changes the value.

Types:
- Algebraic branch points (e.g. z^(1/2) at z=0)
- Logarithmic branch points (e.g. log z at z=0)

Characterization:
A point z0 is a branch point if monodromy around z0 is nontrivial.

## 3. Branch Cuts

A branch cut is a curve removed to make a multivalued function single-valued.

Not unique — depends on convention.

Common choices:
- log z → (-∞, 0]
- sqrt z → (-∞, 0]

## 4. Riemann Surfaces

Branch cuts are artifacts of representing functions on the complex plane.

True object:
→ multi-sheeted Riemann surface

Example:
sqrt(z) lives on a two-sheeted surface.

## 5. Analytic Continuation

Extending a function beyond its initial domain.

Branch cuts define where continuation is *not continuous*.

## 6. Jump Across a Branch Cut

For z0 on a cut:

Jump(f, z0) = lim ε→0+ [f(z0 + iε) - f(z0 - iε)]

Example:
Jump(log z) = 2πi on negative real axis.

## 7. Composition and Pullback

If f(w) has cut C, then f(g(z)) has cut:

g^{-1}(C)

This produces:
- curves
- multiple components
- transcendental sets

## 8. Nested Branching

Example:
sqrt(log(1 - z))

Two layers:
- log branch
- sqrt branch

Cuts interact nontrivially.

## 9. Real Restrictions

Restricting domain removes cuts:

RestrictedExpr(log(z), z > 0) → no jump

## 10. Computational Implications

- symbolic systems must track domains
- naive simplifications break analyticity
- numerical evaluation can cross cuts

## 11. Semialgebraic Structure

Many branch sets are defined by:

Im(g(z)) = 0
Re(g(z)) ≤ 0

These are semialgebraic sets in (x, y).

## 12. Advanced Topics

- Monodromy groups
- Puiseux series
- Algebraic function fields
- Analytic continuation along paths
