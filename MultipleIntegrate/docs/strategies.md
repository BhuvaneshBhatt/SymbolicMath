# Strategies

MultipleIntegrate no longer fits neatly into a single linear story of "nine strategies". The current solver is better described as a **layered planner** with several families of methods:

1. region recognition
2. exact structured formulas
3. symmetry and separability heuristics
4. coordinate changes
5. generic SymPy fallback

This page summarizes the main strategy families and when they are useful.

---

## 1. Region recognition

Before attempting expensive symbolic integration, the solver tries to classify the integration domain from the given iterated bounds.

Recognized families include:

- `BoxRegion`
- `GraphRegion`
- `SimplexRegion`
- `AffineSimplexRegion`
- `DiskRegion`
- `AnnulusRegion`
- `BallRegion`
- `SphericalShellRegion`
- `EllipsoidRegion`
- `UnionRegion` in selected internal constructions

If no stronger structure is found, the domain is represented as `IteratedRegion`.

This stage matters because many exact formulas depend more on the geometry of the region than on the superficial syntax of the bounds.

---

## 2. Exact structured formulas

### Boxes and inactive dimensions

On product regions, the solver can often factor constants, strip inactive finite dimensions, and reduce the problem to lower-dimensional exact integration.

### Simplices and affine simplices

This is one of the most important exact families in the current package.

The solver recognizes many integrands of Dirichlet type,

\[
x_1^{a_1-1}\cdots x_n^{a_n-1}(1-x_1-\cdots-x_n)^{a_{n+1}-1},
\]

including fractional exponents when the convergence conditions are sufficiently clear. In these cases it returns the Gamma-ratio formula directly instead of delegating to SymPy.

Polynomial sums of such terms can also be handled after expansion.

### Disks, balls, annuli, shells, ellipsoids

For standard geometric regions, region-specific methods can compute selected moments and selected transformed integrals exactly.

---

## 3. Symmetry and separability

The solver also uses structure in the integrand itself.

Important examples:

- reflection symmetry leading to vanishing odd contributions
- product separability on boxes and some transformed regions
- detection of inactive variables
- selected decomposition of an integrand into `f(g(...))`

These reductions often simplify a problem enough that later strategies become straightforward.

---

## 4. Coordinate changes

A major recent addition is a structured coordinate-change layer.

### Polar coordinates

Used on disks and annuli when the transformed integrand becomes manageable. The package is no longer limited to purely radial integrands; some angular-factorizable cases are also supported.

### Spherical coordinates

Used on balls and spherical shells for selected radial or separable angular/radial families.

### Affine normalizations

Used for selected ellipsoidal and affine-simplex situations.

### Quadratic Gaussian reductions

Used for selected full-space Gaussian integrals where a structured change of variables simplifies the quadratic form.

---

## 5. Decomposition-based legacy heuristics

The package still contains decomposition-driven logic that tries to write an integrand as

\[
F(x_1,\dots,x_n) = f(g(x_1,\dots,x_n)).
\]

That logic remains useful for several classes of non-region-specific problems, especially monotone, piecewise-monotone, separable, and selected layer-cake reductions.

It is best thought of as one family inside the larger planner rather than the whole architecture.

---

## 6. Fallback

When no structured method succeeds, the solver falls back to nested `sympy.integrate`.

This fallback uses the same public convention as the rest of the package: the ranges are interpreted in **inner-first iterated order**.

Fallback is important, but it is intentionally last because:

- it can be much slower than exact structured methods
- it may introduce branch-sensitive hypergeometric forms
- it can return unevaluated integrals

---

## Practical interpretation

A useful mental model is:

- if the **region** is standard, the solver first tries to exploit that
- if the **integrand** has special structure, the solver tries to exploit that
- if a **coordinate change** makes the problem standard, the solver tries that
- otherwise it falls back to SymPy

That is why the package often succeeds on integrals that look complicated but have strong hidden structure.
