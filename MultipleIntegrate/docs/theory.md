# Theory

This page summarizes the mathematical ideas behind the current package.

MultipleIntegrate is no longer just a layer-cake engine. The present implementation combines several viewpoints:

- direct evaluation on standard regions
- Dirichlet / Beta / Gamma exact formulas
- coordinate changes
- symmetry reductions
- decomposition-based layer-cake style methods
- fallback iterated integration

---

## 1. Iterated integration and order conventions

The public API follows SymPy's convention: ranges are given in **inner-first iterated order**.

So

```python
multiple_integrate(expr, (y, 0, 1 - x), (x, 0, 1))
```

represents

\[
\int_0^1 \int_0^{1-x} \text{expr} \, dy \, dx.
\]

This is only a user-interface convention. Internally, some region recognizers may normalize the data into a different form for pattern matching, but that is not visible at the API level.

---

## 2. Standard special-function families

A large fraction of exact multiple integrals reduce to Gamma and Beta identities.

### Gamma integral

\[
\Gamma(a) = \int_0^\infty x^{a-1} e^{-x} \, dx,
\qquad \Re(a) > 0.
\]

### Beta integral

\[
B(a,b) = \int_0^1 x^{a-1}(1-x)^{b-1} \, dx
= \frac{\Gamma(a)\Gamma(b)}{\Gamma(a+b)}.
\]

### Dirichlet simplex integral

For suitable exponents,

\[
\int_{\Delta_n}
 x_1^{a_1-1}\cdots x_n^{a_n-1}
 (1-x_1-\cdots-x_n)^{a_{n+1}-1} \, dx
=
\frac{\prod_{j=1}^{n+1}\Gamma(a_j)}{\Gamma(a_1+\cdots+a_{n+1})}.
\]

This formula is one of the core exact engines in the current package.

---

## 3. Region-driven exactness

Many multiple integrals are easier to solve once the geometry is recognized.

### Boxes

Product regions support constant-volume extraction, inactive-dimension stripping, and straightforward factorization when the integrand separates.

### Simplices

Simplex geometry naturally leads to Dirichlet integrals, polynomial moments, and affine normalizations.

### Disks and balls

Circular and spherical regions naturally suggest polar and spherical coordinates, symmetry reductions, and radial or angular-factorizable decompositions.

### Ellipsoids

Affine normalization can reduce selected ellipsoid integrals to standard ball or disk integrals.

---

## 4. Coordinate changes

A change of variables rewrites

\[
\int_R F(x) \, dx
\]

as

\[
\int_{T^{-1}(R)} F(T(u)) \, |\det DT(u)| \, du.
\]

In the current package this idea is implemented in curated, structured ways rather than as unrestricted symbolic inversion.

### Polar coordinates

\[
x = r\cos\theta,
\qquad y = r\sin\theta,
\qquad dx\,dy = r\,dr\,d\theta.
\]

### Spherical coordinates

\[
x = \rho\sin\phi\cos\theta,
\quad y = \rho\sin\phi\sin\theta,
\quad z = \rho\cos\phi,
\quad dV = \rho^2\sin\phi\,d\rho\,d\phi\,d\theta.
\]

These are used when the transformed region and integrand become simpler.

---

## 5. Symmetry

Symmetry can eliminate substantial work before any heavy symbolic manipulation.

Examples include:

- odd integrands over reflection-invariant domains
- equal coordinate moments on isotropic regions
- separation of angular and radial factors after a coordinate change

This is especially important on disks, balls, Gaussian full-space integrals, and some box-type domains.

---

## 6. Layer-cake and decomposition methods

The package still uses the classical identity

\[
\int_\Omega f(g(x))\,dx = \int f(y)\,\mu'(y)\,dy
\]

in selected situations, where

\[
\mu(y) = \int_\Omega \mathbf{1}[g(x) \le y] \, dx.
\]

This remains useful for monotone, piecewise-monotone, separable, and selected non-polynomial cases. But it is now one tool among several, not the sole conceptual center of the solver.

---

## 7. Convergence and assumptions

The package does **not** implement a complete general convergence theory for arbitrary symbolic multiple integrals.

However, the current implementation does perform **basic structured-path checks** in some important situations, such as:

- Dirichlet/simplex exponent conditions
- selected Gaussian positive-definiteness checks
- selected safety checks before applying exact formulas or coordinate changes

So the correct statement is not "no convergence screening at all" but rather:

- structured families receive some safety checks
- arbitrary fallback integrals still rely heavily on SymPy's behavior

---

## 8. Why structure matters

The package's central philosophy is that exact multiple integration is usually easiest when one recognizes hidden structure first:

- geometric structure of the region
- special-function structure of the integrand
- symmetry
- suitable coordinates

That is why the package can often solve integrals that look complicated while still deferring difficult unstructured cases to SymPy.
