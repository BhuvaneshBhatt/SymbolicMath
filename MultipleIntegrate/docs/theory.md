# Mathematical Theory

This page derives the mathematical foundations behind every strategy in MultipleIntegrate,
starting from first principles and building up to the co-area formula and its specialisations.

---

## 1. Definite integration

### 1.1 The Riemann integral

For a bounded function $f : [a, b] \to \mathbb{R}$ the **definite integral** is the limit of
Riemann sums over partitions $a = x_0 < x_1 < \cdots < x_n = b$:

$$\int_a^b f(x)\,dx = \lim_{\|\Delta\| \to 0} \sum_{k=1}^n f(x_k^*)\,(x_k - x_{k-1})$$

Geometrically this is the signed area between the graph $y = f(x)$ and the $x$-axis.

### 1.2 The Fundamental Theorem of Calculus

If $F' = f$ on $[a, b]$:

$$\int_a^b f(x)\,dx = F(b) - F(a)$$

This is the bridge between antiderivatives (indefinite integrals) and definite integrals.

### 1.3 Improper integrals

When the domain or integrand is unbounded, the integral is a limit:

$$\int_a^\infty f(x)\,dx = \lim_{R \to \infty} \int_a^R f(x)\,dx$$

**Convergence** requires the limit to be finite. The **Gaussian integral** is the
canonical example of an improper integral with a closed form:

$$\int_{-\infty}^\infty e^{-x^2}\,dx = \sqrt{\pi}$$

This is proved by squaring and converting to polar coordinates — a trick that
generalises directly to Strategy 2.

---

## 2. Multiple integrals

### 2.1 Definition

Let $\Omega \subseteq \mathbb{R}^n$ be a measurable set and $f : \Omega \to \mathbb{R}$
integrable. Partition $\Omega$ into sub-regions $\{\Omega_k\}$ with volumes $|\Omega_k|$
and pick sample points $\mathbf{x}_k^* \in \Omega_k$. The **$n$-fold multiple integral** is:

$$\int_\Omega f(\mathbf{x})\,d\mathbf{x} = \lim_{\|\text{partition}\| \to 0}
\sum_k f(\mathbf{x}_k^*)\,|\Omega_k|$$

This is a single real number independent of the shape of $\Omega$.

### 2.2 Fubini's theorem

For $f \in L^1([a,b] \times [c,d])$:

$$\iint_{[a,b]\times[c,d]} f(x,y)\,dA
= \int_a^b \!\int_c^d f(x,y)\,dy\,dx
= \int_c^d \!\int_a^b f(x,y)\,dx\,dy$$

**Fubini licenses iterated integration as a method of computing multiple integrals.**
The condition $f \in L^1$ (absolute integrability) is essential; without it the
two iterated orders can disagree.

### 2.3 Change of variables

For a diffeomorphism $\mathbf{x} = \Phi(\mathbf{u})$:

$$\int_\Omega f(\mathbf{x})\,d\mathbf{x}
= \int_{\Phi^{-1}(\Omega)} f(\Phi(\mathbf{u}))\,|\det J_\Phi(\mathbf{u})|\,d\mathbf{u}$$

where $J_\Phi$ is the Jacobian matrix of $\Phi$. This underpins Strategy 6
(monotone substitution).

---

## 3. The layer-cake formula

### 3.1 Statement

For a non-negative measurable function $h : \Omega \to [0, \infty)$:

$$\int_\Omega h(\mathbf{x})\,d\mathbf{x} = \int_0^\infty
\lambda\!\left(\{\mathbf{x} \in \Omega : h(\mathbf{x}) > t\}\right) dt$$

where $\lambda$ denotes Lebesgue measure. This says: *the integral of a function
equals the integral of the measures of its superlevel sets.*

**Proof sketch.** Write $h = \int_0^\infty \mathbf{1}[h(\mathbf{x}) > t]\,dt$
(since the integrand is the indicator of the region under the graph), then apply
Tonelli's theorem to exchange the order of integration.

### 3.2 The pushforward measure formulation

Define the **cumulative measure** of $g$ over $\Omega$:

$$\mu(y) = \lambda\!\left(\{\mathbf{x} \in \Omega : g(\mathbf{x}) \le y\}\right)
= \int_\Omega \mathbf{1}[g(\mathbf{x}) \le y]\,d\mathbf{x}
= \int_\Omega \Theta(y - g(\mathbf{x}))\,d\mathbf{x}$$

where $\Theta$ is the Heaviside step function. Then $\mu'(y) = \frac{d\mu}{dy}$
is the **Lebesgue pushforward density** of $g$ on $\Omega$, and:

$$\boxed{
\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x}
= \int_{y_{\min}}^{y_{\max}} f(y)\,\mu'(y)\,dy
}$$

**This identity holds for any measurable $g$** — polynomial, transcendental, or
discontinuous. It reduces an $n$-dimensional integral to a *one-dimensional* integral,
regardless of the dimension of $\Omega$. This is the master formula from which
every strategy in MultipleIntegrate is derived.

**Proof.** By Fubini and the definition of $\mu$:

$$\int_\Omega f(g(\mathbf{x}))\,d\mathbf{x}
= \int f(y) \underbrace{\frac{d}{dy}\int_\Omega \Theta(y - g(\mathbf{x}))\,d\mathbf{x}}_{\mu'(y)} dy
= \int f(y)\,\mu'(y)\,dy$$

The interchange of $d/dy$ and the spatial integral is justified by the dominated
convergence theorem whenever $g$ is locally bounded.

---

## 4. The co-area formula

The co-area formula (Federer, 1959) provides a geometric interpretation of $\mu'(y)$:

$$\int_\Omega |\nabla g(\mathbf{x})|\,d\mathbf{x}
= \int_{-\infty}^\infty \mathcal{H}^{n-1}\!\left(g^{-1}(y)\right) dy$$

where $\mathcal{H}^{n-1}$ is the $(n-1)$-dimensional Hausdorff measure (surface area)
of the level set $\{g = y\}$. More generally, for any integrable $\phi$:

$$\int_\Omega \phi(\mathbf{x})\,|\nabla g(\mathbf{x})|\,d\mathbf{x}
= \int_{-\infty}^\infty \!\int_{g^{-1}(y)} \phi(\mathbf{x})\,d\mathcal{H}^{n-1}(\mathbf{x})\,dy$$

Setting $\phi = 1/|\nabla g|$ and comparing with the pushforward density:

$$\mu'(y) = \int_{g^{-1}(y)} \frac{1}{|\nabla g(\mathbf{x})|}\,d\mathcal{H}^{n-1}(\mathbf{x})$$

For a single-variable $g$, the level set $g^{-1}(y)$ is a finite collection of points
$\{x_k\}$ and $\mathcal{H}^0$ is the counting measure, giving:

$$\mu'(y) = \sum_{\{k\, :\, g(x_k) = y\}} \frac{1}{|g'(x_k)|}$$

This is the formula used in Strategy 6 (monotone substitution) and Strategy 7
(piecewise-monotone).

---

## 5. Strategy derivations

### Strategy 1 — Linear polynomial over $[0,\infty)^n$

**Setting.** $g(\mathbf{x}) = \mathbf{b} \cdot \mathbf{x} + c$ with $b_i > 0$,
domain $\Omega = [0, \infty)^n$.

**Level sets.** The set $\{g(\mathbf{x}) \le y\}$ is the simplex
$\{\mathbf{x} \ge 0 : \mathbf{b} \cdot \mathbf{x} \le y - c\}$.

**Measure.** The volume of the simplex $\{\mathbf{u} \ge 0 : \mathbf{b} \cdot \mathbf{u} \le r\}$
is $r^n / ((\prod_i b_i) \cdot n!)$ for $r \ge 0$. Therefore:

$$\mu(y) = \frac{(y - c)^n}{(\prod_i b_i)\cdot n!}, \qquad y \ge c$$

$$\mu'(y) = \frac{(y - c)^{n-1}}{(\prod_i b_i)\cdot (n-1)!}$$

**Result:**

$$\int_{[0,\infty)^n} f(\mathbf{b}\cdot\mathbf{x}+c)\,d\mathbf{x}
= \frac{1}{\prod_i b_i \cdot (n-1)!}\int_c^\infty (y-c)^{n-1}\,f(y)\,dy$$

### Strategy 2 — Quadratic over $\mathbb{R}^n$

**Setting.** $g(\mathbf{x}) = \mathbf{x}^\top A\mathbf{x} + \mathbf{b}\cdot\mathbf{x} + c$
with $A$ symmetric positive definite, domain $\mathbb{R}^n$.

**Completing the square.** Let $\mathbf{x}_0 = -A^{-1}\mathbf{b}/2$. Then:

$$g(\mathbf{x}) = (\mathbf{x} - \mathbf{x}_0)^\top A (\mathbf{x} - \mathbf{x}_0) + y_{\min},
\qquad y_{\min} = c - \tfrac{1}{4}\mathbf{b}^\top A^{-1} \mathbf{b}$$

**Level sets.** $\{g \le y\}$ is the ellipsoid
$\{(\mathbf{x}-\mathbf{x}_0)^\top A (\mathbf{x}-\mathbf{x}_0) \le y - y_{\min}\}$.

**Volume of ellipsoid.** An ellipsoid $\{\mathbf{u}^\top A \mathbf{u} \le r\}$ has volume:

$$V(r) = \frac{\pi^{n/2}}{\sqrt{\det A}\,\Gamma(n/2 + 1)}\,r^{n/2}$$

Therefore $\mu(y) = V(y - y_{\min})$ and:

$$\mu'(y) = \frac{\pi^{n/2}}{\sqrt{\det A}\,\Gamma(n/2 + 1)} \cdot \frac{n}{2}(y - y_{\min})^{n/2 - 1}$$

**Result:**

$$\int_{\mathbb{R}^n} f(\mathbf{x}^\top A\mathbf{x} + \mathbf{b}\cdot\mathbf{x} + c)\,d\mathbf{x}
= \frac{\pi^{n/2}}{\sqrt{\det A}\,\Gamma(n/2+1)}
\int_{y_{\min}}^\infty \frac{n}{2}(y-y_{\min})^{n/2-1} f(y)\,dy$$

**Verification for $f = e^{-y}$, $A = I$, $n = 2$.** The formula gives
$\pi \int_0^\infty e^{-y}\,dy = \pi$, recovering the 2-D Gaussian.

### Strategy 3 — Quadratic, even function, mixed infinite/half-infinite ranges

**Setting.** Same as Strategy 2 but some dimensions have range $[0, \infty)$.
Requires $f(g(\mathbf{x}))$ to be even in every half-infinite variable $x_i$.

For each half-infinite dimension, the integral over $[0, \infty)$ equals half the
integral over $(-\infty, \infty)$ by symmetry. If $k$ dimensions are half-infinite:

$$\text{result} = \frac{1}{2^k} \times \text{(Strategy 2 result)}$$

### Strategy 4 — General polynomial, Heaviside integral

**Setting.** $g$ is any polynomial; domain is bounded or semi-infinite.

**Algorithm.** Compute $\mu(y)$ by integrating $\Theta(y - g(\mathbf{x}))$
dimension by dimension, delegating each 1-D integral to SymPy's `integrate`.
SymPy can evaluate integrals containing Heaviside in closed form for polynomial
arguments, producing a piecewise-polynomial $\mu(y)$. Differentiate to get
$\mu'(y)$ and integrate $f(y)\mu'(y)$ in the final step.

### Strategy 5 — Separable $g$

**Setting.** $g(\mathbf{x}) = h_1(x_1) + h_2(x_2) + \cdots + h_n(x_n)$
where each $h_i$ depends only on $x_i$.

**Key identity.** The pushforward measure of a sum of independent random variables
is the **convolution** of their individual pushforward measures:

$$\mu'_g(y) = (\nu_1 * \nu_2 * \cdots * \nu_n)(y)$$

where $\nu_i(y) = \frac{d}{dy}\int_{a_i}^{b_i} \Theta(y - h_i(x_i))\,dx_i$
is the Lebesgue pushforward density of $h_i$ on $[a_i, b_i]$.

**Algorithm.**
1. Compute each $\nu_i$ by integrating $\Theta(y - h_i(x))$ over its range.
2. Build the convolution $\nu_1 * \cdots * \nu_n$ iteratively.
3. Integrate $f(y) \cdot (\nu_1 * \cdots * \nu_n)(y)$ over $[g_{\min}, g_{\max}]$.

The convolution integral at step 2 is:

$$(\nu * \rho)(z) = \int \nu(t)\,\rho(z - t)\,dt$$

**Example.** $g = x + y$ on $[0,1]^2$. Each $\nu_i$ is the uniform density on
$[0,1]$, i.e. $\nu_i(t) = 1$ for $t \in [0,1]$. Their convolution is the
triangular distribution on $[0, 2]$:

$$\mu'(y) = \begin{cases} y & 0 \le y \le 1 \\ 2 - y & 1 < y \le 2 \end{cases}$$

### Strategy 6 — Monotone substitution

**Setting.** $g$ depends on a single variable $x_i$, with no critical points
inside the integration interval $[a_i, b_i]$ (so $g$ is strictly monotone).

**Co-area formula in 1-D.** For a monotone $g$, the level set $g^{-1}(y)$ consists
of a single point $x(y) = g^{-1}(y)$, so:

$$\mu'(y) = \frac{1}{|g'(g^{-1}(y))|} = \left|\frac{dx}{dy}\right|$$

**Algorithm.**
1. Invert $g$ analytically: solve $g(x) = y$ for $x$, obtaining $x = g^{-1}(y)$.
2. Compute the Jacobian $|dx/dy|$.
3. Integrate $f(y)\,|dx/dy|$ over $[g(a), g(b)]$ (or $[g(b), g(a)]$ if decreasing).

If $g$ depends on only one of $n$ variables, the other $(n-1)$ dimensions are
integrated out as a volume factor $\prod_{j \ne i}(b_j - a_j)$.

**Example.** $g = e^{-x}$, $x \in [0, \infty)$. Then $g^{-1}(y) = -\log y$,
$|dx/dy| = 1/y$, $g(0) = 1$, $g(\infty) = 0$.

$$\int_0^\infty f(e^{-x})\,dx = \int_0^1 f(y) \cdot \frac{1}{y}\,dy$$

### Strategy 7 — Piecewise-monotone substitution

**Setting.** $g$ depends on a single variable $x_i$ and has $k$ interior critical
points $c_1 < c_2 < \cdots < c_k$ inside $[a_i, b_i]$.

**Algorithm.** Split the interval at critical points:

$$[a_i, b_i] = [a_i, c_1] \cup [c_1, c_2] \cup \cdots \cup [c_k, b_i]$$

On each sub-interval $g$ is monotone; apply Strategy 6 to each piece and sum:

$$\int_{a_i}^{b_i} f(g(x))\,dx = \sum_{j=0}^k \int_{c_j}^{c_{j+1}} f(g(x))\,dx$$

The co-area density is the sum over all branches:

$$\mu'(y) = \sum_{\{k\,:\,g(x_k) = y\}} \frac{1}{|g'(x_k)|}$$

**Example.** $g = \sin(x)$, $x \in [0, \pi]$. Critical point at $x = \pi/2$.
Two branches:

- $[0, \pi/2]$: $\sin$ increasing, $g^{-1}(y) = \arcsin y$, $|dx/dy| = 1/\sqrt{1-y^2}$
- $[\pi/2, \pi]$: $\sin$ decreasing, $g^{-1}(y) = \pi - \arcsin y$, $|dx/dy| = 1/\sqrt{1-y^2}$

$$\mu'(y) = \frac{2}{\sqrt{1-y^2}}, \quad y \in [0, 1]$$

For $f = 1$: $\int_0^1 \frac{2}{\sqrt{1-y^2}}\,dy = 2\arcsin(1) - 2\arcsin(0) = \pi$.
Wait — that is the *length* of the arc. For $\int_0^\pi \sin(x)\,dx = 2$, we integrate
$f(y) = y$ against $\mu'(y)$:

$$\int_0^1 y \cdot \frac{2}{\sqrt{1-y^2}}\,dy = \left[-2\sqrt{1-y^2}\right]_0^1 = 2$$

### Strategy 8 — General non-polynomial, Heaviside layer-cake

Identical algorithm to Strategy 4, but $g$ may be transcendental (e.g. $\sin(x+y)$,
$\log(xy)$). SymPy is asked to integrate $\Theta(y - g(\mathbf{x}))$ dimension by
dimension. This succeeds when SymPy can evaluate each Heaviside integral in closed form;
otherwise the strategy returns `None` and the fallback is used.

### Strategy 9 — Fallback (iterated integration)

Plain iterated SymPy integration in the order the variables were listed:

$$\int_{a_n}^{b_n} \cdots \int_{a_1}^{b_1} f(x_1, \ldots, x_n)\,dx_1 \cdots dx_n$$

Applied when no earlier strategy fires or succeeds. Covers all cases that SymPy
can handle directly, including variable limits of integration, mixed polynomial/transcendental
integrands that don't factorise as $f \circ g$, and product integrands like $x \sin(x)$.

---

## 6. Convergence and divergence

The library does not pre-screen for convergence. Divergent integrals propagate through
the strategies in the same way as convergent ones:

- If SymPy returns `oo` or `-oo` from an internal `integrate` call, the result is
  propagated and returned as-is.
- If SymPy returns an unevaluated `Integral`, the strategy returns `None` and the
  next strategy is tried; eventually the fallback returns the unevaluated `Integral`.

**Integrable singularities** (e.g. $\int_0^1 x^{-1/2}\,dx = 2$) are handled
correctly because SymPy's `integrate` uses the Cauchy definition for improper
integrals.

**Non-integrable singularities** (e.g. $\int_0^1 x^{-1}\,dx$) return `oo`.

---

## 7. Mathematical references

1. **Folland, G. B.** (1999). *Real Analysis: Modern Techniques and Their Applications*
   (2nd ed.). Wiley. — Layer-cake representation: Proposition 6.16.
2. **Federer, H.** (1969). *Geometric Measure Theory*. Springer. — Co-area formula.
3. **Evans, L. C. & Gariepy, R. F.** (2015). *Measure Theory and Fine Properties
   of Functions* (revised ed.). CRC Press. — Co-area formula: Theorem 3.11.
4. **Rudin, W.** (1987). *Real and Complex Analysis* (3rd ed.). McGraw-Hill.
   — Fubini–Tonelli theorem: Chapter 8.
5. **Apostol, T. M.** (1969). *Calculus, Vol. 2*. Wiley. — Multiple integrals.
6. **Risch, R. H.** (1969). The problem of integration in finite terms.
   *Trans. Amer. Math. Soc.*, 139, 167–189. — Algorithm underlying SymPy's `integrate`.
7. **Bronstein, M.** (2005). *Symbolic Integration I* (2nd ed.). Springer.
