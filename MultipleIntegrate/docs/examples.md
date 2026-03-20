# Examples

This page collects worked examples grouped by integrand type. Each example
shows the call, the expected result, and which strategy fires.

All examples assume:

```python
from sympy import symbols, exp, sin, cos, sqrt, log, pi, oo, Abs, Heaviside, E, Rational
from multiple_integrate import multiple_integrate

x, y, z = symbols('x y z', real=True)
```

---

## 1. Polynomial integrands

### 1.1 Basic definite integrals

```python
# ∫_0^1 x^n dx = 1/(n+1)
multiple_integrate(x**3, (x, 0, 1))         # 1/4
multiple_integrate(x**5, (x, 0, 1))         # 1/6
multiple_integrate(x**100, (x, 0, 1))       # 1/101
```

### 1.2 Double integrals on rectangles

```python
# ∫∫_{[0,1]²} x²y dx dy = 1/6
multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 1))           # 1/6

# ∫∫_{[0,1]×[0,2]} x²y dx dy = 2/3
multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 2))           # 2/3

# ∫∫_{[0,1]²} (x+y)³ dx dy
multiple_integrate((x+y)**3, (x, 0, 1), (y, 0, 1))           # 3/2

# ∫∫_{[0,1]²} x²y² dx dy = 1/9
multiple_integrate(x**2 * y**2, (x, 0, 1), (y, 0, 1))        # 1/9
```

### 1.3 Triple integrals

```python
# ∫∫∫_{[0,1]³} xyz dV = 1/8
multiple_integrate(x*y*z, (x, 0, 1), (y, 0, 1), (z, 0, 1))  # 1/8

# ∫∫∫_{[0,1]³} (x+y+z) dV = 3/2
multiple_integrate(x+y+z, (x,0,1),(y,0,1),(z,0,1))           # 3/2
```

### 1.4 Non-rectangular domains (variable limits)

```python
# Triangle {0 ≤ y ≤ 1-x, 0 ≤ x ≤ 1}: ∫∫ (x+y) dA = 1/3
multiple_integrate(x + y, (y, 0, 1-x), (x, 0, 1))            # 1/3

# Same triangle: ∫∫ x²y dA = 1/60
multiple_integrate(x**2 * y, (y, 0, 1-x), (x, 0, 1))         # 1/60

# Tetrahedron: ∫∫∫ 1 dV where 0≤z≤1-y, 0≤y≤1-x, 0≤x≤1 = 1/6
multiple_integrate(1, (z,0,1-y-x),(y,0,1-x),(x,0,1))         # 1/6
```

**Strategy:** S4 (general polynomial Heaviside) for bounded domains.

---

## 2. Gaussian / quadratic exponential integrals

### 2.1 Doubly-infinite Gaussians (Strategy 2)

```python
# 1-D
multiple_integrate(exp(-x**2), (x, -oo, oo))                  # √π

# 2-D isotropic
multiple_integrate(exp(-(x**2+y**2)), (x,-oo,oo),(y,-oo,oo)) # π

# 3-D
multiple_integrate(exp(-(x**2+y**2+z**2)),
    (x,-oo,oo),(y,-oo,oo),(z,-oo,oo))                         # π^(3/2)

# Anisotropic: ∫∫ exp(-(2x²+3y²)) = π/√6
multiple_integrate(exp(-(2*x**2 + 3*y**2)),
    (x,-oo,oo),(y,-oo,oo))                                     # π/√6

# With constant shift: ∫∫ exp(-((x-1)²+(y+2)²)) = π
multiple_integrate(exp(-((x-1)**2 + (y+2)**2)),
    (x,-oo,oo),(y,-oo,oo))                                     # π
```

### 2.2 Half-space Gaussians (Strategy 3)

```python
# Half-line
multiple_integrate(exp(-x**2), (x, 0, oo))                    # √π/2

# Quarter-plane
multiple_integrate(exp(-(x**2+y**2)), (x,0,oo),(y,0,oo))     # π/4

# Mixed full/half
multiple_integrate(exp(-(x**2+y**2)), (x,-oo,oo),(y,0,oo))   # π/2
```

### 2.3 Weighted Gaussians

```python
# ∫_0^∞ x·exp(-x²) dx = 1/2  (not even → S6 handles it)
multiple_integrate(x * exp(-x**2), (x, 0, oo))                # 1/2

# ∫∫_ℝ² (x²+y²)·exp(-(x²+y²)) dx dy = π
multiple_integrate((x**2+y**2)*exp(-(x**2+y**2)),
    (x,-oo,oo),(y,-oo,oo))                                     # π
```

---

## 3. Linear exponential integrals (Strategy 1)

```python
# ∫_0^∞ exp(-2x) dx = 1/2
multiple_integrate(exp(-2*x), (x, 0, oo))                     # 1/2

# ∫∫_[0,∞)² exp(-(x+y)) dx dy = 1
multiple_integrate(exp(-(x+y)), (x,0,oo),(y,0,oo))            # 1

# ∫∫_[0,∞)² exp(-(3x+2y)) dx dy = 1/6
multiple_integrate(exp(-(3*x+2*y)), (x,0,oo),(y,0,oo))        # 1/6

# ∫∫∫_[0,∞)³ exp(-(x+y+z)) dV = 1
multiple_integrate(exp(-(x+y+z)), (x,0,oo),(y,0,oo),(z,0,oo)) # 1
```

**Note:** The moment integral $\int_{[0,\infty)^n} (b\cdot x + c)^k f(b\cdot x + c)$
also routes through S1 since the formula works for any $f$.

---

## 4. Separable non-polynomial integrands (Strategy 5)

### 4.1 Additive trig arguments

```python
# ∫∫_[0,π]² cos(x+y) dx dy = 0
multiple_integrate(cos(x+y), (x,0,pi),(y,0,pi))               # 0

# ∫∫_[0,π]² sin(x+y) dx dy = 0
multiple_integrate(sin(x+y), (x,0,pi),(y,0,pi))               # 0

# ∫∫_[0,π/2]² sin(x+y) dx dy
multiple_integrate(sin(x+y), (x,0,pi/2),(y,0,pi/2))           # 2 - √2·... (sympy result)
```

### 4.2 Additive exponential arguments

```python
# ∫∫_[0,∞)² exp(-(x+y)) dx dy = 1
multiple_integrate(exp(-(x+y)), (x,0,oo),(y,0,oo))            # 1

# ∫∫∫_[0,∞)³ exp(-(x+y+z)) dV = 1
multiple_integrate(exp(-(x+y+z)),
    (x,0,oo),(y,0,oo),(z,0,oo))                               # 1
```

### 4.3 Non-polynomial single-variable terms

```python
# ∫∫_[0,1]² (sin(x)+sin(y))² dx dy
multiple_integrate((sin(x)+sin(y))**2, (x,0,1),(y,0,1))       # sympy result

# ∫_1^e∫_1^e log(x+y) dx dy
multiple_integrate(log(x+y), (x,1,E),(y,1,E))                 # sympy result
```

---

## 5. Monotone substitution (Strategy 6)

### 5.1 Exponential

```python
# ∫_0^1 exp(x) dx = e - 1
multiple_integrate(exp(x), (x, 0, 1))                         # E - 1

# ∫_0^∞ exp(-x) dx = 1
multiple_integrate(exp(-x), (x, 0, oo))                       # 1

# ∫_0^1 exp(-2x) dx = (1 - e^{-2})/2
multiple_integrate(exp(-2*x), (x, 0, 1))                      # (1 - exp(-2))/2
```

### 5.2 Logarithm

```python
# ∫_1^e log(x) dx = 1
multiple_integrate(log(x), (x, 1, E))                         # 1

# ∫_0^1 log(x) dx = -1  (integrable singularity)
multiple_integrate(log(x), (x, 0, 1))                         # -1
```

### 5.3 Rational functions

```python
# ∫_0^1 1/(1+x²) dx = π/4
multiple_integrate(1/(1+x**2), (x, 0, 1))                     # π/4

# ∫_0^∞ 1/(1+x**2) dx = π/2
multiple_integrate(1/(1+x**2), (x, 0, oo))                    # π/2

# ∫_0^1 x/(1+x) dx = 1 - log(2)
multiple_integrate(x/(1+x), (x, 0, 1))                        # 1 - log(2)
```

### 5.4 Algebraic functions

```python
# ∫_0^1 √x dx = 2/3
multiple_integrate(sqrt(x), (x, 0, 1))                        # 2/3

# ∫_0^4 √x dx = 16/3
multiple_integrate(sqrt(x), (x, 0, 4))                        # 16/3

# ∫_0^1 x^(1/3) dx = 3/4
multiple_integrate(x**Rational(1,3), (x, 0, 1))               # 3/4
```

### 5.5 With an extra free dimension

```python
# ∫_0^1∫_0^1 exp(-x) dx dy = 1 - 1/e  (y is free, contributes factor 1)
multiple_integrate(exp(-x), (x, 0, 1), (y, 0, 1))             # 1 - 1/E
```

---

## 6. Piecewise-monotone (Strategy 7)

### 6.1 Trigonometric

```python
# ∫_0^π sin(x) dx = 2
multiple_integrate(sin(x), (x, 0, pi))                        # 2

# ∫_0^{2π} sin(x) dx = 0
multiple_integrate(sin(x), (x, 0, 2*pi))                      # 0

# ∫_0^{2π} cos(x) dx = 0
multiple_integrate(cos(x), (x, 0, 2*pi))                      # 0

# ∫_0^π sin²(x) dx = π/2
multiple_integrate(sin(x)**2, (x, 0, pi))                     # π/2
```

### 6.2 Absolute value (kink detection)

```python
# ∫_{-1}^{1} |x| dx = 1
multiple_integrate(Abs(x), (x, -1, 1))                        # 1

# ∫_{-2}^{2} |x| dx = 4
multiple_integrate(Abs(x), (x, -2, 2))                        # 4

# ∫_{-1}^{1} |x³| dx = 1/2
multiple_integrate(Abs(x**3), (x, -1, 1))                     # 1/2
```

### 6.3 With extra dimension

```python
# ∫_0^π∫_0^1 sin(x) dy dx = 2
multiple_integrate(sin(x), (x, 0, pi), (y, 0, 1))             # 2
```

---

## 7. Non-analytic and discontinuous functions

### 7.1 Heaviside step function

```python
# ∫_0^2 Θ(x-1) dx = 1  (step at x=1)
multiple_integrate(Heaviside(x - 1), (x, 0, 2))               # 1

# ∫_0^1 Θ(x-1/2) dx = 1/2
multiple_integrate(Heaviside(x - Rational(1,2)), (x, 0, 1))   # 1/2
```

### 7.2 Product with Heaviside

```python
# ∫_0^π sin(x)·Θ(x-π/2) dx = ∫_{π/2}^π sin(x) dx = 1
multiple_integrate(sin(x)*Heaviside(x - pi/2), (x, 0, pi))    # 1
```

### 7.3 Absolute value of composite functions

```python
# ∫_0^π |sin(x)| dx = 2  (|sin| = sin on [0,π])
multiple_integrate(Abs(sin(x)), (x, 0, pi))                   # 2

# ∫_0^{2π} |sin(x)| dx = 4
multiple_integrate(Abs(sin(x)), (x, 0, 2*pi))                 # 4

# ∫_0^π |cos(x)| dx = 2
multiple_integrate(Abs(cos(x)), (x, 0, pi))                   # 2
```

### 7.4 Two-variable non-analytic

```python
# ∫∫_{[0,1]²} |x-y| dx dy = 1/3
multiple_integrate(Abs(x-y), (x, 0, 1), (y, 0, 1))           # 1/3

# ∫_{-1}^{1} exp(-|x|) dx = 2(1 - e^{-1})
multiple_integrate(exp(-Abs(x)), (x, -1, 1))                  # 2*(1 - exp(-1))
```

---

## 8. Convergent improper integrals

### 8.1 Power functions (p-test: converges iff p < -1 at ∞, or p > -1 at 0)

```python
# ∫_1^∞ x^{-2} dx = 1  (converges, p = -2 < -1)
multiple_integrate(x**(-2), (x, 1, oo))                       # 1

# ∫_1^∞ x^{-3/2} dx = 2
multiple_integrate(x**Rational(-3,2), (x, 1, oo))             # 2

# ∫_0^1 x^{-1/2} dx = 2  (converges, p = -1/2 > -1)
multiple_integrate(x**Rational(-1,2), (x, 0, 1))              # 2
```

### 8.2 Exponential decay

```python
# ∫_0^∞ x^n·exp(-x) dx = n!  (Gamma function)
multiple_integrate(x**2 * exp(-x), (x, 0, oo))               # 2
multiple_integrate(x**3 * exp(-x), (x, 0, oo))               # 6

# ∫_{-∞}^∞ exp(-x²) dx = √π
multiple_integrate(exp(-x**2), (x, -oo, oo))                  # √π
```

### 8.3 Multi-dimensional improper

```python
# ∫∫_{[0,∞)²} exp(-(x+y)) dx dy = 1
multiple_integrate(exp(-(x+y)), (x,0,oo),(y,0,oo))            # 1

# ∫∫_{[1,∞)²} x^{-2}·y^{-2} dx dy = 1
multiple_integrate(x**(-2)*y**(-2), (x,1,oo),(y,1,oo))        # 1
```

---

## 9. Divergent integrals

These return `oo`, `-oo`, or an unevaluated `sympy.Integral`.

```python
# ∫_1^∞ 1/x dx  — boundary case (p = -1)
multiple_integrate(1/x, (x, 1, oo))                           # oo

# ∫_1^∞ x dx  — polynomial growth
multiple_integrate(x, (x, 1, oo))                             # oo

# ∫_0^1 1/x dx  — non-integrable singularity
multiple_integrate(1/x, (x, 0, 1))                            # oo

# ∫_{-∞}^∞ exp(x²) dx  — wrong sign
multiple_integrate(exp(x**2), (x, -oo, oo))                   # oo

# ∫_{-∞}^∞ |x| dx  — polynomial growth at both ends
multiple_integrate(Abs(x), (x, -oo, oo))                      # oo

# ∫∫_{[0,∞)²} (x+y) dx dy  — 2-D divergence
multiple_integrate(x+y, (x,0,oo),(y,0,oo))                    # oo
```

---

## 10. Fallback (plain iterated integration)

These use Strategy 9 because they are not of the form $f(g(\mathbf{x}))$.

```python
# ∫_0^π x·sin(x) dx = π  (integration by parts)
multiple_integrate(x * sin(x), (x, 0, pi))                    # π

# ∫_0^1 x·log(x) dx = -1/4
multiple_integrate(x * log(x), (x, 0, 1))                     # -1/4

# ∫_0^π∫_0^π sin(x)·cos(y) dx dy = 0
multiple_integrate(sin(x)*cos(y), (x,0,pi),(y,0,pi))          # 0

# ∫_0^1∫_0^x exp(y/x) dy dx
multiple_integrate(exp(y/x), (y, 0, x), (x, 0, 1))           # (e-2)/1 (sympy)
```

---

## 11. Physical applications

### Centre of mass of a 2-D region

```python
# Uniform density disk of radius 1: centre of mass is at origin
# x-coordinate: (1/π) ∫∫_{x²+y²≤1} x dA
# Using polar: ∫_0^{2π}∫_0^1 r·cos(θ)·r dr dθ = 0

theta = symbols('theta', real=True)
r_sym = symbols('r', positive=True)
x_cm = multiple_integrate(r_sym**2 * cos(theta),
    (r_sym, 0, 1), (theta, 0, 2*pi))   # 0
```

### Moment of inertia of a solid cube

```python
# I_z = ∫∫∫_{[0,1]³} (x²+y²) dV = 2/3
multiple_integrate(x**2 + y**2, (x,0,1),(y,0,1),(z,0,1))     # 2/3
```

### Volume of a ball

```python
from sympy import symbols
r_sym, theta, phi = symbols('r_sym theta phi', positive=True)
R = symbols('R', positive=True)

# V = ∫_0^{2π}∫_0^π∫_0^R r²sin(θ) dr dθ dφ = 4πR³/3
vol = multiple_integrate(r_sym**2 * sin(theta),
    (r_sym, 0, R), (theta, 0, pi), (phi, 0, 2*pi))            # 4πR³/3
```

### Probability: E[(X+Y)²] for X, Y ~ Uniform[0,1]

```python
# E[(X+Y)²] = Var(X+Y) + E[X+Y]² = 1/6 + 1/6 + 1 = 7/6
multiple_integrate((x+y)**2, (x,0,1),(y,0,1))                 # 7/6
```
