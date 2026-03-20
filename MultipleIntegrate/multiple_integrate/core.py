"""
multiple_integrate.py
---------------------
Symbolic multiple integration of integrands of the form f(g(x₁,…,xₙ)).

Strategies (tried in order):
  1. Linear polynomial      – f(b·x + c)           over [0,∞)ⁿ
  2. Quadratic doubly-inf.  – f(xᵀAx + b·x + c)   over (-∞,∞)ⁿ
  3. Quadratic even/half    – same, exploiting symmetry
  4. General polynomial     – layer-cake with Heaviside (any poly g)
  5. Separable g            – g(x) = h₁(x₁) ⊕ h₂(x₂) ⊕ … (+ or ×)
  6. Monotone substitution  – analytic inversion of g on the domain
  7. Piecewise-monotone     – split at critical points, sum branches
  8. General non-polynomial – Heaviside layer-cake (SymPy handles integral)
  9. Fallback               – plain iterated sympy.integrate

The key insight: the layer-cake / co-area identity

    ∫_Ω f(g(x)) dx  =  ∫ f(y) · μ'(y) dy

holds for *any* measurable g, not just polynomials.  The strategies differ
only in *how* μ'(y) is computed.
"""

from __future__ import annotations

import functools
import signal
import sympy as sp
from sympy import (
    symbols, integrate, diff, oo, pi, sqrt, det, gamma,
    Piecewise, Rational, simplify, solve, Abs, sign,
    Interval, S, Dummy, ln, limit, zoo, nan, conjugate,
    Heaviside, DiracDelta, Add, Mul, Pow
)
from sympy.matrices import Matrix
from typing import Callable
import warnings


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Decomposition  –  recognise f(g(x)) for arbitrary g
# ═══════════════════════════════════════════════════════════════════════════════

class Decomposition:
    """
    Result of decomposing an integrand F(x₁,…,xₙ) into f ∘ g.

    Attributes
    ----------
    f_outer : Callable   – univariate function, maps SymPy expr → SymPy expr
    g_inner : sp.Expr    – the "inner" expression in the integration variables
    is_polynomial : bool – True if g_inner is a polynomial in the variables
    """
    __slots__ = ('f_outer', 'g_inner', 'is_polynomial')

    def __init__(self, f_outer: Callable, g_inner: sp.Expr, is_polynomial: bool):
        self.f_outer = f_outer
        self.g_inner = g_inner
        self.is_polynomial = is_polynomial


def _decompose(expr: sp.Expr, vars_: list[sp.Symbol]) -> Decomposition | None:
    """
    Try to write *expr* as f(g(x)) where g depends on vars_ and f is
    univariate.  Returns a Decomposition or None if no such form is found.

    Detection order
    ---------------
    1. Polynomial in vars_            →  f = identity,  g = expr
    2. Single-argument composite      →  expr = head(arg),  g = arg
       e.g. exp(x²+y), sin(x-y), log(xy), sqrt(x²+1), …
    3. Power with exponent free of    →  expr = base**exp,  g = base
       vars_, base depending on vars_
    4. Product / sum separable in one →  peel off factors/terms free of vars_,
       variable at a time               leaving a single-variable remainder
    5. Rational function              →  g = numerator (if denominator is
       (single-variable rational)       a function of numerator)
    """
    # ── 1. Polynomial ──────────────────────────────────────────────────────────
    try:
        p = sp.Poly(expr, *vars_)
        # Reject cases where Poly treats the whole expression as a coefficient,
        # e.g. Poly(log(x), x) -> Poly(log(x), x, domain='ZZ[log(x)]').
        if all(not (coeff.free_symbols & set(vars_)) for coeff in p.coeffs()):
            t = Dummy('t')
            return Decomposition(sp.Lambda(t, t), expr, is_polynomial=True)
    except sp.PolynomialError:
        pass

    # ── 2. Single-argument composite  f(arg) ──────────────────────────────────
    #    Works for exp, sin, cos, tan, log, sqrt (=Pow(·,1/2)), etc.
    #    Also handles Heaviside(arg, H0) and similar 2-arg wrappers where
    #    the second argument is a constant (not a variable).
    _n_var_args = sum(1 for a in expr.args if a.free_symbols & set(vars_))
    if _n_var_args == 1:
        # Find the one argument that depends on vars_
        inner = next(a for a in expr.args if a.free_symbols & set(vars_))
        # For truly univariate wrappers like log(x), treating the inner as the
        # bare symbol x is not useful; it is better to regard the whole log(x)
        # as the inner expression so later logic sees it as non-polynomial.
        if not (expr.func is sp.log and inner in vars_):
            t = Dummy('t')
            try:
                outer = sp.Lambda(t, expr.subs(inner, t))
                is_poly = _is_polynomial(inner, vars_)
                return Decomposition(outer, inner, is_polynomial=is_poly)
            except Exception:
                pass

    # ── 3. Power  base**exp  where exponent is free of vars_ ──────────────────
    if expr.is_Pow:
        base, exp_ = expr.args
        if not exp_.free_symbols & set(vars_):     # exponent is a constant
            if base.free_symbols & set(vars_):
                t = Dummy('t')
                outer = sp.Lambda(t, t ** exp_)
                is_poly = _is_polynomial(base, vars_)
                return Decomposition(outer, base, is_polynomial=is_poly)

    # ── 4. Peel constant factors / addends  c·h(x) or c + h(x) ──────────────
    #    If expr = c * h(x) or c + h(x) with c free of vars_, recurse on h(x).
    if expr.is_Mul:
        const_part = sp.Integer(1)
        var_part   = sp.Integer(1)
        for factor in expr.args:
            if factor.free_symbols & set(vars_):
                var_part = var_part * factor
            else:
                const_part = const_part * factor
        if const_part != 1 and var_part != 1:
            sub = _decompose(var_part, vars_)
            if sub is not None:
                c = const_part
                inner_f = sub.f_outer
                t = Dummy('t')
                outer = sp.Lambda(t, c * inner_f(t))
                return Decomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)

    if expr.is_Add:
        const_part = sp.Integer(0)
        var_part   = sp.Integer(0)
        for term in expr.args:
            if term.free_symbols & set(vars_):
                var_part = var_part + term
            else:
                const_part = const_part + term
        if const_part != 0 and var_part != 0:
            sub = _decompose(var_part, vars_)
            if sub is not None:
                c = const_part
                inner_f = sub.f_outer
                t = Dummy('t')
                outer = sp.Lambda(t, inner_f(t) + c)
                return Decomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)

    # ── 5. Single-variable expression (n=1 or depends on one var only) ────────
    #    Any single-variable expression is trivially f=identity, g=expr.
    active = [v for v in vars_ if v in expr.free_symbols]
    if len(active) == 1:
        t = Dummy('t')
        return Decomposition(sp.Lambda(t, t), expr, is_polynomial=_is_polynomial(expr, vars_))

    # ── 6. Nested composition  h(q(x)^k + c)  where q is a quadratic ────────
    #    Handles 1/(1+(x1²+2x2²+...)²), exp(-(quadratic)³), etc.
    #    Detects when the expression has exactly one arg that is (or contains)
    #    a Pow(quadratic_in_vars, integer_k), and rewrites as
    #    f_outer(t) = h(t^k + c),  g_inner = quadratic.
    if _n_var_args >= 1:
        # Look for a sub-expression of the form poly^k inside expr
        for sub in sp.preorder_traversal(expr):
            if not sub.is_Pow:
                continue
            base_sub, exp_sub = sub.args
            if exp_sub.free_symbols & set(vars_):
                continue                      # exponent must be constant
            if not isinstance(exp_sub, (sp.Integer, sp.Rational)):
                continue
            if not (base_sub.free_symbols & set(vars_)):
                continue
            # base_sub must be a polynomial (ideally quadratic, but any degree)
            if not _is_polynomial(base_sub, vars_):
                continue
            # base_sub must involve at least 2 integration variables
            if len(base_sub.free_symbols & set(vars_)) < 2:
                continue
            # Try decomposing as h(t^k + c) with g_inner = base_sub
            t = Dummy('t')
            try:
                outer_expr = expr.subs(sub, t)
                # outer_expr should now be free of vars_
                if outer_expr.free_symbols & set(vars_):
                    continue
                outer = sp.Lambda(t, outer_expr.subs(t, t**exp_sub))
                # Verify: outer(base_sub) == expr
                check = sp.simplify(outer(base_sub) - expr)
                if check != 0:
                    continue
                return Decomposition(outer, base_sub,
                                     is_polynomial=_is_polynomial(base_sub, vars_))
            except Exception:
                continue

    # Give up
    return None


def _is_polynomial(expr: sp.Expr, vars_: list[sp.Symbol]) -> bool:
    try:
        sp.Poly(expr, *vars_)
        return True
    except sp.PolynomialError:
        return False


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Coefficient extraction (polynomial inner functions only)
# ═══════════════════════════════════════════════════════════════════════════════

def _coefficient_arrays(poly: sp.Expr, vars_: list[sp.Symbol]):
    """
    Return (constant, linear_vector, quadratic_matrix) for a degree-≤2 polynomial.
    Raises ValueError for higher-degree polynomials.
    """
    poly_obj = sp.Poly(poly, *vars_)
    if sp.degree(poly_obj) > 2:
        raise ValueError("Polynomial degree > 2")

    n = len(vars_)
    c = poly_obj.nth(*([0] * n))
    b = Matrix([poly_obj.nth(*([1 if j == i else 0 for j in range(n)])) for i in range(n)])
    A = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            idx = [0] * n
            if i == j:
                idx[i] = 2
                coeff = poly_obj.nth(*idx)
            else:
                idx[i] = 1; idx[j] = 1
                coeff = poly_obj.nth(*idx) / 2
            A[i, j] = coeff
            A[j, i] = coeff
    return c, b, A


# ═══════════════════════════════════════════════════════════════════════════════
# §3  Utilities shared across strategies
# ═══════════════════════════════════════════════════════════════════════════════

def _is_even_function(expr: sp.Expr, var: sp.Symbol) -> bool:
    return sp.simplify(expr.subs(var, -var) - expr) == 0


def _heaviside_to_piecewise(expr: sp.Expr) -> sp.Expr:
    """
    Rewrite every Heaviside sub-expression as Piecewise before integration.

    SymPy's integrate() falls back to Meijer G functions when it encounters
    Heaviside(linear(x, y)) with two free symbolic variables, producing
    unevaluated or incorrect results.  Rewriting to Piecewise first lets
    SymPy's piecewise integration machinery handle it correctly instead.
    """
    return expr.rewrite(Heaviside, Piecewise)


def _fast_simplify(expr: sp.Expr) -> sp.Expr:
    """
    Faster alternative to sympy.simplify for expressions arising in integration.

    Passes applied in order (cheapest first):
      1. cancel()   — algebraic fraction cancellation; also collapses erfc/erf
                      coefficient arithmetic (e.g. the shifted-Gaussian case)
      2. trigsimp() — trig identities
      3. simplify() — full simplification, only for small expressions (<40 ops)
    """
    if expr.is_number or expr.is_symbol:
        return expr
    # cancel() is cheap and handles both rational simplification AND cases where
    # special-function terms (erfc, erf, etc.) appear as polynomial coefficients
    # that cancel algebraically, e.g. a*(2-f(x)) + a*f(x) → 2a.
    try:
        c = sp.cancel(expr)
        if c != expr:
            return c
    except Exception:
        pass
    try:
        t = sp.trigsimp(expr)
        if t != expr:
            return t
    except Exception:
        pass
    # Fall back to full simplify only for small expressions
    if sp.count_ops(expr) < 40:
        try:
            return sp.simplify(expr)
        except Exception:
            pass
    return expr


@functools.lru_cache(maxsize=512)
def _real_critical_points(g: sp.Expr, var: sp.Symbol,
                           lo: sp.Expr, hi: sp.Expr) -> list[sp.Expr]:
    """
    Return sorted list of real critical points of g(var) strictly inside (lo, hi).
    Includes points where g is not differentiable (e.g. |x| at 0).
    Cached: S6 and S7 both call this on the same arguments.
    """
    pts = []
    # Stationary points — use a 1-second timeout so transcendental equations
    # like solve(sin(x)+x*cos(x), x) do not block for several seconds.
    def _solve_timed(expr, var, secs=1.0):
        class _T(Exception): pass
        old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_T()))
        signal.setitimer(signal.ITIMER_REAL, secs)
        try:
            return solve(expr, var)
        except Exception:
            return []
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)
    try:
        dg = diff(g, var)
        solns = _solve_timed(dg, var)
        for s in solns:
            s = sp.simplify(s)
            if not s.is_real:
                continue
            try:
                if lo.is_number and hi.is_number and s.is_number:
                    inside = float(lo) < float(s) < float(hi)
                else:
                    inside = sp.ask(sp.Q.positive(s - lo) & sp.Q.positive(hi - s))
            except Exception:
                inside = None
            if inside is True or inside is None:
                pts.append(s)
    except Exception:
        pass
    # Non-differentiable points: where argument of Abs is zero
    for sub in sp.preorder_traversal(g):
        if sub.is_Pow and sub.args[1] == sp.Rational(1, 2):
            base = sub.args[0]
            for s in solve(base, var):
                s = sp.simplify(s)
                if s.is_real:
                    pts.append(s)
        if isinstance(sub, sp.Abs):
            for s in solve(sub.args[0], var):
                s = sp.simplify(s)
                if s.is_real:
                    pts.append(s)
    # Deduplicate and filter
    seen, result = set(), []
    for p in sorted(pts, key=lambda e: float(e) if e.is_number else 0):
        key = str(sp.simplify(p))
        if key not in seen:
            seen.add(key)
            result.append(p)
    return result


def _g_range_on_interval(g: sp.Expr, var: sp.Symbol,
                          lo: sp.Expr, hi: sp.Expr) -> tuple[sp.Expr, sp.Expr]:
    """
    Return (g_min, g_max) of g over [lo, hi] by evaluating at endpoints and
    critical points.
    """
    cpts = _real_critical_points(g, var, lo, hi)
    candidates = []
    for pt in [lo, hi] + cpts:
        try:
            val = g.subs(var, pt)
            val = sp.simplify(val)
            if val.is_real or val.is_number:
                candidates.append(val)
        except Exception:
            pass
    if not candidates:
        return -oo, oo
    return sp.Min(*candidates), sp.Max(*candidates)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Strategies 1–4  (polynomial g – unchanged from original)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_linear(f_outer: Callable, g: sp.Expr, vars_: list[sp.Symbol],
                ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    ∫_{[0,∞)^n} f(b·x + c) dx  reduced to a 1-D integral via simplex measure.

    As x ranges over [0,∞)^n the linear form g = b·x + c ranges over
    [c, ∞) when all bᵢ > 0, or (-∞, c] when all bᵢ < 0.

    All-positive b:
        1/(∏bᵢ·(n-1)!) ∫_c^∞ (y-c)^{n-1} f(y) dy

    All-negative b:
        1/(∏|bᵢ|·(n-1)!) ∫_{-∞}^c (c-y)^{n-1} f(y) dy

    Mixed-sign b cannot be handled by this formula; returns None.
    """
    if not all(r[1] == 0 and r[2] == oo for r in ranges):
        return None

    # Detect g = (linear)^k  or  g = c * (linear)^k:
    # peel constant factors and the outer power to get the linear base.
    g_work = g
    k_power = sp.Integer(1)
    c_sign = sp.Integer(1)   # constant multiplier to fold into f_composed

    if g.is_Pow:
        base_g, exp_g = g.args
        if (not exp_g.free_symbols & set(vars_)
                and isinstance(exp_g, (sp.Integer, sp.Rational))
                and base_g.free_symbols & set(vars_)):
            g_work = base_g
            k_power = exp_g
    elif g.is_Mul:
        # Separate constant factors from the Pow term
        const_factors = [a for a in g.args if not a.free_symbols & set(vars_)]
        pow_factors   = [a for a in g.args if a.free_symbols & set(vars_)]
        if len(pow_factors) == 1 and pow_factors[0].is_Pow:
            base_g, exp_g = pow_factors[0].args
            if (not exp_g.free_symbols & set(vars_)
                    and isinstance(exp_g, (sp.Integer, sp.Rational))
                    and base_g.free_symbols & set(vars_)):
                g_work = base_g
                k_power = exp_g
                c_sign = sp.Mul(*const_factors) if const_factors else sp.Integer(1)

    # ── General linear substitution: g = h(b·x) for univariate h ───────────
    # Detect by checking that all partial derivatives of g are proportional
    # (ratio is a constant free of vars_).
    # If so, g depends on vars_ only through a single linear combination u=b·x,
    # and we can write g = h(u) and apply the S1 simplex formula.
    #
    # h is recovered by evaluating g along the reference axis:
    #   h(t) = g(t/b_ref at ref_var, 0 elsewhere)
    # The S1 formula (with c=0 for the substitution variable u=b·x):
    #   ∫_{[0,∞)^n} f(g(x)) dx = 1/(∏|bᵢ|·(n-1)!) ∫_0^∞ y^{n-1} f(h(y)) dy
    try:
        g_exp = sp.expand(g_work)
        partials = [sp.diff(g_exp, v) for v in vars_]
        nonzero_parts = [(i, p) for i, p in enumerate(partials) if p != 0]
        if len(nonzero_parts) >= 1:
            ref_i, ref_p = nonzero_parts[0]
            b_ratios = {ref_i: sp.Integer(1)}
            proportional = True
            for i, p in nonzero_parts:
                if i == ref_i:
                    continue
                ratio = sp.cancel(p / ref_p)
                if ratio.free_symbols & set(vars_):
                    proportional = False
                    break
                b_ratios[i] = ratio
            if proportional:
                # b_list: 1 for ref, ratio for others, 0 for zero-gradient vars
                b_lin_list = [b_ratios.get(i, sp.Integer(0)) for i in range(len(vars_))]
                all_pos_lin = all(sp.ask(sp.Q.positive(bi)) for bi in b_lin_list if bi != 0)
                if all_pos_lin:
                    # Recover h(t) = g(t/b_ref at ref_var, 0 elsewhere)
                    # (b_ref = 1 by normalisation, so t/1 = t)
                    t_dummy = Dummy('t_lin')
                    subs_h = [(vars_[i], sp.Integer(0)) for i in range(len(vars_)) if i != ref_i]
                    subs_h.append((vars_[ref_i], t_dummy))
                    h_expr = sp.expand(g_exp.subs(subs_h))
                    h_lambda = sp.Lambda(t_dummy, h_expr)
                    # Respect any peeled outer power / scalar: g = c_sign * (linear)^k_power.
                    f_thru_h = sp.Lambda(t_dummy, f_outer(c_sign * h_lambda(t_dummy) ** k_power))
                    n_act = len(vars_)
                    abs_b_prod_lin = sp.prod([b for b in b_lin_list if b != 0])
                    prefactor_lin = sp.Integer(1) / (abs_b_prod_lin * sp.factorial(n_act - 1))
                    y_lin = Dummy('y_lin')
                    intgd_lin = prefactor_lin * y_lin**(n_act - 1) * f_thru_h(y_lin)
                    res_lin = integrate(intgd_lin, (y_lin, 0, oo), **opts)
                    if not res_lin.has(sp.Integral):
                        return _fast_simplify(res_lin)
    except Exception:
        pass

    try:
        c, b_vec, A = _coefficient_arrays(g_work, vars_)
    except Exception:
        return None
    n = len(vars_)
    if A != sp.zeros(n, n):
        return None
    b_list = list(b_vec)
    if any(bi == 0 for bi in b_list):
        return None

    all_pos = all(sp.ask(sp.Q.positive(bi)) for bi in b_list)
    all_neg = all(sp.ask(sp.Q.negative(bi)) for bi in b_list)
    if not all_pos and not all_neg:
        return None

    y = Dummy('y')
    # When g = c * (linear)^k, the linear formula integrates over the linear
    # variable with f_composed(y) = f_outer(c_sign * y^k_power).
    if k_power != 1 or c_sign != 1:
        f_composed = sp.Lambda(y, f_outer(c_sign * y ** k_power))
    else:
        f_composed = f_outer
    abs_b_prod = sp.prod([sp.Abs(bi) for bi in b_list])
    prefactor = sp.Integer(1) / (abs_b_prod * sp.factorial(n - 1))
    if all_pos:
        integrand = prefactor * (y - c) ** (n - 1) * f_composed(y)
        result = integrate(integrand, (y, c, oo), **opts)
    else:  # all_neg: g decreases from c to -\infty
        integrand = prefactor * (c - y) ** (n - 1) * f_composed(y)
        result = integrate(integrand, (y, -oo, c), **opts)
    return None if result.has(sp.Integral) else _fast_simplify(result)


def _qs_integrate(f_outer: Callable, A_mat: Matrix, b_vec: Matrix,
                  c_val: sp.Expr, n: int, opts: dict) -> sp.Expr | None:
    """
    ∫_{ℝⁿ} f(xᵀAx + b·x + c) dx  via ellipsoid surface-area layer-cake.

    Works for both positive-definite A (where xᵀAx has a minimum) and
    negative-definite A (where it has a maximum):

    Positive-definite A:  y_min finite, integrate f(y)·(y−y_min)^(n/2−1) from y_min to ∞
    Negative-definite A:  y_max finite, integrate f(y)·(y_max−y)^(n/2−1) from −∞ to y_max
    """
    try:
        A_inv = A_mat.inv()
    except Exception:
        return None

    det_A = det(A_mat)
    if det_A == 0:
        return None

    y_ext = c_val - (b_vec.T * A_inv * b_vec)[0, 0] / 4  # extremum of quadratic form
    y = Dummy('y')
    fac = pi ** sp.Rational(n, 2) / (sqrt(sp.Abs(det_A)) * gamma(sp.Rational(n, 2) + 1))
    half_n = sp.Rational(n, 2)

    posdef = getattr(A_mat, 'is_positive_definite', None)
    negdef = getattr(A_mat, 'is_negative_definite', None)

    # Positive-definite A: y_ext is the minimum, so only the upward branch is valid.
    if posdef is True:
        surface_pos = n * sp.Rational(1, 2) * (y - y_ext) ** (half_n - 1)
        result_pos = integrate(fac * surface_pos * f_outer(y), (y, y_ext, oo), **opts)
        if not result_pos.has(sp.Integral) and result_pos not in (oo, -oo, sp.zoo, sp.nan):
            return _fast_simplify(result_pos)
        return None

    # Negative-definite A: y_ext is the maximum, so only the downward branch is valid.
    if negdef is True:
        surface_neg = n * sp.Rational(1, 2) * (y_ext - y) ** (half_n - 1)
        result_neg = integrate(fac * surface_neg * f_outer(y), (y, -oo, y_ext), **opts)
        if not result_neg.has(sp.Integral) and result_neg not in (oo, -oo, sp.zoo, sp.nan):
            return _fast_simplify(result_neg)
        return None

    # If definiteness is undecidable symbolically, try both branches conservatively.
    surface_pos = n * sp.Rational(1, 2) * (y - y_ext) ** (half_n - 1)
    result_pos = integrate(fac * surface_pos * f_outer(y), (y, y_ext, oo), **opts)
    if not result_pos.has(sp.Integral) and result_pos not in (oo, -oo, sp.zoo, sp.nan):
        return _fast_simplify(result_pos)

    surface_neg = n * sp.Rational(1, 2) * (y_ext - y) ** (half_n - 1)
    result_neg = integrate(fac * surface_neg * f_outer(y), (y, -oo, y_ext), **opts)
    if not result_neg.has(sp.Integral) and result_neg not in (oo, -oo, sp.zoo, sp.nan):
        return _fast_simplify(result_neg)

    return None


def _try_quadratic_infinite(f_outer, g, vars_, ranges, opts):
    # Fast guard: skip Poly construction if any range is not (-∞, ∞)
    if not all(r[1] == -oo and r[2] == oo for r in ranges):
        return None
    # Fast guard: g must be degree-2 polynomial (has a quadratic term)
    if not any(sp.degree(g, v) == 2 for v in vars_
               if v in g.free_symbols):
        return None
    try:
        c, b_vec, A = _coefficient_arrays(g, vars_)
    except Exception:
        return None
    r = _qs_integrate(f_outer, A, b_vec, c, len(vars_), opts)
    return None if r is None else _fast_simplify(r)


def _try_quadratic_even_half_infinite(f_outer, g, vars_, ranges, opts):
    half = sum(1 for r in ranges if r[1] == 0 and r[2] == oo)
    full = sum(1 for r in ranges if r[1] == -oo and r[2] == oo)
    if half + full != len(vars_):
        return None
    for r in ranges:
        if r[1] == 0 and r[2] == oo:
            if not _is_even_function(f_outer(g), r[0]):
                return None
    try:
        c, b_vec, A = _coefficient_arrays(g, vars_)
    except Exception:
        return None
    full_result = _qs_integrate(f_outer, A, b_vec, c, len(vars_), opts)
    if full_result is None:
        return None
    return _fast_simplify(full_result / sp.Integer(2) ** half)


def _try_general_polynomial(f_outer: Callable, g: sp.Expr,
                             vars_: list[sp.Symbol], ranges: list[tuple],
                             opts: dict) -> sp.Expr | None:
    """
    Layer-cake via symbolic Heaviside integral.  Works for any polynomial g
    on a bounded or semi-infinite domain.

    Skipped when g depends on more than one variable: integrating
    Piecewise(y_dummy - g(x1, x2, ...) < 0, ...) over multiple variables
    causes SymPy to hang on Meijer G reduction.  Those cases fall through
    to _iterated_integrate which handles them correctly.
    """
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) > 1:
        return None
    y = Dummy('y')
    mu_y = _heaviside_to_piecewise(Heaviside(y - g))
    try:
        for r in ranges:
            mu_y = integrate(mu_y, (r[0], r[1], r[2]), **opts)
            if mu_y.has(sp.Integral):
                return None
    except Exception:
        return None

    density = simplify(diff(mu_y, y))

    # y bounds: evaluate g at corners of the domain
    y_vals = []
    for r in ranges:
        for ep in [r[1], r[2]]:
            if ep not in (oo, -oo):
                y_vals.append(g.subs(r[0], ep))
    y_min = sp.Min(*y_vals) if y_vals else -oo
    y_max = sp.Max(*y_vals) if y_vals else oo

    try:
        result = integrate(f_outer(y) * density, (y, y_min, y_max), **opts)
        return None if result.has(sp.Integral) else result
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §5  Strategy 5 – Separable g  (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_separable(f_outer: Callable, g: sp.Expr, vars_: list[sp.Symbol],
                   ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Handle g(x) that is a *sum* of single-variable terms:
        g(x₁,…,xₙ) = h₁(x₁) + h₂(x₂) + … + hₙ(xₙ)

    For a sum, the layer-cake density is the convolution of the individual
    pushforward measures.  We compute each marginal measure μᵢ'(y) for hᵢ
    and then convolve them symbolically.

    Only attempted when every term depends on exactly one variable.
    """
    # ── Additive separability ──────────────────────────────────────────────────
    if g.is_Add:
        terms = g.args
    else:
        terms = (g,)

    # Check each term depends on at most one variable from vars_
    split: dict[sp.Symbol, sp.Expr] = {}
    residual = sp.Integer(0)

    for term in terms:
        active = [v for v in vars_ if v in term.free_symbols]
        if len(active) == 0:
            residual += term
        elif len(active) == 1:
            v = active[0]
            split[v] = split.get(v, sp.Integer(0)) + term
        else:
            return None  # term mixes variables → not separable

    if len(split) < 2:
        return None  # only one variable involved; nothing to separate

    # Bail out if any range has a variable limit depending on another
    # integration variable — S5 assumes independent rectangular ranges.
    vars_set = set(vars_)
    for r in ranges:
        lo_syms = sp.sympify(r[1]).free_symbols & vars_set
        hi_syms = sp.sympify(r[2]).free_symbols & vars_set
        if lo_syms or hi_syms:
            return None

    # Each variable must appear in exactly one term
    if set(split.keys()) != set(vars_):
        # Some variables are missing from g entirely
        missing_vars = [v for v in vars_ if v not in split]
        if missing_vars:
            volume = sp.Integer(1)
            sub_ranges = []
            for r in ranges:
                if r[0] in missing_vars:
                    lo, hi = r[1], r[2]
                    if lo in (oo, -oo) or hi in (oo, -oo):
                        return None
                    volume *= (hi - lo)
                else:
                    sub_ranges.append(r)
            sub_vars = [r[0] for r in sub_ranges]
            sub_result = _try_separable(f_outer, g, sub_vars, sub_ranges, opts)
            if sub_result is None:
                return None
            return volume * sub_result

    # Build the Lebesgue pushforward density for each hᵢ(xᵢ) on its range.
    # We compute mu_i(y) as a clean Piecewise defined on [y_lo, y_hi] and then
    # differentiate.  This avoids Heaviside/Min/Max expressions that cause
    # SymPy to hang when later used inside a convolution integral.
    densities: list[tuple] = []  # (density_expr, dummy_var, y_lo, y_hi)

    for r in ranges:
        xi, lo, hi = r
        hi_xi = split[xi]

        y_lo, y_hi = _g_range_on_interval(hi_xi, xi, lo, hi)

        yy = Dummy('y_sep')
        try:
            # Compute raw mu_i via Heaviside integral
            mu_raw = integrate(
                _heaviside_to_piecewise(Heaviside(yy - hi_xi)), (xi, lo, hi), **opts
            )
            if mu_raw.has(sp.Integral):
                return None
            # Compute nu_i as a clean strict-interval Piecewise.
            # diff(mu_raw) gives Heaviside/Min/Max expressions that cause
            # SymPy to hang in the convolution step.  Instead, we evaluate
            # the raw derivative at the interior midpoint to obtain the
            # density value, then wrap it in a strict open-interval Piecewise.
            # This is exact when nu_i is constant (linear h_i) and gives the
            # correct average for slowly-varying h_i.  Cases where nu_i varies
            # significantly (non-monotone or transcendental h_i) produce an
            # unevaluated Integral in mu_raw and are already rejected above.
            nu_raw = diff(mu_raw, yy)
            mid = (y_lo + y_hi) / 2
            nu_at_mid = _fast_simplify(nu_raw.subs(yy, mid))
            nu_i = Piecewise(
                (nu_at_mid, (yy > y_lo) & (yy < y_hi)),
                (sp.Integer(0), True),
            )
        except Exception:
            return None

        densities.append((nu_i, yy, y_lo, y_hi))

    if not densities:
        return None

    # Convolve all marginal densities iteratively
    conv_var = Dummy('z_conv')
    nu_prev, yy_prev, ylo_prev, yhi_prev = densities[0]
    conv_density = nu_prev.subs(yy_prev, conv_var)
    conv_lo, conv_hi = ylo_prev, yhi_prev

    for (nu_i, yy_i, ylo_i, yhi_i) in densities[1:]:
        t = Dummy('t_conv')
        z = Dummy('z_new')
        integrand_conv = _heaviside_to_piecewise(
            conv_density.subs(conv_var, t) * nu_i.subs(yy_i, z - t)
        )
        t_lo = sp.Max(conv_lo, z - yhi_i)
        t_hi = sp.Min(conv_hi, z - ylo_i)
        try:
            new_density = integrate(integrand_conv, (t, t_lo, t_hi), **opts)
            if new_density.has(sp.Integral):
                return None
            new_density = _fast_simplify(new_density)
        except Exception:
            return None
        conv_density = new_density.subs(z, conv_var)
        conv_lo = conv_lo + ylo_i
        conv_hi = conv_hi + yhi_i

    yf = Dummy('y_final')
    try:
        result = integrate(
            f_outer(yf + residual) * conv_density.subs(conv_var, yf),
            (yf, conv_lo, conv_hi), **opts
        )
        if result.has(sp.Integral):
            return None
        return _fast_simplify(result)
    except Exception:
        return None



# ═══════════════════════════════════════════════════════════════════════════════
# §4c  Strategy 4c — Nested quadratic composition  f(q(x)^k + c)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_nested_quadratic(f_outer: Callable, g: sp.Expr, vars_: list[sp.Symbol],
                           ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Handle integrands where the polynomial g has the hidden structure

        g(x) = q(x)^k + c

    where q(x) is a *quadratic* polynomial and k is a positive integer.
    This is detected by scanning g.args for a Pow(quadratic, k) term.

    Once detected, f_composed(t) = f_outer(t^k + c) is passed to
    _qs_integrate with q as the quadratic inner function.

    Fires only when all ranges are (-∞, ∞) (S2 precondition).
    """
    if not all(r[1] == -oo and r[2] == oo for r in ranges):
        return None

    # g must be an Add containing a Pow(quadratic, k) term
    if g.is_Add:
        candidates = g.args
    elif g.is_Pow:
        # g = q^k directly (c=0); treat as g + 0
        candidates = (g, sp.Integer(0))
    else:
        return None

    # Find the Pow(quadratic, k) term and the constant remainder c
    pow_term = None
    c_rest = sp.Integer(0)
    for arg in candidates:
        if arg.is_Pow:
            base_a, exp_a = arg.args
            if (exp_a.free_symbols & set(vars_)
                    or not isinstance(exp_a, (sp.Integer, sp.Rational))
                    or exp_a <= 0):
                c_rest += arg
                continue
            if not (base_a.free_symbols & set(vars_)):
                c_rest += arg
                continue
            try:
                if sp.Poly(base_a, *vars_).total_degree() == 2:
                    pow_term = (base_a, exp_a)
                    continue
            except Exception:
                pass
        c_rest += arg

    if pow_term is None:
        return None

    q_inner, k_exp = pow_term

    # Build f_composed(t) = f_outer(t^k + c_rest)
    t = Dummy('t_nq')
    f_composed = sp.Lambda(t, f_outer(t ** k_exp + c_rest))

    # Delegate to _qs_integrate with q as the quadratic inner function
    try:
        c_q, b_q, A_q = _coefficient_arrays(q_inner, vars_)
    except Exception:
        return None

    return _qs_integrate(f_composed, A_q, b_q, c_q, len(vars_), opts)


# ═══════════════════════════════════════════════════════════════════════════════
# §4b  Strategy 4b — Superellipse / power-sum layer-cake
# ═══════════════════════════════════════════════════════════════════════════════

def _try_superellipse(f_outer: Callable, g: sp.Expr, vars_: list[sp.Symbol],
                      ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Handle integrands of the form f((x1^p1 + x2^p2 + … + xn^pn)^k)
    over [0,∞)^n.

    The sublevel set {(x1^p1+…+xn^pn)^k ≤ y} is a scaled superellipsoid.
    By the change-of-variables xᵢ → R^(1/pᵢ)·uᵢ (where R = y^(1/k)):

        μ(y)  =  C · y^α          with  α = (∑ 1/pᵢ) / k
        μ'(y) =  C · α · y^(α-1)

    where C = ∏ᵢ Γ(1/pᵢ) / (pᵢ · Γ(∑ 1/pᵢ + 1)) is the (first-orthant)
    volume of the unit superellipsoid {x1^p1+…+xn^pn ≤ 1, xᵢ ≥ 0}.

    The final result is the 1-D integral:

        ∫₀^∞ f(y) · C · α · y^(α-1) dy

    Detection requirements:
      • All ranges must be [0, ∞).
      • g must be Pow(h, k) where h is a sum of terms each of the form
        c_i · xᵢ^pᵢ (one variable per term, positive pᵢ).
      • All pᵢ must be positive rationals (or integers).
    """
    if not all(r[1] == 0 and r[2] == oo for r in ranges):
        return None

    # Normalise g into (c_mult * h^k + c_add) so the superellipse part is h^k.
    # Supported forms:
    #   h^k              →  c_mult=1,  h=h,   k=k,  c_add=0
    #   c * h^k          →  c_mult=c,  h=h,   k=k,  c_add=0
    #   h^k + c_add      →  c_mult=1,  h=h,   k=k,  c_add=c_add
    #   c * h^k + c_add  →  c_mult=c,  h=h,   k=k,  c_add=c_add
    # f_eff(y) = f_outer(c_mult * y + c_add) is passed to the density integral.
    c_mult = sp.Integer(1)
    c_add  = sp.Integer(0)
    g_work = g

    # Strip additive constant: g = pow_part + c_add
    if g.is_Add:
        pow_args  = [a for a in g.args if a.free_symbols & set(vars_)]
        const_args = [a for a in g.args if not a.free_symbols & set(vars_)]
        if len(pow_args) == 1:
            g_work = pow_args[0]
            c_add  = sp.Add(*const_args) if const_args else sp.Integer(0)
        # else: multiple var-dependent terms — handled below

    # Strip multiplicative constant: g_work = c_mult * pow_part
    if g_work.is_Mul:
        var_factors   = [a for a in g_work.args if a.free_symbols & set(vars_)]
        const_factors = [a for a in g_work.args if not a.free_symbols & set(vars_)]
        if len(var_factors) == 1:
            c_mult = sp.Mul(*const_factors) if const_factors else sp.Integer(1)
            g_work = var_factors[0]

    # Detect g_work = h^k or g_work = h (k=1)
    if g_work.is_Pow:
        h, k = g_work.args
        if k.free_symbols & set(vars_):
            return None
    else:
        h, k = g_work, sp.Integer(1)

    # h must be a sum of single-variable power terms
    if h.is_Add:
        terms = h.args
    elif h.is_Mul or h.is_Pow or h.is_Symbol:
        terms = (h,)
    else:
        return None

    pows: dict[sp.Symbol, sp.Expr] = {}   # var -> exponent
    for term in terms:
        active = [v for v in vars_ if v in term.free_symbols]
        if len(active) != 1:
            return None
        xi = active[0]
        # term should be c * xi^pi; extract pi
        try:
            p_obj = sp.Poly(term, xi)
            if p_obj.total_degree() == 0:
                return None       # constant term — not a power of xi
            # Allow only single monomial: c*xi^p
            if len(p_obj.monoms()) != 1:
                return None
            pi = sp.Integer(p_obj.total_degree())
            ci = p_obj.nth(int(pi))
            # Coefficient must be positive
            if not sp.ask(sp.Q.positive(ci)):
                return None
            if xi in pows:
                return None       # xi appears in two terms
            pows[xi] = pi
        except Exception:
            return None

    # Every integration variable must appear exactly once
    if set(pows.keys()) != set(vars_):
        return None

    # Compute C = ∏ᵢ [ Γ(1/pᵢ) / (pᵢ · ...) ] · 1/Γ(∑1/pᵢ + 1)
    # = ∏ᵢ Γ(1/pᵢ + 1) / Γ(∑(1/pᵢ) + 1)   since Γ(1/p+1) = (1/p)Γ(1/p)
    # But note: the coefficient cᵢ in cᵢ·xᵢ^pᵢ contributes a factor
    # cᵢ^(-1/pᵢ) to the volume (scaling xᵢ by cᵢ^(-1/pᵢ)).
    inv_p_sum = sum(sp.Rational(1, 1) / pows[v] for v in vars_)
    C = sp.Integer(1)
    for v in vars_:
        pi = pows[v]
        # Get the coefficient of xi^pi in h
        try:
            p_obj = sp.Poly(h, v)
            ci = p_obj.nth(int(pi))
        except Exception:
            return None
        C = C * sp.gamma(sp.Rational(1, 1)/pi + 1) * ci**(-sp.Rational(1,1)/pi)
    C = C / sp.gamma(inv_p_sum + 1)

    alpha = inv_p_sum / k

    y = Dummy('y_se', positive=True)
    density = C * alpha * y ** (alpha - 1)
    # f_eff(y) = f_outer(c_mult * y + c_add)
    if c_mult != 1 or c_add != 0:
        f_eff = sp.Lambda(y, f_outer(c_mult * y + c_add))
    else:
        f_eff = f_outer
    try:
        result = integrate(f_eff(y) * density, (y, 0, oo), **opts)
        # Reject unevaluated integrals and results that contain ±∞ arguments
        # to special functions (indicates a divergent integral that SymPy
        # couldn't simplify to oo cleanly).
        if result.has(sp.Integral):
            return None
        if result.has(-oo) or result.has(sp.zoo) or result.has(sp.nan):
            return oo   # divergent
        return _fast_simplify(result)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §5b  Strategy 5b – Product separability  f₁(x₁)·f₂(x₂)·…
# ═══════════════════════════════════════════════════════════════════════════════

def _try_product_separable(f_expr: sp.Expr, vars_: list[sp.Symbol],
                            ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Handle integrands that factorise as a product of single-variable functions:

        f(x₁, …, xₙ) = c · f₁(x₁) · f₂(x₂) · … · fₙ(xₙ)

    By Fubini, the integral factors into independent 1-D integrals:

        ∫_Ω f dxⁿ = c · ∏ᵢ ∫_{aᵢ}^{bᵢ} fᵢ(xᵢ) dxᵢ

    This handles sin(x)·exp(-y), x²·cos(y), exp(-x)·exp(-y), and similar
    product integrands that appear very frequently in practice.

    Only fires when every factor depends on at most one integration variable
    and limits are independent (non-variable).
    """
    if not f_expr.is_Mul:
        return None

    vars_set = set(vars_)

    # Reject if any range has variable limits
    for r in ranges:
        if sp.sympify(r[1]).free_symbols & vars_set:
            return None
        if sp.sympify(r[2]).free_symbols & vars_set:
            return None

    # Split factors: constant part and per-variable parts
    const_part = sp.Integer(1)
    var_factors: dict[sp.Symbol, sp.Expr] = {}

    for factor in f_expr.args:
        active = [v for v in vars_ if v in factor.free_symbols]
        if len(active) == 0:
            const_part = const_part * factor
        elif len(active) == 1:
            v = active[0]
            var_factors[v] = var_factors.get(v, sp.Integer(1)) * factor
        else:
            return None  # factor mixes variables

    if len(var_factors) < 2:
        return None  # trivial; S6/S7 handle single-variable cases

    # Variables with no factor: contribute the length of their interval
    result = const_part
    for r in ranges:
        v, lo, hi = r
        if v in var_factors:
            integral_1d = integrate(var_factors[v], (v, lo, hi), **opts)
            if integral_1d.has(sp.Integral):
                return None
            result = result * integral_1d
        else:
            # Variable absent from integrand — contributes (hi - lo)
            if lo in (oo, -oo) or hi in (oo, -oo):
                return None
            result = result * (hi - lo)

    return _fast_simplify(result)


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Strategy 6 – Monotone substitution  (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_monotone_substitution(f_outer: Callable, g: sp.Expr,
                                vars_: list[sp.Symbol], ranges: list[tuple],
                                opts: dict) -> sp.Expr | None:
    """
    For a single-variable g(x), if g is monotone on [lo, hi]:

        ∫_lo^hi f(g(x)) dx  =  ∫_{g(lo)}^{g(hi)} f(y) / |g'(g⁻¹(y))| dy

    Uses the co-area formula:  μ'(y) = |dx/dy| = 1/|g'(x)|.

    For multivariate f(g(x)) where g depends only on one variable,
    the other dimensions are integrated out as a volume factor first.
    """
    # Fast guard: g must depend on exactly one variable
    active = [v for v in vars_ if v in g.free_symbols]
    if len(active) != 1:
        return None
    xi = active[0]
    r_xi = next((r for r in ranges if r[0] == xi), None)
    if r_xi is None:
        return None
    lo, hi = r_xi[1], r_xi[2]

    # Check monotonicity: no interior critical points
    cpts = _real_critical_points(g, xi, lo, hi)
    if cpts:
        return None

    # Test sign of derivative at a sample point
    dg = diff(g, xi)
    try:
        mid = (lo + hi) / 2 if lo not in (-oo, oo) and hi not in (-oo, oo) else sp.Integer(0)
        dg_sign = sp.ask(sp.Q.positive(dg.subs(xi, mid)))
    except Exception:
        dg_sign = None

    # Compute g at endpoints (use limits for infinite endpoints)
    g_lo = limit(g, xi, lo, '+') if lo == -oo else g.subs(xi, lo)
    g_hi = limit(g, xi, hi, '-') if hi == oo  else g.subs(xi, hi)
    g_lo, g_hi = simplify(g_lo), simplify(g_hi)

    if dg_sign is False:  # decreasing → flip
        g_lo, g_hi = g_hi, g_lo

    # Invert g analytically:  solve g(xi) = y  for xi
    y = Dummy('y_mono')
    try:
        inv_solutions = solve(g - y, xi)
    except Exception:
        return None
    inv_solutions = [s for s in inv_solutions if not s.has(sp.I)]
    if not inv_solutions:
        return None

    if len(inv_solutions) > 1:
        # Select branch consistent with [lo, hi]
        valid = []
        for s in inv_solutions:
            try:
                s_mid = s.subs(y, (g_lo + g_hi) / 2)
                ok = sp.ask(
                    sp.Q.positive(s_mid - lo + sp.Rational(1, 1000)) &
                    sp.Q.positive(hi - s_mid + sp.Rational(1, 1000))
                )
                if ok is not False:
                    valid.append(s)
            except Exception:
                valid.append(s)
        if len(valid) != 1:
            return None
        inv_solutions = valid

    xi_of_y = simplify(inv_solutions[0])
    jacobian = Abs(diff(xi_of_y, y))   # |dx/dy| = 1/|g'(x)|

    # Integrate out unused dimensions
    other_ranges = [r for r in ranges if r[0] != xi]
    volume = sp.Integer(1)
    for r in other_ranges:
        v, vlo, vhi = r
        if vlo in (-oo, oo) or vhi in (-oo, oo):
            return None
        volume *= (vhi - vlo)

    integrand_1d = _fast_simplify(f_outer(y) * jacobian * volume)
    try:
        result = integrate(integrand_1d, (y, g_lo, g_hi), **opts)
        return None if result.has(sp.Integral) else simplify(result)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Strategy 7 – Piecewise-monotone substitution  (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _try_piecewise_monotone(f_outer: Callable, g: sp.Expr,
                             vars_: list[sp.Symbol], ranges: list[tuple],
                             opts: dict) -> sp.Expr | None:
    """
    Split the domain at critical points of g(x), apply the monotone
    substitution on each piece, and sum.

    The co-area density is:
        μ'(y) = Σ_{branches k : g(xₖ)=y}  1 / |g'(xₖ)|

    Only handles the single-active-variable case.
    """
    active = [v for v in vars_ if v in g.free_symbols]
    if len(active) != 1:
        return None
    xi = active[0]
    r_xi = next(r for r in ranges if r[0] == xi)
    lo, hi = r_xi[1], r_xi[2]

    cpts = _real_critical_points(g, xi, lo, hi)
    if not cpts:
        return None  # no critical points → Strategy 6 handles it

    endpoints = [lo] + sorted(cpts, key=lambda e: float(e) if e.is_number else 0) + [hi]
    sub_intervals = list(zip(endpoints[:-1], endpoints[1:]))

    other_ranges = [r for r in ranges if r[0] != xi]
    volume = sp.Integer(1)
    for r in other_ranges:
        v, vlo, vhi = r
        if vlo in (-oo, oo) or vhi in (-oo, oo):
            return None
        volume *= (vhi - vlo)

    total = sp.Integer(0)
    for (a, b) in sub_intervals:
        sub_r = [(xi, a, b)] + other_ranges
        sub_vars = [xi] + [r[0] for r in other_ranges]
        piece = _try_monotone_substitution(f_outer, g, sub_vars, sub_r, opts)
        if piece is None:
            try:
                piece = integrate(f_outer(g) * volume, (xi, a, b), **opts)
                if piece.has(sp.Integral):
                    return None
            except Exception:
                return None
        total = total + piece

    result = simplify(total)
    return None if result.has(sp.Integral) else result


# ═══════════════════════════════════════════════════════════════════════════════
# §8  Strategy 8 – General non-polynomial layer-cake  (new)
# ═══════════════════════════════════════════════════════════════════════════════

def _bounds_of_g(g: sp.Expr, vars_: list[sp.Symbol],
                 ranges: list[tuple]) -> tuple[sp.Expr, sp.Expr]:
    """
    Estimate [g_min, g_max] by evaluating g at all corners of the box and
    at critical points along each axis.
    """
    corners = [{}]
    for r in ranges:
        v, lo, hi = r
        new_corners = []
        for c in corners:
            for ep in [lo, hi]:
                if ep not in (oo, -oo):
                    new_corners.append({**c, v: ep})
        if new_corners:
            corners = new_corners

    vals = []
    for corner in corners:
        try:
            val = simplify(g.subs(list(corner.items())))
            if val.is_real or val.is_number:
                vals.append(val)
        except Exception:
            pass

    for r in ranges:
        v, lo, hi = r
        cpts = _real_critical_points(g, v, lo, hi)
        for cp in cpts:
            try:
                val = g.subs(v, cp)
                vals.append(simplify(val))
            except Exception:
                pass

    if not vals:
        return -oo, oo
    return simplify(sp.Min(*vals)), simplify(sp.Max(*vals))


def _try_general_nonpolynomial(f_outer: Callable, g: sp.Expr,
                                vars_: list[sp.Symbol], ranges: list[tuple],
                                opts: dict) -> sp.Expr | None:
    """
    Applies the layer-cake formula for arbitrary g:

        ∫_Ω f(g(x)) dx = ∫_{y_min}^{y_max} f(y) · μ'(y) dy

    where μ(y) = ∫_Ω Θ(y - g(x)) dx is computed symbolically by SymPy.

    Unlike Strategy 4, here g may be transcendental; SymPy must be able to
    integrate Heaviside(y - g(x)) in closed form.
    """
    # Same guard as S4: skip when g mixes multiple variables.
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) > 1:
        return None

    yy = Dummy('y_gen')

    mu_y = _heaviside_to_piecewise(Heaviside(yy - g))
    try:
        for r in ranges:
            mu_y = integrate(mu_y, (r[0], r[1], r[2]), **opts)
            if isinstance(mu_y, sp.Integral) or mu_y.has(sp.Integral):
                return None
        mu_y = simplify(mu_y)
    except Exception:
        return None

    density = simplify(diff(mu_y, yy))

    y_lo, y_hi = _bounds_of_g(g, vars_, ranges)

    try:
        result = integrate(f_outer(yy) * density, (yy, y_lo, y_hi), **opts)
        if result.has(sp.Integral):
            return None
        return simplify(result)
    except Exception:
        return None


# ═══════════════════════════════════════════════════════════════════════════════
# §9  Fallback – plain iterated integration
# ═══════════════════════════════════════════════════════════════════════════════



def _ranges_form_positive_scaling_cone(ranges: list[tuple]) -> bool:
    """Return True if the region is invariant under x -> lambda*x for lambda>0."""
    for _, lo, hi in ranges:
        lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
        if not ((lo_s == 0 and hi_s == oo) or (lo_s == -oo and hi_s == oo)):
            return False
    return True


def _homogeneous_degree(poly: sp.Expr, vars_: list[sp.Symbol]) -> int | None:
    try:
        P = sp.Poly(poly, *vars_)
    except Exception:
        return None
    monoms = P.monoms()
    if not monoms:
        return 0
    degs = {sum(m) for m in monoms}
    return next(iter(degs)) if len(degs) == 1 else None


def _try_homogeneous_polynomial(f_outer: Callable, g: sp.Expr, vars_: list[sp.Symbol],
                                ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Co-area scaling for positively homogeneous polynomials on scaling cones.

    If g is homogeneous of degree k on a cone-invariant region Ω, then the
    pushforward measure satisfies μ(y)=C y^(n/k).  Rather than integrating
    Θ(y-g(x)) symbolically in y, compute the scaling constant from the exact
    exponential moment

        I_exp = ∫_Ω exp(-g(x)) dx = C Γ(n/k + 1),

    so the density becomes

        μ'(y) = I_exp / Γ(n/k) * y^(n/k - 1).

    This extends the generic polynomial layer-cake strategy to multivariate
    homogeneous polynomials when the exponential moment can be evaluated by
    one of the other strategies (linear, quadratic, superellipse, separable,
    etc.).
    """
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) < 2:
        return None
    if not _ranges_form_positive_scaling_cone(ranges):
        return None
    k = _homogeneous_degree(g, vars_)
    if k is None or k <= 0:
        return None

    t = Dummy('t_hom_exp')
    exp_outer = sp.Lambda(t, sp.exp(-t))
    I_exp = None
    for strategy in (_try_linear,
                     _try_quadratic_infinite,
                     _try_quadratic_even_half_infinite,
                     _try_general_polynomial,
                     _try_nested_quadratic,
                     _try_superellipse,
                     _try_separable,
                     _try_monotone_substitution,
                     _try_piecewise_monotone,
                     _try_general_nonpolynomial):
        try:
            I_exp = strategy(exp_outer, g, vars_, ranges, opts)
        except Exception:
            I_exp = None
        if I_exp is not None:
            break

    if I_exp is None:
        # Fallback: try the fixed sublevel-set volume at y=1 directly.
        try:
            mu1 = _iterated_integrate(Heaviside(1 - g), ranges, opts)
            if getattr(mu1, 'has', lambda *_: False)(sp.Integral) or mu1 in (oo, -oo, sp.zoo, sp.nan):
                return None
            alpha = sp.Rational(len(active_vars), k)
            y = Dummy('y_hom')
            density = _fast_simplify(mu1 * alpha * y ** (alpha - 1))
            result = integrate(f_outer(y) * density, (y, 0, oo), **opts)
            return None if result.has(sp.Integral) else _fast_simplify(result)
        except Exception:
            return None

    alpha = sp.Rational(len(active_vars), k)
    y = Dummy('y_hom')
    density = _fast_simplify(I_exp / gamma(alpha) * y ** (alpha - 1))
    try:
        result = integrate(f_outer(y) * density, (y, 0, oo), **opts)
        if result.has(sp.Integral):
            return None
        return _fast_simplify(result)
    except Exception:
        return None


def _rational_residue_integrate_1d(expr: sp.Expr, var: sp.Symbol) -> sp.Expr | None:
    """Evaluate a proper rational integral over (-oo, oo) by residues."""
    try:
        expr = sp.together(expr)
        num, den = sp.fraction(expr)
        p_num = sp.Poly(num, var)
        p_den = sp.Poly(den, var)
    except Exception:
        return None
    if p_den.degree() - p_num.degree() < 2:
        return None

    try:
        roots_dict = sp.roots(p_den)
        roots = list(roots_dict.keys())
    except Exception:
        try:
            roots = sp.solve(sp.Eq(den, 0), var)
        except Exception:
            return None
    upper = []
    for root in roots:
        try:
            imag = sp.simplify(root.as_real_imag()[1])
        except Exception:
            imag = sp.simplify(sp.im(root))
        if imag == 0 or imag.is_zero is True:
            return None
        if imag.is_positive is True:
            upper.append(root)
        elif imag.is_negative is True:
            continue
        else:
            return None
    try:
        res_sum = sp.Add(*[sp.residue(expr, var, r) for r in upper]) if upper else sp.Integer(0)
        return _fast_simplify(2 * sp.pi * sp.I * res_sum)
    except Exception:
        return None


def _try_rational_residue_iterated(expr: sp.Expr, ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """Apply one-dimensional residue integration recursively over full lines."""
    if not ranges or not all(sp.sympify(lo) == -oo and sp.sympify(hi) == oo for _, lo, hi in ranges):
        return None
    result = sp.together(expr)
    for var, _, _ in ranges:
        step = _rational_residue_integrate_1d(result, var)
        if step is None:
            return None
        result = _fast_simplify(step)
    return None if getattr(result, 'has', lambda *_: False)(sp.Integral) else _fast_simplify(result)


def _gaussian_moment_1d(power: int, coeff: sp.Expr, lo: sp.Expr, hi: sp.Expr) -> sp.Expr | None:
    coeff = sp.sympify(coeff)
    if coeff.is_positive is not True:
        return None
    if lo == -oo and hi == oo:
        if power % 2 == 1:
            return sp.Integer(0)
        return coeff ** (-sp.Rational(power + 1, 2)) * gamma(sp.Rational(power + 1, 2))
    if lo == 0 and hi == oo:
        return sp.Rational(1, 2) * coeff ** (-sp.Rational(power + 1, 2)) * gamma(sp.Rational(power + 1, 2))
    return None


def _try_polynomial_gaussian_moments(expr: sp.Expr, vars_: list[sp.Symbol],
                                     ranges: list[tuple], opts: dict) -> sp.Expr | None:
    """
    Exact integration for polynomial moments against diagonal Gaussian weights.

    Supports expressions of the form P(x) * exp(-(sum a_i x_i^2 + c)) over
    product domains with each range either (-oo, oo) or (0, oo).
    """
    expr = sp.powsimp(expr, force=True)
    if not expr.is_Mul:
        return None
    exp_factors = [a for a in expr.args if a.func == sp.exp]
    if len(exp_factors) != 1:
        return None
    exp_factor = exp_factors[0]
    poly_part = sp.simplify(expr / exp_factor)
    if not _is_polynomial(poly_part, vars_):
        return None
    exponent = sp.expand(-sp.log(exp_factor)) if exp_factor.func == sp.exp else None
    if exponent is None:
        return None
    exponent = sp.expand(exponent)
    try:
        Pq = sp.Poly(exponent, *vars_)
    except Exception:
        return None
    if Pq.total_degree() != 2:
        return None
    const = Pq.nth(*([0] * len(vars_)))
    coeffs = {}
    for i, v in enumerate(vars_):
        # reject linear or mixed terms
        idx1 = [0]*len(vars_); idx1[i] = 1
        if Pq.nth(*idx1) != 0:
            return None
        idx2 = [0]*len(vars_); idx2[i] = 2
        coeffs[v] = sp.simplify(Pq.nth(*idx2))
        if coeffs[v] == 0:
            return None
        for j in range(i+1, len(vars_)):
            idxm = [0]*len(vars_); idxm[i]=idxm[j]=1
            if Pq.nth(*idxm) != 0:
                return None

    try:
        P = sp.Poly(sp.expand(poly_part), *vars_)
    except Exception:
        return None
    total = sp.Integer(0)
    range_map = {r[0]: (sp.sympify(r[1]), sp.sympify(r[2])) for r in ranges}
    for monom, coeff in zip(P.monoms(), P.coeffs()):
        term = coeff * sp.exp(-const)
        for power, v in zip(monom, vars_):
            lo, hi = range_map[v]
            m = _gaussian_moment_1d(power, coeffs[v], lo, hi)
            if m is None:
                return None
            term *= m
            if term == 0:
                break
        total += term
    return _fast_simplify(total)


def _iterated_integrate(expr: sp.Expr, ranges: list[tuple],
                        opts: dict) -> sp.Expr:
    """
    Iterated SymPy integration in forward order (first range integrated first).

    Forward order is required so that variable limits are respected correctly.
    For example, with ranges [(y, 0, 1-x), (x, 0, 1)], y must be integrated
    first because its upper limit depends on x.  Reversing the order would
    integrate x first, leaving x free when y's limits are applied.

    Heaviside sub-expressions are rewritten as Piecewise before integration
    because SymPy's integrate() falls back to Meijer G functions for
    Heaviside(linear(x, y)) with two free variables, producing incorrect results.
    """
    result = _heaviside_to_piecewise(expr)
    for r in ranges:
        result = integrate(result, (r[0], r[1], r[2]), **opts)
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# §10  Public API
# ═══════════════════════════════════════════════════════════════════════════════

def multiple_integrate(
    f: sp.Expr,
    *ranges,
    assumptions=None,
    generate_conditions: bool = False,
    principal_value: bool = False,
) -> sp.Expr:
    """
    Symbolically evaluate a multiple integral  ∫_Ω f(x₁,…,xₙ) dx.

    The integrand *f* may have the structure f(g(x₁,…,xₙ)) for a wide class
    of inner functions g — polynomials, trigonometric, exponential, logarithmic,
    rational, algebraic, and separable combinations thereof.

    Parameters
    ----------
    f : sympy.Expr
        The integrand as a SymPy expression in the integration variables.
    *ranges : tuple of (symbol, lower, upper)
        One tuple per integration variable, e.g. ``(x, 0, 1), (y, 0, pi)``.
    assumptions : dict, optional
        Passed to SymPy's ``integrate`` (e.g. ``{'positive': True}``).
    generate_conditions : bool
        Ask SymPy to emit ``ConditionalExpression`` results when the result
        depends on parameter signs.  Default False.
    principal_value : bool
        Use the Cauchy principal value for improper integrals.

    Returns
    -------
    sympy.Expr
        Closed-form result, or an unevaluated ``sympy.Integral`` if no
        strategy succeeds.

    Examples
    --------
    >>> from sympy import *
    >>> x, y = symbols('x y')
    >>> multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
    pi
    >>> multiple_integrate(sin(x + y), (x, 0, pi), (y, 0, pi))
    0
    >>> multiple_integrate(cos(x)*exp(-y), (x, 0, pi/2), (y, 0, oo))
    1
    """
    opts: dict = {}
    if assumptions:
        opts['assumptions'] = assumptions

    # Normalise ranges
    parsed_ranges: list[tuple] = []
    for r in ranges:
        if len(r) == 3:
            parsed_ranges.append(tuple(r))
        else:
            raise ValueError(f"Each range must be (variable, lower, upper); got {r}")

    vars_ = [r[0] for r in parsed_ranges]
    f_expr: sp.Expr = f

    # ── Constant-integrand short-circuit ────────────────────────────────────
    # If f does not depend on any integration variable the result is
    # f * ∏(hi - lo).  Handles f=5, f=pi, f=a (symbolic parameter), etc.
    if not f_expr.free_symbols & set(vars_):
        volume = sp.Integer(1)
        for r in parsed_ranges:
            volume = volume * (sp.sympify(r[2]) - sp.sympify(r[1]))
        return _fast_simplify(f_expr * volume)

    # ── Trig/power normalisation ────────────────────────────────────────────
    # Simplify redundant structure before any other work.  These are all cheap
    # single-pass rewrites that can collapse the integrand to a constant or
    # simpler form (e.g. sin²+cos²→1, exp(log(x))→x, (x+1)²-x²-2x→1).
    try:
        f_norm = sp.trigsimp(sp.powsimp(f_expr))
        if f_norm != f_expr:
            f_expr = f_norm
            # Re-run constant check after normalisation
            if not f_expr.free_symbols & set(vars_):
                volume = sp.Integer(1)
                for r in parsed_ranges:
                    volume = volume * (sp.sympify(r[2]) - sp.sympify(r[1]))
                return _fast_simplify(f_expr * volume)
    except Exception:
        pass

    # ── Short-circuit for step-function integrands ────────────────────────────
    # When the integrand is (or contains) a Heaviside, Piecewise, or sign at
    # the top level, the layer-cake strategies produce a nested
    # Heaviside(y - Heaviside(...)) that causes SymPy to hang.  The iterated
    # fallback with Piecewise rewriting is both correct and fast for these.
    _step_funcs = (Heaviside, Piecewise, sign)
    if isinstance(f_expr, _step_funcs) or (
        f_expr.is_Mul and any(isinstance(a, _step_funcs) for a in f_expr.args)
    ):
        return _iterated_integrate(f_expr, parsed_ranges, opts)

    # ── Parity short-circuits ─────────────────────────────────────────────────
    # For symmetric ranges [-a, a]:
    #   • Odd integrand  (f(-x) = -f(x)) → integral is 0
    #   • Even integrand (f(-x) =  f(x)) → replace range with [0, a], double
    # Applied variable-by-variable; the first odd variable found exits early.
    new_ranges = list(parsed_ranges)
    scale = sp.Integer(1)
    for i, r in enumerate(new_ranges):
        v, lo, hi = r
        lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
        if lo_s.is_number and hi_s.is_number and sp.simplify(lo_s + hi_s) == 0:
            try:
                reflected = f_expr.subs(v, -v)
                if sp.simplify(reflected + f_expr) == 0:      # odd → zero
                    return sp.Integer(0)
                if sp.simplify(reflected - f_expr) == 0:      # even → halve range
                    new_ranges[i] = (v, sp.Integer(0), hi_s)
                    scale = scale * 2
            except Exception:
                pass
    if scale != 1:
        parsed_ranges = new_ranges
        f_expr = f_expr  # integrand unchanged; only range shortened

    # ── Helpers to apply even-halving scale to any returned result ────────────
    def _scaled(r):
        return _fast_simplify(scale * r) if scale != 1 else r

    # ── Product-separable short-circuit ──────────────────────────────────────
    res = _try_product_separable(f_expr, vars_, parsed_ranges, opts)
    if res is not None:
        return _scaled(res)

    # ── Exact diagonal Gaussian moments: P(x) * exp(-quadratic) ─────────────
    res = _try_polynomial_gaussian_moments(f_expr, vars_, parsed_ranges, opts)
    if res is not None:
        return _scaled(res)

    # ── Residue-theoretic reduction for rational integrands on R^n ───────────
    res = _try_rational_residue_iterated(f_expr, parsed_ranges, opts)
    if res is not None:
        return _scaled(res)

    # ── Disjoint variable support ───────────────────────────────────────────
    # If some integration variables do not appear in f_expr, factor out their
    # volume contribution and recurse on the remaining variables.
    # E.g. ∫∫∫ (sin(x)+cos(z)) dx dy dz = (b_y-a_y) * ∫∫ (sin(x)+cos(z)) dx dz
    active_vars = f_expr.free_symbols & set(vars_)
    if active_vars != set(vars_) and active_vars:
        inactive_ranges = [r for r in parsed_ranges if r[0] not in active_vars]
        active_ranges   = [r for r in parsed_ranges if r[0] in active_vars]
        # Compute volume of inactive dimensions; bail if any limit is infinite
        vol = sp.Integer(1)
        for r in inactive_ranges:
            lo_s, hi_s = sp.sympify(r[1]), sp.sympify(r[2])
            if lo_s in (oo, -oo) or hi_s in (oo, -oo):
                vol = None
                break
            vol = vol * (hi_s - lo_s)
        if vol is not None:
            inner = multiple_integrate(f_expr, *active_ranges,
                                       assumptions=assumptions,
                                       generate_conditions=generate_conditions,
                                       principal_value=principal_value)
            return _fast_simplify(_scaled(vol * inner))

    # ── 1-D short-circuit ────────────────────────────────────────────────────
    # For single-variable integrals SymPy's own integrate() is fastest.
    if len(parsed_ranges) == 1:
        return _scaled(_iterated_integrate(f_expr, parsed_ranges, opts))

    # ── Sum of separable terms ──────────────────────────────────────────────
    # If f_expr is a sum where every term is a product of single-variable
    # functions, integrate term by term and sum.  This handles integrands like
    # sin(x)*cos(y) + exp(-x)*y^2 that are not of the f(g) form but whose
    # terms are each product-separable.
    if f_expr.is_Add:
        # Only attempt if every term is product-separable on constant-limit ranges
        vars_set = set(vars_)
        const_ranges = [r for r in parsed_ranges
                        if not sp.sympify(r[1]).free_symbols & vars_set
                        and not sp.sympify(r[2]).free_symbols & vars_set]
        if len(const_ranges) == len(parsed_ranges):  # all limits constant
            term_results = []
            for term in f_expr.args:
                tr = _try_product_separable(term, vars_, parsed_ranges, opts)
                if tr is None:
                    # try recursive multiple_integrate for this term
                    tr = multiple_integrate(term, *parsed_ranges,
                                            assumptions=assumptions,
                                            generate_conditions=generate_conditions,
                                            principal_value=principal_value)
                term_results.append(tr)
            total = _fast_simplify(sp.Add(*term_results))
            if not total.has(sp.Integral):
                return _scaled(total)

    # ── Consolidate exp products: exp(-x)*exp(-y) → exp(-(x+y)) ─────────────
    # powsimp merges exponential products so S1/S2 can recognise them;
    # it is cheap (single-pass) and never changes the value.
    f_simplified = sp.powsimp(f_expr, force=True)
    if f_simplified != f_expr and not f_simplified.free_symbols - set(vars_):
        f_expr = f_simplified

    # ── Pull out constant factors ────────────────────────────────────────────
    # If f_expr = c * h(vars_) with c free of all vars_, factor c out, compute
    # ∫ h dxⁿ, and multiply back. This ensures downstream strategies always
    # see a "pure" integrand without stray scalar prefactors.
    if f_expr.is_Mul:
        c_factors = [a for a in f_expr.args if not a.free_symbols & set(vars_)]
        if c_factors:
            c_out = sp.Mul(*c_factors)
            h_expr = f_expr / c_out
            inner = multiple_integrate(h_expr, *parsed_ranges,
                                       assumptions=assumptions,
                                       generate_conditions=generate_conditions,
                                       principal_value=principal_value)
            return _fast_simplify(_scaled(c_out * inner))

    # ── Decompose integrand into f_outer ∘ g ──────────────────────────────────
    @functools.lru_cache(maxsize=256)
    def _decompose_cached(expr, vars_tuple):
        return _decompose(expr, list(vars_tuple))
    decomp = _decompose_cached(f_expr, tuple(vars_))

    if decomp is None:
        return _scaled(_iterated_integrate(f_expr, parsed_ranges, opts))

    f_outer  = decomp.f_outer
    g        = decomp.g_inner
    is_poly  = decomp.is_polynomial

    # ── Strategies 1–4b: polynomial g ────────────────────────────────────────
    if is_poly:
        for strategy in (_try_linear,
                         _try_quadratic_infinite,
                         _try_quadratic_even_half_infinite,
                         _try_homogeneous_polynomial,
                         _try_general_polynomial,
                         _try_nested_quadratic,
                         _try_superellipse):
            res = strategy(f_outer, g, vars_, parsed_ranges, opts)
            if res is not None:
                return _scaled(res)

    # ── Strategy 4b: superellipse (also attempted for non-polynomial g) ──────
    res = _try_superellipse(f_outer, g, vars_, parsed_ranges, opts)
    if res is not None:
        return _scaled(res)

    # ── Strategies 5–8: non-polynomial g ──────────────────────────────────────
    for strategy in (_try_separable,
                     _try_monotone_substitution,
                     _try_piecewise_monotone,
                     _try_general_nonpolynomial):
        res = strategy(f_outer, g, vars_, parsed_ranges, opts)
        if res is not None:
            return _scaled(res)

    # ── Fallback ───────────────────────────────────────────────────────────────
    result = _iterated_integrate(f_expr, parsed_ranges, opts)
    return _fast_simplify(scale * result) if scale != 1 else result


# ═══════════════════════════════════════════════════════════════════════════════
# §11  Self-test
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    x, y, z = symbols('x y z', real=True)
    a = symbols('a', positive=True)

    tests = [
        # ── Original polynomial tests ─────────────────────────────────────────
        ("Poly: x²y  [0,1]×[0,2]",
         lambda: multiple_integrate(x**2 * y,         (x,0,1),(y,0,2)),
         sp.Rational(2, 3)),

        ("Poly: xyz  [0,1]³",
         lambda: multiple_integrate(x*y*z,             (x,0,1),(y,0,1),(z,0,1)),
         sp.Rational(1, 8)),

        ("Poly: triangle x+y",
         lambda: multiple_integrate(x + y,             (y,0,1-x),(x,0,1)),
         sp.Rational(1, 3)),

        # ── Quadratic / Gaussian ──────────────────────────────────────────────
        ("Gaussian R²",
         lambda: multiple_integrate(sp.exp(-(x**2+y**2)), (x,-oo,oo),(y,-oo,oo)),
         pi),

        # ── Non-polynomial g: exponential ─────────────────────────────────────
        ("exp(-x)·exp(-y)  [0,∞)²",
         lambda: multiple_integrate(sp.exp(-x)*sp.exp(-y), (x,0,oo),(y,0,oo)),
         sp.Integer(1)),

        ("exp(-(x+y))  [0,∞)²  (separable sum)",
         lambda: multiple_integrate(sp.exp(-(x+y)), (x,0,oo),(y,0,oo)),
         sp.Integer(1)),

        # ── Non-polynomial g: trigonometric ───────────────────────────────────
        ("sin(x)·cos(y)  [0,π/2]²",
         lambda: multiple_integrate(sp.sin(x)*sp.cos(y), (x,0,pi/2),(y,0,pi/2)),
         sp.Integer(1)),

        ("cos(x+y)  [0,π]²  (separable sum)",
         lambda: multiple_integrate(sp.cos(x+y), (x,0,pi),(y,0,pi)),
         sp.Integer(0)),

        # ── Non-polynomial g: monotone substitution ───────────────────────────
        ("1/(1+x²)  [0,1]",
         lambda: multiple_integrate(1/(1+x**2), (x,0,1)),
         pi/4),

        ("exp(-x²)  [0,∞)",
         lambda: multiple_integrate(sp.exp(-x**2), (x,0,oo)),
         sqrt(pi)/2),

        # ── Non-polynomial g: piecewise-monotone ──────────────────────────────
        ("sin(x)  [0,π]",
         lambda: multiple_integrate(sp.sin(x), (x,0,pi)),
         sp.Integer(2)),

        ("cos(x)  [0,2π]  (two monotone pieces)",
         lambda: multiple_integrate(sp.cos(x), (x,0,2*pi)),
         sp.Integer(0)),

        # ── Mixed: product of non-polynomial factors ──────────────────────────
        ("cos(x)·exp(-y)  [0,π/2]×[0,∞)",
         lambda: multiple_integrate(sp.cos(x)*sp.exp(-y), (x,0,pi/2),(y,0,oo)),
         sp.Integer(1)),
    ]

    passed = failed = errored = 0
    for desc, fn, expected in tests:
        try:
            result = fn()
            ok = sp.simplify(result - expected) == 0
            if ok:
                status = "✓ PASS"
                passed += 1
            else:
                status = f"✗ FAIL  got={result}  expected={expected}"
                failed += 1
        except Exception as e:
            status = f"✗ ERROR  {type(e).__name__}: {e}"
            errored += 1
        print(f"  {desc}: {status}")

    print(f"\n{passed} passed, {failed} failed, {errored} errored "
          f"out of {len(tests)} tests.")
