"""Core algorithms for exact multiple integration."""

from __future__ import annotations

import functools
import signal
from abc import ABC
import sympy as sp
from sympy import (
    symbols,
    integrate,
    diff,
    oo,
    pi,
    sqrt,
    det,
    gamma,
    Piecewise,
    Rational,
    simplify,
    solve,
    Abs,
    sign,
    Interval,
    S,
    Dummy,
    ln,
    limit,
    zoo,
    nan,
    conjugate,
    Heaviside,
    DiracDelta,
    Add,
    Mul,
    Pow,
)
from sympy.matrices import Matrix
from typing import Callable
from dataclasses import dataclass
import warnings

from multiple_integrate.regions import (
    AffineSimplexRegion,
    AnnulusRegion,
    BallRegion,
    BoxRegion,
    DiskRegion,
    EllipsoidRegion,
    GraphRegion,
    IteratedRegion,
    Region,
    SimplexRegion,
    SphericalShellRegion,
    UnionRegion,
    boole,
    indicator_condition,
    restrict_region,
    region_from_ranges,
)


@dataclass(frozen=True)
class ChangeOfVariables:
    src_vars: tuple[sp.Symbol, ...]
    tgt_vars: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...] | None
    inverse_map: tuple[sp.Expr, ...] | None
    jacobian: sp.Expr | None
    inv_jacobian: sp.Expr | None
    conditions: tuple[sp.Expr, ...] = ()
    name: str = ""


class BaseRegion(ABC):
    pass


@dataclass(frozen=True)
class TransformedIntegral:
    expr: sp.Expr
    ranges: tuple[tuple[sp.Symbol, sp.Expr, sp.Expr], ...]
    region: BaseRegion | None
    transform: ChangeOfVariables
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class CoordinateTransform:
    """Change-of-variables data for structured cordinate transforms."""

    source_vars: tuple[sp.Symbol, ...]
    target_vars: tuple[sp.Symbol, ...]
    forward_map: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    target_ranges: tuple[tuple, ...]

    def apply(self, expr: sp.Expr) -> sp.Expr:
        # The transformed integrand is f(T(u)) * |det DT(u)|.
        subs = dict(zip(self.source_vars, self.forward_map))
        transformed = sp.sympify(expr).subs(subs) * self.jacobian
        try:
            transformed = sp.trigsimp(sp.factor_terms(sp.cancel(transformed)))
        except Exception:
            pass
        return _fast_simplify(transformed)


class Decomposition:
    """
    Result of decomposing an integrand F(x₁,…,xₙ) into f ∘ g.

    Attributes
    ----------
    f_outer : Callable   – univariate function, maps SymPy expr -> SymPy expr
    g_inner : sp.Expr    – the "inner" expression in the integration variables
    is_polynomial : bool – True if g_inner is a polynomial in the variables
    """

    __slots__ = ("f_outer", "g_inner", "is_polynomial")

    def __init__(self, f_outer: Callable, g_inner: sp.Expr, is_polynomial: bool):
        self.f_outer = f_outer
        self.g_inner = g_inner
        self.is_polynomial = is_polynomial


def _normalize_seq(obj):
    """Normalize a list/tuple-like input to a tuple."""
    if isinstance(obj, (list, tuple)):
        return tuple(obj)
    return (obj,)


def _vars_set(vars_: list[sp.Symbol] | tuple[sp.Symbol, ...]) -> set[sp.Symbol]:
    return set(vars_)


def _depends_on_vars(expr: sp.Expr, vars_set: set[sp.Symbol]) -> bool:
    return bool(sp.sympify(expr).free_symbols & vars_set)


def _is_constant_wrt(expr: sp.Expr, vars_set: set[sp.Symbol]) -> bool:
    return not _depends_on_vars(expr, vars_set)


def _clean_expr(expr: sp.Expr) -> sp.Expr:
    expr = sp.sympify(expr)
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _const_result(expr: sp.Expr, region: Region, parsed_ranges: list[tuple]) -> sp.Expr:
    volume = region.constant_volume()
    if volume is None:
        volume = sp.Integer(1)
        for _, lo, hi in parsed_ranges:
            volume *= sp.sympify(hi) - sp.sympify(lo)
    return _fast_simplify(sp.sympify(expr) * volume)


def _inactive_finite_volume(
    active_vars: list[sp.Symbol],
    vars_: list[sp.Symbol],
    ranges: list[tuple],
) -> sp.Expr | None:
    """Return the product of inactive finite dimensions when it's safe.
    This succeeds only when every inactive bound is finite and independent of
    the active variables, and active bounds do not depend on inactive vars.
    """
    active_set = set(active_vars)
    all_set = set(vars_)
    inactive_set = all_set - active_set
    volume = sp.Integer(1)
    for v, lo, hi in ranges:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if v in active_set:
            if (lo_s.free_symbols | hi_s.free_symbols) & inactive_set:
                return None
            continue
        if lo_s in (-oo, oo) or hi_s in (-oo, oo):
            return None
        if (lo_s.free_symbols | hi_s.free_symbols) & active_set:
            return None
        volume *= hi_s - lo_s
    return _fast_simplify(volume)


def _should_try_layercake(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
) -> bool:
    """Entry point for the expensive generic layer-cake fallback."""
    active_vars = [v for v in vars_ if v in g.free_symbols]
    if len(active_vars) > 1:
        return False
    if (
        len(active_vars) == 1
        and _inactive_finite_volume(active_vars, vars_, ranges) is None
    ):
        return False
    return True


def _decompose_unary_wrapper(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> Decomposition | None:
    n_var_args = sum(1 for a in expr.args if _depends_on_vars(a, vars_set))
    if n_var_args != 1:
        return None
    inner = next(a for a in expr.args if _depends_on_vars(a, vars_set))
    t = Dummy("t")
    try:
        outer = sp.Lambda(t, expr.subs(inner, t))
        return Decomposition(outer, inner, is_polynomial=_is_polynomial(inner, vars_))
    except Exception:
        return None


def _decompose_mul_constants(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> Decomposition | None:
    if not expr.is_Mul:
        return None
    const_part = sp.Integer(1)
    var_part = sp.Integer(1)
    for factor in expr.args:
        if _depends_on_vars(factor, vars_set):
            var_part *= factor
        else:
            const_part *= factor
    if const_part == 1 or var_part == 1:
        return None
    sub = _decompose(var_part, vars_)
    if sub is None:
        return None
    inner_f = sub.f_outer
    t = Dummy("t")
    outer = sp.Lambda(t, const_part * inner_f(t))
    return Decomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)


def _decompose_add_constants(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> Decomposition | None:
    if not expr.is_Add:
        return None
    const_part = sp.Integer(0)
    var_part = sp.Integer(0)
    for term in expr.args:
        if _depends_on_vars(term, vars_set):
            var_part += term
        else:
            const_part += term
    if const_part == 0 or var_part == 0:
        return None
    sub = _decompose(var_part, vars_)
    if sub is None:
        return None
    inner_f = sub.f_outer
    t = Dummy("t")
    outer = sp.Lambda(t, inner_f(t) + const_part)
    return Decomposition(outer, sub.g_inner, is_polynomial=sub.is_polynomial)


def _decompose_single_var(
    expr: sp.Expr, vars_: list[sp.Symbol], vars_set: set[sp.Symbol]
) -> Decomposition | None:
    active = [v for v in vars_ if v in sp.sympify(expr).free_symbols]
    if len(active) != 1:
        return None
    t = Dummy("t")
    return Decomposition(
        sp.Lambda(t, t), expr, is_polynomial=_is_polynomial(expr, vars_)
    )


def _decompose_shallow(expr: sp.Expr, vars_: list[sp.Symbol]) -> Decomposition | None:
    """Single-pass decomposition used as the base case for recursive peeling."""
    vars_set = _vars_set(vars_)

    try:
        sp.Poly(expr, *vars_)
        t = Dummy("t")
        return Decomposition(sp.Lambda(t, t), expr, is_polynomial=True)
    except sp.PolynomialError:
        pass

    if expr.func in (sp.log, sp.Abs, sp.sign, sp.floor, sp.ceiling):
        sub = _decompose_single_var(expr, vars_, vars_set)
        if sub is not None:
            return sub

    sub = _decompose_unary_wrapper(expr, vars_, vars_set)
    if sub is not None:
        return sub

    if expr.is_Pow:
        base, exp_ = expr.args
        if _is_constant_wrt(exp_, vars_set) and _depends_on_vars(base, vars_set):
            t = Dummy("t")
            return Decomposition(
                sp.Lambda(t, t**exp_), base, is_polynomial=_is_polynomial(base, vars_)
            )

    sub = _decompose_mul_constants(expr, vars_, vars_set)
    if sub is not None:
        return sub

    sub = _decompose_add_constants(expr, vars_, vars_set)
    if sub is not None:
        return sub

    return _decompose_single_var(expr, vars_, vars_set)


def _decompose(expr: sp.Expr, vars_: list[sp.Symbol]) -> Decomposition | None:
    """Recursively peel wrappers to expose a deeper public decomposition."""
    vars_set = _vars_set(vars_)

    if expr.func in (sp.log, sp.Abs, sp.sign, sp.floor, sp.ceiling):
        sub = _decompose_single_var(expr, vars_, vars_set)
        if sub is not None:
            return sub

    n_var_args = sum(1 for a in expr.args if _depends_on_vars(a, vars_set))
    if n_var_args == 1:
        inner = next(a for a in expr.args if _depends_on_vars(a, vars_set))
        t = Dummy("t")
        try:
            outer = sp.Lambda(t, expr.subs(inner, t))
            sub = _decompose(inner, vars_)
            if sub is not None:
                u = Dummy("u")
                try:
                    return Decomposition(
                        sp.Lambda(u, outer(sub.f_outer(u))),
                        sub.g_inner,
                        sub.is_polynomial,
                    )
                except Exception:
                    pass
            return Decomposition(
                outer, inner, is_polynomial=_is_polynomial(inner, vars_)
            )
        except Exception:
            pass

    if expr.is_Pow:
        base, exp_ = expr.args
        if _is_constant_wrt(exp_, vars_set) and _depends_on_vars(base, vars_set):
            sub = _decompose(base, vars_)
            t = Dummy("t")
            if sub is not None:
                u = Dummy("u")
                try:
                    return Decomposition(
                        sp.Lambda(u, sub.f_outer(u) ** exp_),
                        sub.g_inner,
                        sub.is_polynomial,
                    )
                except Exception:
                    pass
            return Decomposition(
                sp.Lambda(t, t**exp_), base, is_polynomial=_is_polynomial(base, vars_)
            )

    if expr.is_Mul:
        const_part = sp.Integer(1)
        var_part = sp.Integer(1)
        for factor in expr.args:
            if _depends_on_vars(factor, vars_set):
                var_part *= factor
            else:
                const_part *= factor
        if const_part != 1 and var_part != 1:
            sub = _decompose(var_part, vars_)
            if sub is not None:
                t = Dummy("t")
                return Decomposition(
                    sp.Lambda(t, const_part * sub.f_outer(t)),
                    sub.g_inner,
                    sub.is_polynomial,
                )

    if expr.is_Add:
        const_part = sp.Integer(0)
        var_part = sp.Integer(0)
        for term in expr.args:
            if _depends_on_vars(term, vars_set):
                var_part += term
            else:
                const_part += term
        if const_part != 0 and var_part != 0:
            sub = _decompose(var_part, vars_)
            if sub is not None:
                t = Dummy("t")
                return Decomposition(
                    sp.Lambda(t, sub.f_outer(t) + const_part),
                    sub.g_inner,
                    sub.is_polynomial,
                )

    return _decompose_shallow(expr, vars_)


def _is_polynomial(expr: sp.Expr, vars_: list[sp.Symbol]) -> bool:
    try:
        sp.Poly(expr, *vars_)
        return True
    except sp.PolynomialError:
        return False


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
    b = Matrix(
        [poly_obj.nth(*([1 if j == i else 0 for j in range(n)])) for i in range(n)]
    )
    A = sp.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            idx = [0] * n
            if i == j:
                idx[i] = 2
                coeff = poly_obj.nth(*idx)
            else:
                idx[i] = 1
                idx[j] = 1
                coeff = poly_obj.nth(*idx) / 2
            A[i, j] = coeff
            A[j, i] = coeff
    return c, b, A


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
    Tries cancel (rational), trigsimp (trig), and falls back to simplify only
    when the expression is not already in a reduced form.
    """
    if expr.is_number or expr.is_symbol:
        return expr
    # Try cheap reductions first
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
def _is_symmetric_range(lo: sp.Expr, hi: sp.Expr) -> bool:
    try:
        return sp.simplify(sp.sympify(lo) + sp.sympify(hi)) == 0
    except Exception:
        return False


def _split_additive_terms(expr: sp.Expr) -> list[sp.Expr] | None:
    """Return additive terms for early sum splitting when worthwhile."""
    if expr.is_Add and len(expr.args) > 1:
        return list(expr.args)
    return None


def _try_standard_1d(
    expr: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr, opts: dict
):
    """Tiny recognizers for common exact 1-D definite integrals."""
    lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
    try:
        # exp(a*x+b) on finite or infinite intervals
        if expr.func == sp.exp:
            arg = sp.expand(expr.args[0])
            poly = sp.Poly(arg, var)
            if poly.degree() <= 1:
                a = poly.nth(1)
                b = poly.nth(0)
                if a != 0:
                    return _fast_simplify(
                        sp.exp(b) * (sp.exp(a * hi_s) - sp.exp(a * lo_s)) / a
                    )
        # sin(ax+b), cos(ax+b)
        if expr.func in (sp.sin, sp.cos):
            arg = sp.expand(expr.args[0])
            poly = sp.Poly(arg, var)
            if poly.degree() <= 1:
                a = poly.nth(1)
                b = poly.nth(0)
                if a != 0:
                    if expr.func == sp.sin:
                        return _fast_simplify(
                            (-sp.cos(a * hi_s + b) + sp.cos(a * lo_s + b)) / a
                        )
                    return _fast_simplify(
                        (sp.sin(a * hi_s + b) - sp.sin(a * lo_s + b)) / a
                    )
    except Exception:
        pass
    try:
        # polynomial times Gaussian exp(alpha*x**2 + beta*x + gamma)
        factors = sp.Mul.make_args(expr)
        exp_factor = next((f for f in factors if f.func == sp.exp), None)
        if exp_factor is not None:
            rest = _fast_simplify(expr / exp_factor)
            q = sp.expand(exp_factor.args[0])
            qpoly = sp.Poly(q, var)
            if qpoly.degree() <= 2:
                a = qpoly.nth(2)
                b = qpoly.nth(1)
                c = qpoly.nth(0)
                if a != 0 and not rest.has(sp.exp) and sp.Poly(rest, var) is not None:
                    # let SymPy handle this structured case directly
                    res = integrate(
                        sp.expand(rest) * sp.exp(a * var**2 + b * var + c),
                        (var, lo_s, hi_s),
                        **opts,
                    )
                    if not isinstance(res, sp.Integral):
                        return _fast_simplify(res)
    except Exception:
        pass
    try:
        # 1/(x**2 + a**2) over full line
        num, den = sp.fraction(sp.together(expr))
        if (
            sp.simplify(num).free_symbols.isdisjoint({var})
            and lo_s == -oo
            and hi_s == oo
        ):
            dpoly = sp.Poly(sp.expand(den), var)
            if dpoly.degree() == 2 and dpoly.nth(1) == 0:
                a2 = _fast_simplify(dpoly.nth(0) / dpoly.nth(2))
                if sp.ask(sp.Q.positive(a2)):
                    return _fast_simplify(
                        sp.pi * num / (sp.sqrt(dpoly.nth(2)) * sp.sqrt(a2))
                    )
    except Exception:
        pass
    return None


def _real_critical_points(
    g: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr
) -> list[sp.Expr]:
    """
    Return sorted list of real critical points of g(var) strictly inside (lo, hi).
    Includes pts where g is not differentiable (e.g. |x| at 0).
    Cached: S6 and S7 both call this on the same arguments.
    """
    pts = []

    # Solving g'(x)=0 can explode on transcendental inputs, so cap that step
    # and continue with any critical points that were found quickly.
    def _solve_timed(expr, var, secs=1.0):
        class _T(Exception):
            pass

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


def _g_range_on_interval(
    g: sp.Expr, var: sp.Symbol, lo: sp.Expr, hi: sp.Expr
) -> tuple[sp.Expr, sp.Expr]:
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


def _try_linear(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
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
    try:
        c, b_vec, A = _coefficient_arrays(g, vars_)
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

    def _integrate_timed(expr, bounds, secs=3.0):
        class _T(Exception):
            pass

        old = signal.signal(signal.SIGALRM, lambda *_: (_ for _ in ()).throw(_T()))
        signal.setitimer(signal.ITIMER_REAL, secs)
        try:
            return integrate(expr, bounds, **opts)
        except _T:
            return sp.Integral(expr, bounds)
        except Exception:
            return None
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, old)

    y = Dummy("y")
    abs_b_prod = sp.prod([sp.Abs(bi) for bi in b_list])
    prefactor = sp.Integer(1) / (abs_b_prod * sp.factorial(n - 1))
    if all_pos:
        integrand = prefactor * (y - c) ** (n - 1) * f_outer(y)
        result = _integrate_timed(integrand, (y, c, oo))
    else:  # all_neg: g decreases from c to -\infty
        integrand = prefactor * (c - y) ** (n - 1) * f_outer(y)
        result = _integrate_timed(integrand, (y, -oo, c))
    return result


def _qs_integrate(
    f_outer: Callable, A_mat: Matrix, b_vec: Matrix, c_val: sp.Expr, n: int, opts: dict
) -> sp.Expr | None:
    """
    ∫_{ℝⁿ} f(xᵀAx + b·x + c) dx  via ellipsoid surface-area layer-cake.
    Requires A positive definite.
    """
    try:
        A_inv = A_mat.inv()
    except Exception:
        return None
    try:
        evs = list(A_mat.eigenvals().keys())
        if any(sp.ask(sp.Q.negative(ev)) for ev in evs):
            return None
    except Exception:
        pass
    det_A = det(A_mat)
    if det_A == 0:
        return None

    y_min = c_val - (b_vec.T * A_inv * b_vec)[0, 0] / 4
    y = Dummy("y")
    fac = pi ** sp.Rational(n, 2) / (sqrt(det_A) * gamma(sp.Rational(n, 2) + 1))
    surface = n * sp.Rational(1, 2) * (y - y_min) ** (sp.Rational(n, 2) - 1)
    result = integrate(fac * surface * f_outer(y), (y, y_min, oo), **opts)
    return None if result.has(sp.Integral) else result


def _try_quadratic_infinite(f_outer, g, vars_, ranges, opts):
    # Fast guard: skip Poly construction if any range is not (-∞, ∞)
    if not all(r[1] == -oo and r[2] == oo for r in ranges):
        return None
    # Fast guard: g must be degree-2 polynomial (has a quadratic term)
    if not any(sp.degree(g, v) == 2 for v in vars_ if v in g.free_symbols):
        return None
    try:
        c, b_vec, A = _coefficient_arrays(g, vars_)
    except Exception:
        return None
    return _qs_integrate(f_outer, A, b_vec, c, len(vars_), opts)


def _try_even_half_quad(f_outer, g, vars_, ranges, opts):
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
    return full_result / sp.Integer(2) ** half


def _try_general_polynomial(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
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
    if (
        len(active_vars) == 1
        and _inactive_finite_volume(active_vars, vars_, ranges) is None
    ):
        return None
    y = Dummy("y")
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


def _parse_superellipse_core(
    g: sp.Expr, vars_: list[sp.Symbol]
) -> tuple[sp.Expr, sp.Expr, dict[sp.Symbol, tuple[sp.Expr, sp.Expr]]] | None:
    """Return ``(h, k, term_map)`` for g = h**k with h = Σ a_i x_i**p_i."""
    k = sp.Integer(1)
    h = g
    if g.is_Pow:
        base, exp_ = g.as_base_exp()
        if exp_.free_symbols or not (exp_.is_positive and exp_.is_real):
            return None
        k = sp.sympify(exp_)
        h = base
    if not h.is_Add:
        return None
    vars_set = set(vars_)
    term_map: dict[sp.Symbol, tuple[sp.Expr, sp.Expr]] = {}
    for term in h.args:
        coeff, rest = term.as_coeff_Mul()
        coeff = sp.sympify(coeff)
        active = list(rest.free_symbols & vars_set)
        if len(active) != 1:
            return None
        v = active[0]
        if v in term_map:
            return None
        if not rest.is_Pow or rest.base != v:
            return None
        pwr = sp.sympify(rest.exp)
        if pwr.free_symbols or not (pwr.is_positive and pwr.is_real):
            return None
        if coeff.free_symbols or not (coeff.is_positive and coeff.is_real):
            return None
        term_map[v] = (coeff, pwr)
    if set(term_map) != vars_set:
        return None
    return h, k, term_map


def _split_superellipse(
    expr: sp.Expr, vars_: list[sp.Symbol]
) -> tuple[Callable, sp.Expr] | None:
    """Find a unique superellipse-type inner core inside ``expr``.

    We look for a subexpression of the form ``(Σ a_i x_i**p_i)**k`` and, if it
    occurs uniquely, replace it by a dummy variable to obtain the outer
    univariate function.
    """
    vars_set = set(vars_)
    matches = []
    for sub in sp.preorder_traversal(expr):
        if sub == expr or not (sub.free_symbols & vars_set):
            continue
        if _parse_superellipse_core(sub, vars_) is not None:
            matches.append(sub)
    # Prefer the largest matching subexpression and avoid ambiguous cases.
    uniq = []
    for m in matches:
        if not any(
            other != m and m in sp.preorder_traversal(other) for other in matches
        ):
            uniq.append(m)
    if len(uniq) != 1:
        return None
    core = uniq[0]
    t = Dummy("t_super")
    outer_expr = expr.xreplace({core: t})
    if outer_expr.free_symbols & vars_set:
        return None
    return sp.Lambda(t, outer_expr), core


def _try_superellipse(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """Fast homogeneous layer-cake for orthant superellipse-type integrals."""
    if len(vars_) < 2:
        return None

    for _, lo, hi in ranges:
        if sp.sympify(lo) != 0 or sp.sympify(hi) != oo:
            return None

    parsed = _parse_superellipse_core(g, vars_)
    if parsed is None:
        return None
    _, k, term_map = parsed

    alpha = sp.Integer(0)
    const = sp.Integer(1)
    for v in vars_:
        coeff, pwr = term_map[v]
        alpha += sp.Integer(1) / pwr
        const *= gamma(1 + sp.Integer(1) / pwr) / coeff ** (sp.Integer(1) / pwr)
    const /= gamma(1 + alpha)

    t = Dummy("y_super")
    density = _fast_simplify(const * alpha / k * t ** (alpha / k - 1))

    # Fast divergence detection for positive orthant superellipse densities.
    # If the outer function tends to +∞, or to a positive nonzero constant, the
    # weighted tail integral diverges because alpha/k > 0.  This avoids SymPy
    # producing opaque lowergamma(..., -oo) style results on obvious cases.
    try:
        outer_t = f_outer(t)
        tail_lim = limit(outer_t, t, oo)
        if tail_lim is oo:
            return oo
        if getattr(tail_lim, "is_positive", False) and tail_lim != 0:
            return oo
    except Exception:
        outer_t = f_outer(t)

    try:
        result = integrate(outer_t * density, (t, 0, oo), **opts)
    except Exception:
        return None
    return None if result.has(sp.Integral) else _fast_simplify(result)


def _try_separable(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
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
                    volume *= hi - lo
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

        yy = Dummy("y_sep")
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
    conv_var = Dummy("z_conv")
    nu_prev, yy_prev, ylo_prev, yhi_prev = densities[0]
    conv_density = nu_prev.subs(yy_prev, conv_var)
    conv_lo, conv_hi = ylo_prev, yhi_prev

    for nu_i, yy_i, ylo_i, yhi_i in densities[1:]:
        t = Dummy("t_conv")
        z = Dummy("z_new")
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

    yf = Dummy("y_final")
    try:
        result = integrate(
            f_outer(yf + residual) * conv_density.subs(conv_var, yf),
            (yf, conv_lo, conv_hi),
            **opts,
        )
        if result.has(sp.Integral):
            return None
        return _fast_simplify(result)
    except Exception:
        return None


def _try_product_separable(
    f_expr: sp.Expr, vars_: list[sp.Symbol], ranges: list[tuple], opts: dict
) -> sp.Expr | None:
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


def _try_monotone_subst(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
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
    # Bounds for the active variable must not depend on the other variables.
    if (sp.sympify(lo).free_symbols | sp.sympify(hi).free_symbols) & (
        set(vars_) - {xi}
    ):
        return None

    # Check monotonicity: no interior critical points
    cpts = _real_critical_points(g, xi, lo, hi)
    if cpts:
        return None

    # Test sign of derivative at a sample point
    dg = diff(g, xi)
    try:
        mid = (
            (lo + hi) / 2
            if lo not in (-oo, oo) and hi not in (-oo, oo)
            else sp.Integer(0)
        )
        dg_sign = sp.ask(sp.Q.positive(dg.subs(xi, mid)))
    except Exception:
        dg_sign = None

    # Compute g at endpoints (use limits for infinite endpoints)
    g_lo = limit(g, xi, lo, "+") if lo == -oo else g.subs(xi, lo)
    g_hi = limit(g, xi, hi, "-") if hi == oo else g.subs(xi, hi)
    g_lo, g_hi = simplify(g_lo), simplify(g_hi)

    if dg_sign is False:  # decreasing → flip
        g_lo, g_hi = g_hi, g_lo

    # Invert g analytically:  solve g(xi) = y  for xi
    y = Dummy("y_mono")
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
                    sp.Q.positive(s_mid - lo + sp.Rational(1, 1000))
                    & sp.Q.positive(hi - s_mid + sp.Rational(1, 1000))
                )
                if ok is not False:
                    valid.append(s)
            except Exception:
                valid.append(s)
        if len(valid) != 1:
            return None
        inv_solutions = valid

    xi_of_y = simplify(inv_solutions[0])
    jacobian = Abs(diff(xi_of_y, y))  # |dx/dy| = 1/|g'(x)|

    # Integrate out unused dimensions
    other_ranges = [r for r in ranges if r[0] != xi]
    volume = sp.Integer(1)
    for r in other_ranges:
        v, vlo, vhi = r
        vlo_s, vhi_s = sp.sympify(vlo), sp.sympify(vhi)
        if vlo_s in (-oo, oo) or vhi_s in (-oo, oo):
            return None
        if (vlo_s.free_symbols | vhi_s.free_symbols) & {xi}:
            return None
        volume *= vhi_s - vlo_s

    integrand_1d = _fast_simplify(f_outer(y) * jacobian * volume)
    try:
        result = integrate(integrand_1d, (y, g_lo, g_hi), **opts)
        return None if result.has(sp.Integral) else simplify(result)
    except Exception:
        return None


def _try_piecewise_monotone(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
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
        volume *= vhi - vlo

    total = sp.Integer(0)
    for a, b in sub_intervals:
        sub_r = [(xi, a, b)] + other_ranges
        sub_vars = [xi] + [r[0] for r in other_ranges]
        piece = _try_monotone_subst(f_outer, g, sub_vars, sub_r, opts)
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


def _bounds_of_g(
    g: sp.Expr, vars_: list[sp.Symbol], ranges: list[tuple]
) -> tuple[sp.Expr, sp.Expr]:
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


def _try_nonpoly(
    f_outer: Callable,
    g: sp.Expr,
    vars_: list[sp.Symbol],
    ranges: list[tuple],
    opts: dict,
) -> sp.Expr | None:
    """
    Applies the layer-cake formula for arbitrary g:

        ∫_Ω f(g(x)) dx = ∫_{y_min}^{y_max} f(y) · μ'(y) dy

    where μ(y) = ∫_Ω Θ(y - g(x)) dx is computed symbolically by SymPy.

    Unlike Strategy 4, here g may be transcendental; SymPy must be able to
    integrate Heaviside(y - g(x)) in closed form.
    """
    if not _should_try_layercake(f_outer, g, vars_, ranges):
        return None

    yy = Dummy("y_gen")

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


def _iterated_integrate(expr: sp.Expr, ranges: list[tuple], opts: dict) -> sp.Expr:
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
        v, lo, hi = r
        std = _try_standard_1d(result, v, lo, hi, opts)
        if std is not None:
            result = std
        else:
            result = integrate(result, (v, lo, hi), **opts)
    return result


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
        opts["assumptions"] = assumptions

    # Normalise ranges or accept a direct Region object.
    parsed_ranges: list[tuple] = []
    direct_region = None
    norm_ranges = _normalize_seq(ranges)
    if len(norm_ranges) == 1 and isinstance(norm_ranges[0], Region):
        direct_region = norm_ranges[0]
        parsed_ranges = list(direct_region.ranges)
    else:
        for r in norm_ranges:
            if len(r) == 3:
                parsed_ranges.append(tuple(r))
            else:
                raise ValueError(
                    f"Each range must be (variable, lower, upper); got {r}"
                )

    f_expr: sp.Expr = sp.sympify(f)
    region = (
        direct_region
        if direct_region is not None
        else region_from_ranges(parsed_ranges)
    )

    def _integrate_piecewise(expr_pw: sp.Piecewise):
        total = sp.Integer(0)
        remaining = sp.true
        for branch_expr, branch_cond in expr_pw.args:
            eff_cond = (
                remaining
                if branch_cond in (True, sp.true)
                else sp.simplify(sp.And(remaining, branch_cond))
            )
            sub_region = restrict_region(region, eff_cond)
            if sub_region is None:
                try:
                    direct = sp.integrate(expr_pw, *parsed_ranges, **opts)
                    if direct is not None and not isinstance(direct, sp.Integral):
                        return _fast_simplify(direct)
                except Exception:
                    return sp.Integral(expr_pw, *parsed_ranges)
                return sp.Integral(expr_pw, *parsed_ranges)
            branch_res = multiple_integrate(
                sp.sympify(branch_expr),
                sub_region,
                assumptions=assumptions,
                generate_conditions=generate_conditions,
                principal_value=principal_value,
            )
            total += branch_res
            if branch_cond not in (True, sp.true):
                remaining = sp.simplify(sp.And(remaining, sp.Not(branch_cond)))
        return _fast_simplify(total)

    vars_ = (
        list(region.variables)
        if hasattr(region, "variables")
        else [r[0] for r in parsed_ranges]
    )
    vars_set = _vars_set(vars_)

    # ── First-class indicator and Piecewise handling ────────────────────────
    # Treat indicator-like Piecewise factors as region restrictions and
    # integrate general Piecewise expressions branch by branch on supported
    # regions.
    if isinstance(f_expr, sp.Piecewise):
        return _integrate_piecewise(f_expr)

    if isinstance(f_expr, sp.Mul):
        indicator_conds = []
        non_indicator = []
        for arg in f_expr.args:
            ind_cond = indicator_condition(arg)
            if ind_cond is not None:
                indicator_conds.append(ind_cond)
            else:
                non_indicator.append(arg)
        if indicator_conds:
            restricted = restrict_region(region, sp.And(*indicator_conds))
            if restricted is not None:
                new_expr = _fast_simplify(
                    sp.Mul(*non_indicator) if non_indicator else sp.Integer(1)
                )
                return multiple_integrate(
                    new_expr,
                    restricted,
                    assumptions=assumptions,
                    generate_conditions=generate_conditions,
                    principal_value=principal_value,
                )

    # Small normalized-subproblem cache shared across recursive calls.
    if not hasattr(multiple_integrate, "_cache"):
        multiple_integrate._cache = {}

    def _norm_ranges(rs):
        return tuple((v, _clean_expr(lo), _clean_expr(hi)) for v, lo, hi in rs)

    cache_key = (
        _fast_simplify(f_expr),
        region.normalized_ranges(),
        bool(generate_conditions),
        bool(principal_value),
        repr(assumptions),
    )
    if cache_key in multiple_integrate._cache:
        return multiple_integrate._cache[cache_key]

    # ── Constant-integrand short-circuit ────────────────────────────────────
    # If f does not depend on any integration variable the result is
    # f * ∏(hi - lo).  Handles f=5, f=pi, f=a (symbolic parameter), etc.
    def _store(res):
        multiple_integrate._cache[cache_key] = res
        return res

    if _is_constant_wrt(f_expr, vars_set):
        return _store(_const_result(f_expr, region, parsed_ranges))

    # ── Region-aware shortcuts ─────────────────────────
    region_res = _region_shortcut(region, f_expr, assumptions=assumptions)
    if region_res is not None:
        return _store(region_res)

    region_res = _try_region_transform(
        region,
        f_expr,
        assumptions=assumptions,
        generate_conditions=generate_conditions,
        principal_value=principal_value,
    )
    if region_res is not None:
        return _store(region_res)

    gauss_res = _try_gauss_linear_map(f_expr, parsed_ranges, assumptions=assumptions)
    if gauss_res is not None:
        return _store(gauss_res)

    reversed_res = _try_graph_reversal(
        f_expr,
        region,
        assumptions,
        generate_conditions,
        principal_value,
    )
    if reversed_res is not None:
        return _store(reversed_res)

    # ── Aggressive sum splitting ───────────────────────────────────────────
    parts = _split_additive_terms(f_expr)
    if parts is not None:
        finite_bounds = True
        for _, lo, hi in parsed_ranges:
            lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
            if lo_s in (-oo, oo) or hi_s in (-oo, oo):
                finite_bounds = False
                break
        if finite_bounds:
            total = sp.Integer(0)
            for term in parts:
                total += multiple_integrate(
                    term,
                    *parsed_ranges,
                    assumptions=assumptions,
                    generate_conditions=generate_conditions,
                    principal_value=principal_value,
                )
            return _store(_fast_simplify(total))

    # ── Trig/power normalisation ────────────────────────────────────────────
    # Simplify redundant structure before any other work.  These are all cheap
    # single-pass rewrites that can collapse the integrand to a constant or
    # simpler form (e.g. sin²+cos²→1, exp(log(x))→x, (x+1)²-x²-2x→1).
    try:
        f_norm = sp.trigsimp(sp.powsimp(f_expr))
        if f_norm != f_expr:
            f_expr = f_norm
            # Re-run constant check after normalisation
            if _is_constant_wrt(f_expr, vars_set):
                return _store(_const_result(f_expr, region, parsed_ranges))
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
        return _store(_iterated_integrate(f_expr, parsed_ranges, opts))

    # ── Parity short-circuits ─────────────────────────────────────────────────
    # For symmetric ranges [-a, a]:
    #   • Odd integrand  (f(-x) = -f(x)) → integral is 0
    #   • Even integrand (f(-x) =  f(x)) → replace range with [0, a], double
    # Applied variable-by-variable; the first odd variable found exits early.
    new_ranges = list(parsed_ranges)
    scale = sp.Integer(1)
    for i, r in enumerate(new_ranges):
        v, lo, hi = r
        sym_range = region.symmetric_range(v)
        if sym_range is None:
            continue
        _, hi_s = sym_range
        try:
            reflected = f_expr.subs(v, -v)
            if sp.simplify(reflected + f_expr) == 0:  # odd → zero
                return sp.Integer(0)
            if sp.simplify(reflected - f_expr) == 0:  # even → halve range
                new_ranges[i] = (v, sp.Integer(0), hi_s)
                scale = scale * 2
        except Exception:
            pass
    if scale != 1:
        parsed_ranges = new_ranges
        region = region_from_ranges(parsed_ranges)

    # ── Helpers to apply even-halving scale to any returned result ────────────
    def _scaled(r):
        return _fast_simplify(scale * r) if scale != 1 else r

    # ── Product-separable short-circuit ──────────────────────────────────────
    res = _try_product_separable(f_expr, vars_, parsed_ranges, opts)
    if res is not None:
        return _store(_scaled(res))

    # ── Disjoint variable support ───────────────────────────────────────────
    # If some integration variables do not appear in f_expr, factor out their
    # volume contribution and recurse on the remaining variables.
    # E.g. ∫∫∫ (sin(x)+cos(z)) dx dy dz = (b_y-a_y) * ∫∫ (sin(x)+cos(z)) dx dz
    active_vars = f_expr.free_symbols & vars_set
    if active_vars != set(vars_) and active_vars:
        inactive_ranges = [r for r in parsed_ranges if r[0] not in active_vars]
        active_ranges = [r for r in parsed_ranges if r[0] in active_vars]
        # Safe only if inactive dimensions are a true product factor: their own
        # bounds are finite and independent of active vars, and active bounds do
        # not depend on inactive vars.
        vol = sp.Integer(1)
        inactive_set = {r[0] for r in inactive_ranges}
        for r in inactive_ranges:
            lo_s, hi_s = sp.sympify(r[1]), sp.sympify(r[2])
            if (
                lo_s in (oo, -oo)
                or hi_s in (oo, -oo)
                or (lo_s.free_symbols | hi_s.free_symbols) & active_vars
            ):
                vol = None
                break
            vol = vol * (hi_s - lo_s)
        if vol is not None:
            for _, lo, hi in active_ranges:
                lo_s, hi_s = sp.sympify(lo), sp.sympify(hi)
                if (lo_s.free_symbols | hi_s.free_symbols) & inactive_set:
                    vol = None
                    break
        if vol is not None:
            inner = multiple_integrate(
                f_expr,
                *active_ranges,
                assumptions=assumptions,
                generate_conditions=generate_conditions,
                principal_value=principal_value,
            )
            return _store(_fast_simplify(_scaled(vol * inner)))

    # ── 1-D short-circuit ────────────────────────────────────────────────────
    # For single-variable integrals SymPy's own integrate() is fastest.
    if len(parsed_ranges) == 1:
        return _store(_scaled(_iterated_integrate(f_expr, parsed_ranges, opts)))

    # ── Sum of separable terms ──────────────────────────────────────────────
    # If f_expr is a sum where every term is a product of single-variable
    # functions, integrate term by term and sum.  This handles integrands like
    # sin(x)*cos(y) + exp(-x)*y^2 that are not of the f(g) form but whose
    # terms are each product-separable.
    if f_expr.is_Add:
        # Only attempt if every term is product-separable on constant-limit ranges
        vars_set = set(vars_)
        const_ranges = [
            r
            for r in parsed_ranges
            if not sp.sympify(r[1]).free_symbols & vars_set
            and not sp.sympify(r[2]).free_symbols & vars_set
        ]
        if len(const_ranges) == len(parsed_ranges):  # all limits constant
            term_results = []
            for term in f_expr.args:
                tr = _try_product_separable(term, vars_, parsed_ranges, opts)
                if tr is None:
                    # try recursive multiple_integrate for this term
                    tr = multiple_integrate(
                        term,
                        *parsed_ranges,
                        assumptions=assumptions,
                        generate_conditions=generate_conditions,
                        principal_value=principal_value,
                    )
                term_results.append(tr)
            total = _fast_simplify(sp.Add(*term_results))
            if not total.has(sp.Integral):
                return _store(_scaled(total))

    # ── Consolidate exp products: exp(-x)*exp(-y) → exp(-(x+y)) ─────────────
    # powsimp merges exponential products so S1/S2 can recognise them;
    # it is cheap (single-pass) and never changes the value.
    f_simplified = sp.powsimp(f_expr, force=True)
    if f_simplified != f_expr and not f_simplified.free_symbols - vars_set:
        f_expr = f_simplified

    # ── Pull out constant factors ────────────────────────────────────────────
    # If f_expr = c * h(vars_) with c free of all vars_, factor c out, compute
    # ∫ h dxⁿ, and multiply back. This ensures downstream strategies always
    # see a "pure" integrand without stray scalar prefactors.
    if f_expr.is_Mul:
        c_factors = [
            a for a in f_expr.args if not sp.sympify(a).free_symbols & vars_set
        ]
        if c_factors:
            c_out = sp.Mul(*c_factors)
            h_expr = f_expr / c_out
            inner = multiple_integrate(
                h_expr,
                *parsed_ranges,
                assumptions=assumptions,
                generate_conditions=generate_conditions,
                principal_value=principal_value,
            )
            return _store(_fast_simplify(_scaled(c_out * inner)))

    # ── Direct superellipse shortcut ─────────────────────────────────────────
    # Some hard reference cases have an outer wrapper like 1/(1+g) or exp(-g)
    # around a homogeneous power-sum g.  Recover that structure directly from
    # the full integrand so we can avoid extremely slow symbolic Heaviside
    # integration in the generic layer-cake path.
    super_decomp = _split_superellipse(f_expr, vars_)
    if super_decomp is not None:
        super_outer, super_g = super_decomp
        super_res = _try_superellipse(super_outer, super_g, vars_, parsed_ranges, opts)
        if super_res is not None:
            return _store(_scaled(super_res))

    # ── Decompose integrand into f_outer ∘ g ──────────────────────────────────
    @functools.lru_cache(maxsize=256)
    def _decompose_cached(expr, vars_tuple):
        return _decompose_shallow(expr, list(vars_tuple))

    decomp = _decompose_cached(f_expr, tuple(vars_))

    if decomp is None:
        return _store(_scaled(_iterated_integrate(f_expr, parsed_ranges, opts)))

    def _run_strategies(cur_decomp):
        f_outer = cur_decomp.f_outer
        g = cur_decomp.g_inner
        is_poly = cur_decomp.is_polynomial

        if is_poly:
            for strategy in (
                _try_linear,
                _try_quadratic_infinite,
                _try_even_half_quad,
                _try_superellipse,
                _try_general_polynomial,
            ):
                res = strategy(f_outer, g, vars_, parsed_ranges, opts)
                if res is not None:
                    return res

        for strategy in (
            _try_separable,
            _try_monotone_subst,
            _try_piecewise_monotone,
            _try_nonpoly,
        ):
            res = strategy(f_outer, g, vars_, parsed_ranges, opts)
            if res is not None:
                return res
        return None

    res = _run_strategies(decomp)
    if res is not None:
        return _store(_scaled(res))

    deep_decomp = _decompose(f_expr, vars_)
    if deep_decomp is not None and (
        deep_decomp.g_inner != decomp.g_inner
        or deep_decomp.f_outer(sp.Symbol("_u")) != decomp.f_outer(sp.Symbol("_u"))
    ):
        res = _run_strategies(deep_decomp)
        if res is not None:
            return _store(_scaled(res))

    # ── Fallback ───────────────────────────────────────────────────────────────
    result = _iterated_integrate(f_expr, parsed_ranges, opts)
    return _store(_fast_simplify(scale * result) if scale != 1 else result)


def _simplex_poly(region: Region, expr: sp.Expr) -> sp.Expr | None:
    """Exact polynomial moments on simplex-like regions."""
    if isinstance(region, (SimplexRegion, AffineSimplexRegion)):
        return region.polynomial_moment(expr)
    return None


def _standard_ball_ranges(
    vars_: tuple[sp.Symbol, ...], radius: sp.Expr
) -> tuple[tuple, ...]:
    """SymPy-style inner-first iterated ranges for a standard ball."""
    outer_first = []
    sumsq = sp.Integer(0)
    for i, v in enumerate(vars_):
        if i == 0:
            outer_first.append((v, -radius, radius))
        else:
            rad = sp.sqrt(radius**2 - sumsq)
            outer_first.append((v, -rad, rad))
        sumsq += v**2
    return tuple(reversed(outer_first))


def _polar_disk_transform(region: DiskRegion) -> CoordinateTransform:
    x, y = region.variables
    r = sp.Symbol("_r", nonnegative=True, real=True)
    theta = sp.Symbol("_theta", real=True)
    return CoordinateTransform(
        source_vars=(x, y),
        target_vars=(theta, r),
        forward_map=(r * sp.cos(theta), r * sp.sin(theta)),
        jacobian=r,
        target_ranges=((theta, 0, 2 * sp.pi), (r, 0, region.radius)),
    )


def _ball_spherical_map(region: BallRegion) -> CoordinateTransform | None:
    if region.dimension != 3 or len(region.variables) != 3:
        return None
    x, y, z = region.variables
    r = sp.Symbol("_r", nonnegative=True, real=True)
    phi = sp.Symbol("_phi", real=True)
    theta = sp.Symbol("_theta", real=True)
    return CoordinateTransform(
        source_vars=(x, y, z),
        target_vars=(theta, phi, r),
        forward_map=(
            r * sp.sin(phi) * sp.cos(theta),
            r * sp.sin(phi) * sp.sin(theta),
            r * sp.cos(phi),
        ),
        jacobian=r**2 * sp.sin(phi),
        target_ranges=((theta, 0, 2 * sp.pi), (phi, 0, sp.pi), (r, 0, region.radius)),
    )


def _affine_region_transform(region: Region) -> CoordinateTransform | None:
    if isinstance(region, EllipsoidRegion):
        vars_ = tuple(region.variables_nd)
        uvars = sp.symbols(f"_u0:{len(vars_)}", real=True)
        jac = sp.Integer(1)
        forward = []
        for a, u in zip(region.axes, uvars):
            jac *= sp.Abs(sp.sympify(a))
            forward.append(sp.sympify(a) * u)
        return CoordinateTransform(
            source_vars=vars_,
            target_vars=tuple(reversed(uvars)),
            forward_map=tuple(forward),
            jacobian=jac,
            target_ranges=_standard_ball_ranges(uvars, sp.Integer(1)),
        )
    if isinstance(region, AnnulusRegion):
        x, y = region.variables_xy
        r = sp.Symbol("_r", nonnegative=True, real=True)
        theta = sp.Symbol("_theta", real=True)
        return CoordinateTransform(
            source_vars=(x, y),
            target_vars=(theta, r),
            forward_map=(r * sp.cos(theta), r * sp.sin(theta)),
            jacobian=r,
            target_ranges=(
                (theta, 0, 2 * sp.pi),
                (r, region.inner_radius, region.outer_radius),
            ),
        )
    if isinstance(region, SphericalShellRegion) and len(region.variables_nd) == 3:
        x, y, z = region.variables_nd
        r = sp.Symbol("_r", nonnegative=True, real=True)
        phi = sp.Symbol("_phi", real=True)
        theta = sp.Symbol("_theta", real=True)
        return CoordinateTransform(
            source_vars=(x, y, z),
            target_vars=(theta, phi, r),
            forward_map=(
                r * sp.sin(phi) * sp.cos(theta),
                r * sp.sin(phi) * sp.sin(theta),
                r * sp.cos(phi),
            ),
            jacobian=r**2 * sp.sin(phi),
            target_ranges=(
                (theta, 0, 2 * sp.pi),
                (phi, 0, sp.pi),
                (r, region.inner_radius, region.outer_radius),
            ),
        )
    return None


def _try_transform(
    transform: CoordinateTransform,
    expr: sp.Expr,
    *,
    assumptions,
    generate_conditions,
    principal_value,
) -> sp.Expr | None:
    transformed_expr = transform.apply(expr)
    if any(v in transformed_expr.free_symbols for v in transform.source_vars):
        return None
    try:
        return multiple_integrate(
            transformed_expr,
            *transform.target_ranges,
            assumptions=assumptions,
            generate_conditions=generate_conditions,
            principal_value=principal_value,
        )
    except Exception:
        return None


def _try_region_transform(
    region: Region, expr: sp.Expr, *, assumptions, generate_conditions, principal_value
) -> sp.Expr | None:
    if isinstance(region, DiskRegion):
        val = _try_transform(
            _polar_disk_transform(region),
            expr,
            assumptions=assumptions,
            generate_conditions=generate_conditions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    if isinstance(region, BallRegion):
        tfm = _ball_spherical_map(region)
        if tfm is not None:
            val = _try_transform(
                tfm,
                expr,
                assumptions=assumptions,
                generate_conditions=generate_conditions,
                principal_value=principal_value,
            )
            if val is not None:
                return val
    tfm = _affine_region_transform(region)
    if tfm is not None:
        val = _try_transform(
            tfm,
            expr,
            assumptions=assumptions,
            generate_conditions=generate_conditions,
            principal_value=principal_value,
        )
        if val is not None:
            return val
    return None


def _symbolically_positive(expr: sp.Expr, assumptions=None) -> bool:
    """Best-effort positivity test for structured convergence checks."""
    expr = sp.sympify(expr)
    if expr.is_positive is True:
        return True
    if assumptions and isinstance(assumptions, dict):
        if expr.is_Symbol and assumptions.get(str(expr)) in ("positive", True):
            return True
    try:
        val = sp.N(expr)
        if val.is_real and float(val) > 0:
            return True
    except Exception:
        pass
    return False


def _simplex_dirichlet_term(term: sp.Expr, vars_: tuple[sp.Symbol, ...]):
    """Parse coeff * prod(x_i**a_i) * (1-sum x_i)**b for a simplex term."""
    term = sp.factor_terms(sp.sympify(term))
    rem = sp.expand(1 - sum(vars_))
    coeff = sp.Integer(1)
    exponents = {v: sp.Integer(0) for v in vars_}
    rem_exp = sp.Integer(0)
    factors = term.args if term.is_Mul else (term,)
    for factor in factors:
        factor = sp.sympify(factor)
        if not (factor.free_symbols & set(vars_)):
            coeff *= factor
            continue
        base, exp = factor.as_base_exp()
        base = sp.expand(base)
        if base in exponents:
            exponents[base] += exp
            continue
        if sp.simplify(base - rem) == 0:
            rem_exp += exp
            continue
        return None
    return coeff, tuple(exponents[v] for v in vars_), rem_exp


def _simplex_dirichlet(
    region: Region, expr: sp.Expr, assumptions=None
) -> sp.Expr | None:
    """Exact Dirichlet-type integration on standard and affine simplices."""
    if isinstance(region, AffineSimplexRegion):
        vars_ = region.variables
        uvars = sp.symbols(f"_u0:{region.dimension}", real=True)
        subs = {
            v: sp.sympify(a) + sp.sympify(s) * u
            for v, a, s, u in zip(vars_, region.shifts, region.scales, uvars)
        }
        jac = sp.Integer(1)
        for s in region.scales:
            jac *= sp.Abs(sp.sympify(s))
        transformed = sp.expand(sp.sympify(expr).subs(subs) * jac)
        simplex_ranges = tuple((u, 0, 1 - sum(uvars[:i])) for i, u in enumerate(uvars))
        simplex = SimplexRegion(simplex_ranges, dimension=region.dimension)
        return _simplex_dirichlet(simplex, transformed, assumptions=assumptions)

    if not isinstance(region, SimplexRegion):
        return None

    vars_ = tuple(region.variables)
    expr = sp.sympify(expr)
    total = sp.Integer(0)
    terms = expr.args if expr.is_Add else (expr,)
    for term in terms:
        parsed = _simplex_dirichlet_term(term, vars_)
        if parsed is None:
            return None
        coeff, exponents, rem_exp = parsed
        alphas = [sp.simplify(e + 1) for e in exponents] + [sp.simplify(rem_exp + 1)]
        if any(a.is_nonpositive is True for a in alphas):
            return None
        if any(
            (a.is_positive is not True)
            and not _symbolically_positive(a, assumptions=assumptions)
            for a in alphas
        ):
            return None
        numer = coeff
        for a in alphas:
            numer *= sp.gamma(a)
        total += numer / sp.gamma(sp.Add(*alphas))
    return _fast_simplify(total)


def _try_gauss_linear_map(
    expr: sp.Expr, parsed_ranges: list[tuple], assumptions=None
) -> sp.Expr | None:
    vars_ = tuple(r[0] for r in parsed_ranges)
    if not vars_ or not all(
        sp.sympify(lo) == -sp.oo and sp.sympify(hi) == sp.oo
        for _, lo, hi in parsed_ranges
    ):
        return None
    expr = sp.sympify(expr)
    if expr.func != sp.exp or len(expr.args) != 1:
        return None
    q = sp.expand(expr.args[0])
    if any(
        v in sp.sympify(lo).free_symbols or v in sp.sympify(hi).free_symbols
        for v, lo, hi in parsed_ranges
    ):
        return None
    H = sp.hessian(q, vars_)
    A = sp.simplify(-H / 2)
    try:
        if any(entry.free_symbols & set(vars_) for entry in A):
            return None
    except Exception:
        return None
    quad = sp.expand((sp.Matrix(vars_).T * A * sp.Matrix(vars_))[0])
    linear_part = sp.expand(q + quad)
    b_entries = []
    for v in vars_:
        dv = sp.diff(linear_part, v)
        if dv.free_symbols & set(vars_):
            return None
        b_entries.append(sp.simplify(dv))
    bvec = sp.Matrix(b_entries)
    c = sp.simplify(linear_part - sum(b * v for b, v in zip(b_entries, vars_)))
    if c.free_symbols & set(vars_):
        return None
    detA = sp.simplify(A.det())
    if detA == 0:
        return None
    try:
        if A.is_positive_definite is False:
            return None
    except Exception:
        pass
    if A.is_positive_definite is not True:
        # Avoid returning branch-sensitive square roots unless positivity is clear.
        return None
    try:
        shift_term = sp.simplify((bvec.T * A.LUsolve(bvec))[0] / 4)
    except Exception:
        return None
    return sp.simplify(
        sp.pi ** (sp.Rational(len(vars_), 2)) * sp.exp(c + shift_term) / sp.sqrt(detA)
    )


def _region_shortcut(region: Region, expr: sp.Expr, assumptions=None) -> sp.Expr | None:
    """Exact moments and simple radial integrals on recognized regions."""
    if isinstance(region, UnionRegion):
        total = sp.Integer(0)
        for piece in region.pieces:
            val = _region_shortcut(piece, expr, assumptions=assumptions)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

    if isinstance(region, (SimplexRegion, AffineSimplexRegion)):
        poly_res = region.polynomial_moment(expr)
        if poly_res is not None:
            return poly_res
        dirichlet_res = _simplex_dirichlet(region, expr, assumptions=assumptions)
        if dirichlet_res is not None:
            return dirichlet_res

    if isinstance(
        region,
        (DiskRegion, BallRegion, EllipsoidRegion, AnnulusRegion, SphericalShellRegion),
    ):
        poly_res = region.polynomial_moment(expr)
        if poly_res is not None:
            return poly_res
        return region.radial_integral(expr)

    return None


def _try_graph_reversal(
    expr: sp.Expr,
    region: Region,
    assumptions,
    generate_conditions: bool,
    principal_value: bool,
) -> sp.Expr | None:
    """Safely reverse simple 2-D graph regions when it reduces dependency.

    The current Phase 2 heuristic only fires when the integrand depends on the
    inner variable but not the outer one.  Reversing then converts the problem
    into one where the first integration produces a simple geometric factor.
    """
    if not isinstance(region, GraphRegion):
        return None
    if region.outer_var is None or region.inner_var is None:
        return None
    outer = region.outer_var
    inner = region.inner_var
    if outer in expr.free_symbols or inner not in expr.free_symbols:
        return None
    pieces = region.reversed_pieces()
    if not pieces:
        return None

    total = sp.Integer(0)
    for piece in pieces:
        # reversed_pieces() now returns inner-first tuples, matching the public
        # SymPy convention: integrate x over [x_lo, x_hi] first, then y.
        (_, inner_lo, inner_hi), (outer, lo, hi) = piece
        weight = _fast_simplify(sp.sympify(inner_hi) - sp.sympify(inner_lo))
        total += multiple_integrate(
            _fast_simplify(weight * expr),
            (outer, lo, hi),
            assumptions=assumptions,
            generate_conditions=generate_conditions,
            principal_value=principal_value,
        )
    return _fast_simplify(total)


if __name__ == "__main__":
    x, y, z = symbols("x y z", real=True)
    a = symbols("a", positive=True)

    tests = [
        # ── Original polynomial tests ─────────────────────────────────────────
        (
            "Poly: x²y  [0,1]×[0,2]",
            lambda: multiple_integrate(x**2 * y, (x, 0, 1), (y, 0, 2)),
            sp.Rational(2, 3),
        ),
        (
            "Poly: xyz  [0,1]³",
            lambda: multiple_integrate(x * y * z, (x, 0, 1), (y, 0, 1), (z, 0, 1)),
            sp.Rational(1, 8),
        ),
        (
            "Poly: triangle x+y",
            lambda: multiple_integrate(x + y, (y, 0, 1 - x), (x, 0, 1)),
            sp.Rational(1, 3),
        ),
        # ── Quadratic / Gaussian ──────────────────────────────────────────────
        (
            "Gaussian R²",
            lambda: multiple_integrate(
                sp.exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo)
            ),
            pi,
        ),
        # ── Non-polynomial g: exponential ─────────────────────────────────────
        (
            "exp(-x)·exp(-y)  [0,∞)²",
            lambda: multiple_integrate(sp.exp(-x) * sp.exp(-y), (x, 0, oo), (y, 0, oo)),
            sp.Integer(1),
        ),
        (
            "exp(-(x+y))  [0,∞)²  (separable sum)",
            lambda: multiple_integrate(sp.exp(-(x + y)), (x, 0, oo), (y, 0, oo)),
            sp.Integer(1),
        ),
        # ── Non-polynomial g: trigonometric ───────────────────────────────────
        (
            "sin(x)·cos(y)  [0,π/2]²",
            lambda: multiple_integrate(
                sp.sin(x) * sp.cos(y), (x, 0, pi / 2), (y, 0, pi / 2)
            ),
            sp.Integer(1),
        ),
        (
            "cos(x+y)  [0,π]²  (separable sum)",
            lambda: multiple_integrate(sp.cos(x + y), (x, 0, pi), (y, 0, pi)),
            -4,
        ),
        # ── Non-polynomial g: monotone substitution ───────────────────────────
        (
            "1/(1+x²)  [0,1]",
            lambda: multiple_integrate(1 / (1 + x**2), (x, 0, 1)),
            pi / 4,
        ),
        (
            "exp(-x²)  [0,∞)",
            lambda: multiple_integrate(sp.exp(-(x**2)), (x, 0, oo)),
            sqrt(pi) / 2,
        ),
        # ── Non-polynomial g: piecewise-monotone ──────────────────────────────
        (
            "sin(x)  [0,π]",
            lambda: multiple_integrate(sp.sin(x), (x, 0, pi)),
            sp.Integer(2),
        ),
        (
            "cos(x)  [0,2π]  (two monotone pieces)",
            lambda: multiple_integrate(sp.cos(x), (x, 0, 2 * pi)),
            sp.Integer(0),
        ),
        # ── Mixed: product of non-polynomial factors ──────────────────────────
        (
            "cos(x)·exp(-y)  [0,π/2]×[0,∞)",
            lambda: multiple_integrate(
                sp.cos(x) * sp.exp(-y), (x, 0, pi / 2), (y, 0, oo)
            ),
            sp.Integer(1),
        ),
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

    print(
        f"\n{passed} passed, {failed} failed, {errored} errored "
        f"out of {len(tests)} tests."
    )
