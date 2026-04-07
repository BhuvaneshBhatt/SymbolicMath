from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence

import sympy as sp
from sympy.core.function import AppliedUndef


def is_zero(expr: sp.Expr) -> bool:
    expr = sp.expand(expr)
    if expr == 0:
        return True
    zero_flag = getattr(expr, "is_zero", None)
    if zero_flag is not None:
        return bool(zero_flag)
    if expr.is_number:
        try:
            return bool(expr.equals(0))
        except TypeError:
            return False
    return False


def is_nonzero(expr: sp.Expr) -> bool:
    return not is_zero(expr)




def dedupe_preserve_order(values: Iterable):
    seen = set()
    out = []
    for value in values:
        key = sp.srepr(value) if isinstance(value, sp.Basic) else value
        if key in seen:
            continue
        seen.add(key)
        out.append(value)
    return tuple(out)


def collect_applied_functions(expr: sp.Expr):
    found = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative) and isinstance(node.expr, AppliedUndef):
            found.append(node.expr)
        elif isinstance(node, AppliedUndef):
            found.append(node)
    return dedupe_preserve_order(found)

def weak_compositions(total: int, k: int):
    if k == 1:
        yield (total,)
        return
    for i in range(total + 1):
        for rest in weak_compositions(total - i, k - 1):
            yield (i,) + rest


def enumerate_multiindices(k: int, max_order: int):
    out = []
    for n in range(max_order + 1):
        out.extend(weak_compositions(n, k))
    return out


def enumerate_nonzero_multiindices(k: int, max_order: int):
    out = []
    for n in range(1, max_order + 1):
        out.extend(weak_compositions(n, k))
    return out


def add_multiindex(J: Sequence[int], axis: int):
    J = list(J)
    J[axis] += 1
    return tuple(J)


def multiindex_geq(J: Sequence[int], P: Sequence[int]) -> bool:
    return all(J[i] >= P[i] for i in range(len(J)))


def multiindex_subtract(J: Sequence[int], P: Sequence[int]):
    if not multiindex_geq(J, P):
        raise ValueError("Cannot subtract a larger multi-index.")
    return tuple(J[i] - P[i] for i in range(len(J)))


def multiindex_sum(J: Sequence[int]) -> int:
    return sum(J)


def first_nonzero_scale(values: Iterable[sp.Expr]) -> sp.Expr | None:
    for value in values:
        value = sp.expand(value)
        if is_nonzero(value):
            return sp.cancel(1 / value)
    return None


def equation_to_zero_expr(pde: sp.Expr | sp.Equality) -> sp.Expr:
    if isinstance(pde, sp.Equality):
        return sp.expand(pde.lhs - pde.rhs)
    return sp.expand(pde)


def expr_complexity(expr: sp.Expr) -> int:
    return int(sp.count_ops(sp.expand(expr), visual=False))


def safe_solve_for(expr: sp.Expr, target: sp.Symbol):
    try:
        sols = sp.solve(sp.Eq(expr, 0), target, dict=False)
    except (NotImplementedError, ValueError, TypeError):
        return []
    if sols is None:
        return []
    if not isinstance(sols, (list, tuple)):
        sols = [sols]
    out = []
    for sol in sols:
        if isinstance(sol, (list, tuple)):
            if len(sol) == 1:
                out.append(sp.expand(sol[0]))
        else:
            out.append(sp.expand(sol))
    return out


def poly_zero_equations(expr: sp.Expr, gens: Sequence[sp.Expr]):
    expr = sp.expand(expr)
    poly = sp.Poly(expr, *gens, domain="EX")
    return [sp.expand(c) for c in poly.coeffs()]


def substitute_free_parameters_zero(solution_tuple: Sequence[sp.Expr]):
    free = sorted(
        list(set().union(*(expr.free_symbols for expr in solution_tuple))),
        key=lambda s: s.name,
    )
    if not free:
        return tuple(sp.expand(v) for v in solution_tuple)
    sub = {s: 0 for s in free}
    return tuple(sp.expand(v.subs(sub)) for v in solution_tuple)


def matrix_is_zero(M: sp.Matrix) -> bool:
    return all(is_zero(v) for v in M)


def matrix_is_diagonal(M: sp.Matrix) -> bool:
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            if i != j and is_nonzero(M[i, j]):
                return False
    return True


def matrix_rank_symbolic(M: sp.Matrix) -> int | None:
    try:
        return int(M.rank())
    except (TypeError, ValueError, NotImplementedError):
        return None


def right_inverse_columns(A: sp.Matrix):
    r, k = A.shape
    cols = []
    for j in range(r):
        ej = sp.zeros(r, 1)
        ej[j, 0] = 1
        sol, params = A.gauss_jordan_solve(ej)
        if params.shape[0] > 0:
            sub = {params[i, 0]: 0 for i in range(params.shape[0])}
            sol = sp.Matrix([sp.expand(sol[i, 0].subs(sub)) for i in range(k)])
        else:
            sol = sp.Matrix([sp.expand(sol[i, 0]) for i in range(k)])
        cols.append(sol)
    return cols


def nullspace_basis_rows(A: sp.Matrix):
    try:
        return A.nullspace()
    except (NotImplementedError, ValueError, TypeError):
        return []


def replace_applied_undefs(expr: sp.Expr, replacement_value: sp.Expr) -> sp.Expr:
    out = expr
    seen = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.AppliedUndef):
            if node not in seen:
                seen.append(node)
    for node in seen:
        out = out.subs(node, replacement_value)
    return sp.simplify(out)


def extract_rhs_from_pde_solution(sol: sp.Expr | sp.Equality) -> sp.Expr:
    if isinstance(sol, sp.Equality):
        return sol.rhs
    return sol


def extract_first_single_argument_undef_arg(expr: sp.Expr) -> sp.Expr | None:
    if isinstance(expr, sp.AppliedUndef) and len(expr.args) == 1:
        return sp.simplify(expr.args[0])
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.AppliedUndef) and len(node.args) == 1:
            return sp.simplify(node.args[0])
    return None


def factor_prefactor_independent_of_symbols(expr: sp.Expr, keep_symbols: Sequence[sp.Expr]) -> sp.Expr:
    expr = sp.factor_terms(sp.expand(expr))
    if isinstance(expr, sp.Mul):
        kept = []
        for fac in expr.args:
            if fac.free_symbols.isdisjoint(set(keep_symbols)):
                continue
            kept.append(fac)
        if kept:
            return sp.expand(sp.Mul(*kept))
    return sp.expand(expr)

