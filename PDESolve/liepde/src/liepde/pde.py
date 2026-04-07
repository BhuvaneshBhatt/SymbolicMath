from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .utils import (
    add_multiindex,
    enumerate_multiindices,
    enumerate_nonzero_multiindices,
    equation_to_zero_expr,
    expr_complexity,
    multiindex_geq,
    multiindex_subtract,
    multiindex_sum,
    poly_zero_equations,
    safe_solve_for,
    weak_compositions,
)


class ScalarJetSpaceKD:
    """
    Jet space for one dependent variable u over k independent variables.
    Coordinates are indexed by multi-indices J.
    """

    def __init__(self, indep_vars: Sequence[sp.Symbol], dep_name: str = "u", max_order: int = 3):
        self.xs = tuple(indep_vars)
        self.k = len(self.xs)
        self.dep_name = dep_name
        self.max_order = max_order
        self._indices_by_order = {n: list(weak_compositions(n, self.k)) for n in range(max_order + 1)}
        self._cache = {}
        for n in range(max_order + 1):
            for J in self._indices_by_order[n]:
                self._cache[J] = sp.Symbol(self._coord_name(J))

    def _coord_name(self, J):
        parts = [self.dep_name]
        for axis, power in enumerate(J):
            if power == 0:
                continue
            name = str(self.xs[axis])
            parts.append(name if power == 1 else f"{name}{power}")
        return "_".join(parts)

    @property
    def u(self):
        return self._cache[(0,) * self.k]

    def coord(self, J):
        return self._cache[tuple(J)]

    def all_indices(self, max_order=None):
        if max_order is None:
            max_order = self.max_order
        out = []
        for n in range(max_order + 1):
            out.extend(self._indices_by_order[n])
        return out

    def total_derivative(self, expr: sp.Expr, axis: int):
        result = sp.diff(expr, self.xs[axis])
        for J in self.all_indices(self.max_order - 1):
            result += sp.diff(expr, self.coord(J)) * self.coord(add_multiindex(J, axis))
        return sp.expand(result)


@dataclass
class PrincipalMultiindexCandidate:
    principal_multiindex: tuple
    derivative_symbol: sp.Symbol
    solved_rhs: sp.Expr | None
    score: int
    details: dict


@dataclass
class ScalarGeneralSolvedPDEKD:
    """
    Scalar PDE solved for a general principal jet u_P = G.
    """
    jet: ScalarJetSpaceKD
    G: sp.Expr
    principal_multiindex: tuple

    def __post_init__(self):
        P = tuple(self.principal_multiindex)
        if len(P) != self.jet.k:
            raise ValueError("principal_multiindex must have length k.")
        if sum(P) == 0:
            raise ValueError("principal_multiindex must be nonzero.")
        self.principal_multiindex = P

    @property
    def principal_order(self):
        return sum(self.principal_multiindex)

    def equation(self):
        return self.jet.coord(self.principal_multiindex) - self.G

    def differential_consequence_substitutions(self, order_needed=None):
        jet = self.jet
        P = self.principal_multiindex
        if order_needed is None:
            order_needed = jet.max_order
        subs = {jet.coord(P): self.G}
        for total_order in range(self.principal_order + 1, order_needed + 1):
            for J in jet._indices_by_order[total_order]:
                if not multiindex_geq(J, P):
                    continue
                expr = self.G
                Jred = multiindex_subtract(J, P)
                for axis, reps in enumerate(Jred):
                    for _ in range(reps):
                        expr = jet.total_derivative(expr, axis)
                subs[jet.coord(J)] = sp.expand(expr)
        return subs



def _jet_symbols_in_expr(jet, expr):
    expr_syms = expr.free_symbols
    out = []
    for J in jet.all_indices():
        sym = jet.coord(J)
        if sym in expr_syms:
            out.append((J, sym))
    return out


def _contains_principal_family_derivative(jet, expr, P):
    for J, _ in _jet_symbols_in_expr(jet, expr):
        if multiindex_geq(J, P):
            return True
    return False


def _count_principal_family_higher_derivatives(jet, expr, P):
    pord = sum(P)
    count = 0
    for J, _ in _jet_symbols_in_expr(jet, expr):
        if multiindex_geq(J, P) and sum(J) > pord:
            count += 1
    return count


def choose_principal_multiindex_scalar_kd(jet: ScalarJetSpaceKD, pde: sp.Expr | sp.Equality, max_principal_order: int = 3):
    zero_expr = equation_to_zero_expr(pde)
    max_order = min(max_principal_order, jet.max_order)
    candidates = [score_principal_multiindex_candidate(jet, zero_expr, P) for P in enumerate_nonzero_multiindices(jet.k, max_order)]
    def keyfun(c):
        rhs_cx = float("inf") if c.solved_rhs is None else expr_complexity(c.solved_rhs)
        return (c.score, -sum(c.principal_multiindex), -rhs_cx, tuple(-v for v in c.principal_multiindex))
    return max(candidates, key=keyfun)


def build_scalar_general_solved_pde_from_equation(jet: ScalarJetSpaceKD, pde: sp.Expr | sp.Equality, principal_multiindex=None, max_principal_order: int = 3):
    zero_expr = equation_to_zero_expr(pde)
    best = choose_principal_multiindex_scalar_kd(jet, zero_expr, max_principal_order) if principal_multiindex is None else score_principal_multiindex_candidate(jet, zero_expr, tuple(principal_multiindex))
    if best.solved_rhs is None:
        raise ValueError(f"Could not solve PDE for principal derivative {best.derivative_symbol}.")
    return ScalarGeneralSolvedPDEKD(jet=jet, G=sp.expand(best.solved_rhs), principal_multiindex=best.principal_multiindex), best


def infer_sympy_pde_order(indep_vars, dep_func, pde) -> int:
    indep_vars = tuple(indep_vars)
    zero_expr = equation_to_zero_expr(pde)
    uapp = dep_func(*indep_vars)
    inferred = 0
    for node in sp.preorder_traversal(zero_expr):
        if isinstance(node, sp.Derivative) and node.expr == uapp:
            inferred = max(inferred, sum(reps for _, reps in node.variable_count))
    return inferred


def build_scalar_jet_equation_from_sympy_pde(indep_vars, dep_func, pde, max_order=None, dep_name=None):
    """
    Build a ScalarJetSpaceKD and jet-space equation from a SymPy PDE involving
    one dependent function dep_func(*indep_vars).
    """
    indep_vars = tuple(indep_vars)
    if dep_name is None:
        dep_name = getattr(dep_func, '__name__', None) or getattr(getattr(dep_func, 'func', None), '__name__', None) or 'u'
    zero_expr = equation_to_zero_expr(pde)

    # Infer the jet order from the highest total derivative order in the PDE.
    # SymPy stores repeated derivatives as variable/count pairs, so the total order
    # is the sum of repetition counts rather than the number of distinct variables.
    inferred = infer_sympy_pde_order(indep_vars, dep_func, pde)
    if max_order is None:
        max_order = max(1, inferred)
    else:
        max_order = max(int(max_order), int(inferred), 1)

    jet = ScalarJetSpaceKD(indep_vars, dep_name=dep_name, max_order=max_order)
    uapp = dep_func(*indep_vars)

    subs = {uapp: jet.u}
    for node in list(sp.preorder_traversal(zero_expr)):
        if isinstance(node, sp.Derivative) and node.expr == uapp:
            counts = [0] * len(indep_vars)
            ok = True
            for var, reps in node.variable_count:
                if var not in indep_vars:
                    ok = False
                    break
                counts[indep_vars.index(var)] += reps
            if ok:
                subs[node] = jet.coord(tuple(counts))
    jet_expr = sp.expand(zero_expr.xreplace(subs))
    return jet, sp.Eq(jet_expr, 0)


def _principal_family_width(jet, expr, P):
    count = 0
    for J, _ in _jet_symbols_in_expr(jet, expr):
        if multiindex_geq(J, P):
            count += 1
    return count


def score_principal_multiindex_candidate(jet, zero_expr, P):
    """
    Improved score that prefers solved forms with clean substitution families and
    fewer surviving nonprincipal generators.
    """
    target = jet.coord(P)
    details = {
        'principal_present': target in zero_expr.free_symbols,
        'principal_order': sum(P),
        'higher_principal_family_count': _count_principal_family_higher_derivatives(jet, zero_expr, P),
        'principal_family_width': _principal_family_width(jet, zero_expr, P),
        'solvable': False,
        'rhs_contains_principal_family': None,
        'rhs_complexity': None,
        'remaining_generator_count': None,
        'all_solutions': [],
    }
    score = 0
    if details['principal_present']:
        score += 4
    score -= 2 * details['higher_principal_family_count']
    score -= details['principal_family_width']
    score -= (sum(P) - 1)

    sols = safe_solve_for(zero_expr, target)
    details['all_solutions'] = sols
    if not sols:
        return PrincipalMultiindexCandidate(P, target, None, score - 100, details)

    valid, invalid = [], []
    for rhs in sols:
        if _contains_principal_family_derivative(jet, rhs, P):
            invalid.append(rhs)
        else:
            valid.append(rhs)
    details['solvable'] = True

    chosen_pool = valid if valid else invalid
    rhs = min(chosen_pool, key=expr_complexity)
    details['rhs_contains_principal_family'] = (chosen_pool is invalid)
    details['rhs_complexity'] = expr_complexity(rhs)
    details['remaining_generator_count'] = sum(1 for J in jet.all_indices() if multiindex_sum(J) >= 1 and not multiindex_geq(J, P))

    score += 5 if valid else 1
    if chosen_pool is invalid:
        score -= 6
    score -= details['rhs_complexity'] // 12
    score -= details['remaining_generator_count'] // 4

    return PrincipalMultiindexCandidate(P, target, rhs, score, details)
