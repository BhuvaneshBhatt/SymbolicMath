from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import sympy as sp

from .geometry import DistributionKD, VectorFieldKD, distribution_closure
from .frobenius import restricted_local_frobenius_engine, adapted_basis_in_chart
from .utils import expr_complexity, is_nonzero, is_zero
from .optimal import optimal_system_1d, optimal_system_commuting_subalgebras


@dataclass
class SubalgebraCandidate:
    indices: tuple[int, ...]
    distribution: DistributionKD
    commuting: bool
    closed: bool
    score: tuple


@dataclass
class LieAlgebraStructureResult:
    bracket_table: tuple
    closure_closed: bool
    structure_functions: tuple | None
    adapted_struct_fns: tuple | None
    derived_dims: tuple[int, ...]
    lower_central_dims: tuple[int, ...]
    solvable_like: bool
    nilpotent_like: bool


def enumerate_subalgebras(distribution: DistributionKD, max_dim: int | None = None) -> tuple[SubalgebraCandidate, ...]:
    n = distribution.size
    if max_dim is None:
        max_dim = n
    out = []
    for r in range(1, min(max_dim, n) + 1):
        for idxs in combinations(range(n), r):
            sub = DistributionKD(distribution.vars, tuple(distribution.fields[i] for i in idxs))
            commuting = sub.is_commuting()
            closure = distribution_closure(sub)
            diag = sub.diagnostics()
            score = (-r, 0 if diag.translation else 1 if diag.diagonal_scaling else 2 if diag.affine else 3)
            out.append(SubalgebraCandidate(idxs, sub, commuting, closure.closed, score))
    out.sort(key=lambda c: c.score)
    return tuple(out)


def choose_reduction_friendly_subalgebras(distribution: DistributionKD, max_dim: int | None = None) -> tuple[SubalgebraCandidate, ...]:
    cands = [c for c in enumerate_subalgebras(distribution, max_dim=max_dim) if c.commuting]
    cands.sort(key=lambda c: c.score)
    return tuple(cands)



def _field_in_span_coeffs(target: VectorFieldKD, span_fields: tuple[VectorFieldKD, ...]):
    if len(span_fields) == 0:
        return None
    A = sp.Matrix([[f.coeffs[j] for f in span_fields] for j in range(target.dimension)])
    b = sp.Matrix(target.coeffs)
    try:
        sol, params = A.gauss_jordan_solve(b)
    except Exception:
        return None
    if params.shape[0] > 0:
        sub = {params[i, 0]: 0 for i in range(params.shape[0])}
        sol = sp.Matrix([sp.expand(sol[i, 0].subs(sub)) for i in range(len(span_fields))])
    else:
        sol = sp.Matrix([sp.expand(sol[i, 0]) for i in range(len(span_fields))])
    recon = A * sol
    resid = [sp.expand(recon[i, 0] - b[i, 0]) for i in range(target.dimension)]
    if any(is_nonzero(v) for v in resid):
        return None
    return tuple(sol)


def lie_bracket_table(distribution: DistributionKD):
    out = []
    for i in range(distribution.size):
        for j in range(i + 1, distribution.size):
            out.append(((i, j), distribution.fields[i].bracket(distribution.fields[j])))
    return tuple(out)


def structure_functions_in_basis(distribution: DistributionKD):
    table = []
    closed = True
    for (i, j), br in lie_bracket_table(distribution):
        coeffs = _field_in_span_coeffs(br, distribution.fields)
        if coeffs is None:
            closed = False
            table.append(((i, j), None))
        else:
            table.append(((i, j), coeffs))
    return closed, tuple(table)


def structure_functions_in_frobenius_chart(distribution: DistributionKD):
    chart = restricted_local_frobenius_engine(distribution)
    adapted = adapted_basis_in_chart(distribution, chart.invariants + chart.transverse)
    adist = DistributionKD(adapted.coordinate_labels, adapted.adapted_fields)
    closed, table = structure_functions_in_basis(adist)
    return chart, adapted, closed, table


def _closure_span(fields: tuple[VectorFieldKD, ...]):
    """Return a heuristic closure by adjoining pairwise brackets until stable.
    Restricted and local: new brackets are added only when not already in span.
    """
    if len(fields) == 0:
        return tuple()
    basis = list(fields)
    changed = True
    while changed:
        changed = False
        current = list(basis)
        for i in range(len(current)):
            for j in range(i + 1, len(current)):
                br = current[i].bracket(current[j]).simplify()
                if all(is_zero(c) for c in br.coeffs):
                    continue
                if _field_in_span_coeffs(br, tuple(basis)) is None:
                    basis.append(br)
                    changed = True
    out = []
    for f in basis:
        if _field_in_span_coeffs(f, tuple(out)) is None:
            out.append(f)
    return tuple(out)


def derived_series(distribution: DistributionKD, max_steps: int = 8) -> tuple[DistributionKD, ...]:
    series = [distribution]
    current = distribution
    for _ in range(max_steps):
        brackets = []
        for i in range(current.size):
            for j in range(i + 1, current.size):
                br = current.fields[i].bracket(current.fields[j]).simplify()
                if any(is_nonzero(c) for c in br.coeffs):
                    brackets.append(br)
        if not brackets:
            series.append(DistributionKD(distribution.vars, tuple()))
            break
        nxt_fields = _closure_span(tuple(brackets))
        nxt = DistributionKD(distribution.vars, nxt_fields)
        series.append(nxt)
        if nxt.size == 0 or nxt.size == current.size:
            break
        current = nxt
    return tuple(series)


def lower_central_series(distribution: DistributionKD, max_steps: int = 8) -> tuple[DistributionKD, ...]:
    series = [distribution]
    current = distribution
    g0 = distribution
    for _ in range(max_steps):
        brackets = []
        for X in g0.fields:
            for Y in current.fields:
                br = X.bracket(Y).simplify()
                if any(is_nonzero(c) for c in br.coeffs):
                    brackets.append(br)
        if not brackets:
            series.append(DistributionKD(distribution.vars, tuple()))
            break
        nxt_fields = _closure_span(tuple(brackets))
        nxt = DistributionKD(distribution.vars, nxt_fields)
        series.append(nxt)
        if nxt.size == 0 or nxt.size == current.size:
            break
        current = nxt
    return tuple(series)


def lie_algebra_structure_summary(distribution: DistributionKD) -> LieAlgebraStructureResult:
    closure = distribution_closure(distribution)
    closed1, table = structure_functions_in_basis(distribution)
    adapted_table = None
    try:
        _, _, _, adapted_table = structure_functions_in_frobenius_chart(distribution)
    except Exception:
        adapted_table = None

    dser = derived_series(distribution)
    lser = lower_central_series(distribution)
    d_dims = tuple(d.size for d in dser)
    l_dims = tuple(d.size for d in lser)
    solvable_like = (len(d_dims) > 0 and d_dims[-1] == 0)
    nilpotent_like = (len(l_dims) > 0 and l_dims[-1] == 0)
    return LieAlgebraStructureResult(
        bracket_table=lie_bracket_table(distribution),
        closure_closed=closure.closed and closed1,
        structure_functions=table,
        adapted_struct_fns=adapted_table,
        derived_dims=d_dims,
        lower_central_dims=l_dims,
        solvable_like=solvable_like,
        nilpotent_like=nilpotent_like,
    )



def _frobenius_chart_simplicity(distribution: DistributionKD) -> int:
    try:
        chart = restricted_local_frobenius_engine(distribution)
    except Exception:
        return 10**6
    complexity = sum(expr_complexity(v) for v in chart.invariants + chart.transverse)
    jac_penalty = 0 if sp.simplify(chart.jacobian) != 0 else 10**5
    cond_penalty = 5 * len(chart.validity_conditions)
    return complexity + jac_penalty + cond_penalty


def choose_frobenius_friendly_subalgebras(distribution: DistributionKD, max_dim: int | None = None) -> tuple[SubalgebraCandidate, ...]:
    candidates = []
    for cand in enumerate_subalgebras(distribution, max_dim=max_dim):
        if not cand.closed:
            continue
        simp = _frobenius_chart_simplicity(cand.distribution)
        diag = cand.distribution.diagnostics()
        family = 0 if diag.translation else 1 if diag.diagonal_scaling else 2 if diag.affine else 3
        score = (-cand.distribution.size, family, simp)
        candidates.append(SubalgebraCandidate(cand.indices, cand.distribution, cand.commuting, cand.closed, score))
    candidates.sort(key=lambda c: c.score)
    return tuple(candidates)



def choose_optimal_system_style_1d(distribution: DistributionKD, include_combinations: bool = True, max_coeff: int = 2):
    return optimal_system_1d(distribution, include_combinations=include_combinations, max_coeff=max_coeff)


def choose_optimal_system_style_subalgebras(distribution: DistributionKD, max_dim: int = 2):
    return optimal_system_commuting_subalgebras(distribution, max_dim=max_dim)
