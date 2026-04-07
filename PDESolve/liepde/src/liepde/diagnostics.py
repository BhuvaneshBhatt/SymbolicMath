from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .geometry import DistributionKD, distribution_closure
from .utils import expr_complexity, matrix_rank_symbolic


@dataclass
class FailureDiagnostic:
    stage: str
    reason: str
    details: tuple[str, ...] = ()
    conditions: tuple[sp.Expr, ...] = ()


@dataclass
class ChartCandidateDiagnostic:
    method: str
    invariants: tuple[sp.Expr, ...]
    transverse: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    validity_conditions: tuple[sp.Expr, ...]
    score: tuple


@dataclass
class DistributionExplainabilityReport:
    rank: int | None
    commuting: bool
    affine: bool
    closure_closed: bool
    reasons: tuple[str, ...]
    conditions: tuple[sp.Expr, ...]


def local_chart_conditions_from_coords(vars: Sequence[sp.Symbol], coords: Sequence[sp.Expr]) -> tuple[sp.Expr, ...]:
    if len(coords) != len(vars):
        return tuple()
    J = sp.Matrix([[sp.diff(c, v) for v in vars] for c in coords])
    det = sp.factor(sp.simplify(J.det()))
    conds: list[sp.Expr] = []
    if det != 0:
        num, den = sp.fraction(sp.together(det))
        if den != 1:
            conds.append(sp.Ne(den, 0))
        conds.append(sp.Ne(num, 0))
    return tuple(conds)


def score_chart_candidate(method: str, invariants: Sequence[sp.Expr], transverse: Sequence[sp.Expr], jacobian: sp.Expr, validity_conditions: Sequence[sp.Expr]) -> tuple:
    coords = tuple(invariants) + tuple(transverse)
    cx = sum(expr_complexity(c) for c in coords)
    jac_cx = expr_complexity(jacobian)
    cond_penalty = sum(expr_complexity(c) for c in validity_conditions)
    method_priority = {
        'translation_subalgebra': 0,
        'diagonal_scaling_subalgebra': 1,
        'commuting_affine_constant_derivative_coords': 2,
        'commuting_affine_flow_cross_section': 3,
        'involutive_affine_rectified_coords': 4,
        'involutive_affine_constant_derivative_coords': 5,
        'involutive_affine_full_rank_identity_chart': 6,
    }.get(method, 20)
    return (method_priority, cx + jac_cx + cond_penalty, len(validity_conditions), len(transverse))


def explain_distribution(distribution: DistributionKD) -> DistributionExplainabilityReport:
    diag = distribution.diagnostics()
    closure = distribution_closure(distribution)
    reasons: list[str] = []
    conditions: list[sp.Expr] = []
    if diag.rank is None:
        reasons.append('symbolic_rank_unknown')
    else:
        reasons.append(f'generic_rank_{diag.rank}')
    if diag.minor_columns is not None:
        A = distribution.coefficient_matrix()
        cols = diag.minor_columns
        rows = tuple(range(min(distribution.size, len(cols))))
        try:
            if distribution.size <= distribution.dimension and len(cols) == distribution.size:
                det = sp.simplify(A[:, cols].det())
                if det != 0:
                    conditions.append(sp.Ne(det, 0))
                    reasons.append('nonvanishing_minor_selected')
        except Exception:
            pass
    reasons.append('commuting' if diag.commuting else 'noncommuting')
    reasons.append('affine' if diag.affine else 'nonaffine')
    reasons.append('involutive' if closure.closed else 'noninvolutive')
    if not closure.closed and closure.residuals:
        reasons.append(f'closure_residuals_{len(closure.residuals)}')
    return DistributionExplainabilityReport(
        rank=diag.rank,
        commuting=diag.commuting,
        affine=diag.affine,
        closure_closed=closure.closed,
        reasons=tuple(reasons),
        conditions=tuple(dict.fromkeys(conditions)),
    )


def failure(stage: str, reason: str, *details: str, conditions: Sequence[sp.Expr] = ()) -> FailureDiagnostic:
    return FailureDiagnostic(stage=stage, reason=reason, details=tuple(details), conditions=tuple(conditions))
