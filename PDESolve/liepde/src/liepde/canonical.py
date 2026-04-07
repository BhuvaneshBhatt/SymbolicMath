from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import sympy as sp

from .geometry import CharacteristicCoordinatesResult


def _first_nonzero_linear_coefficient(expr: sp.Expr, vars: Sequence[sp.Symbol]):
    for v in vars:
        c = sp.expand(sp.diff(expr, v))
        if sp.simplify(c) != 0:
            return c
    return None


def _is_affine_linear_in(expr: sp.Expr, vars: Sequence[sp.Symbol]) -> bool:
    try:
        poly = sp.Poly(sp.expand(expr), *vars)
    except Exception:
        return False
    return poly.total_degree() <= 1


def canonicalize_coordinate_expression(expr: sp.Expr, vars: Sequence[sp.Symbol]) -> sp.Expr:
    """Best-effort canonicalization for coordinates/invariants.

    Rules:
    - simplify / factor terms
    - for affine-linear expressions, normalize the first nonzero linear coefficient to 1
      and remove additive constants
    - for pure logs/products/powers, apply conservative simplification
    """
    expr = sp.expand_log(sp.logcombine(sp.expand(expr), force=True), force=True)
    expr = sp.factor_terms(sp.expand(expr))
    expr = sp.simplify(expr)

    if _is_affine_linear_in(expr, vars):
        c = _first_nonzero_linear_coefficient(expr, vars)
        if c is not None and sp.simplify(c) != 0:
            expr = sp.expand(expr / c)
        const = sp.expand(expr.subs({v: 0 for v in vars}))
        expr = sp.expand(expr - const)
        expr = sp.simplify(expr)

    return expr


def canonicalize_coordinate_chart(chart: CharacteristicCoordinatesResult, vars: Sequence[sp.Symbol]) -> CharacteristicCoordinatesResult:
    invariants = tuple(canonicalize_coordinate_expression(e, vars) for e in chart.invariants)
    transverse = tuple(canonicalize_coordinate_expression(e, vars) for e in chart.transverse)
    validity = tuple(dict.fromkeys(sp.simplify(v) for v in chart.validity_conditions))
    coordinates = invariants + transverse
    if len(coordinates) != len(vars):
        raise ValueError(
            f"Invalid coordinate chart: expected {len(vars)} total coordinates for variables {tuple(vars)}, "
            f"got {len(coordinates)} (invariants={len(invariants)}, transverse={len(transverse)})."
        )
    jacobian_matrix = sp.Matrix([[sp.diff(c, v) for v in vars] for c in coordinates])
    jacobian = sp.simplify(jacobian_matrix.det())
    return replace(chart, invariants=invariants, transverse=transverse, jacobian=jacobian, validity_conditions=validity)


def _strip_scalar_prefactor(expr: sp.Expr) -> sp.Expr:
    expr = sp.factor_terms(sp.expand(expr))
    if isinstance(expr, sp.Mul):
        kept = []
        for fac in expr.args:
            if fac.free_symbols:
                kept.append(fac)
            else:
                                # Remove scalar prefactors so equivalent differential monomials share the same canonical form.
                if sp.simplify(fac) == 0:
                    return sp.Integer(0)
        if kept:
            return sp.expand(sp.Mul(*kept))
    return sp.expand(expr)




def _scalar_factor_ignoring_sign(expr: sp.Expr) -> sp.Expr:
    expr = sp.factor_terms(sp.expand(expr))
    if isinstance(expr, sp.Mul):
        facs = []
        for fac in expr.args:
            if fac.free_symbols:
                facs.append(fac)
            else:
                                # Treat a global sign separately, but keep meaningful scalar factors attached to the monomial.
                if sp.simplify(fac + 1) == 0:
                    continue
                facs.append(fac)
        return sp.expand(sp.Mul(*facs)) if facs else sp.Integer(1)
    if expr.free_symbols:
        return expr
    return sp.Integer(1)


def _normalize_by_highest_derivative_coefficient(expr: sp.Expr) -> sp.Expr:
    derivs = []
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            try:
                order = sum(c for _, c in node.variable_count)
            except Exception:
                order = len(node.variables)
            derivs.append((order, node))
    if not derivs:
        return expr
    derivs.sort(key=lambda t: (-t[0], sp.srepr(t[1])))
    coeff = sp.expand(sp.diff(expr, derivs[0][1]))
    factor = _scalar_factor_ignoring_sign(coeff)
    if factor != 0 and factor != 1 and all(getattr(n, 'is_Function', False) is False and not isinstance(n, sp.Derivative) for n in sp.preorder_traversal(factor)):
        return sp.expand(expr / factor)
    return expr

def canonicalize_reduced_equation(eq: sp.Equality) -> sp.Equality:
    lhs = sp.expand(eq.lhs - eq.rhs)
    lhs = sp.together(lhs).as_numer_denom()[0]
    lhs = _normalize_by_highest_derivative_coefficient(lhs)
    lhs = sp.factor_terms(lhs)
    lhs = _strip_scalar_prefactor(lhs)
    lhs = sp.expand(lhs)
    lhs = sp.simplify(lhs)
    return sp.Eq(lhs, 0)
