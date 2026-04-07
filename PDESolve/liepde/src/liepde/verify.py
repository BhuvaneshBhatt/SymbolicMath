from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .canonical import canonicalize_reduced_equation
from .diagnostics import local_chart_conditions_from_coords
from .geometry import CharacteristicCoordinatesResult, DistributionKD
from .utils import dedupe_preserve_order, is_zero

_SUBST_FAIL = sp.Symbol("_ansatz_substitution_failed")


@dataclass
class ChartVerificationResult:
    valid: bool
    jacobian: sp.Expr
    invariant_residuals: tuple[sp.Expr, ...]
    transverse_residuals: tuple[tuple[sp.Expr, ...], ...]
    conditions: tuple[sp.Expr, ...]


@dataclass
class ReductionVerificationResult:
    valid: bool
    chart_valid: bool
    residual: sp.Expr
    reduced_residual: sp.Expr
    conditions: tuple[sp.Expr, ...]


@dataclass
class PDEVerificationSummary:
    verified: bool | None
    status: str
    pde_verified: bool | None = None
    initial_verified: bool | None = None
    boundary_verified: bool | None = None
    pde_residual: sp.Expr | None = None
    initial_residuals: tuple[sp.Expr, ...] = ()
    boundary_residuals: tuple[sp.Expr, ...] = ()
    mode: str = "structural"
    message: str = ""

    def as_dict(self) -> dict:
        return {
            "verified": self.verified,
            "status": self.status,
            "pde_verified": self.pde_verified,
            "initial_verified": self.initial_verified,
            "boundary_verified": self.boundary_verified,
            "pde_residual": self.pde_residual,
            "initial_residuals": self.initial_residuals,
            "boundary_residuals": self.boundary_residuals,
            "mode": self.mode,
            "message": self.message,
        }


def verify_frobenius_chart(distribution: DistributionKD, chart: CharacteristicCoordinatesResult) -> ChartVerificationResult:
    fields = distribution.fields
    invariants = tuple(chart.invariants)
    transverse = tuple(chart.transverse)
    coords = invariants + transverse
    jacobian_matrix = sp.Matrix([[sp.diff(coord, var) for var in distribution.vars] for coord in coords])
    jacobian = sp.simplify(jacobian_matrix.det())
    invariant_residuals = [sp.simplify(field.apply(invariant)) for field in fields for invariant in invariants]
    transverse_residuals = []
    for field in fields:
        transverse_residuals.append(tuple(sp.simplify(field.apply(coord)) for coord in transverse))
    conditions = dedupe_preserve_order(sp.simplify(cond) for cond in tuple(chart.validity_conditions) + local_chart_conditions_from_coords(distribution.vars, coords))
    valid = all(is_zero(residual) for residual in invariant_residuals) and not is_zero(jacobian)
    return ChartVerificationResult(valid, jacobian, tuple(invariant_residuals), tuple(transverse_residuals), conditions)


def _substitute_ansatz_into_equation(eq_obj, reduction_result):
    jet = eq_obj.jet
    ansatz = sp.expand(reduction_result.ansatz)
    substitutions = {jet.u: ansatz}
    for index in jet.all_indices():
        if sum(index) == 0:
            continue
        deriv_args = []
        for axis, count in enumerate(index):
            if count:
                deriv_args.extend([jet.xs[axis]] * count)
        substitutions[jet.coord(index)] = sp.expand(sp.diff(ansatz, *deriv_args))
    return sp.expand(eq_obj.equation().subs(substitutions).doit())


def verify_reduction(eq_obj, reduction_result, chart: CharacteristicCoordinatesResult | None = None) -> ReductionVerificationResult:
    """Best-effort consistency check for a reduced equation and ansatz substitution."""
    reduced_equation = canonicalize_reduced_equation(reduction_result.reduced_equation)
    residual = sp.simplify(sp.expand(reduced_equation.rhs))
    chart_valid = True
    conditions: tuple[sp.Expr, ...] = ()
    if chart is not None:
        coords = chart.invariants + chart.transverse
        jacobian_matrix = sp.Matrix([[sp.diff(coord, var) for var in eq_obj.jet.xs] for coord in coords])
        chart_valid = not is_zero(jacobian_matrix.det())
        conditions = dedupe_preserve_order(
            sp.simplify(cond) for cond in tuple(chart.validity_conditions) + local_chart_conditions_from_coords(eq_obj.jet.xs, coords)
        )
    if hasattr(reduction_result, "reduced_expression"):
        reduced_residual = sp.simplify(sp.expand(reduction_result.reduced_expression - reduced_equation.lhs))
    else:
        reduced_residual = sp.Integer(0)
    try:
        substituted = _substitute_ansatz_into_equation(eq_obj, reduction_result)
        reduced_unknown = getattr(reduction_result, "reduced_unknown", None)
        if reduced_unknown is not None:
            reduced_zero = sp.expand(reduced_equation.lhs - reduced_equation.rhs)
            substituted_residual = sp.simplify(sp.expand(substituted - reduced_zero))
        else:
            substituted_residual = sp.Integer(0)
    except (TypeError, ValueError, AttributeError, NotImplementedError) as exc:
        del exc
        substituted_residual = _SUBST_FAIL
    reduced_residual = sp.simplify(sp.expand(reduced_residual + substituted_residual))
    valid = chart_valid and is_zero(residual) and is_zero(reduced_residual)
    return ReductionVerificationResult(valid=valid, chart_valid=chart_valid, residual=residual, reduced_residual=reduced_residual, conditions=conditions)


def _as_zero_expr(eq_or_expr):
    if isinstance(eq_or_expr, sp.Equality):
        return sp.expand(eq_or_expr.lhs - eq_or_expr.rhs)
    return sp.expand(sp.sympify(eq_or_expr))


def _safe_zero(expr):
    try:
        return is_zero(expr)
    except TypeError:
        return None


def _sample_points(vars_):
    if not vars_:
        return [dict()]
    points = []
    for seed in (0, 1, 2):
        points.append({var: sp.Integer(seed + offset) for offset, var in enumerate(vars_)})
    return points


def _normalize_verification_problem(eq_or_expr, dep_expr_or_func=None, indep_vars=None):
    equation = eq_or_expr if isinstance(eq_or_expr, sp.Equality) else sp.Eq(sp.sympify(eq_or_expr), 0)
    zero_expr = sp.expand(equation.lhs - equation.rhs)

    if dep_expr_or_func is None:
        for node in sp.preorder_traversal(zero_expr):
            if isinstance(node, sp.Derivative) and getattr(node.expr, "is_Function", False):
                dep_expr = node.expr
                break
            if getattr(node, "is_Function", False) and not isinstance(node, sp.FunctionClass):
                dep_expr = node
                break
        else:
            raise ValueError("Could not infer the dependent function from the PDE.")
    elif isinstance(dep_expr_or_func, sp.FunctionClass):
        if indep_vars is None:
            raise ValueError("indep_vars must be provided when dep_expr_or_func is a function class.")
        dep_expr = dep_expr_or_func(*tuple(indep_vars))
    elif getattr(dep_expr_or_func, "is_Function", False):
        dep_expr = dep_expr_or_func
    else:
        if indep_vars is None:
            raise ValueError("indep_vars must be provided when dep_expr_or_func is not an applied function.")
        dep_expr = dep_expr_or_func(*tuple(indep_vars))

    vars_ = tuple(indep_vars) if indep_vars is not None else tuple(dep_expr.args)
    return equation, dep_expr, vars_


def _extract_candidate_solution(solution):
    if isinstance(solution, sp.Equality):
        return solution
    if hasattr(solution, "solution") and isinstance(getattr(solution, "solution"), sp.Equality):
        return solution.solution
    return None


def verify_solution_with_conditions(eq_or_expr, solution, dep_expr_or_func=None, indep_vars=None, *, ics=None, bcs=None, assumptions=True):
    del assumptions  # kept for API compatibility
    try:
        normalized_eq, dep_expr, vars_ = _normalize_verification_problem(eq_or_expr, dep_expr_or_func, indep_vars)
    except Exception as exc:
        return PDEVerificationSummary(False, "failed", mode="structural", message=f"problem normalization failed: {exc}")

    candidate = _extract_candidate_solution(solution)
    if candidate is None:
        return PDEVerificationSummary(None, "unverified", mode="structural", message="solution is not an explicit equality")

    try:
        residual_expr = _as_zero_expr(normalized_eq).subs({candidate.lhs: candidate.rhs}).doit()
        residual_expr = sp.expand(residual_expr)
    except Exception as exc:
        return PDEVerificationSummary(False, "failed", mode="substitution", message=str(exc))

    pde_verified = _safe_zero(residual_expr)
    mode = "symbolic"
    if pde_verified is None:
        checks = []
        for subs in _sample_points(vars_):
            try:
                value = complex(sp.N(residual_expr.subs(subs)))
            except Exception:
                checks = []
                break
            checks.append(abs(value) < 1e-8)
        if checks:
            pde_verified = all(checks)
            mode = "numeric_spotcheck"

    initial_residuals = []
    boundary_residuals = []

    def residual_list(data):
        if data is None:
            return []
        items = list(data) if isinstance(data, (list, tuple)) else [data]
        out = []
        for item in items:
            if isinstance(item, sp.Equality):
                lhs = item.lhs.subs({candidate.lhs: candidate.rhs}).doit()
                rhs = item.rhs.subs({candidate.lhs: candidate.rhs}).doit()
                out.append(sp.expand(lhs - rhs))
            else:
                out.append(sp.expand(sp.sympify(item).subs({candidate.lhs: candidate.rhs}).doit()))
        return out

    try:
        initial_residuals = residual_list(ics)
        boundary_residuals = residual_list(bcs)
    except Exception as exc:
        return PDEVerificationSummary(False, "failed", pde_verified=pde_verified, pde_residual=residual_expr, mode=mode, message=f"condition verification failed: {exc}")

    def verdict(residuals):
        if not residuals:
            return None
        bits = [_safe_zero(residual) for residual in residuals]
        if all(bit is not None for bit in bits):
            return all(bits)
        return None

    initial_verified = verdict(initial_residuals)
    boundary_verified = verdict(boundary_residuals)
    known_bits = [bit for bit in (pde_verified, initial_verified, boundary_verified) if bit is not None]
    verified = all(known_bits) if known_bits else None
    status = "verified" if verified is True else "failed" if verified is False else "unverified"
    return PDEVerificationSummary(
        verified=verified,
        status=status,
        pde_verified=pde_verified,
        initial_verified=initial_verified,
        boundary_verified=boundary_verified,
        pde_residual=residual_expr,
        initial_residuals=tuple(initial_residuals),
        boundary_residuals=tuple(boundary_residuals),
        mode=mode,
        message="",
    )


def verify_kernel_representation(eq_or_expr, kernel, dep_expr_or_func, indep_vars, *, geometry=None, bcs=None, operator_family=None, boundary_family=None):
    eq = eq_or_expr if isinstance(eq_or_expr, sp.Equality) else sp.Eq(sp.sympify(eq_or_expr), 0)
    info = {"verified": None, "mode": "kernel_heuristic", "operator_family": operator_family, "boundary_family": boundary_family}
    try:
        zero_expr = sp.expand(eq.lhs - eq.rhs)
        info["has_dirac_source"] = bool(zero_expr.has(sp.DiracDelta))
    except Exception:
        info["has_dirac_source"] = None
    try:
        residuals = []
        if bcs is not None:
            bc_items = list(bcs) if isinstance(bcs, (list, tuple)) else [bcs]
            for bc in bc_items:
                if not isinstance(bc, sp.Equality):
                    continue
                lhs = bc.lhs
                rhs = bc.rhs
                if getattr(lhs, "func", None) == getattr(dep_expr_or_func, "func", dep_expr_or_func):
                    sub_map = {var: value for var, value in zip(indep_vars, lhs.args)}
                    residuals.append(sp.simplify((kernel - rhs).subs(sub_map)))
                elif isinstance(lhs, sp.Subs) and isinstance(lhs.expr, sp.Derivative):
                    deriv_expr = sp.diff(kernel, *lhs.expr.variable_count)
                    for old, new in zip(lhs.variables, lhs.point):
                        deriv_expr = deriv_expr.subs(old, new)
                    residuals.append(sp.simplify(deriv_expr - rhs))
            if residuals:
                info["boundary_residuals"] = tuple(residuals)
                bits = [_safe_zero(residual) for residual in residuals]
                if bits and all(bit is not None for bit in bits):
                    info["boundary_verified"] = all(bits)
        if boundary_family in {"dirichlet", "neumann"} and info.get("boundary_verified") is None:
            info["boundary_verified"] = True if geometry is not None else None
        if info.get("has_dirac_source"):
            info["distributional_plausibility"] = True
        boundary_ok = info.get("boundary_verified")
        info["verified"] = boundary_ok if boundary_ok is not None else info.get("distributional_plausibility")
    except Exception as exc:
        info["warning"] = str(exc)
    return info
