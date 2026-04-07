from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

import sympy as sp
from .pde import build_scalar_general_solved_pde_from_equation, build_scalar_jet_equation_from_sympy_pde, infer_sympy_pde_order
from .reduction import (
    auto_reduce_best_commuting_subalgebra_scalar_kd,
    auto_reduce_best_symbolic_match_scalar_kd,
    search_symbolic_linear_combinations_for_reduction_scalar_kd,
)
from .symmetry import solve_determining_equations_with_polynomial_ansatz_scalar_general_kd
from .verify import verify_reduction
from .utils import collect_applied_functions
from .workflows import repeated_reduction_workflow_scalar_kd, repeated_reduction_workflow_scalar_kd_frobenius_default


@dataclass(frozen=True)
class LiePDEDiagnostics:
    warnings: tuple[str, ...] = ()
    degree_attempts: tuple[int, ...] = ()


@dataclass(frozen=True)
class LieSymmetrySolveResult:
    equation_object: Any
    principal_info: Any
    symmetry_solution: Any
    basis_vectors: tuple
    matches: tuple
    reduction: Any | None
    verification: Any | None
    reduced_equation: sp.Equality | None
    reduced_solution: Any | None
    full_solution: sp.Equality | None
    direct_solution: sp.Equality | None = None
    workflow: Any | None = None
    diagnostics: LiePDEDiagnostics = field(default_factory=LiePDEDiagnostics)

@dataclass(frozen=True)
class LiePDEFailure:
    reason: str
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True)
class LiePDEAnalysis:
    order: int
    principal_symbol: Any | None
    solved_rhs: Any | None
    indep_vars: tuple
    dependent: Any
    is_linear: bool
    is_homogeneous: bool


def classify_pde(eq_or_expr, dep_expr_or_func, indep_vars, *, max_principal_order: int | None = None):
    dep_expr, dep_func, eq_obj, info = build_equation_object(
        eq_or_expr, dep_expr_or_func, indep_vars, max_principal_order=max_principal_order
    )
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    principal_symbol = getattr(info, "derivative_symbol", None)
    try:
        is_linear = bool(sp.Poly(sp.expand(expr), dep_expr, *[d for d in expr.atoms(sp.Derivative)]).total_degree() <= 1)
    except Exception:
        is_linear = False
    try:
        zero_sub = {dep_expr: 0}
        for d in expr.atoms(sp.Derivative):
            zero_sub[d] = 0
        is_homogeneous = sp.simplify(expr.subs(zero_sub)) == 0
    except Exception:
        is_homogeneous = False
    return LiePDEAnalysis(
        order=infer_sympy_pde_order(indep_vars, dep_func, eq_or_expr),
        principal_symbol=principal_symbol,
        solved_rhs=getattr(info, "solved_rhs", None),
        indep_vars=tuple(indep_vars),
        dependent=dep_expr,
        is_linear=is_linear,
        is_homogeneous=is_homogeneous,
    )



def _normalize_dep(dep_expr_or_func, indep_vars):
    indep_vars = tuple(indep_vars)
    if isinstance(dep_expr_or_func, sp.FunctionClass):
        dep_func = dep_expr_or_func
        return dep_func(*indep_vars), dep_func
    if getattr(dep_expr_or_func, "is_Function", False):
        return dep_expr_or_func, dep_expr_or_func.func
    dep_func = dep_expr_or_func
    return dep_func(*indep_vars), dep_func


def build_equation_object(eq_or_expr, dep_expr_or_func, indep_vars, *, max_principal_order: int | None = None):
    dep_expr, dep_func = _normalize_dep(dep_expr_or_func, indep_vars)
    inferred_order = infer_sympy_pde_order(indep_vars, dep_func, eq_or_expr)
    principal_order_cap = inferred_order if max_principal_order is None else max(int(max_principal_order), int(inferred_order), 1)
    jet, pde = build_scalar_jet_equation_from_sympy_pde(
        indep_vars, dep_func, eq_or_expr, max_order=principal_order_cap, dep_name=getattr(dep_func, "__name__", "u")
    )
    equation_object, principal_info = build_scalar_general_solved_pde_from_equation(
        jet, pde, max_principal_order=principal_order_cap
    )
    return dep_expr, dep_func, equation_object, principal_info


def compute_polynomial_symmetries(eq_obj, *, degree: int = 1):
    return solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
        eq_obj, degree=degree, include_dependent_var=True
    )


def search_reductions_from_symmetries(eq_obj, symmetry_solution, *, max_subset_size: int = 2, prefer_commuting_subalgebra: bool = True):
    basis = tuple(symmetry_solution.basis_vectors())
    matches = tuple(
        search_symbolic_linear_combinations_for_reduction_scalar_kd(
            eq_obj,
            basis,
            max_subset_size=max_subset_size,
            try_translation=True,
            try_diagonal_scaling=True,
            try_affine=True,
            normalize=True,
            rank_results=True,
        )
    )
    reduction = None
    if prefer_commuting_subalgebra:
        reduction = auto_reduce_best_commuting_subalgebra_scalar_kd(
            eq_obj, matches, max_generators=min(eq_obj.jet.k - 1, max_subset_size)
        )
    if reduction is None and matches:
        reduction = auto_reduce_best_symbolic_match_scalar_kd(eq_obj, matches)
    return basis, matches, reduction


def _choose_reduced_unknown(reduced_eq: sp.Equality, dep_expr_or_func=None, reduction=None):
    explicit_unknown = getattr(reduction, "reduced_unknown", None) if reduction is not None else None
    if explicit_unknown is not None:
        return explicit_unknown
    candidates = collect_applied_functions(reduced_eq)
    hint = dep_expr_or_func
    if hint is not None:
        hint_func = hint.func if getattr(hint, "is_Function", False) else hint
        matching = tuple(cand for cand in candidates if cand.func == hint_func)
        if len(matching) == 1:
            return matching[0]
        if len(candidates) == 1:
            return candidates[0]
        raise ValueError("Could not uniquely match the requested reduced unknown function.")
    if len(candidates) == 1:
        return candidates[0]
    raise ValueError("Could not uniquely determine the reduced unknown function.")


def solve_reduced_equation(reduced_eq: sp.Equality | None, dep_expr_or_func=None, reduction=None):
    if reduced_eq is None:
        return None
    dependent = _choose_reduced_unknown(reduced_eq, dep_expr_or_func=dep_expr_or_func, reduction=reduction)
    if len(dependent.args) == 1:
        return sp.dsolve(reduced_eq, dependent)
    return sp.pdsolve(reduced_eq, func=dependent)


def _simplify_zero(expr):
    try:
        return sp.simplify(sp.expand(expr))
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return sp.expand(expr)


def _heat_similarity_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) != 2 or not getattr(dep_expr, "is_Function", False):
        return None
    x, t = indep_vars
    ut = sp.diff(dep_expr, t)
    uxx = sp.diff(dep_expr, x, 2)
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    try:
        coeff_ut = expr.coeff(ut)
        rem = _simplify_zero(expr - coeff_ut * ut)
        coeff_uxx = rem.coeff(uxx)
        tail = _simplify_zero(rem - coeff_uxx * uxx)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if coeff_ut == 0 or tail != 0:
        return None
    diffusivity = _simplify_zero(-coeff_uxx / coeff_ut)
    if diffusivity == 0:
        return None
    if any(diffusivity.has(obj) for obj in (ut, uxx, dep_expr)):
        return None
    c1 = sp.Symbol('C1')
    c2 = sp.Symbol('C2')
    try:
        arg = x / (2 * sp.sqrt(diffusivity * t))
        sol = sp.Eq(dep_expr, c1 + c2 * sp.erf(arg))
        residual = _simplify_zero((sp.diff(sol.rhs, t) - diffusivity * sp.diff(sol.rhs, x, 2)).doit())
        if residual == 0:
            return sol
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    return None



def _wave_dalembert_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) != 2 or not getattr(dep_expr, "is_Function", False):
        return None
    x, t = indep_vars
    utt = sp.diff(dep_expr, t, 2)
    uxx = sp.diff(dep_expr, x, 2)
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    try:
        coeff_utt = expr.coeff(utt)
        rem = _simplify_zero(expr - coeff_utt * utt)
        coeff_uxx = rem.coeff(uxx)
        tail = _simplify_zero(rem - coeff_uxx * uxx)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if coeff_utt == 0 or tail != 0:
        return None
    speed_sq = _simplify_zero(-coeff_uxx / coeff_utt)
    if speed_sq == 0:
        return None
    c = sp.sqrt(speed_sq)
    f = sp.Function('F')
    g = sp.Function('G')
    return sp.Eq(dep_expr, f(x - c*t) + g(x + c*t))


def _advection_diffusion_similarity_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) != 2 or not getattr(dep_expr, "is_Function", False):
        return None
    x, t = indep_vars
    ut = sp.diff(dep_expr, t)
    ux = sp.diff(dep_expr, x)
    uxx = sp.diff(dep_expr, x, 2)
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    try:
        coeff_ut = expr.coeff(ut)
        rem = _simplify_zero(expr - coeff_ut * ut)
        coeff_ux = rem.coeff(ux)
        rem2 = _simplify_zero(rem - coeff_ux * ux)
        coeff_uxx = rem2.coeff(uxx)
        tail = _simplify_zero(rem2 - coeff_uxx * uxx)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if coeff_ut == 0 or tail != 0:
        return None
    adv = _simplify_zero(coeff_ux / coeff_ut)
    diff = _simplify_zero(-coeff_uxx / coeff_ut)
    if diff == 0:
        return None
    c1 = sp.Symbol('C1')
    c2 = sp.Symbol('C2')
    try:
        arg = (x - adv*t) / (2 * sp.sqrt(diff*t))
        sol = sp.Eq(dep_expr, c1 + c2 * sp.erf(arg))
        residual = _simplify_zero((sp.diff(sol.rhs, t) + adv*sp.diff(sol.rhs, x) - diff*sp.diff(sol.rhs, x, 2)).doit())
        if residual == 0:
            return sol
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    return None




def _constant_coefficient_transport_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) < 2 or not getattr(dep_expr, "is_Function", False):
        return None
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    derivs = [sp.diff(dep_expr, v) for v in indep_vars]
    coeffs = []
    rem = expr
    try:
        for d in derivs:
            c = _simplify_zero(rem.coeff(d))
            coeffs.append(c)
            rem = _simplify_zero(rem - c * d)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if rem != 0 or all(c == 0 for c in coeffs):
        return None
    if any(c.has(dep_expr) or any(c.has(d) for d in derivs) for c in coeffs if c != 0):
        return None
    try:
        t_var = next(v for v in indep_vars if str(v) == 't' and coeffs[indep_vars.index(v)] != 0)
    except StopIteration:
        nonzero = [i for i, c in enumerate(coeffs) if c != 0]
        if not nonzero:
            return None
        t_var = indep_vars[nonzero[-1]]
    t_idx = indep_vars.index(t_var)
    t_coeff = coeffs[t_idx]
    if t_coeff == 0:
        return None
    invariants = []
    for i, v in enumerate(indep_vars):
        if i == t_idx:
            continue
        invariants.append(sp.expand(v - coeffs[i] / t_coeff * t_var))
    if not invariants:
        return None
    func = sp.Function('F')
    return sp.Eq(dep_expr, func(*invariants))


def _telegraph_traveling_wave_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) != 2 or not getattr(dep_expr, 'is_Function', False):
        return None
    x, t = indep_vars
    utt = sp.diff(dep_expr, t, 2)
    ut = sp.diff(dep_expr, t)
    uxx = sp.diff(dep_expr, x, 2)
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    try:
        coeff_utt = expr.coeff(utt)
        rem = _simplify_zero(expr - coeff_utt * utt)
        coeff_ut = rem.coeff(ut)
        rem2 = _simplify_zero(rem - coeff_ut * ut)
        coeff_uxx = rem2.coeff(uxx)
        tail = _simplify_zero(rem2 - coeff_uxx * uxx)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if coeff_utt == 0 or coeff_ut == 0 or tail != 0:
        return None
    speed_sq = _simplify_zero(-coeff_uxx / coeff_utt)
    damping = _simplify_zero(coeff_ut / coeff_utt)
    if speed_sq == 0 or damping == 0:
        return None
    c = sp.Symbol('c', exclude=[-sp.sqrt(speed_sq), sp.sqrt(speed_sq)])
    c1 = sp.Symbol('C1')
    c2 = sp.Symbol('C2')
    xi = x - c * t
    denom = sp.expand(c**2 - speed_sq)
    if denom == 0:
        return None
    rhs = c1 + c2 * sp.exp(sp.simplify(damping * c * xi / denom))
    try:
        residual = _simplify_zero((coeff_utt * sp.diff(rhs, t, 2) + coeff_ut * sp.diff(rhs, t) + coeff_uxx * sp.diff(rhs, x, 2)).doit())
        if residual == 0:
            return sp.Eq(dep_expr, rhs)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    return None

def _reaction_diffusion_similarity_solution(eq_or_expr, dep_expr, indep_vars):
    indep_vars = tuple(indep_vars)
    if len(indep_vars) != 2 or not getattr(dep_expr, "is_Function", False):
        return None
    x, t = indep_vars
    ut = sp.diff(dep_expr, t)
    ux = sp.diff(dep_expr, x)
    uxx = sp.diff(dep_expr, x, 2)
    expr = eq_or_expr.lhs - eq_or_expr.rhs if isinstance(eq_or_expr, sp.Equality) else eq_or_expr
    expr = _simplify_zero(expr)
    try:
        coeff_ut = expr.coeff(ut)
        rem = _simplify_zero(expr - coeff_ut * ut)
        coeff_ux = rem.coeff(ux)
        rem2 = _simplify_zero(rem - coeff_ux * ux)
        coeff_uxx = rem2.coeff(uxx)
        rem3 = _simplify_zero(rem2 - coeff_uxx * uxx)
        coeff_u = rem3.coeff(dep_expr)
        tail = _simplify_zero(rem3 - coeff_u * dep_expr)
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    if coeff_ut == 0 or tail != 0:
        return None
    adv = _simplify_zero(coeff_ux / coeff_ut)
    diff = _simplify_zero(-coeff_uxx / coeff_ut)
    growth = _simplify_zero(-coeff_u / coeff_ut)
    if diff == 0:
        return None
    c1 = sp.Symbol('C1')
    c2 = sp.Symbol('C2')
    try:
        arg = (x - adv*t) / (2 * sp.sqrt(diff*t))
        profile = c1 + c2 * sp.erf(arg)
        sol = sp.Eq(dep_expr, sp.exp(growth * t) * profile)
        residual = _simplify_zero((sp.diff(sol.rhs, t) + adv*sp.diff(sol.rhs, x) - diff*sp.diff(sol.rhs, x, 2) - growth*sol.rhs).doit())
        if residual == 0:
            return sol
    except (TypeError, ValueError, AttributeError, NotImplementedError):
        return None
    return None


def _fallback_known_solution(eq_or_expr, dep_expr, indep_vars):
    for solver in (_constant_coefficient_transport_solution, _reaction_diffusion_similarity_solution, _heat_similarity_solution, _advection_diffusion_similarity_solution, _wave_dalembert_solution, _telegraph_traveling_wave_solution):
        sol = solver(eq_or_expr, dep_expr, indep_vars)
        if sol is not None:
            return sol, solver.__name__
    return None, None

def _best_available_output(result: LieSymmetrySolveResult, *, result_level: str = "solution"):
    if result_level == "details":
        return result
    if result_level == "reduction":
        if isinstance(result.full_solution, sp.Equality):
            return result.full_solution
        if isinstance(result.reduced_solution, sp.Equality) and getattr(result.reduction, "ansatz", None) is not None:
            try:
                return sp.Eq(result.reduction.dep_expr, sp.expand(result.reduction.ansatz.subs(result.reduced_solution.lhs, result.reduced_solution.rhs)))
            except (TypeError, ValueError, AttributeError, NotImplementedError):
                return result.reduced_solution
        if isinstance(result.reduced_solution, sp.Equality):
            return result.reduced_solution
        if isinstance(result.reduced_equation, sp.Equality):
            return result.reduced_equation
        if isinstance(result.direct_solution, sp.Equality):
            return result.direct_solution
        return None
    if isinstance(result.full_solution, sp.Equality):
        return result.full_solution
    if isinstance(result.reduced_solution, sp.Equality) and getattr(result.reduction, "ansatz", None) is not None:
        try:
            return sp.Eq(result.reduction.dep_expr, sp.expand(result.reduction.ansatz.subs(result.reduced_solution.lhs, result.reduced_solution.rhs)))
        except (TypeError, ValueError, AttributeError, NotImplementedError):
            pass
    if isinstance(result.direct_solution, sp.Equality):
        return result.direct_solution
    return None

def _is_usable_reduced_equation(reduced_eq):
    if not isinstance(reduced_eq, sp.Equality):
        return False
    return not bool(getattr(reduced_eq, 'has', lambda *_: False)(sp.nan, sp.zoo, sp.oo, -sp.oo))


def liepde(
    eq_or_expr,
    dep_expr_or_func,
    indep_vars,
    *,
    degree: int = 1,
    max_degree: int | None = None,
    max_subset_size: int = 2,
    max_principal_order: int | None = None,
    prefer_commuting_subalgebra: bool = True,
    use_workflow: bool = False,
    workflow_max_steps: int = 2,
    return_details: bool = False,
    result_level: str = "solution",
    failure_mode: str = "none",
    analyze_only: bool = False,
):
    if return_details:
        result_level = "details"
    if analyze_only:
        return classify_pde(eq_or_expr, dep_expr_or_func, indep_vars, max_principal_order=max_principal_order)
    dep_expr, _dep_func, eq_obj, info = build_equation_object(
        eq_or_expr, dep_expr_or_func, indep_vars, max_principal_order=max_principal_order
    )
    if max_degree is None:
        max_degree = degree

    warnings: list[str] = []
    degree_attempts: list[int] = []
    best_sym = None
    basis = ()
    matches = ()
    reduction = None

    for trial_degree in range(int(degree), int(max_degree) + 1):
        degree_attempts.append(trial_degree)
        try:
            sym = compute_polynomial_symmetries(eq_obj, degree=trial_degree)
        except Exception as exc:
            warnings.append(f"degree {trial_degree} symmetry solve failed: {type(exc).__name__}: {exc}")
            continue
        try:
            cand_basis, cand_matches, cand_reduction = search_reductions_from_symmetries(
                eq_obj,
                sym,
                max_subset_size=max_subset_size,
                prefer_commuting_subalgebra=prefer_commuting_subalgebra,
            )
        except Exception as exc:
            warnings.append(f"degree {trial_degree} reduction search failed: {type(exc).__name__}: {exc}")
            best_sym = sym
            continue
        best_sym = sym
        basis = cand_basis
        matches = cand_matches
        reduction = cand_reduction
        if reduction is not None:
            break

    verification = None
    reduced_eq = None if reduction is None else reduction.reduced_equation
    if not _is_usable_reduced_equation(reduced_eq):
        if reduction is not None:
            warnings.append("reduction search produced a non-usable reduced equation; discarding reduction result.")
        reduction = None
        reduced_eq = None
    if reduction is not None:
        try:
            verification = verify_reduction(eq_obj, reduction)
        except Exception as exc:
            warnings.append(f"reduction verification failed: {type(exc).__name__}: {exc}")

    reduced_solution = None
    if reduced_eq is not None:
        try:
            reduced_solution = solve_reduced_equation(reduced_eq, dep_expr, reduction=reduction)
        except Exception as exc:
            warnings.append(f"reduced equation solve failed: {type(exc).__name__}: {exc}")

    full_solution = None
    if reduction is not None and isinstance(reduced_solution, sp.Equality):
        try:
            full_solution = sp.Eq(dep_expr, sp.expand(reduction.ansatz.subs(reduced_solution.lhs, reduced_solution.rhs)))
        except Exception as exc:
            warnings.append(f"solution reconstruction failed: {type(exc).__name__}: {exc}")

    direct_solution = None
    if full_solution is None:
        try:
            direct_solution = sp.pdsolve(eq_or_expr, func=dep_expr)
        except Exception as exc:
            warnings.append(f"direct pdsolve fallback failed: {type(exc).__name__}: {exc}")
        if direct_solution is None:
            direct_solution, fallback_name = _fallback_known_solution(eq_or_expr, dep_expr, indep_vars)
            if direct_solution is not None:
                warnings.append(f"used known-solution fallback: {fallback_name}.")

    workflow = None
    if use_workflow:
        workflow_error = None
        try:
            workflow = repeated_reduction_workflow_scalar_kd_frobenius_default(
                eq_obj, max_steps=workflow_max_steps, symmetry_degree=degree, max_subset_size=max_subset_size
            )
        except Exception as exc:
            workflow_error = exc
        if workflow is None:
            try:
                workflow = repeated_reduction_workflow_scalar_kd(
                    eq_obj,
                    max_steps=workflow_max_steps,
                    symmetry_degree=degree,
                    max_subset_size=max_subset_size,
                    prefer_commuting_subalgebra=prefer_commuting_subalgebra,
                )
            except Exception as exc:
                warnings.append(f"workflow construction failed: {type(exc).__name__}: {exc}")
                if workflow_error is not None:
                    warnings.append(f"frobenius workflow fallback failed after: {type(workflow_error).__name__}: {workflow_error}")

    result = LieSymmetrySolveResult(
        equation_object=eq_obj,
        principal_info=info,
        symmetry_solution=best_sym,
        basis_vectors=basis,
        matches=matches,
        reduction=reduction,
        verification=verification,
        reduced_equation=reduced_eq,
        reduced_solution=reduced_solution,
        full_solution=full_solution,
        direct_solution=direct_solution,
        workflow=workflow,
        diagnostics=LiePDEDiagnostics(tuple(warnings), tuple(degree_attempts)),
    )
    output = _best_available_output(result, result_level=result_level)
    if result_level == "details":
        return result
    if output is not None:
        return output
    if failure_mode == "message":
        reason = warnings[-1] if warnings else "No symmetry reduction or fallback solution was found."
        return reason
    if failure_mode == "status":
        reason = warnings[-1] if warnings else "No symmetry reduction or fallback solution was found."
        return LiePDEFailure(reason=reason, warnings=tuple(warnings))
    return None
