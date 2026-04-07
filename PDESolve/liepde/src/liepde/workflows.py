from __future__ import annotations

from itertools import combinations

from dataclasses import dataclass

import sympy as sp
from .pde import build_scalar_jet_equation_from_sympy_pde, build_scalar_general_solved_pde_from_equation
from .symmetry import solve_determining_equations_with_polynomial_ansatz_scalar_general_kd
from .reduction import search_symbolic_linear_combinations_for_reduction_scalar_kd, auto_reduce_best_commuting_subalgebra_scalar_kd, auto_reduce_best_symbolic_match_scalar_kd
from .utils import collect_applied_functions, is_zero


@dataclass
class ReductionStep:
    principal_info: object
    symmetry_solution: object
    matches: tuple
    reduced: object | None
    reduced_pde: sp.Equality | None
    diagnostics: object | None = None
    verification: object | None = None


@dataclass
class RepeatedReductionResult:
    steps: tuple[ReductionStep, ...]
    final_equation: sp.Equality | None


def _reduced_equation_to_new_problem(reduced_eq, dep_name='u', max_order=3):
    """Convert a reduced equation in terms of one applied function to a new jet-space problem."""
    functions = collect_applied_functions(reduced_eq)
    if len(functions) != 1:
        return None
    dep = functions[0]
    indep = dep.args
    jet, pde = build_scalar_jet_equation_from_sympy_pde(indep, dep.func, reduced_eq, max_order=max_order, dep_name=dep_name)
    eq_obj, info = build_scalar_general_solved_pde_from_equation(jet, pde, max_principal_order=max_order)
    return eq_obj, info


def repeated_reduction_workflow_scalar_kd(eq_obj, max_steps=3, symmetry_degree=1, max_subset_size=2, prefer_commuting_subalgebra=True):
    current = eq_obj
    steps = []
    final_eq = None
    for _ in range(max_steps):
        sym = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(current, degree=symmetry_degree, include_dependent_var=True)
        basis = sym.basis_vectors()
        matches = tuple(search_symbolic_linear_combinations_for_reduction_scalar_kd(current, basis, max_subset_size=max_subset_size, try_translation=True, try_diagonal_scaling=True, try_affine=True, normalize=True, rank_results=True))
        if prefer_commuting_subalgebra:
            reduced = auto_reduce_best_commuting_subalgebra_scalar_kd(current, list(matches), max_generators=min(current.jet.k - 1, max_subset_size))
        else:
            reduced = None
        if reduced is None and matches:
            reduced = auto_reduce_best_symbolic_match_scalar_kd(current, list(matches))
        red_eq = None if reduced is None else reduced.reduced_equation
        ver = None if reduced is None else verify_reduction(current, reduced)
        steps.append(ReductionStep(principal_info=getattr(current, 'principal_multiindex', None), symmetry_solution=sym, matches=matches, reduced=reduced, reduced_pde=red_eq, verification=ver))
        if reduced is None or red_eq is None:
            final_eq = red_eq
            break
        nxt = _reduced_equation_to_new_problem(red_eq, dep_name=current.jet.dep_name, max_order=current.jet.max_order)
        if nxt is None:
            final_eq = red_eq
            break
        current, _ = nxt
        final_eq = red_eq
    return RepeatedReductionResult(tuple(steps), final_eq)


from .frobenius import restricted_local_frobenius_engine, restricted_local_frobenius_engine_explain
from .geometry import DistributionKD, VectorFieldKD
from .reduction import reduce_scalar_by_frobenius_chart
from .verify import verify_reduction


def _obvious_translation_distribution(eq_obj):
    xs = list(eq_obj.jet.xs)
    fields = []
    G = sp.expand(eq_obj.G) if hasattr(eq_obj, 'G') else None
    P = getattr(eq_obj, 'principal_multiindex', None)
    for i, x in enumerate(xs):
        if G is None:
            continue
        if is_zero(sp.diff(G, x)):
            coeffs = [sp.Integer(0)] * len(xs)
            coeffs[i] = sp.Integer(1)
            fields.append(VectorFieldKD(tuple(xs), tuple(coeffs)))
    if not fields:
        return None
    # prefer up to k-1 translations to leave at least one invariant
    fields = tuple(fields[: max(1, len(xs)-1)])
    return DistributionKD(tuple(xs), fields)


def repeated_reduction_workflow_scalar_kd_frobenius_default(eq_obj, max_steps=3, symmetry_degree=1, max_subset_size=2):
    """
    Repeated reduction workflow preferring the Frobenius engine as the default backend.

    Strategy per step:
      1. solve polynomial symmetry ansatz;
      2. extract commuting candidate subdistributions from the basis itself;
      3. attempt Frobenius chart construction and reduction;
      4. fall back to existing symbolic-match reducers.
    """
    current = eq_obj
    steps = []
    final_eq = None

    for _ in range(max_steps):
        sym = solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(current, degree=symmetry_degree, include_dependent_var=True)
        basis = sym.basis_vectors()
        matches = tuple(search_symbolic_linear_combinations_for_reduction_scalar_kd(current, basis, max_subset_size=max_subset_size, try_translation=True, try_diagonal_scaling=True, try_affine=True, normalize=True, rank_results=True))

        reduced = None
        diagnostics = None

        # Try Frobenius backend on basis subdistributions with phi=0 first, preferring commuting but allowing involutive affine.
        candidate_dists = []
        for r in range(1, min(current.jet.k - 1, len(basis)) + 1):
            for idxs in combinations(range(len(basis)), r):
                sub = [basis[i] for i in idxs]
                dist = DistributionKD(current.jet.xs, tuple(VectorFieldKD(current.jet.xs, tuple(v for v in Xis)) for Xis, Phi, p in sub if sp.simplify(Phi) == 0))
                if dist.size != r:
                    continue
                report = restricted_local_frobenius_engine_explain(dist)
                if report.get('success'):
                    candidate_dists.append((idxs, dist, report))
            if candidate_dists:
                break
        for idxs, dist, report in candidate_dists:
            try:
                chart = restricted_local_frobenius_engine(dist)
                reduced = reduce_scalar_by_frobenius_chart(current, chart, a_list=[0] * len(chart.transverse), beta_list=[0] * len(chart.transverse))
                diagnostics = report
                break
            except (TypeError, ValueError, NotImplementedError) as exc:
                diagnostics = {'success': False, 'error': f'{type(exc).__name__}: {exc}'}
                continue

        if reduced is None:
            reduced = auto_reduce_best_commuting_subalgebra_scalar_kd(current, list(matches), max_generators=min(current.jet.k - 1, max_subset_size))
        if reduced is None and matches:
            reduced = auto_reduce_best_symbolic_match_scalar_kd(current, list(matches))
        if reduced is None:
            obvious = _obvious_translation_distribution(current)
            if obvious is not None:
                try:
                    chart = restricted_local_frobenius_engine(obvious)
                    reduced = reduce_scalar_by_frobenius_chart(current, chart, a_list=[0] * len(chart.transverse), beta_list=[0] * len(chart.transverse))
                    diagnostics = {'success': True, 'best_method': 'obvious_translation_fallback'}
                except (TypeError, ValueError, NotImplementedError) as exc:
                    diagnostics = {'success': False, 'best_method': 'obvious_translation_fallback', 'error': f'{type(exc).__name__}: {exc}'}

        red_eq = None if reduced is None else reduced.reduced_equation
        ver = None if reduced is None else verify_reduction(current, reduced)
        steps.append(ReductionStep(principal_info=getattr(current, 'principal_multiindex', None), symmetry_solution=sym, matches=matches, reduced=reduced, reduced_pde=red_eq, verification=ver))
        if reduced is None or red_eq is None:
            final_eq = red_eq
            break
        nxt = _reduced_equation_to_new_problem(red_eq, dep_name=current.jet.dep_name, max_order=current.jet.max_order)
        if nxt is None:
            final_eq = red_eq
            break
        current, _ = nxt
        final_eq = red_eq
    return RepeatedReductionResult(tuple(steps), final_eq)


@dataclass
class ReductionHistoryEntry:
    step_index: int
    equation_signature: str | None
    chosen_backend: str | None
    notes: tuple[str, ...] = ()


@dataclass
class ManagedRepeatedReductionResult:
    steps: tuple[ReductionStep, ...]
    final_equation: sp.Equality | None
    history: tuple[ReductionHistoryEntry, ...]
    seen_signatures: tuple[str, ...]


def _equation_signature(eq) -> str | None:
    if eq is None:
        return None
    try:
        lhs = sp.expand(eq.lhs - eq.rhs)
    except Exception:
        lhs = sp.expand(eq)
    return sp.srepr(lhs)


def repeated_reduction_workflow_scalar_kd_managed(eq_obj, max_steps=3, symmetry_degree=1, max_subset_size=2, prefer_frobenius=True, avoid_equivalent=True):
    current = eq_obj
    steps = []
    history = []
    seen = set()
    final_eq = None
    for step_index in range(max_steps):
        sig = _equation_signature(sp.Eq(current.equation(), 0))
        if avoid_equivalent and sig in seen:
            history.append(ReductionHistoryEntry(step_index, sig, None, ('equivalent_equation_seen',)))
            break
        if sig is not None:
            seen.add(sig)
        try:
            if prefer_frobenius:
                result = repeated_reduction_workflow_scalar_kd_frobenius_default(current, max_steps=1, symmetry_degree=symmetry_degree, max_subset_size=max_subset_size)
                step = result.steps[0]
                backend = 'frobenius_default'
            else:
                result = repeated_reduction_workflow_scalar_kd(current, max_steps=1, symmetry_degree=symmetry_degree, max_subset_size=max_subset_size)
                step = result.steps[0]
                backend = 'symbolic_default'
        except (TypeError, ValueError, NotImplementedError) as exc:
            history.append(ReductionHistoryEntry(step_index, sig, None, (f'workflow_error:{type(exc).__name__}:{exc}',)))
            break
        steps.append(step)
        history.append(ReductionHistoryEntry(step_index, sig, backend, tuple()))
        final_eq = step.reduced_pde
        if step.reduced_pde is None:
            break
        nxt = _reduced_equation_to_new_problem(step.reduced_pde, dep_name=current.jet.dep_name, max_order=current.jet.max_order)
        if nxt is None:
            break
        current, _ = nxt
        # Stop early on effectively 1D reduced problems where the polynomial symmetry workflow is still fragile.
        if getattr(current.jet, 'k', len(getattr(current.jet, 'xs', ()))) <= 1:
            break
    return ManagedRepeatedReductionResult(tuple(steps), final_eq, tuple(history), tuple(seen))
