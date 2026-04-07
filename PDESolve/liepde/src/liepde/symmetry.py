from __future__ import annotations

from dataclasses import dataclass

import sympy as sp

from .pde import ScalarGeneralSolvedPDEKD
from .utils import add_multiindex, expr_complexity, multiindex_geq, multiindex_sum
from .performance import prolongation_coefficients_from_cache


def monomials_in_vars(vars_, max_total_degree):
    vars_ = tuple(vars_)
    out = []
    def rec(i, remaining, term):
        if i == len(vars_):
            out.append(term)
            return
        v = vars_[i]
        for d in range(remaining + 1):
            rec(i + 1, remaining - d, term * v**d)
    rec(0, max_total_degree, sp.Integer(1))
    return out


def _prolongation_coefficients_scalar_kd_uncached(jet, xis, phi):
    xis = [sp.expand(xi) for xi in xis]
    phi = sp.expand(phi)
    coeffs = {(0,) * jet.k: phi}
    for total_order in range(0, jet.max_order):
        for J in jet._indices_by_order[total_order]:
            current = coeffs[J]
            for axis in range(jet.k):
                Jn = add_multiindex(J, axis)
                if Jn in coeffs:
                    continue
                D_axis_current = jet.total_derivative(current, axis)
                correction = 0
                for ell in range(jet.k):
                    correction += jet.coord(add_multiindex(J, ell)) * jet.total_derivative(xis[ell], axis)
                coeffs[Jn] = sp.expand(D_axis_current - correction)
    return coeffs


def prolongation_coefficients_scalar_kd(jet, xis, phi):
    return prolongation_coefficients_from_cache(jet, xis, phi)


def prolongation_action_scalar_kd(jet, expr, xis, phi):
    coeffs = prolongation_coefficients_scalar_kd(jet, xis, phi)
    result = sum(xis[i] * sp.diff(expr, jet.xs[i]) for i in range(jet.k)) + phi * sp.diff(expr, jet.u)
    for J in jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        result += coeffs[J] * sp.diff(expr, jet.coord(J))
    return sp.expand(result)


def determining_equations_for_scalar_general_solved_pde_kd(eq_obj: ScalarGeneralSolvedPDEKD):
    jet = eq_obj.jet
    xs = jet.xs
    u = jet.u
    P = eq_obj.principal_multiindex
    args = (*xs, u)
    xis = [sp.Function(f"xi{i+1}")(*args) for i in range(jet.k)]
    phi = sp.Function("phi")(*args)
    raw = prolongation_action_scalar_kd(jet, eq_obj.equation(), xis, phi)
    expr = sp.expand(raw.subs(eq_obj.differential_consequence_substitutions(order_needed=jet.max_order)))
    gens = [jet.coord(J) for J in jet.all_indices() if multiindex_sum(J) >= 1 and not multiindex_geq(J, P)]
    det_eqs = []
    if gens:
        poly = sp.Poly(expr, *gens, domain="EX")
        coeffs = poly.coeffs()
    else:
        coeffs = [expr]
    for c in coeffs:
        c = sp.expand(c)
        if c != 0:
            det_eqs.append(sp.Eq(c, 0))
    return xis, phi, det_eqs


@dataclass
class PolynomialSymmetrySolutionScalarGeneralKD:
    monomials: list
    xi_ansatzes: list
    phi_ansatz: sp.Expr
    unknowns: list
    solution_tuple: tuple
    xi_solutions: list
    phi_solution: sp.Expr
    parameter_symbols: tuple = ()

    def basis_vectors(self):
        if self.parameter_symbols:
            basis = []
            for p in self.parameter_symbols:
                xis = [sp.expand(sp.diff(expr, p)) for expr in self.xi_solutions]
                phi = sp.expand(sp.diff(self.phi_solution, p))
                if any(v != 0 for v in xis) or phi != 0:
                    basis.append((xis, phi, p))
            if basis:
                return basis
        if any(v != 0 for v in self.xi_solutions) or self.phi_solution != 0:
            return [(list(self.xi_solutions), self.phi_solution, None)]
        return []




def _applied_function_signature(expr):
    if hasattr(expr, 'func') and getattr(expr, 'is_Function', False):
        return (expr.func.__name__, tuple(str(a) for a in expr.args))
    return None


def _substitute_function_ansatzes(expr, repl):
    repl_sig = {}
    for k, v in repl.items():
        sig = _applied_function_signature(k)
        if sig is not None:
            repl_sig[sig] = v
    deriv_map = {}
    func_map = {}
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Derivative):
            sig = _applied_function_signature(node.expr)
            if sig in repl_sig:
                replacement = repl_sig[sig]
                for var, count in node.variable_count:
                    replacement = sp.diff(replacement, var, count)
                deriv_map[node] = sp.expand(replacement)
        else:
            sig = _applied_function_signature(node)
            if sig in repl_sig:
                func_map[node] = repl_sig[sig]
    out = expr.xreplace(deriv_map)
    out = out.xreplace(func_map)
    return sp.expand(out)


def solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(eq_obj: ScalarGeneralSolvedPDEKD, degree: int = 2, include_dependent_var: bool = True):
    jet = eq_obj.jet
    xs = list(jet.xs)
    u = jet.u
    P = eq_obj.principal_multiindex
    xi_funs, phi_fun, det_eqs = determining_equations_for_scalar_general_solved_pde_kd(eq_obj)
    ansatz_vars = [*xs, u] if include_dependent_var else [*xs]
    mons = monomials_in_vars(ansatz_vars, degree)
    xi_blocks = [sp.symbols(f"a{i}_0:{len(mons)}") for i in range(jet.k)]
    c = sp.symbols(f"c0:{len(mons)}")
    xi_ans = [sum(xi_blocks[i][r] * mons[r] for r in range(len(mons))) for i in range(jet.k)]
    phi_ans = sum(c[r] * mons[r] for r in range(len(mons)))
    repl = {xi_funs[i]: xi_ans[i] for i in range(jet.k)}
    repl[phi_fun] = phi_ans
    jet_gens = [jet.coord(J) for J in jet.all_indices() if multiindex_sum(J) >= 1 and not multiindex_geq(J, P)]
    poly_gens = [*jet_gens, *xs, u]
    scalar_equations = []
    for det_eq in det_eqs:
        expr = _substitute_function_ansatzes(det_eq.lhs, repl)
        if poly_gens:
            poly = sp.Poly(expr, *poly_gens, domain="EX")
            coeffs = poly.coeffs()
        else:
            coeffs = [expr]
        scalar_equations.extend(sp.expand(coeff) for coeff in coeffs)
    scalar_equations = _deduplicate_scalar_equations(scalar_equations)
    unknowns = [sym for block in xi_blocks for sym in block] + list(c)
    if not scalar_equations:
        raise ValueError("Determining equations vanished for the chosen ansatz; the polynomial family is underdetermined.")
    if len(scalar_equations) > 4 * len(unknowns):
        scalar_equations = sorted(scalar_equations, key=expr_complexity)
    solset = sp.linsolve(scalar_equations, unknowns)
    if not solset:
        raise ValueError("No polynomial symmetry solution found for the chosen ansatz.")
    solution_tuple = tuple(sp.expand(v) for v in list(solset)[0])
    parameter_symbols = tuple(sorted(list(set().union(*(expr.free_symbols for expr in solution_tuple))), key=lambda s: s.name))
    sub_unknowns = dict(zip(unknowns, solution_tuple))
    xi_solutions = [sp.expand(expr.subs(sub_unknowns)) for expr in xi_ans]
    phi_solution = sp.expand(phi_ans.subs(sub_unknowns))
    return PolynomialSymmetrySolutionScalarGeneralKD(
        mons,
        xi_ans,
        phi_ans,
        unknowns,
        solution_tuple,
        xi_solutions,
        phi_solution,
        parameter_symbols=parameter_symbols,
    )



def _deduplicate_scalar_equations(equations):
    seen = set()
    out = []
    for eq in equations:
        expanded = sp.expand(eq)
        if expanded == 0:
            continue
        signature = sp.srepr(expanded)
        if signature in seen:
            continue
        seen.add(signature)
        out.append(expanded)
    return out


def estimate_polynomial_ansatz_size(eq_obj: ScalarGeneralSolvedPDEKD, degree: int = 2, include_dependent_var: bool = True):
    jet = eq_obj.jet
    variable_count = len(jet.xs) + (1 if include_dependent_var else 0)
    monomial_count = len(monomials_in_vars([*jet.xs, jet.u] if include_dependent_var else [*jet.xs], degree))
    unknown_count = (jet.k + 1) * monomial_count
    return {
        "variable_count": variable_count,
        "monomial_count": monomial_count,
        "unknown_count": unknown_count,
        "estimated_cost": monomial_count * unknown_count,
    }
