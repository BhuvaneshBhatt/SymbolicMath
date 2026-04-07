from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import sympy as sp

from .coordinates import probe_transport_coordinates_for_affine_distribution, construct_characteristic_coordinates, construct_characteristic_coordinates_with_diagnostics, solve_invariants_transverse_pdsolve_k2, find_coordinates_translation_subalgebra, find_coordinates_diagonal_scaling_subalgebra
from .geometry import DistributionKD, VectorFieldKD
from .canonical import canonicalize_reduced_equation, canonicalize_coordinate_chart
from .pde import ScalarGeneralSolvedPDEKD, ScalarJetSpaceKD
from .utils import (
    add_multiindex,
    collect_applied_functions,
    dedupe_preserve_order,
    expr_complexity,
    factor_prefactor_independent_of_symbols,
    first_nonzero_scale,
    is_zero,
    matrix_is_diagonal,
    matrix_is_zero,
    multiindex_sum,
    poly_zero_equations,
    right_inverse_columns,
    safe_solve_for,
    substitute_free_parameters_zero,
)


def _is_zero_fast(expr):
    return is_zero(sp.expand(expr))


def _generator_signature(generator):
    xis, phi, _ = generator
    return tuple(sp.srepr(sp.expand(v)) for v in list(xis) + [phi])


def _has_nontrivial_generator(generator):
    xis, phi, _ = generator
    return any(not _is_zero_fast(comp) for comp in list(xis) + [phi])


def _combination_jobs_for_subset(subbasis, try_translation=True, try_diagonal_scaling=True, try_affine=True):
    jobs = []
    if not any(_has_nontrivial_generator(gen) for gen in subbasis):
        return jobs
    if try_translation:
        jobs.append("translation")
    if try_diagonal_scaling:
        jobs.append("scaling")
    if try_affine:
        jobs.append("affine")
    return jobs

@dataclass
class SymbolicCombinationMatchScalarKD:
    basis_indices: tuple
    anchor_slot: int
    coefficients: tuple
    generator_xis: list
    generator_phi: sp.Expr
    match_data: dict


@dataclass
class ScalarReductionResultKD:
    kind: str
    invariants: tuple
    transverse_parameter: sp.Expr | tuple
    ansatz: sp.Expr
    reduced_expression: sp.Expr
    reduced_equation: sp.Equality
    reduced_function: sp.Function
    reduced_unknown: sp.Expr | None = None


@dataclass
class ScalarMultiReductionResultKD:
    kind: str
    subalgebra_dimension: int
    invariants: tuple
    transverse_parameters: tuple
    ansatz: sp.Expr
    reduced_expression: sp.Expr
    reduced_equation: sp.Equality
    reduced_function: sp.Function
    generators: tuple
    reduced_unknown: sp.Expr | None = None



class ReducedFunctionJetKD:
    def __init__(self, x_vars, invariant_symbols, invariant_exprs, max_order):
        self.xs = tuple(x_vars)
        self.zs = tuple(invariant_symbols)
        self.zexprs = tuple(map(sp.expand, invariant_exprs))
        self.k = len(self.xs)
        self.r = len(self.zs)
        self.max_order = max_order
        self._indices_by_order = {n: list(self._weak_comp(n, self.r)) for n in range(max_order + 2)}
        self._cache = {}
        for n in range(max_order + 2):
            for M in self._indices_by_order[n]:
                self._cache[M] = sp.Symbol(self._coord_name(M))

    @staticmethod
    def _weak_comp(total, r):
        if r == 0:
            if total == 0:
                yield ()
            return
        if r == 1:
            yield (total,)
            return
        for i in range(total + 1):
            for rest in ReducedFunctionJetKD._weak_comp(total - i, r - 1):
                yield (i,) + rest

    def _coord_name(self, M):
        if len(M) == 0:
            return "F"
        parts = ["F"]
        for i, power in enumerate(M):
            if power == 0:
                continue
            name = str(self.zs[i])
            parts.append(name if power == 1 else f"{name}{power}")
        return "_".join(parts)

    def coord(self, M):
        return self._cache[tuple(M)]

    def total_derivative(self, expr, axis):
        result = sp.diff(expr, self.xs[axis])
        for a, za in enumerate(self.zs):
            result += sp.diff(expr, za) * sp.diff(self.zexprs[a], self.xs[axis])
        for n in range(self.max_order + 1):
            for M in self._indices_by_order[n]:
                FM = self.coord(M)
                dFM = 0
                for a in range(self.r):
                    Mnext = list(M)
                    Mnext[a] += 1
                    dFM += self.coord(tuple(Mnext)) * sp.diff(self.zexprs[a], self.xs[axis])
                result += sp.diff(expr, FM) * dFM
        return sp.expand(result)

    def to_function_notation(self, expr, f):
        out = sp.expand(expr)
        for n in range(self.max_order + 1, -1, -1):
            for M in self._indices_by_order[n]:
                FM = self.coord(M)
                if sum(M) == 0:
                    out = out.subs(FM, f(*self.zs))
                else:
                    deriv_args = []
                    for i, mi in enumerate(M):
                        if mi > 0:
                            deriv_args.append((self.zs[i], mi))
                    out = out.subs(FM, sp.diff(f(*self.zs), *deriv_args))
        return sp.expand(out)


def _apply_multi_total_derivative(reducer: ReducedFunctionJetKD, expr, J):
    out = expr
    for axis, reps in enumerate(J):
        for _ in range(reps):
            out = reducer.total_derivative(out, axis)
    return sp.expand(out)




def _is_constant_in(expr, vars_):
    expr = sp.expand(expr)
    return expr.free_symbols.isdisjoint(set(vars_))


def _decompose_affine_independent_action_scalar_kd(Xis, xs, u):
    k = len(xs)
    M = sp.zeros(k, k)
    bvec = sp.zeros(k, 1)
    for i, Xi in enumerate(Xis):
        Xi = sp.expand(Xi)
        if u in Xi.free_symbols:
            return None
        for j, xj in enumerate(xs):
            mij = sp.expand(sp.diff(Xi, xj))
            if any(v in mij.free_symbols for v in xs + [u]):
                return None
            M[i, j] = mij
        residual = sp.expand(Xi - sum(M[i, j] * xs[j] for j in range(k)))
        if any(v in residual.free_symbols for v in xs + [u]):
            return None
        bvec[i, 0] = residual
    return M, bvec


def _decompose_affine_u_action_scalar_kd(Phi, xs, u):
    Phi = sp.expand(Phi)
    a = sp.expand(sp.diff(Phi, u))
    if any(v in a.free_symbols for v in xs + [u]):
        return None
    b = sp.expand(Phi - a * u)
    if any(v in b.free_symbols for v in xs + [u]):
        return None
    return a, b


def match_translation_affine_scalar_kd(Xis, Phi, xs, u):
    xis_const = []
    for Xi in Xis:
        if not _is_constant_in(Xi, [*xs, u]):
            return None
        xis_const.append(sp.expand(Xi))
    if all(sp.simplify(v) == 0 for v in xis_const):
        return None
    dec = _decompose_affine_u_action_scalar_kd(Phi, xs, u)
    if dec is None:
        return None
    a, b = dec
    return {"kind": "translation_affine_scalar_kd", "xis": tuple(xis_const), "a": a, "b": b}


def match_diagonal_scaling_affine_scalar_kd(Xis, Phi, xs, u):
    scales = []
    for i, Xi in enumerate(Xis):
        ci = sp.expand(sp.diff(Xi, xs[i]))
        if any(v in ci.free_symbols for v in [*xs, u]):
            return None
        if sp.expand(Xi - ci * xs[i]) != 0:
            return None
        scales.append(ci)
    if all(sp.simplify(v) == 0 for v in scales):
        return None
    dec = _decompose_affine_u_action_scalar_kd(Phi, xs, u)
    if dec is None:
        return None
    a, b = dec
    if sp.simplify(b) != 0:
        return None
    return {"kind": "diagonal_scaling_affine_scalar_kd", "scales": tuple(scales), "a": a}


def match_affine_independent_affine_u_scalar_kd(Xis, Phi, xs, u):
    dec_x = _decompose_affine_independent_action_scalar_kd(Xis, xs, u)
    if dec_x is None:
        return None
    dec_u = _decompose_affine_u_action_scalar_kd(Phi, xs, u)
    if dec_u is None:
        return None
    M, bvec = dec_x
    a, beta = dec_u
    if matrix_is_zero(M) and matrix_is_zero(bvec) and sp.simplify(a) == 0 and sp.simplify(beta) == 0:
        return None
    return {"kind": "affine_independent_affine_u_scalar_kd", "M": M, "bvec": bvec, "a": a, "beta": beta}




def _normalization_candidates_from_affine_action(M, bvec):
    cands = []
    rows, cols = M.shape
    for i in range(rows):
        if sp.simplify(M[i, i]) != 0:
            cands.append(sp.expand(M[i, i]))
    for v in list(bvec):
        if sp.simplify(v) != 0:
            cands.append(sp.expand(v))
    for v in list(M):
        if sp.simplify(v) != 0 and v not in cands:
            cands.append(sp.expand(v))
    return cands


def _normalize_generator_scalar_kd(Xis, Phi, xs, u):
    candidates = []
    for Xi in Xis:
        if _is_constant_in(Xi, [*xs, u]) and not _is_zero_fast(Xi):
            candidates.append(sp.expand(Xi))
    for i, Xi in enumerate(Xis):
        ci = sp.expand(sp.diff(Xi, xs[i]))
        if sp.expand(Xi - ci * xs[i]) == 0 and _is_constant_in(ci, [*xs, u]) and not _is_zero_fast(ci):
            candidates.append(ci)
    dec_x = _decompose_affine_independent_action_scalar_kd(Xis, xs, u)
    if dec_x is not None:
        candidates.extend(_normalization_candidates_from_affine_action(*dec_x))
    dec_u = _decompose_affine_u_action_scalar_kd(Phi, xs, u)
    if dec_u is not None:
        a, b = dec_u
        if not _is_zero_fast(a):
            candidates.append(a)
        if not _is_zero_fast(b):
            candidates.append(b)
    scale = first_nonzero_scale(candidates)
    if scale is None:
        return [sp.expand(v) for v in Xis], sp.expand(Phi), sp.Integer(1)
    return [sp.expand(scale * Xi) for Xi in Xis], sp.expand(scale * Phi), scale


def _count_nonzero_entries_matrix(M):
    return sum(1 for v in list(M) if not _is_zero_fast(v))


def _count_nonzero_entries_vector(v):
    return sum(1 for x in list(v) if not _is_zero_fast(x))


def _matrix_offdiag_nonzero_count(M):
    rows, cols = M.shape
    count = 0
    for i in range(rows):
        for j in range(cols):
            if i != j and sp.simplify(M[i, j]) != 0:
                count += 1
    return count


def _affine_independent_action_complexity(M, bvec):
    return 5 * _matrix_offdiag_nonzero_count(M) + 2 * _count_nonzero_entries_matrix(M) + 2 * _count_nonzero_entries_vector(bvec) + sum(expr_complexity(v) for v in list(M) + list(bvec)) // 10


def _classify_affine_independent_action_type(M, bvec):
    if matrix_is_zero(M):
        return "translation"
    if matrix_is_diagonal(M) and matrix_is_zero(bvec):
        return "diagonal_scaling"
    if matrix_is_diagonal(M):
        return "diagonal_affine"
    return "general_affine"


def _match_priority_scalar_kd(match_data):
    kind = match_data["kind"]
    if kind == "translation_affine_scalar_kd":
        return 0
    if kind == "diagonal_scaling_affine_scalar_kd":
        return 1
    if kind == "affine_independent_affine_u_scalar_kd":
        M, bvec = match_data["M"], match_data["bvec"]
        cls = _classify_affine_independent_action_type(M, bvec)
        return 2 if cls == "diagonal_affine" else 3
    return 10


def _sum_abs_integer_like_coeffs(coeffs):
    total = 0
    for c in coeffs:
        c = sp.expand(c)
        if c.is_Integer:
            total += abs(int(c))
        else:
            total += expr_complexity(c)
    return total


def _match_secondary_complexity_scalar_kd(match_data):
    kind = match_data["kind"]
    if kind == "translation_affine_scalar_kd":
        return sum(expr_complexity(v) for v in match_data["xis"]) + expr_complexity(match_data["a"]) + expr_complexity(match_data["b"])
    if kind == "diagonal_scaling_affine_scalar_kd":
        return sum(expr_complexity(v) for v in match_data["scales"]) + expr_complexity(match_data["a"])
    if kind == "affine_independent_affine_u_scalar_kd":
        return _affine_independent_action_complexity(match_data["M"], match_data["bvec"]) + expr_complexity(match_data["a"]) + expr_complexity(match_data["beta"])
    return 10**6


def rank_symbolic_combination_matches_scalar_kd(matches, xs=None):
    def keyfun(m):
        secondary = _match_secondary_complexity_scalar_kd(m.match_data)
        if xs is not None and m.match_data["kind"] == "affine_independent_affine_u_scalar_kd":
            field = VectorFieldKD(tuple(xs), tuple(m.generator_xis))
            probe = probe_transport_coordinates_for_affine_distribution(DistributionKD(tuple(xs), (field,)))
            secondary += (0 if probe.success else 10**5) + 100 * ({"translation_linear":0, "diagonal_scaling_explicit":1, "linear_first_integrals":2, "affine_eigenfunctions":3, "flow_affine_linear_cross_section":4}.get(probe.method, 9)) + probe.complexity
        return (_match_priority_scalar_kd(m.match_data), secondary, len(m.basis_indices), _sum_abs_integer_like_coeffs(m.coefficients), m.basis_indices, m.coefficients)
    return sorted(matches, key=keyfun)




def _template_equations_translation_affine_scalar_kd(Xis, Phi, xs, u):
    k = len(xs)
    xi_syms = sp.symbols(f"xi_template_0:{k}")
    a = sp.Symbol("a_template")
    b = sp.Symbol("b_template")
    eq_exprs = [sp.expand(Xis[i] - xi_syms[i]) for i in range(k)]
    eq_exprs.append(sp.expand(Phi - a * u - b))
    equations = []
    for expr in eq_exprs:
        equations.extend(poly_zero_equations(expr, [*xs, u]))
    return equations, [*xi_syms, a, b]


def _template_equations_diagonal_scaling_affine_scalar_kd(Xis, Phi, xs, u):
    k = len(xs)
    c_syms = sp.symbols(f"c_template_0:{k}")
    a = sp.Symbol("a_template")
    eq_exprs = [sp.expand(Xis[i] - c_syms[i] * xs[i]) for i in range(k)]
    eq_exprs.append(sp.expand(Phi - a * u))
    equations = []
    for expr in eq_exprs:
        equations.extend(poly_zero_equations(expr, [*xs, u]))
    return equations, [*c_syms, a]


def _template_equations_affine_independent_affine_u_scalar_kd(Xis, Phi, xs, u):
    k = len(xs)
    M_syms = sp.Matrix(k, k, lambda i, j: sp.Symbol(f"M_template_{i}_{j}"))
    b_syms = sp.Matrix(k, 1, lambda i, j: sp.Symbol(f"b_template_{i}"))
    a = sp.Symbol("a_template")
    beta = sp.Symbol("beta_template")
    eq_exprs = [sp.expand(Xis[i] - sum(M_syms[i, j] * xs[j] for j in range(k)) - b_syms[i, 0]) for i in range(k)]
    eq_exprs.append(sp.expand(Phi - a * u - beta))
    equations = []
    for expr in eq_exprs:
        equations.extend(poly_zero_equations(expr, [*xs, u]))
    return equations, [*list(M_syms), *list(b_syms), a, beta]


def _construct_symbolic_combination_scalar_kd(subbasis, lambdas):
    Xis = [sp.expand(sum(lmb * vec[0][i] for lmb, vec in zip(lambdas, subbasis))) for i in range(len(subbasis[0][0]))]
    Phi = sp.expand(sum(lmb * vec[1] for lmb, vec in zip(lambdas, subbasis)))
    return Xis, Phi


def search_symbolic_linear_combinations_for_reduction_scalar_kd(eq_obj, basis_vectors, max_subset_size=3, try_translation=True, try_diagonal_scaling=True, try_affine=True, normalize=True, rank_results=True):
    xs = list(eq_obj.jet.xs)
    u = eq_obj.jet.u
    n = len(basis_vectors)
    subset_seen = set()
    seen = set()
    matches = []
    for r in range(1, min(max_subset_size, n) + 1):
        for idxs in combinations(range(n), r):
            subbasis = [basis_vectors[i] for i in idxs]
            subset_signature = tuple(sorted(_generator_signature(gen) for gen in subbasis))
            if subset_signature in subset_seen:
                continue
            subset_seen.add(subset_signature)
            jobs = _combination_jobs_for_subset(subbasis, try_translation=try_translation, try_diagonal_scaling=try_diagonal_scaling, try_affine=try_affine)
            if not jobs:
                continue
            lambdas = sp.symbols(f"lam0:{r}")
            Xis_raw, Phi_raw = _construct_symbolic_combination_scalar_kd(subbasis, lambdas)
            for anchor in range(r):
                for kind in jobs:
                    if kind == "translation":
                        template_eqs, template_unknowns = _template_equations_translation_affine_scalar_kd(Xis_raw, Phi_raw, xs, u)
                    elif kind == "scaling":
                        template_eqs, template_unknowns = _template_equations_diagonal_scaling_affine_scalar_kd(Xis_raw, Phi_raw, xs, u)
                    else:
                        template_eqs, template_unknowns = _template_equations_affine_independent_affine_u_scalar_kd(Xis_raw, Phi_raw, xs, u)
                    equations = list(template_eqs) + [sp.expand(lambdas[anchor] - 1)]
                    unknowns = list(lambdas) + list(template_unknowns)
                    try:
                        solset = sp.linsolve(equations, unknowns)
                    except Exception:
                        continue
                    if not solset:
                        continue
                    sol_tuple = substitute_free_parameters_zero(list(solset)[0])
                    sub = dict(zip(unknowns, sol_tuple))
                    coeff_vals = tuple(sp.expand(sub[lmb]) for lmb in lambdas)
                    if all(_is_zero_fast(v) for v in coeff_vals):
                        continue
                    Xis = [sp.expand(X.subs(sub)) for X in Xis_raw]
                    Phi = sp.expand(Phi_raw.subs(sub))
                    if normalize:
                        Xis, Phi, _ = _normalize_generator_scalar_kd(Xis, Phi, xs, u)
                    if kind == "translation":
                        match_data = match_translation_affine_scalar_kd(Xis, Phi, xs, u)
                    elif kind == "scaling":
                        match_data = match_diagonal_scaling_affine_scalar_kd(Xis, Phi, xs, u)
                    else:
                        match_data = match_affine_independent_affine_u_scalar_kd(Xis, Phi, xs, u)
                    if match_data is None:
                        continue
                    signature = (match_data["kind"], tuple(sp.srepr(sp.expand(v)) for v in Xis), sp.srepr(sp.expand(Phi)))
                    if signature in seen:
                        continue
                    seen.add(signature)
                    matches.append(SymbolicCombinationMatchScalarKD(tuple(i + 1 for i in idxs), anchor, coeff_vals, Xis, Phi, match_data))
    if rank_results:
        matches = rank_symbolic_combination_matches_scalar_kd(matches, xs=xs)
    return matches




def _solve_transport_coordinates_projectable_affine_k2(Xis, a, beta, xs):
    if len(xs) != 2:
        raise ValueError("This direct transport solver is only for k=2.")
    x, y = xs
    X1, X2 = map(sp.expand, Xis)
    zfun = sp.Function("Z")
    sfun = sp.Function("S")
    lfun = sp.Function("L")
    efun = sp.Function("E")
    z_pde = sp.Eq(X1 * sp.diff(zfun(x, y), x) + X2 * sp.diff(zfun(x, y), y), 0)
    z_sol = sp.pdsolve(z_pde)
    z_rhs = _extract_rhs_from_pde_solution(z_sol)
    z_expr = _extract_first_single_argument_undef_arg(z_rhs)
    if z_expr is None:
        raise NotImplementedError("Could not extract invariant from pdsolve output.")
    s_pde = sp.Eq(X1 * sp.diff(sfun(x, y), x) + X2 * sp.diff(sfun(x, y), y), 1)
    s_sol = sp.pdsolve(s_pde)
    s_rhs = replace_applied_undefs(_extract_rhs_from_pde_solution(s_sol), 0)
    l_pde = sp.Eq(X1 * sp.diff(lfun(x, y), x) + X2 * sp.diff(lfun(x, y), y) + a * lfun(x, y), 0)
    l_sol = sp.pdsolve(l_pde)
    l_rhs = replace_applied_undefs(_extract_rhs_from_pde_solution(l_sol), 1)
    e_pde = sp.Eq(X1 * sp.diff(efun(x, y), x) + X2 * sp.diff(efun(x, y), y) + beta * l_rhs, 0)
    e_sol = sp.pdsolve(e_pde)
    e_rhs = replace_applied_undefs(_extract_rhs_from_pde_solution(e_sol), 0)
    return (sp.expand(z_expr),), sp.expand(s_rhs), sp.expand(l_rhs), sp.expand(e_rhs)


from .utils import extract_rhs_from_pde_solution as _extract_rhs_from_pde_solution
from .utils import extract_first_single_argument_undef_arg as _extract_first_single_argument_undef_arg
from .utils import replace_applied_undefs


def reduce_scalar_by_explicit_transport_coordinates_kd(eq_obj, invariants, transverse, a=0, beta=0):
    jet = eq_obj.jet
    xs = list(jet.xs)
    k = len(xs)
    invariants = tuple(map(sp.expand, invariants))
    if len(invariants) == 0:
        zsyms = tuple()
    else:
        zsyms = tuple(sp.Symbol(f"z{i+1}", real=True) for i in range(len(invariants)))
    reducer = ReducedFunctionJetKD(xs, zsyms, invariants, jet.max_order)
    F0 = reducer.coord(()) if len(zsyms) == 0 else reducer.coord((0,) * len(zsyms))
    a = sp.sympify(a)
    beta = sp.sympify(beta)
    transverse = tuple(map(sp.expand, transverse))
    if sp.simplify(a) != 0:
        phase = sp.expand(sum(a * s for s in transverse)) if len(transverse) > 1 else sp.expand(a * transverse[0])
        U = sp.expand(sp.exp(phase) * F0 - beta / a)
    else:
        U = sp.expand(F0 + sum(beta * s for s in transverse))
    subs = {jet.u: U}
    for J in jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        subs[jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)
    expr = sp.expand(eq_obj.equation().subs(subs))
    keep_syms = list(zsyms) + [reducer.coord(M) for M in reducer._cache]
    expr = factor_prefactor_independent_of_symbols(expr, keep_syms)
    f = sp.Function("f")
    reduced_expr = reducer.to_function_notation(expr, f)
    reduced_eq = canonicalize_reduced_equation(sp.Eq(reduced_expr, 0))
    ansatz = sp.expand(U).subs(F0, f(*zsyms) if len(zsyms) > 0 else f())
    return ScalarReductionResultKD("projectable_affine_scalar_kd", invariants, transverse if len(transverse) != 1 else transverse[0], ansatz, reduced_expr, reduced_eq, f)


def reduce_scalar_by_translation_affine_kd(eq_obj, xis_const, a=0, b=0):
    xs = list(eq_obj.jet.xs)
    field = VectorFieldKD(tuple(xs), tuple(sp.sympify(v) for v in xis_const))
    coords = construct_characteristic_coordinates(DistributionKD(tuple(xs), (field,)))
    return reduce_scalar_by_explicit_transport_coordinates_kd(eq_obj, coords.invariants, coords.transverse, a=a, beta=b)


def reduce_scalar_by_diagonal_scaling_affine_kd(eq_obj, scales, a=0, b=0):
    xs = list(eq_obj.jet.xs)
    field = VectorFieldKD(tuple(xs), tuple(sp.sympify(scales[i]) * xs[i] for i in range(len(xs))))
    coords = construct_characteristic_coordinates(DistributionKD(tuple(xs), (field,)))
    return reduce_scalar_by_explicit_transport_coordinates_kd(eq_obj, coords.invariants, coords.transverse, a=a, beta=b)


def reduce_scalar_by_projectable_affine_generator_transport(eq_obj, Xis, Phi):
    xs = list(eq_obj.jet.xs)
    u = eq_obj.jet.u
    dec_u = _decompose_affine_u_action_scalar_kd(Phi, xs, u)
    if dec_u is None:
        raise ValueError("Dependent-variable action is not affine in u.")
    a, beta = dec_u
    dec_x = _decompose_affine_independent_action_scalar_kd(Xis, xs, u)
    if dec_x is not None:
        field = VectorFieldKD(tuple(xs), tuple(Xis))
        coords = construct_characteristic_coordinates(DistributionKD(tuple(xs), (field,)))
        return reduce_scalar_by_explicit_transport_coordinates_kd(eq_obj, coords.invariants, coords.transverse, a=a, beta=beta)
    if len(xs) == 2:
        invariants, s_expr, lam_expr, eta_expr = _solve_transport_coordinates_projectable_affine_k2(Xis, a, beta, xs)
        reducer = ReducedFunctionJetKD(xs, (sp.Symbol("z1", real=True),), invariants, eq_obj.jet.max_order)
        F0 = reducer.coord((0,))
        U = sp.expand((F0 - eta_expr) / lam_expr)
        subs = {eq_obj.jet.u: U}
        for J in eq_obj.jet.all_indices():
            if multiindex_sum(J) == 0:
                continue
            subs[eq_obj.jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)
        expr = sp.expand(eq_obj.equation().subs(subs))
        keep_syms = [reducer.coord(M) for M in reducer._cache] + [sp.Symbol("z1", real=True)]
        expr = factor_prefactor_independent_of_symbols(expr, keep_syms)
        f = sp.Function("f")
        reduced_expr = reducer.to_function_notation(expr, f)
        reduced_eq = canonicalize_reduced_equation(sp.Eq(reduced_expr, 0))
        return ScalarReductionResultKD("projectable_affine_scalar_k2_pdsolve", invariants, s_expr, sp.expand(U).subs(F0, f(sp.Symbol("z1", real=True))), reduced_expr, reduced_eq, f)
    raise NotImplementedError("General projectable-affine reduction is implemented for affine Xi in any k and arbitrary projectable Xi only for k=2.")




def _u_action_compatibility(a_list, beta_list):
    r = len(a_list)
    for i in range(r):
        for j in range(i + 1, r):
            if sp.simplify(beta_list[i] * a_list[j] - beta_list[j] * a_list[i]) != 0:
                return False
    return True




def reduce_scalar_by_translation_subalgebra_kd(eq_obj, xis_list, a_list=None, beta_list=None):
    xs = list(eq_obj.jet.xs)
    k = len(xs)
    r = len(xis_list)
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    fields = [VectorFieldKD(tuple(xs), tuple(sp.sympify(v) for v in xis)) for xis in xis_list]
    dist = DistributionKD(tuple(xs), tuple(fields))
    coords = find_coordinates_translation_subalgebra(dist)
    reducer = ReducedFunctionJetKD(xs, tuple(sp.Symbol(f"z{i+1}", real=True) for i in range(k - r)), coords.invariants, eq_obj.jet.max_order)
    F0 = reducer.coord(()) if k - r == 0 else reducer.coord((0,) * (k - r))
    U = _common_scalar_ansatz_from_transverse_parameters(F0, coords.transverse, a_list, beta_list)
    subs = {eq_obj.jet.u: U}
    for J in eq_obj.jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        subs[eq_obj.jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)
    expr = sp.expand(eq_obj.equation().subs(subs))
    keep = [reducer.coord(M) for M in reducer._cache] + [sp.Symbol(f"z{i+1}", real=True) for i in range(k - r)]
    expr = factor_prefactor_independent_of_symbols(expr, keep)
    f = sp.Function("f")
    reduced_expr = reducer.to_function_notation(expr, f)
    ansatz = sp.expand(U).subs(F0, f(*tuple(sp.Symbol(f"z{i+1}", real=True) for i in range(k - r))) if k - r > 0 else f())
    return ScalarMultiReductionResultKD("translation_subalgebra_scalar_kd", r, coords.invariants, coords.transverse, ansatz, reduced_expr, sp.Eq(reduced_expr, 0), f, tuple(xis_list))


def reduce_scalar_by_diagonal_scaling_subalgebra_kd(eq_obj, scales_list, a_list=None, beta_list=None):
    xs = list(eq_obj.jet.xs)
    k = len(xs)
    r = len(scales_list)
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    fields = [VectorFieldKD(tuple(xs), tuple(sp.sympify(scales[i]) * xs[i] for i in range(k))) for scales in scales_list]
    dist = DistributionKD(tuple(xs), tuple(fields))
    coords = find_coordinates_diagonal_scaling_subalgebra(dist)
    reducer = ReducedFunctionJetKD(xs, tuple(sp.Symbol(f"z{i+1}", real=True) for i in range(k - r)), coords.invariants, eq_obj.jet.max_order)
    F0 = reducer.coord(()) if k - r == 0 else reducer.coord((0,) * (k - r))
    U = _common_scalar_ansatz_from_transverse_parameters(F0, coords.transverse, a_list, beta_list)
    subs = {eq_obj.jet.u: U}
    for J in eq_obj.jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        subs[eq_obj.jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)
    expr = sp.expand(eq_obj.equation().subs(subs))
    keep = [reducer.coord(M) for M in reducer._cache] + [sp.Symbol(f"z{i+1}", real=True) for i in range(k - r)]
    expr = factor_prefactor_independent_of_symbols(expr, keep)
    f = sp.Function("f")
    reduced_expr = reducer.to_function_notation(expr, f)
    ansatz = sp.expand(U).subs(F0, f(*tuple(sp.Symbol(f"z{i+1}", real=True) for i in range(k - r))) if k - r > 0 else f())
    return ScalarMultiReductionResultKD("diagonal_scaling_subalgebra_scalar_kd", r, coords.invariants, coords.transverse, ansatz, reduced_expr, sp.Eq(reduced_expr, 0), f, tuple(scales_list))


def find_commuting_translation_subalgebras_scalar_kd(matches, k, max_generators=None):
    trans = [m for m in matches if m.match_data["kind"] == "translation_affine_scalar_kd"]
    if max_generators is None:
        max_generators = min(k, len(trans))
    out = []
    for r in range(1, max_generators + 1):
        for idxs in combinations(range(len(trans)), r):
            subset = [trans[i] for i in idxs]
            C = sp.Matrix([list(m.match_data["xis"]) for m in subset])
            if matrix_rank_symbolic(C) is None or matrix_rank_symbolic(C) < r:
                continue
            a_list = [m.match_data["a"] for m in subset]
            b_list = [m.match_data["b"] for m in subset]
            if not _u_action_compatibility(a_list, b_list):
                continue
            out.append((r, subset))
    return sorted(out, key=lambda item: (-item[0], sum(_match_secondary_complexity_scalar_kd(m.match_data) for m in item[1])))


def find_commuting_diagonal_scaling_subalgebras_scalar_kd(matches, k, max_generators=None):
    scal = [m for m in matches if m.match_data["kind"] == "diagonal_scaling_affine_scalar_kd"]
    if max_generators is None:
        max_generators = min(k, len(scal))
    out = []
    for r in range(1, max_generators + 1):
        for idxs in combinations(range(len(scal)), r):
            subset = [scal[i] for i in idxs]
            S = sp.Matrix([list(m.match_data["scales"]) for m in subset])
            if matrix_rank_symbolic(S) is None or matrix_rank_symbolic(S) < r:
                continue
            a_list = [m.match_data["a"] for m in subset]
            b_list = [0 for _ in subset]
            if not _u_action_compatibility(a_list, b_list):
                continue
            out.append((r, subset))
    return sorted(out, key=lambda item: (-item[0], sum(_match_secondary_complexity_scalar_kd(m.match_data) for m in item[1])))




def auto_reduce_symbolic_match_scalar_kd(eq_obj, match):
    md = match.match_data
    if md["kind"] == "translation_affine_scalar_kd":
        return reduce_scalar_by_translation_affine_kd(eq_obj, md["xis"], a=md["a"], b=md["b"])
    if md["kind"] == "diagonal_scaling_affine_scalar_kd":
        return reduce_scalar_by_diagonal_scaling_affine_kd(eq_obj, md["scales"], a=md["a"], b=0)
    if md["kind"] == "affine_independent_affine_u_scalar_kd":
        return reduce_scalar_by_projectable_affine_generator_transport(eq_obj, match.generator_xis, match.generator_phi)
    raise NotImplementedError(f"Unsupported scalar match kind: {md['kind']}")



from .frobenius import restricted_local_frobenius_engine, restricted_local_frobenius_atlas
from .verify import verify_reduction


def _generator_data_from_affine_match(match):
    md = match.match_data
    if md['kind'] == 'translation_affine_scalar_kd':
        return tuple(md['xis']), sp.sympify(md['a']), sp.sympify(md['b'])
    if md['kind'] == 'diagonal_scaling_affine_scalar_kd':
        xs = list(match.generator_xis)
        return tuple(xs), sp.sympify(md['a']), sp.Integer(0)
    if md['kind'] == 'affine_independent_affine_u_scalar_kd':
        return tuple(match.generator_xis), sp.sympify(md['a']), sp.sympify(md['beta'])
    raise NotImplementedError


def reduce_scalar_by_commuting_affine_subalgebra_kd(eq_obj, Xis_list, a_list=None, beta_list=None):
    """General commuting affine-subalgebra reduction using the restricted local engine."""
    xs = list(eq_obj.jet.xs)
    r = len(Xis_list)
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    if len(a_list) != r or len(beta_list) != r:
        raise ValueError('a_list and beta_list must match the number of generators.')
    fields = [VectorFieldKD(tuple(xs), tuple(sp.expand(v) for v in xis)) for xis in Xis_list]
    dist = DistributionKD(tuple(xs), tuple(fields))
    chart = restricted_local_frobenius_engine(dist)
    coords = chart
    k = len(xs)
    reducer = ReducedFunctionJetKD(xs, tuple(sp.Symbol(f'z{i+1}', real=True) for i in range(k - r)), coords.invariants, eq_obj.jet.max_order)
    F0 = reducer.coord(()) if k - r == 0 else reducer.coord((0,) * (k - r))
    U = _common_scalar_ansatz_from_transverse_parameters(F0, coords.transverse, a_list, beta_list)
    subs = {eq_obj.jet.u: U}
    for J in eq_obj.jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        subs[eq_obj.jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)
    expr = sp.expand(eq_obj.equation().subs(subs))
    keep = [reducer.coord(M) for M in reducer._cache] + [sp.Symbol(f'z{i+1}', real=True) for i in range(k - r)]
    expr = factor_prefactor_independent_of_symbols(expr, keep)
    f = sp.Function('f')
    reduced_expr = reducer.to_function_notation(expr, f)
    ansatz = sp.expand(U).subs(F0, f(*tuple(sp.Symbol(f'z{i+1}', real=True) for i in range(k - r))) if k - r > 0 else f())
    return ScalarMultiReductionResultKD('commuting_affine_subalgebra_scalar_kd', r, coords.invariants, coords.transverse, ansatz, reduced_expr, sp.Eq(reduced_expr, 0), f, tuple(Xis_list))


@dataclass
class ScalarAffineSubalgebraCandidateKD:
    kind: str
    match_indices: tuple
    generators: tuple
    dimension: int
    diagnostics: object | None
    score: tuple


def _fields_commute(fields):
    for i in range(len(fields)):
        for j in range(i + 1, len(fields)):
            br = fields[i].bracket(fields[j])
            if any(sp.simplify(c) != 0 for c in br.coeffs):
                return False
    return True


def find_commuting_affine_subalgebras_scalar_kd(matches, xs, max_generators=None):
    aff = [m for m in matches if m.match_data['kind'] in ('translation_affine_scalar_kd', 'diagonal_scaling_affine_scalar_kd', 'affine_independent_affine_u_scalar_kd')]
    n = len(aff)
    k = len(xs)
    if max_generators is None:
        max_generators = min(k, n)
    out = []
    for r in range(1, max_generators + 1):
        for idxs in combinations(range(n), r):
            subset = [aff[i] for i in idxs]
            fields = [VectorFieldKD(tuple(xs), tuple(m.generator_xis)) for m in subset]
            dist = DistributionKD(tuple(xs), tuple(fields))
            if not dist.is_commuting():
                continue
            if dist.rank() is None or dist.rank() < r:
                continue
            a_list = []
            b_list = []
            ok = True
            for m in subset:
                _, a, b = _generator_data_from_affine_match(m)
                a_list.append(a)
                b_list.append(b)
            if not _u_action_compatibility(a_list, b_list):
                continue
            coords, diag = construct_characteristic_coordinates_with_diagnostics(dist)
            atlas = restricted_local_frobenius_atlas(dist)
            best_chart = atlas.best()
            if coords is None or best_chart is None:
                continue
            complexity = sum(_match_secondary_complexity_scalar_kd(m.match_data) for m in subset)
            complexity += sum(expr_complexity(v) for v in best_chart.chart.invariants + best_chart.chart.transverse)
            complexity += len(best_chart.local_conditions) * 5
            score = (-r, complexity)
            out.append(ScalarAffineSubalgebraCandidateKD('commuting_affine_subalgebra_scalar_kd', tuple(idxs), tuple(subset), r, diag, score))
    return sorted(out, key=lambda c: c.score)


def auto_reduce_best_commuting_subalgebra_scalar_kd(eq_obj, matches, max_generators=None):
    xs = list(eq_obj.jet.xs)
    cands = find_commuting_affine_subalgebras_scalar_kd(matches, xs, max_generators=max_generators)
    if not cands:
        return None
    best = cands[0]
    Xis_list = []
    a_list = []
    beta_list = []
    for m in best.generators:
        xis, a, b = _generator_data_from_affine_match(m)
        Xis_list.append(xis)
        a_list.append(a)
        beta_list.append(b)
    return reduce_scalar_by_commuting_affine_subalgebra_kd(eq_obj, Xis_list, a_list=a_list, beta_list=beta_list)


def choose_best_symbolic_match_scalar_kd(eq_obj, matches):
    ranked = rank_symbolic_combination_matches_scalar_kd(matches, xs=list(eq_obj.jet.xs))
    return ranked[0] if ranked else None


def auto_reduce_best_symbolic_match_scalar_kd(eq_obj, matches):
    best = choose_best_symbolic_match_scalar_kd(eq_obj, matches)
    if best is None:
        return None
    return auto_reduce_symbolic_match_scalar_kd(eq_obj, best)




def _common_scalar_ansatz_from_transverse_parameters(F0, s_exprs, a_list, beta_list):
    a_list = [sp.sympify(v) for v in a_list]
    beta_list = [sp.sympify(v) for v in beta_list]
    s_exprs = list(map(sp.expand, s_exprs))
    if not _u_action_compatibility(a_list, beta_list):
        raise ValueError('Dependent affine actions are not mutually commuting.')
    if all(sp.simplify(a) == 0 for a in a_list):
        return sp.expand(F0 + sum(beta_list[j] * s_exprs[j] for j in range(len(s_exprs))))
    kappa = None
    for a, beta in zip(a_list, beta_list):
        if not _is_zero_fast(a):
            cand = sp.simplify(beta / a)
            if kappa is None:
                kappa = cand
            elif sp.simplify(cand - kappa) != 0:
                raise ValueError('Dependent affine actions are not compatible for common scalar reduction.')
        else:
            if sp.simplify(beta) != 0:
                raise ValueError('Dependent affine actions are not compatible for common scalar reduction.')
    phase = sp.expand(sum(a_list[j] * s_exprs[j] for j in range(len(s_exprs))))
    return sp.expand(sp.exp(phase) * F0 - kappa)


def reduce_scalar_by_frobenius_chart(eq_obj, chart, a_list=None, beta_list=None):
    chart = canonicalize_coordinate_chart(chart, eq_obj.jet.xs)
    """Default reduction backend using a Frobenius chart."""
    jet = eq_obj.jet
    xs = list(jet.xs)
    r = len(chart.transverse)
    if a_list is None:
        a_list = [0] * r
    if beta_list is None:
        beta_list = [0] * r
    if len(a_list) != r or len(beta_list) != r:
        raise ValueError('a_list and beta_list must match the number of transverse coordinates.')

    zsyms = tuple(sp.Symbol(f'z{i+1}', real=True) for i in range(len(chart.invariants)))
    reducer = ReducedFunctionJetKD(
        x_vars=xs,
        invariant_symbols=zsyms,
        invariant_exprs=chart.invariants,
        max_order=jet.max_order,
    )
    F0 = reducer.coord((0,) * len(zsyms)) if len(zsyms) > 0 else reducer.coord(())
    U = _common_scalar_ansatz_from_transverse_parameters(F0, chart.transverse, a_list, beta_list)

    subs = {jet.u: U}
    for J in jet.all_indices():
        if multiindex_sum(J) == 0:
            continue
        subs[jet.coord(J)] = _apply_multi_total_derivative(reducer, U, J)

    expr = sp.expand(eq_obj.equation().subs(subs))
    keep_syms = list(zsyms) + [reducer.coord(M) for M in reducer._cache]
    expr = factor_prefactor_independent_of_symbols(expr, keep_syms)

    f = sp.Function('f')
    reduced_expr = reducer.to_function_notation(expr, f)
    reduced_eq = canonicalize_reduced_equation(sp.Eq(reduced_expr, 0))
    ansatz = sp.expand(U).subs(F0, f(*zsyms) if len(zsyms) > 0 else f())

    return ScalarMultiReductionResultKD(
        kind='frobenius_backend_scalar_kd',
        subalgebra_dimension=r,
        invariants=tuple(chart.invariants),
        transverse_parameters=tuple(chart.transverse),
        ansatz=ansatz,
        reduced_expression=reduced_expr,
        reduced_equation=reduced_eq,
        reduced_function=f,
        generators=tuple(),
        reduced_unknown=f(*zsyms) if len(zsyms) > 0 else f(),
    )
