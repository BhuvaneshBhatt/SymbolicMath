from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Sequence

import sympy as sp

from .geometry import CharacteristicCoordinatesResult, DistributionKD, VectorFieldKD
from .utils import (
    extract_first_single_argument_undef_arg,
    extract_rhs_from_pde_solution,
    expr_complexity,
    factor_prefactor_independent_of_symbols,
    matrix_is_diagonal,
    matrix_is_zero,
    matrix_rank_symbolic,
    nullspace_basis_rows,
    replace_applied_undefs,
    right_inverse_columns,
    substitute_free_parameters_zero,
)


@dataclass
class AffineTransportCoordinateProbe:
    success: bool
    method: str | None
    invariants: tuple[sp.Expr, ...] | None
    transverse_parameter: sp.Expr | tuple[sp.Expr, ...] | None
    complexity: int


def _jacobian_det(exprs: Sequence[sp.Expr], vars: Sequence[sp.Symbol]) -> sp.Expr:
    J = sp.Matrix([[sp.diff(expr, var) for var in vars] for expr in exprs])
    try:
        return sp.simplify(J.det())
    except Exception:
        return sp.expand(J.det())


def find_coordinates_translation_subalgebra(distribution: DistributionKD) -> CharacteristicCoordinatesResult:
    vars = list(distribution.vars)
    k = len(vars)
    C = sp.Matrix([list(field.translation_vector()) for field in distribution.fields])
    r = distribution.size
    rank = matrix_rank_symbolic(C)
    if rank is None or rank < r:
        raise ValueError("Translation generators are not linearly independent.")

    null_basis = nullspace_basis_rows(C)
    if len(null_basis) < k - r:
        raise ValueError("Could not construct enough invariants for translation subalgebra.")

    invariants = [sp.expand(sum(null_basis[i][j, 0] * vars[j] for j in range(k))) for i in range(k - r)]
    qcols = right_inverse_columns(C)
    transverse = [sp.expand(sum(qcols[j][i, 0] * vars[i] for i in range(k))) for j in range(r)]
    return _coordinate_result(vars, invariants, transverse, "translation_linear")


def find_coordinates_diagonal_scaling_subalgebra(distribution: DistributionKD) -> CharacteristicCoordinatesResult:
    vars = list(distribution.vars)
    k = len(vars)
    scales = []
    for field in distribution.fields:
        data = field.affine_data()
        if data is None:
            raise ValueError("Not an affine field.")
        M, b = data
        if not matrix_is_diagonal(M) or not matrix_is_zero(b):
            raise ValueError("Not a diagonal scaling field.")
        scales.append([sp.expand(M[i, i]) for i in range(k)])

    S = sp.Matrix(scales)
    r = distribution.size
    rank = matrix_rank_symbolic(S)
    if rank is None or rank < r:
        raise ValueError("Scaling generators are not linearly independent.")

    null_basis = nullspace_basis_rows(S)
    if len(null_basis) < k - r:
        raise ValueError("Could not construct enough invariants for scaling subalgebra.")

    invariants = []
    for i in range(k - r):
        v = null_basis[i]
        invariants.append(sp.expand(sp.prod(vars[j] ** sp.simplify(v[j, 0]) for j in range(k))))

    qcols = right_inverse_columns(S)
    transverse = [sp.expand(sum(qcols[j][i, 0] * sp.log(vars[i]) for i in range(k))) for j in range(r)]
    conditions = tuple(sp.expand(v) for v in vars)
    return _coordinate_result(vars, invariants, transverse, "diagonal_scaling_explicit", conditions=conditions)


def _augmented_affine_matrix(M, bvec):
    k = M.rows
    N = sp.zeros(k + 1, k + 1)
    for i in range(k):
        for j in range(k):
            N[i, j] = M[i, j]
        N[i, k] = bvec[i, 0]
    return N


def _find_linear_transport_coordinates_affine(M, bvec, xs):
    k = len(xs)
    MT = M.T
    stacked = MT.col_join(bvec.T)
    inv_rows = stacked.nullspace()
    if len(inv_rows) < k - 1:
        return None
    invariants = tuple(sp.expand(sum(inv_rows[i][j, 0] * xs[j] for j in range(k))) for i in range(k - 1))

    qsyms = sp.symbols(f"q0:{k}")
    eqs = [sp.Eq(sum(MT[i, j] * qsyms[j] for j in range(k)), 0) for i in range(k)]
    eqs.append(sp.Eq(sum(bvec[j, 0] * qsyms[j] for j in range(k)), 1))
    try:
        sol = sp.solve(eqs, qsyms, dict=True)
    except Exception:
        sol = []
    if not sol:
        return None
    sol = sol[0]
    free = sorted(list(set().union(*(sp.sympify(v).free_symbols for v in sol.values()))), key=lambda s: s.name)
    sub = {p: 0 for p in free}
    q = [sp.expand(sol[qsyms[i]].subs(sub)) for i in range(k)]
    s_expr = sp.expand(sum(q[i] * xs[i] for i in range(k)))
    return invariants, s_expr


def _find_affine_eigenfunction_coordinates(M, bvec, xs):
    k = len(xs)
    N = _augmented_affine_matrix(M, bvec)
    NT = N.T
    try:
        evects = NT.eigenvects()
    except Exception:
        return None
    forms = []
    for eigval, _, vecs in evects:
        if eigval.has(sp.I):
            continue
        for v in vecs:
            v = sp.Matrix(v)
            if v.shape != (k + 1, 1):
                continue
            l = v[:k, 0]
            d = v[k, 0]
            if all(sp.simplify(l[i]) == 0 for i in range(k)):
                continue
            w = sp.expand(sum(l[i] * xs[i] for i in range(k)) + d)
            forms.append((sp.expand(eigval), sp.Matrix(l), sp.expand(d), w))
    if len(forms) < k:
        return None
    chosen = None
    for idxs in combinations(range(len(forms)), k):
        L = sp.Matrix.vstack(*[forms[i][1].T for i in idxs])
        if sp.simplify(L.det()) != 0:
            chosen = [forms[i] for i in idxs]
            break
    if chosen is None:
        return None
    ref = None
    for i, (lam, _, _, _) in enumerate(chosen):
        if sp.simplify(lam) != 0:
            ref = i
            break
    if ref is None:
        return None
    lam_ref, _, _, w_ref = chosen[ref]
    s_expr = sp.expand(sp.log(w_ref) / lam_ref)
    invariants = []
    for i, (lam, _, _, w) in enumerate(chosen):
        if i == ref:
            continue
        if sp.simplify(lam) == 0:
            z = sp.expand(w)
        else:
            z = sp.expand(lam_ref * sp.log(w) - lam * sp.log(w_ref))
        invariants.append(z)
    return tuple(invariants), s_expr


def _candidate_cross_section_normals_affine(M, bvec):
    k = M.rows
    candidates = []
    seen = set()
    def add_vec(v):
        v = sp.Matrix(v)
        if v.shape == (1, k):
            v = v.T
        if v.shape != (k, 1):
            return
        if all(sp.simplify(v[i, 0]) == 0 for i in range(k)):
            return
        scale = None
        for i in range(k):
            if sp.simplify(v[i, 0]) != 0:
                scale = sp.simplify(1 / v[i, 0])
                break
        if scale is not None:
            v = sp.Matrix([sp.expand(scale * v[i, 0]) for i in range(k)])
        sig = tuple(sp.srepr(sp.expand(v[i, 0])) for i in range(k))
        if sig not in seen:
            seen.add(sig)
            candidates.append(v)
    for r in range(k):
        e = sp.zeros(k, 1)
        e[r, 0] = 1
        add_vec(e)
    for r in range(k):
        add_vec(M[r, :].T)
    for c in range(k):
        add_vec(M[:, c])
    add_vec(bvec)
    for i in range(k):
        for j in range(i + 1, k):
            v1 = sp.zeros(k, 1)
            v1[i, 0] = 1
            v1[j, 0] = 1
            add_vec(v1)
            v2 = sp.zeros(k, 1)
            v2[i, 0] = 1
            v2[j, 0] = -1
            add_vec(v2)
    return candidates


def _candidate_cross_section_offsets_affine(ell, M, bvec, xs):
    offsets = [sp.Integer(0)]
    k = len(xs)
    xstars = sp.symbols(f"xstar0:{k}")
    eqs = [sp.Eq(sum(M[i, j] * xstars[j] for j in range(k)) + bvec[i, 0], 0) for i in range(k)]
    try:
        sol = sp.solve(eqs, xstars, dict=True)
    except Exception:
        sol = []
    if sol:
        sol0 = sol[0]
        free = sorted(list(set().union(*(sp.sympify(v).free_symbols for v in sol0.values()))), key=lambda s: s.name)
        sub = {p: 0 for p in free}
        xstar = [sp.expand(sol0[xstars[i]].subs(sub)) for i in range(k)]
        offsets.append(sp.expand(-sum(ell[i, 0] * xstar[i] for i in range(k))))
    offsets.extend([sp.Integer(1), sp.Integer(-1)])
    out = []
    seen = set()
    for d in offsets:
        d = sp.expand(d)
        sig = sp.srepr(d)
        if sig not in seen:
            seen.add(sig)
            out.append(d)
    return out


def _find_flow_affine_linear_cross_section_coordinates(M, bvec, xs):
    k = len(xs)
    N = _augmented_affine_matrix(M, bvec)
    s = sp.Symbol("s_flow", real=True)
    y = sp.Matrix([*xs, 1])
    try:
        Eminus = (-s * N).exp()
    except Exception:
        return None
    y0 = sp.simplify(Eminus * y)
    x0 = y0[:k, 0]
    for ell in _candidate_cross_section_normals_affine(M, bvec):
        for d in _candidate_cross_section_offsets_affine(ell, M, bvec, xs):
            section_eq = sp.expand(sum(ell[i, 0] * x0[i] for i in range(k)) + d)
            try:
                sols = sp.solve(sp.Eq(section_eq, 0), s, dict=False)
            except Exception:
                sols = []
            if sols is None:
                sols = []
            if not isinstance(sols, (list, tuple)):
                sols = [sols]
            explicit_sols = [sp.simplify(sol) for sol in sols if s not in sp.sympify(sol).free_symbols]
            if not explicit_sols:
                continue
            s_expr = min(explicit_sols, key=expr_complexity)
            try:
                x0_sub = [sp.simplify(comp.subs(s, s_expr)) for comp in x0]
            except Exception:
                x0_sub = [sp.expand(comp.subs(s, s_expr)) for comp in x0]
            row = sp.Matrix([list(ell.T)])
            null_basis = row.nullspace()
            if len(null_basis) < k - 1:
                continue
            invariants = [sp.expand(sum(v[j, 0] * x0_sub[j] for j in range(k))) for v in null_basis[: k - 1]]
            return tuple(invariants), sp.expand(s_expr)
    return None


def solve_invariants_transverse_pdsolve_k2(field: VectorFieldKD):
    xs = list(field.vars)
    if len(xs) != 2:
        raise ValueError("pdsolve-based transport fallback is only for k=2.")
    x, y = xs
    X1, X2 = field.coeffs
    zfun = sp.Function("Z")
    sfun = sp.Function("S")
    z_sol = sp.pdsolve(sp.Eq(X1 * sp.diff(zfun(x, y), x) + X2 * sp.diff(zfun(x, y), y), 0))
    z_rhs = extract_rhs_from_pde_solution(z_sol)
    z_expr = extract_first_single_argument_undef_arg(z_rhs)
    if z_expr is None:
        raise NotImplementedError("Could not extract invariant from pdsolve output.")
    s_sol = sp.pdsolve(sp.Eq(X1 * sp.diff(sfun(x, y), x) + X2 * sp.diff(sfun(x, y), y), 1))
    s_rhs = replace_applied_undefs(extract_rhs_from_pde_solution(s_sol), 0)
    return (sp.expand(z_expr),), sp.expand(s_rhs)


def find_coordinates_single_affine(field: VectorFieldKD) -> CharacteristicCoordinatesResult:
    data = field.affine_data()
    if data is None:
        raise ValueError("Field is not affine.")
    M, bvec = data
    xs = list(field.vars)
    k = len(xs)

    coords = _find_linear_transport_coordinates_affine(M, bvec, xs)
    method = "linear_first_integrals"
    if coords is None:
        coords = _find_affine_eigenfunction_coordinates(M, bvec, xs)
        method = "affine_eigenfunctions"
    if coords is None:
        coords = _find_flow_affine_linear_cross_section_coordinates(M, bvec, xs)
        method = "flow_affine_linear_cross_section"
    if coords is None and k == 2:
        coords = solve_invariants_transverse_pdsolve_k2(field)
        method = "pdsolve_k2"
    if coords is None:
        raise NotImplementedError("Could not construct local characteristic coordinates for this affine field.")
    invariants, s_expr = coords
    return _coordinate_result(xs, invariants, (s_expr,), method)





def _expr_nonzero_conditions(expr):
    """Heuristic local validity conditions for an expression used in coordinates."""
    conds = []
    expr = sp.expand(expr)
    num, den = sp.fraction(sp.together(expr))
    den = sp.expand(den)
    if den != 1:
        conds.append(den)
    for node in sp.preorder_traversal(expr):
        if node.func == sp.log and len(node.args) == 1:
            conds.append(sp.expand(node.args[0]))
        elif isinstance(node, sp.Pow) and node.exp.is_negative:
            conds.append(sp.expand(node.base))
    out = []
    seen = set()
    for c in conds:
        sig = sp.srepr(c)
        if sig not in seen and sp.simplify(c) != 0:
            seen.add(sig)
            out.append(c)
    return tuple(out)


def _canonicalize_affine_like(expr, vars):
    expr = sp.factor_terms(sp.expand(expr))
    # remove additive constants only for purely transverse/invariant linear forms? keep them for now
    grad = [sp.expand(sp.diff(expr, v)) for v in vars]
    residual = sp.expand(expr - sum(grad[i] * vars[i] for i in range(len(vars))))
    if all(g.free_symbols.isdisjoint(set(vars)) for g in grad) and residual.free_symbols.isdisjoint(set(vars)):
        # normalize by first nonzero gradient entry
        for g in grad:
            if sp.simplify(g) != 0:
                scale = sp.simplify(1 / g)
                expr = sp.expand(scale * expr)
                break
    return sp.simplify(expr)


def _canonicalize_coordinate_system(invariants, transverse, vars):
    invs = tuple(_canonicalize_affine_like(expr, vars) for expr in invariants)
    trans = tuple(_canonicalize_affine_like(expr, vars) for expr in transverse)
    # de-duplicate purely repeated coordinates
    def dedupe(seq):
        out = []
        seen = set()
        for expr in seq:
            sig = sp.srepr(sp.expand(expr))
            if sig not in seen:
                seen.add(sig)
                out.append(expr)
        return tuple(out)
    return dedupe(invs), dedupe(trans)


def _coordinate_result(vars, invariants, transverse, method, conditions=()):
    invariants, transverse = _canonicalize_coordinate_system(invariants, transverse, vars)
    exprs = tuple(invariants) + tuple(transverse)
    jac = _jacobian_det(exprs, vars) if len(exprs) == len(vars) else sp.Integer(0)
    validity = list(map(sp.expand, conditions))
    for expr in exprs:
        validity.extend(_expr_nonzero_conditions(expr))
    if sp.simplify(jac) != 0:
        validity.append(sp.expand(jac))
    # dedupe validity conditions
    seen = set()
    validity2 = []
    for c in validity:
        sig = sp.srepr(sp.expand(c))
        if sig not in seen and sp.simplify(c) != 0:
            seen.add(sig)
            validity2.append(sp.expand(c))
    return CharacteristicCoordinatesResult(
        invariants=tuple(map(sp.expand, invariants)),
        transverse=tuple(map(sp.expand, transverse)),
        jacobian=sp.expand(jac),
        method=method,
        validity_conditions=tuple(validity2),
    )


@dataclass
class LocalEngineDiagnostics:
    rank: int | None
    commuting: bool
    affine: bool
    method_attempts: tuple[str, ...]
    chosen_method: str | None
    validity_conditions: tuple[sp.Expr, ...]


def _common_affine_data(distribution: DistributionKD):
    data = distribution.affine_data()
    if data is None:
        return None
    Ms, bs = data
    return list(Ms), list(bs)


def _find_common_linear_coordinates_affine_distribution(Ms, bs, xs, r):
    """Common linear first integrals and, when possible, linear transverse coordinates."""
    k = len(xs)
        # Compute linear first integrals shared by every affine generator in the distribution.
    blocks = []
    for M in Ms:
        blocks.append(M.T)
    for b in bs:
        blocks.append(b.T)
    A = sp.Matrix.vstack(*blocks) if blocks else sp.zeros(0, k)
    inv_basis = A.nullspace()
    if len(inv_basis) < k - r:
        return None
    invariants = tuple(sp.expand(sum(inv_basis[i][j, 0] * xs[j] for j in range(k))) for i in range(k - r))

        # Solve for transverse coordinates whose Lie derivatives isolate one generator at a time.
    qcols = []
    for j in range(r):
        rows = []
        rhs = []
        for m in range(r):
            rows.extend(list(Ms[m].T.tolist()))
            rhs.extend([0] * k)
            rows.append([bs[m][c, 0] for c in range(k)])
            rhs.append(1 if m == j else 0)
        Aeq = sp.Matrix(rows)
        beq = sp.Matrix(rhs)
        try:
            sol, params = Aeq.gauss_jordan_solve(beq)
        except Exception:
            return _coordinate_result(xs, invariants, (), 'common_linear_first_integrals')
        if params.shape[0] > 0:
            sub = {params[i, 0]: 0 for i in range(params.shape[0])}
            sol = sp.Matrix([sp.expand(sol[i, 0].subs(sub)) for i in range(k)])
        else:
            sol = sp.Matrix([sp.expand(sol[i, 0]) for i in range(k)])
        qcols.append(sol)
    transverse = tuple(sp.expand(sum(qcols[j][i, 0] * xs[i] for i in range(k))) for j in range(r))
    return _coordinate_result(xs, invariants, transverse, 'common_linear_first_integrals')


def _common_affine_eigenforms_distribution(Ms, bs, xs, r):
    """Find common affine eigenfunctions for commuting affine distributions."""
    k = len(xs)
    Ns = [_augmented_affine_matrix(Ms[i], bs[i]) for i in range(r)]
        # Diagonalize a generic linear combination first, then keep only vectors that remain eigenvectors for every generator.
    coeffs = [i + 1 for i in range(r)]
    A = sp.zeros(k + 1, k + 1)
    for i in range(r):
        A += coeffs[i] * Ns[i].T
    try:
        evects = A.eigenvects()
    except Exception:
        return None
    forms = []
    for eigval, _, vecs in evects:
        if eigval.has(sp.I):
            continue
        for v in vecs:
            v = sp.Matrix(v)
            if v.shape != (k + 1, 1):
                continue
                        # Each candidate must survive as an eigenvector for all augmented affine generators, not just the probe combination.
            lambdas = []
            ok = True
            for m in range(r):
                w = sp.simplify(Ns[m].T * v)
                lam = None
                for idx in range(k + 1):
                    if sp.simplify(v[idx, 0]) != 0:
                        lam = sp.simplify(w[idx, 0] / v[idx, 0])
                        break
                if lam is None:
                    ok = False
                    break
                if any(sp.simplify(w[idx, 0] - lam * v[idx, 0]) != 0 for idx in range(k + 1)):
                    ok = False
                    break
                lambdas.append(lam)
            if not ok:
                continue
            l = v[:k, 0]
            d = v[k, 0]
            if all(sp.simplify(l[i]) == 0 for i in range(k)):
                continue
            wexpr = sp.expand(sum(l[i] * xs[i] for i in range(k)) + d)
            forms.append((tuple(map(sp.expand, lambdas)), sp.Matrix(l), sp.expand(d), wexpr))
    if len(forms) < k:
        return None
    chosen = None
    for idxs in combinations(range(len(forms)), k):
        L = sp.Matrix.vstack(*[forms[i][1].T for i in idxs])
        if sp.simplify(L.det()) != 0:
            chosen = [forms[i] for i in idxs]
            break
    if chosen is None:
        return None
    Lambda = sp.Matrix([[chosen[j][0][m] for j in range(k)] for m in range(r)])
    if matrix_rank_symbolic(Lambda) is None or matrix_rank_symbolic(Lambda) < r:
        return None
    null_basis = Lambda.nullspace()
    if len(null_basis) < k - r:
        return None
    invariants = []
    for i in range(k - r):
        v = null_basis[i]
        inv = 0
        for j in range(k):
            inv += sp.expand(v[j, 0]) * sp.log(chosen[j][3])
        invariants.append(sp.expand(inv))
    qcols = right_inverse_columns(Lambda)
    transverse = []
    for j in range(r):
        s = 0
        for i in range(k):
            s += sp.expand(qcols[j][i, 0]) * sp.log(chosen[i][3])
        transverse.append(sp.expand(s))
    return _coordinate_result(xs, tuple(invariants), tuple(transverse), 'common_affine_eigenfunctions')


def _candidate_linear_cross_section_matrices(Ms, bs, xs, r):
    k = len(xs)
    normals = []
    seen = set()
    def add(v):
        v = sp.Matrix(v)
        if v.shape == (1, k):
            v = v.T
        if v.shape != (k, 1):
            return
        if all(sp.simplify(v[i, 0]) == 0 for i in range(k)):
            return
        for i in range(k):
            if sp.simplify(v[i, 0]) != 0:
                scale = sp.simplify(1 / v[i, 0])
                v = sp.Matrix([sp.expand(scale * v[i, 0]) for i in range(k)])
                break
        sig = tuple(sp.srepr(sp.expand(v[i, 0])) for i in range(k))
        if sig not in seen:
            seen.add(sig)
            normals.append(v)
    for i in range(k):
        e = sp.zeros(k, 1); e[i, 0] = 1; add(e)
    for M in Ms:
        for i in range(k):
            add(M[i, :].T)
        for j in range(k):
            add(M[:, j])
    for b in bs:
        add(b)
    # form candidate matrices of rank r
    mats = []
    for idxs in combinations(range(len(normals)), r):
        L = sp.Matrix.vstack(*[normals[i].T for i in idxs])
        rk = matrix_rank_symbolic(L)
        if rk is not None and rk == r:
            mats.append(L)
            if len(mats) > 20:
                break
    return mats


def _find_joint_flow_cross_section_coordinates_affine_distribution(Ms, bs, xs, r):
    """Commuting affine subalgebra fallback via joint flow and affine-linear cross-section."""
    k = len(xs)
    Ns = [_augmented_affine_matrix(Ms[i], bs[i]) for i in range(r)]
    # verify commuting augmented matrices
    for i in range(r):
        for j in range(i + 1, r):
            if any(sp.simplify(v) != 0 for v in list(Ns[i] * Ns[j] - Ns[j] * Ns[i])):
                return None
    s_syms = sp.symbols(f's0:{r}', real=True)
    y = sp.Matrix([*xs, 1])
    Nsum = sp.zeros(k + 1, k + 1)
    for j in range(r):
        Nsum += s_syms[j] * Ns[j]
    try:
        y0 = sp.simplify((-Nsum).exp() * y)
    except Exception:
        return None
    x0 = y0[:k, 0]
    for L in _candidate_linear_cross_section_matrices(Ms, bs, xs, r):
        # choose offsets 0 only for now, then fixed-point offsets if available could be added later
        eqs = [sp.Eq(sp.expand(sum(L[j, i] * x0[i] for i in range(k))), 0) for j in range(r)]
        try:
            sol = sp.solve(eqs, s_syms, dict=True)
        except Exception:
            sol = []
        if not sol:
            continue
        sol0 = sol[0]
        free = sorted(list(set().union(*(sp.sympify(v).free_symbols for v in sol0.values()))), key=lambda s: s.name)
        sub = {p: 0 for p in free}
        s_exprs = tuple(sp.expand(sol0[s_syms[j]].subs(sub)) for j in range(r))
        try:
            x0_sub = [sp.simplify(comp.subs({s_syms[j]: s_exprs[j] for j in range(r)})) for comp in x0]
        except Exception:
            x0_sub = [sp.expand(comp.subs({s_syms[j]: s_exprs[j] for j in range(r)})) for comp in x0]
        null_basis = L.nullspace()
        if len(null_basis) < k - r:
            continue
        invariants = []
        for i in range(k - r):
            v = null_basis[i]
            inv = sp.expand(sum(v[j, 0] * x0_sub[j] for j in range(k)))
            invariants.append(inv)
        return _coordinate_result(xs, tuple(invariants), s_exprs, 'commuting_affine_flow_cross_section')
    return None





def _common_constant_derivative_coordinate_functions_affine_distribution(Ms, bs, xs, r):
    """
    Construct candidate coordinate functions y_i with constant derivative vectors under all generators:
      - linear forms q^T x with q^T M_m = 0,
      - logs of common affine eigenforms w, where X_m(log w) = lambda_{m}.

    Returns list of tuples (mu_vec, gradient_vec, y_expr).
    """
    k = len(xs)
    candidates = []
    seen = set()

    def add_candidate(mu_vec, grad_vec, y_expr):
        mu_vec = tuple(sp.expand(v) for v in mu_vec)
        grad_vec = sp.Matrix(grad_vec)
        if grad_vec.shape == (1, k):
            grad_vec = grad_vec.T
        if grad_vec.shape != (k, 1):
            return
        if all(sp.simplify(grad_vec[i, 0]) == 0 for i in range(k)):
            return
        y_expr = sp.expand(y_expr)
        sig = (tuple(sp.srepr(v) for v in mu_vec), tuple(sp.srepr(sp.expand(grad_vec[i,0])) for i in range(k)), sp.srepr(y_expr))
        if sig not in seen:
            seen.add(sig)
            candidates.append((mu_vec, grad_vec, y_expr))

    # Family 1: linear forms annihilating all M_m^T
    rows = []
    for M in Ms:
        rows.extend(list(M.T.tolist()))
    A = sp.Matrix(rows) if rows else sp.zeros(0, k)
    for q in A.nullspace():
        mu = [sp.expand(sum(q[i, 0] * bs[m][i, 0] for i in range(k))) for m in range(r)]
        y = sp.expand(sum(q[i, 0] * xs[i] for i in range(k)))
        add_candidate(mu, q, y)

    # Family 2: common affine eigenforms
    Ns = [_augmented_affine_matrix(Ms[i], bs[i]) for i in range(r)]
    Acombo = sp.zeros(k + 1, k + 1)
    for i in range(r):
        Acombo += (i + 1) * Ns[i].T
    try:
        evects = Acombo.eigenvects()
    except Exception:
        evects = []
    for eigval, _, vecs in evects:
        if eigval.has(sp.I):
            continue
        for v in vecs:
            v = sp.Matrix(v)
            if v.shape != (k + 1, 1):
                continue
            l = v[:k, 0]
            d = v[k, 0]
            if all(sp.simplify(l[i]) == 0 for i in range(k)):
                continue
            lambdas = []
            ok = True
            for m in range(r):
                wv = sp.simplify(Ns[m].T * v)
                lam = None
                for idx in range(k + 1):
                    if sp.simplify(v[idx, 0]) != 0:
                        lam = sp.simplify(wv[idx, 0] / v[idx, 0])
                        break
                if lam is None or any(sp.simplify(wv[idx, 0] - lam * v[idx, 0]) != 0 for idx in range(k + 1)):
                    ok = False
                    break
                lambdas.append(sp.expand(lam))
            if not ok:
                continue
            wexpr = sp.expand(sum(l[i] * xs[i] for i in range(k)) + d)
            add_candidate(lambdas, l, sp.expand(sp.log(wexpr)))

    return candidates


def _common_affine_coordinates_distribution(Ms, bs, xs, r):
    """Construct coordinates from constant-derivative candidate functions."""
    k = len(xs)
    cands = _common_constant_derivative_coordinate_functions_affine_distribution(Ms, bs, xs, r)
    if len(cands) < k:
        return None
        # Select a full coordinate system whose gradients are independent and whose derivative data resolves the r generator directions.
    chosen = None
    for idxs in combinations(range(len(cands)), k):
        grads = [cands[i][1] for i in idxs]
        G = sp.Matrix.vstack(*[g.T for g in grads])
        if sp.simplify(G.det()) == 0:
            continue
        Lambda = sp.Matrix([[cands[i][0][m] for i in idxs] for m in range(r)])
        rk = matrix_rank_symbolic(Lambda)
        if rk is None or rk < r:
            continue
        chosen = [cands[i] for i in idxs]
        break
    if chosen is None:
        return None
    Lambda = sp.Matrix([[chosen[i][0][m] for i in range(k)] for m in range(r)])
    null_basis = Lambda.nullspace()
    if len(null_basis) < k - r:
        return None
    invariants = []
    for i in range(k - r):
        v = null_basis[i]
        inv = sp.expand(sum(v[j, 0] * chosen[j][2] for j in range(k)))
        invariants.append(inv)
    qcols = right_inverse_columns(Lambda)
    transverse = []
    for j in range(r):
        s = sp.expand(sum(qcols[j][i, 0] * chosen[i][2] for i in range(k)))
        transverse.append(s)
    return _coordinate_result(xs, tuple(invariants), tuple(transverse), 'commuting_affine_constant_derivative_coords')


def _commuting_affine_distribution_coordinates(distribution: DistributionKD):
    vars = list(distribution.vars)
    r = distribution.size
    data = _common_affine_data(distribution)
    if data is None:
        return None
    Ms, bs = data
        # Prefer the most explicit coordinate construction before falling back to weaker methods.
    coords = _common_affine_coordinates_distribution(Ms, bs, vars, r)
    if coords is not None:
        return coords
        coords = _find_common_linear_coordinates_affine_distribution(Ms, bs, vars, r)
    if coords is not None and len(coords.invariants) == len(vars) - r and len(coords.transverse) == r:
        return coords
        coords = _common_affine_eigenforms_distribution(Ms, bs, vars, r)
    if coords is not None:
        return coords
        coords = _find_joint_flow_cross_section_coordinates_affine_distribution(Ms, bs, vars, r)
    if coords is not None:
        return coords
    return None


def construct_characteristic_coordinates(distribution: DistributionKD):
    """
    Restricted local engine for commuting or affine distributions.

    Supported cases:
    - commuting translation subalgebras
    - commuting diagonal scaling subalgebras
    - commuting affine subalgebras with common linear/eigen/flow coordinates
    - single affine vector field
    - selected low-dimensional single-field commuting cases via pdsolve fallback
    """
    diag = distribution.diagnostics()
    if diag.translation and diag.commuting:
        return find_coordinates_translation_subalgebra(distribution)
    if diag.diagonal_scaling and diag.commuting:
        return find_coordinates_diagonal_scaling_subalgebra(distribution)
    if diag.affine and diag.commuting:
        coords = _commuting_affine_distribution_coordinates(distribution)
        if coords is not None:
            return coords
    if distribution.size == 1 and diag.affine:
        return find_coordinates_single_affine(distribution.fields[0])
    raise NotImplementedError(
        'Restricted local engine currently supports commuting translations, commuting diagonal scalings, commuting affine subalgebras with explicit local coordinates, and single affine fields.'
    )


def construct_characteristic_coordinates_with_diagnostics(distribution: DistributionKD):
    attempts = []
    diag = distribution.diagnostics()
    chosen = None
    result = None
    if diag.translation and diag.commuting:
        attempts.append('translation_linear')
    if diag.diagonal_scaling and diag.commuting:
        attempts.append('diagonal_scaling_explicit')
    if diag.affine and diag.commuting:
        attempts.extend(['common_linear_first_integrals', 'common_affine_eigenfunctions', 'commuting_affine_flow_cross_section'])
    if distribution.size == 1 and diag.affine:
        attempts.extend(['linear_first_integrals', 'affine_eigenfunctions', 'flow_affine_linear_cross_section', 'pdsolve_k2'])
    try:
        result = construct_characteristic_coordinates(distribution)
        chosen = result.method
    except Exception:
        result = None
    return result, LocalEngineDiagnostics(
        rank=diag.rank,
        commuting=diag.commuting,
        affine=diag.affine,
        method_attempts=tuple(attempts),
        chosen_method=chosen,
        validity_conditions=tuple() if result is None else result.validity_conditions,
    )


def probe_transport_coordinates_for_affine_distribution(distribution: DistributionKD) -> AffineTransportCoordinateProbe:
    result, diag = construct_characteristic_coordinates_with_diagnostics(distribution)
    if result is not None:
        cx = sum(expr_complexity(v) for v in result.invariants + result.transverse)
        return AffineTransportCoordinateProbe(
            success=True,
            method=result.method,
            invariants=result.invariants,
            transverse_parameter=result.transverse if len(result.transverse) != 1 else result.transverse[0],
            complexity=cx,
        )
    return AffineTransportCoordinateProbe(False, None, None, None, 10**6)
