from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import sympy as sp

from .geometry import DistributionKD, VectorFieldKD, distribution_closure
from .frobenius import restricted_local_frobenius_engine
from .utils import expr_complexity, matrix_is_zero, matrix_is_diagonal


@dataclass(frozen=True)
class OptimalSubalgebraRepresentative:
    indices: tuple[int, ...]
    distribution: DistributionKD
    signature: tuple
    score: tuple
    kind: str



def _first_nonzero(items):
    for it in items:
        if sp.simplify(it) != 0:
            return sp.expand(it)
    return None


def normalize_vector_field_kd(field: VectorFieldKD) -> VectorFieldKD:
    """Normalize a vector field up to nonzero scalar multiple.

    Preference order:
      1. constant translation coefficients
      2. diagonal affine entries
      3. translation affine entries
      4. first remaining coefficient
    """
    coeffs = list(map(sp.expand, field.coeffs))
    scale_base = None

    # constant coefficients
    consts = [c for c in coeffs if c.free_symbols.isdisjoint(set(field.vars))]
    scale_base = _first_nonzero(consts)

    data = field.affine_data() if scale_base is None else None
    if scale_base is None and data is not None:
        M, b = data
        diags = [sp.expand(M[i, i]) for i in range(M.rows)]
        scale_base = _first_nonzero(diags)
        if scale_base is None:
            scale_base = _first_nonzero([sp.expand(v) for v in list(b)])
        if scale_base is None:
            scale_base = _first_nonzero([sp.expand(v) for v in list(M)])

    if scale_base is None:
        scale_base = _first_nonzero(coeffs)

    if scale_base is None:
        return field

    lam = sp.simplify(1 / scale_base)
    return VectorFieldKD(field.vars, tuple(sp.expand(lam * c) for c in coeffs))



def _matrix_invariants(M: sp.Matrix):
    try:
        charpoly = sp.Poly(M.charpoly().as_expr()).all_coeffs()
        charpoly_sig = tuple(sp.srepr(sp.expand(c)) for c in charpoly)
    except Exception:
        charpoly_sig = None
    try:
        rank = int(M.rank())
    except Exception:
        rank = None
    try:
        trace = sp.expand(M.trace())
    except Exception:
        trace = None
    try:
        det = sp.expand(M.det())
    except Exception:
        det = None
    offdiag = sum(1 for i in range(M.rows) for j in range(M.cols) if i != j and sp.simplify(M[i, j]) != 0)
    return rank, trace, det, offdiag, charpoly_sig


def vector_field_adjoint_signature(field: VectorFieldKD) -> tuple:
    """Heuristic signature for one-dimensional subalgebras.

    This is not a full adjoint-orbit invariant. It is designed to collapse
    obvious scalar multiples and many common affine duplicates.
    """
    f = normalize_vector_field_kd(field)
    data = f.affine_data()

    if f.is_translation():
        v = sp.Matrix(f.coeffs)
        nz = sum(1 for x in list(v) if sp.simplify(x) != 0)
        return ('translation', nz, tuple(sp.srepr(sp.expand(c)) for c in f.coeffs))

    if data is not None:
        M, b = data
        if matrix_is_diagonal(M) and matrix_is_zero(b):
            diag = tuple(sp.srepr(sp.expand(M[i, i])) for i in range(M.rows))
            nz = sum(1 for i in range(M.rows) if sp.simplify(M[i, i]) != 0)
            return ('diagonal_scaling', nz, diag)

        rank, trace, det, offdiag, charpoly_sig = _matrix_invariants(M)
        try:
            aug_rank = int(M.row_join(b).rank())
        except Exception:
            aug_rank = None
        return (
            'affine',
            rank,
            trace is not None and sp.srepr(trace),
            det is not None and sp.srepr(det),
            offdiag,
            aug_rank,
            charpoly_sig,
            tuple(sp.srepr(sp.expand(x)) for x in list(b)),
        )

    return ('general', tuple(sp.srepr(sp.expand(c)) for c in f.coeffs))


def _distribution_family(distribution: DistributionKD) -> str:
    diag = distribution.diagnostics()
    if diag.translation:
        return 'translation'
    if diag.diagonal_scaling:
        return 'diagonal_scaling'
    if diag.affine:
        return 'affine'
    return 'general'


def subalgebra_equivalence_signature(distribution: DistributionKD) -> tuple:
    """Heuristic signature for a small subalgebra.

    Uses generator signatures after normalization and sorting. For commuting
    affine families, also includes coarse distribution invariants.
    """
    fields = tuple(normalize_vector_field_kd(f) for f in distribution.fields)
    field_sigs = tuple(sorted(vector_field_adjoint_signature(f) for f in fields))
    fam = _distribution_family(distribution)
    rank = distribution.rank()
    commuting = distribution.is_commuting()
    closure = distribution_closure(distribution).closed
    try:
        chart = restricted_local_frobenius_engine(distribution)
        chart_sig = (
            len(chart.invariants),
            len(chart.transverse),
            tuple(sp.srepr(sp.expand(v)) for v in chart.validity_conditions),
        )
    except Exception:
        chart_sig = None
    return (distribution.size, rank, commuting, closure, fam, field_sigs, chart_sig)



def _vector_field_score(field: VectorFieldKD) -> tuple:
    sig = vector_field_adjoint_signature(field)
    kind = sig[0]
    fam_prio = 0 if kind == 'translation' else 1 if kind == 'diagonal_scaling' else 2 if kind == 'affine' else 3
    complexity = sum(expr_complexity(c) for c in field.coeffs)
    try:
        data = field.affine_data()
        if data is not None:
            M, b = data
            complexity += sum(expr_complexity(v) for v in list(M) + list(b))
    except Exception:
        pass
    return (fam_prio, complexity, tuple(sp.srepr(sp.expand(c)) for c in field.coeffs))


def _distribution_score(distribution: DistributionKD) -> tuple:
    fam = _distribution_family(distribution)
    fam_prio = 0 if fam == 'translation' else 1 if fam == 'diagonal_scaling' else 2 if fam == 'affine' else 3
    try:
        chart = restricted_local_frobenius_engine(distribution)
        chart_penalty = 0
        chart_complexity = sum(expr_complexity(v) for v in chart.invariants + chart.transverse)
        cond_penalty = 5 * len(chart.validity_conditions)
    except Exception:
        chart_penalty = 10**5
        chart_complexity = 10**5
        cond_penalty = 10**5
    coeff_complexity = sum(sum(expr_complexity(c) for c in f.coeffs) for f in distribution.fields)
    return (-distribution.size, fam_prio, chart_penalty, chart_complexity + cond_penalty + coeff_complexity)



def optimal_system_1d(distribution: DistributionKD, include_combinations: bool = True, max_coeff: int = 2) -> tuple[OptimalSubalgebraRepresentative, ...]:
    """Heuristic representative selection for one-dimensional subalgebras.

    It deduplicates obvious scalar multiples and simple affine-equivalent
    generators by using a normalized adjoint-style signature. When
    ``include_combinations`` is True, it also searches small integer linear
    combinations of the input basis fields.
    """
    candidates: list[VectorFieldKD] = list(distribution.fields)
    if include_combinations and distribution.size > 1:
        coeff_pool = range(-max_coeff, max_coeff + 1)
        for coeffs in product(coeff_pool, repeat=distribution.size):
            if all(c == 0 for c in coeffs):
                continue
            # skip pure basis vectors already present
            if sum(1 for c in coeffs if c != 0) <= 1:
                continue
            coeff_exprs = [sp.Integer(c) for c in coeffs]
            field = VectorFieldKD(
                distribution.vars,
                tuple(sp.expand(sum(coeff_exprs[j] * distribution.fields[j].coeffs[i] for j in range(distribution.size))) for i in range(distribution.dimension)),
            )
            if all(sp.simplify(c) == 0 for c in field.coeffs):
                continue
            candidates.append(field)

    reps = {}
    for idx, field in enumerate(candidates):
        norm = normalize_vector_field_kd(field)
        sig = vector_field_adjoint_signature(norm)
        dist = DistributionKD(distribution.vars, (norm,))
        score = _vector_field_score(norm)
        rep = OptimalSubalgebraRepresentative((idx,), dist, sig, score, sig[0])
        if sig not in reps or score < reps[sig].score:
            reps[sig] = rep

    return tuple(sorted(reps.values(), key=lambda r: r.score))


def optimal_system_commuting_subalgebras(distribution: DistributionKD, max_dim: int = 2) -> tuple[OptimalSubalgebraRepresentative, ...]:
    """Heuristic representative selection for small higher-dimensional commuting subalgebras.

    This enumerates commuting subsets of the supplied generators and selects
    one representative per heuristic equivalence signature.
    """
    n = distribution.size
    reps = {}
    for r in range(2, min(max_dim, n) + 1):
        for idxs in combinations(range(n), r):
            sub = DistributionKD(distribution.vars, tuple(distribution.fields[i] for i in idxs))
            if not sub.is_commuting():
                continue
            if not distribution_closure(sub).closed:
                continue
            sig = subalgebra_equivalence_signature(sub)
            score = _distribution_score(sub)
            rep = OptimalSubalgebraRepresentative(idxs, sub, sig, score, _distribution_family(sub))
            if sig not in reps or score < reps[sig].score:
                reps[sig] = rep
    return tuple(sorted(reps.values(), key=lambda r: r.score))


def choose_optimal_reduction_candidates(distribution: DistributionKD, max_commuting_dim: int = 2):
    """Return preferred one-dimensional representatives followed by higher-dimensional commuting ones."""
    one_d = optimal_system_1d(distribution)
    higher = optimal_system_commuting_subalgebras(distribution, max_dim=max_commuting_dim)
    return one_d, higher
