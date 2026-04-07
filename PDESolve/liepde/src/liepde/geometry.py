from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import sympy as sp

from .utils import expr_complexity, is_nonzero, is_zero, matrix_is_diagonal, matrix_is_zero, matrix_rank_symbolic, nullspace_basis_rows, right_inverse_columns
from .performance import cached_bracket_coeffs, cached_distribution_diagnostics


@dataclass
class VectorFieldKD:
    vars: tuple[sp.Symbol, ...]
    coeffs: tuple[sp.Expr, ...]

    def __post_init__(self):
        self.vars = tuple(self.vars)
        self.coeffs = tuple(sp.expand(c) for c in self.coeffs)
        if len(self.vars) != len(self.coeffs):
            raise ValueError("vars and coeffs must have the same length.")

    @property
    def dimension(self) -> int:
        return len(self.vars)

    def apply(self, expr: sp.Expr) -> sp.Expr:
        return sp.expand(sum(self.coeffs[i] * sp.diff(expr, self.vars[i]) for i in range(self.dimension)))

    def bracket(self, other: "VectorFieldKD") -> "VectorFieldKD":
        if self.vars != other.vars:
            raise ValueError("Vector fields must live on the same coordinate system.")
        vars_sig = tuple(str(v) for v in self.vars)
        coeffs = cached_bracket_coeffs(vars_sig, tuple(str(c) for c in self.coeffs), tuple(str(c) for c in other.coeffs))
        return VectorFieldKD(self.vars, tuple(sp.sympify(c) for c in coeffs))

    def jacobian(self) -> sp.Matrix:
        return sp.Matrix([[sp.diff(self.coeffs[i], self.vars[j]) for j in range(self.dimension)] for i in range(self.dimension)])

    def translation_vector(self) -> sp.Matrix | None:
        for c in self.coeffs:
            if any(v in c.free_symbols for v in self.vars):
                return None
        return sp.Matrix(self.coeffs)

    def affine_data(self) -> tuple[sp.Matrix, sp.Matrix] | None:
        k = self.dimension
        M = sp.zeros(k, k)
        b = sp.zeros(k, 1)
        for i, Xi in enumerate(self.coeffs):
            for j, xj in enumerate(self.vars):
                mij = sp.expand(sp.diff(Xi, xj))
                if any(v in mij.free_symbols for v in self.vars):
                    return None
                M[i, j] = mij
            residual = sp.expand(Xi - sum(M[i, j] * self.vars[j] for j in range(k)))
            if any(v in residual.free_symbols for v in self.vars):
                return None
            b[i, 0] = residual
        return M, b

    def is_translation(self) -> bool:
        return self.translation_vector() is not None

    def is_diagonal_scaling(self) -> bool:
        data = self.affine_data()
        if data is None:
            return False
        M, b = data
        return matrix_is_diagonal(M) and matrix_is_zero(b)

    def simplify(self) -> "VectorFieldKD":
        return VectorFieldKD(self.vars, tuple(sp.simplify(c) for c in self.coeffs))


@dataclass
class DistributionDiagnostics:
    rank: int | None
    commuting: bool
    affine: bool
    translation: bool
    diagonal_scaling: bool
    minor_columns: tuple[int, ...] | None


@dataclass
class CharacteristicCoordinatesResult:
    invariants: tuple[sp.Expr, ...]
    transverse: tuple[sp.Expr, ...]
    jacobian: sp.Expr
    method: str
    validity_conditions: tuple[sp.Expr, ...]


@dataclass
class DistributionKD:
    vars: tuple[sp.Symbol, ...]
    fields: tuple[VectorFieldKD, ...]

    def __post_init__(self):
        self.vars = tuple(self.vars)
        self.fields = tuple(self.fields)
        for field in self.fields:
            if field.vars != self.vars:
                raise ValueError("All vector fields must use the same variables.")

    @property
    def dimension(self) -> int:
        return len(self.vars)

    @property
    def size(self) -> int:
        return len(self.fields)

    def coefficient_matrix(self) -> sp.Matrix:
        return sp.Matrix([[field.coeffs[j] for j in range(self.dimension)] for field in self.fields])

    def rank(self) -> int | None:
        return matrix_rank_symbolic(self.coefficient_matrix())

    def is_commuting(self) -> bool:
        for i in range(self.size):
            for j in range(i + 1, self.size):
                br = self.fields[i].bracket(self.fields[j])
                if any(is_nonzero(c) for c in br.coeffs):
                    return False
        return True

    def affine_data(self) -> tuple[list[sp.Matrix], list[sp.Matrix]] | None:
        Ms: list[sp.Matrix] = []
        bs: list[sp.Matrix] = []
        for field in self.fields:
            data = field.affine_data()
            if data is None:
                return None
            M, b = data
            Ms.append(M)
            bs.append(b)
        return Ms, bs

    def _diagnostics_uncached(self) -> DistributionDiagnostics:
        rank = self.rank()
        A = self.coefficient_matrix()
        minor_cols = None
        if rank is not None and rank > 0:
            from itertools import combinations
            for cols in combinations(range(self.dimension), rank):
                sub = A[:, cols]
                try:
                    det = sp.simplify(sub.det()) if rank == self.size else None
                except Exception:
                    det = None
                if rank < self.size:
                    minor_cols = cols
                    break
                if det is None or det != 0:
                    minor_cols = cols
                    break
        aff = self.affine_data()
        translation = aff is not None and all(matrix_is_zero(M) for M in aff[0])
        diagonal = aff is not None and all(matrix_is_diagonal(M) and matrix_is_zero(b) for M, b in zip(*aff))
        return DistributionDiagnostics(
            rank=rank,
            commuting=self.is_commuting(),
            affine=aff is not None,
            translation=translation,
            diagonal_scaling=diagonal,
            minor_columns=minor_cols,
        )

    def diagnostics(self) -> DistributionDiagnostics:
        coeffs_sig = tuple(tuple(str(c) for c in f.coeffs) for f in self.fields)
        return cached_distribution_diagnostics(coeffs_sig, tuple(str(v) for v in self.vars))




@dataclass
class LieBracketRecord:
    i: int
    j: int
    bracket: VectorFieldKD


@dataclass
class ClosureResult:
    closed: bool
    structure_matrix_data: tuple | None
    residuals: tuple[VectorFieldKD, ...]


def distribution_commutator_table(distribution: DistributionKD) -> tuple[LieBracketRecord, ...]:
    out = []
    for i in range(distribution.size):
        for j in range(i + 1, distribution.size):
            out.append(LieBracketRecord(i, j, distribution.fields[i].bracket(distribution.fields[j])))
    return tuple(out)


def distribution_closure(distribution: DistributionKD) -> ClosureResult:
    """
    Test whether all Lie brackets lie in the span of the distribution and, when
    possible, return one set of coefficient functions expressing those brackets.
    """
    A = distribution.coefficient_matrix().T  # k x r
    residuals = []
    data = []
    for rec in distribution_commutator_table(distribution):
        b = sp.Matrix(rec.bracket.coeffs)
        try:
            sol, params = A.gauss_jordan_solve(b)
            if params.shape[0] > 0:
                sub = {params[i, 0]: 0 for i in range(params.shape[0])}
                sol = sp.Matrix([sp.expand(sol[i, 0].subs(sub)) for i in range(distribution.size)])
            else:
                sol = sp.Matrix([sp.expand(sol[i, 0]) for i in range(distribution.size)])
            recon = A * sol
            resid = sp.Matrix([sp.expand(recon[i, 0] - b[i, 0]) for i in range(distribution.dimension)])
            if any(is_nonzero(v) for v in resid):
                residuals.append(rec.bracket)
            else:
                data.append(((rec.i, rec.j), tuple(sol)))
        except Exception:
            residuals.append(rec.bracket)
    return ClosureResult(closed=len(residuals) == 0, structure_matrix_data=tuple(data) if len(residuals) == 0 else None, residuals=tuple(residuals))


def extract_commuting_subdistributions(distribution: DistributionKD, max_dim: int | None = None) -> tuple[DistributionKD, ...]:
    """
    Brute-force extraction of commuting subdistributions generated by subsets of
    the provided vector fields.
    """
    from itertools import combinations
    n = distribution.size
    if max_dim is None:
        max_dim = n
    out = []
    seen = set()
    for r in range(1, min(max_dim, n) + 1):
        for idxs in combinations(range(n), r):
            sub = DistributionKD(distribution.vars, tuple(distribution.fields[i] for i in idxs))
            if sub.is_commuting():
                sig = tuple(idxs)
                if sig not in seen:
                    seen.add(sig)
                    out.append(sub)
    out.sort(key=lambda d: (-d.size, 0 if d.diagnostics().translation else 1 if d.diagnostics().diagonal_scaling else 2))
    return tuple(out)


def optimal_system_style_selection(distribution: DistributionKD, max_dim: int | None = None) -> tuple[DistributionKD, ...]:
    """
    Heuristic 'optimal-system style' selector. This is not a full adjoint-orbit
    classification. It keeps the most reduction-friendly commuting subdistributions
    first, ranked by dimension, transport simplicity, and affine simplicity.
    """
    subs = list(extract_commuting_subdistributions(distribution, max_dim=max_dim))
    def score(sub: DistributionKD):
        diag = sub.diagnostics()
        aff = sub.affine_data()
        simplicity = 10**6
        if diag.translation:
            simplicity = 0
        elif diag.diagonal_scaling:
            simplicity = 1
        elif aff is not None:
            Ms, bs = aff
            nz = sum(sum(1 for v in list(M) if sp.simplify(v) != 0) for M in Ms)
            off = sum(sum(1 for i in range(M.rows) for j in range(M.cols) if i != j and sp.simplify(M[i, j]) != 0) for M in Ms)
            bz = sum(sum(1 for v in list(b) if sp.simplify(v) != 0) for b in bs)
            simplicity = 10 + 5 * off + 2 * nz + 2 * bz
        return (-sub.size, simplicity)
    subs.sort(key=score)
    return tuple(subs)
