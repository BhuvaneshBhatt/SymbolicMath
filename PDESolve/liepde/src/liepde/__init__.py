"""Standalone Lie-symmetry PDE reduction toolkit."""

from .api import (
    LiePDEAnalysis,
    LiePDEDiagnostics,
    LiePDEFailure,
    LieSymmetrySolveResult,
    build_equation_object,
    classify_pde,
    compute_polynomial_symmetries,
    liepde,
    search_reductions_from_symmetries,
    solve_reduced_equation,
)
from .pde import infer_sympy_pde_order

__all__ = [
    "LiePDEAnalysis",
    "LiePDEDiagnostics",
    "LiePDEFailure",
    "LieSymmetrySolveResult",
    "build_equation_object",
    "classify_pde",
    "compute_polynomial_symmetries",
    "infer_sympy_pde_order",
    "liepde",
    "search_reductions_from_symmetries",
    "solve_reduced_equation",
]
