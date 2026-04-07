from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import sympy as sp


@dataclass
class CacheStats:
    bracket_cache_info: Any
    diagnostics_cache_info: Any
    prolongation_cache_info: Any


@lru_cache(maxsize=4096)
def cached_bracket_coeffs(vars_sig: tuple[str, ...], coeffs1_sig: tuple[str, ...], coeffs2_sig: tuple[str, ...]):
    vars = tuple(sp.sympify(v) for v in vars_sig)
    coeffs1 = tuple(sp.sympify(s) for s in coeffs1_sig)
    coeffs2 = tuple(sp.sympify(s) for s in coeffs2_sig)
    out = []
    for i in range(len(vars)):
        lhs = sum(coeffs1[j] * sp.diff(coeffs2[i], vars[j]) for j in range(len(vars)))
        rhs = sum(coeffs2[j] * sp.diff(coeffs1[i], vars[j]) for j in range(len(vars)))
        out.append(sp.srepr(sp.expand(lhs - rhs)))
    return tuple(out)


@lru_cache(maxsize=2048)
def cached_distribution_diagnostics(coeffs_sig: tuple[tuple[str, ...], ...], vars_sig: tuple[str, ...]):
    from .geometry import DistributionKD, VectorFieldKD
    vars = tuple(sp.sympify(v) for v in vars_sig)
    fields = tuple(VectorFieldKD(vars, tuple(sp.sympify(c) for c in row)) for row in coeffs_sig)
    return DistributionKD(vars, fields)._diagnostics_uncached()


@lru_cache(maxsize=1024)
def cached_prolongation_coefficients_scalar_kd(vars_sig: tuple[str, ...], dep_name: str, max_order: int, xis_sig: tuple[str, ...], phi_sig: str):
    from .pde import ScalarJetSpaceKD
    from .symmetry import _prolongation_coefficients_scalar_kd_uncached
    vars = tuple(sp.Symbol(v) for v in vars_sig)
    jet = ScalarJetSpaceKD(vars, dep_name=dep_name, max_order=max_order)
    xis = [sp.sympify(s) for s in xis_sig]
    phi = sp.sympify(phi_sig)
    coeffs = _prolongation_coefficients_scalar_kd_uncached(jet, xis, phi)
    return tuple((J, sp.srepr(v)) for J, v in coeffs.items())


def prolongation_coefficients_from_cache(jet, xis, phi):
    data = cached_prolongation_coefficients_scalar_kd(tuple(sp.srepr(v) for v in jet.xs), jet.dep_name, jet.max_order, tuple(sp.srepr(e) for e in xis), sp.srepr(phi))
    return {tuple(J): sp.sympify(v) for J, v in data}


def cache_stats() -> CacheStats:
    return CacheStats(
        bracket_cache_info=cached_bracket_coeffs.cache_info(),
        diagnostics_cache_info=cached_distribution_diagnostics.cache_info(),
        prolongation_cache_info=cached_prolongation_coefficients_scalar_kd.cache_info(),
    )


def clear_all_caches() -> None:
    cached_bracket_coeffs.cache_clear()
    cached_distribution_diagnostics.cache_clear()
    cached_prolongation_coefficients_scalar_kd.cache_clear()
