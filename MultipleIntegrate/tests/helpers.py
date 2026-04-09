import os
import time
import sympy as sp
from sympy import oo, simplify


def sym_eq(result, expected, *, tol=1e-12):
    """Return True if two expressions are symbolically or numerically equal."""
    diff = simplify(sp.expand(result - expected))
    if diff == 0:
        return True
    try:
        val = complex(diff.evalf(30))
        return abs(val) < tol
    except Exception:
        return False



def assert_eq(result, expected, label=""):
    assert sym_eq(result, expected), (
        f"{label}\n  got:      {result}\n  expected: {expected}"
    )



def assert_diverges(result):
    """Accept common symbolic divergence outputs or unevaluated integrals."""
    if getattr(result, "is_infinite", False):
        return
    if result in {oo, -oo, sp.zoo, sp.oo + sp.I * sp.pi, -sp.oo + sp.I * sp.pi}:
        return
    assert isinstance(result, sp.Integral), f"Expected divergence marker, got {result!r}"


def run_with_soft_budget(func, seconds, *, label=""):
    """Run ``func`` and assert it finishes within a generous soft budget."""
    scale = float(os.environ.get("MI_PERF_BUDGET_SCALE", "1.0"))
    budget = float(seconds) * scale
    start = time.perf_counter()
    result = func()
    elapsed = time.perf_counter() - start
    assert elapsed <= budget, (
        f"{label}\n  elapsed: {elapsed:.3f}s\n  budget:  {budget:.3f}s"
    )
    return result, elapsed
