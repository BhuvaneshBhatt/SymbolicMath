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
