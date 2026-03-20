from sympy import symbols, log, S
from sympy.sets.contains import Contains
from branchcuts import RestrictedExpr, branch_cut_jumps

def test_restrictedexpr_positive_real_log_runs():
    z = symbols("z", complex=True)
    expr = RestrictedExpr(log(z), Contains(z, S.Reals) & (z > 0))
    out = branch_cut_jumps(expr, z, mode="pairs")
    assert "Jump" in out
    assert all(term[0] == 0 for term in out["Jump"])
