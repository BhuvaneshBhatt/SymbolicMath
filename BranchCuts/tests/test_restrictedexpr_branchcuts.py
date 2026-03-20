
from sympy import symbols, log, Contains, S
from branchcuts import RestrictedExpr, branch_cut_jumps

def test_restricted_log_positive_real_has_zero_jump():
    z = symbols("z", complex=True)
    expr = RestrictedExpr(log(z), Contains(z, S.Reals) & (z > 0))
    out = branch_cut_jumps(expr, z, mode="pairs")
    jumps = out["Jump"]
    # There should be no nonzero jump terms surviving the restriction.
    assert all(j[0] == 0 for j in jumps)

def test_restricted_log_negative_real_keeps_jump():
    z = symbols("z", complex=True)
    expr = RestrictedExpr(log(z), Contains(z, S.Reals) & (z < 0))
    out = branch_cut_jumps(expr, z, mode="pairs")
    jumps = out["Jump"]
    assert any(j[0] != 0 for j in jumps)
