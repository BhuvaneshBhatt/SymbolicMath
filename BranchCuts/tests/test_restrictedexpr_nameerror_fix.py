from sympy import symbols, log, S
from sympy.sets.contains import Contains
from branchcuts import RestrictedExpr, branch_cut_jumps

def test_restrictedexpr_branch_cut_jumps_runs_without_nameerror():
    z = symbols('z', complex=True)
    expr = RestrictedExpr(log(z), Contains(z, S.Reals) & (z > 0))
    out = branch_cut_jumps(expr, z, mode='pairs')
    assert 'Jump' in out
