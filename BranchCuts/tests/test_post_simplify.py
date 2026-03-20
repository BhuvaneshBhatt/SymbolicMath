from sympy import Symbol, Contains, S, I, pi, log, asec
from branchcuts.post_simplify import simplify_branch_condition, rewrite_branch_expr, post_simplify_pairs


def test_affine_interval_simplification():
    z = Symbol("z", complex=True)
    cond = Contains(2*z, S.Reals) & (2*z > 0) & (2*z < 1)
    simp = simplify_branch_condition(cond, z)
    assert str(simp) == str(Contains(z, S.Reals) & (z > 0) & (z < S.Half))


def test_impossible_dual_log_real_conditions():
    z = Symbol("z", complex=True)
    cond = Contains(z, S.Reals) & (z > 1) & Contains(log(z - 1) + I*pi, S.Reals) & Contains(log(z - 1) - I*pi, S.Reals)
    simp = simplify_branch_condition(cond, z)
    assert simp is S.false


def test_rewrite_asech_asec_patterns():
    z = Symbol("z")
    assert rewrite_branch_expr(I*2*log(1)).subs(z, z) == I*2*log(1)  # smoke
    assert rewrite_branch_expr(I*asec(z).rewrite('acosh')) != I*asec(z).rewrite('acosh')


def test_duplicate_merge_exact():
    z = Symbol("z", complex=True)
    pairs = [(asec(2*z), Contains(2*z, S.Reals) & (2*z > 0) & (2*z < 1)), (asec(2*z), Contains(z, S.Reals) & (z > 0) & (z < S.Half))]
    simp = post_simplify_pairs(pairs, z)
    assert len(simp) == 1
from sympy import Not

def test_redundant_log_real_positive_interval():
    z = Symbol('z', complex=True)
    cond = Contains(z, S.Reals) & (z > 1) & (z < 2) & Contains(log(z - 1), S.Reals)
    simp = simplify_branch_condition(cond, z)
    assert str(simp) == str(Contains(z, S.Reals) & (z > 1) & (z < 2))


def test_impossible_shifted_log_real_on_positive_interval():
    z = Symbol('z', complex=True)
    cond = Contains(z, S.Reals) & (z > 1) & (z < 2) & Contains(log(z - 1) + I*pi, S.Reals)
    simp = simplify_branch_condition(cond, z)
    assert simp is S.false


def test_negated_affine_cut_predicate():
    z = Symbol('z', complex=True)
    cond = Not(Contains(1 - z, S.Reals) & (1 - z < 0)) & Contains(z, S.Reals)
    simp = simplify_branch_condition(cond, z)
    # Current dominant normalization should capture z <= 1.
    assert str(simp) == str(Contains(z, S.Reals) & (z <= 1))
