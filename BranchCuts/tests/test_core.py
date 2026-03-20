import pytest
from sympy import S, I, Symbol, acos, acosh, asin, atanh, log, pi, sqrt
from sympy.sets.contains import Contains

from branchcuts import Undefined, RestrictedExpr, apply_restrictions, branch_jump, restrict


def test_restrict_true_false_normalization():
    x = Symbol('x')
    assert restrict(x + 1, S.true) == x + 1
    assert restrict(x + 1, S.false) == Undefined()


def test_nested_restrictions_flatten():
    x = Symbol('x')
    c1 = Contains(x, S.Reals)
    c2 = x > 0
    r = restrict(restrict(log(x), c1), c2)
    assert isinstance(r, RestrictedExpr)
    assert r.cond == (c1 & c2)
    assert r.expr == log(x)


def test_restricted_addition_combines_conditions():
    x = Symbol('x')
    c1 = Contains(x, S.Reals) & (x > 0)
    c2 = Contains(x, S.Reals) & (x >= 0)
    r1 = restrict(log(x), c1)
    r2 = restrict(sqrt(x), c2)
    r3 = r1 + r2
    assert isinstance(r3, RestrictedExpr)
    assert r3.expr == log(x) + sqrt(x)
    assert r3.cond == (Contains(x, S.Reals) & (x > 0))


def test_apply_restrictions_lifts_functions():
    x = Symbol('x')
    c = Contains(x, S.Reals) & (x >= 0)
    r = restrict(sqrt(x), c)
    out = apply_restrictions(log, r)
    assert isinstance(out, RestrictedExpr)
    assert out.expr == log(sqrt(x))
    assert out.cond == c


def test_to_piecewise_defaults_to_undefined():
    x = Symbol('x')
    c = x > 0
    pw = restrict(log(x), c).to_piecewise()
    assert 'Undefined()' in str(pw)


def test_branch_jump_log_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(log(z), z)
    assert '2*I*pi' in str(pw)
    assert 'Contains(z, Reals)' in str(pw)


def test_branch_jump_sqrt_restricted():
    z = Symbol('z', complex=True)
    terms = branch_jump(sqrt(z), z, mode='restricted')
    assert len(terms) == 1
    t = terms[0]
    assert isinstance(t, RestrictedExpr)
    assert t.expr == 2 * I * sqrt(-z)


def test_branch_jump_acos_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(acos(z), z)
    assert str(2 * I * acosh(z)) in str(pw)
    assert str(-2 * I * acosh(-z)) in str(pw)


def test_branch_jump_asin_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(asin(z), z)
    assert str(-2 * I * acosh(z)) in str(pw)
    assert str(2 * I * acosh(-z)) in str(pw)


def test_branch_jump_atanh_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(atanh(z), z)
    assert str(-I * pi) in str(pw) or str(-pi * I) in str(pw)
    assert str(I * pi) in str(pw) or str(pi * I) in str(pw)


def test_inventory_loads():
    from branchcuts import coverage_summary
    summary = coverage_summary()
    assert 'scipy.special' in summary
    assert summary['scipy.special']['total'] > 100
    assert summary['mpmath']['total'] > 100
    assert summary['sympy']['total'] > 50
from sympy import LambertW, Ci, expint, asec, acsc, acot, lerchphi, symbols, besselj


def test_branch_jump_lambertw_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(LambertW(z), z)
    assert 'LambertW(z) - LambertW(z, -1)' in str(pw)
    assert '-exp(-1)' in str(pw) or '-E**(-1)' in str(pw) or '-1/E' in str(pw)


def test_branch_jump_expint_piecewise():
    z = Symbol('z', complex=True)
    nu = Symbol('nu')
    pw = branch_jump(expint(nu, z), z)
    assert '-2*I*pi' in str(pw)
    assert 'gamma(nu)' in str(pw)


def test_branch_jump_besselj_piecewise():
    z = Symbol('z', complex=True)
    nu = Symbol('nu')
    pw = branch_jump(besselj(nu, z), z)
    assert 'sin(pi*nu)' in str(pw)
    assert 'Contains(nu, Integers)' in str(pw)

from sympy import atan, asinh, betainc, betainc_regularized, beta, elliptic_k, elliptic_e, elliptic_f, elliptic_pi, assoc_legendre, hyper, symbols


def test_branch_jump_atan_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(atan(z), z)
    assert 'pi' in str(pw)
    assert 'im(z) > 1' in str(pw)


def test_branch_jump_asinh_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(asinh(z), z)
    assert 'acosh(Abs(im(z)))' in str(pw)


def test_branch_jump_betainc_piecewise():
    z = Symbol('z', complex=True)
    a, b = symbols('a b')
    pw = branch_jump(betainc(a, b, 0, z), z)
    s = str(pw)
    assert 'hyper((a, 1 - b), (a + 1,), z)' in s
    assert 'gamma(1 - b)' in s


def test_branch_jump_betainc_regularized_piecewise():
    z = Symbol('z', complex=True)
    a, b = symbols('a b')
    pw = branch_jump(betainc_regularized(a, b, 0, z), z)
    assert 'beta(a, b)' in str(pw)


def test_branch_jump_elliptic_k_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(elliptic_k(z), z)
    assert 'elliptic_k(1 - 1/z)' in str(pw)


def test_branch_jump_elliptic_e_piecewise():
    z = Symbol('z', complex=True)
    pw = branch_jump(elliptic_e(z), z)
    assert 'elliptic_e(1 - 1/z) - elliptic_k(1 - 1/z)' in str(pw)


def test_branch_jump_elliptic_pi_parameter_piecewise():
    z = Symbol('z', complex=True)
    n = Symbol('n')
    pw = branch_jump(elliptic_pi(n, z), z)
    assert 'elliptic_pi(n/(n - 1), 1 - 1/z)' in str(pw)


def test_branch_jump_assoc_legendre_piecewise():
    z = Symbol('z', complex=True)
    nu, mu = symbols('nu mu')
    pw = branch_jump(assoc_legendre(nu, mu, z), z)
    assert 'assoc_legendre(nu, mu, z)' in str(pw)
    assert 'sin(pi*mu)' in str(pw)


def test_branch_jump_hyp2f1_piecewise():
    z = Symbol('z', complex=True)
    a, b, c = symbols('a b c')
    pw = branch_jump(hyper((a, b), (c,), z), z)
    s = str(pw)
    assert 'gamma(c)' in s and 'hyper((' in s
    assert '1 - z' in s and 're(c) > 0' in s


def test_branch_jump_betainc_lower_endpoint_piecewise():
    z = Symbol('z', complex=True)
    a, b, x = symbols('a b x')
    pw = branch_jump(betainc(a, b, z, x), z)
    s = str(pw)
    assert '-2*I' in s and 'sin(pi*a)' in s


def test_branch_jump_elliptic_f_amplitude_piecewise():
    z = Symbol('z', complex=True)
    m = Symbol('m')
    pw = branch_jump(elliptic_f(z, m), z)
    s = str(pw)
    assert '2*elliptic_k(m)' in s
    assert 'sin(re(z))*cos(re(z))*sinh(im(z))*cosh(im(z))' in s


def test_branch_jump_elliptic_e_amplitude_piecewise():
    z = Symbol('z', complex=True)
    m = Symbol('m')
    pw = branch_jump(elliptic_e(z, m), z)
    s = str(pw)
    assert '2*elliptic_e(m)' in s
    assert 'sin(re(z))*cos(re(z))*sinh(im(z))*cosh(im(z))' in s


def test_branch_jump_elliptic_pi_amplitude_piecewise():
    z = Symbol('z', complex=True)
    n, m = symbols('n m')
    pw = branch_jump(elliptic_pi(n, z, m), z)
    s = str(pw)
    assert 'elliptic_pi(n, m)' in s
    assert 'sqrt(n)' in s
from branchcuts import branch_cut_jumps, branch_cut_jumps_pairs, numeric_branch_cut_jumps


def test_branch_cut_jumps_pairs_log_of_affine():
    z = Symbol('z', complex=True)
    out = branch_cut_jumps_pairs(log(1 - z), z)
    assert 'Jump' in out and out['Jump']
    assert any('2*I*pi' in str(expr) and 'z > 1' in str(cond) for expr, cond in out['Jump'])


def test_branch_cut_jumps_restricted_composite():
    z = Symbol('z', complex=True)
    out = branch_cut_jumps(asec(1 - z) + acsc(2 * z) + sqrt(log(1 - z)), z, mode='restricted')
    assert 'Jump' in out and len(out['Jump']) >= 1
    assert any(isinstance(term, RestrictedExpr) for term in out['Jump'])


def test_numeric_branch_cut_jumps_log():
    z = Symbol('z', complex=True)
    out = numeric_branch_cut_jumps(log(z), [z], wprec=30)
    assert out
    jump = out[0]['Jump']
    expected = (2 * I * pi).evalf(30)
    assert abs(complex(jump) - complex(expected)) < 1e-6
from sympy import Derivative, elliptic_f, elliptic_pi


def test_branch_cut_jumps_pairs_derivative_sqrt():
    z = Symbol('z', complex=True)
    out = branch_cut_jumps_pairs(Derivative(sqrt(z), z), z)
    assert 'Jump' in out and out['Jump']
    assert any(str(expr) == str(-I/sqrt(-z)) for expr, cond in out['Jump'] if 'Contains(z, Reals)' in str(cond))


def test_numeric_branch_cut_jumps_elliptic_f_amplitude():
    z = Symbol('z', complex=True)
    out = numeric_branch_cut_jumps(elliptic_f(z, S(2)), [z], wprec=30)
    assert out
    best = min(out, key=lambda rec: abs(complex(rec['Jump']) - complex(rec['ExpectedJump'])))
    assert abs(complex(best['Jump']) - complex(best['ExpectedJump'])) < 3


def test_numeric_branch_cut_jumps_elliptic_pi_amplitude():
    z = Symbol('z', complex=True)
    out = numeric_branch_cut_jumps(elliptic_pi(S.Half, z, S(3)), [z], wprec=30)
    assert out
    best = min(out, key=lambda rec: abs(complex(rec['Jump']) - complex(rec['ExpectedJump'])))
    assert abs(complex(best['Jump']) - complex(best['ExpectedJump'])) < 2
