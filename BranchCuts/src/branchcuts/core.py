from __future__ import annotations

from dataclasses import dataclass

from sympy import (
    S,
    Abs,
    Add,
    AtomicExpr,
    E,
    Eq,
    Expr,
    I,
    Mul,
    Piecewise,
    Symbol,
    acosh,
    asinh,
    atan,
    acos,
    acot,
    acoth,
    acsc,
    acsch,
    asec,
    asech,
    asin,
    assoc_legendre,
    atanh,
    beta,
    betainc,
    betainc_regularized,
    besseli,
    besselj,
    besselk,
    bessely,
    Ci,
    Chi,
    cos,
    elliptic_e,
    elliptic_f,
    elliptic_k,
    elliptic_pi,
    expint,
    factorial,
    floor,
    gamma,
    hankel1,
    hankel2,
    hyper,
    im,
    LambertW,
    lerchphi,
    li,
    log,
    loggamma,
    meijerg,
    pi,
    polylog,
    re,
    simplify,
    sin,
    sqrt,
    sympify,
)
from sympy.logic.boolalg import And, Boolean, BooleanFalse, BooleanTrue
from sympy.sets.contains import Contains


class _UndefinedType(AtomicExpr):
    """Symbolic undefined value distinct from NaN and ordinary variables."""

    is_commutative = True

    def _sympystr(self, printer):
        return "Undefined()"

    def __repr__(self):
        return "Undefined()"

    def __str__(self):
        return "Undefined()"


_UNDEFINED_SINGLETON = _UndefinedType()


def Undefined() -> _UndefinedType:
    """Return the singleton symbolic undefined value."""
    return _UNDEFINED_SINGLETON


def _to_bool(cond) -> Boolean | BooleanTrue | BooleanFalse:
    """Canonicalize a condition into a SymPy Boolean."""
    cond = sympify(cond)
    if cond is True:
        return S.true
    if cond is False:
        return S.false
    if isinstance(cond, (Boolean, BooleanTrue, BooleanFalse)):
        return cond
    raise TypeError(f"Condition must be a SymPy Boolean, got {type(cond)!r}: {cond!r}")


def _combine_conditions(*conds) -> Boolean | BooleanTrue | BooleanFalse:
    """Combine conditions with logical AND, simplifying trivial parts."""
    normalized = []
    for cond in conds:
        c = _to_bool(cond)
        if c == S.false:
            return S.false
        if c != S.true:
            normalized.append(c)
    if not normalized:
        return S.true
    if len(normalized) == 1:
        return normalized[0]
    combined = simplify(And(*normalized))
    if combined is True:
        return S.true
    if combined is False:
        return S.false
    return _to_bool(combined)


class RestrictedExpr(Expr):
    """
    RestrictedExpr(expr, cond) means

        expr, when cond holds
        Undefined(), otherwise.
    """

    is_commutative = True

    def __new__(cls, expr, cond):
        expr = sympify(expr)
        cond = _to_bool(cond)

        if expr == Undefined():
            return Undefined()

        if isinstance(expr, RestrictedExpr):
            expr, cond = expr.expr, _combine_conditions(expr.cond, cond)

        if cond == S.true:
            return expr
        if cond == S.false:
            return Undefined()

        return Expr.__new__(cls, expr, cond)

    @property
    def expr(self):
        return self.args[0]

    @property
    def cond(self):
        return self.args[1]

    def _sympystr(self, printer):
        return f"RestrictedExpr({printer.doprint(self.expr)}, {printer.doprint(self.cond)})"

    def to_piecewise(self, off_value=None):
        """Export as Piecewise; default off-condition value is Undefined()."""
        if off_value is None:
            off_value = Undefined()
        return Piecewise((self.expr, self.cond), (off_value, True))

    def doit(self, deep=True, **hints):
        expr = self.expr.doit(deep=deep, **hints) if hasattr(self.expr, "doit") else self.expr
        return restrict(expr, simplify(self.cond))

    def _eval_simplify(self, **kwargs):
        return simplify_restricted(self)

    def _combine_with_other(self, other, op):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        if isinstance(other, RestrictedExpr):
            return restrict(op(self.expr, other.expr), _combine_conditions(self.cond, other.cond))
        return restrict(op(self.expr, other), self.cond)

    def __add__(self, other):
        return self._combine_with_other(other, lambda a, b: Add(a, b))

    def __radd__(self, other):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        return restrict(Add(other, self.expr), self.cond)

    def __mul__(self, other):
        return self._combine_with_other(other, lambda a, b: Mul(a, b))

    def __rmul__(self, other):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        return restrict(Mul(other, self.expr), self.cond)

    def __sub__(self, other):
        return self._combine_with_other(other, lambda a, b: Add(a, -b))

    def __rsub__(self, other):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        return restrict(Add(other, -self.expr), self.cond)

    def __truediv__(self, other):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        if isinstance(other, RestrictedExpr):
            return restrict(self.expr / other.expr, _combine_conditions(self.cond, other.cond))
        return restrict(self.expr / other, self.cond)

    def __rtruediv__(self, other):
        other = sympify(other)
        if other == Undefined():
            return Undefined()
        return restrict(other / self.expr, self.cond)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise TypeError("RestrictedExpr does not support modular exponentiation")
        power = sympify(power)
        if power == Undefined():
            return Undefined()
        if isinstance(power, RestrictedExpr):
            return restrict(self.expr ** power.expr, _combine_conditions(self.cond, power.cond))
        return restrict(self.expr ** power, self.cond)

    def __rpow__(self, base):
        base = sympify(base)
        if base == Undefined():
            return Undefined()
        return restrict(base ** self.expr, self.cond)

    def __neg__(self):
        return restrict(-self.expr, self.cond)

    def __pos__(self):
        return self


def restrict(expr, cond):
    """Construct a RestrictedExpr, normalizing trivial cases."""
    expr = sympify(expr)
    cond = _to_bool(cond)

    if expr == Undefined():
        return Undefined()
    if isinstance(expr, RestrictedExpr):
        return RestrictedExpr(expr.expr, _combine_conditions(expr.cond, cond))
    if cond == S.true:
        return expr
    if cond == S.false:
        return Undefined()
    return RestrictedExpr(expr, cond)


def simplify_restricted(obj):
    """Simplify a RestrictedExpr or ordinary SymPy expression conservatively."""
    obj = sympify(obj)
    if obj == Undefined():
        return obj
    if isinstance(obj, RestrictedExpr):
        return restrict(simplify(obj.expr), simplify(obj.cond))
    return simplify(obj)


def propagate_undefined(expr):
    """
    Conservative post-processor that collapses Add/Mul/Pow containing
    Undefined() to Undefined().
    """
    expr = sympify(expr)
    if expr == Undefined():
        return expr
    if not getattr(expr, 'args', None):
        return expr
    new_args = tuple(propagate_undefined(arg) for arg in expr.args)
    if any(arg == Undefined() for arg in new_args) and isinstance(expr, (Add, Mul)):
        return Undefined()
    if expr.is_Pow and any(arg == Undefined() for arg in new_args):
        return Undefined()
    return expr.func(*new_args)


def apply_restrictions(func, *args):
    """Lift a function over possibly restricted arguments."""
    raw_args = []
    conds = []
    for arg in args:
        arg = sympify(arg)
        if arg == Undefined():
            return Undefined()
        if isinstance(arg, RestrictedExpr):
            raw_args.append(arg.expr)
            conds.append(arg.cond)
        else:
            raw_args.append(arg)
    result = func(*raw_args)
    cond = _combine_conditions(*conds) if conds else S.true
    return restrict(result, cond)


@dataclass(frozen=True)
class ConditionalJump:
    expr: Expr
    cond: Boolean | BooleanTrue | BooleanFalse
    note: str = ""

    def to_piecewise(self, off_value=None):
        if off_value is None:
            off_value = Undefined()
        return Piecewise((self.expr, self.cond), (off_value, True))


@dataclass(frozen=True)
class JumpTerm:
    expr: Expr
    cond: Boolean | BooleanTrue | BooleanFalse
    note: str = ""

    def as_restricted(self):
        return restrict(self.expr, self.cond)


@dataclass(frozen=True)
class BranchRule:
    name: str
    jump_terms: tuple[JumpTerm, ...]
    remarks: str = ""

    def restricted_terms(self):
        return tuple(term.as_restricted() for term in self.jump_terms)

    def as_piecewise(self, default=S.Zero):
        return Piecewise(*[(term.expr, term.cond) for term in self.jump_terms], (default, True))


# ---- cut-locus helpers -------------------------------------------------

def real_lt(z, a):
    return Contains(z, S.Reals) & (z < a)


def real_le(z, a):
    return Contains(z, S.Reals) & (z <= a)


def real_gt(z, a):
    return Contains(z, S.Reals) & (z > a)


def real_ge(z, a):
    return Contains(z, S.Reals) & (z >= a)


def real_between(z, a, b):
    return Contains(z, S.Reals) & (z > a) & (z < b)


def imag_axis_gt(z, a):
    return Eq(re(z), 0) & (im(z) > a)


def imag_axis_lt(z, a):
    return Eq(re(z), 0) & (im(z) < a)


def amplitude_vertical_seam(phi):
    return Eq(im(cos(phi) ** 2), 0) & (re(cos(phi) ** 2) < 0)


def elliptic_sqrt_seam(phi, m):
    return Eq(-im(m * sin(phi) ** 2), 0) & (1 - re(m * sin(phi) ** 2) < 0)


def elliptic_pi_pole_seam(phi, n):
    return Eq(-im(n * sin(phi) ** 2), 0) & (1 - re(n * sin(phi) ** 2) <= 0)


# ---- branch rules ------------------------------------------------------

def branch_rule_log(z):
    return BranchRule(
        name="log",
        jump_terms=(JumpTerm(2 * pi * I, real_lt(z, 0), "Principal branch cut on the negative real axis."),),
    )


def branch_rule_sqrt(z):
    return BranchRule(
        name="sqrt",
        jump_terms=(JumpTerm(2 * I * sqrt(-z), real_lt(z, 0), "Principal square root cut on the negative real axis."),),
    )


def branch_rule_power(z, a):
    return BranchRule(
        name="power",
        jump_terms=(
            JumpTerm(
                2 * I * sin(pi * a) * (-z) ** a,
                real_lt(z, 0) & ~Contains(a, S.Integers),
                "Principal z**a jump for non-integer exponent.",
            ),
        ),
    )


def branch_rule_acos(z):
    return BranchRule(
        name="acos",
        jump_terms=(
            JumpTerm(2 * I * acosh(z), real_gt(z, 1), "Right-hand cut."),
            JumpTerm(-2 * I * acosh(-z), real_lt(z, -1), "Left-hand cut."),
        ),
    )


def branch_rule_asin(z):
    return BranchRule(
        name="asin",
        jump_terms=(
            JumpTerm(-2 * I * acosh(z), real_gt(z, 1), "Right-hand cut."),
            JumpTerm(2 * I * acosh(-z), real_lt(z, -1), "Left-hand cut."),
        ),
    )


def branch_rule_atanh(z):
    return BranchRule(
        name="atanh",
        jump_terms=(
            JumpTerm(-pi * I, real_gt(z, 1), "Cut on (1, oo)."),
            JumpTerm(pi * I, real_lt(z, -1), "Cut on (-oo, -1)."),
        ),
    )


def branch_rule_acosh(z):
    return BranchRule(
        name="acosh",
        jump_terms=(
            JumpTerm(2 * I * acos(z), real_between(z, -1, 1), "Interior cut on (-1, 1)."),
            JumpTerm(2 * pi * I, real_le(z, -1), "Exterior cut on (-oo, -1]."),
        ),
    )



def branch_rule_atan(z):
    return BranchRule(
        name="atan",
        jump_terms=(
            JumpTerm(pi, imag_axis_gt(z, 1), "Upper imaginary-axis cut outside i."),
            JumpTerm(-pi, imag_axis_lt(z, -1), "Lower imaginary-axis cut outside -i."),
        ),
    )


def branch_rule_asinh(z):
    return BranchRule(
        name="asinh",
        jump_terms=(
            JumpTerm(
                2 * acosh(Abs(im(z))),
                Eq(re(z), 0) & (Abs(im(z)) > 1),
                "Principal cut on the imaginary axis outside [-I, I].",
            ),
        ),
    )

def branch_rule_acot(z):
    return BranchRule(
        name="acot",
        jump_terms=(
            JumpTerm(-pi, imag_axis_gt(z, 1), "Upper imaginary-axis cut outside i."),
            JumpTerm(pi, imag_axis_lt(z, -1), "Lower imaginary-axis cut outside -i."),
        ),
    )


def branch_rule_acoth(z):
    return BranchRule(
        name="acoth",
        jump_terms=(
            JumpTerm(-pi * I, real_between(z, 0, 1), "Mapped cut from atanh(1/z)."),
            JumpTerm(pi * I, real_between(z, -1, 0), "Mapped cut from atanh(1/z)."),
        ),
    )


def branch_rule_asec(z):
    return BranchRule(
        name="asec",
        jump_terms=(
            JumpTerm(2 * I * acosh(1 / z), real_between(z, 0, 1), "Mapped right-hand cut from acos(1/z)."),
            JumpTerm(-2 * I * acosh(-1 / z), real_between(z, -1, 0), "Mapped left-hand cut from acos(1/z)."),
        ),
    )


def branch_rule_acsc(z):
    return BranchRule(
        name="acsc",
        jump_terms=(
            JumpTerm(-2 * I * acosh(1 / z), real_between(z, 0, 1), "Mapped right-hand cut from asin(1/z)."),
            JumpTerm(2 * I * acosh(-1 / z), real_between(z, -1, 0), "Mapped left-hand cut from asin(1/z)."),
        ),
    )


def branch_rule_asech(z):
    return BranchRule(
        name="asech",
        jump_terms=(
            JumpTerm(2 * I * acos(1 / z), real_gt(z, 1), "Mapped cut from acosh(1/z)."),
            JumpTerm(2 * pi * I, real_ge(z, -1) & real_lt(z, 0), "Mapped logarithmic cut from acosh(1/z)."),
        ),
    )


def branch_rule_acsch(z):
    return BranchRule(
        name="acsch",
        jump_terms=(
            JumpTerm(2 * acosh(Abs(im(z)) / Abs(z)), Eq(re(z), 0) & (Abs(im(z)) < 1) & (Abs(im(z)) > 0), "Mapped imaginary-axis cut from acsch(z)=asinh(1/z)."),
        ),
    )


def branch_rule_expint_ei(z):
    return BranchRule(
        name="Ei",
        jump_terms=(JumpTerm(2 * pi * I, real_lt(z, 0), "Principal cut on the negative real axis."),),
    )


def branch_rule_ci(z):
    return BranchRule(
        name="Ci",
        jump_terms=(JumpTerm(2 * pi * I, real_lt(z, 0), "Principal cut on the negative real axis."),),
    )


def branch_rule_chi(z):
    return BranchRule(
        name="Chi",
        jump_terms=(JumpTerm(2 * pi * I, real_lt(z, 0), "Principal cut on the negative real axis."),),
    )


def branch_rule_li(z):
    return BranchRule(
        name="li",
        jump_terms=(JumpTerm(2 * pi * I, real_between(z, 0, 1), "Inherited from Ei(log(z)) across 0<z<1."),),
    )


def branch_rule_expint(z, nu):
    return BranchRule(
        name="expint",
        jump_terms=(
            JumpTerm(
                -2 * pi * I * (-z) ** (nu - 1) / gamma(nu),
                real_lt(z, 0),
                "Principal cut on the negative real axis for expint(nu, z).",
            ),
        ),
    )


def branch_rule_loggamma(z):
    return BranchRule(
        name="loggamma",
        jump_terms=(
            JumpTerm(
                2 * pi * I * floor(-z),
                real_lt(z, 0) & ~Contains(z, S.Integers),
                "Jump away from poles on the negative real axis.",
            ),
        ),
    )


def branch_rule_polylog(z, s):
    return BranchRule(
        name="polylog",
        jump_terms=(
            JumpTerm(
                2 * pi * I * log(z) ** (s - 1) / gamma(s),
                real_gt(z, 1) & ~Contains(s, S.Integers),
                "Generic principal cut for non-integer order.",
            ),
            JumpTerm(
                2 * pi * I * log(z) ** (s - 1) / factorial(s - 1),
                real_gt(z, 1) & Contains(s, S.PositiveIntegers),
                "Principal cut for positive integer order.",
            ),
        ),
    )


def branch_rule_lerchphi(z, s, a):
    return BranchRule(
        name="lerchphi",
        jump_terms=(
            JumpTerm(
                2 * pi * I * z ** (-a) * log(z) ** (s - 1) / gamma(s),
                real_gt(z, 1),
                "Principal cut at z>1.",
            ),
        ),
    )


def branch_rule_lambertw(z):
    return BranchRule(
        name="LambertW",
        jump_terms=(
            JumpTerm(
                LambertW(z) - LambertW(z, -1),
                real_between(z, -1 / E, 0),
                "Principal-sheet jump across the standard cut on (-1/e, 0).",
            ),
        ),
    )


def branch_rule_hyperu(z, a, b):
    return BranchRule(
        name="hyperu",
        jump_terms=(
            JumpTerm(
                -2 * pi * I * z ** (1 - b) * hyper([1 + a - b], [2 - b], z),
                real_lt(z, 0),
                "Confluent hypergeometric U cut on (-oo, 0].",
            ),
        ),
    )


def branch_rule_besselj(z, nu):
    return BranchRule(
        name="besselj",
        jump_terms=(
            JumpTerm(
                2 * I * sin(pi * nu) * besselj(nu, -z),
                real_lt(z, 0) & ~Contains(nu, S.Integers),
                "Noninteger-order principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_besseli(z, nu):
    return BranchRule(
        name="besseli",
        jump_terms=(
            JumpTerm(
                2 * I * sin(pi * nu) * besseli(nu, -z),
                real_lt(z, 0) & ~Contains(nu, S.Integers),
                "Noninteger-order principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_bessely(z, nu):
    return BranchRule(
        name="bessely",
        jump_terms=(
            JumpTerm(
                -2 * sin(pi * nu) * besselj(nu, -z),
                real_lt(z, 0) & ~Contains(nu, S.Integers),
                "Noninteger-order principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_besselk(z, nu):
    return BranchRule(
        name="besselk",
        jump_terms=(
            JumpTerm(
                -I * (2 * sin(pi * nu) * besselk(nu, -z) + pi * besseli(nu, -z)),
                real_lt(z, 0),
                "Principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_hankel1(z, nu):
    return BranchRule(
        name="hankel1",
        jump_terms=(
            JumpTerm(
                2 * I * sin(pi * nu) * hankel2(nu, -z),
                real_lt(z, 0),
                "Principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_hankel2(z, nu):
    return BranchRule(
        name="hankel2",
        jump_terms=(
            JumpTerm(
                -2 * I * sin(pi * nu) * hankel1(nu, -z),
                real_lt(z, 0),
                "Principal cut on the negative real axis.",
            ),
        ),
    )


def branch_rule_hyper(var, a_params, b_params):
    return BranchRule(
        name="hyper",
        jump_terms=(
            JumpTerm(
                2 * pi * I * (Mul(*[gamma(b) for b in b_params]) / Mul(*[gamma(a) for a in a_params]))
                * meijerg(((), tuple(1 - a for a in a_params)), ((0,), tuple(1 - b for b in b_params)), var),
                real_gt(var, 1),
                "Generic pFq jump for p=q+1 on the principal cut z>1, represented via Meijer G.",
            ),
        ),
    )



def branch_rule_betainc(z, a, b):
    return branch_rule_betainc_endpoint(z, a, b, sign=1, regularized=False)


def branch_rule_betainc_regularized(z, a, b):
    return branch_rule_betainc_endpoint(z, a, b, sign=1, regularized=True)


def branch_rule_elliptic_k(m):
    return BranchRule(
        name="elliptic_k",
        jump_terms=(
            JumpTerm((-2 * I) / sqrt(m) * elliptic_k(1 - 1 / m), real_gt(m, 1), "Principal cut on (1, oo)."),
        ),
    )


def branch_rule_elliptic_e_m(m):
    return BranchRule(
        name="elliptic_e",
        jump_terms=(
            JumpTerm(2 * I * sqrt(m) * (elliptic_e(1 - 1 / m) - elliptic_k(1 - 1 / m)), real_gt(m, 1), "Parameter cut on (1, oo)."),
        ),
    )


def branch_rule_elliptic_pi_m(m, n):
    return BranchRule(
        name="elliptic_pi",
        jump_terms=(
            JumpTerm(2 * I * sqrt(m) * elliptic_pi(n / (n - 1), 1 - 1 / m), real_gt(m, 1) & ~Eq(n, 1), "Parameter cut in m on (1, oo), away from n=1."),
        ),
    )


def branch_rule_elliptic_pi_n(n, m):
    return BranchRule(
        name="elliptic_pi",
        jump_terms=(
            JumpTerm(-pi * I * sqrt(n) / sqrt((n - 1) * (n - m)), real_gt(n, 1) & ~Eq(m, n), "Parameter cut in n on (1, oo), away from coincident pole m=n."),
        ),
    )


def branch_rule_assoc_legendre(z, nu, mu):
    return BranchRule(
        name="assoc_legendre",
        jump_terms=(
            JumpTerm(2 * I * sin(pi * mu) * assoc_legendre(nu, mu, z), real_between(z, -1, 1) & ~Contains(mu, S.Integers), "Principal cut on (-1, 1) for noninteger order."),
        ),
    )


def branch_rule_hyper_2f1(z, a, b, c):
    genericity = (re(c) > 0) & (re(c - a) > 0) & (re(c - b) > 0) & (re(a) > 0) & (re(b) > 0)
    return BranchRule(
        name="hyp2f1",
        jump_terms=(
            JumpTerm(
                2 * pi * I * (gamma(c) / (gamma(a) * gamma(b) * gamma(c - a) * gamma(c - b)))
                * (S.Exp1 ** (I * pi * (c - a - b))) * (z - 1) ** (c - a - b)
                * hyper([c - a, c - b], [c - a - b + 1], 1 - z),
                real_gt(z, 1) & genericity,
                "Gauss hypergeometric principal cut on (1, oo) under generic parameter conditions.",
            ),
        ),
    )


def branch_rule_elliptic_f_phi(phi, m):
    return BranchRule(
        name="elliptic_f",
        jump_terms=(
            JumpTerm(
                2 * elliptic_k(m),
                amplitude_vertical_seam(phi),
                "Amplitude seam where cos(phi)^2 crosses the negative real axis.",
            ),
        ),
    )


def branch_rule_elliptic_e_phi(phi, m):
    return BranchRule(
        name="elliptic_e",
        jump_terms=(
            JumpTerm(
                2 * elliptic_e(m),
                amplitude_vertical_seam(phi),
                "Amplitude seam where cos(phi)^2 crosses the negative real axis.",
            ),
        ),
    )


def branch_rule_elliptic_pi_phi(phi, n, m):
    seam_a = amplitude_vertical_seam(phi) & ~elliptic_pi_pole_seam(phi, n) & ~elliptic_sqrt_seam(phi, m)
    seam_b = elliptic_sqrt_seam(phi, m) & ~elliptic_pi_pole_seam(phi, n)
    seam_c = elliptic_pi_pole_seam(phi, n) & ~Eq(m, n)
    return BranchRule(
        name="elliptic_pi",
        jump_terms=(
            JumpTerm(2 * elliptic_pi(n, m), seam_a, "Amplitude seam from cos(phi)^2."),
            JumpTerm(2 * elliptic_pi(n, m), seam_b, "Square-root seam from 1 - m*sin(phi)^2."),
            JumpTerm(-pi * I * sqrt(n) / sqrt((n - 1) * (n - m)), seam_c, "Pole seam from 1 - n*sin(phi)^2 = 0."),
        ),
    )


def _betainc_jump_terms(var, a, b):
    return (
        JumpTerm(
            2 * I * sin(pi * a) * (-var) ** a / a * hyper([a, 1 - b], [a + 1], var),
            real_lt(var, 0) & ~Contains(a, S.Integers),
            "Incomplete beta cut on (-oo, 0) for noninteger a.",
        ),
        JumpTerm(
            2 * pi * I * var ** a * (S.Exp1 ** (I * pi * b)) / (gamma(1 - b) * gamma(a + b)) * (var - 1) ** b
            * hyper([1, a + b], [b + 1], 1 - var),
            real_gt(var, 1),
            "Incomplete beta continuation across (1, oo).",
        ),
    )


def branch_rule_betainc_endpoint(var, a, b, sign=1, regularized=False):
    terms = []
    factor = 1 / beta(a, b) if regularized else 1
    for term in _betainc_jump_terms(var, a, b):
        note = term.note
        if sign == -1:
            note = "Lower-endpoint contribution with opposite sign. " + note
        if regularized:
            note += " Regularized by division through beta(a,b)."
        terms.append(JumpTerm(sign * factor * term.expr, term.cond, note))
    return BranchRule(
        name="betainc_regularized" if regularized else "betainc",
        jump_terms=tuple(terms),
    )


def get_branch_rule(expr, z: Symbol):
    """Return a BranchRule for a supported expression in the variable z."""
    expr = sympify(expr)

    if expr.func == log and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_log(z)

    if expr.func == acos and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acos(z)

    if expr.func == asin and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_asin(z)

    if expr.func == atanh and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_atanh(z)

    if expr.func == acosh and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acosh(z)

    if expr.func == atan and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_atan(z)

    if expr.func == asinh and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_asinh(z)

    if expr.func == acot and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acot(z)

    if expr.func == acoth and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acoth(z)

    if expr.func == asec and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_asec(z)

    if expr.func == acsc and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acsc(z)

    if expr.func == asech and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_asech(z)

    if expr.func == acsch and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_acsch(z)

    if expr.is_Pow and expr.base == z:
        a = expr.exp
        if a == S.Half:
            return branch_rule_sqrt(z)
        return branch_rule_power(z, a)

    name = expr.func.__name__.lower()
    if name == 'ei' and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_expint_ei(z)
    if expr.func == Ci and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_ci(z)
    if expr.func == Chi and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_chi(z)
    if expr.func == li and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_li(z)
    if expr.func == expint and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_expint(z, expr.args[0])
    if expr.func == loggamma and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_loggamma(z)
    if expr.func == polylog and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_polylog(z, expr.args[0])
    if expr.func == lerchphi and len(expr.args) == 3 and expr.args[0] == z:
        return branch_rule_lerchphi(z, expr.args[1], expr.args[2])
    if expr.func == LambertW and len(expr.args) in {1, 2} and expr.args[0] == z and (len(expr.args) == 1 or expr.args[1] == 0):
        return branch_rule_lambertw(z)
    if name == 'hyperu' and len(expr.args) == 3 and expr.args[2] == z:
        return branch_rule_hyperu(z, expr.args[0], expr.args[1])
    if expr.func == besselj and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_besselj(z, expr.args[0])
    if expr.func == besseli and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_besseli(z, expr.args[0])
    if expr.func == bessely and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_bessely(z, expr.args[0])
    if expr.func == besselk and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_besselk(z, expr.args[0])
    if expr.func == hankel1 and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_hankel1(z, expr.args[0])
    if expr.func == hankel2 and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_hankel2(z, expr.args[0])
    if expr.func == betainc and len(expr.args) == 4 and expr.args[3] == z:
        return branch_rule_betainc_endpoint(z, expr.args[0], expr.args[1], sign=1, regularized=False)
    if expr.func == betainc and len(expr.args) == 4 and expr.args[2] == z:
        return branch_rule_betainc_endpoint(z, expr.args[0], expr.args[1], sign=-1, regularized=False)
    if expr.func == betainc_regularized and len(expr.args) == 4 and expr.args[3] == z:
        return branch_rule_betainc_endpoint(z, expr.args[0], expr.args[1], sign=1, regularized=True)
    if expr.func == betainc_regularized and len(expr.args) == 4 and expr.args[2] == z:
        return branch_rule_betainc_endpoint(z, expr.args[0], expr.args[1], sign=-1, regularized=True)
    if expr.func == elliptic_k and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_elliptic_k(z)
    if expr.func == elliptic_e and len(expr.args) == 1 and expr.args[0] == z:
        return branch_rule_elliptic_e_m(z)
    if expr.func == elliptic_f and len(expr.args) == 2 and expr.args[0] == z:
        return branch_rule_elliptic_f_phi(z, expr.args[1])
    if expr.func == elliptic_e and len(expr.args) == 2 and expr.args[0] == z:
        return branch_rule_elliptic_e_phi(z, expr.args[1])
    if expr.func == elliptic_pi and len(expr.args) == 2 and expr.args[1] == z:
        return branch_rule_elliptic_pi_m(z, expr.args[0])
    if expr.func == elliptic_pi and len(expr.args) == 2 and expr.args[0] == z:
        return branch_rule_elliptic_pi_n(z, expr.args[1])
    if expr.func == elliptic_pi and len(expr.args) == 3 and expr.args[1] == z:
        return branch_rule_elliptic_pi_phi(z, expr.args[0], expr.args[2])
    if expr.func == assoc_legendre and len(expr.args) == 3 and expr.args[2] == z:
        return branch_rule_assoc_legendre(z, expr.args[0], expr.args[1])
    if expr.func == hyper and len(expr.args) == 3 and expr.args[2] == z and len(expr.args[0]) == 2 and len(expr.args[1]) == 1:
        return branch_rule_hyper_2f1(z, expr.args[0][0], expr.args[0][1], expr.args[1][0])
    if expr.func == hyper and len(expr.args) == 3 and expr.args[2] == z and len(expr.args[0]) == len(expr.args[1]) + 1:
        return branch_rule_hyper(z, tuple(expr.args[0]), tuple(expr.args[1]))

    raise NotImplementedError(f"No branch rule registered for expression: {expr!r}")


def branch_jump(expr, z: Symbol, mode: str = 'piecewise'):
    """
    Compute the branch-cut jump representation for a supported expression.

    mode='piecewise'   -> Piecewise(..., (0, True))
    mode='restricted'  -> tuple of RestrictedExpr terms
    mode='terms'       -> tuple of JumpTerm objects
    """
    rule = get_branch_rule(expr, z)
    if mode == 'piecewise':
        return rule.as_piecewise(default=S.Zero)
    if mode == 'restricted':
        return rule.restricted_terms()
    if mode == 'terms':
        return rule.jump_terms
    raise ValueError("mode must be one of: 'piecewise', 'restricted', 'terms'")


IMPLEMENTED_RULE_NAMES = {
    'log',
    'sqrt',
    'power',
    'acos',
    'asin',
    'atanh',
    'acosh',
    'atan',
    'asinh',
    'acot',
    'acoth',
    'asec',
    'acsc',
    'asech',
    'acsch',
    'ei',
    'ci',
    'chi',
    'li',
    'expint',
    'loggamma',
    'polylog',
    'lerchphi',
    'lambertw',
    'hyperu',
    'betainc',
    'betainc_regularized',
    'elliptic_k',
    'elliptic_e',
    'elliptic_f',
    'elliptic_pi',
    'assoc_legendre',
    'hyp2f1',
    'besselj',
    'besseli',
    'bessely',
    'besselk',
    'hankel1',
    'hankel2',
    'hyper',
}
