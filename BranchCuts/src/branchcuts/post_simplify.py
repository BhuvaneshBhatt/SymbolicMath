from __future__ import annotations


def _validate_pair_sequence(pairs, func_name="pair-processing function"):
    if isinstance(pairs, dict):
        hint = ""
        if "Jump" in pairs:
            hint = " Did you mean to pass result['Jump'] from branch_cut_jumps(..., mode='pairs')?"
        raise TypeError(
            f"{func_name} expected a list/tuple of (expr, condition) pairs, got dict.{hint}"
        )
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(
            f"{func_name} expected a list/tuple of (expr, condition) pairs, got {type(pairs).__name__}."
        )
    for i, item in enumerate(pairs):
        if not (isinstance(item, (list, tuple)) and len(item) == 2):
            raise TypeError(
                f"{func_name} expected items of the form (expr, condition), but item {i} is {item!r}."
            )
    return pairs

def _validate_restricted_sequence(exprs, func_name="restricted-expression function"):
    if isinstance(exprs, dict):
        hint = ""
        if "Jump" in exprs:
            hint = " Did you mean to pass result['Jump'] or use mode='restricted'?"
        raise TypeError(
            f"{func_name} expected a list/tuple of RestrictedExpr-like expressions, got dict.{hint}"
        )
    if not isinstance(exprs, (list, tuple)):
        raise TypeError(
            f"{func_name} expected a list/tuple of RestrictedExpr-like expressions, got {type(exprs).__name__}."
        )
    return exprs


from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from sympy import (
    S, I, pi, And, Or, Not, simplify, sympify,
    acosh, asec, asech
)
from sympy.core.relational import Relational, StrictLessThan, StrictGreaterThan, LessThan, GreaterThan, Equality
from sympy.logic.boolalg import Boolean
from sympy.sets.contains import Contains

from .core import RestrictedExpr, restrict
from .computation import clean_pairs

Pair = tuple[object, object]


@dataclass(frozen=True)
class _Bound:
    value: object
    strict: bool


def _is_real_constant(expr) -> bool:
    expr = sympify(expr)
    return bool(expr.is_real and not expr.free_symbols)


def _linear_affine(expr, z):
    expr = sympify(expr).expand()
    if expr.has(z):
        a = expr.coeff(z)
        b = simplify(expr - a*z)
        if not b.has(z) and _is_real_constant(a) and _is_real_constant(b) and a != 0:
            return simplify(a), simplify(b)
    return None


def _contains_affine_real(cond, z):
    return isinstance(cond, Contains) and cond.args[1] == S.Reals and _linear_affine(cond.args[0], z) is not None


def _contains_real_impossible(cond, real_required=False):
    if not isinstance(cond, Contains) or cond.args[1] != S.Reals:
        return False
    target = sympify(cond.args[0])
    if real_required:
        try:
            _, im_part = target.as_real_imag()
            im_part = simplify(im_part)
            if im_part.is_real and not im_part.free_symbols and im_part != 0:
                return True
        except Exception:
            pass
    return False


def _normalize_relational_to_var(rel, z):
    rel = sympify(rel)
    if not isinstance(rel, Relational) or isinstance(rel, Equality):
        return None
    lhs = simplify(rel.lhs - rel.rhs)
    lin = _linear_affine(lhs, z)
    if lin is None:
        return None
    a, b = lin
    thresh = simplify(-b / a)
    if a.is_negative:
        if isinstance(rel, StrictLessThan): kind = '>'
        elif isinstance(rel, LessThan): kind = '>='
        elif isinstance(rel, StrictGreaterThan): kind = '<'
        else: kind = '<='
    else:
        if isinstance(rel, StrictLessThan): kind = '<'
        elif isinstance(rel, LessThan): kind = '<='
        elif isinstance(rel, StrictGreaterThan): kind = '>'
        else: kind = '>='
    return kind, thresh


def _merge_lower(old: _Bound | None, new: _Bound) -> _Bound:
    if old is None:
        return new
    if simplify(new.value - old.value).is_positive:
        return new
    if simplify(old.value - new.value).is_positive:
        return old
    return _Bound(old.value, old.strict or new.strict)


def _merge_upper(old: _Bound | None, new: _Bound) -> _Bound:
    if old is None:
        return new
    if simplify(new.value - old.value).is_negative:
        return new
    if simplify(old.value - new.value).is_negative:
        return old
    return _Bound(old.value, old.strict or new.strict)


def _not_affine_bound(piece, z):
    piece = sympify(piece)
    if piece.func != Not or len(piece.args) != 1:
        return None
    inner = piece.args[0]
    conjuncts = list(And.make_args(inner)) if inner.func == And else [inner]
    contains_aff = None
    rel = None
    for c in conjuncts:
        if isinstance(c, Contains) and c.args[1] == S.Reals and _linear_affine(c.args[0], z) is not None:
            contains_aff = c.args[0]
        elif isinstance(c, Relational) and not isinstance(c, Equality):
            rel = c
    if contains_aff is None or rel is None:
        return None
    aff = contains_aff
    if simplify(rel.lhs - aff) == 0 and _is_real_constant(rel.rhs):
        if isinstance(rel, StrictLessThan): return ('>=', rel.rhs)
        if isinstance(rel, LessThan): return ('>', rel.rhs)
        if isinstance(rel, StrictGreaterThan): return ('<=', rel.rhs)
        if isinstance(rel, GreaterThan): return ('<', rel.rhs)
    if simplify(rel.rhs - aff) == 0 and _is_real_constant(rel.lhs):
        if isinstance(rel, StrictLessThan): return ('<=', rel.lhs)
        if isinstance(rel, LessThan): return ('<', rel.lhs)
        if isinstance(rel, StrictGreaterThan): return ('>=', rel.lhs)
        if isinstance(rel, GreaterThan): return ('>', rel.lhs)
    return None


def _extract_log_real_test(piece):
    """Return (arg, sign) for Contains(log(arg) +/- I*pi, Reals), else None.

    sign = 0  -> Contains(log(arg), Reals)
    sign = +1 -> Contains(log(arg) + I*pi, Reals)
    sign = -1 -> Contains(log(arg) - I*pi, Reals)
    """
    if not isinstance(piece, Contains) or piece.args[1] != S.Reals:
        return None
    target = sympify(piece.args[0])
    if target.func.__name__ == 'log':
        return (target.args[0], 0)
    if target.is_Add:
        lg = None
        for a in target.args:
            if getattr(a, 'func', None) and a.func.__name__ == 'log':
                lg = a
                break
        if lg is not None:
            rest = simplify(target - lg)
            if rest == I*pi:
                return (lg.args[0], +1)
            if rest == -I*pi:
                return (lg.args[0], -1)
    return None


def _affine_interval_sign(arg, z, lower, upper):
    lin = _linear_affine(arg, z)
    if lin is None:
        return None
    a, b = lin
    vals = []
    for bd in (lower, upper):
        if bd is not None:
            vals.append(simplify(a*bd.value + b))
    if not vals:
        return None
    if all(v.is_positive for v in vals):
        return 'positive'
    if all(v.is_negative for v in vals):
        return 'negative'
    if any(v.is_zero for v in vals):
        return 'touches_zero'
    return None


def _redundant_or_impossible_log_real_test(piece, z, lower, upper, real_required):
    lt = _extract_log_real_test(piece)
    if lt is None or not real_required:
        return None
    arg, sign = lt
    implied = _affine_interval_sign(arg, z, lower, upper)
    if implied == 'positive':
        if sign == 0:
            return 'redundant'
        return 'impossible'
    if implied == 'negative':
        if sign == 0:
            return 'impossible'
    return None


def _interval_atoms(lower, upper):
    atoms = []
    if lower is not None:
        atoms.append(('>', lower.value) if lower.strict else ('>=', lower.value))
    if upper is not None:
        atoms.append(('<', upper.value) if upper.strict else ('<=', upper.value))
    return tuple(atoms)


def simplify_branch_condition(cond, z):
    cond = sympify(cond)
    if cond in (S.true, True):
        return S.true
    if cond in (S.false, False):
        return S.false

    pieces = list(And.make_args(cond)) if cond.func == And else [cond]
    lower = None
    upper = None
    real_required = False
    extras = []
    log_tests = []

    for piece in pieces:
        piece = sympify(piece)
        if _contains_affine_real(piece, z):
            real_required = True
            continue
        rb = _normalize_relational_to_var(piece, z)
        if rb is None:
            nb = _not_affine_bound(piece, z)
            if nb is not None:
                rb = nb
        if rb is not None:
            kind, thresh = rb
            if kind == '>':
                lower = _merge_lower(lower, _Bound(thresh, True)); real_required = True; continue
            if kind == '>=':
                lower = _merge_lower(lower, _Bound(thresh, False)); real_required = True; continue
            if kind == '<':
                upper = _merge_upper(upper, _Bound(thresh, True)); real_required = True; continue
            if kind == '<=':
                upper = _merge_upper(upper, _Bound(thresh, False)); real_required = True; continue
        lt = _extract_log_real_test(piece)
        if lt is not None:
            log_tests.append(lt)
        extras.append(piece)

    if lower is not None and upper is not None:
        diff = simplify(upper.value - lower.value)
        if diff.is_negative or (diff == 0 and (lower.strict or upper.strict)):
            return S.false

    if real_required:
        seen = {}
        for arg, sign in log_tests:
            seen.setdefault(str(arg), set()).add(sign)
        for s in seen.values():
            if +1 in s and -1 in s:
                return S.false

    filtered = []
    for piece in extras:
        if _contains_real_impossible(piece, real_required=real_required):
            return S.false
        verdict = _redundant_or_impossible_log_real_test(piece, z, lower, upper, real_required)
        if verdict == 'redundant':
            continue
        if verdict == 'impossible':
            return S.false
        filtered.append(piece)

    out = []
    if real_required:
        out.append(Contains(z, S.Reals))
    if lower is not None:
        out.append(z > lower.value if lower.strict else z >= lower.value)
    if upper is not None:
        out.append(z < upper.value if upper.strict else z <= upper.value)
    out.extend(filtered)
    if not out:
        return S.true
    try:
        return simplify(And(*out))
    except Exception:
        return And(*out)


def rewrite_branch_expr(expr):
    expr = sympify(expr)

    def _rw(e):
        if e.func == acosh and len(e.args) == 1:
            arg = e.args[0]
            num, den = arg.as_numer_denom()
            if num == 1:
                return asech(den)
            if num == -1:
                return acosh(-1/den)
        if e.is_Mul and I in e.args:
            rest = simplify(e / I)
            if rest.func == acosh and len(rest.args) == 1:
                arg = rest.args[0]
                num, den = arg.as_numer_denom()
                if num == 1:
                    return asec(den)
        return e

    prev = None
    cur = expr
    while prev != cur:
        prev = cur
        cur = cur.replace(lambda e: True, _rw)
        try:
            cur = simplify(cur)
        except Exception:
            pass
    return cur


def merge_duplicate_pairs(pairs: Iterable[Pair], z):
    _validate_pair_sequence(pairs, func_name='merge_duplicate_pairs')
    grouped = defaultdict(list)
    for expr, cond in clean_pairs(pairs):
        expr2 = rewrite_branch_expr(expr)
        cond2 = simplify_branch_condition(cond, z)
        if cond2 is S.false:
            continue
        grouped[expr2].append(cond2)

    out = []
    for expr, conds in grouped.items():
        uniq = []
        seen = set()
        for c in conds:
            key = str(c)
            if key not in seen:
                seen.add(key)
                uniq.append(c)
        cond = uniq[0] if len(uniq) == 1 else Or(*uniq)
        out.append((expr, cond))
    return out


def post_simplify_pairs(pairs: Iterable[Pair], z):
    _validate_pair_sequence(pairs, func_name='post_simplify_pairs')
    out = []
    for expr, cond in clean_pairs(pairs):
        expr2 = rewrite_branch_expr(expr)
        cond2 = simplify_branch_condition(cond, z)
        if cond2 is S.false:
            continue
        out.append((expr2, cond2))
    return merge_duplicate_pairs(out, z)


def post_simplify_restricted(exprs: Iterable[object], z):
    _validate_restricted_sequence(exprs, func_name='post_simplify_restricted')
    pairs = []
    for item in exprs:
        if isinstance(item, RestrictedExpr):
            pairs.append((item.expr, item.cond))
        else:
            pairs.append((sympify(item), S.true))
    simp = post_simplify_pairs(pairs, z)
    return [restrict(expr, cond) for expr, cond in simp]
