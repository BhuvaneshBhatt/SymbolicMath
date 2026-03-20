from __future__ import annotations

from itertools import product
from typing import Callable, Iterable

from sympy import (
    S,
    Add,
    Derivative,
    I,
    Mul,
    Piecewise,
    Symbol,
    diff,
    im,
    log,
    pi,
    simplify,
    sympify,
)
from sympy.logic.boolalg import And, Boolean
from sympy.logic.inference import satisfiable
from sympy.sets.contains import Contains

from .core import RestrictedExpr, Undefined, get_branch_rule


Pair = tuple[object, object]


def unsatisfiable_q(cond, symbols, assumptions=True) -> bool:
    """Best-effort unsatisfiability check using SymPy simplification and SAT search."""
    cond = sympify(cond)
    symbols = tuple(symbols) if isinstance(symbols, (list, tuple, set)) else (symbols,)
    ass = sympify(assumptions)
    try:
        simp = simplify(And(cond, ass))
    except Exception:
        simp = And(cond, ass)
    if simp is S.false:
        return True
    if simp is S.true:
        return False
    try:
        return not bool(satisfiable(simp))
    except Exception:
        return False



def condition_simplify(cond, symbols, assumptions=True):
    """Simplify a Boolean condition and drop it to False if provably unsatisfiable."""
    cond = sympify(cond)
    try:
        simp = simplify(cond)
    except Exception:
        simp = cond
    if simp is S.false or simp is False:
        return S.false
    if simp is S.true or simp is True:
        return S.true
    if unsatisfiable_q(simp, symbols, assumptions=assumptions):
        return S.false
    return simp



def condition_false_q(cond, symbols, assumptions=True) -> bool:
    return condition_simplify(cond, symbols, assumptions=assumptions) is S.false



def clean_pairs(pairs: Iterable[Pair]) -> list[Pair]:
    out: list[Pair] = []
    for pair in pairs:
        if not isinstance(pair, (tuple, list)) or len(pair) != 2:
            continue
        expr, cond = pair
        cond = sympify(cond)
        if cond is S.false or cond is False:
            continue
        out.append((sympify(expr), cond))
    return out



def map_pairs(f: Callable, pairs: Iterable[Pair]) -> list[Pair]:
    return [(f(expr), cond) for expr, cond in clean_pairs(pairs)]



def differentiate_pairs(pairs: Iterable[Pair], *vars_and_counts) -> list[Pair]:
    """Differentiate only the expression part of each pair, preserving the condition.

    This implements the standard jump-propagation heuristic
    J[D[f,z], z] = D[J[f,z], z] away from branch points/endpoints.
    """
    out: list[Pair] = []
    for expr, cond in clean_pairs(pairs):
        dexpr = expr
        for var, count in vars_and_counts:
            dexpr = diff(dexpr, var, count)
        out.append((simplify(dexpr), cond))
    return clean_pairs(out)



def gate_pairs(g: Callable, pairs: Iterable[Pair], symbols, assumptions=True) -> list[Pair]:
    out = []
    for expr, cond in clean_pairs(pairs):
        gated = condition_simplify(And(cond, g(expr)), symbols, assumptions=assumptions)
        if gated is not S.false:
            out.append((expr, gated))
    return out



def cross_diff_pairs(f_plus: Callable, f_minus: Callable, plus: Iterable[Pair], minus: Iterable[Pair]) -> list[Pair]:
    out = []
    for p_expr, p_cond in clean_pairs(plus):
        for m_expr, m_cond in clean_pairs(minus):
            out.append((f_plus(p_expr) - f_minus(m_expr), And(p_cond, m_cond)))
    return out



def _combine_head(head, args):
    if head is Add:
        return Add(*args)
    if head is Mul:
        return Mul(*args)
    return head(*args)



def combine_general_capped(head, pair_lists: list[list[Pair]], cap: Callable[[list[Pair]], list[Pair]] | None = None) -> list[Pair]:
    pair_lists = [clean_pairs(lst) for lst in pair_lists]
    if not pair_lists:
        return []
    out: list[Pair] = []
    for combo in product(*pair_lists):
        exprs = [expr for expr, _ in combo]
        conds = [cond for _, cond in combo]
        out.append((_combine_head(head, exprs), And(*conds) if len(conds) > 1 else conds[0]))
    out = clean_pairs(out)
    return cap(out) if cap else out



def cap_pairs(pairs: Iterable[Pair], max_per_expr: int = 2, max_total: int = 128) -> list[Pair]:
    grouped = {}
    for expr, cond in clean_pairs(pairs):
        grouped.setdefault(expr, []).append((expr, cond))
    out: list[Pair] = []
    for expr_pairs in grouped.values():
        out.extend(expr_pairs[:max_per_expr])
    return out[:max_total]



def ensure_neutral_pairs(pairs: Iterable[Pair]) -> list[Pair]:
    pairs = clean_pairs(pairs)
    if any(expr == 0 and cond is S.true for expr, cond in pairs):
        return pairs
    return pairs + [(S.Zero, S.true)]



def simplify_pairs(pairs: Iterable[Pair], symbols, assumptions=True) -> list[Pair]:
    out = []
    for expr, cond in clean_pairs(pairs):
        new_cond = condition_simplify(cond, symbols, assumptions=assumptions)
        if new_cond is not S.false:
            out.append((simplify(expr), new_cond))
    return clean_pairs(out)



def pairs_to_restricted(pairs: Iterable[Pair]):
    from .core import RestrictedExpr, restrict

    return tuple(restrict(expr, cond) for expr, cond in clean_pairs(pairs))



def pairs_to_piecewise_list(pairs: Iterable[Pair], default=None):
    if default is None:
        default = Undefined()
    return tuple(Piecewise((expr, cond), (default, True)) for expr, cond in clean_pairs(pairs))



def jump_pairs_from_table(expr, arg, z, assumptions=True, cap: Callable[[list[Pair]], list[Pair]] | None = None) -> list[Pair]:
    """Instantiate table-driven jumps using the current branch-rule registry."""
    try:
        rule = get_branch_rule(sympify(expr), sympify(arg) if arg is not None else z)
    except Exception:
        return []
    pairs = [(term.expr, term.cond) for term in rule.jump_terms]
    pairs = simplify_pairs(pairs, z, assumptions=assumptions)
    return cap(pairs) if cap else pairs



def _expr_is_z_free(expr, z: Symbol) -> bool:
    expr = sympify(expr)
    return expr.is_Atom or not expr.has(z)



def jump_compose(head, u, inner_assoc, z, assumptions=True, cap: Callable[[list[Pair]], list[Pair]] | None = None):
    expr = head(u)
    plus_pairs = map_pairs(head, inner_assoc['Plus'])
    minus_pairs = map_pairs(head, inner_assoc['Minus'])
    if cap:
        plus_pairs = cap(plus_pairs)
        minus_pairs = cap(minus_pairs)
    self_jumps = jump_pairs_from_table(expr, u, z, assumptions=assumptions, cap=cap)
    inner_jumps = cross_diff_pairs(lambda x: x, lambda x: x, plus_pairs, minus_pairs)
    jumps = self_jumps + inner_jumps
    if cap:
        jumps = cap(jumps)
    return {
        'Plus': plus_pairs,
        'Minus': minus_pairs,
        'Jump': ensure_neutral_pairs(simplify_pairs(jumps, z, assumptions=assumptions)),
    }



def _expanded_variable_counts(deriv: Derivative):
    return list(deriv.variable_count)



def branch_cut_jumps_pairs(expr, z: Symbol, assumptions=True, max_per_expr: int = 2, max_total: int = 128):
    # RestrictedExpr support is handled recursively inside go().
    """Recursively compute plus/minus/jump pair representations for an expression.

    Each entry is a list of ``(expr, condition)`` pairs describing boundary values on
    the two sides of cuts and the resulting jump contributions.

    Derivatives are propagated by differentiating the existing plus/minus/jump
    representatives and preserving their conditions. This follows the usual
    jump rule away from branch points/endpoints.
    """

    expr = sympify(expr)

    def cap(lst: list[Pair]) -> list[Pair]:
        return cap_pairs(clean_pairs(lst), max_per_expr=max_per_expr, max_total=max_total)

    cache = {}

    def go(e):
        if _is_restricted_expr_instance(e):
            inner_expr, inner_cond = e.args
            inner = go(inner_expr)
            return _restricted_assoc(inner, inner_cond, z, opts={'assumptions': assumptions})
        e = sympify(e)
        # Boolean conditions are not analytic expressions and should not be recursed through.
        # They can appear only as metadata inside RestrictedExpr conditions.
        if isinstance(e, Boolean) or getattr(e, 'is_Boolean', False):
            out = {'Plus': [(e, S.true)], 'Minus': [(e, S.true)], 'Jump': [(S.Zero, S.true)]}
            cache[e] = out
            return out
        key = e
        if key in cache:
            return cache[key]

        if _expr_is_z_free(e, z):
            out = {'Plus': [(e, S.true)], 'Minus': [(e, S.true)], 'Jump': [(S.Zero, S.true)]}
            cache[key] = out
            return out

        # Derivative-aware propagation.
        if isinstance(e, Derivative):
            base = e.expr
            inner = go(base)
            var_counts = _expanded_variable_counts(e)
            P = cap(differentiate_pairs(inner['Plus'], *var_counts))
            M = cap(differentiate_pairs(inner['Minus'], *var_counts))
            J = cap(differentiate_pairs(inner['Jump'], *var_counts))
            out = {
                'Plus': P,
                'Minus': M,
                'Jump': ensure_neutral_pairs(simplify_pairs(J, z, assumptions=assumptions)),
            }
            cache[key] = out
            return out

        # Special handling for Log and simple power gives cleaner side values.
        if e.func == log and len(e.args) == 1:
            u = e.args[0]
            inner = go(u)
            cutq = lambda val: Contains(val, S.Reals) & (val < 0)
            plus_cut = map_pairs(lambda val: log(-val) + I * S.Pi, gate_pairs(cutq, inner['Plus'], z, assumptions=assumptions))
            minus_cut = map_pairs(lambda val: log(-val) - I * S.Pi, gate_pairs(cutq, inner['Minus'], z, assumptions=assumptions))
            plus_reg = map_pairs(log, gate_pairs(lambda val: ~cutq(val), inner['Plus'], z, assumptions=assumptions))
            minus_reg = map_pairs(log, gate_pairs(lambda val: ~cutq(val), inner['Minus'], z, assumptions=assumptions))
            P = cap(plus_cut + plus_reg)
            M = cap(minus_cut + minus_reg)
            J = ensure_neutral_pairs(simplify_pairs(cross_diff_pairs(lambda x: x, lambda x: x, P, M), z, assumptions=assumptions))
            out = {'Plus': P, 'Minus': M, 'Jump': J}
            cache[key] = out
            return out

        if e.is_Pow and len(e.args) == 2 and not e.exp.has(z):
            u, a = e.base, e.exp
            inner = go(u)
            cutq = lambda val: Contains(val, S.Reals) & (val < 0) & ~Contains(a, S.Integers)
            plus_cut = map_pairs(lambda val: (-val) ** a * (S.Exp1 ** (I * S.Pi * a)), gate_pairs(cutq, inner['Plus'], z, assumptions=assumptions))
            minus_cut = map_pairs(lambda val: (-val) ** a * (S.Exp1 ** (-I * S.Pi * a)), gate_pairs(cutq, inner['Minus'], z, assumptions=assumptions))
            plus_reg = map_pairs(lambda val: val ** a, gate_pairs(lambda val: ~cutq(val), inner['Plus'], z, assumptions=assumptions))
            minus_reg = map_pairs(lambda val: val ** a, gate_pairs(lambda val: ~cutq(val), inner['Minus'], z, assumptions=assumptions))
            P = cap(plus_cut + plus_reg)
            M = cap(minus_cut + minus_reg)
            J = ensure_neutral_pairs(simplify_pairs(cross_diff_pairs(lambda x: x, lambda x: x, P, M), z, assumptions=assumptions))
            out = {'Plus': P, 'Minus': M, 'Jump': J}
            cache[key] = out
            return out

        if e.func == Add:
            parts = [go(arg) for arg in e.args]
            P = cap(combine_general_capped(Add, [p['Plus'] for p in parts], cap=cap))
            M = cap(combine_general_capped(Add, [p['Minus'] for p in parts], cap=cap))
            J = cap(combine_general_capped(Add, [p['Jump'] for p in parts], cap=cap))
            out = {'Plus': P, 'Minus': M, 'Jump': ensure_neutral_pairs(simplify_pairs(J, z, assumptions=assumptions))}
            cache[key] = out
            return out

        if e.func == Mul:
            parts = [go(arg) for arg in e.args]
            P = cap(combine_general_capped(Mul, [p['Plus'] for p in parts], cap=cap))
            M = cap(combine_general_capped(Mul, [p['Minus'] for p in parts], cap=cap))
            J = cap(cross_diff_pairs(lambda x: x, lambda x: x, P, M))
            out = {'Plus': P, 'Minus': M, 'Jump': ensure_neutral_pairs(simplify_pairs(J, z, assumptions=assumptions))}
            cache[key] = out
            return out

        if len(e.args) == 1:
            u = e.args[0]
            out = jump_compose(e.func, u, go(u), z, assumptions=assumptions, cap=cap)
            cache[key] = out
            return out

        parts = [go(arg) for arg in e.args]
        P = cap(combine_general_capped(e.func, [p['Plus'] for p in parts], cap=cap))
        M = cap(combine_general_capped(e.func, [p['Minus'] for p in parts], cap=cap))
        table_jumps: list[Pair] = []
        for arg in e.args:
            table_jumps.extend(jump_pairs_from_table(e, arg, z, assumptions=assumptions, cap=cap))
        if table_jumps:
            J = cap(table_jumps)
        else:
            J = cap(cross_diff_pairs(lambda x: x, lambda x: x, P, M))
        out = {'Plus': P, 'Minus': M, 'Jump': ensure_neutral_pairs(simplify_pairs(J, z, assumptions=assumptions))}
        cache[key] = out
        return out

    return go(expr)



def branch_cut_jumps(expr, z: Symbol, mode: str = 'restricted', assumptions=True, max_per_expr: int = 2, max_total: int = 128):
    """Convert ``branch_cut_jumps_pairs`` output into restricted expressions or Piecewise objects."""
    if mode not in {'pairs', 'restricted', 'piecewise'}:
        raise ValueError("branch_cut_jumps mode must be one of: 'pairs', 'restricted', 'piecewise'")
    pairs = branch_cut_jumps_pairs(expr, z, assumptions=assumptions, max_per_expr=max_per_expr, max_total=max_total)
    if mode == 'pairs':
        return pairs
    if mode == 'restricted':
        return {k: pairs_to_restricted(v) for k, v in pairs.items()}
    if mode == 'piecewise':
        return {
            'Plus': pairs_to_piecewise_list(pairs['Plus']),
            'Minus': pairs_to_piecewise_list(pairs['Minus']),
            'Jump': pairs_to_piecewise_list(pairs['Jump'], default=S.Zero),
        }
    raise ValueError("mode must be one of: 'pairs', 'restricted', 'piecewise'")



def _condition_holds_numeric(cond, z: Symbol, point) -> bool:
    try:
        sub = simplify(sympify(cond).subs(z, point))
    except Exception:
        return False
    if sub is S.true or sub is True:
        return True
    if sub is S.false or sub is False:
        return False
    try:
        return bool(sub)
    except Exception:
        return False



def _condition_text_sample(cond, z: Symbol):
    text = str(cond)
    # Real-axis samples
    if 'Contains(' in text and 'Reals' in text:
        if f'{z} < -1' in text:
            return -2, I
        if f'{z} < 0' in text and f'{z} > -1' in text:
            return S(-1) / 2, I
        if f'{z} > 1' in text:
            return 2, I
        if f'{z} > 0' in text and f'{z} < 1' in text:
            return S(1) / 2, I
        if f'{z} > -1' in text and f'{z} < 1' in text:
            return S.Zero, I
        if f'{z} < 0' in text:
            return -2, I
        if f'{z} > 0' in text:
            return 2, I
    # Imaginary-axis samples
    if f'Eq(re({z}), 0)' in text or f're({z}) == 0' in text:
        if f'im({z}) > 1' in text:
            return 2 * I, S.One
        if f'im({z}) < -1' in text:
            return -2 * I, S.One
        if 'Abs(im(' in text and '< 1' in text:
            return S.Half * I, S.One
    return None



def _candidate_samples_for_condition(cond, z: Symbol):
    text_sample = _condition_text_sample(cond, z)
    candidates = []
    if text_sample is not None:
        candidates.append(text_sample)

    text = str(cond)
    # Elliptic amplitude seams and related phi-seams are typically crossed by changing Re(z)
    # near pi/2 while holding Im(z) fixed.
    if ('cos(' in text and f'{z}' in text) or ('sin(' in text and f'{z}' in text):
        for yy in (S(1) / 4, S.Half, S.One, S(3) / 2):
            candidates.extend([
                (pi / 2 + I * yy, S.One),
                (pi / 2 - I * yy, S.One),
            ])

    # Generic fallback candidates for common cut geometries.
    candidates.extend([
        (-2, I),
        (-S.Half, I),
        (S.Half, I),
        (2, I),
        (2 * I, S.One),
        (-2 * I, S.One),
        (S.Half * I, S.One),
        (-S.Half * I, S.One),
    ])

    seen = set()
    uniq = []
    for point, normal in candidates:
        key = (str(point), str(normal))
        if key not in seen:
            seen.add(key)
            uniq.append((point, normal))
    return uniq



def _sample_from_condition(cond, z: Symbol):
    """Find a representative point and normal for a branch condition.

    This is heuristic. It now includes explicit support for the more complicated
    seam predicates that appear in elliptic amplitude cases.
    """
    for point, normal in _candidate_samples_for_condition(cond, z):
        if _condition_holds_numeric(cond, z, point):
            return point, normal
    return None



def numeric_branch_cut_jumps(expr, vars_list, wprec: int = 30, step_size=None, assumptions=True):
    """Numerically sample jumps across branch cuts inferred from ``branch_cut_jumps_pairs``.

    Currently this supports one complex variable. It handles common real-axis and
    imaginary-axis cuts, and now also supports the more complicated seam predicates
    that arise in elliptic amplitude cases by searching for representative sample
    points near ``pi/2 + i y`` and crossing them in the real direction.
    """
    if isinstance(vars_list, Symbol):
        vars_list = [vars_list]
    if len(vars_list) != 1:
        raise NotImplementedError('numeric_branch_cut_jumps currently supports exactly one variable')
    z = vars_list[0]
    expr = sympify(expr)
    step = sympify(step_size) if step_size is not None else S('1e-8')

    pairs = branch_cut_jumps_pairs(expr, z, assumptions=assumptions)['Jump']
    out = []
    for jump_expr, cond in clean_pairs(pairs):
        if jump_expr == 0:
            continue
        sample = _sample_from_condition(cond, z)
        if sample is None:
            continue
        point, normal = sample
        f_plus = (expr.subs(z, point + step * normal)).evalf(wprec)
        f_minus = (expr.subs(z, point - step * normal)).evalf(wprec)
        out.append({
            'BranchCondition': cond,
            'SamplePoint': point,
            'Normal': normal,
            'fPlus': f_plus,
            'fMinus': f_minus,
            'Jump': (f_plus - f_minus).evalf(wprec),
            'ExpectedJump': jump_expr.subs(z, point).evalf(wprec) if jump_expr.has(z) else jump_expr.evalf(wprec),
        })
    return out


def _restrict_pairs(pairs, cond, z, opts=None):
    out = []
    for expr, c in clean_pairs(pairs):
        new_c = condition_simplify(And(c, cond), z, **(opts or {}))
        if new_c is not False:
            out.append((expr, new_c))
    return clean_pairs(out)

def _restricted_assoc(inner, cond, z, opts=None):
    plus = _restrict_pairs(inner["Plus"], cond, z, opts=opts)
    minus = _restrict_pairs(inner["Minus"], cond, z, opts=opts)
    jump = _restrict_pairs(inner["Jump"], cond, z, opts=opts)
    if not jump:
        jump = [(0, cond)]
    return ensure_neutral_assoc({"Plus": plus, "Minus": minus, "Jump": jump})

def _is_restricted_expr_instance(e):
    return isinstance(e, RestrictedExpr) or (
        getattr(getattr(e, 'func', None), '__name__', '') == 'RestrictedExpr' and getattr(e, 'args', None) and len(e.args) == 2
    )




def ensure_neutral_assoc(assoc):
    a = dict(assoc)
    a["Jump"] = ensure_neutral_pairs(a.get("Jump", []))
    return a
