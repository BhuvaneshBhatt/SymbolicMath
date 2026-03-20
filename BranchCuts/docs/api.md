
# API Documentation

## branch_cut_jumps(expr, z, mode)

Modes:
- "pairs": returns dict with Plus, Minus, Jump
- "restricted": returns RestrictedExpr
- "piecewise": returns Piecewise

## post_simplify_pairs(pairs, z)

Input:
- list of (expr, condition)

Raises:
- TypeError if input is not list of pairs

## RestrictedExpr(expr, condition)

Represents domain-restricted expression.

Semantics:
branch_cut_jumps(RestrictedExpr(f, c)) =
restrict(branch_cut_jumps(f), c)
