
# Examples

## Basic
branch_cut_jumps(log(z), z)

## Restricted
branch_cut_jumps(RestrictedExpr(log(z), z > 0), z)

## Composite
branch_cut_jumps(sqrt(log(1 - z)), z)

## Correct usage of post_simplify_pairs
pairs = branch_cut_jumps(expr, z, mode="pairs")
post_simplify_pairs(pairs["Jump"], z)
