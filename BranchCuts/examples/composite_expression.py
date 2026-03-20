from sympy import symbols, asec, acsc, sqrt, log
from branchcuts import branch_cut_jumps, post_simplify_pairs

z = symbols('z', complex=True)
expr = asec(1 - z) + acsc(2*z) + sqrt(log(1 - z))

pairs = branch_cut_jumps(expr, z, mode="pairs")
print("Raw Jump pairs:")
for item in pairs["Jump"]:
    print(item)

print("
Post-simplified Jump pairs:")
for item in post_simplify_pairs(pairs["Jump"], z):
    print(item)
