from sympy import symbols, log, sqrt
from branchcuts import branch_jump, branch_cut_jumps, post_simplify_pairs

z = symbols('z', complex=True)
expr = log(1 - z) + sqrt(log(1 - z))

print("Jump for log(z):")
print(branch_jump(log(z), z))

pairs = branch_cut_jumps(expr, z, mode="pairs")
print("
Raw jump pairs:")
print(pairs["Jump"])

print("
Post-simplified jump pairs:")
print(post_simplify_pairs(pairs["Jump"], z))
