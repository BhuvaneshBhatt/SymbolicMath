import sympy as sp
from itertools import product

class JetSpace:
    """
    Jet space for one dependent variable u and several independent variables x_i.
    Coordinates are stored as independent symbols:
        u, u_x, u_t, u_x2, u_x_t, u_t2, ...
    """

    def __init__(self, indep_vars, dep_name="u", max_order=2):
        self.xs = tuple(indep_vars)
        self.n = len(self.xs)
        self.dep_name = dep_name
        self.max_order = max_order

        self.u0 = sp.Symbol(dep_name)
        self._coord_cache = {(): self.u0}
        self._by_order = {0: [()]}

        for order in range(1, max_order + 1):
            mis = []
            for idx in product(range(self.n), repeat=order):
                # keep only nondecreasing multi-indices so mixed partials are canonical
                if tuple(sorted(idx)) == idx:
                    mis.append(idx)
                    self._coord_cache[idx] = sp.Symbol(self._name_for(idx))
            self._by_order[order] = mis

    def _name_for(self, mi):
        """
        Turn a multi-index into a readable symbol name.
        Example with variables (x, t):
            ()      -> u
            (0,)    -> u_x
            (1,)    -> u_t
            (0,0)   -> u_x2
            (0,1)   -> u_x_t
            (1,1)   -> u_t2
        """
        counts = [mi.count(i) for i in range(self.n)]
        parts = [self.dep_name]
        for i, c in enumerate(counts):
            varname = str(self.xs[i])
            if c == 1:
                parts.append(varname)
            elif c > 1:
                parts.append(f"{varname}{c}")
        return "_".join(parts)

    def coord(self, mi):
        return self._coord_cache[tuple(mi)]

    def multiindices(self, order=None):
        if order is None:
            out = []
            for k in range(self.max_order + 1):
                out.extend(self._by_order[k])
            return out
        return list(self._by_order[order])

    def append_index(self, mi, var_index):
        return tuple(sorted(tuple(mi) + (var_index,)))

    def coordinates(self):
        out = []
        for order in range(self.max_order + 1):
            out.extend(self.coord(mi) for mi in self._by_order[order])
        return out

    def total_derivative(self, expr, var_index):
        """
        Truncated total derivative D_i on the jet space:
            D_i = ∂/∂x_i + Σ u_{J,i} ∂/∂u_J
        """
        xi = self.xs[var_index]
        result = sp.diff(expr, xi)

        for order in range(self.max_order):
            for mi in self._by_order[order]:
                result += sp.diff(expr, self.coord(mi)) * self.coord(self.append_index(mi, var_index))

        return sp.expand(result)


def prolongation_coefficients(jet, xis, phi):
    """
    Standard recursive prolongation formula for a point symmetry generator

        X = Σ xi^i(x,u) ∂_{x_i} + phi(x,u) ∂_u

    with coefficients on jet coordinates:
        pr^(n)X = X + Σ phi_J ∂_{u_J}

    recursion:
        phi_{J,i} = D_i(phi_J) - Σ_k u_{J,k} D_i(xi^k)
    """
    xis = tuple(map(sp.expand, xis))
    phi = sp.expand(phi)

    coeffs = {(): phi}

    for order in range(jet.max_order):
        for mi in jet.multiindices(order):
            for i in range(jet.n):
                new_mi = jet.append_index(mi, i)
                if len(new_mi) != order + 1 or new_mi in coeffs:
                    continue

                term = jet.total_derivative(coeffs[mi], i)
                correction = sum(
                    jet.coord(jet.append_index(mi, k)) * jet.total_derivative(xis[k], i)
                    for k in range(jet.n)
                )
                coeffs[new_mi] = sp.expand(term - correction)

    return coeffs


def prolongation_action(jet, expr, xis, phi):
    """
    Apply pr^(n)X to a jet-space expression expr.
    """
    coeffs = prolongation_coefficients(jet, xis, phi)

    result = sum(xis[i] * sp.diff(expr, jet.xs[i]) for i in range(jet.n))
    result += phi * sp.diff(expr, jet.u0)

    for order in range(1, jet.max_order + 1):
        for mi in jet.multiindices(order):
            result += coeffs[mi] * sp.diff(expr, jet.coord(mi))

    return sp.expand(result)


def substitute_heat_solution_manifold(expr, jet):
    """
    For the heat equation u_t - u_xx = 0, substitute only what is needed
    on the 2-jet:
        u_t -> u_xx
    """
    return sp.expand(expr.subs({jet.coord((1,)): jet.coord((0, 0))}))


def demo_heat_scaling():
    # independent variables
    x, t = sp.symbols("x t", positive=True)

    # build 2-jet for u(x,t)
    jet = JetSpace((x, t), dep_name="u", max_order=2)

    # shorthand jet coordinates
    u   = jet.coord(())
    ux  = jet.coord((0,))
    ut  = jet.coord((1,))
    uxx = jet.coord((0, 0))
    uxt = jet.coord((0, 1))
    utt = jet.coord((1, 1))

    print("Jet coordinates:")
    print(jet.coordinates())
    print()

    # Heat equation: u_t - u_xx = 0
    F = ut - uxx
    print("PDE F =")
    print(F)
    print()

    # Scaling symmetry generator:
    #   X = x ∂_x + 2 t ∂_t
    xi = (x, 2 * t)
    phi = sp.Integer(0)

    coeffs = prolongation_coefficients(jet, xi, phi)

    print("First and second prolongation coefficients:")
    print("phi_x  =", coeffs[(0,)])
    print("phi_t  =", coeffs[(1,)])
    print("phi_xx =", coeffs[(0, 0)])
    print("phi_xt =", coeffs[(0, 1)])
    print("phi_tt =", coeffs[(1, 1)])
    print()

    prXF = prolongation_action(jet, F, xi, phi)
    print("pr^(2)X(F) =")
    print(prXF)
    print()

    prXF_on_shell = substitute_heat_solution_manifold(prXF, jet)
    print("On the solution manifold u_t = u_xx:")
    print(prXF_on_shell)
    print()

    # This should simplify to a multiple of F
    print("Compare against -2*F:")
    print(sp.expand(prXF + 2 * F))
    print()

    # Similarity reduction:
    # invariants from dx/x = dt/(2t):
    #   z = x/sqrt(t)
    # and phi = 0 means u itself is invariant, so set u(x,t) = f(z)

    z = sp.symbols("z", real=True)
    f = sp.Function("f")

    # Use x = z*sqrt(t) to express derivatives symbolically
    u_ansatz = f(z)

    # chain rule:
    z_x = sp.diff(x / sp.sqrt(t), x).subs(x / sp.sqrt(t), z)   # 1/sqrt(t)
    z_t = sp.diff(x / sp.sqrt(t), t).subs(x / sp.sqrt(t), z)   # -z/(2 t)

    u_t_reduced = sp.diff(u_ansatz, z) * z_t
    u_x_reduced = sp.diff(u_ansatz, z) * z_x
    u_xx_reduced = sp.diff(u_x_reduced, x).subs({
        sp.diff(z, x): z_x,
        sp.diff(f(z), x): sp.diff(f(z), z) * z_x,
        sp.diff(sp.diff(f(z), z), x): sp.diff(f(z), (z, 2)) * z_x
    })

    # Clean the x-derivatives manually:
    u_xx_reduced = sp.simplify(sp.diff(f(z), (z, 2)) / t)

    reduced_eq = sp.simplify((u_t_reduced - u_xx_reduced) * t)
    print("Reduced ODE after u(x,t) = f(x/sqrt(t)):")
    print(sp.Eq(reduced_eq, 0))
    print()

    ode = sp.Eq(sp.diff(f(z), (z, 2)) + z * sp.diff(f(z), z) / 2, 0)
    print("Canonical reduced ODE:")
    print(ode)
    print()

    # Solve the reduced ODE
    ode_sol = sp.dsolve(ode)
    print("ODE solution:")
    print(ode_sol)
    print()

    # Recover PDE similarity solution
    C1, C2 = sp.symbols("C1 C2")
    similarity_solution = sp.Eq(
        sp.Function("u")(x, t),
        C1 + C2 * sp.erf(z / 2)
    )
    print("One equivalent similarity-family solution:")
    print(similarity_solution.subs(z, x / sp.sqrt(t)))


if __name__ == "__main__":
    demo_heat_scaling()
