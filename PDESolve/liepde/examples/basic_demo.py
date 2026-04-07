import sympy as sp
import liepde as lp


def main():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")

    examples = {
        "transport": sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0),
        "heat": sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0),
        "wave": sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0),
        "reaction_diffusion_linear": sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2) - u(x, t), 0),
    }

    for name, eq in examples.items():
        print(f"\n=== {name} ===")
        print("best result:", lp.liepde(eq, u, (x, t)))
        reduction = lp.liepde(eq, u, (x, t), result_level="reduction")
        print("reduction-level result:", reduction)
        details = lp.liepde(eq, u, (x, t), result_level="details")
        print("basis vector count:", len(details.basis_vectors))
        print("warnings:", details.diagnostics.warnings)


if __name__ == "__main__":
    main()
