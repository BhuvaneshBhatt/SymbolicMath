from __future__ import annotations

from time import perf_counter
import sympy as sp
import liepde as lp


def build_cases():
    x, t, y = sp.symbols("x t y", positive=True)
    u = sp.Function("u")
    return [
        ("transport", sp.Eq(u(x, t).diff(t) + u(x, t).diff(x), 0), (x, t)),
        ("transport_speed2", sp.Eq(u(x, t).diff(t) + 2*u(x, t).diff(x), 0), (x, t)),
        ("heat", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2), 0), (x, t)),
        ("heat_diff2", sp.Eq(u(x, t).diff(t) - 2*u(x, t).diff(x, 2), 0), (x, t)),
        ("advection_diffusion", sp.Eq(u(x, t).diff(t) + u(x, t).diff(x) - u(x, t).diff(x, 2), 0), (x, t)),
        ("wave", sp.Eq(u(x, t).diff(t, 2) - u(x, t).diff(x, 2), 0), (x, t)),
        ("wave_speed2", sp.Eq(u(x, t).diff(t, 2) - 4*u(x, t).diff(x, 2), 0), (x, t)),
        ("reaction_diffusion_linear", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2) - u(x, t), 0), (x, t)),
        ("reaction_diffusion_decay", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2) + u(x, t), 0), (x, t)),
        ("burgers_viscous", sp.Eq(u(x, t).diff(t) + u(x, t)*u(x, t).diff(x) - u(x, t).diff(x, 2), 0), (x, t)),
        ("laplace_2d", sp.Eq(u(x, y).diff(x, 2) + u(x, y).diff(y, 2), 0), (x, y)),
        ("advection_2d", sp.Eq(u(x, y, t).diff(t) + u(x, y, t).diff(x) + u(x, y, t).diff(y), 0), (x, y, t)),
        ("euler_poisson_darboux", sp.Eq(u(x, t).diff(t, 2) + u(x, t).diff(t)/t - u(x, t).diff(x, 2), 0), (x, t)),
        ("telegraph", sp.Eq(u(x, t).diff(t, 2) + u(x, t).diff(t) - u(x, t).diff(x, 2), 0), (x, t)),
        ("linear_potential", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2) + x*u(x, t).diff(x), 0), (x, t)),
        ("fokker_planck_like", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2) - x*u(x, t).diff(x), 0), (x, t)),
        ("kpp_like", sp.Eq(u(x, t).diff(t) - u(x, t).diff(x, 2) - u(x, t) + u(x, t)**2, 0), (x, t)),
        ("porous_medium_like", sp.Eq(u(x, t).diff(t) - (u(x, t)**2).diff(x, 2), 0), (x, t)),
        ("kdv_like", sp.Eq(u(x, t).diff(t) + u(x, t)*u(x, t).diff(x) + u(x, t).diff(x, 3), 0), (x, t)),
        ("beam_like", sp.Eq(u(x, t).diff(t) + u(x, t).diff(x, 4), 0), (x, t)),
    ], u


def classify_result(obj):
    if isinstance(obj, sp.Equality):
        text = str(obj)
        if any(a.func.__name__ in {"F", "G"} for a in obj.atoms(sp.Function)) or obj.has(sp.erf):
            return "full_or_similarity_solution"
        if "Derivative(" in text:
            return "reduced_unsolved"
        return "equation_solution"
    if obj is None:
        return "none"
    if type(obj).__name__ == "LiePDEFailure":
        return "failure"
    return type(obj).__name__


def run_benchmark():
    cases, u = build_cases()
    rows = []
    for name, eq, indep in cases:
        start = perf_counter()
        out = lp.liepde(eq, u, indep, result_level="reduction", failure_mode="status")
        elapsed = perf_counter() - start
        rows.append((name, classify_result(out), elapsed, str(out)[:120]))
    name_w = max(len(r[0]) for r in rows)
    status_w = max(len(r[1]) for r in rows)
    print(f"{'case'.ljust(name_w)}  {'status'.ljust(status_w)}  seconds  preview")
    print("-" * (name_w + status_w + 22))
    for name, status, elapsed, preview in rows:
        print(f"{name.ljust(name_w)}  {status.ljust(status_w)}  {elapsed:7.3f}  {preview}", flush=True)

    solved = sum(status in {"full_or_similarity_solution", "equation_solution"} for _, status, _, _ in rows)
    reduced = sum(status == "reduced_unsolved" for _, status, _, _ in rows)
    failed = sum(status in {"failure", "none"} or status.startswith("EXC:") for _, status, _, _ in rows)
    print()
    print(f"Solved: {solved}")
    print(f"Reduced but unsolved: {reduced}")
    print(f"Failed: {failed}")


if __name__ == "__main__":
    run_benchmark()
