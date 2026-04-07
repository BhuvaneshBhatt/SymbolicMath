from pathlib import Path

import nbformat
import pytest
import sympy as sp

import liepde as lp
from liepde.symmetry import estimate_polynomial_ansatz_size
from liepde.verify import verify_solution_with_conditions


def _transport_equation():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x), 0)
    return x, t, u, eq


def test_import_and_public_api():
    assert hasattr(lp, 'liepde')
    assert not hasattr(lp, 'solve')
    assert not hasattr(lp, 'solve_pde_via_lie_symmetry')
    assert not hasattr(lp, 'build_scalar_jet_equation_from_sympy_pde')


def test_build_scalar_jet_equation_dep_name_regression_function_class():
    x, t, u, eq = _transport_equation()
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x, t), u, eq, max_order=1)
    assert jet.dep_name == 'u'
    assert jet.coord((1, 0)) in jet_eq.free_symbols
    assert jet.coord((0, 1)) in jet_eq.free_symbols


def test_build_scalar_jet_equation_dep_name_regression_explicit_name():
    x, t, u, eq = _transport_equation()
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x, t), u, eq, max_order=1, dep_name='U')
    assert jet.dep_name == 'U'
    assert jet.coord((1, 0)) in jet_eq.free_symbols


def test_low_level_symmetry_pipeline_runs():
    x, t, u, eq = _transport_equation()
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x, t), u, eq, max_order=1, dep_name='u')
    eq_obj, info = lp.pde.build_scalar_general_solved_pde_from_equation(jet, jet_eq, max_principal_order=1)
    sym = lp.symmetry.solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(
        eq_obj, degree=1, include_dependent_var=True
    )
    assert sym is not None
    assert hasattr(sym, 'basis_vectors')
    assert eq_obj.principal_multiindex is not None
    assert info.derivative_symbol is not None


def test_top_level_liepde_accepts_function_class():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(eq, u, (x, t), degree=1, max_degree=1, max_principal_order=1, return_details=True)
    assert result.equation_object.jet.dep_name == 'u'
    assert isinstance(result.basis_vectors, tuple)
    assert isinstance(result.matches, tuple)


def test_top_level_liepde_accepts_applied_function():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(eq, u(x, t), (x, t), degree=1, max_degree=1, max_principal_order=1, return_details=True)
    assert result.equation_object.jet.dep_name == 'u'
    assert result.principal_info.derivative_symbol is not None


def test_top_level_liepde_workflow_mode_runs():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(
        eq,
        u,
        (x, t),
        degree=1,
        max_degree=1,
        max_principal_order=1,
        use_workflow=True,
        workflow_max_steps=1,
        return_details=True,
    )
    assert result.workflow is not None
    assert hasattr(result.workflow, 'steps')


def test_repeated_reduction_workflow_low_level_runs():
    x, t, u, eq = _transport_equation()
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x, t), u, eq, max_order=1, dep_name='u')
    eq_obj, _ = lp.pde.build_scalar_general_solved_pde_from_equation(jet, jet_eq, max_principal_order=1)
    workflow = lp.workflows.repeated_reduction_workflow_scalar_kd_frobenius_default(
        eq_obj, max_steps=1, symmetry_degree=1, max_subset_size=1
    )
    assert workflow is not None
    assert hasattr(workflow, 'steps')


def test_notebook_exists_and_is_valid():
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'demo.ipynb'
    assert notebook_path.exists()
    nb = nbformat.read(notebook_path, as_version=4)
    text = '\n'.join(cell.source for cell in nb.cells if cell.cell_type == 'markdown')
    assert 'Lie symmetry methods for PDEs' in text
    assert 'Worked examples' in text
    assert 'transport equation' in text.lower()
    assert 'heat equation' in text.lower()
    assert 'burgers' in text.lower()


def test_basic_demo_mentions_multiple_examples():
    demo_path = Path(__file__).resolve().parents[1] / 'examples' / 'basic_demo.py'
    text = demo_path.read_text()
    assert 'transport' in text
    assert 'heat' in text
    assert 'reaction_diffusion_linear' in text


def test_high_order_jet_order_inference_repeated_derivative_regression():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x, 3) + u(x), 0)
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x,), u, eq)
    assert jet.max_order >= 3
    assert jet.coord((3,)) in jet_eq.free_symbols


def test_verify_solution_with_conditions_without_classical_core_dependency():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x), 0)
    sol = sp.Eq(u(x), 3)
    summary = verify_solution_with_conditions(eq, sol, u(x), (x,))
    assert summary.verified is True
    assert summary.status == 'verified'


def test_modular_api_exposes_symmetry_and_reduction_steps():
    x, t, u, eq = _transport_equation()
    dep_expr, dep_func, eq_obj, info = lp.build_equation_object(eq, u, (x, t), max_principal_order=1)
    assert dep_expr == u(x, t)
    assert dep_func == u
    assert info.derivative_symbol is not None
    sym = lp.compute_polynomial_symmetries(eq_obj, degree=1)
    basis, matches, reduction = lp.search_reductions_from_symmetries(eq_obj, sym, max_subset_size=1)
    assert isinstance(basis, tuple)
    assert isinstance(matches, tuple)
    assert reduction is None or hasattr(reduction, 'reduced_equation')


def test_top_level_liepde_collects_degree_attempt_diagnostics():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(eq, u, (x, t), degree=1, max_degree=2, max_principal_order=1, return_details=True)
    assert result.diagnostics.degree_attempts[0] == 1
    assert all(isinstance(msg, str) for msg in result.diagnostics.warnings)


def test_symmetry_size_estimate_and_failure_case():
    x, t, u, eq = _transport_equation()
    jet, jet_eq = lp.pde.build_scalar_jet_equation_from_sympy_pde((x, t), u, eq, max_order=1, dep_name='u')
    eq_obj, _ = lp.pde.build_scalar_general_solved_pde_from_equation(jet, jet_eq, max_principal_order=1)
    estimate = estimate_polynomial_ansatz_size(eq_obj, degree=2)
    assert estimate['unknown_count'] > 0
    with pytest.raises(ValueError):
        lp.symmetry.solve_determining_equations_with_polynomial_ansatz_scalar_general_kd(eq_obj, degree=-1)


def test_solve_reduced_equation_handles_none():
    assert lp.solve_reduced_equation(None) is None


def test_top_level_order_inference_handles_fourth_order_pde():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x, 4) + u(x), 0)
    result = lp.liepde(eq, u, (x,), degree=1, max_degree=1, return_details=True)
    assert result.equation_object.jet.max_order >= 4
    assert result.equation_object.principal_order >= 1


def test_public_order_inference_helper_matches_repeated_derivative_order():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x, 5) + u(x), 0)
    assert lp.infer_sympy_pde_order((x,), u, eq) == 5


def test_solve_reduced_equation_prefers_requested_unknown():
    z = sp.symbols('z')
    f = sp.Function('f')
    g = sp.Function('g')
    eq = sp.Eq(sp.diff(f(z), z), 0)
    sol = lp.solve_reduced_equation(eq, f(z))
    assert isinstance(sol, sp.Equality)
    assert sol.lhs == f(z)
    assert g(z) not in sol.free_symbols


def test_solve_reduced_equation_raises_on_ambiguous_unknown_without_hint():
    z = sp.symbols('z')
    f = sp.Function('f')
    g = sp.Function('g')
    eq = sp.Eq(sp.diff(f(z), z) + g(z), 0)
    with pytest.raises(ValueError):
        lp.solve_reduced_equation(eq)


def test_three_variable_jet_build_pipeline_runs():
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, y, t), t) + sp.diff(u(x, y, t), x) + sp.diff(u(x, y, t), y), 0)
    dep_expr, dep_func, eq_obj, info = lp.build_equation_object(eq, u, (x, y, t), max_principal_order=1)
    assert dep_expr == u(x, y, t)
    assert dep_func == u
    assert eq_obj.jet.k == 3
    assert info.derivative_symbol is not None


def test_notebook_contains_no_error_outputs_after_execution():
    notebook_path = Path(__file__).resolve().parents[1] / 'notebooks' / 'demo.ipynb'
    nb = nbformat.read(notebook_path, as_version=4)
    error_outputs = []
    for cell in nb.cells:
        if cell.cell_type != 'code':
            continue
        for output in cell.get('outputs', []):
            if output.get('output_type') == 'error':
                error_outputs.append(output)
    assert error_outputs == []


def test_collect_applied_functions_deduplicates_derivative_occurrences():
    z = sp.symbols('z')
    f = sp.Function('f')
    funcs = lp.utils.collect_applied_functions(sp.Eq(sp.diff(f(z), z) + f(z), 0))
    assert funcs == (f(z),)


def test_verify_reduction_accepts_consistent_reduced_ansatz():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x), 0)
    dep_expr, dep_func, eq_obj, _ = lp.build_equation_object(eq, u, (x,), max_principal_order=1)
    del dep_expr, dep_func
    f = sp.Function('f')
    reduction = lp.reduction.ScalarReductionResultKD(
        kind='unit_test',
        invariants=(),
        transverse_parameter=x,
        ansatz=f(x),
        reduced_expression=sp.diff(f(x), x),
        reduced_equation=sp.Eq(sp.diff(f(x), x), 0),
        reduced_function=f,
        reduced_unknown=f(x),
    )
    summary = lp.verify.verify_reduction(eq_obj, reduction)
    assert summary.valid is True


def test_verify_reduction_marks_substitution_failure_invalid():
    x = sp.symbols('x')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x), x), 0)
    _, _, eq_obj, _ = lp.build_equation_object(eq, u, (x,), max_principal_order=1)
    f = sp.Function('f')
    reduction = lp.reduction.ScalarReductionResultKD(
        kind='unit_test',
        invariants=(),
        transverse_parameter=x,
        ansatz=sp.Integral(f(x), x),
        reduced_expression=sp.diff(f(x), x),
        reduced_equation=sp.Eq(sp.diff(f(x), x), 0),
        reduced_function=f,
        reduced_unknown=f(x),
    )
    summary = lp.verify.verify_reduction(eq_obj, reduction)
    assert summary.valid is False
    assert summary.reduced_residual != 0


def test_top_level_liepde_returns_best_available_value_by_default():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(eq, u, (x, t), degree=1, max_degree=1, max_principal_order=1)
    assert result is None or isinstance(result, sp.Equality)


def test_transport_equation_symmetry_basis_is_nontrivial():
    x, t, u, eq = _transport_equation()
    details = lp.liepde(eq, u, (x, t), degree=1, max_degree=1, max_principal_order=1, return_details=True)
    assert len(details.basis_vectors) >= 1


def test_transport_equation_default_returns_readable_solution():
    x, t, u, eq = _transport_equation()
    result = lp.liepde(eq, u, (x, t), degree=1, max_degree=1, max_principal_order=1)
    assert isinstance(result, sp.Equality)
    assert result.lhs == u(x, t)


def test_heat_equation_similarity_fallback_returns_erf_solution():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    assert sol.rhs.has(sp.erf)
    residual = sp.simplify((sp.diff(sol.rhs, t) - sp.diff(sol.rhs, x, 2)).doit())
    assert residual == 0


def test_heat_equation_details_report_similarity_fallback_warning():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    res = lp.liepde(eq, u, (x, t), return_details=True)
    assert res.direct_solution is not None
    assert any(("_heat_similarity_solution" in msg) or ("_reaction_diffusion_similarity_solution" in msg) for msg in res.diagnostics.warnings)


def test_result_level_reduction_returns_partial_information_when_requested():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    out = lp.liepde(eq, u, (x, t), result_level="reduction")
    assert isinstance(out, sp.Equality)


def test_failure_mode_status_returns_failure_object():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.sin(sp.diff(u(x, t), t)) + sp.exp(sp.diff(u(x, t), x, 2)), 0)
    out = lp.liepde(eq, u, (x, t), failure_mode="status", max_degree=1)
    assert isinstance(out, (lp.LiePDEFailure, type(None), str, sp.Equality))


def test_analyze_only_returns_summary():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    info = lp.liepde(eq, u, (x, t), analyze_only=True)
    assert isinstance(info, lp.LiePDEAnalysis)
    assert info.order == 2
    assert info.is_linear is True


def test_wave_equation_fallback_returns_dalembert_family():
    x, t = sp.symbols("x t")
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t, 2) - sp.diff(u(x, t), x, 2), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    assert 'F' in str(sol.rhs) and 'G' in str(sol.rhs)


def test_advection_diffusion_similarity_fallback_returns_erf_solution():
    x, t = sp.symbols("x t", positive=True)
    u = sp.Function("u")
    eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x) - sp.diff(u(x, t), x, 2), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    assert sol.rhs.has(sp.erf)


def test_readme_mentions_result_levels_and_examples():
    readme = (Path(__file__).resolve().parents[1] / 'README.md').read_text()
    assert 'result_level' in readme
    assert 'Heat equation' in readme
    assert 'Wave equation' in readme
    assert 'Analyze only' in readme


def test_benchmark_script_exists_and_has_twenty_cases():
    bench = Path(__file__).resolve().parents[1] / 'benchmarks' / 'benchmark_common_pdes.py'
    text = bench.read_text()
    assert 'run_benchmark' in text
    assert text.count('("') >= 20


def test_advection_2d_returns_transport_family_solution():
    x, y, t = sp.symbols('x y t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, y, t), t) + sp.diff(u(x, y, t), x) + sp.diff(u(x, y, t), y), 0)
    sol = lp.liepde(eq, u, (x, y, t))
    assert isinstance(sol, sp.Equality)
    assert sol.lhs == u(x, y, t)
    assert len(sol.rhs.args) == 2
    residual = sp.simplify((sp.diff(sol.rhs, t) + sp.diff(sol.rhs, x) + sp.diff(sol.rhs, y)).doit())
    assert residual == 0


def test_reaction_diffusion_linear_known_solution_fallback():
    x, t = sp.symbols('x t', positive=True)
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2) - u(x, t), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    assert sol.has(sp.erf)
    assert sol.has(sp.exp)


def test_telegraph_traveling_wave_fallback_returns_solution_family():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, t), t, 2) + sp.diff(u(x, t), t) - sp.diff(u(x, t), x, 2), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    c = next(sym for sym in sol.free_symbols if sym.name == 'c')
    residual = sp.simplify((sp.diff(sol.rhs, t, 2) + sp.diff(sol.rhs, t) - sp.diff(sol.rhs, x, 2)).doit())
    assert sp.simplify(residual.subs(c, 2)) == 0


def test_first_order_linear_reaction_pde_is_solved():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, y), x) + 3 * sp.diff(u(x, y), y) + u(x, y), 1)
    sol = lp.liepde(eq, u, (x, y))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((sp.diff(sol.rhs, x) + 3 * sp.diff(sol.rhs, y) + sol.rhs - 1).doit())
    assert residual == 0


def test_variable_coefficient_first_order_pde_is_solved():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    eq = sp.Eq(x * sp.diff(u(x, y), y) + y * sp.diff(u(x, y), x), -4 * x * y * u(x, y))
    sol = lp.liepde(eq, u, (x, y))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((x * sp.diff(sol.rhs, y) + y * sp.diff(sol.rhs, x) + 4 * x * y * sol.rhs).doit())
    assert residual == 0


def test_transport_pde_with_symbolic_speed_keeps_characteristic():
    t, x, c = sp.symbols('t x c')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(t, x), t) + c * sp.diff(u(t, x), x), 0)
    sol = lp.liepde(eq, u, (t, x))
    assert isinstance(sol, sp.Equality)
    assert any('c*t - x' in s or 'x - c*t' in s for s in map(str, sol.rhs.args + (sol.rhs,)))


def test_kdv_equation_returns_explicit_solution_family():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, t), t) + sp.diff(u(x, t), x, 3) + 6 * u(x, t) * sp.diff(u(x, t), x), 0)
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((sp.diff(sol.rhs, t) + sp.diff(sol.rhs, x, 3) + 6 * sol.rhs * sp.diff(sol.rhs, x)).doit())
    assert residual == 0


def test_result_level_details_preserves_direct_solution_for_pde():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    eq = sp.Eq(3 * sp.diff(u(x, y), x) + 5 * sp.diff(u(x, y), y), x)
    res = lp.liepde(eq, u, (x, y), result_level='details')
    assert res.direct_solution is None or isinstance(res.direct_solution, sp.Equality)
    assert res.full_solution is None or isinstance(res.full_solution, sp.Equality)


def test_first_order_linear_pde_is_solved():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    eq = sp.Eq(3 * sp.diff(u(x, y), x) + 5 * sp.diff(u(x, y), y), x)
    sol = lp.liepde(eq, u, (x, y))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((3 * sp.diff(sol.rhs, x) + 5 * sp.diff(sol.rhs, y) - x).doit())
    assert residual == 0
    assert 'F' in str(sol.rhs)


def test_constant_speed_transport_pde_is_solved():
    t, x, c = sp.symbols('t x c')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(t, x), t) + c * sp.diff(u(t, x), x), 0)
    sol = lp.liepde(eq, u, (t, x))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((sp.diff(sol.rhs, t) + c * sp.diff(sol.rhs, x)).doit())
    assert residual == 0
    assert 'F' in str(sol.rhs)


def test_wave_pde_is_solved():
    x, t = sp.symbols('x t')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, t), t, 2), sp.diff(u(x, t), x, 2))
    sol = lp.liepde(eq, u, (x, t))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((sp.diff(sol.rhs, t, 2) - sp.diff(sol.rhs, x, 2)).doit())
    assert residual == 0
    assert 'F' in str(sol.rhs) and 'G' in str(sol.rhs)


def test_laplace_pde_is_solved():
    x, y = sp.symbols('x y')
    u = sp.Function('u')
    eq = sp.Eq(sp.diff(u(x, y), x, 2) + sp.diff(u(x, y), y, 2), 0)
    sol = lp.liepde(eq, u, (x, y))
    assert isinstance(sol, sp.Equality)
    residual = sp.simplify((sp.diff(sol.rhs, x, 2) + sp.diff(sol.rhs, y, 2)).doit())
    assert residual == 0
    assert 'F' in str(sol.rhs) and 'G' in str(sol.rhs)
