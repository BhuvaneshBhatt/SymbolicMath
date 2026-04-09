import sympy as sp

from multiple_integrate import multiple_integrate


def _assert_eq(expr, expected):
    assert sp.simplify(expr - expected) == 0


def test_disk_polar_nonradial_structured_case():
    x, y = sp.symbols('x y', real=True)
    expr = x**2 * y**2 / sp.sqrt(1 - x**2 - y**2)
    result = multiple_integrate(
        expr,
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    )
    _assert_eq(result, 2 * sp.pi / 15)


def test_ball_spherical_nonradial_structured_case():
    x, y, z = sp.symbols('x y z', real=True)
    expr = x**2 * y**2 * z**2
    result = multiple_integrate(
        expr,
        (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
        (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
        (x, -1, 1),
    )
    expected = 4 * sp.pi / 945
    _assert_eq(result, expected)


def test_ellipsoid_affine_change_of_variables():
    x, y = sp.symbols('x y', real=True)
    expr = x**2 * y**2
    result = multiple_integrate(
        expr,
        (y, -3 * sp.sqrt(1 - x**2 / 4), 3 * sp.sqrt(1 - x**2 / 4)),
        (x, -2, 2),
    )
    expected = 9 * sp.pi
    _assert_eq(result, expected)


def test_quadratic_gaussian_formula_with_linear_term():
    x, y = sp.symbols('x y', real=True)
    expr = sp.exp(-(2 * x**2 + 2 * x * y + 3 * y**2) + x - 4 * y + 5)
    result = multiple_integrate(expr, (y, -sp.oo, sp.oo), (x, -sp.oo, sp.oo))

    A = sp.Matrix([[2, 1], [1, 3]])
    b = sp.Matrix([1, -4])
    expected = sp.pi * sp.exp(5 + (b.T * A.LUsolve(b))[0] / 4) / sp.sqrt(A.det())
    _assert_eq(result, sp.simplify(expected))
