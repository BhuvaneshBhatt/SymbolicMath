import sympy as sp

from multiple_integrate import multiple_integrate


def test_simplex_dirichlet_half_integer_exact():
    x, y, z = sp.symbols('x y z', positive=True)
    expr = sp.sqrt(x) * y**sp.Rational(3, 2) * sp.sqrt(z) * sp.sqrt(1 - x - y - z)
    result = multiple_integrate(
        expr,
        (z, 0, 1 - x - y),
        (y, 0, 1 - x),
        (x, 0, 1),
    )
    expected = (
        sp.gamma(sp.Rational(3, 2))
        * sp.gamma(sp.Rational(5, 2))
        * sp.gamma(sp.Rational(3, 2))
        * sp.gamma(sp.Rational(3, 2))
        / sp.gamma(7)
    )
    assert sp.simplify(result - expected) == 0


def test_simplex_inverse_square_root_exact():
    x, y, z = sp.symbols('x y z', positive=True)
    expr = 1 / sp.sqrt(x * y * z * (1 - x - y - z))
    result = multiple_integrate(
        expr,
        (z, 0, 1 - x - y),
        (y, 0, 1 - x),
        (x, 0, 1),
    )
    assert sp.simplify(result - sp.pi**2) == 0


def test_four_simplex_dirichlet_fractional_exact():
    x, y, z, w = sp.symbols('x y z w', positive=True)
    expr = (
        x**sp.Rational(1, 2)
        * y**sp.Rational(1, 3)
        * z**sp.Rational(2, 3)
        * w**sp.Rational(1, 4)
        * (1 - x - y - z - w)**sp.Rational(3, 2)
    )
    result = multiple_integrate(
        expr,
        (w, 0, 1 - x - y - z),
        (z, 0, 1 - x - y),
        (y, 0, 1 - x),
        (x, 0, 1),
    )
    expected = (
        sp.gamma(sp.Rational(3, 2))
        * sp.gamma(sp.Rational(4, 3))
        * sp.gamma(sp.Rational(5, 3))
        * sp.gamma(sp.Rational(5, 4))
        * sp.gamma(sp.Rational(5, 2))
        / sp.gamma(sp.Rational(33, 4))
    )
    assert sp.simplify(result - expected) == 0
