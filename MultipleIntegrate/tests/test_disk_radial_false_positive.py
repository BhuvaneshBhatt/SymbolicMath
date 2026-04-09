import sympy as sp

from multiple_integrate import region_from_ranges


def test_disk_radial_integral_handles_axis_vanishing_angular_factor_expr():
    x, y = sp.symbols('x y', real=True)
    region = region_from_ranges([(y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)])
    expr = x**2 * y**2 / sp.sqrt(1 - x**2 - y**2)
    assert sp.simplify(region.radial_integral(expr) - 2 * sp.pi / 15) == 0
