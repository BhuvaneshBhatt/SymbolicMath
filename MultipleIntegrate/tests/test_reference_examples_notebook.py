
import sympy as sp

from multiple_integrate import multiple_integrate, DiskRegion, BallRegion


class TestNotebookReferenceExamples:
    def test_rectangular_double_integral(self):
        x, y = sp.symbols("x y", positive=True)
        result = multiple_integrate(x**2 * y / (x + y), (x, 0, 3), (y, 1, 2))
        expected = sp.Rational(39, 8) + 36 * sp.log(2) - sp.Rational(65, 4) * sp.log(5)
        assert sp.simplify(result - expected) == 0

    def test_rectangular_triple_integral(self):
        x, y, z = sp.symbols("x y z", real=True)
        result = multiple_integrate(sp.cos(z) * sp.exp(x) * y, (x, 0, 1), (y, -1, 2), (z, 0, 3))
        expected = sp.Rational(3, 2) * (sp.E - 1) * sp.sin(3)
        assert sp.simplify(result - expected) == 0

    def test_disk_region_area(self):
        x, y = sp.symbols("x y", real=True)
        region = DiskRegion(((y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)), (x, -1, 1)), radius=1)
        result = multiple_integrate(1, region)
        assert sp.simplify(result - sp.pi) == 0

    def test_ball_integral_sumsq_table_n3(self):
        x, y, z = sp.symbols("x y z", real=True)
        region = BallRegion(
            (
                (x, -1, 1),
                (y, -sp.sqrt(1 - x**2), sp.sqrt(1 - x**2)),
                (z, -sp.sqrt(1 - x**2 - y**2), sp.sqrt(1 - x**2 - y**2)),
            ),
            radius=1,
            dimension=3,
        )
        result = multiple_integrate(x**2 + y**2 + z**2, region)
        assert sp.simplify(result - sp.Rational(4, 5) * sp.pi) == 0
