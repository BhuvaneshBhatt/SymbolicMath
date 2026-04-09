
import sympy as sp

from multiple_integrate import (
    AffineSimplexRegion,
    AnnulusRegion,
    EllipsoidRegion,
    SphericalShellRegion,
    UnionRegion,
    multiple_integrate,
)
from multiple_integrate.regions import region_from_ranges


class TestAdvancedRegions:
    def test_affine_simplex_region_constant(self):
        x, y = sp.symbols("x y", real=True)
        region = AffineSimplexRegion(
            ((x, 2, 5), (y, 1, 1 + 2 * (1 - (x - 2) / 3))),
            shifts=(sp.Integer(2), sp.Integer(1)),
            scales=(sp.Integer(3), sp.Integer(2)),
            dimension=2,
        )
        result = multiple_integrate(1, region)
        assert sp.simplify(result - 3) == 0

    def test_affine_simplex_region_moment(self):
        x, y = sp.symbols("x y", real=True)
        region = AffineSimplexRegion(
            ((x, 2, 5), (y, 1, 1 + 2 * (1 - (x - 2) / 3))),
            shifts=(sp.Integer(2), sp.Integer(1)),
            scales=(sp.Integer(3), sp.Integer(2)),
            dimension=2,
        )
        result = multiple_integrate(x + y, region)
        expected = 14
        assert sp.simplify(result - expected) == 0

    def test_match_affine_simplex_region(self):
        x, y = sp.symbols("x y", real=True)
        region = region_from_ranges([(y, 1, 1 + 2 * (1 - (x - 2) / 3)), (x, 2, 5)])
        assert type(region).__name__ == "AffineSimplexRegion"

    def test_ellipsoid_region_constant(self):
        x, y = sp.symbols("x y", real=True)
        region = EllipsoidRegion(variables_nd=(x, y), axes=(sp.Integer(2), sp.Integer(3)))
        result = multiple_integrate(1, region)
        assert sp.simplify(result - 6 * sp.pi) == 0

    def test_ellipsoid_region_moment(self):
        x, y = sp.symbols("x y", real=True)
        region = EllipsoidRegion(variables_nd=(x, y), axes=(sp.Integer(2), sp.Integer(3)))
        result = multiple_integrate(x**2 + y**2, region)
        expected = sp.Rational(39, 2) * sp.pi
        assert sp.simplify(result - expected) == 0

    def test_annulus_region_constant(self):
        x, y = sp.symbols("x y", real=True)
        region = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
        result = multiple_integrate(1, region)
        assert sp.simplify(result - 3 * sp.pi) == 0

    def test_annulus_region_radial(self):
        x, y = sp.symbols("x y", real=True)
        region = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
        result = multiple_integrate(x**2 + y**2, region)
        assert sp.simplify(result - sp.Rational(15, 2) * sp.pi) == 0

    def test_spherical_shell_region_constant(self):
        x, y, z = sp.symbols("x y z", real=True)
        region = SphericalShellRegion(
            variables_nd=(x, y, z),
            inner_radius=1,
            outer_radius=2,
        )
        result = multiple_integrate(1, region)
        assert sp.simplify(result - sp.Rational(28, 3) * sp.pi) == 0

    def test_union_region_constant(self):
        x, y = sp.symbols("x y", real=True)
        reg1 = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=1)
        reg2 = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
        union = UnionRegion(pieces=(reg1, reg2))
        result = multiple_integrate(1, union)
        assert sp.simplify(result - 4 * sp.pi) == 0

    def test_union_region_polynomial(self):
        x, y = sp.symbols("x y", real=True)
        reg1 = AnnulusRegion(variables_xy=(x, y), inner_radius=0, outer_radius=1)
        reg2 = AnnulusRegion(variables_xy=(x, y), inner_radius=1, outer_radius=2)
        union = UnionRegion(pieces=(reg1, reg2))
        result = multiple_integrate(x**2, union)
        assert sp.simplify(result - 4 * sp.pi) == 0
