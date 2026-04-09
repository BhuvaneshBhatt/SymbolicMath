
from __future__ import annotations

from dataclasses import dataclass, field
import sympy as sp
from sympy.core.relational import Relational


def _normalize_ranges_input(ranges):
    if isinstance(ranges, Region):
        return ranges
    if isinstance(ranges, tuple) and len(ranges) == 1 and isinstance(ranges[0], Region):
        return ranges[0]
    if isinstance(ranges, tuple):
        return list(ranges)
    return ranges


def _clean_expr(expr):
    expr = sp.sympify(expr)
    try:
        return sp.simplify(expr)
    except Exception:
        return expr


def _split_dependence(expr: sp.Expr, radial_vars: tuple[sp.Symbol, ...], angular_vars: tuple[sp.Symbol, ...]):
    """Split a transformed expression into radial and angular factors when possible."""
    expr = _clean_expr(expr)
    radial_set = set(radial_vars)
    angular_set = set(angular_vars)
    if expr.is_Mul:
        radial = sp.Integer(1)
        angular = sp.Integer(1)
        for factor in expr.args:
            fsyms = factor.free_symbols
            if fsyms & radial_set and not (fsyms & angular_set):
                radial *= factor
            elif fsyms & angular_set and not (fsyms & radial_set):
                angular *= factor
            elif not fsyms & (radial_set | angular_set):
                radial *= factor
            else:
                return None
        return _clean_expr(radial), _clean_expr(angular)
    fsyms = expr.free_symbols
    if fsyms & radial_set and not (fsyms & angular_set):
        return _clean_expr(expr), sp.Integer(1)
    if fsyms & angular_set and not (fsyms & radial_set):
        return sp.Integer(1), _clean_expr(expr)
    if not fsyms & (radial_set | angular_set):
        return _clean_expr(expr), sp.Integer(1)
    return None


@dataclass(frozen=True)
class Region:
    """Base region class for structured multiple-integration domains."""

    ranges: tuple[tuple, ...] = field(default_factory=tuple)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return tuple(r[0] for r in self.ranges)

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return tuple(
            (v, _clean_expr(lo), _clean_expr(hi))
            for v, lo, hi in self.ranges
        )

    def constant_volume(self) -> sp.Expr | None:
        return None

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        return None

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        return None

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return False

    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        return None


@dataclass(frozen=True)
class BoxRegion(Region):
    def constant_volume(self) -> sp.Expr | None:
        vol = sp.Integer(1)
        vars_set = set(self.variables)
        for _, lo, hi in self.ranges:
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            if (lo_s.free_symbols | hi_s.free_symbols) & vars_set:
                return None
            vol *= hi_s - lo_s
        return sp.simplify(vol)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return self.symmetric_range(var) is not None

    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        for v, lo, hi in self.ranges:
            if v == var:
                lo_s = sp.sympify(lo)
                hi_s = sp.sympify(hi)
                if sp.simplify(lo_s + hi_s) == 0:
                    return lo_s, hi_s
        return None


@dataclass(frozen=True)
class IteratedRegion(Region):
    def symmetric_range(self, var: sp.Symbol) -> tuple[sp.Expr, sp.Expr] | None:
        for idx, (v, lo, hi) in enumerate(self.ranges):
            if v != var:
                continue
            lo_s = sp.sympify(lo)
            hi_s = sp.sympify(hi)
            if sp.simplify(lo_s + hi_s) != 0:
                return None
            for _, later_lo, later_hi in self.ranges[idx + 1 :]:
                if var in sp.sympify(later_lo).free_symbols or var in sp.sympify(later_hi).free_symbols:
                    return None
            return lo_s, hi_s
        return None

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return self.symmetric_range(var) is not None


@dataclass(frozen=True)
class GraphRegion(IteratedRegion):
    outer_var: sp.Symbol | None = None
    inner_var: sp.Symbol | None = None

    def constant_volume(self) -> sp.Expr | None:
        if len(self.ranges) != 2:
            return None
        (y, lo, hi), (x, a, b) = self.ranges
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if lo_s.free_symbols - {x} or hi_s.free_symbols - {x}:
            return None
        try:
            return sp.simplify(sp.integrate(hi_s - lo_s, (x, sp.sympify(a), sp.sympify(b))))
        except Exception:
            return None

    def _linear_data(self):
        if len(self.ranges) != 2:
            return None
        (y, lo, hi), (x, a, b) = self.ranges
        a = sp.sympify(a)
        b = sp.sympify(b)
        lo = sp.expand(sp.sympify(lo))
        hi = sp.expand(sp.sympify(hi))
        if (a.free_symbols | b.free_symbols):
            return None
        if lo.free_symbols - {x} or hi.free_symbols - {x}:
            return None
        try:
            plo = sp.Poly(lo, x)
            phi = sp.Poly(hi, x)
        except sp.PolynomialError:
            return None
        if plo.degree() > 1 or phi.degree() > 1:
            return None
        m1 = plo.nth(1) if plo.degree() >= 1 else sp.Integer(0)
        c1 = plo.nth(0)
        m2 = phi.nth(1) if phi.degree() >= 1 else sp.Integer(0)
        c2 = phi.nth(0)
        return x, y, a, b, m1, c1, m2, c2

    def reversed_pieces(self) -> list[list[tuple]] | None:
        data = self._linear_data()
        if data is None:
            return None
        x, y, a, b, m1, c1, m2, c2 = data

        def y_at(m, c, xv):
            return sp.simplify(m * xv + c)

        pts = [y_at(m1, c1, a), y_at(m1, c1, b), y_at(m2, c2, a), y_at(m2, c2, b)]
        try:
            if sp.simplify(m1 - m2) != 0:
                x_cross = sp.simplify((c2 - c1) / (m1 - m2))
                if float(sp.N(a)) - 1e-12 <= float(sp.N(x_cross)) <= float(sp.N(b)) + 1e-12:
                    pts.append(y_at(m1, c1, x_cross))
        except Exception:
            pass

        uniq = []
        for p in pts:
            p = sp.simplify(p)
            if p not in uniq:
                uniq.append(p)
        try:
            uniq = sorted(uniq, key=lambda t: float(sp.N(t)))
        except Exception:
            return None
        if len(uniq) < 2:
            return None

        # Reverse the graph by slicing the y-axis into intervals where the
        # active lower and upper x-bounds stay on the same candidate lines.
        def inv(m, c):
            if sp.simplify(m) == 0:
                return None
            return sp.simplify((y - c) / m)

        lower_cands = [a]
        upper_cands = [b]
        inv1 = inv(m1, c1)
        inv2 = inv(m2, c2)
        if inv1 is not None:
            if sp.N(m1) > 0:
                upper_cands.append(inv1)
            else:
                lower_cands.append(inv1)
        if inv2 is not None:
            if sp.N(m2) > 0:
                lower_cands.append(inv2)
            else:
                upper_cands.append(inv2)

        pieces = []
        for left, right in zip(uniq[:-1], uniq[1:]):
            if sp.simplify(left - right) == 0:
                continue
            mid = sp.simplify((left + right) / 2)

            def choose(cands, kind):
                best = None
                best_val = None
                for cand in cands:
                    val = float(sp.N(cand.subs(y, mid) if hasattr(cand, "subs") else cand))
                    if best is None:
                        best, best_val = cand, val
                    elif kind == "max" and val > best_val + 1e-12:
                        best, best_val = cand, val
                    elif kind == "min" and val < best_val - 1e-12:
                        best, best_val = cand, val
                return sp.simplify(best)

            xlo = choose(lower_cands, "max")
            xhi = choose(upper_cands, "min")
            try:
                if float(sp.N(xlo.subs(y, mid))) <= float(sp.N(xhi.subs(y, mid))) + 1e-12:
                    pieces.append([(x, xlo, xhi), (y, left, right)])
            except Exception:
                return None
        return pieces or None


@dataclass(frozen=True)
class SimplexRegion(IteratedRegion):
    dimension: int = 0

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.Integer(1) / sp.factorial(self.dimension)) if self.dimension else None

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = list(self.variables)
        try:
            poly = sp.Poly(sp.expand(expr), *vars_)
        except sp.PolynomialError:
            return None
        total = sp.Integer(0)
        for monom, coeff in poly.terms():
            deg_sum = sum(monom)
            numer = sp.Integer(1)
            for a in monom:
                numer *= sp.factorial(a)
            denom = sp.factorial(self.dimension + deg_sum)
            total += coeff * numer / denom
        return sp.simplify(total)


@dataclass(frozen=True)
class AffineSimplexRegion(Region):
    shifts: tuple[sp.Expr, ...] = field(default_factory=tuple)
    scales: tuple[sp.Expr, ...] = field(default_factory=tuple)
    dimension: int = 0

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return tuple(v for v, _, _ in self.ranges)

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return ("AffineSimplexRegion", tuple(sp.simplify(s) for s in self.shifts), tuple(sp.simplify(s) for s in self.scales))

    def constant_volume(self) -> sp.Expr | None:
        if not self.dimension or len(self.scales) != self.dimension:
            return None
        scale = sp.Integer(1)
        for s in self.scales:
            scale *= sp.Abs(sp.sympify(s))
        return sp.simplify(scale / sp.factorial(self.dimension))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables
        if len(vars_) != self.dimension:
            return None
        uvars = sp.symbols(f"_u0:{self.dimension}", real=True)
        subs = {v: sp.sympify(a) + sp.sympify(s) * u for v, a, s, u in zip(vars_, self.shifts, self.scales, uvars)}
        jac = sp.Integer(1)
        for s in self.scales:
            jac *= sp.Abs(sp.sympify(s))
        transformed = sp.expand(sp.sympify(expr).subs(subs) * jac)
        simplex_ranges = tuple((u, 0, 1 - sum(uvars[:i])) for i, u in enumerate(uvars))
        simplex = SimplexRegion(simplex_ranges, dimension=self.dimension)
        return simplex.polynomial_moment(transformed)


@dataclass(frozen=True)
class DiskRegion(IteratedRegion):
    radius: sp.Expr = sp.Integer(1)

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.pi * self.radius**2)

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = list(self.variables)
        try:
            poly = sp.Poly(sp.expand(expr), *vars_)
        except sp.PolynomialError:
            return None
        total = sp.Integer(0)
        dim = 2
        for monom, coeff in poly.terms():
            if any(a % 2 for a in monom):
                continue
            deg_sum = sum(monom)
            numer = sp.Integer(1)
            for a in monom:
                numer *= sp.gamma(sp.Rational(a + 1, 2))
            denom = sp.gamma(sp.Rational(deg_sum + dim, 2) + 1)
            total += coeff * self.radius ** (deg_sum + dim) * numer / denom
        return sp.simplify(total)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        x, y = self.variables
        r = sp.Symbol("_r", nonnegative=True, real=True)
        theta = sp.Symbol("_theta", real=True)
        # Convert to polar coordinates and split only when the transformed
        # integrand separates into a pure radial factor and a pure angular factor.
        polar_expr = _clean_expr(
            sp.trigsimp(
                sp.expand_trig(
                    sp.sympify(expr).subs({x: r * sp.cos(theta), y: r * sp.sin(theta)})
                )
            )
        )
        pieces = _split_dependence(polar_expr, (r,), (theta,))
        if pieces is None:
            return None
        radial_part, angular_part = pieces
        angular_val = sp.integrate(angular_part, (theta, 0, 2 * sp.pi))
        radial_val = sp.integrate(sp.simplify(radial_part * r), (r, 0, self.radius))
        return _clean_expr(angular_val * radial_val)


@dataclass(frozen=True)
class AnnulusRegion(Region):
    variables_xy: tuple[sp.Symbol, sp.Symbol] = field(default_factory=tuple)
    inner_radius: sp.Expr = sp.Integer(0)
    outer_radius: sp.Expr = sp.Integer(1)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_xy

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return ("AnnulusRegion", sp.simplify(self.inner_radius), sp.simplify(self.outer_radius))

    def constant_volume(self) -> sp.Expr | None:
        return sp.simplify(sp.pi * (self.outer_radius**2 - self.inner_radius**2))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        x, y = self.variables_xy
        outer = DiskRegion(((x, -self.outer_radius, self.outer_radius), (y, -sp.sqrt(self.outer_radius**2 - x**2), sp.sqrt(self.outer_radius**2 - x**2))), radius=self.outer_radius)
        inner = DiskRegion(((x, -self.inner_radius, self.inner_radius), (y, -sp.sqrt(self.inner_radius**2 - x**2), sp.sqrt(self.inner_radius**2 - x**2))), radius=self.inner_radius)
        out = outer.polynomial_moment(expr)
        inn = inner.polynomial_moment(expr)
        if out is None or inn is None:
            return None
        return sp.simplify(out - inn)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        x, y = self.variables_xy
        outer = DiskRegion(((x, -self.outer_radius, self.outer_radius), (y, -sp.sqrt(self.outer_radius**2 - x**2), sp.sqrt(self.outer_radius**2 - x**2))), radius=self.outer_radius)
        inner = DiskRegion(((x, -self.inner_radius, self.inner_radius), (y, -sp.sqrt(self.inner_radius**2 - x**2), sp.sqrt(self.inner_radius**2 - x**2))), radius=self.inner_radius)
        out = outer.radial_integral(expr)
        inn = inner.radial_integral(expr)
        if out is None or inn is None:
            return None
        return sp.simplify(out - inn)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return var in self.variables_xy


@dataclass(frozen=True)
class BallRegion(IteratedRegion):
    radius: sp.Expr = sp.Integer(1)
    dimension: int = 0

    def constant_volume(self) -> sp.Expr | None:
        dim = self.dimension
        return sp.simplify(sp.pi**(sp.Rational(dim, 2)) * self.radius**dim / sp.gamma(sp.Rational(dim, 2) + 1))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = list(self.variables)
        try:
            poly = sp.Poly(sp.expand(expr), *vars_)
        except sp.PolynomialError:
            return None
        total = sp.Integer(0)
        dim = self.dimension
        for monom, coeff in poly.terms():
            if any(a % 2 for a in monom):
                continue
            deg_sum = sum(monom)
            numer = sp.Integer(1)
            for a in monom:
                numer *= sp.gamma(sp.Rational(a + 1, 2))
            denom = sp.gamma(sp.Rational(deg_sum + dim, 2) + 1)
            total += coeff * self.radius ** (deg_sum + dim) * numer / denom
        return sp.simplify(total)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables
        dim = self.dimension
        if dim != 3 or len(vars_) != 3:
            t = sp.Symbol("_rho2", nonnegative=True, real=True)
            probes = []
            for v in vars_:
                subs = {w: sp.Integer(0) for w in vars_}
                subs[v] = sp.sqrt(t)
                probes.append(_clean_expr(expr.subs(subs)))
            first = probes[0]
            if any(sp.simplify(p - first) != 0 for p in probes[1:]):
                return None
            if set(vars_) & first.free_symbols:
                return None
            r = sp.Symbol("_r", nonnegative=True, real=True)
            sphere_area = 2 * sp.pi**(sp.Rational(dim, 2)) / sp.gamma(sp.Rational(dim, 2))
            radial_expr = _clean_expr(first.subs(t, r**2) * sphere_area * r**(dim - 1))
            return sp.integrate(radial_expr, (r, 0, self.radius))

        x, y, z = vars_
        r = sp.Symbol("_r", nonnegative=True, real=True)
        phi = sp.Symbol("_phi", real=True)
        theta = sp.Symbol("_theta", real=True)
        # The 3D case uses spherical coordinates and keeps only products that
        # separate cleanly into radial and angular pieces.
        spherical_expr = _clean_expr(
            sp.trigsimp(
                sp.expand_trig(
                    sp.sympify(expr).subs({
                        x: r * sp.sin(phi) * sp.cos(theta),
                        y: r * sp.sin(phi) * sp.sin(theta),
                        z: r * sp.cos(phi),
                    })
                )
            )
        )
        pieces = _split_dependence(spherical_expr, (r,), (theta, phi))
        if pieces is None:
            return None
        radial_part, angular_part = pieces
        angular_val = sp.integrate(angular_part * sp.sin(phi), (theta, 0, 2 * sp.pi), (phi, 0, sp.pi))
        radial_val = sp.integrate(_clean_expr(radial_part * r**2), (r, 0, self.radius))
        return _clean_expr(angular_val * radial_val)


@dataclass(frozen=True)
class EllipsoidRegion(Region):
    variables_nd: tuple[sp.Symbol, ...] = field(default_factory=tuple)
    axes: tuple[sp.Expr, ...] = field(default_factory=tuple)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_nd

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return ("EllipsoidRegion", tuple(sp.simplify(a) for a in self.axes))

    def constant_volume(self) -> sp.Expr | None:
        dim = len(self.axes)
        scale = sp.Integer(1)
        for a in self.axes:
            scale *= sp.Abs(sp.sympify(a))
        return sp.simplify(scale * sp.pi**(sp.Rational(dim, 2)) / sp.gamma(sp.Rational(dim, 2) + 1))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables_nd
        uvars = sp.symbols(f"_u0:{len(vars_)}", real=True)
        jac = sp.Integer(1)
        subs = {}
        for v, a, u in zip(vars_, self.axes, uvars):
            subs[v] = sp.sympify(a) * u
            jac *= sp.Abs(sp.sympify(a))
        transformed = sp.expand(sp.sympify(expr).subs(subs) * jac)
        ball = BallRegion(tuple((u, -1, 1) for u in uvars), radius=sp.Integer(1), dimension=len(vars_))
        return ball.polynomial_moment(transformed)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        vars_ = self.variables_nd
        t = sp.Symbol("_rho2", nonnegative=True, real=True)
        probes = []
        for v, a in zip(vars_, self.axes):
            subs = {w: sp.Integer(0) for w in vars_}
            subs[v] = sp.sympify(a) * sp.sqrt(t)
            probes.append(sp.simplify(expr.subs(subs)))
        first = probes[0]
        if any(sp.simplify(p - first) != 0 for p in probes[1:]):
            return None
        if set(vars_) & first.free_symbols:
            return None
        dim = len(vars_)
        r = sp.Symbol("_r", nonnegative=True, real=True)
        scale = sp.Integer(1)
        for a in self.axes:
            scale *= sp.Abs(sp.sympify(a))
        sphere_area = 2 * sp.pi**(sp.Rational(dim, 2)) / sp.gamma(sp.Rational(dim, 2))
        radial_expr = sp.simplify(first.subs(t, r**2) * scale * sphere_area * r**(dim - 1))
        return sp.integrate(radial_expr, (r, 0, 1))

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return var in self.variables_nd


@dataclass(frozen=True)
class SphericalShellRegion(Region):
    variables_nd: tuple[sp.Symbol, ...] = field(default_factory=tuple)
    inner_radius: sp.Expr = sp.Integer(0)
    outer_radius: sp.Expr = sp.Integer(1)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.variables_nd

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return ("SphericalShellRegion", sp.simplify(self.inner_radius), sp.simplify(self.outer_radius), len(self.variables_nd))

    def constant_volume(self) -> sp.Expr | None:
        dim = len(self.variables_nd)
        return sp.simplify(sp.pi**(sp.Rational(dim, 2)) * (self.outer_radius**dim - self.inner_radius**dim) / sp.gamma(sp.Rational(dim, 2) + 1))

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        outer = BallRegion(tuple((v, -self.outer_radius, self.outer_radius) for v in self.variables_nd), radius=self.outer_radius, dimension=len(self.variables_nd))
        inner = BallRegion(tuple((v, -self.inner_radius, self.inner_radius) for v in self.variables_nd), radius=self.inner_radius, dimension=len(self.variables_nd))
        out = outer.polynomial_moment(expr)
        inn = inner.polynomial_moment(expr)
        if out is None or inn is None:
            return None
        return sp.simplify(out - inn)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        outer = BallRegion(tuple((v, -self.outer_radius, self.outer_radius) for v in self.variables_nd), radius=self.outer_radius, dimension=len(self.variables_nd))
        inner = BallRegion(tuple((v, -self.inner_radius, self.inner_radius) for v in self.variables_nd), radius=self.inner_radius, dimension=len(self.variables_nd))
        out = outer.radial_integral(expr)
        inn = inner.radial_integral(expr)
        if out is None or inn is None:
            return None
        return sp.simplify(out - inn)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return var in self.variables_nd


@dataclass(frozen=True)
class UnionRegion(Region):
    pieces: tuple[Region, ...] = field(default_factory=tuple)

    @property
    def variables(self) -> tuple[sp.Symbol, ...]:
        return self.pieces[0].variables if self.pieces else tuple()

    def normalized_ranges(self) -> tuple[tuple, ...]:
        return tuple((type(reg).__name__, reg.normalized_ranges()) for reg in self.pieces)

    def constant_volume(self) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            vol = reg.constant_volume()
            if vol is None:
                return None
            total += vol
        return sp.simplify(total)

    def polynomial_moment(self, expr: sp.Expr) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            val = reg.polynomial_moment(expr)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

    def radial_integral(self, expr: sp.Expr) -> sp.Expr | None:
        total = sp.Integer(0)
        for reg in self.pieces:
            val = reg.radial_integral(expr)
            if val is None:
                return None
            total += val
        return sp.simplify(total)

    def is_reflection_invariant(self, var: sp.Symbol) -> bool:
        return all(reg.is_reflection_invariant(var) for reg in self.pieces)


def _poly_is_affine(expr: sp.Expr, var: sp.Symbol) -> bool:
    try:
        poly = sp.Poly(sp.expand(expr), var)
    except sp.PolynomialError:
        return False
    return poly.degree() <= 1


def _structural_ranges(ranges: list[tuple]) -> list[tuple]:
    """Convert public inner-first integration ranges to outer-first structural order."""
    return list(reversed(ranges))


def match_standard_simplex(ranges: list[tuple]) -> SimplexRegion | None:
    seen = []
    for idx, (var, lo, hi) in enumerate(_structural_ranges(ranges)):
        lo_s = sp.sympify(lo)
        hi_s = sp.expand(sp.sympify(hi))
        if lo_s != 0:
            return None
        if idx == 0:
            if sp.simplify(hi_s - 1) != 0:
                return None
        else:
            expected = 1 - sum(seen)
            if sp.simplify(hi_s - expected) != 0:
                return None
        seen.append(var)
    return SimplexRegion(tuple(ranges), dimension=len(ranges))


def match_affine_simplex(ranges: list[tuple]) -> AffineSimplexRegion | None:
    if len(ranges) < 2:
        return None
    sranges = _structural_ranges(ranges)
    # This matcher is intentionally limited to affine simplex bounds.
    for _, lo, hi in sranges:
        if hi is None:
            return None
    shifts = []
    scales = []
    prev_vars = []
    for idx, (var, lo, hi) in enumerate(sranges):
        lo_s = sp.simplify(sp.sympify(lo))
        hi_s = sp.expand(sp.sympify(hi))
        if idx == 0:
            if hi_s.free_symbols:
                return None
            scale = sp.simplify(hi_s - lo_s)
            if scale == 0:
                return None
            shifts.append(lo_s)
            scales.append(scale)
        else:
            if lo_s.free_symbols:
                return None
            if hi_s.free_symbols - set(prev_vars):
                return None
            for pv in prev_vars:
                if not _poly_is_affine(hi_s, pv):
                    return None
            target = sp.Integer(1)
            for pv, sh, sc in zip(prev_vars, shifts, scales):
                target -= sp.simplify((pv - sh) / sc)
            if target == 0:
                return None
            sc_i = sp.simplify((hi_s - lo_s) / target)
            if sc_i.free_symbols:
                return None
            if sp.simplify(lo_s + sc_i * target - hi_s) != 0:
                return None
            shifts.append(lo_s)
            scales.append(sc_i)
        prev_vars.append(var)
    if all(sp.simplify(sh) == 0 for sh in shifts) and all(sp.simplify(sc - 1) == 0 for sc in scales):
        return None
    return AffineSimplexRegion(tuple(ranges), shifts=tuple(reversed(shifts)), scales=tuple(reversed(scales)), dimension=len(ranges))


def match_graph_region(ranges: list[tuple]) -> GraphRegion | None:
    if len(ranges) != 2:
        return None
    (y, lo, hi), (x, a, b) = ranges
    a_s = sp.sympify(a)
    b_s = sp.sympify(b)
    lo_s = sp.sympify(lo)
    hi_s = sp.sympify(hi)
    if (a_s.free_symbols | b_s.free_symbols):
        return None
    if lo_s.free_symbols - {x} or hi_s.free_symbols - {x}:
        return None
    if not ((lo_s.free_symbols | hi_s.free_symbols) & {x}):
        return None
    if not (_poly_is_affine(lo_s, x) and _poly_is_affine(hi_s, x)):
        return None
    return GraphRegion(tuple(ranges), outer_var=x, inner_var=y)


def match_standard_disk(ranges: list[tuple]) -> DiskRegion | None:
    if len(ranges) != 2:
        return None
    (y, lo, hi), (x, a, b) = ranges
    a_s = sp.sympify(a)
    b_s = sp.sympify(b)
    lo_s = sp.sympify(lo)
    hi_s = sp.sympify(hi)
    if a_s.free_symbols or b_s.free_symbols:
        return None
    if sp.simplify(a_s + b_s) != 0:
        return None
    rad = sp.simplify(b_s)
    target = sp.sqrt(rad**2 - x**2)
    if sp.simplify(lo_s + target) == 0 and sp.simplify(hi_s - target) == 0:
        return DiskRegion(tuple(ranges), radius=rad)
    return None


def match_standard_ball(ranges: list[tuple]) -> BallRegion | None:
    if len(ranges) < 3:
        return None
    sranges = _structural_ranges(ranges)
    x0, a0, b0 = sranges[0]
    a0 = sp.sympify(a0)
    b0 = sp.sympify(b0)
    if a0.free_symbols or b0.free_symbols or sp.simplify(a0 + b0) != 0:
        return None
    rad = sp.simplify(b0)
    sumsq = x0**2
    for var, lo, hi in sranges[1:]:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        target = sp.sqrt(rad**2 - sumsq)
        if sp.simplify(lo_s + target) != 0 or sp.simplify(hi_s - target) != 0:
            return None
        sumsq += var**2
    return BallRegion(tuple(ranges), radius=rad, dimension=len(ranges))


def match_standard_ellipsoid(ranges: list[tuple]) -> EllipsoidRegion | None:
    if len(ranges) < 2:
        return None
    sranges = _structural_ranges(ranges)
    vars_ = []
    axes = []
    x0, a0, b0 = sranges[0]
    a0 = sp.sympify(a0)
    b0 = sp.sympify(b0)
    if a0.free_symbols or b0.free_symbols or sp.simplify(a0 + b0) != 0:
        return None
    axis0 = sp.simplify(b0)
    vars_.append(x0)
    axes.append(axis0)
    q = x0**2 / axis0**2
    for var, lo, hi in sranges[1:]:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        ratio = sp.simplify(hi_s / sp.sqrt(1 - q))
        if ratio.free_symbols:
            return None
        target = sp.simplify(ratio * sp.sqrt(1 - q))
        if sp.simplify(lo_s + target) != 0 or sp.simplify(hi_s - target) != 0:
            return None
        vars_.append(var)
        axes.append(sp.simplify(ratio))
        q += var**2 / axes[-1]**2
    if all(sp.simplify(a - axes[0]) == 0 for a in axes):
        return None
    return EllipsoidRegion(tuple(ranges), variables_nd=tuple(reversed(vars_)), axes=tuple(reversed(axes)))



def boole(cond: sp.Expr) -> sp.Piecewise:
    """Indicator-style helper analogous to Mathematica's Boole."""
    return sp.Piecewise((sp.Integer(1), sp.sympify(cond)), (sp.Integer(0), True))


def indicator_condition(expr: sp.Expr) -> sp.Expr | None:
    """Return the condition for an indicator-like Piecewise, else None."""
    expr = sp.sympify(expr)
    if not isinstance(expr, sp.Piecewise) or len(expr.args) != 2:
        return None
    (a1, c1), (a2, c2) = expr.args
    if c2 not in (True, sp.true):
        return None
    a1s = sp.sympify(a1)
    a2s = sp.sympify(a2)
    if a1s == 1 and a2s == 0:
        return sp.sympify(c1)
    if a1s == 0 and a2s == 1:
        return sp.Not(sp.sympify(c1))
    return None


def _extract_rel_bound(cond: sp.Expr):
    cond = sp.sympify(cond)
    if not isinstance(cond, Relational):
        return None
    lhs = sp.sympify(cond.lhs)
    rhs = sp.sympify(cond.rhs)
    return cond.rel_op, lhs, rhs


def _restrict_interval(region: Region, cond: sp.Expr) -> Region | None:
    if len(region.ranges) != 1:
        return None
    x, lo, hi = region.ranges[0]
    lo = sp.sympify(lo)
    hi = sp.sympify(hi)
    data = _extract_rel_bound(cond)
    if data is None:
        return None
    op, lhs, rhs = data
    if lhs == x and not rhs.free_symbols:
        if op in ("<", "<="):
            hi = sp.Min(hi, rhs)
        elif op in (">", ">="):
            lo = sp.Max(lo, rhs)
        else:
            return None
    elif rhs == x and not lhs.free_symbols:
        if op in ("<", "<="):
            lo = sp.Max(lo, lhs)
        elif op in (">", ">="):
            hi = sp.Min(hi, lhs)
        else:
            return None
    else:
        return None
    if sp.simplify(lo - hi) == 0:
        return BoxRegion(((x, lo, hi),))
    if lo.has(sp.Max) or hi.has(sp.Min):
        # Keep a conservative box if the symbolic ordering is not decidable.
        return BoxRegion(((x, lo, hi),))
    if (lo.is_real and hi.is_real and (lo.is_number and hi.is_number) and lo > hi):
        return None
    return BoxRegion(((x, lo, hi),))


def _restrict_box_2d(region: BoxRegion, cond: sp.Expr) -> Region | None:
    if len(region.ranges) != 2:
        return None
    (x, xlo, xhi), (y, ylo, yhi) = region.ranges
    xlo = sp.sympify(xlo); xhi = sp.sympify(xhi); ylo = sp.sympify(ylo); yhi = sp.sympify(yhi)
    data = _extract_rel_bound(cond)
    if data is not None:
        op, lhs, rhs = data
        # Conditions only on x or only on y keep us in a box.
        if lhs == x and not rhs.free_symbols:
            reg = _restrict_interval(BoxRegion(((x, xlo, xhi),)), cond)
            if reg is None:
                return None
            x, xlo, xhi = reg.ranges[0]
            return BoxRegion(((x, xlo, xhi), (y, ylo, yhi)))
        if lhs == y and not rhs.free_symbols:
            reg = _restrict_interval(BoxRegion(((y, ylo, yhi),)), cond)
            if reg is None:
                return None
            y, ylo, yhi = reg.ranges[0]
            return BoxRegion(((x, xlo, xhi), (y, ylo, yhi)))
        # y <= affine(x) / y >= affine(x) become graph regions.
        if lhs == y and (rhs.free_symbols <= {x}):
            if op in ("<", "<="):
                upper = rhs if sp.simplify(rhs - yhi) != 0 else yhi
                return GraphRegion(((y, ylo, upper), (x, xlo, xhi)), outer_var=x, inner_var=y)
            if op in (">", ">="):
                lower = rhs if sp.simplify(rhs - ylo) != 0 else ylo
                return GraphRegion(((y, lower, yhi), (x, xlo, xhi)), outer_var=x, inner_var=y)
        if rhs == y and (lhs.free_symbols <= {x}):
            if op in ("<", "<="):
                lower = lhs if sp.simplify(lhs - ylo) != 0 else ylo
                return GraphRegion(((y, lower, yhi), (x, xlo, xhi)), outer_var=x, inner_var=y)
            if op in (">", ">="):
                upper = lhs if sp.simplify(lhs - yhi) != 0 else yhi
                return GraphRegion(((y, ylo, upper), (x, xlo, xhi)), outer_var=x, inner_var=y)
        # Standard centered disk/annulus restrictions on a square.
        r2 = sp.expand(x**2 + y**2)
        if lhs == r2 and not rhs.free_symbols:
            if op in ("<", "<="):
                return DiskRegion(((y, -sp.sqrt(rhs - x**2), sp.sqrt(rhs - x**2)), (x, -sp.sqrt(rhs), sp.sqrt(rhs))), radius=sp.sqrt(rhs))
            if op in (">", ">=") and xlo == -xhi and ylo == -yhi and xlo == -1*sp.sympify(xhi):
                return AnnulusRegion((x, y), inner_radius=sp.sqrt(rhs), outer_radius=sp.sympify(xhi))
        if rhs == r2 and not lhs.free_symbols:
            if op in (">", ">="):
                return DiskRegion(((y, -sp.sqrt(lhs - x**2), sp.sqrt(lhs - x**2)), (x, -sp.sqrt(lhs), sp.sqrt(lhs))), radius=sp.sqrt(lhs))
    return None


def restrict_region(region: Region, cond: sp.Expr) -> Region | None:
    """Restrict a supported region by a simple Boolean condition.

    This is intentionally conservative and only supports conditions that can be
    represented by the current region model.
    """
    cond = sp.sympify(cond)
    if cond in (True, sp.true):
        return region
    if cond in (False, sp.false):
        return None
    if isinstance(cond, sp.Or):
        pieces = []
        for arg in cond.args:
            reg = restrict_region(region, arg)
            if reg is not None:
                pieces.append(reg)
        if not pieces:
            return None
        if len(pieces) == 1:
            return pieces[0]
        return UnionRegion(tuple(pieces))
    if isinstance(cond, sp.And):
        cur = region
        for arg in cond.args:
            cur = restrict_region(cur, arg)
            if cur is None:
                return None
        return cur
    if isinstance(region, BoxRegion):
        if len(region.ranges) == 1:
            return _restrict_interval(region, cond)
        if len(region.ranges) == 2:
            return _restrict_box_2d(region, cond)
    # Allow restrictions on plain iterated 1D/2D regions by reusing the same logic.
    if isinstance(region, IteratedRegion) and not isinstance(region, (SimplexRegion, GraphRegion, DiskRegion, BallRegion)):
        box_like = BoxRegion(tuple(region.ranges))
        reg = restrict_region(box_like, cond)
        if reg is not None:
            return reg
    return None

def region_from_ranges(ranges) -> Region:
    norm = _normalize_ranges_input(ranges)
    if isinstance(norm, Region):
        return norm
    ranges = norm

    simplex = match_standard_simplex(ranges)
    if simplex is not None:
        return simplex
    disk = match_standard_disk(ranges)
    if disk is not None:
        return disk
    ball = match_standard_ball(ranges)
    if ball is not None:
        return ball
    ellipsoid = match_standard_ellipsoid(ranges)
    if ellipsoid is not None:
        return ellipsoid
    affine_simplex = match_affine_simplex(ranges)
    if affine_simplex is not None:
        return affine_simplex
    graph = match_graph_region(ranges)
    if graph is not None:
        return graph

    vars_set = {r[0] for r in ranges}
    is_box = True
    for _, lo, hi in ranges:
        lo_s = sp.sympify(lo)
        hi_s = sp.sympify(hi)
        if (lo_s.free_symbols | hi_s.free_symbols) & vars_set:
            is_box = False
            break
    if is_box:
        return BoxRegion(tuple(ranges))
    return IteratedRegion(tuple(ranges))
