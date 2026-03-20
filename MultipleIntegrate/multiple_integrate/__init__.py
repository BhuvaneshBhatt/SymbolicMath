"""
multiple_integrate
==================
Symbolic multiple integration for Python.

Evaluates n-dimensional integrals of the form

    ∫_Ω f(g(x₁, …, xₙ)) dx

in closed symbolic form using SymPy.  Nine specialised strategies are tried in
order; the first that applies returns an exact result.  A plain iterated
fallback handles all remaining cases.

Public API
----------
multiple_integrate(f, *ranges, ...)
    Main entry point.  See ``help(multiple_integrate)`` for full signature.

Decomposition
    Dataclass returned by the internal decomposition step.  Exposed here for
    users who want to inspect how an integrand was parsed.

Examples
--------
>>> from sympy import symbols, exp, sin, pi, oo
>>> from multiple_integrate import multiple_integrate
>>> x, y = symbols('x y', real=True)
>>> multiple_integrate(exp(-(x**2 + y**2)), (x, -oo, oo), (y, -oo, oo))
pi
>>> multiple_integrate(sin(x + y), (x, 0, pi), (y, 0, pi))
0
"""

from multiple_integrate.core import (
    multiple_integrate,
    Decomposition,
)

__all__ = [
    "multiple_integrate",
    "Decomposition",
]

__version__ = "2.0.0"
__author__ = "MultipleIntegrate Contributors"
__license__ = "GPL-3.0-or-later"
