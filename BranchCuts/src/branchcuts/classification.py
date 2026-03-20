
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .inventory import (
    scipy_special_inventory,
    mpmath_inventory,
    sympy_inventory,
    implemented_rule_names,
)

CATEGORIES = [
    "implemented_jump_formula",
    "branch_cut_known_needs_formula",
    "meromorphic_or_poles_only",
    "likely_entire_or_single_valued_no_branch_cut",
    "convention_dependent_or_backend_specific",
    "nonanalytic_distributional_real_only_or_utility",
    "unknown_needs_review",
]


@dataclass(frozen=True)
class Classification:
    category: str
    reason: str


# Canonical aliases for names that already have explicit symbolic jump formulas.
IMPLEMENTED_ALIASES: dict[str, set[str]] = {
    "log": {"log", "ln"},
    "sqrt": {"sqrt"},
    "power": {"pow", "power"},
    "acos": {"acos"},
    "asin": {"asin"},
    "atanh": {"atanh"},
    "atan": {"atan"},
    "asinh": {"asinh"},
    "acosh": {"acosh"},
    "acot": {"acot"},
    "acoth": {"acoth"},
    "asec": {"asec"},
    "acsc": {"acsc"},
    "asech": {"asech"},
    "acsch": {"acsch"},
    "Ei": {"ei", "expi"},
    "Ci": {"ci"},
    "Chi": {"chi"},
    "li": {"li", "logintegral"},
    "expint": {"expint", "expn", "exp1", "e1"},
    "loggamma": {"loggamma"},
    "polylog": {"polylog"},
    "lerchphi": {"lerchphi"},
    "lambertw": {"lambertw", "productlog", "wrightomega"},
    "hyperu": {"hyperu"},
    "betainc": {"betainc"},
    "betainc_regularized": {"betainc_regularized"},
    "elliptic_k": {"ellipk", "elliptic_k"},
    "elliptic_e": {"ellipe", "elliptic_e"},
    "elliptic_f": {"ellipf", "elliptic_f"},
    "elliptic_pi": {"ellippi", "elliptic_pi"},
    "assoc_legendre": {"assoc_legendre", "assoc_legendre_p", "lpmv"},
    "besselj": {"besselj", "jv"},
    "besseli": {"besseli", "iv"},
    "bessely": {"bessely", "yv"},
    "besselk": {"besselk", "kv"},
    "hankel1": {"hankel1"},
    "hankel2": {"hankel2"},
    "hyper": {"hyper", "hyp2f1"},
}

IMPLEMENTED_NAMES = {alias for aliases in IMPLEMENTED_ALIASES.values() for alias in aliases}

# Functions that are well known to have branch cuts on their principal branches,
# but for which this package does not yet provide a symbolic jump formula.
KNOWN_BRANCH_CUT_NAMES = {
    # elementary / inverse elementary
    "arg",
    "lambertw", "wrightomega", "li", "ci", "chi", "shi", "spence",
    "exp1", "e1", "expint", "expn",
    # Bessel / Hankel / Kelvin / Legendre / spheroidal families
    "besseli", "besselj", "besselk", "bessely", "jv", "yv", "iv", "kv",
    "hankel1", "hankel2", "airye",
    "bei", "ber", "kei", "ker", "beip", "berp", "keip", "kerp",
    "legendreq", "legenq", "assoc_legendre", "assoc_legendre_p", "lpmv", "lqmn", "lqn",
    "legendre_p", "legendre_p_all", "sph_legendre_p", "sph_legendre_p_all",
    "spherical_in", "spherical_kn", "spherical_yn", "spherical_jn",
    "sph_harm_y", "sph_harm_y_all", "spherharm", "ynm", "znm",
    "mathieuc", "mathieucprime", "mathieus", "mathieusprime",
    "mathieu_cem", "mathieu_sem", "mathieu_modcem1", "mathieu_modcem2", "mathieu_modsem1", "mathieu_modsem2",
    "whitm", "whitw", "webere", "angerj", "coulombf", "coulombg", "coulombc",
    "lommels1", "lommels2", "struve", "modstruve", "struveh", "struvel",
    # hypergeometric / Meijer / Lerch / polylog-adjacent
    "hyp1f2", "hyp2f0", "hyp2f2", "hyp2f3", "hyp3f2",
    "appellf1", "appellf2", "appellf3", "appellf4", "hyper", "hyper2d", "meijerg",
    "lerchphi", "hyp0f1", "hyp1f1", "hyp2f1", "log_wright_bessel", "wright_bessel",
    # gamma/beta incomplete and regularized variants
    "betaincc", "betaincinv", "betainccinv",
    "gammainc", "gammaincc", "gammaincinv", "gammainccinv", "lowergamma", "uppergamma",
    # elliptic families
    "ellipf", "ellipeinc", "ellipkinc", "ellipkm1",
    "elliptic_f",
    "elliprc", "elliprd", "elliprf", "elliprg", "elliprj",
    # zeta / eta / Barnes / related
    "zeta", "altzeta", "hurwitz", "lerchphi", "barnesg", "stieltjes", "kleinj",
    "dirichlet_eta", "riemann_xi",
    # logarithm-like helpers
    "log1p",
}

# Meromorphic (as functions of their main complex variable/parameter) or dominated by poles.
MEROMORPHIC_NAMES = {
    "gamma", "digamma", "polygamma", "trigamma", "factorial", "factorial2", "factorialk",
    "beta", "binom", "binomial", "multigamma", "multigammaln", "poch", "rf", "ff",
    "fac", "fac2", "barnesg", "bernoulli", "euler", "eulernum", "harmonic",
}

# Entire or at least single-valued without branch cuts in the principal variable.
NO_BRANCH_NAMES = {
    "sin", "cos", "tan", "cot", "sec", "csc", "sinh", "cosh", "tanh", "coth", "sech", "csch",
    "exp", "expm1", "exp2", "exp10", "expit", "cosm1", "sinc", "sincpi", "sinpi", "cospi",
    "erf", "erfc", "erfi", "erfinv", "erfcinv", "erf2", "erf2inv",
    "hyp0f1", "hyp1f1",
    "fresnelc", "fresnels", "airyai", "airybi", "airyaiprime", "airybiprime",
    "j0", "j1", "y0", "y1", "i0", "i1", "k0", "k1", "i0e", "i1e", "k0e", "k1e",
    "chebyt", "chebyu", "chebyc", "chebys", "gegenbauer", "genlaguerre", "laguerre",
    "hermite", "hermitenorm", "hermite_prob", "jacobi", "legendre", "chebyshevt", "chebyshevu",
    "assoc_laguerre", "jn", "yn", "jn", "j0", "j1", "ivp", "jvp", "kvp", "yvp", "h1vp", "h2vp",
    "fibonacci", "bell", "catalan", "apery", "glaisher", "khinchin", "mertens",
    "superfac", "hyperfac", "stirling1", "stirling2", "e", "ln2", "ln10", "pi", "degree", "degrees",
    "agm", "dawsn", "wofz", "voigt_profile",
}

# Functions that are clearly utilities, distributional objects, root-finders,
# array helpers, zero tables, or essentially real-only statistical transforms.
UTILITY_OR_REAL_ONLY_NAMES = {
    "abs", "absmax", "absmin", "max", "min", "piecewise", "rem", "adjoint", "conjugate", "transpose",
    "re", "im", "sign", "ceiling", "floor", "frac", "periodic_argument", "principal_branch",
    "polar_lift", "exp_polar", "arg", "heaviside", "diracdelta", "kroneckerdelta", "levicivita",
    "singularityfunction", "contains", "atan2", "boxcox", "boxcox1p", "inv_boxcox", "inv_boxcox1p",
    "xlogy", "xlog1py", "logsumexp", "log_softmax", "softmax", "entr", "kl_div", "rel_entr", "huber",
    "bernoulli", "euler", "comb", "perm", "stirling2",
    "geterr", "seterr", "test", "autoprec", "memoize", "workdps", "workprec", "extradps", "extraprec",
    "almosteq", "identify", "findroot", "limit", "taylor", "diff", "differint", "diffs", "diffs_exp",
    "diffs_prod", "diffun", "jacobian", "quad", "quadgl", "quadosc", "quadsubdiv", "quadts",
    "cholesky", "cholesky_solve", "det", "diag", "eig", "eig_sort", "eigh", "eighe", "eigsy", "expm",
    "sqrtm", "logm", "hessenberg", "hilbert", "lu", "lu_solve", "inverse", "svd", "svd_c", "svd_r",
    "zeros", "eye", "arange", "linspace", "unitroots", "unit_triangle", "unitvector", "swap_row",
    "convert", "extend", "fabs", "fadd", "fdiv", "fdot", "fmul", "fneg", "frexp", "fsub", "fsum",
    "mag", "make_mpc", "make_mpf", "mfrom", "mnorm", "mpc", "mpf", "isfinite", "isinf", "isint",
    "isnan", "isnormal", "chop", "cond", "almosteq", "fourier", "fourierval", "gauss_quadrature",
    "cplot", "splot", "plot", "timing", "invertlaplace", "invlapdehoog", "invlapstehfest", "invlaptalbot",
}

CONVENTION_DEPENDENT_NAMES = {
    "assoc_legendre", "assoc_legendre_p", "assoc_legendre_p_all", "lpmv", "lqmn", "lqn", "legenp", "legenq",
    "sph_harm_y", "sph_harm_y_all", "spherharm", "ynm", "znm",
    "lambertw", "wrightomega",
    "exp_polar", "periodic_argument", "principal_branch", "polar_lift",
    "legendre_p", "legendre_p_all", "sph_legendre_p", "sph_legendre_p_all",
}

# Common suffix/prefix patterns.
UTILITY_PATTERNS = (
    "_zeros", "_roots", "_root", "_coef", "_all", "_inv",
)

REAL_ONLY_PREFIXES = (
    "bdtr", "chdtr", "chndtr", "fdtr", "gdtr", "nbdtr", "ncfdtr", "nctdtr", "ndtr", "nrdtrimn",
    "pdtr", "smirnov", "stdtr", "stdtrit", "tklmbda", "kolmog", "kolmogi", "owens_t",
)

# SciPy statistical distributions and helpers are usually real-only APIs.
REAL_ONLY_SUBSTRINGS = (
    "tri", "idf", "inc",  # only used heuristically after more specific rules fail
)


def _normalize(name: str) -> str:
    return name.strip().lower()


def classify_name(name: str, module: str | None = None) -> Classification:
    n = _normalize(name)
    m = (module or "").lower()

    if n in IMPLEMENTED_NAMES:
        return Classification("implemented_jump_formula", "Explicit symbolic jump formula implemented in this package.")

    if n in CONVENTION_DEPENDENT_NAMES:
        return Classification("convention_dependent_or_backend_specific", "Principal branch or convention differs across libraries or requires explicit branch selection.")

    if n in KNOWN_BRANCH_CUT_NAMES:
        return Classification("branch_cut_known_needs_formula", "Known principal branch cut family, but no explicit symbolic jump formula is implemented yet.")

    if n in MEROMORPHIC_NAMES:
        return Classification("meromorphic_or_poles_only", "Usually meromorphic in its main complex variable; poles/residues are more relevant than additive branch jumps.")

    if n in NO_BRANCH_NAMES:
        return Classification("likely_entire_or_single_valued_no_branch_cut", "Likely entire or single-valued without a principal branch cut in the main variable.")

    if n in UTILITY_OR_REAL_ONLY_NAMES:
        return Classification("nonanalytic_distributional_real_only_or_utility", "Utility, matrix/helper routine, distributional object, or primarily real-valued API.")

    if any(n.endswith(suf) for suf in UTILITY_PATTERNS):
        return Classification("nonanalytic_distributional_real_only_or_utility", "Root/zero table, coefficient generator, inverse helper, or bulk-evaluation utility rather than a single complex-analytic function.")

    if any(n.startswith(pref) for pref in REAL_ONLY_PREFIXES):
        return Classification("nonanalytic_distributional_real_only_or_utility", "Mostly statistical CDF/quantile style API intended for real-valued arguments.")

    if "delta_functions" in m or "tensor_functions" in m:
        return Classification("nonanalytic_distributional_real_only_or_utility", "Distributional/tensor-style symbolic object, not a single-valued branched analytic function.")

    if "special.polynomials" in m or "orthogonal" in m:
        return Classification("likely_entire_or_single_valued_no_branch_cut", "Polynomial family or finite special function; typically no branch cut in the polynomial variable.")

    if "special.error_functions" in m:
        # Some error/integral functions are branched; leave only known entire ones above.
        return Classification("unknown_needs_review", "Special integral/error-function family requires function-specific review.")

    if n.startswith("ellip") or n.startswith("elliptic_"):
        return Classification("branch_cut_known_needs_formula", "Elliptic family; branch cuts are expected but often parameter- and argument-dependent.")

    if n.startswith("hyp") or n in {"hyper", "hyper2d", "meijerg", "appellf1", "appellf2", "appellf3", "appellf4"}:
        return Classification("branch_cut_known_needs_formula", "Hypergeometric/Meijer family; branch cuts depend on parameters and continuation formulas.")

    if n.startswith("bessel") or n in {"jv", "yv", "iv", "kv", "hankel1", "hankel2", "kei", "ker", "bei", "ber", "beip", "berp", "keip", "kerp"}:
        return Classification("branch_cut_known_needs_formula", "Bessel/Kelvin/Hankel family with parameter-dependent principal cuts or phase conventions.")

    if n.startswith("spherical_") or n.startswith("sph_"):
        return Classification("convention_dependent_or_backend_specific", "Spherical/spheroidal harmonics and associated functions often depend on normalization and phase conventions.")

    if n in {"gamma", "beta", "digamma", "polygamma", "trigamma", "zeta", "hurwitz", "dirichlet_eta", "riemann_xi"}:
        return Classification("unknown_needs_review", "Analytic behavior depends strongly on which variable is considered primary; review needed.")

    return Classification("unknown_needs_review", "No reliable automatic classification rule matched; manual review still needed.")


def classify_inventory_names(names: list[str], modules: dict[str, str] | None = None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for name in names:
        module = modules.get(name, "") if modules else ""
        cls = classify_name(name, module=module)
        out[name] = {"category": cls.category, "reason": cls.reason}
    return out


def scipy_special_classification() -> dict[str, dict[str, str]]:
    names = scipy_special_inventory()
    return classify_inventory_names(names)


def mpmath_classification() -> dict[str, dict[str, str]]:
    names = mpmath_inventory()
    return classify_inventory_names(names)


def sympy_classification() -> dict[str, dict[str, str]]:
    items = sympy_inventory()
    modules = {item["name"]: item.get("module", "") for item in items}
    names = [item["name"] for item in items]
    return classify_inventory_names(names, modules=modules)


def summarize_classification(mapping: dict[str, dict[str, str]]) -> dict[str, Any]:
    counts = {cat: 0 for cat in CATEGORIES}
    by_category = {cat: [] for cat in CATEGORIES}
    for name, info in mapping.items():
        cat = info["category"]
        counts[cat] = counts.get(cat, 0) + 1
        by_category.setdefault(cat, []).append(name)
    for cat in by_category:
        by_category[cat].sort()
    return {
        "counts": counts,
        "by_category": by_category,
        "total": sum(counts.values()),
    }


def classification_summary() -> dict[str, Any]:
    scipy_map = scipy_special_classification()
    mpmath_map = mpmath_classification()
    sympy_map = sympy_classification()
    return {
        "implemented_rule_names": implemented_rule_names(),
        "scipy.special": summarize_classification(scipy_map),
        "mpmath": summarize_classification(mpmath_map),
        "sympy": summarize_classification(sympy_map),
        "notes": [
            "These categories are a scaffolding for systematic expansion of the jump registry.",
            "The classifications are partly heuristic and should not be treated as mathematically authoritative for every variable choice or every backend convention.",
        ],
    }
