
from branchcuts import classification_summary, classify_name, scipy_special_classification, mpmath_classification, sympy_classification


def test_classification_summary_has_expected_sections():
    summary = classification_summary()
    assert "scipy.special" in summary
    assert "mpmath" in summary
    assert "sympy" in summary
    assert summary["scipy.special"]["total"] > 0
    assert summary["mpmath"]["total"] > 0
    assert summary["sympy"]["total"] > 0


def test_known_names_classify_sensibly():
    assert classify_name("log").category == "implemented_jump_formula"
    assert classify_name("lambertw").category in {
        "implemented_jump_formula",
        "branch_cut_known_needs_formula",
        "convention_dependent_or_backend_specific",
    }
    assert classify_name("gamma").category in {
        "meromorphic_or_poles_only",
        "unknown_needs_review",
    }


def test_classification_maps_include_examples():
    scipy_map = scipy_special_classification()
    mpmath_map = mpmath_classification()
    sympy_map = sympy_classification()
    assert "lambertw" in scipy_map
    assert "lambertw" in mpmath_map
    assert "LambertW" in sympy_map
