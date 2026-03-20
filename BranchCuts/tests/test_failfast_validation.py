import pytest
from sympy import symbols, log
from branchcuts import branch_cut_jumps, post_simplify_pairs, merge_duplicate_pairs

def test_post_simplify_pairs_rejects_dict_with_hint():
    z = symbols("z", complex=True)
    result = branch_cut_jumps(log(z), z, mode="pairs")
    with pytest.raises(TypeError) as excinfo:
        post_simplify_pairs(result, z)
    msg = str(excinfo.value)
    assert "expected a list/tuple of (expr, condition) pairs" in msg
    assert "result['Jump']" in msg or "Jump" in msg

def test_merge_duplicate_pairs_rejects_non_pairs():
    z = symbols("z", complex=True)
    result = branch_cut_jumps(log(z), z, mode="pairs")
    with pytest.raises(TypeError):
        merge_duplicate_pairs(result, z)

def test_branch_cut_jumps_rejects_bad_mode():
    z = symbols("z", complex=True)
    with pytest.raises(ValueError):
        branch_cut_jumps(log(z), z, mode="badmode")
