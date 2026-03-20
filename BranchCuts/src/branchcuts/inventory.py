from __future__ import annotations

import json
from importlib.resources import files
from typing import Any

from .core import IMPLEMENTED_RULE_NAMES


def _load_json(name: str) -> dict[str, Any]:
    data_path = files('branchcuts').joinpath('data').joinpath(name)
    return json.loads(data_path.read_text())


def scipy_special_inventory() -> list[str]:
    return list(_load_json('scipy_special_public_callables.json')['names'])


def mpmath_inventory() -> list[str]:
    return list(_load_json('mpmath_public_callables.json')['names'])


def sympy_inventory() -> list[dict[str, str]]:
    return list(_load_json('sympy_elementary_special_functions.json')['functions'])


def implemented_rule_names() -> list[str]:
    return sorted(IMPLEMENTED_RULE_NAMES)


def coverage_summary() -> dict[str, Any]:
    scipy_names = scipy_special_inventory()
    mpmath_names = mpmath_inventory()
    sympy_names = [item['name'] for item in sympy_inventory()]
    implemented = set(implemented_rule_names())

    def summarize(names: list[str]) -> dict[str, Any]:
        found = sorted(set(names) & implemented)
        missing = sorted(set(names) - implemented)
        return {
            'total': len(names),
            'implemented_count': len(found),
            'implemented': found,
            'missing_count': len(missing),
            'missing_sample': missing[:50],
        }

    return {
        'implemented_rule_names': sorted(implemented),
        'scipy.special': summarize(scipy_names),
        'mpmath': summarize(mpmath_names),
        'sympy': summarize(sympy_names),
    }
