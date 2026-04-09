# Testing

The test suite now lives in the top-level `tests/` directory and is organized as many focused pytest modules rather than a single monolithic file.

---

## Running tests

From the project root:

```bash
pytest
```

For a quicker targeted run:

```bash
pytest tests/test_simplex_dirichlet_engine.py
pytest tests/test_coordinate_changes.py
pytest tests/test_regions_parse.py
```

If you are not using an editable install, make sure the package is importable from `src/`, for example by using:

```bash
PYTHONPATH=src pytest
```

---

## Current test organization

The suite is split by topic. Representative modules include:

### Region parsing and classification

- `test_regions_parse.py`
- `test_region_order_sympy_convention.py`
- `test_regions_boxes_and_simplex.py`
- `test_regions_disk_and_ball.py`
- `test_regions_graph_and_simplex.py`
- `test_regions_advanced_shapes.py`
- `test_regions_classification_priority.py`
- `test_regions_misclassification.py`

### Region formulas and symmetry

- `test_regions_moments_box_simplex.py`
- `test_regions_moments_disk_ball.py`
- `test_regions_radial.py`
- `test_regions_symmetry.py`
- `test_regions_graph_reversal.py`
- `test_regions_interactions.py`

### Exact-family and strategy regressions

- `test_simplex_dirichlet_engine.py`
- `test_coordinate_changes.py`
- `test_disk_radial_false_positive.py`
- `test_supported_families.py`
- `test_reference_examples.py`
- `test_reference_examples_notebook.py`
- `test_regression_constants_and_caching.py`
- `test_regression_symmetry_and_inner_1d.py`

### Performance and edge cases

- `test_performance_regressions.py`
- `test_singular_and_divergent_cases.py`
- `test_boole_piecewise.py`
- `test_multiple_integrate.py`

The large `test_multiple_integrate.py` file still exists as a broad integration/regression file, but it is no longer the sole test suite.

---

## What the suite is meant to protect

The tests are designed to catch regressions in:

- the SymPy-style inner-first range convention
- region classification
- exact simplex / Dirichlet formulas
- coordinate-change paths
- symmetry reductions
- known radiality edge cases
- representative reference examples from the notebook and docs
- selected performance-sensitive examples

---

## Recommended workflow for contributors

When changing region parsing or structured formulas, run at least:

```bash
pytest tests/test_regions_parse.py \
       tests/test_region_order_sympy_convention.py \
       tests/test_simplex_dirichlet_engine.py \
       tests/test_coordinate_changes.py
```

When changing fallback logic or general integration behavior, also run:

```bash
pytest tests/test_multiple_integrate.py \
       tests/test_supported_families.py \
       tests/test_reference_examples.py
```

---

## Writing new tests

When adding a new capability, prefer a focused test module or extend the most relevant existing one.

Useful principles:

- include at least one direct success case
- include at least one bypass / inapplicable case
- include one regression test if the change fixes a previous bug
- prefer exact symbolic comparisons where practical
- add a timing-oriented regression only for cases known to matter
