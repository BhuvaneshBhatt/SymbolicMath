# Architecture overview

The package is organized around four layers:

1. **Explicit jump rules** in `core.py`
2. **Recursive propagation** in `computation.py`
3. **Condition cleanup** in `post_simplify.py`
4. **Coverage metadata** in `inventory.py` and `classification.py`

The design goal is to provide a practical symbolic and numeric framework for branch-cut reasoning in Python.
