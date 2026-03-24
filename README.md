# SymbolicMath

**SymbolicMath** is a small collection of symbolic-math tools built on top of Python and SymPy.

This repository currently contains two main packages:

- **MultipleIntegrate** — exact symbolic definite integration for many multiple integrals
- **BranchCuts** — symbolic tools for analyzing branch cuts, multivalued expressions, and complex-function behavior

The goal of the repo is to provide practical symbolic methods for problems that are often awkward in standard computer algebra workflows.

---

## Packages

### MultipleIntegrate

`MultipleIntegrate` focuses on **exact definite integration**, especially for structured families of multiple integrals.

It includes support for many cases such as:

- product-region multiple integrals
- Gaussian integrals and moments
- rational full-line integrals
- polynomial moment families
- selected transform-friendly trigonometric and exponential integrals
- special-function outputs involving logarithms, arctangents, gamma, and beta functions

The package uses a strategy-based approach rather than relying only on direct antiderivatives.

---

### BranchCuts

`BranchCuts` focuses on the symbolic analysis of **branch cuts** and related issues for complex elementary and special functions.

Typical goals include:

- identifying where expressions are discontinuous in the complex plane
- determining branch-cut structure under composition
- simplifying expressions while respecting branch behavior
- helping reason about multivalued functions such as logarithms, roots, inverse trig functions, and related expressions

This package is useful when exact symbolic transformations depend on choosing the correct complex branch structure.

---

## Repository structure

```text
SymbolicMath/
├── MultipleIntegrate/
├── BranchCuts/
├── README.md
└── ...
```

---

## Installation

Install the repository in editable mode:

```bash
pip install -e .
```

If the packages are kept as separate subpackages, install dependencies such as:

```bash
pip install sympy pytest
```

---

## Example areas of use

- exact symbolic evaluation of definite integrals
- Gaussian and moment computations
- multivariate integral simplification
- branch-cut detection for logarithms, roots, and inverse functions
- symbolic preprocessing for complex-analysis or special-function workflows

---

## Status

This repository is aimed at **exact symbolic computation**, with an emphasis on practical structured methods rather than fully general computer algebra for every case.

Both packages are designed to be extended incrementally as new symbolic strategies and recognizers are added.

---

## Author

**Bhuvanesh Bhatt**

---

## License

See the license file(s) included in the repository.
