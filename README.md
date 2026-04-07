# SymbolicMath

**SymbolicMath** is a small collection of symbolic math tools built on top of Python and SymPy.

This repository currently contains three main packages:

- **MultipleIntegrate** — symbolic definite integration for many multiple integrals
- **BranchCuts** — symbolic tools for analyzing branch cuts, multivalued expressions, and complex function behavior
- **PDESolve** — symbolic PDE solver, currently using only Lie-symmetry based method

The goal of the repo is to provide practical symbolic methods for problems that are not addressed by SymPy.

---

## Packages

### MultipleIntegrate

`MultipleIntegrate` focuses on **symbolic definite integration**, especially for structured families of multiple integrals.

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

### PDESolve

`liepde`, under PDESolve, is a standalone package for **Lie-symmetry based PDE reduction and symbolic solution workflows**.

It is designed for scalar PDEs written with SymPy and focuses on practical symmetry-based reduction rather than fully general PDE solving. The package can return the best available result from its workflow: a full explicit solution when available, otherwise a lifted similarity solution, otherwise a reduced equation.

Current capabilities include:

- scalar jet-space PDE parsing
- automatic PDE-order inference
- solved-form principal derivative selection
- determining-equation construction for Lie point symmetries
- polynomial ansatz solving for infinitesimals
- symbolic search for reducible symmetry combinations
- Frobenius-chart based reductions
- repeated reduction workflow utilities
- reduced-equation solving
- lifting reduced solutions back to the original variables
- configurable result levels and detailed diagnostics
- analysis-only mode for PDE inspection without a full solve attempt
- direct fallback solving with SymPy where appropriate
- pattern-based fallback solutions for several standard PDE families, including:
  - transport equations
  - heat / diffusion equations
  - advection-diffusion equations
  - wave equations

---

## Repository structure

```text
SymbolicMath/
├── MultipleIntegrate/
├── BranchCuts/
├── PDESolve/
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

## Status

All of the packages are designed to be extended incrementally as new symbolic strategies and recognizers are added.

---

## Author

**Bhuvanesh Bhatt**

---

## License

See the license file(s) included in the repository.
