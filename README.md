# TruncatedPopulations

This repository contains a small Julia investigation of hierarchical inference for
truncated populations, with a focus on latent-variable models where the true
population support is bounded (e.g. a quantity constrained to lie in `[0, 1]`).

The main analysis is in:

- `notebooks/TruncatedQ.ipynb`

The notebook compares two approaches for inferring population parameters:

1. An "exact" model that samples latent true values for each observation.
2. A Monte Carlo marginalization model that integrates over per-observation
   samples and tracks effective sample sizes.

In this setup, both approaches appear to give consistent population inferences
for the truncated-distribution example explored in the notebook.

## Repository layout

- `notebooks/TruncatedQ.ipynb`: exploratory analysis and model comparison.
- `src/TruncatedPopulations.jl`: lightweight Julia module utilities.
- `Project.toml` / `Manifest.toml`: pinned Julia environment and dependencies.

## Getting started

1. Start Julia in this directory.
2. Activate the environment and instantiate dependencies:

```julia
using Pkg
Pkg.activate(".")
Pkg.instantiate()
```

3. Open and run `notebooks/TruncatedQ.ipynb` in your preferred notebook
   environment (e.g. VS Code Jupyter with a Julia kernel).

## Notes

This repository is currently an investigation notebook plus minimal module code,
rather than a polished package API.
