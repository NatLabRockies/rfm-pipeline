---
title: 'rfm-pipeline: A Python package for reduced-form modeling pipelines'
tags:
  - Python
  - reduced-form modeling
  - sensitivity analysis
  - feature selection
  - HPC
authors:
  - name: Dylan Hettinger
    orcid: 0000-0000-0000-0000 # TODO: replace with real ORCID
    affiliation: 1
affiliations:
  - name: National Renewable Energy Laboratory
    index: 1
date: 2025-01-01 # TODO: update to submission date
bibliography: paper.bib
---

# Summary

`rfm-pipeline` is a Python package for constructing and executing reduced-form
modeling (RFM) pipelines on large simulation or observational datasets. A
reduced-form model approximates a high-fidelity simulator's input–output
behavior using a computationally inexpensive statistical surrogate. Building
such models requires systematic identification of which inputs matter, how they
interact, whether nonlinear transformations are needed, and which terms are
stable across resamples of the data.

The package implements a six-stage pipeline: empirical-null screening with
Benjamini–Hochberg FDR control [@BenjaminiHochberg1995], interaction discovery
via tree SHAP interaction values [@LundbergEtAl2020TreeSHAP; @LundbergLee2017],
nonlinear discovery using generalized additive model diagnostics
[@HastieTibshirani1990], sparse selection with LASSO [@Tibshirani1996] and
EBIC model selection [@ChenChen2008], stability filtering
[@MeinshausenBuhlmann2010], and final table/figure generation with
heteroscedasticity-robust OLS [@MacKinnonWhite1985]. Each stage writes
deterministic artifact files so intermediate results can be inspected and the
pipeline can be resumed from any checkpoint. The package supports both
single-machine execution and distributed SLURM array job workflows for
high-performance computing (HPC) environments.

# Statement of need

Reduced-form modeling of complex simulation outputs is a recurring task in
energy systems analysis [@PetersonEtAl2013; @LinEtAl2013BSMDocumentation],
climate and environmental modeling [@RattoEtAl2012; @PianosiEtAl2016], and
other fields where running the high-fidelity simulator for every analysis
scenario is impractical. Existing tools either focus on a single step (e.g.,
LASSO feature selection [@Tibshirani1996]) or require significant custom glue
code to chain steps and manage intermediate artifacts.

`rfm-pipeline` provides a tested, reproducible end-to-end workflow that:

1. handles the full pipeline from raw input/output matrices to publication-ready
   tables and figures;
2. provides a sensitivity study framework for evaluating how pipeline
   hyperparameters affect predictive performance across many synthetic
   data-generating processes (DGPs);
3. supports HPC-scale execution via SLURM array jobs with built-in
   checkpointing and artifact collection;
4. ships with a Random Forest meta-regression model
   [@Breiman2001; @PedregosaEtAl2011] for predicting pipeline runtime and
   holdout NRMSE given dataset and configuration characteristics.

The pipeline builds on established machine learning tools implemented in
scikit-learn [@PedregosaEtAl2011], composing them into a reproducible staged
workflow rather than requiring users to assemble and validate each step
independently.

# Implementation

The pipeline is implemented as a set of composable Python functions in the
`rfm_pipeline` package. Each stage accepts a configuration object and artifact
directory, reads its inputs, writes its outputs, and returns a provenance
record.

The sensitivity study framework generates Latin hypercube samples
[@McKayEtAl1979] over the pipeline hyperparameter space, runs each
configuration on both pure-synthetic and BSM-structure DGPs, and fits a Random
Forest meta-regression [@Breiman2001; @PedregosaEtAl2011] (via
`scripts/fit_meta_regression.py`) to predict normalized root-mean-square error
(NRMSE [@HyndmanKoehler2006]) as a function of configuration and dataset
characteristics. Cross-validation uses group blocking on DGP–configuration
pairs to prevent replicate leakage [@KohaviBecker1995].

The package uses [Pixi](https://pixi.sh) for reproducible environment
management and ships with a `test_repo.sh` script that enforces code
formatting, linting, notebook hygiene, unit tests, workflow smoke tests, and
documentation builds as a single gate.

# Acknowledgements

<!-- TODO: add acknowledgements and funding statement -->

# References
