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
heteroscedasticity-robust (HC3) Wald inference [@MacKinnonWhite1985]
computed directly in NumPy. Each stage writes
deterministic artifact files so intermediate results can be inspected and the
pipeline can be resumed from any checkpoint. The package supports both
single-machine execution and distributed SLURM array job workflows for
high-performance computing (HPC) environments.

# Statement of need

Reduced-form modeling of complex simulation outputs is a recurring task in
energy systems analysis [@PetersonEtAl2013; @LinEtAl2013BSMDocumentation],
climate and environmental modeling [@RattoEtAl2012; @PianosiEtAl2016], and
other fields where running the high-fidelity simulator for every analysis
scenario is impractical.

Existing tools in this space address adjacent but distinct problems. Global
sensitivity analysis packages such as `SALib` [@HermanUsher2017SALib] focus
on variance decomposition of an already-defined surrogate. Generic surrogate
modeling frameworks such as `chaospy` [@FeinbergRingstad2015Chaospy] target
polynomial-chaos and Gaussian-process expansions over low-dimensional input
spaces and do not provide adaptive sparse feature discovery. Automated
feature engineering tools such as `Featuretools` [@KanterVeeramachaneni2015]
generate large candidate sets but do not select, stabilize, or fit
surrogates. Sparse-regression libraries such as the scikit-learn LASSO
[@Tibshirani1996; @PedregosaEtAl2011] implement individual stages but
require significant custom glue code to chain interaction discovery,
nonlinear feature discovery, stability selection, post-selection inference,
ablation comparison, and bootstrap-based uncertainty quantification into a
reproducible end-to-end workflow.

`rfm-pipeline` fills this gap by providing a tested, reproducible end-to-end
workflow that:

1. handles the full pipeline from raw input/output matrices to
   publication-ready tables and figures;
2. supports interaction and nonlinear feature discovery (tree-SHAP and
   GAM smoothing splines) coupled with EBIC-selected L1 stability
   selection and HC3 post-selection inference;
3. provides a sensitivity study framework for evaluating how pipeline
   hyperparameters affect predictive performance across many synthetic
   data-generating processes (DGPs);
4. supports HPC-scale execution via SLURM array jobs with built-in
   checkpointing and artifact collection;
5. ships a pre-fitted Random Forest meta-regression model
   [@Breiman2001; @PedregosaEtAl2011] (in `artifacts/sensitivity/`) for
   predicting pipeline runtime and holdout NRMSE given dataset and
   configuration characteristics.

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
configuration on both pure-synthetic and BSM-structure DGPs, and supports
two complementary meta-regression analyses of the recorded results:
(i) a polynomial response-surface fit
(`scripts/fit_meta_regression.py`) using OLS (degree two) or
LASSO-then-OLS [@Tibshirani1996; @PedregosaEtAl2011] (degree three) for
interpretable coefficients, and (ii) pre-fitted Random Forest models
[@Breiman2001] shipped under `artifacts/sensitivity/` that predict
normalized root-mean-square error
(NRMSE [@HyndmanKoehler2006]) and pipeline wall time as a function of
configuration and dataset characteristics. Cross-validation of the
pre-fitted models uses group blocking on DGP–configuration pairs to
prevent replicate leakage [@KohaviBecker1995]. Two pipeline knobs that are
not part of the sensitivity LHS sweep
(`n_tree_estimators` and `max_tree_depth` for the gradient-boosted SHAP
interaction discovery stage) are held at a faster-than-manuscript baseline
in the synthetic-DGP study configuration
(`configs/sensitivity_study/base_synthetic.yml`) to keep the LHS budget
tractable; the BSM case-study run uses the full manuscript baselines.

The package uses [Pixi](https://pixi.sh) for reproducible environment
management and ships with a `test_repo.sh` script that enforces code
formatting, linting, notebook hygiene, unit tests, workflow smoke tests, and
documentation builds as a single gate.

# Acknowledgements

This work was developed at the National Renewable Energy Laboratory in
support of energy system reduced-form modeling work funded by the U.S.
Department of Energy. The author thanks colleagues at NLR/NREL for
discussions on the methodology and feedback on early drafts.

<!-- TODO: replace funding statement with the exact DOE BETO contract /
     award identifier; confirm final affiliation (NREL vs NLR) and ORCID
     before JOSS submission. -->

# References
