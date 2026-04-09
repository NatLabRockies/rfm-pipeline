# Proposed module breakdown (initial)

## Stage-aware package layout

- `bsm_rfm.data`
  - row alignment
  - scenario flag recovery
  - holdout splitting
  - train-only standardization
- `bsm_rfm.screening_null`
  - future home for recovered delta/permutation-null logic from `null_distribution.py`
- `bsm_rfm.feature_engineering`
  - future home for interaction and nonlinear feature generation from `make_nonlinear_features.ipynb`
- `bsm_rfm.regularized_screen`
  - future home for notebook-derived PCA/debiased-LASSO workflow
- `bsm_rfm.final_ols`
  - recovered notebook-derived final OLS handoff and export contract
  - canonical post-fit artifact schema helpers
- `bsm_rfm.metrics`
  - macro nRMSE, null baseline, bootstrap CIs
- `bsm_rfm.features`
  - selected-feature parser and module summaries
- `bsm_rfm.artifacts`
  - manifest schema and metadata tables
- `bsm_rfm.viz_io`
  - read-only loader for canonical exported artifacts

## Immediate tested slices in this bundle

This initial implementation provides the reusable, tested foundations needed before the heavier modeling code is ported:

- scenario parsing and alignment
- stratified holdout splitting
- train-only scaling
- canonical macro nRMSE and bootstrap CI helpers
- null mean-prediction baseline helper
- selected-feature parser validated against real naming examples
- artifact manifest schema with order-preserving position maps
- visualization-side canonical artifact loader

## Engineering scaffold adopted for the refactor

- `pixi.toml`
  - canonical environment manifest for local development and CI
  - task aliases for fix, check, unit tests, workflow smoke tests, and docs builds
- `test_repo.sh`
  - local gate with `--fix`, `--clean`, and `--ci` modes
- `.github/workflows/ci.yml`
  - GitHub Actions workflow that mirrors the local gate through Pixi
- `docs/`
  - Sphinx + MyST documentation tree built from package code and repo design docs
- `tools/check_repo.py`
  - repository hygiene checks, including generated artifact detection
- `tools/notebook_hygiene.py`
  - notebook output stripping and notebook-cleanliness checks

## Module update

- `bsm_rfm.data`: add canonical balanced subset generation from 300k to 20k by sampling 5,000 rows within each AFSC/UAEORO boolean combination.
- `bsm_rfm.null_screening`: adapter layer that delegates the upstream null-screening stage to the recovered `null_distribution.py` implementation instead of reimplementing notebook-drifted logic.
