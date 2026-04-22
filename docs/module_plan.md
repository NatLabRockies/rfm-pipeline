# Module map and current implementation status

This page records the live package modules and what they currently cover.

## Current package modules

- `bsm_rfm.data`
  - row alignment
  - scenario-flag helpers
  - balanced boolean-stratum subset helpers
  - holdout splitting
  - train-only standardization
- `bsm_rfm.null_screening`
  - stable adapter around the recovered `null_distribution.py` source workflow
- `bsm_rfm.feature_expansion`
  - explicit feature-expansion specification and materialization helpers
- `bsm_rfm.regularized_screening`
  - executable multitask elastic-net screening foundation
- `bsm_rfm.final_ols`
  - final OLS fitting, prediction, evaluation summary, and canonical post-fit artifact assembly
- `bsm_rfm.metrics`
  - macro nRMSE and bootstrap confidence-interval helpers
- `bsm_rfm.features`
  - selected-feature parser and transformation-name utilities
- `bsm_rfm.artifacts`
  - manifest schema and metadata-table helpers
- `bsm_rfm.viz_io`
  - canonical bundle reload helpers for downstream visualization
- `bsm_rfm.workflow`
  - end-to-end workflow orchestration, case-study provenance tables, and bundle writing

## Repository engineering contract

The live repository contract is built around:

- `pixi.toml` for the canonical local and CI environment
- `test_repo.sh` for the local and CI validation entrypoint
- `.github/workflows/ci.yml` for CI orchestration through Pixi
- `tools/check_repo.py` for repository hygiene checks
- `tools/notebook_hygiene.py` for notebook output stripping and syntax validation
- `tools/check_markdown.py` and `tools/format_markdown.py` for shared Markdown checks

## Remaining reconciliation work

The package now covers the canonical screening/final-fit/evaluation/export path directly.
The main remaining scientific reconciliation work is upstream of that path:

- fully promoting the recovered notebook-specific feature-expansion defaults into a
  canonical source-driven default specification
- continuing to document provenance boundaries between source-derived and notebook-derived
  stages
- deciding whether additional case-study-specific reporting helpers belong in the public
  package surface or only in manuscript/reporting artifacts

## Current scope-freeze helper

Use `bsm_rfm.workflow_scope_boundary_table()` for the current machine-readable view of
which canonical workflow stages are still source-derived or only partially ported.
