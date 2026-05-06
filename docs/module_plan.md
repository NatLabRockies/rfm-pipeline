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
  - per-output nRMSE frame and ablation helpers used by final manuscript artifacts
- `bsm_rfm.features`
  - selected-feature parser and transformation-name utilities
- `bsm_rfm.artifacts`
  - manifest schema and metadata-table helpers
- `bsm_rfm.viz_io`
  - canonical bundle reload helpers for downstream visualization
- `bsm_rfm.workflow`
  - end-to-end workflow orchestration, case-study provenance tables, and bundle writing
- `bsm_rfm.manuscript_data_contract`
  - frozen manuscript artifact-table, placeholder-path, and notebook-order contracts
- `bsm_rfm.manuscript_runtime`
  - real/demo manuscript runtime resolution, artifact loading, notebook contexts, and deterministic
    demo artifact generation
- `bsm_rfm.manuscript_stages`
  - source-backed manuscript-stage functions for output conditioning, empirical-null screening,
    interaction discovery, nonlinear discovery, sparse-selection/stability filtering, final
    manuscript artifact generation, the complete reproduction chain, and the QA audit entrypoint
    `run_manuscript_reproduction_audit_stage(...)`

## Repository engineering contract

The live repository contract is built around:

- `pixi.toml` for the canonical local and CI environment
- `test_repo.sh` for the local and CI validation entrypoint
- `.github/workflows/ci.yml` for CI orchestration through Pixi
- `tools/check_repo.py` for repository hygiene checks
- `tools/notebook_hygiene.py` for notebook output stripping and syntax validation
- `tools/check_markdown.py` and `tools/format_markdown.py` for shared Markdown checks

## Remaining reconciliation work

The package now covers the canonical screening/final-fit/evaluation/export path directly and includes
a source-backed executable scaffold for the manuscript-reproduction notebooks. The authoritative
status ledger for manuscript-exactness claims is `docs/manuscript_alignment_audit.md`.

The main remaining scientific reconciliation work is:

- fully promoting the recovered notebook-specific feature-expansion defaults into a
  canonical source-driven default specification
- replacing, porting, or externalizing the tree-SHAP interaction workflow now represented by a
  residualized-product public surrogate
- replacing, porting, or externalizing the GAM EDF/p-value nonlinear workflow now represented by a
  residualized parametric-transform public surrogate
- validating or porting the de-biased-LASSO sparse-selection workflow currently represented by an
  EBIC/L1 public surrogate with stability diagnostics
- verifying the final HC3 filter, retained-feature counts, coefficients, tables, and figures against
  the private real-data manuscript run
- deciding whether additional case-study-specific reporting helpers belong in the public
  package surface or only in manuscript/reporting artifacts
