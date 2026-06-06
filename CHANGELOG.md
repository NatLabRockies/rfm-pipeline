# Changelog

All notable changes to this repository should be documented in this file.

## 0.1.0

Initial public package candidate for the rfm-pipeline reduced-form modeling workflow.

### Added

- tested canonical workflow orchestration from screening through final OLS fitting,
  holdout evaluation, artifact assembly, and bundle writing
- explicit feature-expansion utilities and workflow scope-boundary documentation
- canonical manifest-aware export and reload helpers for downstream visualization
- deterministic end-to-end reproducibility example under `examples/`
- MIT license and public release metadata

### Notes

- the upstream Delta permutation-null screen remains exposed through the
  recovered source adapter in `rfm_pipeline.null_screening`
- the notebook-specific feature-expansion defaults remain only
  partially promoted into the canonical package path
