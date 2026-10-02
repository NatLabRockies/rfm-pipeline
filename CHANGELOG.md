# Changelog

All notable changes to this repository should be documented in this file.

## 0.1.0

Initial public package candidate for the rfm-pipeline reduced-form modeling workflow.

### Added

- tested canonical workflow orchestration from screening through final OLS fitting,
  holdout evaluation, artifact assembly, and bundle writing
- explicit feature-expansion utilities
- canonical manifest-aware export, reload, and bundle-prediction helpers
- one deterministic end-to-end example under `examples/`
- MIT license and public release metadata
- installed-wheel smoke validation for the public API

### Changed

- removed study-specific artifacts, figures, execution wrappers, and internal
  development scaffolding from the public package repository
- made workflow alignment checks fail closed on label or order mismatches
