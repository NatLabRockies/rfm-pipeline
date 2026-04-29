# BSM reduced-form modeling workflow package

This repository provides the tested package modules, documentation, and repository gate for the BSM reduced-form modeling workflow.

## Environment management

The repository is designed around Pixi. The canonical local and CI entrypoint is:

```bash
./test_repo.sh
```

To repair formatting and notebook hygiene before committing:

```bash
./test_repo.sh --fix
```

## What the repo gate enforces

- Ruff linting and formatting for Python and notebook code
- mdformat formatting checks for Markdown
- stripped notebook outputs and execution counts
- repository hygiene checks for generated Python artifacts and text whitespace issues
- unit tests
- workflow smoke tests
- audited manuscript-reproduction smoke checks in a temporary output root
- Sphinx documentation builds

## Documentation

The package documentation is built with Sphinx and MyST. Public Python APIs are expected to use NumPy-style docstrings.

## End-to-end reproducibility example

A deterministic toy example is included at `examples/end_to_end_reproducibility.py`. It runs the canonical workflow, writes a post-fit bundle, and reloads the written artifacts from disk. The toy holdout responses include a fixed deterministic residual so the printed holdout nRMSE is nonzero. It can also run the complete demo manuscript-reproduction stage chain and write every Phase 3 artifact family plus a reproduction audit with artifact hashes and metric QA checks.

Run the canonical workflow example from the repo source tree with:

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py --output-dir artifacts/toy-reproducibility-example
```

Run both the canonical workflow example and the demo manuscript-reproduction chain with:

```bash
PYTHONPATH=src python examples/end_to_end_reproducibility.py \
  --output-dir artifacts/toy-reproducibility-example \
  --run-manuscript-chain \
  --manuscript-output-dir artifacts/toy-manuscript-reproduction-example
```

This command also writes `reproduction_audit/`, which contains a portable artifact manifest,
SHA-256 hashes, metric checks, and an audit summary for the generated manuscript outputs.
The same audited path is exercised by the `manuscript-reproduction-smoke` Pixi task and
by `test_repo.sh` so local and CI gates catch missing artifacts or misleading metrics.

## Current scope

This package now covers the foundational and workflow-level utilities needed for a canonical reduced-form modeling path:

- data alignment, subset, split, and train-only standardization utilities
- executable archived-style multitask elastic-net screening
- executable final OLS fitting, prediction, and post-fit artifact assembly
- end-to-end workflow orchestration from screening through holdout evaluation and export-bundle assembly
- canonical post-fit bundle writing with CSV fallback when a parquet engine is unavailable
- explicit feature-expansion specification and materialization utilities
- selected-feature structure parsing
- nRMSE helpers and bootstrap summaries
- artifact manifest helpers
- visualization-side artifact loading

The scientific workflow is no longer only documented at the screening/final-fit/export boundary.
The repo now includes a tested workflow foundation and executable manuscript-stage scaffold that performs:

- upstream null-screening delegation to the recovered source script
- source-backed manuscript output-conditioning, empirical-null, interaction-discovery, nonlinear-discovery, sparse-selection/stability, and final table/figure stages
- executable archived-style multitask elastic-net screening
- executable final OLS fitting on retained features
- holdout bootstrap nRMSE evaluation
- canonical post-fit artifact assembly and on-disk export writing

This should not yet be described as a full exact implementation of every manuscript method.
`docs/manuscript_alignment_audit.md` is the current status ledger for which notebook stages are
manuscript-exact, partially aligned, or deterministic public approximations. Higher-level
reconciliation still remains for the notebook-derived feature-expansion stage, real-data
verification of the final HC3 filter, and exact manuscript table and figure reproduction. The
interaction and nonlinear stages now write explicit provenance ledgers, but their public surrogate
methods are not yet manuscript-exact.

The release-facing package metadata now records repository URLs, classifiers, and keywords, while the scientific scope boundary remains explicit: upstream Delta null screening is still exposed through a source-derived adapter, and the notebook-specific feature-expansion defaults are still only partially promoted into the canonical package path.

## Provenance notes

- The canonical 20k modeling subset is a balanced stratified sample: 5,000 rows drawn within each AFSC/UAEORO boolean combination.

- The canonical upstream null-screening stage comes from the recovered `null_distribution.py` source script and is wrapped through `bsm_rfm.null_screening` rather than reimplemented ad hoc in notebooks.

- The manuscript empirical-null notebook now uses a deterministic source-backed screening
  implementation that materializes the tracked feature catalog, computes coefficient-row-norm
  statistics against retained PCA component scores, estimates featurewise permutation-null p-values,
  and applies the frozen Benjamini--Hochberg threshold.

- The manuscript interaction-discovery notebook now scores released catalog interaction pairs by
  residualized incremental contribution beyond their first-order factors and writes deterministic
  pair-score, component-score, null-summary, retained-pair, and provenance artifacts. The
  provenance marks this public implementation as a non-equivalent surrogate until the tree-SHAP
  workflow is ported or externalized.

- The manuscript nonlinear-discovery notebook now scores released catalog transformation terms by
  residualized incremental nonlinear contribution beyond each source first-order input and writes
  deterministic transformation-score, component-score, retained-transformation, and provenance
  artifacts. The provenance marks this public implementation as a non-equivalent surrogate until
  the GAM EDF/p-value workflow is ported or externalized.

- The manuscript sparse-selection/stability notebook now fits EBIC-selected L1 models per
  retained PCA component, aggregates support across components, and writes deterministic
  subsample-stability diagnostics, final stable-support artifacts, and provenance that marks
  the public implementation as not-yet-validated against the recovered de-biased-LASSO
  notebook workflow.

- The final manuscript artifact stage now applies the frozen 95% HC3 Wald inferential filter
  to the stable support, refits final OLS on retained terms, computes deterministic holdout
  nRMSE with bootstrap uncertainty for demo execution, and writes manuscript-facing CSV tables
  plus dependency-free SVG figure assets.

## Repository gate

Use `./test_repo.sh` as the canonical local gate.

- `./test_repo.sh` runs the default prepare-and-validate path. It formats Python and Markdown,
  strips notebook outputs, then runs the full validation chain.
- `./test_repo.sh --fix` is an explicit alias for the default prepare-and-validate path.
- `./test_repo.sh --check` runs the same validation chain without mutating the working tree.
- `./test_repo.sh --ci` is the non-mutating GitHub Actions entrypoint.
- `./test_repo.sh --clean` rebuilds the Pixi environment, then runs the default path.

The validation chain includes repository hygiene checks, Ruff lint/format validation,
Markdown formatting validation, notebook hygiene validation, Python compilation, unit tests,
workflow smoke tests, notebook execution tests, Sphinx documentation builds, package builds,
and `git diff --check`.

## Citation and release notes

- Cite the package using the metadata in `CITATION.cff`.
- Track package-facing changes in `CHANGELOG.md`.

## License

This project is released under the MIT License. See `LICENSE`.

## Manuscript reproduction contract

The repo now freezes the manuscript reconstruction contract in `docs/manuscript_contract.md` and
`docs/manuscript_data_contract.md`. Real-data local filenames should be supplied through
`configs/local/manuscript_paths.local.yml`, using `configs/manuscript_paths.template.yml` as the
tracked template.

Phase 2 added tracked manuscript notebook entrypoints under `notebooks/manuscript/` and
runtime/path-resolution helpers in `bsm_rfm.manuscript_runtime` so the notebooks execute on
deterministic demo data in CI and switch to real data once the local path override file is
populated.

Phase 3 has source-backed stages and an end-to-end reproduction chain in `bsm_rfm.manuscript_stages` for output conditioning,
empirical-null screening, interaction discovery, nonlinear discovery, sparse selection with
stability filtering, and final manuscript table/figure regeneration. The
`02_output_conditioning.ipynb` through `08_manuscript_tables_and_figures.ipynb` notebooks now run
deterministic stage functions and write handoff artifacts under the resolved manuscript output root.
The chain is executable and QA-audited, but the manuscript-alignment audit records remaining
scientific exactness gaps before the repo should claim complete manuscript reproduction.
`run_manuscript_reproduction_audit_stage(...)` wraps the full chain with a QA layer that writes
`reproduction_audit/artifact_manifest.csv`, `metric_checks.csv`, and `audit_summary.csv`.
