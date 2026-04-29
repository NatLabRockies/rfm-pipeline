# MEMORY — bsm-public-rf handoff for next chat

You are continuing work on the `bsm-public-rf` repo.

## Core rule

Treat the user's live local repo as the only source of truth.

Do **not** trust prior bundles, summaries, or assumptions over the actual repo on disk.
Start every new chat with a strict live-repo audit.
Make only cumulative fixes.

## Current objective

Continue hardening and finalizing `bsm-public-rf` as a **complete manuscript reproduction package** for the JDS BSM manuscript, not just a public workflow package.

The repo must ultimately:

- implement the manuscript workflow stage for stage,
- run deterministically on a toy/demo dataset for CI and public validation,
- execute real-data notebooks from start to finish once local path placeholders are filled in,
- regenerate manuscript-facing tables and figures,
- support a new user reproducing the documented workflow with clear contracts and documentation.

## Current confirmed project state from this chat

### Phase status

- **Phase 0 complete**: manuscript contract freeze layer exists.
- **Phase 1 complete**: manuscript data/runtime/notebook manifests and placeholder-path contract exist.
- **Phase 2 complete**: manuscript notebook entrypoints, runtime/path-resolution utilities, demo-data fallback, and notebook execution support were added.
- **Phase 3 source-backed workflow implemented**: output conditioning, empirical-null screening, interaction discovery, nonlinear discovery, sparse selection/stability filtering, final manuscript table/figure regeneration, the end-to-end reproduction chain, and the QA audit/smoke gate now have source-backed functions, notebook entrypoints, deterministic demo execution, CSV/SVG handoff artifacts, and regression tests.
- **Interaction-discovery provenance added after empirical-null provenance**: the public interaction stage now records `public_implementation_method=residualized_product_permutation_surrogate`, `public_implementation_status=source_backed_public_surrogate`, `source_workflow_reference=private_tree_shap_interaction_workflow`, and `source_workflow_equivalence_status=not_yet_validated`. This reconciles the provenance gap but does not implement manuscript-exact tree-SHAP interactions.

### Important nuance about validation state

Do **not** assume the repo is fully clean right now without re-running the local gate.

During this chat, multiple late-cycle fixes were made after user-reported local failures, including:

- restoring the reproducibility example public contract (`DATASET_TAG`, `run_reproducibility_example`, CLI output wording),
- fixing bundle-contract helper return types and export-bundle docs references,
- fixing manuscript notebook runtime path normalization,
- fixing notebook key access for nested manuscript config structure,
- fixing Ruff notebook E402 issues by moving imports out of bootstrap cells,
- fixing Ruff import-order issues in `tests/test_public_api.py` and `tests/test_release_scope.py`.

The **latest known local reported failures** at the end of the chat were just Ruff import-order issues in:

- `tests/test_public_api.py`
- `tests/test_release_scope.py`

A final small patch bundle was produced for those import-order fixes, but there was **no final user-confirmed full-gate rerun after that last patch** in this chat.

So the next chat must begin by auditing the live local repo and rerunning the full local validation chain.

## Confirmed manuscript-reproduction contract layers now present

### Phase 0

Files added/maintained around the manuscript contract freeze include:

- `docs/manuscript_contract.md`
- `configs/manuscript_case_study.yml`
- `tests/test_manuscript_contract.py`

The case-study contract now freezes manuscript-explicit values and repo decisions for ambiguous items.

### Phase 1

Files added/maintained around the manuscript data/runtime contract include:

- `docs/manuscript_data_contract.md`
- `configs/manuscript_data_contract.yml`
- `configs/manuscript_paths.template.yml`
- `configs/manuscript_runtime.yml`
- `notebooks/manuscript/README.md`
- `src/bsm_rfm/manuscript_data_contract.py`
- `tests/test_manuscript_data_contract.py`

This layer freezes:

- required real-data artifact tables,
- local placeholder-path policy,
- frozen manuscript notebook execution order.

### Phase 2

Files added/maintained around manuscript runtime and notebook entrypoints include:

- `docs/manuscript_runtime.md`
- `src/bsm_rfm/manuscript_runtime.py`
- `tests/test_manuscript_runtime.py`
- `notebooks/manuscript/00_case_study_data_intake.ipynb`
- `notebooks/manuscript/01_candidate_library_audit.ipynb`
- `notebooks/manuscript/02_output_conditioning.ipynb`
- `notebooks/manuscript/03_empirical_null_screen.ipynb`
- `notebooks/manuscript/04_interaction_discovery.ipynb`
- `notebooks/manuscript/05_nonlinear_discovery.ipynb`
- `notebooks/manuscript/06_sparse_selection_and_stability.ipynb`
- `notebooks/manuscript/07_final_ols_and_bundle_export.ipynb`
- `notebooks/manuscript/08_manuscript_tables_and_figures.ipynb`

The runtime layer now aims to:

- resolve manuscript execution in `real` or `demo` mode,
- use deterministic demo fallback when private case-study files are absent,
- normalize nested notebook paths back to repo root,
- validate artifact-table presence/columns before notebook use,
- give notebooks a shared context object.

## Confirmed public/release-facing layers already added earlier in this project thread

These may already be present in the live repo, but must be audited rather than assumed:

- `LICENSE` (MIT)
- `CHANGELOG.md`
- `CITATION.cff`
- `docs/quickstart.md`
- `docs/export_bundle.md`
- `docs/reproducibility_example.md`
- `docs/scope_boundary.md`
- example script `examples/end_to_end_reproducibility.py`
- bundle contract helpers and tests
- release metadata in `pyproject.toml`

## Current likely live-repo fault line

Because of repeated incremental fixes and merge/conflict resolution during this chat, the most likely remaining issues are **repo-state drift** rather than deep scientific bugs.

Specifically audit for:

- merge-marker residue,
- docs/index drift versus docs tests,
- example-script drift versus `tests/test_reproducibility_example.py`,
- notebook bootstrap/import drift versus Ruff E402 and notebook execution,
- manuscript runtime drift versus the actual YAML config shape,
- `__init__` export drift versus `tests/test_public_api.py`,
- release metadata drift versus `tests/test_release_scope.py`.

## Immediate next step for the new chat

Start with a strict live-repo audit and actual local validation from the user's repo.

### Run first

1. `git status --short`
1. read `docs/MEMORY.md`
1. inspect:
   - `README.md`
   - `pyproject.toml`
   - `pixi.toml`
   - `test_repo.sh`
   - `docs/index.md`
   - `docs/api.rst`
   - `docs/manuscript_contract.md`
   - `docs/manuscript_data_contract.md`
   - `docs/manuscript_runtime.md`
   - `examples/end_to_end_reproducibility.py`
   - `src/bsm_rfm/__init__.py`
   - `src/bsm_rfm/manuscript_runtime.py`
   - `tests/test_public_api.py`
   - `tests/test_release_scope.py`
   - `tests/test_reproducibility_example.py`
   - `tests/test_manuscript_runtime.py`
1. run the real local validation chain:
   - `pixi run lint`
   - `pixi run test`
   - `pixi run notebook-tests`
   - `./test_repo.sh --fix`
   - `./test_repo.sh --check`
   - `pixi run docs`
   - `pixi run package-build`

Do not claim anything passes unless it was actually re-run on the user's live repo.

## Next engineering priority after the live audit

If the gate is clean after the live audit, the next highest-priority step is final release hardening:

1. remove stale Phase 2/Phase 3 wording from public docs,
1. ensure the audited reproduction smoke task remains aligned with emitted QA checks,
1. verify notebook, docs, package-build, and CI gates from the user's live Pixi environment,
1. only then move to real-data dry-run issues that require local private artifact paths.

The Phase 3 scientific stages are now implemented in source-backed demo/real-data entrypoints; future work should treat them as public contracts unless tests and documentation are updated together.

## Rules for future work

- Make only cumulative fixes.
- Fix root causes only.
- No hacks, no shims, no compatibility layers unless explicitly requested.
- Keep local and CI behavior aligned.
- Prefer behavior-oriented tests over brittle implementation-literal tests unless the literal value is itself a frozen contract.
- For notebook work, do not send notebook changes unless they are Ruff-clean and notebook-execution-safe.
- For example scripts, preserve the public contract expected by existing tests unless intentionally changing the contract and updating the tests together.

## Suggested first question for the next chat

Ask the repo to prove its current state first:

- Does the live local repo fully pass `pixi run lint`, `pixi run test`, `pixi run notebook-tests`, `./test_repo.sh --fix`, `./test_repo.sh --check`, `pixi run docs`, and `pixi run package-build` after all late-cycle fixes from this chat?

Only after that should Phase 3 implementation begin.

## Latest Phase 3 continuation note

The sparse-selection/stability slice adds `run_sparse_selection_stability_stage(...)` and related
source-backed helpers in `bsm_rfm.manuscript_stages`. The public demo implementation uses the
ordered union of empirical-null retained terms, retained interaction pairs, and retained nonlinear
transformations as the sparse candidate support, fits EBIC-selected L1 models per retained PCA
component, aggregates nonzero support across components, and writes deterministic stability and
final-support CSV artifacts under `sparse_selection/`. The next Phase 3 slice should implement
final manuscript table and figure regeneration (implemented source-backed slice).

## Latest Phase 3 continuation note

The final manuscript-artifact slice adds `run_final_manuscript_artifacts_stage(...)` and related
source-backed helpers in `bsm_rfm.manuscript_stages`. The public demo implementation recomputes
the upstream manuscript stages for the active context, fits final OLS on the stable sparse-selection
support, computes deterministic holdout macro nRMSE with bootstrap uncertainty against the frozen
`Y_train` normalization contract, and writes final-model CSVs, manuscript-facing summary tables,
figure source-data CSVs, and dependency-free SVG figures under `final_manuscript_artifacts/`.

## Latest Phase 3 continuation note

The finalization slice adds `run_manuscript_reproduction_stage_chain(...)` and the
`ManuscriptReproductionStageChainResult` public result contract. The chain executes the complete
Phase 3 manuscript workflow from one runtime context, writes every upstream handoff artifact family
plus the final table/figure artifacts, and has an integration test that verifies all Phase 3 artifact
families are emitted. This is now the preferred single-call public entry point for validating the
source-backed manuscript reproduction chain before running notebook-by-notebook workflows.

## Latest finalization continuation note

The reproducibility example now has a manuscript-chain smoke-test path. The existing
`run_reproducibility_example(...)` canonical workflow contract and CLI wording are preserved, and
`examples/end_to_end_reproducibility.py` also exposes
`run_manuscript_reproduction_example(...)`. The CLI flag `--run-manuscript-chain` writes the
complete deterministic demo manuscript artifact family tree to `--manuscript-output-dir`, including
output conditioning, empirical-null screening, interaction discovery, nonlinear discovery,
sparse-selection/stability, and final manuscript table/figure artifacts. This is intended as the
public example-level smoke-test companion to the notebook-by-notebook reproduction workflow.

## Latest finalization bug-fix note

The canonical reproducibility example originally used perfectly linear holdout responses, so the
final OLS demo could report a printed holdout macro nRMSE of `0.000000`. That was misleading for a
public evaluation smoke test. The example now adds a small fixed deterministic residual to the toy
holdout responses and tests that the holdout nRMSE point estimate is strictly positive and round
trips through the written `nrmse_summary` artifact. This preserves determinism while exercising the
nonzero holdout-error path.

## Latest finalization continuation note

The next finalization slice adds a manuscript reproduction QA audit layer. The new public API is
`audit_manuscript_reproduction_outputs(...)`, `write_manuscript_reproduction_audit(...)`, and
`run_manuscript_reproduction_audit_stage(...)`, with `ManuscriptReproductionAuditResult` and
`ManuscriptReproductionAuditStageResult` result contracts. The audit wraps the complete Phase 3
stage chain, writes `reproduction_audit/artifact_manifest.csv`, `metric_checks.csv`, and
`audit_summary.csv`, and explicitly checks artifact existence, nonempty files, positive demo
holdout nRMSE, ordered bootstrap intervals, finite null baseline nRMSE, nonempty final support,
registered SVG assets, and final-OLS workflow-summary coverage. The public reproducibility example
now uses this audited path when `--run-manuscript-chain` is supplied and prints the manuscript audit
status.

## Latest finalization continuation note

The next hardening slice adds a gate-level manuscript reproduction smoke check. The new script is
`tools/check_manuscript_reproduction.py`, and the new Pixi task is
`manuscript-reproduction-smoke = "python tools/check_manuscript_reproduction.py"`. The task runs
`run_manuscript_reproduction_audit_stage(...)` from a temporary output root, checks that the audit
status is `pass`, verifies all expected Phase 3 artifact families are emitted, verifies required QA
metric checks are present and passing, and explicitly checks that final holdout nRMSE remains
strictly positive. `test_repo.sh` now includes `manuscript-reproduction-smoke` in
`VALIDATION_TASKS` after `workflow-tests` and before `notebook-tests`, so local and CI gates exercise
the same audited manuscript-reproduction path without leaving generated artifacts in the work tree.

## Latest finalization bug-fix note

The manuscript reproduction smoke script now has a separate pure-Python
`validate_metric_check_records(...)` helper. This prevents stale or incomplete audit-contract
changes from producing a traceback when a required metric check is missing. The smoke script now
reports missing metric checks cleanly, validates required check statuses, and only inspects the
final holdout nRMSE value when the corresponding metric-check row is present. Regression tests cover
missing, nonpositive, and nonnumeric final holdout nRMSE metric-check records.

## Latest finalization continuation note

The public manuscript documentation has been refreshed to describe the notebooks as source-backed
entrypoints rather than Phase 2 skeletons. Keep this wording current: Phase 3 now has implemented
stage functions, notebooks, final table/figure regeneration, an end-to-end reproduction chain, a QA
audit layer, and a gate-level manuscript reproduction smoke task.

## Latest alignment-audit correction

The repo should not yet claim that all manuscript notebooks are complete in the sense of exact
scientific reproduction. They are executable source-backed entrypoints with deterministic demo
artifacts and QA checks, but `docs/manuscript_alignment_audit.md` now records remaining exactness
gaps: interaction discovery does not yet implement tree-SHAP interactions, nonlinear discovery does
not yet implement GAM EDF/p-value diagnostics, sparse selection has not yet been shown equivalent to
the recovered de-biased-LASSO workflow, the final HC3 inferential filter now needs real-data
verification, and final tables and figures are deterministic public artifacts rather than verified
manuscript-exact reproductions. Interaction and nonlinear discovery now both write explicit
provenance tables marking the public methods as surrogates with unvalidated private-workflow
equivalence.
Treat that audit as the current source of truth for manuscript-alignment claims.

### Empirical-null provenance reconciliation slice

The empirical-null screening stage now writes `empirical_null_provenance.csv` next to the screening
statistics, coefficients, permutation-null summaries, retained terms, and summary table. The
provenance ledger records the public implementation method, public implementation status, private
source-script reference, and source-script equivalence status. The current status is intentionally
conservative: `source_backed_public_surrogate` with `not_yet_validated` private-script equivalence.
Do not describe this stage as exact reproduction of the recovered Delta/null-screening script until
that script is available in the repo and validated against the public artifact outputs.

### Nonlinear provenance reconciliation slice

The nonlinear-discovery stage now writes `nonlinear_discovery_provenance.csv` next to the
transformation scores, component scores, retained transformations, and summary table. The
provenance ledger records the frozen GAM/EDF manuscript method, the public residualized parametric
transform surrogate, the private GAM workflow reference, and the equivalence-validation status. The
current status is intentionally conservative: `source_backed_public_surrogate` with
`not_yet_validated` private-workflow equivalence. Do not describe this stage as exact reproduction
of the manuscript GAM EDF/p-value workflow until that workflow is ported or externalized and
validated against the public outputs.

## Sparse-selection provenance reconciliation

The sparse-selection/stability stage now writes `sparse_selection/sparse_selection_provenance.csv` and carries public implementation/equivalence fields through the stage summary. The public implementation remains `ebic_l1_component_union_with_subsample_stability` and is explicitly marked `source_backed_public_surrogate` with `source_workflow_reference=notebook_pca_debiased_lasso`, `source_artifact=LASSO_to_OLS_v9.ipynb`, and `source_workflow_equivalence_status=not_yet_validated`. This is a provenance/equivalence-status hardening slice, not a port of the recovered de-biased-LASSO notebook workflow.
