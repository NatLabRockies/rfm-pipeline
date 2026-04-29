# Manuscript runtime and source-backed notebooks

Phase 2 adds the execution layer that makes the manuscript notebooks runnable in both CI and local
real-data mode.

## Runtime resolution policy

Use `bsm_rfm.resolve_manuscript_runtime(repo_root)` to determine notebook inputs.

- When `configs/local/manuscript_paths.local.yml` exists and every required artifact file is
  present, the runtime resolves to `mode = "real"`.
- Otherwise the runtime falls back to `mode = "demo"` and writes a deterministic toy dataset to a
  temporary directory.

The deterministic demo path exists so the notebook gate can execute the full notebook stack without
shipping the real BSM case-study files.

## Data-intake helpers

The public helpers for the notebook layer are:

- `bsm_rfm.load_manuscript_runtime_manifest(...)`
- `bsm_rfm.load_manuscript_paths_template(...)`
- `bsm_rfm.load_manuscript_local_override(...)`
- `bsm_rfm.resolve_manuscript_runtime(...)`
- `bsm_rfm.load_manuscript_artifact_tables(...)`
- `bsm_rfm.validate_manuscript_artifact_tables(...)`
- `bsm_rfm.build_manuscript_notebook_context(...)`
- `bsm_rfm.manuscript_runtime_summary_table(...)`

## Notebook entrypoints

The tracked manuscript notebooks now exist at the frozen Phase 1 paths under
`notebooks/manuscript/`:

- `00_case_study_data_intake.ipynb`
- `01_candidate_library_audit.ipynb`
- `02_output_conditioning.ipynb`
- `03_empirical_null_screen.ipynb`
- `04_interaction_discovery.ipynb`
- `05_nonlinear_discovery.ipynb`
- `06_sparse_selection_and_stability.ipynb`
- `07_final_ols_and_bundle_export.ipynb`
- `08_manuscript_tables_and_figures.ipynb`

## Phase 3 output-conditioning stage

The first Phase 3 source-backed notebook stage is `02_output_conditioning.ipynb`. It calls
`bsm_rfm.run_output_conditioning_stage(...)`, which reads the frozen output-conditioning settings
from `configs/manuscript_case_study.yml`, applies train-only output filters, computes the PCA
reduced-response representation, and writes deterministic CSV handoff artifacts under
`{output_root}/output_conditioning/`.

The stage currently writes:

- `output_filter_diagnostics.csv`
- `pca_scores.csv`
- `pca_loadings.csv`
- `pca_explained_variance.csv`
- `output_conditioning_summary.csv`

## Phase 3 empirical-null screening stage

The second Phase 3 source-backed notebook stage is `03_empirical_null_screen.ipynb`. It calls
`bsm_rfm.run_empirical_null_screening_stage(...)`, which materializes the tracked feature
catalog, uses the retained PCA component scores as the screening response, computes the
frozen coefficient-row-norm statistic, estimates featurewise empirical-null p-values from
deterministic response permutations, and applies the frozen Benjamini--Hochberg threshold.

The stage currently writes:

- `feature_screening_statistics.csv`
- `component_coefficients.csv`
- `permutation_null_summary.csv`
- `retained_terms.csv`
- `empirical_null_screen_summary.csv`

## Phase 3 interaction-discovery stage

The third Phase 3 source-backed notebook stage is `04_interaction_discovery.ipynb`. It calls
`bsm_rfm.run_interaction_discovery_stage(...)`, which reads interaction-discovery metadata from
the frozen case-study contract, uses the released feature catalog as the authoritative candidate
pair source, scores residualized two-factor product terms against retained PCA component scores,
and estimates deterministic response-permutation null thresholds for CI/demo execution.

The stage currently writes:

- `interaction_pair_scores.csv`
- `component_interaction_scores.csv`
- `interaction_null_summary.csv`
- `retained_interaction_pairs.csv`
- `interaction_discovery_summary.csv`

## Phase 3 nonlinear-discovery stage

The fourth Phase 3 source-backed notebook stage is `05_nonlinear_discovery.ipynb`. It calls
`bsm_rfm.run_nonlinear_discovery_stage(...)`, which reads nonlinear-discovery metadata from the
frozen case-study contract, uses the released feature catalog as the authoritative transformation
candidate source, residualizes each supported transformation against its source first-order input,
and scores the incremental nonlinear contribution against retained PCA component scores.

The stage currently writes:

- `transformation_scores.csv`
- `component_transformation_scores.csv`
- `retained_transformations.csv`
- `nonlinear_discovery_summary.csv`

## Phase 3 sparse-selection and stability stage

The fifth Phase 3 source-backed notebook stage is
`06_sparse_selection_and_stability.ipynb`. It calls
`bsm_rfm.run_sparse_selection_stability_stage(...)`, which reads sparse-selection and stability
metadata from the frozen case-study contract, builds the ordered union of retained empirical-null
terms plus retained interaction and nonlinear candidates, fits EBIC-selected L1 models per retained
PCA component, aggregates nonzero support across components, and evaluates deterministic
subsample-stability diagnostics.

The stage currently writes:

- `support_candidates.csv`
- `component_model_selection.csv`
- `component_coefficients.csv`
- `stability_resample_summary.csv`
- `stability_feature_summary.csv`
- `final_stable_support.csv`
- `sparse_selection_summary.csv`

## Phase 3 final manuscript tables and figures stage

The final Phase 3 source-backed notebook stage is
`08_manuscript_tables_and_figures.ipynb`. It calls
`bsm_rfm.run_final_manuscript_artifacts_stage(...)`, which recomputes the upstream stage outputs
for the active runtime context, fits final OLS on the stable sparse-selection support, evaluates
holdout macro nRMSE against the frozen `Y_train` normalization contract, and writes final model
tables, manuscript-facing summary tables, figure source data, and dependency-free SVG assets.

The stage currently writes under `final_manuscript_artifacts/`:

- `final_model/final_support_features.csv`
- `final_model/final_ols_summary.csv`
- `final_model/coefficient_matrix_raw_scale.csv`
- `final_model/coefficient_matrix_standardized.csv`
- `final_model/x_standardization.csv`
- `final_model/y_standardization.csv`
- `tables/model_performance.csv`
- `tables/workflow_stage_summary.csv`
- `figures/figure_model_performance_data.csv`
- `figures/figure_support_composition_data.csv`
- `figures/figure_specs.csv`
- `figures/figure_model_performance.svg`
- `figures/figure_support_composition.svg`
- `final_artifact_summary.csv`

`07_final_ols_and_bundle_export.ipynb` uses the same stage to display and validate final-model
artifacts before the manuscript-facing table and figure notebook consumes them.

## Scientific alignment status

The tracked notebooks are executable and source-backed, but executable does not mean
manuscript-exact for every stage. The current scientific alignment ledger is
`docs/manuscript_alignment_audit.md`. In particular, the interaction-discovery stage is a
deterministic residualized-product score rather than the frozen tree-SHAP interaction method, the
nonlinear-discovery stage is a residualized parametric-transform score rather than GAM EDF/p-value
diagnostics, sparse selection has not yet been proven equivalent to the recovered de-biased-LASSO
workflow, and the final HC3 inferential filter is not yet implemented.

## End-to-end Phase 3 reproduction chain

For local reproduction runs that should emit every Phase 3 artifact family from one context, use
`bsm_rfm.run_manuscript_reproduction_stage_chain(...)`. The chain executes output conditioning,
empirical-null screening, interaction discovery, nonlinear discovery, sparse selection/stability,
and final table/figure regeneration in dependency order and writes all stage handoff artifacts
under the resolved manuscript output root.

## Manuscript reproduction audit

For final local QA, use `bsm_rfm.run_manuscript_reproduction_audit_stage(...)`. This wraps the
end-to-end Phase 3 chain, then writes `reproduction_audit/` with:

- `artifact_manifest.csv`: one row per written manuscript artifact with portable path, suffix,
  size, existence/nonempty flags, and SHA-256 digest;
- `metric_checks.csv`: explicit pass/fail checks for artifact presence, nonempty files, positive
  demo holdout nRMSE, ordered bootstrap interval, finite null baseline, nonempty final support,
  registered SVG outputs, and final-OLS workflow-summary coverage;
- `audit_summary.csv`: one row with aggregate artifact counts, failed metric-check count, and
  overall `qa_status`.

This audit is intended to catch misleading smoke-test outputs such as a zero holdout nRMSE before
manuscript tables or public examples are interpreted. The repository gate runs the same audited
path through `pixi run manuscript-reproduction-smoke`, which uses a temporary output root so the
check validates the manuscript chain without leaving generated artifacts in the work tree.
