# Manuscript implementation alignment audit

This page is the current scientific-status ledger for the JDS BSM manuscript reproduction work.
It separates three different claims that are easy to conflate:

1. **Executable notebook status**: whether the tracked notebooks run deterministically against the
   resolved manuscript runtime context.
1. **Artifact-chain status**: whether the notebooks and source functions emit handoff artifacts with
   stable schemas for downstream QA.
1. **Manuscript-method exactness**: whether the implemented computation is the same method described
   in the frozen manuscript contract.

## Current answer

The notebooks are implemented as executable entrypoints, and the interaction-discovery and
nonlinear-discovery stages now implement the manuscript-exact methods (tree-based SHAP interaction
values via `GradientBoostingRegressor` + `shap.TreeExplainer`, and GAM cubic smoothing spline
curvature detection via `scipy.interpolate.UnivariateSpline`). The main remaining scientific gaps
are exact equivalence validation of empirical-null screening against the recovered Delta-null
script, the de-biased-LASSO versus plain L1-selection distinction, exact manuscript figure
regeneration, and real-data verification.

## Stage-by-stage alignment

| Stage                          | Notebook                                  | Current implementation                                                                                                                                                                                                                                                                                                     | Alignment status                                                                                                                                                                                                              | Highest-priority remaining work                                                                                                                                                 |
| ------------------------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data intake                    | `00_case_study_data_intake.ipynb`         | Resolves runtime context, validates demo or local artifact tables, and reports table shapes.                                                                                                                                                                                                                               | Executable runtime entrypoint; exactness depends on local private artifact paths.                                                                                                                                             | Add real-data dry-run evidence once private paths are configured.                                                                                                               |
| Candidate library audit        | `01_candidate_library_audit.ipynb`        | Loads the released feature catalog from the runtime artifact tables and summarizes feature types.                                                                                                                                                                                                                          | Partially aligned; it audits an external catalog but does not regenerate the 26,560-row library from primitive rules.                                                                                                         | Decide whether the repo must regenerate the catalog or treat the released catalog as the reproducibility artifact of record.                                                    |
| Output conditioning            | `02_output_conditioning.ipynb`            | Applies train-only variance and dynamic-range filters and computes retained PCA scores.                                                                                                                                                                                                                                    | Mostly aligned with the frozen contract for the public implementation.                                                                                                                                                        | Verify real-data retained output/component counts against the manuscript values.                                                                                                |
| Empirical-null screening       | `03_empirical_null_screen.ipynb`          | Materializes catalog terms, computes coefficient-row-L2 statistics against retained PCA scores, estimates featurewise response-permutation p-values, applies BH q = 0.05, and writes an explicit provenance table.                                                                                                         | Partially aligned with provenance reconciled for the public implementation; exact equivalence to the private Delta/null-screening script is still not validated.                                                              | Validate the public provenance table against the recovered `null_distribution.py`/Delta-null workflow or externalize the private retained-term artifact as the source of truth. |
| Interaction discovery          | `04_interaction_discovery.ipynb`          | Fits `GradientBoostingRegressor` per PCA component, computes SHAP interaction values via `shap.TreeExplainer`, aggregates by max-over-components of mean absolute SHAP interaction per pair, applies a deterministic permutation null threshold, and writes provenance marked `manuscript_aligned`.                        | **Manuscript aligned.** Implements tree-based SHAP interaction values as specified in the frozen contract (`tree_shap_gradient_boosting`, `manuscript_aligned_via_shap_gradient_boosting`).                                   | Validate retained pair counts against the manuscript value (62 discovered, 49 in final support) once real data are available.                                                   |
| Nonlinear discovery            | `05_nonlinear_discovery.ipynb`            | Fits cubic smoothing splines per base feature per PCA component via `scipy.interpolate.UnivariateSpline`, computes EDF and F-test p-value, declares curvature when EDF > 2 and p < 0.01, selects best parametric replacement by minimum RMSE against the fitted smooth, and writes provenance marked `manuscript_aligned`. | **Manuscript aligned.** Implements GAM EDF > 1 / p < 0.01 curvature rule and parametric RMSE replacement as specified in the frozen contract (`gam_cubic_smoothing_spline`, `manuscript_aligned_via_scipy_smoothing_spline`). | Validate identified/retained transformation counts against manuscript values (41 discovered, 29 in final support) once real data are available.                                 |
| Sparse selection and stability | `06_sparse_selection_and_stability.ipynb` | Builds the candidate union from retained screening, interaction, and nonlinear terms; fits EBIC-selected `sklearn.linear_model.Lasso` models per PCA component; computes deterministic subsample support diagnostics; writes `sparse_selection_provenance.csv`.                                                            | Partially aligned. It implements L1 selection, EBIC, stability artifacts, and explicit provenance, but does not yet establish equivalence to the manuscript's recovered de-biased-LASSO workflow.                             | Port or verify the recovered de-biased-LASSO selection logic and preserve the stability handoff.                                                                                |
| Final OLS and export           | `07_final_ols_and_bundle_export.ipynb`    | Applies the frozen 95% HC3 Wald inferential filter to the sparse/stability support, refits final OLS on retained terms, exports coefficients/standardization, and computes holdout macro nRMSE with bootstrap uncertainty.                                                                                                 | More closely aligned for this local stage. The HC3 interval/drop rule is now implemented, but real-data counts and manuscript table values still need verification.                                                           | Validate retained-feature counts and final coefficients against the private manuscript run.                                                                                     |
| Tables and figures             | `08_manuscript_tables_and_figures.ipynb`  | Writes manuscript-facing CSV tables, figure source-data CSVs, and dependency-free SVG diagnostics.                                                                                                                                                                                                                         | Partially aligned. These are deterministic public artifacts, not a guarantee of exact manuscript table and figure reproduction.                                                                                               | Map each manuscript table and figure to a named source artifact and verify values/layouts against the manuscript.                                                               |
| QA audit and smoke gate        | no separate notebook                      | Runs the full demo chain, hashes artifacts, and checks artifact presence, nonempty outputs, positive holdout nRMSE, CI ordering, final support, and SVG outputs.                                                                                                                                                           | Strong demo/gate QA, but not scientific exactness proof.                                                                                                                                                                      | Extend audit checks with real-data manuscript-count and manuscript-metric comparisons once private artifacts are available.                                                     |

## Release claim policy

Use the following wording until the gaps above are closed:

- Acceptable: **source-backed executable manuscript reproduction scaffold**.
- Acceptable: **deterministic demo reproduction chain with QA-audited handoff artifacts**.
- Acceptable: **public implementation of several manuscript-stage contracts with explicitly tracked
  approximation gaps**.
- Not acceptable yet: **full exact reproduction of the manuscript workflow**.
- Not acceptable yet: **all notebooks implement the manuscript exactly**.

## Immediate next technical priorities

1. Verify the implemented final HC3 inferential filter against the private manuscript run, including
   retained-feature counts, dropped-feature counts, and final coefficient tables.
1. Validate empirical-null screening equivalence against the recovered Delta-null script or replace
   the public surrogate with a released private retained-term artifact.
1. Map final manuscript tables and figures one-to-one to manuscript labels and expected values.

## Externalized manuscript-only upstream artifacts

- empirical-null screening stage
  - Source workflow reference: recovered `null_distribution.py` / Delta-null workflow
  - Source artifact availability in this public repo: no
- de-biased-LASSO selection stage
  - Source workflow reference: recovered `LASSO_to_OLS_v9.ipynb` workflow
  - Source artifact availability in this public repo: no

## Manuscript table and figure artifact map

Status: mapping complete, manuscript-value verification pending.

| Manuscript label                                                 | Public artifact source                                                                                                                           | Current verification status                                                                  |
| ---------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------- |
| Table 1 (workflow stage counts and retained support progression) | `final_manuscript_artifacts/tables/workflow_stage_summary.csv`                                                                                   | Artifact mapping complete; values still require private manuscript-run verification.         |
| Table 2 (holdout performance summary)                            | `final_manuscript_artifacts/tables/model_performance.csv`                                                                                        | Artifact mapping complete; values still require private manuscript-run verification.         |
| Figure 1 (model performance figure)                              | `final_manuscript_artifacts/figures/figure_model_performance_data.csv` and `final_manuscript_artifacts/figures/figure_model_performance.svg`     | Artifact mapping complete; plotted values/layout still require manuscript-side verification. |
| Figure 2 (final-support composition figure)                      | `final_manuscript_artifacts/figures/figure_support_composition_data.csv` and `final_manuscript_artifacts/figures/figure_support_composition.svg` | Artifact mapping complete; plotted values/layout still require manuscript-side verification. |

## Private-run verification evidence ledger

| Verification target               | Public artifact source                                                                                                                           | Expected private evidence artifact                           | Verification status          |
| --------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------ | ---------------------------- |
| HC3 retained-feature count parity | `final_manuscript_artifacts/tables/workflow_stage_summary.csv` (`n_final_features`)                                                              | Private manuscript-run final OLS support-count export        | pending private-run evidence |
| HC3 dropped-feature count parity  | `final_manuscript_artifacts/tables/hc3_inferential_filter_summary.csv` (`n_prefilter_features` - `n_final_features`)                             | Private manuscript-run HC3 inferential-filter summary export | pending private-run evidence |
| Final coefficient parity          | `final_manuscript_artifacts/tables/final_model_coefficients.csv`                                                                                 | Private manuscript-run final OLS coefficient table           | pending private-run evidence |
| Table 1 value parity              | `final_manuscript_artifacts/tables/workflow_stage_summary.csv`                                                                                   | Frozen manuscript Table 1 source table                       | pending private-run evidence |
| Table 2 value parity              | `final_manuscript_artifacts/tables/model_performance.csv`                                                                                        | Frozen manuscript Table 2 source table                       | pending private-run evidence |
| Figure 1 value/layout parity      | `final_manuscript_artifacts/figures/figure_model_performance_data.csv` and `final_manuscript_artifacts/figures/figure_model_performance.svg`     | Frozen manuscript Figure 1 plotting data and exported figure | pending private-run evidence |
| Figure 2 value/layout parity      | `final_manuscript_artifacts/figures/figure_support_composition_data.csv` and `final_manuscript_artifacts/figures/figure_support_composition.svg` | Frozen manuscript Figure 2 plotting data and exported figure | pending private-run evidence |

## Recent updates (internal)

- Removed tracked recovered HPC notebooks from docs/final_scripts_from_hpc; these remain reference-only and are now ignored by .gitignore.
- Added deterministic fallback in sparse-selection to prevent empty final_stable_support when stability gates fail (implemented and tested).
- Added a minimal synthetic-notebook fixture for notebook-compare tests to avoid reliance on private HPC notebooks.
- Cleaned and scrubbed the recovered LASSO notebook outputs where it had been force-added temporarily.

### Next steps

- Validate HC3 inferential filter and final table counts against private manuscript-run artifacts.
- Validate empirical-null screening equivalence against the recovered Delta-null script.
- Map manuscript figures/tables to named source artifacts and verify numerical equality.
- Decide on archival policy for recovered notebooks (keep only as offline reference; don't commit to main history in future).
