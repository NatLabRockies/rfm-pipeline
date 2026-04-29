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

The notebooks are implemented as executable entrypoints, but the repository should **not** yet claim
that the full workflow is implemented exactly according to the manuscript.

The present implementation is best described as a source-backed, deterministic reproduction scaffold
with several exact stages and several approximation or placeholder scientific stages. The main
remaining scientific gaps are interaction discovery, nonlinear discovery, the de-biased-LASSO versus
plain L1-selection distinction, exact manuscript figure regeneration, and real-data verification
of the newly implemented HC3 filter.

## Stage-by-stage alignment

| Stage                          | Notebook                                  | Current implementation                                                                                                                                                                                                     | Alignment status                                                                                                                                                                                 | Highest-priority remaining work                                                                                                                         |
| ------------------------------ | ----------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Data intake                    | `00_case_study_data_intake.ipynb`         | Resolves runtime context, validates demo or local artifact tables, and reports table shapes.                                                                                                                               | Executable runtime entrypoint; exactness depends on local private artifact paths.                                                                                                                | Add real-data dry-run evidence once private paths are configured.                                                                                       |
| Candidate library audit        | `01_candidate_library_audit.ipynb`        | Loads the released feature catalog from the runtime artifact tables and summarizes feature types.                                                                                                                          | Partially aligned; it audits an external catalog but does not regenerate the 26,560-row library from primitive rules.                                                                            | Decide whether the repo must regenerate the catalog or treat the released catalog as the reproducibility artifact of record.                            |
| Output conditioning            | `02_output_conditioning.ipynb`            | Applies train-only variance and dynamic-range filters and computes retained PCA scores.                                                                                                                                    | Mostly aligned with the frozen contract for the public implementation.                                                                                                                           | Verify real-data retained output/component counts against the manuscript values.                                                                        |
| Empirical-null screening       | `03_empirical_null_screen.ipynb`          | Materializes catalog terms, computes coefficient-row-L2 statistics against retained PCA scores, estimates featurewise response-permutation p-values, and applies BH q = 0.10.                                              | Partially aligned; this matches the Phase 0 coefficient-row-norm contract, but the recovered workflow audit still records an upstream SALib Delta null-screening script as canonical provenance. | Reconcile the manuscript text, recovered `null_distribution.py`, and this PCA-score coefficient-row-norm screen into one documented scientific path.    |
| Interaction discovery          | `04_interaction_discovery.ipynb`          | Scores catalog interaction pairs by residualized product-term contribution beyond first-order factors and applies a deterministic permutation threshold.                                                                   | Not exact. The frozen manuscript contract specifies tree-based models with SHAP interaction values and max-over-component mean absolute SHAP interaction aggregation.                            | Implement or explicitly externalize the tree/SHAP interaction stage; do not present the residualized-product score as the manuscript-exact SHAP method. |
| Nonlinear discovery            | `05_nonlinear_discovery.ipynb`            | Scores catalog transformations by residualized nonlinear contribution beyond the corresponding first-order input.                                                                                                          | Not exact. The frozen contract specifies GAM diagnostics, EDF > 1, p < 0.01, and parametric replacement by minimum RMSE against the fitted smooth.                                               | Implement GAM-based curvature detection and replacement scoring, or require a released nonlinear-discovery artifact table from the private workflow.    |
| Sparse selection and stability | `06_sparse_selection_and_stability.ipynb` | Builds the candidate union from retained screening, interaction, and nonlinear terms; fits EBIC-selected `sklearn.linear_model.Lasso` models per PCA component; computes deterministic subsample support diagnostics.      | Partially aligned. It implements L1 selection, EBIC, and stability artifacts, but does not yet establish equivalence to the manuscript's recovered de-biased-LASSO workflow.                     | Port or verify the recovered de-biased-LASSO selection logic and preserve the stability handoff.                                                        |
| Final OLS and export           | `07_final_ols_and_bundle_export.ipynb`    | Applies the frozen 95% HC3 Wald inferential filter to the sparse/stability support, refits final OLS on retained terms, exports coefficients/standardization, and computes holdout macro nRMSE with bootstrap uncertainty. | More closely aligned for this local stage. The HC3 interval/drop rule is now implemented, but real-data counts and manuscript table values still need verification.                              | Validate retained-feature counts and final coefficients against the private manuscript run.                                                             |
| Tables and figures             | `08_manuscript_tables_and_figures.ipynb`  | Writes manuscript-facing CSV tables, figure source-data CSVs, and dependency-free SVG diagnostics.                                                                                                                         | Partially aligned. These are deterministic public artifacts, not a guarantee of exact manuscript table and figure reproduction.                                                                  | Map each manuscript table and figure to a named source artifact and verify values/layouts against the manuscript.                                       |
| QA audit and smoke gate        | no separate notebook                      | Runs the full demo chain, hashes artifacts, and checks artifact presence, nonempty outputs, positive holdout nRMSE, CI ordering, final support, and SVG outputs.                                                           | Strong demo/gate QA, but not scientific exactness proof.                                                                                                                                         | Extend audit checks with real-data manuscript-count and manuscript-metric comparisons once private artifacts are available.                             |

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
1. Reconcile empirical-null screening provenance between the recovered Delta-null script and the
   current PCA-score coefficient-row-norm screen.
1. Replace or externalize the interaction stage so the repo no longer substitutes residualized
   product scores for SHAP interaction values while claiming manuscript exactness.
1. Replace or externalize the nonlinear stage so the repo no longer substitutes residualized
   parametric scores for GAM EDF/p-value diagnostics while claiming manuscript exactness.
1. Map final manuscript tables and figures one-to-one to manuscript labels and expected values.
