# De-biased-LASSO implementation contract

This page freezes the implementation contract for the manuscript sparse-selection stage before any de-biased-LASSO code is added. The current public implementation remains the deterministic EBIC/L1 surrogate documented in the sparse-selection provenance artifacts. It must not be renamed or treated as manuscript-exact de-biased LASSO until the source-notebook blockers below are resolved.

## Current status

- Implementation present: The repository now includes a deterministic, test-focused de-biased-LASSO implementation at src/rfm_pipeline/debiased_lasso.py. This implementation is exercised by unit tests and golden fixtures that validate deterministic behavior on synthetic toy data (see tests/test_debiased_lasso_regression.py and tests/test_debiased_lasso_golden_compare.py).

- Validation scope: The current tests exercise numeric shapes, stability of support selection on strong synthetic signal, and reproducibility against locally generated golden fixtures. However, the implementation has NOT yet been validated against the original manuscript notebook outputs; provenance and notebook recovery are still required before claiming manuscript-exact equivalence.

- Public surrogate: The sparse-selection workflow continues to use an EBIC/L1 surrogate with stability diagnostics for the public manuscript handoff. The de-biasing helpers are available as a reproducible, test-validated artifact for downstream use and further auditing.

## Current status

| Item                           | Contract value                                     |
| ------------------------------ | -------------------------------------------------- |
| Manuscript stage               | `sparse_selection_and_stability`                   |
| Public stage entrypoint        | `select_manuscript_sparse_support`                 |
| Current public method          | `ebic_l1_component_union_with_subsample_stability` |
| Target source workflow         | `notebook_pca_debiased_lasso`                      |
| Source artifact                | `LASSO_to_OLS_v9.ipynb`                            |
| Source artifact in public repo | `true`                                             |
| Exact implementation status    | `contract_frozen_not_implemented`                  |
| Equivalence status             | `not_yet_validated`                                |
| De-biasing definition status   | `unresolved_requires_source_notebook_audit`        |

The public sparse-selection function is therefore a handoff-compatible surrogate, not the final manuscript-exact de-biased-LASSO implementation.

## Frozen recovered notebook facts

| Quantity                                        | Contract value |
| ----------------------------------------------- | -------------: |
| Candidate inputs entering notebook sparse stage |            352 |
| Flattened outputs before culling                |         23,495 |
| Outputs retained after culling                  |          9,782 |
| Retained PCA component scores                   |             39 |
| Training rows                                   |         18,000 |
| Holdout rows                                    |          2,000 |
| External holdout fraction                       |           0.10 |
| LASSO mixing value `l1_ratio`                   |           1.00 |
| EBIC `gamma`                                    |            0.5 |
| Recovered selected alpha fraction               |           0.10 |
| Recovered approximate selected absolute alpha   |         2.8541 |
| Recovered notebook selected-feature count       |            346 |
| Final support count after downstream filtering  |            340 |

The recovered fractional alpha grid is:

```text
0.75, 0.50, 0.25, 0.10, 0.05, 0.02, 0.01
```

The recovered model-selection description is **EBIC-guided sparse path search with ALO/KKT diagnostics**. The source notebook `LASSO_to_OLS_v9.ipynb` has been located in `docs/final_scripts_from_hpc/`; the exact de-biasing estimator definition will be frozen after executing and validating the notebook outputs against the public implementation.

## Required implementation boundary

A future implementation may change the executable sparse-selection method only after tests and documentation preserve these boundaries:

1. `select_manuscript_sparse_support(...)` remains the public handoff function, or the API change is explicitly documented and tested.
1. The implementation emits stable tables for alpha-path diagnostics, component-level coefficients, de-biased coefficient diagnostics, support selection, provenance, and stability handoff.
1. The alpha path, EBIC scoring, ALO/KKT diagnostics, and support aggregation behavior are tested independently on deterministic synthetic data.
1. The de-biasing estimator definition is frozen from the recovered source notebook rather than inferred from the current public surrogate.
1. Exact manuscript-equivalence claims remain blocked until outputs are validated against the recovered notebook run.

## Blockers before an exact claim

The package may claim exact de-biased-LASSO implementation only after all of the following are true:

- `LASSO_to_OLS_v9.ipynb` source cells have been recovered and inspected.
- The exact de-biasing estimator definition has been frozen in source, config, and docs.
- Deterministic tests cover alpha-path construction, EBIC selection, ALO/KKT diagnostics, emitted artifact schema, and stage handoff behavior.
- The emitted support and diagnostics have been validated against the recovered notebook run.
- The stability and final-OLS handoff artifact schema remains compatible with the current manuscript notebooks.
