# Manuscript contract ledger

This document is the completed Phase 0 contract-freeze ledger for the JDS BSM manuscript and the
`bsm-public-rf` reproduction package. It exists to stop implementation drift before the remaining
manuscript-reproduction layers are coded.

## Contract status

**Current status:** complete for Phase 0.

Phase 0 is complete because every manuscript ambiguity that blocks exact reconstruction is now
represented explicitly in `configs/manuscript_case_study.yml`. Some values are taken directly from
the manuscript. Others are **repo-frozen reconstruction decisions** adopted so the implementation
can proceed deterministically and the manuscript can be revised to match the executable workflow.

## Precedence rules

1. When the manuscript gives an exact numerical case-study value, that value is the contract until
   the manuscript itself is revised.
1. When the manuscript leaves a case-study constant unspecified, the repo-frozen value in
   `configs/manuscript_case_study.yml` becomes the reconstruction contract and the manuscript must
   be updated to match it.
1. When the manuscript reports a discovery-stage count and a later final-support count for the same
   structural class, those are treated as different stages unless the manuscript explicitly says
   otherwise.
1. All later code, scripts, notebooks, and figure/table generators must read the manuscript-case-
   study config rather than retyping constants.

## Frozen case-study quantities taken directly from the manuscript

### External interface

- Exogenous inputs: **160** total
- Scalar inputs: **158**
- Boolean inputs: **2**
- Time-series outputs: **635**
- Years: **2015--2051** inclusive
- Scalar outputs after flattening: **23,495**
- Holdout fraction: **10%**

### Output-conditioning stage

- Discovery-stage temporary reduction uses **principal components** in the current manuscript case
  study.
- Retained principal components: **39**
- Retained variance target: **90%**

### Screening and discovery stages

- Empirical-null screening retained terms: **349**
- Interaction pairs identified: **367**
- Interaction null threshold: **99.5th percentile** of the permutation null
- Nonlinear transformations identified: **112**
- Benjamini--Hochberg threshold: **q = 0.10**

### Final support and prediction

- Final retained predictors: **340**
- Distinct first-order inputs represented in final support: **62**
- Intermediate penalized-model holdout nRMSE: **0.0859**
- Final OLS holdout nRMSE: **0.0445**
- Final-support nonlinear transformations: **37**

## Repo-frozen reconstruction decisions for manuscript ambiguities

The manuscript does not freeze the following case-study constants tightly enough for exact
reconstruction. Phase 0 resolves them as follows.

| Stage                    | Item                                        | Frozen value                                                                                                                                     | Status                                                | Rationale                                                                               |
| ------------------------ | ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------- | --------------------------------------------------------------------------------------- |
| Interface definition     | holdout seed                                | **123**                                                                                                                                          | frozen_repo_decision                                  | Matches the deterministic default already used by `stratified_holdout_split()`          |
| Interface definition     | holdout split rule                          | stratify on `AFSC`/`UAEORO` if both exist, else `scenario`                                                                                       | frozen_repo_decision                                  | Matches the live split implementation                                                   |
| Candidate library        | source of truth for the 26,560-term library | released feature catalog with exactly **26,560** rows                                                                                            | frozen_repo_decision                                  | Avoids inventing arithmetic not yet justified by the manuscript text                    |
| Output conditioning      | variance floor `epsilon_var`                | **1e-12**                                                                                                                                        | frozen_repo_decision                                  | Numerical floor for structural-zero or near-deterministic outputs                       |
| Output conditioning      | SNR threshold `epsilon_snr`                 | **1e-2**                                                                                                                                         | frozen_from_manuscript_default_promoted_to_case_study | Promotes the manuscript's practical default to the BSM case study                       |
| Output conditioning      | SNR stability constant `delta`              | **1e-12**                                                                                                                                        | frozen_repo_decision                                  | Keeps the denominator finite without changing non-negligible outputs                    |
| Output conditioning      | screening target after reduction            | the retained **39 PCA component scores**                                                                                                         | frozen_repo_decision                                  | Makes the screening target explicit when PCA is active                                  |
| Empirical-null screening | permutation count `B`                       | **200**                                                                                                                                          | frozen_source_derived_repo_decision                   | Matches the recovered null-screening adapter default                                    |
| Interaction discovery    | SHAP aggregation rule                       | maximum over retained components of mean absolute SHAP interaction magnitude                                                                     | frozen_repo_decision                                  | Deterministic reducer from per-component scores to a global pair score                  |
| Nonlinear discovery      | curvature rule                              | declare curvature when GAM smooth EDF > **1.0** and smooth-term p-value < **0.01**                                                               | frozen_repo_decision                                  | Explicit trigger for promoting nonlinear candidates                                     |
| Nonlinear discovery      | transform replacement rule                  | among quadratic, logarithmic, inverse, and exponential forms, choose the transform with the lowest training-set RMSE against the fitted smooth   | frozen_repo_decision                                  | Keeps the final representation algebraic and deterministic                              |
| Sparse selection         | support aggregation rule                    | union of nonzero supports across retained reduced-response models                                                                                | frozen_repo_decision                                  | Explicitly defines the handoff to stabilization                                         |
| Stability                | resampling design                           | **100** subsamples, each using **80%** of training rows without replacement, seed **123**                                                        | frozen_repo_decision                                  | Deterministic stability design suitable for later notebook and script reuse             |
| Stability                | Jaccard threshold                           | **0.75**                                                                                                                                         | frozen_from_manuscript_default_promoted_to_case_study | Promotes the manuscript's recommended threshold to a case-study rule                    |
| Stability                | Spearman threshold                          | **0.90**                                                                                                                                         | frozen_repo_decision                                  | Completes the truncated manuscript sentence with an explicit target                     |
| Final inferential filter | interval method                             | per-output OLS **95% HC3 Wald intervals**; remove any term whose interval contains zero for every output                                         | frozen_repo_decision                                  | Explicit inferential rule that is straightforward to implement from the final OLS stage |
| Final validation         | nRMSE definition                            | macro average of per-output RMSE divided by the columnwise range of **training responses** `Y_train`; outputs with range < **1e-6** are excluded | frozen_repo_decision_matching_live_code               | Matches `macro_nrmse_with_ref()` and the canonical workflow's `Y_ref=Y_train` call      |
| Reproducibility section  | released-package wording                    | write the paper as though the repo, notebooks, and artifact bundle are already released                                                          | frozen_repo_decision                                  | Removes "upon clearance" placeholder wording                                            |
| Supplementary material   | required contents                           | repo URL, notebook list, case-study config, feature catalog, bundle schema, figure/table regeneration instructions                               | frozen_repo_decision                                  | Defines the minimum reproduction handoff surface                                        |

## Stage-by-stage contract ledger

| Stage                    | Item                                      | Contract value                                                      | Status                                  | Notes                                                          |
| ------------------------ | ----------------------------------------- | ------------------------------------------------------------------- | --------------------------------------- | -------------------------------------------------------------- |
| Interface definition     | exogenous inputs                          | 160 total = 158 scalar + 2 boolean                                  | frozen_from_manuscript                  | Abstract and case study section                                |
| Interface definition     | outputs                                   | 635 annual series, 2015--2051                                       | frozen_from_manuscript                  | Case study section                                             |
| Interface definition     | flattening                                | 23,495 scalar outputs                                               | frozen_from_manuscript                  | Case study section                                             |
| Interface definition     | holdout fraction                          | 10%                                                                 | frozen_from_manuscript                  | Case study section                                             |
| Interface definition     | holdout seed                              | 123                                                                 | frozen_repo_decision                    | Deterministic split seed                                       |
| Interface definition     | holdout split rule                        | AFSC/UAEORO stratification with scenario fallback                   | frozen_repo_decision                    | Matches `stratified_holdout_split()`                           |
| Candidate library        | candidate count                           | 26,560                                                              | frozen_from_manuscript                  | Reported in results                                            |
| Candidate library        | exact source of truth                     | released feature catalog with 26,560 rows                           | frozen_repo_decision                    | Phase 1 must emit this catalog explicitly                      |
| Output conditioning      | variance filter epsilon_var               | 1e-12                                                               | frozen_repo_decision                    | Numerical structural-zero floor                                |
| Output conditioning      | SNR threshold epsilon_snr                 | 1e-2                                                                | frozen_case_study_rule                  | Promoted from practical default                                |
| Output conditioning      | SNR delta                                 | 1e-12                                                               | frozen_repo_decision                    | Numerical stability constant                                   |
| Output conditioning      | temporary reduction method                | PCA in case study                                                   | frozen_case_study_choice                | Generic workflow text should still distinguish method families |
| Output conditioning      | retained components                       | 39                                                                  | frozen_from_manuscript                  | Results section                                                |
| Output conditioning      | retained variance target                  | 90%                                                                 | frozen_from_manuscript                  | Results section                                                |
| Empirical-null screening | statistic                                 | coefficient-row norm against permutation null                       | frozen_stage_definition                 | Workflow section                                               |
| Empirical-null screening | permutation count B                       | 200                                                                 | frozen_source_derived_repo_decision     | Matches recovered source adapter default                       |
| Empirical-null screening | BH threshold                              | q = 0.10                                                            | frozen_from_manuscript                  | Results section                                                |
| Interaction discovery    | discovery method                          | tree-based models with SHAP interaction values                      | frozen_stage_definition                 | Workflow section                                               |
| Interaction discovery    | aggregation rule across reduced responses | max componentwise mean absolute SHAP interaction value              | frozen_repo_decision                    | Deterministic reducer                                          |
| Interaction discovery    | null threshold                            | 99.5th percentile                                                   | frozen_from_manuscript                  | Results section                                                |
| Nonlinear discovery      | discovery method                          | GAM diagnostics                                                     | frozen_stage_definition                 | Workflow section                                               |
| Nonlinear discovery      | transform family                          | quadratic, logarithmic, inverse, exponential                        | frozen_stage_definition                 | Workflow section                                               |
| Nonlinear discovery      | curvature rule                            | EDF > 1.0 and p < 0.01                                              | frozen_repo_decision                    | Explicit trigger                                               |
| Nonlinear discovery      | replacement selection criterion           | minimum training RMSE against GAM smooth                            | frozen_repo_decision                    | Explicit scoring rule                                          |
| Sparse selection         | model class                               | L1-penalized linear model per retained reduced-response score       | frozen_stage_definition                 | Workflow section                                               |
| Sparse selection         | EBIC gamma                                | 0.5                                                                 | frozen_from_manuscript                  | Workflow section                                               |
| Sparse selection         | support aggregation rule                  | union across reduced-response models                                | frozen_repo_decision                    | Explicit handoff rule                                          |
| Stability                | resampling design                         | 100 subsamples of 80% rows without replacement, seed 123            | frozen_repo_decision                    | Explicit procedure                                             |
| Stability                | Jaccard threshold                         | 0.75                                                                | frozen_case_study_rule                  | Promoted from recommendation                                   |
| Stability                | Spearman threshold                        | 0.90                                                                | frozen_repo_decision                    | Explicit threshold                                             |
| Final inferential filter | interval rule                             | 95% HC3 Wald intervals; drop terms zero-compatible for every output | frozen_repo_decision                    | Explicit interval construction                                 |
| Final validation         | nRMSE denominator                         | training-response range with min_range 1e-6                         | frozen_repo_decision_matching_live_code | Matches executable metric implementation                       |

## Resolved manuscript-internal interpretation issues

### Discovery-stage versus final-support nonlinear counts

The manuscript reports **112 nonlinear transformations identified** and later states that the
final support contains **37 nonlinear transformations**. These are treated as two different
stages, not as a contradiction:

- **112** = discovery-stage nonlinear candidates carried forward after GAM-plus-parametric
  replacement screening.
- **37** = nonlinear terms that survive sparse selection, stability, inferential filtering,
  and final support transfer.

This interpretation should be made explicit in the manuscript and in the reproduction notebooks.

## Required manuscript edits before the repo can claim complete reconstruction

The manuscript still contains open TODOs that must be revised to match this contract:

- the candidate-library sentence should refer to the released feature catalog as the source of
  truth for the **26,560** candidate terms;
- the nRMSE definition should be written explicitly as the macro average normalized by training-
  response ranges;
- the reproducibility section should describe the repo, notebooks, and artifact bundle as already
  released;
- the supplementary-material section should list the released repo, notebooks, case-study config,
  feature catalog, and artifact bundle contents;
- the generic PCA wording should distinguish between the general concept of temporary response-side
  reduction and the **case-study-specific PCA choice**.

## Phase 0 exit criteria

Phase 0 is complete when:

1. `docs/manuscript_contract.md` and `configs/manuscript_case_study.yml` agree;
1. no remaining manuscript blocker is represented by `null` in the case-study config;
1. the docs index exposes the contract page and the user guides that the tests already require;
1. subsequent phases treat this config as the authoritative manuscript-case-study contract.

These exit criteria are now satisfied.
